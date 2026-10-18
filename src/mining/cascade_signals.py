"""Two day-trading mining candidates, both variations on the same
mechanism: a daily directional bias (price_structure.daily_bias), confirmed
by same-direction structure on the hourly chart (price_structure.
classify_reversal_bar), entered on a 5-minute fair-value-gap (price_
structure.fair_value_gaps) retest-and-reject. LONG-ONLY scope, matching
this catalog's established day-trading convention (e.g. day_trading_
signals.make_orb_5min_atr_stop_scalp_signal_fn's own long-only disclosure).

Both factories below share one private state machine, _make_cascade_
signal_fn -- make_fvg_reversal_cascade_signal_fn (Build 1, single-asset)
calls it with divergence_check=None; make_cross_asset_divergence_fvg_
cascade_signal_fn (Build 2) calls it with a divergence_check closure that
adds exactly one extra gate to the hourly-confirmation step. Everything
else (bias, entry, fill capture, exit) is identical between the two builds
and lives in the shared helper.

Every factory returns a SignalFn matching src/mining/intraday_backtest_
engine.SignalResult/SignalFn's exact contract (see that module's own
docstring): signal_fn(frame, i) receives frame = df.iloc[:i+1] (a slice
ending EXACTLY at bar i), entry=True fires next-bar-open execution,
exit=True fires same-bar-close exit. Position state (a per-day phase, an
"armed but unfilled" flag, an "open" flag, stop/target prices) is tracked
in a closure-local dict and defensively reset on every calendar-date
change, mirroring day_trading_signals.make_orb_5min_atr_stop_scalp_
signal_fn's own closure-state convention exactly.

Disclosed simplifications (stated once here, apply to both builds):
  - Long-only: a bearish/bearish_continuation/none daily bias simply skips
    the day; no short side is implemented.
  - One trade per day maximum, matching this catalog's "only one position
    tracked at a time" convention -- once a trade closes (or once the
    day's bias/confirmation path never qualifies), no further setup is
    taken the same day, even if a second qualifying setup later appears.
  - The fair-value-gap filter only considers gaps whose own middle-candle
    timestamp is AT OR AFTER the hourly confirmation's own end timestamp
    -- a gap that formed before the point of interest was confirmed does
    not count.
  - Gap invalidation: a qualifying gap is dropped (never re-used) once any
    LATER 5-minute bar's Low trades cleanly through it (below its own
    gap_low) before any bar retests and rejects it. Among the gaps that
    still qualify and have not been invalidated, the MOST RECENT one (by
    its own middle-candle timestamp) is the one entry is evaluated
    against.
  - Target rule: max(min_target_r_multiple x initial risk, prior
    completed day's High minus the real fill price) -- so the target
    never sits below the previous day's high once price has already
    broken above it.
  - Build 2's divergence gate defaults PERMISSIVE when the secondary
    instrument's own data does not yet cover the primary's confirming
    hourly bucket (nothing to compare against -- the primary's own
    confirmation is used on its own rather than blocking on missing data).
  - Entry is gated to strictly before `force_flat_et` (step 3 below) so a
    fill -- which always lands on the NEXT bar's open -- cannot spill onto
    the following calendar day: an ungated late entry would fill next-day while
    this closure's own per-day reset had already wiped its "armed" bookkeeping, making the
    engine's own resulting position invisible to this closure's exit
    logic for a full extra session. Residual, disclosed and unresolved:
    a genuine NYSE early-close day (e.g. the day after Thanksgiving) that
    closes BEFORE `force_flat_et` could still exhibit this same failure
    mode in its own final minutes -- this factory has no trading-calendar
    input to detect that case in advance.
"""
from __future__ import annotations

import bisect
import datetime
from typing import Callable, NamedTuple, Optional

import pandas as pd

from .intraday_backtest_engine import SignalFn, SignalResult
from .price_structure import classify_reversal_bar, daily_bias, fair_value_gaps


_SESSION_OPEN_MINUTES = 9 * 60 + 30  # 09:30 ET

# Generous trailing windows -- bounded lookups, never a scan of the whole
# (potentially multi-hundred-thousand-row) history on every 1-minute call.
# Matches the pattern established by day_trading_signals.py's own
# `_compute_atr14` helper.
_DAILY_LOOKBACK_BARS = 20000
_TODAY_LOOKBACK_BARS = 2000

_OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


def _bound_tail(frame: pd.DataFrame, bound_bars: int) -> pd.DataFrame:
    return frame.tail(bound_bars) if len(frame) > bound_bars else frame


def _build_daily_bars(frame: pd.DataFrame, today: datetime.date) -> pd.DataFrame:
    """Daily OHLC bars (Open=first, High=max, Low=min, Close=last) built
    from every row in `frame` whose own date is strictly before `today` --
    bounded to a generous trailing window first, matching day_trading_
    signals.py's own `_compute_atr14` convention. Returns an empty frame
    (same columns) rather than raising when no such rows exist yet."""
    tail = _bound_tail(frame, _DAILY_LOOKBACK_BARS)
    hist = tail[tail.index.date < today]
    if hist.empty:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    return hist.groupby(hist.index.date).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )


def _today_frame(frame: pd.DataFrame, today: datetime.date) -> pd.DataFrame:
    """Rows of `frame` whose own date equals `today`, bounded to a
    generous trailing window first (today's own rows are always near the
    end of `frame`, so this is cheap even against a huge history)."""
    tail = _bound_tail(frame, _TODAY_LOOKBACK_BARS)
    return tail[tail.index.date == today]


def _bucket_bounds(ts: pd.Timestamp, bucket_minutes: int):
    """The [start, end) boundaries of the `bucket_minutes`-wide bucket
    containing `ts`, anchored to the 09:30 ET session open on ts's own
    calendar date. Pure wall-clock-minute arithmetic (not raw Timestamp
    addition) so a bucket's boundaries are exact multiples of
    `bucket_minutes` past 09:30, regardless of `ts`'s own tz offset."""
    minute_of_day = ts.hour * 60 + ts.minute
    elapsed = minute_of_day - _SESSION_OPEN_MINUTES
    bucket_index = elapsed // bucket_minutes
    start_minutes = _SESSION_OPEN_MINUTES + bucket_index * bucket_minutes
    day_start = ts.normalize()
    start = day_start + pd.Timedelta(minutes=int(start_minutes))
    end = start + pd.Timedelta(minutes=int(bucket_minutes))
    return start, end


def _build_complete_buckets(
    today_frame: pd.DataFrame, current_ts: pd.Timestamp, bucket_minutes: int
) -> pd.DataFrame:
    """`bucket_minutes`-wide OHLC bars (Open=first, High=max, Low=min,
    Close=last) built from `today_frame`, anchored to 09:30 ET, indexed by
    each bucket's own START timestamp -- keeping ONLY buckets whose own
    END boundary is at or before `current_ts` (i.e. fully complete as of
    now). A still-forming bucket (the one `current_ts`'s own bar belongs
    to) is always excluded, since its own end boundary is necessarily
    later than `current_ts` by construction. Returns an empty frame (same
    columns) rather than raising on empty input."""
    if today_frame.empty:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    grouped = _group_all_buckets(today_frame, bucket_minutes)
    if grouped.empty:
        return grouped
    ends = grouped.index + pd.Timedelta(minutes=bucket_minutes)
    return grouped[ends <= current_ts]


def _group_all_buckets(today_frame: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    """Every bucket (complete or still forming) `today_frame`'s rows fall in,
    aggregated Open=first / High=max / Low=min / Close=last and indexed by
    bucket START. The reference grouping `_build_complete_buckets` filters
    down, and the seed for `_TodayBuckets`' incremental state."""
    starts = [_bucket_bounds(ts, bucket_minutes)[0] for ts in today_frame.index]
    tmp = today_frame.copy()
    tmp["_bucket_start"] = starts
    grouped = tmp.groupby("_bucket_start").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    return grouped.sort_index()


def _day_midnight(index: pd.DatetimeIndex, day: datetime.date) -> pd.Timestamp:
    """`day`'s local midnight, in `index`'s own timezone (or naive)."""
    return pd.Timestamp(datetime.datetime.combine(day, datetime.time()), tz=index.tz)


def _day_row_bounds(index: pd.DatetimeIndex, day: datetime.date, offset: int = 0):
    """[lo, hi) positions, in `index`, of the rows whose own date is `day`,
    searching only `index[offset:]`. Requires an ascending index (callers
    verify) -- then the day's rows are one contiguous run, found in O(log n)
    instead of the O(n) `index.date == day` scan the reference used."""
    view = index[offset:]
    lo = view.searchsorted(_day_midnight(index, day), side="left")
    hi = view.searchsorted(_day_midnight(index, day + datetime.timedelta(days=1)), side="left")
    return offset + int(lo), offset + int(hi)


class _TodayBuckets:
    """Incremental equivalent of the reference expression
    `_build_complete_buckets(<today's rows>, current_ts, bucket_minutes)`.

    The reference re-groups all of today's bars from scratch (a fresh pandas
    `groupby` plus a Python loop over every timestamp) on EVERY 1-minute call,
    which is costly across a full-history intraday run. The
    engine's contract, though, is that `frame` only ever GROWS at the end
    (`df.iloc[:i+1]`), so this class keeps one running aggregate per bucket
    and folds in just the rows that arrived since the previous call.

    Exactness -- the result is indistinguishable from the reference for the
    engine's call pattern (`df.iloc[:i+1]` growing one bar at a time) and for
    ANY call order over slices of ONE frame (forward, repeated, backward,
    jumping across days -- pinned in test_cascade_signals_incremental.py
    against the frozen reference (uncached) implementation in
    oracle_cascade_signals_frozen.py):
      * The cache trusts its running state only while it can PROVE the frame
        continues the one it consumed: same first row of the day, at least as
        many rows, the last consumed row's timestamp and Close unchanged, and
        every new row strictly later than the latest `current_ts` already
        answered (so a bucket reported complete can never gain rows). Any
        failed proof reseeds from scratch with the reference grouping.
      * Inputs whose groupby semantics differ from plain incremental
        max/min/first/last are declined and served by the reference builder
        directly: non-float64 OHLC (groupby keeps integer dtype), NaN cells
        (first/last skip NaN), and a non-ascending index.
      * `current_ts` is evaluated per call against the bucket END, exactly as
        the reference does -- including a `current_ts` past the frame's last
        row, which the cross-asset gate's permissive default relies on.
    KNOWN LIMIT (disclosed, unreachable through the engine or either driver):
    "continues" is proven from the LAST consumed row only. One instance shown a
    DIFFERENT frame that has the same day, at least as many rows and an
    identical last-consumed row, but differs in earlier rows, would return the
    first frame's aggregates. A signal_fn (which owns one instance) must
    therefore not be reused across different frames; both drivers build a fresh
    one per ticker/run.
    A returned frame is SHARED between calls while the set of complete buckets
    is unchanged; callers must not mutate it (none do).
    """

    def __init__(self, bucket_minutes: int):
        self._m = int(bucket_minutes)
        self._invalidate()

    def _invalidate(self) -> None:
        self._valid = False
        self._lo = -1
        self._n = -1
        self._last_ts = None
        self._last_close = None
        self._max_query_ts = None
        self._starts: list = []
        self._ends: list = []
        self._open: list = []
        self._high: list = []
        self._low: list = []
        self._close: list = []
        self._out_k = -1
        self._out = None

    # -- public entry points -------------------------------------------

    def complete_primary(self, frame: pd.DataFrame, today: datetime.date,
                         current_ts: pd.Timestamp) -> pd.DataFrame:
        """== _build_complete_buckets(_today_frame(frame, today), current_ts, m)"""
        n = len(frame)
        cap_lo = max(0, n - _TODAY_LOOKBACK_BARS)
        if not frame.index[cap_lo:].is_monotonic_increasing:
            return _build_complete_buckets(_today_frame(frame, today), current_ts, self._m)
        lo, hi = _day_row_bounds(frame.index, today, cap_lo)
        return self._complete(frame, lo, hi, current_ts)

    def complete_secondary(self, secondary: pd.DataFrame, today: datetime.date,
                           current_ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        """== the reference divergence-gate expression
        `sec = secondary[secondary.index <= current_ts]` then
        `_build_complete_buckets(sec[sec.index.date == today], current_ts, m)`;
        returns None exactly when `sec` would be empty."""
        index = secondary.index
        if not index.is_monotonic_increasing:
            sec = secondary[index <= current_ts]
            if sec.empty:
                return None
            return _build_complete_buckets(sec[sec.index.date == today], current_ts, self._m)
        upto = int(index.searchsorted(current_ts, side="right"))
        if upto == 0:
            return None
        lo, hi = _day_row_bounds(index, today)
        return self._complete(secondary, lo, min(hi, upto), current_ts)

    # -- core ------------------------------------------------------------

    def _complete(self, frame: pd.DataFrame, lo: int, hi: int,
                  current_ts: pd.Timestamp) -> pd.DataFrame:
        """Buckets of `frame.iloc[lo:hi]` whose end is <= `current_ts`."""
        if hi <= lo:
            self._invalidate()
            return _build_complete_buckets(frame.iloc[lo:hi], current_ts, self._m)
        if not (self._continues(frame, lo, hi) and self._extend(frame, hi)):
            self._invalidate()
            if not self._seed(frame, lo, hi):
                return _build_complete_buckets(frame.iloc[lo:hi], current_ts, self._m)
        if self._max_query_ts is None or current_ts > self._max_query_ts:
            self._max_query_ts = current_ts
        k = bisect.bisect_right(self._ends, current_ts)
        if k != self._out_k:
            # k == 0 keeps the reference's own empty shape (float64 columns, a
            # tz-aware DatetimeIndex named "_bucket_start") by slicing a real
            # one-bucket frame rather than returning a bare empty DataFrame.
            built = max(k, 1)
            frame_k = pd.DataFrame(
                {
                    "Open": self._open[:built],
                    "High": self._high[:built],
                    "Low": self._low[:built],
                    "Close": self._close[:built],
                },
                index=pd.DatetimeIndex(self._starts[:built], name="_bucket_start"),
            )
            self._out = frame_k if k else frame_k.iloc[:0]
            self._out_k = k
        return self._out

    @staticmethod
    def _plain_ohlc(rows: pd.DataFrame) -> bool:
        """True when incremental max/min/first/last provably matches the
        reference groupby: float64 OHLC and no NaN. (An ascending index is
        also required, but both public entry points already refuse a
        non-ascending index before reaching here, so it is not re-checked --
        a redundant guard no test could pin.)"""
        ohlc = rows[_OHLC_COLUMNS]
        return (
            all(dt == "float64" for dt in ohlc.dtypes)
            and not ohlc.isna().to_numpy().any()
        )

    def _seed(self, frame: pd.DataFrame, lo: int, hi: int) -> bool:
        rows = frame.iloc[lo:hi]
        if not self._plain_ohlc(rows):
            return False
        grouped = _group_all_buckets(rows, self._m)
        delta = pd.Timedelta(minutes=self._m)
        self._starts = list(grouped.index)
        self._ends = [s + delta for s in self._starts]
        self._open = grouped["Open"].tolist()
        self._high = grouped["High"].tolist()
        self._low = grouped["Low"].tolist()
        self._close = grouped["Close"].tolist()
        self._lo, self._n = lo, hi
        self._remember_last(frame, hi)
        self._valid = True
        return True

    def _remember_last(self, frame: pd.DataFrame, hi: int) -> None:
        self._last_ts = frame.index[hi - 1]
        self._last_close = frame.iat[hi - 1, frame.columns.get_loc("Close")]

    def _continues(self, frame: pd.DataFrame, lo: int, hi: int) -> bool:
        return (
            self._valid
            and lo == self._lo
            and hi >= self._n
            and frame.index[self._n - 1] == self._last_ts
            and frame.iat[self._n - 1, frame.columns.get_loc("Close")] == self._last_close
        )

    def _extend(self, frame: pd.DataFrame, hi: int) -> bool:
        """Fold rows [self._n, hi) into the running buckets. False -> the
        caller reseeds (state is then discarded, never half-updated)."""
        if hi == self._n:
            return True
        new = frame.iloc[self._n:hi]
        if not self._plain_ohlc(new):
            return False
        if self._max_query_ts is not None and new.index[0] <= self._max_query_ts:
            return False
        if new.index[0] < self._last_ts:
            return False
        delta = pd.Timedelta(minutes=self._m)
        rows = new[_OHLC_COLUMNS].to_numpy().tolist()
        for ts, (o, h, lo_, c) in zip(new.index, rows):
            start = _bucket_bounds(ts, self._m)[0]
            if self._starts and start == self._starts[-1]:
                if h > self._high[-1]:
                    self._high[-1] = h
                if lo_ < self._low[-1]:
                    self._low[-1] = lo_
                self._close[-1] = c
            elif not self._starts or start > self._starts[-1]:
                self._starts.append(start)
                self._ends.append(start + delta)
                self._open.append(o)
                self._high.append(h)
                self._low.append(lo_)
                self._close.append(c)
            else:
                return False
        self._n = hi
        self._remember_last(frame, hi)
        # `_out` (complete buckets only) stays valid: every new row is later
        # than the latest `current_ts` already answered, so it cannot land in
        # a bucket that was reported complete.
        return True


class _DayContext(NamedTuple):
    """What the cascade needs from the completed daily bars: today's bias
    (None while fewer than 2 prior daily bars exist) and the last completed
    day's High (None while no prior daily bar exists)."""
    bias: Optional[str]
    prior_day_high: Optional[float]


class _PriorDayContext:
    """Per-DAY cache of `_DayContext`.

    Today's bias and the prior day's High are functions of the completed
    PRIOR days only, so they are constant for the whole session -- rebuilding
    the daily bars (a 20,000-row `.index.date` + groupby) on every 1-minute
    bar of every no-bias day, as the reference expression does, repeats the
    same work all session.

    Exactness: the reference computes from `_bound_tail(frame,
    _DAILY_LOOKBACK_BARS)`, a window that slides forward one row per bar once
    the frame outgrows the bound. Sliding only ever trims the OLDEST rows, and
    `daily_bias(...).iloc[-1]` / the last High read just the last two (one)
    daily bars, so the cached value stays identical to a fresh reference call
    exactly while the window's start is at or before the first row of the
    older bar those values depend on (`_needed_from`). The cache re-derives
    from the reference the moment the window would touch it, whenever the day
    changes, the bound changes, the frame stops continuing the one consumed
    (last row's timestamp/Close, at least as many rows), or the tail index is
    not ascending (then nothing is cached at all).
    """

    def __init__(self):
        self._day = None
        self._value = None

    def get(self, frame: pd.DataFrame, today: datetime.date) -> _DayContext:
        n = len(frame)
        if self._usable(frame, today, n):
            return self._value
        return self._recompute(frame, today, n)

    def _usable(self, frame: pd.DataFrame, today: datetime.date, n: int) -> bool:
        return (
            self._day == today
            and self._bound == _DAILY_LOOKBACK_BARS
            and n >= self._n
            and max(0, n - self._bound) <= self._needed_from
            and frame.index[self._n - 1] == self._last_ts
            and frame.iat[self._n - 1, frame.columns.get_loc("Close")] == self._last_close
        )

    def _recompute(self, frame: pd.DataFrame, today: datetime.date, n: int) -> _DayContext:
        bars = _build_daily_bars(frame, today)
        bias = daily_bias(bars).iloc[-1] if len(bars) >= 2 else None
        high = float(bars["High"].iloc[-1]) if len(bars) else None
        value = _DayContext(bias, high)

        tail = _bound_tail(frame, _DAILY_LOOKBACK_BARS)
        self._day = None  # only re-armed below once every guard input is known
        if not tail.index.is_monotonic_increasing or n == 0:
            return value
        if len(bars) == 0:
            needed_from = n  # no prior rows exist; sliding cannot change that
        else:
            dependent_day = bars.index[-2] if len(bars) >= 2 else bars.index[-1]
            tail_start = n - len(tail)
            needed_from = tail_start + int(
                tail.index.searchsorted(_day_midnight(tail.index, dependent_day), side="left")
            )
        self._day, self._value = today, value
        self._bound = _DAILY_LOOKBACK_BARS
        self._needed_from = needed_from
        self._n = n
        self._last_ts = frame.index[n - 1]
        self._last_close = frame.iat[n - 1, frame.columns.get_loc("Close")]
        return value


def _parse_et_clock(value: str) -> datetime.time:
    return datetime.datetime.strptime(value, "%H:%M").time()


def _make_cascade_signal_fn(
    min_target_r_multiple: float,
    force_flat_et: str,
    divergence_check: Optional[Callable[[pd.Timestamp, pd.Timestamp, pd.Timestamp], bool]] = None,
) -> SignalFn:
    """Shared state machine behind both public factories in this module.
    `divergence_check(bucket_start, bucket_end, current_ts) -> bool`, when
    given, is consulted at the hourly-confirmation step (see step 2 below)
    and must return True to ALLOW that confirmation to proceed, False to
    BLOCK it (a divergence gate); None (the default, used by the
    single-asset Build 1) means no extra gate -- every real hourly
    confirmation is accepted.

    State machine, reset on every new calendar date seen in
    frame.iloc[-1].index (mirrors day_trading_signals.py's own ORB factory
    per-day reset): a per-day `phase` moves through "no_bias" ->
    "awaiting_confirmation" -> "awaiting_entry" -> (armed, then filled) ->
    "in_position" -> "done_today", plus an independent `armed`/`open`
    pair of flags for the entry-decided-but-not-yet-filled / filled-and-
    open states (independent of `phase` because a position must stay
    exit-checked even after `phase` itself has moved on to "done_today").

    1. BIAS: build daily bars from every row in `frame` with date < today.
       Fewer than 2 completed daily bars -> stay "no_bias" (not enough
       history for daily_bias, which needs a predecessor day). Otherwise
       take price_structure.daily_bias's LAST value on those daily bars --
       today's bias, derived causally from yesterday versus the day
       before. "bullish"/"bullish_continuation" -> "awaiting_confirmation";
       anything else (long-only scope) -> stays "no_bias" for today.

    2. HOURLY CONFIRMATION (only while phase == "awaiting_confirmation"):
       build today's 60-minute buckets anchored to 09:30 ET, keeping only
       FULLY COMPLETE buckets (current bar's own timestamp at or past that
       bucket's end). Fewer than 2 complete buckets -> keep waiting.
       Otherwise classify the LAST complete bucket via price_structure.
       classify_reversal_bar; if it is "bullish_reversal" or
       "bullish_break" AND (when `divergence_check` is given) the
       divergence gate allows it AND this exact bucket's end timestamp has
       not already been used (state["confirmed_hour_end"], preventing
       re-triggering on the same confirmation every later 1-minute bar):
       record state["confirmed_hour_end"] and move to "awaiting_entry".

    3. ENTRY (only while phase == "awaiting_entry" and no position is
       armed/open): build today's 5-minute buckets the same complete-only
       way. Fewer than 3 -> keep waiting (fair_value_gaps needs a
       predecessor and successor of the middle candle). Otherwise call
       price_structure.fair_value_gaps; keep bullish gaps whose own
       middle-candle timestamp is at or after state["confirmed_hour_end"],
       drop any gap a LATER 5-minute bar's Low has undercut (traded
       cleanly through without rejecting -- invalidated, never re-used),
       and take the MOST RECENT surviving one as the point of interest
       (poi_low/poi_high). If the LAST complete 5-minute bar's own Low is
       <= poi_high (traded into the gap zone) AND its own
       classify_reversal_bar reading is "bullish_reversal": record
       state["stop_price"] = that bar's own Low (the protected low), set
       state["armed"] = True, and return entry=True. `phase` stays
       "awaiting_entry" -- it only becomes "in_position" once the fill is
       actually captured (step 4), one bar later.

    4. FILL CAPTURE (checked FIRST on every call, before anything else,
       whenever armed and not yet open): the engine's own next-bar-open
       execution convention means THIS call's frame.iloc[-1] is the bar
       whose Open is the real fill price for the entry decided on the
       PREVIOUS call. target = entry_px + max(min_target_r_multiple x
       (entry_px - stop), prior completed day's High - entry_px) -- so the
       target never sits below the prior day's high once price has
       already broken above it. Sets open=True, armed=False, phase =
       "in_position", and deliberately falls through (no early return) to
       step 5's exit check for this SAME bar -- a freshly-filled position
       must still be checked for an immediate stop/target/force-flat hit
       on its own fill bar (mirrors intraday_backtest_engine.py's own
       documented "no continue" convention for a freshly-opened position).

    5. EXIT (checked whenever open, independent of `phase`): stop checked
       first (bar's Low <= stop), then target (bar's High >= target),
       then independently the force-flat cutoff (bar's own wall-clock
       time at or after `force_flat_et`) regardless of stop/target. Any
       exit resets open=False and phase="done_today" -- one trade per day
       maximum, matching this catalog's "only one position tracked at a
       time" convention; a later qualifying setup the same day is never
       taken.

    Never raises on early bars with insufficient history at any of steps
    1-3 -- simply stays in the current phase and returns SignalResult().
    """
    min_r = float(min_target_r_multiple)
    force_flat_time = _parse_et_clock(force_flat_et)

    state = {
        "last_date": None,
        "phase": "no_bias",
        "confirmed_hour_end": None,
        "armed": False,
        "open": False,
        "stop_price": None,
        "target_price": None,
    }

    # Per-instance caches (a fresh factory call => fresh, empty caches, like
    # `state`). They are exact functions of the frames they are shown -- see
    # each class's docstring for how continuity is proven or the cache is
    # bypassed -- so they change speed only, never a signal.
    prior_day = _PriorDayContext()
    hourly_buckets = _TodayBuckets(60)
    five_min_buckets = _TodayBuckets(5)

    def signal_fn(frame: pd.DataFrame, i: int):
        current_ts = frame.index[-1]
        today = current_ts.date()

        # --- new trading day: defensive reset of all per-day state -----
        if state["last_date"] != today:
            state["last_date"] = today
            state["phase"] = "no_bias"
            state["confirmed_hour_end"] = None
            state["armed"] = False
            state["open"] = False
            state["stop_price"] = None
            state["target_price"] = None

        entry_signal = False
        exit_signal = False

        # --- STEP 4: fill capture -- checked first, before anything else
        if state["armed"] and not state["open"]:
            entry_px = float(frame["Open"].iloc[-1])
            stop = state["stop_price"]
            prior_day_high = prior_day.get(frame, today).prior_day_high
            if prior_day_high is None:
                prior_day_high = entry_px
            target = entry_px + max(
                min_r * (entry_px - stop), prior_day_high - entry_px
            )
            state["target_price"] = target
            state["open"] = True
            state["armed"] = False
            state["phase"] = "in_position"
            # Deliberately NO return -- fall through to step 5's exit
            # check for this same (fill) bar too.

        # --- STEP 1: daily bias ----------------------------------------
        if state["phase"] == "no_bias":
            if prior_day.get(frame, today).bias in ("bullish", "bullish_continuation"):
                state["phase"] = "awaiting_confirmation"

        # --- STEP 2: hourly confirmation ---------------------------------
        if state["phase"] == "awaiting_confirmation":
            hourly_bars = hourly_buckets.complete_primary(frame, today, current_ts)
            if len(hourly_bars) >= 2:
                last_class = classify_reversal_bar(hourly_bars).iloc[-1]
                if last_class in ("bullish_reversal", "bullish_break"):
                    bucket_start = hourly_bars.index[-1]
                    bucket_end = bucket_start + pd.Timedelta(minutes=60)
                    allowed = True
                    if divergence_check is not None:
                        allowed = divergence_check(bucket_start, bucket_end, current_ts)
                    if allowed and state["confirmed_hour_end"] != bucket_end:
                        state["confirmed_hour_end"] = bucket_end
                        state["phase"] = "awaiting_entry"

        # --- STEP 3: entry (fair-value-gap retest-and-reject) -----------
        # Gated to strictly before force_flat_time: the engine fills an
        # entry signal at the NEXT bar's open, which -- for a bar this
        # close to the configured force-flat cutoff -- can land on the
        # FOLLOWING calendar day if today's remaining bars run out first
        # (an entry fired at 15:55+ could fill next-day, and the unconditional
        # per-day reset at the top of this function would then wipe
        # `armed` before step 4 ever captured that fill, leaving the
        # engine's own position invisible to this closure's exit logic --
        # it would then ride an extra full session past the advertised
        # max_hold_days=1 same-day-flat invariant before the engine's own
        # end-of-day backstop eventually closed it). Requiring the
        # decision bar to be strictly before force_flat_time guarantees at
        # least one more same-day bar remains for the fill on any normal
        # session; a genuine NYSE early-close day (e.g. the day after
        # Thanksgiving) that closes before force_flat_time is a disclosed,
        # unresolved residual of this same class -- this factory has no
        # trading-calendar input to detect that case in advance.
        if (
            state["phase"] == "awaiting_entry"
            and not state["armed"]
            and not state["open"]
            and current_ts.time() < force_flat_time
        ):
            five_min_bars = five_min_buckets.complete_primary(frame, today, current_ts)
            if len(five_min_bars) >= 3:
                gaps = fair_value_gaps(five_min_bars)
                bullish_gaps = gaps[gaps["direction"] == "bullish"]
                qualifying = bullish_gaps[bullish_gaps.index >= state["confirmed_hour_end"]]

                live_low = None
                live_high = None
                for gap_ts in sorted(qualifying.index):
                    gap_low = float(qualifying.loc[gap_ts, "gap_low"])
                    later = five_min_bars[five_min_bars.index > gap_ts]
                    undercut = bool((later["Low"] < gap_low).any())
                    if not undercut:
                        live_low = gap_low
                        live_high = float(qualifying.loc[gap_ts, "gap_high"])

                if live_low is not None:
                    last_bar = five_min_bars.iloc[-1]
                    last_class = classify_reversal_bar(five_min_bars).iloc[-1]
                    if float(last_bar["Low"]) <= live_high and last_class == "bullish_reversal":
                        state["stop_price"] = float(last_bar["Low"])
                        state["armed"] = True
                        entry_signal = True

        # --- STEP 5: exit (stop / target / force-flat) ------------------
        if state["open"]:
            row = frame.iloc[-1]
            low = float(row["Low"])
            high = float(row["High"])
            bar_time = current_ts.time()
            if low <= state["stop_price"]:
                exit_signal = True
            elif high >= state["target_price"]:
                exit_signal = True
            elif bar_time >= force_flat_time:
                exit_signal = True

            if exit_signal:
                state["open"] = False
                state["phase"] = "done_today"
                state["stop_price"] = None
                state["target_price"] = None

        return SignalResult(entry=entry_signal, exit=exit_signal)

    return signal_fn


def make_fvg_reversal_cascade_signal_fn(
    min_target_r_multiple: float = 2.0,
    force_flat_et: str = "15:55",
) -> SignalFn:
    """Daily-bias / hourly-confirmation / 5-minute fair-value-gap retest
    cascade, single-asset (LONG SIDE ONLY -- disclosed scope, matching
    this catalog's established convention, e.g. day_trading_signals.
    make_orb_5min_atr_stop_scalp_signal_fn's own long-only disclosure).

    Mechanism: each day, first derive TODAY's directional bias from
    price_structure.daily_bias applied to the last two completed daily
    bars (yesterday versus the day before) -- only a "bullish" or
    "bullish_continuation" reading is tradeable; anything else skips the
    day entirely (the mechanism's bearish case is long-only-scope out of
    bounds, not mirrored as a short). Once biased bullish, wait for the
    first fully-completed 60-minute bucket (anchored 09:30 ET) whose
    price_structure.classify_reversal_bar reading is "bullish_reversal" or
    "bullish_break" versus its own predecessor bucket -- same-direction
    structure on the hourly chart confirming the daily bias. Once
    confirmed, watch 5-minute buckets (same 09:30-anchored, complete-only
    discipline) for a bullish fair value gap (price_structure.
    fair_value_gaps) whose own middle-candle timestamp is at or after the
    hourly confirmation's own end -- a gap formed before confirmation
    doesn't count (disclosed choice: the point of interest must form
    after confirmation). A gap is dropped, never re-used, once a later
    5-minute bar's Low trades cleanly below its own gap_low without
    rejecting (invalidated); among gaps that still qualify, the most
    recent one is the live point of interest. Entry triggers the instant
    the latest complete 5-minute bar's own Low trades into that gap's
    upper edge (<= gap_high) AND classify_reversal_bar reads
    "bullish_reversal" on that same bar versus its own predecessor -- a
    retest-and-reject of the gap zone. The engine fills at the next bar's
    open; the stop is the trigger bar's own Low (the protected low) and
    the target is max(min_target_r_multiple x initial risk, the prior
    completed day's High minus the real fill price) -- so a fill that has
    already broken well above yesterday's high still gets at least that
    many R of room, never a target sitting below a level price already
    cleared. Every bar while a position is believed open, the bar's own
    High/Low is checked against target and stop (stop checked first as
    the conservative tie-break when a single bar's range could plausibly
    have touched both); independently, once the bar's own wall-clock time
    is at or after `force_flat_et`, the position is flattened regardless.
    One trade per day maximum -- once a trade closes (or the day's bias/
    confirmation path never qualifies), no later qualifying setup the same
    day is taken.

    Position state (per-day phase, armed-but-unfilled flag, open/closed
    flag, stop/target prices, the hourly confirmation already used) is
    tracked in a closure-local dict and defensively reset on every date
    change, since every trade this factory opens is mechanically
    guaranteed to close the same session it opened in.
    """
    return _make_cascade_signal_fn(min_target_r_multiple, force_flat_et, divergence_check=None)


def make_cross_asset_divergence_fvg_cascade_signal_fn(
    secondary_bars: pd.DataFrame,
    min_target_r_multiple: float = 2.0,
    force_flat_et: str = "15:55",
) -> SignalFn:
    """Same cascade as make_fvg_reversal_cascade_signal_fn (bias / hourly
    confirmation / 5-minute fair-value-gap retest, LONG SIDE ONLY), with
    exactly one additional gate inserted at the hourly-confirmation step:
    the primary's own hourly confirmation is only ACCEPTED when a second,
    correlated instrument (`secondary_bars` -- a full pre-fetched minute-
    bar OHLCV frame, same shape/index convention as intraday_data.
    fetch_intraday_bars's own output: tz-aware "America/New_York"
    DatetimeIndex) FAILS to show the identical structure at the identical
    hour -- a genuine cross-asset divergence, not just a same-direction
    move both instruments happen to share.

    Mechanism: whenever the primary's own hourly bucket would confirm
    (price_structure.classify_reversal_bar reads "bullish_reversal" or
    "bullish_break" on it), first slice `secondary_bars` to timestamps at
    or before the primary's own current bar (defensive -- the secondary
    series is never allowed to see into the future relative to the
    primary), build the SECONDARY's own 60-minute buckets for today using
    the identical 09:30-ET-anchored, complete-only bucketing rule, and
    classify its SAME bucket (identical start/end timestamps) the same
    way. If the secondary ALSO reads "bullish_reversal" or
    "bullish_break" at that hour, both instruments are moving together --
    not a divergence -- so the primary's own confirmation is REJECTED
    (blocked) and the cascade keeps waiting in "awaiting_confirmation".
    If the secondary's own matching bucket is not covered by
    `secondary_bars` yet (no data reaches that far), the disclosed
    permissive default treats that as "does not confirm" -- there is
    nothing to compare against, so the primary's own confirmation is used
    on its own rather than blocking on missing data. Everything else
    (bias, entry, fill capture, exit) is IDENTICAL to the single-asset
    version and uses only the PRIMARY frame -- `secondary_bars` is
    read-only reference data for this one gate, never traded.
    """
    sec_hourly_buckets = _TodayBuckets(60)

    def divergence_check(
        bucket_start: pd.Timestamp, bucket_end: pd.Timestamp, current_ts: pd.Timestamp
    ) -> bool:
        # Identical to slicing `secondary_bars` to <= current_ts, keeping
        # today's rows and bucketing them -- but served from a cache that
        # neither masks nor date-converts the secondary's FULL history on
        # every call (see _TodayBuckets.complete_secondary).
        sec_hourly = sec_hourly_buckets.complete_secondary(
            secondary_bars, bucket_start.date(), current_ts
        )
        if sec_hourly is None:
            return True
        if bucket_start not in sec_hourly.index:
            return True
        sec_class = classify_reversal_bar(sec_hourly).loc[bucket_start]
        return sec_class not in ("bullish_reversal", "bullish_break")

    return _make_cascade_signal_fn(
        min_target_r_multiple, force_flat_et, divergence_check=divergence_check
    )
