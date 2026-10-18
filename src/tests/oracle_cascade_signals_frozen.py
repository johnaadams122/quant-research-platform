# FROZEN ORACLE -- do not edit its logic, do not import from production code.
#
# The reference (uncached) implementation of src/mining/cascade_signals.py's
# two signal factories: it rebuilds the daily bars and today's buckets from
# scratch on every call (no incremental-bucket or per-day-bias cache), with
# absolute imports so it can live under tests/. It exists solely so
# test_cascade_signals_incremental.py can prove the cached module is
# behaviour-identical to this reference on random frames and random call
# orders. Delete it (and that test) if that equivalence check is ever
# dropped.
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
    the following calendar day: an ungated late entry would fill next-day
    while this closure's own per-day reset had already wiped its "armed"
    bookkeeping, making the engine's own resulting position invisible to
    this closure's exit logic for a full extra session. Residual, disclosed
    and unresolved:
    a genuine NYSE early-close day (e.g. the day after Thanksgiving) that
    closes BEFORE `force_flat_et` could still exhibit this same failure
    mode in its own final minutes -- this factory has no trading-calendar
    input to detect that case in advance.
"""
from __future__ import annotations

import datetime
from typing import Callable, Optional

import pandas as pd

from mining.intraday_backtest_engine import SignalFn, SignalResult
from mining.price_structure import classify_reversal_bar, daily_bias, fair_value_gaps


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
    starts = [_bucket_bounds(ts, bucket_minutes)[0] for ts in today_frame.index]
    tmp = today_frame.copy()
    tmp["_bucket_start"] = starts
    grouped = tmp.groupby("_bucket_start").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    grouped = grouped.sort_index()
    if grouped.empty:
        return grouped
    ends = grouped.index + pd.Timedelta(minutes=bucket_minutes)
    return grouped[ends <= current_ts]


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
            daily_bars = _build_daily_bars(frame, today)
            if daily_bars.empty:
                prior_day_high = entry_px
            else:
                prior_day_high = float(daily_bars["High"].iloc[-1])
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
            daily_bars = _build_daily_bars(frame, today)
            if len(daily_bars) >= 2:
                today_bias = daily_bias(daily_bars).iloc[-1]
                if today_bias in ("bullish", "bullish_continuation"):
                    state["phase"] = "awaiting_confirmation"

        # --- STEP 2: hourly confirmation ---------------------------------
        if state["phase"] == "awaiting_confirmation":
            hourly_bars = _build_complete_buckets(
                _today_frame(frame, today), current_ts, 60
            )
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
            five_min_bars = _build_complete_buckets(
                _today_frame(frame, today), current_ts, 5
            )
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
    def divergence_check(
        bucket_start: pd.Timestamp, bucket_end: pd.Timestamp, current_ts: pd.Timestamp
    ) -> bool:
        sec = secondary_bars[secondary_bars.index <= current_ts]
        if sec.empty:
            return True
        today = bucket_start.date()
        sec_today = sec[sec.index.date == today]
        sec_hourly = _build_complete_buckets(sec_today, current_ts, 60)
        if bucket_start not in sec_hourly.index:
            return True
        sec_class = classify_reversal_bar(sec_hourly).loc[bucket_start]
        return sec_class not in ("bullish_reversal", "bullish_break")

    return _make_cascade_signal_fn(
        min_target_r_multiple, force_flat_et, divergence_check=divergence_check
    )
