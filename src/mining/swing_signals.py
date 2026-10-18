"""Real swing-trading signal_fn factories for the curated swing candidates in
the mining catalog. Every factory returns a
SignalFn matching src/mining/intraday_backtest_engine.SignalResult/SignalFn's
exact contract.
"""
from __future__ import annotations

import datetime as dt
import datetime as _dt
from collections import deque as _deque
from datetime import time as _time
from datetime import time as dtime
from datetime import timedelta

import numpy as np
import pandas as pd

from .intraday_backtest_engine import SignalFn, SignalResult


# --- earnings_reaction_range_breakout_pead ---
def make_earnings_reaction_range_breakout_pead_signal_fn(
    earnings_events: list[dict],
    min_gap_pct: float = 3.0,
    opening_range_minutes: int = 15,
    entry_window_et: str = "09:30-09:45",
    max_hold_days: int = 20,
    trailing_stop_atr_mult: float = 2.5,
    atr_period: int = 14,
):
    """Earnings-reaction opening-range breakout with a post-earnings-drift (PEAD) hold.

    LONG-ONLY DISCLOSED SCOPE: the mechanism describes a mirrored short variant for a
    negative-surprise gap-down; per the engine's whole-call `direction` parameter, this
    factory implements the long side only. Consequently only POSITIVE-surprise earnings
    events (surprise_pct > 0) can ever seed a setup here -- a negative-surprise event is
    simply never scheduled as a reaction day, since its only tradable leg (the short) is
    out of scope for this factory. This is a disclosed scoping decision, not a silent gap.

    Reaction-day resolution (per event, using ONLY that event's own already-realized
    historical date/hour -- never a later event in the list): hour >= 16 (after-close)
    resolves to the first NEW calendar trading day observed in the incoming bar stream
    strictly after the event date (found causally, bar-by-bar, as those days actually
    arrive -- never by scanning ahead in `frame`); hour < 16 (pre-market or during-hours)
    resolves to the event's own date. On that resolved reaction day, if the day's actual
    RTH open gaps up by at least `min_gap_pct` versus the prior COMPLETE trading day's
    close (derived from bars already consumed, never `frame`'s future rows), the setup
    activates for that single day only.

    Opening range: built minute-by-minute from real bars in the window
    [start of `entry_window_et`, start + `opening_range_minutes`) -- both parameters are
    used together to derive the actual range-close clock time. Once that window closes,
    every subsequent bar THE SAME trading day is checked for a break of the range's High
    (real-time, using that bar's own High) -- the first such break fires entry=True. Only
    one entry per event; if no break occurs before the RTH close, that event's setup
    expires unused (no next-day carryover, per the mechanism's own "first 15 minutes of
    that day" framing).

    Hold / exit: since the engine's signal_fn is only ever told whether to emit entry/exit
    per bar (no external position-management input), this closure tracks its own implied
    open-trade state. A daily ATR(atr_period) -- atr_period defaults to 14 trading days,
    a disclosed internal default since the candidate's own `parameters` dict does not name
    one -- is computed from FULLY COMPLETED prior trading days only (never the still-
    forming current day) and refreshed once per new trading day, so the trailing stop is a
    genuinely trailing channel (recomputed daily), not frozen at entry. The stop trails the
    highest intraday High seen since entry by `trailing_stop_atr_mult` ATRs; exit fires
    (exit=True) the bar THIS bar's Close closes at or below that level, consistent with the
    engine's own same-bar-close exit execution.

    max_hold_days is tracked by this closure via its own `state["day_count"]` difference
    (`days_held = state["day_count"] - pos["entry_day_count"]`, firing at
    `days_held >= max_hold_days`) rather than by asking the engine. This lines up exactly
    with `run_intraday_backtest`'s OWN native max_hold_days force-close (which
    `run_swing_mining.py` also applies independently, passing this SAME
    candidate['max_hold_days'] value) -- NOT by coincidence, but because
    `pos["entry_day_count"]` is captured on the call that DECIDES entry (the breakout
    bar, one trading day before the engine's actual next-bar-open fill), so this
    closure's day-count is already "one ahead" of a naive same-day count in exactly the
    way that cancels the engine's own "entry day counts as day 1" convention: the
    closure's own hit_maxhold and the engine's own hit_time_stop always land on the same
    trading day (traceable call-by-call through `run_intraday_backtest`), so this
    closure's internal notion of "position open" never lags what the
    engine has already force-closed. max_hold_days / trailing-stop day counts are
    measured from the entry signal's own trading day rather than the engine's actual
    next-bar fill (the actual fill lands one bar later at the engine's next-open
    execution -- at most a one-minute-bar approximation of the true hold length,
    disclosed).

    Only one position is tracked at a time: a new qualifying reaction-day setup that
    arrives while a prior trade is still open is not evaluated for entry.

    Early-history / insufficient-lookback bars (fewer than atr_period+1 completed daily
    bars, or no prior completed trading day at all) never activate a new setup and simply
    return SignalResult() -- never raise.
    """

    try:
        _start_str, _, _end_str = entry_window_et.partition("-")
        range_start_time = _dt.datetime.strptime(_start_str.strip(), "%H:%M").time()
    except (ValueError, AttributeError):
        range_start_time = _dt.time(9, 30)
    _anchor = _dt.datetime(2000, 1, 1)
    range_end_time = (
        _dt.datetime.combine(_anchor, range_start_time)
        + _dt.timedelta(minutes=opening_range_minutes)
    ).time()

    pending_events = []
    same_day_events = []
    try:
        _sorted_events = sorted(earnings_events or [], key=lambda e: e.get("date", ""))
    except Exception:
        _sorted_events = list(earnings_events or [])
    for ev in _sorted_events:
        try:
            surprise = float(ev.get("surprise_pct", 0.0))
        except (TypeError, ValueError):
            continue
        if surprise <= 0:
            continue
        try:
            ev_date = _dt.datetime.strptime(str(ev["date"]), "%Y-%m-%d").date()
        except (KeyError, ValueError, TypeError):
            continue
        hour = ev.get("hour", 16)
        record = {"event_date": ev_date, "resolved_date": None}
        if hour is not None and hour >= 16:
            pending_events.append(record)
        else:
            same_day_events.append(record)

    state = {
        "cur_day_date": None,
        "cur_day_high": None,
        "cur_day_low": None,
        "last_close": None,
        "daily_bars": [],
        "reaction_by_date": {},
        "today": None,
        "position": None,
        "atr_value": None,
        "day_count": -1,
    }

    def compute_atr():
        bars = state["daily_bars"]
        if len(bars) < atr_period + 1:
            return None
        trs = []
        for k in range(len(bars) - atr_period, len(bars)):
            cur = bars[k]
            prev = bars[k - 1]
            if cur["close"] is None or prev["close"] is None:
                continue
            tr = max(
                cur["high"] - cur["low"],
                abs(cur["high"] - prev["close"]),
                abs(cur["low"] - prev["close"]),
            )
            trs.append(tr)
        if not trs:
            return None
        return sum(trs) / len(trs)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if frame is None or len(frame) == 0:
            return SignalResult()

        row = frame.iloc[-1]
        ts = frame.index[-1]
        cur_date = ts.date()

        o = row.get("Open")
        h = row.get("High")
        l = row.get("Low")
        c = row.get("Close")
        if o is None or h is None or l is None or c is None:
            return SignalResult()
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            return SignalResult()

        is_new_day = cur_date != state["cur_day_date"]

        if is_new_day:
            state["day_count"] += 1

            if state["cur_day_date"] is not None:
                state["daily_bars"].append({
                    "date": state["cur_day_date"],
                    "high": state["cur_day_high"],
                    "low": state["cur_day_low"],
                    "close": state["last_close"],
                })

            state["cur_day_date"] = cur_date
            state["cur_day_high"] = h
            state["cur_day_low"] = l

            for ev in pending_events:
                if ev["resolved_date"] is None and ev["event_date"] < cur_date:
                    ev["resolved_date"] = cur_date
                    state["reaction_by_date"][cur_date] = ev
            for ev in same_day_events:
                if ev["resolved_date"] is None and ev["event_date"] == cur_date:
                    ev["resolved_date"] = cur_date
                    state["reaction_by_date"][cur_date] = ev

            state["atr_value"] = compute_atr()

            today = {"range_high": None, "range_low": None, "setup_active": False, "traded": False}
            if state["position"] is None:
                ev = state["reaction_by_date"].get(cur_date)
                daily_bars = state["daily_bars"]
                if ev is not None and daily_bars:
                    prior_close = daily_bars[-1]["close"]
                    if (
                        prior_close is not None
                        and not pd.isna(prior_close)
                        and prior_close > 0
                        and state["atr_value"] is not None
                    ):
                        gap_pct = (o - prior_close) / prior_close * 100.0
                        if gap_pct >= min_gap_pct:
                            today["setup_active"] = True
            state["today"] = today
        else:
            if h > state["cur_day_high"]:
                state["cur_day_high"] = h
            if l < state["cur_day_low"]:
                state["cur_day_low"] = l

        today = state["today"]
        entry = False
        exit_ = False

        pos = state["position"]
        if pos is not None:
            if h > pos["highest"]:
                pos["highest"] = h
            atr = state["atr_value"]
            hit_stop = False
            if atr is not None:
                stop_level = pos["highest"] - trailing_stop_atr_mult * atr
                hit_stop = c <= stop_level
            days_held = state["day_count"] - pos["entry_day_count"]
            hit_maxhold = days_held >= max_hold_days
            if hit_stop or hit_maxhold:
                exit_ = True
                state["position"] = None
        elif today is not None and today.get("setup_active") and not today.get("traded"):
            t = ts.time()
            if range_start_time <= t < range_end_time:
                if today["range_high"] is None:
                    today["range_high"] = h
                    today["range_low"] = l
                else:
                    if h > today["range_high"]:
                        today["range_high"] = h
                    if l < today["range_low"]:
                        today["range_low"] = l
            elif t >= range_end_time and today["range_high"] is not None:
                if h > today["range_high"]:
                    entry = True
                    today["traded"] = True
                    state["position"] = {
                        "highest": h,
                        "entry_day_count": state["day_count"],
                    }

        state["last_close"] = c
        return SignalResult(entry=entry, exit=exit_)

    return signal_fn


# --- anchored_vwap_reclaim_continuation ---
def make_anchored_vwap_reclaim_continuation_signal_fn(
    anchor_lookback_days: int = 20,
    reclaim_bar_minutes: int = 5,
    min_volume_multiple_of_bar_avg: float = 1.3,
    max_hold_days: int = 20,
    exit_rule: str = "daily_close_below_anchored_vwap",
) -> "SignalFn":
    """
    Long-only Anchored VWAP Reclaim Continuation.

    Anchor: a true "significant multi-week swing low" needs forward
    confirmation (future higher lows) that would require look-ahead, so it
    is approximated -- disclosed, not silent -- as the day with the lowest
    daily Low among the trailing `anchor_lookback_days` COMPLETE trading
    days (strictly before the current day). This rolling-low proxy is
    recomputed on every entry-search bar while flat, so the candidate
    anchor can drift forward as older lows roll out of the window; once a
    trade is opened the anchor that produced entry is locked into
    closure state and used unchanged for that trade's hold/exit checks,
    matching "anchored" VWAP semantics (fixed start point, continuously
    updating running value).

    Anchored VWAP is computed by accumulating Volume-weighted per-bar VWAP
    (Alpaca's own per-bar VWAP column; falls back to Close if that column
    is absent) from the anchor day's first minute bar through the current
    bar `i`, using only real 1-minute bars actually present in `frame` --
    it is recomputed fresh on each evaluation rather than maintained
    incrementally, so cost scales with anchor-to-now distance (caching was
    deliberately not added; not assumed required).

    Entry ("reclaim"): evaluated only on the minute bar that closes a
    `reclaim_bar_minutes`-minute bucket anchored to the 09:30 ET session
    open (e.g. bars ending 09:34, 09:39, ... for the default 5-minute
    bucket). A bucket's aggregated Close is compared to the anchored VWAP
    as of that bucket's close; a reclaim fires when the immediately PRIOR
    confirmed bucket's close was below the anchored VWAP and this bucket's
    close is above it (a same-anchor "cross"). To avoid a spurious cross
    reading when the rolling anchor itself shifts day to day, the first
    bucket evaluated under a newly-selected anchor only records a baseline
    (below/above) and cannot itself fire a cross.

    Volume confirmation: the confirming bucket's summed Volume must exceed
    `min_volume_multiple_of_bar_avg` times the historical average Volume
    for that SAME time-of-day bucket (e.g. all prior days' 09:30-09:34
    bars), using only days strictly before today; requires at least 5 such
    prior-day observations or the entry is skipped (insufficient history,
    not a crash).

    Entry fires SignalResult(entry=True) on the confirming bucket's closing
    minute bar. The mechanism describes filling "at the close of that
    confirmation bar"; this engine instead fills every entry at the NEXT
    bar's open by construction -- a one-bar structural approximation of
    that timing that this signal function does not control.

    Hold / exit: `exit_rule` is read, and only "daily_close_below_anchored_vwap"
    is implemented (the only value this candidate specifies). On each
    day's end-of-session bar -- approximated causally as the bar whose ET
    time-of-day is >= 15:59, since no future bar can be inspected to find
    the true session close; this can miss or delay detection on
    non-standard early-close half-days, a disclosed limitation -- while a
    position is open: exits if that day's Close is below the (fixed-
    anchor) anchored VWAP, or if the count of end-of-day checks since
    entry (entry day counted as day 1) has reached `max_hold_days`. Exit
    fires SignalResult(exit=True) on that same end-of-day bar, matching
    the engine's same-bar-close exit convention.

    Scope: LONG side only, per this pipeline's current single-direction
    backtest convention -- the mechanism does not describe a short mirror
    for this family, so nothing is omitted here.
    """
    import datetime


    state = {
        "in_position": False,
        "anchor_date": None,
        "entry_day": None,
        "hold_day_count": 0,
        "last_checked_day": None,
        "last_anchor_date_seen": None,
        "last_bucket_below_vwap": None,
        "candidate_anchor_computed_for_day": None,
        "candidate_anchor_date": None,
    }

    # Performance bounds (recomputing over the whole frame on every bar is quadratic): the engine hands this function a FRESH SLICE of the
    # whole frame on every call, growing by one row each time, so any
    # operation over the ENTIRE received frame is O(len(frame)) per call
    # and O(n^2) total across a real multi-year backtest. Two separate
    # bounds because the two call sites have very different real needs:
    # the entry-search anchor is always picked fresh from the trailing
    # anchor_lookback_days window, so 400 bars/day (comfortably covers a
    # full ~390-bar RTH session, matching the existing
    # frame.tail(12000) convention in day_trading_signals.py) times that
    # window is enough -- but once locked into a position the SAME
    # anchor stays fixed for up to max_hold_days more trading days, so
    # the held-position check needs the wider bound.
    _ANCHOR_SELECTION_WINDOW_BARS = (anchor_lookback_days + 5) * 400
    _HOLD_ANCHOR_WINDOW_BARS = (anchor_lookback_days + max_hold_days + 10) * 400

    def _anchored_vwap_value(frame, anchor_date, window_bars):
        if anchor_date is None:
            return None
        recent = frame.tail(window_bars)
        if recent.empty:
            return None
        # A Timestamp-range comparison is vectorized; `.index.date` is
        # not -- it builds a Python datetime.date object per row via a
        # Python-level loop, which is costly even inside a bounded
        # window.
        anchor_ts = pd.Timestamp(anchor_date, tz=recent.index.tz)
        sub = recent.loc[recent.index >= anchor_ts]
        if sub.empty:
            return None
        vol = sub["Volume"]
        total_vol = vol.sum()
        if total_vol is None or total_vol <= 0:
            return None
        price = sub["VWAP"] if "VWAP" in sub.columns else sub["Close"]
        return float((price * vol).sum() / total_vol)

    def _time_of_day_avg_volume(frame, today, bucket_time, bar_minutes):
        # This is a genuinely UNBOUNDED "all prior days" historical
        # average by design (see the docstring above: "the historical
        # average Volume for that SAME time-of-day bucket (e.g. all
        # prior days' 09:30-09:34 bars)") -- bounding the window to a
        # fixed recent lookback (for example ~25 trading days) would
        # change ITS VALUE, not just its cost: the average then differs
        # from the all-prior-days definition, which can suppress an entry
        # the definition produces or, on other inputs, produce one it
        # does not. Unlike
        # _anchored_vwap_value above (bounding is provably value-
        # preserving there by construction -- the anchor is always
        # picked fresh from a genuinely bounded trailing window), this
        # one must scan full history to stay correct. The expensive part
        # of an unbounded scan is not its size but
        # `.index.date`/`.index.time`, which build a Python object per
        # row via an unvectorized loop; `.normalize()` and integer
        # hour/minute arithmetic are vectorized and give the identical
        # comparison much faster. This function is also only reached on
        # a crossed_above signal event, not every bucket-close, so its
        # call frequency is far lower than the daily resample's.
        if frame.empty:
            return None
        today_ts = pd.Timestamp(today, tz=frame.index.tz)
        prior = frame.loc[frame.index.normalize() < today_ts]
        if prior.empty:
            return None
        bucket_start_minute = bucket_time.hour * 60 + bucket_time.minute
        bucket_end_minute = bucket_start_minute + bar_minutes - 1
        minute_of_day = prior.index.hour * 60 + prior.index.minute
        mask = (minute_of_day >= bucket_start_minute) & (minute_of_day <= bucket_end_minute)
        same_slot = prior.loc[mask]
        if same_slot.empty:
            return None
        daily_sums = same_slot.groupby(same_slot.index.date)["Volume"].sum()
        if len(daily_sums) < 5:
            return None
        return float(daily_sums.mean())

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        idx = frame.index
        ts = idx[-1]
        today = ts.date()

        # ---- Hold / exit path ----
        # (Runs ahead of the daily-bar resample below -- this branch
        # never reads `daily`/`daily_dates`, only the entry path does, so
        # computing that full-frame resample before checking
        # state["in_position"] would cost an O(len(frame)) recompute on
        # every held-position bar for no reason. Skipping the entry
        # path's daily-history guard here changes no behavior: that
        # guard can never fire for a position that is
        # already open (an entry can only have been taken once
        # `len(daily) >= anchor_lookback_days + 2` was already true on an
        # earlier, smaller frame, and `frame` only grows from there).
        if state["in_position"]:
            is_eod = ts.time() >= datetime.time(15, 59)
            if is_eod and state["last_checked_day"] != today:
                state["last_checked_day"] = today
                state["hold_day_count"] += 1

                exit_signal = False
                if exit_rule == "daily_close_below_anchored_vwap":
                    anchor_vwap_now = _anchored_vwap_value(
                        frame, state["anchor_date"], _HOLD_ANCHOR_WINDOW_BARS
                    )
                    today_close = frame["Close"].iloc[-1]
                    if anchor_vwap_now is not None and today_close < anchor_vwap_now:
                        exit_signal = True

                if not exit_signal and state["hold_day_count"] >= max_hold_days:
                    exit_signal = True

                if exit_signal:
                    state["in_position"] = False
                    state["anchor_date"] = None
                    state["entry_day"] = None
                    state["hold_day_count"] = 0
                    state["last_checked_day"] = None
                    state["last_anchor_date_seen"] = None
                    state["last_bucket_below_vwap"] = None
                    return SignalResult(exit=True)
            return SignalResult()

        # ---- Entry path ----
        minute_of_day = ts.hour * 60 + ts.minute
        offset = minute_of_day - (9 * 60 + 30)
        if offset < 0:
            return SignalResult()
        if (offset % reclaim_bar_minutes) != (reclaim_bar_minutes - 1):
            return SignalResult()

        # Only reached on a bucket-closing bar while flat -- same
        # performance rationale as above: placed after the two early
        # returns just above, it skips the 4-out-of-5 bars whose result
        # would be discarded for the default 5-minute bucket.
        # The rolling anchor depends only on COMPLETED days strictly
        # before today (never today's own partial data -- completed_mask
        # below), so it cannot change between bucket-closes on the same
        # calendar day. Recomputing the daily resample on every one of a
        # session's ~78 bucket-closes would be pure repeated work, so it
        # is cached once per day. Still bounded to a recent tail before
        # the resample itself (recomputing over the whole frame on every
        # bar is quadratic).
        if state["candidate_anchor_computed_for_day"] != today:
            recent_for_anchor = frame.tail(_ANCHOR_SELECTION_WINDOW_BARS)
            daily = recent_for_anchor.resample("1D").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna(subset=["Open"])
            candidate_anchor_date = None
            if len(daily) >= anchor_lookback_days + 2:
                daily_dates = daily.index.date
                completed_mask = daily_dates < today
                completed_daily = daily.loc[completed_mask]
                if len(completed_daily) >= anchor_lookback_days:
                    window = completed_daily.iloc[-anchor_lookback_days:]
                    candidate_anchor_date = window["Low"].idxmin().date()
            state["candidate_anchor_computed_for_day"] = today
            state["candidate_anchor_date"] = candidate_anchor_date

        anchor_date = state["candidate_anchor_date"]
        if anchor_date is None:
            return SignalResult()

        anchor_vwap = _anchored_vwap_value(frame, anchor_date, _ANCHOR_SELECTION_WINDOW_BARS)
        if anchor_vwap is None:
            return SignalResult()

        bucket_start = ts - pd.Timedelta(minutes=offset % reclaim_bar_minutes)
        recent = frame.tail(_ANCHOR_SELECTION_WINDOW_BARS)
        bucket = recent.loc[(recent.index >= bucket_start) & (recent.index <= ts)]
        if bucket.empty:
            return SignalResult()
        bucket_close = float(bucket["Close"].iloc[-1])
        bucket_volume = float(bucket["Volume"].sum())
        is_below_now = bucket_close < anchor_vwap

        if state["last_anchor_date_seen"] != anchor_date:
            state["last_anchor_date_seen"] = anchor_date
            state["last_bucket_below_vwap"] = is_below_now
            return SignalResult()

        was_below = state["last_bucket_below_vwap"]
        state["last_bucket_below_vwap"] = is_below_now
        crossed_above = (was_below is True) and (bucket_close > anchor_vwap)
        if not crossed_above:
            return SignalResult()

        avg_vol = _time_of_day_avg_volume(frame, today, bucket_start.time(), reclaim_bar_minutes)
        if avg_vol is None or avg_vol <= 0:
            return SignalResult()
        if bucket_volume < min_volume_multiple_of_bar_avg * avg_vol:
            return SignalResult()

        state["in_position"] = True
        state["anchor_date"] = anchor_date
        state["entry_day"] = today
        state["hold_day_count"] = 0
        state["last_checked_day"] = None
        state["last_anchor_date_seen"] = None
        state["last_bucket_below_vwap"] = None
        return SignalResult(entry=True)

    return signal_fn


# --- vcp_breakout_volume_confirmation_swing ---
def make_vcp_breakout_volume_confirmation_swing_signal_fn(
    min_contractions: int = 2,
    pivot_confirmation_window_et: str = "09:30-10:00",
    min_opening_volume_pct_of_20d_avg: float = 50.0,
    max_hold_days: int = 20,
    swing_lookback_days: int = 65,
    fractal_k: int = 3,
    target_risk_multiple: float = 2.0,
    trail_ma_days: int = 10,
) -> SignalFn:
    """Long-only Volatility-Contraction-Pattern (VCP) breakout with intraday
    opening-30-minute volume confirmation, swing-held with a risk-multiple
    target / moving-average trailing stop / max-hold-days exit.

    Pattern detection (approximation, disclosed): a formal VCP is a
    discretionary chart pattern, not a closed-form calculation. This
    implementation resamples `frame` (up through the last FULLY COMPLETED
    trading day before "today", never today's own partial day) into daily
    OHLCV bars, finds local swing highs/lows with a simple fractal method
    (a day is a swing high/low if it is the max/min High/Low within
    `fractal_k` days on either side -- both sides always lie strictly in the
    past relative to the current bar `i`, so this is never a look-ahead),
    then scans swing highs from most recent to oldest for a "pivot high"
    whose most recent `min_contractions` subsequent swing lows have
    strictly decreasing pullback percentages measured against that SAME
    pivot high (e.g. 15% then 8% below the pivot), operationalizing
    "successively shallower pullbacks converging toward a pivot high."
    Measured risk = pivot_high - most recent (shallowest) pullback low.
    Pattern is (re)computed at most once per calendar day per ticker (cached
    in the closure) using only that day's already-completed prior history,
    and the daily frame handed to the pivot search is truncated to the most
    recent `swing_lookback_days` calendar days before the search runs --
    without this bound the fractal scan (which walks from the most recent
    swing high backward and returns the first one that qualifies) could
    anchor on a pivot many months old, silently contradicting the
    candidate's own "multi-week volatility contraction pattern" mechanism
    and making swing_lookback_days a buffer-sizing knob only rather than a
    real lookback bound.

    Breakout / volume confirmation: entry is only evaluated while price is
    inside `pivot_confirmation_window_et` (default 09:30-10:00 ET). At each
    minute bar in that window, cumulative same-day volume (from the day's
    first RTH bar through the current bar -- genuine intraday minute-bar
    volume, never an end-of-day total) is compared against
    `min_opening_volume_pct_of_20d_avg`% of the 20-completed-trading-day
    average FULL-day volume. Entry fires (entry=True) at the first minute
    bar where cumulative volume already clears that threshold AND the
    bar's Close is above the day's pivot high. If neither the volume nor
    the breakout condition is satisfied by the end of the window, that
    trading day is skipped entirely (no forced/late entry outside the
    window) and the pattern is freshly recomputed the next day.

    Scope decision: LONG ENTRIES ONLY, per this pipeline's current
    single-direction backtest engine.

    Exit / hold management (mechanism said "target a multiple ... or trail
    a moving-average stop" without naming the multiple or the MA period, so
    both are exposed as disclosed, configurable knobs beyond the candidate's
    own `parameters` dict): target_price = pivot_high +
    `target_risk_multiple` * risk (computed off the pivot high, since the
    actual fill price is realized at the NEXT bar's open and is not known
    to the signal function at decision time) -- exits intrabar the first
    time a bar's High reaches it. Trailing stop is a daily-close-below-
    `trail_ma_days`-day-moving-average rule, evaluated once per day at the
    day's last RTH minute (bar time >= 15:59 ET, used as a proxy for "the
    daily close" since the signal function only ever sees data through the
    current bar; disclosed limitation: on a real early-close trading day
    this proxy never fires, so a trailing-stop/max-hold check due that day
    is deferred to the first bar of the following day instead). Max hold:
    the position is closed once `max_hold_days` TRADING days have elapsed
    since entry, counting the entry day as day 1 -- the SAME counting
    convention `intraday_backtest_engine.run_intraday_backtest`'s own
    independent max_hold_days force-close uses. This has to match exactly:
    the engine enforces its own max_hold_days cap without ever notifying
    signal_fn when it force-closes a position, so if this closure's own
    internal day-count used a different convention and could only fire
    STRICTLY LATER than the engine's cap, the engine would always force-
    close first and this closure's `state["in_position"]` would be left
    stuck `True` for a full extra trading day afterward -- silently
    blocking any new entry that day and evaluating stale exit conditions
    against an already-closed position's frozen target/pivot values.
    `state["days_held"]` is 0 on the entry day and increments by one on the
    first bar of every later day, i.e. it equals (engine's inclusive day
    count) minus 1 -- so the exit check below compares against
    `max_hold_days - 1`, not `max_hold_days`, to land on the identical
    calendar day the engine's own cap would trigger on. Whichever of the
    three exit conditions (target / trailing-stop / max-hold) is met first
    wins; a same-day tie with the engine's own force-close is harmless
    either way since both close the position on the same bar.

    Statefulness: position status, the day's cached pattern, and days-held
    are tracked as mutable closure state across sequential calls, since
    this is inherently a multi-day, multi-bar mechanism -- this assumes the
    caller invokes signal_fn once per bar in chronological order, which is
    how the described engine drives a SignalFn.
    """
    _start_str, _end_str = pivot_confirmation_window_et.split("-")
    _sh, _sm = (int(x) for x in _start_str.strip().split(":"))
    _eh, _em = (int(x) for x in _end_str.strip().split(":"))
    _WINDOW_START = dtime(_sh, _sm)
    _WINDOW_END = dtime(_eh, _em)
    _EOD_TIME = dtime(15, 59)

    state = {
        "in_position": False,
        "pivot_high": None,
        "risk": None,
        "target_price": None,
        "days_held": 0,
        "last_date": None,
        "pattern_cache_date": None,
        "pattern_cache_value": None,
        "avg20_cache_value": None,
    }

    def _to_daily(hist: pd.DataFrame) -> pd.DataFrame:
        d = hist.copy()
        d["__date"] = d.index.date
        daily = d.groupby("__date").agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        daily.sort_index(inplace=True)
        return daily

    def _recent_slice(frame: pd.DataFrame, current_date, lookback_days: int) -> pd.DataFrame:
        max_rows = (lookback_days + 60) * 450
        tail = frame.iloc[-max_rows:] if len(frame) > max_rows else frame
        return tail[tail.index.date < current_date]

    def _find_pattern(daily: pd.DataFrame, min_contractions: int, fractal_k: int):
        n = len(daily)
        min_needed = 2 * fractal_k + min_contractions + 3
        if n < min_needed:
            return None
        highs = daily["High"].to_numpy()
        lows = daily["Low"].to_numpy()
        swings = []
        for pos in range(fractal_k, n - fractal_k):
            wh = highs[pos - fractal_k: pos + fractal_k + 1]
            if highs[pos] >= wh.max():
                swings.append((pos, highs[pos], "H"))
            wl = lows[pos - fractal_k: pos + fractal_k + 1]
            if lows[pos] <= wl.min():
                swings.append((pos, lows[pos], "L"))
        if not swings:
            return None
        swings.sort(key=lambda t: t[0])
        cleaned = []
        for sp in swings:
            if cleaned and cleaned[-1][2] == sp[2]:
                if sp[2] == "H" and sp[1] > cleaned[-1][1]:
                    cleaned[-1] = sp
                elif sp[2] == "L" and sp[1] < cleaned[-1][1]:
                    cleaned[-1] = sp
            else:
                cleaned.append(sp)
        h_positions = [idx for idx, sp in enumerate(cleaned) if sp[2] == "H"]
        for hi in reversed(h_positions):
            pivot_price = cleaned[hi][1]
            lows_after = [sp for sp in cleaned[hi + 1:] if sp[2] == "L"]
            if len(lows_after) < min_contractions:
                continue
            tail = lows_after[-min_contractions:]
            pcts = [(pivot_price - lp[1]) / pivot_price for lp in tail]
            if any(p <= 0 for p in pcts):
                continue
            decreasing = all(pcts[j] > pcts[j + 1] for j in range(len(pcts) - 1))
            if decreasing:
                last_low_price = tail[-1][1]
                risk = pivot_price - last_low_price
                if risk > 0:
                    return {"pivot_high": pivot_price, "risk": risk}
        return None

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if frame.empty:
            return SignalResult()
        row = frame.iloc[-1]
        ts = frame.index[-1]
        today = ts.date()
        t_now = ts.time()

        if state["in_position"]:
            if state["last_date"] != today:
                if state["last_date"] is not None:
                    state["days_held"] += 1
                state["last_date"] = today

            exited = False
            if state["target_price"] is not None and row["High"] >= state["target_price"]:
                exited = True
            elif t_now >= _EOD_TIME:
                # Bounded to a recent tail before the O(len(frame)) `.date`
                # conversion -- only trail_ma_days complete days (plus
                # today) are ever needed for the trailing MA, never the
                # full multi-year frame. This call site runs once per day
                # while a position is open; converting the whole frame on
                # every such call would be quadratic across a long
                # backtest.
                hist = frame.tail((trail_ma_days + 10) * 450)
                hist = hist[hist.index.date <= today]
                daily = _to_daily(hist)
                if len(daily) >= trail_ma_days:
                    ma = daily["Close"].tail(trail_ma_days).mean()
                    if daily["Close"].iloc[-1] < ma:
                        exited = True
                # `- 1`: see the factory docstring's "Max hold" passage --
                # this closure's own days_held is 0 on the entry day, while
                # the engine's own independent max_hold_days force-close
                # counts the entry day as day 1, so this has to fire on
                # `max_hold_days - 1`, not `max_hold_days`, to land on the
                # SAME calendar day the engine's cap would trigger on.
                if not exited and state["days_held"] >= max_hold_days - 1:
                    exited = True

            if exited:
                state["in_position"] = False
                state["pivot_high"] = None
                state["risk"] = None
                state["target_price"] = None
                state["days_held"] = 0
                state["last_date"] = None
                return SignalResult(exit=True)
            return SignalResult()

        if not (_WINDOW_START <= t_now <= _WINDOW_END):
            return SignalResult()

        if state["pattern_cache_date"] != today:
            hist = _recent_slice(frame, today, swing_lookback_days)
            pattern = None
            avg20 = None
            if not hist.empty:
                daily = _to_daily(hist)
                cutoff_date = today - timedelta(days=swing_lookback_days)
                daily = daily[daily.index >= cutoff_date]
                pattern = _find_pattern(daily, min_contractions, fractal_k)
                if len(daily) >= 20:
                    avg20 = daily["Volume"].tail(20).mean()
            state["pattern_cache_date"] = today
            state["pattern_cache_value"] = pattern
            state["avg20_cache_value"] = avg20

        pattern = state["pattern_cache_value"]
        avg20 = state["avg20_cache_value"]
        if pattern is None or avg20 is None or pd.isna(avg20) or avg20 <= 0:
            return SignalResult()

        # Bounded to a recent tail before the O(len(frame)) `.date`
        # conversion -- a single RTH session is at most ~390 one-minute
        # bars, so 1000 comfortably contains all of today's bars.
        # This call site runs on every bar inside the opening
        # confirmation window (up to ~30x/day), so converting the whole
        # frame here would be quadratic across a long backtest.
        today_rows = frame.tail(1000)
        today_rows = today_rows[today_rows.index.date == today]
        if today_rows.empty:
            return SignalResult()
        cum_vol_today = today_rows["Volume"].sum()
        vol_threshold = (min_opening_volume_pct_of_20d_avg / 100.0) * avg20

        if cum_vol_today >= vol_threshold and row["Close"] > pattern["pivot_high"]:
            state["in_position"] = True
            state["pivot_high"] = pattern["pivot_high"]
            state["risk"] = pattern["risk"]
            state["target_price"] = pattern["pivot_high"] + target_risk_multiple * pattern["risk"]
            state["days_held"] = 0
            state["last_date"] = today
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- gap_and_go_momentum_continuation ---
def make_gap_and_go_momentum_continuation_signal_fn(
    gap_threshold_pct: float = 2.0,
    volume_multiple_vs_20d_avg: float = 2.0,
    trailing_stop_atr_mult: float = 3.0,
    atr_period_days: int = 14,
    max_hold_days: int = 20,
    volume_lookback_days: int = 20,
) -> SignalFn:
    """
    Long-only Gap-and-Go Momentum Continuation.

    Designed for `mining.intraday_backtest_engine.run_intraday_backtest`
    (minute-bar OHLCV, RTH-only, tz-aware "America/New_York" index; entry
    at the NEXT bar's open, exit at the SAME bar's close) -- NOT the daily
    engine.

    At the day's 09:30 ET opening print, computes the overnight gap =
    (day_open - prior_completed_session_close) / prior_completed_session_close.
    If gap_pct >= gap_threshold_pct AND the cumulative Volume over the 09:30-09:44
    ET bars (the "first 15 minutes", 09:30 through 09:45 ET) is >=
    volume_multiple_vs_20d_avg times the trailing volume_lookback_days-day average
    of that same first-15-minute cumulative volume AND price never traded back
    down through the exact opening print during 09:31-09:44 (see disclosed
    approximation #1 below), an entry=True is fired on the 09:44 bar so the
    engine's next-open execution fills at the 09:45 ET open, matching the
    mechanism's stated entry time. Position is then held with a 3x-ATR(14-day)
    trailing stop or a max_hold_days trading-day cap, whichever comes first.

    SCOPE (disclosed): only the long side is implemented.
    The mechanism's mirrored short variant (gap <= -2%) is intentionally not
    implemented; the backtest engine's `direction` is a single whole-call
    parameter and this pipeline scopes every swing candidate to its long side.

    Disclosed approximations / interpretive choices:
    1. "Price never trades back through the exact opening print" is tested on
       the confirmation-window bars AFTER the opening bar itself (09:31-09:44),
       not including the 09:30 bar's own Low -- a bar's own Low is trivially
       <= its own Open, so including it would make the condition impossible to
       satisfy on any day. The intent (price holding above the gap after the
       open) is captured by testing the following 14 minutes instead.
    2. ATR(14) uses a simple (non-Wilder-smoothed) rolling mean of True Range
       computed on the frame resampled to one row per COMPLETED trading day
       (today's still-in-progress partial day is always excluded), refreshed
       once per calendar day and reused for every intraday bar that day.
    3. The trailing stop is monitored continuously intrabar: stop_price =
       (highest intraday High observed since the fill bar) - trailing_stop_atr_mult
       * ATR14, and triggers the first bar whose Low trades at or below it. The
       mechanism does not specify EOD-only vs. intraday monitoring for a
       "trailing stop"; continuous intrabar monitoring is the more faithful
       reading of how a real trailing-stop order behaves. Under this engine's
       calling convention, signal_fn IS called with the fill bar itself as
       `frame.iloc[-1]` (entry happens via next-bar-open execution, and the
       engine falls through to evaluate the freshly-opened position on that
       same bar) -- this implementation uses that call only to seed the peak
       from the fill bar's own Open/High and does not stop-check the fill bar
       itself, since the fill already happened at that bar's own Open and the
       same bar's Low cannot be disentangled into before/after the fill
       without tick data. Stop-checking then begins on the very next call
       (the bar after the fill bar), the first bar signal_fn sees under
       "in_position" state.
    4. max_hold_days is counted in trading days from (and including) the fill
       day: the fill day is day 1, and the position is force-exited once the
       count reaches max_hold_days. (The engine also independently enforces
       max_hold_days as an end-of-day time_stop; this internal count can
       therefore fire slightly earlier, intraday on day N, rather than only
       at day N's last bar -- a more conservative reading of "maximum hold N
       trading days", and harmless since the engine's own cap is a backstop,
       not a conflicting rule.)
    5. Purely for computational efficiency (does not change which historical
       days feed any calculation): the daily-resample lookback used for
       ATR/prior-close/20-day-volume is bounded to a generous fixed number of
       the most recent rows rather than re-scanning the entire, potentially
       multi-year frame on every call -- since ATR(14) and the
       volume_lookback_days average only ever need that many trailing trading
       days regardless of how much history exists before them. Safe under
       RTH-only minute data (no pre-market bars to misalign day boundaries).
    6. confirmation_window_et ("09:30-09:45") and entry_time_et ("09:45") from
       the candidate's parameters are reflected as fixed time constants baked
       into the control flow (they define WHEN evaluation/entry happens, not a
       numeric threshold to compare against), rather than exposed as separate
       factory arguments.

    State: this closure tracks its own single-position state (flat /
    pending-fill / in-position) internally, since SignalFn is not given the
    engine's own position state -- only one position is tracked at a time,
    consistent with a non-pyramiding entry rule.
    """
    CONF_START = dtime(9, 30)
    CONF_END = dtime(9, 44)
    WINDOW_BARS = 15  # 09:30 .. 09:44 inclusive, one bar per minute
    APPROX_BARS_PER_DAY = 400
    LOOKBACK_DAY_BUFFER = 40
    LOOKBACK_ROWS = (
        max(atr_period_days, volume_lookback_days) + LOOKBACK_DAY_BUFFER
    ) * APPROX_BARS_PER_DAY

    state = {
        "status": "flat",  # "flat" | "pending_entry" | "in_position"
        "highest_price_since_entry": None,
        "atr_cache_date": None,
        "atr_cache_value": None,
        "days_held": 0,
        "last_bar_date": None,
    }

    def _daily_ohlc(sub: pd.DataFrame) -> pd.DataFrame:
        if sub.empty:
            return sub
        g = sub.groupby(sub.index.date)
        return pd.DataFrame(
            {
                "Open": g["Open"].first(),
                "High": g["High"].max(),
                "Low": g["Low"].min(),
                "Close": g["Close"].last(),
                "Volume": g["Volume"].sum(),
            }
        )

    def _atr(daily: pd.DataFrame):
        if daily is None or len(daily) < atr_period_days + 1:
            return None
        prev_close = daily["Close"].shift(1)
        tr = pd.concat(
            [
                daily["High"] - daily["Low"],
                (daily["High"] - prev_close).abs(),
                (daily["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1).iloc[1:]
        atr_series = tr.rolling(atr_period_days).mean()
        if atr_series.empty:
            return None
        val = atr_series.iloc[-1]
        return None if pd.isna(val) else float(val)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        try:
            ts = frame.index[-1]
            current_date = ts.date()
            bar_time = ts.time()
            row = frame.iloc[-1]

            # ---------------- EXIT PATH ----------------
            if state["status"] == "in_position":
                high = float(row["High"])
                low = float(row["Low"])
                if state["highest_price_since_entry"] is None:
                    state["highest_price_since_entry"] = high
                else:
                    state["highest_price_since_entry"] = max(
                        state["highest_price_since_entry"], high
                    )

                if state["last_bar_date"] != current_date:
                    state["days_held"] += 1
                    state["last_bar_date"] = current_date

                if state["atr_cache_date"] != current_date:
                    hist_start = max(0, i - LOOKBACK_ROWS)
                    prior_slice = frame.iloc[hist_start:i]
                    prior_slice = prior_slice[prior_slice.index.date < current_date]
                    daily_prior = _daily_ohlc(prior_slice)
                    state["atr_cache_value"] = _atr(daily_prior)
                    state["atr_cache_date"] = current_date

                exit_hit = False
                atr_val = state["atr_cache_value"]
                if atr_val is not None:
                    stop_price = (
                        state["highest_price_since_entry"]
                        - trailing_stop_atr_mult * atr_val
                    )
                    if low <= stop_price:
                        exit_hit = True

                if not exit_hit and state["days_held"] >= max_hold_days:
                    exit_hit = True

                if exit_hit:
                    state["status"] = "flat"
                    state["highest_price_since_entry"] = None
                    state["atr_cache_date"] = None
                    state["atr_cache_value"] = None
                    state["days_held"] = 0
                    state["last_bar_date"] = None
                    return SignalResult(exit=True)
                return SignalResult()

            # ---------------- FILL BAR (bar right after an entry signal;
            # under this engine, the fill bar IS this call's own frame.iloc[-1]
            # -- see disclosed approximation #3) ----------------
            if state["status"] == "pending_entry":
                state["status"] = "in_position"
                state["highest_price_since_entry"] = max(
                    float(row["Open"]), float(row["High"])
                )
                state["atr_cache_date"] = None
                state["atr_cache_value"] = None
                state["days_held"] = 1
                state["last_bar_date"] = current_date
                return SignalResult()

            # ---------------- FLAT: only evaluate entry at the 09:44 bar ----------------
            if bar_time != CONF_END:
                return SignalResult()
            if i < WINDOW_BARS - 1:
                return SignalResult()

            window_bars = frame.iloc[i - (WINDOW_BARS - 1) : i + 1]
            if len(window_bars) != WINDOW_BARS:
                return SignalResult()
            if window_bars.index[0].time() != CONF_START:
                return SignalResult()
            if not (window_bars.index.date == current_date).all():
                return SignalResult()

            opening_price = float(window_bars["Open"].iloc[0])
            if opening_price <= 0:
                return SignalResult()

            hist_start = max(0, (i - (WINDOW_BARS - 1)) - LOOKBACK_ROWS)
            hist_end = i - (WINDOW_BARS - 1)
            prior_slice = frame.iloc[hist_start:hist_end]
            if prior_slice.empty:
                return SignalResult()

            daily_prior = _daily_ohlc(prior_slice)
            if daily_prior.empty:
                return SignalResult()
            prior_close = float(daily_prior["Close"].iloc[-1])
            if prior_close <= 0:
                return SignalResult()

            gap_pct = (opening_price - prior_close) / prior_close * 100.0
            if gap_pct < gap_threshold_pct:
                return SignalResult()

            times = prior_slice.index.time
            vol_mask = (times >= CONF_START) & (times <= CONF_END)
            prior_window = prior_slice[vol_mask]
            if prior_window.empty:
                return SignalResult()
            first15_by_day = prior_window.groupby(prior_window.index.date)["Volume"].sum()
            if len(first15_by_day) < volume_lookback_days:
                return SignalResult()
            trailing_avg_vol = float(first15_by_day.iloc[-volume_lookback_days:].mean())
            if trailing_avg_vol <= 0:
                return SignalResult()

            today_window_vol = float(window_bars["Volume"].sum())
            if today_window_vol < volume_multiple_vs_20d_avg * trailing_avg_vol:
                return SignalResult()

            confirmation_bars = window_bars.iloc[1:]  # exclude the 09:30 bar itself
            if (confirmation_bars["Low"] <= opening_price).any():
                return SignalResult()

            state["status"] = "pending_entry"
            return SignalResult(entry=True)

        except (IndexError, KeyError, ValueError):
            return SignalResult()

    return signal_fn


# --- earnings_gap_continuation_intraday_confirm ---
def make_earnings_gap_continuation_intraday_confirm_signal_fn(
    earnings_events: list[dict],
    gap_threshold_pct: float = 3.0,
    confirmation_window_et: str = "09:30-10:00",
    confirmation_tolerance_pct: float = 1.0,
    entry_time_et: str = "10:00",
    max_hold_days: int = 20,
) -> SignalFn:
    """
    Earnings Gap Continuation with Intraday Confirmation (long side only).

    On the first regular session that reacts to a real historical earnings release
    (`earnings_events`), this measures the opening gap at that session's own open vs.
    the prior regular session's own close. It only ever arms the LONG case: a positive
    reported earnings surprise (surprise_pct > 0) combined with a real opening gap UP
    of at least `gap_threshold_pct`. DISCLOSED SCOPE (per the engine's single whole-call
    `direction` parameter): the mechanism's mirrored short case -- negative surprise +
    gap down -- is deliberately out of scope for this factory, not silently dropped.

    Reaction-day resolution uses only each event's own real historical `hour` (ET)
    field, and never a later event in the list: hour < 9 (pre-market) -> the event's
    own calendar date is the reaction day; hour >= 9 (including the after-close
    hour >= 16 case) -> the first real trading day the frame observes strictly after
    the event's date is the reaction day. DISCLOSED APPROXIMATION: a report released
    9 <= hour < 16 (during the regular session itself, e.g. an hour=12
    print) is folded into the same "next trading day" bucket as an after-close report,
    since a same-day mid-session release cannot itself produce a fresh pre-open gap --
    the first tradable gap opportunity is the following session's open.

    On the resolved reaction day: opening_print = that day's first bar Open;
    prior_close = the prior trading day's last observed Close, both built only from
    bars this closure has actually already seen (never a lookahead, and a day-1-of-
    frame event with no real prior close yet is skipped via SignalResult(), never
    faked). gap_pct = (opening_print - prior_close) / prior_close * 100; if
    gap_pct < gap_threshold_pct, or surprise_pct <= 0, the event is never armed.
    DISCLOSED HARDENING: opening_print and prior_close are also required to be real
    finite positive prices before gap_pct is computed at all -- a malformed or missing
    price (NaN, zero, negative) skips arming that event entirely rather than raising
    (a raw ZeroDivisionError on a zero prior_close) or silently failing to reject the
    event (an unguarded NaN gap_pct compares False against the threshold either way in
    Python, which would have let a garbage-data day slip through as "armed").
    Otherwise, every bar inside the parsed `confirmation_window_et` window (default
    09:30-10:00) is checked against a band of +/- confirmation_tolerance_pct around
    opening_print (DISCLOSED: interpreted literally/symmetrically per "hold within 1%
    of the opening print", not one-sided) and against a fade back through prior_close
    (checked against the bar's own intraday Low, consistent with how the tolerance
    band above is also checked against intraday Low/High rather than only the bar's
    Close -- "no immediate intraday fade back through" is naturally an intrabar
    condition, not a close-only one); either violation cancels that event's trade. If
    confirmation survives to the parsed `entry_time_et` (default 10:00), entry=True
    fires on that bar (the engine's own next-open execution then opens the position at
    the following bar's open). While in the resulting position, exit=True fires the
    moment a bar's Close falls back below the entry day's opening_print (the "full
    retracement below the opening print's level" rule), or -- DISCLOSED APPROXIMATION
    for the time-based exit -- at the first bar of the trading day immediately
    following the `max_hold_days`-th trading day held (entry day counts as held day
    1), rather than modeling an exact end-of-session close on day 20 itself. This
    closure-level cap is intentionally redundant with (never looser than) the engine's
    own `max_hold_days` parameter on `run_intraday_backtest` -- the engine's tighter
    same-day force-close at the LAST bar of trading day N always fires no later than
    this closure's own day-(N+1) check, so this is a harmless belt-and-suspenders
    safety net, not a source of divergent behavior.

    Only one position / one armed candidate is tracked at a time: an event whose
    resolved reaction day arrives while already in a position from an earlier event is
    skipped (missed), never queued or retro-armed on a later day. Returns a no-op
    SignalResult() on any bar where there is not yet enough real history to evaluate
    the condition -- this never raises.
    """

    import datetime

    def _parse_time(hhmm: str) -> datetime.time:
        return datetime.datetime.strptime(hhmm.strip(), "%H:%M").time()

    _window_parts = confirmation_window_et.split("-")
    window_start = _parse_time(_window_parts[0])
    window_end = _parse_time(_window_parts[1])
    entry_time = _parse_time(entry_time_et)

    events_sorted = sorted(
        (
            {
                "date": datetime.date.fromisoformat(ev["date"]),
                "hour": int(ev["hour"]),
                "surprise_pct": float(ev["surprise_pct"]),
            }
            for ev in (earnings_events or [])
        ),
        key=lambda e: e["date"],
    )

    state = {
        "last_date": None,
        "running_last_close": None,
        "prev_day_last_close": None,
        "event_ptr": 0,
        "armed": None,
        "in_position": False,
        "position_opening_print": None,
        "position_days_held": 0,
    }

    def signal_fn(frame, i: int) -> "SignalResult":
        row = frame.iloc[-1]
        ts = frame.index[-1]
        today = ts.date()
        t = ts.time()
        close = float(row["Close"])
        open_ = float(row["Open"])
        low = float(row["Low"])
        high = float(row["High"])

        is_new_day = state["last_date"] != today

        if is_new_day:
            if state["last_date"] is not None:
                state["prev_day_last_close"] = state["running_last_close"]
            if state["in_position"]:
                state["position_days_held"] += 1
            if state["armed"] is not None and state["armed"]["day"] != today:
                state["armed"] = None
            state["last_date"] = today

            while state["event_ptr"] < len(events_sorted):
                ev = events_sorted[state["event_ptr"]]
                ev_date = ev["date"]
                if ev["hour"] < 9:
                    trigger = today >= ev_date
                    exact = today == ev_date
                else:
                    trigger = today > ev_date
                    exact = trigger
                if not trigger:
                    break
                if (
                    exact
                    and state["armed"] is None
                    and not state["in_position"]
                    and ev["surprise_pct"] > 0
                    and state["prev_day_last_close"] is not None
                ):
                    prior_close = state["prev_day_last_close"]
                    # Guard against malformed/missing prices (NaN, zero,
                    # negative) before dividing -- a zero prior_close would
                    # otherwise raise ZeroDivisionError (breaking the "this
                    # never raises" contract), and a NaN prior_close/open_
                    # would produce a NaN gap_pct that silently compares
                    # False against gap_threshold_pct, i.e. `rejected` would
                    # wrongly come out False instead of the event just never
                    # arming.
                    if prior_close > 0 and open_ > 0:
                        gap_pct = (open_ - prior_close) / prior_close * 100.0
                        rejected = gap_pct < gap_threshold_pct
                        state["armed"] = {
                            "day": today,
                            "opening_print": open_,
                            "prior_close": prior_close,
                            "gap_pct": gap_pct,
                            "rejected": rejected,
                            "entered": False,
                        }
                state["event_ptr"] += 1

        state["running_last_close"] = close

        if state["in_position"]:
            exit_fired = False
            if close < state["position_opening_print"]:
                exit_fired = True
            elif state["position_days_held"] > max_hold_days:
                exit_fired = True
            if exit_fired:
                state["in_position"] = False
                state["position_opening_print"] = None
                state["position_days_held"] = 0
            return SignalResult(entry=False, exit=exit_fired)

        armed = state["armed"]
        if armed is None or armed["day"] != today or armed["rejected"] or armed["entered"]:
            return SignalResult()

        if window_start <= t < window_end:
            tol = confirmation_tolerance_pct / 100.0
            opening_print = armed["opening_print"]
            upper_bound = opening_print * (1 + tol)
            lower_bound = opening_print * (1 - tol)
            if low < lower_bound or high > upper_bound or low < armed["prior_close"]:
                armed["rejected"] = True
            return SignalResult()

        if t >= entry_time:
            armed["entered"] = True
            state["in_position"] = True
            state["position_opening_print"] = armed["opening_print"]
            state["position_days_held"] = 1
            return SignalResult(entry=True, exit=False)

        return SignalResult()

    return signal_fn


# --- avwap_swing_pivot_reclaim_breakout ---
def make_avwap_swing_pivot_reclaim_breakout_signal_fn(
    anchor_pivot_lookback_days: int = 20,
    volume_confirmation_multiple: float = 1.5,
    max_hold_days: int = 20,
) -> SignalFn:
    """Anchored VWAP Swing-Pivot Reclaim Breakout (LONG side only, see scope note below).

    Once per trading day -- detected at the bar whose America/New_York wall-clock time is
    15:59 (the last regular-hours minute bar under Alpaca's bar-start timestamp convention,
    covering 15:59:00-16:00:00) -- this closure finalizes that day's daily OHLCV from its own
    real minute bars, then: (1) while FLAT, checks whether today is a new rolling
    `anchor_pivot_lookback_days`-day LOW (today's Low is the minimum daily Low over the
    trailing window that includes today); if so it (re)anchors a volume-weighted average
    price line to today's session, accumulating cumulative Volume-weighted price from that
    session's own real minute bars forward (per-bar Alpaca VWAP where present, falling back
    to (High+Low+Close)/3 typical price when VWAP is NaN/missing). While a position is OPEN,
    a fresh pivot low is still tracked in the daily-low history but does NOT re-anchor -- the
    open trade's own AVWAP line stays locked to the session it was entered against and simply
    keeps accumulating forward (see the re-anchoring note below). (2) While flat and an anchor is
    active, checks the reclaim entry: today's daily Close above the running anchored VWAP
    line AND today's real intraday volume (summed from minute bars) at least
    `volume_confirmation_multiple`x the average daily volume of the `anchor_pivot_lookback_days`
    trading days strictly BEFORE today (today's own volume excluded from its own baseline).
    If both hold and this anchor has not already produced a reclaim, it fires entry=True on
    today's last bar so the engine's next-bar-open execution lands at the following trading
    day's open. (3) While in a (closure-tracked) position, it counts trading days held and
    counts CONSECUTIVE daily closes back below the same running anchored VWAP line, firing
    exit=True (same-bar-close execution, per the engine contract) on the second such
    consecutive close-through, or once `max_hold_days` trading days have been held, whichever
    comes first.

    Disclosed scope decision: only the LONG reclaim side is implemented. The mechanism
    describes a mirrored short (AVWAP-loss breakdown from a swing-high anchor), but the
    backtest engine's `direction` is a single whole-call parameter, so this pipeline
    deliberately scopes every swing candidate to its long side only.

    Re-anchoring note: re-anchoring is gated on being FLAT at the start of the bar's
    processing (`was_in_position`). Since a 20-trading-day hold and a 20-trading-day pivot
    lookback are the same length, an unrelated new low can form mid-hold; re-anchoring
    on it would silently reset `cum_pv`/`cum_vol` to just that day's own bars and swap
    the stop-rule's reference line ("second daily close back through the AVWAP line") out from
    under the open trade for a line that has nothing to do with the entry. While
    in a position, the code still updates the trailing daily-low history (so pivot detection
    is caught up once the position closes) but always keeps accumulating onto the SAME anchor
    the trade was entered against.

    Disclosed approximations:
    - "Most recent significant swing pivot ... rolling 20-trading-day high or low" is
      evaluated once per completed trading day (never intraday), using that day's own
      finalized daily Low compared against the trailing `anchor_pivot_lookback_days`-day
      window of daily Lows (window includes today), and only actually triggers a re-anchor
      while flat (see the re-anchoring note above).
    - Day-close detection relies on a fixed 15:59 ET bar-start heuristic for "last regular
      session bar." Genuine early-close (half) sessions end before that time and are simply
      never evaluated by this signal (no entry/exit decision fires on those calendar days) --
      a small, disclosed reduction in sample near market holidays, not a correctness bug.
    - The anchored VWAP is computed by weighting each real minute bar by Alpaca's own per-bar
      VWAP field (falling back to (High+Low+Close)/3 typical price when VWAP is missing/NaN),
      since no running/session VWAP is provided directly.
    - The 20-day average daily volume used for the 1.5x confirmation excludes today's own
      volume from its own baseline (uses the `anchor_pivot_lookback_days` trading days
      strictly before today) -- a standard convention to avoid a day's own volume inflating
      the bar it is being measured against.
    - "Second daily close back through the AVWAP line" (the stop rule) is interpreted as two
      CONSECUTIVE daily closes below the running anchored VWAP; the running count resets to
      zero on any intervening daily close back above the line. This is a disclosed reading of
      an otherwise ambiguous stop-rule description.
    - A reclaim that fires on the SAME session a fresh pivot low anchors (close back above
      that session's own just-started AVWAP, on elevated volume) is accepted as a valid "first
      reclaim" rather than specially excluded -- a capitulation-reversal read of "reclaim" that
      the mechanism text does not rule out, disclosed here since it is the one genuinely
      debatable edge of the reclaim condition.
    - This closure keeps mutable state (daily OHLCV history, the active AVWAP anchor and its
      cumulative price*volume/volume accumulators, and open-position day/through counters)
      across calls. It assumes signal_fn is invoked once per bar, in strictly ascending
      chronological order, for a single ticker's own bar series -- the standard sequential
      backtest calling convention -- and must not be shared across two independent
      chronological streams.
    """

    state = {
        "daily_dates": [],
        "daily_low": [],
        "daily_volume": [],
        "anchor_date": None,
        "cum_pv": 0.0,
        "cum_vol": 0.0,
        "reclaimed_since_anchor": False,
        "in_position": False,
        "days_held": 0,
        "through_count": 0,
        "last_processed_date": None,
    }

    def _weighted_price(day_bars: pd.DataFrame) -> pd.Series:
        typical = (day_bars["High"] + day_bars["Low"] + day_bars["Close"]) / 3.0
        if "VWAP" in day_bars.columns:
            vwap = pd.to_numeric(day_bars["VWAP"], errors="coerce")
            return vwap.fillna(typical)
        return typical

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if frame is None or len(frame) == 0:
            return SignalResult()

        last_ts = frame.index[-1]

        # Only act once per day, at the last regular-hours minute bar (bar-start convention:
        # a full RTH session's final 1-minute bar starts at 15:59 ET). Early-close sessions
        # never reach this cutoff and are simply never evaluated (disclosed above).
        if last_ts.hour != 15 or last_ts.minute != 59:
            return SignalResult()

        today = last_ts.date()
        if state["last_processed_date"] == today:
            return SignalResult()
        state["last_processed_date"] = today

        # Snapshot position status BEFORE today's processing touches it -- this is what
        # gates re-anchoring below (see the re-anchoring note in the docstring): re-anchoring must
        # only happen while flat, never mid-hold.
        was_in_position = state["in_position"]

        # Only scan the recent tail for today's bars (a single trading day never exceeds
        # ~390 minute bars) instead of re-scanning the full, potentially years-long frame.
        tail_n = min(len(frame), 500)
        tail = frame.iloc[-tail_n:]
        day_mask = tail.index.date == today
        day_bars = tail.loc[day_mask]
        if day_bars.empty:
            return SignalResult()

        day_low = float(day_bars["Low"].min())
        day_close = float(day_bars["Close"].iloc[-1])
        day_volume = float(day_bars["Volume"].sum())

        state["daily_dates"].append(today)
        state["daily_low"].append(day_low)
        state["daily_volume"].append(day_volume)

        n_days = len(state["daily_dates"])
        lookback = anchor_pivot_lookback_days

        entry = False
        exit_flag = False

        # --- Rolling N-day swing-low pivot detection (today's own Low included) ---
        is_new_pivot_low = False
        if n_days >= lookback:
            window_low = min(state["daily_low"][-lookback:])
            if day_low <= window_low + 1e-9:
                is_new_pivot_low = True

        # --- Anchored VWAP accumulation from real minute bars ---
        # While a position is open, the anchor stays LOCKED to the exact line the trade was
        # entered against: re-anchoring to a fresh pivot low here would silently swap the
        # stop-rule's reference line out from under an open trade (a new rolling-window low
        # can easily form mid-hold, since the pivot lookback and the max hold are both
        # `anchor_pivot_lookback_days`-scale). Only re-anchor on a new pivot low while FLAT;
        # while in a position, keep accumulating onto the SAME anchor regardless of whether
        # today also happens to make a fresh pivot low (that fresh low is still recorded in
        # `daily_low` above, so pivot detection is caught up correctly once flat again).
        if not was_in_position and is_new_pivot_low:
            wp = _weighted_price(day_bars)
            state["anchor_date"] = today
            state["cum_pv"] = float((wp * day_bars["Volume"]).sum())
            state["cum_vol"] = float(day_bars["Volume"].sum())
            state["reclaimed_since_anchor"] = False
        elif state["anchor_date"] is not None:
            wp = _weighted_price(day_bars)
            state["cum_pv"] += float((wp * day_bars["Volume"]).sum())
            state["cum_vol"] += float(day_bars["Volume"].sum())

        avwap_today = None
        if state["anchor_date"] is not None and state["cum_vol"] > 0:
            avwap_today = state["cum_pv"] / state["cum_vol"]

        # --- Entry: first reclaim of the anchored AVWAP on elevated volume ---
        if not state["in_position"] and avwap_today is not None and not state["reclaimed_since_anchor"]:
            reclaim = day_close > avwap_today
            vol_avg = None
            if n_days > lookback:
                prior_vols = state["daily_volume"][-(lookback + 1):-1]
                if len(prior_vols) >= lookback:
                    vol_avg = sum(prior_vols) / len(prior_vols)
            vol_confirm = vol_avg is not None and vol_avg > 0 and day_volume >= volume_confirmation_multiple * vol_avg
            if reclaim and vol_confirm:
                entry = True
                state["reclaimed_since_anchor"] = True
                state["in_position"] = True
                state["days_held"] = 0
                state["through_count"] = 0
        elif state["in_position"]:
            state["days_held"] += 1
            if avwap_today is not None and day_close < avwap_today:
                state["through_count"] += 1
            else:
                state["through_count"] = 0

            if state["through_count"] >= 2:
                exit_flag = True
                state["in_position"] = False
                state["days_held"] = 0
                state["through_count"] = 0
            elif state["days_held"] >= max_hold_days:
                exit_flag = True
                state["in_position"] = False
                state["days_held"] = 0
                state["through_count"] = 0

        return SignalResult(entry=entry, exit=exit_flag)

    return signal_fn


# --- earnings_avwap_reaction_reclaim ---
def make_earnings_avwap_reaction_reclaim_signal_fn(
    earnings_events,
    reclaim_window_days=10,
    volume_confirmation_multiple=1.3,
    max_hold_days=15,
    avg_volume_lookback_days=20,
):
    """Long-only earnings-reaction anchored-VWAP reclaim.

    Anchors a volume-weighted average price ("AVWAP") to the first full
    trading session after a real historical earnings-reaction event for
    this ticker (the "reaction day"), using this ticker's own
    `earnings_events` list (each item: date, hour ET the report was
    released, surprise_pct). An event's reaction day is resolved as: if
    hour < 9 the report was pre-market and the reaction trades the SAME
    calendar day; otherwise (hour >= 9, which covers both the after-close
    hour >= 16 case the mechanism names explicitly and the rarer
    during-session case it does not) the reaction is treated as trading
    the NEXT real trading day -- disclosed approximation for that
    unspecified middle range. The reaction day is only ever resolved using
    real historical trading days already present in `frame` up to bar
    `i`, so this never looks ahead.

    AVWAP is computed from real minute bars only: cumulative
    sum(VWAP * Volume) / sum(Volume) over every RTH minute bar from the
    reaction day's own opening bar through the bar being evaluated (Alpaca's
    per-bar VWAP is used as each bar's volume-weighted "price", which is a
    closer estimate of the bar's true trade-volume-weighted price than
    Close alone).

    Because the mechanism is stated in terms of DAILY closes, this signal
    only ever makes an entry/exit decision using a session's OWN complete
    daily close/volume -- never a running/partial value from a session
    still in progress. A session is only ever treated as complete once
    EITHER (a) a bar timestamped 15:59 ET or later arrives for it (the
    last RTH minute per Alpaca's standard convention on a normal trading
    day), OR (b) the first bar of the NEXT calendar day arrives without
    (a) ever having happened for the prior day -- which proves, without
    reading anything about the future, that no further bars for that
    prior day are coming (`frame` is time-ascending). Path (b) is what
    correctly handles a scheduled NYSE early close (e.g. the day after
    Thanksgiving, last bar ~13:00) or a minute-bar data gap: the decision
    for that day still fires, using its own real last bar as the close and
    its own real full-session volume, just evaluated on the following
    day's first tick instead of same-day. (Checking ONLY for a >=15:59
    bar is not enough: on a half day that condition is never true, so
    every bar of that session would silently return a flat
    SignalResult() AND the session's volume would never be subtracted
    out of the running position tracker, corrupting the FOLLOWING day's
    own volume figure and, if the reaction day itself fell on a half
    day, permanently dropping that earnings event from consideration.
    Path (b) closes both holes without hardcoding any market-holiday
    calendar.) Every bar that is not one of these two day-closing bars
    returns a flat SignalResult(); at most one decision bar exists per
    calendar day, and it is always evaluated from that day's own real,
    complete OHLCV -- never a mid-session running aggregate.

    Over the "following up to `reclaim_window_days` trading days" after
    the anchor day (day offsets 1..N relative to the reaction day -- the
    reaction day itself, offset 0, is the anchor session and is not
    itself eligible to satisfy the "closed below" prerequisite or the
    reclaim, a disclosed interpretive choice), entry fires long the first
    day the daily close is >= that day's AVWAP AFTER at least one earlier
    day in the same window closed below AVWAP, provided that reclaim
    day's total RTH volume is >= `volume_confirmation_multiple` times the
    trailing `avg_volume_lookback_days`-day average daily volume (the
    `20` named in the mechanism text; the trailing average excludes the
    reclaim day itself, the standard convention). Only one earnings-
    reaction window is tracked at a time: once in a position, no new
    reaction window is considered until the position exits (a disclosed
    simplification -- the mechanism does not describe overlapping
    trades). A same-calendar-day exit immediately followed by a fresh
    entry is not evaluated (the decision for a given day fires exactly
    once); given entries and exits are both keyed to distinct earnings-
    reaction windows roughly a fiscal quarter apart against a
    `reclaim_window_days`/`max_hold_days` scale of days, this is not
    expected to ever bind in practice for this mechanism.

    Exit fires on the first daily close back below the SAME anchored AVWAP
    line (continuing to accumulate from the same reaction-day anchor), or
    once `max_hold_days` trading days have elapsed since the entry
    SIGNAL day (not the next-open fill day -- a disclosed, at-most-one-day
    approximation), whichever comes first.

    Scope: LONG side only, per this pipeline's single-direction
    scoping (the mechanism itself describes only a long reclaim, so this
    is not a scope reduction from a stated short mirror).

    `earnings_events` must be the real historical earnings-reaction event
    list for the ONE ticker this signal_fn's `frame` belongs to (e.g.
    `EVENTS["AAPL"]` for an AAPL-only backtest instance) -- each item a
    dict with at least `date` (ISO "YYYY-MM-DD"), `hour` (int, ET), and
    `surprise_pct` (unused by this mechanism, present for completeness).
    """
    import datetime as _dt
    import pandas as pd

    events = sorted(
        (e for e in earnings_events if "date" in e),
        key=lambda e: e["date"],
    )

    state = {
        "last_seen_date": None,
        "day_start_len": 0,
        "day_dates": [],
        "date_to_idx": {},
        "daily_volume": [],
        "in_position": False,
        "entry_day_idx": None,
        "active_reaction_date": None,
    }

    def _resolve_next_trading_day(event_date, day_dates):
        for d in day_dates:
            if d > event_date:
                return d
        return None

    def _finalize_day(decision_date, decision_frame, start_len):
        """Record `decision_date`'s complete OHLCV into the running daily
        state. `decision_frame` must end exactly at `decision_date`'s own
        last bar (never a bar from any later date), and `start_len` must
        be the position, within `decision_frame`, where `decision_date`'s
        bars begin -- both guaranteed by the two call sites in
        `signal_fn` below, never by this helper itself."""
        day_volume = float(decision_frame["Volume"].iloc[start_len:].sum())
        state["day_dates"].append(decision_date)
        state["date_to_idx"][decision_date] = len(state["day_dates"]) - 1
        state["daily_volume"].append(day_volume)

    def _decide(decision_date, decision_frame):
        """Run the entry/exit decision for `decision_date`, which must
        already have been finalized via `_finalize_day` (so
        `state["date_to_idx"]` and `state["daily_volume"]` both cover it)
        before this is called. `decision_frame` must end exactly at
        `decision_date`'s own last bar."""
        idx_decision = state["date_to_idx"][decision_date]
        day_dates = state["day_dates"]
        tz = decision_frame.index.tz

        # ---- exit branch: already holding a reclaim-triggered long ----
        if state["in_position"]:
            R = state["active_reaction_date"]
            start_ts = pd.Timestamp.combine(R, _dt.time(0, 0)).tz_localize(tz)
            end_ts = pd.Timestamp.combine(decision_date, _dt.time(23, 59, 59)).tz_localize(tz)
            window = decision_frame.loc[start_ts:end_ts]
            vol_sum = window["Volume"].sum()
            if vol_sum <= 0:
                return SignalResult()
            avwap_today = float((window["VWAP"] * window["Volume"]).sum() / vol_sum)
            today_close = float(decision_frame["Close"].iloc[-1])

            hold_days = idx_decision - state["entry_day_idx"]
            if today_close < avwap_today or hold_days >= max_hold_days:
                state["in_position"] = False
                state["entry_day_idx"] = None
                state["active_reaction_date"] = None
                return SignalResult(exit=True)
            return SignalResult()

        # ---- entry branch: look for an active earnings-reaction window ----
        if idx_decision < avg_volume_lookback_days:
            return SignalResult()

        candidate_R = None
        for ev in events:
            try:
                event_date = _dt.date.fromisoformat(ev["date"])
            except (ValueError, TypeError):
                continue
            if event_date > decision_date:
                continue
            hour = ev.get("hour", 16)
            if hour < 9:
                r_date = event_date
            else:
                r_date = _resolve_next_trading_day(event_date, day_dates)
            if r_date is None or r_date > decision_date:
                continue
            r_idx = state["date_to_idx"].get(r_date)
            if r_idx is None:
                continue
            day_offset = idx_decision - r_idx
            if 1 <= day_offset <= reclaim_window_days:
                if candidate_R is None or r_date > candidate_R:
                    candidate_R = r_date

        if candidate_R is None:
            return SignalResult()

        R = candidate_R
        start_ts = pd.Timestamp.combine(R, _dt.time(0, 0)).tz_localize(tz)
        end_ts = pd.Timestamp.combine(decision_date, _dt.time(23, 59, 59)).tz_localize(tz)
        window = decision_frame.loc[start_ts:end_ts]
        if window.empty:
            return SignalResult()

        wdates = window.index.date
        g_close = window.groupby(wdates)["Close"].last()
        g_vol = window.groupby(wdates)["Volume"].sum()
        g_vwapvol = (window["VWAP"] * window["Volume"]).groupby(wdates).sum()
        cum_vol = g_vol.cumsum()
        cum_vwapvol = g_vwapvol.cumsum()
        avwap_series = cum_vwapvol / cum_vol

        ordered_dates = list(g_close.index)
        if len(ordered_dates) < 3 or ordered_dates[-1] != decision_date or ordered_dates[0] != R:
            return SignalResult()

        prior_dates = ordered_dates[1:-1]
        closed_below_seen = False
        for d in prior_dates:
            avwap_d = avwap_series.loc[d]
            close_d = g_close.loc[d]
            if pd.notna(avwap_d) and close_d < avwap_d:
                closed_below_seen = True
                break
        if not closed_below_seen:
            return SignalResult()

        avwap_today = avwap_series.loc[decision_date]
        today_close = g_close.loc[decision_date]
        if pd.isna(avwap_today) or today_close < avwap_today:
            return SignalResult()

        lookback_vols = state["daily_volume"][idx_decision - avg_volume_lookback_days: idx_decision]
        if len(lookback_vols) < avg_volume_lookback_days:
            return SignalResult()
        avg_vol = sum(lookback_vols) / len(lookback_vols)
        today_vol = state["daily_volume"][idx_decision]
        if avg_vol <= 0 or today_vol < volume_confirmation_multiple * avg_vol:
            return SignalResult()

        state["in_position"] = True
        state["entry_day_idx"] = idx_decision
        state["active_reaction_date"] = R
        return SignalResult(entry=True)

    def signal_fn(frame, i):
        if frame.empty:
            return SignalResult()

        n = len(frame)
        last_ts = frame.index[-1]
        today_date = last_ts.date()

        if state["last_seen_date"] is None:
            state["last_seen_date"] = today_date
            state["day_start_len"] = 0
        elif today_date != state["last_seen_date"]:
            # A new calendar day has begun. If the PREVIOUS day never
            # produced a >=15:59 decision bar (a scheduled early close, or
            # a data gap), its session is nonetheless now PROVABLY
            # complete -- frame is time-ascending, so no more bars dated
            # `prev_date` can ever arrive. Finalize and decide it now,
            # using only `frame.iloc[:n - 1]` (every bar strictly before
            # today's first bar) -- this never reads today's own data, so
            # it cannot look ahead; it only recovers a decision that would
            # otherwise be silently skipped.
            prev_date = state["last_seen_date"]
            prev_start_len = state["day_start_len"]
            state["last_seen_date"] = today_date
            state["day_start_len"] = n - 1
            if not state["day_dates"] or state["day_dates"][-1] != prev_date:
                catchup_frame = frame.iloc[: n - 1]
                _finalize_day(prev_date, catchup_frame, prev_start_len)
                return _decide(prev_date, catchup_frame)
            # else: prev_date already got its normal >=15:59 decision on
            # an earlier call this same session -- nothing left to do for
            # it, fall through to evaluate today's own (just-started) bar
            # below via the ordinary path.

        if last_ts.time() < _dt.time(15, 59):
            return SignalResult()

        if not state["day_dates"] or state["day_dates"][-1] != today_date:
            _finalize_day(today_date, frame, state["day_start_len"])

        return _decide(today_date, frame)

    return signal_fn


# --- avwap_band_extension_reversion ---
def make_avwap_band_extension_reversion_signal_fn(
    anchor_pivot_lookback_days: int = 20,
    band_stddev_multiple: float = 2.0,
    deceleration_confirmation_window_min: int = 30,
    max_hold_days: int = 10,
) -> SignalFn:
    """
    Long-only Anchored VWAP band-extension mean-reversion signal (SPY/QQQ/IWM).

    Anchor: for a given evaluation day D, the anchor session is whichever of the
    two extreme days -- the day with the highest daily High, or the day with the
    lowest daily Low -- inside the trailing `anchor_pivot_lookback_days` trading
    days ENDING AT AND INCLUDING D is more RECENT (this is the disclosed reading
    of "the most recent N-day swing pivot": it can be either a swing high or a
    swing low, whichever local extreme last occurred). The anchored VWAP (AVWAP)
    and its +-`band_stddev_multiple` standard-deviation bands are then built from
    real minute bars starting at that anchor session's first bar and accumulating
    forward, using each bar's own Volume and Alpaca-reported per-bar VWAP as the
    volume-weighted price series (cum_pv = cumsum(Volume*VWAP), cum_pv2 =
    cumsum(Volume*VWAP**2); avwap = cum_pv/cum_vol; variance =
    cum_pv2/cum_vol - avwap**2). This is the standard construction for an
    anchored-VWAP standard-deviation-band indicator; it is a disclosed
    approximation of a true tick-level VWAP, since only per-bar OHLCV+VWAP is
    available.

    Direction scope: the mechanism as described is naturally
    symmetric (price can extend to either the upper or lower band and mean-
    revert). This implementation covers ONLY the long side: entry triggers on
    a LOWER-band extension (an oversold dip) with deceleration/reversion back
    up toward the AVWAP centerline. The mirrored short-side variant (upper-band
    extension, fade back down) is NOT implemented here.

    Look-ahead guard: `frame` is trimmed to `frame.iloc[:i+1]` as the very
    first step, before any day-boundary/AVWAP/band computation runs. This
    makes the function correct regardless of whether the caller's `frame`
    already ends at bar `i` or additionally carries bars/days beyond `i` --
    either way, "today" is derived from where bar `i` actually falls, not
    from wherever `frame` happens to end, and today's own daily
    High/Low/Close and running AVWAP are always the genuine partial/running
    values as of bar `i`, never a same-day aggregate that includes bars that
    would not yet have printed in real time.

    Entry trigger, evaluated once per session at that session's close bar
    (identified by a fixed 15:59 America/New_York minute timestamp -- Alpaca's
    standard last regular-hours minute bar; early-close/half trading days, whose
    true last bar falls at a different time, are a disclosed and accepted gap in
    this simplified close-bar detector and are simply never evaluated for entry
    -- enforced uniformly for both today's live evaluation and the backward
    state-replay described below, so a half day can never fire an entry through
    either path):
      1. That day's daily Close is at or below the AVWAP-anchor's lower band
         (avwap - band_stddev_multiple * stdev), both computed through that
         day's own close.
      2. Deceleration confirmation: over the session's final
         `deceleration_confirmation_window_min` minutes, compute distance =
         Close - (per-minute cumulative AVWAP) at every minute in that window
         (both series are the true point-in-time values as of each minute, not
         the day's final value held constant). Split the window into two equal
         halves. Let first_change = last-minus-first distance within the first
         half, second_change = last-minus-first distance within the second half.
         Deceleration (or outright reversal) toward the AVWAP line is confirmed
         when second_change > first_change -- i.e. the down-move's net drift in
         the second half is less negative (or positive) than in the first half.
         If the session does not yet have a full window of minute bars, entry is
         withheld.

    Trade management: the anchor day used for a given trade is LOCKED at entry
    and held fixed for the life of that trade (a real anchored VWAP does not
    re-anchor mid-trade); the AVWAP/band values keep accumulating forward from
    that same anchor as time passes. Exit fires on the first bar (checked at
    every bar, not just session close, since a touch can happen intraday) where
    EITHER (a) that bar's High/Low range brackets the running AVWAP centerline
    value (Low <= avwap <= High -- interpreted as "touch"), OR (b) the current
    trading day index minus the entry-signal's day index reaches
    `max_hold_days` (a forced exit on exactly the `max_hold_days`'th trading day
    of exposure, counting the fill day, which occurs the next session's open, as
    day 1) -- whichever comes first, matching "exit ... or after a maximum hold
    of `max_hold_days` trading days, whichever comes first."

    Statelessness: `signal_fn` receives no persistent position state from the
    caller, so on every call it re-derives whether a position would currently be
    open (and since which trading day) by replaying the same entry/exit rules
    over a bounded backward window. This replay is sufficient to locate the
    entry day of any position that really is still open today (no position can
    be older than `max_hold_days` trading days, or it would already have been
    force-exited): the replay starts `max_hold_days + 1` trading days before
    today (never earlier than the first day with enough anchor history),
    assumes flat at that point, and walks forward day by day. Disclosed caveat,
    left as-is rather than papered over: because there is no cooldown between
    an exit and the next entry, this bounded assume-flat start is not a formal
    guarantee against every adversarial data sequence -- a real position that
    began just before the replay window, overlapping a second, independently
    -qualifying signal inside the window, could in principle be mis-replayed.
    No finite lookback window closes this completely; it is a shared,
    architecture-level tradeoff of the stateless-replay pattern required by
    this interface (not specific to this candidate's mechanism), and is
    considered low-likelihood in practice given how strict this mechanism's
    qualifying setup is. This also assumes the calling engine's fills match
    the contract exactly (next-bar-open entry, same-bar-close exit, one
    position at a time) -- a disclosed modeling assumption required by the
    stateless signal_fn interface.
    """
    N_ANCHOR = int(anchor_pivot_lookback_days)
    K = float(band_stddev_multiple)
    DECEL_WIN = int(deceleration_confirmation_window_min)
    MAX_HOLD = int(max_hold_days)
    SESSION_CLOSE_TIME = _time(15, 59)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        n = len(frame)
        if n == 0 or i < 0 or i >= n or DECEL_WIN < 2 or N_ANCHOR < 1 or MAX_HOLD < 1:
            return SignalResult()

        # Look-ahead guard: trim to bars [0, i] before any computation below.
        # `frame` may or may not already end at bar i depending on the caller's
        # own convention -- rather than assume, make it structurally impossible
        # for anything downstream (day boundaries, "today", the running AVWAP/
        # band, the deceleration window, forced-exit day counting) to read a
        # bar that would not yet have printed at bar i in real time.
        frame = frame.iloc[: i + 1]

        # Performance bound (recomputing over the whole frame on every bar is quadratic): everything below -- the per-day OHLC arrays,
        # the anchor-pivot search, and the stateless day-by-day replay
        # -- only ever needs the trailing N_ANCHOR (anchor lookback)
        # days plus MAX_HOLD (longest possible replay range) days plus
        # a safety margin, never the full multi-year frame. This
        # factory has NO per-day or per-bucket guard at all -- the
        # array construction below (dates = idx.date, the five
        # .to_numpy() conversions) runs on EVERY SINGLE BAR with zero
        # gating, so without this bound it would rescan the whole frame
        # on every call. `i` is rebound to the
        # bounded frame's own last position so every downstream
        # positional reference (idx[i], day_end_pos[-1], etc.) still
        # means "the current bar", just at its new relative offset --
        # safe because `i` is a local parameter, not shared with the
        # caller.
        _RECENT_WINDOW_BARS = (N_ANCHOR + MAX_HOLD + 10) * 400
        frame = frame.tail(_RECENT_WINDOW_BARS)
        i = len(frame) - 1
        n = len(frame)

        idx = frame.index
        dates = idx.date
        uniq_dates, day_start_pos = np.unique(dates, return_index=True)
        n_days = uniq_dates.shape[0]
        if n_days < N_ANCHOR:
            return SignalResult()

        day_end_pos = np.empty(n_days, dtype=np.int64)
        day_end_pos[:-1] = day_start_pos[1:] - 1
        day_end_pos[-1] = n - 1

        today_day_idx = n_days - 1
        is_close_bar = idx[i].time() == SESSION_CLOSE_TIME

        closes = frame['Close'].to_numpy(dtype='float64')
        highs = frame['High'].to_numpy(dtype='float64')
        lows = frame['Low'].to_numpy(dtype='float64')
        vols = frame['Volume'].to_numpy(dtype='float64')
        vwaps = frame['VWAP'].to_numpy(dtype='float64')

        daily_close = closes[day_end_pos]
        daily_high = np.maximum.reduceat(highs, day_start_pos)
        daily_low = np.minimum.reduceat(lows, day_start_pos)

        def anchor_for_day(day_idx):
            start_day = max(0, day_idx - N_ANCHOR + 1)
            window_highs = daily_high[start_day:day_idx + 1]
            window_lows = daily_low[start_day:day_idx + 1]
            hi_day = start_day + int(np.argmax(window_highs))
            lo_day = start_day + int(np.argmin(window_lows))
            return hi_day if hi_day >= lo_day else lo_day

        def cum_avwap_series(anchor_day_idx, end_pos):
            start_pos = int(day_start_pos[anchor_day_idx])
            v = vols[start_pos:end_pos + 1]
            p = vwaps[start_pos:end_pos + 1]
            cv = np.cumsum(v)
            cpv = np.cumsum(v * p)
            cpv2 = np.cumsum(v * p * p)
            safe_cv = np.where(cv <= 0, np.nan, cv)
            avwap = cpv / safe_cv
            var = cpv2 / safe_cv - avwap * avwap
            var = np.where(var < 0, 0.0, var)
            std = np.sqrt(var)
            return avwap, std, start_pos

        def entry_fires(day_idx):
            if day_idx < N_ANCHOR - 1:
                return False, None
            close_pos = int(day_end_pos[day_idx])
            if idx[close_pos].time() != SESSION_CLOSE_TIME:
                # Uniform half-day/early-close gap (disclosed): applies
                # identically whether day_idx is today or a day being replayed
                # from history below, so the two paths can never disagree.
                return False, None
            anchor_day_idx = anchor_for_day(day_idx)
            avwap_s, std_s, start_pos = cum_avwap_series(anchor_day_idx, close_pos)
            avwap_close = avwap_s[-1]
            std_close = std_s[-1]
            if not (np.isfinite(avwap_close) and np.isfinite(std_close)):
                return False, None
            lower_band = avwap_close - K * std_close
            if not (daily_close[day_idx] <= lower_band):
                return False, None

            day_open_pos = int(day_start_pos[day_idx])
            window_start = max(day_open_pos, close_pos - DECEL_WIN + 1)
            avail = close_pos - window_start + 1
            if avail < DECEL_WIN:
                return False, None

            rel_start = window_start - start_pos
            rel_end = close_pos - start_pos
            win_avwap = avwap_s[rel_start:rel_end + 1]
            win_close = closes[window_start:close_pos + 1]
            distance = win_close - win_avwap
            mid = DECEL_WIN // 2
            first_half = distance[:mid]
            second_half = distance[mid:]
            first_change = first_half[-1] - first_half[0]
            second_change = second_half[-1] - second_half[0]
            if not (second_change > first_change):
                return False, None
            return True, anchor_day_idx

        def touched_centerline(anchor_day_idx, day_idx, up_to_pos):
            day_open_pos = int(day_start_pos[day_idx])
            if up_to_pos < day_open_pos:
                return False
            avwap_s, _std_s, start_pos = cum_avwap_series(anchor_day_idx, up_to_pos)
            rel_start = day_open_pos - start_pos
            rel_end = up_to_pos - start_pos
            win_avwap = avwap_s[rel_start:rel_end + 1]
            win_high = highs[day_open_pos:up_to_pos + 1]
            win_low = lows[day_open_pos:up_to_pos + 1]
            return bool(np.any((win_low <= win_avwap) & (win_avwap <= win_high)))

        sim_start = max(N_ANCHOR - 1, today_day_idx - MAX_HOLD - 1)
        in_position = False
        entry_day_idx = None
        anchor_day_idx = None
        d = sim_start
        while d < today_day_idx:
            if not in_position:
                fired, a_idx = entry_fires(d)
                if fired:
                    in_position = True
                    entry_day_idx = d
                    anchor_day_idx = a_idx
            else:
                hold_days = d - entry_day_idx
                if hold_days >= MAX_HOLD:
                    in_position = False
                    entry_day_idx = None
                    anchor_day_idx = None
                elif touched_centerline(anchor_day_idx, d, int(day_end_pos[d])):
                    in_position = False
                    entry_day_idx = None
                    anchor_day_idx = None
            d += 1

        if in_position:
            hold_days_today = today_day_idx - entry_day_idx
            exit_maxhold = hold_days_today >= MAX_HOLD
            exit_touch = touched_centerline(anchor_day_idx, today_day_idx, i)
            return SignalResult(entry=False, exit=bool(exit_maxhold or exit_touch))

        if is_close_bar:
            fired, _a = entry_fires(today_day_idx)
            return SignalResult(entry=bool(fired), exit=False)

        return SignalResult()

    return signal_fn


# --- earnings_reaction_delayed_entry ---
def make_earnings_reaction_delayed_entry_signal_fn(
    earnings_events: list[dict],
    reaction_window_interval_minutes: int = 15,
    reaction_window_skip_intervals: int = 5,
    entry_time_et: str = "10:45",
    surprise_quintile_threshold: float = 0.2,
    max_hold_days: int = 10,
) -> SignalFn:
    """
    Delayed-entry PEAD reaction strategy (Ogneva, Xia & Ye 2021 / Bernard & Thomas 1989
    drift), LONG SIDE ONLY.

    For a ticker's own real historical earnings-reaction days (resolved from
    `earnings_events`: a before-open or during-session report, hour < 16, reacts the SAME
    calendar day; an after-close report, hour >= 16, reacts the NEXT real trading day --
    resolved each call using only trading dates already observed up to bar i, never a date
    beyond it), the function skips the first `reaction_window_skip_intervals` *
    `reaction_window_interval_minutes` minutes after the 09:30 ET open (5*15=75min -> 10:45,
    matching `entry_time_et`) and, at the bar whose time-of-day equals `entry_time_et`,
    measures that day's cumulative reaction return (that bar's Close vs. the day's own 09:30
    Open). It compares this return against the ticker's own EXPANDING, no-lookahead
    distribution of realized reaction returns on all of that ticker's PRIOR real reaction
    days (same open->entry_time_et-close measurement, cached as each day is observed). If
    today's reaction return is at or above the top-`surprise_quintile_threshold` quantile of
    that historical distribution (e.g. 0.2 -> top quintile), it enters long at that bar.
    Position is held for up to `max_hold_days` real trading days, exiting on the first bar at
    or after `entry_time_et` on the day that count is reached, using the same closure state
    used to track "today's" reaction outcome (this signal is necessarily stateful across
    calls -- the SignalFn contract carries no external position state -- and assumes calls
    arrive in ascending bar order, i.e. a normal sequential backtest walk).

    Disclosed scope decisions / approximations:
    - LONG ONLY, by this pipeline's engine-imposed single-direction scope: the mechanism's
      mirrored short leg (entering short on an extreme BOTTOM-quintile reaction) is
      deliberately not implemented; a bottom-quintile day is simply not traded.
    - The `earnings_events` records' own `surprise_pct` field (if present) is not consumed by
      this signal -- the candidate's own mechanism text bases the decision on the realized
      PRICE reaction return's quantile rank, not on the reported earnings-surprise magnitude,
      so only each event's `date`/`hour` are used (to identify real reaction days and to
      time-anchor the real open/close prices around them); `earnings_events` is still
      load-bearing (no reaction days are ever recognized without it), so `uses_earnings_data`
      is true.
    - Reports with 9:00-15:59 ET hours are treated as same-day reactions (bundled with the
      pre-market case) as a disclosed simplification, kept for completeness/robustness.
    - The historical reaction-return distribution requires at least 5 real prior
      reaction-day observations for this ticker before the signal will ever fire (a quintile
      concept is close to meaningless with fewer); until then, returns no entry.
    - The mechanism prose says "enter ... at the 10:45 bar's close" -- this factory instead
      fires entry=True on that exact bar, and (per this pipeline's fixed engine contract)
      actual execution happens at the NEXT bar's open, one minute later; this one-minute fill
      offset versus the idealized "at the close" description is a negligible, disclosed
      approximation forced by the engine's own execution convention, not a choice made here.
    - Exit timing approximates "hold up to max_hold_days trading days" by re-using the same
      `entry_time_et` intraday anchor on the target trading day, rather than requiring
      end-of-day-close detection (which a no-lookahead, growing frame slice cannot verify
      without seeing a following bar). Day counting is INCLUSIVE of the entry day (entry day
      = day 1), matching `run_intraday_backtest`'s own `max_hold_days` convention exactly
      (its docstring: "counting the entry day as day 1") -- this is deliberate, not
      cosmetic: since a signal function has no channel to learn that the engine force-closed
      a position via its own generic `hit_time_stop` (the SignalResult contract carries no
      such feedback), this signal's own exit condition MUST fire on/before the day the
      engine's own time-stop would, using the identical max_hold_days value, or this
      closure's internal `state["in_position"]` silently drifts out of sync with the
      engine's real (external) position state. The day count is INCLUSIVE
      (`entry_date <= d <= today`), which fires this signal's own `signal_exit`
      at `entry_time_et` on the max_hold_days-th day, strictly before the engine's own
      end-of-day `time_stop` on that same day, keeping the two in sync. An EXCLUSIVE
      count (`entry_date < d <= today`) would fire one trading day LATER than the
      engine's own force-close: the engine would always close the position first via
      `hit_time_stop` while this signal's own branch never fired, leaving the closure's
      `state["in_position"]` incorrectly True, so the next trading day's
      entry-decision bar would be consumed by a delayed internal self-heal instead of a
      real entry check.
    """
    import bisect
    from datetime import datetime as _dt, time as _time, timedelta as _timedelta

    hh, mm = (int(x) for x in str(entry_time_et).split(":"))
    entry_time = _time(hh, mm)

    skip_minutes = int(reaction_window_interval_minutes) * int(reaction_window_skip_intervals)
    market_open = _time(9, 30)
    _digest_end_dt = _dt.combine(_dt(2000, 1, 1), market_open) + _timedelta(minutes=skip_minutes)
    digest_end_time = _digest_end_dt.time()

    MIN_HISTORY = 5  # minimum real prior reaction-day observations before this can fire

    events = sorted(earnings_events, key=lambda e: e["date"])

    state = {
        "in_position": False,
        "entry_date": None,
        "last_date": None,
        "trading_dates": [],       # ascending list of distinct dates seen so far
        "trading_dates_set": set(),
        "day_open": {},            # date -> that day's real 09:30 Open
        "day_entry_close": {},     # date -> that day's real entry_time_et-bar Close
    }

    def _reaction_date_for_event(event):
        ev_date = pd.Timestamp(event["date"]).date()
        hour = event["hour"]
        if hour >= 16:
            # after-close report -> real NEXT trading day, resolved only from trading
            # dates already observed (never a date past "today" at call time)
            idx = bisect.bisect_right(state["trading_dates"], ev_date)
            if idx < len(state["trading_dates"]):
                return state["trading_dates"][idx]
            return None
        # pre-market or during-session report -> same real trading day
        return ev_date if ev_date in state["trading_dates_set"] else None

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        row = frame.iloc[-1]
        ts = frame.index[-1]
        today = ts.date()
        now_time = ts.time()

        # --- maintain cheap per-day caches (no full-frame rescans each call) ---
        if today != state["last_date"]:
            state["trading_dates"].append(today)
            state["trading_dates_set"].add(today)
            state["last_date"] = today
            open_px = row["Open"]
            if open_px == open_px and open_px != 0:  # not NaN, not zero
                state["day_open"][today] = float(open_px)

        if now_time == entry_time:
            close_px = row["Close"]
            if close_px == close_px:  # not NaN
                state["day_entry_close"][today] = float(close_px)

        # --- exit handling: currently in a position ---
        if state["in_position"]:
            entry_date = state["entry_date"]
            # INCLUSIVE of the entry day itself (entry day counted as day 1),
            # matching run_intraday_backtest's own max_hold_days convention
            # EXACTLY. This must fire on/before the day the engine's own
            # generic hit_time_stop force-close would (same max_hold_days
            # value) -- see the factory docstring's exit-timing passage for
            # why an exclusive count would let the two drift out of sync.
            elapsed = sum(1 for d in state["trading_dates"] if entry_date <= d <= today)
            if elapsed >= int(max_hold_days) and now_time >= entry_time:
                state["in_position"] = False
                state["entry_date"] = None
                return SignalResult(exit=True)
            return SignalResult()

        # --- entry handling: only ever decide at the designated bar ---
        if now_time != entry_time:
            return SignalResult()
        if now_time < digest_end_time:
            return SignalResult()

        reaction_date_today = None
        for ev in events:
            rd = _reaction_date_for_event(ev)
            if rd == today:
                reaction_date_today = rd
                break
        if reaction_date_today is None:
            return SignalResult()

        day_open = state["day_open"].get(today)
        today_close = state["day_entry_close"].get(today)
        if day_open is None or today_close is None or day_open == 0:
            return SignalResult()
        today_reaction_return = today_close / day_open - 1.0

        hist_returns = []
        for ev in events:
            rd = _reaction_date_for_event(ev)
            if rd is None or rd >= today:
                continue
            o = state["day_open"].get(rd)
            c = state["day_entry_close"].get(rd)
            if o is None or c is None or o == 0:
                continue
            hist_returns.append(c / o - 1.0)

        if len(hist_returns) < MIN_HISTORY:
            return SignalResult()

        top_cut = float(np.quantile(np.asarray(hist_returns, dtype=float), 1.0 - float(surprise_quintile_threshold)))

        if today_reaction_return >= top_cut:
            state["in_position"] = True
            state["entry_date"] = today
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- intraday_seasonality_continuation ---
def make_intraday_seasonality_continuation_signal_fn(
    signal_bar_start_et: str = "09:30",
    signal_bar_minutes: int = 30,
    lookback_days: int = 20,
    signal_tercile_threshold: float = 0.33,
    max_hold_days: int = 15,
):
    """Implements the LONG side of the Heston-Korajczyk-Sadka-style intraday time-of-day
    return continuation effect. Each trading day this closure watches the fixed clock-time
    bucket spanning `signal_bar_minutes` one-minute bars starting at `signal_bar_start_et`
    (default 09:30-10:00 ET); the bucket's return is (close of the bucket's LAST minute bar
    / open of the bucket's FIRST minute bar) - 1. On that day's LAST bucket bar (09:59 ET for
    the default window), today's bucket return is compared against the trailing
    `lookback_days` trading days of that SAME bucket's returns (today excluded); if it sits at
    or above the (1 - signal_tercile_threshold) quantile of that trailing distribution (the
    top tercile when threshold=0.33) a long entry is signaled on that bar, so the engine's
    next-bar-open fill lands within one minute of the bucket's own close -- a disclosed,
    engine-imposed approximation of "enter at the 10:00 ET bar close", since the contract only
    allows next-bar-open entry fills, never same-bar-close fills. The mirrored short side
    (bottom-tercile entries) described in the source mechanism is DELIBERATELY NOT
    implemented -- this factory only ever signals long entries, per this pipeline's
    scoping: direction is a single whole-call engine parameter. A held position is exited
    on the bucket's LAST bar of the trading day that is exactly `max_hold_days` trigger-bar
    occurrences (i.e. real trading days, since the trigger fires once per real trading day and
    naturally skips weekends/holidays with no separate calendar needed) after the entry day;
    because the exit itself fires as exit=True on that same bucket-close bar, the engine fills
    it at that bar's own close, which IS the bucket's close price -- an exact match to
    "exiting at the SAME 09:30-10:00 bucket's close on the exit day", no approximation needed
    for the exit leg. On the exit day itself no new entry is evaluated: this is NOT naturally
    true of the engine's calling convention (see the double-invocation note below) and is enforced by an
    explicit `last_processed_ts` guard rather than assumed; a fresh entry can only be signaled
    starting the following day's trigger bar. Internal state (whether a position is open,
    trigger-bar count since entry, and the last trigger timestamp actually evaluated) is kept
    in the closure and assumes signal_fn is invoked in ascending time order, matching how the
    engine contract describes execution; it also assumes every entry=True signal this closure
    emits is actually filled by the engine (the closure does not observe fills directly, only
    its own signals).

    Double-invocation note (see `src/mining/intraday_backtest_engine.py`'s
    `run_intraday_backtest` loop): the engine does NOT call this closure exactly
    once per bar. On the bar `t` where a position is open, it calls
    `signal_fn(df.iloc[:t+1], t)` (the "position open" exit-check branch). If that call signals
    exit, the position is closed and, on the VERY NEXT loop iteration (bar `t+1`, position now
    None), the engine calls `signal_fn(df.iloc[:t+1], t)` AGAIN via the "position is None"
    entry-check branch (`signal_fn(df.iloc[:i], i-1)` with `i=t+1` reduces to the identical
    `(df.iloc[:t+1], t)` pair). Without a guard, that second, immediately-following call would
    see the closure's own `in_position` flag already reset to False by the first call and would
    fall through to a FRESH entry evaluation using the SAME trigger bar's just-realized bucket
    return -- an undisclosed same-day (effectively same-bar) re-entry that contradicts both the
    stated mechanism (one decision per trigger bar) and this factory's own documented intent.
    The guard records the timestamp of the last trigger bar actually evaluated
    (`state["last_processed_ts"]`) and returns a no-op `SignalResult()` immediately if the
    current bar's timestamp matches it -- this only ever suppresses the exact-duplicate replay
    described above; every other trigger-bar timestamp is distinct across the series and is
    evaluated normally."""
    from datetime import datetime, timedelta

    start_parts = signal_bar_start_et.split(":")
    start_hour, start_minute = int(start_parts[0]), int(start_parts[1])
    window_minutes = int(signal_bar_minutes)

    _base = datetime(2000, 1, 1, start_hour, start_minute)
    window_start_time = _base.time()
    window_end_time = (_base + timedelta(minutes=window_minutes - 1)).time()

    top_quantile = 1.0 - float(signal_tercile_threshold)
    _hold_days = int(max_hold_days)
    _lookback = int(lookback_days)

    # Cap the slice of `frame` we actually scan each trigger-bar call so cost stays bounded
    # regardless of how many total years of minute bars `frame` has grown to by bar i.
    _BARS_PER_DAY_CAP = 500
    _lookback_bars = (_lookback + _hold_days + 10) * _BARS_PER_DAY_CAP

    state = {
        "in_position": False,
        "days_held": 0,  # count of trigger-bar occurrences observed since entry, inclusive
        # Timestamp of the last trigger bar this closure actually evaluated (past the
        # time-of-day gate below). Guards against the engine double-invocation described
        # in the docstring's double-invocation note: an exit-check call and the immediately-following entry-check
        # call can share the IDENTICAL (frame, i) pair on the bar an exit fires. Without this
        # guard the second call would re-evaluate the same trigger bar as a fresh entry
        # opportunity using the return that just closed the position.
        "last_processed_ts": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        idx = frame.index
        current_ts = idx[-1]
        current_time = current_ts.time()

        if current_time != window_end_time:
            return SignalResult()

        if current_ts == state["last_processed_ts"]:
            # Exact-duplicate replay of a trigger bar already evaluated by the
            # immediately-preceding call (see the guard's rationale above) -- no new
            # decision, no state mutation.
            return SignalResult()
        state["last_processed_ts"] = current_ts

        current_date = current_ts.date()

        recent = frame.iloc[-_lookback_bars:] if len(frame) > _lookback_bars else frame
        r_idx = recent.index

        start_mask = r_idx.time == window_start_time
        end_mask = r_idx.time == window_end_time
        if not start_mask.any() or not end_mask.any():
            return SignalResult()

        starts = recent.loc[start_mask, "Open"]
        ends = recent.loc[end_mask, "Close"]

        starts_by_date = {ts.date(): val for ts, val in starts.items()}
        ends_by_date = {ts.date(): val for ts, val in ends.items()}

        bucket_returns = {}
        for d, o in starts_by_date.items():
            c = ends_by_date.get(d)
            if c is None or o is None or pd.isna(o) or pd.isna(c) or o == 0:
                continue
            bucket_returns[d] = (c / o) - 1.0

        if current_date not in bucket_returns:
            return SignalResult()

        today_return = bucket_returns[current_date]
        prior_dates = sorted(d for d in bucket_returns if d < current_date)

        result_entry = False
        result_exit = False

        if state["in_position"]:
            state["days_held"] += 1
            if state["days_held"] >= _hold_days:
                result_exit = True
                state["in_position"] = False
                state["days_held"] = 0
        elif len(prior_dates) >= _lookback:
            trailing = prior_dates[-_lookback:]
            trailing_returns = np.array([bucket_returns[d] for d in trailing], dtype=float)
            hi_thresh = np.quantile(trailing_returns, top_quantile)
            if today_return >= hi_thresh:
                result_entry = True
                state["in_position"] = True
                state["days_held"] = 0

        return SignalResult(entry=result_entry, exit=result_exit)

    return signal_fn


# --- nr7_orb_trend_continuation ---
def make_nr7_orb_trend_continuation_signal_fn(
    nr_lookback_days: int = 7,
    opening_range_minutes: int = 30,
    confirmation_bar_minutes: int = 5,
    max_hold_days: int = 10,
    atr_period: int = 14,
    atr_trailing_stop_mult: float = 2.5,
) -> SignalFn:
    """Crabel-style NR7-conditioned opening-range-breakout trend continuation, LONG side only.

    Each new trading day, a "yesterday's daily range is the narrowest of the trailing
    `nr_lookback_days` sessions" (NR7 when the default of 7 is used) precondition is computed
    once, from a daily OHLC table built by resampling the minute-bar `frame` (grouped by
    calendar date, restricted to days strictly before today so nothing from the still-forming
    current session leaks in). Only a bounded tail slice of `frame` is resampled each day
    (a generous multiple of the days actually needed) rather than the whole multi-year
    history, since re-grouping years of minute bars once per session would be needlessly
    expensive. The opening range is the running High/Low of the first `opening_range_minutes`
    one-minute bars of the day. The "next 5-minute bar" confirmation is interpreted as the
    next `confirmation_bar_minutes` consecutive one-minute bars taken as one block; only that
    block's own final close (the mechanism's actual test) is used -- its intrabar high/low are
    not needed since the mechanism only checks a close-beyond-range condition. If that close is
    above the opening-range high (and NR7 held, and we are flat), entry=True fires on that
    confirmation bar. A downside breakout would symmetrically fire a short entry under the
    mechanism's own description, but `run_intraday_backtest`'s own `direction` is a WHOLE-CALL
    parameter (no per-trade direction on `SignalResult`), so a single signal_fn cannot emit both
    long and short entries in one call without the engine mis-pricing one side's gross_ret --
    per this pipeline's fixed long-only scope, only the upside/long case is implemented; the short
    mirror is deliberately omitted, not silently dropped.

    Because the calling engine executes an entry signal at the FOLLOWING bar's open (fixed,
    engine-level next-open execution) rather than literally at the confirmation bar's own
    close, the actual fill is observed one bar later by reading that next bar's real Open --
    itself already-happened, non-look-ahead information once that bar is the current one --
    and used from then on as the position's cost basis. This one-bar fill delay versus the
    mechanism's literal "enter ... at that confirmation bar's close" wording is a structural
    consequence of the fixed engine contract, not a modeling choice, and is disclosed rather
    than silently reconciled.

    The candidate's own `parameters` (nr_lookback_days, opening_range_minutes,
    confirmation_bar_minutes, max_hold_days) are used exactly as given and unmodified.
    `max_hold_days` is accepted here (kept in the factory signature to match the catalog
    entry's parameter surface) but deliberately NOT re-implemented inside this signal_fn: the
    calling engine (`run_intraday_backtest`) already force-closes a position at the LAST bar of
    its `max_hold_days`-th trading day, fed the identical value by the orchestrating mining
    script -- and that "last bar of the day" determination is something signal_fn structurally
    CANNOT compute causally (it would need to know whether another bar is still coming today,
    which is exactly the information next-bar look-ahead is defined to exclude). An internal
    per-day hold counter that fired an exit as
    soon as the count reached `max_hold_days` would fire on the FIRST bar of the Nth day,
    not the last, silently shortening every max-hold exit by very close to a full trading day
    versus both the mechanism's own "hold ... for up to `max_hold_days` trading days" wording
    and the engine's own matching semantics, and mislabeling the resulting trade
    as a "signal_exit" instead of "time_stop" in `trades_df` -- so this factory relies on the
    engine's own enforcement of the identical parameter value.

    The mechanism names a "trailing-stop exit" as an alternative to the max_hold_days exit but
    the candidate provides no trailing-stop parameter of its own, so a disclosed, reasonably
    faithful default is added as two extra optional factory arguments with sensible defaults: a
    simple (non-Wilder-smoothed) daily ATR(`atr_period`) computed the same way as the NR7 daily
    table, and a trailing stop at (peak close since fill) minus `atr_trailing_stop_mult` times
    that ATR, re-evaluated once per day using the prior day's completed ATR. While in a
    position, exit=True fires the first bar the ATR trailing stop is breached (the
    max_hold_days cap is left to the engine, as above); a fresh entry is never signalled on the
    same bar an exit is signalled, to avoid same-bar re-entry ambiguity. Early bars with
    insufficient daily history for either the NR7 check or the ATR calculation simply skip
    entry evaluation that day (the engine's own max_hold_days force-close still applies to any
    already-open position regardless of ATR availability) rather than raising, per the engine's
    early-bar contract.
    """

    def _compute_daily_stats(frame: pd.DataFrame, today):
        needed_days = max(nr_lookback_days, atr_period + 1) + 15
        tail_n = min(len(frame), needed_days * 400)
        recent = frame.iloc[-tail_n:]
        dates = recent.index.date
        grouped = recent.groupby(dates)
        daily_high = grouped["High"].max()
        daily_low = grouped["Low"].min()
        daily_close = grouped["Close"].last()
        daily = pd.DataFrame(
            {"High": daily_high, "Low": daily_low, "Close": daily_close}
        )
        daily = daily[daily.index < today]

        nr_ok = False
        if len(daily) >= nr_lookback_days:
            ranges = daily["High"] - daily["Low"]
            trailing = ranges.iloc[-nr_lookback_days:]
            yesterday_range = ranges.iloc[-1]
            nr_ok = bool(yesterday_range <= trailing.min())

        atr = None
        if len(daily) >= atr_period + 1:
            prev_close = daily["Close"].shift(1)
            tr = pd.concat(
                [
                    daily["High"] - daily["Low"],
                    (daily["High"] - prev_close).abs(),
                    (daily["Low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_val = tr.iloc[-atr_period:].mean()
            if pd.notna(atr_val):
                atr = float(atr_val)

        return nr_ok, atr

    state = {
        "current_date": None,
        "day_start_i": 0,
        "or_high": None,
        "or_low": None,
        "or_bars_seen": 0,
        "nr7_ok_today": False,
        "atr_today": None,
        "in_position": False,
        "pending_entry_fill": False,
        "entry_price": None,
        "peak_close": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        row = frame.iloc[-1]
        today = frame.index[-1].date()
        close_now = float(row["Close"])

        # 1. Day-change bookkeeping: reset per-day opening-range accumulators and
        # recompute the once-per-day NR7 / ATR daily stats off history strictly
        # before today.
        if today != state["current_date"]:
            state["current_date"] = today
            state["day_start_i"] = i
            state["or_high"] = None
            state["or_low"] = None
            state["or_bars_seen"] = 0
            state["nr7_ok_today"], state["atr_today"] = _compute_daily_stats(frame, today)

        minutes_into_day = i - state["day_start_i"]

        # 2. Resolve a fill flagged on the previous bar: the engine executes an
        # entry at THIS bar's open, which is now real, already-happened data.
        if state["pending_entry_fill"]:
            state["in_position"] = True
            state["entry_price"] = float(row["Open"])
            state["peak_close"] = close_now
            state["pending_entry_fill"] = False

        # 3. Exit check for an open position: ATR-based trailing stop off the
        # peak close since fill. The max_hold_days cap is intentionally NOT
        # re-implemented here -- see the docstring's "max_hold_days" paragraph:
        # the calling engine already force-closes at the last bar of the Nth
        # trading day using the identical parameter value, and that
        # last-bar-of-day determination is not something this function can
        # compute causally.
        exit_signal = False
        if state["in_position"]:
            state["peak_close"] = max(state["peak_close"], close_now)
            stop_hit = False
            if state["atr_today"] is not None:
                stop_level = state["peak_close"] - atr_trailing_stop_mult * state["atr_today"]
                stop_hit = close_now <= stop_level
            if stop_hit:
                exit_signal = True
                state["in_position"] = False

        # 4. Opening-range accumulation for the first `opening_range_minutes`
        # one-minute bars of the day.
        if minutes_into_day < opening_range_minutes:
            if state["or_high"] is None:
                state["or_high"] = float(row["High"])
                state["or_low"] = float(row["Low"])
            else:
                state["or_high"] = max(state["or_high"], float(row["High"]))
                state["or_low"] = min(state["or_low"], float(row["Low"]))
            state["or_bars_seen"] += 1

        # 5. Confirmation check: the block of `confirmation_bar_minutes` one-minute
        # bars immediately after the opening range, evaluated once, at its final bar.
        entry_signal = False
        confirm_bar_offset = opening_range_minutes + confirmation_bar_minutes - 1
        if (
            minutes_into_day == confirm_bar_offset
            and not state["in_position"]
            and not state["pending_entry_fill"]
            and not exit_signal
            and state["or_bars_seen"] >= opening_range_minutes
            and state["or_high"] is not None
            and state["nr7_ok_today"]
        ):
            if close_now > state["or_high"]:
                entry_signal = True
                state["pending_entry_fill"] = True
            # A downside close below state["or_low"] would be the mechanism's
            # mirrored short entry; deliberately not implemented (long-only scope,
            # see docstring).

        return SignalResult(entry=entry_signal, exit=exit_signal)

    return signal_fn


# --- closing_auction_deviation_reversal ---
def make_closing_auction_deviation_reversal_signal_fn(
    preclose_window_minutes: int = 5,
    deviation_threshold_bps: float = 15.0,
    max_hold_days: int = 5,
    retracement_fraction: float = 0.85,
    closing_bar_time: str = "15:59",
) -> "SignalFn":
    """
    Approximates Bogousslavsky & Muravyev (2023)'s closing-auction deviation
    reversal on minute-bar data that has no separately-tagged auction print.

    Each trading day's LAST regular-session minute bar is assumed to be
    timestamped `closing_bar_time` (default 15:59 ET, i.e. the bar covering
    15:59:00-15:59:59 of a standard 09:30-16:00 RTH session). This is a
    disclosed approximation, structurally required by two constraints
    together: Alpaca minute bars do not separately expose the off-exchange
    closing-auction cross price (so that bar's own Close is used as the
    best available proxy for "the eventual closing-auction print"), and the
    entry signal must fire ON that same bar so the caller's next-open
    execution convention lands at the true next session's open (waiting to
    retroactively detect a day boundary, e.g. via the following day's first
    bar, would already be one bar too late to catch that open). Half/early-
    close sessions, where 15:59 is not really the day's last bar, are not
    specially detected and simply fail to trigger that day -- a silent
    missed opportunity, never a silent look-ahead. This also means a held
    position's own day-count and retracement check are silently skipped on
    a half day (no bar exists at wall-clock 15:59 to evaluate) -- the
    caller's own independent `max_hold_days` time-stop (a day-index-based
    cap, not time-of-day-based) is the backstop that still force-closes a
    position that this function's own bookkeeping would otherwise stall on.

    The "last pre-close continuous-market quote/VWAP" is approximated as
    the volume-weighted average price (using each bar's own per-bar VWAP
    column, weighted by that bar's Volume) over the `preclose_window_minutes`
    minute bars immediately preceding the closing bar (the closing bar
    itself is excluded from this average). Any bar in that window whose
    VWAP is missing/non-finite is dropped from BOTH the numerator and the
    denominator (never counted in the weighting volume without also
    contributing its price), so a single bad print cannot silently bias
    the average low or high. The deviation, in bps, is
    `(close_print - preclose_vwap) / preclose_vwap * 1e4`.

    LONG-ONLY SCOPE (disclosed): the paper's mechanism is direction-neutral
    (fade whichever way the auction deviated), but per this pipeline's
    single-direction-per-call constraint only the long side is implemented
    here: a position is opened only when the closing print deviates DOWN
    from the pre-close level by more than `deviation_threshold_bps` (the
    mirrored short -- fading an upward deviation -- is simply skipped, never
    silently converted into a long).

    Entry fires (entry=True) on the deviation day's own closing bar, so the
    caller's next-open execution convention buys at the very next session's
    open, matching the mechanism. The exit is adaptive rather than a flat
    next-day sale: while in a trade, closure state tracks trading days
    elapsed since entry and a retracement target computed at entry as
    `close_print + retracement_fraction * (preclose_vwap - close_print)` --
    i.e. the price level representing an `retracement_fraction` (default
    0.85, mapping the paper's own documented ~85% average reversion into a
    concrete per-trade target, since "the estimated gap" is not itself a
    guarantee of a full 100% retracement) retracement of the estimated gap
    back toward the pre-close level. On each subsequent trading day's own
    closing bar, exit fires (same-bar-close execution, per the caller's
    exit convention) once that day's Close reaches the retracement target,
    or once `max_hold_days` trading days have elapsed since entry,
    whichever comes first.

    KNOWN LIMITATION (shared by any stateful signal_fn under this contract):
    this function has no visibility into whether the backtest engine
    actually acted on a previously emitted entry/exit signal (e.g. a risk
    limit rejecting a fill), so its internal "am I in a trade" bookkeeping
    is advisory and assumes every emitted signal was filled as directed.
    """
    close_h, close_m = (int(x) for x in closing_bar_time.split(":"))
    CLOSING_BAR_TIME = _time(close_h, close_m)

    state = {
        "in_trade": False,
        "days_held": 0,
        "target_price": None,
        "last_seen_date": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        bar_ts = frame.index[i]
        bar_date = bar_ts.date()
        bar_time = bar_ts.time()

        # Both entries and exit checks are evaluated only at daily-close
        # granularity, matching the mechanism's own "by the next session" /
        # "over up to N subsequent trading days" framing.
        if bar_time != CLOSING_BAR_TIME:
            return SignalResult()

        # Guard against double-processing the same calendar day if the feed
        # ever repeats or duplicates a timestamp.
        if state["last_seen_date"] == bar_date:
            return SignalResult()
        state["last_seen_date"] = bar_date

        if state["in_trade"]:
            state["days_held"] += 1
            today_close = float(frame["Close"].iloc[i])
            target = state["target_price"]
            retraced = target is not None and today_close >= target
            capped = state["days_held"] >= max_hold_days
            if retraced or capped:
                state["in_trade"] = False
                state["days_held"] = 0
                state["target_price"] = None
                return SignalResult(exit=True)
            return SignalResult()

        # Not currently in a trade -- evaluate today's closing-auction
        # deviation using only bars from today, all of which are <= i.
        # Bounded to a recent tail before the O(len(frame)) `.date`
        # conversion -- a single RTH session is at most ~390 one-minute
        # bars, so 1000 comfortably contains all of today's bars without
        # re-scanning the full multi-year frame on every closing bar.
        recent = frame.tail(1000)
        day_mask = recent.index.date == bar_date
        today_bars = recent.loc[day_mask]
        if len(today_bars) <= preclose_window_minutes:
            # Not enough of today's own history to build the pre-close
            # window (e.g. a short/gappy session) -- skip, no crash.
            return SignalResult()

        preclose_window = today_bars.iloc[-(preclose_window_minutes + 1):-1]
        if len(preclose_window) < preclose_window_minutes:
            return SignalResult()

        vol = preclose_window["Volume"].astype(float)
        if "VWAP" in preclose_window.columns:
            px = preclose_window["VWAP"].astype(float)
        else:
            px = preclose_window["Close"].astype(float)

        # Drop any bar whose price is missing/non-finite from BOTH the
        # numerator and the denominator together -- summing `vol` alone
        # (which defaults to skipna=True) while `(px * vol)` silently
        # NaNs out that same bar's contribution would count its volume in
        # the weighting base without its price, biasing the average.
        valid = np.isfinite(px.to_numpy(dtype=float))
        vol = vol[valid]
        px = px[valid]

        total_vol = float(vol.sum())
        if total_vol > 0:
            preclose_vwap = float((px * vol).sum() / total_vol)
        else:
            close_fallback = preclose_window["Close"].astype(float)
            close_fallback = close_fallback[np.isfinite(close_fallback.to_numpy(dtype=float))]
            preclose_vwap = float(close_fallback.mean()) if len(close_fallback) else float("nan")

        if not np.isfinite(preclose_vwap) or preclose_vwap <= 0:
            return SignalResult()

        close_print = float(today_bars["Close"].iloc[-1])
        deviation_bps = (close_print - preclose_vwap) / preclose_vwap * 1e4

        if deviation_bps < -abs(deviation_threshold_bps):
            gap = preclose_vwap - close_print
            state["in_trade"] = True
            state["days_held"] = 0
            state["target_price"] = close_print + retracement_fraction * gap
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- buyable_gap_up_continuation ---
def make_buyable_gap_up_continuation_signal_fn(
    min_gap_atr_multiple: float = 0.75,
    atr_lookback_days: int = 40,
    min_volume_multiple_of_50d_avg: float = 1.5,
    confirmation_window_minutes: int = 20,
    entry_time_start_et: str = "09:30",
    entry_time_end_et: str = "09:50",
    max_hold_days: int = 15,
    stop_type: str = "close_below_10d_ma",
) -> SignalFn:
    """Buyable Gap-Up Continuation (long-only, per the mining pipeline's single-direction-per-run
    scope decision -- the mechanism's mirrored short variant is intentionally NOT implemented).

    Each RTH trading day is evaluated for a gap-up "setup" using only daily aggregates built
    incrementally, day by day, from the intraday minute frame itself (no separate daily dataset
    is available): the session qualifies if today's open exceeds the prior close by at least
    `min_gap_atr_multiple` times a simple (non-Wilder, equal-weighted) `atr_lookback_days`-day
    ATR, computed from `atr_lookback_days`+1 finalized daily bars. `entry_time_start_et` /
    `entry_time_end_et` define the opening-range (OR) window (default 09:30-09:50 ET);
    `confirmation_window_minutes` must be consistent with that window (checked at construction)
    and is what actually derives the window's end internally. The OR high is the max High of
    RTH minute bars in [start, end) -- bounded on BOTH sides (not just `< end`), so a non-default
    `entry_time_start_et` later than the session's actual first bar does not pull pre-window bars
    into the OR high (harmless for the default 09:30 start, since the session's first bar already
    IS 09:30 per the data contract below, but load-bearing for any other start time). Since a
    resampled 5-minute bar's Close always equals the Close of its last constituent 1-minute bar,
    "a 5-minute bar closes above the OR high" is evaluated exactly (not approximated) by checking,
    at every clock-aligned 5-minute boundary (minute % 5 == 4) at or after the OR window's end,
    whether the CURRENT 1-minute bar's Close exceeds the OR high.

    Two full-day conditions from the mechanism -- "closes above the open" and "low of day stays
    above the prior day's high" (the gap not filling) -- cannot be known in full until the day's
    close, which would be look-ahead. They are approximated causally: the gap-not-filled
    condition is enforced continuously and irrevocably (if the running day-low ever drops to or
    below the prior day's high, that day's setup is permanently invalidated for the rest of the
    session), and "closes above the open" is checked only at the instant of the entry trigger
    using the then-current bar's Close vs. the session's actual open. "On pace for >= 1.5x the
    50-day average daily volume" is likewise approximated causally: cumulative volume so far
    today is linearly extrapolated to a full 390-minute RTH session and compared against
    `min_volume_multiple_of_50d_avg` times the 50-day average daily volume (the 50-day window is
    fixed at 50, per the parameter's own "50d_avg" naming -- there is no separate numeric
    parameter for it).

    Because this signal function is not told the backtest engine's actual position state, it
    tracks its own entry/exit bookkeeping via internal (closure) state, on the assumption that
    the engine enters at the next bar's open following an entry=True signal and exits at the
    close of the bar where exit=True is returned. It holds at most one position at a time for
    this ticker (a new entry is never signaled while this function's own tracked position is
    open, even if a fresh gap-up setup qualifies on a later day).

    Per-bar accumulation (day-low, cumulative volume, OR high, gap-fill check) is idempotent
    per absolute bar index `i`, tracked via `setup["last_accum_i"]`: `run_intraday_backtest`'s
    loop re-invokes `signal_fn` with the IDENTICAL (frame, i) pair the iteration immediately
    after any position close (an entry-check call re-examining the very bar that just produced
    an exit, structurally so a fresh setup can be considered for re-entry off the same bar) --
    one duplicate call per closed trade.
    Every other accumulated field is naturally idempotent under a repeat call (min/max/bool-OR),
    but `cum_volume` is a plain `+=` and would silently double-count that one bar's volume on
    the repeat call, which can flip the volume-pace check from fail to pass and fabricate a
    same-bar phantom entry that would not fire under correct single-counted volume (e.g. an
    existing position's max-hold exit landing on a bar that also independently qualifies as a
    fresh gap-up, with single-counted volume just under the 1.5x bar and double-counted volume
    clearing it) -- so the whole accumulation block is gated on `last_accum_i != i`.

    The trailing stop ("close below the 10-day moving average" -- only `stop_type` value
    implemented; any other value raises at construction) and the `max_hold_days` cap are both
    checked once per day, at that day's final RTH minute bar (15:59 ET, used as a proxy for the
    official daily close since only minute bars are available). The 10-day MA includes today's
    close. Hold days are counted from the entry day as day 0; the position is force-exited at
    the close of trading day index `max_hold_days - 1` if the MA stop has not already fired.
    Day-boundary and end-of-day detection rely on the data contract's guarantee of one row per
    real RTH minute with the session's first bar at 09:30 ET and last bar at 15:59 ET; unusual
    sessions (e.g. early closes) are not specially handled.
    """
    if stop_type != "close_below_10d_ma":
        raise ValueError(
            f"Unsupported stop_type: {stop_type!r} (only 'close_below_10d_ma' is implemented)"
        )

    window_start = dt.datetime.strptime(entry_time_start_et, "%H:%M").time()
    _anchor = dt.date(2000, 1, 1)
    window_end_computed = (
        dt.datetime.combine(_anchor, window_start) + dt.timedelta(minutes=confirmation_window_minutes)
    ).time()
    window_end_declared = dt.datetime.strptime(entry_time_end_et, "%H:%M").time()
    if window_end_computed != window_end_declared:
        raise ValueError(
            "confirmation_window_minutes (%r) does not reconcile with "
            "entry_time_start_et/entry_time_end_et (%r-%r)"
            % (confirmation_window_minutes, entry_time_start_et, entry_time_end_et)
        )
    window_end = window_end_computed

    EOD_TIME = dt.time(15, 59)
    SESSION_MINUTES = 390
    VOL_LOOKBACK_DAYS = 50
    MA_STOP_DAYS = 10

    keep_last = max(atr_lookback_days + 1, VOL_LOOKBACK_DAYS, MA_STOP_DAYS) + 30

    state = {
        "last_date": None,
        "day_start_pos": None,
        "daily_rows": [],  # finalized days strictly before "today", oldest -> newest
        "setup": {"qualified": False},
        "in_position": False,
        "entry_day_count": 0,
    }

    def _atr(daily_rows):
        if len(daily_rows) < atr_lookback_days + 1:
            return None
        rows = daily_rows[-(atr_lookback_days + 1):]
        trs = []
        for k in range(1, len(rows)):
            h = rows[k]["high"]
            l = rows[k]["low"]
            pc = rows[k - 1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs) / len(trs)

    def _avg_vol_50(daily_rows):
        if len(daily_rows) < VOL_LOOKBACK_DAYS:
            return None
        rows = daily_rows[-VOL_LOOKBACK_DAYS:]
        return sum(r["volume"] for r in rows) / VOL_LOOKBACK_DAYS

    def _evaluate_setup(daily_rows, today_open):
        atr = _atr(daily_rows)
        avg_vol = _avg_vol_50(daily_rows)
        if atr is None or avg_vol is None or not daily_rows:
            return {"qualified": False}
        prior = daily_rows[-1]
        gap = today_open - prior["close"]
        qualified = gap >= min_gap_atr_multiple * atr
        return {
            "qualified": qualified,
            "prior_high": prior["high"],
            "today_open": today_open,
            "avg_vol_50": avg_vol,
            "or_high": None,
            "day_low": None,
            "cum_volume": 0.0,
            "entered_today": False,
            "broken": False,
            "last_accum_i": None,
        }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        bar = frame.iloc[-1]
        ts = frame.index[-1]
        current_date = ts.date()
        current_time = ts.time()

        if state["last_date"] is None:
            state["last_date"] = current_date
            state["day_start_pos"] = i
            state["setup"] = _evaluate_setup(state["daily_rows"], bar["Open"])
        elif current_date != state["last_date"]:
            prev_slice = frame.iloc[state["day_start_pos"]:i]
            if len(prev_slice) > 0:
                state["daily_rows"].append({
                    "open": prev_slice["Open"].iloc[0],
                    "high": prev_slice["High"].max(),
                    "low": prev_slice["Low"].min(),
                    "close": prev_slice["Close"].iloc[-1],
                    "volume": prev_slice["Volume"].sum(),
                })
                if len(state["daily_rows"]) > keep_last + 30:
                    state["daily_rows"] = state["daily_rows"][-keep_last:]
            if state["in_position"]:
                state["entry_day_count"] += 1
            state["day_start_pos"] = i
            state["last_date"] = current_date
            state["setup"] = _evaluate_setup(state["daily_rows"], bar["Open"])

        setup = state["setup"]
        entry = False
        exit_ = False

        if setup.get("qualified"):
            # Idempotency guard: run_intraday_backtest re-invokes signal_fn with the IDENTICAL
            # (frame, i) pair on the iteration immediately after any position close (an
            # entry-check call re-examining the very bar that just closed a trade). Every other
            # field accumulated here is naturally idempotent (min/max/bool-OR), but cum_volume's
            # `+=` is not -- guard the whole block so a repeat call for the same bar never
            # re-applies it. Without this guard, a bar
            # that both closes an old position and independently qualifies as a fresh gap-up can
            # have its own volume double-counted, which can flip the volume-pace check and
            # fabricate a same-bar phantom entry that would not fire under correct accounting.
            if setup.get("last_accum_i") != i:
                low = bar["Low"]
                setup["day_low"] = low if setup["day_low"] is None else min(setup["day_low"], low)
                setup["cum_volume"] += bar["Volume"]
                if window_start <= current_time < window_end:
                    high = bar["High"]
                    setup["or_high"] = high if setup["or_high"] is None else max(setup["or_high"], high)
                if setup["day_low"] <= setup["prior_high"]:
                    setup["broken"] = True
                setup["last_accum_i"] = i

            if (
                not setup["broken"]
                and not setup["entered_today"]
                and not state["in_position"]
                and setup["or_high"] is not None
                and current_time >= window_end
                and current_time.minute % 5 == 4
            ):
                minutes_elapsed = i - state["day_start_pos"] + 1
                projected_volume = setup["cum_volume"] / minutes_elapsed * SESSION_MINUTES
                volume_ok = projected_volume >= min_volume_multiple_of_50d_avg * setup["avg_vol_50"]
                bullish_ok = bar["Close"] > setup["today_open"]
                breakout_ok = bar["Close"] > setup["or_high"]
                if volume_ok and bullish_ok and breakout_ok:
                    entry = True
                    setup["entered_today"] = True
                    state["in_position"] = True
                    state["entry_day_count"] = 0

        if state["in_position"] and current_time == EOD_TIME:
            daily_rows = state["daily_rows"]
            stop_hit = False
            if len(daily_rows) >= MA_STOP_DAYS - 1:
                recent_closes = [r["close"] for r in daily_rows[-(MA_STOP_DAYS - 1):]] + [bar["Close"]]
                ma10 = sum(recent_closes) / MA_STOP_DAYS
                stop_hit = bar["Close"] < ma10
            max_hold_hit = state["entry_day_count"] >= (max_hold_days - 1)
            if stop_hit or max_hold_hit:
                exit_ = True
                state["in_position"] = False
                state["entry_day_count"] = 0

        return SignalResult(entry=entry, exit=exit_)

    return signal_fn


# --- darvas_box_breakout_volume_confirmed ---
"""Darvas Box Breakout (Volume-Confirmed) -- intraday/swing SignalFn for the
darvas_box_breakout_volume_confirmed mining candidate.

Implementation notes:

  (a) Imports. `SignalFn` and `SignalResult` are imported at the top of this
      module from `mining.intraday_backtest_engine`. The outer factory's
      `-> SignalFn` annotation and every `SignalResult(...)` return in
      `signal_fn` depend on them.

  (b) Entry date. `intraday_backtest_engine.run_intraday_backtest` calls
      `signal_fn` TWICE on a fresh-entry bar: once with the frame ending at
      bar i-1 (the flat/entry decision, next-open execution) and, in the SAME
      loop iteration, once more with the frame ending at bar i (the
      exit-check for the position the engine just opened at bar i's open).
      Bar i-1 and bar i fall on different calendar days when the confirming
      breakout bucket is the trading day's LAST bucket. Stamping
      `state["entry_date"]` during the first call would then record one day
      too early, overcount `days_held` from the first exit-check call, and
      let `max_hold_days` fire a trading day early (labelled "signal_exit"
      rather than the engine's own "time_stop"). So only `entry_date` is
      deferred: it is stamped on the first "in position" call after entry,
      which always sees `today` as the real entry bar's own date, in both the
      same-day and day-boundary cases, with no special-casing.

  (c) Floor stop. `floor_at_entry` is NOT deferred like `entry_date`; it is
      seeded in the ENTRY branch from that call's own `box`, the box that was
      actually broken (built from completed_dates strictly before the
      in-progress breakout day, so it never includes the breakout day's own
      high). Reason: on the later "in position" call the breakout day has
      become a completed day, and a confirmed breakout's bucket_close is above
      the OLD ceiling (the window's previous max daily High), so the breakout
      day's own High is the window's new max. `_compute_box` then treats it
      as a fresh peak and resets box_age to a small number, below
      `box_min_days` for roughly the next `box_min_days` trading days after
      essentially every breakout. A floor seeded from that box would stay
      `None`, leaving the stop-loss leg inactive for most of the early hold.
      The ratchet in the "in position" branch skips an update whenever `box`
      is temporarily `None` and only ever raises the floor, so the stop holds
      at the pre-breakout floor while the box is immature and resumes
      ratcheting once a mature box reappears.

Box construction, breakout/volume confirmation, and the trailing-floor stop's
ratchet mechanics follow the catalog entry's mechanism.

Disclosed approximations: (1) Ceiling is defined as the highest daily High
within the trailing box_max_days completed-day window (peak = earliest day
achieving that max, to maximize box_age); since a day's Close can never
exceed its own High, this makes "no completed day has closed above ceiling"
automatically true by construction, which is how the mechanism's daily
non-breach requirement is operationalized rather than via a separate scan.
(2) Because of that, a breakout can only be evaluated against TODAY's
(in-progress) intraday bars, never a prior completed day. (3) Volume
confirmation compares each candidate breakout bucket's volume to the mean
of the same time-of-day 5-minute bucket's volume across however many of the
trailing volume_lookback_days completed days are actually available (no
separate minimum-sample floor beyond box_min_days having already elapsed)
-- avoids over-blocking entries early in a box's tradeable life. (4) The
multi-week hold with a stop trailed to the most recent Darvas box floor is
implemented as closure-held position state (entry date + a floor that only
ratchets up), which is inherently path-dependent and therefore assumes the
engine calls signal_fn once per bar in strictly ascending order for one
continuous single-ticker walk (see (b) above for how entry_date is finalized
on a fresh-entry bar) -- box/ceiling/floor computation itself does not
depend on that assumption (cached per calendar date, content-derived).
(5) trailing_stop
is honored as a real branch: only "prior_box_floor" (the given value)
activates the floor-based stop leg; any other value would leave only the
max_hold_days time-based exit active. (6) Scoped to long entries only per
the fixed-direction engine.
"""

def make_darvas_box_breakout_volume_confirmed_signal_fn(
    box_min_days: int = 15,
    box_max_days: int = 40,
    breakout_bar_minutes: int = 5,
    min_breakout_volume_multiple: float = 2.0,
    volume_lookback_days: int = 20,
    max_hold_days: int = 20,
    trailing_stop: str = "prior_box_floor",
) -> SignalFn:
    """Darvas Box Breakout (Volume-Confirmed), LONG ENTRIES ONLY (disclosed scope decision --
    the mechanism describes no explicit short mirror, but per the fixed-direction backtest
    engine every swing candidate in this pipeline is scoped long-only regardless).

    Box construction (computed once per new calendar day, from COMPLETED trading days only --
    i.e. days strictly before the current bar's own date -- since intraday price action on the
    current day cannot retroactively change a box that was already fully formed):
      - Resample `frame` to daily OHLC via groupby(frame.index.date).
      - ceiling = the highest daily High within the trailing `box_max_days` completed days.
        Disclosed approximation: because any day's Close can never exceed that same day's own
        High, defining the ceiling as the window's max daily High automatically guarantees no
        day in the window can have closed above it -- this operationalizes Darvas' "failed to
        close above the ceiling for N days" requirement directly via a rolling-window-max
        definition, rather than a separate day-by-day invalidation scan for a fixed peak level.
      - box_age = number of trading days from the peak day (the day the window max High was
        set; ties resolve to the EARLIEST such day, maximizing box_age) through the most recent
        completed day, inclusive. Because the peak is always the argmax within a window whose
        length is capped at `box_max_days`, box_age is structurally bounded by box_max_days --
        once the true peak ages out of the window, a new (possibly lower) peak takes over and
        the box re-forms. Box is only "mature" (tradeable) once box_age >= box_min_days.
      - floor = the lowest daily Low from the peak day through the most recent completed day
        (the stable base beneath the consolidation).

    Breakout confirmation (evaluated only on the CURRENT, in-progress trading day, since by
    construction no completed day's close can already be above the ceiling): frame is walked
    bar-by-bar; every `breakout_bar_minutes`-th bar of the day marks the close of one intraday
    bucket (e.g. 09:30-09:34, 09:35-09:39, ... for the default 5-minute bucket, cleanly aligned
    since RTH's 390 minutes divide evenly). For each such completed bucket: if its Close is
    above the current mature box's ceiling AND its total bucket Volume is at least
    `min_breakout_volume_multiple` times the average bucket volume for that IDENTICAL
    time-of-day slot across the trailing `volume_lookback_days` completed trading days (however
    many of those are actually available -- no separate minimum sample floor is enforced beyond
    box_min_days already having elapsed, a disclosed relaxation so entries are not blocked
    early in a freshly-matured box's life purely for lacking a full volume_lookback_days
    history), that bucket is the confirmed breakout bar and long entry fires on it (the caller's
    engine fills at the NEXT bar's open). A bucket that closes above the ceiling without volume
    confirmation is simply skipped -- later buckets (same day or subsequent days, as long as the
    box stays mature) keep getting evaluated fresh until one confirms or the setup expires.

    Hold / exit: this is not evaluated intraday-close-only in the mechanism's sense of a fixed
    swing hold, so exits are checked every bar for responsiveness. A trailing stop is
    maintained as CLOSURE STATE (entry date + a floor level that only ever ratchets UP, never
    loosens) initialized to the box floor at entry and updated to the newest daily box floor
    once per day thereafter, active only when trailing_stop == "prior_box_floor" (the parameter
    is honored -- a different value disables the floor-stop leg entirely, leaving only the
    time-based exit). Exit fires (same-bar close) when the last Close trades at or below the
    current trailing floor, or when `max_hold_days` trading days (inclusive of the entry day)
    have elapsed, whichever comes first.

    IMPORTANT ASSUMPTION: because multi-day holds and a ratcheting trailing stop are inherently
    path-dependent and cannot be reconstructed from a single memoryless (frame, i) call alone,
    this closure keeps internal position state (in_position / entry_date / trailing floor)
    across calls. This requires the caller to invoke signal_fn once per bar, in strictly
    ascending chronological order, for one continuous single-ticker backtest walk -- the
    standard calling convention for a sequential bar-by-bar backtest engine. The box/ceiling/
    floor computation itself does NOT rely on this assumption (it is cached per calendar date,
    content-derived, safe under any call pattern); only the trade-state machine does.

    ENGINE DOUBLE-CALL ON THE ENTRY BAR: `run_intraday_backtest` calls this function TWICE on
    a bar where entry fires -- once with the frame ending at bar i-1 (this closure's flat/entry
    branch, matching next-open execution: the decision is made without seeing bar i), then
    immediately, same loop iteration, once more with the frame ending at bar i itself (the
    exit-check for the position the engine just opened at bar i's own open). `entry_date` is
    therefore deliberately left as sentinel `None` when `entry=True` is returned, and is
    finalized on that SAME-iteration confirming exit-check call instead -- which always sees
    the real entry bar's own date, whether or not it differs from the entry-decision call's
    date (it can differ when the confirming breakout bucket is a trading day's very last
    bucket, putting bar i on the next calendar day). Finalizing at entry-decision time instead
    would silently stamp the wrong day in that edge case and overcount `days_held` by one from
    the first exit-check call onward. `floor_at_entry`, unlike `entry_date`, is NOT deferred --
    it is seeded directly in the entry branch from the box that was actually broken (see that
    branch's own comment for why deferring it would be wrong).
    """
    state = {
        "cache_date": None,
        "completed_dates": None,
        "box": None,
        "in_position": False,
        "entry_date": None,
        "floor_at_entry": None,
    }

    def _compute_box(d_high, d_low, completed_dates):
        if len(completed_dates) < box_min_days:
            return None
        window_dates = completed_dates[-box_max_days:]
        window_high = d_high.loc[window_dates]
        peak_pos = int(np.argmax(window_high.to_numpy()))
        ceiling = float(window_high.iloc[peak_pos])
        box_dates = window_dates[peak_pos:]
        box_age = len(box_dates)
        if box_age < box_min_days:
            return None
        floor = float(d_low.loc[box_dates].min())
        return {"ceiling": ceiling, "floor": floor, "box_age": box_age}

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if frame is None or len(frame) == 0:
            return SignalResult()

        idx = frame.index
        today = idx[-1].date()

        if state["cache_date"] != today:
            # Bounded to a recent tail before the O(len(frame)) groupby
            # -- the box only ever needs the trailing box_max_days
            # completed days, and days_held (below) only ever needs
            # completed_dates back to entry_date, which itself can
            # never be older than max_hold_days days (the exit logic
            # enforces that cap). This call site runs once per day;
            # converting the whole frame each time would be quadratic
            # across a long backtest.
            # volume_lookback_days is included in the bound too:
            # completed_dates below
            # also feeds lookback_dates = completed_dates[-volume_lookback_days:]
            # in the entry-evaluation branch further down, and without
            # this term the bound would silently truncate that window
            # for any candidate parameterization where
            # volume_lookback_days exceeds max(box_max_days,
            # max_hold_days) + 9 (e.g. box_max_days=10,
            # max_hold_days=5, volume_lookback_days=60 would cut
            # lookback_dates from a 60-day span to 20 days).
            recent_for_box = frame.tail(
                (max(box_max_days, max_hold_days, volume_lookback_days) + 10) * 400
            )
            d_high = recent_for_box["High"].groupby(recent_for_box.index.date).max()
            d_low = recent_for_box["Low"].groupby(recent_for_box.index.date).min()
            all_dates = d_high.index
            completed_dates = all_dates[all_dates < today]
            state["cache_date"] = today
            state["completed_dates"] = completed_dates
            state["box"] = _compute_box(d_high, d_low, completed_dates)

        completed_dates = state["completed_dates"]
        box = state["box"]

        # ---- currently in a position: evaluate exit only ----
        if state["in_position"]:
            # Finalize entry_date using THIS call's own `today` -- see the
            # "ENGINE DOUBLE-CALL ON THE ENTRY BAR" docstring section
            # above. This is the first "in position" call the closure sees
            # after entry (same loop iteration as the entry decision), so
            # `today` here is always the real entry bar's own date.
            # (floor_at_entry is NOT deferred the same way -- it is seeded
            # directly in the entry branch below, from the box that was
            # actually broken; see that branch's comment for why.)
            if state["entry_date"] is None:
                state["entry_date"] = today
            entry_date = state["entry_date"]

            if box is not None and trailing_stop == "prior_box_floor":
                if state["floor_at_entry"] is None or box["floor"] > state["floor_at_entry"]:
                    state["floor_at_entry"] = box["floor"]

            days_held = int((completed_dates >= entry_date).sum()) + 1
            current_floor = state["floor_at_entry"]
            last_close = float(frame["Close"].iloc[-1])
            stop_hit = current_floor is not None and last_close <= current_floor
            time_hit = days_held >= max_hold_days

            if stop_hit or time_hit:
                state["in_position"] = False
                state["entry_date"] = None
                state["floor_at_entry"] = None
                return SignalResult(exit=True)
            return SignalResult()

        # ---- flat: evaluate entry (confirmed breakout) only ----
        if box is None:
            return SignalResult()

        tz = idx.tz
        midnight_today = pd.Timestamp(today, tz=tz) if tz is not None else pd.Timestamp(today)
        day_start_pos = int(idx.searchsorted(midnight_today))
        today_bar_count = len(idx) - day_start_pos

        if today_bar_count < breakout_bar_minutes or today_bar_count % breakout_bar_minutes != 0:
            return SignalResult()

        bucket = frame.iloc[-breakout_bar_minutes:]
        ceiling = box["ceiling"]
        bucket_close = float(bucket["Close"].iloc[-1])
        if bucket_close <= ceiling:
            return SignalResult()

        bucket_start_ts = bucket.index[0]
        bucket_start_minute = bucket_start_ts.hour * 60 + bucket_start_ts.minute
        bucket_end_minute = bucket_start_minute + breakout_bar_minutes
        bucket_volume = float(bucket["Volume"].sum())

        lookback_dates = completed_dates[-volume_lookback_days:]
        if len(lookback_dates) == 0:
            return SignalResult()

        # Bounded to a recent tail before the O(len(frame)) `.date`/
        # `.hour`/`.minute` conversions -- only volume_lookback_days
        # completed days (plus today) are ever needed for the
        # historical bucket-volume comparison, never the full
        # multi-year frame. This call site runs on every
        # breakout_bar_minutes-minute bucket while price trades above
        # the box ceiling, so converting the whole frame here would be
        # quadratic across a long backtest.
        recent_for_volume = frame.tail((volume_lookback_days + 10) * 400)
        date_arr = recent_for_volume.index.date
        minute_of_day = recent_for_volume.index.hour * 60 + recent_for_volume.index.minute
        in_lookback = np.isin(date_arr, lookback_dates.to_numpy())
        in_bucket_time = (minute_of_day >= bucket_start_minute) & (minute_of_day < bucket_end_minute)
        mask = in_lookback & in_bucket_time
        if not mask.any():
            return SignalResult()

        hist_bucket_vol = recent_for_volume.loc[mask, "Volume"].groupby(date_arr[mask]).sum()
        if hist_bucket_vol.empty:
            return SignalResult()

        avg_volume = float(hist_bucket_vol.mean())
        if avg_volume <= 0 or bucket_volume < min_breakout_volume_multiple * avg_volume:
            return SignalResult()

        # entry_date deliberately left as sentinel None -- finalized on the
        # same-iteration confirming exit-check call. See the "ENGINE
        # DOUBLE-CALL ON THE ENTRY BAR" docstring section.
        #
        # floor_at_entry, in contrast, is seeded HERE, from THIS call's own
        # `box` -- the box that was actually broken (computed from
        # completed_dates strictly before today, i.e. never including the
        # in-progress breakout day itself). This deliberately does NOT
        # defer to the confirming exit-check call: by the time that call
        # runs, the breakout day has itself become a completed day, and
        # since a confirmed breakout's bucket_close is by construction
        # above the OLD ceiling (the window's previous max daily High),
        # the breakout day's own daily High is now the window's new max --
        # `_compute_box` would treat the breakout day itself as a fresh
        # peak, resetting box_age to a small number and returning None for
        # roughly the next box_min_days trading days on almost every real
        # breakout. Seeding from the deferred (post-breakout) box would
        # leave floor_at_entry stuck at None -- the stop-loss leg silently
        # never activating -- for most of a typical trade's early hold.
        # Seeding from THIS call's pre-breakout box avoids that: the stop
        # starts at the floor of the box that was actually broken, and the
        # ratchet below (which SKIPS an update whenever box is temporarily
        # None, never lowering) picks back up once a mature box reappears.
        state["in_position"] = True
        state["entry_date"] = None
        state["floor_at_entry"] = box["floor"] if trailing_stop == "prior_box_floor" else None
        return SignalResult(entry=True)

    return signal_fn


# --- first_hour_range_breakout_swing ---
def make_first_hour_range_breakout_swing_signal_fn(
    range_window_minutes: int = 60,
    range_start_et: str = "09:30",
    range_end_et: str = "10:30",
    confirmation_bar_minutes: int = 5,
    max_hold_days: int = 10,
    stop_type: str = "first_hour_range_midpoint",
    direction: str = "long_or_short",
):
    """Implements the First-Hour Range Breakout (Swing Hold) mechanism as a LONG-ONLY
    signal. This deliberately scopes out the mechanism's mirrored short
    variant (breakout below the first-hour low) even though the candidate's own
    `direction` parameter reads "long_or_short" -- the `direction` value is still
    validated (must be one of "long", "short", "long_or_short") but is never branched on,
    since this backtest engine's `direction` is a single whole-call parameter
    (`run_intraday_backtest`'s own `direction: Literal["long", "short"]`, applied to
    an entire backtest run rather than per-trade) and this pipeline scopes every swing
    candidate to its long side only. Each real RTH day, the first `range_window_minutes`
    minutes of the session starting at `range_start_et` are used to compute the first-hour
    range (high/low/midpoint) directly from the 1-minute bars actually present in `frame`
    -- the high/low of a window of 1-minute bars is mathematically identical to the
    high/low computed from 5-minute bars over the same window, so no approximation is
    needed for the range itself. Once the window closes (`range_end_et`), later bars are
    checked for a breakout CLOSE above the range high, but only on bars that represent a
    completed `confirmation_bar_minutes`-minute bar close -- approximated from the
    1-minute feed as bars whose elapsed minutes since `range_start_et` are congruent to
    `confirmation_bar_minutes - 1` modulo `confirmation_bar_minutes` (e.g. bars ending
    :34, :39, :44 for the default 5-minute window starting 09:30 -- correct under this
    engine's own bar-timestamp convention, where a 1-minute bar's timestamp marks the
    START of its interval, so the LAST 1-minute bar of each 5-minute window carries the
    same Close a resampled 5-minute bar would). Because a resampled 5-minute bar's close
    equals its last constituent 1-minute bar's close, this reproduces the true 5-minute
    confirmation close exactly, provided there is no missing 1-minute bar exactly at that
    alignment point (an extremely rare condition for SPY/IWM/DIA during RTH); if one is
    missing, that single confirmation check is silently skipped rather than mis-evaluated,
    which is the conservative failure mode. On a qualifying breakout the position enters
    long with a stop fixed at that day's first-hour range midpoint (checked each
    subsequent bar via that bar's Low touching or breaching the level) and a maximum hold
    measured in distinct trading days actually observed in the bar stream (skipping
    weekends/holidays for free, since non-trading days never appear in the feed) -- though
    in practice `run_intraday_backtest`'s OWN `max_hold_days` force-close (using the same
    value, since the orchestrator always passes the candidate's own `max_hold_days` to
    both the factory and the engine) fires first, at the exact last bar of trading day N;
    this signal's own day-count exit is a harmless backstop that in practice is never
    reached before the engine's own force-close.

    This signal is inherently stateful (a closure-held in_position/stop-level/day-count)
    because the mechanism requires a multi-day hold, which is only representable across
    repeated `signal_fn(frame, i)` calls if state persists between them. This signal
    does NOT assume a fresh factory
    instance per ticker/run -- the real orchestrator (`run_swing_mining.py`) reuses ONE
    `candidate["signal_fn"]` object across every ticker in a candidate's pooled backtest
    (`_backtest_tickers`'s own `for ticker in tickers: run_intraday_backtest(bars,
    signal_fn, ...)` loop) and again across the in-sample and holdout phases -- and the
    engine's own end-of-data force-close (`if position is not None: _close_trade(n - 1,
    "end_of_data")`) never calls `signal_fn` again to tell it the position closed, so a
    naive stateful implementation would silently carry a stale open position (at a
    completely different ticker's price scale) into the next ticker's evaluation,
    corrupting or entirely suppressing its trades (it would stay stuck evaluating
    stop/time-stop conditions against the first ticker's stale position instead of ever
    reaching the entry-evaluation branch, so the second ticker's own genuine breakout
    would produce no trade). So all closure state is reset
    whenever `i == 0`: the engine's own loop always skips calling
    `signal_fn` at its own index 0 (`if position is None: if i == 0: continue`), so the
    first-ever call in any fresh `run_intraday_backtest` invocation is always the entry-
    check call `signal_fn(df.iloc[:1], 0)` -- i.e. `i == 0` is passed exactly once per
    run, always as the very first call, making it a reliable "a new run has begun" marker
    with no dependence on which ticker or which phase (in-sample vs. holdout) is running.
    This assumes only that the engine calls signal_fn once per bar in increasing `i` order
    within a single continuous backtest pass -- true for every real call site in this
    repo -- and does not assume one factory instance per ticker/run.
    """
    if stop_type != "first_hour_range_midpoint":
        raise ValueError(
            f"Unsupported stop_type: {stop_type!r}; only 'first_hour_range_midpoint' "
            "is implemented by this signal."
        )
    if direction not in ("long", "short", "long_or_short"):
        raise ValueError(f"Unsupported direction: {direction!r}")

    _range_start_time = dt.datetime.strptime(range_start_et, "%H:%M").time()
    _range_end_time = dt.datetime.strptime(range_end_et, "%H:%M").time()
    _range_start_minutes = _range_start_time.hour * 60 + _range_start_time.minute

    _state = {
        "in_position": False,
        "stop_level": None,
        "last_seen_date": None,
        "trading_days_in_trade": 0,
    }

    _TAIL_ROWS = 500  # comfortably more than one RTH session's ~390 one-minute bars

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        # A fresh backtest run (a new ticker, or a new phase such as
        # in-sample -> holdout) always makes its first-ever call to this
        # function with i == 0 -- see the factory docstring's stateful-
        # reuse paragraph for why this reset is required, not optional.
        if i == 0:
            _state["in_position"] = False
            _state["stop_level"] = None
            _state["last_seen_date"] = None
            _state["trading_days_in_trade"] = 0

        if frame.empty:
            return SignalResult()

        bar = frame.iloc[-1]
        t = frame.index[-1]
        today = t.date()

        # ---- exit management for an existing position ----------------------
        if _state["in_position"]:
            if _state["last_seen_date"] != today:
                _state["trading_days_in_trade"] += 1
                _state["last_seen_date"] = today

            stop_hit = bool(bar["Low"] <= _state["stop_level"])
            time_hit = _state["trading_days_in_trade"] > max_hold_days

            if stop_hit or time_hit:
                _state["in_position"] = False
                _state["stop_level"] = None
                _state["trading_days_in_trade"] = 0
                return SignalResult(exit=True)

            return SignalResult()

        # ---- entry evaluation -----------------------------------------------
        if t.time() < _range_end_time:
            return SignalResult()

        elapsed_now = (t.hour * 60 + t.minute) - _range_start_minutes
        if elapsed_now < 0:
            return SignalResult()
        is_confirmation_close = (
            elapsed_now % confirmation_bar_minutes
        ) == (confirmation_bar_minutes - 1)
        if not is_confirmation_close:
            return SignalResult()

        recent = frame.iloc[-_TAIL_ROWS:] if len(frame) > _TAIL_ROWS else frame
        day_bars = recent[recent.index.date == today]
        if day_bars.empty:
            return SignalResult()
        if day_bars.index[0].time() > _range_start_time:
            # Today's data does not actually start at the configured range open
            # (e.g. the fetch window began mid-session) -- the range cannot be
            # trusted, so skip evaluating a breakout for this day entirely.
            return SignalResult()

        day_elapsed = (
            day_bars.index.hour * 60 + day_bars.index.minute
        ) - _range_start_minutes
        in_range_window = (day_elapsed >= 0) & (day_elapsed < range_window_minutes)
        range_bars = day_bars[in_range_window]
        if range_bars.empty:
            return SignalResult()

        range_high = range_bars["High"].max()
        range_low = range_bars["Low"].min()
        range_mid = (range_high + range_low) / 2.0

        if bar["Close"] > range_high:
            _state["in_position"] = True
            _state["stop_level"] = range_mid
            _state["last_seen_date"] = today
            _state["trading_days_in_trade"] = 1
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- intraday_liquidity_sweep_reversal ---
def make_intraday_liquidity_sweep_reversal_signal_fn(
    sweep_window_start_et: str = "09:30",
    sweep_window_end_et: str = "11:00",
    reclaim_confirm_by_et: str = "12:00",
    swing_low_lookback_sessions: int = 20,
    swing_low_min_age_sessions: int = 4,
    reclaim_volume_multiple: float = 1.5,
    max_hold_days: int = 20,
    bar_granularity_minutes: int = 5,
    volume_avg_lookback_bars: int = 20,
    atr_period: int = 14,
    atr_trailing_multiple: float = 2.5,
) -> SignalFn:
    """Intraday Liquidity-Sweep Failed-Breakdown Reversal (long-only, per the repo-wide
    single-direction engine convention -- the mechanism itself only describes a long
    reversal, so nothing is being omitted by scoping to long).

    `frame` is real 1-minute RTH bars, so this closure resamples it internally into
    5-minute bars (bar_granularity_minutes) and into daily bars, each time using only
    data already present in `frame` up to and including bar i -- never anything past it.

    Each real trading day: watch for a 5-minute bar (09:30-11:00 ET, sweep_window_*)
    whose Low trades below the "prior 20-session swing low" -- the lowest daily Low over
    the swing_low_lookback_sessions sessions before today, restricted to sessions that are
    themselves at least swing_low_min_age_sessions old (i.e. the most recent
    swing_low_min_age_sessions-1 sessions are excluded from forming the level, since a
    level that fresh would not yet be a recognized swing point). Once swept, watch
    subsequent 5-minute bars (up to but not including reclaim_confirm_by_et) for one whose
    Close reclaims back above the swept level on volume >= reclaim_volume_multiple times
    the trailing volume_avg_lookback_bars 5-minute-bar average volume (computed from the
    bars strictly before the candidate bar). The first such bar fires entry=True. The
    mechanism text says to "enter long at the close of that confirming reclaim bar (not at
    the day's close)" -- this closure fires entry=True on that bar; the engine's own
    uniform next-bar-open execution convention (stated in the calling contract, applied to
    every candidate in this pipeline) is what actually times the fill, not a same-bar-close
    fill -- a disclosed consequence of the shared execution model, not a choice made here.

    Only one position is tracked at a time: while in a position, no new sweep/reclaim
    entries are evaluated. Exit uses two rules, checked every bar (finer than the entry
    granularity, for responsiveness) once in a position: (1) a real-data trailing stop --
    the mechanism only says "a trailing stop" with no percentage/distance given, so this
    approximates it with a standard chandelier-style stop: highest Close since entry minus
    atr_trailing_multiple * ATR(atr_period, computed on DAILY bars strictly before the
    entry day, simple-moving-average TR rather than Wilder smoothing -- a common, faithful
    simplification). If there is not yet enough real daily history to compute that ATR
    (very early in the fetched history), the trailing-stop leg is simply skipped for that
    trade (disclosed -- never backfilled with an invented distance) and only rule (2)
    applies: a hard time-stop once the position has been held through max_hold_days
    distinct real trading days (day of entry counts as day 0).

    Time-stop / engine-synchronization note: `run_intraday_backtest` (the calling engine)
    ALSO independently force-closes any position that is still open at the LAST bar of its
    `max_hold_days`-th trading day (entry day counted as day 1 in the engine's own
    convention), regardless of what this closure returns -- and the engine gives no
    callback telling a signal_fn that it did so. Because the engine re-issues an
    IDENTICAL (frame, i) call immediately after any close (once as the exit check on an
    open position, then again -- same bar, same data -- as the very next entry check now
    that the engine's position is None), this closure's own internal `in_position` state
    MUST flip to False no later than the bar the engine force-closes on, or it silently
    stops evaluating new sweep/reclaim entries until it catches up. A naive day-transition
    counter that only increments hold_days_elapsed on the FIRST bar of each new day (and
    fires at `hold_days_elapsed >= max_hold_days`) reaches that threshold one full
    session AFTER the engine's own boundary (the engine closes at day
    entry+`max_hold_days`-1's LAST bar; the naive counter would not reach `max_hold_days`
    until day entry+`max_hold_days`'s FIRST bar) -- a state-desync that
    would silently suppress entries for about a session after every max-hold-days exit.
    So this closure fires its internal time-stop one session earlier
    (`hold_days_elapsed >= max_hold_days - 1`, i.e. at the FIRST bar of the
    `max_hold_days`-th session rather than waiting for a session that never lags behind
    it), so its own exit always fires at or before the engine's, keeping
    `state["in_position"]` synchronized with the engine's real position on every call.
    This holds the position for slightly less than a full `max_hold_days`-th session
    (closes on its first bar instead of its last), which is still faithful to the
    mechanism's "hold for up to max_hold_days" framing.

    Approximations disclosed: (a) 5-minute bars/daily bars are built by resampling the
    1-minute frame inside this closure each call (bounded lookback windows are sliced
    first for cost, not correctness); (b) the swing-low level and the ATR are computed
    once per relevant day/entry and held fixed rather than recomputed every bar; (c) a
    minor real-world data-gap risk: if the underlying minute bars are not perfectly
    contiguous within a session, the 5-minute-bar-close detection (based on clock-minute
    offset from 09:30) could misalign for that one bar.
    """

    import datetime

    sweep_start_time = datetime.datetime.strptime(sweep_window_start_et, "%H:%M").time()
    sweep_end_time = datetime.datetime.strptime(sweep_window_end_et, "%H:%M").time()
    reclaim_cutoff_time = datetime.datetime.strptime(reclaim_confirm_by_et, "%H:%M").time()

    _SWING_LOW_BUFFER_DAYS = 45
    _VOLUME_BAR_BUFFER_DAYS = 10

    state = {
        "current_day": None,
        "day_swept": False,
        "sweep_level": None,
        "swing_low_level_today": None,
        "swing_low_computed_day": None,
        "in_position": False,
        "entry_close": None,
        "entry_day": None,
        "hold_days_elapsed": 0,
        "highest_close_since_entry": None,
        "trade_atr": None,
    }

    def _daily_bars(frame_local, min_date):
        if frame_local.empty:
            return None
        # Bounded to a recent tail before the O(len(frame_local)) `.date`
        # conversion -- callers only ever want days >= min_date, so
        # bound the input to comfortably cover that span (computed from
        # min_date itself, so this stays correct regardless of which
        # caller's own buffer produced it) rather than scanning the
        # full multi-year frame.
        latest_date = frame_local.index[-1].date()
        buffer_calendar_days = max((latest_date - min_date).days, 0) + 20
        frame_local = frame_local.tail(buffer_calendar_days * 400)
        idx_dates = frame_local.index.date
        mask = idx_dates >= min_date
        sl = frame_local.loc[mask]
        if sl.empty:
            return None
        g = sl.groupby(sl.index.date)
        daily = g.agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        return daily

    def _compute_swing_low_level(frame_local, today, lookback_sessions, min_age_sessions, buffer_days):
        min_date = today - datetime.timedelta(days=buffer_days)
        daily = _daily_bars(frame_local, min_date)
        if daily is None:
            return None
        prior_days = [d for d in daily.index if d < today]
        if len(prior_days) < lookback_sessions:
            return None
        last_lookback_days = prior_days[-lookback_sessions:]
        exclude_count = max(0, min_age_sessions - 1)
        if exclude_count >= len(last_lookback_days):
            return None
        eligible_days = (
            last_lookback_days[: len(last_lookback_days) - exclude_count]
            if exclude_count > 0
            else last_lookback_days
        )
        if not eligible_days:
            return None
        level = daily.loc[eligible_days, "Low"].min()
        if pd.isna(level):
            return None
        return float(level)

    def _compute_atr(daily_df, period):
        if daily_df is None or len(daily_df) < period + 1:
            return None
        prev_close = daily_df["Close"].shift(1)
        tr1 = daily_df["High"] - daily_df["Low"]
        tr2 = (daily_df["High"] - prev_close).abs()
        tr3 = (daily_df["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()
        if len(tr) < period:
            return None
        atr_val = tr.iloc[-period:].mean()
        if pd.isna(atr_val) or atr_val <= 0:
            return None
        return float(atr_val)

    def _resample_intraday_bars(frame_local, today, granularity_minutes, buffer_days):
        min_date = today - datetime.timedelta(days=buffer_days)
        # Bounded to a recent tail before the O(len(frame_local)) `.date`
        # conversion -- only buffer_days calendar days (comfortably
        # covering the trading days actually needed) are ever required,
        # never the full multi-year frame. This call site runs on every
        # bar_granularity_minutes-minute bar once a day has swept (up to
        # ~30x/day per the docstring's own sweep-window description),
        # so converting the whole frame here would be quadratic across
        # a long backtest.
        if frame_local.empty:
            return None
        frame_local = frame_local.tail((buffer_days + 20) * 400)
        idx_dates = frame_local.index.date
        mask = (idx_dates >= min_date) & (idx_dates <= today)
        sl = frame_local.loc[mask]
        if sl.empty:
            return None
        freq = "{}min".format(int(granularity_minutes))
        bars = sl.resample(freq, label="left", closed="left").agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        bars = bars.dropna(subset=["Open", "Close"])
        if bars.empty:
            return None
        return bars

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if frame is None or len(frame) == 0:
            return SignalResult()

        ts = frame.index[-1]
        today = ts.date()
        row = frame.iloc[-1]
        close = float(row["Close"])

        if state["current_day"] != today:
            state["current_day"] = today
            state["day_swept"] = False
            state["sweep_level"] = None
            if state["in_position"]:
                state["hold_days_elapsed"] += 1

        # ---- Exit management: checked every bar while a position is open ----
        if state["in_position"]:
            if state["highest_close_since_entry"] is None or close > state["highest_close_since_entry"]:
                state["highest_close_since_entry"] = close

            hit_trailing_stop = False
            if state["trade_atr"] is not None:
                trail_stop_price = state["highest_close_since_entry"] - state["trade_atr"] * atr_trailing_multiple
                hit_trailing_stop = close <= trail_stop_price

            # `max_hold_days - 1` (NOT max_hold_days) is deliberate: the engine
            # (`run_intraday_backtest`) independently force-closes a position at the
            # LAST bar of its max_hold_days-th trading day (entry day = day 1) and
            # gives this closure no notice when it does. `hold_days_elapsed` only
            # advances on the FIRST bar of each new day, so a naive `>= max_hold_days`
            # check would not fire until the FIRST bar of the (max_hold_days+1)-th
            # day -- one full session after the engine's own boundary -- leaving
            # `state["in_position"]` stuck True across that gap and silently
            # suppressing new entries until it caught up (see
            # this function's docstring). Checking `>= max_hold_days - 1` instead
            # fires on the FIRST bar of the max_hold_days-th day, which is always at
            # or before the engine's own LAST-bar boundary on that same day, so this
            # closure's own state transition can never lag the engine's.
            hit_max_hold = state["hold_days_elapsed"] >= max_hold_days - 1

            if hit_trailing_stop or hit_max_hold:
                state["in_position"] = False
                state["entry_close"] = None
                state["entry_day"] = None
                state["hold_days_elapsed"] = 0
                state["highest_close_since_entry"] = None
                state["trade_atr"] = None
                return SignalResult(exit=True)

            return SignalResult()

        # ---- Entry management: only evaluated at 5-minute-bar closes, before the reclaim cutoff ----
        t = ts.time()
        if t >= reclaim_cutoff_time:
            return SignalResult()

        session_open_minute = 9 * 60 + 30
        minute_of_day = ts.hour * 60 + ts.minute
        minutes_since_open = minute_of_day - session_open_minute
        if minutes_since_open < 0:
            return SignalResult()

        is_bar_close = (minutes_since_open % bar_granularity_minutes) == (bar_granularity_minutes - 1)
        if not is_bar_close:
            return SignalResult()

        if state["swing_low_computed_day"] != today:
            state["swing_low_level_today"] = _compute_swing_low_level(
                frame, today, swing_low_lookback_sessions, swing_low_min_age_sessions,
                _SWING_LOW_BUFFER_DAYS,
            )
            state["swing_low_computed_day"] = today

        swing_low_level = state["swing_low_level_today"]
        if swing_low_level is None:
            return SignalResult()

        bars5 = _resample_intraday_bars(frame, today, bar_granularity_minutes, _VOLUME_BAR_BUFFER_DAYS)
        if bars5 is None or len(bars5) == 0:
            return SignalResult()

        current_bar = bars5.iloc[-1]

        if not state["day_swept"]:
            if sweep_start_time <= t <= sweep_end_time:
                if float(current_bar["Low"]) < swing_low_level:
                    state["day_swept"] = True
                    state["sweep_level"] = swing_low_level
            return SignalResult()

        if len(bars5) < volume_avg_lookback_bars + 1:
            return SignalResult()

        prior_bars = bars5.iloc[-(volume_avg_lookback_bars + 1):-1]
        avg_vol = prior_bars["Volume"].mean()
        if pd.isna(avg_vol) or avg_vol <= 0:
            return SignalResult()

        current_close = float(current_bar["Close"])
        current_vol = float(current_bar["Volume"])

        if current_close > state["sweep_level"] and current_vol >= reclaim_volume_multiple * avg_vol:
            atr_min_date = today - datetime.timedelta(days=int((atr_period + 5) * 1.7) + 10)
            daily_for_atr = _daily_bars(frame, atr_min_date)
            entry_atr = None
            if daily_for_atr is not None:
                prior_for_atr_days = [d for d in daily_for_atr.index if d < today]
                if prior_for_atr_days:
                    prior_for_atr = daily_for_atr.loc[prior_for_atr_days]
                    entry_atr = _compute_atr(prior_for_atr, atr_period)

            state["in_position"] = True
            state["entry_close"] = current_close
            state["entry_day"] = today
            state["hold_days_elapsed"] = 0
            state["highest_close_since_entry"] = current_close
            state["trade_atr"] = entry_atr
            state["day_swept"] = False
            state["sweep_level"] = None

            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- intraday_capitulation_reversal_bar ---
def make_intraday_capitulation_reversal_bar_signal_fn(
    bar_granularity_minutes: int = 5,
    decline_threshold_pct: float = 5.0,
    decline_measure_cutoff_et: str = "14:00",
    reversal_bar_volume_multiple: float = 2.0,
    reversal_bar_close_position: str = "top_third",
    max_hold_days: int = 20,
) -> SignalFn:
    """
    Implements the Intraday Capitulation Reversal-Bar Entry mechanism, LONG SIDE ONLY
    (per this pipeline's stated scoping rule: the backtest engine's `direction` is a
    single whole-call parameter, so this pipeline deliberately covers only the
    long entry the mechanism describes -- no short mirror is implemented; the
    mechanism as given does not describe one anyway, so this is a disclosed scope
    decision rather than a dropped requirement).

    Mechanism: synthesize 5-minute bars from the underlying 1-minute Alpaca RTH
    `frame` (a fixed grid anchored to the 09:30 ET session open, i.e. bars
    [09:30,09:35), [09:35,09:40), ... -- NOT a rolling or exchange-published
    5-minute series, since only real 1-minute bars are available). At each
    completed synthetic 5-minute bar, this tracks the percentage move of that
    bar's close versus the PRIOR trading day's last available RTH minute close
    (used as the real-data proxy for "prior session's close", since no separate
    official daily-close field is passed to this function -- `frame` is
    guaranteed gapless across every RTH minute of every trading day, so the row
    immediately preceding today's first row is genuinely the prior session's
    final print). The first time that decline reaches at least
    `decline_threshold_pct` at or before `decline_measure_cutoff_et` ET on a
    given day, this flags a "capitulation" event and waits for exactly the NEXT
    completed 5-minute bar (only that one bar is ever checked, matching "the
    very next 5-minute bar" in the mechanism -- if it fails to qualify, this
    capitulation event produces no trade and the day is done being watched).
    That next bar qualifies as the reversal bar, and entry fires, when its close
    sits in the top fraction of its own high-low range implied by
    `reversal_bar_close_position` ("top_third" -> close position >= 2/3 of the
    bar's range; "top_half" -> >=1/2; "top_quarter" -> >=3/4; any other value
    falls back to the top-third convention, matching the candidate's own default)
    AND its volume is at least `reversal_bar_volume_multiple` times the trailing
    20-completed-5-minute-bar average volume (a volume history that rolls
    continuously across day boundaries once at least 20 real 5-minute bars have
    been observed; before that this signal never fires, per the
    no-crash-on-insufficient-history rule -- it returns a no-op SignalResult
    instead). Entry decisions are only evaluated while this signal considers
    itself flat -- once fired, further capitulation/reversal detection is paused
    until the position is closed, so this function never emits overlapping
    entries regardless of how the calling engine itself tracks position state.

    Exit / max_hold_days: a bar can never look ahead to learn whether it is the
    last bar of its own session (frame is truncated exactly at bar i), so "hold
    for up to `max_hold_days` trading days" is enforced, look-ahead-safely, by
    emitting the exit signal on the FIRST bar of the trading day that is
    `max_hold_days` full trading-day-transitions after the entry day. The engine
    executes an exit signal at that bar's own close, so this exit lands close to
    that day's session open rather than its close -- a disclosed timing
    approximation, chosen because detecting "last bar of today's own session"
    without seeing bar i+1 is structurally impossible, and this alternative
    (unlike a hardcoded 15:59-ET check) also behaves correctly on early-close
    half sessions. If the fetched `frame` ends while a position is still open,
    no further exit is emitted by this function -- there is no later bar to
    emit it from; handling a still-open position at the end of a run is left to
    the calling engine. NOTE: `run_intraday_backtest` (this project's own
    engine) separately and independently enforces `max_hold_days` itself via
    `hit_time_stop`, and (by that engine's own bar indexing) that
    engine-side force-close always fires strictly at or
    before this function's own day-transition exit check would. In the
    ordinary case (entry executes the bar immediately after the reversal bar,
    same day) the engine's own force-close wins by exactly one trading day, so
    this function's own exit path is not the one actually observed by the
    engine in that case; it remains here as the documented, structurally
    look-ahead-safe fallback for the case where the entry bar rolls into the
    NEXT day (reversal bar was that day's last bar), where this function's own
    exit and the engine's force-close land on the same day and this function's
    fires first (at that day's open rather than its close). Either way,
    `held_days` computed by the engine ends up equal to `max_hold_days`.

    Repeat-call idempotency: the engine's own two-call-site convention (see
    `intraday_backtest_engine.py`'s `SignalFn` docstring) calls this function
    TWICE with the IDENTICAL (frame, current-bar) pair whenever a position
    closes -- once via the exit-check call site while the position was still
    open, then immediately again via the entry-check call site on the very
    next outer iteration, which re-evaluates the same bar that just closed the
    position (that call site always looks one bar behind). This closure
    de-duplicates by timestamp so that repeat is a clean no-op rather than
    re-running this bar's completed-bucket processing a second time -- see the
    `last_ts_seen` guard below. Without it, the trailing 20-bucket volume
    history used by the reversal-bar volume check would get this bar's volume
    appended twice on every trade closed by the engine's own `hit_time_stop`,
    which -- since that force-close always lands on a trading day's LAST bar,
    itself always a completed-bucket boundary for a standard 390-minute
    session with a 5-minute bucket -- would happen on very nearly
    every trade this signal takes, not a rare edge case.

    State: this closure keeps its own small internal state (rolling 5-minute
    volume history, day-transition bookkeeping, and whether it currently
    considers itself in a position) across sequential calls, because SignalFn's
    pure (frame, i) signature otherwise gives this function no way to remember
    when ITS OWN prior entry happened in order to enforce max_hold_days, or to
    avoid re-scanning the entire multi-year frame on every single call just to
    rediscover that history. This assumes the engine calls signal_fn once per
    bar, in ascending i order, on ONE fresh factory instance per backtest run
    (the normal walk-forward calling pattern, one instantiation per
    candidate/ticker trial) -- if this same closure were reused across more than
    one ticker's frame, their capitulation tracking would cross-contaminate, so
    callers must instantiate a fresh factory per run.
    """
    try:
        bucket_minutes = int(bar_granularity_minutes)
        if bucket_minutes <= 0:
            bucket_minutes = 5
    except Exception:
        bucket_minutes = 5

    try:
        _cutoff = _dt.datetime.strptime(str(decline_measure_cutoff_et), "%H:%M").time()
    except Exception:
        _cutoff = _dt.time(14, 0)

    _pos_key = str(reversal_bar_close_position).strip().lower()
    if _pos_key in ("top_half", "top-half", "tophalf"):
        _close_pos_threshold = 0.5
    elif _pos_key in ("top_quarter", "top-quarter", "topquarter"):
        _close_pos_threshold = 0.75
    else:
        _close_pos_threshold = 2.0 / 3.0  # "top_third" -- candidate's own default

    try:
        _decline_pct = abs(float(decline_threshold_pct))
    except Exception:
        _decline_pct = 5.0

    try:
        _vol_mult = float(reversal_bar_volume_multiple)
    except Exception:
        _vol_mult = 2.0

    try:
        _max_hold = int(max_hold_days)
        if _max_hold <= 0:
            _max_hold = 20
    except Exception:
        _max_hold = 20

    _session_open = _dt.time(9, 30)
    _vol_hist_len = 20

    _st = {
        "last_date": None,
        "prior_close": None,
        "trading_day_count": 0,
        "decline_hit_today": False,
        "awaiting_reversal": False,
        "position_open": False,
        "entry_trading_day_count": None,
        "vol_hist": _deque(maxlen=_vol_hist_len),
        "last_ts_seen": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        try:
            if frame is None or len(frame) == 0:
                return SignalResult()

            ts = frame.index[-1]

            # --- repeat-call idempotency guard (see docstring) -------------
            # The engine calls this function twice with the identical
            # (frame, current-bar) pair whenever a position closes. Without
            # this guard the second call would re-run this bar's
            # completed-bucket processing (most consequentially,
            # double-appending its volume into `vol_hist`).
            if _st["last_ts_seen"] == ts:
                return SignalResult()
            _st["last_ts_seen"] = ts

            today = ts.date()
            bar_time = ts.time()

            # --- day-transition bookkeeping (also enforces max_hold_days exit) ---
            if _st["last_date"] is None:
                _st["last_date"] = today
                _st["trading_day_count"] = 1
                _st["prior_close"] = None
                _st["decline_hit_today"] = False
                _st["awaiting_reversal"] = False
            elif today != _st["last_date"]:
                if len(frame) >= 2:
                    _st["prior_close"] = float(frame["Close"].iloc[-2])
                else:
                    _st["prior_close"] = None
                _st["last_date"] = today
                _st["trading_day_count"] += 1
                _st["decline_hit_today"] = False
                _st["awaiting_reversal"] = False

                if _st["position_open"] and _st["entry_trading_day_count"] is not None:
                    elapsed = _st["trading_day_count"] - _st["entry_trading_day_count"]
                    if elapsed >= _max_hold:
                        _st["position_open"] = False
                        _st["entry_trading_day_count"] = None
                        return SignalResult(exit=True)

            # --- synthetic 5-minute (or bar_granularity_minutes) bucket completion ---
            minutes_since_open = (bar_time.hour * 60 + bar_time.minute) - (
                _session_open.hour * 60 + _session_open.minute
            )
            if minutes_since_open < 0 or minutes_since_open % bucket_minutes != (bucket_minutes - 1):
                return SignalResult()
            if len(frame) < bucket_minutes:
                return SignalResult()

            five = frame.iloc[-bucket_minutes:]
            five_high = float(five["High"].max())
            five_low = float(five["Low"].min())
            five_close = float(five["Close"].iloc[-1])
            five_volume = float(five["Volume"].sum())

            entry_fired = False

            if not _st["position_open"]:
                prior_close = _st["prior_close"]

                if (
                    not _st["decline_hit_today"]
                    and not _st["awaiting_reversal"]
                    and prior_close is not None
                    and prior_close > 0
                    and bar_time <= _cutoff
                ):
                    decline_pct = (five_close - prior_close) / prior_close * 100.0
                    if decline_pct <= -_decline_pct:
                        _st["decline_hit_today"] = True
                        _st["awaiting_reversal"] = True

                elif _st["awaiting_reversal"]:
                    vol_hist = _st["vol_hist"]
                    if len(vol_hist) >= _vol_hist_len:
                        avg_vol = sum(vol_hist) / len(vol_hist)
                        bar_range = five_high - five_low
                        if bar_range > 0 and avg_vol > 0:
                            close_pos = (five_close - five_low) / bar_range
                            if (
                                close_pos >= _close_pos_threshold
                                and five_volume >= _vol_mult * avg_vol
                            ):
                                _st["position_open"] = True
                                _st["entry_trading_day_count"] = _st["trading_day_count"]
                                entry_fired = True
                    _st["awaiting_reversal"] = False

            _st["vol_hist"].append(five_volume)

            if entry_fired:
                return SignalResult(entry=True)
            return SignalResult()
        except Exception:
            return SignalResult()

    return signal_fn


# --- earnings_reaction_settle_drift ---
def make_earnings_reaction_settle_drift_signal_fn(
    earnings_events: list[dict],
    bar_granularity_minutes: int = 5,
    range_compression_multiple: float = 1.3,
    settle_window_start_et: str = "09:30",
    max_settle_wait_minutes: int = 60,
    max_hold_days: int = 20,
    rolling_lookback_bars: int = 20,
):
    """Post-Earnings Reaction-Settle Drift Entry (LONG side only).

    On a real historical earnings-reaction day for this ticker (resolved from
    `earnings_events`), tracks 5-minute bars built from the underlying 1-minute
    `frame` starting at `settle_window_start_et`. The first COMPLETE 5-minute
    bar (within `max_settle_wait_minutes` of the session open) whose realized
    range (High-Low) compresses to within `range_compression_multiple` times a
    cached baseline range triggers a LONG entry at that bar's close, held for
    up to `max_hold_days` trading days.

    Disclosed scope / approximation decisions:
      1. LONG ONLY. The mechanism describes a mirrored short entry on a
         negative-surprise reaction; per this pipeline's single-direction scoping, only
         the long side is implemented. Negative-surprise reaction days are
         recognized (tracked in `reaction_days`) but never traded -- no entry
         ever fires on them.
      2. "Its own trailing 20-bar rolling average" (the range baseline) is
         interpreted as the mean range of the most recent `rolling_lookback_bars`
         COMPLETE 5-minute bars drawn from the continuous pre-reaction-day bar
         stream (spanning back across prior sessions if one session alone
         doesn't have enough bars), explicitly EXCLUDING the reaction day's own
         bars. This is a disclosed interpretive choice: comparing today's
         settling range against a baseline that already includes today's own
         elevated post-announcement bars would defeat the point of detecting
         "settling back to normal", so the baseline instead reflects the last
         known NORMAL (pre-event) volatility. The baseline is computed once
         per reaction day and cached.
      3. `frame` is assumed to be native 1-minute RTH bars (per the documented
         contract). 5-minute bars are built by grouping consecutive same-day
         minute rows into non-overlapping chunks of `bar_granularity_minutes`
         rows; any trailing partial chunk at day-end is discarded, never
         treated as a "completed" bar.
      4. Reaction-day resolution uses the REAL `surprise_pct` sign from
         `earnings_events` for direction. For an after-close event
         (`hour >= 16`), the reaction date is NOT guessed via a fixed
         calendar +1 day (that would mis-skip weekends/holidays); instead it
         is resolved lazily as the first actual NEW trading date observed in
         `frame` strictly after the announcement date, using only trading
         dates already streamed through `frame` up to the current bar -- this
         is never look-ahead, since resolution only happens once that later
         date has actually arrived in the bar stream. Pre-market/during-hours
         events (`hour < 16`) react the same calendar day.
      5. End-of-day (for both the settle-window cutoff bookkeeping and the
         max-hold-days exit) is approximated as any bar timestamped at or
         after 15:55 ET, since `frame`'s documented RTH session is 09:30-16:00
         and the true last-bar timestamp isn't independently knowable from
         inside the closure. A same-bar safety-net exit also fires if the
         hold window is ever exceeded without having hit that cutoff (e.g. an
         unusual short session).
      6. Single-position discipline: while a position opened by this signal is
         open, no new reaction-day entries are considered, even on a fresh
         reaction day, matching a typical single-position backtest engine.

    Fill-day capture (behaviorally inert in every reachable
    case): `entry_trading_day_count` is
    captured lazily on the first `was_open` call rather than at the moment
    the entry signal fires, so `days_held` is always measured from the frame
    date of the ACTUAL fill bar (next-bar-open execution) rather than the
    settle-bar decision frame -- these only differ if the settle bar were the
    literal last RTH bar of its session, which `max_settle_wait_minutes<=60`
    inside a 09:30-16:00 session makes unreachable, but this removes the
    reliance on that being true rather than assuming it.
    """

    import datetime

    settle_h, settle_m = (int(x) for x in settle_window_start_et.split(":"))
    settle_start_time = datetime.time(settle_h, settle_m)
    settle_open_minutes = settle_h * 60 + settle_m

    same_day_events: dict = {}
    after_close_events: list = []
    for ev in earnings_events:
        ev_date = datetime.date.fromisoformat(ev["date"])
        surprise = ev.get("surprise_pct", 0.0)
        direction = 1 if surprise > 0 else (-1 if surprise < 0 else 0)
        hour = ev.get("hour", 16)
        if hour >= 16:
            after_close_events.append((ev_date, direction))
        else:
            same_day_events[ev_date] = direction
    after_close_events.sort(key=lambda t: t[0])

    state = {
        "last_date": None,
        "pending_after_close": None,
        "reaction_days": {},
        "day_settle_expired": set(),
        "baseline_cache": {},
        "position_open": False,
        "entry_day": None,
        "entry_trading_day_count": None,
        "trading_day_count": 0,
    }

    def _completed_5min_ranges(day_frame: pd.DataFrame, gran: int) -> list:
        n = len(day_frame)
        n_bins = n // gran
        ranges = []
        for b in range(n_bins):
            chunk = day_frame.iloc[b * gran:(b + 1) * gran]
            ranges.append(float(chunk["High"].max() - chunk["Low"].min()))
        return ranges

    def _compute_baseline(frame: pd.DataFrame, current_date):
        # Bounded to a recent tail BEFORE the O(len(frame)) `.date`
        # conversion -- filtering the full frame by `.date` first and
        # only then taking the tail would still pay the unbounded cost
        # on every call. This call
        # is cached per current_date at its one call site, so it only
        # runs once per day, but even once/day over a multi-year
        # backtest is avoidable cost. +450 covers today's own
        # partial session so it never crowds out needed prior bars.
        needed_minutes = rolling_lookback_bars * bar_granularity_minutes * 3 + 500
        prior = frame.tail(needed_minutes + 450)
        prior = prior[prior.index.date < current_date]
        if prior.empty:
            return None
        prior = prior.tail(needed_minutes)
        all_ranges = []
        for _, day_df in prior.groupby(prior.index.date, sort=True):
            all_ranges.extend(
                _completed_5min_ranges(day_df.sort_index(), bar_granularity_minutes)
            )
        if len(all_ranges) < rolling_lookback_bars:
            return None
        recent = all_ranges[-rolling_lookback_bars:]
        return sum(recent) / len(recent)

    def signal_fn(frame: pd.DataFrame, i: int):
        if len(frame) == 0:
            return SignalResult()

        ts = frame.index[-1]
        current_date = ts.date()

        if state["pending_after_close"] is None:
            state["pending_after_close"] = list(after_close_events)

        if state["last_date"] != current_date:
            state["trading_day_count"] += 1
            if current_date in same_day_events and current_date not in state["reaction_days"]:
                state["reaction_days"][current_date] = same_day_events[current_date]
            still_pending = []
            for ev_date, direction in state["pending_after_close"]:
                if ev_date < current_date:
                    if current_date not in state["reaction_days"]:
                        state["reaction_days"][current_date] = direction
                else:
                    still_pending.append((ev_date, direction))
            state["pending_after_close"] = still_pending
            state["last_date"] = current_date

        result_entry = False
        result_exit = False
        was_open = state["position_open"]

        if was_open:
            if state["entry_trading_day_count"] is None:
                state["entry_trading_day_count"] = state["trading_day_count"]
            days_held = state["trading_day_count"] - state["entry_trading_day_count"] + 1
            is_eod = ts.time() >= datetime.time(15, 55)
            if days_held > max_hold_days or (days_held >= max_hold_days and is_eod):
                result_exit = True
                state["position_open"] = False
                state["entry_day"] = None
                state["entry_trading_day_count"] = None

        if not was_open:
            direction = state["reaction_days"].get(current_date)
            if direction == 1:
                minutes_since_open = (ts.hour * 60 + ts.minute) - settle_open_minutes
                if 0 <= minutes_since_open <= max_settle_wait_minutes:
                    recent = frame.tail(minutes_since_open + bar_granularity_minutes + 1)
                    day_today = recent[
                        (recent.index.date == current_date)
                        & (recent.index.time >= settle_start_time)
                    ]
                    n_bars_today = len(day_today)
                    if (
                        n_bars_today >= bar_granularity_minutes
                        and n_bars_today % bar_granularity_minutes == 0
                    ):
                        last_bin = day_today.iloc[
                            n_bars_today - bar_granularity_minutes:n_bars_today
                        ]
                        last_range = float(last_bin["High"].max() - last_bin["Low"].min())
                        if current_date not in state["baseline_cache"]:
                            state["baseline_cache"][current_date] = _compute_baseline(
                                frame, current_date
                            )
                        baseline = state["baseline_cache"][current_date]
                        if (
                            baseline is not None
                            and baseline > 0
                            and last_range <= range_compression_multiple * baseline
                        ):
                            result_entry = True
                            state["position_open"] = True
                            state["entry_day"] = current_date
                            # entry_trading_day_count is deliberately NOT set
                            # here: it is lazily captured on the first
                            # was_open==True call above, which is guaranteed
                            # to see the frame ending at the ACTUAL fill bar
                            # (next-bar-open execution), not the settle-bar
                            # decision frame -- these differ only if the
                            # settle bar were the literal last RTH bar of its
                            # day (unreachable given max_settle_wait_minutes
                            # <= 60 vs a 09:30-16:00 session), but capturing
                            # it lazily makes days_held correct even in that
                            # case rather than relying on that being true.
                elif minutes_since_open > max_settle_wait_minutes:
                    state["day_settle_expired"].add(current_date)

        return SignalResult(entry=result_entry, exit=result_exit)

    return signal_fn


# --- vwap_reclaim_reversal ---
def make_vwap_reclaim_reversal_signal_fn(
    bar_granularity_minutes: int = 5,
    extension_window_end_et: str = "10:30",
    extension_atr_multiple: float = 1.5,
    atr_lookback_days: int = 20,
    reclaim_volume_multiple: float = 1.5,
    max_hold_days: int = 20,
):
    """
    Session-VWAP Reclaim Reversal (long-only).

    Mechanism implemented: during the first 60 minutes of the regular session
    (09:30 ET to `extension_window_end_et`, default 10:30 ET), watch a resampled
    `bar_granularity_minutes`-minute bar series for a bar whose Low trades at
    least `extension_atr_multiple`x the `atr_lookback_days`-trading-day daily ATR
    below the day's own running (session-anchored) VWAP -- this "flags" the name
    as extended below VWAP for the rest of that session. Once flagged, the first
    LATER `bar_granularity_minutes`-minute bar (any time later the same session
    day, not restricted to the flag window -- see approximation #4 below) whose
    Close reclaims back above the running session VWAP, on volume at least
    `reclaim_volume_multiple`x the trailing 20-bar average volume of that same
    bar series, is treated as the entry trigger. A position, once opened, is held
    for up to `max_hold_days` trading days -- enforced entirely by the engine
    (`run_intraday_backtest`'s own `max_hold_days` argument), not by this
    function (see "Closure design" below).

    Scope decision: only the long side is
    implemented. The backtest engine's `direction` is a single whole-call
    parameter, so this candidate is deliberately scoped to long-only;
    no short mirror is built here.

    ## Closure design: fully STATELESS, no mutable closure state

    This is deliberate (see "Implementation notes" below): this
    closure keeps NO mutable state across calls. Every call recomputes
    everything -- the running session VWAP, whether an extension-flag bar
    occurred earlier the same session, the daily ATR, and the trailing
    volume average -- purely from the `frame` (and `i`, unused: `frame`
    always ends exactly at the bar `i` identifies, so `frame.index[-1]` is
    equivalent and is what this function reads) it is handed on THIS call.
    This closure is therefore safe to reuse, unmodified, across multiple
    tickers or multiple concurrent/out-of-order backtest runs -- no per-run
    reset is required, and there is no way for its own bookkeeping to drift
    out of sync with the engine's.

    Nothing here reads past `frame.iloc[-1]` (`ts`, `today`) or past any day
    strictly earlier than `today` -- structurally look-ahead-safe by the same
    convention as the engine's own `SignalFn` contract (see
    `mining.intraday_backtest_engine.SignalFn`'s docstring).

    ## `max_hold_days` is validated here but enforced by the ENGINE, not here

    This function never emits `exit=True` for hold-duration reasons -- the
    engine's own `max_hold_days` argument to `run_intraday_backtest` is the
    SOLE enforcement of "hold for up to max_hold_days trading days" (the
    orchestrating `scripts/run_swing_mining.py` wires
    `candidate["max_hold_days"]` -- the SAME source value this factory
    receives -- directly into that call, so there is no divergence risk).
    This matches the convention already used elsewhere in this
    codebase for an identical "hold N days, then time-stop" mechanism:
    `src/mining/signals.py`'s `make_pead_signal_fn` never sets exit=True
    either, leaving the exit to the engine's time_stop.
    `max_hold_days` is still accepted and validated by this factory (kept in
    the signature for parameter-set completeness against the catalog
    entry's parameters) even though the closure body has no further use for it.

    ## Implementation notes

    1. **Imports.** `SignalResult` is imported from the intraday engine
       module (`from .intraday_backtest_engine import SignalFn,
       SignalResult` at the top of this file, i.e.
       `mining.intraday_backtest_engine`), because every branch of the
       closure, including the bare "no signal yet" return, constructs one.

    2. **No internal exit tracking.** The closure keeps no
       `max_hold_days`/day-tracking state of its own. Anchoring such state to
       the SIGNAL bar's date (the bar that returned `entry=True`) rather than
       the EXECUTION bar's date (the NEXT bar's open, where
       `mining.intraday_backtest_engine.run_intraday_backtest` actually opens
       the position, wired with `max_hold_days=candidate["max_hold_days"]` in
       `scripts/run_swing_mining.py`'s `_backtest_tickers`) can differ by one
       calendar day whenever the signal bar is the session's last bar. That
       mismatch has two failure modes: (a) the engine's own `max_hold_days`
       force-close fires a calendar day before the internal tracking would, so
       the closure's `state["in_position"]` stays stuck `True` until a later
       calendar day's entry check resets it, silently skipping real
       flag/reclaim entries for the rest of that trading day; (b) when the
       reclaim signal bar is the session's last bar, the internal tracking
       fires its own exit one trading day early relative to the intended
       20-day hold. Exit is therefore left entirely to the engine's
       `time_stop` (see "Closure design" above): there is no closure state
       that could disagree with the engine's own position bookkeeping.

    Disclosed approximations / interpretive choices:

    1. Alpaca's `VWAP` column is that bar's OWN per-bar VWAP, not a running
       session VWAP. The "running session VWAP" this strategy needs is
       recomputed on every call from that session's own 1-minute bars
       (cumsum(VWAP*Volume)/cumsum(Volume) from the first RTH minute bar of
       `today` through the current bar), evaluated at 1-minute resolution.
    2. The `bar_granularity_minutes`-minute bars the mechanism is defined on
       are built by resampling the underlying 1-minute RTH bars inside the
       closure (clock-aligned bins, e.g. 09:30-09:34 for 5-minute bars,
       which line up exactly with the RTH open/close since both 09:30 and
       16:00 are multiples of 5 minutes -- no cross-session bin
       contamination). The flag and reclaim conditions are only evaluated
       once per completed `bar_granularity_minutes` window (the 1-minute bar
       that completes that window: `frame` is guaranteed to end exactly at
       that bar, so the resampled bin it completes is guaranteed whole, not
       partial), not on every 1-minute bar.
    3. The 20-day ATR is a simple (non-Wilder) rolling mean of daily True
       Range over the last `atr_lookback_days` COMPLETED trading days
       (today's own still-forming session is excluded), built by resampling
       1-minute RTH bars to daily OHLC.
    4. The mechanism text puts the extension FLAG inside the
       09:30-`extension_window_end_et` window but gives the reclaim leg no
       end time of its own. This implementation restricts the flag
       condition to that window, but allows the reclaim (VWAP-reclaim +
       volume) to fire on any later `bar_granularity_minutes` bar the SAME
       session day, and only on a bar STRICTLY AFTER the one that set the
       flag (never the same bar, re-derived fresh each call by scanning
       today's own bars rather than a persisted flag) -- treating
       flag-then-reclaim as a sequential, two-bar-minimum pattern, matching
       the mechanism's "if price THEN closes back above" wording. A flag not
       reclaimed by the session's close simply expires (never carries into a
       later day).
    5. The reclaim's "trailing 20-bar average volume" is computed over a
       continuous, multi-day series of `bar_granularity_minutes`-minute bars
       (not reset each session), so the filter is well defined even on the
       first few bars of a session, before 20 same-day bars exist yet. Until
       at least 20 trailing bars plus the current bar are available, no
       reclaim/entry is evaluated for that bar.
    6. All `parameters` are used directly: `bar_granularity_minutes` sets
       the resample frequency, `extension_window_end_et` bounds the flag
       window, `extension_atr_multiple` and `atr_lookback_days` define the
       flag threshold, `reclaim_volume_multiple` gates the reclaim's volume
       filter, and `max_hold_days` bounds the hold (via the engine, as
       described above -- and is validated here).
    """
    import datetime

    if (isinstance(max_hold_days, bool) or not isinstance(max_hold_days, int)
            or max_hold_days < 1):
        raise ValueError(f"max_hold_days must be a positive int, got {max_hold_days!r}")

    window_end_h, window_end_m = (int(x) for x in extension_window_end_et.split(":"))
    window_end_time = datetime.time(window_end_h, window_end_m)
    session_open_time = datetime.time(9, 30)
    session_open_minutes = session_open_time.hour * 60 + session_open_time.minute

    def _recent_bars(hist_frame, minutes_back):
        tail = hist_frame.tail(minutes_back) if len(hist_frame) > minutes_back else hist_frame
        bars = tail.resample(f"{bar_granularity_minutes}min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        )
        return bars.dropna(subset=["Open"])

    def _daily_atr(hist_frame, today):
        # Perf: this runs on EVERY bar_granularity_minutes-minute bar,
        # ~78x/day at the default 5-minute granularity, even though this
        # function's own result is invariant across all of those calls
        # (it excludes "today" entirely, so it cannot change intraday).
        # Rebuilding the daily bar series each call with a generic
        # .resample("1D").agg(dict) call, PLUS an .index.date comparison
        # (a per-row Python object conversion, not vectorized -- see
        # _time_of_day_avg_volume's own comment above for the same swap),
        # is too slow at that call rate, so BOTH the date filter and the
        # daily-bar aggregation use vectorized/numpy equivalents that do
        # the exact same bounded amount of work far more cheaply per call,
        # rather than cross-call memoization -- this closure is
        # deliberately kept fully stateless (see "Closure design" above),
        # including across the multiple tickers `scripts/run_swing_mining.py`
        # replays through the SAME closure instance one after another, so a
        # `today`-keyed cache is not a safe option here.
        lookback_minutes = (atr_lookback_days + 15) * 390
        tail = hist_frame.tail(lookback_minutes) if len(hist_frame) > lookback_minutes else hist_frame
        # `tail` is time-sorted, so "every row strictly before today" is a
        # contiguous PREFIX -- `searchsorted` finds its end in O(log n) via
        # a single Timestamp comparison, instead of computing `.normalize()`
        # (itself a real per-row tz round-trip -- .normalize() on a
        # tz-aware index re-localizes through a naive intermediate to
        # handle DST -- not just a cheap vectorized truncation) over every
        # row and then boolean-masking.
        today_ts = pd.Timestamp(today, tz=tail.index.tz)
        pos = tail.index.searchsorted(today_ts, side="left")
        if pos == 0:
            return None
        prior = tail.iloc[:pos]
        # Day-segment boundaries only need to detect WHEN the calendar day
        # changes between consecutive (already time-sorted) rows, not the
        # actual local date -- the raw UTC day number does that just as
        # correctly here, including across DST transitions, because this whole factory
        # already assumes a fixed RTH 09:30-16:00 ET session that never
        # crosses a UTC midnight boundary, and is dramatically cheaper
        # since it skips `.normalize()`'s tz round-trip entirely.
        prior_days = prior.index.asi8 // 86_400_000_000_000

        # Prior is time-sorted, so each calendar day's rows are contiguous --
        # find the start of each day-segment directly instead of grouping.
        day_starts = np.flatnonzero(
            np.r_[True, prior_days[1:] != prior_days[:-1]]
        )
        if len(day_starts) < atr_lookback_days + 1:
            return None
        day_starts = day_starts[-(atr_lookback_days + 1):]
        day_ends = np.r_[day_starts[1:], len(prior_days)]

        high = prior["High"].to_numpy(dtype=float)
        low = prior["Low"].to_numpy(dtype=float)
        close = prior["Close"].to_numpy(dtype=float)
        daily_high = np.maximum.reduceat(high, day_starts)
        daily_low = np.minimum.reduceat(low, day_starts)
        daily_close = close[day_ends - 1]

        prev_close = daily_close[:-1]
        hi = daily_high[1:]
        lo = daily_low[1:]
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_close), np.abs(lo - prev_close)))
        if len(tr) < atr_lookback_days:
            return None
        return float(tr[-atr_lookback_days:].mean())

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        ts = frame.index[-1]
        today = ts.date()

        # Only evaluate on the 1-minute bar that COMPLETES a
        # bar_granularity_minutes-minute window -- `frame` ends exactly at
        # `ts`, so when this holds, the resampled bin `ts` completes is
        # guaranteed whole (never partial).
        minutes_since_open = (ts.hour * 60 + ts.minute) - session_open_minutes
        if minutes_since_open < 0:
            return SignalResult()
        if (minutes_since_open + 1) % bar_granularity_minutes != 0:
            return SignalResult()

        today_ts = pd.Timestamp(today, tz=frame.index.tz)

        # Multi-day bar_granularity_minutes-minute series (approximation #5:
        # the reclaim's trailing-volume average is continuous across days,
        # not reset each session). Only the current bin plus the trailing
        # 20-bin window (at most ~100 minutes) and today's own bins are
        # ever read from this -- 3 trading days of 1-minute bars is a
        # comfortable, still-bounded margin (see _daily_atr's perf comment
        # above for the other expensive per-call scan in this closure).
        recent_bars = _recent_bars(frame, 3 * 390)
        if recent_bars.empty:
            return SignalResult()
        cur_bar = recent_bars.iloc[-1]

        # Running SESSION VWAP, recomputed fresh from today's own 1-minute
        # bars only (approximation #1) -- never persisted across calls.
        # Bounded to a recent tail before finding where today's own bars
        # start -- a single RTH session is at most ~390 one-minute bars,
        # so 1000 comfortably contains all of today's bars. This call site
        # runs on every bar_granularity_minutes-minute bar (~78x/day at the
        # default 5-minute granularity); `searchsorted` (a single O(log n)
        # Timestamp comparison on the already time-sorted tail) is used
        # instead of `.index.normalize() == today_ts` (a per-row tz
        # round-trip) or
        # `.index.date` (a per-row Python object conversion) -- see
        # _daily_atr's own perf comment above.
        today_1min = frame.tail(1000)
        today_1min = today_1min.iloc[today_1min.index.searchsorted(today_ts, side="left"):]
        if today_1min.empty:
            return SignalResult()
        vol = today_1min["Volume"].astype(float)
        vwap_px = today_1min["VWAP"].astype(float)
        cum_vol = vol.cumsum()
        if cum_vol.iloc[-1] <= 0:
            return SignalResult()
        running_vwap = (vwap_px * vol).cumsum() / cum_vol.replace(0, np.nan)
        session_vwap_now = float(running_vwap.iloc[-1])

        # The running session VWAP AS OF the end of each of today's own
        # bar_granularity_minutes-minute bins -- needed to re-derive "was
        # there an extension-flag bar earlier today" purely from `frame`,
        # with no persisted flag state (see "Closure design" above).
        vwap_at_bin_end = running_vwap.resample(
            f"{bar_granularity_minutes}min", label="left", closed="left"
        ).last()
        today_bars = recent_bars.iloc[recent_bars.index.searchsorted(today_ts, side="left"):]
        if today_bars.empty:
            return SignalResult()
        vwap_at_bin_end = vwap_at_bin_end.reindex(today_bars.index)

        atr20 = _daily_atr(frame, today)
        flagged_before_now = False
        if atr20 is not None and atr20 > 0:
            in_window = (
                (today_bars.index.time >= session_open_time)
                & (today_bars.index.time <= window_end_time)
            )
            threshold = (vwap_at_bin_end - extension_atr_multiple * atr20).to_numpy()
            lows = today_bars["Low"].astype(float).to_numpy()
            # NaN thresholds (e.g. a bin with literally zero volume so far)
            # compare False against any real Low, correctly excluding them
            # rather than raising.
            flag_hits = np.flatnonzero(in_window & (lows <= threshold))
            # Strictly BEFORE the current (last) bin -- a reclaim can never
            # fire on the same bar that set the flag (approximation #4).
            if len(flag_hits):
                flagged_before_now = bool(np.any(flag_hits < len(today_bars) - 1))

        if flagged_before_now and len(recent_bars) >= 21:
            trailing_vol_avg = float(recent_bars["Volume"].iloc[-21:-1].mean())
            if (
                trailing_vol_avg > 0
                and cur_bar["Close"] > session_vwap_now
                and cur_bar["Volume"] >= reclaim_volume_multiple * trailing_vol_avg
            ):
                return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


SIGNAL_FACTORIES = {
    "earnings_reaction_range_breakout_pead": make_earnings_reaction_range_breakout_pead_signal_fn,
    "anchored_vwap_reclaim_continuation": make_anchored_vwap_reclaim_continuation_signal_fn,
    "vcp_breakout_volume_confirmation_swing": make_vcp_breakout_volume_confirmation_swing_signal_fn,
    "gap_and_go_momentum_continuation": make_gap_and_go_momentum_continuation_signal_fn,
    "earnings_gap_continuation_intraday_confirm": make_earnings_gap_continuation_intraday_confirm_signal_fn,
    "avwap_swing_pivot_reclaim_breakout": make_avwap_swing_pivot_reclaim_breakout_signal_fn,
    "earnings_avwap_reaction_reclaim": make_earnings_avwap_reaction_reclaim_signal_fn,
    "avwap_band_extension_reversion": make_avwap_band_extension_reversion_signal_fn,
    "earnings_reaction_delayed_entry": make_earnings_reaction_delayed_entry_signal_fn,
    "intraday_seasonality_continuation": make_intraday_seasonality_continuation_signal_fn,
    "nr7_orb_trend_continuation": make_nr7_orb_trend_continuation_signal_fn,
    "closing_auction_deviation_reversal": make_closing_auction_deviation_reversal_signal_fn,
    "buyable_gap_up_continuation": make_buyable_gap_up_continuation_signal_fn,
    "darvas_box_breakout_volume_confirmed": make_darvas_box_breakout_volume_confirmed_signal_fn,
    "first_hour_range_breakout_swing": make_first_hour_range_breakout_swing_signal_fn,
    "intraday_liquidity_sweep_reversal": make_intraday_liquidity_sweep_reversal_signal_fn,
    "intraday_capitulation_reversal_bar": make_intraday_capitulation_reversal_bar_signal_fn,
    "earnings_reaction_settle_drift": make_earnings_reaction_settle_drift_signal_fn,
    "vwap_reclaim_reversal": make_vwap_reclaim_reversal_signal_fn,
}
