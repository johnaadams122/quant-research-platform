"""Real day-trading signal_fn factories for the 12 curated day-trading
candidates of the intraday/swing strategy-mining pipeline. Every
factory returns a SignalFn matching src/mining/intraday_backtest_engine.
SignalResult/SignalFn's exact contract. Every candidate on this track is
called with max_hold_days=1 by its caller, which mechanically forces every
position flat by its own entry session's close (see
run_day_trading_mining.py's own module docstring for why this needs no new
engine support) -- these factories additionally implement their own earlier,
mechanism-specific exit conditions (stops/targets/force_flat_et cutoffs)
rather than relying on that backstop alone.
"""
from __future__ import annotations

import datetime
import datetime as dt
from datetime import date as _date
from datetime import time as _time

import numpy as np
import pandas as pd

from .intraday_backtest_engine import SignalFn, SignalResult


# --- orb_5min_atr_stop_scalp ---
def make_orb_5min_atr_stop_scalp_signal_fn(
    opening_range_minutes: int = 5,
    entry_time_et: str = "09:35",
    stop_loss_pct_of_atr14: float = 5,
    profit_target_r_multiple: float = 10,
    atr_lookback_days: int = 14,
    force_flat_et: str = "15:55",
    flat_threshold_pct: float = 0.05,
):
    """5-minute opening-range-breakout ATR-stop day scalp (LONG SIDE ONLY).

    Mechanism: the opening range is the first `opening_range_minutes` one-minute
    bars of each RTH session (since the underlying frame is 1-minute bars, not a
    native 5-minute bar, this closure reconstructs the equivalent 5-minute OHLC
    exactly: OR-open = the Open of the very first bar of the session,
    OR-close = the Close of the bar immediately preceding `entry_time_et` -- this
    is mathematically identical to what a resampled 5-minute bar's Open/Close
    would be, not an approximation). If OR-close > OR-open (bullish), a long
    entry signal fires on that decision bar, which the engine fills at the NEXT
    bar's open -- i.e. at `entry_time_et` itself, exactly as the mechanism
    describes. If OR-close < OR-open the mechanism calls for a mirrored SHORT
    entry; under the day-trading track's long-only scope that case is
    skipped entirely -- no trade is taken on bearish opening ranges. If OR-open
    and OR-close are "about equal" the day is skipped; "about equal" is
    interpreted as an absolute difference no larger than `flat_threshold_pct`
    percent of OR-open (default 0.05%, i.e. 5 basis points) -- a disclosed
    threshold choice since the candidate names no explicit number for this.

    Risk model: once long, the stop distance is `stop_loss_pct_of_atr14` percent
    (5 -> 5%) of a 14-trading-day ATR measured from DAILY bars resampled from
    this same intraday frame, using only fully-completed sessions strictly
    before the entry day (never today's own in-progress bars). ATR here is a
    simple equal-weighted rolling mean of daily True Range over the last
    `atr_lookback_days` completed sessions (not Wilder's exponential smoothing)
    -- a disclosed simplification, since Wilder smoothing needs continuous
    day-over-day state this per-call closure does not otherwise carry. Stop =
    entry_price - stop_dist; target (profit_target_r_multiple x that risk, 10R
    by default) = entry_price + 10 * stop_dist. Every bar while a position is
    believed open, the bar's own High/Low is checked against target and stop
    (stop checked first as the conservative tie-break when a single bar's range
    could plausibly have touched both, since tick-level sequencing isn't
    available); independently, once the bar's own wall-clock time (already
    tz-aware America/New_York per the frame's own index) is at or after
    `force_flat_et`, the position is flattened even if neither stop nor target
    was hit. `max_hold_days` (1) is a day-trading-track engine parameter
    consumed by the calling backtest engine's own force-close backstop, per
    the day-trading track's contract -- it is not read or enforced inside this
    signal_fn.

    Position state (armed-but-unfilled flag, open/closed flag, entry price,
    stop/target prices, the day's own opening-range open) is tracked in a
    closure-local dict and defensively reset on every date change, since every
    day-trading trade is mechanically guaranteed to close the same
    session it opened in.
    """
    or_minutes = int(opening_range_minutes)
    entry_time = datetime.datetime.strptime(entry_time_et, "%H:%M").time()
    force_flat_time = datetime.datetime.strptime(force_flat_et, "%H:%M").time()
    stop_pct = float(stop_loss_pct_of_atr14) / 100.0
    target_r = float(profit_target_r_multiple)
    lookback = int(atr_lookback_days)
    flat_thresh_pct = float(flat_threshold_pct)

    def _shift_time(t: datetime.time, delta_minutes: int) -> datetime.time:
        total = t.hour * 60 + t.minute + delta_minutes
        total = max(0, min(23 * 60 + 59, total))
        return datetime.time(hour=total // 60, minute=total % 60)

    or_start_time = _shift_time(entry_time, -or_minutes)
    or_end_time = _shift_time(entry_time, -1)

    def _compute_atr14(frame: pd.DataFrame, before_date) -> float:
        # Bounded to a generous trailing window (lookback + 15 sessions'
        # worth of bars) before doing any `.index.date`/`.groupby` work --
        # only the most recent `lookback` completed daily TRs are ever used
        # below, so scanning the entire (potentially 600,000-bar) history on
        # every call is pure waste. Matches the established pattern in
        # swing_signals.py's own `_daily_atr` helper.
        bound_bars = (lookback + 15) * 400
        tail = frame.tail(bound_bars) if len(frame) > bound_bars else frame
        hist = tail[tail.index.date < before_date]
        if hist.empty:
            return None
        daily = hist.groupby(hist.index.date).agg(
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
        )
        if len(daily) < lookback:
            # The threshold is `len(daily) < lookback`, not
            # `len(daily) < lookback + 1`: `tr` is effectively never
            # shortened by the `.dropna()` below (High-Low is always a real,
            # non-NaN value, so pandas' skipna `.max(axis=1)` keeps that row
            # even when the gap-vs-prior-close terms are NaN), so the
            # `len(tr) < lookback` check below reduces to this one. At exactly
            # `len(daily) == lookback` the function therefore returns a real
            # value whose first-day TR is the degenerate High-Low, not None.
            return None
        daily = daily.tail(lookback + 1)
        prev_close = daily["Close"].shift(1)
        tr = pd.concat(
            [
                daily["High"] - daily["Low"],
                (daily["High"] - prev_close).abs(),
                (daily["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        tr = tr.dropna()
        if len(tr) < lookback:
            return None
        return float(tr.tail(lookback).mean())

    state = {
        "last_date": None,
        "or_open": None,
        "or_decided": False,
        "pending_entry_fill": False,
        "pending_atr": None,
        "in_position": False,
        "entry_price": None,
        "stop_price": None,
        "target_price": None,
        "entry_date": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int):
        row = frame.iloc[-1]
        ts = frame.index[-1]
        current_date = ts.date()
        bar_time = ts.time()

        # --- new trading day: defensive reset of all per-day/per-trade state ---
        if state["last_date"] != current_date:
            state["last_date"] = current_date
            state["or_open"] = None
            state["or_decided"] = False
            # Every trade is guaranteed same-day; a position or pending fill
            # surviving into a new date would be a bookkeeping bug -- reset it.
            state["pending_entry_fill"] = False
            state["pending_atr"] = None
            state["in_position"] = False
            state["entry_price"] = None
            state["stop_price"] = None
            state["target_price"] = None
            state["entry_date"] = None

        entry_signal = False
        exit_signal = False

        # --- capture the opening-range open at the first bar of the OR window ---
        if bar_time == or_start_time and state["or_open"] is None:
            state["or_open"] = float(row["Open"])

        # --- consume a pending fill: this bar's open is the real entry price ---
        if state["pending_entry_fill"]:
            entry_price = float(row["Open"])
            atr = state["pending_atr"]
            stop_dist = atr * stop_pct
            state["in_position"] = True
            state["entry_price"] = entry_price
            state["stop_price"] = entry_price - stop_dist
            state["target_price"] = entry_price + target_r * stop_dist
            state["entry_date"] = current_date
            state["pending_entry_fill"] = False
            state["pending_atr"] = None

        # --- manage an open position: stop / target / force-flat-time ---
        if state["in_position"]:
            low = float(row["Low"])
            high = float(row["High"])
            if low <= state["stop_price"]:
                exit_signal = True
            elif high >= state["target_price"]:
                exit_signal = True
            elif bar_time >= force_flat_time:
                exit_signal = True

            if exit_signal:
                state["in_position"] = False
                state["entry_price"] = None
                state["stop_price"] = None
                state["target_price"] = None
                state["entry_date"] = None

        # --- opening-range decision bar: arm a long entry, or skip the day ---
        if (
            bar_time == or_end_time
            and not state["or_decided"]
            and not state["in_position"]
            and not state["pending_entry_fill"]
        ):
            state["or_decided"] = True
            or_open = state["or_open"]
            if or_open is not None and or_open != 0:
                or_close = float(row["Close"])
                diff = or_close - or_open
                threshold = abs(or_open) * (flat_thresh_pct / 100.0)
                if abs(diff) > threshold and diff > 0:
                    # Bullish opening range -> long entry candidate.
                    atr = _compute_atr14(frame, current_date)
                    if atr is not None and atr > 0:
                        state["pending_entry_fill"] = True
                        state["pending_atr"] = atr
                        entry_signal = True
                    # else: not enough daily history yet for a real 14-day ATR
                    # -- skip the day rather than fabricate a stop distance.
                # diff <= 0 (bearish or flat) -> no trade: bearish is the
                # mechanism's short case, out of scope; flat is "about equal".

        return SignalResult(entry=entry_signal, exit=exit_signal)

    return signal_fn


# --- volume_filtered_orb_stocks_in_play_pooled ---
def _parse_et_clock(value: str) -> datetime.time:
    hour_str, minute_str = value.split(":")
    return datetime.time(int(hour_str), int(minute_str))


def make_volume_filtered_orb_stocks_in_play_pooled_signal_fn(
    opening_range_minutes: int = 5,
    relative_volume_lookback_days: int = 20,
    relative_volume_rank_top_pct: float = 10,
    entry_time_et: str = "09:35",
    force_flat_et: str = "15:55",
    atr_period_days: int = 14,
    atr_stop_multiplier: float = 1.0,
    min_relative_volume_history_days: int = 5,
) -> SignalFn:
    """Volume-Filtered ORB (Stocks in Play, Pooled) -- long-only, single-ticker closure.

    Mechanism: each session, forms an opening range from the session's first
    opening_range_minutes bars starting 09:30 ET. Starting at entry_time_et
    (09:35 ET) and until force_flat_et, on the first bar whose Close closes
    above the opening-range High, enters long (actual fill happens at the
    NEXT bar's open, per the engine's own next-open execution contract) --
    but only on a session this ticker is judged "in play" by the relative-
    volume filter below. An open position is exited (signalled on, and per
    the engine's contract filled at, the SAME bar's close) on an ATR-based
    stop-loss below the entry price, or once the current bar's clock time is
    at/after force_flat_et, whichever comes first; the engine's own
    max_hold_days=1 backstop additionally force-closes at the session's last
    bar regardless, so max_hold_days is an engine-level parameter and is
    deliberately NOT referenced in this signal function's own logic, per the
    day-trading track contract.

    FILL-PRICE NOTE: `intraday_backtest_engine.py`'s own SignalFn contract
    fills an entry signal at the FOLLOWING bar's real Open, not the decision
    bar's Close (`run_intraday_backtest` calls the entry-decision signal_fn
    with the frame ending ONE bar before the actual fill bar, then
    immediately makes a SEPARATE exit-check call on the same iteration with
    the frame now extended to include the real fill bar). Setting
    `entry_price`/`stop_price` and flipping `in_position` True INSIDE the
    entry-decision call, from that call's own `frame.iloc[-1]["Close"]` (the
    DECISION bar's close), would therefore not match the price the engine
    actually fills at and uses for real P&L (`opens[i]`) -- on a gapping
    breakout bar it would mis-set the stop-loss level checked on the very
    next call. So `entry_price`/`stop_price`/`in_position` are deferred to a
    `pending_entry_fill` flag resolved on the NEXT call, reading that call's
    own `frame.iloc[-1]["Open"]` (now real, already-happened data) as the
    true fill price -- the same pattern this codebase uses for this engine
    contract elsewhere (see `swing_signals.py`'s own
    `state["pending_entry_fill"]` / `state["entry_price"] = float(row["Open"])`
    convention).

    Disclosed approximations / interpretive choices (several details are
    not fully pinned down by the candidate's catalog entry):

    1. Cross-sectional ranking against a live AAPL/NVDA/QQQ universe ("ranks
       ... restricts trading to the top-ranked stocks in play") is
       structurally impossible from inside this closure -- signal_fn only
       ever receives ONE ticker's own frame, never the pooled universe at
       once. It is approximated as a TIME-SERIES analog: today's early
       cumulative Volume (summed over the same opening_range_minutes window
       used to build the opening range) is compared against this SAME
       ticker's own trailing relative_volume_lookback_days history of that
       identical early-volume figure, and today must fall at/above the
       (100 - relative_volume_rank_top_pct)th percentile of that ticker's
       own historical distribution (top 10% => the 90th percentile) to be
       considered "in play" today. Both relative_volume_lookback_days and
       relative_volume_rank_top_pct are used exactly as given, but this is a
       per-ticker historical-percentile proxy for being "in play," not a
       true live rank against the other pooled tickers on the same morning.
    2. The mechanism explicitly reuses "the same ... opening range breakout
       ... as orb_5min_atr_stop_scalp," but that sibling candidate's own
       exact ATR lookback/multiplier are not available to this factory. A
       standard, disclosed default is used instead: a simple (non-Wilder)
       atr_period_days-length (default 14) daily Average True Range computed
       from frame resampled to daily OHLC over COMPLETED trading days
       strictly before today (never today's own partial day), with the stop
       set at entry_price - atr_stop_multiplier * daily_ATR (default
       multiplier 1.0x). atr_period_days / atr_stop_multiplier are therefore
       factory keyword arguments outside the candidate's own named
       parameters dict, not values recovered from the sibling candidate.
    3. Breakout confirmation uses a bar's CLOSE crossing above the opening-
       range High (not an intrabar High touch) -- a standard, conservative
       ORB convention given only 1-minute OHLC bars are available here, no
       tick data.
    4. Because the engine fills an exit signal at the SAME bar's close
       regardless of what intrabar price crossed the stop, this signal_fn's
       ATR-stop trigger check (current bar's Low breaching the stop level)
       and its actual fill price (that bar's Close) can differ from a real
       stop-loss fill near the stop price itself -- this is inherent to the
       engine's own single-resolution-per-bar exit contract, not a choice
       made here.
    5. A position is only opened while current_time is BEFORE force_flat_et
       as well as at/after entry_time_et, so a fresh position is never
       opened moments before the mandatory flatten -- a disclosed realism
       refinement, not a named parameter.
    6. For performance, the historical relative-volume scan and the daily
       ATR resample are each bounded to a trailing CALENDAR-day window
       (3x the relevant lookback, floored at a small minimum) rather than
       rescanning the entire, possibly multi-year, frame every time --  this
       changes nothing about which real bars are looked at (still only real,
       already-visible past data, still zero look-ahead), it only avoids
       rescanning irrelevant far-past history. Both quantities are also
       cached per calendar date in the closure and reused across that same
       session's remaining minute bars rather than recomputed on every
       single bar within the session, since neither can change intraday
       once each is first computed for that date.
    7. LONG ONLY, per the day-trading track's long-only scope -- no short mirror
       is implemented even though a real ORB is commonly mirrored short.
    8. A session is only usable for the opening range / relative-volume
       calculations if its first available bar in frame is exactly 09:30 ET
       -- a session whose earliest observed bar is some other time (a data
       gap, an unusual partial day, or the very first date in the whole
       fetched history if it does not start at the open) is skipped
       defensively rather than guessed at.
    9. A pending entry whose actual fill bar lands on a DIFFERENT calendar
       date than the decision bar (only possible around an unannounced
       early-close session, since entries are otherwise only decided with
       several minutes of buffer before force_flat_et and the regular
       session close) is dropped silently rather than filled -- the
       opening-range breakout it was based on describes a DIFFERENT
       session than the one the fill would land in, so filling it would
       silently break the "same session" premise the whole mechanism (and
       force_flat_et itself) depends on. A missed opportunity, never a
       look-ahead or a same-day violation.
    """
    or_minutes = int(opening_range_minutes)
    rvol_lookback = int(relative_volume_lookback_days)
    rvol_top_pct = float(relative_volume_rank_top_pct)
    entry_clock = _parse_et_clock(entry_time_et)
    flat_clock = _parse_et_clock(force_flat_et)
    atr_days = int(atr_period_days)
    atr_mult = float(atr_stop_multiplier)
    min_hist_days = int(min_relative_volume_history_days)

    state = {
        "in_position": False,
        "entry_price": None,
        "stop_price": None,
        "position_date": None,
        "pending_entry_fill": False,
        "pending_atr": None,
        "pending_decision_date": None,
        "date_cache": {},
    }

    def _session_early_bars(day_bars):
        if day_bars.empty:
            return None
        start = day_bars.index[0]
        if start.time() != datetime.time(9, 30):
            return None
        cutoff = start + pd.Timedelta(minutes=or_minutes)
        early = day_bars[day_bars.index < cutoff]
        if len(early) < or_minutes:
            return None
        return early

    def _compute_daily_atr(frame, today):
        # Bounded to a generous trailing window before doing any
        # `.index.date` work -- the mechanism itself (docstring item 6)
        # already scopes this to a trailing calendar-day window, so this
        # only avoids re-scanning irrelevant far-past history on every
        # once-per-day call.
        window_days = max(atr_days * 3, 45)
        bound_bars = (window_days + 10) * 400
        tail_frame = frame.tail(bound_bars) if len(frame) > bound_bars else frame
        window_start = today - datetime.timedelta(days=window_days)
        tail_date = tail_frame.index.date
        mask = (tail_date >= window_start) & (tail_date < today)
        windowed = tail_frame.loc[mask]
        if windowed.empty:
            return None
        daily = windowed.resample("D").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        if len(daily) < atr_days + 1:
            return None
        prev_close = daily["Close"].shift(1)
        true_range = pd.concat(
            [
                daily["High"] - daily["Low"],
                (daily["High"] - prev_close).abs(),
                (daily["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_value = true_range.rolling(atr_days).mean().iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            return None
        return float(atr_value)

    def _compute_in_play_and_or(frame, today, today_bars):
        early = _session_early_bars(today_bars)
        if early is None:
            return None
        today_early_vol = float(early["Volume"].sum())
        or_high = float(early["High"].max())
        or_low = float(early["Low"].min())

        # Same bounding rationale as `_compute_daily_atr` above.
        window_days = max(rvol_lookback * 3, 60)
        bound_bars = (window_days + 10) * 400
        tail_frame = frame.tail(bound_bars) if len(frame) > bound_bars else frame
        window_start = today - datetime.timedelta(days=window_days)
        tail_date = tail_frame.index.date
        mask = (tail_date >= window_start) & (tail_date < today)
        recent = tail_frame.loc[mask]
        if recent.empty:
            return {"or_high": or_high, "or_low": or_low, "in_play": False}

        recent_dates = recent.index.date
        hist_dates = sorted(set(recent_dates))
        if len(hist_dates) < min_hist_days:
            return {"or_high": or_high, "or_low": or_low, "in_play": False}
        lookback_dates = (
            hist_dates[-rvol_lookback:] if len(hist_dates) > rvol_lookback else hist_dates
        )

        hist_vols = []
        for d in lookback_dates:
            d_bars = recent[recent_dates == d]
            d_early = _session_early_bars(d_bars)
            if d_early is None:
                continue
            hist_vols.append(float(d_early["Volume"].sum()))

        if len(hist_vols) < min_hist_days:
            return {"or_high": or_high, "or_low": or_low, "in_play": False}

        threshold = float(np.percentile(hist_vols, 100.0 - rvol_top_pct))
        in_play = today_early_vol >= threshold
        return {"or_high": or_high, "or_low": or_low, "in_play": in_play}

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        idx = frame.index
        now_ts = idx[-1]
        today = now_ts.date()
        current_time = now_ts.time()

        # Resolve a pending entry decided on a PRIOR call: the engine fills
        # an entry signal at the FOLLOWING bar's real Open (next-open
        # execution), never at the decision bar's own Close. This bar's
        # Open is now real, already-happened data -- anchor entry_price
        # (and therefore stop_price) to it, matching the ACTUAL fill price
        # the engine uses for real P&L, rather than the earlier decision
        # bar's Close, which the engine never actually fills at.
        if state["pending_entry_fill"]:
            if today == state["pending_decision_date"]:
                fill_price = float(frame.iloc[-1]["Open"])
                state["in_position"] = True
                state["entry_price"] = fill_price
                state["stop_price"] = fill_price - atr_mult * state["pending_atr"]
                state["position_date"] = today
            # else: the decision bar turned out to be that session's last
            # bar (e.g. an unannounced early close) -- the next real bar is
            # a NEW calendar day, so the opening-range breakout this entry
            # was based on no longer describes "today" for that session.
            # Drop the pending fill silently (see docstring item 9).
            state["pending_entry_fill"] = False
            state["pending_atr"] = None
            state["pending_decision_date"] = None

        # Defensive same-day reset: a tracked position must never be believed
        # open on a later calendar date than it was opened. Structurally this
        # should be impossible (force_flat_et fires well before the session
        # close, and the engine's own max_hold_days=1 backstop also force-
        # closes at the entry session's last bar), but this guards against
        # this closure's own bookkeeping bugs rather than relying on that.
        if state["position_date"] is not None and state["position_date"] != today:
            state["in_position"] = False
            state["entry_price"] = None
            state["stop_price"] = None
            state["position_date"] = None

        if state["in_position"]:
            current_bar = frame.iloc[-1]
            exit_signal = False
            if state["stop_price"] is not None and float(current_bar["Low"]) <= state["stop_price"]:
                exit_signal = True
            elif current_time >= flat_clock:
                exit_signal = True
            if exit_signal:
                state["in_position"] = False
                state["entry_price"] = None
                state["stop_price"] = None
                state["position_date"] = None
            return SignalResult(entry=False, exit=exit_signal)

        # Not in a position: only consider entries inside the eligible
        # window (at/after entry_time_et, strictly before force_flat_et so a
        # fresh position is never opened moments before the flatten).
        if current_time < entry_clock or current_time >= flat_clock:
            return SignalResult()

        cache = state["date_cache"].get(today)
        if cache is None:
            # Only today's own bars are needed here -- bound before the
            # `.index.date` scan rather than scanning the whole frame.
            today_tail = frame.tail(_TAIL_BARS) if len(frame) > _TAIL_BARS else frame
            today_bars = today_tail[today_tail.index.date == today]
            info = _compute_in_play_and_or(frame, today, today_bars)
            if info is None:
                return SignalResult()
            atr_value = _compute_daily_atr(frame, today)
            cache = {
                "or_high": info["or_high"],
                "or_low": info["or_low"],
                "in_play": info["in_play"],
                "atr": atr_value,
            }
            state["date_cache"][today] = cache

        if not cache["in_play"] or cache["atr"] is None:
            return SignalResult()

        current_bar = frame.iloc[-1]
        if float(current_bar["Close"]) > cache["or_high"]:
            state["pending_entry_fill"] = True
            state["pending_atr"] = cache["atr"]
            state["pending_decision_date"] = today
            return SignalResult(entry=True, exit=False)

        return SignalResult()

    return signal_fn


# --- vwap_trend_intraday_scalp ---
def make_vwap_trend_intraday_scalp_signal_fn(
    vwap_confirmation_bars: int = 2,
    session_start_et: str = "09:30",
    min_hold_minutes: int = 5,
    force_flat_et: str = "15:55",
    max_hold_days: int = 1,
) -> SignalFn:
    """
    VWAP Trend Intraday Scalp, LONG SIDE ONLY (disclosed scope decision of the day-trading track --
    the mechanism describes a mirrored short leg on a confirmed cross-below-VWAP; this factory
    intentionally never opens that short, and instead uses the same cross-below condition only
    to flatten an existing long).

    Each call recomputes the CURRENT trading session's own cumulative (anchored) VWAP directly
    from `frame`: the running sum of (bar Volume * that bar's own Alpaca-reported per-bar VWAP)
    over every bar at or after `session_start_et` on the current calendar date through bar `i`,
    divided by the running sum of Volume over that same window. Alpaca's per-bar VWAP is itself
    already a volume-weighted average of the trades inside that one bar, so weighting it again
    by that bar's own Volume when accumulating is the standard, faithful way to build a running
    session VWAP from per-bar OHLCV alone -- this is exact if every bar's own mean trade price
    equals its own VWAP, and a close approximation otherwise; reconstructing the true tick-level
    VWAP is not possible from minute bars. This is disclosed as an approximation.

    A "confirmed crossover" requires `vwap_confirmation_bars` CONSECUTIVE closes strictly on one
    side of that running VWAP (Close > VWAP extends the "above" streak, Close < VWAP extends the
    "below" streak, Close == VWAP resets both to zero). On the bar where the "above" streak first
    reaches `vwap_confirmation_bars`, no position is currently believed open, AND the current bar
    is still strictly before `force_flat_et`, entry=True fires (the engine fills this at the NEXT
    bar's open). That last condition is a deliberate guard, not part of the source mechanism as
    literally stated: without it, a force-flat exit near end of day can immediately re-satisfy
    the (still-intact) above-VWAP streak on the very next bar and re-enter, oscillating
    entry/exit for the rest of the session. On the bar where the "below" streak first
    reaches `vwap_confirmation_bars` while a position IS believed open, exit=True fires -- this
    is the mechanism's own short-flip trigger, repurposed as a flatten-only signal for this
    long-only run. `min_hold_minutes` gates ONLY this VWAP-reversal exit: a confirmed cross-below
    less than `min_hold_minutes` after entry is ignored outright (judged too fresh to act on,
    matching a real scalp desk not whipsawing itself out in the first few minutes) -- elapsed
    hold time is measured from the confirmation bar's own timestamp, a small disclosed
    approximation for the true (one-bar-later) fill timestamp, since the closure has no other
    record of the actual fill time.

    Two exits are NEVER gated by `min_hold_minutes`, since both exist to guarantee same-day
    flatness rather than to express the trend view, and both are checked before the reversal
    exit so neither can be suppressed by it: (1) a defensive stale-position safety net, checked
    FIRST -- if the closure's own tracked entry date is `max_hold_days` or more calendar days
    before the current bar's date, the position is treated as stale bookkeeping and force-exited.
    On a max_hold_days=1 day-trading run this should never actually trigger in real operation
    (the engine itself force-closes every position at its own entry session's last bar, so a
    believed-open position should never survive to see a later date), but unlike a silent
    same-day reset, this check is left fully live and reachable -- it inspects `in_position`
    without first having been reset by any day-boundary bookkeeping elsewhere in this closure, so
    if this closure's own state-tracking is ever wrong, this still fires a real exit=True instead
    of quietly forgetting a position the caller may still consider open. (2) once the current
    bar's own America/New_York clock time is at or after `force_flat_et`, exit=True fires
    unconditionally while a position is believed open -- closing out ahead of the illiquid last
    minutes of the session, well before the engine's own last-bar force-close backstop.

    Every parameter is used directly: `vwap_confirmation_bars` sizes both streak counters,
    `session_start_et` anchors the VWAP accumulation window, `min_hold_minutes` gates the
    reversal exit, `force_flat_et` gates the time-based exit, and `max_hold_days` sizes the
    defensive stale-position safety net.
    """
    import datetime as dt
    import numpy as np
    import pandas as pd

    confirm_n = int(vwap_confirmation_bars)
    min_hold = dt.timedelta(minutes=int(min_hold_minutes))
    max_hold = int(max_hold_days)

    _sess_h, _sess_m = (int(x) for x in str(session_start_et).split(":"))
    _flat_h, _flat_m = (int(x) for x in str(force_flat_et).split(":"))
    session_start_t = dt.time(_sess_h, _sess_m)
    force_flat_t = dt.time(_flat_h, _flat_m)

    # Closure state -- reset defensively on every detected session/day boundary. Mirrors the
    # mutable-dict-captured-by-the-inner-function pattern used by the swing-track factories,
    # but deliberately simpler: no multi-day bookkeeping is needed since every trade here is
    # same-day by construction.
    state = {
        "in_position": False,
        "entry_date": None,
        "entry_time": None,
        "above_streak": 0,
        "below_streak": 0,
        "session_date": None,
        "cum_pv": 0.0,  # cumulative Volume * per-bar VWAP for the current session
        "cum_v": 0.0,   # cumulative Volume for the current session
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        row = frame.iloc[-1]
        ts = frame.index[-1]
        cur_date = ts.date()
        cur_time = ts.time()

        if state["session_date"] != cur_date:
            # New session (or first call ever): reset the running VWAP accumulator and the
            # crossover streak counters for the new day. Deliberately does NOT touch
            # in_position/entry_date here -- that stays the sole job of the stale-position
            # safety-net check below, so that check remains a genuinely reachable exit path
            # (with its own real exit=True signal) instead of dead code silently preempted by
            # this reset.
            state["session_date"] = cur_date
            state["cum_pv"] = 0.0
            state["cum_v"] = 0.0
            state["above_streak"] = 0
            state["below_streak"] = 0

        vol = row["Volume"]
        bar_vwap = row["VWAP"]
        if cur_time >= session_start_t and pd.notna(vol) and pd.notna(bar_vwap):
            state["cum_pv"] += float(vol) * float(bar_vwap)
            state["cum_v"] += float(vol)

        if state["cum_v"] <= 0:
            # No real volume yet to anchor a session VWAP -- insufficient history this bar.
            return SignalResult()

        session_vwap = state["cum_pv"] / state["cum_v"]
        close = row["Close"]
        if pd.isna(close) or np.isnan(session_vwap):
            return SignalResult()
        close = float(close)

        if close > session_vwap:
            state["above_streak"] += 1
            state["below_streak"] = 0
        elif close < session_vwap:
            state["below_streak"] += 1
            state["above_streak"] = 0
        else:
            state["above_streak"] = 0
            state["below_streak"] = 0

        # Defensive stale-position safety net -- checked first so a bookkeeping bug can never
        # suppress a needed flatten.
        if state["in_position"] and state["entry_date"] is not None:
            days_held = (cur_date - state["entry_date"]).days
            if days_held >= max_hold:
                state["in_position"] = False
                state["entry_date"] = None
                state["entry_time"] = None
                return SignalResult(exit=True)

        # Unconditional force-flat time, regardless of min_hold_minutes.
        if state["in_position"] and cur_time >= force_flat_t:
            state["in_position"] = False
            state["entry_date"] = None
            state["entry_time"] = None
            return SignalResult(exit=True)

        # Confirmed reversal exit, gated by min_hold_minutes.
        if state["in_position"] and state["below_streak"] >= confirm_n:
            held_long_enough = (
                state["entry_time"] is not None and (ts - state["entry_time"]) >= min_hold
            )
            if held_long_enough:
                state["in_position"] = False
                state["entry_date"] = None
                state["entry_time"] = None
                return SignalResult(exit=True)
            # Too fresh to honor the reversal exit -- no signal this bar.

        # Confirmed entry, long only. Gated on being strictly before force_flat_t: once the
        # session has reached its own closing-out window, no NEW position should be opened even
        # if an above-VWAP streak is (still) intact -- without this guard, a force-flat exit
        # immediately re-satisfies the entry condition on the very next bar (the streak counter
        # isn't reset by an exit) and the signal oscillates entry/exit for the rest of the day.
        if (
            (not state["in_position"])
            and state["above_streak"] >= confirm_n
            and cur_time < force_flat_t
        ):
            state["in_position"] = True
            state["entry_date"] = cur_date
            state["entry_time"] = ts
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- intraday_rsi2_reversion_daytrade ---
def make_intraday_rsi2_reversion_daytrade_signal_fn(
    rsi_period: int = 2,
    bar_minutes: int = 5,
    oversold_level: float = 10.0,
    overbought_level: float = 90.0,
    earliest_entry_et: str = "10:00",
    force_flat_et: str = "15:55",
    sma_period: int = 5,
) -> SignalFn:
    """
    Implements the "Intraday RSI(2) Reversion Day Trade" mechanism
    (family_id intraday_rsi2_reversion_daytrade) on real bar data resampled
    INSIDE this closure from the 1-minute OHLCV `frame` the engine hands us
    into synthetic bar_minutes-minute bars, anchored at each session's own
    first observed bar (assumed to be the true 09:30 ET open, with
    contiguous 1-minute bars and no interior gaps -- a disclosed
    simplification; a real data gap mid-session would shift which 1-minute
    bar this closure treats as completing each bar_minutes bucket for the
    rest of that session). RSI(2) (Wilder's smoothing, period=rsi_period --
    the mechanism names "RSI(2)" but not a smoothing convention, so Wilder's
    is used as the standard convention) and a sma_period-bar SMA of
    bar_minutes-minute closes are both computed fresh from ONLY today's
    completed bar_minutes-minute bars (never carried across a session
    boundary, per the mechanism's own "reset fresh at 09:30" instruction).
    Both are only re-evaluated on the 1-minute bar that completes a
    bar_minutes bucket (i.e. every bar_minutes-th real 1-minute bar since
    the session's first bar) -- between those boundary bars this signal_fn
    is a deliberate no-op for entry/SMA purposes, since there is no new
    completed bar_minutes-minute close to react to yet.

    LONG-ONLY SCOPE (disclosed, the day-trading track's long-only scope):
    the mechanism describes a mirrored short entry when RSI(2) rises above
    overbought_level. This factory implements ONLY the long (oversold)
    side. overbought_level is still accepted and validated (it must exceed
    oversold_level) but is never used to open a short position -- the
    backtest engine's own `direction` is a single whole-call parameter, so
    the day-trading track deliberately scopes every candidate to its long
    side only.

    Entry (long): only evaluated at a completed bar_minutes bucket
    boundary, and only when the current bar's America/New_York clock time
    is at or after earliest_entry_et ("10:00" by default) AND strictly
    before force_flat_et ("15:55" by default). Enters when the day's own
    RSI(2), computed on that session's bar_minutes-minute closes so far,
    drops below oversold_level. Per the engine's own next-open execution
    rule the actual fill happens at the FOLLOWING 1-minute bar's open --
    this closure tracks that as a one-bar-pending state so it does not
    re-signal while a fill is already in flight.

    The `current_time < force_flat_et` half of the entry guard exists for a
    concrete reason, not just symmetry with
    earliest_entry_et: without it, a bar_minutes bucket boundary that
    happens to be the LAST bar of the trading day (this always exists for
    a standard 390-minute RTH session when bar_minutes divides 390, e.g.
    the default bar_minutes=5) can still fire an entry signal off a
    late-day oversold reading. The engine's next-open execution then has
    no same-day bar left to fill on and fills at the FOLLOWING session's
    open instead -- an entry priced off one day's RSI(2) reading but
    filled at a different, potentially heavily gapped, day's open. That
    is not a same-day-guarantee bookkeeping bug (the fill day's own
    entry_date/force_flat bookkeeping below is still internally
    consistent and still force-flattens same-day), but it is unfaithful
    to an intraday reversion mechanism that is supposed to react to and
    trade against TODAY's own price action (for example, a hard sell-off
    into day 1's final 5-minute bucket would produce an entry filled at
    day 2's open, price-unrelated to the signal; the guard prevents
    that). Because signal_fn only ever sees bars already elapsed
    (look-ahead-safe by construction), it cannot know from inside a
    single call whether the CURRENT bucket boundary is the literal last
    bar of the session (that requires seeing whether more bars follow
    today, which is future information) -- so this guard is a
    conservative, disclosed mitigation valid for any standard
    force_flat_et set meaningfully before session close, not a
    structural guarantee: an early-close session (e.g. the day before a
    holiday, ending well before force_flat_et) could still in principle
    hit the same underlying issue on its own last bar. That residual
    edge case is unfixable from inside signal_fn without the engine
    exposing same-day-bars-remaining information, which it deliberately
    does not (that would itself be a look-ahead-safety hole).

    Exit: implements the mechanism's own named exit_rule,
    "5bar_sma_cross_or_force_flat", directly. (exit_rule and max_hold_days
    are candidate/engine-level fields, not per-bar factory parameters, so
    they are not accepted as arguments here: max_hold_days=1 is enforced by
    the engine's own existing same-session force-close backstop, and
    exit_rule is simply the name of the exit logic implemented below --
    the "5bar" in its name is what sma_period defaults to.) Two independent
    exit conditions are checked whenever this closure believes a position
    is open:
      1. SMA cross-back (checked only at a bar_minutes bucket boundary,
         since that is the only time a new completed close exists to
         compare): the latest completed bar_minutes-minute close is at or
         above the sma_period-bar SMA of bar_minutes-minute closes for the
         session so far. Implemented as a level check on the current bar
         (close >= sma), not a strict prior-bar-below / current-bar-above
         crossover test -- a disclosed simplification of "a move back
         through the SMA".
      2. Hard force-flat (checked on every 1-minute bar, not just bucket
         boundaries, since it is a real wall-clock cutoff): the current
         bar's America/New_York time is at or after force_flat_et ("15:55"
         by default) while a position is believed open.
    Whichever condition is detected first fires exit=True for that bar.
    Exit is always meant to fire within the same session it entered; this
    closure also defensively resets its own "position believed open"
    bookkeeping the instant it detects a session-date change while
    believing a position is open, purely as a backstop against its own
    bookkeeping bugs. This backstop is not reachable while any position
    it tracks is genuinely still open (the force-flat check above already
    guarantees a same-day exit before the next session's bars begin) --
    but it CAN observe a transient stale True between the moment the
    engine force-closes a position via its own last-bar-of-day time-stop
    backstop (e.g. on an early-close day, before force_flat_et is ever
    reached) and this closure's own next call. That transient has no
    externally observable effect: while `in_position` is stale-True, the
    entry-check branch below is unreachable (it is gated on NOT being in
    a position), so no spurious entry can fire off it, and the very next
    call whose frame actually reaches a new calendar day clears it before
    any other logic runs.

    Insufficient-history bars never crash: if there are fewer than
    rsi_period + 1 completed bar_minutes-minute closes for the day, RSI is
    treated as unavailable and no entry fires; if there are fewer than
    sma_period completed closes, the SMA-cross exit check is skipped for
    that bar (a position already open with too little session history to
    compute the SMA can still exit via the force-flat clock check).
    """
    if not (0.0 < oversold_level < overbought_level <= 100.0):
        raise ValueError(
            "oversold_level must be positive and strictly below overbought_level "
            "(which must itself be <= 100); got "
            f"oversold_level={oversold_level}, overbought_level={overbought_level}"
        )
    if rsi_period < 1 or bar_minutes < 1 or sma_period < 1:
        raise ValueError("rsi_period, bar_minutes and sma_period must all be >= 1")

    _earliest_entry_time = datetime.datetime.strptime(earliest_entry_et, "%H:%M").time()
    _force_flat_time = datetime.datetime.strptime(force_flat_et, "%H:%M").time()

    # Closure-local "do I believe a position is currently open" bookkeeping.
    # Deliberately simple (no multi-day machinery) since every trade this
    # engine opens for a day-trading candidate is same-session only.
    _state = {"in_position": False, "entry_pending": False, "entry_date": None}

    def _wilder_rsi(closes, period):
        if len(closes) < period + 1:
            return None
        diffs = closes.diff().dropna()
        gains = diffs.clip(lower=0.0)
        losses = (-diffs).clip(lower=0.0)
        avg_gain = float(gains.iloc[:period].mean())
        avg_loss = float(losses.iloc[:period].mean())
        for gain_v, loss_v in zip(gains.iloc[period:].to_numpy(), losses.iloc[period:].to_numpy()):
            avg_gain = (avg_gain * (period - 1) + gain_v) / period
            avg_loss = (avg_loss * (period - 1) + loss_v) / period
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0.0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        last_ts = frame.index[-1]
        today = last_ts.date()
        current_time = last_ts.time()

        # Defensive same-day-exit backstop against our own bookkeeping bugs.
        # See the factory docstring's exit section for the one transient,
        # externally-inert case where this is genuinely reached (an
        # engine-side early-close time-stop closing a position before our
        # own force-flat check ever fires).
        if _state["in_position"] and _state["entry_date"] is not None and _state["entry_date"] != today:
            _state["in_position"] = False
            _state["entry_pending"] = False
            _state["entry_date"] = None

        # A pending entry (signaled on the previous call) fills at THIS
        # bar's open, per the engine's own next-open execution rule.
        if _state["entry_pending"]:
            _state["entry_pending"] = False
            _state["in_position"] = True
            _state["entry_date"] = today

        # Isolate today's session only -- RSI(2) and the SMA both reset
        # fresh at 09:30 ET every session, never carried from a prior day.
        # Bounded to a single-session tail before the `.index.date` scan --
        # this call runs on EVERY bar (no once-per-day gate), so scanning
        # the whole, potentially 600,000-bar, frame here every time would be
        # the most expensive whole-frame scan in this file.
        today_tail = frame.tail(_TAIL_BARS) if len(frame) > _TAIL_BARS else frame
        day_mask = today_tail.index.date == today
        day_frame = today_tail.loc[day_mask]
        minutes_elapsed = len(day_frame)
        at_bucket_boundary = (minutes_elapsed % bar_minutes) == 0

        bar_minutes_close = None
        if at_bucket_boundary:
            bucket_ids = np.arange(minutes_elapsed) // bar_minutes
            bar_minutes_close = day_frame["Close"].groupby(bucket_ids).last()
            # bar_minutes_close now holds one entry per FULLY completed
            # bar_minutes-minute bucket so far today (the final bucket is
            # complete precisely because this call sits on its last minute).

        if _state["in_position"]:
            # 1) Hard force-flat clock cutoff -- checked every 1-minute bar.
            if current_time >= _force_flat_time:
                _state["in_position"] = False
                _state["entry_pending"] = False
                _state["entry_date"] = None
                return SignalResult(exit=True)
            # 2) SMA cross-back -- only checkable at a bucket boundary.
            if at_bucket_boundary and bar_minutes_close is not None and len(bar_minutes_close) >= sma_period:
                sma_val = float(bar_minutes_close.iloc[-sma_period:].mean())
                last_close = float(bar_minutes_close.iloc[-1])
                if last_close >= sma_val:
                    _state["in_position"] = False
                    _state["entry_pending"] = False
                    _state["entry_date"] = None
                    return SignalResult(exit=True)
            return SignalResult()

        # Not in a position and no entry already pending: look for a fresh
        # long entry, only at a bucket boundary, only after
        # earliest_entry_et, and only strictly before force_flat_et (see
        # the factory docstring: without the upper bound, a bucket boundary
        # on the literal last bar of the session can fire an entry that
        # the engine's next-open execution then fills at the FOLLOWING
        # session's open).
        if (
            not _state["entry_pending"]
            and at_bucket_boundary
            and current_time >= _earliest_entry_time
            and current_time < _force_flat_time
            and bar_minutes_close is not None
        ):
            rsi_val = _wilder_rsi(bar_minutes_close, rsi_period)
            if rsi_val is not None and rsi_val < oversold_level:
                _state["entry_pending"] = True
                return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- overnight_gap_meanreversion_fade ---
def make_overnight_gap_meanreversion_fade_signal_fn(
    gap_zscore_threshold: float = 1.0,
    gap_lookback_days: int = 60,
    entry_time_et: str = "09:35",
    target_reversion_window_minutes: int = 120,
    stop_loss_atr_multiple: float = 1.0,
    force_flat_et: str = "15:55",
    atr_lookback_days: int = 14,
):
    """Overnight Gap Mean-Reversion Fade (long-only, day-trading track).

    Each morning, computes today's overnight gap (today's session Open minus the
    prior session's Close) and scales it by the recent overnight-gap distribution
    for this ticker: a rolling mean/std of `gap_lookback_days` historical
    open-vs-prior-close gaps (computed from real daily-resampled OHLC built out of
    `frame` inside the closure, per bar `i`, never looking past `frame.iloc[-1]`).
    When today's gap z-score is at or below `-gap_zscore_threshold` (a sufficiently
    deep DOWN gap), the strategy fades it by going LONG at the `entry_time_et` open,
    betting on reversion back up toward the prior session's close and/or today's
    anchored (09:30-forward) session VWAP computed directly from real per-bar
    Volume/Close in `frame`. Position sizing/stop is ATR-based: the stop-loss is
    `stop_loss_atr_multiple` times a simple (non-Wilder) `atr_lookback_days`-day
    average true range, computed from completed prior daily sessions only. The
    trade is force-flat at `force_flat_et` regardless of outcome, and (as a
    defensive backstop only, not the primary mechanism) the engine itself
    force-closes any day-trading position at the entry session's own last bar via
    `max_hold_days=1` -- that engine-level backstop is not reimplemented here.

    Disclosed scope/approximation choices:
    - LONG-ONLY per the day-trading track's scope: only the downside-gap fade (buy
      the dip) is implemented. An upside gap exceeding the threshold would fade
      SHORT under the described mechanism's mirror case; that short variant is
      deliberately not implemented here.
    - Entry timing: because the engine executes an ENTRY signal at the NEXT bar's
      open, this signal_fn fires entry=True one minute-bar before `entry_time_et`
      (e.g. at 09:34 to achieve a 09:35 open fill) so the resulting fill lands at
      the intended clock time.
    - The stop-loss anchor price uses the last available Close at signal time
      (the bar immediately before the actual next-open fill) as a proxy for the
      real (unknown-at-signal-time) fill price -- a small, disclosed
      approximation, since the true fill price cannot be read from `frame` at
      signal time.
    - Reversion target: exit is triggered when price reaches EITHER the prior
      session's close OR today's anchored session VWAP-so-far, whichever comes
      first -- both are treated as valid "reversion achieved" levels per the
      mechanism's own "prior close/session VWAP" phrasing.
    - `target_reversion_window_minutes` is interpreted as a hard exit deadline
      (if neither reversion target is reached within this many minutes of the
      entry signal, the position is closed then) rather than merely descriptive
      text, so that this explicitly named parameter is actually used in the
      exit logic.
    - `atr_lookback_days` (default 14, a standard ATR window) is not one of the
      candidate's own named parameters -- it is a necessary, disclosed addition
      since the mechanism names an ATR-based stop but no ATR window length.
    - A minimum-sample floor of 5 historical gap observations (in addition to
      the `gap_lookback_days` window itself) is imposed before trusting a
      z-score, as a robustness safeguard against a near-empty/degenerate
      lookback window early in a ticker's history.
    - `max_hold_days` is not accepted as a factory parameter: the day-trading
      engine's own `max_hold_days=1` backstop force-closes any position at the
      entry session's last bar structurally, so this candidate's closure does
      not need (and does not implement) any multi-day bookkeeping.
    """

    entry_hh, entry_mm = (int(x) for x in entry_time_et.split(":"))
    _entry_base = dt.datetime(2000, 1, 1, entry_hh, entry_mm)
    entry_signal_time = (_entry_base - dt.timedelta(minutes=1)).time()

    ff_hh, ff_mm = (int(x) for x in force_flat_et.split(":"))
    force_flat_time = dt.time(ff_hh, ff_mm)

    state = {
        "open": False,
        "entry_day": None,
        "prior_close": None,
        "stop_price": None,
        "deadline": None,
        "last_eval_day": None,
    }

    def _reset():
        state["open"] = False
        state["entry_day"] = None
        state["prior_close"] = None
        state["stop_price"] = None
        state["deadline"] = None

    def _daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
        # Bounded to a generous trailing window before the resample -- only
        # the most recent max(gap_lookback_days, atr_lookback_days) daily
        # sessions are ever used below (both slice from the TAIL of the
        # result), so resampling the entire, potentially multi-year, frame
        # on every once-per-day call is pure waste.
        needed_days = max(gap_lookback_days, atr_lookback_days)
        bound_bars = (needed_days + 10) * 400
        frame = frame.tail(bound_bars) if len(frame) > bound_bars else frame
        daily = frame.resample("1D").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        return daily.dropna(subset=["Open", "High", "Low", "Close"])

    def signal_fn(frame: pd.DataFrame, i: int):
        current_ts = frame.index[-1]
        current_day = current_ts.date()
        current_time = current_ts.time()

        # Defensive safety net: a tracked open position must never survive
        # into a new calendar day (every trade here is same-day by design).
        if state["open"] and state["entry_day"] is not None and current_day != state["entry_day"]:
            _reset()

        if state["open"]:
            row = frame.iloc[-1]
            exit_signal = False

            # 1. ATR-based stop-loss: intrabar Low breaches the stop.
            if state["stop_price"] is not None and row["Low"] <= state["stop_price"]:
                exit_signal = True

            # 2. Reversion target: either the prior session's close or
            #    today's anchored (09:30-forward) session VWAP-so-far.
            if not exit_signal:
                # Only today's own bars are needed here (entry_day is
                # always today while a position is tracked open, per the
                # same-day reset above) -- bound before the `.index.date`
                # scan, which otherwise runs on every bar of an active
                # trade day against the whole growing frame.
                today_tail = frame.tail(_TAIL_BARS) if len(frame) > _TAIL_BARS else frame
                mask = today_tail.index.date == state["entry_day"]
                todays_bars = today_tail.loc[mask]
                vol = todays_bars["Volume"]
                vol_sum = vol.sum()
                if vol_sum > 0:
                    session_vwap_now = (todays_bars["Close"] * vol).sum() / vol_sum
                else:
                    session_vwap_now = todays_bars["Close"].iloc[-1]
                if row["Close"] >= state["prior_close"] or row["Close"] >= session_vwap_now:
                    exit_signal = True

            # 3. Reversion window expired without hitting a target.
            if not exit_signal and state["deadline"] is not None and current_ts >= state["deadline"]:
                exit_signal = True

            # 4. Explicit force-flat time reached.
            if not exit_signal and current_time >= force_flat_time:
                exit_signal = True

            if exit_signal:
                _reset()
                return SignalResult(exit=True)

            return SignalResult()

        # Not currently in a position -- consider a new entry, evaluated at
        # most once per day, at (entry_time_et - 1 minute) or the first bar
        # at/after it (robust to any missing minute bars that day).
        if current_time < entry_signal_time:
            return SignalResult()
        if state["last_eval_day"] == current_day:
            return SignalResult()
        state["last_eval_day"] = current_day

        daily = _daily_bars(frame)
        if len(daily) < 2:
            return SignalResult()

        today_open = daily["Open"].iloc[-1]
        prior_sessions = daily.iloc[:-1]
        if len(prior_sessions) < gap_lookback_days + 1:
            return SignalResult()

        prior_close_today = prior_sessions["Close"].iloc[-1]
        today_gap = today_open - prior_close_today

        hist = prior_sessions.iloc[-(gap_lookback_days + 1):]
        hist_gaps = hist["Open"].values[1:] - hist["Close"].values[:-1]
        if len(hist_gaps) < 5:
            return SignalResult()

        hist_gaps_series = pd.Series(hist_gaps)
        gap_mean = hist_gaps_series.mean()
        gap_std = hist_gaps_series.std(ddof=1)
        if not gap_std or pd.isna(gap_std) or gap_std == 0:
            return SignalResult()

        gap_z = (today_gap - gap_mean) / gap_std

        # Long-only: only a sufficiently deep DOWN gap qualifies (fade it by
        # going long). An UP gap exceeding the threshold would fade short,
        # out of scope for this long-only build (see docstring).
        if gap_z > -gap_zscore_threshold:
            return SignalResult()

        if len(prior_sessions) < atr_lookback_days + 1:
            return SignalResult()
        atr_window = prior_sessions.iloc[-(atr_lookback_days + 1):]
        prev_close = atr_window["Close"].shift(1)
        tr = pd.concat(
            [
                atr_window["High"] - atr_window["Low"],
                (atr_window["High"] - prev_close).abs(),
                (atr_window["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.iloc[1:].mean()
        if not atr or pd.isna(atr) or atr <= 0:
            return SignalResult()

        entry_price_est = frame["Close"].iloc[-1]
        stop_price = entry_price_est - stop_loss_atr_multiple * atr

        state["open"] = True
        state["entry_day"] = current_day
        state["prior_close"] = prior_close_today
        state["stop_price"] = stop_price
        state["deadline"] = current_ts + pd.Timedelta(minutes=target_reversion_window_minutes)

        return SignalResult(entry=True)

    return signal_fn


# --- overnight_gap_fade_intraday_reversal ---
def make_overnight_gap_fade_intraday_reversal_signal_fn(
    min_gap_bps: float = 30,
    entry_time_et: str = "09:30",
    force_flat_et: str = "15:58",
) -> SignalFn:
    """
    Implements the "Overnight-Intraday Tug-of-War Fade" mechanism, LONG SIDE
    ONLY (per the day-trading track's disclosed scope decision -- the mechanism also
    describes a mirrored short variant on up-gaps, which is deliberately NOT
    implemented here since the backtest engine's `direction` is a single
    whole-call parameter, not something this signal can flip per-trade).

    Mechanism (long side only): at the start of each new session, compute the
    overnight return as today's session open vs. the prior session's 16:00 ET
    close (found by resampling `frame` to daily bars inside this closure on
    each entry-evaluation call -- `frame.resample("D")` over a
    tz-aware, RTH-only minute index naturally yields one row per real trading
    day once NaN calendar days are dropped). If the stock gapped DOWN by at
    least `min_gap_bps` (magnitude, in basis points -- `parameters.min_gap_bps`),
    fade the move by going long. `entry_time_et` (`parameters.entry_time_et`)
    is used as a floor: an entry is only ever considered on the first bar of a
    session, and only once that bar's own wall-clock time is at or after
    `entry_time_et`, so a session whose first available bar happens to be
    later than the nominal RTH open (e.g. a partial/short day in the raw feed)
    is still evaluated correctly instead of misfiring on the wrong bar.
    The position is force-flattened once the current bar's own America/New_York
    clock time is at or after `force_flat_et` (`parameters.force_flat_et`,
    default "15:58"), tracked via a same-day-only open/flat flag in this
    closure. Multi-day carry cannot happen by construction here (this exit
    logic always fires same-day, well before the engine's own `max_hold_days=1`
    backstop would ever need to act), but the flag is still defensively reset
    the instant a new calendar date is observed on any call, as a safety net
    against a bug in this closure's own bookkeeping.

    APPROXIMATION DISCLOSED (entry fill timing): the mechanism describes entry
    "at the open" (the 09:30 ET print). This engine fills an entry signal at
    the NEXT bar's open, and no lookahead past `frame.iloc[-1]` is structurally
    possible -- so today's actual open price is not knowable until the 09:30
    bar itself has arrived (that bar's own `Open` field IS the session open
    print). This signal therefore fires `entry=True` on bar i == the first bar
    of today's session (the earliest bar at which the overnight gap is
    computable at all), which the engine then fills at THAT bar's next bar's
    open (approximately the 09:31 print) -- about one minute after the literal
    session open. This is the closest faithful fill achievable under this
    engine's no-lookahead, next-open execution model, and does not change the
    mechanism's substance (an overnight-gap fade captured one minute later is
    still the same trade), but is disclosed here as a real, intentional
    approximation rather than a literal 09:30 fill. The engine
    (`intraday_backtest_engine.run_intraday_backtest`) only ever calls the
    entry-search form of `signal_fn` with `frame` truncated to end EXACTLY at
    bar `i - 1` relative to the fill bar `i`, so the call that observes
    today's first bar as `frame.iloc[-1]` is the one whose RETURNED entry
    fires a fill one bar later.

    SIZING NOTE (disclosed, not implemented): `SignalResult` carries only
    boolean `entry`/`exit` fields -- there is no size field in this contract --
    so "sized in proportion to the gap's magnitude" (per the mechanism text)
    is NOT implemented in this signal_fn; it is a position-sizing concern that,
    if desired, belongs to the caller's own engine/allocator layer, entirely
    outside what a `SignalFn` can express. This signal only decides WHETHER to
    enter (gap direction + magnitude gate) and WHEN to exit, never how large a
    position to take.

    `max_hold_days` (`parameters.max_hold_days`) is deliberately NOT accepted
    as a factory parameter here: it is set by the CALLER on the engine itself
    (a force-close backstop this closure does not need to reimplement), not a
    value this signal's own decision logic consumes. This signal's own
    `force_flat_et` exit independently guarantees the same same-day-only
    outcome the engine's backstop enforces on every NORMAL trading day (the
    mining driver script, `scripts/run_day_trading_mining.py`'s
    `_validate_day_trading_candidate`, mechanically REJECTS any day-trading
    candidate whose own `max_hold_days` parameter is not exactly 1 before
    ever calling this signal_fn, so in the real pipeline the engine's own
    backstop is always armed as designed and always agrees with this
    signal's own force-flat exit). The one edge this closure cannot fully
    pre-empt on its own -- a real early-close/half trading session (e.g. the
    day before Thanksgiving) whose last bar lands BEFORE `force_flat_et` --
    is still caught same-day by the engine's own `max_hold_days=1` backstop;
    see the CROSS-RUN STATE RESET note in `signal_fn` below for why that
    specific edge case, left unhandled, would leave stale state behind for a
    LATER ticker's run rather than affect this one's own same-day
    correctness, and how the closure guards against it.
    """
    state = {"open": False, "entry_date": None}

    try:
        entry_floor_time = datetime.datetime.strptime(entry_time_et, "%H:%M").time()
    except (ValueError, TypeError):
        entry_floor_time = datetime.time(9, 30)

    try:
        flat_time = datetime.datetime.strptime(force_flat_et, "%H:%M").time()
    except (ValueError, TypeError):
        flat_time = datetime.time(15, 58)

    gap_threshold_bps = abs(float(min_gap_bps))

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        # CROSS-RUN STATE RESET.
        # `i == 0` -- the PARAMETER this closure receives, not a loop variable
        # -- can ONLY ever occur on the very first entry-search call of a
        # brand-new `run_intraday_backtest` invocation: that engine's own
        # main loop skips calling signal_fn at all when its own loop index is
        # 0 (`if position is None: if i == 0: continue`), so the smallest `i`
        # ever handed to a signal_fn is 0, produced by the very next
        # iteration's entry-search call `signal_fn(df.iloc[:1], 0)`; a
        # position can only ever open at loop index >= 1, so an exit-check
        # call's own `i` is therefore always >= 1 too. `i == 0` is thus a
        # reliable, run-independent "this is the first bar this closure has
        # ever been asked about in THIS backtest run" marker.
        #
        # Why this matters: this factory's returned closure is REUSED,
        # unmodified, across every ticker AND across both the in-sample and
        # holdout stages of one candidate evaluation
        # (`scripts/run_day_trading_mining.py`'s `_backtest_tickers` passes
        # ONE `signal_fn` object across a `for ticker in tickers` loop, and
        # `compute_day_trading_candidates` reuses that SAME object again for
        # the holdout stage; the same one-object-for-every-ticker convention
        # is also stated in `src/mining/signals.py`). Day-trading's own `max_hold_days=1` is
        # mechanically enforced by that driver, so WITHIN one ticker's
        # own run the engine always force-closes any open position by the
        # last bar of its entry day -- but on a real early-close/half trading
        # session (the day before Thanksgiving, some Christmas Eves) that
        # last bar can land BEFORE `force_flat_et` (15:58 default) ever
        # fires, so the engine's backstop closes the trade WITHOUT this
        # signal_fn's own `if state["open"]:` branch ever returning
        # `exit=True` -- and since the engine never calls `signal_fn` again
        # just to report a force-close, this closure's own `state["open"]`
        # would stay stuck `True` past the end of that ticker's run, ready to
        # be misread as "still holding" on the very first bar of the NEXT
        # ticker's (or the holdout stage's) run. The same failure shape
        # applies to any stateful closure reused this way, and
        # `src/mining/swing_signals.py`'s opening-range-breakout factory
        # guards against it with the same `i == 0` reset (there a stale open
        # position would suppress the second ticker's own genuine breakout).
        # Resetting unconditionally on `i == 0` closes it: a genuinely fresh
        # closure is already in exactly this state (so this is a no-op
        # there), and a reused one is forced clean before it can answer for a
        # new run. For THIS closure specifically, the mid-run reset below,
        # the `force_flat_et` branch's own side-effect clear, and the
        # `len(daily) < 2` guard on a brand-new 1-row frame together mean a
        # stale `state["open"]=True` is expected to clear before it can
        # change an entry/exit decision -- unlike the swing_signals.py case,
        # where it would lose a real trade. The `i == 0` reset is therefore
        # correctness-hardening (replacing a fragile "happens to self-heal"
        # property with "cannot go stale at all"), kept because it is free
        # and matches this codebase's convention for exactly this situation.
        #
        # The same-day-only reset a few lines below, keyed on a calendar-date
        # mismatch rather than on `i`, does not return early: it already
        # resolves a WITHIN-run day-to-day carryover on its own, before any
        # consequential entry decision, because the very call that first
        # observes the new day's data as `frame.iloc[-1]` is also the first
        # (and only) call where an entry for that day could legally fire, and
        # the reset always runs before that entry logic is reached. Returning
        # early on the mismatch would suppress the entry evaluation for the
        # new day's true first bar, giving up a real entry opportunity for no
        # additional safety.
        if i == 0:
            state["open"] = False
            state["entry_date"] = None

        if frame is None or len(frame) == 0:
            return SignalResult()

        current = frame.iloc[-1]
        current_ts = frame.index[-1]
        current_date = current_ts.date()

        # Defensive same-day-only reset: never let tracked state believe a
        # position survived into a new calendar day, regardless of what the
        # rest of this closure's logic does.
        if state["open"] and state["entry_date"] is not None and current_date != state["entry_date"]:
            state["open"] = False
            state["entry_date"] = None

        # ---- If a position is open, only ever check the exit condition. ----
        if state["open"]:
            if current_ts.time() >= flat_time:
                state["open"] = False
                state["entry_date"] = None
                return SignalResult(exit=True)
            return SignalResult()

        # ---- Not in a position: only ever consider entering on the first ----
        # ---- bar of a session, at/after entry_time_et.                   ----
        if i >= 1:
            prev_ts = frame.index[-2]
            is_first_bar_of_session = prev_ts.date() != current_date
        else:
            is_first_bar_of_session = True

        if not is_first_bar_of_session:
            return SignalResult()

        if current_ts.time() < entry_floor_time:
            return SignalResult()

        # Need at least one full prior session's Close, resampled from real
        # bars actually present in `frame` up through bar i -- never fabricated.
        # Only the last 2 daily closes are ever read below (today's, and the
        # immediately preceding session's) -- bound to a small trailing tail
        # before the resample rather than resampling the entire frame on
        # every once-per-session call.
        try:
            bound_bars = 4800  # ~12 RTH sessions' worth, comfortably >= 2
            tail_frame = frame.tail(bound_bars) if len(frame) > bound_bars else frame
            daily = tail_frame.resample("D").agg({"Close": "last"}).dropna(subset=["Close"])
        except Exception:
            return SignalResult()

        if len(daily) < 2:
            # Not enough real history yet to know a prior session close.
            return SignalResult()

        prior_close = daily["Close"].iloc[-2]
        today_open = current["Open"]

        if pd.isna(prior_close) or pd.isna(today_open) or prior_close == 0:
            return SignalResult()

        gap_bps = (today_open - prior_close) / prior_close * 10000.0

        # Fade a DOWN gap only (long side of the mechanism, per the day-trading track's scope).
        if gap_bps <= -gap_threshold_bps:
            state["open"] = True
            state["entry_date"] = current_date
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- intraday_half_hour_momentum_continuation ---
def make_intraday_half_hour_momentum_continuation_signal_fn(
    signal_window_start_et: str = "09:30",
    signal_window_end_et: str = "10:00",
    entry_time_et: str = "15:30",
    force_flat_et: str = "15:59",
) -> SignalFn:
    """Implements the long side of "First-Half-Hour to Last-Half-Hour Intraday
    Momentum": at/after `signal_window_end_et` (10:00 ET) each session, compute
    the return from the PRIOR session's own last available bar Close (a real,
    already-observed price -- no lookahead) to today's own last bar Close at or
    before `signal_window_end_et`, using only bars within
    [`signal_window_start_et`, `signal_window_end_et`] of today's session. If
    that return is positive, the mechanism says buy at `entry_time_et` (15:30
    ET) and hold to the close; under the day-trading track's long-only scope,
    a NEGATIVE signal is simply skipped rather than shorted -- the mechanism's
    mirrored short leg is not implemented. Because the engine only ever acts
    on an entry signal at the following bar's open, the entry condition is
    checked one minute BEFORE `entry_time_et` (falling back to the first bar
    at/after that trigger minute if the exact minute is missing from the feed)
    so the resulting next-open fill lands at, or as close as possible to,
    `entry_time_et` itself, not one bar later than the mechanism intends. The
    position is force-flattened once the current bar's own clock time reaches
    `force_flat_et` (15:59 ET), matching the candidate's own explicit
    execution-backstop language; for standard RTH minute bars this is
    typically the session's final bar, so it also closely approximates
    "exit at the 16:00 ET close." All four ET-time parameters are used
    directly in the entry/signal/exit comparisons below; `max_hold_days` is
    not accepted here because the contract states the calling engine's own
    force-close logic (using max_hold_days=1) already guarantees same-day
    flattening independent of this signal_fn.
    """
    from datetime import time as _time, datetime as _dt, timedelta as _td

    def _parse_hm(s: str) -> _time:
        h, m = s.split(":")
        return _time(int(h), int(m))

    def _minus_one_minute(t: _time) -> _time:
        dummy = _dt(2000, 1, 1, t.hour, t.minute) - _td(minutes=1)
        return dummy.time()

    signal_start_t = _parse_hm(signal_window_start_et)
    signal_end_t = _parse_hm(signal_window_end_et)
    entry_t = _parse_hm(entry_time_et)
    flat_t = _parse_hm(force_flat_et)
    entry_trigger_t = _minus_one_minute(entry_t)

    state = {
        "session_date": None,
        "entered_today": False,
        "in_position": False,
        "signal_positive": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if frame.empty:
            return SignalResult()

        ts = frame.index[-1]
        bar_date = ts.date()
        bar_time = ts.time()

        # Defensive new-day reset. This candidate is same-day-only by
        # construction (the engine's own max_hold_days=1 backstop already
        # guarantees no overnight hold), but we reset our own bookkeeping
        # too so a bug here can never believe a position survived a day
        # boundary.
        if state["session_date"] != bar_date:
            state["session_date"] = bar_date
            state["entered_today"] = False
            state["in_position"] = False
            state["signal_positive"] = None

        # Exit: force-flat backstop named explicitly in this candidate's
        # own mechanism text (parameters.force_flat_et), checked first so
        # it always wins over a same-bar entry attempt.
        if state["in_position"] and bar_time >= flat_t:
            state["in_position"] = False
            return SignalResult(exit=True)

        # Compute today's signal once, the first time we reach the end of
        # the signal window (10:00 ET). Uses only bars already inside
        # `frame` -- today's own morning bars and every prior session's
        # bars -- so there is no lookahead.
        if state["signal_positive"] is None and bar_time >= signal_end_t:
            # Only today's morning window and the immediately preceding
            # session's Close are needed -- bound to a small trailing tail
            # before any `.index.date` scan, which otherwise runs (once per
            # session, but over the whole growing frame) against the entire
            # multi-year history.
            bound_bars = 4800  # ~12 RTH sessions' worth, comfortably >= 2
            tail_frame = frame.tail(bound_bars) if len(frame) > bound_bars else frame
            today_mask = tail_frame.index.date == bar_date
            today_frame = tail_frame[today_mask]
            window_frame = today_frame[
                (today_frame.index.time >= signal_start_t)
                & (today_frame.index.time <= signal_end_t)
            ]
            if window_frame.empty:
                return SignalResult()
            today_signal_close = window_frame["Close"].iloc[-1]

            prior_mask = tail_frame.index.date < bar_date
            prior_frame = tail_frame[prior_mask]
            if prior_frame.empty:
                # No prior session yet in the fetched range -- can't form
                # the anchor return.
                return SignalResult()
            prior_date = prior_frame.index[-1].date()
            prior_day_frame = prior_frame[prior_frame.index.date == prior_date]
            prior_close = prior_day_frame["Close"].iloc[-1]

            if pd.isna(prior_close) or pd.isna(today_signal_close) or prior_close == 0:
                return SignalResult()

            ret = (today_signal_close - prior_close) / prior_close
            state["signal_positive"] = bool(ret > 0)

        # Entry: only once per session, only in the final entry window
        # (checked one minute early so the engine's next-open fill lands
        # at entry_time_et), only on the long side, only if the signal is
        # known and positive.
        if (
            not state["entered_today"]
            and not state["in_position"]
            and state["signal_positive"] is True
            and entry_trigger_t <= bar_time < flat_t
        ):
            state["entered_today"] = True
            state["in_position"] = True
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- fomc_post_statement_reversion_fade ---
def make_fomc_post_statement_reversion_fade_signal_fn(
    event_dates: list,
    reference_time_et: str = "14:00",
    entry_time_et: str = "14:15",
    min_deviation_pct: float = 0.15,
    target_reversion_pct: float = 0.1,
    force_flat_et: str = "15:55",
):
    """
    FOMC Post-Statement Reversion Fade (long-only day-trading candidate).

    On real historical FOMC statement days (supplied via `event_dates`, a list of
    ISO "YYYY-MM-DD" strings the caller must populate with real dates -- this
    factory never fabricates or hardcodes any date itself), this signal records
    the traded instrument's price as of `reference_time_et` (default "14:00" ET,
    the moment of release) using that bar's own Close as the reference level.
    At `entry_time_et` (default "14:15" ET, i.e. the first bar whose time is at
    or after 14:15 -- this tolerates any missing-minute gaps in the feed rather
    than requiring an exact 14:00/14:15 bar to exist), it compares the current
    Close against the reference. If price has fallen away from the reference by
    more than `min_deviation_pct` percent, it enters LONG, betting on reversion
    back up toward the reference level -- this is the only direction implemented
    (see scope note below). If price has instead risen by more than the
    threshold, the mechanism as documented calls for a mirrored SHORT entry;
    under the day-trading track's long-only scope, that case is simply not
    traded (no entry fires).

    Exit is whichever of two conditions the same-day session hits first:
      1. The position's own profit target: Close has moved back in the
         favorable direction by at least `target_reversion_pct` percent
         relative to the recorded entry price (a same-bar-Close approximation
         of the actual next-open fill price, disclosed here).
      2. A hard time-based force-flat at `force_flat_et` ET (default "15:55"):
         once the current bar's own America/New_York clock time is at or after
         this time and this closure still believes a position is open, it
         exits unconditionally -- this is real same-session risk management,
         not just reliance on the caller engine's own end-of-day backstop.

    The evaluation at `entry_time_et` is one-shot per event day: whether or not
    it fires an entry, the closure marks that day's entry decision as made and
    will not re-evaluate entry again that session. Reference-price capture uses
    the same "first bar at or after the target time" rule as entry evaluation,
    for the same feed-gap robustness reason.

    State (event-day flag, reference price, one-shot entry-evaluated flag,
    open-position flag, entry price) is tracked in a plain dict closed over by
    `signal_fn`, and is defensively reset to a flat, no-reference state on every
    new calendar day this closure observes in `frame.index[-1].date()` -- since
    every trade this mechanism opens is same-day by construction (backstopped
    by the caller engine's own `max_hold_days=1`), this reset should be a no-op
    in correct operation, and exists purely as a safety net against this
    closure's own bookkeeping bugs, never as the primary exit path.

    Disclosed approximations:
      - The "14:00 ET price" and "14:15 ET price" are each taken from the first
        available minute bar's Close at or after that clock time, not
        necessarily an exact 14:00:00/14:15:00 bar (handles real feed gaps).
      - The entry price used for the profit-target comparison is this closure's
        own observed Close at the 14:15 decision bar, not the actual next-bar-
        open fill price the caller engine will use to execute the trade --
        real fill price will differ by one bar's worth of drift, and that
        drift is not modeled here.
      - `event_filter` ("FOMC statement day") and `max_hold_days` (1) from the
        candidate's own `parameters` are not read out of a `parameters` dict
        inside this closure because there is no such dict argument in this
        contract -- they are realized structurally instead: `event_filter` by
        requiring the caller to supply real FOMC dates via `event_dates`, and
        `max_hold_days` by this closure's own same-day-only state machine plus
        the caller engine's own independent `max_hold_days=1` backstop.
      - Scope: long-only per the day-trading track's scope -- the mechanism's
        own mirrored short entry (on an upward deviation) is never taken.

    `event_dates` is parsed into `datetime.date` objects with
    `date.fromisoformat`, the same parsing this module's import-time
    `_synthetic_date_match_self_check` exercises against made-up dates. No
    real FOMC date is hardcoded anywhere in this function.
    """
    import datetime as _dt

    event_date_set = {_dt.date.fromisoformat(d) for d in event_dates}
    ref_time = _dt.datetime.strptime(reference_time_et, "%H:%M").time()
    entry_time = _dt.datetime.strptime(entry_time_et, "%H:%M").time()
    flat_time = _dt.datetime.strptime(force_flat_et, "%H:%M").time()

    state = {
        "current_date": None,
        "is_event_day": False,
        "reference_price": None,
        "entry_evaluated": False,
        "position_open": False,
        "entry_price": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if frame.empty:
            return SignalResult()

        bar = frame.iloc[-1]
        ts = frame.index[-1]
        bar_date = ts.date()
        bar_time = ts.time()

        if state["current_date"] != bar_date:
            state["current_date"] = bar_date
            state["is_event_day"] = bar_date in event_date_set
            state["reference_price"] = None
            state["entry_evaluated"] = False
            state["position_open"] = False
            state["entry_price"] = None

        if not state["is_event_day"]:
            return SignalResult()

        close = float(bar["Close"])

        if state["position_open"]:
            if bar_time >= flat_time:
                state["position_open"] = False
                state["entry_price"] = None
                return SignalResult(exit=True)

            entry_price = state["entry_price"]
            if entry_price is not None and entry_price > 0:
                favorable_move_pct = (close - entry_price) / entry_price * 100.0
                if favorable_move_pct >= target_reversion_pct:
                    state["position_open"] = False
                    state["entry_price"] = None
                    return SignalResult(exit=True)

            return SignalResult()

        if state["reference_price"] is None and bar_time >= ref_time:
            state["reference_price"] = close

        if (
            state["reference_price"] is not None
            and not state["entry_evaluated"]
            and bar_time >= entry_time
        ):
            state["entry_evaluated"] = True
            reference_price = state["reference_price"]
            if reference_price > 0:
                deviation_pct = (close - reference_price) / reference_price * 100.0
                if deviation_pct <= -min_deviation_pct:
                    state["position_open"] = True
                    state["entry_price"] = close
                    return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- macro_release_jump_continuation ---
_TAIL_BARS = 400  # more than one full RTH session (390 minutes) + buffer


def _parse_hhmm(s: str) -> _time:
    h, m = s.split(":")
    return _time(int(h), int(m))


def make_macro_release_jump_continuation_signal_fn(
    event_dates: list,
    release_time_et: str = "08:30",
    jump_window_start_et: str = "08:29",
    jump_window_end_et: str = "08:35",
    entry_time_et: str = "08:35",
    min_jump_pct: float = 0.2,
    force_flat_et: str = "15:55",
) -> "SignalFn":
    """
    Implements the "Scheduled Macro-Release Jump Continuation" candidate
    (family_id=macro_release_jump_continuation), LONG side only.

    Real mechanism: on days carrying a scheduled 8:30 AM ET macro release
    (CPI or the monthly jobs report), measure the instrument's return over
    the 8:29-8:35 ET window bracketing the release; if the absolute move
    exceeds min_jump_pct (0.20%), enter at 8:35 ET in the direction of the
    jump and force-flatten by force_flat_et (15:55 ET) the same session.

    DISCLOSED APPROXIMATIONS:

    1. The provided minute-bar frame only contains Regular Trading Hours
       (09:30-16:00 ET). The real 8:30 AM ET release and its 8:29-8:35 AM ET
       reaction window occur entirely PRE-MARKET and are structurally
       unobservable in this data -- there is no way to read a pre-market tick
       from `frame`. As the closest causally-valid RTH analog, this
       implementation treats the day's own RTH open (the first bar's Open,
       09:30 ET) as standing in for the pre-release reference price ("8:29"),
       since the RTH open already embeds the full pre-market reaction to an
       8:30 AM release. The "jump" is then measured as the return from that
       RTH open to the close of the bar reached `signal_offset_minutes`
       minutes later, where `signal_offset_minutes` is DERIVED from the real
       supplied parameters (see below) rather than hardcoded -- this
       preserves the mechanism's real structure (a short bracketing window
       measuring an initial reaction, then testing whether it continues
       through the rest of the day) using only real, available RTH prices.

    2. `signal_offset_minutes` = max(jump_window_end_et - jump_window_start_et,
       entry_time_et - jump_window_start_et, in minutes). For this
       candidate's actual parameter values (window 08:29-08:35, entry 08:35)
       both terms equal 6, so the RTH analog window is 6 minutes: the jump is
       measured from the 09:30 bar's Open to the 09:36 bar's Close, and the
       entry signal fires on that same 09:36 bar. Per the engine's own
       next-open execution rule, the actual fill is the FOLLOWING bar's open
       (~09:37) -- the nearest causally valid execution to "enter as soon as
       the reaction window's move is known," since the jump itself cannot be
       known until the window-end bar has closed. release_time_et is used
       only as a build-time sanity check (asserting it falls inside the
       supplied window) rather than a per-bar comparison, since the release
       clock time itself is pre-market and unobservable the same way.

    3. This candidate's own listed tickers are SPY and QQQ. The mechanism
       names "SPY's return" specifically, but signal_fn only ever receives
       the OHLCV of the single instrument currently being backtested (SPY's
       own frame when testing SPY, QQQ's own frame when testing QQQ) -- there
       is no cross-instrument data available inside this closure. The jump is
       therefore measured on the traded instrument's own RTH open-window
       return in both cases, not literally SPY's tape when trading QQQ.

    4. LONG ONLY: the real mechanism enters in the direction of
       the jump (long on an up-jump, short on a down-jump). This
       implementation only ever fires an entry on a positive (upward) jump of
       at least min_jump_pct; a downward jump of the same magnitude is
       intentionally never traded here, since the engine's `direction` is a
       single whole-call parameter and the day-trading track deliberately
       scopes every candidate to its long side only.

    5. `event_dates` (real historical CPI / jobs-report ISO date strings,
       e.g. "2024-01-11") must be supplied by the caller -- no dates are
       fabricated or hardcoded here. `event_filter` (["CPI", "NFP/jobs
       report"]) is descriptive metadata about WHICH kind of release each
       date is; it cannot be checked from `frame` (there is no per-bar
       "release type" signal), and `event_dates` already fully encodes which
       real calendar days qualify for this backtest, so `event_filter` is
       not used in any comparison here -- it is subsumed by the caller's own
       curation of `event_dates`. `max_hold_days` is the engine's own
       force-close backstop (not this closure's responsibility) and likewise
       appears in no comparison here.

    A small synthetic self-check (using made-up, clearly non-real ISO dates)
    runs once at import time below to check the event-date matching logic:
    event_dates is parsed into `datetime.date` objects at factory-build time,
    because comparing `datetime.date` objects against a set of raw ISO
    strings would never match.

    Every bar, this closure tracks (in `_state`, a plain dict) whether it
    currently believes a position is open. Because `max_hold_days=1` and
    `force_flat_et` both guarantee same-day-only trades, this bookkeeping is
    deliberately simple (no multi-day trailing-stop state): it is reset
    defensively on any new-day boundary as a safety net against its own
    bugs, even though the mechanism itself should never carry state across
    days.
    """
    event_dates_set = frozenset(_date.fromisoformat(d) for d in event_dates)
    min_jump = float(min_jump_pct)

    release_time = _parse_hhmm(release_time_et)
    window_start_time = _parse_hhmm(jump_window_start_et)
    window_end_time = _parse_hhmm(jump_window_end_et)
    entry_time = _parse_hhmm(entry_time_et)
    force_flat_time = _parse_hhmm(force_flat_et)

    _start_min = window_start_time.hour * 60 + window_start_time.minute
    _end_min = window_end_time.hour * 60 + window_end_time.minute
    _entry_min = entry_time.hour * 60 + entry_time.minute
    _release_min = release_time.hour * 60 + release_time.minute

    if not (_start_min <= _release_min <= _end_min):
        raise ValueError(
            "release_time_et must fall within "
            "[jump_window_start_et, jump_window_end_et]"
        )

    window_span_minutes = max(1, _end_min - _start_min)
    entry_offset_minutes = max(0, _entry_min - _start_min)
    signal_offset_minutes = max(window_span_minutes, entry_offset_minutes)

    _state = {
        "position_open": False,
        "session_date": None,
        "traded_dates": set(),
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> "SignalResult":
        if frame.empty:
            return SignalResult()

        current_ts = frame.index[-1]
        current_date = current_ts.date()

        # Defensive new-day reset: a position must never survive into a new
        # calendar day in this closure's own bookkeeping (max_hold_days=1
        # plus force_flat_et both guarantee this at the engine level; this is
        # purely a safety net against our own bookkeeping bugs).
        if _state["session_date"] != current_date:
            _state["session_date"] = current_date
            _state["position_open"] = False

        # -- Exit branch: a position is (from our own tracking) currently open --
        if _state["position_open"]:
            if current_ts.time() >= force_flat_time:
                _state["position_open"] = False
                return SignalResult(exit=True)
            return SignalResult()

        # -- Entry branch: only ever consider entering on a real scheduled
        # macro-release date supplied by the caller, and only once per day --
        if current_date not in event_dates_set:
            return SignalResult()
        if current_date in _state["traded_dates"]:
            return SignalResult()

        tail = frame.iloc[-_TAIL_BARS:]
        today_rows = tail.loc[tail.index.date == current_date]

        if len(today_rows) <= signal_offset_minutes:
            # Not yet far enough into today's session for the RTH-analog
            # window to have closed -- insufficient real history, do
            # nothing rather than guess.
            return SignalResult()
        if len(today_rows) > signal_offset_minutes + 1:
            # We are past the window/entry bar for today (e.g. a data gap
            # skipped straight past it); do not fire a stale entry.
            return SignalResult()

        window_start_price = float(today_rows["Open"].iloc[0])
        window_end_price = float(today_rows["Close"].iloc[-1])

        if not np.isfinite(window_start_price) or window_start_price == 0.0:
            return SignalResult()
        if not np.isfinite(window_end_price):
            return SignalResult()

        jump_pct = (window_end_price / window_start_price - 1.0) * 100.0

        if jump_pct >= min_jump:
            _state["position_open"] = True
            _state["traded_dates"].add(current_date)
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


def _synthetic_date_match_self_check() -> None:
    """
    Small synthetic self-check proving the event-date matching
    logic used above is correct. The dates below are MADE UP for this test
    only -- they are NOT real CPI/NFP release dates and must never be
    treated as such.
    """
    fake_event_dates = ["2099-01-02", "2099-03-04"]
    event_dates_set = frozenset(_date.fromisoformat(d) for d in fake_event_dates)

    assert _date(2099, 1, 2) in event_dates_set
    assert _date(2099, 3, 4) in event_dates_set
    assert _date(2099, 1, 3) not in event_dates_set
    assert _date(2098, 1, 2) not in event_dates_set


_synthetic_date_match_self_check()


# --- closing_auction_pressure_scalp ---
def make_closing_auction_pressure_scalp_signal_fn(
    lookback_window_et: str = "15:30-15:45",
    signal_window_et: str = "15:45-15:58",
    volume_zscore_min: float = 1.5,
    force_flat_et: str = "15:59",
    lookback_days: int = 20,
    min_signal_bars: int = 3,
):
    """Late-session closing-auction pressure scalp (LONG side only).

    Each RTH day, while the current bar's ET clock time falls inside the
    `signal_window_et` window (default 15:45-15:58, both ends inclusive --
    treated as the literal 14 one-minute bars named by that label), the
    closure evaluates, on every such bar until it enters, whether the
    session so far shows an elevated-volume UP move consistent with
    closing-auction-routed buy pressure:

      - Directional: the signal window's own Open-to-Close return
        (open of its first bar to close of the current bar) is positive
        AND larger than the immediately preceding `lookback_window_et`
        window's (default 15:30-15:45) own Open-to-Close return. The
        lookback window's declared end (15:45) is treated EXCLUSIVE so it
        never overlaps the signal window's inclusive start -- a disclosed
        boundary interpretation, since the two labels literally share the
        15:45 endpoint.
      - Volume-elevated vs. the preceding window: the signal window's own
        per-minute volume rate so far (volume-so-far / minutes elapsed in
        the window) exceeds the preceding window's per-minute volume rate
        (its full volume / its full minute count). Using per-minute RATES
        rather than raw sums is a disclosed choice so a still-partial
        signal window is compared fairly against the preceding window's
        complete total.
      - Volume-elevated vs. the stock's own trailing `lookback_days`
        (default 20) trading-day history of this SAME clock window: the
        signal window's per-minute rate so far is scaled up to a
        full-window volume ESTIMATE (rate * the window's total minute
        count) and turned into a z-score against the trailing same-window
        daily volume totals (population std). Entry requires
        z >= volume_zscore_min. This projection-from-a-partial-window is
        a disclosed approximation -- it assumes the window's volume rate
        stays roughly steady for its remainder, which is the best
        estimate obtainable from `frame` alone without waiting for the
        window to fully close (waiting for the literal last bar of the
        window would leave only a single next-bar-open before the
        force-flat time, which the "scalp" framing does not require).
      - At least `min_signal_bars` bars of the signal window must already
        be observed (default 3) and at least 2 lookback-window bars and 5
        trailing historical days must be available, or the bar is treated
        as having insufficient history and no signal fires (never raises).

    On entry (bar i's signal_fn call returning entry=True), the actual
    fill happens at bar i+1's open per the engine's own next-open
    execution rule; the closure marks its own `in_position` bookkeeping
    True starting from that point so that every subsequent bar of the
    SAME calendar day is checked only for the force-flat exit: once the
    current bar's own ET clock time is >= force_flat_et (default 15:59),
    exit=True fires. No separate stop-loss or profit-target is defined by
    this mechanism -- the ONLY specified exit is the 15:59 force-flat, so
    none is invented here. If a new calendar date is EVER observed while
    the closure still believes a position is open (only reachable via a
    genuine data gap -- e.g. the signal-window bar that triggered entry
    turning out to be its own session's actual LAST bar, so the very next
    row in `frame` -- the engine's real entry/next-open bar -- lands on a
    new trading day), the closure immediately emits `SignalResult(exit=
    True)` and resets its own `in_position`/`entry_date` state, rather
    than silently resetting without signaling. A silent reset would
    desync this closure's OWN belief from the engine's real open
    position, permanently disabling this closure's own 15:59 force-flat
    check for that trade for the rest of its life -- leaving it entirely
    to the engine's separate `max_hold_days` backstop, which is real and
    trustworthy but is not a substitute for this factory's own bookkeeping
    being correct. In the ordinary case, with complete same-day bars past
    the signal window, this branch is never taken: the 15:59 check itself
    always fires same-day, before any date could roll over.

    Scope decisions: LONG side only -- the mechanism's implied mirrored
    short (an elevated-volume DOWN move) is deliberately not implemented,
    per the engine's whole-call `direction` scoping. `max_hold_days` is consumed directly by the backtest engine
    itself (the force-close-at-session-end backstop), not by this
    factory, so it is intentionally not a parameter here.

    Efficiency note: the (moderately expensive) per-day grouping and
    trailing-history z-score computation only runs on bars that fall
    inside the ~14-bar signal window and only while no position is
    already open -- not on every bar of every day -- and is bounded to
    `frame.tail(12000)` (roughly 30 RTH trading days of 1-minute bars,
    comfortably covering the 20-day lookback plus the current day) rather
    than re-scanning the full multi-year `frame` on every call.
    """

    def _parse_time(s: str) -> dt.time:
        h, m = s.strip().split(":")
        return dt.time(int(h), int(m))

    def _parse_window(window_str: str):
        start_s, end_s = window_str.split("-")
        return _parse_time(start_s), _parse_time(end_s)

    SIGNAL_START, SIGNAL_END = _parse_window(signal_window_et)
    LOOKBACK_START, _lookback_end_label = _parse_window(lookback_window_et)
    # Exclusive end at SIGNAL_START -- see docstring boundary note.
    FORCE_FLAT = _parse_time(force_flat_et)

    # Number of one-minute bars spanned by the signal window, inclusive of
    # both named endpoints (e.g. "15:45-15:58" -> 14 minutes: :45..:58).
    _sig_start_min = SIGNAL_START.hour * 60 + SIGNAL_START.minute
    _sig_end_min = SIGNAL_END.hour * 60 + SIGNAL_END.minute
    WINDOW_TOTAL_MINUTES = max(1, _sig_end_min - _sig_start_min + 1)

    state = {"in_position": False, "entry_date": None}

    def signal_fn(frame: pd.DataFrame, i: int):
        ts = frame.index[-1]
        t = ts.time()
        d = ts.date()

        # Safety net: never let tracked state believe a position survived
        # into a new calendar day. Only reachable via a genuine data gap
        # (the signal-window bar that triggered entry turning out to be
        # its session's actual LAST bar, so the very next row in `frame`
        # -- the engine's real entry/next-open bar -- lands on a new
        # trading day). Immediately emit exit=True rather than silently
        # resetting: a silent reset here would desync this closure's own
        # `in_position` belief from the engine's real open position,
        # permanently disabling THIS closure's own 15:59 force-flat check
        # for that trade (leaving it solely to the engine's separate
        # max_hold_days backstop, which is real but is not this factory's
        # own bookkeeping being correct).
        if state["in_position"] and state["entry_date"] is not None and d != state["entry_date"]:
            state["in_position"] = False
            state["entry_date"] = None
            return SignalResult(exit=True)

        if state["in_position"]:
            if t >= FORCE_FLAT:
                state["in_position"] = False
                state["entry_date"] = None
                return SignalResult(exit=True)
            return SignalResult()

        # Only look for new entries inside the signal window itself.
        if t < SIGNAL_START or t > SIGNAL_END:
            return SignalResult()

        recent = frame.tail(12000)
        idx_dates = recent.index.date
        idx_times = recent.index.time

        today_mask = idx_dates == d
        today_times = idx_times[today_mask]
        today_bars = recent[today_mask]

        sig_mask = (today_times >= SIGNAL_START) & (today_times <= SIGNAL_END)
        sig_bars = today_bars[sig_mask]
        elapsed = len(sig_bars)
        if elapsed < min_signal_bars:
            return SignalResult()

        lb_mask = (today_times >= LOOKBACK_START) & (today_times < SIGNAL_START)
        lb_bars = today_bars[lb_mask]
        if len(lb_bars) < 2:
            return SignalResult()

        sig_open = float(sig_bars["Open"].iloc[0])
        sig_close = float(sig_bars["Close"].iloc[-1])
        if sig_open == 0:
            return SignalResult()
        sig_return = (sig_close - sig_open) / sig_open

        lb_open = float(lb_bars["Open"].iloc[0])
        lb_close = float(lb_bars["Close"].iloc[-1])
        if lb_open == 0:
            return SignalResult()
        lb_return = (lb_close - lb_open) / lb_open

        sig_volume_so_far = float(sig_bars["Volume"].sum())
        sig_vol_rate = sig_volume_so_far / elapsed
        lb_vol_rate = float(lb_bars["Volume"].sum()) / len(lb_bars)

        # Trailing same-window historical daily volume totals (prior days
        # only -- never today, never any bar beyond i).
        past_mask = idx_dates < d
        past_bars = recent[past_mask]
        if past_bars.empty:
            return SignalResult()
        past_times = idx_times[past_mask]
        past_sig_mask = (past_times >= SIGNAL_START) & (past_times <= SIGNAL_END)
        past_sig = past_bars[past_sig_mask]
        if past_sig.empty:
            return SignalResult()

        daily_vol = past_sig.groupby(past_sig.index.date)["Volume"].sum()
        daily_vol = daily_vol.tail(lookback_days)
        if len(daily_vol) < 5:
            return SignalResult()

        mean_vol = float(daily_vol.mean())
        std_vol = float(daily_vol.std(ddof=0))
        if not np.isfinite(std_vol) or std_vol == 0:
            return SignalResult()

        projected_full_volume = sig_vol_rate * WINDOW_TOTAL_MINUTES
        vol_z = (projected_full_volume - mean_vol) / std_vol

        directional = sig_return > 0 and sig_return > lb_return
        elevated = (vol_z >= volume_zscore_min) and (sig_vol_rate > lb_vol_rate)

        if directional and elevated:
            state["in_position"] = True
            state["entry_date"] = d
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


# --- turtle_soup_failed_breakout_reversal_daytrade ---
def make_turtle_soup_failed_breakout_reversal_daytrade_signal_fn(
    lookback_days: int = 20,
    min_days_since_prior_extreme: int = 4,
    reclaim_confirmation_bar_interval_minutes: int = 5,
    force_flat_et: str = "15:55",
) -> SignalFn:
    """
    Turtle Soup failed-breakout reversal, LONG SIDE ONLY (the mechanism describes a mirrored
    short variant on a failed new-high breakout; per this engine's single whole-call
    `direction` parameter, only the long / failed-new-low-breakdown side is implemented here --
    a disclosed scope decision, not an oversight).

    Each session: using only COMPLETE prior sessions (resampling `frame`'s minute bars to
    daily lows via calendar-day grouping in the America/New_York tz already carried by
    `frame.index`), finds the lowest daily Low over the trailing `lookback_days` sessions
    ("the prior N-day low") and the session date on which that low occurred. Requires that
    extreme to be stale -- at least `min_days_since_prior_extreme` completed sessions must
    have elapsed between that extreme's session and today's session (counted within the same
    lookback window) -- otherwise no trade is considered for the day. For efficiency on
    multi-year minute-bar histories, this daily resample is computed once per NEW SESSION
    (not once per bar) and is bounded to a generous recent window
    (lookback_days + min_days_since_prior_extreme + 15 sessions' worth of RTH minutes, ~390
    bars/session) rather than re-scanning the entire multi-year history every session -- on
    any real, (near-)continuous RTH dataset this produces an identical result to scanning
    full history; it only differs, conservatively (skip the day, never crash or mis-signal),
    in the pathological case of a dataset with far fewer than ~390 bars/session.

    Intraday: tracks (in the closure's own per-session state, reset on every new calendar-day
    boundary as a safety net against this signal_fn's own bookkeeping) whether the current
    session's running Low has broken below the prior N-day low (`broken_below`). Once broken,
    watches for a "5-minute bar close" back above that level. Because the engine calls this
    function on every 1-minute bar rather than on true 5-minute bars, a 5-minute bar's close
    is approximated as the Close of whichever 1-minute bar lands on a wall-clock boundary
    measured every `reclaim_confirmation_bar_interval_minutes` minutes from the 09:30 ET
    session open (09:34, 09:39, 09:44, ...). This is exact when that boundary minute's bar
    exists in the feed (its Close IS the true 5-minute bar's close); if that exact minute is
    missing from the feed, that one confirmation opportunity is simply skipped and the next
    scheduled boundary is checked instead -- never a lookahead, never a crash. On the first
    such boundary bar after `broken_below` where Close > the prior N-day low, enters long
    (once per session only -- this is a single reversal setup, not a repeatable intraday
    signal).

    Exit: this candidate's own `parameters` name no stop-loss or profit target, only
    `force_flat_et` (default "15:55") -- so the only exit condition this signal_fn implements
    is firing exit=True once the current bar's own America/New_York clock time is at or after
    `force_flat_et` AND this closure still believes a position is open. The engine's own
    max_hold_days=1 last-bar force-close backstop remains the final safety net (e.g. covering
    a half trading day that ends before `force_flat_et` would otherwise fire). `max_hold_days`
    is deliberately not a parameter of this factory: per the day-trading track's contract it is consumed
    directly by the engine's own force-close logic, never by signal_fn itself.
    """
    window_days_buffer = lookback_days + min_days_since_prior_extreme + 15
    window_bars = window_days_buffer * 390

    flat_hour_str, flat_minute_str = force_flat_et.split(":")
    flat_hour = int(flat_hour_str)
    flat_minute = int(flat_minute_str)

    state = {
        "session_date": None,
        "session_start_i": None,
        "prior_low": None,
        "prior_low_valid": False,
        "broken_below": False,
        "entered_today": False,
        "position_open": False,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        current_ts = frame.index[-1]
        current_date = current_ts.date()

        if state["session_date"] != current_date:
            # New session boundary -- reset ALL per-session state defensively, so a bug in
            # this closure's own bookkeeping can never let a position "survive" into a new day.
            state["session_date"] = current_date
            state["session_start_i"] = i
            state["broken_below"] = False
            state["entered_today"] = False
            state["position_open"] = False
            state["prior_low"] = None
            state["prior_low_valid"] = False

            start_pos = max(0, i - window_bars)
            prior_slice = frame.iloc[start_pos:i]  # strictly BEFORE today's first bar
            if len(prior_slice) > 0:
                daily_low = prior_slice["Low"].resample("1D").min().dropna()
                if len(daily_low) >= lookback_days:
                    window = daily_low.tail(lookback_days)
                    prior_low_value = window.min()
                    extreme_date = window.idxmin().date()
                    sessions_since_extreme = int((window.index.date > extreme_date).sum())
                    if sessions_since_extreme >= min_days_since_prior_extreme:
                        state["prior_low"] = float(prior_low_value)
                        state["prior_low_valid"] = True

        today_frame = frame.iloc[state["session_start_i"]:]

        if state["prior_low_valid"] and not state["broken_below"]:
            session_low_so_far = today_frame["Low"].min()
            if session_low_so_far < state["prior_low"]:
                state["broken_below"] = True

        exit_signal = False
        if state["position_open"] and (current_ts.hour, current_ts.minute) >= (flat_hour, flat_minute):
            exit_signal = True
            state["position_open"] = False

        entry_signal = False
        if (
            not exit_signal
            and state["prior_low_valid"]
            and state["broken_below"]
            and not state["entered_today"]
        ):
            interval = max(1, int(reclaim_confirmation_bar_interval_minutes))
            elapsed_minutes = (current_ts.hour * 60 + current_ts.minute) - (9 * 60 + 30)
            is_bucket_close = elapsed_minutes >= 0 and (elapsed_minutes + 1) % interval == 0
            if is_bucket_close:
                current_close = today_frame["Close"].iloc[-1]
                if current_close > state["prior_low"]:
                    entry_signal = True
                    state["entered_today"] = True
                    state["position_open"] = True

        return SignalResult(entry=entry_signal, exit=exit_signal)

    return signal_fn


# --- selling_climax_capitulation_bounce_daytrade ---
def make_selling_climax_capitulation_bounce_daytrade_signal_fn(
    volume_multiple_vs_trailing20d: float = 3.0,
    intraday_decline_multiple_of_avg_range: float = 1.5,
    reversal_bar_interval_minutes: int = 5,
    force_flat_et: str = "15:55",
    checkpoint_et: str = "10:30",
    lookback_days: int = 20,
    profit_target_pct: float = 0.01,
    trailing_stop_pct: float = 0.005,
    max_hold_days: int = 1,
) -> "SignalFn":
    """Selling-Climax Capitulation Bounce (Day Trade), LONG ONLY.

    Once per session, at the first bar at/after `checkpoint_et`, flags a same-session
    "selling climax" when BOTH (a) cumulative 1-minute Volume from the session's own
    09:30 ET open through the checkpoint is >= `volume_multiple_vs_trailing20d`x the
    trailing `lookback_days`-day average of that same open-through-checkpoint cumulative
    volume (computed per prior day from the same real 1-minute bars sitting earlier in
    `frame`), and (b) the peak-to-trough decline achieved intraday by the checkpoint
    (today's own running intraday High-so-far minus today's own running Low-so-far) is
    >= `intraday_decline_multiple_of_avg_range`x the trailing `lookback_days`-day average
    daily (High-Low) range, computed by grouping `frame`'s own prior-day rows by
    calendar date. Requires at least min(5, lookback_days) valid prior trading days
    before a climax can ever be flagged (returns SignalResult() rather than raising
    when history is thin). Once flagged, the closure watches raw 1-minute bars for the
    next *complete* `reversal_bar_interval_minutes`-minute bar (aggregated from the raw
    1-minute rows, in bins aligned to the session's own 09:30 open) whose own aggregated
    Low is a NEW session low (below every 1-minute Low seen earlier in the session) and
    whose own aggregated Close is above its own bin midpoint ((High+Low)/2) -- that bin's
    own last bar fires entry=True, so the engine buys at the following bar's open exactly
    as the mechanism describes ("long entry follows that reversal bar"). At most ONE
    reversal-trigger entry is taken per session (interpreting "the entry trigger" as a
    single, one-shot event per climax); after that trade closes the closure does not
    re-arm for the day. Once in a tracked position, exits fire on whichever of these
    comes first: Close >= entry_price * (1 + profit_target_pct); Close <= (running post-
    entry High-water-mark) * (1 - trailing_stop_pct); or the current bar's own
    America/New_York clock time is at/after `force_flat_et`. New entries are also
    suppressed once the current bar's time is at/after `force_flat_et` (no point opening
    a position seconds before it would be force-flattened). The engine's own
    max_hold_days=1 backstop guarantees same-day force-closure regardless of this
    closure's own bookkeeping; `max_hold_days` is accepted here only for interface/
    documentation parity with the candidate's `parameters` dict and is not itself used
    in any comparison, since the engine (not the signal_fn) owns that backstop. All closure state is keyed to `frame.index[-1].date()`
    and defensively reset on every detected new-day boundary (including forcing any
    tracked open position closed in our own bookkeeping, since this mechanism is
    same-day-only and a real position can never legitimately survive that boundary).

    Disclosed approximations/interpretive choices (mechanism did not fully specify these):
      * `checkpoint_et` (the "given intraday checkpoint") is not named in `parameters`,
        so it is exposed as a factory argument, defaulting to "10:30" ET -- a defensible
        single mid-morning checkpoint by which a genuine capitulation flush would already
        be visible in cumulative volume and intraday decline.
      * `profit_target_pct` / `trailing_stop_pct` are not named in `parameters` either
        (the mechanism only says "a fixed profit target or trailing stop" with no
        values) -- exposed as factory arguments with modest intraday defaults (1.0%
        target, 0.5% trailing stop from the post-entry running High).
      * "Decline ... achieved intraday" is measured as today's own running peak-to-trough
        move (session High-so-far minus session Low-so-far) as of the checkpoint, not
        open-to-low, since a gap-down open alone should not count as an intraday decline.
      * The trailing 20-day averages (volume-by-checkpoint and daily range) are computed
        directly from the real 1-minute bars already present in `frame` for prior
        trading days -- no external data is used, and years of history are intentionally
        NOT rescanned every call: the trailing window is computed once per session (first
        bar of each new day) and cached in the closure for the rest of that session.
      * Long-only scope: the mechanism's own "long entry follows that reversal bar"
        language already scopes this candidate to the long/bounce side; no short mirror
        is described to disclose scoping away from.

    Does NOT need event_dates -- this candidate has no FOMC/macro-release dependency.
    """
    checkpoint_time = datetime.datetime.strptime(checkpoint_et, "%H:%M").time()
    force_flat_time = datetime.datetime.strptime(force_flat_et, "%H:%M").time()
    min_days_required = min(5, lookback_days)

    state = {
        "session_date": None,
        "session_low_so_far": None,
        "climax_checked_today": False,
        "climax_flagged_today": False,
        "entry_taken_today": False,
        "entry_pending": False,
        "position_open": False,
        "entry_price": None,
        "trail_high": None,
        "daily_stats_date": None,
        "avg_range_lookback": None,
        "avg_cumvol_ckpt_lookback": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> "SignalResult":
        if frame.empty:
            return SignalResult()

        current = frame.iloc[-1]
        current_ts = frame.index[-1]
        current_date = current_ts.date()
        current_time = current_ts.time()

        # --- defensive new-day reset (this mechanism is strictly same-day) ---
        if state["session_date"] != current_date:
            state["session_date"] = current_date
            state["session_low_so_far"] = None
            state["climax_checked_today"] = False
            state["climax_flagged_today"] = False
            state["entry_taken_today"] = False
            state["entry_pending"] = False
            # Safety net only: a real open position must never survive into a new
            # trading day under this same-day mechanism -- the engine's own
            # max_hold_days=1 backstop already guarantees this; this reset exists
            # purely to protect our own bookkeeping against a bug in it.
            state["position_open"] = False
            state["entry_price"] = None
            state["trail_high"] = None

        # --- if the previous bar fired entry=True, this bar is where the engine's
        # --- next-open execution actually happens; capture that open as entry_price ---
        if state["entry_pending"]:
            state["entry_pending"] = False
            state["position_open"] = True
            state["entry_price"] = float(current["Open"])
            state["trail_high"] = float(current["High"])

        # --- track the running real session low from raw 1-minute Lows ---
        low = float(current["Low"])
        if state["session_low_so_far"] is None or low < state["session_low_so_far"]:
            state["session_low_so_far"] = low

        # --- if a position is tracked open, only evaluate exits this bar ---
        if state["position_open"]:
            state["trail_high"] = max(state["trail_high"], float(current["High"]))
            close = float(current["Close"])
            target_price = state["entry_price"] * (1.0 + profit_target_pct)
            trail_stop_price = state["trail_high"] * (1.0 - trailing_stop_pct)
            force_flat = current_time >= force_flat_time
            if close >= target_price or close <= trail_stop_price or force_flat:
                state["position_open"] = False
                state["entry_price"] = None
                state["trail_high"] = None
                return SignalResult(exit=True)
            return SignalResult()

        # --- no position tracked open: at most one entry per session ---
        if state["entry_taken_today"]:
            return SignalResult()

        # --- compute (once per session, cached) the trailing lookback baselines ---
        if state["daily_stats_date"] != current_date:
            state["daily_stats_date"] = current_date
            # Bounded to a generous trailing window before any `.index.date`
            # work -- only the most recent `lookback_days` (default 20)
            # sessions are ever used below, so scanning the entire,
            # potentially 600,000-bar, history on every once-per-day call
            # (including the Python-level list comprehension further down)
            # is pure waste.
            lookback_bound_bars = (lookback_days + 15) * 400
            tail_frame = (
                frame.tail(lookback_bound_bars)
                if len(frame) > lookback_bound_bars
                else frame
            )
            prior_all = tail_frame[tail_frame.index.date < current_date]
            if prior_all.empty:
                state["avg_range_lookback"] = None
                state["avg_cumvol_ckpt_lookback"] = None
            else:
                unique_dates = pd.unique(prior_all.index.date)
                if len(unique_dates) > lookback_days:
                    recent_dates = unique_dates[-lookback_days:]
                else:
                    recent_dates = unique_dates
                recent_dates_set = set(recent_dates.tolist())
                mask = np.array(
                    [d in recent_dates_set for d in prior_all.index.date]
                )
                prior = prior_all[mask]

                daily_ranges = []
                daily_cumvols_at_ckpt = []
                for _, day_rows in prior.groupby(prior.index.date):
                    day_rows = day_rows.sort_index()
                    daily_ranges.append(
                        float(day_rows["High"].max() - day_rows["Low"].min())
                    )
                    upto_ckpt = day_rows[day_rows.index.time <= checkpoint_time]
                    if len(upto_ckpt) > 0:
                        daily_cumvols_at_ckpt.append(float(upto_ckpt["Volume"].sum()))

                if len(daily_ranges) >= min_days_required:
                    state["avg_range_lookback"] = float(np.mean(daily_ranges))
                else:
                    state["avg_range_lookback"] = None

                if len(daily_cumvols_at_ckpt) >= min_days_required:
                    state["avg_cumvol_ckpt_lookback"] = float(
                        np.mean(daily_cumvols_at_ckpt)
                    )
                else:
                    state["avg_cumvol_ckpt_lookback"] = None

        have_baselines = (
            state["avg_range_lookback"] is not None
            and state["avg_cumvol_ckpt_lookback"] is not None
            and state["avg_range_lookback"] > 0
            and state["avg_cumvol_ckpt_lookback"] > 0
        )

        # --- one-shot climax check, at the first bar at/after the checkpoint ---
        if not state["climax_checked_today"] and current_time >= checkpoint_time:
            state["climax_checked_today"] = True
            if have_baselines:
                # Only today's own bars are needed -- bound before the
                # `.index.date` scan.
                today_tail = frame.tail(_TAIL_BARS) if len(frame) > _TAIL_BARS else frame
                today_rows = today_tail[today_tail.index.date == current_date]
                today_upto_now = today_rows[today_rows.index.time <= current_time]
                cum_vol_today = float(today_upto_now["Volume"].sum())
                session_high_so_far = float(today_upto_now["High"].max())
                decline_so_far = session_high_so_far - state["session_low_so_far"]

                vol_ok = cum_vol_today >= (
                    volume_multiple_vs_trailing20d * state["avg_cumvol_ckpt_lookback"]
                )
                decline_ok = decline_so_far >= (
                    intraday_decline_multiple_of_avg_range
                    * state["avg_range_lookback"]
                )
                if vol_ok and decline_ok:
                    state["climax_flagged_today"] = True

        if not state["climax_flagged_today"]:
            return SignalResult()

        # --- no point arming a fresh entry right before the force-flat cutoff ---
        if current_time >= force_flat_time:
            return SignalResult()

        # --- scan for the reversal trigger on completed N-minute bins, aligned to open ---
        # NOT day-gated -- runs on every bar from the climax flag through
        # force-flat, potentially ~300+ bars on an active day, so bound
        # before the `.index.date` scan the same as the climax-check spot
        # above.
        today_tail = frame.tail(_TAIL_BARS) if len(frame) > _TAIL_BARS else frame
        today_rows = today_tail[today_tail.index.date == current_date].sort_index()
        n = reversal_bar_interval_minutes
        if len(today_rows) < n or len(today_rows) % n != 0:
            return SignalResult()

        bin_rows = today_rows.iloc[-n:]
        prior_rows = today_rows.iloc[:-n]
        if len(prior_rows) == 0:
            return SignalResult()

        bin_high = float(bin_rows["High"].max())
        bin_low = float(bin_rows["Low"].min())
        bin_close = float(bin_rows.iloc[-1]["Close"])
        bin_mid = (bin_high + bin_low) / 2.0

        prior_session_low = float(prior_rows["Low"].min())
        is_new_session_low = bin_low < prior_session_low
        closes_above_midpoint = bin_close > bin_mid

        if is_new_session_low and closes_above_midpoint:
            state["entry_taken_today"] = True
            state["entry_pending"] = True
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


SIGNAL_FACTORIES = {
    "orb_5min_atr_stop_scalp": make_orb_5min_atr_stop_scalp_signal_fn,
    "volume_filtered_orb_stocks_in_play_pooled": make_volume_filtered_orb_stocks_in_play_pooled_signal_fn,
    "vwap_trend_intraday_scalp": make_vwap_trend_intraday_scalp_signal_fn,
    "intraday_rsi2_reversion_daytrade": make_intraday_rsi2_reversion_daytrade_signal_fn,
    "overnight_gap_meanreversion_fade": make_overnight_gap_meanreversion_fade_signal_fn,
    "overnight_gap_fade_intraday_reversal": make_overnight_gap_fade_intraday_reversal_signal_fn,
    "intraday_half_hour_momentum_continuation": make_intraday_half_hour_momentum_continuation_signal_fn,
    "fomc_post_statement_reversion_fade": make_fomc_post_statement_reversion_fade_signal_fn,
    "macro_release_jump_continuation": make_macro_release_jump_continuation_signal_fn,
    "closing_auction_pressure_scalp": make_closing_auction_pressure_scalp_signal_fn,
    "turtle_soup_failed_breakout_reversal_daytrade": make_turtle_soup_failed_breakout_reversal_daytrade_signal_fn,
    "selling_climax_capitulation_bounce_daytrade": make_selling_climax_capitulation_bounce_daytrade_signal_fn,
}
