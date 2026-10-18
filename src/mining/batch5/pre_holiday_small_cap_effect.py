"""Batch5 candidate ``pre_holiday_small_cap_effect`` (shape ``single_long``).

Hold IWM only across the trading session immediately preceding each
scheduled NYSE full-day closure (Ariel 1990; Lakonishok & Smidt 1988; Ko &
Yang 2024 for the small-firm persistence). Flat on every other session.

Rule, exactly as catalogued (the catalog entry's ``rule_pseudocode``)::

    CAL = XNYS sessions from exchange_calendars, intersected with IWM's bar index
    AD_HOC = PARAMETERS["excluded_ad_hoc_closures"]
    is_pre_holiday(s): n = next session after s in CAL
                       closed = weekdays strictly between s and n
                       return len(closed) >= 1 and not any(d in AD_HOC for d in closed)
    signal_fn(frame, i): today = frame.index[-1]
        entry = is_pre_holiday(next_session(today))   # engine buys at the pre-holiday open
        exit  = is_pre_holiday(today)                 # see interpretation choice 2
    run_backtest(IWM, signal_fn, direction='long', time_stop=1, cost_bps=measured)

Everything is precomputed ONCE in the factory from the full-history frame
(calendar arithmetic only -- no price enters the signal), reindexed to the
bucket ``index``, and looked up by position per bar (a few array reads).

Interpretation choices
----------------------
1. **Calendar source.** ``exchange_calendars`` XNYS is the candidate's stated
   ``calendar_source`` and is used directly, intersected with the traded
   frame's own bar index as the pseudocode requires. The repo's vendored
   ``src/trading_calendar.py`` calendar covers 2026-2028 only (752 sessions)
   and fails closed outside that range, so it cannot span IWM's history
   (2000-05-26 onward); it is therefore NOT used here. The intersection
   drops any bar that is not an XNYS session, but it cannot restore a
   session MISSING from the traded frame: a missing bar leaves a weekday
   gap that this rule reads as a closure. The frame must therefore contain
   every XNYS session in its range. The rule is: a session is pre-holiday
   when at least one weekday lies strictly between it and the next session
   in that calendar, unless any of those weekdays is an ad hoc market
   closure listed in ``PARAMETERS["excluded_ad_hoc_closures"]``.
2. **Exit timing (an engine constraint, not a rule change).** The pseudocode
   exits at the pre-holiday session's own close (``exit = is_pre_holiday(
   today)``) after entering at that session's open. ``backtest_engine.
   run_backtest`` never evaluates ``exit`` on the entry bar ("exit checks
   begin the bar after entry"), so that literal exit can never fire, and the
   driver's ``run_single_asset_windows`` forwards no ``time_stop``. The
   earliest exit the engine can execute is the close of the session AFTER
   the pre-holiday session -- which is exactly what the pseudocode's own
   ``time_stop=1`` produces in this engine. That one-bar hold is therefore
   expressed through the signal: ``exit`` is True on the pre-holiday session
   (the literal term, kept for fidelity) OR on the session immediately after
   one. Resulting trade: pre-holiday session OPEN -> post-holiday session
   CLOSE, held == 1, one trade per scheduled holiday. The entry side (the
   catalog's distinctness note says the next-open entry "captures only the
   open-to-close part") is untouched. No parameter is added or tuned.
3. **Closures spanning a scheduled AND an ad hoc day** (2007-01-01 New Year
   + 2007-01-02 Ford mourning) are excluded, per the literal ``not any(...)``.
4. **Ad hoc closures and causality.** Excluding an unscheduled closure is
   what a real-time trader would have done anyway (no holiday was scheduled
   on the signal date), so the exclusion list introduces no look-ahead in
   the direction of taking a trade. Scheduled holidays are known years
   ahead: only their DATES are used, never any future price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "pre_holiday_small_cap_effect"
SHAPE = "single_long"
TRADED = ["IWM"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Copied verbatim from the catalog entry's "parameters" (driver drift check).
PARAMETERS = {
    "holiday_set": "scheduled NYSE full-day closures only: New Year's Day, Martin Luther King Jr. Day (NYSE since 1998), Presidents' Day, Good Friday, Memorial Day, Juneteenth (since 2022), Independence Day, Labor Day, Thanksgiving Day, Christmas Day, with NYSE observance rules",
    "calendar_source": (
        "exchange_calendars XNYS, restricted to IWM's own bar index"
    ),
    "excluded_ad_hoc_closures": [
        "2001-09-11",
        "2001-09-12",
        "2001-09-13",
        "2001-09-14",
        "2004-06-11",
        "2007-01-02",
        "2012-10-29",
        "2012-10-30",
        "2018-12-05",
        "2025-01-09",
    ],
    "entry": "open of the pre-holiday session (entry signal fires at the close of the session before it)",
    "exit": "close of the pre-holiday session",
    "hold_sessions": 1,
    "time_stop": 1,
    "direction": "long",
    "decision_frequency": "one trade per scheduled holiday",
}
EXPECTED_MIN_TRADES = 200

XNYS = "XNYS"
AD_HOC_CLOSURES = frozenset(
    pd.Timestamp(day) for day in PARAMETERS["excluded_ad_hoc_closures"])


# ------------------------------------------------------------------ calendar

def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def xnys_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """XNYS sessions in [start, end] from exchange_calendars (tz-naive)."""
    import exchange_calendars as xc  # local import: heavy, only needed here

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end <= start:
        end = start + pd.Timedelta(days=1)
    calendar = xc.get_calendar(XNYS, start=start, end=end)
    sessions = pd.DatetimeIndex(calendar.sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def session_calendar(frame_index) -> pd.DatetimeIndex:
    """``CAL`` of the pseudocode: XNYS sessions intersected with the traded
    frame's bar index, sorted and unique."""
    bars = _normalize_index(frame_index).unique().sort_values()
    if len(bars) == 0:
        return bars
    sessions = xnys_sessions(bars[0], bars[-1])
    return bars.intersection(sessions).sort_values()


def pre_holiday_mask(cal: pd.DatetimeIndex) -> np.ndarray:
    """``is_pre_holiday`` for every position of ``cal``: at least one weekday
    strictly between this session and the next one in ``cal``, and none of
    those weekdays is an excluded ad hoc closure. The last session has no
    successor and is never pre-holiday."""
    cal = _normalize_index(cal)
    n = len(cal)
    mask = np.zeros(n, dtype=bool)
    if n < 2:
        return mask
    days = cal.values.astype("datetime64[D]")
    # weekdays in [s + 1 day, n) == weekdays strictly between s and n
    between = np.busday_count(days[:-1] + np.timedelta64(1, "D"), days[1:])
    for j in np.flatnonzero(between >= 1):
        closed = pd.bdate_range(cal[j] + pd.Timedelta(days=1),
                                cal[j + 1] - pd.Timedelta(days=1))
        mask[j] = not any(day in AD_HOC_CLOSURES for day in closed)
    return mask


def signal_series(cal: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Entry/exit booleans indexed by ``cal`` (see interpretation choice 2)."""
    cal = _normalize_index(cal)
    pre = pre_holiday_mask(cal)
    entry = np.zeros(len(cal), dtype=bool)
    exit_ = np.zeros(len(cal), dtype=bool)
    if len(cal):
        entry[:-1] = pre[1:]            # is_pre_holiday(next_session(today))
        exit_[:] = pre                  # literal: is_pre_holiday(today)
        exit_[1:] |= pre[:-1]           # time_stop=1: the session after one
    return (pd.Series(entry, index=cal, name="entry"),
            pd.Series(exit_, index=cal, name="exit"))


# ------------------------------------------------------------------- factory

def make_signal_fn(ticker: str, index, frames):
    """SignalFn for the engine: ``index`` is the bucket the engine runs over;
    ``frames[ticker]`` is the full-history OHLC frame (calendar only used)."""
    if ticker not in frames:
        raise KeyError(f"{FAMILY_ID}: no frame for {ticker!r} in frames")
    bucket = _normalize_index(index)
    cal = session_calendar(frames[ticker].index)
    entry_s, exit_s = signal_series(cal)
    entry_arr = entry_s.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    exit_arr = exit_s.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    position_of = {ts: j for j, ts in enumerate(bucket)}
    n = len(bucket)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        today = frame.index[-1]
        if getattr(today, "tzinfo", None) is not None:
            today = today.tz_localize(None)
        if 0 <= i < n and bucket[i] == today:
            j = i
        else:                            # defensive: resolve by date instead
            j = position_of.get(pd.Timestamp(today).normalize())
            if j is None:
                return SignalResult()
        return SignalResult(entry=bool(entry_arr[j]), exit=bool(exit_arr[j]))

    return signal_fn
