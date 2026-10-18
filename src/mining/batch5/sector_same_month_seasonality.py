"""Batch-5 candidate ``sector_same_month_seasonality`` (shape pooled_single_asset).

Same-calendar-month return seasonality across the 11 Select Sector SPDRs
(Heston & Sadka 2008; Keloharju, Linnainmaa & Nyberg 2016). At the close of
the last session of each month every pool member is scored by the mean of
its OWN total return in the coming calendar month over the prior 1..20
years (annual lags), using only complete months and requiring at least
five same-month observations. When at least nine members are eligible the
top three are held for the coming month: entered at the open of the
month's first session, exited at the close of its last session. The whole
cross-sectional rank is precomputed ONCE, strictly point-in-time, into a
selection schedule ``{(year, month): (ticker, ...)}``; each ETF's
per-ticker ``signal_fn`` only looks that schedule up, so the pooled runner
needs no cross-ticker access at run time.

``PARAMETERS`` is the candidate's ``"parameters"`` object, pasted
verbatim (never retyped); the driver checks canonical-JSON identity
against the catalog.

Interpretation choices
----------------------
1. "Last XNYS session of the month" is resolved from the ETF's own bar
   DATES: a bar is a month end when the ETF's next bar falls in a later
   calendar month. For the final bar of a frame (no next bar) the vendored
   NYSE session calendar (src/trading_calendar.py, 2026-2028) decides; a
   final bar outside that coverage is treated as not a month end. Future
   prices are never consulted for this -- only the calendar.
2. "Adjusted closes / total return" = the frame's ``Close`` column as the
   driver delivers it (data_pipeline fetches with ``auto_adjust=True``, so
   Close is the dividend-adjusted total-return close).
   ``month_return(e, m, y)`` = Close at the month-end bar of month m in
   year y / Close at the month-end bar of the preceding month - 1, defined
   only when both month-end bars exist AND the ETF's first bar precedes the
   first calendar day of month m in year y (the candidate's "complete
   months" condition, taken literally).
3. Consecutive selections. The pseudocode is taken literally:
   ``entry = last_of_month and selected_next``, ``exit = last_of_month``
   (unconditional), so every selected ETF-month is exactly one round
   trip (``one_trade_per_selected_etf_month``). This works with the
   mining engine because run_backtest, once flat at bar i, re-reads bar
   i-1's signal (``signal_fn(df.iloc[:i], i - 1)``): a month-end bar
   returning entry=True/exit=True therefore closes at that bar's close
   AND re-enters at the next bar's open, so a sector selected two months
   running is held both months as two separate trades.
   Merging consecutive selections into one holding period instead would
   rest on the false premise that the engine ignores an entry signal on
   an exit bar; it would pool materially fewer trades and change the
   per-trade distribution the DSR scores, so it is deliberately not done.
4. Tie-break "longer history first" = earlier first bar date, then ticker
   alphabetical; ranking is descending on the mean same-month return.
5. A schedule entry exists for every target month from the pool's second
   month of data through the month after the bucket's last bar; months with
   fewer than nine eligible members hold nothing.

Look-ahead and holdout blindness
--------------------------------
The schedule for target month t uses only month returns of month(t) in
years <= year(t) - 1, each of which ends at least eleven months before the
decision session. On top of that, closes are truncated to bars on or
before the last bar of ``index`` before the schedule is built, so a
holdout bucket can never see anything after itself. Month-end flags use
only bar dates. Per-bar cost inside ``signal_fn``: one date equality
check and two boolean array lookups (everything else is precomputed in the
factory).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import trading_calendar
from mining.backtest_engine import SignalResult

FAMILY_ID = "sector_same_month_seasonality"
SHAPE = "pooled_single_asset"
TRADED = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
          "XLRE", "XLC"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {'seasonal_signal': "mean of the ETF's own total return in calendar month m over years y-1 ... "
                    'y-20, using only complete months (both bounding month-end closes exist '
                    "and the ETF's first bar precedes the start of that month)",
 'min_same_month_observations': 5,
 'lookback_years_max': 20,
 'hold_top_n': 3,
 'min_eligible_etfs_to_trade': 9,
 'tie_break': 'longer history first, then ticker alphabetical',
 'decision': 'last XNYS session of the prior month',
 'entry': 'open of the first session of the month',
 'exit': 'close of the last session of the month',
 'one_trade_per_selected_etf_month': True,
 'direction': 'long',
 'execution_note': 'the cross-sectional rank is precomputed once from the 11 adjusted-close '
                   'series into a point-in-time selection schedule (month -> set of 3 tickers, '
                   "using only closes on or before each decision session); each ETF's "
                   'per-ticker signal_fn then only looks up that schedule, so the '
                   'pooled_single_asset runner needs no cross-ticker access at run time -- the '
                   'same pattern as the calendar-driven candidates'}
EXPECTED_MIN_TRADES = 700

MIN_SAME_MONTH_OBSERVATIONS = int(PARAMETERS["min_same_month_observations"])
LOOKBACK_YEARS_MAX = int(PARAMETERS["lookback_years_max"])
HOLD_TOP_N = int(PARAMETERS["hold_top_n"])
MIN_ELIGIBLE_ETFS_TO_TRADE = int(PARAMETERS["min_eligible_etfs_to_trade"])


# ------------------------------------------------------------------ helpers

def _normalize_dates(index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("bar dates must be sorted and unique")
    return dates


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def month_end_flags(dates, sessions=None) -> np.ndarray:
    """Boolean array: True where the bar is the last bar of its calendar
    month (interpretation choice 1). Uses bar dates only. The final bar is
    resolved against the vendored NYSE calendar when covered; outside
    coverage it is False."""
    dates = _normalize_dates(dates)
    n = len(dates)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags
    years = dates.year.to_numpy()
    months = dates.month.to_numpy()
    flags[:-1] = (years[1:] != years[:-1]) | (months[1:] != months[:-1])
    last = dates[-1].date()
    if sessions is None:
        sessions = trading_calendar.load_default_sessions()
    if sessions[0] <= last <= sessions[-1]:
        later = [session for session in sessions if session > last]
        if later:
            flags[-1] = (later[0].year, later[0].month) != (last.year, last.month)
    return flags


def _month_end_closes(frame: pd.DataFrame, cutoff: pd.Timestamp,
                      sessions) -> tuple[dict, pd.Timestamp]:
    """{(year, month): month-end Close} for bars on or before ``cutoff``,
    plus the ETF's first bar date."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame.columns:
        raise ValueError("each pool frame must be a non-empty DataFrame with a Close column")
    dates = _normalize_dates(frame.index)
    closes = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    keep = month_end_flags(dates, sessions) & (dates <= cutoff)
    out = {}
    for date, close in zip(dates[keep], closes[keep]):
        if np.isfinite(close) and close > 0:
            out[(date.year, date.month)] = float(close)
    return out, dates[0]


def build_schedule(frames, cutoff, *, sessions=None) -> dict:
    """Point-in-time selection schedule ``{(year, month): (ticker, ...)}``
    for every target month up to the month after ``cutoff``. The target
    month (y, m), decided at the close of the last session of the month
    before it, scores each member by the mean of month_return(e, m, y - k)
    over k = 1..LOOKBACK_YEARS_MAX where defined; members with fewer than
    MIN_SAME_MONTH_OBSERVATIONS observations are ineligible; fewer than
    MIN_ELIGIBLE_ETFS_TO_TRADE eligible members -> an empty tuple."""
    missing = [ticker for ticker in TRADED if ticker not in frames]
    if missing:
        raise ValueError(f"{FAMILY_ID}: frames missing pool members {missing}")
    if sessions is None:
        sessions = trading_calendar.load_default_sessions()
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    cutoff = cutoff.normalize()
    closes_by, first_bar = {}, {}
    for ticker in TRADED:
        closes_by[ticker], first_bar[ticker] = _month_end_closes(
            frames[ticker], cutoff, sessions)
    keys = [key for closes in closes_by.values() for key in closes]
    if not keys:
        return {}
    target = _next_month(*min(keys))
    last_target = _next_month(cutoff.year, cutoff.month)
    schedule = {}
    while target <= last_target:
        year, month = target
        scored = []
        for ticker in TRADED:
            closes = closes_by[ticker]
            first = first_bar[ticker]
            observations = []
            for lag in range(1, LOOKBACK_YEARS_MAX + 1):
                key = (year - lag, month)
                prev = _prev_month(*key)
                if (key in closes and prev in closes
                        and first < pd.Timestamp(year=key[0], month=key[1], day=1)):
                    observations.append(closes[key] / closes[prev] - 1.0)
            if len(observations) >= MIN_SAME_MONTH_OBSERVATIONS:
                scored.append((-float(np.mean(observations)), first, ticker))
        if len(scored) >= MIN_ELIGIBLE_ETFS_TO_TRADE:
            scored.sort()
            schedule[target] = tuple(ticker for _, _, ticker in scored[:HOLD_TOP_N])
        else:
            schedule[target] = ()
        target = _next_month(*target)
    return schedule


# ------------------------------------------------------------------ factory

def make_signal_fn(ticker: str, index, frames):
    """Per-ticker SignalFn for the pooled runner (contract: called once per
    pool member with that member as ``ticker``; ``index`` is the bucket the
    engine runs over; ``frames`` holds full-history OHLC for all of TRADED)."""
    if ticker not in TRADED:
        raise ValueError(f"{FAMILY_ID}: {ticker!r} is not a pool member")
    if ticker not in frames:
        raise ValueError(f"{FAMILY_ID}: frames missing pool members [{ticker!r}]")
    bucket = _normalize_dates(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index is empty")
    sessions = trading_calendar.load_default_sessions()
    cutoff = bucket[-1]
    schedule = build_schedule(frames, cutoff, sessions=sessions)

    own_dates = _normalize_dates(frames[ticker].index)
    own_flags = pd.Series(month_end_flags(own_dates, sessions), index=own_dates)
    last_of_month = own_flags.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    selected_next = np.fromiter(
        (ticker in schedule.get(_next_month(date.year, date.month), ())
         for date in bucket), dtype=bool, count=len(bucket))
    entry_flags = last_of_month & selected_next
    exit_flags = last_of_month.copy()   # unconditional month-end exit (pseudocode)
    bucket_values = bucket.to_numpy()
    n = len(bucket)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n:
            raise IndexError(f"{FAMILY_ID}: bar position {i} outside the bucket of {n} bars")
        today = pd.Timestamp(frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        if today.normalize().to_datetime64() != bucket_values[i]:
            raise ValueError(
                f"{FAMILY_ID}: frame ends at {today.date()} but position {i} "
                f"of the bucket is {pd.Timestamp(bucket_values[i]).date()}")
        return SignalResult(entry=bool(entry_flags[i]), exit=bool(exit_flags[i]))

    signal_fn.schedule = schedule
    signal_fn.entry_flags = entry_flags
    signal_fn.exit_flags = exit_flags
    signal_fn.last_of_month = last_of_month
    return signal_fn
