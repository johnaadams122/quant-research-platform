"""Candidate: overnight_intraday_tilt_pooled_largecap (pooled_single_asset).

Overnight-minus-intraday return continuation, pooled across individual large-cap
stocks (Lou, Polk & Skouras 2019, JFE 134(1):192-213 -- "A Tug of War: Overnight
Versus Intraday Expected Returns"). Their headline, most-cited, standalone
per-stock decile-sort result (Table 1, t=16.57 on the hedge portfolio) ranks
stocks on the trailing ONE-MONTH (21-trading-day) overnight return (``ret_night``)
and the trailing one-month intraday return (``ret_day``) -- these two names are
given explicitly in the paper's Table 4 regressions. This family combines both
into one cross-sectional spread, ``ret_night_21d - ret_day_21d``, and ranks the
existing 518-name large-cap universe monthly, going long the top decile
(equal-weighted, continuous hold while a name stays in the top decile -- the
same TSMOM-style state machine as ``large_cap_single_stock_pooled_tsmom.py``,
not a forced monthly round trip).

Two OTHER constructions in the same paper are not used by this design:
decomposing standard 12-1 price momentum into its overnight-only leg
(Sharpe 0.77 vs. 0.31 for the full close-to-close version) needs an
overnight-only EXECUTION capability (hold flat intraday, only capture the
close-to-open leg) that this engine's continuous-hold contract does not have.
A 60-month half-life EWMA version of the same spread (the paper's own
``TugOfWar`` regime-timing variable) is not itself a standalone per-stock
decile-sort strategy in the source paper, and its window is impractical against
this repo's much shorter per-trade holdout.

Caveat: a 2026 NY Fed follow-up is reported to find that the closest-studied
version of this class of edge decayed to roughly zero in 2021-2025, so the
published edge may not persist.

Interpretation choices
-----------------------
* "Auto-adjusted daily close (total return)" / Open: the driver's frames come
  from ``data_pipeline.fetch_ohlc`` (``yfinance ... auto_adjust=True``), which
  applies the SAME cumulative split+dividend adjustment factor to the whole
  OHLC bar for a given date -- so ``Open_t`` and ``Close_{t-1}`` are mutually
  consistent (no artificial overnight-return jump from a stock split). A name
  with an ex-dividend date inside the trailing 21-day window will show a real,
  economically-correct depression in its measured overnight return that day --
  not an artifact, the same convention ``ex_dividend_day_capture_pooled.py``
  already treats ex-div mechanics as real, not noise.
* Reference month-end: like ``sector_same_month_seasonality.py``, the bulk of
  month-end determination comes from each ticker's OWN bar dates (a bar is a
  month-end when its own next bar falls in a later calendar month); the
  vendored NYSE session calendar is consulted only for a truncated frame's
  final bar, which has no "next bar" to compare against.
* Cross-sectional ranking is keyed by ``(year, month)`` tuples (not raw dates),
  since every pool member trades the same NYSE/Nasdaq calendar -- their
  month-end BAR DATES coincide whenever both have been trading through that
  month, exactly as ``sector_same_month_seasonality.build_schedule`` already
  relies on.
* A ticker is eligible for a given month-end only if it has >= 21 valid
  (finite, positive Open/Close, with a valid prior close) trailing daily
  component returns ending on/before that bar -- otherwise its signal is
  undefined and it is excluded from that month's ranking, exactly like
  ``large_cap_single_stock_pooled_tsmom.decision_dates``'s "no decision
  without N prior periods" pattern, just at 21 days instead of 12 months.
* TSMOM-style state machine, not ``sector_same_month_seasonality``'s forced
  monthly round trip: ``entry_dates`` = a ticker's own month-end bars where it
  is in that month's top-decile membership; ``exit_dates`` = the bar
  immediately after a month-end bar where it is NOT a member (mirrors
  ``large_cap_single_stock_pooled_tsmom.decision_dates``'s FLAT ->
  exit-next-bar convention exactly). A name selected two months running is
  held continuously as one trade, not two.
* Pool: same 518-name ``TRADED`` list as ``large_cap_single_stock_pooled_
  tsmom.py``, copied verbatim -- same resolved pool, same OHLC cache, no
  re-curation.
* ``min_eligible_names_to_trade`` (100) and ``EXPECTED_MIN_TRADES`` (3000) are
  both configured placeholders, not measured values (this repo's disclosed-
  default convention for an invented constant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import trading_calendar
from mining.backtest_engine import SignalResult
from mining.batch5.large_cap_single_stock_pooled_tsmom import TRADED as _TSMOM_TRADED

FAMILY_ID = "overnight_intraday_tilt_pooled_largecap"
SHAPE = "pooled_single_asset"
TRADED = list(_TSMOM_TRADED)
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied verbatim from the catalog entry's "parameters" (byte-identical as
# canonical JSON; the batch5 driver's parameter_drift check refuses a mismatch).
PARAMETERS = {
    "compounding": "product_of_(1+daily_component)_minus_1_over_trailing_21_sessions",
    "decision_frequency": "monthly_at_reference_month_end",
    "entry": "next_session_open",
    "exit": "next_session_close",
    "intraday_return": "Close_t / Open_t - 1",
    "lookback_trading_days": 21,
    "min_eligible_names_to_trade": 100,
    "overnight_return": "Open_t / Close_{t-1} - 1",
    "pool": (
        "the pipeline's cached expanded universe: current S&P 500 + NASDAQ "
        "100 constituents after dedup, with cached OHLC and dated bid/ask "
        "quotes from the pipeline's universe cache. Same resolved pool as "
        "large_cap_single_stock_pooled_tsmom. Survivorship caveat: current "
        "constituents only, same convention as every other pooled family "
        "drawn from this universe."
    ),
    "pooling": "all_pool_trades_concatenated_one_trial",
    "position_persistence": "continuous_while_in_top_decile",
    "signal": "ret_night_21d_minus_ret_day_21d",
    "tie_break": "longer history first, then ticker alphabetical",
    "top_decile_fraction": 0.10,
}
EXPECTED_MIN_TRADES = 3000

LOOKBACK = int(PARAMETERS["lookback_trading_days"])
TOP_DECILE_FRACTION = float(PARAMETERS["top_decile_fraction"])
MIN_ELIGIBLE_NAMES_TO_TRADE = int(PARAMETERS["min_eligible_names_to_trade"])

_ENTRY = SignalResult(entry=True, exit=False)
_EXIT = SignalResult(entry=False, exit=True)
_NONE = SignalResult()

# {cutoff_ns: {"frames": <identity-checked dict>, "schedule": {...}}} -- the
# identity check (not just a cutoff match) defends against id() reuse after
# garbage collection, not just a stale cutoff.
_schedule_cache: dict = {}


def reset_diagnostics() -> None:
    _schedule_cache.clear()


def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def month_end_flags(dates, sessions=None) -> np.ndarray:
    """Boolean array: True where the bar is the last bar of its calendar
    month, using bar dates only (copied from ``sector_same_month_
    seasonality.month_end_flags`` -- a generic dates-only helper). The final
    bar of a truncated frame is resolved against the vendored NYSE session
    calendar when covered; outside coverage it is False."""
    dates = _normalize_index(dates)
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


def _log_components(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(log(1+overnight_t), log(1+intraday_t)) for every bar t of ``frame``
    (columns Open, Close; index ascending trading days). NaN at t=0 and
    wherever Open/Close/prior-Close aren't finite and positive."""
    open_ = pd.to_numeric(frame["Open"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    prev_close = close.shift(1)
    valid = (open_ > 0) & (close > 0) & (prev_close > 0)
    log_overnight = pd.Series(np.nan, index=frame.index, dtype=float)
    log_intraday = pd.Series(np.nan, index=frame.index, dtype=float)
    log_overnight[valid] = np.log(open_[valid] / prev_close[valid])
    log_intraday[valid] = np.log(close[valid] / open_[valid])
    return log_overnight, log_intraday


def trailing_signal(frame: pd.DataFrame, *, lookback: int = LOOKBACK) -> pd.Series:
    """``ret_night_{lookback}d(t) - ret_day_{lookback}d(t)`` for every bar t of
    ``frame``; NaN wherever fewer than ``lookback`` valid trailing daily
    components exist ending at (and including) t.

    Free internal-consistency invariant (asserted in this module's tests):
    ``(1+ret_night_Nd(t)) *
    (1+ret_day_Nd(t)) == Close_t / Close_{t-N}`` exactly, since
    ``(1+overnight_i)(1+intraday_i) = Close_i/Close_{i-1}`` telescopes across
    the window in log space.
    """
    log_overnight, log_intraday = _log_components(frame)
    roll_night = log_overnight.rolling(lookback, min_periods=lookback).sum()
    roll_day = log_intraday.rolling(lookback, min_periods=lookback).sum()
    ret_night = np.expm1(roll_night)
    ret_day = np.expm1(roll_day)
    return ret_night - ret_day


def _first_valid_date(frame: pd.DataFrame) -> pd.Timestamp:
    close = pd.to_numeric(frame["Close"], errors="coerce")
    valid = close[close > 0]
    if valid.empty:
        return pd.Timestamp.max
    return _normalize_index(valid.index)[0]


def build_target_schedule(frames: dict, cutoff, *, sessions=None) -> dict:
    """``{(year, month): frozenset(tickers in the top decile that month)}``
    for every calendar month-end <= ``cutoff``, across every ``TRADED``
    member present in ``frames`` (a member absent from ``frames``, or with no
    valid signal ever, is simply skipped -- never raises, matching this
    repo's graceful-degradation convention for a young or delisted pool
    member). Fewer than ``MIN_ELIGIBLE_NAMES_TO_TRADE`` scored members in a
    given month -> an empty membership set for that month, not an error."""
    if sessions is None:
        sessions = trading_calendar.load_default_sessions()
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    cutoff = cutoff.normalize()

    per_ticker_month_signal: dict[str, dict[tuple[int, int], float]] = {}
    first_bar: dict[str, pd.Timestamp] = {}
    for ticker in TRADED:
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            continue
        if "Open" not in frame.columns or "Close" not in frame.columns:
            continue
        dates = _normalize_index(frame.index)
        frame = frame.set_axis(dates)
        frame = frame.loc[frame.index <= cutoff]
        if frame.empty:
            continue
        signal = trailing_signal(frame)
        me_flags = month_end_flags(frame.index, sessions)
        keep = me_flags & signal.notna().to_numpy()
        if not keep.any():
            continue
        kept_dates = frame.index[keep]
        kept_values = signal.to_numpy()[keep]
        per_ticker_month_signal[ticker] = {
            (d.year, d.month): float(v) for d, v in zip(kept_dates, kept_values)
        }
        first_bar[ticker] = _first_valid_date(frame)

    all_months = sorted(
        {key for scores in per_ticker_month_signal.values() for key in scores})
    schedule: dict[tuple[int, int], frozenset] = {}
    for year, month in all_months:
        scored = []
        for ticker, scores in per_ticker_month_signal.items():
            value = scores.get((year, month))
            if value is not None:
                scored.append((-value, first_bar[ticker], ticker))
        if len(scored) >= MIN_ELIGIBLE_NAMES_TO_TRADE:
            scored.sort()
            k = max(1, round(TOP_DECILE_FRACTION * len(scored)))
            schedule[(year, month)] = frozenset(ticker for _, _, ticker in scored[:k])
        else:
            schedule[(year, month)] = frozenset()
    return schedule


def decision_dates(frame: pd.DataFrame, ticker: str, schedule: dict, *,
                    sessions=None):
    """TSMOM-style ``(entry_dates, exit_dates)`` for one ticker given the
    already-built cross-sectional ``schedule``. ``entry_dates``: the
    ticker's own month-end bars where it is in that month's top-decile
    membership (the engine buys at the next open). ``exit_dates``: the bar
    immediately after a month-end bar where it is NOT a member (the engine
    sells at that bar's close) -- mirrors ``large_cap_single_stock_pooled_
    tsmom.decision_dates``'s FLAT -> exit-next-bar convention exactly."""
    if sessions is None:
        sessions = trading_calendar.load_default_sessions()
    dates = _normalize_index(frame.index)
    n = len(dates)
    if n == 0:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([])
    flags = month_end_flags(dates, sessions)
    positions = np.flatnonzero(flags)
    long_mask = np.zeros(len(positions), dtype=bool)
    for idx, pos in enumerate(positions):
        d = dates[pos]
        long_mask[idx] = ticker in schedule.get((d.year, d.month), frozenset())
    entry_dates = dates[positions[long_mask]]
    exit_pos = positions[~long_mask] + 1
    exit_pos = exit_pos[exit_pos < n]
    exit_dates = dates[exit_pos]
    return entry_dates, exit_dates


def make_signal_fn(ticker: str, index, frames: dict):
    """SignalFn for one pool member over one bucket (``index``). Builds (or
    reuses a cached) cross-sectional top-decile schedule from the FULL
    ``frames`` dict -- shared across every pool member's call for the same
    bucket -- then converts this ticker's own membership timeline into
    entry/exit flags exactly like ``large_cap_single_stock_pooled_tsmom``'s
    factory."""
    if ticker not in TRADED:
        raise ValueError(f"{FAMILY_ID}: {ticker!r} is not a pool member")
    if ticker not in frames:
        raise KeyError(f"{FAMILY_ID}: no frame for {ticker}")
    frame = frames[ticker]
    if "Open" not in frame.columns or "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame lacks Open/Close columns")

    bucket = _normalize_index(index)
    if len(bucket) == 0:
        def empty_signal_fn(frame_slice, i):
            return _NONE
        return empty_signal_fn

    sessions = trading_calendar.load_default_sessions()
    cutoff = bucket[-1]
    cache_key = int(pd.Timestamp(cutoff).value)
    cached = _schedule_cache.get(cache_key)
    if cached is not None and cached["frames"] is frames:
        schedule = cached["schedule"]
    else:
        schedule = build_target_schedule(frames, cutoff, sessions=sessions)
        _schedule_cache[cache_key] = {"frames": frames, "schedule": schedule}

    own_dates = _normalize_index(frame.index)
    own_frame = frame.set_axis(own_dates)
    own_frame = own_frame.loc[own_frame.index <= cutoff]
    entry_dates, exit_dates = decision_dates(
        own_frame, ticker, schedule, sessions=sessions)

    entry = np.asarray(bucket.isin(entry_dates), dtype=bool)
    exit_ = np.asarray(bucket.isin(exit_dates), dtype=bool)
    results = np.empty(len(bucket), dtype=object)
    for pos in range(len(bucket)):
        if entry[pos] and exit_[pos]:
            # Only reachable when a calendar month holds a single bar (the
            # exit bar after a non-member month-end is itself a month-end):
            # report both flags honestly and let the engine's own collision
            # rule decide (same edge case as large_cap_single_stock_pooled_
            # tsmom.make_signal_fn).
            results[pos] = SignalResult(entry=True, exit=True)
        elif entry[pos]:
            results[pos] = _ENTRY
        elif exit_[pos]:
            results[pos] = _EXIT
        else:
            results[pos] = _NONE
    bucket_ns = bucket.asi8
    by_date = {int(ts): pos for pos, ts in enumerate(bucket_ns)}
    n = len(bucket)

    def signal_fn(frame_slice, i):
        today = pd.Timestamp(frame_slice.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        today_ns = int(today.normalize().value)
        if 0 <= i < n and bucket_ns[i] == today_ns:
            return results[i]
        pos = by_date.get(today_ns)
        if pos is None:
            return _NONE
        return results[pos]

    return signal_fn
