"""us_industry_etf_pooled_tsmom -- pooled 12-month time-series momentum on
11 US sub-industry ETFs (long/flat), batch5 candidate module.

Source: Moskowitz, Ooi & Pedersen (2012), JFE 104(2); Antonacci (2013)
absolute momentum. Rule (verbatim from the curated candidate's
rule_pseudocode, implemented per pool member independently):

    C = auto-adjusted daily close of T (total return)
    M[k] = last available close of T in calendar month k
    at the close of the last trading day of each calendar month m
    (needs 12 prior month-end closes, else no decision):
        ret12 = M[m] / M[m-12] - 1
        target = LONG if ret12 > 0 else FLAT   (ret12 == 0 -> FLAT)
    target LONG and flat  -> BUY at the OPEN of the next trading session
    target FLAT and long  -> SELL at the CLOSE of the next trading session
    otherwise hold; no stops, no time exits.

Engine mapping (mining.backtest_engine.run_backtest): an entry signal on bar
t fills at bar t+1's open, an exit signal on bar t fills at bar t's close.
So the LONG decision is expressed as entry=True on the month-end bar itself
(and on every later bar while the decided state stays LONG, so the engine
keeps holding), and the FLAT decision as exit=True on the session AFTER the
month-end bar -- exactly "next_session_close".

Everything is precomputed ONCE in the factory from the full-history frame
(shift / groupby-last only, never centered); the returned signal_fn does two
positional array lookups per call.

Interpretation choices
----------------------
* "auto-adjusted daily close": the frame's ``Close`` as handed over by the
  driver (data_pipeline.fetch_ohlc uses yfinance ``auto_adjust=True``, so it
  is the dividend-adjusted total-return close). No further adjustment here.
* "last trading day of each calendar month": a bar is a month-end decision
  bar when the NEXT bar in the frame falls in a different calendar month
  (the next session's DATE is exchange-calendar knowledge, never its price).
  The frame's FINAL bar has no successor, so it counts as a month-end only
  when no weekday remains in its calendar month after it (a weekday-holiday
  such as a Good Friday on the 31st would then be missed at the very last
  bar only -- disclosed, immaterial: nothing can fill after the last bar).
* "needs 12 prior month-end closes": each of the 12 calendar months
  m-1 .. m-12 must have a month-end close in the frame; otherwise no
  decision on that month-end (the previous state, initially FLAT, persists).
* Holdout bucket: the state decided at the last month-end BEFORE the bucket
  carries into it (warm-up from earlier history is allowed; nothing after
  the bucket's last bar is read). A bucket therefore starts by entering at
  its second bar's open when the prior decision was LONG.
* Bars of ``index`` absent from the frame (should not happen -- the bucket is
  a slice of the same frame) yield a no-op signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "us_industry_etf_pooled_tsmom"
SHAPE = "pooled_single_asset"
TRADED = ["SOXX", "XBI", "XHB", "KRE", "XME", "XOP", "XRT", "IYT", "GDX",
          "IHI", "ITA"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied verbatim from the candidate record's "parameters" (driver drift check).
PARAMETERS = {
    "lookback_months": 12,
    "decision_frequency": "monthly_at_month_end_close",
    "long_condition": "trailing_12_month_total_return_gt_0",
    "flat_condition": "trailing_12_month_total_return_le_0",
    "entry": "next_session_open",
    "exit": "next_session_close",
    "pool_size": 11,
    "pooling": "all_pool_trades_concatenated_one_trial",
    "judge_edit": (
        "OIH is excluded from the pool: its pre-2011-12-21 yfinance history "
        "is the Merrill Lynch Oil Service HOLDRS fixed-basket trust joined "
        "onto the VanEck ETF, not a continuous index fund; SMH is replaced "
        "by SOXX for the identical reason"
    ),
}
EXPECTED_MIN_TRADES = 100

LOOKBACK_MONTHS = PARAMETERS["lookback_months"]


def _normalized_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: frame index must be sorted and unique")
    return index


def _month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on bars whose next bar is in a different calendar month; the
    final bar is True only when no weekday of its month remains after it."""
    periods = index.to_period("M")
    mask = np.zeros(len(index), dtype=bool)
    if len(index) == 0:
        return mask
    mask[:-1] = periods[:-1] != periods[1:]
    last = index[-1]
    month_end_day = last.to_period("M").to_timestamp(how="end").normalize()
    remaining = pd.bdate_range(last + pd.Timedelta(days=1), month_end_day)
    mask[-1] = len(remaining) == 0
    return mask


def _decisions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per bar of `frame`: (state_long, flat_decision) where state_long[t] is
    the decided target in force at the close of bar t (False before the first
    decision) and flat_decision[t] is True exactly on month-end bars whose
    decision was FLAT."""
    index = _normalized_index(frame)
    closes = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    n = len(index)
    is_month_end = _month_end_mask(index)
    periods = index.to_period("M")

    # Month-end close per calendar month, keyed by Period (only confirmed
    # month-end bars contribute, so a mid-month final bar never counts).
    month_end_close = {}
    for pos in np.flatnonzero(is_month_end):
        month_end_close[periods[pos]] = closes[pos]

    decision = np.full(n, np.nan)  # 1.0 LONG, 0.0 FLAT, nan no decision
    for pos in np.flatnonzero(is_month_end):
        period = periods[pos]
        prior = [period - k for k in range(1, LOOKBACK_MONTHS + 1)]
        if any(p not in month_end_close for p in prior):
            continue
        base = month_end_close[period - LOOKBACK_MONTHS]
        current = closes[pos]
        if not (np.isfinite(base) and np.isfinite(current)) or base <= 0:
            continue
        ret12 = current / base - 1.0
        decision[pos] = 1.0 if ret12 > 0 else 0.0

    state = pd.Series(decision, index=index).ffill().fillna(0.0).to_numpy() > 0.5
    flat_decision = decision == 0.0
    return state, flat_decision


def make_signal_fn(ticker: str, index: pd.DatetimeIndex, frames: dict):
    """SignalFn for one pool member. `frames[ticker]` is the full-history
    OHLC frame; `index` is the bucket the engine will iterate over."""
    frame = frames[ticker]  # KeyError on a missing frame is the right failure
    state, flat_decision = _decisions(frame)
    frame_index = _normalized_index(frame)

    # entry[t]  = decided state in force at bar t is LONG
    # exit[t]   = the PREVIOUS frame bar carried a FLAT decision
    entry_frame = pd.Series(state, index=frame_index)
    exit_frame = pd.Series(flat_decision, index=frame_index).shift(
        1, fill_value=False)

    bucket = pd.DatetimeIndex(pd.to_datetime(index))
    if bucket.tz is not None:
        bucket = bucket.tz_localize(None)
    bucket = bucket.normalize()
    entry = entry_frame.reindex(bucket).fillna(False).to_numpy(dtype=bool)
    exit_ = exit_frame.reindex(bucket).fillna(False).to_numpy(dtype=bool)
    size = len(bucket)

    def signal_fn(frame_slice: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= size:
            return SignalResult()
        return SignalResult(entry=bool(entry[i]), exit=bool(exit_[i]))

    return signal_fn
