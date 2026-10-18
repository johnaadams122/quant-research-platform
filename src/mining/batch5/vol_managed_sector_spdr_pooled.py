"""Batch5 candidate: vol_managed_sector_spdr_pooled (pooled_single_asset).

Moreira & Muir (2017) volatility-managed rule run independently on each of
the 11 Select Sector SPDRs as a two-state in/out switch on the fund's OWN
trailing realized variance:

  RV_t  = sum over the trailing 21 trading days ending at t (inclusive) of
          ln(Close_s / Close_{s-1})^2
  MED_t = expanding median of {RV_u : u <= t}; undefined until at least 252
          RV observations exist (no position while undefined)
  LOW_t = RV_t <= MED_t
  Once per calendar month, on the month's last trading day t: target long if
  LOW_t else flat; executes at the t+1 open; held unchanged until the next
  month's decision executes. No intra-month re-evaluation, no hysteresis.

Engine mapping (mining.backtest_engine.run_backtest): the per-date target
series is precomputed ONCE in the factory from the full-history frame using
only shift/rolling/expanding operations (the value for date t uses closes
<= t), then reindexed to the bucket ``index`` and read by position. Each bar
returns entry=target / exit=not target, so "stay long" is a steady
entry=True/exit=False and the month-end switch to flat is exit=True on the
decision bar (the engine's single-asset convention, as the candidate's
rule_pseudocode directs: exit at that bar's close). Per-call cost is a few
array lookups.

Interpretation choices
----------------------
* "Last trading day of the month" is determined from the fund's own session
  calendar: bar t is a decision bar iff the NEXT bar in the frame falls in a
  different calendar month. This is a calendar fact (the exchange schedule
  is public in advance), not a price fact, so it introduces no price
  look-ahead. The final bar of a frame is never a decision bar -- the frame
  cannot show that its last bar was a month end. Consequence: a frame that
  ends mid-month makes no decision on its last bar (the correct answer), and
  a frame that ends exactly on a month end forgoes that one decision.
* Exit timing: rule_pseudocode says "exit per the engine's single-asset
  convention", i.e. exit=True on the decision bar t fills at t's close (the
  same close that produced RV_t), one session earlier than the t+1 open the
  entry side uses. Entries fill at the t+1 open per the engine.
* Auto-adjusted closes: the signal is the frame's own Close column as
  supplied by the driver (data_pipeline.fetch_ohlc), which is the pipeline's
  adjusted-close convention for every family. No further adjustment here.
* Warm-up for the holdout bucket comes from history BEFORE the bucket (the
  full-history frame), never from anything after the bucket's last bar;
  test_batch5_vol_managed_sector_spdr_pooled proves truncating the frame at
  the bucket's end leaves every signal unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "vol_managed_sector_spdr_pooled"
SHAPE = "pooled_single_asset"
TRADED = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
          "XLRE", "XLC"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Byte-identical (under json.dumps(sort_keys=True)) to the candidate's
# "parameters" in its curated candidate entry (driver drift check).
PARAMETERS = {
    "signal": "realized_variance_21d_log_returns",
    "signal_source": "own_close",
    "rv_lookback_days": 21,
    "threshold": "expanding_median",
    "threshold_min_obs": 252,
    "rebalance": "monthly",
    "decision_bar": "last_trading_day_of_month",
    "low_variance_position": "long",
    "high_variance_position": "flat",
    "pool": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
             "XLRE", "XLC"],
}
EXPECTED_MIN_TRADES = 300

_RV_LOOKBACK = PARAMETERS["rv_lookback_days"]
_THRESHOLD_MIN_OBS = PARAMETERS["threshold_min_obs"]


def realized_variance_and_threshold(close: pd.Series):
    """(RV_t, MED_t) as-of series over ``close``'s own index.

    RV_t is the trailing ``rv_lookback_days`` sum of squared daily log
    returns ending at t inclusive (NaN until that many returns exist);
    MED_t is the expanding median of RV with ``threshold_min_obs`` minimum
    observations (NaN before). Both are strictly causal: rolling and
    expanding windows only, never centered.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    logret = np.log(values).diff()
    rv = (logret ** 2).rolling(_RV_LOOKBACK, min_periods=_RV_LOOKBACK).sum()
    med = rv.expanding(min_periods=_THRESHOLD_MIN_OBS).median()
    return rv, med


def decision_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the month-end decision bars: bar t is a decision bar iff
    the next bar in ``index`` belongs to a different calendar month. The
    final bar is never one (see module docstring, Interpretation choices)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return np.zeros(0, dtype=int)
    months = idx.year.to_numpy() * 12 + idx.month.to_numpy()
    return np.flatnonzero(months[:-1] != months[1:])


def target_series(frame: pd.DataFrame) -> pd.Series:
    """Per-date target over ``frame``'s index: 1.0 long, 0.0 flat, NaN when no
    decision with a defined threshold has been made yet. Decided on each
    month-end bar from RV_t <= MED_t and carried forward unchanged until the
    next decision."""
    idx = pd.DatetimeIndex(frame.index)
    rv, med = realized_variance_and_threshold(frame["Close"])
    rv_arr, med_arr = rv.to_numpy(), med.to_numpy()
    target = np.full(len(idx), np.nan)
    for pos in decision_positions(idx):
        if np.isfinite(rv_arr[pos]) and np.isfinite(med_arr[pos]):
            target[pos] = 1.0 if rv_arr[pos] <= med_arr[pos] else 0.0
    return pd.Series(target, index=idx).ffill()


def make_signal_fn(ticker: str, index: pd.DatetimeIndex, frames: dict):
    """Driver contract factory (pooled_single_asset: called once per pool
    member with that member as ``ticker``). Precomputes the target series
    from the member's full-history frame, reindexes to the bucket ``index``,
    and answers each engine call with O(1) array lookups."""
    if ticker not in frames:
        raise ValueError(f"{FAMILY_ID}: no frame supplied for {ticker}")
    frame = frames[ticker]
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame must be a non-empty "
                         "OHLC DataFrame with a Close column")
    history = frame.copy()
    history.index = pd.DatetimeIndex(pd.to_datetime(history.index)).tz_localize(None)
    if not history.index.is_monotonic_increasing or history.index.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame index must be sorted and unique")
    bucket = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None)
    targets = target_series(history).reindex(bucket).to_numpy()
    bucket_values = bucket.asi8

    def signal_fn(bar_frame: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= len(bucket_values):
            raise ValueError(f"{FAMILY_ID}: bar position {i} outside the bucket")
        today = pd.Timestamp(bar_frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        if today.value != bucket_values[i]:
            raise ValueError(
                f"{FAMILY_ID}: bar {today.date()} at position {i} does not match "
                f"the bucket calendar the factory was built for")
        value = targets[i]
        if not np.isfinite(value):
            return SignalResult()
        long = bool(value >= 0.5)
        return SignalResult(entry=long, exit=not long)

    return signal_fn
