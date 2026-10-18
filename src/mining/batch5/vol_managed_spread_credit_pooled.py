"""Batch5 candidate: vol_managed_spread_credit_pooled (pooled_single_asset).

Volatility-managed credit exposure on HYG and EMB, Moreira-Muir monthly
construction as a two-state long/flat switch on each fund's OWN trailing
realized variance (Harvey et al. 2018 find volatility targeting raises the
Sharpe ratio for equities and credit). Implemented EXACTLY from the
candidate's rule_pseudocode (its data/mining/catalog.json entry):

  SIGNAL     RV_t = sum over the trailing 21 trading days ending at t
             (inclusive) of ln(Close_s / Close_{s-1})^2, from auto-adjusted
             daily closes (frames[ticker]["Close"]; data_pipeline.fetch_ohlc
             pulls yfinance with auto_adjust=True).
  THRESHOLD  MED_t = median of {RV_u : u <= t} over ALL available history
             (expanding, strictly causal); undefined -- no position -- until
             at least 252 RV observations exist.
  STATE      LOW_t = (RV_t <= MED_t).
  DECISION   once per calendar month, on that month's last trading day t;
             the target executes at the t+1 open and is held unchanged until
             the next month's decision executes. No intra-month
             re-evaluation, no hysteresis, no other filter.
  POSITION   LOW_t -> long the fund (enter at the t+1 open if flat, else
             hold); otherwise flat (exit per the engine's single-asset
             convention, i.e. the decision bar's close).
  POOL       run independently on HYG and EMB (signal = each fund's own
             close); trades pooled by the driver into ONE trial.

Engine mapping (mining.backtest_engine.run_backtest): on a decision bar the
signal returns ``SignalResult(entry=LOW_t, exit=not LOW_t)``; on every other
bar it returns ``SignalResult()`` (hold whatever the state is). The engine
fills an entry at the next bar's open, ignores an entry while already long,
and closes an exit at that bar's close.

No look-ahead: every series is precomputed ONCE in the factory from the
ticker's full-history frame using only shift / rolling / expanding
operations, so the value stored for date t uses closes <= t only. It is
then reindexed to ``index`` (the bucket the engine runs over) and each
per-bar call is a dict lookup keyed on the slice's last date -- no rolling
statistic is ever recomputed from the growing slice (O(1) per bar).

Holdout-blind: for a holdout bucket the precomputation warms up on frame
history BEFORE the bucket (the 21-day window and the 252-observation
expanding median need it) but nothing after the bucket's last bar can
influence any value at or before it, because the operations are causal.

Interpretation choices (genuine gaps in the prose, resolved toward the
cited Moreira-Muir monthly construction; no parameters invented):
* "Last trading day of the month" is read off the bucket calendar the
  engine runs over: bar t is a decision bar iff the NEXT bar of ``index``
  falls in a different calendar month. The exchange calendar is known in
  advance, so this uses no future price. The final bar of ``index`` is
  never a decision bar: its "t+1 open" does not exist inside the bucket
  (an entry there would be the engine's documented degenerate held=0
  trade, and an exit there is identical to the engine's own end_of_data
  close).
* While MED_t is undefined (fewer than 252 RV observations) a decision bar
  returns flat (``exit=True, entry=False``) -- "no position is taken".
* A bucket date missing from the ticker's frame (should not happen; the
  bucket is a slice of that frame) is treated as no signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "vol_managed_spread_credit_pooled"
SHAPE = "pooled_single_asset"
TRADED = ["HYG", "EMB"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "HYG"
# Copied verbatim from the catalog entry's "parameters" (driver drift check is
# byte-for-byte on json.dumps(sort_keys=True)).
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
    "pool": [
        "HYG",
        "EMB",
    ],
}
EXPECTED_MIN_TRADES = 50

_RV_LOOKBACK = int(PARAMETERS["rv_lookback_days"])
_MIN_OBS = int(PARAMETERS["threshold_min_obs"])


def _clean_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out


def low_variance_state(close: pd.Series) -> pd.Series:
    """LOW_t as a float series on ``close``'s index: 1.0 low, 0.0 high,
    NaN while RV_t or MED_t is undefined. Strictly causal (shift / rolling
    / expanding only)."""
    values = pd.to_numeric(close, errors="coerce").astype(float)
    log_ret = np.log(values / values.shift(1))
    rv = (log_ret ** 2).rolling(_RV_LOOKBACK, min_periods=_RV_LOOKBACK).sum()
    med = rv.expanding(min_periods=_MIN_OBS).median()
    low = (rv <= med).astype(float)
    low[rv.isna() | med.isna()] = np.nan
    return low


def make_signal_fn(ticker: str, index, frames: dict):
    """Per-instrument SignalFn for ``ticker`` over the bucket ``index``.

    ``frames[ticker]`` is the full-history OHLC frame (tz-naive
    DatetimeIndex; columns Open High Low Close); only its Close is read.
    """
    frame = frames[ticker]  # KeyError if the driver forgot the frame
    close = pd.Series(frame["Close"].to_numpy(dtype=float),
                      index=_clean_index(frame.index)).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    bucket = _clean_index(index)

    low = low_variance_state(close).reindex(bucket).to_numpy(dtype=float)
    n = len(bucket)
    # Decision bar iff the next bucket bar is in a different calendar month;
    # the last bar of the bucket is never a decision bar (see docstring).
    is_decision = np.zeros(n, dtype=bool)
    if n >= 2:
        periods = bucket.to_period("M").asi8
        is_decision[:-1] = periods[1:] != periods[:-1]

    # Per-bar result, fully materialised once; the signal call is a lookup.
    hold = SignalResult()
    go_long = SignalResult(entry=True, exit=False)
    go_flat = SignalResult(entry=False, exit=True)
    results = []
    for pos in range(n):
        if not is_decision[pos]:
            results.append(hold)
        elif np.isnan(low[pos]):
            results.append(go_flat)  # MED_t undefined -> no position
        else:
            results.append(go_long if low[pos] >= 0.5 else go_flat)
    lookup = {int(ts): res for ts, res in zip(bucket.asi8, results)}

    def signal_fn(frame_slice: pd.DataFrame, i: int) -> SignalResult:
        today = frame_slice.index[-1]
        if getattr(today, "tzinfo", None) is not None:
            today = today.tz_localize(None)
        return lookup.get(int(pd.Timestamp(today).value), hold)

    signal_fn.family_id = FAMILY_ID
    signal_fn.ticker = ticker
    return signal_fn
