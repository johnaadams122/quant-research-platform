"""Batch5 candidate: vol_managed_intl_country_etf_pooled (pooled_single_asset).

Moreira & Muir (2017) volatility management applied per single-country
iShares fund (Schwarz 2025 international evidence), as a two-state in/out
switch decided once a month:

  RV_t  = sum over the trailing 21 sessions ending at t (inclusive) of
          ln(Close_s / Close_{s-1})^2        (21 squared log returns)
  MED_t = median of {RV_u : u <= t}, expanding over ALL available history,
          undefined until at least 252 RV observations exist
  LOW_t = RV_t <= MED_t
  On each month's LAST trading day t: if LOW_t -> long (entry at the t+1
  open if flat, else hold); else flat (exit at t's close). No intra-month
  re-evaluation, no hysteresis, no other filter. Run independently on each
  pool fund against its own close; trades pooled as one trial by the driver.

Engine mapping (mining.backtest_engine): on a decision bar the signal is
SignalResult(entry=LOW_t, exit=not LOW_t); the engine fills the entry at the
next open and the exit at the same close, and ignores entry=True while a
position is already open, so consecutive LOW months read as "hold". Every
non-decision bar returns SignalResult() (no change).

No look-ahead: every series is precomputed ONCE in the factory from the
ticker's own full-history frame truncated to <= the bucket's last date,
using only rolling/expanding operations (the value stored for date t uses
closes <= t), then reindexed to the bucket ``index`` and read by position
``i`` inside signal_fn -- two array lookups per call, no per-bar recompute.

Interpretation choices
----------------------
* "Last trading day of the month" is read off the fund's own session
  calendar: bar t is a decision bar iff the next bar in the frame falls in
  a different calendar month. Whether a session is the month's last one is a
  calendar fact (known in advance from the exchange calendar), not a price,
  so this carries no price look-ahead. The final bar of the (truncated)
  frame is never a decision bar: nothing is knowable about the following
  session from the frame alone, and a decision there could not execute in
  any case (no t+1 open exists inside the bucket).
* While MED_t is undefined (fewer than 252 RV observations) the pseudocode
  says "no position is taken": a decision bar then emits entry=False,
  exit=True (flat), which is a no-op when already flat.
* RV uses 21 log returns, so the first RV value needs 22 closes; the
  expanding median starts once 252 such values exist (bar 273 of history).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "vol_managed_intl_country_etf_pooled"
SHAPE = "pooled_single_asset"
TRADED = ["EWJ", "EWY", "EWT", "FXI", "INDA", "EWZ", "EWW", "EWC", "EWA",
          "EWH", "EWU", "EWG"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "VEU"
# Copied verbatim from the candidate record's "parameters" for this
# family_id -- the driver's drift check
# compares json.dumps(..., sort_keys=True) byte-for-byte.
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
        "EWJ",
        "EWY",
        "EWT",
        "FXI",
        "INDA",
        "EWZ",
        "EWW",
        "EWC",
        "EWA",
        "EWH",
        "EWU",
        "EWG",
    ],
    "benchmark_fallback": "EFA if VEU (inception 2007-03-02) cannot be quoted or the panel needs pre-2007 coverage",
}
EXPECTED_MIN_TRADES = 400

_RV_LOOKBACK = PARAMETERS["rv_lookback_days"]
_MIN_OBS = PARAMETERS["threshold_min_obs"]


def _month_end_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """True where the NEXT bar of ``index`` is in a different calendar month;
    the final bar is always False (see module docstring)."""
    months = index.year.to_numpy() * 12 + index.month.to_numpy()
    flags = np.zeros(len(index), dtype=bool)
    if len(index) > 1:
        flags[:-1] = months[1:] != months[:-1]
    return flags


def _precompute(frame: pd.DataFrame, index: pd.DatetimeIndex):
    """Return (entry_flags, exit_flags) as bool arrays aligned to ``index``."""
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        index = index.tz_localize(None)
    history = frame.copy()
    history.index = pd.DatetimeIndex(pd.to_datetime(history.index))
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    history = history.sort_index()
    if len(index):
        # Holdout-blind by construction: nothing after the bucket's last
        # date can enter any statistic.
        history = history.loc[history.index <= index[-1]]

    closes = pd.to_numeric(history["Close"], errors="coerce").astype(float)
    log_returns = np.log(closes / closes.shift(1))
    rv = (log_returns ** 2).rolling(_RV_LOOKBACK, min_periods=_RV_LOOKBACK).sum()
    median = rv.expanding(min_periods=_MIN_OBS).median()
    low = rv.notna() & median.notna() & (rv <= median)
    decision = pd.Series(_month_end_flags(history.index), index=history.index)

    entry = (decision & low).reindex(index, fill_value=False).to_numpy(dtype=bool)
    exit_ = (decision & ~low).reindex(index, fill_value=False).to_numpy(dtype=bool)
    return entry, exit_


def make_signal_fn(ticker: str, index: pd.DatetimeIndex, frames: dict):
    """SignalFn for one pool member. ``frames[ticker]`` must be that fund's
    full-history OHLC frame (tz-naive DatetimeIndex, Open/High/Low/Close);
    ``index`` is the bucket the engine will run over. Raises KeyError when
    the ticker's own frame is absent -- the signal source is the fund's own
    close, never a substitute."""
    frame = frames[ticker]
    entry_flags, exit_flags = _precompute(frame, index)
    n = len(entry_flags)

    def signal_fn(frame_slice: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n:
            return SignalResult()
        return SignalResult(entry=bool(entry_flags[i]), exit=bool(exit_flags[i]))

    return signal_fn
