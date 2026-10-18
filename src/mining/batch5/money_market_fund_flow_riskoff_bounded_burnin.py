"""Batch5 candidate: money_market_fund_flow_riskoff_bounded_burnin
(single_long).

A bounded-burn-in variant of ``money_market_fund_flow_riskoff`` -- SAME
mechanism, SAME data source (FRED RMFSL), SAME ROC/z-score/lag math, but
with the raw RMFSL history capped to
``mining.constants.MAX_PRE_INCEPTION_BURN_IN_YEARS`` years before SPY's
own first tradeable date.

WHY A SEPARATE FAMILY_ID rather than changing the original module: a
trial-ledger row must stay reproducible from the code that produced it,
so a family's recorded signal definition is never changed in place. A
genuinely different signal definition (bounded burn-in is a real change
to what the strategy COMPUTES, not a change to how an unchanged
strategy's returns are SCORED) is a new hypothesis and pays its own
trial, exactly like any other new candidate.

Design: identical to ``money_market_fund_flow_riskoff.py`` except one
extra step -- ``signals.cap_pre_inception_history`` trims the raw
``flow_signal`` to the burn-in cap, anchored at the traded calendar's own
first date, BEFORE the ROC/z-score computation. Everything downstream
(month-end re-anchoring, lag application, external regime lookup) is
byte-identical to the uncapped sibling.
"""
from __future__ import annotations

import pandas as pd

from mining import constants, signals
from mining.backtest_engine import SignalResult

FAMILY_ID = "money_market_fund_flow_riskoff_bounded_burnin"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL: list = []  # external (FRED RMFSL), not a price-frame signal ticker
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied byte-for-byte from the pre-registered catalog candidate's
# "parameters" field (driver drift check via json.dumps(sort_keys=True)).
PARAMETERS = {
    "flow_roc_lookback_months": 3,
    "flow_roc_zscore_riskoff_threshold": 1,
    "max_pre_inception_burn_in_years": constants.MAX_PRE_INCEPTION_BURN_IN_YEARS,
}
ROC_MONTHS = int(PARAMETERS["flow_roc_lookback_months"])
ZSCORE_RISKOFF_THRESHOLD = float(PARAMETERS["flow_roc_zscore_riskoff_threshold"])
MAX_YEARS_BEFORE = int(PARAMETERS["max_pre_inception_burn_in_years"])
# Measured, not pre-registered -- identical to the uncapped sibling's own
# LAG_TRADING_DAYS (the lag itself is unaffected by the burn-in cap).
LAG_TRADING_DAYS = 25
# Conservative sanity floor on how many trades the signal must produce
# before a run is trusted (guards against a degenerate, almost-never-
# trading signal). A configurable threshold, not a reported result.
EXPECTED_MIN_TRADES = 5


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    return frame


def _empty_signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
    return SignalResult()


def make_signal_fn(ticker: str, index, frames: dict, *, flow_signal=None):
    """SignalFn factory. ``flow_signal``, when supplied, is the RAW
    monthly FRED RMFSL level series (never a pre-transformed ROC/z-score/
    lagged/capped series) -- identical convention to the uncapped
    sibling's own ``flow_signal`` keyword."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)
    traded_index = pd.DatetimeIndex(index)
    if len(traded_index) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")

    if flow_signal is None or flow_signal.empty:
        return _empty_signal_fn

    capped = signals.cap_pre_inception_history(
        flow_signal, traded_index.min(), max_years_before=MAX_YEARS_BEFORE)
    if capped.empty:
        return _empty_signal_fn

    raw_regime = signals.money_market_risk_on_series(
        capped, roc_months=ROC_MONTHS,
        riskoff_z_threshold=ZSCORE_RISKOFF_THRESHOLD)
    if raw_regime.empty:
        return _empty_signal_fn

    anchored = raw_regime.copy()
    anchored.index = anchored.index.to_period("M").to_timestamp("M")
    available = signals.lag_boolean_regime_to_availability(
        anchored, traded_index, lag_trading_days=LAG_TRADING_DAYS)
    if available.empty:
        return _empty_signal_fn

    return signals.make_external_regime_signal_fn(available)
