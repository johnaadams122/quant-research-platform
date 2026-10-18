"""Batch5 candidate: margin_debt_growth_regime (single_long).

FINRA aggregate NYSE member-firm customer margin-debt trailing-12-month
growth regime on SPY. Source: Beluska & Vojtko (2025), "Can Margin Debt
Help Predict SPY's Growth & Bear Markets?", QuantPedia/SSRN (catalog.json,
family "margin_debt_growth_regime"). Mechanism (verbatim from the
pre-registered catalog record):

    Compute trailing 12-month % change in FINRA-published aggregate NYSE
    member-firm customer margin-debt balances, lagged ~20 trading days for
    real publication delay. Positive growth (leverage/risk appetite
    expanding) -> hold SPY; growth crosses below zero -> exit to cash.

PROVENANCE: `scripts/run_batch3_mining.py` wires `external_data.
fetch_finra_margin_debt` through `mining.signals.
finra_margin_debt_risk_on_series` / `finra_margin_debt_growth_available_
series` / `make_external_regime_signal_fn` for this family (the
per-trade-count construction). This module is the ABI-1 form of the same
family and does not re-implement the FINRA growth/lag math: a hand-rolled
availability-date calculation can easily disagree with
`signals.finra_margin_debt_growth_available_series` by one trading session
whenever a calendar month-end itself is not a trading day -- both versions
causally safe, but two implementations of one pre-registered parameter for
one family_id would be a defect. This module is therefore a THIN ADAPTER:
all FINRA growth/lag/threshold math is `mining.signals`'s, unchanged,
exactly matching `run_batch3_mining.py`'s own use of the identical
functions for this identical family.

DISCLOSED, NOT INVENTED: ``LAG_TRADING_DAYS = 20`` is copied verbatim
from the pre-registered catalog parameter (``catalog.json``), never
altered here. FINRA's own Margin Statistics
page (corroborated by QuantPedia/Strasmore) indicates that real publication lag
runs roughly 4-7 weeks after month-end (third week of the following
month) -- the low end matches 20 trading sessions, but the high end can
run past it. This is a genuine, unconfirmed-default residual
risk (it lives in `signals.
finra_margin_debt_growth_available_series`, not here) and is
left as is rather than silently widened or ignored.

Design:

* NO LOOK-AHEAD -- `signals.finra_margin_debt_risk_on_series` dates each
  monthly reading at its real lagged availability date; `signals.
  make_external_regime_signal_fn`'s returned `signal_fn` looks up
  `state.index <= today` at call time (the engine's own current bar),
  which can structurally never see a future reading. Both are
  pre-existing, already-tested code (`src/tests/
  test_mining_batch3_signals.py`) -- this module adds no new lookahead
  surface.
* A ticker with no `margin_signal` supplied at all (the shared batch5
  driver's generic 3-positional-arg dispatch, which never threads
  external data through) degrades to "never enter, never exit" --
  matching `capex_cycle_blue_chip_dip_pooled`'s identical
  graceful-degradation convention for a family that needs data the
  generic driver cannot supply. This family's own bespoke driver
  (`scripts/run_margin_debt_mining.py`) always supplies it.
"""
from __future__ import annotations

import pandas as pd

from mining import signals
from mining.backtest_engine import SignalResult

FAMILY_ID = "margin_debt_growth_regime"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL: list = []  # external (FINRA), not a price-frame signal ticker
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied byte-for-byte from the pre-registered catalog candidate's
# "parameters" field (driver drift check via json.dumps(sort_keys=True)).
PARAMETERS = {
    "margin_debt_lookback_months": 12,
    "signal_lag_trading_days": 20,
    "growth_threshold": 0,
}
LOOKBACK_MONTHS = int(PARAMETERS["margin_debt_lookback_months"])
LAG_TRADING_DAYS = int(PARAMETERS["signal_lag_trading_days"])
GROWTH_THRESHOLD = float(PARAMETERS["growth_threshold"])
# Sanity floor on the full-history trade count: the driver refuses a run
# that produces fewer trades than this (a guard against a silently
# degenerate signal), same conservative-floor convention as
# vix_term_structure_carry.py, with room left for whatever slice of history
# the driver's own window/holdout handling covers.
EXPECTED_MIN_TRADES = 5


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    return frame


def _empty_signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
    return SignalResult()


def make_signal_fn(ticker: str, index, frames: dict, *, margin_signal=None):
    """SignalFn factory. ``margin_signal``, when supplied, is the RAW
    monthly FINRA ``finra_margin_debt_usd_millions`` series (never a
    pre-transformed growth/lag series) -- identical convention to
    ``capex_cycle_blue_chip_dip_pooled``'s ``fundamentals`` keyword. All
    growth/lag/threshold math is delegated to ``mining.signals``
    (shared, covered by its own tests), never re-implemented here."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)  # presence check only; SPY's own prices
                                    # never enter the signal computation
    if len(pd.DatetimeIndex(index)) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")

    if margin_signal is None or margin_signal.empty:
        return _empty_signal_fn

    risk_on = signals.finra_margin_debt_risk_on_series(
        margin_signal, index, lookback_months=LOOKBACK_MONTHS,
        lag_sessions=LAG_TRADING_DAYS, growth_threshold=GROWTH_THRESHOLD)
    if risk_on.empty:
        return _empty_signal_fn

    return signals.make_external_regime_signal_fn(risk_on)
