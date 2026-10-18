"""Batch5 candidate: money_market_fund_flow_riskoff (single_long).

Retail money-market fund flow risk-off signal on SPY. Source: Warther, V.A.
(1995), "Aggregate mutual fund flows and security returns", Journal of
Financial Economics 39(2-3), 209-235 (catalog.json, family
"money_market_fund_flow_riskoff"). Mechanism (verbatim from the
pre-registered catalog record):

    Compute a 3-month rate-of-change z-score of the Fed's Retail Money
    Market Fund assets series (H.6 release, FRED ticker RMFSL -- fixed
    one-dollar NAV, so % change is a direct net-flow proxy). Inflows accelerate
    sharply (flight to safety) -> reduce/exit SPY; growth decelerates/
    reverses -> re-enter SPY.

The plain FRED fetch (``external_data.fetch_fred_series``, the public
graph CSV) carries no point-in-time/vintage evidence and is marked
``ledgerable=False`` -- using it directly would risk look-ahead bias. This
family needs a free FRED API key: ``external_data.
fetch_fred_series_point_in_time`` uses FRED's ``fred/series/observations``
endpoint and is marked ``ledgerable=True`` for RMFSL specifically (RMFSL is
on ``external_data.FRED_VERIFIED_UNREVISED_SERIES``; see that constant's
docstring: checked against FRED's own ALFRED vintage history to carry no
revision history).

The z-score/ROC math (``signals.money_market_risk_on_series``) and the
lag-to-availability primitive (``signals.lag_boolean_regime_to_
availability``) are shared, separately tested helpers -- this module is a
THIN ADAPTER, exactly matching ``margin_debt_growth_regime.py``'s own
convention, and adds no new growth/z-score/lag math.

DISCLOSED, MEASURED (not invented): the catalog's ``parameters`` field for
this family has NO pre-registered lag constant (unlike
margin_debt_growth_regime's ``signal_lag_trading_days``). ``LAG_TRADING_
DAYS = 25`` is set from FRED's own public ALFRED vintage history
(``output_type=2``) for a sample of historical months:
RMFSL's publication lag fell roughly 16-20 TRADING days after month-end
in every sampled month. 25 trading days is a deliberate margin above the
observed maximum -- a LONGER assumed lag can
only delay action (safe), never assume knowledge before it was real. A
residual risk is disclosed, not eliminated: a month whose real-world lag
exceeds 25 trading days (not seen in the samples) would still be a
look-ahead exposure, same disclosure shape as margin_debt_growth_regime's
own FINRA lag residual.

DISCLOSED (not invented): the catalog also does not pin a z-score
normalization window. ``signals.money_market_risk_on_series`` (shared
code, unchanged by this module) uses an EXPANDING window from series
start -- the parameter-free, causally
cleanest choice (never adds a look-ahead-widening constant, and every
z-score uses only data available up to that point).

NOTE on the expanding window: fed the full series from its own start date
(simply where FRED's data begins, decades before SPY exists), the
expanding z-score accumulates a long pre-inception burn-in, which makes
it progressively less sensitive to later flow spikes. The start date is
deliberately NOT tuned: trimming the burn-in to a shorter, more
"reactive" window chosen after seeing results would itself be a form of
look-ahead. Feeding the series' own unchosen start date is the one
non-post-hoc-tunable option; bounding the burn-in is a separate design
question for any expanding-window FRED candidate.

Design:

* NO LOOK-AHEAD -- ``signals.money_market_risk_on_series`` computes the
  ROC/z-score regime keyed at each RMFSL reading's own reference
  (month-start) date; this module re-anchors that index to MONTH-END
  (matching the convention the lag was measured against, and
  ``margin_debt_growth_regime.py``'s identical month-end anchor) before
  calling ``signals.lag_boolean_regime_to_availability``, which maps each
  entry to the ``LAG_TRADING_DAYS``-th trading session strictly after
  month-end. ``signals.make_external_regime_signal_fn``'s returned
  ``signal_fn`` looks up ``state.index <= today`` at call time (the
  engine's own current bar), which can structurally never see a future
  reading.
* A ticker with no ``flow_signal`` supplied at all degrades to "never
  enter, never exit" -- matching ``margin_debt_growth_regime``'s identical
  graceful-degradation convention for a family whose data the generic
  batch5 driver cannot supply. This family's own bespoke driver
  (``scripts/run_money_market_mining.py``) always supplies it.
"""
from __future__ import annotations

import pandas as pd

from mining import signals
from mining.backtest_engine import SignalResult

FAMILY_ID = "money_market_fund_flow_riskoff"
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
}
ROC_MONTHS = int(PARAMETERS["flow_roc_lookback_months"])
ZSCORE_RISKOFF_THRESHOLD = float(PARAMETERS["flow_roc_zscore_riskoff_threshold"])
# Measured, not pre-registered -- see module docstring.
LAG_TRADING_DAYS = 25
# Minimal sanity floor on the full-history trade count (at least one
# trade): with a slow expanding-window z-score (see the module docstring's
# NOTE) there is no meaningful higher floor to pre-commit to.
EXPECTED_MIN_TRADES = 1


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    return frame


def _empty_signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
    return SignalResult()


def make_signal_fn(ticker: str, index, frames: dict, *, flow_signal=None):
    """SignalFn factory. ``flow_signal``, when supplied, is the RAW monthly
    FRED RMFSL level series (never a pre-transformed ROC/z-score/lagged
    series) -- identical convention to ``margin_debt_growth_regime``'s
    ``margin_signal`` keyword. All ROC/z-score/lag math is delegated to
    ``mining.signals`` (shared, covered by its own tests), never
    re-implemented here."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)  # presence check only; SPY's own prices
                                    # never enter the signal computation
    if len(pd.DatetimeIndex(index)) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")

    if flow_signal is None or flow_signal.empty:
        return _empty_signal_fn

    raw_regime = signals.money_market_risk_on_series(
        flow_signal, roc_months=ROC_MONTHS,
        riskoff_z_threshold=ZSCORE_RISKOFF_THRESHOLD)
    if raw_regime.empty:
        return _empty_signal_fn

    anchored = raw_regime.copy()
    anchored.index = anchored.index.to_period("M").to_timestamp("M")
    available = signals.lag_boolean_regime_to_availability(
        anchored, index, lag_trading_days=LAG_TRADING_DAYS)
    if available.empty:
        return _empty_signal_fn

    return signals.make_external_regime_signal_fn(available)
