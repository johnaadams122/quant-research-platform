"""Batch5 candidate: fed_net_liquidity_regime (single_long).

Fed net liquidity regime on SPY. Source: Howell, M.J. (2020), "Capital
Wars: The Rise of Global Liquidity" (catalog.json, family
"fed_net_liquidity_regime"). Mechanism (verbatim from the pre-registered
catalog record):

    Compute Net Liquidity = Fed total assets (weekly H.4.1, FRED WALCL)
    minus Treasury General Account (FRED WTREGEN) minus Overnight Reverse
    Repo (FRED RRPONTSYD). Above its own 13-week MA (liquidity expanding,
    e.g. QE or RRP drawdown) -> hold SPY; below (liquidity contracting,
    e.g. QT or TGA rebuild) -> reduce/exit to cash. Rebalance weekly.

Point-in-time inputs are required, as for ``money_market_fund_flow_riskoff``:
the public FRED graph CSV carries no point-in-time/vintage evidence, so
``external_data.fetch_fred_series_point_in_time`` (which needs a free FRED
API key) is used for all three inputs.

The net-liquidity/MA math lives in, and is tested with,
``signals.fed_net_liquidity_risk_on_series`` -- this module is a THIN
ADAPTER, exactly matching ``margin_debt_growth_regime.py`` and
``money_market_fund_flow_riskoff.py``'s own convention, and adds no new
net-liquidity/MA/lag math.

DISCLOSED, NOT BUILT HERE: the catalog lists ``tickers: ["SPY", "QQQ"]``,
but the mechanism text itself only ever says "hold SPY" -- QQQ never
appears in the mechanism's own decision rule. Testing BOTH inside this one
family would need a rotation/pooling rule between the two, which the
mechanism does not give (this pipeline pools multi-ticker tests into one
trial per family). This module builds SPY only, matching the mechanism's
literal text and the same single-ticker precedent as
margin_debt_growth_regime/money_market_fund_flow_riskoff. The QQQ-traded
variant is its own family, ``fed_net_liquidity_regime_qqq.py``, rather
than a guess inside this one.

DISCLOSED, MEASURED (not invented): ``LAG_TRADING_DAYS = 1``. Live FRED
ALFRED vintage queries (spot-checked 2015 and 2026) found
WALCL is published the very next trading session after its own Wednesday
reading (the well-known H.4.1 "Thursday for the prior Wednesday"
convention, stable across a decade). WTREGEN and RRPONTSYD are DAILY
series published same-day-or-next-day (also live-verified),
so by the time WALCL's own weekly figure is known, both daily inputs for
that same week are already known -- WALCL's own lag is the binding
constraint for the whole composite, not a separately-tracked minimum
across three different lags.

Design:

* NO LOOK-AHEAD -- ``signals.fed_net_liquidity_risk_on_series`` computes
  the net-liquidity/MA regime keyed at each WALCL reading's own real
  observation date (already the correct lag-application anchor -- unlike
  ``money_market_fund_flow_riskoff``, no month-start-to-month-end
  re-anchoring is needed here). ``signals.lag_boolean_regime_to_
  availability`` maps each entry to the ``LAG_TRADING_DAYS``-th trading
  session strictly after it. ``signals.make_external_regime_signal_fn``'s
  returned ``signal_fn`` looks up ``state.index <= today`` at call time,
  which can structurally never see a future reading.
* A ticker with no liquidity inputs supplied at all degrades to "never
  enter, never exit" -- matching the sibling FRED/FINRA candidates'
  identical graceful-degradation convention. This family's own bespoke
  driver (``scripts/run_fed_net_liquidity_mining.py``) always supplies them.
"""
from __future__ import annotations

import pandas as pd

from mining import signals
from mining.backtest_engine import SignalResult

FAMILY_ID = "fed_net_liquidity_regime"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL: list = []  # external (FRED WALCL/WTREGEN/RRPONTSYD), not a price-frame ticker
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied byte-for-byte from the pre-registered catalog candidate's
# "parameters" field (driver drift check via json.dumps(sort_keys=True)).
# NOTE: the catalog's own "rebalance": "weekly" is descriptive of WALCL's
# native cadence, not a separate tunable -- mining.signals.
# fed_net_liquidity_risk_on_series is already keyed at WALCL's own weekly
# observation dates, so no extra rebalance parameter is threaded through.
PARAMETERS = {
    "liquidity_ma_weeks": 13,
    "rebalance": "weekly",
}
MA_WEEKS = int(PARAMETERS["liquidity_ma_weeks"])
# Measured, not pre-registered -- see module docstring.
LAG_TRADING_DAYS = 1
# A conservative sanity floor for a full-history run: a weekly regime
# switch over two decades of WALCL history should flip many times, so a
# count below this signals a broken input or lag rather than a real
# result. Same conservative-floor convention as margin_debt_growth_regime.py.
EXPECTED_MIN_TRADES = 100


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    return frame


def _empty_signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
    return SignalResult()


def make_signal_fn(ticker: str, index, frames: dict, *,
                   walcl=None, wtregen=None, rrpontsyd=None):
    """SignalFn factory. ``walcl``/``wtregen``/``rrpontsyd``, when supplied,
    are the RAW FRED level series (never pre-transformed net-liquidity/MA/
    lag series) -- identical convention to the sibling FRED/FINRA
    candidates' own external-data keywords. All net-liquidity/MA/lag math
    is delegated to ``mining.signals`` (and tested there), never
    re-implemented here."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)  # presence check only; SPY's own prices
                                    # never enter the signal computation
    if len(pd.DatetimeIndex(index)) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")

    if walcl is None or wtregen is None or rrpontsyd is None:
        return _empty_signal_fn
    if walcl.empty or wtregen.empty or rrpontsyd.empty:
        return _empty_signal_fn

    raw_regime = signals.fed_net_liquidity_risk_on_series(
        walcl, wtregen, rrpontsyd, ma_weeks=MA_WEEKS)
    if raw_regime.empty:
        return _empty_signal_fn

    available = signals.lag_boolean_regime_to_availability(
        raw_regime, index, lag_trading_days=LAG_TRADING_DAYS)
    if available.empty:
        return _empty_signal_fn

    return signals.make_external_regime_signal_fn(available)
