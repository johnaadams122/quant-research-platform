"""Batch5 candidate: credit_spread_hy_ig_risk_regime (rotation_two_asset).

This family's mechanism and parameters are taken from its
``data/mining/catalog.json`` entry. The single-asset engine
(``mining.backtest_engine.SignalFn``) can only hold ONE ticker long/flat,
so there the catalog's "move into SHY" defensive leg could only be
modeled as going flat (no SHY position at all). This module instead
wraps the mechanism in the batch5 ``rotation_two_asset`` shape, which
CAN hold a real second (defensive) leg -- a more faithful expression of
the catalog's own mechanism text. No math is invented here: the
decision rule is delegated entirely, unchanged, to the shared
``mining.signals.make_ratio_regime_target_fn`` (the exact factory
``vxn_vix_spread_derisking_regime.py`` also uses), and the parameters
are the catalog's.

Mechanism (``data/mining/catalog.json`` entry for this family_id,
verbatim; Chen, Roll & Ross (1986), "Economic Forces and the Stock
Market", Journal of Business 59(3); Fama & French (1989), "Business
Conditions and Expected Returns on Stocks and Bonds", JFE 25(1)):
compute the daily price ratio HYG/LQD as a real-time proxy for the
high-yield-vs-investment-grade credit spread (a falling ratio means
credit spreads are widening -- risk-off). Take a 20-day and a 100-day
simple moving average of this ratio. Hold SPY long while the 20-day
average is above the 100-day average (spreads stable or compressing
-- risk-on); exit to SHY when the 20-day average crosses below the
100-day average (spreads widening -- risk-off); re-enter SPY on
crossback. Recomputed daily (``PARAMETERS["rebalance"] ==
"daily_signal_check"``).

**POLARITY NOTE (the risk/defensive arguments are deliberately
swapped).** The literal factory call --
``signals.make_ratio_regime_target_fn(risk_ticker=TRADED[0],
defensive_ticker=TRADED[1], numerator_ticker="HYG",
denominator_ticker="LQD", fast=20, slow=100)`` -- would produce the
OPPOSITE of the catalog mechanism above. ``make_ratio_regime_target_fn``
(``src/mining/signals.py``) always returns ``risk_ticker`` when
``fast_ma <= slow_ma`` and ``defensive_ticker`` when ``fast_ma >
slow_ma`` -- a FIXED polarity that treats a RISING ratio as risk-off
(correct for ``vxn_vix_spread_derisking_regime``'s VXN/VIX fear ratio,
where a rising ratio IS risk-off). This family's ratio has the OPPOSITE
sign: a rising HYG/LQD ratio is risk-ON (spreads compressing), per the
catalog text above: ``risk_on = bool(fast_ma > slow_ma)``, holding
SPY when ``risk_on`` is True. Because the factory's polarity is fixed,
the risk/defensive roles passed to it are swapped relative to every other
``make_ratio_regime_target_fn`` caller in this repo -- ``risk_ticker``
receives ``TRADED[1]`` (SHY) and ``defensive_ticker`` receives
``TRADED[0]`` (SPY) -- which reproduces the catalog's
``fast > slow -> SPY`` rule bar-for-bar using the LITERAL HYG/LQD ratio
(``numerator_ticker="HYG", denominator_ticker="LQD"``, matching
``PARAMETERS["signal_ratio"]`` exactly). This
module's own test proves the polarity directly against a hand-built
path (mirroring ``usd_strength_risk_regime.py``'s own
``test_uup_above_200sma_selects_the_defensive_leg_matching_catalog_text``
convention), not merely internal self-consistency.

``make_target_fn`` below wraps EXACTLY:

    signals.make_ratio_regime_target_fn(
        risk_ticker=TRADED[1], defensive_ticker=TRADED[0],
        numerator_ticker=SIGNAL[0], denominator_ticker=SIGNAL[1],
        fast=20, slow=100)

BENCHMARK is hard-pinned to "SPY" per this pipeline's standing rule:
ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
catalog "benchmark" field (see ``scripts/run_abi1_mining.py``'s own
``MARKET_BENCHMARK`` and every other batch5 module's identical comment
on this point). For this family SPY is also the risk leg, so there is
no actual drift here -- the comment exists only so a future reader
never treats this attribute as configurable.

Data coverage: a full-history run goes through
``mining.rotation_engine.run_two_asset_rotation`` on the four series
aligned to their common calendar, the same production path
``scripts/run_abi1_mining.py`` uses via
``mining.abi1_driver.build_rotation_curve``. HYG's data start
(2007-04-11, its first daily bar on yfinance) is the binding constraint,
later than SPY/SHY/LQD's own starts. EXPECTED_MIN_TRADES is a
conservative sanity floor on the trade count of a full-history run, the
same convention as every sibling module in this directory.
"""
from __future__ import annotations

import pandas as pd

from mining import signals

FAMILY_ID = "credit_spread_hy_ig_risk_regime"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "SHY"]      # [risk, defensive]
SIGNAL = ["HYG", "LQD"]      # external price-frame tickers the generic
                              # dispatch auto-loads -- [numerator,
                              # denominator] of the credit-spread ratio,
                              # not traded
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (verified via json.dumps(sort_keys=True) byte-identity in this
# module's own test, which reads that file directly rather than
# duplicating a value here that could drift).
PARAMETERS = {
    "ratio_fast_ma_days": 20,
    "ratio_slow_ma_days": 100,
    "signal_ratio": "HYG/LQD",
    "defensive_proxy": "SHY",
    "rebalance": "daily_signal_check",
}
_FAST = int(PARAMETERS["ratio_fast_ma_days"])
_SLOW = int(PARAMETERS["ratio_slow_ma_days"])
# A conservative sanity floor on the trade count of a full-history run on
# the shared SPY/SHY/HYG/LQD calendar (bound by HYG's own data start), the
# same convention as every sibling module in this directory.
EXPECTED_MIN_TRADES = 60


def make_target_fn(frames: dict[str, pd.DataFrame]):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED + SIGNAL; only their presence is
    checked here -- every price the rule reads comes from the truncated
    frames the engine hands over at call time, via
    ``signals.make_ratio_regime_target_fn``'s own per-call validation.

    NOTE the risk_ticker/defensive_ticker arguments below are swapped
    relative to TRADED's own [risk, defensive] order -- see the module
    docstring's "POLARITY NOTE" section for why: this family's
    HYG/LQD ratio has the opposite risk-on/risk-off polarity from
    vxn_vix_spread_derisking_regime's VXN/VIX ratio, and
    make_ratio_regime_target_fn's polarity is fixed, so the roles are
    swapped here to reproduce the catalog's real "fast MA above slow MA
    -> hold SPY" rule using the literal, unmodified HYG/LQD ratio."""
    for key in (TRADED[0], TRADED[1], SIGNAL[0], SIGNAL[1]):
        if key not in frames:
            raise ValueError(
                f"{FAMILY_ID}: make_target_fn requires a frame for {key}")
    return signals.make_ratio_regime_target_fn(
        risk_ticker=TRADED[1], defensive_ticker=TRADED[0],
        numerator_ticker=SIGNAL[0], denominator_ticker=SIGNAL[1],
        fast=_FAST, slow=_SLOW)
