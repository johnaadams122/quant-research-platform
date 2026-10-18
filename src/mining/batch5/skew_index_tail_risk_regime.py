"""Batch5 candidate: skew_index_tail_risk_regime (rotation_two_asset).

CBOE SKEW Index Tail-Risk Regime Filter. Track a 20-day moving average of
the CBOE SKEW Index (``^SKEW``, pricing far-OTM S&P put crash protection
relative to lognormal) against its own 252-day moving average. Fast MA
crosses above slow MA (crash-hedging demand rising) -> exit SPY into SHY;
re-enter SPY once the fast MA falls back below the slow MA. Source: catalog
entry ``data/mining/catalog.json`` (Mora-Valencia, Rodriguez-Raga & Vanegas
(2021), "Skew index: Descriptive analysis, predictive power, and short-term
forecast", North American Journal of Economics and Finance 56).

This implements the catalogued mechanism.
``scripts/run_batch3_mining.py``'s per-trade wiring for this family
(``SUPPORTED_CONTRACTS``/``EXPECTED_PARAMETER_VALUES``/``catalog_parameters``/
``_rotation_target`` in ``run_batch3_mining.py``) calls exactly:

    signals.make_moving_average_regime_target_fn(
        risk_ticker="SPY", defensive_ticker="SHY", signal_ticker="^SKEW",
        fast=20, slow=252)

This module wraps that same ``mining.signals`` factory, with the same
parameters, in the batch5 two-level factory contract. ``mining.signals.
make_moving_average_regime_target_fn`` already returns a callable
``target(frames: dict[str, pd.DataFrame])`` that does its own alignment/
validation (``mining.signals._aligned_frames``) on every call, matching the
truncated-frames per-call contract this driver expects -- so
``make_target_fn`` below only needs to validate the full-history frames up
front (naming the missing key on failure, batch5's own contract convention)
and then return that factory call's output directly, the same "thin
wrapper" shape ``fed_net_liquidity_regime_qqq.py`` uses for its own
``mining.signals`` delegation.

BENCHMARK is hard-pinned to "SPY" per this pipeline's standing rule: ABI-1
ALWAYS benchmarks the account against SPY, never a family's own catalog
"benchmark" field (see any other batch5 module's identical comment, e.g.
``fed_net_liquidity_regime_qqq.py``). For this family SPY is also the risk
leg and the catalog's own benchmark, so there is no actual drift here --
the comment exists only so a future reader never treats this attribute as
configurable.

Calendar: ``mining.rotation_engine.run_two_asset_rotation`` runs on the
three series aligned to their common calendar (SHY's 2002-07-30 inception
binds; ^SKEW and SPY both start earlier). Once the slow moving average is
defined the target is always one of the two legs, so the family is never
flat after warm-up. EXPECTED_MIN_TRADES is a configured, conservative floor
(not a measured count), the same convention as every sibling module in
this directory.
"""
from __future__ import annotations

import pandas as pd

from mining import signals

FAMILY_ID = "skew_index_tail_risk_regime"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "SHY"]
SIGNAL = ["^SKEW"]  # external price-frame ticker the generic dispatch
                    # auto-loads -- CBOE SKEW Index, not traded
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied verbatim from data/mining/catalog.json's real entry for this
# family_id (byte-identical under json.dumps(sort_keys=True); see the test
# file's own drift check, which reads that file directly rather than
# duplicating it here a second time).
PARAMETERS = {
    "signal_series": "SKEW",
    "skew_fast_ma_days": 20,
    "skew_slow_ma_days": 252,
    "defensive_proxy": "SHY",
}
EXPECTED_MIN_TRADES = 70


def make_target_fn(frames: dict[str, pd.DataFrame]):
    """Rotation target factory (driver contract): validate the full-history
    ``frames`` carry every required TRADED/SIGNAL key, then delegate
    directly to ``mining.signals.make_moving_average_regime_target_fn`` --
    the shared, unchanged factory this family's mechanism cites.
    Its returned callable already matches the truncated-frames per-call
    contract (``target(truncated: dict)``), so nothing else wraps it."""
    for key in (TRADED[0], TRADED[1], SIGNAL[0]):
        if key not in frames:
            raise ValueError(
                f"{FAMILY_ID}: make_target_fn requires a frame for {key}")
    return signals.make_moving_average_regime_target_fn(
        risk_ticker=TRADED[0], defensive_ticker=TRADED[1],
        signal_ticker=SIGNAL[0], fast=PARAMETERS["skew_fast_ma_days"],
        slow=PARAMETERS["skew_slow_ma_days"])
