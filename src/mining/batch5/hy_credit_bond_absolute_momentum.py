"""Batch5 candidate: hy_credit_bond_absolute_momentum (rotation_two_asset).

The mechanism is the one the per-trade runner
``scripts/run_batch3_mining.py`` declares for this family_id (its
``SUPPORTED_CONTRACTS`` mechanism class
``"monthly_two_asset_absolute_momentum_rotation"`` and its
``EXPECTED_PARAMETER_VALUES`` entry). This module expresses that mechanism
in the batch5 module contract shape: the decision rule is
``mining.signals.make_monthly_absolute_momentum_target_fn``, called with
the catalog entry's parameters.

Mechanism (Jostova, Nikolova, Philipov & Stahel 2013, "Momentum in
Corporate Bond Returns", Review of Financial Studies 26(7), 1649-1693; see
``data/mining/catalog.json``'s entry for this family_id for the full
citation): compute HYG's own trailing 6-calendar-month total return (this
repo's auto-adjusted yfinance close, per ``data_pipeline`` convention). If
HYG's current close exceeds its 6-month-ago calendar-anchor close, hold
HYG; otherwise rotate fully into BIL (T-bill cash proxy). Re-evaluate
monthly. A single-instrument ABSOLUTE (time-series) momentum filter on the
credit-risk premium itself -- distinct from a cross-asset equity-timing
regime signal.

``make_target_fn`` returns
``mining.signals.make_monthly_absolute_momentum_target_fn(TRADED[0],
TRADED[1], lookback_months=6)`` directly: that factory's own ``target``
callable already takes ``frames: dict[str, pd.DataFrame]`` as its single
per-call argument and computes its decision purely from the frames handed
to it each call -- exactly the batch5 rotation_two_asset per-bar contract
(the driver truncates ``frames`` further each bar before calling), with no
extra wrapping layer needed. See ``fed_net_liquidity_regime_qqq.py``'s own
docstring for this same "a signals.py factory already returns the
correctly-shaped per-call target" convention. The only work this module
adds on top is the one-time required-key check the driver contract asks
``make_target_fn`` itself to perform before any per-bar call.
"""
from __future__ import annotations

from mining import signals

FAMILY_ID = "hy_credit_bond_absolute_momentum"
SHAPE = "rotation_two_asset"
TRADED = ["HYG", "BIL"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks against SPY regardless of the family's own catalog
# "benchmark" field -- this is a hard rule of this pipeline, see any
# existing batch5 module's own comment on this, e.g.
# fed_net_liquidity_regime_qqq.py.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (verified via json.dumps(sort_keys=True) byte-identity in this
# module's own test, which reads that file directly rather than duplicating
# a value here that could drift).
PARAMETERS = {
    "lookback_months": 6,
    "signal": "trailing_total_return",
    "rebalance": "monthly",
    "cash_proxy": "BIL",
}
_LOOKBACK_MONTHS = PARAMETERS["lookback_months"]
# A conservative floor for a monthly-rebalanced two-asset rotation over the
# full fetched HYG/BIL history, set below the count such a rotation
# produces, following the same conservative-floor convention as every
# sibling batch5 module.
EXPECTED_MIN_TRADES = 20


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED; only their presence is checked
    here -- decisions come from the (further-truncated, per the engine's
    own contract) frames the returned callable is called with, delegated
    unchanged to mining.signals.make_monthly_absolute_momentum_target_fn."""
    for key in TRADED:
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")
    return signals.make_monthly_absolute_momentum_target_fn(
        TRADED[0], TRADED[1], lookback_months=_LOOKBACK_MONTHS)
