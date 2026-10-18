"""Batch5 candidate: tips_nominal_breakeven_rotation (rotation_two_asset).

Implements the mechanism the batch3 driver defines for this family_id
(``scripts/run_batch3_mining.py``'s ``SUPPORTED_CONTRACTS`` /
``EXPECTED_PARAMETER_VALUES`` / ``catalog_parameters()`` entries) under the
batch5 module contract (``scripts/run_batch5_mining.py``): the same
mechanism and parameters, expressed in the batch5 shape.

Mechanism (``data/mining/catalog.json``'s real entry for this
family_id): compute TIP's trailing 3-calendar-month total return
relative to IEF's (a market-implied-inflation-expectations proxy). TIP
outperforming IEF (rising breakeven) -> hold TIP; underperforming
(falling breakeven) -> hold IEF. Rebalance monthly. Isolates
inflation-expectations direction from outright nominal-rate direction,
and trades bonds directly (never equities).

By the catalog's own ``"signal": "TIP_minus_IEF_trailing_return"`` field
and by the batch3 driver's own wiring
(``SUPPORTED_CONTRACTS["tips_nominal_breakeven_rotation"]`` names
mechanism class ``monthly_two_asset_relative_momentum_rotation``, and
``catalog_parameters()`` maps this family_id's catalog parameters
straight through to ``{"lookback_months": p["lookback_months"]}``), the
mechanism IS exactly
``signals.make_monthly_relative_momentum_target_fn(risk_ticker="TIP",
defensive_ticker="IEF", lookback_months=3)``. No new math is added
here -- this module is a THIN WRAPPER, not a re-implementation. That
factory's own returned ``target(frames: dict[str, pd.DataFrame])``
already matches the batch5 rotation driver's truncated-frames-per-call
contract, so ``make_target_fn`` below returns it unchanged; no further
wrapping layer is needed (the same "thin wrapper directly returning a
mining.signals factory's own output" convention documented in
``fed_net_liquidity_regime_qqq.py``'s own module docstring). The
factory itself, including its TIP/IEF behavior specifically, is already
covered by ``src/tests/test_mining_batch3_signals.py`` -- this module's
own test proves DELEGATION and contract compliance, not the underlying
momentum math a second time.

BENCHMARK is SPY, not TIP: this pipeline benchmarks every family
against SPY unconditionally, never against whatever is traded (see any
other batch5 module's own comment on this rule, e.g.
``fed_net_liquidity_regime_qqq.py``).
"""
from __future__ import annotations

from mining import signals

FAMILY_ID = "tips_nominal_breakeven_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["TIP", "IEF"]          # [risk, defensive]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (this module's own test verifies json.dumps(sort_keys=True)
# byte-identity against the live catalog file directly, not a hand-typed
# duplicate that could drift from it).
PARAMETERS = {
    "lookback_months": 3,
    "signal": "TIP_minus_IEF_trailing_return",
    "rebalance": "monthly",
}
# Configured, conservative floor (not a measured count), same convention as
# every sibling batch5 module (e.g. monthly_opex_week_drift.py). The curve is
# built via mining.abi1_driver.build_rotation_curve, which trims the two legs
# to their common inception before the engine's own identical-calendar check
# runs.
EXPECTED_MIN_TRADES = 60

_RISK = TRADED[0]
_DEFENSIVE = TRADED[1]
_LOOKBACK_MONTHS = int(PARAMETERS["lookback_months"])


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract): called ONCE by the
    driver with full-history frames for TRADED (+ SIGNAL, none here).
    Every required TRADED key is validated present here so a missing
    frame fails at construction time, naming the missing key, rather
    than inside the first per-bar call. Returns the shared
    ``mining.signals`` factory's own per-call ``target(truncated)``
    callable unchanged -- prices are read only from the truncated
    frames the driver passes to it at call time, never from the
    full-history ``frames`` argument closed over here."""
    for key in TRADED:
        if key not in frames:
            raise ValueError(
                f"{FAMILY_ID}: make_target_fn requires a frame for {key}")
    return signals.make_monthly_relative_momentum_target_fn(
        _RISK, _DEFENSIVE, lookback_months=_LOOKBACK_MONTHS)
