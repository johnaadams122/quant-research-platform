"""Batch5 candidate: vxn_vix_spread_derisking_regime (rotation_two_asset).

The batch3 driver (``scripts/run_batch3_mining.py``,
``SUPPORTED_CONTRACTS["vxn_vix_spread_derisking_regime"]`` /
``EXPECTED_PARAMETER_VALUES`` / ``catalog_parameters()``) defines and wires
this family. This module expresses the same wiring, with the same
parameters, in the batch5 self-describing-module contract
(``scripts/run_batch5_mining.py``).

Mechanism (``data/mining/catalog.json`` entry): compute the
ratio ^VXN/^VIX (Nasdaq-100 vs S&P 500 implied volatility). When the
ratio's 20-day MA rises meaningfully above its 252-day MA (tech-specific
options fear spiking faster than its trailing norm), rotate QQQ into SHY;
re-enter QQQ once the fast MA falls back below the slow MA.

The decision rule is
``mining.signals.make_ratio_regime_target_fn`` -- the same factory the batch3
driver's own ``catalog_parameters()`` wires for this family_id (fast=20,
slow=252, defensive_ticker=SHY, matching ``ratio_fast_ma_days`` /
``ratio_slow_ma_days`` / ``defensive_proxy`` below byte-for-byte). That
factory's returned callable already takes ``frames: dict[str,
pd.DataFrame]`` truncated through the decision bar as its single per-call
argument -- exactly the batch5 rotation contract
(``mining.rotation_engine.run_two_asset_rotation``'s ``target_fn``; frame
validation is ``mining.signals._aligned_frames``, shared by every ratio-
regime family) -- so ``make_target_fn`` below adds nothing beyond a
one-time presence check on the full-history frames it is handed once (see
``fed_net_liquidity_regime_qqq.py``'s own thin-wrapper precedent).

BENCHMARK is SPY, not QQQ: this pipeline benchmarks every family against
the market reference unconditionally, never against whatever is traded
(see e.g. ``fed_net_liquidity_regime_qqq.py``'s own comment on this same
point).
"""
from __future__ import annotations

from mining import signals

FAMILY_ID = "vxn_vix_spread_derisking_regime"
SHAPE = "rotation_two_asset"
TRADED = ["QQQ", "SHY"]           # [risk, defensive]
SIGNAL = ["^VXN", "^VIX"]         # external signal-only price frames, not traded
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (verified via json.dumps(sort_keys=True) byte-identity in this
# module's own test) -- identical to the batch3 driver's
# EXPECTED_PARAMETER_VALUES for this family_id.
PARAMETERS = {
    "signal_ratio": "VXN/VIX",
    "ratio_fast_ma_days": 20,
    "ratio_slow_ma_days": 252,
    "defensive_proxy": "SHY",
}
# A conservative floor for a daily-checked fast/slow-MA ratio rotation over
# the common QQQ/SHY/^VXN/^VIX calendar (common inception trimmed to SHY's
# 2002-07-30 start), same conservative-floor convention as
# fed_net_liquidity_regime_qqq.py's own EXPECTED_MIN_TRADES.
EXPECTED_MIN_TRADES = 100

_RISK, _DEFENSIVE = TRADED[0], TRADED[1]
_NUMERATOR, _DENOMINATOR = SIGNAL[0], SIGNAL[1]
_FAST = int(PARAMETERS["ratio_fast_ma_days"])
_SLOW = int(PARAMETERS["ratio_slow_ma_days"])


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED + SIGNAL; only their presence is
    checked here -- every price the rule reads comes from the truncated
    frames the engine hands over at call time, via
    ``signals.make_ratio_regime_target_fn``'s own per-call validation."""
    for key in (_RISK, _DEFENSIVE, _NUMERATOR, _DENOMINATOR):
        if key not in frames:
            raise ValueError(f"{FAMILY_ID}: make_target_fn requires a frame "
                             f"for {key}")
    return signals.make_ratio_regime_target_fn(
        risk_ticker=_RISK, defensive_ticker=_DEFENSIVE,
        numerator_ticker=_NUMERATOR, denominator_ticker=_DENOMINATOR,
        fast=_FAST, slow=_SLOW)
