"""Batch5 candidate: usd_strength_risk_regime (rotation_two_asset).

The mechanism and its parameters are the ones recorded in
``data/mining/catalog.json``. The module uses the batch5
``rotation_two_asset`` shape, which holds a real second (defensive) leg:
a two-leg SPY/SHY rotation, so the catalog's "move into SHY" defensive leg
is a real SHY position (a single-ticker long/flat ``SignalFn``,
``mining.backtest_engine.SignalFn``, could only express it as going flat).
The decision rule is the shared
``mining.signals.make_moving_average_regime_target_fn`` (the factory the
other fast/slow-MA rotation families in this directory use, e.g.
``skew_index_tail_risk_regime.py``/``vxn_vix_spread_derisking_regime.py``),
called with the catalog entry's parameters.

Mechanism (``data/mining/catalog.json`` entry for this family_id,
verbatim; Bruno, V. & Shin, H.S. (2015), "Cross-Border Banking and
Global Liquidity", Review of Economic Studies 82(2), 535-564): track
UUP's (Invesco DB US Dollar Index Bullish Fund) closing price against
its own 200-day simple moving average as a proxy for broad US dollar
strength. When UUP closes above its 200-day average (dollar in a
sustained uptrend -- historically coinciding with tighter global USD
funding conditions and risk-asset stress), move out of SPY into a
defensive cash/short-Treasury allocation (SHY). When UUP closes below
its 200-day average (dollar weakening or range-bound -- easier global
liquidity), hold SPY at full weight. Recompute the signal daily
(``PARAMETERS["rebalance"] == "daily_signal_check"``).

``make_target_fn`` below wraps EXACTLY:

    signals.make_moving_average_regime_target_fn(
        risk_ticker=TRADED[0], defensive_ticker=TRADED[1],
        signal_ticker="UUP", fast=1, slow=200)

``fast=1`` is the deliberate way this shared factory
expresses "today's raw UUP close versus UUP's own trailing 200-day
SMA", not a two-window moving-average cross: the factory computes
``values.iloc[-fast:].mean()`` for its fast leg, and a length-1 window
rolling mean degenerates algebraically to that single (today's) value
-- verified both symbolically and numerically in this module's own
test file, not merely asserted from this comment. Combined with the
factory's own decision rule (``risk_ticker if fast_ma <= slow_ma else
defensive_ticker``): UUP's close <= its 200-day SMA selects SPY (the
catalog's "dollar weakening or range-bound -> hold SPY"), and UUP's
close > its 200-day SMA selects SHY (the catalog's "dollar in a
sustained uptrend -> move into SHY") -- an exact match to the catalog
mechanism text above, confirmed by this module's own test.

That factory's returned callable already takes ``frames:
dict[str, pd.DataFrame]`` (truncated through the decision bar) as its
single per-call argument and performs its own alignment/validation
internally (``mining.signals._aligned_frames``) -- exactly the batch5
rotation_two_asset per-bar contract, so ``make_target_fn`` below adds
nothing beyond the driver's own required one-time presence check on
the full-history frames it is handed once (the same "thin wrapper"
convention as ``skew_index_tail_risk_regime.py``/
``fed_net_liquidity_regime_qqq.py``).

BENCHMARK is hard-pinned to "SPY" per this pipeline's standing rule:
ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
catalog "benchmark" field (see ``scripts/run_abi1_mining.py``'s own
``MARKET_BENCHMARK`` and every other batch5 module's identical
comment on this point). For this family SPY is also the risk leg, so
there is no actual drift here -- the comment exists only so a future
reader never treats this attribute as configurable.

Data window: UUP is the youngest of the three series (the vendor's first
bar can lag the fund's cited issuer inception by a few days), so UUP's
own data start is the binding constraint on the aligned SPY/SHY/UUP
calendar. EXPECTED_MIN_TRADES is a conservative floor, same convention as
every sibling module in this directory.
"""
from __future__ import annotations

import pandas as pd

from mining import signals

FAMILY_ID = "usd_strength_risk_regime"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "SHY"]      # [risk, defensive]
SIGNAL = ["UUP"]             # external price-frame ticker the generic
                              # dispatch auto-loads -- broad-dollar proxy,
                              # not traded
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (verified via json.dumps(sort_keys=True) byte-identity in this
# module's own test, which reads that file directly rather than
# duplicating a value here that could drift).
PARAMETERS = {
    "uup_ma_days": 200,
    "defensive_proxy": "SHY",
    "rebalance": "daily_signal_check",
}
_FAST = 1
_SLOW = int(PARAMETERS["uup_ma_days"])
# A conservative floor for a daily-checked fast/slow-MA rotation over the
# aligned calendar (bound by UUP's own data start), same convention as
# every sibling module in this directory.
EXPECTED_MIN_TRADES = 100


def make_target_fn(frames: dict[str, pd.DataFrame]):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED + SIGNAL; only their presence is
    checked here -- every price the rule reads comes from the truncated
    frames the engine hands over at call time, via
    ``signals.make_moving_average_regime_target_fn``'s own per-call
    validation."""
    for key in (TRADED[0], TRADED[1], SIGNAL[0]):
        if key not in frames:
            raise ValueError(
                f"{FAMILY_ID}: make_target_fn requires a frame for {key}")
    return signals.make_moving_average_regime_target_fn(
        risk_ticker=TRADED[0], defensive_ticker=TRADED[1],
        signal_ticker=SIGNAL[0], fast=_FAST, slow=_SLOW)
