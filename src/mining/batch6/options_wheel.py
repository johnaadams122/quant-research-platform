"""Catalog-facing module for the pooled cash-secured-put Wheel family
(options-chain simulator family). Calendar/DTE-rule-driven, NOT SignalFn-based -- unlike
a src/mining/batch5/*.py module there is no make_signal_fn here; the
bespoke driver (scripts/run_options_mining.py) calls
wheel_engine.run_wheel directly per pool member.
"""
from .. import constants

FAMILY_ID = "options_wheel_pooled_liquid"
SHAPE = "pooled_options"
TRADED = list(constants.OPTIONS_LIQUID_UNDERLYINGS)
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {"target_delta": 0.15, "dte": 21, "profit_target_pct": 0.5}
