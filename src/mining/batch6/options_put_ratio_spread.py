"""Catalog-facing module for the pooled put ratio spread family (options-chain
simulator family). Calendar/DTE-rule-driven, NOT SignalFn-based -- see
options_wheel.py's own module docstring."""
from .. import constants

FAMILY_ID = "options_put_ratio_spread_pooled_liquid"
SHAPE = "pooled_options"
TRADED = list(constants.OPTIONS_LIQUID_UNDERLYINGS)
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {
    "long_delta": 0.30, "short_delta": 0.15, "short_multiple": 2,
    "dte": 21, "profit_target_pct": 0.5,
}
