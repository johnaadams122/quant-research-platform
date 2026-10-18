"""Catalog-facing module for the pooled PMCC family (options-chain simulator family).
Calendar/DTE-rule-driven, NOT SignalFn-based -- see options_wheel.py's
own module docstring."""
from .. import constants

FAMILY_ID = "options_pmcc_pooled_liquid"
SHAPE = "pooled_options"
TRADED = list(constants.OPTIONS_LIQUID_UNDERLYINGS)
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {
    "long_target_delta": 0.80, "long_dte": 252, "short_target_delta": 0.25,
    "short_dte": 21, "long_roll_dte_floor": 63, "roll_trigger_delta": 0.60,
    "profit_target_pct": 0.5,
    # A value-based LEAPS reset alongside the time-based long_roll_dte_floor:
    # it bounds the drawdown when the long call loses most of its delta.
    # wheel_engine.run_pmcc already defaults this to 0.40 -- listed here
    # explicitly so the catalog stays the single source of truth for every
    # production-tunable PMCC parameter, matching this dict's own role
    # (scripts/run_options_mining.py passes PARAMETERS to run_pmcc as
    # keyword arguments rather than relying on run_pmcc's own defaults).
    "long_reset_delta": 0.40,
}
