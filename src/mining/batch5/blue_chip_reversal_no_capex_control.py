"""Batch6 candidate: blue_chip_reversal_no_capex_control (pooled_single_asset).

MANDATORY control arm for ``capex_cycle_blue_chip_dip_pooled.py``, not a
separate hypothesis -- IDENTICAL watchlist, IDENTICAL underperformance
condition/window, IDENTICAL exit rule, with the capex-growth condition
REMOVED entirely ("buy blue-chip losers vs SPY, no capex filter"). This
is the falsification check for the main family's own honest residual
risk: any edge found in the capex-conditioned arm could simply be
ordinary price-reversal (De Bondt-Thaler and descendants) with the capex
filter contributing nothing, or even actively hurting (Titman/Wei/Xie
2004 and Cooper/Gulen/Schill 2008 both find high-capex firms earn WORSE
forward returns unconditionally, so conditioning on capex growth could be
removing exactly the highest-quality "cheap, safe" losers). If the
capex-conditioned arm does not beat this control arm, that is the honest
answer -- capex added nothing -- reported either way, never spun.

A FALSIFICATION CHECK, NOT A SEPARATE CANDIDACY: its purpose is
falsifying the capex condition specifically, not independent candidacy.
It still has its own catalog entry, and ``scripts/run_capex_cycle_mining.py``
scores it as its own ABI-1 trial (its own DSR score and ledger record)
in the same run as the main family.

Shares its WATCHLIST (``TRADED``) and every threshold with
``capex_cycle_blue_chip_dip_pooled.py`` by construction (imported
directly, never independently duplicated -- a drift between the two
would silently invalidate the control's whole purpose).
"""
from __future__ import annotations

from mining.batch5 import capex_cycle_blue_chip_dip_pooled as _main

FAMILY_ID = "blue_chip_reversal_no_capex_control"
SHAPE = "pooled_single_asset"
BENCHMARK = _main.BENCHMARK
TRADED = list(_main.TRADED)
SIGNAL: list = []
SNAPSHOTS: dict = {}

# Capex-specific keys this control arm's own rule never uses (it removes
# the capex condition entirely) -- everything else is inherited LIVE from
# _main.PARAMETERS (a
# key-by-key hand copy here would silently omit any NEW threshold a
# future edit adds to the main module, a metadata-drift risk even though
# it can never affect actual trading decisions, since the signal logic
# itself always delegates to _main._make_signal_fn).
_CAPEX_ONLY_KEYS = {"capex_growth_threshold", "capex_growth_source"}
PARAMETERS = {
    key: value for key, value in _main.PARAMETERS.items()
    if key not in _CAPEX_ONLY_KEYS
}
PARAMETERS["role"] = "free_falsification_check_control_arm_for_capex_cycle_blue_chip_dip_pooled"
EXPECTED_MIN_TRADES = 0  # same open dependency as the main family

decision_dates = _main.decision_dates          # re-exported for direct testing
underperformance_spread = _main.underperformance_spread


def make_signal_fn(ticker: str, index, frames: dict, *, fundamentals=None):
    """Identical to the main family's own factory except
    ``require_capex=False`` -- ``fundamentals`` is accepted only for
    call-site symmetry with the main module (never read)."""
    return _main._make_signal_fn(
        ticker, index, frames, fundamentals, require_capex=False)
