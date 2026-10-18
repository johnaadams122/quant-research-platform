"""Cost and tax inputs for a mining candidate (the cost-input stage).

Implements the tiny pure spread-to-bps formulas directly, so this module
does not depend on any external project or archived artifact.

The STCG tax rate is NOT taken from a caller-supplied argument with no
internal source. The mining pipeline's own pre-registered constant
(mining.constants.STCG_TAX_RATE) fills that role, pinned from a cited
source rather than a live lookup.
"""
from __future__ import annotations

from . import constants


def compute_spread_bps(bid: float, ask: float) -> float:
    """Quoted spread in bps of mid price. Standard quoted-spread
    formula."""
    mid = (bid + ask) / 2.0
    if mid <= 0:
        raise ValueError(f"non-positive mid price from bid={bid}, ask={ask}")
    return (ask - bid) / mid * 1e4


def session_round_trip(entry_spread_bps: float, exit_spread_bps: float) -> float:
    """Round-trip cost estimate: half the entry spread plus half the exit
    spread, in bps. Standard round-trip spread formula."""
    return entry_spread_bps / 2.0 + exit_spread_bps / 2.0


def cost_bps_from_quotes(entry_bid: float, entry_ask: float,
                          exit_bid: float, exit_ask: float) -> float:
    """Convenience wrapper: round-trip cost in bps from raw entry/exit
    bid-ask quotes."""
    entry_bps = compute_spread_bps(entry_bid, entry_ask)
    exit_bps = compute_spread_bps(exit_bid, exit_ask)
    return session_round_trip(entry_bps, exit_bps)


def tax_haircut_rate_for(account_type: str) -> float:
    """The flat STCG haircut rate to pass to backtest_engine.run_backtest
    for a candidate targeting `account_type`. 0.0 for tax-advantaged
    accounts; the pre-registered STCG_TAX_RATE for 'taxable'. Raises on an
    unrecognized account_type rather than silently defaulting -- that is a
    catalog data-entry bug, not a 0%-tax candidate."""
    if account_type in ("roth", "ira", "hsa"):
        return 0.0
    if account_type == "taxable":
        return constants.STCG_TAX_RATE
    raise ValueError(f"unrecognized account_type: {account_type!r}")
