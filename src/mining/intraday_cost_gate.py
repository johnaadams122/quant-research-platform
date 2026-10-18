"""Pre-registered cost-realism gate for intraday/swing and day-trading
candidates.

This is a PURE decision function -- it does not fetch live quotes itself.
The orchestrating scripts (scripts/run_swing_mining.py,
scripts/run_day_trading_mining.py) are responsible for capturing real
bid/ask and calling this gate with them, matching this pipeline's "never
invent a cost or a candidate's data" rule and the same separation of
concerns as cost_tax_inputs.py's existing pure functions.

Boundary convention (verified against this pipeline's own established
gates, not picked arbitrarily):
- Spread ceiling: a MAXIMUM-allowed value. This project's existing
  ceiling-style gate (scripts/run_batch3_mining.py's
  selected_execution_costs, `if spread >= quote_gate_bps: raise`) fails
  at-or-above the ceiling -- i.e. the spread must be strictly BELOW the
  ceiling to pass. Matched here: spread_bps >= INTRADAY_SPREAD_CEILING_BPS
  fails.
- Edge-over-cost multiple: a MINIMUM-required value, same shape as
  DSR_PASS_THRESHOLD (`passed_dsr = dsr_value >= constants.
  DSR_PASS_THRESHOLD`, scripts/run_index_addition_mining.py) and
  MIN_TRADES_FOR_SCORING (`if trade_count < constants.
  MIN_TRADES_FOR_SCORING: cut`, scripts/run_batch3_mining.py) -- both
  pass exactly at the boundary. Matched here: net_edge_bps >=
  INTRADAY_EDGE_OVER_COST_MULTIPLE * spread_bps passes.
"""
from __future__ import annotations

import math

from . import constants
from . import cost_tax_inputs


def evaluate_cost_realism_gate(
    entry_bid: float, entry_ask: float,
    exit_bid: float, exit_ask: float,
    net_edge_bps: float,
) -> dict:
    """Pre-registered hard pass/fail gate for a short-hold (intraday/
    swing) candidate, using ALREADY-CAPTURED live quotes (never invents
    or estimates a spread).

    Two checks, in order:
    1. The round-trip spread (session_round_trip over the entry/exit
       quotes) must be strictly below constants.INTRADAY_SPREAD_CEILING_BPS.
       At or above the ceiling: proceed=False.
    2. net_edge_bps must be at least
       constants.INTRADAY_EDGE_OVER_COST_MULTIPLE times the spread. Below
       that: proceed=False.

    Returns a dict:
      {'proceed': bool, 'reason': str, 'spread_bps': float,
       'edge_over_cost_ratio': float}

    `reason` always carries the actual measured numbers, never a bare
    code -- this pipeline's convention (catalog.json's ruled_out_reason,
    trial_ledger's outcome notes) is that every skip/cut is logged
    verbatim with a real, specific, human-readable reason.

    Raises ValueError if net_edge_bps is not a finite number (a
    candidate's own bad backtest output must never silently pass this
    gate as if it were real); if any of the four quote inputs is not a
    finite, positive number; or if either quote is crossed (bid > ask).
    A crossed/NaN/non-positive quote is bad data, not a real market
    state -- this pipeline fails closed on bad data rather than let it
    silently produce a number. This matters concretely here: a crossed
    quote produces a NEGATIVE spread_bps, which would otherwise sail
    through both threshold checks below (always under the ceiling, and
    compared against net_edge_bps as if it were a real cost) and
    incorrectly ADMIT a candidate as passing the gate.
    """
    if not isinstance(net_edge_bps, (int, float)) or not math.isfinite(net_edge_bps):
        raise ValueError(
            f"net_edge_bps must be a finite number, got {net_edge_bps!r}")

    for name, value in (
        ("entry_bid", entry_bid), ("entry_ask", entry_ask),
        ("exit_bid", exit_bid), ("exit_ask", exit_ask),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")

    if entry_bid > entry_ask:
        raise ValueError(
            f"entry quote is crossed: entry_bid={entry_bid!r} > "
            f"entry_ask={entry_ask!r}")
    if exit_bid > exit_ask:
        raise ValueError(
            f"exit quote is crossed: exit_bid={exit_bid!r} > "
            f"exit_ask={exit_ask!r}")

    entry_spread_bps = cost_tax_inputs.compute_spread_bps(entry_bid, entry_ask)
    exit_spread_bps = cost_tax_inputs.compute_spread_bps(exit_bid, exit_ask)
    spread_bps = cost_tax_inputs.session_round_trip(entry_spread_bps, exit_spread_bps)

    if spread_bps >= constants.INTRADAY_SPREAD_CEILING_BPS:
        return {
            "proceed": False,
            "reason": (
                f"spread {spread_bps:.2f} bps >= ceiling "
                f"{constants.INTRADAY_SPREAD_CEILING_BPS} bps"
            ),
            "spread_bps": spread_bps,
            "edge_over_cost_ratio": float("nan") if spread_bps == 0 else net_edge_bps / spread_bps,
        }

    required_edge_bps = constants.INTRADAY_EDGE_OVER_COST_MULTIPLE * spread_bps
    edge_over_cost_ratio = (
        float("inf") if spread_bps == 0 else net_edge_bps / spread_bps
    )

    if net_edge_bps < required_edge_bps:
        return {
            "proceed": False,
            "reason": (
                f"net edge {net_edge_bps:.2f} bps is below "
                f"{constants.INTRADAY_EDGE_OVER_COST_MULTIPLE}x round-trip "
                f"spread ({spread_bps:.2f} bps -> requires "
                f"{required_edge_bps:.2f} bps, ratio={edge_over_cost_ratio:.2f})"
            ),
            "spread_bps": spread_bps,
            "edge_over_cost_ratio": edge_over_cost_ratio,
        }

    return {
        "proceed": True,
        "reason": (
            f"spread {spread_bps:.2f} bps within ceiling "
            f"{constants.INTRADAY_SPREAD_CEILING_BPS} bps and net edge "
            f"{net_edge_bps:.2f} bps clears the "
            f"{constants.INTRADAY_EDGE_OVER_COST_MULTIPLE}x cost margin "
            f"(ratio={edge_over_cost_ratio:.2f})"
        ),
        "spread_bps": spread_bps,
        "edge_over_cost_ratio": edge_over_cost_ratio,
    }
