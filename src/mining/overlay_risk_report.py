"""Shared risk-layer report and ledger-note paragraph for benchmark-plus-overlay
families (the risk-basis rule).

A benchmark-plus-overlay family holds SPY at all times plus an overlay. Under the
risk-basis rule, ABI-1's four risk refusals (the 0.35 absolute drawdown cap and the
three SPY-relative ones: drawdown, longest losing-week streak, worst 8-week
stretch) apply to the OVERLAY-ONLY EXCESS account, not to the whole SPY-core
account. ``abi1_driver.evaluate_family`` does that when the curve declares
``account_kind == "benchmark_plus_overlay"``; this module is what every such
family's driver uses to

  * refuse to score if the basis was not applied (`require_overlay_basis`),
    because the ledger note states the basis and must never claim one the
    score did not use;
  * report the four refusals on the excess account together with a report-only
    whole-account reference block (`risk_layer_report`), so the ceiling the
    rule removed stays visible; and
  * write the disclosure paragraph the rule requires into the permanent
    ledger note (`risk_layer_note`).

Each family states its OWN ordering honesty through ``ordering_clause`` -- when
the rule was adopted relative to that family's first ledger row differs between
families, and one shared sentence would be false for one of them.
"""
from __future__ import annotations

import numpy as np

from . import account_relative

OVERLAY_BASIS = "overlay_only_excess"


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _r4(value):
    """A number as it reads in the permanent ledger note: 4 decimals, or None."""
    value = _finite(value)
    return None if value is None else round(value, 4)


def failed_refusals(layer2: dict) -> list:
    """Name every one of the four risk refusals that failed in a layer-2 block."""
    drawdown = layer2["drawdown"]
    streak, stretch = layer2["red_week_streak"], layer2["worst_stretch"]
    weeks = int(stretch["window_weeks"])
    failed = []
    if not drawdown["relative_passed"]:
        failed.append("relative_drawdown")
    if not drawdown["absolute_passed"]:
        failed.append("absolute_drawdown_cap")
    if not streak["passed"]:
        failed.append("red_week_streak")
    if not stretch["passed"]:
        failed.append(f"worst_{weeks}_week_stretch")
    return failed


def require_overlay_basis(result: dict, *, family_id: str) -> None:
    """Raise RuntimeError unless `result` (an ``evaluate_family`` result) was
    scored with the overlay-only-excess risk basis.

    The ledger note written by `risk_layer_note` says the score used that
    basis. A curve that stopped declaring ``account_kind`` would be scored on
    the whole account (the pre-rule basis) while the note claimed otherwise,
    and a permanent one-shot row cannot be corrected afterwards.
    """
    basis = result["layer2"].get("account_basis")
    if basis != OVERLAY_BASIS:
        raise RuntimeError(
            f"the risk-basis rule: {family_id}'s risk layer must be scored on the "
            f"{OVERLAY_BASIS} account, but evaluate_family used {basis!r}; the "
            "curve must carry account_kind='benchmark_plus_overlay'. Nothing "
            "was recorded.")


# Arithmetic residue, not a return: a daily return below this is rounding noise
# (about 4500 ulps of 1.0), far under the smallest real overlay session move. It
# is a fail-closed guard's tolerance only; it never enters a score.
_OVERLAY_ONLY_RESIDUE = 1e-12


def require_excess_is_overlay_only(account_daily, benchmark_daily, overlay_mask, *,
                                   family_id: str,
                                   tolerance: float = _OVERLAY_ONLY_RESIDUE) -> None:
    """Raise unless the account earns exactly the benchmark's return on every
    session the overlay is OFF.

    The overlay-only excess account (the risk-basis rule) is only "the overlay's own
    dollars" if the account's core IS the benchmark. If the core and the
    benchmark the evaluator subtracts ever diverge (a different entry cost, a
    different price series, a different calendar fill), the excess picks up
    non-overlay P&L and the four refusals would be measuring the wrong thing.
    Outside the windows the two must agree to arithmetic residue; a real return
    there is a wiring fault, refused before anything is recorded.

    Two limits, stated rather than hidden. A mask that marks EVERY session
    overlay-on checks nothing, so it is refused (`ValueError`): the guard must
    have at least one off-session to verify. And a benchmark entry cost is
    charged on the benchmark series' FIRST bar, so when SPY lists before the
    account (production) that bar lies outside the account span in both the
    core and the benchmark and a mismatch there is invisible to this guard.
    """
    account = account_daily
    mask = overlay_mask
    if not mask.index.equals(account.index):
        raise ValueError(
            f"{family_id}: overlay_mask must sit on the account's own session index")
    bench = benchmark_daily.reindex(account.index)
    off = ~mask.to_numpy(dtype=bool)
    if not off.any():
        raise ValueError(
            f"{family_id}: overlay_mask marks no session as overlay-off, so the "
            "overlay-only excess cannot be verified")
    if bench[off].isna().any():
        raise RuntimeError(
            f"{family_id}: the benchmark has {int(bench[off].isna().sum())} missing "
            "session(s) outside the overlay windows, so the overlay-only excess "
            "cannot be verified. Nothing was recorded.")
    excess = (account[off] - bench[off]).abs()
    bad = int((excess > float(tolerance)).sum())
    if bad:
        raise RuntimeError(
            f"the risk-basis rule: {family_id}'s account differs from the benchmark on "
            f"{bad} session(s) outside the overlay windows (largest difference "
            f"{float(excess.max()):.3g}), so the excess is not overlay-only. "
            "Nothing was recorded.")


def risk_layer_report(result: dict, benchmark_daily) -> dict:
    """The risk-layer block a driver reports: the four refusals as gated (on
    the basis named in ``result["layer2"]["account_basis"]``), plus a
    report-only ``whole_account_reference`` -- the same four on the whole
    account, which gates nothing."""
    layer2 = result["layer2"]
    drawdown = layer2["drawdown"]
    streak, stretch = layer2["red_week_streak"], layer2["worst_stretch"]
    whole = account_relative.risk_refusals(result["account_daily"], benchmark_daily)
    whole_dd = whole["drawdown"]
    return {
        "account_basis": layer2["account_basis"],
        "account_max_drawdown": _finite(drawdown["account"]),
        "benchmark_max_drawdown": _finite(drawdown["benchmark"]),
        "absolute_cap": _finite(drawdown["absolute_cap"]),
        "absolute_passed": bool(drawdown["absolute_passed"]),
        "relative_passed": bool(drawdown["relative_passed"]),
        "red_week_streak_passed": bool(streak["passed"]),
        "red_week_streak_account": int(streak["account"]),
        "red_week_streak_benchmark": int(streak["benchmark"]),
        "worst_stretch_window_weeks": int(stretch["window_weeks"]),
        "worst_stretch_passed": bool(stretch["passed"]),
        "worst_stretch_account": _finite(stretch["account"]),
        "worst_stretch_benchmark": _finite(stretch["benchmark"]),
        "failed_refusals": failed_refusals(layer2),
        "all_refusals_passed": bool(layer2["all_refusals_passed"]),
        "benchmark_itself_breaches_the_absolute_cap": bool(
            drawdown["benchmark"] < drawdown["absolute_cap"]),
        "whole_account_reference": {
            "basis": whole["account_basis"],
            "gates_verdict": False,
            "account_max_drawdown": _finite(whole_dd["account"]),
            "benchmark_max_drawdown": _finite(whole_dd["benchmark"]),
            "absolute_passed": bool(whole_dd["absolute_passed"]),
            "relative_passed": bool(whole_dd["relative_passed"]),
            "red_week_streak_passed": bool(whole["red_week_streak"]["passed"]),
            "worst_stretch_passed": bool(whole["worst_stretch"]["passed"]),
            "failed_refusals": failed_refusals(whole),
            "all_refusals_passed": bool(whole["all_refusals_passed"]),
        },
    }


def risk_layer_note(risk: dict, *, ordering_clause: str) -> str:
    """The ledger-note paragraph disclosing the risk basis (the risk-basis rule), with
    this run's measured numbers. `ordering_clause` is the family's own,
    accurate statement of when the rule was adopted relative to its first
    ledger row."""
    ref = risk["whole_account_reference"]
    failed_text = ", ".join(risk["failed_refusals"]) or "none"
    whole_failed_text = ", ".join(ref["failed_refusals"]) or "none"
    spy_dd = _r4(risk["benchmark_max_drawdown"])
    if risk["benchmark_itself_breaches_the_absolute_cap"]:
        cap_clause = (
            f"in this window SPY's own drawdown ({spy_dd}) breaches the cap, so the "
            "whole SPY-core account clears it only if the overlay offsets most of that")
    else:
        cap_clause = f"in this window SPY's own drawdown ({spy_dd}) does not breach the cap"
    if [f for f in risk["failed_refusals"] if f != "absolute_drawdown_cap"]:
        relative_clause = (
            "the three SPY-relative refusals compare the overlay's own excess with SPY "
            "itself; a SPY-relative refusal fired in this run, so they were binding here")
    else:
        relative_clause = (
            "the three SPY-relative refusals compare the overlay's own excess with SPY "
            "itself and, for a small overlay measured over a window that holds a large "
            "SPY drawdown, are expected to be much less binding than on the whole "
            "account (this run's numbers above show the slack), so the risk layer here "
            "is mainly the absolute cap on the overlay's own drawdown")
    return (
        "RISK LAYER (the risk-basis rule; risk_account_basis=overlay_only_excess): "
        "ABI-1's four risk refusals (the 0.35 absolute drawdown cap, drawdown no worse than "
        "SPY's, longest losing-week streak no longer than SPY's, worst 8-week stretch no worse "
        "than SPY's) are measured on the overlay-only excess account, the account's daily "
        "arithmetic excess over SPY (the overlay's own dollars, net of costs and financing, over "
        "the prior close's pre-tax account equity, less the realization-tax drag charged on "
        "exit sessions), compounded as its own account, in units of whole-account equity, not of "
        "the overlay's own committed capital, and not on the whole SPY-core account "
        f"({cap_clause}); "
        f"{ordering_clause}; "
        f"this run, overlay-only excess account: max drawdown {_r4(risk['account_max_drawdown'])} "
        f"against SPY's {spy_dd} (cap {_r4(risk['absolute_cap'])}), "
        f"longest losing-week streak {risk['red_week_streak_account']} against "
        f"{risk['red_week_streak_benchmark']}, worst {risk['worst_stretch_window_weeks']}-week "
        f"stretch {_r4(risk['worst_stretch_account'])} against "
        f"{_r4(risk['worst_stretch_benchmark'])}; {relative_clause} "
        f"(refusals failed in this run: {failed_text}); whole-account reference, for reference "
        f"only and gating nothing: max drawdown {_r4(ref['account_max_drawdown'])} against "
        f"SPY's {_r4(ref['benchmark_max_drawdown'])}, whole-account refusals failed: "
        f"{whole_failed_text}")
