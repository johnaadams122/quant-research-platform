"""Tests for mining.overlay_risk_report -- the shared risk-layer report and
ledger-note paragraph for benchmark-plus-overlay families. All data is deterministic and synthetic; no network, no clock."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import account_relative as ar  # noqa: E402
from mining import overlay_risk_report as orr  # noqa: E402

_ORDERING = "the rule was fixed before this family's first ledger row"


def _crash_benchmark(n=2400, seed=99):
    """SPY-like series with real volatility and a 40-session crash (about -55 percent)."""
    idx = pd.bdate_range("2015-01-02", periods=n)
    rng = np.random.default_rng(seed)
    bench = pd.Series(0.00035 + 0.011 * rng.standard_normal(n), index=idx)
    bench.iloc[600:640] = -0.02
    return bench


def _overlay_result(*, basis="overlay_only_excess", overlay_loss=False):
    """(evaluate() result, benchmark) for an SPY-core account plus a small
    overlay that is on 3 sessions of every 21. With `overlay_loss` the overlay
    loses instead of winning, so a refusal can fail."""
    bench = _crash_benchmark()
    idx = bench.index
    rng = np.random.default_rng(5)
    excess = pd.Series(0.0, index=idx)
    sign = -1.0 if overlay_loss else 1.0
    for start in range(0, len(idx), 21):
        block = excess.iloc[start:start + 3]
        excess.iloc[start:start + 3] = sign * 0.004 + 0.0005 * rng.standard_normal(len(block))
    result = ar.evaluate(
        strategy_daily=bench + excess, in_market=pd.Series(True, index=idx),
        benchmark_daily=bench, cash_daily=pd.Series(0.0, index=idx),
        n_trials=100.0, sr_variance=0.0002, tax_policy="none",
        risk_account_basis=basis)
    return result, bench


# ------------------------------------------------------------ failed_refusals

def _clean_layer2():
    result, _ = _overlay_result()
    return json.loads(json.dumps(result["layer2"]))


@pytest.mark.parametrize("which", ["relative_drawdown", "absolute_drawdown_cap",
                                   "red_week_streak", "worst_8_week_stretch"])
def test_failed_refusals_names_exactly_the_refusal_that_failed(which):
    layer2 = _clean_layer2()
    assert orr.failed_refusals(layer2) == []
    if which == "relative_drawdown":
        layer2["drawdown"]["relative_passed"] = False
    elif which == "absolute_drawdown_cap":
        layer2["drawdown"]["absolute_passed"] = False
    elif which == "red_week_streak":
        layer2["red_week_streak"]["passed"] = False
    else:
        layer2["worst_stretch"]["passed"] = False
    assert orr.failed_refusals(layer2) == [which]


def test_failed_refusals_names_the_window_length_actually_used():
    layer2 = _clean_layer2()
    layer2["worst_stretch"].update(passed=False, window_weeks=12)
    assert orr.failed_refusals(layer2) == ["worst_12_week_stretch"]


# ----------------------------------------------------- require_overlay_basis

def test_require_overlay_basis_passes_an_overlay_only_excess_result():
    result, _ = _overlay_result()
    orr.require_overlay_basis(result, family_id="some_family")


def test_require_overlay_basis_refuses_a_whole_account_result_naming_the_family():
    result, _ = _overlay_result(basis="whole_account")
    with pytest.raises(RuntimeError, match="some_family") as caught:
        orr.require_overlay_basis(result, family_id="some_family")
    assert "overlay_only_excess" in str(caught.value) and "account_kind" in str(caught.value)


# --------------------------------------------------------- risk_layer_report

def test_risk_layer_report_gates_on_the_excess_account_and_references_the_whole_account():
    result, bench = _overlay_result()
    risk = orr.risk_layer_report(result, bench)
    ref = risk["whole_account_reference"]
    assert risk["account_basis"] == "overlay_only_excess"
    assert ref["basis"] == "whole_account" and ref["gates_verdict"] is False
    excess = result["account_daily"] - bench.reindex(result["account_daily"].index)
    assert risk["account_max_drawdown"] == pytest.approx(ar.max_drawdown(excess), abs=1e-12)
    assert ref["account_max_drawdown"] == pytest.approx(ar.max_drawdown(result["account_daily"]), abs=1e-12)
    assert risk["account_max_drawdown"] != pytest.approx(ref["account_max_drawdown"])
    direct = ar.risk_refusals(result["account_daily"], bench)
    assert ref["absolute_passed"] == direct["drawdown"]["absolute_passed"]
    assert ref["all_refusals_passed"] == direct["all_refusals_passed"]
    assert ref["failed_refusals"] == orr.failed_refusals(direct)
    # the SPY-core ceiling is a whole-account fact, and only the reference shows it
    assert ref["absolute_passed"] is False and risk["absolute_passed"] is True
    assert risk["benchmark_itself_breaches_the_absolute_cap"] is True
    assert risk["all_refusals_passed"] is True and risk["failed_refusals"] == []
    json.dumps(risk, allow_nan=False)


def test_risk_layer_report_names_a_failed_refusal_on_the_excess_account():
    result, bench = _overlay_result(overlay_loss=True)
    risk = orr.risk_layer_report(result, bench)
    assert risk["all_refusals_passed"] is False
    assert risk["failed_refusals"] == orr.failed_refusals(result["layer2"]) != []


def test_risk_layer_report_reads_the_streak_and_stretch_blocks_of_layer_two():
    """Distinct values, so a swapped key or a wrong source cannot pass."""
    result, bench = _overlay_result()
    result["layer2"]["red_week_streak"].update(passed=False, account=7, benchmark=5)
    result["layer2"]["worst_stretch"].update(passed=True, account=-0.0421, benchmark=-0.0313,
                                             window_weeks=8)
    risk = orr.risk_layer_report(result, bench)
    assert risk["red_week_streak_passed"] is False
    assert (risk["red_week_streak_account"], risk["red_week_streak_benchmark"]) == (7, 5)
    assert risk["worst_stretch_passed"] is True
    assert risk["worst_stretch_account"] == pytest.approx(-0.0421)
    assert risk["worst_stretch_benchmark"] == pytest.approx(-0.0313)
    assert risk["worst_stretch_window_weeks"] == 8


# ----------------------------------------------------------- risk_layer_note

def test_risk_layer_note_states_the_rule_the_basis_and_this_runs_numbers():
    result, bench = _overlay_result()
    risk = orr.risk_layer_report(result, bench)
    note = orr.risk_layer_note(risk, ordering_clause=_ORDERING)
    lowered = note.lower()
    for phrase in ("risk layer (the risk-basis rule", "risk_account_basis=overlay_only_excess",
                   "overlay-only excess account", "0.35 absolute drawdown cap",
                   "spy-relative refusals", "longest losing-week streak", "worst 8-week stretch",
                   "whole-account reference", "gating nothing", "refusals failed in this run",
                   _ORDERING):
        assert phrase in lowered, phrase
    assert f"max drawdown {round(risk['account_max_drawdown'], 4)}" in note
    ref = risk["whole_account_reference"]
    assert f"max drawdown {round(ref['account_max_drawdown'], 4)}" in note
    assert f"streak {risk['red_week_streak_account']} against {risk['red_week_streak_benchmark']}" in note


def test_risk_layer_note_states_what_the_risk_layer_gates_on():
    result, bench = _overlay_result()
    note = orr.risk_layer_note(orr.risk_layer_report(result, bench), ordering_clause=_ORDERING).lower()
    for phrase in ("the risk layer here is mainly the absolute cap on the overlay's own drawdown",
                   "for reference only and gating nothing"):
        assert phrase in note, phrase


def test_risk_layer_note_names_the_failed_refusals_of_each_block():
    result, bench = _overlay_result(overlay_loss=True)
    risk = orr.risk_layer_report(result, bench)
    note = orr.risk_layer_note(risk, ordering_clause=_ORDERING)
    assert "refusals failed in this run: " + ", ".join(risk["failed_refusals"]) in note
    ref_failed = risk["whole_account_reference"]["failed_refusals"]
    assert "whole-account refusals failed: " + (", ".join(ref_failed) or "none") in note


def test_risk_layer_note_states_none_when_nothing_failed():
    result, bench = _overlay_result()
    risk = orr.risk_layer_report(result, bench)
    assert risk["failed_refusals"] == []
    assert "refusals failed in this run: none" in orr.risk_layer_note(risk, ordering_clause=_ORDERING)


def test_the_ordering_clause_is_required_and_appears_verbatim():
    result, bench = _overlay_result()
    risk = orr.risk_layer_report(result, bench)
    with pytest.raises(TypeError):
        orr.risk_layer_note(risk)                       # every family must say its own ordering honestly
    other = "this family's only recorded outcome predates the rule"
    assert other in orr.risk_layer_note(risk, ordering_clause=other)


# ------------------------------------- note wording

def test_risk_layer_note_states_the_scale_of_the_excess_account_and_the_tax_term():
    """The excess account is measured in units of WHOLE-account
    equity, not of the overlay's own committed capital, and it is only 'the
    overlay's dollars' before the realization-tax drag on exit sessions."""
    result, bench = _overlay_result()
    note = orr.risk_layer_note(orr.risk_layer_report(result, bench), ordering_clause=_ORDERING)
    for phrase in ("in units of whole-account equity",
                   "not of the overlay's own committed capital",
                   "less the realization-tax drag charged on exit sessions"):
        assert phrase in note, phrase


def test_risk_layer_note_states_how_binding_the_relative_refusals_are_expected_to_be():
    """The SPY-relative refusals can fire on the excess basis (on a
    calm-benchmark window, for instance), so how much they bind is a property
    of the window, not of the rule. The note says what is expected and points
    at this run's own numbers."""
    result, bench = _overlay_result()
    note = orr.risk_layer_note(orr.risk_layer_report(result, bench), ordering_clause=_ORDERING)
    lowered = note.lower()
    assert "expected to be much less binding than on the whole account" in lowered
    assert "this run's numbers above" in lowered


# ---------------------------------------- require_excess_is_overlay_only

def _overlay_account(n=400):
    idx = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(3)
    bench = pd.Series(0.0004 + 0.01 * rng.standard_normal(n), index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[::21] = True
    mask.iloc[1::21] = True
    excess = pd.Series(0.0, index=idx)
    excess[mask] = 0.003
    return bench + excess, bench, mask


def test_require_excess_is_overlay_only_passes_when_the_excess_is_zero_outside_the_windows():
    acct, bench, mask = _overlay_account()
    acct[~mask] = bench[~mask]                        # exactly the benchmark outside the windows
    orr.require_excess_is_overlay_only(acct, bench, mask, family_id="f")


def test_require_excess_is_overlay_only_tolerates_arithmetic_residue_but_not_a_return():
    acct, bench, mask = _overlay_account()
    acct[~mask] = bench[~mask]
    quiet = acct.copy()
    quiet[~mask] = quiet[~mask] + 1e-16               # rounding residue
    orr.require_excess_is_overlay_only(quiet, bench, mask, family_id="f")
    loud = acct.copy()
    loud.iloc[np.flatnonzero(~mask.to_numpy())[5]] += 1e-6      # a real 0.0001 percent return
    with pytest.raises(RuntimeError, match="f") as caught:
        orr.require_excess_is_overlay_only(loud, bench, mask, family_id="f")
    assert "outside" in str(caught.value) and "1 session" in str(caught.value)


def test_require_excess_is_overlay_only_refuses_a_missing_benchmark_session():
    acct, bench, mask = _overlay_account()
    acct[~mask] = bench[~mask]
    holed = bench.copy()
    holed.iloc[np.flatnonzero(~mask.to_numpy())[3]] = np.nan
    with pytest.raises(RuntimeError, match="benchmark"):
        orr.require_excess_is_overlay_only(acct, holed, mask, family_id="f")


def test_require_excess_is_overlay_only_refuses_a_mask_on_a_different_index():
    acct, bench, mask = _overlay_account()
    with pytest.raises(ValueError, match="overlay_mask"):
        orr.require_excess_is_overlay_only(acct, bench, mask.iloc[:-1], family_id="f")


# ------------------------------------------- further guard tests

def test_require_excess_is_overlay_only_refuses_an_account_below_the_benchmark():
    """The guard compares the ABSOLUTE off-window excess: an account that
    earns LESS than the benchmark off-window (a stale price, an extra cost)
    must be refused too, not only one that earns more."""
    acct, bench, mask = _overlay_account()
    acct[~mask] = bench[~mask]
    below = acct.copy()
    below.iloc[np.flatnonzero(~mask.to_numpy())[5]] -= 1e-6
    with pytest.raises(RuntimeError, match="not overlay-only"):
        orr.require_excess_is_overlay_only(below, bench, mask, family_id="f")


def test_require_excess_is_overlay_only_refuses_a_mask_with_no_off_sessions_it_could_verify():
    """With every session marked overlay-on the guard would check nothing and
    pass whatever the account did, so such a mask is refused."""
    acct, bench, mask = _overlay_account()
    with pytest.raises(ValueError, match="no session"):
        orr.require_excess_is_overlay_only(acct, bench, pd.Series(True, index=acct.index), family_id="f")


def test_risk_layer_report_carries_each_refusals_own_pass_flag():
    """Each pass flag is read from its own refusal block, so a flag
    hard-coded True (or read from the wrong key) fails here."""
    result, bench = _overlay_result(overlay_loss=True)
    for block, key, field in (("drawdown", "absolute_passed", "absolute_passed"),
                              ("drawdown", "relative_passed", "relative_passed"),
                              ("red_week_streak", "passed", "red_week_streak_passed"),
                              ("worst_stretch", "passed", "worst_stretch_passed")):
        tampered = dict(result)
        tampered["layer2"] = json.loads(json.dumps(result["layer2"]))
        tampered["layer2"][block][key] = False
        assert orr.risk_layer_report(tampered, bench)[field] is False, field
        tampered["layer2"][block][key] = True
        assert orr.risk_layer_report(tampered, bench)[field] is True, field


def _note(**risk_overrides):
    result, bench = _overlay_result()
    risk = orr.risk_layer_report(result, bench)
    risk.update(risk_overrides)
    return orr.risk_layer_note(risk, ordering_clause=_ORDERING), risk


def test_risk_layer_note_states_the_measured_spy_drawdown_and_makes_no_hardcoded_2008_claim():
    """A family's data can start after 2008, so the note makes no fixed claim
    about SPY in 2008: it states what THIS window measured."""
    note, risk = _note()
    assert "2008" not in note
    assert risk["benchmark_itself_breaches_the_absolute_cap"] is True
    assert f"in this window SPY's own drawdown ({round(risk['benchmark_max_drawdown'], 4)}) breaches the cap" in note
    assert "whole SPY-core account clears it only if the overlay offsets most of that" in note
    calm, _ = _note(benchmark_itself_breaches_the_absolute_cap=False, benchmark_max_drawdown=-0.12)
    assert "in this window SPY's own drawdown (-0.12) does not breach the cap" in calm
    assert "clears it only if" not in calm


def test_risk_layer_note_does_not_claim_slack_on_a_run_where_a_spy_relative_refusal_fired():
    """The phrase 'this run's numbers above show the slack' must not be
    unconditional, since it is false on exactly the run where a relative
    refusal bit."""
    quiet, risk = _note()
    assert risk["failed_refusals"] == []
    assert "this run's numbers above show the slack" in quiet
    for fired in ("relative_drawdown", "red_week_streak", "worst_8_week_stretch"):
        loud, _ = _note(failed_refusals=[fired], all_refusals_passed=False)
        assert "this run's numbers above show the slack" not in loud, fired
        assert "a SPY-relative refusal fired in this run" in loud, fired
        assert f"refusals failed in this run: {fired}" in loud, fired
    only_cap, _ = _note(failed_refusals=["absolute_drawdown_cap"], all_refusals_passed=False)
    assert "this run's numbers above show the slack" in only_cap       # the cap is not a SPY-relative refusal


def test_risk_layer_note_states_the_compounding_and_the_pre_tax_equity_base():
    """The note states that the excess is compounded as its own account and is
    formed over the PRE-tax prior-close equity; the tax drag is applied after."""
    note, _ = _note()
    assert "compounded as its own account" in note
    assert "pre-tax account equity" in note
