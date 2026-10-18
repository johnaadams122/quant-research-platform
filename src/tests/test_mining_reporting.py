import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import reporting


def test_render_run_report_zero_survivors_states_it_explicitly():
    text = reporting.render_run_report(
        date="2026-08-27", n_considered=12, excluded_ruled_out=["a", "b"],
        excluded_missing_data=[], not_attempted_budget=[], n_passed_dsr=0,
        n_survivors=0, survivor_briefs=[], budget_used="12,000 tokens",
        budget_ceiling="300,000 tokens", n_families_tested_this_run=0,
        n_families_correction=0)
    assert "zero — none of the 12 candidates" in text
    assert "Excluded as already ruled-out (catalog status / past gate check): 2 — a, b" in text


def test_render_run_report_lists_survivor_briefs():
    text = reporting.render_run_report(
        date="2026-08-27", n_considered=5, excluded_ruled_out=[],
        excluded_missing_data=[], not_attempted_budget=[], n_passed_dsr=1,
        n_survivors=1, survivor_briefs=["candidate-brief-fam-QQQ.md"],
        budget_used="5,000", budget_ceiling="300,000",
        n_families_tested_this_run=5, n_families_correction=5)
    assert "**Survivors: candidate-brief-fam-QQQ.md**" in text


def test_render_run_report_distinguishes_tested_count_from_correction_n():
    # The correction N (a cumulative, ledger-floored value) can be far
    # larger than the number of families actually tested in this run (in
    # this synthetic case, 10 families tested against a correction N of 40).
    # The two numbers must never be printed as if they were the same figure
    # under one label.
    text = reporting.render_run_report(
        date="2026-01-15", n_considered=10, excluded_ruled_out=[],
        excluded_missing_data=[], not_attempted_budget=[], n_passed_dsr=0,
        n_survivors=0, survivor_briefs=[], budget_used="n/a (script-driven)",
        budget_ceiling="n/a", n_families_tested_this_run=10,
        n_families_correction=40.0)
    tested_line = next(l for l in text.splitlines()
                        if l.startswith("Distinct signal families tested this run"))
    correction_line = next(l for l in text.splitlines()
                            if l.startswith("Multiple-testing correction N"))
    assert tested_line == "Distinct signal families tested this run: 10"
    assert correction_line.endswith("40.00")
    # The two figures must live on separate, separately-labeled lines:
    # guards against one label claiming both numbers while only the
    # correction value is printed.
    assert tested_line != correction_line


def test_render_candidate_brief_includes_the_not_a_gate_result_disclaimer():
    text = reporting.render_candidate_brief(
        family_name="Test Family", ticker="QQQ", mechanism_hypothesis="x",
        parameters_tested="x", cost_source="x", tax_treatment="x",
        holdout_start_date="2026-06-01", dsr_value=0.97,
        n_families_correction=3, trade_count=10, other_summary_stats="x",
        holdout_window="2026-06-01 to 2026-12-01", holdout_result="pass",
        survival_case="x", account_eligibility_notes="x")
    assert text.startswith("> **NOT a gate result")
    assert "0.9700" in text


def test_brief_filename_matches_the_documented_convention():
    assert reporting.brief_filename("mean_reversion_ibs", "QQQ") == \
        "candidate-brief-mean_reversion_ibs-QQQ.md"


def test_write_run_outputs_creates_directory_and_all_files(tmp_path):
    output_dir = str(tmp_path / "2026-08-27")
    paths = reporting.write_run_outputs(
        output_dir, "REPORT TEXT", {"candidate-brief-a-QQQ.md": "BRIEF TEXT"})
    assert len(paths) == 2
    assert open(paths[0]).read() == "REPORT TEXT"
    assert open(paths[1]).read() == "BRIEF TEXT"
