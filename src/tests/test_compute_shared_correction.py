"""Tests for scripts/compute_shared_correction.py -- the mining pipeline's
multiple-testing correction barrier.

Why this file exists: the correction-N logic must live in a real, executable
file so it can be tested. Two defect classes are pinned here: reading the
PRE-run ledger count (which under-corrects every run) and falling back to
n_trials = 1.0, i.e. NO correction at all, when an upstream step fails.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import compute_shared_correction as csc
from mining import constants, trial_ledger


def _artifact(tmp_path, name, returns, dates=None):
    path = tmp_path / (name + ".json")
    if dates is None:
        dates = ["2020-01-%02d" % (i + 1) for i in range(len(returns))]
    path.write_text(json.dumps({"returns": list(returns), "dates": list(dates)}))
    return str(path)


def _manifest(tmp_path, entries):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entries))
    return str(path)


def test_floor_includes_this_runs_new_families(tmp_path):
    """THE PRE-RUN-COUNT REGRESSION TEST. A run first-testing 3 new families against a
    ledger holding 5 must be corrected at 8, not at the pre-run 5."""
    ledger = str(tmp_path / "ledger.jsonl")
    for i in range(5):
        trial_ledger.record_first_test(
            ledger, "old_%d" % i, "2026-01-01", "s", "e", [], "r0")
    entries = [
        {"family_id": "new_%d" % i,
         "daily_returns_path": _artifact(
             tmp_path, "new_%d" % i, [0.01 * (i + 1), -0.02, 0.03, 0.005])}
        for i in range(3)
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["prior_family_count"] == 5
    assert out["n_new_families"] == 3
    assert out["ledger_floor"] == 8
    assert out["n_trials"] == max(out["n_this_run"], 8.0, 1.0)


def test_already_tested_families_are_not_recharged(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(
        ledger, "fam_a", "2026-01-01", "s", "e", [], "r0")
    entries = [
        {"family_id": "fam_a",
         "daily_returns_path": _artifact(tmp_path, "a", [0.01, -0.02, 0.03])},
        {"family_id": "fam_b",
         "daily_returns_path": _artifact(tmp_path, "b", [-0.01, 0.02, 0.01])},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["prior_family_count"] == 1
    assert out["n_new_families"] == 1
    assert out["ledger_floor"] == 2


def test_reports_candidate_identity_for_caller_cross_check(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "zeta",
         "daily_returns_path": _artifact(tmp_path, "z", [0.01, -0.02, 0.03])},
        {"family_id": "alpha",
         "daily_returns_path": _artifact(tmp_path, "a", [-0.01, 0.02, 0.01])},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["n_candidates"] == 2
    # Sorted, so the caller can compare it exactly against its own list and
    # detect a dropped or mangled manifest entry.
    assert out["family_ids"] == ["alpha", "zeta"]


def test_sr_variance_prefers_the_pooled_estimate_when_available(tmp_path):
    # The pooled estimate is preferred as the most principled one. Seed the
    # ledger with enough real-scored families that pooled_sr_variance stops
    # returning None; the barrier must use THAT, not this run's own
    # (here: single-candidate, degenerate) sample or the floor.
    ledger = str(tmp_path / "ledger.jsonl")
    pooled_values = [0.2, 0.5, 0.9, 0.1, 0.6, 0.3, 0.7, 0.4, 0.8, 0.35]
    assert len(pooled_values) >= constants.MIN_POOLED_SR_VARIANCE_SAMPLES
    for i, v in enumerate(pooled_values):
        trial_ledger.record_outcome(
            ledger, "pool_fam_%d" % i, "prior-run", "2026-09-01", "cut_dsr",
            dsr_value=0.1, sharpe_in_sample=v)
    entries = [
        {"family_id": "solo",
         "daily_returns_path": _artifact(tmp_path, "s", [0.01, -0.02, 0.03]),
         "sharpe_in_sample": 0.5},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["sr_variance"] == pytest.approx(
        trial_ledger.pooled_sr_variance(ledger))
    assert out["sr_variance_source"] == "pooled"
    assert out["sr_variance"] != pytest.approx(
        constants.DEFAULT_SR_VARIANCE_FLOOR)


def test_sr_variance_falls_back_to_this_runs_own_sample_when_pool_is_thin(
        tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "a",
         "daily_returns_path": _artifact(tmp_path, "a", [0.01, -0.02, 0.03]),
         "sharpe_in_sample": 0.4},
        {"family_id": "b",
         "daily_returns_path": _artifact(tmp_path, "b", [-0.01, 0.02, 0.01]),
         "sharpe_in_sample": 0.6},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["sr_variance"] == pytest.approx(np.var([0.4, 0.6], ddof=1))
    assert out["sr_variance_source"] == "this_run"


def test_sr_variance_falls_back_to_the_floor_when_pool_and_run_both_thin(
        tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "solo",
         "daily_returns_path": _artifact(tmp_path, "s", [0.01, -0.02, 0.03]),
         "sharpe_in_sample": 0.5},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["sr_variance"] == pytest.approx(constants.DEFAULT_SR_VARIANCE_FLOOR)
    assert out["sr_variance_source"] == "floor"


def test_sr_variance_floor_constant_is_always_reported(tmp_path):
    # The caller cannot read
    # this repo's constants.py, so it cannot independently verify a
    # floor-sourced sr_variance unless the script tells it what the floor
    # constant actually is. Reported on EVERY call, not just floor-sourced
    # ones, so the caller can sanity-bound it regardless of which tier fired.
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "solo",
         "daily_returns_path": _artifact(tmp_path, "s", [0.01, -0.02, 0.03]),
         "sharpe_in_sample": 0.5},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["sr_variance_floor_constant"] == pytest.approx(
        constants.DEFAULT_SR_VARIANCE_FLOOR)


def test_sr_variance_this_run_tier_ignores_entries_with_no_sharpe(tmp_path):
    # A candidate whose manifest entry omits sharpe_in_sample (e.g. an older
    # caller) must not silently become 0.0 in the this-run sample -- it is
    # excluded, same as pooled_sr_variance excludes unscored ledger records.
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "a",
         "daily_returns_path": _artifact(tmp_path, "a", [0.01, -0.02, 0.03]),
         "sharpe_in_sample": 0.4},
        {"family_id": "b",
         "daily_returns_path": _artifact(tmp_path, "b", [-0.01, 0.02, 0.01])},
        {"family_id": "c",
         "daily_returns_path": _artifact(tmp_path, "c", [0.02, 0.01, -0.01]),
         "sharpe_in_sample": 0.9},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["sr_variance"] == pytest.approx(np.var([0.4, 0.9], ddof=1))
    assert out["sr_variance_source"] == "this_run"


def test_degenerate_series_are_excluded_from_the_correlation(tmp_path):
    """A zero-trade candidate's returns are all zeros; num_independent_trials
    raises on those by design. They must not abort the barrier."""
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "real_a",
         "daily_returns_path": _artifact(tmp_path, "ra", [0.01, -0.02, 0.03, 0.01])},
        {"family_id": "real_b",
         "daily_returns_path": _artifact(tmp_path, "rb", [-0.01, 0.02, 0.01, -0.03])},
        {"family_id": "flat",
         "daily_returns_path": _artifact(tmp_path, "f", [0.0, 0.0, 0.0, 0.0])},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["degenerate_series_count"] == 1
    assert out["n_candidates"] == 3
    # The flat candidate still consumes a trial even though it carries no
    # correlation signal.
    assert out["ledger_floor"] == 3
    assert out["n_this_run"] > 0


def test_all_degenerate_falls_back_to_the_raw_candidate_count(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "f1",
         "daily_returns_path": _artifact(tmp_path, "f1", [0.0, 0.0, 0.0])},
        {"family_id": "f2",
         "daily_returns_path": _artifact(tmp_path, "f2", [0.0, 0.0, 0.0])},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["degenerate_series_count"] == 2
    assert out["n_this_run"] == 2.0
    assert out["n_trials"] == 2.0


def test_n_trials_is_never_below_one(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "solo",
         "daily_returns_path": _artifact(tmp_path, "s", [0.01, -0.02, 0.03])},
    ]
    out = csc.compute(_manifest(tmp_path, entries), ledger)
    assert out["n_trials"] >= 1.0


def test_missing_artifact_file_fails_closed(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "gone",
         "daily_returns_path": str(tmp_path / "does_not_exist.json")},
    ]
    with pytest.raises(csc.CorrectionError):
        csc.compute(_manifest(tmp_path, entries), ledger)


def test_empty_manifest_fails_closed(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    with pytest.raises(csc.CorrectionError):
        csc.compute(_manifest(tmp_path, []), ledger)


def test_manifest_entry_missing_family_id_fails_closed(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"daily_returns_path": _artifact(tmp_path, "x", [0.01, -0.02, 0.03])},
    ]
    with pytest.raises(csc.CorrectionError):
        csc.compute(_manifest(tmp_path, entries), ledger)


def test_compute_never_writes_to_the_ledger(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(
        ledger, "fam_a", "2026-01-01", "s", "e", [], "r0")
    before = open(ledger, "rb").read()
    entries = [
        {"family_id": "brand_new",
         "daily_returns_path": _artifact(tmp_path, "n", [0.01, -0.02, 0.03])},
    ]
    csc.compute(_manifest(tmp_path, entries), ledger)
    assert open(ledger, "rb").read() == before
    assert trial_ledger.has_been_tested(ledger, "brand_new") is False


def test_cli_prints_one_json_object_and_exits_zero(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "c1",
         "daily_returns_path": _artifact(tmp_path, "c1", [0.01, -0.02, 0.03])},
        {"family_id": "c2",
         "daily_returns_path": _artifact(tmp_path, "c2", [-0.01, 0.02, 0.01])},
    ]
    proc = subprocess.run(
        [sys.executable,
         os.path.join(_ROOT, "scripts", "compute_shared_correction.py"),
         "--manifest", _manifest(tmp_path, entries), "--ledger", ledger],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert payload["n_candidates"] == 2
    assert payload["family_ids"] == ["c1", "c2"]
    assert payload["n_trials"] >= 2.0


def test_cli_exits_nonzero_and_prints_no_result_on_failure(tmp_path):
    """Fail closed: a caller must never be able to read a usable n_trials out
    of a failed run's stdout."""
    ledger = str(tmp_path / "ledger.jsonl")
    entries = [
        {"family_id": "gone",
         "daily_returns_path": str(tmp_path / "nope.json")},
    ]
    proc = subprocess.run(
        [sys.executable,
         os.path.join(_ROOT, "scripts", "compute_shared_correction.py"),
         "--manifest", _manifest(tmp_path, entries), "--ledger", ledger],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "n_trials" not in proc.stdout
