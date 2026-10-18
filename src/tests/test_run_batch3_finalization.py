import json
import os
import sys

import pytest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from mining import trial_ledger

import run_batch3_mining as batch3


def _execute(tmp_path, runner, renderer, *, dry_run=False,
             near_miss_renderer=None):
    """`near_miss_renderer` defaults to an inert no-op, NOT
    `batch3.regenerate_near_miss_log` -- most callers here write garbage,
    non-JSONL ledger content (e.g. "baseline\\n") to exercise the
    finalization CONTRACT in isolation, and the real near-miss renderer
    would (correctly) raise a JSON error on that content. Tests that care
    about the real wiring pass an explicit renderer (see the
    near-miss-specific tests below)."""
    ledger = tmp_path / "trial-ledger.jsonl"
    ledger.write_text("baseline\n", encoding="utf-8")
    output = tmp_path / "batch3-results.json"
    results_log = tmp_path / "RESULTS-LOG.md"
    near_miss_log = tmp_path / "NEAR-MISS-LOG.md"
    nm_renderer = near_miss_renderer or (lambda **_kwargs: None)
    return (
        lambda: batch3.execute_batch_run(
            {}, tested_date="2026-08-29", run_id="test-run",
            ledger_path=str(ledger), dry_run=dry_run,
            output_path=str(output), catalog_path="catalog.json",
            research_root="research", results_log_path=str(results_log),
            batch_runner=runner, results_renderer=renderer,
            near_miss_log_path=str(near_miss_log),
            near_miss_renderer=nm_renderer),
        ledger, output, results_log,
    )


def test_mutating_failure_regenerates_results_log_in_finally(tmp_path):
    calls = []

    def runner(_scope, **kwargs):
        with open(kwargs["ledger_path"], "a", encoding="utf-8") as handle:
            handle.write("pinned-and-defensively-closed\n")
        raise RuntimeError("forced post-pin failure")

    def renderer(**kwargs):
        calls.append(kwargs)
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated\n")

    execute, ledger, output, results_log = _execute(tmp_path, runner, renderer)

    with pytest.raises(RuntimeError, match="forced post-pin failure"):
        execute()

    assert ledger.read_text(encoding="utf-8").endswith(
        "pinned-and-defensively-closed\n")
    assert not output.exists()
    assert results_log.read_text(encoding="utf-8") == "regenerated\n"
    assert calls == [{
        "catalog_path": "catalog.json",
        "ledger_path": str(ledger),
        "research_root": "research",
        "out_path": str(results_log),
    }]


def test_mutating_failure_preserves_runner_and_regeneration_errors(tmp_path):
    def runner(_scope, **kwargs):
        with open(kwargs["ledger_path"], "a", encoding="utf-8") as handle:
            handle.write("pinned-and-defensively-closed\n")
        raise RuntimeError("forced post-pin failure")

    def renderer(**_kwargs):
        raise OSError("forced results-log failure")

    execute, _ledger, output, results_log = _execute(
        tmp_path, runner, renderer)

    with pytest.raises(ExceptionGroup) as raised:
        execute()

    assert raised.value.message == (
        "batch run and results-log regeneration both failed")
    assert [type(error) for error in raised.value.exceptions] == [
        RuntimeError, OSError]
    assert [str(error) for error in raised.value.exceptions] == [
        "forced post-pin failure", "forced results-log failure"]
    assert not output.exists()
    assert not results_log.exists()


def test_failure_before_ledger_mutation_does_not_regenerate(tmp_path):
    def runner(_scope, **_kwargs):
        raise RuntimeError("pre-mutation failure")

    def renderer(**_kwargs):
        raise AssertionError("renderer must not run without a ledger change")

    execute, _ledger, output, results_log = _execute(
        tmp_path, runner, renderer)

    with pytest.raises(RuntimeError, match="pre-mutation failure"):
        execute()

    assert not output.exists()
    assert not results_log.exists()


def test_success_writes_artifact_and_regenerates_after_mutation(tmp_path):
    calls = []
    payload = {"schema_version": 1, "results": [], "reconciliation": []}

    def runner(_scope, **kwargs):
        with open(kwargs["ledger_path"], "a", encoding="utf-8") as handle:
            handle.write("successful-events\n")
        return payload

    def renderer(**kwargs):
        calls.append(kwargs)
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated\n")

    execute, _ledger, output, results_log = _execute(
        tmp_path, runner, renderer)

    assert execute() == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert results_log.read_text(encoding="utf-8") == "regenerated\n"
    assert len(calls) == 1


def test_dry_run_never_regenerates_even_if_injected_runner_mutates(tmp_path):
    payload = {"schema_version": 1, "results": [], "reconciliation": []}

    def runner(_scope, **kwargs):
        with open(kwargs["ledger_path"], "a", encoding="utf-8") as handle:
            handle.write("buggy-dry-mutation\n")
        return payload

    def renderer(**_kwargs):
        raise AssertionError("dry mode must not regenerate durable evidence")

    execute, _ledger, output, results_log = _execute(
        tmp_path, runner, renderer, dry_run=True)

    assert execute() == payload
    assert output.exists()
    assert not results_log.exists()


def test_atomic_write_preserves_destination_when_replace_fails(
        tmp_path, monkeypatch):
    destination = tmp_path / "RESULTS-LOG.md"
    destination.write_text("previous evidence\n", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise PermissionError("replace blocked")

    monkeypatch.setattr(batch3.os, "replace", fail_replace)

    with pytest.raises(PermissionError, match="replace blocked"):
        batch3._atomic_write_text(str(destination), "new evidence\n")

    assert destination.read_text(encoding="utf-8") == "previous evidence\n"
    assert list(tmp_path.glob(".RESULTS-LOG.md.*.tmp")) == []


def test_near_miss_log_is_regenerated_alongside_results_log_on_success(
        tmp_path):
    """Guards against a stale near-miss log: after a real (non-dry-run) run
    mutates the ledger, NEAR-MISS-LOG.md must refresh at the SAME
    finalization point that refreshes RESULTS-LOG.md."""
    payload = {"schema_version": 1, "results": [], "reconciliation": []}
    near_miss_calls = []

    def runner(_scope, **kwargs):
        with open(kwargs["ledger_path"], "a", encoding="utf-8") as handle:
            handle.write("successful-events\n")
        return payload

    def renderer(**kwargs):
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated results log\n")

    def nm_renderer(**kwargs):
        near_miss_calls.append(kwargs)
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated near-miss log\n")

    execute, ledger, _output, results_log = _execute(
        tmp_path, runner, renderer, near_miss_renderer=nm_renderer)
    near_miss_log = tmp_path / "NEAR-MISS-LOG.md"

    execute()

    assert results_log.read_text(encoding="utf-8") == "regenerated results log\n"
    assert near_miss_log.read_text(encoding="utf-8") == "regenerated near-miss log\n"
    assert near_miss_calls == [{
        "ledger_path": str(ledger),
        "out_path": str(near_miss_log),
    }]


def test_near_miss_log_is_regenerated_even_when_run_raises(tmp_path):
    """Mirrors test_mutating_failure_regenerates_results_log_in_finally --
    a defensively-outcomed pin must refresh NEAR-MISS-LOG.md too, not just
    RESULTS-LOG.md, before the exception propagates."""
    near_miss_calls = []

    def runner(_scope, **kwargs):
        with open(kwargs["ledger_path"], "a", encoding="utf-8") as handle:
            handle.write("pinned-and-defensively-closed\n")
        raise RuntimeError("forced post-pin failure")

    def renderer(**kwargs):
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated\n")

    def nm_renderer(**kwargs):
        near_miss_calls.append(kwargs)
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated near-miss log\n")

    execute, _ledger, _output, _results_log = _execute(
        tmp_path, runner, renderer, near_miss_renderer=nm_renderer)
    near_miss_log = tmp_path / "NEAR-MISS-LOG.md"

    with pytest.raises(RuntimeError, match="forced post-pin failure"):
        execute()

    assert near_miss_log.read_text(encoding="utf-8") == "regenerated near-miss log\n"
    assert len(near_miss_calls) == 1


def test_near_miss_log_not_regenerated_without_a_ledger_mutation(tmp_path):
    """Mirrors test_dry_run_never_regenerates_even_if_injected_runner_mutates
    for the near-miss renderer specifically."""
    payload = {"schema_version": 1, "results": [], "reconciliation": []}

    def runner(_scope, **kwargs):
        with open(kwargs["ledger_path"], "a", encoding="utf-8") as handle:
            handle.write("buggy-dry-mutation\n")
        return payload

    def renderer(**_kwargs):
        raise AssertionError("dry mode must not regenerate durable evidence")

    def nm_renderer(**_kwargs):
        raise AssertionError(
            "dry mode must not regenerate the near-miss log either")

    execute, _ledger, _output, results_log = _execute(
        tmp_path, runner, renderer, dry_run=True, near_miss_renderer=nm_renderer)
    near_miss_log = tmp_path / "NEAR-MISS-LOG.md"

    assert execute() == payload
    assert not results_log.exists()
    assert not near_miss_log.exists()


def test_default_near_miss_renderer_is_wired_when_not_overridden(tmp_path):
    """The case this whole set guards: a caller that passes no
    `near_miss_renderer` (as the production call sites do) must STILL
    get a refreshed NEAR-MISS-LOG.md via the
    real default `regenerate_near_miss_log` -- not just when a test
    explicitly injects a near_miss_renderer spy. Uses a REAL, valid ledger
    mutation (via trial_ledger.record_outcome, not garbage placeholder
    text) so the real renderer can actually run against it."""
    payload = {"schema_version": 1, "results": [], "reconciliation": []}

    def runner(_scope, **kwargs):
        trial_ledger.record_outcome(
            kwargs["ledger_path"], "fam_real", "test-run", "2026-08-29",
            "cut_dsr", dsr_value=0.5)
        return payload

    def renderer(**kwargs):
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated results log\n")

    ledger = tmp_path / "trial-ledger.jsonl"
    output = tmp_path / "batch3-results.json"
    results_log = tmp_path / "RESULTS-LOG.md"
    near_miss_log = tmp_path / "NEAR-MISS-LOG.md"

    result = batch3.execute_batch_run(
        {}, tested_date="2026-08-29", run_id="test-run",
        ledger_path=str(ledger), dry_run=False,
        output_path=str(output), catalog_path="catalog.json",
        research_root="research", results_log_path=str(results_log),
        near_miss_log_path=str(near_miss_log),
        batch_runner=runner, results_renderer=renderer)

    assert result == payload
    assert near_miss_log.exists()
    text = near_miss_log.read_text(encoding="utf-8")
    assert "near-miss" in text.lower()


def test_near_miss_log_path_defaults_next_to_results_log_when_omitted(
        tmp_path):
    """If a caller doesn't even pass near_miss_log_path, it must default
    to a sibling of results_log_path, not silently do nothing."""
    payload = {"schema_version": 1, "results": [], "reconciliation": []}

    def runner(_scope, **kwargs):
        trial_ledger.record_outcome(
            kwargs["ledger_path"], "fam_real", "test-run", "2026-08-29",
            "cut_dsr", dsr_value=0.5)
        return payload

    def renderer(**kwargs):
        with open(kwargs["out_path"], "w", encoding="utf-8") as handle:
            handle.write("regenerated results log\n")

    ledger = tmp_path / "trial-ledger.jsonl"
    output = tmp_path / "batch3-results.json"
    results_log = tmp_path / "RESULTS-LOG.md"

    batch3.execute_batch_run(
        {}, tested_date="2026-08-29", run_id="test-run",
        ledger_path=str(ledger), dry_run=False,
        output_path=str(output), catalog_path="catalog.json",
        research_root="research", results_log_path=str(results_log),
        batch_runner=runner, results_renderer=renderer)

    assert (tmp_path / "NEAR-MISS-LOG.md").exists()


def test_atomic_write_cleanup_failure_does_not_mask_primary_error(
        tmp_path, monkeypatch):
    destination = tmp_path / "RESULTS-LOG.md"

    def fail_replace(_source, _destination):
        raise PermissionError("replace blocked")

    def fail_cleanup(_path):
        raise OSError("cleanup blocked")

    monkeypatch.setattr(batch3.os, "replace", fail_replace)
    monkeypatch.setattr(batch3.os, "remove", fail_cleanup)

    with pytest.raises(PermissionError, match="replace blocked"):
        batch3._atomic_write_text(str(destination), "new evidence\n")
