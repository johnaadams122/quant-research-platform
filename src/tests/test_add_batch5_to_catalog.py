"""Tests for scripts/add_batch5_to_catalog.py on tmp copies only."""
import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import add_batch5_to_catalog as adder
from mining import catalog as catalog_mod


def _existing_entry(family_id="old_family", parameters=None):
    return {"family_id": family_id, "name": "Old", "mechanism": "old mech",
            "tickers": ["SPY"], "parameters": parameters or {"k": 1},
            "account_types": ["roth"], "status": "eligible", "source": "s",
            "added_date": "2026-09-01"}


def _accepted_candidate(family_id="b5_new", parameters=None, **overrides):
    candidate = {
        "family_id": family_id, "name": "New family",
        "mechanism": "new mechanism", "shape": "pooled_single_asset",
        "tickers": ["AAA", "BBB"], "signal_tickers": ["SPY"],
        "benchmark_ticker": "SPY", "parameters": parameters or {"period": 15},
        "account_types": ["taxable", "roth"], "source": "curation lens",
        "distinctness_note": "distinct", "directional_rationale": "why",
        "expected_in_sample_trades": 40, "rule_pseudocode": "if x then buy",
        "liquidity_basis": "adv", "resolved_pool": ["AAA", "BBB"],
        "ticker_history": "n/a", "added_date": "2026-09-04",
        "status": "eligible", "timeframe": "daily",
    }
    candidate.update(overrides)
    return candidate


def _write(tmp_path, catalog_entries, candidates):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_entries, indent=1), encoding="utf-8")
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps({"batch": "2026-09-04-batch5",
                                         "accepted": candidates}), encoding="utf-8")
    return str(accepted_path), str(catalog_path)


def test_appends_required_and_optional_fields_and_is_idempotent(tmp_path):
    accepted, catalog_path = _write(tmp_path, [_existing_entry()],
                                    [_accepted_candidate()])
    first = adder.add_batch5(accepted, catalog_path, dry_run=False)
    assert first["appended"] == ["b5_new"] and first["written"] is True
    assert (first["before"], first["after"]) == (1, 2)
    entries = catalog_mod.load_catalog(catalog_path)
    assert [e["family_id"] for e in entries] == ["old_family", "b5_new"]
    new = entries[1]
    for field in catalog_mod.REQUIRED_FIELDS:
        assert field in new
    assert new["status"] == "eligible"
    assert new["timeframe"] == "daily"
    assert new["distinctness_note"] == "distinct"
    assert new["pooling_note"].startswith("resolved_pool:") and "AAA" in new["pooling_note"]
    for key in ("shape", "benchmark_ticker", "signal_tickers",
                "directional_rationale", "rule_pseudocode"):
        assert new[key] == _accepted_candidate()[key]
    # Keys that are curation-only stay out of the catalog row.
    assert "liquidity_basis" not in new and "ticker_history" not in new
    before = open(catalog_path, "rb").read()
    second = adder.add_batch5(accepted, catalog_path, dry_run=False)
    assert second["appended"] == [] and second["already_present"] == 1
    assert second["written"] is False
    assert open(catalog_path, "rb").read() == before


def test_existing_family_with_different_parameters_raises_and_writes_nothing(tmp_path):
    accepted, catalog_path = _write(
        tmp_path, [_existing_entry("b5_new", {"period": 99})],
        [_accepted_candidate("b5_new", {"period": 15})])
    before = open(catalog_path, "rb").read()
    with pytest.raises(ValueError, match="DIFFERENT parameters"):
        adder.add_batch5(accepted, catalog_path, dry_run=False)
    assert open(catalog_path, "rb").read() == before


def test_existing_family_with_identical_parameters_is_left_untouched(tmp_path):
    existing = _existing_entry("b5_new", {"period": 15})
    accepted, catalog_path = _write(tmp_path, [existing],
                                    [_accepted_candidate("b5_new", {"period": 15})])
    summary = adder.add_batch5(accepted, catalog_path, dry_run=False)
    assert summary["appended"] == []
    assert catalog_mod.load_catalog(catalog_path) == [existing]


def test_dry_run_reports_the_diff_count_and_writes_nothing(tmp_path, capsys):
    accepted, catalog_path = _write(
        tmp_path, [_existing_entry()],
        [_accepted_candidate("b5_one"), _accepted_candidate("b5_two")])
    before = open(catalog_path, "rb").read()
    assert adder.main(["--accepted", accepted, "--catalog", catalog_path,
                       "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["appended"] == ["b5_one", "b5_two"] and printed["dry_run"] is True
    assert printed["written"] is False
    assert open(catalog_path, "rb").read() == before


def test_candidate_missing_a_required_field_fails_before_any_write(tmp_path):
    bad = _accepted_candidate("b5_bad")
    del bad["mechanism"]
    accepted, catalog_path = _write(tmp_path, [_existing_entry()],
                                    [_accepted_candidate("b5_ok"), bad])
    before = open(catalog_path, "rb").read()
    with pytest.raises(ValueError, match="mechanism"):
        adder.add_batch5(accepted, catalog_path, dry_run=False)
    assert open(catalog_path, "rb").read() == before


def test_invalid_status_is_rejected_by_the_catalog_loader_contract(tmp_path):
    accepted, catalog_path = _write(
        tmp_path, [_existing_entry()],
        [_accepted_candidate("b5_bad", status="pending")])
    with pytest.raises(ValueError, match="invalid status"):
        adder.add_batch5(accepted, catalog_path, dry_run=False)


def test_docstring_carries_the_catalog_pin_note_for_the_main_loop():
    assert "test_mining_catalog.py" in adder.__doc__
    assert "pins the catalog size" in adder.__doc__
