import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import evidence
from mining import ruled_out
from mining import trial_ledger


def _entry(family_id, status="eligible", **overrides):
    base = {"family_id": family_id, "name": "x", "mechanism": "x",
            "tickers": ["QQQ"], "parameters": {}, "account_types": ["roth"],
            "status": status, "source": "x", "added_date": "2026-08-27"}
    base.update(overrides)
    return base


def test_filter_eligible_excludes_non_eligible_status(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_entry("a", status="ruled_out", ruled_out_reason="already tested")]
    eligible, excluded = ruled_out.filter_eligible(catalog, ledger_path)
    assert eligible == []
    assert excluded[0]["family_id"] == "a"
    assert "already tested" in excluded[0]["reason"]


def test_filter_eligible_excludes_survived_families(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(ledger_path, "b", "run-1", "2026-08-27", "survived", dsr_value=0.97)
    catalog = [_entry("b", status="eligible")]
    eligible, excluded = ruled_out.filter_eligible(catalog, ledger_path)
    assert eligible == []
    assert "survived" in excluded[0]["reason"]


def test_filter_eligible_allows_a_prior_near_miss_to_retry(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(ledger_path, "c", "run-1", "2026-08-27", "cut_dsr", dsr_value=0.5)
    catalog = [_entry("c", status="eligible")]
    eligible, excluded = ruled_out.filter_eligible(catalog, ledger_path)
    assert [e["family_id"] for e in eligible] == ["c"]
    assert excluded == []


def test_filter_eligible_allows_fresh_never_tested_family(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_entry("d", status="eligible")]
    eligible, excluded = ruled_out.filter_eligible(catalog, ledger_path)
    assert [e["family_id"] for e in eligible] == ["d"]


def _insufficient_evidence_payload(family_id):
    return evidence.build_insufficient_trades(
        stats={"trade_count": 5},
        catalog_entry={"family_id": family_id, "parameters": {}},
        instrument_set=["SPY"],
        in_sample_start="2020-01-01", in_sample_end="2024-06-01",
        pinned_window={"holdout_start": "2024-07-01", "holdout_end": "2024-12-31"},
        holdout_look="unspent",
        signal_source_path="src/mining/signals.py",
        signal_source_sha256="a" * 64,
        evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "test"})


def test_filter_eligible_excludes_insufficient_evidence_families(tmp_path):
    # insufficient_evidence is a THIRD exclusion category
    # alongside catalog status and a survived outcome -- an ordinary run
    # must not re-mine a family that was already recorded as never having
    # produced enough in-sample trades to judge (the retry path, if any, is
    # a separate, deliberate mechanism, not ordinary re-mining).
    ledger_path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger_path, "e", "run-1", "2026-09-03", "insufficient_evidence",
        dsr_value=None, evidence=_insufficient_evidence_payload("e"))
    catalog = [_entry("e", status="eligible")]
    eligible, excluded = ruled_out.filter_eligible(catalog, ledger_path)
    assert eligible == []
    assert "insufficient" in excluded[0]["reason"].lower()


def test_filter_eligible_excludes_not_evaluated_error_families(tmp_path):
    # Deliberately NOT excluded -- an engine error is a defect, not evidence
    # about the market, so the family must stay eligible and re-enter
    # ordinary runs once the defect is fixed (evidence.py's own
    # build_engine_error docstring). Pinned here as a regression guard: this
    # is the one UNSCORED stage that must NOT behave like the other one.
    ledger_path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger_path, "f", "run-1", "2026-09-03", "not_evaluated_error",
        dsr_value=None,
        evidence=evidence.build_engine_error(
            error=RuntimeError("boom"), instrument_set=["SPY"],
            pinned_window={"holdout_start": "2024-07-01", "holdout_end": "2024-12-31"},
            holdout_look="unspent"))
    catalog = [_entry("f", status="eligible")]
    eligible, excluded = ruled_out.filter_eligible(catalog, ledger_path)
    assert [e["family_id"] for e in eligible] == ["f"]
    assert excluded == []
