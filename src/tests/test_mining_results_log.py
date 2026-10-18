import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import evidence, results_log, trial_ledger


SAMPLE_REPORT = """# Mining run — 2026-01-15

- Catalog entries considered: 6
- Excluded as already ruled-out (catalog status / past gate check): 2 — fam_a, fam_b
- Excluded for missing data: 3 — fam_c: no OHLC data available, fam_d: no cost data available -- skipped rather than guessed, since no cost source was found, : no OHLC data available
- Not attempted (budget/time ceiling reached): 1 — fam_e
- Passed in-sample deflated-Sharpe bar: 0
- Passed holdout confirmation (= survivors): 0

**Survivors: zero — none of the 6 candidates tested this run cleared both the in-sample and holdout bars**

Budget used: not budget-limited of n/a
Distinct signal families tested this run: 0
"""


def test_parse_run_report_extracts_all_three_lists_and_date():
    parsed = results_log.parse_run_report(SAMPLE_REPORT)
    assert parsed["run_date"] == "2026-01-15"
    assert parsed["n_considered"] == 6
    assert parsed["ruled_out"] == ["fam_a", "fam_b"]
    assert parsed["not_attempted"] == ["fam_e"]


def test_parse_run_report_missing_data_survives_comma_inside_reason():
    parsed = results_log.parse_run_report(SAMPLE_REPORT)
    # 3 items even though fam_d's own reason text contains a comma --
    # a naive split(", ") would have produced 4.
    assert len(parsed["missing_data"]) == 3
    by_fam = {m["family_id"]: m["reason"] for m in parsed["missing_data"]}
    assert by_fam["fam_c"] == "no OHLC data available"
    assert "skipped rather than guessed, since no cost source" in by_fam["fam_d"]
    # the blank-family_id item is preserved as-is, not dropped
    assert "" in by_fam


def test_parse_run_report_none_placeholders_become_empty_lists():
    text = """# Mining run — 2026-01-14

- Catalog entries considered: 2
- Excluded as already ruled-out (catalog status / past gate check): 2 — fam_a, fam_b
- Excluded for missing data: 0 — none
- Not attempted (budget/time ceiling reached): 0 — none
- Passed in-sample deflated-Sharpe bar: 0
- Passed holdout confirmation (= survivors): 0

**Survivors: zero — none of the 2 candidates tested this run cleared both the in-sample and holdout bars**

Budget used: 0 of n/a (nothing dispatched)
Distinct signal families tested this run: 0
"""
    parsed = results_log.parse_run_report(text)
    assert parsed["missing_data"] == []
    assert parsed["not_attempted"] == []


def test_load_all_run_reports_reads_and_sorts_by_date(tmp_path):
    (tmp_path / "2026-01-14").mkdir()
    (tmp_path / "2026-01-15").mkdir()
    (tmp_path / "2026-01-15" / "run-report.md").write_text(SAMPLE_REPORT, encoding="utf-8")
    (tmp_path / "2026-01-14" / "run-report.md").write_text(
        SAMPLE_REPORT.replace("2026-01-15", "2026-01-14"), encoding="utf-8")

    reports = results_log.load_all_run_reports(str(tmp_path))
    assert [r["run_date"] for r in reports] == ["2026-01-14", "2026-01-15"]
    assert reports[0]["path"].endswith("run-report.md")


def _catalog_entry(family_id, status="eligible", tickers=None, mechanism="does a thing"):
    return {
        "family_id": family_id, "name": family_id, "mechanism": mechanism,
        "tickers": tickers or ["XYZ"], "parameters": {}, "account_types": ["taxable"],
        "status": status, "source": "test", "added_date": "2026-08-29",
        **({"ruled_out_reason": "because"} if status == "ruled_out" else {}),
    }


def test_resolve_missing_data_attribution_by_elimination_when_counts_match(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    # catalog scope (first 3 entries): fam_ruled (ruled_out), fam_tested
    # (eligible, tested), fam_mystery (eligible, never mentioned anywhere
    # else in this report) -- exactly one leftover for one blank.
    catalog = [
        _catalog_entry("fam_ruled", status="ruled_out"),
        _catalog_entry("fam_tested"),
        _catalog_entry("fam_mystery"),
    ]
    trial_ledger.record_first_test(ledger_path, "fam_tested", "2026-08-29", "s", "e", [], "run-1")

    report = {
        "run_date": "2026-08-29", "n_considered": 3,
        "ruled_out": ["fam_ruled"],
        "missing_data": [{"family_id": "", "reason": "no OHLC data available"}],
        "not_attempted": [],
        "path": "fake/run-report.md",
    }
    resolved = results_log.resolve_missing_data_attribution([report], catalog, ledger_path)
    fam_ids = {m["family_id"] for m in resolved[0]["missing_data"]}
    assert fam_ids == {"fam_mystery"}
    reason = resolved[0]["missing_data"][0]["reason"]
    assert "inferred by elimination" in reason


def test_resolve_missing_data_attribution_leaves_unattributed_on_count_mismatch(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    # Two eligible leftovers, only one blank -- counts don't match, so
    # neither should be silently guessed.
    catalog = [
        _catalog_entry("fam_x"),
        _catalog_entry("fam_y"),
    ]
    report = {
        "run_date": "2026-08-29", "n_considered": 2,
        "ruled_out": [], "not_attempted": [],
        "missing_data": [{"family_id": "", "reason": "no OHLC data available"}],
        "path": "fake/run-report.md",
    }
    resolved = results_log.resolve_missing_data_attribution([report], catalog, ledger_path)
    fam_ids = [m["family_id"] for m in resolved[0]["missing_data"]]
    assert len(fam_ids) == 1
    assert "not attributed" in fam_ids[0]
    assert "fam_x" not in fam_ids[0] and "fam_y" not in fam_ids[0]


def test_build_rows_uses_latest_outcome_and_joins_multiple_test_dates(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_catalog_entry("fam_a", tickers=["ABC"])]
    trial_ledger.record_first_test(ledger_path, "fam_a", "2026-08-28", "s", "e", [], "run-1")
    trial_ledger.record_outcome(ledger_path, "fam_a", "run-1", "2026-08-28", "cut_dsr", dsr_value=0.5)
    trial_ledger.record_outcome(ledger_path, "fam_a", "run-2", "2026-08-29", "survived", dsr_value=0.98)

    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    assert len(rows) == 1
    row = rows[0]
    assert row["family_id"] == "fam_a"
    assert row["ticker"] == "ABC"
    # both the first_test date and every later outcome's date are
    # surfaced -- this synthetic family carries two test dates
    assert row["dates_tested"] == ["2026-08-28", "2026-08-29"]
    assert row["dsr_value"] == 0.98
    assert row["outcome"] == "survived"


def test_build_rows_surfaces_pooled_ticker_label_and_note_from_latest_outcome(tmp_path):
    # Pooled multi-ticker test: when an outcome event carries
    # ticker_label/note (a pooled multi-ticker test), the row must show
    # those instead of the catalog's single placeholder ticker, and the
    # note must be populated from the latest outcome.
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_catalog_entry("fam_a", tickers=["ABC"])]
    trial_ledger.record_first_test(ledger_path, "fam_a", "2026-01-15", "s", "e", [], "run-1")
    trial_ledger.record_outcome(ledger_path, "fam_a", "run-1", "2026-01-15", "cut_dsr", dsr_value=0.25)
    trial_ledger.record_outcome(
        ledger_path, "fam_a", "run-2", "2026-01-15", "survived", dsr_value=0.9,
        note="Pooled trade-level test across 50 synthetic tickers.",
        ticker_label="50 tickers (synthetic pool)")

    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    row = rows[0]
    assert row["ticker"] == "50 tickers (synthetic pool)"
    assert "Pooled trade-level test across 50 synthetic tickers" in row["note"]
    assert row["dsr_value"] == 0.9
    assert row["outcome"] == "survived"


def test_build_rows_tested_row_without_ticker_label_keeps_catalog_ticker_and_empty_note(tmp_path):
    # An ordinary single-ticker outcome (no note/ticker_label) keeps the
    # catalog's ticker and renders an empty note.
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_catalog_entry("fam_a", tickers=["ABC"])]
    trial_ledger.record_first_test(ledger_path, "fam_a", "2026-08-28", "s", "e", [], "run-1")
    trial_ledger.record_outcome(ledger_path, "fam_a", "run-1", "2026-08-28", "cut_dsr", dsr_value=0.5)

    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    row = rows[0]
    assert row["ticker"] == "ABC"
    assert row["note"] == ""


def test_build_rows_marks_catalog_ruled_out_entries_with_no_ledger_data(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_catalog_entry("fam_r", status="ruled_out")]
    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    assert rows[0]["outcome"] == "ruled out before catalog curation"
    assert rows[0]["note"] == "because"
    assert rows[0]["dates_tested"] == []


def test_build_rows_uses_explicit_catalog_timeframe(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_catalog_entry("fam_a", tickers=["ABC"])]
    catalog[0]["timeframe"] = "swing"
    trial_ledger.record_first_test(ledger_path, "fam_a", "2026-08-28", "s", "e", [], "run-1")
    trial_ledger.record_outcome(ledger_path, "fam_a", "run-1", "2026-08-28", "cut_dsr", dsr_value=0.5)

    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    assert rows[0]["timeframe"] == "swing"

    text = results_log.render_markdown(rows, sources=[])
    row_line = next(line for line in text.splitlines() if line.startswith("| fam_a |"))
    cols = [c.strip() for c in row_line.strip("|").split("|")]
    assert cols[2] == "swing"


def test_build_rows_defaults_timeframe_to_daily_when_absent(tmp_path):
    # A catalog entry with no "timeframe" key defaults to "daily".
    # Covers a tested row, a ruled-out row, and the eligible-
    # not-yet-tested fallback row -- three of the five _row_for branches.
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [
        _catalog_entry("fam_tested", tickers=["ABC"]),
        _catalog_entry("fam_ruled", status="ruled_out"),
        _catalog_entry("fam_untested"),
    ]
    trial_ledger.record_first_test(ledger_path, "fam_tested", "2026-08-28", "s", "e", [], "run-1")
    trial_ledger.record_outcome(ledger_path, "fam_tested", "run-1", "2026-08-28", "cut_dsr", dsr_value=0.5)

    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    by_fam = {r["family_id"]: r for r in rows}
    assert by_fam["fam_tested"]["timeframe"] == "daily"
    assert by_fam["fam_ruled"]["timeframe"] == "daily"
    assert by_fam["fam_untested"]["timeframe"] == "daily"

    text = results_log.render_markdown(rows, sources=[])
    assert "| daily |" in text


def test_build_rows_straggler_with_no_catalog_entry_gets_na_timeframe(tmp_path):
    # entry is None for this family_id (the catalog doesn't know about
    # it) -- must fall back to "n/a", matching name's own "(not in
    # catalog)" convention for the same defensive case.
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = []
    trial_ledger.record_first_test(ledger_path, "fam_ghost", "2026-08-28", "s", "e", [], "run-1")

    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    assert len(rows) == 1
    assert rows[0]["family_id"] == "fam_ghost"
    assert rows[0]["timeframe"] == "n/a"

    text = results_log.render_markdown(rows, sources=[])
    row_line = next(line for line in text.splitlines() if line.startswith("| fam_ghost |"))
    cols = [c.strip() for c in row_line.strip("|").split("|")]
    assert cols[2] == "n/a"


def test_build_rows_marks_missing_data_entries_from_reports(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_catalog_entry("fam_m")]
    report = {"run_date": "2026-08-29", "missing_data":
              [{"family_id": "fam_m", "reason": "no OHLC data available"}],
              "not_attempted": []}
    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[report])
    assert rows[0]["outcome"] == "no price data available"
    assert rows[0]["note"] == "no OHLC data available"
    assert rows[0]["dates_tested"] == ["2026-08-29"]


def test_render_markdown_includes_every_row_and_sources():
    rows = [{"family_id": "fam_a", "ticker": "ABC", "timeframe": "daily",
             "dates_tested": ["2026-08-29"],
             "dsr_value": 0.9814, "outcome": "survived", "mechanism": "does a thing",
             "note": ""}]
    text = results_log.render_markdown(rows, sources=["data/mining/trial_ledger.jsonl"])
    assert "fam_a" in text
    assert "0.9814" in text
    assert "survived" in text
    assert "trial_ledger.jsonl" in text
    assert "never hand-edit" in text


def test_render_markdown_header_has_timeframe_column():
    text = results_log.render_markdown([], sources=[])
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("| family_id |"))
    separator = lines[lines.index(header) + 1]
    assert "| timeframe |" in header
    header_cols = [c.strip() for c in header.strip("|").split("|")]
    sep_cols = [c.strip() for c in separator.strip("|").split("|")]
    assert header_cols == ["family_id", "ticker", "timeframe", "date(s) tested",
                            "DSR value", "outcome", "mechanism", "note"]
    assert len(sep_cols) == len(header_cols)


SIMPLE_REPORT = """# Mining run — 2026-01-15

- Catalog entries considered: 2
- Excluded as already ruled-out (catalog status / past gate check): 1 — fam_a
- Excluded for missing data: 1 — fam_c: no OHLC data available
- Not attempted (budget/time ceiling reached): 0 — none
- Passed in-sample deflated-Sharpe bar: 0
- Passed holdout confirmation (= survivors): 0

**Survivors: zero — none of the 2 candidates tested this run cleared both the in-sample and holdout bars**

Budget used: not budget-limited of n/a
Distinct signal families tested this run: 0
"""


def test_render_results_log_end_to_end_smoke(tmp_path):
    research_root = tmp_path / "runs"
    (research_root / "2026-01-15").mkdir(parents=True)
    (research_root / "2026-01-15" / "run-report.md").write_text(SIMPLE_REPORT, encoding="utf-8")
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [
        _catalog_entry("fam_a", status="ruled_out"),
        _catalog_entry("fam_c"),
    ]
    text = results_log.render_results_log(catalog, ledger_path, str(research_root))
    assert "fam_a" in text
    assert "fam_c" in text
    assert "no price data available" in text


# ---------------------------------------------------------------------------
# Evidence lifecycle: OUTCOME_LABELS covers the two
# UNSCORED stages, and the DSR column must render a deferral distinguishably
# from a stage-less null -- otherwise "never scored" and "scored, dsr_value
# happened to be absent" both print as the same "n/a".
# ---------------------------------------------------------------------------

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


def test_outcome_labels_cover_both_unscored_stages():
    assert "insufficient_evidence" in results_log.OUTCOME_LABELS
    assert "not_evaluated_error" in results_log.OUTCOME_LABELS
    # Distinct from each other and from every SCORED label -- a reader must
    # be able to tell a deferral from an engine defect from a real cut.
    labels = results_log.OUTCOME_LABELS
    assert len({labels["insufficient_evidence"], labels["not_evaluated_error"],
                labels["cut_dsr"], labels["survived"]}) == 4


def test_build_rows_labels_insufficient_evidence_and_carries_its_stage(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    catalog = [_catalog_entry("fam_a", tickers=["ABC"])]
    trial_ledger.record_first_test(ledger_path, "fam_a", "2026-09-03", "s", "e", [], "run-1")
    trial_ledger.record_outcome(
        ledger_path, "fam_a", "run-1", "2026-09-03", "insufficient_evidence",
        dsr_value=None, evidence=_insufficient_evidence_payload("fam_a"))

    rows = results_log.build_rows(catalog, ledger_path, resolved_reports=[])
    row = rows[0]
    assert row["dsr_value"] is None
    assert row["outcome"] == results_log.OUTCOME_LABELS["insufficient_evidence"]
    assert row.get("stage") == "insufficient_evidence"


def test_render_markdown_distinguishes_deferred_from_legacy_null_dsr():
    # A stage-less null (no stage recorded at all -- the missing_data/
    # eligible/ruled-out row shapes, which carry no stage) renders "n/a".
    # A deferral WITH a stage renders something else, so the two are never
    # visually identical in the regenerated table.
    deferred_row = {"family_id": "fam_a", "ticker": "ABC", "timeframe": "daily",
                    "dates_tested": ["2026-09-03"], "dsr_value": None,
                    "outcome": results_log.OUTCOME_LABELS["insufficient_evidence"],
                    "mechanism": "x", "note": "", "stage": "insufficient_evidence"}
    legacy_row = {"family_id": "fam_b", "ticker": "XYZ", "timeframe": "daily",
                  "dates_tested": [], "dsr_value": None,
                  "outcome": "no price data available", "mechanism": "x", "note": ""}
    text = results_log.render_markdown([deferred_row, legacy_row], sources=[])
    deferred_line = next(l for l in text.splitlines() if l.startswith("| fam_a |"))
    legacy_line = next(l for l in text.splitlines() if l.startswith("| fam_b |"))
    deferred_dsr = [c.strip() for c in deferred_line.strip("|").split("|")][4]
    legacy_dsr = [c.strip() for c in legacy_line.strip("|").split("|")][4]
    assert deferred_dsr != legacy_dsr
    assert legacy_dsr == "n/a"


def test_render_markdown_labels_not_evaluated_error_distinctly_too():
    error_row = {"family_id": "fam_c", "ticker": "ABC", "timeframe": "daily",
                 "dates_tested": ["2026-09-03"], "dsr_value": None,
                 "outcome": results_log.OUTCOME_LABELS["not_evaluated_error"],
                 "mechanism": "x", "note": "", "stage": "not_evaluated_error"}
    text = results_log.render_markdown([error_row], sources=[])
    line = next(l for l in text.splitlines() if l.startswith("| fam_c |"))
    dsr_col = [c.strip() for c in line.strip("|").split("|")][4]
    assert dsr_col != "n/a"
