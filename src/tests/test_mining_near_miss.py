import math
import os
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import evidence, near_miss, trial_ledger


def _record_not_evaluated_error(path, family_id, run_id, tested_date):
    """A real UNSCORED not_evaluated_error outcome -- dsr_value=None via a
    legitimate stage. cut_dsr is a SCORED_STAGES member and record_outcome
    rejects it with dsr_value=None (insufficient trades has its own UNSCORED
    stage), so a null-dsr_value outcome for these tests must be built
    through a real UNSCORED stage."""
    payload = evidence.build_engine_error(
        error=Exception("simulated engine failure for this test"),
        instrument_set=["ABC"],
        pinned_window={"holdout_start": "2026-01-01", "holdout_end": "2026-06-30"},
        holdout_look="unspent")
    trial_ledger.record_outcome(
        path, family_id, run_id, tested_date,
        "not_evaluated_error", dsr_value=None, evidence=payload)


def _ledger(tmp_path) -> str:
    return str(tmp_path / "ledger.jsonl")


def _record_cost_gate_excluded(path, family_id, run_id, tested_date, *,
                               net_edge_bps: float, spread_bps: float = 1.0,
                               required: float = 3.0) -> None:
    """A real cost_realism_gate_excluded outcome, built through
    evidence.build_cost_realism_gate_excluded like a production caller. `required` overrides
    constants.INTRADAY_EDGE_OVER_COST_MULTIPLE at the constants module
    level so a test can exercise a specific edge_over_cost_ratio-vs-bar
    distance without depending on whatever the live constant happens to be
    (matches this test file's existing style of never hardcoding a
    production constant's current value into a test's own expectation).

    The override must stay in force through `trial_ledger.record_outcome`,
    not just through `build_cost_realism_gate_excluded`: `record_outcome`
    itself re-validates the payload via `evidence.validate(...,
    at_write_time=True)`, which checks the payload's embedded
    `edge_over_cost_multiple_required` against whatever
    `constants.INTRADAY_EDGE_OVER_COST_MULTIPLE` is AT THAT MOMENT.
    Restoring the constant before `record_outcome` would silently discard
    every `required` override by write time, and a call site passing
    `required=3.0` (equal to the live production default) would never
    notice."""
    import mining.constants as constants_mod
    original = constants_mod.INTRADAY_EDGE_OVER_COST_MULTIPLE
    constants_mod.INTRADAY_EDGE_OVER_COST_MULTIPLE = required
    try:
        ratio = net_edge_bps / spread_bps
        payload = evidence.build_cost_realism_gate_excluded(
            stats={"trade_count": 12, "net_edge_bps": net_edge_bps,
                   "sharpe_in_sample": 0.1},
            spread_bps=spread_bps, edge_over_cost_ratio=ratio,
            catalog_entry={"family_id": family_id, "parameters": {}},
            instrument_set=["SPY"],
            in_sample_start="2026-01-01", in_sample_end="2026-06-30",
            pinned_window={"holdout_start": "2026-07-01",
                           "holdout_end": "2026-08-31"},
            holdout_look="unspent", signal_source_path="src/mining/fake.py",
            signal_source_sha256="a" * 64, evidence_source="measured",
            reactivation={"kind": "none_available", "rationale": "test"})
        trial_ledger.record_outcome(
            path, family_id, run_id, tested_date, "insufficient_evidence",
            note="cost-realism gate rejection (test)", ticker_label="SPY",
            evidence=payload, construction="per_trade")
    finally:
        constants_mod.INTRADAY_EDGE_OVER_COST_MULTIPLE = original


def test_flags_a_cut_dsr_family_within_margin(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_first_test(
        path, "fam_close", "2026-08-01", "2026-01-01", "2026-06-30", [0.1], "run-1")
    trial_ledger.record_outcome(
        path, "fam_close", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.91)

    misses = near_miss.find_near_misses(path)

    assert len(misses) == 1
    row = misses[0]
    assert row["family_id"] == "fam_close"
    assert row["dsr_value"] == 0.91
    assert math.isclose(row["distance_from_bar"], 0.04, abs_tol=1e-9)
    assert row["first_tested_date"] == "2026-08-01"
    assert row["first_run_id"] == "run-1"


def test_excludes_dsr_value_exactly_at_threshold(tmp_path):
    path = _ledger(tmp_path)
    # A cut_dsr row whose dsr_value == threshold exactly (distance == 0)
    # represents a PASSING score under the pipeline's own >= convention
    # (constants.DSR_PASS_THRESHOLD's docstring: "A DSR >= this value is
    # required to reach the holdout"). record_outcome only validates the
    # STAGE, not that a cut_dsr row's dsr_value is actually below
    # threshold, so this combination could reach the ledger via an
    # upstream inconsistency -- it must never be reported as a near miss.
    trial_ledger.record_outcome(
        path, "fam_at_bar", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.95)

    misses = near_miss.find_near_misses(path, threshold=0.95)

    assert misses == []


def test_excludes_a_cut_dsr_family_beyond_margin(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_outcome(
        path, "fam_far", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.50)

    misses = near_miss.find_near_misses(path)

    assert misses == []


def test_excludes_null_dsr_value_never_treated_as_near_miss(tmp_path):
    path = _ledger(tmp_path)
    # An unscored outcome -- dsr_value is explicitly None, not a real
    # number. Built via not_evaluated_error (a real UNSCORED stage), since
    # record_outcome rejects a "cut_dsr + dsr_value=None" shape (see
    # _record_not_evaluated_error's own docstring).
    _record_not_evaluated_error(path, "fam_unscored", "run-1", "2026-08-01")

    misses = near_miss.find_near_misses(path)

    assert misses == []


def test_excludes_non_finite_dsr_value(tmp_path):
    path = _ledger(tmp_path)
    # record_outcome itself rejects a non-finite dsr_value on a SCORED
    # stage like cut_dsr, so this cannot reach the ledger through the
    # public API. It can still reach the file through a direct/low-level
    # write or a hand-edited row -- near_miss's own defensive isfinite
    # check exists for exactly that case, so this test writes the
    # malformed record directly via the same unlocked-append primitive
    # record_outcome itself is built on, bypassing its validation on
    # purpose.
    trial_ledger._append_unlocked(path, {
        "event": "outcome", "family_id": "fam_nan", "run_id": "run-1",
        "tested_date": "2026-08-01", "stage": "cut_dsr",
        "dsr_value": float("nan"), "note": None, "ticker_label": None,
        "sharpe_in_sample": None, "evidence": None,
    })

    misses = near_miss.find_near_misses(path)

    assert misses == []


def test_excludes_survived_outcome_even_if_dsr_below_threshold_somehow(tmp_path):
    path = _ledger(tmp_path)
    # Defensive: a survived family is not a "miss" regardless of the number
    # recorded (survived implies it cleared the bar in the pipeline's own
    # judgment; near-miss tracking must never contradict that).
    trial_ledger.record_outcome(
        path, "fam_survived", "run-1", "2026-08-01", "survived", dsr_value=0.97)

    misses = near_miss.find_near_misses(path)

    assert misses == []


def test_excludes_cut_holdout_outcome(tmp_path):
    path = _ledger(tmp_path)
    # cut_holdout means the family CLEARED the DSR bar and failed a later,
    # different check -- not a DSR near-miss at all.
    trial_ledger.record_outcome(
        path, "fam_holdout", "run-1", "2026-08-01", "cut_holdout", dsr_value=0.96)

    misses = near_miss.find_near_misses(path)

    assert misses == []


def test_uses_latest_real_scored_outcome_per_family(tmp_path):
    path = _ledger(tmp_path)
    # Re-tested family: first attempt was a near miss, second (later)
    # attempt scored far from the bar -- the CURRENT status is "far",
    # so it must not appear.
    trial_ledger.record_outcome(
        path, "fam_retested", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.92)
    trial_ledger.record_outcome(
        path, "fam_retested", "run-2", "2026-08-15", "cut_dsr", dsr_value=0.10)

    misses = near_miss.find_near_misses(path)

    assert misses == []


def test_latest_survived_outcome_overrides_earlier_near_miss(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_outcome(
        path, "fam_promoted", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.93)
    trial_ledger.record_outcome(
        path, "fam_promoted", "run-2", "2026-09-01", "survived", dsr_value=0.96)

    misses = near_miss.find_near_misses(path)

    assert misses == []


def test_intervening_null_dsr_retest_does_not_erase_a_real_near_miss(tmp_path):
    path = _ledger(tmp_path)
    # First test: a near-miss score. Second (later) test: cut for
    # insufficient trades, dsr_value None. Design choice (documented in
    # near_miss.py): the family's near-miss status is carried by its most
    # recent REAL score, not silently cleared by a later null-dsr record --
    # mirrors trial_ledger.pooled_sr_variance's own dedup-by-real-value
    # convention.
    trial_ledger.record_outcome(
        path, "fam_still_near", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.93)
    _record_not_evaluated_error(path, "fam_still_near", "run-2", "2026-08-15")

    misses = near_miss.find_near_misses(path)

    assert len(misses) == 1
    assert misses[0]["family_id"] == "fam_still_near"
    assert misses[0]["dsr_value"] == 0.93
    assert misses[0]["run_id"] == "run-1"


def test_sorts_by_distance_ascending_closest_first(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_outcome(
        path, "fam_b_closer", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.94)
    trial_ledger.record_outcome(
        path, "fam_a_farther", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.90)

    misses = near_miss.find_near_misses(path)

    assert [r["family_id"] for r in misses] == ["fam_b_closer", "fam_a_farther"]


def test_empty_ledger_returns_empty_list(tmp_path):
    path = _ledger(tmp_path)
    assert near_miss.find_near_misses(path) == []


def test_missing_first_test_record_does_not_crash(tmp_path):
    path = _ledger(tmp_path)
    # Defensive: an outcome recorded without a matching first_test event
    # (should not happen in production, but must not KeyError here).
    trial_ledger.record_outcome(
        path, "fam_no_first_test", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.91)

    misses = near_miss.find_near_misses(path)

    assert len(misses) == 1
    assert misses[0]["first_tested_date"] is None
    assert misses[0]["first_run_id"] is None


def test_custom_margin_and_threshold_are_respected(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_outcome(
        path, "fam_x", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.80)

    # Default margin (0.05 off a 0.95 bar) would exclude this.
    assert near_miss.find_near_misses(path) == []

    # A wider margin / lower threshold includes it.
    misses = near_miss.find_near_misses(path, margin=0.20, threshold=0.90)
    assert len(misses) == 1
    assert math.isclose(misses[0]["distance_from_bar"], 0.10, abs_tol=1e-9)


def test_ticker_label_and_note_are_carried_through(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_outcome(
        path, "fam_labeled", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.91,
        ticker_label="pooled 500-name universe", note="some methodology note")

    misses = near_miss.find_near_misses(path)

    assert misses[0]["ticker_label"] == "pooled 500-name universe"
    assert misses[0]["note"] == "some methodology note"


def test_render_markdown_smoke(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_first_test(
        path, "fam_close", "2026-08-01", "2026-01-01", "2026-06-30", [0.1], "run-1")
    trial_ledger.record_outcome(
        path, "fam_close", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.91)

    misses = near_miss.find_near_misses(path)
    text = near_miss.render_markdown(misses, threshold=0.95, margin=0.05,
                                      sources=[path])

    assert "fam_close" in text
    assert "0.9100" in text
    assert "0.0400" in text
    assert path.replace(os.sep, "/") in text or path in text
    assert "never re-scores" in text.lower() or "read-only" in text.lower()


def test_render_markdown_empty_says_none_found(tmp_path):
    text = near_miss.render_markdown([], threshold=0.95, margin=0.05, sources=[])
    assert "no near-miss" in text.lower() or "none" in text.lower()


def test_find_near_misses_rejects_non_finite_margin(tmp_path):
    # Defense-in-depth (any direct caller of find_near_misses, not just the
    # CLI script, must be protected): a NaN margin makes `distance > margin`
    # always False, which would silently flag every scored cut as a near
    # miss instead of erroring.
    path = _ledger(tmp_path)
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, margin=float("nan"))
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, margin=float("inf"))


def test_find_near_misses_rejects_non_positive_margin(tmp_path):
    path = _ledger(tmp_path)
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, margin=0.0)
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, margin=-0.05)


def test_find_near_misses_rejects_non_finite_threshold(tmp_path):
    path = _ledger(tmp_path)
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, threshold=float("nan"))
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, threshold=float("inf"))


def test_find_near_misses_rejects_threshold_outside_probability_range(tmp_path):
    path = _ledger(tmp_path)
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, threshold=0.0)
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, threshold=1.0)
    with pytest.raises(ValueError):
        near_miss.find_near_misses(path, threshold=1.5)


def test_render_near_miss_log_end_to_end(tmp_path):
    path = _ledger(tmp_path)
    trial_ledger.record_outcome(
        path, "fam_close", "run-1", "2026-08-01", "cut_dsr", dsr_value=0.92)

    text = near_miss.render_near_miss_log(path)

    assert "fam_close" in text


# --- find_cost_gate_near_misses -------------------------------------------
#
# A SEPARATE category from the DSR-bar near-misses above
# (render_near_miss_log/find_near_misses only ever scan
# stage == "cut_dsr" rows, so a family rejected BEFORE the DSR step ever ran
# -- stage == "insufficient_evidence", reason_code ==
# "cost_realism_gate_excluded" -- is structurally invisible to it regardless
# of how close its measured edge came to clearing the cost gate). Unlike the
# DSR margin, no invented distance constant is needed here: the natural,
# non-arbitrary split is edge_over_cost_ratio > 0 (a real, measured,
# directional edge that just wasn't large enough to survive the spread) vs.
# <= 0 (no real edge at all, not "close" in any meaningful sense).

def test_cost_gate_flags_a_positive_edge_exclusion(tmp_path):
    path = _ledger(tmp_path)
    _record_cost_gate_excluded(
        path, "fam_cost_close", "run-1", "2026-08-31",
        net_edge_bps=7.2, spread_bps=3.0, required=3.0)  # ratio 2.4

    misses = near_miss.find_cost_gate_near_misses(path)

    assert len(misses) == 1
    row = misses[0]
    assert row["family_id"] == "fam_cost_close"
    assert math.isclose(row["edge_over_cost_ratio"], 2.4, abs_tol=1e-9)
    assert row["edge_over_cost_multiple_required"] == 3.0
    assert math.isclose(row["distance_from_required"], 0.6, abs_tol=1e-9)
    assert row["run_id"] == "run-1"
    assert row["tested_date"] == "2026-08-31"
    assert row["ticker_label"] == "SPY"


def test_cost_gate_excludes_a_negative_edge_exclusion(tmp_path):
    path = _ledger(tmp_path)
    # A NEGATIVE net edge means no real directional edge at all, not a
    # near miss in any sense.
    _record_cost_gate_excluded(
        path, "fam_cost_negative", "run-1", "2026-08-31",
        net_edge_bps=-4.2, spread_bps=3.0, required=3.0)

    assert near_miss.find_cost_gate_near_misses(path) == []


def test_cost_gate_excludes_a_zero_edge_exclusion(tmp_path):
    path = _ledger(tmp_path)
    _record_cost_gate_excluded(
        path, "fam_cost_zero", "run-1", "2026-08-31",
        net_edge_bps=0.0, spread_bps=3.0, required=3.0)

    assert near_miss.find_cost_gate_near_misses(path) == []


def test_cost_gate_excludes_non_cost_gate_insufficient_evidence_reason(tmp_path):
    path = _ledger(tmp_path)
    # A DIFFERENT reason_code under the SAME stage (insufficient_evidence)
    # must never be read as a cost-gate near miss -- the family never
    # reached the cost gate at all (build_insufficient_trades, not
    # build_cost_realism_gate_excluded, matching the real reachable
    # reason-code vocabulary under stage == "insufficient_evidence" --
    # not_evaluated_error is a DIFFERENT stage entirely, tested separately
    # below).
    payload = evidence.build_insufficient_trades(
        stats={"trade_count": 1, "sharpe_in_sample": None, "net_edge_bps": None},
        catalog_entry={"family_id": "fam_other_reason", "parameters": {}},
        instrument_set=["SPY"], in_sample_start="2026-01-01",
        in_sample_end="2026-06-30",
        pinned_window={"holdout_start": "2026-07-01", "holdout_end": "2026-08-31"},
        holdout_look="unspent", signal_source_path="src/mining/fake.py",
        signal_source_sha256="a" * 64, evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "test"})
    trial_ledger.record_outcome(
        path, "fam_other_reason", "run-1", "2026-08-01", "insufficient_evidence",
        evidence=payload)

    assert near_miss.find_cost_gate_near_misses(path) == []


def test_cost_gate_excludes_not_evaluated_error_reason(tmp_path):
    path = _ledger(tmp_path)
    # A different STAGE entirely (not_evaluated_error) must also never be
    # read as a cost-gate near miss.
    _record_not_evaluated_error(path, "fam_engine_error", "run-1", "2026-08-01")

    assert near_miss.find_cost_gate_near_misses(path) == []


def test_cost_gate_excludes_family_later_rescored_for_real(tmp_path):
    path = _ledger(tmp_path)
    # Status moved on: a fresh run later cleared the gate and scored the
    # family for real (a SCORED outcome -- evidence.SCORED_STAGES) -- a
    # stale gate exclusion must not shadow a real, current score.
    _record_cost_gate_excluded(
        path, "fam_cost_rescored", "run-1", "2026-08-31",
        net_edge_bps=7.2, spread_bps=3.0, required=3.0)
    trial_ledger.record_outcome(
        path, "fam_cost_rescored", "run-2", "2026-09-22", "cut_dsr",
        dsr_value=0.10)

    assert near_miss.find_cost_gate_near_misses(path) == []


def test_cost_gate_survives_an_unrelated_later_retry_failure(tmp_path):
    path = _ledger(tmp_path)
    # Guards against tracking each family's latest `outcome` event of ANY
    # kind: a later, UNRELATED retry failure (e.g. an engine crash on
    # re-attempt -- nothing to do with the cost gate) would then silently
    # erase a real, still-current cost-gate near miss. The family's
    # cost-gate status has not actually changed -- the retry never even
    # reached the cost gate -- so it must still be reported.
    _record_cost_gate_excluded(
        path, "fam_survives_retry", "run-1", "2026-08-31",
        net_edge_bps=7.2, spread_bps=3.0, required=3.0)  # ratio 2.4
    _record_not_evaluated_error(
        path, "fam_survives_retry", "run-2", "2026-09-10")

    misses = near_miss.find_cost_gate_near_misses(path)

    assert len(misses) == 1
    assert misses[0]["family_id"] == "fam_survives_retry"
    assert math.isclose(misses[0]["edge_over_cost_ratio"], 2.4, abs_tol=1e-9)


def test_cost_gate_a_fresh_exclusion_supersedes_an_older_one(tmp_path):
    path = _ledger(tmp_path)
    # Unlike an unrelated retry failure, a SECOND cost-gate exclusion (a
    # re-measured spread/edge) IS relevant and must replace the first --
    # here the fresher measurement clears the gate (negative edge), so the
    # family must no longer be reported.
    _record_cost_gate_excluded(
        path, "fam_remeasured", "run-1", "2026-08-31",
        net_edge_bps=7.2, spread_bps=3.0, required=3.0)  # ratio 2.4
    _record_cost_gate_excluded(
        path, "fam_remeasured", "run-2", "2026-09-10",
        net_edge_bps=-1.0, spread_bps=3.0, required=3.0)  # ratio negative

    assert near_miss.find_cost_gate_near_misses(path) == []


def test_cost_gate_sorts_closest_to_required_first(tmp_path):
    path = _ledger(tmp_path)
    _record_cost_gate_excluded(
        path, "fam_far", "run-1", "2026-08-31",
        net_edge_bps=1.2, spread_bps=1.0, required=3.0)  # ratio 1.2
    _record_cost_gate_excluded(
        path, "fam_close", "run-1", "2026-08-31",
        net_edge_bps=2.4, spread_bps=1.0, required=3.0)  # ratio 2.4

    misses = near_miss.find_cost_gate_near_misses(path)

    assert [r["family_id"] for r in misses] == ["fam_close", "fam_far"]


def test_render_near_miss_log_includes_cost_gate_section(tmp_path):
    path = _ledger(tmp_path)
    _record_cost_gate_excluded(
        path, "fam_cost_close", "run-1", "2026-08-31",
        net_edge_bps=7.2, spread_bps=3.0, required=3.0)

    text = near_miss.render_near_miss_log(path)

    assert "fam_cost_close" in text
    assert "cost" in text.lower()


def test_cost_gate_render_handles_missing_required_multiple_gracefully(tmp_path):
    path = _ledger(tmp_path)
    # A malformed record missing edge_over_cost_multiple_required --
    # written directly via the unlocked-append primitive, bypassing
    # evidence.validate, the same technique test_excludes_non_finite_dsr_value
    # above uses. find_cost_gate_near_misses and render_markdown must both
    # degrade gracefully (None / "n/a"), not crash.
    trial_ledger._append_unlocked(path, {
        "event": "outcome", "family_id": "fam_malformed_no_required",
        "run_id": "run-1", "tested_date": "2026-08-31",
        "stage": "insufficient_evidence", "dsr_value": None, "note": None,
        "ticker_label": "SPY", "sharpe_in_sample": None,
        "evidence": {
            "reason_code": "cost_realism_gate_excluded",
            "observed": {"edge_over_cost_ratio": 1.5, "spread_bps": 2.0,
                        "net_edge_bps": 3.0},
        },
    })

    misses = near_miss.find_cost_gate_near_misses(path)

    assert len(misses) == 1
    assert misses[0]["edge_over_cost_multiple_required"] is None
    assert misses[0]["distance_from_required"] is None

    text = near_miss.render_near_miss_log(path)
    assert "fam_malformed_no_required" in text
    assert "n/a" in text
