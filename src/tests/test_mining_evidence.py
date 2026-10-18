"""Tests for src/mining/evidence.py -- the semantic core of the
"we could not judge this" outcome states.

The failure mode this whole module exists to prevent: a candidate that was
never scored being recorded identically to a candidate that was scored and
rejected. A ledger that records a family as stage="cut_dsr" with
dsr_value=null, and no note, cannot be told apart from a real rejection.
These tests pin the vocabulary and
the cross-field rules that make that record shape impossible to build.

No fixture files, no network, no market data -- every payload here is
built by a module-level _-prefixed builder.
"""
import ast
import builtins
import os
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import constants
from mining import evidence


# --- builders --------------------------------------------------------------

# Two example parameter sets, copied verbatim from two catalog entries. Used
# as literals rather than read from the file so these tests never depend on a
# shared, concurrently edited data file.
_PFF_PARAMS = {"yield_lookback_years": 3, "entry_zscore": 0.75,
               "exit_zscore": 0, "rebalance": "monthly"}
_UUP_PARAMS = {"lookback_months": 12, "rebalance": "monthly",
               "signal": "trailing_total_return_sign", "long_only": True}


def _catalog_entry(family_id="preferred_stock_yield_spread_reversion",
                   tickers=("PFF",), parameters=None):
    return {"family_id": family_id, "name": "x", "mechanism": "x",
            "tickers": list(tickers),
            "parameters": dict(_PFF_PARAMS if parameters is None else parameters),
            "account_types": ["taxable"], "status": "eligible",
            "source": "x", "added_date": "2026-09-01"}


def _pinned_window():
    return {"holdout_start": "2026-03-01", "holdout_end": "2026-09-01",
            "tested_date": "2026-03-01"}


def _universe_selection(**over):
    block = {"rule": "example rule: US-listed preferred-income ETFs above an "
                     "assets-under-management floor",
             "as_of": "2026-09-02",
             "survivorship_bias": "survivor_only_disclosed"}
    block.update(over)
    return block


def _widen(**over):
    block = {"kind": "widened_instrument_set",
             "candidate_tickers": ["PFF", "PGX", "PSK", "FPE", "PFFD"],
             "universe_selection": _universe_selection(),
             "registered_on": "2026-09-02",
             "registered_by_run_id": "run-amend-x",
             "rationale": "pool across sibling preferred-income ETFs"}
    block.update(over)
    return block


def _observed(**over):
    block = {"trade_count": 5,
             "trades_required": constants.MIN_TRADES_FOR_SCORING,
             "in_sample_start": "2008-01-02",
             "in_sample_end": "2026-02-27",
             "instrument_set": ["PFF"],
             "sharpe_in_sample": 0.5,
             "net_edge_bps": 500.0,
             "parameters": dict(_PFF_PARAMS),
             "parameter_fingerprint": evidence.parameter_fingerprint(
                 _catalog_entry()),
             "signal_source_sha256": "a" * 64,
             "signal_source_path": "runs/x/signal_fn.py"}
    block.update(over)
    return block


def _payload(**over):
    payload = {"schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
               "reason_code": "insufficient_trades",
               "evidence_source": "measured",
               "observed": _observed(),
               "holdout": {"holdout_start": "2026-03-01",
                           "holdout_end": "2026-09-01",
                           "look": "unspent"},
               "reactivation": _widen(),
               "supersedes": None}
    payload.update(over)
    return payload


def _engine_payload(**over):
    payload = {"schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
               "reason_code": "engine_error",
               "evidence_source": "measured",
               "observed": {"error_type": "RuntimeError",
                            "error_message": "boom",
                            "instrument_set": ["PFF"]},
               "holdout": {"holdout_start": "2026-03-01",
                           "holdout_end": "2026-09-01",
                           "look": "unspent"},
               "reactivation": {"kind": "none_available",
                                "rationale": "engine error is a defect, "
                                             "not evidence"},
               "supersedes": None}
    payload.update(over)
    return payload


def _holdout_trade_observed(**over):
    block = {"in_sample_trade_count": 80,
             "holdout_trade_count": 5,
             "holdout_trades_required": constants.MIN_HOLDOUT_TRADES,
             "in_sample_start": "2008-01-02",
             "in_sample_end": "2026-02-27",
             "instrument_set": ["PFF"],
             "sharpe_in_sample": 0.5,
             "net_edge_bps": 500.0,
             "holdout_net_edge_bps": 100.0,
             "parameters": dict(_PFF_PARAMS),
             "parameter_fingerprint": evidence.parameter_fingerprint(
                 _catalog_entry()),
             "signal_source_sha256": "a" * 64,
             "signal_source_path": "runs/x/signal_fn.py"}
    block.update(over)
    return block


def _holdout_trade_payload(**over):
    payload = {"schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
               "reason_code": "insufficient_holdout_trades",
               "evidence_source": "measured",
               "observed": _holdout_trade_observed(),
               "holdout": {"holdout_start": "2026-03-01",
                           "holdout_end": "2026-09-01",
                           "look": "computed_not_scored"},
               "reactivation": {"kind": "none_available",
                                "rationale": "the holdout window is pinned "
                                             "and fixed-length; no amount "
                                             "of waiting adds holdout "
                                             "trades"},
               "supersedes": None}
    payload.update(over)
    return payload


def _write(path, text, newline="\n"):
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
    return str(path)


# --- the two payload shapes are accepted -----------------------------------

def test_validate_accepts_a_well_formed_insufficient_trades_payload():
    """The happy path must actually pass, or every rejection test below is
    vacuous (it would be rejecting for the wrong reason)."""
    evidence.validate(_payload(), stage="insufficient_evidence")


def test_validate_accepts_a_well_formed_engine_error_payload():
    evidence.validate(_engine_payload(), stage="not_evaluated_error")


def test_stage_vocabularies_are_disjoint_and_cover_valid_stages():
    """A stage in both halves would make the biconditional unsatisfiable."""
    assert set(evidence.SCORED_STAGES) & set(evidence.UNSCORED_STAGES) == set()
    assert set(evidence.VALID_STAGES) == (set(evidence.SCORED_STAGES)
                                          | set(evidence.UNSCORED_STAGES))


# --- closed vocabularies: each rejects a plausible near-miss ---------------

def test_validate_rejects_a_near_miss_reason_code():
    """'insufficient_trade' (singular) is the typo a hand-written call site
    makes; it must not be silently stored."""
    with pytest.raises(ValueError, match="reason_code"):
        evidence.validate(_payload(reason_code="insufficient_trade"),
                          stage="insufficient_evidence")


def test_validate_rejects_a_near_miss_evidence_source():
    with pytest.raises(ValueError, match="evidence_source"):
        evidence.validate(_payload(evidence_source="measure"),
                          stage="insufficient_evidence")


def test_validate_rejects_a_near_miss_holdout_look():
    """'spent' reads as a synonym for computed_not_scored but is not in the
    vocabulary; a reader would take it as a claim the holdout was used."""
    p = _payload()
    p["holdout"]["look"] = "spent"
    with pytest.raises(ValueError, match="holdout.look"):
        evidence.validate(p, stage="insufficient_evidence")


def test_validate_rejects_a_near_miss_reactivation_kind():
    with pytest.raises(ValueError, match="reactivation.kind"):
        evidence.validate(_payload(reactivation=_widen(
            kind="widened_instruments")), stage="insufficient_evidence")


def test_validate_rejects_a_near_miss_survivorship_label():
    """'survivorship_free' is a claim no available data source supports."""
    r = _widen(universe_selection=_universe_selection(
        survivorship_bias="survivorship_free"))
    with pytest.raises(ValueError, match="survivorship_bias"):
        evidence.validate(_payload(reactivation=r),
                          stage="insufficient_evidence")


def test_validate_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="stage"):
        evidence.validate(_payload(), stage="insufficient-evidence")


def test_validate_rejects_a_scored_stage():
    """validate() is only ever meaningful for an UNSCORED stage. A scored
    stage reaching it means the ledger's biconditional was bypassed."""
    with pytest.raises(ValueError, match="stage"):
        evidence.validate(_payload(), stage="cut_dsr")


def test_validate_rejects_a_wrong_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        evidence.validate(_payload(schema_version=2),
                          stage="insufficient_evidence")


def test_validate_rejects_an_unknown_top_level_key():
    """Fail closed on an unrecognised key: a typo'd key name would otherwise
    be silently persisted and silently ignored by every reader."""
    with pytest.raises(ValueError, match="unexpected"):
        evidence.validate(_payload(reactivations=_widen()),
                          stage="insufficient_evidence")


# --- the reason_code <-> stage biconditional, both directions -------------

def test_validate_rejects_engine_error_reason_on_the_insufficient_stage():
    with pytest.raises(ValueError, match="engine_error"):
        evidence.validate(_engine_payload(), stage="insufficient_evidence")


def test_validate_rejects_a_trade_count_reason_on_the_error_stage():
    with pytest.raises(ValueError, match="engine_error"):
        evidence.validate(_payload(), stage="not_evaluated_error")


# --- trade-count rules ----------------------------------------------------

def test_validate_rejects_insufficient_trades_with_zero_trades():
    """Zero trades is qualitatively different from a small sample and has
    its own reason code."""
    with pytest.raises(ValueError, match="trade_count"):
        evidence.validate(_payload(observed=_observed(trade_count=0)),
                          stage="insufficient_evidence")


def test_validate_rejects_insufficient_trades_at_the_required_count():
    """15 trades is SCOREABLE. Recording it as insufficient evidence would
    hide a real rejection behind a deferral."""
    n = constants.MIN_TRADES_FOR_SCORING
    with pytest.raises(ValueError, match="trade_count"):
        evidence.validate(_payload(observed=_observed(trade_count=n)),
                          stage="insufficient_evidence")


def test_validate_accepts_insufficient_trades_one_below_the_required_count():
    """The boundary from the other side -- one below the required count
    is still insufficient evidence."""
    n = constants.MIN_TRADES_FOR_SCORING - 1
    evidence.validate(_payload(observed=_observed(trade_count=n)),
                      stage="insufficient_evidence")


def test_validate_rejects_zero_trades_reason_with_a_nonzero_count():
    with pytest.raises(ValueError, match="trade_count"):
        evidence.validate(_payload(reason_code="zero_trades",
                                   observed=_observed(trade_count=1)),
                          stage="insufficient_evidence")


def test_validate_accepts_zero_trades_reason_at_zero():
    p = _payload(reason_code="zero_trades",
                 observed=_observed(trade_count=0, sharpe_in_sample=None,
                                    net_edge_bps=None))
    evidence.validate(p, stage="insufficient_evidence")


def test_validate_at_write_time_rejects_a_stale_trades_required():
    """trades_required records the constant IN FORCE at write time. A
    hand-typed stale value would make a NEW record lie about the bar it was
    judged against -- so the equality is enforced on the WRITE path, where
    "the constant in force" and "now" are the same instant."""
    with pytest.raises(ValueError, match="trades_required"):
        evidence.validate(_payload(observed=_observed(trades_required=10)),
                          stage="insufficient_evidence", at_write_time=True)


def test_validate_by_default_does_not_apply_the_write_time_bar_check():
    """The default is the READ path. If this ever flips, every record written
    under an older bar stops being readable -- the exact regression the
    write-time keyword exists to prevent."""
    evidence.validate(_payload(observed=_observed(trades_required=10)),
                      stage="insufficient_evidence")


def test_validate_reads_back_a_record_written_under_an_older_bar(monkeypatch):
    """observed.trades_required is recorded so an old record stays
    SELF-DESCRIBING when the bar moves. Enforcing equality with the CURRENT
    constant on every call delivers the opposite: the moment the bar moves,
    every historical record becomes unreadable -- and validate() is on the
    READ path (validate_retry_request calls it on a payload read back from
    the ledger). MIN_TRADES_FOR_SCORING is a pre-registered bar this project
    may revisit, as it may the sibling DSR bar."""
    historical = _payload()  # trades_required == the bar in force right now
    monkeypatch.setattr(constants, "MIN_TRADES_FOR_SCORING",
                        constants.MIN_TRADES_FOR_SCORING + 5)
    evidence.validate(historical, stage="insufficient_evidence")


def test_validate_rejects_a_nonpositive_trades_required():
    """The read path still refuses a nonsensical bar -- 'stays readable' is
    not 'accepts anything'. Checked BEFORE the reason_code count rules so the
    error names the field that is actually wrong."""
    with pytest.raises(ValueError, match="trades_required"):
        evidence.validate(_payload(observed=_observed(trades_required=0)),
                          stage="insufficient_evidence")


def test_validate_rejects_a_trade_count_at_or_above_the_recorded_bar():
    """A record claiming to be unscoreable while carrying enough trades to
    score is incoherent no matter which bar was in force."""
    with pytest.raises(ValueError, match="trade_count"):
        evidence.validate(
            _payload(observed=_observed(trade_count=40, trades_required=40)),
            stage="insufficient_evidence")


def test_a_historical_record_can_still_be_amended_after_a_bar_change(
        monkeypatch):
    """Why the bar-equality check is write-time only: validate_amendment
    requires the amendment's observed block to equal the superseded one
    byte-for-byte, so an amendment of a historical record necessarily carries
    the OLD trades_required. If validate() demanded the current constant on
    every call, such a record could be neither retried nor amended --
    permanently frozen."""
    prior = _payload()
    new = _payload(evidence_source="amended",
                   reactivation=_widen(candidate_tickers=[
                       "PFF", "PGX", "PSK", "FPE", "PFFD", "PFFA"]),
                   supersedes={"ledger_line": 12, "line_sha256": "b" * 64,
                               "recorded_stage": "cut_dsr"})
    monkeypatch.setattr(constants, "MIN_TRADES_FOR_SCORING",
                        constants.MIN_TRADES_FOR_SCORING + 5)
    evidence.validate(prior, stage="insufficient_evidence")
    evidence.validate(new, stage="insufficient_evidence")
    evidence.validate_amendment(new, prior)


def test_validate_rejects_a_non_finite_sharpe():
    """NaN is ambiguous garbage; None is an explicit 'not computed'."""
    with pytest.raises(ValueError, match="sharpe_in_sample"):
        evidence.validate(
            _payload(observed=_observed(sharpe_in_sample=float("nan"))),
            stage="insufficient_evidence")


def test_validate_rejects_a_bool_trade_count():
    """isinstance(True, int) is True in Python."""
    with pytest.raises(ValueError, match="trade_count"):
        evidence.validate(_payload(observed=_observed(trade_count=True)),
                          stage="insufficient_evidence")


# --- date ordering --------------------------------------------------------

def test_validate_rejects_in_sample_start_not_before_in_sample_end():
    with pytest.raises(ValueError, match="in_sample_start"):
        evidence.validate(
            _payload(observed=_observed(in_sample_start="2026-02-27",
                                        in_sample_end="2026-02-27")),
            stage="insufficient_evidence")


def test_validate_rejects_in_sample_end_on_or_after_holdout_start():
    """in-sample is index < holdout_start (run_batch3_mining.py's
    _window_masks), so an in-sample bar at or past the boundary is a leak."""
    with pytest.raises(ValueError, match="in_sample_end"):
        evidence.validate(
            _payload(observed=_observed(in_sample_end="2026-03-01")),
            stage="insufficient_evidence")


def test_validate_rejects_holdout_end_before_holdout_start():
    p = _payload()
    p["holdout"]["holdout_end"] = "2026-02-01"
    with pytest.raises(ValueError, match="holdout_end"):
        evidence.validate(p, stage="insufficient_evidence")


def test_validate_rejects_a_malformed_date():
    with pytest.raises(ValueError, match="in_sample_start"):
        evidence.validate(_payload(observed=_observed(
            in_sample_start="2007-13-45")), stage="insufficient_evidence")


# --- reactivation rules ---------------------------------------------------

def test_validate_rejects_a_reactivation_list():
    """Exactly one predicate. A disjunction is a weaker commitment and
    invites taking whichever fires first."""
    with pytest.raises(ValueError, match="never a list"):
        evidence.validate(_payload(reactivation=[_widen()]),
                          stage="insufficient_evidence")


def test_validate_rejects_a_widened_set_that_is_not_a_strict_superset():
    """Retesting the same instrument set is a bit-identical re-run."""
    with pytest.raises(ValueError, match="candidate_tickers"):
        evidence.validate(_payload(reactivation=_widen(
            candidate_tickers=["PFF"])), stage="insufficient_evidence")


def test_validate_rejects_a_widened_set_that_drops_the_original():
    """The instrument that produced the seed result cannot be swapped out."""
    with pytest.raises(ValueError, match="candidate_tickers"):
        evidence.validate(_payload(reactivation=_widen(
            candidate_tickers=["PGX", "PSK", "FPE"])),
            stage="insufficient_evidence")


def test_validate_rejects_duplicate_candidate_tickers():
    with pytest.raises(ValueError, match="candidate_tickers"):
        evidence.validate(_payload(reactivation=_widen(
            candidate_tickers=["PFF", "PGX", "PGX"])),
            stage="insufficient_evidence")


def test_validate_rejects_a_lowercase_candidate_ticker():
    with pytest.raises(ValueError, match="candidate_tickers"):
        evidence.validate(_payload(reactivation=_widen(
            candidate_tickers=["PFF", "pgx"])),
            stage="insufficient_evidence")


def test_validate_rejects_a_widened_set_without_universe_selection():
    """Survivorship bias is not correctable here, so it is made mandatory to
    disclose -- a pooled ETF universe enumerated in 2026 cannot see funds
    that closed."""
    r = _widen()
    del r["universe_selection"]
    with pytest.raises(ValueError, match="universe_selection"):
        evidence.validate(_payload(reactivation=r),
                          stage="insufficient_evidence")


def test_validate_rejects_extended_history_not_earlier_than_in_sample():
    r = {"kind": "extended_history", "earliest_available": "2008-01-02",
         "source": "Cohen & Steers monthly distributions",
         "registered_on": "2026-09-02",
         "registered_by_run_id": "run-x", "rationale": "deeper history"}
    with pytest.raises(ValueError, match="earliest_available"):
        evidence.validate(_payload(reactivation=r),
                          stage="insufficient_evidence")


def test_validate_rejects_extended_history_without_a_source():
    r = {"kind": "extended_history", "earliest_available": "1998-01-02",
         "registered_on": "2026-09-02", "registered_by_run_id": "run-x",
         "rationale": "deeper history"}
    with pytest.raises(ValueError, match="source"):
        evidence.validate(_payload(reactivation=r),
                          stage="insufficient_evidence")


def test_validate_accepts_extended_history_strictly_earlier():
    r = {"kind": "extended_history", "earliest_available": "1998-01-02",
         "source": "Cohen & Steers monthly distributions",
         "registered_on": "2026-09-02", "registered_by_run_id": "run-x",
         "rationale": "deeper history"}
    evidence.validate(_payload(reactivation=r), stage="insufficient_evidence")


def test_validate_rejects_none_available_with_an_empty_rationale():
    with pytest.raises(ValueError, match="rationale"):
        evidence.validate(_payload(reactivation={"kind": "none_available",
                                                 "rationale": "   "}),
                          stage="insufficient_evidence")


def test_validate_rejects_a_time_based_reactivation_kind():
    """There is no time lever, and this is the load-bearing consequence of
    the pinned window: in-sample is `index < holdout_start`, so waiting for
    calendar time can never add an in-sample trade to a pinned family."""
    assert "forward_accrual" not in evidence.REACTIVATION_KINDS
    with pytest.raises(ValueError, match="reactivation.kind"):
        evidence.validate(_payload(reactivation={
            "kind": "forward_accrual", "rationale": "wait a year"}),
            stage="insufficient_evidence")


def test_validate_rejects_an_engine_error_carrying_a_lever():
    """An engine error is a bug, not a research dead end -- the family stays
    eligible and re-enters normally once fixed, so no predicate applies."""
    with pytest.raises(ValueError, match="none_available"):
        evidence.validate(_engine_payload(reactivation=_widen()),
                          stage="not_evaluated_error")


def test_validate_rejects_an_engine_error_observed_with_extra_keys():
    obs = {"error_type": "RuntimeError", "error_message": "boom",
           "instrument_set": ["PFF"], "trade_count": 0}
    with pytest.raises(ValueError, match="observed"):
        evidence.validate(_engine_payload(observed=obs),
                          stage="not_evaluated_error")


# --- free text and supersedes --------------------------------------------

def test_validate_rejects_a_newline_in_a_free_text_field():
    """The ledger is JSON Lines -- one record per line. A newline would not
    corrupt the JSON encoding but does break every grep/sed reader of the
    file, and free-text prose has no need of one."""
    with pytest.raises(ValueError, match="newline"):
        evidence.validate(_payload(reactivation=_widen(
            rationale="pool across siblings\nand hope")),
            stage="insufficient_evidence")


def test_validate_rejects_amended_source_without_supersedes():
    with pytest.raises(ValueError, match="supersedes"):
        evidence.validate(_payload(evidence_source="amended"),
                          stage="insufficient_evidence")


def test_validate_rejects_a_malformed_supersedes_block():
    with pytest.raises(ValueError, match="supersedes"):
        evidence.validate(_payload(evidence_source="amended",
                                   supersedes={"ledger_line": 12}),
                          stage="insufficient_evidence")


def test_validate_accepts_an_amended_payload_with_supersedes():
    evidence.validate(
        _payload(evidence_source="amended",
                 supersedes={"ledger_line": 12, "line_sha256": "b" * 64,
                             "recorded_stage": "cut_dsr"}),
        stage="insufficient_evidence")


# --- constructors ---------------------------------------------------------

def _build_kwargs(**over):
    kwargs = {"stats": {"trade_count": 5, "sharpe_in_sample": 0.5,
                        "net_edge_bps": 500.0},
              "catalog_entry": _catalog_entry(),
              "instrument_set": ["PFF"],
              "in_sample_start": "2008-01-02",
              "in_sample_end": "2026-02-27",
              "pinned_window": _pinned_window(),
              "holdout_look": "unspent",
              "signal_source_path": "runs/x/signal_fn.py",
              "signal_source_sha256": "a" * 64,
              "evidence_source": "measured",
              "reactivation": _widen()}
    kwargs.update(over)
    return kwargs


def test_build_insufficient_trades_computes_reason_code_and_trades_required():
    """Both are DERIVED from data already in hand. A caller that could type
    them by hand could type them wrong, and the record would rot silently."""
    payload = evidence.build_insufficient_trades(**_build_kwargs())
    assert payload["reason_code"] == "insufficient_trades"
    assert payload["observed"]["trades_required"] == \
        constants.MIN_TRADES_FOR_SCORING
    evidence.validate(payload, stage="insufficient_evidence")


def test_build_insufficient_trades_chooses_zero_trades_at_zero():
    payload = evidence.build_insufficient_trades(**_build_kwargs(
        stats={"trade_count": 0, "sharpe_in_sample": None,
               "net_edge_bps": None}))
    assert payload["reason_code"] == "zero_trades"
    evidence.validate(payload, stage="insufficient_evidence")


def test_build_insufficient_trades_refuses_a_hand_typed_trades_required():
    with pytest.raises(TypeError):
        evidence.build_insufficient_trades(
            **_build_kwargs(), trades_required=15)


def test_build_insufficient_trades_refuses_a_hand_typed_reason_code():
    with pytest.raises(TypeError):
        evidence.build_insufficient_trades(
            **_build_kwargs(), reason_code="insufficient_trades")


def test_build_insufficient_trades_computes_the_parameter_fingerprint():
    """The fingerprint is derived from the catalog entry inside the
    constructor -- never accepted as an argument."""
    payload = evidence.build_insufficient_trades(**_build_kwargs())
    assert payload["observed"]["parameter_fingerprint"] == \
        evidence.parameter_fingerprint(_catalog_entry())
    assert payload["observed"]["parameters"] == _PFF_PARAMS


def test_build_insufficient_trades_refuses_a_scoreable_trade_count():
    """Fail closed: 15 trades should have been scored, not deferred."""
    with pytest.raises(ValueError, match="trade_count"):
        evidence.build_insufficient_trades(**_build_kwargs(
            stats={"trade_count": constants.MIN_TRADES_FOR_SCORING,
                   "sharpe_in_sample": 0.9, "net_edge_bps": 10.0}))


def test_build_insufficient_trades_refuses_stats_without_a_trade_count():
    with pytest.raises(ValueError, match="trade_count"):
        evidence.build_insufficient_trades(**_build_kwargs(
            stats={"sharpe_in_sample": 0.9, "net_edge_bps": 10.0}))


def test_build_engine_error_records_the_exception_type_and_message():
    payload = evidence.build_engine_error(
        error=RuntimeError("fetch failed for PFF"), instrument_set=["PFF"],
        pinned_window=_pinned_window(), holdout_look="unspent")
    assert payload["observed"]["error_type"] == "RuntimeError"
    assert "fetch failed for PFF" in payload["observed"]["error_message"]
    assert payload["reason_code"] == "engine_error"
    assert payload["reactivation"]["kind"] == "none_available"
    evidence.validate(payload, stage="not_evaluated_error")


def test_build_engine_error_collapses_newlines_in_the_message():
    """A traceback-bearing exception message would otherwise violate the
    one-line rule and be unwritable."""
    payload = evidence.build_engine_error(
        error=ValueError("line one\nline two"), instrument_set=["PFF"],
        pinned_window=_pinned_window(), holdout_look="unspent")
    assert "\n" not in payload["observed"]["error_message"]
    assert "line one" in payload["observed"]["error_message"]
    assert "line two" in payload["observed"]["error_message"]
    evidence.validate(payload, stage="not_evaluated_error")


def test_build_engine_error_collapses_a_bare_carriage_return():
    r"""_reject_newlines refuses a lone \r exactly as it refuses \n, so a
    collapse that handled only \r\n and \n would raise ValueError INSIDE the
    very exception handler whose job is to record the crash. A bare \r is
    what progress-bar output carries, and this repo's data path is yfinance;
    escaping run_batch3_mining.py's _record_defensive_outcomes loop would
    skip the ledger write for every remaining family in the batch, re-opening
    the cross-run multiple-testing leak the ledger exists to close."""
    payload = evidence.build_engine_error(
        error=RuntimeError("downloading\rfailed for PFF"),
        instrument_set=["PFF"], pinned_window=_pinned_window(),
        holdout_look="unspent")
    message = payload["observed"]["error_message"]
    assert "\r" not in message and "\n" not in message
    assert "downloading" in message and "failed for PFF" in message
    evidence.validate(payload, stage="not_evaluated_error")


def test_build_engine_error_collapses_mixed_terminators_without_empty_parts():
    r"""All three terminators in one message collapse to one separator each.

    TWO mechanisms hold this up and they are mutually redundant -- the
    \r\n-first alternation (so CRLF matches as one separator) and the
    `if part.strip()` filter (which drops an empty segment if one is produced
    anyway). Removing either alone still passes; removing BOTH would yield
    'alpha |  | bravo | charlie | delta' and fail this test. So neither line
    alone is load-bearing here."""
    payload = evidence.build_engine_error(
        error=RuntimeError("alpha\r\nbravo\rcharlie\ndelta"),
        instrument_set=["PFF"], pinned_window=_pinned_window(),
        holdout_look="unspent")
    assert payload["observed"]["error_message"] == \
        "alpha | bravo | charlie | delta"


# --- fingerprint and source hash -----------------------------------------

def test_parameter_fingerprint_ignores_ticker_changes():
    """Widening the instrument set is the sanctioned lever, so it must NOT
    invalidate the hypothesis identity."""
    a = evidence.parameter_fingerprint(_catalog_entry(tickers=["PFF"]))
    b = evidence.parameter_fingerprint(
        _catalog_entry(tickers=["PFF", "PGX", "PSK"]))
    assert a == b


def test_parameter_fingerprint_changes_when_a_parameter_changes():
    """Changing entry_zscore from 0.75 to 0.50 to manufacture trades is a
    NEW hypothesis, not a retry of this one."""
    drifted = dict(_PFF_PARAMS, entry_zscore=0.50)
    assert evidence.parameter_fingerprint(_catalog_entry()) != \
        evidence.parameter_fingerprint(_catalog_entry(parameters=drifted))


def test_parameter_fingerprint_changes_with_family_id():
    assert evidence.parameter_fingerprint(_catalog_entry()) != \
        evidence.parameter_fingerprint(
            _catalog_entry(family_id="fx_dollar_index_trend_following"))


def test_parameter_fingerprint_is_insensitive_to_key_order():
    reordered = {k: _PFF_PARAMS[k] for k in reversed(list(_PFF_PARAMS))}
    assert evidence.parameter_fingerprint(_catalog_entry()) == \
        evidence.parameter_fingerprint(_catalog_entry(parameters=reordered))


def test_source_sha256_hashes_raw_bytes_so_crlf_and_lf_differ(tmp_path):
    """Hash the artifact, not a decoded buffer: line-ending translation can
    move a file's bytes after hashing, and a text-mode read with universal
    newlines would collapse these two files to the same digest."""
    lf = _write(tmp_path / "lf.py", "def f():\n    return 1\n", newline="\n")
    crlf = _write(tmp_path / "crlf.py", "def f():\n    return 1\n",
                  newline="\r\n")
    assert open(lf, "rb").read() != open(crlf, "rb").read()
    assert evidence.source_sha256(lf) != evidence.source_sha256(crlf)


def test_source_sha256_matches_hashlib_on_the_raw_bytes(tmp_path):
    import hashlib
    path = _write(tmp_path / "s.py", "x = 1\r\ny = 2\r\n", newline="")
    expected = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert evidence.source_sha256(path) == expected


def test_source_sha256_raises_on_a_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        evidence.source_sha256(str(tmp_path / "nope.py"))


# --- years_to_threshold (display only) ------------------------------------

def test_years_to_threshold_on_a_sparse_fixture_is_decades():
    """Synthetic fixture: 5 trades over ~18.2 years (~0.28/yr) needs 10
    more -- roughly 36 years. This is the number that makes 'wait for more trades' visibly
    absurd rather than plausible."""
    years = evidence.years_to_threshold(_observed())
    assert 33.0 <= years <= 40.0


def test_years_to_threshold_is_none_at_zero_trades():
    assert evidence.years_to_threshold(_observed(trade_count=0)) is None


def test_years_to_threshold_is_zero_when_the_threshold_is_met():
    obs = _observed(trade_count=constants.MIN_TRADES_FOR_SCORING)
    assert evidence.years_to_threshold(obs) == 0.0


# --- reactivation_satisfiable --------------------------------------------

def test_reactivation_satisfiable_true_for_a_widened_instrument_set():
    ok, reason = evidence.reactivation_satisfiable(
        _payload(), as_of="2026-09-03", catalog_entry=_catalog_entry(),
        retry_ordinal=1)
    assert ok is True
    assert "PGX" in reason


def test_reactivation_satisfiable_false_for_none_available():
    payload = _payload(reactivation={"kind": "none_available",
                                     "rationale": "no deeper history exists"})
    ok, reason = evidence.reactivation_satisfiable(
        payload, as_of="2026-09-03", catalog_entry=_catalog_entry(),
        retry_ordinal=1)
    assert ok is False
    assert "none_available" in reason or "no deeper history" in reason


def test_reactivation_satisfiable_at_the_cap_and_false_one_past_it():
    """House boundary convention: the cap value itself passes."""
    cap = constants.MAX_EVIDENCE_RETRIES_PER_FAMILY
    ok_at, _ = evidence.reactivation_satisfiable(
        _payload(), as_of="2026-09-03", catalog_entry=_catalog_entry(),
        retry_ordinal=cap)
    ok_past, reason = evidence.reactivation_satisfiable(
        _payload(), as_of="2026-09-03", catalog_entry=_catalog_entry(),
        retry_ordinal=cap + 1)
    assert ok_at is True
    assert ok_past is False
    assert "cap" in reason.lower()


def test_reactivation_satisfiable_false_before_the_predicate_was_registered():
    """Pre-registration in the literal sense: a predicate cannot be
    evaluated as of a date before it existed."""
    ok, reason = evidence.reactivation_satisfiable(
        _payload(), as_of="2026-09-01", catalog_entry=_catalog_entry(),
        retry_ordinal=1)
    assert ok is False
    assert "registered_on" in reason


def test_reactivation_satisfiable_false_when_catalog_parameters_drifted():
    drifted = _catalog_entry(parameters=dict(_PFF_PARAMS, entry_zscore=0.50))
    ok, reason = evidence.reactivation_satisfiable(
        _payload(), as_of="2026-09-03", catalog_entry=drifted, retry_ordinal=1)
    assert ok is False
    assert "entry_zscore" in reason


def test_reactivation_satisfiable_requires_an_as_of_date():
    with pytest.raises(ValueError, match="as_of"):
        evidence.reactivation_satisfiable(
            _payload(), as_of=None, catalog_entry=_catalog_entry(),
            retry_ordinal=1)


def test_reactivation_satisfiable_opens_no_file(monkeypatch):
    """The operator surface runs on every render, so it must be pure and
    offline -- no data fetch, no file read."""
    def _boom(*a, **k):
        raise AssertionError("reactivation_satisfiable performed file I/O")
    monkeypatch.setattr(builtins, "open", _boom)
    ok, _ = evidence.reactivation_satisfiable(
        _payload(), as_of="2026-09-03", catalog_entry=_catalog_entry(),
        retry_ordinal=1)
    assert ok is True


def test_evidence_module_imports_only_stdlib_and_constants():
    """Structural proof that nothing in this module can reach the network or
    the market-data layer: the import list itself is whitelisted."""
    path = os.path.join(_SRC, "mining", "evidence.py")
    tree = ast.parse(open(path, "r", encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # `from . import constants` -- name the members
                imported.update(a.name.split(".")[0] for a in node.names)
            else:
                imported.add((node.module or "").split(".")[0])
    assert imported <= {"hashlib", "json", "math", "datetime", "re",
                        "typing", "__future__", "constants"}


# --- validate_amendment ---------------------------------------------------

def test_amended_evidence_must_carry_an_identical_observed_block():
    """No new measurement may be laundered as a predicate amendment. An
    amendment changes the PREDICATE; changing what was observed is a new
    measurement and needs evidence_source='remeasured'."""
    prior = _payload()
    new = _payload(evidence_source="amended",
                   observed=_observed(trade_count=9),
                   supersedes={"ledger_line": 12, "line_sha256": "b" * 64,
                               "recorded_stage": "cut_dsr"})
    with pytest.raises(ValueError, match="observed"):
        evidence.validate_amendment(new, prior)


def test_amendment_may_change_the_reactivation_predicate():
    prior = _payload()
    new = _payload(evidence_source="amended",
                   reactivation=_widen(candidate_tickers=[
                       "PFF", "PGX", "PSK", "FPE", "PFFD", "PFFA"]),
                   supersedes={"ledger_line": 12, "line_sha256": "b" * 64,
                               "recorded_stage": "cut_dsr"})
    evidence.validate_amendment(new, prior)


def test_validate_amendment_refuses_a_non_amended_source():
    with pytest.raises(ValueError, match="evidence_source"):
        evidence.validate_amendment(_payload(), _payload())


def test_validate_amendment_refuses_a_missing_supersedes():
    new = _payload(evidence_source="amended")
    with pytest.raises(ValueError, match="supersedes"):
        evidence.validate_amendment(new, _payload())


# --- validate_retry_request ----------------------------------------------

def _retry_kwargs(tmp_path, **over):
    src = _write(tmp_path / "signal_fn.py", "def signal(df):\n    return 0\n")
    payload = _payload(observed=_observed(
        signal_source_sha256=evidence.source_sha256(src),
        signal_source_path=src.replace("\\", "/")))
    kwargs = {"payload": payload, "catalog_entry": _catalog_entry(),
              "tickers_used": ["PFF", "PGX", "PSK"],
              "signal_source_path": src, "retry_ordinal": 1}
    kwargs.update(over)
    return kwargs


def test_retry_accepts_a_genuine_widening(tmp_path):
    evidence.validate_retry_request(**_retry_kwargs(tmp_path))


def test_retry_of_a_historical_record_survives_a_bar_change(tmp_path,
                                                            monkeypatch):
    """The read path in its real setting: validate_retry_request re-validates
    a payload READ BACK from the ledger. A record deferred under the old bar
    must still be retryable after the bar moves -- otherwise moving the bar
    silently retires every outstanding deferral."""
    kwargs = _retry_kwargs(tmp_path)
    monkeypatch.setattr(constants, "MIN_TRADES_FOR_SCORING",
                        constants.MIN_TRADES_FOR_SCORING + 5)
    evidence.validate_retry_request(**kwargs)


def test_retry_refused_when_a_catalog_parameter_changed(tmp_path):
    """The anti-fishing guard: changing entry_zscore 0.75 -> 0.50 to
    manufacture trades is a new hypothesis and must get its own family_id
    and its own trial charge."""
    drifted = _catalog_entry(parameters=dict(_PFF_PARAMS, entry_zscore=0.50))
    with pytest.raises(evidence.ParameterDriftError, match="entry_zscore"):
        evidence.validate_retry_request(
            **_retry_kwargs(tmp_path, catalog_entry=drifted))


def test_retry_refused_when_the_signal_source_changed(tmp_path):
    """'More data' and 'different code' are different claims. The
    fingerprint survives a legitimate widening; the source hash catches a
    silent code tweak."""
    kwargs = _retry_kwargs(tmp_path)
    _write(tmp_path / "signal_fn.py", "def signal(df):\n    return 1\n")
    with pytest.raises(evidence.ParameterDriftError, match="signal_source"):
        evidence.validate_retry_request(**kwargs)


def test_widened_set_rejects_a_dropped_original_instrument(tmp_path):
    """Instrument shopping. A subset-only rule would let PFF -- the
    instrument that produced the seed result -- be silently dropped."""
    with pytest.raises(evidence.InstrumentDropError, match="PFF"):
        evidence.validate_retry_request(
            **_retry_kwargs(tmp_path, tickers_used=["PGX", "PSK", "FPE"]))


def test_retry_refused_when_tickers_used_are_not_a_widening(tmp_path):
    """Re-running the identical instrument set is bit-identical work."""
    with pytest.raises(evidence.InstrumentDropError):
        evidence.validate_retry_request(
            **_retry_kwargs(tmp_path, tickers_used=["PFF"]))


def test_retry_refused_when_tickers_used_escape_the_candidate_set(tmp_path):
    """The candidate set was pre-registered before the retry's data existed;
    reaching outside it is shopping."""
    with pytest.raises(evidence.InstrumentDropError, match="SPY"):
        evidence.validate_retry_request(
            **_retry_kwargs(tmp_path, tickers_used=["PFF", "PGX", "SPY"]))


def test_retry_allowed_at_the_cap_and_refused_one_past_it(tmp_path):
    cap = constants.MAX_EVIDENCE_RETRIES_PER_FAMILY
    evidence.validate_retry_request(**_retry_kwargs(tmp_path,
                                                    retry_ordinal=cap))
    with pytest.raises(evidence.RetryCapExceeded):
        evidence.validate_retry_request(**_retry_kwargs(tmp_path,
                                                        retry_ordinal=cap + 1))


def test_retry_refused_for_an_engine_error_payload(tmp_path):
    """An engine error re-enters ordinary runs once fixed; it has no
    predicate to satisfy and must not consume a retry."""
    with pytest.raises(ValueError, match="engine_error"):
        evidence.validate_retry_request(
            **_retry_kwargs(tmp_path, payload=_engine_payload()))


def test_retry_refused_for_a_none_available_predicate(tmp_path):
    payload = _payload(reactivation={"kind": "none_available",
                                     "rationale": "no deeper history exists"})
    with pytest.raises(ValueError, match="none_available"):
        evidence.validate_retry_request(
            **_retry_kwargs(tmp_path, payload=payload))


def test_extended_history_retry_requires_the_same_instrument_set(tmp_path):
    """A history extension is not an instrument widening -- doing both at
    once makes the result uninterpretable."""
    src = _write(tmp_path / "signal_fn.py", "def signal(df):\n    return 0\n")
    payload = _payload(
        observed=_observed(signal_source_sha256=evidence.source_sha256(src),
                           signal_source_path=src.replace("\\", "/")),
        reactivation={"kind": "extended_history",
                      "earliest_available": "1998-01-02",
                      "source": "Cohen & Steers monthly distributions",
                      "registered_on": "2026-09-02",
                      "registered_by_run_id": "run-x",
                      "rationale": "deeper history"})
    evidence.validate_retry_request(payload=payload,
                                    catalog_entry=_catalog_entry(),
                                    tickers_used=["PFF"],
                                    signal_source_path=src, retry_ordinal=1)
    with pytest.raises(evidence.InstrumentDropError):
        evidence.validate_retry_request(payload=payload,
                                        catalog_entry=_catalog_entry(),
                                        tickers_used=["PFF", "PGX"],
                                        signal_source_path=src,
                                        retry_ordinal=1)


def test_retry_validates_the_payload_itself(tmp_path):
    """A retry against a malformed record must refuse, not proceed on it."""
    kwargs = _retry_kwargs(tmp_path)
    kwargs["payload"] = dict(kwargs["payload"], schema_version=2)
    with pytest.raises(ValueError, match="schema_version"):
        evidence.validate_retry_request(**kwargs)


# --- insufficient_holdout_trades --------------------------

def test_validate_accepts_a_well_formed_insufficient_holdout_trades_payload():
    evidence.validate(_holdout_trade_payload(), stage="insufficient_evidence",
                      at_write_time=True)


def test_validate_rejects_insufficient_holdout_trades_at_the_required_count():
    """20 holdout trades is SCOREABLE. Recording it as insufficient evidence
    would hide a real holdout-gate outcome behind a deferral."""
    payload = _holdout_trade_payload(
        observed=_holdout_trade_observed(holdout_trade_count=constants.MIN_HOLDOUT_TRADES))
    with pytest.raises(ValueError, match="holdout_trade_count"):
        evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_accepts_insufficient_holdout_trades_one_below_the_required_count():
    payload = _holdout_trade_payload(
        observed=_holdout_trade_observed(
            holdout_trade_count=constants.MIN_HOLDOUT_TRADES - 1))
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_rejects_insufficient_holdout_trades_wrong_required_at_write_time():
    payload = _holdout_trade_payload(
        observed=_holdout_trade_observed(holdout_trades_required=15))
    with pytest.raises(ValueError, match="MIN_HOLDOUT_TRADES"):
        evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_insufficient_holdout_trades_does_not_require_in_sample_end_before_holdout_start():
    """Unlike the in-sample gate, this reason code's evidence describes a
    family that ALREADY cleared the in-sample gate and reached the holdout
    -- in_sample_end preceding holdout_start is still true in practice but
    is not this validator's own concern; it must not crash validating a
    payload whose in_sample_end sits later than a naive reader might
    expect a per-trade in-sample payload to allow."""
    payload = _holdout_trade_payload(
        observed=_holdout_trade_observed(in_sample_end="2026-02-27"))
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_build_insufficient_holdout_trades_computes_the_parameter_fingerprint():
    payload = evidence.build_insufficient_holdout_trades(
        in_sample_stats={"trade_count": 80, "sharpe_in_sample": 0.5,
                        "net_edge_bps": 500.0},
        holdout_stats={"trade_count": 5, "net_edge_bps": 100.0},
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        in_sample_start="2008-01-02", in_sample_end="2026-02-27",
        pinned_window=_pinned_window(), holdout_look="computed_not_scored",
        signal_source_path="runs/x/signal_fn.py",
        signal_source_sha256="a" * 64, evidence_source="measured",
        reactivation={"kind": "none_available",
                     "rationale": "the holdout window is pinned"})
    assert payload["observed"]["parameter_fingerprint"] == \
        evidence.parameter_fingerprint(_catalog_entry())
    assert payload["observed"]["holdout_trade_count"] == 5
    assert payload["observed"]["holdout_trades_required"] == \
        constants.MIN_HOLDOUT_TRADES
    assert payload["reason_code"] == "insufficient_holdout_trades"
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_build_insufficient_holdout_trades_refuses_a_scoreable_holdout_count():
    with pytest.raises(ValueError, match="SCOREABLE"):
        evidence.build_insufficient_holdout_trades(
            in_sample_stats={"trade_count": 80, "sharpe_in_sample": 0.5,
                            "net_edge_bps": 500.0},
            holdout_stats={"trade_count": constants.MIN_HOLDOUT_TRADES,
                          "net_edge_bps": 100.0},
            catalog_entry=_catalog_entry(), instrument_set=["PFF"],
            in_sample_start="2008-01-02", in_sample_end="2026-02-27",
            pinned_window=_pinned_window(), holdout_look="computed_not_scored",
            signal_source_path="x", signal_source_sha256="a" * 64,
            evidence_source="measured",
            reactivation={"kind": "none_available", "rationale": "x"})


def test_record_outcome_accepts_an_insufficient_holdout_trades_payload(tmp_path):
    from mining import trial_ledger
    ledger = str(tmp_path / "ledger.jsonl")
    payload = evidence.build_insufficient_holdout_trades(
        in_sample_stats={"trade_count": 80, "sharpe_in_sample": 0.5,
                        "net_edge_bps": 500.0},
        holdout_stats={"trade_count": 5, "net_edge_bps": 100.0},
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        in_sample_start="2008-01-02", in_sample_end="2026-02-27",
        pinned_window=_pinned_window(), holdout_look="computed_not_scored",
        signal_source_path="x", signal_source_sha256="a" * 64,
        evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "x"})
    trial_ledger.record_outcome(
        ledger, "preferred_stock_yield_spread_reversion", "run-1",
        "2026-09-08", "insufficient_evidence", evidence=payload)
    assert trial_ledger.latest_outcome(
        ledger, "preferred_stock_yield_spread_reversion") == "insufficient_evidence"


def test_cut_risk_refusal_is_a_scored_stage():
    assert "cut_risk_refusal" in evidence.SCORED_STAGES
    assert "cut_risk_refusal" not in evidence.UNSCORED_STAGES


def test_record_outcome_accepts_a_cut_risk_refusal_row(tmp_path):
    from mining import trial_ledger
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        ledger, "some_abi1_family", "run-1", "2026-09-08",
        "cut_risk_refusal", dsr_value=0.97, sharpe_in_sample=0.02,
        construction="ABI-1")
    assert trial_ledger.latest_outcome(ledger, "some_abi1_family") \
        == "cut_risk_refusal"


# --- abi1_not_scoreable / abi1_holdout_not_evaluable ------

def _abi1_not_scoreable_observed(**over):
    block = {"n_sessions": 40, "min_sessions_required": 252,
             "abi1_reason": "only 40 account sessions, below MIN_ACCOUNT_SESSIONS (252)",
             "instrument_set": ["PFF"], "account_start": "2026-07-01",
             "account_end": "2026-09-01", "cash_convention": "sweep_cash_daily",
             "benchmark": "SPY", "parameters": {},
             "parameter_fingerprint": evidence.parameter_fingerprint(_catalog_entry()),
             "signal_source_sha256": "a" * 64,
             "signal_source_path": "scripts/run_abi1_mining.py"}
    block.update(over)
    return block


def _abi1_not_scoreable_payload(**over):
    # holdout_start/end reuse the account's OWN span (account_start/account_end
    # above) -- ABI-1 has no pinned per-trade holdout window, and
    # evidence._validate_holdout requires real ISO dates unconditionally, so
    # None is not legal here.
    payload = {"schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
               "reason_code": "abi1_not_scoreable",
               "evidence_source": "measured",
               "observed": _abi1_not_scoreable_observed(),
               "holdout": {"holdout_start": "2026-07-01", "holdout_end": "2026-09-01",
                           "look": "computed_not_scored"},
               "reactivation": {"kind": "none_available",
                                "rationale": "no registered reactivation mechanism "
                                             "exists yet for an ABI-1 family that is "
                                             "short on account-session history; "
                                             "it may become "
                                             "scoreable in a FUTURE mining run once "
                                             "more sessions accrue, but that is not "
                                             "a reactivation predicate this schema "
                                             "currently models"},
               "supersedes": None}
    payload.update(over)
    return payload


def test_validate_accepts_a_well_formed_abi1_not_scoreable_payload():
    evidence.validate(_abi1_not_scoreable_payload(), stage="insufficient_evidence",
                      at_write_time=True)


def test_build_abi1_not_scoreable_computes_the_parameter_fingerprint():
    payload = evidence.build_abi1_not_scoreable(
        n_sessions=40, abi1_reason="only 40 account sessions, below "
        "MIN_ACCOUNT_SESSIONS (252)", catalog_entry=_catalog_entry(),
        instrument_set=["PFF"], account_start="2026-07-01",
        account_end="2026-09-01", cash_convention="sweep_cash_daily",
        benchmark="SPY", signal_source_path="scripts/run_abi1_mining.py",
        signal_source_sha256="a" * 64, evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "x"})
    assert payload["reason_code"] == "abi1_not_scoreable"
    assert payload["observed"]["n_sessions"] == 40
    assert payload["holdout"]["holdout_start"] == "2026-07-01"
    assert payload["holdout"]["holdout_end"] == "2026-09-01"
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def _abi1_holdout_not_evaluable_observed(**over):
    block = {"in_sample_dsr_value": 0.97, "in_sample_sharpe_daily_excess": 0.05,
             "n_in_sample_sessions": 5000, "n_holdout_sessions": 400,
             "holdout_sessions_required": 756,
             "holdout_refusal": "holdout_too_short",
             "instrument_set": ["PFF"], "account_start": "2008-01-02",
             "account_end": "2026-09-01", "cash_convention": "sweep_cash_daily",
             "benchmark": "SPY", "parameters": {},
             "parameter_fingerprint": evidence.parameter_fingerprint(_catalog_entry()),
             "signal_source_sha256": "a" * 64,
             "signal_source_path": "scripts/run_abi1_mining.py"}
    block.update(over)
    return block


def _abi1_holdout_not_evaluable_payload(**over):
    # holdout_start/end here are the REAL dates of the attempted holdout tail
    # (the caller derives these from account_relative's own account_daily
    # index). Falling back to (pinned_window or {}).get(...)
    # -- the pattern the sibling per-trade builders use -- is NOT safe here
    # because ABI-1 has no guaranteed pinned window and that pattern
    # produces None, which _validate_holdout rejects.
    payload = {"schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
               "reason_code": "abi1_holdout_not_evaluable",
               "evidence_source": "measured",
               "observed": _abi1_holdout_not_evaluable_observed(),
               "holdout": {"holdout_start": "2025-01-01", "holdout_end": "2026-09-01",
                           "look": "computed_not_scored"},
               "reactivation": {"kind": "none_available",
                                "rationale": "no registered reactivation mechanism "
                                             "exists yet for an ABI-1 family whose "
                                             "holdout could not be evaluated"},
               "supersedes": None}
    payload.update(over)
    return payload


def test_validate_accepts_a_well_formed_abi1_holdout_not_evaluable_payload():
    evidence.validate(_abi1_holdout_not_evaluable_payload(),
                      stage="insufficient_evidence", at_write_time=True)


def test_validate_rejects_abi1_holdout_not_evaluable_wrong_required_at_write_time():
    """The sibling insufficient_holdout_trades pins holdout_trades_required to
    constants.MIN_HOLDOUT_TRADES at write time so a stale hand-typed value
    can't land in a new record -- this reason code's holdout_sessions_required
    must get the same guard against constants.ABI1_HOLDOUT_SESSIONS."""
    payload = _abi1_holdout_not_evaluable_payload(
        observed=_abi1_holdout_not_evaluable_observed(holdout_sessions_required=200))
    with pytest.raises(ValueError, match="ABI1_HOLDOUT_SESSIONS"):
        evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_accepts_abi1_holdout_not_evaluable_wrong_required_off_write_time():
    payload = _abi1_holdout_not_evaluable_payload(
        observed=_abi1_holdout_not_evaluable_observed(holdout_sessions_required=200))
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=False)


def test_build_abi1_holdout_not_evaluable_carries_the_real_in_sample_dsr():
    payload = evidence.build_abi1_holdout_not_evaluable(
        in_sample_dsr_value=0.97, in_sample_sharpe_daily_excess=0.05,
        n_in_sample_sessions=5000, n_holdout_sessions=400,
        holdout_sessions_required=756, holdout_refusal="holdout_too_short",
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        account_start="2008-01-02", account_end="2026-09-01",
        pinned_window={"holdout_start": "2025-01-01", "holdout_end": "2026-09-01"},
        cash_convention="sweep_cash_daily", benchmark="SPY",
        signal_source_path="scripts/run_abi1_mining.py",
        signal_source_sha256="a" * 64, evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "x"})
    assert payload["reason_code"] == "abi1_holdout_not_evaluable"
    assert payload["observed"]["in_sample_dsr_value"] == 0.97
    assert payload["holdout"]["holdout_start"] == "2025-01-01"
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_build_abi1_holdout_not_evaluable_requires_a_real_pinned_window():
    """pinned_window is NOT Optional -- an ABI-1 caller must always derive
    real dates from account_relative's own account_daily index;
    there is no safe None-tolerant fallback the way the per-trade builders
    have ((pinned_window or {}).get(...) silently produces
    None, which the shared _validate_holdout rejects anyway, just later and
    less clearly than a direct TypeError here)."""
    with pytest.raises(TypeError):
        evidence.build_abi1_holdout_not_evaluable(
            in_sample_dsr_value=0.97, in_sample_sharpe_daily_excess=0.05,
            n_in_sample_sessions=5000, n_holdout_sessions=400,
            holdout_sessions_required=756, holdout_refusal="holdout_too_short",
            catalog_entry=_catalog_entry(), instrument_set=["PFF"],
            account_start="2008-01-02", account_end="2026-09-01",
            cash_convention="sweep_cash_daily", benchmark="SPY",
            signal_source_path="x", signal_source_sha256="a" * 64,
            evidence_source="measured",
            reactivation={"kind": "none_available", "rationale": "x"})
            # pinned_window omitted entirely -- required positional-or-keyword


def test_record_outcome_accepts_an_abi1_holdout_not_evaluable_payload(tmp_path):
    from mining import trial_ledger
    ledger = str(tmp_path / "ledger.jsonl")
    payload = evidence.build_abi1_holdout_not_evaluable(
        in_sample_dsr_value=0.97, in_sample_sharpe_daily_excess=0.05,
        n_in_sample_sessions=5000, n_holdout_sessions=400,
        holdout_sessions_required=756, holdout_refusal="holdout_too_short",
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        account_start="2008-01-02", account_end="2026-09-01",
        pinned_window={"holdout_start": "2025-01-01", "holdout_end": "2026-09-01"},
        cash_convention="sweep_cash_daily", benchmark="SPY",
        signal_source_path="x", signal_source_sha256="a" * 64,
        evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "x"})
    trial_ledger.record_outcome(
        ledger, "some_abi1_family", "run-1", "2026-09-08",
        "insufficient_evidence", evidence=payload, construction="ABI-1")
    assert trial_ledger.latest_outcome(ledger, "some_abi1_family") \
        == "insufficient_evidence"


# --- cost_realism_gate_excluded --
#
# A candidate rejected by intraday_cost_gate.evaluate_cost_realism_gate
# BEFORE the DSR/holdout step runs needs its own reason code: without one it
# would read as "eligible -- not yet tested" instead of "tested and
# rejected". It follows the same insufficient_evidence pattern as
# abi1_not_scoreable/abi1_holdout_not_evaluable above.

def _cost_realism_gate_excluded_observed(**over):
    block = {"trade_count": 300, "net_edge_bps": -10.0,
             "sharpe_in_sample": -0.1,
             "spread_bps": 2.0,
             "edge_over_cost_ratio": -5.0,
             "edge_over_cost_multiple_required": constants.INTRADAY_EDGE_OVER_COST_MULTIPLE,
             "spread_ceiling_bps": constants.INTRADAY_SPREAD_CEILING_BPS,
             "in_sample_start": "2020-01-01", "in_sample_end": "2026-02-27",
             "instrument_set": ["SPY", "IWM", "DIA"],
             "parameters": dict(_catalog_entry(
                 family_id="synthetic_intraday_family")["parameters"]),
             "parameter_fingerprint": evidence.parameter_fingerprint(
                 _catalog_entry(family_id="synthetic_intraday_family")),
             "signal_source_sha256": "a" * 64,
             "signal_source_path": "src/mining/swing_signals.py"}
    block.update(over)
    return block


def _cost_realism_gate_excluded_payload(**over):
    # holdout_start/holdout_end reuse the RUN'S own operative window (the run
    # report's top-level holdout_start/holdout_end) even though this family
    # was never pinned via record_first_test -- a gate exclusion is not a "tested" outcome
    # (run_swing_mining.py's own module docstring), so there is no
    # per-family pinned window of record; the run's own window is the only
    # real, disclosed one available, and _validate_holdout requires real
    # ISO dates unconditionally (same as the ABI-1
    # builders above). look="unspent" because the cost-realism gate runs
    # BEFORE any holdout fetch/evaluation is ever attempted.
    payload = {"schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
               "reason_code": "cost_realism_gate_excluded",
               "evidence_source": "measured",
               "observed": _cost_realism_gate_excluded_observed(),
               "holdout": {"holdout_start": "2026-02-28",
                           "holdout_end": "2026-08-31",
                           "look": "unspent"},
               "reactivation": {"kind": "none_available",
                                "rationale": "no reactivation predicate exists "
                                             "yet for a cost-realism gate "
                                             "exclusion -- a live spread "
                                             "tightening is not one of this "
                                             "schema's modeled "
                                             "REACTIVATION_KINDS"},
               "supersedes": None}
    payload.update(over)
    return payload


def test_validate_accepts_a_well_formed_cost_realism_gate_excluded_payload():
    evidence.validate(_cost_realism_gate_excluded_payload(),
                      stage="insufficient_evidence", at_write_time=True)


def test_cost_realism_gate_excluded_is_in_reason_codes():
    assert "cost_realism_gate_excluded" in evidence.REASON_CODES


def test_validate_rejects_cost_realism_gate_excluded_wrong_multiple_at_write_time():
    """Same self-describing-at-write-time guard as observed.trades_required
    (_validate_trade_observed) and observed.holdout_trades_required
    (_validate_holdout_trade_observed): a stale hand-typed constant must
    not land in a NEW record, but an old record stays readable if the bar
    later moves."""
    payload = _cost_realism_gate_excluded_payload(
        observed=_cost_realism_gate_excluded_observed(
            edge_over_cost_multiple_required=5.0))
    with pytest.raises(ValueError, match="INTRADAY_EDGE_OVER_COST_MULTIPLE"):
        evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_accepts_cost_realism_gate_excluded_wrong_multiple_off_write_time():
    payload = _cost_realism_gate_excluded_payload(
        observed=_cost_realism_gate_excluded_observed(
            edge_over_cost_multiple_required=5.0))
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=False)


def test_validate_rejects_cost_realism_gate_excluded_wrong_ceiling_at_write_time():
    payload = _cost_realism_gate_excluded_payload(
        observed=_cost_realism_gate_excluded_observed(spread_ceiling_bps=50.0))
    with pytest.raises(ValueError, match="INTRADAY_SPREAD_CEILING_BPS"):
        evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_rejects_cost_realism_gate_excluded_when_ratio_actually_clears_the_gate():
    """A record claiming this reason code while its own numbers say the
    candidate would have PASSED the gate (ratio >= the required multiple
    AND spread under the ceiling) is exactly the mislabeling this module
    exists to prevent -- mirrors build_insufficient_trades' own
    trade_count >= trades_required guard."""
    payload = _cost_realism_gate_excluded_payload(
        observed=_cost_realism_gate_excluded_observed(
            net_edge_bps=4.0, edge_over_cost_ratio=4.0, spread_bps=1.0))
    with pytest.raises(ValueError, match="clears"):
        evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_rejects_cost_realism_gate_excluded_fabricated_ratio():
    """observed.edge_over_cost_ratio must actually equal
    observed.net_edge_bps / observed.spread_bps (the formula
    intraday_cost_gate.evaluate_cost_realism_gate itself computes it with)
    -- a payload whose net_edge_bps/spread_bps pair would clear the gate
    (100 bps edge, 1 bps spread, far past a real multiple requirement) but
    whose own edge_over_cost_ratio field is fabricated to a value that
    reads as a failure (0.0) must be rejected. Without this check, the
    'clears the gate' guard immediately above only inspects the
    (unverified) ratio field, so a caller passing a self-inconsistent
    ratio can mislabel a real pass as an exclusion."""
    payload = _cost_realism_gate_excluded_payload(
        observed=_cost_realism_gate_excluded_observed(
            net_edge_bps=100.0, spread_bps=1.0, edge_over_cost_ratio=0.0))
    with pytest.raises(ValueError, match="edge_over_cost_ratio"):
        evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_validate_rejects_a_near_miss_reason_code_singular_cost_realism():
    with pytest.raises(ValueError, match="reason_code"):
        evidence.validate(_cost_realism_gate_excluded_payload(
            reason_code="cost_realism_gate_exclude"),
            stage="insufficient_evidence")


def test_build_cost_realism_gate_excluded_computes_the_parameter_fingerprint():
    catalog_entry = _catalog_entry(family_id="synthetic_intraday_family",
                                   tickers=("SPY", "IWM", "DIA"),
                                   parameters={"anchor_lookback_days": 20})
    payload = evidence.build_cost_realism_gate_excluded(
        stats={"trade_count": 300, "net_edge_bps": -10.0,
               "sharpe_in_sample": -0.1},
        spread_bps=2.0,
        edge_over_cost_ratio=-5.0,
        catalog_entry=catalog_entry, instrument_set=["SPY", "IWM", "DIA"],
        in_sample_start="2020-01-01", in_sample_end="2026-02-27",
        pinned_window={"holdout_start": "2026-02-28", "holdout_end": "2026-08-31"},
        holdout_look="unspent",
        signal_source_path="src/mining/swing_signals.py",
        signal_source_sha256="a" * 64, evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "x"})
    assert payload["reason_code"] == "cost_realism_gate_excluded"
    assert payload["observed"]["trade_count"] == 300
    assert payload["observed"]["parameter_fingerprint"] == \
        evidence.parameter_fingerprint(catalog_entry)
    assert payload["observed"]["edge_over_cost_multiple_required"] == \
        constants.INTRADAY_EDGE_OVER_COST_MULTIPLE
    assert payload["observed"]["spread_ceiling_bps"] == \
        constants.INTRADAY_SPREAD_CEILING_BPS
    assert payload["holdout"]["holdout_start"] == "2026-02-28"
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_build_cost_realism_gate_excluded_refuses_a_hand_typed_reason_code():
    """Matches build_insufficient_trades' own refusal shape: reason_code is
    ALWAYS 'cost_realism_gate_excluded' for this builder, never a caller
    parameter -- there is nothing to choose between."""
    with pytest.raises(TypeError):
        evidence.build_cost_realism_gate_excluded(
            reason_code="cost_realism_gate_excluded",
            stats={"trade_count": 300, "net_edge_bps": -10.0},
            spread_bps=2.0, edge_over_cost_ratio=-5.0,
            catalog_entry=_catalog_entry(), instrument_set=["SPY"],
            in_sample_start="2020-01-01", in_sample_end="2026-02-27",
            pinned_window={"holdout_start": "2026-02-28", "holdout_end": "2026-08-31"},
            holdout_look="unspent", signal_source_path="x",
            signal_source_sha256="a" * 64, evidence_source="measured",
            reactivation={"kind": "none_available", "rationale": "x"})


def test_build_cost_realism_gate_excluded_requires_trade_count_in_stats():
    with pytest.raises(ValueError, match="trade_count"):
        evidence.build_cost_realism_gate_excluded(
            stats={"net_edge_bps": -10.0}, spread_bps=2.0,
            edge_over_cost_ratio=-5.0, catalog_entry=_catalog_entry(),
            instrument_set=["SPY"], in_sample_start="2020-01-01",
            in_sample_end="2026-02-27",
            pinned_window={"holdout_start": "2026-02-28", "holdout_end": "2026-08-31"},
            holdout_look="unspent", signal_source_path="x",
            signal_source_sha256="a" * 64, evidence_source="measured",
            reactivation={"kind": "none_available", "rationale": "x"})


def test_build_cost_realism_gate_excluded_requires_net_edge_bps_in_stats():
    with pytest.raises(ValueError, match="net_edge_bps"):
        evidence.build_cost_realism_gate_excluded(
            stats={"trade_count": 300}, spread_bps=2.0,
            edge_over_cost_ratio=-5.0, catalog_entry=_catalog_entry(),
            instrument_set=["SPY"], in_sample_start="2020-01-01",
            in_sample_end="2026-02-27",
            pinned_window={"holdout_start": "2026-02-28", "holdout_end": "2026-08-31"},
            holdout_look="unspent", signal_source_path="x",
            signal_source_sha256="a" * 64, evidence_source="measured",
            reactivation={"kind": "none_available", "rationale": "x"})


def test_record_outcome_accepts_a_cost_realism_gate_excluded_payload(tmp_path):
    from mining import trial_ledger
    ledger = str(tmp_path / "ledger.jsonl")
    payload = evidence.build_cost_realism_gate_excluded(
        stats={"trade_count": 300, "net_edge_bps": -10.0,
               "sharpe_in_sample": -0.1},
        spread_bps=2.0,
        edge_over_cost_ratio=-5.0,
        catalog_entry=_catalog_entry(family_id="synthetic_intraday_family",
                                     tickers=("SPY", "IWM", "DIA")),
        instrument_set=["SPY", "IWM", "DIA"],
        in_sample_start="2020-01-01", in_sample_end="2026-02-27",
        pinned_window={"holdout_start": "2026-02-28", "holdout_end": "2026-08-31"},
        holdout_look="unspent",
        signal_source_path="src/mining/swing_signals.py",
        signal_source_sha256="a" * 64, evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "x"})
    trial_ledger.record_outcome(
        ledger, "synthetic_intraday_family", "run-1",
        "2026-03-01", "insufficient_evidence",
        note="net edge -10.0 bps is below the required multiple of the round-trip spread",
        ticker_label="SPY/IWM/DIA", evidence=payload, construction="per_trade")
    assert trial_ledger.latest_outcome(
        ledger, "synthetic_intraday_family") == "insufficient_evidence"
