"""The semantic core of "we could not judge this" -- the outcome states a
mining run must use when NO Deflated Sharpe Ratio was ever computed.

WHY THIS EXISTS. A trial ledger that records "too few trades to score" and
"scored and genuinely rejected" as the SAME record (stage="cut_dsr",
dsr_value=null) lets part of the multiple-testing floor that every future
candidate pays come from families that were never actually judged, with
nothing in the record saying why. Where several scoring implementations
write outcomes, a reason that each call site must remember to pass through
is easily lost, so the cause is made STRUCTURALLY MANDATORY instead: an
unscored outcome cannot be written without a payload this module validated.

A second misleading record shape gets its own state: a defensive outcome
recorder that filed a caught PYTHON EXCEPTION as stage="cut_dsr",
dsr_value=None would record a mechanical failure permanently as a
statistical rejection about the market. That is `not_evaluated_error`, and
splitting it out is not optional: with the biconditional below in force, a
recorder that left that path on "cut_dsr" would raise ValueError inside its
own exception handler and mislabel a whole batch.

THE CONTRACT this module exists to make enforceable, checked by the ledger on
every write, PURE and callable outside its lock:

    stage in SCORED_STAGES    <=>  dsr_value is a finite float AND evidence is None
    stage in UNSCORED_STAGES  <=>  dsr_value is None           AND evidence is a valid payload

WHY THERE IS NO TIME-BASED REACTIVATION LEVER (the counterintuitive part).
In scripts/run_batch3_mining.py's `_window_masks`, in-sample is
`index < holdout_start` and the holdout is `[holdout_start, holdout_end]`;
NOTHING reads data after
holdout_end. Because trial_ledger.record_first_test pins that window
permanently and idempotently, waiting for calendar time can never add an
in-sample trade to an already-pinned family -- a retry a year later is
bit-identical. "Come back later" is therefore not a retry at all; it is a new
hypothesis, needing a new family_id and its own trial charge. The only levers
that can physically work are widening the instrument set and extending
history backwards, and the vocabulary below contains exactly those two plus
an honest terminal "none_available".

Nothing here reads a file except `source_sha256` (which must, and hashes raw
bytes), and nothing here reads the clock: every operator surface takes an
explicit `as_of` so its output is reproducible and its tests deterministic.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from typing import Optional

from . import constants

# Format version stamped into every payload so a future schema change is
# DETECTABLE rather than inferred from which fields happen to be present.
# Deliberately not in constants.py: that file is the register of
# pre-registered decision thresholds, and a format version is neither a
# threshold nor a value whose change can alter a statistical result.
EVIDENCE_SCHEMA_VERSION = 1

# The stage vocabulary has ONE definition -- here -- so the copies scattered
# across the scoring scripts import it rather than retype it.
# "cut_risk_refusal" is for the ABI-1 construction: a family
# whose in-sample DSR clears the bar (a real dsr_value exists) but fails
# one of account_relative.risk_refusals' four checks BEFORE the holdout is
# ever evaluated. It is SCORED, not deferred -- the layer-1 statistic is
# real -- and it is its own stage rather than reusing cut_holdout, because
# "the account cleared the statistical bar but an investor would not have
# held it" is a DIFFERENT claim than "the holdout confirmation failed",
# and a reader must be able to tell them apart from the ledger alone.
SCORED_STAGES = ("cut_dsr", "cut_holdout", "survived", "cut_risk_refusal")
UNSCORED_STAGES = ("insufficient_evidence", "not_evaluated_error")
VALID_STAGES = SCORED_STAGES + UNSCORED_STAGES

# zero_trades is kept distinct from insufficient_trades because zero is
# qualitatively different from a small sample: it is the case
# dsr.num_independent_trials raises on by design (its empty/constant
# return-series check), and it
# carries no Sharpe to report at all.
#
# "cost_realism_gate_excluded" covers families rejected by
# intraday_cost_gate.evaluate_cost_realism_gate before the DSR/holdout step
# ever ran; without a reason code they would never be written to the
# ledger and would read as "eligible -- not yet tested" instead of "tested
# and rejected". This is its own state, not a reuse of insufficient_trades:
# the candidate already cleared MIN_TRADES_FOR_SCORING and had a real
# in-sample net_edge_bps -- what failed is a real captured bid/ask
# spread making that edge un-executable, a market-cost fact, not a sample-
# size fact.
REASON_CODES = ("insufficient_trades", "zero_trades", "engine_error",
                "insufficient_holdout_trades", "abi1_not_scoreable",
                "abi1_holdout_not_evaluable", "cost_realism_gate_excluded")

EVIDENCE_SOURCES = ("measured", "remeasured", "amended")

# Disclosure only -- nothing branches on it. The two mining paths genuinely
# differ: the batch-mining path fetches the holdout only
# inside `if passed_dsr:` so a trade-count cut never looks at it, while
# run_batch3_mining.py's `compute_candidate` computes holdout_net_edge_bps
# unconditionally for every item, before the min-trades gate. Recording which happened lets a
# reader see whether the number was ever computed; it gates nothing, because
# no recorded outcome ever depended on it and the only "cure" -- a fresh
# window -- is precisely the shopping the pin exists to prevent.
HOLDOUT_LOOKS = ("unspent", "computed_not_scored")

REACTIVATION_KINDS = ("widened_instrument_set", "extended_history",
                      "none_available")

# A pooled retry across ETFs that still list in 2026 cannot see funds that
# closed, delisted or merged, and DSR does not correct that bias. No
# survivorship-free database is available here, so the bias is forced to be
# VISIBLE rather than fixed.
SURVIVORSHIP_LABELS = ("survivor_only_disclosed", "point_in_time_verified")

_TOP_LEVEL_KEYS = ("schema_version", "reason_code", "evidence_source",
                   "observed", "holdout", "reactivation", "supersedes")
_TRADE_OBSERVED_KEYS = ("trade_count", "trades_required", "in_sample_start",
                        "in_sample_end", "instrument_set", "sharpe_in_sample",
                        "net_edge_bps", "parameters", "parameter_fingerprint",
                        "signal_source_sha256", "signal_source_path")
_ERROR_OBSERVED_KEYS = ("error_type", "error_message", "instrument_set")
# Deliberately NOT a reuse of _TRADE_OBSERVED_KEYS: that set's trade_count/
# trades_required are the IN-SAMPLE gate's own vocabulary, and its
# write-time check pins them to constants.MIN_TRADES_FOR_SCORING -- wrong
# constant for a holdout-count gate. A family reaching this reason code
# has ALREADY cleared the in-sample gate (a real DSR exists), so this key
# set carries both: the real in-sample figures (informational, read-side
# only -- record_outcome's own top-level sharpe_in_sample/dsr_value stay
# None on this UNSCORED stage, exactly like insufficient_trades' own
# nested sharpe_in_sample) and the holdout figures that triggered the
# deferral.
_HOLDOUT_TRADE_OBSERVED_KEYS = (
    "in_sample_trade_count", "holdout_trade_count", "holdout_trades_required",
    "in_sample_start", "in_sample_end", "instrument_set", "sharpe_in_sample",
    "net_edge_bps", "holdout_net_edge_bps", "parameters",
    "parameter_fingerprint", "signal_source_sha256", "signal_source_path")

# ABI-1's two UNSCORED reason codes carry SESSION vocabulary, never trade
# vocabulary -- account_relative has no per-family trade-count gate of its
# own (MIN_ACCOUNT_SESSIONS governs session count, not trades). Prefixed
# "abi1_" in REASON_CODES (not here, where the field names already say so)
# so a reader scanning REASON_CODES cannot confuse them with the per-trade
# insufficient_trades/insufficient_holdout_trades codes above.
_ABI1_NOT_SCOREABLE_OBSERVED_KEYS = (
    "n_sessions", "min_sessions_required", "abi1_reason", "instrument_set",
    "account_start", "account_end", "cash_convention", "benchmark",
    "parameters", "parameter_fingerprint", "signal_source_sha256",
    "signal_source_path")

# Carries the REAL in-sample DSR (layer 1 passed) nested here, never at the
# top level -- record_outcome's own biconditional forbids a top-level
# dsr_value/sharpe_in_sample on an UNSCORED stage. A family reaching this
# code cleared the statistical bar; only the out-of-sample confirmation
# could not run.
_ABI1_HOLDOUT_NOT_EVALUABLE_OBSERVED_KEYS = (
    "in_sample_dsr_value", "in_sample_sharpe_daily_excess",
    "n_in_sample_sessions", "n_holdout_sessions", "holdout_sessions_required",
    "holdout_refusal", "instrument_set", "account_start", "account_end",
    "cash_convention", "benchmark", "parameters", "parameter_fingerprint",
    "signal_source_sha256", "signal_source_path")

# cost_realism_gate_excluded's own vocabulary -- deliberately NOT a reuse of
# _TRADE_OBSERVED_KEYS: that set's trades_required/its write-time check pin
# to constants.MIN_TRADES_FOR_SCORING, the wrong constant here (a family
# reaching this reason code already CLEARED that bar; what rejected it is
# the separate cost-realism gate). `edge_over_cost_multiple_required` and
# `spread_ceiling_bps` follow trades_required's own established convention
# (_validate_trade_observed's comment): recorded at write time so an old
# record stays self-describing if either bar moves later, rather
# than every historical record going unreadable the instant it does.
_COST_REALISM_GATE_EXCLUDED_OBSERVED_KEYS = (
    "trade_count", "net_edge_bps", "sharpe_in_sample", "spread_bps",
    "edge_over_cost_ratio", "edge_over_cost_multiple_required",
    "spread_ceiling_bps", "in_sample_start", "in_sample_end",
    "instrument_set", "parameters", "parameter_fingerprint",
    "signal_source_sha256", "signal_source_path")

_HOLDOUT_KEYS = ("holdout_start", "holdout_end", "look")
_SUPERSEDES_KEYS = ("ledger_line", "line_sha256", "recorded_stage")
_UNIVERSE_SELECTION_KEYS = ("rule", "as_of", "survivorship_bias")
_REACTIVATION_KEYS = {
    "widened_instrument_set": ("kind", "candidate_tickers",
                               "universe_selection", "registered_on",
                               "registered_by_run_id", "rationale"),
    "extended_history": ("kind", "earliest_available", "source",
                         "registered_on", "registered_by_run_id",
                         "rationale"),
    "none_available": ("kind", "rationale"),
}

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TICKER_RE = re.compile(r"^[A-Z0-9.^\-]+$")

_DAYS_PER_YEAR = 365.25


class ParameterDriftError(ValueError):
    """The hypothesis under retry is not the hypothesis that was deferred --
    a catalog parameter or the signal function's own source changed. That is
    a NEW hypothesis and needs its own family_id and its own trial charge."""


class InstrumentDropError(ValueError):
    """A retry's instrument set is not a real widening of the deferred one:
    it drops an original instrument, adds nothing, or reaches outside the
    pre-registered candidate list."""


class RetryCapExceeded(ValueError):
    """This family has already had its lifetime allowance of evidence
    retries (constants.MAX_EVIDENCE_RETRIES_PER_FAMILY)."""


# --- small, shared checkers ------------------------------------------------

def _require_keys(block, keys, where: str) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"{where} must be a dict (one predicate, never a "
                         f"menu); got {type(block).__name__}")
    missing = sorted(k for k in keys if k not in block)
    unexpected = sorted(k for k in block if k not in keys)
    if missing:
        raise ValueError(f"{where} is missing required key(s): {missing}")
    if unexpected:
        raise ValueError(f"{where} has unexpected key(s): {unexpected}")


def _date(value, field: str) -> datetime.date:
    if not isinstance(value, str) or not _ISO_DATE_RE.match(value):
        raise ValueError(f"{field} must be an ISO date 'YYYY-MM-DD'; "
                         f"got {value!r}")
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a real calendar date: "
                         f"{value!r} ({exc})") from exc


def _int(value, field: str) -> int:
    # isinstance(True, int) is True in Python, so bool must be excluded
    # explicitly or True silently reads as the count 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int; got {value!r}")
    return value


def _optional_float(value, field: str) -> Optional[float]:
    """None means "not computed" -- an honest, readable statement. NaN/inf
    mean nothing a reader can act on, so they are refused."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a float or None; got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite (None means 'not "
                         f"computed'); got {value!r}")
    return float(value)


def _nonempty_str(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string; got {value!r}")
    return value


def _sha256_hex(value, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValueError(f"{field} must be 64 lowercase hex characters; "
                         f"got {value!r}")
    return value


def _tickers(value, field: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of tickers; "
                         f"got {value!r}")
    for t in value:
        if not isinstance(t, str) or not t or not _TICKER_RE.match(t):
            raise ValueError(f"{field} entries must be non-empty uppercase "
                             f"ticker strings; got {t!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} contains duplicate tickers: {value!r}")
    return value


def _reject_newlines(value, path: str) -> None:
    """The ledger is JSON Lines: one record per line. json.dumps would escape
    an embedded newline rather than corrupt the file, but every grep/sed/awk
    reader of the ledger -- and the run reports built from it -- treats a
    line as a record. Free-text prose has no need of one."""
    if isinstance(value, str):
        if "\n" in value or "\r" in value:
            raise ValueError(f"{path} contains a newline; evidence payloads "
                             f"must serialize to exactly one line")
    elif isinstance(value, dict):
        for k, v in value.items():
            _reject_newlines(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _reject_newlines(v, f"{path}[{i}]")


# --- validation ------------------------------------------------------------

def validate(payload: dict, *, stage: str, at_write_time: bool = False) -> None:
    """Raise ValueError naming the exact offending field unless `payload` is
    a well-formed evidence payload for `stage`.

    PURE: no file, no network, no clock. The trial ledger calls this BEFORE
    taking its write lock (`_LedgerLock` is non-reentrant per its own
    docstring), so this function must never touch the ledger itself.

    `at_write_time` defaults to False because this function is on the READ
    path as much as the write path -- `validate_retry_request` runs it on a
    payload read back out of the ledger. The one check that cannot be applied
    to a read is `observed.trades_required == constants.MIN_TRADES_FOR_SCORING`
    (the field is recorded precisely so an old record stays
    self-describing after the bar moves), and demanding today's
    constant on every call delivers the opposite -- every historical record
    becomes unreadable the instant the bar moves, and since `validate_amendment`
    requires an amendment's `observed` block to match the superseded one
    byte-for-byte, such a record could then be neither retried nor amended.
    Pass at_write_time=True from a write path to keep a stale hand-typed value
    from landing in a NEW record; the read path checks shape and internal
    consistency instead, which is all a reader can honestly assert.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"unknown stage: {stage!r}; expected one of "
                         f"{list(VALID_STAGES)}")
    if stage in SCORED_STAGES:
        raise ValueError(f"stage {stage!r} is a SCORED outcome and must not "
                         f"carry an evidence payload -- an evidence payload "
                         f"means no DSR was computed")
    if not isinstance(payload, dict):
        raise ValueError(f"evidence payload must be a dict; "
                         f"got {type(payload).__name__}")

    _require_keys(payload, _TOP_LEVEL_KEYS, "evidence payload")
    _reject_newlines(payload, "evidence")

    if payload["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EVIDENCE_SCHEMA_VERSION}; "
                         f"got {payload['schema_version']!r}")

    reason_code = payload["reason_code"]
    if reason_code not in REASON_CODES:
        raise ValueError(f"reason_code must be one of {list(REASON_CODES)}; "
                         f"got {reason_code!r}")

    # The reason_code <-> stage biconditional, both directions. A mechanical
    # failure must never be filed as a market fact, and a small sample must
    # never be filed as a crash.
    if reason_code == "engine_error" and stage != "not_evaluated_error":
        raise ValueError(f"reason_code 'engine_error' requires stage "
                         f"'not_evaluated_error'; got stage {stage!r}")
    if reason_code != "engine_error" and stage == "not_evaluated_error":
        raise ValueError(f"stage 'not_evaluated_error' requires reason_code "
                         f"'engine_error'; got {reason_code!r}")

    evidence_source = payload["evidence_source"]
    if evidence_source not in EVIDENCE_SOURCES:
        raise ValueError(f"evidence_source must be one of "
                         f"{list(EVIDENCE_SOURCES)}; got {evidence_source!r}")

    holdout_start = _validate_holdout(payload["holdout"])
    if reason_code == "engine_error":
        _validate_error_observed(payload["observed"])
    elif reason_code == "insufficient_holdout_trades":
        _validate_holdout_trade_observed(payload["observed"], at_write_time)
    elif reason_code == "abi1_not_scoreable":
        _validate_abi1_not_scoreable_observed(payload["observed"])
    elif reason_code == "abi1_holdout_not_evaluable":
        _validate_abi1_holdout_not_evaluable_observed(payload["observed"], at_write_time)
    elif reason_code == "cost_realism_gate_excluded":
        _validate_cost_realism_gate_excluded_observed(payload["observed"], at_write_time)
    else:
        _validate_trade_observed(payload["observed"], reason_code,
                                 holdout_start, at_write_time)
    _validate_reactivation(payload["reactivation"], payload["observed"],
                           reason_code)
    _validate_supersedes(payload["supersedes"], evidence_source)


def _validate_holdout(holdout) -> datetime.date:
    _require_keys(holdout, _HOLDOUT_KEYS, "holdout")
    start = _date(holdout["holdout_start"], "holdout.holdout_start")
    end = _date(holdout["holdout_end"], "holdout.holdout_end")
    if end < start:
        raise ValueError(f"holdout.holdout_end ({end}) precedes "
                         f"holdout.holdout_start ({start})")
    if holdout["look"] not in HOLDOUT_LOOKS:
        raise ValueError(f"holdout.look must be one of {list(HOLDOUT_LOOKS)}; "
                         f"got {holdout['look']!r}")
    return start


def _validate_trade_observed(observed, reason_code,
                             holdout_start: datetime.date,
                             at_write_time: bool = False) -> None:
    _require_keys(observed, _TRADE_OBSERVED_KEYS, "observed")
    trade_count = _int(observed["trade_count"], "observed.trade_count")
    trades_required = _int(observed["trades_required"],
                           "observed.trades_required")
    # Recording the constant IN FORCE at write time makes an old record
    # self-describing if the bar changes later, without that change
    # retroactively reinterpreting history -- so the equality below is a
    # WRITE-time check only. On a read, the honest assertions are that the
    # recorded bar is a usable number and that the record is internally
    # consistent with it (checked by the reason_code rules just below); see
    # `validate`'s own docstring for why applying the equality to a read
    # would freeze every historical record the moment the bar moves.
    if trades_required <= 0:
        raise ValueError(f"observed.trades_required must be a positive int "
                         f"(it is the bar this record was judged against); "
                         f"got {trades_required}")
    if at_write_time and trades_required != constants.MIN_TRADES_FOR_SCORING:
        raise ValueError(
            f"observed.trades_required must equal the constant in force at "
            f"write time (constants.MIN_TRADES_FOR_SCORING = "
            f"{constants.MIN_TRADES_FOR_SCORING}); got {trades_required}")
    if trade_count < 0:
        raise ValueError(f"observed.trade_count must be >= 0; "
                         f"got {trade_count}")
    if reason_code == "zero_trades" and trade_count != 0:
        raise ValueError(f"reason_code 'zero_trades' requires "
                         f"observed.trade_count == 0; got {trade_count}")
    if reason_code == "insufficient_trades" and not 0 < trade_count < trades_required:
        raise ValueError(
            f"reason_code 'insufficient_trades' requires 0 < "
            f"observed.trade_count < {trades_required}; got {trade_count}. "
            f"Zero trades is 'zero_trades'; {trades_required} or more is "
            f"SCOREABLE and must be scored, not deferred")

    in_start = _date(observed["in_sample_start"], "observed.in_sample_start")
    in_end = _date(observed["in_sample_end"], "observed.in_sample_end")
    if not in_start < in_end:
        raise ValueError(f"observed.in_sample_start ({in_start}) must precede "
                         f"observed.in_sample_end ({in_end})")
    # in-sample is `index < holdout_start` (run_batch3_mining._window_masks), so
    # an in-sample bar at or past the boundary is a look-ahead leak.
    if not in_end < holdout_start:
        raise ValueError(f"observed.in_sample_end ({in_end}) must precede "
                         f"holdout.holdout_start ({holdout_start}) -- "
                         f"in-sample is `index < holdout_start`")

    _tickers(observed["instrument_set"], "observed.instrument_set")
    _optional_float(observed["sharpe_in_sample"], "observed.sharpe_in_sample")
    _optional_float(observed["net_edge_bps"], "observed.net_edge_bps")
    if not isinstance(observed["parameters"], dict):
        raise ValueError(f"observed.parameters must be a dict copied verbatim "
                         f"from the catalog entry; "
                         f"got {observed['parameters']!r}")
    _sha256_hex(observed["parameter_fingerprint"],
                "observed.parameter_fingerprint")
    _sha256_hex(observed["signal_source_sha256"],
                "observed.signal_source_sha256")
    _nonempty_str(observed["signal_source_path"],
                  "observed.signal_source_path")


def _validate_holdout_trade_observed(observed, at_write_time: bool = False) -> None:
    _require_keys(observed, _HOLDOUT_TRADE_OBSERVED_KEYS, "observed")
    in_sample_trade_count = _int(observed["in_sample_trade_count"],
                                 "observed.in_sample_trade_count")
    if in_sample_trade_count < constants.MIN_TRADES_FOR_SCORING:
        raise ValueError(
            f"observed.in_sample_trade_count ({in_sample_trade_count}) is "
            f"below MIN_TRADES_FOR_SCORING -- this family never cleared the "
            f"in-sample gate, so it belongs to reason_code "
            f"'insufficient_trades'/'zero_trades', not "
            f"'insufficient_holdout_trades'")
    holdout_trade_count = _int(observed["holdout_trade_count"],
                               "observed.holdout_trade_count")
    holdout_trades_required = _int(observed["holdout_trades_required"],
                                   "observed.holdout_trades_required")
    if holdout_trades_required <= 0:
        raise ValueError(f"observed.holdout_trades_required must be a "
                         f"positive int; got {holdout_trades_required}")
    if at_write_time and holdout_trades_required != constants.MIN_HOLDOUT_TRADES:
        raise ValueError(
            f"observed.holdout_trades_required must equal the constant in "
            f"force at write time (constants.MIN_HOLDOUT_TRADES = "
            f"{constants.MIN_HOLDOUT_TRADES}); got {holdout_trades_required}")
    if holdout_trade_count < 0:
        raise ValueError(f"observed.holdout_trade_count must be >= 0; "
                         f"got {holdout_trade_count}")
    if holdout_trade_count >= holdout_trades_required:
        raise ValueError(
            f"observed.holdout_trade_count ({holdout_trade_count}) is >= "
            f"holdout_trades_required ({holdout_trades_required}): this "
            f"family's holdout is SCOREABLE by the trade-count floor and "
            f"must go through backtest_engine.holdout_passes, not be "
            f"recorded as insufficient evidence")

    in_start = _date(observed["in_sample_start"], "observed.in_sample_start")
    in_end = _date(observed["in_sample_end"], "observed.in_sample_end")
    if not in_start < in_end:
        raise ValueError(f"observed.in_sample_start ({in_start}) must "
                         f"precede observed.in_sample_end ({in_end})")

    _tickers(observed["instrument_set"], "observed.instrument_set")
    _optional_float(observed["sharpe_in_sample"], "observed.sharpe_in_sample")
    _optional_float(observed["net_edge_bps"], "observed.net_edge_bps")
    _optional_float(observed["holdout_net_edge_bps"],
                    "observed.holdout_net_edge_bps")
    if not isinstance(observed["parameters"], dict):
        raise ValueError(f"observed.parameters must be a dict copied "
                         f"verbatim from the catalog entry; "
                         f"got {observed['parameters']!r}")
    _sha256_hex(observed["parameter_fingerprint"],
                "observed.parameter_fingerprint")
    _sha256_hex(observed["signal_source_sha256"],
                "observed.signal_source_sha256")
    _nonempty_str(observed["signal_source_path"],
                  "observed.signal_source_path")


def _validate_abi1_not_scoreable_observed(observed) -> None:
    _require_keys(observed, _ABI1_NOT_SCOREABLE_OBSERVED_KEYS, "observed")
    n_sessions = _int(observed["n_sessions"], "observed.n_sessions")
    if n_sessions < 0:
        raise ValueError(f"observed.n_sessions must be >= 0; got {n_sessions}")
    min_required = _int(observed["min_sessions_required"],
                        "observed.min_sessions_required")
    if min_required <= 0:
        raise ValueError("observed.min_sessions_required must be positive")
    if n_sessions >= min_required:
        raise ValueError(
            f"observed.n_sessions ({n_sessions}) is >= "
            f"min_sessions_required ({min_required}): this family has "
            f"enough sessions to be scoreable and must not be recorded as "
            f"abi1_not_scoreable on that ground -- if sr_variance was None "
            f"instead, that is still a real not_scoreable condition, but "
            f"n_sessions alone does not justify it here")
    _nonempty_str(observed["abi1_reason"], "observed.abi1_reason")
    _tickers(observed["instrument_set"], "observed.instrument_set")
    _date(observed["account_start"], "observed.account_start")
    _date(observed["account_end"], "observed.account_end")
    _nonempty_str(observed["cash_convention"], "observed.cash_convention")
    _nonempty_str(observed["benchmark"], "observed.benchmark")
    if not isinstance(observed["parameters"], dict):
        raise ValueError("observed.parameters must be a dict")
    _sha256_hex(observed["parameter_fingerprint"], "observed.parameter_fingerprint")
    _sha256_hex(observed["signal_source_sha256"], "observed.signal_source_sha256")
    _nonempty_str(observed["signal_source_path"], "observed.signal_source_path")


_ABI1_HOLDOUT_REFUSALS = ("holdout_too_short", "in_sample_too_short",
                          "non_finite_excess", "zero_variance")


def _validate_abi1_holdout_not_evaluable_observed(observed, at_write_time: bool = False) -> None:
    _require_keys(observed, _ABI1_HOLDOUT_NOT_EVALUABLE_OBSERVED_KEYS, "observed")
    _optional_float(observed["in_sample_dsr_value"], "observed.in_sample_dsr_value")
    if observed["in_sample_dsr_value"] is None:
        raise ValueError(
            "observed.in_sample_dsr_value must be a real number -- this "
            "reason code means layer 1 PASSED and only the holdout could "
            "not run; a None here belongs to abi1_not_scoreable instead")
    _optional_float(observed["in_sample_sharpe_daily_excess"],
                    "observed.in_sample_sharpe_daily_excess")
    n_in_sample = _int(observed["n_in_sample_sessions"], "observed.n_in_sample_sessions")
    n_holdout = _int(observed["n_holdout_sessions"], "observed.n_holdout_sessions")
    required = _int(observed["holdout_sessions_required"],
                    "observed.holdout_sessions_required")
    if n_in_sample < 0:
        raise ValueError(f"observed.n_in_sample_sessions must be >= 0; got {n_in_sample}")
    if n_holdout < 0:
        raise ValueError(f"observed.n_holdout_sessions must be >= 0; got {n_holdout}")
    if required <= 0:
        raise ValueError(f"observed.holdout_sessions_required must be positive; got {required}")
    if at_write_time and required != constants.ABI1_HOLDOUT_SESSIONS:
        raise ValueError(
            f"observed.holdout_sessions_required must equal the constant in "
            f"force at write time (constants.ABI1_HOLDOUT_SESSIONS = "
            f"{constants.ABI1_HOLDOUT_SESSIONS}); got {required}")
    refusal = observed["holdout_refusal"]
    if refusal not in _ABI1_HOLDOUT_REFUSALS:
        raise ValueError(f"observed.holdout_refusal must be one of "
                         f"{list(_ABI1_HOLDOUT_REFUSALS)}; got {refusal!r}")
    _tickers(observed["instrument_set"], "observed.instrument_set")
    _date(observed["account_start"], "observed.account_start")
    _date(observed["account_end"], "observed.account_end")
    _nonempty_str(observed["cash_convention"], "observed.cash_convention")
    _nonempty_str(observed["benchmark"], "observed.benchmark")
    if not isinstance(observed["parameters"], dict):
        raise ValueError("observed.parameters must be a dict")
    _sha256_hex(observed["parameter_fingerprint"], "observed.parameter_fingerprint")
    _sha256_hex(observed["signal_source_sha256"], "observed.signal_source_sha256")
    _nonempty_str(observed["signal_source_path"], "observed.signal_source_path")


def _finite_number(value, field: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number; got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return value


def _validate_cost_realism_gate_excluded_observed(observed, at_write_time: bool = False) -> None:
    _require_keys(observed, _COST_REALISM_GATE_EXCLUDED_OBSERVED_KEYS, "observed")
    trade_count = _int(observed["trade_count"], "observed.trade_count")
    if trade_count < 0:
        raise ValueError(f"observed.trade_count must be >= 0; got {trade_count}")

    net_edge_bps = _finite_number(observed["net_edge_bps"], "observed.net_edge_bps")
    _optional_float(observed["sharpe_in_sample"], "observed.sharpe_in_sample")

    spread_bps = _finite_number(observed["spread_bps"], "observed.spread_bps")
    if spread_bps < 0:
        raise ValueError(f"observed.spread_bps must be >= 0; got {spread_bps}")

    edge_over_cost_ratio = _finite_number(observed["edge_over_cost_ratio"],
                                          "observed.edge_over_cost_ratio")

    # edge_over_cost_ratio is a DERIVED field (intraday_cost_gate.
    # evaluate_cost_realism_gate computes it as net_edge_bps / spread_bps,
    # never accepts it as an independent input) -- a payload could
    # otherwise pass a self-inconsistent ratio (e.g. a fabricated 0.0)
    # alongside real net_edge_bps/spread_bps values that would actually
    # clear the gate, defeating the "clears the gate" guard below, which
    # only inspects the ratio field (unverified without this check). This
    # is unreachable when spread_bps == 0 (edge_over_cost_ratio would be
    # +/-inf/NaN there, already excluded by the finite-number check above
    # -- a separate, disclosed gap, not this one) so this check only
    # applies for spread_bps > 0.
    if spread_bps > 0:
        expected_ratio = net_edge_bps / spread_bps
        if not math.isclose(edge_over_cost_ratio, expected_ratio,
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"observed.edge_over_cost_ratio ({edge_over_cost_ratio}) does "
                f"not equal observed.net_edge_bps / observed.spread_bps "
                f"({net_edge_bps} / {spread_bps} = {expected_ratio}) -- this "
                f"field must be derived from those two, never independently "
                f"supplied")

    multiple_required = _finite_number(observed["edge_over_cost_multiple_required"],
                                       "observed.edge_over_cost_multiple_required")
    if multiple_required <= 0:
        raise ValueError(f"observed.edge_over_cost_multiple_required must be "
                         f"positive; got {multiple_required}")
    if (at_write_time and
            multiple_required != constants.INTRADAY_EDGE_OVER_COST_MULTIPLE):
        raise ValueError(
            f"observed.edge_over_cost_multiple_required must equal the "
            f"constant in force at write time "
            f"(constants.INTRADAY_EDGE_OVER_COST_MULTIPLE = "
            f"{constants.INTRADAY_EDGE_OVER_COST_MULTIPLE}); got "
            f"{multiple_required}")

    spread_ceiling_bps = _finite_number(observed["spread_ceiling_bps"],
                                        "observed.spread_ceiling_bps")
    if spread_ceiling_bps <= 0:
        raise ValueError(f"observed.spread_ceiling_bps must be positive; "
                         f"got {spread_ceiling_bps}")
    if (at_write_time and
            spread_ceiling_bps != constants.INTRADAY_SPREAD_CEILING_BPS):
        raise ValueError(
            f"observed.spread_ceiling_bps must equal the constant in force "
            f"at write time (constants.INTRADAY_SPREAD_CEILING_BPS = "
            f"{constants.INTRADAY_SPREAD_CEILING_BPS}); got "
            f"{spread_ceiling_bps}")

    # The reason this record exists at all: the gate's own pass condition
    # (intraday_cost_gate.evaluate_cost_realism_gate) must NOT be
    # satisfied. A record claiming this reason code while its own numbers
    # say the candidate would have cleared both checks is exactly the
    # mislabeling evidence.py's module docstring exists to prevent --
    # mirrors build_insufficient_trades' own trade_count >= trades_required
    # guard.
    if edge_over_cost_ratio >= multiple_required and spread_bps < spread_ceiling_bps:
        raise ValueError(
            f"observed.edge_over_cost_ratio ({edge_over_cost_ratio}) clears "
            f"observed.edge_over_cost_multiple_required "
            f"({multiple_required}) and observed.spread_bps "
            f"({spread_bps}) is below observed.spread_ceiling_bps "
            f"({spread_ceiling_bps}): this candidate clears the "
            f"cost-realism gate and must not be recorded as excluded by it")

    in_start = _date(observed["in_sample_start"], "observed.in_sample_start")
    in_end = _date(observed["in_sample_end"], "observed.in_sample_end")
    if not in_start < in_end:
        raise ValueError(f"observed.in_sample_start ({in_start}) must "
                         f"precede observed.in_sample_end ({in_end})")

    _tickers(observed["instrument_set"], "observed.instrument_set")
    if not isinstance(observed["parameters"], dict):
        raise ValueError(f"observed.parameters must be a dict copied "
                         f"verbatim from the catalog entry; "
                         f"got {observed['parameters']!r}")
    _sha256_hex(observed["parameter_fingerprint"],
                "observed.parameter_fingerprint")
    _sha256_hex(observed["signal_source_sha256"],
                "observed.signal_source_sha256")
    _nonempty_str(observed["signal_source_path"],
                  "observed.signal_source_path")


def _validate_error_observed(observed) -> None:
    _require_keys(observed, _ERROR_OBSERVED_KEYS, "observed")
    _nonempty_str(observed["error_type"], "observed.error_type")
    if not isinstance(observed["error_message"], str):
        raise ValueError(f"observed.error_message must be a string; "
                         f"got {observed['error_message']!r}")
    _tickers(observed["instrument_set"], "observed.instrument_set")


def _validate_reactivation(reactivation, observed, reason_code) -> None:
    if isinstance(reactivation, list):
        raise ValueError("reactivation must be a dict holding exactly ONE "
                         "predicate, never a list -- a disjunction is a "
                         "weaker commitment and invites taking whichever "
                         "predicate fires first")
    if not isinstance(reactivation, dict):
        raise ValueError(f"reactivation must be a dict; "
                         f"got {type(reactivation).__name__}")
    kind = reactivation.get("kind")
    if kind not in REACTIVATION_KINDS:
        raise ValueError(f"reactivation.kind must be one of "
                         f"{list(REACTIVATION_KINDS)}; got {kind!r}. There is "
                         f"deliberately NO time-based kind: the holdout "
                         f"window is pinned, so waiting adds no in-sample "
                         f"data")
    _require_keys(reactivation, _REACTIVATION_KEYS[kind], "reactivation")
    _nonempty_str(reactivation["rationale"], "reactivation.rationale")

    # An engine error is a DEFECT, not evidence about the market: the family
    # stays eligible and re-enters ordinary runs once the bug is fixed, so no
    # reactivation predicate applies to it.
    if reason_code == "engine_error" and kind != "none_available":
        raise ValueError(f"reason_code 'engine_error' requires "
                         f"reactivation.kind 'none_available' (a bug is not a "
                         f"research dead end -- the family stays eligible); "
                         f"got {kind!r}")

    if kind == "none_available":
        return

    _date(reactivation["registered_on"], "reactivation.registered_on")
    _nonempty_str(reactivation["registered_by_run_id"],
                  "reactivation.registered_by_run_id")

    if kind == "widened_instrument_set":
        candidates = _tickers(reactivation["candidate_tickers"],
                              "reactivation.candidate_tickers")
        observed_set = set(observed.get("instrument_set") or [])
        if not observed_set < set(candidates):
            raise ValueError(
                f"reactivation.candidate_tickers {sorted(candidates)} must be "
                f"a STRICT superset of observed.instrument_set "
                f"{sorted(observed_set)}: dropping or swapping the instrument "
                f"that produced the seed result is shopping, and an identical "
                f"set is a bit-identical re-run")
        _validate_universe_selection(reactivation["universe_selection"])
    elif kind == "extended_history":
        earliest = _date(reactivation["earliest_available"],
                         "reactivation.earliest_available")
        in_start = _date(observed["in_sample_start"],
                         "observed.in_sample_start")
        if not earliest < in_start:
            raise ValueError(
                f"reactivation.earliest_available ({earliest}) must be "
                f"strictly earlier than observed.in_sample_start ({in_start}) "
                f"-- otherwise there is no additional history to add")
        _nonempty_str(reactivation["source"], "reactivation.source")


def _validate_universe_selection(block) -> None:
    _require_keys(block, _UNIVERSE_SELECTION_KEYS,
                  "reactivation.universe_selection")
    _nonempty_str(block["rule"], "reactivation.universe_selection.rule")
    _date(block["as_of"], "reactivation.universe_selection.as_of")
    if block["survivorship_bias"] not in SURVIVORSHIP_LABELS:
        raise ValueError(
            f"reactivation.universe_selection.survivorship_bias must be one "
            f"of {list(SURVIVORSHIP_LABELS)}; got "
            f"{block['survivorship_bias']!r}")


def _validate_supersedes(supersedes, evidence_source) -> None:
    if supersedes is None:
        if evidence_source == "amended":
            raise ValueError("evidence_source 'amended' requires a "
                             "'supersedes' block naming the record being "
                             "amended")
        return
    _require_keys(supersedes, _SUPERSEDES_KEYS, "supersedes")
    _int(supersedes["ledger_line"], "supersedes.ledger_line")
    _sha256_hex(supersedes["line_sha256"], "supersedes.line_sha256")
    if supersedes["recorded_stage"] not in VALID_STAGES:
        raise ValueError(f"supersedes.recorded_stage must be one of "
                         f"{list(VALID_STAGES)}; got "
                         f"{supersedes['recorded_stage']!r}")


def validate_amendment(new_payload: dict, prior_payload: dict) -> None:
    """Raise unless `new_payload` is a legitimate amendment of
    `prior_payload`.

    Changing your mind about a reactivation predicate is allowed and is
    recorded by APPENDING a new record whose `supersedes` names the old one
    (the ledger is append-only and folds last-wins, so the append IS the
    amendment). What is not allowed is laundering a NEW MEASUREMENT as an
    amendment -- so the observed block must match byte-for-byte. A genuine
    re-measurement is evidence_source 'remeasured', which pays a retry.

    This is the only place both records are in hand.
    """
    if not isinstance(new_payload, dict) or not isinstance(prior_payload, dict):
        raise ValueError("validate_amendment takes two evidence payload dicts")
    if new_payload.get("evidence_source") != "amended":
        raise ValueError(f"validate_amendment applies only to "
                         f"evidence_source 'amended'; got "
                         f"{new_payload.get('evidence_source')!r}")
    if new_payload.get("supersedes") is None:
        raise ValueError("an amendment requires a 'supersedes' block naming "
                         "the record it supersedes")
    if new_payload.get("observed") != prior_payload.get("observed"):
        raise ValueError(
            "an amendment must carry an 'observed' block identical to the "
            "record it supersedes -- a changed measurement is a REMEASUREMENT "
            "(evidence_source 'remeasured'), not an amendment")


# --- identity -------------------------------------------------------------

def parameter_fingerprint(catalog_entry: dict) -> str:
    """sha256 over canonical JSON of {family_id, parameters}.

    TICKERS ARE DELIBERATELY EXCLUDED. Widening the instrument set is the
    sanctioned reactivation lever, so it must not invalidate the hypothesis's
    identity; changing a parameter (entry_zscore 0.75 -> 0.50 to manufacture
    trades) IS a new hypothesis and must get its own family_id and its own
    trial charge.
    """
    if not isinstance(catalog_entry, dict):
        raise ValueError(f"catalog_entry must be a dict; "
                         f"got {type(catalog_entry).__name__}")
    for field in ("family_id", "parameters"):
        if field not in catalog_entry:
            raise ValueError(f"catalog_entry is missing {field!r}")
    blob = json.dumps({"family_id": catalog_entry["family_id"],
                       "parameters": catalog_entry["parameters"]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def source_sha256(path: str) -> str:
    """sha256 of the file's RAW BYTES.

    Opened 'rb' on purpose: a text-mode read applies universal-newline
    translation, so a CRLF and an LF copy of the same signal function would
    hash identically and a line-ending change would move the artifact's bytes
    after it was hashed.

    The one edge in this module that touches the filesystem; every other
    function is pure.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --- constructors ---------------------------------------------------------

def build_insufficient_trades(*, stats: dict, catalog_entry: dict,
                              instrument_set, in_sample_start: str,
                              in_sample_end: str, pinned_window: dict,
                              holdout_look: str, signal_source_path: str,
                              signal_source_sha256: str,
                              evidence_source: str, reactivation: dict,
                              supersedes: Optional[dict] = None) -> dict:
    """Build the payload for a candidate that was never scored because it did
    not produce enough in-sample trades.

    Every DERIVED field is computed here and is NOT a parameter:
    `reason_code` (chosen from the trade count), `trades_required` (read from
    the constant in force), and `parameter_fingerprint` (hashed from the
    catalog entry). A value derived from other recorded data but typed by
    hand rots silently, so passing any of them is a TypeError.

    `signal_source_sha256` IS a parameter because hashing is file I/O and
    this function is pure -- the caller runs `source_sha256` at the edge.
    """
    if not isinstance(stats, dict) or "trade_count" not in stats:
        raise ValueError("stats must be a dict carrying 'trade_count' -- a "
                         "deferral whose trade count is unknown is exactly "
                         "the ambiguity this record exists to remove")
    trade_count = _int(stats["trade_count"], "stats['trade_count']")
    trades_required = constants.MIN_TRADES_FOR_SCORING
    if trade_count >= trades_required:
        raise ValueError(
            f"trade_count {trade_count} is >= MIN_TRADES_FOR_SCORING "
            f"({trades_required}): this candidate is SCOREABLE and must be "
            f"scored, not recorded as insufficient evidence")
    reason_code = "zero_trades" if trade_count == 0 else "insufficient_trades"

    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "reason_code": reason_code,
        "evidence_source": evidence_source,
        "observed": {
            "trade_count": trade_count,
            "trades_required": trades_required,
            "in_sample_start": in_sample_start,
            "in_sample_end": in_sample_end,
            "instrument_set": list(instrument_set or []),
            "sharpe_in_sample": stats.get("sharpe_in_sample"),
            "net_edge_bps": stats.get("net_edge_bps"),
            "parameters": dict(catalog_entry.get("parameters") or {}),
            "parameter_fingerprint": parameter_fingerprint(catalog_entry),
            "signal_source_sha256": signal_source_sha256,
            "signal_source_path": signal_source_path,
        },
        "holdout": {
            "holdout_start": (pinned_window or {}).get("holdout_start"),
            "holdout_end": (pinned_window or {}).get("holdout_end"),
            "look": holdout_look,
        },
        "reactivation": reactivation,
        "supersedes": supersedes,
    }
    # at_write_time=True is belt-and-braces here and CANNOT currently fail:
    # trades_required is read off the constant above, so the two are
    # the same value by construction. It is passed anyway so that a future
    # refactor accepting the value as a parameter is caught by this
    # constructor's own validation instead of landing silently.
    validate(payload, stage="insufficient_evidence", at_write_time=True)
    return payload


def build_insufficient_holdout_trades(
        *, in_sample_stats: dict, holdout_stats: dict, catalog_entry: dict,
        instrument_set, in_sample_start: str, in_sample_end: str,
        pinned_window: dict, holdout_look: str, signal_source_path: str,
        signal_source_sha256: str, evidence_source: str, reactivation: dict,
        supersedes: Optional[dict] = None) -> dict:
    """Build the payload for a family that CLEARED the in-sample DSR bar
    but whose holdout has too few trades for backtest_engine.holdout_passes'
    bare sign check to be trusted (see MIN_HOLDOUT_TRADES).

    `in_sample_stats` is the real, already-computed in-sample stats dict
    (must carry 'trade_count' and 'sharpe_in_sample' -- the same keys
    _score_one already builds via its own stats_for_evidence translation).
    `holdout_stats` must carry 'trade_count' and 'net_edge_bps' (the same
    shape run_batch3_mining._trade_stats/pooling.pooled_trade_stats/
    intraday_holdout.intraday_trade_stats all already return).

    Every DERIVED field is computed here, matching build_insufficient_
    trades' own convention: holdout_trades_required is read from the
    constant in force, parameter_fingerprint is hashed from the catalog
    entry. The in-sample trade count is NOT accepted as a bare int -- it
    must come from in_sample_stats so it stays coupled to the same dict
    the caller already has, rather than a second hand-typed number that
    could drift from it.
    """
    if not isinstance(holdout_stats, dict) or "trade_count" not in holdout_stats:
        raise ValueError(
            "holdout_stats must be a dict carrying 'trade_count' -- a "
            "deferral whose holdout trade count is unknown is exactly the "
            "ambiguity this record exists to remove")
    holdout_trade_count = _int(holdout_stats["trade_count"],
                               "holdout_stats['trade_count']")
    holdout_trades_required = constants.MIN_HOLDOUT_TRADES
    if holdout_trade_count >= holdout_trades_required:
        raise ValueError(
            f"holdout trade_count {holdout_trade_count} is >= "
            f"MIN_HOLDOUT_TRADES ({holdout_trades_required}): this "
            f"family's holdout is SCOREABLE and must go through "
            f"backtest_engine.holdout_passes, not be recorded as "
            f"insufficient evidence")
    if not isinstance(in_sample_stats, dict) or "sharpe_in_sample" not in in_sample_stats:
        raise ValueError(
            "in_sample_stats must be a dict carrying 'sharpe_in_sample' -- "
            "this family already cleared the in-sample gate, and that real "
            "figure must be recorded, not omitted")
    if "trade_count" not in in_sample_stats:
        raise ValueError(
            "in_sample_stats must carry 'trade_count' -- the real in-sample "
            "count this family cleared the gate on must be recorded, not "
            "omitted")

    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "reason_code": "insufficient_holdout_trades",
        "evidence_source": evidence_source,
        "observed": {
            "in_sample_trade_count": _int(
                in_sample_stats["trade_count"], "in_sample_stats['trade_count']"),
            "holdout_trade_count": holdout_trade_count,
            "holdout_trades_required": holdout_trades_required,
            "in_sample_start": in_sample_start,
            "in_sample_end": in_sample_end,
            "instrument_set": list(instrument_set or []),
            "sharpe_in_sample": in_sample_stats.get("sharpe_in_sample"),
            "net_edge_bps": in_sample_stats.get("net_edge_bps"),
            "holdout_net_edge_bps": holdout_stats.get("net_edge_bps"),
            "parameters": dict(catalog_entry.get("parameters") or {}),
            "parameter_fingerprint": parameter_fingerprint(catalog_entry),
            "signal_source_sha256": signal_source_sha256,
            "signal_source_path": signal_source_path,
        },
        "holdout": {
            "holdout_start": (pinned_window or {}).get("holdout_start"),
            "holdout_end": (pinned_window or {}).get("holdout_end"),
            "look": holdout_look,
        },
        "reactivation": reactivation,
        "supersedes": supersedes,
    }
    validate(payload, stage="insufficient_evidence", at_write_time=True)
    return payload


def build_abi1_not_scoreable(
        *, n_sessions: int, abi1_reason: str, catalog_entry: dict,
        instrument_set, account_start: str, account_end: str,
        cash_convention: str, benchmark: str, signal_source_path: str,
        signal_source_sha256: str, evidence_source: str, reactivation: dict,
        supersedes: Optional[dict] = None,
        min_sessions_required: int = 252) -> dict:
    """account_relative.evaluate() returned verdict == "not_scoreable":
    layer 1 (score_account) itself refused -- too few account sessions, or
    (in principle) a missing sr_variance, though this driver always passes
    the fixed constants.DAILY_SR_VARIANCE so that branch is currently
    unreachable in production. `min_sessions_required` defaults to 252
    (account_relative.MIN_ACCOUNT_SESSIONS) as a literal rather than an
    import, matching this module's existing no-cross-import-from-mining-
    submodules convention (evidence.py imports only `constants`).

    `account_start`/`account_end` double as the holdout block's dates:
    there is no separate holdout sub-window for a family
    that was never scoreable at all, and `_validate_holdout` requires real
    ISO dates unconditionally -- reusing the family's own scored span is
    honest (it names the ONLY window this evidence record is about) rather
    than inventing a null convention `_validate_holdout` does not support."""
    if n_sessions >= min_sessions_required:
        raise ValueError(
            f"n_sessions {n_sessions} is >= min_sessions_required "
            f"({min_sessions_required}): this family is SCOREABLE by "
            f"session count and must not be recorded as not_scoreable "
            f"on that ground")
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "reason_code": "abi1_not_scoreable",
        "evidence_source": evidence_source,
        "observed": {
            "n_sessions": _int(n_sessions, "n_sessions"),
            "min_sessions_required": _int(min_sessions_required,
                                          "min_sessions_required"),
            "abi1_reason": abi1_reason,
            "instrument_set": list(instrument_set or []),
            "account_start": account_start, "account_end": account_end,
            "cash_convention": cash_convention, "benchmark": benchmark,
            "parameters": dict(catalog_entry.get("parameters") or {}),
            "parameter_fingerprint": parameter_fingerprint(catalog_entry),
            "signal_source_sha256": signal_source_sha256,
            "signal_source_path": signal_source_path,
        },
        "holdout": {"holdout_start": account_start, "holdout_end": account_end,
                   "look": "computed_not_scored"},
        "reactivation": reactivation,
        "supersedes": supersedes,
    }
    validate(payload, stage="insufficient_evidence", at_write_time=True)
    return payload


def build_abi1_holdout_not_evaluable(
        *, in_sample_dsr_value: float, in_sample_sharpe_daily_excess: float,
        n_in_sample_sessions: int, n_holdout_sessions: int,
        holdout_sessions_required: int, holdout_refusal: str,
        catalog_entry: dict, instrument_set, account_start: str,
        account_end: str, pinned_window: dict, cash_convention: str,
        benchmark: str, signal_source_path: str, signal_source_sha256: str,
        evidence_source: str, reactivation: dict,
        supersedes: Optional[dict] = None) -> dict:
    """account_relative.evaluate() returned verdict == "holdout_not_evaluable":
    layer 1 PASSED (a real dsr_value exists, carried here nested rather than
    at record_outcome's top level -- this is still an UNSCORED stage) and
    layer 2's four risk refusals also passed, but split_holdout/holdout_gate
    could not evaluate the out-of-sample confirmation. `holdout_refusal` is
    account_relative's own refusal code (holdout_too_short/
    in_sample_too_short from split_holdout, or non_finite_excess/
    zero_variance from holdout_gate) -- carried verbatim so a reader can see
    which of the four actually fired.

    `pinned_window` is REQUIRED (no `Optional[dict] = None` with a silent
    `(pinned_window or {}).get(...)` fallback to None, which
    `_validate_holdout` rejects unconditionally). The caller must always
    supply real dates -- the scoring driver derives them from
    `result["account_daily"].index`: the tail `n_holdout_sessions` dates
    when `split_holdout` itself succeeded but `holdout_gate` refused (a
    real holdout sub-window exists, just could not be evaluated), or the
    full account span when `split_holdout` itself refused (no holdout
    sub-window exists at all)."""
    if not isinstance(pinned_window, dict) or not pinned_window.get(
            "holdout_start") or not pinned_window.get("holdout_end"):
        raise TypeError(
            "pinned_window must be a dict with real 'holdout_start'/"
            "'holdout_end' ISO date strings -- there is no safe None "
            "fallback for this reason code")
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "reason_code": "abi1_holdout_not_evaluable",
        "evidence_source": evidence_source,
        "observed": {
            "in_sample_dsr_value": float(in_sample_dsr_value),
            "in_sample_sharpe_daily_excess": float(in_sample_sharpe_daily_excess),
            "n_in_sample_sessions": _int(n_in_sample_sessions, "n_in_sample_sessions"),
            "n_holdout_sessions": _int(n_holdout_sessions, "n_holdout_sessions"),
            "holdout_sessions_required": _int(holdout_sessions_required,
                                              "holdout_sessions_required"),
            "holdout_refusal": holdout_refusal,
            "instrument_set": list(instrument_set or []),
            "account_start": account_start, "account_end": account_end,
            "cash_convention": cash_convention, "benchmark": benchmark,
            "parameters": dict(catalog_entry.get("parameters") or {}),
            "parameter_fingerprint": parameter_fingerprint(catalog_entry),
            "signal_source_sha256": signal_source_sha256,
            "signal_source_path": signal_source_path,
        },
        "holdout": {
            "holdout_start": pinned_window["holdout_start"],
            "holdout_end": pinned_window["holdout_end"],
            "look": "computed_not_scored",
        },
        "reactivation": reactivation,
        "supersedes": supersedes,
    }
    validate(payload, stage="insufficient_evidence", at_write_time=True)
    return payload


def build_cost_realism_gate_excluded(
        *, stats: dict, spread_bps: float, edge_over_cost_ratio: float,
        catalog_entry: dict, instrument_set, in_sample_start: str,
        in_sample_end: str, pinned_window: dict, holdout_look: str,
        signal_source_path: str, signal_source_sha256: str,
        evidence_source: str, reactivation: dict,
        supersedes: Optional[dict] = None) -> dict:
    """Build the payload for a candidate that was never scored because
    `mining.intraday_cost_gate.evaluate_cost_realism_gate` rejected it
    BEFORE the DSR/holdout step ever ran: its measured in-sample edge
    could not clear a real, live-captured bid/ask spread by the
    pre-registered `constants.INTRADAY_EDGE_OVER_COST_MULTIPLE` margin, or
    the spread itself was at/above `constants.INTRADAY_SPREAD_CEILING_BPS`.

    Every DERIVED field is computed here and is NOT a parameter, matching
    `build_insufficient_trades`' own convention:
    `edge_over_cost_multiple_required`/`spread_ceiling_bps` are read from
    the constants in force at write time -- the same "self-describing if the
    bar moves" reasoning `trades_required` already
    follows. `spread_bps`/`edge_over_cost_ratio` ARE parameters -- they
    are the gate's own measured output (`evaluate_cost_realism_gate`'s
    real return values), not something this function can derive from
    `stats` alone.

    `signal_source_sha256` IS a parameter because hashing is file I/O and
    this function is pure -- the caller runs `source_sha256` at the edge,
    matching every other builder in this module.
    """
    if not isinstance(stats, dict) or "trade_count" not in stats:
        raise ValueError(
            "stats must be a dict carrying 'trade_count' -- a gate "
            "exclusion whose in-sample trade count is unknown is exactly "
            "the ambiguity this record exists to remove")
    if "net_edge_bps" not in stats:
        raise ValueError(
            "stats must carry 'net_edge_bps' -- the real in-sample edge "
            "the cost-realism gate measured against must be recorded, not "
            "omitted")
    trade_count = _int(stats["trade_count"], "stats['trade_count']")
    net_edge_bps = _finite_number(stats["net_edge_bps"], "stats['net_edge_bps']")

    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "reason_code": "cost_realism_gate_excluded",
        "evidence_source": evidence_source,
        "observed": {
            "trade_count": trade_count,
            "net_edge_bps": float(net_edge_bps),
            "sharpe_in_sample": stats.get("sharpe_in_sample"),
            "spread_bps": spread_bps,
            "edge_over_cost_ratio": edge_over_cost_ratio,
            "edge_over_cost_multiple_required":
                constants.INTRADAY_EDGE_OVER_COST_MULTIPLE,
            "spread_ceiling_bps": constants.INTRADAY_SPREAD_CEILING_BPS,
            "in_sample_start": in_sample_start,
            "in_sample_end": in_sample_end,
            "instrument_set": list(instrument_set or []),
            "parameters": dict(catalog_entry.get("parameters") or {}),
            "parameter_fingerprint": parameter_fingerprint(catalog_entry),
            "signal_source_sha256": signal_source_sha256,
            "signal_source_path": signal_source_path,
        },
        "holdout": {
            "holdout_start": (pinned_window or {}).get("holdout_start"),
            "holdout_end": (pinned_window or {}).get("holdout_end"),
            "look": holdout_look,
        },
        "reactivation": reactivation,
        "supersedes": supersedes,
    }
    validate(payload, stage="insufficient_evidence", at_write_time=True)
    return payload


def build_engine_error(*, error: BaseException, instrument_set,
                       pinned_window: dict, holdout_look: str,
                       supersedes: Optional[dict] = None) -> dict:
    """Build the payload for a candidate that was never evaluated because the
    engine raised.

    `reactivation.kind` is fixed at 'none_available' and is not a parameter:
    a bug is not a research dead end, so there is no predicate to register.
    The family stays ELIGIBLE and re-enters ordinary runs once the defect is
    fixed -- which is why this state, unlike insufficient_evidence, must not
    exclude anything.

    The message is newline-collapsed rather than truncated: a traceback-
    bearing exception must still fit the one-line ledger record, and dropping
    characters would be dropping the detail.
    """
    if not isinstance(error, BaseException):
        raise ValueError(f"error must be an exception instance; "
                         f"got {error!r}")
    # Split on ALL THREE terminators, not just the two an author on Windows
    # thinks of. `_reject_newlines` refuses a lone \r exactly as it refuses
    # \n, and a bare \r is what progress-bar output carries (this repo's data
    # path is yfinance) -- so a collapse that missed it would raise ValueError
    # inside the very handler whose job is to record the crash, escape
    # run_batch3_mining.py's `_record_defensive_outcomes` loop, and skip the
    # ledger write for every remaining family in the batch. \r\n is matched
    # FIRST so one break yields one separator -- though
    # that ordering is redundant with the `if part.strip()` filter below,
    # which drops the empty segment either way; it is written for the reader,
    # not as the guard.
    message = " | ".join(part.strip() for part in
                         re.split(r"\r\n|\r|\n", str(error))
                         if part.strip())
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "reason_code": "engine_error",
        "evidence_source": "measured",
        "observed": {
            "error_type": type(error).__name__,
            "error_message": message,
            "instrument_set": list(instrument_set or []),
        },
        "holdout": {
            "holdout_start": (pinned_window or {}).get("holdout_start"),
            "holdout_end": (pinned_window or {}).get("holdout_end"),
            "look": holdout_look,
        },
        "reactivation": {
            "kind": "none_available",
            "rationale": "engine error is a defect, not evidence about the "
                         "market: this family stays eligible and re-enters "
                         "ordinary runs once the defect is fixed",
        },
        "supersedes": supersedes,
    }
    validate(payload, stage="not_evaluated_error")
    return payload


# --- operator surface (display only; gates nothing) -----------------------

def years_to_threshold(observed: dict) -> Optional[float]:
    """Years of additional in-sample history needed to reach
    MIN_TRADES_FOR_SCORING at the rate actually observed. None when no trade
    was observed (no rate to extrapolate).

    DISPLAY ONLY -- never persisted, nothing branches on it. It exists
    because it is the number that turns the operator surface from a nag into
    a decision: e.g. (synthetic) 5 trades over 20 years with 12 required reads
    as 28 more years,
    which makes "wait for more trades" visibly absurd rather than plausible.
    And note what it is NOT saying: because the holdout window is pinned,
    even those years could not enter this family's in-sample set. It is a
    measure of how underpowered the mechanism is, not a waiting time.
    """
    trade_count = _int(observed["trade_count"], "observed.trade_count")
    trades_required = _int(observed["trades_required"],
                           "observed.trades_required")
    if trade_count == 0:
        return None
    if trade_count >= trades_required:
        return 0.0
    start = _date(observed["in_sample_start"], "observed.in_sample_start")
    end = _date(observed["in_sample_end"], "observed.in_sample_end")
    years = (end - start).days / _DAYS_PER_YEAR
    if years <= 0:
        raise ValueError(f"observed in-sample span is not positive: "
                         f"{start}..{end}")
    rate_per_year = trade_count / years
    return (trades_required - trade_count) / rate_per_year


def reactivation_satisfiable(payload: dict, *, as_of: str,
                             catalog_entry: dict,
                             retry_ordinal: int) -> tuple:
    """(satisfiable, human-readable reason) for the operator surface.

    PURE and OFFLINE for every kind -- it never fetches market data and never
    opens a file, so an operator status view can call it on every render. `as_of`
    is required rather than defaulted to today: a hidden clock dependency
    makes the surface non-reproducible and its tests non-deterministic.

    `retry_ordinal` is which retry this WOULD be (1 = the first). The cap
    value itself passes, matching the house >=-passes boundary convention.
    """
    as_of_date = _date(as_of, "as_of")
    retry_ordinal = _int(retry_ordinal, "retry_ordinal")
    reactivation = payload.get("reactivation") or {}
    observed = payload.get("observed") or {}
    kind = reactivation.get("kind")

    cap = constants.MAX_EVIDENCE_RETRIES_PER_FAMILY
    if retry_ordinal > cap:
        return (False, f"retry cap reached: this would be retry "
                       f"{retry_ordinal} of a lifetime cap of {cap} "
                       f"(constants.MAX_EVIDENCE_RETRIES_PER_FAMILY). "
                       f"Curate the mechanism as a NEW family_id, which pays "
                       f"its own trial charge")
    if kind == "none_available":
        return (False, f"reactivation kind is 'none_available': "
                       f"{reactivation.get('rationale')}. Waiting cannot "
                       f"help -- the holdout window is pinned. Curate the "
                       f"mechanism as a NEW family_id if it is worth "
                       f"revisiting")
    if kind not in REACTIVATION_KINDS:
        return (False, f"reactivation.kind {kind!r} is not a recognised "
                       f"lever; this record predates the evidence schema or "
                       f"is malformed")

    registered_on = reactivation.get("registered_on")
    if registered_on is None:
        return (False, "reactivation has no registered_on date, so it cannot "
                       "be shown to predate the retry's data")
    if as_of_date < _date(registered_on, "reactivation.registered_on"):
        return (False, f"as_of {as_of} precedes reactivation.registered_on "
                       f"{registered_on}: the predicate did not exist yet")

    drift = _parameter_drift(catalog_entry, observed)
    if drift:
        return (False, f"catalog parameters have drifted since the deferral "
                       f"({drift}) -- this is a NEW hypothesis and needs its "
                       f"own family_id, not a retry")

    if kind == "widened_instrument_set":
        added = sorted(set(reactivation.get("candidate_tickers") or [])
                       - set(observed.get("instrument_set") or []))
        return (True, f"widen the instrument set by adding "
                      f"{', '.join(added)} (keeping "
                      f"{', '.join(sorted(observed.get('instrument_set') or []))})")
    return (True, f"extend history back to "
                  f"{reactivation.get('earliest_available')} from "
                  f"{reactivation.get('source')}")


def _parameter_drift(catalog_entry: dict, observed: dict) -> str:
    """Human-readable description of how the catalog entry now differs from
    what was deferred, or "" if the hypothesis is unchanged. The fingerprint
    is the authority on whether anything moved; the recorded parameter copy
    is what lets the answer NAME the field instead of saying "something"."""
    recorded_fingerprint = observed.get("parameter_fingerprint")
    if recorded_fingerprint is None:
        return ""
    if parameter_fingerprint(catalog_entry) == recorded_fingerprint:
        return ""
    recorded_params = observed.get("parameters")
    current_params = catalog_entry.get("parameters") or {}
    if not isinstance(recorded_params, dict):
        return "parameter fingerprint differs (no recorded parameter copy)"
    changed = []
    for key in sorted(set(recorded_params) | set(current_params)):
        was, now = recorded_params.get(key, "<absent>"), current_params.get(key, "<absent>")
        if was != now:
            changed.append(f"{key}: {was!r} -> {now!r}")
    if changed:
        return "; ".join(changed)
    return (f"family_id changed: {catalog_entry.get('family_id')!r} does not "
            f"match the deferred hypothesis")


def validate_retry_request(*, payload: dict, catalog_entry: dict,
                           tickers_used, signal_source_path: str,
                           retry_ordinal: int, hasher=source_sha256) -> None:
    """Raise unless this retry is the SAME hypothesis, under its OWN
    pre-registered predicate, within its retry cap.

    Raises ParameterDriftError (a catalog parameter or the signal function's
    own source changed), InstrumentDropError (the instrument set is not a
    real widening), or RetryCapExceeded. `hasher` is injected so the pure
    core stays testable; the default is this module's one filesystem edge.
    """
    reason_code = (payload or {}).get("reason_code")
    stage = ("not_evaluated_error" if reason_code == "engine_error"
             else "insufficient_evidence")
    validate(payload, stage=stage)

    if reason_code == "engine_error":
        raise ValueError("reason_code 'engine_error' has no retry path: an "
                         "engine error is a defect, not evidence. Fix the "
                         "defect and the family re-enters ordinary runs")

    reactivation = payload["reactivation"]
    observed = payload["observed"]
    kind = reactivation["kind"]
    if kind == "none_available":
        raise ValueError("reactivation.kind is 'none_available': no retry is "
                         "sanctioned for this family. Curate the mechanism as "
                         "a NEW family_id, which pays its own trial charge")

    cap = constants.MAX_EVIDENCE_RETRIES_PER_FAMILY
    retry_ordinal = _int(retry_ordinal, "retry_ordinal")
    if retry_ordinal > cap:
        raise RetryCapExceeded(
            f"retry {retry_ordinal} exceeds the lifetime cap of {cap} "
            f"(constants.MAX_EVIDENCE_RETRIES_PER_FAMILY). Each retry that "
            f"clears the DSR bar is another look at the SAME pinned holdout "
            f"window; unbounded retries are fishing")

    drift = _parameter_drift(catalog_entry, observed)
    if drift:
        raise ParameterDriftError(
            f"the hypothesis changed since it was deferred ({drift}). That is "
            f"a NEW hypothesis: give it its own family_id and its own trial "
            f"charge rather than retrying this one")

    actual = hasher(signal_source_path)
    if actual != observed["signal_source_sha256"]:
        raise ParameterDriftError(
            f"signal_source_sha256 differs: recorded "
            f"{observed['signal_source_sha256']}, now {actual}. 'More data' "
            f"and 'different code' are different claims -- a changed signal "
            f"function is a new hypothesis")

    used = set(_tickers(list(tickers_used), "tickers_used"))
    original = set(observed["instrument_set"])
    if kind == "widened_instrument_set":
        candidates = set(reactivation["candidate_tickers"])
        if not original < used:
            missing = sorted(original - used)
            raise InstrumentDropError(
                f"tickers_used {sorted(used)} is not a STRICT superset of the "
                f"deferred instrument set {sorted(original)}"
                + (f" (dropped: {missing})" if missing else
                   " -- an identical set is a bit-identical re-run"))
        outside = sorted(used - candidates)
        if outside:
            raise InstrumentDropError(
                f"tickers_used reaches outside the pre-registered "
                f"candidate_tickers: {outside}. The candidate set was "
                f"registered before the retry's data existed; reaching "
                f"outside it is shopping")
    else:  # extended_history
        if used != original:
            raise InstrumentDropError(
                f"reactivation.kind 'extended_history' requires the SAME "
                f"instrument set as the deferral ({sorted(original)}); got "
                f"{sorted(used)}. Extending history and widening instruments "
                f"at once makes the result uninterpretable")
