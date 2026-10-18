"""Persistent, append-only trial ledger for the strategy mining pipeline.
Tracks every distinct family_id ever tested by any mining run:
first-tested date, its PINNED holdout window (never recalculated on
re-test), and outcome history. This is what makes the Deflated Sharpe
correction cumulative ACROSS runs instead of resetting to "this run's
candidate count" every time (a cross-run multiple-testing correction).

Format: JSON Lines, one event dict per line, append-only. Never rewritten
in place -- a family's pinned holdout window is authoritative from its
FIRST "first_test" event; later "first_test" attempts for the same
family_id are no-ops (idempotent), enforced here, not by the caller.

## Cross-process locking

Every WRITE (`record_first_test`'s check-then-act, `record_outcome`'s
append) goes through `_LedgerLock`, a Windows OS byte-range mutex on a
sidecar `<ledger_path>.lock` file (an OS lock is preferred over a lockfile+mtime reclaim because it
auto-releases on process death/crash, closing the two-writer TOCTOU a
stale-reclaim scheme reopens). Without this, two mining runs (e.g. the
daily and swing/intraday pipelines, or two concurrent invocations of
either) racing on the same family_id at close to the same instant could
both observe "not yet tested" via `has_been_tested` and both append a
first_test event (for example via the swing pipeline's own
early-exclusion filter, `run_swing_mining.compute_swing_candidates`).
Reads (`has_been_tested`, `read_all`, `get_pinned_holdout_window`, etc.)
stay lock-free -- they are advisory filters, and the actual
correctness-bearing check happens inside the locked write path.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Optional

from . import evidence as _evidence
# Aliased: record_outcome's own `evidence` PARAMETER (the caller's payload
# for one outcome) shadows the bare module name inside that function's own
# body, so every reference to the module here uses this alias instead.

_LOCK_POLL_INTERVAL_S = 0.02
_LOCK_TIMEOUT_S = 10.0


class LedgerLockTimeout(RuntimeError):
    """Another process held the trial ledger lock past the wait budget."""


class LedgerLockUnsupported(RuntimeError):
    """The trial ledger lock uses a Windows byte-range lock and has no other implementation."""


class OutcomeAlreadyRecorded(RuntimeError):
    """`record_outcome_if_not_scored` lost the race: another caller's
    outcome for this exact (family_id, construction) landed first, between
    this caller's own early already_scored prefilter and this call. The
    caller's freshly computed result must be discarded, not retried --
    retrying would just re-observe the same "already scored" state."""


class _LedgerLock:
    """Windows OS byte-range mutex serializing ledger writes across
    processes -- `msvcrt.locking` on byte 0 of a sidecar `.lock` file,
    held for the duration of a critical section (read-then-write or a
    single append). Mandatory and conflicts across handles even WITHIN one
    process (a Windows locking property), so it must never be
    acquired twice by the same call stack -- callers already holding it
    must use the `_unlocked` primitives directly, not re-enter through a
    locked wrapper.

    Blocks up to `timeout` seconds, polling in `_LOCK_POLL_INTERVAL_S`
    slices (LK_NBLCK + retry, rather than LK_LOCK's fixed ~10-attempt/
    1s-interval schedule) so the wait budget is explicit and testable.
    """

    def __init__(self, ledger_path: str, timeout: float = _LOCK_TIMEOUT_S):
        self._path = ledger_path + ".lock"
        self._timeout = timeout
        self._fh = None

    def __enter__(self):
        if sys.platform != "win32":
            raise LedgerLockUnsupported(
                "the trial ledger lock needs Windows (msvcrt byte-range locking); "
                "no lock implementation exists for this platform")
        import msvcrt
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self._path):
            open(self._path, "a").close()
        fh = open(self._path, "r+b")
        # The whole acquire loop is wrapped so ANY exception out of it
        # (a msvcrt error outside (OSError, PermissionError), a
        # KeyboardInterrupt during the retry sleep, etc.) still closes
        # `fh` before propagating. Relying on GC to eventually
        # finalize an unreferenced file object is not good enough: the
        # OS lock (if already granted) would stay held until whenever
        # that finalization happens, deadlocking every other caller.
        try:
            deadline = time.monotonic() + self._timeout
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except (OSError, PermissionError):
                    # On Windows a taken lock surfaces as
                    # EACCES/EPERM/EBUSY, not one canonical errno -- catch
                    # the family, never a single code.
                    if time.monotonic() >= deadline:
                        raise LedgerLockTimeout(
                            f"could not acquire {self._path} within "
                            f"{self._timeout}s")
                    time.sleep(_LOCK_POLL_INTERVAL_S)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        return self

    def __exit__(self, *exc_info):
        import msvcrt
        fh = self._fh
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass                         # already gone with the process
        finally:
            fh.close()
            self._fh = None
        return False


def _read_lines(ledger_path: str) -> list:
    if not os.path.exists(ledger_path):
        return []
    records = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _append_unlocked(ledger_path: str, record: dict) -> None:
    """Raw write, NO locking -- callers must already hold `_LedgerLock`
    for `ledger_path` (or be a single-writer test) before calling this.
    Exists so a caller already inside a critical section (`record_first_test`)
    never has to re-enter the non-reentrant lock to append."""
    parent = os.path.dirname(ledger_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _append(ledger_path: str, record: dict) -> None:
    with _LedgerLock(ledger_path):
        _append_unlocked(ledger_path, record)


def read_all(ledger_path: str) -> list:
    """All events, in append order. Low-level -- prefer the accessors
    below unless you need the raw event stream."""
    return _read_lines(ledger_path)


def _first_test_exists(records: list, family_id: str) -> bool:
    return any(r["event"] == "first_test" and r["family_id"] == family_id
               for r in records)


def has_been_tested(ledger_path: str, family_id: str) -> bool:
    return _first_test_exists(_read_lines(ledger_path), family_id)


def record_first_test(ledger_path: str, family_id: str, tested_date: str,
                       holdout_start: str, holdout_end: str,
                       representative_returns: list, run_id: str) -> bool:
    """Append a first_test event for `family_id`, UNLESS one already
    exists (idempotent no-op -- returns False in that case, True if it
    recorded a new event). This is the enforcement point for "the window
    doesn't move on re-test". The existence check and
    the append happen inside ONE held `_LedgerLock`, so two processes
    racing on the same family_id can never both observe "not yet tested"
    -- the second one's check runs only after the first's write has
    already landed."""
    with _LedgerLock(ledger_path):
        if _first_test_exists(_read_lines(ledger_path), family_id):
            return False
        _append_unlocked(ledger_path, {
            "event": "first_test",
            "family_id": family_id,
            "tested_date": tested_date,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "representative_returns": list(representative_returns),
            "run_id": run_id,
        })
        return True


def get_pinned_holdout_window(ledger_path: str, family_id: str):
    """{'holdout_start', 'holdout_end', 'tested_date'} from `family_id`'s
    FIRST first_test event, or None if it has never been tested."""
    for r in _read_lines(ledger_path):
        if r["event"] == "first_test" and r["family_id"] == family_id:
            return {"holdout_start": r["holdout_start"],
                    "holdout_end": r["holdout_end"],
                    "tested_date": r["tested_date"]}
    return None


def cumulative_family_count(ledger_path: str) -> int:
    """Count of DISTINCT family_ids ever first-tested, all-time. Used as a
    monotonic FLOOR on the DSR trial count alongside the correlation-based
    `dsr.num_independent_trials` estimate (which only sees the current
    run's aligned return series) -- a floor is needed so the trial count
    can never shrink across runs."""
    seen = {r["family_id"] for r in _read_lines(ledger_path)
            if r["event"] == "first_test"}
    return len(seen)


def post_batch_floor(ledger_path: str, family_ids) -> int:
    """The multiple-testing floor THIS run must be judged against: the
    all-time distinct-family count PLUS the distinct families in
    `family_ids` that this run is first-testing.

    `cumulative_family_count` alone is the PRE-run count, and reading it
    before the run records its own `first_test` events under-corrects by
    exactly the number of new families -- an easier bar. A run that tests
    k new hypotheses is choosing among (prior + k), not (prior).

    Read-only and idempotent: it never records anything, so calling it
    before scoring does not itself consume a trial. Repeated ids collapse
    (one family tested on many tickers is ONE hypothesis -- the same
    pooling rule pooled multi-ticker tests follow), and ids already in
    the ledger are not re-charged, because their trial was paid for on
    first test.

    Lives here, not in a caller, so every scoring path (e.g.
    `scripts/run_batch3_mining.compute_shared_correction`) shares one
    implementation and copies cannot drift apart.
    """
    records = _read_lines(ledger_path)
    already_tested = {r["family_id"] for r in records
                      if r["event"] == "first_test"}
    new_ids = {fid for fid in family_ids if fid not in already_tested}
    return len(already_tested) + len(new_ids)

def all_family_representative_returns(ledger_path: str) -> dict:
    """{family_id: representative_returns} for every family ever
    first-tested. NOTE: series from different runs may be aligned to
    different calendar windows and are not directly comparable across
    runs -- callers combining these with a fresh run's own series should
    only correlate series that share a common index."""
    out = {}
    for r in _read_lines(ledger_path):
        if r["event"] == "first_test":
            out[r["family_id"]] = r["representative_returns"]
    return out


CONSTRUCTIONS = ("per_trade", "ABI-1")


def record_outcome(ledger_path: str, family_id: str, run_id: str,
                    tested_date: str, stage: str,
                    dsr_value: Optional[float] = None,
                    note: Optional[str] = None,
                    ticker_label: Optional[str] = None,
                    sharpe_in_sample: Optional[float] = None,
                    evidence: Optional[dict] = None,
                    construction: str = "per_trade") -> None:
    """`stage` is one of evidence.VALID_STAGES: see evidence.SCORED_STAGES /
    evidence.UNSCORED_STAGES for the current vocabulary -- not enumerated
    here by name, since a hand-copied list goes stale on the next stage
    added there.

    THIS IS THE CHOKEPOINT: every scoring path writes outcomes through
    this one function, so evidence.py's biconditional is enforced HERE,
    on every write, rather than trusted to each call site's own
    discipline:

        stage in SCORED_STAGES   <=> dsr_value is a finite float AND evidence is None
        stage in UNSCORED_STAGES <=> dsr_value is None           AND evidence is a
                                      valid payload for `stage`

    A SCORED stage with `dsr_value=None` (or non-finite) is a record
    shape that would make a family indistinguishable from a real
    statistical rejection -- rejected here, not merely discouraged.
    An UNSCORED stage's `evidence` is validated via `evidence.validate(...,
    at_write_time=True)` so a malformed payload is caught at the point of
    writing, not left for a reader to discover later. `sharpe_in_sample`
    must also stay None on an UNSCORED stage (see the `sharpe_in_sample`
    note below on why a fabricated value there would corrupt
    `pooled_sr_variance`'s pool) -- a real per-trade sharpe MAY still live
    inside `evidence['observed']['sharpe_in_sample']` for a deferral; that
    is a different field with a different, read-only purpose.

    `evidence` is written verbatim as its own ledger key on EVERY record
    (None on a SCORED stage) -- matching note/ticker_label/sharpe_in_
    sample's own established "always present, nullable" convention below,
    so a reader never has to branch on whether the key exists.

    `note` and `ticker_label` are OPTIONAL, additive fields (used by
    pooled multi-ticker tests): when a
    single ticker no longer represents what was tested (e.g. a pooled
    trade-level test across hundreds of tickers), `ticker_label`
    names what was tested instead of a symbol and `note` carries the
    methodology/coverage summary. Call sites that do not need them
    omit both -- they default to None
    and are still WRITTEN as explicit None keys (not omitted), matching
    `dsr_value`'s own always-present-but-nullable convention, so every
    outcome record this function writes has a uniform key set. Readers
    must still use `.get(...)`, not `[...]`, for these two keys -- a
    ledger row is not guaranteed to carry them.

    `sharpe_in_sample` is a THIRD such additive field (it feeds the
    sr_variance pooling): the family's in-sample Sharpe at the
    moment a real DSR was computed. Pass it whenever `dsr_value` is a
    real number; leave it None (the default) on any outcome that never
    reached a real DSR computation (e.g. too few trades, or a defensive
    record filed after an exception) -- a fabricated sharpe on an
    unscored outcome would corrupt `pooled_sr_variance` below with a
    number that was never actually measured. See
    `pooled_sr_variance`'s own docstring for what this feeds.

    A non-finite `sharpe_in_sample` (NaN/inf -- e.g. a degenerate candidate
    whose trades all had byte-identical returns, giving a zero-variance
    denominator) is normalized to None here rather than written verbatim.
    This is the ONE choke point every scoring call site writes through, so
    guarding it here protects all of them without touching each site's own
    control flow -- and keeps the ledger row itself honest, not just
    `pooled_sr_variance`'s own separate read-side guard.

    `construction`: "per_trade" (the default) or "ABI-1" (written by the
    ABI-1 drivers, e.g. `abi1_driver.record_abi1_outcome`). This is
    the ONLY thing that lets `pooled_sr_variance` below tell the two
    scoring constructions' `sharpe_in_sample` values apart -- they are on
    completely different scales (per-trade vs per-session) and must never
    share a pool. It is written on every row because the append-only
    ledger cannot have a schema field retrofitted onto rows already
    written."""
    record = _build_outcome_record(
        family_id, run_id, tested_date, stage, dsr_value=dsr_value,
        note=note, ticker_label=ticker_label,
        sharpe_in_sample=sharpe_in_sample, evidence=evidence,
        construction=construction)
    _append(ledger_path, record)


def _build_outcome_record(family_id: str, run_id: str, tested_date: str,
                          stage: str, *, dsr_value: Optional[float],
                          note: Optional[str], ticker_label: Optional[str],
                          sharpe_in_sample: Optional[float],
                          evidence: Optional[dict],
                          construction: str) -> dict:
    """Validate and build the exact record dict `record_outcome` appends --
    every check `record_outcome`'s own docstring documents, extracted so
    `record_outcome_if_not_scored` below can validate a candidate record
    BEFORE ever touching the lock, without duplicating this logic. Pure: no
    I/O, no lock, never raises anything but the ValueErrors already
    documented on `record_outcome`."""
    if stage not in _evidence.VALID_STAGES:
        raise ValueError(f"invalid stage: {stage!r}; expected one of "
                         f"{list(_evidence.VALID_STAGES)}")
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"invalid construction: {construction!r}; expected "
                         f"one of {list(CONSTRUCTIONS)}")
    if sharpe_in_sample is not None and not math.isfinite(sharpe_in_sample):
        sharpe_in_sample = None

    if stage in _evidence.SCORED_STAGES:
        if evidence is not None:
            raise ValueError(
                f"stage {stage!r} is a SCORED outcome and must not carry an "
                f"evidence payload -- an evidence payload means no DSR was "
                f"computed (see evidence.validate's own check)")
        if dsr_value is None or not math.isfinite(dsr_value):
            raise ValueError(
                f"stage {stage!r} is a SCORED outcome and requires a finite "
                f"dsr_value; got {dsr_value!r}. A never-scored candidate "
                f"belongs to one of evidence.UNSCORED_STAGES "
                f"({list(_evidence.UNSCORED_STAGES)}), not a SCORED stage "
                f"with no DSR")
    else:  # stage in _evidence.UNSCORED_STAGES
        if dsr_value is not None:
            raise ValueError(
                f"stage {stage!r} is an UNSCORED outcome and must not carry "
                f"a dsr_value -- a fabricated DSR on a deferred outcome is "
                f"exactly the ambiguity this lifecycle exists to prevent; "
                f"got {dsr_value!r}")
        if sharpe_in_sample is not None:
            raise ValueError(
                f"stage {stage!r} is an UNSCORED outcome and must not carry "
                f"a top-level sharpe_in_sample -- no real DSR was computed, "
                f"so a fabricated value here would corrupt "
                f"pooled_sr_variance's pool (see this function's own "
                f"docstring above). A real per-trade sharpe may still live "
                f"inside evidence['observed']['sharpe_in_sample'] instead")
        _evidence.validate(evidence, stage=stage, at_write_time=True)

    return {
        "event": "outcome",
        "family_id": family_id,
        "run_id": run_id,
        "tested_date": tested_date,
        "stage": stage,
        "dsr_value": dsr_value,
        "note": note,
        "ticker_label": ticker_label,
        "sharpe_in_sample": sharpe_in_sample,
        "evidence": evidence,
        "construction": construction,
    }


def _outcome_exists(records: list, family_id: str, construction: str) -> bool:
    return any(r["event"] == "outcome" and r["family_id"] == family_id
               and r.get("construction", "per_trade") == construction
               for r in records)


def record_outcome_if_not_scored(ledger_path: str, family_id: str, run_id: str,
                                  tested_date: str, stage: str, *,
                                  dsr_value: Optional[float] = None,
                                  note: Optional[str] = None,
                                  ticker_label: Optional[str] = None,
                                  sharpe_in_sample: Optional[float] = None,
                                  evidence: Optional[dict] = None,
                                  construction: str = "per_trade") -> bool:
    """Atomic check-and-append counterpart to `record_outcome`, closing the
    TOCTOU between an early, unlocked "has this family already been scored
    under this construction" prefilter (every ABI-1 driver script runs one
    before its own, often expensive, scoring work -- see
    `abi1_driver.record_abi1_outcome`) and the append itself. Two processes
    invoked concurrently for the same family could both pass the prefilter
    and both append a permanent outcome row, corrupting the one-shot-per-
    family invariant the mining pipeline depends on.

    Builds and validates the candidate record FIRST, via the same
    `_build_outcome_record` helper `record_outcome` itself uses (pure, no
    I/O, raises the identical ValueErrors on a malformed payload) -- so a
    bad call fails before ever touching the lock, and can never block a
    concurrent, well-formed one. Only THEN does it re-check for an existing
    outcome of this exact (family_id, construction) and append ONLY if none
    exists, both inside ONE held `_LedgerLock` -- mirrors
    `record_first_test`'s own check-then-act pattern (this module's
    docstring, "Cross-process locking"), just applied at the OUTCOME write
    instead of the first_test write.

    Deliberately does NOT hold the lock across the caller's own (often
    expensive -- a real intraday backtest + ABI-1 evaluation) scoring work:
    only the final, cheap check-and-append is atomic. A lost race still
    costs the loser its wasted compute, but the ledger itself -- the actual
    system of record -- can never end up with two permanent outcomes for
    one family. A single ledger-wide lock (not a per-family one) is
    deliberate: these drivers rarely run concurrently in practice, and
    `_LedgerLock` is already held only for the duration of this short
    critical section, not the surrounding compute, so cross-family
    contention is not a real cost.

    Returns True if it recorded a new outcome, False if a prior outcome of
    this exact `construction` already existed for `family_id` (idempotent
    no-op, matching `record_first_test`'s own return convention -- the
    caller lost a race it could not have detected any earlier without
    holding the lock across its own scoring work, which this function
    deliberately does not do). Checks the EXACT `construction` value, not
    merely `family_id` presence -- a prior `"per_trade"` outcome for the
    same family does not block a `"ABI-1"` attempt (matches
    `run_abi1_mining.py`'s own re-run-guard semantics)."""
    record = _build_outcome_record(
        family_id, run_id, tested_date, stage, dsr_value=dsr_value,
        note=note, ticker_label=ticker_label,
        sharpe_in_sample=sharpe_in_sample, evidence=evidence,
        construction=construction)
    with _LedgerLock(ledger_path):
        if _outcome_exists(_read_lines(ledger_path), family_id, construction):
            return False
        _append_unlocked(ledger_path, record)
        return True


def latest_outcome(ledger_path: str, family_id: str):
    """The most recent recorded 'stage' for `family_id` from outcome
    events, in ledger order (last one wins). None if never scored."""
    stage = None
    for r in _read_lines(ledger_path):
        if r["event"] == "outcome" and r["family_id"] == family_id:
            stage = r["stage"]
    return stage


def pooled_sr_variance(ledger_path: str,
                        min_samples: Optional[int] = None) -> Optional[float]:
    """The cross-candidate Sharpe-ratio variance the DSR's `sr_variance`
    argument wants (dsr.expected_max_sharpe_ratio), pooled across every
    real score this ledger has ever recorded, rather than one run's own
    (often small) batch of candidates.

    WHY pool: sr_variance -- not n_trials -- is the DOMINANT term in
    the DSR bar: it is a straight multiplier, while n_trials sits inside
    a slowly-growing factor. A run's own small batch is an unstable
    estimate of it, and DEFAULT_SR_VARIANCE_FLOOR is a rough fallback,
    not an empirically grounded value; pooling every recorded score is
    the empirical estimate.

    Reads every `outcome` event carrying a non-null `sharpe_in_sample`
    (see `record_outcome`'s docstring -- only present on outcomes that
    reached a real DSR computation). Deduplicates by `family_id`, keeping
    the LATEST recorded value (mirrors `latest_outcome`'s own
    last-one-wins convention): a re-tested family's underlying in-sample
    data is pinned and does not change on re-test, so counting it once
    per re-score would give that family disproportionate weight in the
    pool without adding any real information.

    Returns None -- meaning "the pool is not trustworthy yet, fall back"
    -- when fewer than `min_samples` (default
    `constants.MIN_POOLED_SR_VARIANCE_SAMPLES`) distinct families
    qualify, or when the pooled variance is exactly zero (the same
    degenerate-guard every existing sr_variance computation in this
    codebase already applies). Deliberately NOT backfilled from anything
    on disk outside the ledger: per-run result artifacts are not
    version-controlled and may be missing, and building the pool from
    files that may not exist on the next run would not be honest
    pooling. The pool therefore grows only from ledger records that
    carry the field.

    Never writes to the ledger."""
    if min_samples is None:
        from . import constants
        min_samples = constants.MIN_POOLED_SR_VARIANCE_SAMPLES
    latest_by_family: dict = {}
    for r in _read_lines(ledger_path):
        if r["event"] != "outcome":
            continue
        # A row with no "construction" key is read as per_trade (the
        # field's default), the same rule `_outcome_exists` applies.
        if r.get("construction", "per_trade") != "per_trade":
            continue
        sharpe = r.get("sharpe_in_sample")
        if sharpe is None:
            continue
        sharpe = float(sharpe)
        # A NaN/inf value (a corrupted or hand-edited ledger row) must not
        # silently pollute the pool's mean/variance arithmetic.
        if not math.isfinite(sharpe):
            continue
        latest_by_family[r["family_id"]] = sharpe
    values = list(latest_by_family.values())
    if len(values) < min_samples:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if variance <= 0:
        return None
    return variance
