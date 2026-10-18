import os
import sys
import threading
import time

import numpy as np
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import constants, evidence, trial_ledger


def _insufficient_evidence_payload(family_id="fam_x", trade_count=5):
    """A minimal, well-formed insufficient_evidence payload for tests that
    only care about the ledger's write-time enforcement, not the payload's
    own content -- mirrors test_mining_evidence.py's own builder convention
    (no fixtures, a module-level _-prefixed builder)."""
    return evidence.build_insufficient_trades(
        stats={"trade_count": trade_count},
        catalog_entry={"family_id": family_id, "parameters": {}},
        instrument_set=["SPY"],
        in_sample_start="2020-01-01", in_sample_end="2024-06-01",
        pinned_window={"holdout_start": "2024-07-01", "holdout_end": "2024-12-31"},
        holdout_look="unspent",
        signal_source_path="src/mining/signals.py",
        signal_source_sha256="a" * 64,
        evidence_source="measured",
        reactivation={"kind": "none_available", "rationale": "test"})


def _engine_error_payload():
    return evidence.build_engine_error(
        error=RuntimeError("boom"), instrument_set=["SPY"],
        pinned_window={"holdout_start": "2024-07-01", "holdout_end": "2024-12-31"},
        holdout_look="unspent")


def test_first_test_is_idempotent_and_pins_the_window(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    recorded = trial_ledger.record_first_test(
        path, "mean_reversion_ibs", "2026-08-27", "2026-01-01", "2026-06-30",
        [0.01, -0.005, 0.02], "run-1")
    assert recorded is True
    window = trial_ledger.get_pinned_holdout_window(path, "mean_reversion_ibs")
    assert window == {"holdout_start": "2026-01-01", "holdout_end": "2026-06-30",
                       "tested_date": "2026-08-27"}
    recorded_again = trial_ledger.record_first_test(
        path, "mean_reversion_ibs", "2026-09-15", "2026-07-01", "2026-12-31",
        [0.03], "run-2")
    assert recorded_again is False
    assert trial_ledger.get_pinned_holdout_window(path, "mean_reversion_ibs") == window


def test_unknown_family_has_no_pinned_window(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    assert trial_ledger.get_pinned_holdout_window(path, "nonexistent") is None
    assert trial_ledger.has_been_tested(path, "nonexistent") is False


def test_cumulative_family_count_counts_distinct_families(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(path, "fam_a", "2026-08-27", "s", "e", [0.1], "run-1")
    trial_ledger.record_first_test(path, "fam_b", "2026-08-27", "s", "e", [0.2], "run-1")
    trial_ledger.record_first_test(path, "fam_a", "2026-09-01", "s2", "e2", [0.3], "run-2")
    assert trial_ledger.cumulative_family_count(path) == 2


def test_record_outcome_rejects_invalid_stage(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-08-27", "bogus")


def test_record_outcome_appends_and_read_all_sees_it(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-08-27", "survived", dsr_value=0.97)
    events = trial_ledger.read_all(path)
    assert len(events) == 1
    assert events[0]["stage"] == "survived"
    assert events[0]["dsr_value"] == 0.97


def test_all_family_representative_returns_returns_first_test_only(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(path, "fam_a", "2026-08-27", "s", "e", [0.1, 0.2], "run-1")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-08-27", "survived", dsr_value=0.97)
    out = trial_ledger.all_family_representative_returns(path)
    assert out == {"fam_a": [0.1, 0.2]}


def test_latest_outcome_returns_most_recent_stage(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-08-27", "cut_dsr", dsr_value=0.4)
    trial_ledger.record_outcome(path, "fam_a", "run-2", "2026-09-15", "survived", dsr_value=0.97)
    assert trial_ledger.latest_outcome(path, "fam_a") == "survived"


def test_latest_outcome_none_when_never_scored(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    assert trial_ledger.latest_outcome(path, "fam_a") is None


def test_record_outcome_default_call_omits_note_and_ticker_label(tmp_path):
    # A call site that passes only the 5 positional args + dsr_value
    # must not need to know about the optional note/ticker_label fields:
    # both read back as None.
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-08-27", "cut_dsr", dsr_value=0.4)
    events = trial_ledger.read_all(path)
    assert events[0].get("note") is None
    assert events[0].get("ticker_label") is None


def test_record_outcome_accepts_optional_note_and_ticker_label(tmp_path):
    # A pooled multi-ticker outcome needs to say WHAT it tested (universe
    # size, methodology) since there is no single representative ticker --
    # see mining.results_log for how these surface in the cumulative table.
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        path, "fam_a", "run-1", "2026-01-15", "survived", dsr_value=0.9,
        note="Pooled trade-level test across 50 synthetic tickers.",
        ticker_label="50 tickers (synthetic pool)")
    events = trial_ledger.read_all(path)
    assert events[0]["note"] == "Pooled trade-level test across 50 synthetic tickers."
    assert events[0]["ticker_label"] == "50 tickers (synthetic pool)"


# ---------------------------------------------------------------------------
# Cross-process/cross-thread locking. msvcrt OS byte-range locks conflict
# across handles even WITHIN one process,
# so a thread-based test is a faithful stand-in for two real processes racing
# on the same ledger file -- no multiprocessing needed to prove the mutex.
# ---------------------------------------------------------------------------

def test_ledger_lock_blocks_until_released_then_succeeds(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    hold_started = threading.Event()

    def holder():
        with trial_ledger._LedgerLock(path):
            hold_started.set()
            time.sleep(0.2)

    t = threading.Thread(target=holder)
    t.start()
    assert hold_started.wait(timeout=2)

    start = time.monotonic()
    with trial_ledger._LedgerLock(path, timeout=2.0):
        elapsed = time.monotonic() - start
    t.join()

    # A second handle must have BLOCKED until the holder released -- if it
    # had raced past (no real mutual exclusion), elapsed would be ~0.
    assert elapsed >= 0.15


def test_ledger_lock_closes_handle_on_unexpected_enter_exception(tmp_path, monkeypatch):
    """fh is opened before the acquire loop, so ANY exception out of the
    loop (a msvcrt error outside (OSError, PermissionError), a
    KeyboardInterrupt during the retry sleep, etc.) must still close it
    before propagating. A skipped fh.close() would leak the handle (and,
    if the lock had actually been granted moments earlier, the OS lock
    along with it -- a permanent deadlock for every other caller). Checks
    the handle's own .closed state directly rather than inferring it from
    os.remove/delete semantics -- CPython's refcounting GC can incidentally
    finalize (and thus close) an unreferenced file object fast enough to
    hide a leak in a delete-based check, even though relying on that
    timing is exactly the fragility this test exists to rule out."""
    import msvcrt
    path = str(tmp_path / "ledger.jsonl")
    lock_path = path + ".lock"

    opened = []
    real_open = open

    def tracking_open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        if file == lock_path and mode == "r+b":
            opened.append(fh)
        return fh

    monkeypatch.setattr("builtins.open", tracking_open)

    def boom(fd, mode, nbytes):
        raise RuntimeError("simulated unexpected failure mid-acquire")

    monkeypatch.setattr(msvcrt, "locking", boom)

    with pytest.raises(RuntimeError):
        with trial_ledger._LedgerLock(path, timeout=5.0):
            pass

    assert len(opened) == 1
    assert opened[0].closed, (
        "_LedgerLock.__enter__ must close its file handle even when an "
        "unexpected exception interrupts the acquire loop")


def test_ledger_lock_times_out_while_held(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    hold_started = threading.Event()
    keep_holding = threading.Event()

    def holder():
        with trial_ledger._LedgerLock(path):
            hold_started.set()
            keep_holding.wait(timeout=2)

    t = threading.Thread(target=holder)
    t.start()
    try:
        assert hold_started.wait(timeout=2)
        with pytest.raises(trial_ledger.LedgerLockTimeout):
            with trial_ledger._LedgerLock(path, timeout=0.1):
                pass
    finally:
        keep_holding.set()
        t.join()


def test_concurrent_record_first_test_only_one_wins_the_race(tmp_path, monkeypatch):
    """The TOCTOU this lock exists to close: if has_been_tested() (read) and
    _append() (write) were two separate, unlocked operations, two
    processes racing on the same family_id could both observe "not yet
    tested" and both append a first_test event. Widen the window between
    the read and the write (an injected sleep) so a barrier-released pair
    of threads is virtually guaranteed to overlap it if nothing serializes
    them -- then prove the lock prevents exactly that."""
    path = str(tmp_path / "ledger.jsonl")
    original_read_lines = trial_ledger._read_lines

    def slow_read_lines(p):
        result = original_read_lines(p)
        time.sleep(0.1)
        return result

    monkeypatch.setattr(trial_ledger, "_read_lines", slow_read_lines)

    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        recorded = trial_ledger.record_first_test(
            path, "fam_race", "2026-08-31", "2026-01-01", "2026-06-30",
            [0.01], "run-race")
        with results_lock:
            results.append(recorded)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [False, True]
    events = [r for r in trial_ledger.read_all(path)
              if r["event"] == "first_test" and r["family_id"] == "fam_race"]
    assert len(events) == 1


def test_concurrent_writes_are_serialized(tmp_path, monkeypatch):
    """record_outcome's own append must also go through the lock -- not just
    record_first_test's check-then-act -- so two processes appending at the
    same instant can never interleave partial writes onto the same line."""
    path = str(tmp_path / "ledger.jsonl")
    intervals = []
    intervals_lock = threading.Lock()
    original_append_unlocked = trial_ledger._append_unlocked

    def tracked(ledger_path, record):
        start = time.monotonic()
        time.sleep(0.05)
        original_append_unlocked(ledger_path, record)
        end = time.monotonic()
        with intervals_lock:
            intervals.append((start, end))

    monkeypatch.setattr(trial_ledger, "_append_unlocked", tracked)

    barrier = threading.Barrier(2)

    def worker(family_id):
        barrier.wait()
        trial_ledger.record_outcome(
            path, family_id, "run-x", "2026-08-31", "cut_dsr", dsr_value=0.4)

    threads = [threading.Thread(target=worker, args=(f"fam_{i}",))
               for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(intervals) == 2
    (start1, end1), (start2, end2) = sorted(intervals)
    # Non-overlapping: the second write's own critical section did not begin
    # until the first's had fully finished.
    assert start2 >= end1


# --- post_batch_floor: the shared multiple-testing floor helper -------------
# The floor a run must be judged against is the ledger's cumulative
# distinct-family count PLUS the families this run is first-testing -- a run
# that tests k new families is choosing among at least (prior + k)
# hypotheses, not (prior). Reading only the bare pre-run count would
# under-correct. One shared function so callers cannot drift apart.

def test_post_batch_floor_counts_this_runs_new_families(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    assert trial_ledger.post_batch_floor(path, ["fam_a", "fam_b", "fam_c"]) == 3


def test_post_batch_floor_adds_only_the_untested_families(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(path, "fam_a", "2026-08-27", "s", "e", [], "run-1")
    trial_ledger.record_first_test(path, "fam_b", "2026-08-27", "s", "e", [], "run-1")
    # fam_a/fam_b are re-tests (already charged), fam_c is genuinely new.
    assert trial_ledger.post_batch_floor(path, ["fam_a", "fam_b", "fam_c"]) == 3


def test_post_batch_floor_deduplicates_repeated_family_ids(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    # One family tested on three tickers is ONE hypothesis, one trial.
    assert trial_ledger.post_batch_floor(path, ["fam_a", "fam_a", "fam_a"]) == 1


def test_post_batch_floor_equals_cumulative_count_when_nothing_is_new(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(path, "fam_a", "2026-08-27", "s", "e", [], "run-1")
    trial_ledger.record_first_test(path, "fam_b", "2026-08-27", "s", "e", [], "run-1")
    assert trial_ledger.post_batch_floor(path, ["fam_a"]) == 2
    assert trial_ledger.post_batch_floor(path, []) == 2


def test_post_batch_floor_never_writes_to_the_ledger(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(path, "fam_a", "2026-08-27", "s", "e", [], "run-1")
    before = open(path, "rb").read()
    trial_ledger.post_batch_floor(path, ["fam_a", "fam_new"])
    assert open(path, "rb").read() == before
    assert trial_ledger.has_been_tested(path, "fam_new") is False


def test_post_batch_floor_matches_the_batch3_reference_arithmetic(tmp_path):
    # The reference arithmetic, recomputed independently here:
    # cumulative_family_count + |{ids not yet tested}|
    # (scripts/run_batch3_mining.compute_shared_correction delegates to
    # post_batch_floor for it).
    path = str(tmp_path / "ledger.jsonl")
    for fam in ("f1", "f2", "f3"):
        trial_ledger.record_first_test(path, fam, "2026-08-27", "s", "e", [], "run-1")
    submitted = ["f2", "f3", "n1", "n2", "n3", "n4"]
    expected = (trial_ledger.cumulative_family_count(path)
                + len({f for f in submitted
                       if not trial_ledger.has_been_tested(path, f)}))
    assert trial_ledger.post_batch_floor(path, submitted) == expected == 7
# --- sharpe_in_sample capture + pooled_sr_variance --------------------------
# sr_variance pooling. sr_variance is the
# DOMINANT term in the DSR bar (a straight multiplier vs n_trials's
# slowly-growing factor). Estimating it from one run's own 1-3
# candidates, or falling back to an invented floor constant, is unstable. This
# pools sharpe_in_sample across every real DSR score ever recorded in the
# ledger instead.


def test_record_outcome_persists_sharpe_in_sample(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.42,
                                sharpe_in_sample=0.77)
    events = trial_ledger.read_all(path)
    assert events[0]["sharpe_in_sample"] == 0.77


def test_record_outcome_defaults_sharpe_in_sample_to_none(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.42)
    events = trial_ledger.read_all(path)
    # Written explicitly (not omitted), matching note/ticker_label's own
    # convention -- every outcome record has a uniform key set.
    assert "sharpe_in_sample" in events[0]
    assert events[0]["sharpe_in_sample"] is None


def test_pooled_sr_variance_none_below_the_minimum(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    for i in range(3):
        trial_ledger.record_outcome(path, "fam_%d" % i, "run-1", "2026-09-02",
                                    "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=0.1 * i)
    assert trial_ledger.pooled_sr_variance(path, min_samples=5) is None


def test_pooled_sr_variance_computed_once_the_minimum_is_met(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    values = [0.2, 0.5, 0.9, 0.1, 0.6]
    for i, v in enumerate(values):
        trial_ledger.record_outcome(path, "fam_%d" % i, "run-1", "2026-09-02",
                                    "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=v)
    got = trial_ledger.pooled_sr_variance(path, min_samples=5)
    assert got == pytest.approx(np.var(values, ddof=1))


def test_pooled_sr_variance_ignores_records_with_no_sharpe(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    # Two families never reached a real DSR (too few trades): no sharpe.
    # Recorded as insufficient_evidence -- the honest shape for "never
    # scored" (a cut_dsr row with dsr_value=None is refused at write time;
    # see test_record_outcome_rejects_a_scored_stage_with_no_dsr_value).
    trial_ledger.record_outcome(path, "fam_x", "run-1", "2026-09-02",
                                "insufficient_evidence", dsr_value=None,
                                evidence=_insufficient_evidence_payload("fam_x"))
    trial_ledger.record_outcome(path, "fam_y", "run-1", "2026-09-02",
                                "insufficient_evidence", dsr_value=None,
                                evidence=_insufficient_evidence_payload("fam_y"))
    values = [0.2, 0.5, 0.9]
    for i, v in enumerate(values):
        trial_ledger.record_outcome(path, "fam_scored_%d" % i, "run-1",
                                    "2026-09-02", "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=v)
    assert trial_ledger.pooled_sr_variance(path, min_samples=3) ==         pytest.approx(np.var(values, ddof=1))


def test_pooled_sr_variance_dedupes_by_family_keeping_the_latest(tmp_path):
    # A re-tested family (same underlying pinned data) must count ONCE, not
    # once per re-score -- otherwise a heavily re-tested family dominates the
    # pool's weight over families tested only once.
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.5, sharpe_in_sample=0.9)
    trial_ledger.record_outcome(path, "fam_a", "run-2", "2026-09-02",
                                "cut_dsr", dsr_value=0.3, sharpe_in_sample=0.4)
    for i, v in enumerate([0.1, 0.2, 0.6]):
        trial_ledger.record_outcome(path, "fam_%d" % i, "run-1", "2026-09-02",
                                    "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=v)
    got = trial_ledger.pooled_sr_variance(path, min_samples=4)
    # fam_a contributes ONLY its latest sharpe (0.4), not both 0.9 and 0.4.
    expected = np.var([0.4, 0.1, 0.2, 0.6], ddof=1)
    assert got == pytest.approx(expected)


def test_pooled_sr_variance_none_when_every_sharpe_is_identical(tmp_path):
    # Zero-variance pool is degenerate, same guard as every existing
    # sr_variance computation in this codebase (`if variance > 0`).
    path = str(tmp_path / "ledger.jsonl")
    for i in range(5):
        trial_ledger.record_outcome(path, "fam_%d" % i, "run-1", "2026-09-02",
                                    "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=0.5)
    assert trial_ledger.pooled_sr_variance(path, min_samples=5) is None


def test_pooled_sr_variance_never_writes_to_the_ledger(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    for i, v in enumerate([0.2, 0.5, 0.9, 0.1, 0.6]):
        trial_ledger.record_outcome(path, "fam_%d" % i, "run-1", "2026-09-02",
                                    "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=v)
    before = open(path, "rb").read()
    trial_ledger.pooled_sr_variance(path, min_samples=5)
    assert open(path, "rb").read() == before


def test_pooled_sr_variance_defaults_min_samples_to_the_pre_registered_constant(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    for i, v in enumerate([0.2, 0.5, 0.9, 0.1, 0.6, 0.3, 0.7, 0.4]):
        trial_ledger.record_outcome(path, "fam_%d" % i, "run-1", "2026-09-02",
                                    "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=v)
    assert len([0.2, 0.5, 0.9, 0.1, 0.6, 0.3, 0.7, 0.4]) < constants.MIN_POOLED_SR_VARIANCE_SAMPLES
    assert trial_ledger.pooled_sr_variance(path) is None
def test_pooled_sr_variance_ignores_non_finite_sharpes(tmp_path):
    # A NaN/inf sharpe_in_sample (a
    # corrupted or hand-edited ledger row) must not silently pollute the
    # pool's mean/variance arithmetic.
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_bad", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.1,
                                sharpe_in_sample=float("nan"))
    values = [0.2, 0.5, 0.9, 0.1, 0.6]
    for i, v in enumerate(values):
        trial_ledger.record_outcome(path, "fam_%d" % i, "run-1", "2026-09-02",
                                    "cut_dsr", dsr_value=0.1,
                                    sharpe_in_sample=v)
    got = trial_ledger.pooled_sr_variance(path, min_samples=5)
    assert got == pytest.approx(np.var(values, ddof=1))


def test_pooled_sr_variance_ignores_an_abi1_sharpe(tmp_path):
    """The load-bearing protection: an ABI-1
    row's sharpe_in_sample must never enter the per-trade pool. A non-null
    check alone would admit it, so the read loop must also key on the
    row's construction."""
    path = str(tmp_path / "ledger.jsonl")
    for i in range(10):
        trial_ledger.record_outcome(
            path, f"fam_{i}", "run-1", "2026-09-08", "cut_dsr",
            dsr_value=0.5, sharpe_in_sample=0.5 + 0.01 * i)
    baseline = trial_ledger.pooled_sr_variance(path)
    assert baseline is not None
    # An ABI-1 row with a wildly different (per-SESSION) sharpe -- if this
    # entered the same pool it would move the variance a great deal.
    trial_ledger.record_outcome(
        path, "fam_abi1", "run-1", "2026-09-08", "cut_dsr",
        dsr_value=0.1, sharpe_in_sample=0.02, construction="ABI-1")
    after = trial_ledger.pooled_sr_variance(path)
    assert after == pytest.approx(baseline)


def test_record_outcome_normalizes_non_finite_sharpe_to_none(tmp_path):
    # record_outcome is the SINGLE choke point every
    # scoring call site writes through, so guarding it here protects
    # all of them without touching each site's own control flow. A NaN/inf
    # sharpe_in_sample (e.g. a degenerate candidate whose trades all had
    # byte-identical returns, giving std=0) must never reach the ledger --
    # pooled_sr_variance's own read-side guard already filters it, but
    # normalizing at the write side keeps the ledger row itself honest
    # too, and protects any FUTURE reader that is not pooled_sr_variance.
    # dsr_value itself is finite here (0.1) on both rows -- the
    # evidence lifecycle separately REQUIRES a finite dsr_value on
    # every SCORED stage (see test_record_outcome_rejects_a_non_finite_dsr_
    # value_on_a_scored_stage below); this test is only about sharpe_in_
    # sample's own normalization, which is independent of that guard.
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_nan", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.1,
                                sharpe_in_sample=float("nan"))
    trial_ledger.record_outcome(path, "fam_inf", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.1,
                                sharpe_in_sample=float("inf"))
    events = trial_ledger.read_all(path)
    assert events[0]["sharpe_in_sample"] is None
    assert events[1]["sharpe_in_sample"] is None


def test_record_outcome_still_persists_a_finite_sharpe(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.1,
                                sharpe_in_sample=0.77)
    events = trial_ledger.read_all(path)
    assert events[0]["sharpe_in_sample"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Evidence lifecycle: record_outcome is THE chokepoint
# both scoring paths cross -- the
# stage whitelist is evidence.VALID_STAGES and the biconditional
# evidence.py exists to make enforceable is checked HERE, on every write:
#   stage in SCORED_STAGES   <=> dsr_value is a finite float AND evidence is None
#   stage in UNSCORED_STAGES <=> dsr_value is None           AND evidence is valid
# ---------------------------------------------------------------------------

def test_record_outcome_accepts_insufficient_evidence_with_a_valid_payload(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        path, "fam_x", "run-1", "2026-09-02", "insufficient_evidence",
        dsr_value=None, evidence=_insufficient_evidence_payload())
    events = trial_ledger.read_all(path)
    assert events[0]["stage"] == "insufficient_evidence"
    assert events[0]["evidence"]["reason_code"] == "insufficient_trades"


def test_record_outcome_accepts_not_evaluated_error_with_a_valid_payload(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(
        path, "fam_x", "run-1", "2026-09-02", "not_evaluated_error",
        dsr_value=None, evidence=_engine_error_payload())
    events = trial_ledger.read_all(path)
    assert events[0]["stage"] == "not_evaluated_error"
    assert events[0]["evidence"]["reason_code"] == "engine_error"


def test_record_outcome_still_rejects_an_unknown_stage(tmp_path):
    # The whitelist is evidence.VALID_STAGES, not "anything".
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-08-27",
                                    "bogus", dsr_value=0.5)


def test_record_outcome_rejects_insufficient_evidence_without_a_payload(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(
            path, "fam_x", "run-1", "2026-09-02", "insufficient_evidence",
            dsr_value=None)


def test_record_outcome_rejects_insufficient_evidence_with_a_dsr_value(tmp_path):
    # A fabricated DSR on a deferred outcome is exactly the ambiguity this
    # lifecycle exists to make impossible to write.
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(
            path, "fam_x", "run-1", "2026-09-02", "insufficient_evidence",
            dsr_value=0.5, evidence=_insufficient_evidence_payload())


def test_record_outcome_rejects_a_scored_stage_carrying_an_evidence_payload(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(
            path, "fam_x", "run-1", "2026-09-02", "cut_dsr",
            dsr_value=0.5, evidence=_insufficient_evidence_payload())


def test_record_outcome_rejects_a_scored_stage_with_no_dsr_value(tmp_path):
    # A cut_dsr row with dsr_value=None would record an unscored family
    # indistinguishably from a real statistical rejection -- refused at the
    # one chokepoint every scoring path writes through.
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(
            path, "fam_x", "run-1", "2026-09-02", "cut_dsr", dsr_value=None)


def test_record_outcome_rejects_a_non_finite_dsr_value_on_a_scored_stage(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(
            path, "fam_x", "run-1", "2026-09-02", "survived",
            dsr_value=float("nan"))


def test_record_outcome_rejects_an_invalid_evidence_payload_shape(tmp_path):
    # validate() is applied at write time -- a payload with a missing key
    # must be caught here, not merely accepted and left for a reader to
    # discover.
    path = str(tmp_path / "ledger.jsonl")
    bad_payload = _insufficient_evidence_payload()
    del bad_payload["reason_code"]
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(
            path, "fam_x", "run-1", "2026-09-02", "insufficient_evidence",
            dsr_value=None, evidence=bad_payload)


def test_record_outcome_rejects_a_fabricated_top_level_sharpe_on_an_unscored_stage(tmp_path):
    # A real per-trade sharpe can live inside evidence.observed even for a
    # deferral (evidence.py's own schema allows it), but the LEDGER ROW's
    # own top-level sharpe_in_sample must stay None on every unscored
    # outcome -- record_outcome's docstring documents why a fabricated
    # value there would corrupt pooled_sr_variance's pool.
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        trial_ledger.record_outcome(
            path, "fam_x", "run-1", "2026-09-02", "insufficient_evidence",
            dsr_value=None, evidence=_insufficient_evidence_payload(),
            sharpe_in_sample=0.5)


def test_record_outcome_persists_the_evidence_payload_verbatim(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    payload = _insufficient_evidence_payload(trade_count=7)
    trial_ledger.record_outcome(
        path, "fam_x", "run-1", "2026-09-02", "insufficient_evidence",
        dsr_value=None, evidence=payload)
    events = trial_ledger.read_all(path)
    assert events[0]["evidence"] == payload


def test_record_outcome_writes_evidence_key_as_none_on_a_scored_stage(tmp_path):
    # Uniform key set, matching note/ticker_label/sharpe_in_sample's
    # own convention -- a reader should never need to
    # branch on whether the key exists, only on its value.
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-02",
                                "cut_dsr", dsr_value=0.4)
    events = trial_ledger.read_all(path)
    assert events[0]["evidence"] is None


# ---------------------------------------------------------------------------
# construction tag: the ledger schema
# receives an ABI-1 outcome (construction="ABI-1") without polluting the
# per-trade pooled_sr_variance pool above.
# ---------------------------------------------------------------------------

def test_record_outcome_defaults_construction_to_per_trade(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-08",
                                "survived", dsr_value=0.97,
                                sharpe_in_sample=0.9)
    rows = trial_ledger.read_all(path)
    assert rows[-1]["construction"] == "per_trade"


def test_record_outcome_accepts_an_explicit_abi1_construction(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-08",
                                "survived", dsr_value=0.97,
                                sharpe_in_sample=0.9, construction="ABI-1")
    rows = trial_ledger.read_all(path)
    assert rows[-1]["construction"] == "ABI-1"


def test_record_outcome_rejects_an_unknown_construction(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="construction"):
        trial_ledger.record_outcome(path, "fam_a", "run-1", "2026-09-08",
                                    "survived", dsr_value=0.97,
                                    sharpe_in_sample=0.9,
                                    construction="daily_something_else")


# ---------------------------------------------------------------------------
# record_outcome_if_not_scored: the atomic check-and-append counterpart to
# record_outcome, closing the TOCTOU between an ABI-1 driver's own early,
# unlocked `already_scored` prefilter (an optimization to skip expensive
# scoring work, run BEFORE this function is ever called) and the eventual
# outcome append. Mirrors
# record_first_test's own check-then-act pattern (this module's docstring,
# "Cross-process locking"), just applied at the OUTCOME write.
# ---------------------------------------------------------------------------

def test_record_outcome_if_not_scored_records_and_returns_true_first_time(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    recorded = trial_ledger.record_outcome_if_not_scored(
        path, "fam_a", "run-1", "2026-09-22", "cut_dsr",
        dsr_value=0.4, sharpe_in_sample=0.1, construction="ABI-1")
    assert recorded is True
    rows = [r for r in trial_ledger.read_all(path) if r["event"] == "outcome"]
    assert len(rows) == 1
    assert rows[0]["family_id"] == "fam_a"
    assert rows[0]["construction"] == "ABI-1"
    assert rows[0]["stage"] == "cut_dsr"


def test_record_outcome_if_not_scored_no_ops_and_returns_false_when_already_scored(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    first = trial_ledger.record_outcome_if_not_scored(
        path, "fam_a", "run-1", "2026-09-22", "cut_dsr",
        dsr_value=0.4, sharpe_in_sample=0.1, construction="ABI-1")
    assert first is True

    second = trial_ledger.record_outcome_if_not_scored(
        path, "fam_a", "run-2", "2026-09-22", "survived",
        dsr_value=0.99, sharpe_in_sample=0.9, construction="ABI-1")
    assert second is False

    rows = [r for r in trial_ledger.read_all(path) if r["event"] == "outcome"]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"          # the second call never wrote
    assert rows[0]["stage"] == "cut_dsr"


def test_record_outcome_if_not_scored_checks_the_exact_construction(tmp_path):
    """A prior PER-TRADE outcome for the same family must not block an ABI-1
    attempt -- the guard keys on (family_id, construction), matching
    run_abi1_mining.py's own re-run guard semantics
    (test_re_run_guard_ignores_a_pre_trade_construction_row_for_the_same_family)."""
    path = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_outcome(path, "fam_a", "legacy-run", "2026-08-01",
                                "cut_dsr", dsr_value=0.5, sharpe_in_sample=0.01,
                                construction="per_trade")

    recorded = trial_ledger.record_outcome_if_not_scored(
        path, "fam_a", "run-1", "2026-09-22", "cut_dsr",
        dsr_value=0.4, sharpe_in_sample=0.1, construction="ABI-1")
    assert recorded is True
    outcomes = [r for r in trial_ledger.read_all(path) if r["event"] == "outcome"]
    assert len(outcomes) == 2
    assert outcomes[-1]["construction"] == "ABI-1"


def test_record_outcome_if_not_scored_validates_before_touching_the_lock(tmp_path):
    """A malformed payload must raise the SAME ValueError record_outcome
    raises, and must never create the ledger file -- validation happens
    before the lock is even acquired (build-then-lock, not lock-then-build),
    so a bad call can never block a concurrent, well-formed one."""
    path = str(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="finite dsr_value"):
        trial_ledger.record_outcome_if_not_scored(
            path, "fam_a", "run-1", "2026-09-22", "cut_dsr",
            dsr_value=None, construction="ABI-1")
    assert not os.path.exists(path)


def test_concurrent_record_outcome_if_not_scored_only_one_wins_the_race(tmp_path, monkeypatch):
    """The actual race this function exists to close: two callers (e.g. two
    concurrent invocations of the same intraday ABI-1 driver) that both
    already passed their own early already_scored prefilter and both reach
    this function at nearly the same instant. Widen the window between the
    re-check read and the append (an injected sleep, same technique as
    test_concurrent_record_first_test_only_one_wins_the_race above) so a
    barrier-released pair of threads is virtually guaranteed to overlap it
    if the lock does not serialize them -- then prove it does."""
    path = str(tmp_path / "ledger.jsonl")
    original_read_lines = trial_ledger._read_lines

    def slow_read_lines(p):
        result = original_read_lines(p)
        time.sleep(0.1)
        return result

    monkeypatch.setattr(trial_ledger, "_read_lines", slow_read_lines)

    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(run_id):
        barrier.wait()
        recorded = trial_ledger.record_outcome_if_not_scored(
            path, "fam_race", run_id, "2026-09-22", "cut_dsr",
            dsr_value=0.4, sharpe_in_sample=0.1, construction="ABI-1")
        with results_lock:
            results.append(recorded)

    threads = [threading.Thread(target=worker, args=(f"run-{i}",))
               for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [False, True]
    outcomes = [r for r in trial_ledger.read_all(path)
                if r["event"] == "outcome" and r["family_id"] == "fam_race"
                and r.get("construction") == "ABI-1"]
    assert len(outcomes) == 1


def test_ledger_lock_refuses_clearly_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(trial_ledger.sys, "platform", "linux")
    with pytest.raises(trial_ledger.LedgerLockUnsupported, match="Windows"):
        with trial_ledger._LedgerLock(str(tmp_path / "nested" / "ledger.jsonl")):
            pass
    assert not (tmp_path / "nested").exists()
