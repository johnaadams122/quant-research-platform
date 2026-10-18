import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch3_mining as batch3
import run_swing_mining as swing
from mining import constants, trial_ledger
from mining.intraday_backtest_engine import SignalResult, run_intraday_backtest


_ET = "America/New_York"


# ---------------------------------------------------------------------------
# Synthetic data helpers -- no network, no real ledger, ever.
# ---------------------------------------------------------------------------

def _swing_frame(start_date: str, end_date: str, bars_per_day: int = 2,
                  base: float = 100.0, trend: float = 0.10) -> pd.DataFrame:
    """A synthetic RTH-shaped minute-bar frame spanning every business day
    in [start_date, end_date] (matching intraday_data.fetch_intraday_bars's
    output shape: tz-aware "America/New_York" DatetimeIndex,
    Open/High/Low/Close columns). Close increases monotonically bar-over-bar
    (real, nonzero, trending returns) unless a small oscillation is added by
    the caller."""
    days = pd.bdate_range(start=start_date, end=end_date)
    times = ["09:30", "15:59"][:bars_per_day]
    idx = []
    for day in days:
        for t in times:
            idx.append(pd.Timestamp(f"{day.date()} {t}", tz=_ET))
    index = pd.DatetimeIndex(idx)
    n = len(index)
    values = base + np.arange(n, dtype=float) * trend
    return pd.DataFrame({
        "Open": values, "High": values + 0.05, "Low": values - 0.05,
        "Close": values,
    }, index=index)


def _daily_roundtrip_signal(frame, i):
    """Round-trips every day: enters at each day's open, exits at that same
    day's close -- relies on the frame having EXACTLY 2 bars/day starting at
    absolute index 0 (see _swing_frame), so "last bar of day" <=> i % 2 == 1."""
    flag = (i % 2 == 1)
    return SignalResult(entry=flag, exit=flag)


def _index_signal(entry_exec_i: int, exit_exec_i: int):
    """One trade: opens executing at absolute bar `entry_exec_i`, closes
    executing at absolute bar `exit_exec_i` -- pure function of bar
    position, safe to reuse across two independently-fetched frames."""
    def signal_fn(frame, i):
        if i == entry_exec_i - 1:
            return SignalResult(entry=True, exit=False)
        if i == exit_exec_i:
            return SignalResult(entry=False, exit=True)
        return SignalResult(entry=False, exit=False)
    return signal_fn


def _make_dispatch_fetch_fn(frames_by_range: dict):
    """A fake fetch_fn that returns a pre-built frame keyed by the EXACT
    (symbol, start_date, end_date) the orchestrator calls it with -- so a
    test can hand back a DIFFERENT canned frame for the in-sample fetch vs.
    the padded-holdout fetch. Raises loudly (AssertionError) on an
    unexpected call instead of silently returning something wrong."""
    def fetch_fn(symbol, start_date, end_date):
        key = (symbol, start_date, end_date)
        if key not in frames_by_range:
            raise AssertionError(
                f"unexpected fetch_fn call {key}; known keys: "
                f"{sorted(frames_by_range)}")
        return frames_by_range[key]
    return fetch_fn


_TIGHT_QUOTE = {"entry_bid": 99.99, "entry_ask": 100.01,
                "exit_bid": 99.99, "exit_ask": 100.01}  # ~2 bps spread
_WIDE_QUOTE = {"entry_bid": 99.0, "entry_ask": 101.0,
               "exit_bid": 99.0, "exit_ask": 101.0}      # ~200 bps spread


def _base_candidate(family_id="swing_family", tickers=("FAKE",),
                     signal_fn=_daily_roundtrip_signal, in_sample_start="2025-10-01",
                     in_sample_end="2025-11-30", **overrides):
    candidate = {
        "family_id": family_id, "tickers": list(tickers), "signal_fn": signal_fn,
        "cost_bps": 0.0, "tax_haircut_rate": 0.0, "max_hold_days": 5,
        "direction": "long", "account_type": "taxable",
        "in_sample_start": in_sample_start, "in_sample_end": in_sample_end,
    }
    candidate.update(overrides)
    return candidate


def _stable_dsr(**_kwargs):
    return 0.97  # comfortably above constants.DSR_PASS_THRESHOLD (0.95)


# ---------------------------------------------------------------------------
# 1. Candidate that clears the cost gate, full pipeline to a recorded outcome
# ---------------------------------------------------------------------------

def test_gate_passing_candidate_reaches_a_recorded_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")

    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])

    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    result = swing.run_swing_batch(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-1", ledger_path=ledger,
        dry_run=False, fetch_fn=fetch_fn)

    assert result["gate_exclusions"] == []
    assert len(result["results"]) == 1
    assert result["results"][0]["family_id"] == "swing_family"
    assert result["results"][0]["stage"] in ("survived", "cut_dsr", "cut_holdout")

    records = trial_ledger.read_all(ledger)
    events = [(r["event"], r["family_id"]) for r in records]
    assert events == [("first_test", "swing_family"), ("outcome", "swing_family")]
    assert trial_ledger.latest_outcome(ledger, "swing_family") == result["results"][0]["stage"]


def test_gate_passing_candidate_positive_edge_survives_with_stable_dsr(tmp_path, monkeypatch):
    """A concrete, unambiguous happy path: trending-up daily round trips in
    BOTH in-sample and holdout, DSR stabilized above the pass bar -> the
    candidate must reach the passing stage, not just some (possibly cut) stage."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")

    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    result = swing.run_swing_batch(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-1", ledger_path=ledger,
        dry_run=False, fetch_fn=fetch_fn)

    assert result["results"][0]["stage"] == "survived"


def test_missing_quote_excludes_only_that_candidate_not_the_whole_batch(
        tmp_path, monkeypatch):
    """Regression test: if compute_swing_candidates did
    `quotes[family_id]` unguarded, ONE candidate missing a live quote would
    raise an unhandled KeyError and crash the entire batch -- including a second, perfectly fine candidate that
    had nothing to do with the missing quote. A missing quote must be a
    logged exclusion for THAT candidate only, matching every other
    "we cannot proceed without a real input" skip in this pipeline."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")

    frames = {}
    candidates = []
    for name, ticker in (("has_quote", "AAA"), ("no_quote", "BBB")):
        in_sample = _swing_frame("2025-10-01", "2025-11-30")
        holdout = _swing_frame(padded_start, window["holdout_end"])
        frames[(ticker, "2025-10-01", "2025-11-30")] = in_sample
        frames[(ticker, padded_start, window["holdout_end"])] = holdout
        candidates.append(_base_candidate(family_id=name, tickers=(ticker,)))
    fetch_fn = _make_dispatch_fetch_fn(frames)

    # Deliberately NO entry for "no_quote" in the quotes dict.
    quotes = {"has_quote": _TIGHT_QUOTE}

    result = swing.run_swing_batch(
        candidates, quotes=quotes, tested_date="2026-06-15", run_id="run-1",
        ledger_path=ledger, dry_run=False, fetch_fn=fetch_fn)

    # The candidate WITH a quote was scored normally and reached the ledger.
    assert len(result["results"]) == 1
    assert result["results"][0]["family_id"] == "has_quote"
    assert trial_ledger.has_been_tested(ledger, "has_quote")

    # The candidate WITHOUT a quote is a logged exclusion, never pinned.
    exclusion_ids = {e["family_id"] for e in result["gate_exclusions"]}
    assert exclusion_ids == {"no_quote"}
    no_quote_exclusion = next(
        e for e in result["gate_exclusions"] if e["family_id"] == "no_quote")
    assert "quote" in no_quote_exclusion["reason"]
    assert not trial_ledger.has_been_tested(ledger, "no_quote")


# ---------------------------------------------------------------------------
# 2. Candidate that fails the cost gate never reaches record_first_test
# ---------------------------------------------------------------------------

def test_gate_failing_candidate_never_reaches_record_first_test(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
    })
    candidate = _base_candidate()

    result = swing.run_swing_batch(
        [candidate], quotes={"swing_family": _WIDE_QUOTE},
        tested_date="2026-06-15", run_id="run-1", ledger_path=ledger,
        dry_run=False, fetch_fn=fetch_fn)

    assert result["results"] == []
    assert len(result["gate_exclusions"]) == 1
    exclusion = result["gate_exclusions"][0]
    assert exclusion["family_id"] == "swing_family"
    assert "spread" in exclusion["reason"]
    assert exclusion["spread_bps"] >= constants.INTRADAY_SPREAD_CEILING_BPS

    # No ledger file was even created -- nothing was ever pinned.
    assert trial_ledger.read_all(ledger) == []
    assert not trial_ledger.has_been_tested(ledger, "swing_family")


def test_gate_failing_candidate_never_fetches_the_padded_holdout_window(tmp_path):
    """A gate failure must skip the (expensive) holdout fetch entirely --
    if the orchestrator ever called fetch_fn for the padded range, the
    dispatch fake below raises AssertionError since only the in-sample key
    is registered."""
    ledger = str(tmp_path / "ledger.jsonl")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
    })
    candidate = _base_candidate()
    swing.run_swing_batch(
        [candidate], quotes={"swing_family": _WIDE_QUOTE},
        tested_date="2026-06-15", run_id="run-1", ledger_path=ledger,
        dry_run=False, fetch_fn=fetch_fn)  # would raise if holdout were fetched


def test_expected_holdout_sessions_returns_every_real_nyse_session_in_range():
    window = batch3.derive_holdout_window("2026-06-15")
    got = swing._expected_holdout_sessions(
        window["holdout_start"], window["holdout_end"])
    sessions = swing.trading_calendar.load_default_sessions()
    start = pd.Timestamp(window["holdout_start"]).date()
    end = pd.Timestamp(window["holdout_end"]).date()
    expected = [s for s in sessions if start <= s <= end]
    assert got == expected
    assert got == sorted(got)  # ascending
    assert got[-1] == max(expected)
    assert len(got) > 100  # a real 6-month window has well over 100 sessions


def test_incomplete_holdout_fetch_excludes_candidate_not_scored_as_complete(
        tmp_path, monkeypatch):
    """Regression test: `holdout_complete` must not be set to True UNCONDITIONALLY, with
    nothing checking the fetched padded-holdout data actually reached
    through `holdout_end`. A truncated/stale fetch (simulating a real
    Alpaca outage partway through the window) must be excluded with a
    real, specific reason -- never silently scored as a complete holdout
    evaluation, and never reaching record_first_test."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")

    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    # Truncated holdout fetch: the orchestrator ASKS for data through
    # window["holdout_end"], but the fake fetch_fn -- simulating a real
    # feed that silently stopped partway through the window -- returns
    # rows only through 10 calendar days BEFORE that, a genuinely short
    # response, not a legitimate "no trades near the end" outcome.
    truncated_end = (pd.Timestamp(window["holdout_end"])
                      - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    truncated_holdout_frame = _swing_frame(padded_start, truncated_end)

    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): truncated_holdout_frame,
    })
    candidate = _base_candidate()

    result = swing.run_swing_batch(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-1", ledger_path=ledger,
        dry_run=False, fetch_fn=fetch_fn)

    assert result["results"] == []
    assert len(result["gate_exclusions"]) == 1
    exclusion = result["gate_exclusions"][0]
    assert exclusion["family_id"] == "swing_family"
    assert "incomplete" in exclusion["reason"]
    assert "swing_family" not in {r["family_id"] for r in result["results"]}
    assert not trial_ledger.has_been_tested(ledger, "swing_family")


def test_holdout_fetch_reaching_exactly_the_last_session_counts_as_complete(
        tmp_path, monkeypatch):
    """The completeness check is inclusive, not off-by-one: data reaching
    EXACTLY the expected last real holdout session (not one day short) must
    be treated as complete and scored normally -- proven against the SAME
    fixtures every other happy-path test in this file already uses (they
    all pass the completeness check unmodified, but this test makes the
    boundary itself the explicit point)."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    result = swing.run_swing_batch(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-1", ledger_path=ledger,
        dry_run=False, fetch_fn=fetch_fn)

    assert result["gate_exclusions"] == []
    assert len(result["results"]) == 1
    assert result["results"][0]["family_id"] == "swing_family"


def test_interior_holdout_gap_excludes_candidate_not_scored_as_complete(
        tmp_path, monkeypatch):
    """Regression test, companion to the tail-truncation case: a
    completeness check that only validates the
    LAST expected session still passes for a realistic INTERIOR gap -- a
    real outage for several sessions in the MIDDLE of the holdout window,
    with real data present both before and after it. The backtest would
    otherwise silently run "across" the gap as if it were continuous
    history, and the candidate would still be marked
    holdout_complete=True. Must be excluded with a real, specific reason
    instead, the same way a tail-truncated fetch already is."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")

    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")

    # A genuine 5-session gap squarely in the MIDDLE of the real (NYSE
    # calendar-derived, not naive-weekday) expected holdout sessions --
    # not at either end -- simulating a real interior data outage.
    expected_sessions = swing._expected_holdout_sessions(
        window["holdout_start"], window["holdout_end"])
    mid = len(expected_sessions) // 2
    gap_days = expected_sessions[mid:mid + 5]
    assert len(gap_days) == 5, (
        "test assumption broken: expected sessions too short for a "
        "5-session interior gap")
    # The gap must have real coverage on both sides to actually be
    # "interior" (not accidentally touching either boundary).
    assert gap_days[0] > expected_sessions[0]
    assert gap_days[-1] < expected_sessions[-1]

    full_holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    gap_mask = np.isin(full_holdout_frame.index.date, list(gap_days))
    gapped_holdout_frame = full_holdout_frame.loc[~gap_mask]
    # Sanity check on the fixture itself: the gap really did remove rows,
    # and rows genuinely remain both before and after it.
    assert len(gapped_holdout_frame) < len(full_holdout_frame)
    remaining_days = set(pd.unique(gapped_holdout_frame.index.date))
    assert any(d < gap_days[0] for d in remaining_days)
    assert any(d > gap_days[-1] for d in remaining_days)

    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): gapped_holdout_frame,
    })
    candidate = _base_candidate()

    result = swing.run_swing_batch(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-1", ledger_path=ledger,
        dry_run=False, fetch_fn=fetch_fn)

    assert result["results"] == []
    assert len(result["gate_exclusions"]) == 1
    exclusion = result["gate_exclusions"][0]
    assert exclusion["family_id"] == "swing_family"
    assert "missing" in exclusion["reason"]
    assert "incomplete" in exclusion["reason"]
    # The specific missing dates are named, not just a generic message.
    assert gap_days[0].isoformat() in exclusion["reason"]
    assert not trial_ledger.has_been_tested(ledger, "swing_family")


# ---------------------------------------------------------------------------
# 3. Batching: two candidates in one call share ONE compute_shared_correction
# ---------------------------------------------------------------------------

def test_batching_two_candidates_share_one_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")

    frames = {}
    candidates = []
    # trend is scaled proportionally to base so BOTH candidates have the
    # SAME ~10 bps/day relative edge regardless of price level -- otherwise
    # a fixed absolute trend (e.g. 0.10) is a much SMALLER relative edge at
    # a higher base price and can fail the cost gate for one candidate but
    # not the other, which would defeat this test's actual point (both
    # candidates must reach `computed` to prove the shared-correction batching).
    for name, ticker, base in (("fam_a", "AAA", 100.0), ("fam_b", "BBB", 200.0)):
        trend = base * 0.001
        in_sample = _swing_frame("2025-10-01", "2025-11-30", base=base, trend=trend)
        holdout = _swing_frame(padded_start, window["holdout_end"], base=base, trend=trend)
        frames[(ticker, "2025-10-01", "2025-11-30")] = in_sample
        frames[(ticker, padded_start, window["holdout_end"])] = holdout
        candidates.append(_base_candidate(family_id=name, tickers=(ticker,)))

    fetch_fn = _make_dispatch_fetch_fn(frames)
    quotes = {"fam_a": _TIGHT_QUOTE, "fam_b": _TIGHT_QUOTE}

    combined = swing.run_swing_batch(
        candidates, quotes=quotes, tested_date="2026-06-15", run_id="run-batch",
        ledger_path=ledger, dry_run=True, fetch_fn=fetch_fn)

    # Compute the SAME computed items independently and call
    # compute_shared_correction directly -- the two must agree exactly.
    batch = swing.compute_swing_candidates(
        candidates, quotes=quotes, tested_date="2026-06-15",
        ledger_path=ledger, fetch_fn=fetch_fn)
    direct_correction = batch3.compute_shared_correction(batch["computed"], ledger)

    assert combined["shared_correction"] == direct_correction
    assert combined["shared_correction"]["computed_family_count"] == 2
    for item in combined["results"]:
        assert item["shared_n_trials"] == direct_correction["n_trials"]
        assert item["shared_sr_variance"] == direct_correction["sr_variance"]


# ---------------------------------------------------------------------------
# 4. Exception mid-pipeline after pinning -> defensive outcome for every pin
# ---------------------------------------------------------------------------

def test_mid_scoring_exception_defensively_outcomes_every_new_pin(tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")

    frames = {}
    candidates = []
    for name, ticker in (("fam_a", "AAA"), ("fam_b", "BBB"), ("fam_c", "CCC")):
        in_sample = _swing_frame("2025-10-01", "2025-11-30")
        holdout = _swing_frame(padded_start, window["holdout_end"])
        frames[(ticker, "2025-10-01", "2025-11-30")] = in_sample
        frames[(ticker, padded_start, window["holdout_end"])] = holdout
        candidates.append(_base_candidate(family_id=name, tickers=(ticker,)))
    fetch_fn = _make_dispatch_fetch_fn(frames)
    quotes = {name: _TIGHT_QUOTE for name in ("fam_a", "fam_b", "fam_c")}

    count = {"value": 0}

    def fail_second(**_kwargs):
        count["value"] += 1
        if count["value"] == 2:
            raise RuntimeError("injected mid-pipeline failure")
        return 0.5

    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", fail_second)

    with pytest.raises(RuntimeError, match="injected mid-pipeline failure"):
        swing.run_swing_batch(
            candidates, quotes=quotes, tested_date="2026-06-15", run_id="run-fail",
            ledger_path=ledger, dry_run=False, fetch_fn=fetch_fn)

    records = trial_ledger.read_all(ledger)
    pinned = {r["family_id"] for r in records if r["event"] == "first_test"}
    outcomes = {r["family_id"] for r in records if r["event"] == "outcome"}
    assert pinned == {"fam_a", "fam_b", "fam_c"}
    assert outcomes == pinned
    # fam_a scored successfully BEFORE the mid-batch failure (a real,
    # below-bar DSR) -- a genuine DSR cut. fam_b (whose own scoring call
    # raised) and fam_c (never reached) are defensively recorded as
    # not_evaluated_error, not the SAME stage string fam_a carries
    # (the evidence-lifecycle rule, as in run_batch3_mining; this script
    # uses _score_one/_record_defensive_outcomes from it unmodified).
    assert trial_ledger.latest_outcome(ledger, "fam_a") == "cut_dsr"
    assert trial_ledger.latest_outcome(ledger, "fam_b") == "not_evaluated_error"
    assert trial_ledger.latest_outcome(ledger, "fam_c") == "not_evaluated_error"


# ---------------------------------------------------------------------------
# 5. Holdout-boundary reconciliation, exercised through the full orchestrator
# ---------------------------------------------------------------------------

def test_position_opened_during_padding_survives_reconciliation_through_orchestrator(
        tmp_path, monkeypatch):
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    buffer_days = swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    # In-sample: an ordinary, well-separated trending window.
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")

    # Padded-holdout: build the frame first, then derive entry/exit bar
    # indices from ITS OWN actual business-day calendar -- entry deep in
    # the padding period, exit a few real business days into the holdout
    # window, so the trade genuinely spans the boundary.
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    all_days = sorted(pd.unique(holdout_frame.index.date))
    holdout_start_date = pd.Timestamp(window["holdout_start"]).date()
    holdout_start_idx = next(
        i for i, d in enumerate(all_days) if d >= holdout_start_date)
    assert holdout_start_idx >= 5, (
        "test assumption broken: expected several real padding days before "
        "holdout_start given a 90-day buffer")
    d_entry = 1
    d_exit = holdout_start_idx + 3
    assert d_exit < len(all_days) - 1, (
        "test assumption broken: holdout window too narrow for this offset")
    entry_exec_i = 2 * d_entry
    exit_exec_i = 2 * d_exit
    holdout_signal = _index_signal(entry_exec_i, exit_exec_i)

    # The orchestrator uses ONE shared signal_fn per candidate for BOTH
    # fetches (in-sample and padded-holdout). `holdout_signal`'s indices are
    # sized for the (long, ~90-day-padded) holdout frame; applied to the
    # much shorter in-sample frame it simply never reaches its exit index,
    # so the position is force-closed at end_of_data instead -- still
    # produces exactly one trade with a real, positive edge (price is
    # monotonically trending up throughout), which is all the cost gate
    # needs to pass. This keeps the test's ONLY point of interest --
    # reconciliation across the padding/holdout boundary -- isolated to the
    # holdout leg, without inventing a second signal_fn the orchestrator's
    # real one-signal-per-candidate contract wouldn't actually support.
    fetch_fn_in_sample = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate(
        family_id="boundary_family", tickers=("FAKE",),
        signal_fn=holdout_signal, max_hold_days=200)

    batch = swing.compute_swing_candidates(
        [candidate], quotes={"boundary_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", ledger_path=ledger,
        fetch_fn=fetch_fn_in_sample)

    assert batch["gate_exclusions"] == []
    assert len(batch["computed"]) == 1
    item = batch["computed"][0]
    # The pooled holdout trades must show real activity: at least one trade
    # whose exit lands inside the real holdout window, mirroring the
    # dedicated intraday_holdout reconciliation test's own assertion that a
    # position opened during padding and held into holdout is counted, not
    # dropped or zeroed.
    assert item["stats_holdout"]["trade_count"] >= 1
    assert item["holdout_net_edge_bps"] != 0.0

    # Direct proof at the reconciliation layer for the SAME padded frame,
    # confirming the orchestrator's own holdout stats are consistent with
    # calling intraday_holdout.reconcile_holdout_window directly.
    trades_direct, daily_direct = run_intraday_backtest(
        holdout_frame, holdout_signal, max_hold_days=200)
    real_trades, real_daily = swing.intraday_holdout.reconcile_holdout_window(
        trades_direct, daily_direct, holdout_frame, window["holdout_start"],
        window["holdout_end"], cost_bps=candidate["cost_bps"],
        tax_haircut_rate=candidate["tax_haircut_rate"],
        direction=candidate["direction"])
    assert len(real_trades) == item["stats_holdout"]["trade_count"]
    held_day = all_days[holdout_start_idx + 1]
    assert real_daily.loc[held_day] != 0.0

    # The spanning trade must have been RE-PRICED, not kept at its original
    # (padding-inclusive) net_ret -- the orchestrator's own stats_holdout
    # net_edge_bps must match this directly-reconciled trade's net_ret
    # exactly (both derive from the SAME formula), and must differ from
    # the trade's pre-reconciliation net_ret.
    assert len(real_trades) == 1
    assert item["stats_holdout"]["net_edge_bps"] == pytest.approx(
        float(real_trades.iloc[0]["net_ret"]) * 1e4)
    assert real_trades.iloc[0]["net_ret"] != pytest.approx(
        trades_direct.iloc[0]["net_ret"])


# ---------------------------------------------------------------------------
# Two guarded behaviours: (A) has_been_tested-style gates must not
# permanently block a not_evaluated_error retry, and (B) instrument_set
# must come from the real ticker list, never a display label.
# ---------------------------------------------------------------------------

def test_compute_swing_candidates_excludes_a_family_with_a_real_prior_outcome(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    trial_ledger.record_first_test(
        ledger, "swing_family", "2026-06-01", window["holdout_start"],
        window["holdout_end"], [], "older-run")
    trial_ledger.record_outcome(
        ledger, "swing_family", "older-run", "2026-06-01", "cut_dsr", dsr_value=0.4)
    candidate = _base_candidate()

    batch = swing.compute_swing_candidates(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", ledger_path=ledger)

    assert batch["computed"] == []
    assert len(batch["gate_exclusions"]) == 1
    assert batch["gate_exclusions"][0]["family_id"] == "swing_family"


def test_compute_swing_candidates_excludes_a_not_evaluated_error_retry_at_a_different_window(
        tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    # A DIFFERENT window than derive_holdout_window("2026-06-15") produces.
    trial_ledger.record_first_test(
        ledger, "swing_family", "2026-01-01", "2025-01-01", "2025-06-30", [], "older-run")
    trial_ledger.record_outcome(
        ledger, "swing_family", "older-run", "2026-01-01", "not_evaluated_error",
        dsr_value=None,
        evidence=batch3.evidence.build_engine_error(
            error=RuntimeError("earlier bug"), instrument_set=["FAKE"],
            pinned_window={"holdout_start": "2025-01-01", "holdout_end": "2025-06-30"},
            holdout_look="unspent"))
    candidate = _base_candidate()

    batch = swing.compute_swing_candidates(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", ledger_path=ledger)

    assert batch["computed"] == []
    assert len(batch["gate_exclusions"]) == 1
    assert "window mismatch" in batch["gate_exclusions"][0]["reason"]


def test_compute_swing_candidates_allows_a_not_evaluated_error_retry_at_the_same_window(
        tmp_path, monkeypatch):
    # (A): a not_evaluated_error family is a caught DEFECT, not
    # evidence about the market -- it must be able to re-enter scoring once
    # the defect is fixed, so "never re-scored" holds only for real outcomes,
    # not unconditionally.
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    trial_ledger.record_first_test(
        ledger, "swing_family", "2026-06-01", window["holdout_start"],
        window["holdout_end"], [], "older-run")
    trial_ledger.record_outcome(
        ledger, "swing_family", "older-run", "2026-06-01", "not_evaluated_error",
        dsr_value=None,
        evidence=batch3.evidence.build_engine_error(
            error=RuntimeError("earlier bug"), instrument_set=["FAKE"],
            pinned_window={"holdout_start": window["holdout_start"],
                          "holdout_end": window["holdout_end"]},
            holdout_look="unspent"))

    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    batch = swing.compute_swing_candidates(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", ledger_path=ledger, fetch_fn=fetch_fn)

    assert batch["gate_exclusions"] == []
    assert len(batch["computed"]) == 1
    assert batch["computed"][0]["family_id"] == "swing_family"


def test_compute_swing_candidates_carries_real_tickers_independent_of_a_display_label(
        tmp_path, monkeypatch):
    # (B): `ticker_label` (used to build the display `ticker` string)
    # can be an arbitrary human-readable label, not a parseable ticker --
    # the real `tickers` list must be carried through `computed` separately
    # so _instrument_set_for never has to parse the label.
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate(ticker_label="a display label, not a ticker")

    batch = swing.compute_swing_candidates(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", ledger_path=ledger, fetch_fn=fetch_fn)

    assert batch["computed"][0]["ticker"] == "a display label, not a ticker"
    assert batch["computed"][0]["tickers"] == ["FAKE"]


# ---------------------------------------------------------------------------
# Additional coverage: validation, dry_run vs real, results-log regeneration
# ---------------------------------------------------------------------------

def test_validate_candidate_raises_on_missing_field():
    candidate = _base_candidate()
    del candidate["cost_bps"]
    with pytest.raises(ValueError, match="cost_bps"):
        swing._validate_candidate(candidate)


def test_validate_candidate_raises_on_empty_tickers():
    candidate = _base_candidate(tickers=())
    with pytest.raises(ValueError, match="non-empty"):
        swing._validate_candidate(candidate)


def test_in_sample_end_overlapping_holdout_start_raises_look_ahead_error(tmp_path):
    window = batch3.derive_holdout_window("2026-06-15")
    # in_sample_end deliberately AT holdout_start -- a leak, not a valid split.
    candidate = _base_candidate(
        in_sample_start="2025-06-01", in_sample_end=window["holdout_start"])
    with pytest.raises(ValueError, match="look-ahead"):
        swing.compute_swing_candidates(
            [candidate], quotes={"swing_family": _TIGHT_QUOTE},
            tested_date="2026-06-15", ledger_path=str(tmp_path / "ledger.jsonl"),
            fetch_fn=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("fetch_fn should never be called")))


def test_duplicate_family_id_raises_before_any_candidate_is_touched(tmp_path):
    """Regression test:score_batch's pin
    loop aborts the WHOLE batch and defensively records an outcome for every
    already-pinned family if a second candidate shares a family_id with an
    earlier one -- which would silently mis-record an UNRELATED candidate.
    Must fail loudly here, before any fetch/backtest work, not silently."""
    dup_a = _base_candidate(family_id="dup_family", tickers=("AAA",))
    dup_b = _base_candidate(family_id="dup_family", tickers=("BBB",))
    with pytest.raises(ValueError, match="dup_family"):
        swing.compute_swing_candidates(
            [dup_a, dup_b], quotes={"dup_family": _TIGHT_QUOTE},
            tested_date="2026-06-15", ledger_path=str(tmp_path / "ledger.jsonl"),
            fetch_fn=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("fetch_fn should never be called -- the "
                                "uniqueness check must run first")))


def test_already_tested_family_excluded_without_aborting_the_rest_of_the_batch(
        tmp_path, monkeypatch):
    """Regression test:score_batch's own
    pre-pin-loop check (`already_pinned = [... has_been_tested ...]; if
    already_pinned: raise RuntimeError(...)`) aborts the ENTIRE batch the
    instant a candidate list includes even one family the ledger ALREADY
    has a first_test event for -- a realistic scenario (a re-run of an
    overlapping candidate slate), not just an in-batch duplicate. Must be
    filtered out before any fetch/backtest work, without discarding the
    other, genuinely new candidate in the same call."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")

    # Pre-seed the ledger with a first_test for "already_tested_family",
    # simulating an EARLIER, separate run -- not something this batch itself
    # pinned.
    trial_ledger.record_first_test(
        ledger, "already_tested_family", "2026-01-01", "2025-07-01",
        "2026-01-01", [], "earlier-run")
    assert trial_ledger.has_been_tested(ledger, "already_tested_family")

    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidates = [
        _base_candidate(family_id="already_tested_family", tickers=("FAKE",)),
        _base_candidate(family_id="fresh_family", tickers=("FAKE",)),
    ]
    quotes = {"already_tested_family": _TIGHT_QUOTE, "fresh_family": _TIGHT_QUOTE}

    result = swing.run_swing_batch(
        candidates, quotes=quotes, tested_date="2026-06-15", run_id="run-1",
        ledger_path=ledger, dry_run=False, fetch_fn=fetch_fn)

    # The fresh candidate was scored normally -- NOT collaterally aborted.
    assert len(result["results"]) == 1
    assert result["results"][0]["family_id"] == "fresh_family"

    # The already-tested candidate is a clean, non-fetching exclusion.
    exclusion_ids = {e["family_id"] for e in result["gate_exclusions"]}
    assert exclusion_ids == {"already_tested_family"}
    already_tested_exclusion = next(
        e for e in result["gate_exclusions"]
        if e["family_id"] == "already_tested_family")
    assert "already pinned" in already_tested_exclusion["reason"]
    # Its PRE-EXISTING pin/outcome history is untouched (still exactly the
    # one first_test event this test seeded, no defensive outcome added).
    records = [r for r in trial_ledger.read_all(ledger)
               if r["family_id"] == "already_tested_family"]
    assert len(records) == 1
    assert records[0]["event"] == "first_test"


def test_already_tested_filtering_applies_in_dry_run_too(tmp_path):
    """The already-tested filter is NOT conditional on dry_run -- a family
    the pipeline already has a permanent outcome for is never re-scored
    even as a preview (score_batch's own already_pinned check IS skipped
    in dry_run, so without this the family would be silently re-scored as
    if new)."""
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(
        ledger, "already_tested_family", "2026-01-01", "2025-07-01",
        "2026-01-01", [], "earlier-run")

    candidate = _base_candidate(family_id="already_tested_family")
    result = swing.run_swing_batch(
        [candidate], quotes={"already_tested_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-dry", ledger_path=ledger,
        dry_run=True,
        fetch_fn=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("fetch_fn should never be called -- the "
                            "already-tested filter must run first, even "
                            "in dry_run")))
    assert result["results"] == []
    assert len(result["gate_exclusions"]) == 1
    assert result["gate_exclusions"][0]["family_id"] == "already_tested_family"


def test_one_candidates_data_error_does_not_abort_the_whole_batch(
        tmp_path, monkeypatch):
    """Regression test:a real data-fetch
    failure (e.g. an Alpaca gap) for ONE candidate must be a per-candidate
    skip with the real exception recorded as the reason -- not a crash that
    discards already-completed compute work for OTHER, unrelated
    candidates in the same batch."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")

    good_in_sample = _swing_frame("2025-10-01", "2025-11-30")
    good_holdout = _swing_frame(padded_start, window["holdout_end"])

    def flaky_fetch_fn(symbol, start_date, end_date):
        if symbol == "BROKEN":
            raise RuntimeError(
                "simulated Alpaca data gap carrying a fake secret "
                "APCA-API-KEY-ID=AKFAKE123 for BROKEN")
        key = (symbol, start_date, end_date)
        frames = {
            ("GOOD", "2025-10-01", "2025-11-30"): good_in_sample,
            ("GOOD", padded_start, window["holdout_end"]): good_holdout,
        }
        return frames[key]

    candidates = [
        _base_candidate(family_id="broken_family", tickers=("BROKEN",)),
        _base_candidate(family_id="good_family", tickers=("GOOD",)),
    ]
    quotes = {"broken_family": _TIGHT_QUOTE, "good_family": _TIGHT_QUOTE}

    result = swing.run_swing_batch(
        candidates, quotes=quotes, tested_date="2026-06-15", run_id="run-1",
        ledger_path=ledger, dry_run=False, fetch_fn=flaky_fetch_fn)

    # The good candidate was scored and reached the ledger normally.
    assert len(result["results"]) == 1
    assert result["results"][0]["family_id"] == "good_family"
    assert trial_ledger.has_been_tested(ledger, "good_family")

    # The broken candidate is a logged exclusion carrying the real
    # exception, never pinned.
    exclusion_ids = {e["family_id"] for e in result["gate_exclusions"]}
    assert exclusion_ids == {"broken_family"}
    broken_exclusion = next(
        e for e in result["gate_exclusions"] if e["family_id"] == "broken_family")
    # The exception TYPE name is a real, specific reason...
    assert "RuntimeError" in broken_exclusion["reason"]
    # ...but the raw message text (which could carry a secret from an
    # injected fetch_fn -- the leak this regression test guards against)
    # must NEVER appear in the durable reason.
    assert "AKFAKE123" not in broken_exclusion["reason"]
    assert "simulated Alpaca data gap" not in broken_exclusion["reason"]
    assert not trial_ledger.has_been_tested(ledger, "broken_family")


def test_zero_spread_quote_does_not_crash_json_serialization(tmp_path, monkeypatch):
    """Regression test: a zero-measured
    round-trip spread (a real, plausible captured quote where bid == ask)
    makes intraday_cost_gate's edge_over_cost_ratio mathematically
    infinite; json.dumps(..., allow_nan=False) rejects `inf` outright.
    execute_swing_run must still complete and write a valid output file
    even when this happens, rather than crashing after the ledger has
    already been correctly updated."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    research_root = str(tmp_path / "runs-root")
    results_log_path = str(tmp_path / "RESULTS-LOG.md")
    output_path = str(tmp_path / "out.json")
    os.makedirs(research_root, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("[]")

    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()
    zero_spread_quote = {"entry_bid": 100.0, "entry_ask": 100.0,
                          "exit_bid": 100.0, "exit_ask": 100.0}

    def fake_renderer(*, catalog_path, ledger_path, research_root, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# fake results log\n")

    payload = swing.execute_swing_run(
        [candidate], quotes={"swing_family": zero_spread_quote},
        tested_date="2026-06-15", run_id="run-zero-spread", ledger_path=ledger,
        dry_run=False, output_path=output_path, catalog_path=catalog_path,
        research_root=research_root, results_log_path=results_log_path,
        fetch_fn=fetch_fn, results_renderer=fake_renderer)

    assert payload["results"][0]["family_id"] == "swing_family"
    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        written = f.read()
    assert "Infinity" not in written  # non-finite values were sanitized
    import json as _json
    written_payload = _json.loads(written)
    assert written_payload["results"][0]["cost_gate"]["edge_over_cost_ratio"] is None


def test_dry_run_never_writes_to_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    result = swing.run_swing_batch(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-dry", ledger_path=ledger,
        dry_run=True, fetch_fn=fetch_fn)

    assert result["results"][0]["family_id"] == "swing_family"
    assert not os.path.exists(ledger)
    assert trial_ledger.read_all(ledger) == []


def test_execute_swing_run_regenerates_results_log_only_when_ledger_changes(
        tmp_path, monkeypatch):
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    research_root = str(tmp_path / "runs-root")
    results_log_path = str(tmp_path / "RESULTS-LOG.md")
    output_path = str(tmp_path / "out.json")
    os.makedirs(research_root, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("[]")

    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    renderer_calls = []

    def fake_renderer(*, catalog_path, ledger_path, research_root, out_path):
        renderer_calls.append(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# fake results log\n")

    payload = swing.execute_swing_run(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-exec", ledger_path=ledger,
        dry_run=False, output_path=output_path, catalog_path=catalog_path,
        research_root=research_root, results_log_path=results_log_path,
        fetch_fn=fetch_fn, results_renderer=fake_renderer)

    assert len(renderer_calls) == 1
    assert os.path.exists(output_path)
    assert payload["results"][0]["family_id"] == "swing_family"

    # A second dry_run call against the SAME (already-pinned) ledger must
    # not attempt to re-score (already pinned) -- verify it does not call
    # the renderer again is out of scope here (dry_run never mutates the
    # ledger), so instead confirm dry_run truly skips the renderer.
    renderer_calls.clear()
    swing.execute_swing_run(
        [], quotes={}, tested_date="2026-06-15", run_id="run-exec-2",
        ledger_path=ledger, dry_run=True, output_path=output_path,
        catalog_path=catalog_path, research_root=research_root,
        results_log_path=results_log_path, fetch_fn=fetch_fn,
        results_renderer=fake_renderer)
    assert renderer_calls == []


def test_execute_swing_run_regenerates_results_log_even_when_run_raises(
        tmp_path, monkeypatch):
    """Mirrors run_batch3_mining.execute_batch_run's own contract: a
    defensively-outcomed pin from a mid-run
    exception must still be reflected in the durable results log before the
    exception propagates -- the results-log refresh is NOT skipped just
    because the run itself failed."""
    ledger = str(tmp_path / "ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    research_root = str(tmp_path / "runs-root")
    results_log_path = str(tmp_path / "RESULTS-LOG.md")
    output_path = str(tmp_path / "out.json")
    os.makedirs(research_root, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("[]")

    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    def always_raise(**_kwargs):
        raise RuntimeError("injected DSR failure")

    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", always_raise)

    renderer_calls = []

    def fake_renderer(*, catalog_path, ledger_path, research_root, out_path):
        renderer_calls.append(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# fake results log\n")

    with pytest.raises(RuntimeError, match="injected DSR failure"):
        swing.execute_swing_run(
            [candidate], quotes={"swing_family": _TIGHT_QUOTE},
            tested_date="2026-06-15", run_id="run-exc", ledger_path=ledger,
            dry_run=False, output_path=output_path, catalog_path=catalog_path,
            research_root=research_root, results_log_path=results_log_path,
            fetch_fn=fetch_fn, results_renderer=fake_renderer)

    # The pin still happened (score_batch pins unconditionally first) and
    # was defensively outcomed -- the ledger DID change, so the renderer
    # must have been called despite the run itself failing. The sole
    # candidate never reached a real DSR (its own scoring call is what
    # raised), so it is not_evaluated_error, not a DSR cut
    # (the evidence-lifecycle rule).
    assert trial_ledger.latest_outcome(ledger, "swing_family") == "not_evaluated_error"
    assert len(renderer_calls) == 1
    # No output JSON was ever written -- the run itself never completed.
    assert not os.path.exists(output_path)


def test_execute_swing_run_also_refreshes_near_miss_log_by_default(
        tmp_path, monkeypatch):
    """Regression: the near-miss log must refresh
    at the SAME finalization point as RESULTS-LOG.md, and it must do so
    via the REAL default renderer (batch3.regenerate_near_miss_log) when a
    caller only overrides results_renderer, not near_miss_renderer."""
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_dsr)
    ledger = str(tmp_path / "ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    research_root = str(tmp_path / "runs-root")
    results_log_path = str(tmp_path / "RESULTS-LOG.md")
    output_path = str(tmp_path / "out.json")
    os.makedirs(research_root, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("[]")

    window = batch3.derive_holdout_window("2026-06-15")
    padded_start = (pd.Timestamp(window["holdout_start"])
                     - pd.Timedelta(days=swing.intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS)
                     ).strftime("%Y-%m-%d")
    in_sample_frame = _swing_frame("2025-10-01", "2025-11-30")
    holdout_frame = _swing_frame(padded_start, window["holdout_end"])
    fetch_fn = _make_dispatch_fetch_fn({
        ("FAKE", "2025-10-01", "2025-11-30"): in_sample_frame,
        ("FAKE", padded_start, window["holdout_end"]): holdout_frame,
    })
    candidate = _base_candidate()

    def fake_renderer(*, catalog_path, ledger_path, research_root, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# fake results log\n")

    swing.execute_swing_run(
        [candidate], quotes={"swing_family": _TIGHT_QUOTE},
        tested_date="2026-06-15", run_id="run-near-miss", ledger_path=ledger,
        dry_run=False, output_path=output_path, catalog_path=catalog_path,
        research_root=research_root, results_log_path=results_log_path,
        fetch_fn=fetch_fn, results_renderer=fake_renderer)

    # near_miss_log_path was not passed -- it must default to a sibling of
    # results_log_path (tmp_path itself here, NOT research_root, which is a
    # separate directory in this test).
    near_miss_log = str(tmp_path / "NEAR-MISS-LOG.md")
    assert os.path.exists(near_miss_log)
    with open(near_miss_log, "r", encoding="utf-8") as f:
        text = f.read()
    assert "near-miss" in text.lower()


def test_empty_candidate_list_returns_cleanly(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    result = swing.run_swing_batch(
        [], quotes={}, tested_date="2026-06-15", run_id="run-empty",
        ledger_path=ledger, dry_run=False, fetch_fn=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("fetch_fn should never be called")))
    assert result["results"] == []
    assert result["gate_exclusions"] == []
