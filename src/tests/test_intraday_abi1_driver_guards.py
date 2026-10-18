"""Irreversible-path guards shared by the three bespoke intraday ABI-1 drivers.

A real (non --dry-run) invocation of any of these drivers records a ONE-SHOT
outcome through ``abi1_driver.record_abi1_outcome`` (``already_scored_abi1``
then blocks every later attempt), so the guards that stop a wrong real run
from being recorded are exactly the code that must never be wrong. This file
runs ONE parametrized matrix over all three drivers so a guard added to one
sibling cannot silently be missing from another.

Every test is hermetic: the ledger, catalog, quotes file and research root
all live under ``tmp_path`` and the network is monkeypatched (see
``_intraday_abi1_driver_support``). The REAL trial ledger is never touched.
"""
import json
import os
import sys
import threading
import time

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
for _path in (_SRC, _SCRIPTS, os.path.dirname(__file__)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_cross_asset_divergence_fvg_abi1 as cross_asset_driver  # noqa: E402
import run_fvg_reversal_cascade_pooled_abi1 as fvg_pool_driver  # noqa: E402
import run_vwap_cross_decaying_conviction_abi1 as vwap_driver  # noqa: E402
from _intraday_abi1_driver_support import Harness, write_quotes  # noqa: E402

DRIVER_CASES = {
    "vwap": dict(
        module=vwap_driver, quotes=("QQQ", "SPY"), tickers=["QQQ"],
        drifted_tickers=["QQQ", "IWM"], has_start_date=False,
        results_name="vwap-cross-decaying-conviction-results.json",
        dry_name="vwap-cross-decaying-conviction-dry-run.json"),
    "cross_asset": dict(
        module=cross_asset_driver, quotes=("QQQ", "SPY"), tickers=["QQQ"],
        drifted_tickers=["QQQ", "IWM"], has_start_date=True,
        results_name="cross-asset-divergence-fvg-results.json",
        dry_name="cross-asset-divergence-fvg-dry-run.json"),
    "fvg_pool": dict(
        module=fvg_pool_driver, quotes=("QQQ", "SPY", "IWM"),
        tickers=["QQQ", "SPY", "IWM"], drifted_tickers=["QQQ", "SPY"],
        has_start_date=True,
        results_name="fvg-reversal-cascade-pooled-results.json",
        dry_name="fvg-reversal-cascade-pooled-dry-run.json"),
}


@pytest.fixture(params=sorted(DRIVER_CASES))
def case(request):
    return DRIVER_CASES[request.param]


@pytest.fixture
def harness(case, tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch, case["module"],
                   quote_symbols=case["quotes"], catalog_tickers=case["tickers"])


@pytest.fixture(params=sorted(name for name, c in DRIVER_CASES.items() if c["has_start_date"]))
def start_date_case(request):
    """Only the drivers that have a start_date override at all."""
    return DRIVER_CASES[request.param]


@pytest.fixture
def start_date_harness(start_date_case, tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch, start_date_case["module"],
                   quote_symbols=start_date_case["quotes"], catalog_tickers=start_date_case["tickers"])


# ------------------------------------------------ real-run window overrides

def test_real_run_refuses_an_end_date_override(case, harness):
    """A stray/leftover --end-date on a real run would truncate the scored
    history and that shortened result would become the permanent outcome --
    the same hazard the start_date guard exists for."""
    with pytest.raises(ValueError, match="end_date"):
        case["module"].compute_and_record(**harness.kwargs(end_date="2026-03-03"))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_dry_run_still_accepts_an_end_date_override(case, harness):
    """The refusal is real-run-only: --end-date stays a usable dry-run knob."""
    result = case["module"].compute_and_record(
        **harness.kwargs(dry_run=True, end_date="2026-03-03"))
    assert "reason_code" not in result
    assert harness.fetch_calls
    assert all(end == "2026-03-03" for _symbol, _start, end in harness.fetch_calls)
    assert harness.ledger_rows() == []


def test_real_run_refuses_a_start_date_override(start_date_case, start_date_harness):
    case, harness = start_date_case, start_date_harness
    with pytest.raises(ValueError, match="start_date"):
        case["module"].compute_and_record(**harness.kwargs(start_date="2026-03-03"))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_dry_run_still_accepts_a_start_date_override(start_date_case, start_date_harness):
    case, harness = start_date_case, start_date_harness
    result = case["module"].compute_and_record(
        **harness.kwargs(dry_run=True, start_date="2026-03-02"))
    assert "reason_code" not in result
    assert all(start == "2026-03-02" for _symbol, start, _end in harness.fetch_calls)


def test_a_real_run_always_fetches_from_the_measured_floor(case, harness):
    case["module"].compute_and_record(**harness.kwargs())
    assert harness.fetch_calls
    assert all(start == case["module"].REAL_DATA_START
               for _symbol, start, _end in harness.fetch_calls)


# ----------------------------------------------------------- quotes guard

def test_real_run_refuses_a_synthetic_quotes_file(case, harness):
    write_quotes(harness.quotes_path, case["quotes"], synthetic=True)
    with pytest.raises(ValueError, match="SYNTHETIC"):
        case["module"].compute_and_record(**harness.kwargs())
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


# ------------------------------------------------- ledger side effects

def test_a_dry_run_writes_no_ledger_rows(case, harness):
    write_quotes(harness.quotes_path, case["quotes"], synthetic=True)
    result = case["module"].compute_and_record(**harness.kwargs(dry_run=True))
    # The dry run really ran end to end (fetched data, produced a verdict) --
    # so "no ledger rows" is not vacuous.
    assert "reason_code" not in result
    assert harness.fetch_calls
    assert "verdict" in result and "stage" in result
    assert harness.ledger_rows() == []
    assert not os.path.exists(harness.ledger_path)


def test_a_real_run_writes_exactly_one_abi1_outcome_and_a_second_is_blocked(case, harness):
    first = case["module"].compute_and_record(**harness.kwargs())
    assert "reason_code" not in first
    rows_after_first = harness.ledger_rows()
    outcomes = [r for r in rows_after_first if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["family_id"] == case["module"].FAMILY_ID
    assert outcomes[0]["construction"] == "ABI-1"

    calls_before = len(harness.fetch_calls)
    second = case["module"].compute_and_record(**harness.kwargs())
    assert second["reason_code"] == "already_scored_abi1"
    assert len(harness.fetch_calls) == calls_before
    assert harness.ledger_rows() == rows_after_first


def test_two_concurrent_real_runs_only_one_outcome_is_ever_recorded(case, harness, monkeypatch):
    """The race this test covers:
    compute_and_record's own already_scored prefilter (an unlocked
    trial_ledger.read_all call, run BEFORE the -- often expensive --
    backtest/evaluate work below it) and the eventual outcome append are
    separate steps with real compute in between. Two processes racing on
    the SAME family can both pass the prefilter; the outcome append must
    still record only one permanent ABI-1 outcome, or the one-shot-per-
    family invariant this whole pipeline depends on would break.

    Widen the window with an injected sleep right after the prefilter's own
    read -- the same technique test_concurrent_record_first_test_only_one_
    wins_the_race (test_mining_trial_ledger.py) uses for record_first_
    test's own TOCTOU -- release both threads from one barrier, and prove
    only one outcome row is EVER recorded: genuine concurrency (real
    threads, real compute through the real backtest+evaluate+ledger-write
    path, not a mock), matching this repo's own established msvcrt-mutex
    thread-stand-in convention (test_mining_trial_ledger.py's own comment:
    a thread-based test is faithful here because the lock conflicts across
    handles even WITHIN one process)."""
    from mining import trial_ledger

    original_read_all = trial_ledger.read_all

    def slow_read_all(path):
        result = original_read_all(path)
        time.sleep(0.15)
        return result

    monkeypatch.setattr(trial_ledger, "read_all", slow_read_all)

    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        result = case["module"].compute_and_record(**harness.kwargs())
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    outcomes = [r for r in harness.ledger_rows() if r.get("event") == "outcome"]
    assert len(outcomes) == 1, (
        f"expected exactly one permanent ABI-1 outcome row after two "
        f"concurrent real runs, found {len(outcomes)}: {outcomes}")

    succeeded = [r for r in results if "reason_code" not in r]
    skipped = [r for r in results if "reason_code" in r]
    assert len(succeeded) == 1
    assert len(skipped) == 1
    assert skipped[0]["reason_code"] == "already_scored_abi1"


def test_a_seeded_abi1_outcome_blocks_a_real_run_before_any_fetch(case, harness):
    harness.seed_abi1_outcome()
    seeded = harness.ledger_rows()
    result = case["module"].compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "already_scored_abi1"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == seeded


# ----------------------------------------------------- catalog/registration

def test_a_non_eligible_catalog_status_skips_before_any_fetch(case, tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, case["module"],
                      quote_symbols=case["quotes"], catalog_tickers=case["tickers"],
                      status="ruled_out")
    result = case["module"].compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "catalog_status"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_parameter_drift_skips_before_any_fetch(case, tmp_path, monkeypatch):
    drifted = dict(case["module"].PARAMETERS)
    drifted["force_flat_et"] = "15:30"
    harness = Harness(tmp_path, monkeypatch, case["module"],
                      quote_symbols=case["quotes"], catalog_tickers=case["tickers"],
                      catalog_parameters=drifted)
    result = case["module"].compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_ticker_drift_skips_before_any_fetch(case, tmp_path, monkeypatch):
    """parameter_drift compares only the parameters dict; the catalog's own
    `tickers` field is part of the registered definition too, so a real run
    must not proceed if the traded/pool tickers no longer match it."""
    harness = Harness(tmp_path, monkeypatch, case["module"],
                      quote_symbols=case["quotes"],
                      catalog_tickers=case["drifted_tickers"])
    result = case["module"].compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert "tickers" in result["reason"]
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_ticker_order_difference_is_not_drift(case, tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, case["module"],
                      quote_symbols=case["quotes"],
                      catalog_tickers=list(reversed(case["tickers"])))
    result = case["module"].compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result


# ------------------------------------------------------------ main() writes

def test_main_refuses_a_real_run_end_date_before_writing_anything(case, harness):
    with pytest.raises(ValueError, match="end_date"):
        case["module"].main(harness.cli_argv("--end-date", "2026-03-03"))
    assert not os.path.exists(harness.results_dir())
    assert harness.ledger_rows() == []


def test_main_writes_the_results_file_for_a_real_run(case, harness, capsys):
    assert case["module"].main(harness.cli_argv()) == 0
    path = os.path.join(harness.results_dir(), case["results_name"])
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["dry_run"] is False
    assert payload["result"]["family_id"] == case["module"].FAMILY_ID
    assert "verdict" in payload["result"]


def test_main_never_overwrites_the_real_results_file_with_a_skip(case, harness, capsys):
    """An accidental second real invocation returns only a skip dict
    (already_scored_abi1); it must not replace the first run's real results
    file with a reason-only payload."""
    assert case["module"].main(harness.cli_argv()) == 0
    path = os.path.join(harness.results_dir(), case["results_name"])
    with open(path, "rb") as handle:
        first_bytes = handle.read()
    capsys.readouterr()

    assert case["module"].main(harness.cli_argv()) == 0
    with open(path, "rb") as handle:
        assert handle.read() == first_bytes
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "already_scored_abi1"
    assert printed["output"] is None


def test_main_writes_no_results_file_when_the_result_is_a_skip(case, harness, capsys):
    harness.seed_abi1_outcome()
    assert case["module"].main(harness.cli_argv()) == 0
    assert not os.path.exists(os.path.join(harness.results_dir(), case["results_name"]))
    assert not os.path.exists(os.path.join(harness.results_dir(), case["dry_name"]))


def test_main_writes_the_dry_run_file_for_a_dry_run(case, harness, capsys):
    write_quotes(harness.quotes_path, case["quotes"], synthetic=True)
    assert case["module"].main(harness.cli_argv("--dry-run")) == 0
    assert os.path.exists(os.path.join(harness.results_dir(), case["dry_name"]))
    assert not os.path.exists(os.path.join(harness.results_dir(), case["results_name"]))
    assert harness.ledger_rows() == []


# ------------------------------------------- cross-asset secondary ticker

def test_cross_asset_secondary_ticker_constant_must_match_parameters(tmp_path, monkeypatch):
    """PARAMETERS['secondary_ticker'] is what the catalog drift check pins,
    SECONDARY_TICKER is what the driver actually fetches and feeds to the
    signal. If they ever diverge (a catalog/PARAMETERS edit without touching
    the constant, or the reverse), a real run would silently score a
    different reference instrument than the one registered."""
    driver = cross_asset_driver
    harness = Harness(tmp_path, monkeypatch, driver, quote_symbols=("QQQ", "SPY", "IWM"),
                      catalog_tickers=["QQQ"])
    monkeypatch.setattr(driver, "SECONDARY_TICKER", "IWM")
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert "secondary_ticker" in result["reason"]
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []
