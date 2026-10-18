import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_fed_net_liquidity_qqq_mining as driver


def _ohlc_frame(start, end, *, seed=7, drift=0.0003):
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.01, len(index)))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": 1_000_000}, index=index)


# Deliberately DIFFERENT seeds/drift/quotes for QQQ vs SPY in every fixture
# below (identical QQQ/SPY
# fixtures would give the test suite zero power to catch a benchmark_frame/
# qqq_frame or QQQ-cost/SPY-cost mixup -- see
# test_varying_spy_frame_alone_changes_the_recorded_result and
# test_varying_qqq_frame_alone_changes_the_recorded_result).
def _qqq_frame(start="2015-01-02", end="2020-12-30"):
    return _ohlc_frame(start, end, seed=7, drift=0.0006)


def _spy_frame(start="2015-01-02", end="2020-12-30"):
    return _ohlc_frame(start, end, seed=99, drift=0.0001)


def _weekly(seed, base, scale, start="2014-01-01", n=400):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="7D")
    return pd.Series(base * np.cumprod(1.0 + rng.normal(0.0, scale, n)), index=dates)


def _write_fixture(tmp_path, *, status="eligible"):
    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    accepted_path = str(tmp_path / "accepted.json")
    qqq_quotes_path = str(tmp_path / "qqq_quotes.json")
    spy_quotes_path = str(tmp_path / "spy_quotes.json")

    catalog_entry = {
        "family_id": driver.fnl_mod.FAMILY_ID, "name": "test",
        "mechanism": "test", "tickers": ["QQQ"], "status": status,
        "account_types": ["taxable", "roth"], "source": "test",
        "added_date": "2026-09-15",
        "parameters": driver.fnl_mod.PARAMETERS,
    }
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump([catalog_entry], handle)
    accepted_entry = {
        "family_id": driver.fnl_mod.FAMILY_ID, "shape": "single_long",
        "tickers": ["QQQ"], "parameters": driver.fnl_mod.PARAMETERS,
    }
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": [accepted_entry]}, handle)
    with open(qqq_quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-15T22:00:00",
                   "market_state": "closed", "quotes": {"QQQ": {
            "bid": 700.0, "ask": 700.70, "spread_bps": 10.0,
            "security_status": "Closed", "suspect": False}}}, handle)
    with open(spy_quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-15T10:00:00",
                   "market_state": "open", "quotes": {"SPY": {
            "bid": 750.0, "ask": 750.15, "spread_bps": 2.0,
            "security_status": "Normal", "suspect": False}}}, handle)
    return ledger_path, catalog_path, accepted_path, qqq_quotes_path, spy_quotes_path


def test_compute_and_record_real_run_writes_one_ledger_row(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)

    result = driver.compute_and_record(
        qqq_frame=_qqq_frame(),
        spy_frame=_spy_frame(),
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")
    assert result["benchmark"] == "SPY"
    assert result["tickers"] == ["QQQ"]

    with open(ledger_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["family_id"] == driver.fnl_mod.FAMILY_ID


def test_compute_and_record_skips_when_catalog_status_not_eligible(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path, status="ruled_out")

    result = driver.compute_and_record(
        qqq_frame=_qqq_frame(),
        spy_frame=_spy_frame(),
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")
    assert result["reason_code"] == "catalog_status"


def test_compute_and_record_skips_a_second_real_run_as_already_scored(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)

    first = driver.compute_and_record(
        qqq_frame=_qqq_frame(),
        spy_frame=_spy_frame(),
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test-1", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")
    assert "reason_code" not in first

    second = driver.compute_and_record(
        qqq_frame=_qqq_frame(),
        spy_frame=_spy_frame(),
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test-2", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")
    assert second["reason_code"] == "already_scored_abi1"


def _alt_spy_frame():
    # A materially different price path from _spy_frame() (much higher
    # drift), so a call using this MUST produce a different benchmark
    # comparison than one using _spy_frame() -- if it doesn't, spy_frame
    # is not actually reaching evaluate_family's benchmark_frame.
    return _ohlc_frame("2015-01-02", "2020-12-30", seed=99, drift=0.0080)


def _alt_qqq_frame():
    # Same idea, isolating qqq_frame's own effect (the traded curve).
    return _ohlc_frame("2015-01-02", "2020-12-30", seed=7, drift=0.0080)


def test_varying_spy_frame_alone_changes_the_recorded_result(tmp_path):
    """Guards against a benchmark_frame/qqq_frame mixup: a driver that
    silently used qqq_frame as its own benchmark (ignoring the spy_frame
    argument entirely) must fail this test. This isolates spy_frame's own causal
    effect: holding qqq_frame and all costs FIXED, changing only
    spy_frame must change the result. dry_run never touches the ledger,
    so no already_scored_abi1 collision between the two calls."""
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)
    fixed_qqq_frame = _qqq_frame()

    with_spy_a = driver.compute_and_record(
        qqq_frame=fixed_qqq_frame, spy_frame=_spy_frame(),
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test-a", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")

    with_spy_b = driver.compute_and_record(
        qqq_frame=fixed_qqq_frame, spy_frame=_alt_spy_frame(),
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test-b", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")

    assert with_spy_a["sharpe_daily_excess"] != pytest.approx(with_spy_b["sharpe_daily_excess"])
    assert with_spy_a["dsr_value"] != pytest.approx(with_spy_b["dsr_value"])


def test_varying_qqq_frame_alone_changes_the_recorded_result(tmp_path):
    """Companion to the spy_frame isolation test above: holding spy_frame
    and all costs FIXED, changing only qqq_frame must change the result
    -- proves qqq_frame is actually consumed as the traded instrument
    (build_single_asset_curve), not, say, silently ignored in favor of
    always re-deriving a curve from spy_frame."""
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)
    fixed_spy_frame = _spy_frame()

    with_qqq_a = driver.compute_and_record(
        qqq_frame=_qqq_frame(), spy_frame=fixed_spy_frame,
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test-c", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")

    with_qqq_b = driver.compute_and_record(
        qqq_frame=_alt_qqq_frame(), spy_frame=fixed_spy_frame,
        walcl=_weekly(1, 4_000_000.0, 0.01),
        wtregen=_weekly(2, 400_000.0, 0.05), rrpontsyd=_weekly(3, 500.0, 0.1),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path,
        run_id="fed-liq-qqq-test-d", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_fed_net_liquidity_qqq_mining.py",
        signal_source_sha256="deadbeef")

    assert with_qqq_a["sharpe_daily_excess"] != pytest.approx(with_qqq_b["sharpe_daily_excess"])
    assert with_qqq_a["dsr_value"] != pytest.approx(with_qqq_b["dsr_value"])
