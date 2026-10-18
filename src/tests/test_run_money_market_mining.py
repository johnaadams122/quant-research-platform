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

import run_money_market_mining as driver
from mining import external_data


def _spy_frame():
    index = pd.bdate_range("1997-01-02", "2005-12-30")
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(index)))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": 1_000_000}, index=index)


def _monthly_flow():
    months = pd.date_range("1997-01-01", "2005-12-01", freq="MS")
    rng = np.random.default_rng(13)
    level = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.05, len(months)))
    return pd.Series(level, index=months, name="RMFSL")


def test_fetch_money_market_series_wraps_external_data(monkeypatch):
    fake_series = _monthly_flow()
    calls = {}

    def fake_fetch(series_id, **kwargs):
        calls["series_id"] = series_id
        calls.update(kwargs)
        return fake_series

    monkeypatch.setattr(
        driver.external_data, "fetch_fred_series_point_in_time", fake_fetch)
    series = driver.fetch_money_market_series()
    assert series.equals(fake_series)
    assert calls["series_id"] == "RMFSL"


def _write_fixture(tmp_path, *, status="eligible"):
    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    accepted_path = str(tmp_path / "accepted.json")
    quotes_path = str(tmp_path / "quotes.json")

    catalog_entry = {
        "family_id": driver.mm_mod.FAMILY_ID, "name": "test",
        "mechanism": "test", "tickers": ["SPY"], "status": status,
        "account_types": ["taxable", "roth"], "source": "test",
        "added_date": "2026-09-15",
        "parameters": driver.mm_mod.PARAMETERS,
    }
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump([catalog_entry], handle)
    accepted_entry = {
        "family_id": driver.mm_mod.FAMILY_ID, "shape": "single_long",
        "tickers": ["SPY"], "parameters": driver.mm_mod.PARAMETERS,
    }
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": [accepted_entry]}, handle)
    with open(quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-15T10:00:00",
                   "market_state": "open", "quotes": {"SPY": {
            "bid": 100.0, "ask": 100.01, "spread_bps": 1.0,
            "security_status": "Normal", "suspect": False}}}, handle)
    return ledger_path, catalog_path, accepted_path, quotes_path


def test_compute_and_record_dry_run_reports_a_verdict_without_writing_ledger(tmp_path):
    ledger_path, catalog_path, accepted_path, quotes_path = _write_fixture(tmp_path)

    result = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_flow=_monthly_flow(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="money-market-test", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_money_market_mining.py",
        signal_source_sha256="deadbeef")

    assert result["family_id"] == driver.mm_mod.FAMILY_ID
    assert result["verdict"] in {
        "survived", "cut_dsr", "cut_holdout", "cut_risk_refusal",
        "not_scoreable", "holdout_not_evaluable"}
    assert not os.path.exists(ledger_path) or os.path.getsize(ledger_path) == 0


def test_compute_and_record_real_run_writes_one_ledger_row(tmp_path):
    ledger_path, catalog_path, accepted_path, quotes_path = _write_fixture(tmp_path)

    driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_flow=_monthly_flow(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="money-market-test", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_money_market_mining.py",
        signal_source_sha256="deadbeef")

    with open(ledger_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["family_id"] == driver.mm_mod.FAMILY_ID


def test_compute_and_record_skips_when_catalog_status_not_eligible(tmp_path):
    ledger_path, catalog_path, accepted_path, quotes_path = _write_fixture(
        tmp_path, status="ruled_out")

    result = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_flow=_monthly_flow(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="money-market-test", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_money_market_mining.py",
        signal_source_sha256="deadbeef")
    assert result["reason_code"] == "catalog_status"


def test_compute_and_record_skips_a_second_real_run_as_already_scored(tmp_path):
    ledger_path, catalog_path, accepted_path, quotes_path = _write_fixture(tmp_path)

    first = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_flow=_monthly_flow(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="money-market-test-1", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_money_market_mining.py",
        signal_source_sha256="deadbeef")
    assert "reason_code" not in first

    second = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_flow=_monthly_flow(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="money-market-test-2", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_money_market_mining.py",
        signal_source_sha256="deadbeef")
    assert second["reason_code"] == "already_scored_abi1"

    with open(ledger_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
