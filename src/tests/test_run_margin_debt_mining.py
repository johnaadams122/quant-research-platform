import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_margin_debt_mining as driver
from mining import external_data


def _spy_frame():
    index = pd.bdate_range("1997-01-02", "2005-12-30")
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(index)))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": 1_000_000}, index=index)


def _monthly_debt():
    months = pd.date_range("1997-01-31", "2005-12-31", freq="ME")
    rng = np.random.default_rng(11)
    level = 100.0 * np.cumprod(1.0 + rng.normal(0.004, 0.03, len(months)))
    return pd.Series(level, index=months, name="finra_margin_debt_usd_millions")


def test_fetch_margin_debt_series_wraps_external_data(monkeypatch):
    fake_snapshot = external_data.ExternalSeriesSnapshot(
        series=_monthly_debt(),
        metadata={"source_url": "u", "retrieved_at": "t", "raw_sha256": "h",
                  "worksheet": "w", "units": "USD millions",
                  "date_semantics": "reference_month_end"})
    calls = {}

    def fake_fetch(**kwargs):
        calls.update(kwargs)
        return fake_snapshot

    monkeypatch.setattr(driver.external_data, "fetch_finra_margin_debt", fake_fetch)
    series, metadata = driver.fetch_margin_debt_series()
    assert series.equals(fake_snapshot.series)
    assert metadata == fake_snapshot.metadata


def test_compute_and_record_dry_run_reports_a_verdict_without_writing_ledger(tmp_path):
    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    accepted_path = str(tmp_path / "accepted.json")
    quotes_path = str(tmp_path / "quotes.json")

    catalog_entry = {
        "family_id": driver.margin_mod.FAMILY_ID, "name": "test",
        "mechanism": "test", "tickers": ["SPY"], "status": "eligible",
        "account_types": ["taxable", "roth"], "source": "test",
        "added_date": "2026-09-15",
        "parameters": driver.margin_mod.PARAMETERS,
    }
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump([catalog_entry], handle)
    accepted_entry = {
        "family_id": driver.margin_mod.FAMILY_ID, "shape": "single_long",
        "tickers": ["SPY"], "parameters": driver.margin_mod.PARAMETERS,
    }
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": [accepted_entry]}, handle)
    with open(quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-15T10:00:00",
                   "market_state": "open", "quotes": {"SPY": {
            "bid": 100.0, "ask": 100.01, "spread_bps": 1.0,
            "security_status": "Normal", "suspect": False}}}, handle)

    result = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_debt=_monthly_debt(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="margin-debt-test", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_margin_debt_mining.py",
        signal_source_sha256="deadbeef")

    assert result["family_id"] == driver.margin_mod.FAMILY_ID
    assert result["verdict"] in {
        "survived", "cut_dsr", "cut_holdout", "cut_risk_refusal",
        "not_scoreable", "holdout_not_evaluable"}
    assert not os.path.exists(ledger_path) or os.path.getsize(ledger_path) == 0


def test_compute_and_record_real_run_writes_one_ledger_row(tmp_path):
    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    accepted_path = str(tmp_path / "accepted.json")
    quotes_path = str(tmp_path / "quotes.json")

    catalog_entry = {
        "family_id": driver.margin_mod.FAMILY_ID, "name": "test",
        "mechanism": "test", "tickers": ["SPY"], "status": "eligible",
        "account_types": ["taxable", "roth"], "source": "test",
        "added_date": "2026-09-15",
        "parameters": driver.margin_mod.PARAMETERS,
    }
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump([catalog_entry], handle)
    accepted_entry = {
        "family_id": driver.margin_mod.FAMILY_ID, "shape": "single_long",
        "tickers": ["SPY"], "parameters": driver.margin_mod.PARAMETERS,
    }
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": [accepted_entry]}, handle)
    with open(quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-15T10:00:00",
                   "market_state": "open", "quotes": {"SPY": {
            "bid": 100.0, "ask": 100.01, "spread_bps": 1.0,
            "security_status": "Normal", "suspect": False}}}, handle)

    driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_debt=_monthly_debt(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="margin-debt-test", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_margin_debt_mining.py",
        signal_source_sha256="deadbeef")

    with open(ledger_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["family_id"] == driver.margin_mod.FAMILY_ID


def test_compute_and_record_skips_when_catalog_status_not_eligible(tmp_path):
    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    accepted_path = str(tmp_path / "accepted.json")
    quotes_path = str(tmp_path / "quotes.json")
    catalog_entry = {
        "family_id": driver.margin_mod.FAMILY_ID, "name": "test",
        "mechanism": "test", "tickers": ["SPY"], "status": "ruled_out",
        "account_types": ["taxable"], "source": "test",
        "added_date": "2026-09-15", "parameters": driver.margin_mod.PARAMETERS,
    }
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump([catalog_entry], handle)
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": [{
            "family_id": driver.margin_mod.FAMILY_ID, "shape": "single_long",
            "tickers": ["SPY"], "parameters": driver.margin_mod.PARAMETERS}]}, handle)
    with open(quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-15T10:00:00",
                   "market_state": "open", "quotes": {"SPY": {
            "bid": 100.0, "ask": 100.01, "spread_bps": 1.0,
            "security_status": "Normal", "suspect": False}}}, handle)

    result = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_debt=_monthly_debt(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="margin-debt-test", tested_date="2026-09-15", dry_run=True,
        signal_source_path="scripts/run_margin_debt_mining.py",
        signal_source_sha256="deadbeef")
    assert result["reason_code"] == "catalog_status"


def test_compute_and_record_skips_a_second_real_run_as_already_scored(tmp_path):
    """Matches every sibling bespoke driver's own idempotency test
    (test_run_capex_cycle_mining.py::test_already_scored_family_is_skipped_
    independently, test_run_cboe_rotation_mining.py, test_run_abi1_mining.py,
    test_run_synthetic_leveraged_dip_mining.py all assert this identical
    guard) -- a second real run must not duplicate the ledger row."""
    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    accepted_path = str(tmp_path / "accepted.json")
    quotes_path = str(tmp_path / "quotes.json")

    catalog_entry = {
        "family_id": driver.margin_mod.FAMILY_ID, "name": "test",
        "mechanism": "test", "tickers": ["SPY"], "status": "eligible",
        "account_types": ["taxable", "roth"], "source": "test",
        "added_date": "2026-09-15",
        "parameters": driver.margin_mod.PARAMETERS,
    }
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump([catalog_entry], handle)
    accepted_entry = {
        "family_id": driver.margin_mod.FAMILY_ID, "shape": "single_long",
        "tickers": ["SPY"], "parameters": driver.margin_mod.PARAMETERS,
    }
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": [accepted_entry]}, handle)
    with open(quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-15T10:00:00",
                   "market_state": "open", "quotes": {"SPY": {
            "bid": 100.0, "ask": 100.01, "spread_bps": 1.0,
            "security_status": "Normal", "suspect": False}}}, handle)

    first = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_debt=_monthly_debt(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="margin-debt-test-1", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_margin_debt_mining.py",
        signal_source_sha256="deadbeef")
    assert "reason_code" not in first  # a real computed score, not a skip

    second = driver.compute_and_record(
        spy_frame=_spy_frame(), monthly_debt=_monthly_debt(),
        catalog_path=catalog_path, accepted_path=accepted_path,
        quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="margin-debt-test-2", tested_date="2026-09-15", dry_run=False,
        signal_source_path="scripts/run_margin_debt_mining.py",
        signal_source_sha256="deadbeef")
    assert second["reason_code"] == "already_scored_abi1"

    with open(ledger_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1  # the second run must not duplicate the row
