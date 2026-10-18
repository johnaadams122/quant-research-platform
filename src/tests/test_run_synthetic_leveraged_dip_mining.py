"""Tests for scripts/run_synthetic_leveraged_dip_mining.py (the bespoke
ABI-1 driver for the synthetic-leverage family -- a synthetic 2x margin-financed dip-buy pool).

Everything runs on tmp copies: tmp ledger, tmp catalog, tmp accepted.json,
tmp quotes, injected deterministic synthetic OHLC and a monkeypatched
risk-free rate (no network). The real data/mining/trial_ledger.jsonl and
catalog.json are never touched.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import data_pipeline as dp
import run_batch5_mining as batch5
import run_synthetic_leveraged_dip_mining as driver
from mining import trial_ledger
from mining.batch5 import synthetic_leveraged_dip_buy_pooled_largecap as mod

RUN_ID = "synth-lev-test-run"
TESTED_DATE = "2026-09-09"
N_SESSIONS = 700
UNDERLYINGS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


def _index(n: int, start: str = "2015-01-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _frame(n: int, seed: int, dip_at=None) -> pd.DataFrame:
    index = _index(n)
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    if dip_at is not None:
        close[dip_at:dip_at + 3] *= 0.5
    high = close * 1.01
    low = close * 0.99
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=index)


_FRAMES = {
    "SPY": _frame(N_SESSIONS, seed=1),
    "AAA": _frame(N_SESSIONS, seed=11, dip_at=300),
    "BBB": _frame(N_SESSIONS, seed=12, dip_at=350),
    "CCC": _frame(N_SESSIONS, seed=13, dip_at=400),
    "DDD": _frame(N_SESSIONS, seed=14, dip_at=310),
    "EEE": _frame(N_SESSIONS, seed=15, dip_at=360),
    "FFF": _frame(N_SESSIONS, seed=16, dip_at=410),
}


def _loader():
    def loader(symbol):
        if symbol not in _FRAMES:
            raise RuntimeError(f"no synthetic data for {symbol}")
        return _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})
    return loader


def _quote(bid, ask):
    return {"price": (bid + ask) / 2, "bid": bid, "ask": ask,
            "security_status": "Normal",
            "spread_bps": (ask - bid) / ((ask + bid) / 2) * 1e4, "suspect": False}


def _write_quotes(tmp_path, *, drop=()):
    symbols = ("SPY",) + UNDERLYINGS
    quotes = {s: _quote(100.0, 100.02) for s in symbols if s not in drop}
    payload = {"sampled_at_local": "2026-09-09T16:32:00-04:00",
               "source": "test-quotes", "market_state": "closed (test)",
               "quotes": quotes}
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _write_accepted(tmp_path):
    payload = {"batch": "test", "curated_on": TESTED_DATE,
               "accepted": [{"family_id": mod.FAMILY_ID,
                             "parameters": mod.PARAMETERS}],
               "infeasible": [], "rejected_by_judge": [], "merged_notes": ""}
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _write_catalog(tmp_path, *, status="eligible", parameters=None):
    entries = [{
        "family_id": mod.FAMILY_ID, "name": "test", "mechanism": "test",
        "tickers": ["x"], "parameters": parameters or mod.PARAMETERS,
        "account_types": ["taxable"], "status": status, "source": "test",
        "added_date": TESTED_DATE,
    }]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return str(path)


def _patch_universe(monkeypatch):
    monkeypatch.setattr(mod, "SIGNAL", list(UNDERLYINGS))
    monkeypatch.setattr(mod, "TRADED",
                        [mod.synthetic_id_for(t) for t in UNDERLYINGS])
    monkeypatch.setattr(batch5, "default_ohlc_loader", _loader())
    monkeypatch.setattr(dp, "fetch_risk_free_rate",
                        lambda start="max": pd.Series(
                            0.03, index=_index(N_SESSIONS)))


def _build(tmp_path, monkeypatch, *, drop_quote=(), catalog_status="eligible",
          catalog_parameters=None):
    _patch_universe(monkeypatch)
    accepted_path = _write_accepted(tmp_path)
    quotes_path = _write_quotes(tmp_path, drop=drop_quote)
    catalog_path = _write_catalog(
        tmp_path, status=catalog_status, parameters=catalog_parameters)
    ledger_path = str(tmp_path / "ledger.jsonl")
    run_dir = str(tmp_path / "research" / RUN_ID)
    return dict(accepted_path=accepted_path, quotes_path=quotes_path,
               catalog_path=catalog_path, ledger_path=ledger_path,
               run_dir=run_dir)


def _outcomes(ledger_path, family_id=None):
    if not os.path.exists(ledger_path):
        return []
    rows = [r for r in trial_ledger.read_all(ledger_path) if r["event"] == "outcome"]
    if family_id is not None:
        rows = [r for r in rows if r["family_id"] == family_id]
    return rows


# --------------------------------------------------------------- end to end

def test_pool_scores_with_all_members_present(tmp_path, monkeypatch):
    built = _build(tmp_path, monkeypatch)
    result = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False, **built)

    assert not result["skipped"], result["skipped"]
    item = result["computed"][0]
    assert item["family_id"] == mod.FAMILY_ID
    assert set(item["pool_usable"]) == {mod.synthetic_id_for(t) for t in UNDERLYINGS}
    assert item["pool_excluded"] == {}
    assert item["tax_policy"] == "none"
    assert item["n_trades"] >= 1  # each fixture has a forced dip

    rows = _outcomes(built["ledger_path"], mod.FAMILY_ID)
    assert len(rows) == 1
    assert rows[0]["construction"] == "ABI-1"
    assert f"pool: {len(UNDERLYINGS)} of {len(UNDERLYINGS)} declared members usable" in rows[0]["note"]


def test_missing_quote_excludes_member_not_whole_family(tmp_path, monkeypatch):
    built = _build(tmp_path, monkeypatch, drop_quote=("CCC",))
    result = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False, **built)

    assert not result["skipped"], result["skipped"]
    item = result["computed"][0]
    remaining = [t for t in UNDERLYINGS if t != "CCC"]
    assert set(item["pool_usable"]) == {mod.synthetic_id_for(t) for t in remaining}
    assert "CCC" in item["pool_excluded"]


def test_pool_coverage_floor_refuses_when_too_many_members_missing(tmp_path, monkeypatch):
    built = _build(tmp_path, monkeypatch, drop_quote=("AAA", "BBB", "CCC"))
    result = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False, **built)
    assert result["computed"] == []
    assert result["skipped"][0]["reason_code"] == "pool_coverage"


def test_dry_run_writes_no_ledger_rows(tmp_path, monkeypatch):
    built = _build(tmp_path, monkeypatch)
    result = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=True, **built)
    assert not os.path.exists(built["ledger_path"])
    assert result["computed"][0]["stage"] is not None


def test_already_scored_family_is_skipped(tmp_path, monkeypatch):
    built = _build(tmp_path, monkeypatch)
    first = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False, **built)
    assert first["computed"]
    second = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-2", dry_run=False, **built)
    assert second["computed"] == []
    assert second["skipped"][0]["reason_code"] == "already_scored_abi1"
    assert len(_outcomes(built["ledger_path"], mod.FAMILY_ID)) == 1


def test_catalog_status_not_eligible_is_skipped(tmp_path, monkeypatch):
    built = _build(tmp_path, monkeypatch, catalog_status="ruled_out")
    result = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False, **built)
    assert result["computed"] == []
    assert result["skipped"][0]["reason_code"] == "catalog_status"


def test_parameter_drift_against_catalog_is_skipped(tmp_path, monkeypatch):
    drifted = dict(mod.PARAMETERS)
    drifted["dip_threshold_pct"] = -0.30
    built = _build(tmp_path, monkeypatch, catalog_parameters=drifted)
    result = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False, **built)
    assert result["computed"] == []
    assert result["skipped"][0]["reason_code"] == "parameter_drift"


def test_synthetic_frame_uses_the_real_fetched_financing_rate(tmp_path, monkeypatch):
    """The IRX series is fetched once and actually threaded through to
    build_synthetic_leveraged -- not silently defaulted -- verified by
    swapping in two different constant rates and confirming the resulting
    synthetic price path differs."""
    built = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(dp, "fetch_risk_free_rate",
                        lambda start="max": pd.Series(0.01, index=_index(N_SESSIONS)))
    low_rate = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-low", dry_run=True, **built)

    monkeypatch.setattr(dp, "fetch_risk_free_rate",
                        lambda start="max": pd.Series(0.15, index=_index(N_SESSIONS)))
    high_rate = driver.compute_batch_synthetic_leveraged(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-high", dry_run=True, **built)

    assert (low_rate["computed"][0]["sharpe_daily_excess"]
           != high_rate["computed"][0]["sharpe_daily_excess"])
