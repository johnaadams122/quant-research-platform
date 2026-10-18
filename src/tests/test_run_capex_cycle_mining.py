"""Tests for scripts/run_capex_cycle_mining.py -- the bespoke ABI-1 driver
for the capex-cycle family (capex_cycle_blue_chip_dip_pooled) and its mandatory control
arm (blue_chip_reversal_no_capex_control).

Everything runs on tmp copies: tmp ledger, tmp catalog, tmp accepted.json,
tmp quotes, injected deterministic synthetic OHLC, and an injected
``fundamentals_map`` (no network -- no SEC EDGAR calls in this file). The
default data/mining/trial_ledger.jsonl and catalog.json paths are never
touched.
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

import run_batch5_mining as batch5
import run_capex_cycle_mining as driver
from mining import trial_ledger
from mining.batch5 import capex_cycle_blue_chip_dip_pooled as main_mod
from mining.batch5 import blue_chip_reversal_no_capex_control as control_mod

RUN_ID = "capex-cycle-test-run"
TESTED_DATE = "2026-09-10"
N_SESSIONS = 1300
POOL = ("AAA", "BBB", "CCC", "DDD", "EEE")


def _index(n: int, start: str = "2015-01-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _frame(n: int, seed: int, drop_at=None, drop_pct=0.45) -> pd.DataFrame:
    """A mild random walk that permanently drops ``drop_pct`` at ``drop_at``
    and never recovers -- guarantees a large, deterministic underperformance
    spread vs a flat-ish SPY for a long stretch afterward (own_ret dominated
    by the drop, not by noise), regardless of the random seed."""
    index = _index(n)
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, n)))
    if drop_at is not None:
        close[drop_at:] *= (1.0 - drop_pct)
    high = close * 1.01
    low = close * 0.99
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=index)


DROP_AT = 550  # session index of the permanent drop -- see _frame's docstring


def _pool_frames():
    frames = {"SPY": _frame(N_SESSIONS, seed=1)}
    for i, ticker in enumerate(POOL):
        frames[ticker] = _frame(N_SESSIONS, seed=10 + i, drop_at=DROP_AT + i)
    return frames


def _loader(frames):
    def loader(symbol):
        if symbol not in frames:
            raise RuntimeError(f"no synthetic data for {symbol}")
        return frames[symbol].reset_index().rename(columns={"index": "Date"})
    return loader


def _fundamentals_frame(index: pd.DatetimeIndex, *, drop_at: int, growth: float = 0.50):
    """Minimal point-in-time series (matches edgar_point_in_time.build_
    point_in_time_annual_series's output shape, just the two columns
    capex_condition_series actually reads): a usable_asof row well before
    the drop, then re-disclosed every ~300 calendar days (comfortably under
    STALENESS_CAP_DAYS=400) so the growth figure never goes stale for the
    rest of the fixture -- real annual 10-K filings land ~365 days apart."""
    first_asof = index[max(drop_at - 60, 0)]
    asofs = [first_asof]
    while asofs[-1] < index[-1]:
        asofs.append(asofs[-1] + pd.Timedelta(days=300))
    return pd.DataFrame({"usable_asof": asofs, "capex_growth_yoy": growth})


def _fundamentals_map(frames):
    return {ticker: _fundamentals_frame(frames[ticker].index, drop_at=DROP_AT + i)
           for i, ticker in enumerate(POOL)}


def _quote(bid, ask):
    return {"price": (bid + ask) / 2, "bid": bid, "ask": ask,
            "security_status": "Normal",
            "spread_bps": (ask - bid) / ((ask + bid) / 2) * 1e4, "suspect": False}


def _write_quotes(tmp_path, *, drop=()):
    symbols = ("SPY",) + POOL
    quotes = {s: _quote(100.0, 100.02) for s in symbols if s not in drop}
    payload = {"sampled_at_local": "2026-09-10T16:32:00-04:00",
               "source": "test-quotes", "market_state": "closed (test)",
               "quotes": quotes}
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _write_accepted(tmp_path):
    payload = {"batch": "test", "curated_on": TESTED_DATE,
               "accepted": [{"family_id": main_mod.FAMILY_ID},
                            {"family_id": control_mod.FAMILY_ID}],
               "infeasible": [], "rejected_by_judge": [], "merged_notes": ""}
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _catalog_row(module, *, status="eligible", parameters=None):
    return {"family_id": module.FAMILY_ID, "name": "test", "mechanism": "test",
            "tickers": ["x"], "parameters": parameters or module.PARAMETERS,
            "account_types": ["taxable"], "status": status, "source": "test",
            "added_date": TESTED_DATE}


def _write_catalog(tmp_path, *, main_status="eligible", control_status="eligible",
                   main_parameters=None, control_parameters=None):
    entries = [_catalog_row(main_mod, status=main_status, parameters=main_parameters),
              _catalog_row(control_mod, status=control_status, parameters=control_parameters)]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return str(path)


def _patch_pool(monkeypatch, frames):
    monkeypatch.setattr(main_mod, "TRADED", list(POOL))
    monkeypatch.setattr(batch5, "default_ohlc_loader", _loader(frames))


def _build(tmp_path, monkeypatch, *, frames=None, drop_quote=(), main_status="eligible",
          control_status="eligible", main_parameters=None, control_parameters=None):
    frames = frames or _pool_frames()
    _patch_pool(monkeypatch, frames)
    accepted_path = _write_accepted(tmp_path)
    quotes_path = _write_quotes(tmp_path, drop=drop_quote)
    catalog_path = _write_catalog(
        tmp_path, main_status=main_status, control_status=control_status,
        main_parameters=main_parameters, control_parameters=control_parameters)
    ledger_path = str(tmp_path / "ledger.jsonl")
    run_dir = str(tmp_path / "research" / RUN_ID)
    return dict(accepted_path=accepted_path, quotes_path=quotes_path,
               catalog_path=catalog_path, ledger_path=ledger_path, run_dir=run_dir)


def _outcomes(ledger_path, family_id=None):
    if not os.path.exists(ledger_path):
        return []
    rows = [r for r in trial_ledger.read_all(ledger_path) if r["event"] == "outcome"]
    if family_id is not None:
        rows = [r for r in rows if r["family_id"] == family_id]
    return rows


def _by_family(computed, family_id):
    return next(item for item in computed if item["family_id"] == family_id)


# --------------------------------------------------------------- end to end

def test_both_families_score_with_full_pool_and_fundamentals(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)

    assert not result["skipped"], result["skipped"]
    assert len(result["computed"]) == 2
    main_item = _by_family(result["computed"], main_mod.FAMILY_ID)
    control_item = _by_family(result["computed"], control_mod.FAMILY_ID)
    assert set(main_item["pool_usable"]) == set(POOL)
    assert set(control_item["pool_usable"]) == set(POOL)
    assert main_item["pool_excluded"] == {}
    assert main_item["tax_policy"] == "none"
    assert main_item["fundamentals_missing"] == []
    # The capex-conditioned arm requires BOTH conditions, so it can only
    # fire a subset of (or equal to) the control arm's unconditioned trades.
    assert main_item["n_trades"] >= 1
    assert control_item["n_trades"] >= main_item["n_trades"]

    rows = _outcomes(built["ledger_path"])
    assert len(rows) == 2
    assert {r["family_id"] for r in rows} == {main_mod.FAMILY_ID, control_mod.FAMILY_ID}
    assert all(r["construction"] == "ABI-1" for r in rows)


def _flat_frame(index, price=100.0):
    return pd.DataFrame({"Open": price, "High": price * 1.001,
                         "Low": price * 0.999, "Close": price}, index=index)


def test_staggered_inception_dates_do_not_crash(tmp_path, monkeypatch):
    """Guards the case every OTHER fixture in this file skips: they build
    SPY and every pool ticker off the identical shared index, so
    make_signal_fn's own ``own.index.equals(bench.index)`` requirement is
    trivially satisfied and never exercises a ticker whose inception
    differs from SPY's (public examples in this watchlist: CAT/CNP start
    1962, well before SPY's 1993; NCLH starts 2013, well after). Without
    the driver's per-ticker calendar alignment, that requirement would
    raise for such a ticker and the family could not be scored. Flat,
    non-trading prices here on purpose -- this test only proves calendar
    alignment prevents the crash, not the trading logic itself (covered by
    the other tests)."""
    spy_index = _index(N_SESSIONS)
    frames = {"SPY": _flat_frame(spy_index)}
    early_index = pd.bdate_range(end=spy_index[0] - pd.Timedelta(days=1), periods=400)
    frames["AAA"] = _flat_frame(early_index.append(spy_index))  # starts before SPY
    frames["BBB"] = _flat_frame(spy_index[400:])                # starts after SPY
    for ticker in ("CCC", "DDD", "EEE"):
        frames[ticker] = _flat_frame(spy_index)                 # same calendar as SPY

    built = _build(tmp_path, monkeypatch, frames=frames)
    fundamentals = {t: _fundamentals_frame(frames[t].index, drop_at=0) for t in POOL}
    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=True,
        fundamentals_map=fundamentals, **built)

    assert not result["skipped"], result["skipped"]
    assert len(result["computed"]) == 2
    for item in result["computed"]:
        assert set(item["pool_usable"]) == set(POOL)
        assert item["pool_excluded"] == {}
        assert item["stage"] is not None


def test_fundamentals_actually_drive_the_main_family_not_the_control(tmp_path, monkeypatch):
    """The whole reason this bespoke driver exists: run_abi1_mining.py's
    generic dispatch would call make_signal_fn with no fundamentals kwarg,
    silently scoring capex_ok=False for every date. Prove the wiring
    actually matters by running the identical fixture twice -- once with a
    real fundamentals_map, once with an empty one -- and asserting the
    MAIN family's trade count collapses to 0 while the CONTROL family's
    stays identical (it never reads fundamentals at all)."""
    frames = _pool_frames()
    built_with = _build(tmp_path, monkeypatch, frames=frames)
    with_fundamentals = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-with", dry_run=True,
        fundamentals_map=_fundamentals_map(frames), **built_with)

    built_without = _build(tmp_path, monkeypatch, frames=frames)
    without_fundamentals = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-without", dry_run=True,
        fundamentals_map={}, **built_without)

    main_with = _by_family(with_fundamentals["computed"], main_mod.FAMILY_ID)
    main_without = _by_family(without_fundamentals["computed"], main_mod.FAMILY_ID)
    control_with = _by_family(with_fundamentals["computed"], control_mod.FAMILY_ID)
    control_without = _by_family(without_fundamentals["computed"], control_mod.FAMILY_ID)

    assert main_with["n_trades"] >= 1
    assert main_without["n_trades"] == 0
    assert control_with["n_trades"] == control_without["n_trades"]
    assert control_with["n_trades"] >= 1
    assert main_without["fundamentals_missing"] == sorted(POOL)


def test_missing_quote_excludes_member_from_both_families(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames, drop_quote=("CCC",))
    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)

    assert not result["skipped"], result["skipped"]
    remaining = set(POOL) - {"CCC"}
    for family_id in (main_mod.FAMILY_ID, control_mod.FAMILY_ID):
        item = _by_family(result["computed"], family_id)
        assert set(item["pool_usable"]) == remaining
        assert "CCC" in item["pool_excluded"]


def test_pool_coverage_floor_refuses_both_families(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames,
                   drop_quote=("AAA", "BBB", "CCC"))
    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert result["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "pool_coverage"
    assert codes[control_mod.FAMILY_ID] == "pool_coverage"


def test_dry_run_writes_no_ledger_rows(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=True,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert not os.path.exists(built["ledger_path"])
    assert len(result["computed"]) == 2
    assert all(item["stage"] is not None for item in result["computed"])


def test_already_scored_family_is_skipped_independently(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    first = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert len(first["computed"]) == 2

    second = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-2", dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert second["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in second["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "already_scored_abi1"
    assert codes[control_mod.FAMILY_ID] == "already_scored_abi1"
    assert len(_outcomes(built["ledger_path"])) == 2


def test_catalog_status_not_eligible_skips_that_family_only(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames, control_status="ruled_out")
    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert len(result["computed"]) == 1
    assert result["computed"][0]["family_id"] == main_mod.FAMILY_ID
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[control_mod.FAMILY_ID] == "catalog_status"


def test_parameter_drift_against_catalog_skips_that_family_only(tmp_path, monkeypatch):
    frames = _pool_frames()
    drifted = dict(main_mod.PARAMETERS)
    drifted["capex_growth_threshold"] = 0.99
    built = _build(tmp_path, monkeypatch, frames=frames, main_parameters=drifted)
    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert len(result["computed"]) == 1
    assert result["computed"][0]["family_id"] == control_mod.FAMILY_ID
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "parameter_drift"


def test_not_in_accepted_skips_that_family(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    payload = json.loads(open(built["accepted_path"], encoding="utf-8").read())
    payload["accepted"] = [c for c in payload["accepted"]
                           if c["family_id"] != control_mod.FAMILY_ID]
    with open(built["accepted_path"], "w", encoding="utf-8") as f:
        json.dump(payload, f)

    result = driver.compute_batch_capex_cycle(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert len(result["computed"]) == 1
    assert result["computed"][0]["family_id"] == main_mod.FAMILY_ID
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[control_mod.FAMILY_ID] == "not_in_accepted"


# ----------------------------------------------------------- fundamentals

def test_build_fundamentals_map_reports_coverage_without_excluding(monkeypatch):
    calls = []

    def fake_series_for_cik(cik, *, user_agent, sleep, min_interval):
        calls.append(cik)
        if cik == 1:
            return _fundamentals_frame(_index(100), drop_at=50), ["tag1"]
        if cik == 2:
            return None, []  # no usable disclosure
        raise RuntimeError("boom")  # cik == 3: fetch error

    monkeypatch.setattr(driver.watchlist_script, "_build_point_in_time_series_for_ticker",
                        fake_series_for_cik)
    fundamentals_map, coverage = driver.build_fundamentals_map(
        ["OK", "EMPTY", "ERROR", "NOCIK"],
        cik_by_ticker={"OK": 1, "EMPTY": 2, "ERROR": 3, "NOCIK": None},
        sleep=lambda s: None, min_interval=0.0)

    assert set(fundamentals_map) == {"OK"}
    assert coverage["OK"] == "ok"
    assert coverage["EMPTY"] == "no_usable_disclosure"
    assert coverage["ERROR"].startswith("fetch_error")
    assert coverage["NOCIK"] == "no_cik_resolved"
    assert calls == [1, 2, 3]  # NOCIK never reaches the fetch call at all


def test_main_cli_defaults_are_well_formed():
    assert os.path.isabs(driver.DEFAULT_ACCEPTED)
    assert os.path.isabs(driver.DEFAULT_QUOTES)
    assert driver.MARKET_BENCHMARK == "SPY"
    assert driver.TAX_POLICY == "none"
