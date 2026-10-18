"""Tests for scripts/run_accrual_anomaly_mining.py -- the bespoke ABI-1
driver for accrual_anomaly_sloan. Mirrors test_run_capex_cycle_mining.py's
own conventions: everything runs on tmp copies (tmp ledger, tmp catalog,
tmp accepted.json, tmp quotes, injected deterministic synthetic OHLC, an
injected fundamentals_map) -- no network, no real data files touched.
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

import run_accrual_anomaly_mining as driver
import run_batch5_mining as batch5
from mining import trial_ledger
from mining.batch5 import accrual_anomaly_sloan as main_mod

RUN_ID = "accrual-anomaly-test-run"
TESTED_DATE = "2026-09-17"
N_SESSIONS = 1300  # spans several calendar years -- multiple reformation bars
POOL = tuple(f"T{i:02d}" for i in range(12))  # >= MIN_RANK_MEMBERS(10)


def _index(n: int, start: str = "2015-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _flat_frame(index, price=100.0):
    return pd.DataFrame({"Open": price, "High": price * 1.001,
                         "Low": price * 0.999, "Close": price}, index=index)


def _pool_frames():
    spy_index = _index(N_SESSIONS)
    frames = {"SPY": _flat_frame(spy_index)}
    for ticker in POOL:
        frames[ticker] = _flat_frame(spy_index)
    return frames


def _loader(frames):
    def loader(symbol):
        if symbol not in frames:
            raise RuntimeError(f"no synthetic data for {symbol}")
        return frames[symbol].reset_index().rename(columns={"index": "Date"})
    return loader


def _periodic_pit(index, val, *, every=250, extra_cols=None):
    """The SAME value re-disclosed every `every` bars -- keeps a fixed
    accrual leg fresh across a multi-year test window (real annual 10-Ks
    land ~365 days apart; 250 business days ~ 350 calendar days, safely
    under STALENESS_CAP_DAYS=400), matching
    test_batch5_accrual_anomaly_sloan.py's own fixture convention."""
    rows = []
    for i in range(0, len(index), every):
        row = {"usable_asof": pd.Timestamp(index[i]), "latest_val": val}
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return pd.DataFrame(rows)


def _fundamentals_map(frames, *, lowest=-0.60, step=0.10):
    """Each pool ticker gets a FIXED accrual ratio ascending from `lowest`
    (T00 lowest, i.e. the LONG side of Sloan's anomaly). Alphabetical
    ticker order == accrual-ascending order."""
    return {
        ticker: {
            "ni": _periodic_pit(frames[ticker].index, lowest + step * i),
            "cfo": _periodic_pit(frames[ticker].index, 0.0),
            "assets": _periodic_pit(frames[ticker].index, 1.0, extra_cols={"prior_val": np.nan}),
        }
        for i, ticker in enumerate(POOL)
    }


def _quote(bid, ask):
    return {"price": (bid + ask) / 2, "bid": bid, "ask": ask,
            "security_status": "Normal",
            "spread_bps": (ask - bid) / ((ask + bid) / 2) * 1e4, "suspect": False}


def _write_quotes(tmp_path, *, drop=()):
    symbols = ("SPY",) + POOL
    quotes = {s: _quote(100.0, 100.02) for s in symbols if s not in drop}
    payload = {"sampled_at_local": "2026-09-17T16:32:00-04:00",
               "source": "test-quotes", "market_state": "closed (test)",
               "quotes": quotes}
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _write_accepted(tmp_path):
    payload = {"batch": "test", "curated_on": TESTED_DATE,
               "accepted": [{"family_id": main_mod.FAMILY_ID}],
               "infeasible": [], "rejected_by_judge": [], "merged_notes": ""}
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _catalog_row(module, *, status="eligible", parameters=None):
    return {"family_id": module.FAMILY_ID, "name": "test", "mechanism": "test",
            "tickers": ["x"], "parameters": parameters or module.PARAMETERS,
            "account_types": ["taxable"], "status": status, "source": "test",
            "added_date": TESTED_DATE}


def _write_catalog(tmp_path, *, status="eligible", parameters=None):
    entries = [_catalog_row(main_mod, status=status, parameters=parameters)]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return str(path)


def _patch_pool(monkeypatch, frames):
    monkeypatch.setattr(main_mod, "TRADED", list(POOL))
    monkeypatch.setattr(batch5, "default_ohlc_loader", _loader(frames))


def _build(tmp_path, monkeypatch, *, frames=None, drop_quote=(), status="eligible",
          parameters=None):
    frames = frames or _pool_frames()
    _patch_pool(monkeypatch, frames)
    accepted_path = _write_accepted(tmp_path)
    quotes_path = _write_quotes(tmp_path, drop=drop_quote)
    catalog_path = _write_catalog(tmp_path, status=status, parameters=parameters)
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


# --------------------------------------------------------------- end to end

def test_family_scores_with_full_pool_and_fundamentals(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)

    assert not result["skipped"], result["skipped"]
    assert len(result["computed"]) == 1
    item = result["computed"][0]
    assert item["family_id"] == main_mod.FAMILY_ID
    assert set(item["pool_usable"]) == set(POOL)
    assert item["pool_excluded"] == {}
    assert item["tax_policy"] == "none"
    assert item["fundamentals_missing"] == []
    assert item["fundamentals_coverage_fraction"] == pytest.approx(1.0)
    # T00/T01 (bottom quintile of 12 = round(0.2*12)=2) should trade every
    # year-end reformation across the ~5-year synthetic window.
    assert item["n_trades"] >= 1

    rows = _outcomes(built["ledger_path"])
    assert len(rows) == 1
    assert rows[0]["family_id"] == main_mod.FAMILY_ID
    assert rows[0]["construction"] == "ABI-1"


def test_fundamentals_actually_drive_trades(tmp_path, monkeypatch):
    """The whole reason this bespoke driver exists: run_abi1_mining.py's
    generic dispatch would call make_signal_fn with no fundamentals_map
    kwarg, silently scoring every reformation date as UNDEFINED. Prove the
    wiring matters by running the identical fixture twice -- once with a
    real fundamentals_map, once with every ticker PRESENT (so the
    fundamentals_coverage floor is fully cleared -- the separate coverage-
    floor test already covers outright absence) but carrying EMPTY
    per-leg frames, which `accrual_ratio_at` treats as always-undefined --
    and asserting trades collapse to 0 in the second case even though
    every ticker counts as \"covered\"."""
    frames = _pool_frames()
    built_with = _build(tmp_path, monkeypatch, frames=frames)
    with_fundamentals = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-with", dry_run=True,
        fundamentals_map=_fundamentals_map(frames), **built_with)

    built_empty_legs = _build(tmp_path, monkeypatch, frames=frames)
    empty_frame = pd.DataFrame(columns=["usable_asof", "latest_val"])
    always_undefined = {t: {"ni": empty_frame, "cfo": empty_frame, "assets": empty_frame}
                        for t in POOL}
    empty_legs_result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-empty-legs", dry_run=True,
        fundamentals_map=always_undefined, **built_empty_legs)

    item_with = with_fundamentals["computed"][0]
    item_empty = empty_legs_result["computed"][0]
    assert item_with["n_trades"] >= 1
    assert item_empty["fundamentals_missing"] == []  # every ticker "present"
    assert item_empty["n_trades"] == 0


def test_missing_quote_excludes_member(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames, drop_quote=("T05",))
    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)

    assert not result["skipped"], result["skipped"]
    item = result["computed"][0]
    remaining = set(POOL) - {"T05"}
    assert set(item["pool_usable"]) == remaining
    assert "T05" in item["pool_excluded"]


def test_price_excluded_ticker_never_crowds_out_a_tradable_rank_slot(tmp_path, monkeypatch):
    """A ticker excluded from
    pool_frames on price/quote grounds must not still occupy a
    cross-sectional rank slot via a broader fundamentals_map (e.g. one
    cached against the full declared pool before quote exclusions were
    known) -- it can never trade, so it must never be allowed to displace
    a real tradable member from the bottom-fraction selected set.

    Exercises `compute_batch_accrual_anomaly` ITSELF twice (hand-rebuilding
    a signal_fn from an already-filtered map would bypass the driver's own
    filtering and could not catch its absence):

    - "clean": T05 has no quote AND no fundamentals entry at all (never
      even reaches the ranking).
    - "phantom": T05 has no quote (excluded from pool_frames, identical
      to "clean") but DOES carry an EXTREME low accrual entry in the
      fundamentals_map passed to the driver.

    The driver filters fundamentals_map to pool_frames before ranking, so
    "phantom" must produce the IDENTICAL n_trades as "clean" -- T05's
    phantom entry never reaches selection_panel. Without that filter,
    "phantom" would rank 12 defined tickers (round(0.20*12)=2 ->
    {T05, T00}, crowding T01 out) while "clean" ranks 11 (round(0.20*11)=2
    -> {T00, T01}) -- a measurable n_trades difference, which this
    assertion catches."""
    frames = _pool_frames()
    fundamentals_full = _fundamentals_map(frames)
    fundamentals_clean = {t: v for t, v in fundamentals_full.items() if t != "T05"}
    fundamentals_phantom = dict(fundamentals_full)
    fundamentals_phantom["T05"] = {
        "ni": _periodic_pit(frames["T05"].index, -999.0),
        "cfo": _periodic_pit(frames["T05"].index, 0.0),
        "assets": _periodic_pit(frames["T05"].index, 1.0, extra_cols={"prior_val": np.nan}),
    }

    built_clean = _build(tmp_path, monkeypatch, frames=frames, drop_quote=("T05",))
    clean_result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-clean", dry_run=True,
        fundamentals_map=fundamentals_clean, **built_clean)

    built_phantom = _build(tmp_path, monkeypatch, frames=frames, drop_quote=("T05",))
    phantom_result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-phantom", dry_run=True,
        fundamentals_map=fundamentals_phantom, **built_phantom)

    assert not clean_result["skipped"], clean_result["skipped"]
    assert not phantom_result["skipped"], phantom_result["skipped"]
    clean_item = clean_result["computed"][0]
    phantom_item = phantom_result["computed"][0]
    assert "T05" not in clean_item["pool_usable"]
    assert "T05" not in phantom_item["pool_usable"]
    assert clean_item["n_trades"] >= 1
    # The discriminating assertion: T05's phantom fundamentals entry must
    # not change how many trades the REAL, tradable pool produces.
    assert phantom_item["n_trades"] == clean_item["n_trades"]


def test_pool_coverage_floor_refuses_the_family(tmp_path, monkeypatch):
    frames = _pool_frames()
    # Drop enough quotes to fall below the 80% price/quote coverage floor
    # (3 of 12 = 75%).
    built = _build(tmp_path, monkeypatch, frames=frames,
                   drop_quote=("T00", "T01", "T02"))
    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert result["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "pool_coverage"


def test_fundamentals_coverage_floor_refuses_without_forcing(tmp_path, monkeypatch):
    """The family's price/quote pool clears MIN_POOL_COVERAGE but its
    FUNDAMENTALS coverage does not -- the run must refuse rather than
    silently score a too-thin cross-section."""
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    full_fundamentals = _fundamentals_map(frames)
    # Keep fundamentals for only 5 of 12 (< 80% floor), matching every
    # OTHER ticker missing entirely (never a raise, just absent).
    thin_fundamentals = {t: full_fundamentals[t] for t in POOL[:5]}
    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=thin_fundamentals, **built)
    assert result["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "fundamentals_coverage"


def test_dry_run_writes_no_ledger_rows(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=True,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert not os.path.exists(built["ledger_path"])
    assert len(result["computed"]) == 1
    assert result["computed"][0]["stage"] is not None


def test_already_scored_family_is_skipped(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    first = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert len(first["computed"]) == 1

    second = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID + "-2", dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert second["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in second["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "already_scored_abi1"
    assert len(_outcomes(built["ledger_path"])) == 1


def test_catalog_status_not_eligible_skips(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames, status="ruled_out")
    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert result["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "catalog_status"


def test_parameter_drift_against_catalog_skips(tmp_path, monkeypatch):
    frames = _pool_frames()
    drifted = dict(main_mod.PARAMETERS)
    drifted["bottom_fraction"] = 0.99
    built = _build(tmp_path, monkeypatch, frames=frames, parameters=drifted)
    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert result["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "parameter_drift"


def test_not_in_accepted_skips(tmp_path, monkeypatch):
    frames = _pool_frames()
    built = _build(tmp_path, monkeypatch, frames=frames)
    payload = json.loads(open(built["accepted_path"], encoding="utf-8").read())
    payload["accepted"] = []
    with open(built["accepted_path"], "w", encoding="utf-8") as f:
        json.dump(payload, f)

    result = driver.compute_batch_accrual_anomaly(
        tested_date=TESTED_DATE, run_id=RUN_ID, dry_run=False,
        fundamentals_map=_fundamentals_map(frames), **built)
    assert result["computed"] == []
    codes = {row["family_id"]: row["reason_code"] for row in result["skipped"]}
    assert codes[main_mod.FAMILY_ID] == "not_in_accepted"


# ----------------------------------------------------------- fundamentals

def test_build_fundamentals_map_reports_coverage_without_excluding(monkeypatch):
    idx = _index(600)

    def fake_check(ticker, cik, *, fetch_tags_fn, fetch_instant_tags_fn):
        if cik == 1:
            fm = {"ni": _periodic_pit(idx, -0.10), "cfo": _periodic_pit(idx, 0.0),
                 "assets": _periodic_pit(idx, 1.0, extra_cols={"prior_val": np.nan})}
            return {"ticker": ticker, "cik": cik, "status": "ok",
                    "ni_series": fm["ni"], "cfo_series": fm["cfo"],
                    "assets_series": fm["assets"]}
        if cik == 2:
            return {"ticker": ticker, "cik": cik, "status": "no_usable_net_income"}
        if cik == 3:
            return {"ticker": ticker, "cik": cik, "status": "fetch_error", "detail": "boom"}
        return {"ticker": ticker, "cik": None, "status": "no_cik"}

    monkeypatch.setattr(driver.accrual_coverage, "check_ticker_coverage", fake_check)
    fundamentals_map, coverage_results = driver.build_fundamentals_map(
        ["OK", "EMPTY", "ERROR", "NOCIK"],
        cik_by_ticker={"OK": 1, "EMPTY": 2, "ERROR": 3, "NOCIK": None},
        sleep=lambda s: None, min_interval=0.0)

    assert set(fundamentals_map) == {"OK"}
    assert coverage_results["OK"]["status"] == "ok"
    assert coverage_results["EMPTY"]["status"] == "no_usable_net_income"
    assert coverage_results["ERROR"]["status"] == "fetch_error"
    assert coverage_results["NOCIK"]["status"] == "no_cik"


def test_main_cli_defaults_are_well_formed():
    assert os.path.isabs(driver.DEFAULT_ACCEPTED)
    assert os.path.isabs(driver.DEFAULT_QUOTES)
    assert driver.MARKET_BENCHMARK == "SPY"
    assert driver.TAX_POLICY == "none"
    assert driver.MIN_FUNDAMENTALS_COVERAGE == pytest.approx(0.80)
