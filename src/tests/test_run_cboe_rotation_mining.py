"""Tests for scripts/run_cboe_rotation_mining.py (the bespoke ABI-1 driver
for Cboe index-overlay rotation families with an assumed, not measured,
cost on their non-tradable leg).

Everything runs on tmp copies: tmp ledger, tmp catalog, tmp quotes, tmp
accepted.json, a tmp modules dir of fake candidate modules, and injected
deterministic synthetic OHLC (no network). ``ASSUMED_COST_REGISTRY`` is
monkeypatched per test to a synthetic family, so the tests exercise the
driver's own dispatch logic rather than depending on the real
``cboe_putwrite_vol_regime_rotation`` module or its real registry entry.
"""
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_cboe_rotation_mining as cboe_cli
import run_batch5_mining as batch5
from mining import evidence, trial_ledger

RUN_ID = "cboe-rotation-test-run"
TESTED_DATE = "2026-09-09"
N_SESSIONS = 400
SYMBOLS = ("SPY", "RSK", "DEF")


# ------------------------------------------------------------ synthetic data

def _index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _frame(n: int, seed: int, start: str = "2020-01-02") -> pd.DataFrame:
    index = _index(n, start)
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0004, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


_FRAMES = {symbol: _frame(N_SESSIONS, seed)
          for seed, symbol in enumerate(SYMBOLS, start=11)}


def _loader_with_counter():
    calls = []

    def loader(symbol):
        calls.append(symbol)
        if symbol not in _FRAMES:
            raise RuntimeError(f"no synthetic data for {symbol}")
        return _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})
    return loader, calls


# ---------------------------------------------------------- fake module

_HEADER = "from mining.backtest_engine import SignalResult\n"

FAKE_FAMILY_ID = "cboe_test_rotation"
MODULE_ROTATION = _HEADER + f"""
FAMILY_ID = "{FAKE_FAMILY_ID}"
SHAPE = "rotation_two_asset"
TRADED = ["RSK", "DEF"]
SIGNAL = []
SNAPSHOTS = {{}}
BENCHMARK = "RSK"
PARAMETERS = {{"block": 40}}
EXPECTED_MIN_TRADES = 5

def make_target_fn(frames):
    def target(truncated):
        n = len(truncated["RSK"])
        return "RSK" if (n // 40) % 2 == 0 else "DEF"
    return target
"""

CATALOG_PARAMETERS = {FAKE_FAMILY_ID: {"block": 40}}
TRADED_BY_ID = {FAKE_FAMILY_ID: ["RSK", "DEF"]}


def _write_modules(tmp_path, modules: dict) -> str:
    modules_dir = tmp_path / "mods"
    modules_dir.mkdir(exist_ok=True)
    for name, text in modules.items():
        (modules_dir / f"{name}.py").write_text(textwrap.dedent(text), encoding="utf-8")
    return str(modules_dir)


def _write_catalog(tmp_path, family_ids, *, account_types=None) -> str:
    entries = []
    for family_id in family_ids:
        entries.append({
            "family_id": family_id, "name": f"Test {family_id}",
            "mechanism": f"synthetic mechanism for {family_id}",
            "tickers": TRADED_BY_ID[family_id],
            "parameters": CATALOG_PARAMETERS[family_id],
            "account_types": (account_types or {}).get(family_id, ["taxable"]),
            "status": "eligible", "source": "test", "added_date": "2026-09-09",
        })
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return str(path)


def _write_accepted(tmp_path, family_ids) -> str:
    payload = {"batch": "cboe-rotation-test", "curated_on": TESTED_DATE,
               "accepted": [{"family_id": fid, "name": f"Test {fid}",
                             "tickers": TRADED_BY_ID[fid],
                             "parameters": CATALOG_PARAMETERS[fid]}
                            for fid in family_ids],
               "infeasible": [], "rejected_by_judge": [], "merged_notes": ""}
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _quote(bid, ask, suspect=False):
    return {"price": (bid + ask) / 2, "bid": bid, "ask": ask,
            "security_status": "Normal",
            "spread_bps": (ask - bid) / ((ask + bid) / 2) * 1e4,
            "suspect": suspect}


def _write_quotes(tmp_path, *, drop=(), spread=0.02) -> str:
    quotes = {symbol: _quote(100.0, 100.0 + spread)
             for symbol in SYMBOLS if symbol not in drop}
    payload = {"sampled_at_local": "2026-09-09T16:32:00-04:00",
               "source": "test-quotes", "market_state": "closed (test)",
               "quotes": quotes}
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _build_scope(tmp_path, monkeypatch, *, modules: dict, accepted: list,
                 registry: dict, loader=None, ledger=None,
                 quotes_kwargs=None) -> dict:
    loader = loader or _loader_with_counter()[0]
    monkeypatch.setattr(batch5, "default_ohlc_loader", loader)
    monkeypatch.setattr(cboe_cli, "ASSUMED_COST_REGISTRY", registry)
    modules_dir = _write_modules(tmp_path, modules)
    catalog_path = _write_catalog(tmp_path, accepted)
    accepted_path = _write_accepted(tmp_path, accepted)
    quotes_path = _write_quotes(tmp_path, **(quotes_kwargs or {}))
    ledger = ledger or str(tmp_path / "ledger.jsonl")
    research_root = str(tmp_path / "research")
    run_dir = os.path.join(research_root, RUN_ID)
    scope = batch5.build_scope(
        accepted_path=accepted_path, quotes_path=quotes_path,
        catalog_path=catalog_path, ledger_path=ledger,
        modules_dir=modules_dir, run_dir=run_dir,
        families=accepted, ohlc_loader=loader)
    return {"scope": scope, "ledger": ledger, "run_dir": run_dir}


def _by_id(result):
    return {item["family_id"]: item for item in result["computed"]}


def _skips(result):
    return {row["family_id"]: row for row in result["skipped"]}


def _outcomes(ledger_path, family_id=None):
    if not os.path.exists(ledger_path):
        return []
    rows = [r for r in trial_ledger.read_all(ledger_path) if r["event"] == "outcome"]
    if family_id is not None:
        rows = [r for r in rows if r["family_id"] == family_id]
    return rows


# --------------------------------------------------------------- costs_for_family

def _bare_quotes_payload(quotes: dict) -> dict:
    return {"sampled_at_local": "2026-09-09T16:32:00-04:00",
            "source": "test-quotes", "market_state": "closed (test)",
            "quotes": quotes}


def test_costs_for_family_uses_assumed_constant_for_registered_symbol_with_no_quote(monkeypatch):
    monkeypatch.setattr(cboe_cli, "ASSUMED_COST_REGISTRY", {FAKE_FAMILY_ID: {"DEF": 30.0}})
    quotes_payload = _bare_quotes_payload({"RSK": _quote(100.0, 100.02)})  # no DEF quote
    costs = cboe_cli.costs_for_family(
        FAKE_FAMILY_ID, ["RSK", "DEF"], quotes_payload)
    assert "spread_bps" in costs["RSK"] and costs["RSK"]["source"] != "assumed_constant"
    assert costs["DEF"]["source"] == "assumed_constant"
    assert costs["DEF"]["bid"] is None and costs["DEF"]["ask"] is None
    assert costs["DEF"]["cost_bps"] == pytest.approx(30.0)


def test_costs_for_family_still_requires_a_real_quote_for_an_unregistered_symbol():
    quotes_payload = _bare_quotes_payload({})  # RSK has no quote and is not in the registry
    with pytest.raises(ValueError, match="RSK"):
        cboe_cli.costs_for_family(FAKE_FAMILY_ID, ["RSK", "DEF"], quotes_payload)


def test_costs_for_family_with_no_registry_entry_requires_real_quotes_for_everything():
    quotes_payload = _bare_quotes_payload({"RSK": _quote(100.0, 100.02),
                                           "DEF": _quote(100.0, 100.02)})
    costs = cboe_cli.costs_for_family("some_other_family", ["RSK", "DEF"], quotes_payload)
    assert costs["RSK"]["source"] != "assumed_constant"
    assert costs["DEF"]["source"] != "assumed_constant"


# ------------------------------------------------------ calendar alignment

def _plain_frame(index):
    n = len(index)
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5,
                         "Close": close}, index=index)


def test_intersect_traded_calendars_is_a_no_op_on_identical_calendars():
    index = _index(50)
    frames = {"A": _plain_frame(index), "B": _plain_frame(index)}
    aligned, dropped = cboe_cli._intersect_traded_calendars(frames)
    assert dropped == {"A": 0, "B": 0}
    assert aligned["A"].index.equals(index)
    assert aligned["B"].index.equals(index)


def test_intersect_traded_calendars_a_leg_starting_later_is_not_counted_as_dropped():
    """Guards against mistaking a late start for a gap: SPY starts 1993, PUT
    starts 1996 -- a leg simply beginning later must NOT be mistaken for a ~900-day
    'gap' (that's exactly what abi1_driver._align_rotation_frames's own
    common-inception trim already handles correctly one layer down)."""
    long_history = _index(300, start="2010-01-04")
    short_history = long_history[200:]           # the "PUT-like" late starter
    frame_long = _plain_frame(long_history)
    frame_short = _plain_frame(short_history)
    aligned, dropped = cboe_cli._intersect_traded_calendars(
        {"LONG": frame_long, "SHORT": frame_short})
    assert dropped == {"LONG": 0, "SHORT": 0}     # a later start is not a "drop"
    assert len(aligned["LONG"]) == len(aligned["SHORT"]) == len(short_history)
    assert aligned["LONG"].index.equals(short_history)


def test_intersect_traded_calendars_refuses_a_gap_larger_than_the_sparse_allowance():
    full = _index(100)
    # A contiguous 30-session block missing from B -- the ^BXM-style live
    # feed outage this guard exists to catch, not paper over.
    outage = full[40:70]
    frame_a = _plain_frame(full)
    frame_b = _plain_frame(full).drop(index=outage)
    assert len(outage) > cboe_cli.MAX_SPARSE_CALENDAR_GAP_DAYS
    with pytest.raises(ValueError, match="feed outage"):
        cboe_cli._intersect_traded_calendars({"A": frame_a, "B": frame_b})


def test_intersect_traded_calendars_drops_sparse_days_missing_from_either_leg():
    full = _index(50)
    missing_from_b = [full[10], full[30]]      # B has no print on these days
    missing_from_a = [full[5]]                 # A has no print on this day
    frame_a = _plain_frame(full).drop(index=missing_from_a)
    frame_b = _plain_frame(full).drop(index=missing_from_b)
    aligned, dropped = cboe_cli._intersect_traded_calendars({"A": frame_a, "B": frame_b})
    assert dropped == {"A": 2, "B": 1}          # each leg loses the OTHER's gaps
    assert len(aligned["A"]) == len(aligned["B"]) == 47
    assert aligned["A"].index.equals(aligned["B"].index)
    for stamp in missing_from_a + missing_from_b:
        assert stamp not in aligned["A"].index
        assert stamp not in aligned["B"].index


# --------------------------------------------------------------- end to end

def test_rotation_family_scores_with_assumed_cost_on_the_unquoted_leg(tmp_path, monkeypatch):
    """DEF is dropped from the quotes file entirely -- the family must
    still score, proving the assumed-cost path genuinely bypasses the
    quote gate rather than merely working because a real quote happens
    to exist too."""
    built = _build_scope(
        tmp_path, monkeypatch, modules={FAKE_FAMILY_ID: MODULE_ROTATION},
        accepted=[FAKE_FAMILY_ID],
        registry={FAKE_FAMILY_ID: {"DEF": 30.0}},
        quotes_kwargs={"drop": ("DEF",)})

    result = cboe_cli.compute_batch_cboe_rotation(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False)

    assert not result["skipped"], result["skipped"]
    item = _by_id(result)[FAKE_FAMILY_ID]
    assert item["shape"] == "rotation_two_asset"
    assert item["tickers"] == ["RSK", "DEF"]
    assert item["tax_policy"] == "realization"
    assert "DEF=30.00bps (assumed_constant)" in item["cost_model_note"]
    assert item["stage"] in evidence.VALID_STAGES

    rows = _outcomes(built["ledger"], FAKE_FAMILY_ID)
    assert len(rows) == 1
    assert rows[0]["construction"] == "ABI-1"
    # The permanent ledger row itself must carry the assumed-cost
    # disclosure -- not just this run's own JSON/report -- or a future
    # reader of the ledger alone could mistake DEF's cost for a measured
    # quote.
    assert "DEF=30.00bps (assumed_constant)" in rows[0]["note"]


def test_rotation_family_end_to_end_with_a_real_sparse_calendar_gap(tmp_path, monkeypatch):
    """End-to-end case: the isolated _intersect_traded_calendars
    unit tests do not drive a REAL gap through the full compute_batch_
    cboe_rotation -> abi1_driver.build_rotation_curve pipeline (the
    synthetic end-to-end fixture above uses identical calendars for every
    symbol, making the intersection step a no-op there). This test gives
    DEF a handful of missing interior sessions RSK still has -- exactly
    the ^PUT-vs-SPY shape -- and drives it through the real
    driver, not just the helper function in isolation."""
    gap_dates = [_FRAMES["DEF"].index[50], _FRAMES["DEF"].index[120],
                _FRAMES["DEF"].index[121], _FRAMES["DEF"].index[300]]
    gapped_def = _FRAMES["DEF"].drop(index=gap_dates)

    def loader(symbol):
        frame = gapped_def if symbol == "DEF" else _FRAMES[symbol]
        return frame.reset_index().rename(columns={"index": "Date"})

    built = _build_scope(
        tmp_path, monkeypatch, modules={FAKE_FAMILY_ID: MODULE_ROTATION},
        accepted=[FAKE_FAMILY_ID], registry={FAKE_FAMILY_ID: {"DEF": 30.0}},
        loader=loader, quotes_kwargs={"drop": ("DEF",)})

    result = cboe_cli.compute_batch_cboe_rotation(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False)

    assert not result["skipped"], result["skipped"]
    item = _by_id(result)[FAKE_FAMILY_ID]
    assert item["n_sessions"] == N_SESSIONS - len(gap_dates)
    assert "cross-vendor calendar alignment dropped RSK=4" in item["cost_model_note"]
    rows = _outcomes(built["ledger"], FAKE_FAMILY_ID)
    assert "cross-vendor calendar alignment dropped RSK=4" in rows[0]["note"]


def test_intersect_traded_calendars_boundary_at_exactly_the_allowance_passes():
    """Off-by-one guard: exactly MAX_SPARSE_CALENDAR_GAP_DAYS dropped must
    still pass (the guard is '> allowance', not '>= allowance')."""
    full = _index(200)
    outage = full[40:40 + cboe_cli.MAX_SPARSE_CALENDAR_GAP_DAYS]
    assert len(outage) == cboe_cli.MAX_SPARSE_CALENDAR_GAP_DAYS
    frame_a = _plain_frame(full)
    frame_b = _plain_frame(full).drop(index=outage)
    aligned, dropped = cboe_cli._intersect_traded_calendars({"A": frame_a, "B": frame_b})
    assert dropped == {"A": cboe_cli.MAX_SPARSE_CALENDAR_GAP_DAYS, "B": 0}


def test_intersect_traded_calendars_boundary_one_over_the_allowance_refuses():
    full = _index(200)
    outage = full[40:41 + cboe_cli.MAX_SPARSE_CALENDAR_GAP_DAYS]
    assert len(outage) == cboe_cli.MAX_SPARSE_CALENDAR_GAP_DAYS + 1
    frame_a = _plain_frame(full)
    frame_b = _plain_frame(full).drop(index=outage)
    with pytest.raises(ValueError, match="feed outage"):
        cboe_cli._intersect_traded_calendars({"A": frame_a, "B": frame_b})


def test_costs_for_family_assumed_constant_wins_even_when_a_real_quote_exists(monkeypatch):
    """A symbol registered in ASSUMED_COST_REGISTRY always uses the assumed
    constant, even if (unexpectedly) a real quote is also present for it --
    the registry, not quote presence, is what decides the pricing path."""
    monkeypatch.setattr(cboe_cli, "ASSUMED_COST_REGISTRY", {FAKE_FAMILY_ID: {"DEF": 30.0}})
    quotes_payload = _bare_quotes_payload({"RSK": _quote(100.0, 100.02),
                                           "DEF": _quote(50.0, 50.01)})
    costs = cboe_cli.costs_for_family(FAKE_FAMILY_ID, ["RSK", "DEF"], quotes_payload)
    assert costs["DEF"]["source"] == "assumed_constant"
    assert costs["DEF"]["cost_bps"] == pytest.approx(30.0)
    assert costs["DEF"]["bid"] is None  # the real 50.0/50.01 quote is ignored entirely


def test_family_not_in_registry_is_skipped_not_scored(tmp_path, monkeypatch):
    built = _build_scope(
        tmp_path, monkeypatch, modules={FAKE_FAMILY_ID: MODULE_ROTATION},
        accepted=[FAKE_FAMILY_ID], registry={})  # empty registry

    result = cboe_cli.compute_batch_cboe_rotation(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False)

    assert FAKE_FAMILY_ID not in _by_id(result)
    assert _skips(result)[FAKE_FAMILY_ID]["reason_code"] == "not_registered"
    assert _outcomes(built["ledger"]) == []


def test_dry_run_writes_no_ledger_rows(tmp_path, monkeypatch):
    built = _build_scope(
        tmp_path, monkeypatch, modules={FAKE_FAMILY_ID: MODULE_ROTATION},
        accepted=[FAKE_FAMILY_ID], registry={FAKE_FAMILY_ID: {"DEF": 30.0}},
        quotes_kwargs={"drop": ("DEF",)})

    result = cboe_cli.compute_batch_cboe_rotation(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=True)

    assert not os.path.exists(built["ledger"])
    assert _by_id(result)[FAKE_FAMILY_ID]["stage"] in evidence.VALID_STAGES


def test_already_scored_family_is_skipped_and_not_rewritten(tmp_path, monkeypatch):
    built = _build_scope(
        tmp_path, monkeypatch, modules={FAKE_FAMILY_ID: MODULE_ROTATION},
        accepted=[FAKE_FAMILY_ID], registry={FAKE_FAMILY_ID: {"DEF": 30.0}},
        quotes_kwargs={"drop": ("DEF",)})

    first = cboe_cli.compute_batch_cboe_rotation(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False)
    assert FAKE_FAMILY_ID in _by_id(first)
    assert len(_outcomes(built["ledger"], FAKE_FAMILY_ID)) == 1

    second = cboe_cli.compute_batch_cboe_rotation(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID + "-2",
        ledger_path=built["ledger"], dry_run=False)
    assert FAKE_FAMILY_ID not in _by_id(second)
    assert _skips(second)[FAKE_FAMILY_ID]["reason_code"] == "already_scored_abi1"
    assert len(_outcomes(built["ledger"], FAKE_FAMILY_ID)) == 1  # unchanged


def test_non_rotation_shape_is_skipped_as_unsupported(tmp_path, monkeypatch):
    single_module = _HEADER + f"""
FAMILY_ID = "{FAKE_FAMILY_ID}"
SHAPE = "single_long"
TRADED = ["RSK"]
SIGNAL = []
SNAPSHOTS = {{}}
BENCHMARK = "RSK"
PARAMETERS = {{"period": 15, "hold": 7}}
EXPECTED_MIN_TRADES = 5

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""
    built = _build_scope(
        tmp_path, monkeypatch, modules={FAKE_FAMILY_ID: single_module},
        accepted=[FAKE_FAMILY_ID], registry={FAKE_FAMILY_ID: {"RSK": 30.0}})

    result = cboe_cli.compute_batch_cboe_rotation(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False)

    assert FAKE_FAMILY_ID not in _by_id(result)
    assert _skips(result)[FAKE_FAMILY_ID]["reason_code"] == "unsupported_shape"


# ------------------------------------------------------------- real registry

def test_real_registry_names_the_shipped_putwrite_family_and_assumed_cost():
    from mining.batch5 import cboe_putwrite_vol_regime_rotation as putwrite_mod
    assert putwrite_mod.FAMILY_ID in cboe_cli.ASSUMED_COST_REGISTRY
    assert (cboe_cli.ASSUMED_COST_REGISTRY[putwrite_mod.FAMILY_ID]
            == {"PUT": putwrite_mod.ASSUMED_COST_BPS})
