"""Tests for scripts/run_abi1_mining.py (the ABI-1 full-history CLI driver).

Everything runs on tmp copies: tmp ledger, tmp catalog, tmp quotes, tmp
accepted.json, a tmp modules dir of fake candidate modules, and injected
deterministic synthetic OHLC (no network). The real
data/mining/trial_ledger.jsonl and data/mining/catalog.json are never
touched, and no real market data is ever fetched.
"""
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts"),
             os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_abi1_mining as abi1_cli
import run_batch5_mining as batch5
from _portable_paths_support import DRIVE_PATH, workstation_leaks
from mining import account_relative, constants, evidence, trial_ledger

RUN_ID = "abi1-test-run"
TESTED_DATE = "2026-09-09"

# 400 sessions clears account_relative.MIN_ACCOUNT_SESSIONS (252) so layer 1
# actually scores, and is well below ABI1_HOLDOUT_SESSIONS (756) + 252, so
# the holdout split refuses and layer 1 falls back to scoring the FULL
# window -- the identical cheap-fixture recipe test_abi1_driver.py's own
# record_abi1_outcome test uses (no need for a multi-thousand-session
# fixture just to prove the CLI's wiring).
N_SESSIONS = 400
SYMBOLS = ("SPY", "AAA", "BBB", "CCC", "RSK", "DEF")


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
# A recently-listed instrument: only the LAST 40 sessions of the shared
# calendar, so its overlap with SPY's own frame is well-defined and short --
# 40 < MIN_ACCOUNT_SESSIONS (252), which is what drives the not_scoreable
# path end to end.
_FRAMES["SHORT"] = _frame(N_SESSIONS, seed=99).iloc[-40:].copy()


def _loader_with_counter():
    calls = []

    def loader(symbol):
        calls.append(symbol)
        if symbol not in _FRAMES:
            raise RuntimeError(f"no synthetic data for {symbol}")
        # Mimic data_pipeline.fetch_ohlc's shape: a Date COLUMN, not an index.
        return _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})
    return loader, calls


# ---------------------------------------------------------- fake modules

_HEADER = "from mining.backtest_engine import SignalResult\n"

MODULE_SINGLE = _HEADER + """
FAMILY_ID = "abi1_single"
SHAPE = "single_long"
TRADED = ["AAA"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 5

def make_signal_fn(ticker, index, frames):
    assert ticker in frames
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""

MODULE_SINGLE_2 = _HEADER + """
FAMILY_ID = "abi1_single_2"
SHAPE = "single_long"
TRADED = ["DEF"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 5

def make_signal_fn(ticker, index, frames):
    assert ticker in frames
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""

MODULE_SHORT = _HEADER + """
FAMILY_ID = "abi1_short"
SHAPE = "single_long"
TRADED = ["SHORT"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 1

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""

MODULE_POOLED = _HEADER + """
FAMILY_ID = "abi1_pooled"
SHAPE = "pooled_single_asset"
TRADED = ["AAA", "BBB", "CCC"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 15

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""

MODULE_ROTATION = _HEADER + """
FAMILY_ID = "abi1_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["RSK", "DEF"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "RSK"
PARAMETERS = {"block": 40}
EXPECTED_MIN_TRADES = 5

def make_target_fn(frames):
    def target(truncated):
        n = len(truncated["RSK"])
        return "RSK" if (n // 40) % 2 == 0 else "DEF"
    return target
"""

MODULE_ROTATION_WITH_SIGNAL = _HEADER + """
FAMILY_ID = "abi1_rotation_with_signal"
SHAPE = "rotation_two_asset"
TRADED = ["RSK", "DEF"]
SIGNAL = ["SIG"]
SNAPSHOTS = {}
BENCHMARK = "RSK"
PARAMETERS = {"block": 40}
EXPECTED_MIN_TRADES = 5

def make_target_fn(frames):
    assert "SIG" in frames
    def target(truncated):
        n = len(truncated["RSK"])
        return "RSK" if (n // 40) % 2 == 0 else "DEF"
    return target
"""

CATALOG_PARAMETERS = {
    "abi1_single": {"period": 15, "hold": 7},
    "abi1_single_2": {"period": 15, "hold": 7},
    "abi1_short": {"period": 15, "hold": 7},
    "abi1_pooled": {"period": 15, "hold": 7},
    "abi1_rotation": {"block": 40},
    "abi1_rotation_with_signal": {"block": 40},
}
TRADED_BY_ID = {
    "abi1_single": ["AAA"], "abi1_single_2": ["DEF"], "abi1_short": ["SHORT"],
    "abi1_pooled": ["AAA", "BBB", "CCC"], "abi1_rotation": ["RSK", "DEF"],
    "abi1_rotation_with_signal": ["RSK", "DEF"],
}


def _write_modules(tmp_path, modules: dict) -> str:
    modules_dir = tmp_path / "mods"
    modules_dir.mkdir(exist_ok=True)
    for name, text in modules.items():
        (modules_dir / f"{name}.py").write_text(textwrap.dedent(text), encoding="utf-8")
    return str(modules_dir)


def _write_catalog(tmp_path, family_ids, *, account_types=None, filename="catalog.json") -> str:
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
    path = tmp_path / filename
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return str(path)


def _write_accepted(tmp_path, family_ids, filename="accepted.json") -> str:
    payload = {"batch": "abi1-v1-test", "curated_on": TESTED_DATE,
               "accepted": [{"family_id": fid, "name": f"Test {fid}",
                             "tickers": TRADED_BY_ID[fid],
                             "parameters": CATALOG_PARAMETERS[fid]}
                            for fid in family_ids],
               "infeasible": [], "rejected_by_judge": [], "merged_notes": ""}
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _quote(bid, ask, suspect=False):
    return {"price": (bid + ask) / 2, "bid": bid, "ask": ask,
            "security_status": "Normal",
            "spread_bps": (ask - bid) / ((ask + bid) / 2) * 1e4,
            "suspect": suspect}


def _write_quotes(tmp_path, *, drop=(), spread=0.02, filename="quotes.json") -> str:
    symbols = SYMBOLS + ("SHORT",)
    quotes = {symbol: _quote(100.0, 100.0 + spread)
             for symbol in symbols if symbol not in drop}
    payload = {"sampled_at_local": "2026-09-09T16:32:00-04:00",
               "source": "test-quotes", "market_state": "closed (test)",
               "quotes": quotes}
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _build_scope(tmp_path, monkeypatch, *, modules: dict, accepted: list,
                 loader=None, ledger=None, families=None, quotes_kwargs=None,
                 research_root=None) -> dict:
    loader = loader or _loader_with_counter()[0]
    monkeypatch.setattr(batch5, "default_ohlc_loader", loader)
    modules_dir = _write_modules(tmp_path, modules)
    catalog_path = _write_catalog(tmp_path, accepted)
    accepted_path = _write_accepted(tmp_path, accepted)
    quotes_path = _write_quotes(tmp_path, **(quotes_kwargs or {}))
    ledger = ledger or str(tmp_path / "ledger.jsonl")
    research_root = research_root or str(tmp_path / "research")
    run_dir = os.path.join(research_root, RUN_ID)
    scope = batch5.build_scope(
        accepted_path=accepted_path, quotes_path=quotes_path,
        catalog_path=catalog_path, ledger_path=ledger,
        modules_dir=modules_dir, run_dir=run_dir,
        families=families or [], ohlc_loader=loader)
    return {"scope": scope, "ledger": ledger, "run_dir": run_dir,
            "catalog_path": catalog_path}


def _run_compute(tmp_path, monkeypatch, *, modules, accepted, dry_run=False,
                 **kwargs):
    built = _build_scope(tmp_path, monkeypatch, modules=modules,
                         accepted=accepted, **kwargs)
    result = abi1_cli.compute_batch_abi1(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=dry_run)
    return result, built


def _by_id(result):
    return {item["family_id"]: item for item in result["computed"]}


def _skips(result):
    return {row["family_id"]: row for row in result["skipped"]}


def _outcomes(ledger_path, family_id=None):
    rows = [r for r in trial_ledger.read_all(ledger_path) if r["event"] == "outcome"]
    if family_id is not None:
        rows = [r for r in rows if r["family_id"] == family_id]
    return rows


# --------------------------------------------------------- end to end (real)

def test_single_long_end_to_end_writes_exactly_one_abi1_ledger_row(tmp_path, monkeypatch):
    result, built = _run_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"])

    item = _by_id(result)["abi1_single"]
    assert item["shape"] == "single_long"
    assert item["tickers"] == ["AAA"]
    assert item["ticker_label"] is None
    assert item["tax_policy"] == "realization"
    assert item["stage"] in evidence.VALID_STAGES
    assert item["n_sessions"] == N_SESSIONS

    rows = _outcomes(built["ledger"], "abi1_single")
    assert len(rows) == 1
    assert rows[0]["construction"] == "ABI-1"
    assert rows[0]["stage"] == item["stage"]
    assert rows[0]["ticker_label"] is None
    assert "cash_convention=sweep_cash_daily" in (rows[0]["note"] or "")


def test_pooled_shape_scores_untaxed_with_a_pool_ticker_label(tmp_path, monkeypatch):
    result, built = _run_compute(
        tmp_path, monkeypatch, modules={"abi1_pooled": MODULE_POOLED},
        accepted=["abi1_pooled"])

    item = _by_id(result)["abi1_pooled"]
    assert item["shape"] == "pooled_single_asset"
    assert set(item["tickers"]) == {"AAA", "BBB", "CCC"}
    assert item["ticker_label"] == "AAA/BBB/CCC"
    assert item["tax_policy"] == "none"

    rows = _outcomes(built["ledger"], "abi1_pooled")
    assert len(rows) == 1
    assert rows[0]["ticker_label"] == "AAA/BBB/CCC"
    assert rows[0]["family_id"] == "abi1_pooled"   # NEVER the display label
    assert "tax_policy=none" in rows[0]["note"]


def test_rotation_shape_end_to_end(tmp_path, monkeypatch):
    result, built = _run_compute(
        tmp_path, monkeypatch, modules={"abi1_rotation": MODULE_ROTATION},
        accepted=["abi1_rotation"])

    item = _by_id(result)["abi1_rotation"]
    assert item["shape"] == "rotation_two_asset"
    assert item["tickers"] == ["RSK", "DEF"]
    assert item["tax_policy"] == "realization"

    rows = _outcomes(built["ledger"], "abi1_rotation")
    assert len(rows) == 1


def test_rotation_shape_with_a_negative_signal_value_still_scores(tmp_path, monkeypatch):
    """A signal-only symbol (e.g. a real yield index like ^IRX, which had
    genuine negative closes in March 2020) must not make the whole family
    unscoreable: the upfront OHLC loader must load a symbol that is NEVER
    traded anywhere in this run's plans with require_positive=False, so a
    real negative value there does not raise and skip the family."""
    def loader(symbol):
        if symbol == "SIG":
            frame = _frame(N_SESSIONS, seed=77).copy()
            frame.loc[frame.index[5], "Close"] = -0.05
            frame.loc[frame.index[5], "Open"] = -0.04
            return frame.reset_index().rename(columns={"index": "Date"})
        if symbol not in _FRAMES:
            raise RuntimeError(f"no synthetic data for {symbol}")
        return _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})

    result, built = _run_compute(
        tmp_path, monkeypatch,
        modules={"abi1_rotation_with_signal": MODULE_ROTATION_WITH_SIGNAL},
        accepted=["abi1_rotation_with_signal"], loader=loader)

    assert _skips(result) == {}
    item = _by_id(result)["abi1_rotation_with_signal"]
    assert item["shape"] == "rotation_two_asset"
    rows = _outcomes(built["ledger"], "abi1_rotation_with_signal")
    assert len(rows) == 1


def test_a_symbol_traded_by_one_family_stays_positivity_strict_even_if_signal_elsewhere(
        tmp_path, monkeypatch):
    """The require_positive relaxation is scoped to symbols that are NEVER
    traded anywhere in the run -- if the same symbol is TRADED by any family
    in the batch, it must keep the strict positivity gate for every family,
    signal-only uses included. RSK is traded by abi1_rotation; give it a
    negative value and confirm the family is skipped as missing_data rather
    than silently scored on bad data."""
    def loader(symbol):
        if symbol == "RSK":
            frame = _FRAMES["RSK"].copy()
            frame.loc[frame.index[5], "Close"] = -0.05
            return frame.reset_index().rename(columns={"index": "Date"})
        if symbol not in _FRAMES:
            raise RuntimeError(f"no synthetic data for {symbol}")
        return _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})

    result, built = _run_compute(
        tmp_path, monkeypatch, modules={"abi1_rotation": MODULE_ROTATION},
        accepted=["abi1_rotation"], loader=loader)

    assert "abi1_rotation" in _skips(result)
    assert _skips(result)["abi1_rotation"]["reason_code"] == "missing_data"


def test_not_scoreable_family_records_abi1_not_scoreable_reason_code(tmp_path, monkeypatch):
    result, built = _run_compute(
        tmp_path, monkeypatch, modules={"abi1_short": MODULE_SHORT},
        accepted=["abi1_short"])

    item = _by_id(result)["abi1_short"]
    assert item["verdict"] == "not_scoreable"
    assert item["stage"] == "insufficient_evidence"
    assert item["n_sessions"] == 40

    rows = _outcomes(built["ledger"], "abi1_short")
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "insufficient_evidence"
    assert row["dsr_value"] is None
    assert row["evidence"]["reason_code"] == "abi1_not_scoreable"
    assert row["evidence"]["observed"]["n_sessions"] == 40
    evidence.validate(row["evidence"], stage="insufficient_evidence", at_write_time=False)


# ----------------------------------------------------------------- dry run

def test_dry_run_writes_no_ledger_rows_but_still_reports_a_verdict(tmp_path, monkeypatch):
    result, built = _run_compute(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_short": MODULE_SHORT},
        accepted=["abi1_single", "abi1_short"], dry_run=True)

    assert not os.path.exists(built["ledger"])
    by_id = _by_id(result)
    assert by_id["abi1_single"]["stage"] in evidence.VALID_STAGES
    assert by_id["abi1_short"]["stage"] == "insufficient_evidence"


# -------------------------------------------------------------- re-run guard

def test_a_family_already_scored_under_abi1_is_skipped_and_not_rewritten(tmp_path, monkeypatch):
    built = _build_scope(tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
                         accepted=["abi1_single"])
    first = abi1_cli.compute_batch_abi1(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False)
    assert "abi1_single" in _by_id(first)
    first_rows = _outcomes(built["ledger"], "abi1_single")
    assert len(first_rows) == 1

    second = abi1_cli.compute_batch_abi1(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID + "-2",
        ledger_path=built["ledger"], dry_run=False)
    assert "abi1_single" not in _by_id(second)
    skip = _skips(second)["abi1_single"]
    assert skip["reason_code"] == "already_scored_abi1"
    second_rows = _outcomes(built["ledger"], "abi1_single")
    assert len(second_rows) == 1          # unchanged -- no duplicate write


def test_re_run_guard_ignores_a_pre_trade_construction_row_for_the_same_family(tmp_path, monkeypatch):
    # A family that already has a row under the per-trade construction
    # must NOT be treated as already-scored-under-ABI-1 -- the guard keys on
    # construction=="ABI-1" specifically, not merely on family_id presence.
    built = _build_scope(tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
                         accepted=["abi1_single"])
    trial_ledger.record_outcome(
        built["ledger"], "abi1_single", "legacy-run", "2026-08-01",
        "cut_dsr", dsr_value=0.5, sharpe_in_sample=0.01,
        construction="per_trade")

    result = abi1_cli.compute_batch_abi1(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False)

    assert "abi1_single" in _by_id(result)
    rows = _outcomes(built["ledger"], "abi1_single")
    assert len(rows) == 2   # the per-trade row plus the new ABI-1 row
    assert rows[-1]["construction"] == "ABI-1"


def test_one_familys_record_failure_does_not_abandon_its_siblings(tmp_path, monkeypatch):
    """Guards against a record_abi1_outcome failure for ONE family escaping
    compute_batch_abi1 and abandoning the whole batch, which would deny every
    not-yet-processed sibling a score. Two families run; abi1_single's own
    record_abi1_outcome call is made to raise, abi1_single_2's is not
    -- the batch must isolate the failure to abi1_single (a "record_error"
    skip, no ledger row) while abi1_single_2 still gets scored and written
    normally."""
    real_record = abi1_cli.abi1_driver.record_abi1_outcome

    def sabotaged(ledger_path, family_id, *args, **kwargs):
        if family_id == "abi1_single":
            raise RuntimeError("injected ledger-write failure")
        return real_record(ledger_path, family_id, *args, **kwargs)

    monkeypatch.setattr(abi1_cli.abi1_driver, "record_abi1_outcome", sabotaged)

    result, built = _run_compute(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_single_2": MODULE_SINGLE_2},
        accepted=["abi1_single", "abi1_single_2"])

    computed = _by_id(result)
    skips = _skips(result)
    assert "abi1_single" not in computed
    assert skips["abi1_single"]["reason_code"] == "record_error"
    assert "injected ledger-write failure" in skips["abi1_single"]["reason"]
    assert _outcomes(built["ledger"], "abi1_single") == []

    assert "abi1_single_2" in computed
    rows = _outcomes(built["ledger"], "abi1_single_2")
    assert len(rows) == 1
    assert rows[0]["construction"] == "ABI-1"


# --------------------------------------------------------------- cost gates

def test_missing_traded_quote_skips_the_family_as_missing_data(tmp_path, monkeypatch):
    result, built = _run_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], quotes_kwargs={"drop": ("AAA",)})

    assert "abi1_single" not in _by_id(result)
    skip = _skips(result)["abi1_single"]
    assert skip["reason_code"] == "missing_data"
    assert not os.path.exists(built["ledger"])


def test_missing_spy_quote_skips_every_family_as_missing_data(tmp_path, monkeypatch):
    result, built = _run_compute(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_pooled": MODULE_POOLED},
        accepted=["abi1_single", "abi1_pooled"], quotes_kwargs={"drop": ("SPY",)})

    assert result["computed"] == []
    skips = _skips(result)
    assert skips["abi1_single"]["reason_code"] == "missing_data"
    assert skips["abi1_pooled"]["reason_code"] == "missing_data"
    assert "SPY" in skips["abi1_single"]["reason"]


def test_parameter_drift_skips_the_candidate_and_writes_no_ledger_line(tmp_path, monkeypatch):
    modules_dir = _write_modules(tmp_path, {"abi1_single": MODULE_SINGLE})
    catalog_path = _write_catalog(tmp_path, ["abi1_single"])
    # Overwrite the catalog with a DIFFERENT period than the module declares.
    entries = json.loads(open(catalog_path, encoding="utf-8").read())
    entries[0]["parameters"] = {"period": 99, "hold": 7}
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)
    accepted_path = _write_accepted(tmp_path, ["abi1_single"])
    quotes_path = _write_quotes(tmp_path)
    ledger = str(tmp_path / "ledger.jsonl")
    loader, _ = _loader_with_counter()
    monkeypatch.setattr(batch5, "default_ohlc_loader", loader)
    scope = batch5.build_scope(
        accepted_path=accepted_path, quotes_path=quotes_path,
        catalog_path=catalog_path, ledger_path=ledger,
        modules_dir=modules_dir, run_dir=str(tmp_path / "research" / RUN_ID),
        ohlc_loader=loader)

    result = abi1_cli.compute_batch_abi1(
        scope, tested_date=TESTED_DATE, run_id=RUN_ID, ledger_path=ledger,
        dry_run=False)

    skip = _skips(result)["abi1_single"]
    assert skip["reason_code"] == "parameter_drift"
    assert not os.path.exists(ledger)


# ------------------------------------------------------------------- CLI main

def _run_cli(tmp_path, monkeypatch, *, modules, accepted, dry_run=False,
            families=None, quotes_kwargs=None):
    loader, _ = _loader_with_counter()
    monkeypatch.setattr(batch5, "default_ohlc_loader", loader)
    modules_dir = _write_modules(tmp_path, modules)
    catalog_path = _write_catalog(tmp_path, accepted)
    accepted_path = _write_accepted(tmp_path, accepted)
    quotes_path = _write_quotes(tmp_path, **(quotes_kwargs or {}))
    ledger = str(tmp_path / "ledger.jsonl")
    research_root = str(tmp_path / "research")
    argv = ["--accepted", accepted_path, "--quotes", quotes_path,
            "--run-id", RUN_ID, "--tested-date", TESTED_DATE,
            "--ledger", ledger, "--catalog", catalog_path,
            "--modules-dir", modules_dir, "--research-root", research_root]
    if dry_run:
        argv.append("--dry-run")
    if families:
        argv += ["--families", ",".join(families)]
    run_dir = os.path.join(research_root, RUN_ID)
    out = os.path.join(run_dir, "abi1-dry-run.json" if dry_run else "abi1-results.json")
    assert abi1_cli.main(argv) == 0
    with open(out, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, {"run_dir": run_dir, "ledger": ledger, "out": out}


def test_main_writes_results_json_and_run_report(tmp_path, monkeypatch):
    payload, paths = _run_cli(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"])

    assert payload["construction"] == "ABI-1"
    assert payload["dry_run"] is False
    by_id = {item["family_id"]: item for item in payload["results"]}
    assert "abi1_single" in by_id
    report_path = os.path.join(paths["run_dir"], "run-report.md")
    assert os.path.exists(report_path)
    text = open(report_path, encoding="utf-8").read()
    assert "abi1_single" in text
    assert os.path.exists(paths["ledger"])


def test_main_dry_run_writes_a_dry_run_report_and_no_ledger(tmp_path, monkeypatch):
    payload, paths = _run_cli(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], dry_run=True)

    assert payload["dry_run"] is True
    assert not os.path.exists(paths["ledger"])
    report_path = os.path.join(paths["run_dir"], "run-report-dry-run.md")
    assert os.path.exists(report_path)
    real_report = os.path.join(paths["run_dir"], "run-report.md")
    assert not os.path.exists(real_report)


def test_main_families_filter_restricts_the_run(tmp_path, monkeypatch):
    payload, _ = _run_cli(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_pooled": MODULE_POOLED},
        accepted=["abi1_single", "abi1_pooled"], families=["abi1_single"])

    by_id = {item["family_id"]: item for item in payload["results"]}
    assert set(by_id) == {"abi1_single"}


# ------------------------------------------- results JSON names no local paths

MODULE_BROKEN = 'raise ValueError("broken on purpose")\n'


@pytest.mark.parametrize("dry_run", [False, True])
def test_results_json_records_no_absolute_or_home_directory_paths(
        tmp_path, monkeypatch, dry_run):
    payload, paths = _run_cli(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_broken": MODULE_BROKEN},
        accepted=["abi1_single"], dry_run=dry_run)

    # The module_problems field, which carries each broken module's path, is
    # actually exercised.
    assert [p["module"] for p in payload["module_problems"]] == ["abi1_broken"]
    assert workstation_leaks(payload, tmp_path) == []
    # The bytes on disk too (a JSON-escaped backslash path would hide from the
    # decoded-string scan above).
    with open(paths["out"], "r", encoding="utf-8") as handle:
        assert not DRIVE_PATH.search(handle.read())

    # ...while every path field still identifies the artifact it points at.
    provenance = payload["provenance"]
    assert provenance["accepted_path"].endswith("accepted.json")
    assert provenance["quotes_path"].endswith("quotes.json")
    assert provenance["catalog_path"].endswith("catalog.json")
    assert provenance["ledger_path"].endswith("ledger.jsonl")
    assert provenance["modules_dir"].endswith("mods")
    assert payload["module_problems"][0]["path"].endswith("abi1_broken.py")
    written = [os.path.basename(p) for p in payload["outputs_written"]]
    assert ("run-report-dry-run.md" if dry_run else "run-report.md") in written


def test_no_module_skip_reason_names_no_absolute_path(tmp_path, monkeypatch):
    """abi1_single is accepted and cataloged but no module file is written --
    compute_batch_abi1's own "no_module" skip reason interpolates
    scope['modules_dir'] directly (via the shared batch5._skip) and must not
    leak it raw, same shape as run_batch5_mining.py's own version."""
    payload, paths = _run_cli(tmp_path, monkeypatch, modules={},
                              accepted=["abi1_single"])
    assert len(payload["skipped"]) == 1
    assert payload["skipped"][0]["reason_code"] == "no_module"
    assert workstation_leaks(payload, tmp_path) == []


# ------------------------------------------------------------------------
# Opt-in --history-end: end the scored history at a
# chosen NYSE session. Default OFF; only DROPS later sessions, in memory,
# after the per-run OHLC cache write; fail closed; recorded in the
# note/payload/report. If ANY loaded series lacks a bar on the cutoff the WHOLE
# batch is refused (no ledger row, no results file, no report; the OHLC cache is
# kept). A real run also runs the real-run guard (cutoff after the tested date,
# or a drop above --max-history-drop, refuses); a dry run reports the guard's
# answer as history_end_guard_would_refuse instead.
# ------------------------------------------------------------------------

from mining import history_cutoff  # noqa: E402

HE_POS = 327                # synthetic position: flat point of the i % 15 signal
HE = "2021-04-05"           # a real NYSE session, the date at HE_POS
HE_RAW_LAST = "2021-07-14"  # the synthetic frames' own last bar
HE_SHORT_POS = 380          # SHORT holds only positions 360..399 -> 21 bars here
HE_SHORT = "2021-06-17"
HE_DROP = N_SESSIONS - 1 - HE_POS   # sessions every full synthetic frame loses at HE
BASE_NOTE = "cash_convention=sweep_cash_daily; benchmark=SPY; tax_policy=realization"
# The synthetic world drops far more than the real-run default accepts (10), so a
# real-run test that is not ABOUT the guard accepts any drop this fixture can make.
HE_ACCEPT_ANY_DROP = N_SESSIONS
_UNSET = object()           # "do not pass max_history_drop at all" (use the default)
STANDARD_REFUSAL_KEYS = ["reason_code", "reason", "history_end_requested", "symbols"]


def _he_loader(frames):
    def loader(symbol):
        if symbol not in frames:
            raise RuntimeError(f"no synthetic data for {symbol}")
        # Same shape as data_pipeline.fetch_ohlc: a Date COLUMN over an int index.
        return frames[symbol].reset_index().rename(columns={"index": "Date"})
    return loader


def _he_compute(tmp_path, monkeypatch, *, modules, accepted, history_end,
                dry_run=True, frames=None, tested_date=TESTED_DATE,
                max_history_drop=HE_ACCEPT_ANY_DROP, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    loader = _he_loader(frames if frames is not None else _FRAMES)
    built = _build_scope(tmp_path, monkeypatch, modules=modules,
                         accepted=accepted, loader=loader, **kwargs)
    drop_kwargs = ({} if max_history_drop is _UNSET
                   else {"max_history_drop": max_history_drop})
    result = abi1_cli.compute_batch_abi1(
        built["scope"], tested_date=tested_date, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=dry_run, history_end=history_end,
        **drop_kwargs)
    return result, built


def _he_cli(tmp_path, monkeypatch, capsys, *, modules, accepted, history_end,
            dry_run=True, frames=None, tested_date=TESTED_DATE,
            max_history_drop=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(batch5, "default_ohlc_loader",
                        _he_loader(frames if frames is not None else _FRAMES))
    modules_dir = _write_modules(tmp_path, modules)
    catalog_path = _write_catalog(tmp_path, accepted)
    accepted_path = _write_accepted(tmp_path, accepted)
    quotes_path = _write_quotes(tmp_path)
    ledger = str(tmp_path / "ledger.jsonl")
    research_root = str(tmp_path / "research")
    argv = ["--accepted", accepted_path, "--quotes", quotes_path,
            "--run-id", RUN_ID, "--tested-date", tested_date,
            "--ledger", ledger, "--catalog", catalog_path,
            "--modules-dir", modules_dir, "--research-root", research_root]
    if dry_run:
        argv.append("--dry-run")
    if history_end is not None:
        argv += ["--history-end", history_end]
    if max_history_drop is not None:
        argv += ["--max-history-drop", str(max_history_drop)]
    capsys.readouterr()
    code = abi1_cli.main(argv)
    stdout = capsys.readouterr().out
    run_dir = os.path.join(research_root, RUN_ID)
    out = os.path.join(run_dir, "abi1-dry-run.json" if dry_run else "abi1-results.json")
    payload = None
    if os.path.exists(out):
        with open(out, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    return code, stdout, payload, {"run_dir": run_dir, "ledger": ledger, "out": out}


def _files_besides_the_ohlc_cache(run_dir):
    """Every file under the run dir except the per-run OHLC cache (a load
    artifact, kept on purpose). A refused history end must leave this empty:
    no results file, no run report (dry or real)."""
    found = []
    for root, _dirs, names in os.walk(run_dir):
        for name in names:
            rel = os.path.relpath(os.path.join(root, name), run_dir).replace(os.sep, "/")
            if not rel.startswith("artifacts/ohlc/"):
                found.append(rel)
    return sorted(found)


def _assert_standard_refusal(refusal, *, code, requested, symbols=None):
    """The one refusal shape every driver reports: the four standard keys, in this
    order, then any driver-specific extras."""
    assert list(refusal)[:4] == STANDARD_REFUSAL_KEYS
    assert refusal["reason_code"] == code
    assert refusal["history_end_requested"] == requested
    assert isinstance(refusal["reason"], str) and refusal["reason"]
    assert isinstance(refusal["symbols"], list)
    if symbols is not None:
        assert refusal["symbols"] == symbols


def _capture_curves(monkeypatch):
    captured = []
    real = abi1_cli.abi1_driver.evaluate_family

    def spy(curve, **kwargs):
        captured.append(curve)
        return real(curve, **kwargs)

    monkeypatch.setattr(abi1_cli.abi1_driver, "evaluate_family", spy)
    return captured


def _poison_after(frames, cutoff_date):
    """Copies of ``frames`` whose bars dated AFTER ``cutoff_date`` are replaced by
    wildly different values (a per-row factor between 0.2 and 5, OHLC kept
    self-consistent so the loader accepts them)."""
    cutoff = pd.Timestamp(cutoff_date)
    out = {}
    for symbol, frame in frames.items():
        copy = frame.copy()
        mask = np.asarray(copy.index > cutoff)
        rng = np.random.RandomState(sum(map(ord, symbol)))
        factor = rng.uniform(0.2, 5.0, int(mask.sum()))
        columns = ["Open", "High", "Low", "Close"]
        copy.loc[mask, columns] = copy.loc[mask, columns].to_numpy() * factor[:, None]
        out[symbol] = copy
    return out


def test_history_end_fixture_positions_are_what_the_tests_assume():
    index = _index(N_SESSIONS)
    assert index[HE_POS] == pd.Timestamp(HE)
    assert index[HE_SHORT_POS] == pd.Timestamp(HE_SHORT)
    assert index[-1] == pd.Timestamp(HE_RAW_LAST)
    assert HE_POS % 15 == 12          # flat: no trade is open at the cutoff bar
    assert HE_DROP == 72 and HE_DROP > history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS
    assert N_SESSIONS - 1 - HE_SHORT_POS > history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS
    for value in (HE, HE_SHORT):
        assert history_cutoff.parse_history_end(value) == pd.Timestamp(value)


# ------------------------------------------------------------------ flag off

def test_history_end_off_records_nothing_extra_anywhere(tmp_path, monkeypatch, capsys):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=None, dry_run=False)

    rows = _outcomes(built["ledger"], "abi1_single")
    assert rows[0]["note"] == BASE_NOTE
    assert "history_end_provenance" not in result
    assert "history_end_refusal" not in result
    assert "history_end_guard_would_refuse" not in result
    assert "history_end_requested" not in result
    item = _by_id(result)["abi1_single"]
    assert "history_end" not in item
    assert item["account_end"] == HE_RAW_LAST
    assert item["n_sessions"] == N_SESSIONS
    for symbol, cover in result["input_coverage"].items():
        assert not {"raw_last", "kept_last", "dropped_sessions"} & set(cover), symbol
        assert cover["last"] == HE_RAW_LAST

    code, stdout, payload, paths = _he_cli(
        tmp_path / "cli", monkeypatch, capsys, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=None)
    assert code == 0
    assert "history_end" not in payload
    assert "history_end" not in payload["provenance"]
    assert "history_end" not in payload["results"][0]
    assert "history_end_requested" not in payload
    assert "history_end_guard_would_refuse" not in payload
    printed = json.loads(stdout)
    assert not {"history_end", "history_end_requested",
                "history_end_guard_would_refuse"} & set(printed)
    report = open(os.path.join(paths["run_dir"], "run-report-dry-run.md"),
                  encoding="utf-8").read()
    assert history_cutoff.HISTORY_END_MARKER not in report
    assert "History end" not in report
    # With the flag off the report keeps its default full-history wording.
    assert "Every family here was scored on its FULL available price history" in report


# ------------------------------------- a hole that refuses without the flag

def _frames_with_a_rotation_leg_hole():
    """RSK is missing ONE interior session (the one right after HE); DEF is
    complete: the two traded legs no longer share a calendar."""
    frames = dict(_FRAMES)
    frames["RSK"] = _FRAMES["RSK"].drop(_FRAMES["RSK"].index[HE_POS + 1])
    return frames


def test_an_interior_vendor_hole_refuses_without_a_history_end(tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_rotation": MODULE_ROTATION},
        accepted=["abi1_rotation"], history_end=None, dry_run=True,
        frames=_frames_with_a_rotation_leg_hole())

    assert result["computed"] == []
    skip = _skips(result)["abi1_rotation"]
    assert skip["reason_code"] == "compute_error"
    assert "identical calendar" in skip["reason"]


def test_a_history_end_before_the_hole_makes_the_family_scoreable_in_a_dry_run(
        tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_rotation": MODULE_ROTATION},
        accepted=["abi1_rotation"], history_end=HE, dry_run=True,
        frames=_frames_with_a_rotation_leg_hole())

    assert _skips(result) == {}
    item = _by_id(result)["abi1_rotation"]
    assert item["account_end"] == HE
    assert item["stage"] in evidence.VALID_STAGES
    assert not os.path.exists(built["ledger"])
    assert result["input_coverage"]["DEF"]["dropped_sessions"] == N_SESSIONS - 1 - HE_POS
    assert result["input_coverage"]["RSK"]["dropped_sessions"] == N_SESSIONS - 2 - HE_POS


# ------------------------------------------------------- prefix invariance

def test_a_trimmed_run_is_the_exact_prefix_of_an_untrimmed_run(tmp_path, monkeypatch):
    curves = _capture_curves(monkeypatch)
    for name, history_end in (("full", None), ("trimmed", HE)):
        (tmp_path / name).mkdir()
        _he_compute(tmp_path / name, monkeypatch,
                    modules={"abi1_single": MODULE_SINGLE},
                    accepted=["abi1_single"], history_end=history_end)
    full, trimmed = curves
    cutoff = pd.Timestamp(HE)

    assert full["strategy_daily"].index[-1] > cutoff
    assert trimmed["strategy_daily"].index[-1] == cutoff
    pd.testing.assert_series_equal(full["strategy_daily"].loc[:cutoff],
                                   trimmed["strategy_daily"])
    pd.testing.assert_series_equal(full["in_market"].loc[:cutoff],
                                   trimmed["in_market"])


# ---------------------------------------------- recorded everywhere it records

def test_the_ledger_note_names_the_history_end_and_the_raw_last_bars(
        tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=HE, dry_run=False)

    rows = _outcomes(built["ledger"], "abi1_single")
    assert len(rows) == 1 and rows[0]["stage"] in evidence.SCORED_STAGES
    note = rows[0]["note"]
    assert note.startswith(BASE_NOTE + "; ")
    assert history_cutoff.HISTORY_END_MARKER in note
    assert f"ended at {HE}" in note
    assert f"AAA {HE_RAW_LAST}" in note and f"SPY {HE_RAW_LAST}" in note
    assert "BBB" not in note, "the note names only the series this family scored"
    assert "cannot be re-scored on longer history" in note
    item = _by_id(result)["abi1_single"]
    assert item["account_end"] == HE and item["n_sessions"] == HE_POS + 1
    assert item["history_end"]["history_end"] == HE
    assert history_cutoff.HISTORY_END_MARKER in item["history_end"]["clause"]


def test_the_result_payload_and_report_carry_the_history_end(
        tmp_path, monkeypatch, capsys):
    code, stdout, payload, paths = _he_cli(
        tmp_path, monkeypatch, capsys, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=HE)

    assert code == 0
    block = payload["history_end"]
    assert block["history_end"] == HE and block["opt_in"] is True
    assert block["decision"] == history_cutoff.DECISION
    assert history_cutoff.HISTORY_END_MARKER in block["clause"]
    assert block["by_symbol"]["SPY"]["raw_last"] == HE_RAW_LAST
    assert block["by_symbol"]["AAA"]["kept_last"] == HE
    assert block["dropped_dates"][0] > HE and block["dropped_dates"][-1] == HE_RAW_LAST
    assert payload["provenance"]["history_end"] == block
    assert payload["results"][0]["history_end"]["history_end"] == HE
    assert payload["input_coverage"]["AAA"]["raw_last"] == HE_RAW_LAST
    assert payload["input_coverage"]["AAA"]["kept_last"] == HE
    # The raw string the operator typed sits beside the provenance under the
    # same key names every history-end driver uses.
    assert payload["history_end_requested"] == HE
    # This rehearsal used the default drop limit, which the synthetic world's
    # 72-session drop exceeds: the dry run says so and does not refuse.
    would_refuse = payload["history_end_guard_would_refuse"]
    assert isinstance(would_refuse, str) and "--max-history-drop" in would_refuse
    printed = json.loads(stdout)
    assert printed["history_end"] == HE
    assert printed["history_end_guard_would_refuse"] == would_refuse
    assert printed["dry_run"] is True and printed["tested"] == 1
    report = open(os.path.join(paths["run_dir"], "run-report-dry-run.md"),
                  encoding="utf-8").read()
    assert history_cutoff.HISTORY_END_MARKER in report
    assert HE in report
    assert would_refuse in report
    assert workstation_leaks(payload, tmp_path) == []


def test_the_report_puts_the_history_end_section_before_the_candidate_table(
        tmp_path, monkeypatch, capsys):
    code, stdout, payload, paths = _he_cli(
        tmp_path, monkeypatch, capsys, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=HE, max_history_drop=HE_DROP)

    assert code == 0
    report = open(os.path.join(paths["run_dir"], "run-report-dry-run.md"),
                  encoding="utf-8").read()
    history_at = report.index("## History end (opt-in)")
    heading_at = report.index("## ABI-1 candidate table")
    table_at = report.index("| family |")
    assert history_at < heading_at < table_at, "the section sits before the heading"
    assert history_cutoff.HISTORY_END_MARKER in report[history_at:heading_at]
    between = report[heading_at:table_at]        # heading .. first table row
    assert history_cutoff.HISTORY_END_MARKER not in between
    assert "## History end" not in between
    # No "FULL available price history" claim under the flag: the paragraph says
    # the history was ended at the cutoff instead.
    assert "FULL available" not in report
    assert f"ended at {HE}" in between
    # A drop the operator accepted explicitly is not a would-refuse.
    assert payload["history_end_guard_would_refuse"] is None
    assert "would be REFUSED" not in report
    assert "a real run of this setting would be accepted" in report


# ----------------------------------------------------------------- refusals

@pytest.mark.parametrize("value,code", [
    ("2021-04-03", "history_end_not_a_session"),    # a Saturday
    ("2021-04-02", "history_end_not_a_session"),    # Good Friday, exchange closed
    ("2021/04/05", "history_end_invalid"),
    ("2021-4-5", "history_end_invalid"),
    ("", "history_end_invalid"),
    ("not-a-date", "history_end_invalid"),
    ("2021-02-30", "history_end_invalid"),
])
def test_a_bad_history_end_refuses_the_whole_batch_before_anything_is_loaded(
        tmp_path, monkeypatch, capsys, value, code):
    """A malformed or non-session date is refused at the door of the compute
    function, before any series is loaded: nothing at all is written (no ledger,
    no run directory, so no OHLC cache either)."""
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=value, dry_run=False)

    _assert_standard_refusal(result["history_end_refusal"], code=code,
                             requested=value, symbols=[])
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])
    assert not os.path.exists(built["run_dir"])

    rc, stdout, payload, paths = _he_cli(
        tmp_path / "cli", monkeypatch, capsys, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=value, dry_run=False)
    assert rc == 2
    printed = json.loads(stdout)
    assert printed["refused"] is True and printed["reason_code"] == code
    assert printed["history_end_requested"] == value and printed["symbols"] == []
    assert "history_end" not in printed, "a refusal names the request history_end_requested"
    assert payload is None
    assert not os.path.exists(paths["run_dir"])
    assert not os.path.exists(paths["ledger"])


def test_the_run_level_refusal_payload_leads_with_the_standard_keys(
        tmp_path, monkeypatch):
    built = _build_scope(tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
                         accepted=["abi1_single"])
    payload = abi1_cli.run_batch_abi1(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False, history_end="2021-04-03")

    assert list(payload)[:4] == STANDARD_REFUSAL_KEYS
    assert payload["reason_code"] == "history_end_not_a_session"
    assert payload["history_end_requested"] == "2021-04-03" and payload["symbols"] == []
    assert payload["refused"] is True and payload["dry_run"] is False
    assert payload["run_id"] == RUN_ID and payload["tested_date"] == TESTED_DATE
    assert payload["results"] == [] and payload["skipped"] == []
    assert payload["outputs_written"] == []
    json.dumps(payload, allow_nan=False)


def _refused_cli_run(tmp_path, monkeypatch, capsys, *, modules, accepted, code,
                     frames=None, **cli_kwargs):
    """Runs the CLI for real (not a dry run) and checks everything a refused
    history end must leave behind: exit 2, the standard refusal on stdout, no
    ledger, no results file, no run report; only the OHLC cache the load step
    already wrote."""
    rc, stdout, payload, paths = _he_cli(
        tmp_path, monkeypatch, capsys, modules=modules, accepted=accepted,
        frames=frames, dry_run=False, **cli_kwargs)
    assert rc == 2
    printed = json.loads(stdout)
    assert printed["refused"] is True and printed["reason_code"] == code
    assert list(printed)[:5] == ["refused"] + STANDARD_REFUSAL_KEYS
    assert payload is None
    assert not os.path.exists(paths["ledger"])
    assert _files_besides_the_ohlc_cache(paths["run_dir"]) == []
    return printed, paths


def test_a_cutoff_no_series_reaches_refuses_the_whole_batch(
        tmp_path, monkeypatch, capsys):
    # 2021-07-15 is a real session but one day past every synthetic bar.
    modules = {"abi1_single": MODULE_SINGLE, "abi1_pooled": MODULE_POOLED}
    accepted = ["abi1_single", "abi1_pooled"]
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=modules, accepted=accepted,
        history_end="2021-07-15", dry_run=False)

    _assert_standard_refusal(
        result["history_end_refusal"], code="history_end_not_covered",
        requested="2021-07-15", symbols=["AAA", "BBB", "CCC", "SPY"])
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])

    printed, _ = _refused_cli_run(
        tmp_path / "cli", monkeypatch, capsys, modules=modules, accepted=accepted,
        code="history_end_not_covered", history_end="2021-07-15",
        max_history_drop=HE_ACCEPT_ANY_DROP)
    assert printed["symbols"] == ["AAA", "BBB", "CCC", "SPY"]


def _frames_missing_the_cutoff_bar(symbol):
    frames = dict(_FRAMES)
    frames[symbol] = _FRAMES[symbol].drop(_FRAMES[symbol].index[HE_POS])
    return frames


def test_one_series_lacking_the_cutoff_bar_refuses_the_whole_batch_not_just_its_plan(
        tmp_path, monkeypatch, capsys):
    """Two families, one of which (abi1_single_2, DEF) lacks the cutoff bar: the
    sibling (abi1_single) must NOT score either. Scoring it anyway would write a
    permanent ledger row for a batch that was not fully covered."""
    frames = _frames_missing_the_cutoff_bar("DEF")
    modules = {"abi1_single": MODULE_SINGLE, "abi1_single_2": MODULE_SINGLE_2}
    accepted = ["abi1_single", "abi1_single_2"]
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=modules, accepted=accepted,
        history_end=HE, dry_run=False, frames=frames)

    refusal = result["history_end_refusal"]
    _assert_standard_refusal(refusal, code="history_end_not_covered",
                             requested=HE, symbols=["DEF"])
    assert "DEF" in refusal["reason"]
    assert result["computed"] == [] and result["skipped"] == []
    assert "history_end_provenance" not in result
    assert not os.path.exists(built["ledger"])
    # The per-run OHLC cache written by the load step is kept, and the reason
    # says so and how to retry.
    cache_dir = os.path.join(built["run_dir"], "artifacts", "ohlc")
    assert sorted(os.listdir(cache_dir)) == ["AAA.csv", "DEF.csv", "SPY.csv"]
    assert "OHLC cache" in refusal["reason"] and "fresh --run-id" in refusal["reason"]

    printed, paths = _refused_cli_run(
        tmp_path / "cli", monkeypatch, capsys, modules=modules, accepted=accepted,
        code="history_end_not_covered", frames=frames, history_end=HE,
        max_history_drop=HE_ACCEPT_ANY_DROP)
    assert printed["symbols"] == ["DEF"]
    assert "OHLC cache" in printed["reason"]


def test_a_pool_member_lacking_the_cutoff_bar_refuses_the_whole_batch_not_a_shrunken_pool(
        tmp_path, monkeypatch, capsys):
    """The pooled family is NOT scored on the four covered members after silently
    dropping DEF (that would be survivor bias in a permanent one-shot score)."""
    frames = _frames_missing_the_cutoff_bar("DEF")
    modules = {"abi1_pooled_5": MODULE_POOLED_5, "abi1_single": MODULE_SINGLE}
    accepted = ["abi1_pooled_5", "abi1_single"]
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=modules, accepted=accepted,
        history_end=HE, dry_run=False, frames=frames)

    _assert_standard_refusal(result["history_end_refusal"],
                             code="history_end_not_covered", requested=HE,
                             symbols=["DEF"])
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])

    _refused_cli_run(
        tmp_path / "cli", monkeypatch, capsys, modules=modules, accepted=accepted,
        code="history_end_not_covered", frames=frames, history_end=HE,
        max_history_drop=HE_ACCEPT_ANY_DROP)


def test_a_series_that_fails_to_load_for_another_reason_only_skips_its_plans(
        tmp_path, monkeypatch):
    """The whole-batch refusal is about a series that LOADED but lacks the
    cutoff bar. A series that cannot be loaded at all (here: the vendor has no
    data for DEF) still only skips the plans that use it, exactly as without a
    history end; the sibling scores."""
    frames = {symbol: frame for symbol, frame in _FRAMES.items() if symbol != "DEF"}
    result, built = _he_compute(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_single_2": MODULE_SINGLE_2},
        accepted=["abi1_single", "abi1_single_2"], history_end=HE,
        dry_run=False, frames=frames)

    assert "history_end_refusal" not in result
    assert set(_by_id(result)) == {"abi1_single"}
    skip = _skips(result)["abi1_single_2"]
    assert skip["reason_code"] == "missing_data"
    assert "no synthetic data for DEF" in skip["reason"]
    assert _by_id(result)["abi1_single"]["account_end"] == HE
    assert len(_outcomes(built["ledger"], "abi1_single")) == 1
    assert _outcomes(built["ledger"], "abi1_single_2") == []
    assert set(result["history_end_provenance"]["by_symbol"]) == {"AAA", "SPY"}


MODULE_POOLED_5 = _HEADER + """
FAMILY_ID = "abi1_pooled_5"
SHAPE = "pooled_single_asset"
TRADED = ["AAA", "BBB", "CCC", "RSK", "DEF"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 15

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""
CATALOG_PARAMETERS["abi1_pooled_5"] = {"period": 15, "hold": 7}
TRADED_BY_ID["abi1_pooled_5"] = ["AAA", "BBB", "CCC", "RSK", "DEF"]


def test_a_fully_covered_pool_is_scored_on_every_member_and_its_note_ends_with_the_clause(
        tmp_path, monkeypatch):
    """With every member covered the pooled family scores on ALL five members,
    excludes none, and its ledger note ends with the family's history-end clause;
    its history-end record names every member plus the benchmark."""
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_pooled_5": MODULE_POOLED_5},
        accepted=["abi1_pooled_5"], history_end=HE, dry_run=False)

    assert _skips(result) == {}
    item = _by_id(result)["abi1_pooled_5"]
    assert set(item["tickers"]) == {"AAA", "BBB", "CCC", "RSK", "DEF"}
    assert item["pool_excluded"] == {}
    note = _outcomes(built["ledger"], "abi1_pooled_5")[0]["note"]
    assert note.endswith(item["history_end"]["clause"])
    assert set(item["history_end"]["by_symbol"]) == {"AAA", "BBB", "CCC", "RSK",
                                                    "DEF", "SPY"}


def test_a_benchmark_lacking_the_cutoff_bar_refuses_the_whole_batch(
        tmp_path, monkeypatch, capsys):
    frames = _frames_missing_the_cutoff_bar("SPY")
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=HE, dry_run=False, frames=frames)

    _assert_standard_refusal(result["history_end_refusal"],
                             code="history_end_not_covered", requested=HE,
                             symbols=["SPY"])
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])

    _refused_cli_run(
        tmp_path / "cli", monkeypatch, capsys, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], code="history_end_not_covered", frames=frames,
        history_end=HE, max_history_drop=HE_ACCEPT_ANY_DROP)


# The recorder carries the history-end clause on an UNSCORED row too (its
# `unscored_note` keyword), so an unscored verdict under a history end is
# processed like any other verdict: a dry run reports it, and a real run writes
# its row with the clause as the note and spends the family's ABI-1 slot.

def _record_calls(monkeypatch):
    """Spy that passes through to the REAL recorder and remembers, per call, the
    family id and the keyword arguments the driver handed it."""
    calls = []
    real = abi1_cli.abi1_driver.record_abi1_outcome

    def spy(*args, **kwargs):
        calls.append({"family_id": args[1], **kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(abi1_cli.abi1_driver, "record_abi1_outcome", spy)
    return calls


def test_an_unscored_verdict_under_a_history_end_is_processed_in_a_dry_run(
        tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_short": MODULE_SHORT},
        accepted=["abi1_short"], history_end=HE_SHORT, dry_run=True)

    assert result["skipped"] == []
    item = _by_id(result)["abi1_short"]
    assert item["verdict"] == "not_scoreable"
    assert item["stage"] == "insufficient_evidence"
    assert item["account_end"] == HE_SHORT
    assert history_cutoff.HISTORY_END_MARKER in item["history_end"]["clause"]
    assert not os.path.exists(built["ledger"])


def test_an_unscored_verdict_under_a_history_end_writes_a_row_carrying_the_clause(
        tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_short": MODULE_SHORT},
        accepted=["abi1_short"], history_end=HE_SHORT, dry_run=False)

    assert result["skipped"] == []
    item = _by_id(result)["abi1_short"]
    assert item["verdict"] == "not_scoreable"
    assert item["stage"] == "insufficient_evidence"
    rows = _outcomes(built["ledger"], "abi1_short")
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "insufficient_evidence"
    assert row["evidence"]["reason_code"] == "abi1_not_scoreable"
    note = row["note"]
    assert note == item["history_end"]["clause"], "the clause, verbatim, as the note"
    assert history_cutoff.HISTORY_END_MARKER in note
    assert f"ended at {HE_SHORT}" in note
    assert f"SHORT {HE_RAW_LAST}" in note and f"SPY {HE_RAW_LAST}" in note
    assert "cannot be re-scored on longer history" in note
    # The slot really is spent: a later run on this ledger skips the family.
    again = abi1_cli.compute_batch_abi1(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=False, history_end=HE_SHORT,
        max_history_drop=HE_ACCEPT_ANY_DROP)
    assert _skips(again)["abi1_short"]["reason_code"] == "already_scored_abi1"
    assert len(_outcomes(built["ledger"], "abi1_short")) == 1


def test_history_end_unset_hands_the_recorder_no_note_arguments(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    result, built = _he_compute(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_short": MODULE_SHORT},
        accepted=["abi1_single", "abi1_short"], history_end=None, dry_run=False)

    assert _skips(result) == {}
    assert {call["family_id"] for call in calls} == {"abi1_single", "abi1_short"}
    for call in calls:
        assert "unscored_note" not in call, call["family_id"]
        assert "extra_note" not in call, call["family_id"]
    short_rows = _outcomes(built["ledger"], "abi1_short")
    assert len(short_rows) == 1 and short_rows[0]["stage"] == "insufficient_evidence"
    assert short_rows[0].get("note") is None


def test_a_history_end_hands_the_recorder_one_clause_for_scored_and_unscored_rows(
        tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    result, built = _he_compute(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_short": MODULE_SHORT},
        accepted=["abi1_single", "abi1_short"], history_end=HE_SHORT, dry_run=False)

    assert _skips(result) == {}
    by_id = _by_id(result)
    assert by_id["abi1_single"]["stage"] in evidence.SCORED_STAGES
    assert by_id["abi1_short"]["stage"] == "insufficient_evidence"
    assert {call["family_id"] for call in calls} == {"abi1_single", "abi1_short"}
    for call in calls:
        clause = by_id[call["family_id"]]["history_end"]["clause"]
        assert call["extra_note"] == clause
        assert call["unscored_note"] == clause
    # Scored row: the clause follows the base template, exactly once. Unscored
    # row: the clause alone (there is no base template on that row).
    scored_note = _outcomes(built["ledger"], "abi1_single")[0]["note"]
    assert scored_note.startswith(BASE_NOTE + "; ")
    assert scored_note.count(history_cutoff.HISTORY_END_MARKER) == 1
    unscored_note = _outcomes(built["ledger"], "abi1_short")[0]["note"]
    assert unscored_note == by_id["abi1_short"]["history_end"]["clause"]


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_holdout_not_evaluable_verdict_is_processed_and_carries_the_clause(
        tmp_path, monkeypatch, dry_run):
    """holdout_not_evaluable needs a real in-sample DSR above the pass bar plus
    an unevaluable holdout, which no cheap synthetic world produces. So this
    forces that verdict on an otherwise real evaluation and stubs the recorder:
    it proves only the DRIVER's handling of that second unscored verdict."""
    real_evaluate = abi1_cli.abi1_driver.evaluate_family

    def forced(curve, **kwargs):
        return {**real_evaluate(curve, **kwargs), "verdict": "holdout_not_evaluable"}

    seen = []

    def stub_record(*args, **kwargs):
        seen.append({"family_id": args[1], **kwargs})
        return "insufficient_evidence"

    monkeypatch.setattr(abi1_cli.abi1_driver, "evaluate_family", forced)
    monkeypatch.setattr(abi1_cli.abi1_driver, "record_abi1_outcome", stub_record)
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=HE, dry_run=dry_run)

    assert result["skipped"] == []
    item = _by_id(result)["abi1_single"]
    assert item["verdict"] == "holdout_not_evaluable"
    assert item["stage"] == "insufficient_evidence"
    if dry_run:
        assert seen == []
    else:
        [call] = seen
        assert call["unscored_note"] == item["history_end"]["clause"]
        assert history_cutoff.HISTORY_END_MARKER in call["unscored_note"]
    assert not os.path.exists(built["ledger"])


# ----------------------------------------------- the cache is never trimmed

def test_the_per_run_ohlc_cache_keeps_the_full_history_after_a_trimmed_run(
        tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=HE, dry_run=True)
    cache_dir = os.path.join(built["run_dir"], "artifacts", "ohlc")

    assert result["input_coverage"]["AAA"]["origin"] == "fetched"
    for symbol in ("AAA", "SPY"):
        cached = pd.read_csv(os.path.join(cache_dir, f"{symbol}.csv"))
        assert len(cached) == N_SESSIONS
        assert cached["Date"].iloc[-1][:10] == HE_RAW_LAST

    # A later untrimmed run in the same run dir reads that cache: full history.
    later = abi1_cli.compute_batch_abi1(
        built["scope"], tested_date=TESTED_DATE, run_id=RUN_ID,
        ledger_path=built["ledger"], dry_run=True)
    assert later["input_coverage"]["AAA"]["origin"] == "cache"
    item = _by_id(later)["abi1_single"]
    assert item["account_end"] == HE_RAW_LAST and item["n_sessions"] == N_SESSIONS


# ----------------------------------------------------------------- no leak

def test_bars_after_the_cutoff_cannot_change_a_trimmed_score(tmp_path, monkeypatch):
    """Post-cutoff bars replaced by wildly different values must not move any
    scored number, in any shape (single, rotation, pooled)."""
    modules = {"abi1_single": MODULE_SINGLE, "abi1_rotation": MODULE_ROTATION,
               "abi1_pooled": MODULE_POOLED}
    accepted = ["abi1_single", "abi1_rotation", "abi1_pooled"]
    poisoned = _poison_after(_FRAMES, HE)
    fields = ("dsr_value", "sharpe_daily_excess", "n_sessions", "account_start",
              "account_end", "time_in_market", "stage")

    runs = {}
    for name, frames, history_end in (
            ("clean_trimmed", _FRAMES, HE), ("poison_trimmed", poisoned, HE),
            ("clean_full", _FRAMES, None), ("poison_full", poisoned, None)):
        (tmp_path / name).mkdir()
        result, _ = _he_compute(
            tmp_path / name, monkeypatch, modules=modules, accepted=accepted,
            history_end=history_end, frames=frames)
        assert _skips(result) == {}
        runs[name] = {fid: {f: item[f] for f in fields}
                      for fid, item in _by_id(result).items()}

    assert runs["clean_trimmed"] == runs["poison_trimmed"]
    for fid in accepted:
        assert runs["clean_trimmed"][fid]["account_end"] == HE
    # The poison is real: without the trim it changes the score.
    assert runs["clean_full"] != runs["poison_full"]


# ------------- what a scored history end pins in the ledger (new vs already-pinned)

def _long_frames_and_cutoff():
    """1200 synthetic sessions (long enough for the holdout split to apply) and a
    real NYSE session about 1100 bars in to use as the cutoff."""
    n = 1200
    frames = {"SPY": _frame(n, 11, "2016-01-04"), "AAA": _frame(n, 12, "2016-01-04")}
    index = _index(n, "2016-01-04")
    from mining.batch5.treasury_month_end_index_demand_overlay import exchange_sessions
    sessions = set(exchange_sessions(index[0], index[-1]))
    for position in range(1100, 1000, -1):     # walk back off any exchange holiday
        if index[position] in sessions:
            return frames, index[position]
    raise AssertionError("no NYSE session found near position 1100")


def test_a_scored_history_end_pins_a_new_familys_holdout_end_to_the_cutoff(
        tmp_path, monkeypatch):
    frames, cutoff = _long_frames_and_cutoff()

    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=f"{cutoff:%Y-%m-%d}", dry_run=False,
        frames=frames)

    item = _by_id(result)["abi1_single"]
    assert item["window"]["holdout_split_applied"] is True
    assert item["stage"] in evidence.SCORED_STAGES
    pinned = trial_ledger.get_pinned_holdout_window(built["ledger"], "abi1_single")
    assert pinned["holdout_end"] == f"{cutoff:%Y-%m-%d}"
    assert item["account_end"] == f"{cutoff:%Y-%m-%d}"


def test_a_history_end_does_not_move_the_pin_of_a_family_that_already_has_one(
        tmp_path, monkeypatch):
    """The pin is not unconditional: record_first_test is a no-op for a family
    that already has a first_test row, so the earlier pin stands. The scored
    account still ends at the cutoff."""
    frames, cutoff = _long_frames_and_cutoff()
    ledger = str(tmp_path / "ledger.jsonl")
    assert trial_ledger.record_first_test(
        ledger, "abi1_single", "2026-01-05", "2019-01-01", "2020-06-30",
        [0.0, 0.0], "an-earlier-run") is True

    result, built = _he_compute(
        tmp_path, monkeypatch, modules={"abi1_single": MODULE_SINGLE},
        accepted=["abi1_single"], history_end=f"{cutoff:%Y-%m-%d}", dry_run=False,
        frames=frames, ledger=ledger)

    assert _skips(result) == {}
    item = _by_id(result)["abi1_single"]
    assert item["account_end"] == f"{cutoff:%Y-%m-%d}"
    pinned = trial_ledger.get_pinned_holdout_window(ledger, "abi1_single")
    assert pinned["holdout_end"] == "2020-06-30"


# ------------------------- family modules are handed the trimmed frames

# Each probe records the last bar date of every frame the driver hands its
# module, so a module that builds anything from "the full frames" (a target
# built at make_target_fn time, a signal closure over `frames`) provably
# never sees a bar dated after the cutoff -- including a signal-only series.
MODULE_PROBE_ROTATION = _HEADER + """
FAMILY_ID = "abi1_probe_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["RSK", "DEF"]
SIGNAL = ["SIG"]
SNAPSHOTS = {}
BENCHMARK = "RSK"
PARAMETERS = {"block": 40}
EXPECTED_MIN_TRADES = 5
SEEN = {}

def make_target_fn(frames):
    SEEN.update({symbol: frames[symbol].index[-1] for symbol in frames})
    def target(truncated):
        n = len(truncated["RSK"])
        return "RSK" if (n // 40) % 2 == 0 else "DEF"
    return target
"""

MODULE_PROBE_SINGLE = _HEADER + """
FAMILY_ID = "abi1_probe_single"
SHAPE = "single_long"
TRADED = ["AAA"]
SIGNAL = ["SIG"]
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 5
SEEN = {}

def make_signal_fn(ticker, index, frames):
    SEEN.update({symbol: frames[symbol].index[-1] for symbol in frames})
    SEEN["index"] = index[-1]
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""

CATALOG_PARAMETERS.update({"abi1_probe_rotation": {"block": 40},
                           "abi1_probe_single": {"period": 15, "hold": 7}})
TRADED_BY_ID.update({"abi1_probe_rotation": ["RSK", "DEF"],
                     "abi1_probe_single": ["AAA"]})


def _probe_frames():
    frames = dict(_FRAMES)
    frames["SIG"] = _frame(N_SESSIONS, seed=77)
    return frames


@pytest.mark.parametrize("history_end,expected_last", [
    (HE, pd.Timestamp(HE)), (None, pd.Timestamp(HE_RAW_LAST))])
def test_family_modules_are_handed_frames_that_end_at_the_cutoff(
        tmp_path, monkeypatch, history_end, expected_last):
    modules = {"abi1_probe_rotation": MODULE_PROBE_ROTATION,
               "abi1_probe_single": MODULE_PROBE_SINGLE}
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=modules,
        accepted=["abi1_probe_rotation", "abi1_probe_single"],
        history_end=history_end, frames=_probe_frames())

    assert _skips(result) == {}
    mods = built["scope"]["modules"]
    rotation_seen = mods["abi1_probe_rotation"]["module"].SEEN
    single_seen = mods["abi1_probe_single"]["module"].SEEN
    assert set(rotation_seen) == {"RSK", "DEF", "SIG"}
    assert set(rotation_seen.values()) == {expected_last}
    assert set(single_seen) == {"AAA", "SIG", "index"}
    assert set(single_seen.values()) == {expected_last}


# --------------------------------------------------------- the real-run guard
#
# A real run permanently spends a family's one-shot ABI-1 slot, so on a REAL run
# (after the trim) history_cutoff.check_real_run_guard refuses a history end later
# than the run's tested date or one that drops more than --max-history-drop
# sessions from any series. A DRY run never refuses on it: it reports the guard's
# answer as `history_end_guard_would_refuse` so a rehearsal predicts the real run.

SINGLE = {"abi1_single": MODULE_SINGLE}
ONE = ["abi1_single"]


def test_a_real_run_refuses_a_history_end_later_than_the_tested_date(
        tmp_path, monkeypatch, capsys):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=False, tested_date="2021-03-31")

    refusal = result["history_end_refusal"]
    _assert_standard_refusal(refusal, code="history_end_after_tested_date",
                             requested=HE, symbols=[])
    assert HE in refusal["reason"] and "2021-03-31" in refusal["reason"]
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])

    printed, _ = _refused_cli_run(
        tmp_path / "cli", monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        code="history_end_after_tested_date", history_end=HE,
        tested_date="2021-03-31", max_history_drop=HE_ACCEPT_ANY_DROP)
    assert printed["history_end_requested"] == HE


def test_a_history_end_equal_to_the_tested_date_is_accepted(tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=False, tested_date=HE)

    assert "history_end_refusal" not in result
    assert len(_outcomes(built["ledger"], "abi1_single")) == 1


def test_a_real_run_refuses_a_drop_above_the_default_limit(
        tmp_path, monkeypatch, capsys):
    """72 dropped sessions per series against the default limit of 10: refused,
    nothing scored, and the reason names the worst series and how to accept it."""
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=False, max_history_drop=_UNSET)

    refusal = result["history_end_refusal"]
    _assert_standard_refusal(refusal, code="history_end_drop_too_large",
                             requested=HE, symbols=["AAA", "SPY"])
    for needle in ("AAA", str(HE_DROP), str(history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS),
                   "pass --max-history-drop N to accept a larger drop"):
        assert needle in refusal["reason"], needle
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])

    # The same through the CLI, with no --max-history-drop given at all.
    printed, _ = _refused_cli_run(
        tmp_path / "cli", monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        code="history_end_drop_too_large", history_end=HE)
    assert printed["symbols"] == ["AAA", "SPY"]


def test_the_drop_limit_names_only_the_series_over_it_and_the_worst_one(
        tmp_path, monkeypatch):
    frames = dict(_FRAMES)
    frames["AAA"] = _FRAMES["AAA"].iloc[:HE_POS + 1 + 5]      # AAA drops just 5
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=False, frames=frames, max_history_drop=_UNSET)

    refusal = result["history_end_refusal"]
    assert refusal["reason_code"] == "history_end_drop_too_large"
    assert refusal["symbols"] == ["SPY"]                       # AAA is under 10
    assert f"{HE_DROP} sessions from SPY" in refusal["reason"]
    assert not os.path.exists(built["ledger"])


def test_the_drop_guard_looks_at_every_series_the_whole_batch_loaded(
        tmp_path, monkeypatch):
    """The family scoring AAA would pass on its own (AAA drops 5); DEF, used by
    its sibling, drops 72. The guard sees the whole batch, so neither scores."""
    frames = dict(_FRAMES)
    frames["AAA"] = _FRAMES["AAA"].iloc[:HE_POS + 1 + 5]
    result, built = _he_compute(
        tmp_path, monkeypatch,
        modules={"abi1_single": MODULE_SINGLE, "abi1_single_2": MODULE_SINGLE_2},
        accepted=["abi1_single", "abi1_single_2"], history_end=HE, dry_run=False,
        frames=frames, max_history_drop=_UNSET)

    refusal = result["history_end_refusal"]
    assert refusal["reason_code"] == "history_end_drop_too_large"
    assert refusal["symbols"] == ["DEF", "SPY"]
    assert result["computed"] == [] and not os.path.exists(built["ledger"])


def test_a_max_history_drop_that_is_not_a_whole_number_is_refused_not_ignored(
        tmp_path, monkeypatch):
    for bad in (-1, True, 2.5, "5"):
        result, built = _he_compute(
            tmp_path / f"bad_{bad!r}".replace(".", "_"), monkeypatch, modules=SINGLE,
            accepted=ONE, history_end=HE, dry_run=False, max_history_drop=bad)
        refusal = result["history_end_refusal"]
        assert refusal["reason_code"] == "history_end_invalid", bad
        assert "max_drop" in refusal["reason"], bad
        assert not os.path.exists(built["ledger"]), bad


def test_a_dry_run_also_refuses_a_batch_with_an_uncovered_series(
        tmp_path, monkeypatch, capsys):
    """The whole-batch refusal is not a real-run-only rule: a rehearsal that
    would score a partial universe would misreport the real run."""
    frames = _frames_missing_the_cutoff_bar("DEF")
    rc, stdout, payload, paths = _he_cli(
        tmp_path, monkeypatch, capsys,
        modules={"abi1_single": MODULE_SINGLE, "abi1_single_2": MODULE_SINGLE_2},
        accepted=["abi1_single", "abi1_single_2"], history_end=HE, dry_run=True,
        frames=frames)

    assert rc == 2
    printed = json.loads(stdout)
    assert printed["reason_code"] == "history_end_not_covered"
    assert printed["symbols"] == ["DEF"] and printed["dry_run"] is True
    assert payload is None
    assert _files_besides_the_ohlc_cache(paths["run_dir"]) == []
    assert not os.path.exists(paths["ledger"])


def test_an_explicit_max_history_drop_accepts_a_larger_drop_exactly_at_the_limit(
        tmp_path, monkeypatch, capsys):
    accepted_run, built = _he_compute(
        tmp_path / "at_limit", monkeypatch, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=False, max_history_drop=HE_DROP)
    assert "history_end_refusal" not in accepted_run
    assert _by_id(accepted_run)["abi1_single"]["account_end"] == HE
    assert len(_outcomes(built["ledger"], "abi1_single")) == 1

    just_under, built_under = _he_compute(
        tmp_path / "under_limit", monkeypatch, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=False, max_history_drop=HE_DROP - 1)
    assert (just_under["history_end_refusal"]["reason_code"]
            == "history_end_drop_too_large")
    assert not os.path.exists(built_under["ledger"])

    # The same accept through the CLI option: exit 0, results file, ledger row.
    code, stdout, payload, paths = _he_cli(
        tmp_path / "cli", monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=False, max_history_drop=HE_DROP)
    assert code == 0
    assert payload["history_end_requested"] == HE
    assert "history_end_guard_would_refuse" not in payload   # a real run
    assert len(_outcomes(paths["ledger"], "abi1_single")) == 1


def test_max_history_drop_defaults_to_the_shared_constant():
    import inspect
    default = inspect.signature(abi1_cli.compute_batch_abi1).parameters[
        "max_history_drop"].default
    assert default == history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS
    assert history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS == 10
    default_run = inspect.signature(abi1_cli.run_batch_abi1).parameters[
        "max_history_drop"].default
    assert default_run == history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS


@pytest.mark.parametrize("bad", ["-1", "-5", "abc", "1.5", "", " ", "1e3"])
def test_the_cli_rejects_a_max_history_drop_that_is_not_a_whole_number(
        tmp_path, monkeypatch, capsys, bad):
    with pytest.raises(SystemExit) as raised:
        _he_cli(tmp_path, monkeypatch, capsys, modules=SINGLE, accepted=ONE,
                history_end=HE, max_history_drop=bad)
    assert raised.value.code == 2
    assert "max-history-drop" in capsys.readouterr().err
    assert not os.path.exists(str(tmp_path / "research"))


def test_a_dry_run_reports_what_the_guard_would_do_and_does_not_refuse(
        tmp_path, monkeypatch):
    result, built = _he_compute(
        tmp_path / "over", monkeypatch, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=True, max_history_drop=_UNSET)

    assert "history_end_refusal" not in result
    assert _skips(result) == {} and set(_by_id(result)) == {"abi1_single"}
    would_refuse = result["history_end_guard_would_refuse"]
    assert isinstance(would_refuse, str)
    assert f"{HE_DROP} sessions from AAA" in would_refuse
    assert not os.path.exists(built["ledger"])
    # It predicts the real run: the real refusal is the very same text, followed
    # by the cache-kept / fresh-run-id sentence every post-load refusal carries.
    real, _ = _he_compute(
        tmp_path / "real", monkeypatch, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=False, max_history_drop=_UNSET)
    assert real["history_end_refusal"]["reason"].startswith(would_refuse)
    assert "retry with a fresh --run-id" in real["history_end_refusal"]["reason"]

    accepted, _ = _he_compute(
        tmp_path / "accepted", monkeypatch, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=True, max_history_drop=HE_DROP)
    assert accepted["history_end_guard_would_refuse"] is None
    assert set(_by_id(accepted)) == {"abi1_single"}

    late, _ = _he_compute(
        tmp_path / "late", monkeypatch, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=True, tested_date="2021-03-31")
    assert "2021-03-31" in late["history_end_guard_would_refuse"]
    assert set(_by_id(late)) == {"abi1_single"}

    off, _ = _he_compute(
        tmp_path / "off", monkeypatch, modules=SINGLE, accepted=ONE,
        history_end=None, dry_run=True)
    assert "history_end_guard_would_refuse" not in off


def test_a_dry_run_that_the_guard_would_refuse_says_so_in_the_report(
        tmp_path, monkeypatch, capsys):
    code, stdout, payload, paths = _he_cli(
        tmp_path, monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=True)

    assert code == 0
    would_refuse = payload["history_end_guard_would_refuse"]
    report = open(os.path.join(paths["run_dir"], "run-report-dry-run.md"),
                  encoding="utf-8").read()
    assert "a real run of this setting would be REFUSED" in report
    assert would_refuse in report


@pytest.mark.parametrize("code", list(history_cutoff.HISTORY_END_REASON_CODES))
def test_main_exits_2_for_every_history_end_refusal_code_and_writes_nothing(
        tmp_path, monkeypatch, capsys, code):
    """History_end_empty cannot be produced by a loaded frame (the loader rejects
    an empty one first), so the exit-code rule is proven by handing main() a
    refusal of each code the shared module defines."""
    exc = history_cutoff.HistoryCutoffError(code, f"synthetic {code}", symbols=("X",))
    monkeypatch.setattr(abi1_cli.batch5, "build_scope", lambda **kwargs: {})
    monkeypatch.setattr(
        abi1_cli, "run_batch_abi1",
        lambda scope, **kwargs: {"refused": True, **history_cutoff.refusal_payload(
            exc, history_end_requested=HE), "results": [], "skipped": [],
            "outputs_written": []})
    research_root = tmp_path / "research"

    capsys.readouterr()
    rc = abi1_cli.main(["--history-end", HE, "--research-root", str(research_root),
                        "--ledger", str(tmp_path / "ledger.jsonl")])
    printed = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert printed["refused"] is True and printed["reason_code"] == code
    assert printed["history_end_requested"] == HE and printed["symbols"] == ["X"]
    assert not research_root.exists() and not (tmp_path / "ledger.jsonl").exists()


def test_main_exits_0_when_no_history_end_refusal_happened(
        tmp_path, monkeypatch, capsys):
    code, stdout, payload, paths = _he_cli(
        tmp_path, monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        history_end=HE, max_history_drop=HE_DROP)
    assert code == 0
    printed = json.loads(stdout)
    assert printed["history_end"] == HE and printed["tested"] == 1
    assert printed["history_end_guard_would_refuse"] is None


# ------------------------------------------------------------------------
# Further guards. (1) The DOOR of compute_batch_abi1 validates the
# real-run guard's two inputs (tested date, max-history-drop) whenever a history
# end is set, for real AND dry runs, so neither can fail AFTER the load step has
# written the per-run OHLC cache. (2) An empty loaded universe refuses like every
# other history-end refusal. (3) Per-family series lists, a signal-only series
# under the drop guard, and a benchmark load failure. (4) The operator docs.
# ------------------------------------------------------------------------

import re  # noqa: E402

BAD_TESTED_DATES = ["2026-9-25", "2026/09/25", "", "not-a-date", None]
BAD_MAX_DROPS = [-1, True, 2.5, "5", None]


def _cache_dir(built):
    return os.path.join(built["run_dir"], "artifacts", "ohlc")


def _names(text, symbol):
    return re.search(rf"\b{re.escape(symbol)}\b", text) is not None


def _squash(text):
    return re.sub(r"\s+", "", text)


# ------------------------------------------------------------------ the door

@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad_date", BAD_TESTED_DATES)
def test_a_malformed_tested_date_is_refused_at_the_door_before_any_load(
        tmp_path, monkeypatch, dry_run, bad_date):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=dry_run, tested_date=bad_date)

    refusal = result["history_end_refusal"]
    _assert_standard_refusal(refusal, code="history_end_invalid", requested=HE,
                             symbols=[])
    assert "tested_date" in refusal["reason"]
    assert "OHLC cache" not in refusal["reason"], "a door refusal loaded nothing"
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(_cache_dir(built)), "no OHLC cache was written"
    assert not os.path.exists(built["run_dir"])
    assert not os.path.exists(built["ledger"])


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad_drop", BAD_MAX_DROPS)
def test_a_bad_max_history_drop_is_refused_at_the_door_before_any_load(
        tmp_path, monkeypatch, dry_run, bad_drop):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=dry_run, max_history_drop=bad_drop)

    refusal = result["history_end_refusal"]
    _assert_standard_refusal(refusal, code="history_end_invalid", requested=HE,
                             symbols=[])
    assert "max_drop" in refusal["reason"]
    assert "OHLC cache" not in refusal["reason"], "a door refusal loaded nothing"
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(_cache_dir(built)), "no OHLC cache was written"
    assert not os.path.exists(built["run_dir"])
    assert not os.path.exists(built["ledger"])


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad_date", ["2026-9-25", ""])
def test_the_cli_refuses_a_malformed_tested_date_at_the_door_and_writes_nothing(
        tmp_path, monkeypatch, capsys, dry_run, bad_date):
    rc, stdout, payload, paths = _he_cli(
        tmp_path, monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=dry_run, tested_date=bad_date,
        max_history_drop=HE_ACCEPT_ANY_DROP)

    assert rc == 2
    printed = json.loads(stdout)
    assert printed["refused"] is True and printed["reason_code"] == "history_end_invalid"
    assert printed["history_end_requested"] == HE and printed["symbols"] == []
    assert "tested_date" in printed["reason"] and printed["dry_run"] is dry_run
    assert payload is None
    assert not os.path.exists(paths["run_dir"]), "nothing at all was written"
    assert not os.path.exists(paths["ledger"])


@pytest.mark.parametrize("dry_run", [True, False])
def test_with_the_history_end_unset_a_malformed_tested_date_or_drop_limit_is_not_checked(
        tmp_path, monkeypatch, dry_run):
    """The door check belongs to the opt-in only: with no history end the two
    guard inputs are never looked at."""
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=None,
        dry_run=dry_run, tested_date="2026-9-25", max_history_drop=-1)

    assert "history_end_refusal" not in result
    assert _skips(result) == {} and set(_by_id(result)) == {"abi1_single"}
    assert sorted(os.listdir(_cache_dir(built))) == ["AAA.csv", "SPY.csv"], (
        "the load step ran")
    if dry_run:
        assert not os.path.exists(built["ledger"])
    else:
        assert len(_outcomes(built["ledger"], "abi1_single")) == 1


# ------------------------------------------------- an empty loaded universe

@pytest.mark.parametrize("dry_run", [True, False])
def test_a_history_end_with_no_series_loaded_refuses_like_any_other_refusal(
        tmp_path, monkeypatch, capsys, dry_run):
    """Every series fails to load: the setting would otherwise be silently
    ignored (the trim call skipped) and the run would follow the ordinary
    benchmark-missing skip, writing a report and a results file and exiting 0."""
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=dry_run, frames={})

    refusal = result["history_end_refusal"]
    _assert_standard_refusal(refusal, code="history_end_not_covered", requested=HE,
                             symbols=[])
    assert "no series were supplied" in refusal["reason"]
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])
    assert not os.path.exists(_cache_dir(built)), "nothing loaded, so nothing cached"

    rc, stdout, payload, paths = _he_cli(
        tmp_path / "cli", monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=dry_run, frames={},
        max_history_drop=HE_ACCEPT_ANY_DROP)
    assert rc == 2
    printed = json.loads(stdout)
    assert printed["refused"] is True
    assert printed["reason_code"] == "history_end_not_covered"
    assert printed["history_end_requested"] == HE and printed["dry_run"] is dry_run
    assert payload is None, "no results file"
    assert _files_besides_the_ohlc_cache(paths["run_dir"]) == [], "no run report"
    assert not os.path.exists(paths["ledger"])


def test_with_the_history_end_unset_a_run_where_every_series_fails_to_load_skips_on_missing_data(
        tmp_path, monkeypatch, capsys):
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=None,
        dry_run=False, frames={})

    assert "history_end_refusal" not in result and "history_end_requested" not in result
    assert result["computed"] == []
    skip = _skips(result)["abi1_single"]
    assert skip["reason_code"] == "missing_data"
    assert "benchmark OHLC (SPY)" in skip["reason"]
    assert set(result["input_errors"]) == {"AAA", "SPY"}
    assert not os.path.exists(built["ledger"])

    rc, stdout, payload, paths = _he_cli(
        tmp_path / "cli", monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        history_end=None, dry_run=False, frames={})
    assert rc == 0
    assert payload["results"] == [] and len(payload["skipped"]) == 1
    assert "history_end" not in payload and "history_end_requested" not in payload
    assert os.path.exists(os.path.join(paths["run_dir"], "run-report.md"))
    assert not {"refused", "reason_code"} & set(json.loads(stdout))


# ------------------ per-family series, signal-only drops, benchmark failures

def _raw_last(position):
    return f"{_index(N_SESSIONS)[position]:%Y-%m-%d}"


def _frames_with_staggered_raw_last_bars():
    """Every series holds the cutoff bar but each ends at a different later date."""
    frames = _probe_frames()                        # the shared frames + SIG
    frames["AAA"] = frames["AAA"].iloc[:HE_POS + 1 + 5]
    frames["DEF"] = frames["DEF"].iloc[:HE_POS + 1 + 20]
    frames["SIG"] = frames["SIG"].iloc[:HE_POS + 1 + 40]
    return frames


def test_each_familys_ledger_note_names_only_its_own_series_and_the_run_covers_all(
        tmp_path, monkeypatch):
    modules = {"abi1_probe_single": MODULE_PROBE_SINGLE,
               "abi1_single_2": MODULE_SINGLE_2}
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=modules,
        accepted=["abi1_probe_single", "abi1_single_2"], history_end=HE,
        dry_run=False, frames=_frames_with_staggered_raw_last_bars())

    assert _skips(result) == {}
    raw_last = {"AAA": _raw_last(HE_POS + 5), "DEF": _raw_last(HE_POS + 20),
                "SIG": _raw_last(HE_POS + 40), "SPY": HE_RAW_LAST}
    assert len(set(raw_last.values())) == 4, "the fixture really differs per series"

    note_a = _outcomes(built["ledger"], "abi1_probe_single")[0]["note"]
    note_b = _outcomes(built["ledger"], "abi1_single_2")[0]["note"]
    # The family that trades AAA and reads SIG lists exactly AAA, SIG and SPY.
    for symbol in ("AAA", "SIG", "SPY"):
        assert f"{symbol} {raw_last[symbol]}" in note_a, symbol
    assert not _names(note_a, "DEF")
    # The family that trades DEF lists exactly DEF and SPY.
    for symbol in ("DEF", "SPY"):
        assert f"{symbol} {raw_last[symbol]}" in note_b, symbol
    assert not _names(note_b, "AAA") and not _names(note_b, "SIG")

    items = _by_id(result)
    assert set(items["abi1_probe_single"]["history_end"]["by_symbol"]) == {
        "AAA", "SIG", "SPY"}
    assert set(items["abi1_single_2"]["history_end"]["by_symbol"]) == {"DEF", "SPY"}
    # The run-level provenance covers every loaded series, each with its own bar.
    run_level = result["history_end_provenance"]["by_symbol"]
    assert set(run_level) == {"AAA", "DEF", "SIG", "SPY"}
    for symbol, last in raw_last.items():
        assert run_level[symbol]["raw_last"] == last, symbol
        assert run_level[symbol]["kept_last"] == HE, symbol


def _frames_where_only_the_signal_series_drops_a_lot():
    """AAA (traded) and SPY (benchmark) lose 5 sessions at the cutoff, under the
    default limit; the signal-only series SIG loses 72."""
    frames = _probe_frames()
    frames["AAA"] = frames["AAA"].iloc[:HE_POS + 1 + 5]
    frames["SPY"] = frames["SPY"].iloc[:HE_POS + 1 + 5]
    return frames


def test_the_real_run_guard_sees_a_signal_only_series(tmp_path, monkeypatch):
    modules = {"abi1_probe_single": MODULE_PROBE_SINGLE}
    accepted = ["abi1_probe_single"]
    frames = _frames_where_only_the_signal_series_drops_a_lot()

    result, built = _he_compute(
        tmp_path / "real", monkeypatch, modules=modules, accepted=accepted,
        history_end=HE, dry_run=False, frames=frames, max_history_drop=_UNSET)
    refusal = result["history_end_refusal"]
    _assert_standard_refusal(refusal, code="history_end_drop_too_large",
                             requested=HE, symbols=["SIG"])
    assert f"{HE_DROP} sessions from SIG" in refusal["reason"]
    assert result["computed"] == [] and result["skipped"] == []
    assert not os.path.exists(built["ledger"])

    # The rehearsal predicts it: a dry run scores and names the same series.
    dry, _ = _he_compute(
        tmp_path / "dry", monkeypatch, modules=modules, accepted=accepted,
        history_end=HE, dry_run=True, frames=frames, max_history_drop=_UNSET)
    assert "history_end_refusal" not in dry
    assert set(_by_id(dry)) == {"abi1_probe_single"}
    assert f"{HE_DROP} sessions from SIG" in dry["history_end_guard_would_refuse"]


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_benchmark_load_failure_still_reports_the_history_end_that_was_requested(
        tmp_path, monkeypatch, capsys, dry_run):
    frames = {symbol: frame for symbol, frame in _FRAMES.items() if symbol != "SPY"}
    result, built = _he_compute(
        tmp_path, monkeypatch, modules=SINGLE, accepted=ONE, history_end=HE,
        dry_run=dry_run, frames=frames)

    assert "history_end_refusal" not in result and result["computed"] == []
    skip = _skips(result)["abi1_single"]
    assert skip["reason_code"] == "missing_data" and "benchmark OHLC" in skip["reason"]
    assert "SPY" in result["input_errors"]
    assert result["history_end_requested"] == HE
    assert not os.path.exists(built["ledger"])

    # ...and the results file and the printed summary carry it too.
    rc, stdout, payload, paths = _he_cli(
        tmp_path / "cli", monkeypatch, capsys, modules=SINGLE, accepted=ONE,
        history_end=HE, dry_run=dry_run, frames=frames,
        max_history_drop=HE_ACCEPT_ANY_DROP)
    assert rc == 0
    assert payload["history_end_requested"] == HE
    assert payload["history_end"]["history_end"] == HE
    assert json.loads(stdout)["history_end"] == HE


# ------------------------------------------------------------- operator docs

_DOC_NEEDLES = [
    "give a dry run the real --tested-date",
    "default is today's date",
    "no stale-cache refusal",
    "a reused --run-id silently reads the cache",
    "always use a fresh --run-id",
]


def test_the_history_end_help_gives_the_dry_run_and_fresh_run_id_advice(capsys):
    with pytest.raises(SystemExit) as raised:
        abi1_cli.main(["--help"])
    assert raised.value.code == 0
    text = _squash(capsys.readouterr().out).lower()
    for needle in _DOC_NEEDLES:
        assert _squash(needle) in text, needle


def test_the_module_docstring_gives_the_same_advice_and_requires_a_fresh_run_id():
    doc = _squash(abi1_cli.__doc__).lower()
    for needle in _DOC_NEEDLES:
        assert _squash(needle) in doc, needle
    assert _squash("a real retry needs a fresh --run-id") in doc


# ---- door ordering: a bad tested date / max-drop wins over scope and quote errors

def test_main_refuses_a_bad_tested_date_under_a_history_end_before_building_the_scope(tmp_path, capsys):
    """A history-end door refusal must not be preempted by a catalog, quote or
    module problem: with every input path pointing at a file that does not exist,
    main() still gives the standard refusal (exit 2) rather than a file error."""
    missing = str(tmp_path / "does-not-exist")
    argv = ["--accepted", missing, "--quotes", missing, "--catalog", missing,
            "--modules-dir", missing, "--ledger", str(tmp_path / "ledger.jsonl"),
            "--research-root", str(tmp_path / "research"), "--run-id", RUN_ID,
            "--tested-date", "2026-9-25", "--dry-run"]
    capsys.readouterr()
    assert abi1_cli.main(argv + ["--history-end", HE]) == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["refused"] is True and printed["reason_code"] == "history_end_invalid"
    assert printed["history_end_requested"] == HE and printed["dry_run"] is True
    assert printed["symbols"] == [] and "tested_date" in printed["reason"]
    assert not (tmp_path / "research").exists() and not (tmp_path / "ledger.jsonl").exists()
    # Control: with the setting UNSET the same argv never reaches the door, so the
    # missing inputs surface as a raised error, not a standard refusal.
    with pytest.raises(Exception) as unset:
        abi1_cli.main(argv)
    assert not isinstance(unset.value, SystemExit)


def test_run_batch_abi1_refuses_a_bad_tested_date_before_the_synthetic_quote_check(tmp_path, monkeypatch):
    built = _build_scope(tmp_path, monkeypatch, modules=SINGLE, accepted=ONE,
                         loader=_he_loader(_FRAMES))
    scope = built["scope"]
    scope["quotes"] = {**scope["quotes"], "notes": "SYNTHETIC quote file"}
    refused = abi1_cli.run_batch_abi1(
        scope, tested_date="2026-9-25", run_id=RUN_ID, ledger_path=built["ledger"],
        dry_run=False, history_end=HE)
    assert refused["refused"] is True and refused["reason_code"] == "history_end_invalid"
    assert refused["history_end_requested"] == HE and refused["outputs_written"] == []
    # Control: a valid tested date still meets the synthetic-quote refusal first.
    with pytest.raises(ValueError, match="SYNTHETIC"):
        abi1_cli.run_batch_abi1(
            scope, tested_date=TESTED_DATE, run_id=RUN_ID, ledger_path=built["ledger"],
            dry_run=False, history_end=HE)
    # And with the setting UNSET a bad tested date never reaches the door.
    with pytest.raises(ValueError, match="SYNTHETIC"):
        abi1_cli.run_batch_abi1(
            scope, tested_date="2026-9-25", run_id=RUN_ID, ledger_path=built["ledger"],
            dry_run=False)
