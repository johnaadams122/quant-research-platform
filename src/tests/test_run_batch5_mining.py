"""Tests for scripts/run_batch5_mining.py (the batch5 directional driver)
and for run_batch3_mining.score_batch's ``retest_family_ids`` hook.

Everything runs on tmp copies: tmp ledger, tmp catalog, tmp quotes, tmp
accepted.json, a tmp modules dir of fake candidate modules, and injected
deterministic synthetic OHLC (no network). The default
data/mining/trial_ledger.jsonl and data/mining/catalog.json paths are never
touched.
"""
import datetime
import json
import math
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

import trading_calendar
import run_batch3_mining as batch3
import run_batch5_mining as batch5
from _portable_paths_support import DRIVE_PATH, workstation_leaks
from mining import (benchmark_relative, constants, dsr, evidence, reporting,
                    trial_ledger)

TESTED_DATE = "2026-09-04"
RUN_ID = "2026-09-04-batch5-test"
SYMBOLS = ("SPY", "AAA", "BBB", "CCC", "RSK", "DEF", "LATE", "ONEBAR",
           "MID", "ALPH")
HOLDOUT_START = "2026-03-04"     # == derive_holdout_window(TESTED_DATE)


# ------------------------------------------------------------ synthetic data

def _index() -> pd.DatetimeIndex:
    """Business days before the vendored NYSE calendar starts, then the REAL
    NYSE sessions through TESTED_DATE -- so a rotation candidate's holdout
    matches run_batch3_mining._expected_holdout_calendar exactly."""
    sessions = trading_calendar.load_default_sessions()
    end = datetime.date(2026, 9, 4)
    early = pd.bdate_range("2019-01-02",
                           pd.Timestamp(sessions[0]) - pd.Timedelta(days=1))
    late = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions if s <= end])
    return early.append(late)


def _frame(seed: int) -> pd.DataFrame:
    index = _index()
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0004, 0.01, len(index))
    close = 100.0 * np.exp(np.cumsum(steps))
    # Gap-free opens (Open_t == Close_{t-1}) so an always-long strategy's
    # mark-to-market series equals buy-and-hold except for its exit cost.
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


_FRAMES = {symbol: _frame(seed) for seed, symbol in enumerate(SYMBOLS, start=11)}
# Two late-inception pool members, like the recent listings in the shipped
# universe snapshot (FDXF/HONA/SPCX, first bars May-June 2026): LATE has NO
# in-sample bar at all, ONEBAR has exactly one -- both below
# run_batch3_mining._require_bucket's two-bar floor, so without per-member
# exclusion either one would make the whole pooled family unscoreable.
_FRAMES["LATE"] = _FRAMES["LATE"].loc["2026-04-01":].copy()
_ONEBAR_IS, _ONEBAR_HO = batch3._window_masks(
    _FRAMES["ONEBAR"].index, HOLDOUT_START, TESTED_DATE)
_FRAMES["ONEBAR"] = pd.concat([_FRAMES["ONEBAR"].loc[_ONEBAR_IS].tail(1),
                               _FRAMES["ONEBAR"].loc[_ONEBAR_HO]]).copy()
# MID lists in 2022 rather than 2019, but carries plenty of bars in BOTH
# window buckets, so it is a fully KEPT pool member whose calendar is
# genuinely shorter than its pool-mates'. That is what makes the pooled
# panel's union-calendar 0.0-fill (and its equal-weight divisor) observable
# rather than a no-op on three identical calendars.
_FRAMES["MID"] = _FRAMES["MID"].loc["2022-01-03":].copy()


def _alpha_frame(*, up=0.015, down=-0.004, noise=0.002, seed=99) -> pd.DataFrame:
    """A price path with a deterministic exploitable pattern: the first three
    bars of every 10-bar cycle rise hard, the other seven drift down, plus
    small seeded noise so the OLS residuals are not identically zero (exactly
    zero residuals make t_alpha_hac infinite instead of merely large). A
    candidate holding ONLY the up bars has real alpha against its own
    buy-and-hold, which is what makes directional_support able to fire at all.
    """
    index = _index()
    position = np.arange(len(index))
    rng = np.random.RandomState(seed)
    steps = np.where(position % 10 < 3, up, down) + rng.normal(0.0, noise, len(index))
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


_FRAMES["ALPH"] = _alpha_frame()


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

_HEADER = """
from mining.backtest_engine import SignalResult
"""

MODULE_SINGLE = _HEADER + """
FAMILY_ID = "b5_single"
SHAPE = "single_long"
TRADED = ["AAA"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 50

def make_signal_fn(ticker, index, frames):
    assert ticker in frames
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""

MODULE_POOLED = _HEADER + """
FAMILY_ID = "b5_pooled"
SHAPE = "pooled_single_asset"
TRADED = ["AAA", "BBB", "CCC"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 150

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""

MODULE_ROTATION = _HEADER + """
FAMILY_ID = "b5_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["RSK", "DEF"]
SIGNAL = ["SPY"]
SNAPSHOTS = {}
BENCHMARK = "RSK"
PARAMETERS = {"block": 40}
EXPECTED_MIN_TRADES = 20

def make_target_fn(frames):
    def target(truncated):
        n = len(truncated["RSK"])
        return "RSK" if (n // 40) % 2 == 0 else "DEF"
    return target
"""

MODULE_ALWAYS = _HEADER + """
FAMILY_ID = "b5_always"
SHAPE = "single_long"
TRADED = ["AAA"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"always": True}
EXPECTED_MIN_TRADES = 1

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=True, exit=False)
    return signal
"""

MODULE_BAD = _HEADER + """
FAMILY_ID = "b5_bad"
SHAPE = "weird_shape"
TRADED = ["AAA"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {}
EXPECTED_MIN_TRADES = 1

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=True, exit=False)
    return signal
"""

MODULE_DRIFT = MODULE_SINGLE.replace('"b5_single"', '"b5_drift"')

# Five healthy members, so ONE exclusion leaves 4/5 = exactly the coverage
# floor and the family still scores. The 3-member MODULE_POOLED is kept for
# the aggregation test, where one drop would fall below the floor.
MODULE_POOLED5 = (MODULE_POOLED
                  .replace('"b5_pooled"', '"b5_pooled5"')
                  .replace('TRADED = ["AAA", "BBB", "CCC"]',
                           'TRADED = ["AAA", "BBB", "CCC", "RSK", "DEF"]'))
# One late-inception member (LATE has NO in-sample bar) among five.
MODULE_POOLED_LATE = (MODULE_POOLED
                      .replace('"b5_pooled"', '"b5_pooled_late"')
                      .replace('TRADED = ["AAA", "BBB", "CCC"]',
                               'TRADED = ["AAA", "BBB", "CCC", "RSK", "LATE"]'))
# Two unusable members out of four: 50%, below the coverage floor.
MODULE_POOLED_THIN = (MODULE_POOLED
                      .replace('"b5_pooled"', '"b5_pooled_thin"')
                      .replace('TRADED = ["AAA", "BBB", "CCC"]',
                               'TRADED = ["AAA", "BBB", "LATE", "ONEBAR"]'))
# A pooled family whose every member is unusable.
MODULE_POOLED_DEAD = (MODULE_POOLED
                      .replace('"b5_pooled"', '"b5_pooled_dead"')
                      .replace('TRADED = ["AAA", "BBB", "CCC"]',
                               'TRADED = ["LATE", "ONEBAR"]'))
# An event_pooled family: the other pooled shape, same exclusion rule.
MODULE_EVENT_POOLED = (MODULE_POOLED
                       .replace('"b5_pooled"', '"b5_event"')
                       .replace('SHAPE = "pooled_single_asset"',
                                'SHAPE = "event_pooled"')
                       .replace('TRADED = ["AAA", "BBB", "CCC"]',
                                'TRADED = ["AAA", "BBB", "CCC", "RSK", "LATE"]'))

CATALOG_PARAMETERS = {
    "b5_single": {"period": 15, "hold": 7},
    "b5_pooled": {"period": 15, "hold": 7},
    "b5_rotation": {"block": 40},
    "b5_always": {"always": True},
    "b5_bad": {},
    "b5_drift": {"period": 20, "hold": 7},   # deliberately != the module's
    "b5_pooled_late": {"period": 15, "hold": 7},
    "b5_pooled5": {"period": 15, "hold": 7},
    "b5_pooled_thin": {"period": 15, "hold": 7},
    "b5_pooled_raiser": {"period": 15, "hold": 7},
    "b5_pooled_diag": {"period": 15, "hold": 7},
    "b5_pooled_dead": {"period": 15, "hold": 7},
    "b5_event": {"period": 15, "hold": 7},
    "b5_raiser": {"period": 15, "hold": 7},
    "b5_single_ccc": {"period": 15, "hold": 7},
    "b5_pooled_signal": {"period": 15, "hold": 7},
    "b5_pooled_bench": {"period": 15, "hold": 7},
    "b5_pooled_mean": {"period": 15, "hold": 7},
    "b5_alpha": {"cycle": 6, "hold": 3},
}
TRADED_BY_ID = {
    "b5_single": ["AAA"], "b5_pooled": ["AAA", "BBB", "CCC"],
    "b5_rotation": ["RSK", "DEF"], "b5_always": ["AAA"], "b5_bad": ["AAA"],
    "b5_drift": ["AAA"],
    "b5_pooled_late": ["AAA", "BBB", "CCC", "RSK", "LATE"],
    "b5_pooled5": ["AAA", "BBB", "CCC", "RSK", "DEF"],
    "b5_pooled_thin": ["AAA", "BBB", "LATE", "ONEBAR"],
    "b5_pooled_raiser": ["AAA", "BBB", "CCC", "RSK", "DEF"],
    "b5_pooled_diag": ["AAA", "BBB", "CCC"],
    "b5_pooled_dead": ["LATE", "ONEBAR"],
    "b5_event": ["AAA", "BBB", "CCC", "RSK", "LATE"],
    "b5_raiser": ["CCC"], "b5_single_ccc": ["CCC"],
    "b5_pooled_signal": ["AAA", "BBB"], "b5_pooled_bench": ["AAA", "BBB"],
    "b5_pooled_mean": ["AAA", "BBB", "MID"], "b5_alpha": ["ALPH"],
}


def _write_modules(tmp_path, modules: dict) -> str:
    modules_dir = tmp_path / "mods"
    modules_dir.mkdir(exist_ok=True)
    for name, text in modules.items():
        (modules_dir / f"{name}.py").write_text(textwrap.dedent(text), encoding="utf-8")
    return str(modules_dir)


def _write_catalog(tmp_path, family_ids, parameters=None) -> str:
    entries = []
    for family_id in family_ids:
        params = (parameters or {}).get(family_id, CATALOG_PARAMETERS[family_id])
        entries.append({
            "family_id": family_id, "name": f"Test {family_id}",
            "mechanism": f"synthetic mechanism for {family_id}",
            "tickers": TRADED_BY_ID[family_id], "parameters": params,
            "account_types": ["roth"], "status": "eligible",
            "source": "test", "added_date": "2026-09-04",
        })
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return str(path)


def _write_accepted(tmp_path, family_ids) -> str:
    payload = {"batch": "2026-09-04-batch5", "curated_on": "2026-09-04",
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


def _write_quotes(tmp_path, *, drop=(), suspect=(), wide=(),
                  source="test-quotes", market_state="closed (test)",
                  spread=0.02) -> str:
    quotes = {}
    for symbol in SYMBOLS:
        if symbol in drop:
            continue
        if symbol in wide:
            quotes[symbol] = _quote(90.0, 110.0)          # ~2000 bps
        else:
            quotes[symbol] = _quote(100.0, 100.0 + spread,
                                    suspect=(symbol in suspect))
    payload = {"sampled_at_local": "2026-09-04T16:32:00-04:00",
               "source": source, "market_state": market_state,
               "quotes": quotes}
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return str(path)


def _run(tmp_path, monkeypatch, *, modules: dict, accepted: list,
         catalog_ids: list | None = None, dry_run=False, retest=None,
         families=None, quotes_kwargs=None, ledger=None, loader=None,
         expect_error=None):
    """Drive the real CLI (main) end-to-end on tmp inputs. Returns
    (payload, paths)."""
    loader = loader or _loader_with_counter()[0]
    monkeypatch.setattr(batch5, "default_ohlc_loader", loader)
    modules_dir = _write_modules(tmp_path, modules)
    catalog_path = _write_catalog(tmp_path, catalog_ids or accepted)
    accepted_path = _write_accepted(tmp_path, accepted)
    quotes_path = _write_quotes(tmp_path, **(quotes_kwargs or {}))
    ledger = ledger or str(tmp_path / "ledger.jsonl")
    research_root = str(tmp_path / "research")
    argv = ["--accepted", accepted_path, "--quotes", quotes_path,
            "--run-id", RUN_ID, "--tested-date", TESTED_DATE,
            "--ledger", ledger, "--catalog", catalog_path,
            "--modules-dir", modules_dir, "--research-root", research_root,
            "--results-log-out", str(tmp_path / "RESULTS-LOG.md"),
            "--near-miss-log-out", str(tmp_path / "NEAR-MISS-LOG.md")]
    if dry_run:
        argv.append("--dry-run")
    if retest:
        argv += ["--retest", ",".join(retest)]
    if families:
        argv += ["--families", ",".join(families)]
    run_dir = os.path.join(research_root, RUN_ID)
    out = os.path.join(run_dir, "batch5-dry-run.json" if dry_run else "batch5-results.json")
    paths = {"run_dir": run_dir, "ledger": ledger, "out": out,
             "catalog": catalog_path, "modules_dir": modules_dir,
             "results_log": str(tmp_path / "RESULTS-LOG.md")}
    if expect_error is not None:
        with pytest.raises(expect_error) as excinfo:
            batch5.main(argv)
        return excinfo, paths
    assert batch5.main(argv) == 0
    with open(out, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, paths


def _by_id(payload):
    return {item["family_id"]: item for item in payload["results"]}


def _skips(payload):
    return {row["family_id"]: row for row in payload["skipped"]}


# ------------------------------------------------ discovery + contract

def test_bad_module_is_skipped_with_reason_and_good_module_runs(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE, "b5_bad": MODULE_BAD},
                      accepted=["b5_single", "b5_bad"])
    assert set(_by_id(payload)) == {"b5_single"}
    skip = _skips(payload)["b5_bad"]
    assert skip["reason_code"] == "invalid_module"
    assert "SHAPE" in skip["reason"]
    assert [p["module"] for p in payload["module_problems"]] == ["b5_bad"]


def test_accepted_without_module_and_module_without_acceptance_are_both_reported(
        tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE, "b5_always": MODULE_ALWAYS},
                      accepted=["b5_single", "b5_pooled"],
                      catalog_ids=["b5_single", "b5_pooled", "b5_always"])
    assert _skips(payload)["b5_pooled"]["reason_code"] == "no_module"
    assert payload["unmatched_modules"] == ["b5_always"]


@pytest.mark.parametrize("mutation,needle", [
    ('TRADED = ["AAA"]', 'TRADED = []'),
    ('SNAPSHOTS = {}', 'SNAPSHOTS = {"x": {"path": "nope.json", "sha256": "0"}}'),
    ('EXPECTED_MIN_TRADES = 50', 'EXPECTED_MIN_TRADES = True'),
    ('def make_signal_fn', 'def make_other_fn'),
])
def test_contract_validation_names_every_bad_field(tmp_path, mutation, needle):
    modules_dir = _write_modules(
        tmp_path, {"b5_single": MODULE_SINGLE.replace(mutation, needle)})
    found, problems = batch5.discover_modules(modules_dir)
    assert found == {}
    assert len(problems) == 1 and problems[0]["module"] == "b5_single"


def test_rotation_contract_requires_two_traded_and_target_fn(tmp_path):
    modules_dir = _write_modules(tmp_path, {
        "b5_rotation": MODULE_ROTATION.replace('TRADED = ["RSK", "DEF"]',
                                               'TRADED = ["RSK"]')})
    found, problems = batch5.discover_modules(modules_dir)
    assert found == {} and "exactly 2 TRADED" in problems[0]["reason"]


# --------------------------------------------------------- drift + costs

def test_parameter_drift_skips_the_candidate_and_writes_no_ledger_line(
        tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_drift": MODULE_DRIFT, "b5_single": MODULE_SINGLE},
                          accepted=["b5_drift", "b5_single"])
    skip = _skips(payload)["b5_drift"]
    assert skip["reason_code"] == "parameter_drift"
    assert '"period":20' in skip["reason"] and '"period":15' in skip["reason"]
    assert not trial_ledger.has_been_tested(paths["ledger"], "b5_drift")
    assert trial_ledger.has_been_tested(paths["ledger"], "b5_single")


def test_parameter_drift_compares_canonical_json_not_key_order():
    assert batch5.parameter_drift({"b": 1, "a": 2}, {"a": 2, "b": 1}) is None
    assert batch5.parameter_drift({"a": 2}, {"a": 2.5}) is not None


@pytest.mark.parametrize("quotes_kwargs,needle", [
    ({"drop": ("AAA",)}, "no quote"),
    ({"suspect": ("AAA",)}, "suspect"),
    ({"wide": ("AAA",)}, "exceeds the 300 bps gate"),
])
def test_quote_gate_skips_as_missing_data_never_invents_a_cost(
        tmp_path, monkeypatch, quotes_kwargs, needle):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": MODULE_SINGLE},
                          accepted=["b5_single"], quotes_kwargs=quotes_kwargs)
    skip = _skips(payload)["b5_single"]
    assert skip["reason_code"] == "missing_data"
    assert needle in skip["reason"] and "AAA" in skip["reason"]
    assert payload["results"] == []
    assert not os.path.exists(paths["ledger"])


def test_signal_only_tickers_need_no_quote(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_rotation": MODULE_ROTATION},
                      accepted=["b5_rotation"], quotes_kwargs={"drop": ("SPY",)})
    assert "b5_rotation" in _by_id(payload)


def test_cost_bps_comes_from_the_quotes_file_via_cost_tax_inputs():
    payload = {"sampled_at_local": "t", "market_state": "closed", "source": "s",
               "quotes": {"AAA": _quote(100.0, 100.02)}}
    costs = batch5.execution_costs_for(["AAA"], payload)
    expected = batch3.cost_tax_inputs.cost_bps_from_quotes(100.0, 100.02, 100.0, 100.02)
    assert costs["AAA"]["cost_bps"] == expected
    assert "closed" in costs["AAA"]["source"] and "t" in costs["AAA"]["source"]
    with pytest.raises(ValueError, match="0 < bid < ask"):
        batch5.execution_costs_for(
            ["AAA"], {**payload, "quotes": {"AAA": _quote(100.02, 100.0)}})


# ------------------------------------------------------- end to end

def test_single_long_end_to_end_records_first_test_and_outcome(tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": MODULE_SINGLE},
                          accepted=["b5_single"])
    item = _by_id(payload)["b5_single"]
    assert item["stats_in_sample"]["trade_count"] >= 15
    assert item["stats_holdout"]["trade_count"] >= 1
    assert item["stage"] in evidence.VALID_STAGES
    assert item["tickers"] == ["AAA"] and item["ticker"] == "AAA"
    assert item["window_source"] == "derived"
    assert item["holdout_start"] == "2026-03-04" and item["holdout_end"] == TESTED_DATE
    assert item["raw_catalog_parameters"] == CATALOG_PARAMETERS["b5_single"]
    assert "cost source" in item["note"] and "directional batch" in item["note"]
    events = trial_ledger.read_all(paths["ledger"])
    assert [e["event"] for e in events] == ["first_test", "outcome"]
    assert events[0]["holdout_start"] == "2026-03-04"
    assert events[1]["stage"] == item["stage"]
    assert events[1]["ticker_label"] == "AAA"
    # Outputs.
    assert os.path.exists(paths["out"])
    report = os.path.join(paths["run_dir"], "run-report.md")
    assert os.path.exists(report)
    text = open(report, encoding="utf-8").read()
    assert "# Mining run — 2026-09-04" in text
    assert "| b5_single | single_long | AAA |" in text
    trades_dir = os.path.join(paths["run_dir"], "artifacts", "trades")
    for bucket in ("in_sample", "holdout"):
        csv = os.path.join(trades_dir, f"b5_single_{bucket}.csv")
        assert len(pd.read_csv(csv)) == item[f"stats_{bucket}"]["trade_count"]
    # execute_batch_run refreshed the durable logs after the ledger changed.
    assert os.path.exists(paths["results_log"])
    assert "b5_single" in open(paths["results_log"], encoding="utf-8").read()
    # The DSR score used the shared correction, not a per-candidate one.
    assert item["shared_n_trials"] == payload["shared_correction"]["n_trials"]
    assert payload["shared_correction"]["post_batch_ledger_floor"] == 1


def test_pooled_single_asset_pools_trades_across_three_tickers(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled": MODULE_POOLED},
                      accepted=["b5_pooled"])
    item = _by_id(payload)["b5_pooled"]
    assert item["ticker"] == "AAA/BBB/CCC"
    assert item["tickers"] == ["AAA", "BBB", "CCC"]
    assert item["stats_in_sample"]["ticker_count"] == 3

    def factory(_index):
        return lambda frame, i: batch3.backtest_engine.SignalResult(
            entry=(i % 15 == 0), exit=(i % 15 == 7))
    expected_is = expected_ho = 0
    for symbol in ("AAA", "BBB", "CCC"):
        windows = batch3.run_single_asset_windows(
            _FRAMES[symbol], factory, holdout_start="2026-03-04",
            holdout_end=TESTED_DATE, cost_bps=item["execution_costs"][symbol]["cost_bps"],
            tax_haircut_rate=0.0)
        expected_is += windows["stats_in_sample"]["trade_count"]
        expected_ho += windows["stats_holdout"]["trade_count"]
    assert expected_is > 0
    assert item["stats_in_sample"]["trade_count"] == expected_is
    assert item["stats_holdout"]["trade_count"] == expected_ho
    # ONE trial per family, never one per ticker.
    assert payload["shared_correction"]["post_batch_ledger_floor"] == 1
    assert "pooled: equal-weight MEAN" in " ".join(item["directional_panel"]["notes"])


def test_rotation_two_asset_end_to_end(tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_rotation": MODULE_ROTATION},
                          accepted=["b5_rotation"])
    item = _by_id(payload)["b5_rotation"]
    assert item["tickers"] == ["RSK", "DEF"] and item["ticker"] == "RSK/DEF"
    assert item["stats_in_sample"]["trade_count"] >= 15
    assert "signal_alignment" in item
    assert item["signal_alignment"]["holdout"]["lookback_buffer_days"] == (
        batch3.HOLDOUT_LOOKBACK_BUFFER_DAYS)
    assert set(item["execution_costs"]) == {"RSK", "DEF"}
    assert item["data_through"]["SPY"] == TESTED_DATE      # signal leg recorded
    csv = pd.read_csv(os.path.join(paths["run_dir"], "artifacts", "trades",
                                   "b5_rotation_in_sample.csv"))
    assert set(csv["asset"]) <= {"RSK", "DEF"}
    assert "rotation: per-leg" in " ".join(item["directional_panel"]["notes"])
    assert trial_ledger.latest_outcome(paths["ledger"], "b5_rotation") == item["stage"]


def test_dry_run_writes_no_ledger_line_and_no_real_run_report(tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": MODULE_SINGLE},
                          accepted=["b5_single"], dry_run=True)
    assert payload["dry_run"] is True
    assert _by_id(payload)["b5_single"]["stage"] in evidence.VALID_STAGES
    assert not os.path.exists(paths["ledger"])
    assert os.path.exists(os.path.join(paths["run_dir"], "run-report-dry-run.md"))
    assert not os.path.exists(os.path.join(paths["run_dir"], "run-report.md"))
    assert not os.path.exists(paths["results_log"])


def test_ohlc_cache_is_written_and_reused_on_rerun(tmp_path, monkeypatch):
    loader, calls = _loader_with_counter()
    _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
         accepted=["b5_single"], dry_run=True, loader=loader)
    assert sorted(calls) == ["AAA", "SPY"]
    cache_dir = os.path.join(tmp_path, "research", RUN_ID, "artifacts", "ohlc")
    assert sorted(os.listdir(cache_dir)) == ["AAA.csv", "SPY.csv"]
    calls.clear()
    payload, _ = _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True, loader=loader)
    assert calls == []
    assert payload["input_coverage"]["AAA"]["origin"] == "cache"


def test_families_filter_restricts_the_run(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE, "b5_always": MODULE_ALWAYS},
                      accepted=["b5_single", "b5_always"], families=["b5_always"],
                      dry_run=True)
    assert set(_by_id(payload)) == {"b5_always"}
    assert payload["provenance"]["families_filter"] == ["b5_always"]


# ------------------------------------------------ directional panel

_PANEL_KEYS = {"report_only", "strategy_daily_days", "own_benchmark",
               "market_benchmark", "daily_sharpe_annualized", "daily_moments",
               "ols_vs_own", "ols_vs_own_error", "ols_vs_spy", "ols_vs_spy_error",
               "full_window_vs_own", "time_matched_spy_excess_bps", "notes",
               "daily_dsr", "daily_dsr_error", "daily_dsr_inputs", "flags"}


def test_directional_panel_is_present_report_only_and_flags_beta_in_disguise(
        tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_always": MODULE_ALWAYS, "b5_single": MODULE_SINGLE},
                          accepted=["b5_always", "b5_single"])
    always = _by_id(payload)["b5_always"]
    panel = always["directional_panel"]
    assert set(panel) == _PANEL_KEYS
    assert panel["report_only"] is True
    assert panel["own_benchmark"] == ["AAA"] and panel["market_benchmark"] == "SPY"
    ols = panel["ols_vs_own"]
    assert ols is not None and panel["ols_vs_own_error"] is None
    # Always-long on gap-free data IS buy-and-hold minus one exit cost:
    # beta ~ 1, alpha slightly negative, so t_alpha_hac <= 0.
    assert abs(ols["beta"] - 1.0) < 0.02
    assert ols["t_alpha_hac"] <= 0.0 and abs(ols["t_alpha_hac"]) < 5.0
    assert panel["flags"] == {"beta_in_disguise": True, "directional_support": False}
    assert panel["time_matched_spy_excess_bps"]["in_sample"] is not None
    assert panel["daily_dsr_inputs"]["n_trials"] == payload["shared_correction"]["n_trials"]
    assert panel["daily_dsr_inputs"]["sr_variance_daily"] == (
        payload["directional_panel_inputs"]["sr_variance_daily"])
    assert "ddof=1" in payload["directional_panel_inputs"]["sr_variance_source"]
    assert isinstance(panel["daily_dsr"], float)
    assert set(panel["full_window_vs_own"]) >= {
        "strategy_sharpe_ann", "benchmark_sharpe_ann", "strategy_cagr",
        "benchmark_cagr", "strategy_max_dd", "benchmark_max_dd", "n_days",
        "time_in_market", "ols"}
    # REPORT-ONLY: the ledger row carries the pre-registered stage only.
    outcome = [e for e in trial_ledger.read_all(paths["ledger"])
               if e["event"] == "outcome" and e["family_id"] == "b5_always"][0]
    assert outcome["stage"] == always["stage"]
    assert "directional_panel" not in outcome
    assert "beta" not in json.dumps(outcome)


def test_daily_sr_variance_falls_back_to_the_daily_constant_not_the_per_trade_floor(
        tmp_path, monkeypatch):
    """The fallback on the DAILY path must be the daily-construction
    constant, never constants.DEFAULT_SR_VARIANCE_FLOOR.

    That floor's own comment (constants.py) declares it PER-TRADE, and the
    two constructions differ by orders of magnitude (a per-trade Sharpe
    variance is far larger than a daily one), which moves the DSR bar by
    roughly an order of magnitude. Borrowing the
    per-trade floor here is the same unit error account_relative.score_account
    refuses outright rather than committing (it declines to score at all when
    no daily variance exists). The daily value is the pinned constant.
    """
    payload, _ = _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True)
    inputs = payload["directional_panel_inputs"]
    assert inputs["sr_variance_daily"] == constants.DAILY_SR_VARIANCE
    assert "DAILY_SR_VARIANCE" in inputs["sr_variance_source"]
    # The guard that actually bites: the two constants are distinct, so a
    # regression back to the per-trade floor is observable rather than a
    # silent unit swap.
    assert constants.DAILY_SR_VARIANCE != constants.DEFAULT_SR_VARIANCE_FLOOR
    assert inputs["sr_variance_daily"] != constants.DEFAULT_SR_VARIANCE_FLOOR
    assert "DEFAULT_SR_VARIANCE_FLOOR" not in inputs["sr_variance_source"]


def test_panel_failure_never_costs_the_candidate_its_score(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("panel exploded")
    monkeypatch.setattr(batch5, "build_directional_panel", boom)
    payload, paths = _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
                          accepted=["b5_single"])
    item = _by_id(payload)["b5_single"]
    assert item["directional_panel"] is None
    assert "panel exploded" in payload["panel_errors"]["b5_single"]
    assert trial_ledger.latest_outcome(paths["ledger"], "b5_single") == item["stage"]


# ----------------------------------------------- retest hook (driver)

def test_driver_skips_a_pinned_family_unless_listed_in_retest(tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(
        ledger, "b5_single", "2026-09-01", "2026-03-01", "2026-09-01", [], "run-0")
    trial_ledger.record_outcome(
        ledger, "b5_single", "run-0", "2026-09-01", "cut_dsr", dsr_value=0.3,
        sharpe_in_sample=0.1)
    payload, _ = _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], ledger=ledger)
    assert _skips(payload)["b5_single"]["reason_code"] == "already_pinned_not_in_retest"
    assert len(trial_ledger.read_all(ledger)) == 2

    payload, _ = _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], ledger=ledger, retest=["b5_single"])
    item = _by_id(payload)["b5_single"]
    assert item["window_source"] == "pinned"
    assert (item["holdout_start"], item["holdout_end"]) == ("2026-03-01", "2026-09-01")
    events = trial_ledger.read_all(ledger)
    assert len([e for e in events if e["event"] == "first_test"]) == 1
    assert len([e for e in events if e["event"] == "outcome"]) == 2
    assert payload["shared_correction"]["post_batch_ledger_floor"] == 1  # no new charge


# ------------------------------------- retest hook (score_batch itself)

def _item(family_id="family-a", holdout=("2024-07-01", "2024-12-31")):
    returns = pd.Series([0.0, 0.01, 0.0, -0.002, 0.004],
                        index=pd.date_range("2024-01-02", periods=5, freq="B"))
    return {
        "family_id": family_id, "ticker": "SPY", "tickers": ["SPY"],
        "holdout_start": holdout[0], "holdout_end": holdout[1],
        "holdout_complete": True,
        "stats_in_sample": {"trade_count": 40, "net_edge_bps": 100.0,
                            "sharpe": 0.5, "skew": 0.0, "kurtosis": 3.0},
        # MIN_HOLDOUT_TRADES (20) -- this fixture is reused
        # by tests asserting a SCORED outcome (evidence.SCORED_STAGES),
        # so its holdout trade count must clear the floor.
        "stats_holdout": {"trade_count": constants.MIN_HOLDOUT_TRADES,
                          "net_edge_bps": 25.0,
                          "sharpe": 0.1, "skew": 0.0, "kurtosis": 3.0},
        "holdout_net_edge_bps": 25.0, "daily_returns": returns,
        "in_sample_start": "2020-01-02", "in_sample_end": "2024-06-28",
        "catalog_parameters": {}, "raw_catalog_parameters": {},
        "note": "test computation",
    }


def _pin_with_real_outcome(ledger, holdout=("2024-07-01", "2024-12-31")):
    trial_ledger.record_first_test(
        ledger, "family-a", "2024-12-01", holdout[0], holdout[1], [], "run-0")
    trial_ledger.record_outcome(
        ledger, "family-a", "run-0", "2024-12-01", "cut_dsr", dsr_value=0.4,
        sharpe_in_sample=0.5)


def test_score_batch_default_still_refuses_an_already_pinned_family(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    _pin_with_real_outcome(ledger)
    with pytest.raises(RuntimeError, match="already pinned"):
        batch3.score_batch([_item()], ledger_path=ledger, run_id="run-1",
                           tested_date="2024-12-31", dry_run=False)
    assert len(trial_ledger.read_all(ledger)) == 2


def test_score_batch_retest_rescores_against_the_pinned_window_once(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    _pin_with_real_outcome(ledger)
    result = batch3.score_batch(
        [_item()], ledger_path=ledger, run_id="run-1", tested_date="2024-12-31",
        dry_run=False, retest_family_ids=frozenset({"family-a"}))[0]
    assert result["stage"] in evidence.SCORED_STAGES
    events = trial_ledger.read_all(ledger)
    assert len([e for e in events if e["event"] == "first_test"]) == 1
    outcomes = [e for e in events if e["event"] == "outcome"]
    assert len(outcomes) == 2 and outcomes[-1]["run_id"] == "run-1"
    assert trial_ledger.post_batch_floor(ledger, ["family-a"]) == 1


def test_score_batch_retest_with_a_different_window_raises(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    _pin_with_real_outcome(ledger, holdout=("2024-01-01", "2024-06-30"))
    with pytest.raises(RuntimeError, match="explicit retest window mismatch"):
        batch3.score_batch(
            [_item()], ledger_path=ledger, run_id="run-1", tested_date="2024-12-31",
            dry_run=False, retest_family_ids=frozenset({"family-a"}))
    assert len(trial_ledger.read_all(ledger)) == 2   # nothing appended


def test_score_batch_retest_set_does_not_loosen_other_families(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    _pin_with_real_outcome(ledger)
    with pytest.raises(RuntimeError, match="already pinned"):
        batch3.score_batch(
            [_item()], ledger_path=ledger, run_id="run-1", tested_date="2024-12-31",
            dry_run=False, retest_family_ids=frozenset({"some-other-family"}))


def test_score_batch_crash_after_an_explicit_retest_pin_leaves_the_real_outcome_intact(tmp_path, monkeypatch):
    """Guards that a scoring crash after an explicit --retest family passed
    the pin loop appends nothing for it. A defensive not_evaluated_error there
    would flip its latest genuine score into a defect deferral that a later
    DEFAULT run would re-score through the not_evaluated_error retry path
    (bypassing the --retest semantics). A family this run did not pin must get
    no defensive record at all."""
    ledger = str(tmp_path / "ledger.jsonl")
    _pin_with_real_outcome(ledger)
    before = trial_ledger.read_all(ledger)
    real_stage = trial_ledger.latest_outcome(ledger, "family-a")
    assert real_stage in evidence.SCORED_STAGES

    def boom(item, correction):
        raise RuntimeError("scoring exploded mid-batch")

    monkeypatch.setattr(batch3, "_score_one", boom)
    with pytest.raises(RuntimeError, match="scoring exploded"):
        batch3.score_batch(
            [_item()], ledger_path=ledger, run_id="run-1", tested_date="2024-12-31",
            dry_run=False, retest_family_ids=frozenset({"family-a"}))
    after = trial_ledger.read_all(ledger)
    assert after == before, "a crashed explicit retest must append nothing"
    assert trial_ledger.latest_outcome(ledger, "family-a") == real_stage


# ------------------------------------- pooled per-member exclusion
# A pooled family is a POOL: one unusable member must cost the pipeline that
# member, not the whole hypothesis. A large universe carries recent listings with no
# pre-holdout history; without per-member exclusion run_batch3_mining._require_bucket
# would raise and every universe-pool family would be skipped as compute_error.

def test_pooled_member_with_too_few_bars_in_a_bucket_is_excluded_not_the_family(
        tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled_late": MODULE_POOLED_LATE},
                      accepted=["b5_pooled_late"])
    item = _by_id(payload)["b5_pooled_late"]
    assert item["tickers"] == ["AAA", "BBB", "CCC", "RSK"]
    assert item["ticker"] == "AAA/BBB/CCC/RSK"
    assert item["pool_declared"] == ["AAA", "BBB", "CCC", "RSK", "LATE"]
    assert set(item["pool_excluded"]) == {"LATE"}
    assert "fewer than two bars" in item["pool_excluded"]["LATE"]
    assert "LATE" not in item["execution_costs"]
    assert item["stats_in_sample"]["ticker_count"] == 4
    assert item["stats_in_sample"]["trade_count"] > 0
    assert payload["pool_exclusions"]["b5_pooled_late"] == item["pool_excluded"]
    # Still ONE trial for the family, never one per surviving member.
    assert payload["shared_correction"]["post_batch_ledger_floor"] == 1
    assert item["stage"] in evidence.VALID_STAGES


def test_event_pooled_shape_excludes_members_the_same_way(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_event": MODULE_EVENT_POOLED},
                      accepted=["b5_event"])
    item = _by_id(payload)["b5_event"]
    assert item["tickers"] == ["AAA", "BBB", "CCC", "RSK"]
    assert set(item["pool_excluded"]) == {"LATE"}


@pytest.mark.parametrize("quotes_kwargs,needle", [
    ({"drop": ("BBB",)}, "no quote"),
    ({"suspect": ("BBB",)}, "suspect"),
    ({"wide": ("BBB",)}, "exceeds the 300 bps gate"),
])
def test_pooled_member_failing_the_quote_gate_is_excluded_not_the_family(
        tmp_path, monkeypatch, quotes_kwargs, needle):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled5": MODULE_POOLED5},
                      accepted=["b5_pooled5"], quotes_kwargs=quotes_kwargs)
    item = _by_id(payload)["b5_pooled5"]
    assert item["tickers"] == ["AAA", "CCC", "RSK", "DEF"]
    assert needle in item["pool_excluded"]["BBB"]
    assert "BBB" not in item["execution_costs"]
    assert item["stats_in_sample"]["ticker_count"] == 4
    assert _skips(payload) == {}


def test_pooled_member_whose_ohlc_fails_to_load_is_excluded_not_the_family(
        tmp_path, monkeypatch):
    def loader(symbol):
        if symbol == "CCC":
            raise RuntimeError("vendor returned no data for CCC")
        return _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})

    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled5": MODULE_POOLED5},
                      accepted=["b5_pooled5"], loader=loader)
    item = _by_id(payload)["b5_pooled5"]
    assert item["tickers"] == ["AAA", "BBB", "RSK", "DEF"]
    assert "vendor returned no data" in item["pool_excluded"]["CCC"]
    assert "CCC" in payload["input_errors"]


def test_pooled_family_with_every_member_excluded_is_skipped_as_missing_data(
        tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_pooled_dead": MODULE_POOLED_DEAD},
                          accepted=["b5_pooled_dead"])
    skip = _skips(payload)["b5_pooled_dead"]
    assert skip["reason_code"] == "missing_data"
    assert "every pool member excluded" in skip["reason"]
    assert "LATE" in skip["reason"] and "ONEBAR" in skip["reason"]
    assert payload["results"] == []
    assert not os.path.exists(paths["ledger"]), (
        "a family that could not be tested must charge no trial")


def test_a_single_shape_still_fails_the_whole_family_on_a_bad_member(
        tmp_path, monkeypatch):
    """Exclusion is a POOL rule: a single_long or rotation candidate has no
    pool to fall back on, so its one bad leg still skips the family."""
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_rotation": MODULE_ROTATION},
                      accepted=["b5_rotation"], quotes_kwargs={"drop": ("DEF",)})
    skip = _skips(payload)["b5_rotation"]
    assert skip["reason_code"] == "missing_data"
    assert "DEF" in skip["reason"]
    assert payload["results"] == []


def test_pool_exclusions_are_reported_in_the_run_report(tmp_path, monkeypatch):
    _, paths = _run(tmp_path, monkeypatch,
                    modules={"b5_pooled_late": MODULE_POOLED_LATE},
                    accepted=["b5_pooled_late"], dry_run=True)
    report = open(os.path.join(paths["run_dir"], "run-report-dry-run.md"),
                  encoding="utf-8").read()
    assert "Pooled families and their members" in report
    assert "LATE" in report
    assert "pool 4/5" in report
    assert "tested: AAA/BBB/CCC/RSK" in report


def test_a_pooled_family_with_no_exclusions_still_names_its_members(
        tmp_path, monkeypatch):
    """The table cell reads "pool KEPT/DECLARED" rather than the
    instruments, so without this a reader of the report alone could not
    reconstruct what a clean pooled family actually traded."""
    _, paths = _run(tmp_path, monkeypatch,
                    modules={"b5_pooled": MODULE_POOLED},
                    accepted=["b5_pooled"], dry_run=True)
    report = open(os.path.join(paths["run_dir"], "run-report-dry-run.md"),
                  encoding="utf-8").read()
    assert "Pooled families and their members" in report
    assert "tested: AAA/BBB/CCC" in report
    assert "excluded: none" in report


def test_the_quote_derived_cost_actually_reaches_the_engine(tmp_path, monkeypatch):
    """The per-member cost is selected inside the pooled loop, so it must
    be proven to arrive: a wider (still passing) quoted spread has to reduce
    the recorded net edge. Trade counts alone are invariant to cost."""
    edges = {}
    for label, spread in (("tight", 0.02), ("wide", 2.0)):
        work = tmp_path / label
        work.mkdir()
        payload, _ = _run(work, monkeypatch,
                          modules={"b5_pooled": MODULE_POOLED},
                          accepted=["b5_pooled"], dry_run=True,
                          quotes_kwargs={"spread": spread})
        item = _by_id(payload)["b5_pooled"]
        edges[label] = item["stats_in_sample"]["net_edge_bps"]
        assert item["execution_costs"]["AAA"]["cost_bps"] > 0.0
    assert edges["wide"] < edges["tight"], (
        "a wider quoted spread must reduce the recorded net edge")


# ------------------------------------------- all-blank-row hygiene
# The vendor's daily history for HUBB carries one all-blank 1977 session.
# batch3._normalize_ohlc refuses such a row outright, so load_frame drops an
# all-blank row (a missing print, not a bad one) and records it, rather than
# letting it cost a pooled family that member.

def _raw_with_row(frame, position, values):
    raw = frame.reset_index().rename(columns={"index": "Date"})
    for column, value in values.items():
        raw.loc[position, column] = value
    return raw


def test_load_frame_drops_an_all_blank_row_and_records_it(tmp_path):
    blank = {c: np.nan for c in ("Open", "High", "Low", "Close")}
    raw = _raw_with_row(_FRAMES["AAA"], 5, blank)
    dropped_date = pd.Timestamp(raw.loc[5, "Date"])
    report = {}
    frame, origin = batch5.load_frame(
        "AAA", ohlc_loader=lambda _s: raw, cache_dir=str(tmp_path), report=report)
    assert origin == "fetched"
    assert len(frame) == len(_FRAMES["AAA"]) - 1
    assert dropped_date not in frame.index
    assert report["dropped_all_nan_rows"] == [dropped_date.strftime("%Y-%m-%d")]


def test_load_frame_still_refuses_a_partly_blank_row(tmp_path):
    """A row with SOME prices is a bad print, not a missing session, and must
    keep failing closed."""
    raw = _raw_with_row(_FRAMES["AAA"], 5, {"Close": np.nan})
    with pytest.raises(ValueError, match="finite and positive"):
        batch5.load_frame("AAA", ohlc_loader=lambda _s: raw,
                          cache_dir=str(tmp_path))


def test_load_frame_require_positive_false_allows_a_negative_close_on_fetch(tmp_path):
    """A signal-only symbol (e.g. ^IRX, which had real negative closes in
    March 2020) must be loadable via require_positive=False. Default
    (require_positive=True, every existing call site) is unchanged -- see
    the test above."""
    raw = _raw_with_row(_FRAMES["AAA"], 5, {"Close": -0.05, "Open": -0.04})
    frame, origin = batch5.load_frame(
        "^IRX", ohlc_loader=lambda _s: raw, cache_dir=str(tmp_path),
        require_positive=False)
    assert origin == "fetched"
    assert frame["Close"].iloc[5] == -0.05


def test_load_frame_require_positive_false_allows_a_negative_close_on_cache_hit(tmp_path):
    """The cache-hit path re-normalizes via the same _normalize_ohlc call --
    require_positive must reach it there too, not just on first fetch."""
    raw = _raw_with_row(_FRAMES["AAA"], 5, {"Close": -0.05, "Open": -0.04})
    batch5.load_frame("^IRX", ohlc_loader=lambda _s: raw,
                      cache_dir=str(tmp_path), require_positive=False)
    # Second call hits the on-disk cache written by the first.
    frame, origin = batch5.load_frame(
        "^IRX", ohlc_loader=lambda _s: (_ for _ in ()).throw(
            AssertionError("should not re-fetch on a cache hit")),
        cache_dir=str(tmp_path), require_positive=False)
    assert origin == "cache"
    assert frame["Close"].iloc[5] == -0.05


def test_dropped_rows_surface_in_input_coverage(tmp_path, monkeypatch):
    def loader(symbol):
        raw = _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})
        if symbol == "AAA":
            for column in ("Open", "High", "Low", "Close"):
                raw.loc[5, column] = np.nan
        return raw

    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True, loader=loader)
    assert payload["input_coverage"]["AAA"]["dropped_all_nan_rows"]
    assert "dropped_all_nan_rows" not in payload["input_coverage"]["SPY"]


# --------------------------------------------- synthetic-quote guard
# The dry run is exercised on a synthetic quote file so every candidate
# clears the cost gate. A REAL run priced off invented spreads would record a
# pre-registered verdict against a cost that was never measured.

_SYNTHETIC = {"source": ("SYNTHETIC-DRY-RUN-ONLY: flat spread centred on the "
                         "last close"),
              "market_state": "synthetic (dry-run only)"}


def test_a_real_run_refuses_a_synthetic_quotes_file(tmp_path, monkeypatch):
    excinfo, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": MODULE_SINGLE},
                          accepted=["b5_single"],
                          quotes_kwargs=_SYNTHETIC, expect_error=ValueError)
    assert "SYNTHETIC" in str(excinfo.value)
    assert not os.path.exists(paths["ledger"])


# ------------------------------- a pooled SURVIVOR must be writable
# render_briefs feeds item["ticker"] into the brief FILENAME. A label that
# joined every member of a pool of several hundred would exceed the
# filesystem's filename limit, so write_run_outputs would raise OSError --
# AFTER score_batch has already pinned first_test and the survivor outcome in
# the ledger, leaving a re-run refusing the family as already pinned. Only a
# SURVIVOR gets a brief, so compute_candidate bounds the label through
# pool_ticker_label for every pool.

def _long_pool(n: int) -> list:
    return [f"SYM{i:03d}" for i in range(n)]


def test_a_many_member_pool_gets_a_label_short_enough_for_a_filename():
    label = batch5.pool_ticker_label(_long_pool(600))
    assert len(label) <= batch5.POOL_LABEL_MAX_CHARS
    assert "600" in label
    filename = reporting.brief_filename("a_family_id_of_realistic_length",
                                        batch5._brief_ticker_slug(label))
    assert len(filename) < 255


def test_a_short_pool_keeps_its_readable_label():
    assert batch5.pool_ticker_label(["AAA", "BBB", "CCC"]) == "AAA/BBB/CCC"
    assert batch5.pool_ticker_label(["AAA"]) == "AAA"


# ---------------- further pooled-exclusion guards
# A pooled family must not score a full pre-registered verdict on a small
# fraction of its declared pool; a member whose own BACKTEST raises must be
# dropped like any other unusable member; and exclusions must not vanish.

MODULE_POOLED_RAISER = _HEADER + """
FAMILY_ID = "b5_pooled_raiser"
SHAPE = "pooled_single_asset"
TRADED = ["AAA", "BBB", "CCC", "RSK", "DEF"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 150

def make_signal_fn(ticker, index, frames):
    if ticker == "CCC":
        raise RuntimeError("this one member's signal cannot be built")
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
"""


def test_a_pool_member_whose_backtest_raises_is_excluded_not_the_family(
        tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled_raiser": MODULE_POOLED_RAISER},
                      accepted=["b5_pooled_raiser"])
    item = _by_id(payload)["b5_pooled_raiser"]
    assert "CCC" not in item["tickers"]
    assert len(item["tickers"]) == 4
    assert "cannot be built" in item["pool_excluded"]["CCC"]
    assert item["pool_excluded"]["CCC"].startswith("backtest:")
    assert "CCC" not in item["execution_costs"], (
        "never claim a cost for an instrument this run did not trade")
    assert _skips(payload) == {}


def test_a_single_shape_member_whose_backtest_raises_still_fails_the_family(
        tmp_path, monkeypatch):
    """Isolation is a POOL rule. A single-instrument candidate has nothing to
    fall back on, so its failure must stay a whole-family compute_error."""
    raiser = MODULE_SINGLE.replace(
        "def make_signal_fn(ticker, index, frames):\n    assert ticker in frames",
        "def make_signal_fn(ticker, index, frames):\n"
        "    raise RuntimeError('signal cannot be built')")
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": raiser}, accepted=["b5_single"])
    skip = _skips(payload)["b5_single"]
    assert skip["reason_code"] == "compute_error"
    assert "signal cannot be built" in skip["reason"]
    assert not os.path.exists(paths["ledger"])


def test_a_pool_reduced_below_the_coverage_floor_is_refused(tmp_path, monkeypatch):
    """Scoring 2 of 5 members would record a pre-registered verdict for a
    materially narrower hypothesis than the one registered."""
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_pooled_raiser": MODULE_POOLED_RAISER},
                          accepted=["b5_pooled_raiser"],
                          quotes_kwargs={"drop": ("BBB", "RSK")})
    skip = _skips(payload)["b5_pooled_raiser"]
    assert skip["reason_code"] == "pool_coverage"
    assert "below the 80% floor" in skip["reason"]
    assert "BBB" in skip["reason"] and "RSK" in skip["reason"]
    assert payload["results"] == []
    assert not os.path.exists(paths["ledger"]), (
        "a refused family must charge no trial and stay eligible")


def test_the_coverage_floor_admits_a_pool_that_keeps_most_members(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled_thin": MODULE_POOLED_THIN},
                      accepted=["b5_pooled_thin"])
    # 2 of 4 kept is 50%: below the floor, so this must be refused.
    assert _skips(payload)["b5_pooled_thin"]["reason_code"] == "pool_coverage"
    assert batch5._pool_coverage_ok(["A", "B", "C", "D"], ["A", "B", "C", "D", "E"])
    assert not batch5._pool_coverage_ok(["A", "B", "C"], ["A", "B", "C", "D", "E"])
    assert not batch5._pool_coverage_ok([], [])


def test_exclusions_survive_a_whole_family_compute_error(tmp_path, monkeypatch):
    """pool_exclusions is built from SCORED items, so without carrying them
    into the skip reason an excluded member vanishes from every output."""
    every_raiser = MODULE_POOLED_RAISER.replace(
        'if ticker == "CCC":', 'if ticker != "ZZZ":')
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled_raiser": every_raiser},
                      accepted=["b5_pooled_raiser"],
                      quotes_kwargs={"drop": ("BBB",)})
    skip = _skips(payload)["b5_pooled_raiser"]
    assert skip["reason_code"] == "compute_error"
    assert "already excluded" in skip["reason"]
    assert "BBB" in skip["reason"]


def test_the_holdout_half_of_the_two_bar_rule_is_load_bearing(tmp_path, monkeypatch):
    """Pins the HOLDOUT half specifically: LATE and ONEBAR trip only the
    in-sample half (ONEBAR has one in-sample bar), so this fixture -- a full
    in-sample history and only one HOLDOUT bar, which the engine would also
    reject -- is the case that fails if just `or n_hold < 2` is dropped."""
    full = _FRAMES["CCC"]
    is_mask, _ = batch3._window_masks(full.index, HOLDOUT_START, TESTED_DATE)
    # Full in-sample history, then a single holdout bar that IS the required
    # session -- so the data-ends-before-required check cannot fire and the
    # ONLY thing that can exclude this member is the holdout bar count.
    one_holdout_bar = pd.concat([full.loc[is_mask], full.tail(1)]).copy()
    assert len(one_holdout_bar.loc[~batch3._window_masks(
        one_holdout_bar.index, HOLDOUT_START, TESTED_DATE)[0]]) == 1
    assert one_holdout_bar.index.max() == full.index.max()

    def loader(symbol):
        frame = one_holdout_bar if symbol == "CCC" else _FRAMES[symbol]
        return frame.reset_index().rename(columns={"index": "Date"})

    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled5": MODULE_POOLED5},
                      accepted=["b5_pooled5"], loader=loader)
    assert _skips(payload) == {}, "the family itself must still score"
    item = _by_id(payload)["b5_pooled5"]
    assert "CCC" not in item["tickers"], (
        "a member with fewer than two HOLDOUT bars must be excluded")
    assert "holdout 1" in item["pool_excluded"]["CCC"]
    assert "fewer than two bars" in item["pool_excluded"]["CCC"]


@pytest.mark.parametrize("payload_extra", [
    {"notes": ["generated SYNTHETICALLY for a dry run"]},
    {"provenance": "SYNTHETIC sample"},
    {"quotes_note": {"detail": "synthetic"}},
])
def test_the_synthetic_marker_is_refused_wherever_it_appears(tmp_path, payload_extra):
    """Every top-level field is searched, not only a few named ones, so a
    marker in notes or any other key is refused too."""
    base = {"sampled_at_local": "2026-09-08T10:45:00-04:00",
            "source": "a real capture", "market_state": "open",
            "quotes": {"SPY": _quote(100.0, 100.02)}}
    with pytest.raises(ValueError, match="SYNTHETIC"):
        batch5.refuse_synthetic_quotes({**base, **payload_extra}, dry_run=False)
    batch5.refuse_synthetic_quotes({**base, **payload_extra}, dry_run=True)


def test_a_measured_capture_is_not_refused(tmp_path):
    base = {"sampled_at_local": "2026-09-08T10:45:00-04:00",
            "source": "broker quote capture via a builder script",
            "market_state": "open (intraday capture)",
            "notes": ["spread computed from bid and ask", "suspect rules"],
            "quotes": {"SPY": _quote(100.0, 100.02)}}
    batch5.refuse_synthetic_quotes(base, dry_run=False)


def test_a_dropped_blank_row_is_still_reported_on_a_cache_hit(tmp_path):
    """Without the sidecar the fact is lost the moment the cache is warm --
    which is exactly the state the real run reads."""
    blank = {c: np.nan for c in ("Open", "High", "Low", "Close")}
    raw = _raw_with_row(_FRAMES["AAA"], 5, blank)
    first = {}
    batch5.load_frame("AAA", ohlc_loader=lambda _s: raw,
                      cache_dir=str(tmp_path), report=first)
    assert first["dropped_all_nan_rows"]

    def must_not_fetch(_symbol):
        raise AssertionError("the cache should have been used")

    second = {}
    frame, origin = batch5.load_frame("AAA", ohlc_loader=must_not_fetch,
                                      cache_dir=str(tmp_path), report=second)
    assert origin == "cache"
    assert second["dropped_all_nan_rows"] == first["dropped_all_nan_rows"]


MODULE_POOLED_DIAG = MODULE_POOLED.replace(
    '"b5_pooled"', '"b5_pooled_diag"').replace(
    "def make_signal_fn(ticker, index, frames):",
    """_SINK = {}
_RESETS = []

def reset_diagnostics():
    _SINK.clear()
    _RESETS.append(1)

def pooled_diagnostics():
    return {"members": len(_SINK), "totals": {"seen": sum(_SINK.values())},
            "resets": len(_RESETS)}

def make_signal_fn(ticker, index, frames):
    _SINK[ticker] = _SINK.get(ticker, 0) + 1""")


def test_a_modules_own_hygiene_counters_reach_the_run_output(tmp_path, monkeypatch):
    """A module that drops events for data-hygiene reasons must have those
    decisions land in the output: they shape the scored sample, and computing
    them only to discard them makes the module's disclosure unverifiable."""
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled_diag": MODULE_POOLED_DIAG},
                      accepted=["b5_pooled_diag"], dry_run=True)
    diag = _by_id(payload)["b5_pooled_diag"]["event_diagnostics"]
    assert diag["members"] == 3
    assert diag["totals"]["seen"] > 0
    assert diag["resets"] == 1, "the sink must be reset exactly once per family"


def test_a_module_without_counters_is_unaffected(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled": MODULE_POOLED},
                      accepted=["b5_pooled"], dry_run=True)
    assert "event_diagnostics" not in _by_id(payload)["b5_pooled"]


def _survived_pooled_item(pool: list) -> dict:
    """A minimal passing (survivor-stage) pooled item in the shape render_briefs consumes.
    Built directly because manufacturing a real survivor through the CLI
    would require clearing the pre-registered bar, which no synthetic
    fixture does."""
    return {
        "family_id": "b5_pool_survivor", "shape": "pooled_single_asset",
        "ticker": batch5.pool_ticker_label(pool), "tickers": list(pool),
        "pool_declared": list(pool), "pool_excluded": {},
        "stage": "survived", "dsr_value": 0.97, "reason": "cleared the bar",
        "holdout_start": "2026-03-04", "holdout_end": "2026-09-04",
        "in_sample_start": "2019-01-02", "in_sample_end": "2026-03-03",
        "holdout_net_edge_bps": 12.5, "account_type": "taxable",
        "tax_haircut_rate": 0.12,
        "stats_in_sample": {"trade_count": 900, "net_edge_bps": 20.0,
                            "sharpe": 1.4, "skew": 0.1, "kurtosis": 3.2},
        "stats_holdout": {"trade_count": 120, "net_edge_bps": 12.5,
                          "sharpe": 1.1, "skew": 0.0, "kurtosis": 3.0},
        "directional_panel": None,
        "catalog_parameters": {"lookback_months": 12},
        "execution_costs": {pool[0]: {"cost_bps": 3.0, "source": "test-quotes"}},
    }


def test_a_pooled_survivor_writes_its_brief_instead_of_crashing(tmp_path):
    """A surviving many-member pooled family must produce a brief ON DISK
    rather than raise OSError from an over-long filename after score_batch
    has already pinned the survivor."""
    pool = _long_pool(600)
    item = _survived_pooled_item(pool)
    catalog_by_id = {"b5_pool_survivor": {
        "family_id": "b5_pool_survivor", "name": "Pooled survivor",
        "mechanism": "synthetic mechanism", "account_types": ["taxable", "roth"],
        "tickers": ["expanded_universe"], "parameters": {}, "status": "eligible"}}
    correction = {"n_trials": 50.0, "n_this": 10.0,
                  "post_batch_ledger_floor": 50, "sr_variance": 0.1}
    briefs = batch5.render_briefs([item], catalog_by_id, correction)
    assert len(briefs) == 1
    name = next(iter(briefs))
    assert len(name) < 255, f"brief filename is {len(name)} chars"
    written = reporting.write_run_outputs(str(tmp_path), "# report\n", briefs)
    assert any("candidate-brief" in os.path.basename(p) for p in written)
    for path in written:
        assert os.path.exists(path)


def test_the_ledger_label_for_a_many_member_pool_stays_readable(tmp_path):
    """The bounded label is what reaches the append-only ledger's
    ticker_label, so a pooled row stays legible forever."""
    label = batch5.pool_ticker_label(_long_pool(600))
    assert len(label) <= batch5.POOL_LABEL_MAX_CHARS
    assert "/" not in label


def test_a_dry_run_accepts_a_synthetic_quotes_file(tmp_path, monkeypatch):
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True,
                      quotes_kwargs=_SYNTHETIC)
    assert _by_id(payload)["b5_single"]["stage"] in evidence.VALID_STAGES
    assert payload["provenance"]["quotes_market_state"] == _SYNTHETIC["market_state"]


# ------------------------------------- panel_breadth SNAPSHOTS verification
# A panel_breadth candidate's ONLY signal input is a sha-pinned breadth
# snapshot, so discovery must actually LOAD every SNAPSHOTS entry through
# mining.panel.load_breadth_snapshot(expect_sha256=...) -- a pin that is
# merely copied into the spec, never verified, would let a rebuilt snapshot
# silently change what an already-pinned family measures.

_SNAP_DATES = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
_SNAP_VALUES = [0.10, -0.20, 0.35, 0.05]
_SNAP_INDICATOR = "net_advance_ratio"
_SNAP_PARAMETERS = {"min_members": 9}
_SNAP_PROVENANCE = {
    "basis_id": "tiny-test-basis",
    "point_in_time": True,
    "ledgerable": True,
    "built_at": "2026-09-04T00:00:00Z",
    "seasoning_rule": "test fixture",
    "broken_sessions": [],
    "omitted_sessions": [],
}


def _snapshot_sha(dates, values, *, indicator=_SNAP_INDICATOR,
                  parameters=None, provenance=None) -> str:
    """The content sha the file's own bytes imply, recomputed here rather than
    read back out of the file -- so a test asserting the pin is verified is not
    asserting it against the very field under test."""
    from mining import panel as _panel
    return _panel._canonical_sha256(
        schema_version=_panel.SNAPSHOT_SCHEMA_VERSION, indicator=indicator,
        parameters=_SNAP_PARAMETERS if parameters is None else parameters,
        dates=list(dates), values=list(values),
        provenance=_SNAP_PROVENANCE if provenance is None else provenance)


def _write_snapshot(tmp_path, filename="tiny_breadth.json", *,
                    dates=None, values=None) -> tuple:
    """A tiny but genuinely valid breadth snapshot in tmp; returns
    ``(path, content_sha256)``."""
    from mining import panel as _panel
    dates = list(_SNAP_DATES if dates is None else dates)
    values = list(_SNAP_VALUES if values is None else values)
    sha = _snapshot_sha(dates, values)
    payload = {"schema_version": _panel.SNAPSHOT_SCHEMA_VERSION,
               "indicator": _SNAP_INDICATOR, "parameters": _SNAP_PARAMETERS,
               "dates": dates, "values": values, "content_sha256": sha,
               "provenance": _SNAP_PROVENANCE}
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path), sha


_SNAPSHOT_KEY = "tiny_breadth_m9"


def _panel_module(path, sha, *, family_id="b5_panel") -> str:
    """A panel_breadth candidate module pinning ``path`` at ``sha``."""
    return _HEADER + """
FAMILY_ID = {family_id}
SHAPE = "panel_breadth"
TRADED = ["AAA"]
SIGNAL = []
SNAPSHOTS = {{{key}: {{"path": {path}, "sha256": {sha}}}}}
BENCHMARK = "SPY"
PARAMETERS = {{"period": 15, "hold": 7}}
EXPECTED_MIN_TRADES = 50

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=(i % 15 == 0), exit=(i % 15 == 7))
    return signal
""".format(family_id=json.dumps(family_id), key=json.dumps(_SNAPSHOT_KEY),
           path=json.dumps(path), sha=json.dumps(sha))


def test_panel_breadth_snapshot_is_loaded_and_verified_against_its_pinned_sha(
        tmp_path):
    snapshot_path, sha = _write_snapshot(tmp_path)
    modules_dir = _write_modules(
        tmp_path, {"b5_panel": _panel_module(snapshot_path, sha)})
    # An explicit repo root that does not contain tmp_path keeps the snapshot
    # outside the repo even when TEMP itself sits inside the checkout (CI).
    found, problems = batch5.discover_modules(
        modules_dir, repo_root=str(tmp_path / "repo"))
    assert problems == []
    assert set(found) == {"b5_panel"}
    summary = found["b5_panel"]["spec"]["snapshots"]
    assert set(summary) == {_SNAPSHOT_KEY}
    entry = summary[_SNAPSHOT_KEY]
    # The recorded sha is the one the CONTENT implies, recomputed here from the
    # dates/values/provenance rather than read out of the file.
    assert entry["sha256"] == _snapshot_sha(_SNAP_DATES, _SNAP_VALUES)
    assert entry["content_sha256"] == entry["sha256"]
    # describe() fields, each recomputed from the fixture's own inputs.
    assert entry["indicator"] == _SNAP_INDICATOR
    assert entry["n"] == len(_SNAP_VALUES)
    assert entry["first_date"] == min(_SNAP_DATES)
    assert entry["last_date"] == max(_SNAP_DATES)
    # Outside the repo, so the recorded path keeps only the file name -- a
    # committed results JSON must not name the workstation it ran on.
    assert entry["path"] == "<external>/" + os.path.basename(snapshot_path)


def test_panel_breadth_snapshot_pinned_at_a_wrong_sha_is_a_problem_and_is_skipped(
        tmp_path):
    """The pin is what stops a rebuilt snapshot from silently changing an
    already-pinned family's measurement, so a pin the file does not match must
    keep the module OUT of the run."""
    snapshot_path, real_sha = _write_snapshot(tmp_path)
    wrong_sha = "0" * 64
    assert wrong_sha != real_sha
    modules_dir = _write_modules(
        tmp_path, {"b5_panel": _panel_module(snapshot_path, wrong_sha)})
    found, problems = batch5.discover_modules(modules_dir)
    assert found == {}, "a module whose snapshot pin fails must not be runnable"
    assert len(problems) == 1 and problems[0]["module"] == "b5_panel"
    reason = problems[0]["reason"]
    assert _SNAPSHOT_KEY in reason
    assert real_sha in reason and wrong_sha in reason


def test_panel_breadth_snapshot_whose_file_is_absent_is_a_problem_not_a_crash(
        tmp_path):
    missing = str(tmp_path / "not_written.json")
    assert not os.path.exists(missing)
    modules_dir = _write_modules(
        tmp_path, {"b5_panel": _panel_module(missing, "0" * 64)})
    found, problems = batch5.discover_modules(
        modules_dir, repo_root=str(tmp_path / "repo"))
    assert found == {}
    assert len(problems) == 1
    assert "does not exist" in problems[0]["reason"]
    # Outside the repo, so the recorded reason keeps only the file name -- a
    # committed results JSON must not name the workstation it ran on.
    assert "<external>/" + os.path.basename(missing) in problems[0]["reason"]
    assert workstation_leaks(problems, tmp_path) == []


# ------------------------------------------ every _CONTRACT_FIELDS check
# A contract check that nothing exercises can be deleted without a single
# test reddening -- which is exactly how an invalid candidate would reach
# the scorer. One case per check, asserted on the STRUCTURAL outcome (the
# module must not be runnable), not merely on the wording.

def _drop_field(text: str, field: str) -> str:
    """MODULE_SINGLE with ``field``'s assignment line removed entirely."""
    kept = [line for line in text.splitlines()
            if not line.startswith(f"{field} = ")]
    assert len(kept) == len(text.splitlines()) - 1, f"{field} line not found"
    return "\n".join(kept) + "\n"


@pytest.mark.parametrize("field", batch5._CONTRACT_FIELDS)
def test_a_missing_contract_field_is_reported_as_missing(tmp_path, field):
    modules_dir = _write_modules(
        tmp_path, {"b5_single": _drop_field(MODULE_SINGLE, field)})
    found, problems = batch5.discover_modules(modules_dir)
    assert found == {}
    assert len(problems) == 1 and problems[0]["module"] == "b5_single"
    assert f"missing {field}" in problems[0]["reason"]


@pytest.mark.parametrize("check,mutation,replacement", [
    ("FAMILY_ID type", 'FAMILY_ID = "b5_single"', "FAMILY_ID = 5"),
    ("FAMILY_ID empty", 'FAMILY_ID = "b5_single"', 'FAMILY_ID = ""'),
    ("BENCHMARK type", 'BENCHMARK = "SPY"', "BENCHMARK = 5"),
    ("BENCHMARK empty", 'BENCHMARK = "SPY"', 'BENCHMARK = ""'),
    ("SIGNAL not a list", "SIGNAL = []", 'SIGNAL = "SPY"'),
    ("SIGNAL not all str", "SIGNAL = []", 'SIGNAL = ["SPY", 5]'),
    ("SIGNAL repeats", "SIGNAL = []", 'SIGNAL = ["SPY", "SPY"]'),
    ("PARAMETERS not a dict", 'PARAMETERS = {"period": 15, "hold": 7}',
     "PARAMETERS = [15, 7]"),
    ("EXPECTED_MIN_TRADES negative", "EXPECTED_MIN_TRADES = 50",
     "EXPECTED_MIN_TRADES = -1"),
])
def test_each_contract_type_check_keeps_a_bad_module_out_of_the_run(
        tmp_path, check, mutation, replacement):
    text = MODULE_SINGLE.replace(mutation, replacement)
    assert text != MODULE_SINGLE, f"{check}: mutation did not apply"
    modules_dir = _write_modules(tmp_path, {"b5_single": text})
    found, problems = batch5.discover_modules(modules_dir)
    assert found == {}, f"{check}: an invalid module reached the run"
    assert len(problems) == 1 and problems[0]["module"] == "b5_single"
    field = check.split()[0]
    assert field in problems[0]["reason"]


def test_traded_and_signal_must_not_overlap(tmp_path):
    """A symbol on both legs means the signal is reading the very bars it
    trades; each leg is fetched and cost-gated differently, so the overlap is
    a contract error rather than a redundancy."""
    text = MODULE_SINGLE.replace("SIGNAL = []", 'SIGNAL = ["AAA"]')
    modules_dir = _write_modules(tmp_path, {"b5_single": text})
    found, problems = batch5.discover_modules(modules_dir)
    assert found == {}
    assert len(problems) == 1
    assert "TRADED and SIGNAL must not overlap" in problems[0]["reason"]
    # Control: the same module with the overlap removed IS runnable, so the
    # refusal above is the overlap and nothing else about the mutation.
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_dir = _write_modules(clean_root, {"b5_single": MODULE_SINGLE})
    clean_found, clean_problems = batch5.discover_modules(clean_dir)
    assert set(clean_found) == {"b5_single"} and clean_problems == []


def test_two_modules_claiming_one_family_id_keep_the_first_and_report_the_second(
        tmp_path):
    """Two modules for one family_id would make which implementation was
    scored depend on filename order, so the later one is refused by name."""
    modules_dir = _write_modules(tmp_path, {"b5_single": MODULE_SINGLE,
                                            "zz_copy": MODULE_SINGLE})
    found, problems = batch5.discover_modules(modules_dir)
    assert set(found) == {"b5_single"}
    # The FIRST module by discovery order is the one kept -- proven by which
    # file the retained spec came from, not by the absence of an error.
    assert found["b5_single"]["spec"]["module_file"].endswith("b5_single.py")
    assert len(problems) == 1
    assert problems[0]["module"] == "zz_copy"
    assert "duplicate FAMILY_ID b5_single" in problems[0]["reason"]


# --------------------------------------- validate_module_contract aggregates
# The contract promises EVERY problem in one message so an implementer fixes
# them in one pass; a first-problem-only validator would send them round the
# loop once per field.

MODULE_MANY_VIOLATIONS = _HEADER + """
FAMILY_ID = 5
SHAPE = "weird_shape"
TRADED = ["AAA"]
SIGNAL = ["AAA"]
SNAPSHOTS = 5
BENCHMARK = ""
PARAMETERS = [15, 7]
EXPECTED_MIN_TRADES = True

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=True, exit=False)
    return signal
"""


def test_validate_module_contract_names_every_violation_at_once(tmp_path):
    modules_dir = _write_modules(tmp_path, {"b5_many": MODULE_MANY_VIOLATIONS})
    module = batch5._import_module_file(
        "b5_many", os.path.join(modules_dir, "b5_many.py"))
    with pytest.raises(ValueError) as excinfo:
        batch5.validate_module_contract(module)
    message = str(excinfo.value)
    reported = message.split("; ")
    # Seven violations were planted, one per contract check the module trips.
    expected = ["FAMILY_ID", "SHAPE", "TRADED and SIGNAL", "SNAPSHOTS",
                "BENCHMARK", "PARAMETERS", "EXPECTED_MIN_TRADES"]
    for needle in expected:
        assert any(needle in problem for problem in reported), (
            f"{needle} was not named; got {reported}")
    assert len(reported) == len(expected), (
        f"expected exactly {len(expected)} problems, got {reported}")


# ================================================================
# failure isolation, input errors and provenance
# ================================================================
# A batch is many independent hypotheses sharing one pre-registered
# correction. One candidate blowing up, one vendor symbol missing, or one
# short series must cost exactly that candidate -- never the batch, and
# never a trial charge for a test that did not happen.

_RAISER_MESSAGE = "candidate exploded inside make_signal_fn"

# make_signal_fn itself raises, so the failure surfaces from INSIDE
# run_single_asset_windows (the factory is called per window bucket) --
# the same place a real candidate's indicator arithmetic would fail.
MODULE_RAISER = _HEADER + '''
FAMILY_ID = "b5_raiser"
SHAPE = "single_long"
TRADED = ["CCC"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"period": 15, "hold": 7}
EXPECTED_MIN_TRADES = 50

def make_signal_fn(ticker, index, frames):
    raise RuntimeError("''' + _RAISER_MESSAGE + '''")
'''

# A healthy single_long on CCC -- the contrast partner for a pooled family
# that also holds CCC: the pool drops the member, the single dies.
MODULE_SINGLE_CCC = (MODULE_SINGLE
                     .replace('"b5_single"', '"b5_single_ccc"')
                     .replace('TRADED = ["AAA"]', 'TRADED = ["CCC"]'))
# A pooled family with a SIGNAL-only leg: DEF carries the whole family, so
# it is never a pool member and its loss is fatal.
MODULE_POOLED_SIGNAL = (MODULE_POOLED
                        .replace('"b5_pooled"', '"b5_pooled_signal"')
                        .replace('TRADED = ["AAA", "BBB", "CCC"]',
                                 'TRADED = ["AAA", "BBB"]')
                        .replace('SIGNAL = []', 'SIGNAL = ["DEF"]'))
# A pooled family whose declared BENCHMARK is neither traded nor signal,
# so a BENCHMARK failure is separable from a market-benchmark failure.
MODULE_POOLED_BENCH = (MODULE_POOLED
                       .replace('"b5_pooled"', '"b5_pooled_bench"')
                       .replace('TRADED = ["AAA", "BBB", "CCC"]',
                                'TRADED = ["AAA", "BBB"]')
                       .replace('BENCHMARK = "SPY"', 'BENCHMARK = "CCC"'))


def _failing_loader(broken, message="vendor returned no data for"):
    """A loader that raises for ``broken`` and serves synthetic data for
    every other symbol."""
    def loader(symbol):
        if symbol == broken:
            raise RuntimeError(f"{message} {symbol}")
        return _FRAMES[symbol].reset_index().rename(columns={"index": "Date"})
    return loader


# -------------------------------------------- one candidate raising

def test_a_raising_candidate_is_isolated_and_charges_no_trial(
        tmp_path, monkeypatch):
    """A candidate whose signal construction raises must be skipped as
    compute_error while the rest of the batch scores -- and, because the
    skip happens BEFORE score_batch, it must leave no ledger row and add
    nothing to the multiple-testing charge."""
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": MODULE_SINGLE,
                                   "b5_raiser": MODULE_RAISER},
                          accepted=["b5_single", "b5_raiser"])
    # The healthy candidate is unharmed: still scored, still fully computed.
    assert set(_by_id(payload)) == {"b5_single"}
    healthy = _by_id(payload)["b5_single"]
    assert healthy["stage"] in evidence.VALID_STAGES
    assert healthy["stats_in_sample"]["trade_count"] >= 15
    assert healthy["directional_panel"] is not None
    # The raiser is skipped, naming the real exception type and message.
    skip = _skips(payload)["b5_raiser"]
    assert skip["reason_code"] == "compute_error"
    assert "RuntimeError" in skip["reason"]
    assert _RAISER_MESSAGE in skip["reason"]
    # Skipped BEFORE score_batch: score_batch is the only writer of ledger
    # rows, so the raiser having none proves it never reached it.
    events = trial_ledger.read_all(paths["ledger"])
    assert {event["family_id"] for event in events} == {"b5_single"}
    assert not trial_ledger.has_been_tested(paths["ledger"], "b5_raiser")
    assert trial_ledger.latest_outcome(paths["ledger"], "b5_raiser") is None
    # And it charged no trial: the ledger's all-time distinct-family count
    # (recomputed here with an EMPTY submission set, so it is purely what
    # the run recorded) is 1, and that is what the correction used.
    assert trial_ledger.post_batch_floor(paths["ledger"], []) == 1
    assert payload["shared_correction"]["post_batch_ledger_floor"] == 1
    assert payload["shared_correction"]["n_trials"] == 1.0
    assert healthy["shared_n_trials"] == 1.0


def test_a_raising_candidate_does_not_corrupt_a_pooled_neighbour(
        tmp_path, monkeypatch):
    """The neighbour's own numbers must be bit-identical to what it scores
    alone, so the raiser cannot leak state (frames, costs, per-ticker
    windows) into it."""
    alone = tmp_path / "alone"
    alone.mkdir()
    with_raiser = tmp_path / "with_raiser"
    with_raiser.mkdir()
    solo, _ = _run(alone, monkeypatch, modules={"b5_pooled": MODULE_POOLED},
                   accepted=["b5_pooled"], dry_run=True)
    both, _ = _run(with_raiser, monkeypatch,
                   modules={"b5_pooled": MODULE_POOLED,
                            "b5_raiser": MODULE_RAISER},
                   accepted=["b5_pooled", "b5_raiser"], dry_run=True)
    assert _skips(both)["b5_raiser"]["reason_code"] == "compute_error"
    solo_item, both_item = _by_id(solo)["b5_pooled"], _by_id(both)["b5_pooled"]
    for field in ("stats_in_sample", "stats_holdout", "dsr_value", "stage",
                  "tickers", "execution_costs", "holdout_net_edge_bps"):
        assert both_item[field] == solo_item[field], f"{field} was disturbed"


# ------------------------------------------- OHLC load failures
# Support symbols (signal-only, the declared BENCHMARK, and the market
# benchmark) carry the whole family, so losing one is fatal even for a
# pooled shape. A POOL MEMBER is only excluded.

_SUPPORT_MODULES = {"b5_pooled_signal": MODULE_POOLED_SIGNAL,
                    "b5_pooled_bench": MODULE_POOLED_BENCH}


@pytest.mark.parametrize("family_id,broken,role", [
    ("b5_pooled_signal", "DEF", "signal-only leg"),
    ("b5_pooled_bench", "CCC", "declared BENCHMARK"),
    ("b5_pooled_bench", "SPY", "market benchmark"),
])
def test_a_support_symbol_failing_to_load_skips_even_a_pooled_family(
        tmp_path, monkeypatch, family_id, broken, role):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={family_id: _SUPPORT_MODULES[family_id]},
                          accepted=[family_id],
                          loader=_failing_loader(broken))
    skip = _skips(payload)[family_id]
    assert skip["reason_code"] == "missing_data", role
    assert broken in skip["reason"]
    assert "vendor returned no data" in skip["reason"]
    assert payload["results"] == []
    # It is a FAMILY failure, not a per-member exclusion.
    assert payload["pool_exclusions"] == {}
    assert broken in payload["input_errors"]
    assert not os.path.exists(paths["ledger"]), (
        "a family that could not be tested must charge no trial")


def test_a_single_shape_member_failing_to_load_skips_the_family(
        tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": MODULE_SINGLE},
                          accepted=["b5_single"],
                          loader=_failing_loader("AAA"))
    skip = _skips(payload)["b5_single"]
    assert skip["reason_code"] == "missing_data"
    assert "AAA" in skip["reason"]
    assert "vendor returned no data" in skip["reason"]
    assert payload["results"] == []
    assert not os.path.exists(paths["ledger"])


def test_one_broken_symbol_excludes_a_pool_member_but_kills_a_single_family(
        tmp_path, monkeypatch):
    """The two halves of the split, proven in ONE batch off ONE broken
    symbol: CCC is a pool member of b5_pooled (dropped, family scores) and
    the only leg of b5_single_ccc (family skipped)."""
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_pooled5": MODULE_POOLED5,
                                   "b5_single_ccc": MODULE_SINGLE_CCC},
                          accepted=["b5_pooled5", "b5_single_ccc"],
                          loader=_failing_loader("CCC"))
    assert set(_by_id(payload)) == {"b5_pooled5"}
    pooled = _by_id(payload)["b5_pooled5"]
    assert pooled["tickers"] == ["AAA", "BBB", "RSK", "DEF"]
    assert pooled["pool_declared"] == ["AAA", "BBB", "CCC", "RSK", "DEF"]
    assert "vendor returned no data" in pooled["pool_excluded"]["CCC"]
    assert pooled["stats_in_sample"]["ticker_count"] == 4
    assert pooled["stage"] in evidence.VALID_STAGES
    skip = _skips(payload)["b5_single_ccc"]
    assert skip["reason_code"] == "missing_data"
    assert "CCC" in skip["reason"]
    # Exactly one family paid a trial: the pooled one that actually ran.
    assert {event["family_id"] for event in
            trial_ledger.read_all(paths["ledger"])} == {"b5_pooled5"}
    assert trial_ledger.post_batch_floor(paths["ledger"], []) == 1


# ------------------------------- data-coverage (required session) check

def _required_session() -> str:
    """The last real NYSE session at or before TESTED_DATE, recomputed here
    from the vendored calendar rather than read out of
    run_batch3_mining.derive_holdout_window (the thing under test)."""
    end = datetime.date(2026, 9, 4)
    return max(session for session in trading_calendar.load_default_sessions()
               if session <= end).isoformat()


def _truncating_loader(symbol, last_date):
    def loader(name):
        frame = _FRAMES[name]
        if name == symbol:
            frame = frame.loc[:last_date]
        return frame.reset_index().rename(columns={"index": "Date"})
    return loader


def test_data_ending_before_the_required_session_skips_the_candidate(
        tmp_path, monkeypatch):
    required = _required_session()
    sessions = [session.isoformat()
                for session in trading_calendar.load_default_sessions()]
    previous = sessions[sessions.index(required) - 1]
    assert previous < required
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_single": MODULE_SINGLE},
                          accepted=["b5_single"],
                          loader=_truncating_loader("AAA", previous))
    skip = _skips(payload)["b5_single"]
    assert skip["reason_code"] == "missing_data"
    assert "AAA" in skip["reason"], "the short symbol must be named"
    # Both dates are recomputed above, not read back out of the driver.
    assert previous in skip["reason"] and required in skip["reason"]
    assert payload["input_coverage"]["AAA"]["last"] == previous
    assert payload["results"] == []
    assert not os.path.exists(paths["ledger"])


def test_data_ending_exactly_on_the_required_session_is_accepted(
        tmp_path, monkeypatch):
    """The boundary is inclusive: one session short is fatal (above), the
    required session itself is fine."""
    required = _required_session()
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True,
                      loader=_truncating_loader("AAA", required))
    assert payload["input_errors"] == {}
    assert payload["input_coverage"]["AAA"]["last"] == required
    item = _by_id(payload)["b5_single"]
    assert item["data_through"]["AAA"] == required
    assert item["stage"] in evidence.VALID_STAGES


# ------------------------------- market-benchmark quote cross-check

def test_provenance_pins_the_market_benchmark_quote_actually_used(
        tmp_path, monkeypatch):
    """SPY is the panel's market benchmark and is not traded here, so its
    quote passes no gate -- it exists only as a recorded cross-check that
    the quote sample really covered the benchmark. Given a DELIBERATELY
    distinct SPY quote, the recorded value must be SPY's and no other
    symbol's."""
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True,
                      quotes_kwargs={"wide": ("SPY",)})
    cross = payload["provenance"]["spy_quote_cross_check"]
    assert cross is not None
    # Recomputed from the fixture's own bid/ask, not read back off the field.
    assert cross["bid"] == 90.0 and cross["ask"] == 110.0
    assert cross["price"] == pytest.approx((90.0 + 110.0) / 2)
    assert cross["spread_bps"] == pytest.approx(
        (110.0 - 90.0) / ((110.0 + 90.0) / 2) * 1e4)
    assert cross["security_status"] == "Normal"
    assert cross["suspect"] is False
    # It is the MARKET BENCHMARK's quote, not just any symbol's: the traded
    # leg's own quote is a different book entirely.
    traded = _by_id(payload)["b5_single"]["execution_costs"]["AAA"]
    assert traded["bid"] == 100.0 and cross["bid"] != traded["bid"]
    assert cross["spread_bps"] > traded["spread_bps"] * 100
    # And it is exactly what the consumed quotes file holds for SPY.
    with open(str(tmp_path / "quotes.json"), "r", encoding="utf-8") as handle:
        consumed = json.load(handle)["quotes"]["SPY"]
    assert cross == consumed


def test_provenance_market_benchmark_quote_is_none_when_the_file_omits_spy(
        tmp_path, monkeypatch):
    """SPY needs no quote (never traded), so a file without it still runs --
    and the cross-check must say so rather than borrow another symbol."""
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True,
                      quotes_kwargs={"drop": ("SPY",)})
    assert _by_id(payload)["b5_single"]["stage"] in evidence.VALID_STAGES
    assert payload["provenance"]["spy_quote_cross_check"] is None


# ================================================================
# directional-support panel ARITHMETIC
# ================================================================
# The panel is report-only, but it is the evidence a decision turns
# on, so its arithmetic has to be pinned by INDEPENDENT RECOMPUTATION --
# never by asserting that a note or a label is present. A presence check
# alone would stay green if the pooled aggregation switched from mean to
# SUM, the batch variance were multiplied by seven or swapped for the floor
# constant, n_trials=1.0 were passed into the DSR, a constant were added to
# the in-sample Sharpe inside the panel pass, the time-matched-excess
# conjunct were dropped, or the t threshold were lowered to a huge negative
# number.

_MEAN_POOL = ("AAA", "BBB", "MID")
MODULE_POOLED_MEAN = (MODULE_POOLED
                      .replace('"b5_pooled"', '"b5_pooled_mean"')
                      .replace('TRADED = ["AAA", "BBB", "CCC"]',
                               'TRADED = ["AAA", "BBB", "MID"]'))


def _factory_15_7():
    """The 15/7 signal MODULE_POOLED implements, rebuilt here so the test's
    own reconstruction never imports the candidate module under test."""
    def factory(_index):
        def signal(frame, i):
            return batch3.backtest_engine.SignalResult(
                entry=(i % 15 == 0), exit=(i % 15 == 7))
        return signal
    return factory


def _in_sample_bucket(symbol: str) -> pd.DataFrame:
    frame = _FRAMES[symbol]
    return frame.loc[frame.index < pd.Timestamp(HOLDOUT_START)]


def _sleeve(symbol: str, cost_bps: float, factory=None) -> pd.Series:
    """One per-ticker in-sample mark-to-market sleeve, rebuilt from the
    engine's own trade list -- not read out of the driver."""
    windows = batch3.run_single_asset_windows(
        _FRAMES[symbol], factory or _factory_15_7(),
        holdout_start=HOLDOUT_START, holdout_end=TESTED_DATE,
        cost_bps=cost_bps, tax_haircut_rate=0.0)
    return benchmark_relative.mark_to_market_daily_returns(
        _in_sample_bucket(symbol), windows["in_sample_trades"],
        cost_bps=cost_bps)


def _equal_weight_mean(series_by_symbol) -> pd.Series:
    """Equal-weight MEAN on the union calendar with 0.0 fill, computed with
    a pandas DataFrame reindex rather than batch5._aligned, so the expected
    value does not come from the code under test."""
    union = pd.DatetimeIndex([])
    for series in series_by_symbol.values():
        union = union.union(pd.DatetimeIndex(series.index))
    union = union.sort_values()
    wide = pd.DataFrame({symbol: series.reindex(union).fillna(0.0)
                         for symbol, series in series_by_symbol.items()},
                        index=union)
    return wide.sum(axis=1) / float(len(series_by_symbol))


def test_pooled_panel_is_the_equal_weight_mean_of_the_per_ticker_sleeves(
        tmp_path, monkeypatch):
    """The pooled strategy series must be the equal-weight MEAN of the
    per-ticker sleeves, and own buy-and-hold the equal-weight mean of the
    pool -- recomputed here from the sleeves, because Sharpe, t_alpha and r2
    are all scale-invariant and would not notice a SUM at all."""
    payload, _ = _run(tmp_path, monkeypatch,
                      modules={"b5_pooled_mean": MODULE_POOLED_MEAN},
                      accepted=["b5_pooled_mean"], dry_run=True)
    item = _by_id(payload)["b5_pooled_mean"]
    assert item["tickers"] == list(_MEAN_POOL), "no member may be excluded here"
    panel_ = item["directional_panel"]

    costs = {symbol: item["execution_costs"][symbol]["cost_bps"]
             for symbol in _MEAN_POOL}
    sleeves = {symbol: _sleeve(symbol, costs[symbol]) for symbol in _MEAN_POOL}
    expected_strategy = _equal_weight_mean(sleeves)
    expected_own_bh = _equal_weight_mean(
        {symbol: benchmark_relative.buy_and_hold_daily_returns(
            _in_sample_bucket(symbol)) for symbol in _MEAN_POOL})
    # The union calendar is genuinely a union: MID lists later, so its own
    # sleeve is shorter than the pooled series it is averaged into.
    assert len(_in_sample_bucket("MID")) < len(expected_strategy)
    assert panel_["strategy_daily_days"] == len(expected_strategy)
    assert panel_["own_benchmark"] == list(_MEAN_POOL)

    # beta and alpha are the two panel figures a scale change moves, so they
    # are the assertion that separates a MEAN from a SUM.
    expected_ols = benchmark_relative.ols_alpha_beta(
        expected_strategy, expected_own_bh)
    for key, value in expected_ols.items():
        assert panel_["ols_vs_own"][key] == pytest.approx(
            value, rel=1e-9, abs=1e-15), key

    # Each side pinned separately: strategy_cagr/max_dd depend only on the
    # strategy series, benchmark_cagr/max_dd only on own buy-and-hold, so a
    # mean-vs-sum error on either one alone is caught.
    full = panel_["full_window_vs_own"]
    for key, series in (("strategy", expected_strategy),
                        ("benchmark", expected_own_bh)):
        assert full[key + "_cagr"] == pytest.approx(
            benchmark_relative.cagr(series), rel=1e-9), key
        assert full[key + "_max_dd"] == pytest.approx(
            benchmark_relative.max_drawdown(series), rel=1e-9), key
        assert full[key + "_sharpe_ann"] == pytest.approx(
            benchmark_relative.sharpe_annualized(series), rel=1e-9), key
    assert panel_["daily_sharpe_annualized"] == pytest.approx(
        benchmark_relative.sharpe_annualized(expected_strategy), rel=1e-9)
    assert panel_["daily_moments"]["sr"] == pytest.approx(
        benchmark_relative.daily_sharpe_moments(expected_strategy)["sr"],
        rel=1e-9)

    # And the divisor really bites: the SUM of the same sleeves is a
    # measurably different series on every one of those figures.
    summed = expected_strategy * float(len(_MEAN_POOL))
    assert benchmark_relative.cagr(summed) != pytest.approx(
        benchmark_relative.cagr(expected_strategy), rel=1e-6)
    assert benchmark_relative.ols_alpha_beta(summed, expected_own_bh)["beta"] \
        != pytest.approx(expected_ols["beta"], rel=1e-6)


def test_single_ticker_panel_uses_that_ticker_sleeve_unscaled(
        tmp_path, monkeypatch):
    """A one-ticker family is the degenerate case of the same rule: the
    strategy series is that sleeve itself, divided by nothing."""
    payload, _ = _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True)
    item = _by_id(payload)["b5_single"]
    panel_ = item["directional_panel"]
    sleeve = _sleeve("AAA", item["execution_costs"]["AAA"]["cost_bps"])
    own_bh = benchmark_relative.buy_and_hold_daily_returns(
        _in_sample_bucket("AAA"))
    assert panel_["strategy_daily_days"] == len(sleeve)
    expected_ols = benchmark_relative.ols_alpha_beta(sleeve, own_bh)
    for key, value in expected_ols.items():
        assert panel_["ols_vs_own"][key] == pytest.approx(
            value, rel=1e-9, abs=1e-15), key
    assert panel_["full_window_vs_own"]["strategy_cagr"] == pytest.approx(
        benchmark_relative.cagr(sleeve), rel=1e-9)


# --------------------------------------------------- daily-DSR arithmetic

def test_daily_dsr_is_the_dsr_formula_on_the_shared_n_and_the_batch_variance(
        tmp_path, monkeypatch):
    """The daily-frequency DSR is reported beside the pre-registered
    per-trade DSR, so the number has to be right, not merely a float.
    Recomputed here by calling mining.dsr directly with the SHARED
    correction N and the batch daily-Sharpe variance."""
    modules = {"b5_single": MODULE_SINGLE, "b5_always": MODULE_ALWAYS,
               "b5_pooled": MODULE_POOLED}
    payload, _ = _run(tmp_path, monkeypatch, modules=modules,
                      accepted=sorted(modules), dry_run=True)
    items = _by_id(payload)
    assert set(items) == set(modules)
    inputs = payload["directional_panel_inputs"]
    n_trials = payload["shared_correction"]["n_trials"]

    # (1) the batch variance is the ddof=1 sample variance of the batch's own
    # daily Sharpes -- and is NOT the floor constant, so the fallback branch
    # cannot masquerade as the measured one here.
    sharpes = [items[family]["directional_panel"]["daily_moments"]["sr"]
               for family in sorted(items)]
    assert all(np.isfinite(sharpes))
    expected_variance = float(np.var(sharpes, ddof=1))
    assert expected_variance > 0.0
    assert inputs["sr_variance_daily"] == pytest.approx(
        expected_variance, rel=1e-12)
    assert inputs["sr_variance_daily"] != pytest.approx(
        constants.DEFAULT_SR_VARIANCE_FLOOR, rel=1e-6)
    assert n_trials == float(len(modules))

    # (2) every reported daily DSR equals an independent dsr call.
    for family, item in sorted(items.items()):
        panel_ = item["directional_panel"]
        moments = panel_["daily_moments"]
        expected = dsr.deflated_sharpe_ratio(
            sr=moments["sr"], n=moments["n"], skew=moments["skew"],
            kurtosis=moments["kurtosis"], n_trials=n_trials,
            sr_variance=expected_variance)
        assert panel_["daily_dsr_error"] is None, family
        assert panel_["daily_dsr"] == pytest.approx(
            expected, rel=1e-12, abs=1e-15), family
        assert panel_["daily_dsr_inputs"]["n_trials"] == n_trials
        assert panel_["daily_dsr_inputs"]["sr_variance_daily"] == pytest.approx(
            expected_variance, rel=1e-12)

    # (3) that equality is not satisfiable by a constant: the same formula on
    # a WRONG N, a 7x variance, or the floor constant gives a different
    # number, so each of those substitutions is observable.
    probe = items["b5_single"]["directional_panel"]
    moments = probe["daily_moments"]
    for label, wrong in (
            ("n_trials=1", {"n_trials": 1.0, "sr_variance": expected_variance}),
            ("variance x7", {"n_trials": n_trials,
                             "sr_variance": expected_variance * 7.0}),
            ("floor variance", {"n_trials": n_trials,
                                "sr_variance": constants.DEFAULT_SR_VARIANCE_FLOOR})):
        assert dsr.deflated_sharpe_ratio(
            sr=moments["sr"], n=moments["n"], skew=moments["skew"],
            kurtosis=moments["kurtosis"], **wrong) != pytest.approx(
                probe["daily_dsr"], rel=1e-9), label


def test_daily_dsr_moments_come_from_the_strategy_series_itself(
        tmp_path, monkeypatch):
    """The DSR's own inputs (sr, n, skew, kurtosis) must be the moments of
    the mark-to-market series, recomputed here from the rebuilt sleeve."""
    payload, _ = _run(tmp_path, monkeypatch, modules={"b5_single": MODULE_SINGLE},
                      accepted=["b5_single"], dry_run=True)
    item = _by_id(payload)["b5_single"]
    panel_ = item["directional_panel"]
    sleeve = _sleeve("AAA", item["execution_costs"]["AAA"]["cost_bps"])
    expected = benchmark_relative.daily_sharpe_moments(sleeve)
    assert panel_["daily_moments"]["n"] == expected["n"] == len(sleeve)
    for key in ("sr", "skew", "kurtosis"):
        assert panel_["daily_moments"][key] == pytest.approx(
            expected[key], rel=1e-9, abs=1e-15), key
    # One candidate -> the pinned DAILY constant (never the per-trade
    # floor), and the DSR is that formula on it.
    variance = payload["directional_panel_inputs"]["sr_variance_daily"]
    assert variance == constants.DAILY_SR_VARIANCE
    assert panel_["daily_dsr"] == pytest.approx(
        dsr.deflated_sharpe_ratio(
            sr=expected["sr"], n=expected["n"], skew=expected["skew"],
            kurtosis=expected["kurtosis"],
            n_trials=payload["shared_correction"]["n_trials"],
            sr_variance=variance), rel=1e-12, abs=1e-15)


# ------------------------------------------------- REPORT-ONLY invariant

def test_the_panel_influences_no_recorded_stage_dsr_or_ledger_row(
        tmp_path, monkeypatch):
    """Pinned by running the SAME batch twice -- once with the panel
    computed, once with it suppressed -- and comparing the recorded figures
    across the two runs. Comparing two values that both come from the
    panel-computed run cannot see a panel pass that mutates its input."""
    modules = {"b5_single": MODULE_SINGLE, "b5_pooled": MODULE_POOLED,
               "b5_always": MODULE_ALWAYS}
    accepted = sorted(modules)
    with_dir, without_dir = tmp_path / "with_panel", tmp_path / "without_panel"
    with_dir.mkdir()
    without_dir.mkdir()

    with_panel, with_paths = _run(with_dir, monkeypatch, modules=modules,
                                  accepted=accepted)

    def suppressed(*_args, **_kwargs):
        raise RuntimeError("panel suppressed for the report-only control")

    monkeypatch.setattr(batch5, "build_directional_panel", suppressed)
    without, without_paths = _run(without_dir, monkeypatch, modules=modules,
                                  accepted=accepted)

    computed, control = _by_id(with_panel), _by_id(without)
    assert set(computed) == set(control) == set(accepted)
    for family in accepted:
        assert computed[family]["directional_panel"] is not None, family
        assert control[family]["directional_panel"] is None, family
        assert computed[family]["stage"] == control[family]["stage"], family
        assert computed[family]["dsr_value"] == control[family]["dsr_value"], family
        assert computed[family]["shared_n_trials"] == control[family]["shared_n_trials"]
        for key in ("stats_in_sample", "stats_holdout", "holdout_net_edge_bps"):
            assert computed[family][key] == control[family][key], family + " " + key
    assert with_panel["shared_correction"] == without["shared_correction"]

    # ... and the same for what actually reached the two ledgers.
    def _outcomes(path):
        return {event["family_id"]: {key: event.get(key) for key in
                                     ("stage", "dsr_value", "sharpe_in_sample",
                                      "n_families_correction")}
                for event in trial_ledger.read_all(path)
                if event["event"] == "outcome"}

    recorded = _outcomes(with_paths["ledger"])
    assert set(recorded) == set(accepted)
    assert recorded == _outcomes(without_paths["ledger"])


# ------------------------------------------------ directional_support flag

MODULE_ALPHA = _HEADER + """
FAMILY_ID = "b5_alpha"
SHAPE = "single_long"
TRADED = ["ALPH"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
PARAMETERS = {"cycle": 6, "hold": 3}
EXPECTED_MIN_TRADES = 50

def make_signal_fn(ticker, index, frames):
    def signal(frame, i):
        return SignalResult(entry=(i % 6 == 5), exit=(i % 6 == 2))
    return signal
"""


def _alpha_factory():
    def factory(_index):
        def signal(frame, i):
            return batch3.backtest_engine.SignalResult(
                entry=(i % 6 == 5), exit=(i % 6 == 2))
        return signal
    return factory


def test_directional_support_fires_for_a_candidate_with_real_alpha(
        tmp_path, monkeypatch):
    """Both conjuncts genuinely satisfied: a candidate that holds only the
    up bars of its own price path has significant alpha against its own
    buy-and-hold AND a positive time-matched SPY excess. Without a fixture
    that can reach True, neither conjunct is exercised at all."""
    payload, _ = _run(tmp_path, monkeypatch, modules={"b5_alpha": MODULE_ALPHA},
                      accepted=["b5_alpha"], dry_run=True)
    item = _by_id(payload)["b5_alpha"]
    panel_ = item["directional_panel"]
    # The t-statistic is recomputed from the rebuilt sleeve, so "there is
    # real alpha here" is measured, not assumed.
    sleeve = _sleeve("ALPH", item["execution_costs"]["ALPH"]["cost_bps"],
                     factory=_alpha_factory())
    own_bh = benchmark_relative.buy_and_hold_daily_returns(
        _in_sample_bucket("ALPH"))
    expected = benchmark_relative.ols_alpha_beta(sleeve, own_bh)
    assert expected["alpha_daily"] > 0.0
    assert math.isfinite(expected["t_alpha_hac"])
    assert expected["t_alpha_hac"] > batch5.DIRECTIONAL_SUPPORT_T_ALPHA
    assert panel_["ols_vs_own"]["t_alpha_hac"] == pytest.approx(
        expected["t_alpha_hac"], rel=1e-9)
    # The second conjunct: a strictly positive mean time-matched SPY excess.
    assert panel_["time_matched_spy_excess_bps"]["in_sample"] > 0.0
    assert panel_["flags"] == {"beta_in_disguise": False,
                               "directional_support": True}


def _flag_item(family_id, *, t_own, tm_is, sr=0.05, n=600):
    """The minimum finalize_directional_panels reads off a panel: the daily
    moments, the own-benchmark OLS and the time-matched SPY excess."""
    return {"family_id": family_id,
            "directional_panel": {
                "daily_moments": {"sr": sr, "n": n, "skew": 0.0,
                                  "kurtosis": 3.0},
                "ols_vs_own": None if t_own is None else {"t_alpha_hac": t_own},
                "time_matched_spy_excess_bps": {"in_sample": tm_is,
                                                "holdout": None}}}


_T_BAR = batch5.DIRECTIONAL_SUPPORT_T_ALPHA


@pytest.mark.parametrize("case,t_own,tm_is,support,beta", [
    ("both conjuncts hold", _T_BAR + 1.0, 12.5, True, False),
    ("excess negative", _T_BAR + 1.0, -12.5, False, False),
    ("excess exactly zero", _T_BAR + 1.0, 0.0, False, False),
    ("excess unavailable", _T_BAR + 1.0, None, False, False),
    ("t below the bar", _T_BAR - 0.5, 12.5, False, False),
    ("t exactly at the bar", _T_BAR, 12.5, False, False),
    ("t just above the bar", _T_BAR + 1e-9, 12.5, True, False),
    ("t exactly zero", 0.0, 12.5, False, True),
    ("t just above zero", 1e-12, 12.5, False, False),
    ("t negative", -3.0, 12.5, False, True),
], ids=["both_conjuncts_hold", "excess_negative", "excess_exactly_zero", "excess_unavailable", "t_below_the_bar",
        "t_exactly_at_the_bar", "t_just_above_the_bar", "t_exactly_zero", "t_just_above_zero", "t_negative"])
def test_each_directional_support_conjunct_is_load_bearing(
        case, t_own, tm_is, support, beta):
    """One conjunct at a time: a case that fails ONLY the excess test and a
    case that fails ONLY the t test must both come out False, and the
    beta_in_disguise boundary sits at exactly zero (t == 0 IS beta)."""
    items = [_flag_item("subject", t_own=t_own, tm_is=tm_is),
             _flag_item("other", t_own=1.0, tm_is=1.0, sr=0.02)]
    batch5.finalize_directional_panels(items, {"n_trials": 7.0})
    flags = items[0]["directional_panel"]["flags"]
    assert flags["directional_support"] is support, case
    assert flags["beta_in_disguise"] is beta, case


@pytest.mark.parametrize("t_own", [None, float("nan")])
def test_an_unavailable_own_benchmark_t_leaves_both_flags_undetermined(t_own):
    """No own-benchmark regression (or a NaN t) is not evidence either way:
    both flags must be None rather than a default False."""
    items = [_flag_item("subject", t_own=t_own, tm_is=12.5),
             _flag_item("other", t_own=1.0, tm_is=1.0, sr=0.02)]
    batch5.finalize_directional_panels(items, {"n_trials": 7.0})
    assert items[0]["directional_panel"]["flags"] == {
        "beta_in_disguise": None, "directional_support": None}


# ------------------------------------------ run-report candidate table
# A single row PREFIX check ("| b5_single | single_long | AAA |") would stay
# green if a whole cell were dropped from every row, shifting every value one
# column left. The column list, the separator rule's width and each row's
# cell count are asserted together here, so the three cannot drift apart, and
# every column's POSITION is pinned by giving each cell a distinguishable
# value.

_TABLE_COLUMNS = ["family", "shape", "tickers", "trades IS/HO", "per-trade SR",
                  "DSR", "stage", "daily SR ann", "daily DSR", "t_alpha own",
                  "t_alpha SPY", "TM-SPY excess bps IS/HO", "flags"]


def _cells(line: str) -> list:
    """A markdown row's cells, with the leading/trailing pipe stripped, so a
    dropped cell changes the COUNT rather than hiding inside the string."""
    assert line.startswith("|") and line.endswith("|"), line
    return [cell.strip() for cell in line[1:-1].split("|")]


def _table_item(family_id, *, shape, tickers, ticker, stage, dsr_value,
                pool_declared=None, is_trades=101, ho_trades=7, sharpe=0.2345,
                daily_sr=1.5, daily_dsr=0.4321, t_own=2.5, t_spy=-3.25,
                tm_is=12.5, tm_ho=-4.5, flags=("directional_support",)):
    """The minimum render_candidate_table reads, with a DIFFERENT value in
    every numeric column so a one-column shift cannot pass unnoticed."""
    item = {
        "family_id": family_id, "shape": shape, "tickers": list(tickers),
        "ticker": ticker, "stage": stage, "dsr_value": dsr_value,
        "stats_in_sample": {"trade_count": is_trades, "sharpe": sharpe},
        "stats_holdout": {"trade_count": ho_trades},
        "directional_panel": {
            "daily_sharpe_annualized": daily_sr, "daily_dsr": daily_dsr,
            "ols_vs_own": {"t_alpha_hac": t_own},
            "ols_vs_spy": {"t_alpha_hac": t_spy},
            "time_matched_spy_excess_bps": {"in_sample": tm_is,
                                            "holdout": tm_ho},
            "flags": {name: True for name in flags}},
    }
    if pool_declared is not None:
        item["pool_declared"] = list(pool_declared)
    return item


def test_the_candidate_table_header_rule_and_every_row_agree_on_columns():
    pooled = _table_item("b5_a_pooled", shape="pooled_single_asset",
                         tickers=["AAA", "BBB"], ticker="AAA/BBB",
                         stage="cut_holdout", dsr_value=0.9876,
                         pool_declared=["AAA", "BBB", "LATE", "ONEBAR"])
    plain = _table_item("b5_z_plain", shape="single_long", tickers=["AAA"],
                        ticker="AAA", stage="cut_dsr", dsr_value=0.1234)
    lines = batch5.render_candidate_table([plain, pooled]).splitlines()
    header, rule = lines[0], lines[1]
    rows = lines[2:]

    n_columns = len(_TABLE_COLUMNS)
    assert _cells(header) == _TABLE_COLUMNS
    # The separator's width is DERIVED from the header here, so a hand-typed
    # count in the driver that no longer matches the header reddens.
    assert rule == "|" + "---|" * n_columns
    assert len(rows) == 2
    for line in rows:
        assert len(_cells(line)) == n_columns, line

    # Rows are family_id-sorted, so the pooled family comes first.
    pooled_cells, plain_cells = _cells(rows[0]), _cells(rows[1])
    assert pooled_cells[0] == "b5_a_pooled" and plain_cells[0] == "b5_z_plain"

    # The ticker column: "pool KEPT/DECLARED" for a pooled family (both
    # numbers recomputed from the item's own lists), a plain ticker
    # otherwise. A pooled family is never printed member-by-member.
    assert pooled_cells[2] == (
        f"pool {len(pooled['tickers'])}/{len(pooled['pool_declared'])}")
    assert pooled_cells[2] == "pool 2/4"
    assert "AAA/BBB" not in pooled_cells[2]
    assert plain_cells[2] == "AAA"

    # Every remaining column, pinned by POSITION and by recomputed value.
    assert plain_cells[1] == "single_long"
    assert plain_cells[3] == f"{101}/{7}"
    assert plain_cells[4] == f"{0.2345:.3f}"
    assert plain_cells[5] == f"{0.1234:.4f}"
    assert plain_cells[6] == "cut_dsr"
    assert plain_cells[7] == f"{1.5:.3f}"
    assert plain_cells[8] == f"{0.4321:.4f}"
    assert plain_cells[9] == f"{2.5:.3f}"
    assert plain_cells[10] == f"{-3.25:.3f}"
    assert plain_cells[11] == f"{12.5:.1f}/{-4.5:.1f}"
    assert plain_cells[12] == "directional_support"


def test_a_pooled_family_with_no_exclusions_still_renders_a_pool_cell():
    """compute_batch sets pool_declared for EVERY pooled family, not only the
    ones that dropped a member, so kept == declared must still print as a
    pool cell rather than falling back to the slash-joined ticker."""
    item = _table_item("b5_full_pool", shape="event_pooled",
                       tickers=["AAA", "BBB", "CCC"], ticker="AAA/BBB/CCC",
                       stage="cut_dsr", dsr_value=0.5,
                       pool_declared=["AAA", "BBB", "CCC"])
    cells = _cells(batch5.render_candidate_table([item]).splitlines()[2])
    assert cells[2] == "pool 3/3"


def test_a_missing_directional_panel_still_fills_every_table_column():
    """The panel is report-only and may be None (build failure). The row must
    still carry the full column count, with 'n/a' in the panel columns."""
    item = _table_item("b5_nopanel", shape="single_long", tickers=["AAA"],
                       ticker="AAA", stage="cut_dsr", dsr_value=0.5)
    item["directional_panel"] = None
    cells = _cells(batch5.render_candidate_table([item]).splitlines()[2])
    assert len(cells) == len(_TABLE_COLUMNS)
    assert cells[7:] == ["n/a", "n/a", "n/a", "n/a", "n/a/n/a", "none"]


# ------------------------------------------ survivor brief rendering
# Apart from b5_alpha, every CLI-driven fixture in this file ends in a DSR
# cut or an insufficient-evidence deferral, so none of them reaches
# render_briefs' survivor-stage branch. b5_alpha's deterministic
# up-bar pattern clears the DSR bar AND lands a positive holdout edge, so it
# genuinely reaches the survivor stage through the real CLI -- and b5_single, scored
# in the SAME batch, is cut, which pins the guard in both directions from one
# run.

def test_a_real_survivor_gets_a_brief_and_a_cut_family_in_the_same_batch_does_not(
        tmp_path, monkeypatch):
    payload, paths = _run(tmp_path, monkeypatch,
                          modules={"b5_alpha": MODULE_ALPHA,
                                   "b5_single": MODULE_SINGLE},
                          accepted=["b5_alpha", "b5_single"])
    survivor, cut = _by_id(payload)["b5_alpha"], _by_id(payload)["b5_single"]

    # The survivor stage is RECOMPUTED, not read: rebuild the backtest from the
    # engine, re-derive the DSR from the batch's own shared correction, and
    # re-apply the holdout rule.
    cost_bps = survivor["execution_costs"]["ALPH"]["cost_bps"]
    windows = batch3.run_single_asset_windows(
        _FRAMES["ALPH"], _alpha_factory(), holdout_start=HOLDOUT_START,
        holdout_end=TESTED_DATE, cost_bps=cost_bps, tax_haircut_rate=0.0)
    stats = windows["stats_in_sample"]
    correction = payload["shared_correction"]
    expected_dsr = dsr.deflated_sharpe_ratio(
        sr=stats["sharpe"], n=stats["trade_count"], skew=stats["skew"],
        kurtosis=stats["kurtosis"], n_trials=correction["n_trials"],
        sr_variance=correction["sr_variance"])
    assert expected_dsr >= constants.DSR_PASS_THRESHOLD
    assert batch3.backtest_engine.holdout_passes(
        stats["net_edge_bps"], windows["holdout_net_edge_bps"]) is True
    assert survivor["stage"] == "survived"
    assert survivor["dsr_value"] == pytest.approx(expected_dsr, rel=1e-12)
    assert cut["stage"] == "cut_dsr"
    assert trial_ledger.latest_outcome(paths["ledger"], "b5_alpha") == "survived"

    # Exactly one brief was WRITTEN, named for the survivor only.
    brief_name = reporting.brief_filename("b5_alpha", "ALPH")
    written = [os.path.basename(path) for path in payload["outputs_written"]]
    assert [name for name in written
            if name.startswith("candidate-brief-")] == [brief_name]
    assert not os.path.exists(
        os.path.join(paths["run_dir"], reporting.brief_filename("b5_single", "AAA")))

    # ...and it names the family and carries the recomputed numbers.
    brief = open(os.path.join(paths["run_dir"], brief_name),
                 encoding="utf-8").read()
    assert f"# {'Test b5_alpha'} — ALPH" in brief
    assert "synthetic mechanism for b5_alpha" in brief
    assert f"Deflated Sharpe Ratio: {expected_dsr:.4f}" in brief
    assert f"Trade count: {stats['trade_count']}" in brief
    assert f"per-trade Sharpe {stats['sharpe']:.3f}" in brief
    assert (f"{windows['stats_holdout']['trade_count']} trades, net edge "
            f"{windows['holdout_net_edge_bps']:.1f} bps/trade") in brief
    assert f"{HOLDOUT_START} to {TESTED_DATE}" in brief
    assert json.loads(
        brief.split("**Parameters tested:** ")[1].split("\n")[0]
    ) == CATALOG_PARAMETERS["b5_alpha"]

    # The run report's own survivor line names the same family, and its
    # table -- rendered by the REAL pipeline, not a hand-built item list --
    # carries one full-width row per scored candidate.
    report = open(os.path.join(paths["run_dir"], "run-report.md"),
                  encoding="utf-8").read()
    assert "**Survivors: b5_alpha (ALPH)**" in report
    table = [line for line in report.splitlines() if line.startswith("|")]
    assert _cells(table[0]) == _TABLE_COLUMNS
    assert table[1] == "|" + "---|" * len(_TABLE_COLUMNS)
    assert len(table) == 2 + len(payload["results"])
    for line in table[2:]:
        assert len(_cells(line)) == len(_TABLE_COLUMNS), line
    assert _cells(table[2])[0] == "b5_alpha" and _cells(table[2])[6] == "survived"
    assert _cells(table[3])[0] == "b5_single" and _cells(table[3])[6] == "cut_dsr"


# ------------------------------------------------ portable (repo-relative) paths

def test_portable_path_inside_the_repo_is_forward_slash_repo_relative(tmp_path):
    inside = tmp_path / "research" / "run" / "abi1-results.json"
    assert batch5.portable_path(
        str(inside), repo_root=str(tmp_path)) == "research/run/abi1-results.json"
    assert batch5.portable_path(str(tmp_path), repo_root=str(tmp_path)) == "."


def test_portable_path_resolves_a_relative_path_against_the_working_directory(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert batch5.portable_path(
        os.path.join("data", "x.json"), repo_root=str(tmp_path)) == "data/x.json"


def test_portable_path_outside_the_repo_keeps_only_the_file_name(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere" / "quotes.json"
    assert batch5.portable_path(
        str(outside), repo_root=str(repo)) == "<external>/quotes.json"
    # A sibling directory sharing the repo's name as a PREFIX is not inside it.
    sibling = tmp_path / "repo-other" / "f.json"
    assert batch5.portable_path(
        str(sibling), repo_root=str(repo)) == "<external>/f.json"


def test_panel_breadth_snapshot_path_inside_the_repo_is_recorded_repo_relative(
        tmp_path):
    snapshot_path, sha = _write_snapshot(tmp_path)
    relative = os.path.basename(snapshot_path)
    modules_dir = _write_modules(
        tmp_path, {"b5_panel": _panel_module(relative, sha)})
    found, problems = batch5.discover_modules(
        modules_dir, repo_root=str(tmp_path))
    assert problems == []
    entry = found["b5_panel"]["spec"]["snapshots"][_SNAPSHOT_KEY]
    assert entry["path"] == relative


def test_module_problem_paths_are_repo_relative_or_external(tmp_path):
    modules_dir = _write_modules(tmp_path, {"b5_bad": MODULE_BAD})
    _, inside = batch5.discover_modules(modules_dir, repo_root=str(tmp_path))
    assert [p["path"] for p in inside] == ["mods/b5_bad.py"]
    _, outside = batch5.discover_modules(modules_dir, repo_root=str(tmp_path / "repo"))
    assert [p["path"] for p in outside] == ["<external>/b5_bad.py"]


def test_batch5_results_json_records_no_absolute_or_home_directory_paths(
        tmp_path, monkeypatch):
    payload, paths = _run(
        tmp_path, monkeypatch,
        modules={"b5_single": MODULE_SINGLE, "b5_bad": MODULE_BAD},
        accepted=["b5_single"])
    assert [p["module"] for p in payload["module_problems"]] == ["b5_bad"]
    assert workstation_leaks(payload, tmp_path) == []
    with open(paths["out"], "r", encoding="utf-8") as handle:
        assert not DRIVE_PATH.search(handle.read())
    provenance = payload["provenance"]
    assert provenance["accepted_path"].endswith("accepted.json")
    assert provenance["ledger_path"].endswith("ledger.jsonl")
    assert "run-report.md" in [os.path.basename(p)
                               for p in payload["outputs_written"]]


# ---------------------------------- error text must not leak a path
# MODULE_BAD/MODULE_SINGLE above only exercise CONTRACT violations, whose
# messages never happen to embed a path -- so the two tests above do not
# reach these sites. These three do.

def test_snapshot_not_found_reason_names_no_absolute_path(tmp_path):
    missing = str(tmp_path / "does_not_exist.json")
    modules_dir = _write_modules(
        tmp_path, {"b5_panel": _panel_module(missing, "0" * 64)})
    found, problems = batch5.discover_modules(modules_dir, repo_root=str(tmp_path))
    assert found == {}
    assert len(problems) == 1
    assert workstation_leaks(problems, tmp_path) == []


MODULE_RAISES_WITH_OWN_PATH = _HEADER + """
raise ValueError(f"boom in module at {__file__}")
"""


def test_module_top_level_exception_naming_its_own_path_is_scrubbed(tmp_path):
    """A candidate module's own top-level exception can embed __file__ (its
    own absolute path, set by the loader) -- the generic except-handler in
    discover_modules must not let that reach a committed reason verbatim."""
    modules_dir = _write_modules(tmp_path, {"b5_boom": MODULE_RAISES_WITH_OWN_PATH})
    found, problems = batch5.discover_modules(modules_dir, repo_root=str(tmp_path))
    assert found == {}
    assert len(problems) == 1
    assert "boom in module at" in problems[0]["reason"]
    assert workstation_leaks(problems, tmp_path) == []


def test_no_module_skip_reason_names_no_absolute_path(tmp_path, monkeypatch):
    """b5_single is accepted and cataloged but no module file is written for
    it -- compute_batch's own "no_module" skip reason interpolates
    scope['modules_dir'] directly and must not leak it raw."""
    payload, paths = _run(
        tmp_path, monkeypatch, modules={}, accepted=["b5_single"])
    assert len(payload["skipped"]) == 1
    assert payload["skipped"][0]["reason_code"] == "no_module"
    assert workstation_leaks(payload, tmp_path) == []
