"""Tests for scripts/run_commodity_oi_cycle_overlay_abi1.py -- the bespoke ABI-1
driver for commodity_open_interest_cycle_commodity_bond_overlay.

Every run below uses SYNTHETIC price frames injected through the driver's
``ohlc_loader`` hook, a SYNTHETIC CFTC open-interest cache built into ``tmp_path``
through the real fetch-and-cache code path, and a TEMPORARY ledger; nothing here
reads the network or touches data/mining/trial_ledger.jsonl or the real
data/mining/cftc_oi cache (a test below proves both are byte-identical after a
run). The dry-run tests prove wiring only -- never a real cost measurement. The
family's catalog entry is read from data/mining/catalog.json.
"""
import copy
import datetime
import hashlib
import json
import os
import re
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

import run_commodity_oi_cycle_overlay_abi1 as driver  # noqa: E402
from mining import account_relative, cftc_data as cd, cost_tax_inputs, trial_ledger  # noqa: E402
from mining.batch5 import commodity_open_interest_cycle_commodity_bond_overlay as mod  # noqa: E402
from mining.batch5 import treasury_month_end_index_demand_overlay as treasury  # noqa: E402
from mining import history_cutoff as hc  # noqa: E402

_FAMILY = "commodity_open_interest_cycle_commodity_bond_overlay"
_ENTRY_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == _FAMILY)


_DRIVER_PATH = os.path.join(SCRIPTS_ROOT, "run_commodity_oi_cycle_overlay_abi1.py")
# the driver's REAL default ledger, captured at import time: the guard fixture below redirects the driver's own
# default paths for every test, so tests that need the real one must read it from here
_REAL_LEDGER_PATH = driver.LEDGER_PATH

# Deliberately DIFFERENT spreads per symbol everywhere below: with identical fixtures the
# suite would have zero power to catch a leg-cost / SPY-cost or a leg-frame / SPY-frame mixup.
_QUOTES = {
    "DBC": {"bid": 22.00, "ask": 22.02},     # ~9.1 bps
    "TLT": {"bid": 88.00, "ask": 88.05},     # ~5.7 bps
    "SPY": {"bid": 750.00, "ask": 750.15},   # ~2.0 bps
}


# ------------------------------------------------------------ synthetic inputs

def _ohlc(close, index):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999, "Close": close}, index=index)


def _walk(index, seed, vol, drift, start):
    r = np.random.RandomState(seed).normal(drift, vol, len(index))
    r[0] = 0.0
    return start * np.cumprod(1.0 + r)


def _sessions(n):
    return treasury.exchange_sessions(pd.Timestamp("2008-01-02"), pd.Timestamp("2022-12-30"))[:n]


_END = pd.Timestamp("2016-08-31")          # a scheduled month-end: the standard frames end exactly on an entry date
_LEAD = 300


def _default_legs():
    return int(_sessions(4000).get_loc(_END)) + 1 - _LEAD


def _frames(n_legs=None, lead=_LEAD, seed=5, irx_close=None, tail=0):
    """SPY lists ``lead`` sessions before the legs (production: SPY from 1993, DBC from 2006) and
    ends on the SAME last session as the legs (by default a month-end, so a real run is not refused for
    a truncated final window); only ^IRX runs ``tail`` sessions past. Every frame sits on the REAL
    exchange session calendar."""
    n_legs = _default_legs() if n_legs is None else n_legs
    sessions = pd.DatetimeIndex(_sessions(n_legs + lead + tail))
    account_end = len(sessions) - tail
    leg_idx = sessions[lead:account_end]
    spy_idx = sessions[:account_end]
    frames = {
        "SPY": _ohlc(_walk(spy_idx, seed, 0.01, 0.0004, 100.0), spy_idx),
        "DBC": _ohlc(_walk(leg_idx, seed + 1, 0.012, 0.0009, 25.0), leg_idx),
        "TLT": _ohlc(_walk(leg_idx, seed + 2, 0.009, 0.0007, 110.0), leg_idx),
    }
    irx = np.linspace(0.5, 3.0, len(sessions)) if irx_close is None else irx_close(sessions)
    frames["^IRX"] = pd.DataFrame({"Open": irx, "High": irx, "Low": irx, "Close": irx}, index=sessions)
    return frames


def _loader(frames):
    def load(symbol):
        frame = frames[symbol].copy()
        frame.insert(0, "Date", frame.index)
        return frame.reset_index(drop=True)
    return load


_CODE_START = {c.code: "1986-01-07" for c in cd.CONTRACTS}
_CODE_START.update({"002602": "1998-01-06", "005602": "1998-01-06", "001602": "1998-01-06",
                    "054642": "1996-04-02", "023651": "1990-04-10", "085692": "1989-07-25",
                    "111659": "2006-02-14"})


def _oi_rows(end="2023-06-27", start_override=None, seed=7, phase=0.0):
    """Socrata-shaped rows per contract for a synthetic open-interest history."""
    out = {}
    rng = np.random.RandomState(seed)
    for k, contract in enumerate(cd.CONTRACTS):
        first = (start_override or {}).get(contract.code, _CODE_START[contract.code])
        dates = pd.date_range(first, end, freq="7D")
        t = np.arange(len(dates))
        period = 52 * (1 + (k % 3) * 0.5)          # 1 to 2 year cycles: several state blocks shorter than a year
        level = 1000.0 * np.exp(0.7 * np.sin(2 * np.pi * t / period + k + phase) + 0.0008 * t * (1 if k % 2 else -1)
                                + rng.normal(0.0, 0.02, len(t)))
        name = f"{contract.name_keywords[0]} - SYNTHETIC EXCHANGE"
        out[contract.code] = [
            {"report_date_as_yyyy_mm_dd": d.strftime("%Y-%m-%dT00:00:00.000"),
             "cftc_contract_market_code": contract.code, "open_interest_all": str(int(round(v))),
             "market_and_exchange_names": name}
            for d, v in zip(dates, level)]
    return out


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


def _write_cache(directory, rows_by_code):
    def get(url, params=None, timeout=None):
        code = re.search(r"'([^']+)'", params["$where"]).group(1)
        return _Response(json.dumps(rows_by_code[code]).encode("utf-8"))
    cd.fetch_and_cache_all(str(directory), http_get=get, clock=lambda: pd.Timestamp("2026-09-24T09:00:00Z").to_pydatetime())
    return str(directory)


@pytest.fixture(scope="module")
def cftc_cache(tmp_path_factory):
    return _write_cache(tmp_path_factory.mktemp("cftc_cache"), _oi_rows())


def _quotes_payload(*, synthetic: bool, overrides=None) -> dict:
    payload = {
        "sampled_at_local": "2026-09-24T10:00:00",
        "market_state": "SYNTHETIC-test" if synthetic else "open",
        "source": "SYNTHETIC unit-test fixture" if synthetic else "unit test capture",
        "quotes": {s: {**q, "security_status": "Normal", "suspect": False} for s, q in _QUOTES.items()},
    }
    for key, value in (overrides or {}).items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def _fixture(tmp_path, *, synthetic=True, status="eligible", parameters=None, quotes_overrides=None):
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        entry = _catalog_entry_from(fh)
    entry["status"] = status
    if parameters is not None:
        entry["parameters"] = parameters
    catalog_path = str(tmp_path / "catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump([entry], fh)
    quotes_path = str(tmp_path / "quotes.json")
    with open(quotes_path, "w", encoding="utf-8") as fh:
        json.dump(_quotes_payload(synthetic=synthetic, overrides=quotes_overrides), fh)
    return catalog_path, quotes_path, str(tmp_path / "ledger.jsonl"), str(tmp_path / "research")


def _run(tmp_path, cache, *, dry_run=True, synthetic=True, frames=None, run_id="coi-test", loader=None,
         **fixture_kwargs):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=synthetic, **fixture_kwargs)
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id=run_id, tested_date="2026-09-24", dry_run=dry_run,
        signal_source_path="scripts/run_commodity_oi_cycle_overlay_abi1.py",
        signal_source_sha256="deadbeef", research_root=research_root,
        ohlc_loader=loader or _loader(frames or _frames()), cftc_cache_dir=cache)
    return result, ledger_path


def _mk(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def _ledger_paths_seen(monkeypatch, tmp_path, tmp_path_factory):
    """Two safeguards for every test.

    1. The driver's OWN default paths (ledger, catalog, research root) are redirected under tmp_path, so a
       call that forgets a path can never reach a real file.
    2. Every ledger path the driver actually reads or writes is recorded AT THE POINT OF USE
       (``trial_ledger.read_all`` and ``abi1_driver.record_abi1_outcome``), and after the test each must be a
       ledger the test helpers supplied explicitly: named ``ledger.jsonl`` under the pytest temporary base,
       never the real default and never the redirected default. A helper that stopped passing its own temporary
       ledger (and fell back on the driver's default) would therefore fail the test that used it.
    """
    seen = []
    real_read_all = driver.trial_ledger.read_all
    real_record = driver.abi1_driver.record_abi1_outcome

    def read_all(path, *args, **kwargs):
        seen.append(("trial_ledger.read_all", path))
        return real_read_all(path, *args, **kwargs)

    def record(path, *args, **kwargs):
        seen.append(("abi1_driver.record_abi1_outcome", path))
        return real_record(path, *args, **kwargs)

    monkeypatch.setattr(driver.trial_ledger, "read_all", read_all)
    monkeypatch.setattr(driver.abi1_driver, "record_abi1_outcome", record)
    guard_ledger = str(tmp_path / "_guard_ledger.jsonl")
    monkeypatch.setattr(driver, "LEDGER_PATH", guard_ledger)
    monkeypatch.setattr(driver, "CATALOG_PATH", str(tmp_path / "_guard_catalog.json"))
    monkeypatch.setattr(driver, "RESEARCH_ROOT", str(tmp_path / "_guard_research"))
    yield seen
    norm = lambda p: os.path.normcase(os.path.abspath(str(p)))          # noqa: E731
    base = norm(tmp_path_factory.getbasetemp()) + os.sep
    for site, path in seen:
        assert norm(path) != norm(_REAL_LEDGER_PATH), f"{site} touched the REAL default ledger"
        assert norm(path) != norm(guard_ledger), f"{site} fell back on the driver's default ledger path"
        assert norm(path).startswith(base), f"{site} used {path!r}, which is not under the pytest temporary base"
        assert os.path.basename(norm(path)) == "ledger.jsonl", f"{site} used {path!r}, not a helper-supplied ledger"


@pytest.fixture(scope="module")
def standard_dry_run(tmp_path_factory, cftc_cache):
    return _run(tmp_path_factory.mktemp("standard"), cftc_cache)[0]


def _frames_cut_after_entry(cache_snapshot_table, past, *, which=-15):
    """The standard synthetic frames with EVERY frame cut ``past`` sessions after one monthly
    window's entry date."""
    frames = _frames()
    leg_index = frames["DBC"].index
    plan = mod.plan_windows(leg_index, cache_snapshot_table)
    entry = plan["monthly"][which][0]
    last = leg_index[leg_index.get_loc(entry) + past]
    return {symbol: frame.loc[:last] for symbol, frame in frames.items()}, entry


@pytest.fixture(scope="module")
def state_table(cftc_cache):
    snapshot = cd.load_cached_open_interest(cftc_cache)
    return mod.signal_state_table(snapshot.series, mod.month_end_sessions(pd.Timestamp("1986-01-02"),
                                                                         pd.Timestamp("2022-12-30")))


# ------------------------------------------------------------ dry-run wiring

def test_dry_run_evaluates_end_to_end_without_writing_a_ledger_row(tmp_path, cftc_cache):
    result, ledger_path = _run(tmp_path, cftc_cache)
    assert result["family_id"] == _FAMILY
    assert "reason_code" not in result
    assert result["verdict"] in ("not_scoreable", "cut_dsr", "cut_risk_refusal",
                                 "holdout_not_evaluable", "cut_holdout", "survived")
    assert result["tax_policy"] == "realization" and result["benchmark"] == "SPY"
    assert result["tickers"] == ["DBC", "TLT"]
    assert not os.path.exists(ledger_path)


def test_dry_run_reports_the_structural_and_state_details(standard_dry_run):
    result = standard_dry_run
    diag = result["overlay_diagnostics"]
    assert diag["n_windows"] == result["n_trades"] >= 4
    assert diag["n_windows_by_leg"]["DBC"] >= 1 and diag["n_windows_by_leg"]["TLT"] >= 1     # both legs are traded
    # the SCORED convention is monthly-resized: one window per monthly decision; the merged one is the sensitivity
    assert diag["n_monthly_decisions"] == diag["n_windows"]
    assert diag["window_convention"] == {"scored": "monthly-resized", "sensitivity": "merged",
                                         "scored_merge_same_leg": False}
    assert diag["window_convention_sensitivity"]["n_trades"] < diag["n_windows"]
    assert diag["worst_window_net_ret"] < diag["best_window_net_ret"]
    assert diag["overlay_notional_fraction"] == 0.25
    assert diag["first_window_entry"] > "2009-03"
    assert 0.85 < diag["overlay_session_share"] <= 1.0          # the overlay is on in essentially every month
    assert result["time_in_market"] == 1.0
    state = diag["state_diagnostics"]
    assert state["evaluations"] > 60 and 0.0 < state["high_share"] < 1.0
    assert set(state["by_period"]) == {"2006-2008", "2009-2016", "2017-2022", "2023-onward"}
    assert state["holdout"]["evaluations"] > 0
    assert state["switches"] >= diag["window_convention_sensitivity"]["n_trades"] - 2   # blocks = switches + 1 (less end effects)
    json.dumps(result, allow_nan=False, default=str)


def test_the_cftc_cache_provenance_is_reported(standard_dry_run):
    info = standard_dry_run["overlay_diagnostics"]["cftc_cache"]
    assert info["n_contracts"] == 18 and info["retrieved_at"] == "2026-09-24T09:00:00Z"
    assert info["latest_asof"] == "2023-06-27" and len(info["fingerprint_sha256"]) == 64
    assert info["point_in_time_status"].startswith("not_verified_never_revised")


def test_the_cache_fingerprint_changes_when_any_contract_file_changes(tmp_path, cftc_cache):
    a = driver.cache_fingerprint(cd.load_cached_open_interest(cftc_cache))
    rows = _oi_rows()
    rows["088691"][10]["open_interest_all"] = str(int(rows["088691"][10]["open_interest_all"]) + 1)
    other = _write_cache(tmp_path / "other", rows)
    assert driver.cache_fingerprint(cd.load_cached_open_interest(other)) != a


def test_the_account_lives_on_the_legs_span_not_on_spys_longer_history(tmp_path, cftc_cache):
    frames = _frames()
    result, _ = _run(tmp_path, cftc_cache, frames=frames)
    assert result["account_start"] == frames["DBC"].index[0].strftime("%Y-%m-%d")
    assert result["account_end"] == frames["DBC"].index[-1].strftime("%Y-%m-%d")
    assert frames["SPY"].index[0] < frames["DBC"].index[0]


def test_costs_come_from_the_right_symbols_quotes(standard_dry_run):
    diag = standard_dry_run["overlay_diagnostics"]
    for symbol in ("DBC", "TLT"):
        expected = cost_tax_inputs.compute_spread_bps(_QUOTES[symbol]["bid"], _QUOTES[symbol]["ask"])
        assert diag["leg_cost_bps"][symbol] == pytest.approx(expected)
    spy_bps = cost_tax_inputs.compute_spread_bps(_QUOTES["SPY"]["bid"], _QUOTES["SPY"]["ask"])
    assert diag["benchmark_entry_cost_bps"] == pytest.approx(spy_bps / 2.0)
    assert len({round(v, 6) for v in diag["leg_cost_bps"].values()} | {round(spy_bps, 6)}) == 3


def test_the_financing_rate_is_the_bill_series_not_a_constant(tmp_path, cftc_cache):
    frames = _frames(irx_close=lambda idx: np.full(len(idx), 4.0))
    result, _ = _run(tmp_path, cftc_cache, frames=frames)
    expected = float(account_relative.coupon_equivalent_yield(0.04))
    assert result["overlay_diagnostics"]["mean_financing_rate_annual_on_overlay_sessions"] == pytest.approx(
        expected, rel=1e-12)


def test_a_negative_bill_print_is_loaded_without_the_positivity_gate(tmp_path, cftc_cache):
    def with_negative_prints(idx):
        irx = np.full(len(idx), 1.0)
        irx[(idx >= "2012-03-19") & (idx <= "2012-03-27")] = -0.105
        return irx
    frames = _frames(irx_close=with_negative_prints)
    assert (frames["^IRX"]["Close"] < 0).any()
    result, _ = _run(tmp_path, cftc_cache, frames=frames)
    assert "reason_code" not in result and np.isfinite(result["sharpe_daily_excess"])


def test_a_traded_leg_with_a_non_positive_price_still_fails_closed(tmp_path, cftc_cache):
    frames = _frames()
    frames["TLT"].iloc[500, frames["TLT"].columns.get_loc("Close")] = -1.0
    with pytest.raises(ValueError, match="TLT"):
        _run(tmp_path, cftc_cache, frames=frames)


def test_each_leg_frame_really_drives_the_curve(tmp_path, cftc_cache):
    base, _ = _run(_mk(tmp_path, "a"), cftc_cache, frames=_frames(seed=5))
    for leg, seed in (("DBC", 77), ("TLT", 78)):
        other = _frames(seed=5)
        other[leg] = _ohlc(_walk(other[leg].index, seed, 0.02, 0.0005, 50.0), other[leg].index)
        changed, _ = _run(_mk(tmp_path, f"b_{leg}"), cftc_cache, frames=other)
        assert base["sharpe_daily_excess"] != changed["sharpe_daily_excess"], leg


def test_the_spy_frame_really_drives_the_core_and_the_benchmark(tmp_path, cftc_cache):
    a, _ = _run(_mk(tmp_path, "a"), cftc_cache, frames=_frames(seed=5))
    other = _frames(seed=5)
    other["SPY"] = _ohlc(_walk(other["SPY"].index, 41, 0.012, 0.0002, 100.0), other["SPY"].index)
    b, _ = _run(_mk(tmp_path, "b"), cftc_cache, frames=other)
    assert a["sharpe_daily_excess"] != b["sharpe_daily_excess"]
    assert a["overlay_diagnostics"]["scored_excess_beta_to_spy"] != b["overlay_diagnostics"]["scored_excess_beta_to_spy"]


def test_the_open_interest_cache_really_drives_the_state_and_the_score(tmp_path, cftc_cache, standard_dry_run):
    """A different history of the SAME contracts moves the legs held and the score."""
    other_cache = _write_cache(tmp_path / "other_cache", _oi_rows(seed=99, phase=2.0))
    other, _ = _run(_mk(tmp_path, "other"), other_cache)
    assert other["sharpe_daily_excess"] != standard_dry_run["sharpe_daily_excess"]
    assert other["overlay_diagnostics"]["state_diagnostics"]["high_share"] != \
        standard_dry_run["overlay_diagnostics"]["state_diagnostics"]["high_share"]


def test_the_driver_reads_reports_only_as_the_lag_allows(tmp_path, standard_dry_run):
    """Overwrite every report the LAST decision cannot yet use with garbage: the score must not move."""
    rows = _oi_rows()
    cutoff = pd.Timestamp("2022-12-30")
    garbage = str(10 ** 8 - 1)          # built at run time: avoids a long digit literal in source
    for code, series in rows.items():
        for row in series:
            asof = pd.Timestamp(row["report_date_as_yyyy_mm_dd"])
            if cd.availability_date(asof) > cutoff:
                row["open_interest_all"] = garbage
    cache = _write_cache(tmp_path / "garbage_tail", rows)
    result, _ = _run(_mk(tmp_path, "run"), cache)
    assert result["sharpe_daily_excess"] == standard_dry_run["sharpe_daily_excess"]
    assert result["dsr_value"] == standard_dry_run["dsr_value"]


def test_flipping_the_one_switch_flips_the_scored_curve_and_its_sensitivity(tmp_path, cftc_cache, monkeypatch):
    """Both conventions are pinned: with the switch True the scored account is the merged one (fewer windows),
    the sensitivity is the monthly-resized one, and the ledger note says so."""
    base, _ = _run(_mk(tmp_path, "base"), cftc_cache)
    monkeypatch.setattr(driver.mod, "SCORED_MERGE_SAME_LEG", True)
    flipped, _ = _run(_mk(tmp_path, "flipped"), cftc_cache)
    diag = flipped["overlay_diagnostics"]
    assert diag["window_convention"] == {"scored": "merged", "sensitivity": "monthly-resized",
                                         "scored_merge_same_leg": True}
    assert diag["n_windows"] < diag["n_monthly_decisions"] == diag["window_convention_sensitivity"]["n_trades"]
    assert flipped["sharpe_daily_excess"] != base["sharpe_daily_excess"]
    note = flipped["cost_model_note"].lower()
    assert "the scored convention is merged" in note and "the report-only monthly-resized sensitivity" in note


def test_the_switch_is_part_of_the_pre_registered_parameters_so_flipping_it_needs_re_registration(tmp_path, cftc_cache):
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        drifted = dict(_catalog_entry_from(fh)["parameters"])
    assert drifted["scored_merge_same_leg"] is False
    drifted["scored_merge_same_leg"] = True
    result, _ = _run(tmp_path, cftc_cache, parameters=drifted)
    assert result["reason_code"] == "parameter_drift"


def test_the_scored_result_is_the_taxed_realization_evaluation_not_the_untaxed_one(tmp_path, cftc_cache):
    built = _built(cftc_cache)
    taxed, untaxed = built[2], built[3]
    result, _ = _run(tmp_path, cftc_cache)
    assert taxed["layer1"]["sharpe_daily_excess"] != pytest.approx(untaxed["layer1"]["sharpe_daily_excess"], rel=1e-9)
    assert result["sharpe_daily_excess"] == pytest.approx(taxed["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert result["dsr_value"] == pytest.approx(taxed["layer1"]["dsr_value"], rel=1e-9)
    assert result["verdict"] == taxed["verdict"]


def test_dry_run_reports_all_four_risk_refusals_on_the_excess_account(standard_dry_run):
    risk = standard_dry_run["overlay_diagnostics"]["risk_layer"]
    assert risk["account_basis"] == "overlay_only_excess"          # overlay-only excess account basis
    assert risk["absolute_cap"] == -0.35
    for key in ("relative_passed", "red_week_streak_passed", "worst_stretch_passed", "all_refusals_passed",
                "absolute_passed"):
        assert isinstance(risk[key], bool), key
    assert (risk["failed_refusals"] == []) == risk["all_refusals_passed"]
    assert risk["worst_stretch_window_weeks"] == 8
    assert risk["whole_account_reference"]["gates_verdict"] is False


def test_the_excess_is_exactly_zero_before_the_first_window_through_the_real_tax_path_and_entry_cost(cftc_cache):
    built = _built(cftc_cache)
    curve, taxed, bench = built[1], built[2], built[5]
    account = taxed["account_daily"]
    excess = account - bench.reindex(account.index)
    off = ~curve["overlay_mask"].to_numpy()
    assert off.sum() >= 10 and built[7] > 0.0
    assert (excess[off] == 0.0).all()
    assert (excess[curve["overlay_mask"].to_numpy()] != 0.0).any()


def test_a_core_that_differs_from_the_benchmark_outside_the_windows_is_refused(tmp_path, cftc_cache, monkeypatch):
    real = driver.mod.build_curve

    def drifted(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        daily = curve["strategy_daily"].copy()
        off = np.flatnonzero(~curve["overlay_mask"].to_numpy())
        daily.iloc[off[3]] += 1e-4
        curve["strategy_daily"] = daily
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", drifted)
    with pytest.raises(RuntimeError, match="not overlay-only"):
        _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_a_curve_that_lost_its_account_kind_is_refused_before_anything_is_recorded(tmp_path, cftc_cache, monkeypatch):
    real = driver.mod.build_curve

    def stripped(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        curve.pop("account_kind")
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", stripped)
    with pytest.raises(RuntimeError, match="overlay_only_excess"):
        _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_the_gated_risk_layer_is_the_excess_account_and_the_whole_account_is_a_separate_reference(cftc_cache):
    from mining import benchmark_relative
    built = _built(cftc_cache)
    taxed, bench = built[2], built[5]
    account = taxed["account_daily"]
    excess = account - bench.reindex(account.index)
    risk = _diagnostics(built)["risk_layer"]
    ref = risk["whole_account_reference"]
    assert risk["account_max_drawdown"] == pytest.approx(benchmark_relative.max_drawdown(excess), abs=1e-12)
    assert ref["account_max_drawdown"] == pytest.approx(benchmark_relative.max_drawdown(account), abs=1e-12)
    assert risk["account_max_drawdown"] != pytest.approx(ref["account_max_drawdown"])
    direct = account_relative.risk_refusals(account, bench)
    assert ref["all_refusals_passed"] == direct["all_refusals_passed"]


@pytest.mark.parametrize("which", ["relative_drawdown", "absolute_drawdown_cap", "red_week_streak",
                                   "worst_8_week_stretch"])
def test_the_risk_layer_report_names_exactly_the_refusal_that_failed(cftc_cache, which):
    built = _built(cftc_cache)
    taxed = copy.deepcopy(built[2])
    layer2 = taxed["layer2"]
    layer2["drawdown"].update(relative_passed=True, absolute_passed=True, passed=True)
    layer2["red_week_streak"]["passed"] = True
    layer2["worst_stretch"]["passed"] = True
    layer2["all_refusals_passed"] = True
    assert _diagnostics(built, result=taxed)["risk_layer"]["failed_refusals"] == []
    if which == "relative_drawdown":
        layer2["drawdown"].update(relative_passed=False, passed=False)
    elif which == "absolute_drawdown_cap":
        layer2["drawdown"].update(absolute_passed=False, passed=False)
    elif which == "red_week_streak":
        layer2["red_week_streak"]["passed"] = False
    else:
        layer2["worst_stretch"]["passed"] = False
    layer2["all_refusals_passed"] = False
    assert _diagnostics(built, result=taxed)["risk_layer"]["failed_refusals"] == [which]


def test_dry_run_reports_the_disclosed_next_open_sensitivity(standard_dry_run):
    sens = standard_dry_run["overlay_diagnostics"]["next_open_sensitivity"]
    assert "next-open" in sens["convention"]
    assert sens["sharpe_daily_excess"] is not None
    assert sens["sharpe_daily_excess"] != standard_dry_run["sharpe_daily_excess"]


def test_dry_run_reports_the_tax_identity_approximation_it_relies_on(standard_dry_run):
    diag = standard_dry_run["overlay_diagnostics"]
    assert 0.3 < diag["tax_identity_ratio_min"] <= diag["tax_identity_ratio_max"] < 3.0
    taxed, untaxed = diag["sharpe_daily_excess_taxed"], diag["sharpe_daily_excess_untaxed"]
    assert taxed == pytest.approx(standard_dry_run["sharpe_daily_excess"])
    assert diag["tax_effect_on_daily_excess_sharpe"] == pytest.approx(taxed - untaxed)


# ---------------------------------------------------------------- control arm

def test_the_report_carries_both_always_tlt_controls_and_the_merged_sensitivity(standard_dry_run):
    diag = standard_dry_run["overlay_diagnostics"]
    controls = diag["controls"]
    assert set(controls) == {"always_tlt_merged", "always_tlt_monthly_resized"}
    assert controls["always_tlt_merged"]["n_trades"] == 1
    assert controls["always_tlt_monthly_resized"]["n_trades"] == diag["n_monthly_decisions"] > 60
    merged = diag["window_convention_sensitivity"]
    assert merged["name"] == "strategy_merged"
    assert 0 < merged["n_trades"] < diag["n_monthly_decisions"]          # fewer, longer windows
    assert merged["sharpe_daily_excess"] != standard_dry_run["sharpe_daily_excess"]
    assert merged["tax_holding"]["n_trades"] == merged["n_trades"]
    for report in list(controls.values()) + [merged]:
        assert report["verdict"] in ("not_scoreable", "cut_dsr", "cut_risk_refusal",
                                     "holdout_not_evaluable", "cut_holdout", "survived")
        assert report["sharpe_daily_excess"] is not None
    assert "ledger" not in json.dumps(controls).lower()          # report-only: nothing recorded
    # the tax-holding split: the merged control is one multi-year (long-term, untaxed) trade, the
    # monthly-resized control and the strategy's short blocks are short-term
    assert controls["always_tlt_merged"]["tax_holding"]["n_long_term"] == 1
    assert controls["always_tlt_monthly_resized"]["tax_holding"]["n_long_term"] == 0
    assert diag["tax_holding"]["n_long_term"] == 0 and diag["tax_holding"]["n_trades"] == diag["n_windows"]
    assert diag["tax_holding"]["ltcg_tax_rate"] == 0.0 and diag["tax_holding"]["stcg_tax_rate"] == 0.12


def test_the_controls_equal_independent_evaluations_of_the_always_tlt_curves(tmp_path, cftc_cache):
    built = _built(cftc_cache)
    frames, table = built[0], built[8]
    from mining import abi1_driver
    kwargs = dict(benchmark_frame=frames["SPY"], benchmark_entry_cost_bps=built[7], account_type="taxable")
    legs = {s: frames[s] for s in mod.TRADED}
    expected = {}
    for name, merge in (("always_tlt_merged", True), ("always_tlt_monthly_resized", False)):
        curve = mod.build_control_curve(frames["SPY"], legs, frames["^IRX"], table, leg_cost_bps=built[6],
                                        spy_entry_cost_bps=built[7], merge_same_leg=merge)
        expected[name] = abi1_driver.evaluate_family(curve, tax_policy="realization", **kwargs)
    result, _ = _run(tmp_path, cftc_cache)
    for name, evaluation in expected.items():
        got = result["overlay_diagnostics"]["controls"][name]
        assert got["sharpe_daily_excess"] == pytest.approx(evaluation["layer1"]["sharpe_daily_excess"], rel=1e-12)
        assert got["dsr_value"] == pytest.approx(evaluation["layer1"]["dsr_value"], rel=1e-9)
    # the two conventions are genuinely different accounts
    assert expected["always_tlt_merged"]["layer1"]["sharpe_daily_excess"] != pytest.approx(
        expected["always_tlt_monthly_resized"]["layer1"]["sharpe_daily_excess"], rel=1e-9)


def test_the_beats_control_flags_compare_the_strategy_with_each_control_on_the_same_basis(standard_dry_run):
    diag = standard_dry_run["overlay_diagnostics"]
    beats = diag["strategy_beats_control"]
    strategy = standard_dry_run["sharpe_daily_excess"]
    for name, control in diag["controls"].items():
        assert beats[name] is (strategy > control["sharpe_daily_excess"])
    assert beats["beats_both"] is all(beats[n] for n in diag["controls"])


def test_the_control_never_looks_at_the_state(tmp_path, cftc_cache, standard_dry_run):
    """A different open-interest history changes the strategy but leaves both always-TLT controls untouched."""
    other_cache = _write_cache(tmp_path / "other_cache", _oi_rows(seed=99, phase=2.0))
    other, _ = _run(_mk(tmp_path, "other"), other_cache)
    assert other["sharpe_daily_excess"] != standard_dry_run["sharpe_daily_excess"]
    for name in ("always_tlt_merged", "always_tlt_monthly_resized"):
        assert other["overlay_diagnostics"]["controls"][name] == standard_dry_run["overlay_diagnostics"]["controls"][name]


# ---------------------------------------------------------------------- guards

def test_real_run_refuses_a_synthetic_quotes_file(tmp_path, cftc_cache):
    with pytest.raises(ValueError, match="SYNTHETIC"):
        _run(tmp_path, cftc_cache, dry_run=False, synthetic=True)


def test_a_missing_quote_for_any_leg_is_refused(tmp_path, cftc_cache):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    payload = _quotes_payload(synthetic=True)
    del payload["quotes"]["TLT"]
    with open(quotes_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    with pytest.raises(ValueError, match="TLT"):
        driver.compute_and_record(
            catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="x",
            tested_date="2026-09-24", dry_run=True, signal_source_path="p", signal_source_sha256="s",
            research_root=research_root, ohlc_loader=_loader(_frames()), cftc_cache_dir=cftc_cache)


def test_missing_catalog_entry_and_ineligible_status_skip_cleanly(tmp_path, cftc_cache):
    result, _ = _run(_mk(tmp_path, "inelig"), cftc_cache, status="ruled_out")
    assert result["reason_code"] == "catalog_status"
    catalog_path, quotes_path, ledger_path, research_root = _fixture(_mk(tmp_path, "none"))
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump([], fh)
    skipped = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="x",
        tested_date="2026-09-24", dry_run=True, signal_source_path="p", signal_source_sha256="s",
        research_root=research_root, ohlc_loader=_loader(_frames()), cftc_cache_dir=cftc_cache)
    assert skipped["reason_code"] == "not_in_catalog"


def test_parameter_drift_blocks_the_run(tmp_path, cftc_cache):
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        drifted = dict(_catalog_entry_from(fh)["parameters"])
    drifted["min_median_months"] = 24
    result, _ = _run(tmp_path, cftc_cache, parameters=drifted)
    assert result["reason_code"] == "parameter_drift"


def test_an_existing_abi1_outcome_blocks_a_second_trial(tmp_path, cftc_cache):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    trial_ledger.record_outcome(ledger_path, _FAMILY, "earlier", "2026-09-01", stage="cut_dsr", dsr_value=0.1,
                                sharpe_in_sample=0.0, construction="ABI-1", note="prior")
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="again",
        tested_date="2026-09-24", dry_run=True, signal_source_path="p", signal_source_sha256="s",
        research_root=research_root, ohlc_loader=_loader(_frames()), cftc_cache_dir=cftc_cache)
    assert result["reason_code"] == "already_scored_abi1"


# ------------------------------------------------- the CFTC cache guards

@pytest.mark.parametrize("dry_run", [True, False])
def test_a_missing_cftc_cache_is_refused_with_a_reason_code_and_writes_nothing(tmp_path, dry_run):
    result, ledger_path = _run(tmp_path, str(tmp_path / "no_such_cache"), dry_run=dry_run, synthetic=dry_run)
    assert result["reason_code"] == "cftc_cache_missing" and "verdict" not in result
    assert not os.path.exists(ledger_path)
    json.dumps(result, allow_nan=False)


def test_a_corrupt_cftc_cache_is_refused(tmp_path):
    cache = _write_cache(tmp_path / "cache", _oi_rows())
    path = os.path.join(cache, "088691.json")
    rows = json.load(open(path, encoding="utf-8"))
    rows[5]["open_interest_all"] = "1"
    json.dump(rows, open(path, "w", encoding="utf-8"))
    result, ledger_path = _run(_mk(tmp_path, "run"), cache, dry_run=False, synthetic=False)
    assert result["reason_code"] == "cftc_cache_corrupt" and not os.path.exists(ledger_path)


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_cftc_cache_older_than_the_last_decision_is_refused_not_silently_frozen(tmp_path, dry_run):
    cache = _write_cache(tmp_path / "cache", _oi_rows(end="2016-06-28"))
    result, ledger_path = _run(_mk(tmp_path, "run"), cache, dry_run=dry_run, synthetic=dry_run)
    assert result["reason_code"] == "cftc_cache_stale"
    assert "2016-06-28" in result["reason"] and "2016-08-31" in result["reason"] and "verdict" not in result
    assert not os.path.exists(ledger_path)


def test_an_undefined_state_at_a_decision_date_is_refused_not_guessed(tmp_path):
    late = {c.code: "2007-06-05" for c in cd.CONTRACTS}          # first CMI 2008-07, first state 2011-06
    cache = _write_cache(tmp_path / "cache", _oi_rows(start_override=late))
    result, ledger_path = _run(_mk(tmp_path, "run"), cache)
    assert result["reason_code"] == "state_unavailable" and "verdict" not in result
    assert not os.path.exists(ledger_path)


# --------------------- the cache boundary, stale reads and the state-history start

def _drop_reports(rows, code, dates):
    dropped = {pd.Timestamp(d) for d in dates}
    rows[code] = [r for r in rows[code] if pd.Timestamp(r["report_date_as_yyyy_mm_dd"]) not in dropped]
    return rows


def test_a_mid_history_hole_in_one_contract_is_refused_by_the_driver_not_silently_carried_forward(tmp_path):
    rows = _drop_reports(_oi_rows(), "088691", ["2010-03-09", "2010-03-16", "2010-03-23"])   # gold: 3 weekly reports lost
    cache = _write_cache(tmp_path / "cache", rows)                    # the last as-of date is unchanged, so it caches
    result, ledger_path = _run(_mk(tmp_path, "run"), cache)
    assert result["reason_code"] == "cftc_report_stale_at_month_end" and "verdict" not in result
    assert "088691" in result["reason"] and "2010-03-31" in result["reason"] and "2010-03-02" in result["reason"]
    assert not os.path.exists(ledger_path)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dry_run", [True, False])
def test_one_contract_that_stopped_early_is_refused_by_the_driver_even_when_seventeen_are_current(
        tmp_path, monkeypatch, dry_run):
    """The last decision (2016-08-31) needs the report as of 2016-08-23 for EVERY contract."""
    rows = _oi_rows(end="2016-08-23")
    rows["067651"] = [r for r in rows["067651"] if pd.Timestamp(r["report_date_as_yyyy_mm_dd"]) <= pd.Timestamp("2016-07-26")]
    with monkeypatch.context() as context:                                    # build the cache a fetcher of the past wrote
        context.setattr(cd, "require_one_last_asof", lambda _: None)
        cache = _write_cache(tmp_path / "cache", rows)
    result, ledger_path = _run(_mk(tmp_path, "run"), cache, dry_run=dry_run, synthetic=dry_run)
    assert result["reason_code"] == "cftc_contract_stale" and "verdict" not in result
    assert "067651" in result["reason"] and "2016-07-26" in result["reason"] and "2016-08-23" in result["reason"]
    assert not os.path.exists(ledger_path)


def test_a_cache_ending_exactly_on_the_required_report_is_accepted_and_one_week_older_is_refused(tmp_path):
    """The cache boundary at the last decision: that decision (2016-08-31) needs the report as of 2016-08-23."""
    exact = _write_cache(tmp_path / "exact", _oi_rows(end="2016-08-23"))
    result, _ = _run(_mk(tmp_path, "exact_run"), exact)
    assert "reason_code" not in result and result["verdict"] in (
        "not_scoreable", "cut_dsr", "cut_risk_refusal", "holdout_not_evaluable", "cut_holdout", "survived")
    assert result["overlay_diagnostics"]["cftc_cache"]["latest_asof"] == "2016-08-23"
    older = _write_cache(tmp_path / "older", _oi_rows(end="2016-08-16"))
    result, _ = _run(_mk(tmp_path, "older_run"), older)
    assert result["reason_code"] == "cftc_cache_stale" and "verdict" not in result
    assert "needs the report as of 2016-08-23" in result["reason"] and "2016-08-16" in result["reason"]


def test_a_mid_month_dry_run_ignores_the_month_end_that_has_not_happened_yet(tmp_path):
    """Frames end mid-month (2016-08-16) and the cache is fresh for the last decision (2016-07-29 needs the report
    as of 2016-07-19) but not for the coming month-end (2016-08-31 needs 2016-08-23, and would read a report 36 days
    old): the future month-end must not be checked, so the run is accepted."""
    frames = {symbol: frame.loc[:pd.Timestamp("2016-08-16")] for symbol, frame in _frames().items()}
    cache = _write_cache(tmp_path / "cache", _oi_rows(end="2016-07-26"))
    result, _ = _run(_mk(tmp_path, "run"), cache, frames=frames)
    assert "reason_code" not in result and result["overlay_diagnostics"]["final_window_truncated"] is True
    assert result["overlay_diagnostics"]["cftc_cache"]["latest_asof"] == "2016-07-26"
    assert result["account_end"] == "2016-08-16"


def _oi_rows_with_a_pre_2000_regime_break(**kwargs):
    """The synthetic history with a strong growth regime before 2000 (about one log unit a year), continuous at
    the break, so the expanding-median baseline differs visibly depending on where the state history starts."""
    rows = _oi_rows(**kwargs)
    cut, origin = pd.Timestamp("2000-01-01"), pd.Timestamp("1986-01-01")
    for series in rows.values():
        for row in series:
            asof = pd.Timestamp(row["report_date_as_yyyy_mm_dd"])
            weeks = (min(asof, cut) - origin).days / 7.0
            row["open_interest_all"] = str(int(round(float(row["open_interest_all"]) * np.exp(0.02 * weeks))))
    return rows


def test_the_drivers_state_history_start_is_the_pre_registered_parameter():
    assert driver.STATE_HISTORY_START == mod.STATE_HISTORY_START == mod.PARAMETERS["state_history_start"]
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        assert _catalog_entry_from(fh)["parameters"]["state_history_start"] == driver.STATE_HISTORY_START == "1986-01-02"


def test_the_state_history_start_decides_the_scored_states_so_a_different_start_moves_them(tmp_path, monkeypatch):
    """With a growth regime break before 2000 the expanding-median baseline depends on where the history starts:
    the driver's per-year state shares equal those of the registered start and differ from a 1999 start."""
    cache = _write_cache(tmp_path / "break_cache", _oi_rows_with_a_pre_2000_regime_break())
    snapshot = cd.load_cached_open_interest(cache)
    last = _frames()["SPY"].index[-1]
    base, _ = _run(_mk(tmp_path, "base"), cache)
    first_entry, end = pd.Timestamp(base["overlay_diagnostics"]["first_window_entry"]), pd.Timestamp(base["account_end"])

    def by_year(start):
        table = mod.signal_state_table(snapshot.series, mod.month_end_sessions(pd.Timestamp(start), last))
        return mod.state_diagnostics(table, start=first_entry, end=end)["by_year"]

    registered = by_year(mod.PARAMETERS["state_history_start"])
    assert base["overlay_diagnostics"]["state_diagnostics"]["by_year"] == registered
    assert by_year("1999-01-04") != registered                              # the fixture has power to see the start
    monkeypatch.setattr(driver, "STATE_HISTORY_START", "1999-01-04")
    moved, _ = _run(_mk(tmp_path, "moved"), cache)
    assert moved["overlay_diagnostics"]["state_diagnostics"]["by_year"] != registered


def test_the_cache_fingerprint_changes_when_any_one_of_the_eighteen_contracts_changes_and_pins_the_order():
    class _Snap:
        def __init__(self, hashes):
            self.manifest = {"contracts": {code: {"sha256": h} for code, h in hashes.items()}}
    base_hashes = {code: hashlib.sha256(code.encode("ascii")).hexdigest() for code in cd.CONTRACT_CODES}
    base = driver.cache_fingerprint(_Snap(base_hashes))
    seen = {base}
    for code in cd.CONTRACT_CODES:                                           # each of the 18 in turn, first and last included
        changed = dict(base_hashes)
        changed[code] = hashlib.sha256((code + "x").encode("ascii")).hexdigest()
        seen.add(driver.cache_fingerprint(_Snap(changed)))
    assert len(seen) == 19
    joined = "|".join(f"{code}:{base_hashes[code]}" for code in cd.CONTRACT_CODES)      # the pre-registered order
    assert base == hashlib.sha256(joined.encode("ascii")).hexdigest()
    assert list(cd.CONTRACT_CODES) != sorted(cd.CONTRACT_CODES)                          # so a sorted order is a different hash
    in_sorted_order = "|".join(f"{code}:{base_hashes[code]}" for code in sorted(cd.CONTRACT_CODES))
    assert base != hashlib.sha256(in_sorted_order.encode("ascii")).hexdigest()


def test_the_driver_docstring_names_the_new_cache_refusal_reason_codes():
    doc = " ".join(driver.__doc__.split())
    for phrase in ("cftc_cache_stale", "cftc_contract_stale", "cftc_report_stale_at_month_end", "older than 14 days",
                   "EVERY contract"):
        assert phrase in doc, phrase


# --------------------------------------------------- ledger path (tmp ledger)

def test_a_non_dry_run_against_a_temp_ledger_writes_exactly_one_abi1_row_with_every_disclosure(tmp_path, cftc_cache):
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    rows = trial_ledger.read_all(ledger_path)
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and sorted(r.get("event") for r in rows) == ["first_test", "outcome"]
    row = outcomes[0]
    assert row["family_id"] == _FAMILY and row["construction"] == "ABI-1"
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    assert row["stage"] == result["stage"]
    assert row["dsr_value"] == pytest.approx(result["dsr_value"], rel=1e-12)
    assert row["sharpe_in_sample"] == pytest.approx(result["sharpe_daily_excess"], rel=1e-12)
    note = (row.get("note") or "").lower()
    assert note.startswith("cash_convention=sweep_cash_daily; benchmark=spy; tax_policy=realization; cost model:")
    for phrase in (
            # design and the single trial
            "one pooled trial", "open_interest_all", "socrata dataset 6dca-aqww", "contract counts",
            "not dollar open interest", "18 contracts", "expanding median", "regular-holiday schedule",
            "never read off the data",
            # deviations from the paper
            "deviations from hong and yogo", "median split, not a regression", "tlt, not the 10-year note",
            "not about 30 commodities", "five in all", "contract-level growth averaged within each sector",
            "aggregate-then-growth", "contract counts, not dollar open interest",
            # point-in-time status: never labelled never-revised
            "not verified never-revised", "2017-03-28", "2018-09-18", "2019-10-15", "corrected values",
            "never labelled never-revised", "cache fingerprint", "d+6 calendar days",
            # publication blackouts and the holdout
            "publication blackout month-end refuses to update the state and carries the prior state",
            "left out of the expanding median", "of them inside the abi-1 holdout",
            # composition drift and the near-static state
            "composition drift", "grains start 1998", "static long-tlt", "regime switches",
            # the scored window convention (one switch) and its sensitivity
            "window convention", "one switch", "the scored convention is monthly-resized",
            "every calendar month is its own window", "conservative on cost", "the report-only merged sensitivity",
            "fixed share count",
            # the blackout table, the tie month, the stale-base months, the lag margin, composition figures
            "christmas-shifted report as of monday 2018-12-24", "2019-02-28", "accelerated plan",
            "longer than 12 months", "exact tie", "2019-11-29", "three days of margin in a normal week",
            "immaterial to the state", "241 of the 476 cmi values", "84, 168 and 259 days",
            # control arm and the pre-declared rule with this run's numbers
            "control arm", "always-tlt", "always_tlt_merged daily-excess sharpe", "always_tlt_monthly_resized daily-excess sharpe",
            "pre-declared interpretation rule", "exceeds that of both always-tlt controls",
            "strategy beats the controls this run",
            # flattering assumptions
            "no margin spread", "flatters the strategy", "energy-heavy", "equal-weight across four sectors",
            "reg-t margin calls", "tax walk",
            # the risk layer
            "tax-holding effect", "realizes its gain untaxed", "held beyond a year",
            "why merging is not the scored convention", "flatters the strategy through the tax walk",
            "risk layer (the risk-basis rule", "overlay-only excess account", "risk_account_basis=overlay_only_excess",
            "whole-account reference", "gating nothing", "0.35 absolute drawdown cap", "spy-relative refusals",
            "longest losing-week streak", "worst 8-week stretch", "refusals failed in this run",
            "in units of whole-account equity", "realization-tax drag",
            "before this family's driver or any dry-run path of it existed",
            "can only verify the sessions before the first window",
            # sensitivities, coverage, pre-registration honesty, provenance
            "disclosed sensitivity", "next-open", "heuristic sensitivity", "different strategy definition",
            "windows_skipped_before_data=", "windows_not_started=", "final_window_truncated=false",
            "window_entered_on_last_bar=true", "pre-registration limits", "signal-only probes",
            # the pre-registration sentence is scoped to the build step and covers any dry run before the scored run
            "none was tuned", "the build step computed no return, sharpe or excess of this account",
            "any dry run or scratch-ledger rehearsal before the scored run ran the identical code and data as this row",
            "tuned nothing",
            "price data: dbc fetched", "spy fetched", "^irx fetched", "latest-listing leg's first session",
            # five holiday-week decisions, the tie month's early-timing sensitivity, the sixth limitation
            "2009-11-30, 2011-02-28, 2015-11-30, 2020-11-30 and 2022-02-28",
            "about 30 minutes before the close on the decision day", "not verified against a cftc notice",
            "sensitive to the unverified pre-2000 release timing", "an extra 7 or 14 days of lag",
            "sixth disclosed limitation", "a limitation, not a deviation from the paper", "the deviations stay five",
            "present-day survivor set", "no membership rule applied at each historical date"):
        assert phrase in note, phrase
    assert "never_revised" not in note.replace("never-revised", "")
    # the tax-holding split states this run's own counts for the scored convention and its sensitivity
    overlay = result["overlay_diagnostics"]
    tax, alt_tax = overlay["tax_holding"], overlay["window_convention_sensitivity"]["tax_holding"]
    assert f"{tax['n_long_term']} of {tax['n_trades']} scored windows in this run were held beyond a year" in note
    assert f"{alt_tax['n_long_term']} of {alt_tax['n_trades']} windows of the merged sensitivity" in note
    risk = result["overlay_diagnostics"]["risk_layer"]
    assert f"max drawdown {round(risk['account_max_drawdown'], 4)} against spy's {round(risk['benchmark_max_drawdown'], 4)}" in note
    assert "refusals failed in this run: " + (", ".join(risk["failed_refusals"]) or "none") in note
    controls = result["overlay_diagnostics"]["controls"]
    for name, control in controls.items():
        assert f"{name} daily-excess sharpe {round(control['sharpe_daily_excess'], 4)}" in note
    state = result["overlay_diagnostics"]["state_diagnostics"]
    assert f"{state['evaluations']} monthly evaluations" in note and f"{state['switches']} regime switches" in note
    assert os.path.abspath(ledger_path).startswith(os.path.abspath(str(tmp_path)))


def test_an_unscoreable_real_run_records_the_instrument_set_benchmark_and_cash_convention(tmp_path, cftc_cache):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="short-history",
        tested_date="2026-09-24", dry_run=False, signal_source_path="scripts/run_commodity_oi_cycle_overlay_abi1.py",
        signal_source_sha256="a" * 64, research_root=research_root,
        ohlc_loader=_loader(_frames(n_legs=int(_sessions(4000).get_loc(pd.Timestamp("2010-02-26"))) + 1 - _LEAD)),
        cftc_cache_dir=cftc_cache)
    assert "reason_code" not in result, result.get("reason")
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    observed = outcomes[0]["evidence"]["observed"]
    assert observed["instrument_set"] == ["DBC", "TLT", "SPY"]
    assert observed["benchmark"] == "SPY" and observed["cash_convention"] == "sweep_cash_daily"
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        assert observed["parameters"] == _catalog_entry_from(fh)["parameters"]


def test_tax_holding_details_split_at_exactly_the_long_term_threshold_and_weight_by_positive_return():
    entry = pd.Timestamp("2010-01-29")
    trades = pd.DataFrame({
        "entry_date": [entry, entry, entry, entry],
        "exit_date": [entry + pd.Timedelta(days=365), entry + pd.Timedelta(days=366),
                      entry + pd.Timedelta(days=800), entry + pd.Timedelta(days=90)],
        "net_ret": [0.02, 0.03, 0.05, -0.04]})
    out = driver.tax_holding_diagnostics(trades)
    assert out["n_trades"] == 4 and out["n_long_term"] == 2                 # 365 days is still short-term, 366 is not
    assert out["positive_net_return_share_long_term"] == pytest.approx((0.03 + 0.05) / (0.02 + 0.03 + 0.05))
    assert out["long_term_holding_days"] == 365
    empty = driver.tax_holding_diagnostics(trades.iloc[:0])
    assert empty["n_trades"] == 0 and empty["positive_net_return_share_long_term"] is None


def test_a_window_held_beyond_a_year_is_realized_untaxed_and_a_short_window_is_taxed(cftc_cache):
    """The pipeline's tax walk: this is WHY the merge convention flatters the strategy. The same overlay
    account is taxed only on its short-term windows."""
    built = _built(cftc_cache)
    frames, curve, table = built[0], built[1], built[8]
    legs = {s: frames[s] for s in mod.TRADED}
    control = mod.build_control_curve(frames["SPY"], legs, frames["^IRX"], table, leg_cost_bps=built[6],
                                      spy_entry_cost_bps=built[7])
    from mining import abi1_driver
    kwargs = dict(benchmark_frame=frames["SPY"], benchmark_entry_cost_bps=built[7], account_type="taxable")
    taxed = abi1_driver.evaluate_family(control, tax_policy="realization", **kwargs)
    untaxed = abi1_driver.evaluate_family(control, tax_policy="none", **kwargs)
    assert (control["trades_df"]["exit_date"] - control["trades_df"]["entry_date"]).dt.days.iloc[0] > 365
    assert control["trades_df"]["net_ret"].iloc[0] > 0
    assert (taxed["account_daily"] - untaxed["account_daily"]).abs().max() < 1e-12      # a long-term gain pays nothing here
    monthly = mod.build_control_curve(frames["SPY"], legs, frames["^IRX"], table, leg_cost_bps=built[6],
                                      spy_entry_cost_bps=built[7], merge_same_leg=False)
    taxed_m = abi1_driver.evaluate_family(monthly, tax_policy="realization", **kwargs)
    untaxed_m = abi1_driver.evaluate_family(monthly, tax_policy="none", **kwargs)
    assert (taxed_m["account_daily"] - untaxed_m["account_daily"]).abs().max() > 1e-4   # short-term gains are taxed


def test_the_driver_default_ledger_is_never_used_by_these_tests(tmp_path, cftc_cache, monkeypatch, _ledger_paths_seen):
    """Drives the driver through every entry point the helpers use (a dry run, a real run and the CLI) and shows
    at the point of use that each ledger read and write went to the ledger the helper supplied, not to the
    driver's default (the autouse guard fixture asserts the same after EVERY test)."""
    assert os.path.abspath(_REAL_LEDGER_PATH).endswith(os.path.join("data", "mining", "trial_ledger.jsonl"))
    assert driver.LEDGER_PATH != _REAL_LEDGER_PATH          # the guard redirected the default for this test
    _, dry_ledger = _run(_mk(tmp_path, "dry"), cftc_cache)
    _, real_ledger = _run(_mk(tmp_path, "real"), cftc_cache, dry_run=False, synthetic=False, run_id="coi-guard")
    catalog_path, quotes_path, cli_ledger, research_root = _fixture(_mk(tmp_path, "cli"))
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    assert driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path, "--ledger", cli_ledger,
                        "--research-root", research_root, "--run-id", "cli-guard",
                        "--cftc-cache-dir", cftc_cache]) == 0
    supplied = {os.path.normcase(os.path.abspath(p)) for p in (dry_ledger, real_ledger, cli_ledger)}
    assert len(supplied) == 3
    seen_paths = {os.path.normcase(os.path.abspath(path)) for _, path in _ledger_paths_seen}
    assert seen_paths == supplied                           # exactly the helper-supplied ledgers, nothing else
    assert {site for site, _ in _ledger_paths_seen} == {"trial_ledger.read_all", "abi1_driver.record_abi1_outcome"}
    assert len(_ledger_paths_seen) >= 4                     # a read per run plus the real run write
    for path in supplied:
        assert path.startswith(os.path.normcase(os.path.abspath(str(tmp_path))))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def test_running_the_driver_never_touches_the_real_ledger_or_the_real_cftc_cache(tmp_path, cftc_cache):
    real_ledger = _REAL_LEDGER_PATH
    real_cache = cd.DEFAULT_CACHE_DIR
    ledger_before = _sha(real_ledger)
    cache_before = {n: _sha(os.path.join(real_cache, n)) for n in sorted(os.listdir(real_cache))} \
        if os.path.isdir(real_cache) else None
    _run(_mk(tmp_path, "dry"), cftc_cache)
    _run(_mk(tmp_path, "real"), cftc_cache, dry_run=False, synthetic=False, run_id="coi-real")
    assert _sha(real_ledger) == ledger_before
    if cache_before is not None:
        assert {n: _sha(os.path.join(real_cache, n)) for n in sorted(os.listdir(real_cache))} == cache_before


# ------------------------------------------------- window coverage (end of data)

def test_a_data_end_inside_a_month_window_is_reported_and_a_real_run_refuses_it(tmp_path, cftc_cache, state_table):
    frames, entry = _frames_cut_after_entry(state_table, 5)
    dry, _ = _run(_mk(tmp_path, "dry"), cftc_cache, frames=frames)
    diag = dry["overlay_diagnostics"]
    assert diag["final_window_truncated"] is True and diag["window_entered_on_last_bar"] is False
    assert "final_window_truncated=true" in dry["cost_model_note"].lower()
    real, ledger_path = _run(_mk(tmp_path, "real"), cftc_cache, dry_run=False, synthetic=False, frames=frames)
    assert real["reason_code"] == "truncated_final_window" and "verdict" not in real
    assert not os.path.exists(ledger_path)


def test_a_data_end_exactly_on_a_month_end_entry_opens_nothing_and_is_scored(tmp_path, cftc_cache, state_table):
    frames, entry = _frames_cut_after_entry(state_table, 0)
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False, frames=frames)
    assert "reason_code" not in result
    diag = result["overlay_diagnostics"]
    assert diag["window_entered_on_last_bar"] is True and diag["final_window_truncated"] is False
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    note = outcomes[0]["note"].lower()
    assert "window_entered_on_last_bar=true" in note and "final_window_truncated=false" in note


# --------------------------------------------- data-calendar gaps are refused

@pytest.mark.parametrize("dry_run", [True, False])
def test_unequal_trailing_coverage_is_refused_with_a_reason_code_and_the_missing_date(tmp_path, cftc_cache, dry_run):
    frames = _frames()
    last = frames["TLT"].index[-1]
    frames["DBC"] = frames["DBC"].iloc[:-1]
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=dry_run, synthetic=dry_run, frames=frames)
    assert result["reason_code"] == "unequal_trailing_coverage"
    assert result["missing_dates"] == [last.strftime("%Y-%m-%d")] and "DBC" in result["reason"]
    assert "verdict" not in result and not os.path.exists(ledger_path)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dry_run", [True, False])
def test_an_ordinary_session_missing_from_every_series_is_refused_and_named(tmp_path, cftc_cache, dry_run):
    frames = _frames()
    gone = frames["DBC"].index[900]
    frames = {symbol: frame.drop(gone) for symbol, frame in frames.items()}
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=dry_run, synthetic=dry_run, frames=frames)
    assert result["reason_code"] == "missing_exchange_session"
    assert result["missing_dates"] == [gone.strftime("%Y-%m-%d")]
    assert "verdict" not in result and not os.path.exists(ledger_path)


def test_a_known_unscheduled_closure_is_no_gap_and_a_bar_on_it_is_refused(tmp_path, cftc_cache):
    frames = _frames()
    sandy = pd.Timestamp("2012-10-29")
    assert sandy not in frames["DBC"].index
    ok, _ = _run(_mk(tmp_path, "ok"), cftc_cache, frames=frames)
    assert "reason_code" not in ok
    stray = {}
    for symbol, frame in frames.items():
        row = frame.loc[[frame.index[frame.index > sandy][0]]].copy()
        row.index = pd.DatetimeIndex([sandy])
        stray[symbol] = pd.concat([frame, row]).sort_index()
    refused, _ = _run(_mk(tmp_path, "stray"), cftc_cache, frames=stray)
    assert refused["reason_code"] == "bar_on_closed_day" and refused["extra_dates"] == ["2012-10-29"]


# --------------------------------------------- a real run never reuses cache

def test_a_real_run_refuses_ohlc_cached_by_an_earlier_dry_run_under_the_same_run_id(tmp_path, cftc_cache):
    dry, _ = _run(tmp_path, cftc_cache)
    assert set(dry["ohlc_sources"].values()) == {"fetched"}
    real, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    assert real["reason_code"] == "stale_ohlc_cache"
    assert real["cached_symbols"] == ["DBC", "TLT", "SPY", "^IRX"] and "fresh --run-id" in real["reason"]
    assert not os.path.exists(ledger_path)


def test_a_real_run_with_a_fresh_run_id_after_a_dry_run_is_scored_from_freshly_fetched_history(tmp_path, cftc_cache):
    _run(tmp_path, cftc_cache)
    real, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False, run_id="coi-fresh")
    assert "reason_code" not in real and set(real["ohlc_sources"].values()) == {"fetched"}
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


# ------------------------------------------------------------------------ cli

def test_cli_dry_run_writes_a_json_result_and_exits_zero(tmp_path, cftc_cache, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    code = driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--run-id", "cli-test", "--cftc-cache-dir", cftc_cache])
    assert code == 0
    output = os.path.join(research_root, "cli-test", "commodity-oi-cycle-overlay-dry-run.json")
    payload = json.load(open(output, encoding="utf-8"))
    assert payload["dry_run"] is True and payload["construction"] == "ABI-1"
    assert payload["result"]["family_id"] == _FAMILY
    assert not os.path.exists(ledger_path)


def test_a_real_run_requires_an_explicit_tested_date_and_run_id(tmp_path, cftc_cache, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    base = ["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
            "--research-root", research_root, "--cftc-cache-dir", cftc_cache]
    with pytest.raises(SystemExit):
        driver.main(base + ["--run-id", "no-date"])
    with pytest.raises(SystemExit):
        driver.main(base + ["--tested-date", "2026-09-24"])
    assert not os.path.exists(ledger_path) and not os.path.exists(research_root)


def test_a_dry_run_without_a_run_id_falls_back_to_the_default_run_id(tmp_path, cftc_cache, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    assert driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--cftc-cache-dir", cftc_cache]) == 0
    assert os.path.exists(os.path.join(research_root, driver.DEFAULT_RUN_ID, "commodity-oi-cycle-overlay-dry-run.json"))


def test_result_reports_the_provenance_of_every_price_history(tmp_path, cftc_cache):
    frames = _frames(tail=5)
    first, _ = _run(_mk(tmp_path, "one"), cftc_cache, frames=frames)
    assert set(first["ohlc_sources"]) == {"DBC", "TLT", "SPY", "^IRX"}
    assert set(first["ohlc_sources"].values()) == {"fetched"}
    for symbol in ("DBC", "TLT", "SPY", "^IRX"):
        assert first["last_bar_dates"][symbol] == frames[symbol].index[-1].strftime("%Y-%m-%d"), symbol
    assert first["last_bar_dates"]["^IRX"] != first["last_bar_dates"]["DBC"]


# --------------------------------------------- reported details vs independent maths

def _built(cache):
    from mining import abi1_driver, benchmark_relative
    frames = _frames()
    snapshot = cd.load_cached_open_interest(cache)
    table = mod.signal_state_table(snapshot.series, mod.month_end_sessions(pd.Timestamp("1986-01-02"),
                                                                          frames["SPY"].index[-1]))
    costs = {s: cost_tax_inputs.compute_spread_bps(q["bid"], q["ask"]) for s, q in _QUOTES.items()}
    leg_costs = {s: costs[s] for s in mod.TRADED}
    entry_cost = costs["SPY"] / 2.0
    legs = {s: frames[s] for s in mod.TRADED}
    curve = mod.build_curve(frames["SPY"], legs, frames["^IRX"], table, leg_cost_bps=leg_costs,
                            spy_entry_cost_bps=entry_cost)
    kwargs = dict(benchmark_frame=frames["SPY"], benchmark_entry_cost_bps=entry_cost, account_type="taxable")
    taxed = abi1_driver.evaluate_family(curve, tax_policy="realization", **kwargs)
    untaxed = abi1_driver.evaluate_family(curve, tax_policy="none", **kwargs)
    sens_curve = mod.build_next_open_sensitivity(frames["SPY"], legs, frames["^IRX"], table, leg_cost_bps=leg_costs,
                                                 spy_entry_cost_bps=entry_cost)
    sens = abi1_driver.evaluate_family(sens_curve, tax_policy="realization", **kwargs)
    bench = benchmark_relative.cost_adjusted_benchmark_daily_returns(frames["SPY"], entry_cost_bps=entry_cost)
    return frames, curve, taxed, untaxed, sens, bench, leg_costs, entry_cost, table


def _diagnostics(built, *, result=None):
    frames, curve, taxed, untaxed, sens, bench, leg_costs, entry_cost, _ = built
    return driver.overlay_diagnostics(
        curve, taxed if result is None else result, untaxed_result=untaxed, sensitivity_result=sens,
        benchmark_daily=bench, leg_cost_bps=leg_costs, benchmark_entry_cost_bps=entry_cost)


def test_details_equal_independent_aggregates_of_the_trades_frame(cftc_cache):
    built = _built(cftc_cache)
    curve = built[1]
    diag = _diagnostics(built)
    trades = curve["trades_df"]
    assert diag["n_windows"] == len(trades)
    assert diag["n_windows_by_leg"] == {leg: curve["leg_by_trade"].count(leg) for leg in mod.TRADED}
    assert diag["worst_window_net_ret"] == trades["net_ret"].min()
    assert diag["best_window_net_ret"] == trades["net_ret"].max()
    assert diag["sum_window_financing_ret"] == pytest.approx(trades["financing_ret"].sum())
    assert diag["sum_window_cost_ret"] == pytest.approx(trades["cost_ret"].sum())
    assert diag["first_window_entry"] == trades["entry_date"].iloc[0].strftime("%Y-%m-%d")
    assert diag["overlay_sessions"] == int(curve["overlay_mask"].sum())


def test_the_state_details_in_the_report_are_the_state_tables_own_numbers(cftc_cache, standard_dry_run):
    built = _built(cftc_cache)
    curve, table = built[1], built[8]
    account = curve["strategy_daily"].index
    first_entry = curve["window_info"]["monthly"][0][0]
    expected = mod.state_diagnostics(table, start=first_entry, end=account[-1])
    got = standard_dry_run["overlay_diagnostics"]["state_diagnostics"]
    for key in ("evaluations", "switches", "high_share"):
        assert got[key] == pytest.approx(expected[key])
    assert abs(got["holdout"]["evaluations"] - 756 / 21) <= 1                 # the last 756 sessions, about 36 months


def test_scored_beta_is_read_from_the_taxed_result(cftc_cache):
    from mining import benchmark_relative
    built = _built(cftc_cache)
    taxed, untaxed, bench = built[2], built[3], built[5]
    diag = _diagnostics(built)
    from_taxed = benchmark_relative.ols_alpha_beta(taxed["log_excess"], bench)["beta"]
    from_untaxed = benchmark_relative.ols_alpha_beta(untaxed["log_excess"], bench)["beta"]
    assert from_taxed != from_untaxed
    assert diag["scored_excess_beta_to_spy"] == pytest.approx(from_taxed, rel=1e-12)


def test_the_sensitivity_block_is_read_from_the_next_open_result_not_the_scored_one(cftc_cache):
    built = _built(cftc_cache)
    taxed, sens = built[2], built[4]
    diag = _diagnostics(built)
    assert diag["next_open_sensitivity"]["sharpe_daily_excess"] == pytest.approx(
        sens["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert diag["next_open_sensitivity"]["sharpe_daily_excess"] != pytest.approx(
        taxed["layer1"]["sharpe_daily_excess"], rel=1e-9)


# ---------- a REAL run needs a market-hours quote capture dated the run date
#
# refuse_synthetic_quotes (shared) only catches the SYNTHETIC marker, so a closed-market or wrong-date capture
# would pass it (an after-hours cost sample is unusable for pricing a trial). The driver therefore also
# refuses, for REAL runs only, unless the capture is (i) market_state starting with open (any case, surrounding
# whitespace ignored), (ii) a sampled_at_local that parses (naive = US Eastern, offset-aware converted to it),
# (iii) dated the run's tested date in Eastern and (iv) taken between 09:45:00 and 15:45:00 Eastern inclusive.
# Reason code quotes_not_market_hours; nothing is fetched or written; dry runs are exempt.

def _hours_refusal(overrides=None, tested="2026-09-24"):
    payload = {"sampled_at_local": "2026-09-24T10:00:00", "market_state": "open", "quotes": {}}
    for key, value in (overrides or {}).items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return driver.quotes_market_hours_refusal(payload, tested)


def test_an_open_in_window_capture_dated_the_run_date_is_accepted():
    assert _hours_refusal() is None


@pytest.mark.parametrize("sampled,accepted", [
    ("2026-09-24T09:45:00", True), ("2026-09-24T12:00:00", True), ("2026-09-24T15:45:00", True),
    ("2026-09-24T09:44:59", False), ("2026-09-24T15:45:01", False), ("2026-09-24T09:30:00", False),
    ("2026-09-24T16:00:00", False), ("2026-09-24T08:00:00", False), ("2026-09-24T17:30:00", False),
    ("2026-09-24T00:00:00", False), ("2026-09-24T15:45:00.500000", False), ("2026-09-24T09:44:59.999999", False),
    ("2026-09-24 09:45", True), ("2026-09-24 15:45", True)])
def test_the_capture_window_is_0945_to_1545_eastern_inclusive_at_both_ends(sampled, accepted):
    # state and date are valid here, so ONLY the clock time can refuse
    assert (_hours_refusal({"sampled_at_local": sampled}) is None) is accepted


@pytest.mark.parametrize("state,accepted", [
    ("open", True), ("OPEN", True), ("Open", True), ("open (regular session)", True), (" open", True),
    ("\topen\n", True), ("Open - regular", True),
    ("closed", False), ("Closed", False), ("pre-market", False), ("after-hours", False), ("premarket open", False),
    ("market closed", False), ("", False), ("   ", False), ("unknown", False), ("reopened", False)],
    ids=["open", "upper", "title", "open_regular_session", "leading_space", "tab_and_newline", "open_dash_regular",
         "closed", "closed_title", "pre_market", "after_hours", "premarket_open", "market_closed", "empty", "blanks",
         "unknown", "reopened"])
def test_the_market_state_must_start_with_open_in_any_case(state, accepted):
    # the time and date are valid here, so ONLY the state can refuse
    assert (_hours_refusal({"market_state": state}) is None) is accepted


@pytest.mark.parametrize("state", [5, ["open"], {"state": "open"}, True])
def test_a_market_state_that_is_not_a_string_is_a_refusal_not_an_exception(state):
    assert _hours_refusal({"market_state": state}) is not None


@pytest.mark.parametrize("sampled", ["2026-09-23T10:00:00", "2026-09-25T10:00:00", "2025-09-24T10:00:00",
                                     "2026-10-24T10:00:00", "2026-09-04T10:00:00"])
def test_the_capture_must_be_dated_the_run_date(sampled):
    assert _hours_refusal({"sampled_at_local": sampled}) is not None      # time and state are valid


def test_the_capture_date_is_compared_with_the_runs_own_tested_date_not_a_constant():
    assert _hours_refusal({"sampled_at_local": "2026-09-25T10:00:00"}, tested="2026-09-25") is None
    assert _hours_refusal({"sampled_at_local": "2026-09-24T10:00:00"}, tested="2026-09-25") is not None


@pytest.mark.parametrize("sampled", ["not-a-time", "", "   ", "NaT", "2026-13-45T10:00:00", 12345, 12.5, True,
                                     ["2026-09-24T10:00:00"], {"t": "2026-09-24T10:00:00"},
                                     # a capture file only ever holds a string: a time OBJECT is a wrong-shaped payload
                                     datetime.datetime(2026, 9, 24, 10, 0), pd.Timestamp("2026-09-24T10:00:00")])
def test_an_unparseable_sample_time_is_a_refusal_not_an_exception(sampled):
    assert _hours_refusal({"sampled_at_local": sampled}) is not None


@pytest.mark.parametrize("sampled", ["now", "today", "10:00", "12:30:00", "Sep 24 2026 10am", "noon"])
def test_a_sample_time_that_would_be_read_off_the_machine_clock_is_refused(sampled):
    """The loose pandas parser reads a bare clock time or the word now as TODAY on this machine, so such a
    string would turn the date check into a comparison with the wall clock. Only an explicit ISO date and
    time is accepted (a real capture writes one)."""
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    assert _hours_refusal({"sampled_at_local": sampled}, tested=today) is not None


@pytest.mark.parametrize("missing", ["sampled_at_local", "market_state"])
def test_a_payload_missing_either_field_is_a_refusal(missing):
    assert _hours_refusal({missing: None}) is not None


@pytest.mark.parametrize("payload", [None, [], "open", 5, {}, {"quotes": {}}])
def test_a_payload_of_the_wrong_shape_is_a_refusal_not_an_exception(payload):
    assert driver.quotes_market_hours_refusal(payload, "2026-09-24") is not None


@pytest.mark.parametrize("tested", ["not-a-date", "", None, 2026.5, "2026-13-45"])
def test_an_unparseable_tested_date_is_a_refusal_not_an_exception(tested):
    assert _hours_refusal(tested=tested) is not None


@pytest.mark.parametrize("sampled,tested", [("2026-11-01T01:30:00", "2026-11-01"),       # ambiguous: the hour repeats
                                            ("2026-03-08T02:30:00", "2026-03-08")])      # nonexistent: skipped by DST
def test_a_naive_time_inside_a_daylight_saving_gap_or_overlap_is_a_refusal_not_an_exception(sampled, tested):
    assert _hours_refusal({"sampled_at_local": sampled}, tested=tested) is not None


@pytest.mark.parametrize("sampled,accepted", [
    ("2026-09-24T09:50:00Z", False),         # 05:50 Eastern: a naive read of the digits would wrongly accept it
    ("2026-09-24T13:50:00Z", True),          # 09:50 Eastern
    ("2026-09-24T13:44:59Z", False),         # 09:44:59 Eastern
    ("2026-09-24T13:45:00Z", True),          # 09:45:00 Eastern, the inclusive open
    ("2026-09-24T19:45:00+00:00", True),     # 15:45 Eastern, the inclusive close
    ("2026-09-24T19:45:01+00:00", False),    # 15:45:01 Eastern
    ("2026-09-24T19:46:00Z", False),         # 15:46 Eastern
    ("2026-09-24T10:00:00-04:00", True),     # already Eastern daylight time
    ("2026-09-24T10:00:00+02:00", False),    # 04:00 Eastern
    ("2026-09-25T01:00:00Z", False),         # the evening of the 24th Eastern
    ("2026-09-24T02:00:00Z", False)])        # the night of the 23rd Eastern
def test_a_timezone_aware_sample_is_converted_to_eastern_and_a_naive_one_is_read_as_eastern(sampled, accepted):
    assert (_hours_refusal({"sampled_at_local": sampled}) is None) is accepted


@pytest.mark.parametrize("sampled,accepted", [
    ("2026-01-15T20:44:00Z", True),          # 15:44 Eastern STANDARD time: a fixed daylight offset would read 16:44
    ("2026-01-15T20:46:00Z", False),         # 15:46 Eastern standard time
    ("2026-01-15T14:45:00Z", True),          # 09:45 Eastern standard time: a fixed daylight offset would read 10:45
    ("2026-01-15T14:44:00Z", False),         # 09:44 Eastern standard time
    ("2026-01-15T15:00:00-05:00", True)])
def test_the_conversion_uses_the_real_eastern_rules_so_winter_dates_use_standard_time(sampled, accepted):
    assert (_hours_refusal({"sampled_at_local": sampled}, tested="2026-01-15") is None) is accepted


def test_the_refusal_names_what_was_wrong():
    assert "closed" in _hours_refusal({"market_state": "closed"})
    assert "2026-09-23" in _hours_refusal({"sampled_at_local": "2026-09-23T10:00:00"})
    assert "09:45" in _hours_refusal({"sampled_at_local": "2026-09-24T09:00:00"})
    assert "15:45" in _hours_refusal({"sampled_at_local": "2026-09-24T16:00:00"})


class _CountingLoader:
    """An OHLC loader that records every fetch, so a refusal can be shown to have fetched nothing."""

    def __init__(self):
        self.calls = []
        self._inner = _loader(_frames())

    def __call__(self, symbol):
        self.calls.append(symbol)
        return self._inner(symbol)


@pytest.mark.parametrize("overrides", [
    {"market_state": "closed"},                                       # closed, time and date valid
    {"market_state": "pre-market", "sampled_at_local": "2026-09-24T09:00:00"},
    {"market_state": "after-hours", "sampled_at_local": "2026-09-24T17:00:00"},
    {"sampled_at_local": "2026-09-23T10:00:00"},                      # wrong date
    {"sampled_at_local": "yesterday morning"},                        # unparseable
    {"sampled_at_local": "2026-09-24T15:46:00"},                      # a minute after the window
    {"sampled_at_local": "2026-09-24T09:44:00"},                      # a minute before it
    {"sampled_at_local": "2026-09-24T09:50:00Z"}])                    # 05:50 Eastern
def test_a_real_run_off_hours_is_refused_with_a_reason_code_and_fetches_and_writes_nothing(tmp_path, overrides):
    loader = _CountingLoader()
    result, ledger_path = _run(tmp_path, str(tmp_path / "no_such_cache"), dry_run=False, synthetic=False,
                               loader=loader, quotes_overrides=overrides)
    # refused BEFORE the OHLC fetch and BEFORE the CFTC cache is read (that cache does not even exist here)
    assert result["reason_code"] == "quotes_not_market_hours" and result["family_id"] == _FAMILY
    assert result["reason"] and "verdict" not in result and "stage" not in result
    assert loader.calls == []
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(str(tmp_path / "research"))           # no OHLC cache either
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("overrides,field", [({"sampled_at_local": None}, "sampled_at_local"),
                                             ({"market_state": None}, "market_state"),
                                             ({"sampled_at_local": 12345}, "sampled_at_local")])
def test_a_real_run_whose_capture_lacks_a_metadata_field_is_reason_coded_like_an_off_hours_one(
        tmp_path, overrides, field):
    loader = _CountingLoader()
    result, ledger_path = _run(tmp_path, str(tmp_path / "no_such_cache"), dry_run=False, synthetic=False,
                               loader=loader, quotes_overrides=overrides)
    assert result["reason_code"] == "quotes_not_market_hours" and result["family_id"] == _FAMILY
    assert field in result["reason"] and str(tmp_path) not in result["reason"]      # names the field, never a path
    assert "verdict" not in result and "stage" not in result
    assert loader.calls == []
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(str(tmp_path / "research"))


def test_a_dry_run_whose_capture_lacks_a_metadata_field_still_raises_from_the_shared_loader(tmp_path, cftc_cache):
    with pytest.raises(ValueError, match="sampled_at_local"):
        _run(tmp_path, cftc_cache, dry_run=True, synthetic=True, quotes_overrides={"sampled_at_local": None})


def test_a_dry_run_is_exempt_from_the_market_hours_guard(tmp_path, cftc_cache):
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=True, synthetic=True,
                               quotes_overrides={"market_state": "SYNTHETIC-closed",
                                                 "sampled_at_local": "2026-01-02T03:00:00"})
    assert "reason_code" not in result and result["verdict"]
    assert not os.path.exists(ledger_path)


@pytest.mark.parametrize("sampled", ["2026-09-24T09:45:00", "2026-09-24T15:45:00", "2026-09-24T14:00:00Z"])
def test_a_real_run_inside_the_window_is_scored_and_recorded(tmp_path, cftc_cache, sampled):
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False,
                               quotes_overrides={"sampled_at_local": sampled})
    assert "reason_code" not in result and result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


def test_the_cli_refuses_an_after_hours_real_run_and_records_nothing(tmp_path, cftc_cache, monkeypatch, capsys):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(
        tmp_path, synthetic=False, quotes_overrides={"market_state": "closed", "sampled_at_local": "2026-09-24T17:00:00"})
    loader = _CountingLoader()
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", loader)
    assert driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--cftc-cache-dir", cftc_cache,
                        "--run-id", "cli-after-hours", "--tested-date", "2026-09-24"]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "quotes_not_market_hours"
    assert loader.calls == [] and not os.path.exists(ledger_path)


def test_the_cli_refuses_a_capture_dated_a_different_day_than_the_tested_date(tmp_path, cftc_cache, monkeypatch, capsys):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _CountingLoader())
    assert driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--cftc-cache-dir", cftc_cache,
                        "--run-id", "cli-wrong-day", "--tested-date", "2026-09-25"]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "quotes_not_market_hours"
    assert not os.path.exists(ledger_path)


def test_the_driver_docstring_carries_the_market_hours_guard():
    text = driver.__doc__          # the module docstring itself, not any other mention in the source
    assert "quotes_not_market_hours" in text and "09:45 to 15:45 Eastern inclusive" in text
    assert "QUOTE CAPTURE HOURS" in text and "the run's ``--tested-date``" in text


# ---------- the controls gate the PRE-REGISTERED pass, not the recorded stage
#
# The stage vocabulary is closed in shared code and every tested family is recorded, so a passing main ABI-1
# stage is still written to the ledger. What the driver adds is (a) a top-level pre_registered_pass
# (passing stage AND this family's own control condition exactly True: the strategy's taxed daily-excess Sharpe
# exceeds that of BOTH always-TLT controls) and (b) an unmissable first sentence of the driver-supplied note
# (and of the printed summary) whenever the stage passed but the condition did not hold or is missing. The
# synthetic world cannot reach the passing stage on its own, so the real-path tests below FORCE the verdict of the
# main scored evaluation only (compute_and_record's first evaluate_family call) and force the control
# condition through interpret_controls; the decision function, the clause builder and the note builder are
# also tested directly.

_FAILED_FLAG = ("CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under this family's own "
                "pre-registered rule (the strategy's taxed daily-excess Sharpe must exceed that of BOTH always-TLT "
                "controls, the merged and the monthly-resized one). This is not a pass; do not promote it without review.")
_MET_CLAUSE = ("PRE-REGISTERED RULE MET: the ABI-1 stage survived AND the control condition holds (the strategy's "
               "taxed daily-excess Sharpe exceeds that of BOTH always-TLT controls); pre_registered_pass=True.")
_NOT_SURVIVED = ("cut_dsr", "cut_holdout", "cut_risk_refusal", "not_scoreable", "holdout_not_evaluable")
_CONTROL_NAMES = ("always_tlt_merged", "always_tlt_monthly_resized")


def _controls(merged, monthly):
    return {"always_tlt_merged": {"sharpe_daily_excess": merged},
            "always_tlt_monthly_resized": {"sharpe_daily_excess": monthly}}


@pytest.mark.parametrize("verdict,condition,expected", (
    [("survived", True, True), ("survived", False, False), ("survived", None, False)]
    + [(v, True, False) for v in _NOT_SURVIVED] + [(v, False, False) for v in _NOT_SURVIVED]
    + [(v, None, False) for v in _NOT_SURVIVED]))
def test_the_pre_registered_pass_needs_both_the_abi1_survival_and_the_control_condition(verdict, condition, expected):
    assert driver.pre_registered_pass(verdict, condition) is expected


@pytest.mark.parametrize("condition", [1, "yes", 1.0, [True]])
def test_only_an_exact_true_condition_is_a_pass(condition):
    """A truthy-but-not-True value (a stray 1, a string) must never read as a pass."""
    assert driver.pre_registered_pass("survived", condition) is False


@pytest.mark.parametrize("condition", [False, None])
def test_a_survived_stage_without_the_control_condition_carries_the_exact_unmissable_flag(condition):
    assert driver.pre_registered_clause("survived", condition) == _FAILED_FLAG
    assert driver.CONTROL_CONDITION_FAILED_FLAG == _FAILED_FLAG
    assert _FAILED_FLAG.startswith("CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under this "
                                   "family's own pre-registered rule (")
    assert _FAILED_FLAG.endswith("). This is not a pass; do not promote it without review.")


def test_a_survived_stage_whose_controls_hold_says_the_pre_registered_rule_was_met():
    clause = driver.pre_registered_clause("survived", True)
    assert clause == _MET_CLAUSE and "FAILED" not in clause and driver.PRE_REGISTERED_RULE_MET_CLAUSE == _MET_CLAUSE


@pytest.mark.parametrize("verdict", _NOT_SURVIVED)
@pytest.mark.parametrize("condition", [True, False, None])
def test_a_stage_that_did_not_survive_carries_no_clause(verdict, condition):
    assert driver.pre_registered_clause(verdict, condition) == ""


@pytest.mark.parametrize("strategy,merged,monthly,expected", [
    (0.30, 0.10, 0.20, True),        # above both
    (0.30, 0.10, 0.35, False),       # above the merged control only
    (0.30, 0.35, 0.10, False),       # above the monthly-resized control only
    (0.30, 0.40, 0.50, False),       # above neither
    (0.30, 0.30, 0.10, False),       # a tie is not "exceeds"
    (0.30, 0.10, 0.30, False),
    (-0.10, -0.30, -0.20, True),     # negative Sharpes compare the same way
    (-0.30, -0.10, -0.20, False),
    (None, 0.10, 0.20, None),        # a missing strategy or control Sharpe leaves the condition missing
    (0.30, None, 0.20, None),
    (0.30, 0.10, None, None)])
def test_the_control_condition_is_a_strict_comparison_with_both_always_tlt_controls(strategy, merged, monthly, expected):
    beats = driver.interpret_controls(strategy, _controls(merged, monthly))
    assert beats["beats_both"] is expected
    for name, control_sharpe in (("always_tlt_merged", merged), ("always_tlt_monthly_resized", monthly)):
        want = None if strategy is None or control_sharpe is None else bool(strategy > control_sharpe)
        assert beats[name] is want, name


def test_the_note_builder_leads_with_the_clause_only_for_a_survived_stage():
    body = "cost model: spy=1.00bps"
    assert driver.note_with_pre_registered_lead(body, "survived", False) == _FAILED_FLAG + " " + body
    assert driver.note_with_pre_registered_lead(body, "survived", None) == _FAILED_FLAG + " " + body
    assert driver.note_with_pre_registered_lead(body, "survived", True) == _MET_CLAUSE + " " + body
    for verdict in _NOT_SURVIVED:
        assert driver.note_with_pre_registered_lead(body, verdict, True) == body
        assert driver.note_with_pre_registered_lead(body, verdict, False) == body


def _force_main_verdict_survived(monkeypatch):
    """The synthetic world cannot reach the passing stage: relabel the verdict of the MAIN scored evaluation
    only (compute_and_record's first evaluate_family call; every control arm keeps its real verdict)."""
    real = driver.abi1_driver.evaluate_family
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            result = dict(result)
            result["verdict"] = "survived"
        return result
    monkeypatch.setattr(driver.abi1_driver, "evaluate_family", wrapped)


def _force_control_condition(monkeypatch, value):
    real = driver.interpret_controls

    def forced(*args, **kwargs):
        beats = dict(real(*args, **kwargs))
        beats["beats_both"] = value
        return beats
    monkeypatch.setattr(driver, "interpret_controls", forced)


_RECORDED_PREFIX = "cash_convention=sweep_cash_daily; benchmark=SPY; tax_policy=realization; "


@pytest.mark.parametrize("condition", [False, None])
def test_a_recorded_survived_stage_that_failed_the_controls_is_flagged_first_and_is_not_a_pre_registered_pass(
        tmp_path, cftc_cache, monkeypatch, condition):
    _force_main_verdict_survived(monkeypatch)
    _force_control_condition(monkeypatch, condition)
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    assert result["verdict"] == "survived" and result["stage"] == "survived"        # recorded unconditionally
    assert result["pre_registered_pass"] is False
    assert result["pre_registered_clause"] == _FAILED_FLAG
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and outcomes[0]["stage"] == "survived"                # the record is never skipped
    # the shared writer puts its fixed metadata prefix first; the driver's own note starts with the flag
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + _FAILED_FLAG + " cost model:")
    assert outcomes[0]["note"].count("CONTROL CONDITION FAILED") == 1
    assert "PRE-REGISTERED RULE MET" not in outcomes[0]["note"]
    assert "pre_registered_pass=false" in outcomes[0]["note"].lower()
    assert result["cost_model_note"].startswith(_FAILED_FLAG + " cost model:")


def test_a_recorded_survived_stage_that_passed_the_controls_says_the_rule_was_met(tmp_path, cftc_cache, monkeypatch):
    _force_main_verdict_survived(monkeypatch)
    _force_control_condition(monkeypatch, True)
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    assert result["stage"] == "survived" and result["pre_registered_pass"] is True
    assert result["pre_registered_clause"] == _MET_CLAUSE
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + _MET_CLAUSE + " cost model:")
    assert "CONTROL CONDITION FAILED" not in outcomes[0]["note"]
    assert "pre_registered_pass=true" in outcomes[0]["note"].lower()


@pytest.mark.parametrize("condition", [True, False, None])
def test_a_stage_that_did_not_survive_is_not_a_pre_registered_pass_whatever_the_controls_say(
        tmp_path, cftc_cache, monkeypatch, condition):
    _force_control_condition(monkeypatch, condition)
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    assert result["verdict"] != "survived" and result["pre_registered_pass"] is False
    assert result["pre_registered_clause"] == ""
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + "cost model:")        # no clause: unchanged prefix
    assert "CONTROL CONDITION FAILED" not in outcomes[0]["note"] and "PRE-REGISTERED RULE MET" not in outcomes[0]["note"]
    assert "pre_registered_pass=false" in outcomes[0]["note"].lower()


def test_the_pre_registered_pass_is_computed_from_this_runs_own_verdict_and_beats_flags(standard_dry_run):
    """Unforced: the field equals (verdict is the passing stage AND beats_both is exactly True) for the run's own numbers."""
    diag = standard_dry_run["overlay_diagnostics"]
    beats_both = diag["strategy_beats_control"]["beats_both"]
    assert standard_dry_run["pre_registered_pass"] is bool(standard_dry_run["verdict"] == "survived"
                                                            and beats_both is True)
    assert standard_dry_run["pre_registered_pass"] is False and standard_dry_run["pre_registered_clause"] == ""
    assert set(diag["strategy_beats_control"]) == {*_CONTROL_NAMES, "beats_both"}
    strategy = standard_dry_run["sharpe_daily_excess"]
    for name in _CONTROL_NAMES:
        assert diag["strategy_beats_control"][name] is bool(strategy > diag["controls"][name]["sharpe_daily_excess"])


def test_the_note_states_the_pre_registered_pass_and_the_rule_it_is_gated_by(tmp_path, cftc_cache):
    result, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    note = outcomes[0]["note"].lower()
    for phrase in ("pre_registered_pass=false", "true only when the abi-1 verdict is survived and the strategy "
                   "beats both always-tlt controls", f"this run's abi-1 verdict is {result['verdict']}"):
        assert phrase in note, phrase


def test_the_printed_summary_says_the_same_as_the_note_when_the_controls_failed(tmp_path, cftc_cache, monkeypatch,
                                                                                capsys):
    _force_main_verdict_survived(monkeypatch)
    _force_control_condition(monkeypatch, False)
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    code = driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--cftc-cache-dir", cftc_cache,
                        "--run-id", "cli-real", "--tested-date", "2026-09-24"])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert _FAILED_FLAG in lines[:-1]                        # its own printed line, before the JSON line
    printed = json.loads(lines[-1])
    assert printed["result"]["pre_registered_pass"] is False and printed["result"]["stage"] == "survived"
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + lines[0])          # the printed sentence IS the note's lead


def test_the_printed_summary_says_the_rule_was_met_when_it_was(tmp_path, cftc_cache, monkeypatch, capsys):
    _force_main_verdict_survived(monkeypatch)
    _force_control_condition(monkeypatch, True)
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    assert driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--cftc-cache-dir", cftc_cache,
                        "--run-id", "cli-met", "--tested-date", "2026-09-24"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2] == _MET_CLAUSE and json.loads(lines[-1])["result"]["pre_registered_pass"] is True


def test_the_driver_docstring_states_the_control_gate_and_the_closed_stage_vocabulary():
    text = driver.__doc__          # the module docstring itself, not any other mention in the source
    assert "THE CONTROLS GATE THE PRE-REGISTERED PASS, NOT THE RECORDED STAGE" in text
    assert "pre_registered_pass" in text and "CONTROL_CONDITION_FAILED_FLAG" in text and "evidence.VALID_STAGES" in text


def test_the_printed_summary_stays_quiet_when_nothing_survived(tmp_path, cftc_cache, monkeypatch, capsys):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    assert driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--cftc-cache-dir", cftc_cache]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["result"]["pre_registered_pass"] is False


# ------------------------------------------------------------------ hygiene

def test_the_driver_source_and_this_file_have_no_dollar_figure_dated_integer_or_long_digit_run():
    for path in (_DRIVER_PATH, __file__):
        text = open(path, "r", encoding="utf-8").read()
        assert not re.search(r"\$\s?\d", text), path
        assert not re.search(r"(?<![\d.\-_])(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?![\d\-_])",
                             text), path
        # an account-number-shaped value: any 8 to 17 digit run
        assert not re.search(r"(?<!\d\.)\b\d{8,17}\b", text), path


def test_the_only_file_writes_in_the_driver_use_paths_the_caller_supplies():
    """Static inventory of write call sites: each must take its path from an argument (ledger, research root,
    OHLC cache under the research root); none may hard-code a repo path."""
    text = open(_DRIVER_PATH, "r", encoding="utf-8").read()
    writes = re.findall(r"(?:_atomic_write_text|record_abi1_outcome|os\.makedirs|to_csv|json\.dump\(|write_text|open\([^)]*[\"']w)", text)
    assert sorted(set(writes)) == sorted({"_atomic_write_text", "record_abi1_outcome", "os.makedirs"})
    assert "trial_ledger.jsonl" not in text.replace("LEDGER_PATH", "")


# ---------- the metadata-refusal mapping must key on the loader's exact message
# shape, never on words that happen to appear in the quotes FILE PATH the shared loader prefixes to every error

_UNRELATED_LOADER_ERRORS = [
    "{p}/sampled_at_local/quotes.json: 'quotes' must be an object keyed by symbol",
    "{p}/market_state/quotes.json: 'quotes' must be an object keyed by symbol",
    "{p}/sampled_at_local_market_state.json: unreadable",
]


@pytest.mark.parametrize("real_error", _UNRELATED_LOADER_ERRORS)
@pytest.mark.parametrize("dry_run", [False, True])
def test_an_unrelated_loader_error_whose_path_names_a_metadata_field_is_re_raised_not_reason_coded(
        tmp_path, cftc_cache, monkeypatch, real_error, dry_run):
    def fake_load_quotes(path):
        raise ValueError(real_error.format(p=tmp_path))
    monkeypatch.setattr(driver.batch5, "load_quotes", fake_load_quotes)
    with pytest.raises(ValueError, match="must be an object keyed by symbol|unreadable"):
        _run(tmp_path, cftc_cache, dry_run=dry_run, synthetic=dry_run)


def test_a_metadata_error_under_a_path_that_names_the_other_field_reports_only_the_field_that_failed(
        tmp_path, cftc_cache, monkeypatch):
    def fake_load_quotes(path):
        raise ValueError(f"{tmp_path}/market_state/quotes.json: 'sampled_at_local' must be a non-empty string")
    monkeypatch.setattr(driver.batch5, "load_quotes", fake_load_quotes)
    result, _ledger = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False)
    assert result["reason_code"] == "quotes_not_market_hours"
    assert "sampled_at_local" in result["reason"] and "market_state" not in result["reason"]
    assert str(tmp_path) not in result["reason"]


# ---------- opt-in history end
#
# ``--history-end YYYY-MM-DD`` (``history_end=`` on compute_and_record) ENDS the scored history at a chosen NYSE
# session instead of refusing a calendar hole: every loaded series is cut, in memory only and after its OHLC cache
# was written, to the sessions dated on or before the cutoff. Default off. It fails closed (the cutoff must be a
# session and EVERY series must hold a bar dated exactly on it), it is recorded in the ledger note, the result and
# the payload file, and every existing check keeps running on the trimmed frames. The standard synthetic world's
# frames end on 2016-08-31; the cutoffs below are a scheduled month-end (a window is entered on its close) and a
# session inside the window entered there.

_HE_MONTH_END = pd.Timestamp("2015-01-30")
_HE_MID_WINDOW = pd.Timestamp("2015-02-10")
_HE_SERIES = ("DBC", "TLT", "SPY", "^IRX")
_HE_INVALID = ["2015-1-30", "2015/01/30", "", " 2015-01-30", "2015-01-30 ", "2015-01-30T00:00:00", "2015-02-30",
               "yesterday", "2015" "0130", 2015, datetime.datetime(2015, 1, 30, 9, 30),
               pd.Timestamp("2015-01-30 10:00")]
_HE_INVALID_IDS = ["unpadded", "slashes", "empty", "leading_space", "trailing_space", "with_time", "impossible_day",
                   "word", "compact", "integer", "datetime", "timestamp"]
_HE_NOT_SESSIONS = ["2014-06-28", "2014-07-04", "2012-10-29"]      # a Saturday, a holiday, an unscheduled closure


def _he_iso(stamp):
    return f"{pd.Timestamp(stamp):%Y-%m-%d}"


_HE_ACCEPT_ANY_DROP = 10 ** 6       # most history-end tests below cut hundreds of sessions: they accept any drop
_HE_DEFAULT_DROP = object()         # pass no max_history_drop keyword at all, i.e. the driver's own default


def _run_he(tmp_path, cache, history_end, *, dry_run=True, frames=None, run_id="coi-test", loader=None,
            tested_date="2026-09-24", max_history_drop=_HE_ACCEPT_ANY_DROP, **fixture_kwargs):
    """``_run`` plus the history-end setting; also returns the research root (the OHLC cache lives under it). A real
    (non dry) run gets a real-shaped quote capture and a dry run a synthetic one, as with ``_run``. A real run's
    history-end guard (a cutoff after the tested date, or a drop of more than 10 sessions from
    any series, is refused) is told to accept any drop unless the test passes its own ``max_history_drop``;
    ``_HE_DEFAULT_DROP`` leaves the keyword out so the driver's default applies."""
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=dry_run, **fixture_kwargs)
    extra = {} if max_history_drop is _HE_DEFAULT_DROP else {"max_history_drop": max_history_drop}
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id=run_id, tested_date=tested_date, dry_run=dry_run,
        signal_source_path="scripts/run_commodity_oi_cycle_overlay_abi1.py",
        signal_source_sha256="a" * 64, research_root=research_root,
        ohlc_loader=loader or _loader(frames or _frames()), cftc_cache_dir=cache, history_end=history_end, **extra)
    return result, ledger_path, research_root


@pytest.fixture
def main_evaluations(monkeypatch):
    """Every ``evaluate_family`` call of a run as ``(curve, result)``; the FIRST is the scored account."""
    real = driver.abi1_driver.evaluate_family
    seen = []

    def wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append((args[0], result))
        return result
    monkeypatch.setattr(driver.abi1_driver, "evaluate_family", wrapped)
    return seen


@pytest.fixture(scope="module")
def he_untrimmed(tmp_path_factory, cftc_cache):
    """One untrimmed dry run of the standard world with its scored curve and ABI-1 result captured."""
    seen = []
    real = driver.abi1_driver.evaluate_family

    def wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append((args[0], result))
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(driver.abi1_driver, "evaluate_family", wrapped)
        run = _run_he(tmp_path_factory.mktemp("he_untrimmed"), cftc_cache, None)
    return run[0], seen[0][0], seen[0][1]


def _he_poisoned_after(frames, cutoff, seed=11):
    """Every bar dated after ``cutoff`` scaled by a wild positive per-row factor (all four price columns of a row
    alike, so each row stays a valid bar); the bars up to the cutoff are untouched."""
    rng = np.random.RandomState(seed)
    out = {}
    for symbol, frame in frames.items():
        poisoned = frame.copy()
        late = np.asarray(poisoned.index > cutoff)
        factors = rng.uniform(0.5, 2.0, size=int(late.sum()))
        columns = ["Open", "High", "Low", "Close"]
        poisoned.loc[late, columns] = poisoned.loc[late, columns].to_numpy() * factors[:, None]
        out[symbol] = poisoned
    return out


def _he_cache_dir(research_root, run_id):
    return os.path.join(research_root, run_id, "artifacts", "ohlc")


def _he_assert_cache_is_full_history(research_root, run_id, frames):
    for symbol in _HE_SERIES:
        cached = pd.read_csv(os.path.join(_he_cache_dir(research_root, run_id), driver.batch5._cache_name(symbol)))
        assert len(cached) == len(frames[symbol]), symbol
        assert pd.Timestamp(cached["Date"].iloc[-1]) == frames[symbol].index[-1], symbol


# ---- (a) default off: byte-identical

def test_the_history_end_setting_defaults_off_and_a_run_without_it_matches_the_standard_dry_run(tmp_path, cftc_cache, standard_dry_run):
    import inspect
    assert inspect.signature(driver.compute_and_record).parameters["history_end"].default is None
    result, _, _ = _run_he(tmp_path, cftc_cache, None)
    as_json = lambda value: json.dumps(driver.batch5._json_safe(value), sort_keys=True)          # noqa: E731
    assert as_json(result) == as_json(standard_dry_run)
    assert "history_end" not in result and "raw_last_bar_dates" not in result
    assert hc.HISTORY_END_MARKER not in result["cost_model_note"]


def test_with_no_history_end_the_history_cutoff_module_is_never_called(tmp_path, cftc_cache, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("history-end code ran with the setting off")
    for name in ("parse_history_end", "trim_frame_to_history_end", "trim_frames_to_history_end", "build_provenance",
                 "history_end_clause", "check_real_run_guard", "refusal_payload"):
        monkeypatch.setattr(driver.history_cutoff, name, boom)
    result, _, _ = _run_he(tmp_path, cftc_cache, None)
    assert "reason_code" not in result and result["verdict"]


def test_a_real_run_without_the_setting_carries_no_history_end_clause(tmp_path, cftc_cache):
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, None, dry_run=False)
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and hc.HISTORY_END_MARKER not in outcomes[0]["note"]
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + "cost model:")
    assert "history_end" not in result


# ---- (d) recorded everywhere the driver records anything

def test_a_real_run_with_a_history_end_records_it_in_the_note_the_result_and_the_first_test_row(tmp_path, cftc_cache):
    frames = _frames(tail=5)
    cut = _he_iso(_HE_MONTH_END)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, cut, dry_run=False, frames=frames)
    assert "reason_code" not in result, result.get("reason")
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    provenance = result["history_end"]
    assert provenance["history_end"] == cut and provenance["opt_in"] is True and provenance["decision"] == hc.DECISION
    assert set(provenance["by_symbol"]) == set(_HE_SERIES)
    for symbol in _HE_SERIES:
        raw = frames[symbol].index
        entry = provenance["by_symbol"][symbol]
        assert entry["raw_last"] == _he_iso(raw[-1]) and entry["kept_last"] == cut, symbol
        assert entry["dropped_dates"] == [_he_iso(d) for d in raw[raw > _HE_MONTH_END]], symbol
        assert entry["dropped_sessions"] == len(entry["dropped_dates"]), symbol
    assert result["raw_last_bar_dates"] == {s: _he_iso(frames[s].index[-1]) for s in _HE_SERIES}
    assert result["raw_last_bar_dates"]["^IRX"] != result["raw_last_bar_dates"]["DBC"]
    assert result["last_bar_dates"] == {s: cut for s in _HE_SERIES}            # the TRIMMED history's last bars
    assert result["account_end"] == cut
    rows = trial_ledger.read_all(ledger_path)
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    note = outcomes[0]["note"]
    clause = hc.history_end_clause(provenance)
    assert note.startswith(_RECORDED_PREFIX + clause + " cost model:")
    assert result["cost_model_note"].startswith(clause + " cost model:")
    assert hc.HISTORY_END_MARKER in note and note.count(hc.HISTORY_END_MARKER) == 1
    for symbol in _HE_SERIES:
        assert f"{symbol} {_he_iso(frames[symbol].index[-1])}" in clause, symbol       # each series' raw last bar
    assert f"permanently ends at {cut}" in note and "cannot be re-scored on longer history" in note
    assert "The setting added, moved or repaired no bar" in note and "the vendor history extended past it" in note
    first_tests = [r for r in rows if r.get("event") == "first_test"]
    # a NEW family's first_test pin ends at the cutoff (a family that already has a first_test row keeps its own)
    assert len(first_tests) == 1 and first_tests[0]["holdout_end"] == cut


def test_a_dry_run_with_a_history_end_reports_it_in_the_result_and_the_note_and_writes_no_ledger(tmp_path, cftc_cache):
    cut = _he_iso(_HE_MONTH_END)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, cut)
    assert "reason_code" not in result and result["history_end"]["history_end"] == cut
    assert result["cost_model_note"].startswith(hc.history_end_clause(result["history_end"]) + " cost model:")
    assert result["account_end"] == cut and set(result["last_bar_dates"].values()) == {cut}
    assert not os.path.exists(ledger_path)


def test_the_history_end_clause_follows_the_pre_registered_flag_in_the_note(tmp_path, cftc_cache, monkeypatch):
    _force_main_verdict_survived(monkeypatch)
    _force_control_condition(monkeypatch, False)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), dry_run=False)
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    clause = hc.history_end_clause(result["history_end"])
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + _FAILED_FLAG + " " + clause + " cost model:")
    assert outcomes[0]["note"].count("CONTROL CONDITION FAILED") == 1
    assert outcomes[0]["note"].count(hc.HISTORY_END_MARKER) == 1 and result["pre_registered_pass"] is False


# ---- (d2) an UNSCORED real run under a history end carries the clause too (shared recorder's unscored_note keyword)
#
# The shared recorder writes the driver's extra note on the four scored stages only; the two unscored stages
# (not_scoreable, holdout_not_evaluable) take a separate ``unscored_note``. An unscored run still spends the family's
# one permanent slot, so a history end must be readable on that row as well. A month-end cutoff early in the standard
# world leaves too few account sessions to score, which is the unscored scenario built below with a tmp ledger.

_HE_UNSCORED_CUT = "2010-02-26"          # a scheduled month-end about two years into the standard frames


def _he_outcomes(ledger_path):
    return [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]


def test_an_unscored_real_run_with_a_history_end_carries_the_clause_and_the_cutoff_in_its_ledger_note(
        tmp_path, cftc_cache):
    assert pd.Timestamp(_HE_UNSCORED_CUT) in mod.month_end_sessions(
        pd.Timestamp(_HE_UNSCORED_CUT) - pd.Timedelta(days=20), pd.Timestamp(_HE_UNSCORED_CUT))
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, _HE_UNSCORED_CUT, dry_run=False)
    assert "reason_code" not in result, result.get("reason")
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    assert result["account_end"] == _HE_UNSCORED_CUT
    outcomes = _he_outcomes(ledger_path)
    assert len(outcomes) == 1 and outcomes[0]["stage"] == "insufficient_evidence"
    note = outcomes[0]["note"]
    assert note is not None                            # an unscored row under a history end carries a note
    assert hc.HISTORY_END_MARKER in note and note.count(hc.HISTORY_END_MARKER) == 1
    assert note == hc.history_end_clause(result["history_end"])       # the clause verbatim, nothing else on this row
    assert _HE_UNSCORED_CUT in note and f"permanently ends at {_HE_UNSCORED_CUT}" in note
    assert "cannot be re-scored on longer history" in note and "The setting added, moved or repaired no bar" in note
    # the scored-stage note lead is never added to an unscored row
    assert "cost model:" not in note and not note.startswith("cash_convention=")


def test_an_unscored_real_run_without_a_history_end_still_has_no_ledger_note(tmp_path, cftc_cache):
    """Default off: the unscored row carries no note."""
    frames = _frames(n_legs=int(_sessions(4000).get_loc(pd.Timestamp(_HE_UNSCORED_CUT))) + 1 - _LEAD)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, None, dry_run=False, frames=frames)
    assert "reason_code" not in result, result.get("reason")
    assert result["verdict"] == "not_scoreable" and "history_end" not in result
    outcomes = _he_outcomes(ledger_path)
    assert len(outcomes) == 1 and outcomes[0]["stage"] == "insufficient_evidence" and outcomes[0]["note"] is None


@pytest.fixture
def record_spy(monkeypatch):
    """Every ``record_abi1_outcome`` call of a run as ``(args, kwargs)``, delegating to the real recorder underneath
    (which already sits under this file's ledger-path guard)."""
    calls = []
    inner = driver.abi1_driver.record_abi1_outcome

    def spy(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return inner(*args, **kwargs)
    monkeypatch.setattr(driver.abi1_driver, "record_abi1_outcome", spy)
    return calls


_DEFAULT_RECORDER_KEYWORDS = {"catalog_entry", "instrument_set", "cash_convention", "benchmark", "tax_policy",
                             "ticker_label", "extra_note", "signal_source_path", "signal_source_sha256"}


def test_the_recorder_is_given_the_unscored_note_with_the_flag_and_it_equals_the_note_clause(
        tmp_path, cftc_cache, record_spy):
    result, _, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), dry_run=False)
    assert "reason_code" not in result, result.get("reason")
    assert len(record_spy) == 1
    kwargs = record_spy[0][1]
    clause = hc.history_end_clause(result["history_end"])
    assert kwargs["unscored_note"] == clause                                   # exactly the clause, nothing more
    assert kwargs["extra_note"].startswith(clause + " cost model:")           # the scored note keeps its own lead
    assert set(kwargs) == _DEFAULT_RECORDER_KEYWORDS | {"unscored_note"}


def test_the_recorder_is_not_given_an_unscored_note_without_the_flag(tmp_path, cftc_cache, record_spy):
    result, _, _ = _run_he(tmp_path, cftc_cache, None, dry_run=False)
    assert "reason_code" not in result, result.get("reason")
    assert len(record_spy) == 1
    kwargs = record_spy[0][1]
    assert "unscored_note" not in kwargs                                       # not even a None: the default call
    assert set(kwargs) == _DEFAULT_RECORDER_KEYWORDS
    assert hc.HISTORY_END_MARKER not in kwargs["extra_note"]


# ---- (c) prefix invariance

def test_a_month_end_history_end_scores_exactly_the_prefix_of_the_untrimmed_account(
        tmp_path, cftc_cache, he_untrimmed, main_evaluations):
    _, full_curve, full_scored = he_untrimmed
    result, _, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END))
    curve, scored = main_evaluations[0]
    assert curve["strategy_daily"].index[-1] == _HE_MONTH_END and result["account_end"] == _he_iso(_HE_MONTH_END)
    exact = dict(check_exact=True, check_freq=False)
    pd.testing.assert_series_equal(curve["strategy_daily"], full_curve["strategy_daily"].loc[:_HE_MONTH_END], **exact)
    pd.testing.assert_series_equal(curve["overlay_mask"], full_curve["overlay_mask"].loc[:_HE_MONTH_END], **exact)
    pd.testing.assert_series_equal(scored["account_daily"], full_scored["account_daily"].loc[:_HE_MONTH_END], **exact)
    trades = full_curve["trades_df"]
    finished = trades[trades["exit_date"] <= _HE_MONTH_END].reset_index(drop=True)
    assert len(finished) > 20                              # a real history of windows, not a trivial prefix
    pd.testing.assert_frame_equal(curve["trades_df"].reset_index(drop=True), finished, check_exact=True)
    assert curve["leg_by_trade"] == full_curve["leg_by_trade"][:len(finished)]


def test_a_history_end_inside_a_window_scores_the_untrimmed_prefix_up_to_the_early_closed_last_session(
        tmp_path, cftc_cache, he_untrimmed, main_evaluations):
    _, full_curve, _ = he_untrimmed
    result, _, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MID_WINDOW))
    curve, _ = main_evaluations[0]
    assert curve["strategy_daily"].index[-1] == _HE_MID_WINDOW
    exact = dict(check_exact=True, check_freq=False)
    # the open window is closed early at the last close, which can change that last session (and only it)
    pd.testing.assert_series_equal(curve["strategy_daily"].iloc[:-1],
                                   full_curve["strategy_daily"].loc[:_HE_MID_WINDOW].iloc[:-1], **exact)
    trades = full_curve["trades_df"]
    finished = trades[trades["exit_date"] <= _HE_MID_WINDOW].reset_index(drop=True)
    assert len(curve["trades_df"]) == len(finished) + 1                          # plus the early-closed final window
    pd.testing.assert_frame_equal(curve["trades_df"].reset_index(drop=True).iloc[:-1], finished, check_exact=True)
    assert result["overlay_diagnostics"]["final_window_truncated"] is True


# ---- (b) a hole that refuses today is scoreable with a history end before it

@pytest.mark.parametrize("shape,code", [("one_series", "calendar_mismatch"),
                                        ("every_series", "missing_exchange_session")])
def test_an_interior_vendor_hole_that_refuses_by_default_is_scoreable_with_a_history_end_before_it(
        tmp_path, cftc_cache, main_evaluations, shape, code):
    frames = _frames()
    index = frames["DBC"].index
    gone, before = index[1200], index[1199]
    holed = ({s: f.drop(gone) for s, f in frames.items()} if shape == "every_series"
             else {**frames, "DBC": frames["DBC"].drop(gone)})
    refused, _, _ = _run_he(_mk(tmp_path, "today"), cftc_cache, None, frames=holed)
    assert refused["reason_code"] == code and _he_iso(gone) in refused["missing_dates"]
    main_evaluations.clear()
    result, ledger_path, _ = _run_he(_mk(tmp_path, "trimmed"), cftc_cache, _he_iso(before), frames=holed)
    assert "reason_code" not in result, result.get("reason")
    assert result["account_end"] == _he_iso(before) and set(result["last_bar_dates"].values()) == {_he_iso(before)}
    assert main_evaluations[0][0]["strategy_daily"].index[-1] == before          # the account ends exactly at the cutoff
    spy_dropped = result["history_end"]["by_symbol"]["SPY"]["dropped_dates"]
    assert (_he_iso(gone) in spy_dropped) == (shape == "one_series")             # SPY has the bar only in that shape
    assert len(spy_dropped) > 100 and not os.path.exists(ledger_path)


def test_a_series_short_of_the_trailing_sessions_is_scoreable_with_a_history_end_at_its_last_bar(
        tmp_path, cftc_cache):
    frames = _frames()
    short = {**frames, "DBC": frames["DBC"].iloc[:-3]}
    last = short["DBC"].index[-1]
    refused, _, _ = _run_he(_mk(tmp_path, "today"), cftc_cache, None, frames=short)
    assert refused["reason_code"] == "unequal_trailing_coverage"
    result, ledger_path, _ = _run_he(_mk(tmp_path, "trimmed"), cftc_cache, _he_iso(last), frames=short)
    assert "reason_code" not in result, result.get("reason")
    assert result["account_end"] == _he_iso(last)
    by_symbol = result["history_end"]["by_symbol"]
    assert by_symbol["DBC"]["dropped_sessions"] == 0 and by_symbol["DBC"]["dropped_dates"] == []
    assert by_symbol["TLT"]["dropped_sessions"] == 3 and by_symbol["SPY"]["dropped_sessions"] == 3
    assert result["history_end"]["dropped_dates"] == [_he_iso(d) for d in frames["TLT"].index[-3:]]
    assert not os.path.exists(ledger_path)


# ---- (e) refusals: machine-readable reason code, one standard shape (mining.history_cutoff.refusal_payload).
# A malformed or non-session cutoff is refused at the door: nothing is fetched or written. An uncovered cutoff is
# refused after the load step: no ledger row, result file or report, but the per-run OHLC cache stays (see (f)).

@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("setting", _HE_INVALID, ids=_HE_INVALID_IDS)
def test_a_malformed_history_end_is_refused_before_anything_is_fetched_or_written(
        tmp_path, cftc_cache, setting, dry_run):
    loader = _CountingLoader()
    result, ledger_path, research_root = _run_he(tmp_path, cftc_cache, setting, dry_run=dry_run, loader=loader)
    assert result["reason_code"] == hc.REASON_INVALID == "history_end_invalid" and result["family_id"] == _FAMILY
    assert result["reason"] and "verdict" not in result and "stage" not in result
    assert list(result) == _HE_SHAPE and result["history_end_requested"] == str(setting) and result["symbols"] == []
    assert loader.calls == [] and not os.path.exists(ledger_path) and not os.path.exists(research_root)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("setting", _HE_NOT_SESSIONS)
def test_a_cutoff_that_is_not_an_nyse_session_is_refused_before_anything_is_fetched_or_written(
        tmp_path, cftc_cache, setting, dry_run):
    loader = _CountingLoader()
    result, ledger_path, research_root = _run_he(tmp_path, cftc_cache, setting, dry_run=dry_run, loader=loader)
    assert result["reason_code"] == "history_end_not_a_session" and result["history_end_requested"] == setting
    assert "history_end" not in result                          # no provenance on a refusal; the raw setting is history_end_requested
    assert "verdict" not in result and "stage" not in result
    assert loader.calls == [] and not os.path.exists(ledger_path) and not os.path.exists(research_root)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("case,cutoff,expected", [
    ("after_the_data", "2016-09-01", ["DBC", "SPY", "TLT", "^IRX"]),
    ("before_the_legs_list", None, ["DBC", "TLT"]),
    ("hole_at_the_cutoff_in_one_leg", _he_iso(_HE_MONTH_END), ["DBC"]),
    ("hole_at_the_cutoff_in_the_financing_series", _he_iso(_HE_MONTH_END), ["^IRX"])])
def test_a_cutoff_that_not_every_series_covers_is_refused_and_names_the_series(
        tmp_path, cftc_cache, case, cutoff, expected, dry_run):
    frames = _frames()
    if cutoff is None:
        cutoff = _he_iso(frames["SPY"].index[100])                       # SPY and ^IRX have it, the legs list later
    if case == "hole_at_the_cutoff_in_one_leg":
        frames["DBC"] = frames["DBC"].drop(_HE_MONTH_END)
    if case == "hole_at_the_cutoff_in_the_financing_series":
        frames["^IRX"] = frames["^IRX"].drop(_HE_MONTH_END)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, cutoff, dry_run=dry_run, frames=frames)
    assert result["reason_code"] == hc.REASON_NOT_COVERED == "history_end_not_covered" and result["family_id"] == _FAMILY
    assert result["symbols"] == expected and all(s in result["reason"] for s in expected)
    assert "verdict" not in result and "stage" not in result and not os.path.exists(ledger_path)
    json.dumps(result, allow_nan=False)


# ---- (f) the trim never reaches the OHLC cache

def test_a_trimmed_run_leaves_the_ohlc_cache_holding_the_full_vendor_history(tmp_path, cftc_cache):
    frames = _frames(tail=5)
    result, _, research_root = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), frames=frames, run_id="coi-cache")
    assert "reason_code" not in result and set(result["ohlc_sources"].values()) == {"fetched"}
    assert frames["DBC"].index[-1] > _HE_MONTH_END
    _he_assert_cache_is_full_history(research_root, "coi-cache", frames)


def test_a_refused_cutoff_also_leaves_the_full_history_in_the_ohlc_cache(tmp_path, cftc_cache):
    frames = _frames()
    result, ledger_path, research_root = _run_he(tmp_path, cftc_cache, "2016-09-01", frames=frames, run_id="coi-cache2")
    assert result["reason_code"] == "history_end_not_covered" and not os.path.exists(ledger_path)
    _he_assert_cache_is_full_history(research_root, "coi-cache2", frames)


def test_a_later_untrimmed_run_reuses_the_untouched_cache_and_scores_the_full_history(tmp_path, cftc_cache):
    _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), run_id="coi-reuse")
    later, _, _ = _run_he(tmp_path, cftc_cache, None, run_id="coi-reuse")
    assert set(later["ohlc_sources"].values()) == {"cache"}
    assert later["account_end"] == "2016-08-31" and later["last_bar_dates"]["DBC"] == "2016-08-31"


@pytest.mark.parametrize("second_setting", [None, "2015-01-30"])
def test_a_real_run_after_a_trimmed_dry_run_still_refuses_the_cached_history_as_stale(
        tmp_path, cftc_cache, second_setting):
    _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), run_id="coi-stale")
    real, ledger_path, _ = _run_he(tmp_path, cftc_cache, second_setting, dry_run=False, run_id="coi-stale")
    assert real["reason_code"] == "stale_ohlc_cache" and real["cached_symbols"] == list(_HE_SERIES)
    assert "fresh --run-id" in real["reason"] and not os.path.exists(ledger_path)


# ---- (g) nothing dated after the cutoff can reach the score

def test_bars_after_the_cutoff_cannot_change_the_scored_result(tmp_path, cftc_cache, main_evaluations):
    frames = _frames(tail=5)
    poisoned = _he_poisoned_after(frames, _HE_MID_WINDOW)
    cut = _he_iso(_HE_MID_WINDOW)
    clean, _, _ = _run_he(_mk(tmp_path, "clean"), cftc_cache, cut, frames=frames)
    clean_curve = main_evaluations[0][0]
    main_evaluations.clear()
    dirty, _, _ = _run_he(_mk(tmp_path, "poisoned"), cftc_cache, cut, frames=poisoned)
    dirty_curve = main_evaluations[0][0]
    assert "reason_code" not in clean and "reason_code" not in dirty
    as_json = lambda value: json.dumps(driver.batch5._json_safe(value), sort_keys=True)          # noqa: E731
    assert as_json(dirty) == as_json(clean)                     # the whole result, every number and every date
    pd.testing.assert_series_equal(dirty_curve["strategy_daily"], clean_curve["strategy_daily"], check_exact=True)
    assert clean["account_end"] == cut
    # the poison is real: ending ONE session later lets a single poisoned bar in, and the account's last day then differs
    later = _he_iso(frames["DBC"].index[frames["DBC"].index.get_loc(_HE_MID_WINDOW) + 1])
    main_evaluations.clear()
    _run_he(_mk(tmp_path, "later_clean"), cftc_cache, later, frames=frames)
    later_clean = main_evaluations[0][0]["strategy_daily"]
    main_evaluations.clear()
    _run_he(_mk(tmp_path, "later_poisoned"), cftc_cache, later, frames=poisoned)
    later_poisoned = main_evaluations[0][0]["strategy_daily"]
    assert later_clean.index[-1] == later_poisoned.index[-1] == pd.Timestamp(later)
    assert later_poisoned.iloc[-1] != later_clean.iloc[-1]
    pd.testing.assert_series_equal(later_poisoned.iloc[:-1], later_clean.iloc[:-1], check_exact=True)


def test_reports_dated_after_the_cutoff_cannot_change_the_scored_account(tmp_path, cftc_cache, main_evaluations):
    """The open-interest reports are not trimmed (only price histories are); a decision at or before the cutoff can
    read only reports usable by then, so poisoning every report as of a later date changes nothing that is scored."""
    def poisoned_cache(name, start, end):
        rows = _oi_rows()
        for entries in rows.values():
            for row in entries:
                if start < pd.Timestamp(row["report_date_as_yyyy_mm_dd"]) <= end:
                    row["open_interest_all"] = str(int(row["open_interest_all"]) * 7)
        return _write_cache(_mk(tmp_path, name), rows)

    cut = _he_iso(_HE_MID_WINDOW)
    clean, _, _ = _run_he(_mk(tmp_path, "clean"), cftc_cache, cut)
    clean_curve = main_evaluations[0][0]
    main_evaluations.clear()
    dirty, _, _ = _run_he(_mk(tmp_path, "poisoned"), poisoned_cache("cftc_after", _HE_MID_WINDOW, pd.Timestamp.max),
                          cut)
    dirty_curve = main_evaluations[0][0]
    assert "reason_code" not in clean and "reason_code" not in dirty
    assert (dirty["overlay_diagnostics"]["cftc_cache"]["fingerprint_sha256"]
            != clean["overlay_diagnostics"]["cftc_cache"]["fingerprint_sha256"])
    pd.testing.assert_series_equal(dirty_curve["strategy_daily"], clean_curve["strategy_daily"], check_exact=True)
    for key in ("verdict", "dsr_value", "sharpe_daily_excess", "n_sessions", "n_trades", "account_end"):
        assert dirty[key] == clean[key], key
    assert dirty["overlay_diagnostics"]["state_diagnostics"] == clean["overlay_diagnostics"]["state_diagnostics"]
    # the poison is real: the same change to reports BEFORE the cutoff does move the account
    main_evaluations.clear()
    _run_he(_mk(tmp_path, "before"), poisoned_cache("cftc_before", pd.Timestamp("2013-01-01"), pd.Timestamp("2014-06-30")),
            cut)
    assert not main_evaluations[0][0]["strategy_daily"].equals(clean_curve["strategy_daily"])


# ---- every existing check keeps running on the trimmed frames

def test_a_real_run_ending_inside_a_window_is_still_refused_as_a_truncated_final_window(tmp_path, cftc_cache):
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MID_WINDOW), dry_run=False)
    assert result["reason_code"] == "truncated_final_window" and "verdict" not in result
    assert not os.path.exists(ledger_path)


def test_a_dry_run_ending_inside_a_window_reports_the_truncation_and_is_not_refused(tmp_path, cftc_cache):
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MID_WINDOW))
    assert "reason_code" not in result and result["overlay_diagnostics"]["final_window_truncated"] is True
    assert result["account_end"] == _he_iso(_HE_MID_WINDOW) and not os.path.exists(ledger_path)


def test_a_real_run_ending_exactly_on_a_scheduled_month_end_is_not_refused_as_a_truncated_window(
        tmp_path, cftc_cache):
    assert _HE_MONTH_END in mod.month_end_sessions(_HE_MONTH_END - pd.Timedelta(days=20), _HE_MONTH_END)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), dry_run=False)
    assert "reason_code" not in result
    diag = result["overlay_diagnostics"]
    assert diag["window_entered_on_last_bar"] is True and diag["final_window_truncated"] is False
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


def test_the_calendar_checks_still_run_on_the_trimmed_frames(tmp_path, cftc_cache):
    """A hole BEFORE the cutoff is not repaired by a history end: it is still refused with its own reason code."""
    frames = _frames()
    gone = frames["DBC"].index[900]
    assert gone < _HE_MONTH_END
    holed = {s: f.drop(gone) for s, f in frames.items()}
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), frames=holed)
    assert result["reason_code"] == "missing_exchange_session" and result["missing_dates"] == [_he_iso(gone)]
    assert "verdict" not in result and not os.path.exists(ledger_path)


# ---- a later synthetic world: a one-day vendor gap near the data end, cutoff the session before it

_LATE_HOLE = pd.Timestamp("2026-09-22")


@pytest.fixture(scope="module")
def late_world(tmp_path_factory):
    """Synthetic sessions 2008-01-02 to 2026-09-24 on the real calendar, an open-interest cache to 2026-09-22 and a
    one-day gap in two series: DBC and ^IRX have no bar for 2026-09-22 while TLT and SPY do, and every series has
    2026-09-23 and 2026-09-24."""
    sessions = pd.DatetimeIndex(treasury.exchange_sessions(pd.Timestamp("2008-01-02"), pd.Timestamp("2026-09-24")))
    leg_idx = sessions[_LEAD:]
    irx = np.linspace(0.5, 3.0, len(sessions))
    full = {"SPY": _ohlc(_walk(sessions, 5, 0.01, 0.0004, 100.0), sessions),
            "DBC": _ohlc(_walk(leg_idx, 6, 0.012, 0.0009, 25.0), leg_idx),
            "TLT": _ohlc(_walk(leg_idx, 7, 0.009, 0.0007, 110.0), leg_idx),
            "^IRX": pd.DataFrame({"Open": irx, "High": irx, "Low": irx, "Close": irx}, index=sessions)}
    holed = {**full, "DBC": full["DBC"].drop(_LATE_HOLE), "^IRX": full["^IRX"].drop(_LATE_HOLE)}
    return holed, _write_cache(tmp_path_factory.mktemp("late_cftc"), _oi_rows(end="2026-09-22"))


def test_the_vendor_gap_refuses_by_default_and_a_dry_run_with_a_history_end_of_2026_09_21_scores_the_prefix(
        tmp_path, late_world):
    holed, cache = late_world
    refused, _, _ = _run_he(_mk(tmp_path, "today"), cache, None, frames=holed)
    assert refused["reason_code"] == "calendar_mismatch" and "2026-09-22" in refused["reason"]
    result, ledger_path, _ = _run_he(_mk(tmp_path, "trimmed"), cache, "2026-09-21", frames=holed)
    assert "reason_code" not in result, result.get("reason")
    assert result["account_end"] == "2026-09-21" and set(result["last_bar_dates"].values()) == {"2026-09-21"}
    assert result["raw_last_bar_dates"] == {s: "2026-09-24" for s in _HE_SERIES}
    provenance = result["history_end"]
    assert provenance["dropped_dates"] == ["2026-09-22", "2026-09-23", "2026-09-24"]
    assert provenance["by_symbol"]["DBC"]["dropped_sessions"] == 2 and provenance["by_symbol"]["TLT"]["dropped_sessions"] == 3
    assert "2026-09-22..2026-09-24 (3 distinct dates)" in result["cost_model_note"]
    # the existing final-window check still ran on the trimmed frames: 2026-09-21 lies inside the September window
    assert result["overlay_diagnostics"]["final_window_truncated"] is True
    assert not os.path.exists(ledger_path)


def test_a_real_run_with_a_history_end_of_2026_09_21_is_still_refused_as_a_truncated_final_window(
        tmp_path, late_world):
    holed, cache = late_world
    result, ledger_path, _ = _run_he(tmp_path, cache, "2026-09-21", dry_run=False, frames=holed)
    assert result["reason_code"] == "truncated_final_window" and "verdict" not in result
    assert not os.path.exists(ledger_path)


def test_a_real_run_with_a_history_end_on_the_2026_08_31_month_end_is_scored_and_recorded(tmp_path, late_world):
    holed, cache = late_world
    result, ledger_path, _ = _run_he(tmp_path, cache, "2026-08-31", dry_run=False, frames=holed)
    assert "reason_code" not in result, result.get("reason")
    diag = result["overlay_diagnostics"]
    assert diag["window_entered_on_last_bar"] is True and diag["final_window_truncated"] is False
    assert result["account_end"] == "2026-08-31"
    expected_dropped = sorted({_he_iso(d) for f in holed.values() for d in f.index[f.index > pd.Timestamp("2026-08-31")]})
    assert result["history_end"]["dropped_dates"] == expected_dropped and expected_dropped[0] == "2026-09-01"
    rows = trial_ledger.read_all(ledger_path)
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and hc.history_end_clause(result["history_end"]) in outcomes[0]["note"]
    assert "permanently ends at 2026-08-31" in outcomes[0]["note"]
    assert [r["holdout_end"] for r in rows if r.get("event") == "first_test"] == ["2026-08-31"]


# ---- the real-run guard: a history end must not spend the one-shot slot by mistake
#
# A REAL run refuses (history_end_after_tested_date / history_end_drop_too_large, via mining.history_cutoff.
# check_real_run_guard) a cutoff later than its tested date or one that drops more than ``max_history_drop`` sessions
# (default hc.MAX_UNACCEPTED_DROP_SESSIONS) from any series; a DRY run never refuses but predicts the same guard in
# ``history_end_guard_would_refuse``. Every history-end refusal has the ONE standard shape of hc.refusal_payload.

_HE_END = "2016-08-31"          # the standard world's last session and a scheduled month-end; only ^IRX runs past it
_HE_LIMIT = hc.MAX_UNACCEPTED_DROP_SESSIONS
_HE_SHAPE = ["reason_code", "reason", "history_end_requested", "symbols", "family_id"]


def _he_no_run_output(research_root, run_id):
    """No result, payload or report file under the run directory (the per-run OHLC cache the load step wrote stays)."""
    run_dir = os.path.join(research_root, run_id)
    return not os.path.isdir(run_dir) or [n for n in os.listdir(run_dir) if n.endswith(".json")] == []


def test_compute_and_record_takes_a_max_history_drop_defaulting_to_the_shared_constant():
    import inspect
    parameter = inspect.signature(driver.compute_and_record).parameters["max_history_drop"]
    assert parameter.default == hc.MAX_UNACCEPTED_DROP_SESSIONS and parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_real_run_dropping_more_than_the_default_limit_is_refused_in_the_standard_shape_and_writes_nothing(
        tmp_path, cftc_cache):
    frames = _frames(tail=_HE_LIMIT + 2)
    result, ledger_path, research_root = _run_he(tmp_path, cftc_cache, _HE_END, dry_run=False, frames=frames,
                                                 max_history_drop=_HE_DEFAULT_DROP)
    assert list(result) == _HE_SHAPE
    assert result["reason_code"] == hc.REASON_DROP_TOO_LARGE == "history_end_drop_too_large"
    assert result["family_id"] == _FAMILY and result["history_end_requested"] == _HE_END
    assert result["symbols"] == ["^IRX"]                                    # only the financing series ran past
    assert f"{_HE_LIMIT + 2} sessions from ^IRX" in result["reason"] and f"more than the {_HE_LIMIT}" in result["reason"]
    assert "pass --max-history-drop N to accept a larger drop" in result["reason"]
    assert "verdict" not in result and "stage" not in result and "history_end" not in result
    assert not os.path.exists(ledger_path) and _he_no_run_output(research_root, "coi-test")
    _he_assert_cache_is_full_history(research_root, "coi-test", frames)     # the load step's cache is kept, untrimmed
    json.dumps(result, allow_nan=False)


def test_a_real_run_dropping_exactly_the_stated_limit_is_scored_and_one_session_more_is_refused(tmp_path, cftc_cache):
    frames = _frames(tail=5)
    refused, ledger_refused, _ = _run_he(_mk(tmp_path, "refused"), cftc_cache, _HE_END, dry_run=False, frames=frames,
                                         max_history_drop=4)
    assert refused["reason_code"] == "history_end_drop_too_large" and refused["symbols"] == ["^IRX"]
    assert "5 sessions from ^IRX" in refused["reason"] and "more than the 4" in refused["reason"]
    assert not os.path.exists(ledger_refused)
    accepted, ledger_path, _ = _run_he(_mk(tmp_path, "accepted"), cftc_cache, _HE_END, dry_run=False, frames=frames,
                                       max_history_drop=5)
    assert "reason_code" not in accepted, accepted.get("reason")
    assert accepted["history_end_requested"] == _HE_END and accepted["history_end"]["history_end"] == _HE_END
    assert accepted["history_end"]["by_symbol"]["^IRX"]["dropped_sessions"] == 5
    assert "history_end_guard_would_refuse" not in accepted                 # only a dry run predicts the guard
    outcomes = _he_outcomes(ledger_path)
    assert len(outcomes) == 1 and hc.history_end_clause(accepted["history_end"]) in outcomes[0]["note"]


def test_a_real_run_whose_cutoff_is_later_than_its_tested_date_is_refused_and_one_on_the_tested_date_is_scored(
        tmp_path, cftc_cache):
    cut = _he_iso(_HE_MONTH_END)
    refused, ledger_refused, research_root = _run_he(
        _mk(tmp_path, "early"), cftc_cache, cut, dry_run=False, tested_date="2015-01-29",
        quotes_overrides={"sampled_at_local": "2015-01-29T10:00:00"})
    assert list(refused) == _HE_SHAPE and refused["reason_code"] == "history_end_after_tested_date"
    assert refused["symbols"] == [] and refused["history_end_requested"] == cut
    assert "later than the run's tested date 2015-01-29" in refused["reason"]
    assert not os.path.exists(ledger_refused) and _he_no_run_output(research_root, "coi-test")
    # the after-tested-date check comes first: a run that also drops too much is refused for the tested date
    both, _, _ = _run_he(_mk(tmp_path, "both"), cftc_cache, cut, dry_run=False, tested_date="2015-01-29",
                         max_history_drop=_HE_DEFAULT_DROP, quotes_overrides={"sampled_at_local": "2015-01-29T10:00:00"})
    assert both["reason_code"] == "history_end_after_tested_date"
    equal, ledger_path, _ = _run_he(_mk(tmp_path, "equal"), cftc_cache, cut, dry_run=False, tested_date=cut,
                                    quotes_overrides={"sampled_at_local": f"{cut}T10:00:00"})
    assert "reason_code" not in equal, equal.get("reason")                  # equal is fine
    assert len(_he_outcomes(ledger_path)) == 1


def test_a_dry_run_predicts_the_guard_without_refusing_and_the_real_run_then_says_the_same(tmp_path, cftc_cache):
    frames = _frames(tail=_HE_LIMIT + 2)
    would, ledger_path, _ = _run_he(_mk(tmp_path, "would"), cftc_cache, _HE_END, frames=frames,
                                    max_history_drop=_HE_DEFAULT_DROP)
    assert "reason_code" not in would and would["verdict"] and not os.path.exists(ledger_path)
    message = would["history_end_guard_would_refuse"]
    assert isinstance(message, str) and "^IRX" in message
    assert "pass --max-history-drop N to accept a larger drop" in message
    real, _, _ = _run_he(_mk(tmp_path, "real"), cftc_cache, _HE_END, dry_run=False, frames=frames,
                         max_history_drop=_HE_DEFAULT_DROP)
    # a rehearsal predicts it: the real reason is the same words plus the cache-kept sentence
    assert real["reason_code"] == "history_end_drop_too_large" and real["reason"].startswith(message)
    assert "retry with a fresh --run-id" in real["reason"]
    fine, _, _ = _run_he(_mk(tmp_path, "fine"), cftc_cache, _HE_END, frames=frames, max_history_drop=_HE_LIMIT + 2)
    assert "history_end_guard_would_refuse" in fine and fine["history_end_guard_would_refuse"] is None
    assert fine["history_end_requested"] == _HE_END
    early, _, _ = _run_he(_mk(tmp_path, "early"), cftc_cache, _he_iso(_HE_MONTH_END), tested_date="2015-01-29")
    assert "reason_code" not in early
    assert "later than the run's tested date 2015-01-29" in early["history_end_guard_would_refuse"]
    off, _, _ = _run_he(_mk(tmp_path, "off"), cftc_cache, None)
    assert "history_end_guard_would_refuse" not in off and "history_end_requested" not in off


def test_the_guard_is_given_the_provenance_the_runs_tested_date_and_the_stated_limit(tmp_path, cftc_cache, monkeypatch):
    calls = []
    real_guard = driver.history_cutoff.check_real_run_guard

    def spy(provenance, **kwargs):
        calls.append((provenance, kwargs))
        return real_guard(provenance, **kwargs)
    monkeypatch.setattr(driver.history_cutoff, "check_real_run_guard", spy)
    result, _, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), dry_run=False, tested_date="2026-09-24",
                           max_history_drop=777777)
    assert "reason_code" not in result, result.get("reason")
    assert len(calls) == 1 and calls[0][1] == {"tested_date": "2026-09-24", "max_drop": 777777}
    assert calls[0][0] == result["history_end"]


@pytest.mark.parametrize("bad", [-1, True, "5", 2.5])
def test_a_real_run_with_an_invalid_max_history_drop_is_refused_as_history_end_invalid(tmp_path, cftc_cache, bad):
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MONTH_END), dry_run=False, max_history_drop=bad)
    assert list(result) == _HE_SHAPE and result["reason_code"] == "history_end_invalid"
    assert "max_drop" in result["reason"] and not os.path.exists(ledger_path)


def test_an_invalid_max_history_drop_is_ignored_when_the_setting_is_off(tmp_path, cftc_cache):
    result, _, _ = _run_he(tmp_path, cftc_cache, None, max_history_drop=-1)
    assert "reason_code" not in result and result["verdict"]


def test_the_uncovered_cutoff_refusal_uses_the_standard_shape_and_says_the_cache_is_kept(tmp_path, cftc_cache):
    result, ledger_path, research_root = _run_he(tmp_path, cftc_cache, "2016-09-01", dry_run=False)
    assert list(result) == _HE_SHAPE and result["reason_code"] == "history_end_not_covered"
    assert result["symbols"] == ["DBC", "SPY", "TLT", "^IRX"] and result["history_end_requested"] == "2016-09-01"
    assert "OHLC cache" in result["reason"] and "is kept" in result["reason"] and "fresh --run-id" in result["reason"]
    assert not os.path.exists(ledger_path) and _he_no_run_output(research_root, "coi-test")


# ---- the open-interest state table under a history end: only the decisions that have happened

def test_under_a_history_end_the_state_table_is_built_from_the_month_end_decisions_that_happened_only(
        tmp_path, cftc_cache, monkeypatch):
    seen = []
    real_table = driver.mod.signal_state_table

    def spy(series_by_code, decision_dates):
        seen.append(list(pd.DatetimeIndex(decision_dates)))
        return real_table(series_by_code, decision_dates)
    monkeypatch.setattr(driver.mod, "signal_state_table", spy)
    result, _, _ = _run_he(tmp_path, cftc_cache, _he_iso(_HE_MID_WINDOW))      # inside February: the coming month-end
    assert "reason_code" not in result, result.get("reason")                    # is 2015-02-27, after the cutoff
    assert len(seen) == 1 and seen[0][-1] == pd.Timestamp("2015-01-30") and max(seen[0]) <= _HE_MID_WINDOW
    # with the setting OFF and the price data ending on the same day nothing changes: the coming month-end stays in
    seen.clear()
    frames = {symbol: frame.loc[:_HE_MID_WINDOW] for symbol, frame in _frames().items()}
    off, _, _ = _run_he(_mk(tmp_path, "off"), cftc_cache, None, frames=frames)
    assert "reason_code" not in off and seen[0][-1] == pd.Timestamp("2015-02-27")


def test_a_corrupt_report_usable_only_after_the_history_end_is_never_read_and_cannot_crash_the_run(
        tmp_path, cftc_cache, monkeypatch):
    """A report usable only after the cutoff would be read only by the COMING month-end's decision. The cache loader
    already rejects a non-positive or non-finite value, so the corruption is injected at the point of use: every read
    for a decision after the cutoff comes back non-positive (what ``aggregate_growth`` refuses with a ValueError)."""
    cut = _HE_MID_WINDOW
    real_read = cd.month_end_open_interest

    def corrupt_after_the_cutoff(series_by_code, decision_dates):
        values, used = real_read(series_by_code, decision_dates)
        values.loc[pd.DatetimeIndex(values.index) > cut] = -1.0
        return values, used
    clean, _, _ = _run_he(_mk(tmp_path, "clean"), cftc_cache, _he_iso(cut))
    monkeypatch.setattr(cd, "month_end_open_interest", corrupt_after_the_cutoff)
    dirty, _, _ = _run_he(_mk(tmp_path, "dirty"), cftc_cache, _he_iso(cut))
    as_json = lambda value: json.dumps(driver.batch5._json_safe(value), sort_keys=True)          # noqa: E731
    assert "reason_code" not in dirty and as_json(dirty) == as_json(clean)
    # the corruption is real: with the setting off the same data end decides the coming month-end and the run crashes
    frames = {symbol: frame.loc[:cut] for symbol, frame in _frames().items()}
    with pytest.raises(ValueError, match="open interest must be positive"):
        _run_he(_mk(tmp_path, "off"), cftc_cache, None, frames=frames)


def test_the_state_table_rows_up_to_the_cutoff_are_the_same_with_or_without_the_coming_month_end(cftc_cache):
    """Restricting the decisions to those that happened changes no row a scored window reads: the state at a
    month-end uses only reports usable by then and the expanding median only earlier months."""
    snapshot = cd.load_cached_open_interest(cftc_cache)
    every = mod.month_end_sessions(pd.Timestamp(driver.STATE_HISTORY_START), _HE_MID_WINDOW)
    happened = every[every <= _HE_MID_WINDOW]
    assert every[-1] == pd.Timestamp("2015-02-27") and happened[-1] == pd.Timestamp("2015-01-30")
    with_coming = mod.signal_state_table(snapshot.series, every)
    without = mod.signal_state_table(snapshot.series, happened)
    pd.testing.assert_frame_equal(with_coming.loc[happened], without, check_exact=True)


def test_the_scored_note_says_the_cftc_cache_was_not_trimmed_and_that_no_later_report_was_read(tmp_path, cftc_cache):
    cut = _he_iso(_HE_MONTH_END)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, cut, dry_run=False)
    note = _he_outcomes(ledger_path)[0]["note"]
    for text in (note, result["cost_model_note"]):
        assert "the CFTC open-interest cache was not trimmed" in text
        assert f"no report usable only after the history end {cut} was read" in text
    off, off_ledger, _ = _run_he(_mk(tmp_path, "off"), cftc_cache, None, dry_run=False)
    assert "was not trimmed" not in _he_outcomes(off_ledger)[0]["note"] and "was not trimmed" not in off["cost_model_note"]


# ---- cli

def _he_cli(tmp_path, cache, monkeypatch, extra_args, *, dry_run=True, loader=None, run_id="cli-he"):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=dry_run)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", loader or _loader(_frames()))
    argv = ["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
            "--research-root", research_root, "--cftc-cache-dir", cache, "--run-id", run_id]
    argv += ["--dry-run"] if dry_run else ["--tested-date", "2026-09-24"]
    return driver.main(argv + list(extra_args)), ledger_path, research_root


def test_the_cli_takes_a_history_end_and_writes_it_into_the_payload_file(tmp_path, cftc_cache, monkeypatch, capsys):
    code, ledger_path, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, ["--history-end", "2015-01-30"])
    assert code == 0 and not os.path.exists(ledger_path)
    output = os.path.join(research_root, "cli-he", "commodity-oi-cycle-overlay-dry-run.json")
    payload = json.load(open(output, encoding="utf-8"))
    # the same two key names in every driver: history_end is the provenance dict, history_end_requested the raw setting
    assert payload["history_end_requested"] == "2015-01-30"
    assert payload["history_end"]["history_end"] == "2015-01-30" and payload["history_end"]["opt_in"] is True
    assert payload["result"]["history_end_requested"] == "2015-01-30"
    assert payload["result"]["history_end"]["history_end"] == "2015-01-30"
    assert payload["result"]["account_end"] == "2015-01-30" and payload["result"]["raw_last_bar_dates"]["DBC"] == "2016-08-31"
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["history_end"]["history_end"] == "2015-01-30"


def test_the_cli_payload_has_no_history_end_key_without_the_flag(tmp_path, cftc_cache, monkeypatch):
    code, _, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, [])
    payload = json.load(open(os.path.join(research_root, "cli-he", "commodity-oi-cycle-overlay-dry-run.json"),
                             encoding="utf-8"))
    assert code == 0 and "history_end" not in payload and "history_end" not in payload["result"]
    for key in ("history_end_requested", "history_end_guard_would_refuse"):        # nothing of the setting when it is off
        assert key not in payload and key not in payload["result"], key


def test_the_cli_help_describes_the_history_end_option(capsys):
    with pytest.raises(SystemExit):
        driver.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--history-end" in text and "opt-in" in text and "opt-in history-end setting" in text
    assert "permanent" in text and "later sessions" in text
    assert "--max-history-drop" in text and "more than N sessions" in text


@pytest.mark.parametrize("setting", ["2015-1-30", "", "2014-06-28", "2014-07-04"])
def test_the_cli_refuses_a_bad_history_end_and_writes_nothing(tmp_path, cftc_cache, monkeypatch, capsys, setting):
    loader = _CountingLoader()
    code, ledger_path, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, ["--history-end", setting],
                                               dry_run=False, loader=loader)
    assert code == 2                      # the sibling drivers' exit status for a refused history end (not 0)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] in ("history_end_invalid", "history_end_not_a_session")
    assert "output" not in printed and loader.calls == []
    assert not os.path.exists(ledger_path) and not os.path.exists(research_root)


def test_the_cli_refuses_an_uncovered_cutoff_without_a_result_file_or_a_ledger_row(
        tmp_path, cftc_cache, monkeypatch, capsys):
    code, ledger_path, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, ["--history-end", "2016-09-01"],
                                               dry_run=False)
    assert code == 2                      # the sibling drivers' exit status for a refused history end (not 0)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "history_end_not_covered" and "output" not in printed
    assert not os.path.exists(ledger_path)
    run_dir = os.path.join(research_root, "cli-he")
    assert [name for name in os.listdir(run_dir) if name.endswith(".json")] == []       # no result or payload file


def test_the_cli_refuses_an_uncovered_cutoff_on_a_dry_run_with_exit_2_and_no_result_file(
        tmp_path, cftc_cache, monkeypatch, capsys):
    code, ledger_path, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, ["--history-end", "2016-09-01"])
    assert code == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "history_end_not_covered" and "output" not in printed
    assert not os.path.exists(ledger_path)
    run_dir = os.path.join(research_root, "cli-he")
    assert not os.path.exists(os.path.join(run_dir, "commodity-oi-cycle-overlay-dry-run.json"))
    assert [name for name in os.listdir(run_dir) if name.endswith(".json")] == []


@pytest.mark.parametrize("reason_code,expected_exit", [
    *[(code, 2) for code in hc.HISTORY_END_REASON_CODES],
    ("stale_ohlc_cache", 0), ("quotes_not_market_hours", 0), ("truncated_final_window", 0)])
def test_the_cli_exits_2_for_every_history_end_reason_code_and_0_for_any_other_refusal(
        tmp_path, monkeypatch, capsys, reason_code, expected_exit):
    """Only the six history-end reason codes (``hc.HISTORY_END_REASON_CODES``) change the exit status; every other
    refusal exits 0 (and still writes its payload file). With ``compute_and_record`` replaced here
    nothing was loaded, so a history-end refusal leaves no run directory at all; a real refusal after the load step
    keeps only the per-run OHLC cache that step wrote (no ledger row, no result file, no report)."""
    monkeypatch.setattr(driver, "compute_and_record",
                        lambda **kwargs: {"family_id": _FAMILY, "reason_code": reason_code, "reason": "fixture refusal"})
    research_root = tmp_path / "research"
    code = driver.main(["--quotes", str(tmp_path / "quotes.json"), "--dry-run", "--research-root", str(research_root),
                        "--run-id", "exit-status"])
    assert code == expected_exit
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == reason_code
    assert ("output" in printed) == (expected_exit == 0)
    assert os.path.exists(research_root) == (expected_exit == 0)


def test_the_cli_refuses_a_real_history_end_the_guard_refuses_with_exit_2_and_a_larger_limit_scores_it(
        tmp_path, cftc_cache, monkeypatch, capsys):
    frames = _frames(tail=_HE_LIMIT + 3)
    code, ledger_path, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, ["--history-end", _HE_END],
                                               dry_run=False, loader=_loader(frames))
    assert code == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert list(printed["result"]) == _HE_SHAPE and "output" not in printed
    assert printed["result"]["reason_code"] == "history_end_drop_too_large"
    assert printed["result"]["history_end_requested"] == _HE_END and printed["result"]["symbols"] == ["^IRX"]
    assert not os.path.exists(ledger_path) and _he_no_run_output(research_root, "cli-he")   # no row, file or report
    _he_assert_cache_is_full_history(research_root, "cli-he", frames)                       # the cache is kept
    code, ledger_path, _ = _he_cli(_mk(tmp_path, "accepted"), cftc_cache, monkeypatch,
                                   ["--history-end", _HE_END, "--max-history-drop", str(_HE_LIMIT + 3)],
                                   dry_run=False, loader=_loader(frames), run_id="cli-he-2")
    assert code == 0 and len(_he_outcomes(ledger_path)) == 1


def test_the_cli_dry_run_with_a_history_end_the_guard_would_refuse_exits_0_and_says_so(
        tmp_path, cftc_cache, monkeypatch):
    code, _, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, ["--history-end", "2015-01-30"])
    payload = json.load(open(os.path.join(research_root, "cli-he", "commodity-oi-cycle-overlay-dry-run.json"),
                             encoding="utf-8"))
    assert code == 0
    assert "max-history-drop" in payload["result"]["history_end_guard_would_refuse"]
    assert payload["result"]["history_end_requested"] == "2015-01-30" == payload["history_end_requested"]


@pytest.mark.parametrize("value", ["-1", "abc", "1.5", "", "True"])
def test_the_cli_rejects_a_max_history_drop_that_is_not_a_whole_number_of_sessions(capsys, value):
    with pytest.raises(SystemExit) as raised:
        driver.main(["--quotes", "unused.json", "--dry-run", "--max-history-drop", value])
    error = capsys.readouterr().err
    # argparse's own message for a value the option's type rejects (an unknown option would say "unrecognized")
    assert raised.value.code == 2 and "argument --max-history-drop:" in error and "unrecognized" not in error


# ---- documentation

def test_the_driver_docstring_states_the_guard_the_kept_cache_and_what_a_refusal_writes():
    text = " ".join(driver.__doc__.split())
    for phrase in ("--max-history-drop", "history_end_after_tested_date", "history_end_drop_too_large",
                   "history_end_guard_would_refuse", "history_end_requested", "no ledger row, no result file and no report",
                   "per-run OHLC cache", "fresh --run-id", "CFTC cache is not trimmed",
                   # the holdout pin is conditional: only a NEW family's first_test row ends at the cutoff
                   "a NEW family's first_test pin", "The pin is not unconditional",
                   "does nothing for a family that already has a first_test row"):
        assert phrase in text, phrase


def test_the_driver_docstring_states_the_opt_in_history_end_and_its_rules():
    text = " ".join(driver.__doc__.split())
    for phrase in ("OPT-IN HISTORY END", "opt-in history-end setting", "--history-end", "history_end_invalid",
                   "history_end_not_a_session", "history_end_not_covered", "in memory only", "PERMANENT",
                   "cannot be re-scored on longer history", "truncated_final_window", "unless the opt-in history end"):
        assert phrase in text, phrase


def test_the_driver_docstring_states_the_unscored_note_and_the_exit_status_of_a_refused_history_end():
    text = " ".join(driver.__doc__.split())
    for phrase in ("unscored_note", "not_scoreable", "holdout_not_evaluable", "exit status 2", "no ledger row",
                   # an unscored stage carries the history-end clause too
                   "the same clause is also passed as the recorder's ``unscored_note`` keyword"):
        assert phrase in text, phrase


# ---- the door checks the real-run guard's two inputs, and the truncated-window
# refusal names the history end.
#
# With a history end set, a malformed tested date or max_history_drop is refused at the very door (history_end_invalid,
# real and dry runs alike, before the catalog, quotes or any load), so it can never fail AFTER the load step wrote the
# per-run OHLC cache. With the setting off the door checks nothing of the sort.

_HE_BAD_TESTED_DATES = ["2026-9-25", "2026/09/25", "", "2026-09-25T10:00:00", " 2026-09-25", "2026-09-31", 2026, None]
_HE_BAD_DROPS = [-1, True, "5", 2.5, None]
_HE_DOOR_CUT = _he_iso(_HE_MONTH_END)


def _he_assert_door_refusal(result, loader, ledger_path, research_root, *, mentions):
    assert list(result) == _HE_SHAPE and result["reason_code"] == hc.REASON_INVALID == "history_end_invalid"
    assert result["family_id"] == _FAMILY and result["history_end_requested"] == _HE_DOOR_CUT
    assert result["symbols"] == [] and mentions in result["reason"]
    assert "OHLC cache" not in result["reason"]                    # a door refusal has no cache to keep
    assert "verdict" not in result and "stage" not in result and "history_end" not in result
    assert loader.calls == [] and not os.path.exists(ledger_path)
    assert not os.path.exists(research_root) and not os.path.exists(_he_cache_dir(research_root, "coi-test"))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad", _HE_BAD_TESTED_DATES, ids=repr)
def test_a_malformed_tested_date_with_a_history_end_is_refused_at_the_door_before_anything_is_written(
        tmp_path, cftc_cache, bad, dry_run):
    loader = _CountingLoader()
    result, ledger_path, research_root = _run_he(tmp_path, cftc_cache, _HE_DOOR_CUT, dry_run=dry_run, loader=loader,
                                                 tested_date=bad)
    _he_assert_door_refusal(result, loader, ledger_path, research_root, mentions="tested_date")


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad", _HE_BAD_DROPS, ids=repr)
def test_an_invalid_max_history_drop_with_a_history_end_is_refused_at_the_door_before_anything_is_written(
        tmp_path, cftc_cache, bad, dry_run):
    loader = _CountingLoader()
    result, ledger_path, research_root = _run_he(tmp_path, cftc_cache, _HE_DOOR_CUT, dry_run=dry_run, loader=loader,
                                                 max_history_drop=bad)
    _he_assert_door_refusal(result, loader, ledger_path, research_root, mentions="max_drop")


def test_the_door_is_given_the_runs_tested_date_and_the_stated_limit_before_any_load(tmp_path, cftc_cache, monkeypatch):
    loader = _CountingLoader()
    calls = []
    real_validate = driver.history_cutoff.validate_guard_arguments

    def spy(**kwargs):
        calls.append((kwargs, list(loader.calls)))               # what the loader had been asked for by then
        return real_validate(**kwargs)
    monkeypatch.setattr(driver.history_cutoff, "validate_guard_arguments", spy)
    result, _, _ = _run_he(tmp_path, cftc_cache, _HE_DOOR_CUT, loader=loader, tested_date="2026-09-24",
                           max_history_drop=777)
    assert "reason_code" not in result, result.get("reason")
    # check_real_run_guard validates the same two inputs again after the load, so there is a second call; the FIRST
    # call is the door's and must come before the loader was asked for anything
    assert calls and all(kwargs == {"tested_date": "2026-09-24", "max_drop": 777} for kwargs, _ in calls)
    assert calls[0][1] == [] and loader.calls


def test_a_malformed_cutoff_or_non_session_cutoff_is_still_reported_ahead_of_a_malformed_tested_date(
        tmp_path, cftc_cache):
    invalid, _, _ = _run_he(_mk(tmp_path, "invalid"), cftc_cache, "2015-1-30", tested_date="2026-9-25")
    assert invalid["reason_code"] == "history_end_invalid" and "history end must be a real calendar date" in invalid["reason"]
    not_a_session, _, _ = _run_he(_mk(tmp_path, "weekend"), cftc_cache, "2014-06-28", tested_date="2026-9-25")
    assert not_a_session["reason_code"] == "history_end_not_a_session"


def test_a_valid_tested_date_and_limit_pass_the_door_as_a_date_object_too(tmp_path, cftc_cache):
    result, _, _ = _run_he(tmp_path, cftc_cache, _HE_DOOR_CUT, tested_date=datetime.date(2026, 9, 24), max_history_drop=0)
    assert "reason_code" not in result, result.get("reason")


def test_with_no_history_end_a_malformed_tested_date_behaves_exactly_as_it_always_did(
        tmp_path, cftc_cache, standard_dry_run, monkeypatch):
    def boom(**kwargs):
        raise AssertionError("the door validated the guard's arguments with the setting off")
    monkeypatch.setattr(driver.history_cutoff, "validate_guard_arguments", boom)
    as_json = lambda value: json.dumps(driver.batch5._json_safe(value), sort_keys=True)          # noqa: E731
    # a dry run never reads the tested date: the result is byte-identical to the standard one
    dry, ledger_dry, root_dry = _run_he(_mk(tmp_path, "dry"), cftc_cache, None, tested_date="2026-9-25",
                                        max_history_drop=-1)
    assert as_json(dry) == as_json(standard_dry_run) and not os.path.exists(ledger_dry)
    # a real run is refused by the existing quote-capture check, which cannot parse the tested date
    loader = _CountingLoader()
    real, ledger_real, root_real = _run_he(_mk(tmp_path, "real"), cftc_cache, None, dry_run=False, loader=loader,
                                           tested_date="2026-9-25")
    assert real["reason_code"] == "quotes_not_market_hours" and "does not parse as a date" in real["reason"]
    assert loader.calls == [] and not os.path.exists(ledger_real)


def test_the_cli_refuses_a_malformed_tested_date_with_a_history_end_at_the_door_with_exit_2(
        tmp_path, cftc_cache, monkeypatch, capsys):
    for name, dry_run in (("dry", True), ("real", False)):
        loader = _CountingLoader()
        code, ledger_path, research_root = _he_cli(
            _mk(tmp_path, name), cftc_cache, monkeypatch,
            ["--history-end", _HE_DOOR_CUT, "--tested-date", "2026-9-25", "--max-history-drop", "1000000"],
            dry_run=dry_run, loader=loader)
        assert code == 2, name
        printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert printed["result"]["reason_code"] == "history_end_invalid" and "output" not in printed, name
        assert "tested_date" in printed["result"]["reason"], name
        assert loader.calls == [] and not os.path.exists(ledger_path) and not os.path.exists(research_root), name


def test_the_cli_without_a_history_end_still_runs_a_dry_run_with_a_malformed_tested_date(
        tmp_path, cftc_cache, monkeypatch):
    code, _, research_root = _he_cli(tmp_path, cftc_cache, monkeypatch, ["--tested-date", "2026-9-25"])
    assert code == 0
    assert os.path.exists(os.path.join(research_root, "cli-he", "commodity-oi-cycle-overlay-dry-run.json"))


# ---- the truncated_final_window refusal under a history end names the setting

# the flag-unset text, pinned exactly: a run without the setting reads exactly this
_TRUNCATED_TEXT_UNSET = ("the data ends inside an open month window (its scheduled exit lies after the last bar), so the "
                         "final block would be closed early at the last close instead of at the pre-registered "
                         "month-end; run a real score after the close of that month's last scheduled session")


def test_the_truncated_final_window_reason_without_a_history_end_is_the_ordinary_text(tmp_path, cftc_cache, state_table):
    frames, _ = _frames_cut_after_entry(state_table, 5)
    real, ledger_path = _run(tmp_path, cftc_cache, dry_run=False, synthetic=False, frames=frames)
    assert real["reason_code"] == "truncated_final_window" and real["reason"] == _TRUNCATED_TEXT_UNSET
    assert "history end" not in real["reason"] and not os.path.exists(ledger_path)


def test_the_truncated_final_window_reason_under_a_history_end_names_the_cutoff_and_the_way_out(tmp_path, cftc_cache):
    cut = _he_iso(_HE_MID_WINDOW)
    result, ledger_path, _ = _run_he(tmp_path, cftc_cache, cut, dry_run=False)
    assert result["reason_code"] == "truncated_final_window" and "verdict" not in result
    assert not os.path.exists(ledger_path)
    reason = " ".join(result["reason"].split())
    assert f"the history end {cut}" in reason and "--history-end" in reason
    assert "ended the data inside an open month window" in reason
    assert "a real score needs a cutoff that is exactly a scheduled month-end session" in reason
    assert "for example 2015-01-30" in reason                     # the last scheduled month-end before the cutoff
    assert "evening of the month-end" in reason
    assert result["reason"] != _TRUNCATED_TEXT_UNSET


def test_the_truncated_final_window_reason_for_the_2026_09_21_cutoff_offers_the_2026_08_31_month_end(
        tmp_path, late_world):
    holed, cache = late_world
    result, _, _ = _run_he(tmp_path, cache, "2026-09-21", dry_run=False, frames=holed)
    assert result["reason_code"] == "truncated_final_window"
    reason = " ".join(result["reason"].split())
    assert "the history end 2026-09-21" in reason and "for example 2026-08-31" in reason


def test_the_cli_still_writes_the_results_file_and_exits_0_for_a_truncated_final_window_under_a_history_end(
        tmp_path, cftc_cache, monkeypatch, capsys):
    code, ledger_path, research_root = _he_cli(
        tmp_path, cftc_cache, monkeypatch,
        ["--history-end", _he_iso(_HE_MID_WINDOW), "--max-history-drop", "1000000"], dry_run=False)
    assert code == 0 and not os.path.exists(ledger_path)         # exit 0 and a results file, as for every non-history-end refusal
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    output = os.path.join(research_root, "cli-he", "commodity-oi-cycle-overlay-results.json")
    assert printed["result"]["reason_code"] == "truncated_final_window" and "output" in printed
    payload = json.load(open(output, encoding="utf-8"))
    assert payload["result"]["reason_code"] == "truncated_final_window"
    assert f"the history end {_he_iso(_HE_MID_WINDOW)}" in payload["result"]["reason"]


# ---- documentation of the door checks

def test_the_driver_docstring_states_the_door_checks_and_that_other_refusals_still_write_a_results_file():
    text = " ".join(driver.__doc__.split())
    for phrase in ("malformed ``--tested-date`` or ``max_history_drop`` is refused at the same door",
                   "before anything is fetched or written",
                   "``main()`` still writes its small results JSON and exits 0"):
        assert phrase in text, phrase


def test_the_cli_help_says_which_refusals_write_nothing(capsys):
    with pytest.raises(SystemExit):
        driver.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "any refusal of this setting writes no ledger row, result file or report" in text
    assert "a malformed --tested-date is refused up front" in text
