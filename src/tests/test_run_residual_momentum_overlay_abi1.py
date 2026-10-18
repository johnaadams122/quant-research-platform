"""Tests for scripts/run_residual_momentum_overlay_abi1.py -- the bespoke ABI-1
driver for residual_momentum_ff3proxy_top50_monthly.

Every run below uses SYNTHETIC price frames injected through the driver's
``ohlc_loader`` / ``panel_loader`` hooks on the REAL exchange session calendar,
a synthetic panel of 60 names (``traded=`` override in place of the production
list) and a TEMPORARY ledger/research root under ``tmp_path``. Nothing here
reads the network, the frozen panel or data/mining/trial_ledger.jsonl. The
dry-run tests prove wiring only -- never a real cost measurement (the quotes
file is synthetic on purpose). Every file-write call site of the driver is
redirected: the OHLC cache and the JSON output go under ``research_root``, the
outcome under ``ledger_path``.
"""
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

import run_residual_momentum_overlay_abi1 as driver  # noqa: E402
from mining import cost_tax_inputs, trial_ledger  # noqa: E402
from mining.batch5 import residual_momentum_ff3proxy_top50_monthly as mod  # noqa: E402
from mining.batch5 import treasury_month_end_index_demand_overlay as cal  # noqa: E402

_ENTRY_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == mod.FAMILY_ID)


FIRST_ME = "2016-04-29"
NAMES = [f"T{i:02d}" for i in range(60)]
START, END = "2016-04-01", "2026-03-31"          # 10 years: ~1770 scored sessions

# Deliberately DIFFERENT spreads per symbol everywhere: with identical fixtures
# the suite would have zero power to catch a name-cost / SPY-cost mixup.
_QUOTES = {name: {"bid": 50.0 + i, "ask": 50.0 + i + 0.01 + 0.002 * (i % 7)}
           for i, name in enumerate(NAMES)}
_QUOTES["SPY"] = {"bid": 750.00, "ask": 750.15}


def _sessions(start=START, end=END):
    return pd.DatetimeIndex(cal.exchange_sessions(pd.Timestamp(start), pd.Timestamp(end)))


def _ohlc(close, index):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999, "Close": close}, index=index)


def _walk(index, seed, vol, drift, start):
    r = np.random.RandomState(seed).normal(drift, vol, len(index))
    r[0] = 0.0
    return start * np.cumprod(1.0 + r)


def _world(seed=5, start=START, end=END, market_end=None, n=60):
    """``end`` bounds the PANEL; ``market_end`` (default the same) the SPY and
    factor-ETF frames, which a fresh fetch would carry past a frozen panel."""
    idx = _sessions(start, end)
    midx = _sessions(start, market_end or end)
    frames = {"SPY": _ohlc(_walk(midx, seed, 0.009, 0.0004, 100.0), midx)}
    for k, sym in enumerate(mod.FACTOR_SYMBOLS):
        frames[sym] = _ohlc(_walk(midx, seed + 10 + k, 0.011, 0.0002, 90.0 + k), midx)
    irx = np.linspace(0.5, 3.0, len(midx))
    frames["^IRX"] = pd.DataFrame({"Open": irx, "High": irx, "Low": irx, "Close": irx}, index=midx)
    panel = {f"T{i:02d}": _ohlc(_walk(idx, 1000 + 7 * i + seed, 0.014, 0.0003, 40.0 + i), idx)
             for i in range(n)}
    return frames, panel


def _loader(frames):
    def load(symbol):
        frame = frames[symbol].copy()
        frame.insert(0, "Date", frame.index)
        return frame.reset_index(drop=True)
    return load


def _quotes_payload(*, synthetic: bool, drop=(), quotes=None, overrides=None) -> dict:
    """``overrides`` replaces top-level fields of the capture; a value of None removes the field."""
    quotes = quotes or _QUOTES
    payload = {
        "sampled_at_local": "2026-09-24T10:00:00",
        "market_state": "SYNTHETIC-test" if synthetic else "open",
        "source": "SYNTHETIC unit-test fixture" if synthetic else "unit test capture",
        "quotes": {s: {**q, "security_status": "Normal", "suspect": False}
                   for s, q in quotes.items() if s not in drop},
    }
    for key, value in (overrides or {}).items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def _fixture(tmp_path, *, synthetic=True, status="eligible", parameters=None, drop=(), extra_quotes=None,
             quotes_overrides=None):
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
        json.dump(_quotes_payload(synthetic=synthetic, drop=drop, quotes={**_QUOTES, **(extra_quotes or {})},
                                  overrides=quotes_overrides), fh)
    return catalog_path, quotes_path, str(tmp_path / "ledger.jsonl"), str(tmp_path / "research")


def _run(tmp_path, *, dry_run=True, synthetic=True, world=None, run_id="rmo-test", drop=(),
         traded=None, placebo_draws=300, ohlc_loader=None, sleep=lambda s: None, **fixture_kwargs):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(
        tmp_path, synthetic=synthetic, drop=drop, **fixture_kwargs)
    frames, panel = world or _world()
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id=run_id, tested_date="2026-09-24", dry_run=dry_run,
        signal_source_path="scripts/run_residual_momentum_overlay_abi1.py",
        signal_source_sha256="a" * 64, research_root=research_root,
        ohlc_loader=ohlc_loader or _loader(frames), panel_loader=_loader(panel),
        traded=NAMES if traded is None else traded, first_month_end=FIRST_ME,
        placebo_draws=placebo_draws, sleep=sleep)
    return result, ledger_path


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("base"))


# ------------------------------------------------------------- dry-run wiring

def test_dry_run_evaluates_end_to_end_without_writing_a_ledger_row(base):
    result, ledger_path = base
    assert "reason_code" not in result, result
    assert result["family_id"] == mod.FAMILY_ID and result["shape"] == "pooled_single_asset"
    assert result["verdict"] in ("cut_dsr", "survived", "cut_holdout", "cut_risk_refusal")
    assert result["n_trades"] == result["overlay_diagnostics"]["n_windows"] == 83   # Apr 2019 .. Feb 2026 decisions
    assert not os.path.exists(ledger_path)
    assert result["tax_policy"] == "realization" and result["cash_convention"] == "sweep_cash_daily"


def test_the_report_carries_the_control_the_placebo_and_the_reading_rule(base):
    result, _ = base
    control, placebo = result["zero_signal_control"], result["permutation_placebo"]
    assert control["risk_account_basis"] == "whole_account" and control["verdict"]
    assert "RESEARCH CONTROL" in control["scoring"]
    assert placebo["n_draws"] == 300 and placebo["percentile_bar"] == 0.95
    assert 0.0 <= placebo["real_percentile"] <= 1.0 and isinstance(placebo["clears_bar"], bool)
    assert "draws" not in placebo                       # the 300 raw draws stay out of the report
    assert "PRE-DECLARED READING RULE" in result["reading_rule"]
    assert "NOT distinguished from random" in result["reading_rule"]
    assert "RESEARCH CONTROL" in result["reading_rule"]


def test_the_scored_result_is_on_the_overlay_only_excess_risk_basis(base):
    result, _ = base
    risk = result["overlay_diagnostics"]["risk_layer"]
    assert risk["account_basis"] == "overlay_only_excess"
    assert risk["whole_account_reference"]["gates_verdict"] is False
    assert "failed_refusals" in risk and "absolute_cap" in risk


def test_the_account_lives_from_the_first_decision_to_the_last_complete_exit(base):
    result, _ = base
    diag = result["overlay_diagnostics"]
    assert result["account_start"] == "2019-04-30"
    assert diag["first_window_entry"] == "2019-05-01"
    assert diag["last_window_exit"] == result["account_end"] == "2026-03-31"
    assert diag["open_window_excluded"] is None


def test_costs_come_from_the_right_symbols_quotes_and_the_benchmark_from_spy(base):
    result, _ = base
    diag = result["overlay_diagnostics"]
    per_name = np.array([cost_tax_inputs.cost_bps_from_quotes(q["bid"], q["ask"], q["bid"], q["ask"])
                         for n, q in _QUOTES.items() if n != "SPY"])
    assert diag["cost_bps_round_trip"]["n"] == 60
    assert diag["cost_bps_round_trip"]["mean"] == pytest.approx(per_name.mean(), rel=1e-9)
    assert diag["cost_bps_round_trip"]["min"] == pytest.approx(per_name.min(), rel=1e-9)
    assert diag["cost_bps_round_trip"]["max"] == pytest.approx(per_name.max(), rel=1e-9)
    assert len({round(v, 6) for v in per_name}) > 3                 # distinct fixtures
    spy_spread = cost_tax_inputs.compute_spread_bps(750.00, 750.15)
    assert diag["benchmark_entry_cost_bps"] == pytest.approx(spy_spread / 2.0, rel=1e-9)


def test_the_financing_rate_is_the_bill_series_not_a_constant(tmp_path, base):
    frames, panel = _world()
    high = dict(frames)
    high["^IRX"] = frames["^IRX"] * 2.0
    result, _ = _run(tmp_path, world=(high, panel))
    base_rate = base[0]["overlay_diagnostics"]["mean_financing_rate_annual_on_overlay_sessions"]
    rate = result["overlay_diagnostics"]["mean_financing_rate_annual_on_overlay_sessions"]
    assert 1.5 < rate / base_rate < 2.5


def test_a_negative_bill_print_is_loaded_without_the_positivity_gate(tmp_path):
    frames, panel = _world()
    irx = frames["^IRX"].copy()
    irx.iloc[10:14] = -0.05
    frames = {**frames, "^IRX": irx}
    result, _ = _run(tmp_path, world=(frames, panel))
    assert "reason_code" not in result, result


def test_a_panel_name_with_a_non_positive_price_is_excluded_by_the_positivity_gate(tmp_path):
    frames, panel = _world()
    bad = panel["T03"].copy()
    bad.iloc[100:103, bad.columns.get_loc("Close")] = 0.0
    panel = {**panel, "T03": bad}
    result, _ = _run(tmp_path, world=(frames, panel))
    load_excluded = result["overlay_diagnostics"]["panel"]["load_excluded"]
    assert load_excluded["T03"]["reason"] == "panel_load_failed"
    assert result["n_panel_names"] == 59


def test_the_panel_frames_really_drive_the_curve(tmp_path, base):
    frames, panel = _world()
    swapped = dict(panel)
    for name in ("T10", "T11", "T12", "T13", "T14", "T15"):
        swapped[name] = _ohlc(_walk(_sessions(), 999, 0.02, 0.001, 50.0), _sessions())
    result, _ = _run(tmp_path, world=(frames, swapped))
    assert result["sharpe_daily_excess"] != base[0]["sharpe_daily_excess"]


def test_the_spy_frame_really_drives_the_core_and_the_benchmark(tmp_path, base):
    frames, panel = _world()
    other = dict(frames)
    other["SPY"] = _ohlc(_walk(_sessions(), 77, 0.009, 0.0004, 100.0), _sessions())
    result, _ = _run(tmp_path, world=(other, panel))
    assert result["sharpe_daily_excess"] != base[0]["sharpe_daily_excess"]


def test_the_scored_result_is_the_taxed_realization_evaluation_not_the_untaxed_one(base):
    diag = base[0]["overlay_diagnostics"]
    assert base[0]["sharpe_daily_excess"] == pytest.approx(diag["sharpe_daily_excess_taxed"])
    assert diag["sharpe_daily_excess_taxed"] != diag["sharpe_daily_excess_untaxed"]


def test_dry_run_reports_the_tax_identity_approximation_it_relies_on(base):
    diag = base[0]["overlay_diagnostics"]
    assert 0.5 < diag["tax_identity_ratio_min"] <= diag["tax_identity_ratio_max"] < 2.0
    assert diag["tax_identity_sensitivity_on_sharpe"] is not None
    assert diag["scored_excess_beta_to_spy"] is not None


def test_details_equal_independent_aggregates_of_the_trades_frame():
    frames, panel = _world()
    prepared = mod.prepare(frames["SPY"], {s: frames[s] for s in mod.FACTOR_SYMBOLS}, panel,
                           first_month_end=FIRST_ME)
    costs = {n: 2.0 + 0.1 * i for i, n in enumerate(sorted(panel))}
    curve = mod.build_curve(prepared, panel, frames["^IRX"], cost_bps_by_symbol=costs,
                            spy_entry_cost_bps=1.0)
    trades = curve["trades_df"]
    diag = driver.overlay_diagnostics(
        curve, _stub_result(curve), untaxed_result=_stub_result(curve),
        benchmark_daily=_bench(frames["SPY"]), hygiene_excluded={}, quote_excluded={},
        panel_excluded={}, n_traded=60, n_used=60, cost_bps=costs, benchmark_entry_cost_bps=1.0)
    assert diag["n_windows"] == len(trades)
    assert diag["worst_window_net_ret"] == pytest.approx(trades["net_ret"].min())
    assert diag["best_window_net_ret"] == pytest.approx(trades["net_ret"].max())
    assert diag["sum_window_financing_ret"] == pytest.approx(trades["financing_ret"].sum())
    assert diag["sum_window_cost_ret"] == pytest.approx(trades["cost_ret"].sum())
    mask = curve["overlay_mask"].to_numpy()
    assert diag["overlay_sessions"] == int(mask.sum())
    growth = np.cumprod(1.0 + curve["strategy_daily"].to_numpy())
    index = curve["strategy_daily"].index
    entry = index.get_indexer(pd.DatetimeIndex(trades["entry_date"]))
    exit_ = index.get_indexer(pd.DatetimeIndex(trades["exit_date"]))
    ratio = (growth[exit_] / growth[entry - 1]) / (1.0 + trades["net_ret"].to_numpy())
    assert diag["tax_identity_ratio_min"] == pytest.approx(ratio.min())
    assert diag["tax_identity_ratio_max"] == pytest.approx(ratio.max())


def _bench(spy):
    from mining import benchmark_relative
    return benchmark_relative.cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=1.0)


def _stub_result(curve):
    from mining import abi1_driver
    frames, _ = _world()
    return abi1_driver.evaluate_family(
        curve, benchmark_frame=frames["SPY"], benchmark_entry_cost_bps=1.0, account_type="taxable",
        tax_policy="realization")


# ------------------------------------------------------- gates and refusals

def test_real_run_refuses_a_synthetic_quotes_file(tmp_path):
    with pytest.raises(ValueError, match="SYNTHETIC"):
        _run(tmp_path, dry_run=False, synthetic=True, placebo_draws=mod.PLACEBO_DRAWS)


def test_a_missing_benchmark_quote_is_refused(tmp_path):
    result, _ = _run(tmp_path, drop=("SPY",))
    assert result["reason_code"] == "missing_data" and "SPY" in result["reason"]


def test_a_thinly_quoted_panel_is_refused_and_a_few_missing_quotes_are_excluded_and_listed(tmp_path):
    result, _ = _run(tmp_path, drop=tuple(NAMES[:20]))
    assert result["reason_code"] == "insufficient_panel_coverage"
    assert "have a passing quote" in result["reason"]       # the quote gate itself, before any history is loaded
    ok, _ = _run(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, drop=("T05", "T06"))
    assert "reason_code" not in ok
    assert ok["overlay_diagnostics"]["panel"]["n_quote_excluded"] == 2
    assert set(ok["overlay_diagnostics"]["panel"]["quote_excluded"]) == {"T05", "T06"}
    assert ok["n_panel_names"] == 58


def test_missing_catalog_entry_and_ineligible_status_skip_cleanly(tmp_path):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump([], fh)
    common = dict(quotes_path=quotes_path, ledger_path=ledger_path, run_id="x", tested_date="2026-09-24",
                  dry_run=True, signal_source_path="s", signal_source_sha256="0", research_root=research_root)
    assert driver.compute_and_record(catalog_path=catalog_path, **common)["reason_code"] == "not_in_catalog"
    result, _ = _run(tmp_path / "s" if (tmp_path / "s").mkdir() is None else tmp_path, status="ruled_out")
    assert result["reason_code"] == "catalog_status"


def test_parameter_drift_blocks_the_run(tmp_path):
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        params = dict(_catalog_entry_from(fh)["parameters"])
    params["select_top_n"] = 51
    result, _ = _run(tmp_path, parameters=params)
    assert result["reason_code"] == "parameter_drift"


def test_an_existing_abi1_outcome_blocks_a_second_trial(tmp_path):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    trial_ledger.record_outcome(ledger_path, mod.FAMILY_ID, "earlier", "2026-09-01", stage="cut_dsr",
                                dsr_value=0.5, sharpe_in_sample=0.01, ticker_label=None,
                                construction="ABI-1", note="earlier")
    frames, panel = _world()
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="x",
        tested_date="2026-09-24", dry_run=False, signal_source_path="s", signal_source_sha256="0",
        research_root=research_root, ohlc_loader=_loader(frames), panel_loader=_loader(panel),
        traded=NAMES, first_month_end=FIRST_ME)
    assert result["reason_code"] == "already_scored_abi1"


def test_a_real_run_must_use_the_pre_registered_placebo_draw_count(tmp_path):
    with pytest.raises(ValueError, match="500 placebo draws"):
        _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=300)


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_spy_history_missing_an_exchange_session_is_refused_and_named(tmp_path, dry_run):
    frames, panel = _world()
    hole = frames["SPY"].index[400]
    cut = {s: (f.drop(hole) if s != "^IRX" else f) for s, f in frames.items()}
    result, ledger_path = _run(tmp_path, dry_run=dry_run, synthetic=dry_run, world=(cut, panel),
                               placebo_draws=300 if dry_run else mod.PLACEBO_DRAWS)
    assert result["reason_code"] == "missing_exchange_session"
    assert pd.Timestamp(hole).strftime("%Y-%m-%d") in result["missing_dates"]
    assert not os.path.exists(ledger_path)


def test_factor_etfs_that_end_before_spy_are_refused_with_unequal_trailing_coverage(tmp_path):
    frames, panel = _world()
    frames = dict(frames)
    frames["IVE"] = frames["IVE"].iloc[:-1]
    result, ledger_path = _run(tmp_path, world=(frames, panel))
    assert result["reason_code"] == "unequal_trailing_coverage"
    assert not os.path.exists(ledger_path)


def test_a_factor_etf_that_cannot_be_loaded_after_retries_refuses_the_run(tmp_path):
    frames, panel = _world()
    good = _loader(frames)
    calls = []

    def flaky(symbol):
        calls.append(symbol)
        if symbol == "IVE":
            raise RuntimeError("no price data")
        return good(symbol)

    sleeps = []
    result, ledger_path = _run(tmp_path, ohlc_loader=flaky, sleep=sleeps.append)
    assert result["reason_code"] == "ohlc_load_failed" and "IVE" in result["reason"]
    assert calls.count("IVE") == driver.FACTOR_LOAD_ATTEMPTS
    assert len(sleeps) == driver.FACTOR_LOAD_ATTEMPTS - 1
    assert not os.path.exists(ledger_path)


def test_a_transient_factor_etf_failure_is_retried_and_the_run_proceeds(tmp_path):
    frames, panel = _world()
    good = _loader(frames)
    state = {"n": 0}

    def flaky(symbol):
        if symbol == "IVE" and state["n"] < 2:
            state["n"] += 1
            raise RuntimeError("no price data")
        return good(symbol)

    result, _ = _run(tmp_path, ohlc_loader=flaky)
    assert "reason_code" not in result and state["n"] == 2


def test_retrying_loader_retries_empty_frames_then_fails_closed_without_returning_partial_data():
    sleeps = []
    frame = pd.DataFrame({"Date": [pd.Timestamp("2020-01-02")], "Open": [1.0]})
    seq = [pd.DataFrame(), None, frame]
    load = driver.retrying_loader(lambda s: seq.pop(0), attempts=3, sleep=sleeps.append)
    assert load("X") is frame and sleeps == [1.0, 2.0]
    dead = driver.retrying_loader(lambda s: pd.DataFrame(), attempts=2, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="X: OHLC load failed after 2 attempt"):
        dead("X")


def test_the_market_frames_are_cut_to_the_panels_last_session_and_the_run_proceeds(tmp_path):
    frames, panel = _world(end="2026-03-31", market_end="2026-06-30")
    result, _ = _run(tmp_path, world=(frames, panel))
    assert "reason_code" not in result, result
    assert result["as_of"] == "2026-03-31" and result["account_end"] == "2026-03-31"
    assert result["last_bar_dates"]["SPY"] == "2026-06-30"        # the fetched history is longer


def test_a_still_open_final_window_is_excluded_reported_and_never_closed_early(tmp_path):
    frames, panel = _world(end="2026-03-18")
    result, _ = _run(tmp_path, world=(frames, panel))
    diag = result["overlay_diagnostics"]
    assert result["as_of"] == "2026-03-18" and result["account_end"] == "2026-02-27"
    assert diag["open_window_excluded"]["entry"] == "2026-03-02"
    assert diag["open_window_excluded"]["scheduled_exit"] == "2026-03-31"
    assert "open window excluded, never closed early" in result["cost_model_note"]


def test_hygiene_exclusions_are_reported_by_name_and_reason(tmp_path):
    frames, panel = _world()
    gap = panel["T08"].drop(panel["T08"].index[300])
    panel = {**panel, "T08": gap, "HUBB": panel["T01"].copy()}
    result, _ = _run(tmp_path, world=(frames, panel), traded=NAMES + ["HUBB"],
                     extra_quotes={"HUBB": {"bid": 300.0, "ask": 300.05}})
    hyg = result["overlay_diagnostics"]["panel"]
    assert hyg["hygiene_excluded"]["T08"]["reason"] == "interior_gap"
    assert hyg["hygiene_excluded"]["HUBB"]["reason"] == "documented_defect"
    assert hyg["hygiene_excluded_by_reason"] == {"interior_gap": 1, "documented_defect": 1}
    assert result["n_panel_names"] == 59 + 0 and hyg["used"] == 59


def test_a_curve_that_lost_its_account_kind_is_refused_before_anything_is_recorded(tmp_path, monkeypatch):
    """Without the declaration ABI-1 would score the whole SPY-core account
    while the ledger note claims the overlay-only basis."""
    real = driver.mod.build_curve

    def stripped(*args, **kwargs):
        curve = real(*args, **kwargs)
        curve.pop("account_kind")
        return curve

    monkeypatch.setattr(driver.mod, "build_curve", stripped)
    with pytest.raises(RuntimeError, match="risk-basis rule"):
        _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_a_core_that_differs_from_the_benchmark_outside_the_windows_is_refused(tmp_path, monkeypatch):
    """The scored excess is only the overlay's own if the account IS the benchmark
    on every session the overlay is off."""
    real = driver.mod.build_curve

    def drifted(*args, **kwargs):
        curve = real(*args, **kwargs)
        curve["strategy_daily"] = curve["strategy_daily"].copy()
        off = np.flatnonzero(~curve["overlay_mask"].to_numpy())
        curve["strategy_daily"].iloc[off[0]] += 0.001
        return curve

    monkeypatch.setattr(driver.mod, "build_curve", drifted)
    with pytest.raises(RuntimeError, match="not overlay-only"):
        _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))




@pytest.mark.parametrize("dry_run", [True, False])
def test_an_account_that_is_not_the_benchmark_plus_the_overlay_inside_a_window_is_refused_before_anything_is_recorded(
        tmp_path, monkeypatch, dry_run):
    """The shared off-window guard reads ONE session on the real monthly schedule (the windows
    are contiguous), so a fault inside a window passed it. The per-session identity check refuses it; the fault
    below changes the account on one in-window session and leaves the reported overlay series alone."""
    real = driver.mod.build_curve

    def faulty(*args, **kwargs):
        curve = real(*args, **kwargs)
        curve["strategy_daily"] = curve["strategy_daily"].copy()
        on = np.flatnonzero(curve["overlay_mask"].to_numpy())
        curve["strategy_daily"].iloc[on[40]] += 0.0005
        return curve

    monkeypatch.setattr(driver.mod, "build_curve", faulty)
    with pytest.raises(RuntimeError, match="not the benchmark plus the overlay"):
        _run(tmp_path, dry_run=dry_run, synthetic=dry_run, placebo_draws=300 if dry_run else mod.PLACEBO_DRAWS)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_the_shared_off_window_guard_alone_would_have_passed_that_in_window_fault(tmp_path, monkeypatch):
    """The premise of the test above, proven on the driver's own path: with the per-session check switched off the run
    proceeds to a scored result, so only the per-session identity check stands between that fault and the ledger."""
    real = driver.mod.build_curve

    def faulty(*args, **kwargs):
        curve = real(*args, **kwargs)
        curve["strategy_daily"] = curve["strategy_daily"].copy()
        on = np.flatnonzero(curve["overlay_mask"].to_numpy())
        curve["strategy_daily"].iloc[on[40]] += 0.0005
        return curve

    monkeypatch.setattr(driver.mod, "build_curve", faulty)
    monkeypatch.setattr(driver.cohort_overlay, "require_account_is_benchmark_plus_overlay",
                        lambda *args, **kwargs: None)
    result, _ = _run(tmp_path)
    assert "reason_code" not in result and result["verdict"]


def test_the_driver_calls_the_identity_check_with_the_scored_benchmark_and_the_family_id(tmp_path, monkeypatch):
    seen = {}
    real = driver.cohort_overlay.require_account_is_benchmark_plus_overlay

    def spy(curve, benchmark_daily, *, family_id, **kwargs):
        seen.update(curve=curve, benchmark=benchmark_daily, family_id=family_id)
        return real(curve, benchmark_daily, family_id=family_id, **kwargs)

    monkeypatch.setattr(driver.cohort_overlay, "require_account_is_benchmark_plus_overlay", spy)
    result, _ = _run(tmp_path)
    assert "reason_code" not in result, result
    assert seen["family_id"] == mod.FAMILY_ID
    assert seen["curve"]["account_kind"] == "benchmark_plus_overlay"
    frames, _ = _world()
    from mining import benchmark_relative
    expected = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        frames["SPY"], entry_cost_bps=cost_tax_inputs.compute_spread_bps(750.00, 750.15) / 2.0)
    pd.testing.assert_series_equal(seen["benchmark"], expected, check_names=False, check_freq=False)

# ------------------------------------------------------------- real recording

def test_a_non_dry_run_against_a_temp_ledger_writes_exactly_one_abi1_row_with_disclosures(tmp_path):
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert "reason_code" not in result, result
    rows = trial_ledger.read_all(ledger_path)
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and outcomes[0]["family_id"] == mod.FAMILY_ID     # verbatim id
    assert outcomes[0]["construction"] == "ABI-1"
    assert {r["family_id"] for r in rows} == {mod.FAMILY_ID}                    # no control / placebo rows
    note = outcomes[0]["note"]
    for needle in ("RESEARCH CONTROL", "PRE-DECLARED READING RULE", "overlay_only_excess",
                   "not an investable account", "survivorship", "no short borrow fee",
                   "placebo percentile", "Zero-signal equal-weight control",
                   "cash_convention=sweep_cash_daily", "tax_policy=realization",
                   "long-short decile result", "never closed early", "factor-ETF freeze sha256"):
        assert needle in note, needle
    assert result["stage"] == outcomes[0]["stage"]


def test_an_unscoreable_real_run_records_the_instrument_set_benchmark_and_cash_convention(tmp_path):
    """The two UNSCORED verdicts carry what was scored in the ledger row's evidence
    rather than in its note (a short history here is not scoreable)."""
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS,
                               world=_world(end="2019-12-31"))
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    observed = outcomes[0]["evidence"]["observed"]
    assert observed["instrument_set"] == sorted(NAMES) + ["SPY", "IWM", "IVE", "IVW"]   # panel AND market frames
    assert observed["benchmark"] == "SPY" and observed["cash_convention"] == "sweep_cash_daily"
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        assert observed["parameters"] == _catalog_entry_from(fh)["parameters"]
    assert outcomes[0]["construction"] == "ABI-1" and outcomes[0]["family_id"] == mod.FAMILY_ID


def test_the_ledger_note_states_the_ordering_of_the_risk_basis_accurately(tmp_path):
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    note = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"][0]["note"]
    assert "before this family was built and before any of its ledger rows" in note
    assert "risk-basis rule" in note


def test_a_real_run_refuses_ohlc_cached_by_an_earlier_dry_run_under_the_same_run_id(tmp_path):
    frames, panel = _world()
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=True)
    kwargs = dict(catalog_path=catalog_path, ledger_path=ledger_path, run_id="same", tested_date="2026-09-24",
                  signal_source_path="s", signal_source_sha256="0", research_root=research_root,
                  ohlc_loader=_loader(frames), panel_loader=_loader(panel), traded=NAMES,
                  first_month_end=FIRST_ME, placebo_draws=300)
    driver.compute_and_record(quotes_path=quotes_path, dry_run=True, **kwargs)
    with open(quotes_path, "w", encoding="utf-8") as fh:
        json.dump(_quotes_payload(synthetic=False), fh)
    kwargs["placebo_draws"] = mod.PLACEBO_DRAWS
    stale = driver.compute_and_record(quotes_path=quotes_path, dry_run=False, **kwargs)
    assert stale["reason_code"] == "stale_ohlc_cache"
    assert set(stale["cached_symbols"]) == {"SPY", "IWM", "IVE", "IVW", "^IRX"}
    assert not os.path.exists(ledger_path)
    kwargs["run_id"] = "fresh"
    fresh = driver.compute_and_record(quotes_path=quotes_path, dry_run=False, **kwargs)
    assert "reason_code" not in fresh


def test_the_full_result_serializes_to_strict_json_as_main_writes_it(base):
    """`main` writes the result with allow_nan=False AFTER the ledger row is
    recorded: a value that cannot serialize would crash the script after the
    permanent write. Every field of a real result must survive."""
    from run_batch5_mining import _json_safe
    text = json.dumps({"result": _json_safe(base[0])}, indent=2, allow_nan=False)
    back = json.loads(text)["result"]
    assert back["family_id"] == mod.FAMILY_ID
    assert back["overlay_diagnostics"]["first_window_entry"] == "2019-05-01"
    assert isinstance(back["permutation_placebo"]["real_percentile"], float)


def test_the_market_history_is_frozen_with_a_manifest_of_hashes(base):
    manifest = base[0]["factor_freeze"]
    assert set(manifest) == {"SPY", "IWM", "IVE", "IVW", "^IRX"}
    for entry in manifest.values():
        assert len(entry["sha256"]) == 64 and entry["rows"] > 1000 and entry["source"] in ("fetched", "cache")


# ------------------------------------------------------------------- the CLI

def test_cli_dry_run_writes_a_json_result_under_the_research_root_and_exits_zero(tmp_path, capsys,
                                                                                   monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"family_id": mod.FAMILY_ID, "verdict": "cut_dsr", "when": pd.Timestamp("2026-01-02")}

    monkeypatch.setattr(driver, "compute_and_record", fake)
    quotes = tmp_path / "q.json"
    quotes.write_text("{}")
    rc = driver.main(["--dry-run", "--quotes", str(quotes), "--research-root", str(tmp_path / "rr"),
                      "--catalog", str(tmp_path / "c.json"), "--ledger", str(tmp_path / "l.jsonl")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["result"]["when"] == "2026-01-02T00:00:00"
    assert os.path.exists(str(tmp_path / "rr" / driver.DEFAULT_RUN_ID / "residual-momentum-overlay-dry-run.json"))
    assert seen["dry_run"] is True and seen["run_id"] == driver.DEFAULT_RUN_ID


def test_a_real_run_requires_an_explicit_tested_date_and_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "compute_and_record", lambda **k: {})
    quotes = tmp_path / "q.json"
    quotes.write_text("{}")
    with pytest.raises(SystemExit):
        driver.main(["--quotes", str(quotes), "--run-id", "r"])
    with pytest.raises(SystemExit):
        driver.main(["--quotes", str(quotes), "--tested-date", "2026-09-24"])


def test_the_driver_defaults_never_point_a_test_at_the_real_ledger_or_research_root():
    assert driver.LEDGER_PATH.replace("\\", "/").endswith("data/mining/trial_ledger.jsonl")
    assert driver.DEFAULT_RUN_ID == "residual-momentum-overlay-abi1-v1"
    assert not os.path.exists(os.path.join(driver.RESEARCH_ROOT, driver.DEFAULT_RUN_ID))
    assert driver.PANEL_CACHE_DIR.replace("\\", "/").endswith("data/mining/universe_cache/ohlc_2026-08-29")


def test_the_daily_rebalanced_control_is_reported_and_matches_an_independent_evaluation(base):
    from mining import abi1_driver
    result, _ = base
    daily = result["zero_signal_daily_rebalanced_control"]
    assert "REBALANCED EVERY SESSION" in daily["scoring"] and "gross of costs" in daily["scoring"]
    assert daily["risk_account_basis"] == "whole_account"
    frames, panel = _world()
    prepared = mod.prepare(frames["SPY"], {s: frames[s] for s in mod.FACTOR_SYMBOLS}, panel,
                           first_month_end=FIRST_ME)
    curve = mod.build_daily_control_curve(prepared, panel)
    scored = abi1_driver.evaluate_family(
        curve, benchmark_frame=frames["SPY"],
        benchmark_entry_cost_bps=cost_tax_inputs.compute_spread_bps(750.00, 750.15) / 2.0,
        account_type="taxable", tax_policy="none")
    assert daily["dsr_value"] == pytest.approx(scored["layer1"]["dsr_value"], rel=1e-9)
    excess = scored["log_excess"].to_numpy()
    n_hold = scored["window"]["holdout_sessions"]
    ins, hol = excess[:-n_hold], excess[-n_hold:]
    assert daily["annualized"]["in_sample"]["annualized_mean_log_excess"] == pytest.approx(ins.mean() * 252.0)
    assert daily["annualized"]["in_sample"]["annualized_excess_sharpe"] == pytest.approx(
        ins.mean() / ins.std(ddof=1) * np.sqrt(252.0))
    assert daily["annualized"]["holdout"]["annualized_mean_log_excess"] == pytest.approx(hol.mean() * 252.0)
    assert daily["annualized"]["holdout"]["n_sessions"] == n_hold
    assert "DAILY-REBALANCED zero-signal control" in result["reading_rule"]


def test_the_two_controls_are_different_constructions_scored_on_their_own(base):
    result, _ = base
    monthly, daily = result["zero_signal_control"], result["zero_signal_daily_rebalanced_control"]
    assert "SAME monthly windows" in monthly["scoring"]
    assert monthly["sharpe_daily_excess"] != daily["sharpe_daily_excess"]
    assert monthly["annualized"]["in_sample"]["n_sessions"] == daily["annualized"]["in_sample"]["n_sessions"]


# ================================================================ look-ahead and disclosure checks
# Point-in-time (look-ahead) and disclosure checks.

@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """One real (non-dry) run against a temp ledger; shared by the note-wording tests."""
    tmp = tmp_path_factory.mktemp("real")
    result, ledger_path = _run(tmp, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert "reason_code" not in result, result
    note = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"][0]["note"]
    return result, ledger_path, note, str(tmp / "research")


def _flat(text):
    return " ".join(text.split())


def test_the_ledger_note_states_what_the_overlay_does_and_does_not_cancel(real_run):
    """The short leg cancels the LEVEL of the survivorship baseline only; rank-
    conditional survivorship is not cancelled and the placebo cannot test it, so a
    pass is inconclusive and a cut is the informative outcome."""
    note = _flat(real_run[2])
    for needle in ("cancels the LEVEL of the survivorship baseline only", "rank-conditional survivorship",
                   "is NOT cancelled", "cannot test it", "INCONCLUSIVE, not confirmatory",
                   "a cut is the informative outcome", "draws its random cohorts from the same eligible set"):
        assert needle in note, needle


def test_the_ledger_note_pins_every_required_disclosure(real_run):
    note = _flat(real_run[2])
    for needle in ("2026 quoted spreads are applied to every year",
                   "direction unmeasured",
                   "(the overnight gap into it is not earned)",
                   "GOOGL, FOX, NWSA dropped as duplicate share classes, HUBB and MNST as documented vendor defects",
                   "HML-proxy fidelity", "imperfect proxy", "only partly removed",
                   "decided on full-sample or 2026 data and applied to all history",
                   "fixed ex-ante and none was tuned", "design choices",
                   "the regression, formation and skip lengths and the score are from the literature"):
        assert needle in note, needle


def test_the_ledger_note_reports_the_bill_print_age_and_labels_each_unmodelled_item_correctly(real_run):
    result, _, note, _ = real_run
    note = _flat(note)
    age = result["overlay_diagnostics"]["bill_print_max_age_days"]
    assert isinstance(age, int) and 3 <= age <= mod.MAX_BILL_PRINT_AGE_DAYS
    assert f"maximum bill-print age on any financed session {age} calendar days (refused above " in note
    # not crediting a short rebate is conservative (cohort_overlay), so it must not sit under "each flatters"
    assert "no short borrow fee" in note and "flatters the strategy" in note
    assert "no short rebate (conservative)" in note
    assert note.count("no short rebate") == 1 and "fee and no short rebate" not in note   # never under "flatters"


def test_the_freeze_manifest_hash_is_the_sha256_of_the_cached_file(real_run):
    """A manifest that only looks like a hash proves nothing; recompute it."""
    import hashlib
    result, _, _, research_root = real_run
    cache_dir = os.path.join(research_root, "rmo-test", "artifacts", "ohlc")
    manifest = result["factor_freeze"]
    assert set(manifest) == {"SPY", "IWM", "IVE", "IVW", "^IRX"}
    seen = set()
    for symbol, entry in manifest.items():
        with open(os.path.join(cache_dir, driver.batch5._cache_name(symbol)), "rb") as fh:
            expected = hashlib.sha256(fh.read()).hexdigest()
        assert entry["sha256"] == expected, symbol
        assert expected != hashlib.sha256(b"").hexdigest()
        seen.add(expected)
    assert len(seen) == 5                                         # five different histories, five hashes


def test_sha256_of_is_the_sha256_of_the_file_bytes_across_read_blocks(tmp_path):
    import hashlib
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 9000                            # > 2 MB: more than one read block
    path.write_bytes(payload)
    assert driver._sha256_of(str(path)) == hashlib.sha256(payload).hexdigest()
    (tmp_path / "empty.bin").write_bytes(b"")
    assert driver._sha256_of(str(tmp_path / "empty.bin")) == hashlib.sha256(b"").hexdigest()


def test_the_driver_defaults_carry_the_pinned_sample_start():
    import inspect
    assert inspect.signature(driver.compute_and_record).parameters["first_month_end"].default == "2000-05-31"


def test_the_driver_docstring_states_the_survivorship_limit_and_the_unmeasured_direction():
    doc = _flat(driver.__doc__)
    lowered = doc.lower()
    assert "rank-conditional survivorship" in lowered and "is NOT cancelled" in doc
    assert "INCONCLUSIVE, not confirmatory" in doc and "cancels the LEVEL of the survivorship baseline only" in doc
    assert "direction unmeasured" in lowered and "full-sample or 2026 data" in lowered


# ---- a stale bill print refuses the run (nothing written)

@pytest.mark.parametrize("dry_run", [True, False])
def test_a_hole_in_the_bill_series_older_than_ten_days_on_a_financed_session_is_refused(tmp_path, dry_run):
    frames, panel = _world()
    irx = frames["^IRX"]
    start = irx.index.get_loc(pd.Timestamp("2021-03-01"))
    holed = irx.drop(irx.index[start:start + 12])                # ~16 calendar days without a print
    result, ledger_path = _run(tmp_path, dry_run=dry_run, synthetic=dry_run, world=({**frames, "^IRX": holed}, panel),
                               placebo_draws=300 if dry_run else mod.PLACEBO_DRAWS)
    assert result["reason_code"] == "stale_bill_print"
    assert result["max_age_days"] > mod.MAX_BILL_PRINT_AGE_DAYS and result["session"].startswith("2021-03")
    assert not os.path.exists(ledger_path)


# ---- the remaining refusal paths reach the driver's return

def test_no_complete_window_is_refused_with_no_windows_and_nothing_is_written(tmp_path):
    frames, panel = _world(n=49)                                  # 49 names can never fill a 50-name cohort
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, world=(frames, panel),
                               traded=NAMES[:49], placebo_draws=mod.PLACEBO_DRAWS)
    assert result["reason_code"] == "no_windows"
    assert not os.path.exists(ledger_path)


def test_a_cohort_data_error_from_the_builder_is_returned_as_a_reason_coded_refusal(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise driver.cohort_overlay.CohortDataError(
            "missing_member_price", "T05 has no Open on the entry session", tickers=["T05", "T04"],
            dates=["2020-09-01"])

    monkeypatch.setattr(driver.mod, "build_curve", broken)
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert result["reason_code"] == "missing_member_price"
    assert result["tickers"] == ["T04", "T05"] and result["n_tickers"] == 2 and result["dates"] == ["2020-09-01"]
    assert not os.path.exists(ledger_path)


def test_an_eligible_name_missing_a_window_session_that_hygiene_cannot_see_reaches_the_refusal(tmp_path, monkeypatch):
    """End to end through the real builder: hygiene is bypassed (a hole it would catch
    is the normal path), so the missing entry Open of an ELIGIBLE name must be refused
    by the cohort builder itself and reported by the driver, never patched."""
    entry = mod.month_schedule("2020-08-31", "2020-08-31")[0][1]
    frames, panel = _world()
    panel = {**panel, "T05": panel["T05"].drop(entry)}
    monkeypatch.setattr(driver.mod, "panel_hygiene",
                        lambda frames_, **kw: {"kept": dict(frames_), "excluded": {}})
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, world=(frames, panel),
                               placebo_draws=mod.PLACEBO_DRAWS)
    assert result["reason_code"] in ("missing_member_price", "missing_member_frame")
    assert "T05" in result["tickers"] and entry.strftime("%Y-%m-%d") in result["dates"]
    assert not os.path.exists(ledger_path)


# ================================================================ market-hours quote capture
# A REAL run prices its one-shot verdict off a quote capture. The shared loader only refuses the word
# SYNTHETIC, so on its own it would accept a closed-market or wrong-date capture and record it permanently.
# The driver therefore refuses, for REAL runs
# only, with reason code quotes_not_market_hours and nothing fetched or written, unless the capture's
# market_state starts with open (any case, surrounding whitespace ignored), sampled_at_local parses (naive is
# US Eastern, tz-aware is converted), its Eastern date is the run's tested date and its Eastern time of day is
# from 09:45:00 to 15:45:00 inclusive.

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
    ("2026-09-24T09:44:59", False), ("2026-09-24T15:45:01", False), ("2026-09-24T15:45:00.5", False),
    ("2026-09-24T09:30:00", False), ("2026-09-24T16:00:00", False), ("2026-09-24T08:00:00", False),
    ("2026-09-24T17:30:00", False), ("2026-09-24T00:00:00", False)])
def test_the_capture_window_is_0945_to_1545_eastern_inclusive_at_both_ends(sampled, accepted):
    assert (_hours_refusal({"sampled_at_local": sampled}) is None) is accepted


@pytest.mark.parametrize("state,accepted", [
    ("open", True), ("OPEN", True), ("Open", True), ("open (regular session)", True), (" open", True),
    ("\topen\n", True), ("closed", False), ("Closed", False), ("pre-market", False), ("after-hours", False),
    ("market closed", False), ("", False), ("   ", False), ("unknown", False), (None, False), (7, False)],
    ids=["open", "upper", "title", "open_regular_session", "leading_space", "tab_and_newline", "closed",
         "closed_title", "pre_market", "after_hours", "market_closed", "empty", "blanks", "unknown", "none",
         "integer"])
def test_the_market_state_must_start_with_open_in_any_case(state, accepted):
    # the time and date are valid here, so ONLY the state can refuse (None removes the field)
    assert (_hours_refusal({"market_state": state}) is None) is accepted


@pytest.mark.parametrize("sampled", ["2026-09-23T10:00:00", "2026-09-25T10:00:00", "2025-09-24T10:00:00",
                                     "2026-10-24T10:00:00", "2026-08-24T10:00:00"])
def test_the_capture_must_be_dated_the_run_date(sampled):
    assert _hours_refusal({"sampled_at_local": sampled}) is not None      # time and state are valid


def test_the_run_date_is_the_reference_not_a_constant():
    assert _hours_refusal({"sampled_at_local": "2026-09-23T10:00:00"}, tested="2026-09-23") is None
    assert _hours_refusal({"sampled_at_local": "2026-09-23T10:00:00"}, tested="2026-09-24") is not None


@pytest.mark.parametrize("sampled", ["not-a-time", "", "NaT", " ", "2026-13-45T10:00:00", 12345, 12.5, True,
                                     ["2026-09-24T10:00:00"], {"t": "2026-09-24T10:00:00"},
                                     "10:00", "10:00:00", "Sep 24 10:00"])
def test_an_unparseable_or_date_less_sample_time_is_a_refusal_not_an_exception(sampled):
    assert _hours_refusal({"sampled_at_local": sampled}) is not None


@pytest.mark.parametrize("sampled", ["10:00", "10:00:00", "10:00 AM"])
def test_a_time_only_sample_is_refused_even_on_the_day_pandas_fills_in(sampled):
    """pandas reads a time-only string as the CURRENT date, so on the run date itself it would pass the date
    check and the window check: it must be refused for carrying no date. The run date is derived from the same
    parse, so this holds on any day the suite runs."""
    filled_date = pd.Timestamp("10:00").strftime("%Y-%m-%d")
    assert _hours_refusal({"sampled_at_local": sampled}, tested=filled_date) is not None


@pytest.mark.parametrize("sampled,tested", [
    ("2026-03-08T02:30:00", "2026-03-08"),            # a local time that does not exist (spring forward)
    ("9999-12-31T23:59:59Z", "9999-12-31"),           # far past the representable range once converted
    ("2026-09-24T10:00:00+25:00", "2026-09-24")])     # an impossible offset
def test_a_time_that_cannot_be_localized_is_a_refusal_not_an_exception(sampled, tested):
    assert _hours_refusal({"sampled_at_local": sampled}, tested=tested) is not None


@pytest.mark.parametrize("missing", ["sampled_at_local", "market_state"])
def test_a_payload_missing_either_field_is_a_refusal(missing):
    assert _hours_refusal({missing: None}) is not None


@pytest.mark.parametrize("payload", [None, [], "open", 5, {}])
def test_a_non_mapping_or_empty_payload_is_a_refusal_never_an_exception(payload):
    assert driver.quotes_market_hours_refusal(payload, "2026-09-24") is not None


@pytest.mark.parametrize("tested", ["not-a-date", "", None, 20260, "2026-13-45"])
def test_an_unparseable_run_date_is_a_refusal_never_an_exception(tested):
    assert driver.quotes_market_hours_refusal(
        {"sampled_at_local": "2026-09-24T10:00:00", "market_state": "open"}, tested) is not None


@pytest.mark.parametrize("sampled,accepted", [
    ("2026-09-24T09:50:00Z", False),         # 05:50 Eastern: a naive read of the digits would wrongly accept it
    ("2026-09-24T13:50:00Z", True),          # 09:50 Eastern
    ("2026-09-24T13:44:59Z", False),         # 09:44:59 Eastern
    ("2026-09-24T13:45:00Z", True),          # 09:45:00 Eastern, the inclusive open edge
    ("2026-09-24T19:45:00+00:00", True),     # 15:45:00 Eastern, the inclusive close edge
    ("2026-09-24T19:45:01Z", False),         # 15:45:01 Eastern
    ("2026-09-24T10:00:00-04:00", True),     # already Eastern daylight time
    ("2026-09-24T15:00:00Z", True),          # 11:00 Eastern
    ("2026-09-25T01:00:00Z", False)])        # the evening of the 24th Eastern
def test_a_timezone_aware_sample_is_converted_to_eastern_and_a_naive_one_is_read_as_eastern(sampled, accepted):
    assert (_hours_refusal({"sampled_at_local": sampled}) is None) is accepted


def test_the_refusal_names_what_was_wrong():
    assert "closed" in _hours_refusal({"market_state": "closed"})
    assert "2026-09-23" in _hours_refusal({"sampled_at_local": "2026-09-23T10:00:00"})
    assert "09:45" in _hours_refusal({"sampled_at_local": "2026-09-24T09:00:00"})


@pytest.mark.parametrize("overrides", [
    {"market_state": "closed"},                                       # closed, time and date valid
    {"market_state": "pre-market", "sampled_at_local": "2026-09-24T09:00:00"},
    {"market_state": "after-hours", "sampled_at_local": "2026-09-24T17:00:00"},
    {"sampled_at_local": "2026-09-23T10:00:00"},                      # wrong date
    {"sampled_at_local": "yesterday morning"},                        # unparseable
    {"sampled_at_local": "2026-09-24T15:46:00"},                      # a minute after the window
    {"sampled_at_local": "2026-09-24T09:44:00"}])                     # a minute before it
def test_a_real_run_off_hours_is_refused_with_a_reason_code_fetches_nothing_and_writes_nothing(tmp_path, overrides):
    fetched = []

    def spy(symbol):
        fetched.append(symbol)
        raise AssertionError(f"a refused run must not fetch anything ({symbol})")

    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS,
                               quotes_overrides=overrides, ohlc_loader=spy)
    assert result["reason_code"] == "quotes_not_market_hours" and result["family_id"] == mod.FAMILY_ID
    assert result["reason"] and "verdict" not in result and "stage" not in result
    assert fetched == []
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(str(tmp_path / "research"))           # no OHLC cache either: nothing was fetched


@pytest.mark.parametrize("overrides,field", [({"sampled_at_local": None}, "sampled_at_local"),
                                             ({"market_state": None}, "market_state"),
                                             ({"sampled_at_local": 12345}, "sampled_at_local")])
def test_a_real_run_whose_capture_lacks_a_metadata_field_is_reason_coded_like_an_off_hours_one(
        tmp_path, overrides, field):
    fetched = []

    def spy(symbol):
        fetched.append(symbol)
        raise AssertionError(f"a refused run must not fetch anything ({symbol})")

    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS,
                               quotes_overrides=overrides, ohlc_loader=spy)
    assert result["reason_code"] == "quotes_not_market_hours" and result["family_id"] == mod.FAMILY_ID
    assert field in result["reason"] and str(tmp_path) not in result["reason"]      # names the field, never a path
    assert "verdict" not in result and "stage" not in result
    assert fetched == []
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(str(tmp_path / "research"))


def test_a_dry_run_whose_capture_lacks_a_metadata_field_still_raises_from_the_shared_loader(tmp_path):
    with pytest.raises(ValueError, match="sampled_at_local"):
        _run(tmp_path, dry_run=True, synthetic=True, placebo_draws=3, quotes_overrides={"sampled_at_local": None})


def test_a_dry_run_is_exempt_from_the_market_hours_guard(tmp_path):
    result, ledger_path = _run(tmp_path, dry_run=True, synthetic=True,
                               quotes_overrides={"market_state": "SYNTHETIC-closed",
                                                 "sampled_at_local": "2026-01-02T03:00:00"})
    assert "reason_code" not in result and result["verdict"]
    assert not os.path.exists(ledger_path)


def test_a_real_run_at_the_inclusive_close_edge_given_in_utc_is_scored(tmp_path):
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS,
                               quotes_overrides={"sampled_at_local": "2026-09-24T19:45:00+00:00"})
    assert "reason_code" not in result, result
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


def test_the_driver_docstring_states_the_market_hours_guard():
    doc = _flat(driver.__doc__)
    assert "quotes_not_market_hours" in doc and "09:45 to 15:45 Eastern" in doc
    assert "dry runs exempt" in doc


# ================================================================ control condition gates the pre-registered pass
# This family's control condition is the permutation placebo: the real rule's overlay Sharpe percentile among
# the 500 seeded random same-size cohorts must be at least 0.95. The recorded stage still comes solely from the
# ABI-1 verdict (the stage vocabulary is closed in shared code and every tested family is recorded), so a main
# passing stage that failed the placebo is recorded as passing but is NOT a survivor under the family's own
# pre-registered rule. What the driver adds: (a) a top-level pre_registered_pass = (verdict is the passing stage) AND the
# control condition is exactly True; (b) when the stage passed but the condition is False or missing, the FIRST sentence
# of the driver-supplied ledger note (and of the printed summary) is the exact flag below; (c) when the stage passed and
# the condition holds, the note says the pre-registered rule was met. The synthetic world does not reach
# the passing stage on its own, so the real-path tests FORCE the verdict of the main scored evaluation only (the first
# evaluate_family call; every control keeps its real verdict) and force the placebo's clears_bar; the pure
# decision functions are also tested directly.

_FAILED_FLAG = ("CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under this family's own "
                "pre-registered rule (the real rule's overlay Sharpe percentile among the seeded random "
                "same-size cohort placebo draws must be at least 0.95). This is not a pass; do not promote it without review.")
_MET_CLAUSE = ("PRE-REGISTERED RULE MET: the ABI-1 stage survived AND the control condition holds (the real "
               "rule's overlay Sharpe percentile among the seeded random same-size cohort placebo draws must be "
               "at least 0.95); pre_registered_pass=True.")
_NOT_SURVIVED = ("cut_dsr", "cut_holdout", "cut_risk_refusal", "not_scoreable", "holdout_not_evaluable")
_RECORDED_PREFIX = "cash_convention=sweep_cash_daily; benchmark=SPY; tax_policy=realization; "


@pytest.mark.parametrize("verdict,condition,expected", (
    [("survived", True, True), ("survived", False, False), ("survived", None, False),
     ("survived", 1, False), ("survived", "yes", False), ("survived", 1.0, False)]
    + [(v, True, False) for v in _NOT_SURVIVED] + [(v, False, False) for v in _NOT_SURVIVED]
    + [(None, True, False), ("", True, False), ("Survived", True, False)]))
def test_the_pre_registered_pass_needs_both_the_abi1_survival_and_the_control_condition(verdict, condition, expected):
    assert driver.pre_registered_pass(verdict, condition) is expected


@pytest.mark.parametrize("summary,expected", [
    ({"clears_bar": True}, True), ({"clears_bar": False}, False), ({"clears_bar": None}, False),
    ({}, False), ({"clears_bar": 1}, False), (None, False), ({"real_percentile": 0.99}, False)])
def test_the_control_condition_is_exactly_the_placebo_clearing_flag_and_missing_means_not_held(summary, expected):
    assert driver.control_condition_holds(summary) is expected


@pytest.mark.parametrize("condition", [False, None])
def test_a_survived_stage_without_the_control_condition_carries_the_exact_unmissable_flag(condition):
    assert driver.pre_registered_clause("survived", condition) == _FAILED_FLAG


def test_a_survived_stage_whose_control_holds_says_the_pre_registered_rule_was_met():
    clause = driver.pre_registered_clause("survived", True)
    assert clause == _MET_CLAUSE and "FAILED" not in clause


@pytest.mark.parametrize("verdict", _NOT_SURVIVED)
@pytest.mark.parametrize("condition", [True, False, None])
def test_a_stage_that_did_not_survive_carries_no_clause(verdict, condition):
    assert driver.pre_registered_clause(verdict, condition) == ""


def test_the_flag_states_the_rule_of_this_family_and_the_bar_it_uses():
    assert str(driver.PLACEBO_PERCENTILE_BAR) in _FAILED_FLAG
    assert driver.CONTROL_CONDITION_FAILED_FLAG == _FAILED_FLAG
    assert driver.PRE_REGISTERED_RULE_MET_CLAUSE == _MET_CLAUSE
    assert _FAILED_FLAG.startswith("CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under "
                                   "this family's own pre-registered rule (")
    assert _FAILED_FLAG.endswith("). This is not a pass; do not promote it without review.")




@pytest.mark.parametrize("verdict,condition,expected", (
    [("survived", False, _FAILED_FLAG + " "), ("survived", None, _FAILED_FLAG + " "),
     ("survived", True, _MET_CLAUSE + " ")]
    + [(v, c, "") for v in _NOT_SURVIVED for c in (True, False, None)]))
def test_the_note_lead_is_the_clause_and_one_space_or_nothing(verdict, condition, expected):
    """The pure piece of the note builder: what the driver-supplied ledger note is prefixed with."""
    assert driver.ledger_note_lead(verdict, condition) == expected


def test_the_note_lead_puts_the_flag_first_in_any_note_it_prefixes():
    note = driver.ledger_note_lead("survived", False) + "cost model: x"
    assert note.startswith("CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor")
    assert note.endswith(" This is not a pass; do not promote it without review. cost model: x")

def _force_main_verdict_survived(monkeypatch):
    """The synthetic world cannot reach the passing stage: relabel the verdict of the MAIN scored evaluation only
    (compute_and_record's first evaluate_family call; every control arm keeps its real verdict)."""
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


def _force_clears_bar(monkeypatch, value):
    real = driver._placebo_summary

    def forced(placebo):
        summary = dict(real(placebo))
        summary["clears_bar"] = value
        return summary
    monkeypatch.setattr(driver, "_placebo_summary", forced)


def _outcome_note(ledger_path):
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    return outcomes[0]["stage"], outcomes[0]["note"]


@pytest.mark.parametrize("condition", [False, None])
def test_a_recorded_survived_stage_that_failed_its_control_is_flagged_first_and_is_not_a_pre_registered_pass(
        tmp_path, monkeypatch, condition):
    _force_main_verdict_survived(monkeypatch)
    _force_clears_bar(monkeypatch, condition)
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert "reason_code" not in result, result
    assert result["verdict"] == "survived" and result["stage"] == "survived"        # recorded unconditionally
    assert result["pre_registered_pass"] is False
    assert result["pre_registered_clause"] == _FAILED_FLAG
    stage, note = _outcome_note(ledger_path)
    assert stage == "survived"
    # the shared writer puts its fixed metadata prefix first; the driver's own note starts with the flag
    assert note.startswith(_RECORDED_PREFIX + _FAILED_FLAG + " cost model:")
    assert note.count("CONTROL CONDITION FAILED") == 1
    assert "pre_registered_pass=false" in note.lower()


def test_a_recorded_survived_stage_that_passed_its_control_says_the_rule_was_met(tmp_path, monkeypatch):
    _force_main_verdict_survived(monkeypatch)
    _force_clears_bar(monkeypatch, True)
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert "reason_code" not in result, result
    assert result["stage"] == "survived" and result["pre_registered_pass"] is True
    assert result["pre_registered_clause"] == _MET_CLAUSE
    stage, note = _outcome_note(ledger_path)
    assert note.startswith(_RECORDED_PREFIX + _MET_CLAUSE + " cost model:")
    assert "CONTROL CONDITION FAILED" not in note
    assert "pre_registered_pass=true" in note.lower()


def test_a_stage_that_did_not_survive_is_not_a_pre_registered_pass_whatever_the_placebo_says(tmp_path, monkeypatch):
    _force_clears_bar(monkeypatch, True)
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert result["verdict"] != "survived" and result["pre_registered_pass"] is False
    assert result["pre_registered_clause"] == ""
    _, note = _outcome_note(ledger_path)
    assert note.startswith(_RECORDED_PREFIX + "cost model:")                  # no clause: the prefix is unchanged
    assert "CONTROL CONDITION FAILED" not in note and "PRE-REGISTERED RULE MET" not in note
    assert "pre_registered_pass=false" in note.lower()


def test_the_control_condition_is_read_from_the_real_placebo_when_nothing_is_forced(tmp_path, monkeypatch):
    """Only the verdict is relabelled; the placebo is the real seeded one, so the flag must follow its real
    clears_bar (a hard-wired condition would fail one of the two outcomes)."""
    _force_main_verdict_survived(monkeypatch)
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    clears = result["permutation_placebo"]["clears_bar"]
    assert isinstance(clears, bool)
    assert result["pre_registered_pass"] is clears
    assert result["pre_registered_clause"] == (_MET_CLAUSE if clears else _FAILED_FLAG)
    assert _outcome_note(ledger_path)[1].startswith(
        _RECORDED_PREFIX + (_MET_CLAUSE if clears else _FAILED_FLAG) + " cost model:")


def test_a_dry_run_carries_the_same_fields_and_a_non_survivor_is_not_a_pass(base):
    result, _ = base
    assert result["verdict"] != "survived"
    assert result["pre_registered_pass"] is False and result["pre_registered_clause"] == ""
    assert "pre_registered_pass=false" in _flat(result["cost_model_note"]).lower()


def test_the_ledger_note_states_how_the_pass_is_decided(real_run):
    note = _flat(real_run[2])
    assert "pre_registered_pass=" in note
    assert "True only when the ABI-1 verdict is survived AND the placebo percentile bar is cleared" in note


def test_the_printed_summary_says_the_same_as_the_note_when_the_control_failed(tmp_path, monkeypatch, capsys):
    fake_result = {"family_id": mod.FAMILY_ID, "verdict": "survived", "stage": "survived",
                   "pre_registered_pass": False, "pre_registered_clause": _FAILED_FLAG}
    monkeypatch.setattr(driver, "compute_and_record", lambda **kwargs: fake_result)
    quotes = tmp_path / "q.json"
    quotes.write_text("{}")
    rc = driver.main(["--quotes", str(quotes), "--run-id", "r", "--tested-date", "2026-09-24",
                      "--research-root", str(tmp_path / "rr"), "--catalog", str(tmp_path / "c.json"),
                      "--ledger", str(tmp_path / "l.jsonl")])
    lines = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and lines[0] == _FAILED_FLAG                   # its own printed line, before the JSON line
    printed = json.loads(lines[-1])
    assert printed["result"]["pre_registered_pass"] is False and printed["result"]["stage"] == "survived"
    assert len(lines) == 2


def test_the_printed_summary_stays_quiet_when_there_is_no_clause(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(driver, "compute_and_record", lambda **kwargs: {
        "family_id": mod.FAMILY_ID, "verdict": "cut_dsr", "pre_registered_pass": False, "pre_registered_clause": ""})
    quotes = tmp_path / "q.json"
    quotes.write_text("{}")
    assert driver.main(["--dry-run", "--quotes", str(quotes), "--research-root", str(tmp_path / "rr"),
                        "--catalog", str(tmp_path / "c.json"), "--ledger", str(tmp_path / "l.jsonl")]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["result"]["pre_registered_pass"] is False


def test_the_driver_docstring_states_that_the_placebo_gates_the_pass_not_the_recorded_stage():
    doc = _flat(driver.__doc__)
    assert "pre_registered_pass" in doc and "CONTROL CONDITION FAILED" in doc
    assert "recorded stage" in doc.lower() and "closed in shared code" in doc


# ---- the placebo bar itself: characterization of the boundary 

def _placebo_stub(percentile):
    return {"statistic": "s", "n_draws": 500, "seed": 1729, "real_value": 0.1, "draws_mean": 0.0,
            "draws_std": 1.0, "draws_p05": -1.0, "draws_p50": 0.0, "draws_p95": 1.0,
            "real_percentile": percentile, "p_upper": 0.05, "note": "n", "draws": [0.0] * 500}


@pytest.mark.parametrize("percentile,clears", [(1.0, True), (0.951, True), (0.95, True), (0.9499999, False),
                                               (0.9, False), (0.5, False), (0.0, False)])
def test_the_placebo_clears_the_bar_at_exactly_the_pre_declared_percentile_and_not_below(percentile, clears):
    summary = driver._placebo_summary(_placebo_stub(percentile))
    assert summary["clears_bar"] is clears and summary["percentile_bar"] == 0.95
    assert driver.control_condition_holds(summary) is clears
    assert "draws" not in summary and summary["real_percentile"] == percentile


def test_the_percentile_bar_is_the_pre_declared_095_and_the_note_of_a_real_run_states_it(real_run):
    assert driver.PLACEBO_PERCENTILE_BAR == 0.95
    assert "at least 0.95" in _flat(real_run[2])


# ================================================================ builder inputs, gates and the pass flag
# Each test below pins a behaviour that the verdict-shape tests above would not catch if it broke.

def _quote_costs(names=None):
    """The round-trip cost the driver must hand the builders for each panel name, re-derived here from the quotes."""
    return {n: cost_tax_inputs.cost_bps_from_quotes(q["bid"], q["ask"], q["bid"], q["ask"])
            for n, q in _QUOTES.items() if n != "SPY" and (names is None or n in names)}


def _capture_builder_calls(monkeypatch):
    """Wrap the family's scored-curve and placebo builders: run them for real and record what the driver hands
    them (the cost dict, the benchmark entry cost, the prepared windows and the built curve)."""
    seen = {}
    real_build, real_placebo = driver.mod.build_curve, driver.mod.run_placebo

    def build(prepared, *args, **kwargs):
        curve = real_build(prepared, *args, **kwargs)
        seen["build"] = {"costs": dict(kwargs["cost_bps_by_symbol"]), "prepared": prepared, "curve": curve,
                         "spy_entry_cost_bps": kwargs["spy_entry_cost_bps"]}
        return curve

    def placebo(prepared, *args, **kwargs):
        seen["placebo"] = {"costs": dict(kwargs["cost_bps_by_symbol"]),
                           "spy_entry_cost_bps": kwargs["spy_entry_cost_bps"]}
        return real_placebo(prepared, *args, **kwargs)

    monkeypatch.setattr(driver.mod, "build_curve", build)
    monkeypatch.setattr(driver.mod, "run_placebo", placebo)
    return seen


def test_the_scored_curve_and_the_placebo_are_built_with_the_quote_derived_round_trip_costs(tmp_path, monkeypatch):
    """Handing all-zero per-name costs to the scored builder, or to the placebo, would leave every
    check (which reads the real cost dict) and every verdict-shape test green while the permanent ledger
    row is scored cost-free. Pin what each builder RECEIVES, and re-derive the scored curve's per-window cost
    line from the quotes independently of the builder."""
    seen = _capture_builder_calls(monkeypatch)
    result, _ = _run(tmp_path)
    assert "reason_code" not in result, result
    expected = _quote_costs()
    assert len(expected) == 60 and len({round(v, 6) for v in expected.values()}) > 3
    for which in ("build", "placebo"):
        got = seen[which]["costs"]
        assert set(got) == set(expected), which
        for name, value in got.items():
            assert value > 0.0 and value == pytest.approx(expected[name], rel=1e-9), (which, name)
    spy_half_spread = cost_tax_inputs.compute_spread_bps(750.00, 750.15) / 2.0
    assert seen["build"]["spy_entry_cost_bps"] == pytest.approx(spy_half_spread, rel=1e-9)
    assert seen["placebo"]["spy_entry_cost_bps"] == pytest.approx(spy_half_spread, rel=1e-9)
    windows = seen["build"]["prepared"]["windows"]
    trades = seen["build"]["curve"]["trades_df"]
    per_window = [mod.OVERLAY_NOTIONAL * (np.mean([expected[t] for t in w.selected])
                                          + np.mean([expected[t] for t in w.eligible])) / 1e4 for w in windows]
    np.testing.assert_allclose(trades["cost_ret"].to_numpy(dtype=float), per_window, rtol=1e-9)
    assert result["overlay_diagnostics"]["sum_window_cost_ret"] == pytest.approx(sum(per_window), rel=1e-9)
    assert result["overlay_diagnostics"]["sum_window_cost_ret"] > 0.0


def test_the_zero_signal_monthly_control_is_scored_untaxed_with_the_quote_costs_like_its_independent_evaluation(base):
    """Guards against a taxed or cost-free control: the monthly control's verdict, DSR and Sharpe are printed beside the
    result in the permanent note as the survivorship baseline, so it must be the whole-account, untaxed, costed
    evaluation of the monthly control and nothing else (the daily control has its own equivalent test)."""
    from mining import abi1_driver
    result, _ = base
    control = result["zero_signal_control"]
    frames, panel = _world()
    prepared = mod.prepare(frames["SPY"], {s: frames[s] for s in mod.FACTOR_SYMBOLS}, panel,
                           first_month_end=FIRST_ME)
    half_spread = cost_tax_inputs.compute_spread_bps(750.00, 750.15) / 2.0
    kwargs = dict(benchmark_frame=frames["SPY"], benchmark_entry_cost_bps=half_spread, account_type="taxable")
    costed = mod.build_control_curve(prepared, panel, cost_bps_by_symbol=_quote_costs())
    untaxed = abi1_driver.evaluate_family(costed, tax_policy="none", **kwargs)
    assert control["dsr_value"] == pytest.approx(untaxed["layer1"]["dsr_value"], rel=1e-9)
    assert control["sharpe_daily_excess"] == pytest.approx(untaxed["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert control["verdict"] == untaxed["verdict"] and control["risk_account_basis"] == "whole_account"
    # the assertion has power: the same curve scored taxed, and the same basket built cost-free, both differ
    taxed = abi1_driver.evaluate_family(costed, tax_policy=driver.TAX_POLICY, **kwargs)
    assert taxed["layer1"]["sharpe_daily_excess"] != pytest.approx(control["sharpe_daily_excess"], rel=1e-9)
    free = mod.build_control_curve(prepared, panel, cost_bps_by_symbol={n: 0.0 for n in panel})
    free_scored = abi1_driver.evaluate_family(free, tax_policy="none", **kwargs)
    assert free_scored["layer1"]["sharpe_daily_excess"] != pytest.approx(control["sharpe_daily_excess"], rel=1e-9)


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_panel_that_hygiene_thins_below_the_coverage_floor_is_refused_and_nothing_is_recorded(tmp_path, dry_run):
    """The coverage gate AFTER panel hygiene. 70 declared names, all quoted and all loadable, 15 of
    them with an interior gap (hygiene excludes them): 55 of 70 is under the 80 percent floor, and 55 names
    still fill a 50-name cohort, so only this gate stands between a truncated universe and a permanent row."""
    names70 = [f"T{i:02d}" for i in range(70)]
    extra = {n: {"bid": 50.0 + i, "ask": 50.0 + i + 0.02} for i, n in enumerate(names70) if n not in _QUOTES}
    frames, panel = _world(n=70)
    holed = dict(panel)
    for n in names70[:15]:
        holed[n] = panel[n].drop(panel[n].index[300])
    result, ledger_path = _run(tmp_path, dry_run=dry_run, synthetic=dry_run, world=(frames, holed), traded=names70,
                               extra_quotes=extra, placebo_draws=300 if dry_run else mod.PLACEBO_DRAWS)
    assert result.get("reason_code") == "insufficient_panel_coverage", result.get("verdict")
    assert "hygiene" in result["reason"] and "only 55 of 70" in result["reason"]
    assert set(result["hygiene_excluded"]) == set(names70[:15])
    assert all(v["reason"] == "interior_gap" for v in result["hygiene_excluded"].values())
    assert "verdict" not in result and "stage" not in result
    assert not os.path.exists(ledger_path)


def test_a_panel_that_hygiene_thins_to_exactly_the_coverage_floor_is_still_scored(tmp_path):
    """The other side of the same gate: 14 hygiene exclusions of 70 leave exactly 56 (80 percent), which is
    accepted (the floor is a minimum, not a strict bound)."""
    names70 = [f"T{i:02d}" for i in range(70)]
    extra = {n: {"bid": 50.0 + i, "ask": 50.0 + i + 0.02} for i, n in enumerate(names70) if n not in _QUOTES}
    frames, panel = _world(n=70)
    holed = dict(panel)
    for n in names70[:14]:
        holed[n] = panel[n].drop(panel[n].index[300])
    result, _ = _run(tmp_path, world=(frames, holed), traded=names70, extra_quotes=extra)
    assert "reason_code" not in result, result
    assert result["n_panel_names"] == 56


def _frame_ending(last_day, periods=3):
    days = pd.bdate_range(end=last_day, periods=periods)
    return pd.DataFrame({"Close": np.ones(periods)}, index=days)


def test_the_panel_as_of_is_the_most_common_last_bar_not_the_latest_bar_of_any_name():
    """One stray later bar on a single name must not move the panel's as-of date (the market frames
    are cut to it); a tie between two dates goes to the later one."""
    frames = {f"A{i}": _frame_ending("2026-03-30") for i in range(4)}
    frames["STRAY"] = _frame_ending("2026-03-31")
    assert driver._panel_as_of(frames) == pd.Timestamp("2026-03-30")
    tied = {"A": _frame_ending("2026-03-30"), "B": _frame_ending("2026-03-30"),
            "C": _frame_ending("2026-03-31"), "D": _frame_ending("2026-03-31")}
    assert driver._panel_as_of(tied) == pd.Timestamp("2026-03-31")


def test_only_an_abi1_outcome_of_this_family_blocks_the_run_not_another_construction_or_family(tmp_path):
    """The already-scored check is keyed on the (family, construction ABI-1) pair. An older outcome
    of the same family under the per-trade construction, and an ABI-1 outcome of a different family, are not a
    prior trial of this family's ABI-1 score and must not block it."""
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    trial_ledger.record_outcome(ledger_path, mod.FAMILY_ID, "older", "2026-09-01", stage="cut_dsr",
                                dsr_value=0.5, sharpe_in_sample=0.01, ticker_label=None,
                                construction="per_trade", note="an older per-trade outcome")
    trial_ledger.record_outcome(ledger_path, "some_other_family", "other", "2026-09-01", stage="cut_dsr",
                                dsr_value=0.5, sharpe_in_sample=0.01, ticker_label=None,
                                construction="ABI-1", note="a different family")
    before = trial_ledger.read_all(ledger_path)
    frames, panel = _world()
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="rmo-test",
        tested_date="2026-09-24", dry_run=True, signal_source_path="s", signal_source_sha256="0",
        research_root=research_root, ohlc_loader=_loader(frames), panel_loader=_loader(panel),
        traded=NAMES, first_month_end=FIRST_ME, placebo_draws=300)
    assert "reason_code" not in result, result
    assert trial_ledger.read_all(ledger_path) == before          # a dry run adds nothing


@pytest.mark.parametrize("condition", [False, None, True])
def test_a_dry_run_that_survived_but_failed_its_control_is_flagged_and_is_never_a_pre_registered_pass(
        tmp_path, monkeypatch, condition):
    """Guards against a dry run that sets the pass flag True whenever the ABI-1 verdict is the passing stage.
    The dry run is the path a reader sees first; it must apply the same pre-registered rule as a real run."""
    _force_main_verdict_survived(monkeypatch)
    _force_clears_bar(monkeypatch, condition)
    result, ledger_path = _run(tmp_path)                                   # dry run by default
    assert "reason_code" not in result, result
    assert result["verdict"] == "survived" and result["stage"] == "survived"
    assert not os.path.exists(ledger_path)
    if condition is True:
        assert result["pre_registered_pass"] is True and result["pre_registered_clause"] == _MET_CLAUSE
        assert result["cost_model_note"].startswith(_MET_CLAUSE + " cost model:")
        assert "CONTROL CONDITION FAILED" not in result["cost_model_note"]
    else:
        assert result["pre_registered_pass"] is False and result["pre_registered_clause"] == _FAILED_FLAG
        assert result["cost_model_note"].startswith(_FAILED_FLAG + " cost model:")
        assert result["cost_model_note"].count("CONTROL CONDITION FAILED") == 1
    assert f"pre_registered_pass={str(result['pre_registered_pass'])}" in result["cost_model_note"]


def _legacy_flat_merged_payload():
    """A legacy after-hours merged capture shape: a FLAT mapping of symbol to
    quote fields, with no ``quotes`` wrapper, no ``sampled_at_local`` and no ``market_state``."""
    return {name: {"price": q["bid"], "bid": q["bid"], "ask": q["ask"], "security_status": "Normal"}
            for name, q in _QUOTES.items()}


def _refusal_spies(monkeypatch):
    calls = {"ohlc": [], "panel": [], "ledger": []}
    real_record = driver.abi1_driver.record_abi1_outcome

    def ohlc(symbol):
        calls["ohlc"].append(symbol)
        raise AssertionError(f"a refused run must not fetch market history ({symbol})")

    def panel(symbol):
        calls["panel"].append(symbol)
        raise AssertionError(f"a refused run must not read the frozen panel ({symbol})")

    def record(*args, **kwargs):
        calls["ledger"].append(args)
        return real_record(*args, **kwargs)

    monkeypatch.setattr(driver.abi1_driver, "record_abi1_outcome", record)
    return calls, ohlc, panel


@pytest.mark.parametrize("wrapped", [False, True])
def test_a_legacy_flat_merged_after_hours_capture_is_refused_by_a_real_run_with_no_side_effects(
        tmp_path, monkeypatch, wrapped):
    """Guards against a driver without its market-hours refusal call. A legacy merged
    after-hours capture is a FLAT mapping. Handed to a real run as it is, the shared loader refuses it;
    wrapped just far enough to satisfy the loader (its quotes under ``quotes``, an after-hours state and stamp)
    only the driver's own market-hours guard stands in the way. Either way nothing is fetched, read, recorded
    or written."""
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    flat = _legacy_flat_merged_payload()
    payload = ({"quotes": flat, "sampled_at_local": "2026-08-29T20:00:00", "market_state": "after-hours",
                "source": "merged capture"} if wrapped else flat)
    with open(quotes_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    calls, ohlc, panel = _refusal_spies(monkeypatch)
    common = dict(catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="rmo-flat",
                  tested_date="2026-09-24", dry_run=False, signal_source_path="s", signal_source_sha256="0",
                  research_root=research_root, ohlc_loader=ohlc, panel_loader=panel, traded=NAMES,
                  first_month_end=FIRST_ME, placebo_draws=mod.PLACEBO_DRAWS)
    if wrapped:
        result = driver.compute_and_record(**common)
        assert result["reason_code"] == "quotes_not_market_hours" and result["family_id"] == mod.FAMILY_ID
        assert "after-hours" in result["reason"] and "2026-08-29" in result["reason"]
        assert "verdict" not in result and "stage" not in result
    else:
        with pytest.raises(ValueError, match="quotes"):
            driver.compute_and_record(**common)
    assert calls == {"ohlc": [], "panel": [], "ledger": []}
    assert not os.path.exists(ledger_path) and not os.path.exists(research_root)


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
        tmp_path, monkeypatch, real_error, dry_run):
    def fake_load_quotes(path):
        raise ValueError(real_error.format(p=tmp_path))
    monkeypatch.setattr(driver.batch5, "load_quotes", fake_load_quotes)
    with pytest.raises(ValueError, match="must be an object keyed by symbol|unreadable"):
        _run(tmp_path, dry_run=dry_run, synthetic=dry_run, placebo_draws=(3 if dry_run else mod.PLACEBO_DRAWS))


def test_a_metadata_error_under_a_path_that_names_the_other_field_reports_only_the_field_that_failed(
        tmp_path, monkeypatch):
    def fake_load_quotes(path):
        raise ValueError(f"{tmp_path}/market_state/quotes.json: 'sampled_at_local' must be a non-empty string")
    monkeypatch.setattr(driver.batch5, "load_quotes", fake_load_quotes)
    result, _ledger = _run(tmp_path, dry_run=False, synthetic=False, placebo_draws=mod.PLACEBO_DRAWS)
    assert result["reason_code"] == "quotes_not_market_hours"
    assert "sampled_at_local" in result["reason"] and "market_state" not in result["reason"]
    assert str(tmp_path) not in result["reason"]
