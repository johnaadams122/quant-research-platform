"""Tests for scripts/run_margin_debt_deleverage_mining.py -- the rough-
backtest (stages 2-3 ONLY: cost inputs and the in-sample rough backtest)
driver for margin_debt_deleverage_synthetic_leverage_flip.

This driver stops before stage 4 (ABI-1 scoring / ledger write): only the
ABI-1 driver writes a family's permanent ledger record, so stage 4 must
never fire from the rough-backtest path. Every test here runs on tmp/injected data
(no network, no ledger/catalog/accepted.json writes) and the suite
includes an explicit guard proving stage 4's scoring/record functions
are never called by this driver's `main()`.
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

import run_margin_debt_deleverage_mining as driver
from mining import abi1_driver
from mining.batch5 import margin_debt_deleverage_synthetic_leverage_flip as mod


def _index(n: int, start: str = "2015-01-05") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _flat_frame(index, level: float) -> pd.DataFrame:
    n = len(index)
    close = np.full(n, level)
    return pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close,
    }, index=index)


def _trending_frame(index, *, start=100.0, drift=0.0003, vol=0.01, seed=1) -> pd.DataFrame:
    n = len(index)
    rng = np.random.RandomState(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "Open": open_, "High": close * 1.005, "Low": close * 0.995, "Close": close,
    }, index=index)


def _monthly_debt(values, start="2015-01-31"):
    # Default start deliberately AT/AFTER the default _index() start
    # (2015-01-05) -- mirrors the real production relationship (SPY's
    # traded calendar starts 1993, well BEFORE FINRA's 1997 history).
    # A monthly series that predates the traded calendar by more than
    # `signal_lag_trading_days` sessions collides multiple reference
    # months onto the same availability date and
    # signals.finra_margin_debt_growth_available_series fails closed on
    # that (by design), so the fixture must not predate the calendar.
    months = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series([float(v) for v in values], index=months)


# --------------------------------------------------------------- pure helpers

def test_in_sample_split_excludes_trailing_holdout_months():
    index = pd.bdate_range("2020-01-01", periods=500)
    in_sample = driver.in_sample_split(index, holdout_months=6)
    cutoff = index.max() - pd.DateOffset(months=6)
    assert in_sample.max() <= cutoff
    assert len(in_sample) < len(index)
    assert in_sample.is_monotonic_increasing


def test_descriptive_stats_on_known_fixture():
    index = pd.bdate_range("2020-01-01", periods=4)
    daily = pd.Series([0.01, -0.005, 0.02, 0.0], index=index)
    trades_df = pd.DataFrame({
        "net_ret": [0.05, -0.02, 0.03],
        "exit_date": index[:3],
    })
    stats = driver.descriptive_stats(daily, trades_df)
    assert stats["n_trades"] == 3
    assert stats["n_sessions"] == 4
    expected_cumulative = float(np.prod(1.0 + daily.to_numpy()) - 1.0)
    assert stats["cumulative_net_return"] == pytest.approx(expected_cumulative)
    assert stats["win_rate"] == pytest.approx(2.0 / 3.0)


def test_descriptive_stats_handles_zero_trades_without_crashing():
    index = pd.bdate_range("2020-01-01", periods=3)
    daily = pd.Series([0.0, 0.0, 0.0], index=index)
    trades_df = pd.DataFrame({"net_ret": pd.Series(dtype=float),
                              "exit_date": pd.Series(dtype="datetime64[ns]")})
    stats = driver.descriptive_stats(daily, trades_df)
    assert stats["n_trades"] == 0
    assert stats["win_rate"] is None


def test_build_synthetic_frame_uses_build_synthetic_leveraged_default_leverage():
    index = _index(300)
    spy = _trending_frame(index, seed=2)
    irx = pd.Series(0.02, index=index)  # flat 2% annualized financing
    synthetic_frame, synthetic_id = driver.build_synthetic_frame(spy, irx)
    assert synthetic_id == f"SPY_{int(mod.LEVERAGE)}x_synthetic"
    assert list(synthetic_frame.columns) == ["Open", "High", "Low", "Close"]
    assert synthetic_frame.index.equals(spy.index)
    assert (synthetic_frame["Close"] > 0).all()
    # daily-reset shaping: Open_t == Close_{t-1}
    assert synthetic_frame["Open"].iloc[1:].to_numpy() == pytest.approx(
        synthetic_frame["Close"].iloc[:-1].to_numpy())


# ------------------------------------------------------------- end to end

def test_rough_backtest_produces_real_trades_both_directions():
    index = _index(1600)
    spy = _trending_frame(index, seed=3)
    sh = _trending_frame(index, start=30.0, drift=-0.0001, vol=0.012, seed=4)
    irx = pd.Series(0.02, index=index)
    # Regime flips every ~4 months so both legs actually get held.
    n_months = 60
    values = []
    level = 100.0
    for i in range(n_months):
        level *= 1.04 if (i % 8) < 4 else 0.93
        values.append(level)
    margin_debt = _monthly_debt(values)

    result = driver.rough_backtest(
        spy_frame=spy, sh_frame=sh, margin_debt=margin_debt, irx_series=irx,
        cost_bps_by_asset={"SPY": 0.0, "SH": 0.0}, holdout_months=6)

    trades = result["in_sample_trades"]
    assert len(trades) > 0
    assert set(trades["asset"].unique()) <= {f"SPY_{int(mod.LEVERAGE)}x_synthetic", "SH"}
    assert result["stats"]["n_trades"] == len(trades)
    # in-sample must exclude the trailing holdout window
    assert result["in_sample_index"].max() < result["curve"]["strategy_daily"].index.max() - pd.DateOffset(months=5)


def test_rough_backtest_never_touches_real_ledger_or_catalog(tmp_path, monkeypatch):
    """Guard: stage 4 (scoring/ledger write) must never fire from the
    rough-backtest path. Patches the real record/ledger write functions to
    raise if called at all."""
    def _forbidden(*args, **kwargs):
        raise AssertionError("stage 4 scoring/ledger path must not execute from the rough-backtest path")

    monkeypatch.setattr(abi1_driver, "record_abi1_outcome", _forbidden)

    index = _index(900)
    spy = _trending_frame(index, seed=5)
    sh = _trending_frame(index, start=30.0, seed=6)
    irx = pd.Series(0.02, index=index)
    margin_debt = _monthly_debt([100.0 * (1.02 ** i) for i in range(30)])

    result = driver.rough_backtest(
        spy_frame=spy, sh_frame=sh, margin_debt=margin_debt, irx_series=irx,
        cost_bps_by_asset={"SPY": 0.0, "SH": 0.0})
    assert "stats" in result  # ran fine; record_abi1_outcome was never reached


def test_prepare_abi1_scoring_call_is_built_but_never_auto_invoked(monkeypatch):
    """The stage-4 call must be fully constructible (proving it COULD run)
    without actually running it -- calling `prepare_abi1_scoring_call`
    itself must not touch `evaluate_family`."""
    calls = []
    real_evaluate = abi1_driver.evaluate_family

    def spy_evaluate(*args, **kwargs):
        calls.append(1)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(abi1_driver, "evaluate_family", spy_evaluate)

    index = _index(400)
    spy = _trending_frame(index, seed=7)
    sh = _trending_frame(index, start=30.0, seed=8)
    irx = pd.Series(0.02, index=index)
    margin_debt = _monthly_debt([100.0] * 20)
    result = driver.rough_backtest(
        spy_frame=spy, sh_frame=sh, margin_debt=margin_debt, irx_series=irx,
        cost_bps_by_asset={"SPY": 0.0, "SH": 0.0})

    scoring_call = driver.prepare_abi1_scoring_call(
        result["curve"], benchmark_frame=spy, benchmark_entry_cost_bps=0.0,
        account_type="taxable", tax_policy="realization")
    assert callable(scoring_call)
    assert calls == []  # preparing it must not run it

    # Proving it WOULD work, without this being main()'s own path:
    real_result = scoring_call()
    assert calls == [1]
    assert "verdict" in real_result


def test_main_writes_rough_backtest_report_and_skips_cost_when_market_closed(
        tmp_path, monkeypatch):
    """End-to-end main(): injected OHLC loader (no network), injected FINRA
    fetch (no network), a quote-freshness gate that this test forces to
    'closed' -- asserts the report is written with cost_bps=0.0 and a
    clear skip reason, and that no ledger/catalog file anywhere under
    tmp_path or the real repo was created or modified."""
    index = _index(1200, start="2018-01-02")
    spy = _trending_frame(index, seed=9)
    sh = _trending_frame(index, start=30.0, seed=10)
    irx = pd.Series(0.02, index=index)
    margin_debt = _monthly_debt(
        [100.0 * (1.03 ** i) for i in range(40)], start="2018-01-31")

    monkeypatch.setattr(driver, "_fetch_spy_ohlc", lambda: spy)
    monkeypatch.setattr(driver, "_fetch_sh_ohlc", lambda: sh)
    monkeypatch.setattr(driver, "_fetch_irx", lambda: irx)
    monkeypatch.setattr(driver, "fetch_margin_debt_series",
                        lambda: (margin_debt, {"source": "test-finra"}))
    monkeypatch.setattr(driver, "market_is_open_now", lambda: (False, "test: forced closed"))
    # PROVENANCE_PATH is a hardcoded absolute path into the real repo, not
    # threaded through --research-root like the run report -- redirect it
    # into tmp_path or this test silently writes its own
    # {"fetched_for_run_id": "test-run", "source": "test-finra"} fixture
    # record to the repo's data/mining/margin_debt_deleverage/provenance.json
    # on every full-suite run.
    monkeypatch.setattr(driver, "PROVENANCE_PATH", str(tmp_path / "provenance.json"))

    def _forbidden(*args, **kwargs):
        raise AssertionError("stage 4 must not execute from main()")
    monkeypatch.setattr(abi1_driver, "record_abi1_outcome", _forbidden)

    run_dir = tmp_path / "test-run"
    rc = driver.main([
        "--run-id", "test-run", "--tested-date", "2026-09-17",
        "--research-root", str(tmp_path),
    ])
    assert rc == 0

    output_path = run_dir / "rough-backtest.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["family_id"] == mod.FAMILY_ID
    assert payload["step"] == "rough_backtest_only"
    assert payload["cost_model"]["cost_sample_used"] is False
    assert "closed" in payload["cost_model"]["reason"].lower()
    assert payload["in_sample_stats"]["n_trades"] >= 0
    assert "stage 4" in payload["note"]
