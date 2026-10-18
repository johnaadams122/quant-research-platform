import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from mining import options_curve, wheel_engine, account_relative, constants


def _synthetic_inputs(n, price0=100.0, price1=100.0, iv=0.25):
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(np.linspace(price0, price1, n), index=idx)
    iv_series = pd.Series(iv, index=idx)
    r_series = pd.Series(0.04, index=idx)
    return price, iv_series, r_series


def test_build_options_curve_translates_wheel_result_shape():
    price, iv, r = _synthetic_inputs(66)
    cycle = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                    dte=21, profit_target_pct=0.9,
                                    cost_pct_of_premium=0.02)
    curve = options_curve.build_options_curve(cycle)
    assert set(curve) == {"strategy_daily", "in_market", "trades_df"}
    assert curve["strategy_daily"].index.equals(cycle["nav_daily"].index)
    assert curve["in_market"].equals(cycle["in_market"])
    assert list(curve["trades_df"].columns) == [
        "entry_date", "exit_date", "entry_px", "exit_px", "held",
        "reason", "gross_ret", "net_ret"]
    assert len(curve["trades_df"]) == len(cycle["realizations"])
    assert (curve["trades_df"]["gross_ret"] == curve["trades_df"]["net_ret"]).all()


def test_build_options_curve_strategy_daily_first_day_is_zero():
    price, iv, r = _synthetic_inputs(44)
    cycle = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                    dte=21, profit_target_pct=0.9,
                                    cost_pct_of_premium=0.02)
    curve = options_curve.build_options_curve(cycle)
    assert curve["strategy_daily"].iloc[0] == 0.0


@pytest.mark.parametrize("price0,price1,iv", [
    (100.0, 0.01, 0.60),   # crash to near-zero
    (100.0, 400.0, 0.60),  # melt-up
    (100.0, 100.0, 0.25),  # flat/whipsaw-prone
])
def test_build_options_curve_strategy_daily_never_below_negative_one(price0, price1, iv):
    # The property this closes: the design's flagged risk that log_excess
    # (which raises on any daily return <= -100%) could receive a fatal
    # input. Exercised end-to-end through all three drivers.
    n = 130
    price, iv_series, r = _synthetic_inputs(n, price0, price1, iv)
    for cycle in (
        wheel_engine.run_wheel(price, iv_series, r, div_yield=0.0, target_delta=0.15,
                                dte=21, profit_target_pct=0.5, cost_pct_of_premium=0.02),
        wheel_engine.run_pmcc(price, iv_series, r, div_yield=0.0, long_target_delta=0.80,
                               long_dte=252, short_target_delta=0.25, short_dte=21,
                               long_roll_dte_floor=63, roll_trigger_delta=0.60,
                               profit_target_pct=0.5, cost_pct_of_premium=0.03,
                               nav_floor_fraction=0.01),
        wheel_engine.run_put_ratio_spread(price, iv_series, r, div_yield=0.0,
                                           long_delta=0.30, short_delta=0.15,
                                           short_multiple=2, dte=21,
                                           profit_target_pct=0.5, cost_pct_of_premium=0.02),
    ):
        curve = options_curve.build_options_curve(cycle)
        if len(curve["strategy_daily"]) == 0:
            continue
        assert curve["strategy_daily"].min() > -1.0
        assert np.isfinite(curve["strategy_daily"].to_numpy()).all()


def test_build_options_curve_empty_cycle_returns_empty_curve():
    empty_cycle = {"nav_daily": pd.Series(dtype=float),
                   "in_market": pd.Series(dtype=bool, name="in_market"),
                   "realizations": []}
    curve = options_curve.build_options_curve(empty_cycle)
    assert len(curve["strategy_daily"]) == 0
    assert len(curve["trades_df"]) == 0
    assert list(curve["trades_df"].columns) == [
        "entry_date", "exit_date", "entry_px", "exit_px", "held",
        "reason", "gross_ret", "net_ret"]


@pytest.mark.parametrize("driver_name", ["wheel", "pmcc", "ratio_spread"])
def test_build_options_curve_scores_end_to_end_through_account_relative_with_no_errors(driver_name):
    # End-to-end check: feed a real curve straight into
    # account_relative.evaluate() (the actual ABI-1 scoring function, not
    # a mock) and confirm it returns a valid verdict with no exception --
    # proves the translation layer's contract, not just its shape.
    n = 300
    price, iv, r = _synthetic_inputs(n, 100.0, 130.0, 0.25)
    if driver_name == "wheel":
        cycle = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                        dte=21, profit_target_pct=0.5,
                                        cost_pct_of_premium=0.02)
    elif driver_name == "pmcc":
        cycle = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                       long_dte=252, short_target_delta=0.25, short_dte=21,
                                       long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                       profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                       nav_floor_fraction=0.01)
    else:
        cycle = wheel_engine.run_put_ratio_spread(
            price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
            short_multiple=2, dte=21, profit_target_pct=0.5, cost_pct_of_premium=0.02)

    curve = options_curve.build_options_curve(cycle)
    benchmark_daily = pd.Series(0.0003, index=curve["strategy_daily"].index)
    cash_daily = account_relative.sweep_cash_daily(curve["strategy_daily"].index)

    result = account_relative.evaluate(
        strategy_daily=curve["strategy_daily"], in_market=curve["in_market"],
        benchmark_daily=benchmark_daily, cash_daily=cash_daily,
        n_trials=constants.ABI1_N_TRIALS, sr_variance=constants.DAILY_SR_VARIANCE,
        trades_df=curve["trades_df"], tax_policy="realization",
        tax_haircut_rate=0.0, ltcg_tax_rate=0.0)
    assert result["verdict"] in account_relative.VERDICTS


def test_build_options_curve_scores_end_to_end_with_real_tax_netting(monkeypatch):
    # The test above pins tax_haircut_rate=0.0 and ltcg_tax_rate=0.0, which
    # makes evaluate()'s own realization gate --
    # `tax_policy == "realization" and (tax_haircut_rate or ltcg_tax_rate)`
    # (account_relative.py) -- False, so `netted_tax_factors`, the ONLY
    # consumer of trades_df, is never called and trades_df's actual columns
    # (entry_date, exit_date, net_ret) are never validated end-to-end. This
    # test closes that gap: same translation layer, same real (non-mocked)
    # evaluate() call, but at evaluate()'s own default tax rates --
    # constants.STCG_TAX_RATE / constants.LTCG_TAX_RATE, the real production
    # values -- so the gate is True and netted_tax_factors genuinely runs
    # against build_options_curve's trades_df. A monkeypatch spy (wrapping,
    # not replacing, the real implementation) proves it was actually called
    # rather than merely reachable.
    n = 300
    price, iv, r = _synthetic_inputs(n, 100.0, 130.0, 0.25)
    cycle = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                    dte=21, profit_target_pct=0.5,
                                    cost_pct_of_premium=0.02)
    curve = options_curve.build_options_curve(cycle)
    # Guard the premise: with zero trades the gate below would prove nothing.
    assert len(curve["trades_df"]) > 0

    calls = []
    real_netted_tax_factors = account_relative.netted_tax_factors

    def _spy(**kwargs):
        calls.append(kwargs)
        return real_netted_tax_factors(**kwargs)

    monkeypatch.setattr(account_relative, "netted_tax_factors", _spy)

    benchmark_daily = pd.Series(0.0003, index=curve["strategy_daily"].index)
    cash_daily = account_relative.sweep_cash_daily(curve["strategy_daily"].index)

    result = account_relative.evaluate(
        strategy_daily=curve["strategy_daily"], in_market=curve["in_market"],
        benchmark_daily=benchmark_daily, cash_daily=cash_daily,
        n_trials=constants.ABI1_N_TRIALS, sr_variance=constants.DAILY_SR_VARIANCE,
        trades_df=curve["trades_df"], tax_policy="realization",
        tax_haircut_rate=constants.STCG_TAX_RATE, ltcg_tax_rate=constants.LTCG_TAX_RATE)

    # netted_tax_factors was genuinely exercised, not just reachable.
    assert len(calls) == 1
    assert calls[0]["trades_df"] is curve["trades_df"]
    assert result["verdict"] in account_relative.VERDICTS


def test_pool_options_curves_reduces_to_single_member_case():
    price, iv, r = _synthetic_inputs(80)
    cycle = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                    dte=21, profit_target_pct=0.5, cost_pct_of_premium=0.02)
    curve = options_curve.build_options_curve(cycle)
    pooled = options_curve.pool_options_curves({"SPY": curve})
    assert pooled["strategy_daily"].equals(curve["strategy_daily"])
    assert pooled["in_market"].equals(curve["in_market"])
    assert len(pooled["trades_df"]) == len(curve["trades_df"])


def test_pool_options_curves_averages_two_members_and_ors_in_market():
    price_a, iv_a, r_a = _synthetic_inputs(80, 100.0, 105.0)
    price_b, iv_b, r_b = _synthetic_inputs(80, 200.0, 190.0)
    cycle_a = wheel_engine.run_wheel(price_a, iv_a, r_a, div_yield=0.0, target_delta=0.15,
                                      dte=21, profit_target_pct=0.5, cost_pct_of_premium=0.02)
    cycle_b = wheel_engine.run_wheel(price_b, iv_b, r_b, div_yield=0.0, target_delta=0.15,
                                      dte=21, profit_target_pct=0.5, cost_pct_of_premium=0.02)
    curve_a = options_curve.build_options_curve(cycle_a)
    curve_b = options_curve.build_options_curve(cycle_b)
    pooled = options_curve.pool_options_curves({"A": curve_a, "B": curve_b})

    common = curve_a["strategy_daily"].index.intersection(curve_b["strategy_daily"].index)
    day = common[10]
    expected_mean = (curve_a["strategy_daily"].loc[day]
                     + curve_b["strategy_daily"].loc[day]) / 2.0
    assert pooled["strategy_daily"].loc[day] == pytest.approx(expected_mean)
    assert bool(pooled["in_market"].loc[day]) == (
        bool(curve_a["in_market"].loc[day]) or bool(curve_b["in_market"].loc[day]))
    assert len(pooled["trades_df"]) == len(curve_a["trades_df"]) + len(curve_b["trades_df"])


def test_pool_options_curves_rejects_empty_pool():
    with pytest.raises(ValueError, match="per_ticker_curve is empty"):
        options_curve.pool_options_curves({})
