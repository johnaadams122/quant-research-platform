import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from options_engine import bs_price, bs_delta, solve_strike_for_delta
from options_engine import build_iron_condor, BASELINE_PARAMS


def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    parity_rhs = S - K * np.exp(-r * T)
    assert call - put == pytest.approx(parity_rhs, abs=1e-8)


def test_call_delta_bounds():
    for K in [80, 100, 120]:
        d = bs_delta(100.0, K, 0.1, 0.04, 0.20, "call")
        assert 0.0 <= d <= 1.0


def test_put_delta_bounds():
    for K in [80, 100, 120]:
        d = bs_delta(100.0, K, 0.1, 0.04, 0.20, "put")
        assert -1.0 <= d <= 0.0


def test_price_monotonic_in_sigma():
    S, K, T, r = 100.0, 105.0, 0.1, 0.04
    sigmas = [0.10, 0.15, 0.20, 0.25, 0.30]
    call_prices = [bs_price(S, K, T, r, s, "call") for s in sigmas]
    put_prices = [bs_price(S, K, T, r, s, "put") for s in sigmas]
    assert all(b > a for a, b in zip(call_prices, call_prices[1:]))
    assert all(b > a for a, b in zip(put_prices, put_prices[1:]))


def test_zero_time_is_intrinsic():
    assert bs_price(110.0, 100.0, 0.0, 0.04, 0.20, "call") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.0, 0.04, 0.20, "call") == pytest.approx(0.0)
    assert bs_price(90.0, 100.0, 0.0, 0.04, 0.20, "put") == pytest.approx(10.0)
    assert bs_price(110.0, 100.0, 0.0, 0.04, 0.20, "put") == pytest.approx(0.0)


def test_solve_strike_for_call_delta_roundtrip():
    S, T, r, sigma = 580.0, 0.002, 0.04, 0.15
    target = 0.125
    K = solve_strike_for_delta(S, T, r, sigma, target, "call")
    assert K > S  # OTM call strike is above spot
    assert bs_delta(S, K, T, r, sigma, "call") == pytest.approx(target, abs=1e-4)


def test_solve_strike_for_put_delta_roundtrip():
    S, T, r, sigma = 580.0, 0.002, 0.04, 0.15
    target = -0.125
    K = solve_strike_for_delta(S, T, r, sigma, target, "put")
    assert K < S  # OTM put strike is below spot
    assert bs_delta(S, K, T, r, sigma, "put") == pytest.approx(target, abs=1e-4)


def test_build_iron_condor_structure():
    ic = build_iron_condor(S=580.0, iv=0.15, dte=0, **{
        k: v for k, v in BASELINE_PARAMS.items() if k != "stop_pct" and k != "pt_pct"
    })
    assert ic["long_put_K"] < ic["short_put_K"] < 580.0 < ic["short_call_K"] < ic["long_call_K"]
    assert ic["entry_credit"] > 0
    assert ic["max_loss"] > 0
    assert ic["wing_width"] == pytest.approx(580.0 * BASELINE_PARAMS["wing_width_pct"])


def test_build_iron_condor_max_loss_consistency():
    ic = build_iron_condor(S=480.0, iv=0.18, dte=0, **{
        k: v for k, v in BASELINE_PARAMS.items() if k != "stop_pct" and k != "pt_pct"
    })
    assert ic["max_loss"] == pytest.approx((ic["wing_width"] - ic["entry_credit"]) * 100)


from options_engine import simulate_trade


def test_simulate_trade_profit_target_on_vol_crush():
    ic = build_iron_condor(S=580.0, iv=0.20, dte=0, **{
        k: v for k, v in BASELINE_PARAMS.items() if k != "stop_pct" and k != "pt_pct"
    })
    # Flat day, but IV crushes from 0.20 to near zero -> condor cheapens a lot.
    day_ohlc = {"Open": 580.0, "High": 580.0, "Low": 580.0, "Close": 580.0}
    result = simulate_trade(ic, day_ohlc, iv_exit=0.01,
                             pt_pct=BASELINE_PARAMS["pt_pct"],
                             stop_pct=BASELINE_PARAMS["stop_pct"])
    assert result["exit_reason"] == "profit_target"
    expected_r = (ic["entry_credit"] * BASELINE_PARAMS["pt_pct"] * 100) / ic["max_loss"]
    assert result["r_multiple"] == pytest.approx(expected_r)


def test_simulate_trade_stop_on_large_move():
    ic = build_iron_condor(S=580.0, iv=0.15, dte=0, **{
        k: v for k, v in BASELINE_PARAMS.items() if k != "stop_pct" and k != "pt_pct"
    })
    # Huge up-move blows through the call spread.
    day_ohlc = {"Open": 580.0, "High": 870.0, "Low": 580.0, "Close": 850.0}
    result = simulate_trade(ic, day_ohlc, iv_exit=0.15,
                             pt_pct=BASELINE_PARAMS["pt_pct"],
                             stop_pct=BASELINE_PARAMS["stop_pct"])
    assert result["exit_reason"] == "stop"
    expected_r = -(ic["entry_credit"] * BASELINE_PARAMS["stop_pct"] * 100) / ic["max_loss"]
    assert result["r_multiple"] == pytest.approx(expected_r)


def test_simulate_trade_expiry_when_pt_and_stop_unreachable():
    ic = build_iron_condor(S=580.0, iv=0.15, dte=0, **{
        k: v for k, v in BASELINE_PARAMS.items() if k != "stop_pct" and k != "pt_pct"
    })
    # Flat day, same IV. With pt_pct/stop_pct set far outside [-1, 1] (the max
    # possible pnl range as a fraction of entry_credit), neither can trigger.
    day_ohlc = {"Open": 580.0, "High": 580.0, "Low": 580.0, "Close": 580.0}
    result = simulate_trade(ic, day_ohlc, iv_exit=0.15, pt_pct=10.0, stop_pct=10.0)
    assert result["exit_reason"] == "expiry"
    # Close is between the short strikes -> both spreads expire worthless.
    expected_r = (ic["entry_credit"] * 100) / ic["max_loss"]
    assert result["r_multiple"] == pytest.approx(expected_r)


def test_bs_price_with_dividend_yield_matches_dividend_put_call_parity():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.03
    call = bs_price(S, K, T, r, sigma, "call", q=q)
    put = bs_price(S, K, T, r, sigma, "put", q=q)
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert call - put == pytest.approx(parity_rhs, abs=1e-8)


def test_bs_price_q_default_zero_is_byte_identical_to_no_q_call():
    S, K, T, r, sigma = 100.0, 105.0, 0.25, 0.04, 0.22
    assert bs_price(S, K, T, r, sigma, "call") == bs_price(S, K, T, r, sigma, "call", q=0.0)
    assert bs_delta(S, K, T, r, sigma, "put") == bs_delta(S, K, T, r, sigma, "put", q=0.0)


def test_bs_delta_positive_dividend_yield_reduces_call_delta():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    delta_no_div = bs_delta(S, K, T, r, sigma, "call", q=0.0)
    delta_with_div = bs_delta(S, K, T, r, sigma, "call", q=0.03)
    assert delta_with_div < delta_no_div


def test_solve_strike_for_deep_itm_leaps_call_delta():
    # A PMCC long LEAPS leg needs target_delta ~0.80 -- unreachable in a
    # [S, 3S] strike bracket (that bracket only reaches K >= S).
    S, T, r, sigma, q = 100.0, 1.0, 0.04, 0.25, 0.01
    K = solve_strike_for_delta(S, T, r, sigma, 0.80, "call", q=q)
    assert K < S
    assert bs_delta(S, K, T, r, sigma, "call", q=q) == pytest.approx(0.80, abs=1e-4)


def test_solve_strike_for_delta_q_default_zero_still_resolves_pre_widen_target_delta():
    # Backward-compat: an ordinary OTM call target (reachable in the OLD
    # [S,3S] bracket too) still resolves, after widening to [0.05*S, 3S],
    # to a strike that hits the target delta within tolerance -- bisection
    # monotonicity is unaffected. This does NOT assert strike-identity:
    # a wider starting bracket makes bisection visit a different sequence
    # of midpoints, so the resulting K can differ from the pre-widen K by
    # a small amount (~1e-5 in this case) even though both Ks hit the same
    # target delta within tol. See solve_strike_for_delta's docstring.
    S, T, r, sigma = 580.0, 0.002, 0.04, 0.15
    K = solve_strike_for_delta(S, T, r, sigma, 0.125, "call")
    assert bs_delta(S, K, T, r, sigma, "call") == pytest.approx(0.125, abs=1e-4)
