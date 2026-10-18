import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from options_engine import bs_price, bs_delta
from mining import options_positions as op

S, IV, R, Q, DTE = 100.0, 0.20, 0.04, 0.01, 21


def test_build_short_put_strike_matches_target_delta():
    pos = op.build_short_put(S, IV, DTE, target_delta=0.15, r=R, q=Q)
    T = DTE / 252
    assert bs_delta(S, pos["K"], T, R, IV, "put", q=Q) == pytest.approx(-0.15, abs=1e-4)
    assert pos["K"] < S
    assert pos["entry_credit"] == pytest.approx(
        bs_price(S, pos["K"], T, R, IV, "put", q=Q))
    assert pos["entry_credit"] > 0
    assert pos["T_entry"] == pytest.approx(T)


def test_build_covered_call_strike_matches_target_delta():
    pos = op.build_covered_call(S, IV, DTE, target_delta=0.20, r=R, q=Q, cost_basis=98.0)
    T = DTE / 252
    assert bs_delta(S, pos["K"], T, R, IV, "call", q=Q) == pytest.approx(0.20, abs=1e-4)
    assert pos["K"] > S
    assert pos["entry_credit"] == pytest.approx(
        bs_price(S, pos["K"], T, R, IV, "call", q=Q))
    assert pos["cost_basis"] == 98.0


def test_build_pmcc_leaps_is_deep_itm_and_costs_a_debit():
    pos = op.build_pmcc_leaps(S, IV, dte=252, target_delta=0.80, r=R, q=Q)
    T = 252 / 252
    assert pos["K"] < S  # deep ITM
    assert bs_delta(S, pos["K"], T, R, IV, "call", q=Q) == pytest.approx(0.80, abs=1e-4)
    assert pos["entry_cost"] == pytest.approx(bs_price(S, pos["K"], T, R, IV, "call", q=Q))
    assert pos["entry_cost"] > 0


def test_build_pmcc_short_call_is_otm_and_collects_a_credit():
    pos = op.build_pmcc_short_call(S, IV, DTE, target_delta=0.25, r=R, q=Q)
    T = DTE / 252
    assert pos["K"] > S  # OTM
    assert bs_delta(S, pos["K"], T, R, IV, "call", q=Q) == pytest.approx(0.25, abs=1e-4)
    assert pos["entry_credit"] > 0


def test_build_pmcc_leaps_long_strike_is_below_short_call_strike():
    # The defining PMCC shape: deep-ITM long strike well below the OTM
    # short strike (a real "diagonal" spread, not an overlapping pair).
    leaps = op.build_pmcc_leaps(S, IV, dte=252, target_delta=0.80, r=R, q=Q)
    short = op.build_pmcc_short_call(S, IV, DTE, target_delta=0.25, r=R, q=Q)
    assert leaps["K"] < short["K"]


def test_build_put_ratio_spread_structure():
    pos = op.build_put_ratio_spread(
        S, IV, DTE, long_delta=0.30, short_delta=0.15, short_multiple=2, r=R, q=Q)
    T = DTE / 252
    assert pos["K_long"] > pos["K_short"]  # long put closer to money
    assert pos["short_multiple"] == 2
    long_price = bs_price(S, pos["K_long"], T, R, IV, "put", q=Q)
    short_price = bs_price(S, pos["K_short"], T, R, IV, "put", q=Q)
    assert pos["entry_credit"] == pytest.approx(2 * short_price - long_price)


def test_build_put_ratio_spread_rejects_short_multiple_below_two():
    # short_multiple=1 degenerates to a plain (non-ratio) put spread --
    # not the strategy under test; short_multiple must create real
    # "naked excess" short legs for the family this driver curates.
    with pytest.raises(ValueError, match="short_multiple"):
        op.build_put_ratio_spread(
            S, IV, DTE, long_delta=0.30, short_delta=0.15, short_multiple=1, r=R, q=Q)


import numpy as np
import pandas as pd


def _flat_frame(n, price=100.0, start="2020-01-02"):
    idx = pd.date_range(start, periods=n, freq="B")
    return (pd.Series(price, index=idx),  # price_series
            pd.Series(0.20, index=idx),   # iv_series
            pd.Series(0.04, index=idx))   # r_series


def test_short_put_nav_path_nav0_equals_k_minus_entry_cost():
    price, iv, r = _flat_frame(25)
    pos = op.build_short_put(100.0, 0.20, 21, target_delta=0.15, r=0.04, q=0.0)
    path = op.short_put_nav_path(pos["K"], pos["entry_credit"], 0, price, iv, r,
                                  q=0.0, dte=21, profit_target_pct=0.5,
                                  cost_pct_of_premium=0.02)
    expected_nav0 = pos["K"] + pos["entry_credit"] * (1 - 0.02) - pos["entry_credit"]
    assert path["nav"].iloc[0] == pytest.approx(expected_nav0)
    assert path["nav"].iloc[0] > 0


def test_short_put_nav_path_never_goes_non_positive_through_a_crash_to_zero():
    # A deterministic linear crash from 100 straight to 0.01 over the
    # whole cycle -- the worst realistic case (S can't hit exactly 0
    # while T>0 without a log(0) domain error in bs_price's own d1).
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = np.linspace(100.0, 0.01, n)
    price = pd.Series(prices, index=idx)
    iv = pd.Series(0.60, index=idx)  # crash-regime vol spike
    r = pd.Series(0.04, index=idx)
    pos = op.build_short_put(100.0, 0.20, 21, target_delta=0.15, r=0.04, q=0.0)
    path = op.short_put_nav_path(pos["K"], pos["entry_credit"], 0, price, iv, r,
                                  q=0.0, dte=21, profit_target_pct=0.5,
                                  cost_pct_of_premium=0.02)
    assert (path["nav"] > 0).all()
    entry_credit_net = pos["entry_credit"] * 0.98
    assert path["nav"].iloc[-1] == pytest.approx(entry_credit_net, rel=0.05)


def test_short_put_nav_path_profit_target_signal_fires_on_decay():
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(101.0, index=idx)  # drifts slightly OTM, IV crushes
    iv = pd.Series(np.linspace(0.20, 0.02, n), index=idx)
    r = pd.Series(0.04, index=idx)
    pos = op.build_short_put(100.0, 0.20, 21, target_delta=0.15, r=0.04, q=0.0)
    path = op.short_put_nav_path(pos["K"], pos["entry_credit"], 0, price, iv, r,
                                  q=0.0, dte=21, profit_target_pct=0.5,
                                  cost_pct_of_premium=0.02)
    assert path["close_signal"].any()


def test_covered_call_nav_path_nav0_equals_spot_minus_entry_cost():
    price, iv, r = _flat_frame(25)
    pos = op.build_covered_call(100.0, 0.20, 21, target_delta=0.20, r=0.04, q=0.0,
                                 cost_basis=98.0)
    path = op.covered_call_nav_path(pos["K"], pos["entry_credit"], 0, price, iv, r,
                                     q=0.0, dte=21, profit_target_pct=0.5,
                                     cost_pct_of_premium=0.02)
    expected_nav0 = 100.0 - (pos["entry_credit"] - pos["entry_credit"] * 0.98)
    assert path["nav"].iloc[0] == pytest.approx(expected_nav0)


def test_covered_call_nav_path_never_goes_non_positive_through_a_melt_up():
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = np.linspace(100.0, 400.0, n)  # a 4x melt-up
    price = pd.Series(prices, index=idx)
    iv = pd.Series(0.60, index=idx)
    r = pd.Series(0.04, index=idx)
    pos = op.build_covered_call(100.0, 0.20, 21, target_delta=0.20, r=0.04, q=0.0,
                                 cost_basis=98.0)
    path = op.covered_call_nav_path(pos["K"], pos["entry_credit"], 0, price, iv, r,
                                     q=0.0, dte=21, profit_target_pct=0.5,
                                     cost_pct_of_premium=0.02)
    assert (path["nav"] > 0).all()


def test_pmcc_nav_path_nav0_equals_entry_net_debit_cost_inclusive():
    price, iv, r = _flat_frame(25)
    leaps = op.build_pmcc_leaps(100.0, 0.20, dte=252, target_delta=0.80, r=0.04, q=0.0)
    short = op.build_pmcc_short_call(100.0, 0.20, 21, target_delta=0.25, r=0.04, q=0.0)
    assert leaps["entry_cost"] > short["entry_credit"]  # real net-debit precondition
    path = op.pmcc_nav_path(leaps["K"], short["K"], leaps["T_entry"], leaps["entry_cost"],
                             short["entry_credit"], 0, price, iv, r, q=0.0, short_dte=21,
                             roll_trigger_delta=0.60, profit_target_pct=0.5,
                             cost_pct_of_premium=0.02, nav_floor_fraction=0.01,
                             long_leg_is_fresh_entry=True)
    expected_net_debit = (leaps["entry_cost"] * 1.02 - short["entry_credit"] * 0.98)
    assert path["nav"].iloc[0] == pytest.approx(expected_net_debit)
    assert path["nav"].iloc[0] > 0


def test_pmcc_nav_path_never_goes_non_positive_through_a_crash_to_zero():
    # A crash to near-zero drives BOTH call legs toward worthless -- the
    # raw (long - short) mark can dip low; the disclosed floor + the
    # cost_offset's own constant contribution must keep NAV positive.
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = np.linspace(100.0, 0.01, n)
    price = pd.Series(prices, index=idx)
    iv = pd.Series(0.80, index=idx)
    r = pd.Series(0.04, index=idx)
    leaps = op.build_pmcc_leaps(100.0, 0.20, dte=252, target_delta=0.80, r=0.04, q=0.0)
    short = op.build_pmcc_short_call(100.0, 0.20, 21, target_delta=0.25, r=0.04, q=0.0)
    path = op.pmcc_nav_path(leaps["K"], short["K"], leaps["T_entry"], leaps["entry_cost"],
                             short["entry_credit"], 0, price, iv, r, q=0.0, short_dte=21,
                             roll_trigger_delta=0.60, profit_target_pct=0.5,
                             cost_pct_of_premium=0.02, nav_floor_fraction=0.01,
                             long_leg_is_fresh_entry=True)
    assert (path["nav"] > 0).all()


def test_pmcc_nav_path_floor_clamp_engages_with_a_generous_floor():
    # A large nav_floor_fraction (0.5, far above the real disclosed
    # default of 0.01) forces the clamp to engage on any real spread-value
    # decline -- proves the clamp MECHANISM actually works, independent
    # of whether the tiny production-default floor happens to bind in any
    # one particular synthetic scenario (call-price monotonicity in
    # strike/time means the raw mark rarely approaches the small default
    # floor in a realistic PMCC parameter regime).
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(np.linspace(100.0, 60.0, n), index=idx)  # a moderate decline
    iv = pd.Series(0.30, index=idx)
    r = pd.Series(0.04, index=idx)
    leaps = op.build_pmcc_leaps(100.0, 0.20, dte=252, target_delta=0.80, r=0.04, q=0.0)
    short = op.build_pmcc_short_call(100.0, 0.20, 21, target_delta=0.25, r=0.04, q=0.0)
    path = op.pmcc_nav_path(leaps["K"], short["K"], leaps["T_entry"], leaps["entry_cost"],
                             short["entry_credit"], 0, price, iv, r, q=0.0, short_dte=21,
                             roll_trigger_delta=0.60, profit_target_pct=0.99,
                             cost_pct_of_premium=0.02, nav_floor_fraction=0.5,
                             long_leg_is_fresh_entry=True)
    assert path["clamped"].any()
    entry_net_debit = leaps["entry_cost"] * 1.02 - short["entry_credit"] * 0.98
    floor_basis = max(entry_net_debit, leaps["entry_cost"])
    clamped_rows = path.loc[path["clamped"]]
    assert clamped_rows["nav"].to_numpy() == pytest.approx(0.5 * floor_basis)


def test_pmcc_nav_path_roll_trigger_fires_on_deep_itm_short_delta():
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = np.linspace(100.0, 140.0, n)  # rally pushes short call deep ITM
    price = pd.Series(prices, index=idx)
    iv = pd.Series(0.20, index=idx)
    r = pd.Series(0.04, index=idx)
    leaps = op.build_pmcc_leaps(100.0, 0.20, dte=252, target_delta=0.80, r=0.04, q=0.0)
    short = op.build_pmcc_short_call(100.0, 0.20, 21, target_delta=0.25, r=0.04, q=0.0)
    path = op.pmcc_nav_path(leaps["K"], short["K"], leaps["T_entry"], leaps["entry_cost"],
                             short["entry_credit"], 0, price, iv, r, q=0.0, short_dte=21,
                             roll_trigger_delta=0.60, profit_target_pct=0.5,
                             cost_pct_of_premium=0.02, nav_floor_fraction=0.01,
                             long_leg_is_fresh_entry=True)
    assert path["close_signal"].any()


def test_pmcc_nav_path_negative_entry_net_debit_clamps_instead_of_raising():
    # Guards against raising on a negative entry_net_debit. The LEAPS
    # strike is FIXED at its own entry while a fresh short call is priced
    # against the CURRENT spot every sub-cycle -- an ordinary decline
    # within one LEAPS holding window can drive entry_net_debit negative
    # through completely ordinary price action (here: a 40% decline,
    # 100 -> 60, 189 sessions into the LEAPS). This must clamp gracefully
    # (the whole point of the floor mechanism), not raise -- raising here
    # would discard a ticker's ENTIRE curve, not just the affected window,
    # in run_pmcc's while-loop.
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(np.linspace(60.0, 58.0, n), index=idx)
    iv = pd.Series(0.30, index=idx)
    r = pd.Series(0.04, index=idx)
    leaps = op.build_pmcc_leaps(100.0, 0.20, dte=252, target_delta=0.80, r=0.04, q=0.0)
    long_mark_now = bs_price(60.0, leaps["K"], leaps["T_entry"] - 189 / 252, 0.04, 0.30,
                             "call", q=0.0)
    short = op.build_pmcc_short_call(60.0, 0.30, 21, target_delta=0.25, r=0.04, q=0.0)
    assert long_mark_now - short["entry_credit"] < 0  # precondition: a real negative case
    path = op.pmcc_nav_path(leaps["K"], short["K"], leaps["T_entry"] - 189 / 252, long_mark_now,
                             short["entry_credit"], 0, price, iv, r, q=0.0, short_dte=21,
                             roll_trigger_delta=0.60, profit_target_pct=0.5,
                             cost_pct_of_premium=0.03, nav_floor_fraction=0.01,
                             long_leg_is_fresh_entry=False)
    assert (path["nav"] > 0).all()


def test_pmcc_nav_path_double_degenerate_floor_stays_strictly_positive():
    # Guards the DOUBLE degeneracy a deep crash can produce (e.g. a
    # 130-day 100.0 -> 0.01 crash at iv=0.60 fed through run_pmcc).
    # Unlike the negative-entry_net_debit test above (where
    # long_cost_gross stays a real positive re-mark), this constructs: a
    # deep-enough-OTM LEAPS re-mark that genuinely underflows to EXACTLY
    # 0.0 in float64 at extreme moneyness, occurring simultaneously with
    # a non-positive entry_net_debit (the short leg's own fresh credit
    # exceeding the LEAPS' now-worthless value -- an ordinary outcome of
    # the SAME crash, not a rare coincidence). Without the
    # PMCC_FLOOR_BASIS_EPSILON term, max(entry_net_debit, long_cost_gross)
    # is exactly 0.0, so the floor itself would be 0.0 and the clamp would
    # produce nav_t == 0.0 exactly -- which would raise ZeroDivisionError
    # one level up in run_pmcc's _advance_cycle (realized_nav_exit /
    # cycle_base, no zero guard).
    S_crash, K_long, T_long_remaining, iv_val, r_val = 1.56, 75.20, 0.496, 0.60, 0.04
    long_cost_gross = bs_price(S_crash, K_long, T_long_remaining, r_val, iv_val,
                                "call", q=0.0)
    assert long_cost_gross == 0.0  # precondition: a real float64 underflow, not a stub
    short = op.build_pmcc_short_call(S_crash, iv_val, 21, target_delta=0.25, r=r_val, q=0.0)
    n = 5
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(S_crash, index=idx)
    iv = pd.Series(iv_val, index=idx)
    r = pd.Series(r_val, index=idx)
    long_cost_component = long_cost_gross  # long_leg_is_fresh_entry=False below
    short_credit_component = short["entry_credit"] * (1.0 - 0.02)
    entry_net_debit = long_cost_component - short_credit_component
    assert entry_net_debit < 0  # precondition: BOTH terms of max() non-positive
    assert max(entry_net_debit, long_cost_gross) == 0.0  # the degeneracy the epsilon term covers
    path = op.pmcc_nav_path(K_long, short["K"], T_long_remaining, long_cost_gross,
                             short["entry_credit"], 0, price, iv, r, q=0.0, short_dte=21,
                             roll_trigger_delta=0.60, profit_target_pct=0.5,
                             cost_pct_of_premium=0.02, nav_floor_fraction=0.01,
                             long_leg_is_fresh_entry=False)
    assert (path["nav"] > 0).all()  # strictly positive, not just non-negative


def test_ratio_spread_nav_path_nav0_equals_collateral_minus_entry_cost():
    price, iv, r = _flat_frame(25)
    pos = op.build_put_ratio_spread(100.0, 0.20, 21, long_delta=0.30, short_delta=0.15,
                                     short_multiple=2, r=0.04, q=0.0)
    path = op.ratio_spread_nav_path(pos["K_long"], pos["K_short"], pos["short_multiple"],
                                     pos["entry_credit"], 0, price, iv, r, q=0.0, dte=21,
                                     profit_target_pct=0.5, cost_pct_of_premium=0.02)
    collateral = 2 * pos["K_short"]
    # entry_credit is NEGATIVE at these catalog deltas (long_delta=0.30,
    # short_delta=0.15, short_multiple=2, dte=21) -- a real property of a
    # 1x2 put ratio spread here, not an edge case. entry_credit_net uses
    # the abs()-based _cost_adjusted formula so cost still makes a DEBIT
    # bigger, not smaller (see that helper's own docstring).
    assert pos["entry_credit"] < 0
    entry_credit_net = pos["entry_credit"] - 0.02 * abs(pos["entry_credit"])
    expected_nav0 = collateral + entry_credit_net - pos["entry_credit"]
    assert path["nav"].iloc[0] == pytest.approx(expected_nav0)
    assert path["nav"].iloc[0] > 0


def test_ratio_spread_nav_path_never_goes_non_positive_through_a_crash_to_zero():
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = np.linspace(100.0, 0.01, n)
    price = pd.Series(prices, index=idx)
    iv = pd.Series(0.60, index=idx)
    r = pd.Series(0.04, index=idx)
    pos = op.build_put_ratio_spread(100.0, 0.20, 21, long_delta=0.30, short_delta=0.15,
                                     short_multiple=2, r=0.04, q=0.0)
    path = op.ratio_spread_nav_path(pos["K_long"], pos["K_short"], pos["short_multiple"],
                                     pos["entry_credit"], 0, price, iv, r, q=0.0, dte=21,
                                     profit_target_pct=0.5, cost_pct_of_premium=0.02)
    assert (path["nav"] > 0).all()
    # The real floor is K_long + entry_credit_net (see ratio_spread_nav_path's
    # own docstring), not entry_credit_net alone -- assert the crash
    # bottoms out near that bound.
    entry_credit_net = pos["entry_credit"] - 0.02 * abs(pos["entry_credit"])
    assert path["nav"].iloc[-1] == pytest.approx(pos["K_long"] + entry_credit_net, rel=0.02)


def test_ratio_spread_nav_path_cost_always_hurts_a_net_debit_position():
    # Guards the cost sign on a net debit: the naive formula
    # (entry_credit_gross * (1 - cost_pct)) would compute cost BACKWARDS
    # whenever entry_credit_gross is negative (a net debit) -- applying a
    # "cost" would make the debit SMALLER. Assert the formula in use
    # always makes a debit position's entry MORE expensive as
    # cost_pct_of_premium increases.
    price, iv, r = _flat_frame(5)
    pos = op.build_put_ratio_spread(100.0, 0.20, 21, long_delta=0.30, short_delta=0.15,
                                     short_multiple=2, r=0.04, q=0.0)
    assert pos["entry_credit"] < 0
    path_low_cost = op.ratio_spread_nav_path(
        pos["K_long"], pos["K_short"], pos["short_multiple"], pos["entry_credit"],
        0, price, iv, r, q=0.0, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.01)
    path_high_cost = op.ratio_spread_nav_path(
        pos["K_long"], pos["K_short"], pos["short_multiple"], pos["entry_credit"],
        0, price, iv, r, q=0.0, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.05)
    assert path_high_cost["nav"].iloc[0] < path_low_cost["nav"].iloc[0]


def test_ratio_spread_nav_path_close_signal_never_fires_on_entry_day():
    # Guards against a profit-target threshold that fires on day 0: the
    # naive formula (entry_credit_gross * (1 - profit_target_pct)) is
    # already satisfied on day 0 whenever entry_credit_gross is negative
    # (as at these catalog deltas), which would make every cycle close
    # (and immediately re-enter) after a single day, breaking the entire
    # monthly-cycle premise.
    price, iv, r = _flat_frame(25)
    pos = op.build_put_ratio_spread(100.0, 0.20, 21, long_delta=0.30, short_delta=0.15,
                                     short_multiple=2, r=0.04, q=0.0)
    assert pos["entry_credit"] < 0
    path = op.ratio_spread_nav_path(pos["K_long"], pos["K_short"], pos["short_multiple"],
                                     pos["entry_credit"], 0, price, iv, r, q=0.0, dte=21,
                                     profit_target_pct=0.5, cost_pct_of_premium=0.02)
    assert not bool(path["close_signal"].iloc[0])


def test_short_put_nav_path_truncates_at_end_of_available_data():
    price, iv, r = _flat_frame(10)  # only 10 sessions -- shorter than dte=21
    pos = op.build_short_put(100.0, 0.20, 21, target_delta=0.15, r=0.04, q=0.0)
    path = op.short_put_nav_path(pos["K"], pos["entry_credit"], 0, price, iv, r,
                                  q=0.0, dte=21, profit_target_pct=0.5,
                                  cost_pct_of_premium=0.02)
    assert len(path) == 10  # truncated, not padded or raised
