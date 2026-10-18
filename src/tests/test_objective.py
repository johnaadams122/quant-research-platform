import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


def test_terminal_and_annualized_log_wealth():
    from objective import terminal_log_wealth, annualized_log_wealth
    lr = np.array([[0.01, -0.02, 0.03], [0.0, 0.0, 0.0]])
    lw = terminal_log_wealth(lr)
    assert np.allclose(lw, [0.02, 0.0])
    assert annualized_log_wealth(np.array(0.02), 252) == pytest.approx(0.02)
    assert annualized_log_wealth(np.array(0.02), 504) == pytest.approx(0.01)


def test_max_drawdown_matches_metrics_module():
    from objective import max_drawdown_from_lr
    from metrics import max_drawdown
    rng = np.random.default_rng(4)
    lr = rng.normal(0, 0.02, 300)
    prices = pd.Series(100 * np.exp(np.cumsum(lr)),
                       index=pd.bdate_range("2020-01-01", periods=300))
    # metrics.max_drawdown peaks at the first PRICE; seed a 1.0 peak by
    # prepending a zero return so both see the same curve
    got = max_drawdown_from_lr(np.concatenate([[0.0], lr]))
    assert got == pytest.approx(
        max_drawdown(pd.concat([pd.Series([100.0]), prices],
                               ignore_index=True)), abs=1e-12)


def test_cvar_hand_example():
    from objective import cvar
    v = np.array([-0.5, -0.4, -0.3, -0.2, -0.1] + [0.0] * 95)
    assert cvar(v, alpha=0.05) == pytest.approx((-0.5 - 0.4 - 0.3 - 0.2 - 0.1) / 5)
    assert cvar(np.array([3.0]), alpha=0.05) == 3.0


def test_evaluate_point_returns_all_keys_and_respects_gate():
    from objective import evaluate_point
    from gate_engine import bootstrap_indices
    T = 400
    rng = np.random.default_rng(1)
    p = np.clip(rng.uniform(0, 1, T), 0, 1)
    ratio = np.ones(T) * 1.5
    lr1 = rng.normal(0.0004, 0.01, T)
    lr3 = 3 * lr1 - 0.0001
    lc = np.full(T, 1e-5)
    idx = bootstrap_indices(T, 100, block_len=40, seed=0)
    out = evaluate_point((p, ratio, lr1, lr3, lc), idx, 0.6, 0.4, 0, True)
    for k in ("expected_lw_ann", "cvar5_maxdd", "cvar5_lw_ann",
              "median_terminal_ratio_vs_1x", "realized_lw_ann",
              "realized_maxdd", "realized_calmar", "switches_realized",
              "lev_turnover", "cash_occupancy", "cash_zero_use_freq"):
        assert k in out and np.isfinite(out[k])
    # enter_th=1.1 can never arm -> strategy IS 1x -> ratio vs 1x == 1
    never = evaluate_point((p, ratio, lr1, lr3, lc), idx, 1.1, 0.4, 0, True)
    assert never["median_terminal_ratio_vs_1x"] == pytest.approx(1.0)
    assert never["switches_realized"] == 0


def test_switches_exclude_untradeable_final_bar_flip():
    from objective import evaluate_point
    from gate_engine import bootstrap_indices
    T = 4
    p = np.array([0.0, 0.0, 0.0, 1.0])     # arms only on the last bar
    ratio = np.ones(T) * 1.5
    lr1 = np.full(T, 0.01); lr3 = np.full(T, 0.02)
    lc = np.full(T, 1e-5)
    idx = bootstrap_indices(T, 10, block_len=2, seed=0)
    out = evaluate_point((p, ratio, lr1, lr3, lc), idx, 0.6, 0.4, 0, True)
    assert out["switches_realized"] == 0    # decided, never traded/charged


def _ep_fixture(T=400, seed=1, n_boot=100):
    from gate_engine import bootstrap_indices
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(0, 1, T), 0, 1)
    ratio = np.ones(T) * 1.5
    lr1 = rng.normal(0.0004, 0.01, T)
    lr3 = 3 * lr1 - 0.0001
    lc = np.full(T, 1e-5)
    idx = bootstrap_indices(T, n_boot, block_len=40, seed=0)
    return (p, ratio, lr1, lr3, lc), idx


def test_cvar5_excess_dd_is_paired_per_path_not_diff_of_cvars():
    # Pins the WIRING: per-path pairing on the SAME idx rows, then CVaR of
    # the differences -- and rejects the central wrong implementation,
    # CVaR(strategy) - CVaR(1x).
    from objective import evaluate_point, max_drawdown_from_lr, cvar
    from gate_engine import exposure_states, strategy_log_returns
    arrays, idx = _ep_fixture()
    p, ratio, lr1, lr3, lc = arrays
    out = evaluate_point(arrays, idx, 0.6, 0.4, 0, True)
    expo_b = exposure_states(p[idx], ratio[idx], 0.6, 0.4, 0, True)
    lr_b = strategy_log_returns(expo_b, lc[idx], lr1[idx], lr3[idx], 4.0)
    strat_dd = max_drawdown_from_lr(lr_b)
    base_dd = max_drawdown_from_lr(lr1[idx])
    expected = cvar(strat_dd - base_dd, 0.05)
    assert out["cvar5_excess_dd"] == pytest.approx(expected, abs=1e-12)
    diff_of_cvars = cvar(strat_dd, 0.05) - cvar(base_dd, 0.05)
    assert abs(out["cvar5_excess_dd"] - diff_of_cvars) > 1e-6


def test_never_armed_gate_has_exactly_zero_excess():
    # enter_th=1.1 can never arm -> strategy IS 1x on every path -> the
    # paired excess is identically zero (spec: pinned exactly, no approx).
    from objective import evaluate_point
    arrays, idx = _ep_fixture()
    out = evaluate_point(arrays, idx, 1.1, 0.4, 0, True)
    assert out["cvar5_excess_dd"] == 0.0
    assert out["excess_tail_worst"] == 0.0


def test_baseline_maxdd_b_precomputed_equals_inline_and_is_keyword_only():
    from objective import evaluate_point, max_drawdown_from_lr
    arrays, idx = _ep_fixture()
    lr1 = arrays[2]
    base = max_drawdown_from_lr(lr1[idx])
    inline = evaluate_point(arrays, idx, 0.6, 0.4, 0, True)
    pre = evaluate_point(arrays, idx, 0.6, 0.4, 0, True,
                         baseline_maxdd_b=base)
    assert pre["cvar5_excess_dd"] == inline["cvar5_excess_dd"]
    # keyword-only: passing it positionally must be a TypeError, so the
    # existing positional binding through switch_cost_bps is unchanged.
    with pytest.raises(TypeError):
        evaluate_point(arrays, idx, 0.6, 0.4, 0, True, 4.0, base)


def test_baseline_maxdd_b_feeds_through_with_known_effects():
    # A modified baseline changes the result by a KNOWN
    # amount (not just !=), and the pairing is row-wise.
    from objective import evaluate_point, max_drawdown_from_lr, cvar
    from gate_engine import exposure_states, strategy_log_returns
    arrays, idx = _ep_fixture()
    p, ratio, lr1, lr3, lc = arrays
    base = max_drawdown_from_lr(lr1[idx]).copy()
    ok = evaluate_point(arrays, idx, 0.6, 0.4, 0, True,
                        baseline_maxdd_b=base)
    # (a) uniform deepening by 0.10: every excess rises by exactly +0.10,
    # so the CVaR rises by exactly +0.10 -- pins the arithmetic.
    sh = evaluate_point(arrays, idx, 0.6, 0.4, 0, True,
                        baseline_maxdd_b=base - 0.10)
    assert sh["cvar5_excess_dd"] == pytest.approx(
        ok["cvar5_excess_dd"] + 0.10, abs=1e-12)
    # (b) zero out ONE row's baseline: recompute the expected CVaR from
    # the same primitives POSITIONALLY -- proves row-wise pairing.
    expo_b = exposure_states(p[idx], ratio[idx], 0.6, 0.4, 0, True)
    strat_dd = max_drawdown_from_lr(
        strategy_log_returns(expo_b, lc[idx], lr1[idx], lr3[idx], 4.0))
    one = base.copy(); one[7] = 0.0
    sw = evaluate_point(arrays, idx, 0.6, 0.4, 0, True,
                        baseline_maxdd_b=one)
    assert sw["cvar5_excess_dd"] == pytest.approx(
        cvar(strat_dd - one, 0.05), abs=1e-12)
    assert sw["cvar5_excess_dd"] != ok["cvar5_excess_dd"]
    with pytest.raises(ValueError):
        evaluate_point(arrays, idx, 0.6, 0.4, 0, True,
                       baseline_maxdd_b=base[:-1])          # wrong length
    bad = base.copy(); bad[0] = np.nan
    with pytest.raises(ValueError):
        evaluate_point(arrays, idx, 0.6, 0.4, 0, True,
                       baseline_maxdd_b=bad)                 # non-finite


def test_excess_tail_composition_keys():
    from objective import evaluate_point, max_drawdown_from_lr, cvar
    from gate_engine import exposure_states, strategy_log_returns
    import math
    arrays, idx = _ep_fixture()
    p, ratio, lr1, lr3, lc = arrays
    out = evaluate_point(arrays, idx, 0.6, 0.4, 0, True)
    expo_b = exposure_states(p[idx], ratio[idx], 0.6, 0.4, 0, True)
    lr_b = strategy_log_returns(expo_b, lc[idx], lr1[idx], lr3[idx], 4.0)
    strat_dd = max_drawdown_from_lr(lr_b)
    base_dd = max_drawdown_from_lr(lr1[idx])
    excess = strat_dd - base_dd
    k = max(1, int(math.ceil(0.05 * excess.size)))
    tail = np.argsort(excess, kind="stable")[:k]
    assert out["excess_tail_k"] == k == 5                    # n_boot=100
    assert out["excess_tail_worst"] == pytest.approx(excess[tail].min())
    assert out["cvar5_excess_dd"] == pytest.approx(float(excess[tail].mean()))
    assert out["excess_tail_median_maxdd_strategy"] == pytest.approx(
        float(np.median(strat_dd[tail])))
    assert out["excess_tail_worst_maxdd_strategy"] == pytest.approx(
        float(strat_dd[tail].min()))
    assert out["excess_tail_median_maxdd_1x"] == pytest.approx(
        float(np.median(base_dd[tail])))
    assert out["excess_tail_worst_maxdd_1x"] == pytest.approx(
        float(base_dd[tail].min()))


def test_always_armed_3x_on_crash_fixture_is_infeasible_at_default_floor():
    # Behavioral corner case, with EXPLICIT chronological idx
    # rows (not bootstrap draws): a gain-then-crash path guarantees the
    # excess -- crash-only reshuffles would compress it (both legs saturate
    # near -1). Hand-derivation: 1x dd = 1-exp(-1.2) ~= 0.699; 3x dd =
    # 1-exp(-3.6) ~= 0.973; excess ~= -0.274 < -0.15.
    from objective import evaluate_point
    T = 120
    lr1 = np.concatenate([np.full(60, 0.004), np.full(60, -0.02)])
    lr3 = 3 * lr1 - 0.0001
    lc = np.full(T, 1e-5)
    p = np.ones(T)                       # always-bull posterior
    ratio = np.ones(T) * 1.5
    idx = np.tile(np.arange(T), (3, 1))  # 3 identical chronological paths
    out = evaluate_point((p, ratio, lr1, lr3, lc), idx, -1.0, -2.0, 0, False)
    dd1 = 1 - np.exp(-1.2)
    dd3 = 1 - np.exp(-3.6)
    assert out["cvar5_excess_dd"] == pytest.approx(-(dd3 - dd1), abs=2e-3)
    assert out["cvar5_excess_dd"] < -0.15


def test_excess_tail_k_pins_n4_and_n2000():
    # k-contract pins on the actual fixture: n=4 -> k=1,
    # n=2000 -> k=100.
    from objective import evaluate_point
    T = 120
    lr1 = np.concatenate([np.full(60, 0.004), np.full(60, -0.02)])
    lr3 = 3 * lr1 - 0.0001
    lc = np.full(T, 1e-5)
    p = np.ones(T); ratio = np.ones(T) * 1.5
    idx4 = np.tile(np.arange(T), (4, 1))
    out4 = evaluate_point((p, ratio, lr1, lr3, lc), idx4, -1.0, -2.0, 0, False)
    assert out4["excess_tail_k"] == 1
    rng = np.random.default_rng(0)
    idx2k = rng.integers(0, T, size=(2000, T))
    out2k = evaluate_point((p, ratio, lr1, lr3, lc), idx2k, -1.0, -2.0, 0, False)
    assert out2k["excess_tail_k"] == 100


def test_cvar_k_contract_k2_averaging():
    # k = max(1, ceil(0.05*n)): n=40 -> k=2 -> mean of the two worst.
    from objective import cvar
    v = np.linspace(-0.40, -0.01, 40)
    assert cvar(v, alpha=0.05) == pytest.approx((v[0] + v[1]) / 2)


def test_golden_existing_keys_unchanged_by_excess_code():
    # Objective preservation: the pre-existing 8 keys still equal their
    # original formulas (recomputed here from the same primitives).
    from objective import (evaluate_point, terminal_log_wealth,
                           annualized_log_wealth, max_drawdown_from_lr, cvar)
    from gate_engine import exposure_states, strategy_log_returns
    arrays, idx = _ep_fixture()
    p, ratio, lr1, lr3, lc = arrays
    T = p.size
    out = evaluate_point(arrays, idx, 0.6, 0.4, 0, True)
    expo_b = exposure_states(p[idx], ratio[idx], 0.6, 0.4, 0, True)
    lr_b = strategy_log_returns(expo_b, lc[idx], lr1[idx], lr3[idx], 4.0)
    lw_b = terminal_log_wealth(lr_b)
    assert out["expected_lw_ann"] == pytest.approx(
        float(annualized_log_wealth(lw_b, T).mean()), abs=1e-12)
    assert out["cvar5_maxdd"] == pytest.approx(
        cvar(max_drawdown_from_lr(lr_b), 0.05), abs=1e-12)
    assert out["cvar5_lw_ann"] == pytest.approx(
        cvar(annualized_log_wealth(lw_b, T), 0.05), abs=1e-12)
    assert out["median_terminal_ratio_vs_1x"] == pytest.approx(
        float(np.median(np.exp(lw_b - terminal_log_wealth(lr1[idx])))),
        abs=1e-12)
    # realized single-path keys, same formulas as the original module
    expo = exposure_states(p, ratio, 0.6, 0.4, 0, True)
    lr_s = strategy_log_returns(expo, lc, lr1, lr3, 4.0)
    r_lw = float(annualized_log_wealth(terminal_log_wealth(lr_s), T))
    r_dd = float(max_drawdown_from_lr(lr_s))
    assert out["realized_lw_ann"] == pytest.approx(r_lw, abs=1e-12)
    assert out["realized_maxdd"] == pytest.approx(r_dd, abs=1e-12)
    assert out["realized_calmar"] == pytest.approx(
        np.expm1(r_lw) / abs(r_dd), abs=1e-12)
    # switches: count transitions in the EXECUTED posture (posture at bar t
    # is the exposure decided at t-1; start posture is 1x/level-1, matching
    # gate_engine.strategy_log_returns' own pos convention -- the 3-state
    # engine's cash-capable level space starts posture at 1, not 0).
    pos = np.concatenate([[1], expo.astype(int)[:-1]])
    assert out["switches_realized"] == int((pos[1:] != pos[:-1]).sum())


def test_baseline_contract_with_1d_idx_single_path():
    # A 1-D idx is ONE path of length T: a (1,)-shaped baseline must be
    # accepted (and equal the inline result); a (T,)-shaped baseline must
    # raise, never broadcast into an unpaired excess vector.
    from objective import evaluate_point, max_drawdown_from_lr
    arrays, _ = _ep_fixture()
    p = arrays[0]
    idx1d = np.arange(p.size)            # one chronological path
    lr1 = arrays[2]
    good = np.atleast_1d(max_drawdown_from_lr(lr1[idx1d]))
    assert good.shape == (1,)
    inline = evaluate_point(arrays, idx1d, 0.6, 0.4, 0, True)
    pre = evaluate_point(arrays, idx1d, 0.6, 0.4, 0, True,
                         baseline_maxdd_b=good)
    assert pre["cvar5_excess_dd"] == inline["cvar5_excess_dd"]
    assert pre["excess_tail_k"] == 1
    with pytest.raises(ValueError):
        evaluate_point(arrays, idx1d, 0.6, 0.4, 0, True,
                       baseline_maxdd_b=np.full(p.size, -0.1))


def test_legacy_fields_bit_for_bit_when_never_cash():
    # Nesting: with cash_th=-inf the new 3-state engine must never
    # touch cash, and every LEGACY field must equal a computation built from
    # the FROZEN Stage-1 2-state reference (_stage1_reference.py) on the
    # SAME idx -- this is the hard bit-for-bit contract, not weakened.
    from objective import (evaluate_point, annualized_log_wealth,
                           terminal_log_wealth, max_drawdown_from_lr, cvar)
    from _stage1_reference import ref_exposure_states, ref_strategy_log_returns
    rng = np.random.default_rng(21)
    n = 200
    p = rng.random(n); r = rng.random(n) + 0.5
    l1 = rng.normal(0, 0.01, n); l3 = 3 * l1; lc = np.full(n, 1e-5)
    idx = rng.integers(0, n, size=(50, n))
    out = evaluate_point((p, r, l1, l3, lc), idx, 0.6, 0.4, 0, True,
                         cash_th=-np.inf)
    # Reference: build EVERY legacy field from the frozen 2-state functions on
    # the SAME idx, and assert the new engine reproduces the full set.
    old_expo = ref_exposure_states(p, r, 0.6, 0.4, 0, True)
    old_lr = ref_strategy_log_returns(old_expo, l1, l3)
    exp_lw = float(annualized_log_wealth(terminal_log_wealth(old_lr), n))
    exp_maxdd = float(max_drawdown_from_lr(old_lr))
    old_expo_b = ref_exposure_states(p[idx], r[idx], 0.6, 0.4, 0, True)
    old_lr_b = ref_strategy_log_returns(old_expo_b, l1[idx], l3[idx])
    lw_b = terminal_log_wealth(old_lr_b)
    assert abs(out["realized_lw_ann"] - exp_lw) < 1e-12
    assert abs(out["realized_maxdd"] - exp_maxdd) < 1e-12
    assert abs(out["expected_lw_ann"]
               - float(annualized_log_wealth(lw_b, n).mean())) < 1e-12
    assert abs(out["cvar5_maxdd"]
               - cvar(np.atleast_1d(max_drawdown_from_lr(old_lr_b)), 0.05)) < 1e-12
    assert abs(out["cvar5_lw_ann"]
               - cvar(annualized_log_wealth(lw_b, n), 0.05)) < 1e-12
    # switches_realized identical to the 2-state executed-switch count
    old_pos = np.concatenate([[0], old_expo.astype(int)[:-1]])
    exp_switches = int((np.abs(np.diff(np.concatenate([[0], old_pos]))) > 0).sum())
    assert out["switches_realized"] == exp_switches
    assert out["cash_occupancy"] == 0.0
    assert out["cash_zero_use_freq"] == 1.0


def test_static_endpoints_bit_exact():
    from objective import static_leverage_lr
    lc = np.array([1e-4, 1e-4]); l1 = np.array([0.01, -0.02]); l3 = 3 * l1
    assert np.array_equal(static_leverage_lr(lc, l1, l3, 0.0), lc)
    assert np.array_equal(static_leverage_lr(lc, l1, l3, 1.0), l1)
    assert np.array_equal(static_leverage_lr(lc, l1, l3, 3.0), l3)


def test_static_L2_finite_and_L1_zero_excess():
    from objective import static_leverage_lr, max_drawdown_from_lr
    l1 = np.array([0.01, -0.34, 0.02]); l3 = np.log1p(3 * np.expm1(l1))
    lc = np.full(3, 1e-5)
    assert np.isfinite(static_leverage_lr(lc, l1, l3, 2.0)).all()
    x1 = static_leverage_lr(lc, l1, l3, 1.0)
    assert max_drawdown_from_lr(x1) == max_drawdown_from_lr(l1)


def test_cash_config_positive_excess_and_occupancy():
    # A config that arms into cash (cash_th=0.30) on a bull-then-crash
    # fixture: the strategy sidesteps most of the crash drawdown (positive
    # excess vs the always-1x baseline) and spends real bars in cash.
    from objective import evaluate_point
    n = 60
    p = np.concatenate([np.full(30, 0.9), np.full(30, 0.05)])
    r = np.ones(n)
    l1 = np.concatenate([np.zeros(30), np.full(30, -0.03)]); l3 = 3 * l1
    lc = np.full(n, 1e-5)
    idx = np.arange(n)[None, :]
    out = evaluate_point((p, r, l1, l3, lc), idx, 0.8, 0.5, 0, False, cash_th=0.30)
    assert out["excess_tail_worst"] > -1e-9
    assert out["cash_occupancy"] > 0.0
