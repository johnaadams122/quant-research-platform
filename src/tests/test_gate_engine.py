# src/tests/test_gate_engine.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


def test_min_dwell_blocks_reentry_never_exit():
    from gate_engine import exposure_states
    p =     np.array([0.9, 0.9, 0.3, 0.9, 0.9, 0.9, 0.9])
    ratio = np.ones(7) * 1.5
    e = exposure_states(p, ratio, 0.6, 0.4, min_dwell=3)
    #        on   on   EXIT cooldown(3 bars: t3,t4,t5) re-enter t6
    # 3-state ladder encoding {0: cash, 1: 1x, 2: 3x} (default
    # cash_th=-inf, start=1x): on is 3x (2), off is 1x (1)
    assert e.tolist() == [2, 2, 1, 1, 1, 1, 2]
    # exit is immediate even inside a fresh entry
    p2 = np.array([0.9, 0.3, 0.9])
    e2 = exposure_states(p2, np.ones(3) * 1.5, 0.6, 0.4, min_dwell=5)
    assert e2.tolist() == [2, 1, 1]


def test_overlay_off_ignores_ratio():
    from gate_engine import exposure_states
    p = np.array([0.9, 0.9, 0.9])
    ratio = np.array([np.nan, 0.2, 0.1])
    # 3-state ladder encoding {0: cash, 1: 1x, 2: 3x} (default
    # cash_th=-inf, start=1x): on is 3x (2), off is 1x (1)
    assert exposure_states(p, ratio, 0.6, 0.4, overlay=False).tolist() == [2, 2, 2]
    assert exposure_states(p, ratio, 0.6, 0.4, overlay=True).tolist() == [1, 1, 1]


def test_matrix_input_vectorizes_per_path():
    from gate_engine import exposure_states
    p = np.array([[0.9, 0.3, 0.9], [0.5, 0.9, 0.9]])
    ratio = np.ones((2, 3)) * 1.5
    e = exposure_states(p, ratio, 0.6, 0.4)
    assert e.shape == (2, 3)
    # 3-state ladder encoding {0: cash, 1: 1x, 2: 3x} (default
    # cash_th=-inf, start=1x): on is 3x (2), off is 1x (1)
    assert e[0].tolist() == [2, 1, 2] and e[1].tolist() == [1, 2, 2]


def test_strategy_returns_lag_one_bar_and_charge_switches():
    # 3-leg signature: strategy_log_returns(exposure,
    # lr_cash, lr_1x, lr_3x, switch_cost_bps). The 3-state encoding is
    # {0:cash, 1:1x, 2:3x} with pos starting at level 1 (=1x); this path
    # never visits cash, so lr_cash is unused (zeros). Under the
    # risk-notional cost formula, a 1x<->3x switch at switch_cost_bps=100
    # costs (100/2)*|LEVERAGE[2]-LEVERAGE[1]|/1e4 = (100/2)*2/1e4 = 100/1e4,
    # the same as a flat switch_cost_bps/1e4 charge.
    from gate_engine import strategy_log_returns
    lr_cash = np.zeros(4)
    lr1 = np.array([0.01, 0.02, 0.03, 0.04])
    lr3 = np.array([0.10, 0.20, 0.30, 0.40])
    expo = np.array([2, 2, 1, 1], dtype=np.int8)
    got = strategy_log_returns(expo, lr_cash, lr1, lr3, switch_cost_bps=100.0)
    cost = 100.0 / 1e4
    # bar0: initial 1x, no switch. bar1: exposure[0]=2 -> 3x, entered at
    # close0 => charged. bar2: exposure[1]=2 -> 3x. bar3: exposure[2]=1 ->
    # 1x, exited at close2 => charged.
    expected = [0.01, 0.20 - cost, 0.30, 0.04 - cost]
    assert np.allclose(got, expected)


def test_strategy_returns_matrix_shape():
    # Shape probe including the lr_cash leg (3rd positional arg). expo is
    # all-zero, which under the 3-state encoding means "always cash" -- a
    # valid shape probe that also exercises the cash leg's shape; this test
    # asserts only .shape, never values.
    from gate_engine import strategy_log_returns
    lr_cash = np.zeros((5, 3))
    lr1 = np.tile([0.01, 0.02, 0.03], (5, 1))
    lr3 = np.tile([0.1, 0.2, 0.3], (5, 1))
    expo = np.zeros((5, 3), dtype=np.int8)
    assert strategy_log_returns(expo, lr_cash, lr1, lr3).shape == (5, 3)


def test_bootstrap_indices_blocks_and_determinism():
    from gate_engine import bootstrap_indices
    idx = bootstrap_indices(200, 50, block_len=20, seed=7)
    assert idx.shape == (50, 200)
    assert idx.min() >= 0 and idx.max() < 200
    # inside a block, indices are consecutive mod T
    d = np.diff(idx[0][:20])
    assert np.all((d == 1) | (d == 1 - 200))
    idx2 = bootstrap_indices(200, 50, block_len=20, seed=7)
    assert np.array_equal(idx, idx2)
    assert not np.array_equal(idx, bootstrap_indices(200, 50, 20, seed=8))


def test_output_dtype_is_int8():
    from gate_engine import exposure_states
    got1 = exposure_states(np.array([0.9, 0.3]), np.array([1.5, 1.5]), 0.6, 0.4)
    got2 = exposure_states(np.ones((2, 3)) * 0.9, np.ones((2, 3)) * 1.5, 0.6, 0.4)
    assert got1.dtype == np.int8 and got2.dtype == np.int8


def test_min_dwell_is_per_path_in_matrix_input():
    from gate_engine import exposure_states
    # path 0 exits at t2 -> cooldown blocks t3-t4, re-enters t5;
    # path 1 never exits -> stays on throughout (its clock must not be
    # polluted by path 0's exit)
    p = np.array([[0.9, 0.9, 0.3, 0.9, 0.9, 0.9],
                  [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
    ratio = np.ones((2, 6)) * 1.5
    e = exposure_states(p, ratio, 0.6, 0.4, min_dwell=2)
    # 3-state ladder encoding {0: cash, 1: 1x, 2: 3x} (default
    # cash_th=-inf, start=1x): off is 1x (1), on is 3x (2)
    assert e[0].tolist() == [2, 2, 1, 1, 1, 2]
    assert e[1].tolist() == [2, 2, 2, 2, 2, 2]


def test_ladder_trace_one_rung_and_hysteresis():
    from gate_engine import exposure_states
    p = np.array([0.90, 0.90, 0.40, 0.20, 0.55, 0.90])
    lv = exposure_states(p, np.ones_like(p), enter_th=0.80, exit_th=0.50,
                         min_dwell=0, overlay=False, cash_th=0.30)
    assert lv.tolist() == [2, 2, 1, 0, 1, 2]


def test_cash_sticky_band_holds_1x():
    from gate_engine import exposure_states
    p = np.array([0.45, 0.45, 0.45])
    lv = exposure_states(p, np.ones_like(p), 0.80, 0.50, 0, False, cash_th=0.30)
    assert lv.tolist() == [1, 1, 1]


def test_never_cash_equals_stage1_plus_one_battery():
    from gate_engine import exposure_states
    from _stage1_reference import ref_exposure_states
    rng = np.random.default_rng(11)
    cases = [
        dict(p=rng.random(60), r=rng.random(60) + 0.4, e=0.6, x=0.4, d=0, ov=True),
        dict(p=rng.random(60), r=rng.random(60) + 0.4, e=0.7, x=0.3, d=5, ov=True),
        dict(p=rng.random((5, 40)), r=rng.random((5, 40)) + 0.4, e=0.55, x=0.5,
             d=10, ov=False),
    ]
    # inject NaN into a ratio to exercise the overlay nan_to_num path
    cases[0]["r"][3] = np.nan
    for c in cases:
        new = exposure_states(c["p"], c["r"], c["e"], c["x"], c["d"], c["ov"],
                              cash_th=-np.inf)
        old = ref_exposure_states(c["p"], c["r"], c["e"], c["x"], c["d"], c["ov"])
        assert np.array_equal(new, old + 1)


from gate_engine import strategy_log_returns, LEVERAGE
from _stage1_reference import ref_strategy_log_returns

def test_cost_scales_with_leverage_change():
    z = np.zeros(4)
    expo = np.array([0, 1, 2, 0])              # decisions
    out = strategy_log_returns(expo, z, z, z, switch_cost_bps=4.0)
    assert np.allclose(out, -np.array([0, 2, 2, 4]) / 1e4)

def test_never_cash_returns_nest_stage1_reference():
    rng = np.random.default_rng(13)
    l1 = rng.normal(0, 0.01, (4, 30)); l3 = 3 * l1; lc = np.full((4, 30), 1e-5)
    s1 = ref_strategy_log_returns  # 2-state
    from gate_engine import exposure_states
    from _stage1_reference import ref_exposure_states
    p = rng.random((4, 30)); r = rng.random((4, 30)) + 0.4
    old_expo = ref_exposure_states(p, r, 0.6, 0.4, 0, True)
    new_expo = exposure_states(p, r, 0.6, 0.4, 0, True, cash_th=-np.inf)
    assert np.array_equal(new_expo, old_expo + 1)
    new_ret = strategy_log_returns(new_expo, lc, l1, l3, 4.0)
    old_ret = s1(old_expo, l1, l3, 4.0)
    # BIT-for-bit: leg selection is pure indexing and 1x<->3x cost = 4bps under
    # both formulas, so the values are identical, not merely close.
    assert np.array_equal(new_ret, old_ret)


def test_dollar_turnover_flat_per_rung_and_prices_1x3x_at_one_side():
    z = np.zeros(4)
    expo = np.array([0, 1, 2, 0])   # cash,1x,3x,cash decisions
    out = strategy_log_returns(expo, z, z, z, switch_cost_bps=4.0,
                               cost_model="dollar_turnover")
    # pos=[1,0,1,2]; each executed rung-change = one side = 2bps flat
    assert np.allclose(out, -np.array([0, 2, 2, 2]) / 1e4)
    # 1x->3x alone costs 2bps here vs 4bps under risk_notional. The decision
    # array is e2=[2,2], not [1,2]: strategy_log_returns' pos formula is
    # pos=concat([1], e[:-1]) (position HELD lags the DECISION by one bar,
    # as every other test in this file relies on, e.g.
    # test_cost_scales_with_leverage_change) -- for a 2-element decision
    # array only e[0] ever reaches `pos` (e[1] would only affect a pos[2]
    # that doesn't exist in a 2-bar window), so [1,2] would never
    # transition (pos stays [1,1], zero cost either way). e2=[2,2] produces
    # the "1x->3x alone" transition (pos=[1,2]).
    e2 = np.array([2, 2]); rn = strategy_log_returns(e2, z[:2], z[:2], z[:2], 4.0)
    dt = strategy_log_returns(e2, z[:2], z[:2], z[:2], 4.0,
                              cost_model="dollar_turnover")
    assert np.allclose(rn, -np.array([0, 4]) / 1e4)
    assert np.allclose(dt, -np.array([0, 2]) / 1e4)
