"""Active on-leverage as a swept dimension (blend-proxy).

The gate strategy's risk-on rung leverage is a parameter. These tests pin its
behavior: an `on_leverage` L_on in {1.5,2.0,2.5} makes the risk-on leg the
daily-rebalanced 1x/3x blend `static_leverage_lr(...,L_on)`, with cost priced
at L_on, cache-namespaced by L_on, and `on_leverage=3.0` bit-identical to the
default 3x path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from objective import (evaluate_point, static_leverage_lr,
                       annualized_log_wealth, terminal_log_wealth)
from gate_engine import strategy_log_returns
from _synth import two_regime_returns


# --- 1. cost is priced at the passed leverage array ------------------------

def test_strategy_log_returns_respects_explicit_leverage_arg():
    z = np.zeros(4)
    expo = np.array([0, 1, 2, 0])              # decisions; pos lags by one bar
    # default module LEVERAGE=[0,1,3]: the 1x->3x hop at the last bar is |3-1|=2
    out3 = strategy_log_returns(expo, z, z, z, switch_cost_bps=4.0)
    assert np.allclose(out3, -np.array([0, 2, 2, 4]) / 1e4)
    # leverage=[0,1,2]: the same hop is now a 1x->2x hop, |2-1|=1 -> half cost
    out2 = strategy_log_returns(expo, z, z, z, switch_cost_bps=4.0,
                                leverage=[0.0, 1.0, 2.0])
    assert np.allclose(out2, -np.array([0, 2, 2, 2]) / 1e4)


def test_strategy_log_returns_leverage_none_is_module_default():
    rng = np.random.default_rng(5)
    expo = rng.integers(0, 3, (3, 12))
    lc = np.full((3, 12), 1e-5); l1 = rng.normal(0, 0.01, (3, 12)); l3 = 3 * l1
    a = strategy_log_returns(expo, lc, l1, l3, 4.0)
    b = strategy_log_returns(expo, lc, l1, l3, 4.0, leverage=[0.0, 1.0, 3.0])
    assert np.array_equal(a, b)          # explicit [0,1,3] == module default


# --- 2. evaluate_point: on_leverage wires the blend leg + reprices ----------

def _bull_frame(T, seed=3):
    """Always-bull p/ratio over a two-regime (has a real bear) return path, so
    the gate sits on the risk-on rung from bar 1 and leverage bites on the DD."""
    lr1 = two_regime_returns(n_bull=int(T * 0.5), n_bear=int(T * 0.2),
                             n_bull2=T - int(T * 0.5) - int(T * 0.2), seed=seed)
    T = lr1.size
    lr3 = 3.0 * lr1
    lrc = np.full(T, 1e-5)
    p = np.full(T, 0.99)
    ratio = np.full(T, 1.5)
    return (p, ratio, lr1, lr3, lrc), lr1, lr3, lrc, T


def test_evaluate_point_on_leverage_wires_blend_into_realized():
    arrays, lr1, lr3, lrc, T = _bull_frame(300)
    idx = np.arange(T)[None, :]
    out = evaluate_point(arrays, idx, enter_th=0.5, exit_th=0.4, min_dwell=0,
                         overlay=True, switch_cost_bps=0.0, cash_th=-np.inf,
                         on_leverage=2.0)
    on_leg = static_leverage_lr(lrc, lr1, lr3, 2.0)
    hand = np.concatenate([[lr1[0]], on_leg[1:]])       # bar0 = 1x, then blend
    expect = float(annualized_log_wealth(terminal_log_wealth(hand), T))
    assert out["realized_lw_ann"] == pytest.approx(expect, abs=1e-12)


def test_evaluate_point_lower_leverage_is_shallower_excess_dd():
    arrays, *_ , T = _bull_frame(300)
    idx = np.arange(T)[None, :]
    kw = dict(enter_th=0.5, exit_th=0.4, min_dwell=0, overlay=True,
              switch_cost_bps=4.0, cash_th=-np.inf)
    out2 = evaluate_point(arrays, idx, on_leverage=2.0, **kw)
    out3 = evaluate_point(arrays, idx, on_leverage=3.0, **kw)
    # excess-DD is negative; 2x adds strictly less drawdown over 1x than 3x
    assert out2["cvar5_excess_dd"] > out3["cvar5_excess_dd"]


def test_evaluate_point_on_leverage_3_is_default_path():
    arrays, *_, T = _bull_frame(200)
    idx = np.arange(T)[None, :]
    kw = dict(enter_th=0.5, exit_th=0.4, min_dwell=0, overlay=True,
              switch_cost_bps=4.0, cash_th=-np.inf)
    default = evaluate_point(arrays, idx, **kw)
    explicit = evaluate_point(arrays, idx, on_leverage=3.0, **kw)
    assert default == explicit           # 3.0 is the untouched default path


# --- 3. cache key is namespaced by on_leverage (no false hits) --------------

def _tiny_masked(n=24):
    df = pd.DataFrame({
        "Date": pd.bdate_range("2010-01-01", periods=n).date,
        "p_bull": np.linspace(0.1, 0.9, n),
        "trail_ratio": np.linspace(0.8, 1.2, n),
        "lr_1x": np.full(n, 0.001),
        "lr_3x": np.full(n, 0.003),
        "lr_cash": np.full(n, 1e-5)})
    return {21: df, 63: df}


def test_sweep_fingerprint_namespaces_on_leverage():
    from sweep_cache import sweep_fingerprint
    m = _tiny_masked()
    fps = {L: sweep_fingerprint(m, 2000, 63, 0, "2014-12-31", on_leverage=L)
           for L in (1.5, 2.0, 2.5, 3.0)}
    assert len(set(fps.values())) == 4   # all distinct -> no cross-L_on collision


def test_sweep_spec_has_on_leverage_default_3():
    from sweep_cache import SweepSpec
    spec = SweepSpec("QQQ", 63, 0, 2000, "2014-12-31", 15, 63, 1000,
                     "data/raw", "data/backtest")
    assert spec.on_leverage == 3.0      # trailing default: constructors that omit it stay valid


# --- 4. CLI exposes --on-leverage with a validated range --------------------

def test_cli_accepts_on_leverage():
    from strategy_sweep import _build_arg_parser
    args = _build_arg_parser().parse_args(["--on-leverage", "2.0"])
    assert args.on_leverage == 2.0


def test_cli_rejects_out_of_range_on_leverage():
    from strategy_sweep import _build_arg_parser
    # domain is (1, 3]: reject >3, <=1 (rung 2 must out-lever the 1x rung), and junk
    for bad in ("3.5", "1.0", "0.5", "0", "-1"):
        with pytest.raises(SystemExit):
            _build_arg_parser().parse_args(["--on-leverage", bad])


# --- 5. blend leg is NET of daily-rebalance drag (active == passive economics)

def test_static_leverage_lr_net_charges_rebalance_drag():
    from objective import static_leverage_lr, static_leverage_lr_net
    rng = np.random.default_rng(9)
    lr1 = rng.normal(0.0, 0.012, 250); lr3 = 3.0 * lr1; lc = np.full(250, 1e-5)
    gross = static_leverage_lr(lc, lr1, lr3, 2.0)
    net = static_leverage_lr_net(lc, lr1, lr3, 2.0, switch_cost_bps=4.0)
    assert np.all(net <= gross)                     # cost >= 0 every bar
    assert net.sum() < gross.sum()                  # real cumulative drag at L=2
    # endpoints + zero-cost are gross bit-exact -> on_leverage=3.0 stays identical
    assert np.array_equal(static_leverage_lr_net(lc, lr1, lr3, 3.0, 4.0),
                          static_leverage_lr(lc, lr1, lr3, 3.0))
    assert np.array_equal(static_leverage_lr_net(lc, lr1, lr3, 2.0, 0.0),
                          static_leverage_lr(lc, lr1, lr3, 2.0))


# --- 6. cache ABI hashes the blend/cost helpers (no stale hit after an edit) -

def test_abi_digest_includes_blend_cost_helpers(monkeypatch):
    import sweep_cache as sc
    import objective as ob
    sc._ABI_MEMO.clear()
    base = sc.compute_abi_digest()
    orig = ob.static_leverage_lr_net

    def patched(lr_cash, lr_1x, lr_3x, L, switch_cost_bps):   # different source
        return orig(lr_cash, lr_1x, lr_3x, L, switch_cost_bps)
    monkeypatch.setattr(ob, "static_leverage_lr_net", patched)
    sc._ABI_MEMO.clear()
    changed = sc.compute_abi_digest()
    sc._ABI_MEMO.clear()
    assert changed != base            # a blend-helper edit flips the digest


# --- 7. cost_sensitivity re-prices at on_leverage (not silently 3x) ----------

def _mini_joint(n=300, seed=1):
    rng = np.random.default_rng(seed)
    lr1 = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame({
        "Date": pd.bdate_range("2000-01-03", periods=n).date,
        "p_bull": np.clip(rng.random(n), 0.0, 1.0),
        "trail_ratio": 1.0 + rng.normal(0.0, 0.2, n),
        "lr_1x": lr1, "lr_3x": 3.0 * lr1, "lr_cash": np.full(n, 1e-5)})


def test_cost_sensitivity_threads_on_leverage():
    from strategy_sweep import cost_sensitivity
    from gate_engine import bootstrap_indices
    df = _mini_joint()
    jbrbt = {"QQQ": {21: df}}
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
               refit_every=21, cash_arm="never")
    idx = {"QQQ": bootstrap_indices(len(df), 200, block_len=63, seed=0)}
    v3 = cost_sensitivity(jbrbt, {"QQQ": cfg}, idx,
                          on_leverage=3.0)["expected_lw_ann_risk_notional"].iloc[0]
    v2 = cost_sensitivity(jbrbt, {"QQQ": cfg}, idx,
                          on_leverage=2.0)["expected_lw_ann_risk_notional"].iloc[0]
    assert v2 != v3                   # on_leverage actually reaches the re-pricing


# --- 8. full main() pipeline at L_on != 3.0 (subdir + reference label) -------

def _toy_joint_local(T=300, seed=0, start="2014-06-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=T)
    lr1 = rng.normal(0.0004, 0.01, T)
    return pd.DataFrame({"Date": dates.date,
                         "p_bull": np.clip(rng.uniform(0, 1, T), 0, 1),
                         "trail_ratio": rng.uniform(0.8, 1.4, T),
                         "lr_1x": lr1, "lr_3x": 3 * lr1 - 1e-4,
                         "lr_cash": np.full(T, np.log1p(0.02 / 252))})


def test_main_at_active_leverage_subdir_and_reference_label(monkeypatch, tmp_path):
    import strategy_sweep as ss
    import causal_backtest as cb
    tk_joints = {tk: _toy_joint_local(seed=i + 1)
                 for i, tk in enumerate(ss.TICKERS)}

    def fake_build(ticker, refit_every, data_dir, cache_dir, vol_window=63,
                   warmup=1000, n_restarts=15, force=False, strict=False):
        return tk_joints[ticker].copy()
    monkeypatch.setattr(ss, "build_joint_frame", fake_build)
    monkeypatch.setattr(cb, "build_joint_frame", fake_build)
    monkeypatch.setattr(ss, "grid_points",
                        lambda: [dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
                                      overlay=True, refit_every=21, cash_arm=arm)
                                 for arm in ("never", "gap_10", "gap_20")])
    monkeypatch.setattr(ss, "plot_heatmaps", lambda *a, **k: [])     # skip mpl

    out_dir = str(tmp_path / "out")
    result = ss.main(["--on-leverage", "2.0", "--excess-floor", "-0.40",
                      "--quick", "--workers", "1",
                      "--data-dir", str(tmp_path / "raw"),
                      "--cache-dir", str(tmp_path / "cache"),
                      "--out-dir", out_dir])

    # (a) every artifact lands under the L2 subtree, never the bare out_dir
    assert result["report_fp"] == os.path.join(out_dir, "L2",
                                               "leveraged_gate_sweep_report.md")
    assert os.path.exists(result["report_fp"])
    assert os.path.exists(os.path.join(out_dir, "L2", "gate_sweep_QQQ.csv"))
    assert not os.path.exists(os.path.join(out_dir,
                                           "leveraged_gate_sweep_report.md"))
    # (b) provenance records the active leverage
    assert result["provenance"]["on_leverage"] == 2.0
    # (c) the leveraged holdout reference is named for its ACTUAL leverage
    ids = set(result["holdout"]["config_id"])
    assert "always_2x" in ids and "always_3x" not in ids and "always_1x" in ids


# --- 9. passive ladder DD is net-of-cost (matches the active strategy) -------

def test_static_ladder_drawdown_uses_net_leg():
    from strategy_sweep import static_ladder_table
    from objective import (static_leverage_lr, static_leverage_lr_net,
                           max_drawdown_from_lr, cvar)
    from gate_engine import bootstrap_indices
    df = _mini_joint(n=400, seed=2)
    idx = bootstrap_indices(len(df), 200, block_len=63, seed=0)
    base = max_drawdown_from_lr(df["lr_1x"].values[idx])
    tbl = static_ladder_table(df, idx, base, ladder=(1.0, 2.0, 3.0))
    lc, l1, l3 = df["lr_cash"].values, df["lr_1x"].values, df["lr_3x"].values
    net2 = static_leverage_lr_net(lc, l1, l3, 2.0, 4.0)[idx]
    gross2 = static_leverage_lr(lc, l1, l3, 2.0)[idx]
    exp_net = cvar(np.atleast_1d(max_drawdown_from_lr(net2)), 0.05)
    exp_gross = cvar(np.atleast_1d(max_drawdown_from_lr(gross2)), 0.05)
    got = float(tbl[tbl["L"] == 2.0].iloc[0]["cvar5_maxdd"])
    assert got == pytest.approx(exp_net, abs=1e-12)        # ladder DD is NET
    assert exp_net != pytest.approx(exp_gross, abs=1e-12)  # net truly differs at L=2
