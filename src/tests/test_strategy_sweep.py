# src/tests/test_strategy_sweep.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


# --- selection-floor union + per-(floor,ticker) selection matrix ---

def test_selection_floors_union_cli():
    from strategy_sweep import selection_floors, SELECTION_DIAGNOSTIC_FLOORS
    assert set(selection_floors(-0.30)) == set(SELECTION_DIAGNOSTIC_FLOORS) | {-0.30}


def test_selection_matrix_covers_floors_and_tickers():
    from strategy_sweep import run_selection_matrix
    base = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
                refit_every=21, cash_arm="never")
    tr = {"QQQ": pd.DataFrame([{**base, "expected_lw_ann": 0.2,
                                "cvar5_excess_dd": -0.30}])}
    m = run_selection_matrix(tr, None, floors=[-0.15, -0.40])
    assert m[-0.40]["QQQ"]["viable"] is True
    assert m[-0.15]["QQQ"]["viable"] is False


def test_grid_has_360_points_all_exit_below_enter():
    from strategy_sweep import grid_points
    pts = grid_points()
    # The cash_arm dimension: the "never" arm alone is the full 360-point
    # (enter, exit, dwell, overlay, refit) grid; every point (all arms) has
    # exit < enter.
    assert len([p for p in pts if p["cash_arm"] == "never"]) == 360
    assert all(p["exit_th"] < p["enter_th"] for p in pts)


def _toy_joint(T=300, seed=0, start="2010-01-04"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=T)
    lr1 = rng.normal(0.0004, 0.01, T)
    # lr_cash: small constant daily cash log-return (~2%/yr, IRX-like). Its
    # value is inert for every test here (all use the never arm, cash_th=-inf,
    # so the cash leg is never selected) but the column must exist for the
    # 5-tuple _arrays contract. Not drawn from rng -> resample order unchanged.
    return pd.DataFrame({"Date": dates.date,
                         "p_bull": np.clip(rng.uniform(0, 1, T), 0, 1),
                         "trail_ratio": rng.uniform(0.8, 1.4, T),
                         "lr_1x": lr1, "lr_3x": 3 * lr1 - 1e-4,
                         "lr_cash": np.full(T, np.log1p(0.02 / 252))})


def _mini_joint(train_end="2014-12-31"):
    """Shared 1-ticker, 2-refit_every fixture straddling train_end, reused
    by the orchestration tests below and by test_sweep_cache.py."""
    import pandas as pd, numpy as np
    dates = pd.bdate_range("2013-01-01", "2016-12-31")     # straddles train_end
    n = len(dates); rng = np.random.default_rng(5)
    def frame():
        l1 = rng.normal(0, 0.01, n)
        return pd.DataFrame({"Date": dates.date, "p_bull": rng.random(n),
                             "trail_ratio": rng.random(n) + 0.6, "lr_1x": l1,
                             "lr_3x": 3 * l1, "lr_cash": np.full(n, 1e-5)})
    return {"QQQ": {21: frame(), 63: frame()}}


def _mini_joint_small(train_end="2014-12-31"):
    """Compact 2-refit fixture straddling train_end for the fast cache-routing
    equivalence tests (the byte-exact property is n_boot/length-independent)."""
    import pandas as pd, numpy as np
    dates = pd.bdate_range("2014-06-02", "2015-03-31")
    n = len(dates); rng = np.random.default_rng(11)

    def frame():
        l1 = rng.normal(0, 0.01, n)
        return pd.DataFrame({"Date": dates.date, "p_bull": rng.random(n),
                             "trail_ratio": rng.random(n) + 0.6, "lr_1x": l1,
                             "lr_3x": 3 * l1, "lr_cash": np.full(n, 1e-5)})
    return {"QQQ": {21: frame(), 63: frame()}}


# --- route sweep_ticker/seed_tables/block126 through cached_sweep ---

def test_sweep_ticker_cache_dir_matches_direct(tmp_path, monkeypatch):
    """cache_dir set produces the byte-identical full frame (incl feasible) as
    the direct cache_dir=None path -- the routing changes nothing."""
    import strategy_sweep as ss, causal_backtest as cb
    frames = _mini_joint_small()["QQQ"]
    direct = ss.sweep_ticker("QQQ", frames, n_boot=48)
    monkeypatch.setattr(cb, "build_joint_frame",
                        lambda ticker, refit, *a, **k: frames[refit].copy())
    cached = ss.sweep_ticker("QQQ", frames, n_boot=48, cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(direct, cached, check_exact=True)


def test_sweep_ticker_threads_build_params_to_strict_loader(tmp_path,
                                                            monkeypatch):
    """Non-default data_dir/vol_window/warmup/n_restarts must reach the
    strict causal loader through the SweepSpec (else a non-default --data-dir
    run reads the wrong cache or crashes). The loader is called with
    strict=True."""
    import strategy_sweep as ss, causal_backtest as cb
    frames = _mini_joint_small()["QQQ"]
    seen = []

    def spy_build(ticker, refit, data_dir, cache_dir, vol_window=63,
                  warmup=1000, n_restarts=15, force=False, strict=False):
        seen.append((data_dir, vol_window, warmup, n_restarts, strict))
        return frames[refit].copy()
    monkeypatch.setattr(cb, "build_joint_frame", spy_build)
    ss.sweep_ticker("QQQ", frames, n_boot=48, cache_dir=str(tmp_path),
                    data_dir="custom/raw", vol_window=40, warmup=250,
                    n_restarts=7)
    assert seen
    for dd, vw, wu, nr, strict in seen:
        assert (dd, vw, wu, nr) == ("custom/raw", 40, 250, 7)
        assert strict is True


def test_seed_tables_and_block126_route_through_cache(tmp_path, monkeypatch):
    """compute_seed_tables and _block126_sensitivity accept cache_dir and, at
    cache_dir=None, remain byte-identical to the direct seeded core."""
    import strategy_sweep as ss
    frames = _mini_joint_small()["QQQ"]
    # cache_dir=None: seed tables equal the direct seeded core per seed
    tables = ss.compute_seed_tables("QQQ", frames, k=2, n_boot=48)
    for s, tbl in enumerate(tables):
        direct = ss._sweep_ticker_seeded("QQQ", frames, n_boot=48,
                                         excess_floor=0.0, block_len=ss.BLOCK_LEN,
                                         seed=s)
        pd.testing.assert_frame_equal(tbl, direct, check_exact=True)


def _report_train_fixture(feasible, lw=0.1, excess=-0.05):
    from strategy_sweep import grid_points
    rows = []
    for p in grid_points():
        if p["cash_arm"] != "never":     # selection/report tests exercise the
            continue                     # 360-point never arm (Stage-1 grid);
        # cash-arm selection is covered by its own tests.
        rows.append({**p, "expected_lw_ann": lw, "cvar5_maxdd": -0.3,
                     "cvar5_excess_dd": excess, "feasible": feasible,
                     "excess_tail_k": 5, "excess_tail_worst": excess - 0.02,
                     "excess_tail_median_maxdd_strategy": -0.45,
                     "excess_tail_worst_maxdd_strategy": -0.60,
                     "excess_tail_median_maxdd_1x": -0.40,
                     "excess_tail_worst_maxdd_1x": -0.50})
    return pd.DataFrame(rows)


def test_sweep_ticker_masks_train_era(monkeypatch):
    import strategy_sweep as ss
    joint = _toy_joint(T=300, start="2014-06-02")   # spans the 2015 boundary
    seen = {}
    real = ss.evaluate_point
    def spy(arrays, idx, *a, **k):
        seen["T"] = arrays[0].size
        return real(arrays, idx, *a, **k)
    monkeypatch.setattr(ss, "evaluate_point", spy)
    monkeypatch.setattr(ss, "grid_points",
                        lambda: [dict(enter_th=0.6, exit_th=0.4,
                                      min_dwell=0, overlay=True,
                                      refit_every=21, cash_arm="never")])
    res = ss.sweep_ticker("QQQ", {21: joint, 63: joint}, n_boot=50)
    n_train = (pd.to_datetime(joint["Date"]) <= "2014-12-31").sum()
    assert seen["T"] == n_train          # not one holdout row evaluated
    assert len(res) == 1 and "feasible" in res.columns


def test_neighborhood_median_prefers_plateau_over_spike():
    # Membership assertion, NOT point-equality: a flat plateau can tie the
    # neighborhood median across several cells, and
    # idxmax picks an arbitrary one. The property under test is only
    # "plateau beats spike".
    from strategy_sweep import neighborhood_score, grid_points
    rows = []
    for p in grid_points():
        if not (p["overlay"] and p["refit_every"] == 21 and p["min_dwell"] == 0
                and p["cash_arm"] == "never"):
            continue
        # spike at (0.80, 0.30); graded plateau centered on (0.60, 0.40)
        if p["enter_th"] == 0.80 and p["exit_th"] == 0.30:
            lw = 0.50
        elif p["enter_th"] == 0.60 and p["exit_th"] == 0.40:
            lw = 0.30
        elif abs(p["enter_th"] - 0.60) <= 0.051 and abs(p["exit_th"] - 0.40) <= 0.051:
            lw = 0.25
        else:
            lw = 0.05
        rows.append({**p, "expected_lw_ann": lw, "feasible": True})
    df = pd.DataFrame(rows).reset_index(drop=True)
    score = neighborhood_score(df)
    best = df.loc[score.idxmax()]
    plateau_cells = {(e, x) for e in (0.55, 0.60, 0.65)
                     for x in (0.35, 0.40, 0.45)}
    assert (best["enter_th"], best["exit_th"]) in plateau_cells
    assert (best["enter_th"], best["exit_th"]) != (0.80, 0.30)


def test_sweep_ticker_feasibility_boundary_and_baseline_pairing(monkeypatch):
    # Boundary pin from a REPRESENTABLE computed value: feasible at exactly
    # the floor (direct >=, no epsilon), infeasible one ulp-ish above.
    import strategy_sweep as ss
    joint = _toy_joint(T=200, start="2010-01-04")
    monkeypatch.setattr(ss, "grid_points",
                        lambda: [dict(enter_th=0.6, exit_th=0.4,
                                      min_dwell=0, overlay=True,
                                      refit_every=21, cash_arm="never")])
    free = ss.sweep_ticker("QQQ", {21: joint, 63: joint}, n_boot=50,
                           excess_floor=-10.0)
    assert free["feasible"].all()
    assert "cvar5_excess_dd" in free.columns
    v = float(free["cvar5_excess_dd"].iloc[0])
    at = ss.sweep_ticker("QQQ", {21: joint, 63: joint}, n_boot=50,
                         excess_floor=v)
    assert at["feasible"].all()                      # equality IS feasible
    above = ss.sweep_ticker("QQQ", {21: joint, 63: joint}, n_boot=50,
                            excess_floor=v + 1e-9)
    assert not above["feasible"].any()


def test_sweep_ticker_passes_precomputed_baseline(monkeypatch):
    # The per-refit baseline must reach evaluate_point as baseline_maxdd_b
    # (the precompute path, not the inline fallback).
    import strategy_sweep as ss
    joint = _toy_joint(T=200, start="2010-01-04")
    seen = {}
    real = ss.evaluate_point
    def spy(arrays, idx, *a, **k):
        seen["baseline"] = k.get("baseline_maxdd_b")
        return real(arrays, idx, *a, **k)
    monkeypatch.setattr(ss, "evaluate_point", spy)
    monkeypatch.setattr(ss, "grid_points",
                        lambda: [dict(enter_th=0.6, exit_th=0.4,
                                      min_dwell=0, overlay=True,
                                      refit_every=21, cash_arm="never")])
    ss.sweep_ticker("QQQ", {21: joint, 63: joint}, n_boot=50)
    assert seen["baseline"] is not None
    assert seen["baseline"].shape == (50,)


def test_holdout_always_1x_has_exactly_zero_excess_and_is_feasible():
    import strategy_sweep as ss
    joint = _toy_joint(T=250, start="2015-01-05")   # all holdout
    jj = {"QQQ": {21: joint, 63: joint}}
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
              overlay=True, refit_every=21, cash_arm="never")
    out = ss.evaluate_holdout([cfg], jj, n_boot=50)
    a1 = out[(out["config_id"] == "always_1x")].iloc[0]
    assert a1["cvar5_excess_dd"] == 0.0              # exact, by construction
    assert bool(a1["feasible"])                      # 0.0 >= -0.40 (default)


def test_cli_excess_floor_finite_default_and_non_finite_rejected():
    import strategy_sweep as ss
    args = ss._parse_args([])
    assert args.excess_floor == pytest.approx(-0.40)   # primary floor knob
    with pytest.raises(SystemExit):
        ss._parse_args(["--excess-floor", "nan"])
    with pytest.raises(SystemExit):
        ss._parse_args(["--excess-floor", "inf"])


def test_neighborhood_score_excludes_infeasible_neighbors():
    # Corner cell (0.55, 0.30) with ALL its in-fixture grid neighbors
    # infeasible at a huge lw: with proper exclusion its neighborhood median
    # is its own 0.10; a forgotten exclusion yields 9.9. Discriminating by
    # construction.
    from strategy_sweep import neighborhood_score, grid_points
    rows = []
    for p in grid_points():
        if not (p["overlay"] and p["refit_every"] == 21 and p["min_dwell"] == 0
                and p["cash_arm"] == "never"):
            continue
        corner = (p["enter_th"] == 0.55 and p["exit_th"] == 0.30)
        neighbor = (p["enter_th"] in (0.55, 0.60)
                    and p["exit_th"] in (0.30, 0.35) and not corner)
        rows.append({**p,
                     "expected_lw_ann": 9.9 if neighbor else 0.10,
                     "feasible": not neighbor})
    df = pd.DataFrame(rows).reset_index(drop=True)
    score = neighborhood_score(df)
    corner_idx = df[(df["enter_th"] == 0.55) & (df["exit_th"] == 0.30)].index[0]
    assert score.loc[corner_idx] == pytest.approx(0.10)
    assert (score.loc[df[~df["feasible"]].index] == -np.inf).all()
    # explicit-mask variant: same semantics via the feasible= parameter
    from strategy_sweep import neighborhood_score as ns
    score2 = ns(df, feasible=df["feasible"].values)
    assert (score2 == score).all()
    # empty-feasible-neighborhood edge: mask everything but the corner ->
    # its neighborhood is just itself; median = its own value, never -inf
    only_corner = (df["enter_th"] == 0.55) & (df["exit_th"] == 0.30)
    score3 = ns(df, feasible=only_corner.values)
    assert score3.loc[corner_idx] == pytest.approx(0.10)
    assert (score3.loc[df.index[~only_corner]] == -np.inf).all()


def test_select_plateau_requires_feasibility_everywhere():
    from strategy_sweep import select_plateau, grid_points
    base = pd.DataFrame([{**p, "expected_lw_ann": 0.1, "feasible": True}
                         for p in grid_points() if p["cash_arm"] == "never"])
    a, b = base.copy(), base.copy()
    winner = (a["enter_th"] == 0.60) & (a["exit_th"] == 0.40) & \
             (a["min_dwell"] == 0) & a["overlay"] & (a["refit_every"] == 21)
    a.loc[winner, "expected_lw_ann"] = 1.0
    b.loc[winner, "feasible"] = False        # fails the cap on ticker B
    out = select_plateau({"A": a, "B": b})
    c = out["chosen"]
    assert not (c["enter_th"] == 0.60 and c["exit_th"] == 0.40 and
                c["min_dwell"] == 0 and c["overlay"] and
                c["refit_every"] == 21)
    assert len(out["table"]) == 5
    # all surviving rows tie at 0.1 -> the deterministic lean-cautious
    # tie-break must pick the most conservative corner
    assert (c["enter_th"], c["exit_th"], c["min_dwell"],
            c["overlay"], c["refit_every"]) == (0.80, 0.50, 10, True, 63)


def test_holdout_evaluates_only_post_train_rows(monkeypatch):
    import strategy_sweep as ss
    joint = _toy_joint(T=300, start="2014-06-02")
    seen = []
    real = ss.evaluate_point
    def spy(arrays, idx, *a, **k):
        seen.append(arrays[0].size)
        return real(arrays, idx, *a, **k)
    monkeypatch.setattr(ss, "evaluate_point", spy)
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
              overlay=True, refit_every=21, cash_arm="never")
    jj = {tk: {21: joint, 63: joint} for tk in ("QQQ", "SOX", "SPY")}
    out = ss.evaluate_holdout([cfg], jj, n_boot=50)
    n_holdout = (pd.to_datetime(joint["Date"]) > "2014-12-31").sum()
    assert all(T == n_holdout for T in seen)      # never a train row
    assert len(out) == 12                          # 3 tickers x 4 configs
    assert set(out["config_id"]) == {"config_0", "live_default",
                                     "always_1x", "always_3x"}


def test_always_baselines_are_degenerate_gates():
    import strategy_sweep as ss
    joint = _toy_joint(T=250, start="2015-01-05")   # all holdout
    jj = {"QQQ": {21: joint, 63: joint}}
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
              overlay=True, refit_every=21, cash_arm="never")
    out = ss.evaluate_holdout([cfg], jj, n_boot=50)
    a1 = out[(out["config_id"] == "always_1x")].iloc[0]
    a3 = out[(out["config_id"] == "always_3x")].iloc[0]
    assert a1["switches_realized"] == 0
    assert a3["switches_realized"] == 1            # the one arming switch
    assert a1["median_terminal_ratio_vs_1x"] == pytest.approx(1.0)


def test_holdout_dedups_and_keys_configs(monkeypatch):
    """distinct_configs is de-duplicated by the _PARAM_COLS tuple
    before evaluation -- two IDENTICAL viable configs must be evaluated
    exactly ONCE, plus the three explicit references (live_default,
    always_1x, always_3x), for a total of 4 unique evaluate_point calls per
    ticker. Discriminates a forgotten-dedup implementation (which would call
    evaluate_point 5 times: 2 distinct-but-identical + 3 references)."""
    import strategy_sweep as ss
    calls = {"n": 0}
    def counting(*a, **k):
        calls["n"] += 1
        return {kk: 0.0 for kk in ("expected_lw_ann", "cvar5_maxdd",
            "cvar5_lw_ann", "median_terminal_ratio_vs_1x", "realized_lw_ann",
            "realized_maxdd", "realized_calmar", "switches_realized",
            "lev_turnover", "cash_occupancy", "cash_zero_use_freq",
            "cvar5_excess_dd", "excess_tail_k", "excess_tail_worst",
            "excess_tail_median_maxdd_strategy", "excess_tail_worst_maxdd_strategy",
            "excess_tail_median_maxdd_1x", "excess_tail_worst_maxdd_1x")}
    monkeypatch.setattr(ss, "evaluate_point", counting)
    # two IDENTICAL viable configs -> evaluated once, not twice
    c = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
             refit_every=21, cash_arm="never")
    jj = _mini_joint()   # single ticker QQQ
    out = ss.evaluate_holdout([dict(c), dict(c)], jj, n_boot=50)
    assert calls["n"] == 4          # 1 distinct + 3 references, single ticker
    assert "config_id" in out.columns
    assert len(out) == 4            # one row per unique config (1 ticker)
    assert out["config_id"].nunique() == 4
    assert set(out["config_id"]) == {"config_0", "live_default",
                                     "always_1x", "always_3x"}
    # cash_arm must never be None/NaN anywhere in the returned frame
    assert out["cash_arm"].notna().all()
    assert not (out["cash_arm"] == None).any()  # noqa: E711


def test_main_propagates_excess_floor_and_sensitivity_is_train_only(
        monkeypatch, tmp_path):
    """main(argv) wiring: --excess-floor must
    reach sweep_ticker, evaluate_holdout, AND write_report unchanged; the
    block-126 sensitivity step must only ever bootstrap TRAIN-era row counts
    (never the differently sized holdout); and the CLI floor must appear in the
    selection-floor union that write_report + the return dict carry. The
    monkeypatched grid spans all three cash arms at ONE (enter,exit,dwell,
    overlay,refit) cell so the per-cash-arm heatmap loop finds rows for each
    arm (plot_heatmaps fails loud on an empty arm slice, by design)."""
    import strategy_sweep as ss

    joint = _toy_joint(T=300, start="2014-06-02")   # dates for the T-count math
    n_train = int((pd.to_datetime(joint["Date"]) <= "2014-12-31").sum())
    n_holdout = len(joint) - n_train
    assert n_train != n_holdout          # discriminates train vs holdout T

    # Per-ticker DISTINCT frames (same dates) so the 24 cache units get distinct
    # fingerprints -- identical frames would collide (ticker is not in the key)
    # and thrash the cache. Same dates keep n_train/n_holdout valid for all.
    import causal_backtest as cb
    tk_joints = {tk: _toy_joint(T=300, start="2014-06-02", seed=i + 1)
                 for i, tk in enumerate(ss.TICKERS)}

    def fake_build(ticker, refit_every, data_dir, cache_dir, vol_window=63,
                   warmup=1000, n_restarts=15, force=False, strict=False):
        return tk_joints[ticker].copy()
    monkeypatch.setattr(ss, "build_joint_frame", fake_build)     # phase-1 builder
    monkeypatch.setattr(cb, "build_joint_frame", fake_build)     # strict loader
    # all three cash arms at one cell -> the primary-floor heatmap loop (once
    # per GRID["cash_arm"]) has rows for every arm.
    monkeypatch.setattr(ss, "grid_points",
                        lambda: [dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
                                      overlay=True, refit_every=21,
                                      cash_arm=arm)
                                 for arm in ("never", "gap_10", "gap_20")])

    seen = {"sweep_floors": []}
    real_sweep_ticker = ss.sweep_ticker
    def spy_sweep_ticker(ticker, joint_by_refit, n_boot=ss.N_BOOT_SWEEP,
                         excess_floor=ss.EXCESS_FLOOR, train_end=ss.TRAIN_END,
                         block_len=ss.BLOCK_LEN, **kwargs):
        seen["sweep_floors"].append(excess_floor)
        return real_sweep_ticker(ticker, joint_by_refit, n_boot=n_boot,
                                 excess_floor=excess_floor,
                                 train_end=train_end, block_len=block_len,
                                 **kwargs)
    monkeypatch.setattr(ss, "sweep_ticker", spy_sweep_ticker)

    real_evaluate_holdout = ss.evaluate_holdout
    def spy_evaluate_holdout(distinct_configs, jj, n_boot=ss.N_BOOT_FINAL,
                             train_end=ss.TRAIN_END,
                             excess_floor=ss.EXCESS_FLOOR, on_leverage=3.0):
        seen["holdout_floor"] = excess_floor
        seen["holdout_configs_is_list"] = isinstance(distinct_configs, list)
        return real_evaluate_holdout(distinct_configs, jj, n_boot=n_boot,
                                     train_end=train_end,
                                     excess_floor=excess_floor,
                                     on_leverage=on_leverage)
    monkeypatch.setattr(ss, "evaluate_holdout", spy_evaluate_holdout)

    real_write_report = ss.write_report
    def spy_write_report(train_results, selection_matrix, cash_lift,
                         static_ladders, cost_sens, holdout, block126,
                         png_paths, out_fp, excess_floor, selection_floors,
                         provenance=None):
        seen["report_floor"] = excess_floor
        seen["report_floors"] = list(selection_floors)
        seen["provenance"] = provenance
        return real_write_report(train_results, selection_matrix, cash_lift,
                                 static_ladders, cost_sens, holdout, block126,
                                 png_paths, out_fp, excess_floor,
                                 selection_floors, provenance=provenance)
    monkeypatch.setattr(ss, "write_report", spy_write_report)

    sens126_T = []
    real_bootstrap_indices = ss.bootstrap_indices
    def spy_bootstrap_indices(T, n_boot, block_len=63, seed=0):
        if block_len == 126:
            sens126_T.append(T)
        return real_bootstrap_indices(T, n_boot, block_len=block_len,
                                      seed=seed)
    monkeypatch.setattr(ss, "bootstrap_indices", spy_bootstrap_indices)

    out_dir = str(tmp_path / "out")
    result = ss.main(["--excess-floor", "-0.75", "--quick", "--workers", "1",
                      "--data-dir", str(tmp_path / "raw"),
                      "--cache-dir", str(tmp_path / "cache"),
                      "--out-dir", out_dir])

    # --excess-floor reaches sweep_ticker (once per ticker), evaluate_holdout,
    # and write_report unchanged.
    assert seen["sweep_floors"] and all(
        f == pytest.approx(-0.75) for f in seen["sweep_floors"])
    assert seen["holdout_floor"] == pytest.approx(-0.75)
    assert seen["holdout_configs_is_list"]        # LIST of distinct configs
    assert seen["report_floor"] == pytest.approx(-0.75)

    # the CLI floor is unioned into the selection-floor set write_report gets
    assert -0.75 in [pytest.approx(f) for f in seen["report_floors"]]
    assert set(ss.SELECTION_DIAGNOSTIC_FLOORS) <= set(seen["report_floors"])

    # provenance reaches write_report and reflects the actual run params
    assert seen["provenance"]["mode"] == "quick"
    assert seen["provenance"]["excess_floor"] == pytest.approx(-0.75)
    assert seen["provenance"]["n_boot_sweep"] == 500

    # block-126 sensitivity bootstraps saw ONLY train-era row counts
    assert sens126_T
    assert all(T == n_train for T in sens126_T)
    assert n_holdout not in sens126_T

    # Return shape: the per-(floor, ticker) matrix + block-126 dict,
    # both keyed by the selection-floor union (primary floor present).
    assert -0.75 in [pytest.approx(f) for f in result["floors"]]
    assert set(result["selection_matrix"]) == set(result["floors"])
    assert set(result["block126"]) == set(result["floors"])
    for tk in ss.TICKERS:
        assert tk in result["selection_matrix"][-0.75]
    assert os.path.exists(result["report_fp"])
    for tk in ss.TICKERS:
        assert os.path.exists(os.path.join(out_dir, f"gate_sweep_{tk}.csv"))


# --- main() cache wiring, --quick guard, --workers, force lifecycle -

def test_quick_against_decision_cache_is_rejected_side_effect_free(
        tmp_path, monkeypatch):
    """--quick against the decision cache 'data/backtest' is rejected via
    realpath BEFORE any filesystem writes (no out-dir created)."""
    import strategy_sweep as ss
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        ss.main(["--quick", "--cache-dir", "data/backtest",
                 "--data-dir", "data/raw", "--out-dir", "out"])
    assert not (tmp_path / "out").exists()      # rejected before any writes


def test_workers_default_is_2_and_below_one_rejected():
    import strategy_sweep as ss
    from sweep_cache import effective_workers
    assert ss._parse_args([]).workers == 2
    with pytest.raises(ValueError):
        effective_workers(0)


def test_main_cache_lifecycle_24_prefill_zero_recompute_resume_force(
        tmp_path, monkeypatch):
    """A cache-enabled main() computes EXACTLY the 24
    enumerated units once with ZERO recomputes during assembly; a warm rerun
    recomputes 0; a resume (one unit deleted) recomputes 1; --force-rebuild
    recomputes 24. This is what proves the cache is actually enabled and that
    assembly is pure hits (heatmaps stubbed to keep the 4 runs fast)."""
    import strategy_sweep as ss, causal_backtest as cb, glob
    tk_joints = {tk: _toy_joint(T=260, start="2014-06-02", seed=i + 1)
                 for i, tk in enumerate(ss.TICKERS)}

    def fake_build(ticker, refit_every, data_dir, cache_dir, vol_window=63,
                   warmup=1000, n_restarts=15, force=False, strict=False):
        return tk_joints[ticker].copy()
    monkeypatch.setattr(ss, "build_joint_frame", fake_build)
    monkeypatch.setattr(cb, "build_joint_frame", fake_build)
    monkeypatch.setattr(ss, "plot_heatmaps", lambda *a, **k: [])   # skip mpl
    monkeypatch.setattr(ss, "grid_points",
                        lambda: [dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
                                      overlay=True, refit_every=21,
                                      cash_arm=arm)
                                 for arm in ("never", "gap_10", "gap_20")])
    computes = {"n": 0}
    real_compute = ss._sweep_ticker_seeded

    def spy_compute(*a, **k):
        computes["n"] += 1
        return real_compute(*a, **k)
    monkeypatch.setattr(ss, "_sweep_ticker_seeded", spy_compute)

    cache = str(tmp_path / "cache")
    argv = ["--quick", "--workers", "1", "--data-dir", str(tmp_path / "raw"),
            "--cache-dir", cache, "--out-dir", str(tmp_path / "out")]

    computes["n"] = 0
    ss.main(argv)
    assert computes["n"] == 24            # cold: exactly 24 units, 0 in assembly

    computes["n"] = 0
    ss.main(argv)
    assert computes["n"] == 0             # warm: pure resume

    units = sorted(glob.glob(os.path.join(cache, "sweep_*.parquet")))
    assert len(units) == 24               # 24 distinct on-disk units
    os.remove(units[0])
    os.remove(units[0].replace(".parquet", ".meta.json"))
    computes["n"] = 0
    ss.main(argv)
    assert computes["n"] == 1             # resume: only the deleted unit

    computes["n"] = 0
    ss.main(argv + ["--force-rebuild"])
    assert computes["n"] == 24            # force: all 24 recompute


def test_report_and_heatmaps_written(tmp_path, monkeypatch):
    # The multi-floor matrix signature: plot_heatmaps takes a per-ticker
    # record + a fixed cash_arm; write_report consumes the selection_matrix
    # + auxiliary frames built by _report_new_inputs.
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
               refit_every=21, cash_arm="never")
    selection_by_ticker = {tk: {"config": dict(cfg)} for tk in train}
    pngs = ss.plot_heatmaps(train, selection_by_ticker, str(tmp_path),
                            cfg["cash_arm"])
    assert len(pngs) == 3 and all(os.path.exists(p) for p in pngs)
    inp = _report_new_inputs(train)
    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], pngs, fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"])
    text = open(fp, encoding="utf-8").read()
    assert "expected log-wealth" in text
    assert "never optimized" in text
    assert "worst paired excess-drawdown tail" in text
    assert "no absolute-loss guarantee" in text.lower()
    assert "Excess-floor calibration" in text
    assert "-0.15" in text
    assert "--excess-floor -0.15" in text        # reproduction command


def test_report_discloses_nonviable_selection_cells(tmp_path):
    # Selection uses PER-CELL viability rather than one global lean-cautious
    # plateau fallback: a (floor, ticker) cell with no stably-feasible
    # config renders "no viable config at this floor." + its selection_mode,
    # and is NOT given a fabricated winner (neither in the selection-records
    # table nor in the tail-composition section).
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(False, excess=-0.30)
             for tk in ("QQQ", "SOX", "SPY")}
    inp = _report_new_inputs(train, floors=(-0.15, -0.40), primary=-0.15)
    for tk in train:                               # every ticker non-viable
        inp["selection_matrix"][-0.15][tk] = {
            "config": dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
                           overlay=True, refit_every=21, cash_arm="never"),
            "viable": False, "feasible_count": 0,
            "selection_mode": "no_feasible_fallback", "score": float("-inf")}
    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"])
    with open(fp, encoding="utf-8") as f:
        text = f.read()
    assert "no viable config at this floor." in text
    assert "no_feasible_fallback" in text           # selection_mode disclosed
    # rendered in BOTH the per-cell records table AND the tail composition,
    # for all three non-viable tickers -> many occurrences, never a winner
    assert text.count("no viable config at this floor.") >= 3


def test_report_dia_warmup_narrative(tmp_path):
    """DIA is in the universe with a short corpus (DIA data starts 1998-01-20;
    the 1,000-bar HMM warmup pushes its causal eval start to ~2002-01): the
    dot-com honesty bullet must disclose that only the 2002 TAIL of the crash
    is inside DIA's eval window (in-train), unlike SPY/SOX (fully in-train)
    and QQQ (out-of-eval)."""
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True)
             for tk in ("QQQ", "SOX", "SPY", "DIA")}
    inp = _report_new_inputs(train)
    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"])
    with open(fp, encoding="utf-8") as f:
        text = f.read()
    assert "for DIA" in text
    assert "2002 tail" in text
    assert "worst paired excess-drawdown tail" in text


def test_feasible_report_has_no_fallback_and_heatmaps_in_charts_dir(tmp_path):
    # Counterpart guard: an all-viable run carries no
    # global plateau FALLBACK banner (selection is per ticker), and the faceted
    # heatmap links (charts/gate_sweep_{ticker}_{cash_arm}.png) render
    # relative to the report dir with forward slashes, never absolute.
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
               refit_every=21, cash_arm="never")
    selection_by_ticker = {tk: {"config": dict(cfg)} for tk in train}
    pngs = ss.plot_heatmaps(train, selection_by_ticker, str(tmp_path),
                            cfg["cash_arm"])
    assert len(pngs) == 3
    for p, tk in zip(pngs, ("QQQ", "SOX", "SPY")):
        assert os.path.exists(p)
        assert os.path.basename(p) == f"gate_sweep_{tk}_{cfg['cash_arm']}.png"
        assert os.path.basename(os.path.dirname(p)) == "charts"
    inp = _report_new_inputs(train)
    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], pngs, fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"])
    with open(fp, encoding="utf-8") as f:
        text = f.read()
    assert "FALLBACK" not in text
    assert "feasible_all" not in text            # no global-intersection count
    # image links relative to the report's own dir, forward slashes, never
    # absolute/backslashed
    assert (f"![heatmap](charts/gate_sweep_QQQ_{cfg['cash_arm']}.png)"
           in text)
    assert "![heatmap](" + str(tmp_path).replace("\\", "/") not in text
    assert "![heatmap](C:" not in text


# --- cash-arm-faceted heatmaps -----------------------------------

def _cash_arm_facet_fixture():
    """One ticker ("TK"), two (enter_th, exit_th) cells, EACH populated by
    both a `never` row and a `gap_10` row at the SAME cell but with
    DIFFERENT feasibility/expected_lw_ann. min_dwell/overlay/refit_every
    are held fixed (that's the ticker's "selection" slice) -- only cash_arm
    varies within a cell. This is the minimal frame that discriminates a
    cross-arm-averaging bug: pivoting on (enter_th, exit_th) WITHOUT first
    fixing cash_arm would land the never row and the gap_10 row of a cell
    in the same pivot bucket, and pivot_table's default aggfunc='mean'
    would blend their booleans (e.g. True=1.0 and False=0.0 -> 0.5) instead
    of reporting either row's real value."""
    common = dict(min_dwell=0, overlay=True, refit_every=21)
    rows = [
        {**common, "enter_th": 0.60, "exit_th": 0.40, "cash_arm": "never",
         "feasible": True, "expected_lw_ann": 0.12},
        {**common, "enter_th": 0.60, "exit_th": 0.40, "cash_arm": "gap_10",
         "feasible": False, "expected_lw_ann": 0.03},
        {**common, "enter_th": 0.70, "exit_th": 0.30, "cash_arm": "never",
         "feasible": False, "expected_lw_ann": 0.05},
        {**common, "enter_th": 0.70, "exit_th": 0.30, "cash_arm": "gap_10",
         "feasible": True, "expected_lw_ann": 0.09},
    ]
    return pd.DataFrame(rows)


def test_arm_sliced_pivots_never_blend_a_different_arms_row(tmp_path):
    # Headless backend, set BEFORE importing pyplot -- belt
    # and suspenders on top of plot_heatmaps' own matplotlib.use("Agg").
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (import only, unused)
    import strategy_sweep as ss

    df = _cash_arm_facet_fixture()
    cfg = dict(enter_th=0.60, exit_th=0.40, min_dwell=0, overlay=True,
              refit_every=21, cash_arm="never")

    # Genuine no-cross-arm-averaging proof, run against the REAL slicing/
    # pivoting helper plot_heatmaps itself calls -- not a test-side
    # reimplementation of the pivot logic.
    enters_n, exits_n, val_n, feas_n = ss._arm_sliced_pivots(df, cfg, "never")
    enters_g, exits_g, val_g, feas_g = ss._arm_sliced_pivots(df, cfg, "gap_10")

    # The shared (0.60, 0.40) cell must reflect ONLY that arm's own row --
    # exactly 1.0 (True) for never, exactly 0.0 (False) for gap_10. A
    # cross-arm-averaging bug would blend True(1)+False(0) -> 0.5 or NaN.
    never_cell = feas_n.loc[0.60, 0.40]
    gap10_cell = feas_g.loc[0.60, 0.40]
    assert not pd.isna(never_cell)
    assert not pd.isna(gap10_cell)
    assert never_cell == pytest.approx(1.0)
    assert gap10_cell == pytest.approx(0.0)
    assert never_cell != pytest.approx(0.5)   # the blended-average failure mode
    assert val_n.loc[0.60, 0.40] == pytest.approx(0.12)   # never's own value
    assert val_g.loc[0.60, 0.40] == pytest.approx(0.03)   # gap_10's own value,
                                                          # never averaged in

    # The two facets genuinely depend on cash_arm -- they must differ, not
    # just at one cell but as whole pivots.
    assert not feas_n.equals(feas_g)
    assert not val_n.equals(val_g)

    # Files land per-arm under out_dir/charts, named
    # gate_sweep_{ticker}_{cash_arm}.png.
    results = {"TK": df}
    selection_by_ticker = {"TK": {"config": cfg, "viable": True,
                                  "feasible_count": 1,
                                  "selection_mode": "plateau", "score": 1.0}}
    out_dir = str(tmp_path)
    pngs_never = ss.plot_heatmaps(results, selection_by_ticker, out_dir,
                                  "never")
    pngs_gap10 = ss.plot_heatmaps(results, selection_by_ticker, out_dir,
                                  "gap_10")
    assert len(pngs_never) == 1 and os.path.exists(pngs_never[0])
    assert os.path.basename(pngs_never[0]) == "gate_sweep_TK_never.png"
    assert len(pngs_gap10) == 1 and os.path.exists(pngs_gap10[0])
    assert os.path.basename(pngs_gap10[0]) == "gate_sweep_TK_gap_10.png"
    assert pngs_never[0] != pngs_gap10[0]


def test_plot_heatmaps_raises_on_empty_arm_slice(tmp_path):
    # Fail-silent guard: a (ticker, cash_arm) slice with
    # ZERO rows -- e.g. a caller bug passing a cash_arm that doesn't exist
    # in that ticker's results, or a ticker/arm mismatch -- must fail loud
    # instead of ax.imshow()-ing an empty pivot into a plausible-looking but
    # garbage/empty PNG. _cash_arm_facet_fixture's "TK" ticker only has
    # "never" and "gap_10" rows; requesting cash_arm="always" (not present
    # in the fixture at all) reproduces the exact empty-slice condition.
    import strategy_sweep as ss

    df = _cash_arm_facet_fixture()
    cfg = dict(enter_th=0.60, exit_th=0.40, min_dwell=0, overlay=True,
              refit_every=21, cash_arm="never")
    results = {"TK": df}
    selection_by_ticker = {"TK": {"config": cfg, "viable": True,
                                  "feasible_count": 1,
                                  "selection_mode": "plateau", "score": 1.0}}
    out_dir = str(tmp_path)

    with pytest.raises(ValueError, match=r"TK.*always"):
        ss.plot_heatmaps(results, selection_by_ticker, out_dir, "always")

    # No garbage PNG must be written for the failed arm.
    charts_dir = os.path.join(out_dir, "charts")
    if os.path.isdir(charts_dir):
        assert not any("always" in fn for fn in os.listdir(charts_dir))


def test_report_renders_run_provenance(tmp_path):
    # The report must identify the run that produced it
    # (mode, boot counts, floor, corpus max dates), and a --quick run must
    # be branded not-decision-grade. No wall-clock timestamp (deterministic
    # regeneration diffs).
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    inp = _report_new_inputs(train)
    prov = {"mode": "full", "n_boot_sweep": 2000, "n_boot_holdout": 10000,
            "n_restarts": 15, "block_len": 63, "excess_floor": -0.15,
            "train_end": "2014-12-31",
            "corpus_max_date": {"QQQ": "2020-01-31", "SOX": "2020-01-31",
                                "SPY": "2020-01-31"}}
    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"], provenance=prov)
    with open(fp, encoding="utf-8") as f:
        text = f.read()
    assert "Run provenance" in text
    assert "| mode | full |" in text
    assert "| excess_floor | -0.15 |" in text
    assert "QQQ: 2020-01-31" in text               # corpus max dates
    assert "not decision-grade" not in text        # full mode: no warning
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"],
                    provenance={**prov, "mode": "quick"})
    with open(fp, encoding="utf-8") as f:
        text = f.read()
    assert "QUICK MODE" in text
    assert "not decision-grade" in text


def test_report_renders_calibration_and_tail_composition(tmp_path):
    # Calibration + tail composition with per-ticker selection: the tail
    # composition is per ticker at the PRIMARY floor's selection (not one
    # global chosen config), under the "paired excess-drawdown" naming.
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    inp = _report_new_inputs(train, floors=(-0.15, -0.40), primary=-0.15)
    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"])
    text = open(fp, encoding="utf-8").read()
    assert "Excess-floor calibration" in text
    assert "within 0.02 of the floor" in text
    # exact rendered calibration row: constant excess -0.05 -> all
    # percentiles -0.0500, 360/360 feasible, near_floor 0
    assert ("| QQQ | 360/360 | -0.0500 | -0.0500 | -0.0500 | -0.0500 "
            "| -0.0500 | -0.0500 | -0.0500 | 0 |") in text
    assert "Worst paired excess-drawdown tail composition" in text
    for tk in ("QQQ", "SOX", "SPY"):
        assert f"{tk}: k=5" in text
    assert "worst_excess=-0.070" in text       # fixture: excess - 0.02
    assert "Full-grid resweep" in text         # block-126 sensitivity prose
    assert "descriptive only" in text          # holdout flag semantics
    assert "excess_dd" in text


def test_report_mixed_viability_floor_is_per_ticker_independent(tmp_path):
    # Selection is PER TICKER, so there is no global
    # feasible-everywhere intersection to warn on: at one floor some tickers
    # can be viable while others are not, and each cell renders on its own
    # (viable cells show their config; non-viable cells show "no viable
    # config at this floor.").
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    inp = _report_new_inputs(train, floors=(-0.15,), primary=-0.15)
    sm = inp["selection_matrix"][-0.15]
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
               refit_every=21, cash_arm="never")
    sm["QQQ"] = {"config": dict(cfg), "viable": True, "feasible_count": 200,
                 "selection_mode": "plateau", "score": 0.10}
    sm["SOX"] = {"config": dict(cfg), "viable": False, "feasible_count": 0,
                 "selection_mode": "no_feasible_fallback",
                 "score": float("-inf")}
    sm["SPY"] = {"config": dict(cfg), "viable": True, "feasible_count": 360,
                 "selection_mode": "plateau", "score": 0.10}
    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"])
    text = open(fp, encoding="utf-8").read()
    assert "no viable config at this floor." in text   # SOX non-viable cell
    for tk in ("QQQ", "SOX", "SPY"):                    # all three present
        assert f"| {tk} |" in text
    assert "feasible_all" not in text          # no global-intersection banner
    assert "FALLBACK" not in text


def test_report_consumes_selection_matrix_not_recompute(tmp_path, monkeypatch):
    # The report must render selection/config numbers FROM the passed
    # selection_matrix (and block126) as given, never re-derive them: doctor a
    # cell's config with a sentinel enter_th no fixture row carries, and make
    # any recomputation fail loudly.
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    inp = _report_new_inputs(train, floors=(-0.40,), primary=-0.40)
    inp["selection_matrix"][-0.40]["QQQ"] = {
        "config": dict(enter_th=0.123456, exit_th=0.4, min_dwell=0,
                       overlay=True, refit_every=21, cash_arm="never"),
        "viable": True, "feasible_count": 360, "selection_mode": "plateau",
        "score": 0.10}

    def boom(*a, **k):
        raise AssertionError("report must not recompute selection")
    monkeypatch.setattr(ss, "neighborhood_score", boom)
    monkeypatch.setattr(ss, "select_per_ticker", boom)
    monkeypatch.setattr(ss, "select_plateau", boom)
    monkeypatch.setattr(ss, "_block126_sensitivity", boom)

    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.40,
                    selection_floors=inp["selection_floors"])
    text = open(fp, encoding="utf-8").read()
    assert "0.123456" in text        # rendered verbatim from the passed record


# --- multi-floor matrix report, cash-lift, static-L, cost/holdout text ---

def _report_new_inputs(train, floors=(-0.15, -0.40), primary=-0.15,
                       nonviable_ticker=None):
    """Assemble the write_report inputs (selection_matrix, cash_lift,
    static_ladders, cost_sens, holdout, block126) from a train_results dict of
    _report_train_fixture frames. Shapes are the ones main()'s wiring
    produces (documented in write_report's docstring). Optionally
    marks ONE ticker non-viable at the primary floor so the "no viable config
    at this floor" rendering fires on real data."""
    from strategy_sweep import _PARAM_COLS
    cfg = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
               refit_every=21, cash_arm="never")

    def rec(viable):
        return {"config": dict(cfg), "viable": viable,
                "feasible_count": 360 if viable else 0,
                "selection_mode": "plateau" if viable else "no_feasible_fallback",
                "score": 0.10 if viable else float("-inf")}

    selection_matrix = {}
    for fl in floors:
        cells = {}
        for tk in train:
            viable = not (fl == primary and tk == nonviable_ticker)
            cells[tk] = rec(viable)
        selection_matrix[fl] = cells

    cash_lift = {tk: pd.DataFrame([
        {"enter_th": 0.6, "exit_th": 0.4, "min_dwell": 0, "overlay": True,
         "refit_every": 21, "cash_arm": "never", "d_expected_lw_ann": 0.0,
         "d_cvar5_excess_dd": 0.0, "cash_occupancy": 0.0,
         "cash_zero_use_freq": 1.0},
        {"enter_th": 0.6, "exit_th": 0.4, "min_dwell": 0, "overlay": True,
         "refit_every": 21, "cash_arm": "gap_10", "d_expected_lw_ann": -0.02,
         "d_cvar5_excess_dd": 0.05, "cash_occupancy": 0.2,
         "cash_zero_use_freq": 0.5}]) for tk in train}

    static_ladders = {tk: {
        "train": pd.DataFrame([
            {"L": 1.0, "expected_lw_ann": 0.10, "expected_lw_ann_costed": 0.10,
             "cvar5_maxdd": -0.30, "cvar5_excess_dd": 0.0},
            {"L": 1.5, "expected_lw_ann": 0.12, "expected_lw_ann_costed": 0.11,
             "cvar5_maxdd": -0.42, "cvar5_excess_dd": -0.12}]),
        "holdout": pd.DataFrame([
            {"L": 1.0, "expected_lw_ann": 0.05, "expected_lw_ann_costed": 0.05,
             "cvar5_maxdd": -0.20, "cvar5_excess_dd": 0.0}]),
    } for tk in train}

    cost_sens = pd.DataFrame([{
        "ticker": tk, **cfg,
        "expected_lw_ann_risk_notional": 0.10,
        "expected_lw_ann_dollar_turnover": 0.09,
        "d_expected_lw_ann": -0.01,
        "cvar5_excess_dd_risk_notional": -0.05,
        "cvar5_excess_dd_dollar_turnover": -0.04,
        "d_cvar5_excess_dd": 0.01} for tk in sorted(train)])

    holdout = pd.DataFrame([
        {"ticker": tk, "config_id": cid, **cfg, "expected_lw_ann": 0.1,
         "cvar5_maxdd": -0.3, "cvar5_excess_dd": -0.05, "feasible": True}
        for tk in sorted(train)
        for cid in ("config_0", "live_default", "always_1x", "always_3x")])

    block126 = {fl: {tk: {**cfg, "rank_block126": 1 + i,
                          "score_block126": 0.10,
                          "feasible_block126": True}
                     for i, tk in enumerate(sorted(train))}
                for fl in floors}

    return dict(selection_matrix=selection_matrix, cash_lift=cash_lift,
                static_ladders=static_ladders, cost_sens=cost_sens,
                holdout=holdout, block126=block126,
                selection_floors=list(floors))


def test_report_has_holdout_and_cost_disclosures_and_naming(tmp_path):
    # Guard against a trivially-true acceptance test: the first block of
    # assertions below checks prose only -- each substring comes from a
    # section header or description SENTENCE, never from the data itself,
    # so it would pass even on a near-empty report (empty
    # selection_matrix={}, empty cost_sens/holdout, None cash_lift/
    # static_ladders/block126). It is regression cover for the
    # naming/cost/holdout prose; the DATA-dependent block further down
    # carries the acceptance burden.
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    inp = _report_new_inputs(train, floors=(-0.15, -0.40), primary=-0.15,
                             nonviable_ticker="SPY")   # one non-viable cell

    # Sentinels planted directly in the fixture data (never in write_report's
    # static prose) so the assertions below can ONLY pass if write_report
    # actually renders the passed-in DATA for sections 4, 7, and 8.
    SENTINEL_ENTER_TH = 0.123456                # sec 4: viable-cell config
    inp["selection_matrix"][-0.15]["QQQ"]["config"] = dict(
        inp["selection_matrix"][-0.15]["QQQ"]["config"],
        enter_th=SENTINEL_ENTER_TH)

    SENTINEL_DELTA = -0.918273                  # sec 7: cost-sens delta
    cost_sens = inp["cost_sens"].copy()
    cost_sens.loc[cost_sens["ticker"] == "QQQ", "d_expected_lw_ann"] = \
        SENTINEL_DELTA
    inp["cost_sens"] = cost_sens

    SENTINEL_CONFIG_ID = "config_0_SENTINEL77"  # sec 8: holdout config_id
    holdout = inp["holdout"].copy()
    mask = (holdout["ticker"] == "QQQ") & (holdout["config_id"] == "config_0")
    holdout.loc[mask, "config_id"] = SENTINEL_CONFIG_ID
    inp["holdout"] = holdout

    fp = str(tmp_path / "r.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"], provenance=None)
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    low = text.lower()
    assert "worst paired excess-drawdown tail" in text
    # Spent holdout (discriminating): the holdout is SPENT and forward/paper data is the
    # true confirmation -- not merely the words.
    assert "holdout" in low and ("spent" in low or "not fresh" in low
                                 or "already" in low)
    assert "forward" in low and ("paper" in low or "confirmation" in low)
    assert "no viable config" in low
    assert "cost sensitivity" in low
    assert "risk-notional" in low and ("dollar" in low or "turnover" in low)

    # --- Data-dependent (the actual acceptance burden) ---------------------
    # (1) sec 8 holdout: a distinctive config_id planted on a real holdout
    # row is rendered verbatim -- proves the holdout table is populated from
    # the passed `holdout` frame, not merely described.
    assert SENTINEL_CONFIG_ID in text
    # (2) sec 7 cost sensitivity: a distinctive delta planted on a real
    # cost_sens row is rendered (4dp, per _df_md's float_fmt) -- proves the
    # Cost-sensitivity section is populated from the passed `cost_sens` frame.
    assert f"{SENTINEL_DELTA:.4f}" in text
    # (3) sec 4 per-cell records: a distinctive enter_th planted on a VIABLE
    # cell's config is rendered verbatim -- proves viable cells render their
    # own passed-in config (write_report never recomputes selection).
    assert str(SENTINEL_ENTER_TH) in text

    # (4) "no viable config at this floor." must be tied to the ACTUAL
    # non-viable (floor, ticker) cell -- not just the shared report section 4 and 10
    # description prose, which contains the phrase once regardless of any
    # data. Rebuild the identically-shaped report with every cell viable and
    # require the real (SPY non-viable) report to carry MORE occurrences: an
    # all-viable report must not satisfy this assertion.
    n_nonviable = low.count("no viable config at this floor.")
    inp_all_viable = _report_new_inputs(train, floors=(-0.15, -0.40),
                                        primary=-0.15, nonviable_ticker=None)
    fp2 = str(tmp_path / "r_all_viable.md")
    ss.write_report(train, inp_all_viable["selection_matrix"],
                    inp_all_viable["cash_lift"],
                    inp_all_viable["static_ladders"],
                    inp_all_viable["cost_sens"], inp_all_viable["holdout"],
                    inp_all_viable["block126"], [], fp2, excess_floor=-0.15,
                    selection_floors=inp_all_viable["selection_floors"],
                    provenance=None)
    n_viable_baseline = (tmp_path / "r_all_viable.md").read_text(
        encoding="utf-8").lower().count("no viable config at this floor.")
    assert n_nonviable > n_viable_baseline
    # SPY's own record row (not just the surrounding prose) carries the
    # phrase -- pin it to that specific data row.
    assert "| SPY | False |" in text
    assert "no viable config at this floor." in \
        text.split("| SPY | False |")[1].split("\n")[0]


def test_report_cash_lift_section_is_bounded_topn_no_never(tmp_path):
    # cash_lift_table emits ONE row per grid row -- 1080/ticker on the full
    # grid, including 360 identically-zero `never`-arm rows (never vs its OWN
    # never = 0 delta by construction). Dumping the WHOLE per-ticker frame via
    # _df_md would bury the decision content under over a thousand
    # ten-column rows per ticker, so report section 5 must render a BOUNDED
    # top-N summary: drop the `never` rows, keep only the top-N gap-arm rows
    # ranked by d_expected_lw_ann (most log-wealth-improving cash arm first).
    import strategy_sweep as ss
    train = {tk: _report_train_fixture(True) for tk in ("QQQ", "SOX", "SPY")}
    inp = _report_new_inputs(train)

    N_NEVER = 6
    N_GAP = 40                              # >> ss._CASH_LIFT_TOP_N (15)
    BEST_ENTER_TH = 0.9999                  # sentinel id for the top row
    rows = []
    for i in range(N_NEVER):
        rows.append({"enter_th": 0.55 + 0.0001 * i, "exit_th": 0.30,
                     "min_dwell": 0, "overlay": True, "refit_every": 21,
                     "cash_arm": "never", "d_expected_lw_ann": 0.0,
                     "d_cvar5_excess_dd": 0.0, "cash_occupancy": 0.0,
                     "cash_zero_use_freq": 1.0})
    deltas = [round(0.30 - 0.01 * i, 4) for i in range(N_GAP)]  # 0.30..-0.09
    best_i = N_GAP // 2                     # winner mid-list, not an endpoint
    deltas[best_i] = 0.99                   # unambiguous max
    for i in range(N_GAP):
        et = BEST_ENTER_TH if i == best_i else round(0.60 + 0.0001 * i, 6)
        rows.append({"enter_th": et, "exit_th": 0.35, "min_dwell": 0,
                     "overlay": True, "refit_every": 21,
                     "cash_arm": "gap_10" if i % 2 == 0 else "gap_20",
                     "d_expected_lw_ann": float(deltas[i]),
                     "d_cvar5_excess_dd": float(-deltas[i] / 2),
                     "cash_occupancy": 0.1, "cash_zero_use_freq": 0.4})
    inp["cash_lift"]["QQQ"] = pd.DataFrame(rows)

    fp = str(tmp_path / "report.md")
    ss.write_report(train, inp["selection_matrix"], inp["cash_lift"],
                    inp["static_ladders"], inp["cost_sens"], inp["holdout"],
                    inp["block126"], [], fp, excess_floor=-0.15,
                    selection_floors=inp["selection_floors"])
    text = open(fp, encoding="utf-8").read()

    sec5 = text.split("## 5.")[1].split("## 6.")[0]
    qqq_sec = sec5.split("### QQQ")[1].split("### SOX")[0]
    sec_lines = qqq_sec.splitlines()
    hdr_idx = next(i for i, ln in enumerate(sec_lines)
                  if "cash_zero_use_freq" in ln)
    data_lines = [ln for ln in sec_lines[hdr_idx + 2:] if ln.startswith("| ")]
    n_rows = len(data_lines)

    # bounded: at most the cap, never the full 46 rows fed in
    assert n_rows <= 15, f"expected <=15 rendered cash-lift rows, got {n_rows}"
    # never-arm rows dropped entirely (their delta is trivially 0 vs itself)
    assert "never" not in qqq_sec
    # the single highest-d_expected_lw_ann gap row stays after the cut
    assert f"{BEST_ENTER_TH:.4f}" in qqq_sec
    # the note under the header states this is a bounded top-N summary
    assert "top" in sec5.lower() and "15" in sec5


def _grid_slice_rows(lw_fn, feas_fn):
    # one overlay/refit/dwell slice of the real grid, valued per (enter, exit),
    # on the never arm only (Stage-1-equivalent selection surface)
    from strategy_sweep import grid_points
    rows = []
    for p in grid_points():
        if not (p["overlay"] and p["refit_every"] == 21 and p["min_dwell"] == 0
                and p["cash_arm"] == "never"):
            continue
        rows.append({**p, "expected_lw_ann": lw_fn(p), "feasible": feas_fn(p)})
    return pd.DataFrame(rows).reset_index(drop=True)


def test_globally_infeasible_neighbor_cannot_rescue_a_plateau():
    # Neighbor N=(0.60,0.35) has lw 9.9 and is feasible on ticker A but
    # infeasible on ticker B. Center C=(0.55,0.30). The C ("ci") assertion
    # below is regression cover, not discriminating: the median is robust
    # to this single outlier under exclusion, (-inf)-inclusion, or any
    # other treatment of N, so it cannot tell those semantics apart. The
    # discriminating assertions are the "ni" ones below -- the globally-
    # infeasible row N must score -inf on EVERY ticker, including A where
    # it was locally feasible. Exclusion-vs-inclusion semantics for a
    # neighbor that is infeasible on some OTHER ticker are pinned by
    # test_neighborhood_score_excludes_infeasible_neighbors (3-of-4-
    # infeasible fixture) via its feasible= extension.
    from strategy_sweep import select_plateau
    def lw(p):
        return 9.9 if (p["enter_th"], p["exit_th"]) == (0.60, 0.35) else 0.10
    a = _grid_slice_rows(lw, lambda p: True)
    b = _grid_slice_rows(lw, lambda p: (p["enter_th"], p["exit_th"]) != (0.60, 0.35))
    out = select_plateau({"A": a, "B": b})
    ci = a[(a["enter_th"] == 0.55) & (a["exit_th"] == 0.30)].index[0]
    assert out["scores_by_ticker"]["A"].loc[ci] == pytest.approx(0.10)
    # the globally-infeasible row itself scores -inf on every ticker
    ni = a[(a["enter_th"] == 0.60) & (a["exit_th"] == 0.35)].index[0]
    assert out["scores_by_ticker"]["A"].loc[ni] == -np.inf
    assert out["scores_by_ticker"]["B"].loc[ni] == -np.inf


def test_disjoint_per_ticker_feasibility_is_globally_zero_feasible():
    # A feasible only at low enter_th, B only at high: nonempty per-ticker
    # sets, EMPTY intersection -> sum(feasible_all)==0, every global_score
    # -inf, deterministic lean-cautious fallback corner chosen.
    from strategy_sweep import select_plateau
    a = _grid_slice_rows(lambda p: 0.1, lambda p: p["enter_th"] <= 0.65)
    b = _grid_slice_rows(lambda p: 0.1, lambda p: p["enter_th"] >= 0.70)
    assert a["feasible"].any() and b["feasible"].any()
    out = select_plateau({"A": a, "B": b})
    assert int(out["merged"]["feasible_all"].sum()) == 0
    assert (out["merged"]["global_score"] == -np.inf).all()
    c = out["chosen"]
    assert (c["enter_th"], c["exit_th"]) == (0.80, 0.50)   # tie-break corner


def test_select_plateau_tolerates_reordered_frames():
    # select_plateau joins tickers on the parameter KEY, not by positional
    # alignment: a shuffled-but-equivalent ticker frame must select the
    # same config.
    from strategy_sweep import select_plateau
    a = _grid_slice_rows(lambda p: p["enter_th"], lambda p: True)
    b = a.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out1 = select_plateau({"A": a, "B": a.copy()})
    out2 = select_plateau({"A": a, "B": b})
    assert out1["chosen"] == out2["chosen"]
    assert int(out1["merged"]["feasible_all"].sum()) == \
           int(out2["merged"]["feasible_all"].sum())


def test_select_plateau_exports_merged_and_scores():
    from strategy_sweep import select_plateau
    a = _grid_slice_rows(lambda p: 0.1, lambda p: True)
    b = _grid_slice_rows(lambda p: 0.2, lambda p: True)
    out = select_plateau({"A": a, "B": b})
    assert set(out) >= {"chosen", "table", "merged", "scores_by_ticker"}
    assert len(out["merged"]) == len(a)
    assert list(out["scores_by_ticker"]) == ["A", "B"]
    assert len(out["scores_by_ticker"]["A"]) == len(a)
    # a fully-feasible run: no -inf anywhere in the merged global score
    assert np.isfinite(out["merged"]["global_score"]).all()


# --- block-126 sensitivity over the (floor, ticker) selection
# matrix. Selection is PER TICKER (select_per_ticker), evaluated at
# MULTIPLE floors (the selection_floors union) -- so _block126_
# sensitivity's contract is "for each viable (floor, ticker) selection,
# locate its rank within that TICKER's own block-126 ranking", with the
# full 1080-point grid
# resweept ONCE per ticker (never once per floor) and re-masked per floor.
#
# selection_matrix shape (run_selection_matrix output):
# {floor: {ticker: record}}, one
# select_per_ticker()-shaped record per (floor, ticker) cell -- {"config":
# {..._PARAM_COLS}, "viable": bool, "feasible_count": int,
# "selection_mode": str, "score": float}. Only records with viable=True
# contribute an entry to _block126_sensitivity's output.
#
# masked_by_ticker shape (matches main()'s own _mask_train_era(joint_by_
# refit) output and _sweep_ticker_seeded's own `joint_by_refit` argument):
# {ticker: {refit_every: train-era-masked joint frame}}.

def test_block126_sensitivity_ranks_and_resweeps_once_per_ticker(monkeypatch):
    """Checks: (1) rank_block126 present for each viable
    (floor, ticker) selection; (2) block-63 and block-126 genuinely
    DISAGREE on the winner (cfg_B -- the config `selection_matrix` claims
    was picked -- is NOT the block-126 top rank); (3) the resweep is
    PER-TICKER ONCE, not per (ticker, floor) -- the KEY efficiency
    contract. `_sweep_ticker_seeded` is replaced by a counting stub
    returning a fully controlled 2-config-per-ticker block-126 table, so
    the block-63/126 disagreement and the call count are both pinned
    exactly rather than left to incidental bootstrap-draw luck."""
    import strategy_sweep as ss

    base = dict(exit_th=0.40, min_dwell=0, overlay=True, refit_every=21,
               cash_arm="never")
    cfg_A = {**base, "enter_th": 0.55}   # genuinely wins under block-126
    cfg_B = {**base, "enter_th": 0.80}   # block-63's pick -- LOSES under 126
    # enter_th grid step is 0.05 -> neighborhood radius 0.075; |0.80-0.55|
    # = 0.25 is well outside it, so each row scores its own expected_lw_ann
    # (no neighbor rescue/drag), making the A > B ordering unambiguous.

    calls = {"n": 0, "tickers": []}

    def counting(ticker, joint_by_refit, n_boot=0,
                excess_floor=ss.EXCESS_FLOOR, block_len=None, seed=0,
                train_end=ss.TRAIN_END, on_leverage=3.0):
        assert block_len == 126 and seed == 0 and on_leverage == 3.0
        calls["n"] += 1
        calls["tickers"].append(ticker)
        rows = [{**cfg_A, "expected_lw_ann": 0.30, "cvar5_excess_dd": -0.05},
                {**cfg_B, "expected_lw_ann": 0.10, "cvar5_excess_dd": -0.05}]
        return pd.DataFrame(rows)
    monkeypatch.setattr(ss, "_sweep_ticker_seeded", counting)

    masked_by_ticker = {"QQQ": {21: pd.DataFrame()}, "SOX": {21: pd.DataFrame()}}
    # both floors' "selection" is cfg_B at both tickers -- as if a block-63
    # sweep had picked it (rank 1 there, by construction of this fixture).
    selection_matrix = {
        -0.15: {"QQQ": {"config": cfg_B, "viable": True},
               "SOX": {"config": cfg_B, "viable": True}},
        -0.40: {"QQQ": {"config": cfg_B, "viable": True},
               "SOX": {"config": cfg_B, "viable": True}},
    }

    out = ss._block126_sensitivity(selection_matrix, masked_by_ticker,
                                   floors=[-0.15, -0.40], n_boot=50)

    # (3) once per ticker, NOT once per (ticker, floor): 2 tickers x 2
    # floors -- a per-floor-resweep regression would call this 4 times.
    assert calls["n"] == 2
    assert sorted(calls["tickers"]) == ["QQQ", "SOX"]

    # (1) + (2): rank_block126 present, and cfg_B (block-63's claimed
    # winner) is NOT block-126's top (cfg_A -- never present in
    # selection_matrix -- outranks it) at every (floor, ticker) cell.
    for floor in (-0.15, -0.40):
        for tk in ("QQQ", "SOX"):
            rec = out[floor][tk]
            assert "rank_block126" in rec
            assert rec["rank_block126"] == 2
            assert rec["enter_th"] == pytest.approx(0.80)   # still cfg_B


def test_block126_sensitivity_is_a_full_grid_resweep_once_per_ticker(monkeypatch):
    # Full-grid discriminator (not a top-5-only check): the 126-bar
    # sensitivity must (a) evaluate the WHOLE grid via the real
    # evaluate_point/bootstrap_indices pipeline (not a reduced set), (b)
    # draw with block_len=126, (c) do so ONCE PER TICKER regardless of how
    # many floors selection_matrix covers -- the multi-floor matrix
    # must not multiply the (expensive) bootstrap cost by len(floors).
    import strategy_sweep as ss
    joint = _toy_joint(T=300, start="2010-01-04")
    small_grid = [dict(enter_th=e, exit_th=0.40, min_dwell=0, overlay=True,
                       refit_every=21, cash_arm="never")
                  for e in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)]
    monkeypatch.setattr(ss, "grid_points", lambda: small_grid)

    calls = {"n": 0, "block_lens": set()}
    real_ep = ss.evaluate_point
    def spy_ep(arrays, idx, *a, **k):
        calls["n"] += 1
        return real_ep(arrays, idx, *a, **k)
    monkeypatch.setattr(ss, "evaluate_point", spy_ep)
    real_bi = ss.bootstrap_indices
    def spy_bi(T, n_boot, block_len=63, seed=0):
        calls["block_lens"].add(block_len)
        return real_bi(T, n_boot, block_len=block_len, seed=seed)
    monkeypatch.setattr(ss, "bootstrap_indices", spy_bi)

    by_refit = {21: joint, 63: joint}
    res = {tk: ss.sweep_ticker(tk, by_refit, n_boot=30, excess_floor=-10.0)
           for tk in ("A", "B")}
    masked = {tk: ss._mask_train_era(by_refit) for tk in ("A", "B")}
    selection_matrix = {
        floor: {tk: ss.select_per_ticker(res[tk], floor) for tk in ("A", "B")}
        for floor in (-10.0, -5.0)
    }
    calls["n"] = 0
    calls["block_lens"] = set()
    out = ss._block126_sensitivity(selection_matrix, masked,
                                   floors=[-10.0, -5.0], n_boot=30)
    # full grid x 2 tickers, NOT x 2 floors (a per-floor-resweep bug would
    # double calls["n"] to 24 and re-draw block_len=126 twice per ticker).
    assert calls["n"] == len(small_grid) * 2
    assert 126 in calls["block_lens"]
    for floor in (-10.0, -5.0):
        for tk in ("A", "B"):
            assert "rank_block126" in out[floor][tk]
            assert out[floor][tk]["rank_block126"] >= 1


def test_block126_ranking_uses_masked_neighborhood_not_raw_score(monkeypatch):
    # Masked-vs-raw discriminator for
    # PER-TICKER ranking: proves the 126-bar ranking is driven by the
    # per-ticker feasible mask + neighborhood-median statistic (the same
    # rule select_per_ticker uses for primary selection), not a raw
    # expected_lw_ann sort. Sentinel all-zero idx marks 126-draws; a
    # crafted evaluate_point flips winner/feasibility between block
    # lengths. Under 126 draws: 0.55 is infeasible, 0.60 is a raw-score
    # SPIKE (5.0), 0.65 is modest (1.0) -- raw scoring would rank the 0.60
    # spike first; masked neighborhood medians tie 0.60/0.65 at 3.0 and
    # the lean-cautious tie-break picks 0.65, so 0.55 (block-63's own
    # winner, selected below) ranks LAST, not merely "not first".
    #
    # Also pins two further properties:
    #  (A) score_block126's VALUE is exercised, not just its presence -- a
    #      regression hardcoding score_block126 = -inf for feasible rows
    #      would still pass every OTHER assertion here (rank_block126 is
    #      derived from an internal `_score` column, never from the
    #      returned score_block126 field itself), so pin the known
    #      neighborhood-median value (3.0, per the tie above) for both
    #      feasible probe rows and require it to be finite.
    #  (B) a SECOND, tighter floor (0.0) is layered on top of the first
    #      -0.15 floor for the SAME "B"/"C" probe configs. Under floor=0.0,
    #      cvar5_excess_dd=-0.01 for both no longer clears the bar, so both
    #      must flip to infeasible. A frozen-mask regression that computes
    #      the feasible mask once at floors[0] (-0.15) and reuses it for
    #      every later floor would still report them feasible at 0.0 --
    #      this catches that.
    import strategy_sweep as ss
    joint = _toy_joint(T=300, start="2010-01-04")
    small_grid = [dict(enter_th=e, exit_th=0.40, min_dwell=0, overlay=True,
                       refit_every=21, cash_arm="never")
                  for e in (0.55, 0.60, 0.65)]
    monkeypatch.setattr(ss, "grid_points", lambda: small_grid)
    real_bi = ss.bootstrap_indices
    def spy_bi(T, n_boot, block_len=63, seed=0):
        idx = real_bi(T, n_boot, block_len=block_len, seed=seed)
        return np.zeros_like(idx) if block_len == 126 else idx
    monkeypatch.setattr(ss, "bootstrap_indices", spy_bi)
    def fake_ep(arrays, idx, enter_th, *a, **k):
        is126 = bool((np.asarray(idx) == 0).all())
        out = {"cvar5_maxdd": -0.3, "cvar5_lw_ann": 0.0,
               "median_terminal_ratio_vs_1x": 1.0, "realized_lw_ann": 0.0,
               "realized_maxdd": -0.1, "realized_calmar": 0.1,
               "switches_realized": 0, "excess_tail_k": 1,
               "excess_tail_worst": -0.01,
               "excess_tail_median_maxdd_strategy": -0.1,
               "excess_tail_worst_maxdd_strategy": -0.1,
               "excess_tail_median_maxdd_1x": -0.1,
               "excess_tail_worst_maxdd_1x": -0.1}
        if is126:
            # 0.55 infeasible; 0.60 a raw-score SPIKE (5.0); 0.65 modest.
            # Neighborhood medians (0.55 excluded): 0.60 -> median(5,1)=3.0,
            # 0.65 -> median(5,1)=3.0 -- a TIE broken lean-cautious to 0.65.
            out["cvar5_excess_dd"] = -9.0 if enter_th == 0.55 else -0.01
            out["expected_lw_ann"] = 5.0 if enter_th == 0.60 else 1.0
        else:       # under 63 draws: all feasible, 0.55 the winner
            out["cvar5_excess_dd"] = -0.01
            out["expected_lw_ann"] = 1.0 - float(enter_th)
        return out
    monkeypatch.setattr(ss, "evaluate_point", fake_ep)
    by_refit = {21: joint, 63: joint}
    res_a = ss.sweep_ticker("A", by_refit, n_boot=10, excess_floor=-0.15)
    sel_a = ss.select_per_ticker(res_a, -0.15)
    assert sel_a["config"]["enter_th"] == pytest.approx(0.55)   # 63 winner
    assert sel_a["viable"]
    # "B"/"C" are probes: hand-set selections pointing at the 0.60 spike and
    # the 0.65 masked winner respectively, so their block-126 ranks are also
    # observable (the real selection under floor=-0.15 always picks 0.55 --
    # these two entries are not real select_per_ticker output, only a way
    # to read out other rows' ranks from the (floor, ticker)-keyed result).
    selection_matrix = {
        -0.15: {
            "A": sel_a,
            "B": {"config": dict(small_grid[1]), "viable": True},
            "C": {"config": dict(small_grid[2]), "viable": True},
        },
        # (B) tighter floor, same probe configs -- cvar5_excess_dd=-0.01 for
        # both clears -0.15 but not 0.0, so a CORRECT per-floor mask flips
        # both to infeasible here while -0.15 (evaluated first, above)
        # keeps them feasible.
        0.0: {
            "B": {"config": dict(small_grid[1]), "viable": True},
            "C": {"config": dict(small_grid[2]), "viable": True},
        },
    }
    masked = {tk: ss._mask_train_era(by_refit) for tk in ("A", "B", "C")}
    out = ss._block126_sensitivity(selection_matrix, masked,
                                   floors=[-0.15, 0.0], n_boot=10)
    assert not out[-0.15]["A"]["feasible_block126"]      # 126 join flips it
    assert out[-0.15]["A"]["rank_block126"] == 3          # 63 winner ranks LAST
    assert out[-0.15]["C"]["rank_block126"] == 1          # 0.65 masked winner
    assert out[-0.15]["B"]["rank_block126"] == 2          # 0.60 raw spike loses tie

    # (A) pin score_block126's actual value for both feasible probes, not
    # just its presence -- a hardcoded -inf would leave every assertion
    # above untouched (they all read rank_block126 / feasible_block126).
    assert out[-0.15]["B"]["feasible_block126"]
    assert out[-0.15]["C"]["feasible_block126"]
    assert np.isfinite(out[-0.15]["B"]["score_block126"])
    assert np.isfinite(out[-0.15]["C"]["score_block126"])
    assert out[-0.15]["B"]["score_block126"] == pytest.approx(3.0)
    assert out[-0.15]["C"]["score_block126"] == pytest.approx(3.0)

    # (B) same "B"/"C" configs, tighter floor=0.0 -- correct per-floor
    # masking must flip both to infeasible; a frozen mask reusing floors[0]
    # (-0.15)'s feasibility would incorrectly leave both True here.
    assert not out[0.0]["B"]["feasible_block126"]
    assert not out[0.0]["C"]["feasible_block126"]


def test_df_md_renders_nan_rank_blank():
    # Pins the _df_md contract the block-126 union rows rely on: a NaN
    # rank_block63 renders as an empty cell, not "nan".
    from strategy_sweep import _df_md
    md = _df_md(pd.DataFrame({"rank_block63": [np.nan, 2.0]}))
    assert "|  |" in md and "nan" not in md


def test_grid_1080_valid_and_default_floor():
    from strategy_sweep import grid_points, cash_th_of, _PARAM_COLS, EXCESS_FLOOR
    pts = grid_points()
    assert len(pts) == 1080
    assert "cash_arm" in _PARAM_COLS
    assert EXCESS_FLOOR == -0.40
    for p in pts:
        cth = cash_th_of(p)
        if p["cash_arm"] == "never":
            assert cth == float("-inf")
        else:
            assert 0.0 < cth < p["exit_th"]


def test_never_arm_is_360():
    from strategy_sweep import grid_points
    assert len([p for p in grid_points() if p["cash_arm"] == "never"]) == 360


# --- select_per_ticker + cash-arm-fixed neighborhood plateaus ---

def _row(cash_arm, enter, elw, cvar, **kw):
    r = dict(enter_th=enter, exit_th=0.40, min_dwell=0, overlay=True,
             refit_every=21, cash_arm=cash_arm, expected_lw_ann=elw,
             cvar5_excess_dd=cvar)
    r.update(kw)
    return r


def test_neighborhood_excludes_other_cash_arm_neighbor():
    # gap_10 center feasible, its ONLY geometric neighbor (enter +0.05) is a
    # FEASIBLE 'never' arm with a huge elw. Cash-arm-fixed => center scores
    # its own 0.20, NOT the 0.595 avg a non-fixed filter would produce.
    from strategy_sweep import neighborhood_score
    rows = [_row("gap_10", 0.60, 0.20, -0.10),
            _row("never", 0.65, 0.99, -0.10)]
    df = pd.DataFrame(rows)
    s = neighborhood_score(df, feasible=np.array([True, True]))
    assert abs(s.iloc[0] - 0.20) < 1e-9      # would be 0.595 if arm not fixed


def test_zero_feasible_non_viable():
    from strategy_sweep import select_per_ticker
    df = pd.DataFrame([_row("never", 0.6, 0.2, -0.9)])
    sel = select_per_ticker(df, floor=-0.15)
    assert sel["viable"] is False and sel["feasible_count"] == 0


def test_tiebreak_prefers_gap20_over_gap10():
    from strategy_sweep import select_per_ticker
    df = pd.DataFrame([_row("gap_10", 0.6, 0.2, -0.1),
                       _row("gap_20", 0.6, 0.2, -0.1)])
    assert select_per_ticker(df, floor=-0.15)["config"]["cash_arm"] == "gap_20"


def test_stability_is_pure_over_cached_tables():
    from strategy_sweep import feasibility_stability, STABILITY_MIN
    # 2 grid points; point A feasible-with-margin on all 5 seeds, point B on 3.
    tables = []
    for s in range(5):
        tables.append(pd.DataFrame({
            "cvar5_excess_dd": [-0.10, (-0.10 if s < 3 else -0.90)]}))
    stable = feasibility_stability(tables, floor=-0.15, m=STABILITY_MIN,
                                   margin=0.0)
    assert stable.tolist() == [True, False]   # A:5>=4 True; B:3<4 False


def test_stability_margin_applies():
    from strategy_sweep import feasibility_stability
    tables = [pd.DataFrame({"cvar5_excess_dd": [-0.14]}) for _ in range(5)]
    # -0.14 >= -0.15 (raw) but NOT >= -0.15+0.02=-0.13 -> fails with margin
    assert feasibility_stability(tables, floor=-0.15, m=1,
                                 margin=0.02).tolist() == [False]


# --- CRN-paired cash-lift table ---

def test_cash_lift_same_config_paired():
    from strategy_sweep import cash_lift_table
    base = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
                refit_every=21)
    rows = [{**base, "cash_arm": "never", "expected_lw_ann": 0.20,
             "cvar5_excess_dd": -0.40, "cash_occupancy": 0.0,
             "cash_zero_use_freq": 1.0},
            {**base, "cash_arm": "gap_10", "expected_lw_ann": 0.15,
             "cvar5_excess_dd": -0.20, "cash_occupancy": 0.3,
             "cash_zero_use_freq": 0.4}]
    t = cash_lift_table(pd.DataFrame(rows))
    g = t[t["cash_arm"] == "gap_10"].iloc[0]
    assert abs(g["d_expected_lw_ann"] + 0.05) < 1e-9
    assert abs(g["d_cvar5_excess_dd"] - 0.20) < 1e-9
    assert abs(g["cash_zero_use_freq"] - 0.4) < 1e-9


def test_cash_lift_discriminates_pairing_across_two_configs():
    """A single-base-config fixture can't tell correct CRN
    pairing (each cash arm vs its OWN base config's 'never' row) apart from
    a WRONG implementation that uses a global mean of all 'never' rows, or
    the other config's 'never' row -- both would happen to pass a
    one-config fixture identically. Two base configs (differ only in
    enter_th), each with its own never + gap_10 pair carrying DISTINCT
    expected_lw_ann/cvar5_excess_dd, pin numbers that a wrong baseline
    cannot reproduce."""
    from strategy_sweep import cash_lift_table
    base_a = dict(enter_th=0.6, exit_th=0.4, min_dwell=0, overlay=True,
                  refit_every=21)
    base_b = dict(enter_th=0.7, exit_th=0.4, min_dwell=0, overlay=True,
                  refit_every=21)
    rows = [
        {**base_a, "cash_arm": "never", "expected_lw_ann": 0.20,
         "cvar5_excess_dd": -0.40, "cash_occupancy": 0.0,
         "cash_zero_use_freq": 1.0},
        {**base_a, "cash_arm": "gap_10", "expected_lw_ann": 0.15,
         "cvar5_excess_dd": -0.20, "cash_occupancy": 0.3,
         "cash_zero_use_freq": 0.4},
        {**base_b, "cash_arm": "never", "expected_lw_ann": 0.50,
         "cvar5_excess_dd": -0.10, "cash_occupancy": 0.0,
         "cash_zero_use_freq": 1.0},
        {**base_b, "cash_arm": "gap_10", "expected_lw_ann": 0.30,
         "cvar5_excess_dd": -0.05, "cash_occupancy": 0.5,
         "cash_zero_use_freq": 0.2},
    ]
    t = cash_lift_table(pd.DataFrame(rows))

    # a 'never' row itself yields deltas of exactly 0 -- for BOTH base
    # configs (not just the one the single-config test exercises).
    never_a = t[(t["cash_arm"] == "never") & (t["enter_th"] == 0.6)].iloc[0]
    never_b = t[(t["cash_arm"] == "never") & (t["enter_th"] == 0.7)].iloc[0]
    assert abs(never_a["d_expected_lw_ann"]) < 1e-9
    assert abs(never_a["d_cvar5_excess_dd"]) < 1e-9
    assert abs(never_b["d_expected_lw_ann"]) < 1e-9
    assert abs(never_b["d_cvar5_excess_dd"]) < 1e-9

    # correct same-config CRN pairing: config A's and config B's gap_10
    # deltas resolve against their OWN never row and land on DIFFERENT
    # pinned numbers -- a global-mean-of-never (or cross-config) baseline
    # would produce different numbers here and fail these asserts.
    gap_a = t[(t["cash_arm"] == "gap_10") & (t["enter_th"] == 0.6)].iloc[0]
    gap_b = t[(t["cash_arm"] == "gap_10") & (t["enter_th"] == 0.7)].iloc[0]
    assert abs(gap_a["d_expected_lw_ann"] - (-0.05)) < 1e-9
    assert abs(gap_a["d_cvar5_excess_dd"] - 0.20) < 1e-9
    assert abs(gap_b["d_expected_lw_ann"] - (-0.20)) < 1e-9
    assert abs(gap_b["d_cvar5_excess_dd"] - 0.05) < 1e-9


# --- cost-convention sensitivity (risk-notional vs dollar-turnover) ---

def test_cost_sensitivity_reports_both_models():
    # Minimal 1-ticker, 1-refit_every fixture whose chosen config ping-pongs
    # 1x<->3x on every bar (T=4): p_bull=[0.9,0.3,0.9,0.3] with
    # enter_th=0.6/exit_th=0.4/overlay=False crosses every threshold every
    # bar (same crossing shape as
    # test_dollar_turnover_flat_per_rung_and_prices_1x3x_at_one_side in
    # test_gate_engine.py). ALL lr legs are zero, so every basis point of
    # the metric difference between the two cost models is attributable
    # PURELY to the cost convention -- not to any lr noise -- making the
    # nonzero delta an unambiguous discriminator, not a coincidence of a
    # random fixture.
    from strategy_sweep import cost_sensitivity
    T = 4
    dates = pd.bdate_range("2010-01-04", periods=T)
    z = np.zeros(T)
    df = pd.DataFrame({"Date": dates.date,
                       "p_bull": np.array([0.9, 0.3, 0.9, 0.3]),
                       "trail_ratio": np.ones(T),      # ignored, overlay=False
                       "lr_1x": z, "lr_3x": z, "lr_cash": z})
    joint_by_refit_by_ticker = {"QQQ": {21: df}}
    chosen_by_ticker = {"QQQ": dict(enter_th=0.6, exit_th=0.4, min_dwell=0,
                                    overlay=False, refit_every=21,
                                    cash_arm="never")}
    # identity resample (idx = chronological order, single path): the
    # "bootstrap" leg IS the realized path, so the result is exact and
    # deterministic, not subject to resample noise.
    idx_context = {"QQQ": np.array([[0, 1, 2, 3]])}

    out = cost_sensitivity(joint_by_refit_by_ticker, chosen_by_ticker,
                           idx_context)
    assert {"ticker", "expected_lw_ann_risk_notional",
            "expected_lw_ann_dollar_turnover", "d_expected_lw_ann",
            "cvar5_excess_dd_risk_notional", "cvar5_excess_dd_dollar_turnover",
            "d_cvar5_excess_dd"} <= set(out.columns)
    assert len(out) == 1
    row = out[out["ticker"] == "QQQ"].iloc[0]

    # Hand-computed reference: exposure_states(p, ..., 0.6, 0.4, overlay=False,
    # cash_th=-inf) on p=[0.9,0.3,0.9,0.3] gives decisions [2,1,2,1] (up3,
    # drop3, up3, drop3 firing every bar), so positions HELD are [1,2,1,2]
    # (start 1x, per strategy_log_returns' own pos=concat([1],e[:-1])
    # formula) -- 3 executed 1x<->3x switches. risk_notional prices each at
    # (4/2)*|dLev|/1e4 = (4/2)*2/1e4 = 4bps (12bps total, switch_cost_bps=4
    # default); dollar_turnover prices each executed hop at a flat one-side
    # (4/2)/1e4 = 2bps (6bps total) -- HALF the risk_notional cost, since it
    # prices the executed hop, not its |Delta leverage|.
    expected_total_cost_rn = 3 * 4.0 / 1e4
    expected_total_cost_dt = 3 * 2.0 / 1e4
    assert expected_total_cost_dt == pytest.approx(expected_total_cost_rn / 2.0)
    expected_ann_rn = (-expected_total_cost_rn) * 252.0 / T
    expected_ann_dt = (-expected_total_cost_dt) * 252.0 / T
    assert row["expected_lw_ann_risk_notional"] == pytest.approx(expected_ann_rn)
    assert row["expected_lw_ann_dollar_turnover"] == pytest.approx(expected_ann_dt)

    # THE discriminating assertions: nonzero deltas for a switching config --
    # the whole point of the cost sensitivity is that the cost convention is
    # not load-bearing for the SELECTION but IS numerically consequential.
    assert row["d_expected_lw_ann"] != 0.0
    assert row["d_expected_lw_ann"] == pytest.approx(expected_ann_dt - expected_ann_rn)
    assert row["d_cvar5_excess_dd"] != 0.0
    # dollar_turnover charges HALF the risk_notional cost here, so its
    # drawdown-vs-1x excess is shallower (less negative / larger) -- the
    # sign the cost framing predicts.
    assert row["cvar5_excess_dd_dollar_turnover"] > row["cvar5_excess_dd_risk_notional"]


# --- static-L ladder table (per-config frame/CRN, gross + cost) ---

def test_static_ladder_table_crn_paired_gross_and_cost():
    """Discriminates all 4 points:
    (1) L=1.0 has cvar5_excess_dd==0 exactly -- base_dd built from THIS
        frame's own lr_1x resampled by the SAME idx passed in, so
        static_leverage_lr(...,1.0)==lr_1x and maxdd_b - base_dd is 0.
    (2) the table genuinely consumes the PASSED idx/base_dd rather than a
        hardcoded/recomputed refit: (2a) two distinct (idx, matched
        base_dd) draws give distinct L=1 expected_lw_ann -- idx is live,
        not ignored; (2b) pairing idx_a with the MISMATCHED base_dd_b
        breaks the L=1 zero-excess property -- an implementation that
        silently recomputed base_dd from idx internally (ignoring the
        passed base_dd) would still read exactly 0 here and get caught.
    (3) L=3.0's cvar5_maxdd is deeper (more negative) than L=1.0's under
        the SAME idx -- leverage deepens drawdowns.
    (4) endpoints L in {0,1,3} are pure single legs: expected_lw_ann_costed
        == expected_lw_ann exactly (a single holding never drifts off its
        own 100% weight, so the daily-rebalance turnover charge is 0 bit-
        for-bit). An intermediate L (1.5, which blends the 1x and 3x legs)
        has expected_lw_ann_costed strictly < expected_lw_ann:
        two legs with genuinely different daily returns drifting apart
        under a constant-weight target is never free to rebalance back.
    """
    from strategy_sweep import static_ladder_table
    import strategy_sweep as ss
    from objective import max_drawdown_from_lr

    joint = _mini_joint()
    frame = joint["QQQ"][21]           # ONE ticker, ONE refit_every, ONE era
    T = len(frame)

    idx_a = ss.bootstrap_indices(T, n_boot=200, block_len=63, seed=0)
    idx_b = ss.bootstrap_indices(T, n_boot=200, block_len=63, seed=1)
    base_dd_a = max_drawdown_from_lr(frame["lr_1x"].values[idx_a])
    base_dd_b = max_drawdown_from_lr(frame["lr_1x"].values[idx_b])
    assert not np.allclose(base_dd_a, base_dd_b)   # draws are genuinely distinct

    ladder = (0, 1.0, 1.5, 2.0, 2.5, 3.0)
    table_a = static_ladder_table(frame, idx_a, base_dd_a, ladder=ladder)
    table_b = static_ladder_table(frame, idx_b, base_dd_b, ladder=ladder)
    assert list(table_a["L"]) == [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]

    # (1) L=1 excess is exactly 0 when idx/base_dd are correctly paired
    l1_a = table_a[table_a["L"] == 1.0].iloc[0]
    l1_b = table_b[table_b["L"] == 1.0].iloc[0]
    assert l1_a["cvar5_excess_dd"] == 0.0
    assert l1_b["cvar5_excess_dd"] == 0.0

    # (2a) the table genuinely depends on the PASSED idx (not a hardcoded
    # recompute): two distinct resample draws give distinct L=1 stats
    assert l1_a["expected_lw_ann"] != pytest.approx(l1_b["expected_lw_ann"])

    # (2b) base_dd is consumed exactly as passed, never recomputed from idx:
    # pairing idx_a with the mismatched base_dd_b breaks the L=1 zero-excess
    # property.
    mismatched = static_ladder_table(frame, idx_a, base_dd_b, ladder=ladder)
    l1_mismatch = mismatched[mismatched["L"] == 1.0].iloc[0]
    assert l1_mismatch["cvar5_excess_dd"] != 0.0

    # (3) leverage deepens drawdowns: L=3 strictly deeper (more negative)
    # cvar5_maxdd than L=1, same idx.
    l3_a = table_a[table_a["L"] == 3.0].iloc[0]
    assert l3_a["cvar5_maxdd"] < l1_a["cvar5_maxdd"]

    # (4) endpoints cost exactly 0; an intermediate L strictly costs something
    for L in (0.0, 1.0, 3.0):
        row = table_a[table_a["L"] == L].iloc[0]
        assert row["expected_lw_ann_costed"] == row["expected_lw_ann"]
    l15_a = table_a[table_a["L"] == 1.5].iloc[0]
    assert l15_a["expected_lw_ann_costed"] < l15_a["expected_lw_ann"]
