import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import importlib, pytest
import numpy as np
import pandas as pd


@pytest.fixture
def clean_abi_memo():
    """Clear the compute-ABI memo BEFORE and AFTER each test so a
    monkeypatched manifest member never poisons another test via the memo."""
    import sweep_cache as sc
    sc._ABI_MEMO.clear()
    yield
    sc._ABI_MEMO.clear()


# --- compute-ABI digest -------------------------------------------

def test_abi_digest_is_stable_within_a_process(clean_abi_memo):
    import sweep_cache as sc
    assert sc.compute_abi_digest() == sc.compute_abi_digest()


def test_abi_digest_changes_when_switch_cost_changes(clean_abi_memo, monkeypatch):
    import sweep_cache as sc, strategy_sweep as ss
    base = sc.compute_abi_digest()
    sc._ABI_MEMO.clear()
    monkeypatch.setattr(ss, "SWITCH_COST_BPS", ss.SWITCH_COST_BPS + 1.0)
    assert sc.compute_abi_digest() != base


# Every manifest CONSTANT flips the digest ----------------------------------
_CONST_MUTATIONS = [
    ("strategy_sweep", "SWITCH_COST_BPS", lambda old: old + 1.0),
    ("strategy_sweep", "BLOCK_LEN", lambda old: old + 1),
    ("strategy_sweep", "TRAIN_END", lambda old: "2013-12-31"),
    ("strategy_sweep", "CASH_GAP", lambda old: {**old, "gap_10": 0.11}),
    ("strategy_sweep", "GRID", lambda old: {**old, "min_dwell": (0, 5, 15)}),
    ("objective", "TRADING_DAYS", lambda old: old + 1),
]


@pytest.mark.parametrize("mod_name,attr,mutate", _CONST_MUTATIONS,
                         ids=[f"{m}.{a}" for m, a, _ in _CONST_MUTATIONS])
def test_abi_digest_flips_on_constant(clean_abi_memo, monkeypatch, mod_name,
                                      attr, mutate):
    import sweep_cache as sc, importlib
    mod = importlib.import_module(mod_name)
    base = sc.compute_abi_digest()
    sc._ABI_MEMO.clear()
    monkeypatch.setattr(mod, attr, mutate(getattr(mod, attr)))
    assert sc.compute_abi_digest() != base


# Every manifest FUNCTION flips the digest (source-based) --------------------
_FUNC_MEMBERS = [
    ("strategy_sweep", "_sweep_ticker_seeded"),
    ("strategy_sweep", "_mask_train_era"),
    ("strategy_sweep", "_arrays"),
    ("strategy_sweep", "cash_th_of"),
    ("strategy_sweep", "grid_points"),
    ("objective", "evaluate_point"),
    ("objective", "max_drawdown_from_lr"),
    ("objective", "terminal_log_wealth"),
    ("objective", "annualized_log_wealth"),
    ("objective", "cvar"),
    ("gate_engine", "bootstrap_indices"),
    ("gate_engine", "exposure_states"),
    ("gate_engine", "strategy_log_returns"),
]


@pytest.mark.parametrize("mod_name,attr", _FUNC_MEMBERS,
                         ids=[f"{m}.{a}" for m, a in _FUNC_MEMBERS])
def test_abi_digest_flips_on_function_source(clean_abi_memo, monkeypatch,
                                             mod_name, attr):
    import sweep_cache as sc, importlib
    mod = importlib.import_module(mod_name)
    base = sc.compute_abi_digest()
    sc._ABI_MEMO.clear()

    def _replacement(*a, **k):     # distinct source text -> digest must flip
        raise AssertionError("manifest wrapper should never be called")
    monkeypatch.setattr(mod, attr, _replacement)
    assert sc.compute_abi_digest() != base


def test_abi_digest_flips_on_leverage_bytes(clean_abi_memo, monkeypatch):
    import sweep_cache as sc, gate_engine as ge
    base = sc.compute_abi_digest()
    sc._ABI_MEMO.clear()
    monkeypatch.setattr(ge, "LEVERAGE", np.array([0.0, 1.0, 2.999]))
    assert sc.compute_abi_digest() != base


@pytest.mark.parametrize("mod_name,attr", [
    ("numpy", "__version__"), ("pandas", "__version__"),
    ("pyarrow", "__version__")])
def test_abi_digest_flips_on_dep_version(clean_abi_memo, monkeypatch, mod_name,
                                         attr):
    import sweep_cache as sc, importlib
    mod = importlib.import_module(mod_name)
    base = sc.compute_abi_digest()
    sc._ABI_MEMO.clear()
    monkeypatch.setattr(mod, attr, "0.0.0-stub")
    assert sc.compute_abi_digest() != base


def test_abi_digest_flips_on_python_version(clean_abi_memo, monkeypatch):
    import sweep_cache as sc, sys as _sys
    base = sc.compute_abi_digest()
    sc._ABI_MEMO.clear()
    monkeypatch.setattr(_sys, "version", "0.0.0-stub (fake)")
    assert sc.compute_abi_digest() != base


def test_abi_digest_getsource_fail_closed_names_member(clean_abi_memo,
                                                       monkeypatch):
    """Fail-closed: if inspect.getsource cannot read a member, the digest
    RAISES naming it, never silently drops it."""
    import sweep_cache as sc, inspect
    monkeypatch.setattr(sc.inspect, "getsource",
                        lambda obj: (_ for _ in ()).throw(OSError("no src")))
    with pytest.raises(RuntimeError, match="compute_abi_digest"):
        sc.compute_abi_digest()


# --- sweep fingerprint over post-mask numeric arrays ---------------

def _frames():
    from test_strategy_sweep import _mini_joint     # reuse the real fixture
    return _mini_joint()["QQQ"]                      # {21: frame, 63: frame}


def test_frame_fp_masking_is_idempotent_and_matters(clean_abi_memo):
    """Not tautological. (a) masking a full frame vs masking an
    already-masked frame yields the SAME fingerprint (idempotent dedup, guards
    full-vs-pre-masked key drift); (b) the UNMASKED frame's fingerprint
    DIFFERS from the masked one, proving the fingerprint actually depends on
    the rows masking removes (not a constant)."""
    import sweep_cache as sc, strategy_sweep as ss
    full = _frames()
    masked = ss._mask_train_era(full, ss.TRAIN_END)
    fp_full = sc.frame_fingerprint(ss._mask_train_era(full, ss.TRAIN_END))
    fp_pre = sc.frame_fingerprint(ss._mask_train_era(masked, ss.TRAIN_END))
    assert fp_full == fp_pre                          # (a) idempotent dedup
    assert sc.frame_fingerprint(full) != fp_full      # (b) masking matters


def test_sweep_fp_changes_with_each_keyed_field(clean_abi_memo):
    import sweep_cache as sc, strategy_sweep as ss
    m = ss._mask_train_era(_frames(), ss.TRAIN_END)
    base = sc.sweep_fingerprint(m, n_boot=200, block_len=63, seed=0,
                                train_end=ss.TRAIN_END)
    assert base != sc.sweep_fingerprint(m, n_boot=201, block_len=63, seed=0,
                                        train_end=ss.TRAIN_END)
    assert base != sc.sweep_fingerprint(m, n_boot=200, block_len=126, seed=0,
                                        train_end=ss.TRAIN_END)
    assert base != sc.sweep_fingerprint(m, n_boot=200, block_len=63, seed=1,
                                        train_end=ss.TRAIN_END)
    assert base != sc.sweep_fingerprint(m, n_boot=200, block_len=63, seed=0,
                                        train_end="2013-12-31")


def test_sweep_fp_stable_and_incorporates_abi(clean_abi_memo, monkeypatch):
    """Same inputs -> same key; a compute-ABI change (a manifest constant)
    flips the sweep key even with identical frames/params (guards a forgotten
    version bump)."""
    import sweep_cache as sc, strategy_sweep as ss
    m = ss._mask_train_era(_frames(), ss.TRAIN_END)
    base = sc.sweep_fingerprint(m, 200, 63, 0, ss.TRAIN_END)
    assert base == sc.sweep_fingerprint(m, 200, 63, 0, ss.TRAIN_END)
    sc._ABI_MEMO.clear()
    monkeypatch.setattr(ss, "SWITCH_COST_BPS", ss.SWITCH_COST_BPS + 1.0)
    assert base != sc.sweep_fingerprint(m, 200, 63, 0, ss.TRAIN_END)


def test_sweep_fp_flips_on_cache_version(clean_abi_memo, monkeypatch):
    """SWEEP_CACHE_VERSION is the manual escape hatch (outside the ABI
    manifest), so a regression dropping it from sweep_fingerprint would be
    invisible without this test -- bumping it must change the key."""
    import sweep_cache as sc, strategy_sweep as ss
    m = ss._mask_train_era(_frames(), ss.TRAIN_END)
    base = sc.sweep_fingerprint(m, 200, 63, 0, ss.TRAIN_END)
    monkeypatch.setattr(sc, "SWEEP_CACHE_VERSION", sc.SWEEP_CACHE_VERSION + 1)
    assert base != sc.sweep_fingerprint(m, 200, 63, 0, ss.TRAIN_END)


# --- atomic Parquet publish + validated load ----------------------

def _toy_metrics():
    from strategy_sweep import grid_points
    rows = [{**p, "cvar5_excess_dd": -0.3, "expected_lw_ann": 0.1}
            for p in grid_points()]
    return pd.DataFrame(rows)


def test_publish_then_load_roundtrip_exact(tmp_path):
    import sweep_cache as sc
    df = _toy_metrics()
    meta = {"fp": "abc"}
    sc.publish_sweep(str(tmp_path), "abc", df, meta)
    back = sc.load_sweep(str(tmp_path), "abc", meta)
    pd.testing.assert_frame_equal(df, back, check_exact=True)


def test_load_miss_on_absent_or_bad_checksum(tmp_path):
    import sweep_cache as sc, json, os
    df = _toy_metrics()
    meta = {"fp": "abc"}
    assert sc.load_sweep(str(tmp_path), "abc", meta) is None       # absent
    sc.publish_sweep(str(tmp_path), "abc", df, meta)
    metap = os.path.join(str(tmp_path), "sweep_abc.meta.json")
    with open(metap) as fh:
        m = json.load(fh)
    m["payload_sha256"] = "0" * 64
    with open(metap, "w") as fh:
        json.dump(m, fh)
    assert sc.load_sweep(str(tmp_path), "abc", meta) is None       # bad checksum


def test_load_miss_on_meta_mismatch(tmp_path):
    """A meta field that differs from what the caller asks for -> miss (guards
    against loading a different unit's payload under a colliding path)."""
    import sweep_cache as sc
    df = _toy_metrics()
    sc.publish_sweep(str(tmp_path), "abc", df, {"fp": "abc", "seed": 0})
    assert sc.load_sweep(str(tmp_path), "abc", {"fp": "abc", "seed": 1}) is None
    assert sc.load_sweep(str(tmp_path), "abc", {"fp": "abc", "seed": 0}) is not None


def test_load_miss_on_reordered_grid(tmp_path):
    import sweep_cache as sc
    df = _toy_metrics().iloc[::-1].reset_index(drop=True)          # reversed rows
    meta = {"fp": "abc"}
    sc.publish_sweep(str(tmp_path), "abc", df, meta)
    assert sc.load_sweep(str(tmp_path), "abc", meta) is None       # ordered-grid guard


def test_load_miss_on_half_publish_parquet_without_meta(tmp_path):
    """An orphan .parquet with no .meta.json (interrupted publish) must
    not load."""
    import sweep_cache as sc, os
    df = _toy_metrics()
    sc.publish_sweep(str(tmp_path), "abc", df, {"fp": "abc"})
    os.remove(os.path.join(str(tmp_path), "sweep_abc.meta.json"))
    assert sc.load_sweep(str(tmp_path), "abc", {"fp": "abc"}) is None


def test_load_miss_on_schema_broken_parquet(tmp_path):
    """Fail-closed: a parquet whose checksum matches
    meta but whose columns are wrong (schema change under a colliding fp) must
    MISS via the ordered-grid guard, never raise KeyError out of load_sweep."""
    import sweep_cache as sc, os, json
    sc.publish_sweep(str(tmp_path), "abc", _toy_metrics(), {"fp": "abc"})
    pq = os.path.join(str(tmp_path), "sweep_abc.parquet")
    metap = os.path.join(str(tmp_path), "sweep_abc.meta.json")
    pd.DataFrame({"unrelated": [1, 2, 3]}).to_parquet(pq, index=False)
    with open(metap) as fh:
        m = json.load(fh)
    m["payload_sha256"] = sc._sha256_file(pq)      # make the checksum gate pass
    with open(metap, "w") as fh:
        json.dump(m, fh)
    assert sc.load_sweep(str(tmp_path), "abc", {"fp": "abc"}) is None


def test_publish_is_atomic_no_orphan_tmp(tmp_path):
    """After a successful publish, only the .parquet + .meta.json remain (no
    leftover unique .tmp files)."""
    import sweep_cache as sc, os
    sc.publish_sweep(str(tmp_path), "abc", _toy_metrics(), {"fp": "abc"})
    names = os.listdir(str(tmp_path))
    assert not any(n.endswith(".tmp") for n in names)
    assert set(names) == {"sweep_abc.parquet", "sweep_abc.meta.json"}


# --- cached_sweep + SweepSpec (one source per mode, floor-indep) ---

def _frames_small():
    """Compact 1-ticker, 2-refit fixture straddling train_end. The byte-exact
    equivalence property is independent of n_boot and frame length, so a short
    frame + small n_boot proves it fast while STILL sweeping the full 1080-point
    grid (which drives the fingerprint/grid-order guards)."""
    dates = pd.bdate_range("2014-06-02", "2015-03-31")     # straddles 2014-12-31
    n = len(dates)
    rng = np.random.default_rng(7)

    def frame():
        l1 = rng.normal(0, 0.01, n)
        return pd.DataFrame({"Date": dates.date, "p_bull": rng.random(n),
                             "trail_ratio": rng.random(n) + 0.6, "lr_1x": l1,
                             "lr_3x": 3 * l1, "lr_cash": np.full(n, 1e-5)})
    return {21: frame(), 63: frame()}


def _spec(tmp, **kw):
    import sweep_cache as sc, strategy_sweep as ss
    d = dict(ticker="QQQ", block_len=63, seed=0, n_boot=120,
             train_end=ss.TRAIN_END, n_restarts=15, vol_window=63,
             warmup=1000, data_dir="data/raw", cache_dir=str(tmp))
    d.update(kw)
    return sc.SweepSpec(**d)


def test_cached_none_equals_direct_full_frame(clean_abi_memo):
    """cache_dir=None is a byte-identical passthrough to _sweep_ticker_
    seeded on the FULL frame incl feasible, at NON-default seed/block_len/
    train_end (so a dropped or mis-threaded spec field fails)."""
    import sweep_cache as sc, strategy_sweep as ss
    frames = _frames_small()
    spec = sc.SweepSpec("QQQ", block_len=42, seed=2, n_boot=48,
                        train_end="2015-01-30", n_restarts=15, vol_window=63,
                        warmup=1000, data_dir="data/raw", cache_dir=None)
    direct = ss._sweep_ticker_seeded("QQQ", frames, n_boot=48,
                                     excess_floor=-0.4, block_len=42, seed=2,
                                     train_end="2015-01-30")
    viacache, hit = sc.cached_sweep(spec, excess_floor=-0.4,
                                    joint_by_refit=frames)
    assert hit is False
    pd.testing.assert_frame_equal(direct, viacache, check_exact=True)


def test_cached_none_requires_joint():
    import sweep_cache as sc
    spec = _spec("unused", cache_dir=None)
    with pytest.raises(ValueError, match="joint_by_refit"):
        sc.cached_sweep(spec, excess_floor=-0.4, joint_by_refit=None)


def test_cached_dir_roundtrip_floor_independent_and_hit_status(
        clean_abi_memo, tmp_path, monkeypatch):
    """cache_dir set: computes+publishes on the first call (hit False), then a
    DIFFERENT excess_floor is a real HIT (hit True) with NO recompute, and the
    metric columns are byte-identical while feasible tracks each floor.
    cached_sweep returns (frame, hit)."""
    import sweep_cache as sc, strategy_sweep as ss, causal_backtest as cb
    frames = _frames_small()
    monkeypatch.setattr(cb, "build_joint_frame",
                        lambda ticker, refit, *a, **k: frames[refit].copy())
    computes = {"n": 0}
    real = ss._sweep_ticker_seeded

    def spy(*a, **k):
        computes["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(ss, "_sweep_ticker_seeded", spy)

    spec = _spec(tmp_path, n_boot=48)
    r1, hit1 = sc.cached_sweep(spec, excess_floor=-0.40)     # computes + publishes
    r2, hit2 = sc.cached_sweep(spec, excess_floor=-0.15)     # HIT, different floor
    assert hit1 is False and hit2 is True
    assert computes["n"] == 1                                # r2 did NOT recompute
    metric_cols = [c for c in r1.columns if c != "feasible"]
    pd.testing.assert_frame_equal(r1[metric_cols], r2[metric_cols],
                                  check_exact=True)
    assert (r1["feasible"] == (r1["cvar5_excess_dd"] >= -0.40)).all()
    assert (r2["feasible"] == (r2["cvar5_excess_dd"] >= -0.15)).all()


def test_cached_dir_ignores_passed_frame_uses_strict_load(
        clean_abi_memo, tmp_path, monkeypatch):
    """When cache_dir is set the frame comes SOLELY from the strict
    loader; any joint_by_refit passed in is ignored (guaranteeing an in-process
    call and a worker consume identical bytes)."""
    import sweep_cache as sc, strategy_sweep as ss, causal_backtest as cb
    frames = _frames_small()
    monkeypatch.setattr(cb, "build_joint_frame",
                        lambda ticker, refit, *a, **k: frames[refit].copy())
    spec = _spec(tmp_path, n_boot=48)
    # a bogus passed frame must NOT change the result (it is ignored)
    bogus = {21: frames[21].copy(), 63: frames[63].copy()}
    bogus[21].loc[:, "lr_1x"] = 999.0
    r_strict, _ = sc.cached_sweep(spec, excess_floor=-0.40)
    # fresh cache dir so no hit; pass bogus -> ignored -> identical metrics
    spec2 = _spec(tmp_path / "c2", n_boot=48)
    r_bogus, _ = sc.cached_sweep(spec2, excess_floor=-0.40, joint_by_refit=bogus)
    metric_cols = [c for c in r_strict.columns if c != "feasible"]
    pd.testing.assert_frame_equal(r_strict[metric_cols], r_bogus[metric_cols],
                                  check_exact=True)


# --- executor, worker, effective_workers, 18-unit enumeration ------

def test_effective_workers_clamps_and_rejects():
    import sweep_cache as sc
    assert sc.effective_workers(1, free_gb=100) >= 1
    assert sc.effective_workers(999, free_gb=100) <= 24
    with pytest.raises(ValueError):
        sc.effective_workers(0)
    with pytest.raises(ValueError):
        sc.effective_workers(None)


def test_effective_workers_memory_cap(monkeypatch):
    """memory_cap = floor((free - os_reserve) / peak_worker). free=7,
    peak=1.5, reserve=2 -> cap 3, so --workers 8 clamps to 3. The CPU count
    is pinned high so the memory cap, not the machine's cores, binds."""
    import os
    import sweep_cache as sc
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert sc.effective_workers(8, peak_worker_gb=1.5, os_reserve_gb=2.0,
                                free_gb=7.0) == 3


def test_effective_workers_cpu_none(monkeypatch):
    import sweep_cache as sc, os
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert sc.effective_workers(8, free_gb=100) == 1


def _bp(**kw):
    d = dict(n_boot=2000, train_end="2014-12-31", n_restarts=15,
             vol_window=63, warmup=1000, data_dir="data/raw")
    d.update(kw)
    return d


def test_enumerate_specs_is_exactly_18():
    import sweep_cache as sc
    specs = sc.enumerate_specs(("QQQ", "SOX", "SPY"), range(5), "cache", _bp())
    keys = {(s.ticker, s.block_len, s.seed) for s in specs}
    assert len(specs) == 18 and len(keys) == 18
    assert sum(1 for s in specs if s.block_len == 126) == 3
    assert sum(1 for s in specs if s.block_len == 63) == 15


def test_run_one_sweep_computes_then_cached(clean_abi_memo, tmp_path,
                                            monkeypatch):
    """_run_one_sweep strict-loads, computes+publishes on a miss
    ('computed'), reports 'cached' on resume, returns a status tuple (never a
    DataFrame)."""
    import sweep_cache as sc, causal_backtest as cb
    frames = _frames_small()
    monkeypatch.setattr(cb, "build_joint_frame",
                        lambda ticker, refit, *a, **k: frames[refit].copy())
    spec = _spec(tmp_path, n_boot=48)
    st1 = sc._run_one_sweep(spec)
    assert st1[:3] == ("QQQ", 63, 0) and st1[3] == "computed"
    assert isinstance(st1[4], float)
    st2 = sc._run_one_sweep(spec)
    assert st2[3] == "cached"


def test_run_prefill_workers1_resumes_and_logs(clean_abi_memo, tmp_path,
                                               monkeypatch):
    import sweep_cache as sc, causal_backtest as cb, os
    frames = _frames_small()
    monkeypatch.setattr(cb, "build_joint_frame",
                        lambda ticker, refit, *a, **k: frames[refit].copy())
    specs = [_spec(tmp_path, n_boot=48, seed=0),
             _spec(tmp_path, n_boot=48, seed=1)]
    log = str(tmp_path / "sweep_run.log")
    r1 = sc.run_prefill(specs, workers=1, log_path=log)
    assert sorted(x[3] for x in r1) == ["computed", "computed"]
    r2 = sc.run_prefill(specs, workers=1, log_path=log)       # resume
    assert sorted(x[3] for x in r2) == ["cached", "cached"]
    assert os.path.exists(log) and "[sweep]" in open(log).read()


def test_invalidate_sweep_keys_deletes_exactly_those(clean_abi_memo, tmp_path,
                                                     monkeypatch):
    """_invalidate_sweep_keys derives each fp EXACTLY as cached_sweep does
    and removes both files, leaving unrelated keys intact."""
    import sweep_cache as sc, causal_backtest as cb, os
    frames = _frames_small()
    monkeypatch.setattr(cb, "build_joint_frame",
                        lambda ticker, refit, *a, **k: frames[refit].copy())
    specs = [_spec(tmp_path, n_boot=48, seed=0),
             _spec(tmp_path, n_boot=48, seed=1)]
    sc.run_prefill(specs, workers=1)
    before = [f for f in os.listdir(tmp_path) if f.startswith("sweep_")]
    assert len(before) == 4                              # 2 units x (parquet+meta)
    sc._invalidate_sweep_keys([specs[0]], str(tmp_path))
    after = [f for f in os.listdir(tmp_path) if f.startswith("sweep_")]
    assert len(after) == 2                               # only seed-1 survives
    # and it is idempotent (deleting again is a no-op, not an error)
    sc._invalidate_sweep_keys([specs[0]], str(tmp_path))


# --- real causal fixtures that survive a spawned worker process -----

def _write_real_causal_fixtures(tmp_path, ticker="QQQ", seed=1):
    """Build a small real corpus straddling train_end and populate its causal
    cache via build_joint_frame (real HMM, warmup=60/n_restarts=2). Returns
    (raw_dir, causal_dir). No monkeypatch -> the fixtures survive spawn."""
    import causal_backtest as cb
    from _synth import two_regime_returns, prices_from_returns
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    r = two_regime_returns(n_bull=80, n_bear=40, n_bull2=40, seed=seed)
    px = prices_from_returns(r, ticker=ticker, start="2013-06-03")   # straddles
    syn_name = cb.UND_TO_SYN[ticker]
    syn = px.rename(columns={ticker: syn_name}).copy()
    syn[syn_name] = syn[syn_name] ** 1.5
    px.to_csv(raw / f"{ticker}.csv", index=False)
    syn.to_csv(raw / f"{syn_name}.csv", index=False)
    pd.DataFrame({"Date": pd.to_datetime(px["Date"]),
                  "irx_rate_annual": [0.05] * len(px)}).to_csv(
        raw / "IRX.csv", index=False)
    causal = str(tmp_path / "causal")
    for refit in (21, 63):
        cb.build_joint_frame(ticker, refit, str(raw), causal, vol_window=20,
                             warmup=60, n_restarts=2)
    return str(raw), causal
