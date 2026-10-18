"""Content-addressed cache + parallel executor for the gate-parameter sweep.

Makes strategy_sweep.py parallel + resumable WITHOUT changing results: each of
the 24 independent per-(ticker, block_len, seed) sweep units is cached to
lossless Parquet, content-addressed by a fingerprint over the post-mask numeric
arrays plus a compute-ABI digest.
"""
import concurrent.futures as _cf
import hashlib
import inspect
import json
import os
import time
from collections import namedtuple

import numpy as np

SWEEP_CACHE_VERSION = 1          # manual escape hatch: bump to invalidate every cached unit; the ABI
                                 # digest invalidates only on a change to a function, constant or
                                 # library version compute_abi_digest lists, not on other code changes
_ABI_MEMO = {}


def _src(obj, name):
    """inspect.getsource, FAIL-CLOSED: on OSError/TypeError raise naming the
    unavailable member instead of silently dropping it."""
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError) as e:
        raise RuntimeError(
            f"compute_abi_digest: cannot read source of {name}: {e}")


def compute_abi_digest():
    """SHA-1 over the ordered, explicitly transitive sweep manifest: every
    function in the sweep call graph (by source), every keyed constant (by
    repr), gate_engine.LEVERAGE (by bytes), and the numpy/pandas/python
    versions. Memoized per process; tests that patch a manifest member
    clear _ABI_MEMO."""
    if "d" in _ABI_MEMO:
        return _ABI_MEMO["d"]
    import sys
    import numpy
    import pandas
    import pyarrow
    import strategy_sweep as ss
    import gate_engine as ge
    import objective as ob
    funcs = [
        (ss._sweep_ticker_seeded, "ss._sweep_ticker_seeded"),
        (ss._mask_train_era, "ss._mask_train_era"),
        (ss._arrays, "ss._arrays"), (ss.cash_th_of, "ss.cash_th_of"),
        (ss.grid_points, "ss.grid_points"),
        (ob.evaluate_point, "ob.evaluate_point"),
        # blend leg + its rebalance cost enter the compute graph via
        # evaluate_point's on_leverage<3 path; hash their SOURCE explicitly so a
        # later correction to any of them flips the digest (evaluate_point only
        # hashes its OWN source, not its callees').
        (ob.static_leverage_lr, "ob.static_leverage_lr"),
        (ob.static_leverage_lr_net, "ob.static_leverage_lr_net"),
        (ob._static_weights, "ob._static_weights"),
        (ob._rebalance_turnover_cost, "ob._rebalance_turnover_cost"),
        (ob.max_drawdown_from_lr, "ob.max_drawdown_from_lr"),
        (ob.terminal_log_wealth, "ob.terminal_log_wealth"),
        (ob.annualized_log_wealth, "ob.annualized_log_wealth"),
        (ob.cvar, "ob.cvar"),
        (ge.bootstrap_indices, "ge.bootstrap_indices"),
        (ge.exposure_states, "ge.exposure_states"),
        (ge.strategy_log_returns, "ge.strategy_log_returns"),
    ]
    h = hashlib.sha1()
    for obj, name in funcs:
        s = _src(obj, name).encode("utf-8")
        h.update(name.encode())
        h.update(len(s).to_bytes(8, "big"))
        h.update(s)
    consts = [("SWITCH_COST_BPS", repr(ss.SWITCH_COST_BPS)),
              ("BLOCK_LEN", repr(ss.BLOCK_LEN)), ("TRAIN_END", repr(ss.TRAIN_END)),
              ("CASH_GAP", repr(ss.CASH_GAP)), ("GRID", repr(ss.GRID)),
              ("TRADING_DAYS", repr(ob.TRADING_DAYS)),
              ("py", sys.version.split()[0]), ("np", numpy.__version__),
              ("pd", pandas.__version__),
              # pyarrow: the Parquet bit-identical round-trip is a
              # version-specific empirical claim (pyarrow 19.0), so a pyarrow
              # change must invalidate the cache -- nothing else keys it.
              ("pa", pyarrow.__version__)]
    for name, val in consts:
        v = str(val).encode()
        h.update(name.encode())
        h.update(len(v).to_bytes(8, "big"))
        h.update(v)
    h.update(b"LEVERAGE")
    lev = ge.LEVERAGE.astype("<f8").tobytes()
    h.update(len(lev).to_bytes(8, "big"))
    h.update(lev)
    _ABI_MEMO["d"] = h.hexdigest()
    return _ABI_MEMO["d"]


_NUMERIC_COLS = ("p_bull", "trail_ratio", "lr_1x", "lr_3x", "lr_cash")


def frame_fingerprint(masked_by_refit):
    """SHA-1 over ONLY the five numeric float64 arrays of the post-mask frames
    (never the object-dtype Date, which is cross-process unstable). Raw <f8
    bytes make this process-independent; caller masks first."""
    h = hashlib.sha1()
    for refit in sorted(masked_by_refit):
        h.update(b"refit")
        h.update(int(refit).to_bytes(8, "big"))
        m = masked_by_refit[refit]
        for col in _NUMERIC_COLS:
            a = np.ascontiguousarray(m[col].to_numpy(dtype="<f8"))
            h.update(col.encode())
            h.update(repr(a.shape).encode())
            b = a.tobytes()
            h.update(len(b).to_bytes(8, "big"))
            h.update(b)
    return h.hexdigest()


def sweep_fingerprint(masked_by_refit, n_boot, block_len, seed, train_end,
                      on_leverage=3.0):
    """Full sweep-unit key: the post-mask frame fingerprint, the
    ordered grid + CASH_GAP + _PARAM_COLS repr, the keyed scalars (n_boot,
    block_len, seed, train_end, on_leverage), the compute-ABI digest, and
    SWEEP_CACHE_VERSION. excess_floor is deliberately ABSENT (floor
    independence). on_leverage IS present and load-bearing: the L_on
    transform lives inside evaluate_point (not the frame), so frame_fingerprint
    AND the ABI digest are identical across L_on -- without this part L_on=2.0
    would false-hit L_on=1.5's cached parquet."""
    import strategy_sweep as ss
    parts = [
        ("frame", frame_fingerprint(masked_by_refit)),
        ("grid", repr(ss.grid_points())), ("cash_gap", repr(ss.CASH_GAP)),
        ("param_cols", repr(ss._PARAM_COLS)),
        ("n_boot", str(int(n_boot))), ("block_len", str(int(block_len))),
        # train_end: callers MUST pass the frozen string form (TRAIN_END); it is
        # also covered by compute_abi_digest's repr(TRAIN_END), so a change flips
        # the key regardless. str() is deterministic per process either way.
        ("seed", str(int(seed))), ("train_end", str(train_end)),
        ("on_leverage", repr(float(on_leverage))),
        ("abi", compute_abi_digest()), ("ver", str(SWEEP_CACHE_VERSION))]
    h = hashlib.sha1()
    for name, val in parts:
        v = val.encode()
        h.update(name.encode())
        h.update(len(v).to_bytes(8, "big"))
        h.update(v)
    return h.hexdigest()


def _paths(cache_dir, fp):
    base = os.path.join(cache_dir, f"sweep_{fp}")
    return base + ".parquet", base + ".meta.json"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_sweep(cache_dir, fp, metrics_df, meta):
    """Atomically publish a sweep unit: Parquet to a unique temp ->
    os.replace; then the meta (carrying a SHA-256 of the payload) the same way.
    Content-addressing + unique temps + payload checksum eliminate the
    shared-tmp race and half-publish acceptance. metrics_df excludes feasible."""
    os.makedirs(cache_dir, exist_ok=True)
    pq, mp = _paths(cache_dir, fp)
    tmp = f"{pq}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    metrics_df.to_parquet(tmp, index=False)
    os.replace(tmp, pq)
    meta = dict(meta)
    meta["payload_sha256"] = _sha256_file(pq)
    mtmp = f"{mp}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    with open(mtmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(mtmp, mp)


def _grid_ok(df):
    """The cached _PARAM_COLS tuples must equal grid_points() row-for-row,
    exact and in order -- protects feasibility_stability's
    positional votes."""
    import strategy_sweep as ss
    want = [tuple(p[c] for c in ss._PARAM_COLS) for p in ss.grid_points()]
    if len(df) != len(want):
        return False
    got = [tuple(r) for r in
           df[list(ss._PARAM_COLS)].itertuples(index=False, name=None)]
    return got == want


def load_sweep(cache_dir, fp, meta):
    """Return the cached metrics DataFrame on a VALIDATED hit, else None. A hit
    requires both files present, meta consistent (excluding payload_sha256),
    payload SHA-256 match, and an ordered-grid match."""
    import pandas as pd
    pq, mp = _paths(cache_dir, fp)
    if not (os.path.exists(pq) and os.path.exists(mp)):
        return None
    try:
        with open(mp) as fh:
            cached = json.load(fh)
    except Exception:
        return None
    want = dict(meta)
    want.pop("payload_sha256", None)
    got = dict(cached)
    payload = got.pop("payload_sha256", None)
    if got != want:                    # meta (incl. ticker) must match exactly:
        return None                    # the fingerprint alone is ticker-blind
    # Everything past here can raise if a concurrent force-rebuild deleted the
    # parquet mid-flight (TOCTOU) or the file is corrupt/wrong-schema; a resumable
    # multi-process design must treat any such failure as a MISS, never a crash.
    try:
        if payload != _sha256_file(pq):
            return None
        df = pd.read_parquet(pq)
        if not _grid_ok(df):
            return None
    except (OSError, KeyError, ValueError):
        return None
    return df


SweepSpec = namedtuple("SweepSpec",
    "ticker block_len seed n_boot train_end n_restarts vol_window warmup "
    "data_dir cache_dir on_leverage")
# on_leverage is a TRAILING field with a 3.0 default so every existing 10-arg
# positional SweepSpec(...) constructor stays valid and means "hardwired 3x".
SweepSpec.__new__.__defaults__ = (3.0,)


def _strict_load_frames(spec):
    """Obtain both refit frames SOLELY via the strict read-only causal loader
    -- build params come from the spec (so --quick's n_restarts hits
    the right cache), and a miss RAISES rather than refitting."""
    import causal_backtest as cb
    out = {}
    for refit in (21, 63):
        out[refit] = cb.build_joint_frame(
            spec.ticker, refit, spec.data_dir, spec.cache_dir,
            vol_window=spec.vol_window, warmup=spec.warmup,
            n_restarts=spec.n_restarts, strict=True)
    return out


def _with_feasible(metrics_df, excess_floor):
    df = metrics_df.copy()
    df["feasible"] = df["cvar5_excess_dd"] >= excess_floor
    return df


def cached_sweep(spec, excess_floor, joint_by_refit=None):
    """One sweep unit, with exactly ONE frame source per mode.
    Returns (frame, hit): hit True means the metrics came from the cache.

    - cache_dir is None: direct passthrough -- compute from joint_by_refit via
      _sweep_ticker_seeded, no I/O, byte-identical to today; hit is False.
    - cache_dir set: strict-load frames (ignore any passed frame), mask,
      fingerprint, load-or-compute+publish; the returned frame carries
      feasible recomputed at excess_floor (floor-independent cache)."""
    import strategy_sweep as ss
    if spec.cache_dir is None:
        if joint_by_refit is None:
            raise ValueError("cache_dir=None requires joint_by_refit")
        df = ss._sweep_ticker_seeded(
            spec.ticker, joint_by_refit, n_boot=spec.n_boot,
            excess_floor=excess_floor, block_len=spec.block_len,
            seed=spec.seed, train_end=spec.train_end,
            on_leverage=spec.on_leverage)
        return df, False
    frames = _strict_load_frames(spec)
    masked = ss._mask_train_era(frames, spec.train_end)
    fp = sweep_fingerprint(masked, spec.n_boot, spec.block_len, spec.seed,
                           spec.train_end, on_leverage=spec.on_leverage)
    meta = {"ticker": spec.ticker, "block_len": spec.block_len,
            "seed": spec.seed, "n_boot": spec.n_boot,
            "train_end": str(spec.train_end),
            "on_leverage": spec.on_leverage, "fp": fp}
    hit = load_sweep(spec.cache_dir, fp, meta)
    if hit is not None:
        return _with_feasible(hit, excess_floor), True
    full = ss._sweep_ticker_seeded(
        spec.ticker, masked, n_boot=spec.n_boot, excess_floor=excess_floor,
        block_len=spec.block_len, seed=spec.seed, train_end=spec.train_end,
        on_leverage=spec.on_leverage)
    metrics = full.drop(columns=["feasible"])
    publish_sweep(spec.cache_dir, fp, metrics, meta)
    return _with_feasible(metrics, excess_floor), False


# --- Parallel executor over the 24 independent sweep units -------

def effective_workers(requested, peak_worker_gb=1.5, os_reserve_gb=2.0,
                      free_gb=None):
    """max(1, min(requested, 24, cpu_count, memory_cap)) where memory_cap =
    floor((free_GB - os_reserve_GB) / peak_worker_GB). Free RAM is usually
    the binding constraint; reject requested < 1 with a clear error;
    os.cpu_count() is None -> 1. (Unit-count cap 24 includes DIA.)"""
    if requested is None or requested < 1:
        raise ValueError(f"--workers must be >= 1, got {requested!r}")
    if free_gb is None:
        import psutil
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
    mem_cap = max(1, int((free_gb - os_reserve_gb) // max(peak_worker_gb, 1e-9)))
    return max(1, min(requested, 24, (os.cpu_count() or 1), mem_cap))


def enumerate_specs(tickers, block63_seeds, cache_dir, bp, on_leverage=3.0):
    """The 24 SweepSpecs: per ticker, one @ block 63 per seed (20 total) plus
    one @ block 126 seed 0 (4 total). bp is the shared build-params dict.
    on_leverage stamps every spec so the whole run shares one risk-on leverage
    (and one cache namespace)."""
    specs = []
    for tk in tickers:
        for s in block63_seeds:
            specs.append(SweepSpec(tk, 63, s, bp["n_boot"], bp["train_end"],
                                   bp["n_restarts"], bp["vol_window"],
                                   bp["warmup"], bp["data_dir"], cache_dir,
                                   on_leverage))
        specs.append(SweepSpec(tk, 126, 0, bp["n_boot"], bp["train_end"],
                               bp["n_restarts"], bp["vol_window"],
                               bp["warmup"], bp["data_dir"], cache_dir,
                               on_leverage))
    return specs


def _run_one_sweep(spec):
    """Module-level (picklable) worker: strict-load + compute+publish if
    missing (no-op if present = resume); returns a status tuple, NEVER a
    DataFrame. One strict-load per call."""
    t0 = time.perf_counter()
    _, hit = cached_sweep(spec, 0.0)
    return (spec.ticker, spec.block_len, spec.seed,
            "cached" if hit else "computed", time.perf_counter() - t0)


def _invalidate_sweep_keys(specs, cache_dir):
    """Delete both files for each spec's key, deriving the fp EXACTLY as
    cached_sweep does (strict-load -> mask -> sweep_fingerprint) so the
    --force-rebuild lifecycle targets precisely the keys the run will
    regenerate. Missing files are ignored (idempotent).

    NOTE: when --force-rebuild follows a causal-CONTENT change, the
    post-force frames hash to NEW fps, so these deletes hit the not-yet-existing
    new keys (no-op) and the OLD-fp sweep units are left as disk litter. This is
    correctness-safe (content-addressing means they can never be a false hit),
    only unbounded growth in data/backtest/ across content-changing rebuilds."""
    import strategy_sweep as ss
    for spec in specs:
        masked = ss._mask_train_era(_strict_load_frames(spec), spec.train_end)
        fp = sweep_fingerprint(masked, spec.n_boot, spec.block_len, spec.seed,
                               spec.train_end, on_leverage=spec.on_leverage)
        for p in _paths(cache_dir, fp):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


def run_prefill(specs, workers, log_path=None):
    """Prefill every spec's cache unit. workers==1 is a genuine in-process
    path (no pool); workers>1 uses a ProcessPoolExecutor and consumes results
    as_completed for live progress. Each completion line prints to stdout and,
    if given, appends to log_path."""
    def emit(st):
        line = (f"[sweep] {st[0]} b{st[1]} s{st[2]}: {st[3]} "
                f"in {st[4]:.0f}s")
        print(line, flush=True)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(line + "\n")

    n = effective_workers(workers)
    out = []
    if n == 1:
        for s in specs:
            st = _run_one_sweep(s)
            emit(st)
            out.append(st)
        return out
    with _cf.ProcessPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(_run_one_sweep, s): s for s in specs}
        for f in _cf.as_completed(futs):
            st = f.result()
            emit(st)
            out.append(st)
    return out
