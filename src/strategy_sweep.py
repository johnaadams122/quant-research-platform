# src/strategy_sweep.py
"""Gate-parameter sweep orchestration.

Objective (fixed before the sweep): expected log-wealth PRIMARY, paired
excess-drawdown-vs-1x CVaR(5%) floor as the constraint; median terminal +
Calmar reported, never optimized. Train era ONLY inside the sweep; the
holdout (2015+) is touched exactly once, by evaluate_holdout, for the
chosen configuration. Plateau over spike: selection maximizes the
NEIGHBORHOOD MEDIAN, never a single point.

"Stage 1" below names the original two-state gate (1x/3x, no cash state);
the `cash_arm` grid dimension extends it with a cash state, and its `never`
arm reproduces Stage 1 exactly.
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import pandas as pd

from causal_backtest import build_joint_frame, CASH_TRANSFORM_VERSION
from gate_engine import bootstrap_indices
from objective import (evaluate_point, max_drawdown_from_lr,
                       static_leverage_lr, static_leverage_lr_net,
                       _static_weights, _rebalance_turnover_cost,
                       annualized_log_wealth, terminal_log_wealth, cvar)

GRID = {"enter_th": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
        "exit_th": (0.30, 0.35, 0.40, 0.45, 0.50),
        "min_dwell": (0, 5, 10),
        "overlay": (True, False),
        "refit_every": (21, 63),
        "cash_arm": ("never", "gap_10", "gap_20")}
# cash_arm: categorical de-risk leg. "never" reproduces Stage 1
# exactly (cash_th=-inf, no cash state ever entered); gap_10/gap_20 set the
# cash entry level a fixed gap BELOW exit_th. CASH_GAP["never"] is None and
# never enters a param-key tuple/merge/CSV -- only cash_th_of consumes it.
CASH_GAP = {"never": None, "gap_10": 0.10, "gap_20": 0.20}
TRAIN_END = "2014-12-31"
EXCESS_FLOOR = -0.40         # User knob (--excess-floor): paired
                             # excess-drawdown-vs-1x CVaR(5%) floor
SELECTION_DIAGNOSTIC_FLOORS = (-0.15, -0.25, -0.40)   # the
                             # fixed reference floors the selection matrix is
                             # always reported at; the CLI --excess-floor is
                             # unioned in (selection_floors()) so the primary
                             # floor is always present as a matrix key.
N_BOOT_SWEEP = 2000
N_BOOT_FINAL = 10000
BLOCK_LEN = 63
SWITCH_COST_BPS = 4.0
TICKERS = ("QQQ", "SOX", "SPY", "DIA")   # DIA = the Dow leg (its data layer
                                         # + validation live in data_pipeline)
STABILITY_SEEDS = 5         # k seeded train sweeps per ticker,
STABILITY_MIN = 4           # computed once and cached; a grid point is
FEASIBILITY_MARGIN = 0.02   # "stable" if it clears floor+margin on >= m of k
LIVE_DEFAULT = dict(enter_th=0.60, exit_th=0.40, min_dwell=0,
                    overlay=True, refit_every=21)
_PARAM_COLS = ["enter_th", "exit_th", "min_dwell", "overlay", "refit_every",
               "cash_arm"]


def cash_th_of(cfg):
    """Cash-entry p_bull threshold for a config's cash_arm. never -> -inf
    (the cash state is unreachable, nesting Stage 1 exactly); otherwise the
    exit_th minus the arm's gap. Returns a float, never None/NaN."""
    gap = CASH_GAP[cfg["cash_arm"]]
    return float("-inf") if gap is None else float(cfg["exit_th"]) - gap


def grid_points():
    pts = [dict(zip(GRID, combo))
           for combo in itertools.product(*GRID.values())]
    assert all(p["exit_th"] < p["enter_th"] for p in pts)
    return pts


def _arrays(joint):
    return (joint["p_bull"].values, joint["trail_ratio"].values,
            joint["lr_1x"].values, joint["lr_3x"].values,
            joint["lr_cash"].values)


def _mask_train_era(joint_by_refit, train_end=TRAIN_END):
    """{refit_every: joint frame filtered to Date <= train_end}. Shared by
    sweep_ticker and the pipeline's block-len-126 sensitivity step
    so both use the identical TRAIN ERA definition -- the holdout rows are
    never present in the returned frames."""
    masked = {}
    for re_k, joint in joint_by_refit.items():
        m = joint[pd.to_datetime(joint["Date"]) <= train_end].reset_index(drop=True)
        assert len(m) and pd.to_datetime(m["Date"]).max() <= pd.Timestamp(train_end)
        masked[re_k] = m
    return masked


def _sweep_ticker_seeded(ticker, joint_by_refit, n_boot=N_BOOT_SWEEP,
                         excess_floor=EXCESS_FLOOR, block_len=BLOCK_LEN,
                         seed=0, train_end=TRAIN_END, on_leverage=3.0):
    """One row per grid point, TRAIN ERA ONLY -- seed-parameterized core of
    sweep_ticker. The resample seed is a parameter (per-refit CRN still holds
    within one seed) and each config's cash-entry threshold cash_th_of(p) is
    threaded into evaluate_point (never -> -inf, nesting Stage 1). train_end is threaded through solely so
    the public sweep_ticker keeps honoring its own train_end argument; every
    caller uses the TRAIN_END default."""
    masked = _mask_train_era(joint_by_refit, train_end)
    idx_by_refit = {re_k: bootstrap_indices(len(m), n_boot,
                                            block_len=block_len, seed=seed)
                    for re_k, m in masked.items()}
    # 1x baseline maxdd per path: depends only on (lr_1x, idx) -- one
    # computation per refit_every, shared by all configs (CRN pairing).
    base_by_refit = {re_k: max_drawdown_from_lr(
                         masked[re_k]["lr_1x"].values[idx_by_refit[re_k]])
                     for re_k in masked}
    rows = []
    for p in grid_points():
        m = masked[p["refit_every"]]
        out = evaluate_point(_arrays(m), idx_by_refit[p["refit_every"]],
                             p["enter_th"], p["exit_th"], p["min_dwell"],
                             p["overlay"], switch_cost_bps=SWITCH_COST_BPS,
                             baseline_maxdd_b=base_by_refit[p["refit_every"]],
                             cash_th=cash_th_of(p), on_leverage=on_leverage)
        rows.append({**p, **out,
                     "feasible": out["cvar5_excess_dd"] >= excess_floor})
    return pd.DataFrame(rows)


def sweep_ticker(ticker, joint_by_refit, n_boot=N_BOOT_SWEEP,
                 excess_floor=EXCESS_FLOOR, train_end=TRAIN_END,
                 block_len=BLOCK_LEN, *, cache_dir=None, n_restarts=15,
                 vol_window=63, warmup=1000, data_dir="data/raw",
                 on_leverage=3.0):
    """One row per grid point, TRAIN ERA ONLY (seed 0). cache_dir=None is a
    direct passthrough to the seeded core; cache_dir set routes through the
    content-addressed sweep cache, which strict-loads its frames using the
    build params threaded here.
    on_leverage sets the risk-on rung's leverage (3.0 = hardwired-3x)."""
    from sweep_cache import cached_sweep, SweepSpec
    spec = SweepSpec(ticker, block_len, 0, n_boot, train_end, n_restarts,
                     vol_window, warmup, data_dir, cache_dir, on_leverage)
    return cached_sweep(spec, excess_floor, joint_by_refit=joint_by_refit)[0]


def compute_seed_tables(ticker, joint_by_refit, k=STABILITY_SEEDS,
                        n_boot=N_BOOT_SWEEP, block_len=BLOCK_LEN, *,
                        cache_dir=None, n_restarts=15, vol_window=63,
                        warmup=1000, data_dir="data/raw", on_leverage=3.0):
    """The k seeded train sweeps for one ticker, computed ONCE. Feasibility at
    any floor is derived from these cached cvar5_excess_dd columns -- no floor
    re-sweeps (spec: compute metrics once, only masks change). Each seed is a
    distinct cache unit; seed 0 shares its key with sweep_ticker's (dedup)."""
    from sweep_cache import cached_sweep, SweepSpec
    return [cached_sweep(
                SweepSpec(ticker, block_len, s, n_boot, TRAIN_END, n_restarts,
                          vol_window, warmup, data_dir, cache_dir, on_leverage),
                0.0, joint_by_refit=joint_by_refit)[0]
            for s in range(k)]


def feasibility_stability(seed_tables, floor, m=STABILITY_MIN,
                          margin=FEASIBILITY_MARGIN):
    """Per grid point: feasible WITH MARGIN on >= m of the cached seed tables.
    Pure over the tables -- no sweeping."""
    votes = None
    for tbl in seed_tables:
        passed = (tbl["cvar5_excess_dd"].values >= (floor + margin)).astype(int)
        votes = passed if votes is None else votes + passed
    return votes >= m


_BASE_COLS = ["enter_th", "exit_th", "min_dwell", "overlay", "refit_every"]


def cash_lift_table(results):
    """CRN-paired marginal effect of the cash escape hatch: each cash arm vs
    the identical config's 'never' arm (same draws). Isolates cash from every
    other knob."""
    df = results.reset_index(drop=True)
    never = (df[df["cash_arm"] == "never"]
             .set_index(_BASE_COLS)[["expected_lw_ann", "cvar5_excess_dd"]])
    out = []
    for _, r in df.iterrows():
        base = never.loc[tuple(r[c] for c in _BASE_COLS)]
        out.append({**{c: r[c] for c in _BASE_COLS}, "cash_arm": r["cash_arm"],
                    "d_expected_lw_ann": float(r["expected_lw_ann"] - base["expected_lw_ann"]),
                    "d_cvar5_excess_dd": float(r["cvar5_excess_dd"] - base["cvar5_excess_dd"]),
                    "cash_occupancy": float(r.get("cash_occupancy", 0.0)),
                    "cash_zero_use_freq": float(r.get("cash_zero_use_freq", 1.0))})
    return pd.DataFrame(out)


def cost_sensitivity(joint_by_refit_by_ticker, chosen_by_ticker, idx_context,
                     switch_cost_bps=SWITCH_COST_BPS, on_leverage=3.0):
    """Cost-convention sensitivity: re-evaluates each ticker's
    chosen config under BOTH cost conventions -- risk_notional (primary) and
    dollar_turnover (alternate) -- and reports expected_lw_ann and
    cvar5_excess_dd under each plus their deltas, so the choice of cost
    convention is shown to be (or not be) load-bearing for the reported
    metrics rather than assumed. This function does not tune or select
    anything and does not mask dates itself -- joint_by_refit_by_ticker
    should already hold whichever window (train or holdout) the caller wants
    re-priced.

    joint_by_refit_by_ticker: {ticker: {refit_every: joint_frame}}.
    chosen_by_ticker: {ticker: config_dict} -- each config_dict carries the
    _PARAM_COLS keys (enter_th, exit_th, min_dwell, overlay, refit_every,
    cash_arm), i.e. the same shape select_per_ticker/select_plateau return.
    idx_context: {ticker: idx} -- the bootstrap indices for that ticker's
    chosen config's (refit_every, date-window) slice; the SAME idx is reused
    for both cost models per ticker so the delta isolates the convention,
    not the resample draw."""
    rows = []
    for tk in sorted(chosen_by_ticker):
        cfg = chosen_by_ticker[tk]
        j = joint_by_refit_by_ticker[tk][cfg["refit_every"]]
        idx = idx_context[tk]
        cth = cash_th_of(cfg)
        by_model = {}
        for model in ("risk_notional", "dollar_turnover"):
            by_model[model] = evaluate_point(
                _arrays(j), idx, cfg["enter_th"], cfg["exit_th"],
                cfg["min_dwell"], cfg["overlay"],
                switch_cost_bps=switch_cost_bps, cash_th=cth,
                cost_model=model, on_leverage=on_leverage)
        rn, dt = by_model["risk_notional"], by_model["dollar_turnover"]
        rows.append({
            "ticker": tk, **{c: cfg[c] for c in _PARAM_COLS},
            "expected_lw_ann_risk_notional": rn["expected_lw_ann"],
            "expected_lw_ann_dollar_turnover": dt["expected_lw_ann"],
            "d_expected_lw_ann": dt["expected_lw_ann"] - rn["expected_lw_ann"],
            "cvar5_excess_dd_risk_notional": rn["cvar5_excess_dd"],
            "cvar5_excess_dd_dollar_turnover": dt["cvar5_excess_dd"],
            "d_cvar5_excess_dd": dt["cvar5_excess_dd"] - rn["cvar5_excess_dd"],
        })
    return pd.DataFrame(rows)


# _static_weights + _rebalance_turnover_cost live in objective.py (imported
# above) so the passive static-L ladder AND the active on-leverage leg
# (static_leverage_lr_net) share ONE blend-maintenance cost definition -- no
# gross-vs-net asymmetry between the active strategy and its same-L comparator.


def static_ladder_table(frame, idx, base_dd,
                        ladder=(0, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """One row per L in `ladder` for a static (daily-rebalanced constant
    effective leverage) strategy built from `frame`'s OWN lr_cash/lr_1x/
    lr_3x legs (objective.static_leverage_lr) -- no hardcoded refit_every;
    whatever legs `frame` carries are what gets used.

    `frame` must already be the exact single (ticker, refit_every, era)
    joint frame the caller selected (e.g. one config's train-era frame or
    its holdout-era frame -- write_report's caller uses one per era). `idx` and
    `base_dd` are the caller's own CRN bootstrap resample and 1x baseline
    maxdd vector for whatever config the ladder is being compared against
    -- BOTH consumed exactly as passed, NEVER recomputed here, so every row
    of the ladder is CRN-paired against that config's own draws.

    Columns:
      L                       -- the ladder rung.
      expected_lw_ann         -- bootstrap-mean annualized log-wealth,
                                  gross of the rebalance-cost approximation.
      cvar5_maxdd             -- CVaR(5%) of the per-path max drawdown of the
                                  NET (costed) leg -- the implementable figure,
                                  consistent with the active on-leverage
                                  strategy, which also holds the net blend leg
                                  (no gross-vs-net asymmetry when compared).
      cvar5_excess_dd         -- CVaR(5%) of (per-path maxdd - base_dd) on the
                                  NET leg, paired against the PASSED base_dd. At
                                  L=1.0, static_leverage_lr returns lr_1x
                                  bit-exact (cost 0), so if base_dd is itself
                                  max_drawdown_from_lr(lr_1x[idx]) for this
                                  SAME idx, every path's excess is exactly
                                  0 and cvar5_excess_dd == 0.0 exactly.
      expected_lw_ann_costed  -- expected_lw_ann minus the bootstrap-mean
                                  annualized daily-rebalance turnover charge
                                  (see _rebalance_turnover_cost). Endpoints
                                  L in {0,1,3} are pure single legs, so
                                  their cost is exactly 0 and this column
                                  equals expected_lw_ann bit-for-bit;
                                  intermediate L blends two diverging legs
                                  at a constant target weight and pays a
                                  nonzero daily rebalancing turnover charge.
    """
    idx = np.asarray(idx)
    base_dd = np.asarray(base_dd, dtype=float)
    T = idx.shape[1]
    lr_cash = frame["lr_cash"].values
    lr_1x = frame["lr_1x"].values
    lr_3x = frame["lr_3x"].values
    n_paths = idx.shape[0]
    if base_dd.shape != (n_paths,):
        raise ValueError(
            "base_dd shape %s != (%d,); it must be "
            "max_drawdown_from_lr(lr_1x[idx]) for the SAME idx"
            % (base_dd.shape, n_paths))

    rows = []
    for L in ladder:
        Lf = float(L)
        lr_L = static_leverage_lr(lr_cash, lr_1x, lr_3x, Lf)              # gross
        lr_L_costed = static_leverage_lr_net(lr_cash, lr_1x, lr_3x, Lf,
                                             SWITCH_COST_BPS)             # net
        lr_b = lr_L[idx]
        lr_b_costed = lr_L_costed[idx]

        ann_b = annualized_log_wealth(terminal_log_wealth(lr_b), T)
        ann_b_costed = annualized_log_wealth(terminal_log_wealth(lr_b_costed), T)

        # drawdown from the NET (implementable) leg so the passive ladder's risk
        # matches the active on-leverage strategy's net-of-rebalance leg -- no
        # gross-vs-net asymmetry when the two are compared. Endpoints L in
        # {0,1,3} have cost 0, so net == gross there bit-for-bit.
        maxdd_b = np.atleast_1d(max_drawdown_from_lr(lr_b_costed))
        excess_dd = maxdd_b - base_dd

        rows.append({
            "L": Lf,
            "expected_lw_ann": float(ann_b.mean()),
            "expected_lw_ann_costed": float(ann_b_costed.mean()),
            "cvar5_maxdd": cvar(maxdd_b, 0.05),
            "cvar5_excess_dd": cvar(excess_dd, 0.05),
        })
    return pd.DataFrame(rows)


def _step(values):
    v = sorted(set(values))
    return min(b - a for a, b in zip(v, v[1:])) if len(v) > 1 else 1.0


def neighborhood_score(results, feasible=None):
    """Median expected_lw_ann over each row's grid neighborhood (+/-1 step in
    enter/exit/dwell; same overlay+refit). Rows not in `feasible` (default:
    the frame's own per-ticker flag; selection passes the GLOBAL
    feasible_all mask) score -inf and are EXCLUDED from every neighbor's
    median -- they never rescue a neighborhood, and they never drag one
    down as literal -inf values. The neighborhood includes the row itself,
    so a feasible row whose neighbors are all excluded keeps a well-defined
    median (its own value)."""
    df = results.reset_index(drop=True)
    if feasible is None:
        feasible = df["feasible"].values
    feasible = np.asarray(feasible, dtype=bool)
    assert feasible.shape == (len(df),)
    steps = {c: _step(GRID[c]) * 1.5 for c in ("enter_th", "exit_th",
                                               "min_dwell")}
    feas = df[feasible].reset_index()
    scores = pd.Series(-np.inf, index=df.index, dtype=float)
    for i, row in df.iterrows():
        if not feasible[i]:
            continue
        nb = feas[(feas["cash_arm"] == row["cash_arm"]) &
                  (feas["overlay"] == row["overlay"]) &
                  (feas["refit_every"] == row["refit_every"]) &
                  (abs(feas["enter_th"] - row["enter_th"]) <= steps["enter_th"]) &
                  (abs(feas["exit_th"] - row["exit_th"]) <= steps["exit_th"]) &
                  (abs(feas["min_dwell"] - row["min_dwell"]) <= steps["min_dwell"])]
        scores.iloc[i] = float(nb["expected_lw_ann"].median())
    return scores


_CASH_ARM_RANK = {"never": 0, "gap_20": 1, "gap_10": 2}    # least-cash first


def select_per_ticker(results, floor, seeds_feasible=None):
    """One config for one ticker at one floor. Eligibility = seeds_feasible if
    given (the >=m/k multi-seed vote mask, which already includes seed 0),
    else this ticker's own cvar5_excess_dd >= floor. Returns an explicit
    viability record; zero-feasible is NOT given a fake winner."""
    df = results.reset_index(drop=True)
    if seeds_feasible is not None:
        feas = np.asarray(seeds_feasible, dtype=bool)
    else:
        feas = (df["cvar5_excess_dd"].values >= floor)
    feasible_count = int(feas.sum())
    scores = neighborhood_score(df, feasible=feas)
    d = df.copy()
    d["_score"] = scores.values
    d["_arm_rank"] = d["cash_arm"].map(_CASH_ARM_RANK)
    ranked = d.sort_values(
        ["_score", "enter_th", "exit_th", "min_dwell", "overlay",
         "refit_every", "_arm_rank"],
        ascending=[False, False, False, False, False, False, True])
    top = ranked.iloc[0]
    cfg = {c: top[c] for c in _PARAM_COLS}
    cfg["min_dwell"] = int(cfg["min_dwell"])
    cfg["refit_every"] = int(cfg["refit_every"])
    cfg["overlay"] = bool(cfg["overlay"])
    viable = feasible_count > 0 and np.isfinite(top["_score"])
    return {"config": cfg, "viable": bool(viable),
            "feasible_count": feasible_count,
            "selection_mode": "plateau" if viable else "no_feasible_fallback",
            "score": float(top["_score"])}


def selection_floors(cli_floor):
    """The selection-floor union: the fixed reference floors
    (SELECTION_DIAGNOSTIC_FLOORS) unioned with the CLI --excess-floor,
    de-duplicated and sorted ascending (deepest/most-negative first). The CLI
    floor is guaranteed present, so the PRIMARY floor is always a key of the
    matrix run_selection_matrix builds. Returns a tuple of floats."""
    return tuple(sorted(set(SELECTION_DIAGNOSTIC_FLOORS) | {float(cli_floor)}))


def run_selection_matrix(train_results_seed0, seed_tables_by_ticker, floors):
    """The per-(floor, ticker) selection matrix.

    For each floor in `floors` and each ticker in `train_results_seed0`, derive
    that ticker's multi-seed stability mask via
    feasibility_stability(seed_tables_by_ticker[ticker], floor) and pass it as
    `seeds_feasible` into select_per_ticker(train_results_seed0[ticker], floor).
    The seed-0 full-grid train sweep (train_results_seed0[ticker]) and the k
    cached seed tables (seed_tables_by_ticker[ticker]) are both produced by
    _sweep_ticker_seeded over grid_points() in the identical order, so the
    >=m/k vote mask aligns row-for-row with the train frame.

    `seed_tables_by_ticker` may be None (check-only / no-stability path): every
    cell then falls back to select_per_ticker's own per-frame
    cvar5_excess_dd >= floor eligibility (seeds_feasible=None).

    Returns {floor: {ticker: record}} where record is a select_per_ticker()
    output: {"config": {.._PARAM_COLS}, "viable": bool, "feasible_count": int,
    "selection_mode": str, "score": float}."""
    out = {}
    for floor in floors:
        cells = {}
        for ticker, res in train_results_seed0.items():
            if seed_tables_by_ticker is None:
                seeds_feasible = None
            else:
                seeds_feasible = feasibility_stability(
                    seed_tables_by_ticker[ticker], floor)
            cells[ticker] = select_per_ticker(res, floor,
                                              seeds_feasible=seeds_feasible)
        out[floor] = cells
    return out


def select_plateau(per_ticker_results):
    """ONE global config. Order of operations:
    (i) global feasible_all FIRST via a parameter-key inner join across
    tickers (tolerates reordered-but-equivalent frames, like the original);
    (ii) per-ticker neighborhood scores masked by feasible_all -- a config
    infeasible anywhere neither is selected nor lends plateau support;
    (iii) global score = mean of per-ticker scores."""
    tickers = sorted(per_ticker_results)
    merged = None
    for tk in tickers:
        r = per_ticker_results[tk]
        f = r[_PARAM_COLS].copy()
        f[f"feasible_{tk}"] = r["feasible"].values.astype(bool)
        merged = f if merged is None else merged.merge(f, on=_PARAM_COLS)
    feas_cols = [c for c in merged.columns if c.startswith("feasible_")]
    merged["feasible_all"] = merged[feas_cols].all(axis=1)

    def _key(row):
        return tuple(row[c] for c in _PARAM_COLS)

    fa_by_key = {_key(row): bool(row["feasible_all"])
                 for _, row in merged.iterrows()}
    scores_by_ticker = {}
    s_by_key_by_tk = {}
    for tk in tickers:
        r = per_ticker_results[tk].reset_index(drop=True)
        mask = np.array([fa_by_key[_key(row)] for _, row in r.iterrows()],
                        dtype=bool)
        s = neighborhood_score(r, feasible=mask)
        scores_by_ticker[tk] = s
        s_by_key_by_tk[tk] = {_key(row): float(s.iloc[i])
                              for i, (_, row) in enumerate(r.iterrows())}
    for tk in tickers:
        merged[f"score_{tk}"] = [s_by_key_by_tk[tk][_key(row)]
                                 for _, row in merged.iterrows()]
    score_cols = [f"score_{tk}" for tk in tickers]
    merged["global_score"] = merged[score_cols].mean(axis=1)
    merged.loc[~merged["feasible_all"], "global_score"] = -np.inf
    # deterministic lean-cautious tie-break: plateaus tie by construction
    top = merged.sort_values(
        ["global_score", "enter_th", "exit_th", "min_dwell", "overlay",
         "refit_every"],
        ascending=[False, False, False, False, False, False]).head(5)
    chosen = top.iloc[0][_PARAM_COLS].to_dict()
    chosen["min_dwell"] = int(chosen["min_dwell"])
    chosen["refit_every"] = int(chosen["refit_every"])
    chosen["overlay"] = bool(chosen["overlay"])
    return {"chosen": chosen, "table": top.reset_index(drop=True),
            "merged": merged, "scores_by_ticker": scores_by_ticker}


def _dedupe_param_configs(configs):
    """De-duplicate a list of _PARAM_COLS config dicts by their param tuple
    (incl. cash_arm), preserving first-seen order. Pure -- no evaluation."""
    seen = set()
    out = []
    for cfg in configs:
        key = tuple(cfg[c] for c in _PARAM_COLS)
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def evaluate_holdout(distinct_configs, joint_by_refit_by_ticker,
                     n_boot=N_BOOT_FINAL, train_end=TRAIN_END,
                     excess_floor=EXCESS_FLOOR, on_leverage=3.0):
    """The ONLY function that touches rows dated after train_end. Evaluates
    each DE-DUPLICATED viable config (distinct_configs -- full _PARAM_COLS
    dicts incl. cash_arm; dedup by the _PARAM_COLS tuple) plus three fixed,
    explicit reference configs -- live_default, always_1x, always_3x, each
    cash_arm="never" with an explicit refit_every=21 (never inherited from
    any distinct config) -- EXACTLY ONCE per ticker; cash_th=cash_th_of(cfg)
    is threaded into every evaluate_point call. Rows are keyed by a
    config_id column ("config_0", "config_1", ... for the deduped distinct
    configs in first-seen order; the reference names otherwise) so the
    caller can map results back to (floor, ticker) cells. No tuning decision
    may consume these results."""
    deduped = _dedupe_param_configs(distinct_configs)
    configs = {f"config_{i}": cfg for i, cfg in enumerate(deduped)}
    # The always-on reference is named for its ACTUAL leverage: "always_3x" at
    # the default 3.0 (a stable config_id that row-matching keys on),
    # "always_2x"/"always_1.5x" at a lower L_on. always_1x stays the fixed 1x baseline regardless.
    lev_ref = f"always_{on_leverage:g}x"
    configs.update({
        "live_default": {**LIVE_DEFAULT, "refit_every": 21,
                         "cash_arm": "never"},
        "always_1x": dict(enter_th=2.0, exit_th=0.4, min_dwell=0,
                          overlay=True, refit_every=21, cash_arm="never"),
        lev_ref: dict(enter_th=-1.0, exit_th=-2.0, min_dwell=0,
                      overlay=False, refit_every=21, cash_arm="never"),
    })
    rows = []
    for tk, by_refit in joint_by_refit_by_ticker.items():
        cache = {}                       # refit_every -> (idx, base_dd)
        for config_id, cfg in configs.items():
            j = by_refit[cfg["refit_every"]]
            h = j[pd.to_datetime(j["Date"]) > train_end].reset_index(drop=True)
            assert len(h) and pd.to_datetime(h["Date"]).min() > pd.Timestamp(train_end)
            key = cfg["refit_every"]
            if key not in cache:
                idx = bootstrap_indices(len(h), n_boot, block_len=BLOCK_LEN,
                                        seed=1)   # independent of train's seed=0
                cache[key] = (idx,
                              max_drawdown_from_lr(h["lr_1x"].values[idx]))
            idx, base_dd = cache[key]
            out = evaluate_point(_arrays(h), idx, cfg["enter_th"],
                                 cfg["exit_th"], cfg["min_dwell"],
                                 cfg["overlay"],
                                 switch_cost_bps=SWITCH_COST_BPS,
                                 baseline_maxdd_b=base_dd,
                                 cash_th=cash_th_of(cfg),
                                 on_leverage=on_leverage)
            rows.append({"ticker": tk, "config_id": config_id,
                        **{c: cfg[c] for c in _PARAM_COLS}, **out,
                        "feasible": out["cvar5_excess_dd"] >= excess_floor})
    return pd.DataFrame(rows)


def _arm_sliced_pivots(df, cfg, cash_arm):
    """Slice df at the FIXED (min_dwell, overlay, refit_every) from cfg AND
    the FIXED cash_arm, THEN pivot enter_th x exit_th -> expected_lw_ann /
    feasible. Fixing cash_arm as part of the slice -- before either
    pivot_table call -- is the load-bearing step: rows
    from a DIFFERENT cash_arm at the same (enter_th, exit_th) cell must
    never reach the same pivot bucket, because pivot_table's default
    aggfunc='mean' would silently blend a feasible cell (1.0) with an
    infeasible one (0.0) into a meaningless 0.5 (or NaN, depending on
    grouping), rather than reporting either arm's real value. Factored out
    of plot_heatmaps so the no-cross-arm-averaging property is directly
    testable against the real slicing/pivoting code.

    Returns (enters, exits, val, feas) -- val/feas are DataFrames indexed
    by enters, columned by exits."""
    sl = df[(df["min_dwell"] == cfg["min_dwell"]) &
            (df["overlay"] == cfg["overlay"]) &
            (df["refit_every"] == cfg["refit_every"]) &
            (df["cash_arm"] == cash_arm)]
    enters = sorted(sl["enter_th"].unique())
    exits = sorted(sl["exit_th"].unique())
    val = (sl.pivot_table(index="enter_th", columns="exit_th",
                          values="expected_lw_ann")
             .reindex(index=enters, columns=exits))
    feas = (sl.pivot_table(index="enter_th", columns="exit_th",
                           values="feasible")
              .reindex(index=enters, columns=exits))
    return enters, exits, val, feas


def plot_heatmaps(results, selection_by_ticker, out_dir, cash_arm):
    """One PNG per ticker at
    {out_dir}/charts/gate_sweep_{ticker}_{cash_arm}.png (the plan's File
    Structure location; charts/ is gitignored): expected_lw_ann on the
    enter_th x exit_th grid, sliced at that ticker's own selection
    (selection_by_ticker[ticker]["config"] -- a select_per_ticker()-shaped
    record, run_selection_matrix()'s per-floor slice) AND the
    single FIXED cash_arm this call renders (the caller
    loops GRID["cash_arm"] and calls this once per arm -- this function
    never loops arms itself). Infeasible cells are hatched/greyed. The
    ticker's selected (enter_th, exit_th) cell is outlined ONLY on the
    facet whose cash_arm matches that ticker's own selected cash_arm --
    outlining it on a different arm's facet would mark a cell as "chosen"
    in an arm where it was never actually selected."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    png_paths = []
    for tk in sorted(results.keys()):
        df = results[tk]
        cfg = selection_by_ticker[tk]["config"]
        enters, exits, val, feas = _arm_sliced_pivots(df, cfg, cash_arm)
        if len(enters) == 0 or len(exits) == 0:
            # Fail-silent guard: an empty (ticker, cash_arm)
            # slice would otherwise hit ax.imshow() on a 0-row pivot and write
            # a plausible-looking but empty/garbage PNG. This never fires on
            # the real pipeline -- main() loops GRID["cash_arm"] and
            # grid_points() guarantees every arm exists for every ticker's
            # results -- it only converts a future caller misuse (ticker/arm
            # mismatch, partial results, a typo) into a clear error.
            raise ValueError(
                f"plot_heatmaps: no rows for ticker={tk!r} "
                f"cash_arm={cash_arm!r} — cannot render heatmap")

        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        im = ax.imshow(val.values.astype(float), cmap="viridis",
                       aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax, label="expected_lw_ann (bootstrap mean)")
        ax.set_xticks(range(len(exits)))
        ax.set_xticklabels([f"{x:.2f}" for x in exits])
        ax.set_yticks(range(len(enters)))
        ax.set_yticklabels([f"{e:.2f}" for e in enters])
        ax.set_xlabel("exit_th")
        ax.set_ylabel("enter_th")
        ax.set_title(f"{tk} [{cash_arm}]: expected_lw_ann  "
                    f"(min_dwell={cfg['min_dwell']}, "
                    f"overlay={cfg['overlay']}, "
                    f"refit_every={cfg['refit_every']})")

        feas_arr = feas.values
        for i in range(feas_arr.shape[0]):
            for j in range(feas_arr.shape[1]):
                cell_feasible = feas_arr[i, j]
                if pd.isna(cell_feasible) or not bool(cell_feasible):
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="lightgrey",
                        edgecolor="dimgray", hatch="//", alpha=0.9,
                        zorder=3))
        if (cfg.get("cash_arm") == cash_arm and
                cfg["enter_th"] in enters and cfg["exit_th"] in exits):
            ci = enters.index(cfg["enter_th"])
            cj = exits.index(cfg["exit_th"])
            ax.add_patch(Rectangle(
                (cj - 0.5, ci - 0.5), 1, 1, fill=False, edgecolor="crimson",
                linewidth=2.5, zorder=5))

        fig.tight_layout()
        fp = os.path.join(charts_dir, f"gate_sweep_{tk}_{cash_arm}.png")
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        png_paths.append(fp)
    return png_paths


def _df_md(df, float_fmt="{:.4f}"):
    """Minimal markdown-table renderer -- no external table-formatting
    dependency. Floats fixed-precision; everything else stringified."""
    if df is None or len(df) == 0:
        return "_(no rows)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (bool, np.bool_)):
                s = str(bool(v))
            elif isinstance(v, (float, np.floating)):
                s = "" if pd.isna(v) else float_fmt.format(float(v))
            else:
                s = str(v)
            cells.append(s.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


_CASH_LIFT_TOP_N = 15   # bounded top-N table in report section 5 (CRN cash-lift)


def _cash_lift_summary(df, n=_CASH_LIFT_TOP_N):
    """Bounded, decision-useful RENDER-TIME summary of one ticker's
    cash_lift_table() output for report section 5 (CRN cash-lift). cash_lift_table itself is
    untouched (still one row per grid point, the correct full data) -- this
    only shapes what write_report prints:

      1. DROP the `never`-arm rows: a `never` row is that same config's
         `never` arm compared to itself, so its delta is trivially 0 by
         construction and carries no information.
      2. Rank the remaining gap-arm rows (`gap_10`/`gap_20`) by
         `d_expected_lw_ann` DESCENDING -- most log-wealth-improving cash
         arm first, the sweep's PRIMARY objective.
      3. Cap at the top `n` rows.

    Returns None/empty unchanged (caller's _df_md already renders those as
    "_(no rows)_")."""
    if df is None or len(df) == 0:
        return df
    gap = df[df["cash_arm"] != "never"]
    return gap.sort_values("d_expected_lw_ann", ascending=False).head(n)


def write_report(train_results, selection_matrix, cash_lift, static_ladders,
                 cost_sens, holdout, block126, png_paths, out_fp,
                 excess_floor, selection_floors, provenance=None):
    """Render the cash-state markdown report and write it to out_fp.

    Pure string-building + one file write. Every number is taken AS GIVEN from
    the passed frames/dicts -- write_report never recomputes selection,
    feasibility, neighborhood scores, or metrics.

    The report is organised around a PER-(ticker, floor) SELECTION MATRIX
    rather than one global chosen config: each (floor, ticker) cell carries
    its own select_per_ticker() record, and non-viable cells are disclosed as
    "no viable config at this floor." rather than fabricating a winner.

    Assumed input shapes (main()'s wiring produces exactly these):
      train_results   : {ticker: DataFrame} -- the FULL 1080-row seed-0 train
                        sweep per ticker (ALL cash_arms), carrying _PARAM_COLS
                        + cvar5_excess_dd + expected_lw_ann + feasible + the
                        excess_tail_* composition columns. (sweep_ticker output
                        over the full 1080-point grid.)
      selection_matrix: {floor: {ticker: record}} -- one select_per_ticker()
                        record per cell: {"config": {.._PARAM_COLS}, "viable":
                        bool, "feasible_count": int, "selection_mode": str,
                        "score": float}. (run_selection_matrix output.)
      cash_lift       : {ticker: DataFrame} -- cash_lift_table() per ticker
                        (_BASE_COLS + cash_arm + d_expected_lw_ann +
                        d_cvar5_excess_dd + cash_occupancy + cash_zero_use_freq).
                        The FULL per-config frame (one row per grid point incl.
                        `never`) -- report section 5 renders a bounded top-N summary
                        of it at RENDER TIME (_cash_lift_summary), never the
                        whole thing.
      static_ladders  : {ticker: {"train": DataFrame, "holdout": DataFrame}}
                        -- static_ladder_table() per ticker per era (L +
                        expected_lw_ann + expected_lw_ann_costed + cvar5_maxdd +
                        cvar5_excess_dd).
      cost_sens       : DataFrame -- cost_sensitivity() over the primary-floor
                        per-ticker configs (ticker + _PARAM_COLS + the
                        risk_notional/dollar_turnover metric pair + deltas).
      holdout         : DataFrame -- evaluate_holdout() rows keyed by config_id
                        (NOT a "config" column).
      block126        : {floor: {ticker: {.._PARAM_COLS, rank_block126,
                        score_block126, feasible_block126}}} --
                        _block126_sensitivity() output.
      png_paths       : [str] -- faceted heatmap PNGs
                        gate_sweep_{ticker}_{cash_arm}.png.
      excess_floor    : float -- the PRIMARY floor knob (the CLI
                        --excess-floor); drives the objective text, the
                        calibration table, the primary-floor tail composition,
                        and the reproduction command.
      selection_floors: iterable of floats -- the selection-floor union
                        (SELECTION_DIAGNOSTIC_FLOORS union the CLI floor);
                        drives the multi-floor feasible-rate, selection-record,
                        and block-126 tables.
      provenance      : dict|None -- run provenance.
    Absent metric columns fail loud with KeyError by design.
    """
    floors = list(selection_floors)
    primary = excess_floor
    tickers = sorted(train_results)
    arms = list(GRID["cash_arm"])

    lines = []

    lines.append("# Cash-State Gate Sweep -- Report\n")

    # (0) run provenance: identify the run that produced these
    # numbers. Deliberately NO wall-clock timestamp -- regeneration diffs stay
    # deterministic.
    lines.append("## 0. Run provenance\n")
    if provenance is None:
        lines.append(
            "_Not recorded -- write_report was called outside the pipeline; "
            "run src/strategy_sweep.py to embed the run parameters._\n")
    else:
        if provenance.get("mode") == "quick":
            lines.append(
                "**QUICK MODE -- reduced bootstrap/restarts; not "
                "decision-grade; regenerate without `--quick` before relying "
                "on numbers.**\n")
        prov_rows = ["| parameter | value |", "| --- | --- |"]
        for k, v in provenance.items():
            if isinstance(v, dict):
                v = "; ".join(f"{kk}: {vv}" for kk, vv in sorted(v.items()))
            prov_rows.append(f"| {k} | {v} |")
        lines.append("\n".join(prov_rows) + "\n")

    # (1) locked objective + banned-objective statement
    lines.append("## 1. Objective (locked)\n")
    if excess_floor <= 0:
        floor_policy_frag = (
            f"Floor policy: {excess_floor} means "
            f"{abs(excess_floor) * 100:.0f} percentage points of full-horizon "
            "max drawdown added over 1x, averaged across the worst 5% of "
            "futures -- a user-set knob (`--excess-floor`) and policy constraint.")
    else:
        floor_policy_frag = (
            f"Floor policy: {excess_floor} requires the strategy's worst-5% "
            "average drawdown to be at least "
            f"{excess_floor * 100:.0f} percentage points SHALLOWER than 1x -- "
            "an unusual, stricter-than-index constraint.")
    lines.append(
        "The sweep selects, PER TICKER at each selection floor, the "
        "configuration that maximizes **expected log-wealth** -- the "
        "bootstrap-mean annualized terminal log-wealth -- subject to a "
        "**paired excess-drawdown-vs-1x CVaR(5%) floor** "
        f"(`excess_floor = {excess_floor}`) as a hard feasibility constraint: "
        "for each bootstrap path, the strategy's max drawdown minus 1x's max "
        "drawdown on the SAME path; the average of the worst 5% of these "
        "excesses must be no deeper than the floor. This is the **worst "
        "paired excess-drawdown tail**, not a general drawdown cap -- it "
        "provides NO absolute-loss guarantee (a config can pass with "
        "near-total drawdowns whenever plain 1x is similarly catastrophic; the "
        "never-cash design owns the market's own risk, the gate owns only the "
        "increment; a config that sits in cash through a crash earns a "
        "POSITIVE excess on those paths -- the metric credits it). An "
        "individual path may add more than the floor value -- the floor caps "
        "the tail AVERAGE. The absolute `cvar5_maxdd` is reported as context "
        "in the calibration and holdout tables; it is no longer a constraint. "
        + floor_policy_frag +
        " Median terminal-wealth ratio and realized Calmar ratio are computed "
        "and reported as cross-checks; they are **never optimized**. "
        "Arithmetic-mean terminal wealth is explicitly BANNED as an "
        "optimization objective -- it degenerates to always-max-leverage. This "
        "objective and the floor formulation were fixed before the sweep "
        "was run.\n")

    # (2) grid + multiple-testing note (cash arm + per-ticker
    # multi-floor selection).
    lines.append("## 2. Grid and multiple-testing control\n")
    lines.append(
        "The grid has 1080 points (6 `enter_th` x 5 `exit_th` x 3 `min_dwell` "
        "x 2 `overlay` x 2 `refit_every` x 3 `cash_arm`). The `never` cash arm "
        "reproduces Stage 1 exactly (cash state unreachable, `cash_th = -inf`); "
        "`gap_10`/`gap_20` set the cash-entry level a fixed gap below "
        "`exit_th`. Selection maximizes the **neighborhood-median** "
        "`expected_lw_ann` (a plateau over a spike), computed WITHIN a fixed "
        "cash_arm and PER TICKER, with infeasible neighbors scoring `-inf`; "
        "ties break deterministically, leaning cautious (highest `enter_th`, "
        "then `exit_th`, `min_dwell`, `overlay`, `refit_every`, then least-cash "
        "arm). Per-ticker selection over 1080 points x "
        f"{len(tickers)} tickers x up to {len(floors)} floors materially "
        "inflates the search versus Stage 1's single joint gate; the two "
        "honest limits (the holdout is spent; forward/paper data is the only "
        "fresh confirmation) are stated in report section 8.\n")

    # (3) NEW -- multi-floor feasible-RATE table (never vs cash, per ticker,
    # per selection floor). Rates are the fraction of that (ticker, arm) grid
    # slice whose seed-0 cvar5_excess_dd clears the floor -- a descriptive
    # grid rate, NOT the multi-seed stability vote select_per_ticker uses for
    # viability (that lives in report section 4's records).
    lines.append("## 3. Feasible-rate matrix (never vs cash, per ticker, "
                 "per selection floor)\n")
    lines.append(
        "Fraction of each ticker's grid slice (by cash arm) whose seed-0 "
        "`cvar5_excess_dd` clears the floor. `never` is the Stage-1 arm; "
        "`gap_10`/`gap_20` are the cash arms. This is a descriptive grid rate; "
        "actual per-cell viability (multi-seed stability vote) is in section "
        "4.\n")
    header = "| ticker | floor | " + " | ".join(arms) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in arms) + " |"
    rate_rows = [header, sep]
    for tk in tickers:
        df = train_results[tk]
        for fl in floors:
            cells = []
            for arm in arms:
                sl = df[df["cash_arm"] == arm]
                if len(sl) == 0:
                    cells.append("n/a")
                else:
                    ok = int((sl["cvar5_excess_dd"].values >= fl).sum())
                    n = len(sl)
                    cells.append(f"{ok}/{n} ({ok / n:.0%})")
            rate_rows.append(f"| {tk} | {fl} | " + " | ".join(cells) + " |")
    lines.append("\n".join(rate_rows) + "\n")

    # (4) NEW -- per-(ticker, floor) selection records. Non-viable cell ->
    # "no viable config at this floor."
    lines.append("## 4. Per-(ticker, floor) selection records\n")
    lines.append(
        "One record per (floor, ticker) cell, taken verbatim from the "
        "selection matrix (`viable` / `selection_mode` / chosen `cash_arm` + "
        "config). A cell with no stably-feasible configuration renders **no "
        "viable config at this floor.** -- it is NOT given a fabricated "
        "winner.\n")
    for fl in floors:
        lines.append(f"### Floor {fl}\n")
        srows = ["| ticker | viable | selection_mode | cash_arm | config |",
                 "| --- | --- | --- | --- | --- |"]
        for tk in tickers:
            rec = selection_matrix.get(fl, {}).get(tk)
            if rec is None:
                srows.append(f"| {tk} | -- | (absent) | -- | -- |")
                continue
            if not rec.get("viable"):
                srows.append(
                    f"| {tk} | False | {rec.get('selection_mode', '')} | -- | "
                    "no viable config at this floor. |")
                continue
            cfg = rec["config"]
            cfg_txt = ", ".join(f"{c}={cfg[c]}" for c in _PARAM_COLS)
            srows.append(
                f"| {tk} | True | {rec.get('selection_mode', '')} | "
                f"{cfg['cash_arm']} | {cfg_txt} |")
        lines.append("\n".join(srows) + "\n")

    # (5) CRN cash-lift + occupancy / zero-use frequency.
    lines.append("## 5. CRN cash-lift (each cash arm vs the same config's "
                 "`never` arm)\n")
    lines.append(
        "Common-random-number-paired marginal effect of the cash escape "
        "hatch: `d_expected_lw_ann` / `d_cvar5_excess_dd` are each cash arm "
        "MINUS the identical config's `never` arm on the SAME draws (a `never` "
        "row is 0 by construction). `cash_occupancy` is the mean fraction of "
        "bars in cash; `cash_zero_use_freq` is the fraction of bootstrap paths "
        "that never touched cash.\n")
    lines.append(
        f"_Top-{_CASH_LIFT_TOP_N} summary per ticker: `never`-arm rows "
        "dropped (trivially 0 vs themselves) and the remaining gap-arm rows "
        "ranked by `d_expected_lw_ann` descending (most log-wealth-improving "
        "first), capped at the top "
        f"{_CASH_LIFT_TOP_N}. The full per-config cash-lift (all grid rows, "
        "every cash arm) is reproducible from the sweep's per-ticker data "
        "(`cash_lift_table()` over `gate_sweep_{ticker}.csv`) and is not "
        "dumped here._\n")
    if cash_lift is None:
        lines.append("_Not supplied in this report run._\n")
    else:
        for tk in tickers:
            lines.append(f"### {tk}\n")
            lines.append(_df_md(_cash_lift_summary(cash_lift.get(tk))))
            lines.append("")

    # (6) static-L ladder: gross + costed, per era.
    lines.append("## 6. Static-leverage ladder (daily-rebalanced constant L)\n")
    lines.append(
        "Per era, one row per ladder rung L on that ticker's own "
        "lr_cash/lr_1x/lr_3x legs, CRN-paired against the same config's draws. "
        "`expected_lw_ann` is GROSS; `expected_lw_ann_costed` subtracts the "
        "daily-rebalance turnover charge (0 at the pure endpoints L in "
        "{0,1,3}). `cvar5_maxdd`/`cvar5_excess_dd` are computed on the NET "
        "(costed) leg -- the same implementable leg the active on-leverage "
        "strategy holds, so passive and active risk are apples-to-apples. "
        "`cvar5_excess_dd` is paired vs 1x, so L=1.0 is exactly 0.\n")
    if static_ladders is None:
        lines.append("_Not supplied in this report run._\n")
    else:
        for tk in tickers:
            lines.append(f"### {tk}\n")
            eras = static_ladders.get(tk) or {}
            for era in ("train", "holdout"):
                lines.append(f"**{era} era**\n")
                lines.append(_df_md(eras.get(era)))
                lines.append("")

    # (7) cost-convention sensitivity.
    lines.append("## 7. Cost sensitivity (risk-notional vs "
                 "dollar-turnover)\n")
    lines.append(
        "Each primary-floor per-ticker selection re-priced under BOTH cost "
        "conventions -- **risk-notional** (primary: bps on |delta leverage|) "
        "and **dollar-turnover** (alternate: flat bps on the executed hop) -- "
        "with the SAME resample draws, so the delta isolates the convention, "
        "not the draw. The convention is shown to be (or not be) load-bearing "
        "for `expected_lw_ann` / `cvar5_excess_dd` rather than assumed. It "
        "does NOT change the selection.\n")
    lines.append(_df_md(cost_sens))
    lines.append("")

    # (8) holdout keyed by config_id + spent-holdout disclosure.
    lines.append("## 8. Holdout (post-train_end) results vs baselines\n")
    lines.append(
        "Rows are keyed by `config_id` -- the deduplicated union of distinct "
        "viable per-(ticker,floor) configs (`config_0`, `config_1`, ...) plus "
        "the three fixed references (`live_default`, `always_1x`, "
        "`always_3x`), evaluated ONCE each on `Date > train_end` rows only. "
        "The `feasible` flag here is **descriptive only** -- computed per row "
        "against the same-row 1x baseline; it neither selects nor invalidates "
        "any train-selected config. `always_1x` rows have excess_dd "
        "identically 0 by construction.\n")
    # The holdout is SPENT -- forward / paper data is the only fresh
    # confirmation.
    lines.append(
        "> **The holdout is spent.** It was already read while designing "
        "the -0.40 excess-floor policy, so this "
        "holdout read is **not fresh** out-of-sample confirmation and must not "
        "be treated as one. The genuinely fresh confirmation for any selected "
        "winner is **forward / paper data** collected after the fact, not "
        "this backtest -- weigh the holdout numbers below as confirmatory of an "
        "already-spent read, never as a pristine one-shot.\n")
    lines.append(_df_md(holdout))
    lines.append("")

    # (9) excess-floor calibration (per ticker, train era) --
    # percentiles over the grid, no plateau dependency.
    lines.append("## 9. Excess-floor calibration (per ticker, train era)\n")
    prov_note = ""
    if provenance is not None:
        prov_note = (f"Provenance: n_boot={provenance.get('n_boot_sweep')}, "
                     f"block_len={provenance.get('block_len')}, seed=0, "
                     f"era Date <= {provenance.get('train_end')}. ")
    lines.append(
        prov_note + "min = worst (deepest) excess, max = best. Percentiles are "
        "computed across each ticker's grid configurations' own "
        "`cvar5_excess_dd` values (each already a tail-average over bootstrap "
        "paths), NOT across individual bootstrap paths; CRN pairing makes "
        "these values correlated, not independent draws. `feasible` counts the "
        "grid rows clearing the primary floor "
        f"({excess_floor}); `near_floor` counts configs within 0.02 of the "
        "floor -- a small bootstrap movement there can flip a config's "
        "feasibility.\n")
    cal = ["| ticker | feasible | min | p5 | p25 | p50 | p75 | p95 | max "
           "| near_floor |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tk in tickers:
        df = train_results[tk]
        n_total = len(df)
        n_feas = int(df["feasible"].sum())
        v = df["cvar5_excess_dd"].astype(float).values
        q = np.percentile(v, [5, 25, 50, 75, 95])
        near = int((np.abs(v - excess_floor) <= 0.02).sum())
        cal.append(
            f"| {tk} | {n_feas}/{n_total} | {v.min():.4f} "
            f"| {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {q[3]:.4f} "
            f"| {q[4]:.4f} | {v.max():.4f} | {near} |")
    lines.append("\n".join(cal) + "\n")

    # (10) worst paired excess-drawdown tail composition -- for each ticker's
    # PRIMARY-floor selection (the per-ticker analogue of a single global
    # chosen config). Non-viable primary-floor cell -> disclosed, not sliced.
    lines.append("## 10. Worst paired excess-drawdown tail composition "
                 "(primary-floor selection, train era)\n")
    lines.append(
        "The k binding futures per ticker: what the strategy and plain 1x "
        "actually drew down on the paths where the gate added the most "
        "increment over 1x. `worst_excess` is the single worst path -- "
        "individual paths can exceed the floor value.\n")
    primary_cells = selection_matrix.get(primary, {})
    for tk in tickers:
        rec = primary_cells.get(tk)
        if rec is None or not rec.get("viable"):
            lines.append(f"- {tk}: no viable config at this floor.\n")
            continue
        cfg = rec["config"]
        sel = train_results[tk]
        for c in _PARAM_COLS:
            sel = sel[sel[c] == cfg[c]]
        if len(sel) == 0:
            lines.append(f"- {tk}: selected config row not present in "
                         "train_results (unexpected)\n")
            continue
        r = sel.iloc[0]
        lines.append(
            f"- {tk}: k={int(r['excess_tail_k'])}, worst_excess="
            f"{r['excess_tail_worst']:.3f}, strategy maxdd median/worst "
            f"{r['excess_tail_median_maxdd_strategy']:.3f}/"
            f"{r['excess_tail_worst_maxdd_strategy']:.3f}, 1x maxdd "
            f"median/worst {r['excess_tail_median_maxdd_1x']:.3f}/"
            f"{r['excess_tail_worst_maxdd_1x']:.3f}\n")

    # (11) block-length-126 sensitivity over the (floor, ticker) matrix
    # ({floor: {ticker: ...}} dict shape).
    lines.append("## 11. Block-length-126 train-era sensitivity "
                 "(per viable selection)\n")
    lines.append(
        "Full-grid resweep under a longer (126-bar) bootstrap block length, "
        "train era only: all 1080 configurations re-evaluated per ticker (ONCE "
        "per ticker, re-masked per floor), and each VIABLE (floor, ticker) "
        "selection located within ITS OWN ticker's block-126 ranking. "
        "`rank_block126` is 1-indexed; a rank far from 1 or "
        "`feasible_block126 = False` means the 63-bar block-seam bias may be "
        "flattering that selection. Non-viable cells are absent.\n")
    if block126 is None:
        lines.append("_Not supplied in this report run._\n")
    else:
        for fl in floors:
            floor_out = block126.get(fl, {})
            if not floor_out:
                lines.append(f"### Floor {fl}\n")
                lines.append("_No viable selection at this floor._\n")
                continue
            lines.append(f"### Floor {fl}\n")
            brows = ["| ticker | cash_arm | enter_th | exit_th | min_dwell | "
                     "overlay | refit_every | rank_block126 | score_block126 "
                     "| feasible_block126 |",
                     "| --- | --- | --- | --- | --- | --- | --- | --- | --- "
                     "| --- |"]
            for tk in tickers:
                rec = floor_out.get(tk)
                if rec is None:
                    continue
                brows.append(
                    f"| {tk} | {rec.get('cash_arm')} | {rec.get('enter_th')} "
                    f"| {rec.get('exit_th')} | {rec.get('min_dwell')} "
                    f"| {rec.get('overlay')} | {rec.get('refit_every')} "
                    f"| {int(rec['rank_block126'])} "
                    f"| {float(rec['score_block126']):.4f} "
                    f"| {bool(rec['feasible_block126'])} |")
            lines.append("\n".join(brows) + "\n")

    # (12) faceted heatmap links: gate_sweep_{ticker}_{cash_arm}.png.
    if png_paths:
        lines.append("## 12. Faceted train-era heatmaps "
                     "(`expected_lw_ann`, per cash arm)\n")
        lines.append(
            "One PNG per (ticker, cash_arm) at the primary floor's selection "
            "slice (local artifact -- regenerate via report section 14; not "
            "committed).\n")
        report_dir = os.path.dirname(os.path.abspath(out_fp))
        for p in png_paths:
            try:
                rel = os.path.relpath(p, report_dir)
            except ValueError:      # e.g. different drive on Windows
                rel = p
            lines.append(f"![heatmap]({rel.replace(os.sep, '/')})")
        lines.append("")

    # (13) honesty notes
    lines.append("## 13. Honesty notes\n")
    lines.append(
        "- **Multiple testing / spent holdout:** per-ticker selection "
        "over 1080 points x tickers x floors inflates the search relative to "
        "Stage 1's single joint gate. The holdout was already spent designing "
        "the -0.40 policy; the only genuinely fresh out-of-sample confirmation "
        "is forward/paper data (report section 8).\n"
        "- **Feasibility stability:** a config feasible on seed 0 but "
        "infeasible on more than (k-m) of k seeds (k=5, m=4) is NOT marked "
        "viable; a 0.02 feasibility margin guards floor-boundary flips.\n"
        "- **Dot-com era coverage:** the 2000-2002 crash is out-of-eval for "
        "QQQ only (causal eval starts ~2003-03 after the 1,000-bar HMM "
        "warmup); for SPY (~1997-07) and SOX (~1998-04) it is fully in-train "
        "and dominates SOX's very deep train-era drawdown tail; for DIA "
        "(data starts 1998-01-20, causal eval ~2002-01 after warmup) only "
        "the 2002 tail of the crash is in-eval (in-train), so DIA's "
        "train-era tail UNDERSTATES a full dot-com round trip.\n"
        "- **always-3x baseline is a 1-bar approximation:** its degenerate "
        "thresholds arm at bar 0 and never exit, so it is one 1x bar plus a "
        "single arming switch, then 3x for the remainder.\n"
        "- **Holdout cold-start:** `evaluate_holdout` restarts every gate OFF "
        "at the first post-train_end bar, the same fail-closed posture the "
        "gate always starts from; identical in kind across all configs (a "
        "differential bias, not a train-era leak).\n"
        "- **Block-seam turnover bias:** the moving-block bootstrap stitches "
        "non-adjacent days at block seams, manufacturing spurious transitions "
        "that penalize low-dwell/tight-threshold configs; CRN cannot remove "
        "it -- hence the block-126 sensitivity (report section 11).\n"
        "- **Costs:** 4 bps flat per executed switch (`SWITCH_COST_BPS`); no "
        "bid-ask, slippage, or market-impact model. The static-L ladder "
        "(report section 6) charges a separate daily-rebalance turnover approximation.\n"
        "- **No tax modeling (a tax-advantaged account is assumed):** returns are pre-tax; no wash-sale or "
        "capital-gains modeling anywhere in this sweep.\n")

    # (14) reproduction commands
    lines.append("## 14. Reproduction\n")
    lines.append(
        "```powershell\n"
        "# Full test suite for this module\n"
        'python -m pytest '
        "src/tests/test_strategy_sweep.py -q\n\n"
        "# Full real-data pipeline (cache build -> per-ticker sweep -> "
        "per-floor selection matrix -> cash-lift -> cost sensitivity -> "
        "holdout -> static-L -> block-126 -> heatmaps -> report)\n"
        'python src/strategy_sweep.py '
        f"--excess-floor {excess_floor}\n"
        "```\n")

    with open(out_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _block126_sensitivity(selection_matrix, masked_by_ticker, floors,
                          n_boot=N_BOOT_SWEEP, *, cache_dir=None,
                          n_restarts=15, vol_window=63, warmup=1000,
                          data_dir="data/raw", on_leverage=3.0):
    """Block-126 robustness check over the PER-TICKER,
    MULTI-FLOOR selection matrix. Unlike a cross-ticker
    select_plateau-based check, the sweep selects one config per
    (floor, ticker) cell (select_per_ticker), not one global
    config, so sensitivity is "where does each viable (floor, ticker)
    selection rank within ITS OWN ticker's block-126 ranking", never a
    cross-ticker joint rank.

    selection_matrix: {floor: {ticker: record}} -- the
        run_selection_matrix() output shape, i.e. one select_per_ticker()
        record per (floor, ticker) cell: {"config": {..._PARAM_COLS},
        "viable": bool, "feasible_count": int, "selection_mode": str,
        "score": float}. Only cells with record["viable"] is True produce
        an entry in the output -- a non-viable cell has no train-era
        selection to locate.
    masked_by_ticker: {ticker: {refit_every: joint frame}} -- TRAIN-ERA
        masked joint frames per ticker (main()'s own
        _mask_train_era(joint_by_refit) output; the exact shape
        _sweep_ticker_seeded consumes as its `joint_by_refit` argument).
    floors: iterable of floor floats (the selection_floors union) to
        derive block-126 feasibility + per-ticker ranking for.
    n_boot: bootstrap path count for the per-ticker block-126 resweep.

    EFFICIENCY CONTRACT (the key property): the full 1080-point grid is
    resweept via _sweep_ticker_seeded(..., block_len=126, seed=0) EXACTLY
    ONCE PER TICKER -- before the floor loop, cached, and re-masked (a
    fresh feasible mask + neighborhood_score pass, no new sweeping) for
    every floor. cvar5_excess_dd/expected_lw_ann do not depend on the
    SELECTION floor, only the feasibility threshold applied to them does,
    so a per-floor resweep would multiply the (expensive) bootstrap cost
    by len(floors) for no new information.

    Returns {floor: {ticker: {**_PARAM_COLS, "rank_block126": int,
                              "score_block126": float,
                              "feasible_block126": bool}}} -- one inner
    entry per (floor, ticker) cell where
    selection_matrix[floor][ticker]["viable"] is True. rank_block126 is
    1-indexed, computed within that TICKER's own block-126 ranking using
    the identical neighborhood-score + deterministic lean-cautious
    tie-break as select_per_ticker's own sort (cash_arm rank included).
    Non-viable (floor, ticker) cells are absent from the output -- callers
    should treat a missing key the same as its selection_matrix cell
    (no viable train selection, nothing to rank)."""
    from sweep_cache import cached_sweep, SweepSpec
    tickers = sorted(masked_by_ticker)
    results126 = {tk: cached_sweep(
                      SweepSpec(tk, 126, 0, n_boot, TRAIN_END, n_restarts,
                                vol_window, warmup, data_dir, cache_dir,
                                on_leverage),
                      0.0, joint_by_refit=masked_by_ticker[tk])[0]
                  for tk in tickers}

    out = {}
    for floor in floors:
        floor_out = {}
        for tk in tickers:
            sel = selection_matrix.get(floor, {}).get(tk)
            if sel is None or not sel.get("viable"):
                continue
            res = results126[tk]
            feas = res["cvar5_excess_dd"].values >= floor
            scores = neighborhood_score(res, feasible=feas)
            d = res.copy()
            d["_score"] = scores.values
            d["_feasible126"] = feas
            d["_arm_rank"] = d["cash_arm"].map(_CASH_ARM_RANK)
            ranked = d.sort_values(
                ["_score", "enter_th", "exit_th", "min_dwell", "overlay",
                 "refit_every", "_arm_rank"],
                ascending=[False, False, False, False, False, False, True]
            ).reset_index(drop=True)
            ranked["rank_block126"] = ranked.index + 1

            cfg = sel["config"]
            m = ranked
            for c in _PARAM_COLS:
                m = m[m[c] == cfg[c]]
            if len(m) == 0:
                raise ValueError(
                    "viable selection config for (floor=%r, ticker=%r) "
                    "not found in the block-126 grid: %r"
                    % (floor, tk, cfg))
            r = m.iloc[0]
            floor_out[tk] = {**{c: cfg[c] for c in _PARAM_COLS},
                             "rank_block126": int(r["rank_block126"]),
                             "score_block126": float(r["_score"]),
                             "feasible_block126": bool(r["_feasible126"])}
        out[floor] = floor_out
    return out


def _finite_floor(s):
    v = float(s)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(
            "--excess-floor must be a finite float, got %r" % s)
    return v


def _on_leverage(s):
    v = float(s)
    # Domain is (1, 3]: the risk-on rung must be MORE levered than the 1x
    # middle rung. At L<=1 the "risk-on" leg is not risk-on (rung 2 <= rung 1),
    # and static_leverage_lr's L<=1 branch would blend cash/1x, a different
    # regime. 3.0 is the hardwired-3x default.
    if not (1.0 < v <= 3.0):
        raise argparse.ArgumentTypeError(
            "--on-leverage must be in (1.0, 3.0], got %r" % s)
    return v


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Gate-parameter sweep: cache build -> train "
                    "sweep -> plateau selection -> block-126 sensitivity -> "
                    "holdout (once) -> heatmaps -> report.")
    p.add_argument("--excess-floor", type=_finite_floor, default=EXCESS_FLOOR,
                   help="Paired excess-drawdown-vs-1x CVaR(5%%) feasibility "
                        f"floor (default {EXCESS_FLOOR}): the average excess "
                        "drawdown among the worst 5%% of bootstrap paths "
                        "must be no deeper than this. Propagates to "
                        "sweep_ticker, evaluate_holdout, "
                        "_block126_sensitivity, and write_report. "
                        "A user-set knob.")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Force cache rebuild (build_joint_frame force=True) "
                        "even if a matching-meta cache already exists.")
    p.add_argument("--quick", action="store_true",
                   help="Iteration mode: n_boot=500 for the sweep, "
                        "block-126 sensitivity, and holdout; n_restarts=5 "
                        "for cache builds. NOTE: a --quick cache writes "
                        "n_restarts=5 into the cache meta, so a later full "
                        "run auto-rebuilds via meta mismatch -- this is BY "
                        "DESIGN, do not \"optimize\" the rebuild away.")
    p.add_argument("--data-dir", default="data/raw",
                   help="Directory holding data/raw corpus CSVs.")
    p.add_argument("--cache-dir", default="data/backtest",
                   help="Directory for build_joint_frame's cache (gitignored).")
    p.add_argument("--out-dir", default="outputs",
                   help="Directory for sweep CSVs, heatmap PNGs, and the "
                        "report.")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel sweep workers (default 2; 1 = in-process, no "
                        "pool). RAM-bounded and clamped to <=24; size to "
                        "available RAM.")
    p.add_argument("--on-leverage", type=_on_leverage, default=3.0,
                   help="Risk-on rung leverage L_on in (1.0, 3.0] (default 3.0 = "
                        "hardwired 3x, byte-identical to the fixed-3x path). L_on<3 "
                        "makes the risk-on leg the daily-rebalanced 1x/3x blend "
                        "static_leverage_lr(...,L_on), prices its switch cost "
                        "at L_on, namespaces its own sweep cache, and suffixes "
                        "the report path _L{L_on}. A glide-path knob.")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated ticker override (default: the "
                        "module TICKERS). Additive research escape hatch "
                        "(e.g. --tickers NDX_TR,SPX_TR) so a deep-corpus "
                        "run never perturbs the 4-ticker report-of-record.")
    return p


def _parse_args(argv=None):
    return _build_arg_parser().parse_args(argv)


def _read_irx_sha1(cache_dir):
    """Best-effort provenance: the IRX.csv content fingerprint from any cached
    joint-frame meta sidecar (build_joint_frame writes {base}.meta.json with an
    irx_sha1). Returns None if unavailable -- e.g. a monkeypatched build in
    tests that never wrote a meta file -- so provenance stays robust."""
    try:
        # cache_dir now also holds sweep_<fp>.meta.json (no irx_sha1); the
        # `if sha1:` guard skips those, and causal metas (TICKER_re..) sort
        # before sweep_.. so the first hit is still a causal meta.
        for fn in sorted(os.listdir(cache_dir)):
            if fn.endswith(".meta.json"):
                with open(os.path.join(cache_dir, fn)) as fh:
                    sha1 = json.load(fh).get("irx_sha1")
                if sha1:
                    return sha1
    except Exception:
        pass
    return None


def main(argv=None):
    """Cash-state sweep pipeline. For each ticker x refit_every, build/cache
    the joint frame; per ticker run the seed-0 full-grid train sweep (saved to
    CSV) and the k seeded stability tables (computed ONCE); build the
    per-(floor, ticker) selection matrix over the selection-floor union; derive
    the CRN cash-lift, cost sensitivity, and static-L ladders (train +
    holdout) for each ticker's primary-floor selection; evaluate the
    de-duplicated union of distinct viable configs on the holdout exactly once;
    run the block-126 robustness resweep (once per ticker); render the
    per-cash-arm faceted heatmaps at the primary floor; write the report."""
    args = _parse_args(argv)
    # Additive override: default is the module-level TICKERS, so omitting the flag
    # gives the default universe exactly (no behaviour change, no ABI impact).
    tickers = (tuple(t.strip() for t in args.tickers.split(",") if t.strip())
               if args.tickers else TICKERS)
    if not tickers:
        raise SystemExit("--tickers parsed to an empty ticker set")
    n_boot = 500 if args.quick else N_BOOT_SWEEP
    n_boot_holdout = 500 if args.quick else N_BOOT_FINAL
    n_restarts = 5 if args.quick else 15

    # --quick must NEVER touch the decision cache (its n_restarts=5 would poison
    # the causal meta); reject BEFORE any filesystem writes (side-effect-free).
    # realpath, fail-closed, on both sides.
    if (args.quick and
            os.path.realpath(args.cache_dir) == os.path.realpath("data/backtest")):
        raise SystemExit(
            "--quick must not use the decision cache 'data/backtest'; "
            "pass --cache-dir data/backtest-quick.")

    # L_on != 3.0 gets its own out-dir subtree so its CSVs, heatmap PNGs, sweep
    # log, and report never collide with (or clobber) the 3x report-of-record.
    # 3.0 writes directly to out_dir (unchanged).
    if args.on_leverage != 3.0:
        args.out_dir = os.path.join(args.out_dir, f"L{args.on_leverage:g}")
    os.makedirs(args.out_dir, exist_ok=True)

    primary = args.excess_floor
    floors = selection_floors(primary)

    # (1) build/cache the joint frames once per (ticker, refit_every).
    joint_by_refit_by_ticker = {}
    for ticker in tickers:
        joint_by_refit = {}
        for refit_every in GRID["refit_every"]:
            t0 = time.time()
            joint = build_joint_frame(ticker, refit_every, args.data_dir,
                                      args.cache_dir, n_restarts=n_restarts,
                                      force=args.force_rebuild)
            print(f"[build] {ticker} refit_every={refit_every}: "
                 f"{len(joint)} rows in {time.time() - t0:.1f}s")
            joint_by_refit[refit_every] = joint
        joint_by_refit_by_ticker[ticker] = joint_by_refit

    # (1b) PARALLEL PREFILL: populate the 24 independent sweep-cache units
    # (per ticker: seeds 0-4 @ block 63, plus seed 0 @ block 126). Only cache
    # population is parallel; the phase-2/9 assembly below is then pure cache
    # hits (0 recomputes). --force-rebuild deletes the 24 keys once (the causal
    # frames were already force-rebuilt in phase 1) before recomputing.
    from sweep_cache import (enumerate_specs, run_prefill,
                             _invalidate_sweep_keys)
    build_params = dict(n_boot=n_boot, train_end=TRAIN_END,
                        n_restarts=n_restarts, vol_window=63, warmup=1000,
                        data_dir=args.data_dir)
    specs = enumerate_specs(tickers, range(STABILITY_SEEDS), args.cache_dir,
                            build_params, on_leverage=args.on_leverage)
    if args.force_rebuild:
        _invalidate_sweep_keys(specs, args.cache_dir)
    # Pin BLAS threads NOW -- AFTER the (serial, thread-sensitive) causal HMM
    # build, BEFORE the pool spawns inside run_prefill. Workers still inherit the
    # env; phase 1 provably ran in the SAME unpinned environment as the serial
    # oracle (KMeans-init thread sensitivity is why phase 1 stays
    # serial -- pinning it earlier could drift p_bull ~1e-15 and flip a
    # selection).
    if args.workers > 1:
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[_v] = "1"
    run_prefill(specs, args.workers,
                log_path=os.path.join(args.out_dir, "sweep_run.log"))

    # (2) per ticker: the seed-0 full-grid train sweep (report surface +
    # selection frame), the k seeded stability tables (computed ONCE), and the
    # train-era masked joint frames.
    train_results, masked_by_ticker, seed_tables_by_ticker = {}, {}, {}
    for ticker in tickers:
        joint_by_refit = joint_by_refit_by_ticker[ticker]
        # Build params come from the SAME build_params dict enumerate_specs used
        # for the prefill, so prefill and assembly fingerprints match by
        # construction (no 4-place 63/1000 duplication to drift).
        res = sweep_ticker(ticker, joint_by_refit, n_boot=n_boot,
                           excess_floor=args.excess_floor,
                           cache_dir=args.cache_dir,
                           n_restarts=build_params["n_restarts"],
                           vol_window=build_params["vol_window"],
                           warmup=build_params["warmup"],
                           data_dir=build_params["data_dir"],
                           on_leverage=args.on_leverage)
        train_results[ticker] = res
        res.to_csv(os.path.join(args.out_dir, f"gate_sweep_{ticker}.csv"),
                  index=False)
        masked_by_ticker[ticker] = _mask_train_era(joint_by_refit)
        seed_tables_by_ticker[ticker] = compute_seed_tables(
            ticker, joint_by_refit, n_boot=n_boot, cache_dir=args.cache_dir,
            n_restarts=build_params["n_restarts"],
            vol_window=build_params["vol_window"],
            warmup=build_params["warmup"], data_dir=build_params["data_dir"],
            on_leverage=args.on_leverage)

    # (3) per-(floor, ticker) selection matrix over the selection-floor union;
    # each cell's stability mask comes from that ticker's cached seed tables.
    selection_matrix = run_selection_matrix(train_results,
                                            seed_tables_by_ticker, floors)

    # (4) CRN cash-lift per ticker (each cash arm vs its own 'never' arm).
    cash_lift = {tk: cash_lift_table(train_results[tk]) for tk in tickers}

    # (5) primary-floor per-ticker chosen configs + their CRN train draws --
    # shared by the cost sensitivity AND the static-L ladders so both are
    # paired against the identical (config, seed-0 resample) the selection used.
    primary_cells = selection_matrix.get(primary, {})
    chosen_by_ticker, idx_context = {}, {}
    for tk in tickers:
        rec = primary_cells.get(tk)
        if rec is None or not rec.get("viable"):
            continue
        cfg = rec["config"]
        m = masked_by_ticker[tk][cfg["refit_every"]]
        chosen_by_ticker[tk] = cfg
        idx_context[tk] = bootstrap_indices(len(m), n_boot,
                                            block_len=BLOCK_LEN, seed=0)

    # (6) Cost-convention sensitivity over the primary-floor selections,
    # re-priced on the TRAIN-era frames with the SAME CRN draws.
    cost_sens = cost_sensitivity(masked_by_ticker, chosen_by_ticker,
                                idx_context, on_leverage=args.on_leverage)

    # (7) static-L ladders per viable-at-primary ticker, train + holdout era,
    # each CRN-paired against that config's own draws (train seed 0, holdout
    # seed 1 -- the same seeds sweep_ticker / evaluate_holdout use).
    static_ladders = {}
    for tk, cfg in chosen_by_ticker.items():
        re_k = cfg["refit_every"]
        mtrain = masked_by_ticker[tk][re_k]
        idx_train = idx_context[tk]
        base_train = max_drawdown_from_lr(mtrain["lr_1x"].values[idx_train])
        full = joint_by_refit_by_ticker[tk][re_k]
        hold = full[pd.to_datetime(full["Date"]) > TRAIN_END].reset_index(
            drop=True)
        idx_hold = bootstrap_indices(len(hold), n_boot_holdout,
                                     block_len=BLOCK_LEN, seed=1)
        base_hold = max_drawdown_from_lr(hold["lr_1x"].values[idx_hold])
        static_ladders[tk] = {
            "train": static_ladder_table(mtrain, idx_train, base_train),
            "holdout": static_ladder_table(hold, idx_hold, base_hold)}

    # (8) the ONLY holdout touch: the de-duplicated union of distinct viable
    # per-(floor, ticker) configs (evaluate_holdout dedups + adds the three
    # fixed references). An empty union still evaluates the references.
    distinct_viable = [selection_matrix[fl][tk]["config"]
                       for fl in floors for tk in tickers
                       if selection_matrix[fl][tk].get("viable")]
    holdout = evaluate_holdout(distinct_viable, joint_by_refit_by_ticker,
                               n_boot=n_boot_holdout,
                               excess_floor=args.excess_floor,
                               on_leverage=args.on_leverage)

    # (9) block-126 robustness: full-grid resweep ONCE per ticker, re-masked
    # per floor over the selection matrix (train era only).
    block126 = _block126_sensitivity(selection_matrix, masked_by_ticker,
                                     floors, n_boot=n_boot,
                                     cache_dir=args.cache_dir,
                                     n_restarts=build_params["n_restarts"],
                                     vol_window=build_params["vol_window"],
                                     warmup=build_params["warmup"],
                                     data_dir=build_params["data_dir"],
                                     on_leverage=args.on_leverage)

    # (10) faceted train-era heatmaps at the PRIMARY floor: one PNG per
    # (ticker, cash_arm) -- loop the arms, plot_heatmaps renders one arm each.
    png_paths = []
    for arm in GRID["cash_arm"]:
        png_paths.extend(plot_heatmaps(train_results, primary_cells,
                                       args.out_dir, arm))

    # run provenance: identifies quick vs full runs, the corpus
    # state, the selection-floor union, and the cash-transform version, derived
    # from the frames/config already in memory. NO wall-clock timestamp --
    # regeneration diffs stay deterministic.
    provenance = {
        "mode": "quick" if args.quick else "full",
        "n_boot_sweep": n_boot,
        "n_boot_holdout": n_boot_holdout,
        "n_restarts": n_restarts,
        "block_len": BLOCK_LEN,
        "excess_floor": args.excess_floor,
        "on_leverage": args.on_leverage,
        "selection_floors": ", ".join(str(f) for f in floors),
        "cash_transform_version": CASH_TRANSFORM_VERSION,
        "train_end": TRAIN_END,
        "corpus_max_date": {
            tk: str(max(pd.to_datetime(j["Date"]).max()
                        for j in by_refit.values()).date())
            for tk, by_refit in joint_by_refit_by_ticker.items()},
    }
    irx_sha1 = _read_irx_sha1(args.cache_dir)
    if irx_sha1 is not None:
        provenance["irx_sha1"] = irx_sha1

    report_fp = os.path.join(args.out_dir, "leveraged_gate_sweep_report.md")
    write_report(train_results, selection_matrix, cash_lift, static_ladders,
                 cost_sens, holdout, block126, png_paths, report_fp,
                 excess_floor=args.excess_floor, selection_floors=floors,
                 provenance=provenance)

    print(f"\nSelection matrix (primary floor {primary}):")
    for tk in tickers:
        rec = primary_cells.get(tk, {})
        print(f"  {tk}: viable={rec.get('viable')} "
              f"mode={rec.get('selection_mode')} config={rec.get('config')}")
    print("\nHoldout results:")
    print(holdout.to_string(index=False))
    print(f"\nReport written to {report_fp}")

    return {"train_results": train_results,
            "selection_matrix": selection_matrix, "cash_lift": cash_lift,
            "static_ladders": static_ladders, "cost_sens": cost_sens,
            "holdout": holdout, "block126": block126, "png_paths": png_paths,
            "floors": floors, "provenance": provenance,
            "report_fp": report_fp}


if __name__ == "__main__":
    main()
