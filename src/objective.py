"""Objective + risk metrics for the gate-parameter sweep (numpy only).

Fixed objective: the optimizer maximizes EXPECTED LOG-WEALTH
subject to a CVaR cap. Median terminal and Calmar are cross-checks --
computed, reported, never optimized. Arithmetic-mean terminal wealth is
BANNED as an objective (it degenerates to always-max-leverage).
"""
import math

import numpy as np

from gate_engine import exposure_states, strategy_log_returns

TRADING_DAYS = 252


def terminal_log_wealth(lr):
    return np.asarray(lr, dtype=float).sum(axis=-1)


def annualized_log_wealth(lw, T):
    return np.asarray(lw, dtype=float) * TRADING_DAYS / float(T)


def max_drawdown_from_lr(lr):
    """Per-path max drawdown (negative decimal) with the peak seeded at 1.0."""
    lr2 = np.atleast_2d(np.asarray(lr, dtype=float))
    equity = np.exp(np.cumsum(lr2, axis=1))
    equity = np.concatenate([np.ones((equity.shape[0], 1)), equity], axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    dd = (equity / peak - 1.0).min(axis=1)
    return dd[0] if np.asarray(lr).ndim == 1 else dd


def static_leverage_lr(lr_cash, lr_1x, lr_3x, L):
    """Daily-rebalanced constant-effective-leverage log-returns from the actual
    cash/1x/3x legs (the implementable static-L strategy). Endpoints return the
    leg array directly (bit-exact). Convex weights keep log1p finite."""
    lr_cash = np.asarray(lr_cash, dtype=float)
    lr_1x = np.asarray(lr_1x, dtype=float)
    lr_3x = np.asarray(lr_3x, dtype=float)
    if L == 0.0:
        return lr_cash
    if L == 1.0:
        return lr_1x
    if L == 3.0:
        return lr_3x
    Rc, R1, R3 = np.expm1(lr_cash), np.expm1(lr_1x), np.expm1(lr_3x)
    if L <= 1.0:
        w_cash, w_1x, w_3x = 1.0 - L, L, 0.0
    else:
        w_3x = (L - 1.0) / 2.0
        w_1x, w_cash = 1.0 - w_3x, 0.0
    return np.log1p(w_cash * Rc + w_1x * R1 + w_3x * R3)


def _static_weights(L):
    """(w_cash, w_1x, w_3x) constant target weights for static_leverage_lr's L
    convention -- L in [0,1] blends cash/1x, L in (1,3] blends 1x/3x. Priced
    (never used to build the return series -- that goes through
    static_leverage_lr) so the daily-rebalance turnover cost matches the blend
    the strategy actually holds."""
    if L <= 1.0:
        return 1.0 - L, L, 0.0
    w_3x = (L - 1.0) / 2.0
    return 0.0, 1.0 - w_3x, w_3x


def _rebalance_turnover_cost(w, lr_cash, lr_1x, lr_3x, switch_cost_bps):
    """Per-bar cost FRACTION (length T) of maintaining constant target weights
    w=(w_cash, w_1x, w_3x) via daily rebalancing (the SAME flat-bps convention
    as the gate's switch cost). A single-leg weight (one entry == 1.0, the case
    at every endpoint L in {0,1,3}) drifts back to itself EXACTLY, so the cost
    is identically 0.0 at every bar -- not merely small -- which is what keeps
    on_leverage=3.0 byte-identical. Intermediate L pays a nonzero daily
    rebalancing charge on the two diverging legs."""
    w_cash, w_1x, w_3x = w
    Rc = np.expm1(np.asarray(lr_cash, dtype=float))
    R1 = np.expm1(np.asarray(lr_1x, dtype=float))
    R3 = np.expm1(np.asarray(lr_3x, dtype=float))
    Rp = w_cash * Rc + w_1x * R1 + w_3x * R3
    denom = 1.0 + Rp
    d_cash = w_cash - w_cash * (1.0 + Rc) / denom
    d_1x = w_1x - w_1x * (1.0 + R1) / denom
    d_3x = w_3x - w_3x * (1.0 + R3) / denom
    turnover = np.abs(d_cash) + np.abs(d_1x) + np.abs(d_3x)
    return switch_cost_bps / 1e4 * turnover


def static_leverage_lr_net(lr_cash, lr_1x, lr_3x, L, switch_cost_bps):
    """The implementable daily-rebalanced constant-L leg NET of its own daily
    rebalance-turnover cost -- the SAME net leg the passive static-L ladder
    holds (static_ladder_table's lr_L - cost). An active gate that sits in this
    leg and the passive constant-L comparator therefore pay identical blend-
    maintenance economics (no free edge from a gross-vs-net asymmetry). At L in
    {0,1,3} the leg is a single instrument, the turnover cost is identically 0,
    and this returns static_leverage_lr(...) bit-exact -- so on_leverage=3.0
    stays byte-identical."""
    gross = static_leverage_lr(lr_cash, lr_1x, lr_3x, L)
    cost = _rebalance_turnover_cost(_static_weights(L), lr_cash, lr_1x, lr_3x,
                                    switch_cost_bps)
    return gross - cost


def cvar(values, alpha=0.05):
    v = np.sort(np.asarray(values, dtype=float).ravel())
    k = max(1, int(math.ceil(alpha * v.size)))
    return float(v[:k].mean())


def evaluate_point(joint_arrays, idx, enter_th, exit_th, min_dwell, overlay,
                   switch_cost_bps=4.0, *, baseline_maxdd_b=None,
                   cash_th=-np.inf, cost_model="risk_notional",
                   on_leverage=3.0):
    """cost_model (switch-cost model): forwarded to strategy_log_returns for BOTH
    the realized and bootstrap legs below. Default "risk_notional" reproduces
    every prior result unchanged (regression-safety contract).

    on_leverage (active on-leverage sweep): the risk-on rung's effective
    leverage. 3.0 (default) is BIT-IDENTICAL to the hardwired-3x behavior --
    it uses the raw lr_3x/l3b legs, the module LEVERAGE, and leverage=None into
    strategy_log_returns. For L_on in (1,3) the risk-on leg becomes the daily-
    rebalanced 1x/3x blend static_leverage_lr(...,L_on) and the switch cost +
    the lev_turnover side measure are priced at [0,1,L_on]. The excess-drawdown
    is still paired vs 1x (base_dd, lw_1x unchanged)."""
    from gate_engine import LEVERAGE
    p, ratio, lr1, lr3, lrcash = (np.asarray(a, dtype=float)
                                  for a in joint_arrays)
    T = p.size

    # Risk-on rung leg + leverage array. 3.0 short-circuits to the exact
    # pre-sweep path (raw lr_3x, module LEVERAGE, leverage=None) so every prior
    # number is reproduced bit-for-bit; L_on<3 substitutes the blend leg.
    if on_leverage == 3.0:
        on_leg, lev_arr, lev_kw = lr3, LEVERAGE, None
    else:
        on_leg = static_leverage_lr_net(lrcash, lr1, lr3, on_leverage,
                                        switch_cost_bps)
        lev_arr = np.array([0.0, 1.0, float(on_leverage)])
        lev_kw = lev_arr

    expo = exposure_states(p, ratio, enter_th, exit_th, min_dwell, overlay,
                           cash_th=cash_th)
    lr_s = strategy_log_returns(expo, lrcash, lr1, on_leg, switch_cost_bps,
                                cost_model=cost_model, leverage=lev_kw)
    realized_lw_ann = float(annualized_log_wealth(terminal_log_wealth(lr_s), T))
    realized_maxdd = float(max_drawdown_from_lr(lr_s))
    cagr = math.expm1(realized_lw_ann)
    realized_calmar = cagr / abs(realized_maxdd) if realized_maxdd < 0 else float("inf")
    pos = np.concatenate([[1], expo.astype(int)[:-1]])       # executed posture
    lev = lev_arr[pos]
    switches = int((np.abs(np.diff(np.concatenate([[1], pos]))) > 0).sum())
    lev_turnover = float(np.abs(np.diff(np.concatenate([[1.0], lev]))).sum())
    cash_occupancy = float((pos == 0).mean())

    pb, rb, l1b, l3b, lcb = p[idx], ratio[idx], lr1[idx], lr3[idx], lrcash[idx]
    on_leg_b = (l3b if on_leverage == 3.0
                else static_leverage_lr_net(lcb, l1b, l3b, on_leverage,
                                            switch_cost_bps))
    expo_b = exposure_states(pb, rb, enter_th, exit_th, min_dwell, overlay,
                             cash_th=cash_th)
    # a 1-D idx (single chronological path) makes exposure_states return a
    # 1-D array (see gate_engine.exposure_states' own ndim-preserving
    # return); atleast_2d normalizes to (n_paths, T) before the row-wise
    # cash-touch reduction, matching max_drawdown_from_lr's own pattern.
    expo_b2 = np.atleast_2d(expo_b)
    pos_b = np.concatenate([np.ones((expo_b2.shape[0], 1), dtype=int),
                            expo_b2.astype(int)[:, :-1]], axis=1)
    cash_zero_use_freq = float((pos_b == 0).any(axis=1).mean())
    cash_zero_use_freq = 1.0 - cash_zero_use_freq            # never-touch fraction
    lr_b = strategy_log_returns(expo_b, lcb, l1b, on_leg_b, switch_cost_bps,
                                cost_model=cost_model, leverage=lev_kw)
    lw_b = terminal_log_wealth(lr_b)
    lw_1x = terminal_log_wealth(l1b)

    maxdd_b = np.atleast_1d(max_drawdown_from_lr(lr_b))
    n_paths = maxdd_b.shape[0]
    if baseline_maxdd_b is None:
        base_dd = np.atleast_1d(max_drawdown_from_lr(l1b))
    else:
        base_dd = np.asarray(baseline_maxdd_b, dtype=float)
        if base_dd.shape != (n_paths,):
            raise ValueError(
                "baseline_maxdd_b shape %s != (%d,); it must be "
                "max_drawdown_from_lr(lr1[idx]) for the SAME idx"
                % (base_dd.shape, n_paths))
        if not np.isfinite(base_dd).all():
            raise ValueError("baseline_maxdd_b contains non-finite values")
    excess_dd = maxdd_b - base_dd
    if not np.isfinite(excess_dd).all():
        raise ValueError("excess drawdown vector contains non-finite values")
    k = max(1, int(math.ceil(0.05 * excess_dd.size)))
    tail = np.argsort(excess_dd, kind="stable")[:k]

    return {
        "expected_lw_ann": float(annualized_log_wealth(lw_b, T).mean()),
        "cvar5_maxdd": cvar(maxdd_b, 0.05),
        "cvar5_lw_ann": cvar(annualized_log_wealth(lw_b, T), 0.05),
        "median_terminal_ratio_vs_1x": float(np.median(np.exp(lw_b - lw_1x))),
        "realized_lw_ann": realized_lw_ann,
        "realized_maxdd": realized_maxdd,
        "realized_calmar": realized_calmar,
        "switches_realized": switches,
        "lev_turnover": lev_turnover,
        "cash_occupancy": cash_occupancy,
        "cash_zero_use_freq": cash_zero_use_freq,
        "cvar5_excess_dd": cvar(excess_dd, 0.05),
        "excess_tail_k": int(k),
        "excess_tail_worst": float(excess_dd[tail].min()),
        "excess_tail_median_maxdd_strategy": float(np.median(maxdd_b[tail])),
        "excess_tail_worst_maxdd_strategy": float(maxdd_b[tail].min()),
        "excess_tail_median_maxdd_1x": float(np.median(base_dd[tail])),
        "excess_tail_worst_maxdd_1x": float(base_dd[tail].min()),
    }
