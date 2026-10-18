"""Vectorized gate/exposure engine for the gate-parameter sweep (numpy only).

exposure_states implements the regime gate's hysteresis ladder: the 1x/3x
enter/exit hysteresis; with overlay=True, the trailing-ratio gate on top
(stepping up to 3x needs trail_ratio > 1, and losing it forces the drop back
to 1x); and, with a finite cash_th, a cash rung below 1x. The two sweep
knobs are a re-entry cooldown (min_dwell) and that overlay on/off switch.
Loops over TIME, vectorizes across bootstrap paths -- (n_paths, T) in one pass.
"""
import numpy as np


def exposure_states(p_bull, trail_ratio, enter_th, exit_th,
                    min_dwell=0, overlay=True, cash_th=-np.inf):
    """Monotone 3-state ladder {0:cash, 1:1x, 2:3x}. Each bar moves at most one
    rung; every transition keys off the START-of-bar level so exactly one fires
    per path/bar. Boundaries cash_th < exit_th < enter_th; exit_th doubles as
    the cash->1x re-entry level (hysteresis for free). Start=1x so cash_th=-inf
    reproduces the frozen Stage-1 exposure + 1 exactly (see _stage1_reference)."""
    p = np.atleast_2d(np.asarray(p_bull, dtype=float))
    r = np.atleast_2d(np.asarray(trail_ratio, dtype=float))
    n, T = p.shape
    gate = (np.nan_to_num(r, nan=0.0) > 1.0) if overlay \
        else np.ones_like(p, dtype=bool)
    level = np.ones(n, dtype=np.int8)
    last_exit = np.full(n, -(10 ** 9), dtype=np.int64)
    out = np.zeros((n, T), dtype=np.int8)
    for t in range(T):
        pt = p[:, t]
        gt = gate[:, t]
        prev = level
        drop3 = (prev == 2) & ((pt < exit_th) | ~gt)
        up3 = (prev == 1) & (pt > enter_th) & gt & (t - last_exit > min_dwell)
        downc = (prev == 1) & (pt < cash_th)
        upc = (prev == 0) & (pt >= exit_th)
        level = prev.copy()
        level[drop3] = 1
        level[up3] = 2
        level[downc] = 0
        level[upc] = 1
        last_exit[drop3] = t
        out[:, t] = level
    return out[0] if np.asarray(p_bull).ndim == 1 else out

# The `t - last_exit > min_dwell` guard IS the semantics: exit at bar t
# blocks entry through bar t+min_dwell, first legal re-entry t+min_dwell+1;
# at min_dwell=0 it forbids only same-bar re-entry, which is plain
# hysteresis. A cooldown counter is easy to get off by one on the exit bar,
# so the hand example below is authoritative over any refactor.
#
# Ladder hand example (enter_th=0.80, exit_th=0.50, cash_th=0.30, dwell=0,
# overlay=False), matching test_ladder_trace_one_rung_and_hysteresis:
#   p:     0.90  0.90  0.40  0.20  0.55  0.90
#   level:  2     2     1     0     1     2     (start level=1, i.e. 1x)
#   t0: p>enter_th, prev=1 -> up3 fires -> 2 (3x)
#   t1: prev=2, p>=exit_th -> holds at 2
#   t2: prev=2, p<exit_th -> drop3 fires -> 1 (1x), last_exit=t2
#   t3: prev=1, p<cash_th -> downc fires -> 0 (cash)
#   t4: prev=0, p>=exit_th (0.55>=0.50) -> upc fires -> 1 (1x); note this
#       bar does NOT also evaluate up3 off level=1 (transitions key off
#       prev/start-of-bar level, so cash->1x and 1x->3x can't both fire
#       on the same bar even though p>exit_th here too)
#   t5: prev=1, p>enter_th -> up3 fires -> 2 (3x)
# Cash-sticky band (test_cash_sticky_band_holds_1x): p held at 0.45 with
# cash_th=0.30 stays at level=1 forever -- 0.45 is above cash_th (no
# downc) and below enter_th (no up3), so the START posture (1x) just
# holds; there's no p-band that reads as "enter cash" on its own, only
# downc from an existing 1x position falling below cash_th.


LEVERAGE = np.array([0.0, 1.0, 3.0])          # index by level {0:cash,1:1x,2:3x}


def strategy_log_returns(exposure, lr_cash, lr_1x, lr_3x, switch_cost_bps=4.0,
                         cost_model="risk_notional", leverage=None):
    """Bar t earns the leg of the posture DECIDED at t-1; start posture is 1x
    (level 1), so bar 0 earns 1x and cash_th=-inf nests Stage 1.

    cost_model="risk_notional" (default, primary model): (bps/2)*|dLev|, Lev in
    {0,1,3} -- 1x<->3x costs 4bps (Stage-1 preserved), cash<->1x 2bps,
    cash<->3x 6bps.
    cost_model="dollar_turnover" (sensitivity model): each EXECUTED rung-change
    costs a flat one-side bps/2 regardless of |dLev| -- so 1x<->3x costs HALF
    what risk_notional charges (2bps vs 4bps at switch_cost_bps=4), while a
    forced two-hop ladder (cash->1x->3x) still pays for BOTH executed hops
    under either model (it cannot teleport straight from cash to 3x).

    leverage (active on-leverage sweep): an explicit [L0,L1,L2] array used for
    the risk_notional |dLev| cost; None (default) uses the module LEVERAGE
    [0,1,3] and is BIT-IDENTICAL to the pre-sweep behavior. Passing
    [0,1,L_on] with a blended L_on risk-on leg (as the lr_3x argument) prices
    the switch cost at that lower leverage -- rung 2 still earns whatever the
    caller supplies as the third leg (the caller substitutes the blend)."""
    e = np.atleast_2d(np.asarray(exposure, dtype=np.int8))
    legs = np.stack([np.atleast_2d(np.asarray(x, dtype=float))
                     for x in (lr_cash, lr_1x, lr_3x)], axis=0)   # (3,n,T)
    n, T = e.shape
    pos = np.concatenate([np.ones((n, 1), dtype=np.int8), e[:, :-1]], axis=1)
    prev = np.concatenate([np.ones((n, 1), dtype=np.int8), pos[:, :-1]], axis=1)
    sel = np.take_along_axis(legs, pos[None, :, :], axis=0)[0]
    lev = LEVERAGE if leverage is None else np.asarray(leverage, dtype=float)
    dlev = np.abs(lev[pos] - lev[prev])
    if cost_model == "risk_notional":
        cost = (switch_cost_bps / 2.0) * dlev / 1e4
    elif cost_model == "dollar_turnover":
        cost = (switch_cost_bps / 2.0) * (dlev > 0).astype(float) / 1e4
    else:
        raise ValueError("cost_model must be 'risk_notional' or "
                         "'dollar_turnover', got %r" % (cost_model,))
    out = sel - cost
    return out[0] if np.asarray(exposure).ndim == 1 else out


def bootstrap_indices(T, n_boot, block_len=63, seed=0):
    """Circular moving-block bootstrap over row indices. Deterministic per
    (T, n_boot, block_len, seed): every sweep point sees the SAME resamples
    (common random numbers), so between-point comparisons are low-noise."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    idx = (starts[:, :, None] + offsets[None, None, :]) % T
    return idx.reshape(n_boot, n_blocks * block_len)[:, :T]
