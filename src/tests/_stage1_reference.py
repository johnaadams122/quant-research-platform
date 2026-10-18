"""FROZEN Stage-1 gate reference. Do not edit. Nesting tests compare the new
3-state engine (cash_arm='never') against these to prove bit-for-bit equality."""
import numpy as np


def ref_exposure_states(p_bull, trail_ratio, enter_th, exit_th,
                        min_dwell=0, overlay=True):
    p = np.atleast_2d(np.asarray(p_bull, dtype=float))
    r = np.atleast_2d(np.asarray(trail_ratio, dtype=float))
    n, T = p.shape
    gate = (np.nan_to_num(r, nan=0.0) > 1.0) if overlay \
        else np.ones_like(p, dtype=bool)
    on = np.zeros(n, dtype=bool)
    last_exit = np.full(n, -(10 ** 9), dtype=np.int64)
    out = np.zeros((n, T), dtype=np.int8)
    for t in range(T):
        exit_now = on & ((p[:, t] < exit_th) | ~gate[:, t])
        last_exit[exit_now] = t
        on = on & ~exit_now
        can_enter = (~on) & (t - last_exit > min_dwell)
        on = on | (can_enter & (p[:, t] > enter_th) & gate[:, t])
        out[:, t] = on
    return out[0] if np.asarray(p_bull).ndim == 1 else out


def ref_strategy_log_returns(exposure, lr_1x, lr_3x, switch_cost_bps=4.0):
    e = np.atleast_2d(np.asarray(exposure, dtype=np.int8))
    l1 = np.atleast_2d(np.asarray(lr_1x, dtype=float))
    l3 = np.atleast_2d(np.asarray(lr_3x, dtype=float))
    pos = np.concatenate([np.zeros((e.shape[0], 1), dtype=np.int8),
                          e[:, :-1]], axis=1)
    prev = np.concatenate([np.zeros((e.shape[0], 1), dtype=np.int8),
                           pos[:, :-1]], axis=1)
    cost = (np.abs(pos - prev) > 0).astype(float) * (switch_cost_bps / 1e4)
    out = pos * l3 + (1 - pos) * l1 - cost
    return out[0] if np.asarray(exposure).ndim == 1 else out
