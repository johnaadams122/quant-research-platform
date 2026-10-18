"""Pure-numpy guards for the HMM regime signal.

Three jobs, deliberately dependency-light (numpy only) so this module is
importable in every environment:
  1. params_to_dict / save_params / load_params -- freeze a fitted GaussianHMM's
     parameters to JSON (the frozen-model fallback).
  2. forward_filter_pbull -- hand-rolled, scaled forward algorithm. Used (a) as
     the frozen-fallback inference path and (b) as the machine-checkable proof
     that the terminal predict_proba read is causal (test suite).
  3. fit_is_sane -- hard-failure detection for a refit (degenerate/collapsed
     fits must fall back, never silently emit).
"""
import json
import os

import numpy as np

# fit_is_sane thresholds (daily-return units)
MIN_MEAN_GAP = 5e-4     # bull/bear daily means closer than 5 bps = no real regimes
MIN_STATE_FRAC = 0.02   # each state must own >= 2% of decoded days
MAX_STAY = 0.9999       # a diagonal this high is an absorbing state


def params_to_dict(model):
    """Extract (and relabel: index 1 = bull = higher mean) HMM parameters."""
    means = model.means_.ravel().astype(float)
    variances = np.array(
        [float(np.atleast_2d(c)[0, 0]) for c in model.covars_], dtype=float
    )
    transmat = np.asarray(model.transmat_, dtype=float)
    startprob = np.asarray(model.startprob_, dtype=float)
    if means[0] > means[1]:
        order = [1, 0]
        means = means[order]
        variances = variances[order]
        transmat = transmat[np.ix_(order, order)]
        startprob = startprob[order]
    return {
        "means": means.tolist(),
        "variances": variances.tolist(),
        "transmat": transmat.tolist(),
        "startprob": startprob.tolist(),
    }


def save_params(params, fp, asof):
    payload = {"schema": 1, "asof": asof, "params": params}
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, fp)


def load_params(fp):
    """Return the frozen params dict, or None if missing/corrupt (fail soft).

    "Corrupt" includes structurally-valid JSON whose params block is broken:
    missing fields, nulls, wrong shapes, non-numeric or non-finite entries.
    Every field must convert cleanly to a float array of the exact expected
    shape with all entries finite, or the whole load fails closed -- a
    poisoned frozen-model file must never reach the forward filter.
    """
    try:
        with open(fp, "r", encoding="ascii") as f:
            payload = json.load(f)
        params = payload["params"]
        # full-schema validation before anyone trusts it
        means = np.asarray(params["means"], dtype=float)
        variances = np.asarray(params["variances"], dtype=float)
        transmat = np.asarray(params["transmat"], dtype=float)
        startprob = np.asarray(params["startprob"], dtype=float)
        assert means.shape == (2,)
        assert variances.shape == (2,)
        assert transmat.shape == (2, 2)
        assert startprob.shape == (2,)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(variances))
        assert np.all(np.isfinite(transmat))
        assert np.all(np.isfinite(startprob))
        return params
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        return None


def _gauss_pdf(x, means, variances):
    return np.exp(-0.5 * (x - means) ** 2 / variances) / np.sqrt(
        2.0 * np.pi * variances
    )


def forward_filter_pbull(params, log_returns):
    """Filtered P(state=bull | r_1..r_T) via the scaled forward algorithm.

    Uses ONLY past-and-present observations by construction -- this is the
    causal quantity a regime signal is allowed to act on.
    """
    means = np.asarray(params["means"], dtype=float)
    variances = np.asarray(params["variances"], dtype=float)
    transmat = np.asarray(params["transmat"], dtype=float)
    startprob = np.asarray(params["startprob"], dtype=float)
    x = np.asarray(log_returns, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("forward_filter_pbull: empty return series")

    alpha = startprob * _gauss_pdf(x[0], means, variances)
    s = alpha.sum()
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("forward_filter_pbull: zero/invalid emission at t=0")
    alpha /= s
    for xt in x[1:]:
        alpha = (alpha @ transmat) * _gauss_pdf(xt, means, variances)
        s = alpha.sum()
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("forward_filter_pbull: underflow/invalid emission")
        alpha /= s
    return float(alpha[1])


def fit_is_sane(model, states):
    """Hard-failure screen for a fresh refit. Returns (ok, reasons).

    The finiteness check comes FIRST and short-circuits: every threshold
    comparison below evaluates False on NaN (NaN-blind), so without this a
    NaN-poisoned fit would sail through as "sane" -- the exact fail-open this
    guard exists to prevent.
    """
    variances_arr = np.array(
        [float(np.atleast_2d(c)[0, 0]) for c in model.covars_], dtype=float
    )
    all_params = np.concatenate([
        np.ravel(model.means_).astype(float),
        variances_arr,
        np.ravel(model.transmat_).astype(float),
        np.ravel(model.startprob_).astype(float),
    ])
    if not np.all(np.isfinite(all_params)):
        return (False, ["nonfinite_params"])

    reasons = []
    mu = model.means_.ravel()
    if abs(float(mu[1] - mu[0])) < MIN_MEAN_GAP:
        reasons.append("means_too_close")
    frac_bull = float(np.mean(np.asarray(states) == 1))
    if not (MIN_STATE_FRAC <= frac_bull <= 1.0 - MIN_STATE_FRAC):
        reasons.append("state_collapse")
    tm = np.asarray(model.transmat_, dtype=float)
    if np.abs(tm.sum(axis=1) - 1.0).max() > 1e-8:
        reasons.append("transmat_rows")
    if (np.diag(tm) >= MAX_STAY).any():
        reasons.append("absorbing_state")
    variances = np.array([float(np.atleast_2d(c)[0, 0]) for c in model.covars_])
    if (variances <= 0.0).any():
        reasons.append("nonpositive_variance")
    return (len(reasons) == 0, reasons)
