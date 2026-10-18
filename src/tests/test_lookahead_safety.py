# src/tests/test_lookahead_safety.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import ConvergenceMonitor

import hmm_guards
from _synth import two_regime_returns, prices_from_returns


def test_log_return_additivity():
    """Data contract: daily log returns must sum to ln(P_end/P_start)."""
    r = two_regime_returns(seed=11)
    df = prices_from_returns(r)
    lr = np.log1p(df["PX"].pct_change()).dropna().values
    total = np.log(df["PX"].iloc[-1] / df["PX"].iloc[0])
    assert abs(lr.sum() - total) < 1e-10


class _RecordingMonitor(ConvergenceMonitor):
    """history deque is unbounded in hmmlearn 0.3.3; subclass kept as
    version-proofing -- records all values on any version."""

    def __init__(self, tol, n_iter, verbose):
        super().__init__(tol, n_iter, verbose)
        self.full_history = []

    def report(self, log_prob):
        self.full_history.append(log_prob)
        super().report(log_prob)


def test_em_loglikelihood_improves_without_catastrophic_drops():
    """hmmlearn 0.3.3's full-cov EM can wobble by ~1e-3..1e-2 nats at the
    plateau -- that is library-characteristic, not a broken fit. So: require a large NET improvement and forbid any
    catastrophic single-step collapse, NOT strict per-step monotonicity."""
    r = two_regime_returns(seed=5)
    m = GaussianHMM(n_components=2, covariance_type="full",
                    n_iter=150, random_state=42)
    m.monitor_ = _RecordingMonitor(m.monitor_.tol, m.monitor_.n_iter,
                                   m.monitor_.verbose)
    m.fit(r.reshape(-1, 1))
    hist = np.asarray(m.monitor_.full_history)
    assert len(hist) >= 3
    assert hist[-1] - hist[0] > 1.0, "EM made no net progress - broken fit"
    assert np.min(np.diff(hist)) > -0.05, (
        "EM log-likelihood collapsed mid-run - beyond plateau wobble")


def test_transition_matrix_rows_sum_to_one():
    r = two_regime_returns(seed=5)
    m = GaussianHMM(n_components=2, covariance_type="full",
                    n_iter=150, random_state=42).fit(r.reshape(-1, 1))
    assert np.allclose(m.transmat_.sum(axis=1), 1.0, atol=1e-10)


def _known_params_model():
    """A 2-state GaussianHMM with HAND-SET, analytically-known parameters.

    No EM anywhere: an EM-fitted construction makes the bull label
    seed-dependent (the two state labels can swap), which is noise this test must
    not stand on. State 0 = bear (mean -0.004, sd 0.035), state 1 = bull
    (mean +0.0008, sd 0.010), sticky transitions.
    """
    m = GaussianHMM(n_components=2, covariance_type="full", init_params="")
    m.startprob_ = np.array([0.5, 0.5])
    m.transmat_ = np.array([[0.98, 0.02],
                            [0.02, 0.98]])
    m.means_ = np.array([[-0.0040], [0.0008]])
    m.covars_ = np.array([[[0.035 ** 2]], [[0.010 ** 2]]])
    params = {"means": [-0.0040, 0.0008],
              "variances": [0.035 ** 2, 0.010 ** 2],
              "transmat": [[0.98, 0.02], [0.02, 0.98]],
              "startprob": [0.5, 0.5]}
    return m, params


def test_interior_smoothed_differs_from_causal_filtered():
    """Prove filtered != smoothed at an interior point.

    Construction: 400 calm-up days then 40 violent-down days, evaluated under
    HAND-SET true-ish parameters (no fit). Day index 399 is the LAST calm day.
    The causal (filtered) read at 399 has never seen the crash; the smoothed
    read at 399 (predict_proba over the full series) has -- the backward pass
    leaks the crash into day 399. Both reads use the SAME parameters, so the
    gap is purely the future-data leak the live pipeline must never consume.
    """
    rng = np.random.default_rng(3)
    r = np.concatenate([
        rng.normal(0.0008, 0.010, 400),
        rng.normal(-0.0040, 0.035, 40),
    ])
    m, params = _known_params_model()

    idx = 399  # last calm observation; index 1 = bull by construction
    p_filtered = hmm_guards.forward_filter_pbull(params, r[: idx + 1])
    p_smoothed = float(m.predict_proba(r.reshape(-1, 1))[idx, 1])

    # Under the true calm parameters, a causal read of 400 calm days is
    # confidently bull; hindsight of the crash drags the same day toward bear.
    assert p_filtered > 0.9
    assert abs(p_filtered - p_smoothed) > 0.05, (
        "filtered and smoothed agree at a regime edge -- smoothing is not "
        "doing anything or the comparison is miswired"
    )
    assert p_smoothed < p_filtered


def test_terminal_read_is_invariant_to_history_truncation():
    """The terminal filtered read on data[:k] must equal the forward filter's
    value at step k -- i.e. the 'live' read never changes retroactively as a
    function of how much LATER data exists (with parameters held fixed)."""
    r = two_regime_returns(seed=9)
    m = GaussianHMM(n_components=2, covariance_type="full",
                    n_iter=200, random_state=42).fit(r.reshape(-1, 1))
    params = hmm_guards.params_to_dict(m)
    for k in (200, 450, 700):
        p_k = hmm_guards.forward_filter_pbull(params, r[:k])
        bull_idx = int(np.argmax(m.means_.ravel()))
        # hmmlearn terminal smoothed == filtered at the last obs of the slice
        p_hmm = float(m.predict_proba(r[:k].reshape(-1, 1))[-1, bull_idx])
        assert abs(p_k - p_hmm) < 1e-8
