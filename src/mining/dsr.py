"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) for the mining
pipeline's multiple-testing correction (applied at the scoring stage).

All Sharpe ratios and variances in this module must be in the RETURN
SERIES' OWN FREQUENCY (e.g. daily), never annualized -- mixing annualized
and per-period units is the single most common DSR mistake (confirmed
against the paper's own worked example, which only reproduces if
annualized inputs are converted to daily before calling in). Callers are
responsible for annualizing/de-annualizing outside this module.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI = 0.577_215_664_901_532_9


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n: int,
                                skew: float, kurtosis: float) -> float:
    """PSR: P(true Sharpe > sr_benchmark | observed sr, n obs, skew,
    kurtosis). Bailey & Lopez de Prado (2014), Eq. 2's inner Z[...] with
    an arbitrary benchmark. `kurtosis` uses the PEARSON convention
    (Normal == 3.0), not excess kurtosis (Normal == 0.0) -- pass
    `scipy.stats.kurtosis(returns, fisher=False)`, or add 3 to an excess
    value.

    Validates `kurtosis >= 1 + skew**2` (a mathematical identity that
    holds EXACTLY for any two moments computed from the same empirical
    distribution under a CONSISTENT plug-in/moment convention -- provably
    forces the denominator's radicand non-negative: denom**2 >= 1 -
    skew*sr + skew**2*sr**2/4 = (1 - skew*sr/2)**2 >= 0) and raises
    ValueError if violated, rather than letting an inconsistent input
    either silently corrupt the result or crash later on an unguarded
    sqrt of a negative radicand.

    Two DISTINCT ways to violate this are both named in the error: (1) passing EXCESS
    kurtosis (Fisher convention, Normal==0) instead of Pearson (Normal==3)
    -- fix: add 3. (2) passing skew and kurtosis computed with DIFFERENT
    bias conventions (e.g. `scipy.stats.skew(x, bias=False)` paired with
    `scipy.stats.kurtosis(x, fisher=False)`, which defaults to
    `bias=True`) -- which violates this guard in the large majority of
    ordinary two-outcome trade-return samples, because the identity only
    holds when BOTH moments come from the same convention. Fix: use
    `bias=True` (the plug-in/moment estimator) for BOTH skew and
    kurtosis, not a bias-corrected skew paired with a plug-in kurtosis."""
    if n < 2:
        raise ValueError(f"probabilistic_sharpe_ratio needs n >= 2, got {n}")
    if kurtosis < 1 + skew ** 2:
        raise ValueError(
            f"kurtosis={kurtosis} is impossible for skew={skew} under a "
            f"consistent moment convention (requires kurtosis >= "
            f"1 + skew**2 = {1 + skew ** 2}). Two known causes: (1) EXCESS "
            f"kurtosis (Fisher, Normal==0) was passed instead of Pearson "
            f"(Normal==3) -- add 3 and retry; (2) skew and kurtosis were "
            f"computed with DIFFERENT bias conventions (e.g. "
            f"scipy.stats.skew(x, bias=False) with "
            f"scipy.stats.kurtosis(x, fisher=False) -- the latter defaults "
            f"to bias=True) -- recompute BOTH with bias=True.")
    denom = math.sqrt(1 - skew * sr + ((kurtosis - 1) / 4.0) * sr ** 2)
    if denom == 0.0:
        # Kurtosis
        # exactly at the Pearson floor (kurtosis == 1 + skew**2) is a
        # VALID two-point/degenerate-distribution moment combination --
        # allowed by the guard above, which only rejects kurtosis STRICTLY
        # less than the floor. At that exact boundary, the radicand is the
        # perfect square (1 - skew*sr/2)**2 (see the guard's own docstring
        # proof), which collapses to exactly 0 when sr == 2/skew -- a
        # realistic small-sample mining input, not an invalid one. Rather
        # than an unguarded ZeroDivisionError, return the mathematically
        # well-defined limiting value: as denom -> 0+, z -> +-inf per the
        # sign of (sr - sr_benchmark), so PSR -> 1.0 / 0.0 / 0.5.
        if sr > sr_benchmark:
            return 1.0
        if sr < sr_benchmark:
            return 0.0
        return 0.5
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe_ratio(n_trials: float, sr_variance: float,
                               mean_sr: float = 0.0) -> float:
    """E[max SR] under `n_trials` (possibly non-integer, per Eq. 9)
    independent trials whose SR estimates are Normal(mean_sr,
    sr_variance). Bailey & Lopez de Prado (2014), Eq. 1.

    Clamped to never fall below `mean_sr`: the raw formula's weighted
    z-score can go NEGATIVE for `n_trials` in roughly (1, 1.28) (verified
    by hand) --
    a case reachable on a mining run's very first call, since
    `num_independent_trials` can land exactly there for highly
    correlated candidates with an empty ledger floor. An uncorrected
    negative SR0 would make the multiple-testing "correction" easier to
    clear than no correction at all, which is the opposite of its
    purpose.

    Raises ValueError if `n_trials` is non-finite or `sr_variance` is
    non-finite/negative: a NaN
    `sr_variance` from a degenerate/insufficient trial-Sharpe estimate
    would otherwise silently propagate to a NaN SR0 and then a NaN DSR
    instead of failing closed, and a negative `sr_variance` would crash
    opaquely inside `math.sqrt` instead of naming the actual problem."""
    if not math.isfinite(n_trials):
        raise ValueError(
            f"expected_max_sharpe_ratio: n_trials must be finite, got "
            f"{n_trials!r}")
    if not math.isfinite(sr_variance) or sr_variance < 0:
        raise ValueError(
            f"expected_max_sharpe_ratio: sr_variance must be finite and "
            f">= 0, got {sr_variance!r}")
    if n_trials <= 1:
        return mean_sr
    sigma_sr = math.sqrt(sr_variance)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    max_z = (1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    return mean_sr + sigma_sr * max(0.0, max_z)


def num_independent_trials(trial_returns: Sequence[np.ndarray]) -> float:
    """Estimated number of INDEPENDENT trials N-hat from M actual
    (possibly correlated) trials, via Bailey & Lopez de Prado (2014)
    Appendix A.3, Eq. 9: N-hat = rho-hat + (1 - rho-hat) * M, where
    rho-hat is the average pairwise correlation among the M trials'
    return series. `trial_returns` must all be aligned to the same
    index/length (e.g. via backtest_engine.daily_returns_by_exit_day over
    a shared in-sample window) for correlation to be meaningful. A single
    trial (M=1) returns 1.0. Raises ValueError on an empty list -- zero
    is not a valid trial count and would silently disable the
    correction downstream (expected_max_sharpe_ratio's n_trials<=1 guard
    returns mean_sr, i.e. no deflation at all).

    `rho_hat` is floored at 0 before use: a negative average pairwise
    correlation would otherwise push N-hat ABOVE the raw count M, which
    has no valid interpretation under this estimator (you cannot have
    more effectively-independent trials than trials actually run).

    Raises ValueError if any trial's return series is empty, constant
    (zero variance, including all-zero -- e.g. a candidate with no
    trades in the sample window), or contains a non-finite value (NaN /
    +-inf), since Pearson correlation is mathematically undefined in
    all three cases and `np.corrcoef` returns NaN for them. Silently
    dropping these NaNs and defaulting the average to rho=0 would treat
    a degenerate candidate as perfectly independent of every other
    trial (N-hat=M, no correlation discount at all) instead of
    surfacing the bad input -- callers must clean/exclude zero-trade,
    zero-variance, or NaN-containing candidates from the trial set
    before calling.

    Honesty note: this project pairs the
    correlation estimate with backtest_engine.daily_returns_by_exit_day,
    which is zero everywhere except each trade's exit date. Two
    candidates with genuinely similar mechanisms but offset trade timing
    can measure near-zero correlation under this proxy, pulling N-hat
    back toward the raw count M -- i.e. toward the naive family-count
    approach this estimator was adopted specifically to improve on. This
    is a conservative failure direction (never UNDER-corrects), not a
    dangerous one, but it means the correlation correction may deliver
    less benefit in practice than the formula alone suggests."""
    m = len(trial_returns)
    if m == 0:
        raise ValueError("num_independent_trials requires at least 1 trial")
    arrays = [np.asarray(r) for r in trial_returns]
    shapes = {r.shape for r in arrays}
    if len(shapes) > 1:
        # Mismatched
        # lengths reach np.vstack and fail with an opaque NumPy
        # ValueError instead of this module's own actionable message --
        # this docstring already requires callers to align trial_returns
        # (e.g. via backtest_engine.daily_returns_by_exit_day over a
        # shared in-sample window), so a mismatch is a caller bug worth
        # naming explicitly.
        raise ValueError(
            f"num_independent_trials: all trial return series must share "
            f"the same shape (e.g. via "
            f"backtest_engine.daily_returns_by_exit_day over a shared "
            f"in-sample window) -- got shapes {sorted(shapes)}")
    degenerate = [i for i, r in enumerate(arrays)
                  if r.size == 0 or not np.all(np.isfinite(r))
                  or np.std(r) == 0.0]
    if degenerate:
        raise ValueError(
            f"num_independent_trials: trial(s) at index {degenerate} have "
            f"an empty, non-finite (NaN/inf), or constant return series "
            f"(e.g. a candidate with no trades in the sample window) -- "
            f"pairwise correlation is undefined for any of these. Clean "
            f"or exclude such candidates from the trial set before "
            f"calling; do not let them silently default to an "
            f"assumed-independent (rho=0) correlation.")
    if m == 1:
        return 1.0
    matrix = np.corrcoef(np.vstack(arrays))
    off_diag = matrix[~np.eye(m, dtype=bool)]
    off_diag = off_diag[~np.isnan(off_diag)]
    rho = float(np.mean(off_diag)) if len(off_diag) else 0.0
    rho = max(0.0, rho)
    return rho + (1.0 - rho) * m


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurtosis: float,
                           n_trials: float, sr_variance: float,
                           mean_sr: float = 0.0) -> float:
    """DSR = PSR evaluated at the benchmark SR0 = E[max SR] under the
    null of zero true skill across n_trials independent trials. Bailey &
    Lopez de Prado (2014), Eq. 2 -- "DSR is a PSR where the rejection
    threshold is adjusted to reflect the multiplicity of trials"."""
    sr0 = expected_max_sharpe_ratio(n_trials, sr_variance, mean_sr)
    return probabilistic_sharpe_ratio(sr, sr0, n, skew, kurtosis)
