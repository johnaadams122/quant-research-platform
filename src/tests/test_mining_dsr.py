import math
import os
import sys

import numpy as np
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import dsr


def test_reproduces_bailey_lopez_de_prado_worked_example():
    # Bailey & Lopez de Prado (2014) "A NUMERICAL EXAMPLE": SR=2.5/yr,
    # T=1250 daily obs (5y), skew=-3, kurtosis=10 (Pearson convention,
    # Normal==3), N=100 trials, SR variance across trials = 0.5/yr.
    # Paper reports SR0~=0.1132, DSR~=0.9004 -- both computed in DAILY
    # units after annualized inputs are converted (SR/sqrt(250),
    # variance/250). Independently hand-recomputed to SR0=0.1131719,
    # DSR=0.9003974 -- tolerance abs=1e-5 is tight enough to catch a
    # sqrt(n) vs sqrt(n-1) transcription error.
    periods_per_year = 250
    sr_daily = 2.5 / math.sqrt(periods_per_year)
    var_daily = 0.5 / periods_per_year

    sr0 = dsr.expected_max_sharpe_ratio(n_trials=100, sr_variance=var_daily)
    assert sr0 == pytest.approx(0.1131719, abs=1e-5)

    result = dsr.deflated_sharpe_ratio(sr=sr_daily, n=1250, skew=-3,
                                        kurtosis=10, n_trials=100,
                                        sr_variance=var_daily)
    assert result == pytest.approx(0.9003974, abs=1e-5)


def test_expected_max_sharpe_ratio_single_trial_is_just_the_mean():
    assert dsr.expected_max_sharpe_ratio(n_trials=1, sr_variance=0.1,
                                          mean_sr=0.05) == pytest.approx(0.05)


def test_expected_max_sharpe_ratio_never_drops_below_mean_sr():
    # Guarded case: for n_trials in roughly (1, 1.28), the unclamped
    # formula returns a value BELOW mean_sr (even negative for
    # mean_sr=0) -- making the "correction" easier to pass than no
    # correction at all. This is reachable on a mining run's FIRST call:
    # num_independent_trials() can land in exactly this range for highly
    # correlated candidates (common in a mining context) with an empty
    # ledger floor. SR0 must be monotone non-decreasing in n_trials and
    # never below mean_sr.
    for n_trials in (1.01, 1.05, 1.1, 1.2, 1.28, 1.5, 2.0, 5.0):
        sr0 = dsr.expected_max_sharpe_ratio(n_trials=n_trials, sr_variance=0.002)
        assert sr0 >= 0.0, f"SR0 went negative at n_trials={n_trials}: {sr0}"
    values = [dsr.expected_max_sharpe_ratio(n_trials=n, sr_variance=0.002)
              for n in (1.0, 1.1, 1.28, 2.0, 10.0, 100.0)]
    assert values == sorted(values)


def test_expected_max_sharpe_ratio_rejects_nonfinite_n_trials():
    # Guards against a NaN n_trials propagating silently through
    # norm.ppf instead of failing closed.
    with pytest.raises(ValueError):
        dsr.expected_max_sharpe_ratio(n_trials=float("nan"), sr_variance=0.002)


def test_expected_max_sharpe_ratio_rejects_bad_sr_variance():
    # Guards against a NaN sr_variance propagating silently to a NaN SR0
    # (no crash, no error) and a negative sr_variance crashing opaquely
    # inside math.sqrt instead of naming the actual problem.
    with pytest.raises(ValueError):
        dsr.expected_max_sharpe_ratio(n_trials=10, sr_variance=float("nan"))
    with pytest.raises(ValueError):
        dsr.expected_max_sharpe_ratio(n_trials=10, sr_variance=-0.001)


def test_probabilistic_sharpe_ratio_at_benchmark_equal_to_sr_is_half():
    # PSR(sr == sr_benchmark) = Z[0] = 0.5, regardless of n/skew/kurtosis.
    result = dsr.probabilistic_sharpe_ratio(sr=0.1, sr_benchmark=0.1, n=100,
                                             skew=0.0, kurtosis=3.0)
    assert result == pytest.approx(0.5)


def test_probabilistic_sharpe_ratio_rejects_kurtosis_fisher_convention_mixup():
    # Guarded case: passing EXCESS kurtosis (Normal==0) instead of the
    # required Pearson convention (Normal==3) either
    # silently corrupts the result or, for positive skew, makes the
    # radicand negative -> unguarded math.sqrt domain error. Any real
    # distribution satisfies kurtosis >= 1 + skew**2 (Pearson); violating
    # that inequality is the tell that Fisher convention was passed by
    # mistake, so it is rejected explicitly instead of crashing opaquely.
    with pytest.raises(ValueError, match="convention"):
        dsr.probabilistic_sharpe_ratio(sr=0.1, sr_benchmark=0.0, n=100,
                                        skew=2.0, kurtosis=1.0)


def test_probabilistic_sharpe_ratio_handles_zero_denominator_boundary():
    # Kurtosis exactly equal to the Pearson floor (1 + skew**2) is a
    # VALID two-point/degenerate-distribution moment combination -- allowed by the
    # kurtosis-convention guard, which only rejects kurtosis STRICTLY
    # below 1 + skew**2. At that exact boundary, if sr == 2/skew too, the
    # radicand (a perfect square) collapses to exactly 0 -- an unguarded
    # ZeroDivisionError instead of the mathematically well-defined
    # limiting PSR (1.0 / 0.0 / 0.5 per the sign of sr - sr_benchmark).
    skew = 2.0
    kurtosis = 1 + skew ** 2  # exact Pearson-floor boundary
    sr = 2.0 / skew  # makes the radicand collapse to exactly 0
    assert dsr.probabilistic_sharpe_ratio(sr=sr, sr_benchmark=sr - 0.1, n=100,
                                           skew=skew, kurtosis=kurtosis) == 1.0
    assert dsr.probabilistic_sharpe_ratio(sr=sr, sr_benchmark=sr + 0.1, n=100,
                                           skew=skew, kurtosis=kurtosis) == 0.0
    assert dsr.probabilistic_sharpe_ratio(sr=sr, sr_benchmark=sr, n=100,
                                           skew=skew, kurtosis=kurtosis) == 0.5


def test_num_independent_trials_single_trial_is_one():
    assert dsr.num_independent_trials([np.array([0.01, 0.02, -0.01])]) == 1.0


def test_num_independent_trials_empty_list_raises():
    with pytest.raises(ValueError):
        dsr.num_independent_trials([])


def test_num_independent_trials_perfectly_correlated_collapses_to_one():
    a = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    b = a * 2.0
    n = dsr.num_independent_trials([a, b, a.copy()])
    assert n == pytest.approx(1.0, abs=1e-9)


def test_num_independent_trials_uncorrelated_stays_near_raw_count():
    rng = np.random.default_rng(7)
    trials = [rng.normal(size=500) for _ in range(5)]
    n = dsr.num_independent_trials(trials)
    assert 3.5 < n < 6.5


def test_num_independent_trials_never_exceeds_raw_count():
    # A negative average pairwise correlation would otherwise push
    # N-hat ABOVE the raw trial count M, which has no valid
    # interpretation (you cannot have more effectively-independent
    # trials than trials actually run) -- rho is floored at 0 for this
    # estimator's purposes.
    a = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    b = -a  # perfectly ANTI-correlated, rho = -1
    n = dsr.num_independent_trials([a, b])
    assert n <= 2.0


def test_num_independent_trials_rejects_constant_return_series():
    # A constant/zero-variance return series (e.g. a candidate with no
    # trades in the sample window) makes Pearson correlation undefined -- np.corrcoef returns
    # NaN for it. Silently dropping those NaNs and defaulting the
    # average to rho=0 would treat the degenerate candidate as perfectly
    # INDEPENDENT of every other trial (N-hat=M, no correlation discount
    # at all) instead of surfacing the bad input. Must raise instead.
    normal = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    constant = np.zeros(5)  # a no-trade candidate's all-zero daily returns
    with pytest.raises(ValueError):
        dsr.num_independent_trials([normal, constant])


def test_num_independent_trials_rejects_nonfinite_return_series():
    # A NaN/inf value inside an otherwise-normal series is NOT caught by
    # the zero-variance guard (np.std of a NaN-containing array is itself
    # NaN, not 0.0) but still makes np.corrcoef emit NaN correlations,
    # which the mean-of-off-diagonal step would otherwise silently drop.
    normal = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    has_nan = np.array([0.01, np.nan, -0.01, 0.03, 0.0])
    with pytest.raises(ValueError):
        dsr.num_independent_trials([normal, has_nan])


def test_num_independent_trials_single_degenerate_trial_still_validates():
    # Guards against an m==1 fast path that returns 1.0 without the
    # validation m>=2 applies: a single zero-trade/constant/NaN candidate
    # must be rejected, not silently passed. Validation runs regardless
    # of trial count.
    with pytest.raises(ValueError):
        dsr.num_independent_trials([np.zeros(5)])


def test_num_independent_trials_rejects_mismatched_lengths():
    # Guards against mismatched trial lengths reaching np.vstack and
    # failing with an opaque NumPy ValueError instead of this module's
    # own actionable message. The docstring requires callers to align
    # trial_returns to the same shape (e.g. via
    # backtest_engine.daily_returns_by_exit_day over a shared in-sample
    # window).
    a = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    b = np.array([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        dsr.num_independent_trials([a, b])
