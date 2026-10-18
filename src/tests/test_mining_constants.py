import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import constants


def test_dsr_threshold_in_valid_probability_range():
    assert 0.0 < constants.DSR_PASS_THRESHOLD < 1.0


def test_holdout_window_is_six_months():
    assert constants.HOLDOUT_WINDOW_MONTHS == 6


def test_candidate_count_target_is_ordered_range():
    assert constants.CANDIDATE_COUNT_TARGET_MIN < constants.CANDIDATE_COUNT_TARGET_MAX


def test_stcg_tax_rate_is_a_plausible_fraction():
    assert 0.0 < constants.STCG_TAX_RATE < 1.0


def test_budget_ceiling_is_positive():
    assert constants.DEFAULT_BUDGET_CEILING_TOKENS > 0


def test_min_trades_for_scoring_is_a_sane_small_positive_int():
    assert 2 <= constants.MIN_TRADES_FOR_SCORING <= 100


def test_default_sr_variance_floor_is_positive():
    assert constants.DEFAULT_SR_VARIANCE_FLOOR > 0.0


def test_daily_returns_correlation_window_is_a_sane_positive_day_count():
    assert 100 <= constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS <= 5000


def test_near_miss_margin_is_a_small_positive_fraction_of_the_dsr_scale():
    # DSR lives on [0, 1] -- the margin must be a small positive slice of
    # that scale, strictly narrower than the gap between the bar and zero.
    assert 0.0 < constants.NEAR_MISS_DSR_MARGIN < constants.DSR_PASS_THRESHOLD

def test_idle_sweep_apy_is_the_pinned_value():
    """The flat brokerage-sweep yield, pinned EXACTLY rather than by
    range. A silent drift here moves every account curve in the pipeline,
    and moves it only for candidates that sit out -- so it would re-rank
    patient strategies against high-turnover ones without any test failing.
    Changing it is a deliberate modelling choice, not a tuning step."""
    assert constants.IDLE_SWEEP_APY == 0.001


def test_max_pre_inception_burn_in_years_is_a_sane_small_positive_int():
    """The RULE is to cap expanding-window burn-in; the specific value is a
    proposal -- see the constant's own docstring. Range-checked, not pinned
    exactly, so this test does not silently bless a still-open number as final."""
    assert 1 <= constants.MAX_PRE_INCEPTION_BURN_IN_YEARS <= 30
