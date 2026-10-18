"""Tests for src/mining/batch5/move_vix_divergence_of_fear_gauges_derisking.py.

PRE-REGISTERED RULE (every constant is fixed a priori in the catalog entry;
nothing below may be tuned after looking at a backtest)::

    zM_t, zV_t = expanding z-scores of ^MOVE and ^VIX, statistics taken over
                 the observations STRICTLY BEFORE t, at least 756 of them
    DFG_t      = (zM_t - zV_t) / sqrt(2)
    S_t        = mean of the last 10 DFG observations ending at t
    at each scheduled weekly decision date F (the Friday, or the last
    scheduled session of the week when Friday is a scheduled holiday):
        derisk = S_F > 0.5             (strict)
        target = SHY if derisk else SPY   # the engine fills at the NEXT
                                          # session's open

The tests are grouped: contract, ex-ante calendar, pure building blocks,
engine-level behaviour against an INDEPENDENT oracle, look-ahead / hold-out
blindness (with power checks that inject a deliberate look-ahead), degraded
data, and the fetched-frame loader (exercised on synthetic fetches only).
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import data_pipeline  # noqa: E402
import run_batch3_mining as batch3  # noqa: E402
from mining import abi1_driver, catalog as mining_catalog, rotation_engine  # noqa: E402
from mining.batch5 import move_vix_divergence_of_fear_gauges_derisking as mod  # noqa: E402

FAMILY_ID = "move_vix_divergence_of_fear_gauges_derisking"
RISK, DEFENSIVE = "SPY", "SHY"
MOVE, VIX = "^MOVE", "^VIX"

# Pre-registered constants, restated here on purpose.
MIN_HISTORY = 756
SMOOTHING = 10
THRESHOLD = 0.5

CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
THIS_FILE = os.path.abspath(__file__)

# A dollar-sign-then-digit pattern, written so this file does not contain one.
_DOLLAR_DIGIT = re.compile(r"\$\s?\d")

# Every scheduled full-day NYSE closure on a weekday 2010-2016, LISTED BY HAND
# (from the exchange's published holiday rules; the exchange_calendars XNYS
# calendar differs only by the two unscheduled Hurricane Sandy days, which are
# deliberately absent).
_HAND_CLOSURES = pd.DatetimeIndex(pd.to_datetime("""
2010-01-01 2010-01-18 2010-02-15 2010-04-02 2010-05-31 2010-07-05 2010-09-06
2010-11-25 2010-12-24
2011-01-17 2011-02-21 2011-04-22 2011-05-30 2011-07-04 2011-09-05 2011-11-24
2011-12-26
2012-01-02 2012-01-16 2012-02-20 2012-04-06 2012-05-28 2012-07-04 2012-09-03
2012-11-22 2012-12-25
2013-01-01 2013-01-21 2013-02-18 2013-03-29 2013-05-27 2013-07-04 2013-09-02
2013-11-28 2013-12-25
2014-01-01 2014-01-20 2014-02-17 2014-04-18 2014-05-26 2014-07-04 2014-09-01
2014-11-27 2014-12-25
2015-01-01 2015-01-19 2015-02-16 2015-04-03 2015-05-25 2015-07-03 2015-09-07
2015-11-26 2015-12-25
2016-01-01 2016-01-18 2016-02-15 2016-03-25 2016-05-30 2016-07-04 2016-09-05
2016-11-24 2016-12-26
""".split()))


# ------------------------------------------------------------------ helpers

def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    """Weekdays minus the hand-listed closures above: a synthetic calendar
    with exactly the sessions the real exchange had (2010-2016)."""
    days = pd.bdate_range(start, end)
    return days[~days.isin(_HAND_CLOSURES)]


def _ohlc_from_close(close, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _entry() -> dict:
    """This family's entry in the shared catalog."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    matches = [e for e in entries if e.get("family_id") == FAMILY_ID]
    assert len(matches) == 1, f"expected one catalog entry for {FAMILY_ID}"
    return matches[0]


# -------------------------------------------------------------- contract

def test_module_constants_match_the_preregistered_spec():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]        # [risk/default, defensive]
    assert mod.SIGNAL == [MOVE, VIX]
    assert mod.BENCHMARK == "SPY"
    assert mod.SNAPSHOTS == {}
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert not isinstance(mod.EXPECTED_MIN_TRADES, bool)
    assert mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_target_fn)


def test_preregistered_constants_are_exactly_756_10_and_0_5():
    assert mod.PARAMETERS == {
        "bond_vol_ticker": MOVE,
        "equity_vol_ticker": VIX,
        "zscore_window": "expanding_ending_t_minus_1",
        "zscore_min_history_sessions": MIN_HISTORY,
        "smoothing_sessions": SMOOTHING,
        "derisk_threshold": THRESHOLD,
        "defensive_proxy": DEFENSIVE,
        "rebalance": "weekly_friday_close",
    }
    assert isinstance(mod.PARAMETERS["zscore_min_history_sessions"], int)
    assert isinstance(mod.PARAMETERS["smoothing_sessions"], int)
    assert isinstance(mod.PARAMETERS["derisk_threshold"], float)
    # The values the code actually USES must be the pre-registered ones, not
    # just the dict that documents them.
    assert mod._MIN_HISTORY == MIN_HISTORY
    assert mod._SMOOTHING_SESSIONS == SMOOTHING
    assert mod._DERISK_THRESHOLD == THRESHOLD


# ------------------------------------------------------ ex-ante calendar

def test_scheduled_closures_equal_the_hand_listed_nyse_closures_2010_2016():
    weekdays = pd.bdate_range("2010-01-01", "2016-12-31")
    closed = [d for d in weekdays if mod.is_scheduled_closure(d)]
    assert list(pd.DatetimeIndex(closed)) == list(_HAND_CLOSURES)


@pytest.mark.parametrize("day, closed", [
    ("2021-12-24", True),    # Christmas on a Saturday: the Friday is closed
    ("2021-12-31", False),   # New Year's Day on a Saturday: NOT observed Friday
    ("2027-12-24", True),
    ("2027-12-31", False),
    ("2023-01-02", True),    # New Year's Day on a Sunday: the Monday is closed
    ("2022-12-26", True),    # Christmas on a Sunday: the Monday is closed
    ("2021-06-18", False),   # Juneteenth was not an NYSE holiday before 2022
    ("2022-06-20", True),    # first NYSE Juneteenth (a Sunday): the Monday
    ("2026-06-19", True),    # Juneteenth on a Friday
    ("2027-06-18", True),    # Juneteenth on a Saturday: the Friday
    ("2026-07-03", True),    # July 4 on a Saturday: the Friday
    ("2020-07-03", True),
    ("2026-12-25", True),    # Christmas on a Friday
    ("2027-01-01", True),    # New Year's Day on a Friday
    ("2026-04-03", True),    # Good Friday
    ("2026-11-27", False),   # the day after Thanksgiving is a session
    ("2026-11-26", True),    # Thanksgiving
    ("2005-12-26", True),
    ("2006-01-02", True),
])
def test_scheduled_closure_observance_rules(day, closed):
    assert mod.is_scheduled_closure(pd.Timestamp(day)) is closed


def test_weekends_are_scheduled_closures():
    assert mod.is_scheduled_closure(pd.Timestamp("2024-03-02")) is True
    assert mod.is_scheduled_closure(pd.Timestamp("2024-03-03")) is True


@pytest.mark.parametrize("day, expected", [
    ("2024-03-06", "2024-03-08"),    # ordinary week: the Friday
    ("2024-03-08", "2024-03-08"),    # any day of the week gives the same date
    ("2024-03-10", "2024-03-08"),    # ... Sunday included (ISO week Mon-Sun)
    ("2024-03-25", "2024-03-28"),    # Good Friday 2024-03-29: the Thursday
    ("2015-06-29", "2015-07-02"),    # July 3 2015 closed (July 4 a Saturday)
    ("2010-12-20", "2010-12-23"),    # December 24 2010 closed (Christmas Sat)
    ("2010-12-27", "2010-12-31"),    # December 31 2010 is a session
    ("2015-12-28", "2015-12-31"),    # January 1 2016 (a Friday) closed
    ("2026-06-15", "2026-06-18"),    # Juneteenth Friday
    ("2026-12-21", "2026-12-24"),    # Christmas Friday
    ("2014-11-24", "2014-11-28"),    # Thanksgiving week: Friday is a session
])
def test_scheduled_decision_date_is_the_friday_or_last_scheduled_session(
        day, expected):
    assert mod.scheduled_decision_date(pd.Timestamp(day)) == pd.Timestamp(expected)


def test_scheduled_decision_dates_are_one_per_week_and_never_after_the_end():
    dates = mod.scheduled_decision_dates(
        pd.Timestamp("2024-03-04"), pd.Timestamp("2024-04-12"))
    assert list(dates) == list(pd.to_datetime([
        "2024-03-08", "2024-03-15", "2024-03-22", "2024-03-28",
        "2024-04-05", "2024-04-12"]))
    # a decision date after `end` has not happened yet: not returned
    partial = mod.scheduled_decision_dates(
        pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-14"))
    assert list(partial) == [pd.Timestamp("2024-03-08")]
    # ... but a Thursday IS the decision date when Friday is scheduled closed,
    # and it is known at the Thursday close
    thursday = mod.scheduled_decision_dates(
        pd.Timestamp("2024-03-25"), pd.Timestamp("2024-03-28"))
    assert list(thursday) == [pd.Timestamp("2024-03-28")]


def test_scheduled_decision_dates_start_bound_and_empty_ranges():
    # weeks whose decision date is before `start` are dropped
    dates = mod.scheduled_decision_dates(
        pd.Timestamp("2024-03-09"), pd.Timestamp("2024-03-23"))
    assert list(dates) == list(pd.to_datetime(["2024-03-15", "2024-03-22"]))
    assert len(mod.scheduled_decision_dates(
        pd.Timestamp("2024-03-11"), pd.Timestamp("2024-03-13"))) == 0
    assert len(mod.scheduled_decision_dates(
        pd.Timestamp("2024-03-13"), pd.Timestamp("2024-03-01"))) == 0


def test_scheduled_decision_dates_come_from_no_data_file():
    """The schedule is a pure function of the calendar: identical whatever any
    data frame holds (holes, unscheduled closures) -- proven by calling it with
    nothing but two dates, and by comparing it to the hand-listed 2010-2016
    closures rather than to any frame."""
    dates = mod.scheduled_decision_dates(
        pd.Timestamp("2010-01-04"), pd.Timestamp("2016-12-30"))
    sessions = _bdays("2010-01-04", "2016-12-30")
    weeks = pd.Series(sessions, index=sessions).groupby(
        [sessions.isocalendar().year.values,
         sessions.isocalendar().week.values]).max()
    assert list(dates) == list(pd.DatetimeIndex(weeks.values))
    assert dates.is_monotonic_increasing and dates.is_unique
    assert dates.tz is None


def test_scheduled_decision_dates_match_exchange_calendars_xnys():
    """An independent oracle: exchange_calendars' XNYS last session of every
    ISO week 2005-2027. The two unscheduled closures near a Friday are outside
    the window (the 2004 mourning Friday) or on other weekdays (Sandy 2012,
    2018, 2025), so no exception is needed."""
    xc = pytest.importorskip("exchange_calendars")
    calendar = xc.get_calendar("XNYS", start="2004-01-01", end="2028-12-31")
    sessions = calendar.sessions_in_range(
        "2005-01-03", "2027-06-25").tz_localize(None)
    iso = sessions.isocalendar()
    last = pd.Series(sessions, index=sessions).groupby(
        [iso.year.values, iso.week.values]).max()
    expected = pd.DatetimeIndex(last.values)
    ours = mod.scheduled_decision_dates(sessions[0], sessions[-1])
    assert list(ours) == list(expected)


# ------------------------------------------------- pure building blocks

def _close_series(values, index=None, start="2020-01-01") -> pd.Series:
    values = np.asarray(values, dtype=float)
    if index is None:
        index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index)


def _oracle_smoothed(move: pd.Series, vix: pd.Series, *, min_history: int,
                     window: int) -> pd.Series:
    """The pre-registered arithmetic re-implemented from the rule text in this
    file's docstring with plain loops, sharing no code with the module:
    z-statistics from the
    observations strictly before t (numpy, ddof 1), DFG, then the mean of the
    last `window` DFG observations."""
    common = move.index.intersection(vix.index)
    m = move.loc[common].to_numpy(float)
    v = vix.loc[common].to_numpy(float)
    n = len(common)
    dfg = np.full(n, np.nan)
    for t in range(min_history, n):
        zm = (m[t] - m[:t].mean()) / m[:t].std(ddof=1)
        zv = (v[t] - v[:t].mean()) / v[:t].std(ddof=1)
        dfg[t] = (zm - zv) / np.sqrt(2.0)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        chunk = dfg[t - window + 1:t + 1]
        if not np.isnan(chunk).any():
            out[t] = chunk.mean()
    return pd.Series(out, index=common)


def test_clean_close_is_tz_naive_sorted_deduplicated_finite_and_never_floored():
    index = pd.DatetimeIndex(pd.to_datetime([
        "2020-03-26", "2020-03-24", "2020-03-25", "2020-03-25",
        "2020-03-27", "2020-03-30", "2020-03-31"])).tz_localize("UTC")
    frame = pd.DataFrame({"Close": [-0.105, 33.0, 1.0, 70.0,
                                    np.nan, np.inf, 0.0]}, index=index)
    cleaned = mod._clean_close(frame)
    assert cleaned.index.tz is None
    assert cleaned.index.is_monotonic_increasing and cleaned.index.is_unique
    # 03-25 appears twice: the LAST occurrence (70.0) wins; NaN and inf are
    # dropped; zero and negative prints survive unmodified (a signal-only
    # series is never forced through a traded price's positivity gate).
    assert list(cleaned.index) == list(pd.to_datetime(
        ["2020-03-24", "2020-03-25", "2020-03-26", "2020-03-31"]))
    assert list(cleaned.to_numpy()) == [33.0, 70.0, -0.105, 0.0]


def test_expanding_zscore_is_hand_worked_from_the_earlier_observations_only():
    x = _close_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    z = mod.expanding_zscore(x, min_history=3)
    assert z.iloc[:3].isna().all()               # fewer than 3 prior obs
    assert z.iloc[3] == pytest.approx(2.0)       # prior [1,2,3]: mean 2, sd 1
    assert z.iloc[4] == pytest.approx((5.0 - 2.5) / np.sqrt(5.0 / 3.0))
    assert z.iloc[5] == pytest.approx((6.0 - 3.0) / np.sqrt(10.0 / 4.0))


def test_expanding_zscore_needs_exactly_min_history_prior_observations():
    x = _close_series(np.arange(1.0, 13.0) ** 1.5)
    for min_history in (2, 5, 9):
        z = mod.expanding_zscore(x, min_history=min_history)
        assert z.iloc[:min_history].isna().all()
        assert z.iloc[min_history:].notna().all()
        assert z.first_valid_index() == x.index[min_history]


def test_expanding_zscore_uses_the_sample_standard_deviation_ddof_1():
    x = _close_series([1.0, 2.0, 3.0])
    z = mod.expanding_zscore(x, min_history=2)
    # prior [1,2]: mean 1.5, sample sd 0.70711 -> 2.1213; ddof 0 would be 3.0
    assert z.iloc[2] == pytest.approx(1.5 / np.sqrt(0.5))
    assert abs(z.iloc[2] - 3.0) > 0.5


def test_expanding_zscore_matches_a_brute_force_loop_at_every_position():
    rng = np.random.RandomState(11)
    x = _close_series(50.0 + np.cumsum(rng.normal(0.0, 1.0, 80)))
    z = mod.expanding_zscore(x, min_history=7)
    for t in range(len(x)):
        if t < 7:
            assert np.isnan(z.iloc[t])
        else:
            prior = x.to_numpy()[:t]
            expected = (x.iloc[t] - prior.mean()) / prior.std(ddof=1)
            assert z.iloc[t] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_expanding_zscore_window_ends_at_t_minus_1_never_at_t():
    """If the statistics included x_t, nudging x_t would move its
    own mean and standard deviation and the change in z_t would not be
    d / sd(prior). With the window ending at t-1 it is exactly that, and no
    earlier z moves at all."""
    rng = np.random.RandomState(5)
    x = _close_series(20.0 + np.cumsum(rng.normal(0.0, 1.0, 40)))
    t, bump = 25, 3.0
    bumped = x.copy()
    bumped.iloc[t] += bump
    z0 = mod.expanding_zscore(x, min_history=6)
    z1 = mod.expanding_zscore(bumped, min_history=6)
    prior_sd = x.to_numpy()[:t].std(ddof=1)
    assert z1.iloc[t] - z0.iloc[t] == pytest.approx(bump / prior_sd, rel=1e-12)
    assert np.array_equal(z0.iloc[:t].to_numpy(), z1.iloc[:t].to_numpy(),
                          equal_nan=True)


def test_expanding_zscore_is_prefix_invariant():
    rng = np.random.RandomState(3)
    x = _close_series(30.0 + np.cumsum(rng.normal(0.0, 1.0, 70)))
    full = mod.expanding_zscore(x, min_history=8)
    for cut in (9, 10, 31, 50):
        cut_z = mod.expanding_zscore(x.iloc[:cut], min_history=8)
        assert np.array_equal(full.iloc[:cut].to_numpy(), cut_z.to_numpy(),
                              equal_nan=True)


def test_expanding_zscore_of_a_constant_series_is_nan_never_inf():
    z = mod.expanding_zscore(_close_series([5.0] * 12), min_history=3)
    assert z.isna().all()
    # a constant prefix followed by a move: undefined until the sd is > 0
    z2 = mod.expanding_zscore(_close_series([5.0] * 6 + [6.0, 7.0]),
                              min_history=3)
    assert np.isfinite(z2.dropna().to_numpy()).all()
    assert np.isnan(z2.iloc[6])                    # prior sd is zero
    assert np.isfinite(z2.iloc[7])


def test_divergence_is_hand_worked_scaled_by_sqrt_2_and_signed_bond_minus_equity():
    move = _close_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    vix = _close_series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    dfg = mod.divergence_series(move, vix, min_history=3)
    assert dfg.iloc[:3].isna().all()
    # zMOVE = +2.0, zVIX = -2.0 at position 3: (2 - -2) / sqrt(2)
    assert dfg.iloc[3] == pytest.approx(4.0 / np.sqrt(2.0))
    assert dfg.iloc[3] == pytest.approx(2.0 * np.sqrt(2.0))
    # sign: bond-market fear up, equity fear flat -> positive divergence
    rng = np.random.RandomState(2)
    base_m = _close_series(90.0 + np.cumsum(rng.normal(0.0, 1.0, 40)))
    base_v = _close_series(18.0 + np.cumsum(rng.normal(0.0, 0.3, 40)))
    ref = mod.divergence_series(base_m, base_v, min_history=8)
    up_m = base_m.copy()
    up_m.iloc[-1] += 20.0
    up_v = base_v.copy()
    up_v.iloc[-1] += 20.0
    assert (mod.divergence_series(up_m, base_v, min_history=8).iloc[-1]
            > ref.iloc[-1])
    assert (mod.divergence_series(base_m, up_v, min_history=8).iloc[-1]
            < ref.iloc[-1])


def test_divergence_lives_on_the_two_indices_common_calendar_only():
    """Both indices are standardized over the SAME sample: a date where only
    one of them prints (a VIX print on a bond-market holiday, a MOVE hole) is
    in neither statistic and in no output."""
    rng = np.random.RandomState(9)
    index = pd.bdate_range("2020-01-06", periods=60)
    move = _close_series(90.0 + np.cumsum(rng.normal(0.0, 1.0, 60)), index)
    vix = _close_series(18.0 + np.cumsum(rng.normal(0.0, 0.3, 60)), index)
    clean = mod.divergence_series(move, vix, min_history=10)
    # a VIX-only extra date, with a VIX value there wildly different
    extra_day = pd.Timestamp("2020-04-04")           # a Saturday: not in index
    vix_extra = pd.concat([vix, pd.Series([999.0], index=[extra_day])]).sort_index()
    with_extra = mod.divergence_series(move, vix_extra, min_history=10)
    assert with_extra.index.equals(clean.index)
    assert np.array_equal(with_extra.to_numpy(), clean.to_numpy(), equal_nan=True)
    # a MOVE hole shrinks the common calendar (and shifts what counts as the
    # 10th prior observation), but the result is still exactly the oracle's
    hole = index[30]
    move_hole = move.drop(hole)
    result = mod.divergence_series(move_hole, vix, min_history=10)
    assert hole not in result.index
    assert result.index.equals(index.drop(hole))
    oracle = _oracle_smoothed(move_hole, vix, min_history=10, window=1)
    assert np.allclose(result.dropna().to_numpy(), oracle.dropna().to_numpy())


def test_smoothed_divergence_is_the_mean_of_the_last_window_observations():
    rng = np.random.RandomState(4)
    index = pd.bdate_range("2021-01-04", periods=70)
    move = _close_series(90.0 + np.cumsum(rng.normal(0.0, 1.0, 70)), index)
    vix = _close_series(18.0 + np.cumsum(rng.normal(0.0, 0.3, 70)), index)
    smoothed = mod.smoothed_divergence(move, vix, min_history=12, window=5)
    oracle = _oracle_smoothed(move, vix, min_history=12, window=5)
    assert np.array_equal(np.isnan(smoothed.to_numpy()),
                          np.isnan(oracle.to_numpy()))
    assert np.allclose(smoothed.dropna().to_numpy(), oracle.dropna().to_numpy(),
                       rtol=1e-12, atol=1e-12)
    # undefined until the WHOLE window is defined: 12 prior obs + 4 more
    assert smoothed.first_valid_index() == index[12 + 5 - 1]


def test_smoothed_divergence_counts_observations_not_calendar_days():
    """A hole in the common calendar means the 'last 10' reach one session
    further back in time; they are the last 10 OBSERVATIONS."""
    rng = np.random.RandomState(6)
    index = pd.bdate_range("2021-01-04", periods=50)
    move = _close_series(90.0 + np.cumsum(rng.normal(0.0, 1.0, 50)), index)
    vix = _close_series(18.0 + np.cumsum(rng.normal(0.0, 0.3, 50)), index)
    hole = index[40]
    move_hole = move.drop(hole)
    smoothed = mod.smoothed_divergence(move_hole, vix, min_history=10, window=10)
    dfg = mod.divergence_series(move_hole, vix, min_history=10)
    at = index[45]
    position = dfg.index.get_loc(at)
    assert smoothed.loc[at] == pytest.approx(
        dfg.iloc[position - 9:position + 1].mean())
    # the 10 observations ending at index[45] span 11 calendar sessions
    assert dfg.index[position - 9] == index[35]


def test_derisk_threshold_edge_is_strict_and_undefined_is_never_derisk():
    assert bool(mod.is_derisk(0.5)) is False
    assert bool(mod.is_derisk(np.nextafter(0.5, 1.0))) is True
    assert bool(mod.is_derisk(np.nextafter(0.5, 0.0))) is False
    assert bool(mod.is_derisk(0.0)) is False
    assert bool(mod.is_derisk(-3.0)) is False
    assert bool(mod.is_derisk(float("nan"))) is False
    assert list(mod.is_derisk(np.array([0.2, 0.5, 0.5000001, 9.0]))) == [
        False, False, True, True]


def _state_case():
    """A calendar with Good Friday 2024-03-29 closed, small constants."""
    index = pd.bdate_range("2024-01-08", "2024-05-10")
    index = index[index != pd.Timestamp("2024-03-29")]
    rng = np.random.RandomState(21)
    move = _close_series(90.0 + np.cumsum(rng.normal(0.0, 2.0, len(index))), index)
    vix = _close_series(18.0 + np.cumsum(rng.normal(0.0, 0.5, len(index))), index)
    return move, vix


def test_derisk_state_series_is_dated_only_at_scheduled_decision_dates():
    move, vix = _state_case()
    kwargs = dict(min_history=15, window=3, threshold=0.5)
    state = mod.derisk_state_series(move, vix, **kwargs)
    smoothed = _oracle_smoothed(move, vix, min_history=15, window=3)
    schedule = mod.scheduled_decision_dates(move.index[0], move.index[-1])
    expected_dates = [d for d in schedule if not np.isnan(smoothed.get(d, np.nan))]
    assert len(expected_dates) >= 8
    assert list(state.index) == expected_dates
    assert pd.Timestamp("2024-03-28") in state.index       # Good Friday week
    assert pd.Timestamp("2024-03-29") not in state.index
    assert all(d.weekday() in (3, 4) for d in state.index)
    assert set(state.unique()) <= {0.0, 1.0}
    for date in state.index:
        assert state.loc[date] == float(smoothed.loc[date] > 0.5)


def test_derisk_state_series_has_no_reading_for_a_decision_date_without_a_print():
    move, vix = _state_case()
    friday = pd.Timestamp("2024-04-12")
    assert friday in mod.scheduled_decision_dates(move.index[0], move.index[-1])
    for missing_frame in ("move", "vix"):
        m, v = move.copy(), vix.copy()
        if missing_frame == "move":
            m = m.drop(friday)
        else:
            v = v.drop(friday)
        state = mod.derisk_state_series(m, v, min_history=15, window=3,
                                        threshold=0.5)
        assert friday not in state.index
        # ... and NO fallback to Thursday: 2024-04-11 is not a decision date
        assert pd.Timestamp("2024-04-11") not in state.index
        # every other decision date keeps its reading
        assert pd.Timestamp("2024-04-05") in state.index
        assert pd.Timestamp("2024-04-19") in state.index


def test_derisk_state_series_threshold_is_strict_on_the_smoothed_reading():
    move, vix = _state_case()
    # the module's OWN reading at the date (the oracle agrees to rounding but
    # not bit for bit, and this test is about the comparison at the edge)
    smoothed = mod.smoothed_divergence(move, vix, min_history=15, window=3)
    schedule = mod.scheduled_decision_dates(move.index[0], move.index[-1])
    date = [d for d in schedule if not np.isnan(smoothed.get(d, np.nan))][4]
    reading = float(smoothed.loc[date])
    at_edge = mod.derisk_state_series(move, vix, min_history=15, window=3,
                                      threshold=reading)
    above = mod.derisk_state_series(move, vix, min_history=15, window=3,
                                    threshold=float(np.nextafter(reading, -np.inf)))
    assert at_edge.loc[date] == 0.0          # reading == threshold: NOT derisk
    assert above.loc[date] == 1.0            # reading just above the threshold


def test_weeks_without_a_reading_lists_only_post_warmup_complete_weeks():
    move, vix = _state_case()
    common = move.index.intersection(vix.index)
    state = mod.derisk_state_series(move, vix, min_history=15, window=3,
                                    threshold=0.5)
    assert mod.weeks_without_a_reading(state, common) == []
    friday = pd.Timestamp("2024-04-12")
    holed = mod.derisk_state_series(move.drop(friday), vix, min_history=15,
                                    window=3, threshold=0.5)
    missing = mod.weeks_without_a_reading(holed, common.drop(friday))
    assert missing == [friday]
    # the week still in progress (its decision date is after the data's last
    # date) and the warm-up before the first reading are never demanded
    cut = common[common <= pd.Timestamp("2024-04-10")]
    cut_state = mod.derisk_state_series(move.loc[cut], vix.loc[cut],
                                        min_history=15, window=3, threshold=0.5)
    assert mod.weeks_without_a_reading(cut_state, cut) == []


# --------------------- engine-level behaviour vs an INDEPENDENT oracle

# Six synthetic stress episodes (the bond-market gauge jumps while the equity
# gauge does not). Some start on the Monday right after a decision Friday and
# some on a Wednesday, so a look-ahead of one session AND of several sessions
# both change a decision somewhere. The jump grows because the expanding
# standard deviation grows after every earlier episode.
SPIKE_STARTS = ["2013-06-10", "2014-02-05", "2014-09-15", "2015-03-11",
                "2015-10-12", "2016-05-16"]
SPIKE_JUMPS = [150.0, 500.0, 1200.0, 2500.0, 5000.0, 9000.0]
SPIKE_LENGTH = 25


def _synthetic_frames(*, seed: int = 1) -> dict:
    index = _bdays("2010-01-04", "2016-12-30")
    n = len(index)
    rng = np.random.RandomState(seed)
    vix = np.empty(n)
    move = np.empty(n)
    v, m = 18.0, 90.0
    for i in range(n):
        v = 18.0 + 0.9 * (v - 18.0) + rng.normal(0.0, 0.4)
        m = 90.0 + 0.9 * (m - 90.0) + rng.normal(0.0, 1.2)
        vix[i], move[i] = v, m
    for start, jump in zip(SPIKE_STARTS, SPIKE_JUMPS):
        position = int(index.get_loc(pd.Timestamp(start)))
        move[position:position + SPIKE_LENGTH] += jump
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    shy = 80.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.001, n)))
    return {RISK: _ohlc_from_close(spy, index),
            DEFENSIVE: _ohlc_from_close(shy, index),
            MOVE: _ohlc_from_close(move, index),
            VIX: _ohlc_from_close(vix, index)}


def _run_like_abi1(frames: dict, **factory_kwargs):
    """The wiring scripts/run_abi1_mining.py + abi1_driver.build_rotation_curve
    use: the target factory gets the FULL, UNALIGNED frames; the engine gets
    the two traded legs and the two signal frames intersected onto one
    calendar."""
    target_fn = mod.make_target_fn(frames, **factory_kwargs)
    return _run_engine_with(frames, target_fn)


def _run_engine_with(frames: dict, target_fn):
    aligned_assets, aligned_signals = abi1_driver._align_rotation_frames(
        {RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]},
        {MOVE: frames[MOVE], VIX: frames[VIX]})
    trades, active = rotation_engine.run_two_asset_rotation(
        aligned_assets, target_fn,
        cost_bps_by_asset={RISK: 1.0, DEFENSIVE: 1.0},
        signal_frames=aligned_signals)
    return trades, active, aligned_assets


def _oracle_decision_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last session of every ISO week of a HOLE-FREE calendar (the synthetic
    calendar equals the exchange's scheduled sessions), i.e. the intended
    weekly decision dates -- derived here from the calendar, sharing no code
    with the module's closed-form schedule."""
    iso = index.isocalendar()
    last = pd.Series(index, index=index).groupby(
        [iso.year.values, iso.week.values]).max()
    return pd.DatetimeIndex(last.values)


def _oracle_states(frames: dict, decision_dates) -> dict:
    """{decision date: asset for the coming week}, from the rule text."""
    smoothed = _oracle_smoothed(
        frames[MOVE]["Close"], frames[VIX]["Close"],
        min_history=MIN_HISTORY, window=SMOOTHING)
    states = {}
    for date in decision_dates:
        reading = smoothed.get(date, np.nan)
        if not np.isnan(reading):
            states[date] = DEFENSIVE if reading > THRESHOLD else RISK
    return states


def _oracle_active_assets(calendar: pd.DatetimeIndex, states: dict) -> list:
    """The asset held DURING each session: the latest reading dated on or
    before the previous session's close (else the default, SPY); nothing is
    held during the engine's very first session."""
    dates = sorted(states)
    held_after_bar = []
    for day in calendar:
        applicable = [d for d in dates if d <= day]
        held_after_bar.append(states[applicable[-1]] if applicable else RISK)
    return [None] + held_after_bar[:-1]


@pytest.fixture(scope="module")
def synth():
    frames = _synthetic_frames()
    trades, active, aligned = _run_like_abi1(frames)
    decisions = _oracle_decision_dates(frames[RISK].index)
    states = _oracle_states(frames, decisions)
    return {"frames": frames, "trades": trades, "active": active,
            "aligned": aligned, "decisions": decisions, "states": states,
            "index": frames[RISK].index}


def _flip_dates(states: dict) -> list:
    """Decision dates at which the asset for the coming week CHANGES -- the
    only places a lost or peeked decision is visible."""
    dates = sorted(states)
    return [d for prev, d in zip(dates[:-1], dates[1:])
            if states[d] != states[prev]]


def test_engine_positions_match_the_independent_oracle_on_a_multi_regime_path(synth):
    expected = _oracle_active_assets(synth["index"], synth["states"])
    assert list(synth["active"]) == expected
    # Power guard: the scenario must exercise both assets and many switches.
    held = [a for a in expected if a is not None]
    assert set(held) == {RISK, DEFENSIVE}
    flips = _flip_dates(synth["states"])
    assert len(flips) >= 8
    switches = sum(1 for a, b in zip(held, held[1:]) if a != b)
    assert switches >= 8
    assert len(synth["trades"]) == switches + 1
    assert set(synth["trades"]["asset"]) == {RISK, DEFENSIVE}


def test_a_switch_only_ever_fills_on_the_session_after_a_decision_date(synth):
    index = synth["index"]
    decisions = set(mod.scheduled_decision_dates(index[0], index[-1]))
    trades = synth["trades"]
    assert len(trades) >= 9
    # the first entry is the default holding at the engine's first fill
    assert trades.iloc[0]["entry_date"] == index[1]
    for entry_date in trades["entry_date"].iloc[1:]:
        previous_session = index[index.get_loc(entry_date) - 1]
        assert previous_session in decisions, entry_date


def test_the_closed_form_schedule_equals_the_calendar_derived_decision_dates(synth):
    index = synth["index"]
    assert list(mod.scheduled_decision_dates(index[0], index[-1])
                ) == list(synth["decisions"])


def test_before_the_first_reading_the_account_holds_the_default_spy_never_flat(synth):
    active = list(synth["active"])
    index = synth["index"]
    first_reading = min(synth["states"])
    assert first_reading > index[MIN_HISTORY + SMOOTHING]      # a long warm-up
    assert active[0] is None                  # nothing is filled before bar 1
    assert None not in active[1:]             # ... and never flat afterwards
    first_switch = next(k for k, a in enumerate(active) if a == DEFENSIVE)
    assert index[first_switch - 1] >= first_reading
    assert set(active[1:first_switch]) == {RISK}
    trades = synth["trades"]
    assert trades.iloc[0]["asset"] == RISK
    assert trades.iloc[0]["entry_date"] == index[1]


def test_a_midweek_stress_signal_moves_the_position_only_after_the_decision_date(synth):
    """The 2015-03-11 (Wednesday) episode is visible in the smoothed reading
    from Wednesday on, but the account keeps SPY until the Friday decision
    and switches at the NEXT session's open."""
    index = synth["index"]
    start = pd.Timestamp("2015-03-11")
    decision = next(d for d in _flip_dates(synth["states"])
                    if d > start and synth["states"][d] == DEFENSIVE)
    assert decision - start <= pd.Timedelta(days=4)
    active = synth["active"]
    held_during = active.loc[start:index[index.get_loc(decision) + 1]]
    assert set(held_during.iloc[:-1]) == {RISK}          # through the Friday
    assert held_during.iloc[-1] == DEFENSIVE             # the next session
    trades = synth["trades"]
    entry = trades.loc[(trades["asset"] == DEFENSIVE)
                       & (trades["entry_date"] > start)].iloc[0]
    assert entry["entry_date"] == index[index.get_loc(decision) + 1]


def _hole_case(synth, *, at_flip: bool):
    """A decision Friday to delete from ^MOVE: a state flip whose NEXT decision
    keeps the new state, so a whole week's delay is visible."""
    states = synth["states"]
    dates = sorted(states)
    for previous, date, following in zip(dates[:-2], dates[1:-1], dates[2:]):
        if states[date] != states[previous] and states[following] == states[date]:
            return date
    raise AssertionError("no suitable flip found")


def test_a_decision_friday_without_a_move_print_carries_the_previous_regime_one_week(synth):
    frames = synth["frames"]
    index = synth["index"]
    hole = _hole_case(synth, at_flip=True)
    holed = dict(frames)
    holed[MOVE] = frames[MOVE].drop(hole)
    trades, active, aligned = _run_like_abi1(holed)
    # the oracle: readings only where the decision date has a print; NO
    # fallback to the previous session
    states = _oracle_states(holed, synth["decisions"])
    assert hole not in states
    assert list(active) == _oracle_active_assets(aligned[RISK].index, states)
    # the flip the hole swallowed: the un-holed run switches on the Monday
    # after the hole Friday; the holed run still holds the PREVIOUS week's
    # asset that Monday (carried), never a fallback to the Thursday's data
    dates = sorted(synth["states"])
    previous_asset = synth["states"][dates[dates.index(hole) - 1]]
    flip_asset = synth["states"][hole]
    assert previous_asset != flip_asset
    monday = index[index.get_loc(hole) + 1]
    assert synth["active"].loc[monday] == flip_asset
    assert active.loc[monday] == previous_asset


def test_the_hole_friday_has_no_reading_in_the_state_series(synth):
    hole = _hole_case(synth, at_flip=True)
    frames = synth["frames"]
    move = mod._clean_close(frames[MOVE].drop(hole))
    vix = mod._clean_close(frames[VIX])
    state = mod.derisk_state_series(
        move, vix, min_history=MIN_HISTORY, window=SMOOTHING, threshold=THRESHOLD)
    assert hole not in state.index
    thursday = hole - pd.Timedelta(days=1)
    assert thursday not in state.index


def test_a_decision_friday_missing_from_the_traded_legs_uses_its_reading_from_the_next_bar(synth):
    """The two indices print on the Friday, the traded funds do not (the
    engine's calendar has no such session). The reading dated that Friday is
    still causal and is applied at the next available bar."""
    frames = synth["frames"]
    hole = _hole_case(synth, at_flip=True)
    holed = dict(frames)
    holed[RISK] = frames[RISK].drop(hole)
    holed[DEFENSIVE] = frames[DEFENSIVE].drop(hole)
    trades, active, aligned = _run_like_abi1(holed)
    assert hole not in aligned[RISK].index
    states = _oracle_states(holed, synth["decisions"])
    assert hole in states                       # the reading exists
    assert list(active) == _oracle_active_assets(aligned[RISK].index, states)


def test_make_target_fn_requires_every_traded_and_signal_frame(synth):
    frames = synth["frames"]
    for key in (RISK, DEFENSIVE, MOVE, VIX):
        partial = {k: v for k, v in frames.items() if k != key}
        with pytest.raises(ValueError, match=key.replace("^", r"\^")):
            mod.make_target_fn(partial)


def test_make_target_fn_rejects_an_empty_or_closeless_frame(synth):
    frames = synth["frames"]
    empty = dict(frames)
    empty[MOVE] = frames[MOVE].iloc[:0]
    with pytest.raises(ValueError, match="non-empty"):
        mod.make_target_fn(empty)
    closeless = dict(frames)
    closeless[VIX] = frames[VIX].drop(columns=["Close"])
    with pytest.raises(ValueError, match="Close"):
        mod.make_target_fn(closeless)


def test_make_target_fn_refuses_to_build_when_no_decision_date_has_a_reading(synth):
    frames = synth["frames"]
    short = {key: frame.iloc[:MIN_HISTORY + SMOOTHING - 3]
             for key, frame in frames.items()}
    with pytest.raises(ValueError, match="no reading"):
        mod.make_target_fn(short)


def test_make_target_fn_refuses_a_constant_signal_series(synth):
    frames = synth["frames"]
    flat = dict(frames)
    flat[VIX] = _ohlc_from_close(np.full(len(frames[VIX]), 18.0), frames[VIX].index)
    with pytest.raises(ValueError, match="no reading"):
        mod.make_target_fn(flat)


def test_require_complete_decision_weeks_raises_naming_the_missing_week(synth):
    frames = synth["frames"]
    hole = _hole_case(synth, at_flip=True)
    holed = dict(frames)
    holed[MOVE] = frames[MOVE].drop(hole)
    with pytest.raises(ValueError, match=hole.strftime("%Y-%m-%d")):
        mod.make_target_fn(holed, require_complete_decision_weeks=True)
    # the default carries the previous regime through it instead
    assert callable(mod.make_target_fn(holed))
    assert callable(mod.make_target_fn(holed, require_complete_decision_weeks=False))


def test_require_complete_decision_weeks_does_not_demand_warmup_or_in_progress_weeks(synth):
    frames = synth["frames"]
    index = synth["index"]
    # a hole deep inside the warm-up (before any reading exists)
    warm_hole = index[index.get_loc(pd.Timestamp("2011-06-10"))]
    holed = dict(frames)
    holed[MOVE] = frames[MOVE].drop(warm_hole)
    assert callable(mod.make_target_fn(holed, require_complete_decision_weeks=True))
    # data ending mid-week: that week's decision date is still in the future
    cut = pd.Timestamp("2015-06-10")                     # a Wednesday
    truncated = _truncate(frames, cut)
    assert callable(mod.make_target_fn(truncated, require_complete_decision_weeks=True))
    # data ending on the Thursday before Good Friday: Thursday IS the decision
    thursday = pd.Timestamp("2015-04-02")
    assert thursday in index
    truncated = _truncate(frames, thursday)
    assert callable(mod.make_target_fn(truncated, require_complete_decision_weeks=True))


def test_a_vix_print_on_an_exchange_holiday_changes_nothing(synth):
    """A VIX row on a session the other series lack (real ^VIX carries prints
    on a couple of US holidays) is in no statistic and no decision."""
    frames = synth["frames"]
    extra_days = [d for d in _HAND_CLOSURES
                  if pd.Timestamp("2013-01-01") <= d <= pd.Timestamp("2016-12-31")]
    assert len(extra_days) >= 20
    extra = pd.DataFrame(
        {"Open": 999.0, "High": 999.0, "Low": 999.0, "Close": 999.0},
        index=pd.DatetimeIndex(extra_days))
    variant = dict(frames)
    variant[VIX] = pd.concat([frames[VIX], extra]).sort_index()
    _, active, _ = _run_like_abi1(variant)
    assert list(active) == list(synth["active"])


def test_zero_and_negative_signal_prints_flow_through_the_real_engine(synth):
    """A signal-only index is never forced through a traded price's positivity
    gate: the real engine takes it (signal frames are exempt), and the
    decisions still equal the oracle's."""
    frames = synth["frames"]
    index = synth["index"]
    bad_day = index[1300]
    assert bad_day.weekday() != 4
    variant = dict(frames)
    move = frames[MOVE].copy()
    move.loc[bad_day, ["Open", "High", "Low", "Close"]] = -5.0
    variant[MOVE] = move
    _, active, aligned = _run_like_abi1(variant)
    states = _oracle_states(variant, synth["decisions"])
    assert list(active) == _oracle_active_assets(aligned[RISK].index, states)


def test_a_tz_aware_unsorted_duplicated_signal_frame_is_cleaned_not_trusted(synth):
    frames = synth["frames"]
    move = frames[MOVE]
    shuffled = move.sample(frac=1.0, random_state=0)
    duplicated = pd.concat([shuffled, move.iloc[100:110]])   # duplicate rows
    duplicated.index = duplicated.index.tz_localize("UTC")
    variant = dict(frames)
    variant[MOVE] = duplicated
    baseline = mod.make_target_fn(frames)
    cleaned = mod.make_target_fn(variant)
    view = _truncate(frames, synth["index"][1500])
    assert cleaned(view) == baseline(view)


# ------------------------------------- no look-ahead / hold-out blindness

def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _corrupt_after(frame: pd.DataFrame, cutoff: pd.Timestamp, value) -> pd.DataFrame:
    out = frame.copy(deep=True)
    out.loc[out.index > cutoff, ["Open", "High", "Low", "Close"]] = value
    return out


def _calendar_cutoffs(index: pd.DatetimeIndex, flips, offsets=(-2, -1, 0, 1, 2, 3, 4)):
    """Cutoff sessions placed RELATIVE TO the regime-changing decision dates
    -- a peek of h sessions is only visible when a cutoff lands
    between a decision and the h sessions after it. Never round dates."""
    cutoffs = set()
    for date in flips:
        base = int(index.get_loc(date))
        for offset in offsets:
            cutoffs.add(base + offset)
    return sorted(c for c in cutoffs if 0 < c < len(index))


def _prefix_violations(frames: dict, cutoffs) -> list:
    """Cutoff dates at which the target rebuilt from data that ENDS at the
    cutoff differs from the full-history target on the same view."""
    index = frames[RISK].index
    full_target = mod.make_target_fn(frames)
    bad = []
    for cutoff in cutoffs:
        when = index[cutoff]
        view = _truncate(frames, when)
        cut_target = mod.make_target_fn(_truncate(frames, when))
        if cut_target(view) != full_target(view):
            bad.append(when)
    return bad


def test_a_target_rebuilt_from_data_ending_at_the_cutoff_equals_the_full_history_target(synth):
    index = synth["index"]
    flips = _flip_dates(synth["states"])
    cutoffs = _calendar_cutoffs(index, flips)
    assert len(cutoffs) >= 100
    assert _prefix_violations(synth["frames"], cutoffs) == []
    # power guard: the decision genuinely varies across the cutoffs
    full_target = mod.make_target_fn(synth["frames"])
    decisions = {full_target(_truncate(synth["frames"], index[c])) for c in cutoffs}
    assert decisions == {RISK, DEFENSIVE}


@pytest.mark.parametrize("garbage", [-50.0, 0.0, 1.0e6])
def test_a_decision_does_not_depend_on_any_later_signal_row(synth, garbage):
    frames = synth["frames"]
    index = synth["index"]
    original = mod.make_target_fn(frames)
    flips = _flip_dates(synth["states"])
    tested, decisions = 0, set()
    for cutoff in _calendar_cutoffs(index, flips, offsets=(-1, 0, 1, 2))[::3]:
        when = index[cutoff]
        variant = dict(frames)
        variant[MOVE] = _corrupt_after(frames[MOVE], when, garbage)
        variant[VIX] = _corrupt_after(frames[VIX], when, garbage)
        view = _truncate(frames, when)
        assert mod.make_target_fn(variant)(view) == original(view), when
        decisions.add(original(view))
        tested += 1
    assert tested >= 15
    assert decisions == {RISK, DEFENSIVE}


def test_a_decision_does_not_depend_on_any_later_signal_rows_at_all(synth):
    """Removing every later signal row (a shorter file) must not move the
    decision either: decision dates come from the ex-ante schedule, so the
    availability of future rows cannot change which session is a decision
    date."""
    frames = synth["frames"]
    index = synth["index"]
    original = mod.make_target_fn(frames)
    for cutoff in _calendar_cutoffs(index, _flip_dates(synth["states"]),
                                    offsets=(0, 1, 2))[::4]:
        when = index[cutoff]
        variant = dict(frames)
        for key in (MOVE, VIX):
            variant[key] = frames[key].loc[frames[key].index <= when]
        view = _truncate(frames, when)
        assert mod.make_target_fn(variant)(view) == original(view), when


def test_a_decision_does_not_depend_on_later_traded_prices(synth):
    frames = synth["frames"]
    index = synth["index"]
    original = mod.make_target_fn(frames)
    for cutoff in _calendar_cutoffs(index, _flip_dates(synth["states"]),
                                    offsets=(0, 1))[::4]:
        when = index[cutoff]
        variant = dict(frames)
        for leg in (RISK, DEFENSIVE):
            variant[leg] = _corrupt_after(frames[leg], when, 1.0e6)
        view = _truncate(frames, when)
        assert mod.make_target_fn(variant)(view) == original(view), when


def test_engine_run_on_truncated_data_equals_the_prefix_of_the_full_run(synth):
    """Real engine, real target factory: run everything on data that ENDS at
    the cutoff (target factory built from the truncated frames too, i.e. all
    that anyone could know then) and require the held asset on every session
    up to the cutoff to equal the full-history run's."""
    index = synth["index"]
    full = list(synth["active"])
    flips = _flip_dates(synth["states"])
    picks = [flips[0], flips[3], flips[6], flips[-1]]
    cutoffs = _calendar_cutoffs(index, picks, offsets=(0, 1))
    assert len(cutoffs) >= 6
    saw_switch = False
    for cutoff in cutoffs:
        truncated = _truncate(synth["frames"], index[cutoff])
        _, active, _ = _run_like_abi1(truncated)
        assert list(active) == full[:cutoff + 1], (
            f"held asset changed with data after {index[cutoff].date()}")
        saw_switch |= any(a != b for a, b in zip(full[1:cutoff], full[2:cutoff + 1]))
    assert saw_switch


def _peeking(original, sessions_ahead):
    """A deliberately LOOK-AHEAD smoothed divergence: the reading dated t is
    the one that will only exist `sessions_ahead` sessions later."""
    def wrapped(*args, **kwargs):
        return original(*args, **kwargs).shift(-sessions_ahead)
    return wrapped


def _full_sample_zscore(close, *, min_history):
    """A deliberately LOOK-AHEAD standardization: full-sample mean and sd."""
    values = pd.Series(np.asarray(close, dtype=float), index=close.index)
    z = (values - values.mean()) / values.std(ddof=1)
    z.iloc[:min_history] = np.nan
    return z


def test_the_prefix_test_has_power_a_control_passes_and_look_ahead_mutants_fail(
        synth, monkeypatch):
    """Cutoff placement, proven rather than assumed: the same cutoff set that passes
    the real code FAILS when a one-session peek, a three-session peek, or a
    full-sample standardization is injected."""
    index = synth["index"]
    cutoffs = _calendar_cutoffs(index, _flip_dates(synth["states"]))
    frames = synth["frames"]
    assert _prefix_violations(frames, cutoffs) == []          # control
    original_smoothed = mod.smoothed_divergence
    for sessions_ahead in (1, 3):
        monkeypatch.setattr(mod, "smoothed_divergence",
                            _peeking(original_smoothed, sessions_ahead))
        violations = _prefix_violations(frames, cutoffs)
        assert violations, f"a {sessions_ahead}-session peek went undetected"
    monkeypatch.setattr(mod, "smoothed_divergence", original_smoothed)
    monkeypatch.setattr(mod, "expanding_zscore", _full_sample_zscore)
    assert _prefix_violations(frames, cutoffs), (
        "full-sample standardization went undetected")
    monkeypatch.undo()
    assert _prefix_violations(frames, cutoffs) == []          # restored


# ------------- restricted bucket starting MID-WEEK, through the real engine

def _restricted_bucket_frames(frames: dict, start, end) -> dict:
    return {key: frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
            for key, frame in frames.items()}


def _mid_regime_start(index, held, want_asset, decisions, after=800):
    """A WEDNESDAY inside a stretch where `want_asset` is held for the whole
    surrounding three weeks and no decision date is on the Monday, Tuesday or
    that Wednesday, so a decision-bar-only signal would have nothing to go on
    at the bucket start."""
    dates = set(decisions)
    for k in range(after, len(index) - 260):
        if index[k].weekday() != 2:
            continue
        if any(a != want_asset for a in held[k - 3:k + 13]):
            continue
        if any(index[j] in dates for j in range(k - 2, k + 1)):
            continue
        return k
    raise AssertionError(f"no mid-week {want_asset} start found")


@pytest.mark.parametrize("want_asset", [RISK, DEFENSIVE])
def test_a_bucket_starting_mid_week_holds_the_carried_in_asset_from_its_second_bar(
        synth, want_asset):
    """REAL engine on a sliced bucket whose first session is mid-week inside a
    regime. The target factory is built once from the FULL frames (exactly as
    both real drivers do); the engine only sees the slice. The carried-in
    state must be usable immediately: the very first fill (bar 1's open) is
    the regime's asset, and every later session matches the full run."""
    frames = synth["frames"]
    index = synth["index"]
    target = mod.make_target_fn(frames)
    full = list(synth["active"])
    k = _mid_regime_start(index, full, want_asset, synth["decisions"])
    bucket = _restricted_bucket_frames(frames, index[k], index[k + 120])
    trades, active, _ = _run_engine_with(bucket, target)
    held = list(active)
    assert held[0] is None
    assert held[1] == want_asset                  # carried in, not flat
    assert held[1:] == full[k + 1:k + 121]
    assert None not in held[1:]
    assert trades.iloc[0]["asset"] == want_asset
    assert trades.iloc[0]["entry_date"] == index[k + 1]


def test_a_bucket_whose_own_signal_slice_is_far_too_short_still_gets_the_full_history_state(synth):
    """The bucket driver slices the SIGNAL frames to the bucket too, leaving
    far fewer than 756 observations inside it. The state was computed from
    the full history at build time, so the slice's length cannot matter."""
    frames = synth["frames"]
    index = synth["index"]
    target = mod.make_target_fn(frames)
    full = list(synth["active"])
    k = _mid_regime_start(index, full, DEFENSIVE, synth["decisions"])
    bucket = _restricted_bucket_frames(frames, index[k], index[k + 60])
    assert len(bucket[MOVE]) < MIN_HISTORY
    _, active, _ = _run_engine_with(bucket, target)
    assert list(active)[1:] == full[k + 1:k + 61]


def _bucket_driver(frames: dict, holdout_start: str, holdout_end: str, target):
    return batch3.run_rotation_windows(
        {RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]},
        {MOVE: frames[MOVE], VIX: frames[VIX]},
        target,
        holdout_start=holdout_start, holdout_end=holdout_end,
        cost_bps_by_asset={RISK: 1.0, DEFENSIVE: 1.0},
        tax_haircut_rate=0.0)


def test_real_bucket_driver_is_holdout_blind_and_carries_the_regime_into_the_holdout(synth):
    """scripts/run_batch3_mining.run_rotation_windows is a per-trade bucket
    driver: in-sample bucket =
    everything before the holdout start, holdout bucket = a lookback-padded
    window with spanning trades marked to market at the first real holdout
    session."""
    frames = synth["frames"]
    index = synth["index"]
    target = mod.make_target_fn(frames)
    full = list(synth["active"])
    k = _mid_regime_start(index, full, DEFENSIVE, synth["decisions"])
    holdout_start, holdout_end = index[k], index[k + 100]
    result = _bucket_driver(frames, holdout_start.strftime("%Y-%m-%d"),
                            holdout_end.strftime("%Y-%m-%d"), target)
    first = result["holdout_trades"].iloc[0]
    assert first["asset"] == DEFENSIVE
    assert first["entry_date"] == holdout_start
    assert result["alignment"]["holdout"]["warmup_trades_marked_to_market"] >= 1
    # blind: wreck EVERYTHING dated on or after the holdout start; the
    # in-sample trades cannot move
    wrecked = {key: _corrupt_after(frame, holdout_start - pd.Timedelta(days=1), 7.0)
               for key, frame in frames.items()}
    wrecked_result = _bucket_driver(
        wrecked, holdout_start.strftime("%Y-%m-%d"),
        holdout_end.strftime("%Y-%m-%d"), mod.make_target_fn(wrecked))
    pd.testing.assert_frame_equal(
        result["in_sample_trades"], wrecked_result["in_sample_trades"])
    assert len(result["in_sample_trades"]) >= 5


# ------------------------- fetched-frame loader (synthetic fetches, no network)

def _load_real_frame(raw, ticker):
    """One fetched OHLC frame -> the normalized frame the driver would load.

    Mirrors run_batch5_mining.load_frame's uncached path: blank rows first,
    through the driver's OWN helper (``_drop_all_nan_rows``), then
    run_batch3_mining._normalize_ohlc with the strict positivity gate ON for
    the two TRADED symbols and OFF for the signal-only volatility indices.
    The driver's helper is used on purpose: ``DataFrame.dropna`` cannot do
    this job, because the fetch carries ``Date`` as a COLUMN and a blank
    row's Date is not null."""
    import run_batch5_mining as batch5
    frame, _dropped = batch5._drop_all_nan_rows(raw)
    return batch3._normalize_ohlc(
        frame, ticker, require_positive=ticker not in (MOVE, VIX))


def _load_real_frames(fetch):
    """The four fetched, driver-cleaned frames, or a loud pytest skip.

    ``fetch(ticker, start="max")`` has data_pipeline.fetch_ohlc's signature;
    the tests here pass fakes. A fetch that returns nothing, or that cannot be
    normalized (a partly-blank bad print, a non-positive TRADED price), skips
    with the reason instead of erroring at setup."""
    frames = {}
    for ticker in (RISK, DEFENSIVE, MOVE, VIX):
        raw = fetch(ticker, start="max")
        if raw is None or raw.empty:
            pytest.skip(f"real data fetch for {ticker!r} returned no data")
        try:
            frames[ticker] = _load_real_frame(raw, ticker)
        except ValueError as exc:
            pytest.skip(f"real data fetch for {ticker!r} did not normalize "
                        f"({exc}); checks that need this fetch cannot run")
    return frames


def _synthetic_fetch_with_blank_trailing_row():
    """A fetch shaped exactly like data_pipeline.fetch_ohlc's return: ``Date``
    is a COLUMN on a RangeIndex, and the last row is a still-in-progress
    session the vendor has not printed (every OHLC field blank)."""
    dates = pd.bdate_range("2026-09-14", periods=6)
    raw = pd.DataFrame({
        "Date": dates,
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0, np.nan],
        "High": [101.0, 102.0, 103.0, 104.0, 105.0, np.nan],
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0, np.nan],
        "Close": [100.5, 101.5, 102.5, 103.5, 104.5, np.nan],
    })
    assert raw["Date"].notna().all() and isinstance(raw.index, pd.RangeIndex)
    return raw


def test_real_fixture_loader_drops_a_blank_trailing_row_like_the_driver(tmp_path):
    """A fetched frame carries ``Date`` as a column, so a plain
    ``DataFrame.dropna(how="all")`` sees a non-null Date on the blank row and
    drops NOTHING, and the strict-positivity normalizer would then refuse the
    blank row. The fixture loader must agree with the driver's own loader on
    the identical fetch."""
    import run_batch5_mining as batch5
    raw = _synthetic_fetch_with_blank_trailing_row()
    driver_frame, source = batch5.load_frame(
        RISK, ohlc_loader=lambda symbol: raw.copy(),
        cache_dir=str(tmp_path / "cache"))
    assert source == "fetched" and len(driver_frame) == 5
    fixture_frame = _load_real_frame(raw.copy(), RISK)
    pd.testing.assert_frame_equal(fixture_frame, driver_frame)
    assert fixture_frame.index[-1] == pd.Timestamp("2026-09-18")


def test_real_fixture_loader_keeps_failing_closed_on_a_partly_blank_row():
    """Only a row with NO prices at all is a session the vendor never
    printed. A row with SOME prices is a bad print and must still be refused,
    exactly as the driver refuses it (never silently repaired)."""
    raw = _synthetic_fetch_with_blank_trailing_row()
    raw.loc[5, "Open"] = 105.0        # last row now has one price, three blanks
    with pytest.raises(ValueError, match="finite"):
        _load_real_frame(raw, RISK)


def test_real_frames_loader_cleans_all_four_symbols_and_exempts_signal_only():
    """The whole loading loop on a FAKE fetch (no network): every symbol's
    blank trailing row is dropped, and the positivity gate is ON for the two
    traded symbols but OFF for the signal-only volatility indices."""
    def fake_fetch(ticker, start="max"):
        raw = _synthetic_fetch_with_blank_trailing_row()
        if ticker == MOVE:
            raw.loc[2, ["Open", "High", "Low", "Close"]] = 0.0   # legitimate
        return raw
    try:
        frames = _load_real_frames(fake_fetch)
    except pytest.skip.Exception as skipped:
        # A skip here would hide the failure this test guards against (a
        # blank row the loader failed to drop makes normalization raise,
        # which the loader turns into a skip), so it must FAIL, never skip.
        pytest.fail(f"the loader skipped on clean synthetic input: {skipped}")
    assert set(frames) == {RISK, DEFENSIVE, MOVE, VIX}
    for ticker, frame in frames.items():
        assert len(frame) == 5, ticker
        assert frame.index[-1] == pd.Timestamp("2026-09-18"), ticker
    assert (frames[MOVE]["Close"] == 0.0).any()


def test_real_frames_loader_skips_instead_of_erroring_when_a_fetch_is_unusable():
    """A fetch that returns nothing, or that cannot be normalized (a bad
    print with some prices, or a non-positive TRADED price), becomes a loud
    skip with the reason, never a setup error."""
    def empty_fetch(ticker, start="max"):
        return pd.DataFrame()
    with pytest.raises(pytest.skip.Exception, match="returned no data"):
        _load_real_frames(empty_fetch)

    def partly_blank_fetch(ticker, start="max"):
        raw = _synthetic_fetch_with_blank_trailing_row()
        raw.loc[5, "Open"] = 105.0
        return raw
    with pytest.raises(pytest.skip.Exception, match="did not normalize"):
        _load_real_frames(partly_blank_fetch)

    def non_positive_traded_fetch(ticker, start="max"):
        raw = _synthetic_fetch_with_blank_trailing_row()
        if ticker == RISK:
            raw.loc[2, ["Open", "High", "Low", "Close"]] = 0.0
        return raw
    with pytest.raises(pytest.skip.Exception, match=r"SPY.*did not normalize"):
        _load_real_frames(non_positive_traded_fetch)


# --------------------------------------------------- small guard rails

def test_a_week_with_every_weekday_closed_raises_instead_of_looping(monkeypatch):
    monkeypatch.setattr(mod, "is_scheduled_closure", lambda day: True)
    with pytest.raises(ValueError, match="no scheduled session"):
        mod.scheduled_decision_date(pd.Timestamp("2024-03-06"))


def test_scheduled_closure_accepts_tz_aware_and_time_of_day_timestamps():
    assert mod.is_scheduled_closure(
        pd.Timestamp("2024-03-29 15:30", tz="America/New_York")) is True
    assert mod.is_scheduled_closure(pd.Timestamp("2024-03-28 09:31")) is False


def test_signal_building_blocks_validate_their_arguments():
    series = _close_series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="at least 2"):
        mod.expanding_zscore(series, min_history=1)
    with pytest.raises(ValueError, match="finite"):
        mod.expanding_zscore(_close_series([1.0, np.nan, 3.0]), min_history=2)
    with pytest.raises(ValueError, match="window"):
        mod.smoothed_divergence(series, series, min_history=2, window=0)


def test_a_series_shorter_than_the_window_has_no_reading_and_no_crash():
    short = _close_series([1.0, 2.0, 3.0])
    smoothed = mod.smoothed_divergence(short, short, min_history=2, window=10)
    assert smoothed.isna().all() and len(smoothed) == 3
    state = mod.derisk_state_series(short, short, min_history=2, window=10,
                                    threshold=0.5)
    assert state.empty
    empty = _close_series([])
    assert mod.derisk_state_series(empty, empty, min_history=2, window=3,
                                   threshold=0.5).empty


def test_make_target_fn_rejects_a_frames_argument_that_is_not_a_dict():
    with pytest.raises(ValueError, match="non-empty"):
        mod.make_target_fn([1, 2, 3])
    with pytest.raises(ValueError, match="non-empty"):
        mod.make_target_fn(None)


def test_the_module_reads_only_close_from_the_signal_frames(synth):
    """Open/High/Low of the two volatility indices are never read: replacing
    them with garbage cannot move a decision."""
    frames = synth["frames"]
    variant = dict(frames)
    for key in (MOVE, VIX):
        frame = frames[key].copy()
        frame[["Open", "High", "Low"]] = -123.0
        variant[key] = frame
    index = synth["index"]
    original = mod.make_target_fn(frames)
    rebuilt = mod.make_target_fn(variant)
    for cutoff in _calendar_cutoffs(index, _flip_dates(synth["states"]),
                                    offsets=(0, 1))[::5]:
        view = _truncate(frames, index[cutoff])
        assert rebuilt(view) == original(view)


# --------------------------------------------------------- boundary pins

def test_smoothed_divergence_is_an_exact_direct_mean_not_a_running_mean():
    """After a huge one-off outlier leaves the window a running (add/remove)
    mean keeps a rounding residue that can be far larger than any threshold
    margin; the module's mean is recomputed directly from each window."""
    rng = np.random.RandomState(0)
    n = 400
    index = pd.bdate_range("2015-01-05", periods=n)
    move = _close_series(90.0 + rng.normal(0.0, 1.5, n), index)
    vix = _close_series(18.0 + rng.normal(0.0, 0.5, n), index)
    move.iloc[200] = 1.0e15
    dfg = mod.divergence_series(move, vix, min_history=50)
    smoothed = mod.smoothed_divergence(move, vix, min_history=50, window=10)
    values = dfg.to_numpy()
    worst = 0.0
    for t in range(50 + 9, n):
        worst = max(worst, abs(smoothed.iloc[t] - np.mean(values[t - 9:t + 1])))
    assert worst <= 1e-12
    # power guard: a running mean really does drift on this input
    running = dfg.rolling(10, min_periods=10).mean().to_numpy()
    assert np.nanmax(np.abs(running[59:] - smoothed.to_numpy()[59:])) > 1e-6


def _spike_frames_with_friday_at(position: int, *, n: int = 800) -> dict:
    """A plain weekday calendar whose session number `position` (0-based) is a
    non-holiday Friday, with a huge, persistent ^MOVE jump starting at session
    740 so that every smoothed reading defined after the warm-up is far above
    the threshold."""
    weekday_needed = (4 - position) % 5           # Monday=0 .. Friday=4
    start = pd.Timestamp("2010-01-04")
    start += pd.Timedelta(days=(weekday_needed - start.weekday()) % 7)
    index = pd.bdate_range(start, periods=n)
    assert index[position].weekday() == 4
    assert not mod.is_scheduled_closure(index[position])
    rng = np.random.RandomState(4)
    move = 90.0 + rng.normal(0.0, 1.0, n)
    move[740:] += 300.0
    vix = 18.0 + rng.normal(0.0, 0.3, n)
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    shy = 80.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.001, n)))
    return {RISK: _ohlc_from_close(spy, index),
            DEFENSIVE: _ohlc_from_close(shy, index),
            MOVE: _ohlc_from_close(move, index),
            VIX: _ohlc_from_close(vix, index)}


@pytest.mark.parametrize("position, expected", [
    (764, RISK),        # 756 + 10 - 1 = 765 is the first defined reading: none yet
    (765, DEFENSIVE),   # ... and it lands exactly here (a Friday)
])
def test_the_first_reading_lands_exactly_at_min_history_plus_window_minus_one(
        position, expected):
    """Pins the two constants together at the factory: the first smoothed
    reading exists at observation 756 + 10 - 1 (0-based) and not before or
    after, so an off-by-one in either constant moves a decision."""
    frames = _spike_frames_with_friday_at(position)
    index = frames[RISK].index
    target = mod.make_target_fn(frames)
    assert target(_truncate(frames, index[position])) == expected
    if expected == DEFENSIVE:
        # the reading carries through the following (non-decision) session
        assert target(_truncate(frames, index[position + 1])) == DEFENSIVE
