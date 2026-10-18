"""Tests for src/mining/batch5/preferred_stock_yield_spread_reversion.py.

Synthetic series only (no network, deterministic); the module's data file,
the ledger and the catalog are never read. The fetcher-script tests feed
hand-written publisher payloads to its parsers.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import backtest_engine  # noqa: E402
from mining.batch5 import preferred_stock_yield_spread_reversion as mod  # noqa: E402

# First bar that can carry a z-score: the yield needs 252 bars (valid from
# position 251), the z-score 756 yields on top (positions 251..1006).
FIRST_Z = mod.TRAILING_YIELD_TRADING_DAYS + mod.ZSCORE_WINDOW_TRADING_DAYS - 2


# ------------------------------------------------------------ fixtures

def _synthetic_inputs(n_bars: int = 1700, *, drop_at: int = 1200,
                      recover_at: int = 1330, start="2018-01-02"):
    """Unadjusted close + monthly-ish distributions with a KNOWN shape.

    Distributions: 0.2 per share every 21 bars (positions 20, 41, ...), so
    any 252-bar trailing window holds exactly 12 of them (sum 2.4).
    Close: 40 plus a small deterministic sinusoid (so the yield has a
    non-zero rolling std) with TWO depressed stretches:
      * a variance ANCHOR [200, 480) at 34, entirely inside the warm-up,
        which sits in every 756-bar z window up to bar ~1235 and keeps the
        baseline z inside the hysteresis band (a pure sinusoid's own z has
        amplitude ~1.4 and would trip the 0.75 entry by itself);
      * the tested DROP [drop_at, recover_at) at 30 -- the yield jumps from
        ~0.06 to 0.08 (z far above 0.75), then on recovery the yield falls
        back below its now-inflated trailing mean (z below 0): exactly one
        entry and one exit.
    """
    index = pd.bdate_range(start=start, periods=n_bars)
    t = np.arange(n_bars)
    close = 40.0 + 0.5 * np.sin(2 * np.pi * t / 126.0)
    close[200:480] = 34.0
    close[drop_at:recover_at] = 30.0
    close = pd.Series(close, index=index)
    div_positions = np.arange(20, n_bars, 21)
    dividends = pd.Series(0.2, index=index[div_positions])
    return close, dividends


def _frame_from_close(close: pd.Series) -> pd.DataFrame:
    values = close.to_numpy(dtype=float)
    return pd.DataFrame({"Open": values, "High": values * 1.001,
                         "Low": values * 0.999, "Close": values},
                        index=close.index)


def _first_bar_of_month_positions(index: pd.DatetimeIndex) -> list:
    periods = index.to_period("M")
    return [0] + [i for i in range(1, len(index)) if periods[i] != periods[i - 1]]


def _flags_via_factory(close, dividends, bucket, ticker="PFF"):
    frame = _frame_from_close(close)
    inputs = {ticker: {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn(ticker, bucket, {ticker: frame}, inputs=inputs)
    entries = np.array([signal_fn(frame.loc[:bucket[i]], i).entry
                        for i in range(len(bucket))])
    exits = np.array([signal_fn(frame.loc[:bucket[i]], i).exit
                      for i in range(len(bucket))])
    return entries, exits


# ------------------------------------------------------- yield arithmetic

def test_trailing_distribution_yield_known_value_and_warmup():
    close, dividends = _synthetic_inputs(600, drop_at=10**9, recover_at=10**9)
    close[:] = 40.0
    yld = mod.trailing_distribution_yield(
        close, dividends, trailing_days=mod.TRAILING_YIELD_TRADING_DAYS)
    assert yld.index.equals(close.index)
    # No full 252-bar window yet -> NaN, never an understated yield.
    assert yld.iloc[:mod.TRAILING_YIELD_TRADING_DAYS - 1].isna().all()
    # Exactly 12 distributions of 0.2 in every full window -> 2.4 / 40.
    filled = yld.iloc[mod.TRAILING_YIELD_TRADING_DAYS - 1:]
    assert filled.notna().all()
    assert np.allclose(filled.to_numpy(), 2.4 / 40.0)


def test_dividend_on_a_non_trading_date_counts_from_the_next_session_only():
    close, _ = _synthetic_inputs(300, drop_at=10**9, recover_at=10**9)
    close[:] = 50.0
    saturday = pd.Timestamp("2018-06-02")            # not in a bdate index
    assert saturday not in close.index
    dividends = pd.Series([1.0], index=[saturday])
    yld = mod.trailing_distribution_yield(close, dividends, trailing_days=5)
    next_session = close.index[close.index.searchsorted(saturday)]
    before = close.index[close.index.get_loc(next_session) - 1]
    assert yld.loc[before] == 0.0
    assert yld.loc[next_session] == pytest.approx(1.0 / 50.0)


# ------------------------------------------------------- standing state

def test_standing_state_hand_built_hysteresis_at_first_bar_of_month():
    # One z value per calendar month; decisions land on the month's FIRST bar.
    #   m0 NaN  -> FLAT (no history)
    #   m1 0.2  -> FLAT (inside the band while flat: no entry)
    #   m2 0.8  -> LONG (z >= 0.75)
    #   m3 0.5  -> LONG (hold inside the band)
    #   m4 0.1  -> LONG (still > 0)
    #   m5 0.0  -> FLAT (z <= 0 exits; equality exits)
    #   m6 0.3  -> FLAT (band while flat)
    #   m7 0.75 -> LONG (equality enters)
    #   m8 -0.5 -> FLAT
    #   m9 NaN  -> FLAT (a NaN decision goes flat)
    monthly_z = [np.nan, 0.2, 0.8, 0.5, 0.1, 0.0, 0.3, 0.75, -0.5, np.nan]
    expected = [False, False, True, True, True, False, False, True, False, False]
    months = pd.period_range("2020-01", periods=len(monthly_z), freq="M")
    dates, values = [], []
    for period, z in zip(months, monthly_z):
        days = pd.bdate_range(period.start_time, period.end_time)
        dates.extend(days)
        values.extend([z] * len(days))
    zscore = pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)
    # Make the z on NON-decision bars wildly different: they must be ignored.
    periods = zscore.index.to_period("M")
    first_of_month = np.r_[True, periods[1:] != periods[:-1]]
    zscore[~first_of_month] = 99.0
    state = mod.standing_long(zscore, entry_z=0.75, exit_z=0.0)
    assert state.index.equals(zscore.index)
    assert state.dtype == bool
    for period, want in zip(months, expected):
        in_month = state[periods == period]
        assert in_month.all() == want and in_month.any() == want, str(period)


def test_signal_flags_fire_where_they_should_and_stay_flat_elsewhere():
    close, dividends = _synthetic_inputs()
    flags = mod.signal_flags(close, dividends)
    assert list(flags.columns) == ["entry", "exit"]
    assert flags.index.equals(close.index)
    entry = flags["entry"].to_numpy()
    exit_ = flags["exit"].to_numpy()
    # Warm-up: no z-score before position FIRST_Z -> never long.
    assert not entry[:FIRST_Z].any()
    assert exit_[:FIRST_Z].all()
    decisions = _first_bar_of_month_positions(close.index)
    first_decision_after_drop = min(d for d in decisions if d >= 1200)
    first_decision_after_recovery = min(d for d in decisions if d >= 1330)
    # Flat right up to the first decision bar after the price drop ...
    assert not entry[FIRST_Z:first_decision_after_drop].any()
    # ... LONG from that decision bar through the first decision after recovery
    assert entry[first_decision_after_drop:first_decision_after_recovery].all()
    assert not exit_[first_decision_after_drop:first_decision_after_recovery].any()
    # ... and FLAT from the recovery decision onward (z fell to/below 0).
    assert not entry[first_decision_after_recovery:].any()
    assert exit_[first_decision_after_recovery:].all()
    # exit is the complement of the standing LONG state on every bar, so a
    # position the engine opened one bar late still finds an exit flag.
    assert np.array_equal(exit_, ~entry)


# ---------------------------------------------------------- look-ahead

def test_look_ahead_probe_perturbing_inputs_after_k_never_changes_flags_through_k():
    close, dividends = _synthetic_inputs()
    k = 1250                                    # inside the LONG stretch
    base = mod.signal_flags(close, dividends)
    close_p = close.copy()
    close_p.iloc[k + 1:] *= 1.3
    extra = pd.Series(5.0, index=close.index[k + 1::7])   # huge future payouts
    dividends_p = pd.concat([dividends, extra]).sort_index()
    perturbed = mod.signal_flags(close_p, dividends_p)
    assert base.iloc[:k + 1].equals(perturbed.iloc[:k + 1])
    # and the perturbation is real: something after k DID change
    assert not base.iloc[k + 1:].equals(perturbed.iloc[k + 1:])


def test_look_ahead_probe_through_the_factory_on_frames_and_inputs():
    close, dividends = _synthetic_inputs()
    bucket = close.index[1100:]
    k = 150                                     # bucket position
    base_e, base_x = _flags_via_factory(close, dividends, bucket)
    close_p = close.copy()
    close_p.iloc[1100 + k + 1:] *= 1.3
    frame_p = _frame_from_close(close_p)
    frame_p["High"] = frame_p["High"] * 1.2
    frame_p["Low"] = frame_p["Low"] * 0.8
    inputs = {"PFF": {"unadjusted_close": close_p, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("PFF", bucket, {"PFF": frame_p}, inputs=inputs)
    for i in range(k + 1):
        result = signal_fn(frame_p.loc[:bucket[i]], i)
        assert result.entry == base_e[i] and result.exit == base_x[i], i


def test_holdout_blind_inputs_after_the_bucket_end_are_irrelevant():
    close, dividends = _synthetic_inputs()
    bucket = close.index[1100:1400]
    base_e, base_x = _flags_via_factory(close, dividends, bucket)
    close_p = close.copy()
    close_p.iloc[1400:] = 5.0                   # absurd future prices
    dividends_p = pd.concat([dividends, pd.Series(9.0, index=close.index[1400::3])]).sort_index()
    pert_e, pert_x = _flags_via_factory(close_p, dividends_p, bucket)
    assert np.array_equal(base_e, pert_e) and np.array_equal(base_x, pert_x)


# --------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_one_trade_with_the_expected_dates():
    close, dividends = _synthetic_inputs()
    frame = _frame_from_close(close)
    bucket_frame = frame.iloc[1100:].copy()           # ~600 bars
    inputs = {"PFF": {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("PFF", bucket_frame.index, {"PFF": frame},
                                   inputs=inputs)
    trades, in_market = backtest_engine.run_backtest(bucket_frame, signal_fn)
    assert len(trades) >= 1
    assert len(trades) == 1
    decisions = _first_bar_of_month_positions(close.index)
    entry_decision = min(d for d in decisions if d >= 1200)
    exit_decision = min(d for d in decisions if d >= 1330)
    # entry fills at the NEXT open after the LONG decision bar; exit at the
    # FLAT decision bar's own close.
    assert trades["entry_date"].iloc[0] == close.index[entry_decision + 1]
    assert trades["exit_date"].iloc[0] == close.index[exit_decision]
    assert trades["reason"].iloc[0] == "signal_exit"
    assert trades["entry_px"].iloc[0] == pytest.approx(30.0)
    assert in_market.sum() == exit_decision - entry_decision


def test_bucket_that_opens_inside_a_long_regime_enters_at_first_fill():
    close, dividends = _synthetic_inputs()
    frame = _frame_from_close(close)
    bucket_frame = frame.iloc[1260:1500].copy()        # opens mid-position
    inputs = {"PFF": {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("PFF", bucket_frame.index, {"PFF": frame},
                                   inputs=inputs)
    trades, _ = backtest_engine.run_backtest(bucket_frame, signal_fn)
    assert len(trades) == 1
    assert trades["entry_date"].iloc[0] == bucket_frame.index[1]


def test_signal_fn_is_a_lookup_and_ignores_the_slice():
    close, dividends = _synthetic_inputs()
    bucket = close.index[1100:]
    frame = _frame_from_close(close)
    inputs = {"PFF": {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("PFF", bucket, {"PFF": frame}, inputs=inputs)
    a = signal_fn(frame.loc[:bucket[120]], 120)
    b = signal_fn(None, 120)
    assert (a.entry, a.exit) == (b.entry, b.exit)
    with pytest.raises(IndexError):
        signal_fn(None, len(bucket))


# ------------------------------------------------------------ guards

def test_factory_fails_closed_when_the_data_file_does_not_cover_the_bucket():
    close, dividends = _synthetic_inputs()
    frame = _frame_from_close(close)
    stale_close = close.iloc[:1500]
    inputs = {"PFF": {"unadjusted_close": stale_close, "dividends": dividends}}
    with pytest.raises(ValueError, match="ends"):
        mod.make_signal_fn("PFF", close.index[1100:], {"PFF": frame}, inputs=inputs)


def test_factory_fails_closed_on_a_missing_pool_member():
    close, dividends = _synthetic_inputs()
    frame = _frame_from_close(close)
    inputs = {"PFF": {"unadjusted_close": close, "dividends": dividends}}
    with pytest.raises(ValueError, match="PGX"):
        mod.make_signal_fn("PGX", close.index[1100:], {"PGX": frame}, inputs=inputs)


def test_zero_rolling_std_yields_no_decision_not_a_spurious_long():
    close, dividends = _synthetic_inputs(1300, drop_at=10**9, recover_at=10**9)
    close[:] = 40.0                              # perfectly constant yield
    flags = mod.signal_flags(close, dividends)
    assert not flags["entry"].any()


# ------------------------------------------- distribution-calendar hygiene
#
# Completeness cannot be checked against the vendor itself: comparing Yahoo's
# Dividends column with Yahoo's own Ticker.dividends endpoint agrees even
# though Yahoo is missing five monthly distributions that independent
# publishers list. The trailing yield is a SUM, so each hole depresses the
# yield (and drags the z-score) for a full 252 sessions; the checks below
# catch a hole from the calendar's own shape.

CURRENCY = chr(36)          # the publishers' currency prefix, never a literal


def _monthly_series(start="2015-01-15", months=48, amount=0.07):
    stamps = [pd.Timestamp(start) + pd.DateOffset(months=k)
              for k in range(months)]
    return pd.Series([amount] * len(stamps), index=pd.DatetimeIndex(stamps))


def _series_from_dates(dates, amount=0.07):
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates]).sort_values()
    return pd.Series([amount] * len(index), index=index)


EMPTY_PROBLEMS = {"month_gaps": [], "short_years": {}, "tail_gaps": []}


def test_calendar_check_accepts_a_clean_monthly_series():
    problems = mod.distribution_calendar_problems(_monthly_series())
    assert problems == EMPTY_PROBLEMS
    mod.validate_distribution_calendar(_monthly_series(), "clean")


def test_calendar_check_accepts_the_year_end_pull_forward_pattern():
    """A late-December second ex-date followed by an empty January is the ONE
    benign gap in these funds' real histories (it is why PFF/PFFD show an
    empty January every year) -- it must not raise, or the guard would be
    unusable on the real data."""
    dates = []
    for year in (2019, 2020, 2021, 2022):
        dates += [f"{year}-{month:02d}-03" for month in range(2, 13)]
        dates.append(f"{year}-12-27")            # pulled-forward January
    series = _series_from_dates(dates)
    assert mod.distribution_calendar_problems(series) == EMPTY_PROBLEMS
    mod.validate_distribution_calendar(series, "pull_forward")


def test_calendar_check_flags_an_unexplained_month_gap():
    series = _monthly_series()
    dropped = series.index[20]                   # a plain interior month
    series = series.drop(dropped)
    problems = mod.distribution_calendar_problems(series)
    assert problems["month_gaps"] == [f"{dropped.year:04d}-{dropped.month:02d}"]
    with pytest.raises(ValueError, match="distribution calendar is incomplete"):
        mod.validate_distribution_calendar(series, "gappy")


def test_calendar_check_flags_a_complete_year_short_of_twelve_events():
    """A dropped January is EXPLAINED by the preceding December double, so the
    month-gap test alone cannot see it -- the per-complete-year count is what
    catches a vendor that drops one event from such a year."""
    dates = [f"2018-{month:02d}-03" for month in range(1, 13)]
    dates.append("2018-12-27")                   # pulls January 2019 forward
    dates += [f"2019-{month:02d}-03" for month in range(2, 13)]   # 11 events
    dates += [f"2020-{month:02d}-03" for month in range(1, 13)]
    series = _series_from_dates(dates)
    problems = mod.distribution_calendar_problems(series)
    assert problems["month_gaps"] == []
    assert problems["short_years"] == {2019: 11}
    with pytest.raises(ValueError, match="distribution calendar is incomplete"):
        mod.validate_distribution_calendar(series, "short_year")


# ------------------------------------- stale TAIL (the end-of-series hole)
#
# The interior month-gap and per-complete-year checks both derive their span
# FROM the dividend series, so a distribution dropped at the END moves the
# span end with it and neither check can see it. The newest months are the
# pinned holdout window, so that is the one place a hole moves holdout trades.


def test_calendar_check_cannot_see_a_dropped_tail_without_the_last_session():
    """Pins the blind spot the tail check exists to close: with no
    ``last_session`` the series looks perfectly complete."""
    series = _monthly_series(start="2018-01-15", months=30)   # ends 2020-06-15
    assert mod.distribution_calendar_problems(series) == EMPTY_PROBLEMS


def test_calendar_check_flags_a_tail_dropped_before_the_last_price_bar():
    series = _monthly_series(start="2018-01-15", months=30)   # ends 2020-06-15
    problems = mod.distribution_calendar_problems(
        series, last_session=pd.Timestamp("2020-09-10"))
    # required through the last COMPLETE month before the final bar
    assert problems["month_gaps"] == [] and problems["short_years"] == {}
    assert problems["tail_gaps"] == ["2020-07", "2020-08"]
    with pytest.raises(ValueError, match="distribution calendar is incomplete"):
        mod.validate_distribution_calendar(
            series, "stale_tail", last_session=pd.Timestamp("2020-09-10"))


def test_tail_check_does_not_require_the_incomplete_final_month():
    """Only COMPLETE months are required -- the same rule the per-year check
    uses. A not-yet-due ex-date in the month containing the final price bar
    must not fail-close a healthy member."""
    series = _monthly_series(start="2018-01-15", months=33)   # ends 2020-09-15
    for last in ("2020-09-16", "2020-09-30", "2020-10-01", "2020-10-31"):
        assert mod.distribution_calendar_problems(
            series, last_session=pd.Timestamp(last)) == EMPTY_PROBLEMS, last
        mod.validate_distribution_calendar(
            series, "current_month", last_session=pd.Timestamp(last))
    # ... and the month after that one IS required
    problems = mod.distribution_calendar_problems(
        series, last_session=pd.Timestamp("2020-11-02"))
    assert problems["tail_gaps"] == ["2020-10"]


def test_tail_check_still_applies_to_a_one_event_series():
    """A series collapsed to a single event is the same defect further along;
    the interior checks are vacuous there, so only the tail check can see it.
    An empty series stays a no-op (no cadence to reason from)."""
    single = _monthly_series(start="2019-04-15", months=1)
    assert mod.distribution_calendar_problems(single) == EMPTY_PROBLEMS
    assert mod.distribution_calendar_problems(
        single, last_session=pd.Timestamp("2019-05-20"))["tail_gaps"] == []
    problems = mod.distribution_calendar_problems(
        single, last_session=pd.Timestamp("2019-08-02"))
    assert problems["tail_gaps"] == ["2019-05", "2019-06", "2019-07"]
    with pytest.raises(ValueError, match="distribution calendar is incomplete"):
        mod.validate_distribution_calendar(
            single, "one_event", last_session=pd.Timestamp("2019-08-02"))
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert mod.distribution_calendar_problems(
        empty, last_session=pd.Timestamp("2019-08-02")) == EMPTY_PROBLEMS


def test_tail_check_honours_the_year_end_pull_forward():
    """A two-ex-date December followed by an empty January is benign at the
    TAIL exactly as it is in the interior."""
    dates = [f"2019-{month:02d}-03" for month in range(1, 13)]
    dates.append("2019-12-27")                   # pulls January 2020 forward
    series = _series_from_dates(dates)
    assert mod.distribution_calendar_problems(
        series, last_session=pd.Timestamp("2020-02-14")) == EMPTY_PROBLEMS


# ------------------------------------------------------------- fetcher script

def _fetcher():
    import importlib.util
    path = os.path.join(_ROOT, "scripts",
                        f"fetch_batch5_events_{mod.FAMILY_ID}.py")
    assert os.path.exists(path), path
    spec = importlib.util.spec_from_file_location("_batch5_pref_fetcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)              # no network at import
    return module


def test_fetcher_parsers_read_each_publisher_shape():
    fetcher = _fetcher()
    assert fetcher.parse_digrin(
        "<tr>\n<td>2016-03-15</td>\n<td>2016-03-31</td>\n<td>0.0728 " "USD"
    ) == {"2016-03-15": 0.0728}
    assert fetcher.parse_dividendhistory(
        "<tr class><td>2016-04-15</td><td>2016-04-29</td><td>"
        + CURRENCY + "0.0731</td>") == {"2016-04-15": 0.0731}
    assert fetcher.parse_stockanalysis(
        '{dt:"2021-09-20",amt:"' + CURRENCY + '0.05923",dec:"n/a"}'
    ) == {"2021-09-20": 0.05923}
    payload = json.dumps({"data": {"dividends": {"rows": [
        {"exOrEffDate": "11/01/2012", "type": "Cash",
         "amount": CURRENCY + "0.211135"},
        {"exOrEffDate": "12/01/2012", "type": "Capital Gain",
         "amount": CURRENCY + "0.5"}]}}})
    # only CASH rows are kept: a capital-gains distribution is not cash yield
    assert fetcher.parse_nasdaq(payload) == {"2012-11-01": 0.211135}


def test_fetcher_nasdaq_unavailable_symbol_is_absent_not_agreement():
    fetcher = _fetcher()
    payload = json.dumps({
        "data": {"dividends": {"asOf": None, "headers": None, "rows": None}},
        "message": "Dividend History for Non-Nasdaq symbols is not available"})
    assert fetcher.parse_nasdaq(payload) == {}


def test_fetcher_only_adds_events_confirmed_by_two_independent_sources():
    fetcher = _fetcher()
    vendor = ["2020-01-15", "2020-03-16"]
    independent = {
        "a": {"2020-02-14": 0.07, "2020-04-15": 0.07},
        "b": {"2020-02-14": 0.0701},
        "c": {"2020-04-16": 0.07},
    }
    additions, rejections = fetcher.confirmed_missing(
        vendor, independent, cutoff="2020-12-31")
    assert sorted(additions) == ["2020-02-14"]
    assert additions["2020-02-14"]["amount"] == 0.0701   # most precise agreeing
    assert additions["2020-02-14"]["sources"] == ["a", "b"]
    assert rejections["2020-04-15"] == "single_source"
    assert rejections["2020-04-16"] == "single_source"


def test_fetcher_rejects_a_publisher_date_shift_and_a_future_ex_date():
    fetcher = _fetcher()
    vendor = ["2020-01-15", "2020-02-14"]
    independent = {
        "a": {"2020-02-13": 0.07, "2020-06-15": 0.07},
        "b": {"2020-02-13": 0.07, "2020-06-15": 0.07},
    }
    additions, rejections = fetcher.confirmed_missing(
        vendor, independent, cutoff="2020-03-31")
    assert additions == {}
    assert rejections["2020-02-13"] == "near_duplicate_of_2020-02-14"
    assert rejections["2020-06-15"] == "after_cutoff"


def test_fetcher_reconcile_fails_closed_when_too_few_sources_answer():
    fetcher = _fetcher()
    dates = [pd.Timestamp("2015-01-15") + pd.DateOffset(months=k)
             for k in range(36)]
    calendar = pd.DatetimeIndex(dates)
    vendor = pd.Series([0.07] * len(dates), index=calendar).drop(dates[18])
    with pytest.raises(fetcher.ReconciliationError, match="only 0 independent"):
        fetcher.reconcile("X", vendor, calendar, {"a": {}, "b": {}})


def test_fetcher_reconcile_fails_closed_on_a_surviving_month_gap():
    """A gap NO independent source can fill is an error, never a silent zero."""
    fetcher = _fetcher()
    dates = [pd.Timestamp("2015-01-15") + pd.DateOffset(months=k)
             for k in range(36)]
    calendar = pd.DatetimeIndex(dates)
    vendor = pd.Series([0.07] * len(dates), index=calendar).drop(dates[18])
    echo = {str(d.date()): 0.07 for d in vendor.index}
    with pytest.raises(fetcher.ReconciliationError, match="still incomplete"):
        fetcher.reconcile("X", vendor, calendar,
                          {"a": dict(echo), "b": dict(echo)})


def test_fetcher_reconcile_merges_and_passes_once_the_gap_is_filled():
    fetcher = _fetcher()
    dates = [pd.Timestamp("2015-01-15") + pd.DateOffset(months=k)
             for k in range(36)]
    calendar = pd.DatetimeIndex(dates)
    full = {str(d.date()): 0.07 for d in dates}
    missing = str(dates[18].date())
    vendor = pd.Series([0.07] * len(dates), index=calendar).drop(dates[18])
    merged, report = fetcher.reconcile(
        "X", vendor, calendar, {"a": dict(full), "b": dict(full)})
    assert len(merged) == len(dates)
    assert pd.Timestamp(missing) in merged.index
    assert sorted(report["cross_source_additions"]) == [missing]
    assert report["month_gaps_after_merge"] == []
    assert report["calendar_years_off_expected_count_after_merge"] == {}
    assert report["amount_cross_check_report_only"] is True


def test_fetcher_tail_gaps_matches_the_loaders_tail_rule():
    """The fetcher and the loader must agree on the tail invariant, or a
    regenerated file could pass the fetch and then be refused at load."""
    fetcher = _fetcher()
    series = _monthly_series(start="2018-01-15", months=30)   # ends 2020-06-15
    for last in ("2020-06-30", "2020-07-31", "2020-09-10", "2020-12-01"):
        stamp = pd.Timestamp(last)
        assert fetcher.tail_gaps(series.index, stamp) == \
            mod.distribution_calendar_problems(
                series, last_session=stamp)["tail_gaps"], last
    assert fetcher.tail_gaps(series.index, pd.Timestamp("2020-09-10")) == \
        ["2020-07", "2020-08"]


def test_fetcher_reconcile_fails_closed_on_a_stale_tail():
    """A vendor that dropped the NEWEST month(s) leaves a calendar the
    interior checks call complete -- the fetch must still refuse to write."""
    fetcher = _fetcher()
    dates = [pd.Timestamp("2015-01-15") + pd.DateOffset(months=k)
             for k in range(36)]
    calendar = pd.DatetimeIndex(dates + [pd.Timestamp("2018-03-15"),
                                         pd.Timestamp("2018-04-16")])
    vendor = pd.Series([0.07] * len(dates), index=pd.DatetimeIndex(dates))
    assert fetcher.month_gaps(vendor.index) == []
    assert fetcher.incomplete_years(vendor.index) == {}
    echo = {str(d.date()): 0.07 for d in vendor.index}
    with pytest.raises(fetcher.ReconciliationError, match="stale tail"):
        fetcher.reconcile("X", vendor, calendar,
                          {"a": dict(echo), "b": dict(echo)})


def test_fetcher_reconcile_reports_a_clean_tail():
    fetcher = _fetcher()
    dates = [pd.Timestamp("2015-01-15") + pd.DateOffset(months=k)
             for k in range(36)]
    calendar = pd.DatetimeIndex(dates)
    full = {str(d.date()): 0.07 for d in dates}
    vendor = pd.Series([0.07] * len(dates), index=calendar)
    _, report = fetcher.reconcile("X", vendor, calendar,
                                  {"a": dict(full), "b": dict(full)})
    assert report["tail_gaps_after_merge"] == []


def test_fetcher_reconcile_refuses_an_addition_that_is_not_a_session():
    fetcher = _fetcher()
    dates = [pd.Timestamp("2015-01-15") + pd.DateOffset(months=k)
             for k in range(36)]
    calendar = pd.DatetimeIndex([d for d in dates if d != dates[18]])
    vendor = pd.Series([0.07] * len(calendar), index=calendar)
    full = {str(d.date()): 0.07 for d in dates}
    with pytest.raises(fetcher.ReconciliationError, match="not sessions"):
        fetcher.reconcile("X", vendor, calendar,
                          {"a": dict(full), "b": dict(full)})
