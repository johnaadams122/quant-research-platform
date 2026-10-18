"""Tests for src/mining/batch5/cross_asset_financial_turbulence_derisking.py.

PRE-REGISTERED RULE (every constant fixed a priori; nothing below may be
tuned after looking at any backtest number)::

    panel  = SPY, EEM, TLT, GLD, LQD  (daily log returns, dates on which all
             five have a genuine one-session return)
    d_t    = (r_t - mean)' inv(cov) (r_t - mean), with mean and cov the
             EXPANDING sample statistics over sessions through t-1 only
             (undefined until 500 earlier sessions exist)
    m_t    = trailing 10-session mean of d_t
    at each SCHEDULED last session of the week (Friday, stepping back over a
    scheduled Friday closure):
        pct = share of strictly earlier m values that are strictly below m_t
              (needs 250 earlier m values)
        hold SHY for the next week if pct > 0.80, otherwise SPY
    the engine fills at the next session's open.

The tests are grouped: contract and catalog parity, the ex-ante week
schedule, panel construction, the turbulence statistic against a brute-force
oracle, fail-closed covariance handling, smoothing and percentile rank,
engine-level behaviour against an INDEPENDENT oracle, look-ahead and hold-out
blindness (with injected one-session look-ahead mutations that the harness
must catch), restricted-bucket consistency, degraded data, and the real ABI-1 driver wiring.
"""
import copy
import datetime
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
import run_batch5_mining as batch5_run  # noqa: E402
from mining import abi1_driver, rotation_engine  # noqa: E402
from mining.batch5 import cross_asset_financial_turbulence_derisking as mod  # noqa: E402

FAMILY_ID = "cross_asset_financial_turbulence_derisking"
RISK, DEFENSIVE = "SPY", "SHY"
PANEL = ["SPY", "EEM", "TLT", "GLD", "LQD"]
SIGNAL_ONLY = ["EEM", "TLT", "GLD", "LQD"]

# Pre-registered constants, restated here on purpose (not imported).
MIN_COV_SESSIONS = 500
WINDOW = 10
MIN_PCT_HISTORY = 250
THRESHOLD = 0.80
MAX_COND_EXPONENT = 8
MAX_COND = 10.0 ** MAX_COND_EXPONENT

REAL_CATALOG = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
THIS_FILE = os.path.abspath(__file__)


# ------------------------------------------------------------------ helpers

def _xnys(start, end) -> pd.DatetimeIndex:
    """XNYS sessions from the locally installed exchange_calendars package,
    an INDEPENDENT source of the exchange calendar (no network)."""
    import exchange_calendars as xc
    calendar = xc.get_calendar("XNYS", start=pd.Timestamp(start),
                               end=pd.Timestamp(end))
    index = pd.DatetimeIndex(calendar.sessions)
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


_VOL = np.array([0.010, 0.014, 0.008, 0.009, 0.004])
_CORR = np.array([
    [1.00, 0.70, -0.30, 0.05, 0.10],
    [0.70, 1.00, -0.20, 0.10, 0.15],
    [-0.30, -0.20, 1.00, 0.20, 0.55],
    [0.05, 0.10, 0.20, 1.00, 0.10],
    [0.10, 0.15, 0.55, 0.10, 1.00],
])


def _returns(n: int, seed: int, bursts=()) -> np.ndarray:
    """Synthetic daily log returns for the five panel assets; ``bursts`` is
    a list of (start_row, stop_row, scale) multiplying that stretch's moves,
    a turbulence episode."""
    rng = np.random.RandomState(seed)
    chol = np.linalg.cholesky(np.outer(_VOL, _VOL) * _CORR)
    out = rng.normal(0.0, 1.0, (n, 5)) @ chol.T
    for start, stop, scale in bursts:
        out[start:stop] *= scale
    return out + 0.0003


def _panel_df(n: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame(_returns(n, seed),
                        index=pd.bdate_range("2010-01-04", periods=n),
                        columns=PANEL)


def _ohlc_from_close(close, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _frames(index: pd.DatetimeIndex, returns: np.ndarray) -> dict:
    """The six frames the driver hands the target factory: the five panel
    assets (their log returns are exactly ``returns``) and SHY."""
    frames = {}
    for column, ticker in enumerate(PANEL):
        frames[ticker] = _ohlc_from_close(
            100.0 * np.exp(np.cumsum(returns[:, column])), index)
    rng = np.random.RandomState(5)
    shy = 80.0 * np.exp(np.cumsum(0.00005 + rng.normal(0.0, 0.0002, len(index))))
    frames[DEFENSIVE] = _ohlc_from_close(shy, index)
    return frames


def _scenario(seed: int = 11):
    """Five calm years with two turbulence bursts inside the readings era."""
    index = _xnys("2010-01-04", "2014-12-26")          # ends on a Friday
    returns = _returns(len(index), seed,
                       bursts=[(900, 915, 5.0), (1100, 1112, 6.0)])
    return index, returns, _frames(index, returns)


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _run_like_abi1(frames: dict, **factory_kwargs):
    """The wiring scripts/run_abi1_mining.py + abi1_driver.build_rotation_curve
    use: the target factory gets the FULL, UNALIGNED frames; the engine gets
    the two traded legs plus the signal-only frames intersected onto one
    calendar."""
    target_fn = mod.make_target_fn(frames, **factory_kwargs)
    aligned_assets, aligned_signals = abi1_driver._align_rotation_frames(
        {RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]},
        {ticker: frames[ticker] for ticker in SIGNAL_ONLY})
    trades, active = rotation_engine.run_two_asset_rotation(
        aligned_assets, target_fn,
        cost_bps_by_asset={RISK: 1.0, DEFENSIVE: 1.0},
        signal_frames=aligned_signals)
    return trades, active, aligned_assets


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------- the independent oracle

def _oracle_panel(frames: dict) -> pd.DataFrame:
    closes = pd.DataFrame({ticker: frames[ticker]["Close"] for ticker in PANEL})
    return np.log(closes).diff().dropna()


def _oracle_turbulence(returns: np.ndarray) -> np.ndarray:
    """Brute force straight from the definition: for each t, the sample mean
    and np.cov (ddof=1) of the returns BEFORE t, then a plain inverse."""
    n = len(returns)
    out = np.full(n, np.nan)
    for k in range(MIN_COV_SESSIONS, n):
        history = returns[:k]
        mean = history.mean(axis=0)
        cov = np.cov(history, rowvar=False)
        deviation = returns[k] - mean
        out[k] = float(deviation @ np.linalg.inv(cov) @ deviation)
    return out


def _oracle_readings(frames: dict) -> dict:
    """{week-end date: True if risk-on (hold SPY)} for every week whose last
    panel session has a defined reading. Sharing no code with the module:
    pandas rolling mean, ISO-week grouping of the panel's own dates (the
    calendar handed in is a real XNYS calendar with no holes, so the last
    panel session of a complete week IS the scheduled week end), a plain
    count for the percentile."""
    panel = _oracle_panel(frames)
    d = _oracle_turbulence(panel.to_numpy())
    m = pd.Series(d, index=panel.index).rolling(WINDOW, min_periods=WINDOW).mean()
    values = m.to_numpy()
    iso = panel.index.isocalendar()
    last_position = {}
    for position, key in enumerate(zip(iso["year"], iso["week"])):
        last_position[key] = position
    out = {}
    for position in sorted(last_position.values()):
        if not np.isfinite(values[position]):
            continue
        prior = values[:position]
        prior = prior[np.isfinite(prior)]
        if len(prior) < MIN_PCT_HISTORY:
            continue
        percentile = float((prior < values[position]).sum()) / len(prior)
        out[panel.index[position]] = not (percentile > THRESHOLD)
    return out


def _oracle_active(frames: dict, *, warmup=RISK) -> list:
    """The asset held DURING each session: a reading dated at week end w
    becomes the target on bar w and fills at the next session's open."""
    readings = _oracle_readings(frames)
    target_after_bar = []
    current = warmup
    for date in frames[RISK].index:
        if date in readings:
            current = RISK if readings[date] else DEFENSIVE
        target_after_bar.append(current)
    return [None] + target_after_bar[:-1]


# ------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]
    assert mod.SIGNAL == SIGNAL_ONLY
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_target_fn)
    assert not hasattr(mod, "make_signal_fn")     # rotation shape, no signal fn


def test_parameters_are_the_preregistered_values():
    assert mod.PARAMETERS == {
        "panel_tickers": ["SPY", "EEM", "TLT", "GLD", "LQD"],
        "risk_on_ticker": "SPY",
        "defensive_ticker": "SHY",
        "return_definition": "daily_log_return",
        "covariance_estimator": "expanding_sample_ddof1_through_t_minus_1",
        "covariance_min_sessions": 500,
        "max_covariance_condition_number_exponent": 8,
        "smoothing_window_sessions": 10,
        "percentile_min_history_sessions": 250,
        "percentile_definition":
            "share_of_strictly_earlier_smoothed_values_strictly_below",
        "defensive_percentile_above": 0.8,
        "decision_day": "scheduled_last_session_of_each_week",
        "rebalance": "weekly",
        "warmup_holding": "risk_on_ticker",
    }


def test_the_module_constants_equal_the_restated_preregistered_constants():
    p = mod.PARAMETERS
    assert p["covariance_min_sessions"] == MIN_COV_SESSIONS
    assert p["smoothing_window_sessions"] == WINDOW
    assert p["percentile_min_history_sessions"] == MIN_PCT_HISTORY
    assert p["defensive_percentile_above"] == THRESHOLD
    assert p["max_covariance_condition_number_exponent"] == MAX_COND_EXPONENT
    assert p["panel_tickers"] == [RISK] + SIGNAL_ONLY


def test_the_derived_condition_limit_is_exactly_the_preregistered_ten_to_the_eighth():
    assert mod.PARAMETERS["max_covariance_condition_number_exponent"] == 8
    assert isinstance(mod.PARAMETERS["max_covariance_condition_number_exponent"], int)
    assert "max_covariance_condition_number" not in mod.PARAMETERS
    assert mod._MAX_CONDITION == MAX_COND == 10.0 ** 8


def test_family_is_not_duplicated_in_the_real_catalog_and_agrees_if_merged():
    """The family appears at most once in the real catalog, and where it
    appears it carries the module's exact parameters."""
    matches = [c for c in _read_json(REAL_CATALOG) if c["family_id"] == FAMILY_ID]
    assert len(matches) <= 1
    if matches:
        assert json.dumps(matches[0]["parameters"], sort_keys=True) == \
            json.dumps(mod.PARAMETERS, sort_keys=True)


# ------------------------------------------------------ the week schedule

@pytest.mark.parametrize("any_day, expected", [
    ("2012-05-14", "2012-05-18"),      # an ordinary week: Friday
    ("2012-05-19", "2012-05-18"),      # a Saturday belongs to the week before
    ("2012-05-20", "2012-05-18"),      # ... and so does a Sunday
    ("2015-04-01", "2015-04-02"),      # Good Friday 2015-04-03 -> Thursday
    ("2013-03-25", "2013-03-28"),      # Good Friday 2013-03-29 -> Thursday
    ("2014-06-30", "2014-07-03"),      # July 4 on a Friday -> Thursday
    ("2015-06-29", "2015-07-02"),      # July 4 on a Saturday: Friday July 3 shut
    ("2020-06-29", "2020-07-02"),      # same again in 2020
    ("2015-12-21", "2015-12-24"),      # Christmas on a Friday -> Thursday
    ("2010-12-20", "2010-12-23"),      # Christmas on a Saturday: Friday 24th shut
    ("2009-12-28", "2009-12-31"),      # New Year's Day on a Friday -> Thursday
    ("2021-12-27", "2021-12-31"),      # ... Jan 1 on a Saturday: the 31st STAYS open
    ("2010-12-27", "2010-12-31"),      # same in 2010
    ("2026-06-15", "2026-06-18"),      # Juneteenth on a Friday (observed since 2022)
    ("2027-06-14", "2027-06-17"),      # Juneteenth on a Saturday: Friday 18th shut
    ("2015-06-15", "2015-06-19"),      # Juneteenth was NOT a holiday before 2022
    ("2011-07-04", "2011-07-08"),      # a Monday holiday does not touch the Friday
])
def test_scheduled_week_end_examples(any_day, expected):
    assert mod.scheduled_week_end(pd.Timestamp(any_day)) == pd.Timestamp(expected)


def test_schedule_equals_the_xnys_last_session_of_every_week_2004_to_2028():
    """Independent oracle: the exchange_calendars XNYS calendar, compared as
    a whole set over every ISO week from November 2004 (the panel's start) to
    the end of 2028, including every observed-holiday Friday rule."""
    sessions = _xnys("2004-11-15", "2028-12-29")
    iso = sessions.isocalendar()
    last = {}
    for position, key in enumerate(zip(iso["year"], iso["week"])):
        last[key] = sessions[position]
    assert len(last) > 1200
    bad = []
    for (year, week), actual in last.items():
        monday = pd.Timestamp(datetime.date.fromisocalendar(int(year), int(week), 1))
        scheduled = mod.scheduled_week_end(monday)
        if scheduled != actual:
            bad.append((monday.date().isoformat(), actual.date().isoformat(),
                        scheduled.date().isoformat()))
    assert bad == []


def test_week_end_dates_are_the_scheduled_dates_present_in_the_calendar():
    calendar = _xnys("2013-01-02", "2013-06-28")
    ends = mod.week_end_dates(calendar)
    assert ends.is_monotonic_increasing and ends.is_unique
    assert set(ends) <= set(calendar)
    assert pd.Timestamp("2013-03-28") in ends           # Good Friday week -> Thursday
    assert pd.Timestamp("2013-03-29") not in ends
    assert len(ends) == 26      # 26 ISO weeks; the first is partial but its Friday is in


def test_a_scheduled_week_end_missing_from_the_calendar_is_skipped_not_moved_back():
    calendar = _xnys("2013-01-02", "2013-06-28")
    missing = pd.Timestamp("2013-05-17")                # an ordinary Friday
    holed = calendar[calendar != missing]
    ends = mod.week_end_dates(holed)
    assert missing not in ends
    assert pd.Timestamp("2013-05-16") not in ends       # never falls back to Thursday
    assert len(ends) == len(mod.week_end_dates(calendar)) - 1


# --------------------------------------------------------- panel building

def test_panel_log_returns_are_one_session_returns_on_dates_all_five_have():
    index = _xnys("2012-01-03", "2012-03-30")
    returns = _returns(len(index), 3)
    frames = _frames(index, returns)
    closes = {ticker: mod.clean_close(frames[ticker]) for ticker in PANEL}
    out = mod.panel_log_returns(closes)
    assert list(out.columns) == PANEL
    assert out.index.equals(index[1:])
    np.testing.assert_allclose(out.to_numpy(), returns[1:], atol=1e-12)


def test_a_missing_print_drops_that_day_and_the_next_and_never_mixes_horizons():
    index = _xnys("2012-01-03", "2012-03-30")
    returns = _returns(len(index), 3)
    frames = _frames(index, returns)
    closes = {ticker: mod.clean_close(frames[ticker]) for ticker in PANEL}
    hole = index[20]
    closes["EEM"] = closes["EEM"].drop(hole)
    out = mod.panel_log_returns(closes)
    assert hole not in out.index
    assert index[21] not in out.index     # EEM's return there would span two sessions
    assert out.index.equals(index[1:].drop([index[20], index[21]]))
    positions = index.get_indexer(out.index)
    np.testing.assert_allclose(out.to_numpy(), returns[positions], atol=1e-12)


def test_a_date_absent_from_every_asset_is_a_closure_not_a_hole():
    index = _xnys("2012-01-03", "2012-03-30")
    returns = _returns(len(index), 3)
    frames = _frames(index, returns)
    closes = {ticker: mod.clean_close(frames[ticker]).drop(index[20])
              for ticker in PANEL}
    out = mod.panel_log_returns(closes)
    assert index[20] not in out.index
    assert index[21] in out.index         # a return over the closure is legitimate
    assert len(out) == len(index) - 2


def test_panel_starts_after_the_latest_inception():
    index = _xnys("2012-01-03", "2012-03-30")
    returns = _returns(len(index), 3)
    frames = _frames(index, returns)
    closes = {ticker: mod.clean_close(frames[ticker]) for ticker in PANEL}
    closes["GLD"] = closes["GLD"].iloc[30:]
    out = mod.panel_log_returns(closes)
    assert out.index[0] == index[31]      # GLD's first RETURN, not its first price


def test_clean_close_sorts_dedupes_keep_last_and_rejects_nonpositive():
    index = pd.DatetimeIndex(["2020-01-03", "2020-01-02", "2020-01-03"])
    frame = pd.DataFrame({"Close": [11.0, 10.0, 12.0]}, index=index)
    out = mod.clean_close(frame)
    assert list(out.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert list(out) == [10.0, 12.0]
    bad = pd.DataFrame({"Close": [10.0, 0.0]},
                       index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]))
    with pytest.raises(ValueError, match="positive"):
        mod.clean_close(bad)


# ------------------------------------------------- the turbulence statistic

def test_turbulence_matches_the_brute_force_definition_at_every_defined_session():
    df = _panel_df(640, 1)
    d = mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=MAX_COND)
    oracle = _oracle_turbulence(df.to_numpy())
    assert d.index.equals(df.index)
    np.testing.assert_allclose(d.to_numpy(), oracle, rtol=1e-8, equal_nan=True)
    assert np.isnan(d.to_numpy()[:MIN_COV_SESSIONS]).all()
    assert np.isfinite(d.to_numpy()[MIN_COV_SESSIONS:]).all()


def test_turbulence_needs_exactly_min_sessions_earlier_sessions():
    df = _panel_df(560, 4)
    d = mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=MAX_COND)
    assert np.isnan(d.iloc[MIN_COV_SESSIONS - 1])      # only 499 earlier sessions
    assert np.isfinite(d.iloc[MIN_COV_SESSIONS])       # exactly 500 earlier sessions
    shorter = mod.turbulence_series(df, min_sessions=50,
                                    max_condition_number=MAX_COND)
    assert np.isnan(shorter.iloc[49]) and np.isfinite(shorter.iloc[50])


def test_turbulence_at_t_uses_no_later_return_and_excludes_r_t_from_its_own_estimate():
    df = _panel_df(640, 6)
    base = mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                                 max_condition_number=MAX_COND)
    k = 560
    future = df.copy()
    future.iloc[k + 1:] *= 3.0                          # rewrite every later session
    d_future = mod.turbulence_series(future, min_sessions=MIN_COV_SESSIONS,
                                     max_condition_number=MAX_COND)
    assert d_future.iloc[:k + 1].equals(base.iloc[:k + 1])   # exactly identical

    spike = df.copy()
    spike.iloc[k] = spike.iloc[k] + 0.2                 # one huge move on day k
    d_spike = mod.turbulence_series(spike, min_sessions=MIN_COV_SESSIONS,
                                    max_condition_number=MAX_COND)
    assert d_spike.iloc[:k].equals(base.iloc[:k])
    history = df.to_numpy()[:k]                         # UNSPIKED history, k rows
    deviation = spike.to_numpy()[k] - history.mean(axis=0)
    expected = float(deviation @ np.linalg.inv(np.cov(history, rowvar=False)) @ deviation)
    assert d_spike.iloc[k] == pytest.approx(expected, rel=1e-8)
    assert d_spike.iloc[k] > 5.0 * base.iloc[k]         # the spike is visible on day k


def test_singular_covariance_fails_closed_naming_the_first_failing_date():
    df = _panel_df(600, 1)
    df["EEM"] = df["SPY"]                               # exact duplicate column
    with pytest.raises(ValueError, match=r"singular|ill-conditioned") as caught:
        mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=MAX_COND)
    assert df.index[MIN_COV_SESSIONS].date().isoformat() in str(caught.value)


def test_a_constant_asset_makes_the_covariance_singular_and_fails_closed():
    df = _panel_df(600, 1)
    df["GLD"] = 0.0
    with pytest.raises(ValueError, match=r"singular|ill-conditioned"):
        mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=MAX_COND)


def test_a_nearly_collinear_panel_fails_closed_instead_of_regularizing():
    df = _panel_df(600, 1)
    noise = np.random.RandomState(9).normal(0.0, 1.0, len(df))
    df["EEM"] = df["SPY"] + 1e-9 * noise                # condition number ~1e14
    with pytest.raises(ValueError, match="ill-conditioned"):
        mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=MAX_COND)


def test_the_condition_limit_argument_is_really_applied():
    df = _panel_df(600, 2)
    mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                          max_condition_number=MAX_COND)
    with pytest.raises(ValueError, match="ill-conditioned"):
        mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=1.5)


def test_no_ridge_or_shrinkage_a_moderately_conditioned_panel_matches_the_exact_inverse():
    df = _panel_df(600, 1)
    noise = np.random.RandomState(9).normal(0.0, 1.0, len(df))
    df["EEM"] = 0.9 * df["SPY"] + 1e-5 * noise          # condition number ~1e5-1e7
    history = df.to_numpy()[:MIN_COV_SESSIONS]
    condition = np.linalg.cond(np.cov(history, rowvar=False))
    assert 1e4 < condition < MAX_COND                    # inside the limit, and stiff
    d = mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=MAX_COND)
    oracle = _oracle_turbulence(df.to_numpy())
    np.testing.assert_allclose(d.to_numpy(), oracle, rtol=1e-5, equal_nan=True)


def test_nonfinite_returns_are_refused():
    df = _panel_df(600, 1)
    df.iloc[10, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mod.turbulence_series(df, min_sessions=MIN_COV_SESSIONS,
                              max_condition_number=MAX_COND)


# ----------------------------------------- smoothing and percentile rank

def test_smoothed_mean_is_the_direct_trailing_mean_undefined_until_window_values():
    values = pd.Series([np.nan] * 3 + [float(v) for v in range(1, 21)],
                      index=pd.bdate_range("2020-01-01", periods=23))
    out = mod.smoothed_mean(values, window=10)
    assert out.index.equals(values.index)
    assert out.iloc[:12].isna().all()
    assert out.iloc[12] == pytest.approx(np.mean(range(1, 11)))
    assert out.iloc[22] == pytest.approx(np.mean(range(11, 21)))
    other = mod.smoothed_mean(values, window=5)          # the window is an argument
    assert other.iloc[7] == pytest.approx(np.mean(range(1, 6)))
    assert other.iloc[7:].notna().all() and other.iloc[:7].isna().all()


def test_percentile_rank_counts_strictly_earlier_values_strictly_below():
    smoothed = pd.Series(np.concatenate([np.arange(250, dtype=float),
                                         [199.5, 200.5, 199.5]]))
    ranks = mod.percentile_ranks(smoothed, np.array([250, 251, 252]),
                                 min_history=MIN_PCT_HISTORY)
    assert ranks[0] == pytest.approx(200 / 250)          # values 0..199 are below
    assert ranks[1] == pytest.approx(202 / 251)          # 0..200 plus the 199.5
    assert ranks[2] == pytest.approx(200 / 252)          # ties are NOT below


def test_percentile_rank_needs_min_history_finite_earlier_values():
    smoothed = pd.Series([np.nan] * 20 + [float(v) for v in range(300)])
    ranks = mod.percentile_ranks(smoothed, np.array([268, 269, 270, 271]),
                                 min_history=MIN_PCT_HISTORY)
    assert np.isnan(ranks[0]) and np.isnan(ranks[1])     # 248 and 249 earlier values
    assert np.isfinite(ranks[2]) and np.isfinite(ranks[3])   # 250 and 251
    assert ranks[2] == pytest.approx(1.0) and ranks[3] == pytest.approx(1.0)
    other = mod.percentile_ranks(smoothed, np.array([50]), min_history=20)
    assert np.isfinite(other[0])                         # min_history is an argument


def test_percentile_rank_is_undefined_where_the_value_itself_is_undefined():
    smoothed = pd.Series(list(range(300)) + [np.nan], dtype=float)
    ranks = mod.percentile_ranks(smoothed, np.array([300]),
                                 min_history=MIN_PCT_HISTORY)
    assert np.isnan(ranks[0])


def test_defensive_only_when_the_percentile_is_strictly_above_the_threshold():
    assert mod.is_defensive(200 / 250, threshold=THRESHOLD) is False   # exactly 0.8
    assert mod.is_defensive(201 / 250, threshold=THRESHOLD) is True
    assert mod.is_defensive(0.8000001, threshold=THRESHOLD) is True
    assert mod.is_defensive(0.7999999, threshold=THRESHOLD) is False
    assert mod.is_defensive(float("nan"), threshold=THRESHOLD) is False
    assert mod.is_defensive(0.6, threshold=0.5) is True                # an argument


# ------------------------------------------- readings against the oracle

def test_readings_equal_the_independent_oracle_on_a_multi_regime_path():
    _, _, frames = _scenario()
    readings = mod.risk_on_readings(frames, require_complete_weeks=True)
    oracle = _oracle_readings(frames)
    assert list(readings.index) == list(oracle)
    assert [bool(v) for v in readings] == list(oracle.values())
    # Power guard: both states occur and there are real flips.
    states = list(oracle.values())
    flips = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert True in states and False in states and flips >= 6


def test_first_reading_is_the_first_week_end_with_250_earlier_smoothed_values():
    index, _, frames = _scenario()
    readings = mod.risk_on_readings(frames, require_complete_weeks=True)
    panel = _oracle_panel(frames)
    # panel position p has its smoothed value from p = 500 + 10 - 1; a reading
    # needs 250 finite earlier smoothed values, i.e. p >= 509 + 250.
    first_position = MIN_COV_SESSIONS + WINDOW - 1 + MIN_PCT_HISTORY
    week_ends = mod.week_end_dates(panel.index)
    expected = [d for d in week_ends if panel.index.get_loc(d) >= first_position][0]
    assert readings.index[0] == expected
    assert readings.index[0] > pd.Timestamp("2012-06-01")


def test_readings_only_ever_sit_on_scheduled_week_ends():
    _, _, frames = _scenario()
    readings = mod.risk_on_readings(frames, require_complete_weeks=True)
    scheduled = set(mod.week_end_dates(frames[RISK].index))
    assert set(readings.index) <= scheduled
    assert set(readings.unique()) <= {0.0, 1.0}


# ------------------------------------------------ engine-level behaviour

def test_engine_positions_match_the_independent_oracle_through_the_abi1_wiring():
    index, _, frames = _scenario()
    trades, active, _ = _run_like_abi1(frames)
    expected = _oracle_active(frames)
    assert list(active) == expected
    held = [a for a in expected if a is not None]
    assert set(held) == {RISK, DEFENSIVE}
    switches = sum(1 for a, b in zip(held, held[1:]) if a != b)
    assert switches >= 6
    assert len(trades) == switches + 1
    assert set(trades["asset"]) == {RISK, DEFENSIVE}


def test_a_switch_only_ever_fills_on_the_first_session_after_a_scheduled_week_end():
    index, _, frames = _scenario()
    trades, _, _ = _run_like_abi1(frames)
    ends = set(mod.week_end_dates(index))
    assert len(trades) >= 6
    for entry_date in trades["entry_date"].iloc[1:]:
        previous_session = index[index.get_loc(entry_date) - 1]
        assert previous_session in ends, entry_date


def test_a_turbulence_burst_moves_the_account_into_shy_the_following_week():
    """Behavioural sanity, not an oracle: the second burst (rows 1100-1111,
    six times the normal moves) must put the account in SHY within a week or
    two of its start, and back in SPY afterwards."""
    index, _, frames = _scenario()
    _, active, _ = _run_like_abi1(frames)
    window = active.loc[index[1100]:index[1125]]
    assert DEFENSIVE in set(window)
    assert active.iloc[-1] in (RISK, DEFENSIVE)
    assert RISK in set(active.loc[index[1180]:])


def test_warmup_holds_spy_from_the_first_execution_session_by_default():
    index, _, frames = _scenario()
    readings = mod.risk_on_readings(frames, require_complete_weeks=True)
    trades, active, _ = _run_like_abi1(frames)
    assert active.iloc[0] is None                        # the engine's own first bar
    first_reading_pos = index.get_loc(readings.index[0])
    assert (active.iloc[1:first_reading_pos + 1] == RISK).all()
    assert trades.iloc[0]["asset"] == RISK
    assert trades.iloc[0]["entry_date"] == index[1]


def test_flat_warmup_variant_stays_in_cash_until_the_first_reading():
    index, _, frames = _scenario()
    readings = mod.risk_on_readings(frames, require_complete_weeks=True)
    _, active, _ = _run_like_abi1(frames, warmup_holding="flat")
    first_reading_pos = index.get_loc(readings.index[0])
    assert active.iloc[:first_reading_pos + 1].isna().all()
    assert list(active) == _oracle_active(frames, warmup=None)


def test_an_unknown_warmup_holding_is_refused():
    _, _, frames = _scenario()
    with pytest.raises(ValueError, match="warmup_holding"):
        mod.make_target_fn(frames, warmup_holding="leveraged")


# ------------------------------ look-ahead and hold-out blindness harness

def _build_readings(frames: dict) -> pd.Series:
    return mod.risk_on_readings(frames, require_complete_weeks=False)


def _transition_cutoffs(frames: dict, *, offsets=range(0, 6), max_flips=6,
                        ordinary_every=25) -> list:
    """Cutoffs 0..5 sessions AFTER each state change (the only places a
    peeked decision shows), plus ordinary week ends spread through the
    readings era. Round dates alone would miss a short-reach look-ahead."""
    index = frames[RISK].index
    readings = _build_readings(frames)
    dates = list(readings.index)
    flips = [d for prev, d in zip(dates, dates[1:])
             if readings.loc[prev] != readings.loc[d]]
    assert len(flips) >= 4                                # power guard
    cuts = set()
    for date in flips[:max_flips]:
        position = index.get_loc(date)
        for offset in offsets:
            if position + offset < len(index):
                cuts.add(index[position + offset])
    for date in dates[::ordinary_every]:
        cuts.add(date)
    return sorted(cuts)


def _prefix_violations(frames: dict, cutoffs, *, stop_at_first=False) -> list:
    """Cutoffs at which building from frames cut off there gives a different
    reading history than the full build restricted to that date."""
    full = _build_readings(frames)
    bad = []
    for cut in cutoffs:
        part = _build_readings(_truncate(frames, cut))
        expected = full.loc[full.index <= cut]
        if not (part.index.equals(expected.index) and part.equals(expected)):
            bad.append(cut)
            if stop_at_first:
                break
    return bad


def test_readings_are_prefix_invariant_at_cutoffs_zero_to_five_sessions_after_every_flip():
    _, _, frames = _scenario()
    cutoffs = _transition_cutoffs(frames)
    assert len(cutoffs) >= 30
    assert _prefix_violations(frames, cutoffs) == []


def test_harness_catches_a_one_session_lookahead_in_the_smoothed_series(monkeypatch):
    _, _, frames = _scenario()
    cutoffs = _transition_cutoffs(frames)
    real = mod.smoothed_mean

    def peeking(values, *, window):
        return real(values, window=window).shift(-1)      # tomorrow's value today

    monkeypatch.setattr(mod, "smoothed_mean", peeking)
    assert _prefix_violations(frames, cutoffs, stop_at_first=True) != []


def test_harness_catches_a_one_session_lookahead_in_the_turbulence_series(monkeypatch):
    _, _, frames = _scenario()
    cutoffs = _transition_cutoffs(frames)
    real = mod.turbulence_series

    def peeking(returns, *, min_sessions, max_condition_number):
        return real(returns, min_sessions=min_sessions,
                    max_condition_number=max_condition_number).shift(-1)

    monkeypatch.setattr(mod, "turbulence_series", peeking)
    assert _prefix_violations(frames, cutoffs, stop_at_first=True) != []


@pytest.mark.parametrize("last_row_counts", [False, True])
def test_harness_catches_a_week_end_derived_from_the_data(monkeypatch, last_row_counts):
    """The classic calendar look-ahead: 'the next row is in a new week' (or
    'the last row is a week end'). Both variants must be caught."""
    _, _, frames = _scenario()
    cutoffs = _transition_cutoffs(frames)

    def data_derived(calendar):
        iso = pd.DatetimeIndex(calendar).isocalendar()
        keys = list(zip(iso["year"], iso["week"]))
        keep = [keys[i] != keys[i + 1] for i in range(len(keys) - 1)]
        keep.append(bool(last_row_counts))
        return pd.DatetimeIndex(calendar)[np.array(keep, dtype=bool)]

    monkeypatch.setattr(mod, "week_end_dates", data_derived)
    assert _prefix_violations(frames, cutoffs, stop_at_first=True) != []


def test_harness_catches_a_percentile_ranked_against_later_values_too(monkeypatch):
    _, _, frames = _scenario()
    cutoffs = _transition_cutoffs(frames)

    def two_sided(smoothed, positions, *, min_history):
        values = np.asarray(smoothed, dtype=float)
        out = np.full(len(positions), np.nan)
        for slot, position in enumerate(positions):
            others = np.delete(values, position)
            others = others[np.isfinite(others)]
            if np.isfinite(values[position]) and len(others) >= min_history:
                out[slot] = float((others < values[position]).sum()) / len(others)
        return out

    monkeypatch.setattr(mod, "percentile_ranks", two_sided)
    assert _prefix_violations(frames, cutoffs, stop_at_first=True) != []


def test_cutting_every_frame_off_at_a_regime_changing_week_end_keeps_the_decision():
    """Truncating all frames exactly at a flip date must still produce that
    date's reading, and the engine run over the cut frames must hold the same
    asset the full run held through the cut."""
    index, _, frames = _scenario()
    readings = _build_readings(frames)
    dates = list(readings.index)
    flip = [d for prev, d in zip(dates, dates[1:])
            if readings.loc[prev] != readings.loc[d]][1]
    cut = _truncate(frames, flip)
    assert _build_readings(cut).loc[flip] == readings.loc[flip]
    _, full_active, _ = _run_like_abi1(frames)
    _, cut_active, _ = _run_like_abi1(cut)
    assert list(cut_active) == list(full_active.loc[:flip])


# ----------------------------------------------- restricted-bucket carry-in

def _tail_run(frames: dict, start: pd.Timestamp):
    """The target factory sees the FULL history; the engine sees only a tail
    bucket that starts at ``start`` (a restricted evaluation window)."""
    target_fn = mod.make_target_fn(frames)
    tail = {key: frame.loc[start:].copy() for key, frame in frames.items()}
    assets, signals = abi1_driver._align_rotation_frames(
        {RISK: tail[RISK], DEFENSIVE: tail[DEFENSIVE]},
        {ticker: tail[ticker] for ticker in SIGNAL_ONLY})
    return rotation_engine.run_two_asset_rotation(
        assets, target_fn, cost_bps_by_asset={RISK: 1.0, DEFENSIVE: 1.0},
        signal_frames=signals)


def test_a_bucket_starting_mid_regime_gets_the_carried_in_state_through_the_engine():
    index, _, frames = _scenario()
    _, full_active, _ = _run_like_abi1(frames)
    held = list(full_active)
    # A start 2 sessions after the first flip INTO the defensive asset (still
    # inside that held stretch) and one inside a later SPY stretch.
    first_shy = next(i for i, a in enumerate(held) if a == DEFENSIVE)
    starts = [index[first_shy + 1]]
    later_spy = next(i for i in range(first_shy + 30, len(held)) if held[i] == RISK
                     and held[i - 1] == RISK and held[i - 2] == RISK)
    starts.append(index[later_spy])
    assert held[index.get_loc(starts[0]) + 1] == DEFENSIVE     # power guard
    for start in starts:
        _, tail_active = _tail_run(frames, start)
        assert list(tail_active.iloc[1:]) == list(full_active.loc[start:].iloc[1:])
    _, first_tail = _tail_run(frames, starts[0])
    assert first_tail.iloc[1] == DEFENSIVE                     # carried in at once


# ---------------------------------------------------------- degraded data

def test_a_missing_scheduled_week_end_raises_by_default_and_the_opt_out_carries_state():
    index, returns, frames = _scenario()
    readings = _build_readings(frames)
    victim = readings.index[10]                          # inside the readings era
    holed = {key: frame.loc[frame.index != victim].copy()
             for key, frame in frames.items()}
    with pytest.raises(ValueError, match=victim.date().isoformat()):
        mod.make_target_fn(holed)
    carried = _build_readings(holed)
    assert victim not in carried.index
    assert (victim - pd.Timedelta(days=1)) not in carried.index   # never moved back
    assert carried.index.equals(readings.index.drop(victim))
    trades, active, _ = _run_like_abi1(holed, require_complete_weeks=False)
    assert active.notna().sum() > 0


def test_too_little_history_for_any_reading_raises_instead_of_scoring_nothing():
    index = _xnys("2010-01-04", "2012-06-29")            # ~630 sessions
    frames = _frames(index, _returns(len(index), 1))
    with pytest.raises(ValueError, match="reading"):
        mod.make_target_fn(frames)


def test_a_missing_frame_or_bad_close_is_refused():
    _, _, frames = _scenario()
    for key in (RISK, DEFENSIVE, *SIGNAL_ONLY):
        broken = {k: v for k, v in frames.items() if k != key}
        with pytest.raises(ValueError, match=key):
            mod.make_target_fn(broken)
    bad = copy.deepcopy(frames)
    bad["GLD"].iloc[5, bad["GLD"].columns.get_loc("Close")] = -1.0
    with pytest.raises(ValueError, match="positive"):
        mod.make_target_fn(bad)


def test_building_the_target_does_not_mutate_the_frames():
    _, _, frames = _scenario()
    before = copy.deepcopy(frames)
    mod.make_target_fn(frames)
    for key in frames:
        pd.testing.assert_frame_equal(frames[key], before[key])


# ------- helpers shared with the real-data checks (those checks are not part
# ------- of this public file; nothing here reaches the network)

def _abi1_driver_scope(tmp_path, *, catalog_parameters=None, loader=None):
    """A scope for scripts/run_abi1_mining.py built from tmp files only: the
    family's real catalog entry (optionally with drifted parameters), a
    private copy of the module, a tmp ledger that is never written (every
    call below is a dry run or a skip), and clearly SYNTHETIC quotes."""
    import run_batch5_mining as batch5
    entry = next(c for c in _read_json(REAL_CATALOG) if c["family_id"] == FAMILY_ID)
    if catalog_parameters is not None:
        entry = dict(entry, parameters=catalog_parameters)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps([entry]), encoding="utf-8")
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps({
        "batch": "test", "curated_on": "2026-09-23", "accepted": [entry]}),
        encoding="utf-8")
    quote = {"price": 100.01, "bid": 100.0, "ask": 100.02,
             "security_status": "Normal", "spread_bps": 2.0, "suspect": False}
    quotes_path = tmp_path / "quotes.json"
    quotes_path.write_text(json.dumps({
        "sampled_at_local": "2026-09-23T10:00:00-04:00",
        "source": "SYNTHETIC wiring proof only -- not a cost measurement",
        "market_state": "SYNTHETIC (test)",
        "quotes": {RISK: quote, DEFENSIVE: quote}}), encoding="utf-8")
    modules_dir = tmp_path / "mods"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    ledger = str(tmp_path / "ledger.jsonl")

    def refuse(symbol):
        raise AssertionError(f"OHLC requested for {symbol!r} before the drift skip")

    scope = batch5.build_scope(
        accepted_path=str(accepted_path), quotes_path=str(quotes_path),
        catalog_path=str(catalog_path), ledger_path=ledger,
        modules_dir=str(modules_dir), run_dir=str(tmp_path / "run"),
        ohlc_loader=loader or refuse)
    return scope, ledger


# --------------------------------------- vendor-frame loading (no network)

# A fixed cut-off for real-data checks (not part of this public file). A
# vendor history can briefly miss a recent session for some panel tickers but
# not others, and the ABI-1 driver refuses two traded legs whose calendars
# differ, so real frames are cut here to keep such checks about the rule and
# not about the state of a vendor feed. Later data only ever appends.
REAL_THROUGH = pd.Timestamp("2026-09-18")


def _normalize_real_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One raw vendor frame, loaded the way scripts/run_abi1_mining.py loads it
    (run_batch5_mining.load_frame): rows whose four price fields are ALL blank
    are dropped, judged on the price columns only (the raw frame also carries a
    Date column, so a whole-row dropna never sees such a row as empty), then the
    frame goes through run_batch3_mining._normalize_ohlc with the strict
    positivity gate ON for the two traded symbols and OFF for the four
    signal-only ones."""
    frame, _dropped = batch5_run._drop_all_nan_rows(raw)
    return batch3._normalize_ohlc(
        frame, ticker, require_positive=ticker in (RISK, DEFENSIVE))


def _vendor_shaped_frame(*, blank_tail_rows: int, bad_print: bool = False):
    """A raw frame shaped like data_pipeline.fetch_ohlc output (Date column as
    datetime.date, then Open/High/Low/Close), optionally ending in placeholder
    rows blank in all four price columns, the way the live vendor feed ends on
    the current session before it has printed."""
    dates = list(pd.bdate_range("2026-09-14", periods=5 + blank_tail_rows).date)
    closes = [100.0, 101.0, 100.5, 102.0, 101.5] + [float("nan")] * blank_tail_rows
    frame = pd.DataFrame({
        "Date": dates,
        "Open": closes, "High": closes, "Low": closes, "Close": closes,
    })
    if bad_print:                          # SOME prices present: a bad print
        frame.loc[2, "Low"] = float("nan")
    return frame


def test_a_blank_placeholder_row_at_the_end_of_a_vendor_frame_is_dropped_not_fatal():
    """The live vendor frames for all six tickers end with a row that is blank
    in Open/High/Low/Close; the real driver drops it, so a real-data load
    must produce the same frame instead of erroring at setup."""
    raw = _vendor_shaped_frame(blank_tail_rows=1)
    for ticker in (RISK, DEFENSIVE, *SIGNAL_ONLY):
        loaded = _normalize_real_frame(raw, ticker)
        assert len(loaded) == 5
        assert loaded.index[-1] == pd.Timestamp("2026-09-18")
        assert list(loaded.columns) == ["Open", "High", "Low", "Close"]
        assert np.isfinite(loaded.to_numpy(dtype=float)).all()


def test_the_real_data_loader_still_fails_closed_on_a_partially_blank_row():
    """Only ALL-blank rows are a session the vendor never printed. A row with
    some prices present is a bad print and must keep failing for a traded
    symbol (the gate is not weakened by dropping the placeholder rows)."""
    raw = _vendor_shaped_frame(blank_tail_rows=0, bad_print=True)
    with pytest.raises(ValueError, match="finite and positive"):
        _normalize_real_frame(raw, RISK)


