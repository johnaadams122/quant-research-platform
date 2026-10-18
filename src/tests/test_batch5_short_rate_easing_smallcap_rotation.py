"""Tests for src/mining/batch5/short_rate_easing_smallcap_rotation.py.

PRE-REGISTERED RULE (both constants are fixed a priori in the catalog entry;
nothing below may be tuned after looking at a backtest):

    at each month-end m (last trading session of the month):
        rtb    = irx_close[m] - mean(irx_close over the trailing 252
                 sessions ending at m, m included)          # percentage points
        easing = rtb <= -0.10
        target = IWM if easing else SPY   # engine executes at the NEXT
                                          # session's open (verified from
                                          # rotation_engine.py source)

The tests are grouped: contract, pure building blocks, engine-level
behaviour against an INDEPENDENT oracle, look-ahead / hold-out blindness,
restricted-bucket consistency through the REAL engine and the REAL bucket
driver, degraded-data handling, and the REAL ABI-1 driver on synthetic
frames (dry run, no network).
"""
import json
import os
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
from mining import abi1_driver, rotation_engine  # noqa: E402
from mining.batch5 import short_rate_easing_smallcap_rotation as mod  # noqa: E402

FAMILY_ID = "short_rate_easing_smallcap_rotation"
EASING_TARGET, OTHERWISE_TARGET = "IWM", "SPY"
IRX = "^IRX"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")

WINDOW = 252            # pre-registered, restated here on purpose
THRESHOLD = -0.10       # pre-registered, restated here on purpose


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


# The two scheduled NYSE closures that can land on the last weekday of a
# month (Good Friday in March, Memorial Day on May 31), listed by hand so the
# synthetic calendars below do not share code with the module under test.
_GOOD_FRIDAYS = ["2010-04-02", "2011-04-22", "2012-04-06", "2013-03-29",
                 "2014-04-18", "2015-04-03", "2016-03-25", "2017-04-14",
                 "2018-03-30", "2019-04-19", "2020-04-10", "2021-04-02",
                 "2022-04-15", "2023-04-07"]
_MEMORIAL_DAYS = ["2010-05-31", "2011-05-30", "2012-05-28", "2013-05-27",
                  "2014-05-26", "2015-05-25", "2016-05-30", "2017-05-29",
                  "2018-05-28", "2019-05-27", "2020-05-25", "2021-05-31",
                  "2022-05-30", "2023-05-29"]
_SCHEDULED_CLOSURES = pd.DatetimeIndex(pd.to_datetime(_GOOD_FRIDAYS + _MEMORIAL_DAYS))


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    """Weekdays minus the scheduled closures above, so a synthetic calendar
    has the same last-session-of-the-month dates a real NYSE calendar has."""
    days = pd.bdate_range(start, end)
    return days[~days.isin(_SCHEDULED_CLOSURES)]


def _random_frame(index: pd.DatetimeIndex, seed: int,
                  level: float = 100.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0002, 0.01, len(index))
    return _ohlc_from_close(level * np.exp(np.cumsum(steps)), index)


def _irx_frame(values, index: pd.DatetimeIndex) -> pd.DataFrame:
    """A signal frame. Yields may be zero/negative, so no positivity is
    imposed; High/Low bracket Open/Close like the real frames."""
    return _ohlc_from_close(values, index)


def _frames(index: pd.DatetimeIndex, irx_values, *, seed_iwm: int = 1,
            seed_spy: int = 2) -> dict:
    return {
        EASING_TARGET: _random_frame(index, seed=seed_iwm, level=100.0),
        OTHERWISE_TARGET: _random_frame(index, seed=seed_spy, level=80.0),
        IRX: _irx_frame(irx_values, index),
    }


def _step_down_irx(index: pd.DatetimeIndex, step_date: str,
                   high: float = 2.0, low: float = 1.5) -> np.ndarray:
    values = np.full(len(index), high, dtype=float)
    values[index >= pd.Timestamp(step_date)] = low
    return values


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _catalog_entry() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == FAMILY_ID]
    assert len(matches) == 1, (
        f"expected exactly one catalog.json entry for {FAMILY_ID}, found "
        f"{len(matches)}")
    return matches[0]


def _run_engine(frames: dict, *, target_fn=None):
    """The REAL rotation engine, exactly as the ABI-1 driver wires it: the
    target factory sees the full frames, the engine sees the two traded legs
    plus the signal-only IRX frame."""
    if target_fn is None:
        target_fn = mod.make_target_fn(frames)
    assets = {EASING_TARGET: frames[EASING_TARGET],
              OTHERWISE_TARGET: frames[OTHERWISE_TARGET]}
    return rotation_engine.run_two_asset_rotation(
        assets, target_fn,
        cost_bps_by_asset={EASING_TARGET: 1.0, OTHERWISE_TARGET: 1.0},
        signal_frames={IRX: frames[IRX]})


# -------------------------------------------------------------- contract

def test_module_constants_match_the_preregistered_spec():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [EASING_TARGET, OTHERWISE_TARGET]
    assert mod.SIGNAL == [IRX]
    assert mod.BENCHMARK == "SPY"
    assert mod.SNAPSHOTS == {}
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert not isinstance(mod.EXPECTED_MIN_TRADES, bool)
    assert mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_target_fn)


def test_preregistered_constants_are_exactly_252_sessions_and_minus_0_10():
    assert mod.PARAMETERS["detrend_window_sessions"] == WINDOW
    assert isinstance(mod.PARAMETERS["detrend_window_sessions"], int)
    assert mod.PARAMETERS["easing_threshold_pp"] == THRESHOLD
    assert mod.PARAMETERS["short_rate_ticker"] == IRX
    assert mod.PARAMETERS["rebalance"] == "monthly"
    assert set(mod.PARAMETERS) == {
        "short_rate_ticker", "detrend_window_sessions",
        "easing_threshold_pp", "rebalance"}
    # The values the code actually USES must be the pre-registered ones, not
    # just the dict that documents them.
    assert mod._WINDOW == WINDOW
    assert mod._EASING_THRESHOLD_PP == THRESHOLD


def test_parameters_are_byte_identical_to_the_real_catalog_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert entry["tickers"] == [EASING_TARGET, OTHERWISE_TARGET]
    assert entry["status"] == "eligible"
    assert entry["added_date"] == "2026-09-23"
    assert entry["account_types"] == ["taxable", "roth"]


def test_parameter_drift_failure_is_reported_for_every_kind_of_drift():
    import run_batch5_mining as batch5
    real = _catalog_entry()["parameters"]
    assert batch5.parameter_drift(mod.PARAMETERS, real) is None
    drifted = [
        {**real, "detrend_window_sessions": 251},
        {**real, "easing_threshold_pp": -0.11},
        {**real, "easing_threshold_pp": -0.09},
        {**real, "short_rate_ticker": "^FVX"},
        {**real, "rebalance": "weekly"},
        {**real, "extra_key": 1},
        {k: v for k, v in real.items() if k != "easing_threshold_pp"},
    ]
    for catalog_side in drifted:
        message = batch5.parameter_drift(mod.PARAMETERS, catalog_side)
        assert message is not None and "!=" in message, catalog_side
    # ... and the same in the other direction (module drifts, catalog fixed).
    module_side = dict(mod.PARAMETERS, detrend_window_sessions=253)
    assert batch5.parameter_drift(module_side, real) is not None


def test_driver_discovery_validates_this_module_in_isolation(tmp_path):
    """Discovery on a private copy of the module (other modules in the
    real batch5 dir may change independently, so the real dir is not
    asserted on)."""
    import run_batch5_mining as batch5
    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["family_id"] == FAMILY_ID
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == [EASING_TARGET, OTHERWISE_TARGET]
    assert spec["signal"] == [IRX]
    assert spec["benchmark"] == "SPY"
    assert spec["expected_min_trades"] == mod.EXPECTED_MIN_TRADES
    assert batch5.parameter_drift(
        spec["parameters"], _catalog_entry()["parameters"]) is None


# ------------------------------------------------- pure building blocks

def test_clean_close_is_tz_naive_sorted_deduplicated_finite_and_keeps_negatives():
    index = pd.DatetimeIndex(pd.to_datetime([
        "2020-03-26", "2020-03-24", "2020-03-25", "2020-03-25",
        "2020-03-27", "2020-03-30", "2020-03-31"])).tz_localize("UTC")
    frame = pd.DataFrame({"Close": [-0.105, -0.033, 1.0, -0.070,
                                    np.nan, np.inf, 0.0]}, index=index)
    cleaned = mod._clean_close(frame)
    assert cleaned.index.tz is None
    assert cleaned.index.is_monotonic_increasing and cleaned.index.is_unique
    # 03-25 appears twice: the LAST occurrence (-0.070) wins; NaN and inf are
    # dropped; negative and zero prints survive unmodified (no floor).
    assert list(cleaned.index) == list(pd.to_datetime(
        ["2020-03-24", "2020-03-25", "2020-03-26", "2020-03-31"]))
    assert list(cleaned.to_numpy()) == [-0.033, -0.070, -0.105, 0.0]


@pytest.mark.parametrize("year, month, expected", [
    (2013, 3, "2013-03-28"),   # Good Friday 03-29
    (2018, 3, "2018-03-29"),   # Good Friday 03-30
    (2024, 3, "2024-03-28"),   # Good Friday 03-29
    (2002, 3, "2002-03-28"),   # Good Friday 03-29
    (2010, 5, "2010-05-28"),   # Memorial Day is Monday 05-31
    (2021, 5, "2021-05-28"),   # Memorial Day is Monday 05-31
    (2004, 5, "2004-05-28"),   # Memorial Day is Monday 05-31
    (2016, 5, "2016-05-31"),   # Memorial Day 05-30; Tuesday 05-31 is open
    (2013, 4, "2013-04-30"),   # an ordinary Tuesday
    (2011, 12, "2011-12-30"),  # 12-31 is a Saturday
    (2010, 12, "2010-12-31"),  # a Friday; open although 2011-01-01 is a Saturday
    (2020, 8, "2020-08-31"),   # a Monday
    (2015, 2, "2015-02-27"),   # 02-28 is a Saturday
    (2020, 2, "2020-02-28"),   # leap year, 02-29 is a Saturday
    (2024, 2, "2024-02-29"),   # leap day, a Thursday
    (2000, 2, "2000-02-29"),   # leap day, a Tuesday
])
def test_scheduled_month_end_is_the_ex_ante_last_session_of_the_month(
        year, month, expected):
    """Month-ends come from the exchange's published schedule, not from
    whichever dates a data file happens to contain. Every expectation is
    hand-known; none is computed by the module."""
    assert mod.scheduled_month_end(year, month) == pd.Timestamp(expected)


def test_month_end_dates_are_the_scheduled_month_ends_present_in_the_calendar():
    index = pd.DatetimeIndex(pd.to_datetime([
        "2021-01-27", "2021-01-28", "2021-01-29",     # 01-29 is January's month-end
        "2021-02-01", "2021-02-26",                   # 02-26 is February's
        "2021-03-01", "2021-03-02"]))                 # March's (03-31) is not here
    assert list(mod.month_end_dates(index)) == list(
        pd.to_datetime(["2021-01-29", "2021-02-26"]))


def test_the_last_date_of_a_calendar_is_a_month_end_when_it_is_the_scheduled_one():
    """Nothing after the date is needed to know it is the month-end."""
    index = pd.DatetimeIndex(pd.to_datetime(
        ["2021-02-24", "2021-02-25", "2021-02-26"]))
    assert list(mod.month_end_dates(index)) == list(pd.to_datetime(["2021-02-26"]))


def test_a_scheduled_month_end_missing_from_the_calendar_is_skipped_not_moved_back():
    index = pd.DatetimeIndex(pd.to_datetime([
        "2021-01-27", "2021-01-28",                   # Fri 01-29 is MISSING
        "2021-02-01", "2021-02-26", "2021-03-01"]))
    ends = mod.month_end_dates(index)
    assert list(ends) == list(pd.to_datetime(["2021-02-26"]))
    assert pd.Timestamp("2021-01-28") not in set(ends)


def test_month_end_dates_cross_year_and_same_month_number_gap():
    index = pd.DatetimeIndex(pd.to_datetime([
        "2020-12-30", "2020-12-31", "2021-01-04",
        "2021-03-31", "2022-03-01", "2022-03-31"]))   # same month NUMBER, next year
    assert list(mod.month_end_dates(index)) == list(pd.to_datetime(
        ["2020-12-31", "2021-03-31", "2022-03-31"]))


def test_month_end_dates_of_an_empty_calendar_is_empty():
    assert len(mod.month_end_dates(pd.DatetimeIndex([]))) == 0


def test_traded_calendar_is_the_intersection_of_both_legs_sessions():
    index = _bdays("2015-01-02", "2015-03-31")
    frames = _frames(index, np.full(len(index), 2.0))
    iwm_only, spy_only = index[10], index[20]
    frames[EASING_TARGET] = frames[EASING_TARGET].drop(spy_only)
    frames[OTHERWISE_TARGET] = frames[OTHERWISE_TARGET].drop(iwm_only)
    calendar = mod.traded_calendar(frames)
    assert list(calendar) == [d for d in index if d not in (iwm_only, spy_only)]


def test_a_month_end_missing_from_only_one_leg_is_not_a_month_end():
    index = _bdays("2015-01-02", "2015-03-31")
    frames = _frames(index, np.full(len(index), 2.0))
    february_end = pd.Timestamp("2015-02-27")
    assert february_end in set(mod.month_end_dates(mod.traded_calendar(frames)))
    frames[OTHERWISE_TARGET] = frames[OTHERWISE_TARGET].drop(february_end)
    assert february_end not in set(mod.month_end_dates(mod.traded_calendar(frames)))


def test_relative_bill_rate_is_a_hand_worked_difference_and_includes_the_month_end():
    index = _bdays("2018-01-02", "2019-06-28")
    values = np.full(len(index), 4.0)
    values[-1] = 3.8
    irx = pd.Series(values, index=index)
    rtb = mod.relative_bill_rate_at(irx, index[-1:])
    # window = 251 prints of 4.0 plus the month-end print itself (3.8):
    # mean = 4.0 - 0.2/252, so rtb = 3.8 - mean = -0.2 * 251 / 252.
    assert rtb.iloc[-1] == pytest.approx(-0.2 * 251 / 252, abs=1e-12)


def test_relative_bill_rate_window_is_exactly_252_sessions():
    index = _bdays("2016-01-04", "2019-12-31")
    rng = np.random.RandomState(11)
    values = rng.normal(2.0, 0.3, len(index))
    position = 700
    when = index[position:position + 1]

    def rtb_of(series_values):
        return float(mod.relative_bill_rate_at(
            pd.Series(series_values, index=index), when).iloc[0])

    base = rtb_of(values)
    expected = values[position] - values[position - 251:position + 1].mean()
    assert base == pytest.approx(expected, abs=1e-12)

    outside = values.copy()
    outside[position - 252] += 100.0          # one session OUTSIDE the window
    assert rtb_of(outside) == pytest.approx(base, abs=1e-12)

    inside = values.copy()
    inside[position - 251] += 100.0           # the oldest session INSIDE it
    assert rtb_of(inside) == pytest.approx(base - 100.0 / 252, abs=1e-9)

    after = values.copy()
    after[position + 1:] += 100.0             # everything AFTER the date
    assert rtb_of(after) == pytest.approx(base, abs=1e-12)


def test_relative_bill_rate_is_undefined_until_a_full_252_session_window():
    index = _bdays("2016-01-04", "2018-12-31")
    irx = pd.Series(np.linspace(1.0, 2.0, len(index)), index=index)
    rtb = mod.relative_bill_rate_at(irx, index)
    assert rtb.iloc[:251].isna().all()        # 251 prints: still warming up
    assert rtb.iloc[251:].notna().all()       # 252nd print: first defined


def test_relative_bill_rate_uses_differences_and_never_floors_negative_prints():
    index = _bdays("2019-01-02", "2020-06-30")
    values = np.full(len(index), -0.05)       # a genuinely sub-zero regime
    values[-1] = -0.105                       # the real March-2020 minimum
    irx = pd.Series(values, index=index)
    rtb = float(mod.relative_bill_rate_at(irx, index[-1:]).iloc[0])
    expected = -0.105 - (251 * -0.05 + -0.105) / 252
    assert np.isfinite(rtb)
    assert rtb == pytest.approx(expected, abs=1e-12)
    # Power check: flooring every print at zero would give exactly 0.0, so a
    # floored implementation is nowhere near the pre-registered number.
    assert abs(rtb) > 0.05
    # Zero prints are legitimate values too (a ratio or log would explode).
    zero = pd.Series(np.zeros(len(index)), index=index)
    assert float(mod.relative_bill_rate_at(zero, index[-1:]).iloc[0]) == 0.0


def test_relative_bill_rate_is_nan_for_a_date_without_a_print():
    index = _bdays("2016-01-04", "2018-12-31")
    irx = pd.Series(np.linspace(1.0, 2.0, len(index)), index=index)
    hole = index[600]
    holey = irx.drop(hole)
    rtb = mod.relative_bill_rate_at(holey, pd.DatetimeIndex([hole, index[601]]))
    assert np.isnan(rtb.iloc[0])              # no print at the month-end
    assert np.isfinite(rtb.iloc[1])           # the next date still has one


def test_easing_band_edge_is_inclusive_at_exactly_minus_0_10():
    assert bool(mod.is_easing(-0.10)) is True
    assert bool(mod.is_easing(np.nextafter(-0.10, -1.0))) is True
    assert bool(mod.is_easing(np.nextafter(-0.10, 0.0))) is False
    assert bool(mod.is_easing(0.0)) is False
    assert bool(mod.is_easing(-5.0)) is True
    assert bool(mod.is_easing(float("nan"))) is False
    flags = mod.is_easing(np.array([-0.2, -0.10, -0.0999999, 0.3]))
    assert list(flags) == [True, True, False, False]


# --------------------------- engine-level behaviour vs an INDEPENDENT oracle

def _regime_irx(index: pd.DatetimeIndex) -> np.ndarray:
    """A deterministic bill-rate path with several easing and non-easing
    stretches (piecewise drift plus noise). Positive throughout."""
    rng = np.random.RandomState(7)
    n = len(index)
    drift = np.zeros(n)
    blocks = [(300, 460, -0.006), (460, 800, 0.002), (800, 900, -0.008),
              (900, 1200, 0.003), (1200, 1290, -0.007), (1290, n, 0.002)]
    for lo, hi, step in blocks:
        drift[lo:hi] = step
    return 3.0 + np.cumsum(drift + rng.normal(0.0, 0.004, n))


def _oracle_month_end_flags(frames: dict) -> list:
    """The pre-registered rule re-implemented from the rule text in this
    file's docstring, sharing no
    code with the module: plain loops, pandas rolling mean (the module uses a
    direct numpy mean per window), month-ends from a next-date scan.

    Returns [(month_end_date, easing_flag), ...] for every traded month-end
    that has a defined reading."""
    irx = frames[IRX]["Close"]
    mean252 = irx.rolling(WINDOW, min_periods=WINDOW).mean()
    rtb = irx - mean252
    index = frames[EASING_TARGET].index
    out = []
    for k in range(len(index) - 1):
        if index[k + 1].to_period("M") != index[k].to_period("M"):
            reading = rtb.loc[index[k]]
            if not np.isnan(reading):
                out.append((index[k], bool(reading <= THRESHOLD)))
    return out


def _oracle_active_assets(frames: dict) -> list:
    """Returns the asset held DURING each session (None before the first
    reading): the reading dated at month-end m becomes the target on bar m and
    fills at the open of the next session, so it is held from session m+1."""
    flags = dict(_oracle_month_end_flags(frames))
    index = frames[EASING_TARGET].index
    target_after_bar = []          # target known at the close of each bar
    current = None
    for date in index:
        if date in flags:
            current = EASING_TARGET if flags[date] else OTHERWISE_TARGET
        target_after_bar.append(current)
    return [None] + target_after_bar[:-1]


def _transition_month_ends(frames: dict, *, extra: int = 0) -> list:
    """Month-ends at which the regime CHANGES (the only places a lost or
    peeked decision is visible), plus `extra` ordinary ones in between."""
    flags = _oracle_month_end_flags(frames)
    dates = [d for (d, f), (_, prev) in zip(flags[1:], flags[:-1]) if f != prev]
    assert len(dates) >= 4                       # power guard
    ordinary = [d for (d, f), (_, prev) in zip(flags[1:], flags[:-1]) if f == prev]
    return sorted(dates + ordinary[10:10 + extra])


def test_engine_positions_match_the_independent_oracle_on_a_multi_regime_path():
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    trades, active = _run_engine(frames)
    expected = _oracle_active_assets(frames)
    assert list(active) == expected
    # Power guard: the scenario must actually exercise both assets and
    # several switches, otherwise agreement proves little.
    held = [a for a in expected if a is not None]
    assert set(held) == {EASING_TARGET, OTHERWISE_TARGET}
    switches = sum(1 for a, b in zip(held, held[1:]) if a != b)
    assert switches >= 4
    assert len(trades) == switches + 1
    assert set(trades["asset"]) == {EASING_TARGET, OTHERWISE_TARGET}


def test_a_switch_only_ever_fills_on_the_first_session_after_a_month_end():
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    trades, _ = _run_engine(frames)
    ends = set(mod.month_end_dates(index))
    assert len(trades) >= 5
    for entry_date in trades["entry_date"]:
        previous_session = index[index.get_loc(entry_date) - 1]
        assert previous_session in ends, entry_date


def test_hand_worked_step_down_holds_iwm_exactly_while_rtb_is_at_or_below_minus_0_10():
    """Analytic scenario, no pandas rolling involved. The bill rate sits at
    2.0 then steps to 1.5. With k sessions at 1.5 in the trailing window
    (month-end included): rtb = -0.5 * (252 - k) / 252, which is at or
    below -0.10 exactly while 252 - k >= 50.4, i.e. k <= 201."""
    index = _bdays("2012-01-02", "2014-12-31")
    step_pos = int(index.get_loc(pd.Timestamp("2013-03-15")))
    values = np.full(len(index), 2.0)
    values[step_pos:] = 1.5
    frames = _frames(index, values)

    ends = list(mod.month_end_dates(index))
    after_step = [m for m in ends if index.get_loc(m) >= step_pos]
    def k_of(m):
        return int(index.get_loc(m)) - step_pos + 1
    easing_ends = [m for m in after_step if k_of(m) <= 201]
    first_off = [m for m in after_step if k_of(m) > 201][0]
    assert len(easing_ends) >= 5           # scenario has real length

    trades, active = _run_engine(frames)
    iwm_segments = trades[trades["asset"] == EASING_TARGET]
    assert len(iwm_segments) == 1
    segment = iwm_segments.iloc[0]
    # decision on bar m1 (close) -> fill at the open of the next session
    assert segment["entry_date"] == index[index.get_loc(easing_ends[0]) + 1]
    assert segment["entry_px"] == frames[EASING_TARGET]["Open"].loc[segment["entry_date"]]
    # the first non-easing month-end hands the account back to SPY, again at
    # the NEXT session's open
    assert segment["exit_date"] == index[index.get_loc(first_off) + 1]
    # and the sessions in between are held in IWM, no other asset
    held_between = active.loc[segment["entry_date"]:first_off]
    assert (held_between == EASING_TARGET).all()
    assert active.loc[index[index.get_loc(first_off) + 1]] == OTHERWISE_TARGET


def test_an_intra_month_flip_does_not_move_the_position_before_the_month_end():
    index = _bdays("2012-01-02", "2014-12-31")
    flip = pd.Timestamp("2013-03-15")             # mid-month
    frames = _frames(index, _step_down_irx(index, "2013-03-15"))
    _, active = _run_engine(frames)
    march_end = mod.month_end_dates(index)[
        mod.month_end_dates(index) >= flip][0]
    assert march_end.month == 3
    # SPY all the way through the flip month, including the month-end bar
    assert (active.loc[flip:march_end] == OTHERWISE_TARGET).all()
    assert active.loc[index[index.get_loc(march_end) + 1]] == EASING_TARGET


# ------------------------------------------------------- factory validation

def test_make_target_fn_requires_every_traded_and_signal_frame():
    index = _bdays("2015-01-02", "2016-12-30")
    frames = _frames(index, np.full(len(index), 2.0))
    for missing in (EASING_TARGET, OTHERWISE_TARGET, IRX):
        partial = {k: v for k, v in frames.items() if k != missing}
        with pytest.raises(ValueError, match=missing.replace("^", r"\^")):
            mod.make_target_fn(partial)


def test_make_target_fn_rejects_an_empty_or_closeless_frame():
    index = _bdays("2015-01-02", "2016-12-30")
    frames = _frames(index, np.full(len(index), 2.0))
    empty = dict(frames, **{IRX: frames[IRX].iloc[0:0]})
    with pytest.raises(ValueError, match=r"\^IRX"):
        mod.make_target_fn(empty)
    closeless = dict(frames, **{IRX: frames[IRX].drop(columns=["Close"])})
    with pytest.raises(ValueError, match="Close"):
        mod.make_target_fn(closeless)


def test_make_target_fn_refuses_to_build_when_no_month_end_has_a_reading():
    """A family that could never take a position must not be scored as if it
    had been tested: with fewer than 252 bill-rate prints before every traded
    month-end there is no reading at all, and the factory says so instead of
    returning a target that is flat forever."""
    index = _bdays("2015-01-02", "2015-09-30")           # ~195 sessions
    frames = _frames(index, np.full(len(index), 2.0))
    assert len(index) < WINDOW
    with pytest.raises(ValueError, match="no month-end"):
        mod.make_target_fn(frames)


@pytest.mark.parametrize("hole", ["irx_only", "every_frame"])
def test_make_target_fn_refuses_a_complete_month_that_has_no_reading(hole):
    """A one-shot score must never be recorded with a silently skipped
    decision: once readings have begun, a COMPLETE month whose scheduled
    month-end has no reading (a vendor hole in the bill-rate file, or a
    session missing from the traded frames) makes the factory raise, naming
    the month. The explicit opt-out carries the previous regime instead."""
    index = _bdays("2012-01-02", "2014-12-31")
    frames = _frames(index, _step_down_irx(index, "2013-03-15"))
    march_end = [m for m in mod.month_end_dates(index)
                 if m.year == 2013 and m.month == 3][0]
    if hole == "irx_only":
        frames[IRX] = frames[IRX].drop(march_end)
    else:
        frames = {key: frame.drop(march_end) for key, frame in frames.items()}
    with pytest.raises(ValueError, match="2013-03"):
        mod.make_target_fn(frames)
    assert callable(mod.make_target_fn(frames, require_complete_month_ends=False))


def test_make_target_fn_does_not_demand_a_reading_for_the_month_still_in_progress():
    """Data that ends mid-month has not reached that month's scheduled
    month-end yet, so nothing is missing."""
    index = _bdays("2012-01-02", "2014-06-18")
    frames = _frames(index, _step_down_irx(index, "2013-03-15"))
    assert index[-1] < mod.scheduled_month_end(2014, 6)
    assert callable(mod.make_target_fn(frames))


def test_make_target_fn_does_not_demand_readings_during_the_warm_up():
    """Before the first reading (the first 252 bill-rate prints) there is
    nothing to be missing."""
    index = _bdays("2012-01-02", "2013-12-31")
    frames = _frames(index, np.full(len(index), 2.0))
    assert mod.easing_state_series(
        mod._clean_close(frames[IRX]), index).index[0] > index[200]
    assert callable(mod.make_target_fn(frames))


# ------------------------------- no look-ahead / hold-out blindness

def _calendar_derived_cutoffs(index: pd.DatetimeIndex, month_end_dates,
                              offsets=(-1, 0, 1, 2)) -> list:
    """Cutoff sessions placed RELATIVE TO regime-changing month-ends of the
    calendar (the only place a lost or peeked decision is visible), never at
    round numbers."""
    cutoffs = set()
    for date in month_end_dates:
        base = int(index.get_loc(date))
        for offset in offsets:
            cutoffs.add(base + offset)
    return sorted(cutoffs)


def test_engine_run_on_truncated_data_equals_the_prefix_of_the_full_run():
    """Real engine, real target factory: run everything on data that ENDS at
    the cutoff (target factory built from the truncated frames too, i.e. all
    that anyone could know then) and require the held asset on every session
    up to the cutoff to equal the full-history run's."""
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    _, full_active = _run_engine(frames)
    full = list(full_active)
    cutoffs = _calendar_derived_cutoffs(
        index, _transition_month_ends(frames)[:4] + _transition_month_ends(frames)[-1:])
    assert len(cutoffs) >= 16
    checked_switch_neighbourhood = False
    for cutoff in cutoffs:
        truncated = _truncate(frames, index[cutoff])
        _, active = _run_engine(truncated)
        assert list(active) == full[:cutoff + 1], (
            f"held asset changed with data after {index[cutoff].date()}")
        if any(a != b for a, b in zip(full[:cutoff], full[1:cutoff + 1])):
            checked_switch_neighbourhood = True
    assert checked_switch_neighbourhood      # at least one cutoff saw a switch


def _corrupt_after(frame: pd.DataFrame, cutoff: pd.Timestamp, value) -> pd.DataFrame:
    out = frame.copy(deep=True)
    out.loc[out.index > cutoff, ["Open", "High", "Low", "Close"]] = value
    return out


@pytest.mark.parametrize("garbage", [-50.0, 0.0, 1.0e6])
def test_month_end_decision_does_not_depend_on_any_later_irx_print(garbage):
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    original_target = mod.make_target_fn(frames)
    tested = 0
    decisions = set()
    for m in _transition_month_ends(frames, extra=2):
        variant = dict(frames)
        variant[IRX] = _corrupt_after(frames[IRX], m, garbage)
        variant_target = mod.make_target_fn(variant)
        view = _truncate(frames, m)
        assert variant_target(view) == original_target(view), m
        decisions.add(original_target(view))
        tested += 1
    assert tested >= 6
    assert decisions == {EASING_TARGET, OTHERWISE_TARGET}   # power guard


def test_month_end_decision_does_not_depend_on_any_later_irx_rows_at_all():
    """Removing every later IRX row (a shorter signal file) must not move the
    decision either: month-ends come from the TRADED calendar, so the
    availability of future IRX rows cannot change which session is a
    month-end."""
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    original_target = mod.make_target_fn(frames)
    for m in _transition_month_ends(frames, extra=2):
        variant = dict(frames)
        variant[IRX] = frames[IRX].loc[frames[IRX].index <= m]
        view = _truncate(frames, m)
        # (the shortened bill-rate file leaves every later month without a
        # reading, so the strict completeness check is opted out on purpose)
        rebuilt = mod.make_target_fn(variant, require_complete_month_ends=False)
        assert rebuilt(view) == original_target(view), m


def test_month_end_decision_does_not_depend_on_later_traded_prices():
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    original_target = mod.make_target_fn(frames)
    for m in _transition_month_ends(frames, extra=2):
        variant = dict(frames)
        for leg in (EASING_TARGET, OTHERWISE_TARGET):
            variant[leg] = _corrupt_after(frames[leg], m, 1.0e6)
        view = _truncate(frames, m)
        assert mod.make_target_fn(variant)(view) == original_target(view), m


def test_a_month_end_decision_survives_truncating_every_frame_at_that_month_end():
    """The strongest form of no-look-ahead: cut ALL THREE frames (both traded
    legs and the bill rate) off exactly at a regime-changing month-end, so
    not even the existence of a later session remains, and rebuild the
    target from that. Because month-ends come from an ex-ante schedule
    rather than from a next date in the data, the decision is unchanged."""
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    original = mod.make_target_fn(frames)
    switches = 0
    ends = list(mod.month_end_dates(index))
    for m in _transition_month_ends(frames, extra=2):
        truncated = _truncate(frames, m)
        assert m == truncated[EASING_TARGET].index[-1]
        decision = mod.make_target_fn(truncated)(truncated)
        assert decision == original(_truncate(frames, m)), m
        previous = original(_truncate(frames, ends[ends.index(m) - 1]))
        switches += decision != previous
    assert switches >= 4          # the cut month-ends were real regime switches


# ------------- restricted bucket starting MID-REGIME, through the real engine

def _restricted_bucket_frames(frames: dict, start: pd.Timestamp,
                              end: pd.Timestamp) -> dict:
    return {key: frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
            for key, frame in frames.items()}


def _mid_regime_start(index, oracle_held, want_asset, after=300):
    """A session at least 5 sessions AFTER a month-end and at least 5 before
    the next one, inside a stretch where the held asset is `want_asset` for
    the whole surrounding month (so a naive decision-bar-only signal has
    nothing to go on at the bucket start)."""
    ends = set(mod.month_end_dates(index))
    for k in range(after, len(index) - 260):
        window = oracle_held[k - 3:k + 15]
        if any(a != want_asset for a in window):
            continue
        if any(index[j] in ends for j in range(k - 4, k + 6)):
            continue
        return k
    raise AssertionError(f"no mid-regime {want_asset} start found")


@pytest.mark.parametrize("want_asset", [EASING_TARGET, OTHERWISE_TARGET])
def test_a_bucket_starting_mid_regime_holds_the_carried_in_asset_from_its_second_bar(
        want_asset):
    """REAL engine on a sliced bucket whose first session is mid-month inside
    a regime. The target factory is built once from the FULL frames (exactly
    as both real drivers do); the engine only sees the slice. The carried-in
    state must be usable immediately: the very first fill (bar 1's open)
    is the regime's asset, not flat, and every later session matches the
    full-history run."""
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    target = mod.make_target_fn(frames)
    _, full_active = _run_engine(frames, target_fn=target)
    full = list(full_active)
    k = _mid_regime_start(index, full, want_asset)
    start, end = index[k], index[k + 250]
    bucket = _restricted_bucket_frames(frames, start, end)
    trades, active = _run_engine(bucket, target_fn=target)
    held = list(active)
    assert held[0] is None                      # nothing filled before bar 1
    assert held[1] == want_asset                # ... then the carried-in state
    assert held[1:] == full[k + 1:k + 251]
    assert None not in held[1:]
    assert trades.iloc[0]["asset"] == want_asset
    assert trades.iloc[0]["entry_date"] == index[k + 1]


def test_a_bucket_whose_own_irx_slice_is_too_short_still_gets_the_full_history_state():
    """The bucket driver slices the SIGNAL frame to the bucket too, leaving
    fewer than 252 IRX sessions inside it. The state was computed from the
    full history at build time, so the slice's length cannot matter; a rule
    that recomputed from the sliced frame would lose its warm-up."""
    index = _bdays("2012-01-02", "2017-12-29")
    frames = _frames(index, _regime_irx(index))
    target = mod.make_target_fn(frames)
    _, full_active = _run_engine(frames, target_fn=target)
    full = list(full_active)
    k = _mid_regime_start(index, full, EASING_TARGET)
    bucket = _restricted_bucket_frames(frames, index[k], index[k + 120])
    assert len(bucket[IRX]) < WINDOW
    _, active = _run_engine(bucket, target_fn=target)
    assert list(active)[1:] == full[k + 1:k + 121]


def _bucket_driver(frames: dict, holdout_start: str, holdout_end: str, target):
    return batch3.run_rotation_windows(
        {EASING_TARGET: frames[EASING_TARGET],
         OTHERWISE_TARGET: frames[OTHERWISE_TARGET]},
        {IRX: frames[IRX]},
        target,
        holdout_start=holdout_start, holdout_end=holdout_end,
        cost_bps_by_asset={EASING_TARGET: 1.0, OTHERWISE_TARGET: 1.0},
        tax_haircut_rate=0.0)


def test_real_bucket_driver_is_holdout_blind_and_carries_the_regime_into_the_holdout():
    """scripts/run_batch3_mining.run_rotation_windows is a per-trade bucket
    driver: in-sample bucket =
    everything before the holdout start, holdout bucket = a lookback-padded
    window with spanning trades marked to market at the first real holdout
    session. (It normalizes signal frames with the strict positivity gate,
    so this test uses an all-positive bill-rate path.)"""
    index = _bdays("2012-01-02", "2017-12-29")
    irx = _regime_irx(index)
    assert irx.min() > 0.0
    frames = _frames(index, irx)
    target = mod.make_target_fn(frames)
    _, full_active = _run_engine(frames, target_fn=target)
    full = list(full_active)
    k = _mid_regime_start(index, full, EASING_TARGET, after=900)
    holdout_start, holdout_end = index[k], index[k + 200]
    result = _bucket_driver(frames, holdout_start.strftime("%Y-%m-%d"),
                            holdout_end.strftime("%Y-%m-%d"), target)

    # the regime in force at the holdout start is carried in and marked to
    # market at the first REAL holdout session
    holdout_trades = result["holdout_trades"]
    first = holdout_trades.iloc[0]
    assert first["asset"] == EASING_TARGET
    assert first["entry_date"] == holdout_start
    assert result["alignment"]["holdout"]["warmup_trades_marked_to_market"] >= 1

    # blind: wreck EVERYTHING dated on or after the holdout start (both
    # traded legs and the bill rate); the in-sample trades cannot move
    wrecked = {key: _corrupt_after(frame, holdout_start - pd.Timedelta(days=1), 7.0)
               for key, frame in frames.items()}
    wrecked_result = _bucket_driver(
        wrecked, holdout_start.strftime("%Y-%m-%d"),
        holdout_end.strftime("%Y-%m-%d"), mod.make_target_fn(wrecked))
    pd.testing.assert_frame_equal(
        result["in_sample_trades"], wrecked_result["in_sample_trades"])
    assert len(result["in_sample_trades"]) >= 3


# ----------------------------------------------- degraded / awkward data

def _run_like_abi1(frames: dict, **factory_kwargs):
    """The wiring scripts/run_abi1_mining.py + abi1_driver.build_rotation_curve
    use: the target factory gets the FULL, UNALIGNED frames; the engine gets
    the two traded legs and the IRX frame intersected onto one calendar."""
    target_fn = mod.make_target_fn(frames, **factory_kwargs)
    aligned_assets, aligned_signals = abi1_driver._align_rotation_frames(
        {EASING_TARGET: frames[EASING_TARGET],
         OTHERWISE_TARGET: frames[OTHERWISE_TARGET]},
        {IRX: frames[IRX]})
    trades, active = rotation_engine.run_two_asset_rotation(
        aligned_assets, target_fn,
        cost_bps_by_asset={EASING_TARGET: 1.0, OTHERWISE_TARGET: 1.0},
        signal_frames=aligned_signals)
    return trades, active, aligned_assets


def _zirp_irx_with_negative_month_end(index: pd.DatetimeIndex):
    """A bill rate that falls to near zero and then dips genuinely NEGATIVE,
    including a negative print exactly ON a month-end (the real minimum,
    -0.105, is used)."""
    rng = np.random.RandomState(3)
    n = len(index)
    values = np.full(n, 1.0)
    values[300:420] = np.linspace(1.0, 0.02, 120)
    values[420:] = 0.02 + rng.normal(0.0, 0.003, n - 420)
    ends = list(mod.month_end_dates(index))
    bad = int(index.get_loc(ends[45]))
    values[bad - 8:bad + 1] = np.linspace(-0.02, -0.105, 9)
    values[bad + 1:bad + 12] = np.linspace(-0.05, 0.03, 11)
    return values, ends[45]


def test_negative_bill_rate_prints_flow_through_the_real_engine_and_match_the_oracle():
    index = _bdays("2012-01-02", "2017-12-29")
    values, negative_month_end = _zirp_irx_with_negative_month_end(index)
    frames = _frames(index, values)
    assert frames[IRX].loc[negative_month_end, "Close"] == pytest.approx(-0.105)
    assert (frames[IRX]["Close"] < 0.0).sum() >= 15
    trades, active, _ = _run_like_abi1(frames)
    assert list(active) == _oracle_active_assets(frames)
    # the negative print really was a decision input: its own rtb is well
    # below zero and on the easing side of the band
    rtb = mod.relative_bill_rate_at(
        mod._clean_close(frames[IRX]), pd.DatetimeIndex([negative_month_end]))
    assert np.isfinite(rtb.iloc[0]) and rtb.iloc[0] <= THRESHOLD
    assert len(trades) >= 2


def test_a_single_bad_low_month_end_print_moves_the_account_for_exactly_one_month():
    """Constant 2.0 bill rate (rtb = 0 everywhere, so SPY throughout) with ONE
    month-end print of 0.0. That month-end reads rtb = -2*251/252 (easing);
    the NEXT month-end reads +2/252 (the bad print sits in its window but
    barely moves the mean), so the account is back in SPY. One IWM segment,
    one month long."""
    index = _bdays("2012-01-02", "2014-12-31")
    values = np.full(len(index), 2.0)
    ends = list(mod.month_end_dates(index))
    bad_end, next_end = ends[20], ends[21]
    values[int(index.get_loc(bad_end))] = 0.0
    frames = _frames(index, values)
    trades, active, _ = _run_like_abi1(frames)
    iwm = trades[trades["asset"] == EASING_TARGET]
    assert len(iwm) == 1
    assert iwm.iloc[0]["entry_date"] == index[index.get_loc(bad_end) + 1]
    assert iwm.iloc[0]["exit_date"] == index[index.get_loc(next_end) + 1]
    assert len(trades) == 3                    # SPY, one IWM month, SPY again


def test_a_single_high_outlier_print_inflates_the_trailing_mean_for_a_year_KNOWN_LIMITATION():
    """Pins a DISCLOSED limitation of the pre-registered rule, not a defect
    fix: the rule applies no outlier handling, so one wild HIGH print (30.0
    against a flat 2.0) shifts the trailing mean by 28/252 = 0.111 and every
    month-end whose window still contains it reads rtb = -0.111 <= -0.10
    (easing) until it leaves the window 252 sessions later."""
    index = _bdays("2012-01-02", "2014-12-31")
    values = np.full(len(index), 2.0)
    ends = list(mod.month_end_dates(index))
    spike_pos = int(index.get_loc(ends[20])) + 5           # mid-month
    values[spike_pos] = 30.0
    frames = _frames(index, values)
    _, active, _ = _run_like_abi1(frames)
    easing_ends = [m for m in ends
                   if spike_pos < index.get_loc(m) <= spike_pos + 251]
    assert len(easing_ends) >= 10                         # about a year
    trades, _, _ = _run_like_abi1(frames)
    iwm = trades[trades["asset"] == EASING_TARGET]
    assert len(iwm) == 1
    assert iwm.iloc[0]["entry_date"] == index[index.get_loc(easing_ends[0]) + 1]
    first_clear = [m for m in ends if index.get_loc(m) > spike_pos + 251][0]
    assert iwm.iloc[0]["exit_date"] == index[index.get_loc(first_clear) + 1]


def test_a_missing_irx_print_at_a_traded_month_end_skips_exactly_that_decision():
    """The bill rate steps down mid-March so March's month-end is the first
    easing reading. Delete the IRX row at that month-end only (a data hole
    in the signal file, traded frames intact): that month has NO reading, the
    account carries the previous regime (SPY) through April, and the switch
    happens one month later off April's reading."""
    index = _bdays("2012-01-02", "2014-12-31")
    frames = _frames(index, _step_down_irx(index, "2013-03-15"))
    ends = list(mod.month_end_dates(index))
    march_end = [m for m in ends if m.year == 2013 and m.month == 3][0]
    april_end = [m for m in ends if m.year == 2013 and m.month == 4][0]

    control_trades, _, _ = _run_like_abi1(frames)
    control_iwm = control_trades[control_trades["asset"] == EASING_TARGET]
    assert control_iwm.iloc[0]["entry_date"] == index[index.get_loc(march_end) + 1]

    holed = dict(frames)
    holed[IRX] = frames[IRX].drop(march_end)
    calendar = mod._clean_close(holed[EASING_TARGET]).index
    assert march_end in set(mod.month_end_dates(calendar))      # traded month-end
    state = mod.easing_state_series(mod._clean_close(holed[IRX]), calendar)
    assert march_end not in state.index                          # ... no reading

    trades, active, aligned = _run_like_abi1(
        holed, require_complete_month_ends=False)
    iwm = trades[trades["asset"] == EASING_TARGET]
    aligned_index = aligned[EASING_TARGET].index
    assert iwm.iloc[0]["entry_date"] == aligned_index[
        aligned_index.get_loc(april_end) + 1]
    assert (active.loc[index[index.get_loc(march_end) + 1]:april_end]
            == OTHERWISE_TARGET).all()


def test_a_missing_scheduled_month_end_in_every_frame_skips_that_months_decision():
    """A whole-market hole on the last scheduled session of March (all three
    frames): March gets NO reading (never a guess from an earlier day, which
    would need foreknowledge of the closure), the previous regime is carried
    through April, and the switch happens one month later off April's
    reading."""
    index = _bdays("2012-01-02", "2014-12-31")
    frames = _frames(index, _step_down_irx(index, "2013-03-15"))
    ends = list(mod.month_end_dates(index))
    march_end = [m for m in ends if m.year == 2013 and m.month == 3][0]
    april_end = [m for m in ends if m.year == 2013 and m.month == 4][0]
    previous = index[index.get_loc(march_end) - 1]

    holed = {key: frame.drop(march_end) for key, frame in frames.items()}
    holed_calendar = holed[EASING_TARGET].index
    assert march_end not in holed_calendar
    assert previous not in set(mod.month_end_dates(holed_calendar))
    state = mod.easing_state_series(mod._clean_close(holed[IRX]), holed_calendar)
    assert not any(d.year == 2013 and d.month == 3 for d in state.index)

    trades, active, aligned = _run_like_abi1(
        holed, require_complete_month_ends=False)
    iwm = trades[trades["asset"] == EASING_TARGET]
    assert len(iwm) == 1
    aligned_index = aligned[EASING_TARGET].index
    assert iwm.iloc[0]["entry_date"] == aligned_index[
        aligned_index.get_loc(april_end) + 1]


# ------------------ the REAL ABI-1 driver preflight and dry-run (no network)

def _abi1_driver_scope(tmp_path, *, catalog_parameters=None, loader=None):
    """A scope for scripts/run_abi1_mining.py built from tmp files only: the
    real catalog entry for this family (optionally with drifted parameters),
    a private copy of the module, a tmp ledger that is never written (every
    call below is a dry run or a skip), and clearly SYNTHETIC quotes."""
    import run_batch5_mining as batch5
    entry = _catalog_entry()
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
        "quotes": {"IWM": quote, "SPY": quote}}), encoding="utf-8")
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


def test_real_abi1_driver_skips_a_drifted_catalog_entry_before_loading_any_data(tmp_path):
    import run_abi1_mining as abi1_cli
    real = _catalog_entry()["parameters"]
    for number, drifted in enumerate((
            {**real, "detrend_window_sessions": 251},
            {**real, "easing_threshold_pp": -0.11})):
        case = tmp_path / f"case{number}"
        case.mkdir()
        scope, ledger = _abi1_driver_scope(case, catalog_parameters=drifted)
        result = abi1_cli.compute_batch_abi1(
            scope, tested_date="2026-09-09", run_id="drift-test",
            ledger_path=ledger, dry_run=True)
        assert result["computed"] == []
        skip = {row["family_id"]: row for row in result["skipped"]}[FAMILY_ID]
        assert skip["reason_code"] == "parameter_drift"
        assert not os.path.exists(ledger)


def test_real_abi1_driver_dry_run_scores_the_family_with_negative_irx_and_writes_no_ledger_row(
        tmp_path):
    """The whole real dispatch (module discovery -> drift check -> quote gate
    -> OHLC load with the signal-only positivity exemption -> rotation curve
    -> ABI-1 evaluation) on SYNTHETIC frames whose bill rate really goes
    negative. A traded-symbol positivity gate applied to ^IRX would reject it
    at load; scoring proves ^IRX is treated as signal-only. Dry run: nothing
    is written to any ledger, and the synthetic verdict says nothing about
    real data."""
    import run_abi1_mining as abi1_cli
    index = _bdays("2015-01-02", "2022-12-30")
    values, _ = _zirp_irx_with_negative_month_end(index)
    frames = _frames(index, values)
    assert (frames[IRX]["Close"] < 0.0).any()

    def loader(symbol):
        return frames[symbol].reset_index().rename(columns={"index": "Date"})

    scope, ledger = _abi1_driver_scope(tmp_path, loader=loader)
    result = abi1_cli.compute_batch_abi1(
        scope, tested_date="2026-09-09", run_id="dry-run-wiring",
        ledger_path=ledger, dry_run=True)
    assert result["skipped"] == [], result["skipped"]
    assert [row["family_id"] for row in result["computed"]] == [FAMILY_ID]
    assert result["input_errors"] == {}
    assert not os.path.exists(ledger)
