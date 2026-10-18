"""Tests for src/mining/batch5/month_end_rebalancing_flow_rotation.py.

Synthetic frames only (no network).
"""
import datetime
import importlib
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import trading_calendar
from mining import rotation_engine
from mining.batch5 import month_end_rebalancing_flow_rotation as mod

FAMILY_ID = "month_end_rebalancing_flow_rotation"
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           f"{FAMILY_ID}.py")


# ------------------------------------------------------------ synthetic data

def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _flat_frame(index: pd.DatetimeIndex, level: float = 100.0) -> pd.DataFrame:
    values = np.full(len(index), float(level))
    return pd.DataFrame({"Open": values, "High": values * 1.001,
                         "Low": values * 0.999, "Close": values}, index=index)


def _random_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0003, 0.01, len(index))
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _frames(index: pd.DatetimeIndex, *, random: bool = False) -> dict:
    if random:
        return {"SPY": _random_frame(index, 1), "SH": _random_frame(index, 2),
                "IEF": _random_frame(index, 3)}
    return {"SPY": _flat_frame(index), "SH": _flat_frame(index, 50.0),
            "IEF": _flat_frame(index, 80.0)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _month_sessions(index: pd.DatetimeIndex, year: int, month: int) -> list:
    return [ts for ts in index if ts.year == year and ts.month == month]


def _set_close(frame: pd.DataFrame, date: pd.Timestamp, close: float) -> None:
    frame.loc[date, "Close"] = close
    frame.loc[date, "High"] = max(frame.loc[date, "High"], close)
    frame.loc[date, "Low"] = min(frame.loc[date, "Low"], close)


# ------------------------------------------------------------- contract

# --------------------------------------------------------- calendar roles

def test_session_roles_label_the_last_seven_sessions_of_each_month():
    index = _bdays("2020-01-01", "2020-12-31")
    roles = mod.session_roles(index)
    march = _month_sessions(index, 2020, 3)
    feb = _month_sessions(index, 2020, 2)
    t = march[-1]
    for k in range(7):
        role = roles[march[-1 - k]]
        assert role.offset == k
        assert role.month == pd.Period(t, freq="M")
        assert role.decision_date == march[-7]          # T-6
        assert role.prev_month_end == feb[-1]            # prev_T
    assert roles[march[-8]].offset == 7                  # T-7: outside the window
    # January has no previous month in this calendar.
    jan = _month_sessions(index, 2020, 1)
    assert roles[jan[-1]].prev_month_end is None


def test_calendar_extends_past_the_frames_with_the_vendored_nyse_sessions():
    """Frames ending mid-September 2026 must still see 2026-09-30 as T,
    because the XNYS calendar is known in advance (trading_calendar)."""
    sessions = trading_calendar.load_default_sessions()
    end = datetime.date(2026, 9, 4)
    early = pd.bdate_range("2025-01-02",
                           pd.Timestamp(sessions[0]) - pd.Timedelta(days=1))
    late = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions if s <= end])
    index = early.append(late)
    frames = _frames(index)
    calendar = mod.build_session_calendar(frames)
    sept = [ts for ts in calendar if ts.year == 2026 and ts.month == 9]
    assert sept[-1] == pd.Timestamp("2026-09-30")
    assert pd.Timestamp("2026-09-07") not in calendar   # Labor Day
    roles = mod.session_roles(calendar)
    role = roles[pd.Timestamp("2026-09-04")]
    assert role.offset > 6                                # not yet in the window
    # The vendored calendar must be the source of the extension, not bdays.
    assert calendar.max() == pd.Timestamp(sessions[-1])


# ---------------------------------------------------------- the rule itself

def _prepare_signal_month(frames: dict, index: pd.DatetimeIndex, year: int,
                          month: int, *, spy_ret: float, ief_ret: float):
    """Set SPY/IEF closes from prev_T to T-6 so the month-to-date returns at
    T-6 are exactly spy_ret / ief_ret (flat everywhere else)."""
    sessions = _month_sessions(index, year, month)
    t6 = sessions[-7]
    for date in sessions[:len(sessions) - 6]:
        _set_close(frames["SPY"], date, 100.0 * (1.0 + spy_ret))
        _set_close(frames["IEF"], date, 80.0 * (1.0 + ief_ret))
    return sessions, t6


def test_equities_overweight_targets_sh_for_the_five_hold_sessions():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _frames(index)
    sessions, t6 = _prepare_signal_month(frames, index, 2020, 5,
                                         spy_ret=0.05, ief_ret=0.01)
    target = mod.make_target_fn(frames)
    # T-7 and earlier: flat.
    assert target(_truncate(frames, sessions[-8])) is None
    # T-6 decision through T-2 hold: SH (stocks beat bonds month-to-date).
    for date in sessions[-7:-2]:
        assert target(_truncate(frames, date)) == "SH", date
    # T-1 clears the target (position exits at T's open); T itself flat.
    assert target(_truncate(frames, sessions[-2])) is None
    assert target(_truncate(frames, sessions[-1])) is None


def test_equities_underweight_targets_spy():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _frames(index)
    sessions, _ = _prepare_signal_month(frames, index, 2020, 6,
                                        spy_ret=-0.03, ief_ret=0.02)
    target = mod.make_target_fn(frames)
    for date in sessions[-7:-2]:
        assert target(_truncate(frames, date)) == "SPY", date
    assert target(_truncate(frames, sessions[-2])) is None


def test_zero_deviation_and_missing_prev_month_close_stay_flat():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _frames(index)                        # flat: R_e == R_b == 0
    target = mod.make_target_fn(frames)
    july = _month_sessions(index, 2020, 7)
    for date in july[-7:-2]:
        assert target(_truncate(frames, date)) is None
    # A frame that starts inside the month has no prev_T close -> flat, even
    # though the month-to-date deviation would otherwise be positive.
    frames2 = _frames(index)
    sessions, _ = _prepare_signal_month(frames2, index, 2020, 8,
                                        spy_ret=0.04, ief_ret=0.0)
    target2 = mod.make_target_fn(frames2)
    late_start = {key: frame.loc[frame.index >= sessions[1]]
                  for key, frame in frames2.items()}
    assert target2(_truncate(late_start, sessions[-7])) is None
    # Sanity: with prev_T present the same month targets SH.
    assert target2(_truncate(frames2, sessions[-7])) == "SH"


def test_w_dev_formula_matches_the_candidate_definition():
    spy_ret, ief_ret = 0.02, -0.01
    w_dev = mod.equity_weight_deviation(spy_ret, ief_ret)
    expected = 0.6 * 1.02 / (0.6 * 1.02 + 0.4 * 0.99) - 0.6
    assert w_dev == pytest.approx(expected)
    assert mod.equity_weight_deviation(0.0, 0.0) == 0.0
    assert mod.equity_weight_deviation(float("nan"), 0.0) != \
        mod.equity_weight_deviation(float("nan"), 0.0)  # nan propagates


# ------------------------------------------------------------ look-ahead

def test_perturbing_bars_after_k_never_changes_targets_through_k():
    index = _bdays("2019-01-01", "2021-06-30")
    frames = _frames(index, random=True)
    k = 420
    target = mod.make_target_fn(frames)
    before = [target(_truncate(frames, index[j])) for j in range(k + 1)]
    perturbed = {}
    for key, frame in frames.items():
        copy = frame.copy(deep=True)
        after = copy.index > index[k]
        copy.loc[after, "Close"] *= 1.3
        copy.loc[after, "Open"] *= 1.2
        copy.loc[after, "High"] = copy.loc[after, "High"] * 1.5
        copy.loc[after, "Low"] = copy.loc[after, "Low"] * 0.7
        perturbed[key] = copy
    target_p = mod.make_target_fn(perturbed)
    after = [target_p(_truncate(perturbed, index[j])) for j in range(k + 1)]
    assert before == after
    assert any(value is not None for value in before)


def test_target_uses_only_a_bounded_tail_of_the_truncated_frames():
    """Prices more than TAIL_SESSIONS bars back must be irrelevant: zeroing
    everything before the tail does not change the decision."""
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _frames(index, random=True)
    target = mod.make_target_fn(frames)
    dec = _month_sessions(index, 2020, 9)[-7]
    truncated = _truncate(frames, dec)
    full = target(truncated)
    n = len(truncated["SPY"])
    for frame in truncated.values():
        frame.iloc[:n - mod.TAIL_SESSIONS, :] = 1e-9
    assert target(truncated) == full


# ------------------------------------------------------------- end to end

def test_end_to_end_rotation_produces_five_session_segments():
    index = _bdays("2019-01-01", "2021-06-30")         # ~650 bars
    frames = _frames(index, random=True)
    target = mod.make_target_fn(frames)
    segments, active = rotation_engine.run_two_asset_rotation(
        {"SPY": frames["SPY"], "SH": frames["SH"]}, target,
        cost_bps_by_asset={"SPY": 1.0, "SH": 3.0},
        signal_frames={"IEF": frames["IEF"]})
    assert len(segments) >= 1
    assert set(segments["asset"]).issubset({"SPY", "SH"})
    closed = segments.loc[segments["reason"] != "end_of_data"]
    assert (closed["held"] == 5).all()
    assert (closed["reason"] == "to_flat").all()
    # Every entry is a T-5 session and every exit the month's last session.
    roles = mod.session_roles(mod.build_session_calendar(frames))
    assert all(roles[ts].offset == 5 for ts in closed["entry_date"])
    assert all(roles[ts].offset == 0 for ts in closed["exit_date"])
    # Roughly one segment per month over ~29 months, minus flat months.
    assert len(segments) >= 20
    assert (active.notna().sum()) == closed["held"].sum() + (
        segments.loc[segments["reason"] == "end_of_data", "held"].sum())
