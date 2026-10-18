"""Tests for src/mining/batch5/utilities_beta_rotation_relative_strength.py.

Synthetic frames only (no network).
"""
import json
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

from mining import rotation_engine
from mining.batch5 import utilities_beta_rotation_relative_strength as mod

FAMILY_ID = "utilities_beta_rotation_relative_strength"
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           f"{FAMILY_ID}.py")
XLU, SPY = "XLU", "SPY"


# ------------------------------------------------------------ synthetic data

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


def _frame_from_close(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002,
                         "Low": np.minimum(open_, close) * 0.998,
                         "Close": close}, index=index)


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _perturb_after(frame: pd.DataFrame, k: int,
                   close_scale: float = 1.3) -> pd.DataFrame:
    """Scale every bar AFTER position k (closes x close_scale, highs/lows
    shifted). Callers probing look-ahead MUST scale the two legs by
    DIFFERENT factors: the rule's decision variable is the XLU/SPY ratio,
    so scaling both legs by the same factor leaves every future ratio
    unchanged and a factory reading a FUTURE ratio would still pass."""
    out = frame.copy(deep=True)
    tail = out.index[k + 1:]
    out.loc[tail, "Close"] = out.loc[tail, "Close"] * close_scale
    out.loc[tail, "Open"] = out.loc[tail, "Open"] * 1.25
    out.loc[tail, "High"] = out.loc[tail, "High"] * 1.5
    out.loc[tail, "Low"] = out.loc[tail, "Low"] * 0.7
    return out


def _naive_targets(frames: dict) -> pd.Series:
    """Independent hand computation of the pseudocode on the shared calendar:
    decision bar = last session of each ISO week (next session in a later
    ISO week); ratio vs the ratio 4 decision bars earlier; strict > -> XLU."""
    calendar = frames[XLU].index.intersection(frames[SPY].index).sort_values()
    iso = calendar.isocalendar()
    codes = [int(y) * 100 + int(w) for y, w in zip(iso["year"], iso["week"])]
    decisions = [calendar[i] for i in range(len(calendar) - 1)
                 if codes[i] != codes[i + 1]]
    ratio = frames[XLU]["Close"] / frames[SPY]["Close"]
    decided = {}
    for j, date in enumerate(decisions):
        if j < 4:
            decided[date] = None
        else:
            ref = float(ratio.loc[decisions[j - 4]])
            decided[date] = XLU if float(ratio.loc[date]) > ref else SPY
    standing, out = None, []
    for date in calendar:
        if date in decided:
            standing = decided[date]
        out.append(standing)
    return pd.Series(out, index=calendar, dtype=object)


# ------------------------------------------------------------- contract

# ------------------------------------------------------ decision calendar

def test_weekly_decision_sessions_are_fridays_on_a_plain_calendar():
    calendar = pd.bdate_range("2024-01-01", "2024-03-29")
    decisions = mod.weekly_decision_sessions(calendar)
    # Every decision is a Friday except none is missing; the final session
    # (a Friday) is NOT a decision because there is no next session to fill.
    assert all(ts.dayofweek == 4 for ts in decisions)
    assert calendar[-1] not in decisions
    fridays = [ts for ts in calendar if ts.dayofweek == 4]
    assert list(decisions) == fridays[:-1]


def test_weekly_decision_falls_back_to_thursday_when_friday_is_a_holiday():
    calendar = pd.bdate_range("2024-03-18", "2024-04-12")
    holiday = pd.Timestamp("2024-03-29")  # Good Friday 2024
    calendar = calendar[calendar != holiday]
    decisions = mod.weekly_decision_sessions(calendar)
    assert pd.Timestamp("2024-03-28") in decisions  # Thursday steps in
    assert holiday not in decisions
    assert pd.Timestamp("2024-03-22") in decisions
    assert pd.Timestamp("2024-04-05") in decisions


def test_weekly_decision_handles_iso_year_boundary():
    # 2024-12-30 / 31 belong to ISO week 1 of 2025, not to week 52/53 of 2024.
    calendar = pd.bdate_range("2024-12-16", "2025-01-17")
    decisions = mod.weekly_decision_sessions(calendar)
    assert pd.Timestamp("2024-12-27") in decisions
    assert pd.Timestamp("2024-12-31") not in decisions
    assert pd.Timestamp("2025-01-03") in decisions


# ----------------------------------------------------------- the rule

def _ratio_frames(index: pd.DatetimeIndex, ratio_path: np.ndarray) -> dict:
    """SPY flat at 100, XLU = 100 * ratio_path, so XLU/SPY == ratio_path."""
    spy = _flat_frame(index, 100.0)
    xlu = _frame_from_close(index, 100.0 * np.asarray(ratio_path, dtype=float))
    return {XLU: xlu, SPY: spy}


def test_warm_up_returns_none_until_the_fourth_prior_decision_exists():
    index = pd.bdate_range("2024-01-01", "2024-06-28")
    frames = _ratio_frames(index, np.linspace(1.0, 1.5, len(index)))  # rising
    target = mod.make_target_fn(frames)
    decisions = mod.weekly_decision_sessions(index)
    # Bars through the 4th decision bar (ordinal 3) have no reference -> None.
    for date in index[index <= decisions[3]]:
        assert target(_truncate(frames, date)) is None
    # The 5th decision bar (ordinal 4) is the first decidable one.
    assert target(_truncate(frames, decisions[4])) == XLU
    # A non-decision bar between decisions carries the standing target.
    next_bar = index[index > decisions[4]][0]
    assert next_bar not in decisions
    assert target(_truncate(frames, next_bar)) == XLU


def test_rising_ratio_holds_xlu_and_falling_ratio_holds_spy():
    index = pd.bdate_range("2024-01-01", "2024-12-31")
    n = len(index)
    up = _ratio_frames(index, np.linspace(1.0, 2.0, n))
    down = _ratio_frames(index, np.linspace(2.0, 1.0, n))
    decisions = mod.weekly_decision_sessions(index)
    up_target = mod.make_target_fn(up)
    down_target = mod.make_target_fn(down)
    for date in index[index >= decisions[4]]:
        assert up_target(_truncate(up, date)) == XLU
        assert down_target(_truncate(down, date)) == SPY


def test_tie_goes_to_the_broad_market():
    index = pd.bdate_range("2024-01-01", "2024-12-31")
    frames = {XLU: _flat_frame(index, 60.0), SPY: _flat_frame(index, 400.0)}
    target = mod.make_target_fn(frames)
    decisions = mod.weekly_decision_sessions(index)
    for date in index[index >= decisions[4]]:
        assert target(_truncate(frames, date)) == SPY


def test_signal_compares_against_the_fourth_prior_weekly_decision_exactly():
    """Ratio rises for 4 weeks then drops sharply in the last week: a 1-week
    rule would say SPY, the 4-week rule says XLU (ratio still above the
    4th-prior decision). Then the ratio keeps falling so that 4 weeks later
    the comparison flips to SPY."""
    index = pd.bdate_range("2024-01-01", "2024-12-31")
    decisions = mod.weekly_decision_sessions(index)
    ratio = np.full(len(index), 1.0)
    pos = {ts: i for i, ts in enumerate(index)}
    # Weeks 0..3 (decision ordinals 0-3) ratio 1.00; week 4 rises to 1.10 on
    # its decision bar; week 5..8 drop back a little each week.
    d = decisions
    ratio[pos[d[4]]:] = 1.10
    ratio[pos[d[5]]:] = 1.08
    ratio[pos[d[6]]:] = 1.06
    ratio[pos[d[7]]:] = 1.04
    ratio[pos[d[8]]:] = 1.02
    ratio[pos[d[9]]:] = 1.00
    frames = _ratio_frames(index, ratio)
    target = mod.make_target_fn(frames)
    # ordinal 4: 1.10 > ref(ordinal 0)=1.00 -> XLU
    assert target(_truncate(frames, d[4])) == XLU
    # ordinal 5: 1.08 > ref(ordinal 1)=1.00 -> XLU even though it fell 1 week
    assert target(_truncate(frames, d[5])) == XLU
    # ordinal 8: 1.02 > ref(ordinal 4)=1.10 ? no -> SPY
    assert target(_truncate(frames, d[8])) == SPY
    # ordinal 9: 1.00 > ref(ordinal 5)=1.08 ? no -> SPY
    assert target(_truncate(frames, d[9])) == SPY
    # ordinal 7: 1.04 > ref(ordinal 3)=1.00 -> XLU
    assert target(_truncate(frames, d[7])) == XLU


def test_factory_matches_an_independent_naive_computation_on_random_data():
    index = pd.bdate_range("2019-01-01", "2021-06-30")
    frames = {XLU: _random_frame(index, 11), SPY: _random_frame(index, 12)}
    target = mod.make_target_fn(frames)
    expected = _naive_targets(frames)
    got = [target(_truncate(frames, date)) for date in index]
    assert got == list(expected.to_numpy(dtype=object))
    assert XLU in got and SPY in got  # both legs actually occur


def test_decision_uses_the_shared_calendar_when_one_leg_misses_a_session():
    """A session missing from one leg is not on the shared calendar, so the
    weekly boundary is judged on the sessions BOTH legs actually have."""
    index = pd.bdate_range("2024-01-01", "2024-06-28")
    friday = pd.Timestamp("2024-02-09")
    xlu_index = index[index != friday]
    frames = {XLU: _random_frame(xlu_index, 5), SPY: _random_frame(index, 6)}
    decisions = mod.make_target_fn(frames).decision_dates
    assert friday not in decisions
    assert pd.Timestamp("2024-02-08") in decisions  # Thursday is the shared last session


# ------------------------------------------------------- look-ahead probe

def test_perturbing_bars_after_k_never_changes_signals_through_k():
    index = pd.bdate_range("2019-01-01", "2021-06-30")
    frames = {XLU: _random_frame(index, 21), SPY: _random_frame(index, 22)}
    target = mod.make_target_fn(frames)
    scales = {XLU: 1.3, SPY: 0.8}  # asymmetric: the future RATIO must move
    for k in (60, 200, 333, 500):
        before = [target(_truncate(frames, date)) for date in index[:k + 1]]
        perturbed = {key: _perturb_after(frame, k, close_scale=scales[key])
                     for key, frame in frames.items()}
        # Sanity: the perturbation really changed the future, INCLUDING the
        # decision variable itself (a symmetric scaling would not, and would
        # leave a future-ratio look-ahead undetected).
        assert not perturbed[XLU]["Close"].equals(frames[XLU]["Close"])
        ratio_before = frames[XLU]["Close"] / frames[SPY]["Close"]
        ratio_after = perturbed[XLU]["Close"] / perturbed[SPY]["Close"]
        assert (ratio_after.iloc[:k + 1] == ratio_before.iloc[:k + 1]).all()
        assert (ratio_after.iloc[k + 1:] != ratio_before.iloc[k + 1:]).all()
        rebuilt = mod.make_target_fn(perturbed)
        after = [rebuilt(_truncate(perturbed, date)) for date in index[:k + 1]]
        assert before == after, f"signals through bar {k} changed"


# ------------------------------------------------------ fail-closed guards

def test_missing_leg_raises():
    index = pd.bdate_range("2024-01-01", "2024-06-28")
    with pytest.raises(ValueError):
        mod.make_target_fn({XLU: _flat_frame(index)})
    with pytest.raises(TypeError):
        mod.make_target_fn([_flat_frame(index)])


def test_target_refuses_a_slice_whose_close_differs_from_the_factory_data():
    index = pd.bdate_range("2024-01-01", "2024-06-28")
    frames = {XLU: _random_frame(index, 31), SPY: _random_frame(index, 32)}
    target = mod.make_target_fn(frames)
    sliced = _truncate(frames, index[100])
    sliced[SPY].loc[index[100], "Close"] *= 1.01
    with pytest.raises(ValueError, match="does not match"):
        target(sliced)
    # And a date the factory never saw is refused too.
    other = pd.bdate_range("2030-01-01", "2030-03-01")
    with pytest.raises(ValueError):
        target({XLU: _flat_frame(other), SPY: _flat_frame(other)})


# --------------------------------------------------------- end to end

def test_end_to_end_rotation_engine_produces_trades_and_is_never_flat():
    index = pd.bdate_range("2021-01-01", "2023-05-31")  # ~625 sessions
    assert 600 <= len(index) <= 650
    frames = {XLU: _random_frame(index, 41), SPY: _random_frame(index, 42)}
    target = mod.make_target_fn(frames)
    segments, active = rotation_engine.run_two_asset_rotation(
        frames, target, cost_bps_by_asset={XLU: 5.0, SPY: 1.0})
    assert len(segments) >= 1
    assert set(segments["asset"]).issubset({XLU, SPY})
    assert set(segments["asset"]) == {XLU, SPY}  # a rotation, not a static hold
    # The first fill is the session AFTER the first decidable decision bar.
    decisions = mod.weekly_decision_sessions(index)
    first_fill = index[index > decisions[4]][0]
    assert segments["entry_date"].iloc[0] == first_fill
    # After the first fill the strategy is never flat (no cash state).
    assert active.loc[active.index >= first_fill].notna().all()
    assert active.loc[active.index < first_fill].isna().all()
    # Segments chain without gaps: each exit is the next entry.
    for prev, nxt in zip(segments.itertuples(), segments.iloc[1:].itertuples()):
        assert prev.exit_date == nxt.entry_date
    assert segments["reason"].iloc[-1] == "end_of_data"
    assert (segments["reason"].iloc[:-1] == "switch").all()


def test_decisions_execute_at_next_open_via_the_engine():
    """Weekly step ratio: XLU/SPY jumps on a Friday decision bar, so the
    engine must switch into XLU at the following Monday's open."""
    index = pd.bdate_range("2024-01-01", "2024-12-31")
    decisions = mod.weekly_decision_sessions(index)
    ratio = np.full(len(index), 1.0)
    pos = {ts: i for i, ts in enumerate(index)}
    ratio[pos[decisions[10]]:] = 1.2       # rises at decision 10 -> XLU
    ratio[pos[decisions[15]]:] = 0.9       # falls at decision 15 -> SPY
    frames = _ratio_frames(index, ratio)
    segments, _ = rotation_engine.run_two_asset_rotation(
        frames, mod.make_target_fn(frames),
        cost_bps_by_asset={XLU: 0.0, SPY: 0.0})
    by_entry = {ts: asset for ts, asset in zip(segments["entry_date"], segments["asset"])}
    fill_10 = index[index > decisions[10]][0]
    fill_14 = index[index > decisions[14]][0]
    fill_15 = index[index > decisions[15]][0]
    assert by_entry[fill_10] == XLU
    assert by_entry[fill_14] == SPY   # ordinal 14: 1.2 vs ref ordinal 10 = 1.2 -> tie -> SPY
    assert fill_15 not in by_entry  # already in SPY since fill_14: no churn
    assert segments["entry_date"].iloc[0] == index[index > decisions[4]][0]
    assert segments["asset"].iloc[0] == SPY  # flat ratio during warm-up -> tie -> SPY
