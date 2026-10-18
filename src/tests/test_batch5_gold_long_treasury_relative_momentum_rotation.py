"""Tests for src/mining/batch5/gold_long_treasury_relative_momentum_rotation.py.

Synthetic frames only (no network).
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import rotation_engine, signals
from mining.batch5 import gold_long_treasury_relative_momentum_rotation as mod

FAMILY_ID = "gold_long_treasury_relative_momentum_rotation"
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           f"{FAMILY_ID}.py")
RISK, DEFENSIVE = "GLD", "TLT"


# ------------------------------------------------------------ synthetic data

def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _frame_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _trend_frame(index: pd.DatetimeIndex, daily_rate: float,
                 level: float = 100.0) -> pd.DataFrame:
    close = level * np.exp(daily_rate * np.arange(len(index)))
    return _frame_from_close(close, index)


def _random_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0002, 0.012, len(index))
    return _frame_from_close(100.0 * np.exp(np.cumsum(steps)), index)


def _random_frames(index: pd.DatetimeIndex, seed: int = 7) -> dict:
    return {RISK: _random_frame(index, seed),
            DEFENSIVE: _random_frame(index, seed + 100)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _month_sessions(index: pd.DatetimeIndex, year: int, month: int) -> list:
    return [ts for ts in index if ts.year == year and ts.month == month]


def _reference():
    return signals.make_monthly_relative_momentum_target_fn(
        RISK, DEFENSIVE, lookback_months=12)


# ------------------------------------------------------------- contract

def test_factory_requires_both_legs():
    index = _bdays("2020-01-01", "2020-06-30")
    frames = _random_frames(index)
    try:
        mod.make_target_fn({RISK: frames[RISK]})
    except ValueError as exc:
        assert DEFENSIVE in str(exc)
    else:
        raise AssertionError("make_target_fn accepted a missing TLT frame")


# ---------------------------------------------------------- the rule itself

def test_warm_up_returns_none_until_twelve_calendar_months_exist():
    # bdate_range has no holidays: 2019-01-01 and 2020-01-01 are "sessions".
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _random_frames(index)
    target = mod.make_target_fn(frames)
    # Any decision bar whose anchor (first session of month minus 12 months)
    # predates the first session -> None. 2019-12-02's anchor is 2018-12-02.
    for date in index[index < pd.Timestamp("2020-01-01")]:
        assert target(_truncate(frames, date)) is None, date
    # 2020-01-01's anchor is 2019-01-01 == the first session -> base exists.
    assert target(_truncate(frames, pd.Timestamp("2020-01-01"))) is not None
    # A frame that starts one session later has no close on/before the anchor.
    later = {key: frame.iloc[1:] for key, frame in frames.items()}
    assert mod.make_target_fn(later)(
        _truncate(later, pd.Timestamp("2020-01-01"))) is None


def test_leading_leg_is_held_and_ties_go_to_the_defensive_leg():
    index = _bdays("2019-01-02", "2020-12-31")
    gold_leads = {RISK: _trend_frame(index, 0.001),
                  DEFENSIVE: _trend_frame(index, 0.0)}
    target = mod.make_target_fn(gold_leads)
    for date in index[index >= pd.Timestamp("2020-02-01")]:
        assert target(_truncate(gold_leads, date)) == RISK, date

    bonds_lead = {RISK: _trend_frame(index, 0.0),
                  DEFENSIVE: _trend_frame(index, 0.001)}
    target = mod.make_target_fn(bonds_lead)
    for date in index[index >= pd.Timestamp("2020-02-01")]:
        assert target(_truncate(bonds_lead, date)) == DEFENSIVE, date

    # Identical paths -> identical trailing returns -> strict '>' fails ->
    # the defensive leg (tie_rule == "defensive").
    same = _random_frame(index, 3)
    tie = {RISK: same.copy(), DEFENSIVE: same.copy()}
    target = mod.make_target_fn(tie)
    for date in index[index >= pd.Timestamp("2020-02-01")]:
        assert target(_truncate(tie, date)) == DEFENSIVE, date


def test_decision_is_pinned_to_the_months_first_session():
    """Within a month the target must not drift with later closes: it is the
    first session's decision until the next month's first session."""
    index = _bdays("2019-01-02", "2020-12-31")
    frames = {RISK: _trend_frame(index, 0.0), DEFENSIVE: _trend_frame(index, 0.0)}
    june = _month_sessions(index, 2020, 6)
    july = _month_sessions(index, 2020, 7)
    # Gold leads at June's first session only; afterwards gold collapses so
    # that on every later June close bonds would win a fresh comparison.
    frames[RISK].loc[june[0], "Close"] = 200.0
    frames[RISK].loc[june[0], "High"] = 200.0
    for date in june[1:]:
        frames[RISK].loc[date, "Close"] = 50.0
        frames[RISK].loc[date, "Low"] = 50.0
    target = mod.make_target_fn(frames)
    for date in june:
        assert target(_truncate(frames, date)) == RISK, date
    # July's first session sees gold at 50 vs its year-ago 100 -> bonds.
    assert target(_truncate(frames, july[0])) == DEFENSIVE
    # Sanity: a fresh comparison on a late-June close WOULD have said bonds.
    late = _truncate(frames, june[-1])
    assert float(late[RISK]["Close"].iloc[-1]) < float(late[DEFENSIVE]["Close"].iloc[-1])


def test_matches_the_repo_reference_factory_on_every_bar():
    """The candidate cites signals.make_monthly_relative_momentum_target_fn
    (risk GLD, defensive TLT, 12 months) as the exact rule."""
    index = _bdays("2018-01-02", "2021-06-30")
    frames = _random_frames(index, seed=11)
    ours = mod.make_target_fn(frames)
    reference = _reference()
    decisions = []
    for date in index:
        truncated = _truncate(frames, date)
        decisions.append(ours(truncated))
        assert decisions[-1] == reference(truncated), date
    assert RISK in decisions and DEFENSIVE in decisions and None in decisions


def test_matches_the_reference_when_the_anchor_lies_beyond_the_tail():
    """A calendar dense enough that twelve months exceed TAIL_SESSIONS bars
    (daily incl. weekends) must take the exact full-history path."""
    index = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    assert len(index[index <= pd.Timestamp("2020-01-01")]) > mod.TAIL_SESSIONS
    frames = _random_frames(index, seed=5)
    ours = mod.make_target_fn(frames)
    reference = _reference()
    for date in index[index >= pd.Timestamp("2020-01-01")]:
        truncated = _truncate(frames, date)
        assert ours(truncated) == reference(truncated), date


# ------------------------------------------------------------ look-ahead

def test_perturbing_bars_after_k_never_changes_targets_through_k():
    index = _bdays("2018-01-02", "2021-06-30")
    frames = _random_frames(index, seed=21)
    k = 600
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
    """On a business-day calendar the anchor close is inside TAIL_SESSIONS,
    so garbage before the tail must be irrelevant."""
    index = _bdays("2016-01-04", "2020-12-31")
    frames = _random_frames(index, seed=9)
    target = mod.make_target_fn(frames)
    for decision in (pd.Timestamp("2020-09-01"), pd.Timestamp("2020-09-15")):
        truncated = _truncate(frames, decision)
        full = target(truncated)
        assert full is not None
        n = len(truncated[RISK])
        for frame in truncated.values():
            frame.iloc[:n - mod.TAIL_SESSIONS, :] = 1e-9
        assert target(truncated) == full


def test_factory_ignores_prices_in_the_full_history_frames():
    """Holdout-blindness by construction: the factory's ``frames`` argument
    fixes nothing but the required keys; decisions come from the truncated
    frames the engine hands over."""
    index = _bdays("2018-01-02", "2020-12-31")
    frames = _random_frames(index, seed=2)
    scrambled = {key: frame * 3.0 for key, frame in frames.items()}
    a, b = mod.make_target_fn(frames), mod.make_target_fn(scrambled)
    for date in index[index >= pd.Timestamp("2019-06-01")][::7]:
        truncated = _truncate(frames, date)
        assert a(truncated) == b(truncated)


# ------------------------------------------------------------- end to end

def test_end_to_end_rotation_switches_at_month_starts_and_never_goes_flat():
    index = _bdays("2018-01-01", "2020-06-30")          # ~650 bars
    frames = _random_frames(index, seed=17)
    target = mod.make_target_fn(frames)
    segments, active = rotation_engine.run_two_asset_rotation(
        {RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]}, target,
        cost_bps_by_asset={RISK: 3.7, DEFENSIVE: 2.4})
    assert len(segments) >= 1
    assert set(segments["asset"]).issubset({RISK, DEFENSIVE})
    # Always invested after warm-up: no to_flat exits, ever.
    assert set(segments["reason"]).issubset({"switch", "end_of_data"})
    # A decision made on a month's first session executes at the second
    # session's open, so every segment starts on the 2nd session of a month.
    second_sessions = set()
    for period in index.to_period("M").unique():
        month = index[index.to_period("M") == period]
        if len(month) >= 2:
            second_sessions.add(month[1])
    assert all(ts in second_sessions for ts in segments["entry_date"])
    # First entry: 2019-01-01's decision (anchor 2018-01-01 == first session)
    # executes at the next session's open, 2019-01-02.
    assert segments["entry_date"].min() == pd.Timestamp("2019-01-02")
    # After the first entry the position is never flat.
    first = segments["entry_date"].min()
    assert active.loc[active.index >= first].notna().all()
    assert active.loc[active.index < first].isna().all()
    # Two random legs over 18 post-warm-up months should switch at least once.
    assert (segments["reason"] == "switch").sum() >= 1
