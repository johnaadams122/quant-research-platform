"""Tests for src/mining/batch5/low_vol_high_beta_factor_momentum_rotation.py.

Synthetic frames only (no network).
"""
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
from mining.batch5 import low_vol_high_beta_factor_momentum_rotation as mod

FAMILY_ID = "low_vol_high_beta_factor_momentum_rotation"
RISK, DEFENSIVE = "SPHB", "SPLV"
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
    steps = rng.normal(0.0003, 0.012, len(index))
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _frames(index: pd.DatetimeIndex, *, random: bool = False) -> dict:
    if random:
        return {RISK: _random_frame(index, 11), DEFENSIVE: _random_frame(index, 22)}
    return {RISK: _flat_frame(index, 40.0), DEFENSIVE: _flat_frame(index, 60.0)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _month_sessions(index: pd.DatetimeIndex, year: int, month: int) -> list:
    return [ts for ts in index if ts.year == year and ts.month == month]


def _set_closes(frame: pd.DataFrame, dates, close: float) -> None:
    for date in dates:
        frame.loc[date, "Close"] = close
        frame.loc[date, "High"] = max(frame.loc[date, "High"], close)
        frame.loc[date, "Low"] = min(frame.loc[date, "Low"], close)


def _reference_target():
    return signals.make_monthly_relative_momentum_target_fn(
        RISK, DEFENSIVE, lookback_months=mod.PARAMETERS["lookback_months"])


# ---------------------------------------------------------- the rule itself

def test_risk_leg_wins_when_its_trailing_month_return_is_higher():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _frames(index)
    # SPHB rises from 40 -> 42 (+5%) between the April anchor and every May
    # session; SPLV stays flat -> r_SPHB > r_SPLV -> hold SPHB all of May.
    may = _month_sessions(index, 2020, 5)
    _set_closes(frames[RISK], may, 42.0)
    target = mod.make_target_fn(frames)
    for date in may:
        assert target(_truncate(frames, date)) == RISK, date


def test_defensive_leg_wins_when_the_risk_leg_lags():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _frames(index)
    june = _month_sessions(index, 2020, 6)
    _set_closes(frames[RISK], june, 38.0)          # SPHB -5%
    _set_closes(frames[DEFENSIVE], june, 60.6)     # SPLV +1%
    target = mod.make_target_fn(frames)
    for date in june:
        assert target(_truncate(frames, date)) == DEFENSIVE, date


def test_a_tie_goes_to_the_defensive_leg_and_warmup_is_none():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _frames(index)                        # flat: both returns 0.0
    target = mod.make_target_fn(frames)
    july = _month_sessions(index, 2020, 7)
    for date in july:
        assert target(_truncate(frames, date)) == DEFENSIVE, date
    # Warm-up: the first calendar month of data has no anchor close on or
    # before (first session - 1 month) -> None, never a trade.
    jan = _month_sessions(index, 2020, 1)
    for date in jan:
        assert target(_truncate(frames, date)) is None, date
    # February's first session anchors to 2020-01-03 (<= 2020-01-03), so the
    # rule is live from February on.
    feb = _month_sessions(index, 2020, 2)
    assert target(_truncate(frames, feb[0])) == DEFENSIVE


def test_decision_is_fixed_at_the_first_session_of_the_month():
    """Within a month the target is the FIRST session's decision: later
    price moves inside the month do not change it until the next month."""
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _frames(index)
    aug = _month_sessions(index, 2020, 8)
    _set_closes(frames[RISK], aug[:1], 42.0)       # SPHB +5% on the decision bar
    _set_closes(frames[RISK], aug[1:], 30.0)       # then collapses -25%
    target = mod.make_target_fn(frames)
    for date in aug:
        assert target(_truncate(frames, date)) == RISK, date
    # September's first session sees SPHB 30 vs anchor ~30 (flat) and SPLV
    # flat -> tie -> defensive.
    sept = _month_sessions(index, 2020, 9)
    _set_closes(frames[RISK], sept, 30.0)
    target = mod.make_target_fn(frames)
    assert target(_truncate(frames, sept[0])) == DEFENSIVE


def test_matches_the_repo_reference_factory_bar_for_bar():
    """The family's rule is exactly signals.make_monthly_relative_momentum_
    target_fn(risk='SPHB', defensive='SPLV', lookback_months=1); this module
    must agree with it on every bar of a random tape."""
    index = _bdays("2019-01-01", "2021-06-30")      # ~650 bars
    frames = _frames(index, random=True)
    ours = mod.make_target_fn(frames)
    reference = _reference_target()
    mine = [ours(_truncate(frames, ts)) for ts in index]
    theirs = [reference(_truncate(frames, ts)) for ts in index]
    assert mine == theirs
    assert RISK in mine and DEFENSIVE in mine


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


def test_factory_holds_no_price_state_from_the_full_history_frames():
    """The factory may only fix the rule; prices come from the truncated
    frames the engine hands over. A factory built on unrelated frames must
    give the same answers as one built on the real frames."""
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _frames(index, random=True)
    other = {RISK: _random_frame(index, 99), DEFENSIVE: _random_frame(index, 98)}
    target_a = mod.make_target_fn(frames)
    target_b = mod.make_target_fn(other)
    for ts in index[::7]:
        truncated = _truncate(frames, ts)
        assert target_a(truncated) == target_b(truncated)


def test_target_uses_only_a_bounded_tail_of_the_truncated_frames():
    """Prices more than TAIL_SESSIONS bars back must be irrelevant: zeroing
    everything before the tail does not change the decision."""
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _frames(index, random=True)
    target = mod.make_target_fn(frames)
    for ts in _month_sessions(index, 2020, 9)[::5]:
        truncated = _truncate(frames, ts)
        full = target(truncated)
        n = len(truncated[RISK])
        for frame in truncated.values():
            frame.iloc[:n - mod.TAIL_SESSIONS, :] = 1e-9
        assert target(truncated) == full


# ------------------------------------------------------------- end to end

def test_end_to_end_rotation_is_fully_invested_after_warmup():
    index = _bdays("2019-01-01", "2021-06-30")         # ~650 bars
    frames = _frames(index, random=True)
    target = mod.make_target_fn(frames)
    segments, active = rotation_engine.run_two_asset_rotation(
        {RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]}, target,
        cost_bps_by_asset={RISK: 5.0, DEFENSIVE: 3.0})
    assert len(segments) >= 1
    assert set(segments["asset"]).issubset({RISK, DEFENSIVE})
    closed = segments.loc[segments["reason"] != "end_of_data"]
    # No cash state: every closed segment is a switch, never to_flat.
    assert (closed["reason"] == "switch").all()
    # Once invested (first execution after warm-up) the strategy never goes
    # flat again.
    first_active = active.first_valid_index()
    assert first_active is not None
    assert active.loc[first_active:].notna().all()
    # Every switch executes on the session AFTER a first-session-of-month
    # decision bar (decision on the first session, fill at the next open).
    periods = index.to_period("M")
    first_sessions = {index[int(pos)] for pos in
                      np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]])}
    for entry in closed["entry_date"].iloc[1:]:
        position = index.get_loc(entry)
        assert index[position - 1] in first_sessions, entry
    # ~29 months of random tape: a handful of switches at least.
    assert len(segments) >= 3
