"""Tests for src/mining/batch5/long_treasury_auction_cycle_3x_inverse_rotation.py.

Synthetic frames and synthetic auction calendars only (no network); no
events file is read.
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

import trading_calendar
from mining import rotation_engine
from mining.batch5 import long_treasury_auction_cycle_3x_inverse_rotation as mod

FAMILY_ID = "long_treasury_auction_cycle_3x_inverse_rotation"
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")


# ------------------------------------------------------------ synthetic data

def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _random_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0002, 0.02, len(index))
    close = 50.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _frames(index: pd.DatetimeIndex) -> dict:
    return {"TMF": _random_frame(index, 11), "TMV": _random_frame(index, 12)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _monthly_auctions(index: pd.DatetimeIndex, *, offset: int = 8,
                      every: int = 21) -> list:
    """Synthetic auction calendar: one session every ``every`` sessions."""
    return [index[i] for i in range(offset, len(index) - 6, every)]


# ------------------------------------------- announcement-lead disclosure

def _record(auction: pd.Timestamp, announced: pd.Timestamp) -> dict:
    return {"auction_date": auction.strftime("%Y-%m-%d"),
            "announcement_date": announced.strftime("%Y-%m-%d"),
            "issue_date": (auction + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            "cusip": "912810XX0", "security_term": "30-Year",
            "original_security_term": "30-Year", "reopening": False}


def test_announcement_lead_report_counts_the_two_precedence_cases():
    """Hand-computed: the a-5 DECISION precedes the announcement whenever the
    session lead is under 5; the a-4 EXECUTION does whenever it is under 4."""
    index = _bdays("2020-01-01", "2020-12-31")
    records = [
        _record(index[40], index[40 - 5]),   # lead 5: neither precedes
        _record(index[80], index[80 - 4]),   # lead 4: decision only
        _record(index[120], index[120 - 3]),  # lead 3: decision and execution
    ]
    report = mod.announcement_lead_report(index, records)
    assert report["in_frame_auctions"] == 3
    assert report["session_lead_distribution"] == {"3": 1, "4": 1, "5": 1}
    assert report["decision_precedes_announcement"] == 2
    assert report["first_execution_precedes_announcement"] == 1
    examples = report["first_execution_precedes_announcement_examples"]
    assert len(examples) == 1
    assert examples[0]["auction_date"] == index[120].strftime("%Y-%m-%d")
    assert examples[0]["first_pre_leg_execution_session"] == (
        index[120 - 4].strftime("%Y-%m-%d"))
    # An auction outside the calendar is not counted at all.
    outside = mod.announcement_lead_report(
        index[:20], records)
    assert outside["in_frame_auctions"] == 0


def test_a_holiday_makes_the_weekday_basis_understate_the_session_basis():
    """Why the report counts leads in SESSIONS: five WEEKDAYS of lead can be
    four SESSIONS, so a weekday-basis count that looks safe hides a decision
    taken before the announcement."""
    index = _bdays("2020-01-01", "2020-12-31")
    p = 60
    holiday = index[p - 2]                       # a session between the two
    calendar = index.drop(holiday)
    records = [_record(index[p], index[p - 5])]
    report = mod.announcement_lead_report(calendar, records)
    assert report["weekday_lead_distribution"] == {"5": 1}     # looks safe
    assert report["session_lead_distribution"] == {"4": 1}     # is not
    assert report["decision_precedes_announcement"] == 1
    assert report["first_execution_precedes_announcement"] == 0
    # Without the holiday the same auction is clean on both bases.
    clean = mod.announcement_lead_report(index, records)
    assert clean["session_lead_distribution"] == {"5": 1}
    assert clean["decision_precedes_announcement"] == 0


def test_the_report_is_report_only_and_cannot_reach_a_target(monkeypatch):
    """Make the report raise: targets must be unaffected, which is what
    'report only' has to mean."""
    index = _bdays("2020-01-01", "2020-06-30")
    frames = _frames(index)
    auctions = _monthly_auctions(index)
    before = mod.make_target_fn(frames, auction_dates=auctions, extra_sessions=[])
    expected = [before(_truncate(frames, date)) for date in index]

    def explode(*_args, **_kwargs):
        raise AssertionError("announcement_lead_report must not be on the "
                             "target path")

    monkeypatch.setattr(mod, "announcement_lead_report", explode)
    target = mod.make_target_fn(frames, auction_dates=auctions, extra_sessions=[])
    assert [target(_truncate(frames, date)) for date in index] == expected
    assert "TMV" in expected and "TMF" in expected


# ---------------------------------------------------------- the rule itself

def test_legs_fire_around_a_single_auction_and_stay_flat_elsewhere():
    index = _bdays("2020-01-01", "2020-06-30")
    frames = _frames(index)
    p = 40                                   # auction on session index[p]
    target = mod.make_target_fn(frames, auction_dates=[index[p]],
                                extra_sessions=[])
    # Decision on t targets execution session t+1.
    # Pre-leg: execution sessions p-4..p  <=> decisions p-5..p-1 -> TMV.
    for t in range(p - 5, p):
        assert target(_truncate(frames, index[t])) == "TMV", t
    # Post-leg: execution sessions p+1..p+5 <=> decisions p..p+4 -> TMF.
    for t in range(p, p + 5):
        assert target(_truncate(frames, index[t])) == "TMF", t
    # Outside: decision p-6 (executes p-5) and p+5 (executes p+6) -> flat.
    assert target(_truncate(frames, index[p - 6])) is None
    assert target(_truncate(frames, index[p + 5])) is None
    for t in (0, 10, p - 20, p + 30, len(index) - 1):
        assert target(_truncate(frames, index[t])) is None, t


def test_post_leg_of_the_earlier_auction_wins_an_overlap():
    index = _bdays("2020-01-01", "2020-06-30")
    frames = _frames(index)
    a, b = 40, 47                            # 7 sessions apart (never on the
    target = mod.make_target_fn(              # real schedule; exercises the rule)
        frames, auction_dates=[index[a], index[b]], extra_sessions=[])
    execution = mod.execution_targets(index, [index[a], index[b]])
    # A's post-leg 41..45 overlaps B's pre-leg 43..47 on 43..45 -> TMF.
    assert execution[a + 1:a + 6] == ["TMF"] * 5
    assert execution[46:48] == ["TMV", "TMV"]          # B's pre-leg remainder
    assert execution[48:53] == ["TMF"] * 5             # B's post-leg
    assert execution[a - 4:a + 1] == ["TMV"] * 5       # A's pre-leg
    # Same thing through the factory (decision t -> execution t+1).
    assert target(_truncate(frames, index[43 - 1])) == "TMF"
    assert target(_truncate(frames, index[46 - 1])) == "TMV"


def test_auction_on_a_non_session_day_maps_to_the_prior_session():
    index = _bdays("2020-01-01", "2020-06-30")
    # Drop one Wednesday to make it a non-session; auction dated that day.
    hole = index[50]
    calendar = index.drop(hole)
    frames = {key: frame.loc[calendar] for key, frame in _frames(index).items()}
    execution = mod.execution_targets(calendar, [hole])
    s_a = calendar.searchsorted(hole, side="right") - 1    # last session <= a
    assert calendar[s_a] < hole < calendar[s_a + 1]
    assert execution[s_a - 4:s_a + 1] == ["TMV"] * 5
    assert execution[s_a + 1:s_a + 6] == ["TMF"] * 5
    assert execution[s_a - 5] is None and execution[s_a + 6] is None
    target = mod.make_target_fn(frames, auction_dates=[hole], extra_sessions=[])
    assert target(_truncate(frames, calendar[s_a - 1])) == "TMV"
    assert target(_truncate(frames, calendar[s_a])) == "TMF"


def test_auctions_before_the_calendar_are_ignored_and_a_late_one_is_partial():
    index = _bdays("2020-01-01", "2020-03-31")
    execution = mod.execution_targets(
        index, [index[0] - pd.Timedelta(days=30), index[-2]])
    assert execution[:5] == [None] * 5
    # Auction on the second-to-last session: pre-leg present, post-leg cut.
    assert execution[-6:-1] == ["TMV"] * 5
    assert execution[-1] == "TMF"
    # Decision on the last session has no t+1 -> None.
    lookup = mod.decision_targets(index, [index[-2]])
    assert index[-1] not in lookup
    assert lookup[index[-2]] == "TMF"


def test_calendar_extends_past_the_frames_with_the_vendored_nyse_sessions():
    sessions = trading_calendar.load_default_sessions()
    early = _bdays("2025-01-02", pd.Timestamp(sessions[0]) - pd.Timedelta(days=1))
    late = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions
                             if s <= pd.Timestamp("2026-09-04").date()])
    index = early.append(late)
    frames = _frames(index)
    calendar = mod.build_session_calendar(frames)
    assert calendar.max() == pd.Timestamp(sessions[-1])
    assert pd.Timestamp("2026-09-07") not in calendar          # Labor Day
    # The 2026-09-10 auction's pre-leg is decided on real sessions the
    # frames do not yet have: 09-10 -> s(a); pre-leg 09-03,04,08,09,10.
    execution = mod.execution_targets(calendar, [pd.Timestamp("2026-09-10")])
    pos = {date: i for i, date in enumerate(calendar)}
    assert execution[pos[pd.Timestamp("2026-09-03")]] == "TMV"
    assert execution[pos[pd.Timestamp("2026-09-02")]] is None
    assert execution[pos[pd.Timestamp("2026-09-11")]] == "TMF"
    assert execution[pos[pd.Timestamp("2026-09-17")]] == "TMF"
    assert execution[pos[pd.Timestamp("2026-09-18")]] is None
    # Frames ending 09-04 -> the decision on 09-04 (execute 09-08) is TMV;
    # so is the decision on 09-02 (execute 09-03 = a-4); the decision on
    # 09-01 (execute 09-02 = a-5) is still flat.
    target = mod.make_target_fn(frames, auction_dates=[pd.Timestamp("2026-09-10")])
    assert target(_truncate(frames, pd.Timestamp("2026-09-04"))) == "TMV"
    assert target(_truncate(frames, pd.Timestamp("2026-09-02"))) == "TMV"
    assert target(_truncate(frames, pd.Timestamp("2026-09-01"))) is None


# ------------------------------------------------------------ look-ahead

def test_perturbing_bars_after_k_never_changes_targets_through_k():
    index = _bdays("2019-01-01", "2021-06-30")
    frames = _frames(index)
    auctions = _monthly_auctions(index)
    k = 400
    target = mod.make_target_fn(frames, auction_dates=auctions, extra_sessions=[])
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
    target_p = mod.make_target_fn(perturbed, auction_dates=auctions,
                                  extra_sessions=[])
    after = [target_p(_truncate(perturbed, index[j])) for j in range(k + 1)]
    assert before == after
    assert "TMV" in before and "TMF" in before


def test_target_reads_no_prices_at_all():
    """A pure calendar rule: zeroing every price leaves every target intact."""
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _frames(index)
    auctions = _monthly_auctions(index)
    target = mod.make_target_fn(frames, auction_dates=auctions, extra_sessions=[])
    dead = {key: frame.copy(deep=True) for key, frame in frames.items()}
    for frame in dead.values():
        frame.iloc[:, :] = 1e-9
    target_dead = mod.make_target_fn(dead, auction_dates=auctions,
                                     extra_sessions=[])
    for date in index[::7]:
        assert target(_truncate(frames, date)) == target_dead(_truncate(dead, date))


# ------------------------------------------------------------- end to end

def test_end_to_end_rotation_produces_two_five_session_segments_per_auction():
    index = _bdays("2019-01-01", "2021-06-30")         # ~650 bars
    frames = _frames(index)
    auctions = _monthly_auctions(index)
    target = mod.make_target_fn(frames, auction_dates=auctions, extra_sessions=[])
    segments, active = rotation_engine.run_two_asset_rotation(
        frames, target, cost_bps_by_asset={"TMF": 2.0, "TMV": 6.0})
    assert len(segments) >= 1
    assert set(segments["asset"]) == {"TMF", "TMV"}
    closed = segments.loc[segments["reason"] != "end_of_data"]
    assert (closed["held"] == 5).all()
    tmv = closed.loc[closed["asset"] == "TMV"]
    tmf = closed.loc[closed["asset"] == "TMF"]
    assert (tmv["reason"] == "switch").all()             # TMV -> TMF at a+1 open
    assert (tmf["reason"] == "to_flat").all()            # TMF -> flat at a+6 open
    pos = {date: i for i, date in enumerate(index)}
    auction_pos = {pos[a] for a in auctions}
    assert all(pos[d] + 4 in auction_pos for d in tmv["entry_date"])   # a-4
    assert all(pos[d] - 1 in auction_pos for d in tmf["entry_date"])   # a+1
    assert all(pos[d] - 6 in auction_pos for d in tmf["exit_date"])    # a+6
    # Two segments per auction (the final one may still be open).
    assert len(segments) >= 2 * len(auctions) - 1
    assert active.notna().sum() == segments["held"].sum()
