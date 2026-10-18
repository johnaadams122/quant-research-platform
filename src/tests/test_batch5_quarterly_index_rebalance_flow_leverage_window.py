"""Tests for src/mining/batch5/quarterly_index_rebalance_flow_leverage_window.py.

Synthetic frames only (no network). The candidate JSON is read from the
real data/mining/catalog.json for the PARAMETERS byte-identity check
(this family has no dedicated curation record -- it was added
directly to the catalog, mirroring the convention every later single-
candidate module in src/mining/batch5/ already follows, e.g.
fed_net_liquidity_regime / money_market_fund_flow_riskoff).
"""
import importlib
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
from mining.batch5 import quarterly_index_rebalance_flow_leverage_window as mod

FAMILY_ID = "quarterly_index_rebalance_flow_leverage_window"
CATALOG = os.path.join(_ROOT, "data", "mining", "catalog.json")
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
        return {"SPXL": _random_frame(index, 1), "SPXS": _random_frame(index, 2),
                "IEF": _random_frame(index, 3), "SPY": _random_frame(index, 4)}
    return {"SPXL": _flat_frame(index, 30.0), "SPXS": _flat_frame(index, 20.0),
            "IEF": _flat_frame(index, 80.0), "SPY": _flat_frame(index, 100.0)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _quarter_sessions(index: pd.DatetimeIndex, year: int, quarter: int) -> list:
    months = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}[quarter]
    return [ts for ts in index
            if ts.year == year and months[0] <= ts.month <= months[1]]


def _set_close(frame: pd.DataFrame, date: pd.Timestamp, close: float) -> None:
    frame.loc[date, "Close"] = close
    frame.loc[date, "High"] = max(frame.loc[date, "High"], close)
    frame.loc[date, "Low"] = min(frame.loc[date, "Low"], close)


# ------------------------------------------------------------- contract

def _candidate() -> dict:
    with open(CATALOG, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload if c["family_id"] == FAMILY_ID]
    assert len(matches) == 1, (
        f"expected exactly one catalog entry for {FAMILY_ID}, found "
        f"{len(matches)}")
    return matches[0]


def test_parameters_are_byte_identical_to_the_catalog_entry():
    candidate = _candidate()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(candidate["parameters"], sort_keys=True))
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == candidate["shape"] == "rotation_two_asset"
    assert mod.TRADED == candidate["tickers"] == ["SPXL", "SPXS"]
    assert set(mod.SIGNAL) == set(candidate["signal_tickers"]) == {"IEF", "SPY"}
    assert mod.BENCHMARK == candidate["benchmark_ticker"] == "SPY"
    assert mod.SNAPSHOTS == {}


def test_module_passes_the_driver_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module, so the result depends only
    on this module and not on whatever else is in the real batch5 dir."""
    import run_batch5_mining as batch5
    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == ["SPXL", "SPXS"]
    assert set(spec["signal"]) == {"IEF", "SPY"}
    assert batch5.parameter_drift(spec["parameters"],
                                  _candidate()["parameters"]) is None


# --------------------------------------------------------- calendar roles

def test_session_roles_label_the_last_seven_sessions_of_each_quarter():
    index = _bdays("2019-01-01", "2020-12-31")
    roles = mod.session_roles(index)
    q1_2020 = _quarter_sessions(index, 2020, 1)
    q4_2019 = _quarter_sessions(index, 2019, 4)
    t = q1_2020[-1]
    for k in range(7):
        role = roles[q1_2020[-1 - k]]
        assert role.offset == k
        assert role.quarter == pd.Period(t, freq="Q")
        assert role.decision_date == q1_2020[-7]           # Q-6
        assert role.prev_quarter_end == q4_2019[-1]         # prev_Q
    assert roles[q1_2020[-8]].offset == 7                   # Q-7: outside window
    # The very first quarter in this calendar has no previous quarter.
    q1_2019 = _quarter_sessions(index, 2019, 1)
    assert roles[q1_2019[-1]].prev_quarter_end is None


def test_calendar_extends_past_the_frames_with_the_vendored_nyse_sessions():
    """Frames ending mid-quarter must still see the quarter's true last
    session, because the XNYS calendar is known in advance
    (trading_calendar)."""
    sessions = trading_calendar.load_default_sessions()
    import datetime
    end = datetime.date(2026, 9, 4)
    early = pd.bdate_range("2025-01-02",
                           pd.Timestamp(sessions[0]) - pd.Timedelta(days=1))
    late = pd.DatetimeIndex([pd.Timestamp(s) for s in sessions if s <= end])
    index = early.append(late)
    frames = _frames(index)
    calendar = mod.build_session_calendar(frames)
    q3_2026 = [ts for ts in calendar if ts.year == 2026 and 7 <= ts.month <= 9]
    assert q3_2026[-1] == pd.Timestamp("2026-09-30")
    roles = mod.session_roles(calendar)
    role = roles[pd.Timestamp("2026-09-04")]
    assert role.offset > 6                                  # not yet in the window
    assert calendar.max() == pd.Timestamp(sessions[-1])


# ---------------------------------------------------------- the rule itself

def _prepare_signal_quarter(frames: dict, index: pd.DatetimeIndex, year: int,
                            quarter: int, *, spy_ret: float, ief_ret: float):
    """Set SPY/IEF closes from prev_Q to Q-6 so the quarter-to-date returns
    at Q-6 are exactly spy_ret / ief_ret (flat everywhere else)."""
    sessions = _quarter_sessions(index, year, quarter)
    q6 = sessions[-7]
    for date in sessions[:len(sessions) - 6]:
        _set_close(frames["SPY"], date, 100.0 * (1.0 + spy_ret))
        _set_close(frames["IEF"], date, 80.0 * (1.0 + ief_ret))
    return sessions, q6


def test_equities_overweight_targets_spxs_for_the_five_hold_sessions():
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _frames(index)
    sessions, q6 = _prepare_signal_quarter(frames, index, 2020, 2,
                                           spy_ret=0.06, ief_ret=0.01)
    target = mod.make_target_fn(frames)
    # Q-7 and earlier: flat.
    assert target(_truncate(frames, sessions[-8])) is None
    # Q-6 decision through Q-2 hold: SPXS (stocks beat bonds quarter-to-date).
    for date in sessions[-7:-2]:
        assert target(_truncate(frames, date)) == "SPXS", date
    # Q-1 clears the target (position exits at Q's open); Q itself flat.
    assert target(_truncate(frames, sessions[-2])) is None
    assert target(_truncate(frames, sessions[-1])) is None


def test_equities_underweight_targets_spxl():
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _frames(index)
    sessions, _ = _prepare_signal_quarter(frames, index, 2020, 3,
                                          spy_ret=-0.04, ief_ret=0.02)
    target = mod.make_target_fn(frames)
    for date in sessions[-7:-2]:
        assert target(_truncate(frames, date)) == "SPXL", date
    assert target(_truncate(frames, sessions[-2])) is None


def test_zero_deviation_and_missing_prev_quarter_close_stay_flat():
    index = _bdays("2019-01-01", "2020-12-31")
    frames = _frames(index)                        # flat: R_e == R_b == 0
    target = mod.make_target_fn(frames)
    q3 = _quarter_sessions(index, 2020, 3)
    for date in q3[-7:-2]:
        assert target(_truncate(frames, date)) is None
    # A frame that starts inside the quarter has no prev_Q close -> flat,
    # even though the quarter-to-date deviation would otherwise be positive.
    frames2 = _frames(index)
    sessions, _ = _prepare_signal_quarter(frames2, index, 2020, 4,
                                          spy_ret=0.05, ief_ret=0.0)
    target2 = mod.make_target_fn(frames2)
    late_start = {key: frame.loc[frame.index >= sessions[1]]
                  for key, frame in frames2.items()}
    assert target2(_truncate(late_start, sessions[-7])) is None
    # Sanity: with prev_Q present the same quarter targets SPXS.
    assert target2(_truncate(frames2, sessions[-7])) == "SPXS"


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
    index = _bdays("2017-01-01", "2021-06-30")
    frames = _frames(index, random=True)
    k = 700
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
    index = _bdays("2017-01-01", "2020-12-31")
    frames = _frames(index, random=True)
    target = mod.make_target_fn(frames)
    dec = _quarter_sessions(index, 2020, 3)[-7]
    truncated = _truncate(frames, dec)
    full = target(truncated)
    n = len(truncated["SPY"])
    for frame in truncated.values():
        frame.iloc[:n - mod.TAIL_SESSIONS, :] = 1e-9
    assert target(truncated) == full


# ------------------------------------------------------------- end to end

def test_end_to_end_rotation_produces_five_session_segments():
    index = _bdays("2015-01-01", "2021-06-30")         # ~1600 bars, ~26 quarters
    frames = _frames(index, random=True)
    target = mod.make_target_fn(frames)
    segments, active = rotation_engine.run_two_asset_rotation(
        {"SPXL": frames["SPXL"], "SPXS": frames["SPXS"]}, target,
        cost_bps_by_asset={"SPXL": 2.5, "SPXS": 4.0},
        signal_frames={"IEF": frames["IEF"], "SPY": frames["SPY"]})
    assert len(segments) >= 1
    assert set(segments["asset"]).issubset({"SPXL", "SPXS"})
    closed = segments.loc[segments["reason"] != "end_of_data"]
    assert (closed["held"] == 5).all()
    assert (closed["reason"] == "to_flat").all()
    # Every entry is a Q-5 session and every exit the quarter's last session.
    roles = mod.session_roles(mod.build_session_calendar(frames))
    assert all(roles[ts].offset == 5 for ts in closed["entry_date"])
    assert all(roles[ts].offset == 0 for ts in closed["exit_date"])
    # Roughly one segment per quarter over ~26 quarters, minus flat ones.
    assert len(segments) >= 5
    assert (active.notna().sum()) == closed["held"].sum() + (
        segments.loc[segments["reason"] == "end_of_data", "held"].sum())
