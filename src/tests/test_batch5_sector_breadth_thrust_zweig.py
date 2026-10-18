"""Tests for src/mining/batch5/sector_breadth_thrust_zweig.py.

Synthetic snapshots only (no network, deterministic, fast): a
panel.AsOfSeries is built directly from hand-designed net-advance-ratio
values.
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

from mining import backtest_engine, panel  # noqa: E402
from mining.batch5 import sector_breadth_thrust_zweig as mod  # noqa: E402

# One washout/recovery episode: three sessions of every member declining
# (NAR -1, share 0) followed by every member advancing (NAR +1, share 1).
# From a steady z of 0.5 the EMA runs 0.409, 0.335, 0.274, then 0.406,
# 0.514, 0.602, 0.675 -- so the thrust (z > 0.615 with a value < 0.40 among
# the ten prior sessions) fires on the SIXTH session after the washout
# starts, and keeps firing while a sub-0.40 value remains in the window.
_EPISODE_WASHOUT = 3
_EPISODE_RECOVERY = 4
_FIRST_FIRE_OFFSET = 6


# ------------------------------------------------------------ synthetic data

def _calendar(periods=600, start="2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _ohlc(index: pd.DatetimeIndex, seed=3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _nar_with_episodes(n: int, starts) -> np.ndarray:
    """Flat NAR (share 0.5) with a washout/recovery episode at each start."""
    nar = np.zeros(n)
    for start in starts:
        nar[start:start + _EPISODE_WASHOUT] = -1.0
        nar[start + _EPISODE_WASHOUT:
            start + _EPISODE_WASHOUT + _EPISODE_RECOVERY] = 1.0
    return nar


def _snapshot(index: pd.DatetimeIndex, nar, unknown=()) -> mod.BreadthSnapshot:
    series = panel.AsOfSeries(dates=index, values=np.asarray(nar, dtype=float),
                              indicator="net_advance_ratio",
                              content_sha256="synthetic")
    return mod.BreadthSnapshot(
        dates=pd.DatetimeIndex(index), series=series,
        unknown_sessions=frozenset(pd.Timestamp(d) for d in unknown))


def _reference_thrust_positions(nar) -> list:
    """Independent, plain-loop statement of the family's thrust test."""
    share = (1.0 + np.asarray(nar, dtype=float)) / 2.0
    z = [share[0]]
    for value in share[1:]:
        z.append(2.0 / 11.0 * value + 9.0 / 11.0 * z[-1])
    out = []
    for t in range(len(z)):
        if t < 11:
            continue
        if z[t] > 0.615 and min(z[t - 10:t]) < 0.40:
            out.append(t)
    return out


def _record(signal_fn, frame: pd.DataFrame, through: int) -> list:
    return [(signal_fn(frame.iloc[:i + 1], i).entry,
             signal_fn(frame.iloc[:i + 1], i).exit) for i in range(through + 1)]


# ------------------------------------------------------------------- rule

def test_thrust_fires_after_washout_then_recovery_and_not_when_flat():
    index = _calendar()
    nar = _nar_with_episodes(len(index), [50])
    thrusts = mod.thrust_dates(_snapshot(index, nar), index[-1])
    positions = [index.get_loc(d) for d in thrusts]
    assert positions[0] == 50 + _FIRST_FIRE_OFFSET
    assert positions == _reference_thrust_positions(nar)
    # Flat panel: never fires.
    assert len(mod.thrust_dates(_snapshot(index, np.zeros(len(index))),
                                index[-1])) == 0


def test_slow_recovery_over_more_than_ten_sessions_does_not_fire():
    index = _calendar()
    nar = np.zeros(len(index))
    nar[50:53] = -1.0
    nar[53:] = 0.24  # share 0.62: crosses 0.615 only ~22 sessions later
    thrusts = mod.thrust_dates(_snapshot(index, nar), index[-1])
    assert len(thrusts) == 0
    assert _reference_thrust_positions(nar) == []


def test_eleven_prior_defined_sessions_rule_is_literal():
    index = _calendar()
    # Seeded at share 0 (washout on the very first sessions), then twelve
    # all-advance sessions: z = 0, 0, 0, .18, .33, .45, .55, .63, ... so the
    # raw crossing is at position 7 and a sub-0.40 value stays inside the
    # ten-session window through position 14.
    nar = np.zeros(len(index))
    nar[0:3] = -1.0
    nar[3:15] = 1.0
    share = (1.0 + nar) / 2.0
    z = [share[0]]
    for value in share[1:]:
        z.append(2.0 / 11.0 * value + 9.0 / 11.0 * z[-1])
    raw = [t for t in range(10, len(z))
           if z[t] > 0.615 and min(z[t - 10:t]) < 0.40]
    assert raw[0] == 10  # the ten-value window alone would admit position 10
    thrusts = mod.thrust_dates(_snapshot(index, nar), index[-1])
    positions = [index.get_loc(d) for d in thrusts]
    # The rule's literal ">= 11 prior defined sessions" admits nothing before 11.
    assert positions[0] == 11
    assert positions == _reference_thrust_positions(nar)


def test_hold_schedule_enters_on_thrust_bar_and_exits_63_sessions_later():
    index = _calendar()
    nar = _nar_with_episodes(len(index), [50])
    entry, exit_ = mod.hold_schedule(
        mod.thrust_dates(_snapshot(index, nar), index[-1]), index)
    t = 50 + _FIRST_FIRE_OFFSET
    assert np.flatnonzero(entry).tolist() == [t]
    assert np.flatnonzero(exit_).tolist() == [t + 63]


def test_thrust_during_hold_is_ignored_and_reentry_allowed_after_exit():
    index = _calendar()
    # Episodes at 50 (fires 56, hold through 119), 86 (inside the hold) and
    # 200 (after the hold): exactly two entries.
    nar = _nar_with_episodes(len(index), [50, 86, 200])
    thrusts = mod.thrust_dates(_snapshot(index, nar), index[-1])
    assert index[86 + _FIRST_FIRE_OFFSET] in thrusts  # the raw thrust exists
    entry, exit_ = mod.hold_schedule(thrusts, index)
    assert np.flatnonzero(entry).tolist() == [56, 206]
    assert np.flatnonzero(exit_).tolist() == [119, 269]


def test_thrust_on_the_exit_session_itself_is_ignored():
    index = _calendar()
    # First hold: 56 -> exit 119. Second episode at 113 fires first at 119
    # (the exit session, ignored) and again at 120 (flat again, acted on).
    nar = _nar_with_episodes(len(index), [50, 113])
    thrusts = mod.thrust_dates(_snapshot(index, nar), index[-1])
    assert index[119] in thrusts and index[120] in thrusts
    entry, exit_ = mod.hold_schedule(thrusts, index)
    assert not entry[119] and exit_[119]
    assert entry[120]
    assert np.flatnonzero(entry).tolist() == [56, 120]


def test_thrust_whose_fill_bar_is_beyond_the_bucket_produces_no_entry():
    index = _calendar()
    nar = _nar_with_episodes(len(index), [len(index) - 1 - _FIRST_FIRE_OFFSET])
    thrusts = mod.thrust_dates(_snapshot(index, nar), index[-1])
    assert index[-1] in thrusts
    entry, exit_ = mod.hold_schedule(thrusts, index)
    assert not entry.any() and not exit_.any()


def test_unknown_session_inside_span_is_refused_but_outside_is_not():
    index = _calendar()
    nar = _nar_with_episodes(len(index), [50])
    hole_inside = _snapshot(index, nar, unknown=[index[300]])
    with pytest.raises(ValueError, match="unknown session"):
        mod.thrust_dates(hole_inside, index[-1])
    # A hole dated after the consulted span is not inside it.
    assert len(mod.thrust_dates(hole_inside, index[250])) > 0
    hole_after_end = _snapshot(index, nar,
                               unknown=[index[-1] + pd.Timedelta(days=1)])
    assert len(mod.thrust_dates(hole_after_end, index[-1])) > 0


# ------------------------------------------------------------- look-ahead

def test_look_ahead_probe_perturbing_bars_after_k_leaves_signals_through_k():
    index = _calendar()
    rng = np.random.RandomState(11)
    nar = rng.uniform(-1.0, 1.0, len(index))
    nar[100:103] = -1.0
    nar[103:110] = 1.0  # guarantee at least one thrust well before k
    frame = _ohlc(index)
    frames = {"SPY": frame}
    k = 300
    base = mod.make_signal_fn("SPY", index, frames, snapshot=_snapshot(index, nar))
    before = _record(base, frame, k)
    assert any(e for e, _ in before), "probe needs at least one entry before k"

    perturbed_nar = nar.copy()
    perturbed_nar[k + 1:] = rng.uniform(-1.0, 1.0, len(index) - k - 1)
    perturbed_nar[k + 1:k + 4] = -1.0
    perturbed_nar[k + 4:k + 12] = 1.0
    perturbed_frame = frame.copy()
    perturbed_frame.iloc[k + 1:, perturbed_frame.columns.get_loc("Close")] *= 1.3
    perturbed_frame.iloc[k + 1:, perturbed_frame.columns.get_loc("High")] *= 1.4
    perturbed_frame.iloc[k + 1:, perturbed_frame.columns.get_loc("Low")] *= 0.9
    rebuilt = mod.make_signal_fn("SPY", index, {"SPY": perturbed_frame},
                                 snapshot=_snapshot(index, perturbed_nar))
    after = _record(rebuilt, perturbed_frame, k)
    assert after == before
    # And the perturbation was real: something after k differs.
    full_before = _record(base, frame, len(index) - 1)
    full_after = _record(rebuilt, perturbed_frame, len(index) - 1)
    assert full_before != full_after


def test_holdout_blind_shorter_bucket_matches_longer_one_through_its_end():
    index = _calendar()
    rng = np.random.RandomState(5)
    nar = rng.uniform(-1.0, 1.0, len(index))
    frame = _ohlc(index)
    snapshot = _snapshot(index, nar)
    k = 400
    long_fn = mod.make_signal_fn("SPY", index, {"SPY": frame}, snapshot=snapshot)
    short_index = index[:k + 1]
    short_fn = mod.make_signal_fn("SPY", short_index,
                                  {"SPY": frame.iloc[:k + 1]}, snapshot=snapshot)
    assert _record(short_fn, frame.iloc[:k + 1], k) == _record(long_fn, frame, k)


# ------------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_63_session_trades():
    index = _calendar()
    nar = _nar_with_episodes(len(index), [50, 200, 400])
    frame = _ohlc(index)
    signal_fn = mod.make_signal_fn("SPY", index, {"SPY": frame},
                                   snapshot=_snapshot(index, nar))
    trades, in_market = backtest_engine.run_backtest(frame, signal_fn)
    assert len(trades) == 3
    for start, (_, trade) in zip((50, 200, 400), trades.iterrows()):
        t = start + _FIRST_FIRE_OFFSET
        assert trade["entry_date"] == index[t + 1]
        assert trade["exit_date"] == index[t + 63]
        assert trade["held"] == 62  # 63 sessions in position
        assert trade["reason"] == "signal_exit"
        assert trade["entry_px"] == frame["Open"].iloc[t + 1]
        assert trade["exit_px"] == frame["Close"].iloc[t + 63]
    assert int(in_market.sum()) == 3 * 63


def test_factory_input_validation():
    index = _calendar()
    frame = _ohlc(index)
    snapshot = _snapshot(index, np.zeros(len(index)))
    with pytest.raises(ValueError, match="expects ticker"):
        mod.make_signal_fn("QQQ", index, {"SPY": frame, "QQQ": frame},
                           snapshot=snapshot)
    with pytest.raises(ValueError, match="frames lacks"):
        mod.make_signal_fn("SPY", index, {}, snapshot=snapshot)
    signal_fn = mod.make_signal_fn("SPY", index, {"SPY": frame}, snapshot=snapshot)
    with pytest.raises(ValueError, match="misaligned"):
        signal_fn(frame.iloc[:10], 12)
