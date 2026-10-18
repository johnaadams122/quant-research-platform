"""Tests for src/mining/batch5/overnight_intraday_tilt_pooled_largecap.py.

Synthetic frames only (no network, deterministic).
"""
import importlib
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
from mining.backtest_engine import SignalResult  # noqa: E402

FAMILY_ID = "overnight_intraday_tilt_pooled_largecap"

mod = importlib.import_module(f"mining.batch5.{FAMILY_ID}")


def setup_function(_fn=None):
    mod.reset_diagnostics()


# ------------------------------------------------------------ synthetic data

def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _frame(open_: np.ndarray, close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _random_walk_frame(n_days: int, *, start: str = "2015-01-01",
                        seed: int = 0, drift: float = 0.0003,
                        overnight_bias: float = 0.0,
                        intraday_bias: float = 0.0) -> pd.DataFrame:
    """A synthetic OHLC frame with independently-perturbable overnight and
    intraday legs, so a caller can construct a name that is deliberately
    strong on one leg and weak on the other."""
    rng = np.random.default_rng(seed)
    index = _bdays(start, n_days)
    overnight = rng.normal(drift / 2 + overnight_bias, 0.004, n_days)
    intraday = rng.normal(drift / 2 + intraday_bias, 0.004, n_days)
    close = np.empty(n_days)
    open_ = np.empty(n_days)
    close[0] = 100.0
    open_[0] = 100.0
    for i in range(1, n_days):
        open_[i] = close[i - 1] * (1.0 + overnight[i])
        close[i] = open_[i] * (1.0 + intraday[i])
    return _frame(open_, close, index)


def _month_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    periods = index.to_period("M")
    is_last = np.append(periods[1:] != periods[:-1], True)
    return np.flatnonzero(is_last)


# ----------------------------------------------------------- contract fields

def test_contract_constants():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "pooled_single_asset"
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)
    assert not hasattr(mod, "make_target_fn")


# --------------------------------------------------------- trailing_signal

def test_trailing_signal_telescopes_exactly_to_the_21_day_return():
    """(1+ret_night_21d(t)) * (1+ret_day_21d(t)) == Close_t / Close_{t-21}
    exactly, since (1+overnight_i)(1+intraday_i) = Close_i/Close_{i-1}
    telescopes across the window in log space -- the free internal-
    consistency invariant the module's own docstring promises."""
    frame = _random_walk_frame(300, seed=1)
    signal = mod.trailing_signal(frame)
    close = frame["Close"].to_numpy()
    lookback = mod.LOOKBACK
    for t in range(lookback, len(frame)):
        if np.isnan(signal.iloc[t]):
            continue
        ret_night = None  # recovered algebraically below, not directly observable
    # Recompute ret_night/ret_day independently via plain compounding and
    # check the telescoping identity directly against the module's signal.
    log_overnight, log_intraday = mod._log_components(frame)
    roll_night = log_overnight.rolling(lookback, min_periods=lookback).sum()
    roll_day = log_intraday.rolling(lookback, min_periods=lookback).sum()
    ret_night = np.expm1(roll_night)
    ret_day = np.expm1(roll_day)
    valid = signal.notna()
    assert valid.sum() > 0
    lhs = (1.0 + ret_night[valid]) * (1.0 + ret_day[valid])
    rhs = close[valid.to_numpy()] / np.roll(close, lookback)[valid.to_numpy()]
    np.testing.assert_allclose(lhs.to_numpy(), rhs, rtol=1e-9)
    np.testing.assert_allclose((ret_night - ret_day)[valid].to_numpy(),
                                signal[valid].to_numpy(), rtol=1e-12)


def test_trailing_signal_undefined_before_21_valid_bars():
    frame = _random_walk_frame(60, seed=2)
    signal = mod.trailing_signal(frame)
    # Bar 0 has no prior close, so its overnight/intraday components are
    # NaN -- a 21-wide trailing window only accumulates 21 VALID components
    # once it stops covering bar 0, i.e. starting at bar 21 (bars 1..21).
    assert signal.iloc[:21].isna().all()
    assert signal.iloc[21:].notna().all()


def test_trailing_signal_favors_a_name_with_a_strong_overnight_bias():
    strong_overnight = _random_walk_frame(
        100, seed=3, overnight_bias=0.01, intraday_bias=-0.01)
    strong_intraday = _random_walk_frame(
        100, seed=3, overnight_bias=-0.01, intraday_bias=0.01)
    sig_night = mod.trailing_signal(strong_overnight).iloc[-1]
    sig_day = mod.trailing_signal(strong_intraday).iloc[-1]
    assert sig_night > sig_day


# --------------------------------------------------------- build_target_schedule

def _three_ticker_universe(seed=10):
    """Three tickers with deliberately different overnight/intraday tilts,
    used to test cross-sectional ranking directly."""
    frames = {
        "HI": _random_walk_frame(400, seed=seed, overnight_bias=0.015,
                                  intraday_bias=-0.015),
        "MID": _random_walk_frame(400, seed=seed + 1),
        "LO": _random_walk_frame(400, seed=seed + 2, overnight_bias=-0.015,
                                  intraday_bias=0.015),
    }
    return frames


def test_schedule_ranks_the_strongest_overnight_tilt_on_top_when_threshold_is_low(monkeypatch):
    frames = _three_ticker_universe()
    monkeypatch.setattr(mod, "TRADED", list(frames.keys()))
    monkeypatch.setattr(mod, "MIN_ELIGIBLE_NAMES_TO_TRADE", 3)
    monkeypatch.setattr(mod, "TOP_DECILE_FRACTION", 1.0 / 3.0)
    cutoff = frames["HI"].index[-1]
    schedule = mod.build_target_schedule(frames, cutoff)
    assert schedule
    for (year, month), members in schedule.items():
        if members:
            assert members == frozenset({"HI"})


def test_schedule_empty_when_fewer_than_min_eligible_names(monkeypatch):
    frames = _three_ticker_universe()
    monkeypatch.setattr(mod, "TRADED", list(frames.keys()))
    monkeypatch.setattr(mod, "MIN_ELIGIBLE_NAMES_TO_TRADE", 100)
    cutoff = frames["HI"].index[-1]
    schedule = mod.build_target_schedule(frames, cutoff)
    assert schedule
    assert all(members == frozenset() for members in schedule.values())


def test_schedule_skips_a_missing_or_too_short_member_gracefully(monkeypatch):
    frames = _three_ticker_universe()
    frames["NEW"] = _random_walk_frame(5, seed=99)  # far too short to ever be eligible
    monkeypatch.setattr(mod, "TRADED", list(frames.keys()) + ["ABSENT"])
    monkeypatch.setattr(mod, "MIN_ELIGIBLE_NAMES_TO_TRADE", 3)
    cutoff = frames["HI"].index[-1]
    # Must not raise despite "ABSENT" missing from frames and "NEW" too short.
    schedule = mod.build_target_schedule(frames, cutoff)
    assert schedule
    for members in schedule.values():
        assert "ABSENT" not in members
        assert "NEW" not in members


# --------------------------------------------------------------- decision_dates

def test_decision_dates_entry_on_member_month_ends_exit_bar_after_non_member():
    frame = _random_walk_frame(200, seed=5)
    dates = frame.index
    # Use the module's OWN month-end determination (calendar-aware tail-bar
    # resolution), not the naive always-True-at-the-final-bar helper used
    # elsewhere in this file for period-boundary bookkeeping -- decision_
    # dates() calls mod.month_end_flags internally, and a frame dated 2015
    # falls outside the vendored 2026-2028 NYSE calendar, so (unlike large_
    # cap_single_stock_pooled_tsmom's own calendar-agnostic decision_dates)
    # this module's tail bar is correctly NOT treated as a month end here.
    me = np.flatnonzero(mod.month_end_flags(dates))
    # A hand-built schedule: member in odd-indexed month-ends, not in even.
    schedule = {}
    for k, pos in enumerate(me):
        d = dates[pos]
        schedule[(d.year, d.month)] = frozenset({"AAA"}) if k % 2 else frozenset()
    entry_dates, exit_dates = mod.decision_dates(frame, "AAA", schedule)
    member_positions = {pos for k, pos in enumerate(me) if k % 2}
    nonmember_positions = {pos for k, pos in enumerate(me) if not k % 2}
    assert set(dates.get_indexer(entry_dates)) == member_positions
    expected_exit = {pos + 1 for pos in nonmember_positions if pos + 1 < len(dates)}
    assert set(dates.get_indexer(exit_dates)) == expected_exit


def test_decision_dates_empty_schedule_means_never_long():
    frame = _random_walk_frame(150, seed=6)
    entry_dates, exit_dates = mod.decision_dates(frame, "AAA", {})
    assert len(entry_dates) == 0


# ------------------------------------------------------------ make_signal_fn

def test_make_signal_fn_rejects_non_pool_ticker():
    frame = _random_walk_frame(50, seed=7)
    with pytest.raises(ValueError):
        mod.make_signal_fn("NOT_IN_POOL", frame.index, {"NOT_IN_POOL": frame})


def test_make_signal_fn_uses_the_shared_cross_sectional_schedule(monkeypatch):
    frames = _three_ticker_universe()
    monkeypatch.setattr(mod, "TRADED", list(frames.keys()))
    monkeypatch.setattr(mod, "MIN_ELIGIBLE_NAMES_TO_TRADE", 3)
    monkeypatch.setattr(mod, "TOP_DECILE_FRACTION", 1.0 / 3.0)
    index = frames["HI"].index
    fn_hi = mod.make_signal_fn("HI", index, frames)
    fn_lo = mod.make_signal_fn("LO", index, frames)
    me = _month_end_positions(index)
    seasoned = [pos for pos in me if pos >= mod.LOOKBACK][2]
    assert fn_hi(frames["HI"].iloc[:seasoned + 1], seasoned).entry is True
    assert fn_lo(frames["LO"].iloc[:seasoned + 1], seasoned).entry is False


def test_schedule_cache_is_keyed_by_frames_identity_not_just_cutoff(monkeypatch):
    """Two distinct frames dicts sharing the same cutoff timestamp must not
    cross-contaminate -- the cache's identity check, not just a cutoff
    match, is what this test exercises."""
    frames_a = _three_ticker_universe(seed=20)
    frames_b = _three_ticker_universe(seed=20)  # different dict object, same content by construction
    # Make frames_b's "HI" actually the weakest instead, so a wrongly-shared
    # cache would misclassify it.
    frames_b["HI"], frames_b["LO"] = frames_b["LO"], frames_b["HI"]
    monkeypatch.setattr(mod, "TRADED", ["HI", "MID", "LO"])
    monkeypatch.setattr(mod, "MIN_ELIGIBLE_NAMES_TO_TRADE", 3)
    monkeypatch.setattr(mod, "TOP_DECILE_FRACTION", 1.0 / 3.0)
    index = frames_a["HI"].index
    assert index.equals(frames_b["HI"].index)

    fn_a_hi = mod.make_signal_fn("HI", index, frames_a)
    fn_b_hi = mod.make_signal_fn("HI", index, frames_b)
    me = _month_end_positions(index)
    seasoned = [pos for pos in me if pos >= mod.LOOKBACK][2]
    # frames_a's real "HI" (strong overnight tilt) should be selected;
    # frames_b's "HI" (originally the weak "LO" series) should not be --
    # if the cache ignored object identity, both calls would share
    # frames_a's schedule and this would wrongly pass for frames_b too.
    assert fn_a_hi(frames_a["HI"].iloc[:seasoned + 1], seasoned).entry is True
    assert fn_b_hi(frames_b["HI"].iloc[:seasoned + 1], seasoned).entry is False


# ----------------------------------------------------------- end to end

def test_end_to_end_backtest_produces_a_real_holding_period(monkeypatch):
    frames = _three_ticker_universe(seed=30)
    monkeypatch.setattr(mod, "TRADED", list(frames.keys()))
    monkeypatch.setattr(mod, "MIN_ELIGIBLE_NAMES_TO_TRADE", 3)
    monkeypatch.setattr(mod, "TOP_DECILE_FRACTION", 1.0 / 3.0)
    index = frames["HI"].index
    fn = mod.make_signal_fn("HI", index, frames)
    trades, in_market = backtest_engine.run_backtest(frames["HI"], fn)
    assert len(trades) >= 1
    assert in_market.sum() >= 1


def test_look_ahead_probe_perturbing_future_bars_leaves_past_signals_unchanged(monkeypatch):
    frames = _three_ticker_universe(seed=40)
    monkeypatch.setattr(mod, "TRADED", list(frames.keys()))
    monkeypatch.setattr(mod, "MIN_ELIGIBLE_NAMES_TO_TRADE", 3)
    monkeypatch.setattr(mod, "TOP_DECILE_FRACTION", 1.0 / 3.0)
    index = frames["HI"].index
    k = 250
    truncated_index = index[:k + 1]

    def _signals(fr):
        fn = mod.make_signal_fn("HI", truncated_index, fr)
        bucket = fr["HI"].loc[truncated_index]
        return [fn(bucket.iloc[:i + 1], i) for i in range(len(bucket))]

    before = _signals(frames)
    mod.reset_diagnostics()
    perturbed = {k_: v.copy() for k_, v in frames.items()}
    tail = index[k + 1:]
    for name in perturbed:
        perturbed[name].loc[tail, ["Open", "High", "Low", "Close"]] *= 1.5
    after = _signals(perturbed)
    assert before == after


# ------------------------------------------------------------- driver
