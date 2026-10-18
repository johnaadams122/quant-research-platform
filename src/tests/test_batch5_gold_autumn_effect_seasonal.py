"""Tests for src/mining/batch5/gold_autumn_effect_seasonal.py.

Synthetic data only (no network), deterministic, fast. The PARAMETERS
drift check here compares against a FROZEN literal copy of the
pre-registered catalog record -- same convention as
test_batch5_gasoline_driving_season_seasonal.py.
"""
import json
import os
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

from mining.backtest_engine import SignalResult, run_backtest  # noqa: E402
from mining.batch5 import gold_autumn_effect_seasonal as mod  # noqa: E402

# Frozen copy of this family's pre-registered catalog "parameters" (must
# match PARAMETERS byte-for-byte as JSON).
_CATALOG_PARAMETERS = {
    "entry_month": 9,
    "entry_day": 1,
    "exit_month": 11,
    "exit_day": 30,
    "rebalance": "annual",
}


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2022-01-03", periods=1200) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _gld_close(index: pd.DatetimeIndex, seed: int = 5) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 120.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 5) -> dict:
    return {"GLD": _ohlc_from_close(_gld_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "gold_autumn_effect_seasonal"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["GLD"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "GLD"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_frozen_catalog_entry():
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(_CATALOG_PARAMETERS, sort_keys=True))


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["GLD"]
    assert spec["signal"] == []


# ------------------------------------------------------------ in_season_mask

def test_in_season_mask_hand_verified_boundaries():
    # 2025: entry_target=2025-09-01 (a Monday), exit_target=2025-11-30 (a
    # Sunday). 2026: entry_target=2026-09-01 (a Tuesday),
    # exit_target=2026-11-30 (a Monday). Both boundaries are LITERAL
    # calendar dates (no "nearest weekday"/"last Monday" shift rule) --
    # in_season_mask is a pure elementwise predicate and does not care
    # whether the date given is itself a trading day.
    dates = pd.DatetimeIndex([
        "2025-08-31", "2025-09-01", "2025-09-02",   # around 2025 entry
        "2025-11-29", "2025-11-30", "2025-12-01",   # around 2025 exit
        "2026-08-31", "2026-09-01",                 # around 2026 entry
        "2026-11-29", "2026-11-30", "2026-12-01",   # around 2026 exit
    ])
    expected = np.array([
        False, True, True,      # before Sep1 / on Sep1 / after Sep1
        True, True, False,      # before exit / on exit (inclusive) / after
        False, True,
        True, True, False,
    ])
    result = mod.in_season_mask(dates)
    assert result.tolist() == expected.tolist()


def test_in_season_mask_empty():
    assert mod.in_season_mask(pd.DatetimeIndex([])).tolist() == []


def test_in_season_mask_never_crosses_year_boundary():
    # Dec, Jan, Feb, Aug are always off-season; only Sep-Nov is in-season.
    dates = pd.DatetimeIndex([
        "2024-12-15", "2025-01-15", "2025-02-15", "2025-08-15",
        "2025-09-15", "2025-10-15", "2025-11-15",
    ])
    expected = [False, False, False, False, True, True, True]
    assert mod.in_season_mask(dates).tolist() == expected


# --------------------------------------------------------- season_decisions

def test_season_decisions_entry_one_bar_early_exit_on_last_in_season_bar():
    # Hand-verified fixture spanning one full 2025 season transition on
    # each side. entry_target(2025)=2025-09-01 (Mon) so the first
    # in-season trading day IS 2025-09-01 itself; exit_target(2025)=
    # 2025-11-30 (Sun) so the last in-season trading day is 2025-11-28
    # (Fri).
    dates = pd.DatetimeIndex([
        "2025-08-28",  # off-season
        "2025-08-29",  # off-season (day before season starts)
        "2025-09-01",  # in-season (first day)
        "2025-09-02",  # in-season
        "2025-11-26",  # in-season
        "2025-11-28",  # in-season (last day, <= exit_target 2025-11-30)
        "2025-12-01",  # off-season (first day after exit_target)
        "2025-12-02",  # off-season
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    expected_entry = [False, True, True, True, True, False, False, False]
    expected_exit = [False, False, False, False, False, True, False, False]
    assert entry_s.tolist() == expected_entry
    assert exit_s.tolist() == expected_exit


def test_no_immediate_reentry_the_session_after_exit():
    # Same guard as the gasoline family's (interpretation choice 2): entry[t]
    # must be in_season[t+1], never in_season[t] directly, or the exit
    # bar itself (last in-season day) would re-raise "enter" and cause an
    # immediate re-entry the very next session.
    dates = pd.DatetimeIndex([
        "2025-11-24", "2025-11-26", "2025-11-28",  # tail of 2025 season
        "2025-12-01", "2025-12-02", "2025-12-03",  # first sessions after exit
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    exit_idx = exit_s.index[exit_s.to_numpy()]
    assert list(exit_idx) == [pd.Timestamp("2025-11-28")]
    assert entry_s.loc["2025-11-28"] == False  # noqa: E712
    assert not entry_s.loc["2025-11-28":].any()


def test_final_bar_of_truncated_history_never_signals():
    dates = pd.DatetimeIndex(["2025-10-01", "2025-10-02", "2025-10-03"])
    entry_s, exit_s = mod.season_decisions(dates)
    assert entry_s.iloc[-1] == False  # noqa: E712
    assert exit_s.iloc[-1] == False  # noqa: E712


def test_off_season_bars_never_signal():
    dates = pd.bdate_range("2025-12-01", periods=180)  # Dec-Aug, no season
    entry_s, exit_s = mod.season_decisions(dates)
    assert not entry_s.any()
    assert not exit_s.any()


# ------------------------------------------------------------------ factory

def test_signal_fn_matches_season_decisions_over_full_history():
    index = _calendar(start="2022-01-03", periods=1200)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("GLD", index, frames)
    sigs = _signals(signal_fn, frames["GLD"])
    entry_s, exit_s = mod.season_decisions(index)
    for i, (date, sig) in enumerate(zip(index, sigs)):
        assert sig == SignalResult(entry=bool(entry_s.loc[date]),
                                    exit=bool(exit_s.loc[date])), (i, date)


def test_price_never_enters_the_signal():
    index = _calendar(periods=800)
    a = mod.make_signal_fn("GLD", index, _frames(index, seed=1))
    b = mod.make_signal_fn("GLD", index, _frames(index, seed=999))
    frame_a = _frames(index, seed=1)["GLD"]
    frame_b = _frames(index, seed=999)["GLD"]
    assert _signals(a, frame_a) == _signals(b, frame_b)


def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=900)
    frames = _frames(index)
    k = 500
    base_fn = mod.make_signal_fn("GLD", index, frames)
    base = _signals(base_fn, frames["GLD"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.6
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.6
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("GLD", index, perturbed)
    after = _signals(pert_fn, perturbed["GLD"], upto=k)
    assert base == after


def test_holdout_blind_extra_future_dates_do_not_change_past_signals():
    index = _calendar(start="2023-01-03", periods=1000)
    frames = _frames(index)
    bucket = index[:600]
    fn_full = mod.make_signal_fn("GLD", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("GLD", bucket, truncated)
    bucket_frame = frames["GLD"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_holdout_bucket_opening_mid_season_still_enters_at_first_opportunity():
    # A holdout bucket that OPENS mid-season (here, Oct 1 -- after the
    # Sept-1 entry target but well before the Nov-30 exit) must still
    # enter GLD at its own first fill opportunity, not skip the season.
    full_index = pd.bdate_range("2024-01-02", periods=260)
    frames = _frames(full_index, seed=3)
    bucket = full_index[full_index >= pd.Timestamp("2024-10-01")]
    assert len(bucket) > 5
    signal_fn = mod.make_signal_fn("GLD", bucket, frames)
    bucket_frame = frames["GLD"].loc[bucket]
    trades, in_market = run_backtest(bucket_frame, signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    first_entry_pos = bucket.get_loc(trades.iloc[0]["entry_date"])
    assert first_entry_pos <= 2


def test_price_history_starting_mid_season_enters_at_first_opportunity_without_crashing():
    # Mirrors GLD's real situation: the TICKER'S OWN price history (not
    # just a holdout bucket) begins mid-way through what would otherwise
    # be its first season -- GLD's real inception is Nov 18 2004, inside
    # the 2004 Sep1-Nov30 window. The frame handed to make_signal_fn has
    # no data before that first bar, so traded_dates itself starts
    # mid-season. This must not crash, and (same mid-season-entry
    # interpretation as the holdout-bucket case above, since
    # season_decisions is computed on traded_dates directly) must enter
    # at the first available fill opportunity.
    full_index = pd.bdate_range("2004-11-18", periods=1500)
    frames = _frames(full_index, seed=9)
    signal_fn = mod.make_signal_fn("GLD", full_index, frames)
    trades, in_market = run_backtest(frames["GLD"], signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    first_entry_pos = full_index.get_loc(trades.iloc[0]["entry_date"])
    assert first_entry_pos <= 2
    # The partial 2004 season must exit in November 2004 (<= exit_target),
    # not spill into a later year.
    assert trades.iloc[0]["exit_date"].year == 2004
    assert trades.iloc[0]["exit_date"].month == 11


def test_price_history_starting_after_season_end_detects_no_partial_season():
    # If a ticker's price history begins entirely AFTER a would-be
    # season's exit_target (here, Dec 1 2004 -- after the 2004 season
    # already ended), in_season_mask must correctly show no season for
    # the remainder of 2004 (it only evaluates dates that actually exist
    # in the array) and pick up the very next real season (2005)
    # without crashing.
    full_index = pd.bdate_range("2004-12-01", periods=400)
    frames = _frames(full_index, seed=13)
    signal_fn = mod.make_signal_fn("GLD", full_index, frames)
    trades, in_market = run_backtest(frames["GLD"], signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    first_entry_date = trades.iloc[0]["entry_date"]
    # No trade may be dated inside the already-missed 2004 season window.
    assert not (pd.Timestamp("2004-12-01") <= first_entry_date
                <= pd.Timestamp("2005-08-31"))
    assert first_entry_date >= pd.Timestamp("2005-09-01") - pd.Timedelta(days=3)


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_annual_seasonal_trades():
    index = _calendar(start="2015-01-02", periods=2700)  # ~10.7 years
    frames = _frames(index, seed=7)
    signal_fn = mod.make_signal_fn("GLD", index, frames)
    trades, in_market = run_backtest(frames["GLD"], signal_fn, cost_bps=1.0)
    assert len(trades) >= 8  # roughly one trade per covered year
    for _, row in trades.iterrows():
        assert row["entry_date"].month in (8, 9)
        assert row["exit_date"].month == 11 or row["reason"] == "end_of_data"
        # The season is ~60-64 trading days; end_of_data trades can be shorter.
        if row["reason"] == "signal_exit":
            assert 40 <= row["held"] <= 75, row
    assert in_market.any()
    assert not in_market.all()  # most of the year must be flat


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("GLD", index, {})
    with pytest.raises(ValueError):
        mod.make_signal_fn("GLD", index, {"GLD": frames["GLD"].drop(columns=["Close"])})


def test_factory_rejects_empty_or_unsorted_index():
    frames = _frames(_calendar(periods=50))
    with pytest.raises(ValueError):
        mod.make_signal_fn("GLD", pd.DatetimeIndex([]), frames)
    bad = _calendar(periods=50)[::-1]  # descending, not sorted
    with pytest.raises(ValueError):
        mod.make_signal_fn("GLD", bad, frames)


def test_non_finite_close_does_not_affect_signal():
    # The rule never reads Close, so corrupting some Close values (while
    # keeping the Close COLUMN present, since the contract requires it)
    # must not change a single signal.
    index = _calendar(periods=500)
    clean = _frames(index, seed=2)
    dirty = {s: f.copy() for s, f in clean.items()}
    dirty["GLD"].iloc[10:20, dirty["GLD"].columns.get_loc("Close")] = np.nan
    dirty["GLD"].iloc[300, dirty["GLD"].columns.get_loc("Close")] = np.inf
    fn_clean = mod.make_signal_fn("GLD", index, clean)
    fn_dirty = mod.make_signal_fn("GLD", index, dirty)
    assert _signals(fn_clean, clean["GLD"]) == _signals(fn_dirty, dirty["GLD"])


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2008-09-01", periods=5000)
    frames = _frames(index, seed=11)
    signal_fn = mod.make_signal_fn("GLD", index, frames)
    frame = frames["GLD"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
