"""Tests for src/mining/batch5/gasoline_driving_season_seasonal.py.

Synthetic data only (no network), deterministic, fast. The PARAMETERS drift
check here compares against a FROZEN literal copy of the pre-registered
catalog record.
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
from mining.batch5 import gasoline_driving_season_seasonal as mod  # noqa: E402

# Frozen copy of this family's pre-registered catalog "parameters"; the
# drift check compares against this literal.
_CATALOG_PARAMETERS = {
    "entry_month": 3,
    "entry_day": 1,
    "exit_rule": "last_monday_of_may",
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


def _uga_close(index: pd.DatetimeIndex, seed: int = 5) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 60.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.02, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 5) -> dict:
    return {"UGA": _ohlc_from_close(_uga_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "gasoline_driving_season_seasonal"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["UGA"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "UGA"
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
    assert spec["traded"] == ["UGA"]
    assert spec["signal"] == []


# ------------------------------------------------------- last_monday_of_may

def test_last_monday_of_may_hand_computed():
    # Hand-computed via ISO weekday arithmetic and cross-checked against
    # real-world Memorial Day dates (US federal holiday, last Monday of
    # May): 2008-05-26, 2009-05-25, 2024-05-27, 2025-05-26, 2026-05-25.
    years = np.array([2008, 2009, 2024, 2025, 2026])
    expected = pd.DatetimeIndex([
        "2008-05-26", "2009-05-25", "2024-05-27", "2025-05-26", "2026-05-25",
    ])
    result = mod.last_monday_of_may(years)
    assert list(result) == list(expected)


def test_last_monday_of_may_is_always_a_monday_and_in_may():
    years = np.arange(2000, 2031)
    result = mod.last_monday_of_may(years)
    assert (result.dayofweek == 0).all()
    assert (result.month == 5).all()
    assert (result.day >= 25).all()  # the last Monday of May is always >= 25


# ------------------------------------------------------------ in_season_mask

def test_in_season_mask_hand_verified_boundaries():
    # 2025: entry_target=2025-03-01 (a Saturday), exit_target=2025-05-26.
    # 2026: entry_target=2026-03-01 (a Sunday), exit_target=2026-05-25.
    dates = pd.DatetimeIndex([
        "2025-02-28", "2025-03-01", "2025-03-02",   # around 2025 entry
        "2025-05-25", "2025-05-26", "2025-05-27",   # around 2025 exit
        "2026-02-28", "2026-03-01",                 # around 2026 entry
        "2026-05-24", "2026-05-25", "2026-05-26",   # around 2026 exit
    ])
    expected = np.array([
        False, True, True,      # before Mar1 / on Mar1 / after Mar1
        True, True, False,      # before exit / on exit (inclusive) / after
        False, True,
        True, True, False,
    ])
    result = mod.in_season_mask(dates)
    assert result.tolist() == expected.tolist()


def test_in_season_mask_empty():
    assert mod.in_season_mask(pd.DatetimeIndex([])).tolist() == []


# --------------------------------------------------------- season_decisions

def test_season_decisions_entry_one_bar_early_exit_on_last_in_season_bar():
    # Hand-verified fixture spanning one full 2025 season transition on
    # each side. entry_target(2025)=2025-03-01 (Sat) so the first in-season
    # trading day is 2025-03-03 (Mon); exit_target(2025)=2025-05-26 (Mon,
    # itself a market holiday) so the last in-season trading day is
    # 2025-05-23 (Fri).
    dates = pd.DatetimeIndex([
        "2025-02-27",  # off-season
        "2025-02-28",  # off-season (day before season starts)
        "2025-03-03",  # in-season (first day)
        "2025-03-04",  # in-season
        "2025-05-22",  # in-season
        "2025-05-23",  # in-season (last day, <= exit_target 2025-05-26)
        "2025-05-27",  # off-season (first day after exit_target)
        "2025-05-28",  # off-season
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    expected_entry = [False, True, True, True, True, False, False, False]
    expected_exit = [False, False, False, False, False, True, False, False]
    assert entry_s.tolist() == expected_entry
    assert exit_s.tolist() == expected_exit


def test_no_immediate_reentry_the_session_after_exit():
    # Guards against an immediate re-entry the session after an exit. A
    # "standing flag" entry[t] = in_season[t] would ignore the fact that the
    # EXIT bar is itself in-season: it would re-raise the entry flag on the
    # exit bar, and since the engine reads the entry flag from the bar
    # immediately preceding a prospective fill, it would cause an IMMEDIATE
    # re-entry the very next trading session after every single exit
    # (one-season round trips merged into multi-year holds). The module
    # uses entry[t] = in_season[t+1] (interpretation choice 3).
    dates = pd.DatetimeIndex([
        "2025-05-21", "2025-05-22", "2025-05-23",  # tail of 2025 season
        "2025-05-27", "2025-05-28", "2025-05-29",  # first sessions after exit
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    exit_idx = exit_s.index[exit_s.to_numpy()]
    assert list(exit_idx) == [pd.Timestamp("2025-05-23")]
    # The engine reads the entry flag from the bar IMMEDIATELY BEFORE a
    # prospective fill: to open a position at 2025-05-27's open, it reads
    # entry at 2025-05-23 (the exit bar itself). That is exactly the flag
    # the naive "entry = in_season" construction would get wrong (in_season on
    # the exit day is True, since the exit day IS the last in-season day),
    # so this is the precise position that must be False.
    assert entry_s.loc["2025-05-23"] == False  # noqa: E712
    # And, as a full behavioral check, no bar from the exit day onward may
    # carry an entry flag in this fixture (the next season is ~9 months
    # away, entirely outside this date range).
    assert not entry_s.loc["2025-05-23":].any()


def test_final_bar_of_truncated_history_never_signals():
    # Mid-season final bar: without a following bar the module cannot know
    # whether the season continues, so both flags must stay False on it
    # (the engine's own end_of_data path force-closes any open position
    # regardless).
    dates = pd.DatetimeIndex(["2025-04-01", "2025-04-02", "2025-04-03"])
    entry_s, exit_s = mod.season_decisions(dates)
    assert entry_s.iloc[-1] == False  # noqa: E712
    assert exit_s.iloc[-1] == False  # noqa: E712


def test_off_season_bars_never_signal():
    dates = pd.bdate_range("2025-06-02", periods=180)  # Jun-Feb, no season
    entry_s, exit_s = mod.season_decisions(dates)
    assert not entry_s.any()
    assert not exit_s.any()


# ------------------------------------------------------------------ factory

def test_signal_fn_matches_season_decisions_over_full_history():
    index = _calendar(start="2022-01-03", periods=1200)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("UGA", index, frames)
    sigs = _signals(signal_fn, frames["UGA"])
    entry_s, exit_s = mod.season_decisions(index)
    for i, (date, sig) in enumerate(zip(index, sigs)):
        assert sig == SignalResult(entry=bool(entry_s.loc[date]),
                                    exit=bool(exit_s.loc[date])), (i, date)


def test_price_never_enters_the_signal():
    index = _calendar(periods=800)
    a = mod.make_signal_fn("UGA", index, _frames(index, seed=1))
    b = mod.make_signal_fn("UGA", index, _frames(index, seed=999))
    frame_a = _frames(index, seed=1)["UGA"]
    frame_b = _frames(index, seed=999)["UGA"]
    assert _signals(a, frame_a) == _signals(b, frame_b)


def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=900)
    frames = _frames(index)
    k = 500
    base_fn = mod.make_signal_fn("UGA", index, frames)
    base = _signals(base_fn, frames["UGA"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.6
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.6
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("UGA", index, perturbed)
    after = _signals(pert_fn, perturbed["UGA"], upto=k)
    assert base == after


def test_holdout_blind_extra_future_dates_do_not_change_past_signals():
    # Extending the AVAILABLE DATE RANGE past the bucket (not just prices)
    # matters for this module specifically, because the exit flag is
    # derived from comparing a bar to its successor -- an un-truncated
    # frame could let a future bar leak into that comparison at the
    # boundary.
    index = _calendar(start="2023-01-03", periods=1000)
    frames = _frames(index)
    bucket = index[:600]
    fn_full = mod.make_signal_fn("UGA", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("UGA", bucket, truncated)
    bucket_frame = frames["UGA"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_holdout_bucket_opening_mid_season_still_enters_at_first_opportunity():
    # Interpretation choice 3: a holdout bucket that OPENS mid-season (here,
    # April 1 2024 -- after the March-1 entry target but well before the
    # Memorial Day exit) must still enter UGA at its own first fill
    # opportunity, not skip the season entirely.
    full_index = pd.bdate_range("2024-01-02", periods=260)
    frames = _frames(full_index, seed=3)
    bucket = full_index[full_index >= pd.Timestamp("2024-04-01")]
    assert len(bucket) > 5
    signal_fn = mod.make_signal_fn("UGA", bucket, frames)
    bucket_frame = frames["UGA"].loc[bucket]
    trades, in_market = run_backtest(bucket_frame, signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    # The entry must land within the first few sessions of the bucket, not
    # somewhere deep inside it (proving it wasn't waiting for a fresh
    # March-1 transition that no longer exists inside this bucket).
    first_entry_pos = bucket.get_loc(trades.iloc[0]["entry_date"])
    assert first_entry_pos <= 2


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_annual_seasonal_trades():
    index = _calendar(start="2015-01-02", periods=2700)  # ~10.7 years
    frames = _frames(index, seed=7)
    signal_fn = mod.make_signal_fn("UGA", index, frames)
    trades, in_market = run_backtest(frames["UGA"], signal_fn, cost_bps=1.0)
    assert len(trades) >= 8  # roughly one trade per covered year
    for _, row in trades.iterrows():
        assert row["entry_date"].month in (2, 3)
        assert row["exit_date"].month in (5, 6) or row["reason"] == "end_of_data"
        # A season is ~58-65 trading days; end_of_data trades can be shorter.
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
        mod.make_signal_fn("UGA", index, {})
    with pytest.raises(ValueError):
        mod.make_signal_fn("UGA", index, {"UGA": frames["UGA"].drop(columns=["Close"])})


def test_factory_rejects_empty_or_unsorted_index():
    frames = _frames(_calendar(periods=50))
    with pytest.raises(ValueError):
        mod.make_signal_fn("UGA", pd.DatetimeIndex([]), frames)
    bad = _calendar(periods=50)[::-1]  # descending, not sorted
    with pytest.raises(ValueError):
        mod.make_signal_fn("UGA", bad, frames)


def test_non_finite_close_does_not_affect_signal():
    # The rule never reads Close, so corrupting some Close values (while
    # keeping the Close COLUMN present, since the contract requires it)
    # must not change a single signal.
    index = _calendar(periods=500)
    clean = _frames(index, seed=2)
    dirty = {s: f.copy() for s, f in clean.items()}
    dirty["UGA"].iloc[10:20, dirty["UGA"].columns.get_loc("Close")] = np.nan
    dirty["UGA"].iloc[300, dirty["UGA"].columns.get_loc("Close")] = np.inf
    fn_clean = mod.make_signal_fn("UGA", index, clean)
    fn_dirty = mod.make_signal_fn("UGA", index, dirty)
    assert _signals(fn_clean, clean["UGA"]) == _signals(fn_dirty, dirty["UGA"])


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2008-03-01", periods=5000)
    frames = _frames(index, seed=11)
    signal_fn = mod.make_signal_fn("UGA", index, frames)
    frame = frames["UGA"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
