"""Tests for src/mining/batch5/presidential_cycle_year3_seasonal.py.

Synthetic data only (no network), deterministic, fast. This family is
registered in data/mining/catalog.json. The PARAMETERS drift check below
compares against a FROZEN literal copy of the pre-registered record rather
than reading the live file (the catalog entry is byte-identical to this
literal) -- same convention as test_batch5_gold_autumn_effect_seasonal.py.
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
from mining.batch5 import presidential_cycle_year3_seasonal as mod  # noqa: E402

# Frozen copy of this family's pre-registered catalog "parameters" (must
# match PARAMETERS byte-for-byte as JSON).
_CATALOG_PARAMETERS = {
    "cycle_length_years": 4,
    "election_year_modulus": 0,
    "target_year_offset_from_election": 3,
    "rebalance": "quadrennial",
}


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2010-01-04", periods=1200) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _spy_close(index: pd.DatetimeIndex, seed: int = 5) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 120.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 5) -> dict:
    return {"SPY": _ohlc_from_close(_spy_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "presidential_cycle_year3_seasonal"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SPY"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
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
    assert spec["traded"] == ["SPY"]
    assert spec["signal"] == []


# ------------------------------------------------------------ in_season_mask

def test_in_season_mask_hand_verified_election_cycle_years():
    # 2023 % 4 == 3 -> Year 3 (pre-election, 2024 is the election year):
    # in-season. 2024 (election year, %4==0), 2025 (Year 1, %4==1), 2026
    # (Year 2 / midterm, %4==2) are all off-season. 2027 (%4==3) is the
    # NEXT Year-3 year: in-season again.
    dates = pd.DatetimeIndex([
        "2022-06-15",  # Year 2 (midterm), %4==2 -- off
        "2023-01-02",  # Year 3, %4==3 -- on
        "2023-06-15",  # Year 3 -- on
        "2023-12-29",  # Year 3 -- on
        "2024-01-02",  # election year, %4==0 -- off
        "2025-06-15",  # Year 1, %4==1 -- off
        "2026-06-15",  # Year 2, %4==2 -- off
        "2027-01-04",  # Year 3, %4==3 -- on
    ])
    expected = [False, True, True, True, False, False, False, True]
    assert mod.in_season_mask(dates).tolist() == expected


def test_in_season_mask_empty():
    assert mod.in_season_mask(pd.DatetimeIndex([])).tolist() == []


def test_in_season_mask_depends_only_on_calendar_year():
    # Every date within a given calendar year must agree, regardless of
    # month/day -- unlike gold_autumn_effect_seasonal, there is no
    # sub-year month/day boundary to this family's rule at all.
    on_year_dates = pd.DatetimeIndex(
        ["2023-01-01", "2023-04-15", "2023-07-04", "2023-12-31"])
    off_year_dates = pd.DatetimeIndex(
        ["2022-01-01", "2024-04-15", "2025-07-04", "2026-12-31"])
    assert mod.in_season_mask(on_year_dates).all()
    assert not mod.in_season_mask(off_year_dates).any()


# --------------------------------------------------------- season_decisions

def test_season_decisions_entry_one_bar_early_exit_on_last_in_season_bar():
    # Hand-verified fixture spanning one full Year-3 (2023) transition on
    # each side, using ordinary business days (2023-01-02 is the first
    # business day of 2023; 2023-12-29 is the last).
    dates = pd.DatetimeIndex([
        "2022-12-28",  # off-season (2022, Year 2)
        "2022-12-29",  # off-season (day before season starts)
        "2023-01-02",  # in-season (first day of Year 3)
        "2023-01-03",  # in-season
        "2023-12-27",  # in-season
        "2023-12-29",  # in-season (last day of Year 3)
        "2024-01-02",  # off-season (first day after Year 3 ends)
        "2024-01-03",  # off-season
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    expected_entry = [False, True, True, True, True, False, False, False]
    expected_exit = [False, False, False, False, False, True, False, False]
    assert entry_s.tolist() == expected_entry
    assert exit_s.tolist() == expected_exit


def test_no_immediate_reentry_the_session_after_exit():
    # entry[t] must be in_season[t+1], never in_season[t] directly, or the
    # exit bar itself (last in-season day) would re-raise "enter" and
    # cause an immediate re-entry the very next session.
    dates = pd.DatetimeIndex([
        "2023-12-26", "2023-12-27", "2023-12-29",  # tail of 2023 season
        "2024-01-02", "2024-01-03", "2024-01-04",  # first sessions after exit
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    exit_idx = exit_s.index[exit_s.to_numpy()]
    assert list(exit_idx) == [pd.Timestamp("2023-12-29")]
    assert entry_s.loc["2023-12-29"] == False  # noqa: E712
    assert not entry_s.loc["2023-12-29":].any()


def test_final_bar_of_truncated_history_never_signals():
    dates = pd.DatetimeIndex(["2023-06-01", "2023-06-02", "2023-06-05"])
    entry_s, exit_s = mod.season_decisions(dates)
    assert entry_s.iloc[-1] == False  # noqa: E712
    assert exit_s.iloc[-1] == False  # noqa: E712


def test_off_years_never_signal():
    dates = pd.bdate_range("2024-01-02", periods=500)  # 2024-2025, all off
    entry_s, exit_s = mod.season_decisions(dates)
    assert not entry_s.any()
    assert not exit_s.any()


# ------------------------------------------------------------------ factory

def test_signal_fn_matches_season_decisions_over_full_history():
    index = _calendar(start="2010-01-04", periods=1200)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])
    entry_s, exit_s = mod.season_decisions(index)
    for i, (date, sig) in enumerate(zip(index, sigs)):
        assert sig == SignalResult(entry=bool(entry_s.loc[date]),
                                    exit=bool(exit_s.loc[date])), (i, date)


def test_holdout_blind_bucket_ending_exactly_at_season_close_does_not_leak_future_exit():
    # A bucket whose boundary does not fall on a season transition cannot
    # detect a missing holdout-blind truncation:
    # test_holdout_blind_extra_future_dates_do_not_change_past_signals
    # would still pass with the truncation REMOVED, because in_season_mask
    # depends only on each bar's own year, not its neighbors. The one place
    # adjacency (and
    # therefore future information) actually enters is season_decisions'
    # one-bar shift for entry/exit -- so this deliberately engineers the
    # bucket's LAST bar to be the real Dec-29-2023 close of a Year-3
    # season (2023 %4==3), with frames extending into 2024 beyond it.
    # WITHOUT truncation, a naive computation would see 2024-01-02
    # (off-season) as "the next date" and flag exit=True on that last
    # bar using information a real holdout run at that date could not
    # legitimately use for cross-checking; WITH the factory's truncation
    # (see the module docstring's interpretation choice 4),
    # the truncated traded_dates ends AT that bar, so it is the final
    # element of the truncated array and season_decisions' own "never
    # signal on the final bar" rule applies -- exit must be False there,
    # deferred to the engine's end_of_data close.
    full_index = pd.bdate_range("2022-01-03", periods=900)  # spans 2022-2025
    frames = _frames(full_index, seed=17)
    season_close = pd.Timestamp("2023-12-29")
    assert season_close in full_index
    bucket = full_index[full_index <= season_close]
    assert bucket[-1] == season_close
    # Sanity: the very next REAL trading day after the bucket is off-season
    # (2024), confirming this bucket boundary really does coincide with a
    # season transition (i.e. this fixture actually exercises the gap).
    next_real_date = full_index[full_index > season_close][0]
    assert not mod.in_season_mask(pd.DatetimeIndex([next_real_date]))[0]

    signal_fn = mod.make_signal_fn("SPY", bucket, frames)
    bucket_frame = frames["SPY"].loc[bucket]
    last_sig = signal_fn(bucket_frame, len(bucket) - 1)
    assert last_sig.exit is False, (
        "truncated bucket must defer the season-close exit to the "
        "engine's own end_of_data handling, never flag it using a real "
        "trading day beyond the bucket's own last bar")


def test_price_never_enters_the_signal():
    index = _calendar(periods=800)
    a = mod.make_signal_fn("SPY", index, _frames(index, seed=1))
    b = mod.make_signal_fn("SPY", index, _frames(index, seed=999))
    frame_a = _frames(index, seed=1)["SPY"]
    frame_b = _frames(index, seed=999)["SPY"]
    assert _signals(a, frame_a) == _signals(b, frame_b)


def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=900)
    frames = _frames(index)
    k = 500
    base_fn = mod.make_signal_fn("SPY", index, frames)
    base = _signals(base_fn, frames["SPY"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.6
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.6
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("SPY", index, perturbed)
    after = _signals(pert_fn, perturbed["SPY"], upto=k)
    assert base == after


def test_holdout_blind_extra_future_dates_do_not_change_past_signals():
    index = _calendar(start="2011-01-03", periods=1000)
    frames = _frames(index)
    bucket = index[:600]
    fn_full = mod.make_signal_fn("SPY", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("SPY", bucket, truncated)
    bucket_frame = frames["SPY"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_holdout_bucket_opening_mid_season_still_enters_at_first_opportunity():
    # A holdout bucket that OPENS mid-season (here, mid-2023 -- well
    # inside the 2023 Year-3 season) must still enter SPY at its own
    # first fill opportunity, not skip the season. This code path is
    # never exercised by SPY's own REAL full history (SPY's real
    # inception year, 1993, is off-season -- see the module docstring's
    # interpretation choice 6), but any holdout bucket carved out of a
    # longer history can still open mid-season, so it must work.
    full_index = pd.bdate_range("2023-01-03", periods=260)  # 2023-2024
    frames = _frames(full_index, seed=3)
    bucket = full_index[full_index >= pd.Timestamp("2023-06-01")]
    assert len(bucket) > 5
    signal_fn = mod.make_signal_fn("SPY", bucket, frames)
    bucket_frame = frames["SPY"].loc[bucket]
    trades, in_market = run_backtest(bucket_frame, signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    first_entry_pos = bucket.get_loc(trades.iloc[0]["entry_date"])
    assert first_entry_pos <= 2


def test_price_history_starting_mid_season_enters_at_first_opportunity_without_crashing():
    # If a ticker's OWN price history (not just a holdout bucket) begins
    # mid-way through a Year-3 season, make_signal_fn must not crash and
    # must enter at the first available fill opportunity (same
    # mid-season-entry interpretation as the holdout-bucket case above,
    # since season_decisions is computed on traded_dates directly).
    full_index = pd.bdate_range("2023-06-01", periods=1500)
    frames = _frames(full_index, seed=9)
    signal_fn = mod.make_signal_fn("SPY", full_index, frames)
    trades, in_market = run_backtest(frames["SPY"], signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    first_entry_pos = full_index.get_loc(trades.iloc[0]["entry_date"])
    assert first_entry_pos <= 2
    # The partial 2023 season must exit before 2024 begins.
    assert trades.iloc[0]["exit_date"].year == 2023


def test_price_history_starting_after_season_end_detects_no_partial_season():
    # If a ticker's price history begins entirely AFTER a would-be
    # season already ended (here, 2024-02-01 -- after the 2023 Year-3
    # season is over and before the next one in 2027), in_season_mask
    # must correctly show no season until 2027 and pick up that next
    # real season without crashing.
    full_index = pd.bdate_range("2024-02-01", periods=800)  # ~2024-2027
    frames = _frames(full_index, seed=13)
    signal_fn = mod.make_signal_fn("SPY", full_index, frames)
    trades, in_market = run_backtest(frames["SPY"], signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    first_entry_date = trades.iloc[0]["entry_date"]
    # No trade may be dated inside the off-season gap 2024-2026.
    assert not (pd.Timestamp("2024-02-01") <= first_entry_date
                <= pd.Timestamp("2026-12-31"))
    assert first_entry_date.year == 2027


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_quadrennial_seasonal_trades():
    index = _calendar(start="2012-01-03", periods=4200)  # ~16.7 years
    frames = _frames(index, seed=7)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    trades, in_market = run_backtest(frames["SPY"], signal_fn, cost_bps=1.0)
    assert len(trades) >= 3  # roughly one trade per ~4-year cycle covered
    for _, row in trades.iterrows():
        assert row["entry_date"].year % 4 == 3
        if row["reason"] == "signal_exit":
            assert row["exit_date"].year == row["entry_date"].year
            # A full calendar year is roughly 250-252 trading sessions.
            assert 230 <= row["held"] <= 262, row
    assert in_market.any()
    assert not in_market.all()  # most of the 4-year cycle must be flat


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {})
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {"SPY": frames["SPY"].drop(columns=["Close"])})


def test_factory_rejects_empty_or_unsorted_index():
    frames = _frames(_calendar(periods=50))
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", pd.DatetimeIndex([]), frames)
    bad = _calendar(periods=50)[::-1]  # descending, not sorted
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", bad, frames)


def test_non_finite_close_does_not_affect_signal():
    # The rule never reads Close, so corrupting some Close values (while
    # keeping the Close COLUMN present, since the contract requires it)
    # must not change a single signal.
    index = _calendar(periods=500)
    clean = _frames(index, seed=2)
    dirty = {s: f.copy() for s, f in clean.items()}
    dirty["SPY"].iloc[10:20, dirty["SPY"].columns.get_loc("Close")] = np.nan
    dirty["SPY"].iloc[300, dirty["SPY"].columns.get_loc("Close")] = np.inf
    fn_clean = mod.make_signal_fn("SPY", index, clean)
    fn_dirty = mod.make_signal_fn("SPY", index, dirty)
    assert _signals(fn_clean, clean["SPY"]) == _signals(fn_dirty, dirty["SPY"])


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2008-09-01", periods=5000)
    frames = _frames(index, seed=11)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    frame = frames["SPY"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
