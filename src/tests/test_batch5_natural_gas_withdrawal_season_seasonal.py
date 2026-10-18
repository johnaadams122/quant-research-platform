"""Tests for src/mining/batch5/natural_gas_withdrawal_season_seasonal.py.

Synthetic data only (no network), deterministic, fast. Like
test_batch5_gasoline_driving_season_seasonal.py's convention, the
PARAMETERS drift check here compares against a FROZEN literal copy of the
pre-registered catalog record.

This family's season CROSSES A CALENDAR YEAR BOUNDARY (Nov 1 of year Y
through Mar 31 of year Y+1), unlike gasoline_driving_season_seasonal
(March-May, no boundary crossing). "year(bar) = bar's own calendar year"
is NOT a valid season key here -- a bar in Jan/Feb/March of year Y+1
belongs to the season keyed Y, not Y+1. The explicit boundary-crossing
tests below (December vs. February-of-next-year vs. June) exist
specifically to catch a naive same-calendar-year implementation.
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
from mining.batch5 import natural_gas_withdrawal_season_seasonal as mod  # noqa: E402

# Frozen copy of this family's pre-registered catalog "parameters" (see
# this module's own docstring for why the comparison is against a
# literal). Byte-for-byte copy of the catalog record.
_CATALOG_PARAMETERS = {
    "entry_month": 11,
    "entry_day": 1,
    "exit_month": 3,
    "exit_day": 31,
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


def _ung_close(index: pd.DatetimeIndex, seed: int = 5) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 20.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.025, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 5) -> dict:
    return {"UNG": _ohlc_from_close(_ung_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "natural_gas_withdrawal_season_seasonal"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["UNG"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "UNG"
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
    assert spec["traded"] == ["UNG"]
    assert spec["signal"] == []


# --------------------------------------------------- season_year_for_date

def test_season_key_december_and_february_share_same_season_year():
    # THE case a naive "year(bar) = bar's own calendar year" implementation
    # gets wrong: a December bar and the FOLLOWING February's bar belong to
    # the SAME season (keyed by the November-1 START year), not to two
    # different seasons keyed by their own calendar years.
    dates = pd.DatetimeIndex([
        "2024-12-15",  # December of year Y=2024 -> season 2024
        "2025-02-15",  # February of year Y+1=2025 -> ALSO season 2024
        "2025-06-15",  # June -> the gap month, no season at all
    ])
    years, valid = mod.season_year_for_date(dates)
    assert valid.tolist() == [True, True, False]
    assert int(years[0]) == 2024
    assert int(years[1]) == 2024  # NOT 2025 -- the naive-bug case
    in_season = mod.in_season_mask(dates)
    assert in_season.tolist() == [True, True, False]


def test_season_year_for_date_full_year_cycle():
    # One full calendar year of month-representative dates, hand-verified
    # against season_key(date) = date.year if month>=11 else
    # (date.year-1 if month<=3 else no season).
    dates = pd.DatetimeIndex([
        "2025-01-10", "2025-02-10", "2025-03-10",  # tail of season 2024
        "2025-04-10", "2025-05-10", "2025-06-10",  # gap
        "2025-07-10", "2025-08-10", "2025-09-10", "2025-10-10",  # gap
        "2025-11-10", "2025-12-10",  # head of season 2025
    ])
    years, valid = mod.season_year_for_date(dates)
    expected_valid = [True, True, True,
                      False, False, False, False, False, False, False,
                      True, True]
    expected_years_where_valid = {0: 2024, 1: 2024, 2: 2024, 10: 2025, 11: 2025}
    assert valid.tolist() == expected_valid
    for idx, year in expected_years_where_valid.items():
        assert int(years[idx]) == year


# ------------------------------------------------------------ in_season_mask

def test_in_season_mask_hand_verified_boundaries():
    # Season 2024: entry_target=2024-11-01, exit_target=2025-03-31.
    # Season 2025: entry_target=2025-11-01, exit_target=2026-03-31.
    dates = pd.DatetimeIndex([
        "2024-10-31", "2024-11-01", "2024-11-02",   # around 2024 entry
        "2025-03-30", "2025-03-31", "2025-04-01",   # around 2024-season exit
        "2025-10-31", "2025-11-01",                 # around 2025 entry
    ])
    expected = np.array([
        False, True, True,      # before Nov1 / on Nov1 / after Nov1
        True, True, False,      # before exit / on exit (inclusive) / after
        False, True,
    ])
    result = mod.in_season_mask(dates)
    assert result.tolist() == expected.tolist()


def test_in_season_mask_empty():
    assert mod.in_season_mask(pd.DatetimeIndex([])).tolist() == []


def test_in_season_mask_every_month_of_the_5_season_months_is_fully_in_season():
    # entry_day=1 and exit_day=31 (the last possible day of March) mean
    # EVERY day of Nov/Dec/Jan/Feb/Mar is in-season -- not just the
    # boundary dates.
    dates = pd.bdate_range("2024-11-01", "2025-03-31")
    assert mod.in_season_mask(dates).all()


def test_in_season_mask_every_gap_month_day_is_out_of_season():
    dates = pd.bdate_range("2025-04-01", "2025-10-31")
    assert not mod.in_season_mask(dates).any()


# --------------------------------------------------------- season_decisions

def test_season_decisions_entry_one_bar_early_exit_on_last_in_season_bar():
    # Hand-verified fixture spanning one full 2024-2025 season transition
    # on each side. entry_target(2024)=2024-11-01 (Friday, a trading day)
    # so the first in-season trading day is 2024-11-01 itself;
    # exit_target(2024)=2025-03-31 (Monday, a trading day) so the last
    # in-season trading day is 2025-03-31 itself.
    dates = pd.DatetimeIndex([
        "2024-10-30",  # off-season
        "2024-10-31",  # off-season (day before season starts)
        "2024-11-01",  # in-season (first day)
        "2024-11-04",  # in-season
        "2025-03-28",  # in-season
        "2025-03-31",  # in-season (last day, == exit_target)
        "2025-04-01",  # off-season (first day after exit_target)
        "2025-04-02",  # off-season
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    expected_entry = [False, True, True, True, True, False, False, False]
    expected_exit = [False, False, False, False, False, True, False, False]
    assert entry_s.tolist() == expected_entry
    assert exit_s.tolist() == expected_exit


def test_no_immediate_reentry_the_session_after_exit():
    # Guards against an immediate re-entry: entry[t] must be in_season[t+1],
    # never in_season[t] directly, or the exit bar (itself in-season) would
    # re-raise the entry flag and cause an immediate re-entry the very
    # next trading session after every exit.
    dates = pd.DatetimeIndex([
        "2025-03-27", "2025-03-28", "2025-03-31",  # tail of season 2024
        "2025-04-01", "2025-04-02", "2025-04-03",  # first sessions after exit
    ])
    entry_s, exit_s = mod.season_decisions(dates)
    exit_idx = exit_s.index[exit_s.to_numpy()]
    assert list(exit_idx) == [pd.Timestamp("2025-03-31")]
    assert entry_s.loc["2025-03-31"] == False  # noqa: E712
    assert not entry_s.loc["2025-03-31":].any()


def test_final_bar_of_truncated_history_never_signals():
    dates = pd.DatetimeIndex(["2025-01-06", "2025-01-07", "2025-01-08"])
    entry_s, exit_s = mod.season_decisions(dates)
    assert entry_s.iloc[-1] == False  # noqa: E712
    assert exit_s.iloc[-1] == False  # noqa: E712


def test_off_season_bars_never_signal():
    dates = pd.bdate_range("2025-04-01", periods=140)  # Apr-Oct, no season
    entry_s, exit_s = mod.season_decisions(dates)
    assert not entry_s.any()
    assert not exit_s.any()


def test_two_consecutive_seasons_produce_two_clean_round_trips():
    # A calendar spanning two winters back-to-back, run through the real
    # engine (not the raw season_decisions arrays -- entry[t] stays True
    # through most of a season by design, per interpretation choice 2, so
    # counting raw True entries would overcount; run_backtest is what
    # collapses the standing flag into actual trade round trips).
    index = pd.bdate_range("2024-09-02", "2026-05-29")  # starts in a gap month
    frames = _frames(index, seed=17)
    signal_fn = mod.make_signal_fn("UNG", index, frames)
    trades, in_market = run_backtest(frames["UNG"], signal_fn, cost_bps=0.0)
    assert len(trades) == 2
    for _, row in trades.iterrows():
        assert row["entry_date"].month in (10, 11)
        assert row["exit_date"].month in (3, 4)


def test_season_decisions_resolves_correct_season_without_seeing_nov_dec_bars():
    # A calendar slice that starts well INTO a season (mid-January) and
    # extends a few sessions past that season's own exit boundary, with
    # NO November/December bars visible at all. The season year must
    # still resolve correctly (2024, not 2025) purely from month
    # arithmetic -- proving the module never relies on having "seen" the
    # season's own November start to know which season a bar belongs to.
    dates = pd.bdate_range("2025-01-15", "2025-04-04")
    entry_s, exit_s = mod.season_decisions(dates)
    exit_dates = exit_s.index[exit_s.to_numpy()]
    assert len(exit_dates) == 1
    assert exit_dates[0].month == 3
    assert exit_dates[0].day >= 28


# ------------------------------------------------------------------ factory

def test_signal_fn_matches_season_decisions_over_full_history():
    index = _calendar(start="2022-01-03", periods=1200)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("UNG", index, frames)
    sigs = _signals(signal_fn, frames["UNG"])
    entry_s, exit_s = mod.season_decisions(index)
    for i, (date, sig) in enumerate(zip(index, sigs)):
        assert sig == SignalResult(entry=bool(entry_s.loc[date]),
                                    exit=bool(exit_s.loc[date])), (i, date)


def test_price_never_enters_the_signal():
    index = _calendar(periods=800)
    a = mod.make_signal_fn("UNG", index, _frames(index, seed=1))
    b = mod.make_signal_fn("UNG", index, _frames(index, seed=999))
    frame_a = _frames(index, seed=1)["UNG"]
    frame_b = _frames(index, seed=999)["UNG"]
    assert _signals(a, frame_a) == _signals(b, frame_b)


def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=900)
    frames = _frames(index)
    k = 500
    base_fn = mod.make_signal_fn("UNG", index, frames)
    base = _signals(base_fn, frames["UNG"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.6
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.6
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("UNG", index, perturbed)
    after = _signals(pert_fn, perturbed["UNG"], upto=k)
    assert base == after


def test_holdout_blind_extra_future_dates_do_not_change_past_signals():
    index = _calendar(start="2023-01-03", periods=1000)
    frames = _frames(index)
    bucket = index[:600]
    fn_full = mod.make_signal_fn("UNG", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("UNG", bucket, truncated)
    bucket_frame = frames["UNG"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_holdout_bucket_opening_mid_season_still_enters_at_first_opportunity():
    # A holdout bucket that OPENS mid-season (here, January 2 2025 --
    # after the Nov-1 entry date but well before the Mar-31 exit) must
    # still enter UNG at its own first fill opportunity.
    full_index = pd.bdate_range("2024-09-02", periods=260)
    frames = _frames(full_index, seed=3)
    bucket = full_index[full_index >= pd.Timestamp("2025-01-02")]
    assert len(bucket) > 5
    signal_fn = mod.make_signal_fn("UNG", bucket, frames)
    bucket_frame = frames["UNG"].loc[bucket]
    trades, in_market = run_backtest(bucket_frame, signal_fn, cost_bps=0.0)
    assert len(trades) >= 1
    first_entry_pos = bucket.get_loc(trades.iloc[0]["entry_date"])
    assert first_entry_pos <= 2


def test_holdout_bucket_spanning_the_year_boundary_uses_the_correct_season():
    # A holdout bucket that starts inside Jan-Mar of year Y+1 (no Nov/Dec
    # bars visible at all) and extends a few sessions past that season's
    # own exit boundary must still resolve to season Y, not attempt to
    # find a (nonexistent) season keyed Y+1. Exercises the exact
    # naive-bug shape inside make_signal_fn/season_decisions together,
    # not just the raw season-key helper.
    full_index = pd.bdate_range("2024-09-02", periods=400)
    frames = _frames(full_index, seed=9)
    bucket = full_index[(full_index >= pd.Timestamp("2025-01-15"))
                        & (full_index <= pd.Timestamp("2025-04-04"))]
    assert len(bucket) > 5
    signal_fn = mod.make_signal_fn("UNG", bucket, frames)
    bucket_frame = frames["UNG"].loc[bucket]
    trades, in_market = run_backtest(bucket_frame, signal_fn, cost_bps=0.0)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_date"].month == 3
    assert trades.iloc[0]["reason"] == "signal_exit"


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_annual_seasonal_trades():
    # Starts in April (a gap month) rather than January -- January would
    # land mid-season (the tail of the PRECEDING winter), which correctly
    # enters immediately per interpretation choice 2 but would make the
    # first trade's entry_date an outlier for this test's per-trade month
    # assertion (see test_holdout_bucket_opening_mid_season_still_enters_
    # at_first_opportunity for that documented mid-season-start behavior).
    index = _calendar(start="2015-04-01", periods=2900)  # ~11.5 years
    frames = _frames(index, seed=7)
    signal_fn = mod.make_signal_fn("UNG", index, frames)
    trades, in_market = run_backtest(frames["UNG"], signal_fn, cost_bps=1.0)
    assert len(trades) >= 8  # roughly one trade per covered winter
    for _, row in trades.iterrows():
        assert row["entry_date"].month in (10, 11)
        assert row["exit_date"].month in (3, 4) or row["reason"] == "end_of_data"
        # A season is ~100 trading days (5 months); end_of_data trades can
        # be shorter.
        if row["reason"] == "signal_exit":
            assert 90 <= row["held"] <= 115, row
    assert in_market.any()
    assert not in_market.all()  # most of the year must be flat


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("UNG", index, {})
    with pytest.raises(ValueError):
        mod.make_signal_fn("UNG", index, {"UNG": frames["UNG"].drop(columns=["Close"])})


def test_factory_rejects_empty_or_unsorted_index():
    frames = _frames(_calendar(periods=50))
    with pytest.raises(ValueError):
        mod.make_signal_fn("UNG", pd.DatetimeIndex([]), frames)
    bad = _calendar(periods=50)[::-1]  # descending, not sorted
    with pytest.raises(ValueError):
        mod.make_signal_fn("UNG", bad, frames)


def test_non_finite_close_does_not_affect_signal():
    index = _calendar(periods=500)
    clean = _frames(index, seed=2)
    dirty = {s: f.copy() for s, f in clean.items()}
    dirty["UNG"].iloc[10:20, dirty["UNG"].columns.get_loc("Close")] = np.nan
    dirty["UNG"].iloc[300, dirty["UNG"].columns.get_loc("Close")] = np.inf
    fn_clean = mod.make_signal_fn("UNG", index, clean)
    fn_dirty = mod.make_signal_fn("UNG", index, dirty)
    assert _signals(fn_clean, clean["UNG"]) == _signals(fn_dirty, dirty["UNG"])


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2007-04-18", periods=5000)  # UNG's real inception
    frames = _frames(index, seed=11)
    signal_fn = mod.make_signal_fn("UNG", index, frames)
    frame = frames["UNG"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0


def test_real_ung_inception_does_not_crash_truncation_logic():
    # UNG's real inception is April 2007 (mid gap-month, before the first
    # reachable season's Nov-1 entry). The first bar of the whole history
    # is itself an off-season bar; this must not crash the vectorized
    # array construction (e.g. an out-of-range Y-1 or Y+1 at the very
    # start/end of the frame).
    index = pd.bdate_range("2007-04-18", periods=60)  # April/May 2007 only
    frames = _frames(index, seed=13)
    signal_fn = mod.make_signal_fn("UNG", index, frames)
    sigs = _signals(signal_fn, frames["UNG"])
    assert not any(s.entry or s.exit for s in sigs)  # entirely off-season
