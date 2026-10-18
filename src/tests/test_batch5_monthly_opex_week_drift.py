"""Tests for src/mining/batch5/monthly_opex_week_drift.py.

Synthetic data only (no network), deterministic, fast. Like
test_batch5_gasoline_driving_season_seasonal.py, the PARAMETERS drift
check here compares against a FROZEN literal copy of the pre-registered
catalog record.

Third-Friday / entry-position ground truth used below is independently
hand-verified with Python's stdlib ``datetime`` (Monday=0..Sunday=6,
Friday=4), e.g.:
  2024-01: day1=Mon -> third Friday 2024-01-19; entry bar (-4 sessions)
           2024-01-15 (Mon); entry flag (one bar earlier) 2024-01-12 (Fri).
  2024-06: day1=Sat -> third Friday 2024-06-21; entry bar 2024-06-17 (Mon);
           entry flag 2024-06-14 (Fri).
  2024-09: day1=Sun -> third Friday 2024-09-20; entry bar 2024-09-16 (Mon);
           entry flag 2024-09-13 (Fri).
  2024-11: day1=Fri -> third Friday 2024-11-15; entry bar 2024-11-11 (Mon);
           entry flag 2024-11-08 (Fri).
These four months deliberately span four different first-of-month
weekdays (Mon/Sat/Sun/Fri) to rule out an off-by-one in the "first Friday"
computation.
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
from mining.batch5 import monthly_opex_week_drift as mod  # noqa: E402

# Frozen copy of this family's pre-registered catalog "parameters" (byte-
# identical JSON check against a literal copy).
_CATALOG_PARAMETERS = {
    "opex_week_definition": "week_containing_third_friday_of_month",
    "entry_trading_days_before_opex_friday": 4,
    "exit_at": "opex_friday_close",
}


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2015-01-02", end=None, periods=None) -> pd.DatetimeIndex:
    if end is not None:
        return pd.bdate_range(start, end)
    return pd.bdate_range(start, periods=periods)


def _spy_close(index: pd.DatetimeIndex, seed: int = 5) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 200.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 5) -> dict:
    return {"SPY": _ohlc_from_close(_spy_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "monthly_opex_week_drift"
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


# ------------------------------------------------------- third_friday_of_month

def test_third_friday_of_month_hand_computed():
    # Independently hand-verified via stdlib datetime (see module docstring
    # above) -- four different first-of-month weekdays (Mon/Sat/Sun/Fri).
    years = np.array([2024, 2024, 2024, 2024])
    months = np.array([1, 6, 9, 11])
    expected = pd.DatetimeIndex([
        "2024-01-19", "2024-06-21", "2024-09-20", "2024-11-15",
    ])
    result = mod.third_friday_of_month(years, months)
    assert list(result) == list(expected)


def test_third_friday_of_month_is_always_a_friday_in_day_15_to_21():
    years = np.repeat(np.arange(2000, 2031), 12)
    months = np.tile(np.arange(1, 13), 31)
    result = mod.third_friday_of_month(years, months)
    assert (result.dayofweek == 4).all()
    assert (result.day >= 15).all()
    assert (result.day <= 21).all()


def test_third_friday_of_month_empty():
    result = mod.third_friday_of_month(np.array([], dtype=np.int64),
                                        np.array([], dtype=np.int64))
    assert len(result) == 0


# --------------------------------------------------- opex_window_decisions

def test_opex_window_decisions_hand_verified_two_different_first_weekdays():
    # Jan 2024 (day1=Monday) and Jun 2024 (day1=Saturday) -- deliberately
    # different first-of-month weekdays, confirmed by later bars (Feb/Jul).
    dates = _calendar(start="2024-01-02", end="2024-07-31")
    entry_s, exit_s = mod.opex_window_decisions(dates)
    assert entry_s.loc[pd.Timestamp("2024-01-12")] == True  # noqa: E712
    assert exit_s.loc[pd.Timestamp("2024-01-19")] == True  # noqa: E712
    assert entry_s.loc[pd.Timestamp("2024-06-14")] == True  # noqa: E712
    assert exit_s.loc[pd.Timestamp("2024-06-21")] == True  # noqa: E712
    # Entry is raised on EVERY bar from the entry flag through the bar
    # immediately before the exit (a continuous range -- see
    # opex_window_decisions' own docstring) -- five business days
    # 01-12..01-18 for January (Fri through Thu), five for June
    # (Fri 06-14..Thu 06-20). Exit is a single bar.
    jan_window = dates[(dates >= "2024-01-08") & (dates <= "2024-01-24")]
    assert entry_s.loc[jan_window].sum() == 5
    assert exit_s.loc[jan_window].sum() == 1
    jun_window = dates[(dates >= "2024-06-10") & (dates <= "2024-06-26")]
    assert entry_s.loc[jun_window].sum() == 5
    assert exit_s.loc[jun_window].sum() == 1


def test_opex_window_decisions_exit_lands_on_opex_friday_itself():
    dates = _calendar(start="2024-08-01", end="2024-10-31")
    entry_s, exit_s = mod.opex_window_decisions(dates)
    exit_dates = exit_s.index[exit_s.to_numpy()]
    assert pd.Timestamp("2024-09-20") in exit_dates  # Sep 2024 third Friday


def test_month_with_insufficient_prior_history_is_skipped_not_crashed():
    # Data starts 2024-01-17 (Wed), only 3 bars exist on/before Jan's
    # third Friday (2024-01-19) -- not enough for a 4-session-back entry
    # bar (the explicit "position would be negative" guard). Feb is
    # confirmed normally (data runs through Feb 2024).
    dates = _calendar(start="2024-01-17", end="2024-02-29")
    entry_s, exit_s = mod.opex_window_decisions(dates)
    assert not entry_s.loc[:"2024-01-31"].any()
    assert not exit_s.loc[:"2024-01-31"].any()
    # Feb 2024 third Friday=2024-02-16; entry bar (-4 sessions)=2024-02-12
    # (Mon); entry FLAG (one bar earlier, previous Friday)=2024-02-09.
    assert entry_s.loc[pd.Timestamp("2024-02-09")] == True  # noqa: E712
    assert exit_s.loc[pd.Timestamp("2024-02-16")] == True  # noqa: E712


def test_month_with_no_bar_before_entry_bar_is_skipped():
    # Data starts EXACTLY on Jan's entry bar (2024-01-15, Monday): the
    # entry bar position itself is non-negative (it's bar 0), but there is
    # no bar before it to raise the one-bar-early entry flag on, so the
    # whole month must be skipped (the entry flag needs a preceding bar).
    dates = _calendar(start="2024-01-15", end="2024-02-29")
    entry_s, exit_s = mod.opex_window_decisions(dates)
    assert not entry_s.loc[:"2024-01-31"].any()
    assert not exit_s.loc[:"2024-01-31"].any()
    assert entry_s.loc[pd.Timestamp("2024-02-09")] == True  # noqa: E712
    assert exit_s.loc[pd.Timestamp("2024-02-16")] == True  # noqa: E712


def test_trailing_unconfirmed_month_produces_no_signal():
    # Holdout-blindness guard: data ends 2024-10-10, BEFORE October's own
    # third Friday (2024-10-18). Without a bar strictly after 2024-10-18
    # confirming the search, naively treating the last available bar
    # (2024-10-10) as "the last bar <= third Friday" would fabricate a
    # phantom October opex-reference bar out of running out of data, not
    # out of the real calendar event having happened -- and that phantom
    # signal would retroactively change (once more data arrived) if this
    # same bucket were later re-run with a longer frame, breaking
    # holdout-blindness. Sep (confirmed by Oct bars) must still fire.
    dates = _calendar(start="2024-01-02", end="2024-10-10")
    entry_s, exit_s = mod.opex_window_decisions(dates)
    assert exit_s.loc[pd.Timestamp("2024-10-10")] == False  # noqa: E712
    assert not entry_s.loc["2024-10-01":].any()
    assert not exit_s.loc["2024-10-01":].any()
    assert exit_s.loc[pd.Timestamp("2024-09-20")] == True  # noqa: E712
    assert entry_s.loc[pd.Timestamp("2024-09-13")] == True  # noqa: E712
    # Exactly 9 confirmed months (Jan..Sep), none for the trailing partial
    # October. Entry is a 5-bar continuous range per confirmed month,
    # so 9 months * 5 bars = 45; exit is one bar per month.
    assert entry_s.sum() == 45
    assert exit_s.sum() == 9


def test_opex_window_decisions_empty():
    entry_s, exit_s = mod.opex_window_decisions(pd.DatetimeIndex([]))
    assert entry_s.tolist() == []
    assert exit_s.tolist() == []


def _contiguous_true_blocks(mask: np.ndarray) -> list:
    """[(start_pos, end_pos_inclusive), ...] for each maximal run of True
    values in a 1-D boolean array, in order."""
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(positions) - 1]))
    return [(positions[s], positions[e]) for s, e in zip(starts, ends)]


def test_windows_never_overlap_across_a_full_year():
    dates = _calendar(start="2020-01-02", end="2021-06-30")
    entry_s, exit_s = mod.opex_window_decisions(dates)
    # Entry is a contiguous multi-bar range per confirmed month,
    # so exactly one BLOCK of entry bars per exit, each
    # block ending immediately before its own month's exit bar, blocks
    # never touching each other, and months spaced far enough apart that
    # no block bleeds into the next month's window.
    entry_blocks = _contiguous_true_blocks(entry_s.to_numpy())
    exit_positions = np.flatnonzero(exit_s.to_numpy())
    assert len(entry_blocks) == len(exit_positions)
    block_ends = np.array([end for _, end in entry_blocks])
    block_starts = np.array([start for start, _ in entry_blocks])
    # Each block ends exactly one bar before its own month's exit (the
    # entry range is [entry_flag_pos, opex_pos) -- see opex_window_
    # decisions' own docstring).
    assert (exit_positions - block_ends == 1).all()
    # Each block is a handful of sessions wide (the ~4-5 session hold),
    # never a multi-week span that would suggest two months' ranges merged.
    assert ((block_ends - block_starts) <= 6).all()
    if len(entry_blocks) > 1:
        assert (np.diff(block_starts) >= 15).all()
        # No block starts before the PRIOR month's own exit bar (strict
        # non-overlap between consecutive months' windows).
        assert (block_starts[1:] > exit_positions[:-1]).all()


# ------------------------------------------------------------------ factory

def test_signal_fn_matches_opex_window_decisions_over_full_history():
    index = _calendar(start="2018-01-02", end="2020-12-31")
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])
    entry_s, exit_s = mod.opex_window_decisions(index)
    for i, (date, sig) in enumerate(zip(index, sigs)):
        assert sig == SignalResult(entry=bool(entry_s.loc[date]),
                                    exit=bool(exit_s.loc[date])), (i, date)


def test_price_never_enters_the_signal():
    index = _calendar(start="2019-01-02", end="2020-12-31")
    a = mod.make_signal_fn("SPY", index, _frames(index, seed=1))
    b = mod.make_signal_fn("SPY", index, _frames(index, seed=999))
    frame_a = _frames(index, seed=1)["SPY"]
    frame_b = _frames(index, seed=999)["SPY"]
    assert _signals(a, frame_a) == _signals(b, frame_b)


def test_non_finite_close_does_not_affect_signal():
    index = _calendar(start="2019-01-02", end="2019-12-31")
    clean = _frames(index, seed=2)
    dirty = {s: f.copy() for s, f in clean.items()}
    dirty["SPY"].iloc[10:20, dirty["SPY"].columns.get_loc("Close")] = np.nan
    dirty["SPY"].iloc[100, dirty["SPY"].columns.get_loc("Close")] = np.inf
    fn_clean = mod.make_signal_fn("SPY", index, clean)
    fn_dirty = mod.make_signal_fn("SPY", index, dirty)
    assert _signals(fn_clean, clean["SPY"]) == _signals(fn_dirty, dirty["SPY"])


def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(start="2017-01-03", end="2019-12-31")
    frames = _frames(index)
    k = 300
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
    # A bucket ending 2019-10-10 (BEFORE October's own third Friday) must
    # produce identical in-bucket signals whether the caller hands
    # make_signal_fn a frame that stops exactly at the bucket boundary or
    # one that continues on for another year -- the internal truncation
    # step (and the trailing-unconfirmed-month guard) must fully absorb
    # the extra future dates, not just extra future prices.
    index = _calendar(start="2018-01-02", end="2020-12-31")
    frames = _frames(index)
    bucket = index[index <= pd.Timestamp("2019-10-10")]
    fn_full = mod.make_signal_fn("SPY", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("SPY", bucket, truncated)
    bucket_frame = frames["SPY"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(start="2019-01-02", end="2019-06-30")
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {})
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {"SPY": frames["SPY"].drop(columns=["Close"])})


def test_factory_rejects_empty_or_unsorted_index():
    frames = _frames(_calendar(start="2019-01-02", periods=100))
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", pd.DatetimeIndex([]), frames)
    bad = _calendar(start="2019-01-02", periods=100)[::-1]
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", bad, frames)


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2008-01-02", periods=5000)
    frames = _frames(index, seed=11)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    frame = frames["SPY"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_roughly_monthly_trades():
    index = _calendar(start="2010-01-04", end="2019-12-31")  # 10 years
    frames = _frames(index, seed=7)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    trades, in_market = run_backtest(frames["SPY"], signal_fn, cost_bps=1.0)
    # ~10 years * 12 months, minus early-history guard months and the
    # trailing unconfirmed month.
    assert len(trades) >= 100
    for _, row in trades.iterrows():
        if row["reason"] == "signal_exit":
            # Entry (next-open) to exit (opex Friday close) is a handful
            # of trading sessions -- never a multi-week hold.
            assert 3 <= row["held"] <= 8, row
    assert in_market.any()
    assert not in_market.all()  # most of the month must be flat


def test_end_to_end_holdout_bucket_late_start_still_finds_its_own_windows():
    full_index = _calendar(start="2015-01-02", end="2016-12-30")
    frames = _frames(full_index, seed=3)
    bucket = full_index[full_index >= pd.Timestamp("2016-01-01")]
    assert len(bucket) > 100
    signal_fn = mod.make_signal_fn("SPY", bucket, frames)
    bucket_frame = frames["SPY"].loc[bucket]
    trades, _ = run_backtest(bucket_frame, signal_fn, cost_bps=0.0)
    assert len(trades) >= 8  # most months of 2016 covered


def test_end_to_end_bucket_starting_mid_window_still_enters_the_open_trade():
    """Guards against a single-bar entry flag: if opex_window_decisions
    raised entry=True on exactly ONE bar per month (entry_flag_pos), a
    bucket that starts AFTER that bar but BEFORE the month's opex exit
    would never see an entry flag at all -- the engine would stay flat all
    month, then (harmlessly, with no open position) ignore the exit flag,
    silently missing a real window the un-truncated data does contain. The
    "late start" test above does not cover this because its bucket starts
    on 2016-01-01, before every month's own entry flag -- every window
    there is seen in full.

    Ground truth (hand-verified with stdlib datetime, matching this file's
    own module docstring table): January 2024's third Friday is 2024-01-19,
    entry bar 2024-01-15 (Mon), entry flag 2024-01-12 (Fri). A bucket
    starting 2024-01-16 (the business day AFTER the entry bar, so the
    entry-flag bar 01-12 is entirely outside it) must still open a
    position on its own first available bar and exit at the 01-19 close --
    it must NOT stay flat and skip January entirely."""
    full_index = _calendar(start="2023-11-01", end="2024-02-29")
    frames = _frames(full_index, seed=11)
    bucket = full_index[full_index >= pd.Timestamp("2024-01-16")]
    assert bucket[0] == pd.Timestamp("2024-01-16")
    assert pd.Timestamp("2024-01-19") in bucket  # the opex exit bar is visible
    assert pd.Timestamp("2024-01-12") not in bucket  # the entry-flag bar is NOT
    signal_fn = mod.make_signal_fn("SPY", bucket, frames)
    bucket_frame = frames["SPY"].loc[bucket]
    trades, in_market = run_backtest(bucket_frame, signal_fn, cost_bps=0.0)
    january_trades = trades[trades["exit_date"] <= pd.Timestamp("2024-01-22")]
    assert len(january_trades) == 1, (
        "the January window must still be captured even though the bucket "
        "starts after the month's entry-flag bar")
    row = january_trades.iloc[0]
    assert row["reason"] == "signal_exit"
    assert row["exit_date"] == pd.Timestamp("2024-01-19")
    assert bool(in_market.loc[pd.Timestamp("2024-01-17")])
