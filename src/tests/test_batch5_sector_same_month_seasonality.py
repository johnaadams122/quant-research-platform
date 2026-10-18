"""Tests for src/mining/batch5/sector_same_month_seasonality.py.

Synthetic data only (no network): each pool member's closes are built so
that its total return in every calendar month is a known constant, which
makes the expected same-month ranking computable by hand in the test.
"""
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
from mining.batch5 import sector_same_month_seasonality as mod  # noqa: E402

POOL = list(mod.TRADED)
DEFAULT_START = "2005-01-03"   # a Monday; January 2005 is therefore INCOMPLETE
DEFAULT_END = "2016-12-30"


# ------------------------------------------------------------ synthetic data

def _month_return_table() -> dict:
    """ticker -> month -> constant total return for that calendar month.
    Ticker j is 'favored' in month (j % 12) + 1; every other month carries a
    small, ticker- and month-specific return so rankings have no ties."""
    table = {}
    for j, ticker in enumerate(POOL):
        favored = (j % 12) + 1
        table[ticker] = {
            month: 0.001 * ((j * 7 + month * 3) % 11) + (0.05 if month == favored else 0.0)
            for month in range(1, 13)}
    return table


def _build_frame(start: str, end: str, month_ret) -> pd.DataFrame:
    """Business-day OHLC whose total return in each calendar month equals
    month_ret(year, month) exactly (constant daily factor within the month)."""
    index = pd.bdate_range(start, end)
    periods = index.to_period("M")
    close = np.empty(len(index))
    level = 100.0
    for period in periods.unique():
        mask = np.asarray(periods == period, dtype=bool)
        nbars = int(mask.sum())
        step = (1.0 + month_ret(period.year, period.month)) ** (1.0 / nbars)
        values = level * step ** np.arange(1, nbars + 1)
        close[mask] = values
        level = values[-1]
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=index)


def _frames(starts=None, end=DEFAULT_END, table=None) -> dict:
    table = table if table is not None else _month_return_table()
    starts = starts or {}
    return {
        ticker: _build_frame(starts.get(ticker, DEFAULT_START), end,
                             lambda y, m, t=ticker: table[t][m])
        for ticker in POOL}


def _expected_top3(table: dict, month: int, tickers=POOL, starts=None) -> tuple:
    starts = starts or {}
    ranked = sorted(
        tickers,
        key=lambda t: (-table[t][month], pd.Timestamp(starts.get(t, DEFAULT_START)), t))
    return tuple(ranked[:3])


def _bucket(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index <= end)].copy()


def _signals(signal_fn, df: pd.DataFrame, upto: int) -> list:
    """Call signal_fn exactly the way run_backtest does, for i = 0..upto."""
    return [(bool(s.entry), bool(s.exit))
            for s in (signal_fn(df.iloc[:i + 1], i) for i in range(upto + 1))]


# ----------------------------------------------------------- month-end flags

def test_month_end_flags_use_dates_and_calendar_for_last_bar():
    dates = pd.bdate_range("2026-01-02", "2026-03-31")  # inside the vendored calendar
    flags = mod.month_end_flags(dates)
    flagged = list(dates[flags].strftime("%Y-%m-%d"))
    # 2026-03-31 is the final bar: resolved via the NYSE calendar (Apr 1 is a session)
    assert flagged == ["2026-01-30", "2026-02-27", "2026-03-31"]

    outside = pd.bdate_range("2016-11-01", "2016-12-30")  # before calendar coverage
    flags_out = mod.month_end_flags(outside)
    assert list(outside[flags_out].strftime("%Y-%m-%d")) == ["2016-11-30"]
    assert not flags_out[-1]  # final bar outside coverage: not a month end


# ----------------------------------------------------------------- schedule

def test_schedule_ranks_top_three_by_same_month_history():
    table = _month_return_table()
    frames = _frames(table=table)
    schedule = mod.build_schedule(frames, pd.Timestamp(DEFAULT_END))
    for month in range(1, 13):
        assert schedule[(2012, month)] == _expected_top3(table, month), month
    # XLB (j=0) is favored in January and must be the January leader.
    assert schedule[(2012, 1)][0] == "XLB"
    # January 2005 is incomplete (first bar 2005-01-03 > 2005-01-01), so the
    # fifth January observation only arrives with January 2010 -> first
    # tradable January is 2011; February has 2005..2009 -> tradable from 2010.
    assert schedule[(2010, 1)] == ()
    assert schedule[(2011, 1)] == _expected_top3(table, 1)
    assert schedule[(2010, 2)] == _expected_top3(table, 2)
    assert schedule[(2009, 12)] == ()   # only 2005..2008 = 4 observations


def test_schedule_requires_nine_eligible_members():
    table = _month_return_table()
    late = {"XLRE": "2013-01-02", "XLC": "2013-01-02"}
    nine_eligible = mod.build_schedule(_frames(starts=late, table=table),
                                       pd.Timestamp(DEFAULT_END))
    founders = [t for t in POOL if t not in late]
    assert nine_eligible[(2012, 6)] == _expected_top3(table, 6, tickers=founders)

    late_three = {**late, "XLY": "2013-01-02"}
    eight_eligible = mod.build_schedule(_frames(starts=late_three, table=table),
                                        pd.Timestamp(DEFAULT_END))
    assert eight_eligible[(2012, 6)] == ()
    assert all(v == () for k, v in eight_eligible.items() if k < (2018, 1))

    # Once the late members have five complete same-month observations
    # (2013..2017 -> 2018) every member competes again.
    schedule_2019 = mod.build_schedule(_frames(starts=late_three, table=table,
                                               end="2019-12-31"),
                                       pd.Timestamp("2019-12-31"))
    assert schedule_2019[(2018, 6)] == _expected_top3(table, 6, starts=late_three)


def test_tie_break_longer_history_then_alphabetical():
    table = _month_return_table()
    for ticker in POOL:
        table[ticker][6] = 0.01   # every member ties in June
    starts = {"XLB": "2004-01-02"}  # XLB has the longest history
    schedule = mod.build_schedule(_frames(starts=starts, table=table),
                                  pd.Timestamp(DEFAULT_END))
    assert schedule[(2012, 6)] == ("XLB", "XLC", "XLE")


# --------------------------------------------------------- per-bar signals

def test_signal_fires_on_month_end_for_selected_and_stays_flat_otherwise():
    table = _month_return_table()
    frames = _frames(table=table)
    selected = _expected_top3(table, 6)
    winner = selected[0]
    loser = next(t for t in POOL if t not in selected)
    bucket = _bucket(frames[winner], "2011-01-01", "2013-12-31")
    index = bucket.index

    fn_win = mod.make_signal_fn(winner, index, frames)
    fn_lose = mod.make_signal_fn(loser, index, frames)

    may_end = index.get_loc(pd.Timestamp("2012-05-31"))
    # Month end: exit is unconditional; entry iff selected for the coming month.
    assert fn_win(bucket.iloc[:may_end + 1], may_end) == backtest_engine.SignalResult(
        entry=True, exit=True)
    loser_bucket = _bucket(frames[loser], "2011-01-01", "2013-12-31")
    assert fn_lose(loser_bucket.iloc[:may_end + 1], may_end) == backtest_engine.SignalResult(
        entry=False, exit=True)

    mid = index.get_loc(pd.Timestamp("2012-06-15"))
    assert fn_win(bucket.iloc[:mid + 1], mid) == backtest_engine.SignalResult()
    assert fn_lose(loser_bucket.iloc[:mid + 1], mid) == backtest_engine.SignalResult()

    # Every non-month-end bar is flat for everybody. Month ends are derived
    # from the FULL frame's dates (the bucket's last bar has a successor there).
    full_dates = frames[winner].index
    full_flags = pd.Series(mod.month_end_flags(full_dates), index=full_dates)
    flags = full_flags.reindex(index).to_numpy(dtype=bool)
    assert flags[-1]  # 2013-12-31 is a month end because 2014-01-01 follows it
    signals = _signals(fn_win, bucket, len(index) - 1)
    for i, (entry, exit_) in enumerate(signals):
        if not flags[i]:
            assert (entry, exit_) == (False, False), index[i]
        else:
            assert exit_, index[i]  # month end: exit is unconditional
            selected = winner in fn_win.schedule.get(
                mod._next_month(index[i].year, index[i].month), ())
            assert entry == selected, index[i]


def test_signal_fn_rejects_a_misaligned_bar():
    frames = _frames()
    bucket = _bucket(frames["XLB"], "2011-01-01", "2013-12-31")
    fn = mod.make_signal_fn("XLB", bucket.index, frames)
    with pytest.raises(ValueError):
        fn(bucket.iloc[:11], 5)  # frame ends at position 10, caller claims 5


def test_make_signal_fn_requires_every_pool_member_frame():
    frames = _frames()
    frames.pop("XLC")
    with pytest.raises(ValueError, match="XLC"):
        mod.make_signal_fn("XLB", frames["XLB"].index, frames)


# ------------------------------------------------------------- look-ahead

def _perturb_after(frames: dict, cutoff: pd.Timestamp, factors: dict) -> dict:
    out = {}
    for ticker, frame in frames.items():
        frame = frame.copy()
        after = frame.index > cutoff
        factor = factors.get(ticker, 1.0)
        frame.loc[after, "Close"] *= factor
        frame.loc[after, "Open"] *= factor
        frame.loc[after, "High"] = frame.loc[after, "High"] * factor + 5.0
        frame.loc[after, "Low"] = frame.loc[after, "Low"] * factor - 1.0
        out[ticker] = frame
    return out


@pytest.mark.parametrize("ticker", ["XLB", "XLK", "XLRE"])
def test_lookahead_probe_future_bars_never_change_past_signals(ticker):
    frames = _frames()
    bucket = _bucket(frames[ticker], "2011-01-01", "2013-12-31")
    index = bucket.index
    k = index.get_loc(pd.Timestamp("2012-06-15"))

    before = _signals(mod.make_signal_fn(ticker, index, frames), bucket, k)

    perturbed = _perturb_after(frames, index[k], {t: 1.3 for t in POOL})
    bucket_p = _bucket(perturbed[ticker], "2011-01-01", "2013-12-31")
    after = _signals(mod.make_signal_fn(ticker, index, perturbed), bucket_p, k)
    assert before == after
    assert any(entry for entry, _ in before)  # the probe covered real signals


def test_lookahead_probe_bites_future_only_schedules():
    """Perturbing ONE member's future changes rankings twelve months later
    (proving the schedule really reads the closes) but nothing on or before k."""
    frames = _frames()
    ticker = "XLB"
    bucket = _bucket(frames[ticker], "2011-01-01", "2013-12-31")
    index = bucket.index
    k = index.get_loc(pd.Timestamp("2012-06-15"))
    fn_before = mod.make_signal_fn(ticker, index, frames)
    perturbed = _perturb_after(frames, index[k], {"XLU": 1.3})
    fn_after = mod.make_signal_fn(ticker, index, perturbed)
    assert _signals(fn_before, bucket, k) == _signals(fn_after, bucket, k)
    sched_before, sched_after = fn_before.schedule, fn_after.schedule
    assert all(sched_before[key] == sched_after[key] for key in sched_before
               if key <= (2012, 7))
    assert sched_before[(2013, 6)] != sched_after[(2013, 6)]


# ------------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_monthly_trades():
    table = _month_return_table()
    frames = _frames(table=table)
    ticker = "XLB"   # favored in January
    bucket = _bucket(frames[ticker], "2014-07-01", DEFAULT_END)  # ~650 bars
    assert 600 <= len(bucket) <= 700
    signal_fn = mod.make_signal_fn(ticker, bucket.index, frames)
    trades, in_market = backtest_engine.run_backtest(bucket, signal_fn)
    assert len(trades) >= 1
    periods = bucket.index.to_period("M")
    for _, trade in trades.iterrows():
        entry_pos = bucket.index.get_loc(trade["entry_date"])
        assert entry_pos > 0 and periods[entry_pos] != periods[entry_pos - 1]
        if trade["reason"] == "signal_exit":
            exit_pos = bucket.index.get_loc(trade["exit_date"])
            assert periods[exit_pos + 1] != periods[exit_pos]
    entered_months = {pd.Timestamp(d).month for d in trades["entry_date"]}
    assert 1 in entered_months
    assert in_market.any()


def test_consecutive_selection_is_one_trade_per_selected_month():
    """The family's exit is UNCONDITIONAL at
    every month end, so a sector selected two months running yields two
    round trips (exit at the first month's last close, re-entry at the next
    open), never one merged holding period. The engine supports this: once
    flat at bar i it re-reads bar i-1's signal, so entry=True/exit=True on
    a month-end bar exits there and re-enters at the next open."""
    table = _month_return_table()
    for ticker in POOL:
        table[ticker][3] = 0.0
        table[ticker][4] = 0.0
    table["XLB"][3] = 0.05
    table["XLB"][4] = 0.05
    frames = _frames(table=table)
    schedule = mod.build_schedule(frames, pd.Timestamp(DEFAULT_END))
    assert "XLB" in schedule[(2012, 3)] and "XLB" in schedule[(2012, 4)]
    assert "XLB" not in schedule[(2012, 5)]

    bucket = _bucket(frames["XLB"], "2012-01-01", "2012-12-31")
    signal_fn = mod.make_signal_fn("XLB", bucket.index, frames)
    # The exit flag is exactly the month-end flag, independent of selection.
    assert np.array_equal(signal_fn.exit_flags, signal_fn.last_of_month)
    mar_end = bucket.index.get_loc(pd.Timestamp("2012-03-30"))
    assert signal_fn(bucket.iloc[:mar_end + 1], mar_end) == backtest_engine.SignalResult(
        entry=True, exit=True)

    trades, _ = backtest_engine.run_backtest(bucket, signal_fn)
    march = trades[trades["entry_date"] == pd.Timestamp("2012-03-01")]
    april = trades[trades["entry_date"] == pd.Timestamp("2012-04-02")]
    assert len(march) == 1 and len(april) == 1
    assert march.iloc[0]["exit_date"] == pd.Timestamp("2012-03-30")
    assert march.iloc[0]["reason"] == "signal_exit"
    assert april.iloc[0]["exit_date"] == pd.Timestamp("2012-04-30")
    # No trade ever spans a month boundary.
    periods = bucket.index.to_period("M")
    for _, trade in trades.iterrows():
        assert (periods[bucket.index.get_loc(trade["entry_date"])]
                == periods[bucket.index.get_loc(trade["exit_date"])]), trade


def test_no_trades_while_fewer_than_nine_members_are_eligible():
    late = {"XLRE": "2013-01-02", "XLC": "2013-01-02", "XLY": "2013-01-02"}
    frames = _frames(starts=late)
    bucket = _bucket(frames["XLB"], "2011-01-01", "2012-12-31")
    signal_fn = mod.make_signal_fn("XLB", bucket.index, frames)
    trades, in_market = backtest_engine.run_backtest(bucket, signal_fn)
    assert len(trades) == 0
    assert not in_market.any()
