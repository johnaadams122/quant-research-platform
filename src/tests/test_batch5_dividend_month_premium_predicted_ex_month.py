"""Tests for src/mining/batch5/dividend_month_premium_predicted_ex_month.py.

Synthetic frames and hand-built ex-dividend calendars only (no network,
deterministic). Wherever the factory would read the ex-dividend events file,
the loader is monkeypatched with a hand-built calendar, so no data file is read.
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

from mining import backtest_engine  # noqa: E402
from mining.batch5 import dividend_month_premium_predicted_ex_month as mod  # noqa: E402

FAMILY_ID = "dividend_month_premium_predicted_ex_month"


# ------------------------------------------------------------ synthetic data

def _frame(index: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(index)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _bdays(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _signals(signal_fn, frame: pd.DataFrame, upto=None):
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


def _last_bar_of_month(index: pd.DatetimeIndex, year: int, month: int):
    sub = index[(index.year == year) & (index.month == month)]
    return sub[-1]


def _first_bar_of_month(index: pd.DatetimeIndex, year: int, month: int):
    sub = index[(index.year == year) & (index.month == month)]
    return sub[0]


# --------------------------------------------------------------- predictions

def test_predicted_months_from_quarterly_calendar():
    ex = frozenset(pd.to_datetime(["2015-02-10", "2015-05-12", "2015-08-11",
                                   "2015-11-10"]))
    # 2016-02 is predicted via m-3 (Nov), m-6 (Aug), m-9 (May), m-12 (Feb)
    assert mod.is_predicted_month(ex, pd.Period("2016-02", "M"))
    # 2016-01 has ex-dates in none of Oct-15, Jul-15, Apr-15, Jan-15
    assert not mod.is_predicted_month(ex, pd.Period("2016-01", "M"))
    # 2015-05 is predicted by 2015-02 (m-3) alone
    assert mod.is_predicted_month(ex, pd.Period("2015-05", "M"))
    # 2015-03: m-3 = 2014-12, nothing there
    assert not mod.is_predicted_month(ex, pd.Period("2015-03", "M"))
    # annual payer: 2016-02 predicted from 2015-02 via m-12 only
    annual = frozenset(pd.to_datetime(["2015-02-10"]))
    assert mod.is_predicted_month(annual, pd.Period("2016-02", "M"))
    assert not mod.is_predicted_month(annual, pd.Period("2016-03", "M"))
    # a m-1 or m-2 dividend does NOT predict (only 3/6/9/12)
    assert not mod.is_predicted_month(annual, pd.Period("2015-03", "M"))
    assert not mod.is_predicted_month(annual, pd.Period("2015-04", "M"))
    # month arithmetic crosses the year boundary correctly
    dec = frozenset(pd.to_datetime(["2015-12-15"]))
    assert mod.is_predicted_month(dec, pd.Period("2016-03", "M"))
    assert mod.is_predicted_month(dec, pd.Period("2016-12", "M"))


# ------------------------------------------------------------- rule firing

def _quarterly_ex_dates(start_year=2014, end_year=2018):
    """Deterministic quarterly payer: ex-dates on the 15th of Feb/May/Aug/Nov."""
    out = []
    for year in range(start_year, end_year + 1):
        for month in (2, 5, 8, 11):
            out.append(pd.Timestamp(year=year, month=month, day=15))
    return frozenset(out)


def test_rule_fires_at_month_end_before_predicted_month_and_exits_on_ex_day():
    index = _bdays("2014-01-01", "2016-12-30")
    frame = _frame(index)
    ex = _quarterly_ex_dates()
    ex_in_frame = frozenset(d for d in ex if d in index)
    signal_fn = mod.build_signal_fn(frame, index, ex_in_frame)
    sigs = _signals(signal_fn, frame)
    by_date = dict(zip(index, sigs))

    # Last bar of Jan-2016: Feb-2016 is predicted (Nov/Aug/May/Feb-2015) -> entry
    assert by_date[_last_bar_of_month(index, 2016, 1)].entry is True
    # Last bar of Dec-2015: Jan-2016 predicted? m-3=Oct, m-6=Jul, m-9=Apr,
    # m-12=Jan-2015 -> none -> no entry
    assert by_date[_last_bar_of_month(index, 2015, 12)].entry is False
    # Mid-month bars never carry an entry
    mid = index[(index.year == 2016) & (index.month == 1)][5]
    assert by_date[mid].entry is False and by_date[mid].exit is False
    # Ex-day inside the predicted month -> exit; the bar before it does not
    ex_day = pd.Timestamp("2016-02-15")  # a Monday
    assert ex_day in index
    assert by_date[ex_day].exit is True
    pos = index.get_loc(ex_day)
    assert by_date[index[pos - 1]].exit is False
    # Month-end bars always carry exit=True (the "no ex-day" fallback)
    assert by_date[_last_bar_of_month(index, 2016, 2)].exit is True
    assert by_date[_last_bar_of_month(index, 2016, 3)].exit is True
    # Before the first ex-date + 3 months nothing is predicted -> flat
    for d in index[index < pd.Timestamp("2014-04-30")]:
        assert by_date[d].entry is False


def test_no_ex_dates_means_no_entries_ever():
    index = _bdays("2014-01-01", "2016-12-30")
    frame = _frame(index)
    signal_fn = mod.build_signal_fn(frame, index, frozenset())
    assert all(s.entry is False for s in _signals(signal_fn, frame))


def test_ex_date_on_non_trading_day_maps_to_next_bar_in_same_month():
    index = _bdays("2016-01-01", "2016-06-30")
    frame = _frame(index)
    # 2016-04-16 is a Saturday -> exit should fire on Monday 2016-04-18
    ex = frozenset([pd.Timestamp("2016-04-16")])
    signal_fn = mod.build_signal_fn(frame, index, ex)
    by_date = dict(zip(index, _signals(signal_fn, frame)))
    assert by_date[pd.Timestamp("2016-04-18")].exit is True
    assert by_date[pd.Timestamp("2016-04-15")].exit is False
    # an ex-date after the frame's last bar is silently ignored
    ex_late = frozenset([pd.Timestamp("2016-07-05")])
    signal_fn2 = mod.build_signal_fn(frame, index, ex_late)
    sigs = _signals(signal_fn2, frame)
    exits = [d for d, s in zip(index, sigs) if s.exit]
    month_ends = [_last_bar_of_month(index, 2016, m) for m in range(1, 7)]
    assert exits == month_ends


def test_holdout_bucket_is_blind_to_bars_after_it_but_uses_prior_history():
    full_index = _bdays("2014-01-01", "2017-12-29")
    frame = _frame(full_index)
    ex = _quarterly_ex_dates(2014, 2017)
    bucket = full_index[(full_index >= "2016-03-01") & (full_index <= "2016-08-31")]
    bucket_frame = frame.loc[bucket]
    signal_fn = mod.build_signal_fn(frame.loc[frame.index <= bucket[-1]], bucket, ex)
    sigs = _signals(signal_fn, bucket_frame)
    by_date = dict(zip(bucket, sigs))
    # entry at end of Apr-2016 for May-2016 (predicted by Feb-16/Nov-15/Aug-15/May-15)
    assert by_date[_last_bar_of_month(bucket, 2016, 4)].entry is True
    # the bucket's own signal positions are indexed by the bucket, not the frame
    assert len(sigs) == len(bucket)
    # perturbing the future beyond the bucket changes nothing
    frame2 = frame.copy()
    frame2.loc[frame2.index > bucket[-1], ["Open", "High", "Low", "Close"]] *= 1.7
    signal_fn2 = mod.build_signal_fn(frame2, bucket, ex)
    sigs2 = _signals(signal_fn2, bucket_frame)
    assert [(s.entry, s.exit) for s in sigs] == [(s.entry, s.exit) for s in sigs2]


# ------------------------------------------------------------ look-ahead probe

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _bdays("2014-01-01", "2016-12-30")
    frame = _frame(index)
    ex = _quarterly_ex_dates()
    k = index.get_loc(pd.Timestamp("2015-09-30"))
    base = _signals(mod.build_signal_fn(frame, index, ex), frame, upto=k)
    perturbed = frame.copy()
    after = perturbed.index > index[k]
    perturbed.loc[after, "Close"] *= 1.3
    perturbed.loc[after, "High"] = perturbed.loc[after, "High"] * 1.5 + 10.0
    perturbed.loc[after, "Low"] = perturbed.loc[after, "Low"] * 0.5
    perturbed.loc[after, "Open"] *= 0.9
    probe = _signals(mod.build_signal_fn(perturbed, index, ex), perturbed, upto=k)
    assert [(s.entry, s.exit) for s in base] == [(s.entry, s.exit) for s in probe]


def test_signal_fn_is_cheap_per_call_and_ignores_slice_content():
    """The engine hands a growing slice; the factory precomputes everything,
    so the per-call work is a positional lookup independent of slice length."""
    index = _bdays("2014-01-01", "2016-12-30")
    frame = _frame(index)
    signal_fn = mod.build_signal_fn(frame, index, _quarterly_ex_dates())
    i = index.get_loc(_last_bar_of_month(index, 2016, 1))
    full = signal_fn(frame.iloc[:i + 1], i)
    # the same position with a garbage slice gives the same answer
    garbage = frame.iloc[:i + 1].copy()
    garbage[["Open", "High", "Low", "Close"]] = 1.0
    assert signal_fn(garbage, i) == full
    assert signal_fn(frame.iloc[:1], len(index) + 5) == backtest_engine.SignalResult()


# --------------------------------------------------------------- end to end

def test_end_to_end_backtest_produces_dividend_month_trades(monkeypatch):
    index = _bdays("2014-01-01", "2016-06-30")  # ~650 bars
    frame = _frame(index)
    frames = {"XYZQ": frame}
    ex = _quarterly_ex_dates(2014, 2016)
    monkeypatch.setattr(mod, "load_events", lambda: {"XYZQ": ex})
    signal_fn = mod.make_signal_fn("XYZQ", index, frames)
    trades, in_market = backtest_engine.run_backtest(
        frame, signal_fn, time_stop=mod.TIME_STOP_BARS)
    assert len(trades) >= 1
    # every entry is the first session of a predicted month (Feb/May/Aug/Nov)
    for entry in trades["entry_date"]:
        assert entry == _first_bar_of_month(index, entry.year, entry.month)
        assert entry.month in (2, 5, 8, 11)
    # exits are the ex-day close (the 15th or the next session) in that month
    for _, row in trades.iterrows():
        assert row["exit_date"].month == row["entry_date"].month
        assert row["exit_date"].day >= 15
        assert row["exit_date"].day <= 17
        assert row["reason"] == "signal_exit"
    # first eligible entry: May-2014, the first month predicted by an
    # ex-date 3+ months back (Feb-2014)
    assert trades["entry_date"].iloc[0] == _first_bar_of_month(index, 2014, 5)
    # 2014: May/Aug/Nov ; 2015: 4 ; 2016: Feb/May -> 9 trades
    assert len(trades) == 9
    assert (trades["held"] <= mod.TIME_STOP_BARS).all()


def test_end_to_end_without_ex_day_in_month_exits_at_month_end(monkeypatch):
    index = _bdays("2015-01-01", "2016-12-30")
    frame = _frame(index)
    # a single historical ex-date in Mar-2015 predicts Jun/Sep/Dec-2015 and
    # Mar-2016, but no realized ex-day ever occurs in those months
    ex = frozenset([pd.Timestamp("2015-03-16")])
    monkeypatch.setattr(mod, "load_events", lambda: {"ABCD": ex})
    signal_fn = mod.make_signal_fn("ABCD", index, {"ABCD": frame})
    trades, _ = backtest_engine.run_backtest(frame, signal_fn,
                                             time_stop=mod.TIME_STOP_BARS)
    entries = [(d.year, d.month) for d in trades["entry_date"]]
    assert entries == [(2015, 6), (2015, 9), (2015, 12), (2016, 3)]
    for _, row in trades.iterrows():
        assert row["exit_date"] == _last_bar_of_month(
            index, row["entry_date"].year, row["entry_date"].month)
        assert row["reason"] == "signal_exit"


def test_make_signal_fn_unknown_ticker_is_flat(monkeypatch):
    index = _bdays("2015-01-01", "2015-12-31")
    frame = _frame(index)
    monkeypatch.setattr(mod, "load_events", lambda: {})
    signal_fn = mod.make_signal_fn("NOPE", index, {"NOPE": frame})
    assert all(s.entry is False for s in _signals(signal_fn, frame))


def test_make_signal_fn_requires_frame():
    with pytest.raises(KeyError):
        mod.make_signal_fn("AAPL", _bdays("2015-01-01", "2015-03-01"), {})
