"""Tests for src/mining/batch5/fx_dollar_index_trend_following.py.

Synthetic frames only (no network, deterministic); the ledger and catalog
are never touched.
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
from mining.batch5 import fx_dollar_index_trend_following as mod  # noqa: E402

CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")


# ------------------------------------------------------------ fixtures

def _flat_month_frame(monthly_closes, start="2015-01-01") -> pd.DataFrame:
    """Daily OHLC frame where every bar of calendar month k carries the
    constant close ``monthly_closes[k]`` (Open=High=Low=Close), so the
    month-end closes are exactly the hand-typed numbers."""
    months = pd.period_range(start=start, periods=len(monthly_closes), freq="M")
    dates, closes = [], []
    for period, close in zip(months, monthly_closes):
        days = pd.bdate_range(period.start_time, period.end_time)
        dates.extend(days)
        closes.extend([float(close)] * len(days))
    index = pd.DatetimeIndex(dates)
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": values, "High": values, "Low": values,
                         "Close": values}, index=index)


def _random_walk_frame(n_bars: int, seed: int = 7, start="2018-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_bars)
    close = 25.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, n_bars)))
    open_ = close * (1 + rng.normal(0, 0.001, n_bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n_bars))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=index)


def _month_end_positions(index: pd.DatetimeIndex) -> list:
    """Positions of the last bar of every calendar month, INCLUDING the
    frame's final bar (so the list has one entry per calendar month)."""
    periods = index.to_period("M")
    return [i for i in range(len(index) - 1) if periods[i] != periods[i + 1]] + [len(index) - 1]


def _signals(frame: pd.DataFrame, index: pd.DatetimeIndex, ticker="UUP"):
    signal_fn = mod.make_signal_fn(ticker, index, {ticker: frame})
    entries = np.array([signal_fn(frame.loc[:index[i]], i).entry for i in range(len(index))])
    exits = np.array([signal_fn(frame.loc[:index[i]], i).exit for i in range(len(index))])
    return entries, exits


# Hand-built monthly path. Month k's trailing-12 comparison is against month
# k-12; the first decision is possible at month 12 (needs month 0).
#   month 12: 26.0 vs 25.0 -> +4%  LONG  (buy at the NEXT session's open)
#   month 13: 25.5 vs 25.0 -> +2%  LONG  (hold)
#   month 14: 24.0 vs 25.0 -> -4%  CASH  (sell at THIS month-end close)
#   month 15: 25.0 vs 25.0 ->  0%  CASH  (zero is not positive)
#   month 16: 27.0 vs 25.0 -> +8%  LONG  (buy at the NEXT session's open)
#   months 17..22: 27.0 vs 25.0 -> +8%  LONG (hold)
#   month 23: the frame's final bar -> no new decision (standing LONG)
MONTHLY = [25.0] * 12 + [26.0, 25.5, 24.0, 25.0, 27.0] + [27.0] * 7


# --------------------------------------------------------- rule fires

def test_rule_fires_at_month_ends_where_it_should_and_stays_flat_elsewhere():
    frame = _flat_month_frame(MONTHLY)
    index = frame.index
    entries, exits = _signals(frame, index)
    periods = index.to_period("M")
    month_ends = _month_end_positions(index)
    assert len(month_ends) == len(MONTHLY) == 24

    # No decision before month 12's month-end: everything flat.
    first_decision = month_ends[12]
    assert not entries[:first_decision].any()
    assert not exits[:first_decision].any()

    # Month-end targets: LONG at 12, 13, 16..23; CASH at 14, 15.
    expected_long = {12: True, 13: True, 14: False, 15: False, 16: True}
    for month, is_long in expected_long.items():
        assert bool(entries[month_ends[month]]) is is_long, month
    for month in range(17, 23):          # 27.0 vs <= 25.0 -> still positive
        assert entries[month_ends[month]], month
    # month 23 is the frame's final bar: no NEW decision is made there (the
    # calendar cannot tell whether it is a real month-end without a next
    # bar), so it carries month 22's standing LONG and never an exit.
    assert entries[month_ends[23]]
    assert not exits[month_ends[23]]
    assert np.isnan(mod.month_end_targets(frame["Close"], 12).iloc[-1])

    # The entry flag persists through the following month while the
    # standing target is LONG; a CASH month (15) has no entry flag anywhere.
    month15 = np.flatnonzero(periods == periods[month_ends[15]])
    assert not entries[month15].any()
    month13 = np.flatnonzero(periods == periods[month_ends[13]])
    assert entries[month13].all()

    # Exit fires ON the CASH month-end decision bar (sell at that close),
    # never on the bar after it and never on a LONG month-end.
    assert exits[month_ends[14]]
    assert exits[month_ends[15]]
    assert not exits[month_ends[14] + 1]
    assert not exits[month_ends[12]]
    assert not exits[month_ends[16]]
    # Exits never fire inside a month (only month-end bars decide).
    inside = np.ones(len(index), dtype=bool)
    inside[month_ends] = False
    assert not exits[inside].any()


def test_exact_zero_trailing_return_is_cash():
    frame = _flat_month_frame(MONTHLY)
    entries, exits = _signals(frame, frame.index)
    month_ends = _month_end_positions(frame.index)
    assert MONTHLY[15] == MONTHLY[3]          # the hand-built 0% month
    assert not entries[month_ends[15]]
    assert exits[month_ends[15]]


def test_end_to_end_trade_dates_match_next_open_entry_and_month_end_close_exit():
    frame = _flat_month_frame(MONTHLY)
    assert len(frame) >= 500
    index = frame.index
    signal_fn = mod.make_signal_fn("UUP", index, {"UUP": frame})
    trades, in_market = backtest_engine.run_backtest(frame, signal_fn)
    month_ends = _month_end_positions(index)
    assert len(trades) == 2
    first, second = trades.iloc[0], trades.iloc[1]
    # Trade 1: decided LONG at month 12's close, filled at the next open;
    # decided CASH at month 14's close, sold at THAT month-end close.
    assert first["entry_date"] == index[month_ends[12] + 1]
    assert first["exit_date"] == index[month_ends[14]]
    assert first["reason"] == "signal_exit"
    assert first["entry_px"] == 25.5 and first["exit_px"] == 24.0
    # Trade 2: LONG at month 16's close, held to the end of data.
    assert second["entry_date"] == index[month_ends[16] + 1]
    assert second["reason"] == "end_of_data"
    assert in_market.sum() > 0


def test_random_walk_end_to_end_produces_at_least_one_trade():
    frame = _random_walk_frame(600, seed=3)
    signal_fn = mod.make_signal_fn("FXE", frame.index, {"FXE": frame})
    trades, _ = backtest_engine.run_backtest(frame, signal_fn)
    assert len(trades) >= 1
    # Rule properties independent of the path: every entry fills on the bar
    # right after a month-end decision bar; every signal exit IS a month-end.
    month_ends = set(_month_end_positions(frame.index))
    for entry_date in trades["entry_date"]:
        assert (frame.index.get_loc(entry_date) - 1) in month_ends
    for _, row in trades.iterrows():
        if row["reason"] == "signal_exit":
            assert frame.index.get_loc(row["exit_date"]) in month_ends


# ------------------------------------------------------- look-ahead

@pytest.mark.parametrize("bucket", ["full", "holdout_tail"])
def test_look_ahead_probe_perturbing_future_bars_never_changes_past_signals(bucket):
    frame = _random_walk_frame(760)
    if bucket == "full":
        index = frame.index
        k = 500
    else:
        index = frame.index[-130:]
        k = 60
    before_e, before_x = _signals(frame, index)
    cut = index[k]
    perturbed = frame.copy()
    after = perturbed.index > cut
    perturbed.loc[after, "Close"] *= 1.3
    perturbed.loc[after, "Open"] *= 1.3
    perturbed.loc[after, "High"] = perturbed.loc[after, "High"] * 1.3 + 5.0
    perturbed.loc[after, "Low"] = perturbed.loc[after, "Low"] * 1.3 - 5.0
    after_e, after_x = _signals(perturbed, index)
    assert np.array_equal(before_e[:k + 1], after_e[:k + 1])
    assert np.array_equal(before_x[:k + 1], after_x[:k + 1])
    # and the perturbation was real: something after k did change
    assert not (np.array_equal(before_e, after_e) and np.array_equal(before_x, after_x))


def test_holdout_bucket_uses_pre_bucket_history_for_warm_up():
    frame = _random_walk_frame(760)
    holdout = frame.index[-130:]
    entries, exits = _signals(frame, holdout)
    # 130 bars is far less than 12 months, so any signal at all proves the
    # factory warmed up on frame history BEFORE the bucket.
    assert entries.any() or exits.any()
    # A bucket-only frame (no history) can make no decision at all.
    bucket_frame = frame.loc[holdout]
    e2, x2 = _signals(bucket_frame, holdout)
    assert not e2.any() and not x2.any()


# ------------------------------------------------------- contract

def test_signal_fn_is_a_pure_lookup_and_per_ticker():
    frame_a = _random_walk_frame(700, seed=3)
    frame_b = _random_walk_frame(700, seed=4)
    frames = {"UUP": frame_a, "FXY": frame_b}
    fn_a = mod.make_signal_fn("UUP", frame_a.index, frames)
    fn_b = mod.make_signal_fn("FXY", frame_b.index, frames)
    # O(1) per call: the slice argument is never read (no per-bar recompute).
    sig_a = [fn_a(None, i) for i in range(len(frame_a))]
    sig_b = [fn_b(None, i) for i in range(len(frame_b))]
    assert any(s.entry for s in sig_a) or any(s.exit for s in sig_a)
    assert ([s.entry for s in sig_a] != [s.entry for s in sig_b]
            or [s.exit for s in sig_a] != [s.exit for s in sig_b])
    # Direct check against the ticker's own frame.
    ea, xa = _signals(frame_a, frame_a.index, ticker="UUP")
    assert [s.entry for s in sig_a] == list(ea)
    assert [s.exit for s in sig_a] == list(xa)


def test_index_outside_the_frame_fails_closed():
    frame = _random_walk_frame(300)
    bad_index = frame.index.union(pd.DatetimeIndex(["2030-01-02"]))
    with pytest.raises(ValueError):
        mod.make_signal_fn("UUP", bad_index, {"UUP": frame})
    with pytest.raises(KeyError):
        mod.make_signal_fn("UUP", frame.index, {"UDN": frame})


def test_month_end_targets_helper_requires_the_twelfth_prior_month():
    frame = _flat_month_frame(MONTHLY)
    targets = mod.month_end_targets(frame["Close"], 12)
    month_ends = _month_end_positions(frame.index)
    assert targets.index.equals(frame.index)
    assert np.isnan(targets.iloc[month_ends[11]])
    assert targets.iloc[month_ends[12]] == 1.0
    assert targets.iloc[month_ends[14]] == 0.0
    assert targets.iloc[month_ends[15]] == 0.0
    # non-month-end bars carry no decision; neither does the final bar
    assert np.isnan(targets.iloc[month_ends[12] + 1])
    assert np.isnan(targets.iloc[-1])


def test_missing_anchor_month_means_no_decision():
    # Drop every bar of month 2 entirely: month 14 (whose anchor is month 2)
    # must make NO decision rather than interpolate to a neighbour.
    frame = _flat_month_frame(MONTHLY)
    periods = frame.index.to_period("M")
    gone = periods == periods.unique()[2]
    frame = frame.loc[~gone]
    targets = mod.month_end_targets(frame["Close"], 12)
    month_ends = _month_end_positions(frame.index)
    # after the drop, list position 13 is calendar month 14
    assert periods.unique()[14] == frame.index[month_ends[13]].to_period("M")
    assert np.isnan(targets.iloc[month_ends[13]])
    # calendar month 13 (anchor month 1, present) still decides
    assert targets.iloc[month_ends[12]] == 1.0
