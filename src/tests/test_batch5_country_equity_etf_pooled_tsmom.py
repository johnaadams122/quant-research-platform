"""Tests for src/mining/batch5/country_equity_etf_pooled_tsmom.py.

Synthetic frames only (no network, deterministic); the ledger and catalog
are never touched.
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
from mining.batch5 import country_equity_etf_pooled_tsmom as mod  # noqa: E402

# ------------------------------------------------------------ fixtures

def _flat_month_frame(monthly_closes, start="2020-01-01") -> pd.DataFrame:
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
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_bars)))
    open_ = close * (1 + rng.normal(0, 0.002, n_bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n_bars))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=index)


def _month_end_positions(index: pd.DatetimeIndex) -> list:
    periods = index.to_period("M")
    return [i for i in range(len(index) - 1) if periods[i] != periods[i + 1]] + [len(index) - 1]


def _signals(frame: pd.DataFrame, index: pd.DatetimeIndex, ticker="EWJ"):
    signal_fn = mod.make_signal_fn(ticker, index, {ticker: frame})
    entries = np.array([signal_fn(frame.loc[:index[i]], i).entry for i in range(len(index))])
    exits = np.array([signal_fn(frame.loc[:index[i]], i).exit for i in range(len(index))])
    return entries, exits


# Hand-built monthly path. Month k's trailing-12 comparison is against month
# k-12; the first decision is possible at month 12 (needs month 0).
#   month 12: 110 vs 100 -> +10%  LONG
#   month 13: 105 vs 100 -> +5%   LONG (hold)
#   month 14:  90 vs 100 -> -10%  FLAT -> sell at the NEXT session's close
#   month 15: 100 vs 100 ->  0%   FLAT (ret12 == 0 is FLAT, strict > 0)
#   month 16: 120 vs 100 -> +20%  LONG -> buy at the NEXT session's open
#   months 17..27: 120 vs 100/110/105/90/100/120 -> all > 0, stay LONG
MONTHLY = [100.0] * 12 + [110.0, 105.0, 90.0, 100.0, 120.0] + [120.0] * 11


# --------------------------------------------------------- rule fires

def test_rule_fires_at_month_ends_where_it_should_and_stays_flat_elsewhere():
    frame = _flat_month_frame(MONTHLY)
    index = frame.index
    entries, exits = _signals(frame, index)
    periods = index.to_period("M")
    month_ends = _month_end_positions(index)
    assert len(month_ends) == len(MONTHLY)

    # No decision before month 12's month-end: everything flat.
    first_decision = month_ends[12]
    assert not entries[:first_decision].any()
    assert not exits[:first_decision].any()

    # Month-end targets: LONG at 12, 13, 16+; FLAT at 14, 15.
    expected_long = {12: True, 13: True, 14: False, 15: False, 16: True}
    for month, is_long in expected_long.items():
        assert bool(entries[month_ends[month]]) is is_long, month
    for month in range(17, len(MONTHLY)):
        assert entries[month_ends[month]]

    # Entry persists through the following month while the target is LONG;
    # month 15 (a FLAT month) has no entry flag anywhere.
    month15 = np.flatnonzero(periods == periods[month_ends[15]])
    assert not entries[month15].any()
    month13 = np.flatnonzero(periods == periods[month_ends[13]])
    assert entries[month13].all()

    # Exit fires on the bar AFTER a FLAT month-end decision, never on the
    # month-end bar itself (sell at the next session's close).
    flat_decision = month_ends[14]
    assert not exits[flat_decision]
    assert exits[flat_decision + 1]
    # The month-end decision that flips LONG (month 16) is not an exit bar
    # for the following session.
    assert not exits[month_ends[16] + 1]
    # Exits never fire while the target is LONG and was LONG the bar before.
    month13_after_first = month13[1:]
    assert not exits[month13_after_first].any()


def test_exact_zero_trailing_return_is_flat():
    frame = _flat_month_frame(MONTHLY)
    entries, _ = _signals(frame, frame.index)
    month_ends = _month_end_positions(frame.index)
    assert MONTHLY[15] == MONTHLY[3]          # the hand-built 0% month
    assert not entries[month_ends[15]]


def test_end_to_end_trade_dates_match_next_open_entry_and_next_close_exit():
    frame = _flat_month_frame(MONTHLY)
    assert len(frame) >= 550
    index = frame.index
    signal_fn = mod.make_signal_fn("EWJ", index, {"EWJ": frame})
    trades, in_market = backtest_engine.run_backtest(frame, signal_fn)
    month_ends = _month_end_positions(index)
    assert len(trades) == 2
    first, second = trades.iloc[0], trades.iloc[1]
    # Trade 1: decided LONG at month 12's close, filled at the next open;
    # decided FLAT at month 14's close, sold at the NEXT session's close.
    assert first["entry_date"] == index[month_ends[12] + 1]
    assert first["exit_date"] == index[month_ends[14] + 1]
    assert first["reason"] == "signal_exit"
    assert first["entry_px"] == 105.0 and first["exit_px"] == 100.0
    # Trade 2: LONG at month 16's close, held to the end of data.
    assert second["entry_date"] == index[month_ends[16] + 1]
    assert second["reason"] == "end_of_data"
    assert in_market.sum() > 0


def test_random_walk_end_to_end_produces_at_least_one_trade():
    # seed 1: 16 decidable month-ends, decisions FFFFFFFFFLLLLLLL (seed 7's
    # path is negative-trailing throughout and can never enter -- a fixture
    # fact, checked once, not a rule property)
    frame = _random_walk_frame(600, seed=1)
    signal_fn = mod.make_signal_fn("EWG", frame.index, {"EWG": frame})
    trades, _ = backtest_engine.run_backtest(frame, signal_fn)
    assert len(trades) >= 1
    # Rule property independent of the path: every entry fills on the bar
    # right after a month-end decision bar.
    month_ends = set(_month_end_positions(frame.index))
    for entry_date in trades["entry_date"]:
        assert (frame.index.get_loc(entry_date) - 1) in month_ends


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
    frame_a = _random_walk_frame(700, seed=1)
    frame_b = _random_walk_frame(700, seed=2)
    frames = {"EWJ": frame_a, "EWZ": frame_b}
    fn_a = mod.make_signal_fn("EWJ", frame_a.index, frames)
    fn_b = mod.make_signal_fn("EWZ", frame_b.index, frames)
    # O(1) per call: the slice argument is never read (no per-bar recompute).
    sig_a = [fn_a(None, i) for i in range(len(frame_a))]
    sig_b = [fn_b(None, i) for i in range(len(frame_b))]
    assert any(s.entry for s in sig_a)
    assert [s.entry for s in sig_a] != [s.entry for s in sig_b]
    # Direct check against the ticker's own frame.
    ea, _ = _signals(frame_a, frame_a.index, ticker="EWJ")
    assert [s.entry for s in sig_a] == list(ea)


def test_index_outside_the_frame_fails_closed():
    frame = _random_walk_frame(300)
    bad_index = frame.index.union(pd.DatetimeIndex(["2030-01-02"]))
    with pytest.raises(ValueError):
        mod.make_signal_fn("EWJ", bad_index, {"EWJ": frame})
    with pytest.raises(KeyError):
        mod.make_signal_fn("EWJ", frame.index, {"EWG": frame})


def test_month_end_targets_helper_requires_the_twelfth_prior_month():
    frame = _flat_month_frame(MONTHLY)
    targets = mod.month_end_targets(frame["Close"], 12)
    month_ends = _month_end_positions(frame.index)
    assert targets.index.equals(frame.index)
    assert targets.iloc[month_ends[11]] is np.nan or np.isnan(targets.iloc[month_ends[11]])
    assert targets.iloc[month_ends[12]] == 1.0
    assert targets.iloc[month_ends[14]] == 0.0
    # non-month-end bars carry no decision
    assert np.isnan(targets.iloc[month_ends[12] + 1])
