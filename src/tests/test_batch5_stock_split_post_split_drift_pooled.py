"""Tests for src/mining/batch5/stock_split_post_split_drift_pooled.py.

Synthetic frames and injected event calendars only (no network,
deterministic). The split-calendar fetch test uses an injected fetcher and
writes only under tmp_path; nothing under data/mining is read or modified.
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

FAMILY_ID = "stock_split_post_split_drift_pooled"

mod = importlib.import_module(f"mining.batch5.{FAMILY_ID}")

TS = mod.TIME_STOP_BARS  # 252


# ------------------------------------------------------------ synthetic data

def _frame_from_close(close, index):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _frame(n_days: int = 600, start: str = "2018-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, n_days))
    return _frame_from_close(close, index)


def _signals(frame, events, index=None, ticker="AAA", upto=None):
    index = frame.index if index is None else index
    fn = mod.make_signal_fn(ticker, index, {ticker: frame}, events=events)
    bucket = frame.loc[index]
    n = len(bucket) if upto is None else upto
    return fn, [fn(bucket.iloc[:i + 1], i) for i in range(n)]


def _flags(results):
    return ([i for i, r in enumerate(results) if r.entry],
            [i for i, r in enumerate(results) if r.exit])


# --------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "event_pooled"
    assert mod.SIGNAL == [] and mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)
    assert len(mod.TRADED) == len(set(mod.TRADED))
    assert set(mod.TRADED) <= set(mod.RESOLVED_POOL)


# ------------------------------------------------------------- ratio rule

@pytest.mark.parametrize("ratio", [1.25, 4.0 / 3.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                                   7.0, 10.0, 20.0, 1.5 + 0.0004],
                         ids=["five_for_four", "four_for_three", "three_for_two", "two_for_one", "three_for_one",
                              "four_for_one", "five_for_one", "seven_for_one", "ten_for_one", "twenty_for_one",
                              "three_for_two_rounded"])
def test_ratio_rule_accepts_true_forward_splits(ratio):
    assert mod.passes_ratio_rule(ratio)


@pytest.mark.parametrize("ratio", [1.281, 1.324, 1.589, 1.0, 1.2, 0.5, 0.1,
                                   1.5 + 0.0006, float("nan"), float("inf"),
                                   None, "x"])
def test_ratio_rule_rejects_reverse_splits_and_adjustment_factors(ratio):
    assert not mod.passes_ratio_rule(ratio)


def test_events_from_raw_applies_the_rule_and_keeps_every_ticker():
    raw = {"GE": {"2023-01-04": 1.281, "2000-05-08": 3.0},
           "RTX": {"2020-04-03": 1.589},
           "AAPL": {"2020-08-31": 4.0, "2014-06-09": 7.0, "1987-06-16": 2.0}}
    events = mod.events_from_raw(raw)
    assert events == {"GE": ["2000-05-08"], "RTX": [],
                      "AAPL": ["1987-06-16", "2014-06-09", "2020-08-31"]}


def test_normalize_split_series_handles_tz_aware_index():
    idx = pd.DatetimeIndex(["2020-08-31", "2014-06-09"], tz="America/New_York")
    series = pd.Series([4.0, 7.0], index=idx)
    assert mod.normalize_split_series(series) == {"2014-06-09": 7.0, "2020-08-31": 4.0}
    assert mod.normalize_split_series(None) == {}
    assert mod.normalize_split_series(pd.Series([], dtype=float)) == {}


# ------------------------------------------------------------- rule fires

def test_fires_on_split_date_and_exits_252_bars_after_fill():
    frame = _frame(600)
    k = 100
    events = {"AAA": [frame.index[k].strftime("%Y-%m-%d")]}
    _, results = _signals(frame, events)
    entries, exits = _flags(results)
    assert entries == [k]
    # fill at k+1 (next open); exit at the close of the 252nd bar after it
    assert exits == [k + 1 + TS]
    assert results[k] == SignalResult(entry=True, exit=False)
    assert results[k + 1 + TS] == SignalResult(entry=False, exit=True)
    assert all(r == SignalResult() for i, r in enumerate(results)
               if i not in (k, k + 1 + TS))


def test_flat_when_no_events_for_ticker():
    frame = _frame(300)
    _, results = _signals(frame, {"OTHER": [frame.index[10].strftime("%Y-%m-%d")]})
    assert _flags(results) == ([], [])
    _, results = _signals(frame, {})
    assert _flags(results) == ([], [])


def test_split_during_open_hold_is_skipped_including_on_exit_bar():
    frame = _frame(900)
    k = 50
    exit_bar = k + 1 + TS
    dates = [frame.index[p].strftime("%Y-%m-%d")
             for p in (k, k + 1, k + 120, exit_bar)]
    _, results = _signals(frame, {"AAA": dates})
    entries, exits = _flags(results)
    assert entries == [k] and exits == [exit_bar]
    # the first split dated AFTER the exit bar is eligible again
    dates.append(frame.index[exit_bar + 1].strftime("%Y-%m-%d"))
    _, results = _signals(frame, {"AAA": dates})
    entries, exits = _flags(results)
    assert entries == [k, exit_bar + 1]
    assert exits == [exit_bar, exit_bar + 2 + TS]


def test_hold_open_at_bucket_end_has_no_exit_and_blocks_later_splits():
    frame = _frame(400)
    k = 300
    dates = [frame.index[p].strftime("%Y-%m-%d") for p in (k, 350, 399)]
    _, results = _signals(frame, {"AAA": dates})
    assert _flags(results) == ([k], [])


@pytest.mark.parametrize("n_days,offset", [(400, 2), (128, 2), (3, 2), (2, 2),
                                            (400, 1), (128, 1), (2, 1)])
def test_zero_hold_entry_at_bucket_tail_reports_no_entry(n_days, offset):
    """Data hygiene item 1: a split whose FILL bar would be the bucket's last
    bar (position n-2) -- or that has no fill bar at all (n-1) -- reports no
    entry, because the engine would open and force-close it on the same bar
    (held=0, end_of_data), which measures one intraday session rather than
    the 252-bar post-split drift the rule specifies.

    Guards against an implementation that keeps the offset=2 entry:
    run_backtest would then record a held=0 trade.
    """
    frame = _frame(n_days)
    k = n_days - offset
    events = {"AAA": [frame.index[k].strftime("%Y-%m-%d")]}
    entries, exits = mod.entry_exit_positions(
        [frame.index[k]], frame.index)
    assert entries.tolist() == [] and exits.tolist() == []
    _, results = _signals(frame, events)
    assert _flags(results) == ([], [])
    fn = mod.make_signal_fn("AAA", frame.index, {"AAA": frame}, events=events)
    trades, in_market = backtest_engine.run_backtest(frame, fn)
    assert len(trades) == 0
    assert not in_market.any()


def test_zero_hold_exclusion_does_not_touch_a_one_bar_hold():
    """The exclusion is exactly "no hold of even one bar is possible": a
    split at position n-3 still fills at n-2 and is closed by end_of_data on
    the last bar with held=1, and is kept."""
    frame = _frame(200)
    k = len(frame) - 3
    events = {"AAA": [frame.index[k].strftime("%Y-%m-%d")]}
    entries, exits = mod.entry_exit_positions([frame.index[k]], frame.index)
    assert entries.tolist() == [k] and exits.tolist() == []
    fn = mod.make_signal_fn("AAA", frame.index, {"AAA": frame}, events=events)
    trades, _ = backtest_engine.run_backtest(frame, fn)
    assert len(trades) == 1
    assert trades.iloc[0]["held"] == 1
    assert trades.iloc[0]["reason"] == "end_of_data"


def test_zero_hold_tail_split_does_not_block_an_earlier_eligible_split():
    """The tail exclusion closes out only the calendar AFTER it: an earlier
    split still enters and its own hold is unaffected."""
    frame = _frame(400)
    k, tail = 100, len(frame) - 2
    dates = [frame.index[p].strftime("%Y-%m-%d") for p in (k, tail)]
    _, results = _signals(frame, {"AAA": dates})
    assert _flags(results) == ([k], [k + 1 + TS])
    fn = mod.make_signal_fn("AAA", frame.index, {"AAA": frame},
                            events={"AAA": dates})
    trades, _ = backtest_engine.run_backtest(frame, fn)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == frame.index[k + 1]
    assert trades.iloc[0]["exit_date"] == frame.index[k + 1 + TS]
    assert trades.iloc[0]["reason"] == "signal_exit"


def test_holdout_length_bucket_cannot_reach_the_pre_registered_252_bar_exit():
    """Data hygiene item 2: the derived 6-month holdout bucket is about half
    the 252 bars the time stop needs, so NO holdout trade exercises the
    rule's own exit -- every one is an end_of_data truncation. Pinned here
    (and in the docstring) so the disclosure cannot silently go stale.

The sizes exercised here, including lengths typical of a 6-month bucket,
    show the property is not knife-edged at any one length -- it holds for
    any bucket up to and including 252 bars.
    """
    for bucket_bars in (60, 123, 128, TS):
        frame = _frame(bucket_bars)
        k = 5
        events = {"AAA": [frame.index[k].strftime("%Y-%m-%d")]}
        entries, exits = mod.entry_exit_positions([frame.index[k]], frame.index)
        assert entries.tolist() == [k]
        # the rule's exit bar (k + 1 + 252) is past the bucket -> unreachable
        assert exits.tolist() == []
        assert k + 1 + TS >= bucket_bars
        fn = mod.make_signal_fn("AAA", frame.index, {"AAA": frame}, events=events)
        trades, _ = backtest_engine.run_backtest(frame, fn)
        assert len(trades) == 1
        assert trades.iloc[0]["reason"] == "end_of_data"
        assert trades.iloc[0]["held"] < TS
    # and one bar more than the bucket the exit needs DOES reach it
    frame = _frame(k + 2 + TS)
    events = {"AAA": [frame.index[k].strftime("%Y-%m-%d")]}
    _, exits = mod.entry_exit_positions([frame.index[k]], frame.index)
    assert exits.tolist() == [k + 1 + TS]


def test_docstring_discloses_both_data_hygiene_items():
    """The two disclosures are load-bearing for how a reader interprets this
    family's holdout verdict; assert they are actually present."""
    doc = mod.__doc__
    assert "Data hygiene" in doc
    for phrase in ("ZERO-HOLD ENTRIES ARE EXCLUDED",
                   "THE 252-BAR EXIT IS UNREACHABLE IN THE HOLDOUT BUCKET",
                   "held = 0", "end_of_data"):
        assert phrase in doc, phrase


def test_docstring_states_the_holdout_length_as_derived():
    """The holdout-bucket claim must describe the pool that actually
    trades (members with a full history all share one holdout length) and
    must not pin a run-date-dependent session count: a hand-measured literal
    rots the moment the run date moves.
    """
    doc = mod.__doc__
    assert "derived, never" in doc, "the docstring must not pin a run-date-dependent count"
    assert "ZERO in-sample bars" in doc


def test_docstring_carries_no_measured_backtest_figures():
    """The data-hygiene disclosures are mechanical; they must not quote
    measured edge figures (basis-point returns, pooled trade counts) from
    any backtest run."""
    doc = mod.__doc__
    assert "bps" not in doc
    assert "trades /" not in doc


def test_split_date_outside_bucket_is_ignored_and_offbar_maps_forward():
    frame = _frame(600)
    # a weekend date inside the bucket -> first following session
    saturday = frame.index[200] + pd.Timedelta(days=(5 - frame.index[200].weekday()))
    assert saturday.weekday() == 5
    nxt = int(frame.index.searchsorted(saturday))
    before = (frame.index[0] - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    after = (frame.index[-1] + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    _, results = _signals(frame, {"AAA": [before, saturday.strftime("%Y-%m-%d"), after]})
    assert _flags(results) == ([nxt], [nxt + 1 + TS])


def test_holdout_bucket_is_independent_of_pre_bucket_splits():
    frame = _frame(900)
    holdout = frame.index[600:]
    # split 100 bars before the bucket: its hold would still be open, but the
    # bucket runs independently -> no position; a split inside fires normally
    dates = [frame.index[500].strftime("%Y-%m-%d"),
             holdout[20].strftime("%Y-%m-%d")]
    fn = mod.make_signal_fn("AAA", holdout, {"AAA": frame}, events={"AAA": dates})
    bucket = frame.loc[holdout]
    results = [fn(bucket.iloc[:i + 1], i) for i in range(len(bucket))]
    assert _flags(results) == ([20], [20 + 1 + TS])


def test_entry_exit_positions_direct():
    index = pd.bdate_range("2020-01-01", periods=600)
    entries, exits = mod.entry_exit_positions(
        [index[10], index[15], index[300]], index, time_stop_bars=100)
    assert entries.tolist() == [10, 300] and exits.tolist() == [111, 401]
    entries, exits = mod.entry_exit_positions([], index)
    assert entries.size == 0 and exits.size == 0
    with pytest.raises(ValueError):
        mod.entry_exit_positions([index[1]], index, time_stop_bars=0)


def test_factory_contract_errors():
    frame = _frame(50)
    with pytest.raises(KeyError):
        mod.make_signal_fn("ZZZ", frame.index, {"AAA": frame}, events={})
    with pytest.raises(ValueError):
        mod.make_signal_fn("AAA", frame.index, {"AAA": frame.drop(columns="Close")},
                           events={})
    fn = mod.make_signal_fn("AAA", frame.index[:0], {"AAA": frame}, events={})
    assert fn(frame.iloc[:1], 0) == SignalResult()


# ------------------------------------------------------------ look-ahead

def test_lookahead_probe_future_bars_do_not_change_past_signals():
    frame = _frame(700)
    k = 320
    events = {"AAA": [frame.index[p].strftime("%Y-%m-%d") for p in (30, 60, 400)]}
    _, base = _signals(frame, events, upto=k + 1)
    perturbed = frame.copy()
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] *= 1.3
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("High")] *= 1.5
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Low")] *= 0.7
    _, again = _signals(perturbed, events, upto=k + 1)
    assert base == again
    # and the signal is a pure function of the calendar: perturbing EVERY
    # bar changes nothing either
    _, all_perturbed = _signals(frame * 3.0, events, upto=k + 1)
    assert base == all_perturbed


def test_per_bar_call_is_position_lookup_not_recompute():
    frame = _frame(600)
    fn, _ = _signals(frame, {"AAA": [frame.index[5].strftime("%Y-%m-%d")]}, upto=0)
    assert hasattr(fn, "entry_positions") and fn.entry_positions.tolist() == [5]
    # calling with a one-row slice (i mismatch) falls back to date lookup
    assert fn(frame.iloc[5:6], 0) == SignalResult(entry=True, exit=False)
    assert fn(frame.iloc[[5 + 1 + TS]], 0) == SignalResult(entry=False, exit=True)
    # a date not in the bucket is a no-op
    other = frame.copy()
    other.index = other.index + pd.Timedelta(days=5000)
    assert fn(other.iloc[:1], 0) == SignalResult()


# ------------------------------------------------------------ end to end

def test_end_to_end_run_backtest_one_trade_with_252_bar_hold():
    frame = _frame(600)
    k = 100
    dates = [frame.index[p].strftime("%Y-%m-%d") for p in (k, k + 40)]
    fn = mod.make_signal_fn("AAA", frame.index, {"AAA": frame}, events={"AAA": dates})
    trades, in_market = backtest_engine.run_backtest(frame, fn, cost_bps=5.0)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_date"] == frame.index[k + 1]
    assert trade["exit_date"] == frame.index[k + 1 + TS]
    assert trade["held"] == TS
    assert trade["entry_px"] == pytest.approx(float(frame["Open"].iloc[k + 1]))
    assert trade["exit_px"] == pytest.approx(float(frame["Close"].iloc[k + 1 + TS]))
    assert trade["gross_ret"] == pytest.approx(
        float(frame["Close"].iloc[k + 1 + TS]) / float(frame["Open"].iloc[k + 1]) - 1.0)
    assert int(in_market.sum()) == TS + 1


def test_end_to_end_matches_engine_time_stop_semantics():
    """The module's exit bar is the bar the engine's own time_stop=252 would
    close on (only the reason label differs)."""
    frame = _frame(700)
    k = 20
    events = {"AAA": [frame.index[k].strftime("%Y-%m-%d"),
                      frame.index[k + 1 + TS + 3].strftime("%Y-%m-%d")]}
    fn = mod.make_signal_fn("AAA", frame.index, {"AAA": frame}, events=events)
    ours, _ = backtest_engine.run_backtest(frame, fn)

    def entry_only(frame_slice, i):
        return SignalResult(entry=fn(frame_slice, i).entry, exit=False)
    engine_ts, _ = backtest_engine.run_backtest(frame, entry_only, time_stop=TS)
    assert len(ours) == 2 and len(engine_ts) == 2
    for column in ("entry_date", "exit_date", "held", "gross_ret"):
        assert ours[column].tolist() == engine_ts[column].tolist()
    assert set(ours["reason"]) == {"signal_exit"}
    assert set(engine_ts["reason"]) == {"time_stop"}


def test_end_to_end_pooled_two_tickers_independent():
    frame_a, frame_b = _frame(600), _frame(600, start="2018-01-03")
    events = {"AAA": [frame_a.index[10].strftime("%Y-%m-%d")],
              "BBB": [frame_b.index[300].strftime("%Y-%m-%d")]}
    frames = {"AAA": frame_a, "BBB": frame_b}
    counts = {}
    for ticker, frame in frames.items():
        fn = mod.make_signal_fn(ticker, frame.index, frames, events=events)
        trades, _ = backtest_engine.run_backtest(frame, fn)
        counts[ticker] = (len(trades), trades["entry_date"].iloc[0])
    assert counts["AAA"] == (1, frame_a.index[11])
    assert counts["BBB"] == (1, frame_b.index[301])


# ------------------------------------------------------------- data files

def test_load_events_missing_file_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.load_events(str(tmp_path / "nope.json"))


def test_fetch_split_events_with_injected_fetcher_writes_files(tmp_path):
    def fetcher(ticker):
        if ticker == "BAD":
            raise RuntimeError("boom")
        idx = pd.DatetimeIndex(["2020-08-31", "2023-01-04"], tz="America/New_York")
        return pd.Series([4.0, 1.281], index=idx)
    prov = mod.fetch_split_events(["AAA", "BAD"], out_dir=str(tmp_path), retries=1,
                                  pause_seconds=0.0, fetcher=fetcher, log=None)
    assert prov["tickers_fetched"] == 1 and list(prov["tickers_failed"]) == ["BAD"]
    assert prov["raw_split_count"] == 2 and prov["event_count"] == 1
    with open(tmp_path / "events.json", encoding="utf-8") as handle:
        assert json.load(handle) == {"AAA": ["2020-08-31"]}
    loaded = mod.load_events(str(tmp_path / "events.json"))
    assert loaded["AAA"].tolist() == [pd.Timestamp("2020-08-31")]
