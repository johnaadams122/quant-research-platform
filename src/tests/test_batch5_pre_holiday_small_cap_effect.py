"""Tests for src/mining/batch5/pre_holiday_small_cap_effect.py.

Synthetic prices only (deterministic RNG); the trading calendar comes from
the locally installed exchange_calendars XNYS calendar (no network). The
real ledger/catalog are never touched.
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
from mining.batch5 import pre_holiday_small_cap_effect as mod  # noqa: E402

# ------------------------------------------------------------------ helpers

def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    return mod.xnys_sessions(pd.Timestamp(start), pd.Timestamp(end))


def _frame(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=pd.DatetimeIndex(index))


def _signals(frame: pd.DataFrame, index=None):
    """Drive the factory the way the engine does: growing slice, position i."""
    index = frame.index if index is None else index
    fn = mod.make_signal_fn("IWM", index, {"IWM": frame})
    out = {}
    for i in range(len(frame)):
        res = fn(frame.iloc[:i + 1], i)
        out[frame.index[i]] = (bool(res.entry), bool(res.exit))
    return out


def _dates(signals, which: int) -> set:
    return {pd.Timestamp(d).strftime("%Y-%m-%d")
            for d, flags in signals.items() if flags[which]}


# ----------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == "pre_holiday_small_cap_effect"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["IWM"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES == 200
    assert callable(mod.make_signal_fn)


def test_module_passes_driver_contract_validation():
    import run_batch5_mining as batch5
    spec = batch5.validate_module_contract(mod)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["IWM"]
    assert spec["snapshots"] == {}


def test_ad_hoc_closures_come_from_parameters_and_are_weekday_non_sessions():
    listed = mod.PARAMETERS["excluded_ad_hoc_closures"]
    assert len(listed) == 10
    assert mod.AD_HOC_CLOSURES == frozenset(pd.Timestamp(d) for d in listed)
    sessions = set(_sessions("2001-01-01", "2025-12-31"))
    for date in mod.AD_HOC_CLOSURES:
        assert date.weekday() < 5
        assert date not in sessions


# --------------------------------------------------------------------- rule

def test_fires_before_scheduled_holidays_and_stays_flat_otherwise():
    # Q4 2016: Thanksgiving Thu 2016-11-24 and Christmas observed Mon
    # 2016-12-26 are the only weekday closures (Columbus/Veterans Day are
    # NYSE sessions). Weekends alone never count.
    frame = _frame(_sessions("2016-10-03", "2016-12-30"))
    signals = _signals(frame)
    # entry fires the session BEFORE the pre-holiday session
    assert _dates(signals, 0) == {"2016-11-22", "2016-12-22"}
    # exit: the literal pre-holiday-session term plus the first evaluable
    # bar after it (the post-holiday session's close)
    assert _dates(signals, 1) == {"2016-11-23", "2016-11-25",
                                  "2016-12-23", "2016-12-27"}
    # a Friday before an ordinary weekend is not a pre-holiday session
    assert signals[pd.Timestamp("2016-11-17")] == (False, False)
    assert signals[pd.Timestamp("2016-11-18")] == (False, False)


def test_ad_hoc_closure_is_excluded_by_rule():
    # 2018-12-05 (national day of mourning) is a weekday closure NOT in the
    # scheduled set -> 2018-12-04 is not a pre-holiday session.
    frame = _frame(_sessions("2018-11-01", "2019-01-31"))
    signals = _signals(frame)
    assert _dates(signals, 0) == {"2018-11-20",   # Thanksgiving 11-22
                                  "2018-12-21",   # Christmas Tue 12-25
                                  "2018-12-28",   # New Year Tue 01-01
                                  "2019-01-17"}   # MLK Mon 01-21
    assert signals[pd.Timestamp("2018-12-03")] == (False, False)
    assert signals[pd.Timestamp("2018-12-06")] == (False, False)


def test_closure_mixing_scheduled_and_ad_hoc_days_is_excluded():
    # 2007-01-01 (New Year) + 2007-01-02 (ad hoc) form ONE closure between
    # 2006-12-29 and 2007-01-03; any ad hoc day inside it excludes the trade.
    frame = _frame(_sessions("2006-12-01", "2007-01-31"))
    signals = _signals(frame)
    assert _dates(signals, 0) == {"2006-12-21",   # Christmas Mon 12-25
                                  "2007-01-11"}   # MLK Mon 01-15
    assert signals[pd.Timestamp("2006-12-28")] == (False, False)
    assert signals[pd.Timestamp("2006-12-29")] == (False, False)


def test_first_and_last_bars_have_no_signal_from_beyond_the_frame():
    # Frame ends ON a pre-holiday session (Wed 2016-11-23): the calendar is
    # XNYS intersected with the frame's bars, so no next session exists ->
    # no pre-holiday status, nothing to enter or exit.
    frame = _frame(_sessions("2016-11-01", "2016-11-23"))
    signals = _signals(frame)
    assert signals[pd.Timestamp("2016-11-22")] == (False, False)
    assert signals[pd.Timestamp("2016-11-23")] == (False, False)


def test_non_session_bars_are_dropped_from_the_calendar():
    base = _frame(_sessions("2016-10-03", "2016-12-30"))
    saturday = pd.Timestamp("2016-11-19")
    extra = base.copy()
    extra.loc[saturday] = base.iloc[0].to_numpy()
    extra = extra.sort_index()
    signals_base = _signals(base)
    signals_extra = _signals(extra)
    assert signals_extra[saturday] == (False, False)
    for date, flags in signals_base.items():
        assert signals_extra[date] == flags


# ------------------------------------------------------------- causality

def test_look_ahead_probe_future_prices_never_change_past_signals():
    frame = _frame(_sessions("2015-01-02", "2017-05-31"))
    assert len(frame) >= 600
    k = 300
    before = _signals(frame)
    perturbed = frame.copy()
    tail = perturbed.index[k + 1:]
    perturbed.loc[tail, "Close"] *= 1.3
    perturbed.loc[tail, "High"] = perturbed.loc[tail, "High"] * 1.5 + 7.0
    perturbed.loc[tail, "Low"] = perturbed.loc[tail, "Low"] * 0.5
    perturbed.loc[tail, "Open"] *= 0.9
    after = _signals(perturbed)
    for date in frame.index[:k + 1]:
        assert after[date] == before[date]


def test_truncating_the_future_never_changes_past_signals():
    # Stricter than the price probe: remove the future bars entirely. The
    # entry at bar k needs the calendar two sessions ahead, so compare
    # through k - 2 (calendar knowledge, never price).
    frame = _frame(_sessions("2015-01-02", "2017-05-31"))
    k = 300
    full = _signals(frame)
    truncated = _signals(frame.iloc[:k + 1])
    for date in frame.index[:k - 1]:
        assert truncated[date] == full[date]


def test_signal_ignores_the_slice_contents_after_precomputation():
    frame = _frame(_sessions("2016-10-03", "2016-12-30"))
    fn = mod.make_signal_fn("IWM", frame.index, {"IWM": frame})
    poisoned = frame.copy()
    poisoned.loc[:, ["Open", "High", "Low", "Close"]] = np.nan
    for i in range(len(frame)):
        clean = fn(frame.iloc[:i + 1], i)
        dirty = fn(poisoned.iloc[:i + 1], i)
        assert (clean.entry, clean.exit) == (dirty.entry, dirty.exit)


def test_bucket_index_does_not_change_signals_inside_the_bucket():
    frame = _frame(_sessions("2015-01-02", "2017-05-31"))
    full = _signals(frame)
    bucket = frame.iloc[-120:]
    fn = mod.make_signal_fn("IWM", bucket.index, {"IWM": frame})
    for i in range(len(bucket)):
        res = fn(bucket.iloc[:i + 1], i)
        assert (res.entry, res.exit) == full[bucket.index[i]]


def test_position_mismatch_falls_back_to_date_lookup():
    frame = _frame(_sessions("2016-10-03", "2016-12-30"))
    fn = mod.make_signal_fn("IWM", frame.index, {"IWM": frame})
    i = frame.index.get_loc(pd.Timestamp("2016-11-22"))
    res = fn(frame.iloc[:i + 1], i + 5)          # wrong position, right date
    assert (res.entry, res.exit) == (True, False)
    res = fn(frame.iloc[:i + 1], 10_000)         # out of range position
    assert (res.entry, res.exit) == (True, False)


# ------------------------------------------------------------- end to end

def test_end_to_end_run_backtest_one_trade_per_scheduled_holiday():
    index = _sessions("2015-01-02", "2017-05-31")
    frame = _frame(index, seed=3)
    assert len(frame) >= 600
    fn = mod.make_signal_fn("IWM", frame.index, {"IWM": frame})
    trades, in_market = backtest_engine.run_backtest(frame, fn, cost_bps=1.0)
    assert len(trades) >= 1
    cal = mod.session_calendar(frame.index)
    pre = mod.pre_holiday_mask(cal)
    # one trade per pre-holiday session whose entry-signal bar is in the frame
    assert len(trades) == int(pre[1:].sum())
    assert len(trades) >= 20
    positions = {ts: j for j, ts in enumerate(cal)}
    for row in trades.itertuples(index=False):
        j = positions[row.entry_date]
        assert pre[j]                              # entered AT the pre-holiday open
        assert row.exit_date == cal[j + 1]         # out at the next session's close
        assert row.held == 1
        assert row.reason == "signal_exit"
        assert row.entry_px == pytest.approx(float(frame.loc[row.entry_date, "Open"]))
        assert row.exit_px == pytest.approx(float(frame.loc[row.exit_date, "Close"]))
    assert int(in_market.sum()) == 2 * len(trades)


def test_end_to_end_driver_shaped_windows():
    import run_batch3_mining as batch3
    frame = _frame(_sessions("2015-01-02", "2017-12-29"), seed=5)

    def factory(index):
        return mod.make_signal_fn("IWM", index, {"IWM": frame})

    out = batch3.run_single_asset_windows(
        frame, factory, holdout_start="2017-07-01", holdout_end="2017-12-29",
        cost_bps=1.0, tax_haircut_rate=0.0)
    assert out["stats_in_sample"]["trade_count"] >= 20
    assert out["stats_holdout"]["trade_count"] >= 3   # Labor Day, Thanksgiving, Christmas
    assert (out["in_sample_trades"]["held"] == 1).all()
    assert (out["holdout_trades"]["held"] == 1).all()
