"""Tests for src/mining/batch5/us_industry_etf_pooled_tsmom.py.

Synthetic frames only (no network, deterministic).
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
from mining.backtest_engine import SignalResult  # noqa: E402
from mining.batch5 import us_industry_etf_pooled_tsmom as mod  # noqa: E402

FAMILY_ID = "us_industry_etf_pooled_tsmom"


# ------------------------------------------------------------ synthetic data

def _frame_from_closes(closes: pd.Series) -> pd.DataFrame:
    close = closes.to_numpy(dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=closes.index)


def _piecewise_frame() -> pd.DataFrame:
    """Business days 2015-01-02 .. 2018-12-31. Close is a deterministic
    piecewise path: rises 1%/month-equivalent through 2016-06, then falls
    hard through 2017-03, then rises again. Built so month-end 12-month
    returns are unambiguously > 0 or < 0 at the bars the tests inspect."""
    index = pd.bdate_range("2015-01-02", "2018-12-31")
    values = np.empty(len(index))
    level = 100.0
    for k, day in enumerate(index):
        if day < pd.Timestamp("2016-07-01"):
            level *= 1.0006
        elif day < pd.Timestamp("2017-04-01"):
            level *= 0.9970
        else:
            level *= 1.0008
        values[k] = level
    return _frame_from_closes(pd.Series(values, index=index))


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None):
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


def _month_end_positions(index: pd.DatetimeIndex) -> dict:
    """{Period('YYYY-MM'): position of that month's last bar}."""
    periods = index.to_period("M")
    out = {}
    for pos, period in enumerate(periods):
        out[period] = pos
    return out


# ------------------------------------------------------------------ contract

def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    found, problems = driver.discover_modules(driver.DEFAULT_MODULES_DIR)
    own = [p for p in problems if p["module"] == FAMILY_ID]
    assert own == [], own
    assert FAMILY_ID in found
    spec = found[FAMILY_ID]["spec"]
    assert spec["shape"] == "pooled_single_asset"
    assert spec["traded"] == mod.TRADED


# ------------------------------------------------------------------ the rule

def test_rule_fires_at_month_end_and_stays_flat_where_it_should():
    frame = _piecewise_frame()
    signal_fn = mod.make_signal_fn("SOXX", frame.index, {"SOXX": frame})
    sigs = _signals(signal_fn, frame)
    me = _month_end_positions(frame.index)
    closes = frame["Close"]

    # Fewer than 12 prior month-end closes: no decision, flat everywhere.
    for period in pd.period_range("2015-01", "2015-12", freq="M"):
        for pos in range(me[period] - 3, me[period] + 1):
            assert sigs[pos] == SignalResult(entry=False, exit=False)
    # Bars strictly before the first decision bar are all flat.
    first_decision = me[pd.Period("2016-01", freq="M")]
    assert all(s == SignalResult() for s in sigs[:first_decision])

    # 2016-01 month-end: trailing 12-month return is positive (rising leg)
    # -> LONG decision at that bar (entry=True), and the state persists
    # into the following month (entry=True mid-month, exit=False).
    pos = me[pd.Period("2016-01", freq="M")]
    assert closes.iloc[pos] > closes.iloc[me[pd.Period("2015-01", freq="M")]]
    assert sigs[pos] == SignalResult(entry=True, exit=False)
    assert sigs[pos + 1] == SignalResult(entry=True, exit=False)
    assert sigs[pos + 7] == SignalResult(entry=True, exit=False)

    # Find the first month-end whose 12-month return turns negative: the
    # decision is FLAT there (entry=False), the exit fires on the NEXT
    # session (next_session_close), and the bar between them is quiet.
    negative = None
    for period in pd.period_range("2016-07", "2017-06", freq="M"):
        pos = me[period]
        base = closes.iloc[me[period - 12]]
        if closes.iloc[pos] / base - 1.0 <= 0:
            negative = period
            break
    assert negative is not None
    pos = me[negative]
    prev_pos = me[negative - 1]
    assert sigs[prev_pos] == SignalResult(entry=True, exit=False)
    assert sigs[pos] == SignalResult(entry=False, exit=False)
    assert sigs[pos + 1] == SignalResult(entry=False, exit=True)
    assert sigs[pos + 2] == SignalResult(entry=False, exit=False)
    # Mid-month bars in a FLAT month never signal.
    assert sigs[pos + 9] == SignalResult(entry=False, exit=False)

    # A later month-end where the 12-month return is positive again -> LONG.
    pos = me[pd.Period("2018-06", freq="M")]
    assert closes.iloc[pos] > closes.iloc[me[pd.Period("2017-06", freq="M")]]
    assert sigs[pos] == SignalResult(entry=True, exit=False)


def test_zero_twelve_month_return_is_flat():
    index = pd.bdate_range("2015-01-02", "2016-12-30")
    # Constant close: every 12-month return is exactly zero -> FLAT always.
    frame = _frame_from_closes(pd.Series(50.0, index=index))
    signal_fn = mod.make_signal_fn("XBI", frame.index, {"XBI": frame})
    sigs = _signals(signal_fn, frame)
    assert all(not s.entry for s in sigs)


def test_signal_is_a_month_end_decision_not_a_daily_one():
    """A one-day spike mid-month must not flip the state until month-end."""
    frame = _piecewise_frame()
    me = _month_end_positions(frame.index)
    # In the falling leg, 2016-11 is FLAT at its month-end; spike a
    # mid-December bar far above every prior close.
    spike_pos = me[pd.Period("2016-11", freq="M")] + 8
    spiked = frame.copy()
    spiked.iloc[spike_pos, spiked.columns.get_loc("Close")] *= 5.0
    spiked.iloc[spike_pos, spiked.columns.get_loc("High")] *= 5.0
    signal_fn = mod.make_signal_fn("XHB", spiked.index, {"XHB": spiked})
    sigs = _signals(signal_fn, spiked)
    assert sigs[spike_pos] == SignalResult(entry=False, exit=False)
    assert sigs[spike_pos + 1] == SignalResult(entry=False, exit=False)


# --------------------------------------------------------------- look-ahead

def test_lookahead_probe_future_bars_never_change_past_signals():
    frame = _piecewise_frame()
    k = len(frame) // 2  # well inside the data, mid-month
    base_fn = mod.make_signal_fn("KRE", frame.index, {"KRE": frame})
    before = _signals(base_fn, frame, upto=k)

    perturbed = frame.copy()
    tail = perturbed.index > frame.index[k]
    perturbed.loc[tail, "Close"] *= 1.3
    perturbed.loc[tail, "High"] = perturbed.loc[tail, "High"] * 1.5 + 7.0
    perturbed.loc[tail, "Low"] = perturbed.loc[tail, "Low"] * 0.5 - 1.0
    perturbed.loc[tail, "Open"] *= 0.9
    pert_fn = mod.make_signal_fn("KRE", perturbed.index, {"KRE": perturbed})
    after = _signals(pert_fn, perturbed, upto=k)
    assert before == after

    # Also at a month-end bar k (the decision bar itself).
    me = _month_end_positions(frame.index)
    k2 = me[pd.Period("2016-08", freq="M")]
    before2 = _signals(base_fn, frame, upto=k2)
    after2 = _signals(pert_fn, perturbed, upto=k2)
    assert before2 == after2


def test_holdout_bucket_is_blind_to_bars_after_its_last_bar():
    """When `index` is a holdout bucket, signals over the bucket are the
    same whether the frame ends at the bucket's last bar or extends past it
    -- except that the frame's final bar cannot be confirmed a month-end
    without a later bar, so the check runs on a bucket ending at a
    month-end confirmed by calendar (the last weekday of the month)."""
    frame = _piecewise_frame()
    start = pd.Timestamp("2017-07-03")
    end = pd.Timestamp("2017-12-29")  # Friday, last weekday of December
    bucket_index = frame.index[(frame.index >= start) & (frame.index <= end)]
    bucket = frame.loc[bucket_index]

    full_fn = mod.make_signal_fn("XME", bucket_index, {"XME": frame})
    truncated = frame.loc[frame.index <= end]
    trunc_fn = mod.make_signal_fn("XME", bucket_index, {"XME": truncated})
    assert _signals(full_fn, bucket) == _signals(trunc_fn, bucket)

    # Warm-up: the state decided BEFORE the bucket carries in, so the first
    # bucket bar already says entry=True (the rising leg resumed 2017-04
    # but the 12-month return only turns positive later; assert consistency
    # against a full-frame run instead of a hand value).
    full_run = _signals(mod.make_signal_fn("XME", frame.index, {"XME": frame}), frame)
    offset = int(frame.index.get_loc(start))
    assert _signals(full_fn, bucket) == full_run[offset:offset + len(bucket)]


def test_frame_final_bar_is_a_decision_only_when_no_weekday_remains():
    frame = _piecewise_frame()
    # Truncate mid-month: the final bar is NOT a month-end -> no decision.
    mid = frame.loc[frame.index <= pd.Timestamp("2018-03-14")]
    fn_mid = mod.make_signal_fn("XOP", mid.index, {"XOP": mid})
    full_fn = mod.make_signal_fn("XOP", frame.index, {"XOP": frame})
    n = len(mid)
    assert _signals(fn_mid, mid) == _signals(full_fn, frame, upto=n - 1)
    # Truncate at the last weekday of a month: the final bar IS a decision.
    eom = frame.loc[frame.index <= pd.Timestamp("2018-03-30")]  # Friday
    fn_eom = mod.make_signal_fn("XOP", eom.index, {"XOP": eom})
    n = len(eom)
    assert _signals(fn_eom, eom) == _signals(full_fn, frame, upto=n - 1)


# -------------------------------------------------------------- end-to-end

def test_end_to_end_run_backtest_produces_trades():
    index = pd.bdate_range("2016-01-04", periods=600)
    rng = np.random.RandomState(7)
    steps = rng.normal(0.0008, 0.008, len(index))
    frame = _frame_from_closes(pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index))
    signal_fn = mod.make_signal_fn("GDX", frame.index, {"GDX": frame})
    trades, in_market = backtest_engine.run_backtest(frame, signal_fn, cost_bps=5.0)
    assert len(trades) >= 1
    assert in_market.any()
    # Entries execute at an open on the session after a month-end bar.
    me_positions = set(_month_end_positions(frame.index).values())
    for entry_date in trades["entry_date"]:
        pos = int(frame.index.get_loc(entry_date))
        assert pos - 1 in me_positions, entry_date
    # Signal exits execute at the close of the session AFTER the FLAT
    # month-end decision (two sessions after the last held month-end bar).
    for _, row in trades.iterrows():
        if row["reason"] == "signal_exit":
            pos = int(frame.index.get_loc(row["exit_date"]))
            assert pos - 1 in me_positions, row["exit_date"]


def test_factory_rejects_missing_ticker_frame():
    frame = _piecewise_frame()
    with pytest.raises(KeyError):
        mod.make_signal_fn("ITA", frame.index, {"SOXX": frame})
