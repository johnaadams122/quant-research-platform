"""Tests for the batch5 candidate ``customer_supplier_leadlag_semiconductors``.

Synthetic frames only -- no network, no real OHLC.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(SRC, ".."))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mining import backtest_engine  # noqa: E402
from mining.batch5 import customer_supplier_leadlag_semiconductors as mod  # noqa: E402



# ------------------------------------------------------------ fixtures

def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _frame(index: pd.DatetimeIndex, closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Open": closes * 0.999,
        "High": closes * 1.01,
        "Low": closes * 0.99,
        "Close": closes,
    }, index=index)


def _flat_frames(index: pd.DatetimeIndex) -> dict:
    """Both series flat -- the spread is never > 0."""
    return {"SMH": _frame(index, np.full(len(index), 100.0)),
            "QQQ": _frame(index, np.full(len(index), 200.0))}


def _month_ends(index: pd.DatetimeIndex) -> list:
    periods = index.to_period("M")
    return [index[i] for i in range(len(index) - 1) if periods[i] != periods[i + 1]]


def _decisions(signal_fn, frame: pd.DataFrame) -> list:
    out = []
    for i in range(len(frame)):
        result = signal_fn(frame.iloc[:i + 1], i)
        out.append((bool(result.entry), bool(result.exit)))
    return out


# ------------------------------------------------------------ contract

def test_contract_constants():
    assert mod.FAMILY_ID == "customer_supplier_leadlag_semiconductors"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SMH"]
    assert mod.SIGNAL == ["QQQ"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SMH"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)
    assert not hasattr(mod, "make_target_fn")


# ------------------------------------------------------------ the rule

def test_fires_on_month_end_when_customer_outperformed_and_exits_when_not():
    index = _bdays("2020-01-01", 130)  # ~6 months
    month_ends = _month_ends(index)
    assert len(month_ends) >= 5
    # Month by month: QQQ gains 5% in month 1 (SMH flat) -> True at m1 end.
    # Month 2: SMH +5%, QQQ flat -> False at m2 end.  Month 3: both +2% -> False
    # (spread exactly 0 is NOT > 0).  Month 4: QQQ +1%, SMH -1% -> True.
    periods = index.to_period("M")
    uniq = list(periods.unique())
    qqq_month_ret = {0: 0.0, 1: 0.05, 2: 0.0, 3: 0.02, 4: 0.01, 5: 0.0, 6: 0.0}
    smh_month_ret = {0: 0.0, 1: 0.0, 2: 0.05, 3: 0.02, 4: -0.01, 5: 0.0, 6: 0.0}
    qqq, smh = [], []
    q_level, s_level = 200.0, 100.0
    for k, period in enumerate(uniq):
        n = int((periods == period).sum())
        q_target = q_level * (1 + qqq_month_ret[k])
        s_target = s_level * (1 + smh_month_ret[k])
        # linear ramp inside the month, ending exactly at the target close
        qqq.extend(np.linspace(q_level, q_target, n + 1)[1:])
        smh.extend(np.linspace(s_level, s_target, n + 1)[1:])
        q_level, s_level = q_target, s_target
    frames = {"SMH": _frame(index, smh), "QQQ": _frame(index, qqq)}
    signal_fn = mod.make_signal_fn("SMH", index, frames)
    decisions = _decisions(signal_fn, frames["SMH"])

    expected_by_month = {1: (True, False), 2: (False, True), 3: (False, True),
                         4: (True, False)}
    for k, month_end in enumerate(month_ends):
        pos = index.get_loc(month_end)
        if k == 0:
            # first month-end has no prior month-end close -> no decision
            assert decisions[pos] == (False, False)
        elif k in expected_by_month:
            assert decisions[pos] == expected_by_month[k], (k, month_end)
    # every non-month-end bar is silent
    month_end_set = set(month_ends)
    for pos, ts in enumerate(index):
        if ts not in month_end_set:
            assert decisions[pos] == (False, False), ts


def test_stays_flat_when_spread_never_positive():
    index = _bdays("2020-01-01", 260)
    frames = _flat_frames(index)
    signal_fn = mod.make_signal_fn("SMH", index, frames)
    decisions = _decisions(signal_fn, frames["SMH"])
    assert not any(entry for entry, _ in decisions)
    trades, in_market = backtest_engine.run_backtest(frames["SMH"], signal_fn)
    assert len(trades) == 0
    assert not in_market.any()


def test_signal_only_on_common_calendar_month_end():
    # QQQ is missing SMH's last session of January, so on the COMMON
    # calendar January's month-end is the prior session.
    index = _bdays("2020-01-01", 60)
    smh_close = np.linspace(100.0, 90.0, len(index))   # SMH falling
    qqq_close = np.linspace(200.0, 220.0, len(index))  # QQQ rising -> True
    smh = _frame(index, smh_close)
    jan = index[index.to_period("M") == pd.Period("2020-01", "M")]
    qqq_index = index.drop(jan[-1])
    qqq = _frame(qqq_index, qqq_close[index.get_indexer(qqq_index)])
    signal_fn = mod.make_signal_fn("SMH", index, {"SMH": smh, "QQQ": qqq})
    decisions = _decisions(signal_fn, smh)
    # SMH's true Jan month-end is not in the common calendar -> silent there
    assert decisions[index.get_loc(jan[-1])] == (False, False)
    # the common-calendar Jan month-end is the previous session, but it has no
    # prior month-end -> silent; February's month-end fires (QQQ outperformed)
    feb = index[index.to_period("M") == pd.Period("2020-02", "M")]
    assert decisions[index.get_loc(feb[-1])] == (True, False)


# ------------------------------------------------------------ look-ahead

def test_lookahead_probe_future_prices_never_change_past_signals():
    rng = np.random.default_rng(7)
    index = _bdays("2018-01-01", 500)
    smh_close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    qqq_close = 200.0 * np.exp(np.cumsum(rng.normal(0, 0.015, len(index))))
    base = {"SMH": _frame(index, smh_close), "QQQ": _frame(index, qqq_close)}
    k = 300
    before = _decisions(mod.make_signal_fn("SMH", index, base), base["SMH"])[:k + 1]
    perturbed = {}
    for symbol, frame in base.items():
        frame = frame.copy()
        frame.iloc[k + 1:, frame.columns.get_loc("Close")] *= 1.3
        frame.iloc[k + 1:, frame.columns.get_loc("High")] *= 1.5
        frame.iloc[k + 1:, frame.columns.get_loc("Low")] *= 0.7
        frame.iloc[k + 1:, frame.columns.get_loc("Open")] *= 1.2
        perturbed[symbol] = frame
    after = _decisions(mod.make_signal_fn("SMH", index, perturbed),
                       perturbed["SMH"])[:k + 1]
    assert before == after
    assert any(entry for entry, _ in before), "probe frame must actually fire"


def test_holdout_bucket_uses_only_warmup_history_before_it():
    rng = np.random.default_rng(11)
    full_index = _bdays("2018-01-01", 700)
    smh_close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(full_index))))
    qqq_close = 200.0 * np.exp(np.cumsum(rng.normal(0, 0.015, len(full_index))))
    frames = {"SMH": _frame(full_index, smh_close), "QQQ": _frame(full_index, qqq_close)}
    holdout = full_index[500:620]  # a bucket strictly inside the frames
    signal_fn = mod.make_signal_fn("SMH", holdout, frames)
    bucket = frames["SMH"].loc[holdout]
    decisions = _decisions(signal_fn, bucket)
    # the bucket's first month-end still decides, using the pre-bucket
    # month-end close as its anchor (warm-up from history before the bucket)
    first_me = _month_ends(holdout)[0]
    assert decisions[holdout.get_loc(first_me)] != (False, False)
    # and an identical bucket run on frames truncated right after it agrees
    truncated = {s: f.loc[: holdout[-1] + pd.Timedelta(days=45)] for s, f in frames.items()}
    same = _decisions(mod.make_signal_fn("SMH", holdout, truncated), bucket)
    assert same == decisions


# ------------------------------------------------------------ end-to-end

def test_end_to_end_run_backtest_produces_trades():
    rng = np.random.default_rng(3)
    index = _bdays("2019-01-01", 600)
    smh_close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    qqq_close = 200.0 * np.exp(np.cumsum(rng.normal(0, 0.015, len(index))))
    frames = {"SMH": _frame(index, smh_close), "QQQ": _frame(index, qqq_close)}
    signal_fn = mod.make_signal_fn("SMH", index, frames)
    trades, in_market = backtest_engine.run_backtest(frames["SMH"], signal_fn, cost_bps=8.7)
    assert len(trades) >= 1
    month_end_set = set(_month_ends(index))
    for _, row in trades.iterrows():
        # entry fills the bar after a month-end decision
        entry_pos = index.get_loc(row["entry_date"])
        assert index[entry_pos - 1] in month_end_set
        # signal exits land on a month-end close
        if row["reason"] == "signal_exit":
            assert row["exit_date"] in month_end_set
    # a real holding period, never a one-bar flip
    assert (trades["held"] >= 15).all()


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _bdays("2020-01-01", 40)
    frames = _flat_frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SMH", index, {"SMH": frames["SMH"]})


def test_signal_fn_is_cheap_per_call():
    """The engine calls once per bar with a growing slice: the per-bar work
    must be array lookups, so a whole pass over 5000 bars is fast."""
    import time
    index = _bdays("2005-01-01", 5000)
    rng = np.random.default_rng(1)
    frames = {"SMH": _frame(index, 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 5000)))),
              "QQQ": _frame(index, 200.0 * np.exp(np.cumsum(rng.normal(0, 0.015, 5000))))}
    signal_fn = mod.make_signal_fn("SMH", index, frames)
    frame = frames["SMH"]
    started = time.perf_counter()
    for i in range(0, 5000, 50):  # 100 calls with slices of growing size
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 2.0
