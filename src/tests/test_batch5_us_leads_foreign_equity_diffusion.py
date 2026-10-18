"""Tests for src/mining/batch5/us_leads_foreign_equity_diffusion.py.

Synthetic data only (no network), deterministic, fast.
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
from mining.batch5 import us_leads_foreign_equity_diffusion as mod  # noqa: E402

MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           "us_leads_foreign_equity_diffusion.py")


# ------------------------------------------------------------ synthetic data

def _calendar(start="2020-01-01", periods=600) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _spy_alternating(index: pd.DatetimeIndex, up_first=True) -> pd.DataFrame:
    """SPY whose calendar-month return alternates sign: months with an even
    ordinal (counting from the first month in `index`) finish ABOVE the prior
    month-end close, odd ordinals finish BELOW. Within a month the close
    walks linearly from the prior month-end level to the new one, so the
    sign of the month return is unambiguous and never exactly zero."""
    periods = index.to_period("M")
    uniq = list(dict.fromkeys(periods))
    levels = {}
    level = 100.0
    for k, p in enumerate(uniq):
        up = (k % 2 == 0) if up_first else (k % 2 == 1)
        level = level * (1.05 if up else 0.95)
        levels[p] = level
    close = np.empty(len(index))
    prev_level = 100.0 / (1.05 if up_first else 0.95)
    for k, p in enumerate(uniq):
        pos = np.flatnonzero(periods == p)
        start = prev_level if k > 0 else 100.0
        end = levels[p]
        close[pos] = np.linspace(start, end, len(pos) + 1)[1:]
        prev_level = end
    return _ohlc_from_close(close, index)


def _efa_noise(index: pd.DatetimeIndex, seed=7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index))))
    return _ohlc_from_close(close, index)


def _month_end_positions(index: pd.DatetimeIndex) -> list:
    """Positions whose NEXT bar is in a different calendar month (the final
    bar is excluded -- there is no next bar to compare against)."""
    periods = index.to_period("M")
    return [i for i in range(len(index) - 1) if periods[i] != periods[i + 1]]


def _collect(signal_fn, df: pd.DataFrame, upto=None):
    n = len(df) if upto is None else upto + 1
    return [(bool(s.entry), bool(s.exit)) for s in
            (signal_fn(df.iloc[:i + 1], i) for i in range(n))]


# -------------------------------------------------------------------- tests

def test_contract_constants():
    assert mod.FAMILY_ID == "us_leads_foreign_equity_diffusion"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["EFA"]
    assert mod.SIGNAL == ["SPY"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "EFA"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES == 60


def test_rule_fires_on_month_ends_with_positive_spy_month_and_flat_otherwise():
    index = _calendar()
    spy = _spy_alternating(index, up_first=True)
    efa = _efa_noise(index)
    frames = {"EFA": efa, "SPY": spy}
    fn = mod.make_signal_fn("EFA", index, frames)
    sigs = _collect(fn, efa)

    me = _month_end_positions(index)
    assert len(me) >= 20
    periods = index.to_period("M")
    uniq = list(dict.fromkeys(periods))
    fired_true = fired_false = 0
    for pos in me:
        k = uniq.index(periods[pos])
        if k == 0:
            # First month-end has no prior month-end anchor in the data:
            # no decision is possible, must be a no-op.
            assert sigs[pos] == (False, False)
            continue
        spy_up = (k % 2 == 0)
        if spy_up:
            assert sigs[pos] == (True, False), (pos, index[pos])
            fired_true += 1
        else:
            assert sigs[pos] == (False, True), (pos, index[pos])
            fired_false += 1
    assert fired_true >= 9 and fired_false >= 9
    non_me = set(range(len(index))) - set(me)
    for pos in non_me:
        assert sigs[pos] == (False, False), (pos, index[pos])


def test_signal_ignores_efa_own_history():
    index = _calendar()
    spy = _spy_alternating(index)
    a = mod.make_signal_fn("EFA", index, {"EFA": _efa_noise(index, 1), "SPY": spy})
    b = mod.make_signal_fn("EFA", index, {"EFA": _efa_noise(index, 2), "SPY": spy})
    efa1, efa2 = _efa_noise(index, 1), _efa_noise(index, 2)
    assert _collect(a, efa1) == _collect(b, efa2)


def test_zero_month_return_is_not_a_hold():
    index = _calendar(periods=130)
    # Flat SPY: every month return is exactly 0 -> never > 0 -> exit on
    # every decision bar, never entry.
    spy = _ohlc_from_close(np.full(len(index), 100.0), index)
    efa = _efa_noise(index)
    fn = mod.make_signal_fn("EFA", index, {"EFA": efa, "SPY": spy})
    sigs = _collect(fn, efa)
    me = _month_end_positions(index)
    assert all(sigs[p] == (False, True) for p in me[1:])
    assert not any(s[0] for s in sigs)


def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar()
    spy = _spy_alternating(index)
    efa = _efa_noise(index)
    fn = mod.make_signal_fn("EFA", index, {"EFA": efa, "SPY": spy})
    k = 337
    before = _collect(fn, efa, upto=k)

    def perturb(df):
        out = df.copy()
        out.iloc[k + 1:, out.columns.get_loc("Close")] *= 1.3
        out.iloc[k + 1:, out.columns.get_loc("High")] *= 1.45
        out.iloc[k + 1:, out.columns.get_loc("Low")] *= 0.7
        out.iloc[k + 1:, out.columns.get_loc("Open")] *= 1.2
        return out
    spy2, efa2 = perturb(spy), perturb(efa)
    fn2 = mod.make_signal_fn("EFA", index, {"EFA": efa2, "SPY": spy2})
    after = _collect(fn2, efa2, upto=k)
    assert before == after
    # And the perturbation was real: at least one later signal differs or
    # the SPY closes differ (guards against a vacuous probe).
    assert not spy2["Close"].iloc[k + 1:].equals(spy["Close"].iloc[k + 1:])


def test_holdout_bucket_uses_prior_history_for_warmup_and_nothing_after():
    full = _calendar(periods=700)
    spy = _spy_alternating(full)
    efa = _efa_noise(full)
    # Bucket = a later slice; frames carry history before AND after it.
    bucket = full[400:560]
    truncated = {"EFA": efa.loc[:bucket[-1]], "SPY": spy.loc[:bucket[-1]]}
    fn_full = mod.make_signal_fn("EFA", bucket, {"EFA": efa, "SPY": spy})
    fn_trunc = mod.make_signal_fn("EFA", bucket, truncated)
    efa_bucket = efa.loc[bucket]
    s_full, s_trunc = _collect(fn_full, efa_bucket), _collect(fn_trunc, efa_bucket)
    assert s_full == s_trunc
    # Warm-up: the FIRST month-end inside the bucket must already carry a
    # real decision because the prior month-end anchor comes from history
    # before the bucket.
    me = _month_end_positions(bucket)
    assert s_full[me[0]] in ((True, False), (False, True))
    assert any(s[0] for s in s_full) and any(s[1] for s in s_full)


def test_no_decision_without_spy_data():
    index = _calendar(periods=300)
    spy_short = _spy_alternating(index).iloc[150:]   # SPY history starts late
    efa = _efa_noise(index)
    fn = mod.make_signal_fn("EFA", index, {"EFA": efa, "SPY": spy_short})
    sigs = _collect(fn, efa)
    assert all(s == (False, False) for s in sigs[:150])
    assert any(s != (False, False) for s in sigs[150:])


def test_end_to_end_run_backtest_produces_monthly_trades():
    index = _calendar()
    spy = _spy_alternating(index, up_first=True)
    efa = _efa_noise(index)
    fn = mod.make_signal_fn("EFA", index, {"EFA": efa, "SPY": spy})
    trades, in_market = backtest_engine.run_backtest(efa, fn, cost_bps=5.0)
    assert len(trades) >= 1
    # Alternating months -> roughly one trade every two months over ~28
    # months of data.
    assert 10 <= len(trades) <= 16, len(trades)
    periods = index.to_period("M")
    for _, row in trades.iterrows():
        e = index.get_loc(row["entry_date"])
        # Entry fills at the FIRST session of a month (next open after the
        # month-end decision bar).
        assert periods[e] != periods[e - 1]
        if row["reason"] == "signal_exit":
            x = index.get_loc(row["exit_date"])
            assert periods[x] != periods[x + 1]
            # Held one calendar month of sessions (alternating pattern).
            assert 15 <= row["held"] <= 25
    assert in_market.sum() > 0


def test_consecutive_true_months_merge_into_one_trade():
    index = _calendar(periods=260)
    # SPY strictly rising every month -> always hold -> one trade, closed
    # only by end_of_data.
    spy = _ohlc_from_close(np.linspace(100, 150, len(index)), index)
    efa = _efa_noise(index)
    fn = mod.make_signal_fn("EFA", index, {"EFA": efa, "SPY": spy})
    trades, _ = backtest_engine.run_backtest(efa, fn)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "end_of_data"


def test_signal_fn_rejects_misaligned_frame():
    index = _calendar(periods=100)
    spy = _spy_alternating(index)
    efa = _efa_noise(index)
    fn = mod.make_signal_fn("EFA", index, {"EFA": efa, "SPY": spy})
    other = _calendar(start="2021-06-01", periods=100)
    efa_other = _efa_noise(other)
    with pytest.raises(ValueError):
        fn(efa_other.iloc[:5], 4)


def test_factory_requires_traded_ticker_and_frames():
    index = _calendar(periods=50)
    spy = _spy_alternating(index)
    efa = _efa_noise(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {"EFA": efa, "SPY": spy})
    with pytest.raises(ValueError):
        mod.make_signal_fn("EFA", index, {"EFA": efa})
