"""Tests for src/mining/batch5/large_cap_single_stock_pooled_tsmom.py.

Synthetic frames only (no network, deterministic); nothing under
data/mining is touched.
"""
import importlib
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

FAMILY_ID = "large_cap_single_stock_pooled_tsmom"

mod = importlib.import_module(f"mining.batch5.{FAMILY_ID}")


# ------------------------------------------------------------ synthetic data

def _frame_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _rising(n_days: int = 560, start: str = "2020-01-01") -> pd.DataFrame:
    index = _bdays(start, n_days)
    close = 100.0 * (1.0 + 0.001) ** np.arange(n_days)
    return _frame_from_close(close, index)


def _falling(n_days: int = 560, start: str = "2020-01-01") -> pd.DataFrame:
    index = _bdays(start, n_days)
    close = 100.0 * (1.0 - 0.001) ** np.arange(n_days)
    return _frame_from_close(close, index)


def _flat(n_days: int = 560, start: str = "2020-01-01") -> pd.DataFrame:
    index = _bdays(start, n_days)
    return _frame_from_close(np.full(n_days, 50.0), index)


def _month_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    periods = index.to_period("M")
    is_last = np.append(periods[1:] != periods[:-1], True)
    return np.flatnonzero(is_last)


def _signals(frame: pd.DataFrame, index=None, upto=None) -> list:
    index = frame.index if index is None else index
    fn = mod.make_signal_fn("AAA", index, {"AAA": frame})
    bucket = frame.loc[index]
    n = len(bucket) if upto is None else upto
    return [fn(bucket.iloc[:i + 1], i) for i in range(n)]


# ----------------------------------------------------------- contract fields

def test_contract_constants():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "pooled_single_asset"
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)
    assert not hasattr(mod, "make_target_fn")


# --------------------------------------------------------------- rule fires

def test_rising_stock_enters_at_month_ends_after_twelve_months_and_never_exits():
    frame = _rising()
    sigs = _signals(frame)
    me = _month_end_positions(frame.index)
    periods = frame.index.to_period("M")
    first_period = periods[0]
    for pos in me:
        months_of_history = (periods[pos] - first_period).n  # prior months present
        sig = sigs[pos]
        if months_of_history >= 12:
            assert sig.entry is True, frame.index[pos]
        else:
            assert sig.entry is False, frame.index[pos]
        assert sig.exit is False
    # Nothing fires away from a month-end decision bar (exits would land on
    # the bar after a month end; there are none for a rising stock).
    for pos, sig in enumerate(sigs):
        if pos not in set(me):
            assert sig == SignalResult(), frame.index[pos]


def test_falling_stock_never_enters_and_exits_on_the_bar_after_month_end():
    frame = _falling()
    sigs = _signals(frame)
    me = _month_end_positions(frame.index)
    periods = frame.index.to_period("M")
    first_period = periods[0]
    exit_positions = set()
    for pos in me:
        if (periods[pos] - first_period).n >= 12 and pos + 1 < len(frame):
            exit_positions.add(pos + 1)
    assert exit_positions, "test frame must reach a decision"
    for pos, sig in enumerate(sigs):
        assert sig.entry is False, frame.index[pos]
        assert sig.exit is (pos in exit_positions), frame.index[pos]


def test_zero_twelve_month_return_is_flat():
    frame = _flat()
    sigs = _signals(frame)
    assert not any(sig.entry for sig in sigs)
    assert any(sig.exit for sig in sigs)


def test_missing_calendar_month_suppresses_the_decision():
    frame = _rising(n_days=700)
    # Remove every bar of one calendar month in the middle of the history.
    drop_period = frame.index.to_period("M")[300]
    keep = frame.index.to_period("M") != drop_period
    gapped = frame.loc[keep]
    sigs = _signals(gapped)
    periods = gapped.index.to_period("M")
    me = _month_end_positions(gapped.index)
    for pos in me:
        period = periods[pos]
        needs_gap = any((period - k) == drop_period for k in range(1, 13))
        if needs_gap:
            assert sigs[pos] == SignalResult(), gapped.index[pos]
        elif (period - periods[0]).n >= 12 and period > drop_period:
            # Once the gap month has rolled out of the 12-month lookback the
            # rule decides again.
            assert sigs[pos].entry is True, gapped.index[pos]


def test_decision_uses_month_end_close_ratio_exactly():
    """ret12 = M[m] / M[m-12] - 1 with M[k] the LAST close of month k.
    Build a series where the month-end close alone flips the sign."""
    index = _bdays("2020-01-01", 420)
    close = np.full(len(index), 100.0)
    me = _month_end_positions(index)
    # Month-end closes: 100 for the first 13 months, then the 14th month's
    # last close is pushed above 100 by a single-bar spike while every other
    # bar stays at 100 -> LONG decided at that month end only.
    spike_pos = me[13]
    close[spike_pos] = 101.0
    frame = _frame_from_close(close, index)
    sigs = _signals(frame)
    for pos in me:
        expected_long = bool(pos == spike_pos)
        assert sigs[pos].entry is expected_long, index[pos]
    # The month AFTER the spike compares 100 vs the 12-months-prior close 100
    # -> FLAT -> exit on the following bar.
    nxt = me[14]
    assert sigs[nxt].entry is False
    assert sigs[nxt + 1].exit is True


# ------------------------------------------------------------ look-ahead

def test_look_ahead_probe_perturbing_future_bars_leaves_past_signals_unchanged():
    frame = _rising(n_days=600)
    k = 430
    before = _signals(frame, upto=k + 1)
    perturbed = frame.copy()
    after = perturbed.index[k + 1:]
    perturbed.loc[after, "Close"] *= 1.3
    perturbed.loc[after, "Open"] *= 1.3
    perturbed.loc[after, "High"] = perturbed.loc[after, "High"] * 1.3 + 5.0
    perturbed.loc[after, "Low"] = perturbed.loc[after, "Low"] * 1.3 - 5.0
    # Also crash the stock after k so a look-ahead would change decisions:
    # closes fall to half their level and keep falling, which turns the
    # 12-month return negative at every later month end.
    crash = float(frame["Close"].iloc[k]) * 0.5 * (0.99 ** np.arange(len(after)))
    perturbed.loc[after, "Close"] = crash
    again = _signals(perturbed, upto=k + 1)
    assert before == again
    # Sanity: the perturbation actually changes at least one later signal.
    assert _signals(frame) != _signals(perturbed)


def test_signal_depends_only_on_the_bar_date_not_the_slice_contents():
    """O(1) per call: the factory precomputes everything; the per-bar call
    only reads the slice's last date and the position."""
    frame = _rising()
    index = frame.index
    fn = mod.make_signal_fn("AAA", index, {"AAA": frame})
    me = _month_end_positions(index)
    pos = int(me[15])
    full = fn(frame.iloc[:pos + 1], pos)
    one_row = fn(frame.iloc[pos:pos + 1], pos)
    assert full == one_row
    assert full.entry is True


def test_holdout_bucket_positions_map_to_the_same_dates_as_the_full_frame():
    frame = _rising(n_days=700)
    full = _signals(frame)
    bucket_index = frame.index[520:]
    bucket_sigs = _signals(frame, index=bucket_index)
    for j, date in enumerate(bucket_index):
        pos = frame.index.get_loc(date)
        assert bucket_sigs[j] == full[pos], date


def test_holdout_blind_history_after_the_bucket_is_ignored():
    frame = _rising(n_days=700)
    bucket_index = frame.index[300:520]
    reference = _signals(frame, index=bucket_index)
    extended = frame.copy()
    tail = extended.index[520:]
    extended.loc[tail, ["Open", "High", "Low", "Close"]] *= 0.01
    assert _signals(extended, index=bucket_index) == reference


# ----------------------------------------------------------- end to end

def test_end_to_end_backtest_produces_a_real_holding_period():
    n = 600
    index = _bdays("2021-01-01", n)
    up = 100.0 * (1.0 + 0.0015) ** np.arange(320)
    down = up[-1] * (1.0 - 0.003) ** np.arange(1, n - 320 + 1)
    frame = _frame_from_close(np.concatenate([up, down]), index)
    fn = mod.make_signal_fn("AAA", index, {"AAA": frame})
    trades, in_market = backtest_engine.run_backtest(frame, fn)
    assert len(trades) >= 1
    assert (trades["held"] > 1).all()
    assert "signal_exit" in set(trades["reason"])
    first = trades.iloc[0]
    # Entered at the open of the bar after a month-end decision.
    entry_pos = frame.index.get_loc(first["entry_date"])
    assert entry_pos - 1 in set(_month_end_positions(index))
    assert first["entry_px"] == pytest.approx(float(frame["Open"].iloc[entry_pos]))
    # Exit fills at the close of the first session of a new month.
    exit_pos = frame.index.get_loc(first["exit_date"])
    assert exit_pos - 1 in set(_month_end_positions(index))
    assert in_market.sum() >= 2


def test_pooled_factory_is_per_ticker():
    rising, falling = _rising(), _falling()
    frames = {"UP": rising, "DN": falling}
    up_fn = mod.make_signal_fn("UP", rising.index, frames)
    dn_fn = mod.make_signal_fn("DN", falling.index, frames)
    pos = int(_month_end_positions(rising.index)[15])
    assert up_fn(rising.iloc[:pos + 1], pos).entry is True
    assert dn_fn(falling.iloc[:pos + 1], pos).entry is False
