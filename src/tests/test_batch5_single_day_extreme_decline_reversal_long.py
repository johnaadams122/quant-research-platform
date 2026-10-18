"""Tests for src/mining/batch5/single_day_extreme_decline_reversal_long.py.

Synthetic data only (no network), deterministic, fast. Mirrors the
conventions of test_batch5_oil_price_change_equity_underreaction.py
(contract tests, look-ahead probe, holdout-blind probe, end-to-end
run_backtest) and test_batch5_stock_split_post_split_drift_pooled.py
(entry_exit_positions overlap/tail-hold edge cases for a fixed-hold-days
mechanism the ABI-1 driver runs without an engine-level time_stop).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining.backtest_engine import SignalResult, run_backtest  # noqa: E402
from mining.batch5 import (  # noqa: E402
    single_day_extreme_decline_reversal_long as mod)

VOL_WINDOW = mod.VOL_WINDOW_DAYS if hasattr(mod, "VOL_WINDOW_DAYS") else 20
HOLD_DAYS = mod.HOLD_DAYS if hasattr(mod, "HOLD_DAYS") else 3


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2015-01-02", periods=300) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


AMPLITUDE = 0.003  # ordinary day-to-day return magnitude, see _ordinary_returns
# Closed-form sample std (ddof=1) of a perfectly alternating +/-AMPLITUDE
# sequence over VOL_WINDOW points: n*AMPLITUDE^2/(n-1) under the radical.
_EFFECTIVE_TRAILING_STD = AMPLITUDE * np.sqrt(VOL_WINDOW / (VOL_WINDOW - 1))


def _ordinary_returns(n: int, *, amplitude: float = AMPLITUDE) -> np.ndarray:
    """Deterministic alternating +/-amplitude daily returns -- bounded and
    exactly reproducible (no random tail risk that could spuriously cross
    the z-score threshold on an "ordinary" day, unlike a Gaussian draw)."""
    pattern = np.empty(n)
    pattern[0::2] = amplitude
    pattern[1::2] = -amplitude
    return pattern


def _frame_with_shock(n: int, shock_pos: int, shock_return: float, *,
                      amplitude: float = AMPLITUDE) -> pd.DataFrame:
    """Ordinary deterministic-vol synthetic SPY closes with ONE manufactured
    single-session return of `shock_return` at `shock_pos`; every other
    session (before AND after) keeps the ordinary alternating return, so the
    trailing vol window is only ever contaminated by the shock once the
    window has rolled fully past it."""
    index = _calendar(periods=n)
    rets = _ordinary_returns(n, amplitude=amplitude)
    rets = rets.copy()
    rets[shock_pos] = shock_return
    level = np.empty(n)
    level[0] = 100.0
    for i in range(1, n):
        level[i] = level[i - 1] * (1.0 + rets[i])
    return _ohlc_from_close(level, index)


def _frames(frame: pd.DataFrame) -> dict:
    return {"SPY": frame}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_expected_contract():
    assert mod.FAMILY_ID == "single_day_extreme_decline_reversal_long"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SPY"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 0
    assert callable(mod.make_signal_fn)
    assert mod.PARAMETERS["entry_zscore_threshold"] == -2.5
    assert mod.PARAMETERS["holding_period_days"] == 3


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["SPY"]
    assert spec["signal"] == []


# ------------------------------------------------------------------- rule

def test_manufactured_extreme_decline_day_fires_entry():
    n = 120
    shock_pos = 80  # well past the VOL_WINDOW+1 warm-up
    frame = _frame_with_shock(n, shock_pos, shock_return=-0.10)
    signal_fn = mod.make_signal_fn("SPY", frame.index, _frames(frame))
    sigs = _signals(signal_fn, frame)
    assert sigs[shock_pos].entry is True
    assert sigs[shock_pos].exit is False


def test_no_signal_on_ordinary_days():
    n = 150
    shock_pos = 100
    frame = _frame_with_shock(n, shock_pos, shock_return=-0.12)
    signal_fn = mod.make_signal_fn("SPY", frame.index, _frames(frame))
    sigs = _signals(signal_fn, frame)
    fill = shock_pos + 1
    exit_pos = fill + HOLD_DAYS
    for i, sig in enumerate(sigs):
        if i == shock_pos:
            assert sig == SignalResult(entry=True, exit=False), i
        elif i == exit_pos:
            assert sig == SignalResult(entry=False, exit=True), i
        else:
            assert sig == SignalResult(entry=False, exit=False), i


def test_moderate_decline_does_not_trigger():
    # A decline that is real but not extreme (|z| well under 2.5) must not
    # fire -- target z == -1.5 exactly, using the ordinary pattern's own
    # closed-form trailing std so the margin below -2.5 is not accidental.
    n = 120
    shock_pos = 80
    shock_return = -1.5 * _EFFECTIVE_TRAILING_STD
    frame = _frame_with_shock(n, shock_pos, shock_return=shock_return)
    signal_fn = mod.make_signal_fn("SPY", frame.index, _frames(frame))
    sigs = _signals(signal_fn, frame)
    assert all(not s.entry for s in sigs)


def test_hold_is_exactly_three_sessions_end_to_end():
    n = 150
    shock_pos = 90
    frame = _frame_with_shock(n, shock_pos, shock_return=-0.11)
    signal_fn = mod.make_signal_fn("SPY", frame.index, _frames(frame))
    trades, in_market = run_backtest(frame, signal_fn)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["held"] == HOLD_DAYS == 3
    assert row["entry_date"] == frame.index[shock_pos + 1]
    assert row["reason"] == "signal_exit"
    assert in_market.iloc[shock_pos + 1:shock_pos + 1 + HOLD_DAYS + 1].all()


def test_overlapping_trigger_during_open_hold_is_skipped():
    n = 200
    first = 60
    frame = _frame_with_shock(n, first, shock_return=-0.10)
    # Manufacture a SECOND shock two sessions after the first (inside the
    # 3-session hold window) directly on the already-built frame.
    close = frame["Close"].to_numpy().copy()
    second = first + 2
    close[second:] = close[second:] / close[second] * close[second - 1] * (1.0 - 0.10)
    frame2 = _ohlc_from_close(close, frame.index)
    signal_fn = mod.make_signal_fn("SPY", frame2.index, _frames(frame2))
    trades, _ = run_backtest(frame2, signal_fn)
    # Only the first trigger produces a trade; the engine never even asks
    # about entry while a position is open, and this family's own
    # precomputed exit array must not be confused by the second trigger.
    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == frame2.index[first + 1]
    assert trades.iloc[0]["held"] == HOLD_DAYS


def test_trigger_at_bucket_tail_reports_no_entry_when_no_fill_bar_exists():
    n = 100
    shock_pos = n - 1  # last bar: no next-bar open exists to fill at
    frame = _frame_with_shock(n, shock_pos, shock_return=-0.15)
    signal_fn = mod.make_signal_fn("SPY", frame.index, _frames(frame))
    sigs = _signals(signal_fn, frame)
    assert all(not s.entry for s in sigs)


def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    n = 200
    frame = _frame_with_shock(n, 90, shock_return=-0.09)
    k = 120
    base_fn = mod.make_signal_fn("SPY", frame.index, _frames(frame))
    base = _signals(base_fn, frame, upto=k)

    perturbed = frame.copy()
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] *= 1.4
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Open")] *= 1.4
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("High")] *= 1.6
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Low")] *= 0.6
    pert_fn = mod.make_signal_fn("SPY", frame.index, _frames(perturbed))
    after = _signals(pert_fn, perturbed, upto=k)
    assert base == after


def test_holdout_blind_frames_extending_past_index_are_ignored():
    n = 250
    frame = _frame_with_shock(n, 90, shock_return=-0.10)
    bucket = frame.index[:180]
    fn_full = mod.make_signal_fn("SPY", bucket, _frames(frame))
    truncated = frame.loc[frame.index <= bucket[-1]]
    fn_trunc = mod.make_signal_fn("SPY", bucket, _frames(truncated))
    bucket_frame = frame.loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_warm_up_history_before_bucket_start_is_used():
    # A holdout bucket whose FIRST bar is the shock day itself must still
    # fire: its trailing VOL_WINDOW-session realized vol can only be
    # computed from history strictly BEFORE the bucket, which is available
    # only because the factory reads the full frame (truncated at the end,
    # never at the bucket's own start) rather than the bucket alone.
    n = 200
    shock_pos = 90
    frame = _frame_with_shock(n, shock_pos, shock_return=-0.10)
    bucket = frame.index[shock_pos:]  # the shock day is the bucket's bar 0
    signal_fn = mod.make_signal_fn("SPY", bucket, _frames(frame))
    bucket_frame = frame.loc[bucket]
    sigs = _signals(signal_fn, bucket_frame)
    assert sigs[0] == SignalResult(entry=True, exit=False)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    frame = _frame_with_shock(60, 40, shock_return=-0.10)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", frame.index, _frames(frame))
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", frame.index, {})


def test_signal_fn_is_constant_time_per_call():
    import time
    n = 3000
    frame = _frame_with_shock(n, 1500, shock_return=-0.10)
    signal_fn = mod.make_signal_fn("SPY", frame.index, _frames(frame))
    started = time.perf_counter()
    for i in range(n):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
