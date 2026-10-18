"""Tests for src/mining/batch5/long_treasury_3x_fomc_window_drift.py.

Synthetic data only (no network), deterministic, fast. Statement dates are
synthetic calendar positions passed to the factory directly; no events file
is read.
"""
import json
import os
import sys
import time

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
from mining.batch5 import long_treasury_3x_fomc_window_drift as mod  # noqa: E402


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2015-01-02", periods=600) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _tmf_close(index: pd.DatetimeIndex, seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 30.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 3) -> dict:
    return {"TMF": _ohlc_from_close(_tmf_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto=None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


def _synthetic_statements(index: pd.DatetimeIndex, positions) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([index[p] for p in positions])


LONG = SignalResult(entry=True, exit=False)
FLAT = SignalResult(entry=False, exit=True)


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "long_treasury_3x_fomc_window_drift"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["TMF"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "TMF"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["TMF"]
    assert spec["signal"] == []
    assert driver.parameter_drift(spec["parameters"], mod.PARAMETERS) is None


# ------------------------------------------------------------ parameters

def test_window_arithmetic_matches_hold_sessions():
    p = mod.PARAMETERS
    assert (p["window_sessions_before"] + 1 + p["window_sessions_after"]
            == p["hold_sessions"] == 3)
    assert p["include_unscheduled_meetings"] is False


# ------------------------------------------------------------------- rule

def test_rule_fires_only_in_the_three_session_window():
    index = _calendar(periods=300)
    frames = _frames(index)
    statements = _synthetic_statements(index, [40, 120, 210])
    signal_fn = mod.make_signal_fn("TMF", index, frames,
                                   statement_dates=statements)
    sigs = _signals(signal_fn, frames["TMF"])
    long_positions = set()
    for s in (40, 120, 210):
        long_positions.update({s - 2, s - 1, s})
    for i, sig in enumerate(sigs):
        if i in long_positions:
            assert sig == LONG, (i, sig)
        else:
            assert sig == FLAT, (i, sig)


def test_engine_holds_open_of_e_minus_1_through_close_of_e_plus_1():
    index = _calendar(periods=300)
    frames = _frames(index)
    statements = _synthetic_statements(index, [40, 120, 210])
    signal_fn = mod.make_signal_fn("TMF", index, frames,
                                   statement_dates=statements)
    trades, in_market = run_backtest(frames["TMF"], signal_fn, cost_bps=5.0)
    assert len(trades) == 3
    for (_, row), s in zip(trades.iterrows(), (40, 120, 210)):
        assert row["entry_date"] == index[s - 1]
        assert row["exit_date"] == index[s + 1]
        assert row["held"] == 2  # exactly 3 sessions: e-1, e, e+1
        assert row["reason"] == "signal_exit"
        assert row["entry_px"] == frames["TMF"]["Open"].iloc[s - 1]
        assert row["exit_px"] == frames["TMF"]["Close"].iloc[s + 1]
    assert int(in_market.sum()) == 9


def test_statement_not_a_session_anchors_to_next_session():
    # Interpretation choice: a statement date absent from the traded calendar
    # (never the case on real data) anchors to the first session on/after it.
    full = _calendar(periods=200)
    missing = full[100]
    index = full.drop(missing)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("TMF", index, frames,
                                   statement_dates=pd.DatetimeIndex([missing]))
    sigs = _signals(signal_fn, frames["TMF"])
    anchor = int(index.get_loc(full[101]))
    for i, sig in enumerate(sigs):
        assert sig == (LONG if i in (anchor - 2, anchor - 1, anchor) else FLAT)


def test_event_beyond_last_bar_and_event_at_start_are_safe():
    index = _calendar(periods=100)
    frames = _frames(index)
    beyond = index[-1] + pd.Timedelta(days=7)
    at_start = index[1]  # s(e)-2 would be negative
    signal_fn = mod.make_signal_fn(
        "TMF", index, frames,
        statement_dates=pd.DatetimeIndex([at_start, beyond]))
    sigs = _signals(signal_fn, frames["TMF"])
    assert sigs[0] == LONG and sigs[1] == LONG
    assert all(sig == FLAT for sig in sigs[2:])


def test_statement_before_calendar_start_is_skipped_not_anchored():
    # A pre-inception statement must not be anchored to TMF's first bar.
    index = _calendar(periods=100)
    frames = _frames(index)
    before = index[0] - pd.Timedelta(days=10)
    signal_fn = mod.make_signal_fn("TMF", index, frames,
                                   statement_dates=pd.DatetimeIndex([before]))
    assert all(sig == FLAT for sig in _signals(signal_fn, frames["TMF"]))


def test_prices_never_enter_the_signal():
    index = _calendar(periods=300)
    statements = _synthetic_statements(index, [50, 150, 250])
    a = mod.make_signal_fn("TMF", index, _frames(index, seed=1),
                           statement_dates=statements)
    b = mod.make_signal_fn("TMF", index, _frames(index, seed=99),
                           statement_dates=statements)
    assert (_signals(a, _frames(index, seed=1)["TMF"])
            == _signals(b, _frames(index, seed=99)["TMF"]))


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=500)
    frames = _frames(index, seed=7)
    statements = _synthetic_statements(index, [30, 100, 170, 240, 320, 400, 470])
    k = 310
    base_fn = mod.make_signal_fn("TMF", index, frames, statement_dates=statements)
    base = _signals(base_fn, frames["TMF"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.3
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.3
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.7
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("TMF", index, perturbed, statement_dates=statements)
    after = _signals(pert_fn, perturbed["TMF"], upto=k)
    assert base == after


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=600)
    frames = _frames(index, seed=11)
    statements = _synthetic_statements(index, [30, 100, 170, 240, 320, 400,
                                               449, 452, 520, 590])
    bucket = index[:450]
    fn_full = mod.make_signal_fn("TMF", bucket, frames, statement_dates=statements)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("TMF", bucket, truncated, statement_dates=statements)
    bucket_frame = frames["TMF"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)
    # The statement two sessions past the bucket end (position 452 is 3
    # past 449, the last bar) produces no signal inside the bucket; the one
    # AT the last bar does.
    sigs = _signals(fn_full, bucket_frame)
    assert sigs[449] == LONG and sigs[448] == LONG and sigs[447] == LONG
    assert sigs[446] == FLAT


def test_holdout_bucket_uses_full_history_calendar_positions():
    # A bucket starting mid-history: statement positions are resolved on the
    # full traded calendar, then reindexed onto the bucket.
    index = _calendar(periods=400)
    frames = _frames(index)
    statements = _synthetic_statements(index, [250, 320])
    bucket = index[249:]
    signal_fn = mod.make_signal_fn("TMF", bucket, frames, statement_dates=statements)
    bucket_frame = frames["TMF"].loc[bucket]
    sigs = _signals(signal_fn, bucket_frame)
    # bucket[0] == index[249] == s(e)-1 -> still inside the window.
    assert sigs[0] == LONG and sigs[1] == LONG and sigs[2] == FLAT
    assert sigs[320 - 249 - 2] == LONG and sigs[320 - 249] == LONG
    assert sigs[320 - 249 + 1] == FLAT


# ----------------------------------------------------------- end to end

def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("TLT", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("TMF", index, {"SPY": frames["TMF"]})
    with pytest.raises(ValueError):
        mod.make_signal_fn("TMF", index[:0], frames)


def test_signal_fn_is_constant_time_per_call():
    index = _calendar(start="2006-01-03", periods=5000)
    frames = _frames(index, seed=5)
    statements = _synthetic_statements(index, range(40, 5000, 35))
    signal_fn = mod.make_signal_fn("TMF", index, frames, statement_dates=statements)
    frame = frames["TMF"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
