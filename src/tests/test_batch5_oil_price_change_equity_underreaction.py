"""Tests for src/mining/batch5/oil_price_change_equity_underreaction.py.

Synthetic data only (no network), deterministic, fast. The candidate's
contract constants are pinned here; the driver's own drift check compares
PARAMETERS against the catalog at run time.
"""
import json
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
from mining.batch5 import oil_price_change_equity_underreaction as mod  # noqa: E402

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


def _month_ends(index: pd.DatetimeIndex) -> list:
    """Last bar of every calendar month in ``index`` EXCEPT the final month
    (whose last bar cannot be known to be a month-end without a later bar)."""
    periods = index.to_period("M")
    out = []
    for k in range(len(index) - 1):
        if periods[k + 1] != periods[k]:
            out.append(index[k])
    return out


def _uso_close_with_monthly_signs(index: pd.DatetimeIndex, signs: dict) -> np.ndarray:
    """USO closes that are flat within a month and step at each month start so
    the month-over-month (month-end to month-end) return has the sign given
    in ``signs`` (keyed by pd.Period month -> +1 / -1 / 0). Months not listed
    get +1."""
    periods = index.to_period("M")
    level = 50.0
    close = np.empty(len(index))
    prev_period = None
    for k, period in enumerate(periods):
        if prev_period is not None and period != prev_period:
            sign = signs.get(period, 1)
            level = level * (1.0 + 0.05 * sign)
        close[k] = level
        prev_period = period
    return close


def _spy_close(index: pd.DatetimeIndex, seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 200.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))


def _frames(index: pd.DatetimeIndex, uso_close: np.ndarray, spy_seed: int = 3) -> dict:
    return {"SPY": _ohlc_from_close(_spy_close(index, spy_seed), index),
            "USO": _ohlc_from_close(uso_close, index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "oil_price_change_equity_underreaction"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SPY"]
    assert mod.SIGNAL == ["USO"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["SPY"]
    assert spec["signal"] == ["USO"]


# ------------------------------------------------------------------- rule

def test_rule_fires_at_month_end_and_only_there():
    index = _calendar(periods=400)
    periods = index.to_period("M")
    months = list(dict.fromkeys(periods))
    # Month k's sign = return from month-end(k-1) to month-end(k).
    signs = {months[1]: -1, months[2]: -1, months[3]: +1, months[4]: 0,
             months[5]: +1, months[6]: -1}
    uso = _uso_close_with_monthly_signs(index, signs)
    frames = _frames(index, uso)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])

    month_end_dates = _month_ends(index)
    by_date = {index[i]: sigs[i] for i in range(len(index))}
    me_by_period = {d.to_period("M"): d for d in month_end_dates}

    # First month-end has no prior month-end -> no signal at all.
    first = me_by_period[months[0]]
    assert by_date[first] == SignalResult(entry=False, exit=False)
    # Oil fell -> hold (entry=True, exit=False).
    assert by_date[me_by_period[months[1]]] == SignalResult(entry=True, exit=False)
    assert by_date[me_by_period[months[2]]] == SignalResult(entry=True, exit=False)
    # Oil rose -> flat (entry=False, exit=True).
    assert by_date[me_by_period[months[3]]] == SignalResult(entry=False, exit=True)
    # Oil unchanged (r_oil == 0) -> the rule says <= 0 -> hold.
    assert by_date[me_by_period[months[4]]] == SignalResult(entry=True, exit=False)
    assert by_date[me_by_period[months[5]]] == SignalResult(entry=False, exit=True)
    assert by_date[me_by_period[months[6]]] == SignalResult(entry=True, exit=False)

    # Every non-month-end bar is silent.
    month_end_set = set(month_end_dates)
    for i, date in enumerate(index):
        if date not in month_end_set:
            assert sigs[i] == SignalResult(entry=False, exit=False), date


def test_final_bar_of_index_is_never_a_decision_bar():
    # The last bar cannot be known to be a month-end without a later bar, and
    # the engine force-closes at end_of_data anyway.
    index = _calendar(periods=300)
    uso = _uso_close_with_monthly_signs(index, {})
    frames = _frames(index, uso)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    last = signal_fn(frames["SPY"], len(index) - 1)
    assert last == SignalResult(entry=False, exit=False)


def test_spy_prices_never_enter_the_signal():
    index = _calendar(periods=400)
    uso = _uso_close_with_monthly_signs(
        index, {p: (-1 if k % 2 else 1) for k, p in
                enumerate(dict.fromkeys(index.to_period("M")))})
    a = mod.make_signal_fn("SPY", index, _frames(index, uso, spy_seed=1))
    b = mod.make_signal_fn("SPY", index, _frames(index, uso, spy_seed=99))
    frame_a = _frames(index, uso, spy_seed=1)["SPY"]
    frame_b = _frames(index, uso, spy_seed=99)["SPY"]
    assert _signals(a, frame_a) == _signals(b, frame_b)


def test_dates_before_signal_history_stay_flat():
    # SPY exists long before USO: SPY bars with no USO bar must be silent.
    index = _calendar(periods=500)
    uso_index = index[250:]
    uso = _uso_close_with_monthly_signs(
        uso_index, {p: -1 for p in dict.fromkeys(uso_index.to_period("M"))})
    frames = {"SPY": _ohlc_from_close(_spy_close(index), index),
              "USO": _ohlc_from_close(uso, uso_index)}
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])
    for i in range(250):
        assert sigs[i] == SignalResult(entry=False, exit=False)
    assert any(s.entry for s in sigs[250:])


def test_missing_prior_month_end_gives_no_signal():
    # A gap month in the USO calendar breaks the one-month lookback: the
    # month-end after the gap must NOT compare against a two-month-old close.
    index = _calendar(periods=300)
    periods = index.to_period("M")
    months = list(dict.fromkeys(periods))
    uso_index = index[periods != months[3]]
    uso = _uso_close_with_monthly_signs(
        uso_index, {p: -1 for p in dict.fromkeys(uso_index.to_period("M"))})
    frames = {"SPY": _ohlc_from_close(_spy_close(index), index),
              "USO": _ohlc_from_close(uso, uso_index)}
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])
    by_date = {index[i]: sigs[i] for i in range(len(index))}
    me = {d.to_period("M"): d for d in _month_ends(uso_index)}
    assert by_date[me[months[2]]] == SignalResult(entry=True, exit=False)
    assert by_date[me[months[4]]] == SignalResult(entry=False, exit=False)
    assert by_date[me[months[5]]] == SignalResult(entry=True, exit=False)


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=500)
    rng = np.random.RandomState(7)
    uso = 60.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(index))))
    frames = _frames(index, uso)
    k = 310
    base_fn = mod.make_signal_fn("SPY", index, frames)
    base = _signals(base_fn, frames["SPY"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.3
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.3
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.7
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("SPY", index, perturbed)
    after = _signals(pert_fn, perturbed["SPY"], upto=k)
    assert base == after
    # The perturbation is real: SOME later signal differs.
    assert (_signals(base_fn, frames["SPY"])
            != _signals(pert_fn, perturbed["SPY"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=600)
    rng = np.random.RandomState(11)
    uso = 60.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(index))))
    frames = _frames(index, uso)
    bucket = index[:450]
    fn_full = mod.make_signal_fn("SPY", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("SPY", bucket, truncated)
    bucket_frame = frames["SPY"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_warm_up_history_before_the_bucket_is_used():
    # A holdout bucket starting mid-month must still see the prior month-end's
    # USO close (which lies BEFORE the bucket) to decide at its first month-end.
    index = _calendar(periods=400)
    periods = index.to_period("M")
    months = list(dict.fromkeys(periods))
    uso = _uso_close_with_monthly_signs(index, {p: -1 for p in months})
    frames = _frames(index, uso)
    start = int(np.flatnonzero(periods == months[5])[3])  # 4th bar of month 5
    bucket = index[start:]
    signal_fn = mod.make_signal_fn("SPY", bucket, frames)
    bucket_frame = frames["SPY"].loc[bucket]
    sigs = _signals(signal_fn, bucket_frame)
    first_me = _month_ends(bucket)[0]
    assert sigs[list(bucket).index(first_me)] == SignalResult(entry=True, exit=False)


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_monthly_trades():
    index = _calendar(periods=600)
    periods = index.to_period("M")
    months = list(dict.fromkeys(periods))
    # Alternate two down months, two up months.
    signs = {p: (-1 if (k // 2) % 2 == 0 else 1) for k, p in enumerate(months)}
    uso = _uso_close_with_monthly_signs(index, signs)
    frames = _frames(index, uso)
    bucket_frame = frames["SPY"]
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    trades, in_market = run_backtest(bucket_frame, signal_fn, cost_bps=1.0)
    assert len(trades) >= 1
    month_end_set = set(_month_ends(index))
    first_of_month = {index[i + 1] for i in range(len(index) - 1)
                      if periods[i + 1] != periods[i]}
    for _, row in trades.iterrows():
        assert row["entry_date"] in first_of_month
        if row["reason"] == "signal_exit":
            assert row["exit_date"] in month_end_set
        # Consecutive holds merged: each trade spans >= ~1 calendar month.
        assert row["held"] >= 15
    # Some trade spans two months (consecutive hold months merge).
    assert (trades["held"] >= 35).any()
    assert in_market.any()


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    uso = _uso_close_with_monthly_signs(index, {})
    frames = _frames(index, uso)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {"SPY": frames["SPY"]})


def test_signal_fn_is_constant_time_per_call():
    # ~5000 bars: the per-call body must be a few array lookups, so the whole
    # sweep (including the engine-style slicing) stays well under a second.
    import time
    index = _calendar(start="2006-01-03", periods=5000)
    rng = np.random.RandomState(5)
    uso = 60.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(index))))
    frames = _frames(index, uso)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    frame = frames["SPY"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
