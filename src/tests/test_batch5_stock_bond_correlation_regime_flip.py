"""Tests for src/mining/batch5/stock_bond_correlation_regime_flip.py.

Synthetic data only (no network), deterministic, fast. The candidate's
PARAMETERS are compared against the REAL curation file so drift is caught
here before the driver's own drift check ever runs.
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
from mining.batch5 import stock_bond_correlation_regime_flip as mod  # noqa: E402


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


def _close_from_returns(returns: np.ndarray, start_level: float = 100.0) -> np.ndarray:
    return start_level * np.exp(np.cumsum(returns))


def _corr_controlled_closes(index: pd.DatetimeIndex, *, window: int,
                             regimes: list) -> tuple:
    """Build SPY/TLT close series whose rolling W-day return correlation is
    driven to a controlled sign/magnitude in successive contiguous blocks.

    ``regimes`` is a list of (n_bars, target) tuples; ``target`` is either
    "pos" (SPY and TLT move together -> correlation strongly positive) or
    "neg" (SPY and TLT move oppositely -> correlation strongly negative).
    A tiny amount of idiosyncratic noise keeps returns non-degenerate
    (pandas .corr() on a perfectly collinear pair is still well-defined,
    but real markets never are, and the noise avoids any accidental exact
    zero-variance edge case)."""
    rng = np.random.RandomState(0)
    n = len(index)
    common_shock = rng.normal(0.0, 0.01, n)
    spy_idio = rng.normal(0.0, 0.0005, n)
    tlt_idio = rng.normal(0.0, 0.0005, n)
    sign = np.empty(n)
    pos = 0
    for n_bars, target in regimes:
        sign[pos:pos + n_bars] = 1.0 if target == "pos" else -1.0
        pos += n_bars
    assert pos == n, "regimes must cover the whole index"
    spy_ret = common_shock + spy_idio
    tlt_ret = sign * common_shock + tlt_idio
    spy_close = _close_from_returns(spy_ret, 300.0)
    tlt_close = _close_from_returns(tlt_ret, 90.0)
    return spy_close, tlt_close


def _frames(index: pd.DatetimeIndex, spy_close: np.ndarray,
           tlt_close: np.ndarray) -> dict:
    return {"SPY": _ohlc_from_close(spy_close, index),
            "TLT": _ohlc_from_close(tlt_close, index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "stock_bond_correlation_regime_flip"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SPY"]
    assert mod.SIGNAL == ["TLT"]
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
    assert spec["signal"] == ["TLT"]


def test_generic_abi1_dispatch_signature_is_usable():
    # run_abi1_mining.py's _build_curve calls make_signal_fn(symbol,
    # full_frame.index, module_frames) POSITIONALLY with no extra kwargs --
    # this module must be directly callable that way (unlike the bespoke
    # external-data drivers, e.g. bitcoin_extreme_decline_spillover_riskoff,
    # which need an extra kwarg and are NOT reachable through the generic
    # dispatch).
    index = _calendar(periods=200)
    spy_close, tlt_close = _corr_controlled_closes(
        index, window=mod._WINDOW, regimes=[(200, "neg")])
    frames = _frames(index, spy_close, tlt_close)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    assert callable(signal_fn)


# ------------------------------------------------------------------- rule

def test_warm_up_period_before_full_window_emits_no_signal():
    window = mod._WINDOW
    index = _calendar(periods=window + 40)
    spy_close, tlt_close = _corr_controlled_closes(
        index, window=window, regimes=[(window + 40, "neg")])
    frames = _frames(index, spy_close, tlt_close)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])
    # Fewer than `window` common-calendar return observations exist ->
    # rolling correlation is undefined for at least the first `window` bars.
    for i in range(window):
        assert sigs[i] == SignalResult(entry=False, exit=False), i


def test_negative_correlation_regime_holds_risk_on_entry():
    window = mod._WINDOW
    index = _calendar(periods=window + 80)
    spy_close, tlt_close = _corr_controlled_closes(
        index, window=window, regimes=[(window + 80, "neg")])
    frames = _frames(index, spy_close, tlt_close)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])
    # Once the window is full, a strongly negative correlation regime must
    # emit entry=True (risk-on, hold SPY) on every subsequent bar (the
    # signal function is stateless -- the engine only reads `entry` while
    # flat, so repeating entry=True every day is harmless and correct).
    assert any(s.entry for s in sigs[window:])
    assert not any(s.exit for s in sigs[window:])


def test_crossing_above_threshold_triggers_exit_then_reentry_below():
    window = mod._WINDOW
    # Long negative-correlation warm-up, then a positive-correlation block
    # long enough to push the rolling correlation above the threshold, then
    # back to negative.
    regimes = [(window + 60, "neg"), (window + 20, "pos"), (window + 60, "neg")]
    n = sum(r[0] for r in regimes)
    index = _calendar(periods=n)
    spy_close, tlt_close = _corr_controlled_closes(index, window=window, regimes=regimes)
    frames = _frames(index, spy_close, tlt_close)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])

    # Recompute the rolling correlation directly (independent of the
    # module's own precompute) to assert against ground truth rather than
    # against the module's own internal state.
    spy_ret = pd.Series(np.log(spy_close)).diff()
    tlt_ret = pd.Series(np.log(tlt_close)).diff()
    corr = spy_ret.rolling(window).corr(tlt_ret).to_numpy()

    saw_exit = False
    saw_reentry = False
    for i in range(window, n):
        if np.isfinite(corr[i]) and corr[i] > mod._THRESHOLD:
            assert sigs[i] == SignalResult(entry=False, exit=True), i
            saw_exit = True
        elif np.isfinite(corr[i]) and corr[i] <= mod._THRESHOLD:
            assert sigs[i] == SignalResult(entry=True, exit=False), i
            if saw_exit:
                saw_reentry = True
    assert saw_exit, "the positive-correlation block never crossed the threshold"
    assert saw_reentry, "correlation never dropped back at/below threshold after the exit"


def test_boundary_correlation_equal_to_threshold_is_risk_on():
    # threshold crossing uses > for exit / <= for entry -- an exact match
    # at C is risk-on (hold), not risk-off. Constructed via a direct call
    # to the module's own precompute helper (a real market series landing
    # on the EXACT float threshold is not reliably constructible).
    window = mod._WINDOW
    index = _calendar(periods=window + 10)
    corr = pd.Series(np.nan, index=mod._naive_index(index))
    corr.iloc[window:] = mod._THRESHOLD
    entry, exit_ = mod.correlation_regime_flags(mod._naive_index(index), corr,
                                                mod._THRESHOLD)
    assert bool(entry.iloc[window])
    assert not bool(exit_.iloc[window])


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    window = mod._WINDOW
    regimes = [(window + 60, "neg"), (window + 30, "pos"), (window + 40, "neg")]
    n = sum(r[0] for r in regimes)
    index = _calendar(periods=n)
    spy_close, tlt_close = _corr_controlled_closes(index, window=window, regimes=regimes)
    frames = _frames(index, spy_close, tlt_close)
    k = window + 70
    base_fn = mod.make_signal_fn("SPY", index, frames)
    base = _signals(base_fn, frames["SPY"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.6
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.6
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("SPY", index, perturbed)
    after = _signals(pert_fn, perturbed["SPY"], upto=k)
    assert base == after
    # The perturbation is real: SOME later signal differs.
    assert (_signals(base_fn, frames["SPY"])
            != _signals(pert_fn, perturbed["SPY"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    window = mod._WINDOW
    regimes = [(window + 60, "neg"), (window + 30, "pos"), (window + 110, "neg")]
    n = sum(r[0] for r in regimes)
    index = _calendar(periods=n)
    spy_close, tlt_close = _corr_controlled_closes(index, window=window, regimes=regimes)
    frames = _frames(index, spy_close, tlt_close)
    bucket = index[:window + 130]
    fn_full = mod.make_signal_fn("SPY", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("SPY", bucket, truncated)
    bucket_frame = frames["SPY"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_dates_before_signal_history_stay_silent():
    # SPY exists long before TLT (TLT inception ~2002): SPY bars with no
    # TLT bar must be silent (entry=False, exit=False), matching this
    # catalog's "undefined is not flat by rule, it is simply no signal
    # today" convention.
    window = mod._WINDOW
    index = _calendar(periods=300)
    tlt_index = index[150:]
    spy_full, _ = _corr_controlled_closes(index, window=window, regimes=[(300, "neg")])
    _, tlt_tail = _corr_controlled_closes(tlt_index, window=window,
                                          regimes=[(len(tlt_index), "neg")])
    frames = {"SPY": _ohlc_from_close(spy_full, index),
              "TLT": _ohlc_from_close(tlt_tail, tlt_index)}
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])
    for i in range(150):
        assert sigs[i] == SignalResult(entry=False, exit=False), i


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_trades():
    window = mod._WINDOW
    regimes = [(window + 60, "neg"), (window + 20, "pos"), (window + 20, "neg"),
              (window + 20, "pos"), (window + 60, "neg")]
    n = sum(r[0] for r in regimes)
    index = _calendar(periods=n)
    spy_close, tlt_close = _corr_controlled_closes(index, window=window, regimes=regimes)
    frames = _frames(index, spy_close, tlt_close)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    trades, in_market = run_backtest(frames["SPY"], signal_fn, cost_bps=1.0)
    assert len(trades) >= 2
    assert in_market.any()
    assert (trades["reason"] == "signal_exit").any()


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    spy_close, tlt_close = _corr_controlled_closes(index, window=mod._WINDOW,
                                                    regimes=[(100, "neg")])
    frames = _frames(index, spy_close, tlt_close)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {"SPY": frames["SPY"]})


def test_signal_fn_is_constant_time_per_call():
    import time
    window = mod._WINDOW
    index = _calendar(start="2006-01-03", periods=3000)
    spy_close, tlt_close = _corr_controlled_closes(
        index, window=window, regimes=[(3000, "neg")])
    frames = _frames(index, spy_close, tlt_close)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    frame = frames["SPY"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
