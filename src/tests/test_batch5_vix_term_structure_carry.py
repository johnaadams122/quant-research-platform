"""Tests for src/mining/batch5/vix_term_structure_carry.py.

Synthetic data only (no network), deterministic, fast. PARAMETERS are
compared against a hardcoded transcription of the pre-registered catalog
entry, so this test file does not depend on any curation file's current
contents.
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
from mining.batch5 import vix_term_structure_carry as mod  # noqa: E402

# Transcribed verbatim from this family's pre-registered catalog record
# (its frozen candidate JSON), NOT read from a curation file.
_FROZEN_PARAMETERS = {
    "signal": "vix_spot_over_vix3m_ratio",
    "contango_threshold": 1,
    "flat_position": "cash",
    "rebalance": "daily",
}
_FROZEN_TICKERS = ["SVXY", "^VIX", "^VIX3M"]


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    with np.errstate(invalid="ignore"):
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = np.where(np.isfinite(prev), prev, close)
        open_ = np.where(np.isfinite(open_), open_, 1.0)
        high = np.where(np.isfinite(close), np.maximum(open_, close) * 1.002, np.nan)
        low = np.where(np.isfinite(close), np.minimum(open_, close) * 0.998, np.nan)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2015-01-02", periods=10) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _svxy_close(index: pd.DatetimeIndex, seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 80.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(index))))


def _signals(signal_fn, frame: pd.DataFrame, upto=None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


def _hand_fixture():
    """10-bar fixture with every ratio regime hand-computed below; see the
    module docstring test for the day-by-day derivation. Day indices:
      0: vix=15  vix3m=20  -> ratio 0.75  contango       -> entry
      1: vix=18  vix3m=20  -> ratio 0.90  contango       -> entry
      2: vix=25  vix3m=20  -> ratio 1.25  backwardation  -> exit
      3: vix=20  vix3m=20  -> ratio 1.00  boundary (>=)  -> exit
      4: vix=19  vix3m=20  -> ratio 0.95  contango       -> entry
      5: vix=NaN vix3m=20  -> non-finite vix             -> silent
      6: vix=15  vix3m=0   -> non-positive denominator   -> silent
      7: vix=15  vix3m=<row absent from the ^VIX3M frame>-> silent
      8: vix=10  vix3m=20  -> ratio 0.50  contango       -> entry
      9: vix=30  vix3m=20  -> ratio 1.50  backwardation  -> exit (final bar)
    """
    index = _calendar(periods=10)
    vix_vals = [15, 18, 25, 20, 19, np.nan, 15, 15, 10, 30]
    vix_frame = _ohlc_from_close(vix_vals, index)

    vix3m_index = index.delete(7)  # day 7's ^VIX3M row is entirely absent
    vix3m_vals = [20, 20, 20, 20, 20, 20, 0, 20, 20]
    vix3m_frame = _ohlc_from_close(vix3m_vals, vix3m_index)

    svxy_frame = _ohlc_from_close(_svxy_close(index), index)
    frames = {"SVXY": svxy_frame, "^VIX": vix_frame, "^VIX3M": vix3m_frame}
    expected_entry = [True, True, False, False, True, False, False, False, True, False]
    expected_exit = [False, False, True, True, False, False, False, False, False, True]
    return index, frames, expected_entry, expected_exit


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "vix_term_structure_carry"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SVXY"]
    assert mod.SIGNAL == ["^VIX", "^VIX3M"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SVXY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 0
    assert callable(mod.make_signal_fn)


def test_parameters_match_frozen_catalog_entry():
    import json
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(_FROZEN_PARAMETERS, sort_keys=True))
    assert mod.TRADED + mod.SIGNAL == _FROZEN_TICKERS


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["SVXY"]
    assert spec["signal"] == ["^VIX", "^VIX3M"]


# ------------------------------------------------------------------- rule

def test_rule_entry_on_contango_exit_on_backwardation_hand_verified():
    index, frames, expected_entry, expected_exit = _hand_fixture()
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    sigs = _signals(signal_fn, frames["SVXY"])
    got_entry = [s.entry for s in sigs]
    got_exit = [s.exit for s in sigs]
    assert got_entry == expected_entry
    assert got_exit == expected_exit


def test_ratio_exactly_at_threshold_is_backwardation_not_contango():
    # Day 3 of the hand fixture is ratio == 1.0 exactly (20/20); the rule
    # text says "1.0 or above" is backwardation, so this must be exit=True,
    # entry=False -- never the reverse.
    index, frames, expected_entry, expected_exit = _hand_fixture()
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    sig = signal_fn(frames["SVXY"].iloc[:4], 3)
    assert sig == SignalResult(entry=False, exit=True)


def test_missing_or_invalid_signal_bar_is_silent():
    index, frames, expected_entry, expected_exit = _hand_fixture()
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    sigs = _signals(signal_fn, frames["SVXY"])
    # Day 5 (non-finite ^VIX), day 6 (non-positive ^VIX3M) and day 7 (^VIX3M
    # row entirely absent) must all be silent on both flags.
    for day in (5, 6, 7):
        assert sigs[day] == SignalResult(entry=False, exit=False), day


def test_svxy_prices_never_enter_the_signal():
    index, frames, expected_entry, expected_exit = _hand_fixture()
    frames_a = dict(frames)
    frames_a["SVXY"] = _ohlc_from_close(_svxy_close(index, seed=1), index)
    frames_b = dict(frames)
    frames_b["SVXY"] = _ohlc_from_close(_svxy_close(index, seed=99), index)
    fn_a = mod.make_signal_fn("SVXY", index, frames_a)
    fn_b = mod.make_signal_fn("SVXY", index, frames_b)
    assert (_signals(fn_a, frames_a["SVXY"])
            == _signals(fn_b, frames_b["SVXY"]))


def test_final_bar_of_bucket_is_a_real_exit_decision():
    # Unlike the batch's monthly-rebalance candidates, this rule re-decides
    # every day, so the bucket's own final bar carries a real exit reading.
    index, frames, expected_entry, expected_exit = _hand_fixture()
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    last = signal_fn(frames["SVXY"], len(index) - 1)
    assert last == SignalResult(entry=False, exit=True)  # ratio 1.5 at day 9


def test_single_bar_bucket_does_not_crash():
    index = _calendar(periods=1)
    frames = {
        "SVXY": _ohlc_from_close(_svxy_close(index), index),
        "^VIX": _ohlc_from_close([15.0], index),
        "^VIX3M": _ohlc_from_close([20.0], index),
    }
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    sig = signal_fn(frames["SVXY"], 0)
    assert sig == SignalResult(entry=True, exit=False)  # ratio 0.75


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=500)
    t = np.arange(len(index))
    vix3m_vals = np.full(len(index), 20.0)
    ratio = 1.0 + 0.3 * np.sin(t / 9.0)  # oscillates through 1.0 repeatedly
    vix_vals = ratio * vix3m_vals
    vix_frame = _ohlc_from_close(vix_vals, index)
    vix3m_frame = _ohlc_from_close(vix3m_vals, index)
    svxy_frame = _ohlc_from_close(_svxy_close(index), index)
    frames = {"SVXY": svxy_frame, "^VIX": vix_frame, "^VIX3M": vix3m_frame}

    k = 310
    base_fn = mod.make_signal_fn("SVXY", index, frames)
    base = _signals(base_fn, svxy_frame, upto=k)

    # Perturb ONLY ^VIX post-cutoff: scaling both signal legs by the same
    # factor would leave their ratio (and thus every signal) unchanged, so
    # that would not actually probe look-ahead at all.
    perturbed = {"SVXY": svxy_frame.copy(), "^VIX3M": vix3m_frame.copy()}
    pert_vix = vix_frame.copy()
    pert_vix.iloc[k + 1:, pert_vix.columns.get_loc("Close")] *= 1.7
    perturbed["^VIX"] = pert_vix
    pert_fn = mod.make_signal_fn("SVXY", index, perturbed)
    after = _signals(pert_fn, perturbed["SVXY"], upto=k)
    assert base == after
    # The perturbation is real: some later signal differs.
    assert (_signals(base_fn, svxy_frame)
            != _signals(pert_fn, perturbed["SVXY"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=400)
    t = np.arange(len(index))
    vix3m_vals = np.full(len(index), 20.0)
    ratio = 1.0 + 0.25 * np.sin(t / 7.0)
    vix_vals = ratio * vix3m_vals
    frames = {
        "SVXY": _ohlc_from_close(_svxy_close(index), index),
        "^VIX": _ohlc_from_close(vix_vals, index),
        "^VIX3M": _ohlc_from_close(vix3m_vals, index),
    }
    bucket = index[:300]
    fn_full = mod.make_signal_fn("SVXY", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("SVXY", bucket, truncated)
    bucket_frame = frames["SVXY"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


# ------------------------------------------------------------ contract guards

def test_factory_rejects_wrong_ticker_or_missing_frames():
    index, frames, _, _ = _hand_fixture()
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SVXY", index, {"SVXY": frames["SVXY"]})
    with pytest.raises(ValueError):
        mod.make_signal_fn("SVXY", index,
                           {"SVXY": frames["SVXY"], "^VIX": frames["^VIX"]})


def test_factory_rejects_empty_index():
    frames = {
        "SVXY": _ohlc_from_close([80.0], _calendar(periods=1)),
        "^VIX": _ohlc_from_close([15.0], _calendar(periods=1)),
        "^VIX3M": _ohlc_from_close([20.0], _calendar(periods=1)),
    }
    with pytest.raises(ValueError):
        mod.make_signal_fn("SVXY", pd.DatetimeIndex([]), frames)


def test_factory_rejects_unsorted_or_duplicate_index():
    index, frames, _, _ = _hand_fixture()
    reversed_index = index[::-1]
    with pytest.raises(ValueError):
        mod.make_signal_fn("SVXY", reversed_index, frames)
    dup_index = index.append(pd.DatetimeIndex([index[0]]))
    with pytest.raises(ValueError):
        mod.make_signal_fn("SVXY", dup_index, frames)


def test_signal_fn_rejects_out_of_range_position():
    index, frames, _, _ = _hand_fixture()
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    with pytest.raises(IndexError):
        signal_fn(frames["SVXY"], len(index))
    with pytest.raises(IndexError):
        signal_fn(frames["SVXY"], -1)


def test_signal_fn_rejects_mismatched_bar_date():
    index, frames, _, _ = _hand_fixture()
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    with pytest.raises(ValueError):
        signal_fn(frames["SVXY"].iloc[:1], 3)  # frame's last bar is day 0, not day 3


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_trades_on_threshold_crossings():
    index = _calendar(periods=250)
    t = np.arange(len(index))
    vix3m_vals = np.full(len(index), 20.0)
    ratio = 1.0 + 0.35 * np.sin(t / 11.0)
    vix_vals = ratio * vix3m_vals
    svxy_frame = _ohlc_from_close(_svxy_close(index), index)
    frames = {"SVXY": svxy_frame, "^VIX": _ohlc_from_close(vix_vals, index),
              "^VIX3M": _ohlc_from_close(vix3m_vals, index)}
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    trades, in_market = run_backtest(svxy_frame, signal_fn, cost_bps=2.0)
    # Hand-verified via mod.vix_ratio_flags on this exact series: ratio-1.0
    # changes sign 7 times over 250 bars, alternating backwardation/contango
    # starting in backwardation, giving exactly 4 contiguous contango
    # (entry) regions -- one trade per region (open at the region's start,
    # close -- by signal or by end_of_data for the final region -- at its
    # end).
    assert len(trades) == 4
    assert in_market.any()
    assert (trades["held"] >= 1).all()
    for _, row in trades.iterrows():
        assert row["entry_date"] in index
        assert row["exit_date"] in index
        assert row["entry_date"] < row["exit_date"] or row["held"] == 0


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2007-12-04", periods=5000)
    rng = np.random.RandomState(5)
    vix3m_vals = 20.0 + np.cumsum(rng.normal(0.0, 0.05, len(index)))
    vix_vals = vix3m_vals * (1.0 + rng.normal(0.0, 0.05, len(index)))
    frames = {
        "SVXY": _ohlc_from_close(_svxy_close(index), index),
        "^VIX": _ohlc_from_close(vix_vals, index),
        "^VIX3M": _ohlc_from_close(vix3m_vals, index),
    }
    signal_fn = mod.make_signal_fn("SVXY", index, frames)
    frame = frames["SVXY"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
