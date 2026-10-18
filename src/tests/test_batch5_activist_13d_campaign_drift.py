"""Tests for src/mining/batch5/activist_13d_campaign_drift.py.

Synthetic data only (no network), deterministic, fast. The candidate's
PARAMETERS are compared against the REAL pre-registered catalog entry
(data/mining/catalog.json), the frozen source of truth for
PARAMETERS/tickers/source, so drift is caught here before the driver's own
drift check runs.

This family's rule (see the module docstring) is a pure buy-and-hold: the
signal is a CONSTANT (entry=True, exit=False on every bar the engine can
ask about), because the 13D-filing detection / basket construction /
campaign-hold timing are ACTV's own internal, issuer-side mechanism --
nothing in this module computes it. That constant-ness is itself the thing
under test: a broken implementation that accidentally introduces ANY
conditional exit, ANY conditional entry, or ANY dependence on price data
would be caught by every test below except the pure boundary/contract ones.
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
from mining.batch5 import activist_13d_campaign_drift as mod  # noqa: E402

CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2020-11-02", periods=50) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _actv_close(index: pd.DatetimeIndex, seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 20.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 3) -> dict:
    return {"ACTV": _ohlc_from_close(_actv_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "activist_13d_campaign_drift"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["ACTV"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "ACTV"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 1
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_frozen_catalog_entry():
    # catalog.json is the pre-registered, already-frozen source this module
    # must not reinterpret.
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == mod.FAMILY_ID]
    assert len(matches) == 1
    entry = matches[0]
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert mod.TRADED == entry["tickers"]
    assert mod.BENCHMARK == entry["tickers"][0]


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["ACTV"]
    assert spec["signal"] == []


# ------------------------------------------------------------------- rule

def test_rule_is_constant_always_long_never_exit_on_every_bar():
    index = _calendar(periods=60)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("ACTV", index, frames)
    sigs = _signals(signal_fn, frames["ACTV"])
    assert len(sigs) == 60
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


def test_non_finite_close_values_do_not_affect_the_signal():
    # The signal reads no price data at all (docstring choice 3): a NaN/inf
    # Close must not change, nor crash, the constant entry/exit decision.
    index = _calendar(periods=30)
    frame = _frames(index)["ACTV"]
    frame = frame.copy()
    frame.loc[frame.index[5], "Close"] = np.nan
    frame.loc[frame.index[10], "Close"] = np.inf
    frame.loc[frame.index[15], "Close"] = -np.inf
    frames = {"ACTV": frame}
    signal_fn = mod.make_signal_fn("ACTV", index, frames)
    sigs = _signals(signal_fn, frame)
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


def test_signal_identical_regardless_of_price_path():
    # A second, wildly different ACTV price path must produce the IDENTICAL
    # signal sequence -- the strongest possible statement that this signal
    # is price-independent (a bug that made it price-dependent would very
    # likely break this).
    index = _calendar(periods=80)
    fn_a = mod.make_signal_fn("ACTV", index, _frames(index, seed=1))
    fn_b = mod.make_signal_fn("ACTV", index, _frames(index, seed=999))
    frame_a = _frames(index, seed=1)["ACTV"]
    frame_b = _frames(index, seed=999)["ACTV"]
    assert _signals(fn_a, frame_a) == _signals(fn_b, frame_b)


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=80)
    frames = _frames(index)
    k = 40
    base_fn = mod.make_signal_fn("ACTV", index, frames)
    base = _signals(base_fn, frames["ACTV"], upto=k)

    perturbed = frames["ACTV"].copy()
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] *= 5.0
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Open")] *= 5.0
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("High")] *= 6.0
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Low")] *= 0.2
    pert_fn = mod.make_signal_fn("ACTV", index, {"ACTV": perturbed})
    after = _signals(pert_fn, perturbed, upto=k)
    assert base == after


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=90)
    frames = _frames(index)
    bucket = index[:60]
    fn_full = mod.make_signal_fn("ACTV", bucket, frames)
    truncated = {"ACTV": frames["ACTV"].loc[frames["ACTV"].index <= bucket[-1]]}
    fn_trunc = mod.make_signal_fn("ACTV", bucket, truncated)
    bucket_frame = frames["ACTV"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_frame_with_no_bars_at_or_before_bucket_end_raises():
    # All ACTV history lies AFTER the bucket -- nothing to truncate down to,
    # so the factory must fail closed rather than silently produce signals
    # with no underlying coverage.
    index = _calendar(periods=40)
    bucket = index[:10]
    late_index = index[20:]
    frames = {"ACTV": _ohlc_from_close(_actv_close(late_index), late_index)}
    with pytest.raises(ValueError):
        mod.make_signal_fn("ACTV", bucket, frames)


# ----------------------------------------------------------- end to end

def test_end_to_end_single_trade_opens_bar1_closes_end_of_data():
    # Hand-verified: with the engine's next-open entry / same-bar-close
    # exit convention, a constant entry=True/exit=False signal MUST produce
    # exactly one trade: the engine's first entry check happens at bar 1
    # (frame=df.iloc[:1], position=0), entry fires, so the position opens at
    # bar 1's OPEN; every subsequent bar's exit check reads exit=False, so
    # the position is only closed by the engine's own end_of_data path at
    # the FINAL bar's CLOSE.
    index = _calendar(periods=20)
    close = _actv_close(index, seed=5)
    frame = _ohlc_from_close(close, index)
    frames = {"ACTV": frame}
    signal_fn = mod.make_signal_fn("ACTV", index, frames)
    trades, in_market = run_backtest(frame, signal_fn, cost_bps=1.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_date"] == index[1]
    assert row["exit_date"] == index[-1]
    assert row["reason"] == "end_of_data"
    assert row["held"] == len(index) - 2

    expected_entry_px = float(frame["Open"].iloc[1])
    expected_exit_px = float(frame["Close"].iloc[-1])
    assert row["entry_px"] == pytest.approx(expected_entry_px)
    assert row["exit_px"] == pytest.approx(expected_exit_px)
    expected_gross = expected_exit_px / expected_entry_px - 1.0
    expected_net = expected_gross - 1.0 / 1e4
    assert row["gross_ret"] == pytest.approx(expected_gross)
    assert row["net_ret"] == pytest.approx(expected_net)

    # in_market is True from the entry bar through the exit bar inclusive,
    # False on bar 0 (no position could exist before the first entry check).
    assert not in_market.iloc[0]
    assert in_market.iloc[1:].all()


def test_end_to_end_tiny_hand_computed_series():
    # A fully hand-computed 4-bar series: Open/Close chosen so the expected
    # arithmetic can be verified by hand, not derived from the code under
    # test.
    index = _calendar(periods=4)
    frame = pd.DataFrame({
        "Open":  [10.0, 10.5, 11.0, 9.0],
        "High":  [10.2, 10.8, 11.3, 9.3],
        "Low":   [9.8, 10.2, 10.7, 8.7],
        "Close": [10.1, 10.9, 9.5, 9.2],
    }, index=index)
    frames = {"ACTV": frame}
    signal_fn = mod.make_signal_fn("ACTV", index, frames)
    trades, in_market = run_backtest(frame, signal_fn, cost_bps=0.0)

    # Entry at bar 1's open (10.5), position force-closed at bar 3's close
    # (9.2) via end_of_data. gross_ret = 9.2/10.5 - 1 = -0.1238095...
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(10.5)
    assert row["exit_px"] == pytest.approx(9.2)
    assert row["held"] == 2
    assert row["reason"] == "end_of_data"
    assert row["gross_ret"] == pytest.approx(9.2 / 10.5 - 1.0, abs=1e-12)
    assert row["net_ret"] == pytest.approx(9.2 / 10.5 - 1.0, abs=1e-12)


# --------------------------------------------------------------- edges

def test_single_bar_bucket_never_enters():
    # The engine's own convention: entry is never checked on bar 0 of a
    # bucket (i > 0 required), so a 1-bar bucket can never open a position.
    index = _calendar(periods=1)
    frame = _frames(index)["ACTV"]
    signal_fn = mod.make_signal_fn("ACTV", index, {"ACTV": frame})
    trades, in_market = run_backtest(frame, signal_fn, cost_bps=0.0)
    assert len(trades) == 0
    assert not in_market.any()
    # The signal function itself is still constant if ever queried directly.
    assert signal_fn(frame.iloc[:1], 0) == SignalResult(entry=True, exit=False)


def test_empty_bucket_index_raises():
    index = _calendar(periods=10)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("ACTV", index[:0], frames)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=20)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("ACTV", index, {})
    with pytest.raises(ValueError):
        mod.make_signal_fn("ACTV", index, {"ACTV": pd.DataFrame()})


def test_frame_missing_close_column_raises():
    index = _calendar(periods=10)
    frame = _frames(index)["ACTV"].drop(columns=["Close"])
    with pytest.raises(ValueError):
        mod.make_signal_fn("ACTV", index, {"ACTV": frame})


def test_signal_fn_out_of_bounds_position_raises():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("ACTV", index, frames)
    frame = frames["ACTV"]
    with pytest.raises(IndexError):
        signal_fn(frame.iloc[:1], -1)
    with pytest.raises(IndexError):
        signal_fn(frame.iloc[:1], 20)


def test_signal_fn_bucket_date_mismatch_raises():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("ACTV", index, frames)
    other_index = _calendar(start="2022-01-03", periods=20)
    other_frame = _ohlc_from_close(_actv_close(other_index), other_index)
    with pytest.raises(ValueError):
        signal_fn(other_frame.iloc[:5], 4)


def test_bucket_index_must_be_sorted_and_unique():
    index = _calendar(periods=10)
    frames = _frames(index)
    scrambled = index[::-1]
    with pytest.raises(ValueError):
        mod.make_signal_fn("ACTV", scrambled, frames)
    dup = index.insert(0, index[0])
    with pytest.raises(ValueError):
        mod.make_signal_fn("ACTV", dup, frames)


def test_signal_fn_is_fast_over_a_large_bucket():
    import time
    index = _calendar(start="2020-11-02", periods=5000)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("ACTV", index, frames)
    frame = frames["ACTV"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
