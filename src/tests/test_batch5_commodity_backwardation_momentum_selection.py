"""Tests for src/mining/batch5/commodity_backwardation_momentum_selection.py.

Mirrors test_batch5_value_factor_book_to_market.py's convention (the same
always-long-single-proxy shape, "the issuer already implements periodic
reformation" reasoning) for the synthetic-data contract/rule/look-ahead/
holdout-blind sections, plus a contract-isolation test copying the module
into a tmp_path dir (matching the sibling batch5 tests' convention of not
asserting on the real shared src/mining/batch5/ dir directly, since other
families are present there too).

PARAMETERS is checked against data/mining/catalog.json DIRECTLY (this
family IS already registered there, under family_id
"commodity_backwardation_momentum_selection") -- read fresh in the test
rather than transcribed as a local frozen literal.

The real-data end-to-end checks (live market-history fetches) are not part of
this public file; every test here is offline and uses synthetic data.
"""
import importlib.util
import json
import os
import shutil
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
from mining.batch5 import commodity_backwardation_momentum_selection as mod  # noqa: E402

_CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
_MODULE_PATH = os.path.join(
    _SRC, "mining", "batch5", "commodity_backwardation_momentum_selection.py")


def _catalog_entry() -> dict:
    with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    for entry in catalog:
        if entry.get("family_id") == "commodity_backwardation_momentum_selection":
            return entry
    raise AssertionError(
        "commodity_backwardation_momentum_selection not found in "
        "data/mining/catalog.json")


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2013-05-06", periods=600) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _usci_close(index: pd.DatetimeIndex, seed: int = 41) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 40.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 41) -> dict:
    return {"USCI": _ohlc_from_close(_usci_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "commodity_backwardation_momentum_selection"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["USCI"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    # ABI-1 always benchmarks against SPY, never a family's own catalog
    # "benchmark" concept -- see scripts/run_abi1_mining.py's
    # MARKET_BENCHMARK comment and value_factor_book_to_market.py's own
    # identical disclosure for this same shape.
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_catalog_json():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert set(mod.TRADED) | set(mod.SIGNAL) <= set(entry["tickers"])
    assert mod.EXPECTED_MIN_TRADES == 1


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["USCI"]
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    assert spec["parameters"] == mod.PARAMETERS


def test_module_satisfies_discover_modules_in_isolation(tmp_path):
    # Copy ONLY this module into an isolated batch5-shaped dir and run the
    # real driver's discover_modules over it -- proves this module is
    # self-contained and does not accidentally depend on a sibling being
    # present in the real shared src/mining/batch5/ dir.
    import run_batch5_mining as driver

    isolated_dir = tmp_path / "batch5_isolated"
    isolated_dir.mkdir()
    (isolated_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy(_MODULE_PATH,
                isolated_dir / "commodity_backwardation_momentum_selection.py")

    modules, problems = driver.discover_modules(str(isolated_dir), repo_root=_ROOT)
    assert problems == []
    assert "commodity_backwardation_momentum_selection" in modules
    entry = modules["commodity_backwardation_momentum_selection"]
    assert entry["spec"]["family_id"] == "commodity_backwardation_momentum_selection"
    assert entry["spec"]["shape"] == "single_long"
    assert entry["spec"]["traded"] == ["USCI"]


# ------------------------------------------------------------------- rule

def test_entry_true_exit_false_on_every_reachable_bar():
    # Hand-verified: this rule has no per-bar decision, so EVERY call must
    # return exactly SignalResult(entry=True, exit=False), regardless of
    # position in the bucket or the bar's own price data.
    index = _calendar(periods=250)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("USCI", index, frames)
    sigs = _signals(signal_fn, frames["USCI"])
    assert len(sigs) == 250
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


def test_single_bar_bucket_still_signals_entry():
    index = _calendar(periods=1)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("USCI", index, frames)
    assert signal_fn(frames["USCI"], 0) == SignalResult(entry=True, exit=False)


def test_empty_index_is_rejected():
    frames = _frames(_calendar(periods=5))
    empty_index = pd.DatetimeIndex([])
    with pytest.raises(ValueError):
        mod.make_signal_fn("USCI", empty_index, frames)


def test_non_finite_prices_do_not_change_the_signal():
    # The rule never reads a price at all, so injecting NaN/inf into USCI's
    # own Close column must leave the signal completely unaffected.
    index = _calendar(periods=120)
    frames = _frames(index)
    dirty = frames["USCI"].copy()
    dirty.iloc[10, dirty.columns.get_loc("Close")] = np.nan
    dirty.iloc[40, dirty.columns.get_loc("Close")] = np.inf
    dirty.iloc[70, dirty.columns.get_loc("Open")] = -np.inf
    clean_fn = mod.make_signal_fn("USCI", index, frames)
    dirty_fn = mod.make_signal_fn("USCI", index, {"USCI": dirty})
    assert _signals(clean_fn, frames["USCI"]) == _signals(dirty_fn, dirty)


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=300)
    frames = _frames(index)
    k = 150
    base_fn = mod.make_signal_fn("USCI", index, frames)
    base = _signals(base_fn, frames["USCI"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.7
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.5
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("USCI", index, perturbed)
    after = _signals(pert_fn, perturbed["USCI"], upto=k)
    assert base == after
    assert (_signals(base_fn, frames["USCI"])
            == _signals(pert_fn, perturbed["USCI"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=400)
    frames = _frames(index)
    bucket = index[:300]
    fn_full = mod.make_signal_fn("USCI", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("USCI", bucket, truncated)
    bucket_frame = frames["USCI"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_holdout_bucket_starting_mid_history_still_enters_immediately():
    index = _calendar(periods=400)
    frames = _frames(index)
    bucket = index[250:]
    signal_fn = mod.make_signal_fn("USCI", bucket, frames)
    bucket_frame = frames["USCI"].loc[bucket]
    sigs = _signals(signal_fn, bucket_frame)
    assert sigs[0] == SignalResult(entry=True, exit=False)
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_exactly_one_trade():
    # Hand-verified arithmetic fixture: 10 bars, closes chosen by hand.
    index = _calendar(periods=10)
    closes = np.array([40.0, 40.4, 40.8, 40.3, 41.2, 41.6, 41.0, 42.0, 42.4, 43.0])
    frame = _ohlc_from_close(closes, index)
    frames = {"USCI": frame}
    signal_fn = mod.make_signal_fn("USCI", index, frames)
    trades, in_market = run_backtest(frame, signal_fn, cost_bps=1.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    entry_open = frame["Open"].iloc[1]
    exit_close = frame["Close"].iloc[9]
    assert row["entry_date"] == index[1]
    assert row["exit_date"] == index[9]
    assert row["entry_px"] == pytest.approx(entry_open)
    assert row["exit_px"] == pytest.approx(exit_close)
    assert row["held"] == 8
    assert row["reason"] == "end_of_data"
    expected_gross = exit_close / entry_open - 1.0
    assert row["gross_ret"] == pytest.approx(expected_gross)
    assert row["net_ret"] == pytest.approx(expected_gross - 1.0 / 1e4)
    assert not in_market.iloc[0]
    assert in_market.iloc[1:].all()


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=50)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("USCI", index, {})


def test_signal_fn_rejects_out_of_range_position():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("USCI", index, frames)
    with pytest.raises(IndexError):
        signal_fn(frames["USCI"].iloc[:1], -1)
    with pytest.raises(IndexError):
        signal_fn(frames["USCI"], 20)


def test_signal_fn_rejects_mismatched_bucket_bar():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("USCI", index, frames)
    wrong_frame = frames["USCI"].iloc[:5].copy()
    with pytest.raises(ValueError):
        signal_fn(wrong_frame, 10)


def test_signal_fn_is_constant_time_per_call():
    index = _calendar(start="2013-05-06", periods=5000)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("USCI", index, frames)
    frame = frames["USCI"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
