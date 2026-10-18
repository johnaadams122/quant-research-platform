"""Tests for src/mining/batch5/managed_futures_trend_replication.py.

Mirrors test_batch5_free_cash_flow_yield_cows.py's / test_batch5_wide_moat_
quality_investing.py's convention (same always-long-since-inception shape)
for the synthetic-data contract/rule/look-ahead/holdout-blind sections, plus
test_batch5_hy_credit_bond_absolute_momentum.py's tmp_path-isolated driver-
discovery convention (so the result does not depend on whatever else sits
in the real batch5 dir).
Deviates from the two single_long siblings in the same two places they
themselves deviate from test_batch5_faith_based_sin_stock_screen.py:

1. PARAMETERS is checked against data/mining/catalog.json DIRECTLY (this
   family IS registered there, under family_id
   "managed_futures_trend_replication"), read fresh in the test rather than
   transcribed as a local frozen literal.
2. Real-data end-to-end checks (network fetches of DBMF's history) are not
   part of this file; everything here runs offline on synthetic fixtures.
"""
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
from mining.batch5 import managed_futures_trend_replication as mod  # noqa: E402

_CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
_MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                            "managed_futures_trend_replication.py")


def _catalog_entry() -> dict:
    with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    matches = [e for e in catalog
               if e.get("family_id") == "managed_futures_trend_replication"]
    assert len(matches) == 1, (
        "expected exactly one catalog.json entry for "
        f"managed_futures_trend_replication, found {len(matches)}")
    return matches[0]


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2019-05-08", periods=600) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _dbmf_close(index: pd.DatetimeIndex, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 16.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 7) -> dict:
    return {"DBMF": _ohlc_from_close(_dbmf_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "managed_futures_trend_replication"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["DBMF"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    # ABI-1 always benchmarks against SPY, never a family's own catalog
    # "benchmark" concept -- see scripts/run_abi1_mining.py::MARKET_BENCHMARK
    # and every sibling batch5 module's own identical disclosure.
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES == 1
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_catalog_json():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    # TRADED + SIGNAL must resolve to the catalog's own tickers list.
    assert set(mod.TRADED) | set(mod.SIGNAL) == set(entry["tickers"])
    assert entry["tickers"] == ["DBMF"]


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["DBMF"]
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    assert spec["parameters"] == mod.PARAMETERS
    assert spec["expected_min_trades"] == 1


def test_module_passes_the_driver_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module, so the result does not
    depend on whatever else sits in the real batch5 dir -- matches
    test_batch5_hy_credit_bond_absolute_momentum.py's own isolation
    convention."""
    import run_batch5_mining as driver

    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(_MODULE_PATH, modules_dir / "managed_futures_trend_replication.py")
    found, problems = driver.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == ["managed_futures_trend_replication"]
    spec = found["managed_futures_trend_replication"]["spec"]
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["DBMF"]
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    catalog_params = _catalog_entry()["parameters"]
    assert driver.parameter_drift(spec["parameters"], catalog_params) is None


# ------------------------------------------------------------------- rule

def test_entry_true_exit_false_on_every_reachable_bar():
    # Hand-verified: this rule has no per-bar decision, so EVERY call must
    # return exactly SignalResult(entry=True, exit=False), regardless of
    # position in the bucket or the bar's own price data.
    index = _calendar(periods=250)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("DBMF", index, frames)
    sigs = _signals(signal_fn, frames["DBMF"])
    assert len(sigs) == 250
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


def test_single_bar_bucket_still_signals_entry():
    index = _calendar(periods=1)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("DBMF", index, frames)
    assert signal_fn(frames["DBMF"], 0) == SignalResult(entry=True, exit=False)


def test_empty_index_is_rejected():
    frames = _frames(_calendar(periods=5))
    empty_index = pd.DatetimeIndex([])
    with pytest.raises(ValueError):
        mod.make_signal_fn("DBMF", empty_index, frames)


def test_non_finite_prices_do_not_change_the_signal():
    # The rule never reads a price at all, so injecting NaN/inf into DBMF's
    # own Close column must leave the signal completely unaffected.
    index = _calendar(periods=120)
    frames = _frames(index)
    dirty = frames["DBMF"].copy()
    dirty.iloc[10, dirty.columns.get_loc("Close")] = np.nan
    dirty.iloc[40, dirty.columns.get_loc("Close")] = np.inf
    dirty.iloc[70, dirty.columns.get_loc("Open")] = -np.inf
    clean_fn = mod.make_signal_fn("DBMF", index, frames)
    dirty_fn = mod.make_signal_fn("DBMF", index, {"DBMF": dirty})
    assert _signals(clean_fn, frames["DBMF"]) == _signals(dirty_fn, dirty)


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=300)
    frames = _frames(index)
    k = 150
    base_fn = mod.make_signal_fn("DBMF", index, frames)
    base = _signals(base_fn, frames["DBMF"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.7
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.5
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("DBMF", index, perturbed)
    after = _signals(pert_fn, perturbed["DBMF"], upto=k)
    assert base == after
    assert (_signals(base_fn, frames["DBMF"])
            == _signals(pert_fn, perturbed["DBMF"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=400)
    frames = _frames(index)
    bucket = index[:300]
    fn_full = mod.make_signal_fn("DBMF", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("DBMF", bucket, truncated)
    bucket_frame = frames["DBMF"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_holdout_bucket_starting_mid_history_still_enters_immediately():
    index = _calendar(periods=400)
    frames = _frames(index)
    bucket = index[250:]
    signal_fn = mod.make_signal_fn("DBMF", bucket, frames)
    bucket_frame = frames["DBMF"].loc[bucket]
    sigs = _signals(signal_fn, bucket_frame)
    assert sigs[0] == SignalResult(entry=True, exit=False)
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


# ----------------------------------------------------------- end to end
# (synthetic)

def test_end_to_end_run_backtest_produces_exactly_one_trade():
    # Hand-verified arithmetic fixture: 10 bars, closes chosen by hand.
    index = _calendar(periods=10)
    closes = np.array([16.0, 16.2, 16.4, 16.1, 16.6, 16.8, 16.5, 17.0, 17.2, 17.5])
    frame = _ohlc_from_close(closes, index)
    frames = {"DBMF": frame}
    signal_fn = mod.make_signal_fn("DBMF", index, frames)
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
        mod.make_signal_fn("DBMF", index, {})


def test_signal_fn_rejects_out_of_range_position():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("DBMF", index, frames)
    with pytest.raises(IndexError):
        signal_fn(frames["DBMF"].iloc[:1], -1)
    with pytest.raises(IndexError):
        signal_fn(frames["DBMF"], 20)


def test_signal_fn_rejects_mismatched_bucket_bar():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("DBMF", index, frames)
    wrong_frame = frames["DBMF"].iloc[:5].copy()
    with pytest.raises(ValueError):
        signal_fn(wrong_frame, 10)


def test_signal_fn_is_constant_time_per_call():
    index = _calendar(start="2019-05-08", periods=5000)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("DBMF", index, frames)
    frame = frames["DBMF"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
