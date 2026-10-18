"""Tests for src/mining/batch5/ipo_long_run_underperformance_fade.py.

Mirrors test_batch5_value_factor_book_to_market.py's convention (same
always-in-position single-instrument shape, just direction="short"
instead of the default "long") for the synthetic-data contract/rule/
look-ahead/holdout-blind sections, plus test_batch5_retail_inventory_
sales_breadth_short.py's convention of NOT calling
run_batch5_mining.validate_module_contract/discover_modules as a
generic-dispatch-readiness proof for every test -- this family, like
that one, is always run through its own bespoke driver
(scripts/run_ipo_fade_abi1.py), never scripts/run_abi1_mining.py's
generic SINGLE_SHAPES path (see the module's own "SHAPE and dispatch"
docstring section for why). One contract test below DOES still call
validate_module_contract -- unlike retail's panel_breadth (which needs a
non-empty SNAPSHOTS dict panel_breadth requires), this module's fields
are well-formed against the generic schema even though nothing calls it
that way in production.

PARAMETERS is checked against data/mining/catalog.json DIRECTLY (this
family IS already registered there, under family_id
"ipo_long_run_underperformance_fade") -- read fresh in the test rather
than transcribed as a local frozen literal.

The real-data end-to-end checks (live market-history fetches) are not part of
this public file; every test here is offline and uses synthetic data.
"""
import importlib.util
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
from mining.batch5 import ipo_long_run_underperformance_fade as mod  # noqa: E402

_CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")


def _catalog_entry() -> dict:
    with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    for entry in catalog:
        if entry.get("family_id") == "ipo_long_run_underperformance_fade":
            return entry
    raise AssertionError(
        "ipo_long_run_underperformance_fade not found in data/mining/catalog.json")


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2013-10-16", periods=600) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _ipo_close(index: pd.DatetimeIndex, seed: int = 29) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 20.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index))))


def _frames(index: pd.DatetimeIndex, seed: int = 29) -> dict:
    return {"IPO": _ohlc_from_close(_ipo_close(index, seed), index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "ipo_long_run_underperformance_fade"
    assert mod.SHAPE == "single_inverse_product"
    assert mod.TRADED == ["IPO"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    # ABI-1 always benchmarks against SPY, never a family's own catalog
    # "benchmark" concept -- see scripts/run_abi1_mining.py's
    # MARKET_BENCHMARK comment and every sibling single-instrument
    # module's identical disclosure.
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_catalog_json():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert set(mod.TRADED) | set(mod.SIGNAL) <= set(entry["tickers"])
    assert mod.PARAMETERS["position"] == "short"
    assert mod.EXPECTED_MIN_TRADES == 1


def test_driver_discovery_validates_this_module_contract_shape():
    # This family is NOT in accepted.json and is never run through the
    # generic driver in production (see module docstring "SHAPE and
    # dispatch") -- this test only proves the contract fields are
    # well-formed against the generic schema, not that generic dispatch
    # is safe to use for this family (it is not: it would default to
    # direction="long", which is wrong for this family).
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_inverse_product"
    assert spec["traded"] == ["IPO"]
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    assert spec["parameters"] == mod.PARAMETERS


# ------------------------------------------------------------------- rule

def test_entry_true_exit_false_on_every_reachable_bar():
    # Hand-verified: this rule has no per-bar decision (see module
    # docstring's DISCLOSED SIMPLIFICATION), so EVERY call must return
    # exactly SignalResult(entry=True, exit=False) -- direction is NOT
    # encoded here (that's the bespoke driver's job, matching
    # retail_inventory_sales_breadth_short.py's own convention).
    index = _calendar(periods=250)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    sigs = _signals(signal_fn, frames["IPO"])
    assert len(sigs) == 250
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


def test_single_bar_bucket_still_signals_entry():
    index = _calendar(periods=1)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    assert signal_fn(frames["IPO"], 0) == SignalResult(entry=True, exit=False)


def test_empty_index_is_rejected():
    frames = _frames(_calendar(periods=5))
    empty_index = pd.DatetimeIndex([])
    with pytest.raises(ValueError):
        mod.make_signal_fn("IPO", empty_index, frames, driver_acknowledges_short_direction=True)


def test_non_finite_prices_do_not_change_the_signal():
    # The rule never reads a price at all, so injecting NaN/inf into IPO's
    # own Close column must leave the signal completely unaffected.
    index = _calendar(periods=120)
    frames = _frames(index)
    dirty = frames["IPO"].copy()
    dirty.iloc[10, dirty.columns.get_loc("Close")] = np.nan
    dirty.iloc[40, dirty.columns.get_loc("Close")] = np.inf
    dirty.iloc[70, dirty.columns.get_loc("Open")] = -np.inf
    clean_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    dirty_fn = mod.make_signal_fn("IPO", index, {"IPO": dirty}, driver_acknowledges_short_direction=True)
    assert _signals(clean_fn, frames["IPO"]) == _signals(dirty_fn, dirty)


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=300)
    frames = _frames(index)
    k = 150
    base_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    base = _signals(base_fn, frames["IPO"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.5
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.7
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.5
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("IPO", index, perturbed, driver_acknowledges_short_direction=True)
    after = _signals(pert_fn, perturbed["IPO"], upto=k)
    assert base == after
    assert (_signals(base_fn, frames["IPO"])
            == _signals(pert_fn, perturbed["IPO"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=400)
    frames = _frames(index)
    bucket = index[:300]
    fn_full = mod.make_signal_fn("IPO", bucket, frames, driver_acknowledges_short_direction=True)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("IPO", bucket, truncated, driver_acknowledges_short_direction=True)
    bucket_frame = frames["IPO"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_holdout_bucket_starting_mid_history_still_enters_immediately():
    index = _calendar(periods=400)
    frames = _frames(index)
    bucket = index[250:]
    signal_fn = mod.make_signal_fn("IPO", bucket, frames, driver_acknowledges_short_direction=True)
    bucket_frame = frames["IPO"].loc[bucket]
    sigs = _signals(signal_fn, bucket_frame)
    assert sigs[0] == SignalResult(entry=True, exit=False)
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_short_produces_exactly_one_trade():
    # Hand-verified arithmetic fixture: 10 bars, closes chosen by hand,
    # RISING -- so a SHORT position must show a NEGATIVE gross_ret (the
    # opposite sign of the long fixture in the value_factor sibling test).
    index = _calendar(periods=10)
    closes = np.array([20.0, 20.2, 20.4, 20.17, 20.6, 20.8, 20.5, 21.0, 21.2, 21.5])
    frame = _ohlc_from_close(closes, index)
    frames = {"IPO": frame}
    signal_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    trades, in_market = run_backtest(frame, signal_fn, cost_bps=1.0, direction="short")

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
    long_ret = exit_close / entry_open - 1.0
    expected_gross = -long_ret  # short: negated long return, per
                                # backtest_engine.run_backtest's own docstring
    assert row["gross_ret"] == pytest.approx(expected_gross)
    assert row["gross_ret"] < 0.0  # the fixture prices rise, so the short loses
    assert row["net_ret"] == pytest.approx(expected_gross - 1.0 / 1e4)
    assert not in_market.iloc[0]
    assert in_market.iloc[1:].all()


def test_generic_dispatch_call_shape_raises_instead_of_silently_going_long():
    """DISPATCH-SAFETY REGRESSION TEST: scripts/run_abi1_mining.py's generic
    SINGLE_SHAPES path calls make_signal_fn with EXACTLY 3 positional args
    (ticker, index, frames), no keyword arguments -- module.make_signal_fn(
    symbol, full_frame.index, module_frames). That call shape has no
    direction="short" passthrough (abi1_driver.build_single_asset_curve
    always defaults to "long"), so if this family were ever added to
    accepted.json and reached the generic dispatch, a module that accepted
    this call shape would silently produce a LONG IPO position -- the
    wrong direction, with no error and a plausible-looking result. This
    test proves the module raises a clear, actionable error on exactly that
    call shape instead."""
    index = _calendar(periods=50)
    frames = _frames(index)
    with pytest.raises(RuntimeError, match="bespoke driver"):
        mod.make_signal_fn("IPO", index, frames)  # exactly the generic dispatch's call shape


def test_bespoke_driver_acknowledgment_flag_restores_normal_behavior():
    index = _calendar(periods=50)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn(
        "IPO", index, frames, driver_acknowledges_short_direction=True)
    sigs = _signals(signal_fn, frames["IPO"])
    assert all(s == SignalResult(entry=True, exit=False) for s in sigs)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=50)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("IPO", index, {})


def test_signal_fn_rejects_out_of_range_position():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    with pytest.raises(IndexError):
        signal_fn(frames["IPO"].iloc[:1], -1)
    with pytest.raises(IndexError):
        signal_fn(frames["IPO"], 20)


def test_signal_fn_rejects_mismatched_bucket_bar():
    index = _calendar(periods=20)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    wrong_frame = frames["IPO"].iloc[:5].copy()
    with pytest.raises(ValueError):
        signal_fn(wrong_frame, 10)


def test_signal_fn_is_constant_time_per_call():
    index = _calendar(start="2013-10-16", periods=5000)
    frames = _frames(index)
    signal_fn = mod.make_signal_fn("IPO", index, frames, driver_acknowledges_short_direction=True)
    frame = frames["IPO"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
