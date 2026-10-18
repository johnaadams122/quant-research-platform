"""Tests for scripts/run_vix1d_vix9d_deleverage_mining.py.

Everything runs on injected, deterministic synthetic frames (no network) --
the real data_pipeline fetch functions are monkeypatched. This checks the
driver's OWN wiring (alignment, holdout split, engine call) independently
of what yfinance returns for ^VIX1D/^VIX9D (see the family module's
docstring).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_vix1d_vix9d_deleverage_mining as driver  # noqa: E402
from mining.batch5 import vix1d_vix9d_inversion_deleverage_trigger as mod  # noqa: E402


def _index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _ohlc(index: pd.DatetimeIndex, close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "Open": open_, "High": close * 1.001, "Low": close * 0.999, "Close": close,
    }, index=index)


# ------------------------------------------------------------- split_holdout

def test_split_holdout_last_n_months_is_holdout():
    index = pd.bdate_range("2020-01-01", periods=500)
    in_sample, holdout = driver.split_holdout(index, holdout_months=6)
    assert in_sample.max() < holdout.min()
    assert (in_sample.union(holdout)).equals(index)
    cutoff = index.max() - pd.DateOffset(months=6)
    assert (holdout >= cutoff).all()
    assert (in_sample < cutoff).all()


def test_split_holdout_empty_index():
    empty = pd.DatetimeIndex([])
    in_sample, holdout = driver.split_holdout(empty, holdout_months=6)
    assert len(in_sample) == 0 and len(holdout) == 0


# ------------------------------------------------------------ alignment

def test_align_to_common_calendar_trims_to_intersection():
    asset_index = _index(100, start="2020-01-02")
    signal_index = _index(80, start="2020-01-20")  # starts later, real overlap
    asset_frames = {
        mod.TRADED[0]: _ohlc(asset_index, np.linspace(100, 200, 100)),
        mod.TRADED[1]: _ohlc(asset_index, np.linspace(50, 60, 100)),
    }
    signal_frames = {
        mod.SIGNAL[0]: _ohlc(signal_index, np.full(80, 12.0)),
        mod.SIGNAL[1]: _ohlc(signal_index, np.full(80, 14.0)),
    }
    assets, signals = driver.align_to_common_calendar(asset_frames, signal_frames)
    assert assets[mod.TRADED[0]].index.equals(assets[mod.TRADED[1]].index)
    assert assets[mod.TRADED[0]].index.equals(signals[mod.SIGNAL[0]].index)
    assert signals[mod.SIGNAL[0]].index.equals(signals[mod.SIGNAL[1]].index)
    assert len(assets[mod.TRADED[0]]) == len(signal_index)  # bounded by the shorter leg


def test_align_to_common_calendar_raises_below_floor():
    asset_index = _index(100)
    signal_index = _index(5, start="2020-06-01")  # far too short an overlap
    asset_frames = {
        mod.TRADED[0]: _ohlc(asset_index, np.linspace(100, 200, 100)),
        mod.TRADED[1]: _ohlc(asset_index, np.linspace(50, 60, 100)),
    }
    signal_frames = {
        mod.SIGNAL[0]: _ohlc(signal_index, np.full(5, 12.0)),
        mod.SIGNAL[1]: _ohlc(signal_index, np.full(5, 14.0)),
    }
    with pytest.raises(ValueError, match="insufficient real signal history"):
        driver.align_to_common_calendar(asset_frames, signal_frames)


def test_align_to_common_calendar_raises_on_missing_signal_data():
    asset_index = _index(100)
    asset_frames = {
        mod.TRADED[0]: _ohlc(asset_index, np.linspace(100, 200, 100)),
        mod.TRADED[1]: _ohlc(asset_index, np.linspace(50, 60, 100)),
    }
    signal_frames = {mod.SIGNAL[0]: None, mod.SIGNAL[1]: pd.DataFrame()}
    with pytest.raises(ValueError, match="no OHLC data returned"):
        driver.align_to_common_calendar(asset_frames, signal_frames)


# ------------------------------------------------------ end-to-end rough backtest

def test_run_rough_backtest_end_to_end_with_real_engine():
    n = 500
    index = _index(n)
    xlk_close = 100.0 * np.cumprod(1.0 + np.full(n, 0.0003))
    synth_close = 100.0 * np.cumprod(1.0 + np.full(n, 0.0003) * mod.LEVERAGE)
    # Alternate contango/inversion in blocks so real switches occur.
    block = np.resize(np.concatenate([np.full(40, 12.0), np.full(40, 16.0)]), n)
    vix1d = block
    vix9d = np.full(n, 14.0)

    asset_frames = {
        mod.TRADED[0]: _ohlc(index, synth_close),
        mod.TRADED[1]: _ohlc(index, xlk_close),
    }
    signal_frames = {
        mod.SIGNAL[0]: _ohlc(index, vix1d),
        mod.SIGNAL[1]: _ohlc(index, vix9d),
    }
    aligned_assets, aligned_signals = driver.align_to_common_calendar(
        asset_frames, signal_frames)
    result = driver.run_rough_backtest(aligned_assets, aligned_signals, cost_bps=1.0)

    assert result["n_trades"] > 0
    assert result["in_sample_sessions"] > 0
    assert result["holdout_sessions_excluded"] > 0
    assert result["sessions_total"] == result["in_sample_sessions"]
    assert set(result["trades_by_asset"]).issubset(set(mod.TRADED))
    assert result["net_ret_mean"] is not None


def test_prepare_score_call_description_names_the_real_chain_but_is_inert():
    desc = driver.prepare_score_call_description()
    assert desc["module"] == "mining.abi1_driver"
    assert "does not import or call" in desc["status"]
    assert mod.FAMILY_ID in desc["ledger_writer"]
    # The driver's OWN namespace never binds the scoring module -- checked
    # on the driver module itself (order-independent), not on the
    # process-wide sys.modules cache, since some OTHER test file in a full
    # suite run legitimately imports mining.abi1_driver first and would
    # make a sys.modules-based assertion pass or fail depending on test
    # collection order rather than on what this driver actually does.
    assert "abi1_driver" not in vars(driver)
    # A parsed-AST check (not a raw text search -- the module's own
    # docstring deliberately NAMES mining.abi1_driver in prose, which a
    # plain substring search would misfire on) confirms no real import
    # statement anywhere in the file references it either.
    import ast
    with open(driver.__file__, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=driver.__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("abi1_driver" in alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            assert "abi1_driver" not in module_name
            assert not any("abi1_driver" in alias.name for alias in node.names)


# ------------------------------------------------------------- live fetch wiring

def test_fetch_traded_legs_builds_synthetic_from_injected_xlk(monkeypatch):
    n = 60
    index = _index(n)
    xlk_close = 100.0 * np.cumprod(1.0 + np.full(n, 0.0004))
    xlk_ohlc_with_date = pd.DataFrame({
        "Date": index, "XLK": xlk_close,
    })
    real_ohlc = _ohlc(index, xlk_close).reset_index().rename(columns={"index": "Date"})

    import data_pipeline

    def fake_fetch_ohlc(ticker, start="max"):
        assert ticker == "XLK"
        return real_ohlc.copy()

    monkeypatch.setattr(data_pipeline, "fetch_ohlc", fake_fetch_ohlc)

    irx = pd.Series(0.02, index=pd.DatetimeIndex(index))
    legs = driver.fetch_traded_legs(irx_series=irx)
    assert set(legs) == set(mod.TRADED)
    assert list(legs[mod.TRADED[1]]["Close"]) == list(xlk_close)
    assert len(legs[mod.TRADED[0]]) == n
    assert list(legs[mod.TRADED[0]].columns) == ["Open", "High", "Low", "Close"]
