import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import signals
from mining.backtest_engine import SignalResult
from mining.batch5 import fed_net_liquidity_regime as mod


def _weekly(values, start="2019-01-02"):
    dates = pd.date_range(start, periods=len(values), freq="7D")
    return pd.Series([float(v) for v in values], index=dates)


def _price_frame(index):
    n = len(index)
    return pd.DataFrame({"Close": np.linspace(100.0, 200.0, n)}, index=index)


def test_make_signal_fn_rejects_wrong_ticker():
    index = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=index)
    with pytest.raises(ValueError, match="SPY"):
        mod.make_signal_fn("QQQ", index, {"QQQ": frame})


def test_make_signal_fn_rejects_missing_traded_frame():
    index = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(ValueError, match="frames must contain"):
        mod.make_signal_fn("SPY", index, {})


def test_make_signal_fn_rejects_empty_index():
    frame = pd.DataFrame({"Close": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError, match="index must not be empty"):
        mod.make_signal_fn("SPY", pd.DatetimeIndex([]), {"SPY": frame})


def test_make_signal_fn_degrades_gracefully_with_no_liquidity_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn("SPY", index, {"SPY": frame})
    for i in range(5):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_degrades_gracefully_with_empty_liquidity_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = _price_frame(index)
    empty = pd.Series(dtype=float)
    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame},
        walcl=empty, wtregen=empty, rrpontsyd=empty)
    for i in range(5):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_degrades_gracefully_when_history_too_short_for_ma():
    # Only 5 weekly observations -- fewer than ma_weeks=13, so
    # signals.fed_net_liquidity_risk_on_series can never find enough
    # trailing weeks and returns an empty series (no crash).
    walcl = _weekly([100.0] * 5)
    wtregen = _weekly([0.0] * 5)
    rrp = _weekly([0.0] * 5)
    index = pd.bdate_range("2019-01-01", periods=200)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame},
        walcl=walcl, wtregen=wtregen, rrpontsyd=rrp)
    for i in (0, 50, 199):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_delegates_to_signals_module_and_matches_it_exactly():
    """The whole point of this module is that it adds no new net-liquidity/
    MA/lag math -- it must produce IDENTICAL entry/exit decisions to calling
    ``signals.fed_net_liquidity_risk_on_series`` +
    ``signals.lag_boolean_regime_to_availability`` +
    ``signals.make_external_regime_signal_fn`` directly, for every real
    parameter this module declares."""
    rng = np.random.default_rng(1)
    n = 80
    walcl = _weekly(8_000_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n)))
    wtregen = _weekly(600_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.05, n)))
    rrp = _weekly(500.0 * np.cumprod(1.0 + rng.normal(0.0, 0.1, n)))
    index = pd.bdate_range(walcl.index.min(), walcl.index.max() + pd.Timedelta(days=30))
    frame = _price_frame(index)

    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, walcl=walcl, wtregen=wtregen, rrpontsyd=rrp)

    raw_regime = signals.fed_net_liquidity_risk_on_series(
        walcl, wtregen, rrp, ma_weeks=mod.MA_WEEKS)
    available = signals.lag_boolean_regime_to_availability(
        raw_regime, index, lag_trading_days=mod.LAG_TRADING_DAYS)
    expected_signal_fn = signals.make_external_regime_signal_fn(available)

    saw_entry, saw_exit = False, False
    for i in range(len(index)):
        actual = signal_fn(frame.iloc[: i + 1], i)
        expected = expected_signal_fn(frame.iloc[: i + 1], i)
        assert actual == expected, f"bar {i} ({index[i].date()}): {actual} != {expected}"
        saw_entry = saw_entry or actual.entry
        saw_exit = saw_exit or actual.exit
    assert saw_entry and saw_exit


def test_make_signal_fn_passes_catalog_parameters_verbatim(monkeypatch):
    captured = {}
    real = signals.fed_net_liquidity_risk_on_series

    def spy(walcl, wtregen, rrpontsyd, **kwargs):
        captured.update(kwargs)
        return real(walcl, wtregen, rrpontsyd, **kwargs)

    monkeypatch.setattr(mod.signals, "fed_net_liquidity_risk_on_series", spy)
    walcl = _weekly([8_000_000.0] * 30)
    wtregen = _weekly([600_000.0] * 30)
    rrp = _weekly([500.0] * 30)
    index = pd.bdate_range("2019-01-01", periods=400)
    frame = _price_frame(index)
    mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, walcl=walcl, wtregen=wtregen, rrpontsyd=rrp)

    assert captured == {"ma_weeks": mod.PARAMETERS["liquidity_ma_weeks"]}
