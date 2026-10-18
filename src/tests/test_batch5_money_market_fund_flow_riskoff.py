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
from mining.batch5 import money_market_fund_flow_riskoff as mod


def _monthly_flow(values, start="2019-01-01"):
    months = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series([float(v) for v in values], index=months)


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


def test_make_signal_fn_degrades_gracefully_with_no_flow_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn("SPY", index, {"SPY": frame})
    for i in range(5):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_degrades_gracefully_with_empty_flow_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, flow_signal=pd.Series(dtype=float))
    for i in range(5):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_degrades_gracefully_when_history_too_short_for_roc():
    # Only 2 monthly observations -- fewer than roc_months=3, so
    # signals.money_market_risk_on_series can never find a prior-period
    # match and returns an empty series (no crash).
    monthly = _monthly_flow([100.0, 101.0])
    index = pd.bdate_range("2019-01-01", periods=200)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, flow_signal=monthly)
    for i in (0, 50, 199):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_delegates_to_signals_module_and_matches_it_exactly():
    """The whole point of this module is that it adds no new ROC/z-score/lag
    math -- it must produce IDENTICAL entry/exit decisions to calling
    ``signals.money_market_risk_on_series`` +
    ``signals.lag_boolean_regime_to_availability`` +
    ``signals.make_external_regime_signal_fn`` directly, for every real
    parameter this module declares."""
    rng = np.random.default_rng(0)
    # A long enough, varied enough monthly series that both regimes appear
    # under a 3-month ROC z-score threshold of 1.0.
    steps = rng.normal(loc=0.01, scale=0.08, size=48)
    levels = 100.0 * np.cumprod(1.0 + steps)
    monthly = _monthly_flow(levels)
    index = pd.bdate_range("2019-01-01", "2023-06-30")
    frame = _price_frame(index)

    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, flow_signal=monthly)

    raw_regime = signals.money_market_risk_on_series(
        monthly, roc_months=mod.ROC_MONTHS,
        riskoff_z_threshold=mod.ZSCORE_RISKOFF_THRESHOLD)
    anchored = raw_regime.copy()
    anchored.index = anchored.index.to_period("M").to_timestamp("M")
    available = signals.lag_boolean_regime_to_availability(
        anchored, index, lag_trading_days=mod.LAG_TRADING_DAYS)
    expected_signal_fn = signals.make_external_regime_signal_fn(available)

    saw_entry, saw_exit = False, False
    for i in range(len(index)):
        actual = signal_fn(frame.iloc[: i + 1], i)
        expected = expected_signal_fn(frame.iloc[: i + 1], i)
        assert actual == expected, f"bar {i} ({index[i].date()}): {actual} != {expected}"
        saw_entry = saw_entry or actual.entry
        saw_exit = saw_exit or actual.exit
    # both regimes must actually appear in this fixture, or the test
    # proves nothing about real entry/exit wiring
    assert saw_entry and saw_exit


def test_make_signal_fn_passes_catalog_parameters_verbatim(monkeypatch):
    captured = {}
    real = signals.money_market_risk_on_series

    def spy(series, **kwargs):
        captured.update(kwargs)
        return real(series, **kwargs)

    monkeypatch.setattr(mod.signals, "money_market_risk_on_series", spy)
    monthly = _monthly_flow([100.0] * 20)
    index = pd.bdate_range("2019-01-01", periods=400)
    frame = _price_frame(index)
    mod.make_signal_fn("SPY", index, {"SPY": frame}, flow_signal=monthly)

    assert captured == {
        "roc_months": mod.PARAMETERS["flow_roc_lookback_months"],
        "riskoff_z_threshold": float(
            mod.PARAMETERS["flow_roc_zscore_riskoff_threshold"]),
    }


def test_month_start_index_is_reanchored_to_month_end_before_lag():
    """RMFSL is dated at month-START (FRED convention); this module must
    re-anchor to month-END before applying the lag, matching how the real
    publication lag was measured (from month-end, same convention as
    margin_debt_growth_regime.py's FINRA lag)."""
    monthly = _monthly_flow([100.0, 90.0, 80.0, 70.0, 200.0])
    index = pd.bdate_range("2019-01-01", periods=400)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, flow_signal=monthly)

    raw_regime = signals.money_market_risk_on_series(
        monthly, roc_months=mod.ROC_MONTHS,
        riskoff_z_threshold=mod.ZSCORE_RISKOFF_THRESHOLD)
    assert not raw_regime.empty
    # If the module failed to re-anchor, lag_boolean_regime_to_availability
    # would be fed month-START dates directly.
    month_start_available = signals.lag_boolean_regime_to_availability(
        raw_regime, index, lag_trading_days=mod.LAG_TRADING_DAYS)
    anchored = raw_regime.copy()
    anchored.index = anchored.index.to_period("M").to_timestamp("M")
    month_end_available = signals.lag_boolean_regime_to_availability(
        anchored, index, lag_trading_days=mod.LAG_TRADING_DAYS)
    assert not month_start_available.index.equals(month_end_available.index)

    expected_signal_fn = signals.make_external_regime_signal_fn(month_end_available)
    for i in range(len(index)):
        assert signal_fn(frame.iloc[: i + 1], i) == expected_signal_fn(
            frame.iloc[: i + 1], i)
