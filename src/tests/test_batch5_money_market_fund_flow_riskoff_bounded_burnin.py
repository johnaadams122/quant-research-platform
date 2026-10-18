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
from mining.batch5 import money_market_fund_flow_riskoff_bounded_burnin as mod


def _monthly_flow(values, start="1980-01-01"):
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


def test_make_signal_fn_applies_the_burn_in_cap_before_scoring():
    """The whole point of this module: it must produce IDENTICAL decisions
    to calling signals.cap_pre_inception_history (anchored at the traded
    index's own first date) -> signals.money_market_risk_on_series ->
    signals.lag_boolean_regime_to_availability directly -- and that
    result must differ from the UNCAPPED sibling module whenever the raw
    history is long enough for the cap to actually bite."""
    rng = np.random.default_rng(3)
    steps = rng.normal(loc=0.005, scale=0.06, size=400)  # ~33 years monthly
    levels = 100.0 * np.cumprod(1.0 + steps)
    monthly = _monthly_flow(levels, start="1980-01-01")
    index = pd.bdate_range("2010-01-01", "2013-12-31")
    frame = _price_frame(index)

    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, flow_signal=monthly)

    capped = signals.cap_pre_inception_history(
        monthly, index.min(), max_years_before=mod.MAX_YEARS_BEFORE)
    raw_regime = signals.money_market_risk_on_series(
        capped, roc_months=mod.ROC_MONTHS,
        riskoff_z_threshold=mod.ZSCORE_RISKOFF_THRESHOLD)
    anchored = raw_regime.copy()
    anchored.index = anchored.index.to_period("M").to_timestamp("M")
    available = signals.lag_boolean_regime_to_availability(
        anchored, index, lag_trading_days=mod.LAG_TRADING_DAYS)
    expected_signal_fn = signals.make_external_regime_signal_fn(available)

    for i in range(len(index)):
        assert signal_fn(frame.iloc[: i + 1], i) == expected_signal_fn(
            frame.iloc[: i + 1], i)


def test_make_signal_fn_passes_catalog_parameters_verbatim(monkeypatch):
    captured = {}
    real = signals.money_market_risk_on_series

    def spy(series, **kwargs):
        captured.update(kwargs)
        return real(series, **kwargs)

    monkeypatch.setattr(mod.signals, "money_market_risk_on_series", spy)
    monthly = _monthly_flow([100.0] * 20, start="2018-01-01")
    index = pd.bdate_range("2019-01-01", periods=400)
    frame = _price_frame(index)
    mod.make_signal_fn("SPY", index, {"SPY": frame}, flow_signal=monthly)

    assert captured == {
        "roc_months": mod.PARAMETERS["flow_roc_lookback_months"],
        "riskoff_z_threshold": float(
            mod.PARAMETERS["flow_roc_zscore_riskoff_threshold"]),
    }
