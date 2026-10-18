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
from mining.batch5 import margin_debt_growth_regime as mod


def _monthly_debt(values, start="2019-01-31"):
    months = pd.date_range(start, periods=len(values), freq="ME")
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


def test_make_signal_fn_degrades_gracefully_with_no_margin_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn("SPY", index, {"SPY": frame})
    for i in range(5):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_degrades_gracefully_with_empty_margin_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, margin_signal=pd.Series(dtype=float))
    for i in range(5):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_degrades_gracefully_when_history_too_short_for_lookback():
    # Only 6 monthly observations -- fewer than LOOKBACK_MONTHS=12, so
    # signals.finra_margin_debt_growth_available_series can never find a
    # prior-period match and returns an empty series (no crash).
    monthly = _monthly_debt([100.0] * 6)
    index = pd.bdate_range("2019-01-01", periods=200)
    frame = _price_frame(index)
    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, margin_signal=monthly)
    for i in (0, 50, 199):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


def test_make_signal_fn_delegates_to_signals_module_and_matches_it_exactly():
    """The whole point of this module is that it adds no new growth/lag
    math -- it must produce IDENTICAL entry/exit decisions to calling
    ``signals.finra_margin_debt_risk_on_series`` +
    ``signals.make_external_regime_signal_fn`` directly, for every real
    parameter this module declares."""
    ramp = [100.0 * (1.03 ** i) for i in range(13)]
    monthly = _monthly_debt(ramp + [50.0] * 12)  # rises, then drops
    index = pd.bdate_range("2019-01-01", "2021-06-30")
    frame = _price_frame(index)

    signal_fn = mod.make_signal_fn(
        "SPY", index, {"SPY": frame}, margin_signal=monthly)

    expected_risk_on = signals.finra_margin_debt_risk_on_series(
        monthly, index, lookback_months=mod.LOOKBACK_MONTHS,
        lag_sessions=mod.LAG_TRADING_DAYS, growth_threshold=mod.GROWTH_THRESHOLD)
    expected_signal_fn = signals.make_external_regime_signal_fn(expected_risk_on)

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
    real = signals.finra_margin_debt_risk_on_series

    def spy(monthly_balances, traded_calendar, **kwargs):
        captured.update(kwargs)
        return real(monthly_balances, traded_calendar, **kwargs)

    monkeypatch.setattr(mod.signals, "finra_margin_debt_risk_on_series", spy)
    monthly = _monthly_debt([100.0] * 20)
    index = pd.bdate_range("2019-01-01", periods=400)
    frame = _price_frame(index)
    mod.make_signal_fn("SPY", index, {"SPY": frame}, margin_signal=monthly)

    assert captured == {
        "lookback_months": mod.PARAMETERS["margin_debt_lookback_months"],
        "lag_sessions": mod.PARAMETERS["signal_lag_trading_days"],
        "growth_threshold": float(mod.PARAMETERS["growth_threshold"]),
    }
