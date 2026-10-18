"""Tests for mining.batch5.margin_debt_deleverage_synthetic_leverage_flip.

Mirrors test_batch5_margin_debt_growth_regime.py's own convention: this
module must add NO new FINRA growth/lag math of its own -- every test here
proves it is a thin adapter delegating to the already-tested
``mining.signals`` functions (``finra_margin_debt_risk_on_series`` +
``make_external_regime_target_fn``), with the ONE genuine parameter
difference being ``margin_debt_lookback_months=1`` (month-over-month
deceleration/contraction) instead of the sibling family's 12
(trailing-12-month growth regime).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import signals
from mining.batch5 import margin_debt_deleverage_synthetic_leverage_flip as mod


def _monthly_debt(values, start="2019-01-31"):
    months = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series([float(v) for v in values], index=months)


def _price_frame(index, start=100.0):
    n = len(index)
    return pd.DataFrame({
        "Open": np.linspace(start, start * 2, n),
        "High": np.linspace(start, start * 2, n) * 1.001,
        "Low": np.linspace(start, start * 2, n) * 0.999,
        "Close": np.linspace(start, start * 2, n),
    }, index=index)


def test_contract_fields():
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [f"SPY_{int(mod.LEVERAGE)}x_synthetic", "SH"]
    assert mod.SIGNAL == []
    assert set(mod.TRADED).isdisjoint(mod.SIGNAL)
    assert mod.BENCHMARK == "SPY"
    # module PARAMETERS must be JSON-serializable (canonical-drift-check
    # convention every sibling bespoke driver relies on)
    import json
    json.dumps(mod.PARAMETERS, sort_keys=True)


def test_make_target_fn_rejects_empty_index():
    with pytest.raises(ValueError, match="index must not be empty"):
        mod.make_target_fn(pd.DatetimeIndex([]), margin_signal=_monthly_debt([100.0] * 5))


def test_make_target_fn_degrades_gracefully_with_no_margin_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    risk = _price_frame(index)
    defensive = _price_frame(index, start=30.0)
    target_fn = mod.make_target_fn(index)
    for i in range(1, 5):
        truncated = {mod.TRADED[0]: risk.iloc[:i], mod.TRADED[1]: defensive.iloc[:i]}
        assert target_fn(truncated) is None


def test_make_target_fn_degrades_gracefully_with_empty_margin_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    risk = _price_frame(index)
    defensive = _price_frame(index, start=30.0)
    target_fn = mod.make_target_fn(index, margin_signal=pd.Series(dtype=float))
    for i in range(1, 5):
        truncated = {mod.TRADED[0]: risk.iloc[:i], mod.TRADED[1]: defensive.iloc[:i]}
        assert target_fn(truncated) is None


def test_make_target_fn_degrades_gracefully_when_history_too_short_for_lookback():
    # Only 1 monthly observation -- fewer than LOOKBACK_MONTHS+1=2 needed to
    # ever find a prior-period match, so finra_margin_debt_growth_available_
    # series can never compute a MoM change and returns an empty series.
    monthly = _monthly_debt([100.0])
    index = pd.bdate_range("2019-01-01", periods=200)
    risk = _price_frame(index)
    defensive = _price_frame(index, start=30.0)
    target_fn = mod.make_target_fn(index, margin_signal=monthly)
    for i in (10, 100, 199):
        truncated = {mod.TRADED[0]: risk.iloc[:i], mod.TRADED[1]: defensive.iloc[:i]}
        assert target_fn(truncated) is None


def test_make_target_fn_delegates_to_signals_module_and_matches_it_exactly():
    """The whole point of this module: identical decisions to calling
    signals.finra_margin_debt_risk_on_series + signals.
    make_external_regime_target_fn directly, for every real parameter this
    module declares (lookback_months=1, not the sibling's 12)."""
    # Alternating MoM growth/contraction so both regimes actually occur.
    values = []
    level = 100.0
    for i in range(30):
        level *= 1.05 if (i % 4) < 2 else 0.90
        values.append(level)
    monthly = _monthly_debt(values)
    index = pd.bdate_range("2019-01-01", "2021-12-31")
    risk = _price_frame(index)
    defensive = _price_frame(index, start=30.0)

    target_fn = mod.make_target_fn(index, margin_signal=monthly)

    expected_risk_on = signals.finra_margin_debt_risk_on_series(
        monthly, index, lookback_months=mod.LOOKBACK_MONTHS,
        lag_sessions=mod.LAG_TRADING_DAYS, growth_threshold=mod.GROWTH_THRESHOLD)
    expected_target_fn = signals.make_external_regime_target_fn(
        expected_risk_on, risk_ticker=mod.TRADED[0], defensive_ticker=mod.TRADED[1])

    saw_risk_on, saw_risk_off = False, False
    for i in range(2, len(index) + 1):
        truncated = {mod.TRADED[0]: risk.iloc[:i], mod.TRADED[1]: defensive.iloc[:i]}
        actual = target_fn(truncated)
        expected = expected_target_fn(truncated)
        assert actual == expected, f"bar {i}: {actual} != {expected}"
        saw_risk_on = saw_risk_on or actual == mod.TRADED[0]
        saw_risk_off = saw_risk_off or actual == mod.TRADED[1]
    assert saw_risk_on and saw_risk_off


def test_make_target_fn_passes_catalog_parameters_verbatim(monkeypatch):
    captured = {}
    real = signals.finra_margin_debt_risk_on_series

    def spy(monthly_balances, traded_calendar, **kwargs):
        captured.update(kwargs)
        return real(monthly_balances, traded_calendar, **kwargs)

    monkeypatch.setattr(mod.signals, "finra_margin_debt_risk_on_series", spy)
    monthly = _monthly_debt([100.0] * 20)
    index = pd.bdate_range("2019-01-01", periods=400)
    mod.make_target_fn(index, margin_signal=monthly)

    assert captured == {
        "lookback_months": mod.PARAMETERS["margin_debt_lookback_months"],
        "lag_sessions": mod.PARAMETERS["signal_lag_trading_days"],
        "growth_threshold": float(mod.PARAMETERS["growth_threshold"]),
    }
    assert mod.PARAMETERS["margin_debt_lookback_months"] == 1  # MoM, not YoY


def test_lookback_is_genuinely_different_mechanism_than_growth_regime_sibling():
    """Distinctness guard: this family must NOT share the sibling's
    lookback_months=12 trailing-year construction -- that would make it a
    parameter-tweak re-test of margin_debt_growth_regime, not a genuinely
    different mechanism, and would risk double-charging one trial for two
    near-identical hypotheses."""
    from mining.batch5 import margin_debt_growth_regime as sibling
    assert mod.LOOKBACK_MONTHS != sibling.LOOKBACK_MONTHS
    assert mod.LAG_TRADING_DAYS == sibling.LAG_TRADING_DAYS  # same disclosed
                                                              # publication-lag
                                                              # constant, reused
                                                              # not reinvented
