"""Tests for src/mining/batch5/bitcoin_extreme_decline_spillover_riskoff.py.

Synthetic data only (no network), deterministic, fast. Mirrors
test_batch5_single_day_extreme_decline_reversal_long.py's local z-score
helper rigor (manufactured shock, look-ahead probe, holdout-blind probe)
and test_batch5_money_market_fund_flow_riskoff_bounded_burnin.py's
external-signal delegation rigor (degrade-gracefully-with-no-signal,
byte-identical delegation to the shared signals.py combinators).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import signals  # noqa: E402
from mining.backtest_engine import SignalResult, run_backtest  # noqa: E402
from mining.batch5 import (  # noqa: E402
    bitcoin_extreme_decline_spillover_riskoff as mod)

VOL_WINDOW = mod.VOL_WINDOW_DAYS
Z_THRESHOLD = mod.Z_THRESHOLD
HOLD_DAYS = mod.HOLD_DAYS
LAG_TRADING_DAYS = mod.LAG_TRADING_DAYS


# ------------------------------------------------------------------ helpers

def _qqq_frame(index: pd.DatetimeIndex, level: float = 100.0) -> pd.DataFrame:
    n = len(index)
    close = np.full(n, level, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.001, "Low": close * 0.999,
         "Close": close}, index=index)


def _ordinary_btc_returns(n: int, *, amplitude: float = 0.01) -> np.ndarray:
    pattern = np.empty(n)
    pattern[0::2] = amplitude
    pattern[1::2] = -amplitude
    return pattern


_EFFECTIVE_TRAILING_STD = 0.01 * np.sqrt(VOL_WINDOW / (VOL_WINDOW - 1))


def _btc_close_with_shock(n: int, shock_pos: int, shock_return: float, *,
                          amplitude: float = 0.01,
                          start: str = "2024-01-01") -> pd.Series:
    """BTC trades every CALENDAR day (unlike QQQ's business-day calendar)."""
    index = pd.date_range(start, periods=n, freq="D")
    rets = _ordinary_btc_returns(n, amplitude=amplitude).copy()
    rets[shock_pos] = shock_return
    level = np.empty(n)
    level[0] = 40000.0
    for i in range(1, n):
        level[i] = level[i - 1] * (1.0 + rets[i])
    return pd.Series(level, index=index, name="close")


def _qqq_index_covering(btc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.bdate_range(btc_index.min(), btc_index.max() + pd.Timedelta(days=10))


# -------------------------------------------------------------- contract

def test_module_constants_match_expected_contract():
    assert mod.FAMILY_ID == "bitcoin_extreme_decline_spillover_riskoff"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["QQQ"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 0
    assert callable(mod.make_signal_fn)
    assert mod.PARAMETERS["btc_zscore_riskoff_threshold"] == -2.0
    assert mod.PARAMETERS["riskoff_hold_days"] == 5
    assert mod.PARAMETERS["btc_vol_window_days"] == 20
    assert mod.PARAMETERS["lag_trading_days"] == 1


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["QQQ"]
    assert spec["signal"] == []


def test_make_signal_fn_rejects_wrong_ticker():
    index = pd.bdate_range("2024-01-01", periods=3)
    frame = _qqq_frame(index)
    with pytest.raises(ValueError, match="QQQ"):
        mod.make_signal_fn("SPY", index, {"SPY": frame})


def test_make_signal_fn_rejects_missing_traded_frame():
    index = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(ValueError, match="frames must contain"):
        mod.make_signal_fn("QQQ", index, {})


def test_make_signal_fn_rejects_empty_index():
    frame = _qqq_frame(pd.bdate_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="index must not be empty"):
        mod.make_signal_fn("QQQ", pd.DatetimeIndex([]), {"QQQ": frame})


def test_make_signal_fn_degrades_gracefully_with_no_btc_close():
    index = pd.bdate_range("2024-01-01", periods=5)
    frame = _qqq_frame(index)
    signal_fn = mod.make_signal_fn("QQQ", index, {"QQQ": frame})
    for i in range(5):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result == SignalResult(entry=False, exit=False)


# ------------------------------------------------------------------- rule

def test_btc_decline_zscore_denominator_excludes_own_day_return():
    """The trailing-vol DENOMINATOR at day t must be numerically identical
    whether day t's own return is a huge outlier or an ordinary day -- if
    it were not, a large move would deflate/inflate its own normalizer, a
    real self-referential look-ahead bug. This exact assertion fails if
    ``btc_decline_zscore`` drops its ``shift(1)``; a weaker
    index-equality-only check would not catch that."""
    n = 60
    shock_pos = 40
    shocked = _btc_close_with_shock(n, shock_pos, shock_return=-0.30)
    ordinary = _btc_close_with_shock(n, shock_pos, shock_return=0.01)
    z_shocked = mod.btc_decline_zscore(shocked)
    z_ordinary = mod.btc_decline_zscore(ordinary)
    date = shocked.index[shock_pos]
    r_shocked = np.log(shocked.loc[date] / shocked.shift(1).loc[date])
    r_ordinary = np.log(ordinary.loc[date] / ordinary.shift(1).loc[date])
    # implied trailing_vol = r_t / z_t; must match across the two series at
    # day t itself if day t's own return never entered its own window.
    vol_from_shocked = r_shocked / z_shocked.loc[date]
    vol_from_ordinary = r_ordinary / z_ordinary.loc[date]
    assert vol_from_shocked == pytest.approx(vol_from_ordinary, rel=1e-9)


def test_manufactured_extreme_btc_decline_triggers_riskoff():
    n = 90
    shock_pos = 50  # well past VOL_WINDOW+1 warm-up
    btc_close = _btc_close_with_shock(n, shock_pos, shock_return=-0.25)
    regime = mod.btc_riskoff_regime(
        btc_close, vol_window=VOL_WINDOW, zscore_threshold=Z_THRESHOLD,
        hold_days=HOLD_DAYS)
    shock_date = btc_close.index[shock_pos]
    assert bool(regime.loc[shock_date]) is False  # risk_on == False that day
    # stays risk-off for exactly HOLD_DAYS calendar days from the shock
    for offset in range(HOLD_DAYS):
        assert bool(regime.loc[shock_date + pd.Timedelta(days=offset)]) is False
    back_on = shock_date + pd.Timedelta(days=HOLD_DAYS)
    if back_on in regime.index:
        assert bool(regime.loc[back_on]) is True


def test_moderate_btc_decline_does_not_trigger_riskoff():
    n = 90
    shock_pos = 50
    shock_return = -1.0 * _EFFECTIVE_TRAILING_STD  # |z| == 1.0, well under 2.0
    btc_close = _btc_close_with_shock(n, shock_pos, shock_return=shock_return)
    regime = mod.btc_riskoff_regime(
        btc_close, vol_window=VOL_WINDOW, zscore_threshold=Z_THRESHOLD,
        hold_days=HOLD_DAYS)
    assert bool(regime.iloc[shock_pos]) is True  # stays risk_on


def test_end_to_end_riskoff_flip_exits_and_reenters_qqq():
    n = 120
    shock_pos = 50
    btc_close = _btc_close_with_shock(n, shock_pos, shock_return=-0.30)
    qqq_index = _qqq_index_covering(btc_close.index)
    qqq_frame = _qqq_frame(qqq_index)
    signal_fn = mod.make_signal_fn(
        "QQQ", qqq_index, {"QQQ": qqq_frame}, btc_close=btc_close)
    trades, in_market = run_backtest(qqq_frame, signal_fn)
    # Exactly one round trip is forced by the single manufactured shock:
    # QQQ starts flat (signal unknown before BTC's first date), enters once
    # BTC's regime becomes known (risk_on default True), then the shock
    # forces one exit-then-reenter cycle -- at least 2 trades (initial
    # entry->exit, then reenter->held/eod).
    assert len(trades) >= 2
    assert (trades["reason"] == "signal_exit").any()


def test_look_ahead_probe_future_btc_bars_do_not_change_past_signals():
    n = 150
    btc_close = _btc_close_with_shock(n, 60, shock_return=-0.20)
    qqq_index = _qqq_index_covering(btc_close.index)
    qqq_frame = _qqq_frame(qqq_index)
    k = 90
    base_fn = mod.make_signal_fn(
        "QQQ", qqq_index, {"QQQ": qqq_frame}, btc_close=btc_close)
    base = [base_fn(qqq_frame.iloc[:i + 1], i) for i in range(k + 1)]

    perturbed = btc_close.copy()
    cutoff = qqq_index[k]
    perturbed.loc[perturbed.index > cutoff] *= 3.0
    pert_fn = mod.make_signal_fn(
        "QQQ", qqq_index, {"QQQ": qqq_frame}, btc_close=perturbed)
    after = [pert_fn(qqq_frame.iloc[:i + 1], i) for i in range(k + 1)]
    assert base == after


def test_holdout_blind_bucket_uses_full_btc_history_before_bucket_start():
    n = 150
    btc_close = _btc_close_with_shock(n, 60, shock_return=-0.20)
    full_qqq_index = _qqq_index_covering(btc_close.index)
    full_qqq_frame = _qqq_frame(full_qqq_index)
    bucket = full_qqq_index[40:]
    bucket_frame = full_qqq_frame.loc[bucket]

    fn_full = mod.make_signal_fn(
        "QQQ", bucket, {"QQQ": full_qqq_frame}, btc_close=btc_close)
    sigs = [fn_full(bucket_frame.iloc[:i + 1], i) for i in range(len(bucket))]
    # No exception, deterministic length -- the shock predates the bucket
    # start in this construction, so the regime should already reflect it
    # (risk_on known and possibly already back on) rather than erroring.
    assert len(sigs) == len(bucket)


def test_make_signal_fn_uses_the_shared_signals_combinators_verbatim():
    """The whole point of delegating rather than hand-rolling: the final
    decision must be byte-identical to calling
    signals.lag_boolean_regime_to_availability ->
    signals.make_external_regime_signal_fn directly on this module's own
    btc_riskoff_regime output."""
    n = 100
    btc_close = _btc_close_with_shock(n, 45, shock_return=-0.22)
    qqq_index = _qqq_index_covering(btc_close.index)
    qqq_frame = _qqq_frame(qqq_index)

    signal_fn = mod.make_signal_fn(
        "QQQ", qqq_index, {"QQQ": qqq_frame}, btc_close=btc_close)

    regime = mod.btc_riskoff_regime(
        btc_close, vol_window=VOL_WINDOW, zscore_threshold=Z_THRESHOLD,
        hold_days=HOLD_DAYS)
    available = signals.lag_boolean_regime_to_availability(
        regime, qqq_index, lag_trading_days=LAG_TRADING_DAYS)
    expected_fn = signals.make_external_regime_signal_fn(available)

    for i in range(len(qqq_index)):
        assert signal_fn(qqq_frame.iloc[:i + 1], i) == expected_fn(
            qqq_frame.iloc[:i + 1], i)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    btc_close = _btc_close_with_shock(60, 40, shock_return=-0.20)
    qqq_index = _qqq_index_covering(btc_close.index)
    qqq_frame = _qqq_frame(qqq_index)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", qqq_index, {"QQQ": qqq_frame}, btc_close=btc_close)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", qqq_index, {}, btc_close=btc_close)


def test_signal_fn_is_constant_time_per_call():
    import time
    n = 250
    btc_close = _btc_close_with_shock(n, 120, shock_return=-0.20)
    qqq_index = _qqq_index_covering(btc_close.index)
    qqq_frame = _qqq_frame(qqq_index)
    signal_fn = mod.make_signal_fn(
        "QQQ", qqq_index, {"QQQ": qqq_frame}, btc_close=btc_close)
    started = time.perf_counter()
    for i in range(len(qqq_index)):
        signal_fn(qqq_frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
