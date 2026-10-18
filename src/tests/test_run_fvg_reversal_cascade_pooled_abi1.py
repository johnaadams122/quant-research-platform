"""Integration test for scripts/run_fvg_reversal_cascade_pooled_abi1.py --
proves the FULL composition (real signal_fn -> real intraday_backtest_engine
-> mining.day_trading_abi1_bridge.build_pooled_day_trading_curve)
works end to end on small, synthetic, hand-constructed minute-bar frames for
all three pool members (QQQ/SPY/IWM), independent of any network call (no
real Alpaca fetch, no real SPY fetch, no ledger write).

QQQ's own rows MIRROR test_cascade_signals.py's own scenario 1 (Build 1
happy path) fixture bar-for-bar (reused, not re-derived -- same convention
run_vwap_cross_decaying_conviction_abi1's own test file established). SPY
and IWM are deliberately no-bias days (see test_cascade_signals._no_bias_
rows' own docstring) -- they never enter, so this test also exercises the
pooled bridge's own zero-fill behavior for members that traded nothing.
"""
import os
import sys

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import day_trading_abi1_bridge as bridge  # noqa: E402
import run_fvg_reversal_cascade_pooled_abi1 as driver  # noqa: E402

TZ = "America/New_York"
DAY_T_MINUS_2 = "2026-03-02"
DAY_T_MINUS_1 = "2026-03-03"
DAY_T = "2026-03-04"


def _ts(date_str, time_str):
    return pd.Timestamp(f"{date_str} {time_str}", tz=TZ)


def _row(o, h, l, c, v=1000.0):
    return {"Open": float(o), "High": float(h), "Low": float(l), "Close": float(c), "Volume": float(v)}


def _frame(rows):
    index = pd.DatetimeIndex([ts for ts, _ in rows])
    data = {col: [r[col] for _, r in rows] for col in ["Open", "High", "Low", "Close", "Volume"]}
    return pd.DataFrame(data, index=index).sort_index()


def _bullish_bias_rows():
    # test_cascade_signals._bullish_bias_rows, mirrored bar-for-bar.
    return [
        (_ts(DAY_T_MINUS_2, "09:30"), _row(100, 100, 98, 99)),
        (_ts(DAY_T_MINUS_2, "09:31"), _row(99, 102, 99, 101)),
        (_ts(DAY_T_MINUS_1, "09:30"), _row(99, 99, 97, 97.5)),
        (_ts(DAY_T_MINUS_1, "09:31"), _row(97.5, 100, 97.5, 99)),
    ]


def _happy_path_today_rows():
    # test_cascade_signals._happy_path_today_rows, mirrored bar-for-bar.
    return [
        (_ts(DAY_T, "09:30"), _row(100, 101, 99, 100)),
        (_ts(DAY_T, "10:30"), _row(100.5, 102.5, 100, 102.5)),
        (_ts(DAY_T, "11:30"), _row(100, 100.2, 99.8, 100)),
        (_ts(DAY_T, "11:35"), _row(101, 103, 100.9, 102.5)),
        (_ts(DAY_T, "11:40"), _row(102.6, 103.5, 102.4, 103)),
        (_ts(DAY_T, "11:45"), _row(103, 103.2, 102, 102.8)),
        (_ts(DAY_T, "11:50"), _row(102.85, 102.9, 102.75, 102.85)),
        (_ts(DAY_T, "11:51"), _row(102.9, 103.0, 102.8, 102.95)),
        (_ts(DAY_T, "12:00"), _row(103.0, 105.0, 102.85, 104.8)),
    ]


def _no_bias_rows():
    # test_cascade_signals._no_bias_rows, mirrored bar-for-bar -- T-1's
    # daily aggregate stays fully inside T-2's range, so daily_bias is
    # "none" and no entry can ever fire for this member.
    return [
        (_ts(DAY_T_MINUS_2, "09:30"), _row(100, 100, 98, 99)),
        (_ts(DAY_T_MINUS_2, "09:31"), _row(99, 102, 99, 101)),
        (_ts(DAY_T_MINUS_1, "09:30"), _row(100, 100, 99, 100)),
        (_ts(DAY_T_MINUS_1, "09:31"), _row(100, 101, 99.5, 100.5)),
    ]


def test_real_signal_fn_through_real_engine_and_pooled_bridge_end_to_end():
    """QQQ plays out the real scenario-1 happy-path trade; SPY and IWM are
    both no-bias days that never enter -- proving the pooled bridge's
    equal-weight-mean/zero-fill combination on a REAL mix of one active
    and two idle members, wired through THIS driver's own
    run_real_intraday_backtest (not just the underlying factory/engine,
    already proven by test_cascade_signals.py itself)."""
    qqq_bars = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    spy_bars = _frame(_no_bias_rows())
    iwm_bars = _frame(_no_bias_rows())

    qqq_trades = driver.run_real_intraday_backtest(qqq_bars, cost_bps=0.0)
    spy_trades = driver.run_real_intraday_backtest(spy_bars, cost_bps=0.0)
    iwm_trades = driver.run_real_intraday_backtest(iwm_bars, cost_bps=0.0)
    assert len(qqq_trades) == 1
    assert len(spy_trades) == 0
    assert len(iwm_trades) == 0
    qqq_net_ret = float(qqq_trades.iloc[0]["net_ret"])

    calendar = pd.DatetimeIndex(sorted(set(qqq_bars.index.date)
                                        | set(spy_bars.index.date)
                                        | set(iwm_bars.index.date)))
    curves = {
        "QQQ": bridge.build_day_trading_curve(qqq_trades, calendar),
        "SPY": bridge.build_day_trading_curve(spy_trades, calendar),
        "IWM": bridge.build_day_trading_curve(iwm_trades, calendar),
    }
    pooled = bridge.build_pooled_day_trading_curve(curves)

    day_t_ts = pd.Timestamp(DAY_T)
    day_minus2_ts = pd.Timestamp(DAY_T_MINUS_2)
    # Equal-weight mean across 3 members, 2 of which are 0.0 that day --
    # NOT the undiluted single-member return.
    assert pooled["strategy_daily"].loc[day_t_ts] == pytest.approx(qqq_net_ret / 3.0)
    assert pooled["strategy_daily"].loc[day_minus2_ts] == 0.0
    assert bool(pooled["in_market"].loc[day_t_ts]) is True
    assert bool(pooled["in_market"].loc[day_minus2_ts]) is False
    assert len(pooled["trades_df"]) == 1
    assert pooled["trades_df"]["ticker"].iloc[0] == "QQQ"


def test_parameters_match_factory_defaults_and_catalog_shape():
    assert set(driver.PARAMETERS) == {
        "min_target_r_multiple", "force_flat_et", "max_hold_days", "direction"}
    assert driver.POOL_TICKERS == ("QQQ", "SPY", "IWM")
    assert driver.SHAPE == "pooled_single_asset"
    assert driver.TAX_POLICY == "none"
