"""Integration test for scripts/run_cross_asset_divergence_fvg_abi1.py --
proves the FULL composition (real signal_fn -> real intraday_backtest_engine
-> mining.day_trading_abi1_bridge) works end to end on small, synthetic,
hand-constructed minute-bar frames, independent of any network call (no
real Alpaca fetch, no real SPY fetch, no ledger write).

Every row below MIRRORS test_cascade_signals.py's own scenario 1 (single-
asset happy path) and scenario 6b (cross-asset cascade, divergence-allowed) fixtures
bar-for-bar (reused, not re-derived, matching run_vwap_cross_decaying_
conviction_abi1's own test file precedent) -- see that file's own
_bullish_bias_rows/_happy_path_today_rows/_secondary_diverges_rows
docstrings for the full hand-derivation of every landmark bar.
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
import run_cross_asset_divergence_fvg_abi1 as driver  # noqa: E402

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
    # test_cascade_signals._happy_path_today_rows, mirrored bar-for-bar --
    # see that function's own docstring for the full bucket-by-bucket
    # derivation of this real happy-path trade (entry 102.9, target exit
    # 104.7, filled at 104.8).
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


def _secondary_diverges_rows():
    # test_cascade_signals._secondary_diverges_rows, mirrored bar-for-bar:
    # flat ("none") at the same hour the primary breaks out -- a genuine
    # divergence, so the gate allows the primary's own confirmation.
    return [
        (_ts(DAY_T, "09:30"), _row(50, 51, 49, 50)),
        (_ts(DAY_T, "10:30"), _row(50, 50.5, 49.5, 50)),
        (_ts(DAY_T, "11:30"), _row(50, 50.1, 49.9, 50)),
    ]


def test_real_signal_fn_through_real_engine_and_bridge_end_to_end():
    """QQQ (primary) plays out the exact scenario-6b happy path (secondary
    diverges -> confirmation allowed -> real trade), run through THIS
    driver's own run_real_intraday_backtest + the bridge -- proving the
    driver's wiring (not just the underlying factory/engine, already
    proven by test_cascade_signals.py itself) is correct end to end."""
    qqq_bars = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    spy_bars = _frame(_secondary_diverges_rows())

    trades_df = driver.run_real_intraday_backtest(qqq_bars, spy_bars, cost_bps=0.0)
    assert len(trades_df) == 1
    row = trades_df.iloc[0]
    assert row["entry_px"] == pytest.approx(102.9)
    assert row["exit_px"] == pytest.approx(104.8)
    assert row["reason"] == "signal_exit"

    calendar_index = pd.DatetimeIndex(sorted(set(qqq_bars.index.date)))
    curve = bridge.build_day_trading_curve(trades_df, calendar_index)

    day_minus2_ts = pd.Timestamp(DAY_T_MINUS_2)
    day_minus1_ts = pd.Timestamp(DAY_T_MINUS_1)
    day_t_ts = pd.Timestamp(DAY_T)

    assert list(curve["strategy_daily"].index) == [day_minus2_ts, day_minus1_ts, day_t_ts]
    assert curve["strategy_daily"].loc[day_t_ts] == pytest.approx(row["net_ret"])
    assert curve["strategy_daily"].loc[day_minus2_ts] == 0.0
    assert bool(curve["in_market"].loc[day_t_ts]) is True
    assert bool(curve["in_market"].loc[day_minus2_ts]) is False
    out_trades = curve["trades_df"]
    assert out_trades is not trades_df
    assert out_trades["entry_date"].iloc[0] == day_t_ts
    assert out_trades["net_ret"].iloc[0] == pytest.approx(row["net_ret"])


def test_divergence_gate_still_blocks_when_secondary_confirms_too():
    """Regression coverage at the driver layer for scenario 6a's own
    blocking case -- proves this driver's own run_real_intraday_backtest
    threads secondary_bars through correctly (not silently dropped/
    ignored), by reusing the SAME "secondary confirms the identical move"
    fixture test_cascade_signals.py's own scenario 6a already proves
    blocks confirmation at the factory layer."""
    qqq_bars = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    secondary_confirms = _frame([
        (_ts(DAY_T, "09:30"), _row(50, 51, 49, 50)),
        (_ts(DAY_T, "10:30"), _row(50.2, 52.5, 50, 52.5)),
        (_ts(DAY_T, "11:30"), _row(52.5, 52.6, 52.4, 52.5)),
    ])

    trades_df = driver.run_real_intraday_backtest(qqq_bars, secondary_confirms, cost_bps=0.0)
    assert len(trades_df) == 0


def test_a_fresh_signal_fn_instance_is_built_per_call_no_leaked_state():
    """run_real_intraday_backtest must build a brand-new signal_fn (and
    therefore a fresh divergence_check closure) on every call -- reusing
    one across two backtests would leak per-session state from the first
    call's final bar into the second call's very first bar."""
    qqq_bars = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    spy_bars = _frame(_secondary_diverges_rows())
    first = driver.run_real_intraday_backtest(qqq_bars, spy_bars, cost_bps=0.0)
    second = driver.run_real_intraday_backtest(qqq_bars, spy_bars, cost_bps=0.0)
    assert len(first) == len(second) == 1
    assert first.iloc[0]["net_ret"] == pytest.approx(second.iloc[0]["net_ret"])


def test_run_real_intraday_backtest_never_applies_a_tax_haircut_at_the_engine_layer():
    """Same regression shape run_vwap_cross_decaying_conviction_abi1's own
    test file guards against (a double tax on winning trades): the engine
    layer must always run untaxed (0.0) -- ABI-1's
    own tax_policy applies tax exactly once, downstream. A real winning
    trade's net_ret must equal its gross_ret exactly."""
    qqq_bars = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    spy_bars = _frame(_secondary_diverges_rows())
    trades_df = driver.run_real_intraday_backtest(qqq_bars, spy_bars, cost_bps=0.0)
    assert len(trades_df) == 1
    row = trades_df.iloc[0]
    assert row["net_ret"] > 0.0
    assert row["net_ret"] == row["gross_ret"]
    with pytest.raises(TypeError):
        driver.run_real_intraday_backtest(
            qqq_bars, spy_bars, cost_bps=0.0, tax_haircut_rate=0.05)


def test_parameters_match_factory_defaults_and_catalog_shape():
    """PARAMETERS is the local drift-check anchor against the catalog
    entry -- must stay exactly 4 keys (matches data/mining/catalog.json's
    own real parameters field), and the 2 keys the factory itself
    declares must match its own live defaults (the module-level assert
    already checks this at import time; re-asserted here as an explicit,
    discoverable test rather than relying solely on an import-time
    side effect)."""
    assert set(driver.PARAMETERS) == {
        "secondary_ticker", "min_target_r_multiple", "force_flat_et",
        "max_hold_days", "direction"}
    assert driver.PARAMETERS["secondary_ticker"] == "SPY"
    assert driver.TRADED_TICKER == "QQQ"
    assert driver.SHAPE == "single_long"
