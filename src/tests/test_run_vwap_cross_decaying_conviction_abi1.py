"""Integration test for scripts/run_vwap_cross_decaying_conviction_abi1.py --
proves the FULL composition (real signal_fn -> real intraday_backtest_engine
-> mining.day_trading_abi1_bridge) works end to end on a small,
synthetic, hand-constructed 2-day minute-bar frame, independent of any
network call (no real Alpaca fetch, no real SPY fetch, no ledger write).

Day 1 mirrors test_day_trading_vwap_decaying_conviction.py's own
test_scenario_a_fast_favorable_move_exits_close_to_base_target bar-for-bar
(a VWAP-cross entry that exits close to the base target the same session)
-- reused rather than re-derived, so this test's own expected values are
anchored to an existing, independently tested scenario. Day 2 is deliberately flat and never crosses the session VWAP, so
it never arms an entry -- the "day with zero trades" calendar case.
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
import run_vwap_cross_decaying_conviction_abi1 as driver  # noqa: E402


def _minute_index(date_str: str, start="09:30", end="15:55") -> pd.DatetimeIndex:
    day = pd.Timestamp(date_str, tz="America/New_York")
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    start_ts = day.replace(hour=sh, minute=sm)
    end_ts = day.replace(hour=eh, minute=em)
    return pd.date_range(start_ts, end_ts, freq="1min")


def _flat_frame(date_str: str, start="09:30", end="15:55", price=100.0, vwap=100.0):
    idx = _minute_index(date_str, start=start, end=end)
    n = len(idx)
    return pd.DataFrame(
        {"Open": [price] * n, "High": [price] * n, "Low": [price] * n,
         "Close": [price] * n, "Volume": [100.0] * n, "VWAP": [vwap] * n},
        index=idx)


def _set_bar(df, ts_str, date_str, **cols):
    ts = pd.Timestamp(f"{date_str} {ts_str}", tz="America/New_York")
    for col, val in cols.items():
        df.loc[ts, col] = float(val)
    return df


def _two_day_frame():
    day1 = "2026-03-02"  # Monday, mirrors test_scenario_a_...
    day1_frame = _flat_frame(day1, start="09:30", end="09:40", price=99.0, vwap=100.0)
    _set_bar(day1_frame, "09:31", day1, Open=100.4, High=100.6, Low=100.3, Close=100.5)
    _set_bar(day1_frame, "09:32", day1, Open=100.5, High=100.7, Low=100.4, Close=100.6)
    _set_bar(day1_frame, "09:33", day1, Open=101.0, High=101.05, Low=100.95, Close=101.0)
    _set_bar(day1_frame, "09:34", day1, Open=102.0, High=102.20, Low=101.95, Close=102.10)

    day2 = "2026-03-03"  # Tuesday -- flat, always below session VWAP, never confirms.
    day2_frame = _flat_frame(day2, start="09:30", end="09:40", price=99.0, vwap=100.0)

    return pd.concat([day1_frame, day2_frame])


def test_real_signal_fn_through_real_engine_and_bridge_end_to_end():
    bars = _two_day_frame()

    trades_df = driver.run_real_intraday_backtest(bars, cost_bps=0.0)
    assert len(trades_df) == 1
    row = trades_df.iloc[0]
    assert row["entry_px"] == pytest.approx(101.0)
    assert row["reason"] == "signal_exit"
    assert row["net_ret"] == pytest.approx(0.01, abs=0.003)

    calendar_index = pd.DatetimeIndex(sorted(set(bars.index.date)))
    curve = bridge.build_day_trading_curve(trades_df, calendar_index)

    day1_ts = pd.Timestamp("2026-03-02")
    day2_ts = pd.Timestamp("2026-03-03")

    assert list(curve["strategy_daily"].index) == [day1_ts, day2_ts]
    # Day 1's strategy_daily must equal the real trade's own net_ret exactly
    # (single trade, single day, no cost/tax) -- the SAME value the trade's
    # own net_ret column already carries, proving the bridge did not
    # silently transform it.
    assert curve["strategy_daily"].loc[day1_ts] == pytest.approx(row["net_ret"])
    assert curve["strategy_daily"].loc[day2_ts] == 0.0
    assert bool(curve["in_market"].loc[day1_ts]) is True
    assert bool(curve["in_market"].loc[day2_ts]) is False
    # trades_df comes back as an augmented COPY (entry_date/exit_date added,
    # required downstream by account_relative.netted_tax_factors -- see
    # mining.day_trading_abi1_bridge's own docstring), not the same object.
    out_trades = curve["trades_df"]
    assert out_trades is not trades_df
    assert out_trades["entry_date"].iloc[0] == day1_ts
    assert out_trades["exit_date"].iloc[0] == day1_ts
    assert out_trades["net_ret"].iloc[0] == pytest.approx(row["net_ret"])


def test_a_fresh_signal_fn_instance_is_built_per_call_no_leaked_state():
    """run_real_intraday_backtest must build a brand-new signal_fn on every
    call -- reusing one across two separate backtests would leak its
    closure's per-session VWAP/streak/position state from the first call's
    final bar into the second call's very first bar."""
    bars = _two_day_frame()
    first = driver.run_real_intraday_backtest(bars, cost_bps=0.0)
    second = driver.run_real_intraday_backtest(bars, cost_bps=0.0)
    assert len(first) == len(second) == 1
    assert first.iloc[0]["net_ret"] == pytest.approx(second.iloc[0]["net_ret"])


def test_run_real_intraday_backtest_never_applies_a_tax_haircut_at_the_engine_layer():
    """The engine layer always runs untaxed: run_real_intraday_backtest has
    no tax_haircut_rate parameter (always 0.0 internally). If
    compute_and_record passed cost_tax_inputs.tax_haircut_rate_for(
    ACCOUNT_TYPE) (nonzero for "taxable") into
    run_intraday_backtest's own tax_haircut_rate kwarg, every winning
    trade would be taxed twice: once
    inside the engine's own _close_trade, and AGAIN by
    account_relative.netted_tax_factors (reached through
    abi1_driver.evaluate_family's tax_policy="realization" path), which
    expects trades_df["net_ret"] to still be PRE-tax -- exactly the
    hazard abi1_driver's own module docstring warns every curve builder
    against. This test pins it with a real winning trade: with
    cost_bps=0.0, net_ret must equal gross_ret EXACTLY -- a nonzero tax
    haircut at the engine layer would make a real winning trade's net_ret
    strictly less than its gross_ret."""
    bars = _two_day_frame()
    trades_df = driver.run_real_intraday_backtest(bars, cost_bps=0.0)
    assert len(trades_df) == 1
    row = trades_df.iloc[0]
    assert row["net_ret"] > 0.0  # a real winning trade, not a degenerate case
    assert row["net_ret"] == row["gross_ret"]
    # And run_real_intraday_backtest must not even ACCEPT the parameter --
    # a caller passing it gets a loud TypeError, not a silent double-tax
    # path.
    with pytest.raises(TypeError):
        driver.run_real_intraday_backtest(bars, cost_bps=0.0, tax_haircut_rate=0.05)
