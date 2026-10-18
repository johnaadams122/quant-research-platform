"""The net-return identity between the daily mark-to-market path and the
engine's per-trade path, with a NON-ZERO round-trip cost.

`account_relative.tax_factor_by_exit_day` and `netted_tax_factors` both
rely on: over a trade, the daily terms compound to exactly (1 + net_ret),
where net_ret = gross_ret - cost_bps / 1e4 is the engine's figure on ENTRY
capital (backtest_engine.run_backtest). Charging the cost as a flat
subtraction from the exit session's return charges it on the equity the
trade has grown (or shrunk) to by then, which breaks the identity for any
multi-session trade. Synthetic data only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mining import account_relative as ar  # noqa: E402
from mining import backtest_engine  # noqa: E402
from mining import benchmark_relative as br  # noqa: E402
from mining.backtest_engine import SignalResult  # noqa: E402


def _trend_frame():
    index = pd.date_range("2024-01-02", periods=6, freq="B")
    close = np.array([100.0, 100.0, 110.0, 120.0, 130.0, 130.0])
    open_ = np.array([100.0, 100.0, 100.0, 120.0, 130.0, 130.0])
    return pd.DataFrame({"Open": open_, "High": close + 1.0,
                         "Low": np.minimum(open_, close) - 1.0,
                         "Close": close}, index=index)


def _enter_bar0_exit_bar4(frame, i):
    return SignalResult(entry=(i == 0), exit=(i == 4))


def _random_frame(n=160, seed=11):
    rng = np.random.default_rng(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.01, n))
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + 0.1,
                         "Low": np.minimum(open_, close) - 0.1,
                         "Close": close}, index=index)


def _always_entry(frame, i):
    return SignalResult(entry=True, exit=False)


def _per_trade_compounded(mtm, trades):
    out = []
    for _, row in trades.iterrows():
        span = mtm.loc[row["entry_date"]:row["exit_date"]].to_numpy()
        out.append(float(np.prod(1.0 + span)) - 1.0)
    return np.array(out)


def test_long_multiday_trade_compounds_to_engine_net_ret_with_cost():
    df = _trend_frame()
    cost_bps = 10.0
    trades, _ = backtest_engine.run_backtest(
        df, _enter_bar0_exit_bar4, cost_bps=cost_bps)
    assert len(trades) == 1 and trades["held"].iloc[0] > 0
    mtm = br.mark_to_market_daily_returns(df, trades, cost_bps=cost_bps)
    compounded = float(np.prod(1.0 + mtm.to_numpy())) - 1.0
    assert compounded == pytest.approx(trades["net_ret"].iloc[0],
                                       rel=0, abs=1e-12)


def test_long_identity_holds_for_every_trade_on_a_random_path():
    df = _random_frame()
    cost_bps = 25.0
    trades, _ = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=cost_bps, time_stop=7)
    assert (trades["held"] > 0).sum() >= 5
    mtm = br.mark_to_market_daily_returns(df, trades, cost_bps=cost_bps)
    np.testing.assert_allclose(_per_trade_compounded(mtm, trades),
                               trades["net_ret"].to_numpy(),
                               rtol=0, atol=1e-12)


def test_realization_tax_on_daily_path_matches_engine_taxed_return():
    df = _trend_frame()
    cost_bps = 10.0
    rate = 0.3
    pre, _ = backtest_engine.run_backtest(
        df, _enter_bar0_exit_bar4, cost_bps=cost_bps)
    post, _ = backtest_engine.run_backtest(
        df, _enter_bar0_exit_bar4, cost_bps=cost_bps, tax_haircut_rate=rate)
    mtm = br.mark_to_market_daily_returns(df, pre, cost_bps=cost_bps)
    factors = ar.tax_factor_by_exit_day(df.index, pre, tax_haircut_rate=rate)
    taxed = float(np.prod((1.0 + mtm.to_numpy()) * factors.to_numpy())) - 1.0
    assert taxed == pytest.approx(post["net_ret"].iloc[0], rel=0, abs=1e-12)


def test_short_cost_is_charged_on_entry_capital():
    # The short price path keeps its documented simple-vs-compounded gap,
    # but the COST must still reduce the trade's compounded multiple by
    # exactly cost_bps / 1e4 of entry capital, as in the engine.
    df = _random_frame(seed=5)
    cost_bps = 30.0
    trades, _ = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=0.0, time_stop=6, direction="short")
    free = br.mark_to_market_daily_returns(
        df, trades, direction="short", cost_bps=0.0)
    charged = br.mark_to_market_daily_returns(
        df, trades, direction="short", cost_bps=cost_bps)
    np.testing.assert_allclose(
        _per_trade_compounded(free, trades) - _per_trade_compounded(charged, trades),
        cost_bps / 1e4, rtol=0, atol=1e-12)


def test_same_session_trade_charges_the_full_cost_on_its_one_session():
    # held == 0: a flat subtraction on the one session is on entry capital.
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({"Open": [10.0, 10.0, 11.0], "High": [12.0] * 3,
                       "Low": [9.0] * 3, "Close": [10.0, 10.5, 11.0]},
                      index=index)
    trades = pd.DataFrame({"entry_date": [index[1]], "exit_date": [index[1]],
                           "exit_px": [10.5]})
    mtm = br.mark_to_market_daily_returns(df, trades, cost_bps=20.0)
    assert mtm.iloc[1] == pytest.approx(10.5 / 10.0 - 1.0 - 0.002,
                                        rel=0, abs=1e-15)
