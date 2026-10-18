import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import pooling


def _trades_df(rows):
    """rows: list of (entry_date_str, exit_date_str, net_ret). Builds a
    trades_df with the same columns backtest_engine.run_backtest
    produces (only the columns pooling.py actually reads are populated
    with real values; the rest are filler)."""
    if not rows:
        return pd.DataFrame({
            "entry_date": pd.Series(dtype="datetime64[ns]"),
            "exit_date": pd.Series(dtype="datetime64[ns]"),
            "entry_px": pd.Series(dtype="float64"),
            "exit_px": pd.Series(dtype="float64"),
            "held": pd.Series(dtype="int64"),
            "reason": pd.Series(dtype="object"),
            "gross_ret": pd.Series(dtype="float64"),
            "net_ret": pd.Series(dtype="float64"),
        })
    return pd.DataFrame({
        "entry_date": [pd.Timestamp(r[0]) for r in rows],
        "exit_date": [pd.Timestamp(r[1]) for r in rows],
        "entry_px": [100.0] * len(rows),
        "exit_px": [100.0] * len(rows),
        "held": [5] * len(rows),
        "reason": ["signal_exit"] * len(rows),
        "gross_ret": [r[2] for r in rows],
        "net_ret": [r[2] for r in rows],
    })


# ---------------------------------------------------------------------------
# pool_trades
# ---------------------------------------------------------------------------

def test_pool_trades_concatenates_and_tags_ticker():
    by_ticker = {
        "AAPL": _trades_df([("2020-01-02", "2020-01-10", 0.01)]),
        "MSFT": _trades_df([("2020-01-05", "2020-01-12", -0.02)]),
    }
    pooled = pooling.pool_trades(by_ticker)
    assert len(pooled) == 2
    assert set(pooled["ticker"]) == {"AAPL", "MSFT"}
    assert "net_ret" in pooled.columns


def test_pool_trades_sorts_chronologically_by_entry_date_then_ticker():
    by_ticker = {
        "MSFT": _trades_df([("2020-01-05", "2020-01-12", -0.02)]),
        "AAPL": _trades_df([("2020-01-02", "2020-01-10", 0.01)]),
        "NVDA": _trades_df([("2020-01-02", "2020-01-11", 0.03)]),  # tie w/ AAPL on entry_date
    }
    pooled = pooling.pool_trades(by_ticker)
    assert list(pooled["ticker"]) == ["AAPL", "NVDA", "MSFT"]


def test_pool_trades_drops_zero_trade_tickers_without_error():
    by_ticker = {
        "AAPL": _trades_df([("2020-01-02", "2020-01-10", 0.01)]),
        "EMPTY": _trades_df([]),
    }
    pooled = pooling.pool_trades(by_ticker)
    assert len(pooled) == 1
    assert list(pooled["ticker"]) == ["AAPL"]


def test_pool_trades_rejects_empty_input_dict():
    with pytest.raises(ValueError):
        pooling.pool_trades({})


def test_pool_trades_all_tickers_zero_trades_returns_empty_not_error():
    # A dict with real tickers but zero trades everywhere is a legitimate
    # (if disappointing) outcome, not a caller bug -- distinct from an
    # empty dict, which IS a caller bug.
    pooled = pooling.pool_trades({"AAPL": _trades_df([]), "MSFT": _trades_df([])})
    assert len(pooled) == 0
    assert "ticker" in pooled.columns


# ---------------------------------------------------------------------------
# pooled_trade_stats -- must match the single-candidate backtest-stage
# formulas (net_edge_bps, PER-TRADE sharpe, skew/kurtosis both bias=True)
# exactly, just evaluated over the pooled set.
# ---------------------------------------------------------------------------

def test_pooled_trade_stats_basic_values():
    pooled = pooling.pool_trades({
        "A": _trades_df([("2020-01-02", "2020-01-10", 0.02)]),
        "B": _trades_df([("2020-01-03", "2020-01-11", 0.04)]),
        "C": _trades_df([("2020-01-04", "2020-01-12", -0.01)]),
    })
    stats = pooling.pooled_trade_stats(pooled)
    assert stats["trade_count"] == 3
    assert stats["net_edge_bps"] == pytest.approx(pooled["net_ret"].mean() * 1e4)
    assert stats["sharpe"] == pytest.approx(
        pooled["net_ret"].mean() / pooled["net_ret"].std(ddof=1))
    import scipy.stats
    assert stats["skew"] == pytest.approx(
        scipy.stats.skew(pooled["net_ret"], bias=True))
    assert stats["kurtosis"] == pytest.approx(
        scipy.stats.kurtosis(pooled["net_ret"], fisher=False, bias=True))
    assert stats["ticker_count"] == 3


def test_pooled_trade_stats_fallback_below_two_trades():
    pooled = pooling.pool_trades({"A": _trades_df([("2020-01-02", "2020-01-10", 0.02)])})
    stats = pooling.pooled_trade_stats(pooled)
    assert stats["trade_count"] == 1
    assert stats["sharpe"] == 0.0
    assert stats["skew"] == 0.0
    assert stats["kurtosis"] == 3.0


def test_pooled_trade_stats_zero_trades():
    pooled = pooling.pool_trades({"A": _trades_df([]), "B": _trades_df([])})
    stats = pooling.pooled_trade_stats(pooled)
    assert stats["trade_count"] == 0
    assert stats["net_edge_bps"] == 0.0
    assert stats["sharpe"] == 0.0
    assert stats["skew"] == 0.0
    assert stats["kurtosis"] == 3.0
    assert stats["ticker_count"] == 0


# ---------------------------------------------------------------------------
# pooled_daily_returns_by_exit_day -- correlation-purpose series only.
# ---------------------------------------------------------------------------

def test_pooled_daily_returns_sums_overlapping_exit_days_across_tickers():
    idx_a = pd.date_range("2020-01-01", periods=10, freq="B")
    idx_b = pd.date_range("2020-01-06", periods=10, freq="B")  # overlaps idx_a's tail
    trades_a = _trades_df([("2020-01-01", idx_a[5].strftime("%Y-%m-%d"), 0.01)])
    trades_b = _trades_df([("2020-01-06", idx_a[5].strftime("%Y-%m-%d"), 0.02)])  # same exit day as A
    per_ticker = {"A": (trades_a, idx_a), "B": (trades_b, idx_b)}
    combined = pooling.pooled_daily_returns_by_exit_day(per_ticker)
    # Union calendar covers every date either ticker's OWN index has.
    assert set(idx_a).issubset(set(combined.index))
    assert set(idx_b).issubset(set(combined.index))
    # Both trades exit on the same calendar day -> summed.
    assert combined.loc[idx_a[5]] == pytest.approx(0.03)
    # A day with no exits anywhere is zero, not missing.
    other_day = idx_a[1]
    if other_day != idx_a[5]:
        assert combined.loc[other_day] == pytest.approx(0.0)


def test_pooled_daily_returns_window_days_takes_the_tail():
    idx_a = pd.date_range("2020-01-01", periods=20, freq="B")
    trades_a = _trades_df([("2020-01-01", idx_a[15].strftime("%Y-%m-%d"), 0.01)])
    per_ticker = {"A": (trades_a, idx_a)}
    full = pooling.pooled_daily_returns_by_exit_day(per_ticker)
    windowed = pooling.pooled_daily_returns_by_exit_day(per_ticker, window_days=5)
    assert len(windowed) == 5
    assert list(windowed.index) == list(full.index[-5:])


def test_pooled_daily_returns_rejects_empty_input_dict():
    with pytest.raises(ValueError):
        pooling.pooled_daily_returns_by_exit_day({})
