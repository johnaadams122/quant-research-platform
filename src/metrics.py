"""
metrics.py
----------
Performance and risk metrics for the Leveraged ETF Backtest project.
All functions accept pd.Series with a DatetimeIndex.

Conventions:
  - Prices: positive, indexed by trading date
  - Returns: pct_change() of prices (arithmetic, not log)
  - CAGR, Sharpe, Sortino: annualized (TRADING_DAYS = 252)
  - Drawdown: expressed as a decimal, max_drawdown() returns a negative number
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Drawdown ───────────────────────────────────────────────────────────────────

def drawdown_series(prices: pd.Series) -> pd.Series:
    """Rolling drawdown from peak. All values <= 0. -0.50 means 50% below prior peak."""
    roll_max = prices.cummax()
    return (prices - roll_max) / roll_max


def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough loss as a negative decimal. E.g., -0.80 = 80% drawdown."""
    return float(drawdown_series(prices).min())


def max_drawdown_duration(prices: pd.Series) -> int:
    """
    Longest contiguous period (in trading days) spent below the previous peak.
    This is the worst 'underwater' stretch, including both the fall and the recovery.
    """
    dd = drawdown_series(prices)
    is_underwater = (dd < 0).astype(int)
    # Count contiguous runs via cumsum trick
    group = (is_underwater != is_underwater.shift()).cumsum()
    longest = is_underwater.groupby(group).sum().max()
    return int(longest)


def drawdown_periods(prices: pd.Series) -> list:
    """
    Return a list of dicts describing each complete drawdown episode:
      start     : first date below prior peak
      trough    : date of maximum loss
      end       : date of full recovery (NaT if never recovered)
      depth     : trough drawdown (negative decimal)
      duration  : total days from start to end (or to last data if unrecovered)
    """
    dd = drawdown_series(prices)
    is_dd = dd < 0
    periods = []
    i = 0
    vals = is_dd.values
    n = len(prices)

    while i < n:
        if vals[i]:
            j = i
            while j < n and vals[j]:
                j += 1
            sub_dd = dd.iloc[i:j]
            trough_loc = sub_dd.idxmin()
            depth = sub_dd.min()
            end_date = dd.index[j - 1] if j == n else dd.index[j]
            periods.append({
                "start":    dd.index[i],
                "trough":   trough_loc,
                "end":      end_date,
                "depth":    float(depth),
                "duration": j - i,
            })
            i = j
        else:
            i += 1
    return periods


# ── Return-based metrics ───────────────────────────────────────────────────────

def cagr(prices: pd.Series) -> float:
    """Compound annual growth rate (annualized by trading days)."""
    n = len(prices)
    return float((prices.iloc[-1] / prices.iloc[0]) ** (TRADING_DAYS / n) - 1)


def sharpe(returns: pd.Series, rf_annual: float = 0.025) -> float:
    """
    Annualized Sharpe ratio.
    rf_annual: constant annual risk-free rate as a decimal (default 2.5%).
    """
    rf_daily = rf_annual / TRADING_DAYS
    excess = returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_annual: float = 0.025) -> float:
    """
    Annualized Sortino ratio — penalises only downside deviation.
    rf_annual: constant annual risk-free rate as a decimal (default 2.5%).
    """
    rf_daily = rf_annual / TRADING_DAYS
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    dstd = downside.std()
    if dstd == 0 or np.isnan(dstd):
        return np.nan
    return float(excess.mean() / dstd * np.sqrt(TRADING_DAYS))


def calmar(prices: pd.Series) -> float:
    """Calmar ratio = CAGR / |max drawdown|. Higher is better."""
    mdd = max_drawdown(prices)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr(prices) / abs(mdd))


def profit_factor(returns: pd.Series) -> float:
    """
    Profit Factor = sum(positive returns) / sum(|negative returns|).
    > 1.0 required; > 1.5 is considered strong.
    """
    gains  = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


# ── Convenience aggregator ─────────────────────────────────────────────────────

def summary_metrics(prices: pd.Series, name: str = "",
                    rf_annual: float = 0.025) -> dict:
    """
    Compute all key metrics for a price series and return as a dict.
    Keys: name, cagr, max_dd, max_dd_dur, sharpe, sortino, calmar, profit_factor.
    """
    returns = prices.pct_change().dropna()
    return {
        "name":          name,
        "cagr":          cagr(prices),
        "max_dd":        max_drawdown(prices),
        "max_dd_dur":    max_drawdown_duration(prices),
        "sharpe":        sharpe(returns, rf_annual),
        "sortino":       sortino(returns, rf_annual),
        "calmar":        calmar(prices),
        "profit_factor": profit_factor(returns),
    }
