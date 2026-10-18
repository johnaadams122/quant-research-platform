"""Benchmark-relative and daily-frequency metrics for the strategy-mining
pipeline. REPORT-ONLY analytics.

Motivation
----------
`backtest_engine.run_backtest` returns per-TRADE returns. The pipeline's
pre-registered score is a per-trade Sharpe with n = trade count, and its
holdout test (`backtest_engine.holdout_passes`) is a bare sign check with
no benchmark. That combination alone can let a long-only family pass on
pure index drift. Directional strategies stay admissible, so every
candidate is also reported with, ALONGSIDE the pre-registered score:

  (a) alpha/beta versus buy-and-hold of its own instrument and versus SPY
      (`ols_alpha_beta`);
  (b) time-matched benchmark excess per trade
      (`time_matched_benchmark_excess`);
  (c) a full-window risk-adjusted comparison (`full_window_summary`);
  (d) the same DSR formula applied to a DAILY mark-to-market return series
      -- the standard construction in Bailey & Lopez de Prado (2014),
      where n is the number of return OBSERVATIONS, not trades
      (`mark_to_market_daily_returns` -> `daily_sharpe_moments` ->
      `daily_frequency_dsr`).

Boundaries (deliberate, and pinned by a test)
---------------------------------------------
This module never imports `trial_ledger`, never writes a file, and never
reads the clock. It changes no recorded score; a caller decides what to do
with these numbers.

Engine conventions this module mirrors (see backtest_engine.run_backtest)
------------------------------------------------------------------------
* Entry fills at the NEXT bar's Open: `trades_df.entry_date` is the bar
  whose Open was the fill, and `in_market` is True from that bar through
  `exit_date` inclusive.
* Exit fills at `exit_date`'s Close.
* gross_ret = exit_px / entry_px - 1 for long, negated for short.
* net_ret = gross_ret - cost_bps / 1e4 (round trip charged once), then
  the tax haircut applies to a positive net_ret only.
* Positions never overlap.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.stats

from mining import dsr

TRADING_DAYS_PER_YEAR = 252
OLS_MIN_OBSERVATIONS = 30


# --------------------------------------------------------------- returns

def mark_to_market_daily_returns(
        df: pd.DataFrame, trades_df: pd.DataFrame, *,
        direction: Literal["long", "short"] = "long",
        cost_bps: float = 0.0) -> pd.Series:
    """Daily mark-to-market return series for a trade list, aligned 1:1
    with `df.index` (name "mtm"), 0.0 on every flat day.

    Per trade (entry bar e, exit bar x, e <= x, both in `df.index`):
      * on e:                 r_e = Close_e / Open_e - 1
      * on each d in (e, x):  r_d = Close_d / Close_{d-1} - 1
      * on x: use the recorded exit_px when present, otherwise Close_x.
        Rotation exits at the open; its later close is no longer owned.
      * on x additionally:    r_x -= (cost_bps / 1e4) / G, where G is the
        trade's compounded growth over sessions e..x-1 (G = 1 for a
        same-session trade). This is the engine's round-trip cost, charged
        ONCE, at exit, on ENTRY capital: the trade's compounded multiple
        drops by exactly cost_bps / 1e4, so for LONG trades
        prod_{d=e..x} (1 + r_d) = 1 + net_ret (the identity
        account_relative's realization tax relies on).
    `direction="short"` NEGATES every price-return term above; the cost
    term is still subtracted (a cost is a cost either way).

    Exactness versus the engine, stated honestly
    --------------------------------------------
    LONG: compounding the daily price terms over a trade reproduces the
    engine's gross_ret exactly (up to float rounding):
        prod_{d=e..x} (1 + r_d) = (Close_e/Open_e) * prod (Close_d/Close_{d-1})
                                = Close_x / Open_e = 1 + gross_ret.
    SHORT: the engine defines gross_ret as the SIMPLE one-share P&L on
    entry capital, `1 - exit_px / entry_px`, i.e. the negated long path
    return. Compounding the negated daily terms gives
    prod (1 - r_d), which is NOT `1 - prod(1 + r_d)` in general (they agree
    to first order and diverge at second order in the daily moves). So a
    short trade's compounded mtm series does not reproduce the engine's
    gross_ret bit-for-bit; the per-day terms are the correct daily P&L of
    a constant-one-share short, and the discrepancy is the engine's
    simple-return convention, not a defect here. Callers comparing short
    families to the engine's per-trade figures should expect a small,
    sign-indeterminate gap that grows with the size of the daily moves.

    Validation (ValueError): every entry/exit date must be in `df.index`
    (which must be unique); exit >= entry; trades must not overlap (a
    trade entering on or before a previous trade's exit bar is an
    overlap -- the engine never re-enters on the exit bar); `direction`
    must be "long" or "short".
    """
    if direction not in ("long", "short"):
        raise ValueError(
            f"direction must be 'long' or 'short', got {direction!r}")
    if not df.index.is_unique:
        raise ValueError("df.index must be unique")
    index = df.index
    out = np.zeros(len(index), dtype=float)
    result = pd.Series(out, index=index, name="mtm")
    if trades_df is None or len(trades_df) == 0:
        return result

    entry_pos = index.get_indexer(pd.DatetimeIndex(trades_df["entry_date"]))
    exit_pos = index.get_indexer(pd.DatetimeIndex(trades_df["exit_date"]))
    if (entry_pos < 0).any() or (exit_pos < 0).any():
        bad = trades_df.loc[(entry_pos < 0) | (exit_pos < 0),
                            ["entry_date", "exit_date"]]
        raise ValueError(
            f"trade dates not in df.index -- pass the same df that produced "
            f"trades_df. Offending rows:\n{bad}")
    if (exit_pos < entry_pos).any():
        raise ValueError("every trade must have exit_date >= entry_date")
    order = np.argsort(entry_pos, kind="stable")
    entry_pos, exit_pos = entry_pos[order], exit_pos[order]
    if len(entry_pos) > 1 and (entry_pos[1:] <= exit_pos[:-1]).any():
        raise ValueError(
            "trades overlap (a trade enters on or before a previous trade's "
            "exit bar); this engine never holds overlapping positions")

    open_ = df["Open"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    sign = 1.0 if direction == "long" else -1.0
    cost = float(cost_bps) / 1e4
    exits = (trades_df["exit_px"].to_numpy(dtype=float)[order]
             if "exit_px" in trades_df else close[exit_pos])
    for e, x, exit_px in zip(entry_pos, exit_pos, exits):
        out[e] = sign * (close[e] / open_[e] - 1.0)
        if x > e:
            out[e + 1:x + 1] = sign * (close[e + 1:x + 1] / close[e:x] - 1.0)
        out[x] = sign * (exit_px / (open_[e] if x == e else close[x - 1]) - 1.0)
        # The engine charges the cost on ENTRY capital (net = gross - cost).
        # By the exit session the trade's equity has compounded to `grown`
        # times entry capital, so the same dollar cost is cost / grown of the
        # exit session's opening equity. A flat `-= cost` would charge it on
        # grown equity and break the (1 + net_ret) identity the realization
        # tax in account_relative relies on.
        grown = float(np.prod(1.0 + out[e:x])) if x > e else 1.0
        out[x] -= cost / grown if grown > 0.0 else cost
    return pd.Series(out, index=index, name="mtm")


def buy_and_hold_daily_returns(df: pd.DataFrame) -> pd.Series:
    """Close-to-close daily return, `Close / Close.shift(1) - 1`, with the
    first bar set to 0.0 so the series aligns 1:1 with `df.index`."""
    close = df["Close"].to_numpy(dtype=float)
    out = np.zeros(len(close), dtype=float)
    if len(close) > 1:
        out[1:] = close[1:] / close[:-1] - 1.0
    return pd.Series(out, index=df.index, name="buy_and_hold")


def cost_adjusted_benchmark_daily_returns(
        df: pd.DataFrame, *, entry_cost_bps: float = 0.0) -> pd.Series:
    """Daily buy-and-hold benchmark return, net of the benchmark's OWN
    entry-side transaction cost only. No tax is ever applied.

    ABI-1's account-vs-
    benchmark comparison must be net-vs-net, not net-vs-gross -- the
    strategy side already carries measured cost and tax
    (backtest_engine.run_backtest), while the report-only panel's
    time_matched_benchmark_excess compares that against a raw benchmark
    price return (no cost), which flatters the strategy. That report-only
    panel keeps its raw-benchmark convention by design (the two
    conventions are deliberately not collapsed) -- this function exists for
    an ABI-1 call site only.

    JUDGMENT CALL, flagged as such (same convention as this module's
    other invented numbers): only HALF the
    round-trip spread is charged (the cost of crossing the spread ONCE to
    establish the position), not a full round trip, because a benchmark
    held for an account's entire scored life is never actually exited
    inside the window being measured -- charging a hypothetical future
    sale that has not happened would invent a cost as speculative as the
    untaxed gain that is likewise excluded. This is the CONSERVATIVE
    direction for a discovery pipeline: undercharging the benchmark makes
    it look BETTER, which makes it HARDER, not easier, for a candidate to
    out-score it -- the same direction the net-vs-net comparison itself moves,
    so this cannot manufacture a false pass. Callers pass
    `entry_cost_bps` already halved (e.g.
    cost_tax_inputs.compute_spread_bps(bid, ask) / 2.0) -- this function
    does not halve anything itself, to keep the halving decision visible
    at the call site rather than hidden in here.

    Charges the ENTIRE `entry_cost_bps` once, on the series' first bar
    (which buy_and_hold_daily_returns always sets to 0.0 before this
    charge, since there is no prior close to compare against) -- the
    position's one and only crossing of the spread. No tax anywhere: this
    convention models a benchmark that is never sold inside the scored
    window, so no gain is ever realized.
    """
    daily = buy_and_hold_daily_returns(df)
    if len(daily) == 0 or entry_cost_bps == 0.0:
        return daily
    out = daily.to_numpy(dtype=float).copy()
    out[0] -= float(entry_cost_bps) / 1e4
    return pd.Series(out, index=daily.index, name="benchmark_net_of_cost")


# -------------------------------------------------------------- summaries

def _values(daily) -> np.ndarray:
    return np.asarray(daily, dtype=float).ravel()


def sharpe_annualized(daily, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """mean / std(ddof=1) * sqrt(periods); 0.0 when n < 2 or std is 0."""
    x = _values(daily)
    if x.size < 2:
        return 0.0
    std = float(np.std(x, ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return 0.0
    return float(np.mean(x) / std * math.sqrt(periods))


def cagr(daily, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate of the compounded equity curve:
    (prod(1 + r)) ** (periods / n) - 1. 0.0 for an empty series; -1.0 if
    the equity curve is wiped out (terminal equity <= 0)."""
    x = _values(daily)
    if x.size == 0:
        return 0.0
    terminal = float(np.prod(1.0 + x))
    if terminal <= 0.0:
        return -1.0
    return terminal ** (periods / x.size) - 1.0


def max_drawdown(daily) -> float:
    """Maximum peak-to-trough drawdown of the compounded equity curve,
    as a non-positive fraction (0.0 = never below a prior peak)."""
    x = _values(daily)
    if x.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + x)
    # Starting equity (1.0) is itself a peak, so a series that only ever
    # falls still reports its drawdown from the start.
    peak = np.maximum(np.maximum.accumulate(equity), 1.0)
    dd = equity / peak - 1.0
    return float(min(0.0, dd.min()))


def ols_alpha_beta(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """OLS of strategy daily returns on benchmark daily returns:
    strategy = alpha + beta * benchmark + e. Inner-joins the two series on
    their index first (rows with a NaN on either side are dropped).

    Returns a dict: alpha_daily, alpha_annualized (= alpha_daily * 252,
    the simple per-period convention, not compounded), beta, t_alpha_ols
    (classical homoskedastic standard error, dof n - 2), t_alpha_hac
    (Newey-West / Bartlett-kernel HAC standard error with lag
    floor(4 * (n / 100) ** (2 / 9)), no small-sample correction -- the
    same default statsmodels applies), hac_lag (that lag, reported so a
    caller can see how much autocorrelation correction was applied),
    r2, n.

    Raises ValueError if fewer than OLS_MIN_OBSERVATIONS (30) joint
    observations remain or the benchmark has zero variance.

    Degenerate t-statistics: when the residuals are exactly zero (e.g. a
    strategy that is identically zero, or an exact multiple of the
    benchmark) the standard error is 0. Then t is 0.0 if alpha is exactly
    0.0, and +/-inf otherwise -- a perfectly fitted line is infinitely
    "significant", not undefined. r2 is reported as 0.0 when the
    strategy series is constant (SST == 0), since the regression explains
    nothing there.
    """
    joined = pd.concat([pd.Series(strategy, name="s"),
                        pd.Series(benchmark, name="b")],
                       axis=1, join="inner").dropna()
    n = int(len(joined))
    if n < OLS_MIN_OBSERVATIONS:
        raise ValueError(
            f"ols_alpha_beta needs at least {OLS_MIN_OBSERVATIONS} joint "
            f"observations, got {n}")
    s = joined["s"].to_numpy(dtype=float)
    b = joined["b"].to_numpy(dtype=float)
    b_mean, s_mean = float(np.mean(b)), float(np.mean(s))
    b_dev = b - b_mean
    sxx = float(np.dot(b_dev, b_dev))
    if sxx <= 0.0 or not math.isfinite(sxx):
        raise ValueError("ols_alpha_beta: benchmark variance is 0")
    beta = float(np.dot(b_dev, s - s_mean) / sxx)
    alpha = s_mean - beta * b_mean
    resid = s - (alpha + beta * b)
    sse = float(np.dot(resid, resid))
    sst = float(np.dot(s - s_mean, s - s_mean))
    r2 = 1.0 - sse / sst if sst > 0.0 else 0.0

    # Classical OLS: var(alpha_hat) = sigma^2 * sum(b^2) / (n * Sxx).
    sigma2 = sse / (n - 2)
    var_alpha_ols = sigma2 * float(np.dot(b, b)) / (n * sxx)

    # Newey-West HAC: V = (X'X)^-1 S (X'X)^-1, Bartlett weights.
    lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    X = np.column_stack([np.ones(n), b])
    xtx_inv = np.linalg.inv(X.T @ X)
    xe = X * resid[:, None]
    S = xe.T @ xe
    for ell in range(1, lag + 1):
        w = 1.0 - ell / (lag + 1.0)
        gamma = xe[ell:].T @ xe[:-ell]
        S += w * (gamma + gamma.T)
    var_alpha_hac = float((xtx_inv @ S @ xtx_inv)[0, 0])

    def _t(alpha_hat: float, var: float) -> float:
        if var > 0.0 and math.isfinite(var):
            return float(alpha_hat / math.sqrt(var))
        if alpha_hat == 0.0:
            return 0.0
        return math.copysign(math.inf, alpha_hat)

    return {
        "alpha_daily": float(alpha),
        "alpha_annualized": float(alpha) * TRADING_DAYS_PER_YEAR,
        "beta": beta,
        "t_alpha_ols": _t(alpha, var_alpha_ols),
        "t_alpha_hac": _t(alpha, var_alpha_hac),
        "hac_lag": lag,
        "r2": float(r2),
        "n": n,
    }


def time_matched_benchmark_excess(trades_df: pd.DataFrame,
                                  bench_df: pd.DataFrame) -> pd.Series:
    """Per-trade excess over the benchmark held for the SAME days:
    net_ret - (bench Close at exit_date / bench Open at entry_date - 1).

    ALWAYS subtracts the benchmark's LONG return regardless of the
    strategy's direction: the question is "did this trade beat simply
    holding the benchmark over the same days", and that comparison is the
    same one whether the strategy was long or short. Returns a float
    Series indexed like `trades_df` (name "excess"). Raises ValueError if
    any entry/exit date is missing from `bench_df.index`.
    """
    result = pd.Series(np.zeros(len(trades_df), dtype=float),
                       index=trades_df.index, name="excess")
    if len(trades_df) == 0:
        return result
    if not bench_df.index.is_unique:
        raise ValueError("bench_df.index must be unique")
    entry_pos = bench_df.index.get_indexer(
        pd.DatetimeIndex(trades_df["entry_date"]))
    exit_pos = bench_df.index.get_indexer(
        pd.DatetimeIndex(trades_df["exit_date"]))
    missing = (entry_pos < 0) | (exit_pos < 0)
    if missing.any():
        raise ValueError(
            f"trade dates missing from the benchmark frame:\n"
            f"{trades_df.loc[missing, ['entry_date', 'exit_date']]}")
    bench_open = bench_df["Open"].to_numpy(dtype=float)[entry_pos]
    bench_close = bench_df["Close"].to_numpy(dtype=float)[exit_pos]
    bench_ret = bench_close / bench_open - 1.0
    net = trades_df["net_ret"].to_numpy(dtype=float)
    return pd.Series(net - bench_ret, index=trades_df.index, name="excess")


# --------------------------------------------------------- daily-freq DSR

def daily_sharpe_moments(daily) -> dict:
    """Per-period Sharpe and plug-in moments of a daily return series, in
    the series' own frequency (NOT annualized -- dsr.py requires that).

    Mirrors scripts/run_batch3_mining._trade_stats exactly: n < 2 or a
    zero/non-finite std gives sr 0.0, skew 0.0, kurtosis 3.0; otherwise
    skew = scipy.stats.skew(x, bias=True), kurtosis =
    scipy.stats.kurtosis(x, fisher=False, bias=True) (Pearson, plug-in),
    non-finite moments -> (0.0, 3.0), kurtosis floored at 1 + skew**2 so
    the pair always satisfies dsr's consistency guard. Raises ValueError
    on a non-finite input value.
    """
    x = _values(daily)
    n = int(x.size)
    if n == 0:
        return {"sr": 0.0, "n": 0, "skew": 0.0, "kurtosis": 3.0}
    if not np.isfinite(x).all():
        raise ValueError("daily returns must be finite")
    std = float(np.std(x, ddof=1)) if n >= 2 else 0.0
    if n < 2 or std <= 0.0 or not math.isfinite(std):
        return {"sr": 0.0, "n": n, "skew": 0.0, "kurtosis": 3.0}
    skew = float(scipy.stats.skew(x, bias=True))
    kurtosis = float(scipy.stats.kurtosis(x, fisher=False, bias=True))
    if not math.isfinite(skew) or not math.isfinite(kurtosis):
        skew, kurtosis = 0.0, 3.0
    kurtosis = max(kurtosis, 1.0 + skew ** 2)
    return {"sr": float(np.mean(x) / std), "n": n, "skew": skew,
            "kurtosis": kurtosis}


def daily_frequency_dsr(*, sr: float, n: int, skew: float, kurtosis: float,
                        n_trials: float, sr_variance: float) -> float:
    """Deflated Sharpe ratio of a DAILY return series -- a thin wrapper
    over `dsr.deflated_sharpe_ratio` so the daily construction (Bailey &
    Lopez de Prado 2014, n = number of return observations) uses the
    identical formula and guards as the pipeline's per-trade score. All
    inputs must be in daily units (sr and sr_variance per day)."""
    return dsr.deflated_sharpe_ratio(sr=sr, n=n, skew=skew,
                                     kurtosis=kurtosis, n_trials=n_trials,
                                     sr_variance=sr_variance)


# ------------------------------------------------------------ full window

def full_window_summary(strategy_daily: pd.Series,
                        benchmark_daily: pd.Series, *,
                        in_market: Optional[pd.Series] = None) -> dict:
    """Full-window risk-adjusted comparison of a strategy's daily series
    against a benchmark's, over their common index (inner join; NaNs
    dropped). Keys: strategy_sharpe_ann, benchmark_sharpe_ann,
    strategy_cagr, benchmark_cagr, strategy_max_dd, benchmark_max_dd,
    n_days, time_in_market (fraction of joint days with `in_market` True
    when given, else the fraction with a non-zero strategy return), and
    `ols` = the `ols_alpha_beta` dict -- or None plus an `ols_error`
    string when the regression cannot be run (too few days, flat
    benchmark)."""
    joined = pd.concat([pd.Series(strategy_daily, name="s"),
                        pd.Series(benchmark_daily, name="b")],
                       axis=1, join="inner").dropna()
    s, b = joined["s"], joined["b"]
    n_days = int(len(joined))
    if in_market is not None:
        # reindex with fill_value keeps the bool dtype (a reindex-then-
        # fillna round trip goes through object dtype and trips pandas'
        # silent-downcasting deprecation).
        mask = pd.Series(in_market).reindex(joined.index, fill_value=False)
        time_in_market = float(mask.astype(bool).mean()) if n_days else 0.0
    else:
        time_in_market = float((s != 0.0).mean()) if n_days else 0.0
    summary = {
        "strategy_sharpe_ann": sharpe_annualized(s),
        "benchmark_sharpe_ann": sharpe_annualized(b),
        "strategy_cagr": cagr(s),
        "benchmark_cagr": cagr(b),
        "strategy_max_dd": max_drawdown(s),
        "benchmark_max_dd": max_drawdown(b),
        "n_days": n_days,
        "time_in_market": time_in_market,
        "ols": None,
    }
    try:
        summary["ols"] = ols_alpha_beta(s, b)
    except ValueError as exc:
        summary["ols_error"] = str(exc)
    return summary
