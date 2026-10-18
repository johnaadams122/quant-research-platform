"""Cross-ticker trade pooling for expanded-universe mining: a catalog
candidate whose signal is derived purely from one ticker's own
price/earnings history is run on every ticker of a broad universe (e.g. the
S&P 500 and NASDAQ 100) and scored as ONE pooled hypothesis.

METHODOLOGY (why trade-level pooling, not daily-return pooling or a
plain average of per-ticker summary statistics):

This pipeline's statistical apparatus (mining.dsr's PSR/DSR, and the
mining pipeline's own backtest-phase formulas for net_edge_bps/sharpe/
skew/kurtosis) is built to consume a SAMPLE OF TRADE RETURNS -- see
backtest_engine.run_backtest's trades_df.net_ret and dsr.py's own
docstring. A single-ticker candidate's evidence IS its own trades_df.
The most direct, least-invented way to test "does this mechanism
generalize across the real market" as ONE hypothesis (not hundreds) is
to concatenate every ticker's realized trades into ONE combined sample
and feed THAT to the exact same formulas already used for a single
ticker -- extending the sample from one ticker's few dozen trades to (with
a broad universe) plausibly thousands, which is exactly the statistical
power gain pooling is for. No new statistical machinery is needed; the
DSR/skew/kurtosis inputs are computed identically to how a single-ticker
candidate already computes them, just over a larger pooled set.

Two alternatives were considered and rejected:

1. Average each ticker's OWN daily-return series across the whole
   universe into one cross-sectional daily portfolio return (closer to
   the literature's actual decile-portfolio construction). Rejected for
   THIS pipeline's purpose: on the vast majority of trading days, only a
   small fraction of ~500 tickers are in a live position under either
   signal (both are discrete-trigger mechanisms -- a monthly nearness-to-
   high rebalance, or a real earnings date -- not an always-invested
   exposure), so a naive per-day average across the FULL universe would
   dilute genuine trade P&L by roughly the ticker count on most days,
   collapsing the measured effect toward zero regardless of whether the
   underlying per-trade edge is real. That would silently answer a
   different question ("how does an always-in-the-market average of this
   universe perform") than the one being asked ("does this mechanism's
   edge hold up over many real instruments").

2. Average each ticker's own summary Sharpe/edge_bps (one number per
   ticker, then average across tickers). Rejected: gives a ticker with 2
   trades the same weight as one with 60, throws away the trade-level
   distribution DSR's skew/kurtosis inputs need, and shrinks the
   effective sample size back down to "number of tickers" rather than
   "number of trades" -- the opposite of what a larger, more
   representative test is supposed to buy.

The SEPARATE, narrower `pooled_daily_returns_by_exit_day` below exists
only for the DSR module's cross-candidate correlation estimate
(dsr.num_independent_trials, via the SAME per-run "shared correction N"
procedure the mining pipeline's scoring phase already uses across multiple
candidates tested in one run) -- it plays no role in the trade-level
DSR/skew/kurtosis inputs above, and pooling that series by SUMMING across
tickers (not averaging) is the natural analogue of a single candidate's
own daily_returns_by_exit_day output: "this family's aggregate realized
P&L, across every ticker it touched, on each calendar day."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats

from .backtest_engine import daily_returns_by_exit_day

_TRADES_COLUMNS = ["entry_date", "exit_date", "entry_px", "exit_px",
                   "held", "reason", "gross_ret", "net_ret"]


def pool_trades(trades_by_ticker: dict) -> pd.DataFrame:
    """Concatenate every ticker's trades_df (backtest_engine.run_backtest
    output shape) into one pooled DataFrame tagged with a `ticker`
    column, sorted chronologically by entry_date (ticker name as a
    deterministic tie-break for same-day entries). A ticker that
    produced zero trades contributes nothing (its empty frame
    concatenates to nothing) -- not an error, since "this ticker never
    triggered the signal" is a legitimate real outcome. Raises
    ValueError only if `trades_by_ticker` itself is empty (a caller
    passing no tickers at all is a bug, distinct from tickers that ran
    and produced zero trades)."""
    if not trades_by_ticker:
        raise ValueError("pool_trades: trades_by_ticker is empty -- no "
                          "tickers were passed in at all")

    frames = []
    for ticker, trades_df in trades_by_ticker.items():
        if len(trades_df) == 0:
            continue
        tagged = trades_df.copy()
        tagged["ticker"] = ticker
        frames.append(tagged)

    if not frames:
        empty = pd.DataFrame({col: pd.Series(dtype=trades_df.dtypes.get(col, "object"))
                               for col in _TRADES_COLUMNS})
        empty["ticker"] = pd.Series(dtype="object")
        return empty

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(["entry_date", "ticker"], kind="stable").reset_index(drop=True)
    return pooled


def pooled_trade_stats(pooled_trades_df: pd.DataFrame) -> dict:
    """net_edge_bps/sharpe/skew/kurtosis over a pooled trades DataFrame
    (from pool_trades), using the EXACT SAME formulas the mining
    pipeline's backtest phase uses per-candidate -- just evaluated
    over the pooled set instead of one ticker's own
    trades. `sharpe` is PER-TRADE, not annualized, matching that
    convention. Same <2-trade fallback (sharpe=0.0, skew=0.0,
    kurtosis=3.0 -- the Pearson-convention Normal value) as the
    per-candidate formulas, so a degenerate pool never NaNs downstream."""
    n = len(pooled_trades_df)
    ticker_count = pooled_trades_df["ticker"].nunique() if n else 0
    if n == 0:
        return {"trade_count": 0, "net_edge_bps": 0.0, "sharpe": 0.0,
                "skew": 0.0, "kurtosis": 3.0, "ticker_count": ticker_count}

    net_ret = pooled_trades_df["net_ret"]
    net_edge_bps = float(net_ret.mean() * 1e4)
    # Zero-variance guard: std==0 (every
    # pooled trade returning the identical value -- near-impossible with real
    # ticker diversity but not structurally impossible) would otherwise divide
    # to inf/NaN here and propagate into dsr.probabilistic_sharpe_ratio's
    # sqrt(1 - skew*sr + ...), which raises an unhandled ValueError on a
    # negative radicand. dsr.py guards this class of input everywhere else
    # (num_independent_trials, expected_max_sharpe_ratio); this module should
    # too. Same <2-trade fallback values as the degenerate-pool branch below.
    std = float(net_ret.std(ddof=1)) if n >= 2 else 0.0
    if n >= 2 and std > 0:
        sharpe = float(net_ret.mean() / std)
        skew = float(scipy.stats.skew(net_ret, bias=True))
        kurtosis = float(scipy.stats.kurtosis(net_ret, fisher=False, bias=True))
    else:
        sharpe = 0.0
        skew = 0.0
        kurtosis = 3.0

    return {"trade_count": n, "net_edge_bps": net_edge_bps, "sharpe": sharpe,
            "skew": skew, "kurtosis": kurtosis, "ticker_count": ticker_count}


def pooled_daily_returns_by_exit_day(per_ticker: dict, window_days: int = None) -> pd.Series:
    """`per_ticker`: {ticker: (trades_df, ohlc_index)} -- each ticker's
    OWN OHLC DatetimeIndex, required because
    backtest_engine.daily_returns_by_exit_day validates every trade's
    exit_date against the index that produced it (see that function's
    own docstring); there is no single shared index across ~500
    different tickers' different listing histories/trading calendars.

    For each ticker, builds its own daily_returns_by_exit_day series,
    then aligns every ticker's series onto the UNION of all tickers' OHLC
    calendars (zero-filled where a ticker has no data or no exit that
    day) and SUMS across tickers -- see this module's docstring for why
    summing (not averaging) is the right aggregation for this
    correlation-purpose series specifically. If `window_days` is given,
    returns only the trailing `window_days` rows of the combined series
    (mirrors mining.constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS
    usage elsewhere in this pipeline). Raises ValueError if `per_ticker`
    is empty."""
    if not per_ticker:
        raise ValueError("pooled_daily_returns_by_exit_day: per_ticker is "
                          "empty -- no tickers were passed in at all")

    per_ticker_series = []
    union_index = None
    for ticker, (trades_df, ohlc_index) in per_ticker.items():
        s = daily_returns_by_exit_day(ohlc_index, trades_df)
        per_ticker_series.append(s)
        union_index = ohlc_index if union_index is None else union_index.union(ohlc_index)

    union_index = union_index.sort_values()
    combined = pd.Series(0.0, index=union_index)
    for s in per_ticker_series:
        combined = combined.add(s.reindex(union_index, fill_value=0.0), fill_value=0.0)

    if window_days is not None:
        combined = combined.tail(window_days)
    return combined
