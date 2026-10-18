"""Batch5 candidate: 12-month time-series momentum on currency ETFs
(family ``fx_dollar_index_trend_following``, shape ``pooled_single_asset``).

The catalog entry's mechanism names UUP alone; this module runs that same
rule independently on each of seven currency ETFs (UUP among them) and
pools the trades into one trial. PARAMETERS is copied verbatim from the
catalog entry.

Rule (the catalog mechanism applied per pool member; ``make_signal_fn``
is called once per pool member by the driver, each member judged ONLY on
its own history):

    At each month-end, compute the fund's own trailing 12-month total
    return; if positive, hold the fund long over the next month, if
    negative, move to cash. Re-check and rebalance every month-end. Run
    the identical rule independently on each currency ETF in the pool;
    pool the trades.

Source: Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", JFE
104(2): the 12-month look-back / 1-month hold specification, long-only
here (parameters.long_only = true).

Engine mapping (mining.backtest_engine): an ``entry`` flag on bar t fills
at bar t+1's open, an ``exit`` flag on bar t fills at bar t's close. So a
LONG decision at a month-end raises the entry flag on that month-end bar
(fill = the next session's open, the earliest the engine can buy) and a
CASH decision raises the exit flag on that same month-end bar (fill = that
month-end close, the source's month-end rebalance point). Both flags are
precomputed once per ticker in the factory from grouped / shifted
operations only, reindexed to the bucket and looked up by position -- the
per-bar signal function is two array reads and never touches the slice.

Interpretation choices
----------------------
1. "Month-end" is the last bar of each calendar month in the fund's OWN bar
   calendar (the repo's vendored NYSE calendar covers 2026-2028 only and
   cannot serve a 2007+ history). The frame's FINAL bar makes no decision:
   without a following bar the calendar cannot say whether it is a real
   month-end, and no fill could differ anyway (an entry there has no next
   open; an exit there equals the engine's own end_of_data close).
2. "Trailing 12-month total return" = month-end close of month m divided by
   the month-end close of calendar month m-12 (period arithmetic), minus
   one. If that anchor month has no bars in the frame there is no decision
   that month -- never an interpolation to a neighbouring month. Total
   return is the pipeline's Close as delivered (data_pipeline fetches
   ``auto_adjust=True``, i.e. distribution-adjusted).
3. "If positive ... if negative": a trailing return of EXACTLY zero is not
   positive, so it is CASH (parameters.signal = trailing_total_return_sign,
   long only when the sign is strictly positive).
4. "Move to cash" executes at the deciding month-end's close (the engine's
   pinned exit fill); "hold over the next month" enters at the next
   session's open (the engine's pinned entry fill). No stops, no time exit.
5. While the standing decision is LONG the entry flag stays raised (the
   "stay long" idiom the engine needs). Its only extra effect is that a
   bucket which OPENS inside a LONG month enters at its first fill
   opportunity -- the closest reading of "hold the fund over the month"
   given the engine cannot start a bucket already in a position.
6. The exit flag is raised ONLY on CASH month-end bars. Between month-ends
   the function is a no-op on the exit side, so the engine's own state
   persists exactly as the rule intends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "fx_dollar_index_trend_following"
SHAPE = "pooled_single_asset"
TRADED = ["UUP", "UDN", "FXE", "FXY", "FXB", "FXA", "FXC"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "UUP"
# Copied verbatim from the existing data/mining/catalog.json entry; the
# driver rejects any drift under json.dumps(sort_keys=True).
PARAMETERS = {
    "lookback_months": 12,
    "rebalance": "monthly",
    "signal": "trailing_total_return_sign",
    "long_only": True,
}
EXPECTED_MIN_TRADES = 60

LOOKBACK_MONTHS = int(PARAMETERS["lookback_months"])


def _tz_naive(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out


def month_end_targets(close: pd.Series, lookback_months: int) -> pd.Series:
    """Per-bar decision series aligned to ``close.index``: 1.0 = LONG, 0.0 =
    CASH on a month-end bar whose trailing ``lookback_months``-month return
    is decidable; NaN on every other bar (non-month-end, no anchor month,
    or the frame's final bar -- see docstring choice 1).

    Only grouped / shifted operations: M[m] is the close of the LAST bar in
    calendar month m and is placed on that bar's own date, so the value at
    date t never uses a bar after t.
    """
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    index = _tz_naive(close.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("close index must be sorted and unique")
    values = pd.to_numeric(pd.Series(close).to_numpy(), errors="coerce")
    if len(values) == 0:
        return pd.Series(dtype=float, index=index)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("close values must be finite and positive")

    periods = index.to_period("M")
    codes = periods.asi8
    is_month_end = np.zeros(len(codes), dtype=bool)
    is_month_end[:-1] = codes[:-1] != codes[1:]     # final bar stays False

    monthly = pd.Series(values[is_month_end], index=periods[is_month_end])
    anchor_periods = monthly.index - lookback_months   # PeriodIndex arithmetic
    anchors = monthly.reindex(anchor_periods).to_numpy(dtype=float)
    ret = monthly.to_numpy(dtype=float) / anchors - 1.0   # NaN where no anchor
    decision = np.where(np.isnan(ret), np.nan, (ret > 0).astype(float))

    targets = np.full(len(values), np.nan)
    targets[is_month_end] = decision
    return pd.Series(targets, index=index, name="target")


def signal_flags(frame: pd.DataFrame, lookback_months: int) -> pd.DataFrame:
    """Boolean ``entry`` / ``exit`` columns aligned to ``frame.index``.

    standing[t] = the most recent month-end decision at or before t
    (forward fill). entry[t] = standing[t] is LONG (fills at t+1 open).
    exit[t] = the decision made ON bar t is CASH (fills at t's close);
    only month-end bars carry a decision, so exits fire only there.
    """
    targets = month_end_targets(frame["Close"], lookback_months)
    standing = targets.ffill()
    entry = standing == 1.0
    exit_ = targets == 0.0
    return pd.DataFrame({"entry": entry.to_numpy(bool),
                         "exit": exit_.to_numpy(bool)}, index=targets.index)


def make_signal_fn(ticker: str, index, frames):
    """Driver factory (called once per pool member with that member as
    ``ticker``). ``index`` is the bucket calendar the engine will run over;
    ``frames[ticker]`` is that member's full-history OHLC frame. Everything
    is precomputed here; the returned closure is O(1) per bar."""
    frame = frames[ticker]                       # KeyError if absent: fail closed
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{ticker}: frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ValueError(f"{ticker}: frame is missing Close")
    flags = signal_flags(frame, LOOKBACK_MONTHS)
    bucket = _tz_naive(index)
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{ticker}: bucket index must be sorted and unique")
    missing = bucket.difference(flags.index)
    if len(missing):
        raise ValueError(
            f"{ticker}: bucket has {len(missing)} date(s) not in the frame "
            f"(first {missing[0].date()})")
    aligned = flags.reindex(bucket)
    entry_flags = aligned["entry"].to_numpy(dtype=bool)
    exit_flags = aligned["exit"].to_numpy(dtype=bool)
    n = len(bucket)

    def signal_fn(_frame_slice, i: int) -> SignalResult:
        if i < 0 or i >= n:
            raise IndexError(f"{ticker}: bar position {i} outside bucket of {n}")
        return SignalResult(entry=bool(entry_flags[i]), exit=bool(exit_flags[i]))

    signal_fn.family_id = FAMILY_ID
    signal_fn.ticker = ticker
    return signal_fn
