"""Batch5 candidate: pooled 12-month time-series momentum on single-country
equity ETFs (family ``country_equity_etf_pooled_tsmom``, shape
``pooled_single_asset``).

Rule (verbatim from the curated candidate's ``rule_pseudocode``, applied
independently to every pool member; ``make_signal_fn`` is called once per
ticker by the driver):

    C = auto-adjusted daily close of T (total return)
    M[k] = last available close of T in calendar month k
    at the close of the last trading day of each calendar month m
    (needs 12 prior month-end closes, else no decision):
        ret12 = M[m] / M[m-12] - 1
        target = LONG if ret12 > 0 else FLAT   (ret12 == 0 -> FLAT)
    target LONG and flat  -> BUY at the OPEN of the next trading session
    target FLAT and long  -> SELL at the CLOSE of the next trading session
    otherwise hold; no stops, no time exits

Engine mapping (mining.backtest_engine): an ``entry`` flag on bar t fills at
bar t+1's open, an ``exit`` flag on bar t fills at bar t's close. So the
entry flag is raised on the month-end decision bar itself (fill = next
session's open, as the rule says) and the exit flag is raised on the bar
AFTER a FLAT decision (fill = that next session's close). Both flags are
precomputed once per ticker in the factory from shifted / grouped
operations only, reindexed to the bucket, and looked up by position -- the
per-bar signal function is two array reads and never touches the slice.

Interpretation choices
----------------------
1. "Last trading day of each calendar month" is taken from the fund's OWN
   bar calendar (the last bar of each calendar month present in its
   frame). The repo's vendored NYSE calendar covers 2026-2028 only, so it
   cannot serve a 1996+ history. The frame's final bar is therefore treated
   as its month's month-end; the only flag that can differ from a
   true-calendar reading is an ENTRY flag on that final bar, which the
   engine never reads (there is no next bar to fill at).
2. "Needs 12 prior month-end closes" is read as: the month-end close of the
   calendar month exactly ``lookback_months`` before m must exist in the
   frame (period arithmetic, not a count of rows). No interpolation across
   a missing month; a missing anchor means no decision that month.
3. "Total return" is the pipeline's Close as delivered
   (``data_pipeline.fetch_ohlc`` fetches ``auto_adjust=True``, i.e.
   dividend- and split-adjusted).
4. While the standing target is LONG the entry flag stays raised (the
   "stay long" idiom the engine needs). Its only extra effect is that a
   bucket which OPENS inside a LONG month enters at its first fill
   opportunity -- the closest reading of "hold the fund for the following
   month" given the engine cannot start a bucket already in a position.
5. Exit flags stay raised for every bar whose PREVIOUS bar's standing
   target was FLAT; the engine ignores exits while flat, so this changes
   nothing beyond the single sell at the next session's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "country_equity_etf_pooled_tsmom"
SHAPE = "pooled_single_asset"
TRADED = ["EWJ", "EWG", "EWU", "EWC", "EWA", "EWH", "EWW", "EWS", "EWY",
          "EWT", "EWZ", "FXI", "EZA", "INDA", "EWL"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "VEU"
# Copied from the curated candidate's "parameters" (== the
# data/mining/catalog.json entry); the driver rejects any drift under
# json.dumps(sort_keys=True).
PARAMETERS = {
    "lookback_months": 12,
    "decision_frequency": "monthly_at_month_end_close",
    "long_condition": "trailing_12_month_total_return_gt_0",
    "flat_condition": "trailing_12_month_total_return_le_0",
    "entry": "next_session_open",
    "exit": "next_session_close",
    "pool_size": 15,
    "pooling": "all_pool_trades_concatenated_one_trial",
    "benchmark_note": ("VEU starts 2007-04-30; earlier trades are benchmarked "
                       "against the member fund's own buy-and-hold and SPY "
                       "(both already in the batch5 directional panel)"),
}
EXPECTED_MIN_TRADES = 200

LOOKBACK_MONTHS = int(PARAMETERS["lookback_months"])


def _tz_naive(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out


def month_end_targets(close: pd.Series, lookback_months: int) -> pd.Series:
    """Per-bar decision series aligned to ``close.index``: 1.0 = LONG, 0.0 =
    FLAT on a month-end bar whose trailing ``lookback_months``-month return
    is decidable; NaN on every other bar (non-month-end, or no anchor).

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
    is_month_end = np.empty(len(codes), dtype=bool)
    is_month_end[:-1] = codes[:-1] != codes[1:]
    is_month_end[-1] = True                       # see docstring choice 1

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
    exit[t] = standing[t-1] is FLAT (fills at t's close, i.e. the session
    after the FLAT decision).
    """
    targets = month_end_targets(frame["Close"], lookback_months)
    standing = targets.ffill()
    entry = standing == 1.0
    exit_ = standing.shift(1) == 0.0
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
