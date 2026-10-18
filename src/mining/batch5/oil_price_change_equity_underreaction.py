"""Batch5 candidate: oil_price_change_equity_underreaction (single_long).

Lagged monthly oil-price change timing broad equities (USO signal -> SPY).
Source: Driesprong, Jacobsen and Maat (2008), "Striking oil: Another
puzzle?", JFE 89(2). Rule, verbatim from the curated candidate's
rule_pseudocode:

    On the last trading day t_m of each calendar month m (SPY and USO
    aligned on their common calendar):
        r_oil    = close_USO[t_m] / close_USO[t_(m-1)] - 1
        signal_m = (r_oil <= 0)
    on bar t_m: signal_m      -> entry=True  (fills next bar open)
                not signal_m  -> exit=True   (fills at t_m's close)
    every other bar: entry=False, exit=False. No stop, no time stop;
    consecutive hold months merge into one trade.

The engine only reads ``entry`` while flat and only reads ``exit`` while
long, so emitting entry=signal_m / exit=not signal_m on every decision bar
is exactly the pseudocode's "if ... and no open position" / "if ... and a
position is open" without the signal function tracking position state.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- every series is precomputed ONCE in the factory from the
  USO closes at month-end bars only; the value stored for date t uses USO
  closes <= t. SPY's own prices never enter the signal.
* PERFORMANCE -- the returned signal_fn does two numpy lookups by position
  plus one scalar date check; nothing is recomputed over the growing slice.
* HOLDOUT-BLIND -- frames are truncated to <= the last bar of ``index``
  before anything is computed, so a holdout bucket's signals cannot depend
  on data after the bucket; history BEFORE the bucket is kept as warm-up
  (the prior month-end close the first decision needs).
* DIRECTION -- always long SPY; the "oil rose" state is flat, never short.

Interpretation choices (the pseudocode leaves these to the implementer):

1. "Last trading day of each calendar month" is decided on the COMMON
   calendar of SPY and USO (the intersection of their bar dates), as the
   pseudocode's alignment note says: bar t is a month-end iff the NEXT
   common-calendar bar falls in a different calendar month. The final bar
   of the (truncated) history therefore never counts as a month-end -- it
   cannot be known to be one without a later bar, and the engine's
   end_of_data path closes any open position at that same close anyway,
   so no P&L is changed by this choice (only the recorded exit reason).
2. t_(m-1) must be the month-end of the IMMEDIATELY preceding calendar
   month (lookback_months = 1). If the common calendar has no bar in the
   preceding month, no signal is emitted at t_m (never a two-month return).
3. The first month-end in USO's history has no prior month-end and emits
   no signal; the first decision is the month-end of USO's first FULL
   month after inception (May 2006 on real data -> first hold June 2006,
   matching the candidate's "in-sample from June 2006").
4. SPY bars with no USO bar (all of 1993-2006 in-sample) are silent
   (entry=False, exit=False): the signal is undefined there, not "flat by
   rule". A non-finite USO close at a month-end also yields no signal.
5. r_oil == 0 exactly is a hold (the rule says ``<= 0``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "oil_price_change_equity_underreaction"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL = ["USO"]
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from the candidate JSON "parameters" (driver drift check).
PARAMETERS = {
    "signal_ticker": "USO",
    "traded_ticker": "SPY",
    "lookback_months": 1,
    "decision_frequency": "monthly (last trading day of each calendar month)",
    "hold_condition": "prior-month USO close-to-close return <= 0",
    "position_when_oil_rose": "flat",
    "extra_lag_days": 0,
    "holding_period": "one calendar month, consecutive holds merge into one trade",
}
EXPECTED_MIN_TRADES = 50

_LOOKBACK_MONTHS = PARAMETERS["lookback_months"]


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def month_end_decisions(traded_dates: pd.DatetimeIndex,
                        signal_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Precompute (entry, exit) boolean Series on the common calendar of the
    traded and signal instruments.

    ``traded_dates`` and ``signal_close.index`` must already be tz-naive,
    normalized, sorted and unique. Both returned Series are indexed by the
    common calendar; True only on month-end bars with a valid one-month USO
    return (entry where r_oil <= 0, exit where r_oil > 0).
    """
    common = traded_dates.intersection(signal_close.index).sort_values()
    n = len(common)
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    if n >= 2:
        periods = common.to_period("M")
        is_month_end = np.zeros(n, dtype=bool)
        # Bar k is a month-end iff bar k+1 is in a different month; the last
        # bar has no successor and is never a month-end (docstring choice 1).
        is_month_end[:-1] = np.asarray(periods[1:] != periods[:-1])
        closes = pd.to_numeric(signal_close.reindex(common),
                               errors="coerce").to_numpy(dtype=float)
        me_pos = np.flatnonzero(is_month_end)
        for k in range(1, len(me_pos)):
            cur, prev = me_pos[k], me_pos[k - 1]
            # Exactly one calendar month apart (docstring choice 2).
            if (periods[cur] - periods[prev]).n != _LOOKBACK_MONTHS:
                continue
            c_now, c_prev = closes[cur], closes[prev]
            if not (np.isfinite(c_now) and np.isfinite(c_prev)) or c_prev <= 0:
                continue
            r_oil = c_now / c_prev - 1.0
            if r_oil <= 0.0:
                entry[cur] = True
            else:
                exit_[cur] = True
    return (pd.Series(entry, index=common, name="entry"),
            pd.Series(exit_, index=common, name="exit"))


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED + SIGNAL."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    traded_frame = _require_frame(frames, ticker)
    signal_frame = _require_frame(frames, PARAMETERS["signal_ticker"])

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    traded_dates = _naive_index(traded_frame.index)
    traded_dates = traded_dates[traded_dates <= last_bar].unique().sort_values()

    signal_close = pd.Series(signal_frame["Close"].to_numpy(),
                             index=_naive_index(signal_frame.index))
    signal_close = signal_close[~signal_close.index.duplicated(keep="last")]
    signal_close = signal_close.sort_index()
    signal_close = signal_close[signal_close.index <= last_bar]

    entry_series, exit_series = month_end_decisions(traded_dates, signal_close)
    entry_arr = entry_series.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    exit_arr = exit_series.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    bucket_values = bucket.asi8  # int64 ns for an O(1) scalar date check
    n_bucket = len(bucket)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n_bucket:
            raise IndexError(f"{FAMILY_ID}: bar position {i} outside the "
                             f"{n_bucket}-bar bucket this signal was built for")
        today = frame.index[-1]
        if pd.Timestamp(today).value != bucket_values[i]:
            raise ValueError(f"{FAMILY_ID}: bar {i} is {today}, but the bucket "
                             f"index has {bucket[i]} there -- the engine must "
                             "run over the same index handed to the factory")
        return SignalResult(entry=bool(entry_arr[i]), exit=bool(exit_arr[i]))

    return signal_fn
