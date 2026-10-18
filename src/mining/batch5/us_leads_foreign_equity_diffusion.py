"""us_leads_foreign_equity_diffusion -- batch5 candidate (shape single_long).

US-leads-the-world lead-lag (Rapach, Strauss and Zhou 2013, J. Finance
68(4)): at each month-end, if SPY's return over the just-ended calendar
month is positive, hold EFA (MSCI EAFE) through the next month; otherwise
stay flat. SPY is signal-only; EFA is the traded instrument and its own
history never enters the signal.

Rule (verbatim from the curated candidate's rule_pseudocode):
    On the last trading day t_m of each calendar month m (SPY and EFA
    aligned on their common calendar):
        r_us     = close_SPY[t_m] / close_SPY[t_(m-1)] - 1
        signal_m = (r_us > 0)
    Engine mapping (SignalResult per bar):
        on bar t_m:  signal_m      -> entry=True,  exit=False
                     not signal_m  -> entry=False, exit=True
        every other bar: entry=False, exit=False
    (consecutive True months merge into one trade; no stop, no time stop.)

Interpretation choices
----------------------
* "Last trading day of each calendar month" is decided on the COMMON
  SPY/EFA calendar (intersection of the two frames' dates), truncated to
  the last bar of the bucket `index` the engine runs over. A bar is a
  decision bar when the NEXT common bar belongs to a different calendar
  month. The final bar of the truncated calendar has no next bar and is
  therefore never a decision bar -- this changes no trade: an entry
  signalled there could never fill (no next open) and an exit there is
  identical to the engine's own end_of_data close on the same bar. It
  keeps the factory holdout-blind (nothing after index[-1] is consulted,
  not even the calendar).
* The anchor close t_(m-1) is the previous decision bar's SPY close on the
  same common calendar, i.e. `lookback_months` decision bars back
  (lookback_months=1 per the candidate). The first month-end in the data
  has no anchor and is a no-op; warm-up history BEFORE the bucket supplies
  the anchor for the bucket's first month-end.
* A bar in `index` that has no SPY observation (or is not a decision bar)
  is a no-op (entry=False, exit=False), never a forward fill.
* r_us exactly 0 is NOT > 0 -> flat (strict inequality as written).

Look-ahead: the value stored for a decision bar t uses SPY closes at t and
at an earlier decision bar only. Whether t is a month-end is a calendar
fact (the next scheduled session's month), never a future PRICE.

Performance: every series is precomputed ONCE in the factory and reindexed
to `index`; the per-bar signal_fn is two boolean array lookups plus one
timestamp equality guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "us_leads_foreign_equity_diffusion"
SHAPE = "single_long"
TRADED = ["EFA"]
SIGNAL = ["SPY"]
SNAPSHOTS = {}
BENCHMARK = "EFA"
# Copied byte-for-byte from the curated candidate entry's "parameters"
# (family us_leads_foreign_equity_diffusion; driver drift check).
PARAMETERS = {
    "signal_ticker": "SPY",
    "traded_ticker": "EFA",
    "lookback_months": 1,
    "decision_frequency": "monthly (last trading day of each calendar month)",
    "hold_condition": "prior-month SPY close-to-close return > 0",
    "position_otherwise": "flat",
    "holding_rule": "consecutive True months merge into one trade",
}
EXPECTED_MIN_TRADES = 60


def _close_series(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(
        dtype=float), index=index.normalize())
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def make_signal_fn(ticker: str, index: pd.DatetimeIndex, frames: dict):
    """Build the per-bar SignalFn for the bucket `index` (see module doc)."""
    traded, signal_symbol = PARAMETERS["traded_ticker"], PARAMETERS["signal_ticker"]
    lookback = int(PARAMETERS["lookback_months"])
    if ticker != traded:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker {traded!r}, "
                         f"got {ticker!r}")
    for symbol in (traded, signal_symbol):
        if symbol not in frames:
            raise ValueError(f"{FAMILY_ID}: frames lacks {symbol}")

    bucket = pd.DatetimeIndex(pd.to_datetime(index))
    if bucket.tz is not None:
        bucket = bucket.tz_localize(None)
    bucket = bucket.normalize()
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index is empty")
    last = bucket[-1]

    spy_close = _close_series(frames[signal_symbol], signal_symbol)
    efa_close = _close_series(frames[traded], traded)
    # Common SPY/EFA calendar, holdout-blind: nothing after the bucket's
    # last bar is consulted.
    common = spy_close.index.intersection(efa_close.index)
    common = common[common <= last]
    spy_on_common = spy_close.reindex(common)

    entry_by_date = pd.Series(False, index=common)
    exit_by_date = pd.Series(False, index=common)
    if len(common) >= 2:
        periods = common.to_period("M")
        # Decision bar: the NEXT common bar is in a different calendar month.
        is_decision = np.zeros(len(common), dtype=bool)
        is_decision[:-1] = periods[:-1] != periods[1:]
        decision_close = spy_on_common[is_decision]
        anchor = decision_close.shift(lookback)
        valid = decision_close.notna() & anchor.notna() & (anchor > 0)
        r_us = decision_close[valid] / anchor[valid] - 1.0
        hold = r_us > 0
        entry_by_date.loc[hold.index] = hold.to_numpy()
        exit_by_date.loc[hold.index] = (~hold).to_numpy()

    entry_arr = entry_by_date.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    exit_arr = exit_by_date.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    bucket_values = bucket.asi8

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        today = frame.index[-1]
        if today.value != bucket_values[i]:
            raise ValueError(
                f"{FAMILY_ID}: signal_fn bar {i} is {today} but the bucket "
                f"index has {bucket[i]} -- frame/index misaligned")
        return SignalResult(entry=bool(entry_arr[i]), exit=bool(exit_arr[i]))

    return signal_fn
