"""batch5 candidate: customer_supplier_leadlag_semiconductors (single_long).

Customer-momentum lead-lag (Cohen & Frazzini 2008; Menzly & Ozbas 2010):
Nasdaq-100 customers (QQQ, signal only) timing semiconductor suppliers (SMH,
traded). On the last trading day t_m of each calendar month, on the QQQ/SMH
COMMON calendar::

    r_cust   = close_QQQ[t_m] / close_QQQ[t_(m-1)] - 1
    r_supp   = close_SMH[t_m] / close_SMH[t_(m-1)] - 1
    signal_m = (r_cust - r_supp) > 0

Engine mapping (backtest_engine.SignalResult per bar of the traded SMH
bucket): on t_m return ``entry=signal_m, exit=not signal_m`` -- the engine
itself only reads ``entry`` while flat and ``exit`` while long, so consecutive
True months merge into one holding and a False month-end closes at that
bar's close; every other bar is ``(False, False)``. No stop, no time stop.

Everything is precomputed ONCE in the factory into an int8 array aligned to
``index`` (+1 entry, -1 exit, 0 nothing); the per-bar call is one array
lookup.

Interpretation choices
----------------------
* "Last trading day of each calendar month" is a calendar fact, known in
  advance from the exchange calendar: bar t is a month-end iff the NEXT bar
  of the common QQQ/SMH calendar falls in a different calendar month. This
  uses only the DATES of the frames (never a future price). The very last
  bar of the common calendar has no successor, so it is never treated as a
  decision bar (no decision is invented without a confirmed month boundary;
  the engine's end_of_data close handles an open position there anyway).
* t_(m-1) is the previous month-end bar of the common calendar (the prior
  element of the month-end series); the first month-end has no anchor and
  makes no decision.
* A bucket bar absent from the common calendar (QQQ has no bar that day) is
  silent; the common-calendar month-end of that month decides instead.
* "> 0" is strict: a spread of exactly zero is a non-signal (exit while
  long, no entry while flat).
* History caveat (from the candidate parameters): SMH's series before
  2011-12-21 is the Semiconductor HOLDRS fixed-basket trust; disclosed in the
  run report, not filtered here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "customer_supplier_leadlag_semiconductors"
SHAPE = "single_long"
TRADED = ["SMH"]
SIGNAL = ["QQQ"]
SNAPSHOTS = {}
BENCHMARK = "SMH"
# Copied verbatim from the curated candidate entry's "parameters" (driver drift check).
PARAMETERS = {
    "customer_signal_ticker": "QQQ",
    "traded_supplier_ticker": "SMH",
    "formation_months": 1,
    "holding_months": 1,
    "signal": "prior-month QQQ return minus prior-month SMH return > 0",
    "decision_frequency": "monthly (last trading day of each calendar month)",
    "position_otherwise": "flat",
    "holding_rule": "consecutive True months merge into one trade",
    "history_caveat": (
        "SMH's yfinance series before 2011-12-21 is the Semiconductor HOLDRS "
        "fixed-basket trust (real traded prices, not an index fund); any "
        "result must disclose this"
    ),
}
EXPECTED_MIN_TRADES = 80

_ENTRY = np.int8(1)
_EXIT = np.int8(-1)


def _closes(frames, symbol: str) -> pd.Series:
    if symbol not in frames:
        raise ValueError(f"{FAMILY_ID}: frames must contain {symbol}")
    frame = frames[symbol]
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame must be a non-empty OHLC DataFrame")
    closes = pd.to_numeric(frame["Close"], errors="coerce")
    closes.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    closes = closes.dropna()
    closes = closes[~closes.index.duplicated(keep="last")].sort_index()
    if closes.empty or not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any():
        raise ValueError(f"{FAMILY_ID}: {symbol} Close values must be finite and positive")
    return closes.astype(float)


def month_end_decisions(frames) -> pd.Series:
    """Decision per common-calendar month-end date: True = long next month,
    False = flat. Dates with no prior month-end anchor are absent."""
    supplier = _closes(frames, PARAMETERS["traded_supplier_ticker"])
    customer = _closes(frames, PARAMETERS["customer_signal_ticker"])
    common = supplier.index.intersection(customer.index).sort_values()
    if len(common) < 2:
        raise ValueError(f"{FAMILY_ID}: QQQ/SMH share fewer than two sessions")
    periods = common.to_period("M")
    # bar t is a month-end iff the NEXT common bar is in a different month;
    # the final bar has no successor and is never a decision bar.
    is_month_end = np.zeros(len(common), dtype=bool)
    is_month_end[:-1] = periods[:-1] != periods[1:]
    month_ends = common[is_month_end]
    cust = customer.reindex(month_ends)
    supp = supplier.reindex(month_ends)
    # shift(1) = the previous month-end close (t_(m-1)); the first is NaN.
    r_cust = cust / cust.shift(1) - 1.0
    r_supp = supp / supp.shift(1) - 1.0
    spread = (r_cust - r_supp).dropna()
    return (spread > 0.0).rename("long_next_month")


def make_signal_fn(ticker: str, index, frames):
    """SignalFn for the SMH bucket ``index``; precomputes every decision once."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker {TRADED[0]!r}, got {ticker!r}")
    bucket = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()
    decisions = month_end_decisions(frames)
    state = np.zeros(len(bucket), dtype=np.int8)
    # NaN where the bucket bar is not a common-calendar decision bar
    aligned = decisions.astype(float).reindex(bucket).to_numpy(dtype=float)
    known = np.isfinite(aligned)
    longs = aligned == 1.0
    state[known & longs] = _ENTRY
    state[known & ~longs] = _EXIT
    n = len(state)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n:
            return SignalResult()
        value = state[i]
        if value == _ENTRY:
            return SignalResult(entry=True, exit=False)
        if value == _EXIT:
            return SignalResult(entry=False, exit=True)
        return SignalResult()

    signal_fn.decision_states = state  # exposed for tests / audit only
    return signal_fn
