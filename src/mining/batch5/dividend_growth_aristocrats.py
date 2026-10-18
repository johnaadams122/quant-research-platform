"""Batch5 candidate: dividend_growth_aristocrats (single_long).

Implements an already-catalogued mechanism from
``data/mining/catalog.json`` in the ABI-1 batch5 module contract. The
mechanism, the parameters and the traded instrument are exactly the
catalog entry's; nothing is tuned here.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "dividend_growth_aristocrats"): S&P Dow Jones Indices research,
"Dividend Investing and a Look Inside the S&P Dow Jones Dividend Indices"
and the S&P 500 Dividend Aristocrats research note (backtest Dec-1999 to
Jun-2019 showing +4.3%/yr annualized outperformance and a higher Sharpe
ratio vs. the S&P Composite 1500). Ticker verified: NOBL is a real,
currently-trading ProShares ETF launched October 9, 2013, tracking the
S&P 500 Dividend Aristocrats Index; its first daily bar is its
2013-10-09 inception date.

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "Hold an
equal-weighted basket of S&P 500 companies that have raised their dividend
every year for at least 25 consecutive years (the \"Dividend Aristocrats\"
methodology), with the basket reconstituted and rebalanced annually. The
screen selects purely for a multi-decade streak of consecutive dividend
increases -- evidence of durable earnings/cash-flow growth and
management's shareholder-return commitment -- not for current yield level
or a single-period profitability or leverage ratio; a high-yield stock
with a spotty or cut dividend history is excluded, while a low-current-
yield compounder with 25+ straight annual raises qualifies."

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``spinoff_outperformance_effect.py``'s and
``free_cash_flow_yield_cows.py``'s own disclosures of their limitations):
a real from-scratch screen that re-derives the 25-year consecutive-
dividend-increase streak for every S&P 500 constituent, reconstitutes an
equal-weighted basket, and rebalances it annually is NOT implemented
here -- this repo's backtest engine is single-ticker, not a
portfolio/basket simulator capable of that kind of periodic
reconstitution. NOBL is the real, already-constructed ETF whose own index
methodology already runs the described screen-and-annual-rebalance
mechanism internally (``PARAMETERS["screen"] ==
"min_25yr_consecutive_dividend_increases"``, ``PARAMETERS["rebalance"] ==
"annual"``), so buy-and-hold NOBL (since its own 2013-10-09 inception) is
used as a DIRECT substitute for simulating the screen from scratch -- the
same substitution ``spinoff_outperformance_effect.py`` makes for CSD's own
continuous-reconstitution index.

NOBL -- the traded ticker -- already IS the continuously re-screened,
annually-rebalanced basket: the issuer (via the S&P methodology it
licenses) runs the streak-screen-and-rebalance rule internally. There is
therefore no per-bar decision left for a signal function to make; the
testable hypothesis reduces to "does buy-and-hold NOBL clear the mining
pipeline's bars" (own-instrument alpha, ABI-1 excess over SPY, etc.), not
a market-timing question.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even NOBL's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long NOBL, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices:

1. Entry is NOT gated on an annual-rebalance-day check. This module
   follows the simpler batch5 convention established by
   ``spinoff_outperformance_effect.py``'s own "Interpretation choices"
   item 1: since ``backtest_engine.run_backtest`` reads ``entry`` only
   while flat and ``exit`` only while long, a bare "always entry=True,
   exit=False on EVERY bar" behaves exactly like a rebalance-day-gated
   entry check once the position is open -- the engine ignores every
   subsequent entry=True signal regardless of which bars it is raised
   on. Both produce exactly one continuous holding, closed only by the
   engine's end_of_data path; the always-on form opens it at the earliest
   reachable bar.
2. The engine enters at the EARLIEST bar it can structurally reach:
   ``run_backtest`` first asks about bar 0's decision while flat (the
   call ``signal_fn(df.iloc[:1], 0)`` made when evaluating bar 1), so
   that very first check already returns entry=True and the position
   opens at bar 1's open -- one bar of unavoidable engine latency common
   to every candidate in this pipeline, not a modeling choice made here.
3. No time stop, no signal exit, no cash state -- "buy-and-hold, annually
   rebalanced internally by the issuer" is taken completely literally at
   this module's level: the position is closed only by the engine's own
   end_of_data path (or an external time_stop the driver may separately
   impose), never by this module raising `exit`. The annual rebalance is
   entirely inside NOBL's own NAV and never surfaces as a signal-level
   event.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's end_of_data path
   at the bucket's last bar -- so each bucket the driver runs this module
   over (in-sample, holdout, or ABI-1's full history) produces exactly one
   trade. This follows from the structure alone: a pure buy-and-hold
   rule cannot produce more than one trade per contiguous bucket.
5. NOBL's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "dividend_growth_aristocrats"
SHAPE = "single_long"
TRADED = ["NOBL"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# spinoff_outperformance_effect.py's own identical disclosure for this
# same shape (a module attribute that would otherwise contradict the
# pipeline's hard rule -- here it happens to equal the catalog's own
# "benchmark": "SPY" entry too, purely coincidentally, not because this
# module reads the catalog field).
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "dividend_growth_aristocrats" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "screen": "min_25yr_consecutive_dividend_increases",
    "weighting": "equal",
    "rebalance": "annual",
    "benchmark": "SPY",
}
# Exactly 1 trade per contiguous bucket (buy at the first reachable bar,
# held to end_of_data) -- the structural argument in "Interpretation
# choices" item 4 above. Set at the honest floor: a pure buy-and-hold rule
# structurally cannot produce more than one trade per contiguous bucket.
EXPECTED_MIN_TRADES = 1


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED (SIGNAL is
    empty for this family). Returns a closure that always signals
    entry=True, exit=False -- literal buy-and-hold, see module docstring."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)  # fail closed if the traded frame is absent

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
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
        return SignalResult(entry=True, exit=False)

    return signal_fn
