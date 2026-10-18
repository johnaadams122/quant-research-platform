"""Batch5 candidate: faith_based_sin_stock_screen (single_long).

Buy-and-hold CATH (Global X S&P 500 Catholic Values ETF). Source: Hong &
Kacperczyk (2009), "The Price of Sin: The Effects of Social Norms on
Markets", JFE 93(1), 15-36. Catalog mechanism, verbatim from the curated
candidate's "mechanism" field:

    Hold the S&P 500 with companies excluded per US Conference of
    Catholic Bishops guidelines (abortion, contraceptives, embryonic
    stem-cell research, for-profit weapons), reweighting remaining
    constituents to track the broad index. Tests whether norm-constrained
    institutional neglect of excluded 'sin' names creates a return
    premium the screened index forgoes (a performance-drag test vs. its
    unscreened parent).

CATH -- the traded ticker -- already IS the screened, already-rebalanced
index fund: the issuer runs the quarterly exclusion/reweight internally
(parameters.rebalance = "quarterly" describes CATH's OWN mechanics, not a
rule this module computes). There is therefore no per-bar decision left
for a signal function to make; the testable hypothesis reduces to "does
buy-and-hold CATH clear the mining pipeline's bars" (own-instrument alpha,
ABI-1 excess over SPY, etc.), not a market-timing question. SPY appears
in the catalog's own `tickers` list only as ABI-1's fixed benchmark
(wired by the driver) -- it is never read as a SIGNAL ticker by this
module.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even
  CATH's own), so there is nothing for a future bar to leak into a past
  decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency,
  truncating ``frames`` to <= the bucket's last bar cannot change
  anything the signal does.
* DIRECTION -- always long CATH, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (the catalog mechanism leaves these to the implementer):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first.
   backtest_engine.py's own docstring says the engine "only reads
   `entry` while flat and only reads `exit` while long", so a constant
   True entry is simply ignored once a position is open -- the same
   "stay long" idiom fx_dollar_index_trend_following uses for its
   standing-LONG months (that module's docstring choice 5), but simpler
   here because there is no month-end gate raising or lowering it: there
   is no signal at all, by design (see module docstring above).
2. The engine enters at the EARLIEST bar it can structurally reach:
   `run_backtest` first asks about bar 0's decision while flat (the call
   `signal_fn(df.iloc[:1], 0)` made when evaluating bar 1), so that very
   first check already returns entry=True and the position opens at bar
   1's open -- one bar of unavoidable engine latency common to every
   candidate in this pipeline, not a modeling choice made here.
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is
   taken completely literally: the position is closed only by the engine's own end_of_data path (or an
   external time_stop the driver may separately impose), never by this
   module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always
   False, exactly one continuous holding period opens (at the bucket's
   first reachable bar) and is only ever closed by the engine's
   end_of_data path at the bucket's last bar -- so each bucket the
   driver runs this module over (in-sample, holdout) produces exactly
   one trade. This is an honest count, not a floor pulled from a
   pooled/event-driven family's convention: a pure buy-and-hold rule
   structurally cannot produce more than one trade per contiguous
   bucket, because nothing ever raises `exit` to close and reopen it.
5. CATH's own frame is required to be present in `frames` (fail closed
   if it is missing, matching every other module's convention) even
   though its prices are never read by the signal -- this keeps the
   factory's contract identical to every other candidate's (the caller
   always passes full-history OHLC for every TRADED symbol) and gives a
   clear error if the driver's data plumbing ever fails to supply it,
   rather than silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "faith_based_sin_stock_screen"
SHAPE = "single_long"
TRADED = ["CATH"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "CATH"
# Copied byte-for-byte from the catalog entry's "parameters" (driver drift check).
PARAMETERS = {
    "screen_type": "faith_based_exclusionary",
    "base_universe": "sp500",
    "rebalance": "quarterly",
}
# See docstring "Interpretation choices" item 4: a pure buy-and-hold rule
# with no rule-driven exit produces exactly one continuous holding period
# per bucket the driver runs this module over.
EXPECTED_MIN_TRADES = 1


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


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED (+ SIGNAL,
    empty here). Returns a closure that always signals entry=True,
    exit=False -- literal buy-and-hold, see module docstring."""
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
