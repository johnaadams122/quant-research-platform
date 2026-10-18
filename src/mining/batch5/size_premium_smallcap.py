"""Batch5 candidate: size_premium_smallcap (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json``
(status "eligible") under the ABI-1 batch5 module contract. Nothing about
the mechanism, the parameters, or the traded instrument is re-tuned here.

The rule is the simple bare-constant shape (entry=True, exit=False on
every bar, no rebalance-day gating at all), which matches
``data/mining/catalog.json``'s own mechanism prose for this family ("Entry is
at backtest start and the position stays fully invested throughout"). There
is nothing ambiguous about the rule.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "size_premium_smallcap"): Banz, R. W. (1981), "The Relationship
Between Return and Market Value of Common Stocks," Journal of Financial
Economics 9(1), 3-18 (per the catalog entry's own "source" field). IJR is
a currently-trading BlackRock/iShares ETF, tracking the S&P SmallCap 600
index. A vendor's daily history for IJR may begin a few calendar days
after the catalog note's cited 2000-05-22 launch date (e.g. a
listing-vs-first-printed-close distinction, or a data-vendor coverage
gap); that is disclosed rather than reconciled or silently overwritten.

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "Maintain a
long-only, continuously-held position in a broad small-capitalization
index fund (the S&P SmallCap 600), capturing the historically documented
'size effect' where smaller-capitalization firms earn higher average
risk-adjusted returns than large-cap firms. No cross-sectional stock-
picking is performed by this pipeline -- the ETF itself already holds
roughly 600 small-cap constituents that S&P reconstitutes quarterly -- so
the test is whether persistent exposure to this cap-weighted small-cap
basket earns a premium over a large-cap benchmark like SPY, independent of
any trend or momentum timing overlay. Entry is at backtest start and the
position stays fully invested throughout (subject only to the pipeline's
own generic risk-management overlay, if applied)."

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``free_cash_flow_yield_cows.py``'s and
``spinoff_outperformance_effect.py``'s own disclosures of their
limitations): the catalog's "quarterly" reconstitution is S&P's OWN index
methodology, executed inside the ETF, not a decision this signal makes --
there is no per-bar rebalance-day gating logic implemented here because
none is called for: IJR itself already holds the reconstituted basket, and
the strategy under test is plain continuous exposure to that basket, not a
timing overlay on top of it. This module never reads IJR's price at all,
identically to the spinoff/free-cash-flow-yield precedent for this
always-long shape.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even IJR's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long IJR, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to
``spinoff_outperformance_effect.py``'s own choices for the same
always-long shape -- restated here rather than imported, since each
batch5 module is self-contained per this pipeline's convention):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first.
   ``backtest_engine.py``'s own docstring says the engine "only reads
   `entry` while flat and only reads `exit` while long", so a constant
   True entry is simply ignored once a position is open.
2. The engine enters at the EARLIEST bar it can structurally reach:
   ``run_backtest`` first asks about bar 0's decision while flat (the
   call ``signal_fn(df.iloc[:1], 0)`` made when evaluating bar 1), so
   that very first check already returns entry=True and the position
   opens at bar 1's open -- one bar of unavoidable engine latency common
   to every candidate in this pipeline, not a modeling choice made here.
3. No time stop, no signal exit, no cash state -- "stays fully invested
   throughout" is taken completely literally: the position is closed only
   by the engine's own end_of_data path (or an external time_stop the
   driver may separately impose), never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's end_of_data path
   at the bucket's last bar -- so each bucket the driver runs this module
   over (in-sample, holdout, or ABI-1's full history) produces exactly one
   trade: a pure buy-and-hold rule cannot produce more than one trade per
   contiguous bucket.
5. IJR's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "size_premium_smallcap"
SHAPE = "single_long"
TRADED = ["IJR"]
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
# for family_id "size_premium_smallcap" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "index": "S&P SmallCap 600",
    "rebalance": "quarterly",
    "benchmark": "SPY",
    "position": "long_only_full_exposure",
}
# Structural value: exactly 1 trade per contiguous bucket (buy at the first
# reachable bar, held to end_of_data) -- see "Interpretation choices" item 4
# above. A pure buy-and-hold rule cannot produce more than one trade per
# contiguous bucket.
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
