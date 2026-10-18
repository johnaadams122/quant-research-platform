"""Batch5 candidate: value_factor_book_to_market (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json``
under the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instrument is re-tuned here.

SOURCING: this module is built from ``data/mining/catalog.json``'s own
mechanism/parameters/source/PROXY NOTE for this family_id (quoted below).
Nothing about the mechanism is guessed: a buy-and-hold-VLUE proxy for an
annual-rebalance cross-sectional value screen is exactly what the
catalog's own PROXY NOTE already prescribes.

This module ports the catalog's VLUE-proxy substitution decision forward;
it does not invent one from a blank slate.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "value_factor_book_to_market"): Fama, E.F. & French, K.R.
(1992), "The Cross-Section of Expected Stock Returns," Journal of Finance
47(2), 427-465 (verified 2026-08-29, per the catalog entry's own
"source" field).

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "Rank equities
by book-to-market ratio (or a composite of book-to-market/earnings-to-
price/cash-flow-to-price); buy the highest-book-to-market ('cheap') decile
and hold with periodic reformation, expecting price to mean-revert toward
fundamental value over a multi-year horizon."

Catalog PROXY NOTE (from the catalog entry's "source" field): "the
backtest engine only fetches a single ticker's OHLC (tickers[0]), so this
cross-sectional decile strategy is tested via VLUE (iShares MSCI USA Value
Factor ETF, real/verified 2026-08-29) as a real, tradeable
stand-in basket -- VLUE tracks a broad value-factor INDEX (MSCI's own
methodology), not a literal book-to-market decile sort, so a pass/fail
here says something about 'a real value-factor product's daily-bar
behavior', not a direct replication of Fama-French's original portfolio
construction." Catalog ``tickers`` are ``["VLUE", "screened equity
universe (book-to-market ranked)"]`` -- the second entry is a generic
label for the unimplemented cross-sectional universe, not a second real
ticker; it is not treated as one and is not part of TRADED/SIGNAL below.

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``spinoff_outperformance_effect.py``'s and
``free_cash_flow_yield_cows.py``'s own disclosures of their
simplifications): the catalog's own ``PARAMETERS["rebalance"]`` is
"annual", but a per-bar annual-rebalance-day entry gate is NOT
reproduced here. Per the same reasoning already
applied to the sibling ``dividend_growth_aristocrats`` port and to this
module's own direct precedent ``spinoff_outperformance_effect.py``, a
bare constant entry=True/exit=False on every bar is BEHAVIORALLY IDENTICAL
given this engine's own contract: ``backtest_engine.py`` "only reads
`entry` while flat and only reads `exit` while long," so once the single
position opens at the first reachable bar, a later rebalance-day check
that would only ever re-affirm the same always-long decision (this
mechanism has no short/flat/rotate branch -- it is "buy the cheap decile
and hold with periodic reformation," not a timing rule) can never fire
differently than doing nothing. VLUE itself already runs its own real
periodic reformation internally (MSCI's index methodology), exactly the
same shape as ``spinoff_outperformance_effect.py``'s CSD (continuous
index tracking) and ``free_cash_flow_yield_cows.py``'s COWZ (quarterly
reconstitution) -- the issuer, not this signal, is what already implements
"periodic reformation." There is therefore no per-bar decision left for a
signal function to make; the testable hypothesis reduces to "does
buy-and-hold VLUE clear the mining pipeline's bars" (own-instrument alpha,
ABI-1 excess over SPY, etc.), not a market-timing question.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even VLUE's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long VLUE, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to ``spinoff_outperformance_effect.py``'s
and ``free_cash_flow_yield_cows.py``'s own choices for the same
always-long shape -- restated here rather than imported, since each
batch5 module is self-contained per this pipeline's convention):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first --
   see the DISCLOSED SIMPLIFICATION section above for why this is
   behaviorally identical to an annual rebalance-day entry gate given this
   engine's own contract.
2. The engine enters at the EARLIEST bar it can structurally reach:
   ``run_backtest`` first asks about bar 0's decision while flat (the
   call ``signal_fn(df.iloc[:1], 0)`` made when evaluating bar 1), so
   that very first check already returns entry=True and the position
   opens at bar 1's open -- one bar of unavoidable engine latency common
   to every candidate in this pipeline, not a modeling choice made here.
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is taken
   completely literally: the position is closed only by the engine's own
   end_of_data path (or an external time_stop the driver may separately
   impose), never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's end_of_data path
   at the bucket's last bar -- so each bucket the driver runs this module
   over (in-sample, holdout, or ABI-1's full history) produces exactly one
   trade. This is the structural argument (a pure buy-and-hold rule cannot
   produce more than one trade per contiguous bucket), which this module's
   own synthetic end-to-end test checks through
   ``mining.backtest_engine.run_backtest``.
5. VLUE's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "value_factor_book_to_market"
SHAPE = "single_long"
TRADED = ["VLUE"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# free_cash_flow_yield_cows.py's own identical disclosure for this same
# shape. This family's own catalog entry has no "benchmark" parameter key
# at all (see PARAMETERS below), so there is no coincidental agreement to
# note here, unlike spinoff_outperformance_effect.py's CSD.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "value_factor_book_to_market" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "sort_metric": "book_to_market",
    "rank_decile": "top",
    "rebalance": "annual",
}
# Exactly 1 trade per contiguous bucket (buy at the first reachable bar,
# held to end_of_data) -- the structural argument in "Interpretation
# choices" item 4 above. Set at that structural floor, not a safety margin
# below a larger count: a pure buy-and-hold rule structurally cannot
# produce more than one trade per contiguous bucket.
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
