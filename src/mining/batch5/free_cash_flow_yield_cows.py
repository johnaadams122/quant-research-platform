"""Batch5 candidate: free_cash_flow_yield_cows (single_long).

Implements the catalog mechanism under the ABI-1 batch5 module
contract. Nothing about the mechanism, the parameters, or the traded
instrument is re-tuned here. The per-trade runner
(``scripts/run_batch3_mining.py``) wires this same family_id, in its
``_single_factory`` (around line 783), to ``_always_long_factory``
(around line 777):

    def _always_long_factory(_index):
        def signal_fn(_frame, _i):
            return backtest_engine.SignalResult(entry=True, exit=False)
        return signal_fn

This module applies that same always-long substitution in the
batch5/ABI-1 contract shape. It is not a new design decision.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "free_cash_flow_yield_cows"): Hackel, Livnat & Rai (2000), "A
Free Cash Flow Investment Anomaly", Journal of Accounting, Auditing &
Finance 15(1). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=240540
(verified, per the catalog entry's own "source" field). Ticker verified: COWZ (Pacer US Cash Cows 100 ETF), issuer Pacer
Advisors, inception 2016-12-16, actively trading (per the same catalog
"source" field).

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "Rank
large/mid-cap US stocks by trailing free cash flow yield (FCF / enterprise
value) and hold the 100 highest-yielding names, reconstituted quarterly --
the exact Pacer US Cash Cows 100 Index rule."

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``faith_based_sin_stock_screen.py``'s and
``spinoff_outperformance_effect.py``'s own disclosures of their limitations):
a real point-in-time cross-sectional "free cash flow yield" screen over
the large/mid-cap US universe is NOT implemented here -- this repo's
backtest engine is single-ticker, not a portfolio/basket simulator capable
of ranking and reconstituting a 100-name universe every quarter. COWZ is
the real, already-constructed ETF that runs the exact Pacer US Cash Cows
100 Index rule the catalog mechanism describes, so buy-and-hold COWZ
(since its own 2016-12-16 inception) is used as a DIRECT substitute for
simulating the screen from scratch. This is the same substitution
``scripts/run_batch3_mining.py`` makes for this family (see
``_always_long_factory`` above). DSTL (also listed in the catalog's
"tickers") is an auxiliary comparator only and is never read by this
module, matching that script's own disposition (COWZ is its only traded
symbol).

COWZ -- the traded ticker -- already IS the screened, already-rebalanced
index fund: the issuer runs the quarterly FCF-yield ranking/reconstitution
internally (``PARAMETERS["rebalance"] == "quarterly"`` describes COWZ's OWN
mechanics, not a rule this module computes). There is therefore no per-bar
decision left for a signal function to make; the testable hypothesis
reduces to "does buy-and-hold COWZ clear the mining pipeline's bars"
(own-instrument alpha, ABI-1 excess over SPY, etc.), not a market-timing
question.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even COWZ's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long COWZ, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to ``faith_based_sin_stock_screen.py``'s
own choices for the same always-long shape -- restated here rather than
imported, since each batch5 module is self-contained per this pipeline's
convention):

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
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is taken
   completely literally: the position is closed only by the engine's own
   end_of_data path (or an external time_stop the driver may separately
   impose), never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's end_of_data path
   at the bucket's last bar -- so each bucket the driver runs this module
   over (in-sample, holdout, or ABI-1's full history) produces exactly one
   trade. This follows from the structure alone (a pure buy-and-hold
   rule cannot produce more than one trade per contiguous bucket).
5. COWZ's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "free_cash_flow_yield_cows"
SHAPE = "single_long"
TRADED = ["COWZ"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# fed_net_liquidity_regime_qqq.py's own identical disclosure for this same
# shape (a module attribute that would otherwise contradict the pipeline's
# hard rule if it echoed the catalog's own COWZ-relative framing).
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "free_cash_flow_yield_cows" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "screen_metric": "free_cash_flow_yield",
    "portfolio_size": 100,
    "rebalance": "quarterly",
}
# Exactly 1 trade per bucket (buy at the first reachable bar, held to
# end_of_data) -- the structural argument in "Interpretation choices"
# item 4 above. Set at the honest floor, not a
# safety margin below a larger measurement: a pure buy-and-hold rule
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
