"""Batch5 candidate: merger_arbitrage_risk_arb (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json`` under
the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instrument is re-tuned here -- the catalog
entry's own PROXY NOTE is treated as
authoritative and quoted verbatim rather than re-deriving the proxy
decision.

Source (verbatim from the pre-registered catalog entry,
``data/mining/catalog.json``, ``family_id == "merger_arbitrage_risk_arb"``,
``added_date`` 2026-08-29): Mitchell, M. & Pulvino, T. (2001),
"Characteristics of Risk and Return in Risk Arbitrage", Journal of Finance
56(6), 2135-2175.
https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00401 (verified via
web search 2026-08-29, per the catalog entry's own "source" field).

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "On announcement
of a definitive M&A deal, buy the target (in a cash deal) or buy the target
and short the acquirer at the stock-swap ratio (in a stock deal), capturing
the spread between the deal price and the target's market price as
compensation for deal-completion risk; exit at deal close or on deal-break.
Long-only or long/short depending on deal structure, no options -- distinct
event-driven mechanism from every other entry in this catalog, with a
documented return profile resembling a short index put (crash risk
concentrated in market downturns)."

Catalog PROXY NOTE (verbatim): "tested via
MNA (IQ Merger Arbitrage ETF, real/verified via web search 2026-08-29 --
launched 2009, tracks a systematic long-target/short-acquirer merger-arb
index) as a real, tradeable stand-in for the single-deal-by-deal strategy
described above -- MNA holds a diversified basket across many concurrent
deals rather than one target/acquirer pair at a time, so this measures a
real merger-arb product's daily-bar behavior, not a direct replication of
Mitchell & Pulvino's single-deal event-study construction. A future real
gate on any survivor would need to decide whether the ETF-proxy result is
sufficient or whether the literal per-deal construction needs separate
validation." Catalog ``tickers`` are ``["MNA", "announced M&A target/
acquirer pairs (screened universe)"]`` -- the second entry is a generic
label for the unimplemented per-deal screened universe, not a second real
ticker; it is not treated as one and is not
part of TRADED/SIGNAL below.

This module applies the catalog's MNA-proxy substitution decision; it does
not invent one. It reads MNA's historical daily OHLC only, never a live
intraday quote.

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``value_factor_book_to_market.py``'s and
``free_cash_flow_yield_cows.py``'s own disclosures of their simplifications):
the catalog's own mechanism is a per-deal, event-driven long/short
construction with a deal-by-deal entry (on announcement) and exit (at deal
close or break). None of that per-deal machinery is implemented here. Per
the catalog's own PROXY NOTE, MNA is a diversified, continuously-rebalanced
basket product that already runs its OWN internal per-deal entry/exit
machinery (the index methodology behind the ETF) -- so testing "does
buy-and-hold MNA clear the mining pipeline's bars" is exactly what the PROXY
NOTE licenses, and there is no per-bar decision left for a signal function to
make on top of that: a bare entry=True/exit=False on every bar is
BEHAVIORALLY the correct representation of "hold this basket product",
matching the same reasoning already applied to
``value_factor_book_to_market.py`` (VLUE's own periodic reformation) and
``free_cash_flow_yield_cows.py`` (COWZ's own quarterly reconstitution) for
the identical always-long-proxy-product shape. ``backtest_engine.py`` "only
reads `entry` while flat and only reads `exit` while long," so once the
single position opens at the first reachable bar, there is no rule left to
re-evaluate -- the testable hypothesis reduces to "does buy-and-hold MNA
clear the mining pipeline's bars" (own-instrument alpha, ABI-1 excess over
SPY, etc.), not the literal per-deal event-study question Mitchell & Pulvino
studied.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even MNA's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult; nothing
  is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the signal
  does.
* DIRECTION -- always long MNA, entered at the first structural opportunity
  and never exited by rule.

Interpretation choices (identical in shape to
``value_factor_book_to_market.py``'s own choices for the same always-long
single-proxy shape -- restated here rather than imported, since each batch5
module is self-contained per this pipeline's convention):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first -- see
   the DISCLOSED SIMPLIFICATION section above for why this is the correct
   representation of "hold the MNA basket product" given the PROXY NOTE and
   this engine's own contract.
2. The engine enters at the EARLIEST bar it can structurally reach:
   ``run_backtest`` first asks about bar 0's decision while flat (the call
   ``signal_fn(df.iloc[:1], 0)`` made when evaluating bar 1), so that very
   first check already returns entry=True and the position opens at bar 1's
   open -- one bar of unavoidable engine latency common to every candidate
   in this pipeline, not a modeling choice made here.
3. No time stop, no signal exit, no cash state -- "buy-and-hold the proxy
   basket" is taken completely literally: the position is closed only by the
   engine's own end_of_data path (or an external time_stop the driver may
   separately impose), never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's end_of_data path
   at the bucket's last bar -- so each bucket the driver runs this module
   over (in-sample, holdout, or ABI-1's full history) produces exactly one
   trade: a pure buy-and-hold rule cannot produce more than one trade per
   contiguous bucket (see the EXPECTED_MIN_TRADES comment below). MNA's
   price history begins in 2009, matching the catalog PROXY NOTE's own
   "launched 2009" claim.
5. MNA's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's contract
   identical to every other candidate's and gives a clear error if the
   driver's data plumbing ever fails to supply it, rather than silently
   trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "merger_arbitrage_risk_arb"
SHAPE = "single_long"
TRADED = ["MNA"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# value_factor_book_to_market.py's own identical disclosure for this same
# shape. This family's own catalog entry has no "benchmark" parameter key at
# all (see PARAMETERS below), so there is no coincidental agreement to note
# here.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "merger_arbitrage_risk_arb" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "event": "announced_merger_definitive_agreement",
    "position": "long_target_or_long_target_short_acquirer",
    "exit": "deal_close_or_break",
}
# Structural floor, not a measured count: this module's make_signal_fn run
# through mining.backtest_engine.run_backtest buys at the first reachable
# bar and holds to end_of_data, i.e. exactly 1 trade per contiguous bucket
# ("Interpretation choices" item 4 above). A pure buy-and-hold rule
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
