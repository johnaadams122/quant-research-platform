"""Batch5 candidate: high_dividend_yield_factor (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json``
under the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instrument is re-tuned here.

A literal reading of the catalog's semi-annual rebalance would gate entry
to a semi-annual rebalance day. This module is built directly from: (1)
the catalog's own verbatim mechanism/parameters text (quoted below), and
(2) the same collapse of that gate that the
``spinoff_outperformance_effect.py`` sibling module applies to its own
always-long, never-exited mechanism -- see "DISCLOSED SIMPLIFICATION"
below for why that collapse is behaviorally exact, not a guess.

Catalog mechanism (verbatim, ``data/mining/catalog.json``, family_id
"high_dividend_yield_factor"): "Hold a market-cap-weighted basket of U.S.
common stocks drawn from the top half of the market by trailing dividend
yield (REITs excluded), rebalanced semi-annually, on the thesis from the
dividend-yield literature that stocks with above-average current yield
earn above-average future total returns. This is a cross-sectional
RANK-AND-HOLD on the CURRENT yield level, not a growth-history screen: a
stock qualifies purely by paying a high dividend relative to its price
today, regardless of how many consecutive years it has raised that
dividend."

Source (pre-registered catalog entry, verbatim citation): Fama, E. F., and
French, K. R. (1988), "Dividend Yields and Expected Stock Returns,"
Journal of Financial Economics 22(1), 3-25 (verified via web search
2026-08-29, per the catalog entry's own "source" field). Ticker verified
by the catalog entry: VYM is a real, currently-trading Vanguard ETF
launched November 10, 2006, tracking the FTSE High Dividend Yield Index.
VYM's yfinance history begins 2006-11-16 and carries no gaps or
non-finite prices once a fetch day's own in-progress (not-yet-closed)
session bar is excluded -- see "REAL-DATA FETCH CAVEAT" below.

DISTINCTNESS NOTE (catalog's own, verbatim): ranks purely on current
dividend YIELD LEVEL, distinct from the growth-streak screen
(``dividend_growth_aristocrats``) and from every other fundamental sort
already in the catalog -- yield-level and payout-growth-history are
separately-studied, only loosely correlated dividend factors.

Traded ticker: VYM only (catalog ``tickers[0]``, by design). Catalog ``tickers[1]``/``tickers[2]`` (SPYD, SCHD) are alternative
proxies for the same factor, not additional legs of a combined
strategy -- consistent with every other batch5 single-proxy factor-ETF
module in this pipeline (e.g. ``free_cash_flow_yield_cows.py`` trades only
COWZ though its own catalog entry lists alternates too), they are not
fetched or traded here.

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see ``spinoff_outperformance_effect.py``'s and
``free_cash_flow_yield_cows.py``'s own disclosures of their own
limitations): a semi-annual-rebalance-day entry gate is NOT implemented
here. VYM is the real, already-constructed ETF whose own
index methodology already runs the described cross-sectional
rank-and-hold internally (semi-annual reconstitution of the underlying
FTSE High Dividend Yield Index is the FUND's own process, not something
this signal needs to re-simulate) -- buy-and-hold VYM since its own real
inception is used as a DIRECT substitute for simulating the yield-rank
screen from scratch, exactly the same substitution shape as every other
single-proxy factor-ETF module in this batch. Given that substitution, a
"gate entry to the semi-annual rebalance day" rule and a bare
``entry=True`` constant are BEHAVIORALLY IDENTICAL under this engine's own
contract: ``backtest_engine.py``'s docstring states the engine "only reads
`entry` while flat and only reads `exit` while long," so once the first
semi-annual gate opens the position, every later gate day is read while
already long (a no-op) -- and this mechanism has no exit condition at all,
so the position, once opened, is never closed by rule. The two
formulations therefore produce the identical single trade over any
history; the semi-annual gate adds no behavior the constant does not
already have, so this module follows the simpler batch5 convention
(``spinoff_outperformance_effect.py``) rather than adding a gate that
cannot change the outcome.

REAL-DATA FETCH CAVEAT (a property of the vendor feed; not handled in
this module or in any shared loader): a
``data_pipeline.fetch_ohlc("VYM")`` call made during a live trading
session (market open) returns an all-NaN final row dated the fetch day
itself (an in-progress, not-yet-closed session bar from yfinance).
``run_batch3_mining._load_ohlc`` fails closed on such a row (by design,
and correctly so -- it is a general-purpose finite-value guard this
module does not touch or weaken). A caller fetching during market hours
drops that one trailing all-NaN row itself before handing the rest of the
frame to the shared ``_normalize_ohlc``, rather than disabling or
loosening the shared guard.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even VYM's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long VYM, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to ``spinoff_outperformance_effect.py``'s
and ``free_cash_flow_yield_cows.py``'s own choices for the same
always-long shape -- restated here rather than imported, since each
batch5 module is self-contained per this pipeline's convention):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first --
   `backtest_engine.py` only reads `entry` while flat, so a constant True
   entry is simply ignored once a position is open.
2. The engine enters at the EARLIEST bar it can structurally reach:
   ``run_backtest`` first asks about bar 0's decision while flat (the
   call ``signal_fn(df.iloc[:1], 0)`` made when evaluating bar 1), so
   that very first check already returns entry=True and the position
   opens at bar 1's open -- one bar of unavoidable engine latency common
   to every candidate in this pipeline, not a modeling choice made here.
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is taken
   completely literally: the position is closed only by the engine's own
   end_of_data path, never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's end_of_data path
   at the bucket's last bar -- so each bucket the driver runs this module
   over produces exactly one trade. This follows from the structure
   alone: a pure buy-and-hold rule cannot produce more than one trade
   per contiguous bucket.
5. VYM's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "high_dividend_yield_factor"
SHAPE = "single_long"
TRADED = ["VYM"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# spinoff_outperformance_effect.py's / free_cash_flow_yield_cows.py's own
# identical disclosure for this same shape (a module attribute that would
# otherwise contradict the pipeline's hard rule -- here it happens to equal
# the catalog's own "benchmark": "SPY" entry too, purely coincidentally,
# not because this module reads the catalog field).
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "high_dividend_yield_factor" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "sort_metric": "trailing_dividend_yield",
    "rank_half": "top",
    "rebalance": "semi_annual",
    "benchmark": "SPY",
}
# Structural floor (see "Interpretation choices" item 4 above): a pure
# buy-and-hold rule produces exactly one trade per contiguous bucket.
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
