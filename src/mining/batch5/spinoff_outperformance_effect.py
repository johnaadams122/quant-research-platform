"""Batch5 candidate: spinoff_outperformance_effect (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json``
under the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instrument is re-tuned here.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "spinoff_outperformance_effect"): Cusatis, P., Miles, J.,
Woolridge, J.R. (1993), "Restructuring Through Spinoffs: The Stock Market
Evidence," Journal of Financial Economics, 33(3), 293-311 (found spinoffs
outperformed industry-matched benchmarks by ~30% over 3 years for 161
spinoffs, 1965-1988). Verified via web search, per the catalog
entry's own "source" field: Invesco S&P Spin-Off ETF (CSD), issuer
Invesco, inception December 15, 2006, confirmed currently trading on NYSE
Arca. CSD's daily price history begins on that inception date
(2006-12-15).

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "When a
company spins off a division into an independently-traded stock, both the
spinoff and its former parent tend to outperform the market for roughly
the following one to three years -- index funds and institutional holders
that only mandate the parent's profile are forced to sell the
newly-listed, often smaller/off-index spinoff regardless of its
fundamentals, creating temporary underpricing that corrects over time.
The strategy holds a continuously-refreshed basket of these post-spinoff
names, proxied by an ETF that adds each new U.S. spinoff shortly after it
begins trading and holds it for up to four years before rotating it out,
rather than trying to time individual spinoff events by hand."

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``free_cash_flow_yield_cows.py``'s and
``faith_based_sin_stock_screen.py``'s own disclosures of their
limitations): a real event-driven screen that identifies each individual
spinoff, tracks its own age since separation, and continuously
adds/removes names from a basket is NOT implemented here -- this repo's
backtest engine is single-ticker, not a portfolio/basket simulator capable
of that kind of continuous reconstitution. CSD is the real,
already-constructed ETF whose own index methodology already runs the
described mechanism internally: it adds newly completed U.S. spinoffs and
removes each holding once it has been held for
``PARAMETERS["holding_period_years"]`` (4) years, on an ongoing basis
(``PARAMETERS["rebalance"] == "continuous_index_tracking"``), so
buy-and-hold CSD (since its own 2006-12-15 inception) is used as a DIRECT
substitute for simulating the screen from scratch.

CSD -- the traded ticker -- already IS the continuously-refreshed,
already-managed basket: the issuer runs the add/hold-4-years/roll-off rule
internally. There is therefore no per-bar decision left for a signal
function to make; the testable hypothesis reduces to "does buy-and-hold
CSD clear the mining pipeline's bars" (own-instrument alpha, ABI-1 excess
over SPY, etc.), not a market-timing question. Note also: unlike most
other batch5 single-proxy candidates, this family's own catalog
``parameters["benchmark"]`` is ALSO "SPY" -- purely coincidental with
ABI-1's own hard SPY rule (see BENCHMARK below), not derived from it.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even CSD's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long CSD, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to ``free_cash_flow_yield_cows.py``'s
own choices for the same always-long shape -- restated here rather than
imported, since each batch5 module is self-contained per this pipeline's
convention):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first.
   ``backtest_engine.py``'s own docstring says the engine "only reads
   `entry` while flat and only reads `exit` while long", so a constant
   True entry is simply ignored once a position is open. (Special-casing
   ``i == 0`` instead would be behaviorally identical given that contract;
   the constant form matches the sibling batch5 modules.)
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
   trade, entered at the open of the second bar and held to end_of_data
   (a pure buy-and-hold rule cannot produce more than one trade per
   contiguous bucket).
5. CSD's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "spinoff_outperformance_effect"
SHAPE = "single_long"
TRADED = ["CSD"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# free_cash_flow_yield_cows.py's own identical disclosure for this same
# shape (a module attribute that would otherwise contradict the pipeline's
# hard rule -- here it happens to equal the catalog's own "benchmark":
# "SPY" entry too, purely coincidentally, not because this module reads
# the catalog field).
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "spinoff_outperformance_effect" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "holding_period_years": 4,
    "rebalance": "continuous_index_tracking",
    "benchmark": "SPY",
}
# Exactly 1 trade per contiguous bucket (buy at the first reachable bar,
# held to end_of_data) -- the structural argument in "Interpretation
# choices" item 4 above, which the synthetic end-to-end test checks through
# mining.backtest_engine.run_backtest. Set at that structural floor, not a
# safety margin below a larger count: a pure buy-and-hold rule
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
