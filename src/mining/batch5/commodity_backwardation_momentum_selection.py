"""Batch5 candidate: commodity_backwardation_momentum_selection (single_long).

Implements an already-catalogued mechanism from
``data/mining/catalog.json`` in the ABI-1 batch5 module contract. The
mechanism, the parameters and the traded instrument are exactly the
catalog entry's; nothing is tuned here. Same always-long single-proxy
shape as ``value_factor_book_to_market.py`` (and
``spinoff_outperformance_effect.py`` / ``free_cash_flow_yield_cows.py``)
-- the SIMPLEST shape in this batch.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "commodity_backwardation_momentum_selection"): Gorton, Gary B.,
Fumio Hayashi, and K. Geert Rouwenhorst (2013), "The Fundamentals of
Commodity Futures Returns," Review of Finance, 17(1), 35-105. SummerHaven
Dynamic Commodity Index methodology:
https://summerhavenindex.com/sdci_methodology/ . Ticker fact (catalog's
own "source" field): USCI (United States Commodity Index Fund, issuer
USCF Investments/Marygold), catalog-cited launch Aug 10, 2010, real and
currently exchange-traded.

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "Hold USCI, an
ETF that itself implements a monthly commodity-SELECTION rule (not a
price-trend rule applied by us): each month its underlying SummerHaven
Dynamic Commodity Index screens 27 eligible commodity futures and picks
the 7 showing the greatest backwardation (or least contango) plus the 7
showing the strongest 12-month momentum, subject to a minimum-sector-
diversification constraint, then equal-weights the resulting 14. The
strategy simply buys and holds this reformulated basket, testing whether
the backwardation/roll-yield-and-momentum contract-selection premium
(distinct from a fixed-weight commodity index) earns excess return."

DISCLOSED SIMPLIFICATION / "the issuer already implements periodic
reformation" reasoning (identical shape to
``value_factor_book_to_market.py``'s own disclosure, and to
``spinoff_outperformance_effect.py``'s CSD / ``free_cash_flow_yield_cows.py``'s
COWZ): the catalog's own ``PARAMETERS["rebalance"]`` is "monthly" -- USCI's
underlying SummerHaven Dynamic Commodity Index re-screens its 27-contract
universe and re-selects the 14-contract backwardation+momentum basket every
month. That monthly re-selection is a REAL FUND MECHANIC already carried
out by the issuer/index provider inside USCI itself -- it is not a
per-bar decision this module's signal needs to make: the strategy is
simply "buy and hold USCI" (the same "the issuer already implements
periodic reformation" reasoning as VLUE/CSD/COWZ in the sibling modules).
A bare constant entry=True/exit=False on every bar is therefore BEHAVIORALLY
IDENTICAL to any hypothetical per-bar "is today a reformation day" check,
given this engine's own contract: ``backtest_engine.run_backtest`` reads
``entry`` only while flat and ``exit`` only while long, so once the single
position opens at the first reachable bar, a later reformation-day check
that would only ever re-affirm the same always-long decision (this
mechanism has no short/flat/rotate branch -- it is "buy the reformulated
basket and hold," not a timing rule) can never fire differently than doing
nothing. There is therefore no per-bar decision left for a signal
function to make; the testable hypothesis reduces to "does buy-and-hold
USCI clear the mining pipeline's bars" (own-instrument alpha, ABI-1 excess
over SPY, etc.), not a market-timing or contract-selection-replication
question. The catalog's second-metric split (7 backwardation / 7 momentum,
14-contract basket, 27-name universe, min-sector-diversification) is real
USCI fund mechanics, not something this signal implements or approximates
-- it is carried in PARAMETERS (byte-identical to the catalog) purely for
provenance/driver-drift-check purposes, same convention as every other
single_long module in this directory.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even USCI's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long USCI, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical shape to
``value_factor_book_to_market.py``'s own choices -- restated here rather
than imported, since each batch5 module is self-contained per this
pipeline's convention):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first --
   see the DISCLOSED SIMPLIFICATION section above for why this is
   behaviorally identical to any hypothetical per-bar reformation-day gate
   given this engine's own contract.
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
   rule cannot produce more than one trade per contiguous bucket); the
   module's own tests check it on synthetic data.
5. USCI's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "commodity_backwardation_momentum_selection"
SHAPE = "single_long"
TRADED = ["USCI"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# value_factor_book_to_market.py's own identical disclosure for this same
# shape. This family's own catalog entry has no "benchmark" parameter key
# at all (see PARAMETERS below), so there is no coincidental agreement to
# note here.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "commodity_backwardation_momentum_selection" (driver drift
# check; also verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "selection_metrics": ["backwardation", "12_month_momentum"],
    "contracts_held": 14,
    "universe_size": 27,
    "rebalance": "monthly",
    "min_sector_diversification": True,
}
# Structural: exactly 1 trade per contiguous bucket (buy at the first
# reachable bar, held to end_of_data) -- the argument in "Interpretation
# choices" item 4 above. Set at the honest floor, not a
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
