"""Batch5 candidate: managed_futures_trend_replication (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json``
(status "eligible") under the ABI-1 batch5 module contract. Nothing about
the mechanism, the parameters, or the traded instrument is re-tuned here.
The rule is a literal always-long DBMF holding
(``def signal_fn(frame, i): return SignalResult(entry=True, exit=False)``),
because DBMF already runs the described mechanism internally and leaves no
per-bar decision (see the disclosed simplification below).

Catalog mechanism (verbatim, ``data/mining/catalog.json``, family_id
"managed_futures_trend_replication"): "Hold DBMF, an ETF that does not
follow a single-asset price rule but instead runs a rolling regression each
month to replicate the aggregate long/short futures positioning of a basket
of the largest managed-futures (CTA/trend-following) hedge funds across
equity indices, fixed income, currencies, and commodities simultaneously.
The strategy is simply to buy and hold this replicated basket, testing
whether the diversified, dynamically-net-long-or-short trend premium
documented in the managed-futures literature is captured by an investable
single ticker."

Source (verbatim, ``data/mining/catalog.json``'s own "source" field):
Hasanhodzic, Jasmina, and Andrew W. Lo (2007), "Can Hedge-Fund Returns Be
Replicated?: The Linear Case," Journal of Investment Management, 5(2), 5-45
(SSRN abstract_id=924565). Ticker verified: DBMF (iMGP DBi Managed Futures
Strategy ETF, issuer iM Global Partner / Dynamic Beta Investments, launched
May 2019); its daily price history starts 2019-05-08, consistent with the
catalog's "launched May 2019" note.

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``free_cash_flow_yield_cows.py``'s and
``wide_moat_quality_investing.py``'s own disclosures of the identical
shape): DBMF already IS the replicated basket -- the issuer runs the
monthly rolling regression against aggregate CTA/trend-follower positioning
internally (``PARAMETERS["rebalance"] == "monthly"`` describes DBMF's OWN
mechanics, not a rule this module computes). There is no per-bar decision
left for a signal function to make and no way for this single-ticker engine
to reach inside DBMF's own monthly regression re-fit; the testable
hypothesis reduces to "does buy-and-hold DBMF clear the mining pipeline's
bars" (own-instrument alpha, ABI-1 excess over SPY, etc.), not a
market-timing question.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same shape as every other pure buy-and-hold batch5
candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even DBMF's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long DBMF, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to ``free_cash_flow_yield_cows.py``'s and
``wide_moat_quality_investing.py``'s own choices for the same always-long
shape -- restated here rather than imported, since each batch5 module is
self-contained per this pipeline's convention):

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
   On DBMF's history (first traded bar 2019-05-08) the earliest entry is
   therefore the next session's open; the first bar's own Open is never
   used as an entry.
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is taken
   completely literally: the position is closed only by the engine's own
   end_of_data path (or an external time_stop the driver may separately
   impose), never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's own end_of_data
   path at the bucket's last bar -- so each bucket the driver runs this
   module over (in-sample, holdout, or ABI-1's full history) produces
   exactly one trade: a pure buy-and-hold rule cannot produce more than
   one trade per contiguous bucket (see the EXPECTED_MIN_TRADES comment
   below).
5. DBMF's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "managed_futures_trend_replication"
SHAPE = "single_long"
TRADED = ["DBMF"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's own
# MARKET_BENCHMARK comment and every sibling batch5 module's identical
# disclosure for this same rule (e.g. free_cash_flow_yield_cows.py,
# wide_moat_quality_investing.py).
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's real "parameters"
# field for family_id "managed_futures_trend_replication" (driver drift
# check; also verified byte-identical by this module's own test via
# json.dumps(sort_keys=True) read fresh from that file).
PARAMETERS = {
    "rebalance": "monthly",
    "asset_classes_replicated": ["equity_index", "fixed_income", "currency",
                                  "commodity"],
    "exposure_type": "regression_based_replication",
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
