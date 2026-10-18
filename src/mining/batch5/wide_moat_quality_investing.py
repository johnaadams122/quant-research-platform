"""Batch5 candidate: wide_moat_quality_investing (single_long).

The batch3 driver defines this family (``scripts/run_batch3_mining.py``,
``SUPPORTED_CONTRACTS["wide_moat_quality_investing"]``, mechanism_class
``single_proxy_buy_and_hold``). A pure buy-and-hold rule structurally cannot
produce more than one trade in any bucket. This module expresses the same
always-long rule, with the same parameters, in the batch5 module-contract
shape (``make_signal_fn`` returning a per-bar ``signal_fn``) instead of
batch3's ``_always_long_factory``.

Catalog mechanism (verbatim from ``data/mining/catalog.json``'s
"wide_moat_quality_investing" entry): "Hold US companies Morningstar's
equity-research team rates as having a durable 'Wide' Economic Moat,
filtered to the cheapest wide-moat names by price-to-fair-value, split into
two staggered-reconstituting 40-stock sub-portfolios." Source (verbatim from
the same catalog entry): Morningstar Wide Moat Focus Index methodology;
economic-moat framework in Dorsey, P. (2004), "The Little Book That Builds
Wealth", Wiley. Ticker verified: MOAT (VanEck Morningstar Wide Moat ETF),
issuer VanEck, inception 2012-04-24, actively trading (~13 billion USD AUM).

DISCLOSED SUBSTITUTION (the same simplification the batch3 mechanism class
makes, disclosed under this pipeline's convention, e.g.
faith_based_sin_stock_screen.py's own disclosed limitations): a real
point-in-time cross-sectional "morningstar_economic_moat_rating" screen with
"price_to_fair_value" valuation filtering and semiannual-staggered
reconstitution is NOT implemented here -- this engine is single-ticker, not
a portfolio simulator, so there is nothing in this codebase capable of
running that screen bar-by-bar. Instead this family tests "does MOAT (the
real, already-constructed proxy ETF that already runs this exact screen
internally, the same way faith_based_sin_stock_screen's CATH already runs
its own exclusionary screen) beat SPY buy-and-hold since inception" as a
DIRECT substitute for the named screen -- exactly the substitution the
batch3 mechanism class ``single_proxy_buy_and_hold`` expresses.
MOAT is the traded ticker
and already IS the screened, already-rebalanced portfolio: VanEck runs the
semiannual-staggered reconstitution internally (PARAMETERS.rebalance =
"semiannual_staggered" describes MOAT's OWN mechanics, not a rule this
module computes). There is therefore no per-bar decision left for a signal
function to make; the testable hypothesis reduces to "does buy-and-hold MOAT
clear the mining pipeline's bars" (own-instrument alpha, ABI-1 excess over
SPY, etc.), not a market-timing question.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same shape as faith_based_sin_stock_screen.py):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even MOAT's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult; nothing
  is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the signal
  does.
* DIRECTION -- always long MOAT, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (the pseudocode leaves these to the implementer, same
choices faith_based_sin_stock_screen.py already made for the identical
shape):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first.
   backtest_engine.py's own docstring says the engine "only reads `entry`
   while flat and only reads `exit` while long", so a constant True entry is
   simply ignored once a position is open.
2. The engine enters at the EARLIEST bar it can structurally reach:
   `run_backtest` first asks about bar 0's decision while flat (the call
   `signal_fn(df.iloc[:1], 0)` made when evaluating bar 1), so that very
   first check already returns entry=True and the position opens at bar 1's
   open -- one bar of unavoidable engine latency common to every candidate
   in this pipeline, not a modeling choice made here. On MOAT's full
   history the entry is therefore its second traded bar's open (the first
   traded bar's OWN Open is never used as an entry).
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is taken
   completely literally: the position is closed only by the engine's own
   end_of_data path (or an external time_stop the driver may separately
   impose), never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's own end_of_data
   path at the bucket's last bar -- so each bucket the driver runs this
   module over (in-sample, holdout, or ABI-1's full history) produces
   exactly one trade. This is an honest count, not a floor pulled from a
   pooled/event-driven family's convention: a pure buy-and-hold rule
   structurally cannot produce more than one trade per contiguous bucket,
   because nothing ever raises `exit` to close and reopen it.
   EXPECTED_MIN_TRADES is set to exactly that structural floor (1),
   matching faith_based_sin_stock_screen.py's own identical-shape
   convention -- there is no safety margin to set below 1.
5. MOAT's own frame is required to be present in `frames` (fail closed if
   it is missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's contract
   identical to every other candidate's and gives a clear error if the
   driver's data plumbing ever fails to supply it, rather than silently
   trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "wide_moat_quality_investing"
SHAPE = "single_long"
TRADED = ["MOAT"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 always benchmarks the account against SPY regardless of this field
# (see scripts/run_abi1_mining.py's own MARKET_BENCHMARK comment) -- the
# attribute records that pipeline-wide benchmark; no price (SPY's or MOAT's)
# is read as a signal or timing reference anywhere in this module.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's real
# "wide_moat_quality_investing" entry (driver drift check) -- verified
# via json.dumps(sort_keys=True) byte-identity against that file.
PARAMETERS = {
    "screen_metric": "morningstar_economic_moat_rating",
    "moat_rating_required": "wide",
    "valuation_filter": "price_to_fair_value",
    "rebalance": "semiannual_staggered",
}
# See docstring "Interpretation choices" item 4: a pure buy-and-hold rule
# with no rule-driven exit produces exactly one continuous holding period
# per bucket the driver runs this module over -- so a full-history
# run has exactly 1 trade by construction.
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
    exit=False -- literal buy-and-hold, see module docstring. This is the
    batch5-contract re-expression of batch3's ``_always_long_factory``
    (``def signal_fn(_frame, _i): return backtest_engine.SignalResult(
    entry=True, exit=False)``) -- same rule, same always-long behavior,
    only the factory shape changed."""
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
