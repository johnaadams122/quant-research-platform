"""Batch5 candidate: buyback_yield_repurchase_anomaly (single_long).

Implements an already-catalogued mechanism from
``data/mining/catalog.json`` in the ABI-1 batch5 module contract. The
mechanism, the parameters and the traded instrument are exactly the
catalog entry's; nothing is tuned here.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "buyback_yield_repurchase_anomaly"): Ikenberry, D., Lakonishok,
J. & Vermaelen, T. (1995), "Market Underreaction to Open Market Share
Repurchases", Journal of Financial Economics 39(2-3), 181-208.
https://www.sciencedirect.com/science/article/abs/pii/0304405X9500826Z
(verified, per the catalog entry's own "source" field).

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "Buy equities
following open-market share-repurchase announcements, weighted toward
value-tilted (high book-to-market) repurchasers where the undervaluation
signal is strongest per the original study; hold multiple years as the
market slowly incorporates the repurchase's informational content, since
the market is documented to underreact to the announcement rather than
overreact. Event-driven single-name signal, distinct from the
cross-sectional book-to-market ranking of value_factor_book_to_market and
from the earnings-surprise event in pead_earnings_surprise_drift."

Catalog PROXY NOTE -- this family's own catalog entry does not carry
one the way ``merger_arbitrage_risk_arb``'s and ``value_factor_book_to_
market``'s do (its "source" field has no "PROXY NOTE" text), so it is
written here instead, analogous to those two modules' own PROXY NOTE
sections: the backtest engine
only fetches a single ticker's OHLC (``tickers[0]``), so this event-driven,
cross-sectional, value-tilted repurchase strategy is tested via PKW
(Invesco BuyBack Achievers ETF, issuer Invesco, real/verified) as a real, tradeable stand-in basket. PKW tracks the
NASDAQ US BuyBack Achievers Index, which requires a constituent to have
effected a net reduction in shares outstanding of 5% or more over the
trailing 12 months, and is float-adjusted-market-cap weighted with a
semi-annual (Feb/Aug) reconstitution (verified against the index methodology)
-- a real, continuously-maintained basket of buyback-heavy issuers, not a
literal per-announcement event study with a value tilt and a 4-year hold.
So a pass/fail here says something about "a real buyback-basket product's
daily-bar behavior", not a direct replication of Ikenberry, Lakonishok &
Vermaelen's original event-study construction. Any later use of a
passing result would need to decide whether the ETF-proxy result is itself
sufficient or whether the literal per-announcement, value-tilted,
multi-year-hold construction needs separate validation. Catalog
``tickers`` are ``["PKW", "screened equity universe (buyback-announcement
ranked)"]`` -- the second entry is a generic label for the unimplemented
cross-sectional universe, not a second real ticker (same convention as
``value_factor_book_to_market.py``'s own VLUE/"screened equity universe"
disclosure); it is not treated as one and is
not part of TRADED/SIGNAL below.

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``value_factor_book_to_market.py``'s and
``free_cash_flow_yield_cows.py``'s own disclosures of their
simplifications): a real point-in-time, per-announcement event study
(buy on repurchase announcement, tilt toward high book-to-market
repurchasers, hold ~4 years, reformed as new announcements arrive) is NOT
implemented here -- this repo's backtest engine is single-ticker, not an
event-driven portfolio simulator capable of entering/exiting individual
names on their own announcement dates. PKW is the real, already-constructed
ETF that runs a continuous, index-level version of "own a basket of
issuers that are aggressively buying back stock" (its own semi-annual
reconstitution is what already implements "periodic reformation" here),
so buy-and-hold PKW (since its own 2006-12-20 inception) is used as a
DIRECT substitute for simulating the
per-announcement screen from scratch -- same substitution shape as
``value_factor_book_to_market.py``'s VLUE-for-a-decile-sort and
``free_cash_flow_yield_cows.py``'s COWZ-for-a-quarterly-screen. TWO
catalog parameters are NOT literally reproduced by this substitution and
are disclosed rather than silently reconciled:

* ``PARAMETERS["hold_years"] == 4`` describes a PER-ANNOUNCEMENT holding
  period in the original event-study construction (buy at announcement,
  hold that name ~4 years, then roll into the next announcement). PKW has
  no such per-position hold-then-exit mechanic -- its own semi-annual
  reconstitution turns individual constituents over on the issuer's index
  schedule, not on a 4-year-per-name clock. The number is preserved
  byte-identically in PARAMETERS below (it is what the catalog says), but
  it does not correspond to any timer this module implements.
* ``PARAMETERS["value_tilt_filter"] == "high_book_to_market"`` describes a
  tilt toward value-priced repurchasers that PKW's own index methodology
  does NOT apply -- PKW's real selection rule is a trailing-12-month net
  share-count reduction of >=5%, float-adjusted-market-cap weighted, with
  no book-to-market screen at all (verified, see
  PROXY NOTE above). Same shape as ``value_factor_book_to_market.py``'s own
  disclosure that VLUE "tracks a broad value-factor INDEX ... not a literal
  book-to-market decile sort" -- here the value tilt described by the
  catalog mechanism is simply absent from the real proxy instrument, not
  approximated by it.

PKW's own semi-annual reconstitution already implements "periodic
reformation" at the issuer level (``PARAMETERS["hold_years"]`` describes
the ORIGINAL per-name event-study design, not PKW's own mechanics). There
is therefore no per-bar decision left for a signal function to make; the
testable hypothesis reduces to "does buy-and-hold PKW clear the mining
pipeline's bars" (own-instrument alpha, ABI-1 excess over SPY, etc.), not
a market-timing or per-announcement-entry question.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even PKW's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long PKW, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical shape to ``value_factor_book_to_market.
py``'s and ``free_cash_flow_yield_cows.py``'s own choices for the same
always-long shape -- restated here rather than imported, since each
batch5 module is self-contained per this pipeline's convention):

1. "Always long, never exited by rule" is implemented as entry=True /
   exit=False on EVERY bar the engine asks about, not just the first --
   ``backtest_engine.run_backtest`` reads ``entry`` only while flat and
   ``exit`` only while long, so a constant True entry is simply ignored
   once a position is open.
2. The engine enters at the EARLIEST bar it can structurally reach:
   ``run_backtest`` first asks about bar 0's decision while flat (the
   call ``signal_fn(df.iloc[:1], 0)`` made when evaluating bar 1), so
   that very first check already returns entry=True and the position
   opens at bar 1's open -- one bar of unavoidable engine latency common
   to every candidate in this pipeline, not a modeling choice made here.
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is taken
   completely literally: the position is closed only by the engine's own
   end_of_data path (or an external time_stop the driver may separately
   impose), never by this module raising `exit`. In particular, the
   catalog's ``hold_years: 4`` is NOT wired in as a time_stop here -- see
   the DISCLOSED SIMPLIFICATION section above for why that parameter does
   not correspond to any timer PKW's own mechanics run.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's end_of_data path
   at the bucket's last bar -- so each bucket the driver runs this module
   over (in-sample, holdout, or ABI-1's full history) produces exactly one
   trade. This follows from the structure alone: a pure buy-and-hold
   rule cannot produce more than one trade per contiguous bucket.
5. PKW's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "buyback_yield_repurchase_anomaly"
SHAPE = "single_long"
TRADED = ["PKW"]
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
# for family_id "buyback_yield_repurchase_anomaly" (driver drift check;
# also verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)). See the DISCLOSED SIMPLIFICATION section
# above for why "value_tilt_filter" and "hold_years" are preserved here
# verbatim but do not correspond to any mechanic this module implements.
PARAMETERS = {
    "signal": "open_market_repurchase_announcement",
    "value_tilt_filter": "high_book_to_market",
    "hold_years": 4,
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
