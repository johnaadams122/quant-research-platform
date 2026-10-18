"""Batch5 candidate: hedge_fund_13f_best_ideas (single_long).

Hedge Fund High-Conviction 13F Copycat: hold Global X Guru Index ETF (GURU)
outright. Source: Cohen, Polk, Silli, "Best Ideas" (SSRN 1364827), finding
that hedge funds' highest-conviction holdings outperform the market and the
rest of the same manager's own portfolio. GURU (Global X Guru Index ETF,
issuer Global X, launched 2012-06-04) tracks the Solactive Guru Index, which
is built directly from a curated panel of hedge funds' Form 13F filings:
each quarter it selects the largest, most-concentrated ("best idea")
positions across the panel and reconstitutes the basket as new filings are
processed, honoring the SEC's mandatory 45-day disclosure lag (see the
pre-registered candidate's "source" field,
data/mining/catalog.json, family_id "hedge_fund_13f_best_ideas").

Same shape as faith_based_sin_stock_screen: the 13F-copycat construction
and quarterly reconstitution are GURU's own internal mechanism, not
something this module computes, so the testable hypothesis is buy-and-hold
GURU vs the benchmark -- always entered, never exited by rule. The rule
this module implements is:

    Always long GURU: enter as soon as the engine's own next-open-fill
    convention allows (the first bar the engine evaluates while flat), and
    exit NEVER by rule -- the position is closed only by the engine's own
    end_of_data force-close. No per-bar decision is made from GURU's own
    price history, because the quarterly reconstitution / 45-day filing lag
    / conviction-ranking selection is Solactive's and Global X's own
    index-construction and portfolio-management process, external to this
    module and to the OHLC bars this signal function is ever shown.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- trivially satisfied: the signal is a constant literal
  (entry=True, exit=False on every bar the engine asks about), computed
  from no price data at all, so it cannot depend on a future value it never
  reads in the first place.
* PERFORMANCE -- the returned signal_fn does one bounds check plus one
  scalar date-consistency check and returns a fixed SignalResult; nothing
  is recomputed per bar or over a growing slice.
* HOLDOUT-BLIND -- the traded frame is still truncated to <= the bucket's
  last bar inside the factory, for parity with every other batch5 module
  and so a future edit that made the signal data-dependent would fail the
  same look-ahead-safe way its siblings do -- not because today's constant
  rule needs any of that data.
* DIRECTION -- always long GURU, never short, never flat by rule.

Interpretation choices (the catalog entry leaves these to the implementer):

1. "Reconstituted each quarter as new 13F filings are processed" and the
   "45-day disclosure lag" are GURU's/Solactive's own index-construction
   rules, executed by the ETF issuer -- not a computation performed here on
   GURU's own daily bars. There is no separate signal ticker (SIGNAL = [])
   and no rebalance-date arithmetic in this module. This mirrors the batch5
   ``faith_based_sin_stock_screen`` precedent: a fund whose entire active
   mechanism lives inside the product itself is tested as a pure
   buy-and-hold of that product, not reimplemented from the underlying
   13F/panel-selection process.
2. "Always entered, never exited by rule" is implemented literally: entry
   is True on every bar the engine evaluates while flat (so the position
   opens at the earliest bar the engine's next-open-fill convention
   permits) and exit is always False (so an open position is closed only
   by the engine's own end_of_data path, never a signal-driven exit). No
   stop, no time stop, no rebalancing logic, no cash arm.
3. A bar with a non-finite or missing Close (or any other column) is not
   specially handled, because the signal never reads price data to make
   its decision -- entry/exit do not depend on price at all. Data quality
   for GURU is the driver's own OHLC-loading concern
   (batch3._normalize_ohlc), not this module's.
4. GURU's inception is 2012-06-04 (catalog "source" field); bars before inception simply do not exist in
   ``frames`` and are never presented to ``signal_fn`` by the engine, so no
   explicit "before inception" branch is needed. If a bucket's history has
   no GURU bars at or before its own last date at all (a data-coverage
   problem, not a rule question), the factory fails closed rather than
   emitting signals with no underlying coverage -- see
   ``test_frame_with_no_bars_at_or_before_bucket_end_raises``.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "hedge_fund_13f_best_ideas"
SHAPE = "single_long"
TRADED = ["GURU"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "GURU"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" for this
# family_id (driver drift check via
# test_parameters_are_byte_identical_to_frozen_catalog_entry). Do not
# invent, rename, or add parameters.
PARAMETERS = {
    "rebalance": "quarterly",
    "filing_lag_days": 45,
    "selection": "top_conviction_holdings_across_hedge_fund_panel",
}
# A pure buy-and-hold opens exactly one position (at the earliest bar the
# engine can fill) and never closes it by rule -- any bucket with >= 2 bars
# therefore produces EXACTLY one trade (closed only by the engine's own
# end_of_data path at the bucket's final bar), never more, never fewer once
# there are 2+ bars. EXPECTED_MIN_TRADES = 1 is the honest, exact floor for
# this shape; GURU's history since inception spans thousands of bars,
# far more than the 2 needed for that single trade to exist.
EXPECTED_MIN_TRADES = 1


def _normalize_index(index) -> pd.DatetimeIndex:
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
    will run over; ``frames`` holds full-history OHLC for TRADED (SIGNAL is
    empty for this family, so ``frames`` carries only the GURU key). The
    returned closure is a CONSTANT -- this family has no per-bar decision
    (see module docstring, interpretation choice 2)."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    traded_frame = _require_frame(frames, ticker)

    bucket = _normalize_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    # HOLDOUT-BLIND parity with every other batch5 module (see docstring
    # Design section): truncate the traded frame's dates to <= the bucket's
    # last bar before the factory returns, even though the constant signal
    # below reads none of it. This also gives a real, honest failure mode
    # (below) when a bucket has no GURU coverage at all.
    traded_dates = _normalize_index(traded_frame.index)
    traded_dates = traded_dates[traded_dates <= last_bar].unique().sort_values()
    if len(traded_dates) == 0:
        raise ValueError(f"{FAMILY_ID}: {ticker} has no bars on or before "
                         f"{last_bar.date()}")

    n_bucket = len(bucket)
    bucket_values = bucket.asi8  # int64 ns for an O(1) scalar date check

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
