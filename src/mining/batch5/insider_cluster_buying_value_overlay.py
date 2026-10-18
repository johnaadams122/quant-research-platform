"""Batch5 candidate: insider_cluster_buying_value_overlay (single_long).

Insider Cluster Buying with Value Overlay: hold Tweedy, Browne Insider +
Value ETF (COPY) outright. Source: Jeng, Metrick & Zeckhauser (2003),
"Estimating the Returns to Insider Trading", Review of Economics and
Statistics 85(2), 453-471, finding that stocks with recent discretionary
open-market Form 4 insider purchases outperform. COPY (issuer Tweedy
Browne, inception 2024-12-27, actively trading) implements this directly
as a product: it screens for companies with recent discretionary
open-market SEC Form 4 insider purchases (excluding automatic comp-plan
buys), then within that subset further ranks on a multi-factor deep-value
composite (price/book, price/earnings, price/cash-flow), holding names
that pass both filters (see the
pre-registered candidate's "source" field, data/mining/catalog.json,
family_id "insider_cluster_buying_value_overlay").

This is the same shape as this batch's own hedge_fund_13f_best_ideas
precedent: the insider-cluster screen and the deep-value ranking overlay
are COPY's own internal, issuer-side security-selection and
reconstitution mechanism -- not something computed here from price data.
The testable hypothesis is buy-and-hold COPY vs the benchmark -- always
entered, never exited by rule -- so the rule this module implements is:

    Always long COPY: enter as soon as the engine's own next-open-fill
    convention allows (the first bar the engine evaluates while flat), and
    exit NEVER by rule -- the position is closed only by the engine's own
    end_of_data force-close. No per-bar decision is made from COPY's own
    price history, because the Form-4 insider-purchase screen and the
    price/book, price/earnings, price/cash-flow value-composite ranking
    are Tweedy Browne's own security-selection process, external to this
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
* DIRECTION -- always long COPY, never short, never flat by rule.

Interpretation choices (the catalog entry leaves these to the implementer):

1. "Screen for open-market discretionary Form 4 insider purchases" and
   "rank on a deep-value composite" are COPY's/Tweedy Browne's own
   security-selection and portfolio-construction rules, executed by the
   issuer -- not a computation performed here on COPY's own daily bars.
   There is no separate signal ticker (SIGNAL = []) and no per-name
   Form-4/valuation arithmetic in this module, mirroring the batch5
   ``hedge_fund_13f_best_ideas`` precedent: a fund whose entire active
   mechanism lives inside the product itself is tested as a pure
   buy-and-hold of that product, not reimplemented from the underlying
   insider-filing/value-screen process.
2. "Hold names passing both filters" (always-on, no per-bar decision) is
   implemented literally: entry is True on every bar the engine evaluates
   while flat (so the position opens at the earliest bar the engine's
   next-open-fill convention permits) and exit is always False (so an open
   position is closed only by the engine's own end_of_data path, never a
   signal-driven exit). No stop, no time stop, no rebalancing logic, no
   cash arm.
3. A bar with a non-finite or missing Close (or any other column) is not
   specially handled, because the signal never reads price data to make
   its decision -- entry/exit do not depend on price at all. Data quality
   for COPY is the driver's own OHLC-loading concern
   (batch3._normalize_ohlc), not this module's.
4. COPY's inception is 2024-12-27 (catalog "source" field; disclosed
   SHORT-HISTORY CAVEAT: a short live history, materially shorter than
   most catalog tickers -- may not clear the pipeline's own
   MIN_TRADES_FOR_SCORING or provide a meaningful holdout split. The
   pipeline's own eligibility gating decides this, not a pre-judgment
   here -- this module is built and tested normally, with no padding or
   artificial extension of anything to compensate). Bars before inception
   simply do not exist in ``frames`` and are never presented to
   ``signal_fn`` by the engine, so no explicit "before inception" branch
   is needed. If a bucket's history has no COPY bars at or before its own
   last date at all (a data-coverage problem, not a rule question), the
   factory fails closed rather than emitting signals with no underlying
   coverage -- see ``test_frame_with_no_bars_at_or_before_bucket_end_raises``.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "insider_cluster_buying_value_overlay"
SHAPE = "single_long"
TRADED = ["COPY"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "COPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" for this
# family_id (driver drift check via
# test_parameters_are_byte_identical_to_frozen_catalog_entry). Do not
# invent, rename, or add parameters.
PARAMETERS = {
    "insider_filter": "open_market_discretionary_purchase_not_comp_plan",
    "secondary_filter": "deep_value_composite_score",
}
# A pure buy-and-hold opens exactly one position (at the earliest bar the
# engine can fill) and never closes it by rule -- any bucket with >= 2 bars
# therefore produces EXACTLY one trade (closed only by the engine's own
# end_of_data path at the bucket's final bar), never more, never fewer once
# there are 2+ bars. EXPECTED_MIN_TRADES = 1 is the honest, exact floor for
# this shape. Unlike GURU's thousands of in-sample bars, COPY's history
# is short (inception 2024-12-27) -- still
# far more than the 2 bars needed for that single trade to exist, but likely
# too short for the pipeline's own MIN_TRADES_FOR_SCORING / holdout-split
# eligibility gates, which this module deliberately does not pre-judge.
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
    empty for this family, so ``frames`` carries only the COPY key). The
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
    # (below) when a bucket has no COPY coverage at all.
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
