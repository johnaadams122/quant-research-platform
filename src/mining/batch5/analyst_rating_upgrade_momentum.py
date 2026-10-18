"""Batch5 candidate: analyst_rating_upgrade_momentum (single_long).

Analyst Rating Upgrade Momentum: hold Invesco Bloomberg Analyst Rating
Improvers ETF (UPGD) outright. Source: Womack, K. L. (1996), "Do Brokerage
Analysts' Recommendations Have Investment Value?", Journal of Finance 51(1),
137-167, finding that sell-side analyst recommendation changes carry real
investment value. UPGD (issuer Invesco, inception 2006-05-19) tracks the
Solactive/Bloomberg-style analyst-rating-improvement index, which each
quarter (March/June/September/December) ranks all Bloomberg US Large & Mid
Cap Index members meeting a minimum daily-dollar-volume liquidity floor by
how much their average sell-side buy/hold/sell rating has improved over the
trailing 6- and 12-month windows, and holds an equal-weighted basket of the
50 biggest upgraders (see the pre-registered candidate's "source" field,
data/mining/catalog.json, family_id "analyst_rating_upgrade_momentum").

This is the same shape as the batch5 ``hedge_fund_13f_best_ideas``
module: the analyst-rating-improvement ranking, the
quarterly reconstitution, and the equal-weight basket construction are
UPGD's own internal mechanism, not something computed here. The testable
hypothesis is buy-and-hold UPGD vs the benchmark -- always entered, never
exited by rule -- the rule this module implements is:

    Always long UPGD: enter as soon as the engine's own next-open-fill
    convention allows (the first bar the engine evaluates while flat), and
    exit NEVER by rule -- the position is closed only by the engine's own
    end_of_data force-close. No per-bar decision is made from UPGD's own
    price history, because the analyst-rating scoring / quarterly
    reconstitution / equal-weight-50 basket selection is Solactive's and
    Invesco's own index-construction and portfolio-management process,
    external to this module and to the OHLC bars this signal function is
    ever shown.

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
* DIRECTION -- always long UPGD, never short, never flat by rule.

Interpretation choices (the catalog entry leaves these to the implementer):

1. "Ranked each quarter by trailing 6- and 12-month analyst-rating
   improvement, held as an equal-weighted 50-name basket" is UPGD's/
   Solactive's own index-construction rule, executed by the ETF issuer --
   not a computation performed here on UPGD's own daily bars. There is no
   separate signal ticker (SIGNAL = []) and no ranking/rebalance-date
   arithmetic in this module.
   This mirrors the batch5 ``hedge_fund_13f_best_ideas`` precedent: a fund
   whose entire active mechanism lives inside the product itself is tested
   as a pure buy-and-hold of that product, not reimplemented from the
   underlying analyst-rating/reconstitution process.
2. "Always entered, never exited by rule" is implemented literally: entry
   is True on every bar the engine evaluates while flat (so the position
   opens at the earliest bar the engine's next-open-fill convention
   permits) and exit is always False (so an open position is closed only
   by the engine's own end_of_data path, never a signal-driven exit). No
   stop, no time stop, no rebalancing logic, no cash arm.
3. A bar with a non-finite or missing Close (or any other column) is not
   specially handled, because the signal never reads price data to make
   its decision -- entry/exit do not depend on price at all. Data quality
   for UPGD is the driver's own OHLC-loading concern
   (batch3._normalize_ohlc), not this module's.
4. UPGD's inception is 2006-05-19 (catalog "source" field); bars before inception simply do not exist in
   ``frames`` and are never presented to ``signal_fn`` by the engine, so no
   explicit "before inception" branch is needed. If a bucket's history has
   no UPGD bars at or before its own last date at all (a data-coverage
   problem, not a rule question), the factory fails closed rather than
   emitting signals with no underlying coverage -- see
   ``test_frame_with_no_bars_at_or_before_bucket_end_raises``.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "analyst_rating_upgrade_momentum"
SHAPE = "single_long"
TRADED = ["UPGD"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "UPGD"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" for this
# family_id (driver drift check via
# test_parameters_are_byte_identical_to_frozen_catalog_entry). Do not
# invent, rename, or add parameters.
PARAMETERS = {
    "rank_metric": "consensus_analyst_rating_improvement_6_12mo",
    "basket_size": 50,
    "rebalance": "quarterly",
}
# A pure buy-and-hold opens exactly one position (at the earliest bar the
# engine can fill) and never closes it by rule -- any bucket with >= 2 bars
# therefore produces EXACTLY one trade (closed only by the engine's own
# end_of_data path at the bucket's final bar), never more, never fewer once
# there are 2+ bars. EXPECTED_MIN_TRADES = 1 is the honest, exact floor for
# this shape; UPGD's fetched history spans thousands of bars
# in-sample, far more than the 2 needed for that single trade to exist.
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
    empty for this family, so ``frames`` carries only the UPGD key). The
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
    # (below) when a bucket has no UPGD coverage at all.
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
