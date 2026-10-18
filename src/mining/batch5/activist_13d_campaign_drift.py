"""Batch5 candidate: activist_13d_campaign_drift (single_long).

Activist 13D Campaign Drift: hold LeaderShares Activist Leaders ETF (ACTV)
outright. Source: Brav, Jiang, Partnoy & Thomas (2008), "Hedge Fund
Activism, Corporate Governance, and Firm Performance", Journal of Finance
63(4), 1729-1775, finding that the market slowly reprices target companies
after a hedge fund files a Schedule 13D disclosing 5%+ ownership and a
stated intent to seek board/operational/financial changes. ACTV (issuer
Redwood Investment Management / LeaderShares, inception 2020-10-27)
implemented this directly as a product: an actively-managed basket that
added a company as soon as a new 13D activist-intent filing was identified
against it, holding through the campaign (data/mining/catalog.json,
family_id "activist_13d_campaign_drift"; catalog "source" field verified
via web search 2026-08-29).

**Real-data finding, NOT reflected in the
catalog's "actively trading" claim:** ACTV was liquidated. Per Redwood
Investment Management's own PR Newswire liquidation announcement, the
fund's last day of trading on NYSE Arca was 2025-10-24 and it dissolved on
or about 2025-10-30/31. A real
`data_pipeline.fetch_ohlc("ACTV", start="max")` call returned only 5 daily bars,
2025-10-27 through 2025-10-31, every one an IDENTICAL flat Open/High/
Low/Close of 33.143002 -- consistent with a fund frozen at its final NAV
during liquidation wind-down, not five days of real market trading. There
is no OHLC history at all for the 2020-10-27..2025-10-24 window this
module's own mechanism describes; yfinance (this driver's only OHLC
source) simply does not carry it. **This family cannot be meaningfully
backtested or traded going forward** -- disclosed here, not silently
worked around.

Following the batch5 ``hedge_fund_13f_best_ideas`` and
``faith_based_sin_stock_screen`` precedent -- when an ETF's construction
and reconstitution are the fund's own internal mechanism, the testable
hypothesis is buy-and-hold of the ETF vs the benchmark, always entered,
never exited by rule -- the rule this module implements is:

    Always long ACTV: enter as soon as the engine's own next-open-fill
    convention allows (the first bar the engine evaluates while flat), and
    exit NEVER by rule -- the position is closed only by the engine's own
    end_of_data force-close. No per-bar decision is made from ACTV's own
    price history, because the 13D-filing detection, basket construction,
    and campaign-hold/exit timing are Redwood/LeaderShares' own
    active-management process, external to this module and to the OHLC
    bars this signal function is ever shown.

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
* DIRECTION -- always long ACTV, never short, never flat by rule.

Interpretation choices (the catalog entry leaves these to the implementer):

1. "Adds a company as soon as a new 13D activist-intent filing is
   identified against it, holding through the campaign" is ACTV's/
   Redwood's own active-management rule, executed by the fund manager --
   not a computation performed here on ACTV's own daily bars. There is no
   separate signal ticker (SIGNAL = []) and no filing-date arithmetic in
   this module, mirroring the batch5 ``hedge_fund_13f_best_ideas`` and
   ``faith_based_sin_stock_screen`` precedents: a fund whose entire active
   mechanism lives inside the product itself is tested as a pure
   buy-and-hold of that product, not reimplemented from the underlying
   13D-filing/basket-construction process. The catalog's own
   ``holding_horizon_months: 12`` parameter is likewise the fund's
   internal per-position hold rule (when to roll a name out of the
   basket), not a per-bucket exit this module fires -- the module's own
   holding period is "as long as the engine holds the position", per the
   always-on/never-exit convention below.
2. "Always entered, never exited by rule" is implemented literally: entry
   is True on every bar the engine evaluates while flat (so the position
   opens at the earliest bar the engine's next-open-fill convention
   permits) and exit is always False (so an open position is closed only
   by the engine's own end_of_data path, never a signal-driven exit). No
   stop, no time stop, no rebalancing logic, no cash arm.
3. A bar with a non-finite or missing Close (or any other column) is not
   specially handled, because the signal never reads price data to make
   its decision -- entry/exit do not depend on price at all. Data quality
   for ACTV is the driver's own OHLC-loading concern
   (batch3._normalize_ohlc), not this module's.
4. ACTV's inception is 2020-10-27 (catalog "source" field, verified via
   web search 2026-08-29) and its real trading history ends 2025-10-24
   (liquidation, verified via web search -- see the module
   docstring's "Real-data finding" section above); bars outside that
   window simply do not exist in ``frames`` and are never presented to
   ``signal_fn`` by the engine, so no explicit "before inception" or
   "after liquidation" branch is needed. If a bucket's history has no ACTV
   bars at or before its own last date at all (a data-coverage problem,
   not a rule question), the factory fails closed rather than emitting
   signals with no underlying coverage -- see
   ``test_frame_with_no_bars_at_or_before_bucket_end_raises``.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "activist_13d_campaign_drift"
SHAPE = "single_long"
TRADED = ["ACTV"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "ACTV"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" for this
# family_id (driver drift check via
# test_parameters_are_byte_identical_to_frozen_catalog_entry). Do not
# invent, rename, or add parameters.
PARAMETERS = {
    "position": "long",
    "signal": "new_13d_activist_filing",
    "holding_horizon_months": 12,
}
# A pure buy-and-hold opens exactly one position (at the earliest bar the
# engine can fill) and never closes it by rule -- any bucket with >= 2 bars
# therefore produces EXACTLY one trade (closed only by the engine's own
# end_of_data path at the bucket's final bar), never more, never fewer once
# there are 2+ bars. EXPECTED_MIN_TRADES = 1 is the honest, exact floor for
# this shape -- BUT see the module docstring's "Real-data finding": ACTV's
# real fetched history is only 5 daily bars total (2025-10-27 through
# 2025-10-31, the fund's liquidation window), so while the *rule* still
# guarantees >=1 trade for any bucket with >=2 bars, there may be no bucket
# with real, non-liquidation ACTV coverage at all in this driver's corpus.
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
    empty for this family, so ``frames`` carries only the ACTV key). The
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
    # (below) when a bucket has no ACTV coverage at all.
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
