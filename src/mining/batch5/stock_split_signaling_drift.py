"""Batch5 candidate: stock_split_signaling_drift (single_long).

Stock Split Signaling Drift: hold USCF Stock Split ETF (TOFR) outright.
Source: Ikenberry & Ramnath (2002), "Underreaction to Self-Selected News
Events: The Case of Stock Splits", Review of Financial Studies 15(2),
489-526, finding that the market underreacts to a forward stock split as a
costly, voluntary signal of management confidence. TOFR (USCF ETF Trust's
Stock Split Index Fund, issuer USCF Investments, inception 2014-09-15)
tracked a monthly-rebalanced, equally-weighted index of U.S.-listed
companies that had recently split 2-for-1-or-greater, honoring a
post-split inclusion window before names roll off the index (catalog
"parameters" field; data/mining/catalog.json, family_id
"stock_split_signaling_drift").

This family's shape is single_long, always-on (no per-bar decision), and
it follows ``src/mining/batch5/hedge_fund_13f_best_ideas.py`` as its
template -- the rule this module implements mirrors that precedent
exactly: the monthly reconstitution / post-split inclusion-window roll-off
is TOFR's/Solactive's own index-construction and portfolio-management
process, external to this module and to the OHLC bars this signal
function is ever shown. The testable hypothesis is buy-and-hold TOFR vs
the benchmark -- always entered, never exited by rule:

    Always long TOFR: enter as soon as the engine's own next-open-fill
    convention allows (the first bar the engine evaluates while flat), and
    exit NEVER by rule -- the position is closed only by the engine's own
    end_of_data force-close. No per-bar decision is made from TOFR's own
    price history, because the monthly rebalance / inclusion-window
    selection is USCF's/Solactive's own index-construction process,
    external to this module and to the OHLC bars this signal function is
    ever shown.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- trivially satisfied: the signal is a constant literal
  (entry=True, exit=False on every bar the engine asks about), computed
  from no price data at all, so it cannot depend on a future value it
  never reads in the first place.
* PERFORMANCE -- the returned signal_fn does one bounds check plus one
  scalar date-consistency check and returns a fixed SignalResult; nothing
  is recomputed per bar or over a growing slice.
* HOLDOUT-BLIND -- the traded frame is still truncated to <= the bucket's
  last bar inside the factory, for parity with every other batch5 module
  and so a future edit that made the signal data-dependent would fail the
  same look-ahead-safe way its siblings do -- not because today's constant
  rule needs any of that data.
* DIRECTION -- always long TOFR, never short, never flat by rule.

Interpretation choices (the catalog entry leaves these to the implementer):

1. "Monthly-rebalanced" and "post_split_inclusion_window_months: 6" are
   TOFR's/Solactive's own index-construction rules, executed by the ETF
   issuer -- not a computation performed here on TOFR's own daily bars.
   There is no separate signal ticker (SIGNAL = []) and no rebalance-date
   or inclusion-window arithmetic in this module, mirroring the batch5
   ``hedge_fund_13f_best_ideas`` / ``faith_based_sin_stock_screen``
   precedent: a fund whose entire active mechanism lives inside the
   product itself is tested as a pure buy-and-hold of that product, not
   reimplemented from the underlying split-detection/index-construction
   process.
2. "Always entered, never exited by rule" is implemented literally: entry
   is True on every bar the engine evaluates while flat (so the position
   opens at the earliest bar the engine's next-open-fill convention
   permits) and exit is always False (so an open position is closed only
   by the engine's own end_of_data path, never a signal-driven exit). No
   stop, no time stop, no rebalancing logic, no cash arm.
3. A bar with a non-finite or missing Close (or any other column) is not
   specially handled, because the signal never reads price data to make
   its decision -- entry/exit do not depend on price at all. Data quality
   for TOFR is the driver's own OHLC-loading concern
   (batch3._normalize_ohlc), not this module's.
4. TOFR's catalog-stated inception is 2014-09-15; bars before inception
   simply do not exist in ``frames`` and are never presented to
   ``signal_fn`` by the engine, so no explicit "before inception" branch
   is needed. If a bucket's history has no TOFR bars at or before its own
   last date at all, the factory fails closed rather than emitting
   signals with no underlying coverage -- see
   ``test_frame_with_no_bars_at_or_before_bucket_end_raises``. This is not
   a hypothetical for this family -- see the data-availability finding
   below, which this exact guard clause makes into a real, honest failure
   for every bucket the current data pipeline can produce.

Data-availability finding
-------------------------
The catalog entry's "source" field states TOFR is "actively trading".
That claim does NOT
hold up against a real fetch: ``data_pipeline.fetch_ohlc("TOFR",
start="max")`` under the full-requirements interpreter returns exactly ONE row -- 2017-10-11,
Open = 0.0 (a degenerate placeholder bar, not a plausible traded price) --
and a direct ``yfinance.Ticker("TOFR").history(...)`` call over any
explicit 2014-2017 date range returns zero rows and prints "possibly
delisted; no price data found". A web search of SEC
EDGAR filings for USCF ETF Trust confirms why: the Trust's Board approved
liquidation and closure of the fund on 2017-09-22, TOFR stopped accepting
creation orders 2017-10-10, trading on NYSE Arca ceased before market open
2017-10-13, and the fund's assets were liquidated with proceeds
distributed to shareholders on or about 2017-10-20. TOFR has not traded
since.

Consequence for this module, mechanical and unavoidable: every bucket the
mining driver can construct from real 2026-era data (in-sample or
holdout) has ZERO TOFR bars at or before its last date, so
``make_signal_fn`` below -- via the exact same "no bars at or before the
bucket end" guard this family's own template (``hedge_fund_13f_best_ideas``)
already carries for a fully general reason -- raises ``ValueError`` for
every real call the driver would make. No trade can ever be produced by
this module against real data as the catalog entry is currently written;
this is a delisted-product data gap, not a signal-design defect, and is
NOT something a signal-function change can fix. It is reported as a
blocking finding rather than papered over with a synthetic substitute --
the pre-registered catalog entry's
ticker is FROZEN and this module implements it exactly, so the honest
report is that it cannot be backtested against real data, not a quiet
reinterpretation. EXPECTED_MIN_TRADES = 0 reflects this (see below);
absent the delisting, the always-long constant-signal shape would make
EXPECTED_MIN_TRADES = 1 the honest floor for any bucket with >= 2 bars, as
it is for GURU in this family's own template.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "stock_split_signaling_drift"
SHAPE = "single_long"
TRADED = ["TOFR"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "TOFR"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" for this
# family_id (driver drift check via
# test_parameters_are_byte_identical_to_frozen_catalog_entry). Do not
# invent, rename, or add parameters.
PARAMETERS = {
    "position": "long",
    "rebalance": "monthly",
    "post_split_inclusion_window_months": 6,
}
# Honest floor given the VERIFIED data-availability finding above: TOFR was
# liquidated 2017-10-20 and data_pipeline.fetch_ohlc("TOFR", start="max")
# returns exactly one degenerate bar (2017-10-11, Open=0.0) -- zero real
# bars exist at or before any 2026-era bucket boundary this pipeline would
# construct, so make_signal_fn's own bar-coverage guard raises for every
# real call rather than ever reaching a fillable trade. 0 is the honest
# count of trades this module can produce against real data today; it is
# NOT a claim that the always-long rule itself would only trade once given
# live coverage (see docstring: a bucket with >= 2 real bars would still
# yield exactly 1 trade, same as this family's GURU template) -- there is
# no such bucket to give it.
EXPECTED_MIN_TRADES = 0


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
    empty for this family, so ``frames`` carries only the TOFR key). The
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
    # (below) when a bucket has no TOFR coverage at all -- which, per the
    # data-availability finding above, is EVERY bucket the driver would
    # build against real data today.
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
