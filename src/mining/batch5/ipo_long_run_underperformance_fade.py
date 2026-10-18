"""Batch5 candidate: ipo_long_run_underperformance_fade (bespoke driver;
run only via scripts/run_ipo_fade_abi1.py, never the generic
scripts/run_abi1_mining.py dispatch -- see "SHAPE and dispatch" below).

Implements the catalogued mechanism from ``data/mining/catalog.json``
under the ABI-1 batch5 module contract. The parameters and the traded
instrument are the catalog entry's; the mechanism's simplification is
described under "DISCLOSED SIMPLIFICATION" below.

POINTER: the two problems documented below (the -100% floor and
the SPY-beta confound) are addressed by a DIFFERENT strategy definition, the
beta-neutral benchmark-plus-overlay construction ``ipo_long_run_underperformance_
fade_beta_neutral`` (own family_id, own module, own bespoke driver
``scripts/run_ipo_fade_beta_neutral_abi1.py``). This naked-short module and its driver are
the simple primitive that shows why that construction is needed.

Catalog mechanism (verbatim, ``data/mining/catalog.json``, family_id
"ipo_long_run_underperformance_fade"): "Newly public companies tend to
underperform the broad market and matched non-issuing peers over the
one-to-three years following their IPO, once the initial first-day pop
and post-IPO 'honeymoon' period fade -- largely because issuers and
underwriters time offerings to windows of investor over-optimism. The
strategy takes a SHORT position in a continuously-refreshed basket of
recent U.S. IPOs, proxied by an ETF that adds new offerings on a
fast-entry basis shortly after listing and removes each holding after
roughly two years, harvesting the documented long-run underperformance
spread versus the broad market."

Source (pre-registered catalog entry): Ritter, J.R. (1991), "The
Long-Run Performance of Initial Public Offerings," Journal of Finance,
46(1), 3-27 -- documents IPOs returning 34.5% vs. 61.9% for a matched
non-issuing control sample over the 3 years from first-day close,
1975-84. Renaissance IPO ETF (IPO), issuer Renaissance Capital,
inception October 14, 2013 (catalog's own "source" field).

SHAPE and dispatch
-------------------
``SHAPE = "single_inverse_product"`` -- this module's own INTERPRETATION
CHOICE, disclosed explicitly rather than picked silently: of the six
registered shapes in ``scripts/run_batch5_mining.py`` (``single_long``,
``single_inverse_product``, ``rotation_two_asset``,
``pooled_single_asset``, ``panel_breadth``, ``event_pooled``), none
literally describes "hold one plain (non-inverse) instrument, directly
SHORT, forever" -- ``single_inverse_product`` is reserved for a
single-instrument BEARISH-exposure family, which is the closest real
category fit even though this family's actual mechanism shorts IPO
DIRECTLY rather than holding a real inverse ETF long. ``single_long``
was rejected as strictly more misleading (it names the wrong direction
outright). This choice carries NO behavioral consequence in this
module's own code -- ``make_signal_fn`` below never encodes direction
either way (see "Interpretation choices" item 2) -- so it is purely a
contract-shape label, matching ``retail_inventory_sales_breadth_short.
py``'s own precedent of a real registered SHAPE value under a family
that is always run by its own bespoke driver, never generic dispatch.

DISPATCH-SAFETY GUARD, code-level: ``make_signal_fn`` requires an
explicit ``driver_acknowledges_short_direction=True`` keyword, defaulting
to False. ``scripts/run_abi1_mining.py``'s generic SINGLE_SHAPES dispatch
calls ``make_signal_fn`` with EXACTLY 3 positional args (``module.
make_signal_fn(symbol, full_frame.index, module_frames)``, no keywords),
so that exact call shape raises a clear ``RuntimeError`` naming the
real problem (no ``direction`` passthrough in that dispatch, would
silently build a LONG position instead of SHORT) instead of silently
producing a plausible-looking but wrong-direction result. Only this
family's own bespoke driver (``scripts/run_ipo_fade_abi1.py``) passes the
acknowledgment flag. This mirrors retail_inventory_sales_breadth_short.
py's own ``fundamentals_map=None``-defaults-to-inert pattern in INTENT
(a wrong invocation must be harmless, never a silent wrong answer) but
differs in mechanism, since this family (unlike that one) has no natural
data input whose absence would make the signal inert -- an explicit,
purpose-built acknowledgment flag is the more honest fit than
overloading an unrelated parameter. The guard is defense in depth: it
holds even if a future change adds this family_id to the generic
dispatch's accepted scope without realizing the bespoke driver is
still required.

DISCLOSED SIMPLIFICATION (same reasoning already applied to every
buy-and-hold sibling in this directory, e.g. ``value_factor_book_to_
market.py`` and ``spinoff_outperformance_effect.py``): the catalog's own
``PARAMETERS`` describe "hold multiple years, exit after ~2 years,
continuous refresh" -- i.e. a rolling ~2-year holding period, not a
single position held forever. This module reduces that to the simpler
"short-and-hold forever" rule (constant entry=True/exit=False, direction
handled entirely by the driver -- see "Interpretation choices" below),
for the SAME reason every sibling buy-and-hold module gives: IPO's own
real index methodology ALREADY implements the "remove each holding after
roughly two years" reformation internally (Renaissance's own published
methodology continuously adds new listings and retires ~2-year-old
ones) -- the issuer, not this signal, is what already implements
periodic reformation. There is therefore no per-bar decision left for a
signal function to make; the testable hypothesis reduces to "does a
continuous short position in this continuously-refreshed basket clear
the mining pipeline's bars" (own-instrument-vs-SPY alpha, ABI-1 excess
over SPY, etc.), not a market-timing question.

FIRST PROBLEM, THE -100% FLOOR: over a sample dominated by a strong
bull market, the IPO basket's own absolute price can rise a lot even
while it underperforms SPY in relative terms. A naive, un-hedged,
continuously-held SHORT position on IPO itself then shows a LARGE
NOMINAL LOSS under the engine's simple one-share-P&L short convention
(``backtest_engine.run_backtest``'s own docstring: "Short gross return
is simple one-share P&L on entry capital"), which is NOT compounded and
can go below -100%. Such a loss does NOT by itself contradict the
academic finding, which is about RELATIVE underperformance versus the
broad market (Ritter 1991), not about the basket's absolute direction.

SECOND, DEEPER PROBLEM (ABI-1's SPY-excess scoring alone is NOT the
correct measure for the relative question): a raw, un-hedged,
single-instrument short does not merely
express market beta "first" alongside the relative-underperformance
signal -- for ABI-1's own scoring construction specifically, it
CONFOUNDS the two. ABI-1 scores an account's DAILY excess return over
SPY's own daily return. For every OTHER single-instrument family in this
pipeline, that account is LONG the traded instrument -- SPY-excess there
genuinely isolates "does holding X beat holding SPY," which is exactly
the intended comparison. For a NAKED SHORT of a single equity-like,
market-correlated instrument, the account's own daily return is
approximately the NEGATIVE of that instrument's daily return -- so
SPY-excess there measures "does shorting X beat holding SPY," which is
dominated by the account's own NEGATIVE overall equity-market beta
exposure (a naked short of almost any broad-market-correlated instrument
looks bad against a long-SPY benchmark during a bull market, and good
during a bear market, REGARDLESS of whether that instrument specifically
underperforms SPY on a relative basis). This is NOT the same question
Ritter (1991)'s documented finding answers, and no fix inside this
module or its bespoke driver corrects it -- a genuine test of the
catalog's actual relative-underperformance thesis would need a real
BETA-NEUTRAL construction (e.g. simultaneously long SPY and short IPO,
sized to net out market exposure), which is a materially different,
larger construction than a single-instrument short-and-hold. This is a
real, disclosed methodology limitation, like the -100%-floor problem
above: under either problem a score of this naked-short form does not
answer the relative-underperformance question, so this module is the
naked-short primitive, and the beta-neutral sibling named in the POINTER
above is the construction that addresses both.

FOLLOW-ON: a net return below the engine's -100% floor makes
``account_relative.netted_tax_factors`` fail closed for this family in
its reduced naked-short form -- see ``scripts/run_ipo_fade_abi1.py``'s
own docstring for the full account. The driver handles that failure
gracefully: a scoring attempt returns a clean, structured skip
(``reason_code="compute_error"``) instead of crashing -- but this does
NOT resolve either problem above; it only makes the FAILURE MODE clean.
A meaningful score of the relative thesis needs BOTH the -100%-floor
problem and the beta-confound problem above resolved, which is what the
beta-neutral construction does.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold/short-and-hold
batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all, so there is
  nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency,
  truncating ``frames`` to <= the bucket's last bar cannot change
  anything the signal does.
* DIRECTION -- always short IPO (enforced by the DRIVER, not this
  module -- see "Interpretation choices" item 2), entered at the first
  structural opportunity and never exited by rule.

Interpretation choices
-----------------------
1. "Short, never exited by rule" is implemented as entry=True / exit=
   False on EVERY bar the engine asks about, not just the first -- same
   "the issuer already implements periodic reformation" reasoning as
   every buy-and-hold sibling (see DISCLOSED SIMPLIFICATION above).
2. Direction is NOT encoded in this module's SignalResult at all --
   matching ``retail_inventory_sales_breadth_short.py``'s own explicit
   convention ("This module's own SignalResult never encodes direction
   ... direction is the DRIVER's job"). ``make_signal_fn`` below is
   IDENTICAL in shape to a plain always-long buy-and-hold module; the
   bespoke driver (``scripts/run_ipo_fade_abi1.py``) is what calls
   ``mining.backtest_engine.run_backtest`` and ``mining.
   benchmark_relative.mark_to_market_daily_returns`` with
   ``direction="short"`` explicitly (see "SHAPE and dispatch" above).
3. The engine enters at the EARLIEST bar it can structurally reach:
   ``run_backtest`` first asks about bar 0's decision while flat (the
   call ``signal_fn(df.iloc[:1], 0)`` made when evaluating bar 1), so
   that very first check already returns entry=True and the position
   opens at bar 1's open -- one bar of unavoidable engine latency common
   to every candidate in this pipeline, not a modeling choice made here.
4. No time stop, no signal exit, no cash state -- "short-and-hold" is
   taken completely literally: the position is closed only by the
   engine's own end_of_data path (or an external time_stop the driver
   may separately impose), never by this module raising `exit`.
5. EXPECTED_MIN_TRADES = 1: with entry always True and exit always
   False, exactly one continuous holding period opens (at the bucket's
   first reachable bar) and is only ever closed by the engine's own
   end_of_data path -- so each bucket the driver runs this module over
   produces exactly one trade. This follows from the structure alone
   (a pure short-and-hold rule cannot produce more than one trade per
   contiguous bucket).
6. IPO's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
7. Issuer inception vs vendor data-start: the catalog cites IPO's
   issuer inception as 2013-10-14; a vendor's daily history may start a
   session or two later -- a normal, small vendor-coverage lag
   (same pattern already disclosed in ``value_factor_book_to_market.py``
   for VLUE), not a data-quality problem.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "ipo_long_run_underperformance_fade"
SHAPE = "single_inverse_product"
TRADED = ["IPO"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK comment and value_factor_book_to_market.py's own
# identical disclosure for this same shape. This family's own catalog
# PARAMETERS happens to also carry a "benchmark": "SPY" key (see below),
# which agrees here but is coincidental, not load-bearing.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "ipo_long_run_underperformance_fade" (driver drift
# check; also verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)).
PARAMETERS = {
    "position": "short",
    "holding_period_years": 2,
    "rebalance": "continuous_index_tracking",
    "benchmark": "SPY",
}
# Structural value: exactly 1 trade per contiguous bucket (short at the
# first reachable bar, held to end_of_data) -- see "Interpretation
# choices" item 5 above. A pure short-and-hold rule cannot produce more
# than one trade per contiguous bucket.
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


def make_signal_fn(ticker: str, index, frames, *,
                    driver_acknowledges_short_direction: bool = False):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED (SIGNAL is
    empty for this family). Returns a closure that always signals
    entry=True, exit=False -- literal short-and-hold, see module
    docstring. Direction is NOT encoded here (see "Interpretation
    choices" item 2) -- the bespoke driver calls the backtest engine and
    the mark-to-market helper with ``direction="short"`` directly.

    DISPATCH-SAFETY GUARD: ``scripts/run_abi1_mining.py``'s generic
    SINGLE_SHAPES dispatch calls this with EXACTLY 3 positional args --
    ``module.make_signal_fn(symbol, full_frame.index, module_frames)`` --
    and that dispatch has no ``direction`` passthrough (always defaults to
    "long"), which would silently score this family in the WRONG direction
    with no error. ``driver_acknowledges_short_direction`` defaults to
    False, so that exact generic-dispatch call shape raises a clear,
    actionable error instead of silently producing a plausible-looking but
    wrong-direction result. Only ``scripts/run_ipo_fade_abi1.py`` (this
    family's own bespoke driver) passes ``driver_acknowledges_short_
    direction=True``."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)  # fail closed if the traded frame is absent
    if not driver_acknowledges_short_direction:
        raise RuntimeError(
            f"{FAMILY_ID}: this family must only be run through its own "
            "bespoke driver (scripts/run_ipo_fade_abi1.py), never the "
            "generic scripts/run_abi1_mining.py dispatch -- that dispatch "
            "has no direction passthrough and would silently score a LONG "
            "position instead of the catalog's SHORT one. Pass "
            "driver_acknowledges_short_direction=True only from that "
            "bespoke driver.")

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
