"""Batch5 candidate: fallen_angel_bond_downgrade_drift (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json``
under the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instrument is re-tuned here.

An alternative reading of this family is a monthly-rebalance entry that
(re-)opens a position only when currently flat and the calendar month has
rolled over, closed by the *engine's own*
``time_stop=holding_period_years*252`` argument at the run_backtest call
site (i.e. reading ``PARAMETERS["holding_period_years"]`` as an
ENGINE-ENFORCED per-position exit, re-entering monthly after each 3-year
rotation). This module deliberately does NOT use that shape -- see
"Design decision, and why there is no engine time stop" below.

Source (pre-registered catalog entry, ``data/mining/catalog.json``,
family_id "fallen_angel_bond_downgrade_drift"): Ben Dor, A., Xu, Z.
(2011), "Fallen Angels: Characteristics, Performance, and Implications for
Investors," Journal of Fixed Income, 20(4), 33-58. Per the catalog
entry's own "source" field: VanEck Fallen
Angel High Yield Bond ETF (ANGL), issuer VanEck, launched April 10, 2012,
confirmed actively trading. Catalog note: "this is a credit-rating-
downgrade event on corporate BONDS, not an equity anomaly ... a genuine
corporate/credit-action anomaly (forced-selling overreaction on a rating
downgrade) with no overlap with any other candidate."

Catalog mechanism (verbatim, ``data/mining/catalog.json``): "When a
corporate bond issuer is downgraded from investment-grade to high-yield
("fallen angel" status), forced selling by investment-grade-only mandates
and insurers pushes the bond's price below fair value; documented research
shows an initial post-downgrade dip followed by a multi-year reversal/
recovery that outperforms both the broad high-yield index and newly-issued
junk bonds of similar credit quality. The strategy holds a continuously-
rebalanced basket of these recently-downgraded issuers, proxied by an ETF
reconstituted monthly to add newly-downgraded names and remove issuers
once upgraded back to investment grade or that default."

DISCLOSED SIMPLIFICATION (matching this pipeline's own established honesty
convention -- see e.g. ``spinoff_outperformance_effect.py``'s and
``free_cash_flow_yield_cows.py``'s own disclosures of their limitations):
a real event-driven screen that identifies each individual downgraded
bond issuer, tracks it from downgrade through the documented multi-year
recovery, and continuously adds/removes names from a basket is NOT
implemented here -- this repo's backtest engine is single-ticker, not a
portfolio/basket simulator capable of that kind of continuous
reconstitution. ANGL is the real, already-constructed ETF whose own index
methodology already runs the described mechanism internally: it adds
newly-downgraded issuers and removes issuers "once upgraded back to
investment grade or that default", on an ongoing monthly-reconstituted
basis (``PARAMETERS["rebalance"] == "monthly"``), so buy-and-hold ANGL
(since its own real, fetched inception) is used as a DIRECT substitute for
simulating the screen from scratch -- the same substitution
``spinoff_outperformance_effect.py`` makes for CSD.

Design decision, and why there is no engine time stop:
``PARAMETERS["holding_period_years"] == 3`` here documents the SOURCE
RESEARCH's own finding (Ben Dor & Xu 2011: the documented reversal/
recovery plays out over a multi-year window) and the catalog's
own ``benchmark`` field for comparing that finding -- it is NOT read as an
engine-enforced per-position exit the way the alternative reading above
would read an explicit holding-period parameter. The reason is the same reason
``spinoff_outperformance_effect.py`` does not implement CSD's own
``holding_period_years=4`` as an engine time_stop either: ANGL's index
methodology already handles per-issuer entry and exit timing internally
(monthly reconstitution; removal "once upgraded back to investment grade
or that default", not on a fixed calendar), so re-implementing a
position-level 3-year rotation on top of that would be double-counting a
rule the traded instrument already enforces on its own underlying
holdings -- exactly the same "already-managed basket" argument this
pipeline's other single-proxy buy-and-hold candidates (CSD, COWZ, MOAT)
already establish. This module follows that batch5 single-proxy
convention (buy-and-hold the real, already-managed proxy: always long,
entering on the very first bar and never exiting) rather than an
engine-time-stop re-entering shape -- a deliberate, disclosed choice, not
an oversight.

CSD -- the traded ticker's own sibling precedent -- already IS the
continuously-refreshed, already-managed basket; ANGL is the same shape for
fallen-angel bonds: the issuer runs the add/remove-on-upgrade-or-default
rule internally. There is therefore no per-bar decision left for a signal
function to make; the testable hypothesis reduces to "does buy-and-hold
ANGL clear the mining pipeline's bars" (own-instrument alpha, ABI-1 excess
over SPY, etc.), not a market-timing or position-rotation question. Note
also: unlike ``spinoff_outperformance_effect.py`` (whose catalog
"benchmark" happens to equal ABI-1's hard SPY rule), this family's own
catalog ``parameters["benchmark"]`` is "HYG" -- genuinely DIFFERENT from
ABI-1's hard SPY rule (see BENCHMARK below); PARAMETERS still stores the
catalog's own "HYG" value byte-for-byte since PARAMETERS is a faithful
copy of the catalog record, not a restatement of what this driver scores
against.

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even ANGL's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long ANGL, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to ``spinoff_outperformance_effect.py``'s
own choices for the same always-long shape -- restated here rather than
imported, since each batch5 module is self-contained per this pipeline's
convention):

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
3. No time stop, no signal exit, no cash state -- "buy-and-hold" is taken
   completely literally: the position is closed only by the engine's own
   end_of_data path (or an external time_stop the driver may separately
   impose), never by this module raising `exit`.
4. EXPECTED_MIN_TRADES = 1: with entry always True and exit always False,
   exactly one continuous holding period opens (at the bucket's first
   reachable bar) and is only ever closed by the engine's own end_of_data
   path at the bucket's last bar -- so each bucket the driver runs this
   module over (in-sample, holdout, or ABI-1's full history) produces
   exactly one trade. This module's own test checks it end to end on a
   synthetic fixture.
5. ANGL's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "fallen_angel_bond_downgrade_drift"
SHAPE = "single_long"
TRADED = ["ANGL"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK ("ABI-1 always compares the account to SPY -- NOT a
# family's own catalog benchmark/BENCHMARK field") and
# spinoff_outperformance_effect.py's own identical disclosure for this
# same shape. Unlike that sibling, this family's own catalog "benchmark"
# ("HYG") is genuinely DIFFERENT from SPY -- BENCHMARK below is still
# hard-pinned to "SPY" per the pipeline rule, not copied from the catalog.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "fallen_angel_bond_downgrade_drift" (driver drift check;
# also verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)). "holding_period_years" and "rebalance" are
# stored as-is from the catalog record; see the module docstring's "Design
# decision" section for why they are NOT read as an engine-level time_stop
# by make_signal_fn below.
PARAMETERS = {
    "benchmark": "HYG",
    "holding_period_years": 3,
    "rebalance": "monthly",
}
# Structural floor: a pure buy-and-hold rule enters once and holds to
# end_of_data (see "Interpretation choices" item 4 above), so it
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
