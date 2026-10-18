"""Batch5 candidate: quality_minus_junk_factor (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json`` under
the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instrument is re-tuned here.

Catalog mechanism (verbatim, ``data/mining/catalog.json``, family_id
"quality_minus_junk_factor"): "Rank equities by a composite quality score
(profitability, earnings growth, balance-sheet safety/low leverage, low
earnings volatility, high payout); long the highest-quality decile (short
or underweight the lowest, 'junk', decile in a long/short construction).
Quality's premium is attributed to persistent mispricing/behavioral
underpricing of safe, profitable firms, not the leverage-constraint
mechanism behind betting_against_beta_low_vol already in this catalog --
the two factors are empirically correlated but mechanistically distinct in
the source literature." Source (verbatim from the same catalog entry):
Asness, C.S., Frazzini, A. & Pedersen, L.H. (2019, orig. working paper
2013), "Quality Minus Junk", Review of Accounting Studies 24(1), 34-112.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432. Catalog
status "eligible".

Catalog ``tickers``: ``["QUAL", "screened equity universe
(quality-ranked)"]``. Only ``tickers[0]`` (QUAL, iShares MSCI USA Quality
Factor ETF) is a real, tradeable instrument -- ``tickers[1]`` is a generic
label for the cross-sectional universe the mechanism describes, not a
second real ticker, and is not treated as one here.

DISCLOSED SUBSTITUTION: ``mining.backtest_engine.run_backtest`` drives exactly ONE
ticker's OHLC frame through a binary long/flat signal -- there is no
cross-sectional universe here from which to actually rank stocks into
deciles or fund a short "junk" leg. QUAL is used as the single-ticker
long-only proxy for "the top-quality-decile portfolio" itself: its index
selects US large/mid-cap stocks on high ROE, stable earnings growth, and
low leverage -- the same profitability/growth/safety composite
``PARAMETERS["quality_score"]`` names. The short/underweight "junk" leg
described in the mechanism is NOT implemented (no proxy ticker for it
exists in this engine's single-ticker contract) -- disclosed, not
silently approximated. This module is therefore a long-only
quality-decile read, not the full long/short QMJ construction.

DISCLOSED SIMPLIFICATION (the annual rebalance is not gated): like
``dividend_growth_aristocrats`` and ``value_factor_book_to_market`` (also
always-long single-proxy modules with an annual-rebalance mechanism
parameter), this module uses a bare constant ``entry=True, exit=False`` on
every bar, matching ``spinoff_outperformance_effect.py`` and
``wide_moat_quality_investing.py``, rather than gating entry on the first
trading day of each new calendar year. For this engine the two are
equivalent in the outcome that matters -- QUAL is bought once and held to
end_of_data either way, since neither form ever raises ``exit`` -- because
an annual-rebalance gate would only affect WHEN the position first opens,
not whether more than one trade can result. The bare-constant form opens
at the earliest structurally reachable bar, never later than the first
rebalance day, so it cannot manufacture a trade that would not otherwise
occur. ``PARAMETERS["rebalance"] == "annual"`` is preserved byte-for-byte
below as the catalog's own recorded value (the source-of-truth parameters
are never edited here), even though signal_fn does not gate on it -- the
same relationship every other always-long batch5 module has to a catalog
``rebalance`` field that describes the proxy ETF's or the named screen's
own mechanics, not a rule this module computes (see e.g.
``wide_moat_quality_investing.py``'s identical treatment of
"semiannual_staggered").

Design (the four non-negotiable properties, all trivially satisfied by a
constant rule -- same as every other pure buy-and-hold batch5 candidate):

* NO LOOK-AHEAD -- the signal never reads a price at all (not even QUAL's
  own), so there is nothing for a future bar to leak into a past decision.
* PERFORMANCE -- the returned closure returns a fixed SignalResult;
  nothing is computed per bar, let alone recomputed over a growing slice.
* HOLDOUT-BLIND -- likewise vacuous: with zero data dependency, truncating
  ``frames`` to <= the bucket's last bar cannot change anything the
  signal does.
* DIRECTION -- always long QUAL, entered at the first structural
  opportunity and never exited by rule.

Interpretation choices (identical to ``spinoff_outperformance_effect.py``'s
and ``wide_moat_quality_investing.py``'s own choices for the same
always-long shape -- restated here rather than imported, since each
batch5 module is self-contained per this pipeline's convention):

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
   exactly one trade (an all-blank trailing row for a still-forming
   session is dropped by the same ``run_batch5_mining._drop_all_nan_rows``
   cleaning step the real driver applies before normalizing).
   EXPECTED_MIN_TRADES is set to exactly that structural floor (1) --
   there is no safety margin to set below 1.
5. QUAL's own frame is required to be present in `frames` (fail closed if
   missing, matching every other module's convention) even though its
   prices are never read by the signal -- this keeps the factory's
   contract identical to every other candidate's and gives a clear error
   if the driver's data plumbing ever fails to supply it, rather than
   silently trading blind.
"""
from __future__ import annotations

import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "quality_minus_junk_factor"
SHAPE = "single_long"
TRADED = ["QUAL"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK comment and spinoff_outperformance_effect.py's /
# wide_moat_quality_investing.py's own identical disclosures for this same
# shape. This family's catalog entry does not even carry its own
# "benchmark" parameter key, so there is no coincidental agreement to note
# either way.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "quality_minus_junk_factor" (driver drift check; also
# verified byte-identical by this module's own test via
# json.dumps(sort_keys=True)). "rebalance": "annual" is preserved as
# recorded even though this module's own signal_fn does not gate on it
# -- see the module docstring's "DISCLOSED SIMPLIFICATION" section.
PARAMETERS = {
    "quality_score": "profitability_growth_safety_composite",
    "rank_decile": "top",
    "rebalance": "annual",
}
# Structural floor, not a measured count (docstring "Interpretation choices"
# item 4): a pure buy-and-hold rule with no rule-driven exit structurally
# cannot produce more than one trade per contiguous bucket.
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
    entry=True, exit=False -- literal buy-and-hold, see the module
    docstring's "DISCLOSED SIMPLIFICATION" section for why the catalog's
    annual rebalance is not gated."""
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
