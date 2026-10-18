"""Batch5 candidate: em_dm_relative_strength_rotation (rotation_two_asset).

Implements the mechanism recorded in ``data/mining/catalog.json`` under
the ABI-1 batch5 module contract (``scripts/run_batch5_mining.py``). The
mechanism, parameters and traded instruments are exactly the catalog
entry's.

Mechanism (``data/mining/catalog.json``'s real entry for this family_id,
verbatim; Faber, Mebane T. (2010), "Relative Strength Strategies for
Investing," Cambria Investment Management working paper, SSRN
abstract_id=1585517): at the end of each month, compute the trailing
6-calendar-month total return of EEM (iShares MSCI Emerging Markets ETF)
and EFA (iShares MSCI EAFE ETF -- developed ex-US equities). Hold
whichever of the two had the higher trailing return for the following
month, fully invested at all times (no cash/bond fallback, no absolute
trend filter). Rebalance monthly by re-ranking the pair and switching
holdings only when the ranking flips.

By the catalog's own ``parameters`` (``lookback_months=6``,
``num_assets=2``, ``hold_top_n=1``, ``cash_fallback=false``,
``rebalance="monthly"``), the mechanism IS exactly
``signals.make_monthly_relative_momentum_target_fn(risk_ticker="EEM",
defensive_ticker="EFA", lookback_months=6)``: that factory computes
``(close - base) / base`` for BOTH legs and returns whichever ticker's
trailing return is larger -- an exactly-symmetric "hold the leader" rule
with no risk/defensive asymmetry, matching ``num_assets=2`` /
``hold_top_n=1`` / ``cash_fallback=False`` (unlike, say,
``hy_credit_bond_absolute_momentum.py``'s single-instrument ABSOLUTE
momentum filter against a cash proxy, or
``credit_spread_hy_ig_risk_regime.py``'s fixed-polarity regime rule).
Calling the factory with (EEM, EFA) in either argument order produces
IDENTICAL decisions, since the rule only ever compares the two trailing
returns and returns whichever ticker's is larger -- proved directly in
this module's own test file, not merely asserted here. ``TRADED`` is kept
in the catalog's own ``["EEM", "EFA"]`` order for traceability, not
because the order is decision-relevant.

The ``PARAMETERS`` dict and factory call are checked directly against
the ``data/mining/catalog.json`` entry for this family_id (this
module's own test proves byte-identity via
``json.dumps(sort_keys=True)``, never a hand-typed duplicate that could
drift) and against ``mining/signals.py``'s real source for
``make_monthly_relative_momentum_target_fn``
(``src/mining/signals.py:454-479``). Unlike the
``credit_spread_hy_ig_risk_regime.py`` sibling in this same directory,
there is no polarity or argument-order subtlety for this family, and this
module wraps the factory unchanged.

``make_target_fn`` below returns
``signals.make_monthly_relative_momentum_target_fn(TRADED[0], TRADED[1],
lookback_months=6)`` directly, unchanged -- that factory's own
``target(frames: dict[str, pd.DataFrame])`` callable already matches the
batch5 rotation driver's truncated-frames-per-call contract, so no
further wrapping layer is needed (the same "thin wrapper directly
returning a mining.signals factory's own output" convention as
``tips_nominal_breakeven_rotation.py`` / ``hy_credit_bond_absolute_
momentum.py``).

``SIGNAL`` is empty: both EEM and EFA are TRADED legs, so no external
price-frame ticker is needed beyond them. The catalog's own
"Implementation note" ("signal_fn must fetch EFA OHLC itself...") describes
the per-trade single-asset ``signal_fn`` contract, under which only one
ticker (``tickers[0]``) is harness-supplied -- the ``rotation_two_asset``
shape used here hands ``make_target_fn`` full frames for both ``TRADED``
legs, so that extra fetch is unnecessary here.

``BENCHMARK`` is hard-pinned to "SPY" per this pipeline's standing rule:
ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
catalog "benchmark" field (this family's catalog entry has no
``benchmark_ticker`` field at all, predating the batch5 curation schema --
see any other batch5 module's own identical comment on this point, e.g.
``tips_nominal_breakeven_rotation.py``).

The rule is always invested (no cash state, matching ``cash_fallback:
false``); EEM's later listing start bounds the common EEM/EFA calendar.
``EXPECTED_MIN_TRADES`` is a configured, conservative floor, not a
measured count.
"""
from __future__ import annotations

from mining import signals

FAMILY_ID = "em_dm_relative_strength_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["EEM", "EFA"]           # [EEM, EFA] -- symmetric rule, order is
                                   # not decision-relevant (see docstring)
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (this module's own test verifies json.dumps(sort_keys=True)
# byte-identity against the live catalog file directly, not a hand-typed
# duplicate that could drift from it).
PARAMETERS = {
    "cash_fallback": False,
    "hold_top_n": 1,
    "lookback_months": 6,
    "num_assets": 2,
    "rebalance": "monthly",
}
# Configured, conservative floor (not a measured count): a monthly
# re-ranking of two equity ETFs over two decades is expected to flip well
# more often than this; a run below it points at a data or wiring problem.
EXPECTED_MIN_TRADES = 20

_RISK = TRADED[0]
_DEFENSIVE = TRADED[1]
_LOOKBACK_MONTHS = int(PARAMETERS["lookback_months"])


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract): called ONCE by the
    driver with full-history frames for TRADED (+ SIGNAL, none here).
    Every required TRADED key is validated present here so a missing
    frame fails at construction time, naming the missing key, rather
    than inside the first per-bar call. Returns the shared
    ``mining.signals`` factory's own per-call ``target(truncated)``
    callable unchanged -- prices are read only from the truncated
    frames the driver passes to it at call time, never from the
    full-history ``frames`` argument closed over here."""
    for key in TRADED:
        if key not in frames:
            raise ValueError(
                f"{FAMILY_ID}: make_target_fn requires a frame for {key}")
    return signals.make_monthly_relative_momentum_target_fn(
        _RISK, _DEFENSIVE, lookback_months=_LOOKBACK_MONTHS)
