"""Batch5 candidate: reit_sector_absolute_momentum_trend (rotation_two_asset).

Implements the catalogued mechanism (``data/mining/catalog.json``, status
"eligible", ``added_date`` 2026-09-01) under the ABI-1 batch5 module
contract. The decision rule is
``mining.signals.make_monthly_absolute_momentum_target_fn``, called with
the catalog entry's parameters.

Mechanism (``data/mining/catalog.json`` entry for this family_id, verbatim;
Hung, K., Glascock, J.L. (2008), "Momentum Profitability and Market Trend:
Evidence from REITs," Journal of Real Estate Finance and Economics, 37(1),
51-69): each month, compute VNQ's own trailing 6-calendar-month total return
(this repo's auto-adjusted yfinance close, per ``data_pipeline`` convention).
If VNQ's current close exceeds its 6-month-ago calendar-anchor close, hold
VNQ; otherwise rotate fully into BIL (T-bill cash proxy) until the trailing
return turns positive again. Re-evaluate monthly. A single-sector ABSOLUTE
(time-series) momentum/trend filter on VNQ's own price history -- same
mechanism shape as ``hy_credit_bond_absolute_momentum``'s HYG absolute
momentum, with VNQ in place of HYG.

``make_target_fn`` returns
``mining.signals.make_monthly_absolute_momentum_target_fn(TRADED[0],
TRADED[1], lookback_months=6)`` directly: that factory's own ``target``
callable already takes ``frames: dict[str, pd.DataFrame]`` as its single
per-call argument and computes its decision purely from the frames handed
to it each call -- exactly the batch5 rotation_two_asset per-bar contract
(the driver truncates ``frames`` further each bar before calling), with no
extra wrapping layer needed. See ``hy_credit_bond_absolute_momentum.py``'s
own docstring for this same "a signals.py factory already returns the
correctly-shaped per-call target" convention. The only work this module
adds on top is the one-time required-key check the driver contract asks
``make_target_fn`` itself to perform before any per-bar call.

Catalog entry and factory call: the catalog entry for this family_id has
``tickers`` == ``["VNQ", "BIL"]`` and ``parameters`` ==
``{"lookback_months": 6, "rebalance": "monthly", "signal":
"absolute_momentum_own_trailing_return", "cash_proxy": "BIL"}`` (checked
via ``json.dumps(sort_keys=True)`` in this module's own test, which reads
the catalog file directly rather than duplicating a value here that could
drift). The catalog entry has no ``benchmark_ticker`` field, consistent
with ``hy_credit_bond_absolute_momentum`` and the pipeline's own hard
"always SPY" benchmark rule below. The factory call and argument order
(``TRADED[0]``=risk=VNQ, ``TRADED[1]``=defensive=BIL,
``lookback_months=6``) match ``make_monthly_absolute_momentum_target_fn``'s
own signature (``src/mining/signals.py``) and its sibling caller's usage in
``hy_credit_bond_absolute_momentum.py`` exactly. This factory has no
risk/defensive polarity ambiguity to get backwards: it returns
``risk_ticker`` when the current close exceeds the calendar-anchor close and
``defensive_ticker`` otherwise (``None`` before a usable anchor exists),
matching the catalog mechanism text directly.

BENCHMARK is hard-pinned to "SPY" per this pipeline's standing rule: ABI-1
ALWAYS benchmarks the account against SPY, never a family's own catalog
"benchmark" field (see ``scripts/run_abi1_mining.py``'s own
``MARKET_BENCHMARK`` and every other batch5 module's identical comment on
this point).
"""
from __future__ import annotations

from mining import signals

FAMILY_ID = "reit_sector_absolute_momentum_trend"
SHAPE = "rotation_two_asset"
TRADED = ["VNQ", "BIL"]      # [risk, defensive]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's entry for this
# family_id (checked via json.dumps(sort_keys=True) byte-identity in this
# module's own test, which reads that file directly rather than duplicating
# a value here that could drift).
PARAMETERS = {
    "lookback_months": 6,
    "rebalance": "monthly",
    "signal": "absolute_momentum_own_trailing_return",
    "cash_proxy": "BIL",
}
_LOOKBACK_MONTHS = PARAMETERS["lookback_months"]
# Configured, conservative floor (not a measured count), same convention as
# every sibling batch5 module. The common VNQ/BIL calendar is bound by the
# later of the two data starts, and the first entry follows the 6-month
# warm-up.
EXPECTED_MIN_TRADES = 30


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED; only their presence is checked
    here -- decisions come from the (further-truncated, per the engine's
    own contract) frames the returned callable is called with, delegated
    unchanged to mining.signals.make_monthly_absolute_momentum_target_fn."""
    for key in TRADED:
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")
    return signals.make_monthly_absolute_momentum_target_fn(
        TRADED[0], TRADED[1], lookback_months=_LOOKBACK_MONTHS)
