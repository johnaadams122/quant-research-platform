"""Batch5 candidate: treasury_duration_volatility_timing (rotation_two_asset).

Implements a catalogued family under the batch5 module contract. The
mechanism is also defined in the batch3 ``run_batch3_mining.py`` driver
(``SUPPORTED_CONTRACTS["treasury_duration_volatility_timing"]``,
mechanism class ``monthly_two_asset_realized_volatility_rotation``). This
module wraps the shared ``mining.signals.make_monthly_volatility_target_fn``
factory in the batch5
``make_target_fn(frames) -> target(truncated)`` two-level shape, the same
thin-wrapper pattern used by ``fed_net_liquidity_regime_qqq.py``: the
factory's own ``target(frames: dict[str, pd.DataFrame])`` closure already
takes the truncated per-call frames dict as its single argument, so no
extra wrapping layer is needed beyond validating the full-history frames
carry both required legs.

Source (from the pre-registered catalog entry,
data/mining/catalog.json): Viceira, L.M. (2012), "Bond Risk, Bond Return
Volatility, and the Term Structure of Interest Rates", International
Journal of Forecasting 28, 97-117.
https://www.sciencedirect.com/science/article/abs/pii/S0169207011000446
(verified 2026-08-29). Tickers verified: TLT/SHY (iShares
Treasury ETFs), both launched 2002-07-22, actively trading.

Mechanism (verbatim from the catalog): "Compute TLT's trailing 20-day
realized annualized volatility against its own trailing 252-day median.
Below median (calm) -> hold long-duration TLT; above median (elevated
bond-market stress) -> rotate to short-duration SHY. Re-evaluate monthly. A
pure volatility-regime filter on the bond market's own risk, not a
curve-slope/momentum signal."

PARAMETERS is copied byte-for-byte from the catalog entry
(vol_lookback_days=20, vol_threshold="trailing_252d_median" ->
median_lookback_days=252 via the same
``re.fullmatch(r"trailing_(\\d+)d_median", ...)`` translation
``run_batch3_mining.catalog_parameters`` already used for this exact
family, rebalance="monthly" -- the factory's own calendar-month decision
cadence, no extra rebalance logic needed).

BENCHMARK is "SPY", per this pipeline's hard ABI-1 rule that every family
is benchmarked against the market reference unconditionally, never against
a family's own catalog "benchmark" field (see any other batch5 module's own
comment on this, e.g. ``fed_net_liquidity_regime_qqq.py``).
"""
from __future__ import annotations

from mining import signals

FAMILY_ID = "treasury_duration_volatility_timing"
SHAPE = "rotation_two_asset"
TRADED = ["TLT", "SHY"]
SIGNAL: list = []  # no external price-frame ticker beyond the two legs
SNAPSHOTS: dict = {}
# ABI-1 always benchmarks against SPY, not a family's own catalog field --
# see module docstring.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's
# "treasury_duration_volatility_timing" entry (verified in
# test_batch5_treasury_duration_volatility_timing.py via
# json.dumps(sort_keys=True) byte-identity against that file directly).
PARAMETERS = {
    "vol_lookback_days": 20,
    "vol_threshold": "trailing_252d_median",
    "rebalance": "monthly",
}
_VOL_LOOKBACK_DAYS = PARAMETERS["vol_lookback_days"]
# vol_threshold = "trailing_252d_median" -> median_lookback_days=252, the
# same translation run_batch3_mining.catalog_parameters already used for
# this exact family (its treasury_duration_volatility_timing branch in
# scripts/run_batch3_mining.py).
_MEDIAN_LOOKBACK_DAYS = 252
# The rule is always invested after the ~252-session warm-up (there is no
# cash state). Configured, conservative floor (not a measured count): a
# monthly TLT/SHY volatility rotation over two decades is expected to
# switch well more often than this.
EXPECTED_MIN_TRADES = 40


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED; only their presence is checked
    here -- the returned ``target`` reads only the truncated frames the
    engine hands it at call time (mining.signals.make_monthly_volatility_
    target_fn's own contract), so this factory never looks at prices past
    a holdout bucket's own boundary.
    """
    for key in TRADED:
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")
    return signals.make_monthly_volatility_target_fn(
        TRADED[0], TRADED[1],
        vol_lookback_days=_VOL_LOOKBACK_DAYS,
        median_lookback_days=_MEDIAN_LOOKBACK_DAYS)
