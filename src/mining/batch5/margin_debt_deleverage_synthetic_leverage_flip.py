"""Candidate: margin_debt_deleverage_synthetic_leverage_flip
(rotation_two_asset).

FINRA aggregate NYSE member-firm customer margin-debt MONTH-OVER-MONTH
deceleration/contraction flips equity exposure between a SYNTHETIC
leveraged-long proxy and a REAL inverse ETF -- a leverage-DIRECTION flip,
not a long/cash regime hold. Mechanism:

    Flip equity exposure between the repo's existing synthetic leveraged-
    long proxy (data_pipeline.build_synthetic_leveraged on SPY, the same
    function synthetic_leveraged_dip_buy_pooled_largecap.py already uses)
    and the real inverse ETF SH, based on month-over-month deceleration/
    contraction in the FINRA aggregate customer margin-debt series.
    Positive MoM growth (leverage/risk appetite still expanding, even if
    decelerating in RATE) -> hold the synthetic 2x SPY position; MoM
    growth crosses to zero or below (the debt level itself contracted
    month over month -- forced deleveraging, margin calls, reduced risk
    appetite) -> hold SH (real short S&P 500 exposure), not cash.

DISTINCT FROM margin_debt_growth_regime (see that module's own
docstring): that family holds SPY outright and exits to CASH on a
TRAILING-12-MONTH (YoY) growth sign flip. This family (a) uses a
MONTH-OVER-MONTH (1-month) lookback -- a genuinely faster, noisier read on
credit-cycle deceleration, not a parameter tweak of the 12-month construct
-- and (b) never sits in cash: it flips the full leverage DIRECTION,
2x-long to -1x, so a false signal costs a real round-trip through both
legs' spreads and the daily-reset inverse fund's own volatility decay,
which the YoY/cash sibling never pays. No other family_id or mechanism in
``data/mining/catalog.json`` uses this construction;
`margin_debt_growth_regime` is the only other "margin" family and is the
cited, textually-distinguished sibling above.

PROVENANCE, reused not reimplemented: all FINRA growth/lag/threshold math
is `mining.signals`'s, unchanged --
`signals.finra_margin_debt_growth_available_series` (covered by its own
tests, and shared with `margin_debt_growth_regime.py` and
`scripts/run_batch3_mining.py`) is fully generic in `lookback_months`; this
module calls it with `lookback_months=1` instead of the sibling's 12. No
new FINRA math is written here, so one pre-registered parameter for one
family_id never has two implementations.

DISCLOSED, NOT INVENTED: `LAG_TRADING_DAYS = 20` is the SAME
pre-registered catalog constant `margin_debt_growth_regime` uses (FINRA's
~4-5-week publication delay) -- reused verbatim, not re-derived, and
carries the SAME unconfirmed-default residual risk that module's docstring
records (the true lag can run past 20 sessions in some months). See that
module's docstring for the full note.

Rotation target, not a single-asset signal: `mining.rotation_engine.
run_two_asset_rotation`'s `target_fn` is `Callable[[dict[str,
pd.DataFrame]], str | None]` -- this module's `make_target_fn` is a THIN
factory wrapping the already-tested `signals.finra_margin_debt_risk_on_
series` (boolean risk-on state) + `signals.make_external_regime_target_fn`
(the rotation-shape counterpart of `make_external_regime_signal_fn`, which
the single_long sibling uses) -- both reused unchanged, exactly the
established "thin adapter" convention.

Traded legs (built by the bespoke driver, `scripts/
run_margin_debt_deleverage_mining.py` -- the generic `run_batch5_mining.py`
dispatch cannot supply FINRA data or a synthetic frame, same reason
`margin_debt_growth_regime` and `synthetic_leveraged_dip_buy_pooled_
largecap` both need bespoke drivers):

* `SPY_2x_synthetic` -- built via `data_pipeline.build_synthetic_leveraged`
  on SPY's own real OHLC history at the function's own default leverage
  (2.0x) and default retail-margin financing spread (100bps over the real
  time-varying ^IRX) -- the SAME construction and SAME defaults
  `synthetic_leveraged_dip_buy_pooled_largecap.py` already uses (a disclosed
  default there, not re-litigated here). Trading cost basis:
  SPY's own real quoted spread (this is realized as real margin-financed
  SPY shares, not a packaged product -- the leverage's true extra cost is
  financing, already inside `build_synthetic_leveraged`), matching that
  sibling's own disclosed convention.
* `SH` -- ProShares Short S&P500 (real, -1x SPY inverse ETF, daily reset).
  Trading cost basis: SH's own real quoted spread.

Both legs benchmark against SPY (`BENCHMARK = "SPY"`), matching every
other FINRA-margin-debt family and the market-wide nature of the signal.

Account type: margin-financed leverage and a short ETF are not generally
meaningful in a tax-advantaged account (same reasoning `synthetic_
leveraged_dip_buy_pooled_largecap.py` gives) -- the bespoke driver scores
this as `account_type="taxable"`.

No look-ahead: `signals.finra_margin_debt_risk_on_series` dates each
monthly reading at its real lagged availability date (never a future
reading); `signals.make_external_regime_target_fn`'s returned `target`
looks up `state.index <= today` at the rotation engine's own decision bar
(`frames[risk_ticker].index[-1]`), which can structurally never see a
future FINRA release. Both are pre-existing, already-tested code.

A missing `margin_signal` (the shared batch5 driver's generic dispatch,
which never threads external data through -- this family's own bespoke
driver always supplies it) degrades to "never switch, always flat"
(`target_fn` returns `None` for every bar) -- matching `margin_debt_
growth_regime`'s identical graceful-degradation convention.
"""
from __future__ import annotations

import pandas as pd

from mining import signals

FAMILY_ID = "margin_debt_deleverage_synthetic_leverage_flip"
SHAPE = "rotation_two_asset"
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied from the pre-registered mechanism
# verbatim, plus the SAME disclosed lag constant margin_debt_growth_regime
# already carries (reused, not re-derived) and build_synthetic_leveraged's
# OWN default leverage (2.0, unchanged from that function's signature).
PARAMETERS = {
    "margin_debt_lookback_months": 1,   # MoM deceleration/contraction --
                                        # the genuinely different axis vs
                                        # margin_debt_growth_regime's 12
                                        # (trailing-12-month / YoY)
    "signal_lag_trading_days": 20,      # reused verbatim from the sibling
                                        # family's pre-registered constant
    "growth_threshold": 0,
    "leverage": 2.0,                    # data_pipeline.build_synthetic_
                                        # leveraged's own default
    "financing_spread_bps": 100,        # data_pipeline.build_synthetic_
                                        # leveraged's own default (0.0100);
                                        # an explicit, drift-checked
                                        # parameter, so a change to that
                                        # function's default cannot silently
                                        # change this family (same shape as
                                        # post_opex_gamma_release_
                                        # leverage_window's own catalog
                                        # entry)
    "annual_expense_ratio": 0.0,        # data_pipeline.build_synthetic_
                                        # leveraged's own default -- real
                                        # margin-financed shares, no fund
                                        # wrapper/expense ratio
    "risk_on_position": "SPY_2x_synthetic",
    "risk_off_position": "SH",
}
LOOKBACK_MONTHS = int(PARAMETERS["margin_debt_lookback_months"])
LAG_TRADING_DAYS = int(PARAMETERS["signal_lag_trading_days"])
GROWTH_THRESHOLD = float(PARAMETERS["growth_threshold"])
LEVERAGE = float(PARAMETERS["leverage"])
FINANCING_SPREAD = PARAMETERS["financing_spread_bps"] / 1e4
ANNUAL_EXPENSE_RATIO = float(PARAMETERS["annual_expense_ratio"])

TRADED = [f"SPY_{int(LEVERAGE)}x_synthetic", "SH"]
SIGNAL: list = []  # external (FINRA), not a price-frame signal ticker
# Conservative floor on the expected in-sample trade count (SH's
# inception is the binding common-inception floor, not SPY's own
# 1993 start), same conservative-floor convention as
# vix_term_structure_carry.py / margin_debt_growth_regime.py, leaving room
# for whatever slice of history a driver's own window/holdout handling
# covers. Reported alongside a run's results; it gates nothing.
EXPECTED_MIN_TRADES = 70


def _empty_target_fn(frames):
    return None


def make_target_fn(index, *, margin_signal=None):
    """Rotation TargetFn factory. ``index`` is the traded calendar used to
    date FINRA availability (the driver's full SPY index, matching
    ``margin_debt_growth_regime.make_signal_fn``'s identical convention).
    ``margin_signal``, when supplied, is the RAW monthly FINRA
    ``finra_margin_debt_usd_millions`` series (never a pre-transformed
    growth/lag series) -- identical convention to the sibling family's
    ``margin_signal`` keyword. All growth/lag/threshold math is delegated
    to ``mining.signals`` (shared, covered by its own tests), never
    re-implemented here."""
    if len(pd.DatetimeIndex(index)) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")

    if margin_signal is None or margin_signal.empty:
        return _empty_target_fn

    risk_on = signals.finra_margin_debt_risk_on_series(
        margin_signal, index, lookback_months=LOOKBACK_MONTHS,
        lag_sessions=LAG_TRADING_DAYS, growth_threshold=GROWTH_THRESHOLD)
    if risk_on.empty:
        return _empty_target_fn

    return signals.make_external_regime_target_fn(
        risk_on, risk_ticker=TRADED[0], defensive_ticker=TRADED[1])
