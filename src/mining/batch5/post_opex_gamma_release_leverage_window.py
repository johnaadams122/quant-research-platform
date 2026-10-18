"""Batch5 candidate: post_opex_gamma_release_leverage_window
(pooled_single_asset).

Hold a SYNTHETIC 2x margin-financed position in each of the 11 Select
Sector SPDRs ONLY during the trading week immediately AFTER the ordinary
monthly (3rd-Friday) options-expiration Friday -- flat the rest of the
month. Pooled across the same 11-member panel as vol_managed_sector_spdr_
pooled so the mechanism is tested with hundreds of round trips instead of
12/year on a single ticker.

MECHANISM
---------
Dealer delta-hedge rebalancing around the largest recurring monthly
options rolloff pushes underlying prices toward heavily-open strikes into
expiration (Ni, Pearson & Poteshman 2005, JFE, "stock price clustering on
option expiration dates" -- hedge rebalancing by option market makers
measurably alters expiration-date closing prices). That pinning/hedging
pressure is mechanically tied to the SOON-TO-EXPIRE contracts' gamma; once
they expire the large block of gamma dealers were hedging against them
rolls off ("gamma release" / "gamma unpin" in practitioner terms -- see
source field) and the suppressing force is gone. Klemkosky (1978) and
Chiang (2014) independently document a short-term REVERSAL in the days/
week immediately following expiration that partially offsets the
pre-expiration price-pressure move -- this family targets exactly that
post-expiration window, amplified with leverage to clear trading costs on
what the literature shows is a modest single-week effect.

RULE
----
    hold_trading_days_after_opex = 5   # one trading week (disclosed
                                        # default -- Klemkosky's own
                                        # framing is "the week following
                                        # expiration"; pending
                                        # validation, same convention as
                                        # every other invented-schedule
                                        # constant in this pipeline)
    leverage = 2.0                     # Reg-T margin-financed synthetic
                                        # position (see "Why synthetic,
                                        # not a real leveraged product"
                                        # below)

    For each T in POOL independently, using T's own real auto-adjusted
    daily calendar:
      expiry = the month's 3rd-Friday option-expiration session
      signal(expiry) = True (entry); next-open execution enters the
        SYNTHETIC 2x position the FIRST post-opex session
      signal fires exit=True at the close of the session
        hold_trading_days_after_opex actual sessions after expiry
      one trade per (ticker, month) pair with a resolvable exit; a month
        whose expiry falls too close to the end of available history for
        the exit session to exist is skipped, never truncated
      trades of all 11 pool members concatenated into one sample, scored
        as ONE trial

Why SYNTHETIC, not a real leveraged product: real leveraged sector/country
products are mostly thin (e.g. DPST, DFEN), with no reliably usable dated
quote, and no broadly liquid 3x fund exists for most Select Sector SPDRs.
A synthetic 2x
Reg-T-margin position, built off each SPDR's own real, liquid, decades-
long price history via ``data_pipeline.build_synthetic_leveraged``
(already used for exactly this purpose by
synthetic_leveraged_dip_buy_pooled_largecap.py, the synthetic 2x leveraged dip-buy family), sidesteps
that liquidity problem entirely -- the trading-cost basis is the
UNDERLYING SPDR's own real, liquid, quoted spread (see PARAMETERS), never
a thin leveraged-product's.

DISTINCTNESS
------------
vs. vol_managed_sector_spdr_pooled (SAME 11-member panel): that family's signal is each fund's OWN TRAILING REALIZED
VARIANCE against an expanding median, re-evaluated once a month on the
month's last session -- a continuous, price-derived regime signal. This
family's signal is a fixed CALENDAR EVENT (the 3rd-Friday expiration
date), known in advance from the public options calendar, with NO price
input into the entry/exit decision at all -- reusing the same panel with a
genuinely different signal is an established, already-accepted pattern in
this catalog (see e.g. sector_same_month_seasonality's own distinctness
note against this identical sibling). Wrapping the SAME realized-variance
switch in leverage would be the same mechanism (the catalog records
leveraged_sector_country_3x_vol_managed_pooled as merged into
vol_managed_sector_spdr_pooled for that reason); this candidate's SIGNAL
itself is different in kind (event/calendar vs. price/variance), not just
its wrapper.

vs. monthly_opex_week_drift (SPY): that family enters 4
sessions BEFORE expiry and exits AT expiry's close -- the pre-expiration
pinning/pressure window (Johnson & So 2013). This family enters AT expiry
and exits a week AFTER -- the opposite side of the same recurring event,
on a POOL of 11 sector funds instead of one SPY position, with 2x
synthetic leverage instead of unlevered SPY. ``tests/test_mining_batch3_
signals.py::test_post_opex_window_dates_never_shares_a_date_with_
monthly_opex_dates_pre_window`` pins that the two families' entry/exit
dates never overlap (post-opex entries occur exactly on pre-opex exits --
both windows share expiry Friday as their pivot, never as overlapping
holding days).

vs. long_treasury_3x_fomc_window_drift (TMF, status eligible): the closest
sibling in SHAPE (a leveraged product held only in a short calendar window
around a recurring macro/market-structure event, flat otherwise) but a
different asset class, a different event (options expiry vs. FOMC
statements), and a different literature.

Engine mapping (mining.backtest_engine.run_backtest): the per-ticker
entry/exit date sets are precomputed ONCE per pool member from that
member's own real calendar (holdout-blind truncated to <= the bucket's
last bar, same convention as long_treasury_3x_fomc_window_drift and
vol_managed_sector_spdr_pooled) via ``mining.signals.post_opex_window_
dates``, then answered by ``mining.signals.make_date_window_signal_fn`` --
no new signal-evaluation code, both helpers are pre-existing, independently
tested (src/tests/test_mining_batch3_signals.py).
"""
from __future__ import annotations

import pandas as pd

from mining import signals
from mining.backtest_engine import SignalFn
from mining.batch5.synthetic_leveraged_dip_buy_pooled_largecap import (
    synthetic_ohlc_from_close)

FAMILY_ID = "post_opex_gamma_release_leverage_window"
SHAPE = "pooled_single_asset"
_POOL = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
         "XLRE", "XLC"]
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied byte-for-byte from this family's catalog "parameters" field
# (data/mining/catalog.json) -- driver drift check via
# json.dumps(sort_keys=True), same convention as every other batch5 module.
PARAMETERS = {
    "event": "ordinary_monthly_third_friday_option_expiration",
    "hold_trading_days_after_opex": 5,
    "leverage": 2.0,
    "financing_spread_bps": 100,
    "annual_expense_ratio": 0.0,
    "entry": "synthetic_open_first_session_after_opex_friday",
    "exit": "synthetic_close_n_sessions_after_opex_friday",
    "trading_cost_basis": "underlying_spdr_real_quoted_spread",
    "pool": list(_POOL),
}
LEVERAGE = float(PARAMETERS["leverage"])
HOLD_TRADING_DAYS_AFTER_OPEX = int(PARAMETERS["hold_trading_days_after_opex"])
FINANCING_SPREAD = PARAMETERS["financing_spread_bps"] / 1e4
ANNUAL_EXPENSE_RATIO = float(PARAMETERS["annual_expense_ratio"])

# SIGNAL is each pool member's real underlying ticker (needed only to
# derive the trading calendar the opex dates are computed against -- no
# price ever enters the decision); TRADED is the SYNTHETIC id actually
# priced/held, matching synthetic_leveraged_dip_buy_pooled_largecap's
# identical SIGNAL/TRADED split convention (validate_module_contract
# requires the two disjoint, and they are: e.g. "XLK" vs
# "XLK_2x_synthetic").
SIGNAL = list(_POOL)
TRADED = [f"{ticker}_{int(LEVERAGE)}x_synthetic" for ticker in _POOL]
# Structural count (calendar only): the window opens once per monthly
# expiration cycle, so each pool member contributes about one trade per
# calendar month of its own history (the 9 founding funds start in late
# 1998; XLRE and XLC list much later). Set at a conservative floor below
# that calendar-derived pooled count, same convention as every other batch5
# module's EXPECTED_MIN_TRADES.
EXPECTED_MIN_TRADES = 2800


def underlying_of(synthetic_ticker: str) -> str:
    """Inverse of ``synthetic_id_for`` -- the real ticker a synthetic id
    (e.g. 'XLK_2x_synthetic') derives from. Identical convention to
    synthetic_leveraged_dip_buy_pooled_largecap.underlying_of."""
    suffix = f"_{int(LEVERAGE)}x_synthetic"
    if not synthetic_ticker.endswith(suffix):
        raise ValueError(
            f"{synthetic_ticker!r} is not a {suffix!r}-suffixed synthetic id")
    return synthetic_ticker[: -len(suffix)]


def synthetic_id_for(underlying_ticker: str) -> str:
    return f"{underlying_ticker}_{int(LEVERAGE)}x_synthetic"


def _naive_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def make_signal_fn(ticker: str, index, frames: dict) -> SignalFn:
    """Driver contract factory (pooled_single_asset: called once per pool
    member with that member's SYNTHETIC id as ``ticker``). ``frames`` must
    contain ``frames[ticker]`` (the synthetic OHLC the engine will actually
    price, built by the driver via data_pipeline.build_synthetic_leveraged
    + synthetic_ohlc_from_close) -- only its DATE INDEX is used here, since
    the opex signal is pure calendar arithmetic with no price input at all.
    ``frames[underlying_of(ticker)]`` is accepted too if present (not
    required) purely so a caller reusing the dip-buy family's frames dict
    shape (real underlying + synthetic) does not need a special case.

    Holdout-blind by construction: the calendar is truncated to <= the
    bucket's (``index``'s) last bar before entry/exit dates are resolved,
    same convention as long_treasury_3x_fomc_window_drift.make_signal_fn
    and vol_managed_sector_spdr_pooled.make_signal_fn.
    """
    if ticker not in frames:
        raise KeyError(f"{FAMILY_ID}: no synthetic frame for {ticker}")
    frame = frames[ticker]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame must be a non-empty "
                         "OHLC DataFrame")

    bucket = pd.DatetimeIndex(pd.to_datetime(index))
    if bucket.tz is not None:
        bucket = bucket.tz_localize(None)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    cutoff = bucket[-1]

    calendar = _naive_index(frame).unique().sort_values()
    calendar = calendar[calendar <= cutoff]  # holdout-blind truncation

    entries, exits = signals.post_opex_window_dates(
        calendar, hold_trading_days_after_opex=HOLD_TRADING_DAYS_AFTER_OPEX)
    return signals.make_date_window_signal_fn(entries, exits)
