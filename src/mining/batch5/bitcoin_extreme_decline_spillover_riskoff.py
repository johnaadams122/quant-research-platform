"""New candidate: bitcoin_extreme_decline_spillover_riskoff (single_long).

A cross-asset spillover mechanism: an extreme, single-day BTC-USD price
shock (measured against BTC's own trailing realized volatility) is used as
a leading risk-off trigger for QQQ, on the hypothesis that Bitcoin's
increasingly tight, asymmetric-in-stress correlation with the Nasdaq
complex means a crypto-market deleveraging event is informative about
equity-market risk appetite before or concurrently with an equity-market
move -- not a restatement of "buy QQQ after QQQ itself drops" (see
DISTINCTNESS NOTE below).

Mechanism
---------

    Fetch BTC-USD daily spot OHLC (mining.crypto_data.fetch_kraken_spot_ohlcv,
    interval=1440, Kraken's real public spot market). Compute BTC's own daily
    log return r_t = ln(Close_t / Close_{t-1}) on BTC's own CALENDAR (crypto
    trades every day, unlike QQQ's NYSE business-day calendar).
    Compute a trailing 20-day realized volatility as the sample std (ddof=1)
    of the 20 daily log returns STRICTLY BEFORE day t (r_{t-20}..r_{t-1} --
    never r_t itself), exactly mirroring
    single_day_extreme_decline_reversal_long.py's own no-look-ahead
    convention, applied to BTC instead of SPY.
    z_t = r_t / trailing_vol_t.
    Risk-off trigger: z_t <= -2.0 on BTC calendar day t. Once triggered,
    QQQ exposure is reduced to CASH for a fixed 5-CALENDAR-day cooldown
    window starting at day t (inclusive) -- a persistence rule, not a
    single-day flag: any BTC day within 5 calendar days of a trigger is
    also "risk off", so overlapping/repeat triggers extend the cooldown
    rather than shortening it (mining.rolling(...).max() semantics, see
    ``btc_riskoff_regime``).
    Outside a risk-off window, QQQ is held at full weight (the "always long
    unless triggered" shape shared by every existing FRED-driven regime
    family in this catalog -- usd_strength_risk_regime,
    fed_net_liquidity_regime, money_market_fund_flow_riskoff,
    margin_debt_growth_regime).
    Each BTC-calendar-day risk-off/on READING becomes tradeable on the
    FIRST QQQ trading session STRICTLY AFTER that BTC calendar day
    (``signals.lag_boolean_regime_to_availability``, lag_trading_days=1) --
    see "No look-ahead" below for why this is causally safe with a
    comfortable margin, not a bare minimum.

Economic hypothesis and citation
---------------------------------
Two real, independent, cited facts combine into this hypothesis:

(1) Bitcoin's correlation with the Nasdaq-100 has structurally increased
    since 2020 and is ASYMMETRIC -- it strengthens specifically during
    risk-off/stress episodes, as BTC has come to trade as "a leveraged
    risk asset" alongside high-beta tech exposure rather than as an
    uncorrelated or safe-haven instrument. (Public reporting:
    BTC-Nasdaq-100 correlation reached roughly 0.87 during
    2024's most correlated stretches, versus near-zero 2014-2019; CME
    Group's own 2025 research notes "Bitcoin behaves much like a
    leveraged version of major stock market indices" and that this
    correlation "is alive and well (during risk-off)".)
(2) IMF Working Paper 2023/213, Iyer, R. & Popescu, A. (2023), "New
    Evidence on Spillovers Between Crypto Assets and Financial Markets",
    International Monetary Fund
    (elibrary.imf.org/view/journals/001/2023/213/article-A001-en.xml): crypto assets are found to be a PREDOMINANT NET TRANSMITTER
    of return/volatility spillovers to conventional financial markets,
    with the magnitude of these spillovers INCREASING specifically during
    periods of heightened turbulence, and increased crypto-equity
    correlation during risk-off episodes flagged as a genuine financial-
    stability transmission channel.

Combined: an extreme, statistically unusual single-day BTC decline is a
real-time, publicly observable proxy for a crypto-market deleveraging
event, and per (2) such events are a net spillover SOURCE into equity
markets whose transmission is strongest exactly when this signal fires
(a stress period) -- so temporarily de-risking QQQ around such an event is
a directionally motivated, falsifiable bet, not an arbitrary threshold
rule. Kraken's own trading-education
article ("Funding rates as a trading signal", kraken.com/learn/futures-
trading-funding-rate-strategy) independently corroborates the general
"extreme move -> crowded, unstable positioning -> reversal risk" pattern
in crypto derivatives markets specifically, though this family's own
mechanism uses BTC SPOT price action (not funding rate -- see "Why spot,
not funding rate" below), so that source is corroborating context, not the
load-bearing citation for the mechanism actually implemented.

Why spot, not funding rate (data availability):
``mining.crypto_data.fetch_kraken_perp_funding_rate`` is the more literal
"crowded leverage" measure, but Kraken's futures
funding-rate history endpoint has no pagination and returns only its own
retained history -- roughly a year, per that module's own docstring.
Kraken's public SPOT OHLC endpoint has an
UNDOCUMENTED, real constraint of its own -- ``fetch_kraken_spot_ohlcv``
with ``interval=1440`` (daily) returns at most ~720 rows and, measured
directly against ``since`` values from 2015 through 2023, silently ignores
any ``since`` older than roughly 720 days ago and returns the SAME most-
recent ~720-row window regardless (only a ``since`` within that live
window, e.g. 2025, is honored). So real usable daily BTC-USD history via
this connector is bounded to roughly the trailing two years before the
fetch date, not spot's full multi-year
listing history -- still meaningfully longer than the funding-rate
endpoint's ~1 year, which is why this family uses spot, but the
~2-year real ceiling is a genuine, load-bearing constraint on this
family's own tradeable window, disclosed here rather than hidden. This
module uses no Kraken-side historical bulk-data or paginated-since
workaround; the limit is a connector characteristic any crypto-touching
mining work should expect, not something specific to this family.

DISTINCTNESS NOTE (verbatim mechanism text from the closest catalog
siblings, data/mining/catalog.json; no other catalog family references
"crypto"/"bitcoin"/"btc"/"kraken"):

* ``single_day_extreme_decline_reversal_long`` -- "Compute daily log returns
  for SPY ... z_t <= -2.5 on session t -> enter long ... Hold for a fixed
  time stop of exactly 3 trading sessions". This family's trigger asset
  (SPY) and traded asset (SPY) are the SAME instrument, and a trigger is a
  LONG-ENTRY dip-buy on that instrument. This family's trigger asset (BTC)
  and traded asset (QQQ) are DIFFERENT instruments in DIFFERENT asset
  classes, and a trigger is a temporary EXIT-TO-CASH (risk reduction) on
  the traded instrument, not a buy -- opposite polarity and a genuinely
  cross-asset (not self-referential) mechanism.
* ``usd_strength_risk_regime`` -- "Track UUP's closing price against its
  own 200-day simple moving average ... When UUP closes above its 200-day
  average ... move out of SPY into a defensive cash/short-Treasury (SHY)
  allocation." A SLOW, TREND-FOLLOWING regime signal (200-day MA cross) on
  a currency-strength proxy, defensive posture = SHY, traded = SPY. This
  family is an ASYMMETRIC-SPIKE/EVENT trigger (a single extreme day, not a
  slow moving-average cross) on a completely different asset class
  (crypto, not currency), defensive posture = cash (not SHY), traded =
  QQQ.
* ``fed_net_liquidity_regime`` / ``money_market_fund_flow_riskoff`` /
  ``margin_debt_growth_regime`` -- all FRED/FINRA-sourced, slow-moving
  aggregate-liquidity or leverage TREND regimes (13-week MA of a weekly
  Fed balance-sheet series; expanding-window z-score of a monthly fund-
  flow series; trailing-12-month margin-debt growth), each with a
  publication lag measured in WEEKS. This family's trigger is a single
  EXTREME DAILY EVENT in a real-time, 24/7-traded market with a 1-session
  lag, an entirely different data source (crypto exchange price data, not
  a government/SRO aggregate release) and a much faster signal cadence.
* ``margin_debt_deleverage_synthetic_leverage_flip`` -- "FINRA aggregate
  ... margin-debt month-over-month deceleration/contraction flips equity
  exposure between a synthetic 2x margin-financed long position ... and
  the real inverse ETF SH ... Never sits in cash". This family's
  defensive posture is CASH, never a full directional flip to an inverse
  position, and its trigger is a monthly FINRA aggregate, not a daily
  crypto-market price shock.
* ``vix_spike_reversal_equity_buy`` / ``vvix_vix_derisking_regime`` --
  both use an OPTIONS-IMPLIED volatility index (VIX/VVIX) of the SAME
  asset class being traded (US equities) as the regime trigger. This
  family's trigger is a REALIZED price shock in a genuinely different
  asset class (crypto spot), not an implied-vol index on equities --
  the spillover hypothesis this family tests (a real, cited, cross-asset
  transmission channel) is structurally different from "equity options
  markets are pricing in more near-term risk".

None of the above can fire the way this family does: every equity-implied-
vol or equity-price-based sibling is blind to a crypto-market event that
produces no equity-index move of its own (yet), and every macro/FRED
sibling is blind to intra-month, real-time crypto price action entirely.

Engine mapping (mining.backtest_engine.run_backtest)
-----------------------------------------------------
This family uses ``signals.make_external_regime_signal_fn`` (the SAME
generic long/cash regime combinator every other FRED-driven single_long
family in this catalog already uses), so entry/exit semantics are
identical to those families': the engine fills a
"become risk-on" entry at the next bar's OPEN and a "become risk-off"
exit at the SAME bar's CLOSE, per that combinator's own no-backward-fill,
as-of-today lookup. No engine ``time_stop`` is used or needed -- the
5-calendar-day cooldown is fully expressed in the regime series itself
(``btc_riskoff_regime``'s own rolling-max persistence), exactly the same
"engine never sees time_stop; the module's own precomputed series already
encodes the hold" pattern as
``money_market_fund_flow_riskoff_bounded_burnin.py``.

Interpretation choices
-----------------------
* No look-ahead: BTC's day-t candle (Kraken's own "open time" convention)
  covers the full UTC calendar day t and only becomes fully known at UTC
  midnight of day t+1 -- roughly 19:00-20:00 ET the EVENING of day t
  (ET trails UTC by 4-5 hours), which is already hours before that
  evening's close and a full overnight before the NEXT NYSE session even
  opens (09:30 ET). Requiring the regime reading to become tradeable on
  the FIRST QQQ SESSION STRICTLY AFTER BTC calendar day t
  (lag_trading_days=1 via ``signals.lag_boolean_regime_to_availability``)
  is therefore a comfortably conservative mapping, not a bare-minimum one
  -- it never assumes information before Kraken's own candle for day t is
  actually complete, with hours of real margin to spare.
* Data-availability window: this family's regime series exists ONLY over
  the real fetched BTC history (see "Why spot, not funding rate" above).
  QQQ trading sessions before that window has any BTC signal produce
  ``SignalResult()`` (no entry, no exit) via
  ``signals.make_external_regime_signal_fn``'s own "no backward fill"
  rule -- the engine never opens a position before the signal is known,
  so this family is a no-op (flat, zero trades) over the entire multi-
  decade QQQ history before BTC data begins. This is DISCLOSED, not
  hidden: EXPECTED_MIN_TRADES below applies to the full QQQ history but
  every trade is produced within the ~2-year BTC-covered tail (a signal's
  real informative window can be much shorter than the traded
  instrument's own listing history).
* Default regime before any trigger has occurred within the available BTC
  window is risk-ON (hold QQQ) -- a day whose trailing-vol warm-up is not
  yet complete (the first ``vol_window`` BTC days) computes z_t as NaN,
  and ``NaN <= threshold`` is False in NumPy/pandas comparisons, so the
  warm-up period is automatically, correctly treated as "no trigger" with
  no special-casing required.
* Trigger persistence is a CAUSAL backward-looking rolling max
  (``raw_trigger.rolling(hold_days, min_periods=1).max()``), never a
  forward-looking window -- day t's risk-off flag depends only on BTC
  triggers at day t and BEFORE, so a later trigger extends the cooldown
  forward in time but never reaches backward to relabel an earlier day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining import signals
from mining.backtest_engine import SignalResult

FAMILY_ID = "bitcoin_extreme_decline_spillover_riskoff"
SHAPE = "single_long"
TRADED = ["QQQ"]
SIGNAL: list = []  # external (Kraken BTC-USD spot OHLC), not a price-frame ticker
BENCHMARK = "SPY"  # abi1_driver.evaluate_family's own documented contract:
# ABI-1 always benchmarks against SPY unconditionally, never against
# whatever is traded (same convention as fed_net_liquidity_regime_qqq.py).
SNAPSHOTS: dict = {}
PARAMETERS = {
    "btc_vol_window_days": 20,
    "btc_zscore_riskoff_threshold": -2.0,
    "riskoff_hold_days": 5,
    "lag_trading_days": 1,
}
VOL_WINDOW_DAYS = int(PARAMETERS["btc_vol_window_days"])
Z_THRESHOLD = float(PARAMETERS["btc_zscore_riskoff_threshold"])
HOLD_DAYS = int(PARAMETERS["riskoff_hold_days"])
LAG_TRADING_DAYS = int(PARAMETERS["lag_trading_days"])
# The threshold is a genuinely "extreme" |z| >= 2 reading. Every trade of
# a full-history run falls inside the BTC-covered tail of the QQQ history:
# QQQ sessions before any BTC reading exists produce no signal (see
# "Data-availability window" in the module docstring).
# EXPECTED_MIN_TRADES is a conservative sanity floor on the trade count of a
# full-history run (it sits just above constants.MIN_TRADES_FOR_SCORING), the
# same convention as every sibling family in this catalog.
EXPECTED_MIN_TRADES = 16


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    return frame


def _empty_signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
    return SignalResult()


def btc_decline_zscore(close: pd.Series, *,
                       vol_window: int = VOL_WINDOW_DAYS) -> pd.Series:
    """z_t = r_t / trailing_vol_t over ``close``'s own (BTC calendar-day)
    index. r_t is BTC's own single-day log return ending at t
    (``np.log(close).diff()``). trailing_vol_t is the sample std (ddof=1)
    of the ``vol_window`` log returns STRICTLY BEFORE t -- a one-day
    ``shift(1)`` ahead of the rolling window, so r_t itself is never part
    of its own normalizer (identical no-look-ahead convention to
    single_day_extreme_decline_reversal_long.trigger_zscore, applied to
    BTC-USD instead of SPY). NaN wherever r_t or the trailing window is
    undefined (the first ``vol_window + 1`` days of ``close``, plus any
    non-finite input)."""
    values = pd.to_numeric(close, errors="coerce").astype(float)
    logret = np.log(values).diff()
    trailing_vol = logret.shift(1).rolling(
        vol_window, min_periods=vol_window).std(ddof=1)
    return logret / trailing_vol


def btc_riskoff_regime(btc_close: pd.Series, *, vol_window: int,
                       zscore_threshold: float, hold_days: int) -> pd.Series:
    """Boolean risk_on regime on BTC's OWN calendar-day index: False
    ("risk off" -- reduce QQQ to cash) for ``hold_days`` CALENDAR days
    starting the day BTC's z-score first crosses ``zscore_threshold``
    (inclusive of the trigger day itself), True otherwise. A CAUSAL
    backward-only rolling max of the raw trigger flag -- a later trigger
    extends the cooldown forward but a day's own flag never depends on a
    later day (see module docstring, "Trigger persistence"). NaN z-scores
    (the warm-up period) compare False under ``<=`` and are therefore
    automatically risk_on, with no special-casing required."""
    if isinstance(hold_days, bool) or not isinstance(hold_days, int) or hold_days <= 0:
        raise ValueError(f"{FAMILY_ID}: hold_days must be a positive int")
    z = btc_decline_zscore(btc_close, vol_window=vol_window)
    raw_trigger = (z <= zscore_threshold).fillna(False)
    risk_off = raw_trigger.rolling(hold_days, min_periods=1).max().astype(bool)
    return (~risk_off).rename("risk_on")


def make_signal_fn(ticker: str, index, frames: dict, *, btc_close=None):
    """SignalFn factory per the batch5 contract. ``btc_close`` is BTC-USD's
    RAW daily Close price series (Kraken spot, any calendar-day index) --
    never a pre-transformed z-score/regime/lagged series, matching
    ``money_market_fund_flow_riskoff_bounded_burnin.make_signal_fn``'s own
    ``flow_signal=`` convention. ``frames[ticker]`` is QQQ's own OHLC
    frame, required for the module contract but not read for anything
    beyond validating it exists (this family's signal never reads QQQ's
    own price).

    NOT REACHABLE THROUGH THE GENERIC scripts/run_abi1_mining.py DISPATCH:
    that driver's
    ``_build_curve`` calls ``module.make_signal_fn(symbol, index, frames)``
    positionally, with no ``btc_close`` argument -- ``discover_modules``'s
    own contract validation only checks that ``make_signal_fn`` is
    callable, never that it is USABLE through the real call signature.
    Under the generic path this factory therefore silently returns
    ``_empty_signal_fn`` (zero entries, zero trades) rather than raising,
    and the trade-count floor documented above applies to calling this
    factory directly with a real fetched ``btc_close`` series -- exactly
    the same shape already true of every other external-data-driven
    single_long family in this catalog (money_market_fund_flow_riskoff*,
    fed_net_liquidity_regime*, margin_debt_growth_regime -- none of which
    ``run_abi1_mining.py`` references any external-series kwarg for
    either). Scoring this family under ABI-1, like those, uses its
    own bespoke driver script, ``scripts/run_bitcoin_spillover_abi1.py``,
    which fetches real BTC data and calls this factory directly; the
    cost-realism gate also needs a live quote regardless."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)
    traded_index = pd.DatetimeIndex(index)
    if len(traded_index) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")

    if btc_close is None or btc_close.empty:
        return _empty_signal_fn

    regime = btc_riskoff_regime(
        btc_close, vol_window=VOL_WINDOW_DAYS, zscore_threshold=Z_THRESHOLD,
        hold_days=HOLD_DAYS)
    if regime.empty:
        return _empty_signal_fn

    available = signals.lag_boolean_regime_to_availability(
        regime, traded_index, lag_trading_days=LAG_TRADING_DAYS)
    if available.empty:
        return _empty_signal_fn

    return signals.make_external_regime_signal_fn(available)
