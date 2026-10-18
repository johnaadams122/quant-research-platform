"""Batch5 candidate: defensive_sector_recession_signal_rotation
(rotation_two_asset).

Implements an already-catalogued mechanism from
``data/mining/catalog.json`` in the ABI-1 batch5 module contract. The
mechanism, the parameters and the traded instruments are exactly what the
catalog entry specifies; nothing is tuned here.

Catalog mechanism (verbatim, ``data/mining/catalog.json``, family_id
"defensive_sector_recession_signal_rotation", added_date 2026-08-29):
"Rotate into Utilities (XLU) as a defensive holding when the yield curve
signals rising recession risk, and hold the broad market (SPY) otherwise.
The recession-risk signal is the 10-year minus 3-month Treasury yield
spread (10y ~ ^TNX, 3-month ~ ^IRX); when the spread inverts (short rates
exceed long rates), switch into XLU, following the documented pattern that
defensive sectors like utilities outperform around economic slowdowns.
Rebalance monthly on signal changes; a positively-sloped curve holds SPY
instead." Source: Estrella & Mishkin (1996), "The Yield Curve as a
Predictor of U.S. Recessions," FRBNY Current Issues in Economics and
Finance, Vol. 2, No. 7; defensive-sector response per Sam Stovall's S&P's
Guide to Sector Rotation (both cited verbatim in the catalog entry).

Rotation target, not a single-asset signal: this module's ``make_target_fn``
is a THIN factory wrapping ``mining.signals.make_external_regime_target_fn``
(the rotation-shape counterpart of ``make_external_regime_signal_fn``, and
the SAME factory ``margin_debt_deleverage_synthetic_leverage_flip.py``
already uses for its own external-regime rotation) fed a precomputed
boolean risk-on ``pd.Series`` -- NOT ``make_moving_average_regime_target_fn``
(this batch's usual MA-crossover factory, used by e.g.
``usd_strength_risk_regime.py``), because this family's regime rule is a
YIELD-CURVE SPREAD between two external Treasury-yield series, not a single
ticker's price versus its own moving average. All regime-decision math is
delegated to that one shared, separately tested factory; this module
only builds the boolean series it is fed.

Real measurement 1 -- units: yfinance's ``^TNX``/``^IRX`` ``Close`` values are
ALREADY directly comparable percentage-point yields -- no ``/100`` or
``/10`` rescale is needed (unlike, say, ``data_pipeline.fetch_iv_proxy``'s
own ``^VIX``/``^VXN`` handling, which DOES divide by 100). Both series
quote in the low-single-digit percent range a 10-year/3-month Treasury
yield pair should be in, and are directly subtractable.

Real measurement 2 -- curve_lookback_days=20 is applied as a rolling
20-TRADING-DAY (not calendar-day) MEAN of the raw daily spread
(``^TNX.Close - ^IRX.Close``), computed once over the full ``^TNX``/``^IRX``
common calendar (their real fetched history intersected, tz-naive, sorted,
de-duplicated keep-last). This is a smoothing window on the spread itself,
not a lookback on anything else the catalog parameter name could otherwise
be read as (e.g. a moving-average window on either yield individually) --
the catalog's own mechanism text says "when the spread inverts", so the
smoothing target is the spread.

Real measurement 3 -- negative signal yields: yfinance's ``^IRX``
history carries a handful of sessions in March 2020 (the COVID
flight-to-quality episode, when 3-month T-bill demand briefly pushed the
quoted yield NEGATIVE) with a negative Open and/or Close. This is
economically genuine data (a well-documented historical event), not a data
error -- but it would break TWO independent, unconditional positivity
checks in this pipeline's shared code:

  1. ``scripts/run_batch5_mining.py``'s ``load_frame`` normalizes every
     symbol (traded OR signal) via ``run_batch3_mining._normalize_ohlc``.
  2. ``mining.rotation_engine.run_two_asset_rotation``'s own
     ``_validate_frame`` (called on every traded AND signal frame).

  HOW IT IS HANDLED: both
  ``_normalize_ohlc`` and ``_validate_frame`` take an explicit
  ``require_positive: bool = True`` parameter -- True by default, so a
  TRADED instrument's price must be strictly positive -- and
  ``scripts/run_abi1_mining.py``'s own upfront OHLC-loading loop
  computes, once per run, which symbols are traded by ANY plan in the
  batch and passes ``require_positive=False`` only for a symbol that is
  signal-only across the WHOLE run (never traded anywhere) -- exactly
  ``^TNX``/``^IRX`` for this family, never SPY/XLU. ``mining.
  rotation_engine.run_two_asset_rotation`` itself always validates its
  ``signal_frames`` with ``require_positive=False`` (per its own
  docstring, "signal_frames are decision inputs only" -- never priced or
  traded, so a tradeable price's positivity check does not apply to
  them). This module's
  OWN code never requires TNX/IRX positivity -- it reads only ``Close`` and
  only checks finiteness (see ``_require_frame``/``_clean_close`` below) --
  so nothing here depends on it. A negative signal Close therefore
  survives normalization unmodified (no floor, no workaround), and this
  family is scorable through the generic ABI-1 driver.

Real measurement 4 -- curve inversion is economically real, not an
artifact: over the full ``^TNX``/``^IRX`` common calendar the smoothed
spread inverts (goes <= 0) in several distinct multi-month episodes,
matching the well-documented historical yield-curve inversions, so the
signal is not degenerate.

Design decision -- how "rebalance monthly" is implemented (resolved and
disclosed here): ``PARAMETERS["rebalance"] ==
"monthly"`` means "only re-evaluate/switch on month-end decision bars";
non-decision bars carry the prior state forward unchanged (the same
"last trading day of the month" convention ``vol_managed_intl_country_etf_
pooled.py``'s own docstring establishes: bar t is a decision bar iff the
NEXT bar in the calendar falls in a different month). This module achieves
that WITHOUT any extra machinery inside ``make_target_fn``'s returned
closure: the boolean risk-on ``pd.Series`` built once in ``make_target_fn``
is defined (non-NaN) ONLY at month-end dates of the ``^TNX``/``^IRX``
common calendar -- every other date is left NaN. ``signals.
make_external_regime_target_fn`` already (a) drops every NaN entry from the
series it is handed (``state.dropna()``, in its own already-tested code)
and (b) looks up the LATEST remaining (i.e. latest defined, i.e. latest
month-end) entry at or before "today" on every call (`known =
state.loc[state.index <= today]; ... known.iloc[-1]`). So a sparse,
month-end-only input series is ALREADY forward-carried across every
intra-month trading day by that factory's own existing "as-of" mechanism --
restricting TRANSITIONS to month-end dates a second time, inside this
module's own closure, would be redundant machinery duplicating what the
shared factory already does. This is a deliberate interpretation
choice, not an oversight: the alternative (re-evaluate the smoothed spread
literally every day, still only checking cross above zero once, without
using the sparse-series trick) would need to pass the traded
calendar itself into ``make_target_fn`` and hand-roll the day-by-day
carry-forward logic that ``make_external_regime_target_fn`` was already
built to provide.

Interpretation choice -- WHICH calendar defines "last trading day of the
month": this module computes month-end flags on the ``^TNX``/``^IRX``
common calendar (the bond-market signal's own aligned calendar), not on the
SPY/XLU traded calendar. The two calendars occasionally differ by a session
around a bond-market-only holiday (e.g. Columbus Day/Veterans Day, when
SIFMA closes but NYSE does not). This is not a defect: ``make_external_
regime_target_fn``'s own "as-of" design is explicitly built to accept an
external signal series dated on a DIFFERENT calendar than the traded legs
(see ``margin_debt_deleverage_synthetic_leverage_flip.py``'s identical
FINRA-calendar-vs-traded-calendar precedent) -- a month-end reading dated
one bond-market session earlier or later than the equivalent equity-market
month-end merely shifts, by at most one session, WHEN the new month's
regime reading first becomes visible to the traded legs; it never changes
WHAT that reading is or lets a future date leak into it.

No look-ahead: the smoothed spread at date t is a rolling
``curve_lookback_days``-session mean of ``^TNX.Close - ^IRX.Close`` ending
AT t (inclusive), using only ``.rolling(window, min_periods=window)`` --
structurally incapable of reading a future close. The boolean risk-on
series is computed ONCE, over the full available ``^TNX``/``^IRX`` history,
at ``make_target_fn`` build time (the SAME "precompute once from full
history, then as-of filter per call" pattern ``margin_debt_deleverage_
synthetic_leverage_flip.py`` already uses) --
safe specifically because every value in that precomputed series is itself
causal (uses only data <= its own date), so the later per-call "as-of"
filter (``state.index <= today``) can never expose a future date's already-
computed reading before that date has actually occurred in the truncated
frames the engine hands to ``target()`` at each decision bar.

Interpretation choices beyond the catalog's own explicit text:

1. A month-end date whose smoothed spread is still undefined (fewer than
   ``curve_lookback_days`` common-calendar observations, i.e. only possible
   in the earliest few weeks of ``^TNX``/``^IRX`` history -- both series'
   real history predates SPY's and XLU's by decades, so this warm-up window
   is never reached by any real traded session) is left NaN rather than
   resolved to a default direction, and is dropped by ``make_external_
   regime_target_fn``'s own ``dropna()`` exactly like every other undefined
   reading -- "no signal yet" for that month, not a silently-chosen
   default.
2. Duplicate dates in either yield series keep the LAST occurrence before
   sorting, matching the de-duplication convention already used elsewhere
   in this batch (e.g. ``betting_against_beta_low_vol.py``).
3. The smoothed spread compares to a strict ``> 0.0`` threshold for
   risk-on (SPY); ``<= 0.0`` (a flat or inverted curve) selects XLU --
   matching the catalog text's own polarity ("when the spread inverts ...
   switch into XLU ... a positively-sloped curve holds SPY instead") and
   treating an exactly-flat curve (spread == 0, a measure-zero edge case)
   as risk-off, the conservative reading of "not positively-sloped."
4. ``^TNX``/``^IRX`` Close values are read directly with NO positivity
   requirement (see Real measurement 3 above) -- only finite-numeric
   coercion (``pd.to_numeric(..., errors="coerce")``) and a
   presence/"Close column exists" check, matching every other module's
   ``_require_frame`` convention but deliberately NOT reusing ``batch3.
   _normalize_ohlc``'s stricter equity-oriented positivity gate for these
   two signal tickers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining import signals

FAMILY_ID = "defensive_sector_recession_signal_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "XLU"]        # [risk, defensive]
SIGNAL = ["^TNX", "^IRX"]      # external price-frame tickers the generic
                                # dispatch auto-loads -- long/short Treasury
                                # yield proxies, never traded
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (verified via json.dumps(sort_keys=True) byte-identity in this
# module's own test, which reads that file directly rather than
# duplicating a value here that could drift).
PARAMETERS = {
    "short_yield_ticker": "^IRX",
    "long_yield_ticker": "^TNX",
    "curve_lookback_days": 20,
    "rebalance": "monthly",
}
_CURVE_LOOKBACK_DAYS = int(PARAMETERS["curve_lookback_days"])
# A conservative sanity floor for a full-history run: this is a genuinely
# low-frequency regime rotation (it switches only when the yield curve
# inverts or un-inverts, a handful of times per few decades), so the floor
# is small. Same conservative-floor convention as every sibling module.
EXPECTED_MIN_TRADES = 5


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series -- Real
    measurement 3's deliberate choice: NO positivity requirement (a
    Treasury yield's genuine range includes zero and slightly-negative
    prints), only finite-numeric coercion."""
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                       index=index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[close.notna()]
    return close


def _month_end_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """True where the NEXT date in ``index`` falls in a different calendar
    month than the current one -- the last date is always False (nothing
    is knowable about a following month from this calendar alone). Same
    convention as ``vol_managed_intl_country_etf_pooled.py``'s own
    ``_month_end_flags``, defined locally here per this batch's
    self-contained-module convention."""
    months = index.year.to_numpy() * 12 + index.month.to_numpy()
    flags = np.zeros(len(index), dtype=bool)
    if len(index) > 1:
        flags[:-1] = months[1:] != months[:-1]
    return flags


def curve_risk_on_series(tnx_close: pd.Series, irx_close: pd.Series) -> pd.Series:
    """The precomputed, sparse boolean risk-on series this module feeds to
    ``signals.make_external_regime_target_fn``: defined (0.0/1.0) ONLY at
    month-end dates of the ``tnx_close``/``irx_close`` common calendar whose
    smoothed spread is already warm; NaN everywhere else (both non-month-
    end dates and undefined/warm-up month-ends), which that factory's own
    ``.dropna()`` strips before use -- see the module docstring's "Design
    decision" section for why a sparse month-end-only series already
    achieves monthly rebalancing with no further restriction needed.

    ``daily_spread = tnx_close - irx_close`` (10y minus 3mo, in percentage
    points -- see the module docstring's "Real measurement 1" on why no
    unit rescale is needed), smoothed by a rolling ``curve_lookback_days``-
    session mean ending at each date (causal, no look-ahead). risk_on =
    smoothed_spread > 0.0 (a positively-sloped curve holds SPY); <= 0.0
    (flat or inverted) selects XLU.
    """
    common = tnx_close.index.intersection(irx_close.index).sort_values()
    if len(common) == 0:
        return pd.Series(dtype=float)
    tnx = tnx_close.reindex(common)
    irx = irx_close.reindex(common)
    daily_spread = tnx - irx
    smoothed = daily_spread.rolling(
        _CURVE_LOOKBACK_DAYS, min_periods=_CURVE_LOOKBACK_DAYS).mean()

    month_end = _month_end_flags(common)
    defined = smoothed.notna().to_numpy() & month_end
    risk_on = (smoothed > 0.0).to_numpy()
    values = np.where(defined, risk_on.astype(float), np.nan)
    return pd.Series(values, index=common)


def make_target_fn(frames: dict[str, pd.DataFrame]):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED + SIGNAL, handed once; only their
    presence is checked here -- every price the rule reads comes from
    ``^TNX``/``^IRX``'s own full Close history, precomputed once into a
    causal boolean series (see ``curve_risk_on_series``), then delegated
    entirely to ``signals.make_external_regime_target_fn``'s own already-
    tested per-call "as-of" lookup (which itself only reads TRADED[0]/
    TRADED[1] out of whatever truncated ``frames`` it receives per call --
    it never re-touches SIGNAL directly)."""
    for key in (TRADED[0], TRADED[1], SIGNAL[0], SIGNAL[1]):
        if key not in frames:
            raise ValueError(
                f"{FAMILY_ID}: make_target_fn requires a frame for {key}")
    tnx_frame = _require_frame(frames, SIGNAL[0])
    irx_frame = _require_frame(frames, SIGNAL[1])
    tnx_close = _clean_close(tnx_frame)
    irx_close = _clean_close(irx_frame)

    risk_on = curve_risk_on_series(tnx_close, irx_close)
    return signals.make_external_regime_target_fn(
        risk_on, risk_ticker=TRADED[0], defensive_ticker=TRADED[1])
