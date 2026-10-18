"""Candidate: vix1d_vix9d_inversion_deleverage_trigger (rotation_two_asset).

VIX1D/VIX9D short-end term-structure deleverage trigger on a Select Sector
SPDR. Default to holding a SYNTHETIC leveraged version of the underlying
sector SPDR whenever the very-short-end vol term structure is in normal
contango (``^VIX1D`` < ``^VIX9D``); switch to the UNLEVERED underlying
whenever the short end inverts (``^VIX1D`` >= ``^VIX9D``, signaling
unusually large near-dated (~24h) event risk); revert to the leveraged
position once contango resumes. This family has no entry in
``data/mining/catalog.json``, so there is no catalog entry to pin
PARAMETERS against (the catalogued modules' tests pin theirs byte-for-byte
against their catalog entry). Its driver,
``scripts/run_vix1d_vix9d_deleverage_mining.py``, runs the data fetch, the
cost input and the rough backtest.

Distinctness from existing catalog families (no catalog entry mentions
"vix1d"/"vix9d"/"deleverage"):

* ``vix_term_structure_carry`` (SVXY long/flat on ^VIX/^VIX3M, i.e. the
  30-day VIX against the 3-MONTH tenor) is a different pair of tenors (30d/3mo, not 1d/9d), a
  different traded instrument (a packaged short-vol ETF, not a synthetic
  leveraged sector position), and a different action (long/flat, not a
  leverage toggle on a still-always-invested sector position).
* ``vvix_vix_derisking_regime`` (VVIX/VIX ratio -> SPY de-risk to cash) and
  ``vix_spike_reversal_equity_buy`` (VIX spike mean-reversion buy) are both
  economically unrelated triggers (vol-of-vol and spike-mean-reversion,
  not term-structure ordering) on a different underlying (SPY, not a
  sector SPDR).
* ``gayed_200dma_letf_rotation`` (catalog) rotates
  leveraged-ETF exposure on a 200-day PRICE trend, not an implied
  volatility term-structure signal.
* ``vol_managed_smallcap_leverage_switch`` is the
  same rotation_two_asset SHAPE (a leverage toggle between a levered and
  unlevered sibling) but gated on REALIZED variance vs. its own expanding
  median -- a different signal family, and the polarity is opposite: that
  family switches leverage ON when realized vol is LOW; this one switches
  leverage OFF when IMPLIED near-dated vol is inverted (elevated relative
  to the 9-day tenor), a distinct economic mechanism (term-structure
  inversion pricing event risk, not a realized-variance regime).

VENDOR DATA. ``^VIX1D`` and ``^VIX9D`` can return only the latest
snapshot (a single row) from yfinance via every call shape
(``Ticker.history(period=...)``, ``Ticker.history(start=..., end=...)`` and
``yf.download(...)``), while ``^VIX`` itself, fetched via the IDENTICAL
``data_pipeline.fetch_ohlc`` call, returns its full history -- so this is
specific to these short-tenor CBOE index tickers, not a general
yfinance/network outage. The pipeline never fabricates history or
substitutes another vendor's data: the driver reports what came back and
stops before the backtest when too little common history remains.
``EXPECTED_MIN_TRADES`` below is 0: this module asserts no minimum trade
count. Nothing in this module is specific to that vendor response: given
full history for both tickers the same code backtests on real data. ``^VIX1D``'s history
begins 2023-04-24 (per this repo's ``fetch_vix1d`` docstring), so even full
coverage gives a short in-sample window.

Interpretation choices
-----------------------
* Underlying = XLK (Technology Select Sector SPDR, State Street, inception
  1998-12-16) -- the rule names no specific Select Sector SPDR; XLK was
  chosen for deep, liquid, uninterrupted real OHLC
  history and thematic consistency with this repo's leveraged-index
  research. Not otherwise reserved by a conflicting existing
  catalog mechanism for this same VIX1D/VIX9D idea (the sector-rotation
  families that DO use XLK, e.g. absolute/relative sector momentum, are
  economically unrelated two-asset rotations, not a leverage toggle).
* Leverage = 2.0 -- ``data_pipeline.build_synthetic_leveraged``'s own
  default (the synthetic builder this family uses, not
  ``build_synthetic_3x``); the rule specifies no multiple.
* Financing: ``build_synthetic_leveraged`` is called with
  ``annual_financing_rate=`` the REAL, time-varying ``^IRX`` series (via
  ``data_pipeline.fetch_risk_free_rate``), matching this repo's own
  established driver precedent (``scripts/run_synthetic_leveraged_dip_
  mining.py``'s ``_build_underlying_and_synthetic``) rather than the
  function's constant fallback. Its own defaults otherwise apply
  (``annual_expense_ratio=0.0``, ``financing_spread=0.0100`` -- 100bps,
  the single-stock/retail-margin convention, not ``build_synthetic_3x``'s
  50bps institutional-swap ETF convention -- a disclosed mismatch, since
  XLK is a packaged fund not a single stock, kept because this family is
  defined on ``build_synthetic_leveraged`` (PARAMETERS["synthetic_builder"]);
  ``dividend_yield=0.0``,
  correct since ``data_pipeline.fetch_ohlc``'s ``auto_adjust=True`` close
  is already total-return).
* Trading cost basis for BOTH legs = XLK's own real measured spread (same
  rationale as ``synthetic_leveraged_dip_buy_pooled_largecap``'s own
  precedent: the leveraged leg is realized as real margin-financed shares
  of the SAME real fund, not a separately-quoted packaged leveraged
  product, so there is no distinct "leveraged product spread" to model --
  leverage's true extra cost is financing, already inside the synthetic
  build).
* Decision cadence = DAILY, re-checked and re-decided every session (no
  monthly/expanding-window smoothing) -- the rule's own language
  ("switch ... whenever the short end inverts ... revert ... once contango
  resumes") describes an immediate, same-day reaction, matching
  ``vix_term_structure_carry``'s own "rebalance: daily" precedent for a
  level-comparison (not moving-average) VIX-family signal.
* Tie (``VIX1D == VIX9D``) resolves to INVERSION (unlevered) -- the rule's
  own explicit ``>=`` on the inversion side.
* Undefined signal (a date with no ``^VIX1D``/``^VIX9D`` bar, or a
  non-finite/non-positive close) is FLAT (``None``), never fabricated as
  either leg -- matching ``vix_term_structure_carry``'s "undefined is not
  flat by rule, it is simply no signal today" convention. Since the
  engine holds the PRIOR position when ``target`` repeats the same value,
  and a rotation's ``target_fn`` returning ``None`` explicitly closes to
  cash (``rotation_engine.run_two_asset_rotation``), an undefined session
  goes to cash rather than silently carrying forward the last known
  target -- disclosed, not a claim that the mechanism ever intends a cash
  state (the raw mechanism never mentions one).

Engine contract (``mining.rotation_engine.run_two_asset_rotation``): on
decision bar t, ``target_fn`` receives every traded AND signal frame
truncated through bar t (``frame.iloc[:i+1]``, so bar t is the last row
and there is no bar t+1 to accidentally read) and returns the asset key to
hold from bar t+1's open. This candidate's rule needs no history beyond
today's own VIX1D/VIX9D close, so ``make_target_fn`` does no precompute at
factory time beyond a presence/shape check -- the returned callable reads
only ``truncated[SIGNAL[i]]["Close"].iloc[-1]`` each call, which is
already structurally look-ahead-safe by the engine's own contract (see
this module's own look-ahead test for a direct proof, not just an
argument from the contract).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FAMILY_ID = "vix1d_vix9d_inversion_deleverage_trigger"
SHAPE = "rotation_two_asset"

UNDERLYING = "XLK"
LEVERAGE = 2.0
_SYNTHETIC_ID = f"{UNDERLYING}_{int(LEVERAGE)}x_synthetic"

# [risk, defensive] per the driver contract's own ordering convention
# (mirrors vol_managed_smallcap_leverage_switch.TRADED = ["TNA", "IWM"]).
TRADED = [_SYNTHETIC_ID, UNDERLYING]
SIGNAL = ["^VIX1D", "^VIX9D"]
SNAPSHOTS: dict = {}
BENCHMARK = UNDERLYING

PARAMETERS = {
    "signal": "vix1d_vs_vix9d_term_structure",
    "signal_source_contango": "^VIX1D",
    "signal_source_backwardation_reference": "^VIX9D",
    "contango_condition": "VIX1D < VIX9D",
    "inversion_condition": "VIX1D >= VIX9D",
    "rebalance": "daily",
    "leveraged_position": _SYNTHETIC_ID,
    "unlevered_position": UNDERLYING,
    "underlying": UNDERLYING,
    "leverage": LEVERAGE,
    "synthetic_builder": "data_pipeline.build_synthetic_leveraged",
    "financing": "time_varying_real_irx_via_fetch_risk_free_rate",
    "financing_spread_bps": 100,
    "annual_expense_ratio": 0.0,
    "dividend_yield": 0.0,
    "trading_cost_basis": "underlying_real_quoted_spread_both_legs",
}

# No minimum trade count is asserted for this family (see the module
# docstring's "VENDOR DATA" paragraph).
EXPECTED_MIN_TRADES = 0

_TICKER_COLUMN_ATTR = "Close"


def synthetic_id_for(underlying_ticker: str) -> str:
    return f"{underlying_ticker}_{int(LEVERAGE)}x_synthetic"


def underlying_of(synthetic_ticker: str) -> str:
    suffix = f"_{int(LEVERAGE)}x_synthetic"
    if not synthetic_ticker.endswith(suffix):
        raise ValueError(
            f"{synthetic_ticker!r} is not a {suffix!r}-suffixed synthetic id")
    return synthetic_ticker[: -len(suffix)]


def synthetic_ohlc_from_close(dates, close, *, name: str) -> pd.DataFrame:
    """A daily-reset synthetic instrument has no real intraday range:
    ``Open_t = Close_{t-1}`` (``Open_0 = Close_0``), ``High = Low =
    Close`` -- the same economically-correct convention this repo already
    uses for every other synthetic leveraged series (matches
    ``synthetic_leveraged_dip_buy_pooled_largecap.synthetic_ohlc_from_
    close`` exactly; duplicated here rather than cross-imported so this
    module stays self-contained, the same convention every other
    candidate module in this package already follows -- see e.g.
    ``vix_term_structure_carry.py``/``vol_managed_smallcap_leverage_
    switch.py``, neither of which imports from a sibling module either)."""
    close = np.asarray(close, dtype=float)
    if len(close) == 0:
        return pd.DataFrame(
            {"Open": [], "High": [], "Low": [], "Close": []},
            index=pd.DatetimeIndex([], name="Date"))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": close, "Low": close, "Close": close},
        index=index)


def _require_ohlc_frame(frames, ticker: str, *,
                         require_finite_positive: bool) -> pd.DataFrame:
    """Structural validation always applies (frame present, non-empty, has
    every OHLC column, sorted/unique tz-naive-comparable index) -- a
    malformed FRAME is a real contract violation and fails closed at
    factory time regardless of which kind of ticker it is.

    ``require_finite_positive`` is True for the TRADED legs (real prices
    the engine will fill and mark trades against must never be NaN/<=0
    anywhere in history) and False for the SIGNAL tickers: an occasional
    non-finite ^VIX1D/^VIX9D print is a live-feed gap, not a contract
    violation, and must degrade to "no decision that day" (handled
    per-bar inside ``target`` below), matching ``vix_term_structure_
    carry``'s own established "an occasional data glitch should degrade
    to 'no decision today,' not halt the whole backtest" convention --
    NOT raise at factory time and halt every other bar's decision too."""
    if not isinstance(frames, dict) or ticker not in frames:
        raise ValueError(f"{FAMILY_ID}: missing required frame for {ticker}")
    frame = frames[ticker]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(
            f"{FAMILY_ID}: {ticker} frame must be a non-empty DataFrame")
    missing = [c for c in ("Open", "High", "Low", "Close") if c not in frame.columns]
    if missing:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame is missing {missing}")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame index must be sorted and unique")
    if require_finite_positive:
        close = pd.to_numeric(frame["Close"], errors="coerce")
        if close.isna().any() or not np.isfinite(close.to_numpy()).all():
            raise ValueError(f"{FAMILY_ID}: {ticker} Close values must be finite numeric")
        if (close <= 0).any():
            raise ValueError(f"{FAMILY_ID}: {ticker} Close values must be positive")
    return frame


def make_target_fn(frames: dict[str, pd.DataFrame]):
    """Rotation target factory per the driver contract: ``frames`` carries
    full-history OHLC for both TRADED legs and both SIGNAL tickers. No
    precompute is needed (a pure per-bar level comparison, no lookback
    window) beyond validating every required frame exists and is
    structurally well-formed -- done once here so a malformed input fails
    at factory time rather than silently degrading every bar of a real
    backtest. Only the TRADED legs are required finite/positive at this
    stage; see ``_require_ohlc_frame``'s own docstring for why the SIGNAL
    tickers are not."""
    for ticker in TRADED:
        _require_ohlc_frame(frames, ticker, require_finite_positive=True)
    for ticker in SIGNAL:
        _require_ohlc_frame(frames, ticker, require_finite_positive=False)

    def target(truncated: dict[str, pd.DataFrame]):
        vix1d_frame = truncated.get(SIGNAL[0])
        vix9d_frame = truncated.get(SIGNAL[1])
        if vix1d_frame is None or vix9d_frame is None:
            raise ValueError(
                f"{FAMILY_ID}: target input must include both {SIGNAL[0]!r} "
                f"and {SIGNAL[1]!r}")
        if vix1d_frame.empty or vix9d_frame.empty:
            return None
        vix1d_close = float(vix1d_frame["Close"].iloc[-1])
        vix9d_close = float(vix9d_frame["Close"].iloc[-1])
        if not (math.isfinite(vix1d_close) and math.isfinite(vix9d_close)):
            return None
        if vix1d_close <= 0.0 or vix9d_close <= 0.0:
            return None
        # Contango (VIX1D < VIX9D) -> leveraged; inversion (VIX1D >= VIX9D,
        # tie included) -> unlevered.
        return TRADED[0] if vix1d_close < vix9d_close else TRADED[1]

    return target
