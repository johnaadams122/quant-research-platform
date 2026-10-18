"""Batch6 candidate ``cboe_putwrite_vol_regime_rotation`` (rotation_two_asset).

Cboe S&P 500 PutWrite Index (``^PUT``) systematic overlay, rotating with
SPY on a realized-volatility regime signal. Structurally copied verbatim
from ``vol_managed_smallcap_leverage_switch.py`` (same
``realized_variance`` / ``expanding_median`` / ``month_end_mask`` /
``decision_series`` / ``make_target_fn`` shape, engine =
``mining.rotation_engine.run_two_asset_rotation``); only the traded pair,
signal source and PARAMETERS differ.

Why ``rotation_two_asset`` and not a naive buy-and-hold: a pure long
``^PUT`` position produces ~1 trade for the whole history and would be
refused outright under ``constants.MIN_HOLDOUT_TRADES=20``. Rotating
against SPY on SPY's own realized-vol regime (elevated vol -> hold the
overlay; calm vol -> hold SPY for uncapped upside) generates real,
measurable turnover. Economic rationale sourced directly from Cboe's own
PUT index dashboard text: PUT-write/buy-write
strategies "tend to outperform a related stock index when stock markets
are in range-bound or bearish regimes ... and underperform in bull market
regimes."

COST MODEL (real gap, resolved per this repo's standing
"disclosed default" convention -- see
``ASSUMED_COST_BPS`` below): ``^PUT`` is a non-tradable calculated index
with no real bid/ask, so the pipeline's usual
``cost_tax_inputs.cost_bps_from_quotes`` (fed by a real captured quote
sample) cannot price it, and ``scripts/run_abi1_mining.py``'s own
per-candidate cost gate (``execution_costs_for``) would refuse the whole
family with a "missing_data" skip the moment it tried to look up a ^PUT
quote that does not exist in the captured universe. This family is
therefore run through a bespoke driver
(``scripts/run_cboe_rotation_mining.py``), not the shared
``run_abi1_mining.py`` dispatch. SPY's leg still prices off a REAL captured quote (SPY is an
ordinary tradable ETF, already in the batch5 quote universe); only the
``^PUT`` leg uses the assumed constant.

DATA-QUALITY NOTE: ``^PUT``'s yfinance/Yahoo feed is expected to be dense
across its full history (apart from market closures such as the
September 2001 NYSE shutdown). The sibling overlay ``^BXM`` has shown
multi-week feed gaps, which is why a ``cboe_buywrite_vol_regime_rotation``
sibling is not built here.

Interpretation choices (matching the sibling's own documented choices)
-----------------------------------------------------------------------
* Signal source is SPY's OWN realized variance (self-referential: SPY is
  already a TRADED leg, so SIGNAL is empty, exactly like
  ``vol_managed_smallcap_leverage_switch`` using IWM's own RV).
* "ALL available history" for the expanding median = every SPY bar in the
  frame the driver hands the factory (SPY history starts 1993, well
  before PUT's 1996-08-02 start) -- the same reading the sibling module
  uses for IWM vs. TNA's later start. ``abi1_driver._align_rotation_
  frames`` trims the traded pair to their common inception (PUT's start)
  before the engine ever sees a bar, so an SPY-only decision computed
  before PUT existed is precomputed but never acted upon.
* "Last trading day of month" / no intra-month re-evaluation / as-of
  lookup on a bucket starting mid-month: identical to the sibling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILY_ID = "cboe_putwrite_vol_regime_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "PUT"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"

# Conservative, deliberately unflattering round-trip cost assumption for
# the non-tradable ^PUT leg (bps of notional), since there is no real
# quote to measure. Set well above a typical liquid equity or ETF quoted
# spread, on the theory that actually
# executing this exposure (via a proxy ETF or options) should cost MORE
# than a liquid single stock, not less. Disclosed default
# -- same convention as every other invented constant in this pipeline
# (e.g. IDLE_SWEEP_APY, MIN_POOLED_SR_VARIANCE_SAMPLES).
ASSUMED_COST_BPS = 25.0

PARAMETERS = {
    "signal": "realized_variance_21d_log_returns",
    "signal_source": "SPY",
    "rv_lookback_days": 21,
    "threshold": "expanding_median",
    "threshold_min_obs": 252,
    "rebalance": "monthly",
    "decision_bar": "last_trading_day_of_month",
    "low_variance_position": "SPY",
    "high_variance_position": "PUT",
    "assumed_cost_bps_round_trip": ASSUMED_COST_BPS,
}
EXPECTED_MIN_TRADES = 40  # configured, conservative floor (not a measured
# count) on completed segments across both legs: a monthly rotation over
# multi-decade SPY/PUT history should clear MIN_HOLDOUT_TRADES=20 easily,
# so a run below this floor points at a data or wiring problem.

_SIGNAL_SOURCE = PARAMETERS["signal_source"]
_RV_LOOKBACK_DAYS = int(PARAMETERS["rv_lookback_days"])
_THRESHOLD_MIN_OBS = int(PARAMETERS["threshold_min_obs"])
_LOW_VARIANCE_POSITION = PARAMETERS["low_variance_position"]
_HIGH_VARIANCE_POSITION = PARAMETERS["high_variance_position"]


def _signal_close(frames) -> pd.Series:
    """Validated SPY close series with a sorted, unique, tz-naive index."""
    if not isinstance(frames, dict) or _SIGNAL_SOURCE not in frames:
        raise ValueError(f"missing required frame for {_SIGNAL_SOURCE}")
    frame = frames[_SIGNAL_SOURCE]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{_SIGNAL_SOURCE} frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ValueError(f"{_SIGNAL_SOURCE} frame is missing Close")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{_SIGNAL_SOURCE} frame index must be sorted and unique")
    close = pd.to_numeric(frame["Close"], errors="coerce").astype(float)
    values = close.to_numpy()
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{_SIGNAL_SOURCE} Close values must be finite and positive")
    close.index = index
    return close


def realized_variance(close: pd.Series) -> pd.Series:
    """RV_t: trailing ``rv_lookback_days`` sum of squared log returns
    ending at t inclusive; NaN until the window is full."""
    log_returns = np.log(close).diff()
    return (log_returns ** 2).rolling(
        _RV_LOOKBACK_DAYS, min_periods=_RV_LOOKBACK_DAYS).sum()


def expanding_median(rv: pd.Series) -> pd.Series:
    """MED_t: median of all RV observations <= t; NaN until
    ``threshold_min_obs`` RV observations exist."""
    return rv.expanding(min_periods=_THRESHOLD_MIN_OBS).median()


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last bar of each calendar month present in ``index``
    (the final bar counts as its month's last bar)."""
    periods = index.to_period("M")
    mask = np.ones(len(index), dtype=bool)
    if len(index) > 1:
        mask[:-1] = periods[1:] != periods[:-1]
    return mask


def decision_series(close: pd.Series) -> pd.Series:
    """Object Series indexed by month-end dates: the target decided on that
    bar ('SPY' / 'PUT'), or None while the median is undefined."""
    rv = realized_variance(close)
    med = expanding_median(rv)
    defined = rv.notna() & med.notna()
    state = pd.Series([None] * len(close), index=close.index, dtype=object)
    low = defined & (rv <= med)
    high = defined & (rv > med)
    state[low.to_numpy()] = _LOW_VARIANCE_POSITION
    state[high.to_numpy()] = _HIGH_VARIANCE_POSITION
    return state[month_end_mask(close.index)]


def make_target_fn(frames: dict[str, pd.DataFrame]):
    """Rotation target factory: precompute every month-end decision once from
    the full-history SPY frame; the returned callable does one as-of lookup
    per bar on the truncated frames the engine hands it."""
    close = _signal_close(frames)
    decisions = decision_series(close)
    decision_dates = decisions.index.to_numpy(dtype="datetime64[ns]")
    decision_values = decisions.to_numpy(dtype=object)

    def target(truncated: dict[str, pd.DataFrame]):
        try:
            frame = truncated[_SIGNAL_SOURCE]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"target input must include the {_SIGNAL_SOURCE} frame") from exc
        if frame.empty:
            return None
        today = pd.Timestamp(frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        stamp = np.datetime64(today.to_datetime64(), "ns")
        position = int(np.searchsorted(decision_dates, stamp, side="right")) - 1
        if position < 0:
            return None
        value = decision_values[position]
        return None if value is None else str(value)

    return target
