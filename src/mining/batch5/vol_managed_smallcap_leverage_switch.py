"""Batch5 candidate ``vol_managed_smallcap_leverage_switch`` (rotation_two_asset).

Volatility-Managed Small-Cap Leverage Switch (TNA/IWM) -- a two-state
discretization of Moreira & Muir (2017), "Volatility-Managed Portfolios",
J. Finance 72(4). Source of record for the rule: the candidate's
``rule_pseudocode`` from its accepted curation record, implemented verbatim with the given parameters:

  SIGNAL     RV_t = sum over the trailing 21 trading days ending at t
             (inclusive) of ln(Close_s / Close_{s-1})**2, from IWM's
             auto-adjusted daily closes.
  THRESHOLD  MED_t = median of {RV_u : u <= t} over ALL available history
             (expanding window); undefined -- and no position taken --
             until at least 252 RV observations exist.
  STATE      LOW_t = (RV_t <= MED_t)              (inclusive tie -> low)
  DECISION   once per calendar month on that month's last trading day t;
             the target executes at the t+1 open and is held unchanged
             until the next month's decision executes. No intra-month
             re-evaluation, no hysteresis, no other filter.
  TARGET     'TNA' if LOW_t else 'IWM'; None while MED_t is undefined.

Engine contract (mining.rotation_engine.run_two_asset_rotation): the
target function receives every traded frame truncated through bar t and
returns the asset key to hold from the t+1 open. Both traded legs share
one calendar; IWM is itself a traded leg, so the signal frame is always in
the dictionary (SIGNAL is empty).

No look-ahead / performance: every series is precomputed ONCE in the
factory from the full-history frames with rolling / expanding operations
only, so the value stored for date t uses closes <= t. The per-bar target
call is a single binary search on the month-end decision dates (as-of
lookup) -- no statistic is ever recomputed over the growing slice.

Holdout-blind: the factory's precomputation may use history BEFORE a bucket
(the median needs 252 RV observations of warm-up) but the value at any date
t depends only on closes <= t, so nothing after the last bar of a bucket can
reach a decision inside it.

Interpretation choices (where the pseudocode leaves room):
* "Last trading day of month" = the last bar of each calendar month present
  in the IWM frame's own calendar. The frame's final bar is treated as its
  month's decision bar; the engine never calls target_fn on a bucket's final
  bar, so this can never execute -- it only keeps the series total.
* "ALL available history" = every IWM bar in the frame the driver hands the
  factory (yfinance IWM history starts 2000-05-26, well before TNA's
  2008-11-19). A defined median at TNA's first month-end requires exactly
  this reading; using
  only post-TNA history would leave the first ~13 months flat.
* The target on bar t is the decision of the LATEST month-end <= t (as-of
  lookup). A bucket that starts mid-month (the lookback-padded holdout
  bucket) therefore inherits the standing decision instead of going flat
  until its first in-bucket month-end, matching "held unchanged until the
  next month's decision executes".
* "Auto-adjusted daily closes" = the Close column as delivered by the
  driver (data_pipeline.fetch_ohlc uses yfinance ``auto_adjust=True``); the
  module does no further adjustment.
* Frame validation mirrors mining.signals._aligned_frames: a missing IWM
  frame or a non-finite / non-positive close raises ValueError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILY_ID = "vol_managed_smallcap_leverage_switch"
SHAPE = "rotation_two_asset"
TRADED = ["TNA", "IWM"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "IWM"
# Copied verbatim from the accepted curation record (driver drift check compares
# json.dumps(sort_keys=True) byte-for-byte).
PARAMETERS = {
    "signal": "realized_variance_21d_log_returns",
    "signal_source": "IWM",
    "rv_lookback_days": 21,
    "threshold": "expanding_median",
    "threshold_min_obs": 252,
    "rebalance": "monthly",
    "decision_bar": "last_trading_day_of_month",
    "low_variance_position": "TNA",
    "high_variance_position": "IWM",
}
EXPECTED_MIN_TRADES = 60

_SIGNAL_SOURCE = PARAMETERS["signal_source"]
_RV_LOOKBACK_DAYS = int(PARAMETERS["rv_lookback_days"])
_THRESHOLD_MIN_OBS = int(PARAMETERS["threshold_min_obs"])
_LOW_VARIANCE_POSITION = PARAMETERS["low_variance_position"]
_HIGH_VARIANCE_POSITION = PARAMETERS["high_variance_position"]


def _signal_close(frames) -> pd.Series:
    """Validated IWM close series with a sorted, unique, tz-naive index."""
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
    bar ('TNA' / 'IWM'), or None while the median is undefined."""
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
    the full-history IWM frame; the returned callable does one as-of lookup
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
