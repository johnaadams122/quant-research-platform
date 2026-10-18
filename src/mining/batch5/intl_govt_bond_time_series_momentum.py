"""Batch5 candidate ``intl_govt_bond_time_series_momentum`` (rotation_two_asset).

International Government Bond Absolute Momentum (BWX/BIL) -- single-asset
absolute (time-series) momentum applied to ex-US developed-market sovereign
debt, following Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum",
JFE 104(2) (the family's catalog `source` cites Asness, Moskowitz & Pedersen
(2013), "Value and Momentum Everywhere", which documents the same trailing-
return time-series-momentum construction across asset classes including
international bonds). Source of record for the rule: the catalog
mechanism ("Positive -> hold BWX; negative -> rotate to BIL. Rebalance
monthly."), implemented with the given parameters as:

  SIGNAL     BWX's OWN trailing 12-calendar-month total return, evaluated on
             BWX's own bar calendar: the month-end close of month m divided
             by the month-end close of month m-12, minus one.
  RULE       positive (strictly > 0) -> hold BWX; negative, zero, or
             undefined (no month m-12 anchor exists yet) -> hold BIL.
  DECISION   once per calendar month on that month's last trading day t; the
             target executes at the t+1 open and is held unchanged until the
             next month's decision executes. No intra-month re-evaluation,
             no hysteresis, no other filter.
  TARGET     'BWX' if the trailing return is defined and positive, else
             'BIL' (this covers negative, exactly zero, AND undefined --
             see the Interpretation choice on zero and undefined returns).

Engine contract (mining.rotation_engine.run_two_asset_rotation): the target
function receives every traded frame truncated through bar t and returns the
asset key to hold from the t+1 open. Both traded legs share one calendar;
BWX is itself a traded leg (the risk asset), so the signal frame is always
in the dictionary handed to ``make_target_fn`` (SIGNAL is empty). BIL is
modeled as a REAL traded rotation leg with its own OHLC price series and its
own empirical round-trip cost -- not a synthetic zero-yield cash placeholder
-- per the catalog's own "rotate to BIL" wording and BIL's status as a real,
liquid, small-positive-return T-bill ETF.

No look-ahead / performance: every series is precomputed ONCE in the factory
from the full-history frames with grouped / shifted operations only, so the
value stored for month m uses closes <= that month's own last bar. The
per-bar target call is a single binary search on the month-end decision
dates (as-of lookup) -- no statistic is ever recomputed over the growing
slice.

Holdout-blind: the factory's precomputation may use history BEFORE a bucket
(month m's decision needs month m-12's close, i.e. up to 12 months of
warm-up), but the value at any date t depends only on closes <= t, so
nothing after the last bar of a bucket can reach a decision inside it.

Interpretation choices (where the pseudocode leaves room):
* "Trailing 12-month total return" = month-end close of month m divided by
  the month-end close of calendar month m-12 (PeriodIndex arithmetic,
  adapted from fx_dollar_index_trend_following's ``month_end_targets``),
  minus one. Total return is the pipeline's Close as delivered
  (data_pipeline fetches ``auto_adjust=True``, i.e. distribution-adjusted,
  so no further dividend adjustment is applied here).
* "Positive -> hold BWX; negative -> rotate to BIL" leaves two cases
  unstated: an EXACTLY ZERO trailing return, and an UNDEFINED trailing
  return (the anchor month m-12 has no bars in the frame -- true for every
  month-end inside the first ``lookback_months`` of BWX's own history).
  Both resolve to BIL: a return that is not STRICTLY positive is treated as
  the "negative" branch (mirrors fx_dollar_index_trend_following's identical
  choice for its own long/cash rule), and an undefined return is read the
  same way: "negative or undefined -> target 'BIL'". This is a
  deliberate departure from vol_managed_smallcap_leverage_switch's
  convention (undefined threshold -> None / no position): here undefined
  is not a distinct third state, it is folded into the same BIL branch as
  negative, so ``make_target_fn`` never returns None once the engine hands
  it at least one bar of history -- there is no "no position taken" state
  for this family, only "hold BWX" or "hold BIL".
* "Last trading day of month" = the last bar of each calendar month present
  in BWX's own frame's calendar (mirrors vol_managed_smallcap_leverage_
  switch's ``month_end_mask``, including its final-bar convention -- the
  engine never calls ``target_fn`` on a bucket's final bar, per
  ``run_two_asset_rotation``'s truncation loop, so which way that single
  edge case resolves cannot change any decision it actually returns).
* Before the FIRST month-end decision ever fires (the truncated frame is too
  short to contain even one full calendar month), the as-of lookup finds no
  prior decision date; this also resolves to BIL, consistent with "undefined
  -> BIL" above rather than introducing a separate flat/None state.
* "Auto-adjusted daily closes" = the Close column as delivered by the driver
  (data_pipeline.fetch_ohlc uses yfinance ``auto_adjust=True``); the module
  does no further adjustment.
* Frame validation mirrors mining.signals._aligned_frames: a missing BWX
  frame or a non-finite / non-positive close raises ValueError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILY_ID = "intl_govt_bond_time_series_momentum"
SHAPE = "rotation_two_asset"
TRADED = ["BWX", "BIL"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "BWX"
# Copied verbatim from the pre-registered catalog entry (driver drift check
# compares json.dumps(sort_keys=True) byte-for-byte). "cash_proxy" is a
# catalog LABEL, not an instruction to model BIL as synthetic cash -- see
# the module docstring's "Engine contract" paragraph.
PARAMETERS = {
    "lookback_months": 12,
    "signal": "trailing_total_return",
    "rebalance": "monthly",
    "cash_proxy": "BIL",
}
# Expected segment count over the full history of a monthly two-asset
# rotation (one segment per switch plus the initial open). Uses the
# full-history view (same convention as the other batch5 rotation modules
# in this directory), not a partial in-sample-only count, since the driver's
# own holdout split is applied on top of this module. Reported alongside a
# run's results as a sanity reference; it gates nothing.
EXPECTED_MIN_TRADES = 30

_LOOKBACK_MONTHS = int(PARAMETERS["lookback_months"])
_RISK_ASSET = TRADED[0]        # "BWX": held on a strictly positive signal
_DEFENSIVE_ASSET = TRADED[1]   # "BIL": held otherwise, incl. undefined
_SIGNAL_SOURCE = _RISK_ASSET   # BWX judges its OWN trailing return
_DEFAULT_TARGET = _DEFENSIVE_ASSET


def _signal_close(frames) -> pd.Series:
    """Validated BWX close series with a sorted, unique, tz-naive index."""
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


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last bar of each calendar month present in ``index``
    (the final bar counts as its month's last bar -- see Interpretation
    choice on "last trading day of month")."""
    periods = index.to_period("M")
    mask = np.ones(len(index), dtype=bool)
    if len(index) > 1:
        mask[:-1] = periods[1:] != periods[:-1]
    return mask


def trailing_total_return(monthly_close: pd.Series) -> np.ndarray:
    """Trailing ``lookback_months``-month total return for each month-end
    value in ``monthly_close`` (indexed by calendar-month Period): this
    month's close over the close ``lookback_months`` periods earlier, minus
    one. NaN wherever that anchor month has no entry (pure period-arithmetic
    lookup, never an interpolation to a neighbouring month)."""
    anchor_periods = monthly_close.index - _LOOKBACK_MONTHS
    anchors = monthly_close.reindex(anchor_periods).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return monthly_close.to_numpy(dtype=float) / anchors - 1.0


def decision_series(close: pd.Series) -> pd.Series:
    """Object Series indexed by month-end dates: the target decided on that
    bar ('BWX' if the trailing return is defined and strictly positive,
    'BIL' otherwise -- negative, exactly zero, or undefined all fold into
    the same BIL branch, per the Interpretation choice on zero and
    undefined returns)."""
    index = close.index
    mask = month_end_mask(index)
    month_end_dates = index[mask]
    month_end_periods = index.to_period("M")[mask]
    monthly_close = pd.Series(close.to_numpy(dtype=float)[mask],
                              index=month_end_periods)
    ret = trailing_total_return(monthly_close)
    decision = np.where(ret > 0.0, _RISK_ASSET, _DEFENSIVE_ASSET)
    return pd.Series(decision, index=month_end_dates, dtype=object)


def make_target_fn(frames: dict[str, pd.DataFrame]):
    """Rotation target factory: precompute every month-end decision once from
    the full-history BWX frame; the returned callable does one as-of lookup
    per bar on the truncated frames the engine hands it. Never returns None
    -- 'BIL' is the default whenever no decision has fired yet (see
    Interpretation choices)."""
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
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return _DEFAULT_TARGET
        today = pd.Timestamp(frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        stamp = np.datetime64(today.to_datetime64(), "ns")
        position = int(np.searchsorted(decision_dates, stamp, side="right")) - 1
        if position < 0:
            return _DEFAULT_TARGET
        return str(decision_values[position])

    return target
