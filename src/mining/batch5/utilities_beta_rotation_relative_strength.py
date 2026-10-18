"""Batch5 candidate: utilities beta rotation on 4-week XLU/SPY relative
strength, weekly (family ``utilities_beta_rotation_relative_strength``,
shape ``rotation_two_asset``; Gayed & Bilello 2014, Dow Award).

Rule (verbatim from the curated candidate's ``rule_pseudocode``):

    DATA: daily auto-adjusted OHLC for XLU and SPY from yfinance, restricted
    to their common calendar (starts 1998-12-22). No other series.
    DECISION BARS: the last trading session of each ISO week (normally
    Friday; the preceding session if Friday is a holiday). On every other
    session, the target is the most recent decision's target.
    FOR each decision bar t (using only closes <= t):
      ratio_t   = Close_XLU[t] / Close_SPY[t]
      ratio_ref = ratio on the 4th prior weekly decision bar
      IF ratio_ref is unavailable (fewer than 4 prior decision bars):
          target = None (warm-up only)
      ELSE IF ratio_t > ratio_ref: target = 'XLU'
      ELSE:                        target = 'SPY'   # ties and negatives
    EXECUTION (rotation_engine.run_two_asset_rotation convention): the
    target decided on bar t executes at bar t+1's open. An unchanged target
    does not trade; a changed target closes the old leg and opens the new
    leg at that open. After warm-up the strategy is never flat.
    PARAMETERS ARE FIXED: 4-week lookback, weekly decision, no threshold
    band, no cash state, no MA or absolute-momentum filter.

Engine mapping (mining.rotation_engine.run_two_asset_rotation): on every
decision bar t the engine hands ``target_fn`` the frames truncated through
t and executes the returned asset at t+1's open. The weekly decision is
therefore returned ON the week's last session (fill = first session of the
next week) and every other bar returns the standing decision so the engine
holds without churn. Everything is precomputed ONCE in the factory from the
full-history frames using only same-bar and earlier closes; the per-bar
closure is one binary search plus a consistency read of the slice's last
XLU and SPY closes -- it never recomputes over the slice.

Interpretation choices
----------------------
1. "Last trading session of each ISO week" is judged on the SHARED traded
   calendar (intersection of the XLU and SPY bar calendars, which is exactly
   the calendar the rotation engine runs on after its common-inception
   trim): a session is a decision bar when the NEXT shared session falls in
   a later ISO week (``isocalendar`` year+week, so the 30/31 December bars
   that ISO assigns to week 1 of the next year are handled correctly). The
   exchange schedule is public in advance, so knowing on bar t that it is
   the week's final session is causal; only calendar DATES are used this
   way, never a future price. The final session of the data is never a
   decision date (there is no next session to fill at), which is also why a
   frame truncated mid-week yields the identical decisions through its last
   bar. The repo's vendored NYSE calendar covers 2026-2028 only, so it
   cannot serve a 1998+ history.
2. "4th prior weekly decision bar" is counted in DECISION BARS on that
   shared calendar (ordinal j compares with ordinal j-4), not in calendar
   days, so a holiday-shortened week still counts as one week.
3. Warm-up: the first four decision bars have no reference and yield
   ``None``; no earlier standing target exists, so the closure returns
   ``None`` (flat -- the only state the engine can start in) until the
   fifth decision bar has been reached. Every later decision is decidable
   (all closes are finite and positive), so the standing target is never
   ``None`` again.
4. Tie rule: ``ratio_t > ratio_ref`` strictly; equality goes to SPY, per
   the paper and the candidate's ``tie_rule = broad_market``.
5. TRADED order follows the candidate's own ``tickers`` list, [XLU, SPY]:
   XLU is the signal-on leg (the paper "positions into Utilities" when the
   4-week relative strength is positive) and SPY the default leg. Both
   legs are 100% equity, so "risk/defensive" is a labelling convention
   here, not a cash state.
6. "Auto-adjusted close" is the pipeline's Close as delivered
   (``data_pipeline.fetch_ohlc`` fetches ``auto_adjust=True``).
7. Fail-closed guard: the closure compares the slice's last XLU and SPY
   closes with the factory's own closes for that date and raises if they
   differ (or if the date is unknown), so a factory built on one dataset
   can never silently drive an engine run on another.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FAMILY_ID = "utilities_beta_rotation_relative_strength"
SHAPE = "rotation_two_asset"
TRADED = ["XLU", "SPY"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Copied from the curated candidate record ("parameters"); the driver rejects any drift under
# json.dumps(sort_keys=True).
PARAMETERS = {
    "ratio": "XLU/SPY",
    "lookback_weeks": 4,
    "rebalance": "weekly",
    "decision_bar": "last_session_of_iso_week",
    "signal": "ratio_rate_of_change_sign_over_prior_4_weekly_decisions",
    "num_assets": 2,
    "hold_top_n": 1,
    "cash_fallback": False,
    "tie_rule": "broad_market",
}
EXPECTED_MIN_TRADES = 300

UTILITIES = TRADED[0]           # numerator of the ratio, the signal-on leg
BROAD_MARKET = TRADED[1]        # denominator, the default / tie leg
LOOKBACK_WEEKS = int(PARAMETERS["lookback_weeks"])
_CLOSE_REL_TOL = 1e-9


def _tz_naive(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out


def _epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.to_numpy(dtype="datetime64[ns]").view("int64"))


def _validated_close(frame: pd.DataFrame, key: str) -> pd.Series:
    """Finite, positive Close on a sorted, unique, tz-naive calendar."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{key}: frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ValueError(f"{key}: frame is missing Close")
    index = _tz_naive(frame.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{key}: frame index must be sorted and unique")
    values = pd.to_numeric(pd.Series(frame["Close"]).to_numpy(), errors="coerce")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{key}: Close values must be finite and positive")
    return pd.Series(values, index=index, name="Close")


def weekly_decision_sessions(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Decision dates: every session whose NEXT session lies in a later ISO
    week (year, week). The final session is never a decision date."""
    index = _tz_naive(calendar)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("shared calendar must be sorted and unique")
    if len(index) < 2:
        return index[:0]
    iso = index.isocalendar()
    codes = (iso["year"].to_numpy(dtype="int64") * 100
             + iso["week"].to_numpy(dtype="int64"))
    is_decision = np.zeros(len(codes), dtype=bool)
    is_decision[:-1] = codes[:-1] != codes[1:]
    return index[is_decision]


def decision_series(utilities_close: pd.Series, market_close: pd.Series,
                    shared_calendar: pd.DatetimeIndex, *,
                    lookback_weeks: int = LOOKBACK_WEEKS,
                    utilities: str = UTILITIES,
                    market: str = BROAD_MARKET) -> pd.Series:
    """Target decided AT each weekly decision session (object Series
    indexed by decision date; values ``utilities`` / ``market`` / ``None``).

    ratio_t is the XLU/SPY close ratio at the decision bar; ratio_ref the
    ratio at the decision bar ``lookback_weeks`` decisions earlier. Strict
    ``>`` selects utilities; the first ``lookback_weeks`` decisions are
    ``None`` (warm-up).
    """
    if lookback_weeks <= 0:
        raise ValueError("lookback_weeks must be positive")
    dates = weekly_decision_sessions(shared_calendar)
    if len(dates) == 0:
        return pd.Series([], index=dates, dtype=object, name="target")
    missing = dates.difference(utilities_close.index).union(
        dates.difference(market_close.index))
    if len(missing):
        raise ValueError(
            f"shared decision session {missing[0].date()} is not a session "
            "of both legs")
    ratio = (utilities_close.reindex(dates).to_numpy(dtype=float)
             / market_close.reindex(dates).to_numpy(dtype=float))
    out = np.empty(len(dates), dtype=object)
    out[:] = None
    if len(dates) > lookback_weeks:
        current = ratio[lookback_weeks:]
        reference = ratio[:-lookback_weeks]
        out[lookback_weeks:] = np.where(current > reference, utilities, market)
    return pd.Series(out, index=dates, dtype=object, name="target")


def make_target_fn(frames):
    """Driver factory: ``frames`` = {symbol: full-history OHLC} for TRADED
    (+ SIGNAL, none here). Returns ``target(truncated) -> asset | None``
    for mining.rotation_engine.run_two_asset_rotation."""
    if not isinstance(frames, dict):
        raise TypeError("frames must be a dict of OHLC DataFrames")
    for key in (UTILITIES, BROAD_MARKET):
        if key not in frames:
            raise ValueError(f"missing required frame for {key}")
    utilities_close = _validated_close(frames[UTILITIES], UTILITIES)
    market_close = _validated_close(frames[BROAD_MARKET], BROAD_MARKET)
    shared = utilities_close.index.intersection(market_close.index).sort_values()
    if len(shared) < 2:
        raise ValueError("traded legs share fewer than two sessions")

    decisions = decision_series(utilities_close, market_close, shared)
    decision_ns = _epoch_ns(decisions.index)
    targets = decisions.to_numpy(dtype=object)
    closes_by_date = {
        UTILITIES: {int(ts): float(v) for ts, v in
                    zip(_epoch_ns(utilities_close.index),
                        utilities_close.to_numpy(dtype=float))},
        BROAD_MARKET: {int(ts): float(v) for ts, v in
                       zip(_epoch_ns(market_close.index),
                           market_close.to_numpy(dtype=float))},
    }

    def target(truncated):
        frame = truncated[UTILITIES]            # KeyError if absent: fail closed
        today = pd.Timestamp(frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        key_ns = int(today.value)
        for symbol in (UTILITIES, BROAD_MARKET):
            expected = closes_by_date[symbol].get(key_ns)
            actual = float(truncated[symbol]["Close"].iloc[-1])
            if expected is None or not math.isclose(expected, actual,
                                                    rel_tol=_CLOSE_REL_TOL):
                raise ValueError(
                    f"{FAMILY_ID}: {symbol} close on {today.date()} "
                    f"({actual!r}) does not match the factory's data "
                    f"({expected!r}); the factory and the engine must run on "
                    "the same frames")
        pos = int(np.searchsorted(decision_ns, key_ns, side="right")) - 1
        if pos < 0:
            return None
        return targets[pos]

    target.family_id = FAMILY_ID
    target.decision_dates = decisions.index
    return target
