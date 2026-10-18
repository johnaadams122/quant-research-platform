"""Batch5 candidate: small-cap time-series momentum, long IWM or long RWM
(inverse) rotation (family ``small_cap_tsmom_long_inverse_rotation``, shape
``rotation_two_asset``).

Rule (verbatim from the curated candidate's ``rule_pseudocode``):

    ASSETS = {IWM, RWM}; shared calendar from RWM's first session
    C = auto-adjusted daily close of IWM; M[k] = last IWM close in
    calendar month k
    at the close of the last trading day of each calendar month m
    (needs 12 prior IWM month-end closes):
        s1  = +1 if M[m]/M[m-1]  - 1 > 0 else -1
        s3  = +1 if M[m]/M[m-3]  - 1 > 0 else -1
        s12 = +1 if M[m]/M[m-12] - 1 > 0 else -1
        composite = (s1 + s3 + s12) / 3
        target = IWM if composite > 0 else RWM
    if target != current holding: the rotation engine closes the current
    holding and opens the target at the next session (engine convention)
    if target == current holding: hold; no stops, no cash state

Engine mapping (mining.rotation_engine.run_two_asset_rotation): on every
decision bar t the engine hands ``target_fn`` the frames truncated through
t and executes the returned asset at t+1's open. The month-end decision is
therefore returned ON the month-end bar (fill = first session of the next
month), and every other bar returns the standing decision so the engine
holds without churn. Everything is precomputed ONCE in the factory from
grouped / as-of operations; the per-bar closure is one binary search plus
one consistency read of the slice's last close -- it never recomputes over
the slice.

Interpretation choices
----------------------
1. "Last trading day of each calendar month" is taken from the SHARED
   traded calendar (intersection of the IWM and RWM bar calendars, which is
   exactly the calendar the rotation engine runs on): a session is a
   month-end when the next shared session falls in a later calendar month.
   The exchange schedule is public in advance, so knowing on bar t that it
   is the month's final session is causal; only calendar DATES are used
   this way, never a future price. The final session of the data is never
   a decision date (there is no next session to fill at), which is also
   why a frame truncated mid-month yields the identical decisions through
   its last bar. The repo's vendored NYSE calendar covers 2026-2028 only,
   so it cannot serve a 2007+ history.
2. M[m] (the current month-end close) is IWM's close AT the shared
   month-end session itself. The anchors M[m-1], M[m-3], M[m-12] are IWM's
   last close in those calendar months on IWM's OWN bar calendar, so the
   months BEFORE RWM's inception feed the lookbacks exactly as the
   candidate states ("in-sample starts at RWM's first month-end,
   2007-01-31" -- IWM's 2006 closes supply the anchors). Every anchor is
   dated strictly inside an earlier calendar month, hence before the
   decision bar.
3. "Needs 12 prior IWM month-end closes" is read as period arithmetic: the
   calendar months exactly 1, 3 and 12 before m must each have an IWM
   month-end close. No interpolation across a missing month. A month-end
   with no decision leaves the standing target UNCHANGED (hold); before
   the first decidable month-end there is no standing target and the
   factory returns None (flat), the only state the engine can start in.
4. A trailing return of exactly zero votes -1 (the pseudocode's strict
   ``> 0``). The composite is never zero with three +/-1 votes, so
   ``target = IWM iff at least two of the three votes are +1``.
5. "Auto-adjusted close" is the pipeline's Close as delivered
   (``data_pipeline.fetch_ohlc`` fetches ``auto_adjust=True``).
6. Fail-closed guard: the closure compares the slice's last IWM close with
   the factory's own close for that date and raises if they differ, so a
   factory built on one dataset can never silently drive an engine run on
   another.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FAMILY_ID = "small_cap_tsmom_long_inverse_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["IWM", "RWM"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "IWM"
# Copied from the curated candidate entry's "parameters"; the driver rejects any drift under
# json.dumps(sort_keys=True).
PARAMETERS = {
    "signal_source": "IWM_own_trailing_returns",
    "lookback_months": [
        1,
        3,
        12,
    ],
    "combination": "majority_sign_of_three_lookbacks (at least 2 of 3 positive -> IWM, else RWM)",
    "decision_frequency": "monthly_at_month_end_close",
    "risk_asset": "IWM",
    "risk_off_asset": "RWM",
    "cash_proxy": "none",
    "switch_execution": "next_session (rotation engine convention)",
}
EXPECTED_MIN_TRADES = 60

RISK_ASSET = str(PARAMETERS["risk_asset"])
DEFENSIVE_ASSET = str(PARAMETERS["risk_off_asset"])
LOOKBACK_MONTHS = tuple(int(k) for k in PARAMETERS["lookback_months"])
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


def month_end_closes(close: pd.Series) -> pd.Series:
    """M[k]: the last close of each calendar month on the series' OWN
    calendar, indexed by monthly Period. The series' final (possibly
    partial) month is included; it only ever serves as an ANCHOR for a
    decision in a LATER month, never for a decision at or before its own
    date."""
    index = _tz_naive(close.index)
    periods = index.to_period("M")
    codes = periods.asi8
    is_last = np.empty(len(codes), dtype=bool)
    if len(codes):
        is_last[:-1] = codes[:-1] != codes[1:]
        is_last[-1] = True
    values = np.asarray(close.to_numpy(), dtype=float)
    return pd.Series(values[is_last], index=periods[is_last], name="month_end_close")


def shared_month_end_sessions(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Decision dates: every session whose NEXT session lies in a later
    calendar month. The final session is never a decision date."""
    index = _tz_naive(calendar)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("shared calendar must be sorted and unique")
    if len(index) < 2:
        return index[:0]
    codes = index.to_period("M").asi8
    is_decision = np.zeros(len(codes), dtype=bool)
    is_decision[:-1] = codes[:-1] != codes[1:]
    return index[is_decision]


def composite_target(current: float, anchors, risk: str, defensive: str):
    """Majority sign of the trailing returns; ``None`` when any anchor is
    missing. A zero return votes -1 (strict ``> 0``)."""
    votes = 0
    for anchor in anchors:
        if anchor is None or not np.isfinite(anchor) or anchor <= 0:
            return None
        votes += 1 if (current / anchor - 1.0) > 0.0 else -1
    return risk if votes > 0 else defensive


def decision_series(risk_close: pd.Series, shared_calendar: pd.DatetimeIndex,
                    *, lookback_months=LOOKBACK_MONTHS, risk: str = RISK_ASSET,
                    defensive: str = DEFENSIVE_ASSET) -> pd.Series:
    """Standing target as of each shared month-end session (object Series
    indexed by decision date; values ``risk`` / ``defensive`` / ``None``).

    The raw decision at month-end m uses M[m] = the risk close AT that
    session and anchors M[m-k] from ``month_end_closes`` (period
    arithmetic). An undecidable month-end carries the previous standing
    target forward; with no prior decision it stays ``None``.
    """
    monthly = month_end_closes(risk_close)
    by_period = {period: float(value) for period, value in monthly.items()}
    close_by_date = risk_close
    dates = shared_month_end_sessions(shared_calendar)
    standing = None
    out = []
    for date in dates:
        if date not in close_by_date.index:
            raise ValueError(
                f"shared month-end {date.date()} is not a {risk} session")
        current = float(close_by_date.loc[date])
        period = date.to_period("M")
        anchors = [by_period.get(period - k) for k in lookback_months]
        decision = composite_target(current, anchors, risk, defensive)
        if decision is not None:
            standing = decision
        out.append(standing)
    return pd.Series(out, index=dates, dtype=object, name="target")


def make_target_fn(frames):
    """Driver factory: ``frames`` = {symbol: full-history OHLC} for TRADED
    (+ SIGNAL, none here). Returns ``target(truncated) -> asset | None``
    for mining.rotation_engine.run_two_asset_rotation."""
    if not isinstance(frames, dict):
        raise TypeError("frames must be a dict of OHLC DataFrames")
    for key in (RISK_ASSET, DEFENSIVE_ASSET):
        if key not in frames:
            raise ValueError(f"missing required frame for {key}")
    risk_close = _validated_close(frames[RISK_ASSET], RISK_ASSET)
    defensive_close = _validated_close(frames[DEFENSIVE_ASSET], DEFENSIVE_ASSET)
    shared = risk_close.index.intersection(defensive_close.index).sort_values()
    if len(shared) < 2:
        raise ValueError("traded legs share fewer than two sessions")

    decisions = decision_series(risk_close, shared)
    decision_ns = _epoch_ns(decisions.index)
    targets = decisions.to_numpy(dtype=object)
    close_by_date = {int(ts): float(v) for ts, v in
                     zip(_epoch_ns(risk_close.index),
                         risk_close.to_numpy(dtype=float))}

    def target(truncated):
        frame = truncated[RISK_ASSET]           # KeyError if absent: fail closed
        today = pd.Timestamp(frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        expected = close_by_date.get(int(today.value))
        actual = float(frame["Close"].iloc[-1])
        if expected is None or not math.isclose(expected, actual,
                                                rel_tol=_CLOSE_REL_TOL):
            raise ValueError(
                f"{FAMILY_ID}: {RISK_ASSET} close on {today.date()} "
                f"({actual!r}) does not match the factory's data "
                f"({expected!r}); the factory and the engine must run on the "
                "same frames")
        pos = int(np.searchsorted(decision_ns, today.value, side="right")) - 1
        if pos < 0:
            return None
        return targets[pos]

    target.family_id = FAMILY_ID
    target.decision_dates = decisions.index
    return target
