"""Batch5 candidate: month_end_rebalancing_flow_rotation (rotation_two_asset).

Month-end 60/40 calendar-rebalancing flow (Harvey, Mazzoleni & Melone 2025,
NBER w33554): at the close of the sixth-to-last XNYS session of each month
(T-6) compute the month-to-date deviation of a simulated 60/40 SPY/IEF
portfolio's equity weight from 60%. Equities overweight (w_dev > 0) -> hold
the inverse ETF SH; underweight (w_dev < 0) -> hold SPY; zero or missing ->
flat. The target is returned on T-6 (engine executes at T-5 open) and held
through T-2; on T-1 the target is None so the engine exits at T's open.
Flat the rest of the month. One five-session segment per month.

Implemented exactly per the candidate's ``rule_pseudocode`` with the
candidate's ``parameters``; no tuning, no extra filters.

Interpretation choices
----------------------
1. Session calendar (which session is T-6 of a month) is a calendar fact
   known in advance, not a price. It is built ONCE in ``make_target_fn``
   from the DATES of the full-history frames handed to the factory (union
   of the SPY/SH/IEF session dates, i.e. the observed XNYS session
   calendar) and extended past the last observed bar with the vendored,
   sha-pinned XNYS calendar in ``trading_calendar`` (2026-2028), so the
   trailing partial month at the end of the data still has its true T.
   No PRICE after the decision bar is ever read: the per-bar decision uses
   only closes at prev_T and T-6 taken from the TRUNCATED frames the
   engine hands over. The observed calendar differs from the ex-ante XNYS
   calendar only on unscheduled closures; within SH's life the one such
   closure inside a month's last seven sessions is Hurricane Sandy
   (2012-10-29/30), which shifts October 2012's T-k labels by two
   sessions relative to the ex-ante schedule.
2. "Adjusted closes": the pipeline's OHLC comes from yfinance with
   ``auto_adjust=True`` (data_pipeline.fetch_ohlc), so the frames' Close IS
   the adjusted close and is used as given.
3. The hold sessions T-5..T-2 recompute the SAME decision from the same
   two closes (prev_T and T-6) rather than carrying hidden state, so a hold
   bar cannot drift from the T-6 decision and the target is a pure
   function of the truncated frames (the rotation engine's contract).
4. If either the prev_T or the T-6 close is absent from the truncated
   frame (e.g. the first month after the engine's common inception, whose
   prev_T predates the bucket) the target is None -- the pseudocode's
   "return None if either close is missing".
5. w_dev exactly 0.0 or non-finite -> None (``target_if_w_dev_zero_or_
   missing: flat``).
6. Calendar months are keyed by the session date's calendar month; a
   month with fewer than seven sessions (never the case on XNYS) simply
   has no T-6 and stays flat.

Performance: the engine calls ``target_fn`` once per bar with a growing
truncated copy. Each call does one dict lookup keyed by the decision date,
then reads at most ``TAIL_SESSIONS`` closes from the tail of two frames --
never a whole-slice computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import trading_calendar

FAMILY_ID = "month_end_rebalancing_flow_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "SH"]
SIGNAL = ["IEF"]
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Copied verbatim from the candidate's pre-registered "parameters"
# (byte-identical under json.dumps(sort_keys=True)).
PARAMETERS = {
    "equity_target_weight": 0.6,
    "bond_target_weight": 0.4,
    "equity_proxy": "SPY (traded and signal)",
    "bond_proxy": "IEF, iShares 7-10 Year Treasury Bond ETF -- the ETF nearest the paper's 10-year Treasury note futures; signal only",
    "signal": "w_dev = 0.6*(1+R_e) / (0.6*(1+R_e) + 0.4*(1+R_b)) - 0.6, where R_e and R_b are SPY and IEF adjusted-close cumulative returns from the previous month's last session close to the close of session T-6 (T = the month's last XNYS session)",
    "decision_session": "T-6",
    "hold_window": "sessions T-5 through T-1: enter at T-5 open, exit at T open (target set at T-6, cleared at T-1)",
    "target_if_w_dev_positive": "SH",
    "target_if_w_dev_negative": "SPY",
    "target_if_w_dev_zero_or_missing": "flat",
    "cash_leg": "flat (rotation engine None target)",
    "quarter_end_scaling": "none -- one uniform rule every month",
}
EXPECTED_MIN_TRADES = 200

_EQUITY = "SPY"
_INVERSE = "SH"
_BOND = "IEF"
_DECISION_OFFSET = 6          # T-6: decision session
_LAST_HOLD_OFFSET = 2         # T-2: last session on which the target is held
# A calendar month has at most 23 XNYS sessions, so prev_T is never more
# than 23 sessions behind T-2; 40 is a lookup bound with margin, not a
# rule parameter.
TAIL_SESSIONS = 40


@dataclass(frozen=True)
class SessionRole:
    """Where a session sits relative to its month's last session T."""
    month: pd.Period
    offset: int                      # 0 = T, 1 = T-1, ..., 6 = T-6, ...
    decision_date: pd.Timestamp | None   # the month's T-6 (None if < 7 sessions)
    prev_month_end: pd.Timestamp | None  # prev_T (None for the first month)


def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def build_session_calendar(frames: dict, *,
                           extra_sessions=None) -> pd.DatetimeIndex:
    """Union of every frame's session dates, extended past the last observed
    bar with the vendored XNYS calendar (or ``extra_sessions`` when given).
    Dates only -- no prices are read here."""
    calendar = pd.DatetimeIndex([])
    for frame in frames.values():
        calendar = calendar.union(_normalize_index(frame.index))
    if calendar.empty:
        raise ValueError("cannot build a session calendar from empty frames")
    last_observed = calendar.max()
    if extra_sessions is None:
        extra = pd.DatetimeIndex(
            [pd.Timestamp(s) for s in trading_calendar.load_default_sessions()])
    else:
        extra = _normalize_index(extra_sessions)
    extra = extra[extra > last_observed]
    calendar = calendar.union(extra)
    return pd.DatetimeIndex(calendar).sort_values().unique()


def session_roles(calendar) -> dict:
    """Map each session date -> SessionRole, once, from dates alone."""
    calendar = _normalize_index(calendar)
    periods = calendar.to_period("M")
    roles = {}
    prev_month_end = None
    for period in periods.unique():
        sessions = calendar[periods == period]
        count = len(sessions)
        decision_date = (sessions[count - 1 - _DECISION_OFFSET]
                         if count > _DECISION_OFFSET else None)
        for position, date in enumerate(sessions):
            roles[date] = SessionRole(
                month=period, offset=count - 1 - position,
                decision_date=decision_date, prev_month_end=prev_month_end)
        prev_month_end = sessions[-1]
    return roles


def equity_weight_deviation(equity_return: float, bond_return: float) -> float:
    """w_dev = we*(1+R_e) / (we*(1+R_e) + wb*(1+R_b)) - we."""
    we = PARAMETERS["equity_target_weight"]
    wb = PARAMETERS["bond_target_weight"]
    equity_value = we * (1.0 + equity_return)
    bond_value = wb * (1.0 + bond_return)
    return equity_value / (equity_value + bond_value) - we


def _tail_close(frame: pd.DataFrame, date: pd.Timestamp):
    """Close at ``date`` from the frame's bounded tail, or None if absent."""
    tail = frame["Close"].iloc[-TAIL_SESSIONS:]
    position = tail.index.get_indexer([date])[0]
    if position < 0:
        return None
    value = float(tail.iloc[position])
    return value if math.isfinite(value) and value > 0.0 else None


def make_target_fn(frames: dict, *, extra_sessions=None):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED + SIGNAL; only their DATES are used
    here, to fix the session calendar. Prices come from the truncated
    frames at call time."""
    for key in (_EQUITY, _INVERSE, _BOND):
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")
    calendar = build_session_calendar(frames, extra_sessions=extra_sessions)
    roles = session_roles(calendar)

    def target(truncated: dict):
        equity = truncated[_EQUITY]
        bond = truncated[_BOND]
        today = pd.Timestamp(equity.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        role = roles.get(today.normalize())
        if role is None:
            return None
        if role.offset > _DECISION_OFFSET or role.offset < _LAST_HOLD_OFFSET:
            return None                       # T-7 and earlier, T-1, T
        if role.decision_date is None or role.prev_month_end is None:
            return None
        e_base = _tail_close(equity, role.prev_month_end)
        e_now = _tail_close(equity, role.decision_date)
        b_base = _tail_close(bond, role.prev_month_end)
        b_now = _tail_close(bond, role.decision_date)
        if None in (e_base, e_now, b_base, b_now):
            return None
        w_dev = equity_weight_deviation(e_now / e_base - 1.0,
                                        b_now / b_base - 1.0)
        if not np.isfinite(w_dev) or w_dev == 0.0:
            return None
        return _INVERSE if w_dev > 0.0 else _EQUITY

    return target
