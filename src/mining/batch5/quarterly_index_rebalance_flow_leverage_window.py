"""Batch5 candidate: quarterly_index_rebalance_flow_leverage_window
(rotation_two_asset).

Quarter-end calendar-rebalancing flow, expressed through a leveraged pair
rather than the 1x/inverse pair the sibling family
``month_end_rebalancing_flow_rotation`` uses. Same underlying economic
mechanism as that family's own source -- Harvey, Mazzoleni & Melone (2025,
NBER w33554), "The Unintended Consequences of Rebalancing": calendar-based
rebalancers must sell equities into a scheduled review point when stocks
have outperformed bonds since the last reset (equities overweight) and buy
when they have underperformed, and this creates predictable price pressure
concentrated in the final sessions before the reset. That paper's own
empirical measurement is at MONTHLY frequency; this family generalizes the
same signal construction to the QUARTER-END reset point instead --
disclosed as an extrapolation, not a direct finding of the cited paper (see
REFORMULATION NOTE below and the catalog's own ``source`` field).

At the close of the sixth-to-last XNYS session of each CALENDAR QUARTER
(Q-6) compute the quarter-to-date deviation of a simulated 60/40 SPY/IEF
portfolio's equity weight from 60%, using the SAME formula as the sibling
family but resetting each calendar quarter (Jan/Apr/Jul/Oct) rather than
each calendar month. Equities overweight (w_dev > 0, quarter-end
rebalancers must sell) -> hold the 3x-short S&P 500 fund SPXS; underweight
(w_dev < 0, rebalancers must buy) -> hold the 3x-long S&P 500 fund SPXL;
zero or missing -> flat. The target is returned on Q-6 (engine executes at
Q-5 open) and held through Q-2; on Q-1 the target is None so the engine
exits at Q's open. Flat the rest of the quarter -- four decision windows a
year, not twelve.

REFORMULATION NOTE (why a leveraged pair, not SPY/SH). A quarter-to-date
deviation accumulates over roughly three times as many sessions as the
sibling's month-to-date deviation, so it is expected to reach a larger
typical magnitude at the Q-6 decision point -- but the position is held
only four times a year instead of twelve, a lower-frequency, larger-
per-trade expression of the same clientele-flow logic. SPXL/SPXS (Direxion
Daily S&P 500 Bull/Bear 3X Shares, real exchange-traded products, inception
2008-11-05/2008-11-19) are the leveraged long/short pair over the SAME
underlying (S&P 500) the sibling's SPY/SH pair tracks, so the sign
convention and traded index are identical -- only the leverage and the
reset frequency differ. This is the "leverage window" in the family name: a
narrow multi-session trade expressed at 3x leverage around the quarterly
reset instead of a 1x/inverse pair around the monthly one.

Quarter-end is a real, recognized institutional calendar-rebalancing
review point distinct from month-end (pension funds, endowments and other
mandates commonly review and rebalance on a quarterly cadence rather than
monthly -- a separate clientele from the sibling's monthly rebalancers, not
the same effect resampled), but the specific 17-bps/day magnitude the
source paper measures is a MONTHLY-frequency finding; no equivalent
quarterly magnitude is claimed or available from that source, and none is
invented here.

Interpretation choices (mirrors the sibling's, substituting quarter for
month)
----------------------
1. Session calendar (which session is Q-6 of a quarter) is a calendar fact
   known in advance, not a price. It is built ONCE in ``make_target_fn``
   from the DATES of the full-history frames handed to the factory (union
   of the SPXL/SPXS/IEF session dates) and extended past the last observed
   bar with the vendored, sha-pinned XNYS calendar in ``trading_calendar``
   (2026-2028), so the trailing partial quarter at the end of the data
   still has its true Q. No PRICE after the decision bar is ever read: the
   per-bar decision uses only closes at prev_Q and Q-6 taken from the
   TRUNCATED frames the engine hands over.
2. "Adjusted closes": the pipeline's OHLC comes from yfinance with
   ``auto_adjust=True``, so the frames' Close IS the adjusted close and is
   used as given.
3. The hold sessions Q-5..Q-2 recompute the SAME decision from the same
   two closes (prev_Q and Q-6) rather than carrying hidden state, so a hold
   bar cannot drift from the Q-6 decision and the target is a pure function
   of the truncated frames (the rotation engine's contract).
4. If either the prev_Q or the Q-6 close is absent from the truncated frame
   (e.g. the first quarter after the engine's common inception, whose
   prev_Q predates the bucket) the target is None.
5. w_dev exactly 0.0 or non-finite -> None.
6. Calendar quarters are keyed by the session date's calendar quarter
   (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec); a quarter with fewer than seven
   sessions (never the case on XNYS) simply has no Q-6 and stays flat.

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

FAMILY_ID = "quarterly_index_rebalance_flow_leverage_window"
SHAPE = "rotation_two_asset"
TRADED = ["SPXL", "SPXS"]
# IEF is the bond-weight signal input (as in the sibling); SPY is the
# UNLEVERAGED equity-weight signal input -- see signal_price_source in
# PARAMETERS for why the signal is computed on SPY rather than on SPXL's
# own daily-reset 3x closes.
SIGNAL = ["IEF", "SPY"]
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Must equal the catalog entry's "parameters" byte-for-byte under
# json.dumps(sort_keys=True) -- see test_parameters_are_byte_identical_to_
# the_catalog_entry.
PARAMETERS = {
    "equity_target_weight": 0.6,
    "bond_target_weight": 0.4,
    "equity_proxy": "SPXL (traded, 3x S&P 500 long -- ProShares/Direxion leveraged expression of the SPY signal)",
    "bond_proxy": "IEF, iShares 7-10 Year Treasury Bond ETF -- signal only, unleveraged",
    "signal": "w_dev = 0.6*(1+R_e) / (0.6*(1+R_e) + 0.4*(1+R_b)) - 0.6, where R_e and R_b are the UNLEVERAGED S&P 500 (SPY-equivalent, read from SPXL's own 3x daily-reset series is NOT used for the signal -- see signal_price_source) and IEF adjusted-close cumulative returns from the previous quarter's last session close to the close of session Q-6 (Q = the quarter's last XNYS session)",
    "signal_price_source": "SPY (unleveraged) closes are used to compute R_e -- SPXL/SPXS are daily-reset 3x products whose multi-session cumulative return is NOT 3x the underlying's cumulative return (volatility drag), so using SPXL's own closes for a quarter-long deviation would mix the rebalancing signal with leverage decay; the signal is computed on the unleveraged index and only the EXECUTED position is leveraged",
    "decision_session": "Q-6",
    "hold_window": "sessions Q-5 through Q-1: enter at Q-5 open, exit at Q open (target set at Q-6, cleared at Q-1)",
    "reset_frequency": "calendar quarter (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec), not calendar month",
    "target_if_w_dev_positive": "SPXS",
    "target_if_w_dev_negative": "SPXL",
    "target_if_w_dev_zero_or_missing": "flat",
    "cash_leg": "flat (rotation engine None target)",
    "quarter_end_scaling": "none -- one uniform rule every quarter",
}
EXPECTED_MIN_TRADES = 60

_EQUITY_SIGNAL = "SPY"
_BOND = "IEF"
_LONG = "SPXL"
_SHORT = "SPXS"
_DECISION_OFFSET = 6          # Q-6: decision session
_LAST_HOLD_OFFSET = 2         # Q-2: last session on which the target is held
# A calendar quarter has at most ~64 XNYS sessions; 80 is a lookup bound
# with margin, not a rule parameter.
TAIL_SESSIONS = 80


@dataclass(frozen=True)
class SessionRole:
    """Where a session sits relative to its quarter's last session Q."""
    quarter: pd.Period
    offset: int                      # 0 = Q, 1 = Q-1, ..., 6 = Q-6, ...
    decision_date: pd.Timestamp | None   # the quarter's Q-6 (None if < 7 sessions)
    prev_quarter_end: pd.Timestamp | None  # prev_Q (None for the first quarter)


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
    periods = calendar.to_period("Q")
    roles = {}
    prev_quarter_end = None
    for period in periods.unique():
        sessions = calendar[periods == period]
        count = len(sessions)
        decision_date = (sessions[count - 1 - _DECISION_OFFSET]
                         if count > _DECISION_OFFSET else None)
        for position, date in enumerate(sessions):
            roles[date] = SessionRole(
                quarter=period, offset=count - 1 - position,
                decision_date=decision_date, prev_quarter_end=prev_quarter_end)
        prev_quarter_end = sessions[-1]
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
    full-history OHLC frames for TRADED + SIGNAL + the unleveraged equity
    signal series; only their DATES are used here, to fix the session
    calendar. Prices come from the truncated frames at call time. The
    unleveraged equity signal series (SPY) must be supplied under
    ``_EQUITY_SIGNAL`` even though SPY is not itself traded by this
    family -- see ``signal_price_source`` in PARAMETERS."""
    for key in (_LONG, _SHORT, _BOND, _EQUITY_SIGNAL):
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")
    calendar = build_session_calendar(frames, extra_sessions=extra_sessions)
    roles = session_roles(calendar)

    def target(truncated: dict):
        equity = truncated[_EQUITY_SIGNAL]
        bond = truncated[_BOND]
        today = pd.Timestamp(equity.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        role = roles.get(today.normalize())
        if role is None:
            return None
        if role.offset > _DECISION_OFFSET or role.offset < _LAST_HOLD_OFFSET:
            return None                       # Q-7 and earlier, Q-1, Q
        if role.decision_date is None or role.prev_quarter_end is None:
            return None
        e_base = _tail_close(equity, role.prev_quarter_end)
        e_now = _tail_close(equity, role.decision_date)
        b_base = _tail_close(bond, role.prev_quarter_end)
        b_now = _tail_close(bond, role.decision_date)
        if None in (e_base, e_now, b_base, b_now):
            return None
        w_dev = equity_weight_deviation(e_now / e_base - 1.0,
                                        b_now / b_base - 1.0)
        if not np.isfinite(w_dev) or w_dev == 0.0:
            return None
        return _SHORT if w_dev > 0.0 else _LONG

    return target
