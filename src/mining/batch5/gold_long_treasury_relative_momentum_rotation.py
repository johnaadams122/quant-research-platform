"""Batch5 candidate: gold_long_treasury_relative_momentum_rotation
(rotation_two_asset).

The relative-momentum half of Antonacci's (2012) "economic stress" dual
momentum module, always invested: on the first trading session of each
calendar month compare the trailing twelve-calendar-month return of GLD and
TLT and hold the leader until the next month's decision. GLD if its return
is STRICTLY greater, otherwise TLT (a tie goes to the defensive leg). No
T-bill / absolute-momentum hurdle, no skip-month, no cash state.

Implemented exactly per the candidate's ``rule_pseudocode`` with the
candidate's ``parameters``; no tuning, no extra filters. The rule is, by
the candidate's own citation, ``signals.make_monthly_relative_momentum_
target_fn(risk_ticker="GLD", defensive_ticker="TLT", lookback_months=12)``
-- this module reproduces that factory's decisions bar-for-bar (pinned by
a test that compares the two on every bar of a synthetic history) while
reading only a bounded tail of the truncated frames per call.

Rule, per decision bar t (the engine hands frames truncated through t):
  decision_date d = the first session of t's calendar month
                    (``signals._monthly_decision_frame`` convention)
  anchor          = d minus 12 calendar months (``pd.DateOffset(months=12)``)
  base_X          = Close_X on the last session on or before anchor
                    (``signals._calendar_anchor_close``); if none exists or
                    it is <= 0 for either leg -> None (12-month warm-up only)
  r_X             = (Close_X[d] - base_X) / base_X
  target          = "GLD" if r_GLD > r_TLT else "TLT"
The target is returned on every bar of the month (not just the decision
bar) so the engine holds the position; it executes at t+1 open, unchanged
targets do not trade.

Interpretation choices
----------------------
1. Arithmetic form: the trailing return is computed as ``(close - base) /
   base`` -- the reference factory's exact expression -- rather than the
   pseudocode's algebraically equal ``close / base - 1``; the two can differ
   in the last ulp, which matters only for the strict-'>' tie rule, and the
   candidate names the reference factory as the exact rule.
2. Closes are the pipeline's auto-adjusted yfinance closes (data_pipeline.
   fetch_ohlc, ``auto_adjust=True``), used as given -- "total return" in
   the pseudocode's DATA line.
3. The full-history ``frames`` handed to the factory are used only to
   confirm both legs are present. Every decision is a pure function of the
   truncated frames the rotation engine passes at call time (its documented
   contract), so a holdout bucket can never see a price after its last bar
   and the bucket's own 600-day lookback buffer supplies the warm-up.
4. Per call the module reads ``TAIL_SESSIONS`` (320) trailing bars. Twelve
   calendar months span at most ~262 NYSE sessions and the current month at
   most 23 more, so on any real trading calendar the anchor close lies
   inside the tail. If it does not (a denser-than-daily-session calendar),
   the lookup falls back to the whole truncated frame, so the result is
   IDENTICAL to the reference in every case; only the cost differs.
   TAIL_SESSIONS is a lookup bound with margin, not a rule parameter.

Performance: the engine calls ``target_fn`` once per bar with a growing
truncated copy; each call does two ``iloc`` tail slices, a same-month mask
over <= 320 dates and two anchor lookups -- never a whole-slice computation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FAMILY_ID = "gold_long_treasury_relative_momentum_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["GLD", "TLT"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "GLD"
# Copied verbatim from the curated candidate record's "parameters"
# (byte-identical under json.dumps(sort_keys=True)).
PARAMETERS = {
    "lookback_months": 12,
    "rebalance": "monthly",
    "signal": "relative_trailing_return",
    "num_assets": 2,
    "hold_top_n": 1,
    "cash_fallback": False,
    "risk_ticker": "GLD",
    "defensive_ticker": "TLT",
    "tie_rule": "defensive",
    "skip_month": False,
}
EXPECTED_MIN_TRADES = 30

_RISK = PARAMETERS["risk_ticker"]
_DEFENSIVE = PARAMETERS["defensive_ticker"]
_LOOKBACK_MONTHS = PARAMETERS["lookback_months"]
# Twelve calendar months <= ~262 NYSE sessions, plus <= 23 sessions of the
# current month after its first session; 320 is a lookup bound with margin.
TAIL_SESSIONS = 320


def _naive(timestamp) -> pd.Timestamp:
    out = pd.Timestamp(timestamp)
    return out.tz_localize(None) if out.tzinfo is not None else out


def _decision_position(index: pd.DatetimeIndex) -> int:
    """Position (within ``index``) of the first session of the last bar's
    calendar month. Sessions of one month form a contiguous suffix of a
    sorted calendar, so the first True of the same-month mask is it."""
    last = index[-1]
    same_month = (index.year == last.year) & (index.month == last.month)
    return int(np.argmax(same_month))


def _anchor_close(frame: pd.DataFrame, tail: pd.DataFrame,
                  anchor: pd.Timestamp):
    """Close on the last session on or before ``anchor`` -- from the tail
    when it reaches back that far, else from the whole truncated frame
    (exactly ``signals._calendar_anchor_close``). None if no such session."""
    if len(tail) < len(frame) and _naive(tail.index[0]) > anchor:
        source = frame                      # tail too short for this calendar
    else:
        source = tail
    eligible = source["Close"].to_numpy()[
        np.asarray(source.index <= anchor)]
    if eligible.size == 0:
        return None
    return float(eligible[-1])


def _trailing_return(frame: pd.DataFrame, decision_position_from_end: int,
                     anchor: pd.Timestamp):
    """(Close[d] - base) / base with base = the anchor close, or None."""
    tail = frame.iloc[-TAIL_SESSIONS:]
    base = _anchor_close(frame, tail, anchor)
    if base is None or not math.isfinite(base) or base <= 0.0:
        return None
    current = float(tail["Close"].iloc[-decision_position_from_end])
    return (current - base) / base


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED; only their presence is checked.
    Prices come from the truncated frames at call time."""
    for key in (_RISK, _DEFENSIVE):
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")

    def target(truncated: dict):
        risk = truncated[_RISK]
        defensive = truncated[_DEFENSIVE]
        risk_tail_index = risk.index[-TAIL_SESSIONS:]
        if risk_tail_index.tz is not None:
            risk_tail_index = risk_tail_index.tz_localize(None)
        decision_pos = _decision_position(risk_tail_index)
        decision_date = risk_tail_index[decision_pos]
        from_end = len(risk_tail_index) - decision_pos
        anchor = decision_date - pd.DateOffset(months=_LOOKBACK_MONTHS)
        risk_return = _trailing_return(risk, from_end, anchor)
        if risk_return is None:
            return None
        defensive_return = _trailing_return(defensive, from_end, anchor)
        if defensive_return is None:
            return None
        return _RISK if risk_return > defensive_return else _DEFENSIVE

    return target
