"""Batch5 candidate: low_vol_high_beta_factor_momentum_rotation (rotation_two_asset).

Time-series factor momentum on the low-beta factor (Gupta & Kelly 2019;
Ehsani & Linnainmaa 2022), expressed long-only as a two-leg rotation
between SPHB (100 highest-beta S&P 500 stocks, the risk leg) and SPLV
(100 lowest-volatility S&P 500 stocks, the defensive leg).

Rule (exactly the candidate's ``rule_pseudocode``):
  * DECISION BAR: the first trading session of each calendar month
    (``mining.signals._monthly_decision_frame`` convention), using data
    through that session's close; the target holds until the next month's
    decision.
  * anchor = decision_date minus 1 calendar month; base_X = Close_X on the
    last session on or before the anchor (``_calendar_anchor_close`` with
    lookback_months=1). Either base unavailable or <= 0 -> None (warm-up).
  * r_X = Close_X[decision] / base_X - 1 for X in {SPHB, SPLV};
    target = SPHB if r_SPHB > r_SPLV else SPLV (strict '>': a tie goes to
    the defensive leg, ``tie_rule: defensive``).
  * Execution is the rotation engine's: a target decided on bar t fills at
    bar t+1's open; unchanged targets do not trade; fully invested after
    warm-up, no cash state, no absolute-momentum filter, no skip-month.

This is by construction the same rule as
``signals.make_monthly_relative_momentum_target_fn(risk_ticker="SPHB",
defensive_ticker="SPLV", lookback_months=1)`` (the pseudocode names it),
re-implemented here ONLY so each per-bar call reads a bounded tail of the
truncated frames instead of re-scanning the whole growing slice (the
reference computes ``to_period("M")`` over the full slice on every call,
which is O(n^2) over a run). The test file asserts bar-for-bar agreement
with the reference factory on a random tape.

Interpretation choices
----------------------
1. TRADED follows the batch5 contract order [risk, defensive] =
   ["SPHB", "SPLV"] (the candidate's ``tickers`` list the same two symbols
   in the other order; the driver only requires exactly two).
2. "Adjusted closes": the pipeline's OHLC comes from yfinance with
   ``auto_adjust=True`` (data_pipeline.fetch_ohlc), so the frames' Close IS
   the total-return-adjusted close and is used as given.
3. The decision date is the first session of the CURRENT month present in
   the truncated frame (so the first bar of a bucket that starts mid-month
   is that month's decision bar -- identical to the reference convention).
   Every later session of the month recomputes the SAME decision from the
   same two closes rather than carrying hidden state, so the target is a
   pure function of the truncated frames (rotation_engine's contract).
4. The bounded tail (``TAIL_SESSIONS`` = 80 sessions) always covers the
   current month (<= 23 sessions) plus the one-month anchor lookback
   (<= 23 sessions), so it changes no answer; in the pathological case
   where the tail does not reach the anchor date while the full frame
   does, the anchor lookup falls back to the full frame so the result is
   still exactly the reference's.

Performance: one ``to_period`` over at most ``TAIL_SESSIONS`` rows and a
handful of scalar lookups per bar.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FAMILY_ID = "low_vol_high_beta_factor_momentum_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["SPHB", "SPLV"]     # [risk, defensive]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Copied verbatim from the curated candidate's "parameters" (byte-identical
# under json.dumps(sort_keys=True)).
PARAMETERS = {
    "lookback_months": 1,
    "rebalance": "monthly",
    "signal": "relative_trailing_return",
    "num_assets": 2,
    "hold_top_n": 1,
    "cash_fallback": False,
    "risk_ticker": "SPHB",
    "defensive_ticker": "SPLV",
    "tie_rule": "defensive",
}
EXPECTED_MIN_TRADES = 80

_RISK = PARAMETERS["risk_ticker"]
_DEFENSIVE = PARAMETERS["defensive_ticker"]
_LOOKBACK_MONTHS = int(PARAMETERS["lookback_months"])
# A calendar month has at most 23 XNYS sessions; the current month plus a
# one-month anchor lookback is therefore < 50 sessions. 80 is a lookup
# bound with margin, not a rule parameter.
TAIL_SESSIONS = 80


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(index)
    if out.tz is not None:
        out = out.tz_localize(None)
    return out


def _decision_position(tail_index: pd.DatetimeIndex) -> int:
    """Position (within the tail) of the first session of the current month --
    the ``_monthly_decision_frame`` convention applied to the tail."""
    periods = tail_index.to_period("M")
    return int(np.flatnonzero(periods == periods[-1])[0])


def _anchor_close(frame: pd.DataFrame, tail_closes: pd.Series,
                  decision_date: pd.Timestamp):
    """Close on the last session on or before (decision_date - 1 month), read
    from the tail; falls back to the full frame only if the tail does not
    reach the anchor while the frame does (never the case on a real
    calendar with TAIL_SESSIONS = 80)."""
    anchor = decision_date - pd.DateOffset(months=_LOOKBACK_MONTHS)
    eligible = tail_closes.loc[tail_closes.index <= anchor]
    if eligible.empty and len(frame) > len(tail_closes):
        full = frame["Close"].copy()
        full.index = _naive_index(full.index)
        eligible = full.loc[full.index <= anchor]
    if eligible.empty:
        return None
    value = float(eligible.iloc[-1])
    return value if math.isfinite(value) and value > 0.0 else None


def _tail_closes(frame: pd.DataFrame) -> pd.Series:
    tail = frame["Close"].iloc[-TAIL_SESSIONS:]
    tail = pd.Series(pd.to_numeric(tail, errors="coerce").to_numpy(dtype=float),
                     index=_naive_index(tail.index))
    return tail


def make_target_fn(frames: dict):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED (+ SIGNAL, none here); only their
    presence is checked -- every price the rule reads comes from the
    TRUNCATED frames the engine hands over at call time."""
    for key in (_RISK, _DEFENSIVE):
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")

    def target(truncated: dict):
        risk_frame = truncated[_RISK]
        defensive_frame = truncated[_DEFENSIVE]
        if risk_frame.empty or defensive_frame.empty:
            return None
        risk_tail = _tail_closes(risk_frame)
        defensive_tail = _tail_closes(defensive_frame)
        position = _decision_position(risk_tail.index)
        decision_date = risk_tail.index[position]
        risk_now = float(risk_tail.iloc[position])
        defensive_now = float(defensive_tail.iloc[position])
        risk_base = _anchor_close(risk_frame, risk_tail, decision_date)
        defensive_base = _anchor_close(defensive_frame, defensive_tail,
                                       decision_date)
        if risk_base is None or defensive_base is None:
            return None
        if not (math.isfinite(risk_now) and math.isfinite(defensive_now)):
            return None
        risk_return = (risk_now - risk_base) / risk_base
        defensive_return = (defensive_now - defensive_base) / defensive_base
        return _RISK if risk_return > defensive_return else _DEFENSIVE

    return target
