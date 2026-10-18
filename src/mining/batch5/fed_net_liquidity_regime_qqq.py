"""Batch5 candidate: fed_net_liquidity_regime_qqq (single_long).

A QQQ-traded variant of ``fed_net_liquidity_regime`` -- IDENTICAL
mechanism, IDENTICAL data (FRED WALCL/WTREGEN/RRPONTSYD), IDENTICAL
net-liquidity/MA/lag math, traded on QQQ instead of SPY. This builds the QQQ variant the original
catalog entry's tickers list mentioned but the mechanism text never
exercised (see
``fed_net_liquidity_regime.py``'s own disclosure of this gap).

WHY A SEPARATE FAMILY_ID rather than pooling SPY+QQQ into one family:
this pipeline's own standing rule is that multi-ticker tests are pooled
into ONE trial per family, never one family split across tickers with no
stated pooling/rotation rule between them. The
mechanism gives no rule for choosing between SPY and QQQ or rotating
between them -- it is the SAME regime rule applied to a DIFFERENT single
traded instrument, which this pipeline treats as a genuinely distinct
hypothesis (a different real edge could exist on a different underlying)
paying its own trial, exactly like this catalog's many other
same-mechanism-different-ticker family pairs.

``fed_net_liquidity_risk_on_series`` does NOT need the pre-inception
burn-in cap money_market_fund_flow_riskoff_bounded_burnin required: it
uses a fixed 13-week TRAILING moving average, not an expanding window, so
it cannot accumulate multi-decade desensitization regardless of how much
WALCL/WTREGEN/RRPONTSYD history is fed in.

Design: byte-identical to ``fed_net_liquidity_regime.py`` except
``TRADED``/``FAMILY_ID``.
"""
from __future__ import annotations

import pandas as pd

from mining import signals
from mining.backtest_engine import SignalResult

FAMILY_ID = "fed_net_liquidity_regime_qqq"
SHAPE = "single_long"
TRADED = ["QQQ"]
SIGNAL: list = []  # external (FRED WALCL/WTREGEN/RRPONTSYD), not a price-frame ticker
# NOT "QQQ" -- this family's real ABI-1 benchmark is SPY (see module
# docstring and scripts/run_fed_net_liquidity_qqq_mining.py), which
# benchmarks every family against the market reference unconditionally,
# never against whatever is traded. A module attribute contradicting its
# own driver/docstring is a trap for a future reader even though nothing currently reads this
# specific attribute for a bespoke-driver family.
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
# Identical to the SPY sibling's own PARAMETERS -- the mechanism is
# unchanged, only the traded instrument differs.
PARAMETERS = {
    "liquidity_ma_weeks": 13,
    "rebalance": "weekly",
}
MA_WEEKS = int(PARAMETERS["liquidity_ma_weeks"])
LAG_TRADING_DAYS = 1
# A conservative floor for a weekly-rebalanced regime flag; the signal
# itself never looks at the traded instrument's own price, so the floor
# matches the SPY sibling's. Same conservative-floor convention as every
# sibling.
EXPECTED_MIN_TRADES = 100


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    return frame


def _empty_signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
    return SignalResult()


def make_signal_fn(ticker: str, index, frames: dict, *,
                   walcl=None, wtregen=None, rrpontsyd=None):
    """SignalFn factory -- see fed_net_liquidity_regime.py's own
    docstring for the walcl/wtregen/rrpontsyd keyword convention. All
    net-liquidity/MA/lag math is delegated to mining.signals unchanged."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    _require_frame(frames, ticker)
    if len(pd.DatetimeIndex(index)) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")

    if walcl is None or wtregen is None or rrpontsyd is None:
        return _empty_signal_fn
    if walcl.empty or wtregen.empty or rrpontsyd.empty:
        return _empty_signal_fn

    raw_regime = signals.fed_net_liquidity_risk_on_series(
        walcl, wtregen, rrpontsyd, ma_weeks=MA_WEEKS)
    if raw_regime.empty:
        return _empty_signal_fn

    available = signals.lag_boolean_regime_to_availability(
        raw_regime, index, lag_trading_days=LAG_TRADING_DAYS)
    if available.empty:
        return _empty_signal_fn

    return signals.make_external_regime_signal_fn(available)
