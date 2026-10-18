"""Pure, stateless, vectorized price-structure primitives shared by several
independent price-action trading frameworks -- the mechanical core survives
translation across their differing vocabularies (this generalizes concepts
common to several independent price-action educators; no single source is
named). Three primitives, plus one thin composition:

1. fair_value_gaps(bars): the standard 3-candle imbalance / untraded price
   void, where one candle's high (or low) never reaches a candle two bars
   later's low (or high).
2. equilibrium(bars): the midpoint of each bar's own High/Low range.
3. classify_reversal_bar(bars): a "swept the prior bar's extreme, then
   closed back inside it" rejection pattern versus a clean directional
   break, bar over bar.
4. daily_bias(daily_bars): classify_reversal_bar applied to daily bars and
   relabeled for readability at that timeframe (bullish/bearish/
   bullish_continuation/bearish_continuation/none).

This module implements ONLY these primitives. It does NOT implement a
tradeable signal, is NOT registered in the mining catalog, and does NOT run
any backtest -- it is reusable groundwork, matching this repo's existing
style in swing_signals.py / day_trading_signals.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_GAP_COLUMNS = ["direction", "gap_low", "gap_high"]


def fair_value_gaps(bars: pd.DataFrame) -> pd.DataFrame:
    """Detect 3-candle price gaps (an untraded price void left behind by a
    displacement candle).

    Mechanism: for every candle i that has both a predecessor (i-1) and a
    successor (i+1) in `bars`, compare the predecessor's extreme against the
    successor's opposite extreme:
      - BULLISH gap when bars.High[i-1] < bars.Low[i+1] -- the predecessor's
        high never reaches the successor's low, leaving an untraded void
        that price is expected to react to from below. Spans
        (bars.High[i-1], bars.Low[i+1]).
      - BEARISH gap when bars.Low[i-1] > bars.High[i+1] -- the mirror case.
        Spans (bars.High[i+1], bars.Low[i-1]).
    These two conditions are mutually exclusive for real OHLC data (High >=
    Low on every bar): if both held, chaining the inequalities gives
    prev_low > next_high >= next_low > prev_high >= prev_low, a
    contradiction. No runtime assertion is needed for this (spec-directed);
    it is a property of well-formed OHLC input, not something this function
    must defend against.

    Uses `bars.shift(1)`/`bars.shift(-1)` to compare each candle against its
    neighbors in one vectorized pass -- boundary rows (no predecessor, no
    successor) get NaN neighbors, and a NaN comparison is False, so the
    first and last row (and every row of a 0/1/2-row frame) simply never
    qualify without any special-cased length check.

    Returns a DataFrame indexed by the MIDDLE candle's own timestamp
    (bars.index[i], the displacement candle) with columns ["direction"
    ("bullish"/"bearish"), "gap_low", "gap_high"] (gap_low < gap_high
    always, regardless of direction -- direction tells you which side price
    is expected to react from). Rows where no gap exists are simply absent
    (a sparse event table, not a per-bar frame with NaN rows). Never raises
    on an empty, 1-row, or 2-row input -- returns an empty frame with the
    same columns.
    """
    prev_high = bars["High"].shift(1)
    prev_low = bars["Low"].shift(1)
    next_high = bars["High"].shift(-1)
    next_low = bars["Low"].shift(-1)

    bullish_mask = prev_high < next_low
    bearish_mask = prev_low > next_high

    direction = pd.Series(
        np.where(bullish_mask, "bullish", np.where(bearish_mask, "bearish", None)),
        index=bars.index,
    )
    gap_low = pd.Series(
        np.where(bullish_mask, prev_high, np.where(bearish_mask, next_high, np.nan)),
        index=bars.index,
    )
    gap_high = pd.Series(
        np.where(bullish_mask, next_low, np.where(bearish_mask, prev_low, np.nan)),
        index=bars.index,
    )

    has_gap = bullish_mask | bearish_mask
    out = pd.DataFrame(
        {
            "direction": direction[has_gap],
            "gap_low": gap_low[has_gap],
            "gap_high": gap_high[has_gap],
        }
    )
    return out.reindex(columns=_GAP_COLUMNS)


def equilibrium(bars: pd.DataFrame) -> pd.Series:
    """The midpoint of each bar's own High/Low range.

    Mechanism: trivial per-bar average, (High + Low) / 2.0 -- no lookback,
    no state. Returns a pd.Series aligned to `bars.index`. Never raises on
    an empty input -- returns an empty Series.
    """
    return (bars["High"] + bars["Low"]) / 2.0


def classify_reversal_bar(bars: pd.DataFrame) -> pd.Series:
    """Classify each bar, versus its immediate predecessor, as a rejection
    of a swept extreme, a clean directional break, or neither.

    Mechanism: for every bar i with a predecessor i-1:
      - BULLISH_REVERSAL: bars.Low[i] < bars.Low[i-1] (bar i traded below
        the prior bar's low -- swept it) AND bars.Close[i] > bars.Low[i-1]
        (closed back above it -- a rejection).
      - BEARISH_REVERSAL: bars.High[i] > bars.High[i-1] (swept the prior
        high) AND bars.Close[i] < bars.High[i-1] (closed back below it).
      - BULLISH_BREAK: bars.Close[i] > bars.High[i-1] -- closed clean above
        the prior high, no rejection.
      - BEARISH_BREAK: bars.Close[i] < bars.Low[i-1] -- closed clean below
        the prior low, no rejection.
      - Otherwise "none" (bar stayed within the prior bar's range).

    Disclosed tie-break rule for an OUTSIDE BAR (bars.Low[i] <
    bars.Low[i-1] AND bars.High[i] > bars.High[i-1] -- i.e. bar i swept
    BOTH the prior low and the prior high at once, which is the only shape
    that can otherwise satisfy both reversal conditions simultaneously):
    classify by which extreme bar i's own CLOSE ended up rejecting, i.e.
    where Close[i] sits relative to the midpoint of bar i's OWN
    [Low[i], High[i]] range (its own `equilibrium`) -- upper half ->
    BULLISH_REVERSAL, lower half -> BEARISH_REVERSAL, exact midpoint tie ->
    "none" (a rare, disclosed edge case). This rule is checked first and
    takes priority over the four plain rules above for an outside bar.

    Returns a pd.Series indexed like `bars`, values in
    {"bullish_reversal", "bearish_reversal", "bullish_break",
    "bearish_break", "none"}. The first row is always "none" (no
    predecessor) -- this falls out of `shift(1)` producing NaN neighbors
    for row 0, and every comparison against NaN is False, so no special
    first-row check is needed. Never raises on an empty or 1-row input.
    """
    high = bars["High"]
    low = bars["Low"]
    close = bars["Close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    swept_low = low < prev_low
    swept_high = high > prev_high
    outside = swept_low & swept_high
    own_mid = equilibrium(bars)

    bullish_reversal_cond = (~outside) & swept_low & (close > prev_low)
    bearish_reversal_cond = (~outside) & swept_high & (close < prev_high)
    bullish_break_cond = (~outside) & (close > prev_high)
    bearish_break_cond = (~outside) & (close < prev_low)

    conditions = [
        outside & (close > own_mid),
        outside & (close < own_mid),
        bullish_reversal_cond,
        bearish_reversal_cond,
        bullish_break_cond,
        bearish_break_cond,
    ]
    choices = [
        "bullish_reversal",
        "bearish_reversal",
        "bullish_reversal",
        "bearish_reversal",
        "bullish_break",
        "bearish_break",
    ]
    result = np.select(conditions, choices, default="none")
    return pd.Series(result, index=bars.index)


_DAILY_BIAS_LABELS = {
    "bullish_reversal": "bullish",
    "bearish_reversal": "bearish",
    "bullish_break": "bullish_continuation",
    "bearish_break": "bearish_continuation",
    "none": "none",
}


def daily_bias(daily_bars: pd.DataFrame) -> pd.Series:
    """Relabel `classify_reversal_bar`'s output for readability at the
    daily timeframe -- a thin wrapper, not a re-derivation.

    Mechanism: bullish_reversal -> "bullish" (swept the previous day's low
    and closed back above it -- expect a reversal higher); bearish_reversal
    -> "bearish" (mirror); bullish_break -> "bullish_continuation" (closed
    clean above the previous day's high -- expect continuation higher);
    bearish_break -> "bearish_continuation" (closed clean below the
    previous day's low -- expect continuation lower); none -> "none".

    Returns a pd.Series indexed like `daily_bars`. Never raises on an empty
    input; the first row is always "none" (inherited from
    classify_reversal_bar).
    """
    raw = classify_reversal_bar(daily_bars)
    return raw.map(_DAILY_BIAS_LABELS)
