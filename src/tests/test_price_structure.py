"""Tests for src/mining/price_structure.py -- three pure, stateless,
vectorized primitives shared by several independent price-action trading
frameworks (the mechanical core survives translation across their differing
vocabularies): a 3-candle price gap, the midpoint of a bar's own range, and
a "swept the prior extreme then closed back inside" reversal classification.

RED-first (TDD): every case below is hand-computed against a small synthetic
OHLC frame, written BEFORE price_structure.py exists. This file must fail to
collect (ModuleNotFoundError) until the module is created.
"""
import os
import sys

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.price_structure import (
    fair_value_gaps,
    equilibrium,
    classify_reversal_bar,
    daily_bias,
)


def _bars(high, low, close=None, index=None):
    """Build a minimal OHLC DataFrame. Close defaults to the midpoint of
    High/Low when a test doesn't care about it (fair_value_gaps and
    equilibrium never read Close)."""
    n = len(high)
    if index is None:
        index = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {"High": list(high), "Low": list(low)}
    if close is not None:
        data["Close"] = list(close)
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# PRIMITIVE 1: fair_value_gaps
# ---------------------------------------------------------------------------

def test_fair_value_gaps_worked_example_all_bullish():
    """High=[10,12,15,20,18], Low=[8,10,13,17,16] at positions 0..4.
    i=1: High[0]=10 < Low[2]=13 -> BULLISH gap (10, 13)
    i=2: High[1]=12 < Low[3]=17 -> BULLISH gap (12, 17)
    i=3: High[2]=15 < Low[4]=16 -> BULLISH gap (15, 16)
         (also: High[4]=18 > Low[2]=13, so i=3 is NOT also bearish)
    """
    bars = _bars(high=[10, 12, 15, 20, 18], low=[8, 10, 13, 17, 16])
    out = fair_value_gaps(bars)

    assert list(out.columns) == ["direction", "gap_low", "gap_high"]
    assert len(out) == 3

    row1 = out.loc[bars.index[1]]
    assert row1["direction"] == "bullish"
    assert row1["gap_low"] == pytest.approx(10)
    assert row1["gap_high"] == pytest.approx(13)

    row2 = out.loc[bars.index[2]]
    assert row2["direction"] == "bullish"
    assert row2["gap_low"] == pytest.approx(12)
    assert row2["gap_high"] == pytest.approx(17)

    row3 = out.loc[bars.index[3]]
    assert row3["direction"] == "bullish"
    assert row3["gap_low"] == pytest.approx(15)
    assert row3["gap_high"] == pytest.approx(16)


def test_fair_value_gaps_worked_example_bearish():
    """High=[20,18,15,10,12], Low=[18,16,13,8,10] at positions 0..4.
    i=1: Low[0]=18 > High[2]=15 -> BEARISH gap spanning (High[2]=15, Low[0]=18)
    """
    bars = _bars(high=[20, 18, 15, 10, 12], low=[18, 16, 13, 8, 10])
    out = fair_value_gaps(bars)

    row1 = out.loc[bars.index[1]]
    assert row1["direction"] == "bearish"
    # gap_low < gap_high always, regardless of direction
    assert row1["gap_low"] == pytest.approx(15)
    assert row1["gap_high"] == pytest.approx(18)


def test_fair_value_gaps_no_gap_monotonic_overlapping_ranges():
    """High=[10,11,12], Low=[8,9,10] -- consecutive ranges always overlap
    (each bar's low sits inside the prior bar's range), so no candle ever
    creates an untraded void. Must return an empty frame with the right
    columns, not raise."""
    bars = _bars(high=[10, 11, 12], low=[8, 9, 10])
    out = fair_value_gaps(bars)

    assert list(out.columns) == ["direction", "gap_low", "gap_high"]
    assert len(out) == 0


def test_fair_value_gaps_empty_input_returns_empty_frame():
    bars = _bars(high=[], low=[])
    out = fair_value_gaps(bars)
    assert list(out.columns) == ["direction", "gap_low", "gap_high"]
    assert len(out) == 0


def test_fair_value_gaps_one_row_input_returns_empty_frame():
    """A single bar has neither a predecessor nor a successor -- never
    raises, never emits a row."""
    bars = _bars(high=[10], low=[8])
    out = fair_value_gaps(bars)
    assert list(out.columns) == ["direction", "gap_low", "gap_high"]
    assert len(out) == 0


def test_fair_value_gaps_two_row_input_returns_empty_frame():
    """Two bars: neither index has BOTH a predecessor and a successor."""
    bars = _bars(high=[10, 12], low=[8, 10])
    out = fair_value_gaps(bars)
    assert list(out.columns) == ["direction", "gap_low", "gap_high"]
    assert len(out) == 0


def test_fair_value_gaps_indexed_by_middle_displacement_candle():
    """The returned frame is indexed by the middle candle's own timestamp
    (bars.index[i]), not by position or by the gap's edges."""
    bars = _bars(high=[10, 12, 15], low=[8, 10, 13])
    out = fair_value_gaps(bars)
    assert len(out) == 1
    assert out.index[0] == bars.index[1]


# ---------------------------------------------------------------------------
# PRIMITIVE 2: equilibrium
# ---------------------------------------------------------------------------

def test_equilibrium_hand_computed_three_rows():
    bars = _bars(high=[10, 20, 100], low=[8, 10, 50])
    out = equilibrium(bars)

    assert isinstance(out, pd.Series)
    assert list(out.index) == list(bars.index)
    assert out.iloc[0] == pytest.approx((10 + 8) / 2.0)
    assert out.iloc[1] == pytest.approx((20 + 10) / 2.0)
    assert out.iloc[2] == pytest.approx((100 + 50) / 2.0)


def test_equilibrium_empty_input_returns_empty_series_never_raises():
    bars = _bars(high=[], low=[])
    out = equilibrium(bars)
    assert isinstance(out, pd.Series)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# PRIMITIVE 3: classify_reversal_bar
# ---------------------------------------------------------------------------

def test_classify_reversal_bar_bullish_reversal_worked_example():
    """prior bar: Low=100, High=110. Current bar Low=98, High=108, Close=105
    -> Low(98)<100 and Close(105)>100 -> BULLISH_REVERSAL."""
    bars = _bars(high=[110, 108], low=[100, 98], close=[105, 105])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "bullish_reversal"


def test_classify_reversal_bar_bearish_reversal_worked_example():
    """prior bar: Low=100, High=110. Current bar Low=105, High=115, Close=107
    -> High(115)>110 and Close(107)<110 -> BEARISH_REVERSAL."""
    bars = _bars(high=[110, 115], low=[100, 105], close=[105, 107])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "bearish_reversal"


def test_classify_reversal_bar_none_when_fully_inside_prior_range():
    """prior bar: Low=100, High=110. Current bar Low=104, High=106, Close=105
    -> stays fully inside prior range -> "none"."""
    bars = _bars(high=[110, 106], low=[100, 104], close=[105, 105])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "none"


def test_classify_reversal_bar_bullish_break_worked_example():
    """prior bar: Low=100, High=110. Current bar Low=106, High=120,
    Close=118 -> Close(118)>High[i-1](110), clean break, no rejection since
    Low[i]=106 is NOT < Low[i-1]=100 -> BULLISH_BREAK."""
    bars = _bars(high=[110, 120], low=[100, 106], close=[105, 118])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "bullish_break"


def test_classify_reversal_bar_bearish_break():
    """Mirror of the bullish-break example: prior bar Low=100, High=110.
    Current bar Low=80, High=94, Close=90 -> Close(90) < Low[i-1](100),
    clean break lower, no rejection since High[i]=94 is NOT > High[i-1]=110
    -> BEARISH_BREAK."""
    bars = _bars(high=[110, 94], low=[100, 80], close=[105, 90])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "bearish_break"


def test_classify_reversal_bar_outside_bar_tiebreak_close_upper_half_bullish():
    """Outside bar: sweeps BOTH the prior low and prior high (Low[i]<Low[i-1]
    AND High[i]>High[i-1]). prior bar: Low=100, High=110. Current bar
    Low=95, High=120, Close=116 -> Close(116) is in the upper half of the
    current bar's own [95,120] range (midpoint 107.5) and Close>Low[i-1]=100
    -> tie-break rule calls this BULLISH_REVERSAL."""
    bars = _bars(high=[110, 120], low=[100, 95], close=[105, 116])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "bullish_reversal"


def test_classify_reversal_bar_outside_bar_tiebreak_close_lower_half_bearish():
    """Same outside-bar shape, Close in the LOWER half of the current bar's
    own range this time. prior bar: Low=100, High=110. Current bar Low=95,
    High=120, Close=99 -> midpoint of [95,120] is 107.5, Close(99) is below
    it -> tie-break rule calls this BEARISH_REVERSAL."""
    bars = _bars(high=[110, 120], low=[100, 95], close=[105, 99])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "bearish_reversal"


def test_classify_reversal_bar_outside_bar_tiebreak_exact_midpoint_is_none():
    """Same outside-bar shape, Close landing EXACTLY on the current bar's
    own midpoint -> documented rare edge case -> "none". Current bar
    Low=95, High=120 -> midpoint (95+120)/2.0 = 107.5."""
    bars = _bars(high=[110, 120], low=[100, 95], close=[105, 107.5])
    out = classify_reversal_bar(bars)
    assert out.iloc[1] == "none"


def test_classify_reversal_bar_first_row_is_always_none():
    """No predecessor for the first row -- must always be "none",
    regardless of the first bar's own values."""
    bars = _bars(high=[110, 120, 130], low=[100, 95, 125], close=[105, 116, 128])
    out = classify_reversal_bar(bars)
    assert out.iloc[0] == "none"


def test_classify_reversal_bar_single_row_input_is_none_never_raises():
    bars = _bars(high=[110], low=[100], close=[105])
    out = classify_reversal_bar(bars)
    assert len(out) == 1
    assert out.iloc[0] == "none"


def test_classify_reversal_bar_empty_input_returns_empty_series_never_raises():
    bars = _bars(high=[], low=[], close=[])
    out = classify_reversal_bar(bars)
    assert isinstance(out, pd.Series)
    assert len(out) == 0


def test_classify_reversal_bar_index_matches_input():
    bars = _bars(high=[110, 106], low=[100, 104], close=[105, 105])
    out = classify_reversal_bar(bars)
    assert list(out.index) == list(bars.index)


# ---------------------------------------------------------------------------
# FUNCTION 4: daily_bias (composition over classify_reversal_bar)
# ---------------------------------------------------------------------------

def test_daily_bias_relabels_bullish_reversal_to_bullish():
    """Reuses the bullish-reversal worked example: prior day Low=100,
    High=110; today Low=98, High=108, Close=105 -- swept the prior day's
    low and closed back above it -> daily_bias relabels this "bullish"
    (failing to close below the previous day's low reads as a bullish
    reversal)."""
    daily_bars = _bars(high=[110, 108], low=[100, 98], close=[105, 105])
    out = daily_bias(daily_bars)
    assert out.iloc[1] == "bullish"


def test_daily_bias_relabels_bearish_reversal_to_bearish():
    daily_bars = _bars(high=[110, 115], low=[100, 105], close=[105, 107])
    out = daily_bias(daily_bars)
    assert out.iloc[1] == "bearish"


def test_daily_bias_relabels_bullish_break_to_bullish_continuation():
    """(A close below the previous day's low reads as a continuation lower --
    mirrored here on the bullish side.) Reuses the
    bullish-break worked example: prior day Low=100, High=110; today
    Low=106, High=120, Close=118 -- clean break above, no rejection."""
    daily_bars = _bars(high=[110, 120], low=[100, 106], close=[105, 118])
    out = daily_bias(daily_bars)
    assert out.iloc[1] == "bullish_continuation"


def test_daily_bias_relabels_bearish_break_to_bearish_continuation():
    """Direct implementation of the rule: a close below the previous day's
    low reads as a continuation lower."""
    daily_bars = _bars(high=[110, 94], low=[100, 80], close=[105, 90])
    out = daily_bias(daily_bars)
    assert out.iloc[1] == "bearish_continuation"


def test_daily_bias_relabels_none_to_none():
    daily_bars = _bars(high=[110, 106], low=[100, 104], close=[105, 105])
    out = daily_bias(daily_bars)
    assert out.iloc[1] == "none"


def test_daily_bias_first_row_is_always_none():
    daily_bars = _bars(high=[110, 120], low=[100, 95], close=[105, 116])
    out = daily_bias(daily_bars)
    assert out.iloc[0] == "none"


def test_daily_bias_empty_input_returns_empty_series_never_raises():
    daily_bars = _bars(high=[], low=[], close=[])
    out = daily_bias(daily_bars)
    assert isinstance(out, pd.Series)
    assert len(out) == 0


def test_daily_bias_is_a_thin_relabel_not_duplicated_logic():
    """daily_bias must agree with classify_reversal_bar under the relabel
    map on every bar of a mixed frame -- it should never diverge by
    re-deriving the classification independently."""
    daily_bars = _bars(
        high=[110, 108, 115, 120, 94, 106],
        low=[100, 98, 105, 106, 80, 104],
        close=[105, 105, 107, 118, 90, 105],
    )
    raw = classify_reversal_bar(daily_bars)
    relabeled = daily_bias(daily_bars)

    label_map = {
        "bullish_reversal": "bullish",
        "bearish_reversal": "bearish",
        "bullish_break": "bullish_continuation",
        "bearish_break": "bearish_continuation",
        "none": "none",
    }
    expected = raw.map(label_map)
    assert list(relabeled) == list(expected)
