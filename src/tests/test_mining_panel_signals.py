"""Tests for mining.panel_signals -- the breadth reductions of a panel.

House conventions followed deliberately: no @pytest.fixture, module-level
`_`-prefixed builders, no network, and every synthetic panel small enough that
the expected value is hand-computed in the test's own docstring rather than
asserted as a shape.

Every expected value in this file is chosen to be exactly representable in
binary floating point (quarters, halves, eighths), so the assertions use `==`
and need no tolerance -- the same exact-value convention
test_mining_intraday_cost_gate.py follows.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.constants import PANEL_MIN_RESOLVED_MEMBERS
from mining import panel_signals


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def _panel(rows, tickers, *, start="2024-01-02"):
    """Build a price panel. `None` in a cell means NO BAR (NaN), never 0.0."""
    index = pd.date_range(start, periods=len(rows), freq="B")
    values = [[np.nan if cell is None else float(cell) for cell in row]
              for row in rows]
    return pd.DataFrame(values, index=index, columns=list(tickers))


def _mask(prices, value=True):
    return pd.DataFrame(bool(value), index=prices.index,
                        columns=prices.columns, dtype=bool)


def _participation_panel():
    """Four tickers, six sessions, window=3. Hand-computed 3-session means:

        A  10 20 30 40 50 60   ma3 t2..t5 = 20,30,40,50 -> ABOVE on all four
        B  60 50 40 30 20 10   ma3 t2..t5 = 50,40,30,20 -> BELOW on all four
        C  25 25 25 25 25 25   ma3 = 25 == px, and `>` is strict -> BELOW
        D  10 10 10 40 10 40   ma3 t2..t5 = 10,20,20,30 -> BELOW,ABOVE,BELOW,ABOVE

    so participation is 1/4, 2/4, 1/4, 2/4 at t2..t5 -- all exact in binary.
    """
    prices = _panel(
        [
            [10, 60, 25, 10],
            [20, 50, 25, 10],
            [30, 40, 25, 10],
            [40, 30, 25, 40],
            [50, 20, 25, 10],
            [60, 10, 25, 40],
        ],
        ["A", "B", "C", "D"],
    )
    seasoned = _mask(prices)
    seasoned.iloc[0:2, :] = False          # three own bars are needed first
    return prices, seasoned


def _advance_decline_panel():
    """Four tickers, four sessions, seasoned throughout. Session-over-session:

        t1: A +1 B -1 C  0 D +2  -> 2 adv, 1 dec, 4 eligible -> +1/4 = 0.25
        t2: A +1 B -1 C  0 D -1  -> 1 adv, 2 dec, 4 eligible -> -1/4 = -0.25
        t3: A  0 B  0 C +1 D -1  -> 1 adv, 1 dec, 4 eligible ->  0/4 = 0.0

    t0 has no prior session, so it is unknown by construction.
    """
    prices = _panel(
        [
            [10, 10, 10, 10],
            [11, 9, 10, 12],
            [12, 8, 10, 11],
            [12, 8, 11, 10],
        ],
        ["A", "B", "C", "D"],
    )
    return prices, _mask(prices)


def _hole_then_clean_tail_panel():
    """Four tickers, seven sessions, seasoned throughout, with B missing a bar
    at t2. Session-over-session, once the gap is behind:

        t1: A +1 B -1 C  0 D +2  ->  2 adv, 1 dec, 4 eligible ->  0.25
        t2: B has no bar          ->  unknown (broken session)
        t3: unknown too           ->  every member's move spans the gap
        t4: A +1 B -1 C +1 D -1  ->  2 adv, 2 dec, 4 eligible ->  0.0
        t5: A +1 B -1 C  0 D -1  ->  1 adv, 2 dec, 4 eligible -> -0.25
        t6: A +1 B -1 C +1 D  0  ->  2 adv, 1 dec, 4 eligible ->  0.25

    so the ratio series is [nan, 0.25, nan, nan, 0.0, -0.25, 0.25] and the
    clean tail t4..t6 is [0.0, -0.25, 0.25] -- every value a binary-exact
    quarter. t0 has no predecessor and is unknown by construction.
    """
    prices = _panel(
        [
            [10, 10, 10, 10],
            [11, 9, 10, 12],
            [12, None, 10, 11],
            [13, 8, 11, 10],
            [14, 7, 12, 9],
            [15, 6, 12, 8],
            [16, 5, 13, 8],
        ],
        ["A", "B", "C", "D"],
    )
    return prices, _mask(prices)


def _eleven_rising(n_sessions=4):
    """Eleven tickers, each rising 10, 20, 30, ... so every session advances."""
    tickers = [f"T{i:02d}" for i in range(11)]
    rows = [[10 * (step + 1)] * 11 for step in range(n_sessions)]
    return _panel(rows, tickers), tickers


# --------------------------------------------------------------------------
# participation above a moving average -- hand-computed
# --------------------------------------------------------------------------

def test_participation_matches_a_hand_computed_three_session_average():
    prices, seasoned = _participation_panel()
    result = panel_signals.breadth_participation(
        prices, seasoned, window=3, min_members=1)

    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == 0.25
    assert result.iloc[3] == 0.5
    assert result.iloc[4] == 0.25
    assert result.iloc[5] == 0.5
    assert result.index.equals(prices.index)


def test_participation_treats_price_equal_to_its_average_as_not_above():
    """Ticker C sits exactly on its own mean every session. `>` is strict, so
    it is never counted as a participant -- pinned because flipping to `>=`
    would silently add a member to every numerator on a flat tape."""
    prices, seasoned = _participation_panel()
    single = prices.loc[:, ["C"]]
    result = panel_signals.breadth_participation(
        single, seasoned.loc[:, ["C"]], window=3, min_members=1)
    assert result.iloc[2] == 0.0
    assert result.iloc[5] == 0.0


def test_participation_denominator_excludes_unseasoned_and_non_members():
    """A/C rise, B/E fall, and D rises but is NOT seasoned. The denominator
    must be 4 (A, B, C, E) and the value 2/4 = 0.5. D is deliberately RISING,
    so a leak is visible in the numerator too: were D counted it would be
    3/4 = 0.75, and were it counted on both sides 3/5 = 0.6. All three are
    exactly representable, so the outcomes are distinguishable without
    tolerance."""
    prices = _panel(
        [
            [10, 40, 10, 10, 40],
            [20, 30, 20, 20, 30],
            [30, 20, 30, 30, 20],
        ],
        ["A", "B", "C", "D", "E"],
    )
    seasoned = _mask(prices)
    seasoned.iloc[0, :] = False
    seasoned.loc[:, "D"] = False           # a member of nothing: never counted

    result = panel_signals.breadth_participation(
        prices, seasoned, window=2, min_members=1)
    counts = panel_signals.resolved_member_count(prices, seasoned)

    assert counts.iloc[1] == 4
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 0.5


def test_participation_moving_average_uses_only_that_tickers_own_bars():
    """D's gap at t3 makes t3 itself unknown, but it must NOT poison the next
    `window - 1` sessions. A plain frame-level rolling mean returns NaN for
    `window - 1` sessions after any gap, which would silently demote that
    ticker to 'below its average' -- the NaN-is-not-zero failure in its
    quietest form. D's own bars through t4 are 10, 20, 30, 60, so its trailing
    3-bar mean at t4 is (20 + 30 + 60) / 3 = 110/3, which px 60 exceeds; A and
    C are above theirs and B is below, so t4 is 3/4."""
    prices = _panel(
        [
            [10, 40, 10, 10],
            [20, 30, 20, 20],
            [30, 20, 30, 30],
            [40, 10, 40, None],
            [50, 5, 50, 60],
            [60, 1, 60, 70],
        ],
        ["A", "B", "C", "D"],
    )
    seasoned = _mask(prices)
    seasoned.iloc[0:2, :] = False

    result = panel_signals.breadth_participation(
        prices, seasoned, window=3, min_members=1)

    assert math.isnan(result.iloc[3])      # the gap session is unknown
    assert result.iloc[4] == 0.75
    assert result.iloc[5] == 0.75


def test_participation_raises_when_the_seasoned_mask_outruns_the_bars():
    """The seasoned mask is the caller's; if it claims a ticker is seasoned
    before that ticker has `window` bars of its own, the moving average is
    undefined and the honest answer is a refusal -- never a member silently
    scored 'below' because its average is NaN."""
    prices = _panel(
        [
            [None, 60, 25, 10],
            [20, 50, 25, 10],
            [30, 40, 25, 10],
        ],
        ["A", "B", "C", "D"],
    )
    seasoned = _mask(prices)
    seasoned.iloc[0:2, :] = False

    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            prices, seasoned, window=3, min_members=1)
    assert "A" in str(excinfo.value)
    assert "seasoned" in str(excinfo.value)


# --------------------------------------------------------------------------
# broken sessions -- NaN is not zero
# --------------------------------------------------------------------------

def test_a_missing_bar_for_a_seasoned_member_makes_the_whole_session_unknown():
    """B is a seasoned member with no bar at t3. The other three tickers are
    fine, so a 'compute over whoever resolved' implementation would return a
    clean, plausible 1/3. It must be NaN: a ratio with an unknown member is
    unknown, not approximate."""
    prices, seasoned = _participation_panel()
    prices.iloc[3, 1] = np.nan             # B has no bar at t3

    broken = panel_signals.broken_sessions(prices, seasoned)
    result = panel_signals.breadth_participation(
        prices, seasoned, window=3, min_members=1)

    assert bool(broken.iloc[3]) is True
    assert broken.sum() == 1
    assert math.isnan(result.iloc[3])
    # The sessions on either side are untouched -- brokenness is not contagious
    # for a point-in-time indicator.
    assert result.iloc[2] == 0.25


def test_broken_sessions_ignores_a_missing_bar_for_an_unseasoned_ticker():
    """A ticker that is not yet a seasoned member has no bar to miss. Only a
    seasoned member's absence makes the session unknown."""
    prices, seasoned = _participation_panel()
    prices.iloc[0, 1] = np.nan             # t0, where nothing is seasoned yet
    broken = panel_signals.broken_sessions(prices, seasoned)
    assert broken.sum() == 0


def test_a_zero_price_is_refused_rather_than_read_as_a_hundred_percent_decline():
    """pooling.py's `fill_value=0.0` union is correct for summing P&L and
    catastrophic on prices: a 0.00 close is a -100% return and an automatic
    'below its 200-day average'. Refuse it at the door, naming the cell."""
    prices, seasoned = _participation_panel()
    prices.iloc[3, 1] = 0.0

    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            prices, seasoned, window=3, min_members=1)
    message = str(excinfo.value)
    assert "B" in message
    assert "positive" in message


def test_a_non_finite_price_is_refused_rather_than_read_as_a_real_print():
    """Zero and negative are only ONE half of "is this a real price". `+inf`
    is not NaN and does satisfy `> 0.0`, so a detector written as
    `~isnan(v) & ~(v > 0.0)` admits it.

    With such a detector, on the fully-seasoned panel
    A = [10, 11, 12, 13, 14] / B = [10, 11, 12, inf, 14],
    `breadth_net_advance_ratio(..., min_members=1)` would return
    [nan, 1.0, 1.0, 1.0, 0.0] with no raise -- the inf print counted as B
    advancing on its own session and, because `14 - inf = -inf`, as B
    declining on the next -- and `breadth_cumulative_ad_line` would integrate
    both into a running total of 2.0: one garbage value read as two market
    facts a downstream DSR score would believe. `breadth_participation` would
    raise only incidentally, via the undefined-trailing-mean guard
    (`inf - inf` is NaN inside the rolling sum), blaming the seasoned mask
    for a bar count that is not wrong.

    -inf is covered in the same loop so the detector's converse cannot regress
    either: it fails `> 0.0` anyway, but a finiteness test is what actually
    states the intent.
    """
    for garbage in (np.inf, -np.inf):
        prices, seasoned = _advance_decline_panel()
        prices.iloc[2, 1] = garbage

        with pytest.raises(ValueError) as excinfo:
            panel_signals.breadth_net_advance_ratio(
                prices, seasoned, min_members=1)
        message = str(excinfo.value)
        assert "B" in message, garbage
        assert "finite" in message, garbage
        assert str(prices.index[2].date()) in message, garbage


# --------------------------------------------------------------------------
# net advance ratio -- hand-computed
# --------------------------------------------------------------------------

def test_net_advance_ratio_matches_hand_counted_advancers_and_decliners():
    prices, seasoned = _advance_decline_panel()
    result = panel_signals.breadth_net_advance_ratio(
        prices, seasoned, min_members=1)

    assert math.isnan(result.iloc[0])      # no prior session exists
    assert result.iloc[1] == 0.25
    assert result.iloc[2] == -0.25
    assert result.iloc[3] == 0.0


def test_net_advance_ratio_is_unknown_on_the_session_after_a_gap_too():
    """B, a seasoned member, has no bar at t2. That makes t2 unknown -- but it
    also makes t3 unknown, because B's move INTO t3 would have to span the
    gap. Dropping B from t3's count instead would publish a clean, plausible
    1/3 built from three of four members while the fourth's move is
    unmeasured. t1 (0.25) and t4 (0.0) are unaffected."""
    prices = _panel(
        [
            [10, 10, 10, 10],
            [11, 9, 10, 12],
            [12, None, 10, 11],
            [13, 8, 11, 10],
            [14, 7, 12, 9],
        ],
        ["A", "B", "C", "D"],
    )
    result = panel_signals.breadth_net_advance_ratio(
        prices, _mask(prices), min_members=1)

    assert result.iloc[1] == 0.25
    assert math.isnan(result.iloc[2])
    assert math.isnan(result.iloc[3])
    assert result.iloc[4] == 0.0


def test_net_advance_counts_report_advancers_decliners_and_the_denominator():
    prices, seasoned = _advance_decline_panel()
    counts = panel_signals.net_advance_counts(prices, seasoned)

    assert list(counts.columns) == ["advancers", "decliners", "eligible"]
    assert counts["advancers"].tolist() == [0, 2, 1, 1]
    assert counts["decliners"].tolist() == [0, 1, 2, 1]
    assert counts["eligible"].tolist() == [0, 4, 4, 4]


def test_net_advance_ratio_has_no_level_shift_across_a_membership_change():
    """Nine members become eleven, with identical per-member behaviour on both
    sides (every member advances). The RATIO is 1.0 on both sides; the raw net
    advance COUNT is 9 then 11. That level shift is exactly why the ratio form
    is mandatory rather than stylistic on a panel whose membership grows."""
    prices, tickers = _eleven_rising(6)
    seasoned = _mask(prices)
    seasoned.loc[:, tickers[9:]] = False
    seasoned.iloc[3:, :] = True            # the last two join at t3

    result = panel_signals.breadth_net_advance_ratio(
        prices, seasoned, min_members=PANEL_MIN_RESOLVED_MEMBERS)
    counts = panel_signals.net_advance_counts(prices, seasoned)

    assert result.iloc[2] == 1.0
    assert result.iloc[4] == 1.0
    raw_net = counts["advancers"] - counts["decliners"]
    assert raw_net.iloc[2] == 9
    assert raw_net.iloc[4] == 11


def test_min_resolved_members_boundary_is_inclusive():
    """Exactly PANEL_MIN_RESOLVED_MEMBERS resolved members is enough; one
    fewer is not. Constructed with integer counts and an exactly representable
    value (1.0), so no float tolerance is involved."""
    prices, tickers = _eleven_rising(4)
    seasoned = _mask(prices)
    seasoned.iloc[0, :] = False
    seasoned.loc[seasoned.index[2], tickers[9:]] = False        # 9 resolved
    seasoned.loc[seasoned.index[3], tickers[8:]] = False        # 8 resolved

    counts = panel_signals.resolved_member_count(prices, seasoned)
    result = panel_signals.breadth_participation(
        prices, seasoned, window=2, min_members=PANEL_MIN_RESOLVED_MEMBERS)

    assert counts.iloc[2] == PANEL_MIN_RESOLVED_MEMBERS
    assert counts.iloc[3] == PANEL_MIN_RESOLVED_MEMBERS - 1
    assert result.iloc[2] == 1.0
    assert math.isnan(result.iloc[3])


# --------------------------------------------------------------------------
# path-dependent indicators -- hand-computed, and refusing to bridge a gap
# --------------------------------------------------------------------------

def test_net_advance_oscillator_matches_a_hand_rolled_ewma_difference():
    """fast=1 (alpha = 1, so the fast leg IS the ratio) and slow=3
    (alpha = 0.5) are chosen so every intermediate is exact in binary.

        nar          0.25   -0.25    0.0
        slow ewma    0.25     0.0    0.0     (e_t = 0.5 x_t + 0.5 e_{t-1})
        oscillator   0.00   -0.25    0.0
    """
    prices, seasoned = _advance_decline_panel()
    result = panel_signals.breadth_net_advance_oscillator(
        prices, seasoned, fast=1, slow=3, min_members=1)

    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == -0.25
    assert result.iloc[3] == 0.0


def test_cumulative_ad_line_matches_a_hand_computed_running_total():
    """cumsum of 0.25, -0.25, 0.0 is 0.25, 0.0, 0.0."""
    prices, seasoned = _advance_decline_panel()
    result = panel_signals.breadth_cumulative_ad_line(
        prices, seasoned, min_members=1)

    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == 0.25
    assert result.iloc[2] == 0.0
    assert result.iloc[3] == 0.0


def test_cumulative_ad_trend_is_the_line_minus_its_own_trailing_mean():
    """line 0.25, 0.0, 0.0; trailing 2-session mean NaN, 0.125, 0.0;
    difference NaN, -0.125, 0.0."""
    prices, seasoned = _advance_decline_panel()
    result = panel_signals.breadth_cumulative_ad_trend(
        prices, seasoned, trend_days=2, min_members=1)

    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == -0.125
    assert result.iloc[3] == 0.0


def test_pandas_cumsum_and_ewm_silently_bridge_a_gap():
    """The platform fact the path-dependent refusal exists for, pinned here so
    a future pandas upgrade that changed it would be noticed. Measured on
    pandas 2.2.3: cumsum carries the running total straight across a NaN
    (the value after the gap is 2.0, as though the gap contributed 0.0)
    and ewm(adjust=False) holds its prior level through it."""
    gapped = pd.Series([1.0, np.nan, 1.0, 1.0])
    assert gapped.cumsum().iloc[2] == 2.0
    assert gapped.ewm(span=3, adjust=False).mean().iloc[1] == 1.0


def test_path_dependent_indicators_refuse_to_accumulate_across_a_broken_session():
    """A seasoned member with no bar mid-span makes that session's breadth
    genuinely unknown. A point-in-time indicator may yield NaN there; a
    cumulative or EWMA one may not, because pandas would bridge the gap
    (see the test above) and every subsequent value would silently rest on a
    session nobody measured."""
    prices, seasoned = _advance_decline_panel()
    prices.iloc[2, 1] = np.nan             # B, a seasoned member, has no bar

    broken_date = prices.index[2]
    for call in (
        lambda: panel_signals.breadth_net_advance_oscillator(
            prices, seasoned, fast=1, slow=3, min_members=1),
        lambda: panel_signals.breadth_cumulative_ad_line(
            prices, seasoned, min_members=1),
        lambda: panel_signals.breadth_cumulative_ad_trend(
            prices, seasoned, trend_days=2, min_members=1),
    ):
        with pytest.raises(ValueError) as excinfo:
            call()
        assert str(broken_date.date()) in str(excinfo.value)

    # The point-in-time indicator does NOT raise: it drops the session.
    seasoned_for_window_2 = seasoned.copy()
    seasoned_for_window_2.iloc[0, :] = False   # two own bars are needed first
    pointwise = panel_signals.breadth_participation(
        prices, seasoned_for_window_2, window=2, min_members=1)
    assert math.isnan(pointwise.iloc[2])


def test_path_dependent_indicators_tolerate_unknown_sessions_before_the_span():
    """The leading sessions of any panel are unknown (no prior bar, members not
    yet seasoned). Those precede the accumulation and are simply not part of
    it -- only a hole INSIDE the span is a refusal."""
    prices, seasoned = _advance_decline_panel()
    seasoned.iloc[0:2, :] = False          # nothing is known until t2

    result = panel_signals.breadth_cumulative_ad_line(
        prices, seasoned, min_members=1)
    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert math.isnan(result.iloc[2])      # t2 has no resolved prior session
    assert result.iloc[3] == 0.0           # first defined ratio: 1 adv, 1 dec


def test_path_dependent_indicators_raise_when_nothing_is_ever_known():
    prices, seasoned = _advance_decline_panel()
    seasoned.loc[:, :] = False

    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_cumulative_ad_line(
            prices, seasoned, min_members=1)
    assert "no known" in str(excinfo.value)


# --------------------------------------------------------------------------
# span_start -- the caller-stated escape from a hole
# --------------------------------------------------------------------------

def test_the_refusal_offers_span_start_by_name_so_a_caller_can_find_it():
    """The refusal is the ONLY place a caller meets this wall, so it has to
    say what the escape is called. Without this the escape exists but is
    undiscoverable, which is the same as not shipping it."""
    prices, seasoned = _hole_then_clean_tail_panel()
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_cumulative_ad_line(
            prices, seasoned, min_members=1)
    message = str(excinfo.value)
    assert "span_start" in message
    assert str(prices.index[2].date()) in message      # the hole, named


def test_a_caller_stated_span_start_accumulates_over_the_clean_tail_only():
    """A hole in the MIDDLE of a panel has no workaround by truncation --
    building to a date before it throws the whole tail away, which on a
    long-history basis whose member count sits at its floor can void a
    decade or more of history over a single missing print. The escape is not a
    tolerance: the caller NAMES the session the accumulation restarts at, and
    that date is what the snapshot's provenance records.

    Over the clean tail [0.0, -0.25, 0.25]:
        cumsum       0.0, -0.25,  0.0
        slow ewma (alpha 0.5)  0.0, -0.125, 0.0625
        oscillator (fast=1)    0.0, -0.125, 0.1875
        trailing 2-mean        nan, -0.125, -0.125
        trend                  nan, -0.125,  0.125
    every value a binary-exact sixteenth.
    """
    prices, seasoned = _hole_then_clean_tail_panel()
    restart = prices.index[4]

    line = panel_signals.breadth_cumulative_ad_line(
        prices, seasoned, min_members=1, span_start=restart)
    assert math.isnan(line.iloc[3])        # before the restart: not accumulated
    assert line.iloc[4] == 0.0
    assert line.iloc[5] == -0.25
    assert line.iloc[6] == 0.0

    oscillator = panel_signals.breadth_net_advance_oscillator(
        prices, seasoned, fast=1, slow=3, min_members=1, span_start=restart)
    assert math.isnan(oscillator.iloc[3])
    assert oscillator.iloc[4] == 0.0
    assert oscillator.iloc[5] == -0.125
    assert oscillator.iloc[6] == 0.1875

    trend = panel_signals.breadth_cumulative_ad_trend(
        prices, seasoned, trend_days=2, min_members=1, span_start=restart)
    assert math.isnan(trend.iloc[4])       # the trailing window is not full yet
    assert trend.iloc[5] == -0.125
    assert trend.iloc[6] == 0.125


def test_a_span_start_that_still_leaves_a_hole_inside_the_span_is_refused():
    """The escape moves the span's floor; it never relaxes the hole rule. A
    span_start BEFORE the gap must fail exactly as the unset case does, or the
    keyword would be a way to ask for the silent bridging the module exists to
    prevent."""
    prices, seasoned = _hole_then_clean_tail_panel()
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_cumulative_ad_line(
            prices, seasoned, min_members=1, span_start=prices.index[1])
    message = str(excinfo.value)
    assert str(prices.index[2].date()) in message
    assert str(prices.index[3].date()) in message


def test_a_span_start_may_name_a_session_that_is_itself_unknown():
    """t3 is unknown -- every member's move into it spans the gap -- so naming
    it starts the accumulation at the first KNOWN session at or after it. That
    is the same leading-unknowns rule the unset case already follows, applied
    from the caller's floor instead of from the panel's first session; the
    result is identical to naming t4 directly."""
    prices, seasoned = _hole_then_clean_tail_panel()
    from_unknown = panel_signals.breadth_cumulative_ad_line(
        prices, seasoned, min_members=1, span_start=prices.index[3])
    from_known = panel_signals.breadth_cumulative_ad_line(
        prices, seasoned, min_members=1, span_start=prices.index[4])
    pd.testing.assert_series_equal(from_unknown, from_known)


def test_a_span_start_on_a_hole_free_panel_is_honoured_not_ignored():
    """No hole exists, so the caller does not need the escape -- but a stated
    floor is still a stated floor. Accumulating from t2 of the clean
    advance-decline panel drops t1's +0.25 from the running total, which the
    unset call includes."""
    prices, seasoned = _advance_decline_panel()
    unset = panel_signals.breadth_cumulative_ad_line(
        prices, seasoned, min_members=1)
    started = panel_signals.breadth_cumulative_ad_line(
        prices, seasoned, min_members=1, span_start=prices.index[2])

    assert unset.iloc[1] == 0.25
    assert math.isnan(started.iloc[1])
    assert unset.iloc[2] == 0.0
    assert started.iloc[2] == -0.25


def test_a_span_start_with_no_known_session_at_or_after_it_is_refused():
    """Past the end of the calendar there is nothing to accumulate, and a
    silently empty series would read downstream as 'this indicator has no
    signal' rather than 'you asked for a window that does not exist'."""
    prices, seasoned = _hole_then_clean_tail_panel()
    beyond = prices.index[-1] + pd.Timedelta(days=30)
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_cumulative_ad_line(
            prices, seasoned, min_members=1, span_start=beyond)
    message = str(excinfo.value)
    assert "span_start" in message
    assert str(beyond.date()) in message


def test_span_start_must_be_a_timezone_naive_timestamp():
    """A string date or a tz-aware stamp would slice the calendar by luck
    rather than by contract -- and a tz-aware bound against the tz-naive
    calendar this module already insists on is a comparison error waiting for
    a pandas version bump."""
    prices, seasoned = _hole_then_clean_tail_panel()
    for bad in ("2024-01-08", 4, prices.index[4].tz_localize("America/New_York")):
        with pytest.raises(ValueError) as excinfo:
            panel_signals.breadth_cumulative_ad_line(
                prices, seasoned, min_members=1, span_start=bad)
        assert "span_start" in str(excinfo.value), bad


def test_span_start_is_keyword_only_on_every_path_dependent_indicator():
    """Keyword-only and defaulting to None: an accumulation with no stated
    floor stays the fail-closed default, and the escape is never taken by a
    positional argument slipping into place."""
    import inspect
    for name in (
        "breadth_net_advance_oscillator",
        "breadth_cumulative_ad_line",
        "breadth_cumulative_ad_trend",
    ):
        signature = inspect.signature(getattr(panel_signals, name))
        parameter = signature.parameters["span_start"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is None, name


# --------------------------------------------------------------------------
# no look-ahead
# --------------------------------------------------------------------------

def test_indicators_computed_on_a_truncated_panel_match_the_full_panel():
    """Every indicator here is trailing-only, so its value at session T must be
    identical whether or not sessions after T exist in the frame it was handed.
    This is the whole-frame analogue of the engine's own i+1 slice."""
    prices, seasoned = _participation_panel()
    cut = 4
    full = panel_signals.breadth_participation(
        prices, seasoned, window=3, min_members=1)
    truncated = panel_signals.breadth_participation(
        prices.iloc[:cut], seasoned.iloc[:cut], window=3, min_members=1)
    pd.testing.assert_series_equal(full.iloc[:cut], truncated)

    ad_prices, ad_seasoned = _advance_decline_panel()
    cut = 3
    for name, kwargs in (
        ("breadth_net_advance_ratio", {"min_members": 1}),
        ("breadth_net_advance_oscillator",
         {"fast": 1, "slow": 3, "min_members": 1}),
        ("breadth_cumulative_ad_line", {"min_members": 1}),
        ("breadth_cumulative_ad_trend", {"trend_days": 2, "min_members": 1}),
    ):
        fn = getattr(panel_signals, name)
        full = fn(ad_prices, ad_seasoned, **kwargs)
        truncated = fn(ad_prices.iloc[:cut], ad_seasoned.iloc[:cut], **kwargs)
        pd.testing.assert_series_equal(
            full.iloc[:cut], truncated, obj=name)

    # A stated span_start floor must not smuggle look-ahead in either: it moves
    # where the accumulation begins, never what any session already inside it
    # is worth.
    gap_prices, gap_seasoned = _hole_then_clean_tail_panel()
    restart = gap_prices.index[4]
    cut = 6
    for name, kwargs in (
        ("breadth_net_advance_oscillator",
         {"fast": 1, "slow": 3, "min_members": 1, "span_start": restart}),
        ("breadth_cumulative_ad_line",
         {"min_members": 1, "span_start": restart}),
        ("breadth_cumulative_ad_trend",
         {"trend_days": 2, "min_members": 1, "span_start": restart}),
    ):
        fn = getattr(panel_signals, name)
        full = fn(gap_prices, gap_seasoned, **kwargs)
        truncated = fn(gap_prices.iloc[:cut], gap_seasoned.iloc[:cut], **kwargs)
        pd.testing.assert_series_equal(
            full.iloc[:cut], truncated, obj=f"{name} with span_start")


# --------------------------------------------------------------------------
# fail-closed input validation
# --------------------------------------------------------------------------

def test_breadth_functions_reject_an_unsorted_or_duplicated_calendar():
    prices, seasoned = _participation_panel()
    shuffled = prices.iloc[[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            shuffled, seasoned.iloc[[1, 0, 2, 3, 4, 5]],
            window=3, min_members=1)
    assert "sorted unique" in str(excinfo.value)


def test_breadth_functions_reject_a_timezone_aware_calendar():
    prices, seasoned = _participation_panel()
    aware = prices.copy()
    aware.index = prices.index.tz_localize("America/New_York")
    seasoned_aware = seasoned.copy()
    seasoned_aware.index = aware.index
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            aware, seasoned_aware, window=3, min_members=1)
    assert "timezone-naive" in str(excinfo.value)


def test_breadth_functions_reject_a_seasoned_mask_that_does_not_line_up():
    prices, seasoned = _participation_panel()
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            prices, seasoned.iloc[:, :3], window=3, min_members=1)
    assert "columns" in str(excinfo.value)


def test_breadth_functions_reject_a_non_boolean_seasoned_mask():
    prices, seasoned = _participation_panel()
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            prices, seasoned.astype(float), window=3, min_members=1)
    assert "boolean" in str(excinfo.value)


def test_breadth_functions_reject_an_empty_panel():
    empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["A"], dtype=float)
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            empty, empty.astype(bool), window=3, min_members=1)
    assert "nonempty" in str(excinfo.value)


def test_breadth_functions_reject_duplicate_ticker_columns():
    prices = _panel([[10, 10], [20, 20], [30, 30]], ["A", "A"])
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            prices, _mask(prices), window=2, min_members=1)
    assert "unique" in str(excinfo.value)


def test_breadth_functions_reject_nonpositive_windows_and_member_floors():
    prices, seasoned = _participation_panel()
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            prices, seasoned, window=0, min_members=1)
    assert "window" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_participation(
            prices, seasoned, window=3, min_members=0)
    assert "min_members" in str(excinfo.value)


def test_oscillator_rejects_a_fast_span_that_is_not_shorter_than_the_slow_one():
    prices, seasoned = _advance_decline_panel()
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_net_advance_oscillator(
            prices, seasoned, fast=5, slow=5, min_members=1)
    assert "fast" in str(excinfo.value)


def test_cumulative_ad_trend_rejects_a_nonpositive_trend_window():
    prices, seasoned = _advance_decline_panel()
    with pytest.raises(ValueError) as excinfo:
        panel_signals.breadth_cumulative_ad_trend(
            prices, seasoned, trend_days=0, min_members=1)
    assert "trend_days" in str(excinfo.value)


def test_min_members_is_keyword_only_and_has_no_default():
    """A default would quietly tie this module to one membership basis;
    PANEL_MIN_RESOLVED_MEMBERS is derived from the sector_spdr_11 founding
    cohort and its own comment says it is not portable. The caller must state
    the floor its basis derives."""
    import inspect
    for name in (
        "breadth_participation",
        "breadth_net_advance_ratio",
        "breadth_net_advance_oscillator",
        "breadth_cumulative_ad_line",
        "breadth_cumulative_ad_trend",
    ):
        signature = inspect.signature(getattr(panel_signals, name))
        parameter = signature.parameters["min_members"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is inspect.Parameter.empty, name


def test_breadth_functions_do_not_mutate_their_inputs():
    prices, seasoned = _participation_panel()
    prices_before = prices.copy()
    seasoned_before = seasoned.copy()
    panel_signals.breadth_participation(
        prices, seasoned, window=3, min_members=1)
    panel_signals.breadth_net_advance_ratio(prices, seasoned, min_members=1)
    pd.testing.assert_frame_equal(prices, prices_before)
    pd.testing.assert_frame_equal(seasoned, seasoned_before)
