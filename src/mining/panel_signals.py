"""Breadth indicators reduced from a cross-sectional price panel.

Built for breadth candidates that need a panel capability: a cumulative
advance-decline line, a McClellan-style oscillator (the difference of two
EWMAs of net advances), and percent-of-members-above-their-own moving average.

WHY THIS MODULE IS A PURE REDUCTION. A breadth statistic is one number per
session computed from many tickers; it is not a new kind of backtest. So the
panel is reduced OFFLINE to a dated scalar series and the strategy still trades
a single ticker through the untouched engine. Everything here therefore takes
plain pandas frames and returns a plain pandas Series: no network, no
filesystem, no wall clock, no dependency on any panel container type. The
caller assembles the frames; this module only reduces them.

THE INPUT CONTRACT, stated once and validated on every call:

  prices    float DataFrame, index = the reference session calendar
            (DatetimeIndex, tz-naive, sorted, unique), columns = unique
            tickers. NaN means NO BAR. It never means zero, and anything that
            is not a FINITE POSITIVE number is refused outright -- pooling.py's
            `fill_value=0.0` union is right for summing P&L and catastrophic
            on prices, where a 0.00 close is a -100% return and an automatic
            "below its moving average", while an inf close reads as an advance
            on its own session and a decline on the next.
  seasoned  bool DataFrame with the IDENTICAL index and columns. True iff the
            ticker is a member of the basis on that session AND has already
            accumulated enough of its own bars for the indicator to be
            defined. Deriving that mask is the caller's job precisely because
            it must come from the hand-dated membership file plus a cumulative
            count of that ticker's own resolved bars -- never from a rolling
            statistic over the prices, which would let a fetch gap masquerade
            as a real market fact by quietly dropping the member from the
            denominator.

WHY NaN IS NOT ZERO, AND WHY THAT SPLITS THE INDICATORS IN TWO. If a seasoned
member has no bar on a session, that session's breadth value is genuinely
unknown -- a ratio with an unknown member is unknown, not approximate. A
POINT-IN-TIME indicator may simply yield NaN there and carry on. A
PATH-DEPENDENT one may not, because pandas silently bridges the gap:
`pd.Series([1.0, nan, 1.0, 1.0]).cumsum()` is `[1.0, nan, 2.0, 3.0]` -- the
running total after the gap is exactly what it would have been had the gap
contributed 0.0 -- and `.ewm(adjust=False).mean()` holds its prior level
straight through (both observed on pandas 2.2.3, and pinned by
test_pandas_cumsum_and_ewm_silently_bridge_a_gap). So every accumulating
indicator here REFUSES, naming the offending sessions, rather than tolerating
a fraction of them.

THE ESCAPE FROM THAT REFUSAL IS A DATE, NOT A TOLERANCE.
A hole at the end of a panel can be dodged by building to an earlier date; a
hole in the middle cannot, and refusing with no way out would leave a
cumulative line and an oscillator uncomputable on the very basis they were
written for -- nine founding sector funds against PANEL_MIN_RESOLVED_MEMBERS
= 9 leaves no slack before XLRE joins in 2015. So the path-dependent
indicators take an optional `span_start`: the caller NAMES the session the
accumulation restarts at, that date goes into the snapshot's provenance
beside the broken dates it answers, and the hole rule is untouched at and
above it. Unset remains the default and remains a refusal.

NO LOOK-AHEAD. Every computation is trailing-only, so an indicator's value at
session T is identical whether or not sessions after T exist in the frame it
was handed. That is asserted directly, by computing on a truncated frame and
on the full one and comparing, rather than argued in prose.

NO min_members DEFAULT. PANEL_MIN_RESOLVED_MEMBERS is derived from the
sector_spdr_11 basis's own founding cohort and its own comment in constants.py
says it is not portable to another basis. A default here would quietly weld
this module to one membership file, so the floor is a required keyword and the
caller states the number its basis derives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_MAX_NAMED_SESSIONS = 10


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def _validate(fn_name: str, prices: pd.DataFrame,
              seasoned: pd.DataFrame, *, min_members: int) -> None:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError(f"{fn_name}: prices must be a nonempty DataFrame of "
                         "session closes")
    if not isinstance(seasoned, pd.DataFrame):
        raise ValueError(f"{fn_name}: seasoned must be a boolean DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(f"{fn_name}: prices must use a DatetimeIndex session "
                         "calendar")
    if prices.index.tz is not None:
        raise ValueError(f"{fn_name}: the session calendar must be "
                         "timezone-naive; the mining pipeline normalizes "
                         "market calendars before any reduction")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError(f"{fn_name}: the session calendar must be sorted "
                         "unique")
    if prices.columns.has_duplicates:
        raise ValueError(f"{fn_name}: prices columns (tickers) must be unique")
    if not seasoned.index.equals(prices.index):
        raise ValueError(f"{fn_name}: seasoned must share the identical "
                         "session calendar as prices")
    if not seasoned.columns.equals(prices.columns):
        raise ValueError(f"{fn_name}: seasoned must carry the identical "
                         "columns (tickers) as prices, in the same order")
    if not (seasoned.dtypes == bool).all():
        raise ValueError(f"{fn_name}: seasoned must be a boolean mask; a "
                         "numeric mask would make an ambiguous 0.0/NaN "
                         "membership claim")

    try:
        values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fn_name}: prices must be numeric") from exc
    # FINITENESS is half of this check, not a flourish. `+inf > 0.0` is True,
    # so a narrower `~isnan & ~(v > 0.0)` form would admit an inf print
    # outright: on a fully-seasoned two-member panel whose second column
    # carried one inf, breadth_net_advance_ratio would return a clean-looking
    # series and breadth_cumulative_ad_line would integrate it, with no raise
    # anywhere. The module's whole reason for
    # validating prices at all is that it does not trust its caller, and
    # "is this a real price" has two failure directions, not one.
    bad = (~np.isnan(values)) & ~(np.isfinite(values) & (values > 0.0))
    if bad.any():
        row, column = (int(i) for i in np.argwhere(bad)[0])
        raise ValueError(
            f"{fn_name}: prices must be a finite positive number where a bar "
            f"exists; {prices.columns[column]!r} is {values[row, column]!r} "
            f"on {prices.index[row].date()}. A missing bar is NaN, never 0.0 "
            "and never an infinity -- a 0.00 close reads as a -100% return "
            "and an automatic 'below its moving average', and an inf close "
            "reads as an advance on its own session and a decline on the "
            "next")

    if not isinstance(min_members, (int, np.integer)) or isinstance(min_members, bool):
        raise ValueError(f"{fn_name}: min_members must be an int")
    if min_members < 1:
        raise ValueError(f"{fn_name}: min_members must be >= 1, got "
                         f"{min_members!r}")


def _validate_positive_int(fn_name: str, field: str, value) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"{fn_name}: {field} must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{fn_name}: {field} must be >= 1, got {value!r}")
    return int(value)


def _validate_span_start(fn_name: str, value):
    """A stated accumulation floor is a DECISION, so it is typed strictly."""
    if value is None:
        return None
    if not isinstance(value, pd.Timestamp) or isinstance(value, bool):
        raise ValueError(
            f"{fn_name}: span_start must be a pandas Timestamp naming the "
            f"session the accumulation restarts at, got {value!r}. A string "
            "or an integer would slice the calendar by coincidence rather "
            "than by contract")
    if value.tz is not None:
        raise ValueError(
            f"{fn_name}: span_start must be timezone-naive, matching the "
            "session calendar this module already requires")
    return value


def _name_sessions(index: pd.DatetimeIndex) -> str:
    named = ", ".join(str(stamp.date()) for stamp in index[:_MAX_NAMED_SESSIONS])
    if len(index) > _MAX_NAMED_SESSIONS:
        named += f", ... ({len(index)} sessions in total)"
    return named


# --------------------------------------------------------------------------
# panel-shaped helpers -- public because the snapshot builder and the coverage
# report both need them, and re-deriving them at the call site is how the two
# would drift apart.
# --------------------------------------------------------------------------

def resolved_mask(prices: pd.DataFrame, seasoned: pd.DataFrame) -> pd.DataFrame:
    """Members that BOTH count on a session and actually printed a bar on it."""
    _validate("resolved_mask", prices, seasoned, min_members=1)
    return seasoned & prices.notna()


def resolved_member_count(prices: pd.DataFrame,
                          seasoned: pd.DataFrame) -> pd.Series:
    """Per-session denominator: how many seasoned members resolved."""
    _validate("resolved_member_count", prices, seasoned, min_members=1)
    return (seasoned & prices.notna()).sum(axis=1).astype("int64")


def broken_sessions(prices: pd.DataFrame,
                    seasoned: pd.DataFrame) -> pd.Series:
    """True on any session where a SEASONED member has no bar.

    Such a session's breadth value is unknown, not approximate. A ticker that
    is not yet a seasoned member has no bar to miss, so its absence never
    breaks a session -- that asymmetry is the whole reason `seasoned` is
    derived from the membership file rather than from the price data.

    Exposed publicly so the snapshot builder can measure the broken fraction
    against PANEL_MAX_BROKEN_SESSION_FRACTION and name every broken date in
    the snapshot's provenance. This module deliberately does not enforce that
    ceiling itself: it is a whole-snapshot policy, not a per-reduction one.
    """
    _validate("broken_sessions", prices, seasoned, min_members=1)
    return (seasoned & prices.isna()).any(axis=1)


def net_advance_counts(prices: pd.DataFrame,
                       seasoned: pd.DataFrame) -> pd.DataFrame:
    """Per-session advancers / decliners / eligible members.

    A member is ELIGIBLE for a session's advance-decline count only if it
    resolved on that session AND on the immediately preceding one -- otherwise
    the "change" spans a gap and is not a one-session move. Unchanged members
    stay in the denominator and count as neither, the standard convention.
    The first session of any panel has no predecessor, so its eligible count
    is zero by construction.
    """
    _validate("net_advance_counts", prices, seasoned, min_members=1)
    resolved = seasoned & prices.notna()
    previous = resolved.shift(1, fill_value=False)
    eligible = resolved & previous
    change = prices.diff()
    return pd.DataFrame(
        {
            "advancers": ((change > 0.0) & eligible).sum(axis=1).astype("int64"),
            "decliners": ((change < 0.0) & eligible).sum(axis=1).astype("int64"),
            "eligible": eligible.sum(axis=1).astype("int64"),
        },
        index=prices.index,
    )


# --------------------------------------------------------------------------
# point-in-time indicator: participation above a trailing moving average
# --------------------------------------------------------------------------

def _trailing_mean_over_own_bars(prices: pd.DataFrame,
                                 window: int) -> pd.DataFrame:
    """Each ticker's trailing `window`-BAR mean, over that ticker's OWN bars.

    Deliberately NOT `prices.rolling(window).mean()` over the session frame.
    A frame-level rolling mean returns NaN for `window - 1` sessions after any
    gap in that ticker's column, which would silently demote a perfectly
    healthy member to "below its average" for months after a single missing
    print. Averaging over the ticker's own resolved bars also matches the
    seasoning rule the mask is built from -- "cumulative count of that
    ticker's own resolved bars >= window" -- so a cell that the mask calls
    seasoned always has a defined mean here, and `breadth_participation`
    refuses if that ever turns out not to hold.
    """
    columns = {}
    for ticker in prices.columns:
        column = prices[ticker]
        own = column.dropna()
        columns[ticker] = own.rolling(window, min_periods=window).mean(
            ).reindex(column.index)
    return pd.DataFrame(columns, index=prices.index, columns=prices.columns)


def breadth_participation(prices: pd.DataFrame, seasoned: pd.DataFrame, *,
                          window: int, min_members: int) -> pd.Series:
    """Fraction of resolved seasoned members trading above their own trailing
    `window`-bar mean.

    Point-in-time: a session whose value is unknown -- because a seasoned
    member had no bar, or because too few members resolved -- yields NaN and
    is dropped by the caller rather than estimated. `>` is strict, so a price
    exactly on its own mean is not a participant.
    """
    _validate("breadth_participation", prices, seasoned,
              min_members=min_members)
    window = _validate_positive_int("breadth_participation", "window", window)

    resolved = seasoned & prices.notna()
    averages = _trailing_mean_over_own_bars(prices, window)

    undefined = resolved & averages.isna()
    if undefined.to_numpy().any():
        row, column = (int(i) for i in np.argwhere(undefined.to_numpy())[0])
        raise ValueError(
            f"breadth_participation: the seasoned mask claims "
            f"{prices.columns[column]!r} is seasoned on "
            f"{prices.index[row].date()}, but that ticker has fewer than "
            f"window={window} bars of its own by then, so its trailing mean "
            "is undefined. Refusing rather than scoring it 'below'")

    above = (prices > averages) & resolved
    denominator = resolved.sum(axis=1)
    unknown = ((seasoned & prices.isna()).any(axis=1)
               | (denominator < min_members))
    fraction = above.sum(axis=1) / denominator.where(denominator > 0)
    return fraction.where(~unknown).astype("float64")


# --------------------------------------------------------------------------
# point-in-time indicator: net advance ratio
# --------------------------------------------------------------------------

def _net_advance_ratio(fn_name: str, prices: pd.DataFrame,
                       seasoned: pd.DataFrame, *,
                       min_members: int) -> pd.Series:
    counts = net_advance_counts(prices, seasoned)
    eligible = counts["eligible"]
    broken = (seasoned & prices.isna()).any(axis=1)
    # A gap at session t also destroys t+1's advance-decline reading, because
    # the change into t+1 would span the gap. Both sessions are unknown.
    broken_either_side = broken | broken.shift(1, fill_value=True)
    unknown = broken_either_side | (eligible < min_members)
    ratio = ((counts["advancers"] - counts["decliners"])
             / eligible.where(eligible > 0))
    return ratio.where(~unknown).astype("float64")


def breadth_net_advance_ratio(prices: pd.DataFrame, seasoned: pd.DataFrame, *,
                              min_members: int) -> pd.Series:
    """(advancers - decliners) / eligible members, per session.

    The RATIO form is mandatory rather than stylistic on any panel whose
    membership grows: the denominator genuinely moves as funds are added, and
    a raw net-advance COUNT would carry a permanent level shift across every
    membership change, which an accumulating indicator would then integrate
    into a trend that is an artefact of the basis rather than of the market.
    """
    _validate("breadth_net_advance_ratio", prices, seasoned,
              min_members=min_members)
    return _net_advance_ratio("breadth_net_advance_ratio", prices, seasoned,
                              min_members=min_members)


# --------------------------------------------------------------------------
# path-dependent indicators
# --------------------------------------------------------------------------

def _accumulation_span(fn_name: str, series: pd.Series,
                       span_start: pd.Timestamp | None) -> pd.Series:
    """The contiguous tail of `series` that may be accumulated over.

    Leading unknown sessions -- the panel's first session has no predecessor,
    and early members are not yet seasoned -- precede the accumulation and are
    simply not part of it. A hole INSIDE the span is a refusal, because pandas
    `.cumsum()` and `.ewm()` skip NaN and would bridge it silently, leaving
    every subsequent value resting on a session nobody measured.

    `span_start` is the caller's escape from that refusal, because the
    refusal alone can be very costly. A hole
    at the END of a panel can be dodged by building to an earlier date, but a
    hole in the MIDDLE cannot: truncating before it discards the entire tail.
    On the real sector_spdr_11 basis that is not hypothetical -- nine founding
    funds against PANEL_MIN_RESOLVED_MEMBERS = 9 leaves the pre-2015 era with
    zero slack, so one missing print anywhere in sixteen years would void a
    cumulative line and an oscillator outright.

    It is an escape, not a tolerance, and the distinction is the whole design:

      * The caller NAMES the session the accumulation restarts at, so the
        decision is a date the snapshot's provenance records verbatim next to
        the broken dates it is answering. Restarting automatically would be
        an unrecorded decision changing every downstream number.
      * The hole rule is unchanged at and after that floor. A `span_start`
        with a hole still inside its span is refused exactly as an unset one
        is, so the keyword can never buy the silent bridging this module
        exists to prevent.
      * Unset stays the default and stays fail-closed.
    """
    span_start = _validate_span_start(fn_name, span_start)
    region = series if span_start is None else series.loc[span_start:]
    known = region.notna()
    if not known.to_numpy().any():
        if span_start is not None:
            raise ValueError(
                f"{fn_name}: no known session at or after span_start="
                f"{span_start.date()}, so there is nothing to accumulate "
                "over. Name a session inside the panel's own calendar, or "
                "leave span_start unset to accumulate from the panel's first "
                "known session")
        raise ValueError(
            f"{fn_name}: no known session in the whole panel -- there is "
            "nothing to accumulate over. Check the seasoned mask and the "
            "min_members floor before rebuilding")
    span = region.loc[region.index[known.to_numpy().argmax()]:]
    holes = span.index[span.isna()]
    if len(holes) > 0:
        remedy = (
            "Re-fetch the missing bars, build to a date before the first "
            f"gap, or restart the accumulation after it by passing "
            f"span_start={holes[-1].date()!s} or later -- a stated restart "
            "is recorded in the snapshot's provenance, a bridged gap is not"
            if span_start is None else
            "Re-fetch the missing bars, or move span_start past the last of "
            "them -- moving the floor cannot excuse a hole above it")
        raise ValueError(
            f"{fn_name}: refusing to accumulate across unknown sessions "
            f"inside the span {span.index[0].date()}..{span.index[-1].date()}: "
            f"{_name_sessions(holes)}. pandas .cumsum()/.ewm() skip NaN, so "
            "accumulating anyway would bridge these gaps invisibly and every "
            f"later value would rest on a session nobody measured. {remedy}")
    return span


def breadth_net_advance_oscillator(prices: pd.DataFrame,
                                   seasoned: pd.DataFrame, *,
                                   fast: int, slow: int,
                                   min_members: int,
                                   span_start: pd.Timestamp | None = None,
                                   ) -> pd.Series:
    """Difference of a fast and a slow EWMA of the net advance ratio.

    A McClellan-STYLE oscillator: same construction, computed over whatever
    membership basis the caller supplies, which on a small fund basis is a
    coarser analogue of the indicator its literature constructs over a full
    index. Both legs use `ewm(span=..., adjust=False)` -- the recursive form,
    `e_t = a * x_t + (1 - a) * e_{t-1}` with `a = 2 / (span + 1)`, seeded at
    the span's first value -- so the series is reproducible by hand and does
    not depend on how much history precedes the window.

    Path-dependent: raises if any session inside the accumulated span is
    unknown, unless the caller states a `span_start` floor above the last of
    them (see `_accumulation_span`).
    """
    _validate("breadth_net_advance_oscillator", prices, seasoned,
              min_members=min_members)
    fast = _validate_positive_int("breadth_net_advance_oscillator", "fast", fast)
    slow = _validate_positive_int("breadth_net_advance_oscillator", "slow", slow)
    if fast >= slow:
        raise ValueError(
            "breadth_net_advance_oscillator: fast must be strictly shorter "
            f"than slow, got fast={fast}, slow={slow}. Equal spans give an "
            "identically-zero oscillator and a crossed pair inverts its sign")

    ratio = _net_advance_ratio("breadth_net_advance_oscillator", prices,
                               seasoned, min_members=min_members)
    span = _accumulation_span("breadth_net_advance_oscillator", ratio,
                              span_start)
    oscillator = (span.ewm(span=fast, adjust=False).mean()
                  - span.ewm(span=slow, adjust=False).mean())
    return oscillator.reindex(prices.index).astype("float64")


def breadth_cumulative_ad_line(prices: pd.DataFrame, seasoned: pd.DataFrame, *,
                               min_members: int,
                               span_start: pd.Timestamp | None = None,
                               ) -> pd.Series:
    """Running total of the net advance ratio -- the advance-decline line.

    Accumulated over the RATIO rather than the raw net count for the reason
    given on `breadth_net_advance_ratio`: a raw count integrates every
    membership change into a permanent drift of the line itself.

    Path-dependent: raises if any session inside the accumulated span is
    unknown, unless the caller states a `span_start` floor above the last of
    them (see `_accumulation_span`).
    """
    _validate("breadth_cumulative_ad_line", prices, seasoned,
              min_members=min_members)
    ratio = _net_advance_ratio("breadth_cumulative_ad_line", prices, seasoned,
                               min_members=min_members)
    span = _accumulation_span("breadth_cumulative_ad_line", ratio, span_start)
    return span.cumsum().reindex(prices.index).astype("float64")


def breadth_cumulative_ad_trend(prices: pd.DataFrame, seasoned: pd.DataFrame,
                                *, trend_days: int,
                                min_members: int,
                                span_start: pd.Timestamp | None = None,
                                ) -> pd.Series:
    """The advance-decline line less its own trailing `trend_days` mean.

    The de-trended form is what a regime filter can threshold at zero: the
    raw line's level depends on where the panel's history happens to start,
    while its distance from its own trailing mean does not.

    Path-dependent: raises if any session inside the accumulated span is
    unknown, unless the caller states a `span_start` floor above the last of
    them (see `_accumulation_span`). Sessions before the trailing window has
    filled yield NaN -- that is "not defined yet", never a bridged gap, and a
    stated `span_start` restarts that window too rather than reaching back
    across the hole for the observations it is short of.
    """
    _validate("breadth_cumulative_ad_trend", prices, seasoned,
              min_members=min_members)
    trend_days = _validate_positive_int(
        "breadth_cumulative_ad_trend", "trend_days", trend_days)

    ratio = _net_advance_ratio("breadth_cumulative_ad_trend", prices, seasoned,
                               min_members=min_members)
    span = _accumulation_span("breadth_cumulative_ad_trend", ratio, span_start)
    line = span.cumsum()
    trend = line - line.rolling(trend_days, min_periods=trend_days).mean()
    return trend.reindex(prices.index).astype("float64")
