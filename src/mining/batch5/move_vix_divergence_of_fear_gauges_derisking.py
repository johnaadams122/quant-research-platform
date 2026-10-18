"""Batch5 candidate: move_vix_divergence_of_fear_gauges_derisking
(rotation_two_asset).

PRE-REGISTERED RULE (every constant fixed a priori -- the 10-observation
smoothing and the +0.5 threshold chosen for switch-count and time-in-market
feasibility, no returns examined -- and never to be tuned after a backtest
result)::

    common     = sessions on which BOTH ^MOVE and ^VIX print
    for each index x on common, for observation t with >= 756 observations
    before it:
        z_t    = (x_t - mean(x before t)) / std(x before t, ddof=1)
    DFG_t      = (zMOVE_t - zVIX_t) / sqrt(2)
    S_t        = mean of the last 10 DFG observations ending at t
    at each scheduled weekly decision date F (the Friday close, or the last
    scheduled session of the week when Friday is a scheduled holiday):
        derisk = S_F > 0.5                      (strict)
        target = SHY if derisk else SPY         (held for the following week)

Mechanism: when bond-market implied volatility (^MOVE) is high RELATIVE to
equity-market implied volatility (^VIX) investors reallocate from bonds
into stocks, which puts price pressure on stocks now and compresses the
subsequent equity risk premium; a high divergence therefore predicts lower
equity returns. Citation (Ruan and Wei, "Divergence of Fear Gauges and Stock
Returns", SSRN 5001821, sample 2004-2022; published as "When fears diverge:
Bond-equity relative uncertainty and stock market returns", ScienceDirect PII
S1057521926002188) was verified ONLY through search-result summaries (the
abstract pages were not accessible). UNVERIFIED: the published version's author
order and journal, the predictive horizon, the R-squared and the paper's
exact sign convention. The paper is a regression forecast, NOT a trading
rule: the weekly holding period, the 10-observation smoothing and the 0.5
threshold are this pipeline's own pre-registered constants, not literature
values, and the paper's whole 2004-2022 sample lies inside this test window.

Conventions, each verified from source or by test:

* Timing. ``mining.rotation_engine.run_two_asset_rotation`` hands the target
  factory frames truncated through decision bar t and fills its answer at bar
  t+1's Open. A reading dated at decision date F is therefore held from the
  FIRST session after F, at that session's Open.
* Decision dates are an EX-ANTE schedule, never derived from data:
  ``scheduled_decision_dates`` gives one date per ISO week, the Friday or,
  when the exchange's published holiday rules close that Friday (Good Friday;
  New Year's Day, Independence Day, Christmas and, from 2022, Juneteenth when
  they fall on or are observed on a Friday), the last scheduled session
  before it. New Year's Day on a Saturday is NOT observed on the Friday (the
  exchange stays open on that year-end Friday). Unscheduled closures
  (mourning days, storms) are unknowable in advance and are deliberately not
  modelled; none fell on a Friday after 2004. Verified against the
  exchange_calendars XNYS calendar for every week 2005-2027 and against a
  hand-listed table of every closure 2010-2016 (both in this module's tests).
* The signal lives on the two indices' OWN common calendar, intersected with
  no traded calendar, so a restricted evaluation bucket, a different traded
  calendar or a later data vendor cannot change it. Both indices are
  standardized over that same sample (which begins at the ^MOVE inception,
  2002-11-12); a date only one of them prints (real ^VIX prints on a couple
  of US holidays; real ^MOVE has holes and no bond-market-only-holiday
  prints) is in neither statistic.
* The z-score window ends at t-1, never at t, and needs at least 756
  observations before t (Welford running update, exact for a constant
  series; ddof 1). A constant prior history has no defined z (NaN, never
  inf). The 10-observation mean is a direct numpy mean per window (not a
  running mean), over the last 10 OBSERVATIONS of the common calendar, so a
  hole makes the window reach one session further back.
* Sign: DFG = (zMOVE - zVIX) / sqrt(2), so a HIGH value means bond-market
  fear relative to equity-market fear and is the de-risk side, matching the
  paper's negative predictive sign. For two standardized, positively
  correlated series this is exactly their second principal component (the
  paper's construction).
* Threshold: de-risk only when S_F is STRICTLY above 0.5 (plain float
  comparison; NaN is never de-risk).
* A decision date on which either index has no print has NO reading -- never
  an earlier day picked with hindsight. The previous regime is carried
  through that week (one decision skipped), exactly the sibling families'
  rule for a missing reading; ``make_target_fn(...,
  require_complete_decision_weeks=True)`` raises instead, naming the dates
  (neither driver passes it). On real data a small number of scheduled
  decision dates have no reading (Veterans Day, a bond-market-only holiday,
  is one recurring cause).
* Before the first defined reading (until the first Friday reading,
  2005-12-16) the account holds the DEFAULT, SPY -- the rule's "otherwise hold
  SPY" -- instead of sitting in cash for the roughly three warm-up years,
  which against the SPY benchmark ABI-1 always uses would be a large
  artificial drag with nothing to do with the signal. The default is a state
  entry dated before any data exists, so it needs no data and can leak
  nothing.
* The state series is computed ONCE from the full frames at build time, from
  data at or before each decision date only, and looked up as-of each bar by
  ``signals.make_external_regime_target_fn`` (the sibling rotation families'
  shared mechanism), so (a) no decision can see a later observation of any
  kind (tested by rebuilding the target from frames cut exactly at
  regime-changing decision dates and the sessions after them, and by
  injecting one-session and three-session look-aheads and a full-sample
  standardization and proving the same test FAILS on them), (b) a restricted
  bucket that starts mid-week, and whose signal slice is far shorter than 756
  observations, gets the carried-in regime from its first bar (verified
  through the real engine and the real per-trade bucket driver).
* ^MOVE and ^VIX are signal-only frames: never traded, read only through
  Close, with no positivity gate and no floor (the ABI-1 driver loads a
  symbol no plan in the run trades with ``require_positive=False``, and the
  rotation engine exempts signal frames).
* The ABI-1 driver intersects the two signal frames onto the traded calendar
  before the engine runs, so a session where ^MOVE has no print vanishes from
  the ACCOUNT. The signal itself is unaffected (it is built from the raw
  frames).

DISCLOSURES (read before believing any score):

1. Very low statistical power. A slow weekly regime switch produces few
   transitions, and those collapse into a handful of independent stress
   clusters, so the effective sample is far smaller than the switch count
   suggests, and the paper's
   sample lies wholly inside this window.
2. Not a crisis hedge by construction. The divergence measures bond-market
   fear RELATIVE to equity-market fear, so an equity panic that lifts both
   gauges together need not trigger it, while a long bond-volatility regime
   can hold the account in SHY through a rising equity market that the SPY
   benchmark ABI-1 uses keeps. The family is a bet on relative-uncertainty
   regimes, not on crisis protection.
3. Overlap with the other implied-volatility de-risk families
   (``vxn_vix_spread_derisking_regime``, ``vvix_vix_derisking_regime``,
   ``skew_index_tail_risk_regime``): ABI-1 scores every family at the fixed
   ``constants.ABI1_N_TRIALS`` / ``DAILY_SR_VARIANCE`` with no pairwise
   correlation correction, so that overlap is DISCLOSED, not corrected. Two
   sibling variants of the same idea (a trailing-252-session standardization,
   and a leverage overlay) are treated as duplicates and deliberately NOT
   built as separate families.
4. Cash drag is small (the default state is SPY) but TAX is on top: the catalog lists ``taxable`` first, so ABI-1
   applies realization tax to the account and never to the SPY benchmark, and
   the final open position is taxed as if closed at the end of the data.
5. Real-data caveat: the vendor can lag on the most recent session for
   ^MOVE, ^VIX or SHY. ``abi1_driver._align_rotation_frames``
   refuses two traded legs whose calendars differ, so a real ABI-1 run
   against data that still has that SHY hole fails closed for this family
   (and for any other SPY/SHY rotation); refetch after the vendor fills it.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from dateutil.easter import easter

from mining import signals

FAMILY_ID = "move_vix_divergence_of_fear_gauges_derisking"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "SHY"]          # [risk / default, defensive]
SIGNAL = ["^MOVE", "^VIX"]       # signal-only volatility indices, never traded
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {
    "bond_vol_ticker": "^MOVE",
    "equity_vol_ticker": "^VIX",
    "zscore_window": "expanding_ending_t_minus_1",
    "zscore_min_history_sessions": 756,
    "smoothing_sessions": 10,
    "derisk_threshold": 0.5,
    "defensive_proxy": "SHY",
    "rebalance": "weekly_friday_close",
}
_MIN_HISTORY = int(PARAMETERS["zscore_min_history_sessions"])
_SMOOTHING_SESSIONS = int(PARAMETERS["smoothing_sessions"])
_DERISK_THRESHOLD = float(PARAMETERS["derisk_threshold"])
# Conservative floor on the expected segment count for a slow weekly regime
# switch (one segment per regime transition plus the default SPY entry).
# Reported alongside a run's results as a sanity reference; it gates
# nothing. Same convention as every sibling module.
EXPECTED_MIN_TRADES = 30


# ------------------------------------------------------ ex-ante calendar

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """The n-th (1-based) given weekday (Monday=0) of a month."""
    first = pd.Timestamp(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    while day.weekday() != weekday:
        day -= pd.Timedelta(days=1)
    return day


def _observed(holiday: pd.Timestamp) -> pd.Timestamp:
    """A fixed-date holiday observed the exchange's way: a Saturday holiday
    closes the preceding Friday, a Sunday holiday the following Monday."""
    if holiday.weekday() == 5:
        return holiday - pd.Timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + pd.Timedelta(days=1)
    return holiday


@functools.lru_cache(maxsize=None)
def _scheduled_closures(year: int) -> frozenset:
    closures = set()
    new_year = pd.Timestamp(year, 1, 1)
    # New Year's Day: a Sunday closes the Monday; a SATURDAY is NOT observed
    # on the preceding Friday (the exchange stays open on that year-end
    # Friday, the last business day of the accounting year).
    if new_year.weekday() != 5:
        closures.add(_observed(new_year))
    closures.add(_nth_weekday(year, 1, 0, 3))            # Martin Luther King
    closures.add(_nth_weekday(year, 2, 0, 3))            # Presidents' Day
    closures.add(pd.Timestamp(easter(year)) - pd.Timedelta(days=2))  # Good Friday
    closures.add(_last_weekday(year, 5, 0))              # Memorial Day
    if year >= 2022:                                     # first NYSE Juneteenth
        closures.add(_observed(pd.Timestamp(year, 6, 19)))
    closures.add(_observed(pd.Timestamp(year, 7, 4)))    # Independence Day
    closures.add(_nth_weekday(year, 9, 0, 1))            # Labor Day
    closures.add(_nth_weekday(year, 11, 3, 4))           # Thanksgiving
    closures.add(_observed(pd.Timestamp(year, 12, 25)))  # Christmas
    return frozenset(closures)


def is_scheduled_closure(day) -> bool:
    """True when ``day`` is a weekend or a full-day closure the exchange
    publishes IN ADVANCE (its holiday rules). Deliberately independent of any
    data file, and blind to unscheduled closures (mourning days, storms)."""
    day = pd.Timestamp(day)
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    day = day.normalize()
    if day.weekday() >= 5:
        return True
    return day in _scheduled_closures(day.year)


def scheduled_decision_date(day) -> pd.Timestamp:
    """The weekly decision date of the ISO week (Monday-Sunday) containing
    ``day``: that week's Friday, or, when Friday is a scheduled closure, the
    last scheduled session before it."""
    day = pd.Timestamp(day).normalize()
    monday = day - pd.Timedelta(days=day.weekday())
    candidate = monday + pd.Timedelta(days=4)
    while is_scheduled_closure(candidate):
        candidate -= pd.Timedelta(days=1)
        if candidate < monday:
            raise ValueError(f"no scheduled session in the week of {monday.date()}")
    return candidate


def scheduled_decision_dates(start, end) -> pd.DatetimeIndex:
    """Every weekly decision date in ``[start, end]`` (both inclusive): one
    per ISO week, none dated after ``end`` (that decision has not happened
    yet -- but a Thursday IS returned once known, when Friday is a scheduled
    closure)."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    monday = start - pd.Timedelta(days=start.weekday())
    dates = []
    while monday <= end:
        decision = scheduled_decision_date(monday)
        if start <= decision <= end:
            dates.append(decision)
        monday += pd.Timedelta(days=7)
    return pd.DatetimeIndex(dates)


# ------------------------------------------------------------- the signal

def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series of finite
    values. NO positivity requirement and no floor: a signal-only series is
    a decision input, never a traded price."""
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                      index=index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[np.isfinite(close.to_numpy())]


def expanding_zscore(close: pd.Series, *, min_history: int) -> pd.Series:
    """z_t = (x_t - mean(x before t)) / std(x before t, ddof=1).

    The statistics are taken over the observations STRICTLY BEFORE t (the
    window ends at t-1, never at t) and only once at least ``min_history`` of
    them exist; earlier positions, and any position whose prior standard
    deviation is zero, are NaN. Welford's running update (exact for a
    constant series, no cumulative-sum cancellation)."""
    if min_history < 2:
        raise ValueError("min_history must be at least 2 (a sample standard "
                         "deviation needs two observations)")
    values = np.asarray(close, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("expanding_zscore needs finite observations")
    out = np.full(len(values), np.nan)
    mean = 0.0
    m2 = 0.0
    for t, x in enumerate(values):
        if t >= min_history:
            std = (m2 / (t - 1)) ** 0.5
            if std > 0.0:
                out[t] = (x - mean) / std
        delta = x - mean
        mean += delta / (t + 1)
        m2 += delta * (x - mean)
    return pd.Series(out, index=close.index)


def divergence_series(move_close: pd.Series, vix_close: pd.Series, *,
                      min_history: int) -> pd.Series:
    """DFG_t = (zMOVE_t - zVIX_t) / sqrt(2) on the sessions where BOTH
    indices print. Both are standardized over that same common sample, so a
    date only one index has (a VIX print on a bond-market holiday, a MOVE
    hole) is in neither statistic and in no output. Positive = bond-market
    fear high relative to equity-market fear."""
    common = move_close.index.intersection(vix_close.index).sort_values()
    z_move = expanding_zscore(move_close.reindex(common), min_history=min_history)
    z_vix = expanding_zscore(vix_close.reindex(common), min_history=min_history)
    return (z_move - z_vix) / np.sqrt(2.0)


def smoothed_divergence(move_close: pd.Series, vix_close: pd.Series, *,
                        min_history: int, window: int) -> pd.Series:
    """Mean of the last ``window`` DFG observations ending at t (t's own
    included), undefined until all ``window`` of them are. A direct numpy mean
    per window -- not a running mean -- so no accumulated rounding can move a
    decision that sits on the threshold."""
    if window < 1:
        raise ValueError("window must be at least 1")
    dfg = divergence_series(move_close, vix_close, min_history=min_history)
    values = dfg.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1:] = windows.mean(axis=1)      # NaN if any obs is NaN
    return pd.Series(out, index=dfg.index)


def is_derisk(reading, *, threshold: float = _DERISK_THRESHOLD):
    """The pre-registered band: de-risk when the smoothed reading is STRICTLY
    above the threshold, plain float comparison (NaN is never de-risk)."""
    return np.asarray(reading, dtype=float) > threshold


def derisk_state_series(move_close: pd.Series, vix_close: pd.Series, *,
                        min_history: int, window: int,
                        threshold: float) -> pd.Series:
    """1.0 (de-risk) / 0.0 (hold SPY) at every scheduled weekly decision
    date whose own reading is defined: the decision date must be present on
    the two indices' common calendar (a date either index skipped has NO
    reading -- never a fallback to an earlier day) and the smoothed
    divergence must be warm there."""
    smoothed = smoothed_divergence(
        move_close, vix_close, min_history=min_history, window=window)
    if smoothed.empty:
        return pd.Series(dtype=float)
    dates = scheduled_decision_dates(smoothed.index[0], smoothed.index[-1])
    present = dates[dates.isin(smoothed.index)]
    readings = smoothed.reindex(present).dropna()
    flags = is_derisk(readings.to_numpy(), threshold=threshold)
    return pd.Series(flags.astype(float), index=readings.index)


def weeks_without_a_reading(state: pd.Series,
                            common: pd.DatetimeIndex) -> list:
    """Scheduled decision dates from the first reading onward, up to the last
    date of ``common``, that have no reading. The warm-up before the first
    reading and the week still in progress (its decision date is after the
    data's last date) are never counted."""
    if state.empty or len(common) == 0:
        return []
    dates = scheduled_decision_dates(state.index.min(), common.max())
    return [d for d in dates if d not in state.index]


# The regime in force before the first defined reading is the DEFAULT holding
# ("otherwise hold SPY"): a state entry dated before any data can exist.
_DEFAULT_REGIME_DATE = pd.Timestamp("1900-01-01")


def make_target_fn(frames, *, require_complete_decision_weeks: bool = False):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history frames for SPY, SHY, ^MOVE and ^VIX, handed once, UNALIGNED:
    the state is computed here, once, on the two indices' own common
    calendar, from data at or before each decision date only, then looked up
    as-of the current bar by ``signals.make_external_regime_target_fn``.

    ``require_complete_decision_weeks`` (default False; neither driver passes
    it): when True the factory RAISES, naming the first few dates, if any
    scheduled decision date after the first reading has no reading (an index
    did not print that day). The default carries the previous regime through
    such a week, one decision skipped -- real ^MOVE has such holes."""
    for symbol in (*TRADED, *SIGNAL):
        _require_frame(frames, symbol)
    move_close = _clean_close(frames[SIGNAL[0]])
    vix_close = _clean_close(frames[SIGNAL[1]])
    state = derisk_state_series(
        move_close, vix_close, min_history=_MIN_HISTORY,
        window=_SMOOTHING_SESSIONS, threshold=_DERISK_THRESHOLD)
    if state.empty:
        raise ValueError(
            f"{FAMILY_ID}: no reading -- no scheduled weekly decision date has "
            f"a defined smoothed divergence (needs at least {_MIN_HISTORY} "
            f"prior observations plus {_SMOOTHING_SESSIONS - 1} more, on "
            "sessions where both indices print, and a non-constant series); "
            "a family that could never trade must not be scored")
    if require_complete_decision_weeks:
        common = move_close.index.intersection(vix_close.index)
        missing = weeks_without_a_reading(state, common)
        if missing:
            shown = ", ".join(d.strftime("%Y-%m-%d") for d in missing[:6])
            raise ValueError(
                f"{FAMILY_ID}: {len(missing)} scheduled decision date(s) after "
                f"the first reading have no reading (first: {shown}); refetch "
                "the data, or pass require_complete_decision_weeks=False to "
                "carry the previous regime through them")
    risk_on = pd.concat([
        pd.Series([True], index=pd.DatetimeIndex([_DEFAULT_REGIME_DATE])),
        pd.Series(state.to_numpy() == 0.0, index=state.index)])
    return signals.make_external_regime_target_fn(
        risk_on, risk_ticker=TRADED[0], defensive_ticker=TRADED[1])
