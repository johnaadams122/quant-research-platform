"""Batch5 candidate: cross_asset_financial_turbulence_derisking
(rotation_two_asset).

PRE-REGISTERED RULE (every constant fixed a priori in the catalog entry,
chosen without examining any return, Sharpe, drawdown or other performance
number of this strategy, and never to be tuned after a backtest result)::

    panel   = SPY, EEM, TLT, GLD, LQD      # stocks, emerging markets, long
                                           # Treasuries, gold, credit
    r_t     = daily log returns of the five, on sessions where ALL five have
              a genuine one-session return
    d_t     = (r_t - mean)' inv(cov) (r_t - mean)      # Mahalanobis turbulence
              mean, cov = EXPANDING sample mean and covariance (ddof = 1) over
              the sessions strictly BEFORE t; d_t needs 500 earlier sessions
    m_t     = trailing 10-session mean of d_t, t included
    at each SCHEDULED last session of the week w (Friday, or Thursday when
    the exchange calendar shuts that Friday):
        pct = share of the STRICTLY EARLIER m values that are STRICTLY BELOW
              m_w (needs 250 earlier m values)
        target = SHY if pct > 0.80 else SPY
    the rotation engine fills the target at the NEXT session open.

Mechanism (catalog entry): financial turbulence is the multivariate distance
of one day's returns across asset classes from their historical mean and
covariance. It flags days when prices move unusually far AND days when assets
break their usual correlations (stocks and bonds falling together, gold
failing to hedge), which a volatility reading on one asset misses. Source:
Kritzman, M. and Li, Y. (2010), "Skulls, Financial Turbulence, and Risk
Management", Financial Analysts Journal 66(5), 30-41. Citation status (as
recorded in the catalog entry's "source" field, restated here): the paper's use
of turbulence as an exposure-scaling filter is confirmed; any claim about the
size of post-turbulence drawdowns or the paper's own filter threshold is
UNVERIFIED. The catalog entry also records that the paper publishes a
filtering concept and not a trading rule (its only scaling test being on a
currency carry strategy, per one reading of the PDF), so the SPY to
SHY use, the weekly Friday decision, the 10-session smoothing, the 0.80
percentile and the 500/250-session minimums are THIS family's own
pre-registered construction, not literature values.

Where each constant comes from (all fixed a priori):

* 500 sessions of covariance history, 250 earlier smoothed values for the
  percentile: the catalog entry's own stated minimums.
* 10-session smoothing and the 0.80 percentile: the catalog entry fixes them
  "from switch-count and time-in-market feasibility with no returns
  examined" (a rule that switches every few weeks or sits defensive most of
  the time is not a de-risk filter). Unchanged here.
* ddof = 1: the standard unbiased sample covariance (numpy and pandas
  default). A uniform rescaling of the covariance by n/(n-1) changes every d_t
  by a factor within 0.2 percent at n >= 500, so this cannot decide anything.
* Condition-number limit 1e8 (ten to the eighth power; stored as the integer
  exponent 8 in PARAMETERS so that no long run of digits, and no float that
  would re-serialize into one, ever appears in the catalog entry or the source):
  chosen for numerical reasons alone,
  roughly the reciprocal of the square root of double-precision machine
  epsilon (about 67 million), rounded up; beyond it a solve loses about half
  its significant digits. It is a guard, not a tuned value: a well-behaved
  covariance of liquid ETFs is expected to sit orders of magnitude below it.

Conventions, each verified by this module's test file:

* Timing. ``mining.rotation_engine.run_two_asset_rotation`` hands the target
  factory frames truncated through decision bar t and fills its answer at bar
  t+1's Open. A reading dated at a week end w is held from the first session
  of the next week, at that session's Open. d_t at a Friday uses the mean and
  covariance through Thursday and the Friday return itself (all known at the
  Friday close).
* Week end = the exchange's SCHEDULED last session of each Monday-to-Friday
  week (``scheduled_week_end``): the Friday, stepping back one day over a
  scheduled Friday closure -- Good Friday; a holiday that falls on a Friday;
  a holiday on a Saturday observed the Friday before (July 4, Christmas and,
  from 2022, Juneteenth), except New Year's Day, whose Saturday case leaves
  December 31 open. It is fixed in advance and read from no data file, so no
  later observation, not even the existence of a later session, can decide
  it. It is checked as a whole set against the installed exchange_calendars
  XNYS calendar for every week from November 2004 to the end of 2028, which
  must match exactly. Since 1993 only two unscheduled closures have moved a
  week's actual last session (the September 2001 shutdown and the June 2004
  day of mourning), both before the panel begins. A
  scheduled week end absent from the data has NO reading; the code never
  falls back to an earlier day, which would need foreknowledge of the
  closure.
* Panel sessions. Each asset's log return is taken against the PREVIOUS
  session of the joined calendar and a session is kept only if all five have
  one. A single missing print therefore drops that session and the next one
  (the next return would span two sessions for that asset only) instead of
  mixing a two-session return into the vector; a date missing from every
  asset is a closure and the return across it is a legitimate multi-day
  return. The 500 / 10 / 250 counts are counts of panel sessions.
* Fail closed, never regularize. If the expanding covariance is not positive
  definite or its condition number exceeds the limit, ``turbulence_series``
  raises ValueError naming the first failing session; no ridge, shrinkage,
  pseudo-inverse or pivot is ever applied, because a repaired matrix would
  score a different statistic from the one pre-registered. A non-positive
  close is refused for the same reason (a log return is undefined).
* Percentile rank. The share of strictly earlier smoothed values that are
  strictly below today's; ties are not below; the comparison is a strict
  "above 0.80", so a percentile of exactly 0.80 keeps SPY. "Strictly earlier"
  means every earlier session's smoothed value (overlapping windows
  included), read literally from the catalog's rule.
* The state is computed ONCE from the full frames at build time, entirely from
  data at or before each week end (Welford updates, ``[:k]`` prefixes), and
  handed to ``signals.make_external_regime_target_fn``, whose as-of lookup
  carries each reading through the week and into a restricted evaluation
  bucket that starts mid-regime (verified through the real engine on tail
  buckets, not only by comparing the factory with itself). Cutting every frame
  off 0 to 5 sessions after each state change reproduces the full build's
  readings on every date up to the cut, and injected one-session look-aheads
  (in the turbulence series, in the smoothed series, in the percentile, and a
  week end derived from the data) are each caught by that harness.
* Missing scheduled week end after the first reading: a complete week with no
  reading (a vendor hole, or an unscheduled closure landing on the decision
  Friday) makes ``make_target_fn`` RAISE by default, naming the week
  (``require_complete_weeks``), because the score this feeds is one-shot and
  permanent and a silently skipped decision must never reach it; the opt-out
  (False) carries the previous state through such a week. No week end has a
  reading at all -> the factory raises too.
* Before the first reading the account holds SPY (see disclosure 2).
* Every load-bearing constant is passed EXPLICITLY to the building blocks
  (none relies on a default) and ``PARAMETERS`` equals the catalog entry byte
  for byte.

DISCLOSURES (read before believing any score):

1. Low prior. The catalog entry places this in the realized-risk de-risk
   neighbourhood of simpler rules
   (single-asset volatility timers, VIX-term-structure and skew switches, a
   stock-bond correlation flip), with the same "avoids the crash early, lags
   the rebound after" profile: the distance reacts after a shock has hit, so
   it can sit in SHY through V-shaped rebounds. State changes
   cluster around a small number of independent stress episodes, and a
   small share of time in SHY leaves little room to earn the excess Sharpe
   the ABI-1 bar needs. The statistic (a five-asset covariance-aware
   distance) is distinct, so this is a cheap, low-prior probe, not a strong
   candidate.
2. Warm-up. The rule defines no reading before 2007-11-30, about three years
   after the panel starts on 2004-11-19 (until 500 + 9 + 250 panel sessions
   exist). ``PARAMETERS["warmup_holding"] = "risk_on_ticker"`` holds the
   DEFAULT holding, SPY, through it, so there the account equals the
   benchmark apart from one entry and no signal is being traded. The
   alternative reading, "no signal, no position" (pass
   ``warmup_holding="flat"``; the pipeline's usual convention for an external
   regime series), leaves the account idle on cash while SPY runs, a handicap
   of the construction rather than of the hypothesis. Changing it means
   changing ``PARAMETERS`` and the catalog entry together.
3. The first reading is 2007-11-30: that is the date the literal minimums
   (500 earlier sessions, 250 earlier smoothed values, both counted in panel
   sessions) give on this panel, and no constant is adjusted to move it.
4. The five-asset panel is a judgment call and any change to it is a new
   trial. SHY is the traded defensive leg and is not in the panel.
5. Only the ABI-1 driver is supported (rotation shape, ``TRADED`` SPY and SHY,
   ``SIGNAL`` the other four panel assets). The driver requires the two
   traded legs to share one identical calendar after trimming to their common
   inception; a vendor hole in one of them makes
   the driver refuse the whole family, which is a data-freshness problem
   outside this module.
6. ABI-1 taxes the account's realized gains (taxable is the first listed
   account type) and never the SPY benchmark; the long warm-up SPY segment is
   realized at the first switch, so a genuine but small edge can be eaten by
   it.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from dateutil.easter import easter

from mining import signals

FAMILY_ID = "cross_asset_financial_turbulence_derisking"
SHAPE = "rotation_two_asset"
TRADED = ["SPY", "SHY"]                 # [risk (default), defensive]
SIGNAL = ["EEM", "TLT", "GLD", "LQD"]   # the other four panel assets, never traded
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {
    "panel_tickers": ["SPY", "EEM", "TLT", "GLD", "LQD"],
    "risk_on_ticker": "SPY",
    "defensive_ticker": "SHY",
    "return_definition": "daily_log_return",
    "covariance_estimator": "expanding_sample_ddof1_through_t_minus_1",
    "covariance_min_sessions": 500,
    "max_covariance_condition_number_exponent": 8,      # limit = 10 ** 8
    "smoothing_window_sessions": 10,
    "percentile_min_history_sessions": 250,
    "percentile_definition":
        "share_of_strictly_earlier_smoothed_values_strictly_below",
    "defensive_percentile_above": 0.8,
    "decision_day": "scheduled_last_session_of_each_week",
    "rebalance": "weekly",
    "warmup_holding": "risk_on_ticker",
}
# Configured, conservative floor (not a measured count): a weekly de-risk
# filter over two decades of history is expected to complete well over this
# many segments; a run below it points at a data or wiring problem.
EXPECTED_MIN_TRADES = 40

_PANEL = list(PARAMETERS["panel_tickers"])
_RISK = str(PARAMETERS["risk_on_ticker"])
_DEFENSIVE = str(PARAMETERS["defensive_ticker"])
_MIN_SESSIONS = int(PARAMETERS["covariance_min_sessions"])
_MAX_CONDITION = 10.0 ** int(PARAMETERS["max_covariance_condition_number_exponent"])
_WINDOW = int(PARAMETERS["smoothing_window_sessions"])
_MIN_HISTORY = int(PARAMETERS["percentile_min_history_sessions"])
_THRESHOLD = float(PARAMETERS["defensive_percentile_above"])
_WARMUP = str(PARAMETERS["warmup_holding"])
_WARMUP_CHOICES = ("risk_on_ticker", "flat")
_JUNETEENTH_FIRST_YEAR = 2022        # first year the exchange observed it


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, Mapping) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def clean_close(frame: pd.DataFrame) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series of finite
    values. A blank close is dropped like a missing session (the panel then
    treats it as a hole); a non-positive close is refused because a log
    return is undefined there and no ETF price can be zero or negative."""
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(
        dtype=float), index=index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[np.isfinite(close.to_numpy())]
    if (close <= 0.0).any():
        raise ValueError(f"{FAMILY_ID}: Close values must be positive "
                         f"(first offender {close[close <= 0.0].index[0].date()})")
    return close


def panel_log_returns(closes: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Daily log returns of every asset on the sessions where ALL of them
    have a genuine one-session return.

    The assets' calendars are outer-joined, each asset's log return is the
    difference from the PREVIOUS session of the joined calendar, and a row
    is kept only when every asset has a finite value. So one asset missing a
    print drops that session (no return there) AND the next one (its return
    there would span two sessions while the others' span one): a return of a
    different horizon is never mixed into the vector. A session absent from
    every asset is a market closure, not a hole, and the return across it is
    a legitimate multi-day return."""
    joined = pd.concat({name: series for name, series in closes.items()},
                       axis=1).sort_index()
    logs = np.log(joined.astype(float))
    return logs.diff().dropna(how="any")


def turbulence_series(returns: pd.DataFrame, *, min_sessions: int,
                      max_condition_number: float) -> pd.Series:
    """Kritzman-Li financial turbulence, one value per session.

    d_t = (r_t - mean)' inv(cov) (r_t - mean) with ``mean`` and ``cov`` the
    EXPANDING sample mean and covariance (ddof = 1) of the sessions strictly
    BEFORE t -- r_t is never part of its own estimate -- and undefined until
    ``min_sessions`` earlier sessions exist. The statistics are updated
    sequentially (Welford), so a session can only ever see its past.

    A covariance that is not positive definite, or whose condition number
    (largest over smallest eigenvalue) exceeds ``max_condition_number``,
    raises ValueError naming the session: the matrix is NEVER regularized,
    shrunk or repaired, because a silently repaired matrix would score a
    different statistic than the pre-registered one."""
    values = returns.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{FAMILY_ID}: returns must be a 2-D table of assets")
    if not np.isfinite(values).all():
        raise ValueError(f"{FAMILY_ID}: returns must be finite")
    if min_sessions < 2:
        raise ValueError(f"{FAMILY_ID}: min_sessions must be at least 2")
    n, p = values.shape
    out = np.full(n, np.nan)
    mean = np.zeros(p)
    scatter = np.zeros((p, p))
    count = 0
    for k in range(n):
        x = values[k]
        if count >= min_sessions:
            cov = scatter / (count - 1)
            cov = (cov + cov.T) / 2.0
            eigenvalues = np.linalg.eigvalsh(cov)
            low, high = float(eigenvalues[0]), float(eigenvalues[-1])
            condition = high / low if low > 0.0 else float("inf")
            if not condition <= max_condition_number:
                date = pd.Timestamp(returns.index[k]).date().isoformat()
                raise ValueError(
                    f"{FAMILY_ID}: covariance is singular or ill-conditioned "
                    f"at {date} (smallest eigenvalue {low:.3g}, condition "
                    f"number {condition:.3g}, limit {max_condition_number:.3g})"
                    "; refusing to regularize")
            deviation = x - mean
            out[k] = float(deviation @ np.linalg.solve(cov, deviation))
        count += 1
        delta = x - mean
        mean = mean + delta / count
        scatter = scatter + np.outer(delta, x - mean)
    return pd.Series(out, index=returns.index, name="turbulence")


def smoothed_mean(values, *, window: int) -> pd.Series:
    """Trailing ``window``-session mean ending at each session (that session
    included), a direct numpy mean per window; undefined until ``window``
    values exist and while any value in the window is undefined."""
    if window < 1:
        raise ValueError(f"{FAMILY_ID}: window must be at least 1")
    array = np.asarray(values, dtype=float)
    out = np.full(len(array), np.nan)
    if len(array) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(array, window)
        out[window - 1:] = windows.mean(axis=1)
    index = values.index if isinstance(values, pd.Series) else None
    return pd.Series(out, index=index)


def percentile_ranks(smoothed, positions, *, min_history: int) -> np.ndarray:
    """For each position: the share of STRICTLY EARLIER finite values of
    ``smoothed`` that are STRICTLY BELOW the value at that position (ties are
    not below). NaN where the value itself is undefined or fewer than
    ``min_history`` earlier finite values exist."""
    array = np.asarray(smoothed, dtype=float)
    out = np.full(len(positions), np.nan)
    for slot, position in enumerate(positions):
        position = int(position)
        current = array[position]
        if not np.isfinite(current):
            continue
        prior = array[:position]
        prior = prior[np.isfinite(prior)]
        if len(prior) < min_history:
            continue
        out[slot] = np.count_nonzero(prior < current) / len(prior)
    return out


def is_defensive(percentile: float, *, threshold: float) -> bool:
    """True only when the percentile is strictly above the threshold (an
    undefined percentile is never defensive)."""
    return bool(np.isfinite(percentile) and percentile > threshold)


def _good_friday(year: int) -> pd.Timestamp:
    return pd.Timestamp(easter(year)) - pd.Timedelta(days=2)


def _friday_closures(year: int) -> set:
    """The scheduled full-day closures that land on a Friday, fixed in
    advance by the exchange's published rules: Good Friday; a holiday that
    falls ON a Friday; and a holiday that falls on a Saturday is observed on
    the preceding Friday -- except New Year's Day, whose Saturday case
    leaves the exchange OPEN on December 31 (year-end accounting). Juneteenth
    counts from its first observance in 2022."""
    closed = {_good_friday(year)}
    holidays = [(1, 1), (7, 4), (12, 25)]
    if year >= _JUNETEENTH_FIRST_YEAR:
        holidays.append((6, 19))
    for month, day in holidays:
        holiday = pd.Timestamp(year, month, day)
        if holiday.weekday() == 4:
            closed.add(holiday)
        elif holiday.weekday() == 5:
            closed.add(holiday - pd.Timedelta(days=1))
    # New Year's Day needs no special case for its Saturday: the Friday before
    # it is December 31 of the PREVIOUS year, a date this year's set is never
    # consulted for (``scheduled_week_end`` keys the set on the Friday's own
    # year), so the exchange staying open that day falls out of the keying.
    # Pinned by the December 31 examples and the whole-set XNYS comparison.
    return closed


def scheduled_week_end(day) -> pd.Timestamp:
    """The exchange's SCHEDULED last session of the Monday-to-Friday week
    containing ``day`` (a Saturday or Sunday belongs to the week just ended):
    that week's Friday, stepping back one day over a scheduled Friday
    closure. Read from no data file, so no later observation can decide it."""
    day = pd.Timestamp(day).normalize()
    friday = day + pd.Timedelta(days=4 - day.weekday())
    while friday.weekday() >= 5 or friday in _friday_closures(friday.year):
        friday -= pd.Timedelta(days=1)
    return friday


def week_end_dates(calendar) -> pd.DatetimeIndex:
    """The scheduled week ends of every week ``calendar`` spans that are
    actually PRESENT in ``calendar``. A scheduled week end missing from the
    data (a vendor hole or an unscheduled closure) yields no week end for
    that week -- never an earlier day picked with hindsight."""
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    if len(cal) == 0:
        return cal
    first_monday = cal[0].normalize() - pd.Timedelta(days=cal[0].weekday())
    mondays = pd.date_range(first_monday, cal[-1], freq="7D")
    scheduled = pd.DatetimeIndex([scheduled_week_end(m) for m in mondays])
    return cal[cal.isin(scheduled)]


def _weeks_without_a_reading(readings: pd.Series,
                             last_date: pd.Timestamp) -> list:
    """Scheduled week ends of COMPLETE weeks, from the week of the first
    reading onward, that have no reading. A week is complete when its
    scheduled week end is not after the last panel date; the week still in
    progress and the warm-up before the first reading are never counted."""
    have = set(readings.index)
    first = readings.index[0]
    monday = first.normalize() - pd.Timedelta(days=first.weekday())
    missing = []
    for week_start in pd.date_range(monday, last_date, freq="7D"):
        scheduled = scheduled_week_end(week_start)
        if scheduled <= last_date and scheduled not in have:
            missing.append(scheduled)
    return missing


def weekly_readings(returns: pd.DataFrame, *, min_sessions: int,
                    max_condition_number: float, window: int,
                    min_history: int, threshold: float,
                    require_complete_weeks: bool) -> pd.Series:
    """1.0 (hold the risk asset) / 0.0 (hold the defensive asset) at every
    scheduled week end of the panel that has a defined reading; weeks
    without one are simply absent. Every load-bearing parameter is an
    explicit argument -- nothing relies on a default."""
    turbulence = turbulence_series(
        returns, min_sessions=min_sessions,
        max_condition_number=max_condition_number)
    smoothed = smoothed_mean(turbulence, window=window)
    ends = week_end_dates(returns.index)
    positions = returns.index.get_indexer(ends)
    ranks = percentile_ranks(smoothed, positions, min_history=min_history)
    defined = np.isfinite(ranks)
    values = [0.0 if is_defensive(rank, threshold=threshold) else 1.0
              for rank in ranks[defined]]
    readings = pd.Series(values, index=ends[defined], dtype=float)
    if require_complete_weeks and not readings.empty:
        missing = _weeks_without_a_reading(readings, returns.index[-1])
        if missing:
            shown = ", ".join(d.date().isoformat() for d in missing[:6])
            raise ValueError(
                f"{FAMILY_ID}: {len(missing)} complete week(s) after the first "
                f"reading have no reading at their scheduled week end "
                f"(first: {shown}); refetch the data, or pass "
                "require_complete_weeks=False to carry the previous state "
                "through them")
    return readings


def risk_on_readings(frames, *, require_complete_weeks: bool = True) -> pd.Series:
    """The pre-registered weekly state from the full-history frames: 1.0 at
    a scheduled week end whose turbulence percentile is NOT above the
    threshold (hold the risk asset), 0.0 when it is (hold the defensive
    asset). Every value depends only on sessions at or before its own date."""
    closes = {ticker: clean_close(_require_frame(frames, ticker))
              for ticker in _PANEL}
    returns = panel_log_returns(closes)
    if returns.empty:
        raise ValueError(f"{FAMILY_ID}: no session on which all "
                         f"{len(_PANEL)} panel assets have a return")
    return weekly_readings(
        returns, min_sessions=_MIN_SESSIONS,
        max_condition_number=_MAX_CONDITION, window=_WINDOW,
        min_history=_MIN_HISTORY, threshold=_THRESHOLD,
        require_complete_weeks=require_complete_weeks)


def make_target_fn(frames, *, require_complete_weeks: bool = True,
                   warmup_holding: str | None = None):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history frames for the two traded legs and the four signal-only
    assets, handed once; the weekly state is computed ONCE from them (every
    entry uses only sessions at or before its own date) and delegated to
    ``signals.make_external_regime_target_fn``'s as-of lookup, which is what
    carries a reading through the week and into a restricted bucket that
    starts mid-regime."""
    warmup = _WARMUP if warmup_holding is None else warmup_holding
    if warmup not in _WARMUP_CHOICES:
        raise ValueError(f"{FAMILY_ID}: warmup_holding must be one of "
                         f"{_WARMUP_CHOICES}, got {warmup!r}")
    for symbol in (*TRADED, *SIGNAL):
        _require_frame(frames, symbol)
    readings = risk_on_readings(frames,
                                require_complete_weeks=require_complete_weeks)
    if readings.empty:
        raise ValueError(
            f"{FAMILY_ID}: no week end has a defined reading (needs "
            f"{_MIN_SESSIONS} earlier sessions for the covariance, "
            f"{_WINDOW} for the smoothing and {_MIN_HISTORY} earlier smoothed "
            "values for the percentile); a family that could never trade "
            "must not be scored")
    state = readings
    if warmup == "risk_on_ticker":
        seed_date = min(pd.DatetimeIndex(frames[symbol].index).min()
                        for symbol in (*TRADED, *SIGNAL))
        state = pd.concat([pd.Series([1.0], index=[seed_date]), readings])
    return signals.make_external_regime_target_fn(
        state, risk_ticker=_RISK, defensive_ticker=_DEFENSIVE)
