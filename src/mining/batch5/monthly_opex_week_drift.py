"""Batch5 candidate: monthly_opex_week_drift (single_long).

Monthly options-expiration-week rotation on SPY: long from 4 trading
sessions before the ordinary monthly (3rd-Friday) options-expiration
Friday, hold through that Friday's close, then flat/cash the rest of the
month. Recurs all 12 months a year (not just the 4 quarterly
triple-witching months).

Source (verbatim from the pre-registered catalog entry): Johnson, T.L. and
So, E.C. (2013), "Returns and option activity over the option-expiration
week for S&P 100 stocks", Journal of Banking and Finance 37(11),
4226-4240. https://www.sciencedirect.com/science/article/abs/pii/S0378426613003051
(also on Quantpedia). Ticker verified:
SPY.

Mechanism (verbatim from the catalog): "Buy SPY 4 trading sessions before
the ordinary monthly (3rd-Friday) options-expiration Friday, hold through
that Friday's close, then exit to cash for the rest of the month. Recurs
all 12 months/year (not just the 4 quarterly triple-witching months),
attributed to concentrated dealer delta-hedge rebalancing/pinning flows
around the largest recurring monthly options rolloff."

Rule, restated precisely enough to reimplement (the catalog's three
PARAMETERS: opex_week_definition="week_containing_third_friday_of_month",
entry_trading_days_before_opex_friday=4, exit_at="opex_friday_close"):

    For every (year, month) pair touching SPY's own (already
    truncated-to-bucket) bar dates:
        third_friday(Y, M) = the calendar date of the third Friday of
                              month M, year Y (pure date arithmetic: the
                              first day within days 1-7 of the month whose
                              weekday is Friday, plus 14 days)
        opex_ref_bar(Y, M)  = the LAST bar in SPY's own truncated bar
                              sequence whose own date is
                              <= third_friday(Y, M)
        entry_bar(Y, M)      = the SPY bar exactly 4 POSITIONS earlier
                              than opex_ref_bar(Y, M) in SPY's own
                              truncated bar sequence (a trading-session
                              count offset, not a calendar-day offset)
    entry = True on the bar immediately BEFORE entry_bar(Y, M)
                                                     [fills at THIS bar's
                                                       next open -- lands
                                                       the actual fill
                                                       exactly on
                                                       entry_bar's own
                                                       open]
    exit  = True ON opex_ref_bar(Y, M) itself
                                                     [fills at THIS bar's
                                                       own close, matching
                                                       "hold through that
                                                       Friday's close,
                                                       then exit"]
    every other bar: entry=False, exit=False (flat/cash)

A month whose opex_ref_bar or entry_bar position would not exist inside
the truncated bar sequence (not enough prior history, or -- see
interpretation choice 3 below -- the calendar third Friday has not yet
been reached by the truncated data) contributes NO window for that month
rather than crashing or fabricating one.

Design (the four non-negotiable properties, matching every other batch5
single_long candidate):

* NO LOOK-AHEAD -- entry/exit are derived only from SPY's OWN bar dates
  and pure calendar arithmetic on the third-Friday target; no future bar
  is ever consulted to decide a past bar's flag.
* PERFORMANCE -- entry/exit are two precomputed boolean arrays built with
  vectorized numpy/pandas ops (one np.searchsorted call over all months at
  once); the returned signal_fn does two numpy lookups by position plus
  one scalar date check.
* HOLDOUT-BLIND -- SPY's frame is truncated to <= the last bar of
  ``index`` before anything is computed (same convention as every sibling
  calendar-seasonal module), AND (interpretation choice 3 below) a
  trailing month whose own third-Friday session has not yet been
  confirmed by a LATER bar in the truncated sequence contributes no
  window -- without this a not-yet-complete month would silently invent a
  window out of "we ran out of data", and that window would retroactively
  change once more data arrived, which is exactly the leak the
  HOLDOUT-BLIND property forbids.
* DIRECTION -- always long SPY during the opex window, flat (cash)
  otherwise; never short.

Interpretation choices (the pseudocode leaves these to the implementer):

1. "The third Friday of the month" is pure calendar arithmetic: the first
   day within days 1-7 of the month whose weekday is Friday ("the first
   Friday"), plus 14 calendar days. Vectorized over every (year, month)
   pair at once via ``third_friday_of_month`` -- never per-row, never a
   per-calendar-day loop.
2. "Options-expiration Friday" is read as the calendar third Friday
   itself, not "the nearest trading day to it" computed by a running
   comparison against "today" -- the opex reference bar is derived
   directly as the LAST bar whose own date is <= that calendar target
   (mirrors gasoline_driving_season_seasonal's own interpretation choice
   1 for the analogous "target calendar date is not itself a trading
   day" case, e.g. Good Friday coinciding with a third Friday). This is
   almost always the third Friday itself; it only differs on the rare
   month where that exact date is a market holiday.
3. HOLDOUT-BLIND trailing-month guard (not spelled out in the bare
   mechanism prose, but required by the batch's own non-negotiable
   HOLDOUT-BLIND property -- see the Design section above): a month's
   opex_ref_bar is only trusted if there EXISTS a bar in the truncated
   sequence strictly AFTER that month's calendar third-Friday target.
   Without this, a truncated sequence that simply ends partway through a
   month (before that month's real third Friday has occurred) would have
   its LAST available bar treated as "the last bar <= target" purely
   because there is no more data -- not because the calendar event
   actually happened there -- fabricating a phantom opex_ref_bar (and
   therefore a phantom entry 4-5 sessions earlier) that would move once
   more data arrived. This is the direct analogue of
   gasoline_driving_season_seasonal's "final bar of the whole truncated
   array never triggers a fresh entry flag on its own account" property,
   generalized from "the very last bar" to "the whole trailing
   not-yet-confirmed month."
4. "The entry_bar position would not exist" (the rule above) is read to
   cover BOTH of the two ways a month's window can run off the start of
   the truncated sequence: (a) the entry bar's own position
   (opex_ref_pos - 4) is negative (fewer than 4 prior bars exist), and
   (b) the entry bar's position is exactly 0 but there is no bar BEFORE
   it to raise the required one-bar-early entry flag on (the entry flag
   sits on the bar before entry_bar). Both are "skip that month entirely
   (no window)", not "raise the exit anyway with no matching entry" --
   consistent with the rule's own "contributes NO window for that month"
   wording.
5. Windows from different months never overlap (the hold is ~5 trading
   sessions out of ~21 in a month), so there is no cross-month merging
   logic -- each confirmed month independently contributes at most one
   entry flag and one exit flag.
6. A non-finite or missing Close is never consulted by this rule at all
   (it is pure calendar + bar-position arithmetic), so a bad bar's Close
   cannot corrupt entry/exit; the module still validates that a Close
   column exists (contract hygiene, matching every other batch5 module)
   but never reads its values for the signal itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "monthly_opex_week_drift"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from the pre-registered candidate's "parameters"
# (driver drift check).
PARAMETERS = {
    "opex_week_definition": "week_containing_third_friday_of_month",
    "entry_trading_days_before_opex_friday": 4,
    "exit_at": "opex_friday_close",
}
# Structurally, the rule makes one round trip per calendar month of SPY
# history, each closed by reason="signal_exit" and held EXACTLY 4 sessions
# (held is the positional entry-bar-to-opex-bar gap, which PARAMETERS pins
# at 4).
# The newest calendar month touching the fetch correctly produces no trade
# when its third Friday falls past the fetch's own last bar, so
# interpretation choice 3's confirmation guard withholds it rather than
# fabricating a trailing signal. EXPECTED_MIN_TRADES is set below that
# calendar-derived count as a safety margin against small holdout-boundary shifts
# and a trailing month or two dropping out under the confirmation guard:
EXPECTED_MIN_TRADES = 350

_ENTRY_DAYS_BEFORE = int(PARAMETERS["entry_trading_days_before_opex_friday"])
if PARAMETERS["opex_week_definition"] != "week_containing_third_friday_of_month":
    raise ValueError(
        f"{FAMILY_ID}: unsupported opex_week_definition "
        f"{PARAMETERS['opex_week_definition']!r}")
if PARAMETERS["exit_at"] != "opex_friday_close":
    raise ValueError(f"{FAMILY_ID}: unsupported exit_at {PARAMETERS['exit_at']!r}")


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _dates_from_ymd(years: np.ndarray, months: np.ndarray,
                     days: np.ndarray) -> pd.DatetimeIndex:
    """Vectorized calendar-date construction: one Timestamp per (year,
    month, day) triple. Pure calendar arithmetic, no data dependency."""
    frame = pd.DataFrame({
        "year": np.asarray(years, dtype=np.int64),
        "month": np.asarray(months, dtype=np.int64),
        "day": np.asarray(days, dtype=np.int64),
    })
    return pd.DatetimeIndex(pd.to_datetime(frame))


def third_friday_of_month(years, months) -> pd.DatetimeIndex:
    """The calendar date of the third Friday of each (year, month) pair in
    the two parallel arrays -- pure calendar arithmetic, no data
    dependency (interpretation choice 1). Vectorized: the first Friday of
    the month is day 1 plus ``(4 - weekday(day1)) % 7`` days (pandas
    dayofweek: Monday=0 ... Sunday=6, so Friday=4); the third Friday is 14
    days after that."""
    years = np.asarray(years, dtype=np.int64)
    months = np.asarray(months, dtype=np.int64)
    n = len(years)
    if n == 0:
        return pd.DatetimeIndex([])
    first_of_month = _dates_from_ymd(years, months, np.ones(n, dtype=np.int64))
    dow = first_of_month.dayofweek.to_numpy()
    offset_to_first_friday = (4 - dow) % 7
    first_friday = first_of_month + pd.to_timedelta(offset_to_first_friday, unit="D")
    return pd.DatetimeIndex(first_friday + pd.Timedelta(days=14))


def opex_reference_positions(traded_dates: pd.DatetimeIndex):
    """For every (year, month) pair touching ``traded_dates`` (sorted,
    unique, tz-naive, normalized), compute the calendar third Friday and
    the position in ``traded_dates`` of the "opex reference bar" -- the
    LAST bar whose own date is <= that Friday (interpretation choice 2).

    Returns ``(years, months, third_friday_dates, opex_pos)`` as four
    parallel numpy/pandas arrays, one entry per unique (year, month)
    touching ``traded_dates``, in chronological order.
    ``opex_pos[i] == -1`` when no bar in ``traded_dates`` exists on/before
    that month's third Friday (no opex_ref_bar exists: that month has no
    window)."""
    n = len(traded_dates)
    if n == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                pd.DatetimeIndex([]), np.zeros(0, dtype=np.int64))
    years_all = traded_dates.year.to_numpy().astype(np.int64)
    months_all = traded_dates.month.to_numpy().astype(np.int64)
    year_month = years_all * 100 + months_all
    unique_ym = np.unique(year_month)  # sorted ascending
    years = unique_ym // 100
    months = unique_ym % 100
    third_friday = third_friday_of_month(years, months)
    bar_values = traded_dates.asi8
    target_values = third_friday.asi8
    # Position of the LAST bar whose date is <= target: searchsorted with
    # side="right" gives the insertion point just past any equal
    # element(s); subtracting 1 lands on that last <= element (-1 if none).
    opex_pos = np.searchsorted(bar_values, target_values, side="right") - 1
    return years, months, third_friday, opex_pos


def opex_window_decisions(traded_dates: pd.DatetimeIndex) -> tuple:
    """Precompute (entry, exit) boolean Series on SPY's own (already
    truncated) bar calendar -- fully vectorized, no per-calendar-day and
    no per-month Python loop (a cumulative-sum range-fill over the whole
    months array at once).

    entry[t] = True on EVERY bar from the bar immediately before each
    confirmed month's entry bar (opex_ref_pos - entry_trading_days_before_
    opex_friday) through the bar immediately before opex_ref_bar itself
    (inclusive) -- NOT just the single one-bar-early bar. This mirrors the
    "stay raised while the position should be held" idiom every sibling
    batch5 single_long module uses (see e.g.
    fx_dollar_index_trend_following's or gasoline_driving_season_
    seasonal's own in-season/standing-decision flags): a bucket that opens
    mid-window (after the entry bar but before the opex exit -- a real
    scenario for any caller that hands this module a truncated `index`,
    not only a full-history one) must still see an entry flag on its own
    first bar in order to open the position at all, since the engine has
    no memory of flags before the bucket it was given. A single-bar entry
    flag would silently miss every such window (a bucket starting after the
    entry flag but before the opex exit would receive only the exit signal
    and stay flat).
    exit[t] = True on each confirmed month's opex_ref_bar itself. A month
    is "confirmed" only when (a) its opex_ref_bar exists at all
    (opex_pos >= 0), (b) a LATER bar exists in ``traded_dates`` to confirm
    that bar wasn't just "the data ran out" (interpretation choice 3),
    (c) the entry bar position is non-negative, and (d) a bar exists
    before the entry bar to carry the one-bar-early entry flag
    (interpretation choice 4). An unconfirmed month contributes neither
    flag.

    For a run starting from ``traded_dates[0]`` (e.g. ABI-1's full-history
    scoring path), the standing flag trades exactly like a single-bar
    entry flag: once the engine is in a position, a repeated entry=True is
    a no-op, so the trade count and dates are the same either way."""
    n = len(traded_dates)
    entry_arr = np.zeros(n, dtype=bool)
    exit_arr = np.zeros(n, dtype=bool)
    if n == 0:
        return (pd.Series(entry_arr, index=traded_dates, name="entry"),
                pd.Series(exit_arr, index=traded_dates, name="exit"))
    _, _, _, opex_pos = opex_reference_positions(traded_dates)
    exists = opex_pos >= 0
    confirmed_by_later_bar = opex_pos <= (n - 2)
    entry_pos = opex_pos - _ENTRY_DAYS_BEFORE
    entry_flag_pos = entry_pos - 1
    valid = (exists & confirmed_by_later_bar
             & (entry_pos >= 0) & (entry_flag_pos >= 0))
    exit_arr[opex_pos[valid]] = True
    # Range-fill entry_arr[entry_flag_pos : opex_pos] = True for every
    # valid month at once via a cumulative-sum delta trick (start +1 at
    # entry_flag_pos, -1 at opex_pos, then entry_arr = cumsum(delta) > 0)
    # -- vectorized, no per-month Python loop.
    delta = np.zeros(n + 1, dtype=np.int64)
    starts = entry_flag_pos[valid]
    ends = opex_pos[valid]  # exclusive upper bound of the True range
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    entry_arr = (np.cumsum(delta[:-1]) > 0)
    return (pd.Series(entry_arr, index=traded_dates, name="entry"),
            pd.Series(exit_arr, index=traded_dates, name="exit"))


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED (SIGNAL
    is empty for this family)."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    traded_frame = _require_frame(frames, ticker)

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    traded_dates = _naive_index(traded_frame.index)
    traded_dates = traded_dates[traded_dates <= last_bar].unique().sort_values()

    entry_series, exit_series = opex_window_decisions(traded_dates)
    entry_arr = entry_series.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    exit_arr = exit_series.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    bucket_values = bucket.asi8  # int64 ns for an O(1) scalar date check
    n_bucket = len(bucket)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n_bucket:
            raise IndexError(f"{FAMILY_ID}: bar position {i} outside the "
                             f"{n_bucket}-bar bucket this signal was built for")
        today = frame.index[-1]
        if pd.Timestamp(today).value != bucket_values[i]:
            raise ValueError(f"{FAMILY_ID}: bar {i} is {today}, but the bucket "
                             f"index has {bucket[i]} there -- the engine must "
                             "run over the same index handed to the factory")
        return SignalResult(entry=bool(entry_arr[i]), exit=bool(exit_arr[i]))

    return signal_fn
