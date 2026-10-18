"""Batch5 candidate: natural_gas_withdrawal_season_seasonal (single_long).

Calendar-only seasonal rotation on the United States Natural Gas Fund
(UNG): long from November 1 through March 31 of the following year (the
physical storage WITHDRAWAL season), flat/cash the rest of the calendar
year (the INJECTION season, April-October). Repeats every winter
independently ("rebalance": "annual" in the catalog's own PARAMETERS).

Source (verbatim from the pre-registered catalog entry; see this
module's own PARAMETERS/EXPECTED_MIN_TRADES for the drift-checked
values): Mu, X. (2007), "Weather, storage, and natural gas price
dynamics," Energy Economics; CME Group, "Introduction to Natural Gas
Seasonality". Ticker verified:
UNG, United States Natural Gas Fund, inception April 2007, ~504 million USD AUM.
K-1-issuing commodity pool, taxable-only scope.

Mechanism (verbatim from the catalog): "Henry Hub natural gas follows the
physical storage cycle: April-October injection season sees price
weakness, November-March withdrawal season sees heating-driven demand
draw storage down, historically pushing a seasonal premium. Holds UNG
long Nov 1 - Mar 31 each winter, flat/cash the rest of the year."

Rule, restated precisely enough to reimplement (the catalog's five
PARAMETERS: entry_month=11, entry_day=1, exit_month=3, exit_day=31,
rebalance="annual"):

    THIS FAMILY'S SEASON CROSSES A CALENDAR YEAR BOUNDARY, unlike
    gasoline_driving_season_seasonal's March-May window. "year(bar) =
    bar's own calendar year" is NOT a valid season key -- a bar in
    Jan/Feb/March of year Y+1 belongs to the season keyed Y, not Y+1.

    season_key(date) = date.year        if date.month >= entry_month (11)
                      = date.year - 1   if date.month <= exit_month (3)
                      = None (no season) otherwise (Apr-Oct, the gap)

    For every reachable season year Y:
        entry_target(Y) = November 1 of year Y       (entry_month/day)
        exit_target(Y)  = March 31 of year Y+1        (exit_month/day)
        in_season(bar)  = season_key(bar) is not None
                           AND entry_target(season_key(bar)) <= bar_date
                               <= exit_target(season_key(bar))
    entry = in_season(next bar in UNG's own (truncated) calendar)
                                                     [fills at THIS bar's
                                                       next open -- raising
                                                       the flag one bar
                                                       EARLY is what makes
                                                       the engine's fixed
                                                       "fill at next bar's
                                                       open" convention land
                                                       exactly on the first
                                                       in-season bar's own
                                                       open]
    exit  = in_season(bar) AND NOT in_season(next bar in UNG's own
                                              (truncated) calendar)
                                                     [fills at THIS bar's
                                                       close]
    every other bar: entry=False, exit=False (flat/cash)

This is a pure CALENDAR rule -- no UNG price ever enters the signal, and
the season-year window dates are computable from the calendar alone (no
"trailing return", no external ticker), so there is no warm-up
requirement the way a returns-based rule needs a prior observation.

Design (the four non-negotiable properties, same as every other batch5
calendar-only module):

* NO LOOK-AHEAD -- entry_target/exit_target are pure date arithmetic
  (never derived from price data); the only history-dependent fact
  computed is which of UNG's OWN (truncated) bar dates fall in-season.
* PERFORMANCE -- entry/exit are two precomputed boolean arrays (built via
  vectorized per-season-year entry_target/exit_target arrays, the same
  construction gasoline_driving_season_seasonal.last_monday_of_may uses
  per calendar year -- never a slow per-row Python loop); the returned
  signal_fn does two numpy lookups by position plus one scalar date
  check.
* HOLDOUT-BLIND -- UNG's frame is truncated to <= the last bar of
  ``index`` before anything is computed, so a holdout bucket's signals
  cannot depend on data after the bucket.
* DIRECTION -- always long UNG when in-season, flat (cash) otherwise;
  never short.

Interpretation choices (the pseudocode leaves these to the implementer;
follows gasoline_driving_season_seasonal's own conventions except where
the year-boundary crossing forces a difference, noted explicitly):

1. entry_month/entry_day (11/1) and exit_month/exit_day (3/31) are read
   literally as calendar dates, not "the first/last trading day of the
   month": if November 1 itself is not a trading day (weekend or a rare
   weekday closure), the entry flag is raised on every bar from the
   first available trading session on/after November 1 through the exit
   bar, so the position opens at the FIRST fill opportunity on/after
   November 1 regardless of weekday. Symmetrically, since exit_day=31 is
   the LAST possible calendar day of March, every trading day of March
   is <= exit_target -- the exit fires on the LAST trading session on/
   before March 31 (almost always March 31 itself when it is a weekday,
   else the preceding Friday).
2. Entry is a ONE-BAR-EARLY shift of ``in_season`` (``entry[t] =
   in_season[t+1]`` in the ticker's own truncated bar sequence), not
   ``in_season[t]`` itself -- identical reasoning to
   gasoline_driving_season_seasonal's interpretation choice 3: this lands
   the actual fill (always the bar AFTER the flagged one) exactly on the
   first in-season bar's own open, and it goes False on the season's own
   last bar (whose successor is off-season), so there is no flag left
   over to trigger an immediate re-entry the session right after exit.
   The flag still stays raised on every OTHER in-season bar, which is
   what lets a holdout bucket that opens mid-season -- e.g. a window
   starting January 2, after that winter's November-1 entry date but
   before its March-31 exit -- still enter at its own first fill
   opportunity rather than never entering at all.
3. season_key(date) -- THE INTERPRETATION CHOICE THIS FAMILY ADDS beyond
   gasoline_driving_season_seasonal (whose March-May window never crosses
   a year boundary, so its "year(bar) = bar's own calendar year" is
   unambiguous and would be WRONG here). Because entry_month (11) >
   exit_month (3), a bar's season year is date.year when the bar's month
   is >= entry_month (November/December bars start a new season keyed to
   their own year) and date.year - 1 when the bar's month is <=
   exit_month (January/February/March bars finish the PRECEDING year's
   season). A bar in the gap months (April-October) has no season at
   all. This is computed vectorized via season_year_for_date(), which
   returns both the candidate season-year array and a validity mask
   (Apr-Oct bars are invalid/no-season, not merely "wrong season").
4. The FINAL bar of the (truncated) UNG history never raises an exit
   flag on its own account, by the same reasoning as every other
   calendar-only candidate in this batch: without a following bar the
   module cannot know whether the season has ended, and the engine's own
   end_of_data path force-closes any still-open position at that same
   close anyway, so no P&L is changed by leaving it undetermined (only
   the recorded exit reason, "end_of_data" instead of "signal_exit").
5. UNG's real inception is April 2007 -- squarely inside a GAP month
   (Apr-Oct), before the first reachable season's own November-1 entry
   target. This does not special-case the truncation logic: a frame
   whose entire history is off-season simply produces all-False
   entry/exit arrays (verified by a dedicated unit test), because
   season_year_for_date's validity mask depends only on each bar's own
   month, never on whether an earlier or later bar exists.
6. A non-finite or missing Close is never consulted by this rule at all
   (it is pure calendar arithmetic on bar DATES), so a bad bar's Close
   cannot corrupt entry/exit; the module still validates that a Close
   column exists (contract hygiene, matching every other batch5 module)
   but never reads its values for the signal itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "natural_gas_withdrawal_season_seasonal"
SHAPE = "single_long"
TRADED = ["UNG"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "UNG"
# Copied byte-for-byte from the pre-registered candidate's "parameters"
# (driver drift check); this module's own test compares against a frozen
# literal.
PARAMETERS = {
    "entry_month": 11,
    "entry_day": 1,
    "exit_month": 3,
    "exit_day": 31,
    "rebalance": "annual",
}
# One trade per winter season: UNG's history spans roughly two decades of
# seasons, and no consecutive-hold merging is possible (seasons never touch;
# there is always a flat Apr-Oct gap between them). EXPECTED_MIN_TRADES is
# set a few seasons below that count as a safety margin against small
# holdout-boundary shifts (the same convention
# gasoline_driving_season_seasonal's own EXPECTED_MIN_TRADES comment
# documents):
EXPECTED_MIN_TRADES = 15

_ENTRY_MONTH = int(PARAMETERS["entry_month"])
_ENTRY_DAY = int(PARAMETERS["entry_day"])
_EXIT_MONTH = int(PARAMETERS["exit_month"])
_EXIT_DAY = int(PARAMETERS["exit_day"])
if _ENTRY_MONTH <= _EXIT_MONTH:
    raise ValueError(
        f"{FAMILY_ID}: this module's season_key math assumes a "
        f"year-crossing window (entry_month > exit_month), got "
        f"entry_month={_ENTRY_MONTH}, exit_month={_EXIT_MONTH}")


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


def _dates_from_ymd(years: np.ndarray, month: int, day: int) -> pd.DatetimeIndex:
    """Vectorized calendar-date construction: one Timestamp per entry of
    ``years``, all sharing the same (month, day). Pure calendar
    arithmetic, no data dependency."""
    n = len(years)
    frame = pd.DataFrame({
        "year": years,
        "month": np.full(n, month, dtype=np.int64),
        "day": np.full(n, day, dtype=np.int64),
    })
    return pd.DatetimeIndex(pd.to_datetime(frame))


def season_year_for_date(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """The candidate season-year Y for each date in ``dates``, plus a
    validity mask (interpretation choice 3): Y = date.year when the bar's
    month is >= entry_month (a November/December bar starts a season
    keyed to its own year), Y = date.year - 1 when the bar's month is <=
    exit_month (a January/February/March bar finishes the PRECEDING
    year's season). A bar in a gap month (April-October here) has no
    season at all -- its ``valid`` entry is False and its ``years`` entry
    is a meaningless placeholder (never consumed downstream).

    This is the exact mechanism that catches a naive "year(bar) = bar's
    own calendar year" implementation: a December-of-Y bar and a
    February-of-(Y+1) bar both return Y here, not two different years."""
    n = len(dates)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    months = dates.month.to_numpy()
    calendar_years = dates.year.to_numpy()
    starts_new_season = months >= _ENTRY_MONTH
    finishes_prior_season = months <= _EXIT_MONTH
    valid = starts_new_season | finishes_prior_season
    years = np.where(starts_new_season, calendar_years, calendar_years - 1)
    return years, valid


def in_season_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """True where ``dates[k]`` falls within its own season window
    [entry_target(Y), exit_target(Y)] for the season year Y that
    season_year_for_date assigns it (or False if the bar has no season
    at all -- a gap-month bar). ``dates`` must already be tz-naive and
    normalized (not required to be sorted/unique -- this is a pure
    elementwise predicate)."""
    n = len(dates)
    if n == 0:
        return np.zeros(0, dtype=bool)
    years, valid = season_year_for_date(dates)
    entry_target = _dates_from_ymd(years, _ENTRY_MONTH, _ENTRY_DAY)
    exit_target = _dates_from_ymd(years + 1, _EXIT_MONTH, _EXIT_DAY)
    dates_arr = dates.to_numpy()
    in_range = ((dates_arr >= entry_target.to_numpy())
                & (dates_arr <= exit_target.to_numpy()))
    return in_range & valid


def season_decisions(traded_dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Precompute (entry, exit) boolean Series on UNG's own (already
    truncated) bar calendar. ``traded_dates`` must be tz-naive,
    normalized, sorted and unique.

    entry[t] = in_season[t+1] -- True one bar BEFORE every in-season bar
    (choice 2), which is what lands the engine's next-bar-open fill on
    the season's own first open and what stops the flag re-firing the
    session right after exit. exit[t] = in_season[t] and not
    in_season[t+1] -- True only on the LAST in-season bar of each season
    block. Neither is ever True on the final bar of the whole array
    (choice 4) -- there is no known "next bar" to compare against without
    looking past the truncation cutoff."""
    n = len(traded_dates)
    in_season = in_season_mask(traded_dates)
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    if n >= 2:
        entry[:-1] = in_season[1:]
        exit_[:-1] = in_season[:-1] & ~in_season[1:]
    return (pd.Series(entry, index=traded_dates, name="entry"),
            pd.Series(exit_, index=traded_dates, name="exit"))


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

    entry_series, exit_series = season_decisions(traded_dates)
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
