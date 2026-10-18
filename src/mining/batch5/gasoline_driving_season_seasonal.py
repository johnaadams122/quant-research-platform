"""Batch5 candidate: gasoline_driving_season_seasonal (single_long).

Calendar-only seasonal rotation on the United States Gasoline Fund (UGA):
long from the first trading session on/after March 1 through the last
trading session on/before the last Monday of May (Memorial Day -- an NYSE
Arca full-day closure every year the market trades in May), flat/cash the
rest of the calendar year. Repeats every year independently
("rebalance": "annual" in the catalog's own PARAMETERS).

Source (verbatim from the pre-registered catalog entry; see this
module's own PARAMETERS/EXPECTED_MIN_TRADES for the drift-checked
values): US EIA, "Gasoline price fluctuations"; US BLS, "Gasoline
prices: cyclical trends and market developments". Ticker: UGA,
United States Gasoline Fund, inception Feb 2008, K-1-issuing commodity
pool, taxable-only scope.

Mechanism (verbatim from the catalog): "US gasoline prices typically
bottom in Jan-Feb and rise into a late-spring/early-summer peak, driven
by spring refinery-maintenance turnarounds, the mandated summer-blend
fuel switch around March 1, and rising driving demand into summer. Holds
UGA long from March 1 through Memorial Day each year, flat/cash the rest
of the year."

Rule, restated precisely enough to reimplement (the catalog's four
PARAMETERS: entry_month=3, entry_day=1, exit_rule="last_monday_of_may",
rebalance="annual"):

    For every calendar year Y touching UGA's own bar history:
        entry_target(Y) = March 1 of year Y                  (entry_month/day)
        exit_target(Y)  = the last Monday of May in year Y    (exit_rule)
        in_season(bar)  = entry_target(Y) <= bar_date <= exit_target(Y),
                          where Y = bar_date.year (the window never crosses
                          a year boundary, so this is unambiguous)
    entry = in_season(next bar in UGA's own (truncated) calendar)
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
    exit  = in_season(bar) AND NOT in_season(next bar in UGA's own
                                              (truncated) calendar)
                                                     [fills at THIS bar's
                                                       close]
    every other bar: entry=False, exit=False (flat/cash)

This is a pure CALENDAR rule -- no UGA price ever enters the signal, and
the year-Y window dates are computable from the calendar alone (no
"trailing return", no external ticker), so there is no warm-up
requirement the way a returns-based rule needs a prior observation.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- entry_target/exit_target are pure date arithmetic
  (never derived from price data); the only history-dependent fact
  computed is which of UGA's OWN (truncated) bar dates fall in-season.
* PERFORMANCE -- entry/exit are two precomputed boolean arrays; the
  returned signal_fn does two numpy lookups by position plus one scalar
  date check.
* HOLDOUT-BLIND -- UGA's frame is truncated to <= the last bar of
  ``index`` before anything is computed, so a holdout bucket's signals
  cannot depend on data after the bucket (see interpretation choice 5 for
  why this matters even though the rule never reads price).
* DIRECTION -- always long UGA when in-season, flat (cash) otherwise;
  never short.

Interpretation choices (the pseudocode leaves these to the implementer):

1. "Last trading session before Memorial Day." Memorial Day (the calendar
   target of ``exit_rule="last_monday_of_may"``) is itself an NYSE Arca
   full-day closure every year the exchange trades in May, so there is
   never a bar exactly on that date. Rather than search for "the nearest
   trading day" as a running comparison against "today" (which, naively
   implemented, would re-fire an exit signal on every remaining day of
   the year once the season had technically ended), the exit is derived
   directly from the ``in_season`` predicate: it fires on the LAST bar
   whose own date is <= exit_target(Y), i.e. the bar immediately
   preceding the first bar whose date exceeds exit_target(Y). This is
   almost always the Friday before Memorial Day.
2. "Entry ... March 1" is read literally as a calendar date, not "the
   first trading day of March": if March 1 itself is not a trading day
   (weekend, or a rare weekday closure), the entry flag is simply raised
   on every bar from the first available trading session on/after March
   1 through the exit bar (see choice 3), so the position opens at the
   FIRST fill opportunity on/after March 1 regardless of which weekday it
   falls on.
3. Entry is a ONE-BAR-EARLY shift of ``in_season`` (``entry[t] =
   in_season[t+1]`` in the ticker's own truncated bar sequence), not
   ``in_season[t]`` itself. Two things this gets right that the naive
   "entry = in_season" construction gets wrong (the naive form would
   re-enter the very next trading session after every single exit --
   because the exit bar is itself the last IN-season bar, so using
   ``in_season`` directly as the entry flag re-raises "enter" on the day
   the engine is free to read it again):
     (a) it lands the ACTUAL fill (which the engine always places at the
         bar AFTER the one carrying the flag) exactly on the first
         in-season bar's own open, rather than one session late; and
     (b) it goes False on the season's own last bar (whose successor is
         off-season), so there is no flag left over to trigger an
         immediate re-entry the session after exit.
   The flag still stays raised on every OTHER in-season bar (each one's
   successor is also in-season, except the last), which is what lets a
   holdout bucket that opens mid-season -- e.g. a window starting April 1,
   after this year's March 1 entry date but before its Memorial Day exit
   -- still enter at its own first fill opportunity rather than never
   entering at all, the same reading fx_dollar_index_trend_following's
   convention (its interpretation choice 5) adopted for an identical
   mid-window boundary.
4. year(bar) is simply the bar's own calendar year: the season never
   crosses a year boundary (March through May), so there is no
   end-of-December / New Year ambiguity to resolve, unlike a rule whose
   window straddles Dec 31.
5. The FINAL bar of the (truncated) UGA history never raises an exit flag
   on its own account, by the same reasoning as every other calendar-only
   candidate in this batch: without a following bar the module cannot
   know whether the season has ended, and the engine's own end_of_data
   path force-closes any still-open position at that same close anyway,
   so no P&L is changed by leaving it undetermined (only the recorded
   exit reason, "end_of_data" instead of "signal_exit").
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

FAMILY_ID = "gasoline_driving_season_seasonal"
SHAPE = "single_long"
TRADED = ["UGA"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "UGA"
# Copied byte-for-byte from the pre-registered candidate's "parameters"
# (driver drift check); this module's own test compares against a frozen
# literal copy of that record.
PARAMETERS = {
    "entry_month": 3,
    "entry_day": 1,
    "exit_rule": "last_monday_of_may",
    "rebalance": "annual",
}
# UGA history (data_pipeline.fetch_ohlc("UGA", start="max")) begins before
# the first season's March 1 entry target (the fund's inception is Feb
# 2008), so every calendar year from the first through the most recent
# contributes one full season. By construction the rule makes one trade
# per season, one round trip each (no consecutive-hold merging is possible
# here -- seasons never touch, there is always a flat gap June-February
# between them); each exit lands on the last session on/before that year's
# Memorial Day, usually the Friday before it (interpretation choice 1), so
# a season holds for roughly three months of sessions. Under an approximate
# 6-month holdout ending at the history's last date, only the final
# season straddles the holdout boundary; all earlier seasons are entirely
# before it. EXPECTED_MIN_TRADES is set below the in-sample season count
# as a safety margin against small holdout-boundary shifts:
EXPECTED_MIN_TRADES = 15

_ENTRY_MONTH = int(PARAMETERS["entry_month"])
_ENTRY_DAY = int(PARAMETERS["entry_day"])
if PARAMETERS["exit_rule"] != "last_monday_of_may":
    raise ValueError(
        f"{FAMILY_ID}: unsupported exit_rule {PARAMETERS['exit_rule']!r}")


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
    ``years``, all sharing the same (month, day). Pure calendar arithmetic,
    no data dependency."""
    n = len(years)
    frame = pd.DataFrame({
        "year": years,
        "month": np.full(n, month, dtype=np.int64),
        "day": np.full(n, day, dtype=np.int64),
    })
    return pd.DatetimeIndex(pd.to_datetime(frame))


def last_monday_of_may(years: np.ndarray) -> pd.DatetimeIndex:
    """The last Monday of May (Memorial Day) for each year in ``years`` --
    pure calendar arithmetic, no data dependency. Vectorized: May 31 minus
    its own ISO weekday offset (Monday=0 ... Sunday=6)."""
    may31 = _dates_from_ymd(years, 5, 31)
    dow = may31.dayofweek.to_numpy()
    return may31 - pd.to_timedelta(dow, unit="D")


def in_season_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """True where ``dates[k]`` falls on/after March 1 and on/before the
    last Monday of May of its OWN calendar year (interpretation choices
    1, 2, 4). ``dates`` must already be tz-naive and normalized (not
    required to be sorted/unique -- this is a pure elementwise predicate)."""
    n = len(dates)
    if n == 0:
        return np.zeros(0, dtype=bool)
    years = dates.year.to_numpy()
    entry_target = _dates_from_ymd(years, _ENTRY_MONTH, _ENTRY_DAY)
    exit_target = last_monday_of_may(years)
    dates_arr = dates.to_numpy()
    return ((dates_arr >= entry_target.to_numpy())
            & (dates_arr <= exit_target.to_numpy()))


def season_decisions(traded_dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Precompute (entry, exit) boolean Series on UGA's own (already
    truncated) bar calendar. ``traded_dates`` must be tz-naive, normalized,
    sorted and unique.

    entry[t] = in_season[t+1] -- True one bar BEFORE every in-season bar
    (choice 3), which is what lands the engine's next-bar-open fill on the
    season's own first open and what stops the flag re-firing the session
    right after exit. exit[t] = in_season[t] and not in_season[t+1] --
    True only on the LAST in-season bar of each season block (choice 1).
    Neither is ever True on the final bar of the whole array (choice 5) --
    there is no known "next bar" to compare against without looking past
    the truncation cutoff."""
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
