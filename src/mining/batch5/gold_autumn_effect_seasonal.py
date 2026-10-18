"""Batch5 candidate: gold_autumn_effect_seasonal (single_long).

Calendar-only seasonal rotation on SPDR Gold Shares (GLD): long from
September 1 through November 30 each year, flat/cash the rest of the
calendar year. Repeats every year independently ("rebalance": "annual"
in PARAMETERS).

Source (verbatim from the pre-registered catalog entry): Baur, D.G.
(2013), "The autumn effect of gold", Research in International Business
and Finance. https://www.sciencedirect.com/science/article/abs/pii/
S0275531912000323. Ticker verified:
GLD, SPDR Gold Shares, inception Nov 2004, ~130-153 billion USD net assets --
largest gold ETF.

Mechanism (verbatim from the catalog): "Academic analysis of gold
returns 1980-2010 found September and November are the only calendar
months with statistically significant positive average returns (Indian
wedding-season/Diwali demand, equity-market Halloween-effect hedging).
Holds GLD long from September 1 through November 30 each year, flat/cash
the rest of the year."

Rule, restated precisely enough to reimplement (the catalog's five
PARAMETERS: entry_month=9, entry_day=1, exit_month=11, exit_day=30,
rebalance="annual"):

    For every calendar year Y touching GLD's own bar history:
        entry_target(Y) = September 1 of year Y   (entry_month/entry_day)
        exit_target(Y)  = November 30 of year Y    (exit_month/exit_day)
        in_season(bar)  = entry_target(Y) <= bar_date <= exit_target(Y),
                          where Y = bar_date.year (the window never
                          crosses a year boundary, so this is
                          unambiguous)
    entry = in_season(next bar in GLD's own (truncated) calendar)
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
    exit  = in_season(bar) AND NOT in_season(next bar in GLD's own
                                              (truncated) calendar)
                                                     [fills at THIS bar's
                                                       close]
    every other bar: entry=False, exit=False (flat/cash)

This is a pure CALENDAR rule -- no GLD price ever enters the signal, and
the year-Y window dates are computable from the calendar alone (no
"trailing return", no external ticker), so there is no warm-up
requirement the way a returns-based rule needs a prior observation.

This family is SIMPLER than gasoline_driving_season_seasonal.py (its own
reference template solving the same class of problem): both boundaries
(Sep 1, Nov 30) are LITERAL calendar dates, not a "nearest weekday" or
"last Monday of <month>" derived rule -- November always has 30 days, so
exit_target(Y) is simply "November 30 of year Y", a real calendar date
every year. No last_monday_of_may-style helper is needed.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- entry_target/exit_target are pure date arithmetic
  (never derived from price data); the only history-dependent fact
  computed is which of GLD's OWN (truncated) bar dates fall in-season.
* PERFORMANCE -- entry/exit are two precomputed boolean arrays; the
  returned signal_fn does two numpy lookups by position plus one scalar
  date check.
* HOLDOUT-BLIND -- GLD's frame is truncated to <= the last bar of
  ``index`` before anything is computed, so a holdout bucket's signals
  cannot depend on data after the bucket.
* DIRECTION -- always long GLD when in-season, flat (cash) otherwise;
  never short.

Interpretation choices (the pseudocode leaves these to the implementer):

1. Both "entry ... September 1" and "exit ... November 30" are read
   literally as calendar dates, not "the first/last trading day of the
   month": if the target date itself is not a trading day (weekend, or a
   rare weekday closure), the season boundary simply falls on the
   nearest trading day the ``in_season`` predicate naturally selects --
   entry on the FIRST available session on/after Sep 1, exit on the LAST
   available session on/before Nov 30. Unlike gasoline's Memorial Day
   exit, Nov 30 is not itself a recurring market holiday, so on most
   years the exit bar IS Nov 30 exactly when Nov 30 is a weekday.
2. Entry is a ONE-BAR-EARLY shift of ``in_season`` (``entry[t] =
   in_season[t+1]`` in the ticker's own truncated bar sequence), not
   ``in_season[t]`` itself -- same reasoning as gasoline's interpretation
   choice 3 (reused verbatim here, not re-derived): it lands the actual
   fill (always placed at the bar AFTER the flagged one) exactly on the
   first in-season bar's own open, and it goes False on the season's own
   last bar (whose successor is off-season), so there is no flag left
   over to trigger an immediate re-entry the session after exit.
3. Exit fires on the LAST bar whose own date is <= exit_target(Y), i.e.
   the bar immediately preceding the first bar whose date exceeds
   exit_target(Y) -- derived directly from the ``in_season`` predicate,
   same construction as gasoline's exit rule.
4. year(bar) is simply the bar's own calendar year: the season never
   crosses a year boundary (September through November), so there is no
   end-of-December / New Year ambiguity to resolve.
5. The FINAL bar of the (truncated) GLD history never raises an exit
   flag on its own account, by the same reasoning as every other
   calendar-only candidate in this batch: without a following bar the
   module cannot know whether the season has ended, and the engine's own
   end_of_data path force-closes any still-open position at that same
   close anyway, so no P&L is changed by leaving it undetermined (only
   the recorded exit reason, "end_of_data" instead of "signal_exit").
6. A non-finite or missing Close is never consulted by this rule at all
   (it is pure calendar arithmetic on bar DATES), so a bad bar's Close
   cannot corrupt entry/exit; the module still validates that a Close
   column exists (contract hygiene, matching every other batch5 module)
   but never reads its values for the signal itself.
7. GLD's REAL inception (Nov 18, 2004) falls INSIDE the 2004
   Sep1-Nov30 window, so GLD's own bar history (traded_dates) itself
   starts mid-season rather than at a fresh entry_target. This reuses
   the SAME mid-season-entry interpretation gasoline's own choice 3
   established for a holdout bucket opening mid-season: since
   season_decisions is computed directly on traded_dates (whatever its
   own first date is), a first bar that is already in-season still
   raises entry[0] = in_season[1] (True, as long as the second bar is
   also in-season) -- there is no explicit "was this the season's
   nominal start" check, only "is the NEXT bar in-season". This lets the
   partial 2004 season enter at GLD's own first trading day rather than
   being skipped entirely (this module's own test covers a history that
   starts on 2004-11-18; see also the EXPECTED_MIN_TRADES comment below).
   Conversely,
   if a ticker's history began entirely AFTER a would-be season's
   exit_target, in_season_mask simply returns False for every date in
   that gap (it only ever evaluates dates that exist in the array), so
   no phantom season is invented -- the next real season is picked up
   normally once the calendar reaches its own entry_target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "gold_autumn_effect_seasonal"
SHAPE = "single_long"
TRADED = ["GLD"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "GLD"
# Copied byte-for-byte from the pre-registered catalog entry's "parameters"
# (driver drift check); this module's own test compares against a frozen
# literal copy of that record.
PARAMETERS = {
    "entry_month": 9,
    "entry_day": 1,
    "exit_month": 11,
    "exit_day": 30,
    "rebalance": "annual",
}
# GLD's very first bar (2004-11-18) falls INSIDE that year's Sep1-Nov30
# window (interpretation choice 7), so the 2004 season DOES enter as a short
# partial position rather than being unreachable.
# Structurally the rule gives a short 2004 partial season
# (reason=signal_exit), one full ~62-session round trip per later season
# (all reason=signal_exit), and, when the history ends inside a season, one
# final in-progress season (reason=end_of_data). Under an approximate
# 6-month holdout ending at the history's last date, only the final
# season falls inside the holdout window; all earlier seasons are entirely
# before it. EXPECTED_MIN_TRADES is set below that one-per-year count
# as a safety margin against small holdout-boundary shifts, same
# convention as the gasoline template:
EXPECTED_MIN_TRADES = 18

_ENTRY_MONTH = int(PARAMETERS["entry_month"])
_ENTRY_DAY = int(PARAMETERS["entry_day"])
_EXIT_MONTH = int(PARAMETERS["exit_month"])
_EXIT_DAY = int(PARAMETERS["exit_day"])
if PARAMETERS["rebalance"] != "annual":
    raise ValueError(
        f"{FAMILY_ID}: unsupported rebalance {PARAMETERS['rebalance']!r}")


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


def in_season_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """True where ``dates[k]`` falls on/after September 1 and on/before
    November 30 of its OWN calendar year (interpretation choices 1, 4).
    ``dates`` must already be tz-naive and normalized (not required to be
    sorted/unique -- this is a pure elementwise predicate)."""
    n = len(dates)
    if n == 0:
        return np.zeros(0, dtype=bool)
    years = dates.year.to_numpy()
    entry_target = _dates_from_ymd(years, _ENTRY_MONTH, _ENTRY_DAY)
    exit_target = _dates_from_ymd(years, _EXIT_MONTH, _EXIT_DAY)
    dates_arr = dates.to_numpy()
    return ((dates_arr >= entry_target.to_numpy())
            & (dates_arr <= exit_target.to_numpy()))


def season_decisions(traded_dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Precompute (entry, exit) boolean Series on GLD's own (already
    truncated) bar calendar. ``traded_dates`` must be tz-naive, normalized,
    sorted and unique.

    entry[t] = in_season[t+1] -- True one bar BEFORE every in-season bar
    (choice 2), which is what lands the engine's next-bar-open fill on the
    season's own first open and what stops the flag re-firing the session
    right after exit. exit[t] = in_season[t] and not in_season[t+1] --
    True only on the LAST in-season bar of each season block (choice 3).
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
