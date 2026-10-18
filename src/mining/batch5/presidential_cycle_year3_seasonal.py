"""New candidate: presidential_cycle_year3_seasonal (single_long).

Calendar-only seasonal rotation on SPDR S&P 500 ETF Trust (SPY): long SPY
for the ENTIRE calendar year of "Year 3" of the 4-year U.S. presidential
term -- the pre-election year -- flat/cash every other year. Repeats once
every 4 years ("rebalance": "quadrennial" in PARAMETERS), never within a
single calendar year.

Source: Wong, W.-K. & McAleer, M. (2009), "Mapping the Presidential
Election Cycle in US Stock Markets", Mathematics and Computers in
Simulation, 79(11), 3267-3277, DOI 10.1016/j.matcom.2009.05.007 (DOI
verified) -- using weekly S&P 500 index data 1965-2003,
finds that US equity prices systematically peak in the third and fourth
years of a presidential term and are weakest in the first two, and
identifies the presidential election cycle as the single most dominant
cycle detected in the data (their spectral/periodicity analysis compares
it against several other candidate cycles). This is this family's
PRIMARY, falsifiable, peer-reviewed citation.

Corroborating practitioner source (background only, not itself the
falsifiable citation): Yale Hirsch's "Stock Trader's Almanac" four-year
cycle tables (popularized since the 1960s) report Year 3 (the pre-election
year) as the strongest of the four cycle years by a wide margin --
period-dependent secondary compilations put Year 3's average S&P
500/Dow return at roughly 15-17% versus single-digit averages for Years
1, 2 and 4 (figures vary by source window and compilation). The standard
explanation in this literature is that
incumbent administrations and the Federal Reserve have an incentive to
front-load stimulative fiscal/monetary policy in the run-up to a
re-election year, concentrating economic tailwinds in Year 3 -- an
INCUMBENT-INCENTIVE mechanism, not a claim about either political party,
so this family's rule (a fixed arithmetic year-of-cycle test) never reads
which party or individual holds office.

Ticker verified: SPY, SPDR S&P 500 ETF Trust, State Street, inception
1993-01-29.

Mechanism
---------

    US presidential elections fall in years divisible by 4 (2024, 2028,
    ...). "Year 3" of a term -- the pre-election year -- is therefore any
    calendar year Y where (Y - 3) % 4 == 0, equivalently Y % 4 == 3
    (1995, 1999, 2003, ..., 2023, 2027, ...).
    Hold SPY long for every trading day whose OWN calendar year is a
    Year-3 year; flat/cash every other calendar year. Rebalances (enters
    or exits) at most twice per 4-year cycle -- once into the season,
    once out -- never mid-year.

DISTINCTNESS NOTE (verbatim mechanism text from the three nearest
catalog siblings by theme -- broad-market calendar/seasonal equity rules,
from data/mining/catalog.json; no other family in this catalog keys off
any multi-year political or election cycle at all):

* ``halloween_indicator_seasonal`` -- "Hold equity index ETF exposure
  only November-April (the 'winter' half of the calendar year), flat or
  in cash May-October." An ANNUAL, EVERY-YEAR six-month split -- this
  family fires in only ONE of every FOUR calendar years, keyed to the
  U.S. presidential election calendar, not a month-of-year boundary at
  all, and holds the WHOLE calendar year (roughly 250 trading days), not
  a recurring half-year window.
* ``santa_claus_rally_yearend`` -- "Go long a broad equity index ETF at
  the close of the 6th-to-last trading day of the calendar year (i.e.,
  entering before the last 5 trading days of December) and hold through
  the close of the 2nd trading day of the new January, then exit to cash
  -- a fixed 7-trading-day seasonal window each year end." An EVERY-YEAR,
  7-TRADING-DAY window straddling New Year's -- this family's window is
  an entire calendar year, occurs only once per 4 years, and by
  construction NEVER straddles a year boundary (a Year-3 season starts
  and ends within the same single calendar year).
* ``january_small_cap_seasonal`` -- "Hold small-cap equity exposure only
  during the calendar month of January (positions opened at the January
  1st trading session's open or the prior December close, closed at
  January's final trading session's close); stay in cash or a risk-free
  proxy the rest of the year." An EVERY-YEAR, ONE-MONTH, SMALL-CAP-
  specific window attributed to a post-tax-loss-selling rebound -- this
  family trades the BROAD-MARKET SPY, holds a full calendar year, fires
  only once every 4 years, and its driver (Wong & McAleer's documented
  political-economic cycle in aggregate market prices) has nothing to do
  with tax-loss selling or any single calendar month.

None of the three, nor any other catalog entry (verified by the grep
above), expresses a rule whose periodicity is 4 CALENDAR YEARS rather
than 1 -- every other seasonal/calendar family in this catalog recurs
every single year (or every week/month/quarter within a year). This
family is the only one keyed to WHICH of the four years within a
recurring 4-year political cycle a given calendar year is.

Engine mapping (mining.backtest_engine.run_backtest)
-----------------------------------------------------
Same calendar-precomputed-signal shape as gold_autumn_effect_seasonal.py
and natural_gas_withdrawal_season_seasonal.py: entry/exit are two boolean
arrays precomputed once per call to make_signal_fn's factory, never a
rolling/time_stop mechanism, so the ABI-1 driver's own run_backtest call
(without a time_stop argument -- verified by reading abi1_driver.py, no
time_stop= appears anywhere in that module) needs no extra machinery from
this module.

Interpretation choices
-----------------------
1. Both the season's start and end are read as literal calendar-year
   boundaries (Jan 1 through Dec 31 of a Year-3 year), never "first/last
   trading day of the month" the way gasoline's Memorial Day rule is --
   if Jan 1 is not itself a trading day the season's first bar is simply
   the first trading day on/after it, and if Dec 31 is not a trading day
   the season's last bar is the last trading day on/before it; this falls
   directly out of in_season_mask being a pure per-bar predicate on
   dates.year, same construction as gold_autumn's month/day predicate.
2. Entry is a ONE-BAR-EARLY shift of in_season (entry[t] =
   in_season[t+1]); exit fires on the LAST in-season bar (exit[t] =
   in_season[t] and not in_season[t+1]) -- identical convention and
   identical reasoning to gold_autumn_effect_seasonal.season_decisions
   (reused verbatim here, not re-derived): it lands the actual fill
   (always the bar AFTER the flagged one) on the season's own first open,
   and it never re-raises entry on the season's own last bar.
3. A Year-3 season never crosses a calendar-year boundary by construction
   (it IS a calendar year), so unlike gold_autumn there is no cross-year
   edge case to resolve for the season's own definition; dates.year is
   read directly, with no separate entry_target/exit_target date
   arithmetic needed.
4. The FINAL bar of the (truncated) SPY history never raises an exit flag
   on its own account (there is no known "next bar" to compare in-season
   status against) -- the engine's own end_of_data path force-closes any
   still-open position at that close, so no P&L changes, only the
   recorded exit reason ("end_of_data" instead of "signal_exit").
5. A non-finite or missing Close is never consulted by this rule at all
   (it is pure calendar arithmetic on bar DATES), so a bad bar's Close
   cannot corrupt entry/exit; the module still validates that a Close
   column exists (contract hygiene, matching every other batch5 module)
   but never reads its values for the signal itself.
6. SPY's REAL inception (1993-01-29) falls in 1993 (1993 % 4 == 1, an
   OFF-season "Year 1"), so SPY's own real bar history does NOT begin
   mid-season -- unlike GLD's real inception in
   gold_autumn_effect_seasonal, there is no partial first season in
   SPY's trade count below (see EXPECTED_MIN_TRADES). The
   mid-season-entry code path (entry[0] = in_season[1] even when a
   ticker's OWN history begins already in-season) is still exercised and
   tested with synthetic data for correctness, because make_signal_fn
   must handle it correctly for any holdout bucket that opens mid-season
   even though SPY's full, untruncated history itself never does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "presidential_cycle_year3_seasonal"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL: list = []  # self-contained: only SPY's own OHLC is ever read
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {
    "cycle_length_years": 4,
    "election_year_modulus": 0,
    "target_year_offset_from_election": 3,
    "rebalance": "quadrennial",
}
CYCLE_LENGTH_YEARS = int(PARAMETERS["cycle_length_years"])
ELECTION_YEAR_MODULUS = int(PARAMETERS["election_year_modulus"])
TARGET_YEAR_OFFSET = int(PARAMETERS["target_year_offset_from_election"])
if PARAMETERS["rebalance"] != "quadrennial":
    raise ValueError(
        f"{FAMILY_ID}: unsupported rebalance {PARAMETERS['rebalance']!r}")
if CYCLE_LENGTH_YEARS <= 0:
    raise ValueError(f"{FAMILY_ID}: cycle_length_years must be positive")

# One trade per Year-3 calendar year fully contained in SPY's history
# (1995, 1999, ..., 2023 by calendar arithmetic alone: 8 such years from
# SPY's 1993 inception), each entered on the year's first session and
# exited on its last. This is a genuinely low-frequency family BY
# CONSTRUCTION (the underlying phenomenon recurs once every 4 years, not
# every year), so the trade count is small regardless of the data; this
# module deliberately does not attempt to score itself. Set a bit below
# the calendar count as a conservative floor:
EXPECTED_MIN_TRADES = 6


def _normalize_index(index) -> pd.DatetimeIndex:
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


def in_season_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """True where ``dates[k]``'s OWN calendar year is a "Year 3" (pre-
    election) year of the 4-year U.S. presidential cycle, i.e.
    ``(year - TARGET_YEAR_OFFSET) % CYCLE_LENGTH_YEARS ==
    ELECTION_YEAR_MODULUS``. Pure calendar arithmetic -- no price data,
    no sub-year month/day boundary."""
    n = len(dates)
    if n == 0:
        return np.zeros(0, dtype=bool)
    years = dates.year.to_numpy()
    return ((years - TARGET_YEAR_OFFSET) % CYCLE_LENGTH_YEARS
             == ELECTION_YEAR_MODULUS)


def season_decisions(traded_dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Precompute (entry, exit) boolean Series on SPY's own (already
    truncated) bar calendar. ``traded_dates`` must be tz-naive,
    normalized, sorted and unique.

    entry[t] = in_season[t+1] -- True one bar BEFORE every in-season bar,
    which is what lands the engine's next-bar-open fill on the season's
    own first open and what stops the flag re-firing the session right
    after exit. exit[t] = in_season[t] and not in_season[t+1] -- True
    only on the LAST in-season bar of each Year-3 season. Neither is ever
    True on the final bar of the whole array -- there is no known "next
    bar" to compare against without looking past the truncation cutoff."""
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

    bucket = _normalize_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    traded_dates = _normalize_index(traded_frame.index)
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
