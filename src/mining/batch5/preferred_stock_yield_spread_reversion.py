"""Batch5 candidate: preferred-stock ETF trailing-yield z-score reversion,
pooled (family ``preferred_stock_yield_spread_reversion``, shape
``pooled_single_asset``; pool PFF / PGX / PFFD). The catalog mechanism names
PFF; this module runs the identical rule on each of three preferred-stock
ETFs and pools the trades.

Rule (applied independently to every pool member; the driver calls
``make_signal_fn`` once per ticker):

    Track the fund's trailing 12-month distribution yield and compute a
    z-score vs its own trailing 3-year rolling mean/std, recomputed
    monthly. Go long when the yield z-score rises to 0.75+ above its own
    average (price depressed relative to its own recent yield regime);
    exit to cash when the z-score falls back to zero or below. Run the
    identical rule on each pool member using that member's OWN
    trailing-12-month distribution yield (yfinance Ticker.dividends,
    ex-dates as-of) and its own 3-year rolling z-score; pool the trades.

Parameters (byte-identical to the catalog entry): yield_lookback_years=3,
entry_zscore=0.75, exit_zscore=0, rebalance="monthly".

Data (committed with provenance, never touched by tests):
``data/mining/batch5_events/preferred_stock_yield_spread_reversion/
distributions_and_unadjusted_close.json`` holds, per pool member, the
per-share cash distribution on its EX-DATE row and the UNADJUSTED daily
close. Regenerate it with ``scripts/fetch_batch5_events_
preferred_stock_yield_spread_reversion.py``, which pulls amounts/closes from
``yfinance Ticker.history(period="max", auto_adjust=False)`` and then
reconciles the distribution calendar against INDEPENDENT publishers
(digrin.com, dividendhistory.org, stockanalysis.com, and the Nasdaq dividend
API for the Nasdaq-listed member) -- see "Data hygiene" below.
``provenance.json`` records source, fetched_at, regenerated_by, the
independent sources and the full reconciliation result.

Engine mapping (mining.backtest_engine): an ``entry`` flag on bar t fills at
bar t+1's open, an ``exit`` flag on bar t fills at bar t's close. The
standing LONG/FLAT state is precomputed once per member in the factory from
rolling / shifted operations only; ``entry[t] = LONG[t]`` and
``exit[t] = not LONG[t]``; both are reindexed as-of onto the bucket and the
per-bar signal function is two array reads (never touches the slice).

Interpretation choices
----------------------
Every choice below applies identically to each pool member, so the pool runs
the SAME rule as the catalog's single-fund mechanism, not a redefinition:

1. "Trailing 12-month distribution yield" = trailing 252-trading-day SUM of
   per-share cash distributions (ex-date dated) divided by that day's
   UNADJUSTED close. The engine's frame carries a dividend-ADJUSTED close
   (data_pipeline.fetch_ohlc, auto_adjust=True), which is back-adjusted for
   every FUTURE payout: for a ~6%-yielder that inflates past yields by
   ~17-20% across any 3-year window and biases the z-score negative. The
   yield therefore uses the unadjusted close from the data file; the
   engine's adjusted frame is used only for fills (as every other family).
   A full 252-observation window is required (NaN before), so a fund's
   first year never shows an understated yield.
2. z-score = (yield - rolling mean) / rolling std over 3 x 252 = 756
   trading days of the yield series, full window required (NaN before,
   ddof=1). A zero rolling std (below a 1e-9 relative float-noise floor)
   gives no decision (NaN), never a signal.
3. "Recomputed monthly" = the z-score is CHECKED on the first trading bar
   of each calendar month (the fund's own calendar); state changes only
   there and carries forward between checks (hysteresis: enter at
   z >= 0.75, exit at z <= 0, hold inside the band). A NaN z on a check
   bar means FLAT (no history yet).
4. "Exit to cash" = flat; no cash instrument is priced (single-instrument
   engine, same simplification every other regime candidate uses).
5. Bucket edges: a bucket that OPENS inside a LONG regime enters at its
   first fill opportunity (entry stays raised while LONG); the exit flag is
   raised on EVERY non-LONG bar so a position the engine opened one bar
   after the decision (its ``continue`` skips the exit check on the fill
   bar) is still closed at the next FLAT bar.
6. A distribution whose ex-date is not on the close calendar counts from
   the first close ON OR AFTER it (never before). Inputs are truncated to
   dates <= the bucket's last bar before any computation, so a holdout
   bucket structurally cannot see later data.

Data hygiene
------------
The trailing-yield numerator is a SUM, so a distribution the vendor never
published is not noise -- it is a hole that depresses the yield for a full
252 sessions and drags the z-score with it. Yahoo's
``history()['Dividends']`` column and Yahoo's own ``Ticker.dividends``
endpoint agree with each other, yet both lack FIVE real monthly
distributions that independent publishers list (PFF 2012-11-01, PGX
2016-03-15 / 2016-04-15 / 2021-09-20, PFFD 2017-11-01). A vendor-vs-itself
comparison cannot see that class of defect.

a. Completeness is therefore established against INDEPENDENT publishers by
   the fetcher script, and it FAILS CLOSED: an ex-date that at least two
   independent sources publish and the vendor lacks is merged in at the most
   precise agreeing amount; a single-source event is NOT added; a candidate
   within five days of an event the vendor already carries is rejected as a
   publisher date-shift, never merged. All five events above were confirmed
   on two or three independent sources each.
b. The same completeness invariants are re-checked HERE, on every load, by
   ``validate_distribution_calendar``: for a monthly distributor, every
   calendar month inside the series' span must carry a distribution, and
   every COMPLETE calendar year must carry exactly twelve. The one benign
   pattern in these funds' real histories is a year-end distribution pulled
   forward into late December, so December carries two ex-dates and the
   following January carries none -- a month whose predecessor has two or
   more ex-dates is explained; every other empty month raises. A monthly
   distributor with a missing month is an error, not a zero.
b2. Those two checks both derive their span FROM the dividend series, so a
   distribution dropped at the END of the series moves the span end with it
   and neither check can see it -- the file passes while its newest month is
   missing, and the newest months are precisely the holdout window,
   where a hole moves the trailing yield and
   therefore the holdout trades. ``validate_distribution_calendar`` therefore
   also takes the member's own LAST PRICE BAR and requires a distribution in
   every month from the one after the last ex-date through the last COMPLETE
   calendar month before that bar (``tail_gaps``), same pull-forward
   exemption, same fail-closed raise. Only complete months are required --
   the identical "partial units are not evidence" rule the per-year check
   uses -- so the residual blind spot is bounded at the one calendar month
   containing the final price bar. Requiring that partial month too would
   need an invented day-of-month threshold, and a threshold that fired early
   would fail-close a healthy member on a schedule shift; a bounded
   one-month blind spot is the safer error. Every member of the committed
   data file passes this check.
c. Only CASH distributions enter the sum. The fetcher reads Yahoo's
   ``Dividends`` column and never ``Capital Gains`` or ``Stock Splits``: a
   capital-gains distribution and a split-type adjustment are not cash
   distribution yield. (All three members report zero capital-gains rows
   over their full history in the committed file, so this is a standing guard,
   not a data edit.)
c2. The guard is scoped PER MEMBER, not per file. ``load_inputs`` takes a
   ``members`` argument and the factory passes only the ticker it is building,
   because the driver turns any exception out of ``make_signal_fn`` into a
   whole-FAMILY ``compute_error`` skip. Validating all three members on each
   of the three per-member loads would therefore convert one fund's data
   defect into a family-wide failure -- backwards from the driver's own rule
   that a pooled family excludes an unusable member and scores the remainder.
   With the scoping, a broken calendar in one fund raises only for THAT
   fund (the other two still load clean), and the driver excludes a
   pooled member whose own backtest raises rather than failing the
   family. That does not let this pool survive the loss: the
   pool is THREE funds, so losing one leaves 67% coverage, under the
   driver's MIN_POOL_COVERAGE floor, and the family is refused as
   pool_coverage -- charging no trial and staying eligible. Isolation
   protects a LARGE pool; a three-member pool still needs all three.
d. NOT changed, and visible rather than silent: Yahoo publishes these three
   funds' amounts at 3 decimals, so a vendor amount can differ from an
   independent publisher by a rounding step. The fetcher cross-checks every
   shared ex-date against the closest independent amount and writes every
   disagreement beyond one rounding step into the provenance -- report-only.
   One event is flagged that way and is DELIBERATELY LEFT at the vendor's
   value: PFF 2020-02-03, vendor 0.16 against 0.16376 on dividendhistory and
   nasdaq and 0.1638 on digrin. The line this code draws is between a
   STRUCTURAL defect and a disputed amount. A missing ex-date makes the
   numerator a sum of eleven months called a twelve-month sum, so filling it
   from independent publishers restores the quantity the rule names; the
   rule above also names ``yfinance Ticker.dividends`` as the amount
   source, and overwriting a figure that vendor actually published would
   substitute this code's judgement for a pre-registered input. So the
   disagreement is surfaced, never silently overwritten.
   Size of the disputed amount: the trailing 12-month sum is understated by
   0.00376 per share -- 0.177% to 0.210% of the sum -- on exactly 252
   sessions, 2020-02-03 through 2021-02-01 (the z-score inherits it through
   the 756-session z window). It is also structurally outside the holdout:
   the perturbation ends years before the holdout window opens, so it cannot
   reach a holdout trade at all.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "preferred_stock_yield_spread_reversion"
SHAPE = "pooled_single_asset"
TRADED = ["PFF", "PGX", "PFFD"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "PFF"
# Copied from the catalog entry's "parameters"; the driver rejects any drift
# as canonical JSON.
PARAMETERS = {
    "yield_lookback_years": 3,
    "entry_zscore": 0.75,
    "exit_zscore": 0,
    "rebalance": "monthly",
}
EXPECTED_MIN_TRADES = 30

# Every pool member is a MONTHLY distributor (verified against each fund's
# published distribution frequency and against the independent publishers the
# fetcher reconciles with). Used only for the data-completeness guard below --
# it is not a rule parameter and never enters a signal.
PAYMENTS_PER_YEAR = 12

TRAILING_YIELD_TRADING_DAYS = 252
ZSCORE_WINDOW_TRADING_DAYS = TRAILING_YIELD_TRADING_DAYS * int(
    PARAMETERS["yield_lookback_years"])
ENTRY_ZSCORE = float(PARAMETERS["entry_zscore"])
EXIT_ZSCORE = float(PARAMETERS["exit_zscore"])

_REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "mining", "batch5_events", FAMILY_ID)
DATA_PATH = os.path.join(DATA_DIR, "distributions_and_unadjusted_close.json")
PROVENANCE_PATH = os.path.join(DATA_DIR, "provenance.json")


# ------------------------------------------------------------- helpers

def _tz_naive(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _clean_positive_series(series: pd.Series, name: str) -> pd.Series:
    out = pd.Series(series, copy=True)
    out.index = _tz_naive(out.index)
    out = pd.to_numeric(out, errors="coerce").astype(float)
    if out.isna().any() or not np.isfinite(out.to_numpy()).all():
        raise ValueError(f"{name} must be finite numeric")
    if (out <= 0).any():
        raise ValueError(f"{name} must be strictly positive")
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"{name} index must be unique")
    return out


def _month_key(stamp) -> str:
    stamp = pd.Timestamp(stamp)
    return f"{stamp.year:04d}-{stamp.month:02d}"


def _previous_month(key: str) -> str:
    year, month = int(key[:4]), int(key[5:7])
    month -= 1
    if month == 0:
        year, month = year - 1, 12
    return f"{year:04d}-{month:02d}"


def _next_month(key: str) -> str:
    year, month = int(key[:4]), int(key[5:7])
    month += 1
    if month == 13:
        year, month = year + 1, 1
    return f"{year:04d}-{month:02d}"


def _month_span(first: str, last: str) -> list:
    year, month = int(first[:4]), int(first[5:7])
    end = (int(last[:4]), int(last[5:7]))
    out = []
    while (year, month) <= end:
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def distribution_calendar_problems(dividends: pd.Series, *,
                                   per_year: int = PAYMENTS_PER_YEAR,
                                   last_session=None) -> dict:
    """Completeness problems in a MONTHLY distributor's ex-date series --
    ``{"month_gaps": [...], "short_years": {year: count}, "tail_gaps": [...]}``
    (all three empty when the series is complete).

    ``month_gaps`` / ``short_years`` are INTERIOR checks, bounded by the
    dividend series' own span: a month inside that span with no ex-date is a
    gap UNLESS the preceding month carries two or more (the real year-end
    pull-forward: a two-ex-date December followed by an empty January), and a
    complete calendar year -- one strictly between the first and last event
    years, so neither the partial inception year nor the partial current year
    -- must carry exactly ``per_year`` events. See the module's "Data hygiene"
    section: Yahoo lacks five real monthly distributions that independent
    publishers list, and the trailing yield is a SUM, so a hole depresses it
    for 252 sessions.

    ``tail_gaps`` closes the blind spot those two share. Both derive their
    span from the dividend series itself, so a distribution dropped at the
    END of the series moves the span end with it and is invisible -- the file
    looks complete while its newest month is missing, which is exactly where
    the holdout window sits. Pass ``last_session`` (the last row of the
    member's own price series) and every month from the one after the last
    ex-date through the last COMPLETE calendar month before ``last_session``
    is required too, with the same pull-forward exemption. Only complete
    months are required -- the same "partial units are not evidence" rule the
    per-year check already uses -- so the residual blind spot is bounded at
    the single calendar month containing ``last_session`` (a genuinely
    not-yet-due ex-date is indistinguishable from a dropped one without
    inventing a day-of-month threshold, and a threshold that fires early
    would fail-close a member for a schedule shift). Together with the
    factory's own "close data must reach the bucket's last bar" check, that
    bounds staleness in BOTH series."""
    keys = sorted(_month_key(stamp) for stamp in pd.DatetimeIndex(
        pd.to_datetime(list(dividends.index))))
    if not keys:
        return {"month_gaps": [], "short_years": {}, "tail_gaps": []}
    # A ONE-event series is not exempt. The interior checks are vacuous on it
    # (a one-month span holds its own event; no year is complete), but the
    # tail check is not -- and "the tail collapsed to a single event" is the
    # same defect as "the tail lost its newest month", just further along.
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1

    def _unexplained(key: str) -> bool:
        # empty month, and its predecessor did not pull it forward
        return not counts.get(key) and counts.get(_previous_month(key), 0) < 2

    gaps = [key for key in _month_span(keys[0], keys[-1]) if _unexplained(key)]
    per_year_counts = {}
    for key in keys:
        year = int(key[:4])
        per_year_counts[year] = per_year_counts.get(year, 0) + 1
    first_year, last_year = int(keys[0][:4]), int(keys[-1][:4])
    short = {year: per_year_counts.get(year, 0)
             for year in range(first_year + 1, last_year)
             if per_year_counts.get(year, 0) != per_year}
    tail = []
    if last_session is not None:
        # _month_span returns [] when the start is past the end, so a series
        # already current needs no special case.
        tail = [key for key in _month_span(
            _next_month(keys[-1]),
            _previous_month(_month_key(last_session)))
            if _unexplained(key)]
    return {"month_gaps": gaps, "short_years": short, "tail_gaps": tail}


def validate_distribution_calendar(dividends: pd.Series, name: str, *,
                                   per_year: int = PAYMENTS_PER_YEAR,
                                   last_session=None) -> None:
    """Raise ``ValueError`` when ``dividends`` has an unexplained interior
    month gap, a complete calendar year off ``per_year`` events, or (when
    ``last_session`` is given) a missing TAIL month. Fail-closed on purpose,
    and identically for all three: a monthly distributor with a missing month
    is an error, not a zero, and a missing newest month is the one that lands
    inside the holdout window."""
    problems = distribution_calendar_problems(
        dividends, per_year=per_year, last_session=last_session)
    if (problems["month_gaps"] or problems["short_years"]
            or problems["tail_gaps"]):
        raise ValueError(
            f"{name} distribution calendar is incomplete: month(s) with no "
            f"distribution {problems['month_gaps']}, complete calendar "
            f"year(s) not at {per_year} events {problems['short_years']}, "
            f"month(s) missing at the END of the series (stale tail, through "
            f"the last complete month before the final price bar) "
            f"{problems['tail_gaps']}. "
            "These funds distribute monthly and the trailing yield is a sum, "
            "so a missing month understates the yield for a full 252 "
            "sessions -- regenerate the data file with scripts/"
            f"fetch_batch5_events_{FAMILY_ID}.py (it reconciles the calendar "
            "against independent publishers) rather than accepting a zero.")


def load_inputs(path: str = DATA_PATH, *, members=None) -> dict:
    """{ticker: {"unadjusted_close": Series, "dividends": Series}} from the
    committed data file (dates ISO strings -> tz-naive DatetimeIndex).

    ``members`` scopes the load to the named pool members (default: every
    member in the file). This is what keeps a POOLED family a pool: the
    validation below fails closed, and the driver treats an exception out of
    ``make_signal_fn`` as a whole-FAMILY compute error, so validating all
    three members on every per-member load would turn one fund's data defect
    into a family-wide failure -- the opposite of the driver's own rule that a
    pooled family drops an unusable member and scores on the remainder.

    Fails closed on an incomplete distribution calendar, tail included -- see
    ``validate_distribution_calendar`` and the module's "Data hygiene"
    section."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if members is None:
        wanted = list(payload)
    else:
        wanted = list(members)
        missing = [ticker for ticker in wanted if ticker not in payload]
        if missing:
            raise ValueError(
                f"{', '.join(missing)}: no distribution/close inputs in {path}")
    inputs = {}
    for ticker in wanted:
        block = payload[ticker]
        close = pd.Series(block["unadjusted_close"], dtype=float)
        close.index = pd.DatetimeIndex(pd.to_datetime(list(close.index)))
        close = _clean_positive_series(close, f"{ticker} unadjusted_close")
        dividends = pd.Series(block["dividends"], dtype=float)
        dividends.index = pd.DatetimeIndex(pd.to_datetime(list(dividends.index)))
        dividends = _clean_positive_series(dividends, f"{ticker} dividends")
        # last_session is this member's own last price bar: a distribution
        # dropped at the END of the series is invisible to a span derived
        # from the series itself.
        validate_distribution_calendar(dividends, ticker,
                                       last_session=close.index[-1])
        inputs[ticker] = {"unadjusted_close": close, "dividends": dividends}
    return inputs


# --------------------------------------------------------- computation

def _distributions_on_calendar(dividends: pd.Series,
                               calendar: pd.DatetimeIndex) -> pd.Series:
    """Per-share distributions summed onto ``calendar``; an ex-date that is
    not a session counts on the first session ON OR AFTER it (causal), and
    one after the last session is dropped."""
    out = pd.Series(0.0, index=calendar)
    if len(dividends) == 0:
        return out
    dates = _tz_naive(dividends.index)
    positions = calendar.searchsorted(dates, side="left")
    keep = positions < len(calendar)
    if keep.any():
        binned = pd.Series(dividends.to_numpy(dtype=float)[keep],
                           index=calendar[positions[keep]]).groupby(level=0).sum()
        out.loc[binned.index] = out.loc[binned.index] + binned
    return out


def trailing_distribution_yield(unadjusted_close: pd.Series, dividends: pd.Series,
                                *, trailing_days: int = TRAILING_YIELD_TRADING_DAYS
                                ) -> pd.Series:
    """Trailing ``trailing_days``-session sum of distributions divided by the
    same day's unadjusted close; NaN until a full window exists. Aligned to
    ``unadjusted_close.index``; a value at t uses only data dated <= t."""
    if trailing_days <= 0:
        raise ValueError("trailing_days must be positive")
    close = _clean_positive_series(unadjusted_close, "unadjusted_close")
    payouts = _distributions_on_calendar(pd.Series(dividends, dtype=float), close.index)
    trailing = payouts.rolling(trailing_days, min_periods=trailing_days).sum()
    return (trailing / close).rename("trailing_yield")


def trailing_yield_zscore(unadjusted_close: pd.Series, dividends: pd.Series, *,
                          trailing_days: int = TRAILING_YIELD_TRADING_DAYS,
                          zscore_window: int = ZSCORE_WINDOW_TRADING_DAYS
                          ) -> pd.Series:
    """Rolling z-score of the trailing yield over ``zscore_window`` sessions
    (full window required, ddof=1). A zero std is NaN (no decision)."""
    if zscore_window < 2:
        raise ValueError("zscore_window must be at least 2")
    yld = trailing_distribution_yield(unadjusted_close, dividends,
                                      trailing_days=trailing_days)
    mean = yld.rolling(zscore_window, min_periods=zscore_window).mean()
    std = yld.rolling(zscore_window, min_periods=zscore_window).std(ddof=1)
    # pandas' online rolling std returns float noise (~1e-17), not 0, for an
    # exactly constant window; a std below this relative floor is "no
    # dispersion" -> NaN (no decision), never a spurious +/-1e15 z-score.
    tolerance = 1e-9 * mean.abs() + 1e-15
    z = (yld - mean) / std.where(std > tolerance)
    return z.rename("yield_zscore")


def decision_bars(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the first bar of each calendar month (and on bar 0)."""
    codes = _tz_naive(index).to_period("M").asi8
    out = np.empty(len(codes), dtype=bool)
    if len(codes):
        out[0] = True
        out[1:] = codes[1:] != codes[:-1]
    return out


def standing_long(zscore: pd.Series, *, entry_z: float = ENTRY_ZSCORE,
                  exit_z: float = EXIT_ZSCORE) -> pd.Series:
    """Per-bar standing state (True = LONG) from the z-score series: checked
    only on decision bars (first bar of each calendar month), hysteresis
    enter at z >= entry_z / exit at z <= exit_z, hold inside the band, NaN
    -> FLAT; carried forward between checks. Sequential by construction, so
    the state at t depends only on decision bars <= t."""
    values = pd.to_numeric(pd.Series(zscore), errors="coerce").to_numpy(dtype=float)
    index = _tz_naive(zscore.index)
    checks = decision_bars(index)
    state = np.zeros(len(values), dtype=bool)
    is_long = False
    for i in range(len(values)):
        if checks[i]:
            z = values[i]
            if np.isnan(z):
                is_long = False
            elif z >= entry_z:
                is_long = True
            elif z <= exit_z:
                is_long = False
        state[i] = is_long
    return pd.Series(state, index=index, name="long")


def signal_flags(unadjusted_close: pd.Series, dividends: pd.Series) -> pd.DataFrame:
    """Boolean ``entry`` / ``exit`` columns aligned to the close calendar:
    entry[t] = standing LONG at t (fills t+1 open); exit[t] = not LONG at t
    (fills t close). See docstring choice 5 for why exit is the complement."""
    zscore = trailing_yield_zscore(unadjusted_close, dividends)
    longs = standing_long(zscore)
    return pd.DataFrame({"entry": longs.to_numpy(bool),
                         "exit": ~longs.to_numpy(bool)}, index=longs.index)


# ------------------------------------------------------------- factory

def make_signal_fn(ticker: str, index, frames, *, inputs: dict | None = None):
    """Driver factory (called once per pool member with that member as
    ``ticker``). ``index`` is the bucket calendar the engine will run over;
    ``frames[ticker]`` is that member's full-history OHLC frame (used for
    validation only -- fills happen in the engine; the yield uses the data
    file's unadjusted close, docstring choice 1). ``inputs`` defaults to the
    committed data file; tests inject synthetic series. Everything is
    precomputed here; the returned closure is O(1) per bar."""
    frame = frames[ticker]                       # KeyError if absent: fail closed
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{ticker}: frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ValueError(f"{ticker}: frame is missing Close")
    bucket = _tz_naive(index)
    if len(bucket) == 0:
        raise ValueError(f"{ticker}: bucket index is empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{ticker}: bucket index must be sorted and unique")

    if inputs is None:
        # Scoped to THIS member only: the driver turns any exception here into
        # a whole-family compute error, so loading (and validating) the other
        # pool members would let one fund's data defect take down the pool.
        # DATA_PATH is read at call time so a test can point it elsewhere.
        inputs = load_inputs(DATA_PATH, members=(ticker,))
    if ticker not in inputs:
        raise ValueError(f"{ticker}: no distribution/close inputs in {DATA_PATH}")
    close = _clean_positive_series(inputs[ticker]["unadjusted_close"],
                                   f"{ticker} unadjusted_close")
    dividends = pd.Series(inputs[ticker]["dividends"], dtype=float)
    last_bucket_bar = bucket[-1]
    if close.index[-1] < last_bucket_bar:
        raise ValueError(
            f"{ticker}: unadjusted close data ends {close.index[-1].date()}, "
            f"before the bucket's last bar {last_bucket_bar.date()} -- refetch "
            f"the data file")
    # Holdout-blind by construction: nothing after the bucket's last bar is
    # ever loaded into the computation.
    close = close.loc[close.index <= last_bucket_bar]
    if len(dividends):
        div_index = _tz_naive(dividends.index)
        dividends = pd.Series(dividends.to_numpy(dtype=float), index=div_index)
        dividends = dividends.loc[div_index <= last_bucket_bar]

    flags = signal_flags(close, dividends)
    aligned = flags.reindex(bucket, method="ffill")   # as-of the bucket date
    entry_flags = aligned["entry"].fillna(False).to_numpy(dtype=bool)
    exit_flags = aligned["exit"].fillna(True).to_numpy(dtype=bool)
    n = len(bucket)

    def signal_fn(_frame_slice, i: int) -> SignalResult:
        if i < 0 or i >= n:
            raise IndexError(f"{ticker}: bar position {i} outside bucket of {n}")
        return SignalResult(entry=bool(entry_flags[i]), exit=bool(exit_flags[i]))

    signal_fn.family_id = FAMILY_ID
    signal_fn.ticker = ticker
    return signal_fn
