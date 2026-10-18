"""Batch5 candidate: fx_carry_distribution_yield_regime (single_long).

Currency carry via FXA's (Invesco CurrencyShares Australian Dollar Trust)
own realized distribution yield. FXA passes through accrued short-term AUD
interest as periodic cash distributions, so its own trailing 12-month
distribution yield is a direct read on AUD's interest-rate carry vs USD.
Source: Lustig, Roussanov & Verdelhan (2011), "Common Risk Factors in
Currency Markets", Review of Financial Studies 24(11), 3731-3777 (cited by
the frozen catalog entry).
Rule, verbatim from the frozen catalog entry's "mechanism" field:

    At each month-end, compute trailing-12-month distributions divided by
    price; go long FXA when that yield is above its own trailing 36-month
    median, move to cash otherwise.

UUP appears in the catalog's pre-registered "tickers" list, but the
mechanism never reads it -- the signal is FXA's OWN distribution yield only.
TRADED = ["FXA"], SIGNAL = [] (no external signal ticker; this module
deliberately departs from the catalog's tickers field here, since the
mechanism never reads UUP).

Engine mapping (mining.backtest_engine): the rule is checked only at
MONTH-END bars ("at each month-end" -- the catalog's own wording, unlike
the sibling preferred_stock_yield_spread_reversion template, which checks on
the FIRST bar of the month); the decision then holds for every bar until the
next month-end check. On every bar t: entry[t] = standing LONG[t] (fills at
bar t+1's open, the earliest the engine can buy) and exit[t] = not LONG[t]
(fills at bar t's close) -- the same "exit is the complement of the standing
state" idiom preferred_stock_yield_spread_reversion uses, so a position the
engine opened one bar after the decision (its own ``continue`` on the entry
bar skips that bar's exit check) is still closed at the next FLAT bar.

Interpretation choices
-----------------------
1. "Trailing 12-month distributions" = the sum of FXA's own per-share cash
   distributions (ex-date indexed, ``yfinance Ticker.dividends``) whose
   EX-DATE lands, after a causal forward-shift to the first trading session
   on or after it (an ex-date that is not itself a session counts on the
   next session; never before), within the trailing 12 CALENDAR MONTHS
   ending at the current month-end decision bar's own calendar month
   (inclusive). Divided by that month-end bar's UNADJUSTED close (the
   family's data file, fetched via ``yfinance
   Ticker.history(auto_adjust=False)``) -- the
   engine's own frame carries a dividend-ADJUSTED close (back-adjusted for
   every FUTURE distribution), which would distort a trailing yield computed
   over past dates exactly as documented in
   preferred_stock_yield_spread_reversion's docstring choice 1. A full
   12-calendar-month span of price/distribution history is required (NaN,
   no decision, before the 12th calendar month of the series): the *span*
   requirement is positional (12 calendar months must have elapsed), not a
   requirement that every one of those months carried a distribution --
   FXA's real carry income is genuinely zero for stretches (whole calendar
   years can carry ZERO recorded distributions in the fetched data; see
   the fetcher script's own docstring), and a month with no
   distribution contributes 0 to the trailing sum, which is the correct
   economic reading of a near-zero-carry regime, never an error.
2. "Month-end" = the LAST trading day of each calendar month present in the
   (possibly holdout-truncated) close series -- the same ``_month_end_mask``
   idiom ``dividend_month_premium_predicted_ex_month`` uses: a bar counts as
   month-end when the NEXT bar in the series falls in a different calendar
   month; the series' own final bar counts as month-end only when no
   weekday of its calendar month remains after it (a genuinely partial final
   month, e.g. mid-month truncation for a holdout bucket, is never
   mis-flagged as month-end).
3. Regime threshold = a rolling MEDIAN of the monthly trailing-yield series,
   over the trailing 36 MONTH-END decision values ending at (and including)
   the current one -- a full 36-value window required (``rolling(36,
   min_periods=36)``, so a single undefined monthly yield inside the window
   propagates as "no median yet", never a median computed over fewer than
   36 real observations). No mean/std/z-score enters anywhere here (unlike
   the sibling preferred_stock template): the catalog's own wording names a
   plain median threshold, not a standardized distance from one.
4. Decision rule, evaluated ONLY on month-end bars: is_long = (yield is
   defined) AND (median is defined) AND (yield > median) -- a STRICT
   inequality ("above" its median, catalog wording), so a yield exactly
   equal to its own trailing median is CASH, not long. Either input
   undefined (insufficient history) makes the bar CASH, never long from
   missing data. There is no separate entry/exit threshold or hysteresis
   band here (unlike preferred_stock_yield_spread_reversion's asymmetric
   0.75/0 z-score bands) -- the catalog names one plain condition, so the
   standing state simply tracks the sign of (yield - median) at each
   month-end check and holds between checks.
5. "Move to cash" = flat; no cash instrument is priced (single-instrument
   engine, the same simplification every other regime candidate in this
   batch uses).
6. Bucket edges: a bucket that OPENS inside a standing LONG regime enters at
   its first fill opportunity (entry stays raised while LONG, mirroring
   preferred_stock_yield_spread_reversion's docstring choice 5); the exit
   flag is raised on every non-LONG bar so a position the engine opened one
   bar after the decision is still closed at the next FLAT bar.
7. Holdout-blind by construction: both series are truncated to dates <= the
   bucket's own last bar before any computation runs (month-end detection,
   monthly aggregation, trailing sum, rolling median all recompute from that
   truncated view), so a holdout bucket structurally cannot see later data.
   History before the bucket is kept as warm-up (the 12+35=47-month combined
   lookback the first in-bucket decision may need).

Data hygiene -- disclosed limitation
-------------------------------------
This is a single-ticker build, and its data file (written by the fetcher)
is a SINGLE VENDOR pull (yfinance) with
NO independent cross-source reconciliation -- stated plainly in
``scripts/fetch_batch5_events_fx_carry_distribution_yield_regime.py`` and
in the provenance file it writes, never silently implied to be verified.
The structural guards this module and its fetcher DO apply: every stored
distribution amount is finite and strictly positive; the stored close
series is finite and strictly positive, sorted, and de-duplicated. No "every month must carry a
distribution" completeness invariant is imposed (see interpretation choice
1) -- FXA's real cadence includes genuine zero-distribution years, and such
a guard would fail-close a healthy file for a true economic fact.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "fx_carry_distribution_yield_regime"
SHAPE = "single_long"
TRADED = ["FXA"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "FXA"
# Copied verbatim from the frozen catalog entry's "parameters" (the driver's
# drift check requires byte identity under json.dumps(sort_keys=True)).
PARAMETERS = {
    "yield_lookback_months": 12,
    "regime_lookback_months": 36,
    "rebalance": "monthly",
    "long_only": True,
}
# Conservative floor on the expected trade count: a slow multi-year regime
# family trades rarely. Reported alongside a run's results as a sanity
# reference; it gates nothing.
EXPECTED_MIN_TRADES = 5

YIELD_LOOKBACK_MONTHS = int(PARAMETERS["yield_lookback_months"])
REGIME_LOOKBACK_MONTHS = int(PARAMETERS["regime_lookback_months"])

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


def load_inputs(path: str = DATA_PATH, *, ticker: str = TRADED[0]) -> dict:
    """{ticker: {"unadjusted_close": Series, "dividends": Series}} from the
    family's data file (dates ISO strings -> tz-naive DatetimeIndex)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if ticker not in payload:
        raise ValueError(f"{ticker}: no distribution/close inputs in {path}")
    block = payload[ticker]
    close = pd.Series(block["unadjusted_close"], dtype=float)
    close.index = pd.DatetimeIndex(pd.to_datetime(list(close.index)))
    close = _clean_positive_series(close, f"{ticker} unadjusted_close")
    dividends = pd.Series(block["dividends"], dtype=float)
    dividends.index = pd.DatetimeIndex(pd.to_datetime(list(dividends.index)))
    dividends = _clean_positive_series(dividends, f"{ticker} dividends")
    return {ticker: {"unadjusted_close": close, "dividends": dividends}}


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


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on bars whose NEXT bar in ``index`` falls in a different
    calendar month; the final bar is True only when no weekday of its own
    calendar month remains after it (a genuinely partial final month, e.g. a
    holdout-truncated series ending mid-month, is never mis-flagged)."""
    n = len(index)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    periods = index.to_period("M")
    mask[:-1] = periods[:-1] != periods[1:]
    last = index[-1]
    month_end_day = last.to_period("M").to_timestamp(how="end").normalize()
    remaining = pd.bdate_range(last + pd.Timedelta(days=1), month_end_day)
    mask[-1] = len(remaining) == 0
    return mask


def monthly_trailing_yield(unadjusted_close: pd.Series, dividends: pd.Series, *,
                           lookback_months: int = YIELD_LOOKBACK_MONTHS
                           ) -> pd.Series:
    """Trailing ``lookback_months``-calendar-month sum of distributions
    divided by that month-end's own unadjusted close, indexed by the ACTUAL
    trading-day timestamp of each month-end bar (one row per calendar month
    present in ``unadjusted_close``'s index). NaN until ``lookback_months``
    calendar months have elapsed (docstring choice 1) -- a month with no
    distribution contributes 0 to the sum, never NaN, never a gap."""
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    close = _clean_positive_series(unadjusted_close, "unadjusted_close")
    on_calendar = _distributions_on_calendar(pd.Series(dividends, dtype=float),
                                             close.index)
    periods = close.index.to_period("M")
    monthly_sum = on_calendar.groupby(periods).sum()
    monthly_sum = monthly_sum.sort_index()
    trailing = monthly_sum.rolling(lookback_months,
                                  min_periods=lookback_months).sum()

    me_mask = month_end_mask(close.index)
    me_positions = np.flatnonzero(me_mask)
    me_dates = close.index[me_positions]
    me_periods = periods[me_positions]
    trailing_at_me = trailing.reindex(me_periods).to_numpy()
    price_at_me = close.to_numpy()[me_positions]
    yld = trailing_at_me / price_at_me
    return pd.Series(yld, index=me_dates, name="trailing_yield")


def regime_median(monthly_yield: pd.Series, *,
                  regime_lookback_months: int = REGIME_LOOKBACK_MONTHS
                  ) -> pd.Series:
    """Rolling median of ``monthly_yield`` (one row per month-end decision)
    over the trailing ``regime_lookback_months`` values, full window
    required (NaN before, docstring choice 3)."""
    if regime_lookback_months < 1:
        raise ValueError("regime_lookback_months must be at least 1")
    return monthly_yield.rolling(regime_lookback_months,
                                 min_periods=regime_lookback_months).median()


def standing_long(unadjusted_close: pd.Series, dividends: pd.Series, *,
                  yield_lookback_months: int = YIELD_LOOKBACK_MONTHS,
                  regime_lookback_months: int = REGIME_LOOKBACK_MONTHS
                  ) -> pd.Series:
    """Per-bar standing state (True = LONG), indexed by ``unadjusted_close``'s
    own full calendar: checked only at month-end bars (docstring choice 2),
    is_long = yield > median (strict, docstring choice 4), NaN in either
    input -> False; carried forward between checks. Sequential by
    construction (state at bar t depends only on month-end checks <= t)."""
    close = _clean_positive_series(unadjusted_close, "unadjusted_close")
    monthly_yield = monthly_trailing_yield(
        close, dividends, lookback_months=yield_lookback_months)
    median = regime_median(monthly_yield,
                           regime_lookback_months=regime_lookback_months)
    # Strict inequality; NaN on either side resolves to False automatically
    # (numpy/pandas float comparison against NaN is always False, never NaN),
    # which is exactly "undefined -> cash" (docstring choice 4).
    decision = (monthly_yield > median)
    me_positions = close.index.get_indexer(decision.index)
    values = decision.to_numpy(dtype=bool)

    n = len(close)
    if len(values) == 0:
        # No month-end bar has occurred at all (a bucket/history shorter than
        # one calendar month): never a decision, never long.
        return pd.Series(np.zeros(n, dtype=bool), index=close.index, name="long")
    bar_positions = np.arange(n)
    idx_of_last_decision = np.searchsorted(
        me_positions, bar_positions, side="right") - 1
    has_decision = idx_of_last_decision >= 0
    clipped = np.clip(idx_of_last_decision, 0, len(values) - 1)
    state = np.where(has_decision, values[clipped], False)
    return pd.Series(state, index=close.index, name="long")


def signal_flags(unadjusted_close: pd.Series, dividends: pd.Series) -> pd.DataFrame:
    """Boolean ``entry`` / ``exit`` columns aligned to the close calendar:
    entry[t] = standing LONG at t (fills t+1 open); exit[t] = not LONG at t
    (fills t close). See docstring choice 6 for why exit is the complement."""
    longs = standing_long(unadjusted_close, dividends)
    return pd.DataFrame({"entry": longs.to_numpy(bool),
                         "exit": ~longs.to_numpy(bool)}, index=longs.index)


# ------------------------------------------------------------- factory

def make_signal_fn(ticker: str, index, frames, *, inputs: dict | None = None):
    """Driver factory (single_long contract): ``index`` is the bucket
    calendar the engine will run over; ``frames[ticker]`` is the full-history
    OHLC frame (used for validation only -- fills happen in the engine on
    that frame; the yield uses the data file's unadjusted close, docstring
    choice 1). ``inputs`` defaults to the family's data file; tests inject
    synthetic series. Everything is precomputed here; the returned closure is
    O(1) per bar."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
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
        inputs = load_inputs(DATA_PATH, ticker=ticker)
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
    # Holdout-blind by construction (docstring choice 7): nothing after the
    # bucket's last bar is ever loaded into the computation.
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
