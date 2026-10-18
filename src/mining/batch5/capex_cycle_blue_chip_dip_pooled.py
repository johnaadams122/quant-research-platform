"""Batch6 candidate: capex_cycle_blue_chip_dip_pooled (pooled_single_asset).

Rule: buy blue-chip stocks that are BOTH ramping up capital
spending AND currently underperforming SPY -- the thesis being that
today's capex ramp pays off later, and the market hasn't yet priced that
in for names that are concurrently lagging.

Judgment call, disclosed not hedged: this is NOT a relabeling of the well-documented OPPOSITE-
signed capex/investment-growth anomaly (Titman/Wei/Xie 2004; Cooper/
Gulen/Schill 2008, now the CMA factor) -- those are UNCONDITIONAL sorts
on capex/asset growth; this rule adds a second, independent axis
(concurrent underperformance vs SPY) that neither paper tests, and there
IS a real, distinct economic story for the sign to flip on that joint
condition (their mechanism is the market still extrapolating an
overinvesting firm's growth optimism; requiring concurrent price
weakness selects specifically for names where that optimism looks
already priced out). The serious, honest residual risk this family
cannot rule out on its own: any edge found here could simply be ordinary
price-reversal with the capex filter contributing nothing (or hurting,
since TWX/CGS says high-capex firms do WORSE unconditionally). This is
why ``blue_chip_reversal_no_capex_control.py`` (identical minus the capex
condition) is MANDATORY, not optional: a falsification check rather than
a second candidacy, scored as its own trial in the same run -- see that
module's own docstring.

RULE
----
    capex_growth_threshold = 0.30       # YoY capex growth, >= this
    underperformance_threshold = -0.15  # trailing 2yr return vs SPY, <= this
    underperformance_window_sessions = 504
    max_hold_sessions = 252

    for each T in WATCHLIST independently, at each month-end bar m:
      capex_ok = latest known capex_growth_yoy (point-in-time, EDGAR
                 filed-date gated) >= 0.30
      underperf_ok = (Close_m/Close_{m-504} - 1) - (SPY_m/SPY_{m-504} - 1)
                     <= -0.15
      if capex_ok AND underperf_ok AND position==FLAT:
        BUY at the OPEN of the next trading session
      if position==LONG and (NOT capex_ok OR NOT underperf_ok):
        SELL at the CLOSE of the next trading session
      if position==LONG and held >= 252 sessions since fill (and the
        above signal-exit has not already fired): SELL at that bar's
        close (max_hold_sessions -- "a capex cycle plays out" implies a
        bounded horizon, not an open-ended hold)
      one trade = one contiguous holding period; trades of every
        watchlist member concatenated into one sample, scored as ONE trial

Fundamentals: point-in-time capex_growth_yoy from
``mining.edgar_point_in_time`` (filed-date gated, never fiscal-period-end
-- see that module's own docstring for the lookahead hazard this avoids),
joined onto the daily price calendar. A ticker with NO fundamentals known
yet (before its first annual filing enters the pipeline, or once its most
recent filing is stale beyond ``edgar_point_in_time.STALENESS_CAP_DAYS``)
simply never satisfies ``capex_ok`` -- graceful degradation, no crash.

Disclosed defaults (this repo's standing convention):
``capex_growth_threshold``, ``underperformance_threshold``/window, and
``max_hold_sessions`` are all proposed defaults chosen to match the
cited literature's reasoning, not independently confirmed.
``WATCHLIST`` (this module's ``TRADED``) is the one piece this module
explicitly refuses to silently invent -- see the module-level
``WATCHLIST_SOURCE`` note below for exactly how it was built and why.

Engine mapping (mining.backtest_engine.run_backtest): matches
``synthetic_leveraged_dip_buy_pooled_largecap.py``'s own verified
convention (this driver never passes ``time_stop`` for the
``pooled_single_asset`` shape) -- both the signal-off exit and the
252-session max-hold exit are precomputed by this module itself, walked
sequentially exactly like that module's ``dip_entry_exit_events``, and
should be held to the same "verify against the real engine, don't just
derive it" discipline (see this module's own test file).

Interpretation choices
-----------------------
* "Trailing 2-year return" = 504 TRADING sessions (not calendar days),
  matching this repo's existing session-based lookback convention
  throughout (e.g. the 252-session "52-week high").
* Both conditions are re-evaluated only at MONTH-END bars (matching
  ``large_cap_single_stock_pooled_tsmom.py``'s own monthly cadence) --
  fundamentals only update ~1x/year and daily re-evaluation of the price
  condition alone would add noise, not information.
* The month-end scan structurally EXCLUDES a bucket's own final bar from
  ever deciding (mirrors every sibling pooled module); the engine's own
  next-open/same-close fill conventions govern precisely as documented in
  ``dip_entry_exit_events``'s own verified derivation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult
from mining.edgar_point_in_time import point_in_time_asof

FAMILY_ID = "capex_cycle_blue_chip_dip_pooled"
SHAPE = "pooled_single_asset"
BENCHMARK = "SPY"

# WATCHLIST_SOURCE: an OBJECTIVE, MECHANICAL selection rule, never a hand-picked list,
# so the watchlist cannot be second-guessed as "chosen because these are
# the names that later did well." Built by scripts/build_capex_watchlist.py
# (unit-tested core in mining.edgar_cik_resolution / mining.capex_watchlist).
#
# Methodology: rank by capex GROWTH (recent disclosure / 2019-12-31
# disclosure), not by trailing capex LEVEL as of a fixed date -- a
# level-as-of-base-date ranking is blind to a company whose capex only
# ramped up LATER, which is exactly the "capex ramp not yet priced in"
# case this rule targets. Growth is gated by a floor requiring a company
# to be in the top LEVEL_FLOOR_RANK=100 S&P 500 names by CURRENT absolute
# capex level before its growth even counts -- without that floor, a tiny
# company's huge percentage swing off a trivial base would dominate the
# ranking. One ticker per CIK (a company with multiple share classes,
# e.g. GOOG/GOOGL, is deduped to its alphabetically-first ticker).
# Universe: live Wikipedia fetch, current S&P 500 constituents only -- no
# point-in-time index-membership history exists anywhere in this repo, so
# this carries the SAME current-constituents-only survivorship-bias caveat
# every other pooled family here already discloses.
#
# Underlying capex figures: SEC EDGAR XBRL companyconcept, 5-tag
# fallback union, filed-date point-in-time (NEVER end-date). A same-
# period conflict across two different tags is resolved by preferring
# the LARGER value, because the narrowest tag can capture only an
# irrelevant subset of a company's real spend and manufacture a false
# "growth" reading. A base-date figure below 50 million USD is treated
# as unreliable rather than genuinely tiny (such figures can be restated
# to zero in a later filing).
#
# N=25, the 2019-12-31 base date, and the level-floor rank are
# disclosed defaults (same convention as every other invented constant
# in this pipeline), not independently confirmed.
TRADED = [
    "AES", "AMAT", "AMZN", "APD", "ATO", "CAT", "CNP", "EVRG", "FCX",
    "GOOG", "LIN", "LLY", "META", "MSFT", "NCLH", "NEM", "NUE", "ORCL",
    "PNW", "SRE", "TSLA", "TXN", "VTR", "WMT", "XEL",
]
SIGNAL: list = []
SNAPSHOTS: dict = {}
PARAMETERS = {
    "capex_growth_threshold": 0.30,
    "capex_growth_source": "edgar_point_in_time_filed_date_gated_annual",
    "decision_frequency": "monthly_at_month_end_close",
    "entry": "next_session_open",
    "exit": "next_session_close_or_252_session_max_hold",
    "max_hold_sessions": 252,
    "underperformance_threshold": -0.15,
    "underperformance_window_sessions": 504,
    "watchlist_source": (
        "PENDING_CONFIRMATION -- objective mechanical rule: among S&P 500 "
        "names in the top 100 by CURRENT absolute capex, the 25 with the "
        "largest capex GROWTH multiple (recent disclosure / 2019-12-31 "
        "disclosure). N, base date, and level-floor rank are disclosed "
        "defaults, not yet confirmed; built by "
        "scripts/build_capex_watchlist.py"
    ),
    "pooling": "all_pool_trades_concatenated_one_trial",
}
EXPECTED_MIN_TRADES = 0  # no minimum trade count is asserted for this family

CAPEX_GROWTH_THRESHOLD = PARAMETERS["capex_growth_threshold"]
UNDERPERFORMANCE_THRESHOLD = PARAMETERS["underperformance_threshold"]
UNDERPERFORMANCE_WINDOW = int(PARAMETERS["underperformance_window_sessions"])
MAX_HOLD_SESSIONS = int(PARAMETERS["max_hold_sessions"])

_ENTRY = SignalResult(entry=True, exit=False)
_EXIT = SignalResult(entry=False, exit=True)
_NONE = SignalResult()


def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last bar of each calendar month present in ``index``
    (matches every sibling pooled module's identical helper)."""
    periods = index.to_period("M")
    mask = np.ones(len(index), dtype=bool)
    if len(index) > 1:
        mask[:-1] = periods[1:] != periods[:-1]
    return mask


def underperformance_spread(close: pd.Series, benchmark_close: pd.Series, *,
                            window: int = UNDERPERFORMANCE_WINDOW) -> pd.Series:
    """``(Close_t/Close_{t-window} - 1) - (Benchmark_t/Benchmark_{t-window} - 1)``
    on a shared calendar; NaN until ``window`` prior sessions exist for
    BOTH series."""
    own_ret = close / close.shift(window) - 1.0
    bench_ret = benchmark_close / benchmark_close.shift(window) - 1.0
    return own_ret - bench_ret


def capex_condition_series(price_index: pd.DatetimeIndex,
                           fundamentals: pd.DataFrame | None, *,
                           threshold: float = CAPEX_GROWTH_THRESHOLD) -> pd.Series:
    """Bool Series on ``price_index``: True where the point-in-time-joined
    ``capex_growth_yoy`` (from ``edgar_point_in_time.point_in_time_asof``,
    already filed-date gated) is >= ``threshold``. ``fundamentals is
    None`` (a ticker with no EDGAR series at all) -> always False, never
    an error -- graceful degradation, matching every other lookback-gated
    module's treatment of a too-young/unsupported member."""
    if fundamentals is None or fundamentals.empty:
        return pd.Series(False, index=price_index)
    joined = point_in_time_asof(fundamentals, price_index)
    growth = joined["capex_growth_yoy"]
    return (growth >= threshold).fillna(False)


def decision_dates(close: pd.Series, benchmark_close: pd.Series,
                   fundamentals: pd.DataFrame | None, *,
                   require_capex: bool = True,
                   capex_threshold: float = CAPEX_GROWTH_THRESHOLD,
                   underperformance_threshold: float = UNDERPERFORMANCE_THRESHOLD,
                   underperformance_window: int = UNDERPERFORMANCE_WINDOW,
                   max_hold_sessions: int = MAX_HOLD_SESSIONS):
    """Pure sequential walk (mirrors ``synthetic_leveraged_dip_buy_pooled_
    largecap.dip_entry_exit_events``'s own verified bar-accounting
    convention exactly): returns ``(entry_decision_bars, exit_bars)``,
    0-indexed bar positions into ``close``'s own calendar, paired 1:1.

    ``require_capex=False`` builds the CONTROL ARM (underperformance
    condition only) -- see ``blue_chip_reversal_no_capex_control.py``.

    Re-evaluated only at month-end bars. While FLAT, the first month-end
    bar where both (or, control arm, just the one) condition(s) hold
    fires an entry, filling next-open. While LONG, exits at the first of:
    the NEXT month-end bar where a condition no longer holds (signal
    exit, filling that bar's close per the engine's own same-bar
    decide-and-fill convention -- see the docstring on bar accounting in
    the sibling synthetic-leverage module), or ``max_hold_sessions`` sessions held
    (whichever first). After an exit, the walk resumes checking AT the
    exit bar itself (same chained-reentry convention as
    ``synthetic_leveraged_dip_buy_pooled_largecap`` -- the engine's own
    FLAT branch on the next iteration is queried with decision index ==
    the exit bar).

    Disclosed limitation, not fixed,
    same as the synthetic-leverage module's own ``dip_entry_exit_events``: both the signal
    exit and the ``max_hold_sessions`` exit reach the engine as an
    ordinary ``exit=True``, so ``trades_df["reason"]`` always reads
    ``"signal_exit"``, never ``"time_stop"`` -- ``net_ret``/DSR scoring
    is unaffected; distinguishing the two causes later needs ``held``,
    not ``reason``."""
    n = len(close)
    if not (len(benchmark_close) == n):
        raise ValueError("close and benchmark_close must share one calendar")
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    index = close.index
    spread = underperformance_spread(close, benchmark_close, window=underperformance_window)
    underperf_ok = (spread <= underperformance_threshold).fillna(False).to_numpy()
    if require_capex:
        capex_ok = capex_condition_series(
            index, fundamentals, threshold=capex_threshold).to_numpy()
        signal_ok = underperf_ok & capex_ok
    else:
        signal_ok = underperf_ok
    me_mask = month_end_mask(_normalize_index(index))

    entries: list = []
    exits: list = []
    i = 0
    while i < n:
        if not (me_mask[i] and signal_ok[i]):
            i += 1
            continue
        decision_bar = i
        fill_bar = i + 1
        if fill_bar >= n:
            break
        entries.append(decision_bar)

        exit_bar = None
        j = fill_bar + 1
        while j < n:
            held = j - fill_bar
            hit_time_stop = held >= max_hold_sessions
            hit_signal_exit = me_mask[j] and not signal_ok[j]
            if hit_signal_exit or hit_time_stop:
                exit_bar = j
                break
            j += 1
        if exit_bar is None:
            entries.pop()
            break
        exits.append(exit_bar)
        i = exit_bar

    return np.array(entries, dtype=np.int64), np.array(exits, dtype=np.int64)


def _make_signal_fn(ticker: str, index, frames: dict, fundamentals_map, *,
                    require_capex: bool) -> callable:
    if ticker not in frames:
        raise KeyError(f"missing frame for {ticker}")
    if BENCHMARK not in frames:
        raise KeyError(f"missing benchmark frame for {BENCHMARK}")
    frame = frames[ticker]
    benchmark_frame = frames[BENCHMARK]
    if "Close" not in frame.columns or "Close" not in benchmark_frame.columns:
        raise ValueError(f"{ticker}/{BENCHMARK} frames must have a Close column")

    bucket = _normalize_index(index)
    if len(bucket) == 0:
        def empty_signal_fn(frame_slice, i):
            return _NONE
        return empty_signal_fn
    cutoff = bucket[-1]

    def _prep(f):
        idx = _normalize_index(f.index)
        out = f.set_axis(idx)
        return out.loc[out.index <= cutoff]

    own = _prep(frame)
    bench = _prep(benchmark_frame)
    if not own.index.equals(bench.index):
        raise ValueError(f"{ticker} and {BENCHMARK} frames must share one identical calendar")

    fundamentals = (fundamentals_map or {}).get(ticker)
    entries, exit_bars = decision_dates(
        own["Close"], bench["Close"], fundamentals, require_capex=require_capex)

    full_dates = own.index
    entry_dates = full_dates[entries] if len(entries) else pd.DatetimeIndex([])
    exit_dates = full_dates[exit_bars] if len(exit_bars) else pd.DatetimeIndex([])
    bucket_entry = np.asarray(bucket.isin(entry_dates), dtype=bool)
    bucket_exit = np.asarray(bucket.isin(exit_dates), dtype=bool)
    results = np.empty(len(bucket), dtype=object)
    for pos in range(len(bucket)):
        if bucket_entry[pos] and bucket_exit[pos]:
            results[pos] = SignalResult(entry=True, exit=True)
        elif bucket_entry[pos]:
            results[pos] = _ENTRY
        elif bucket_exit[pos]:
            results[pos] = _EXIT
        else:
            results[pos] = _NONE
    bucket_ns = bucket.asi8
    by_date = {int(ts): pos for pos, ts in enumerate(bucket_ns)}
    n = len(bucket)

    def signal_fn(frame_slice, i):
        today = pd.Timestamp(frame_slice.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        today_ns = int(today.normalize().value)
        if 0 <= i < n and bucket_ns[i] == today_ns:
            return results[i]
        pos = by_date.get(today_ns)
        if pos is None:
            return _NONE
        return results[pos]

    return signal_fn


def make_signal_fn(ticker: str, index, frames: dict, *, fundamentals=None):
    """SignalFn factory matching every sibling module's ``(ticker, index,
    frames)`` contract, with an OPTIONAL keyword-only ``fundamentals``
    (``{ticker: point-in-time capex series}``) -- this family's own
    bespoke driver always passes it explicitly; the default keeps
    ``validate_module_contract``'s generic 3-positional-arg expectation
    satisfied without a shared-driver signature change."""
    return _make_signal_fn(ticker, index, frames, fundamentals, require_capex=True)
