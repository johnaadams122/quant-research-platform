"""Core of the range-bound spike-fade study, Stage 1 (pre-registered: every
grid value and parameter is a fixed constant below).

WHAT THIS MODULE IS. The deterministic, causal building blocks the study
stands on: the sealed loader, the data-quality rules, the three trigger
families, gap20 episodes, the episode-level trade simulator with its 72
exit-parameter variants and the per-cell statistics. It is a RESEARCH
module: it places no trade, writes no ledger row and spends no trial slot.
The null machinery, the plateau / vehicle / put screens are built ON TOP of
it and are not here.

CONVENTIONS (read once, they hold everywhere):
  * Positions. Every function works on POSITIONS (0-based integer offsets
    into one series' sealed, ascending date index), never on labels, so a
    result is reproducible bit-for-bit and cheap to slice.
  * Causality. A value at session t may use closes up to and including t
    and baselines that end at t-1. `test_prefix_*` in
    src/tests/test_spike_fade_study.py proves that truncating a series at
    ANY session leaves every value at or before it unchanged.
  * Sessions. A session is a row of the series' own valid-observation
    calendar: rows with no observation are dropped at load (and
    counted), so 'the close 20 sessions back' is 20 valid rows back.
  * Invalid closes. `SpikeSeries.p` holds the cleaned closes with NaN for
    every invalid session (non-positive, or set missing by the external
    same-day reference check, the ONLY data correction). `nonpos`
    marks the non-positive ones separately because they are treated
    differently from an ordinary missing value. NO data rule reads a value
    after the close it judges.
  * Sides. `up_fade` = the series jumped UP and the trade is SHORT it
    (sign s = +1); `down_fade` = it crashed and the trade is LONG (s = -1).
    P&L per unit initial notional is -s * (P_exit / P_entry - 1), simple
    return, uncapped. A vehicle (tradable-vehicle test) is traded with its own sign
    passed as `side` (an int) to `evaluate_variants`.
  * Cells. `CELL_GRID` is the 72 (entry, H, TP, SL) variants in a fixed
    order; 4 triggers x 72 = 288 cells per (series, side).
  * No dollar figures anywhere in this file.

Everything a reader could need to audit a number is returned, not hidden:
each simulation returns a STATUS code per (cell, episode) so every dropped,
censored, undefined, not-applicable or externally corrected observation is countable
in the report.

PUBLIC ENTRY POINTS, in the order a caller meets them:
  build_series(close, name=, cutoff=, drop_missing=, reference=, extra_lag=)
                                         sealed load + data rules -> SpikeSeries
  compute_triggers(series)               4 triggers -> {name: TriggerSignals}
  extract_episodes(flags) / censor_episodes(t0, n) / build_episode_set(...)
  series.context() -> SeriesContext      median M, sigma21 (+ floor) per session
  evaluate_variants(series, t0, side, cells=)  numpy: any t0 (real or null
                                         pseudo-episodes), any subset of the 72
                                         cells -> VariantResults
  simulate_episode(...) / simulate_from_series(...)  the sequential reference
  cell_stats(x, pnl, dates) / cell_stats_from_results(...)   per-cell statistics
  evaluate_group / evaluate_series / series_frame      the 72-cell reports
  trigger_diagnostics / removed_observations / cell_status_counts  the counts
  entry_d_e0(series, side)               D at an E0 entry (side-of-median match)
  map_anchors_to_vehicle(dates, vehicle_index, extra_lag=)   vehicle-test anchors
Null-machinery callers pass pseudo-episode positions to `evaluate_variants`
with `cells=[PRIMARY_CELL]` and read `res.x[0]` / `res.status[0]`; a vehicle
(tradable-vehicle test) is evaluated on its OWN SpikeSeries with `side` = its position
sign (+1 short, -1 long) and event dates mapped to that series' positions.
"""
from __future__ import annotations

import bisect
import contextlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------------------
# Constants: every grid value and parameter the pre-registration names
# ---------------------------------------------------------------------------

# Sealed loader: last usable date, inclusive. The official ABI-1 holdout (the
# last ABI1_HOLDOUT_SESSIONS = 756 scored sessions, src/mining/constants.py)
# starts in late September 2023, so about twelve weeks are used by neither
# this study nor the official score for selection.
DISCOVERY_CUTOFF = pd.Timestamp("2023-06-30")

# The four triggers, two sides.
Z_GRID = (2.0, 2.5)
TRIGGERS = ("z20@2.0", "z20@2.5", "pct756", "pctexp")
SIDES = ("up_fade", "down_fade")
SIDE_SIGN = {"up_fade": +1, "down_fade": -1}   # +1 = short the series

Z_LOOKBACK = 20            # r20(t) = ln(P(t) / P(t - 20)); fresh-extreme window
BASELINE_WINDOW = 252      # mean / sd of r20 over the 252 sessions ending t-1
MIN_VALID_BASELINE = 200   # NaN-tolerant: at least 200 valid of 252
PCT_WINDOW = 756           # rolling percentile window (sessions ending t-1)
MIN_VALID_PCT = 500        # NaN-tolerant: at least 500 valid of 756
PCT_UP = 95.0              # numpy 'linear' percentiles
PCT_DOWN = 5.0
Z20_MIN_CLOSES = 273       # 20 for r20 + 252 baseline + the signal close
PCTEXP_MIN_CLOSES = 756    # expanding percentile needs this many valid closes
GAP_SESSIONS = 20          # de-clustering: >= 20 sessions since ANY prior flag

# Trade construction.
E1_CONFIRM_SESSIONS = 5    # E1 looks for a confirming close in (t0, t0 + 5]
ENTRY_RULES = ("E0", "E1")
H_GRID = (10, 21, 42, 63)
TP_GRID = ("none", "half", "full")
TP_FRACTION = {"none": None, "half": 0.5, "full": 1.0}
SL_GRID = (None, 1.5, 3.0)              # None = no stop; else multiple of sigma21
MEDIAN_WINDOW = 252                     # M = median of P over 252 sessions ending t0
SIGMA_WINDOW = 63                       # daily log returns ending t0
SIGMA_HORIZON = 21                      # sigma21 = sd * sqrt(21)
SIGMA_FLOOR_PERCENTILE = 10.0           # causal expanding floor
# The pre-registration fixes 200-of-252, 500-of-756 and 50-of-63 (the sigma21
# window) as minimum valid counts, and the sigma floor's requirement as at
# least 200 valid PRIOR values.
MIN_VALID_SIGMA_RETURNS = 50
MIN_SIGMA_FLOOR_HISTORY = 200

# Uniform censoring: t0 + 6 + 63 + 1 sessions must exist.
CENSOR_TAIL = (E1_CONFIRM_SESSIONS + 1) + max(H_GRID) + 1

# External same-day reference check: the only data correction is an EXTERNAL same-day cross-check
# (JPY=X against FRED DEXJPUS): a close differing from the reference by more
# than this fraction is set missing and counted. (No rule based on the NEXT
# session's reversal is used: such a rule would delete genuine spikes.)
EXTERNAL_CHECK_TOLERANCE = 0.01
EXTERNAL_CHECK_SERIES = ("JPY=X",)   # series whose ONLY correction is the reference check: it is mandatory
LARGEST_RETURNS_LISTED = 20    # per series, the largest |one-session log return| the report lists

# Per-cell statistics.
WINSOR_BOUND = 5.0
SD_ZERO_REL_TOL = 1e-12    # sd(x) at or below this x max(1, |S|) is float noise, i.e. zero
MIN_EPISODES = 8
CLUSTER_DAYS = 28
# The statistics call for a cluster bootstrap 90 percent interval but fix
# neither the number of replications nor a seed; frozen here.
CLUSTER_BOOTSTRAP_B = 2000
CLUSTER_BOOTSTRAP_SEED = 2026_0925
CLUSTER_BOOTSTRAP_LEVEL = 0.90

# Null-test / plateau parameters other modules build on (declared once, here).
NULL_REPLICATIONS = 500
N1_SEED = 2026_0923
N2_SEED = 2026_0924
HEADLINE_P_MAX = 0.10
MIN_HALF_EPISODES = 4      # stability rule: each half needs >= 4 episodes
MIN_PLATEAU_NEIGHBORS = 3  # plateau score

# The 72 exit-parameter variants, in one fixed order used by every array.
CELL_GRID = tuple((e, h, tp, sl)
                  for e in ENTRY_RULES for h in H_GRID
                  for tp in TP_GRID for sl in SL_GRID)
CELL_INDEX = {c: i for i, c in enumerate(CELL_GRID)}
PRIMARY_CELL = ("E0", 21, "none", None)   # the max-t test's pre-declared cell

# Sequential status codes for one (cell, episode). Precedence is the order of
# the checks in `simulate_episode` (the reference implementation).
STATUS_TRADED = 0
STATUS_UNDEFINED_SIGMA = 1        # sigma21 not computable (too few valid returns / no floor history)
STATUS_SIGMA_TOUCH_NONPOS = 2     # sigma window touches a non-positive close
STATUS_NO_E1_CONFIRM = 3          # E1 found no confirming session in (t0, t0 + 5]
STATUS_TOUCH_NONPOS = 4           # entry or path touches a non-positive close
STATUS_UNDEFINED_M = 5            # TP cell: median M not computable
STATUS_M_TOUCH_NONPOS = 6         # TP cell: median window touches a non-positive close
STATUS_CENSORED = 7               # path runs past the last close (never after uniform censoring)
STATUS_TOUCH_MISSING = 8          # entry / exit price (or, for stop cells, the path) is a missing close
STATUS_NAMES = {
    STATUS_TRADED: "traded",
    STATUS_UNDEFINED_SIGMA: "undefined_sigma",
    STATUS_SIGMA_TOUCH_NONPOS: "sigma_touch_nonpositive",
    STATUS_NO_E1_CONFIRM: "no_e1_confirmation",
    STATUS_TOUCH_NONPOS: "touch_nonpositive",
    STATUS_UNDEFINED_M: "undefined_median",
    STATUS_M_TOUCH_NONPOS: "median_touch_nonpositive",
    STATUS_CENSORED: "censored_data_end",
    STATUS_TOUCH_MISSING: "touch_missing",
}

# Exit reasons for a traded (cell, episode).
EXIT_TIME, EXIT_TP, EXIT_STOP = 0, 1, 2
EXIT_NAMES = {EXIT_TIME: "time_stop", EXIT_TP: "take_profit", EXIT_STOP: "stop_loss"}

_BIG = 10 ** 9   # "no such offset" sentinel for first-flag searches


# ---------------------------------------------------------------------------
# The sealed loader
# ---------------------------------------------------------------------------

class SealedWindowError(AssertionError):
    """A date after DISCOVERY_CUTOFF reached code that must never see one.

    Subclasses AssertionError because the contract is an assertion ("the
    maximum loaded date is at or before the cutoff"); a dedicated class keeps
    it distinguishable from an unrelated failed assert and survives
    `python -O`, which strips bare `assert` statements."""


@dataclass(frozen=True)
class SealedSeries:
    """One series, truncated at the cutoff BEFORE any computation.

    `close` is the RAW close (NaN allowed, never adjusted here): data rules
    are applied by `prepare_series`. Counts record what the seal removed, so
    nothing is dropped silently.
    """
    name: str
    index: pd.DatetimeIndex
    close: np.ndarray
    cutoff: pd.Timestamp
    n_post_cutoff_dropped: int
    n_pre_start_dropped: int
    n_missing_dropped: int = 0

    @property
    def n_rows(self) -> int:
        return int(len(self.index))

    @property
    def max_date(self) -> pd.Timestamp:
        return self.index[-1]


# The seal is NOT a default a caller can move: no public function takes a way
# around it. `_CUTOFF_CEILING` is the latest cutoff any loader, builder or guard
# will accept; it equals DISCOVERY_CUTOFF except inside the private, test-only
# context manager below.
_CUTOFF_CEILING = DISCOVERY_CUTOFF


@contextlib.contextmanager
def _allow_cutoff_beyond_discovery(until: pd.Timestamp):
    """PRIVATE, TEST-ONLY. Temporarily raise the accepted cutoff ceiling to
    `until` (never below DISCOVERY_CUTOFF), restoring it on exit even if the
    block raises. Exists so a test can prove the seal holds and that the helper
    is the only way past it; production code must never call it."""
    global _CUTOFF_CEILING
    until = pd.Timestamp(until)
    if until < DISCOVERY_CUTOFF:
        raise ValueError("the ceiling can only be raised, never lowered below DISCOVERY_CUTOFF")
    previous = _CUTOFF_CEILING
    _CUTOFF_CEILING = until
    try:
        yield
    finally:
        _CUTOFF_CEILING = previous


def _check_cutoff(cutoff: pd.Timestamp) -> None:
    if pd.Timestamp(cutoff) > _CUTOFF_CEILING:
        raise SealedWindowError(
            f"cutoff {pd.Timestamp(cutoff).date()} is later than the discovery cutoff "
            f"{DISCOVERY_CUTOFF.date()}: the sealed window cannot be loaded")


def assert_sealed(index: pd.DatetimeIndex, cutoff: pd.Timestamp = DISCOVERY_CUTOFF) -> None:
    """Raise SealedWindowError if `cutoff` is later than the discovery cutoff, or
    if any date is strictly after `cutoff`.

    Called by `load_sealed` on its own output, by `prepare_series` on the
    object it is handed, and at the top of `compute_triggers`,
    `evaluate_variants` and `series_context` on the series they receive, so a
    hand-built object (or a caller-chosen far-future cutoff) cannot smuggle a
    later date past the guard."""
    _check_cutoff(cutoff)
    if len(index) and pd.Timestamp(index.max()) > pd.Timestamp(cutoff):
        raise SealedWindowError(
            f"date {pd.Timestamp(index.max()).date()} is after the sealed "
            f"cutoff {pd.Timestamp(cutoff).date()}")


def load_sealed(close: Union[pd.Series, pd.DataFrame], *, name: str = "series",
                cutoff: pd.Timestamp = DISCOVERY_CUTOFF,
                first_date: Optional[pd.Timestamp] = None,
                drop_missing: bool = True) -> SealedSeries:
    """Truncate a price frame at `cutoff` (inclusive) and assert the result.

    The truncation is the FIRST thing that happens to the data: no
    adjustment factor, percentile or return is computed on rows after the
    cutoff, because those rows are gone before this function returns. A
    DataFrame is reduced to its `Close` column. Dates are normalized to
    tz-naive midnights. The index must be strictly ascending (unsorted or
    duplicate dates raise ValueError: a silent re-sort could hide a
    look-ahead). `first_date` drops rows before a series' first usable
    date (the study's series table) and counts them.

    `drop_missing` (default True): a SESSION is a row of the
    series' own valid-observation calendar, so rows whose close is not finite
    (FRED holiday gaps) are dropped and counted (`n_missing_dropped`); 'P(t-20)'
    is then the valid observation 20 rows back. A non-positive close is a
    valid observation (a session) whose logarithm is undefined; it is kept.
    """
    _check_cutoff(cutoff)                 # before anything is read: the seal cannot be moved
    if isinstance(close, pd.DataFrame):
        if "Close" not in close.columns:
            raise ValueError("DataFrame input needs a 'Close' column")
        close = close["Close"]
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("dates must be strictly ascending and unique")
    cutoff = pd.Timestamp(cutoff)
    values = np.asarray(close.to_numpy(dtype=float), dtype=float)

    keep = idx <= cutoff
    n_post = int((~keep).sum())
    n_pre = 0
    if first_date is not None:
        early = idx < pd.Timestamp(first_date)
        n_pre = int((early & keep).sum())
        keep = keep & ~early
    n_missing = 0
    if drop_missing:
        gap = keep & ~np.isfinite(values)
        n_missing = int(gap.sum())
        keep = keep & ~gap
    sealed = SealedSeries(name=name, index=idx[keep], close=values[keep].copy(),
                          cutoff=cutoff, n_post_cutoff_dropped=n_post,
                          n_pre_start_dropped=n_pre, n_missing_dropped=n_missing)
    assert_sealed(sealed.index, cutoff)
    return sealed


# ---------------------------------------------------------------------------
# Rolling-window primitives (NaN-tolerant, causal by construction)
# ---------------------------------------------------------------------------

_CHUNK_ROWS = 2048     # rows of a sliding window materialized at once


def rolling_apply(x: np.ndarray, window: int, lag: int, reducer) -> np.ndarray:
    """Apply `reducer(win)` to the window of `x` ending at session t - lag.

    For session t the window is x[t - lag - window + 1 : t - lag + 1] (so
    `lag = 1` is a BASELINE ending at t-1 and `lag = 0` includes t itself).
    `reducer` receives a 2-D array (rows = sessions, columns = the window,
    oldest first) and returns one float per row. Sessions whose full window
    does not fit inside the series get NaN. Every row is reduced
    independently of every other row, so the value at t depends only on the
    window's own contents -- the property the prefix-invariance test
    proves. Rows are materialized in chunks to bound memory.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0: a window ending after t would read the future")
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    first_t = lag + window - 1
    if n <= first_t:
        return out
    view = sliding_window_view(x, window)          # row s = x[s : s + window]
    n_rows = n - first_t                           # t = s + first_t <= n - 1
    for a in range(0, n_rows, _CHUNK_ROWS):
        b = min(a + _CHUNK_ROWS, n_rows)
        out[first_t + a:first_t + b] = reducer(np.ascontiguousarray(view[a:b]))
    return out


def _count_valid(win: np.ndarray) -> np.ndarray:
    return np.isfinite(win).sum(axis=1)


def _rows_mean(win: np.ndarray) -> np.ndarray:
    valid = np.isfinite(win)
    c = valid.sum(axis=1)
    s = np.where(valid, win, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def _rows_sd(win: np.ndarray) -> np.ndarray:
    """Sample standard deviation (ddof = 1) of the finite values in each row."""
    valid = np.isfinite(win)
    c = valid.sum(axis=1)
    s = np.where(valid, win, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
        dev = np.where(valid, win - m[:, None], 0.0)
        var = (dev * dev).sum(axis=1) / (c - 1)
    return np.sqrt(var)


def lagged_window_stats(x: np.ndarray, window: int, min_valid: int):
    """NaN-tolerant mean and sd (ddof = 1) of `x` over the `window` sessions
    ENDING AT t-1, plus the valid count. Returns (mean, sd, count), each of
    length len(x); the statistics are NaN where the full window does not fit
    or where fewer than `min_valid` (and fewer than 2) values are valid.
    `count` is reported even when the statistic is withheld so a caller can
    tell "too little data" from "window not yet full".

    This is THE causal baseline primitive for the z20 trigger. A test
    monkeypatches it with a one-session-ahead version to prove the
    prefix-invariance test has power.
    """
    x = np.asarray(x, dtype=float)
    mean = rolling_apply(x, window, 1, _rows_mean)
    sd = rolling_apply(x, window, 1, _rows_sd)
    cnt = rolling_apply(x, window, 1, lambda w: _count_valid(w).astype(float))
    ok = (cnt >= min_valid) & (cnt >= 2)
    return np.where(ok, mean, np.nan), np.where(ok, sd, np.nan), cnt


# ---------------------------------------------------------------------------
# One prepared series
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SpikeSeries:
    """A sealed series after the data-quality rules.

    Arrays are aligned to `dates`:
      raw                raw closes as loaded (NaN only when drop_missing=False)
      p                  cleaned closes: NaN for every invalid session
      nonpos             raw close finite and <= 0 (kept distinct)
      missing            raw close not finite
      external_corrected set missing by the external same-day reference check;
                         the row stays in the calendar
    `extra_lag` (FRED publication lag): sessions every entry (E0 at t0 + 1 + L, E1 at
    u + 1 + L for u found in the unchanged (t0, t0 + 5] window) and every
    signal-driven fill (k + 1 + L) is delayed (1 for FRED series, else 0);
    baselines M and sigma21 stay computed at t0. Nothing in this class or in the data rules
    reads a value after the close it describes.
    """
    name: str
    cutoff: pd.Timestamp
    dates: pd.DatetimeIndex
    raw: np.ndarray
    p: np.ndarray
    nonpos: np.ndarray
    missing: np.ndarray
    external_corrected: Optional[np.ndarray] = None
    extra_lag: int = 0
    n_post_cutoff_dropped: int = 0
    n_pre_start_dropped: int = 0
    n_missing_dropped: int = 0
    n_reference_unchecked: int = 0
    _context: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.external_corrected is None:
            self.external_corrected = np.zeros(len(self.dates), dtype=bool)

    @property
    def n(self) -> int:
        return int(len(self.dates))

    @property
    def nonpos_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.nonpos]

    @property
    def external_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.external_corrected]

    @property
    def missing_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.missing]

    def context(self) -> "SeriesContext":
        """Per-session median M and sigma21 arrays (computed once, memoized)."""
        if self._context is None:
            self._context = series_context(self)
        return self._context

    def counts(self) -> dict:
        """Every data-quality count the report needs."""
        return {
            "rows": self.n,
            "post_cutoff_dropped": int(self.n_post_cutoff_dropped),
            "pre_start_dropped": int(self.n_pre_start_dropped),
            "missing_rows_dropped": int(self.n_missing_dropped),
            "external_corrections": int(self.external_corrected.sum()),
            "reference_unchecked": int(self.n_reference_unchecked),
            "nonpositive": int(self.nonpos.sum()),
            "missing": int(self.missing.sum()),
        }


def _external_cross_check(index: pd.DatetimeIndex, p: np.ndarray, reference,
                          tolerance: float, cutoff: pd.Timestamp):
    """External same-day reference check: flag every valid close that differs from the SAME-DAY value of
    an independent reference series by more than `tolerance` (relative to the
    reference). Returns (mask, n_unchecked). The reference is truncated at the
    cutoff first, so a reference value after it cannot matter; a session with
    no usable reference value (absent row, NaN, non-positive) is left alone and
    counted as unchecked. Only the same day's two values are compared, never
    any later return."""
    ref = pd.Series(np.asarray(reference.to_numpy(dtype=float)),
                    index=pd.DatetimeIndex(reference.index))
    if ref.index.tz is not None:
        ref.index = ref.index.tz_localize(None)
    ref.index = ref.index.normalize()
    if not ref.index.is_unique:
        raise ValueError("reference dates must be unique")
    ref = ref[ref.index <= pd.Timestamp(cutoff)]
    r = ref.reindex(index).to_numpy(dtype=float)
    checkable = np.isfinite(p) & (p > 0)
    has_ref = np.isfinite(r) & (r > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.abs(p - r) / r
    external = checkable & has_ref & (off > tolerance)
    return external, int((checkable & ~has_ref).sum())


def prepare_series(sealed: SealedSeries, *, reference: Optional[pd.Series] = None,
                   reference_tolerance: float = EXTERNAL_CHECK_TOLERANCE,
                   extra_lag: int = 0) -> SpikeSeries:
    """Apply the study's data rules to a sealed series.

    Re-asserts the seal first (a hand-built SealedSeries cannot smuggle a
    later date past the loader). Non-finite closes are `missing`;
    non-positive closes are `nonpos` (both become NaN in `p`). The ONLY
    correction is the optional external cross-check: with a `reference`
    series (JPY=X against FRED DEXJPUS) a close more than
    `reference_tolerance` (1 percent) away from the same-day reference value
    is set missing, listed and counted (`external_corrected`). No rule looks
    at any later close. `extra_lag` >= 0 is carried on the result (FRED
    publication lag). Returned `p` is what every trigger,
    baseline and simulation reads. A series named in EXTERNAL_CHECK_SERIES
    (JPY=X) cannot be prepared without a `reference`.
    """
    assert_sealed(sealed.index, sealed.cutoff)
    if sealed.name in EXTERNAL_CHECK_SERIES and reference is None:
        raise ValueError(
            f"{sealed.name} must be built with its same-day external reference (FRED DEXJPUS): "
            "its only data correction is that cross-check, and without it uncorrected ticks pass silently")
    if not (isinstance(extra_lag, (int, np.integer)) and not isinstance(extra_lag, bool)
            and extra_lag >= 0):
        raise ValueError(f"extra_lag must be a non-negative integer, got {extra_lag!r}")
    raw = np.asarray(sealed.close, dtype=float).copy()
    finite = np.isfinite(raw)
    nonpos = finite & (raw <= 0)
    missing = ~finite
    p = np.where(finite & (raw > 0), raw, np.nan)
    external = np.zeros(len(raw), dtype=bool)
    unchecked = 0
    if reference is not None:
        external, unchecked = _external_cross_check(sealed.index, p, reference,
                                                    reference_tolerance, sealed.cutoff)
        p = np.where(external, np.nan, p)
    return SpikeSeries(name=sealed.name, cutoff=sealed.cutoff, dates=sealed.index,
                       raw=raw, p=p, nonpos=nonpos, missing=missing,
                       external_corrected=external, extra_lag=int(extra_lag),
                       n_post_cutoff_dropped=sealed.n_post_cutoff_dropped,
                       n_pre_start_dropped=sealed.n_pre_start_dropped,
                       n_missing_dropped=sealed.n_missing_dropped,
                       n_reference_unchecked=unchecked)


def build_series(close: Union[pd.Series, pd.DataFrame], *, name: str = "series",
                 cutoff: pd.Timestamp = DISCOVERY_CUTOFF,
                 first_date: Optional[pd.Timestamp] = None, drop_missing: bool = True,
                 reference: Optional[pd.Series] = None,
                 reference_tolerance: float = EXTERNAL_CHECK_TOLERANCE,
                 extra_lag: int = 0) -> SpikeSeries:
    """Convenience: `load_sealed` then `prepare_series`."""
    return prepare_series(load_sealed(close, name=name, cutoff=cutoff, first_date=first_date,
                                      drop_missing=drop_missing),
                          reference=reference, reference_tolerance=reference_tolerance,
                          extra_lag=extra_lag)


def largest_abs_returns(series: SpikeSeries, k: int = LARGEST_RETURNS_LISTED) -> pd.DataFrame:
    """The `k` (20) largest absolute one-session log returns of a series, for
    the reader to inspect (nothing is altered on the strength of
    them). One row per adjacent pair of sessions whose two closes are both
    valid; columns date, prev_date, log_return, abs_log_return; ordered by
    absolute size, ties by date."""
    p = series.p
    prev = np.r_[np.nan, p[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(p / prev)
    idx = np.flatnonzero(np.isfinite(r))
    order = idx[np.argsort(-np.abs(r[idx]), kind="stable")][:k]
    return pd.DataFrame({"date": series.dates[order], "prev_date": series.dates[order - 1],
                         "log_return": r[order], "abs_log_return": np.abs(r[order])},
                        columns=["date", "prev_date", "log_return", "abs_log_return"])


# ---------------------------------------------------------------------------
# The three trigger families (causal boolean series)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class TriggerSignals:
    """One trigger's causal signal arrays, aligned to the series dates.

      up / down          the flags (what `extract_episodes` consumes)
      raw_up / raw_down  the THRESHOLD criterion alone (for z20: z >= Z or
                         z <= -Z before the fresh-extreme requirement)
      fresh_up / _down   P(t) equals the max / min close of the last 20
                         sessions (t-19 .. t)
      defined            the flag question can be answered at t: the
                         baseline exists and P(t) (and, for z20, P(t-20))
                         is valid. Sessions where it is False are the
                         "undefined sessions" the report lists.
      values             the statistic itself: {"r20", "z"} for z20,
                         {"hi", "lo"} thresholds for the percentile triggers
    For the percentile triggers the fresh-extreme filter is NOT part of the
    rule, so `up == raw_up`; `fresh_*` is kept only so the
    roll-off share below is defined for every trigger.
    """
    trigger: str
    up: np.ndarray
    down: np.ndarray
    raw_up: np.ndarray
    raw_down: np.ndarray
    fresh_up: np.ndarray
    fresh_down: np.ndarray
    defined: np.ndarray
    values: dict

    def flags(self, side: str) -> np.ndarray:
        """Flag array for a side name ('up_fade' or 'down_fade')."""
        if side not in SIDE_SIGN:
            raise ValueError(f"unknown side {side!r}")
        return self.up if side == "up_fade" else self.down


def _rows_nanmax(win: np.ndarray) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(win, axis=1)


def _rows_nanmin(win: np.ndarray) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(win, axis=1)


def _fresh_extremes(p: np.ndarray):
    """(fresh_up, fresh_down): P(t) is finite and equals the max (min) of the
    closes at t-19 .. t (the last 20 sessions, the signal close included)."""
    hi = rolling_apply(p, Z_LOOKBACK, 0, _rows_nanmax)
    lo = rolling_apply(p, Z_LOOKBACK, 0, _rows_nanmin)
    with np.errstate(invalid="ignore"):
        return np.isfinite(p) & (p == hi), np.isfinite(p) & (p == lo)


def _z20_values(p: np.ndarray):
    """(r20, z). r20(t) = ln(P(t) / P(t-20)); z uses the mean and sd (ddof =
    1) of r20 over the 252 sessions ending t-1 (NaN-tolerant, >= 200 valid),
    and only from the 273rd close on."""
    n = len(p)
    r20 = np.full(n, np.nan)
    if n > Z_LOOKBACK:
        with np.errstate(invalid="ignore", divide="ignore"):
            r20[Z_LOOKBACK:] = np.log(p[Z_LOOKBACK:] / p[:-Z_LOOKBACK])
    mean, sd, _ = lagged_window_stats(r20, BASELINE_WINDOW, MIN_VALID_BASELINE)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (r20 - mean) / sd
    z = np.where(np.isfinite(z) & (sd > 0), z, np.nan)
    z[:Z20_MIN_CLOSES - 1] = np.nan
    return r20, z


def _pct_rows(q: float, min_valid: int):
    def reducer(win: np.ndarray) -> np.ndarray:
        valid = np.isfinite(win)
        cnt = valid.sum(axis=1)
        out = np.full(len(win), np.nan)
        full = cnt == win.shape[1]
        if full.any():
            out[full] = np.percentile(win[full], q, axis=1, method="linear")
        for i in np.flatnonzero(~full & (cnt >= min_valid)):
            out[i] = np.percentile(win[i][valid[i]], q, method="linear")
        return out
    return reducer


def _pct_sorted(srt: list, q: float) -> float:
    """numpy 'linear' percentile of an ascending python list, replicating
    numpy's virtual-index and lerp arithmetic step for step so the result is
    bit-identical to np.percentile(srt, q, method='linear')."""
    m = len(srt)
    virt = (m - 1) * (q / 100.0)
    if virt >= m - 1:
        return srt[-1]
    prev = int(math.floor(virt))
    a, b = srt[prev], srt[prev + 1]
    g = virt - prev
    diff = b - a
    return b - diff * (1 - g) if g >= 0.5 else a + diff * g


def expanding_percentiles(x: np.ndarray, qs: Sequence[float], min_count: int):
    """For each session t, the numpy-linear percentiles `qs` of the valid
    values of x[0 .. t-1] (the window ENDS at t-1), or NaN while fewer than
    `min_count` valid values exist. Returns one array per q. Incremental
    (sorted insertion), so a full history costs far less than re-sorting at
    every session; results are bit-identical to np.percentile.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    outs = [np.full(n, np.nan) for _ in qs]
    srt: list = []
    for t in range(n):
        if len(srt) >= min_count:
            for out, q in zip(outs, qs):
                out[t] = _pct_sorted(srt, q)
        v = x[t]
        if np.isfinite(v):
            bisect.insort(srt, float(v))
    return tuple(outs)


def compute_triggers(series: SpikeSeries) -> dict:
    """All four triggers for one series: {trigger name: TriggerSignals}.

    z20@2.0 / z20@2.5: up when z >= Z AND P(t) is the
    20-session maximum close (a fresh extreme, so the anchor is the spike and
    not the roll-off of an old low); down when z <= -Z AND P(t) is the
    20-session minimum. pct756: P(t) at or above the 95th (at or below the
    5th) percentile of the 756 closes ending t-1, rolling. pctexp: the same
    thresholds over an expanding window from the first close through t-1,
    needing at least 756 valid closes.

    Everything is causal: baselines end at t-1, P(t) is the only same-day
    input. `series.p` is used as is (invalid sessions are NaN).
    """
    assert_sealed(series.dates, series.cutoff)
    p = series.p
    fresh_up, fresh_down = _fresh_extremes(p)
    out = {}

    r20, z = _z20_values(p)
    for Z in Z_GRID:
        name = f"z20@{Z:.1f}"
        with np.errstate(invalid="ignore"):
            raw_up = np.isfinite(z) & (z >= Z)
            raw_down = np.isfinite(z) & (z <= -Z)
        out[name] = TriggerSignals(
            trigger=name, up=raw_up & fresh_up, down=raw_down & fresh_down,
            raw_up=raw_up, raw_down=raw_down, fresh_up=fresh_up,
            fresh_down=fresh_down, defined=np.isfinite(z),
            values={"r20": r20, "z": z})

    def _pct_signals(name, hi, lo):
        with np.errstate(invalid="ignore"):
            raw_up = np.isfinite(hi) & np.isfinite(p) & (p >= hi)
            raw_down = np.isfinite(lo) & np.isfinite(p) & (p <= lo)
        return TriggerSignals(
            trigger=name, up=raw_up, down=raw_down, raw_up=raw_up,
            raw_down=raw_down, fresh_up=fresh_up, fresh_down=fresh_down,
            defined=np.isfinite(hi) & np.isfinite(p), values={"hi": hi, "lo": lo})

    out["pct756"] = _pct_signals(
        "pct756",
        rolling_apply(p, PCT_WINDOW, 1, _pct_rows(PCT_UP, MIN_VALID_PCT)),
        rolling_apply(p, PCT_WINDOW, 1, _pct_rows(PCT_DOWN, MIN_VALID_PCT)))
    hi, lo = expanding_percentiles(p, (PCT_UP, PCT_DOWN), PCTEXP_MIN_CLOSES)
    out["pctexp"] = _pct_signals("pctexp", hi, lo)
    return out


def trigger_diagnostics(series: SpikeSeries, sig: TriggerSignals) -> dict:
    """The per-(series, trigger) report items: first usable
    signal date, undefined sessions, roll-off share. Per side: raw threshold
    days, flagged days, first flag date and the ROLL-OFF SHARE.

    Roll-off share = the fraction of threshold-crossing days on which P(t) was
    NOT a fresh 20-session extreme, i.e. days the z (or percentile) crossed
    because of what dropped out of the anchor or because the level simply
    persisted, not because a new extreme was printed. For z20 those are the
    days the fresh-extreme rule removes (0 = the rule removed nothing, 1 =
    everything raw was roll-off). For the percentile triggers the rule is not
    applied; the same fraction is reported as a descriptive statistic.
    (The pre-registration names the statistic but does not define it: this
    definition is an open choice.)
    """
    n = series.n
    idx = np.flatnonzero(sig.defined)
    first = int(idx[0]) if len(idx) else None
    n_undefined = int((~sig.defined).sum())
    d = {
        "trigger": sig.trigger,
        "n_sessions": int(n),
        "first_usable_index": first,
        "first_usable_date": series.dates[first] if first is not None else None,
        "n_undefined": n_undefined,
        "n_warmup": int(first) if first is not None else int(n),
        "n_undefined_after_first_usable": (
            int((~sig.defined[first:]).sum()) if first is not None else 0),
    }
    for side, raw, fresh, flags in (
            ("up_fade", sig.raw_up, sig.fresh_up, sig.up),
            ("down_fade", sig.raw_down, sig.fresh_down, sig.down)):
        n_raw = int(raw.sum())
        flagged = np.flatnonzero(flags)
        d[side] = {
            "n_raw": n_raw,
            "n_flagged": int(len(flagged)),
            "first_flag_date": series.dates[flagged[0]] if len(flagged) else None,
            "rolloff_share": (1.0 - float((raw & fresh).sum()) / n_raw) if n_raw else float("nan"),
        }
    return d


# ---------------------------------------------------------------------------
# gap20 episodes and uniform censoring
# ---------------------------------------------------------------------------

def extract_episodes(flags: np.ndarray, gap: int = GAP_SESSIONS) -> np.ndarray:
    """Episode anchors from a boolean flag array.

    A flagged day starts a NEW episode only if at least `gap` sessions have
    elapsed since the previous flagged day, ANY flagged day and not merely
    the previous episode anchor, so a run of flags no more than `gap - 1`
    apart chains into one episode. The episode is anchored on its first
    flagged day. Returns the anchors as ascending int positions.
    """
    pos = np.flatnonzero(np.asarray(flags, dtype=bool))
    if len(pos) == 0:
        return pos.astype(np.int64)
    new = np.empty(len(pos), dtype=bool)
    new[0] = True
    new[1:] = np.diff(pos) >= gap
    return pos[new].astype(np.int64)


def censor_tail(extra_lag: int = 0) -> int:
    """Sessions after t0 that must exist for an episode to be kept: 6 + 63 + 1
    = 70 plus the series' extra publication lag,
    which pushes every entry, and so every scheduled exit, that much later."""
    return CENSOR_TAIL + int(extra_lag)


def censor_episodes(t0: np.ndarray, n: int, tail: int = CENSOR_TAIL):
    """Uniform censoring: keep an episode only if session
    t0 + `tail` (= 6 + 63 + 1 = 70) exists, i.e. t0 + tail <= n - 1, so every
    cell sees the same episode set. Returns (kept, n_dropped); dropped
    episodes are counted, never truncated."""
    t0 = np.asarray(t0, dtype=np.int64)
    keep = t0 + tail <= n - 1
    return t0[keep], int((~keep).sum())


@dataclass(eq=False)
class EpisodeSet:
    """The episodes of one (series, trigger, side) group.

    t0_all           every anchor before censoring
    t0 / dates       the anchors kept after uniform censoring (positions, dates)
    n_flagged_days   flagged sessions before de-clustering
    dropped_censored anchors dropped by uniform censoring
    """
    series: str
    trigger: str
    side: str
    t0_all: np.ndarray
    t0: np.ndarray
    dates: pd.DatetimeIndex
    n_flagged_days: int
    dropped_censored: int

    @property
    def n_episodes(self) -> int:
        return int(len(self.t0_all))

    @property
    def n_kept(self) -> int:
        return int(len(self.t0))


def build_episode_set(series: SpikeSeries, sig: TriggerSignals, side: str) -> EpisodeSet:
    """gap20 episodes for one trigger and side, uniformly censored."""
    flags = sig.flags(side)
    t0_all = extract_episodes(flags)
    kept, dropped = censor_episodes(t0_all, series.n, tail=censor_tail(series.extra_lag))
    return EpisodeSet(series=series.name, trigger=sig.trigger, side=side,
                      t0_all=t0_all, t0=kept, dates=series.dates[kept],
                      n_flagged_days=int(flags.sum()), dropped_censored=dropped)


def build_all_episode_sets(series: SpikeSeries, signals: Optional[dict] = None) -> dict:
    """{(trigger, side): EpisodeSet} for all 4 triggers x 2 sides."""
    signals = signals if signals is not None else compute_triggers(series)
    return {(trig, side): build_episode_set(series, signals[trig], side)
            for trig in TRIGGERS for side in SIDES}


# ---------------------------------------------------------------------------
# Per-session context (median M, sigma21 with its floor)
# ---------------------------------------------------------------------------

CTX_OK = 0
CTX_UNDEFINED = 1          # window not full, too few valid values, or no floor history
CTX_TOUCH_NONPOS = 2       # the window contains a non-positive close


@dataclass(eq=False)
class SeriesContext:
    """Per-session arrays every episode context is read from.

      med / med_status          M = median of P over the 252 sessions ending
                                t (inclusive), NaN-tolerant with >= 200 valid
      sigma_raw                 sd (ddof 1) of the 63 daily log returns ending
                                t, times sqrt(21); NaN unless status OK
      sigma_floor               the series' causal expanding 10th percentile
                                of sigma_raw over sessions 0..t (inclusive)
      sigma                     max(sigma_raw, sigma_floor): the trade's sigma21
      sigma_status              CTX_OK / CTX_UNDEFINED / CTX_TOUCH_NONPOS

    Because every array at t uses only data through t, an episode's context
    is simply indexed at its anchor t0; the null machinery reads pseudo-
    episodes' context from the same arrays.
    """
    med: np.ndarray
    med_status: np.ndarray
    sigma_raw: np.ndarray
    sigma_floor: np.ndarray
    sigma: np.ndarray
    sigma_status: np.ndarray


def _window_touch(flag: np.ndarray, window: int) -> np.ndarray:
    """touch[t] = any(flag[t - window + 1 .. t]); False where the window does
    not fit inside the series."""
    n = len(flag)
    out = np.zeros(n, dtype=bool)
    if n >= window:
        cs = np.r_[0, np.cumsum(flag.astype(np.int64))]
        t = np.arange(window - 1, n)
        out[window - 1:] = (cs[t + 1] - cs[t + 1 - window]) > 0
    return out


def _median_rows(min_valid: int):
    def reducer(win: np.ndarray) -> np.ndarray:
        valid = np.isfinite(win)
        cnt = valid.sum(axis=1)
        out = np.full(len(win), np.nan)
        full = cnt == win.shape[1]
        if full.any():
            out[full] = np.median(win[full], axis=1)
        for i in np.flatnonzero(~full & (cnt >= min_valid)):
            out[i] = np.median(win[i][valid[i]])
        return out
    return reducer


def series_context(series: SpikeSeries) -> SeriesContext:
    """Compute the per-session context arrays for one prepared series.

    A non-positive close removes only the CELLS whose own median
    window, sigma window or path touches it, so a touched window is flagged
    CTX_TOUCH_NONPOS rather than silently recomputed without the bad value
    (a sigma that skips the extreme returns around a negative price would be
    biased low). Sigma windows touched this way are also left out of the
    floor's history. A merely missing (NaN) close is tolerated: the median
    window needs >= 200 valid closes, the sigma window >= 50 valid returns
    (the pre-registration is silent on the 63-return window; same 200/252
    ratio, an open choice), the floor >= 200 valid sigma values.
    """
    assert_sealed(series.dates, series.cutoff)
    p = series.p
    n = len(p)

    # median M -------------------------------------------------------------
    med = rolling_apply(p, MEDIAN_WINDOW, 0, _median_rows(MIN_VALID_BASELINE))
    med_touch = _window_touch(series.nonpos, MEDIAN_WINDOW)
    med_status = np.where(med_touch, CTX_TOUCH_NONPOS,
                          np.where(np.isfinite(med), CTX_OK, CTX_UNDEFINED)).astype(np.int8)
    med = np.where(med_status == CTX_OK, med, np.nan)

    # sigma21 --------------------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
    r = np.full(n, np.nan)
    r[1:] = lp[1:] - lp[:-1]
    sd = rolling_apply(r, SIGMA_WINDOW, 0, _rows_sd)
    cnt = rolling_apply(r, SIGMA_WINDOW, 0, lambda w: _count_valid(w).astype(float))
    sigma_touch = _window_touch(series.nonpos, SIGMA_WINDOW + 1)   # 64 closes = 63 returns
    ok_raw = (cnt >= MIN_VALID_SIGMA_RETURNS) & np.isfinite(sd) & ~sigma_touch
    ok_raw[:SIGMA_WINDOW] = False                                  # needs a full 63 returns
    sigma_raw = np.where(ok_raw, sd * math.sqrt(SIGMA_HORIZON), np.nan)
    # causal expanding floor through t INCLUSIVE: `expanding_percentiles`
    # ends its window at t-1, so pad one NaN and read it shifted by one session.
    # Numerical guard: the floor needs at least 200 valid PRIOR values (sessions before
    # t), otherwise the episode is removed.
    floor = expanding_percentiles(np.r_[sigma_raw, np.nan], (SIGMA_FLOOR_PERCENTILE,), 1)[0][1:]
    n_prior = np.r_[0, np.cumsum(np.isfinite(sigma_raw))[:-1]]
    floor = np.where(n_prior >= MIN_SIGMA_FLOOR_HISTORY, floor, np.nan)
    with np.errstate(invalid="ignore"):
        combined = np.maximum(sigma_raw, floor)
        usable = np.isfinite(combined) & (combined > 0)      # numerical guard: finite and positive
    sigma_status = np.where(sigma_touch, CTX_TOUCH_NONPOS,
                            np.where(usable, CTX_OK, CTX_UNDEFINED)).astype(np.int8)
    sigma = np.where(sigma_status == CTX_OK, combined, np.nan)
    return SeriesContext(med=med, med_status=med_status, sigma_raw=sigma_raw,
                         sigma_floor=floor, sigma=sigma, sigma_status=sigma_status)


# ---------------------------------------------------------------------------
# The episode-level trade simulator (REFERENCE implementation)
# ---------------------------------------------------------------------------
# `simulate_episode` is the plain, sequential, rule-literal simulator: one
# episode, one variant, one close at a time. It is the definition the rest of
# the module is held to: `evaluate_variants` (numpy, all 72 variants at once)
# must give identical answers, which the test-suite proves on random data with
# NaN, non-positive and end-of-data cases.

NO_FLAG = _BIG      # "no TP / stop flag on this path" offset sentinel


def _sign(side: Union[str, int]) -> int:
    """+1 = short the traded series (up_fade), -1 = long (down_fade)."""
    if isinstance(side, str):
        if side not in SIDE_SIGN:
            raise ValueError(f"unknown side {side!r}")
        return SIDE_SIGN[side]
    if side in (1, -1):
        return int(side)
    raise ValueError(f"side must be a side name or +1/-1, got {side!r}")


def resolve_first_flag(k_tp, k_sl):
    """First-flag-wins with stop-wins-on-tie.

    `k_tp` / `k_sl` are the offsets (sessions after entry) of the first
    take-profit / stop flag, `NO_FLAG` when there is none. Returns
    (k, reason): the earlier flag; on the same close the STOP wins; with no
    flag at all (`NO_FLAG`, EXIT_TIME). Works on scalars (returns ints) and
    on arrays (returns arrays). Note: under the pre-registered definitions
    the two flags can never coincide on one close (a take-profit needs a
    favourable move, a stop an adverse one), so the tie branch is defensive.
    """
    a = np.asarray(k_tp)
    b = np.asarray(k_sl)
    k = np.minimum(a, b)
    reason = np.where(k >= NO_FLAG, EXIT_TIME, np.where(b <= a, EXIT_STOP, EXIT_TP))
    if k.ndim == 0:
        return int(k), int(reason)
    return k, reason


@dataclass(frozen=True)
class TradeOutcome:
    """One (episode, variant) result. `status` is a STATUS_* code; the other
    fields are meaningful only when `status == STATUS_TRADED`."""
    status: int
    pnl: float = float("nan")          # -s * (P_exit / P_entry - 1), per unit notional
    entry_idx: int = -1
    exit_idx: int = -1
    exit_reason: int = -1              # EXIT_TIME / EXIT_TP / EXIT_STOP
    tp_not_applicable: bool = False    # a TP cell whose D <= 0: treated as TP = none


def _not_traded(status: int, **kw) -> TradeOutcome:
    return TradeOutcome(status=status, **kw)


def simulate_episode(p: np.ndarray, nonpos, t0: int, side: Union[str, int],
                     entry: str, H: int, tp: str, sl: Optional[float], *,
                     M: float = float("nan"), med_status: int = CTX_OK,
                     sigma: float = float("nan"), sigma_status: int = CTX_OK,
                     extra_lag: int = 0) -> TradeOutcome:
    """Simulate ONE episode under ONE (entry, H, TP, SL) variant.

    `p` = cleaned closes (NaN invalid), `nonpos` = boolean mask of the
    non-positive ones (or None), `t0` = the anchor position, `side` = a side
    name or the position sign (+1 short / -1 long). `M`, `sigma` and their
    statuses come from `SeriesContext` at t0 (baselines are never lagged).
    `extra_lag` L >= 0 (FRED publication lag): every order fills
    L sessions later than it otherwise would. E0 enters at t0 + 1 + L. E1
    scans the UNCHANGED confirmation window (t0, t0 + 5] for the first
    confirming session u and enters at u + 1 + L (the lag sits between the
    confirming observation and the fill, not on the scan). A stop / take-profit
    flag fills at k + 1 + L. The scheduled time stop is measured from the
    entry session and carries no lag; a flag whose delayed fill would land
    after it is overtaken by it (the position leaves at the time stop).

    Sequence and status precedence (the vectorized code replicates it):
      1. sigma21 unusable -> STATUS_UNDEFINED_SIGMA / STATUS_SIGMA_TOUCH_NONPOS
         (every cell of the episode, x = P&L / sigma21 would be undefined);
      2. entry: E0 = the close of t0 + 1 + L. E1 = the first u in (t0, t0 + 5]
         whose close is STRICTLY below (up side) / above (down side) the
         previous session's close (equal closes do not confirm), entering at
         the close of u + 1 + L; none ->
         STATUS_NO_E1_CONFIRM; data ends first -> STATUS_CENSORED. A NaN
         close cannot confirm;
      3. an entry close that is invalid -> STATUS_TOUCH_NONPOS /
         STATUS_TOUCH_MISSING;
      4. TP cells: median unusable -> STATUS_UNDEFINED_M / _M_TOUCH_NONPOS;
         D = s * (P_entry - M) <= 0 -> the cell is TP = none and
         `tp_not_applicable` is set;
      5. walk k = 1, 2, ... (cell-specific): a STOP cell (SL not
         none) needs every close on its entry-to-exit path valid, because the
         stop uses a log return, and is removed (STATUS_TOUCH_*) by an invalid
         (non-positive or missing) one; a time-only or TP cell needs only its
         ENTRY and EXIT prices (and, for TP, its median window), so an
         invalid close on the way is skipped (it can fire no flag) and only an
         invalid exit / fill close removes it. Past the last close ->
         STATUS_CENSORED. k = H exits at that close
         (a TP or stop condition on it resolves to the time stop); for k < H
         a stop (s * ln(P_k / P_entry) >= SL * sigma21) or take-profit
         (s * (P_entry - P_k) >= fraction * D) flag fills at the close of
         k + 1 + L (a gap fills at the next close, not at the level). The
         walk stops at the exit, so an invalid close AFTER it is harmless.
    """
    if not (isinstance(extra_lag, (int, np.integer)) and extra_lag >= 0):
        raise ValueError(f"extra_lag must be a non-negative integer, got {extra_lag!r}")
    L = int(extra_lag)
    s = _sign(side)
    p = np.asarray(p, dtype=float)
    n = len(p)
    npos = np.zeros(n, dtype=bool) if nonpos is None else np.asarray(nonpos, dtype=bool)

    def _touch(i: int) -> int:
        return STATUS_TOUCH_NONPOS if npos[i] else STATUS_TOUCH_MISSING

    # 1. sigma
    if sigma_status == CTX_TOUCH_NONPOS:
        return _not_traded(STATUS_SIGMA_TOUCH_NONPOS)
    if sigma_status != CTX_OK or not (np.isfinite(sigma) and sigma > 0):   # numerical guard
        return _not_traded(STATUS_UNDEFINED_SIGMA)

    # 2. entry
    if entry == "E0":
        e = t0 + 1 + L
    elif entry == "E1":
        e = -1
        for u in range(t0 + 1, t0 + E1_CONFIRM_SESSIONS + 1):
            if u > n - 1:
                return _not_traded(STATUS_CENSORED)
            a, b = p[u], p[u - 1]
            if np.isfinite(a) and np.isfinite(b) and ((s > 0 and a < b) or (s < 0 and a > b)):
                e = u + 1 + L
                break
        if e < 0:
            return _not_traded(STATUS_NO_E1_CONFIRM)
    else:
        raise ValueError(f"unknown entry rule {entry!r}")
    if e > n - 1:
        return _not_traded(STATUS_CENSORED)

    # 3. entry close
    if not np.isfinite(p[e]):
        return _not_traded(_touch(e))
    pe = float(p[e])

    # 4. take-profit applicability
    frac = TP_FRACTION[tp]
    D = float("nan")
    tp_na = False
    if frac is not None:
        if med_status == CTX_TOUCH_NONPOS:
            return _not_traded(STATUS_M_TOUCH_NONPOS)
        if med_status != CTX_OK or not np.isfinite(M):
            return _not_traded(STATUS_UNDEFINED_M)
        D = s * (pe - float(M))
        if D <= 0:
            frac = None
            tp_na = True

    # 5. walk
    needs_path = sl is not None           # stop cells use a log return at every close
    for k in range(1, H + 1):
        idx = e + k
        if idx > n - 1:
            return _not_traded(STATUS_CENSORED)
        if not np.isfinite(p[idx]):
            if needs_path or k == H:      # the time-stop close is the exit price
                return _not_traded(_touch(idx))
            continue                      # a close a time-only / TP cell cannot use
        if k == H:
            xi, reason = idx, EXIT_TIME
            break
        pk = float(p[idx])
        stop_flag = sl is not None and s * float(np.log(pk / pe)) >= sl * sigma
        tp_flag = frac is not None and s * (pe - pk) >= frac * D
        if stop_flag or tp_flag:
            f = k + 1 + L
            if f > H:
                continue                    # the scheduled exit comes before the delayed fill
            reason = EXIT_STOP if stop_flag else EXIT_TP
            for kk in range(k + 1, f if needs_path else k + 1):   # flag-to-fill closes are on a stop path
                j = e + kk
                if j > n - 1:
                    return _not_traded(STATUS_CENSORED)
                if not np.isfinite(p[j]):
                    return _not_traded(_touch(j))
            xi = e + f
            if xi > n - 1:
                return _not_traded(STATUS_CENSORED)
            if not np.isfinite(p[xi]):
                return _not_traded(_touch(xi))
            break
    return TradeOutcome(status=STATUS_TRADED, pnl=-s * (float(p[xi]) / pe - 1.0),
                        entry_idx=int(e), exit_idx=int(xi), exit_reason=int(reason),
                        tp_not_applicable=tp_na)


def simulate_from_series(series: SpikeSeries, t0: int, side: Union[str, int],
                         cell: tuple) -> TradeOutcome:
    """`simulate_episode` for one variant `cell = (entry, H, TP, SL)`, reading
    M and sigma21 from the series' own context at t0."""
    ctx = series.context()
    entry, H, tp, sl = cell
    return simulate_episode(series.p, series.nonpos, int(t0), side, entry, H, tp, sl,
                            M=float(ctx.med[t0]), med_status=int(ctx.med_status[t0]),
                            sigma=float(ctx.sigma[t0]), sigma_status=int(ctx.sigma_status[t0]),
                            extra_lag=series.extra_lag)


def winsorise(x, bound: float = WINSOR_BOUND):
    """Clip x to [-bound, +bound] (plus and minus 5)."""
    return np.clip(np.asarray(x, dtype=float), -bound, bound)


def episode_x(outcome: TradeOutcome, sigma: float) -> float:
    """x = winsorised(P&L / sigma21) for a traded outcome, else NaN."""
    if outcome.status != STATUS_TRADED:
        return float("nan")
    return float(winsorise(outcome.pnl / sigma))


# ---------------------------------------------------------------------------
# Trade simulation at scale: all 72 variants for a set of episodes (numpy)
# ---------------------------------------------------------------------------

_EVAL_CHUNK = 20000        # episodes evaluated per numpy block (bounds memory)


@dataclass(eq=False)
class VariantResults:
    """`evaluate_variants` output. Every array is (n_cells, n_episodes) with
    rows in the order of `cells` (CELL_GRID by default) and columns in the
    order of `t0`.

      status       STATUS_* code (STATUS_NAMES); 0 = traded
      pnl          -s * (P_exit / P_entry - 1), NaN unless traded
      x            winsorised(pnl / sigma21), NaN unless traded
      entry_idx / exit_idx   positions, -1 unless traded
      exit_reason  EXIT_TIME / EXIT_TP / EXIT_STOP, -1 unless traded
      tp_na        traded TP cell whose D <= 0 (treated as TP = none, counted)
    `sigma` (n_episodes,) is the sigma21 used for x and for the stops.
    """
    t0: np.ndarray
    sign: int
    cells: tuple
    status: np.ndarray
    pnl: np.ndarray
    x: np.ndarray
    entry_idx: np.ndarray
    exit_idx: np.ndarray
    exit_reason: np.ndarray
    tp_na: np.ndarray
    sigma: np.ndarray

    def row(self, cell: tuple) -> int:
        """Row index of a cell in this result."""
        return self.cells.index(cell)

    @property
    def traded(self) -> np.ndarray:
        return self.status == STATUS_TRADED


def _first_offset(mask: np.ndarray) -> np.ndarray:
    """mask[:, j] refers to offset j + 1 after entry; the first True offset per
    row, or NO_FLAG."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1) + 1, NO_FLAG).astype(np.int64)


def _evaluate_chunk(p, nonpos, t0, s, sigma, sigma_status, M, med_status, cells, L=0):
    """Numpy core of `evaluate_variants` for one block of episodes. Replicates
    `simulate_episode`'s sequence and status precedence exactly; see that
    function's docstring for the rules. The trick that makes all 72 variants
    cheap: the first stop flag (per multiple) and first take-profit flag (per
    fraction) over the whole 63-session path do not depend on H, so each is
    found once per episode and entry rule, and every (H, TP, SL) cell is then a
    lookup: a flag at offset k < H exits at k + 1, otherwise the time stop."""
    n = len(p)
    m = len(t0)
    nc = len(cells)
    o_status = np.full((nc, m), STATUS_TRADED, dtype=np.int8)
    o_pnl = np.full((nc, m), np.nan)
    o_entry = np.full((nc, m), -1, dtype=np.int64)
    o_exit = np.full((nc, m), -1, dtype=np.int64)
    o_reason = np.full((nc, m), -1, dtype=np.int8)
    o_tpna = np.zeros((nc, m), dtype=bool)
    if m == 0:
        return o_status, o_pnl, o_entry, o_exit, o_reason, o_tpna

    with np.errstate(invalid="ignore"):
        sigma_ok = (sigma_status == CTX_OK) & np.isfinite(sigma) & (sigma > 0)   # numerical guard
    base = np.full(m, STATUS_TRADED, dtype=np.int64)
    base[~sigma_ok] = STATUS_UNDEFINED_SIGMA
    base[sigma_status == CTX_TOUCH_NONPOS] = STATUS_SIGMA_TOUCH_NONPOS
    m_touch = med_status == CTX_TOUCH_NONPOS
    m_bad = (med_status != CTX_OK) | ~np.isfinite(M)

    by_entry: dict = {}
    for ci, cell in enumerate(cells):
        by_entry.setdefault(cell[0], []).append((ci, cell))

    rows = np.arange(m)
    for entry, items in by_entry.items():
        st = base.copy()
        alive = st == STATUS_TRADED
        # entry ------------------------------------------------------------
        if entry == "E0":
            e = t0 + 1 + L
        else:
            e = np.full(m, -1, dtype=np.int64)
            found = np.zeros(m, dtype=bool)
            for u in range(1, E1_CONFIRM_SESSIONS + 1):
                idx = t0 + u
                inside = idx <= n - 1
                a = p[np.minimum(idx, n - 1)]
                b = p[np.minimum(idx - 1, n - 1)]
                with np.errstate(invalid="ignore"):
                    cmp_ = (a < b) if s > 0 else (a > b)
                newly = inside & np.isfinite(a) & np.isfinite(b) & cmp_ & ~found
                e = np.where(newly, idx + 1 + L, e)
                found |= newly
            nof = alive & ~found
            cens = nof & (t0 + E1_CONFIRM_SESSIONS > n - 1)
            st = np.where(cens, STATUS_CENSORED, st)
            st = np.where(nof & ~cens, STATUS_NO_E1_CONFIRM, st)
            alive = st == STATUS_TRADED
        st = np.where(alive & (e > n - 1), STATUS_CENSORED, st)
        alive = st == STATUS_TRADED
        e_safe = np.where(alive, np.minimum(e, n - 1), 0)
        pe = p[e_safe]
        st = np.where(alive & ~np.isfinite(pe),
                      np.where(nonpos[e_safe], STATUS_TOUCH_NONPOS, STATUS_TOUCH_MISSING), st)
        alive = st == STATUS_TRADED

        # path matrix over offsets 0..Hmax after entry ----------------------
        hmax = max(c[1] for _, c in items)
        K = hmax + 1
        idxs = e_safe[:, None] + np.arange(K)[None, :]
        beyond = idxs > n - 1
        W = np.where(beyond, np.nan, p[np.minimum(idxs, n - 1)])
        with np.errstate(invalid="ignore", divide="ignore"):
            adv = s * np.log(W / pe[:, None])          # adverse move, log terms
            fav = s * (pe[:, None] - W)                # favourable move, price terms
        fk = _first_offset(np.isnan(W)[:, 1:])         # first invalid close on the path
        hit_pos = e_safe + fk
        hit_inside = hit_pos <= n - 1
        hit_nonpos = nonpos[np.minimum(hit_pos, n - 1)]
        Dv = s * (pe - M)

        ksl = {}
        for mult in {c[3] for _, c in items if c[3] is not None}:
            with np.errstate(invalid="ignore"):
                ksl[mult] = _first_offset((adv >= mult * sigma[:, None])[:, 1:])
        ktp = {}
        for fr in {TP_FRACTION[c[2]] for _, c in items if TP_FRACTION[c[2]] is not None}:
            with np.errstate(invalid="ignore"):
                ktp[fr] = _first_offset((fav >= fr * Dv[:, None])[:, 1:])
        no_flag = np.full(m, NO_FLAG, dtype=np.int64)

        # per cell ----------------------------------------------------------
        for ci, (_, H, tp, sl) in items:
            cst = st.copy()
            frac = TP_FRACTION[tp]
            with np.errstate(invalid="ignore"):
                d_pos = Dv > 0
            if frac is not None:
                cst = np.where(alive & m_touch, STATUS_M_TOUCH_NONPOS, cst)
                cst = np.where(alive & m_bad & ~m_touch, STATUS_UNDEFINED_M, cst)
                row_alive = cst == STATUS_TRADED
                kt = np.where(row_alive & d_pos, ktp[frac], no_flag)
                na_candidate = row_alive & ~d_pos
            else:
                row_alive = cst == STATUS_TRADED
                kt = no_flag
                na_candidate = np.zeros(m, dtype=bool)
            ks = ksl[sl] if sl is not None else no_flag
            k, rsn = resolve_first_flag(kt, ks)
            flagged = (k + 1 + L) <= H
            xk = np.where(flagged, k + 1 + L, H)
            rsn = np.where(flagged, rsn, EXIT_TIME)
            px = W[rows, np.minimum(xk, K - 1)]
            if sl is not None:
                # a stop cell needs its whole entry-to-exit path valid (log return)
                touched = row_alive & (fk <= xk)
                out_of_data = ~hit_inside
                bad_nonpos = hit_nonpos
            else:
                # a time-only / TP cell needs only its exit price; unusable closes on
                # the way are skipped (NaN never satisfies a flag test)
                exit_pos = e_safe + xk
                touched = row_alive & np.isnan(px)
                out_of_data = exit_pos > n - 1
                bad_nonpos = nonpos[np.minimum(exit_pos, n - 1)]
            cst = np.where(touched & out_of_data, STATUS_CENSORED, cst)
            cst = np.where(touched & ~out_of_data,
                           np.where(bad_nonpos, STATUS_TOUCH_NONPOS, STATUS_TOUCH_MISSING), cst)
            traded = cst == STATUS_TRADED
            o_status[ci] = cst
            o_pnl[ci] = np.where(traded, -s * (px / pe - 1.0), np.nan)
            o_entry[ci] = np.where(traded, e_safe, -1)
            o_exit[ci] = np.where(traded, e_safe + xk, -1)
            o_reason[ci] = np.where(traded, rsn, -1)
            o_tpna[ci] = traded & na_candidate
    return o_status, o_pnl, o_entry, o_exit, o_reason, o_tpna


def _validate_cell(cell) -> None:
    """A cell is (entry, H, TP, SL). The pre-registered grid is CELL_GRID, but
    any H in 1..63 and any stop multiple > 0 is accepted so that tiny
    hand-computed cases can be run through the numpy engine as well."""
    try:
        entry, H, tp, sl = cell
    except (TypeError, ValueError):
        raise ValueError(f"a cell is (entry, H, TP, SL), got {cell!r}") from None
    if entry not in ENTRY_RULES or tp not in TP_FRACTION:
        raise ValueError(f"unknown entry rule or take-profit in cell {cell!r}")
    if not (isinstance(H, (int, np.integer)) and 1 <= H <= max(H_GRID)):
        raise ValueError(f"H must be an integer in 1..{max(H_GRID)}, got {H!r}")
    if sl is not None and not sl > 0:
        raise ValueError(f"stop multiple must be positive or None, got {sl!r}")


def evaluate_variants(series: SpikeSeries, t0, side: Union[str, int],
                      cells: Iterable[tuple] = CELL_GRID, *,
                      context: Optional[SeriesContext] = None,
                      extra_lag: Optional[int] = None) -> VariantResults:
    """Evaluate exit-parameter variants for a set of episode anchors.

    `t0` = anchor positions in `series` (real episodes, or the pseudo-episodes
    of the null machinery); `side` = a side name or the position sign (+1
    short / -1 long: a vehicle traded against the trigger passes its own sign
    and its own `series`); `cells` = the variants to compute (default all 72;
    pass `[PRIMARY_CELL]` for the max-t test, which is far cheaper);
    `context` = override for the series' own median / sigma21 arrays;
    `extra_lag` = override for the series' own publication lag.

    Identical, cell for cell and episode for episode, to running
    `simulate_from_series` in a loop (proved on random messy data in the
    tests); vectorized across episodes and variants, in blocks of
    `_EVAL_CHUNK` episodes. Pure numpy: there is no separate accelerated
    path to keep in sync with it.
    """
    assert_sealed(series.dates, series.cutoff)
    cells = tuple(cells)
    for c in cells:
        _validate_cell(c)
    t0 = np.asarray(t0, dtype=np.int64)
    if t0.ndim != 1:
        raise ValueError("t0 must be 1-D")
    if len(t0) and (t0.min() < 0 or t0.max() > series.n - 1):
        raise ValueError("episode anchor outside the series")
    s = _sign(side)
    L = series.extra_lag if extra_lag is None else extra_lag
    if not (isinstance(L, (int, np.integer)) and L >= 0):
        raise ValueError(f"extra_lag must be a non-negative integer, got {L!r}")
    L = int(L)
    ctx = context if context is not None else series.context()
    sigma = ctx.sigma[t0]
    parts = []
    step = max(1, int(_EVAL_CHUNK))
    for a in range(0, max(len(t0), 1), step):
        sel = slice(a, a + step)
        parts.append(_evaluate_chunk(series.p, series.nonpos, t0[sel], s, sigma[sel],
                                     ctx.sigma_status[t0[sel]], ctx.med[t0[sel]],
                                     ctx.med_status[t0[sel]], cells, L))
    status, pnl, entry_idx, exit_idx, reason, tp_na = (
        np.concatenate([part[i] for part in parts], axis=1) for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(status == STATUS_TRADED, winsorise(pnl / sigma[None, :]), np.nan)
    return VariantResults(t0=t0, sign=s, cells=cells, status=status, pnl=pnl, x=x,
                          entry_idx=entry_idx, exit_idx=exit_idx, exit_reason=reason,
                          tp_na=tp_na, sigma=sigma)


def cell_status_counts(res: VariantResults, cell: tuple) -> dict:
    """Every observation of one cell accounted for by name: the
    per-status counts sum to the number of episodes, plus how many traded
    episodes had a TP treated as none because D <= 0."""
    i = res.row(cell)
    out = {name: int((res.status[i] == code).sum()) for code, name in STATUS_NAMES.items()}
    out["tp_not_applicable"] = int(res.tp_na[i].sum())
    out["n_episodes"] = int(res.status.shape[1])
    return out


# ---------------------------------------------------------------------------
# Cell statistics
# ---------------------------------------------------------------------------

def cluster_ids(dates, gap_days: int = CLUSTER_DAYS) -> np.ndarray:
    """Macro-event cluster label per episode: episodes whose
    t0 lie within `gap_days` (28) calendar days of each other share a cluster.
    Single-linkage by chain: after sorting by date, a gap of more than
    `gap_days` starts a new cluster, so 1 Jan, 20 Jan and 15 Feb are one
    cluster (19 and 26 days apart) even though the ends are 45 days apart.
    Returns ids 0, 1, ... in the ORDER OF THE INPUT."""
    d = pd.DatetimeIndex(dates).normalize()
    n = len(d)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(d.values, kind="stable")
    gaps = np.diff(d.values[order]) / np.timedelta64(1, "D")
    label_sorted = np.r_[0, np.cumsum(gaps > gap_days)]
    out = np.empty(n, dtype=np.int64)
    out[order] = label_sorted
    return out


@dataclass(frozen=True)
class CellStats:
    """Per-cell statistics of one cell over its TRADED episodes.

      n / eligible     traded episodes; eligible iff n >= MIN_EPISODES (8) AND t
                       is defined (sd(x) = 0 leaves the cell ineligible)
      mean_x = S       mean of x = winsorised(P&L / sigma21)
      t                S / (sd(x, ddof=1) / sqrt(n)); NaN when n < 2 or sd = 0
      median_x, hit_rate (P&L > 0), mean_pnl, worst_pnl / worst_date
      n_clusters       macro-event clusters among the traded episodes
      ci_lo, ci_hi     cluster-bootstrap 90 percent interval for S (NaN with
                       fewer than 2 clusters or bootstrap disabled)
      break_even_bps   the round-trip cost, in basis points of notional, that
                       brings mean raw P&L to zero: 1e4 * mean_pnl. Negative
                       means the cell already loses before any cost. No cost
                       is invented.
    """
    n: int
    eligible: bool
    mean_x: float
    t: float
    median_x: float
    hit_rate: float
    mean_pnl: float
    worst_pnl: float
    worst_date: Optional[pd.Timestamp]
    n_clusters: int
    ci_lo: float
    ci_hi: float
    break_even_bps: float


def cell_stats(x, pnl, dates, *, bootstrap: bool = True,
               seed: int = CLUSTER_BOOTSTRAP_SEED, B: int = CLUSTER_BOOTSTRAP_B,
               level: float = CLUSTER_BOOTSTRAP_LEVEL) -> CellStats:
    """Per-cell statistics for one cell. `x`, `pnl`, `dates` are aligned
    arrays over the cell's TRADED episodes (entries where either is NaN are
    dropped). The cluster bootstrap resamples whole macro-event clusters with
    replacement (`B` times, frozen `seed`) and recomputes S as the mean of x
    over the resampled episodes; the interval is its central `level`
    percentile range. The bootstrap is skipped (NaN interval) when
    `bootstrap` is False, which the null machinery uses to stay fast."""
    x = np.asarray(x, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    dates = pd.DatetimeIndex(dates)
    keep = np.isfinite(x) & np.isfinite(pnl)
    x, pnl, dates = x[keep], pnl[keep], dates[keep]
    n = int(len(x))
    nan = float("nan")
    if n == 0:
        return CellStats(n=0, eligible=False, mean_x=nan, t=nan, median_x=nan, hit_rate=nan,
                         mean_pnl=nan, worst_pnl=nan, worst_date=None, n_clusters=0,
                         ci_lo=nan, ci_hi=nan, break_even_bps=nan)
    S = float(x.mean())
    sd = float(x.std(ddof=1)) if n >= 2 else nan
    # Numerical guard: sd(x) = 0 leaves t undefined. Identical values give a float-noise
    # sd (np.full(12, 0.1).std(ddof=1) is about 1e-17, not 0), so identical x, or an
    # sd at the size of float noise relative to the mean, counts as zero.
    degenerate = n < 2 or float(x.max()) == float(x.min()) or sd <= SD_ZERO_REL_TOL * max(1.0, abs(S))
    t = S / (sd / math.sqrt(n)) if not degenerate else nan
    ids = cluster_ids(dates)
    n_clusters = int(ids.max()) + 1
    ci_lo = ci_hi = nan
    if bootstrap and n_clusters >= 2:
        rng = np.random.default_rng(seed)
        sums = np.bincount(ids, weights=x, minlength=n_clusters)
        cnts = np.bincount(ids, minlength=n_clusters).astype(float)
        draw = rng.integers(0, n_clusters, size=(B, n_clusters))
        Sb = sums[draw].sum(axis=1) / cnts[draw].sum(axis=1)
        lo, hi = 100.0 * (1 - level) / 2, 100.0 * (1 + level) / 2
        ci_lo, ci_hi = (float(v) for v in np.percentile(Sb, [lo, hi], method="linear"))
    worst = int(np.argmin(pnl))
    mean_pnl = float(pnl.mean())
    return CellStats(n=n, eligible=(n >= MIN_EPISODES and math.isfinite(t)), mean_x=S, t=t,
                     median_x=float(np.median(x)), hit_rate=float((pnl > 0).mean()),
                     mean_pnl=mean_pnl, worst_pnl=float(pnl[worst]),
                     worst_date=pd.Timestamp(dates[worst]), n_clusters=n_clusters,
                     ci_lo=ci_lo, ci_hi=ci_hi, break_even_bps=1e4 * mean_pnl)


def cell_stats_from_results(res: VariantResults, cell: tuple, dates: pd.DatetimeIndex,
                            **kw) -> CellStats:
    """`cell_stats` for one cell of a `VariantResults`, over its traded
    episodes; `dates` = the full date index of the series the episodes came
    from (episode dates are `dates[res.t0]`)."""
    i = res.row(cell)
    ok = res.status[i] == STATUS_TRADED
    return cell_stats(res.x[i][ok], res.pnl[i][ok], pd.DatetimeIndex(dates)[res.t0[ok]], **kw)


def entry_d_e0(series: SpikeSeries, side: Union[str, int]) -> np.ndarray:
    """D = s * (P(t + 1 + L) - M(t)) at every session t (L = the series' extra
    lag): the side-of-median quantity of an E0 entry (a TP
    applies only if D > 0; the N1 null matches on the sign of D). NaN
    where the median is unusable, there is no entry session, or its close is
    invalid."""
    s = _sign(side)
    ctx = series.context()
    n = series.n
    d = np.full(n, np.nan)
    shift = 1 + series.extra_lag                 # the E0 entry session is t + 1 + L
    if n > shift:
        with np.errstate(invalid="ignore"):
            d[:-shift] = s * (series.p[shift:] - ctx.med[:-shift])
    return d


# ---------------------------------------------------------------------------
# Vehicle test: mapping trigger-series anchors onto a vehicle's own calendar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleAnchors:
    """`map_anchors_to_vehicle` output.

      positions          vehicle-calendar anchor positions of the KEPT anchors
                         (last vehicle session on or before the trigger date,
                         plus the extra lag), ready for `evaluate_variants` on
                         the vehicle's own SpikeSeries
      source_index       index of each kept anchor in the input anchor dates
      trigger_dates      the kept trigger-series anchor dates
      session_dates      the vehicle date at each kept position (the lagged
                         anchor session; entry is the next vehicle session)
      n_input            anchors offered
      dropped_no_session       no vehicle session on or before the date
      dropped_short_history    fewer than 252 vehicle sessions up to it
      dropped_beyond_end       the lagged position falls past the last session
      n_duplicate_positions    kept anchors sharing a vehicle position (a sparse
                               vehicle calendar can collapse two anchors; they are
                               reported, not merged)
    """
    positions: np.ndarray
    source_index: np.ndarray
    trigger_dates: pd.DatetimeIndex
    session_dates: pd.DatetimeIndex
    n_input: int
    dropped_no_session: int
    dropped_short_history: int
    dropped_beyond_end: int
    n_duplicate_positions: int


def map_anchors_to_vehicle(anchor_dates, vehicle_index, extra_lag: int = 0, *,
                           min_history: int = MEDIAN_WINDOW) -> VehicleAnchors:
    """Tradable-vehicle mapping (deterministic). The vehicle test runs entirely on the
    vehicle's OWN calendar and closes; the trigger series supplies only the
    anchor DATE. Each anchor maps to the vehicle position of the last vehicle
    session on or before that date, plus the trigger series' `extra_lag`
    (FRED publication lag); E0 then enters at the close of the next
    vehicle session, E1 confirms on the vehicle's closes, and D, M, the take-
    profit and stop levels, sigma21 and H all come from the vehicle's own
    SpikeSeries (`evaluate_variants(vehicle, positions, side)`; side = -1 for a
    LONG vehicle position, +1 for a SHORT one).

    Anchors are dropped, and counted, when no vehicle session exists on or
    before the date, when fewer than `min_history` (252) vehicle sessions exist
    up to that session (the median / sigma21 baselines would not fit), or when
    the lagged position falls past the vehicle's last session. Uniform cutoff
    censoring is left to the caller (`censor_episodes(positions, vehicle.n)`).
    `vehicle_index` must be strictly ascending and unique.
    """
    if not (isinstance(extra_lag, (int, np.integer)) and not isinstance(extra_lag, bool)
            and extra_lag >= 0):
        raise ValueError(f"extra_lag must be a non-negative integer, got {extra_lag!r}")
    vi = pd.DatetimeIndex(vehicle_index)
    if vi.tz is not None:
        vi = vi.tz_localize(None)
    vi = vi.normalize()
    if not vi.is_monotonic_increasing or not vi.is_unique:
        raise ValueError("vehicle dates must be strictly ascending and unique")
    ad = pd.DatetimeIndex(anchor_dates)
    if ad.tz is not None:
        ad = ad.tz_localize(None)
    ad = ad.normalize()
    pos0 = np.searchsorted(vi.values, ad.values, side="right") - 1
    no_session = pos0 < 0
    short = ~no_session & (pos0 + 1 < min_history)
    pos = pos0 + int(extra_lag)
    beyond = ~no_session & ~short & (pos > len(vi) - 1)
    keep = ~(no_session | short | beyond)
    kept = pos[keep].astype(np.int64)
    return VehicleAnchors(
        positions=kept, source_index=np.flatnonzero(keep).astype(np.int64),
        trigger_dates=ad[keep], session_dates=vi[kept], n_input=int(len(ad)),
        dropped_no_session=int(no_session.sum()), dropped_short_history=int(short.sum()),
        dropped_beyond_end=int(beyond.sum()),
        n_duplicate_positions=int(len(kept) - len(np.unique(kept))))


# ---------------------------------------------------------------------------
# One group, one series: the 72-cell report
# ---------------------------------------------------------------------------

def sl_label(sl: Optional[float]) -> str:
    """Report label of a stop multiple: 'none', '1.5', '3.0'."""
    return "none" if sl is None else f"{float(sl):.1f}"


@dataclass(eq=False)
class GroupResult:
    """All 72 cells of one (series, trigger, side) group.

      episodes    the EpisodeSet (t0_all / t0 / dropped_censored / ...)
      results     VariantResults for the kept episodes
      stats       {cell: CellStats}
      counts      group-level counts (flagged days, episodes, censored drops,
                  the series' external-correction / non-positive / missing counts)
      checks      `trigger_diagnostics` for this trigger (first usable date,
                  undefined sessions, roll-off share per side)
    """
    series: str
    trigger: str
    side: str
    episodes: EpisodeSet
    results: VariantResults
    stats: dict
    counts: dict
    checks: dict

    def to_frame(self) -> pd.DataFrame:
        """One row per cell: identity, per-cell statistics, per-status counts
        (every removed observation by name), tp_not_applicable, the group
        counts and the series-level counters (`series_*`: rows, dropped rows,
        external corrections, non-positive and missing closes, ...)."""
        return pd.DataFrame(self._rows())

    def _rows(self) -> list:
        d_side = self.checks[self.side]
        rows = []
        for cell in self.results.cells:
            entry, H, tp, sl = cell
            cs = self.stats[cell]
            row = {"series": self.series, "trigger": self.trigger, "side": self.side,
                   "entry": entry, "H": H, "tp": tp, "sl": sl_label(sl),
                   "sl_mult": float("nan") if sl is None else float(sl),
                   "n": cs.n, "eligible": cs.eligible, "mean_x": cs.mean_x, "t": cs.t,
                   "median_x": cs.median_x, "hit_rate": cs.hit_rate,
                   "mean_pnl": cs.mean_pnl, "worst_pnl": cs.worst_pnl,
                   "worst_date": cs.worst_date if cs.worst_date is not None else pd.NaT,
                   "n_clusters": cs.n_clusters, "ci_lo": cs.ci_lo, "ci_hi": cs.ci_hi,
                   "break_even_bps": cs.break_even_bps}
            row.update({k: v for k, v in cell_status_counts(self.results, cell).items()
                        if k != "n_episodes"})
            row.update({"n_flagged_days": self.counts["n_flagged_days"],
                        "n_episodes": self.counts["n_episodes"],
                        "dropped_censored": self.counts["dropped_censored"],
                        "first_usable_date": (self.checks["first_usable_date"]
                                              if self.checks["first_usable_date"] is not None else pd.NaT),
                        "n_undefined": self.checks["n_undefined"],
                        "rolloff_share": d_side["rolloff_share"]})
            row.update({k: v for k, v in self.counts.items() if k.startswith("series_")})
            rows.append(row)
        return rows


def evaluate_group(series: SpikeSeries, trigger: str, side: str,
                   cells: Iterable[tuple] = CELL_GRID, *,
                   signals: Optional[dict] = None, bootstrap: bool = True) -> GroupResult:
    """Episodes, all variants and per-cell statistics for one group."""
    signals = signals if signals is not None else compute_triggers(series)
    sig = signals[trigger]
    es = build_episode_set(series, sig, side)
    res = evaluate_variants(series, es.t0, side, cells)
    stats = {c: cell_stats_from_results(res, c, series.dates, bootstrap=bootstrap)
             for c in res.cells}
    diag = trigger_diagnostics(series, sig)
    counts = {"n_flagged_days": es.n_flagged_days, "n_episodes": es.n_episodes,
              "n_kept": es.n_kept, "dropped_censored": es.dropped_censored}
    counts.update({f"series_{k}": v for k, v in series.counts().items()})
    return GroupResult(series=series.name, trigger=trigger, side=side, episodes=es,
                       results=res, stats=stats, counts=counts, checks=diag)


def evaluate_series(series: SpikeSeries, cells: Iterable[tuple] = CELL_GRID, *,
                    bootstrap: bool = True) -> dict:
    """{(trigger, side): GroupResult} for all 4 triggers x 2 sides of a series
    (triggers computed once)."""
    signals = compute_triggers(series)
    cells = tuple(cells)
    return {(trig, side): evaluate_group(series, trig, side, cells, signals=signals,
                                         bootstrap=bootstrap)
            for trig in TRIGGERS for side in SIDES}


def series_frame(groups: dict) -> pd.DataFrame:
    """One frame of every cell of `evaluate_series` output in the fixed
    (trigger, side) order: 8 x 72 = 576 rows for a full series. Built from the
    row records directly (not by concatenating per-group frames), so groups with
    no episodes, whose statistic columns are all NaN / NaT, cannot trigger
    pandas' empty-or-all-NA concat behaviour change."""
    rows = []
    for trig in TRIGGERS:
        for side in SIDES:
            if (trig, side) in groups:
                rows.extend(groups[(trig, side)]._rows())
    return pd.DataFrame(rows)


def removed_observations(series: SpikeSeries, res: VariantResults) -> pd.DataFrame:
    """Every (episode, cell) that did NOT yield a plain trade, listed by date
    and by name (every affected date is listed, and every dropped /
    undefined / not-applicable observation appears in the report).

    Columns: date, t0, entry, H, tp, sl, status, traded. `status` is a
    STATUS_NAMES value for removed cells, or 'tp_not_applicable' for a traded
    TP cell whose D <= 0 (treated as TP = none; `traded` is True). Sorted by
    anchor, then by CELL_GRID order. Uniform-censoring drops are a property of
    the EpisodeSet (`dropped_censored`), not of a cell, and are not listed here.
    """
    ci, ej = np.nonzero((res.status != STATUS_TRADED) | res.tp_na)
    order = np.lexsort((ci, res.t0[ej]))
    rows = []
    for c, j in zip(ci[order], ej[order]):
        entry, H, tp, sl = res.cells[c]
        st = int(res.status[c, j])
        rows.append({"date": series.dates[res.t0[j]], "t0": int(res.t0[j]), "entry": entry,
                     "H": H, "tp": tp, "sl": sl_label(sl),
                     "status": STATUS_NAMES[st] if st != STATUS_TRADED else "tp_not_applicable",
                     "traded": st == STATUS_TRADED})
    return pd.DataFrame(rows, columns=["date", "t0", "entry", "H", "tp", "sl", "status", "traded"])
