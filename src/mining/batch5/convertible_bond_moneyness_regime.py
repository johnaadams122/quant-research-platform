"""Batch5 candidate: convertible_bond_moneyness_regime (single_long).

Convertible-bond ETF moneyness/delta regime timing: hold CWB (SPDR Bloomberg
Convertible Securities ETF) only in its "equity-like" regime, move to cash
otherwise. Source: RBC Global Asset Management, "Convertible Bonds" investor
primer; Advent Capital Management, "Optimizing an Asset Allocation with
Convertible Securities" (per the curated candidate's own "source"
field). Rule, verbatim from the frozen
catalog entry's "mechanism" field:

    Proxies CWB's aggregate equity-vs-bond delta regime using CWB's own
    trailing 63-day rolling beta to SPY, recomputed daily. Hold CWB when
    rolling beta is at/above its own trailing 2-year median AND trailing
    63-day return is positive (equity-like regime); move to cash when beta
    is below median with negative trailing return (busted/bond-like
    regime).

Engine mapping (mining.backtest_engine): the rule is re-evaluated EVERY bar
("rebalance: daily" -- parameters.rebalance), not just at transitions, so
the per-bar target state IS the per-bar signal: on every bar t,
entry[t] = is_long[t] (fills at t+1's open, the earliest the engine can
buy) and exit[t] = not is_long[t] (fills at t's close). The engine only
reads `entry` while flat and only reads `exit` while a position is open, so
emitting both flags on every bar (not just on state changes) reproduces
"stay long through the equity-like regime, stay in cash through the busted
regime" without the signal function tracking position state itself -- the
same idiom the batch5 FX/vol-managed modules use for their "standing
decision" bars, simplified here because there is no month-end ambiguity:
every bar is a decision bar by construction.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- beta, its trailing median, and CWB's trailing return are
  all computed from right-aligned (trailing, inclusive-of-today) windows
  only; the value stored for date t never reads a close after t.
* PERFORMANCE -- entry/exit are precomputed ONCE in the factory as plain
  numpy arrays; the returned signal_fn does two array lookups plus one
  scalar date check per call, nothing is recomputed over a growing slice.
* HOLDOUT-BLIND -- both frames are truncated to <= the last bar of `index`
  before any computation runs, so a holdout bucket's signals cannot depend
  on data after the bucket; history BEFORE the bucket is kept as warm-up
  (the 63/504-bar lookbacks the first in-bucket decision needs).
* DIRECTION -- always long CWB or flat cash, never short; SPY is a signal
  input only (SIGNAL = ["SPY"]) and is never traded (BENCHMARK == TRADED[0]
  == "CWB", so the candidate is judged against buy-and-hold CWB, not SPY).

Interpretation choices (the catalog entry leaves these to the implementer):

1. CWB and SPY closes are aligned on their COMMON calendar (the
   intersection of their bar dates) before anything is computed -- both are
   NYSE-listed US ETFs so the calendars are expected to agree almost every
   day, but a bucket date absent from the common calendar carries no
   decision (entry=False, exit=False), the same convention the oil-price
   module uses for its SPY/USO alignment.
2. Returns feeding the rolling beta are plain daily simple returns
   (pct_change of the pipeline's auto-adjusted Close, i.e. total-return),
   not log returns -- the catalog's parameters name no return convention,
   and simple returns are the textbook choice for a CAPM-style rolling
   beta.
3. beta_t = 63-trading-day trailing OLS slope of CWB's return on SPY's
   return, i.e. Cov(CWB_ret, SPY_ret) / Var(SPY_ret) over the RIGHT-ALIGNED
   window [t-62, t] (today inclusive) -- the ordinary rolling-beta
   definition. The window is a FULL window only: fewer than
   ``beta_window_days`` valid (finite) paired returns, or a non-finite or
   non-positive SPY variance over the window, yields NaN (no beta) at that
   bar rather than a manufactured value from a degenerate denominator.
   Because Cov/Var is invariant to the covariance/variance ddof convention
   (it cancels in the ratio), beta is computed directly as a sum-of-squares
   slope, sidestepping the ddof choice entirely.
4. The trailing "2-year" (504-trading-day) median of the beta series is
   computed EXPANDING-then-rolling and STRICTLY on already-observed beta
   values: at bar t it is the median of every non-NaN beta value in the
   trailing window ending at t, that window growing from bar 1 up to a cap
   of 504 bars (never wider). Concretely: a plain rolling median with
   ``min_periods=1`` over the (NaN-containing) beta series -- pandas'
   median already skips NaNs within the window, which is exactly
   "expanding until 504 valid observations accumulate, then rolling" once
   beta itself has warmed up (the median needs at least one non-NaN beta in
   its own window to be defined; requiring 504 *valid* beta values before
   ever answering would silently re-describe the window as much wider than
   504 calendar bars whenever beta is intermittently undefined, which the
   catalog's "504-day window" phrasing does not ask for).
5. trailing_return_t = CWB_close_t / CWB_close_{t-63} - 1 (the close
   exactly ``trailing_return_window_days`` common-calendar bars before t);
   undefined (NaN, no decision from this input) if the anchor close is
   non-finite or <= 0, or if fewer than that many prior common-calendar
   bars exist yet.
6. The AND is literal and both-sided inclusive/exclusive exactly as
   written: is_long_t = (beta_t >= median_t) AND (trailing_return_t > 0).
   beta_t == median_t counts as "at/above" (catalog: "at/above its own
   trailing 2-year median", inclusive). trailing_return_t == 0 exactly does
   NOT count as positive (catalog: "trailing 63-day return is positive"),
   so an exactly-flat trailing return is cash, matching the catalog's own
   asymmetric wording (it never calls zero "positive"). is_long_t is False
   whenever ANY of beta_t, median_t or trailing_return_t is undefined (the
   busted regime is the default during warm-up, not the equity-like one) --
   the catalog names only the two regimes the rule cycles between once
   defined, not a third "no data yet" state, and False is also the
   engine-safe default (it never opens a position on undefined inputs).
7. The final bar of the (possibly truncated) bucket DOES make a decision
   (unlike the batch5 month-end modules) -- there is no "is this a
   month-end" ambiguity here; every bar is a decision bar by construction,
   so suppressing the last one would only differ from the engine's own
   end_of_data close in the recorded exit reason, never in P&L, and there
   is no reason to add that suppression for a rule that has no monthly
   grouping to begin with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "convertible_bond_moneyness_regime"
SHAPE = "single_long"
TRADED = ["CWB"]
SIGNAL = ["SPY"]
SNAPSHOTS = {}
BENCHMARK = "CWB"
# Copied byte-for-byte from the candidate JSON "parameters" (driver drift check).
PARAMETERS = {
    "beta_window_days": 63,
    "regime_median_window_days": 504,
    "trailing_return_window_days": 63,
    "rebalance": "daily",
}
# Sanity floor on the in-sample trade count the batch5 windowing machinery
# (run_batch3_mining.run_single_asset_windows with the run's derived
# holdout window) must produce; the driver refuses a run below it as a
# guard against a silently degenerate signal. A daily-re-evaluated regime
# rule switches often, so the floor is correspondingly high.
EXPECTED_MIN_TRADES = 80

_BETA_WINDOW = int(PARAMETERS["beta_window_days"])
_MEDIAN_WINDOW = int(PARAMETERS["regime_median_window_days"])
_RETURN_WINDOW = int(PARAMETERS["trailing_return_window_days"])


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


def _clean_close(frame: pd.DataFrame, last_bar: pd.Timestamp) -> pd.Series:
    """A sorted, de-duplicated, holdout-truncated Close series indexed by a
    tz-naive normalized DatetimeIndex."""
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                       index=_naive_index(frame.index))
    close = close[~close.index.duplicated(keep="last")]
    close = close.sort_index()
    return close[close.index <= last_bar]


def rolling_beta(signal_ret: np.ndarray, traded_ret: np.ndarray,
                 window: int) -> np.ndarray:
    """Right-aligned (causal, inclusive-of-today) trailing ``window``-bar
    OLS-slope beta of ``traded_ret`` on ``signal_ret`` at every position.

    ``beta[t]`` uses only ``signal_ret[t-window+1 : t+1]`` and the matching
    slice of ``traded_ret`` -- never a value after ``t``. NaN wherever the
    window is short of ``window`` observations, contains a non-finite
    return, or has a non-finite/non-positive signal variance (docstring
    choice 3): a degenerate denominator never manufactures a beta.
    """
    if window <= 1:
        raise ValueError("window must be > 1")
    signal_ret = np.asarray(signal_ret, dtype=float)
    traded_ret = np.asarray(traded_ret, dtype=float)
    if signal_ret.shape != traded_ret.shape:
        raise ValueError("signal_ret and traded_ret must be the same shape")
    n = signal_ret.shape[0]
    beta = np.full(n, np.nan, dtype=float)
    for t in range(window - 1, n):
        sig_win = signal_ret[t - window + 1: t + 1]
        trd_win = traded_ret[t - window + 1: t + 1]
        if not (np.isfinite(sig_win).all() and np.isfinite(trd_win).all()):
            continue
        sig_mean = sig_win.mean()
        denom = float(np.sum((sig_win - sig_mean) ** 2))
        if not np.isfinite(denom) or denom <= 0.0:
            continue
        trd_mean = trd_win.mean()
        numer = float(np.sum((sig_win - sig_mean) * (trd_win - trd_mean)))
        beta[t] = numer / denom
    return beta


def trailing_return(close: np.ndarray, window: int) -> np.ndarray:
    """``close[t] / close[t-window] - 1`` (docstring choice 5): NaN when the
    anchor is out of range, non-finite, or <= 0."""
    if window <= 0:
        raise ValueError("window must be positive")
    close = np.asarray(close, dtype=float)
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n <= window:
        return out
    anchor = close[:-window]
    current = close[window:]
    valid = np.isfinite(anchor) & (anchor > 0.0) & np.isfinite(current)
    out[window:] = np.where(valid, current / np.where(valid, anchor, 1.0) - 1.0,
                            np.nan)
    return out


def trailing_median_expanding_then_rolling(values: np.ndarray,
                                           window: int) -> np.ndarray:
    """Docstring choice 4: at position t, the median of every non-NaN value
    in ``values[max(0, t-window+1) : t+1]`` (NaN if none). Delegates to
    pandas' ``rolling(window, min_periods=1).median()``, which is exactly
    this expanding-then-rolling, NaN-skipping definition."""
    if window <= 0:
        raise ValueError("window must be positive")
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window=window, min_periods=1).median().to_numpy(dtype=float)


def regime_state(traded_close: pd.Series, signal_close: pd.Series, *,
                 beta_window: int = _BETA_WINDOW,
                 median_window: int = _MEDIAN_WINDOW,
                 return_window: int = _RETURN_WINDOW) -> pd.Series:
    """Boolean Series, indexed by the common calendar of ``traded_close`` and
    ``signal_close``: True on bars in the equity-like regime (long CWB),
    False everywhere else including every warm-up bar (docstring choice 6)."""
    common = traded_close.index.intersection(signal_close.index).sort_values()
    cwb = pd.to_numeric(traded_close.reindex(common), errors="coerce").to_numpy(dtype=float)
    spy = pd.to_numeric(signal_close.reindex(common), errors="coerce").to_numpy(dtype=float)

    cwb_ret = np.concatenate(([np.nan], np.where(
        np.isfinite(cwb[:-1]) & (cwb[:-1] > 0.0) & np.isfinite(cwb[1:]),
        cwb[1:] / np.where(cwb[:-1] > 0.0, cwb[:-1], 1.0) - 1.0, np.nan))) \
        if len(cwb) else np.asarray([], dtype=float)
    spy_ret = np.concatenate(([np.nan], np.where(
        np.isfinite(spy[:-1]) & (spy[:-1] > 0.0) & np.isfinite(spy[1:]),
        spy[1:] / np.where(spy[:-1] > 0.0, spy[:-1], 1.0) - 1.0, np.nan))) \
        if len(spy) else np.asarray([], dtype=float)

    beta = rolling_beta(spy_ret, cwb_ret, beta_window)
    median = trailing_median_expanding_then_rolling(beta, median_window)
    ret = trailing_return(cwb, return_window)

    is_long = (np.isfinite(beta) & np.isfinite(median) & np.isfinite(ret)
              & (beta >= median) & (ret > 0.0))
    return pd.Series(is_long, index=common)


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED + SIGNAL."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    traded_frame = _require_frame(frames, ticker)
    signal_frame = _require_frame(frames, SIGNAL[0])

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    traded_close = _clean_close(traded_frame, last_bar)
    signal_close = _clean_close(signal_frame, last_bar)

    state = regime_state(traded_close, signal_close)
    entry_series = state
    exit_series = pd.Series(~state.to_numpy(), index=state.index)
    entry_arr = entry_series.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    exit_arr = exit_series.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    # A bucket bar absent from the common calendar carries NO decision at
    # all (docstring choice 1): both flags stay False, so a bar the signal
    # cannot see neither opens a position blind nor force-closes one that is
    # already open -- in practice this branch is unreachable for two
    # NYSE-listed ETFs (see this module's tests), which trade the same
    # calendar essentially always.
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
