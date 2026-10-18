"""Batch5 candidate: long_horizon_index_mean_reversion (pooled_single_asset).

Implements the catalogued mechanism from ``data/mining/catalog.json``
under the ABI-1 batch5 module contract; the points the catalog's prose
leaves open are resolved under "Design choices" below.

Source (verbatim, ``data/mining/catalog.json``, family_id
"long_horizon_index_mean_reversion"): Fama, E.F. &
French, K.R. (1988), "Permanent and Temporary Components of Stock Prices",
Journal of Political Economy 96(2), 246-273. Poterba, J.M. & Summers, L.H.
(1988), "Mean Reversion in Stock Prices: Evidence and Implications",
Journal of Financial Economics 22(1), 27-59 (NBER Working Paper 2343).

Catalog mechanism (verbatim): "Fama & French (1988) and Poterba & Summers
(1988) found that aggregate stock index returns contain a slowly
mean-reverting component, with returns negatively autocorrelated at
roughly 3-to-5-year horizons -- unusually weak (strong) trailing multi-year
performance tends to be followed by above-(below-)average subsequent
returns. Applied to a single index ETF: compute the trailing 36-month
(3-year) cumulative total return and z-score it against its own trailing
15-year rolling history of 36-month returns; when the z-score falls below
-1.0 (an unusually poor 3-year run), go long or overweight for the
following year, and when it rises above +1.0, reduce or flatten the
position. Rebalance monthly."

The mechanism is self-referential (a ticker's own trailing-return
distribution is both the signal and the thing it is scored against,
needing no external fetch -- SIGNAL = [] below), and hysteresis (not a
single toggle) is the reading of "go long below -1.0 ... reduce/flatten
above +1.0" used here.

Design choices (where the catalog's prose leaves room; disclosed, not
silent):

1. TRADING-DAY PROXIES, not calendar ``DateOffset`` windows. "36 months
   ago" / "15 years ago" could be resolved via
   ``pd.DateOffset(months=36)`` / ``pd.DateOffset(years=15)`` and a
   searchsorted lookup on CALENDAR dates. This module instead follows this
   batch's own daily-bar convention (e.g. ``betting_against_beta_low_
   vol.py``'s 252-trading-day beta window, ``preferred_stock_yield_spread_
   reversion.py``'s 756-trading-day z-score window) and works in
   TRADING-DAY proxies throughout -- see "Design constants" below. Simpler
   and consistent with every other daily-bar module in this directory;
   does not require resampling to monthly OHLC.
2. MONTH-END decision bars, not month-START. This module follows the
   ``vol_managed_intl_country_etf_pooled.py`` convention: a decision bar is
   one whose NEXT bar falls in a different calendar month (the LAST
   trading day of the month), and the final bar of a truncated frame is
   never a decision bar. An entry raised there fills at the following
   session's open -- the first session of the new month -- which matches
   this batch's more common monthly-rebalance convention.
3. INCLUSIVE threshold comparisons, not strict: enter at
   ``z <= entry_zscore_threshold`` (-1), exit at
   ``z >= exit_zscore_threshold`` (+1), rather than strictly below -1.0 /
   strictly above +1.0 (which would leave z exactly at a threshold a
   no-op). An exact boundary hit is a measure-zero event on real daily
   closes.

Also by design: a self-referential single-series mechanism (no external
SIGNAL ticker, no bespoke driver script -- every TRADED symbol is a normal
``data_pipeline.fetch_ohlc``-fetchable ticker); hysteresis (not a single
toggle) between the two thresholds; a full lookback window is required
before the rule can fire at all (a partial early window is "not enough
history yet", never scored against a shorter, biased sample).

Design constants
----------------
* ``_TRADING_DAYS_PER_MONTH = 21``, ``_TRADING_DAYS_PER_YEAR = 252`` --
  standard daily-bar proxies used throughout this batch.
* ``RETURN_WINDOW_DAYS = 36 * 21 = 756`` (``PARAMETERS
  ["return_window_months"]``).
* ``ZSCORE_LOOKBACK_DAYS = 15 * 252 = 3780`` (``PARAMETERS
  ["zscore_lookback_years"]``).
* ``trailing_return[t] = Close[t] / Close[t - RETURN_WINDOW_DAYS] - 1``
  (a simple cumulative return over the trailing window; NaN until
  ``RETURN_WINDOW_DAYS`` prior closes exist).
* ``z[t] = (trailing_return[t] - rolling_mean[t]) / rolling_std[t]``, where
  the mean/std are computed over a ROLLING (not expanding)
  ``ZSCORE_LOOKBACK_DAYS``-bar window of trailing_return VALUES ending at
  t, ddof=1, full window required (NaN before). See "Rolling vs expanding"
  below for why ROLLING is the reading used here.
* A rolling std below a relative float-noise floor (matching this batch's
  own established zero-dispersion guard, e.g.
  ``preferred_stock_yield_spread_reversion.trailing_yield_zscore``) is
  treated as undefined (NaN), never a spurious near-infinite z.

Rolling vs expanding (explicit design choice)
---------------------------------------------
The catalog's own mechanism text says "z-score it against its own trailing
15-year ROLLING history of 36-month returns" -- "rolling" is the catalog's
own word, and the more literal reading is a FIXED-length rolling window of
past trailing-return observations (mean/std recomputed over exactly the
most recent 3780 such observations at each t), not an EXPANDING window
that accumulates every trailing-return observation since the ticker's own
inception. This module therefore uses a rolling (not expanding) mean/std.
Consequence, disclosed rather than silently absorbed: an expanding window
would eventually let an old regime keep diluting the statistic forever --
the exact "expanding-window burn-in" pathology a bounded-burn-in variant
exists to fix elsewhere in this pipeline. A rolling window instead
keeps comparing "now" against a genuinely recent 15-year past, matching
the catalog's own word and avoiding that pathology by construction rather
than by a cap.

Hysteresis band (design decision)
---------------------------------
The catalog is silent on the in-between region (-1 < z < +1). This module
reads it the standard way, matching this batch's own entry_zscore/
exit_zscore hysteresis precedent (``preferred_stock_yield_spread_
reversion.standing_long``): a simple two-state (LONG / FLAT) switch with
hysteresis, checked ONLY on decision bars (below). Enter/stay LONG while
``z <= entry_zscore_threshold`` (-1) has fired and z has not since risen to
``exit_zscore_threshold`` (+1) or above; flatten and stay FLAT once
``z >= exit_zscore_threshold`` (+1), until z falls back to -1 or below.
Strictly inside the band, the prior state carries forward unchanged. An
UNDEFINED z reading (before the rolling window is full) at a decision bar
resolves to FLAT -- matching this batch's established "undefined = flat"
convention for a two-state regime rule with no third "no signal" state
(e.g. ``betting_against_beta_low_vol.hold_flags``,
``preferred_stock_yield_spread_reversion.standing_long``).

Decision bars = MONTH-END only (matching
``vol_managed_intl_country_etf_pooled.py``'s own convention exactly -- see
"Design choices" item 2 above): a bar is a decision bar iff the NEXT bar of
the (holdout-truncated) frame falls in a different calendar month; the
final bar of a truncated frame is never a decision bar (nothing about the
following session is knowable from the frame alone, and a decision there
could not execute inside the bucket in any case). The LONG/FLAT state
changes only at a decision bar and is carried forward unchanged between
them; the signal re-raises that standing state on EVERY bar (as e.g.
``preferred_stock_yield_spread_reversion.py`` does). The engine ignores a
repeated ``entry`` while already in position and a repeated ``exit`` while
already flat, so each real transition still fires exactly once, and a
bucket that opens mid-regime picks up the carried-in state at its first
fill opportunity instead of waiting for the next decision bar (see the
comment in ``_precompute``).

Engine mapping (mining.backtest_engine): on every bar the signal is
``SignalResult(entry=is_long, exit=not is_long)`` for the standing state;
the engine fills an entry at the next session's open and an exit at that
bar's own close.

No look-ahead: every series (trailing_return, the rolling z-score, the
month-end decision flags, the hysteresis state) is precomputed ONCE per
ticker in the factory from that ticker's OWN full-history frame truncated
to <= the bucket's last date (``_clean_close``), using only rolling
operations whose value at t reads closes dated <= t, then reindexed onto
the bucket ``index`` and read by position ``i`` inside ``signal_fn`` -- two
array lookups per call, no per-bar recompute. The hysteresis state is
walked forward from the START of the truncated history (not from the
bucket's own start), so a restricted holdout bucket beginning mid-history
still resolves the TRUE prior state carried in from before the bucket
began -- the same correctness property ``betting_against_beta_low_vol.py``
documents for its ``rebalance_dates`` holds here by construction, because
the factory always computes over the full truncated calendar and only
reindexes onto the bucket at the very end.

Pooling (per this pipeline's standing rule --
see e.g. ``vol_managed_intl_country_etf_pooled.py``'s own docstring): run
each of the three pool members (SPY, QQQ, IWM) INDEPENDENTLY against its
OWN close series -- no cross-ticker signal, no external SIGNAL ticker
(SIGNAL = [] -- self-referential by construction). The driver calls
``make_signal_fn`` once per ticker; the three tickers' trades are pooled
into ONE trial, matching this pipeline's standing rule that a multi-
instance backtest is pooled into one trial per family, never one family
per ticker.

Expected trade frequency (structural, from the rule itself)
-----------------------------------------------------------
* The first trailing_return is defined 756 sessions in, and the first z a
  further 3780 sessions later: a 4536-trading-day (~18-year) warm-up, so a
  newer fund's z-score is only defined late in its history -- expected,
  not a bug.
* Checking only at month-end (design choice above) means a z-score that
  dips below the -1 entry threshold intra-month but recovers above -1 by
  the next MONTH-END decision check never triggers an entry: a
  monthly-cadence hysteresis rule can miss an intra-month extreme that a
  daily-checked rule would catch. A pool member can therefore produce no
  trades at all; that is a property of the rule, not a bug.
EXPECTED_MIN_TRADES is accordingly a small pooled floor -- a genuinely
rare multi-year regime signal, matching the mechanism's own "roughly
3-to-5-year horizon" framing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "long_horizon_index_mean_reversion"
SHAPE = "pooled_single_asset"
TRADED = ["SPY", "QQQ", "IWM"]
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK and any sibling module's own identical disclosure (e.g.
# betting_against_beta_low_vol.py). Here it happens to also be
# TRADED[0] and this family's own most natural reference index, which is
# incidental, not a contract violation.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field
# for family_id "long_horizon_index_mean_reversion" (verified by this
# module's own test via json.dumps(sort_keys=True) byte-identity against
# that file directly -- same convention as betting_against_beta_low_vol.py
# and treasury_duration_volatility_timing.py).
PARAMETERS = {
    "return_window_months": 36,
    "zscore_lookback_years": 15,
    "entry_zscore_threshold": -1,
    "exit_zscore_threshold": 1,
    "rebalance": "monthly",
}
# A small pooled floor: the ~18-year warm-up and month-end-only checks
# leave only a handful of entries across the pool (see the module
# docstring's "Expected trade frequency" section).
EXPECTED_MIN_TRADES = 2

_TRADING_DAYS_PER_MONTH = 21
_TRADING_DAYS_PER_YEAR = 252
_RETURN_WINDOW_DAYS = int(PARAMETERS["return_window_months"]) * _TRADING_DAYS_PER_MONTH
_ZSCORE_LOOKBACK_DAYS = int(PARAMETERS["zscore_lookback_years"]) * _TRADING_DAYS_PER_YEAR
_ENTRY_Z = float(PARAMETERS["entry_zscore_threshold"])
_EXIT_Z = float(PARAMETERS["exit_zscore_threshold"])


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _clean_close(frame: pd.DataFrame, last_bar: pd.Timestamp) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series truncated to
    <= ``last_bar`` -- the holdout-blind step, done once per ticker (matches
    betting_against_beta_low_vol.py's own ``_clean_close`` convention)."""
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                       index=_naive_index(frame.index))
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[close.index <= last_bar]


def trailing_return(close: pd.Series, *,
                    window: int = _RETURN_WINDOW_DAYS) -> pd.Series:
    """``Close[t] / Close[t - window] - 1`` for every t in ``close.index``
    (a plain trading-day-lagged cumulative return); NaN until ``window``
    prior closes exist. Equivalent to ``close.pct_change(periods=window)``,
    written out via ``pct_change`` directly so the formula is legible
    without cross-referencing pandas' own definition."""
    if window <= 0:
        raise ValueError(f"{FAMILY_ID}: window must be positive, got {window}")
    return close.pct_change(periods=window).rename("trailing_return")


def rolling_zscore(series: pd.Series, *,
                   window: int = _ZSCORE_LOOKBACK_DAYS) -> pd.Series:
    """Rolling ``window``-bar z-score of ``series`` (full window required,
    ddof=1) -- see the module docstring's "Rolling vs expanding" section for
    why this is rolling, not expanding. A rolling std below a relative
    float-noise floor (matching ``preferred_stock_yield_spread_reversion.
    trailing_yield_zscore``'s own guard) is treated as undefined (NaN),
    never a spurious near-infinite z."""
    if window < 2:
        raise ValueError(f"{FAMILY_ID}: window must be >= 2, got {window}")
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1)
    tolerance = 1e-9 * mean.abs() + 1e-15
    z = (series - mean) / std.where(std > tolerance)
    return z.rename("zscore")


def _month_end_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """True where the NEXT bar of ``index`` is in a different calendar
    month; the final bar is always False. Byte-for-byte the same rule as
    ``vol_managed_intl_country_etf_pooled._month_end_flags`` (see this
    module's "Decision bars" docstring section and "Design choices" item
    2)."""
    months = index.year.to_numpy() * 12 + index.month.to_numpy()
    flags = np.zeros(len(index), dtype=bool)
    if len(index) > 1:
        flags[:-1] = months[1:] != months[:-1]
    return flags


def standing_long_state(zscore: pd.Series, decision_flags: np.ndarray, *,
                        entry_z: float = _ENTRY_Z,
                        exit_z: float = _EXIT_Z) -> np.ndarray:
    """Per-bar hysteresis state (True = LONG), aligned to ``zscore.index``.

    Checked ONLY where ``decision_flags`` is True (month-end bars) and
    carried forward unchanged on every other bar. At a decision bar: enter
    (become/stay LONG) at ``z <= entry_z``; exit (become/stay FLAT) at
    ``z >= exit_z``; strictly inside the band, hold whatever the state
    already is. An undefined (NaN) z at a decision bar resolves to FLAT --
    this family's own "undefined = flat" convention (see module docstring).
    Sequential by construction (a plain forward loop over decision bars),
    so the state at bar i depends only on decision bars <= i -- no
    look-ahead."""
    values = pd.to_numeric(zscore, errors="coerce").to_numpy(dtype=float)
    n = len(values)
    if len(decision_flags) != n:
        raise ValueError(f"{FAMILY_ID}: zscore and decision_flags length "
                         f"mismatch ({n} vs {len(decision_flags)})")
    state = np.zeros(n, dtype=bool)
    is_long = False
    for i in range(n):
        if decision_flags[i]:
            z = values[i]
            if np.isnan(z):
                is_long = False
            elif z <= entry_z:
                is_long = True
            elif z >= exit_z:
                is_long = False
            # else: strictly inside the band -- hold the prior state.
        state[i] = is_long
    return state


def _precompute(frame: pd.DataFrame, index: pd.DatetimeIndex) -> tuple:
    """Return ``(entry_flags, exit_flags)`` bool arrays aligned to
    ``index`` (the bucket the engine will run over) for ONE ticker's own
    frame. Everything is computed over the ticker's full history truncated
    to <= the bucket's last bar (holdout-blind), so the hysteresis state
    carried INTO the bucket is always the true prior state, even when the
    bucket itself starts mid-history."""
    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")

    close = _clean_close(frame, bucket[-1])
    trailing = trailing_return(close)
    z = rolling_zscore(trailing)
    decision = _month_end_flags(close.index)
    is_long = standing_long_state(z, decision)

    # entry/exit reflect the CURRENT state on every bar (long_series is
    # already a step function, constant between decision-bar transitions --
    # see standing_long_state's own "carried forward unchanged on every
    # other bar" contract), not just on decision bars themselves. This
    # matters at a bucket boundary: a restricted bucket (e.g. a holdout
    # window) can start mid-regime, already long from a transition that
    # happened BEFORE the bucket's own first bar -- gating on the
    # decision flags here would silently wait for the next decision bar to
    # merely re-confirm a state that was already true, missing the carried-
    # in position for every bar in between (a bucket beginning mid-cycle
    # must resolve its TRUE prior state, not just react to transitions
    # visible inside the bucket -- the same property
    # betting_against_beta_low_vol.py's rebalance_dates() provides). The
    # engine's own contract ("entry only read while
    # flat, exit only read while long") already makes a continuously-True
    # entry/exit series safe -- it fires the underlying open/close exactly
    # once per real transition, same as every other module in this
    # directory that returns a per-bar state directly (e.g.
    # betting_against_beta_low_vol.hold_flags).
    long_series = pd.Series(is_long, index=close.index)
    entry = long_series.reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    exit_ = (~long_series).reindex(bucket, fill_value=False).to_numpy(dtype=bool)
    return entry, exit_


def make_signal_fn(ticker: str, index: pd.DatetimeIndex, frames: dict):
    """SignalFn factory for one pool member (the driver calls this once per
    ticker in TRADED, per the ``pooled_single_asset`` contract -- see
    ``vol_managed_intl_country_etf_pooled.make_signal_fn``'s own identical
    shape). ``frames[ticker]`` must be that ticker's full-history OHLC
    frame; raises ``KeyError`` when absent (fail closed, matching
    ``vol_managed_intl_country_etf_pooled.py``'s own convention) -- the
    signal source is the ticker's own close, never a substitute. ``index``
    is the bucket the engine will run over."""
    if ticker not in TRADED:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects one of "
                         f"{TRADED}, got {ticker!r}")
    frame = frames[ticker]
    entry_flags, exit_flags = _precompute(frame, index)
    n = len(entry_flags)

    def signal_fn(frame_slice: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n:
            return SignalResult()
        return SignalResult(entry=bool(entry_flags[i]), exit=bool(exit_flags[i]))

    return signal_fn
