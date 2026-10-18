"""Batch5 candidate: vix_spike_reversal_equity_buy (single_long).

Implements the catalogued mechanism from ``data/mining/catalog.json``
under the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instrument is re-tuned here -- this is a
genuinely conditional, bespoke signal
(same complexity tier as ``betting_against_beta_low_vol.py``), not a
simple factory reuse.

Source (verbatim from the pre-registered catalog entry, ``data/mining/
catalog.json``, ``family_id == "vix_spike_reversal_equity_buy"``,
``added_date`` 2026-08-29): Whaley, R.E. (2000), "The Investor Fear
Gauge," Journal of Portfolio Management, 26(3), 12-17; Giot, P. (2005),
"Relationships Between Implied Volatility Indexes and Stock Index
Returns," Journal of Portfolio Management, 31(3), 92-100.

Catalog mechanism (verbatim): "Track the VIX index's own 10-day moving
average. When VIX closes at least 50% above that moving average (a
volatility spike, historically most common when VIX also crosses above
roughly 30), buy SPY the next session and hold for a fixed 10-trading-day
window before exiting, regardless of what VIX does in between. No new
entry is taken on additional spikes while a position is already open.
Implementation note: signal_fn must fetch ^VIX via data_pipeline.
fetch_ohlc and align by date to the SPY frame." Catalog ``tickers`` are
``["SPY", "^VIX"]``.

Rule (this module's own reading of the catalog prose into code)
-----------------------------------------------------------------------
Each VIX trading day t: ``vix_ma10[t]`` = VIX's own trailing 10-day mean
of its Close (``PARAMETERS["vix_ma_window_days"] == 10``,
``min_periods=10`` -- undefined during warm-up). ``spike[t]`` = ``VIX.
Close[t] >= 1.5 * vix_ma10[t]`` (``PARAMETERS["spike_threshold_pct_over_
ma"] == 0.5`` read as "at least 50% above," i.e. a multiplicative
threshold of ``1.0 + 0.5 == 1.5``, boundary INCLUSIVE via ``>=`` --
matching the catalog's own "at least" wording).

Aligned onto SPY's own trading calendar (see Interpretation choice 1
below), a state machine tracks entry/exit array positions exactly the
way ``stock_split_post_split_drift_pooled.py``'s own ``entry_exit_
positions`` tracks a fixed-length hold from a triggering event -- ported
here rather than the monthly-rebalance-then-hold-constant pattern of
``betting_against_beta_low_vol.py``, because this family's hold length is
a FIXED BAR COUNT from a discrete trigger, not a regime read re-evaluated
at the next periodic checkpoint:

* While flat (per this module's OWN forward-scan bookkeeping, which
  mirrors the engine's real flat/long state machine exactly -- see
  Interpretation choice 4), a spike at bucket position ``k`` reports
  ``entry[k] = True``. Per the engine's own next-open execution
  convention (``mining.backtest_engine.run_backtest``'s docstring: "acts
  on an entry signal at the NEXT bar's open"), this is exactly "buy SPY
  the next session" -- no extra lag is added here beyond what the engine
  already provides.
* The fill bar is therefore bucket position ``k + 1``. The position is
  held for EXACTLY ``PARAMETERS["holding_period_days"] == 10`` bars as
  measured by the engine's own ``held`` field (``i - entry_idx`` in
  ``run_backtest``), so ``exit[fill + 10] = True`` -- closing at that
  bar's close. See Interpretation choice 3 for why the exit position is
  ``fill + 10``, not ``k + 10``.
* ``PARAMETERS["reentry_while_in_position"] == False``: from the fill bar
  through the exit bar INCLUSIVE, any further spike is ignored -- ported
  verbatim from ``stock_split_post_split_drift_pooled.py``'s own
  ``busy_through`` convention (see Interpretation choice 2).

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- ``vix_spike_flags`` and ``entry_exit_positions`` are
  both precomputed ONCE in the factory from VIX/SPY history truncated to
  <= the bucket's last bar (``_clean_close``); ``entry_exit_positions``
  is a pure forward scan (each position's outcome depends only on EARLIER
  positions' spike/state), never a backward or centered computation. The
  returned ``signal_fn`` only ever does an O(1) array lookup by position.
* PERFORMANCE -- the rolling MA, the spike comparison, and the forward
  scan are each computed ONCE in the factory; ``signal_fn`` does two
  boolean array lookups plus one scalar date check per call.
* HOLDOUT-BLIND -- both VIX's and SPY's close series are truncated to <=
  the bucket's own last bar before anything is computed
  (``_clean_close``), so a holdout bucket's signals cannot depend on data
  after the bucket; history before the bucket is kept as warm-up (the
  10-day MA window needs it).
* DIRECTION -- always long SPY or flat cash; never short, never trades
  ``^VIX`` itself.

Interpretation choices beyond the catalog's own prose (left to this
implementer, each disclosed rather than silently picked):

1. "Align by date to the SPY frame" (the catalog's own implementation
   note): VIX's trailing 10-day MA and spike flag are computed on VIX's
   OWN trading calendar first (never restricted to SPY's calendar before
   the rolling window is computed -- a rolling mean must walk VIX's own
   consecutive trading days, not gapped ones), and the resulting boolean
   ``spike`` series is then looked up by date on SPY's bucket via
   ``reindex`` -- exactly ``vix_term_structure_carry.py``'s own
   "aligned via reindex, not intersected first" convention for a
   signal-ticker series. Calendar fact (full-history OHLC via
   ``data_pipeline.fetch_ohlc``): SPY's own
   trading calendar (1993-01-29 .. today) is a STRICT SUBSET of ^VIX's
   (1990-01-02 .. today) -- zero SPY dates are missing from ^VIX's
   calendar (782 ^VIX-only dates all predate SPY's 1993 inception) -- so
   this reindex is a clean lookup with no gap-induced ``NaN`` anywhere a
   bucket could actually start. A gap is handled anyway (silent
   ``spike=False``, the same "undefined is not a signal" convention used
   throughout this batch) for robustness against a corrupted or partial
   frame, not because one was observed.
2. Re-entry blocking (Interpretation of "no new entry ... while a
   position is already open"): ``entry_exit_positions`` ports
   ``stock_split_post_split_drift_pooled.py``'s own ``busy_through``
   mechanism verbatim -- a spike at position ``k`` is skipped if
   ``k <= busy_through``, where ``busy_through`` is set to the exit
   position once a hold is scheduled. This blocks a spike on the EXIT bar
   itself from starting a fresh hold that same bar (the position is still
   "open" through that bar's close), not only spikes strictly between fill
   and exit. This narrowing only matters in the rare case of a spike
   landing exactly on a prior hold's exit bar; it is the interpretation
   the established precedent uses.
3. Fill-then-hold exit-position arithmetic (why ``exit = fill + hold``,
   not ``spike_position + hold``): ``run_backtest``'s own ``held`` field
   is computed as ``i - entry_idx``, where ``entry_idx`` is the FILL
   position, not the signal position. Reading "hold for exactly 10
   trading bars" as "the engine's own ``held`` field equals 10 exactly"
   (the same reading ``stock_split_post_split_drift_pooled.py`` gives its
   own ``time_stop_bars``) requires ``exit_position = fill + hold``. This
   module's synthetic tests pin it: a single spike exits exactly ``hold``
   bars after its fill, and end to end through ``run_backtest`` every trade
   closes with ``held == 10`` and ``reason == "signal_exit"`` -- not off by
   one in either direction.
4. A spike whose FILL bar would land at or beyond the bucket's second-to-
   last bar (``fill >= n - 1``) reports NO entry -- the same rule as
   ``stock_split_post_split_drift_pooled.py``'s own "Data hygiene" choice
   (a fill this close to the bucket's end cannot produce a real held
   position; the engine would force-close it the same bar it opens, held
   ``0``, which is not this family's rule). The rest of that bucket's
   calendar is then closed out (nothing later could fill either).
5. Duplicate dates in either close series keep the LAST occurrence before
   sorting -- the same de-duplication convention used throughout this
   batch (e.g. ``betting_against_beta_low_vol.py``).
6. VIX values as fetched via ``data_pipeline.fetch_ohlc("^VIX")`` are
   already in raw index "vol points" -- NOT decimal fractions the way ``data_pipeline.
   fetch_iv_proxy`` divides by 100 for direct Black-Scholes sigma use.
   ``fetch_ohlc`` is used here (per the catalog's own implementation
   note), so the spike comparison operates directly on raw vol points on
   both sides of the ``>=`` (the ratio is scale-invariant to a constant
   multiplier anyway, so this only matters for readability, not
   correctness).
7. Data-freshness disclosure: during market hours SPY's still-forming
   current-session bar can arrive from ``fetch_ohlc`` as an all-``NaN``
   row, while ``^VIX``'s same-date bar can carry a real, non-``NaN``
   close -- a live intraday index-
   vs-security data-timing artifact of the two instruments' different
   quote mechanics, not a defect in this module. This cannot affect a
   real driver run: the driver always hands ``make_signal_fn`` a bucket of
   already-closed SPY sessions, and VIX is only ever looked up BY THOSE
   DATES via ``reindex``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "vix_spike_reversal_equity_buy"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL = ["^VIX"]
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see any other batch5 module's own comment
# on this (e.g. presidential_cycle_year3_seasonal.py). Here it happens to
# equal TRADED[0] too, which is incidental (SPY is both the traded
# instrument and the hard ABI-1 benchmark), not a contract violation.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's
# "vix_spike_reversal_equity_buy" entry (verified in this module's own
# test via json.dumps(sort_keys=True) byte-identity against that file
# directly -- this family has no separate curated candidate record,
# same situation/convention as betting_against_beta_low_vol.py).
PARAMETERS = {
    "vix_ma_window_days": 10,
    "spike_threshold_pct_over_ma": 0.5,
    "holding_period_days": 10,
    "reentry_while_in_position": False,
}
if PARAMETERS["reentry_while_in_position"] is not False:
    # This module only implements the False (no-reentry-while-held)
    # semantics described in the catalog and in the module docstring's
    # "Rule" section -- a future catalog edit flipping this parameter
    # would silently change the intended mechanism out from under an
    # unmodified entry_exit_positions(), so fail loudly instead.
    raise NotImplementedError(
        f"{FAMILY_ID}: only reentry_while_in_position=False is implemented")

_MA_WINDOW_DAYS = int(PARAMETERS["vix_ma_window_days"])
_SPIKE_THRESHOLD_MULT = 1.0 + float(PARAMETERS["spike_threshold_pct_over_ma"])
_HOLD_DAYS = int(PARAMETERS["holding_period_days"])

# Sanity floor on the full-history trade count (the driver refuses a run
# below it). Only a subset of spikes open a position: VIX spikes cluster in
# time around a single stress event, and spikes that fire while a hold is
# already open are ignored by the reentry-blocking rule (Interpretation
# choice 2). Kept conservative (the same convention as
# presidential_cycle_year3_seasonal.py and
# single_day_extreme_decline_reversal_long.py), since this is a rare-event
# family.
EXPECTED_MIN_TRADES = 10


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
    """Tz-naive, sorted, de-duplicated (keep last) Close series truncated to
    <= ``last_bar`` -- the holdout-blind step, done once per ticker."""
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                       index=_naive_index(frame.index))
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[close.index <= last_bar]


def vix_spike_flags(vix_close: pd.Series, ma_window: int,
                     threshold_mult: float) -> pd.Series:
    """Boolean Series, aligned to ``vix_close``'s own index: ``spike[t]``
    = ``vix_close[t] >= threshold_mult * trailing_ma[t]``, where
    ``trailing_ma`` is ``vix_close``'s own ``ma_window``-day rolling mean
    (``min_periods=ma_window`` -- ``NaN``/``False`` during warm-up, never
    a partial-window estimate). The comparison uses ONLY ``vix_close``'s
    trailing values through and including position t (pandas
    ``.rolling(window, min_periods=window)``), so it structurally cannot
    read a value past its own current row -- no look-ahead. Boundary is
    INCLUSIVE (``>=``), matching the catalog's own "at least 50% above"
    wording."""
    if ma_window <= 0:
        raise ValueError(f"{FAMILY_ID}: ma_window must be > 0, got {ma_window}")
    ma = vix_close.rolling(ma_window, min_periods=ma_window).mean()
    vix_arr = vix_close.to_numpy(dtype=float)
    ma_arr = ma.to_numpy(dtype=float)
    valid = np.isfinite(vix_arr) & np.isfinite(ma_arr)
    spike = np.zeros(len(vix_close), dtype=bool)
    spike[valid] = vix_arr[valid] >= threshold_mult * ma_arr[valid]
    return pd.Series(spike, index=vix_close.index)


def entry_exit_positions(spike: np.ndarray, hold_days: int) -> tuple:
    """Forward scan over bucket positions ``0 .. len(spike) - 1``: a
    ``True`` at position ``k`` (and no hold currently open, per this
    function's own ``busy_through`` bookkeeping) schedules an entry at
    ``k``, a fill at ``k + 1``, and an exit at ``k + 1 + hold_days`` --
    see the module docstring's "Rule" section and Interpretation choices
    2-4 for the exact semantics and their justification. Ported
    from ``stock_split_post_split_drift_pooled.py``'s own
    ``entry_exit_positions`` (there keyed on real split-event dates,
    here keyed on this family's own precomputed ``spike`` boolean array)
    -- same complexity tier as that module's function and as
    ``betting_against_beta_low_vol.py``'s own ``rebalance_dates``/
    ``hold_flags``.

    A hold that would still be open past the array's end (``fill +
    hold_days >= len(spike)``) is entered but reports NO exit position --
    the engine's own ``end_of_data`` path closes it, matching every other
    batch5 module's convention for a bucket that ends mid-hold.

    Returns ``(entry_positions, exit_positions)`` as sorted int arrays
    (each strictly increasing, since a new hold can only start after the
    previous one's ``busy_through`` boundary)."""
    if hold_days <= 0:
        raise ValueError(f"{FAMILY_ID}: hold_days must be > 0, got {hold_days}")
    spike = np.asarray(spike, dtype=bool)
    n = len(spike)
    entries: list = []
    exits: list = []
    busy_through = -1
    for k in range(n):
        if k <= busy_through:
            continue
        if not spike[k]:
            continue
        fill = k + 1
        if fill >= n - 1:
            # The fill bar is the bucket's LAST bar, or there is no fill
            # bar at all -- neither can produce a real held position (see
            # Interpretation choice 4). Nothing later in this bucket
            # could fill either, so close out the rest of the calendar.
            busy_through = n - 1
            continue
        entries.append(k)
        exit_pos = fill + hold_days
        if exit_pos < n:
            exits.append(exit_pos)
            busy_through = exit_pos
        else:
            busy_through = n - 1  # held to end of data, no exit reported
    return np.asarray(entries, dtype=int), np.asarray(exits, dtype=int)


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over (SPY's own calendar, per the generic single_long
    dispatch); ``frames`` holds full-history OHLC for TRADED + SIGNAL."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    spy_frame = _require_frame(frames, ticker)
    vix_frame = _require_frame(frames, SIGNAL[0])

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    # spy_frame's presence was already confirmed by _require_frame(frames,
    # ticker) above -- SPY's own frame is used only to satisfy the driver's
    # fail-closed convention; SPY's OWN prices never enter the signal (VIX
    # alone drives entry/exit).
    del spy_frame

    vix_close = _clean_close(vix_frame, last_bar)
    spike_full = vix_spike_flags(vix_close, _MA_WINDOW_DAYS, _SPIKE_THRESHOLD_MULT)
    # Interpretation choice 1: the spike flag is computed on VIX's OWN
    # calendar (immediately above), THEN looked up by date on the bucket
    # -- never the reverse, since the rolling MA must walk VIX's own
    # consecutive trading days. A bucket date absent from VIX's calendar
    # (not expected for two real, currently-trading NYSE/CBOE
    # instruments -- see Interpretation choice 1's calendar fact) is
    # silently no-spike via reindex-to-False, never a KeyError.
    spike_on_bucket = spike_full.reindex(bucket).fillna(False).to_numpy(dtype=bool)

    entries, exits = entry_exit_positions(spike_on_bucket, _HOLD_DAYS)
    n_bucket = len(bucket)
    entry_arr = np.zeros(n_bucket, dtype=bool)
    entry_arr[entries] = True
    exit_arr = np.zeros(n_bucket, dtype=bool)
    exit_arr[exits] = True
    bucket_values = bucket.asi8  # int64 ns for an O(1) scalar date check

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
