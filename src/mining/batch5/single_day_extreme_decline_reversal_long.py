"""Batch5 candidate: single_day_extreme_decline_reversal_long (single_long).

Single-SESSION extreme-decline reversal on SPY: a Cox & Peterson (1994) /
Bremer & Sweeney (1991) style short-term reversal mechanism, deliberately
scoped to the smallest, most literal reading of "one bad day, one reversal
attempt" so it does not restate any of the three catalog families it is
closest to (see DISTINCTNESS NOTE below, restated in the catalog entry this
module supports).

Mechanism
---------

    Compute daily log returns for SPY: r_t = ln(Close_t / Close_{t-1}).
    Compute a trailing 20-session realized volatility as the sample std
    (ddof=1) of the 20 daily log returns STRICTLY BEFORE session t (r_{t-20}
    .. r_{t-1} -- never r_t itself, so the divisor cannot be inflated or
    deflated by the very session being measured).
    z_t = r_t / trailing_vol_t.
    Entry trigger: z_t <= -2.5 on session t -> enter long, filled at the
    engine's own next-bar-open execution (session t+1's open -- see Engine
    mapping below).
    Exit: a fixed time stop of exactly 3 trading sessions held from the
    fill, then exit at that session's close. No other exit condition.

DISTINCTNESS NOTE (verbatim mechanism text from the three siblings this
family is scoped against, data/mining/catalog.json):

* ``large_decline_rebound`` -- "compute the trailing 10-trading-day
  cumulative close-to-close return; when it drops below a large-negative
  threshold (-10%), enter long at the next close ... Hold for a short fixed
  window (5 trading days)". A 10-DAY CUMULATIVE trigger and a 5-day hold --
  this family's trigger is one session's own return, not a 10-day sum, and
  its hold is 3 sessions, not 5.
* ``short_term_reversal_weekly`` -- "compute the trailing 5-trading-day
  (1-week) cumulative close-to-close return; when it falls below a
  large-negative threshold (a z-score below -1.5 versus its own trailing
  1-year distribution of 5-day returns) ... Hold for a fixed 5-trading-day
  period". A 5-DAY CUMULATIVE return, z-scored against a trailing 1-YEAR
  distribution of 5-day returns and a -1.5 threshold -- this family
  z-scores a SINGLE session's own return against a trailing 20-session
  volatility (not a distribution of multi-day returns) at a much stricter
  -2.5 threshold, and holds 3 sessions, not 5.
* ``intraday_capitulation_reversal_bar`` -- "track the intraday percentage
  decline from the prior session's close using 5-minute bars. When that
  decline first reaches at least 5% by 14:00 ET ... and the very next
  5-minute bar closes in the top third of its own range on volume at least
  2x ... enter long at that reversal bar's close, same day. Hold for up to
  20 trading days". An INTRADAY decline-plus-reversal-bar trigger measured
  on 5-minute bars against a 14:00 ET cutoff, entered same-day, held up to
  20 days (swing-shaped) -- this family reads only daily OHLC (never an
  intraday bar), triggers on the CLOSE-to-close return only after the full
  session has printed, enters the following session (never same-day), and
  holds a short, fixed 3 sessions (not "up to" a longer horizon).

None of the three can fire on a single ordinary trading day the way this
family's trigger does: the 10-day and 5-day siblings require a multi-day
cumulative decline (one very bad day alone cannot trip either threshold in
this family's synthetic tests, see test_moderate_decline_does_not_trigger),
and the intraday sibling requires minute-bar data this family never reads.

Engine mapping (mining.backtest_engine.run_backtest)
-----------------------------------------------------
The shared engine ALWAYS fills an entry at the NEXT bar's OPEN --
structural, not a per-family knob (backtest_engine.py's own module
docstring: "The engine acts on an entry signal at the NEXT bar's open").
Every other single_long family in this catalog that specifies its own
entry timing defers to this same constraint (see e.g.
pre_holiday_small_cap_effect.py's "Interpretation choices" item 2, an
identical engine-imposed constraint). Entry therefore fills at session
t+1's OPEN, as the catalog mechanism states -- earlier than a fill at
t+1's close would be, so less of the reversal can have happened before the
fill; neither the trigger nor the hold length is redefined.

The ABI-1 driver (mining.abi1_driver.build_single_asset_curve) calls
``run_backtest`` WITHOUT its ``time_stop`` argument (no ``time_stop=``
appears anywhere in abi1_driver.py), so
the fixed 3-session hold is expressed by THIS module's own precomputed
exit schedule, exactly matching stock_split_post_split_drift_pooled.py's
own documented reason for doing the same thing for its 252-bar hold. Every
other bar returns entry=False, exit=False; the engine's own state machine
already refuses to consider a new entry while a position is open, but this
module's own entry/exit precomputation still tracks one-position-at-a-time
explicitly (mirroring stock_split's ``entry_exit_positions``) so that the
EXIT it schedules always corresponds to a trigger the engine actually
acted on, never a later trigger the engine silently ignored because a
hold from an earlier trigger was still open.

Interpretation choices
-----------------------
* A trigger whose fill bar (position + 1) does not exist inside the bucket
  (the trigger is on the bucket's very last bar) reports NO entry -- there
  is no next-bar open to fill at. This is the same "no fill bar" half of
  stock_split_post_split_drift_pooled's "Data hygiene item 1"; unlike that
  family's 252-bar hold, this family's short 3-session hold makes the
  "fill bar is the bucket's last bar" (held=0) sub-case vanishingly rare in
  practice (it needs a trigger on the SECOND-to-last bar specifically), so
  it is handled identically rather than carved out as a separate rule --
  the fill-existence check subsumes both.
* A hold still open at the bucket's last bar is left for the engine's own
  ``end_of_data`` close -- no exit position is scheduled past the bucket.
* NO LOOK-AHEAD -- trailing_vol_t uses only log returns strictly before
  session t (a one-session shift before the rolling window), and the
  factory truncates the supplied frame to <= the bucket's own last bar
  before computing anything (test_holdout_blind_frames_extending_past_
  index_are_ignored), matching every other single_long family's own
  holdout-blind convention (see oil_price_change_equity_underreaction.py).
* Warm-up: the trailing-vol window is computed from FULL history before the
  bucket (never re-zeroed at the bucket's own start), so a trigger that lands
  on the bucket's very FIRST bar still fires correctly
  (test_warm_up_history_before_bucket_start_is_used) -- this module never
  needs to resolve a HOLD that was already open before the bucket began,
  because it never receives one: run_single_asset_windows
  (scripts/run_batch3_mining.py, the driver for this SHAPE) splits
  in_sample/holdout with a hard, unpadded cutoff and runs each bucket as its
  own independent run_backtest call, and the engine's own end_of_data path
  (backtest_engine.py) force-closes any position still open at a bucket's
  own last bar -- so the NEXT bucket's signal_fn always starts flat, the
  same "buckets are independent" contract
  stock_split_post_split_drift_pooled.py documents verbatim ("a hold is
  never carried across the bucket boundary"). No cross-bucket carry happens
  or is needed for this SHAPE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "single_day_extreme_decline_reversal_long"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL: list = []  # self-contained: only SPY's own OHLC is ever read
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Parameter names follow this codebase's own naming convention, read from
# the two closest siblings (large_decline_rebound /
# short_term_reversal_weekly, both in
# data/mining/catalog.json): "entry_zscore_threshold" and
# "holding_period_days" are their exact field names; "trailing_vol_window_
# days" distinguishes this family's SINGLE-session vol-normalization window
# from those two families' own "lookback_days" (a multi-day CUMULATIVE
# return window, a different quantity this family does not compute).
PARAMETERS = {
    "trailing_vol_window_days": 20,
    "entry_zscore_threshold": -2.5,
    "holding_period_days": 3,
    "direction": "long",
}
VOL_WINDOW_DAYS = int(PARAMETERS["trailing_vol_window_days"])
Z_THRESHOLD = float(PARAMETERS["entry_zscore_threshold"])
HOLD_DAYS = int(PARAMETERS["holding_period_days"])
# Configured conservative floor on the full-history trade count (triggers
# whose prior 3-session hold is still open are skipped, so the count is below
# the raw trigger count), the same conservative-floor convention as
# fed_net_liquidity_regime.py and margin_debt_growth_regime.py.
EXPECTED_MIN_TRADES = 100


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


def trigger_zscore(close: pd.Series, *,
                   vol_window: int = VOL_WINDOW_DAYS) -> pd.Series:
    """z_t = r_t / trailing_vol_t over ``close``'s own index.

    r_t is the single-session log return ending at t
    (``np.log(close).diff()``). trailing_vol_t is the sample std (ddof=1) of
    the ``vol_window`` log returns STRICTLY BEFORE t -- a one-session
    ``shift(1)`` ahead of the rolling window, so r_t itself is never part of
    its own normalizer. NaN wherever r_t or the trailing window is
    undefined (the first ``vol_window + 1`` sessions of ``close``, plus any
    non-finite input)."""
    values = pd.to_numeric(close, errors="coerce").astype(float)
    logret = np.log(values).diff()
    trailing_vol = logret.shift(1).rolling(
        vol_window, min_periods=vol_window).std(ddof=1)
    return logret / trailing_vol


def trigger_positions(z: pd.Series, *,
                      threshold: float = Z_THRESHOLD) -> np.ndarray:
    """Integer positions (into ``z``'s own index) where ``z_t <= threshold``,
    ascending, ignoring NaN/inf."""
    arr = z.to_numpy(dtype=float)
    mask = np.isfinite(arr) & (arr <= threshold)
    return np.flatnonzero(mask)


def entry_exit_positions(trigger_pos, n: int, *,
                         hold_days: int = HOLD_DAYS):
    """One-position-at-a-time simulation over a bucket of length ``n``:
    a trigger at position k fills at k+1 and holds through k+1+hold_days
    inclusive (``held == hold_days`` exactly, matching the engine's own
    time_stop semantics); a trigger while a hold is open (including on its
    own exit bar) is skipped, mirroring
    stock_split_post_split_drift_pooled.entry_exit_positions. A trigger
    with no fill bar inside the bucket (k is the bucket's last bar) reports
    no entry and closes out the rest of the bucket's calendar (nothing
    later could fill either). A hold still open at the bucket's end is left
    for the engine's own end_of_data close (no exit position emitted).
    Returns ``(entry_positions, exit_positions)`` as int arrays."""
    if isinstance(hold_days, bool) or not isinstance(hold_days, int) or hold_days <= 0:
        raise ValueError(f"{FAMILY_ID}: hold_days must be a positive int")
    entries, exits = [], []
    busy_through = -1
    for k in trigger_pos:
        k = int(k)
        if k <= busy_through:
            continue
        fill = k + 1
        if fill >= n:
            busy_through = n - 1
            continue
        entries.append(k)
        exit_pos = fill + hold_days
        if exit_pos < n:
            exits.append(exit_pos)
            busy_through = exit_pos
        else:
            busy_through = n - 1  # held open to end of data
    return np.asarray(entries, dtype=int), np.asarray(exits, dtype=int)


def make_signal_fn(ticker: str, index, frames: dict):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames[ticker]`` is the (possibly full-history) OHLC
    frame for SPY. No other frame is read (SIGNAL is empty)."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    frame = _require_frame(frames, ticker)

    bucket = _normalize_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    history = frame.copy()
    history.index = _normalize_index(history.index)
    history = history[~history.index.duplicated(keep="last")].sort_index()
    history = history[history.index <= last_bar]  # holdout-blind truncation

    z_full = trigger_zscore(history["Close"])
    z_bucket = z_full.reindex(bucket)
    n = len(bucket)
    trig_pos = trigger_positions(z_bucket)
    entry_pos, exit_pos = entry_exit_positions(trig_pos, n, hold_days=HOLD_DAYS)

    entry_arr = np.zeros(n, dtype=bool)
    exit_arr = np.zeros(n, dtype=bool)
    entry_arr[entry_pos] = True
    exit_arr[exit_pos] = True

    bucket_values = bucket.asi8
    by_date = {int(v): p for p, v in enumerate(bucket_values)}

    def signal_fn(frame_slice: pd.DataFrame, i: int) -> SignalResult:
        today = pd.Timestamp(frame_slice.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        today_ns = int(today.normalize().value)
        if 0 <= i < n and bucket_values[i] == today_ns:
            pos = i
        else:
            pos = by_date.get(today_ns)
            if pos is None:
                return SignalResult()
        return SignalResult(entry=bool(entry_arr[pos]), exit=bool(exit_arr[pos]))

    return signal_fn
