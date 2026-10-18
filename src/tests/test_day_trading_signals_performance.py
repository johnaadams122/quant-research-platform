"""Performance and value-equivalence tests for the bounded per-call work in
src/mining/day_trading_signals.py's signal_fn factories.

The intraday backtest engine hands each signal_fn a FRESH SLICE of the frame
ending at the current bar on every call (`df.iloc[:i]` or `df.iloc[:i+1]`,
in run_intraday_backtest), which GROWS by one row every call. A factory that
recomputes something (a daily ATR, a relative-volume baseline, a same-day
mask, a resampled daily-close series, a selling-climax lookback baseline)
over the ENTIRE received frame on every call, instead of a bounded recent
window, costs O(len(frame)) per call x O(len(frame)) calls = O(n^2) total.
Seven factories therefore bound that work --
`make_orb_5min_atr_stop_scalp_signal_fn`,
`make_volume_filtered_orb_stocks_in_play_pooled_signal_fn`,
`make_intraday_rsi2_reversion_daytrade_signal_fn`,
`make_overnight_gap_meanreversion_fade_signal_fn`,
`make_overnight_gap_fade_intraday_reversal_signal_fn`,
`make_intraday_half_hour_momentum_continuation_signal_fn`,
`make_selling_climax_capitulation_bounce_daytrade_signal_fn`.

Two kinds of test below, both relative comparisons rather than a single fixed
wall-clock number so they stay meaningful across machines:

(1) For the factory most sensitive to unbounded per-call cost (the RSI(2)
factory, gated on NO once-per-day check at all), a marginal-call-cost
comparison: a fixed-size batch of calls near the START of a long frame vs.
the SAME-size batch near the END. A total-runtime-vs-bar-count ratio would be
confounded by an internal lookback window's "filling up" transient (see
test_swing_signals_performance.py).

(2) The `Value-equivalence` section proves that the bound in each of the 4
factories that cap row count BEFORE a `.groupby`/`.resample` lookback
aggregation (ATR, relative-volume threshold, gap z-score inputs,
selling-climax baselines) does not change the computed value, not just its
cost -- a bound that shrinks a lookback window too far would silently change
the value. Each pairs a REFERENCE (unbounded) implementation against a FAST
(production, bounded) implementation and asserts them numerically identical
across many sampled points on a shared realistic multi-year frame. The 3
factories whose bound is "today's bars only" are correct by construction
(_TAIL_BARS=400 > one 390-minute RTH session), so they get no
value-equivalence test.
"""
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.day_trading_signals import (
    make_intraday_half_hour_momentum_continuation_signal_fn,
    make_intraday_rsi2_reversion_daytrade_signal_fn,
    make_orb_5min_atr_stop_scalp_signal_fn,
    make_overnight_gap_fade_intraday_reversal_signal_fn,
    make_overnight_gap_meanreversion_fade_signal_fn,
    make_selling_climax_capitulation_bounce_daytrade_signal_fn,
    make_volume_filtered_orb_stocks_in_play_pooled_signal_fn,
)

BARS_PER_SESSION = 390  # 09:30-15:59 ET inclusive, RTH-only


def _rth_minute_frame(n_days, start_date="2020-01-02", seed=11):
    """Synthetic RTH-shaped 1-minute bar frame spanning `n_days` business
    days (09:30-15:59 ET, BARS_PER_SESSION bars/day, weekends skipped --
    real NYSE holidays are not modeled, irrelevant for a pure timing/
    value-equivalence harness). Deterministic (seeded) slow random walk in
    Close; Open/High/Low/Volume derived from it so every column is sane and
    every bar has real Volume. Mirrors test_swing_signals_performance.py's
    own fixture builder (minus the VWAP column, which none of the 7 bounded
    factories in this file read).
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start=start_date, periods=n_days, tz="America/New_York")
    idx = []
    for day in days:
        session_start = day.replace(hour=9, minute=30)
        idx.extend(pd.date_range(session_start, periods=BARS_PER_SESSION, freq="1min"))
    n = len(idx)
    close = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    open_ = close + rng.normal(0, 0.005, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.01, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.01, n))
    volume = rng.integers(100, 1000, n).astype(float)
    index = pd.DatetimeIndex(idx, name="timestamp")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def _drive_every_bar(signal_fn, frame):
    """Replicate the real engine's per-bar dispatch at every position
    (signal_fn(df.iloc[:i+1], i) for i in range(len(frame))) -- simpler than
    the engine's own flat-vs-in-position double-call convention, but
    sufficient for a completion-time smoke test: it is not testing exact
    entry/exit timing, only that the function does not hang at real scale.
    """
    n = len(frame)
    for i in range(n):
        signal_fn(frame.iloc[: i + 1], i)


# ---------------------------------------------------------------------
# Real-scale marginal-call-cost test
# ---------------------------------------------------------------------


def test_intraday_rsi2_reversion_daytrade_marginal_call_cost_is_bounded():
    """Marginal-cost proof for the most sensitive factory: a fixed-size batch of
    calls near the START of a long frame must cost about the same as the
    SAME-size batch near the END. A naive total-runtime-vs-bar-count ratio
    is confounded by pandas' own per-call constant overhead at small n; this
    isolates the real question -- does an identical amount of work cost the
    same regardless of how much history sits behind it.
    """
    frame = _rth_minute_frame(n_days=120)
    n = len(frame)
    batch = 2000
    assert n > 3 * batch

    start_fn = make_intraday_rsi2_reversion_daytrade_signal_fn()
    t0 = time.perf_counter()
    for i in range(batch):
        start_fn(frame.iloc[: i + 1], i)
    start_elapsed = time.perf_counter() - t0

    end_fn = make_intraday_rsi2_reversion_daytrade_signal_fn()
    for i in range(n - batch):
        end_fn(frame.iloc[: i + 1], i)
    t0 = time.perf_counter()
    for i in range(n - batch, n):
        end_fn(frame.iloc[: i + 1], i)
    end_elapsed = time.perf_counter() - t0

    ratio = end_elapsed / max(start_elapsed, 1e-6)
    print(f"\n[intraday_rsi2] start={start_elapsed:.4f}s end={end_elapsed:.4f}s ratio={ratio:.2f}x")
    assert end_elapsed < max(5 * start_elapsed, start_elapsed + 2.0), (
        f"a {batch}-call batch near the end of a {n}-bar frame took {ratio:.1f}x "
        f"longer than the same-size batch near the start (start={start_elapsed:.3f}s, "
        f"end={end_elapsed:.3f}s) -- per-call cost is still growing with total "
        "history, not bounded"
    )


# ---------------------------------------------------------------------
# Value-equivalence tests: proves each row-count-bounded helper returns the
# SAME numbers as the unbounded algorithm, not just faster ones.
# Each REFERENCE function below is the reference (unbounded)
# implementation; each FAST function mirrors the current, bounded
# production code exactly. If production changes, these copies must be
# updated to match, or these tests stop proving anything.
# ---------------------------------------------------------------------


def _reference_atr14(frame, before_date, lookback):
    """Reference (unbounded) `_compute_atr14` (make_orb_5min_atr_stop_scalp_signal_fn)."""
    hist = frame[frame.index.date < before_date]
    if hist.empty:
        return None
    daily = hist.groupby(hist.index.date).agg(
        High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"),
    )
    if len(daily) < 2:
        return None
    prev_close = daily["Close"].shift(1)
    tr = pd.concat(
        [
            daily["High"] - daily["Low"],
            (daily["High"] - prev_close).abs(),
            (daily["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.dropna()
    if len(tr) < lookback:
        return None
    return float(tr.tail(lookback).mean())


def _fast_atr14(frame, before_date, lookback):
    """Mirrors the bounded production `_compute_atr14` exactly."""
    bound_bars = (lookback + 15) * 400
    tail = frame.tail(bound_bars) if len(frame) > bound_bars else frame
    hist = tail[tail.index.date < before_date]
    if hist.empty:
        return None
    daily = hist.groupby(hist.index.date).agg(
        High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"),
    )
    if len(daily) < lookback:
        return None
    daily = daily.tail(lookback + 1)
    prev_close = daily["Close"].shift(1)
    tr = pd.concat(
        [
            daily["High"] - daily["Low"],
            (daily["High"] - prev_close).abs(),
            (daily["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.dropna()
    if len(tr) < lookback:
        return None
    return float(tr.tail(lookback).mean())


def test_atr14_boundary_guard_matches_reference_at_exactly_lookback_days():
    """Boundary test at exactly `len(daily) == lookback` prior trading
    days: there `tr`'s `.dropna()` is effectively a no-op (High-Low is
    always real/non-NaN, so pandas' skipna `.max(axis=1)` keeps that row
    even when both gap-vs-prior-close terms are NaN), so the reference
    `len(tr) < lookback` check reduces to `len(daily) < lookback`, NOT
    `len(daily) < lookback + 1`. A bounded version using the `+ 1`
    threshold (as a superficially similar guard in swing_signals.py's
    sibling helper does) would return None where the reference returns a
    real (if degenerate-first-day) ATR value. Checked across many
    `n_prior_days` values around the boundary, not just the exact edge,
    so a future regression in either direction is caught.
    """
    lookback = 14
    for n_prior_days in (12, 13, 14, 15, 16, 17, 20, 30):
        rng = np.random.default_rng(n_prior_days)
        days = pd.bdate_range(start="2020-01-02", periods=n_prior_days + 5, tz="America/New_York")
        idx = []
        for day in days:
            session_start = day.replace(hour=9, minute=30)
            idx.extend(pd.date_range(session_start, periods=BARS_PER_SESSION, freq="1min"))
        n = len(idx)
        close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
        high = close + np.abs(rng.normal(0, 0.05, n))
        low = close - np.abs(rng.normal(0, 0.05, n))
        volume = rng.integers(100, 1000, n).astype(float)
        frame = pd.DataFrame(
            {"Open": close.copy(), "High": high, "Low": low, "Close": close, "Volume": volume},
            index=pd.DatetimeIndex(idx, name="timestamp"),
        )
        all_days = sorted(set(frame.index.date))
        before_date = all_days[n_prior_days]  # exactly n_prior_days strictly-prior days

        ref = _reference_atr14(frame, before_date, lookback)
        fast = _fast_atr14(frame, before_date, lookback)
        assert ref == fast or (
            ref is not None and fast is not None and abs(ref - fast) < 1e-9
        ), f"divergence at n_prior_days={n_prior_days}: ref={ref} fast={fast}"


def test_atr14_fast_path_matches_unbounded_reference():
    """`frame` is sliced to end at each sampled date, mirroring the real
    engine's own invariant that signal_fn's `frame` argument never contains
    data past "now" -- comparing against the untouched full-250-day frame
    regardless of `before_date` would test a scenario the real engine can
    never actually produce (an early `before_date` paired with a frame that
    already extends hundreds of days into its own future), which is not
    what the fast path's tail-bound needs to be correct against.
    """
    full_frame = _rth_minute_frame(n_days=250)
    all_days = sorted(set(full_frame.index.date))
    sample_days = all_days[10::13] + [all_days[-1]]

    checked = 0
    for before_date in sample_days:
        frame = full_frame[full_frame.index.date <= before_date]
        ref = _reference_atr14(frame, before_date, lookback=14)
        fast = _fast_atr14(frame, before_date, lookback=14)
        assert ref == fast or (
            ref is not None and fast is not None and abs(ref - fast) < 1e-9
        ), f"divergence at before_date={before_date}: ref={ref} fast={fast}"
        if ref is not None:
            checked += 1
    assert checked >= 15


def _session_early_bars_ref(day_bars, or_minutes):
    if day_bars.empty:
        return None
    start = day_bars.index[0]
    if start.time() != datetime.time(9, 30):
        return None
    cutoff = start + pd.Timedelta(minutes=or_minutes)
    early = day_bars[day_bars.index < cutoff]
    if len(early) < or_minutes:
        return None
    return early


def _reference_orb_pooled_rvol_in_play(frame, today, today_bars, rvol_lookback, min_hist_days, or_minutes):
    """Reference (unbounded) `_compute_in_play_and_or`'s relative-volume
    half (the or_high/or_low computation has no bound and is shared as-is)."""
    early = _session_early_bars_ref(today_bars, or_minutes)
    if early is None:
        return None
    today_early_vol = float(early["Volume"].sum())

    idx = frame.index
    window_start = today - datetime.timedelta(days=max(rvol_lookback * 3, 60))
    mask = (idx.date >= window_start) & (idx.date < today)
    recent = frame.loc[mask]
    if recent.empty:
        return False

    recent_dates = recent.index.date
    hist_dates = sorted(set(recent_dates))
    if len(hist_dates) < min_hist_days:
        return False
    lookback_dates = (
        hist_dates[-rvol_lookback:] if len(hist_dates) > rvol_lookback else hist_dates
    )

    hist_vols = []
    for d in lookback_dates:
        d_bars = recent[recent_dates == d]
        d_early = _session_early_bars_ref(d_bars, or_minutes)
        if d_early is None:
            continue
        hist_vols.append(float(d_early["Volume"].sum()))

    if len(hist_vols) < min_hist_days:
        return False

    threshold = float(np.percentile(hist_vols, 100.0 - 10.0))
    return today_early_vol >= threshold


def _fast_orb_pooled_rvol_in_play(frame, today, today_bars, rvol_lookback, min_hist_days, or_minutes):
    """Mirrors the bounded production `_compute_in_play_and_or` exactly."""
    early = _session_early_bars_ref(today_bars, or_minutes)
    if early is None:
        return None
    today_early_vol = float(early["Volume"].sum())

    window_days = max(rvol_lookback * 3, 60)
    bound_bars = (window_days + 10) * 400
    tail_frame = frame.tail(bound_bars) if len(frame) > bound_bars else frame
    window_start = today - datetime.timedelta(days=window_days)
    tail_date = tail_frame.index.date
    mask = (tail_date >= window_start) & (tail_date < today)
    recent = tail_frame.loc[mask]
    if recent.empty:
        return False

    recent_dates = recent.index.date
    hist_dates = sorted(set(recent_dates))
    if len(hist_dates) < min_hist_days:
        return False
    lookback_dates = (
        hist_dates[-rvol_lookback:] if len(hist_dates) > rvol_lookback else hist_dates
    )

    hist_vols = []
    for d in lookback_dates:
        d_bars = recent[recent_dates == d]
        d_early = _session_early_bars_ref(d_bars, or_minutes)
        if d_early is None:
            continue
        hist_vols.append(float(d_early["Volume"].sum()))

    if len(hist_vols) < min_hist_days:
        return False

    threshold = float(np.percentile(hist_vols, 100.0 - 10.0))
    return today_early_vol >= threshold


def _reference_orb_pooled_daily_atr(frame, today, atr_days):
    """Reference (unbounded) `_compute_daily_atr`
    (make_volume_filtered_orb_stocks_in_play_pooled_signal_fn)."""
    idx = frame.index
    window_start = today - datetime.timedelta(days=max(atr_days * 3, 45))
    mask = (idx.date >= window_start) & (idx.date < today)
    windowed = frame.loc[mask]
    if windowed.empty:
        return None
    daily = windowed.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    if len(daily) < atr_days + 1:
        return None
    prev_close = daily["Close"].shift(1)
    true_range = pd.concat(
        [
            daily["High"] - daily["Low"],
            (daily["High"] - prev_close).abs(),
            (daily["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_value = true_range.rolling(atr_days).mean().iloc[-1]
    if pd.isna(atr_value) or atr_value <= 0:
        return None
    return float(atr_value)


def _fast_orb_pooled_daily_atr(frame, today, atr_days):
    """Mirrors the bounded production `_compute_daily_atr` exactly."""
    window_days = max(atr_days * 3, 45)
    bound_bars = (window_days + 10) * 400
    tail_frame = frame.tail(bound_bars) if len(frame) > bound_bars else frame
    window_start = today - datetime.timedelta(days=window_days)
    tail_date = tail_frame.index.date
    mask = (tail_date >= window_start) & (tail_date < today)
    windowed = tail_frame.loc[mask]
    if windowed.empty:
        return None
    daily = windowed.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    if len(daily) < atr_days + 1:
        return None
    prev_close = daily["Close"].shift(1)
    true_range = pd.concat(
        [
            daily["High"] - daily["Low"],
            (daily["High"] - prev_close).abs(),
            (daily["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_value = true_range.rolling(atr_days).mean().iloc[-1]
    if pd.isna(atr_value) or atr_value <= 0:
        return None
    return float(atr_value)


def test_orb_pooled_daily_atr_fast_path_matches_unbounded_reference():
    """`frame` sliced to end at each sampled date -- see the rationale in
    test_atr14_fast_path_matches_unbounded_reference above."""
    full_frame = _rth_minute_frame(n_days=250)
    all_days = sorted(set(full_frame.index.date))
    sample_days = all_days[10::13] + [all_days[-1]]

    checked = 0
    for today in sample_days:
        frame = full_frame[full_frame.index.date <= today]
        ref = _reference_orb_pooled_daily_atr(frame, today, atr_days=14)
        fast = _fast_orb_pooled_daily_atr(frame, today, atr_days=14)
        assert ref == fast or (
            ref is not None and fast is not None and abs(ref - fast) < 1e-9
        ), f"divergence at today={today}: ref={ref} fast={fast}"
        if ref is not None:
            checked += 1
    assert checked >= 15


def test_orb_pooled_rvol_in_play_fast_path_matches_unbounded_reference():
    """`frame` sliced to end at each sampled date -- see the rationale in
    test_atr14_fast_path_matches_unbounded_reference above."""
    full_frame = _rth_minute_frame(n_days=250)
    all_days = sorted(set(full_frame.index.date))
    sample_days = all_days[10::13] + [all_days[-1]]

    checked = 0
    for today in sample_days:
        frame = full_frame[full_frame.index.date <= today]
        today_bars = frame[frame.index.date == today]
        ref = _reference_orb_pooled_rvol_in_play(
            frame, today, today_bars, rvol_lookback=20, min_hist_days=5, or_minutes=5
        )
        fast = _fast_orb_pooled_rvol_in_play(
            frame, today, today_bars, rvol_lookback=20, min_hist_days=5, or_minutes=5
        )
        assert ref == fast, f"divergence at today={today}: ref={ref} fast={fast}"
        if ref is not None:
            checked += 1
    assert checked >= 15


def _reference_daily_bars(frame):
    """Reference (unbounded) `_daily_bars` (make_overnight_gap_meanreversion_fade_signal_fn)."""
    daily = frame.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return daily.dropna(subset=["Open", "High", "Low", "Close"])


def _fast_daily_bars(frame, needed_days):
    """Mirrors the bounded production `_daily_bars` exactly."""
    bound_bars = (needed_days + 10) * 400
    tail = frame.tail(bound_bars) if len(frame) > bound_bars else frame
    daily = tail.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return daily.dropna(subset=["Open", "High", "Low", "Close"])


def test_daily_bars_fast_path_matches_unbounded_reference():
    """Compares the tail portion of the two DataFrames -- only the last
    max(gap_lookback_days, atr_lookback_days) + 1 rows are ever read by
    production (today's open, and the trailing gap/ATR lookback windows),
    so that is exactly what must match.
    """
    frame = _rth_minute_frame(n_days=250)
    needed_days = max(60, 14)  # factory defaults: gap_lookback_days=60, atr_lookback_days=14

    all_days = sorted(set(frame.index.date))
    sample_day_positions = list(range(80, len(all_days), 11)) + [len(all_days) - 1]

    checked = 0
    for pos in sample_day_positions:
        cutoff_date = all_days[pos]
        # Slice the frame at a realistic point: through the last bar of
        # cutoff_date (mirrors what signal_fn's own `frame` looks like when
        # the once-per-day entry check runs).
        sliced = frame[frame.index.date <= cutoff_date]

        ref_daily = _reference_daily_bars(sliced)
        fast_daily = _fast_daily_bars(sliced, needed_days)

        ref_tail = ref_daily.tail(needed_days + 1)
        fast_tail = fast_daily.tail(needed_days + 1)
        pd.testing.assert_frame_equal(ref_tail, fast_tail, check_exact=False, rtol=1e-9)
        checked += 1

    assert checked > 10


def _reference_climax_baselines(frame, current_date, lookback_days, checkpoint_time, min_days_required):
    """Reference (unbounded) daily-baseline block
    (make_selling_climax_capitulation_bounce_daytrade_signal_fn)."""
    prior_all = frame[frame.index.date < current_date]
    if prior_all.empty:
        return None, None
    unique_dates = pd.unique(prior_all.index.date)
    if len(unique_dates) > lookback_days:
        recent_dates = unique_dates[-lookback_days:]
    else:
        recent_dates = unique_dates
    recent_dates_set = set(recent_dates.tolist())
    mask = np.array([d in recent_dates_set for d in prior_all.index.date])
    prior = prior_all[mask]

    daily_ranges = []
    daily_cumvols_at_ckpt = []
    for _, day_rows in prior.groupby(prior.index.date):
        day_rows = day_rows.sort_index()
        daily_ranges.append(float(day_rows["High"].max() - day_rows["Low"].min()))
        upto_ckpt = day_rows[day_rows.index.time <= checkpoint_time]
        if len(upto_ckpt) > 0:
            daily_cumvols_at_ckpt.append(float(upto_ckpt["Volume"].sum()))

    avg_range = float(np.mean(daily_ranges)) if len(daily_ranges) >= min_days_required else None
    avg_cumvol = (
        float(np.mean(daily_cumvols_at_ckpt))
        if len(daily_cumvols_at_ckpt) >= min_days_required
        else None
    )
    return avg_range, avg_cumvol


def _fast_climax_baselines(frame, current_date, lookback_days, checkpoint_time, min_days_required):
    """Mirrors the bounded production daily-baseline block exactly."""
    lookback_bound_bars = (lookback_days + 15) * 400
    tail_frame = frame.tail(lookback_bound_bars) if len(frame) > lookback_bound_bars else frame
    prior_all = tail_frame[tail_frame.index.date < current_date]
    if prior_all.empty:
        return None, None
    unique_dates = pd.unique(prior_all.index.date)
    if len(unique_dates) > lookback_days:
        recent_dates = unique_dates[-lookback_days:]
    else:
        recent_dates = unique_dates
    recent_dates_set = set(recent_dates.tolist())
    mask = np.array([d in recent_dates_set for d in prior_all.index.date])
    prior = prior_all[mask]

    daily_ranges = []
    daily_cumvols_at_ckpt = []
    for _, day_rows in prior.groupby(prior.index.date):
        day_rows = day_rows.sort_index()
        daily_ranges.append(float(day_rows["High"].max() - day_rows["Low"].min()))
        upto_ckpt = day_rows[day_rows.index.time <= checkpoint_time]
        if len(upto_ckpt) > 0:
            daily_cumvols_at_ckpt.append(float(upto_ckpt["Volume"].sum()))

    avg_range = float(np.mean(daily_ranges)) if len(daily_ranges) >= min_days_required else None
    avg_cumvol = (
        float(np.mean(daily_cumvols_at_ckpt))
        if len(daily_cumvols_at_ckpt) >= min_days_required
        else None
    )
    return avg_range, avg_cumvol


def test_selling_climax_baselines_fast_path_matches_unbounded_reference():
    """`frame` sliced to end at each sampled date -- see the rationale in
    test_atr14_fast_path_matches_unbounded_reference above."""
    full_frame = _rth_minute_frame(n_days=250)
    checkpoint_time = datetime.time(10, 30)
    lookback_days = 20
    min_days_required = min(5, lookback_days)

    all_days = sorted(set(full_frame.index.date))
    sample_days = all_days[10::13] + [all_days[-1]]

    checked = 0
    for current_date in sample_days:
        frame = full_frame[full_frame.index.date <= current_date]
        ref_range, ref_vol = _reference_climax_baselines(
            frame, current_date, lookback_days, checkpoint_time, min_days_required
        )
        fast_range, fast_vol = _fast_climax_baselines(
            frame, current_date, lookback_days, checkpoint_time, min_days_required
        )
        assert ref_range == fast_range or (
            ref_range is not None and fast_range is not None and abs(ref_range - fast_range) < 1e-9
        ), f"avg_range divergence at current_date={current_date}: ref={ref_range} fast={fast_range}"
        assert ref_vol == fast_vol or (
            ref_vol is not None and fast_vol is not None and abs(ref_vol - fast_vol) < 1e-6
        ), f"avg_cumvol divergence at current_date={current_date}: ref={ref_vol} fast={fast_vol}"
        if ref_range is not None:
            checked += 1

    assert checked > 15
