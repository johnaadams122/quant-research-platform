"""Real-scale performance regression tests for swing signal_fn factories in
mining/swing_signals.py whose per-call work must stay bounded: timing
checks for `make_anchored_vwap_reclaim_continuation_signal_fn`,
`make_closing_auction_deviation_reversal_signal_fn` and
`make_avwap_band_extension_reversion_signal_fn`; value-identity checks for
two vectorized helpers (`_time_of_day_avg_volume` inside the anchored-VWAP
factory and `_daily_atr` inside `make_vwap_reclaim_reversal_signal_fn`);
and shared frame builders/drivers for the same bounded-cost pattern in the
other factories (`make_vcp_breakout_volume_confirmation_swing_signal_fn`,
`make_darvas_box_breakout_volume_confirmed_signal_fn`,
`make_intraday_liquidity_sweep_reversal_signal_fn`,
`make_earnings_reaction_settle_drift_signal_fn`,
`make_vwap_reclaim_reversal_signal_fn`).

Why per-call cost matters:
the intraday backtest engine (`run_intraday_backtest` in
mining/intraday_backtest_engine.py) hands each signal_fn a FRESH SLICE of
the frame ending at the current bar on every single call
(`df.iloc[:i+1]`/`df.iloc[:i]`), which GROWS by one row every call. A
helper that recomputes something (a daily resample, an anchored-VWAP sum,
a full-frame boolean day mask) over the ENTIRE received frame on every
call instead of a bounded recent window costs O(len(frame)) per call x
O(len(frame)) calls = O(n^2) total. At a multi-year 1-minute history
(hundreds of thousands of bars) that is not "slow", it does not
realistically finish.

The timing tests are relative comparisons rather than a single fixed
wall-clock number, so they stay meaningful across machines: a fixed-size
batch of calls near the START of a long frame (shallow history) must cost
about the same as the SAME-size batch near the END (deep history). A
total-runtime-vs-bar-count ratio would be confounded by the internal
lookback window's "filling up" transient, which makes a short run look
artificially cheap relative to a long one even when per-call cost is
bounded. Every test drives frames at real scale (over ten thousand bars,
not a few hundred), since a correctness-only check at fixture
scale is silent on this failure mode by construction.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.swing_signals import (
    make_anchored_vwap_reclaim_continuation_signal_fn,
    make_avwap_band_extension_reversion_signal_fn,
    make_closing_auction_deviation_reversal_signal_fn,
    make_darvas_box_breakout_volume_confirmed_signal_fn,
    make_earnings_reaction_settle_drift_signal_fn,
    make_intraday_liquidity_sweep_reversal_signal_fn,
    make_vcp_breakout_volume_confirmation_swing_signal_fn,
    make_vwap_reclaim_reversal_signal_fn,
)

BARS_PER_SESSION = 390  # 09:30-15:59 ET inclusive, RTH-only (regular-session 1-minute bars)


def _rth_minute_frame(n_days, start_date="2020-01-02", seed=7):
    """Synthetic RTH-shaped 1-minute bar frame spanning `n_days` business
    days (09:30-15:59 ET, BARS_PER_SESSION bars/day, weekends skipped --
    real NYSE holidays are not modeled, irrelevant for a pure timing
    harness). Deterministic (seeded) slow random walk in Close; Open/
    High/Low/Volume derived from it so every column is sane and every
    bar has real Volume (needed by _time_of_day_avg_volume's >=5-
    observation floor). Includes a VWAP column (a small, deterministic
    offset from Close) since several factories read `frame["VWAP"]`
    directly with no Close fallback.
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
    vwap = close + rng.normal(0, 0.003, n)
    index = pd.DatetimeIndex(idx, name="timestamp")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume, "VWAP": vwap},
        index=index,
    )


def _synthetic_earnings_events(n_days, start_date="2020-01-02", every_n_days=10):
    """Synthetic earnings_events for make_earnings_reaction_settle_drift_signal_fn
    -- real earnings only happen ~4x/year, but a performance smoke test
    needs its once-per-reaction-day `_compute_baseline` call exercised
    repeatedly across the driven range, not just zero or one time.
    """
    days = pd.bdate_range(start=start_date, periods=n_days, tz="America/New_York")
    events = []
    for pos in range(0, len(days), every_n_days):
        d = days[pos].date()
        events.append({"date": d.isoformat(), "surprise_pct": 5.0, "hour": 16})
    return events


def _bucket_close_indices(frame, bucket_minutes=5):
    """Positions of every bucket-closing bar (the only bars
    make_anchored_vwap_reclaim_continuation_signal_fn's entry path does
    real work on -- all other bars hit an O(1) early return on the
    minute-of-day check, so driving only these positions reproduces the
    exact same expensive-call pattern and frame-length growth the real
    engine would produce, at a fraction of the wall-clock setup cost).
    """
    minute_of_day = frame.index.hour * 60 + frame.index.minute
    offset = minute_of_day - (9 * 60 + 30)
    mask = (offset >= 0) & (offset % bucket_minutes == bucket_minutes - 1)
    return np.flatnonzero(mask)


def _drive_flat_search(signal_fn, frame, indices):
    """Replicate the real engine's flat-path dispatch
    (signal_fn(df.iloc[:i], i - 1), in `run_intraday_backtest`) at
    each 1-based position in `indices`."""
    for i in indices:
        pos = int(i) + 1
        signal_fn(frame.iloc[:pos], pos - 1)


def _closing_bar_indices(frame, closing_time_str="15:59"):
    h, m = (int(x) for x in closing_time_str.split(":"))
    mask = (frame.index.hour == h) & (frame.index.minute == m)
    return np.flatnonzero(mask)


def _drive_closing_bars(signal_fn, frame, indices):
    """Replicate the real engine's per-bar dispatch
    (signal_fn(df.iloc[:i+1], i), in `run_intraday_backtest`) at
    each closing-bar position -- the only bars
    make_closing_auction_deviation_reversal_signal_fn does real work on;
    every other bar hits an O(1) early return on the exact-time check."""
    for i in indices:
        signal_fn(frame.iloc[: int(i) + 1], int(i))


def _reference_time_of_day_avg_volume(frame, today, bucket_time, bar_minutes):
    """The reference (unbounded) implementation of
    `_time_of_day_avg_volume`, written with the slow `.index.date`/
    `.index.time` object-array conversions -- kept here as a test oracle
    so the fast, vectorized production version (`.normalize()` +
    hour/minute integer arithmetic) can be asserted value-identical to
    it, not merely faster. The function's own contract is an unbounded
    "all prior days" average, so a speed-up that ALSO bounded the window
    to a fixed recent lookback (for example ~25 trading days) would
    change its value and could flip an entry decision; this oracle
    catches exactly that. Keep it the plain unbounded algorithm; do not
    "improve" it.
    """
    import datetime as _dt

    prior = frame.loc[frame.index.date < today]
    if prior.empty:
        return None
    end_minute = bucket_time.hour * 60 + bucket_time.minute + bar_minutes - 1
    end_h, end_m = divmod(end_minute, 60)
    end_time = _dt.time(end_h, end_m)
    mask = (prior.index.time >= bucket_time) & (prior.index.time <= end_time)
    same_slot = prior.loc[mask]
    if same_slot.empty:
        return None
    daily_sums = same_slot.groupby(same_slot.index.date)["Volume"].sum()
    if len(daily_sums) < 5:
        return None
    return float(daily_sums.mean())


def _fast_time_of_day_avg_volume(frame, today, bucket_time, bar_minutes):
    """Mirrors the production `_time_of_day_avg_volume` in
    src/mining/swing_signals.py exactly -- kept as a standalone copy
    here (rather than importing the private nested closure, which isn't
    reachable from outside its factory) purely so this test can compare
    it against `_reference_time_of_day_avg_volume` above on the same
    inputs. If production's implementation changes, this copy must be
    updated to match, or this test stops proving anything.
    """
    if frame.empty:
        return None
    today_ts = pd.Timestamp(today, tz=frame.index.tz)
    prior = frame.loc[frame.index.normalize() < today_ts]
    if prior.empty:
        return None
    bucket_start_minute = bucket_time.hour * 60 + bucket_time.minute
    bucket_end_minute = bucket_start_minute + bar_minutes - 1
    minute_of_day = prior.index.hour * 60 + prior.index.minute
    mask = (minute_of_day >= bucket_start_minute) & (minute_of_day <= bucket_end_minute)
    same_slot = prior.loc[mask]
    if same_slot.empty:
        return None
    daily_sums = same_slot.groupby(same_slot.index.date)["Volume"].sum()
    if len(daily_sums) < 5:
        return None
    return float(daily_sums.mean())


def test_time_of_day_avg_volume_fast_path_matches_unbounded_reference():
    """Proves the fast vectorized version is value-IDENTICAL to the
    unbounded reference algorithm, not just faster -- the property a
    speed-up that bounded the window (instead of only speeding up the
    scan) would violate. Checked across many distinct
    (today, bucket_time) pairs spanning a multi-year synthetic frame,
    including bucket times crossing an hour boundary.
    """
    import datetime as _dt

    frame = _rth_minute_frame(n_days=300)
    all_days = sorted(set(frame.index.date))
    # Skip the earliest days (need >=5 prior days of history for either
    # implementation to return a real number rather than None) and
    # sample across the rest, including the very last day (maximum
    # prior-history size, where a bounded-window bug would be most
    # visible).
    sample_days = all_days[10::17] + [all_days[-1]]
    bucket_times = [_dt.time(9, 30), _dt.time(9, 47), _dt.time(11, 3), _dt.time(15, 55)]

    checked = 0
    for today in sample_days:
        for bucket_time in bucket_times:
            for bar_minutes in (5, 7):
                ref = _reference_time_of_day_avg_volume(frame, today, bucket_time, bar_minutes)
                fast = _fast_time_of_day_avg_volume(frame, today, bucket_time, bar_minutes)
                assert ref == fast or (
                    ref is not None and fast is not None and abs(ref - fast) < 1e-9
                ), f"divergence at today={today} bucket_time={bucket_time} bar_minutes={bar_minutes}: ref={ref} fast={fast}"
                if ref is not None:
                    checked += 1
    # Sanity: the sampling actually exercised the >=5-prior-days real
    # path many times, not just the None/insufficient-history branch.
    assert checked > 20


def _reference_daily_atr(frame, today, atr_lookback_days=20):
    """The reference (resample-based) implementation of
    `make_vwap_reclaim_reversal_signal_fn`'s internal `_daily_atr`, using
    the slow `.index.date` conversion -- kept here as a test oracle so the
    fast, numpy-segment production version can be asserted value-identical
    to it, not merely faster. Same convention as
    `_reference_time_of_day_avg_volume` above: keep it the plain
    resample-based algorithm; do not "improve" it.
    """
    lookback_minutes = (atr_lookback_days + 15) * 390
    tail = frame.tail(lookback_minutes) if len(frame) > lookback_minutes else frame
    prior = tail[tail.index.date < today]
    if prior.empty:
        return None
    daily = prior.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    daily = daily.dropna(subset=["Open"])
    if len(daily) < atr_lookback_days + 1:
        return None
    daily = daily.tail(atr_lookback_days + 1)
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
    if len(tr) < atr_lookback_days:
        return None
    return float(tr.tail(atr_lookback_days).mean())


def _fast_daily_atr(frame, today, atr_lookback_days=20):
    """Mirrors the production `_daily_atr` inside
    `make_vwap_reclaim_reversal_signal_fn` in src/mining/swing_signals.py
    exactly -- kept as a standalone copy here for the same reason
    `_fast_time_of_day_avg_volume` is: the private nested closure isn't
    reachable from outside its factory. If production's implementation
    changes, this copy must be updated to match, or this test stops
    proving anything.
    """
    lookback_minutes = (atr_lookback_days + 15) * 390
    tail = frame.tail(lookback_minutes) if len(frame) > lookback_minutes else frame
    today_ts = pd.Timestamp(today, tz=tail.index.tz)
    pos = tail.index.searchsorted(today_ts, side="left")
    if pos == 0:
        return None
    prior = tail.iloc[:pos]
    prior_days = prior.index.asi8 // 86_400_000_000_000

    day_starts = np.flatnonzero(np.r_[True, prior_days[1:] != prior_days[:-1]])
    if len(day_starts) < atr_lookback_days + 1:
        return None
    day_starts = day_starts[-(atr_lookback_days + 1):]
    day_ends = np.r_[day_starts[1:], len(prior_days)]

    high = prior["High"].to_numpy(dtype=float)
    low = prior["Low"].to_numpy(dtype=float)
    close = prior["Close"].to_numpy(dtype=float)
    daily_high = np.maximum.reduceat(high, day_starts)
    daily_low = np.minimum.reduceat(low, day_starts)
    daily_close = close[day_ends - 1]

    prev_close = daily_close[:-1]
    hi = daily_high[1:]
    lo = daily_low[1:]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_close), np.abs(lo - prev_close)))
    if len(tr) < atr_lookback_days:
        return None
    return float(tr[-atr_lookback_days:].mean())


def test_vwap_reclaim_reversal_daily_atr_fast_path_matches_unbounded_reference():
    """Proves the numpy-segment version of
    `make_vwap_reclaim_reversal_signal_fn`'s internal `_daily_atr` is
    value-IDENTICAL to the resample-based reference algorithm, not merely
    faster -- same convention as
    `test_time_of_day_avg_volume_fast_path_matches_unbounded_reference`
    above (an optimization can be faster yet silently wrong, so speed
    alone proves nothing). Checked across
    many distinct (today, atr_lookback_days) pairs spanning a multi-year
    synthetic frame, including lookbacks short and long enough to change
    how many prior trading days are actually consumed.
    """
    frame = _rth_minute_frame(n_days=120)
    all_days = sorted(set(frame.index.date))
    sample_days = all_days[10::13] + [all_days[-1]]

    checked = 0
    for today in sample_days:
        # Both `_daily_atr` implementations are always called (in
        # production, via signal_fn) with a frame that ends AT `today` --
        # `ts = frame.index[-1]; today = ts.date()`. Passing the FULL
        # multi-year frame instead (which extends well past `today`) would
        # make the `< today` filter vacuous on the wrong end and silently
        # test nothing.
        sub = frame[frame.index.date <= today]
        for atr_lookback_days in (5, 20, 40):
            ref = _reference_daily_atr(sub, today, atr_lookback_days)
            fast = _fast_daily_atr(sub, today, atr_lookback_days)
            assert ref == fast or (
                ref is not None and fast is not None and abs(ref - fast) < 1e-9
            ), (
                f"divergence at today={today} atr_lookback_days="
                f"{atr_lookback_days}: ref={ref} fast={fast}"
            )
            if ref is not None:
                checked += 1
    # Sanity: the sampling actually exercised the real (non-None) path
    # many times, not just the insufficient-history branch.
    assert checked > 20


def test_anchored_vwap_reclaim_continuation_marginal_call_cost_is_bounded():
    """The entry-search path's PER-CALL cost must not grow with how much
    history precedes the call -- once frame length exceeds the internal
    lookback bound, adding more history must not make each subsequent
    call any slower. Measured as a marginal-cost comparison rather than
    a total-runtime-vs-bar-count ratio: a naive small-vs-large comparison
    is confounded by the internal window's own "filling up" transient
    (a short run never reaches the steady-state bounded cost at all,
    making it look artificially cheap relative to a long run that does)
    -- timing a fixed-size batch of calls near the START of one long
    frame against the SAME-size batch near the END isolates the real
    question: does an identical amount of work cost the same regardless
    of how much history sits behind it. A bug that still scans the
    whole received frame on every call would show the END batch taking
    much longer than the START batch even though both process the same
    number of calls.
    """
    frame = _rth_minute_frame(n_days=200)
    kwargs = dict(min_volume_multiple_of_bar_avg=1_000_000.0)
    bucket_indices = _bucket_close_indices(frame)
    batch = 400
    assert len(bucket_indices) > 2 * batch

    start_fn = make_anchored_vwap_reclaim_continuation_signal_fn(**kwargs)
    start_indices = bucket_indices[:batch]
    t0 = time.perf_counter()
    _drive_flat_search(start_fn, frame, start_indices)
    start_elapsed = time.perf_counter() - t0

    end_fn = make_anchored_vwap_reclaim_continuation_signal_fn(**kwargs)
    # Drive through everything before the measured batch first (untimed)
    # so state reflects deep history, matching what a real backtest's
    # late calls would see.
    warm_up_indices = bucket_indices[:-batch]
    _drive_flat_search(end_fn, frame, warm_up_indices)
    end_indices = bucket_indices[-batch:]
    t0 = time.perf_counter()
    _drive_flat_search(end_fn, frame, end_indices)
    end_elapsed = time.perf_counter() - t0

    ratio = end_elapsed / max(start_elapsed, 1e-6)
    print(f"\n[anchored_vwap] start={start_elapsed:.4f}s end={end_elapsed:.4f}s ratio={ratio:.2f}x")
    assert end_elapsed < max(5 * start_elapsed, start_elapsed + 2.0), (
        f"a {batch}-call batch near the end of a {len(frame)}-bar frame "
        f"took {ratio:.1f}x longer than the same-size batch near the "
        f"start (start={start_elapsed:.3f}s, end={end_elapsed:.3f}s) -- "
        "per-call cost is still growing with total history, not bounded"
    )


def test_closing_auction_deviation_reversal_marginal_call_cost_is_bounded():
    """The once-per-day full-frame boolean day mask
    (`frame.index.date == bar_date`) must be bounded to a recent window,
    not O(len(frame)). Same marginal-cost methodology as the anchored-
    VWAP test above (a fixed-size batch of closing-bar calls near the
    START of a long frame vs the SAME-size batch near the END) rather
    than a total-runtime-vs-day-count ratio, which is confounded by the
    internal window's own "filling up" transient.
    """
    frame = _rth_minute_frame(n_days=800)
    closing_indices = _closing_bar_indices(frame)
    batch = 200
    assert len(closing_indices) > 2 * batch

    start_fn = make_closing_auction_deviation_reversal_signal_fn(deviation_threshold_bps=1e9)
    start_indices = closing_indices[:batch]
    t0 = time.perf_counter()
    _drive_closing_bars(start_fn, frame, start_indices)
    start_elapsed = time.perf_counter() - t0

    end_fn = make_closing_auction_deviation_reversal_signal_fn(deviation_threshold_bps=1e9)
    warm_up_indices = closing_indices[:-batch]
    _drive_closing_bars(end_fn, frame, warm_up_indices)
    end_indices = closing_indices[-batch:]
    t0 = time.perf_counter()
    _drive_closing_bars(end_fn, frame, end_indices)
    end_elapsed = time.perf_counter() - t0

    ratio = end_elapsed / max(start_elapsed, 1e-6)
    print(f"\n[closing_auction] start={start_elapsed:.4f}s end={end_elapsed:.4f}s ratio={ratio:.2f}x")
    assert end_elapsed < max(5 * start_elapsed, start_elapsed + 2.0), (
        f"a {batch}-call batch near the end of a {len(frame)}-bar frame "
        f"took {ratio:.1f}x longer than the same-size batch near the "
        f"start (start={start_elapsed:.3f}s, end={end_elapsed:.3f}s) -- "
        "per-call cost is still growing with total history, not bounded"
    )


def test_anchored_vwap_reclaim_continuation_plateaus_once_past_the_lookback_window():
    """Stronger proof than the start-vs-end test above: once the internal
    lookback window (anchor_lookback_days=20 -> _ANCHOR_SELECTION_WINDOW_BARS
    ~= 26 trading days) has fully filled, per-call cost must STAY flat no
    matter how much MORE history accumulates beyond that point -- a small
    residual O(n) per-call leak (milder than the whole-frame O(n^2)
    pattern, but still unbounded) would still matter at multi-year
    1-minute scale even if it passes the coarser start-vs-end comparison
    above. Compares a batch at
    the frame's own midpoint against a batch at its end, both already
    hundreds of trading days past the window-fill point.
    """
    frame = _rth_minute_frame(n_days=400)
    kwargs = dict(min_volume_multiple_of_bar_avg=1_000_000.0)
    bucket_indices = _bucket_close_indices(frame)
    batch = 400
    mid = len(bucket_indices) // 2
    assert mid > batch and (len(bucket_indices) - mid) > batch

    mid_fn = make_anchored_vwap_reclaim_continuation_signal_fn(**kwargs)
    _drive_flat_search(mid_fn, frame, bucket_indices[: mid - batch])
    t0 = time.perf_counter()
    _drive_flat_search(mid_fn, frame, bucket_indices[mid - batch : mid])
    mid_elapsed = time.perf_counter() - t0

    end_fn = make_anchored_vwap_reclaim_continuation_signal_fn(**kwargs)
    _drive_flat_search(end_fn, frame, bucket_indices[:-batch])
    t0 = time.perf_counter()
    _drive_flat_search(end_fn, frame, bucket_indices[-batch:])
    end_elapsed = time.perf_counter() - t0

    ratio = end_elapsed / max(mid_elapsed, 1e-6)
    print(f"\n[anchored_vwap plateau] mid={mid_elapsed:.4f}s end={end_elapsed:.4f}s ratio={ratio:.2f}x")
    assert end_elapsed < max(3 * mid_elapsed, mid_elapsed + 1.0), (
        f"a {batch}-call batch at the frame's end took {ratio:.1f}x longer "
        f"than the same-size batch at its midpoint, both already well past "
        f"the lookback window filling -- per-call cost has not plateaued"
    )


# ---------------------------------------------------------------------
# Bounded-cost checks for the remaining factories. The stateless
# avwap-band factory is sampled directly at two depths (below).
#
# `_drive_every_bar` drives a factory bar-by-bar over the full range
# (signal_fn(df.iloc[:i+1], i) for every i) rather than the bucket/
# closing-bar subsampling the anchored-VWAP and closing-auction tests
# above use: the other factories don't share one single cheap early-
# return gate to subsample against, and every-bar driving exercises both
# their cheap and expensive code paths the way the real engine actually
# would, so a change that reintroduces the O(n^2) pattern shows up as a
# run that no longer completes in bounded time.
# ---------------------------------------------------------------------


def _drive_every_bar(signal_fn, frame):
    """Replicate the real engine's per-bar dispatch at every position
    (signal_fn(df.iloc[:i+1], i) for i in range(len(frame))) -- simpler
    than the engine's own flat-vs-in-position double-call convention,
    but sufficient for a completion-time smoke test: it is not testing
    exact entry/exit timing, only that the function does not hang at
    real scale.
    """
    n = len(frame)
    for i in range(n):
        signal_fn(frame.iloc[: i + 1], i)


def test_avwap_band_extension_reversion_marginal_call_cost_is_bounded():
    """This factory is STATELESS (a pure function of (frame, i), no
    `state` dict persisted across calls) and has no internal per-day/
    per-bucket gate at all -- every bar takes its full day-by-day
    replay path, making it the most sensitive of these factories to any
    regression toward unbounded per-call cost. Its per-call cost is
    inherently higher than the simpler factories' (a bounded day-by-day
    replay simulation runs on every call, by design) but must still be
    FLAT regardless of total frame depth. Driving every bar of a ~50-day
    frame (~19,500 bars) under a fixed ceiling would not test that: the
    frame is not comfortably past this factory's own internal bound
    ((N_ANCHOR+MAX_HOLD+10)*400 = 16,000 bars at defaults), so most of
    such a run sits in the window-filling transient. The cost is instead
    sampled directly at two depths (no warm-up needed, since the
    function is memoryless) -- immune to that confound and far cheaper
    to run than driving every bar of a much larger frame would be.
    """
    N_SAMPLES = 30
    shallow_frame = _rth_minute_frame(n_days=45)  # just past the internal bound
    deep_frame = _rth_minute_frame(n_days=200)  # comfortably past it

    def _sample_cost(frame):
        fn = make_avwap_band_extension_reversion_signal_fn()
        n = len(frame)
        t0 = time.perf_counter()
        for i in range(n - N_SAMPLES, n):
            fn(frame.iloc[: i + 1], i)
        return time.perf_counter() - t0

    shallow_elapsed = _sample_cost(shallow_frame)
    deep_elapsed = _sample_cost(deep_frame)

    ratio = deep_elapsed / max(shallow_elapsed, 1e-6)
    print(
        f"\n[avwap_band_extension] shallow={shallow_elapsed:.3f}s "
        f"({len(shallow_frame)} bars) deep={deep_elapsed:.3f}s "
        f"({len(deep_frame)} bars) ratio={ratio:.2f}x"
    )
    assert deep_elapsed < max(3 * shallow_elapsed, shallow_elapsed + 2.0), (
        f"a {N_SAMPLES}-call sample at {len(deep_frame)} bars took {ratio:.1f}x "
        f"longer than the same-size sample at {len(shallow_frame)} bars -- "
        "per-call cost is still growing with total history, not bounded"
    )
