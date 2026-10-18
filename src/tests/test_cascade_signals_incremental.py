"""Equivalence + structure tests for the per-day / incremental caching inside
mining.cascade_signals.

WHAT THE CACHES REPLACE. Today's daily bias is a per-DAY constant, yet the
uncached computation rebuilds `_build_daily_bars`/`daily_bias` from a
20,000-row trailing window on every 1-minute bar of a no-bias day; it
re-aggregates today's hourly and 5-minute buckets with
`_build_complete_buckets` on each call; and its cross-asset gate slices the
FULL secondary history (boolean mask + `.index.date` over every row) on each
divergence check. The module caches all
three, and the ONLY acceptable outcome is "identical signals on every input" --
so the tests below drive the optimized module and a frozen uncached reference
implementation (`oracle_cascade_signals_frozen.py`) side by side and require
identical `SignalResult`s at every bar, including under adversarial
(non-sequential) call orders where a naive cache would silently go stale.

Three groups:
  * Characterization tests (optimized vs reference) -- the equivalence net.
  * Unit equivalence of the two caches against the reference expressions,
    including guard-pinning tests: each pins one cache guard, so removing that
    guard makes a test here fail.
  * Structure tests -- the optimized module must build the daily bars at
    most once per day and never call the reference bucket builder on the
    sequential hot path.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TESTS = os.path.dirname(__file__)
for _p in (_SRC, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oracle_cascade_signals_frozen as oracle  # noqa: E402
from mining import cascade_signals as cs  # noqa: E402
from mining.intraday_backtest_engine import run_intraday_backtest  # noqa: E402
from mining.price_structure import daily_bias  # noqa: E402

TZ = "America/New_York"
FORCE_FLAT = "12:45"  # inside the short synthetic session below


# ---------------------------------------------------------------------
# Random real-shaped minute-bar frames
# ---------------------------------------------------------------------

def _random_frame(seed, n_days=6, start="2026-01-05", drop_prob=0.02,
                  session_minutes=210, sigma=0.0004):
    """Heavy-tailed random-walk 1-minute OHLC over `n_days` business days,
    `session_minutes` bars per day from 09:30 ET, with random missing minutes
    (real vendor feeds have gaps) and overnight gaps. Heavy-tailed 5-minute
    block drift makes fair-value gaps, hourly reversals and daily reversals
    all occur, so the cascade's entry path is genuinely exercised."""
    rng = np.random.default_rng(seed)
    price = 100.0
    stamps, rows = [], []
    for d in pd.bdate_range(start, periods=n_days):
        open_ts = pd.Timestamp(d.strftime("%Y-%m-%d") + " 09:30", tz=TZ)
        block = rng.standard_t(2, size=session_minutes // 5 + 1) * 0.0007
        day_drift = rng.normal(0, 0.0001)
        for m in range(session_minutes):
            if rng.random() < drop_prob:
                continue
            o = price
            c = o * (1 + rng.normal(day_drift, sigma) + block[m // 5])
            h = max(o, c) * (1 + abs(rng.normal(0, 0.0002)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.0002)))
            stamps.append(open_ts + pd.Timedelta(minutes=m))
            rows.append((o, h, lo, c, 1000.0))
            price = c
        price *= 1 + rng.normal(0, 0.003)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(stamps),
                        columns=["Open", "High", "Low", "Close", "Volume"])


def _drive(signal_fn, frame, order):
    out = []
    for i in order:
        r = signal_fn(frame.iloc[: i + 1], i)
        out.append((bool(r.entry), bool(r.exit)))
    return out


def _new_and_oracle(kind, frame, secondary=None, **kw):
    kw = {"min_target_r_multiple": 2.0, "force_flat_et": FORCE_FLAT, **kw}
    if kind == "single":
        return (cs.make_fvg_reversal_cascade_signal_fn(**kw),
                oracle.make_fvg_reversal_cascade_signal_fn(**kw))
    return (cs.make_cross_asset_divergence_fvg_cascade_signal_fn(secondary_bars=secondary, **kw),
            oracle.make_cross_asset_divergence_fvg_cascade_signal_fn(secondary_bars=secondary, **kw))


# Seeds chosen by SEARCH (seeds 1-40, both variants) for real coverage, not by
# luck: random frames rarely complete the whole cascade (0-2 entries over four
# tradeable sessions), and seeds that never enter would make "identical to the
# reference" mean only "identically inert". 13 and 18 enter in both variants;
# on 18 the cross-asset divergence gate BLOCKS both single-asset entries (2
# single vs 0 cross), so the gate's own cache path decides a real outcome.
SEEDS_FAST = [13, 18]
SEEDS_SLOW = [pytest.param(s, marks=pytest.mark.slow) for s in (1, 9, 25)]
SEEDS = SEEDS_FAST + SEEDS_SLOW


def _secondary_for(seed, primary):
    """An independent random secondary that only partially covers the
    primary's range (drops its first day) so the permissive missing-data
    default is exercised alongside real comparisons."""
    sec = _random_frame(seed + 1000, drop_prob=0.05)
    return sec[sec.index >= primary.index[0] + pd.Timedelta(days=1)]


# The frozen reference is much slower than the optimized module, so its
# per-(kind, seed) outputs are computed once and shared across tests.
_ORACLE_MEMO: dict = {}


def _oracle_sequential(kind, seed):
    key = (kind, seed)
    if key not in _ORACLE_MEMO:
        frame = _random_frame(seed)
        sec = _secondary_for(seed, frame) if kind == "cross" else None
        _, old_fn = _new_and_oracle(kind, frame, sec)
        _ORACLE_MEMO[key] = _drive(old_fn, frame, range(len(frame)))
    return _ORACLE_MEMO[key]


# ---------------------------------------------------------------------
# A. Characterization: optimized module == frozen reference, bar by bar
# ---------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["single", "cross"])
@pytest.mark.parametrize("seed", SEEDS)
def test_sequential_signals_identical_to_frozen_reference(kind, seed):
    frame = _random_frame(seed)
    sec = _secondary_for(seed, frame) if kind == "cross" else None
    new_fn, _ = _new_and_oracle(kind, frame, sec)
    assert _drive(new_fn, frame, range(len(frame))) == _oracle_sequential(kind, seed)


def test_random_frames_actually_exercise_entries_and_exits():
    """Guard against a vacuous equivalence suite: across the fast seed set the
    REFERENCE module must produce real entries and exits (otherwise 'identical'
    only proves 'identically inert')."""
    res = [r for seed in SEEDS_FAST for r in _oracle_sequential("single", seed)]
    entries, exits = sum(e for e, _ in res), sum(x for _, x in res)
    assert entries >= 3 and exits >= 3, (entries, exits)


def test_divergence_gate_actually_blocks_something_on_a_fast_seed():
    """Second vacuity guard: on at least one fast seed the REFERENCE cross-asset
    module must trade strictly less than the single-asset one, i.e. the
    secondary-frame path really changes an outcome (not merely runs)."""
    blocked = [
        sum(e for e, _ in _oracle_sequential("single", s))
        - sum(e for e, _ in _oracle_sequential("cross", s))
        for s in SEEDS_FAST
    ]
    assert max(blocked) >= 1, blocked


@pytest.mark.parametrize("kind", ["single", "cross"])
@pytest.mark.parametrize("seed", SEEDS_FAST)
def test_future_rows_never_influence_earlier_signals(kind, seed):
    """Causality, pinned permanently:
    perturb every PRIMARY and SECONDARY row from a cut onward, re-drive a fresh
    optimized signal_fn, and require every signal BEFORE the cut to be
    unchanged. Covers the caches' persistence across calls and the
    searchsorted-based secondary slice (min(hi, upto)) -- neither may let a
    row later than the current bar/current_ts leak into a decision.

    Scope: this is a regression guard, not a discriminating proof. The gate
    reads only the secondary bucket the primary just confirmed on, whose rows
    all precede current_ts, so causality there holds by structure -- which is
    also why dropping the min(hi, upto) cap would change no output (see the
    note at the bottom of this file). It would catch a future change that starts
    reading a later secondary bucket or today's rows into the daily context."""
    frame = _random_frame(seed)
    sec = _secondary_for(seed, frame) if kind == "cross" else None
    cut_ts = frame.index[int(len(frame) * 0.7)]
    cut = int((frame.index < cut_ts).sum())

    def perturbed(df):
        out = df.copy()
        later = out.index >= cut_ts
        noise = 1 + np.random.default_rng(5).normal(0, 0.05, size=int(later.sum()))
        for col in ("Open", "High", "Low", "Close"):
            out.loc[later, col] = out.loc[later, col].to_numpy() * noise
        return out

    base_fn, _ = _new_and_oracle(kind, frame, sec)
    base = _drive(base_fn, frame, range(cut))
    frame2 = perturbed(frame)
    sec2 = perturbed(sec) if sec is not None else None
    other_fn, _ = _new_and_oracle(kind, frame2, sec2)
    assert _drive(other_fn, frame2, range(cut)) == base


@pytest.mark.parametrize("kind", ["single", "cross"])
@pytest.mark.parametrize("seed", SEEDS_FAST)
def test_engine_trades_identical_to_frozen_reference(kind, seed):
    """Whole-engine equivalence. The oracle side REPLAYS the frozen reference's
    own recorded bar-by-bar signals (from `_oracle_sequential`) through the
    same engine, so the comparison is on the engine's trades/daily returns
    without paying for the slow reference a second time."""
    from mining.intraday_backtest_engine import SignalResult

    frame = _random_frame(seed)
    sec = _secondary_for(seed, frame) if kind == "cross" else None
    new_fn, _ = _new_and_oracle(kind, frame, sec)
    recorded = _oracle_sequential(kind, seed)
    replay = lambda f, i: SignalResult(entry=recorded[i][0], exit=recorded[i][1])  # noqa: E731
    kw = dict(cost_bps=1.5, tax_haircut_rate=0.0, max_hold_days=1, direction="long")
    t_new, d_new = run_intraday_backtest(frame, new_fn, **kw)
    t_old, d_old = run_intraday_backtest(frame, replay, **kw)
    pd.testing.assert_frame_equal(t_new, t_old, check_exact=True)
    pd.testing.assert_series_equal(d_new, d_old, check_exact=True)


def _adversarial_order(n, seed):
    """Forward runs, repeats, backward steps and cross-day jumps. The engine
    never does this, but a cache keyed on 'the frame only ever grows' must not
    go stale if some future caller (or a test) does."""
    rng = random.Random(seed)
    order, i = [], rng.randrange(n)
    for _ in range(3 * n // 2):
        order.append(i)
        move = rng.random()
        if move < 0.70:
            i = min(n - 1, i + 1)
        elif move < 0.80:
            pass                                   # repeat the same call
        elif move < 0.90:
            i = max(0, i - rng.randrange(1, 8))    # step backwards
        else:
            i = rng.randrange(n)                   # jump anywhere (often another day)
    return order


@pytest.mark.parametrize("kind", ["single", "cross"])
@pytest.mark.parametrize("seed", SEEDS_FAST)
def test_non_sequential_call_orders_identical_to_frozen_reference(kind, seed):
    frame = _random_frame(seed)
    sec = _secondary_for(seed, frame) if kind == "cross" else None
    new_fn, old_fn = _new_and_oracle(kind, frame, sec)
    order = _adversarial_order(len(frame), seed)
    assert _drive(new_fn, frame, order) == _drive(old_fn, frame, order)


def test_identical_when_trailing_windows_are_tiny(monkeypatch):
    """Signal-level smoke under tiny trailing windows: shrink both bounds so
    the 'today' cap and the daily tail truncate on this data, and require the
    optimized module and the oracle -- patched identically -- to agree. NOT
    load-bearing on its own (this frame's cascade may never reach the later
    phases); the tail-shift and today-cap branches themselves are pinned by the
    unit tests `test_prior_day_context_tail_shift_guard` and
    `test_today_buckets_primary_honors_the_trailing_today_cap`."""
    for mod in (cs, oracle):
        monkeypatch.setattr(mod, "_DAILY_LOOKBACK_BARS", 470)   # ~2.2 sessions: slides into day t-2 mid-day
        monkeypatch.setattr(mod, "_TODAY_LOOKBACK_BARS", 90)    # < a session
    frame = _random_frame(21, n_days=7)
    sec = _secondary_for(21, frame)
    for kind, secondary in (("single", None), ("cross", sec)):
        new_fn, old_fn = _new_and_oracle(kind, frame, secondary)
        order = range(len(frame))
        assert _drive(new_fn, frame, order) == _drive(old_fn, frame, order), kind


def test_identical_with_nan_and_integer_ohlc_falls_back_exactly():
    """NaN cells make groupby first/last skip values, and integer OHLC keeps
    integer dtype through groupby: both must produce reference-identical
    output (the incremental path declines these inputs and defers to the
    reference builder)."""
    frame = _random_frame(31)
    nan_frame = frame.copy()
    nan_frame.iloc[600, nan_frame.columns.get_loc("High")] = np.nan
    nan_frame.iloc[601, nan_frame.columns.get_loc("Open")] = np.nan
    int_frame = (frame[["Open", "High", "Low", "Close"]] * 100).round().astype("int64")
    int_frame["Volume"] = 1000
    for f in (nan_frame, int_frame):
        new_fn, old_fn = _new_and_oracle("single", f)
        order = range(len(f))
        assert _drive(new_fn, f, order) == _drive(old_fn, f, order)


# ---------------------------------------------------------------------
# B. Unit equivalence of the two caches against the reference expressions
# ---------------------------------------------------------------------

def _require(name):
    if not hasattr(cs, name):
        raise AttributeError(f"cascade_signals.{name} does not exist")
    return getattr(cs, name)


@pytest.mark.parametrize("minutes", [5, 60])
@pytest.mark.parametrize("seed", SEEDS_FAST)
def test_today_buckets_primary_equals_reference_at_every_bar(minutes, seed):
    cache = _require("_TodayBuckets")(minutes)
    frame = _random_frame(seed)
    for i in range(len(frame)):
        f = frame.iloc[: i + 1]
        ts = f.index[-1]
        got = cache.complete_primary(f, ts.date(), ts)
        want = cs._build_complete_buckets(cs._today_frame(f, ts.date()), ts, minutes)
        assert len(got) == len(want)
        pd.testing.assert_frame_equal(got, want, check_exact=True)  # empty results too


@pytest.mark.parametrize("minutes", [5, 60])
def test_today_buckets_secondary_equals_reference_expression(minutes):
    """The secondary path must equal the REFERENCE divergence_check expression
    (full-history boolean mask + `.index.date` filter), evaluated at query
    times that are NOT the secondary's own last bar -- current_ts can sit past
    the secondary's final row (missing bars), which is what the permissive
    default in the divergence gate relies on."""
    cache = _require("_TodayBuckets")(minutes)
    primary = _random_frame(41)
    sec = _random_frame(1041, drop_prob=0.08)
    for i in range(0, len(primary), 3):
        ts = primary.index[i]
        today = ts.date()
        s = sec[sec.index <= ts]
        want = cs._build_complete_buckets(s[s.index.date == today], ts, minutes)
        got = cache.complete_secondary(sec, today, ts)
        assert len(got) == len(want)
        pd.testing.assert_frame_equal(got, want, check_exact=True)  # empty results too


def test_today_buckets_returns_reference_shape_index_name_and_dtypes():
    cache = _require("_TodayBuckets")(5)
    frame = _random_frame(51)
    f = frame.iloc[:150]
    ts = f.index[-1]
    got = cache.complete_primary(f, ts.date(), ts)
    want = cs._build_complete_buckets(cs._today_frame(f, ts.date()), ts, 5)
    assert list(got.columns) == list(want.columns) == ["Open", "High", "Low", "Close"]
    assert got.index.name == want.index.name
    assert str(got.index.dtype) == str(want.index.dtype)
    assert list(got.dtypes) == list(want.dtypes)


def test_today_buckets_reseeds_when_a_row_lands_in_an_already_complete_bucket():
    """Both real call sites guarantee the frame a query sees already contains
    every row <= its `current_ts`, so a bucket reported complete can never gain
    a row later. `_TodayBuckets` does not merely trust that: shown a frame that
    VIOLATES it (current_ts far past the last row, then a new row still inside
    the bucket that was reported complete), it must reseed and agree with the
    reference rather than keep serving the stale complete bucket."""
    cache = _require("_TodayBuckets")(60)
    day = "2026-03-04"
    ts = lambda hm: pd.Timestamp(f"{day} {hm}", tz=TZ)  # noqa: E731

    def frame_of(stamps_and_closes):
        idx = pd.DatetimeIndex([ts(t) for t, _ in stamps_and_closes])
        rows = [(c, c + 1.0, c - 1.0, c) for _, c in stamps_and_closes]
        return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])

    rows = [("09:30", 100.0), ("10:00", 101.0), ("10:15", 102.0), ("10:20", 250.0)]
    first = frame_of(rows[:3])
    late = ts("10:45")  # far past the last row: bucket 09:30-10:30 counts as complete
    got1 = cache._complete(first, 0, len(first), late)
    assert list(got1.index) == [ts("09:30")]
    assert got1["High"].iloc[0] == 103.0

    # A legal in-order append (10:20 >= 10:15) that still falls INSIDE the
    # bucket already reported complete (end 10:30 <= late). A cache that
    # trusted the earlier answer would keep serving High=103 / Close=102.
    second = frame_of(rows)
    want = cs._build_complete_buckets(second, late, 60)
    assert want["High"].iloc[0] == 251.0            # the reference DOES see the new row
    got2 = cache._complete(second, 0, len(second), late)
    pd.testing.assert_frame_equal(got2, want, check_exact=True)


# --- guard-pinning unit tests (each pins one cache guard: removing that guard
# makes the matching test fail) ----------------------------------------------

def _assert_primary_matches_reference(cache, frame, today, minutes, positions):
    for k in positions:
        f = frame.iloc[:k]
        ts = f.index[-1]
        got = cache.complete_primary(f, today, ts)
        want = cs._build_complete_buckets(cs._today_frame(f, today), ts, minutes)
        assert len(got) == len(want), k
        pd.testing.assert_frame_equal(got, want, check_exact=True)  # empty results too


def test_today_buckets_nan_cells_match_reference_first_last_skip_semantics():
    """groupby first/last skip NaN (a NaN Open on a bucket's first row makes the
    NEXT row's Open the bucket Open); a naive incremental fold would keep the
    NaN. The cache must decline NaN input and defer to the reference."""
    frame = _random_frame(51).iloc[:150].copy()
    starts = [cs._bucket_bounds(ts, 5)[0] for ts in frame.index]
    first_of_bucket = [i for i in range(1, len(frame)) if starts[i] != starts[i - 1]]
    i_open = first_of_bucket[3]
    # NaN Open on a bucket's FIRST row (reference: next row's Open wins) and NaN
    # High/Low/Close on INTERIOR rows. Deliberately NOT a NaN Close on a bucket's
    # last row: the cache's continuity check compares last_close with `==`, and
    # NaN != NaN forces a reseed exactly when that bucket completes, which would
    # mask the NaN-Open case this test exists to pin.
    frame.iloc[i_open, frame.columns.get_loc("Open")] = np.nan
    frame.iloc[i_open + 1, frame.columns.get_loc("High")] = np.nan
    frame.iloc[first_of_bucket[5] + 1, frame.columns.get_loc("Low")] = np.nan
    frame.iloc[first_of_bucket[6] + 1, frame.columns.get_loc("Close")] = np.nan
    cache = _require("_TodayBuckets")(5)
    today = frame.index[0].date()
    _assert_primary_matches_reference(cache, frame, today, 5, range(1, 151))
    ref = cs._build_complete_buckets(cs._today_frame(frame, today), frame.index[-1], 5)
    assert not ref["Open"].isna().any(), "reference must skip the NaN Open (premise of this test)"


def test_today_buckets_float32_ohlc_keeps_reference_dtype():
    """groupby on float32 stays float32; the incremental fold works in Python
    floats. The cache must decline non-float64 OHLC so dtype and rounding match."""
    frame = _random_frame(52).iloc[:150].copy()
    frame[["Open", "High", "Low", "Close"]] = frame[["Open", "High", "Low", "Close"]].astype("float32")
    cache = _require("_TodayBuckets")(5)
    _assert_primary_matches_reference(cache, frame, frame.index[0].date(), 5, range(20, 151, 7))
    got = cache.complete_primary(frame, frame.index[0].date(), frame.index[-1])
    assert str(got["Close"].dtype) == "float32"


def test_today_buckets_unsorted_index_defers_to_reference():
    frame = _random_frame(53, n_days=2)
    day1 = frame.index[0].date()
    n1 = int((frame.index.date == day1).sum())
    order = list(range(len(frame)))
    # one day-2 row lands BEFORE the last day-1 row: a non-ascending index whose
    # date boundaries a searchsorted-based slice would place wrongly.
    order[n1 - 1], order[n1] = order[n1], order[n1 - 1]
    f = frame.iloc[order]
    cache = _require("_TodayBuckets")(5)
    for today in (day1, frame.index[-1].date()):
        _assert_primary_matches_reference(cache, f, today, 5, range(n1 - 5, min(len(f), n1 + 30)))


def test_today_buckets_rows_of_other_days_never_leak_into_todays_buckets():
    """Query day 1's buckets from a frame whose LAST rows are day 2 (current_ts
    on day 2, so every day-1 bucket is complete). Only rows dated `today` may be
    aggregated -- an unbounded upper day bound would fold day-2 rows in."""
    frame = _random_frame(54, n_days=2)
    day1 = frame.index[0].date()
    cache = _require("_TodayBuckets")(60)
    f = frame.iloc[: len(frame) - 3]
    got = cache.complete_primary(f, day1, f.index[-1])
    want = cs._build_complete_buckets(cs._today_frame(f, day1), f.index[-1], 60)
    pd.testing.assert_frame_equal(got, want, check_exact=True)
    assert all(ts.date() == day1 for ts in got.index)


@pytest.mark.parametrize("minutes", [5, 60])
def test_today_buckets_primary_honors_the_trailing_today_cap(monkeypatch, minutes):
    """With a cap smaller than a session the reference sees only the LAST
    `_TODAY_LOOKBACK_BARS` rows of today (its partial-window buckets differ from
    full-day buckets). The cache must reproduce that exactly, not the full day."""
    monkeypatch.setattr(cs, "_TODAY_LOOKBACK_BARS", 90)
    cache = _require("_TodayBuckets")(minutes)
    frame = _random_frame(55)
    day = frame.index[len(frame) // 2].date()
    pos = [i for i, d in enumerate(frame.index.date) if d == day]
    _assert_primary_matches_reference(cache, frame, day, minutes, range(pos[0] + 1, pos[-1] + 2, 3))


def test_today_buckets_reseeds_when_shown_a_different_frame_with_the_same_length():
    """The cache trusts its running state only while the frame CONTINUES the one
    it consumed. A different frame (all prices scaled) shown after the first --
    whose row at the cache's last position differs in Close -- must reseed."""
    a = _random_frame(56).iloc[:120]
    b = a.copy()
    b[["Open", "High", "Low", "Close"]] = b[["Open", "High", "Low", "Close"]] * 1.01
    cache = _require("_TodayBuckets")(5)
    today = a.index[0].date()
    cache.complete_primary(a.iloc[:100], today, a.index[99])
    f = b.iloc[:110]
    got = cache.complete_primary(f, today, f.index[-1])
    want = cs._build_complete_buckets(cs._today_frame(f, today), f.index[-1], 5)
    pd.testing.assert_frame_equal(got, want, check_exact=True)


def test_today_buckets_backward_query_matches_reference():
    """A query on an EARLIER slice of the same day than one already answered
    (fewer rows) must not be served from the later state."""
    frame = _random_frame(58).iloc[:150]
    today = frame.index[0].date()
    cache = _require("_TodayBuckets")(5)
    _assert_primary_matches_reference(cache, frame, today, 5, [140, 60, 61, 149, 30])


def test_prior_day_context_backward_query_matches_reference(monkeypatch):
    """Same for the daily context. Constructed so the answer genuinely differs
    with position: with a 250-row daily tail, late in the day the window covers
    only part of the prior day (< 2 daily bars -> no bias), early in the day it
    covers a full prior day plus part of the one before (-> a bias)."""
    monkeypatch.setattr(cs, "_DAILY_LOOKBACK_BARS", 250)
    frame = _random_frame(59, n_days=4)
    today = frame.index[-1].date()
    rows_today = [i for i, d in enumerate(frame.index.date) if d == today]
    late, early = rows_today[0] + 190, rows_today[0] + 20

    def reference(k):
        f = frame.iloc[: k + 1]
        bars = cs._build_daily_bars(f, today)
        return (daily_bias(bars).iloc[-1] if len(bars) >= 2 else None,
                float(bars["High"].iloc[-1]) if len(bars) else None)

    assert reference(late) != reference(early), "test data must make position observable"
    ctx = _require("_PriorDayContext")()
    for k in (late, early):
        got = ctx.get(frame.iloc[: k + 1], today)
        assert (got.bias, got.prior_day_high) == reference(k), k


def test_prior_day_context_reseeds_when_shown_a_different_frame():
    a = _random_frame(57, n_days=5)
    b = a.copy()
    b[["Open", "High", "Low", "Close"]] = b[["Open", "High", "Low", "Close"]] * 1.01
    k = len(a) - 60
    today = a.index[k].date()
    ctx = _require("_PriorDayContext")()
    ctx.get(a.iloc[:k], today)
    got = ctx.get(b.iloc[: k + 5], today)
    bars = cs._build_daily_bars(b.iloc[: k + 5], today)
    want = (daily_bias(bars).iloc[-1] if len(bars) >= 2 else None, float(bars["High"].iloc[-1]))
    assert (got.bias, got.prior_day_high) == want


@pytest.mark.parametrize("seed", SEEDS_FAST)
def test_prior_day_context_equals_reference_at_every_bar(seed):
    ctx = _require("_PriorDayContext")()
    frame = _random_frame(seed, n_days=7)
    for i in range(len(frame)):
        f = frame.iloc[: i + 1]
        today = f.index[-1].date()
        bars = cs._build_daily_bars(f, today)
        want_bias = daily_bias(bars).iloc[-1] if len(bars) >= 2 else None
        want_high = float(bars["High"].iloc[-1]) if len(bars) else None
        got = ctx.get(f, today)
        assert (got.bias, got.prior_day_high) == (want_bias, want_high), i


def test_prior_day_context_recomputes_when_the_bound_changes(monkeypatch):
    """The cache is keyed on the trailing-window bound it was computed under,
    because the reference reads the module constant on every call."""
    ctx = _require("_PriorDayContext")()
    frame = _random_frame(63, n_days=8)
    f = frame.iloc[: len(frame) - 5]
    today = f.index[-1].date()

    def reference():
        bars = cs._build_daily_bars(f, today)
        return (daily_bias(bars).iloc[-1] if len(bars) >= 2 else None,
                float(bars["High"].iloc[-1]) if len(bars) else None)

    first = ctx.get(f, today)
    before = reference()
    assert (first.bias, first.prior_day_high) == before
    monkeypatch.setattr(cs, "_DAILY_LOOKBACK_BARS", 300)   # window now cuts into day t-1
    after = reference()
    assert after != before, "test data must make the bound change observable"
    again = ctx.get(f, today)
    assert (again.bias, again.prior_day_high) == after


def test_prior_day_context_tail_shift_guard(monkeypatch):
    """With a daily tail of only ~2.2 sessions the trailing window slides INTO
    the second-to-last prior day mid-session, changing what the reference
    computes. A cache without the slide guard would keep serving the stale
    value; this pins that it recomputes exactly when the reference changes."""
    monkeypatch.setattr(cs, "_DAILY_LOOKBACK_BARS", 470)
    ctx = _require("_PriorDayContext")()
    frame = _random_frame(63, n_days=8)
    changes = 0
    prev = None
    for i in range(len(frame)):
        f = frame.iloc[: i + 1]
        today = f.index[-1].date()
        bars = cs._build_daily_bars(f, today)
        want = (daily_bias(bars).iloc[-1] if len(bars) >= 2 else None,
                float(bars["High"].iloc[-1]) if len(bars) else None)
        got = ctx.get(f, today)
        assert (got.bias, got.prior_day_high) == want, i
        if prev is not None and want != prev and f.index[-1].date() == frame.index[i - 1].date():
            changes += 1                       # reference changed WITHIN a day
        prev = want
    assert changes >= 1, "test data never made the daily tail slide within a day"


# ---------------------------------------------------------------------
# C. Structure: per-day work is done once per day, not once per bar
# ---------------------------------------------------------------------

def _count_calls(monkeypatch, name):
    calls = {"n": 0}
    real = getattr(cs, name)

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(cs, name, spy)
    return calls


@pytest.mark.parametrize("kind", ["single", "cross"])
def test_daily_bars_built_once_per_day_not_once_per_bar(monkeypatch, kind):
    frame = _random_frame(71, n_days=6)
    sec = _secondary_for(71, frame) if kind == "cross" else None
    calls = _count_calls(monkeypatch, "_build_daily_bars")
    new_fn, _ = _new_and_oracle(kind, frame, sec)
    _drive(new_fn, frame, range(len(frame)))
    n_days = len(set(frame.index.date))
    assert calls["n"] <= n_days, (calls["n"], n_days, len(frame))


@pytest.mark.parametrize("kind", ["single", "cross"])
def test_reference_bucket_builder_not_used_on_sequential_hot_path(monkeypatch, kind):
    frame = _random_frame(72, n_days=6)
    sec = _secondary_for(72, frame) if kind == "cross" else None
    calls = _count_calls(monkeypatch, "_build_complete_buckets")
    new_fn, _ = _new_and_oracle(kind, frame, sec)
    _drive(new_fn, frame, range(len(frame)))
    assert calls["n"] == 0, calls["n"]


# ---------------------------------------------------------------------
# Notes on what no test here can observe, on purpose:
#   * `_TodayBuckets.complete_secondary` caps the secondary rows it folds at
#     current_ts (the min(hi, upto) cap). Dropping it would change no output:
#     rows later than current_ts can only land in buckets whose end is also
#     later than current_ts, and those are excluded from every result. The cap
#     is kept because it makes the cache causal BY CONSTRUCTION (it never
#     stores a row the reference could not see), not because a test can
#     observe it.
#   * `_plain_ohlc` does not re-check for an ascending index (see its
#     docstring): both entry points already refuse a non-ascending index.
# Two test-design traps worth knowing: a NaN Close on a bucket's LAST row makes
# the continuity check's `==` False (NaN != NaN), forcing a reseed exactly when
# that bucket completes -- which would hide a missing NaN guard; and seeds 11/12
# never enter a trade (hence the coverage guards above).
# ---------------------------------------------------------------------
