"""Tests for src/mining/spike_fade_study.py -- the CORE of the range-bound
spike-fade study (pre-registered design: sealed loader, data rules,
triggers, episodes, trade simulation, per-cell statistics and the
acceptance tests for cutoff independence, causality, data-quality
handling and counting).

Tests are hand-derived on small synthetic inputs, so
each expected value can be checked by hand. Synthetic data only -- no network,
no real prices.

Layout (one block per acceptance test):
  1. constants and the sealed loader
  2. data rules: valid-observation calendar, external cross-check, non-positive
     close, NaN-tolerant baselines, no forward-looking rule
  3. the three trigger families
  4. gap20 episodes and uniform censoring
  5. per-episode context
  6. the exit engine, hand-computed
  7. vectorized evaluation == reference
  8. cell statistics
  9. group evaluation, counts, determinism
 10. prefix invariance + power of that test
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import spike_fade_study as sfs


# ---------------------------------------------------------------------------
# Synthetic-data helpers (shared by every block)
# ---------------------------------------------------------------------------

def _bdays(n, start="2000-01-03"):
    return pd.bdate_range(start, periods=n)


def _series(closes, start="2000-01-03", name="SYN"):
    """A close Series on consecutive business days."""
    closes = np.asarray(closes, dtype=float)
    return pd.Series(closes, index=_bdays(len(closes), start), name=name)


def _walk(n, seed=0, vol=0.01, df=4.0, start_level=100.0):
    """Fat-tailed random-walk closes (Student t innovations), positive."""
    rng = np.random.default_rng(seed)
    scale = vol / math.sqrt(df / (df - 2.0))
    r = rng.standard_t(df, size=n) * scale
    return start_level * np.exp(np.cumsum(r))


# ---------------------------------------------------------------------------
# 1. Constants and the sealed loader
# ---------------------------------------------------------------------------

def test_constants_match_the_pre_registration():
    assert sfs.DISCOVERY_CUTOFF == pd.Timestamp("2023-06-30")
    assert sfs.TRIGGERS == ("z20@2.0", "z20@2.5", "pct756", "pctexp")
    assert sfs.Z_GRID == (2.0, 2.5)
    assert sfs.SIDES == ("up_fade", "down_fade")
    assert sfs.SIDE_SIGN == {"up_fade": +1, "down_fade": -1}
    assert sfs.ENTRY_RULES == ("E0", "E1")
    assert sfs.H_GRID == (10, 21, 42, 63)
    assert sfs.TP_GRID == ("none", "half", "full")
    assert sfs.TP_FRACTION == {"none": None, "half": 0.5, "full": 1.0}
    assert sfs.SL_GRID == (None, 1.5, 3.0)
    assert sfs.MIN_EPISODES == 8
    assert sfs.WINSOR_BOUND == 5.0
    assert sfs.GAP_SESSIONS == 20
    assert sfs.CENSOR_TAIL == 6 + 63 + 1
    assert sfs.CLUSTER_DAYS == 28
    assert sfs.Z_LOOKBACK == 20
    assert sfs.BASELINE_WINDOW == 252 and sfs.MIN_VALID_BASELINE == 200
    assert sfs.PCT_WINDOW == 756 and sfs.MIN_VALID_PCT == 500
    assert sfs.PCT_UP == 95.0 and sfs.PCT_DOWN == 5.0
    assert sfs.Z20_MIN_CLOSES == 273
    assert sfs.PCTEXP_MIN_CLOSES == 756
    assert sfs.SIGMA_WINDOW == 63 and sfs.SIGMA_HORIZON == 21
    assert sfs.SIGMA_FLOOR_PERCENTILE == 10.0
    assert sfs.MEDIAN_WINDOW == 252
    assert sfs.EXTERNAL_CHECK_TOLERANCE == 0.01 and sfs.LARGEST_RETURNS_LISTED == 20
    assert sfs.E1_CONFIRM_SESSIONS == 5
    assert sfs.PRIMARY_CELL == ("E0", 21, "none", None)


def test_cell_grid_is_the_72_variants_in_a_fixed_order():
    grid = sfs.CELL_GRID
    assert len(grid) == 72 == len(set(grid))
    assert grid[0] == ("E0", 10, "none", None)
    assert grid[-1] == ("E1", 63, "full", 3.0)
    assert sfs.PRIMARY_CELL in grid
    assert sfs.CELL_INDEX[sfs.PRIMARY_CELL] == grid.index(sfs.PRIMARY_CELL)
    # 288 cells per (series, side) = 4 triggers x 72 variants
    assert len(sfs.TRIGGERS) * len(grid) == 288


def test_load_sealed_truncates_at_the_cutoff_and_counts_the_rest():
    idx = pd.bdate_range("2023-06-26", periods=10)  # runs past 2023-06-30
    s = pd.Series(np.arange(10, dtype=float) + 100.0, index=idx)
    sealed = sfs.load_sealed(s, name="X", cutoff=pd.Timestamp("2023-06-30"))
    assert sealed.max_date == pd.Timestamp("2023-06-30")
    assert sealed.n_rows == 5                       # 26,27,28,29,30 June
    assert sealed.n_post_cutoff_dropped == 5
    assert list(sealed.close) == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert sealed.index.max() <= pd.Timestamp("2023-06-30")


def test_load_sealed_default_cutoff_is_the_discovery_cutoff():
    idx = pd.bdate_range("2023-06-20", periods=20)
    sealed = sfs.load_sealed(pd.Series(np.linspace(100, 110, 20), index=idx),
                             name="X")
    assert sealed.cutoff == sfs.DISCOVERY_CUTOFF
    assert sealed.max_date <= sfs.DISCOVERY_CUTOFF


def test_the_seal_assertion_fires_on_any_date_after_the_cutoff():
    """The loader ASSERTS the max date (not merely truncates): the guard is a
    separate callable so it can be proven to fire."""
    idx = pd.DatetimeIndex(["2023-06-29", "2023-06-30", "2023-07-03"])
    with pytest.raises(sfs.SealedWindowError):
        sfs.assert_sealed(idx, pd.Timestamp("2023-06-30"))
    # the exact cutoff date is allowed (inclusive)
    sfs.assert_sealed(idx[:2], pd.Timestamp("2023-06-30"))
    # SealedWindowError is an AssertionError so a bare `except AssertionError`
    # (and python -O aside, an assert-style contract) catches it
    assert issubclass(sfs.SealedWindowError, AssertionError)


def test_load_sealed_accepts_a_dataframe_with_a_close_column_and_tz_aware_index():
    idx = pd.bdate_range("2020-01-01", periods=5, tz="America/New_York")
    df = pd.DataFrame({"Open": 1.0, "Close": [10, 11, 12, 13, 14.0]}, index=idx)
    sealed = sfs.load_sealed(df, name="X")
    assert list(sealed.close) == [10, 11, 12, 13, 14.0]
    assert sealed.index.tz is None                  # dates, not timestamps


def test_load_sealed_rejects_unsorted_or_duplicate_dates():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError):
        sfs.load_sealed(pd.Series([1.0, 2.0, 3.0], index=idx), name="X")
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError):
        sfs.load_sealed(pd.Series([1.0, 2.0, 3.0], index=idx), name="X")


def test_load_sealed_first_date_drops_early_rows_and_counts_them():
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series(np.arange(10, dtype=float) + 1.0, index=idx)
    sealed = sfs.load_sealed(s, name="X", first_date=idx[3])
    assert sealed.index[0] == idx[3]
    assert sealed.n_pre_start_dropped == 3
    assert sealed.n_rows == 7


def test_the_sealed_series_is_independent_of_every_post_cutoff_value():
    """A series with post-cutoff spikes and the same
    series with the post-cutoff rows deleted outright load to the IDENTICAL
    sealed object -- nothing after the cutoff can influence anything built on
    it."""
    idx = pd.bdate_range("2023-05-01", periods=80)
    base = 100.0 + np.arange(80) * 0.1
    spiked = base.copy()
    spiked[idx > pd.Timestamp("2023-06-30")] *= 25.0    # post-cutoff spike
    a = sfs.load_sealed(pd.Series(spiked, index=idx), name="X")
    b = sfs.load_sealed(pd.Series(base[idx <= pd.Timestamp("2023-06-30")],
                                  index=idx[idx <= pd.Timestamp("2023-06-30")]),
                        name="X")
    assert a.index.equals(b.index)
    assert np.array_equal(a.close, b.close)
    assert a.n_post_cutoff_dropped == int((idx > pd.Timestamp("2023-06-30")).sum())
    assert b.n_post_cutoff_dropped == 0


# ---------------------------------------------------------------------------
# 2. Data rules
# ---------------------------------------------------------------------------

def test_lagged_window_stats_is_hand_computed_and_ends_at_t_minus_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    mean, sd, cnt = sfs.lagged_window_stats(x, window=4, min_valid=4)
    assert np.isnan(mean[:4]).all() and np.isnan(sd[:4]).all()
    # t = 4 uses x[0:4] = 1,2,3,4 (window ends at t-1, x[4] itself is excluded)
    assert mean[4] == pytest.approx(2.5)
    assert sd[4] == pytest.approx(math.sqrt(5.0 / 3.0))     # ddof = 1
    assert cnt[4] == 4
    assert mean[6] == pytest.approx(4.5)                    # 3,4,5,6


def test_lagged_window_stats_is_nan_tolerant_with_a_minimum_valid_count():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    m3, s3, c3 = sfs.lagged_window_stats(x, window=4, min_valid=3)
    assert c3[4] == 3
    assert m3[4] == pytest.approx((1 + 3 + 4) / 3.0)        # skips the NaN
    m4, s4, c4 = sfs.lagged_window_stats(x, window=4, min_valid=4)
    assert np.isnan(m4[4]) and np.isnan(s4[4])              # 3 valid < 4 required
    assert c4[4] == 3                                       # the count is still reported


def _prep(closes, name="SYN", start="2000-01-03", **kw):
    """Build a series with the study's own pipeline. `drop_missing` defaults to
    False HERE so tests that put NaN closes at known positions keep them; the
    study default (True: sessions are rows of the valid-observation calendar,
    missing-session drop rule) has its own tests below."""
    kw.setdefault("drop_missing", False)
    return sfs.build_series(_series(closes, start, name), name=name, **kw)


def test_prepare_series_on_clean_data_changes_nothing():
    s = _prep(_walk(400, seed=1))
    assert s.n == 400
    assert not s.external_corrected.any() and not s.nonpos.any()
    assert np.array_equal(s.p, s.raw)
    c = s.counts()
    assert c["external_corrections"] == 0 and c["nonpositive"] == 0 and c["missing"] == 0


def test_a_non_positive_close_is_flagged_kept_out_of_p_and_listed():
    closes = _walk(300, seed=2)
    closes[150] = -37.0
    closes[200] = 0.0
    s = _prep(closes)
    assert s.nonpos[150] and s.nonpos[200] and s.nonpos.sum() == 2
    assert np.isnan(s.p[150]) and np.isnan(s.p[200])
    assert s.raw[150] == -37.0                            # raw record is kept
    assert list(s.nonpos_dates) == [s.dates[150], s.dates[200]]
    assert s.counts()["nonpositive"] == 2
    assert s.n == 300                                     # a non-positive close is still a session


def test_a_missing_close_kept_in_the_calendar_is_counted_and_is_not_nonpositive():
    closes = _walk(300, seed=2)
    closes[120] = np.nan
    s = _prep(closes)                                     # drop_missing=False keeps the row
    assert s.missing[120] and not s.nonpos[120] and np.isnan(s.p[120])
    assert s.counts()["missing"] == 1 and s.n == 300


def test_sessions_are_rows_of_the_valid_observation_calendar_by_default():
    """Missing-session drop rule: 'P(t-20)' is the valid observation 20 ROWS back. Rows with no
    valid observation (FRED holiday gaps) are not sessions: the loader drops
    them by default, counts them, and r20 spans the surviving rows."""
    closes = _walk(400, seed=3)
    closes[[100, 101, 250]] = np.nan
    ser = _series(closes)
    s = sfs.build_series(ser, name="X")
    v = closes[np.isfinite(closes)]
    assert s.n == 397 and s.counts()["missing_rows_dropped"] == 3 and not s.missing.any()
    assert np.array_equal(s.raw, v)
    assert s.dates.equals(ser.index[np.isfinite(closes)])
    r20 = sfs.compute_triggers(s)["z20@2.5"].values["r20"]
    assert r20[300] == pytest.approx(math.log(v[300] / v[280]))        # 20 valid rows back
    kept = sfs.build_series(ser, name="X", drop_missing=False)
    assert kept.n == 400 and kept.counts()["missing_rows_dropped"] == 0
    r20k = sfs.compute_triggers(kept)["z20@2.5"].values["r20"]
    assert r20k[300] == pytest.approx(math.log(closes[300] / closes[280]))


def test_dropping_missing_rows_still_counts_what_the_seal_removed():
    idx = pd.bdate_range("2023-06-26", periods=10)
    vals = np.arange(10, dtype=float) + 100.0
    vals[1] = np.nan
    s = sfs.load_sealed(pd.Series(vals, index=idx), name="X")
    assert s.n_post_cutoff_dropped == 5 and s.n_missing_dropped == 1 and s.n_rows == 4


# --- no data rule may read a later value -----------------------------

def _spike_pair(seed=12, n=500, t=350, jump=0.15):
    """Two series identical through session t, which carries a genuine spike.
    In A the next session gives the whole spike back; in B it does not."""
    base = _walk(n, seed=seed, vol=0.005)
    r = np.r_[0.0, np.diff(np.log(base))]
    r[t] = jump
    a, b = r.copy(), r.copy()
    a[t + 1] = -jump
    b[t + 1] = 0.001
    return (base[0] * np.exp(np.cumsum(a)), base[0] * np.exp(np.cumsum(b)))


def test_no_next_session_reversal_rule_sets_a_genuine_spike_missing():
    # the only data correction is the external same-day cross-check: a spike
    # that the next session gives back stays a valid close
    s = _prep(_spike_pair()[0])
    t = 350
    assert np.isfinite(s.p[t]) and s.p[t] == s.raw[t] and not s.missing.any()


def test_a_genuine_spike_that_fully_reverses_next_session_is_still_a_spike_and_the_same_episode():
    """Series A gives the whole +15 percent back on the next session,
    series B does not. Everything computed at or before the spike session is
    identical, the spike close survives, and both trigger the same episode."""
    a_close, b_close = _spike_pair()
    t = 350
    a, b = _prep(a_close), _prep(b_close)
    assert np.isfinite(a.p[t]) and a.p[t] == b.p[t] == a.raw[t]
    assert a.counts()["external_corrections"] == 0 and not a.missing.any()
    sa, sb = sfs.compute_triggers(a), sfs.compute_triggers(b)
    for trig in ("z20@2.0", "z20@2.5"):
        assert sa[trig].up[t] and sb[trig].up[t]                       # the spike fires
        assert _same(sa[trig].values["z"][:t + 1], sb[trig].values["z"][:t + 1])
        assert _same(sa[trig].up[:t + 1], sb[trig].up[:t + 1])
    ea = sfs.build_episode_set(a, sa["z20@2.5"], "up_fade")
    eb = sfs.build_episode_set(b, sb["z20@2.5"], "up_fade")
    upto = lambda e: e.t0_all[e.t0_all <= t].tolist()
    assert upto(ea) == upto(eb) and len(upto(ea)) >= 1
    anchor = upto(ea)[-1]
    assert anchor <= t and t - anchor < sfs.GAP_SESSIONS               # the spike belongs to that episode
    # the context the trade would use is the same too
    ca, cb = a.context(), b.context()
    assert _same(ca.sigma[:t + 1], cb.sigma[:t + 1]) and _same(ca.med[:t + 1], cb.med[:t + 1])


def test_no_data_rule_uses_a_value_after_the_close_it_judges():
    """Every per-session output of the data pipeline (cleaned closes and every
    mask) computed from closes[:k] equals the full-series output on the same
    sessions, for cutoffs that straddle a fully reversed spike, a non-positive
    close, a missing close and an external-reference mismatch."""
    base = _walk(600, seed=21, vol=0.008, start_level=110.0)
    r = np.r_[0.0, np.diff(np.log(base))]
    r[320], r[321] = 0.2, -0.2                                        # a spike that reverses
    closes = base[0] * np.exp(np.cumsum(r))
    ref = closes * (1 + np.random.default_rng(2).normal(0, 0.0004, len(closes)))
    closes[400] = -3.0
    closes[450] = np.nan
    closes[480] = closes[480] * 1.03                                  # disagrees with the reference
    ref[500] = np.nan
    ser, refser = _series(closes), _series(ref)
    kw = dict(name="J", drop_missing=False, reference=refser)
    full = sfs.build_series(ser, **kw)
    assert full.external_corrected[480] and full.nonpos[400] and full.missing[450]
    for k in (260, 300, 321, 322, 323, 400, 401, 450, 451, 480, 481, 500, 501, 599):
        part = sfs.build_series(ser.iloc[:k], cutoff=ser.index[k - 1], **kw)
        assert part.n == k
        for name in ("p", "nonpos", "missing", "external_corrected"):
            assert _same(getattr(part, name), getattr(full, name)[:k]), (k, name)


# --- the only correction is an external, same-day cross-check --------

def _jpy_like(n=300, seed=7):
    close = _walk(n, seed=seed, vol=0.004, start_level=110.0)
    ref = close * (1 + np.random.default_rng(1).normal(0.0, 0.0005, n))   # same-day reference
    return close, ref


def test_external_check_sets_a_close_that_disagrees_with_the_reference_missing_and_counts_it():
    close, ref = _jpy_like()
    close[200] = ref[200] * 1.03                       # 3 percent above the same-day reference
    close[210] = ref[210] * 0.97                       # 3 percent below
    close[220] = ref[220] * 1.005                      # within the 1 percent tolerance
    s = _prep(close, reference=_series(ref))
    assert s.external_corrected[200] and s.external_corrected[210]
    assert s.external_corrected.sum() == 2
    assert np.isnan(s.p[200]) and np.isnan(s.p[210]) and np.isfinite(s.p[220])
    assert s.raw[200] == close[200]                    # the raw record is kept
    assert s.counts()["external_corrections"] == 2
    assert list(s.external_dates) == [s.dates[200], s.dates[210]]
    assert s.n == 300                                  # the row stays in the calendar
    assert not s.missing[200] and not s.nonpos[200]


def test_external_check_tolerance_is_a_parameter_and_defaults_to_one_percent():
    assert sfs.EXTERNAL_CHECK_TOLERANCE == 0.01
    close, ref = _jpy_like()
    close[200] = ref[200] * 1.015
    assert _prep(close, reference=_series(ref)).external_corrected[200]
    assert not _prep(close, reference=_series(ref), reference_tolerance=0.02).external_corrected[200]


def test_a_session_with_no_reference_value_is_left_alone_and_counted_as_unchecked():
    close, ref = _jpy_like()
    close[220] = ref[220] * 1.10                       # wildly off, but the reference has no value
    ref[220] = np.nan
    refser = _series(ref).drop(_series(ref).index[230])   # and no row at all for another date
    s = _prep(close, reference=refser)
    assert not s.external_corrected.any() and np.isfinite(s.p[220])
    assert s.counts()["reference_unchecked"] == 2


def test_no_reference_means_no_external_correction():
    close, _ref = _jpy_like()
    close[200] = close[200] * 1.5
    s = _prep(close)
    assert not s.external_corrected.any() and s.counts()["reference_unchecked"] == 0


def test_reference_values_after_the_cutoff_are_ignored():
    close, ref = _jpy_like()
    ser = _series(close)
    cut = ser.index[249]
    ref_bad = ref.copy()
    ref_bad[250:] = ref_bad[250:] * 3.0                # garbage AFTER the cutoff
    a = sfs.build_series(ser, name="J", cutoff=cut, reference=_series(ref))
    b = sfs.build_series(ser, name="J", cutoff=cut, reference=_series(ref_bad))
    assert a.n == b.n == 250
    assert _same(a.p, b.p) and not b.external_corrected.any()


def test_largest_absolute_one_session_log_returns_are_listed_for_inspection():
    s = _prep([100, 110, 99, 100, 130, 104])
    df = sfs.largest_abs_returns(s, k=3)
    assert list(df.columns) == ["date", "prev_date", "log_return", "abs_log_return"]
    assert df["date"].tolist() == [s.dates[4], s.dates[5], s.dates[2]]
    assert df["log_return"].iloc[0] == pytest.approx(math.log(1.3))
    assert df["log_return"].iloc[1] == pytest.approx(math.log(104 / 130))
    assert (df["abs_log_return"].diff().dropna() <= 0).all()
    assert len(sfs.largest_abs_returns(_prep(_walk(400, seed=4)))) == sfs.LARGEST_RETURNS_LISTED == 20
    # invalid closes give no return: a non-positive close removes the two returns around it
    z = _prep([100, 101, -5, 103, 104])
    assert set(sfs.largest_abs_returns(z, k=20)["date"]) == {z.dates[1], z.dates[4]}


def test_prepare_series_reasserts_the_seal():
    idx = pd.bdate_range("2023-06-20", periods=20)
    sealed = sfs.SealedSeries(name="X", index=idx, close=np.full(20, 100.0),
                              cutoff=pd.Timestamp("2023-06-30"),
                              n_post_cutoff_dropped=0, n_pre_start_dropped=0)
    with pytest.raises(sfs.SealedWindowError):
        sfs.prepare_series(sealed)                      # hand-built, unsealed


# ---------------------------------------------------------------------------
# 3. The three trigger families
# ---------------------------------------------------------------------------

def _oracle_z20(p, Z):
    """Rule-literal per-session loop, deliberately unlike the vectorized code.
    Returns (z, up, down). p has NaN for invalid closes."""
    n = len(p)
    r20 = np.full(n, np.nan)
    for t in range(20, n):
        if np.isfinite(p[t]) and np.isfinite(p[t - 20]):
            r20[t] = math.log(p[t] / p[t - 20])
    z = np.full(n, np.nan)
    up = np.zeros(n, bool)
    down = np.zeros(n, bool)
    for t in range(272, n):                           # needs 273 closes
        w = r20[t - 252:t]                            # the 252 sessions ending t-1
        w = w[np.isfinite(w)]
        if len(w) < 200 or not np.isfinite(r20[t]):
            continue
        sd = w.std(ddof=1)
        if sd == 0:
            continue
        z[t] = (r20[t] - w.mean()) / sd
        last20 = p[t - 19:t + 1]
        if z[t] >= Z and p[t] == np.nanmax(last20):
            up[t] = True
        if z[t] <= -Z and p[t] == np.nanmin(last20):
            down[t] = True
    return z, up, down


def _oracle_pct(p, expanding):
    n = len(p)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    up = np.zeros(n, bool)
    down = np.zeros(n, bool)
    for t in range(756, n):
        w = p[:t] if expanding else p[t - 756:t]
        w = w[np.isfinite(w)]
        if expanding:
            if len(w) < 756:
                continue
        elif len(w) < 500:
            continue
        hi[t] = np.percentile(w, 95, method="linear")
        lo[t] = np.percentile(w, 5, method="linear")
        if np.isfinite(p[t]):
            up[t] = p[t] >= hi[t]
            down[t] = p[t] <= lo[t]
    return hi, lo, up, down


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_z20_matches_a_naive_spec_oracle_on_random_walks(seed):
    s = _prep(_walk(900, seed=seed, vol=0.012))
    sigs = sfs.compute_triggers(s)
    z_ref = None
    for trig, Z in (("z20@2.0", 2.0), ("z20@2.5", 2.5)):
        z_ref, up_ref, down_ref = _oracle_z20(s.p, Z)
        got = sigs[trig]
        assert _same(got.up, up_ref) and _same(got.down, down_ref)
        assert np.allclose(got.values["z"], z_ref, rtol=0, atol=1e-10, equal_nan=True)
        assert got.up.any() or got.down.any()        # the comparison is not vacuous


def test_z20_2_0_flags_are_a_superset_of_z20_2_5_flags():
    s = _prep(_walk(1200, seed=7, vol=0.012))
    sigs = sfs.compute_triggers(s)
    assert (sigs["z20@2.0"].up | ~sigs["z20@2.5"].up).all()
    assert (sigs["z20@2.0"].down | ~sigs["z20@2.5"].down).all()
    assert sigs["z20@2.0"].up.sum() > sigs["z20@2.5"].up.sum()


def test_z20_first_usable_session_is_the_273rd_close():
    s = _prep(_walk(400, seed=4))
    sig = sfs.compute_triggers(s)["z20@2.5"]
    assert not sig.defined[:272].any()
    assert sig.defined[272]                     # position 272 = the 273rd close
    assert not sig.up[:272].any() and not sig.down[:272].any()


def test_z20_is_symmetric_under_price_inversion():
    """Down flags of 1/P are the up flags of P (z changes sign, max <-> min)."""
    p = _walk(1200, seed=9, vol=0.012)
    a = sfs.compute_triggers(_prep(p))["z20@2.0"]
    b = sfs.compute_triggers(_prep(1.0 / p))["z20@2.0"]
    assert _same(a.up, b.down) and _same(a.down, b.up)
    assert a.up.any()


def _sine_with_crash(n=420, crash_at=280, width=3, crash_level=90.0):
    """Quiet sine plus a crash that lasts `width` sessions (three, so the
    roll-off arithmetic in the hand case below is easy to follow)."""
    t = np.arange(n)
    p = 100.0 * (1 + 0.002 * np.sin(0.9 * t))
    p[crash_at:crash_at + width] = crash_level
    return p


def _ou(n, seed, phi=0.985, vol=0.012, df=4.0):
    """Mean-reverting fat-tailed log-price, so percentile triggers fire often
    enough that an oracle comparison cannot pass vacuously."""
    rng = np.random.default_rng(seed)
    scale = vol / math.sqrt(df / (df - 2.0))
    e = rng.standard_t(df, size=n) * scale
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return 100.0 * np.exp(x)


def test_z20_crash_is_a_down_flag_and_its_rolloff_twenty_sessions_later_is_not_an_up_flag():
    """Hand case. A quiet sine (r20 sd about 0.003) with a crash to 90 lasting
    sessions 280, 281, 282. r20(280) = ln(90 / P[260]) = about -0.105: z far
    below -2.5 and P[280] is the 20-session minimum -> a genuine down flag
    (281 and 282 follow, ties with the running minimum count). Twenty
    sessions later r20(300..302) = ln(P / 90) = about +0.105 is again a huge
    z, but only because the crash DROPPED OUT of the r20 anchor: those closes
    are not 20-session maxima, so the fresh-extreme rule refuses the up flag
    (fresh-extreme rule of the z20 trigger)."""
    s = _prep(_sine_with_crash())
    sig = sfs.compute_triggers(s)["z20@2.5"]
    assert np.flatnonzero(sig.down).tolist() == [280, 281, 282]
    assert not sig.up.any()
    assert sig.raw_up[300:303].all() and not sig.fresh_up[300:303].any()
    assert sig.values["z"][300] > 2.5
    assert sig.values["z"][280] < -2.5
    d = sfs.trigger_diagnostics(s, sig)
    assert d["up_fade"]["n_raw"] == 3 and d["up_fade"]["n_flagged"] == 0
    assert d["up_fade"]["rolloff_share"] == pytest.approx(1.0)
    assert d["down_fade"]["n_raw"] == 3 and d["down_fade"]["n_flagged"] == 3
    assert d["down_fade"]["rolloff_share"] == pytest.approx(0.0)


def test_z20_baseline_excludes_the_signal_session_itself():
    """The 252-session baseline ends at t-1: the size of r20(t) cannot dilute
    its own threshold. Compare z at the crash with a baseline recomputed by
    hand over r20[t-252 .. t-1]."""
    s = _prep(_sine_with_crash())
    p = s.p
    t = 280
    r20 = np.full(len(p), np.nan)
    r20[20:] = np.log(p[20:] / p[:-20])
    w = r20[t - 252:t]
    z_hand = (r20[t] - w.mean()) / w.std(ddof=1)
    assert sfs.compute_triggers(s)["z20@2.5"].values["z"][t] == pytest.approx(z_hand, abs=1e-10)


def test_z20_value_at_t_ignores_data_after_t():
    """Corrupt-the-future direct test (a prefix-invariance test cannot
    see a same-session peek). z and both flags at every checked t are
    unchanged when every close after t is replaced by garbage."""
    p = _walk(700, seed=13, vol=0.012)
    base = sfs.compute_triggers(_prep(p))["z20@2.0"]
    rng = np.random.default_rng(0)
    for t in (272, 300, 411, 520, 690):
        q = p.copy()
        q[t + 1:] = q[t] * np.exp(np.cumsum(rng.normal(0, 0.2, size=len(q) - t - 1)))
        got = sfs.compute_triggers(_prep(q))["z20@2.0"]
        assert _same(base.values["z"][:t + 1], got.values["z"][:t + 1])
        assert _same(base.up[:t + 1], got.up[:t + 1])
        assert _same(base.down[:t + 1], got.down[:t + 1])


def test_pct756_hand_computed_on_a_ramp():
    """p[i] = 100 + i for i < 756 (values 100..855). At t = 756 the window is
    exactly those 756 closes: numpy linear 95th percentile = value at virtual
    index 0.95 * 755 = 717.25 -> 817.25; 5th = index 37.75 -> 137.75."""
    p = 100.0 + np.arange(900.0)
    sig = sfs.compute_triggers(_prep(p))["pct756"]
    assert not sig.defined[:756].any() and sig.defined[756]
    assert sig.values["hi"][756] == pytest.approx(817.25)
    assert sig.values["lo"][756] == pytest.approx(137.75)
    assert sig.up[756] and not sig.down[756]             # 856 >= 817.25
    # at t = 800 the window is p[44..799] = 144..899: 95th = 144 + 717.25
    assert sig.values["hi"][800] == pytest.approx(861.25)


def test_pctexp_hand_computed_on_a_ramp_uses_history_through_t_minus_1():
    p = 100.0 + np.arange(900.0)
    sig = sfs.compute_triggers(_prep(p))["pctexp"]
    assert not sig.defined[:756].any() and sig.defined[756]
    assert sig.values["hi"][756] == pytest.approx(817.25)     # same 756 closes
    # at t = 800 the expanding window is all 800 closes 100..899
    assert sig.values["hi"][800] == pytest.approx(100.0 + 0.95 * 799)
    assert sig.values["lo"][800] == pytest.approx(100.0 + 0.05 * 799)


def test_pct_down_side_on_a_falling_ramp():
    p = 1000.0 - np.arange(900.0)
    sigs = sfs.compute_triggers(_prep(p))
    for name in ("pct756", "pctexp"):
        assert sigs[name].down[756] and not sigs[name].up[756]


@pytest.mark.parametrize("seed", [21, 22])
def test_pct_triggers_match_naive_oracles_on_mean_reverting_series(seed):
    s = _prep(_ou(1300, seed=seed))
    sigs = sfs.compute_triggers(s)
    for name, expanding in (("pct756", False), ("pctexp", True)):
        hi, lo, up, down = _oracle_pct(s.p, expanding)
        got = sigs[name]
        assert _same(got.values["hi"], hi) and _same(got.values["lo"], lo)
        assert _same(got.up, up) and _same(got.down, down)
        assert up.sum() + down.sum() > 10         # plenty of flags: not vacuous


def test_expanding_percentile_helper_equals_numpy_linear_exactly():
    rng = np.random.default_rng(5)
    x = rng.normal(size=400)
    x[[50, 51, 200]] = np.nan
    hi, lo = sfs.expanding_percentiles(x, (95.0, 5.0), min_count=30)
    for t in (30, 31, 100, 250, 399):
        w = x[:t][np.isfinite(x[:t])]
        if len(w) >= 30:
            assert hi[t] == np.percentile(w, 95, method="linear")
            assert lo[t] == np.percentile(w, 5, method="linear")
    assert np.isnan(hi[:30]).all()


def test_a_single_non_positive_close_does_not_blank_a_year_of_signals():
    """One non-positive close at session 400 removes r20 at the
    close itself (400) and at 420 (where it is the anchor) and NOTHING else:
    the baseline windows keep at least 250 valid values."""
    p = _walk(800, seed=31, vol=0.012)
    p[400] = -37.0
    s = _prep(p)
    for name in ("z20@2.0", "z20@2.5"):
        sig = sfs.compute_triggers(s)[name]
        undefined_after_warmup = np.flatnonzero(~sig.defined[272:]) + 272
        assert undefined_after_warmup.tolist() == [400, 420]
        assert sig.defined[421:652].all()          # the year that must not go blank
    sig = sfs.compute_triggers(s)["pct756"]
    assert sig.defined[756:].all()          # 400 < 756: the NaN only thins the window
    sig = sfs.compute_triggers(s)["pctexp"]
    # "at least 756 closes" counts VALID closes: 755 exist before session 756
    assert not sig.defined[756] and sig.defined[757:].all()


def test_the_200_valid_rule_boundary_is_exact():
    """Hand-derived boundary. Closes 300..359 are non-positive (60 of them).
    r20 is NaN at every session 300..379 (a NaN close kills r20 at itself
    and 20 sessions later) = 80 sessions. The baseline for t covers r20 over
    [t-252, t-1]; its NaN count is 632 - t for t in 380..631. It needs
    <= 52 NaN (>= 200 valid): t = 580 has 52 -> defined, t = 579 has 53 ->
    undefined."""
    p = _walk(700, seed=11, vol=0.012)
    p[300:360] = -1.0
    sig = sfs.compute_triggers(_prep(p))["z20@2.5"]
    assert not sig.defined[579]
    assert sig.defined[580] and sig.defined[581]
    assert not sig.defined[400] and not sig.defined[500]
    assert sig.defined[290]                                  # before the block


def test_a_missing_close_run_counts_as_undefined_not_as_a_signal():
    p = _walk(800, seed=41, vol=0.012)
    p[500:505] = np.nan
    s = _prep(p)
    sig = sfs.compute_triggers(s)["pct756"]
    assert not sig.up[500:505].any() and not sig.down[500:505].any()


def test_trigger_details_reports_first_usable_undefined_and_flags():
    s = _prep(_ou(1000, seed=51))
    sig = sfs.compute_triggers(s)["pct756"]
    d = sfs.trigger_diagnostics(s, sig)
    assert d["first_usable_index"] == 756
    assert d["first_usable_date"] == s.dates[756]
    assert d["n_sessions"] == 1000
    assert d["n_warmup"] == 756
    assert d["n_undefined"] == 756 + d["n_undefined_after_first_usable"]
    assert d["up_fade"]["n_flagged"] == int(sig.up.sum())
    assert d["up_fade"]["first_flag_date"] == s.dates[np.flatnonzero(sig.up)[0]]
    assert 0.0 <= d["up_fade"]["rolloff_share"] <= 1.0


# ---------------------------------------------------------------------------
# 4. gap20 episodes and uniform censoring
# ---------------------------------------------------------------------------

def _flags(n, positions):
    f = np.zeros(n, bool)
    f[list(positions)] = True
    return f


def test_extract_episodes_gap_is_measured_from_the_previous_flagged_day_not_the_anchor():
    """Hand case (ANY flagged day, not the previous episode anchor).
    Flags 10, 35, 36, 55, 56, 80: 10 starts an episode; 35 is 25 sessions
    after 10 -> new; 36 is 1 after 35 -> same; 55 is 19 after 36 -> SAME
    (an anchor-based rule would have started a new one, 55 - 35 = 20); 56 same;
    80 is 24 after 56 -> new."""
    t0 = sfs.extract_episodes(_flags(100, [10, 35, 36, 55, 56, 80]))
    assert t0.tolist() == [10, 35, 80]
    assert t0.dtype.kind == "i"


def test_extract_episodes_boundary_exactly_20_sessions_starts_a_new_episode():
    assert sfs.extract_episodes(_flags(50, [0, 20])).tolist() == [0, 20]
    assert sfs.extract_episodes(_flags(50, [0, 19])).tolist() == [0]
    assert sfs.extract_episodes(_flags(60, [0, 19, 38, 57])).tolist() == [0]  # chain 19,19,19


def test_extract_episodes_empty_and_all_flagged():
    assert sfs.extract_episodes(np.zeros(40, bool)).tolist() == []
    assert sfs.extract_episodes(np.ones(40, bool)).tolist() == [0]


def test_extract_episodes_is_anchored_on_the_first_flagged_day():
    t0 = sfs.extract_episodes(_flags(60, [5, 6, 7, 8]))
    assert t0.tolist() == [5]


def test_censor_episodes_keeps_t0_plus_70_on_or_before_the_last_close():
    """Uniform censoring: t0 + 6 + 63 + 1 sessions must exist. With
    100 closes the last position is 99: t0 = 29 needs position 99 (kept),
    t0 = 30 needs 100 (dropped)."""
    kept, dropped = sfs.censor_episodes(np.array([10, 29, 30, 60, 99]), 100)
    assert kept.tolist() == [10, 29]
    assert dropped == 3
    kept, dropped = sfs.censor_episodes(np.array([], dtype=int), 100)
    assert kept.tolist() == [] and dropped == 0


def test_episode_set_counts_flagged_days_episodes_and_censored_drops():
    """Episodes failing censoring are dropped and counted."""
    s = _prep(_sine_with_crash(n=340))                  # down flags at 280, 281, 282
    sig = sfs.compute_triggers(s)["z20@2.5"]
    es = sfs.build_episode_set(s, sig, "down_fade")
    assert es.n_flagged_days == 3 and es.t0_all.tolist() == [280]
    assert es.dropped_censored == 1 and es.t0.tolist() == []      # 280 + 70 = 350 > 339
    assert es.n_episodes == 1
    # the same crash with enough history after it survives censoring
    s2 = _prep(_sine_with_crash(n=420))
    es2 = sfs.build_episode_set(s2, sfs.compute_triggers(s2)["z20@2.5"], "down_fade")
    assert es2.t0.tolist() == [280] and es2.dropped_censored == 0
    # a crash before the 273rd close is never a signal at all
    s3 = _prep(_sine_with_crash(n=420, crash_at=200))
    es3 = sfs.build_episode_set(s3, sfs.compute_triggers(s3)["z20@2.5"], "down_fade")
    assert es3.n_flagged_days == 0 and es3.n_episodes == 0


def test_episode_set_carries_dates_and_side():
    s = _prep(_sine_with_crash(n=500))
    sig = sfs.compute_triggers(s)["z20@2.5"]
    es = sfs.build_episode_set(s, sig, "down_fade")
    assert es.t0.tolist() == [280] and es.side == "down_fade"
    assert es.trigger == "z20@2.5" and es.series == "SYN"
    assert list(es.dates) == [s.dates[280]]
    assert es.dropped_censored == 0


# ---------------------------------------------------------------------------
# 5. Per-episode context: median M and sigma21
# ---------------------------------------------------------------------------

def test_median_M_is_the_252_session_median_ending_at_t_inclusive():
    """p = 1..400. Window for t = 300 is positions 49..300 = values 50..301:
    median = (175 + 176) / 2 = 175.5."""
    s = _prep(np.arange(1.0, 401.0))
    ctx = sfs.series_context(s)
    assert ctx.med[300] == pytest.approx(175.5)
    assert ctx.med_status[300] == sfs.CTX_OK
    assert ctx.med_status[250] == sfs.CTX_UNDEFINED and np.isnan(ctx.med[250])
    assert ctx.med_status[251] == sfs.CTX_OK              # first full window
    assert ctx.med[251] == pytest.approx(126.5)           # values 1..252


def test_median_M_is_nan_tolerant_with_a_200_valid_minimum():
    p = np.arange(1.0, 501.0)
    p[300:352] = np.nan                                   # 52 missing in window of t = 400
    ctx = sfs.series_context(_prep(p))
    assert ctx.med_status[400] == sfs.CTX_OK              # 200 valid
    p = np.arange(1.0, 501.0)
    p[300:353] = np.nan                                   # 53 missing -> 199 valid
    ctx = sfs.series_context(_prep(p))
    assert ctx.med_status[400] == sfs.CTX_UNDEFINED


def test_nonpositive_close_touches_median_and_sigma_windows_exactly():
    """A non-positive close at q removes (status TOUCH_NONPOS) every session
    whose median window (252 closes ending t) or sigma window (63 returns =
    64 closes ending t) contains it, and nothing else."""
    p = _walk(800, seed=61, vol=0.012)
    q = 400
    p[q] = -37.0
    ctx = sfs.series_context(_prep(p))
    med_touch = np.flatnonzero(ctx.med_status == sfs.CTX_TOUCH_NONPOS)
    assert med_touch.tolist() == list(range(q, q + 252))
    sig_touch = np.flatnonzero(ctx.sigma_status == sfs.CTX_TOUCH_NONPOS)
    assert sig_touch.tolist() == list(range(q, q + 64))
    assert np.isnan(ctx.med[q]) and np.isnan(ctx.sigma[q + 10])
    assert ctx.med_status[q - 1] == sfs.CTX_OK and ctx.med_status[q + 252] == sfs.CTX_OK
    assert ctx.sigma_status[q + 64] == sfs.CTX_OK


def test_sigma21_is_63_return_sd_times_sqrt_21_floored_at_the_expanding_10th_percentile():
    s = _prep(_walk(600, seed=62, vol=0.012))
    ctx = sfs.series_context(s)
    p = s.p
    r = np.r_[np.nan, np.log(p[1:] / p[:-1])]
    t = 400
    raw = r[t - 62:t + 1].std(ddof=1) * math.sqrt(21)
    assert ctx.sigma_raw[t] == pytest.approx(raw, rel=1e-12)
    # floor: 10th percentile of every sigma_raw from the start through t (inclusive)
    hist = ctx.sigma_raw[:t + 1]
    hist = hist[np.isfinite(hist)]
    assert ctx.sigma_floor[t] == pytest.approx(np.percentile(hist, 10, method="linear"),
                                               rel=1e-12)
    assert ctx.sigma[t] == pytest.approx(max(raw, ctx.sigma_floor[t]))
    # window needs a full 63 returns: first computable session is 63
    assert np.isnan(ctx.sigma_raw[62]) and np.isfinite(ctx.sigma_raw[63])


def test_sigma_floor_binds_after_a_regime_change_to_very_low_volatility():
    """First 300 returns are noisy, then almost flat. Once sigma_raw falls
    below the causal expanding 10th percentile the floor takes over."""
    rng = np.random.default_rng(3)
    r = np.r_[rng.normal(0, 0.02, 300), rng.normal(0, 0.0005, 200)]
    p = 100.0 * np.exp(np.cumsum(r))
    ctx = sfs.series_context(_prep(p))
    t = 370
    assert ctx.sigma_raw[t] < ctx.sigma_floor[t]
    assert ctx.sigma[t] == ctx.sigma_floor[t]
    ok = ctx.sigma_status == sfs.CTX_OK
    assert np.allclose(ctx.sigma[ok], np.maximum(ctx.sigma_raw[ok], ctx.sigma_floor[ok]))


def test_sigma_needs_50_valid_returns_of_63():
    p = _walk(600, seed=63, vol=0.012)
    p[[300 + 2 * k for k in range(7)]] = np.nan          # 7 isolated NaN closes -> 14 NaN returns
    ctx = sfs.series_context(_prep(p))
    # window ending 320 covers returns 258..320 and contains all 14: 49 valid < 50
    assert ctx.sigma_status[320] == sfs.CTX_UNDEFINED
    p2 = _walk(600, seed=63, vol=0.012)
    p2[[300 + 2 * k for k in range(6)]] = np.nan         # 12 NaN returns -> 51 valid
    ctx2 = sfs.series_context(_prep(p2))
    assert ctx2.sigma_status[320] == sfs.CTX_OK


def test_series_context_is_memoized_on_the_series():
    s = _prep(_walk(400, seed=64))
    assert s.context() is s.context()


def test_context_at_t_ignores_data_after_t():
    p = _walk(700, seed=65, vol=0.012)
    base = sfs.series_context(_prep(p))
    rng = np.random.default_rng(1)
    for t in (300, 411, 520):
        q = p.copy()
        q[t + 1:] = q[t] * np.exp(np.cumsum(rng.normal(0, 0.2, size=len(q) - t - 1)))
        got = sfs.series_context(_prep(q))
        for name in ("med", "sigma_raw", "sigma_floor", "sigma", "med_status", "sigma_status"):
            assert _same(getattr(base, name)[:t + 1], getattr(got, name)[:t + 1]), name


# ---------------------------------------------------------------------------
# 6. The exit engine, hand-computed
# ---------------------------------------------------------------------------

def _vec_outcome(closes, *, t0, side, entry, H, tp, sl, M, sigma, nonpos, med_status,
                 sigma_status, extra_lag=0):
    """The SAME hand case pushed through the numpy engine: a SpikeSeries built
    directly from the closes plus a context whose median and sigma21 are the
    case's own numbers at every session."""
    p = np.asarray(closes, dtype=float)
    n = len(p)
    npos = np.zeros(n, bool) if nonpos is None else np.asarray(nonpos, bool)
    s = sfs.SpikeSeries(name="HAND", cutoff=sfs.DISCOVERY_CUTOFF,
                        dates=pd.bdate_range("2020-01-01", periods=n), raw=p.copy(), p=p,
                        nonpos=npos, missing=np.isnan(p) & ~npos, extra_lag=extra_lag)
    ctx = sfs.SeriesContext(
        med=np.full(n, M, dtype=float), med_status=np.full(n, med_status, dtype=np.int8),
        sigma_raw=np.full(n, sigma, dtype=float), sigma_floor=np.full(n, sigma, dtype=float),
        sigma=np.full(n, sigma, dtype=float), sigma_status=np.full(n, sigma_status, dtype=np.int8))
    return sfs.evaluate_variants(s, np.array([t0]), side, cells=[(entry, H, tp, sl)], context=ctx)


def _sim(closes, *, t0=0, side="up_fade", entry="E0", H=3, tp="none", sl=None,
         M=np.nan, sigma=0.05, nonpos=None, med_status=sfs.CTX_OK, sigma_status=sfs.CTX_OK,
         extra_lag=0):
    """Run one hand-computed case through BOTH engines (the sequential
    reference and the numpy evaluator), require them to agree exactly, and
    return the reference outcome for the test's own hand-computed assertions.
    Random-data equivalence cannot hit an exact float tie (a close exactly on
    a take-profit level); these hand cases can, so a strict-versus-inclusive
    slip in either engine is caught here."""
    p = np.asarray(closes, dtype=float)
    npos = np.zeros(len(p), bool) if nonpos is None else np.asarray(nonpos, bool)
    ref = sfs.simulate_episode(p, npos, t0, side, entry, H, tp, sl, M=M, med_status=med_status,
                               sigma=sigma, sigma_status=sigma_status, extra_lag=extra_lag)
    vec = _vec_outcome(closes, t0=t0, side=side, entry=entry, H=H, tp=tp, sl=sl, M=M,
                       sigma=sigma, nonpos=nonpos, med_status=med_status,
                       sigma_status=sigma_status, extra_lag=extra_lag)
    assert vec.status[0, 0] == ref.status, "numpy engine status differs from the reference"
    assert bool(vec.tp_na[0, 0]) == ref.tp_not_applicable
    if ref.status == sfs.STATUS_TRADED:
        assert vec.pnl[0, 0] == ref.pnl
        assert (vec.entry_idx[0, 0], vec.exit_idx[0, 0], vec.exit_reason[0, 0]) == \
            (ref.entry_idx, ref.exit_idx, ref.exit_reason)
    return ref


def test_time_stop_short_exits_at_entry_plus_H_with_no_extra_lag():
    """E0 enters at the close of t0 + 1 (position 1, close 100). H = 3: exit
    at the close of position 4 (97). Short: P&L = -(97/100 - 1) = +0.03."""
    o = _sim([100, 100, 99, 98, 97, 96], H=3)
    assert o.status == sfs.STATUS_TRADED
    assert (o.entry_idx, o.exit_idx, o.exit_reason) == (1, 4, sfs.EXIT_TIME)
    assert o.pnl == pytest.approx(0.03)
    assert not o.tp_not_applicable


def test_time_stop_long_pnl_sign():
    """down_fade is LONG: P&L = +(P_exit/P_entry - 1)."""
    o = _sim([100, 100, 101, 102, 104, 90], side="down_fade", H=3)
    assert o.pnl == pytest.approx(0.04)
    o = _sim([100, 100, 99, 98, 97, 96], side="down_fade", H=3)
    assert o.pnl == pytest.approx(-0.03)


def test_pnl_is_uncapped_a_naked_short_can_lose_more_than_100_percent():
    o = _sim([100, 100, 150, 210, 250, 300], H=3)
    assert o.pnl == pytest.approx(-1.5)


def test_take_profit_half_short_flags_at_k_and_fills_at_the_next_close():
    """M = 90, entry 100 -> D = +10; half = a gain of 5, i.e. a close <= 95.
    Entry at position 1. k = 1: 99 (no); k = 2: 94 (FLAG) -> fills at the
    close of k = 3 (position 4) = 96: P&L = -(96/100 - 1) = +0.04."""
    o = _sim([100, 100, 99, 94, 96, 97, 98], H=5, tp="half", M=90.0)
    assert o.status == sfs.STATUS_TRADED
    assert (o.exit_idx, o.exit_reason) == (4, sfs.EXIT_TP)
    assert o.pnl == pytest.approx(0.04)


def test_take_profit_full_short():
    """full = a gain of D = 10, i.e. a close <= 90. First at k = 4 (position 5,
    close 89); fills at k = 5 (position 6, close 92): P&L +0.08."""
    o = _sim([100, 100, 99, 94, 91, 89, 92, 93], H=6, tp="full", M=90.0)
    assert (o.exit_idx, o.exit_reason) == (6, sfs.EXIT_TP)
    assert o.pnl == pytest.approx(0.08)


def test_take_profit_long_side_half_and_full():
    """down_fade (long), M = 110, entry 100: D = -1 * (100 - 110) = +10.
    half = a close >= 105: first at k = 1 (exactly 105: the test is >=) so it
    fills at the k = 2 close 111 -> +0.11. full = >= 110: first at k = 2
    (111) -> fills at k = 3 (108) -> +0.08."""
    o = _sim([100, 100, 105, 111, 108, 107], side="down_fade", H=4, tp="half", M=110.0)
    assert (o.exit_idx, o.exit_reason) == (3, sfs.EXIT_TP)
    assert o.pnl == pytest.approx(0.11)
    o = _sim([100, 100, 105, 111, 108, 107], side="down_fade", H=4, tp="full", M=110.0)
    assert (o.exit_idx, o.exit_reason) == (4, sfs.EXIT_TP)
    assert o.pnl == pytest.approx(0.08)


def test_stop_loss_short_flags_on_the_close_and_fills_at_the_next_close():
    """sigma21 = 0.05, SL 1.5 -> flag when ln(P_k / 100) >= 0.075 (P >= 107.79).
    k = 2 close 108 (ln 1.08 = 0.0770) flags; fills at k = 3 close 110:
    P&L = -(110/100 - 1) = -0.10."""
    o = _sim([100, 100, 103, 108, 110, 111], H=5, sl=1.5)
    assert (o.exit_idx, o.exit_reason) == (4, sfs.EXIT_STOP)
    assert o.pnl == pytest.approx(-0.10)
    # SL 3.0 -> 0.15: the same path never flags; time stop at position 6
    o = _sim([100, 100, 103, 108, 110, 111, 116], H=5, sl=3.0)
    assert (o.exit_idx, o.exit_reason) == (6, sfs.EXIT_TIME)
    assert o.pnl == pytest.approx(-0.16)


def test_stop_loss_long_side_flags_on_a_fall():
    """down_fade: adverse = price falls; -ln(P_k / 100) >= 0.075 -> P <= 92.77.
    k = 2 close 92 flags; fills at close 90 -> P&L = -0.10 (long loses)."""
    o = _sim([100, 100, 97, 92, 90, 89], side="down_fade", H=5, sl=1.5)
    assert (o.exit_idx, o.exit_reason) == (4, sfs.EXIT_STOP)
    assert o.pnl == pytest.approx(-0.10)


def test_a_gap_through_the_stop_fills_at_the_next_close_not_at_the_level_or_the_gap_close():
    o = _sim([100, 100, 101, 120, 118, 117], H=4, sl=1.5)
    assert (o.exit_idx, o.exit_reason) == (4, sfs.EXIT_STOP)      # k = 2 flag, fill k = 3
    assert o.pnl == pytest.approx(-0.18)                          # 118, not 120, not 107.79


def test_a_flag_at_k_equal_H_resolves_to_the_time_stop():
    """H = 3: a stop-sized close at k = 3 is the time-stop close itself."""
    o = _sim([100, 100, 101, 102, 108, 109], H=3, sl=1.5)
    assert (o.exit_idx, o.exit_reason) == (4, sfs.EXIT_TIME)
    assert o.pnl == pytest.approx(-0.08)
    # the SAME close flagged one session earlier (k = 3 < H = 4) is a stop
    # fill, though it lands on the same close as the time stop would
    o = _sim([100, 100, 101, 102, 108, 109], H=4, sl=1.5)
    assert (o.exit_idx, o.exit_reason) == (5, sfs.EXIT_STOP)
    assert o.pnl == pytest.approx(-0.09)


def test_first_flag_wins_take_profit_before_stop_and_stop_before_take_profit():
    # short, M = 90: TP-half at k = 1 (close 94), a stop-sized spike at k = 3
    o = _sim([100, 100, 94, 96, 109, 110, 111], H=5, tp="half", sl=1.5, M=90.0)
    assert (o.exit_idx, o.exit_reason) == (3, sfs.EXIT_TP)
    # stop at k = 1 (close 108), a profit-sized close later
    o = _sim([100, 100, 108, 110, 90, 89, 88], H=5, tp="half", sl=1.5, M=90.0)
    assert (o.exit_idx, o.exit_reason) == (3, sfs.EXIT_STOP)


def test_same_close_tie_rule_stop_wins_at_the_resolver():
    """If two flags occur on the same close, the stop wins."""
    big = sfs.NO_FLAG
    assert sfs.resolve_first_flag(2, 2) == (2, sfs.EXIT_STOP)
    assert sfs.resolve_first_flag(2, 3) == (2, sfs.EXIT_TP)
    assert sfs.resolve_first_flag(3, 2) == (2, sfs.EXIT_STOP)
    assert sfs.resolve_first_flag(big, big) == (big, sfs.EXIT_TIME)
    assert sfs.resolve_first_flag(4, big) == (4, sfs.EXIT_TP)
    assert sfs.resolve_first_flag(big, 4) == (4, sfs.EXIT_STOP)


def test_tp_and_stop_flags_are_mutually_exclusive_on_one_close_under_the_spec_definitions():
    """Evidence for an open choice: TP needs a FAVOURABLE move of at least
    frac * D > 0, the stop an ADVERSE move of at least m * sigma21 > 0, so one
    close can never trigger both and the tie rule cannot bind. (The rule is
    still implemented, and tested at the resolver, as written.)"""
    rng = np.random.default_rng(0)
    for s in (+1, -1):
        pe = 100.0
        path = pe * np.exp(rng.normal(0, 0.3, size=20000))
        D = rng.uniform(0.5, 50.0, size=20000)
        frac = rng.choice([0.5, 1.0], size=20000)
        m = rng.choice([1.5, 3.0], size=20000)
        sigma = rng.uniform(0.001, 0.3, size=20000)
        tp_flag = s * (pe - path) >= frac * D
        sl_flag = s * np.log(path / pe) >= m * sigma
        assert not (tp_flag & sl_flag).any()
        assert tp_flag.any() and sl_flag.any()


def test_take_profit_not_applicable_when_D_is_not_positive_is_treated_as_none_and_counted():
    """Short with M = 110 above the entry 100: D = -10. Were it applied, a
    'half' target (gain >= -5) would flag at once. Instead the episode is
    treated as TP = none: time stop at position 5 (70): +0.30."""
    o = _sim([100, 100, 99, 90, 80, 70], H=4, tp="half", M=110.0)
    assert o.status == sfs.STATUS_TRADED and o.tp_not_applicable
    assert (o.exit_idx, o.exit_reason) == (5, sfs.EXIT_TIME)
    assert o.pnl == pytest.approx(0.30)
    o = _sim([100, 100, 99, 90, 80, 70], H=4, tp="full", M=100.0)   # D = 0 exactly
    assert o.tp_not_applicable and o.exit_reason == sfs.EXIT_TIME
    # TP = none is never 'not applicable'
    assert not _sim([100, 100, 99, 90, 80, 70], H=4, tp="none", M=110.0).tp_not_applicable
    # and an applicable TP is not counted
    assert not _sim([100, 100, 99, 94, 96, 97, 98], H=5, tp="half", M=90.0).tp_not_applicable


def test_E1_with_no_confirming_session_gives_no_trade():
    """Short (up_fade): E1 wants a close BELOW the previous close within
    (t0, t0 + 5]. A path that rises for all five sessions has none."""
    o = _sim([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111], entry="E1", H=3)
    assert o.status == sfs.STATUS_NO_E1_CONFIRM
    assert np.isnan(o.pnl) and o.entry_idx == -1


def test_E1_confirmation_at_the_first_session_enters_one_session_later():
    """u = 1: p[1] = 99 < p[0] = 100 (the signal close is the 'previous'
    close). Enter at the close of u + 1 = 2 (98); H = 3 -> exit position 5
    (95): P&L = -(95/98 - 1)."""
    o = _sim([100, 99, 98, 97, 96, 95, 94, 93], entry="E1", H=3)
    assert o.status == sfs.STATUS_TRADED and o.entry_idx == 2 and o.exit_idx == 5
    assert o.pnl == pytest.approx(-(95 / 98 - 1))


def test_E1_confirmation_at_t0_plus_5_enters_at_t0_plus_6():
    """Rising through position 4, then a lower close at u = 5: entry at
    position 6, the latest possible."""
    o = _sim([100, 101, 102, 103, 104, 103, 104, 105, 106, 107, 108, 109], entry="E1", H=3)
    assert o.status == sfs.STATUS_TRADED and o.entry_idx == 6 and o.exit_idx == 9
    assert o.pnl == pytest.approx(-(107 / 104 - 1))


def test_E1_a_confirming_close_only_at_t0_plus_6_is_too_late():
    o = _sim([100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99], entry="E1", H=3)
    assert o.status == sfs.STATUS_NO_E1_CONFIRM


def test_E1_long_side_wants_a_close_above_the_previous_close():
    """down_fade: falls at u = 1, 2 do not confirm; u = 3 (99 > 98) does ->
    entry at position 4."""
    o = _sim([100, 99, 98, 99, 100, 101, 102, 103, 104, 105], side="down_fade", entry="E1", H=3)
    assert o.status == sfs.STATUS_TRADED and o.entry_idx == 4 and o.exit_idx == 7


def test_E1_M_and_D_use_the_entry_close_not_t0():
    """D = s * (P_entry - M) with P_entry the E1 fill close (position 2, 98):
    short with M = 90: D = 8, half = gain of 4 = a close <= 94."""
    o = _sim([100, 99, 98, 96, 94, 93, 92, 91], entry="E1", H=5, tp="half", M=90.0)
    assert o.status == sfs.STATUS_TRADED and not o.tp_not_applicable
    # k = 1 (pos 3) 96: gain 2 no; k = 2 (pos 4) 94: gain 4 -> flag; fill pos 5 = 93
    assert (o.exit_idx, o.exit_reason) == (5, sfs.EXIT_TP)
    assert o.pnl == pytest.approx(-(93 / 98 - 1))


def test_a_non_positive_or_missing_entry_close_removes_the_cell():
    nan_entry = [100, np.nan, 99, 98, 97, 96]
    nonpos = [False, True, False, False, False, False]
    o = _sim(nan_entry, nonpos=nonpos, H=3)
    assert o.status == sfs.STATUS_TOUCH_NONPOS
    o = _sim(nan_entry, H=3)                                  # merely missing
    assert o.status == sfs.STATUS_TOUCH_MISSING


def test_time_only_cell_keeps_the_episode_when_its_path_crosses_a_non_positive_or_missing_close():
    """Non-positive closes are kept distinct. A time-only cell (TP none, SL none) uses only its entry and
    exit prices. Invalid close at position 6; entry at 1 (100), H = 8 exits at
    position 9 (92): the episode stays, P&L +0.08, whether the bad close is
    non-positive or merely missing."""
    p = [100, 100, 99, 98, 97, 96, np.nan, 94, 93, 92, 91, 90]
    npos = np.zeros(len(p), bool)
    npos[6] = True
    for mask in (npos, None):
        o = _sim(p, nonpos=mask, H=8)
        assert o.status == sfs.STATUS_TRADED and (o.entry_idx, o.exit_idx) == (1, 9)
        assert o.pnl == pytest.approx(0.08)
        o = _sim(p, nonpos=mask, H=3)                     # exits at 4, before the bad close
        assert o.status == sfs.STATUS_TRADED and o.pnl == pytest.approx(0.03)


def test_time_only_cell_is_removed_when_its_exit_price_is_invalid():
    p = [100, 100, 99, 98, 97, np.nan, 95]
    npos = [False, False, False, False, False, True, False]
    assert _sim(p, nonpos=npos, H=4).status == sfs.STATUS_TOUCH_NONPOS       # exit at position 5
    assert _sim(p, H=4).status == sfs.STATUS_TOUCH_MISSING
    o = _sim(p, nonpos=npos, H=5)                                            # exit at 6 (95), crosses 5
    assert o.status == sfs.STATUS_TRADED and o.pnl == pytest.approx(0.05)


def test_stop_cell_is_removed_when_any_close_on_its_entry_to_exit_path_is_invalid():
    """Stop-loss uses a log return at every close, so its whole entry-to-exit
    path must be valid. Multiple 3.0 (level 0.15) never flags here."""
    p = [100, 100, 99, 98, 97, 96, np.nan, 94, 93, 92, 91, 90]
    npos = np.zeros(len(p), bool)
    npos[6] = True
    assert _sim(p, nonpos=npos, H=8, sl=3.0).status == sfs.STATUS_TOUCH_NONPOS
    assert _sim(p, H=8, sl=3.0).status == sfs.STATUS_TOUCH_MISSING
    o = _sim(p, nonpos=npos, H=3, sl=3.0)                 # exits at 4 before the bad close
    assert o.status == sfs.STATUS_TRADED and o.pnl == pytest.approx(0.03)
    # an early stop exit (flag k = 1, fill at position 3) ahead of the bad close
    early = _sim([100, 100, 110, 111, 112, 113, np.nan, 90, 89], nonpos=npos[:9], H=8, sl=1.5)
    assert early.status == sfs.STATUS_TRADED and early.exit_idx == 3
    # the bad close IS the fill close of the flag -> the exit price is invalid
    fill_hit = _sim([100, 100, 110, np.nan, 112, 113], nonpos=[False, False, False, True, False, False],
                    H=4, sl=1.5)
    assert fill_hit.status == sfs.STATUS_TOUCH_NONPOS
    # a take-profit target on the same path does not rescue a stop cell
    both = _sim([100, 100, 99, np.nan, 97, 96, 95, 94], nonpos=[False, False, False, True] + [False] * 4,
                H=6, tp="half", sl=3.0, M=90.0)
    assert both.status == sfs.STATUS_TOUCH_NONPOS


def test_take_profit_cell_skips_an_invalid_close_and_needs_only_entry_exit_and_median():
    """A TP cell without a stop needs its median window and its entry
    and exit prices. A non-positive / missing close on the way cannot fire a flag
    (there is no usable price there) and does not remove the episode. Short, M = 90,
    half = a gain of 5: closes 99, [invalid], 97, 96 do not reach it; 95 does
    (k = 5, gain exactly 5) and the fill at k = 6 is 94: +0.06."""
    p = [100, 100, 99, np.nan, 97, 96, 95, 94]
    npos = [False, False, False, True, False, False, False, False]
    for mask in (npos, None):
        o = _sim(p, nonpos=mask, H=6, tp="half", M=90.0)
        assert o.status == sfs.STATUS_TRADED
        assert (o.exit_idx, o.exit_reason) == (7, sfs.EXIT_TP)
        assert o.pnl == pytest.approx(0.06)


def test_take_profit_cell_whose_fill_close_is_invalid_is_removed():
    p = [100, 100, 94, np.nan, 96, 97, 98, 99]           # flag at k = 1 (94), fill at position 3
    npos = [False, False, False, True, False, False, False, False]
    assert _sim(p, nonpos=npos, H=5, tp="half", M=90.0).status == sfs.STATUS_TOUCH_NONPOS
    assert _sim(p, H=5, tp="half", M=90.0).status == sfs.STATUS_TOUCH_MISSING


def test_extra_lag_a_stop_cell_also_needs_the_closes_between_its_flag_and_its_fill():
    """extra_lag = 1: the flag at k = 1 fills at k = 3, so position 4 (k = 2) is on
    the entry-to-exit path. A stop cell is removed by an invalid close there; a
    take-profit cell, which uses only its entry, exit and median, is not."""
    npos = [False, False, False, False, True, False, False, False]
    o = _sim([100, 100, 100, 108, np.nan, 110, 111, 112], nonpos=npos, H=5, sl=1.5, extra_lag=1)
    assert o.status == sfs.STATUS_TOUCH_NONPOS
    o = _sim([100, 100, 100, 94, np.nan, 97, 98, 99], nonpos=npos, H=5, tp="half", M=90.0, extra_lag=1)
    assert o.status == sfs.STATUS_TRADED and o.exit_idx == 5 and o.pnl == pytest.approx(0.03)


def test_a_path_that_runs_past_the_last_close_is_censored_unless_it_exits_first():
    p = [100, 100, 99, 98, 97]                       # last position 4
    assert _sim(p, H=5).status == sfs.STATUS_CENSORED
    assert _sim(p, H=3).status == sfs.STATUS_TRADED   # exit at position 4 = last close
    # an early stop exit inside the data is fine even though H reaches beyond it
    o = _sim([100, 100, 110, 111, 112], H=10, sl=1.5)
    assert o.status == sfs.STATUS_TRADED and o.exit_idx == 3
    # a flag whose FILL close would be beyond the data is censored
    assert _sim([100, 100, 110], H=10, sl=1.5).status == sfs.STATUS_CENSORED
    # E1 still scanning when the data ends
    assert _sim([100, 101, 102], entry="E1", H=3).status == sfs.STATUS_CENSORED
    # entry itself beyond the data
    assert _sim([100], H=3).status == sfs.STATUS_CENSORED


def test_TP_cells_need_a_usable_median_but_TP_none_cells_do_not():
    p = [100, 100, 99, 94, 96, 97, 98]
    o = _sim(p, H=5, tp="half", M=90.0, med_status=sfs.CTX_TOUCH_NONPOS)
    assert o.status == sfs.STATUS_M_TOUCH_NONPOS
    o = _sim(p, H=5, tp="half", M=np.nan, med_status=sfs.CTX_UNDEFINED)
    assert o.status == sfs.STATUS_UNDEFINED_M
    o = _sim(p, H=5, tp="none", M=np.nan, med_status=sfs.CTX_TOUCH_NONPOS)
    assert o.status == sfs.STATUS_TRADED             # the median window is not this cell's


def test_sigma_problems_remove_every_cell_of_the_episode():
    p = [100, 100, 99, 98, 97, 96]
    assert _sim(p, sigma=np.nan, sigma_status=sfs.CTX_UNDEFINED).status == sfs.STATUS_UNDEFINED_SIGMA
    assert _sim(p, sigma=np.nan, sigma_status=sfs.CTX_TOUCH_NONPOS).status == sfs.STATUS_SIGMA_TOUCH_NONPOS
    # sigma is checked first: it wins over an E1 that would have had no trade
    o = _sim([100, 101, 102, 103, 104, 105, 106], entry="E1", sigma=np.nan,
             sigma_status=sfs.CTX_UNDEFINED)
    assert o.status == sfs.STATUS_UNDEFINED_SIGMA


def test_x_is_pnl_over_sigma21_winsorised_at_plus_and_minus_5():
    assert sfs.winsorise(np.array([-9.0, -5.0, 0.3, 4.99, 5.0, 12.0])).tolist() == \
        [-5.0, -5.0, 0.3, 4.99, 5.0, 5.0]
    o = _sim([100, 100, 99, 98, 97, 96], H=3, sigma=0.01)      # pnl 0.03 / 0.01 = 3
    assert sfs.episode_x(o, 0.01) == pytest.approx(3.0)
    o = _sim([100, 100, 99, 98, 97, 96], H=3, sigma=0.001)     # 30 -> capped at 5
    assert sfs.episode_x(o, 0.001) == pytest.approx(5.0)
    o = _sim([100, 101, 102, 103, 104, 105, 106], entry="E1")
    assert np.isnan(sfs.episode_x(o, 0.01))


def test_simulate_from_series_reads_the_series_context_at_t0():
    s = _prep(_walk(500, seed=71, vol=0.012))
    ctx = s.context()
    t0 = 300
    cell = ("E0", 21, "half", 1.5)
    a = sfs.simulate_from_series(s, t0, "up_fade", cell)
    b = sfs.simulate_episode(s.p, s.nonpos, t0, "up_fade", "E0", 21, "half", 1.5,
                             M=ctx.med[t0], med_status=int(ctx.med_status[t0]),
                             sigma=ctx.sigma[t0], sigma_status=int(ctx.sigma_status[t0]))
    assert a == b
    assert a.status == sfs.STATUS_TRADED


# ---------------------------------------------------------------------------
# 7. Vectorized evaluation of all 72 variants == the reference simulator
# ---------------------------------------------------------------------------

def _messy_series(seed, n=700, vol=0.02):
    """A random walk with a non-positive close, a couple of missing closes and
    a run of NaN, so every status code has a chance to occur."""
    p = _walk(n, seed=seed, vol=vol)
    rng = np.random.default_rng(seed)
    p[int(rng.integers(300, 400))] = -5.0
    for q in rng.integers(300, n - 10, size=3):
        p[int(q)] = np.nan
    return p


def _assert_results_match_reference(series, t0, side, res):
    """Every (cell, episode) of `res` equals the sequential simulator."""
    for ci, cell in enumerate(res.cells):
        for j, t in enumerate(t0):
            o = sfs.simulate_from_series(series, int(t), side, cell)
            assert res.status[ci, j] == o.status, (cell, int(t))
            assert res.tp_na[ci, j] == o.tp_not_applicable, (cell, int(t))
            if o.status == sfs.STATUS_TRADED:
                assert res.pnl[ci, j] == o.pnl, (cell, int(t))
                assert res.entry_idx[ci, j] == o.entry_idx
                assert res.exit_idx[ci, j] == o.exit_idx
                assert res.exit_reason[ci, j] == o.exit_reason
            else:
                assert np.isnan(res.pnl[ci, j]) and np.isnan(res.x[ci, j])
                assert res.entry_idx[ci, j] == -1 and res.exit_idx[ci, j] == -1


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("side", ["up_fade", "down_fade"])
def test_evaluate_variants_equals_the_reference_on_messy_random_data(seed, side):
    p = _messy_series(seed)
    s = _prep(p)
    rng = np.random.default_rng(100 + seed)
    # anchors everywhere, including the last 80 sessions (censoring) and the
    # windows around the non-positive close (touch statuses)
    t0 = np.unique(np.r_[rng.integers(280, len(p) - 1, size=45),
                         np.arange(len(p) - 75, len(p) - 60), 305, 350, 399])
    res = sfs.evaluate_variants(s, t0, side)
    assert res.pnl.shape == (72, len(t0))
    assert res.cells == sfs.CELL_GRID
    _assert_results_match_reference(s, t0, side, res)
    # the comparison is not vacuous: many statuses and many traded cells occur
    assert (res.status == sfs.STATUS_TRADED).sum() > 500
    assert len(np.unique(res.status)) >= 4


def test_evaluate_variants_exercises_every_exit_reason_and_tp_not_applicable():
    s = _prep(_walk(900, seed=8, vol=0.02))
    t0 = np.arange(300, 820, 7)
    res = sfs.evaluate_variants(s, t0, "up_fade")
    traded = res.status == sfs.STATUS_TRADED
    for reason in (sfs.EXIT_TIME, sfs.EXIT_TP, sfs.EXIT_STOP):
        assert (res.exit_reason[traded] == reason).any(), reason
    assert res.tp_na.any() and (~res.tp_na).any()
    _assert_results_match_reference(s, t0[::5], "up_fade", sfs.evaluate_variants(s, t0[::5], "up_fade"))


def test_evaluate_variants_accepts_a_vehicle_sign_and_matches_the_reference():
    s = _prep(_walk(800, seed=9, vol=0.02))
    t0 = np.arange(300, 700, 23)
    res = sfs.evaluate_variants(s, t0, -1)                  # long the series
    res2 = sfs.evaluate_variants(s, t0, "down_fade")
    assert np.array_equal(res.status, res2.status)
    assert np.array_equal(res.pnl, res2.pnl, equal_nan=True)
    _assert_results_match_reference(s, t0, -1, res)


def test_evaluate_variants_subset_of_cells_equals_the_matching_rows_of_the_full_grid():
    s = _prep(_walk(800, seed=10, vol=0.02))
    t0 = np.arange(300, 700, 11)
    full = sfs.evaluate_variants(s, t0, "up_fade")
    one = sfs.evaluate_variants(s, t0, "up_fade", cells=[sfs.PRIMARY_CELL])
    i = sfs.CELL_INDEX[sfs.PRIMARY_CELL]
    assert one.pnl.shape == (1, len(t0)) and one.cells == (sfs.PRIMARY_CELL,)
    assert np.array_equal(one.pnl[0], full.pnl[i], equal_nan=True)
    assert np.array_equal(one.status[0], full.status[i])
    picks = [("E1", 42, "full", 3.0), ("E0", 10, "half", None), ("E1", 21, "none", 1.5)]
    part = sfs.evaluate_variants(s, t0, "up_fade", cells=picks)
    for r, c in enumerate(picks):
        assert np.array_equal(part.pnl[r], full.pnl[sfs.CELL_INDEX[c]], equal_nan=True)


def test_evaluate_variants_is_chunk_size_independent(monkeypatch):
    s = _prep(_walk(800, seed=11, vol=0.02))
    t0 = np.arange(290, 720, 3)
    big = sfs.evaluate_variants(s, t0, "up_fade")
    monkeypatch.setattr(sfs, "_EVAL_CHUNK", 7)
    small = sfs.evaluate_variants(s, t0, "up_fade")
    for name in ("status", "pnl", "x", "entry_idx", "exit_idx", "exit_reason", "tp_na"):
        assert np.array_equal(getattr(big, name), getattr(small, name), equal_nan=True), name


def test_evaluate_variants_with_no_episodes_returns_empty_arrays():
    s = _prep(_walk(500, seed=12))
    res = sfs.evaluate_variants(s, np.array([], dtype=int), "up_fade")
    assert res.pnl.shape == (72, 0) and res.status.shape == (72, 0)


def test_x_is_winsorised_pnl_over_sigma21_and_only_for_traded_cells():
    s = _prep(_walk(800, seed=13, vol=0.02))
    ctx = s.context()
    t0 = np.arange(300, 700, 13)
    res = sfs.evaluate_variants(s, t0, "up_fade")
    traded = res.status == sfs.STATUS_TRADED
    want = np.clip(res.pnl / ctx.sigma[t0][None, :], -5, 5)
    assert np.array_equal(res.x[traded], want[traded])
    assert np.isnan(res.x[~traded]).all()
    assert np.abs(res.x[traded]).max() <= 5.0
    assert np.array_equal(res.sigma, ctx.sigma[t0], equal_nan=True)


def test_status_counts_report_every_removed_observation():
    p = _messy_series(5)
    s = _prep(p)
    t0 = np.unique(np.r_[np.arange(280, 400, 3), np.arange(len(p) - 75, len(p) - 60)])
    res = sfs.evaluate_variants(s, t0, "up_fade")
    c = sfs.cell_status_counts(res, sfs.PRIMARY_CELL)
    assert sum(c[name] for name in sfs.STATUS_NAMES.values()) == len(t0)
    assert c["traded"] == int((res.status[sfs.CELL_INDEX[sfs.PRIMARY_CELL]] == 0).sum())
    assert "tp_not_applicable" in c
    # cells that need the median list their own removals separately
    tp_cell = ("E0", 21, "half", None)
    c2 = sfs.cell_status_counts(res, tp_cell)
    assert sum(c2[name] for name in sfs.STATUS_NAMES.values()) == len(t0)


def test_the_non_positive_close_removes_only_the_cells_that_use_it():
    """End to end: a non-positive close at position 400. Anchored at
    380 (E0 entry 381) a time-only cell keeps the episode even though its path
    crosses 400 (H = 10 exits at 391, H = 21 at 402); a stop cell whose path
    reaches 400 loses it; anchored at 378 the H = 21 exit IS the bad close and
    the cell is removed; anchored at 399 the entry is the bad close."""
    p = _walk(700, seed=14, vol=0.012)
    p[400] = -3.0
    s = _prep(p)
    res = sfs.evaluate_variants(s, np.array([378, 380, 399]), "up_fade")
    st = lambda cell, j: res.status[res.row(cell), j]
    assert st(("E0", 10, "none", None), 1) == sfs.STATUS_TRADED
    assert st(("E0", 21, "none", None), 1) == sfs.STATUS_TRADED            # crosses 400, exits at 402
    assert st(("E0", 21, "none", 3.0), 1) == sfs.STATUS_TOUCH_NONPOS       # stop path reaches 400
    assert st(("E0", 21, "none", None), 0) == sfs.STATUS_TOUCH_NONPOS      # the exit price is the bad close
    assert st(("E0", 10, "none", None), 0) == sfs.STATUS_TRADED            # exits at 389
    for cell in sfs.CELL_GRID:
        if cell[0] == "E0":
            assert st(cell, 2) in (sfs.STATUS_TOUCH_NONPOS, sfs.STATUS_UNDEFINED_M,
                                   sfs.STATUS_M_TOUCH_NONPOS), cell     # the entry price is the bad close
    # an anchor AT the bad close, or within its sigma window after it, loses every cell
    for t in (400, 401, 430, 463):
        r = sfs.evaluate_variants(s, np.array([t]), "up_fade")
        assert (r.status == sfs.STATUS_SIGMA_TOUCH_NONPOS).all(), t
    r = sfs.evaluate_variants(s, np.array([464]), "up_fade")     # sigma window clear
    assert (r.status != sfs.STATUS_SIGMA_TOUCH_NONPOS).all()
    # TP cells anchored 100 sessions later: the median window still contains 400
    r = sfs.evaluate_variants(s, np.array([500]), "up_fade")
    tp_row = sfs.CELL_INDEX[("E0", 21, "half", None)]
    none_row = sfs.CELL_INDEX[("E0", 21, "none", None)]
    assert r.status[tp_row, 0] == sfs.STATUS_M_TOUCH_NONPOS
    assert r.status[none_row, 0] == sfs.STATUS_TRADED


# ---------------------------------------------------------------------------
# 8. Cell statistics
# ---------------------------------------------------------------------------

def _dates(*days):
    return pd.DatetimeIndex(pd.to_datetime(list(days)))


def test_cluster_ids_link_episodes_within_28_calendar_days_by_chain():
    d = _dates("2020-01-01", "2020-01-20", "2020-02-15", "2020-04-01")
    # 19 days, then 26 days (chained into the same macro event), then 46 days
    assert sfs.cluster_ids(d).tolist() == [0, 0, 0, 1]


def test_cluster_ids_boundary_is_28_days_inclusive():
    assert sfs.cluster_ids(_dates("2020-01-01", "2020-01-29")).tolist() == [0, 0]   # 28 days
    assert sfs.cluster_ids(_dates("2020-01-01", "2020-01-30")).tolist() == [0, 1]   # 29 days


def test_cluster_ids_keep_the_input_order_for_unsorted_dates():
    d = _dates("2020-04-01", "2020-01-01", "2020-01-20")
    assert sfs.cluster_ids(d).tolist() == [1, 0, 0]
    assert sfs.cluster_ids(_dates()).tolist() == []


def test_cell_stats_hand_computed_mean_t_median_hit_rate_and_breakeven():
    """x = 1,2,3,4 (S = 2.5, sd ddof=1 = sqrt(5/3), t = S / (sd / 2) = 3.873);
    pnl = 0.02, -0.01, 0.04, 0.03: mean 0.02 -> break-even 200 bps, hit rate
    3 of 4, worst = -0.01 on the second episode."""
    d = _dates("2010-01-01", "2010-06-01", "2011-01-01", "2011-06-01")
    cs = sfs.cell_stats(x=np.array([1.0, 2.0, 3.0, 4.0]),
                        pnl=np.array([0.02, -0.01, 0.04, 0.03]), dates=d, bootstrap=False)
    assert cs.n == 4 and not cs.eligible                # needs at least 8
    assert cs.mean_x == pytest.approx(2.5)
    assert cs.t == pytest.approx(2.5 / (math.sqrt(5.0 / 3.0) / 2.0))
    assert cs.median_x == pytest.approx(2.5)
    assert cs.hit_rate == pytest.approx(0.75)
    assert cs.mean_pnl == pytest.approx(0.02)
    assert cs.break_even_bps == pytest.approx(200.0)
    assert cs.worst_pnl == pytest.approx(-0.01)
    assert cs.worst_date == pd.Timestamp("2010-06-01")
    assert cs.n_clusters == 4


def test_cell_stats_eligibility_needs_at_least_8_traded_episodes():
    d = pd.date_range("2010-01-01", periods=8, freq="60D")
    x = np.linspace(0.1, 0.8, 8)
    assert sfs.cell_stats(x=x, pnl=x / 10, dates=d, bootstrap=False).eligible
    assert not sfs.cell_stats(x=x[:7], pnl=(x / 10)[:7], dates=d[:7], bootstrap=False).eligible


def test_cell_stats_t_is_nan_when_undefined():
    d = _dates("2010-01-01")
    one = sfs.cell_stats(x=np.array([2.0]), pnl=np.array([0.1]), dates=d, bootstrap=False)
    assert one.n == 1 and np.isnan(one.t)
    d = pd.date_range("2010-01-01", periods=8, freq="60D")
    const = sfs.cell_stats(x=np.full(8, 1.5), pnl=np.full(8, 0.1), dates=d, bootstrap=False)
    assert np.isnan(const.t)                              # sd = 0
    none = sfs.cell_stats(x=np.array([]), pnl=np.array([]), dates=_dates(), bootstrap=False)
    assert none.n == 0 and np.isnan(none.mean_x) and not none.eligible


def test_cell_stats_drops_nan_entries():
    d = pd.date_range("2010-01-01", periods=5, freq="60D")
    cs = sfs.cell_stats(x=np.array([1.0, np.nan, 3.0, np.nan, 2.0]),
                        pnl=np.array([0.1, np.nan, 0.3, np.nan, 0.2]), dates=d, bootstrap=False)
    assert cs.n == 3 and cs.mean_x == pytest.approx(2.0)


def test_cluster_bootstrap_interval_reflects_the_cluster_not_the_episode_count():
    """Hand case. Two macro events of five episodes each: every x in the first
    is +1 and in the second -1. S = 0. Resampling CLUSTERS gives means of
    +1 (25%), 0 (50%) or -1 (25%), so the 90 percent interval is exactly
    [-1, +1]; an episode-level bootstrap would be about 10 times narrower."""
    d = _dates(*["2010-01-%02d" % (1 + 3 * i) for i in range(5)],
               *["2011-01-%02d" % (1 + 3 * i) for i in range(5)])
    x = np.r_[np.ones(5), -np.ones(5)]
    cs = sfs.cell_stats(x=x, pnl=x / 100.0, dates=d)
    assert cs.n_clusters == 2
    assert cs.mean_x == pytest.approx(0.0)
    assert cs.ci_lo == pytest.approx(-1.0) and cs.ci_hi == pytest.approx(1.0)


def test_cluster_bootstrap_is_deterministic_and_seed_dependent():
    rng = np.random.default_rng(0)
    d = pd.date_range("2005-01-01", periods=30, freq="40D")
    x = rng.normal(0.3, 1.0, size=30)
    a = sfs.cell_stats(x=x, pnl=x / 50, dates=d)
    b = sfs.cell_stats(x=x, pnl=x / 50, dates=d)
    assert (a.ci_lo, a.ci_hi) == (b.ci_lo, b.ci_hi)
    c = sfs.cell_stats(x=x, pnl=x / 50, dates=d, seed=1)
    assert (c.ci_lo, c.ci_hi) != (a.ci_lo, a.ci_hi)
    assert a.ci_lo < a.mean_x < a.ci_hi


def test_cluster_bootstrap_interval_is_nan_with_fewer_than_two_clusters():
    d = _dates("2010-01-01", "2010-01-05", "2010-01-09")
    cs = sfs.cell_stats(x=np.array([1.0, 2.0, 3.0]), pnl=np.array([0.1, 0.2, 0.3]), dates=d)
    assert cs.n_clusters == 1 and np.isnan(cs.ci_lo) and np.isnan(cs.ci_hi)


def test_cell_stats_from_results_selects_the_traded_episodes_of_one_cell():
    s = _prep(_messy_series(3))
    t0 = np.arange(300, 640, 9)
    res = sfs.evaluate_variants(s, t0, "up_fade")
    cell = ("E0", 21, "half", 1.5)
    cs = sfs.cell_stats_from_results(res, cell, s.dates, bootstrap=False)
    row = res.row(cell)
    ok = res.status[row] == sfs.STATUS_TRADED
    assert cs.n == int(ok.sum())
    assert cs.mean_x == pytest.approx(res.x[row][ok].mean())
    assert cs.mean_pnl == pytest.approx(res.pnl[row][ok].mean())
    assert cs.worst_pnl == pytest.approx(res.pnl[row][ok].min())


def test_entry_d_e0_is_side_signed_entry_close_minus_median():
    s = _prep(np.arange(1.0, 401.0))
    d_up = sfs.entry_d_e0(s, "up_fade")
    d_dn = sfs.entry_d_e0(s, "down_fade")
    t = 300
    # M(300) = 175.5 (window 50..301); entry close = p[301] = 302
    assert d_up[t] == pytest.approx(302.0 - 175.5)
    assert d_dn[t] == pytest.approx(-(302.0 - 175.5))
    assert np.isnan(d_up[-1])                             # no entry session after the last close
    assert np.isnan(d_up[100])                            # no median yet


# ---------------------------------------------------------------------------
# 9. Group evaluation, counts and determinism
# ---------------------------------------------------------------------------

def _planted(n=2600, seed=0, first=300, every=200, jump=0.15, drift=-0.004,
             hold=21, vol=0.005):
    """Random walk with a big one-day jump every `every` sessions followed by
    `hold` sessions of daily `drift` (a planted reversion when the drift
    opposes the jump). Returns (closes, spike positions)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, vol, n)
    spikes = list(range(first, n - 90, every))
    for t in spikes:
        r[t] = jump + rng.normal(0.0, 0.002)
        r[t + 1:t + 1 + hold] += drift
    return 100.0 * np.exp(np.cumsum(r)), spikes


def test_a_planted_up_spike_reversion_is_a_positive_short_result():
    """End-to-end sign check. Jumps of +15 percent followed by a 20-session
    drift of -0.4 percent a day: shorting entry-next-close and holding 21
    sessions must show a clearly positive mean x."""
    p, spikes = _planted(seed=1)
    s = _prep(p)
    g = sfs.evaluate_group(s, "z20@2.5", "up_fade", bootstrap=False)
    assert g.episodes.n_kept >= 8
    cs = g.stats[sfs.PRIMARY_CELL]
    # x is P&L over sigma21, and the planted jump itself sits inside the 63-session
    # sigma window, so x is near 0.75 rather than the naive 3; the t-stat and sign carry it
    assert cs.eligible and cs.mean_x > 0.5 and cs.t > 5
    assert cs.hit_rate > 0.9
    assert cs.break_even_bps > 300


def test_a_planted_crash_rebound_is_a_positive_long_result():
    p, _ = _planted(seed=2, jump=-0.15, drift=+0.004)
    g = sfs.evaluate_group(_prep(p), "z20@2.5", "down_fade", bootstrap=False)
    cs = g.stats[sfs.PRIMARY_CELL]
    assert cs.eligible and cs.mean_x > 0.5 and cs.t > 5


def test_continuation_after_an_up_spike_is_a_negative_short_result():
    p, _ = _planted(seed=3, jump=0.15, drift=+0.004)
    g = sfs.evaluate_group(_prep(p), "z20@2.5", "up_fade", bootstrap=False)
    cs = g.stats[sfs.PRIMARY_CELL]
    assert cs.eligible and cs.mean_x < -0.5 and cs.t < -3


def test_evaluate_group_covers_all_72_cells_and_counts_every_episode():
    s = _prep(_planted(seed=4)[0])
    g = sfs.evaluate_group(s, "z20@2.0", "up_fade")
    assert list(g.stats) == list(sfs.CELL_GRID)
    fr = g.to_frame()
    assert len(fr) == 72
    for col in ("series", "trigger", "side", "entry", "H", "tp", "sl", "n", "eligible",
                "mean_x", "t", "median_x", "hit_rate", "mean_pnl", "worst_pnl",
                "n_clusters", "ci_lo", "ci_hi", "break_even_bps", "tp_not_applicable"):
        assert col in fr.columns, col
    status_cols = list(sfs.STATUS_NAMES.values())
    assert (fr[status_cols].sum(axis=1) == g.episodes.n_kept).all()
    assert (fr["n"] == fr["traded"]).all()
    e = g.episodes
    assert e.n_episodes == e.n_kept + e.dropped_censored
    assert g.counts["n_flagged_days"] == e.n_flagged_days


def test_evaluate_series_returns_all_eight_groups():
    s = _prep(_planted(seed=5)[0])
    groups = sfs.evaluate_series(s, bootstrap=False)
    assert set(groups) == {(t, sd) for t in sfs.TRIGGERS for sd in sfs.SIDES}
    frame = sfs.series_frame(groups)
    assert len(frame) == 8 * 72 == 576
    assert set(frame["trigger"]) == set(sfs.TRIGGERS)


def test_determinism_identical_inputs_give_identical_outputs():
    """Independently built objects from the same data give
    bit-identical frames, including the seeded bootstrap intervals."""
    p, _ = _planted(seed=6)
    a = sfs.series_frame(sfs.evaluate_series(_prep(p)))
    b = sfs.series_frame(sfs.evaluate_series(_prep(p.copy())))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a["ci_lo"].notna().any()


def _row(frame, entry, H, tp, sl):
    sel = frame[(frame["entry"] == entry) & (frame["H"] == H)
                & (frame["tp"] == tp) & (frame["sl"] == sl)]
    assert len(sel) == 1
    return sel.iloc[0]


def test_counts_of_dropped_censored_undefined_not_applicable_and_external_corrections_all_appear():
    """Counting. Plant one of each observation type and require
    it to show up by name in the report structures."""
    # an external-reference correction: listed by date and counted
    jclose, jref = _jpy_like()
    jclose[200] = jref[200] * 1.03
    sb = _prep(jclose, reference=_series(jref))
    assert sb.counts()["external_corrections"] == 1 and list(sb.external_dates) == [sb.dates[200]]

    # a non-positive close 10 sessions after the spike at 1100, and a missing close
    p, spikes = _planted(n=2600, seed=7)
    p = p.copy()
    assert 1100 in spikes
    p[1110] = -2.0
    p[1500] = np.nan
    s = _prep(p)
    c = s.counts()
    assert c["nonpositive"] == 1 and c["missing"] == 1 and c["external_corrections"] == 0
    assert list(s.nonpos_dates) == [s.dates[1110]] and list(s.missing_dates) == [s.dates[1500]]
    g = sfs.evaluate_group(s, "z20@2.5", "up_fade", bootstrap=False)
    fr = g.to_frame()
    stop_cell = _row(fr, "E0", 21, "none", "1.5")
    assert stop_cell["touch_nonpositive"] >= 1       # a stop path from 1100 reaches 1110
    assert _row(fr, "E0", 21, "none", "none")["touch_nonpositive"] == 0   # a time-only cell does not care
    status_cols = list(sfs.STATUS_NAMES.values())
    assert (fr[status_cols].sum(axis=1) == g.episodes.n_kept).all()

    # censoring: cut the same history 31 sessions after the spike at 2500
    s_end = _prep(_planted(n=2600, seed=7)[0][:2531])
    g_end = sfs.evaluate_group(s_end, "z20@2.5", "up_fade", bootstrap=False)
    assert 2500 in g_end.episodes.t0_all.tolist() and 2500 not in g_end.episodes.t0.tolist()
    assert g_end.episodes.dropped_censored == 1
    assert g_end.episodes.n_kept == g_end.episodes.n_episodes - 1
    fr_end = g_end.to_frame()
    assert (fr_end["dropped_censored"] == 1).all()

    # not-applicable TP is counted per cell and equals the raw array's count
    res = g.results
    tp_cell = ("E0", 21, "half", None)
    row = fr[(fr["entry"] == "E0") & (fr["H"] == 21) & (fr["tp"] == "half") & (fr["sl"] == "none")].iloc[0]
    assert row["tp_not_applicable"] == int(res.tp_na[res.row(tp_cell)].sum())

    # undefined sessions / first usable date / roll-off share ride along
    for col in ("first_usable_date", "n_undefined", "rolloff_share", "n_flagged_days",
                "n_episodes", "dropped_censored"):
        assert col in fr.columns, col


def test_the_whole_pipeline_is_independent_of_everything_after_the_cutoff():
    """End to end. The SAME price history, one copy carrying huge
    post-cutoff spikes, the other with the post-cutoff rows deleted, yields
    identical series, triggers, episodes, contexts and result frames."""
    idx = pd.bdate_range(end="2023-12-29", periods=1500)
    base, _ = _planted(n=1500, seed=8, first=300, every=200)
    cut = pd.Timestamp("2023-06-30")
    spiked = base.copy()
    late = idx > cut
    spiked[late] = spiked[late] * np.exp(np.cumsum(np.where(np.arange(late.sum()) % 17 == 0, 0.5, -0.02)))
    a = sfs.build_series(pd.Series(spiked, index=idx), name="X")
    b = sfs.build_series(pd.Series(base[~late], index=idx[~late]), name="X")
    assert a.dates.equals(b.dates) and a.n == b.n
    assert a.dates.max() <= sfs.DISCOVERY_CUTOFF
    assert a.n_post_cutoff_dropped == int(late.sum()) and b.n_post_cutoff_dropped == 0
    assert np.array_equal(a.p, b.p, equal_nan=True)
    ta, tb = sfs.compute_triggers(a), sfs.compute_triggers(b)
    for k in sfs.TRIGGERS:
        assert _same(ta[k].up, tb[k].up) and _same(ta[k].down, tb[k].down)
        for name, arr in ta[k].values.items():
            assert _same(arr, tb[k].values[name])
    ca, cb = a.context(), b.context()
    for name in ("med", "sigma_raw", "sigma_floor", "sigma", "med_status", "sigma_status"):
        assert _same(getattr(ca, name), getattr(cb, name)), name
    fa = sfs.series_frame(sfs.evaluate_series(a))
    fb = sfs.series_frame(sfs.evaluate_series(b))
    # the only column allowed to differ is the counter of what the seal removed
    assert (fa["series_post_cutoff_dropped"] == int(late.sum())).all()
    assert (fb["series_post_cutoff_dropped"] == 0).all()
    pd.testing.assert_frame_equal(fa.drop(columns="series_post_cutoff_dropped"),
                                  fb.drop(columns="series_post_cutoff_dropped"), check_exact=True)
    ea = sfs.build_all_episode_sets(a)
    eb = sfs.build_all_episode_sets(b)
    for k in ea:
        assert ea[k].t0_all.tolist() == eb[k].t0_all.tolist()
        assert ea[k].dropped_censored == eb[k].dropped_censored


# ---------------------------------------------------------------------------
# 10. Prefix invariance, and proof that the test has power
# ---------------------------------------------------------------------------

_PREFIX_JUMPS = {330: 0.14, 480: -0.14, 640: 0.12, 800: -0.12, 900: 0.15}


def _prefix_fixture(seed, n=1150):
    """Fat-tailed walk with a few big up and down jumps that partly revert, so
    all four triggers fire on both sides at several places."""
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, size=n) * 0.012 / math.sqrt(2.0)
    for t, j in _PREFIX_JUMPS.items():
        r[t] += j
        r[t + 1:t + 16] -= np.sign(j) * 0.003
    return 100.0 * np.exp(np.cumsum(r))


_PREFIX_OFFSETS = (1, 2, 3, 4, 6, 9, 13, 18, 22, 27, 33, 40, 48, 57, 63, 66, 70)


def _pick_anchors(full, k=4):
    """Up to k episode anchors spread across the full series' groups."""
    eps = sfs.build_all_episode_sets(full)
    anchors = sorted({int(t) for e in eps.values() for t in e.t0_all})
    if len(anchors) <= k:
        return anchors
    idx = np.linspace(0, len(anchors) - 1, k).round().astype(int)
    return [anchors[i] for i in idx]


def _prefix_cuts(full, k=4, offsets=_PREFIX_OFFSETS):
    """Cutoff positions (last kept session) 1..70 sessions AFTER each anchor's
    entry (t0 + 1), plus the anchor session itself."""
    cuts = set()
    for t0 in _pick_anchors(full, k):
        cuts.add(t0)
        for off in offsets:
            c = t0 + 1 + off
            if c <= full.n - 1:
                cuts.add(c)
    return sorted(cuts)


_TRADE_FIELDS = ("status", "pnl", "x", "entry_idx", "exit_idx", "exit_reason", "tp_na")


def _prefix_mismatches(closes, cuts, extra_lag=0):
    """Compare everything computed from closes[:cut + 1] with the same
    quantities from the full series restricted to the same sessions. Returns
    (n_checks, mismatch descriptions). Uses the module's public functions
    only, looked up at call time, so a monkeypatched (mutated) module is
    exercised."""
    full = _prep(closes, extra_lag=extra_lag)
    f_sigs = sfs.compute_triggers(full)
    f_eps = sfs.build_all_episode_sets(full, f_sigs)
    f_ctx = full.context()
    f_res = {k: sfs.evaluate_variants(full, e.t0_all, k[1]) for k, e in f_eps.items()}
    n_checks, bad = 0, []

    def check(ok, what):
        nonlocal n_checks
        n_checks += 1
        if not ok:
            bad.append(what)

    for cut in cuts:
        tr = _prep(closes[:cut + 1], extra_lag=extra_lag)
        sigs = sfs.compute_triggers(tr)
        ctx = tr.context()
        eps = sfs.build_all_episode_sets(tr, sigs)
        for trig in sfs.TRIGGERS:
            a, b = f_sigs[trig], sigs[trig]
            for name in ("up", "down", "raw_up", "raw_down", "fresh_up", "fresh_down", "defined"):
                check(_same(getattr(a, name)[:cut + 1], getattr(b, name)), (cut, trig, name))
            for name in a.values:
                check(_same(a.values[name][:cut + 1], b.values[name]), (cut, trig, name))
        for name in ("med", "sigma_raw", "sigma_floor", "sigma", "med_status", "sigma_status"):
            check(_same(getattr(f_ctx, name)[:cut + 1], getattr(ctx, name)), (cut, "ctx", name))
        for key, e in eps.items():
            fe = f_eps[key]
            want = fe.t0_all[fe.t0_all <= cut]
            check(e.t0_all.tolist() == want.tolist(), (cut, key, "episodes"))
            check(e.n_flagged_days == int(f_sigs[key[0]].flags(key[1])[:cut + 1].sum()),
                  (cut, key, "flagged days"))
            if len(e.t0_all):
                tres = sfs.evaluate_variants(tr, e.t0_all, key[1])
                fres = f_res[key]
                pos = np.searchsorted(fe.t0_all, e.t0_all)
                live = tres.status != sfs.STATUS_CENSORED       # a path cut short is censored, not compared
                for name in _TRADE_FIELDS:
                    ta, fa = getattr(tres, name), getattr(fres, name)[:, pos]
                    check(np.array_equal(np.where(live, ta, 0), np.where(live, fa, 0),
                                         equal_nan=(ta.dtype.kind == "f")),
                          (cut, key, "trades:" + name))
    return n_checks, bad


@pytest.mark.parametrize("seed", [101, 102, 103, 104])
def test_prefix_invariance_at_many_cutoffs_after_real_entries(seed):
    """Triggers, contexts, episode sets and every non-censored trade
    outcome computed from a series truncated at a cutoff equal those of the
    full series on the same sessions, for cutoffs 1 to 70 sessions after real
    episode entries (not round dates)."""
    closes = _prefix_fixture(seed)
    full = _prep(closes)
    cuts = _prefix_cuts(full)
    assert len(cuts) >= 40
    n_checks, bad = _prefix_mismatches(closes, cuts)
    assert bad == []
    assert n_checks > 1500                      # four series -> well past a 384-check single-series baseline
    # the fixture really exercises both sides and all triggers
    eps = sfs.build_all_episode_sets(full)
    assert sum(1 for e in eps.values() if len(e.t0_all)) >= 6


def _future_corruption_mismatches(closes, cuts, seed=9):
    """Corrupt-the-future harness: for each cutoff, replace every close AFTER it
    with garbage and list every trigger / context value at or before the cutoff
    that moved. Returns a list of (cut, what, name) descriptors.

    The first garbage close is placed a factor of 100 above and then a factor of
    100 below the cutoff close (two runs). A median ignores a changed element
    that stays on the same side of it, so a one-session peek of a median-based
    value is only visible when the garbage lands on the OTHER side; using both
    extremes guarantees one run does."""
    full = _prep(closes)
    f_sigs = sfs.compute_triggers(full)
    f_ctx = full.context()
    rng = np.random.default_rng(seed)
    bad = []
    for cut in cuts:
        for direction in (+1.0, -1.0):
            junk = closes.copy()
            steps = rng.normal(0.0, 0.06, size=len(junk) - cut - 1)
            steps[0] = direction * math.log(100.0)
            junk[cut + 1:] = closes[cut] * np.exp(np.cumsum(steps))
            s = _prep(junk)
            sigs = sfs.compute_triggers(s)
            for trig in sfs.TRIGGERS:
                for name in ("up", "down", "defined"):
                    if not _same(getattr(f_sigs[trig], name)[:cut + 1],
                                 getattr(sigs[trig], name)[:cut + 1]):
                        bad.append((cut, trig, name))
                for name in f_sigs[trig].values:
                    if not _same(f_sigs[trig].values[name][:cut + 1],
                                 sigs[trig].values[name][:cut + 1]):
                        bad.append((cut, trig, name))
            c = s.context()
            for name in ("med", "sigma", "sigma_floor", "med_status", "sigma_status"):
                if not _same(getattr(f_ctx, name)[:cut + 1], getattr(c, name)[:cut + 1]):
                    bad.append((cut, "ctx", name))
    return bad


def test_prefix_invariance_also_holds_when_the_future_is_replaced_by_garbage():
    """Corrupt-the-future variant: identical series up to a cutoff, wildly
    different afterwards. Everything at or before the cutoff must not move."""
    closes = _prefix_fixture(105)
    cuts = _prefix_cuts(_prep(closes), k=3, offsets=(1, 5, 12, 25, 47, 70))
    assert _future_corruption_mismatches(closes, cuts) == []


def _baseline_window_ending_at(orig, shift):
    """Mutation of the z20 baseline: the value for session t is read from
    session t + shift instead of t. The correct baseline ends at t-1, so
    shift = 1 is a SAME-SESSION baseline (window ending t, includes the signal
    return) and shift = 2 is a genuine ONE-SESSION LOOK-AHEAD (window ending
    t + 1)."""
    def mutated(x, window, min_valid):
        outs = orig(x, window, min_valid)
        return tuple(np.r_[a[shift:], np.full(shift, np.nan)] for a in outs)
    return mutated


def test_the_prefix_test_has_power_a_one_session_look_ahead_makes_it_fail(monkeypatch):
    """Power requirement for the prefix test. Inject a one-session look-ahead into the
    z20 baseline (window ending t + 1, i.e. the baseline series shifted by -2
    from the correct one) and show BOTH harnesses, truncation and
    corrupt-the-future, report mismatches, then that the un-mutated module
    passes them on the same cutoffs."""
    closes = _prefix_fixture(106)
    full = _prep(closes)
    cuts = _prefix_cuts(full, k=2, offsets=(1, 9, 33))
    clean_checks, clean_bad = _prefix_mismatches(closes, cuts)
    assert clean_bad == [] and _future_corruption_mismatches(closes, cuts) == []
    monkeypatch.setattr(sfs, "lagged_window_stats",
                        _baseline_window_ending_at(sfs.lagged_window_stats, 2))
    n_checks, bad = _prefix_mismatches(closes, cuts)
    assert n_checks > 300 and len(bad) > 0
    assert any(b[2] in ("z", "up", "down", "raw_up", "raw_down", "defined") for b in bad)
    assert len(_future_corruption_mismatches(closes, cuts)) > 0


def _estimator_reads_data_from_t_onward(fn, seed=3):
    """Direct estimator test (a prefix-invariance test cannot see a
    same-session slip, so corrupt the future and look at t): True if the value
    of `fn(x, window, min_valid)` at any checked t moves when x[t:] is replaced."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=120)
    base = fn(x, 30, 25)
    for t in (30, 31, 55, 90, 119):
        y = x.copy()
        y[t:] = rng.normal(size=len(y) - t) * 1000.0
        got = fn(y, 30, 25)
        for a, b in zip(base, got):
            if not (a[t] == b[t] or (np.isnan(a[t]) and np.isnan(b[t]))):
                return True
    return False


def test_lagged_window_stats_value_at_t_ignores_every_value_from_t_onward():
    assert not _estimator_reads_data_from_t_onward(sfs.lagged_window_stats)


def test_the_direct_estimator_test_has_power_a_same_session_baseline_fails_it():
    """A baseline whose window ends at t (it includes the signal-session return)
    is invisible to prefix invariance except through an edge effect; the direct
    test must catch it on its own."""
    same_session = _baseline_window_ending_at(sfs.lagged_window_stats, 1)
    assert _estimator_reads_data_from_t_onward(same_session)


def test_the_prefix_test_has_power_against_a_median_window_that_peeks(monkeypatch):
    """A second injected look-ahead: M's window ends at t + 1 instead of t."""
    closes = _prefix_fixture(107)
    full = _prep(closes)
    cuts = _prefix_cuts(full, k=2, offsets=(1, 9, 33))
    assert _prefix_mismatches(closes, cuts)[1] == []
    orig = sfs.rolling_apply

    def peeking(x, window, lag, reducer):
        out = orig(x, window, lag, reducer)
        if window == sfs.MEDIAN_WINDOW and lag == 0:
            return np.r_[out[1:], np.nan]          # M(t) read from the window ending t + 1
        return out

    monkeypatch.setattr(sfs, "rolling_apply", peeking)
    _n, bad = _prefix_mismatches(closes, cuts)
    assert any(b[1] == "ctx" for b in bad)
    assert any(b[1] == "ctx" for b in _future_corruption_mismatches(closes, cuts))


def test_trade_outcome_ignores_every_close_after_the_exit():
    """Direct engine causality test (prefix invariance cannot see a
    same-session peek; corrupt the future and look). For every traded
    (episode, cell) the outcome is unchanged when all closes after its exit
    session are replaced by garbage."""
    s = _prep(_prefix_fixture(108, n=1200))
    ctx = s.context()
    rng = np.random.default_rng(3)
    t0s = np.arange(330, 1100, 37)
    checked = 0
    for side in ("up_fade", "down_fade"):
        res = sfs.evaluate_variants(s, t0s, side)
        for ci, cell in enumerate(res.cells):
            if ci % 5:
                continue
            for j, t0 in enumerate(t0s):
                if res.status[ci, j] != sfs.STATUS_TRADED:
                    continue
                x = int(res.exit_idx[ci, j])
                junk = s.p.copy()
                steps = rng.normal(0, 0.05, size=len(junk) - x - 1)
                junk[x + 1:] = s.p[x] * np.exp(np.cumsum(steps))
                o = sfs.simulate_episode(junk, s.nonpos, int(t0), side, cell[0], cell[1], cell[2], cell[3],
                                         M=float(ctx.med[t0]), med_status=int(ctx.med_status[t0]),
                                         sigma=float(ctx.sigma[t0]), sigma_status=int(ctx.sigma_status[t0]))
                assert o.status == sfs.STATUS_TRADED and o.pnl == res.pnl[ci, j]
                assert o.exit_idx == x and o.exit_reason == res.exit_reason[ci, j]
                checked += 1
    assert checked > 200


# ---------------------------------------------------------------------------
# Extra guards: exact ties, strict hit rate, per-date removal lists,
# malformed cells
# ---------------------------------------------------------------------------

def test_stop_flag_is_inclusive_at_an_exact_tie_in_both_engines():
    """ln(125 / 100) equals the stop level exactly (multiple 1.0 x sigma21 =
    ln 1.25), so the >= test flags at k = 1 and fills at the next close."""
    sigma = float(np.log(1.25))
    o = _sim([100, 100, 125, 126, 127, 128], H=4, sl=1.0, sigma=sigma)
    assert (o.exit_idx, o.exit_reason) == (3, sfs.EXIT_STOP)
    assert o.pnl == pytest.approx(-0.26)


def test_cell_hit_rate_counts_only_strictly_positive_pnl():
    d = pd.date_range("2010-01-01", periods=4, freq="60D")
    cs = sfs.cell_stats(x=np.array([0.0, 1.0, -1.0, 2.0]), pnl=np.array([0.0, 0.01, -0.01, 0.02]),
                        dates=d, bootstrap=False)
    assert cs.hit_rate == pytest.approx(0.5)          # a zero P&L is not a hit


def test_removed_observations_lists_every_affected_date_by_name():
    """Every removed (episode, cell) is listed by date and
    reason, and the tp_not_applicable cells that still traded are listed too."""
    p = _walk(700, seed=14, vol=0.012)
    p[400] = -3.0
    s = _prep(p)
    res = sfs.evaluate_variants(s, np.array([378, 400, 500]), "up_fade")
    df = sfs.removed_observations(s, res)
    assert list(df.columns) == ["date", "t0", "entry", "H", "tp", "sl", "status", "traded"]
    assert len(df) == int(((res.status != sfs.STATUS_TRADED) | res.tp_na).sum())
    assert set(df["status"]) <= set(sfs.STATUS_NAMES.values()) | {"tp_not_applicable"}
    # the episode AT the non-positive close loses every one of the 72 cells
    e400 = df[df["t0"] == 400]
    assert len(e400) == 72 and (e400["status"] == "sigma_touch_nonpositive").all()
    assert (e400["date"] == s.dates[400]).all()
    # the anchor whose H = 21 exit IS the bad close loses that cell
    row = df[(df["t0"] == 378) & (df["entry"] == "E0") & (df["H"] == 21)
             & (df["tp"] == "none") & (df["sl"] == "none")]
    assert row["status"].tolist() == ["touch_nonpositive"]
    # E0 TP cells anchored 100 sessions later lose their median window to the same close
    med = df[(df["t0"] == 500) & (df["entry"] == "E0") & (df["tp"] != "none")]
    assert len(med) > 0 and (med["status"] == "median_touch_nonpositive").all()
    assert df["t0"].is_monotonic_increasing
    assert df[df["status"] == "tp_not_applicable"]["traded"].all()


def test_evaluate_variants_rejects_malformed_cells():
    s = _prep(_walk(400, seed=15))
    for bad in [("E2", 21, "none", None), ("E0", 0, "none", None), ("E0", 64, "none", None),
                ("E0", 21, "third", None), ("E0", 21, "none", -1.0), ("E0", 21, "none")]:
        with pytest.raises(ValueError):
            sfs.evaluate_variants(s, np.array([300]), "up_fade", cells=[bad])
    with pytest.raises(ValueError):
        sfs.evaluate_variants(s, np.array([400]), "up_fade")          # anchor beyond the series


# ---------------------------------------------------------------------------
# Extra publication lag   (FRED publication lag: FRED series enter and exit one session later)
# ---------------------------------------------------------------------------

def test_extra_lag_is_carried_on_the_series_and_validated():
    assert _prep(_walk(300, seed=1)).extra_lag == 0
    assert _prep(_walk(300, seed=1), extra_lag=1).extra_lag == 1
    for bad in (-1, 1.5, True, "1"):
        with pytest.raises(ValueError):
            _prep(_walk(300, seed=1), extra_lag=bad)


def test_extra_lag_shifts_the_E0_entry_and_the_time_stop_follows_the_entry():
    """t0 = 0. Without the lag E0 enters at position 1; with extra_lag = 1 at
    position 2 (close 100), and the H = 3 time stop is measured from THAT entry:
    exit at position 5 (97), short P&L +0.03 (no extra lag on a scheduled exit)."""
    p = [100, 100, 100, 99, 98, 97, 96, 95]
    o0 = _sim(p, H=3)
    assert (o0.entry_idx, o0.exit_idx) == (1, 4)
    o1 = _sim(p, H=3, extra_lag=1)
    assert (o1.entry_idx, o1.exit_idx, o1.exit_reason) == (2, 5, sfs.EXIT_TIME)
    assert o1.pnl == pytest.approx(0.03)
    o2 = _sim(p, H=3, extra_lag=2)
    assert (o2.entry_idx, o2.exit_idx) == (3, 6)
    assert o2.pnl == pytest.approx(-(96 / 99 - 1))


def test_extra_lag_E1_scans_the_unchanged_window_and_enters_at_u_plus_1_plus_L():
    """FRED publication lag, correct reading: the confirmation window (t0, t0 + 5] is NOT
    shifted; the publication lag is added between the confirming observation u
    and the fill, so the entry is at u + 1 + L. Discriminating input:
    t0 = 10, closes flat at 100 except p[11] = 99 (a fall: u = 11 confirms), p[12] =
    101, p[13] = 100, p[14] = 100.5, p[15] = 99. With extra_lag = 1 the entry is
    at 13 (a shifted-window reading would find u = 13 and enter at 14)."""
    p = np.full(40, 100.0)
    p[11], p[12], p[13], p[14], p[15] = 99.0, 101.0, 100.0, 100.5, 99.0
    literal = sfs.simulate_episode(p, None, 10, "up_fade", "E1", 10, "none", None,
                                   sigma=0.1, extra_lag=1)
    assert literal.status == sfs.STATUS_TRADED and literal.entry_idx == 13
    for lag, entry in ((0, 12), (1, 13), (2, 14), (3, 15)):
        o = _sim(p, t0=10, entry="E1", H=10, sigma=0.1, extra_lag=lag)     # both engines
        assert o.status == sfs.STATUS_TRADED and o.entry_idx == entry, lag
        assert o.exit_idx == entry + 10


def test_extra_lag_E1_confirmation_window_is_not_shifted():
    """Rising through position 5 and a fall only at position 6: outside
    (t0, t0 + 5] whatever the lag, so there is no confirmation."""
    p = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99, 98]
    for lag in (0, 1, 2):
        assert _sim(p, entry="E1", H=3, extra_lag=lag).status == sfs.STATUS_NO_E1_CONFIRM, lag
    # a fall at u = 1 confirms whatever the lag (a shifted window would miss it)
    q = [100, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    for lag in (0, 1, 2):
        o = _sim(q, entry="E1", H=3, extra_lag=lag)
        assert o.status == sfs.STATUS_TRADED and o.entry_idx == 2 + lag, lag


def test_extra_lag_E1_scan_that_runs_out_of_data_is_censored_whatever_the_lag():
    """The scan needs position t0 + 5 to exist: 6 closes (last position 5) hold a
    complete scan with no confirmation; 5 closes do not, for every lag."""
    six = [100, 101, 102, 103, 104, 105]
    five = [100, 101, 102, 103, 104]
    for lag in (0, 1, 2):
        assert _sim(six, entry="E1", H=3, extra_lag=lag).status == sfs.STATUS_NO_E1_CONFIRM
        assert _sim(five, entry="E1", H=3, extra_lag=lag).status == sfs.STATUS_CENSORED


def test_extra_lag_E1_entry_beyond_the_data_is_censored():
    """Confirmed at u = 1 but the lagged entry u + 1 + L must exist."""
    p = [100, 99, 98, 97, 96, 95, 94]                     # last position 6
    o = _sim(p, entry="E1", H=1, extra_lag=2)             # entry 4, exit 5
    assert o.status == sfs.STATUS_TRADED and (o.entry_idx, o.exit_idx) == (4, 5)
    assert _sim(p, entry="E1", H=1, extra_lag=4).status == sfs.STATUS_CENSORED   # entry 6, exit 7


def test_extra_lag_delays_a_stop_fill_by_the_extra_sessions():
    """Entry at position 2 (100). The 108 close at k = 1 flags the 1.5 stop
    (0.075 level). No lag would fill at k + 1; extra_lag = 1 fills at k + 2 =
    position 5 (110): P&L -0.10; extra_lag = 2 fills at position 6 (111)."""
    p = [100, 100, 100, 108, 109, 110, 111, 112]
    o = _sim(p, H=5, sl=1.5, extra_lag=1)
    assert (o.entry_idx, o.exit_idx, o.exit_reason) == (2, 5, sfs.EXIT_STOP)
    assert o.pnl == pytest.approx(-0.10)
    q = [100, 100, 100, 100, 108, 109, 110, 111, 112]      # extra_lag = 2: entry at position 3
    o = _sim(q, H=5, sl=1.5, extra_lag=2)
    assert (o.entry_idx, o.exit_idx, o.exit_reason) == (3, 7, sfs.EXIT_STOP)      # flag k = 1, fill k + 3
    assert o.pnl == pytest.approx(-0.11)


def test_extra_lag_delays_a_take_profit_fill_too():
    """Short, M = 90: half target = a gain of 5 (a close <= 95). Entry at 2 (100),
    flag at k = 1 (94), fill at k + 1 + 1 = 3 -> position 5 (97): +0.03."""
    o = _sim([100, 100, 100, 94, 96, 97, 98, 99], H=5, tp="half", M=90.0, extra_lag=1)
    assert (o.exit_idx, o.exit_reason) == (5, sfs.EXIT_TP)
    assert o.pnl == pytest.approx(0.03)


def test_extra_lag_a_flag_whose_delayed_fill_lands_after_the_time_stop_resolves_to_the_time_stop():
    """The scheduled exit comes first. H = 3, extra_lag = 1, flag at k = 2: the
    fill would be at k + 2 = 4 > 3, so the position leaves at the time stop (k = 3,
    position 5, close 109). A flag at k = 1 fills at exactly k = 3 and keeps its
    stop reason."""
    o = _sim([100, 100, 100, 101, 108, 109, 110, 111], H=3, sl=1.5, extra_lag=1)
    assert (o.exit_idx, o.exit_reason) == (5, sfs.EXIT_TIME)
    assert o.pnl == pytest.approx(-0.09)
    o = _sim([100, 100, 100, 108, 109, 110, 111, 112], H=3, sl=1.5, extra_lag=1)
    assert (o.exit_idx, o.exit_reason) == (5, sfs.EXIT_STOP)
    assert o.pnl == pytest.approx(-0.10)


def test_extra_lag_censoring_tail_grows_with_the_lag():
    assert sfs.censor_tail(0) == sfs.CENSOR_TAIL == 70 and sfs.censor_tail(1) == 71
    kept, dropped = sfs.censor_episodes(np.array([28, 29]), 100, tail=sfs.censor_tail(1))
    assert kept.tolist() == [28] and dropped == 1           # 28 + 71 = 99 fits, 29 + 71 does not
    # end to end: 280 + 70 = 350 fits in 351 closes, 280 + 71 does not
    p = _sine_with_crash(n=351)
    e0 = sfs.build_episode_set(_prep(p), sfs.compute_triggers(_prep(p))["z20@2.5"], "down_fade")
    s1 = _prep(p, extra_lag=1)
    e1 = sfs.build_episode_set(s1, sfs.compute_triggers(s1)["z20@2.5"], "down_fade")
    assert e0.t0.tolist() == [280] and e0.dropped_censored == 0
    assert e1.t0.tolist() == [] and e1.dropped_censored == 1


def test_extra_lag_leaves_the_baselines_at_t0():
    """M and sigma21 stay computed at t0: the context arrays do not depend on
    the lag, and a lagged episode reads the same M and sigma21 as an unlagged one."""
    p = _walk(600, seed=81, vol=0.012)
    s0, s1 = _prep(p), _prep(p, extra_lag=2)
    c0, c1 = s0.context(), s1.context()
    for name in ("med", "sigma_raw", "sigma_floor", "sigma", "med_status", "sigma_status"):
        assert _same(getattr(c0, name), getattr(c1, name)), name
    a = sfs.evaluate_variants(s0, np.array([400]), "up_fade", cells=[("E0", 21, "none", 1.5)])
    b = sfs.evaluate_variants(s1, np.array([400]), "up_fade", cells=[("E0", 21, "none", 1.5)])
    assert a.sigma[0] == b.sigma[0]
    assert b.entry_idx[0, 0] == a.entry_idx[0, 0] + 2


@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("side", ["up_fade", "down_fade"])
def test_evaluate_variants_equals_the_reference_with_an_extra_lag(lag, side):
    p = _messy_series(3 + lag)
    s = _prep(p, extra_lag=lag)
    rng = np.random.default_rng(300 + lag)
    t0 = np.unique(np.r_[rng.integers(280, len(p) - 1, size=30),
                         np.arange(len(p) - 80, len(p) - 60), 305, 350, 399])
    res = sfs.evaluate_variants(s, t0, side)
    _assert_results_match_reference(s, t0, side, res)
    assert (res.status == sfs.STATUS_TRADED).sum() > 300
    assert (res.exit_reason[res.traded] == sfs.EXIT_STOP).any()
    # the override argument is the same thing as the series attribute
    s0 = _prep(p)
    res2 = sfs.evaluate_variants(s0, t0, side, extra_lag=lag)
    for name in ("status", "pnl", "entry_idx", "exit_idx", "exit_reason"):
        assert np.array_equal(getattr(res, name), getattr(res2, name), equal_nan=True), name


def test_entry_d_e0_uses_the_lagged_entry_close():
    s = _prep(np.arange(1.0, 401.0), extra_lag=1)
    d = sfs.entry_d_e0(s, "up_fade")
    assert d[300] == pytest.approx(303.0 - 175.5)       # entry at 300 + 1 + 1 = 302 -> close 303
    assert np.isnan(d[-1]) and np.isnan(d[-2])


def test_prefix_invariance_holds_with_an_extra_lag():
    closes = _prefix_fixture(109)
    full = _prep(closes, extra_lag=1)
    cuts = _prefix_cuts(full, k=3, offsets=(1, 2, 5, 9, 22, 40, 63, 66, 70, 71))
    n_checks, bad = _prefix_mismatches(closes, cuts, extra_lag=1)
    assert bad == [] and n_checks > 500


# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_sigma", [0.0, -0.05, np.nan, np.inf, -np.inf])
def test_a_sigma21_that_is_not_finite_and_positive_removes_the_episode_in_both_engines(bad_sigma):
    o = _sim([100, 100, 99, 98, 97, 96], H=3, sigma=bad_sigma)
    assert o.status == sfs.STATUS_UNDEFINED_SIGMA and np.isnan(o.pnl)
    assert _sim([100, 100, 99, 98, 97, 96], H=3, sigma=0.05).status == sfs.STATUS_TRADED


def test_a_zero_sigma_from_a_flat_stretch_removes_the_episode_end_to_end():
    """300 identical closes then a random walk. While every close in the 63-session
    window is identical the sample sd is 0, the floor (the 10th percentile of a
    history that is all zero) is 0, and sigma21 = 0 is not positive: the episode
    is removed, not scored at x = +-infinity."""
    rng = np.random.default_rng(5)
    p = np.r_[np.full(300, 100.0), 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))]
    s = _prep(p)
    ctx = s.context()
    assert ctx.sigma_raw[290] == 0.0 and ctx.sigma_status[290] == sfs.CTX_UNDEFINED
    assert np.isnan(ctx.sigma[290])
    res = sfs.evaluate_variants(s, np.array([290]), "up_fade")
    assert (res.status == sfs.STATUS_UNDEFINED_SIGMA).all()
    assert ctx.sigma_status[460] == sfs.CTX_OK and ctx.sigma[460] > 0


def test_the_sigma_floor_needs_200_valid_prior_values():
    """sigma_raw exists from session 63 on. The floor at session t needs at least
    200 valid values BEFORE t: sessions 63 .. t-1, i.e. t - 63 >= 200, t >= 263."""
    ctx = _prep(_walk(400, seed=91, vol=0.012)).context()
    assert ctx.sigma_status[262] == sfs.CTX_UNDEFINED and np.isnan(ctx.sigma[262])
    assert np.isfinite(ctx.sigma_raw[262])                # the raw value exists; the floor does not
    assert ctx.sigma_status[263] == sfs.CTX_OK
    s = _prep(_walk(400, seed=91, vol=0.012))
    res = sfs.evaluate_variants(s, np.array([262, 263]), "up_fade")
    assert (res.status[:, 0] == sfs.STATUS_UNDEFINED_SIGMA).all()
    assert (res.status[:, 1] != sfs.STATUS_UNDEFINED_SIGMA).all()


def test_the_sigma_floor_prior_count_skips_windows_touching_a_non_positive_close():
    """A non-positive close at 100 makes sigma_raw undefined for sessions 100..163
    (their 64-close windows contain it). Valid sigma_raw sessions: 63..99 (37 of
    them) and 164 onward, so 200 valid values exist before t = 327 exactly."""
    p = _walk(500, seed=92, vol=0.012)
    p[100] = -4.0
    ctx = _prep(p).context()
    assert ctx.sigma_status[326] == sfs.CTX_UNDEFINED
    assert ctx.sigma_status[327] == sfs.CTX_OK
    assert ctx.sigma_status[150] == sfs.CTX_TOUCH_NONPOS


def test_cell_with_sd_of_x_equal_to_zero_has_no_t_and_is_ineligible():
    d = pd.date_range("2010-01-01", periods=8, freq="60D")
    cs = sfs.cell_stats(x=np.full(8, 1.5), pnl=np.full(8, 0.1), dates=d, bootstrap=False)
    assert cs.n == 8 and np.isnan(cs.t) and not cs.eligible
    assert cs.mean_x == pytest.approx(1.5)                     # the descriptive numbers remain
    ok = sfs.cell_stats(x=1.5 + 1e-3 * np.arange(8), pnl=np.full(8, 0.1), dates=d, bootstrap=False)
    assert np.isfinite(ok.t) and ok.eligible
    # 7 episodes with spread: still ineligible on count
    assert not sfs.cell_stats(x=np.arange(7.0), pnl=np.arange(7.0) / 10, dates=d[:7], bootstrap=False).eligible


# ---------------------------------------------------------------------------
# Vehicle mapping
# ---------------------------------------------------------------------------

def _vehicle_calendar(periods=700, start="2012-01-02", holidays=("2013-03-06", "2013-07-04")):
    """A vehicle calendar: business days minus a few 'holidays' the trigger
    series (which trades every business day) does not share."""
    idx = pd.bdate_range(start, periods=periods)
    return idx.drop(pd.DatetimeIndex(list(holidays)))


def test_anchors_map_to_the_last_vehicle_session_on_or_before_the_trigger_date():
    vi = _vehicle_calendar()
    hol = pd.Timestamp("2013-03-06")                       # a trigger-calendar day the vehicle skips
    assert hol not in vi
    pos_prev = int(vi.get_loc(pd.Timestamp("2013-03-05")))
    fri = vi[vi.dayofweek == 4][60]                        # a Friday session; the day after is a Saturday
    sat = fri + pd.Timedelta(days=1)
    assert sat.dayofweek == 5 and sat not in vi
    pos_fri = int(vi.get_loc(fri))
    anchors = pd.DatetimeIndex([hol, vi[400], sat])
    va = sfs.map_anchors_to_vehicle(anchors, vi)
    assert va.positions.tolist() == [pos_prev, 400, pos_fri]
    assert va.source_index.tolist() == [0, 1, 2]
    assert list(va.trigger_dates) == list(anchors)
    assert list(va.session_dates) == [vi[pos_prev], vi[400], fri]
    assert va.n_input == 3 and va.dropped_no_session == va.dropped_short_history == 0
    assert va.dropped_beyond_end == 0 and va.n_duplicate_positions == 0


def test_anchors_with_no_vehicle_session_or_too_little_vehicle_history_are_dropped_and_counted():
    vi = _vehicle_calendar()
    anchors = pd.DatetimeIndex([
        pd.Timestamp("2011-06-01"),                        # before the vehicle exists
        vi[100],                                           # 101 sessions of history < 252
        vi[250],                                           # 251 sessions < 252
        vi[251],                                           # exactly 252 sessions: kept
        vi[300],
    ])
    va = sfs.map_anchors_to_vehicle(anchors, vi)
    assert va.positions.tolist() == [251, 300]
    assert va.source_index.tolist() == [3, 4]
    assert va.dropped_no_session == 1 and va.dropped_short_history == 2
    assert va.n_input == 5


def test_anchor_mapping_adds_the_extra_lag_and_drops_anchors_pushed_past_the_end():
    vi = _vehicle_calendar()
    last = len(vi) - 1
    anchors = pd.DatetimeIndex([vi[300], vi[last - 1], vi[last], vi[last] + pd.Timedelta(days=10)])
    va0 = sfs.map_anchors_to_vehicle(anchors, vi)
    assert va0.positions.tolist() == [300, last - 1, last, last]         # the last two collapse
    assert va0.n_duplicate_positions == 1 and va0.dropped_beyond_end == 0
    va1 = sfs.map_anchors_to_vehicle(anchors, vi, extra_lag=1)
    assert va1.positions.tolist() == [301, last]                         # the last two fall off the end
    assert va1.dropped_beyond_end == 2 and va1.source_index.tolist() == [0, 1]
    with pytest.raises(ValueError):
        sfs.map_anchors_to_vehicle(anchors, vi, extra_lag=-1)


def test_anchor_mapping_rejects_an_unsorted_or_duplicated_vehicle_calendar():
    vi = _vehicle_calendar()
    with pytest.raises(ValueError):
        sfs.map_anchors_to_vehicle(pd.DatetimeIndex([vi[300]]), vi[::-1])
    with pytest.raises(ValueError):
        sfs.map_anchors_to_vehicle(pd.DatetimeIndex([vi[300]]), vi.append(vi[[5]]).sort_values())


def _rising_vehicle_values(n):
    """A steadily rising vehicle with a small wobble (so its sigma21 is positive)."""
    t = np.arange(n)
    return 50.0 * np.exp(0.0015 * t + 0.002 * np.sin(0.7 * t))


def _vehicle_series(vi, values, name="VEH"):
    return sfs.build_series(pd.Series(values, index=vi), name=name)


def test_entry_is_the_next_vehicle_session_and_a_long_vehicle_position_gains_when_the_vehicle_rises():
    """The trigger series trades every business day; the vehicle
    skips 2013-03-06. An anchor on the trigger date 2013-03-05 maps to the vehicle
    session 2013-03-05 and E0 enters at the close of the NEXT VEHICLE session
    (2013-03-07, not the trigger calendar's 03-06). A long vehicle position (side
    = -1) earns the vehicle's own return, which is positive here, and the same
    trades as a short (side = +1) lose it."""
    vi = _vehicle_calendar()
    v = _rising_vehicle_values(len(vi))                        # a steadily rising vehicle
    veh = _vehicle_series(vi, v)
    anchor = pd.DatetimeIndex([pd.Timestamp("2013-03-05"), vi[400]])
    va = sfs.map_anchors_to_vehicle(anchor, vi)
    pos = va.positions
    assert vi[pos[0] + 1] == pd.Timestamp("2013-03-07")        # next VEHICLE session after the gap
    long_ = sfs.evaluate_variants(veh, pos, -1, cells=[sfs.PRIMARY_CELL])
    short = sfs.evaluate_variants(veh, pos, +1, cells=[sfs.PRIMARY_CELL])
    assert (long_.entry_idx[0] == pos + 1).all() and (long_.exit_idx[0] == pos + 1 + 21).all()
    want = v[pos + 1 + 21] / v[pos + 1] - 1.0
    assert np.allclose(long_.pnl[0], want) and (long_.pnl[0] > 0).all()
    assert np.allclose(short.pnl[0], -want) and (short.pnl[0] < 0).all()
    assert (long_.x[0] > 0).all()


def test_the_vehicle_test_uses_the_vehicles_own_D_M_take_profit_stop_and_sigma():
    """Every level comes from the vehicle's OWN closes. Vehicle: flat near 100 with a
    tiny wobble; anchor A at a session where the next close dips to 90 and then climbs
    3 a session; anchor B where it falls 4 percent and keeps falling. Long vehicle
    (side = -1): at A, D = M - P_entry = about 10 > 0 and the half target (a gain of
    about 5) is first met at k = 2 (96), filling at k = 3 (99); at B the stop level
    is 1.5 x the vehicle's own sigma21 and the fall flags at k = 1."""
    vi = _vehicle_calendar()
    n = len(vi)
    wob = 0.05 * np.sin(0.7 * np.arange(n))             # keeps sigma21 positive everywhere
    v = 100.0 + wob
    pa, pb = 300, 500
    v[pa + 1] = 90.0
    v[pa + 2:pa + 12] = 90.0 + 3.0 * np.arange(1, 11)
    v[pa + 12:pb + 1] = 100.0 + wob[pa + 12:pb + 1]
    v[pb + 1] = 96.0
    v[pb + 2:pb + 30] = 95.0 + wob[pb + 2:pb + 30]
    veh = _vehicle_series(vi, v)
    ctx = veh.context()
    va = sfs.map_anchors_to_vehicle(vi[[pa, pb]], vi)
    assert va.positions.tolist() == [pa, pb]
    # D at an E0 entry, long vehicle: -(P_entry - M) with the vehicle's own median
    d = sfs.entry_d_e0(veh, -1)
    assert d[pa] == pytest.approx(ctx.med[pa] - 90.0) and 9.9 < d[pa] < 10.1
    a = sfs.evaluate_variants(veh, np.array([pa]), -1, cells=[("E0", 21, "half", None)])
    assert a.status[0, 0] == sfs.STATUS_TRADED and not a.tp_na[0, 0]
    assert a.exit_reason[0, 0] == sfs.EXIT_TP
    assert a.entry_idx[0, 0] == pa + 1 and a.exit_idx[0, 0] == pa + 1 + 3      # flag k = 2, fill k = 3
    assert a.pnl[0, 0] == pytest.approx(99.0 / 90.0 - 1.0)
    # the stop: level = 1.5 x the vehicle's own sigma21 at the anchor
    b = sfs.evaluate_variants(veh, np.array([pb]), -1, cells=[("E0", 21, "none", 1.5)])
    assert b.sigma[0] == ctx.sigma[pb] and b.sigma[0] > 0
    assert 1.5 * b.sigma[0] < abs(np.log(95.0 / 96.0))                            # the fall clears the stop level
    assert b.status[0, 0] == sfs.STATUS_TRADED and b.exit_reason[0, 0] == sfs.EXIT_STOP
    assert b.exit_idx[0, 0] == pb + 1 + 2                                        # flag k = 1, fill k = 2
    assert b.pnl[0, 0] == pytest.approx(v[pb + 3] / v[pb + 1] - 1.0)             # a long loses on the fall
    assert b.pnl[0, 0] < 0


def test_E1_on_a_long_vehicle_confirms_on_the_vehicles_own_up_close():
    """E1 for a long position waits for a close ABOVE the previous close of the
    VEHICLE: after the anchor 100 -> 99 (down, no) -> 100 (up, confirms at u = 2),
    entering at u + 1."""
    vi = _vehicle_calendar()
    v = 100.0 + 0.05 * np.sin(0.7 * np.arange(len(vi)))
    pos = 300
    v[pos], v[pos + 1], v[pos + 2] = 100.0, 99.0, 100.0
    veh = _vehicle_series(vi, v)
    r = sfs.evaluate_variants(veh, np.array([pos]), -1, cells=[("E1", 10, "none", None)])
    assert r.status[0, 0] == sfs.STATUS_TRADED and r.entry_idx[0, 0] == pos + 3


def test_vehicle_extra_lag_is_carried_by_the_mapped_anchor():
    """For a FRED trigger series the mapped anchor already contains the lag, so
    the vehicle series itself keeps extra_lag = 0 and E0 enters the session after
    the LAGGED anchor."""
    vi = _vehicle_calendar()
    v = _rising_vehicle_values(len(vi))
    veh = _vehicle_series(vi, v)
    va = sfs.map_anchors_to_vehicle(pd.DatetimeIndex([vi[400]]), vi, extra_lag=1)
    r = sfs.evaluate_variants(veh, va.positions, -1, cells=[sfs.PRIMARY_CELL])
    assert va.positions.tolist() == [401] and r.entry_idx[0, 0] == 402


def test_rolling_apply_refuses_a_lag_that_would_read_the_future():
    """A window ending after session t is a look-ahead by construction; the
    primitive every baseline goes through rejects it (no data rule
    may use a later value)."""
    x = np.arange(50.0)
    with pytest.raises(ValueError, match="future"):
        sfs.rolling_apply(x, 5, -1, lambda w: w.sum(axis=1))
    assert not np.isnan(sfs.rolling_apply(x, 5, 0, lambda w: w.sum(axis=1))[4])


# ---------------------------------------------------------------------------
# Float-fragile sd guard, E1 ties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,ns", [(0.1, (8, 12)), (0.3, (10, 11, 12, 13)), (0.7, (11, 12)),
                                      (0.123_456_789, (9, 10)), (-0.42, (8, 9)), (1e-9, (10,))],
                         ids=["tenth", "three_tenths", "seven_tenths", "long_fraction", "negative", "tiny"])
def test_identical_x_values_are_a_zero_sd_not_a_float_noise_t(value, ns):
    """np.full(12, 0.1).std(ddof=1) is about 1e-17, not 0, which would naively yield
    t of about 1e16 and an eligible cell. Identical x is sd = 0 (numerical guard): t is
    undefined and the cell ineligible, at any n and for any repeated value."""
    for n in ns:
        d = pd.date_range("2010-01-01", periods=n, freq="60D")
        cs = sfs.cell_stats(x=np.full(n, value), pnl=np.full(n, 0.01), dates=d, bootstrap=False)
        assert cs.n == n and np.isnan(cs.t) and not cs.eligible, (value, n)
        assert cs.mean_x == pytest.approx(value)


def test_a_relative_sd_of_float_noise_size_is_also_a_zero_sd():
    """Values that differ only in the last bit or so are still 'identical': sd at
    or below 1e-12 * max(1, |S|) is treated as zero."""
    d = pd.date_range("2010-01-01", periods=10, freq="60D")
    x = 5.0 + 1e-15 * np.arange(10)                       # sd about 3e-15 <= 1e-12 * 5
    cs = sfs.cell_stats(x=x, pnl=np.full(10, 0.01), dates=d, bootstrap=False)
    assert np.isnan(cs.t) and not cs.eligible
    y = 5.0 + 1e-6 * np.arange(10)                        # tiny but genuine spread
    cs2 = sfs.cell_stats(x=y, pnl=np.full(10, 0.01), dates=d, bootstrap=False)
    assert np.isfinite(cs2.t) and cs2.eligible


# --- E1 ties -----------------------------------------------------------------------

def test_E1_equal_consecutive_closes_do_not_confirm_on_either_side():
    """A confirming session needs a STRICTLY lower (up side) / higher (down side)
    close than the previous session; a tie is not a confirmation, including the
    tie with the signal-session close at u = 1. Both engines."""
    # up_fade (short): a tie at u = 1 with the signal close, then a rise: no confirmation
    p = [100, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    assert _sim(p, entry="E1", H=3).status == sfs.STATUS_NO_E1_CONFIRM
    # ... but the first STRICT fall after the tie confirms (u = 2, entry 3)
    p = [100, 100, 99, 100, 101, 102, 103, 104, 105, 106]
    o = _sim(p, entry="E1", H=3)
    assert o.status == sfs.STATUS_TRADED and o.entry_idx == 3
    # a tie in the middle of the window
    p = [100, 101, 102, 102, 103, 104, 105, 106, 107, 108]
    assert _sim(p, entry="E1", H=3).status == sfs.STATUS_NO_E1_CONFIRM
    # down_fade (long): equal closes do not confirm, a strict rise does
    p = [100, 100, 99, 98, 97, 96, 95, 94, 93, 92]
    assert _sim(p, side="down_fade", entry="E1", H=3).status == sfs.STATUS_NO_E1_CONFIRM
    p = [100, 100, 101, 100, 99, 98, 97, 96, 95, 94]
    o = _sim(p, side="down_fade", entry="E1", H=3)
    assert o.status == sfs.STATUS_TRADED and o.entry_idx == 3
    p = [100, 99, 98, 98, 97, 96, 95, 94, 93, 92]
    assert _sim(p, side="down_fade", entry="E1", H=3).status == sfs.STATUS_NO_E1_CONFIRM


def _tie_series(seed, n=700):
    """Integer-valued closes from a lazy walk (a third of the steps are zero), so
    equal consecutive closes are common."""
    rng = np.random.default_rng(seed)
    p = 100.0 + np.cumsum(rng.integers(-1, 2, size=n))
    return np.maximum(p, 40.0)


@pytest.mark.parametrize("side", ["up_fade", "down_fade"])
def test_evaluate_variants_equals_the_reference_on_data_full_of_ties(side):
    p = _tie_series(31)
    assert (np.diff(p) == 0).mean() > 0.25                # ties really are common
    s = _prep(p)
    t0 = np.arange(280, len(p) - 1, 5)
    res = sfs.evaluate_variants(s, t0, side)
    _assert_results_match_reference(s, t0, side, res)
    e1 = [i for i, c in enumerate(res.cells) if c[0] == "E1"]
    assert (res.status[e1] == sfs.STATUS_NO_E1_CONFIRM).any() and res.traded[e1].any()


# ---------------------------------------------------------------------------
# The seal is not a default a caller can move   
# ---------------------------------------------------------------------------

def _seal_probe_series():
    idx = pd.bdate_range("2023-06-01", periods=60)               # runs past 2023-06-30
    return pd.Series(np.linspace(100.0, 110.0, 60), index=idx)


def test_a_caller_cannot_unseal_the_loader_or_any_builder_with_a_later_cutoff():
    s = _seal_probe_series()
    later = sfs.DISCOVERY_CUTOFF + pd.Timedelta(days=1)
    for bad in (later, pd.Timestamp("2099-01-01")):
        with pytest.raises(sfs.SealedWindowError):
            sfs.load_sealed(s, name="X", cutoff=bad)
        with pytest.raises(sfs.SealedWindowError):
            sfs.build_series(s, name="X", cutoff=bad)
        with pytest.raises(sfs.SealedWindowError):
            sfs.assert_sealed(s.index[:5], bad)               # even when every date is <= that cutoff
    hand = sfs.SealedSeries(name="X", index=s.index[:5], close=np.full(5, 100.0),
                            cutoff=pd.Timestamp("2099-01-01"), n_post_cutoff_dropped=0,
                            n_pre_start_dropped=0)
    with pytest.raises(sfs.SealedWindowError):
        sfs.prepare_series(hand)                              # a hand-built sealed object too
    # the discovery cutoff itself and any earlier one are allowed
    assert sfs.load_sealed(s, name="X", cutoff=sfs.DISCOVERY_CUTOFF).max_date == sfs.DISCOVERY_CUTOFF
    assert sfs.load_sealed(s, name="X", cutoff=pd.Timestamp("2023-06-15")).max_date <= pd.Timestamp("2023-06-15")


def test_the_only_way_past_the_seal_is_a_private_test_helper():
    import inspect
    words = ("unseal", "ceiling", "override", "allow", "force")
    public = [n for n in dir(sfs) if not n.startswith("_")]
    assert not [n for n in public if any(w in n.lower() for w in words)]
    for fn in (sfs.load_sealed, sfs.build_series, sfs.prepare_series, sfs.assert_sealed):
        params = list(inspect.signature(fn).parameters)
        assert not [p for p in params if any(w in p.lower() for w in words)], fn.__name__
    assert sfs._allow_cutoff_beyond_discovery.__name__.startswith("_")
    s = _seal_probe_series()
    far = pd.Timestamp("2099-01-01")
    with sfs._allow_cutoff_beyond_discovery(far):
        loaded = sfs.load_sealed(s, name="X", cutoff=far)
        assert loaded.n_post_cutoff_dropped == 0 and loaded.max_date == s.index[-1]
    with pytest.raises(sfs.SealedWindowError):                # restored on normal exit
        sfs.load_sealed(s, name="X", cutoff=far)
    with pytest.raises(RuntimeError):
        with sfs._allow_cutoff_beyond_discovery(far):
            raise RuntimeError("boom")
    with pytest.raises(sfs.SealedWindowError):                # and when the block raises
        sfs.load_sealed(s, name="X", cutoff=far)
    with pytest.raises(ValueError):                           # the helper cannot LOWER the seal
        with sfs._allow_cutoff_beyond_discovery(pd.Timestamp("2020-01-01")):
            pass


def test_compute_triggers_evaluate_variants_and_context_reject_a_hand_built_series_past_the_cutoff():
    """A SpikeSeries assembled by hand with post-cutoff dates (or with a cutoff
    later than the discovery cutoff) cannot reach the trigger, context or trade code."""
    n = 300
    p = _walk(n, seed=1)
    zeros = np.zeros(n, bool)
    for dates, cutoff in ((pd.bdate_range("2023-06-20", periods=n), sfs.DISCOVERY_CUTOFF),   # dates after it
                          (pd.bdate_range("2023-06-20", periods=n), pd.Timestamp("2099-01-01")),
                          (pd.bdate_range(end="2023-06-30", periods=n), pd.Timestamp("2099-01-01"))):
        s = sfs.SpikeSeries(name="H", cutoff=cutoff, dates=dates, raw=p.copy(), p=p, nonpos=zeros,
                            missing=zeros)
        with pytest.raises(sfs.SealedWindowError):
            sfs.compute_triggers(s)
        with pytest.raises(sfs.SealedWindowError):
            sfs.evaluate_variants(s, np.array([100]), "up_fade")
        with pytest.raises(sfs.SealedWindowError):
            s.context()
    ok = sfs.SpikeSeries(name="H", cutoff=sfs.DISCOVERY_CUTOFF,
                         dates=pd.bdate_range(end="2023-06-30", periods=n), raw=p.copy(), p=p,
                         nonpos=zeros, missing=zeros)
    assert set(sfs.compute_triggers(ok)) == set(sfs.TRIGGERS)
    assert sfs.evaluate_variants(ok, np.array([100]), "up_fade").pnl.shape == (72, 1)


# ---------------------------------------------------------------------------
# JPY reference is mandatory, frame counters, pandas warning,
# window-length and validity-boundary pins
# ---------------------------------------------------------------------------

def test_the_jpy_series_cannot_be_built_without_its_external_reference():
    """JPY=X is the one series whose only data correction is the same-day FRED
    DEXJPUS check; building it without the reference would silently allow
    uncorrected ticks, so it is an error."""
    close, ref = _jpy_like()
    ser = _series(close)
    with pytest.raises(ValueError, match="DEXJPUS"):
        sfs.build_series(ser, name="JPY=X")
    with pytest.raises(ValueError, match="DEXJPUS"):
        sfs.prepare_series(sfs.load_sealed(ser, name="JPY=X"))
    ok = sfs.build_series(ser, name="JPY=X", reference=_series(ref))
    assert ok.name == "JPY=X" and ok.n == 300
    assert sfs.build_series(ser, name="^TNX").n == 300              # other series need no reference
    assert sfs.EXTERNAL_CHECK_SERIES == ("JPY=X",)


def test_the_group_frame_carries_the_series_level_counters():
    close, ref = _jpy_like()
    close[200] = ref[200] * 1.03
    s = _prep(close, reference=_series(ref))
    counts = s.counts()
    assert counts["external_corrections"] == 1
    fr = sfs.evaluate_group(s, "z20@2.5", "up_fade", bootstrap=False).to_frame()
    assert (fr["series_external_corrections"] == 1).all()
    for key, value in counts.items():
        assert f"series_{key}" in fr.columns and (fr[f"series_{key}"] == value).all(), key
    whole = sfs.series_frame(sfs.evaluate_series(s, bootstrap=False))
    assert len(whole) == 576 and (whole["series_external_corrections"] == 1).all()


def test_series_frame_handles_empty_and_all_na_groups_without_a_pandas_FutureWarning():
    import warnings
    quiet = sfs.evaluate_series(_prep(_walk(250, seed=3)), bootstrap=False)      # too short for any trigger: no episodes
    assert all(g.episodes.n_kept == 0 for g in quiet.values())
    mixed = sfs.evaluate_series(_prep(_planted(seed=5)[0]), bootstrap=False)     # some groups empty
    assert any(g.episodes.n_kept == 0 for g in mixed.values()) and any(g.episodes.n_kept for g in mixed.values())
    mixed_walk = sfs.evaluate_series(_prep(_walk(400, seed=3)), bootstrap=False)   # a FutureWarning-prone mix
    for groups in (quiet, mixed, mixed_walk):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fr = sfs.series_frame(groups)
        assert len(fr) == 576
        assert str(fr["worst_date"].dtype).startswith("datetime64") and fr["mean_x"].dtype.kind == "f"
    assert quiet and (sfs.series_frame(quiet)["n"] == 0).all()
    assert sfs.series_frame({}).empty


# --- the fresh-extreme window is exactly 20 closes --------------------------------------

def _sine_with_jump(level, prior=None, t=300, n=420):
    """Quiet sine (r20 sd about 0.003) with a jump to `level` at session t; `prior`
    optionally places a more extreme close exactly 19 sessions earlier (t - 19)."""
    p = 100.0 * (1 + 0.002 * np.sin(0.9 * np.arange(n)))
    p[t] = level
    if prior is not None:
        p[t - 19] = prior
    return p


@pytest.mark.parametrize("side,level,prior", [("up", 101.5, 102.0), ("down", 98.5, 98.0)])
def test_the_fresh_extreme_window_is_20_closes_a_close_19_sessions_back_still_counts(side, level, prior):
    """Session 300 jumps (z far beyond 2.5) but the close 19 sessions earlier, at
    281, is more extreme: inside the 20-close window t-19..t, so 300 is NOT a fresh
    extreme and does not flag. A 19-close window (t-18..t) would flag it. Without
    the extreme close at 281 the same jump flags (control)."""
    sig = sfs.compute_triggers(_prep(_sine_with_jump(level, prior)))["z20@2.5"]
    raw, fresh, flag = getattr(sig, f"raw_{side}"), getattr(sig, f"fresh_{side}"), getattr(sig, side)
    assert raw[300] and not fresh[300] and not flag[300]
    ctrl = sfs.compute_triggers(_prep(_sine_with_jump(level)))["z20@2.5"]
    assert getattr(ctrl, f"raw_{side}")[300] and getattr(ctrl, f"fresh_{side}")[300]
    assert getattr(ctrl, side)[300]


# --- validity minimums are exact -------------------------------------------------------------

@pytest.mark.parametrize("valid,expect", [(499, False), (500, True), (501, True)])
def test_pct756_needs_exactly_500_valid_closes_of_756(valid, expect):
    """Window for t = 800 is positions 44..799 (756 closes). A NaN block leaves
    exactly `valid` of them."""
    p = _walk(900, seed=7, vol=0.01)
    p[44:44 + (756 - valid)] = np.nan
    sig = sfs.compute_triggers(_prep(p))["pct756"]
    assert bool(sig.defined[800]) == expect
    if expect:
        w = p[44:800]
        w = w[np.isfinite(w)]
        assert len(w) == valid
        assert sig.values["hi"][800] == pytest.approx(np.percentile(w, 95, method="linear"))
        assert sig.values["lo"][800] == pytest.approx(np.percentile(w, 5, method="linear"))


def test_sigma_window_with_exactly_50_valid_returns_is_usable_and_49_is_not():
    """Boundary case: random walk n = 600, seed 63, NaN closes at 257, 300, 302,
    304, 306, 308, 310. The 63 returns ending at 320 are r[258..320]; r258 (from the
    NaN at 257) and r300..r311 (from the six NaN closes) are missing: 13 missing, so
    exactly 50 valid. One more missing return (NaN close at 320 kills r320) leaves 49."""
    def valid_returns(p):
        r = np.r_[np.nan, np.log(p[1:] / p[:-1])]
        return int(np.isfinite(r[258:321]).sum())
    p = _walk(600, seed=63, vol=0.012)
    p[[257, 300, 302, 304, 306, 308, 310]] = np.nan
    assert valid_returns(p) == 50
    assert _prep(p).context().sigma_status[320] == sfs.CTX_OK
    q = p.copy()
    q[320] = np.nan
    assert valid_returns(q) == 49
    assert _prep(q).context().sigma_status[320] == sfs.CTX_UNDEFINED


def test_the_zero_sd_tolerance_scales_with_the_size_of_the_mean():
    """The tolerance is 1e-12 * max(1, |S|). x = base + 1e-12 * k (sd about 3.0e-12):
    at S = 5 the threshold is 5e-12 so it is float noise (t undefined, ineligible);
    at S = 0.5 the threshold is 1e-12 so the same spread is genuine (t defined)."""
    d = pd.date_range("2010-01-01", periods=10, freq="60D")
    steps = 1e-12 * np.arange(10)
    big = sfs.cell_stats(x=5.0 + steps, pnl=np.full(10, 0.01), dates=d, bootstrap=False)
    assert np.isnan(big.t) and not big.eligible
    small = sfs.cell_stats(x=0.5 + steps, pnl=np.full(10, 0.01), dates=d, bootstrap=False)
    assert np.isfinite(small.t) and small.eligible


def test_evaluate_variants_asserts_the_seal_itself_even_when_handed_a_ready_made_context():
    """`series.context()` already asserts the seal, which would mask a missing
    guard in `evaluate_variants`; passing a context explicitly isolates it."""
    n = 300
    p = _walk(n, seed=1)
    zeros = np.zeros(n, bool)
    good = sfs.SpikeSeries(name="G", cutoff=sfs.DISCOVERY_CUTOFF,
                           dates=pd.bdate_range(end="2023-06-30", periods=n), raw=p.copy(), p=p,
                           nonpos=zeros, missing=zeros)
    ctx = good.context()
    bad = sfs.SpikeSeries(name="B", cutoff=sfs.DISCOVERY_CUTOFF,
                          dates=pd.bdate_range("2023-06-20", periods=n), raw=p.copy(), p=p,
                          nonpos=zeros, missing=zeros)
    with pytest.raises(sfs.SealedWindowError):
        sfs.evaluate_variants(bad, np.array([100]), "up_fade", context=ctx)
    assert sfs.evaluate_variants(good, np.array([100]), "up_fade", context=ctx).pnl.shape == (72, 1)
