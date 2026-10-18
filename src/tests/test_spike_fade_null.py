"""Tests for src/mining/spike_fade_null.py -- the NULL / TEST machinery of the
range-bound spike-fade study (the confirmatory max-t test and its two nulls,
the same-cell null quantile helper the exploratory map uses for context, and
the synthetic calibration tests).

Tests are hand-derived on small synthetic inputs, so
each expected value can be checked by hand.
Synthetic data only -- no network, no real prices. Sizes are tiny on purpose
(a few thousand sessions, tens of replications); the full calibration run of
acceptance test 5 lives in scripts/validate_spike_fade_null.py.

Layout:
  1. constants, the per-series RNG streams
  2. observation: the frozen tested set and T*
  3. N1: exact valid-n matching, D-sign strata, coupling, independence, failure
  4. N2: pure translation, no wrap, feasible offsets, skipping
  5. p-values, the headline, adjusted p, determinism
  6. the same-cell N1 null distribution for the exploratory map
  7. the synthetic calibration harness (test 5): generator, injection, size
     and power smoke checks, summaries
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

from mining import spike_fade_null as sfn
from mining import spike_fade_study as sfs


# ---------------------------------------------------------------------------
# Synthetic-data helpers (shared by every block)
# ---------------------------------------------------------------------------

def _sv_walk(n, seed, vol=0.012, df=3.8, phi=0.97, eta=0.2):
    """Stochastic-volatility, fat-tailed random-walk closes (no edge)."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        h[i] = phi * h[i - 1] + eta * e[i]
    r = rng.standard_t(df, n) / math.sqrt(df / (df - 2.0)) * vol * np.exp(h / 2.0)
    return 100.0 * np.exp(np.cumsum(r))


def _spike_series(n, seed, name):
    idx = pd.bdate_range("2000-01-03", periods=n)
    return sfs.build_series(pd.Series(_sv_walk(n, seed), index=idx), name=name)


@pytest.fixture(scope="module")
def universe():
    """Two independent 3500-session series with several tested groups each
    (verified when the fixture was chosen: 6 tested groups per series)."""
    return [_spike_series(3500, 0, "A"), _spike_series(3500, 2, "B")]


@pytest.fixture(scope="module")
def obs(universe):
    return sfn.observe_family(universe)


def _key_str(k):
    return "|".join(k)


def _independent_x(series, side, anchors):
    """x of the primary cell computed the SLOW reference way (sequential
    simulator), independent of the numpy engine the module uses."""
    out = []
    for a in anchors:
        o = sfs.simulate_from_series(series, int(a), side, sfs.PRIMARY_CELL)
        out.append(sfs.episode_x(o, float(series.context().sigma[int(a)]))
                   if o.status == sfs.STATUS_TRADED else float("nan"))
    return np.asarray(out, dtype=float)


def _independent_strata(series, side, anchors):
    d = sfs.entry_d_e0(series, side)[np.asarray(anchors, dtype=int)]
    return (int((d > 0).sum()), int((d <= 0).sum()), int(np.isnan(d).sum()))


# ---------------------------------------------------------------------------
# 1. Constants and the per-series RNG streams
# ---------------------------------------------------------------------------

def test_frozen_constants_match_the_pre_registration():
    assert sfn.N1_SEED == 2026_0923
    assert sfn.N2_SEED == 2026_0924
    assert sfn.MAP_SEED == 2026_0926
    assert sfn.NULL_B == 500
    assert sfn.MAP_B == 200
    assert sfn.N2_GRID_STEP == 21
    assert sfn.N2_MIN_OFFSETS == 10
    assert sfn.HEADLINE_P_MAX == pytest.approx(0.10)


def test_series_streams_are_deterministic_and_independent_of_the_family():
    a1 = sfn.series_rng(2026_0923, "A").permutation(50)
    a2 = sfn.series_rng(2026_0923, "A").permutation(50)
    b = sfn.series_rng(2026_0923, "B").permutation(50)
    other_seed = sfn.series_rng(2026_0924, "A").permutation(50)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)
    assert not np.array_equal(a1, other_seed)


# ---------------------------------------------------------------------------
# 2. Observation: the frozen tested set and T*
# ---------------------------------------------------------------------------

def test_t_stat_matches_the_core_cell_statistic_exactly():
    rng = np.random.default_rng(3)
    for n in (2, 5, 9, 40):
        x = np.clip(rng.standard_t(4, size=n), -5, 5)
        dates = pd.bdate_range("2001-01-01", periods=n)
        ref = sfs.cell_stats(x, x, dates, bootstrap=False).t
        assert sfn.t_stat(x) == ref
    assert math.isnan(sfn.t_stat(np.array([1.0])))            # n < 2
    assert math.isnan(sfn.t_stat(np.array([2.0, 2.0, 2.0])))  # sd = 0


def test_observe_family_builds_all_56_style_groups(obs, universe):
    # 2 series x 2 sides x 4 triggers = 16 groups here (56 for the real 7)
    assert len(obs.groups) == 2 * 2 * 4
    for key in obs.groups:
        series, side, trigger = key
        assert series in ("A", "B") and side in sfs.SIDES and trigger in sfs.TRIGGERS
    assert obs.series_names == ("A", "B")


def test_observed_group_matches_an_independent_recomputation(obs, universe):
    by_name = {s.name: s for s in universe}
    for key, g in obs.groups.items():
        s = by_name[g.series]
        sig = sfs.compute_triggers(s)[g.trigger]
        es = sfs.build_episode_set(s, sig, g.side)
        x = _independent_x(s, g.side, es.t0)
        ok = np.isfinite(x)
        assert g.n == int(ok.sum())
        assert g.n_kept == len(es.t0)
        assert np.array_equal(g.anchors, es.t0[ok])
        assert np.allclose(g.x, x[ok], rtol=0, atol=0)
        assert g.strata == _independent_strata(s, g.side, g.anchors)
        assert sum(g.strata) == g.n
        cs = sfs.cell_stats(x[ok], x[ok], s.dates[es.t0[ok]], bootstrap=False)
        assert g.t == cs.t or (math.isnan(g.t) and math.isnan(cs.t))
        assert g.mean_x == cs.mean_x or (math.isnan(g.mean_x) and math.isnan(cs.mean_x))


def test_tables_agree_with_the_engine_at_every_eligible_anchor(universe):
    s = universe[0]
    tab = sfn.build_tables(s)
    # censoring: exactly the anchors with t0 + CENSOR_TAIL <= n - 1
    assert len(tab.elig) == s.n - sfs.CENSOR_TAIL
    assert tab.elig[0] == 0 and tab.elig[-1] == s.n - 1 - sfs.CENSOR_TAIL
    for side in sfs.SIDES:
        sample = tab.elig[::97]
        ref = _independent_x(s, side, sample)
        assert np.array_equal(np.isfinite(ref), tab.valid[side][sample])
        assert np.allclose(np.nan_to_num(ref, nan=0.0), np.nan_to_num(tab.x[side][sample], nan=0.0),
                           rtol=0, atol=0)
        d = sfs.entry_d_e0(s, side)
        code = tab.stratum[side]
        assert np.all(code[np.flatnonzero(d > 0)] == 0)
        assert np.all(code[np.flatnonzero(d <= 0)] == 1)
        assert np.all(code[np.flatnonzero(np.isnan(d))] == 2)
        # nothing outside the censoring window is ever valid
        assert not tab.valid[side][s.n - sfs.CENSOR_TAIL:].any()


def test_extra_publication_lag_moves_the_censoring_tail_and_the_pseudo_episodes():
    """FRED publication lag: a FRED series (extra_lag = 1) needs one more session after
    t0. The null must censor, and evaluate, exactly as the observed data do."""
    idx = pd.bdate_range("2000-01-03", periods=3000)
    s = sfs.build_series(pd.Series(_sv_walk(3000, 5), index=idx), name="F", extra_lag=1)
    tab = sfn.build_tables(s)
    tail = sfs.censor_tail(1)
    assert tail == sfs.CENSOR_TAIL + 1
    assert len(tab.elig) == s.n - tail and tab.elig[-1] == s.n - 1 - tail
    for side in sfs.SIDES:
        sample = tab.elig[::113]
        ref = _independent_x(s, side, sample)          # the reference honours s.extra_lag
        assert np.array_equal(np.isfinite(ref), tab.valid[side][sample])
        assert np.allclose(np.nan_to_num(ref), np.nan_to_num(tab.x[side][sample]), rtol=0, atol=0)
        assert not tab.valid[side][s.n - tail:].any()
    # D uses the lagged entry close (t + 1 + L)
    d = sfs.entry_d_e0(s, "up_fade")
    assert np.all(tab.stratum["up_fade"][np.flatnonzero(d > 0)] == sfn.STRATUM_POS)
    o = sfn.observe_family([s], min_n=3)
    run = sfn.run_null_n1(o, B=5, seed=sfn.N1_SEED, collect_draws=True)
    for b in range(5):
        for key, anchors in run.draws[b].items():
            assert np.all(np.asarray(anchors) + tail <= s.n - 1)


def test_tested_set_is_the_groups_with_n_at_least_8_and_defined_t(obs):
    expected = tuple(k for k, g in obs.groups.items()
                     if g.n >= sfs.MIN_EPISODES and math.isfinite(g.t))
    assert obs.tested == expected
    assert isinstance(obs.tested, tuple)
    assert len(obs.tested) >= 6            # fixture is chosen to have several
    assert len(obs.tested) < len(obs.groups)   # ... and some untested groups
    for k in obs.tested:
        assert obs.groups[k].tested
    for k in set(obs.groups) - set(obs.tested):
        assert not obs.groups[k].tested


def test_min_n_is_a_parameter_but_defaults_to_the_pre_registered_8(universe):
    lo = sfn.observe_family(universe, min_n=3)
    hi = sfn.observe_family(universe, min_n=8)
    assert set(hi.tested) < set(lo.tested)
    assert sfn.observe_family(universe).min_n == 8


def test_tstar_is_the_max_t_over_the_tested_set_only(obs):
    t, key = obs.tstar()
    assert key in obs.tested
    assert t == max(obs.groups[k].t for k in obs.tested)


def test_tstar_ignores_an_untested_group_however_large_its_t(universe):
    o = sfn.observe_family(universe)
    untested = [k for k, g in o.groups.items() if not g.tested and g.n >= 2]
    assert untested, "fixture needs an untested group with a defined t"
    victim = o.groups[untested[0]]
    real_t, real_key = o.tstar()
    saved = victim.t
    victim.t = 1e9                       # would win if the max were over all groups
    try:
        t, key = o.tstar()
    finally:
        victim.t = saved
    assert key == real_key and t == real_t


def test_select_tstar_tie_rule_larger_n_then_lexicographic_order():
    # (t, n, key) with key = (series, side, trigger)
    a = (2.0, 10, ("B", "up_fade", "pct756"))
    b = (2.0, 12, ("C", "up_fade", "pct756"))       # same t, larger n wins
    assert sfn.select_tstar([a, b])[1] == ("C", "up_fade", "pct756")
    c = (2.0, 12, ("A", "up_fade", "pctexp"))       # same t and n: lexicographic first
    d = (2.0, 12, ("A", "down_fade", "z20@2.0"))
    assert sfn.select_tstar([b, c, d])[1] == ("A", "down_fade", "z20@2.0")
    e = (2.5, 8, ("Z", "up_fade", "pctexp"))        # a larger t beats everything
    assert sfn.select_tstar([a, b, c, e])[1] == ("Z", "up_fade", "pctexp")
    assert sfn.select_tstar([(float("nan"), 9, ("A", "up_fade", "pctexp"))]) is None
    assert sfn.select_tstar([]) is None
    # order of the input never matters
    assert sfn.select_tstar([d, c, b, a]) == sfn.select_tstar([a, b, c, d])


# ---------------------------------------------------------------------------
# 3. N1: random dates matched on exact valid n and on the D-sign strata
# ---------------------------------------------------------------------------

import dataclasses


def _with_groups(o, changes):
    """Copy of an ObservedFamily with some ObservedGroups tampered with:
    changes = {key: {field: value}}."""
    groups = dict(o.groups)
    for key, ch in changes.items():
        groups[key] = dataclasses.replace(groups[key], **ch)
    return dataclasses.replace(o, groups=groups)


def _check_anchors_for_group(o, key, anchors, quota=None):
    """Independent verification of one pseudo-episode set for one group:
    exact valid n, every anchor a valid uncensored traded episode, D strata
    equal to the requested quota, pairwise spacing at least 20."""
    g = o.groups[key]
    s = o.series[g.series]
    anchors = np.asarray(anchors, dtype=int)
    quota = quota if quota is not None else g.strata
    assert len(anchors) == sum(quota) == len(np.unique(anchors))
    assert np.all(anchors + sfs.censor_tail(s.extra_lag) <= s.n - 1) and anchors.min() >= 0
    assert np.all(np.isfinite(_independent_x(s, g.side, anchors)))     # valid data rules
    assert _independent_strata(s, g.side, anchors) == tuple(quota)     # D-sign match
    if len(anchors) > 1:
        assert np.diff(np.sort(anchors)).min() >= sfs.GAP_SESSIONS     # gap20 spacing


def test_draw_matched_anchors_walks_the_permutation_with_spacing_and_quota():
    n = 100
    valid = np.ones(n, dtype=bool)
    stratum = np.zeros(n, dtype=np.int8)
    perm = np.array([50, 51, 10, 70, 30, 90, 5])
    # 50 accepted (blocks 31..69), 51 blocked, 10 accepted (blocks up to 29),
    # 70 is exactly 20 away from 50: allowed, done at 3 anchors.
    got = sfn.draw_matched_anchors(perm, valid, stratum, (3, 0, 0), n)
    assert list(got) == [10, 50, 70]
    # 19 apart is NOT allowed
    got = sfn.draw_matched_anchors(np.array([50, 69, 31, 90]), valid, stratum, (2, 0, 0), n)
    assert list(got) == [50, 90]        # 69 blocked (19 away), 31 blocked (19 away)


def test_draw_matched_anchors_matches_strata_quota_exactly():
    n = 100
    valid = np.ones(n, dtype=bool)
    stratum = np.where(np.arange(n) < 50, sfn.STRATUM_POS, sfn.STRATUM_NONPOS).astype(np.int8)
    perm = np.array([5, 25, 60, 45, 80])
    got = sfn.draw_matched_anchors(perm, valid, stratum, (1, 1, 0), n)
    assert list(got) == [5, 60]         # 25 rejected: the D > 0 quota was full
    got = sfn.draw_matched_anchors(perm, valid, stratum, (2, 1, 0), n)
    assert list(got) == [5, 25, 60]


def test_draw_matched_anchors_skips_invalid_dates_and_fails_when_short():
    n = 100
    valid = np.ones(n, dtype=bool)
    valid[[10, 30]] = False
    stratum = np.zeros(n, dtype=np.int8)
    got = sfn.draw_matched_anchors(np.array([10, 30, 55, 80]), valid, stratum, (2, 0, 0), n)
    assert list(got) == [55, 80]
    # not enough valid, spaced dates: a failure is None, never a short draw
    assert sfn.draw_matched_anchors(np.array([10, 30, 55, 80]), valid, stratum, (3, 0, 0), n) is None
    # a stratum with no dates cannot be filled
    assert sfn.draw_matched_anchors(np.array([55, 80]), valid, stratum, (1, 0, 1), n) is None


def test_n1_replications_have_exact_valid_n_and_match_the_d_strata(obs):
    run = sfn.run_null_n1(obs, B=12, seed=sfn.N1_SEED, collect_draws=True)
    assert run.method == "N1" and run.B == 12
    assert run.groups == obs.tested
    assert run.t.shape == (12, len(obs.tested))
    assert run.n_failed == 0
    for b in range(12):
        for key in obs.tested:
            _check_anchors_for_group(obs, key, run.draws[b][key])
    nonuniform = sum(1 for key in obs.tested
                     if obs.groups[key].strata[0] != obs.groups[key].strata[1])
    assert nonuniform >= 3           # the strata are far from a coin flip: matching is not vacuous


def test_n1_null_t_equals_an_independent_recomputation_from_the_drawn_anchors(obs):
    run = sfn.run_null_n1(obs, B=6, seed=sfn.N1_SEED, collect_draws=True)
    for b in range(6):
        for j, key in enumerate(obs.tested):
            g = obs.groups[key]
            s = obs.series[g.series]
            anchors = np.sort(np.asarray(run.draws[b][key]))
            x = _independent_x(s, g.side, anchors)
            ref = sfs.cell_stats(x, x, s.dates[anchors], bootstrap=False).t
            assert run.t[b, j] == ref


def test_n1_statistic_is_the_max_over_the_frozen_tested_set_only(obs):
    run = sfn.run_null_n1(obs, B=10, seed=sfn.N1_SEED, collect_draws=True)
    assert run.t.shape[1] == len(obs.tested) < len(obs.groups)
    for b in range(10):
        assert set(run.draws[b]) == set(obs.tested)       # untested groups never drawn
        finite = run.t[b][np.isfinite(run.t[b])]
        assert run.stat[b] == finite.max()


def test_n1_is_deterministic_under_a_fixed_seed_and_differs_across_seeds(obs):
    r1 = sfn.run_null_n1(obs, B=8, seed=sfn.N1_SEED)
    r2 = sfn.run_null_n1(obs, B=8, seed=sfn.N1_SEED)
    r3 = sfn.run_null_n1(obs, B=8, seed=sfn.N1_SEED + 1)
    assert np.array_equal(r1.t, r2.t, equal_nan=True)
    assert np.array_equal(r1.stat, r2.stat)
    assert not np.array_equal(r1.stat, r3.stat)


def test_n1_groups_of_one_series_share_one_permutation_coupling(obs):
    """Nested triggers are coupled as in the data: every group of a series
    walks the SAME permutation, so identical quotas give identical anchors and
    single-stratum groups always share the first accepted date."""
    ka = ("A", "up_fade", "z20@2.0")
    kb = ("A", "up_fade", "z20@2.5")
    assert ka in obs.tested and kb in obs.tested
    ga, gb = obs.groups[ka], obs.groups[kb]
    twin = _with_groups(obs, {kb: {"strata": ga.strata, "n": ga.n}})
    run = sfn.run_null_n1(twin, B=10, seed=sfn.N1_SEED, collect_draws=True)
    for b in range(10):
        assert np.array_equal(np.sort(run.draws[b][ka]), np.sort(run.draws[b][kb]))
    # single-stratum groups of different size: the first valid date is shared
    one = _with_groups(obs, {ka: {"strata": (ga.n, 0, 0)}, kb: {"strata": (gb.n, 0, 0)}})
    run = sfn.run_null_n1(one, B=10, seed=sfn.N1_SEED, collect_draws=True)
    for b in range(10):
        assert set(run.draws[b][ka]) & set(run.draws[b][kb])
    # the natural nested pair overlaps heavily (far more than chance)
    run = sfn.run_null_n1(obs, B=20, seed=sfn.N1_SEED, collect_draws=True)
    shared = np.mean([len(set(run.draws[b][ka]) & set(run.draws[b][kb])) for b in range(20)])
    assert shared >= 3


def test_n1_series_are_independent_of_one_another(universe):
    both = sfn.observe_family(universe)
    only_a = sfn.observe_family([universe[0]])
    ra = sfn.run_null_n1(both, B=8, seed=sfn.N1_SEED, collect_draws=True)
    rb = sfn.run_null_n1(only_a, B=8, seed=sfn.N1_SEED, collect_draws=True)
    cols_a = [j for j, k in enumerate(both.tested) if k[0] == "A"]
    assert [both.tested[j] for j in cols_a] == list(only_a.tested)
    assert np.array_equal(ra.t[:, cols_a], rb.t, equal_nan=True)
    for b in range(8):
        for key in only_a.tested:
            assert np.array_equal(ra.draws[b][key], rb.draws[b][key])
    # ... and reversing the family order does not move any series' draws
    rev = sfn.observe_family(universe[::-1])
    rr = sfn.run_null_n1(rev, B=8, seed=sfn.N1_SEED)
    for j, key in enumerate(both.tested):
        jr = rev.tested.index(key)
        assert np.array_equal(ra.t[:, j], rr.t[:, jr], equal_nan=True)


def test_a_group_that_cannot_reach_n_makes_the_replication_minus_infinity(obs):
    key = ("A", "up_fade", "z20@2.0")
    g = obs.groups[key]
    # demand n episodes whose D is undefined: no valid date has an undefined D
    starved = _with_groups(obs, {key: {"strata": (0, 0, g.n)}})
    run = sfn.run_null_n1(starved, B=6, seed=sfn.N1_SEED)
    assert np.all(run.stat == -np.inf)                 # the WHOLE replication
    assert run.n_failed == 6
    a_idx = obs.series_names.index("A")
    assert run.fail[:, a_idx].all() and not run.fail[:, 1 - a_idx].any()
    # the other series' groups still produced finite t (informational only)
    other = [j for j, k in enumerate(obs.tested) if k[0] == "B"]
    assert np.isfinite(run.t[:, other]).any()
    # -inf is never at or above a finite T*
    assert sfn.p_value(1.0, run.stat) == pytest.approx(1.0 / 7.0)


# ---------------------------------------------------------------------------
# 4. N2: pure translation, no wrap, one common offset per series, skipping
# ---------------------------------------------------------------------------

def _brute_feasible(anchor_sets, n, step=21):
    """Reference: try every non-zero multiple of `step` one anchor at a time."""
    out = []
    for m in range(-(n // step) - 1, n // step + 2):
        d = m * step
        if d == 0:
            continue
        ok = True
        for anchors, valid in anchor_sets:
            for a in anchors:
                p = int(a) + d
                if not (0 <= p < n and valid[p]):
                    ok = False
        if ok:
            out.append(d)
    return out


def test_feasible_offsets_hand_computed_grid_and_no_wrap():
    n = 100
    valid = np.ones(n, dtype=bool)
    # anchors 10 and 30: 21, 42, 63 keep both inside; 84 pushes 30 to 114 (out
    # of range); -21 pushes 10 to -11 (out of range); 0 is excluded.
    assert list(sfn.feasible_offsets([([10, 30], valid)], n)) == [21, 42, 63]
    # an anchor at the far end cannot wrap around to the start: 95 + 21 = 116
    # would wrap to 16 under a modulo, which is NOT a translation
    assert len(sfn.feasible_offsets([([10, 95], valid)], n)) == 0
    # every returned offset is a multiple of 21, so any two differ by >= 21
    got = sfn.feasible_offsets([([50], valid)], n)
    assert list(got) == [-42, -21, 21, 42]
    assert np.all(np.diff(got) >= 21) and np.all(got % 21 == 0) and 0 not in got


def test_feasible_offsets_require_every_shifted_anchor_to_be_valid_for_its_group():
    n = 100
    ok = np.ones(n, dtype=bool)
    bad = ok.copy()
    bad[51] = False                                     # 30 + 21
    assert list(sfn.feasible_offsets([([10, 30], bad)], n)) == [42, 63]
    # two groups with different validity tables: the AND of both
    other = ok.copy()
    other[73] = False                                   # 10 + 63
    assert list(sfn.feasible_offsets([([10, 30], bad), ([10], other)], n)) == [42]


def test_feasible_offsets_agree_with_brute_force_on_random_tables():
    rng = np.random.default_rng(11)
    for _ in range(25):
        n = int(rng.integers(60, 400))
        valid = rng.random(n) > 0.1
        anchors = np.sort(rng.choice(n, size=int(rng.integers(1, 6)), replace=False))
        assert list(sfn.feasible_offsets([(anchors, valid)], n)) == _brute_feasible([(anchors, valid)], n)


def _clustered(o, start=1500, stop=2100, step=30):
    """Copy of an ObservedFamily whose tested groups sit in ONE compact block
    of valid dates, so plenty of translations are feasible."""
    changes = {}
    for key in o.tested:
        g = o.groups[key]
        tab = o.tables[g.series]
        pos = np.arange(start, stop, step)
        pos = pos[tab.valid[g.side][pos]]
        changes[key] = {"anchors": pos, "n": len(pos), "x": tab.x[g.side][pos],
                        "strata": (len(pos), 0, 0)}
    return _with_groups(o, changes)


@pytest.fixture(scope="module")
def cobs(obs):
    return _clustered(obs)


def test_natural_fixture_has_fewer_than_10_offsets_so_n2_is_skipped_and_reported(obs):
    for name in obs.series_names:
        sets = [(obs.groups[k].anchors, obs.tables[name].valid[k[1]])
                for k in obs.tested if k[0] == name]
        assert len(_brute_feasible(sets, obs.tables[name].n)) < sfn.N2_MIN_OFFSETS
    run = sfn.run_null_n2(obs, B=6, seed=sfn.N2_SEED)
    assert run.method == "N2"
    assert run.skipped == obs.series_names          # both series are reported as skipped
    assert not run.covered.any()
    assert np.all(run.stat == -np.inf)              # nothing to take a maximum over
    assert all(len(run.offsets[nm]) < sfn.N2_MIN_OFFSETS for nm in obs.series_names)


def test_n2_min_offsets_is_the_documented_threshold(cobs):
    # plenty of offsets: not skipped at the default threshold ...
    run = sfn.run_null_n2(cobs, B=4, seed=sfn.N2_SEED)
    assert run.skipped == () and run.covered.all()
    # ... but the same family IS skipped when the threshold is raised above
    # the number of feasible offsets (the count is reported either way)
    n_off = min(len(v) for v in run.offsets.values())
    assert n_off >= sfn.N2_MIN_OFFSETS
    run2 = sfn.run_null_n2(cobs, B=4, seed=sfn.N2_SEED, min_offsets=n_off + 1)
    assert set(run2.skipped) == {nm for nm, v in run.offsets.items() if len(v) < n_off + 1}
    assert len(run2.skipped) >= 1
    # exactly the threshold is enough ("fewer than 10" skips, 10 does not)
    run3 = sfn.run_null_n2(cobs, B=4, seed=sfn.N2_SEED, min_offsets=n_off)
    assert set(run3.skipped) == {nm for nm, v in run.offsets.items() if len(v) < n_off}
    assert any(len(v) == n_off for v in run.offsets.values())


def test_n2_is_a_pure_translation_that_preserves_count_and_spacing_and_never_wraps(cobs):
    B = 10
    run = sfn.run_null_n2(cobs, B=B, seed=sfn.N2_SEED, collect_draws=True)
    for name in cobs.series_names:
        offs = run.offsets[name]
        assert len(offs) >= sfn.N2_MIN_OFFSETS
        assert np.all(offs % sfn.N2_GRID_STEP == 0) and 0 not in offs
        s = cobs.series[name]
        chosen = run.draws[name]
        assert chosen.shape == (B,) and set(chosen) <= set(offs.tolist())
        for b in range(B):
            d = int(chosen[b])
            for j, key in enumerate(cobs.tested):
                if key[0] != name:
                    continue
                g = cobs.groups[key]
                shifted = g.anchors + d                      # ONE common offset
                assert shifted.min() >= 0                    # no wrap below ...
                assert shifted.max() + sfs.CENSOR_TAIL <= s.n - 1   # ... or above (censoring)
                assert len(shifted) == g.n                   # count preserved
                assert np.array_equal(np.diff(shifted), np.diff(g.anchors))   # spacing preserved
                x = _independent_x(s, g.side, shifted)
                assert np.all(np.isfinite(x))                # every shifted anchor is valid
                assert run.t[b, j] == sfs.cell_stats(x, x, s.dates[shifted], bootstrap=False).t


def test_n2_does_not_rematch_the_d_strata(cobs):
    """N2 is the different-regime sensitivity check: strata are NOT matched."""
    run = sfn.run_null_n2(cobs, B=30, seed=sfn.N2_SEED, collect_draws=True)
    differs = 0
    for name in cobs.series_names:
        s = cobs.series[name]
        for key in cobs.tested:
            if key[0] != name:
                continue
            g = cobs.groups[key]
            for d in set(run.draws[name].tolist()):
                if _independent_strata(s, g.side, g.anchors + d) != g.strata:
                    differs += 1
    assert differs > 0


def test_n2_one_offset_per_series_and_series_independent(universe, cobs):
    a_only = _clustered(sfn.observe_family([universe[0]]))
    r_both = sfn.run_null_n2(cobs, B=12, seed=sfn.N2_SEED, collect_draws=True)
    r_a = sfn.run_null_n2(a_only, B=12, seed=sfn.N2_SEED, collect_draws=True)
    assert np.array_equal(r_both.draws["A"], r_a.draws["A"])       # independent of B's presence
    assert not np.array_equal(r_both.draws["A"], r_both.draws["B"])  # each series its own stream
    cols_a = [j for j, k in enumerate(cobs.tested) if k[0] == "A"]
    assert np.array_equal(r_both.t[:, cols_a], r_a.t, equal_nan=True)


def test_n2_statistic_is_the_max_over_covered_tested_groups_only(obs, cobs):
    # series A clustered (covered), series B natural (skipped)
    changes = {k: {"anchors": cobs.groups[k].anchors, "n": cobs.groups[k].n,
                   "x": cobs.groups[k].x, "strata": cobs.groups[k].strata}
               for k in obs.tested if k[0] == "A"}
    mixed = _with_groups(obs, changes)
    run = sfn.run_null_n2(mixed, B=10, seed=sfn.N2_SEED)
    assert run.skipped == ("B",)
    cov = np.array([k[0] == "A" for k in mixed.tested])
    assert np.array_equal(run.covered, cov)
    for b in range(10):
        assert run.stat[b] == run.t[b][cov][np.isfinite(run.t[b][cov])].max()
    assert np.all(np.isnan(run.t[:, ~cov]))
    assert run.t.shape[1] == len(mixed.tested)


def test_n2_is_deterministic_under_fixed_seed(cobs):
    r1 = sfn.run_null_n2(cobs, B=8, seed=sfn.N2_SEED)
    r2 = sfn.run_null_n2(cobs, B=8, seed=sfn.N2_SEED)
    r3 = sfn.run_null_n2(cobs, B=8, seed=sfn.N2_SEED + 1)
    assert np.array_equal(r1.t, r2.t, equal_nan=True)
    assert np.array_equal(r1.stat, r2.stat)
    assert not np.array_equal(r1.stat, r3.stat)


# ---------------------------------------------------------------------------
# 5. p-values, the headline, adjusted p, determinism
# ---------------------------------------------------------------------------

def test_p_value_formula_is_one_plus_k_over_one_plus_b_with_inclusive_ties():
    null = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
    assert sfn.p_value(3.0, null) == pytest.approx(4.0 / 6.0)     # 3, 3, 4 count: ties INCLUDED
    assert sfn.p_value(3.5, null) == pytest.approx(2.0 / 6.0)
    assert sfn.p_value(10.0, null) == pytest.approx(1.0 / 6.0)    # never below 1 / (1 + B)
    assert sfn.p_value(0.0, null) == pytest.approx(1.0)
    assert sfn.p_value(1.0, np.array([-np.inf, -np.inf, 5.0])) == pytest.approx(2.0 / 4.0)
    assert math.isnan(sfn.p_value(float("nan"), null))
    assert sfn.p_value(1.0, np.array([])) == pytest.approx(1.0)   # (1 + 0) / (1 + 0)


def test_monte_carlo_standard_error():
    assert sfn.mc_se(0.1, 100) == pytest.approx(math.sqrt(0.09 / 100))
    assert sfn.mc_se(1.0, 100) == 0.0
    assert math.isnan(sfn.mc_se(float("nan"), 100))


def test_headline_is_the_n1_p_alone_and_n2_cannot_be_passed_in():
    """HEADLINE p = the N1 value; N2 is a sensitivity."""
    assert sfn.headline_p(0.05) == 0.05
    assert sfn.headline_p(0.20) == 0.20
    assert math.isnan(sfn.headline_p(float("nan")))
    # a two-argument caller (the v3 max(N1, N2) headline definition) fails LOUDLY
    with pytest.raises(TypeError):
        sfn.headline_p(0.05, 0.20)
    assert sfn.N2_SENSITIVITY_LABEL == "sensitivity, not gating"


def test_decision_language():
    assert sfn.decision(0.10).startswith("confirmatory evidence")      # at or below 0.10
    assert sfn.decision(0.0999).startswith("confirmatory evidence")
    assert sfn.decision(0.1001).startswith("inconclusive at the study")
    assert "not" in sfn.decision(0.5).lower() or "NOT" in sfn.decision(0.5)   # never worded as "no edge"
    assert sfn.decision(float("nan")).startswith("no tested group")


def test_confirmatory_test_ties_the_pieces_together(obs):
    res = sfn.run_confirmatory_test(obs, B=30)
    t, key = obs.tstar()
    assert res.tstar == t and res.tstar_key == key
    assert res.p_n1 == sfn.p_value(t, res.n1.stat)
    assert res.n1.B == 30 and res.n2.B == 30
    assert res.n1.seed == sfn.N1_SEED and res.n2.seed == sfn.N2_SEED
    assert res.mc_se_n1 == sfn.mc_se(res.p_n1, 30)
    assert 1.0 / 31.0 <= res.p_n1 <= 1.0
    # natural fixture: N2 is skipped everywhere -> N1 alone, and the report says so
    assert res.p_n2 is None
    assert res.p_headline == res.p_n1
    assert "N1 alone" in res.headline_rule and "v3.1" in res.headline_rule
    assert res.n2_skipped == list(obs.series_names)
    assert set(res.n2_offsets) == set(obs.series_names)
    assert res.n_failed_n1 == 0


def test_headline_is_n1_alone_even_when_n2_is_available(cobs):
    res = sfn.run_confirmatory_test(cobs, B=40)
    assert res.p_n2 is not None and res.n2_skipped == []
    assert res.p_n2 == sfn.p_value(res.tstar, res.n2.stat)            # N2 is still computed and reported
    assert res.p_headline == res.p_n1                                  # ... but never enters the headline
    assert res.p_n2 != res.p_n1                                        # (fixture: the two really differ)
    assert res.mc_se_headline == res.mc_se_n1 == sfn.mc_se(res.p_n1, 40)
    assert "N1 alone" in res.headline_rule and "max" not in res.headline_rule
    assert res.decision == sfn.decision(res.p_n1)
    assert all(v >= sfn.N2_MIN_OFFSETS for v in res.n2_offsets.values())


def test_headline_and_decision_do_not_move_when_only_n2_changes(cobs, monkeypatch):
    a = sfn.run_confirmatory_test(cobs, B=30)
    for delta in (1, 2, 3, 4):                                    # other N2 streams
        b = sfn.run_confirmatory_test(cobs, B=30, n2_seed=sfn.N2_SEED + delta)
        assert (a.p_headline, a.mc_se_headline, a.decision) == (b.p_headline, b.mc_se_headline,
                                                               b.decision)
        for k in a.per_series:
            assert a.per_series[k]["p_headline"] == b.per_series[k]["p_headline"]
    # ... with N2 switched off entirely (threshold above any offset count) ...
    c = sfn.run_confirmatory_test(cobs, B=30, n2_min_offsets=10 ** 6)
    assert c.p_n2 is None and c.p_headline == a.p_headline
    # ... and with an N2 null pushed to BOTH extremes: p_n2 far above, then far below,
    # the N1 p -- the headline is the N1 p either way (the test has teeth in both directions)
    orig = sfn.run_null_n2

    def forced(value):
        def run(*args, **kw):
            r = orig(*args, **kw)
            r.stat = np.full_like(r.stat, value)
            return r
        return run
    for value, expect_above in ((1e9, True), (-np.inf, False)):
        monkeypatch.setattr(sfn, "run_null_n2", forced(value))
        d = sfn.run_confirmatory_test(cobs, B=30)
        assert (d.p_n2 > d.p_n1) if expect_above else (d.p_n2 < d.p_n1)
        assert d.p_headline == d.p_n1 == a.p_n1 and d.decision == a.decision
        for k, item in d.per_series.items():
            assert item["p_headline"] == item["p_n1"] == a.per_series[k]["p_n1"]
        sub = sfn.family_subset_p(d, ["A", "B"])
        assert sub["p_headline"] == sub["p_n1"]
        rec_line = [ln for ln in d.report_lines() if ln.startswith("Headline")][0]
        assert f"{d.p_n1:.4f}" in rec_line


def test_headline_is_n1_when_the_tstar_series_has_no_n2(obs, cobs):
    # A is clustered (N2 available), B natural (N2 skipped); make B hold T*
    changes = {k: {"anchors": cobs.groups[k].anchors, "n": cobs.groups[k].n,
                   "x": cobs.groups[k].x, "strata": cobs.groups[k].strata}
               for k in obs.tested if k[0] == "A"}
    mixed = _with_groups(obs, changes)
    b_key = next(k for k in mixed.tested if k[0] == "B")
    mixed = _with_groups(mixed, {b_key: {"t": 50.0}})
    res = sfn.run_confirmatory_test(mixed, B=20)
    assert res.tstar_key == b_key
    assert res.n2_skipped == ["B"]
    assert res.p_n2 is None and res.p_headline == res.p_n1
    assert "N1 alone" in res.headline_rule


def test_confirmatory_test_is_deterministic_and_seeded(obs):
    a = sfn.run_confirmatory_test(obs, B=25)
    b = sfn.run_confirmatory_test(obs, B=25)
    c = sfn.run_confirmatory_test(obs, B=25, n1_seed=sfn.N1_SEED + 5)
    assert a.to_dict() == b.to_dict()
    assert np.array_equal(a.n1.stat, b.n1.stat)
    assert not np.array_equal(a.n1.stat, c.n1.stat)


def test_adjusted_p_values_use_the_same_global_null_max_distribution(obs):
    res = sfn.run_confirmatory_test(obs, B=40)
    # per series: the series maximum against the SAME global null maxima
    for name, item in res.per_series.items():
        ts = [obs.groups[k].t for k in obs.tested if k[0] == name]
        assert item["t_max"] == max(ts)
        assert item["p_n1"] == sfn.p_value(item["t_max"], res.n1.stat)
        assert item["p_n1"] >= res.p_n1                     # a subset maximum cannot beat T*
        assert item["n_tested_groups"] == len(ts)
    for side, item in res.per_side.items():
        ts = [obs.groups[k].t for k in obs.tested if k[1] == side]
        assert item["t_max"] == max(ts)
        assert item["p_n1"] == sfn.p_value(item["t_max"], res.n1.stat)
    for (name, side), item in res.per_pair.items():
        ts = [obs.groups[k].t for k in obs.tested if k[0] == name and k[1] == side]
        assert item["t_max"] == max(ts)
    for key, item in res.per_group.items():
        assert item["t_max"] == obs.groups[key].t
    # the pair that holds T* has the headline p exactly
    pair = (res.tstar_key[0], res.tstar_key[1])
    assert res.per_pair[pair]["p_n1"] == res.p_n1
    # adjusted p values are labeled descriptive
    assert res.per_series and "descriptive" in sfn.ADJUSTED_P_LABEL.lower()


def test_adjusted_headline_is_n1_and_the_n2_p_is_a_separate_sensitivity_field(cobs):
    res = sfn.run_confirmatory_test(cobs, B=30)
    items = (list(res.per_series.values()) + list(res.per_side.values())
             + list(res.per_pair.values()) + list(res.per_group.values()))
    for item in items:
        assert item["p_headline"] == item["p_n1"]                      # headline = N1, per item too
        assert "p_n2" not in item                                      # the N2 p has its own labelled key
        assert item["p_n2_sensitivity"] is not None                    # everything covered here
        assert item["p_n2_sensitivity"] == sfn.p_value(item["t_max"], res.n2.stat)
    assert any(it["p_n2_sensitivity"] != it["p_n1"] for it in items)   # non-vacuous


def test_result_serialises_to_strict_json_and_flat_frame(obs):
    import json
    res = sfn.run_confirmatory_test(obs, B=12)
    text = json.dumps(res.to_dict(), allow_nan=False)      # no NaN / inf leaks
    assert "tstar" in text and "p_headline" in text
    fr = res.to_frame()
    assert len(fr) == len(obs.groups)
    assert set(fr.columns) >= {"series", "side", "trigger", "n_kept", "n", "mean_x", "t",
                               "tested", "adj_p_n1", "adj_p_headline", "adj_p_n2_sensitivity"}
    assert fr["tested"].sum() == len(obs.tested)


def test_n2_is_labelled_sensitivity_not_gating_in_dict_frame_and_report_lines(cobs, obs):
    import json
    res = sfn.run_confirmatory_test(cobs, B=30)
    d = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    # the headline block: N1 alone
    assert d["p_headline"] == d["p_n1"] and "N1 alone" in d["headline_rule"]
    assert d["mc_se_headline"] == d["mc_se_n1"]
    # N2 lives ONLY under one labelled key, never at the top level
    for stale in ("p_n2", "mc_se_n2", "n2_offsets", "n2_skipped", "n2_coverage", "p_n2_covered",
                  "tstar_n2_covered", "n2_seed", "n2_null_max_quantiles"):
        assert stale not in d, stale
    n2 = d["n2_sensitivity"]
    assert n2["label"] == "sensitivity, not gating" and n2["gating"] is False
    assert n2["p_n2"] == pytest.approx(res.p_n2) and n2["mc_se_n2"] == pytest.approx(res.mc_se_n2)
    assert n2["available_for_tstar"] is True
    assert n2["offsets"] == res.n2_offsets and n2["skipped"] == res.n2_skipped
    assert n2["p_n2_covered"] == pytest.approx(res.p_n2_covered)
    assert n2["tstar_n2_covered"] == pytest.approx(res.tstar_n2_covered)
    assert set(n2) >= {"coverage", "seed", "null_max_quantiles"}
    for grp in ("per_series", "per_side", "per_pair", "per_group"):
        for item in d[grp].values():
            assert "p_n2" not in item and "p_n2_sensitivity" in item
    # the frame: N2 column is explicitly labelled, no bare adj_p_n2
    fr = res.to_frame()
    assert "adj_p_n2_sensitivity" in fr.columns and "adj_p_n2" not in fr.columns
    tested = fr[fr["tested"]]
    assert (tested["adj_p_headline"] == tested["adj_p_n1"]).all()
    # report lines: every N2 line carries the label, the headline line does not mention N2
    for r in (res, sfn.run_confirmatory_test(obs, B=12)):
        lines = r.report_lines()
        n2_lines = [ln for ln in lines if ln.startswith("N2")]
        assert n2_lines and all("sensitivity, not gating" in ln for ln in n2_lines)
        head = [ln for ln in lines if ln.startswith("Headline")]
        assert len(head) == 1 and "N1 alone" in head[0] and "N2" not in head[0]
        assert not any("sensitivity, not gating" in ln for ln in lines if not ln.startswith("N2"))


def test_confirmatory_test_from_series_builds_the_observed_family(universe):
    res = sfn.confirmatory_test(universe, B=10)
    assert res.observed.series_names == ("A", "B")
    assert res.tstar_key in res.observed.tested


def test_n2_like_for_like_p_uses_the_covered_groups_only(obs, cobs):
    # everything covered: like-for-like equals the specified N2 p
    res = sfn.run_confirmatory_test(cobs, B=30)
    assert res.p_n2_covered == res.p_n2
    assert res.tstar_n2_covered == res.tstar
    # nothing covered (natural fixture): no like-for-like p either
    res0 = sfn.run_confirmatory_test(obs, B=12)
    assert res0.p_n2_covered is None and res0.tstar_n2_covered is None
    # A covered, B skipped, and T* sits in B: the specified N2 p does not apply
    # but the like-for-like one compares A's maximum with the null over A only
    changes = {k: {"anchors": cobs.groups[k].anchors, "n": cobs.groups[k].n,
                   "x": cobs.groups[k].x, "strata": cobs.groups[k].strata}
               for k in obs.tested if k[0] == "A"}
    mixed = _with_groups(obs, changes)
    b_key = next(k for k in mixed.tested if k[0] == "B")
    mixed = _with_groups(mixed, {b_key: {"t": 50.0}})
    res1 = sfn.run_confirmatory_test(mixed, B=20)
    assert res1.p_n2 is None
    tcov = max(mixed.groups[k].t for k in mixed.tested if k[0] == "A")
    assert res1.tstar_n2_covered == tcov
    assert res1.p_n2_covered == sfn.p_value(tcov, res1.n2.stat)


def test_report_lines_state_the_key_facts_and_the_skips(obs):
    res = sfn.run_confirmatory_test(obs, B=12)
    text = " | ".join(res.report_lines())
    assert "T*" in text and "N1" in text and "Headline" in text and "Monte Carlo" in text
    for nm in obs.series_names:
        assert f"N2 skipped for {nm}" in text          # the required per-series skip line
    assert "feasible" in text
    assert "sensitivity, not gating" in text


def test_no_dollar_figure_anywhere_in_the_null_module_its_tests_or_its_script():
    import re
    pat = re.compile(r"\$\s?\d")
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, "..", "mining", "spike_fade_null.py"),
             os.path.join(here, "test_spike_fade_null.py"),
             os.path.join(here, "..", "..", "scripts", "validate_spike_fade_null.py")]
    for p in paths:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                assert not pat.search(fh.read()), p


# ---------------------------------------------------------------------------
# 6. The same-cell N1 null distribution for the exploratory map
# ---------------------------------------------------------------------------

def test_map_constants():
    assert sfn.MAP_B == 200 and sfn.MAP_SEED == 2026_0926


def test_map_null_shape_defaults_and_quantile(universe):
    s = universe[0]
    m = sfn.n1_cell_null(s, "z20@2.0", "up_fade", B=10)
    n_cells = len(sfs.CELL_GRID)
    assert m.S.shape == (n_cells, 10) and m.n_traded.shape == (n_cells, 10)
    assert m.cells == sfs.CELL_GRID and m.seed == sfn.MAP_SEED and m.B == 10
    for i in range(n_cells):
        row = m.S[i][np.isfinite(m.S[i])]
        if len(row):
            assert m.q90[i] == pytest.approx(np.percentile(row, 90, method="linear"))
        else:
            assert math.isnan(m.q90[i])
    assert m.q90_of(sfs.PRIMARY_CELL) == m.q90[sfs.CELL_INDEX[sfs.PRIMARY_CELL]]
    import inspect
    sig = inspect.signature(sfn.n1_cell_null)
    assert sig.parameters["B"].default == 200 and sig.parameters["seed"].default == 2026_0926


def test_map_null_pseudo_episodes_match_valid_n_and_strata_exactly(universe):
    s = universe[0]
    o = sfn.observe_family([s])
    for trig, side in (("z20@2.0", "up_fade"), ("z20@2.0", "down_fade"), ("pct756", "down_fade")):
        g = o.groups[(s.name, side, trig)]
        m = sfn.n1_cell_null(s, trig, side, B=8, collect_draws=True)
        assert m.n_target == g.n and m.strata == g.strata and m.n_failed == 0
        assert m.anchors.shape == (8, g.n)
        for b in range(8):
            _check_anchors_for_group(o, (s.name, side, trig), m.anchors[b])


def test_map_null_equals_a_slow_reference_cell_by_cell(universe):
    s = universe[0]
    cells = sfs.CELL_GRID[::7]                                    # E0 and E1, TP and SL variants
    m = sfn.n1_cell_null(s, "z20@2.0", "up_fade", cells=cells, B=6, collect_draws=True)
    sigma = s.context().sigma
    for ci, cell in enumerate(cells):
        for b in range(6):
            xs = []
            for a in m.anchors[b]:
                o = sfs.simulate_from_series(s, int(a), "up_fade", cell)
                if o.status == sfs.STATUS_TRADED:
                    xs.append(sfs.episode_x(o, float(sigma[int(a)])))
            assert m.n_traded[ci, b] == len(xs)
            if xs:
                assert m.S[ci, b] == pytest.approx(float(np.mean(xs)), abs=1e-12)
            else:
                assert math.isnan(m.S[ci, b])


def test_map_null_is_deterministic_and_seeded(universe):
    s = universe[0]
    a = sfn.n1_cell_null(s, "z20@2.5", "up_fade", B=6)
    b = sfn.n1_cell_null(s, "z20@2.5", "up_fade", B=6)
    c = sfn.n1_cell_null(s, "z20@2.5", "up_fade", B=6, seed=sfn.MAP_SEED + 1)
    assert np.array_equal(a.S, b.S, equal_nan=True) and np.array_equal(a.q90, b.q90, equal_nan=True)
    assert not np.array_equal(a.S, c.S, equal_nan=True)
    # a different group draws from its own stream
    d = sfn.n1_cell_null(s, "z20@2.0", "down_fade", B=6)
    assert not np.array_equal(a.S, d.S, equal_nan=True)


def test_map_null_group_with_no_valid_episodes_is_reported_not_faked(universe):
    s = universe[1]
    o = sfn.observe_family([s])
    assert o.groups[("B", "down_fade", "pctexp")].n == 0      # fixture: no episode at all
    m = sfn.n1_cell_null(s, "pctexp", "down_fade", B=5)
    assert m.n_target == 0 and np.all(np.isnan(m.S)) and np.all(np.isnan(m.q90))


def test_map_null_series_helper_covers_all_eight_groups(universe):
    s = universe[1]
    out = sfn.n1_cell_null_series(s, cells=[sfs.PRIMARY_CELL], B=5)
    assert set(out) == {(t, sd) for t in sfs.TRIGGERS for sd in sfs.SIDES}
    single = sfn.n1_cell_null(s, "pct756", "up_fade", cells=[sfs.PRIMARY_CELL], B=5)
    assert np.array_equal(out[("pct756", "up_fade")].S, single.S, equal_nan=True)


def test_map_null_failure_replications_are_excluded_and_counted(universe):
    s = universe[0]
    o = sfn.observe_family([s])
    g = o.groups[(s.name, "up_fade", "z20@2.0")]
    tampered = dataclasses.replace(g, strata=(0, 0, g.n))         # cannot be filled
    m = sfn.n1_cell_null(s, "z20@2.0", "up_fade", B=4, observed=tampered)
    assert m.n_failed == 4 and np.all(np.isnan(m.S))


# ---------------------------------------------------------------------------
# 7. The synthetic calibration harness
# ---------------------------------------------------------------------------

def test_calibration_constants_follow_the_spec():
    assert sfn.CALIB_SERIES == 13 and sfn.CALIB_SESSIONS == 5000
    assert sfn.CALIB_ALPHA == pytest.approx(0.10)
    assert sfn.SIZE_WINDOW == (0.06, 0.14)
    assert sfn.CALIB_EFFECTS == (0.25, 0.5, 1.0)
    assert sfn.POWER_REQUIRED_AT == (1.0, 0.5)          # at least 50 percent power at 1.0
    assert sfn.CALIB_DF_RANGE == (3.5, 5.0)             # Student t, 3.5 to 5 degrees of freedom


def test_series_parameters_are_drawn_within_the_spec_ranges_and_deterministically():
    for k in range(20):
        p = sfn.calibration_params(7, k)
        assert 3.5 <= p["df"] <= 5.0
        assert 0.90 <= p["phi"] < 1.0 and p["eta"] > 0 and p["vol"] > 0
    assert sfn.calibration_params(7, 3) == sfn.calibration_params(7, 3)
    assert sfn.calibration_params(7, 3) != sfn.calibration_params(7, 4)
    assert sfn.calibration_params(7, 3) != sfn.calibration_params(8, 3)


def test_make_dataset_is_13_independent_fat_tailed_series_by_default():
    ds = sfn.make_dataset(101, n_series=13, n_sessions=5000)
    assert list(ds) == [f"S{i:02d}" for i in range(13)]
    rets = []
    for nm, s in ds.items():
        assert len(s) == 5000 and (s > 0).all() and s.index[-1] <= sfs.DISCOVERY_CUTOFF
        r = np.diff(np.log(s.to_numpy()))
        rets.append(r)
        z = (r - r.mean()) / r.std()
        assert np.mean(z ** 4) > 3.5                     # fat tails (excess kurtosis)
    R = np.corrcoef(np.vstack(rets))
    assert np.abs(R[np.triu_indices(13, 1)]).max() < 0.08    # independent series
    again = sfn.make_dataset(101, n_series=13, n_sessions=5000)
    assert all(np.array_equal(ds[k].to_numpy(), again[k].to_numpy()) for k in ds)
    other = sfn.make_dataset(102, n_series=2, n_sessions=300)
    assert not np.array_equal(ds["S00"].to_numpy()[:300], other["S00"].to_numpy())


def test_a_series_does_not_depend_on_how_many_series_the_dataset_has():
    small = sfn.make_dataset(5, n_series=2, n_sessions=400)
    big = sfn.make_dataset(5, n_series=5, n_sessions=400)
    assert np.array_equal(small["S01"].to_numpy(), big["S01"].to_numpy())


def test_injection_with_zero_effect_is_the_identity():
    ds = sfn.make_dataset(11, n_series=1, n_sessions=2500)
    out, info = sfn.inject_post_spike_reversion(ds["S00"], 0.0)
    assert np.array_equal(out.to_numpy(), ds["S00"].to_numpy())
    assert info["delta"] == 0.0


def test_injection_adds_the_stated_reversion_to_every_injected_trade_exactly():
    close = sfn.make_dataset(11, n_series=1, n_sessions=2500)["S00"]
    delta = 0.7
    out, info = sfn.inject_post_spike_reversion(close, delta)
    assert info["n_injected"] >= 10 and info["hold"] == 21 and info["max_error"] < 1e-8
    lp0, lp1 = np.log(close.to_numpy()), np.log(out.to_numpy())
    a_min = min(a for a, _ in info["anchors"])
    assert np.array_equal(lp0[:a_min + 2], lp1[:a_min + 2])            # nothing before the first window
    for a, sgn in info["anchors"]:
        # the primary cell enters at the close of a + 1 and exits at the close of a + 22
        shift = (lp1[a + 22] - lp1[a + 1]) - (lp0[a + 22] - lp0[a + 1])
        assert shift == pytest.approx(-sgn * delta * info["sigma21"][a], abs=1e-8)
    assert {s for _, s in info["anchors"]} == {+1, -1}                 # both sides are injected


def test_injection_finds_a_fixed_point_so_flagged_anchors_carry_the_effect():
    close = sfn.make_dataset(11, n_series=1, n_sessions=5000)["S00"]
    _, one = sfn.inject_post_spike_reversion(close, 1.0, iterations=1)
    _, many = sfn.inject_post_spike_reversion(close, 1.0)
    assert many["iterations"] == sfn.INJECT_ITERATIONS
    assert many["coincidence"] > one["coincidence"]                    # iteration helps ...
    assert many["coincidence"] >= 0.75                                 # ... a lot


def test_injection_creates_the_stated_positive_fade_edge():
    close = sfn.make_dataset(11, n_series=1, n_sessions=5000)["S00"]
    out, info = sfn.inject_post_spike_reversion(close, 1.0)
    inj = sfn.observe_family([sfs.build_series(out, name="S00")])
    means = [inj.groups[k].mean_x for k in inj.tested]
    assert len(means) >= 4
    assert np.mean(means) > 0.7                    # about the injected 1.0 sigma21 per trade
    for side in sfs.SIDES:
        assert inj.groups[("S00", side, "z20@2.0")].mean_x > 0.4


def test_subset_p_equals_the_full_result_for_the_full_family_and_shrinks_the_family(universe):
    res = sfn.confirmatory_test(universe, B=30)
    full = sfn.family_subset_p(res, [s.name for s in universe])
    assert full["p_n1"] == res.p_n1 and full["p_headline"] == res.p_headline
    assert full["p_headline"] == full["p_n1"]                          # headline = N1
    assert full["tstar"] == res.tstar and full["tstar_key"] == res.tstar_key
    sub = sfn.family_subset_p(res, ["A"])
    ts = [res.observed.groups[k].t for k in res.observed.tested if k[0] == "A"]
    assert sub["tstar"] == max(ts) and sub["tstar_key"][0] == "A"
    # a subset family has a smaller null maximum, so its p can only be smaller or equal for the same T*
    assert sub["p_n1"] == sfn.p_value(sub["tstar"], res.n1.statistic(
        np.array([k[0] == "A" for k in res.observed.tested])))


def test_subset_headline_is_n1_even_when_n2_covers_the_subset(cobs):
    res = sfn.run_confirmatory_test(cobs, B=30)
    for names in (["A"], ["A", "B"]):
        sub = sfn.family_subset_p(res, names)
        assert sub["p_n2"] is not None and sub["p_headline"] == sub["p_n1"]
    assert sfn.family_subset_p(res, ["A"])["p_n2"] != sfn.family_subset_p(res, ["A"])["p_n1"]


def test_min_detectable_effect_interpolates_the_power_curve():
    pts = [(0.25, 0.10), (0.5, 0.30), (1.0, 0.70)]
    assert sfn.min_detectable_effect(pts, 0.5) == pytest.approx(0.75)
    assert sfn.min_detectable_effect(pts, 0.3) == pytest.approx(0.5)
    assert sfn.min_detectable_effect(pts, 0.8) is None              # not reached in the tested range
    assert sfn.min_detectable_effect(pts, 0.05) == pytest.approx(0.25)   # already met at the first point
    assert sfn.min_detectable_effect([], 0.5) is None


def test_summarize_size_does_not_judge_a_rate_from_too_few_datasets():
    recs = [{"p_n1": 0.05, "p_n2": 0.5, "p_n2_covered": None, "p_headline": 0.5}] * 40
    out = sfn.summarize_size(recs)
    assert out["p_n2"]["n_available"] == 40 and out["p_n2"]["in_window"] is None   # < 50: not judged
    assert out["n2_judged"] is False
    assert out["p_n2_covered"]["n_available"] == 0 and out["p_n2_covered"]["in_window"] is None
    many = ([{"p_n1": 0.1, "p_n2": 0.1, "p_n2_covered": 0.1, "p_headline": 0.1}] * 8
            + [{"p_n1": 0.1, "p_n2": 0.9, "p_n2_covered": 0.9, "p_headline": 0.1}] * 1
            + [{"p_n1": 0.9, "p_n2": 0.9, "p_n2_covered": 0.9, "p_headline": 0.9}] * 91)
    out = sfn.summarize_size(many)
    assert out["n2_judged"] is True and out["p_n2"]["in_window"] is True
    assert out["p_n2_covered"]["in_window"] is True and out["p_n2_covered"]["rate"] == 0.08
    assert out["all_in_window"] is True


def test_summarize_size_requires_n1_and_headline_but_only_reports_n2():
    """The window is REQUIRED for N1 and the headline (= N1);
    N2's rate is reported, not required."""
    def rec(p1, p2):
        return {"p_n1": p1, "p_n2": p2, "p_n2_covered": p2, "p_headline": p1}
    out = sfn.summarize_size([rec(0.05, 0.9)] * 9 + [rec(0.9, 0.9)] * 91)      # N1 rate 0.09, N2 rate 0
    assert out["p_n1"]["in_window"] is True and out["p_headline"]["in_window"] is True
    assert out["p_n2"]["in_window"] is False and out["n2_judged"] is True       # N2 judged, out of window
    assert out["all_in_window"] is True                                         # ... and it does not matter
    assert (out["p_n1"]["gating"], out["p_headline"]["gating"]) == (True, True)
    assert (out["p_n2"]["gating"], out["p_n2_covered"]["gating"]) == (False, False)
    assert "sensitivity, not gating" in out["n2_label"]
    # the other way round: N2 perfect, N1 (= headline) out of window -> the requirement fails
    bad = sfn.summarize_size([rec(0.05, 0.05)] * 30 + [rec(0.9, 0.9)] * 70)     # N1 rate 0.30, N2 rate 0.30
    bad2 = sfn.summarize_size([{"p_n1": 0.05, "p_n2": (0.05 if k < 8 else 0.9),
                                "p_n2_covered": (0.05 if k < 8 else 0.9), "p_headline": 0.05}
                               for k in range(100)])                               # N1 rate 1.0, N2 rate 0.08
    assert bad["all_in_window"] is False
    assert bad2["p_n2"]["in_window"] is True and bad2["all_in_window"] is False
    # N1 and the headline are BOTH required: whichever one leaves the window fails the requirement
    def mixed(n_n1, n_head):
        return [{"p_n1": 0.05 if k < n_n1 else 0.9, "p_n2": 0.9, "p_n2_covered": 0.9,
                 "p_headline": 0.05 if k < n_head else 0.9} for k in range(100)]
    assert sfn.summarize_size(mixed(9, 9))["all_in_window"] is True
    assert sfn.summarize_size(mixed(50, 9))["all_in_window"] is False          # N1 out, headline in
    assert sfn.summarize_size(mixed(9, 50))["all_in_window"] is False          # N1 in, headline out


def test_summarize_size_counts_rejections_and_checks_the_window():
    recs = ([{"p_n1": 0.05, "p_n2": 0.5, "p_headline": 0.05}] * 3 +
            [{"p_n1": 0.5, "p_n2": None, "p_headline": 0.5}] * 17)
    out = sfn.summarize_size(recs, alpha=0.10)
    assert out["n_datasets"] == 20
    assert out["p_n1"]["rejections"] == 3 and out["p_n1"]["rate"] == pytest.approx(0.15)
    assert out["p_n1"]["in_window"] is False                        # 0.15 is above 0.14
    assert out["p_n2"]["n_available"] == 3 and out["p_n2"]["rejections"] == 0
    assert out["p_headline"]["rejections"] == 3 and out["p_headline"]["in_window"] is False
    assert out["p_n1"]["se"] == pytest.approx(math.sqrt(0.15 * 0.85 / 20))
    ok = sfn.summarize_size([{"p_n1": 0.1, "p_n2": 0.1, "p_headline": 0.1}] * 10
                            + [{"p_n1": 0.9, "p_n2": 0.9, "p_headline": 0.9}] * 90)
    assert ok["p_headline"]["in_window"] is True and ok["all_in_window"] is True   # p = alpha counts


def test_summarize_power_builds_the_curve_and_the_requirement():
    def rec(p):
        return {"p_n1": p, "p_n2": p, "p_headline": p, "realized_mean_x": 1.0}
    by = {0.25: [rec(0.05)] * 1 + [rec(0.5)] * 9,
          0.5: [rec(0.05)] * 3 + [rec(0.5)] * 7,
          1.0: [rec(0.05)] * 7 + [rec(0.5)] * 3}
    out = sfn.summarize_power(by, alpha=0.10)
    assert out["curve"][1.0]["power_headline"] == pytest.approx(0.7)
    assert out["curve"][0.25]["power_headline"] == pytest.approx(0.1)
    assert out["min_detectable_effect_50"] == pytest.approx(0.75)
    assert out["min_detectable_effect_80"] is None
    assert out["passes_required_power"] is True                      # 0.7 >= 0.5 at 1.0
    by[1.0] = [rec(0.05)] * 4 + [rec(0.5)] * 6
    assert sfn.summarize_power(by)["passes_required_power"] is False


def test_calibration_dataset_record_and_determinism():
    kw = dict(n_series=3, n_sessions=2500, B=24)
    r1 = sfn.run_calibration_dataset(3, **kw)
    r2 = sfn.run_calibration_dataset(3, **kw)
    assert r1 == r2
    for f in ("dataset_seed", "effect", "n_series", "n_tested", "tstar", "tstar_key", "p_n1",
              "p_n2", "p_headline", "n_failed_n1", "n2_covers_tstar", "n2_offsets", "subset"):
        assert f in r1
    assert r1["effect"] == 0.0 and 1.0 / 25.0 <= r1["p_n1"] <= 1.0
    for k in range(6):                                                # headline = N1 in every record
        r = sfn.run_calibration_dataset(k, **kw)
        assert r["p_headline"] == r["p_n1"]
    r3 = sfn.run_calibration_dataset(4, **kw)
    assert r3["tstar"] != r1["tstar"]
    import json
    json.dumps(r1, allow_nan=False)                                   # strict JSON


def test_calibration_task_wrapper_is_picklable_and_matches_the_direct_call():
    import pickle
    pickle.dumps(sfn.calibration_task)
    task = {"dataset_seed": 3, "n_series": 3, "n_sessions": 2500, "B": 24, "effect": 0.0}
    assert sfn.calibration_task(task) == sfn.run_calibration_dataset(3, n_series=3,
                                                                     n_sessions=2500, B=24)
    bad = sfn.calibration_task({"dataset_seed": 3, "n_series": 0, "n_sessions": 10, "B": 5})
    assert "error" in bad and bad["dataset_seed"] == 3                # an error is recorded, not raised
    labelled = sfn.calibration_task(dict(task, kind="size"))          # the script labels its tasks
    assert labelled["kind"] == "size" and {k: v for k, v in labelled.items() if k != "kind"} ==         sfn.run_calibration_dataset(3, n_series=3, n_sessions=2500, B=24)
    assert sfn.calibration_task({"dataset_seed": 3, "n_series": 0, "kind": "size"})["kind"] == "size"


def test_harness_power_smoke_an_injected_effect_is_detected_and_no_edge_is_not():
    """Tiny sizes (3 series, 2500 sessions, B = 30, 5 seeded datasets). The
    real size / power calibration is scripts/validate_spike_fade_null.py; this
    proves the harness has power to see an effect and does not invent one."""
    kw = dict(n_series=3, n_sessions=2500, B=30)
    null = [sfn.run_calibration_dataset(k, **kw) for k in range(5)]
    edge = [sfn.run_calibration_dataset(k, effect=2.0, **kw) for k in range(5)]
    reject = lambda recs: np.mean([r["p_headline"] <= 0.10 for r in recs])     # noqa: E731
    assert reject(edge) >= 0.8                       # the injected effect is found
    assert reject(null) <= 0.4                       # and no-edge data mostly is not
    assert np.mean([r["tstar"] for r in edge]) > np.mean([r["tstar"] for r in null]) + 1.0
    assert all(r["realized_mean_x"] > 1.0 for r in edge)      # the effect really was delivered
    assert all(r["n_injected"] > 10 and r["effect_series"] == "S00" for r in edge)
    assert all(r["n_injected"] == 0 and r["realized_mean_x"] is None for r in null)


def test_subset_of_the_first_seven_series_is_reported_when_the_dataset_is_larger():
    rec = sfn.run_calibration_dataset(2, n_series=4, n_sessions=2000, B=16, subset_size=2)
    assert rec["subset"]["n_series"] == 2
    assert rec["subset"]["tstar_key"].split("|")[0] in ("S00", "S01")
    assert rec["subset"]["p_headline"] == rec["subset"]["p_n1"]
    assert sfn.run_calibration_dataset(2, n_series=2, n_sessions=2000, B=16,
                                       subset_size=2)["subset"] is None


# ---------------------------------------------------------------------------
# 8. The runnable calibration script (scripts/validate_spike_fade_null.py)
# ---------------------------------------------------------------------------

def _load_script():
    import importlib.util
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                                        "validate_spike_fade_null.py"))
    spec = importlib.util.spec_from_file_location("validate_spike_fade_null_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_worker_count_is_clamped_to_at_most_six():
    sc = _load_script()
    assert [sc.clamp_workers(v) for v in (0, -3, 1, 4, 6, 7, 99)] == [1, 1, 1, 4, 6, 6, 6]
    assert sc.MAX_WORKERS == 6


def test_script_defaults_follow_the_spec_and_the_output_folder_is_not_created_on_import():
    sc = _load_script()
    assert sc.DEFAULT_OUT.replace("\\", "/").endswith(
        "runs/spike-fade-study/calibration/null_calibration.json")
    assert sc.DEFAULT_SEED == 2026_0927


def test_script_builds_unique_seeded_tasks_for_size_and_power():
    import argparse
    sc = _load_script()
    args = argparse.Namespace(datasets=5, power_datasets=3, effects=[0.25, 0.5, 1.0], B=200,
                              n_series=13, sessions=5000, seed=777)
    tasks = sc.build_tasks(args)
    assert len(tasks) == 5 + 3 * 3
    assert len({t["dataset_seed"] for t in tasks}) == len(tasks)        # no two share a stream
    assert [t["kind"] for t in tasks[:5]] == ["size"] * 5 and all(t["effect"] == 0.0 for t in tasks[:5])
    assert sorted({t["effect"] for t in tasks[5:]}) == [0.25, 0.5, 1.0]
    assert all(t["B"] == 200 and t["n_series"] == 13 and t["n_sessions"] == 5000 for t in tasks)


def test_script_clean_makes_strict_json():
    import json
    sc = _load_script()
    out = sc.clean({1.0: [float("nan"), float("inf"), 2.5], "a": {"b": (1, float("-inf"))}})
    assert out == {"1.0": [None, None, 2.5], "a": {"b": [1, None]}}
    json.dumps(out, allow_nan=False)


def test_script_end_to_end_tiny_run_writes_a_strict_json_report(tmp_path):
    import json
    sc = _load_script()
    out = tmp_path / "calibration" / "report.json"
    rc = sc.main(["--datasets", "4", "--power-datasets", "2", "--effects", "2.0", "--n-series", "3",
                  "--sessions", "2000", "--B", "16", "--workers", "1", "--checkpoint-every", "2",
                  "--smoke", "--out", str(out)])
    assert rc == 0 and out.exists()
    with open(out, encoding="utf-8") as fh:
        rep = json.load(fh)
    assert rep["finished"] is True and rep["verdict"]["n_errors"] == 0
    assert rep["size"]["n_datasets"] == 4
    assert set(rep["size"]) >= {"p_n1", "p_n2", "p_headline", "all_in_window", "n2_judged"}
    assert rep["power"]["curve"]["2.0"]["n_datasets"] == 2
    assert rep["verdict"]["size_datasets_ok"] is False        # 4 datasets is not the required 300
    assert len(rep["records"]["size"]) == 4 and "2.0" in rep["records"]["power"]
    assert rep["open_choices"] and rep["spec"]["size_window"] == [0.06, 0.14]
    assert not (tmp_path / "calibration" / "report.json.tmp").exists()      # atomic write cleaned up
    # the same seed reproduces the same records (order-independent)
    out2 = tmp_path / "again.json"
    sc.main(["--datasets", "4", "--power-datasets", "2", "--effects", "2.0", "--n-series", "3",
             "--sessions", "2000", "--B", "16", "--workers", "1", "--smoke", "--out", str(out2)])
    with open(out2, encoding="utf-8") as fh:
        rep2 = json.load(fh)
    key = lambda r: r["dataset_seed"]                                         # noqa: E731
    assert sorted(rep["records"]["size"], key=key) == sorted(rep2["records"]["size"], key=key)


def _fake_records(n_size=300, n1_rate=0.09, n2_rate=0.0, n2_available=100, powers=None):
    """Hand-made calibration records: size records with the given N1 (= headline)
    and N2 rejection rates, plus power records at three effects."""
    recs = []
    n1_rej = int(round(n1_rate * n_size))
    n2_rej = int(round(n2_rate * n2_available))
    for k in range(n_size):
        p1 = 0.05 if k < n1_rej else 0.6
        p2 = None
        if k < n2_available:
            p2 = 0.05 if k < n2_rej else 0.6
        recs.append({"kind": "size", "dataset_seed": k, "effect": 0.0, "n_series": 13, "p_n1": p1,
                     "p_n2": p2, "p_n2_covered": p2, "p_headline": p1, "n2_covers_tstar": p2 is not None,
                     "n2_offsets": {"S00": 12, "S01": 3},
                     "subset": {"p_n1": p1, "p_n2": p2, "p_n2_covered": p2, "p_headline": p1}})
    powers = powers or {0.25: 0.2, 0.5: 0.5, 1.0: 0.95}
    for eff, pw in powers.items():
        for k in range(20):
            p = 0.05 if k < int(round(pw * 20)) else 0.6
            recs.append({"kind": "power", "dataset_seed": 10000 + k, "effect": eff, "n_series": 13,
                         "p_n1": p, "p_n2": None, "p_n2_covered": None, "p_headline": p,
                         "n2_covers_tstar": False, "n2_offsets": {"S00": 1}, "realized_mean_x": eff,
                         "subset": {"p_n1": p, "p_n2": None, "p_n2_covered": None, "p_headline": p}})
    return recs


def _fake_args(sc):
    import argparse
    return argparse.Namespace(datasets=300, power_datasets=20, effects=[0.25, 0.5, 1.0], B=200,
                              n_series=13, sessions=5000, seed=1, workers=4)


def test_script_acceptance_requires_n1_and_headline_but_never_n2():
    sc = _load_script()
    rep = sc.build_report(_fake_args(sc), _fake_records(n1_rate=0.09, n2_rate=0.0), 1.0, True)
    v = rep["verdict"]
    assert v["size_n1_in_window"] is True and v["size_headline_in_window"] is True
    assert v["size_required_in_window"] is True                    # N2 rate 0.0 (out of window) does not matter
    assert v["size_n2_sensitivity"]["gating"] is False
    assert v["size_n2_sensitivity"]["in_window_informational"] is False
    assert v["size_n2_sensitivity"]["label"] == "sensitivity, not gating"
    assert v["power_at_1_passes"] is True and v["size_datasets_ok"] is True
    assert rep["spec_version"] == "3.1" and rep["headline_definition"] == "N1 alone (p_headline = p_n1)"
    # N1 (= headline) out of the window -> the requirement fails even with a perfect N2
    bad = sc.build_report(_fake_args(sc), _fake_records(n1_rate=0.30, n2_rate=0.10), 1.0, True)
    assert bad["verdict"]["size_required_in_window"] is False
    assert bad["verdict"]["size_n2_sensitivity"]["in_window_informational"] is True
    # the power requirement stays: 50 percent at 1.0
    weak = sc.build_report(_fake_args(sc), _fake_records(powers={0.25: 0.1, 0.5: 0.2, 1.0: 0.4}), 1.0, True)
    assert weak["verdict"]["power_at_1_passes"] is False


def test_script_restates_a_v3_headline_as_n1_alone():
    sc = _load_script()
    r = {"kind": "size", "p_n1": 0.04, "p_n2": 0.5, "p_headline": 0.5,           # v3: max(N1, N2)
         "subset": {"p_n1": 0.04, "p_n2": 0.5, "p_headline": 0.5}}
    out = sc.restate_headline(r)
    assert out["p_headline"] == 0.04 and out["p_headline_v3_stored"] == 0.5
    assert out["subset"]["p_headline"] == 0.04 and out["subset"]["p_headline_v3_stored"] == 0.5
    assert r["p_headline"] == 0.5                                                 # the input is not mutated


def test_script_resummarize_rewrites_the_headline_from_raw_p_values_and_writes_the_v3_1_summary(tmp_path):
    import json
    sc = _load_script()
    recs = _fake_records(n1_rate=0.09, n2_rate=0.0)
    for r in recs:                                                                # stored the v3 way
        if r["p_n2"] is not None:
            r["p_headline"] = max(r["p_n1"], r["p_n2"])
            r["subset"]["p_headline"] = max(r["subset"]["p_n1"], r["subset"]["p_n2"])
    size = [r for r in recs if r["kind"] == "size"]
    power = {}
    for r in recs:
        if r["kind"] == "power":
            power.setdefault(str(r["effect"]), []).append(r)
    src = tmp_path / "calibration" / "null_calibration.json"
    src.parent.mkdir()
    stored = {"finished": True, "elapsed_seconds": 5.0,
              "args": {"datasets": 300, "power_datasets": 20, "effects": [0.25, 0.5, 1.0], "B": 200,
                       "n_series": 13, "sessions": 5000, "seed": 1, "workers": 4},
              "records": {"size": size, "power": power}}
    src.write_text(json.dumps(stored), encoding="utf-8")
    v3_headline_rate = sfn.summarize_size(size)["p_headline"]["rate"]
    assert v3_headline_rate < 0.09                                                # v3 (max) rate is lower
    rc = sc.main(["--resummarize", str(src)])
    dst = tmp_path / "calibration" / "null_calibration_v3_1_summary.json"
    assert rc == 0 and dst.exists()
    out = json.loads(dst.read_text(encoding="utf-8"))
    assert out["spec_version"] == "3.1" and out["resummarized_from"].endswith("null_calibration.json")
    assert out["size"]["p_headline"]["rate"] == out["size"]["p_n1"]["rate"] == pytest.approx(0.09)
    assert out["size"]["p_headline"]["in_window"] is True
    assert out["v3_stored_headline_size"]["rate"] == pytest.approx(v3_headline_rate)   # provenance kept
    assert "gating" not in out["v3_stored_headline_size"]      # a superseded definition gates nothing
    assert out["verdict"]["size_required_in_window"] is True
    assert out["verdict"]["size_n2_sensitivity"]["gating"] is False
    assert "records" not in out                                                   # a summary, not the raw dump
    assert out["power"]["curve"]["1.0"]["power_headline"] == out["power"]["curve"]["1.0"]["power_n1"]
    # an explicit --summary-out is honoured
    alt = tmp_path / "elsewhere" / "s.json"
    assert sc.main(["--resummarize", str(src), "--summary-out", str(alt)]) == 0 and alt.exists()


# ---------------------------------------------------------------------------
# 9. Zero-SD tolerance, boundary and call-site pins,
#    calibration hardening and the vol-of-vol robustness arm
# ---------------------------------------------------------------------------

def _core_t(x):
    x = np.asarray(x, dtype=float)
    return sfs.cell_stats(x, x, pd.bdate_range("2001-01-01", periods=len(x)), bootstrap=False).t


def _same_t(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def test_t_stat_is_undefined_for_identical_and_near_constant_x_exactly_like_the_core():
    for x in ([0.3] * 10, [0.3] * 11, [0.7] * 8, [1.1] * 9):
        assert math.isnan(sfn.t_stat(np.array(x))) and math.isnan(_core_t(x))
    # identical values are undefined by the "all equal" rule even where the sd itself overflows
    # (squared deviations of about 1e306 values are inf, so the tolerance alone would give t = 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for x in ([2.0070509537366427e306] * 11, [-2.0070509537366427e306] * 50):
            assert math.isnan(_core_t(x)) and math.isnan(sfn.t_stat(np.array(x)))
    near = 1.0 + 1e-13 * np.arange(8)                 # float-noise spread: the core says undefined
    assert math.isnan(_core_t(near))
    assert math.isnan(sfn.t_stat(near))               # not a t of about 1e13
    ok = 1.0 + 1e-9 * np.arange(8)                    # genuine (tiny) spread: defined in both
    assert math.isfinite(_core_t(ok)) and sfn.t_stat(ok) == _core_t(ok)


def test_t_stat_reads_the_cores_tolerance_constant_it_does_not_copy_it(monkeypatch):
    x = 1.0 + 1e-4 * np.arange(8)                     # sd about 2.4e-4
    assert math.isfinite(sfn.t_stat(x)) and math.isfinite(_core_t(x))       # default tolerance 1e-12
    monkeypatch.setattr(sfs, "SD_ZERO_REL_TOL", 1e-3)                        # the core's tolerance moves ...
    assert math.isnan(_core_t(x)) and math.isnan(sfn.t_stat(x))              # ... and t_stat follows it


def test_t_stat_equals_the_core_statistic_including_the_nan_pattern_on_random_and_degenerate_vectors():
    rng = np.random.default_rng(2026_0923)
    cases = []
    for _ in range(400):
        n = int(rng.integers(1, 30))
        centre = float(rng.choice([0.0, 1.0, -5.0, 100.0, 0.3]))
        eps = float(10.0 ** rng.integers(-18, 1))
        kind = int(rng.integers(0, 6))
        if kind == 0:
            x = np.clip(rng.standard_t(4, size=n), -5, 5)                        # ordinary
        elif kind == 1:
            x = centre + eps * rng.standard_normal(n)                            # tiny spread
        elif kind == 2:
            x = np.full(n, centre)                                               # identical
        elif kind == 3:
            x = centre + eps * np.arange(n)                                      # arithmetic ramp
        elif kind == 4:
            x = np.r_[np.full(max(n - 1, 0), 0.3), 0.3 + eps][:n]                # constant but one
        else:
            x = np.clip(rng.standard_t(4, size=n), -5, 5)
            x[rng.random(n) < 0.3] = np.nan                                      # non-finite entries
        cases.append(x)
    cases += [np.array([]), np.array([np.nan, np.nan]), np.array([1.0]), np.array([1.0, 2.0])]
    # the tolerance boundary itself, relative to max(1, |S|): just below and just above it
    ramp = np.arange(8, dtype=float) - 3.5
    ramp /= ramp.std(ddof=1)
    for centre in (0.0, 0.3, 1.0, -100.0, 100.0, 1.0e4):
        for factor in (0.3, 0.9, 1.1, 3.0):
            cases.append(centre + factor * sfs.SD_ZERO_REL_TOL * max(1.0, abs(centre)) * ramp)
    n_nan = n_def = 0
    for x in cases:
        a = sfn.t_stat(x)
        fin = x[np.isfinite(x)]
        b = _core_t(fin) if len(fin) else float("nan")
        assert _same_t(a, b), (x, a, b)
        n_nan += math.isnan(a)
        n_def += math.isfinite(a)
    assert n_nan >= 60 and n_def >= 60                                          # both patterns exercised
    # a large centre moves the boundary (it is relative to max(1, |S|), not absolute): pin one pair
    big = 1.0e4 + 0.5 * sfs.SD_ZERO_REL_TOL * 1.0e4 * ramp
    assert math.isnan(sfn.t_stat(big)) and math.isnan(_core_t(big))
    big2 = 1.0e4 + 3.0 * sfs.SD_ZERO_REL_TOL * 1.0e4 * ramp
    assert math.isfinite(sfn.t_stat(big2)) and sfn.t_stat(big2) == _core_t(big2)


def test_stratum_codes_split_at_exactly_zero():
    d = np.array([1.0, 0.0, -0.0, -1.0, np.nan, 1e-300, -1e-300])
    code = sfn._stratum_codes(d)
    assert list(code) == [sfn.STRATUM_POS, sfn.STRATUM_NONPOS, sfn.STRATUM_NONPOS, sfn.STRATUM_NONPOS,
                          sfn.STRATUM_UNDEFINED, sfn.STRATUM_POS, sfn.STRATUM_NONPOS]


def test_observed_group_keeps_only_valid_episodes_when_a_kept_episode_cannot_trade():
    n = 3500
    idx = pd.bdate_range("2000-01-03", periods=n)
    close = _sv_walk(n, 0)
    key = ("A", "up_fade", "z20@2.0")
    base = sfn.observe_family([sfs.build_series(pd.Series(close, index=idx), name="A",
                                                drop_missing=False)])
    g0 = base.groups[key]
    assert g0.n == g0.n_kept >= 8
    found = False
    for a in g0.anchors[2:8]:
        broken = close.copy()
        broken[int(a) + 1 + 21] = np.nan                    # the primary cell's exit close is missing
        s2 = sfs.build_series(pd.Series(broken, index=idx), name="A", drop_missing=False)
        es = sfs.build_episode_set(s2, sfs.compute_triggers(s2)["z20@2.0"], "up_fade")
        if int(a) not in es.t0:
            continue
        g2 = sfn.observe_family([s2]).groups[key]
        assert g2.n_kept == len(es.t0) and g2.n < g2.n_kept          # kept by censoring, not valid
        assert int(a) not in g2.anchors and np.all(np.isfinite(g2.x))
        assert sum(g2.strata) == g2.n and len(g2.anchors) == g2.n
        found = True
        break
    assert found


def test_a_group_with_undefined_t_is_never_in_the_tested_set_however_many_episodes(universe, monkeypatch):
    real = sfs.cell_stats

    def undefined_t(*a, **k):
        return dataclasses.replace(real(*a, **k), t=float("nan"))
    monkeypatch.setattr(sfs, "cell_stats", undefined_t)
    o = sfn.observe_family(universe)
    assert any(g.n >= sfs.MIN_EPISODES for g in o.groups.values())         # enough valid episodes ...
    assert o.tested == () and all(not g.tested for g in o.groups.values())  # ... but t undefined
    t, key = o.tstar()
    assert math.isnan(t) and key is None


def test_undefined_group_t_counts_as_minus_infinity_in_the_replication_maximum_not_zero():
    keys = (("A", "up_fade", "z20@2.0"), ("A", "down_fade", "z20@2.0"))
    t = np.array([[np.nan, -2.0], [np.nan, np.nan], [-1.5, np.nan], [0.5, np.nan]])
    run = sfn.NullRun(method="N1", seed=1, B=4, groups=keys, series_names=("A",), t=t,
                      covered=np.ones(2, dtype=bool), fail=np.zeros((4, 1), dtype=bool),
                      stat=np.zeros(4))
    assert list(run.statistic()) == [-2.0, -np.inf, -1.5, 0.5]             # NaN ignored, never read as 0


def test_null_max_is_minus_infinity_when_every_pseudo_group_t_is_undefined(obs, cobs, monkeypatch):
    monkeypatch.setattr(sfn, "t_stat", lambda x: float("nan"))
    r1 = sfn.run_null_n1(obs, B=4, seed=sfn.N1_SEED)
    r2 = sfn.run_null_n2(cobs, B=4, seed=sfn.N2_SEED)
    assert np.all(r1.stat == -np.inf) and np.all(r2.stat == -np.inf)
    assert sfn.p_value(-0.5, r1.stat) == pytest.approx(1.0 / 5.0)          # never at or above a finite T*


def test_every_call_site_gives_each_series_its_own_stream(universe, obs, cobs, monkeypatch):
    calls = []
    real = sfn.series_rng

    def recorder(seed, name, salt=0):
        calls.append((seed, name, salt))
        return real(seed, name, salt)
    monkeypatch.setattr(sfn, "series_rng", recorder)
    sfn.run_null_n1(obs, B=3, seed=111)
    assert sorted(calls) == [(111, "A", 0), (111, "B", 0)]                  # one stream per series, by name
    calls.clear()
    sfn.run_null_n2(cobs, B=3, seed=222)
    assert sorted(calls) == [(222, "A", 0), (222, "B", 0)]
    calls.clear()
    sfn.n1_cell_null_series(universe[0], cells=[sfs.PRIMARY_CELL], B=3, seed=333)
    assert len(calls) == 8 and {c[0] for c in calls} == {333} and {c[1] for c in calls} == {"A"}
    import zlib
    want = {zlib.crc32(f"{t}|{sd}".encode("utf-8")) & 0xFFFFFFFF for t in sfs.TRIGGERS for sd in sfs.SIDES}
    assert {c[2] for c in calls} == want and len(want) == 8                 # one salted stream per group
    calls.clear()
    sfn.n1_cell_null(universe[0], "pct756", "up_fade", cells=[sfs.PRIMARY_CELL], B=3, seed=444)
    assert calls == [(444, "A", zlib.crc32(b"pct756|up_fade") & 0xFFFFFFFF)]


def test_calibration_seeds_are_derived_per_dataset_and_differ_between_the_two_nulls(monkeypatch):
    assert sfn._derived_seed(sfn.N1_SEED, 3) == sfn.N1_SEED * 1_000_003 + 3
    seen = []
    real = sfn.confirmatory_test

    def recorder(spikes, **kw):
        seen.append((kw["n1_seed"], kw["n2_seed"]))
        return real(spikes, **kw)
    monkeypatch.setattr(sfn, "confirmatory_test", recorder)
    for k in (3, 4):
        sfn.run_calibration_dataset(k, n_series=1, n_sessions=900, B=8)
    assert seen[0] == (sfn._derived_seed(sfn.N1_SEED, 3), sfn._derived_seed(sfn.N2_SEED, 3))
    assert seen[1] == (sfn._derived_seed(sfn.N1_SEED, 4), sfn._derived_seed(sfn.N2_SEED, 4))
    assert len({seen[0][0], seen[0][1], seen[1][0], seen[1][1]}) == 4        # four distinct streams


# ---- calibration hardening: the default generator is frozen, the robustness arm is additive ----

def _hash_dataset(ds):
    import hashlib
    m = hashlib.sha1()
    for k in sorted(ds):
        m.update(k.encode())
        m.update(ds[k].to_numpy().tobytes())
    return m.hexdigest()


def test_default_generator_is_byte_identical_to_the_one_the_v3_1_calibration_used():
    """Pinned hashes of the default generator (no vol-of-vol multiplier): it must stay
    byte-identical to the generator the v3.1 calibration used."""
    assert _hash_dataset(sfn.make_dataset(101, n_series=2, n_sessions=300)) == \
        "2aedaddfdc7c3b291862e17db57dd10abafde6ca"
    assert _hash_dataset(sfn.make_dataset(2026_0927)) == "ba73c6dfb014570432f5a1326410d81043863157"
    assert _hash_dataset(sfn.make_dataset(7, n_series=7, n_sessions=5000)) == \
        "7c13bf8553391189441717c6da119f9439cd3c38"
    assert sfn.calibration_params(7, 3) == {"df": 4.768_834_311_887_255, "vol": 0.010_416_729_010_241_68,
                                             "phi": 0.956_087_049_372_005, "eta": 0.286_939_133_737_804_2}
    # an explicit multiplier of 1 changes nothing
    assert _hash_dataset(sfn.make_dataset(101, n_series=2, n_sessions=300, vol_of_vol_multiplier=1.0)) == \
        "2aedaddfdc7c3b291862e17db57dd10abafde6ca"


def test_vol_of_vol_multiplier_scales_the_log_variance_innovation_and_only_that():
    p = {"df": 4.5, "vol": 0.01, "phi": 0.9, "eta": 0.2}
    base = sfn.simulate_no_edge_series(300000, 5, 0, params=p)
    for m in (2.0, 3.0):
        x = sfn.simulate_no_edge_series(300000, 5, 0, params=p, vol_of_vol_multiplier=m)
        assert not np.array_equal(base, x)
        assert np.array_equal(x, sfn.simulate_no_edge_series(300000, 5, 0, params=dict(p, eta=0.2 * m)))
        # var(log|r|) = var(log|z|) + var(h) / 4 with var(h) = (eta m)^2 / (1 - phi^2): the increase
        # over the default arm is the documented one, so the innovation of h scales by m
        lb = np.log(np.abs(np.diff(np.log(base))))
        lx = np.log(np.abs(np.diff(np.log(x))))
        want = (m * m - 1.0) * (0.2 ** 2 / (1.0 - 0.9 ** 2)) / 4.0
        assert np.var(lx) - np.var(lb) == pytest.approx(want, rel=0.15)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            sfn.simulate_no_edge_series(50, 5, 0, params=p, vol_of_vol_multiplier=bad)


def test_log_volatility_innovation_sd_is_documented_as_eta_over_two():
    p = {"df": 4.0, "vol": 0.01, "phi": 0.95, "eta": 0.3}
    assert sfn.log_vol_innovation_sd(p) == pytest.approx(0.15)                # volatility = vol * exp(h / 2)
    assert sfn.log_vol_innovation_sd(p, vol_of_vol_multiplier=3.0) == pytest.approx(0.45)
    assert "eta / 2" in sfn.CALIB_ETA_DOC and "exp(h / 2)" in sfn.CALIB_ETA_DOC


def test_calibration_record_carries_the_vol_of_vol_multiplier():
    a = sfn.run_calibration_dataset(3, n_series=2, n_sessions=1500, B=8)
    b = sfn.run_calibration_dataset(3, n_series=2, n_sessions=1500, B=8, vol_of_vol_multiplier=3.0)
    assert a["vol_of_vol_multiplier"] == 1.0 and b["vol_of_vol_multiplier"] == 3.0
    assert a["tstar"] != b["tstar"]                                          # a different arm really ran
    with pytest.raises(ValueError):
        sfn.run_calibration_dataset(3, n_series=2, n_sessions=1500, B=8, vol_of_vol_multiplier=0.0)
    assert sfn.calibration_task({"dataset_seed": 3, "n_series": 2, "n_sessions": 1500, "B": 8,
                                 "vol_of_vol_multiplier": 3.0}) == b


# ---- the script: B >= 200 for a claim, errored datasets fail the run, the robustness arm ----

def test_script_rejects_a_claim_run_with_fewer_than_200_replications():
    import argparse
    sc = _load_script()
    ns = lambda B, smoke: argparse.Namespace(B=B, smoke=smoke, vol_of_vol_multipliers=[1.0])   # noqa: E731
    assert "200" in sc.validate_args(ns(199, False))
    assert "200" in sc.validate_args(ns(16, False))
    assert sc.validate_args(ns(200, False)) is None
    assert sc.validate_args(ns(16, True)) is None                        # --smoke: tiny test runs only
    bad = argparse.Namespace(B=200, smoke=False, vol_of_vol_multipliers=[1.0, 0.0])
    assert "multiplier" in sc.validate_args(bad)
    with pytest.raises(SystemExit):                                       # and main refuses before running
        sc.main(["--datasets", "1", "--power-datasets", "0", "--B", "16", "--workers", "1",
                 "--out", "unused.json"])


def test_errored_datasets_fail_the_run_loudly(tmp_path, capsys):
    import json
    sc = _load_script()
    out = tmp_path / "bad.json"
    rc = sc.main(["--datasets", "2", "--power-datasets", "0", "--n-series", "0", "--sessions", "500",
                  "--B", "8", "--workers", "1", "--smoke", "--out", str(out)])
    assert rc == 1                                                        # non-zero exit
    text = capsys.readouterr().out
    assert "FAILED" in text and "2 of 2" in text and "errored" in text   # loud summary line
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["verdict"]["n_errors"] == 2 and rep["verdict"]["run_ok"] is False
    assert rep["verdict"]["size_datasets_ok"] is False and len(rep["errors"]) == 2


def test_errors_also_fail_the_verdict_when_most_datasets_completed():
    sc = _load_script()
    recs = _fake_records(n_size=300) + [{"kind": "size", "dataset_seed": 999, "effect": 0.0,
                                         "error": "ValueError: boom"}]
    rep = sc.build_report(_fake_args(sc), recs, 1.0, True)
    assert rep["verdict"]["n_errors"] == 1 and rep["verdict"]["run_ok"] is False
    assert rep["verdict"]["size_datasets_ok"] is False                    # 300 completed + 1 error is not clean
    assert sc.build_report(_fake_args(sc), _fake_records(n_size=300), 1.0, True)["verdict"]["run_ok"] is True


def test_script_builds_paired_tasks_for_the_vol_of_vol_arms():
    import argparse
    sc = _load_script()
    args = argparse.Namespace(datasets=4, power_datasets=2, effects=[1.0], B=200, n_series=7,
                              sessions=5000, seed=50, vol_of_vol_multipliers=[1.0, 2.0, 3.0])
    tasks = sc.build_tasks(args)
    size = [t for t in tasks if t["kind"] == "size"]
    power = [t for t in tasks if t["kind"] == "power"]
    assert len(size) == 12 and len(power) == 2                             # power only on the default arm
    assert {t["vol_of_vol_multiplier"] for t in size} == {1.0, 2.0, 3.0}
    assert all(t["vol_of_vol_multiplier"] == 1.0 for t in power)
    for m in (1.0, 2.0, 3.0):                                              # the SAME dataset seeds per arm
        assert sorted(t["dataset_seed"] for t in size if t["vol_of_vol_multiplier"] == m) == \
            [50, 51, 52, 53]


def test_script_reports_size_by_vol_of_vol_arm_and_keeps_the_verdict_on_the_default_arm():
    sc = _load_script()
    recs = []
    for m, rate in ((1.0, 0.09), (2.0, 0.12), (3.0, 0.22)):
        for r in _fake_records(n_size=300, n1_rate=rate, powers={1.0: 0.9}):
            if r["kind"] == "size":
                recs.append(dict(r, vol_of_vol_multiplier=m))
    rep = sc.build_report(_fake_args(sc), recs, 1.0, True)
    by = rep["size_by_vol_of_vol_multiplier"]
    assert set(by) == {"1.0", "2.0", "3.0"}
    assert [by[k]["p_n1"]["rate"] for k in ("1.0", "2.0", "3.0")] == pytest.approx([0.09, 0.12, 0.22])
    assert by["3.0"]["p_n1"]["in_window"] is False and by["2.0"]["p_n1"]["in_window"] is True
    assert rep["size"]["p_n1"]["rate"] == pytest.approx(0.09)             # the default arm is THE size claim
    assert rep["verdict"]["size_required_in_window"] is True             # arms x2, x3 never gate
    assert "REPORTED caveat" in rep["robustness_caveat"] and "not a re-tuning" in rep["robustness_caveat"]
    single = sc.build_report(_fake_args(sc), _fake_records(n_size=300), 1.0, True)
    assert "size_by_vol_of_vol_multiplier" not in single                  # one arm: nothing extra


def test_script_end_to_end_vol_of_vol_arms(tmp_path):
    import json
    sc = _load_script()
    out = tmp_path / "rob.json"
    rc = sc.main(["--datasets", "3", "--power-datasets", "0", "--n-series", "3", "--sessions", "1800",
                  "--B", "8", "--workers", "1", "--smoke", "--vol-of-vol-multipliers", "1", "2",
                  "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert set(rep["size_by_vol_of_vol_multiplier"]) == {"1.0", "2.0"}
    assert rep["size"]["n_datasets"] == 3 and rep["args"]["vol_of_vol_multipliers"] == [1.0, 2.0]
    assert {r["vol_of_vol_multiplier"] for r in rep["records"]["size"]} == {1.0, 2.0}
    assert rep["claim_run"] is False                                      # --smoke marks it
