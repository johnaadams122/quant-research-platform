"""Null / test machinery of the range-bound spike-fade study, Stage 1
(the confirmatory max-t test and its two nulls, the same-cell null quantile
helper for the exploratory map, and the synthetic calibration tests).

WHAT THIS MODULE IS. The confirmatory yes/no test built on the deterministic
core in `spike_fade_study`: the observed primary-cell t of every group, the
frozen TESTED SET, the maximum-t statistic T*, the two nulls (N1 random dates,
N2 translation), the p-values, the headline, the adjusted per-series /
per-side p, and the synthetic calibration harness (the null-calibration
acceptance test). The HEADLINE p is the N1 p ALONE; N2 is a
reported SENSITIVITY, not gating (its p, its availability, its feasible-offset
counts and its like-for-like p are printed beside the headline, labelled
"sensitivity, not gating"). It is a RESEARCH module: it places no trade, writes no ledger row and
spends no trial slot. No dollar figures anywhere in this file.

VOCABULARY. A GROUP is (series, side, trigger); its key is
the tuple (series, side, trigger) so that Python's tuple order IS the
"lexicographic order of series, side, trigger" the tie rules use. A CELL is a group plus
(entry, H, TP, SL). The PRIMARY CELL is E0 / H = 21 / TP none / SL none.

HOW A NULL REPLICATION WORKS (one replication, one series):
  * The observed data fix, per tested group, its valid n (traded episodes
    with a finite x) and its D-sign strata (how many of those episodes had
    D > 0, D <= 0 or D undefined at the E0 entry close). The tested set is
    frozen in `observe_family`, before any draw.
  * Every possible anchor date of the series is pre-evaluated ONCE per side
    (`SeriesTables`): its primary-cell x, whether it is a valid traded
    episode, and its D stratum, all computed by the core's own engine. A
    pseudo-episode is therefore exactly what the identical machinery would
    return for that anchor; nothing is re-implemented.
  * N1: one random permutation of the series' censor-eligible dates is drawn;
    each tested group then walks that SAME permutation in order and accepts
    a date only if it is valid for the group's side, at least GAP_SESSIONS
    (20) sessions from every date the group has already accepted, and its D
    stratum still has quota, until the group's strata quotas are met exactly.
    A group that exhausts the permutation first is a failed replication and
    the replication statistic is minus infinity. The statistic is the maximum
    t over the tested set only.
  * N2: ONE offset per series from a grid spaced 21 sessions apart is added
    to every tested anchor of the series (a pure translation, no wrap-around);
    only offsets for which every shifted anchor is still a valid, uncensored
    episode are feasible; fewer than 10 feasible offsets skips N2 for the
    series (reported). D strata are NOT re-matched under N2. N2 is a
    SENSITIVITY only: it never enters the headline or the decision.
  * p = (1 + k) / (1 + B), k = replications with statistic at or above T*
    (ties count). Headline p = the N1 value.

STREAMS. Every series has its OWN random stream, keyed by (seed, series
name): draws never depend on which other series are in the family (series are
independent under the null) and identical seeds give identical outputs.

CHOICES WHERE THE DESIGN IS SILENT (the design = the study's fixed rules, the
constants of the `spike_fade_*` modules) are listed in the docstrings of the
functions that make them and collected in `OPEN_CHOICES` (reported, never
made silently).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import spike_fade_study as sfs

# ---------------------------------------------------------------------------
# Constants (literal values, pinned by the tests)
# ---------------------------------------------------------------------------

N1_SEED = 2026_0923            # frozen seed of the random-date null
N2_SEED = 2026_0924            # frozen seed of the translation null
MAP_SEED = 2026_0926           # exploratory-map context column: same-cell N1 null
NULL_B = 500                  # replications per null
MAP_B = 200                   # map context: replications per group
N2_GRID_STEP = 21             # offsets spaced at least 21 sessions apart
N2_MIN_OFFSETS = 10           # fewer feasible offsets skips N2
HEADLINE_P_MAX = 0.10         # decision language
N2_SENSITIVITY_LABEL = "sensitivity, not gating"   # how N2 is always labelled

# D-sign strata of an E0 entry close (the N1 matching): two are specified
# (D > 0 and D <= 0); a third, "D undefined" (the median window unusable
# while the trade itself is valid), is matched as its own stratum so that the
# counts add up to the valid n exactly. See OPEN_CHOICES.
STRATUM_POS, STRATUM_NONPOS, STRATUM_UNDEFINED = 0, 1, 2
N_STRATA = 3

OPEN_CHOICES = (
    "D-sign strata: the design names D > 0 and D <= 0; a valid observed "
    "episode whose median window is unusable has D undefined and is matched "
    "as a third stratum, exactly (the null draws the same three counts).",
    "Random-anchor spacing: pairwise at least GAP_SESSIONS (20) sessions "
    "between accepted pseudo-anchors of one group (the analogue of gap20 for "
    "dates that are not a run of flags).",
    "Eligible dates: every censor-eligible position of the series (t0 + 70 "
    "<= last index), NOT restricted to dates after a trigger's own first "
    "usable date; each side then discards dates that fail the data rules.",
    "N2 anchors: the VALID observed anchors of the series' tested groups "
    "(the episodes that define n) are translated; observed-but-invalid "
    "episodes are not part of n and are not shifted.",
    "N2 grid: offsets are non-zero multiples of 21 sessions (offset 0 is the "
    "observed configuration itself and is excluded).",
    "Headline: the headline p, its Monte Carlo SE, the decision "
    "and every per-item adjusted headline are the N1 values ALONE. N2 is "
    "computed and reported as a sensitivity (sensitivity, not gating) and can "
    "never change the headline or the decision.",
    "N2 coverage: when a series is skipped (fewer than 10 feasible offsets) "
    "the N2 replication statistic is the maximum over the tested groups of "
    "the covered series only; the specified N2 p exists only if the T* "
    "group's series is covered (otherwise it is None, and that is reported).",
    "A group that cannot reach its n makes the WHOLE replication statistic "
    "minus infinity (the design's wording); that never counts toward k, so failures "
    "are anti-conservative and are counted and reported.",
    "A pseudo-group whose t is undefined (sd = 0) does not contribute to the "
    "replication maximum (NaN is ignored; all-NaN gives minus infinity).",
    "Monte Carlo standard error: sqrt(p * (1 - p) / B) of the reported p.",
    "Map context null: pseudo-episodes are the group's N1 draws matched on "
    "the PRIMARY cell's valid n and strata; another cell's null S averages "
    "the pseudo-episodes that TRADE in that cell (NaN if none).",
    "Like-for-like N2 (sensitivity, not gating): besides the specified N2 p "
    "(T* over the whole tested set), the result reports the maximum over the "
    "N2-covered groups against the N2 null over the same groups "
    "(p_n2_covered), so the translation null can be calibrated even where N2 "
    "skips the T* series. Its size is reported by the synthetic calibration "
    "(the shifted hold can land on the spike's own run-up); no exclusion or "
    "grid change is made for it.",
    "Publication lag: a series carries its own extra_lag (FRED publication lag); "
    "censoring, the E0 entry, the D strata and the primary-cell x all use it "
    "through the core, so pseudo-episodes are censored and evaluated exactly "
    "as the observed ones.",
    "Calibration injection (the synthetic calibration test leaves the mechanism open): the "
    "minimum-norm daily log-return vector that gives EVERY injected primary-"
    "cell trade a window sum of minus s * delta * sigma21, found by a fixed-"
    "point iteration over the anchors that flag on the modified path; the "
    "realized mean x is reported beside the nominal delta.",
    "Zero spread: t is undefined exactly where the core says so (n < 2, "
    "identical values, or sd <= the core's SD_ZERO_REL_TOL x max(1, |S|)); "
    "the tolerance is read from the core, not copied.",
    "Robustness caveat (reported, not a re-tuning): with the log-volatility "
    "innovation multiplied by 2 and 3 the N1 test is anti-conservative for "
    "rejections (a convexity drift makes the simple-return P&L over sigma21 "
    "larger on spike dates than on random dates for long positions); a "
    "non-rejection is unaffected. Measured by "
    "scripts/validate_spike_fade_null.py (calibration/null_size_robustness.json "
    "under the study output folder).",
    "Calibration seeds: each dataset has its own deterministic seeds for the "
    "data and for both nulls (the frozen study seeds are for the real data); "
    "the calibration family has 13 series and the sub-family "
    "of the first 7 (the real family size) is reported from the same "
    "replications.",
)

GroupKey = Tuple[str, str, str]      # (series, side, trigger)


# ---------------------------------------------------------------------------
# Random streams: one per (seed, series name)
# ---------------------------------------------------------------------------

def series_rng(seed: int, name: str, salt: int = 0) -> np.random.Generator:
    """The random stream of one series under one frozen seed.

    Keyed by the series NAME (crc32), never by its position in the family, so
    the draws of a series do not depend on which other series are present and
    series are independent of one another. `salt` separates streams that
    share (seed, name) (the map context null uses one per group)."""
    key = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(key, int(salt)))
    return np.random.default_rng(ss)


# ---------------------------------------------------------------------------
# The cell statistic t
# ---------------------------------------------------------------------------

def t_stat(x) -> float:
    """t = S / (sd(x) / sqrt(n)), ddof = 1, over the finite entries of x
    with the CORE's zero-spread rule: t is
    undefined (NaN) when n < 2, when every value is identical, or when
    sd <= `spike_fade_study.SD_ZERO_REL_TOL` x max(1, |S|) (float noise:
    x = 1 + 1e-13 i is a constant, not a t of 1e13). The tolerance constant is
    READ from the core at call time, not copied, and the operations run in the
    same order as `spike_fade_study.cell_stats`, so a null replication's t is
    bit-for-bit what the core reports for the same episodes in anchor order,
    NaN pattern included (the tests pin this on random and degenerate
    vectors)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n < 2:
        return float("nan")
    S = float(x.mean())
    sd = float(x.std(ddof=1))
    if float(x.max()) == float(x.min()) or sd <= sfs.SD_ZERO_REL_TOL * max(1.0, abs(S)):
        return float("nan")
    return S / (sd / math.sqrt(n))


# ---------------------------------------------------------------------------
# Per-series pseudo-episode tables (computed once per series and side)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SeriesTables:
    """Everything a pseudo-anchor needs, for EVERY possible anchor position.

      n         number of sessions of the series
      elig      censor-eligible anchors: t0 + CENSOR_TAIL <= n - 1
      x         {side: (n,) primary-cell x at each anchor, NaN unless traded}
      valid     {side: (n,) bool}: a valid traded episode (finite x)
      stratum   {side: (n,) int8}: 0 = D > 0, 1 = D <= 0, 2 = D undefined

    All three come from the core's own engine (`evaluate_variants` and
    `entry_d_e0`), so a pseudo-episode is exactly what the identical
    primary-cell machinery returns for that anchor."""
    name: str
    n: int
    elig: np.ndarray
    x: Dict[str, np.ndarray]
    valid: Dict[str, np.ndarray]
    stratum: Dict[str, np.ndarray]


def _stratum_codes(d: np.ndarray) -> np.ndarray:
    """D stratum per session: D > 0 -> 0, D <= 0 -> 1 (exactly zero is NOT
    positive), D undefined (NaN) -> 2. One comparison decides the
    sign, so the zero boundary has a single definition."""
    d = np.asarray(d, dtype=float)
    with np.errstate(invalid="ignore"):
        code = np.where(d > 0, STRATUM_POS, STRATUM_NONPOS).astype(np.int8)
    code[np.isnan(d)] = STRATUM_UNDEFINED
    return code


def build_tables(series: sfs.SpikeSeries) -> SeriesTables:
    """Pre-evaluate the primary cell at every censor-eligible anchor, both
    sides (see `SeriesTables`)."""
    n = series.n
    tail = sfs.censor_tail(getattr(series, "extra_lag", 0))    # uniform censoring plus the FRED publication lag
    elig, _ = sfs.censor_episodes(np.arange(n, dtype=np.int64), n, tail=tail)
    x, valid, stratum = {}, {}, {}
    for side in sfs.SIDES:
        xs = np.full(n, np.nan)
        ok = np.zeros(n, dtype=bool)
        if len(elig):
            res = sfs.evaluate_variants(series, elig, side, [sfs.PRIMARY_CELL])
            good = (res.status[0] == sfs.STATUS_TRADED) & np.isfinite(res.x[0])
            xs[elig[good]] = res.x[0][good]
            ok[elig[good]] = True
        x[side], valid[side] = xs, ok
        stratum[side] = _stratum_codes(sfs.entry_d_e0(series, side))
    return SeriesTables(name=series.name, n=n, elig=elig, x=x, valid=valid,
                        stratum=stratum)


# ---------------------------------------------------------------------------
# Observation: observed t of every group, the frozen tested set, T*
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class ObservedGroup:
    """One group's OBSERVED primary cell.

      anchors / x   the VALID traded episodes (finite x), ascending by anchor
      n / t / mean_x   valid n and the per-cell statistics over them
      n_kept        episodes kept by uniform censoring, before the data rules
      strata        (n with D > 0, n with D <= 0, n with D undefined)
      tested        n >= min_n AND t defined (frozen at observation time)
    """
    key: GroupKey
    series: str
    side: str
    trigger: str
    n_kept: int
    anchors: np.ndarray
    x: np.ndarray
    n: int
    mean_x: float
    t: float
    strata: Tuple[int, int, int]
    tested: bool


def select_tstar(candidates: Iterable[Tuple[float, int, GroupKey]]):
    """T* over (t, n, key) candidates: the largest t; ties broken by the
    larger n, then by the lexicographically first (series, side, trigger)
    key. Groups whose t is NaN are ignored. Returns (t, key, n) or
    None if no candidate has a defined t. Exact float equality defines a tie."""
    best = None
    for t, n, key in candidates:
        if t is None or not math.isfinite(float(t)):
            continue
        rank = (-float(t), -int(n), tuple(key))
        if best is None or rank < best[0]:
            best = (rank, float(t), tuple(key), int(n))
    return None if best is None else (best[1], best[2], best[3])


@dataclass(eq=False)
class ObservedFamily:
    """The observed side of the confirmatory test.

      series / tables   {name: SpikeSeries / SeriesTables}
      groups            {key: ObservedGroup}, every (series, side, trigger)
      tested            the FROZEN tested set: a tuple of keys fixed here,
                        before any null draw; nothing later changes it
    """
    series_names: Tuple[str, ...]
    series: Dict[str, sfs.SpikeSeries]
    tables: Dict[str, SeriesTables]
    groups: Dict[GroupKey, ObservedGroup]
    tested: Tuple[GroupKey, ...]
    min_n: int

    def tested_groups(self) -> List[ObservedGroup]:
        return [self.groups[k] for k in self.tested]

    def tstar(self):
        """(T*, key) over the tested set only (tie rule in `select_tstar`),
        or (nan, None) when the tested set is empty."""
        best = select_tstar((self.groups[k].t, self.groups[k].n, k) for k in self.tested)
        return (float("nan"), None) if best is None else (best[0], best[1])


def _observe_one(series: sfs.SpikeSeries, tab: SeriesTables, sig, side: str, trigger: str,
                 min_n: int) -> ObservedGroup:
    """Observed primary cell of ONE group: gap20 episodes (uniformly
    censored), the valid traded ones (finite x), n, t and the D strata."""
    es = sfs.build_episode_set(series, sig, side)
    t0 = np.asarray(es.t0, dtype=np.int64)
    ok = tab.valid[side][t0]
    anchors = t0[ok]
    x = tab.x[side][anchors]
    d = tab.stratum[side][anchors]
    strata = (int((d == STRATUM_POS).sum()), int((d == STRATUM_NONPOS).sum()),
              int((d == STRATUM_UNDEFINED).sum()))
    n = int(len(anchors))
    cs = sfs.cell_stats(x, x, series.dates[anchors], bootstrap=False) if n else None
    t = float(cs.t) if cs is not None else float("nan")
    key = (series.name, side, trigger)
    return ObservedGroup(
        key=key, series=series.name, side=side, trigger=trigger, n_kept=int(len(t0)),
        anchors=anchors, x=x, n=n,
        mean_x=float(cs.mean_x) if cs is not None else float("nan"), t=t, strata=strata,
        tested=bool(n >= min_n and math.isfinite(t)))


def observe_family(series_list: Sequence[sfs.SpikeSeries], *,
                   min_n: int = sfs.MIN_EPISODES,
                   signals: Optional[dict] = None) -> ObservedFamily:
    """Observed primary-cell statistics of every group of every series, and
    the frozen tested set.

    For each series x side x trigger: gap20 episodes (uniformly censored),
    the primary cell on their anchors through the core's engine, the VALID
    episodes (traded, finite x), n, t and the D strata. TESTED = the groups
    whose observed valid n is at least `min_n` (the pre-registered 8) with a
    defined t; this tuple is created here and never recomputed, so nothing a
    null replication does can change which groups are in the maximum.
    `signals` = optional {series name: compute_triggers output} to avoid
    recomputing the triggers. Series names must be unique."""
    names = tuple(s.name for s in series_list)
    if len(set(names)) != len(names):
        raise ValueError("series names must be unique")
    signals = signals or {}
    series_d, tables_d, groups = {}, {}, {}
    for s in series_list:
        series_d[s.name] = s
        tab = build_tables(s)
        tables_d[s.name] = tab
        sig_all = signals.get(s.name) or sfs.compute_triggers(s)
        for side in sfs.SIDES:
            for trig in sfs.TRIGGERS:
                groups[(s.name, side, trig)] = _observe_one(s, tab, sig_all[trig], side, trig,
                                                            min_n)
    tested = tuple(k for k, g in groups.items() if g.tested)
    return ObservedFamily(series_names=names, series=series_d, tables=tables_d,
                          groups=groups, tested=tested, min_n=int(min_n))


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------

def p_value(tstar: float, null_stat) -> float:
    """p = (1 + k) / (1 + B), k = the number of null replications whose
    statistic is at or above `tstar` (inclusive ties), B = len(null_stat).
    A replication whose statistic is minus infinity (a failed draw) never
    counts toward k. NaN when T* is undefined."""
    null_stat = np.asarray(null_stat, dtype=float)
    if tstar is None or not math.isfinite(float(tstar)):
        return float("nan")
    k = int(np.sum(null_stat >= float(tstar)))
    return (1.0 + k) / (1.0 + len(null_stat))


def mc_se(p: float, B: int) -> float:
    """Monte Carlo standard error of a reported p: sqrt(p * (1 - p) / B)."""
    if not math.isfinite(p) or B <= 0:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / B)


def headline_p(p_n1: float) -> float:
    """HEADLINE p = the N1 value ALONE. The function takes no N2 argument on
    purpose: a caller that passes one fails with a TypeError instead of
    silently mixing N2 into the headline."""
    return p_n1


# ---------------------------------------------------------------------------
# One null run: replication statistics for the frozen tested set
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class NullRun:
    """B replications of one null over the frozen tested set.

      method   "N1" or "N2"
      groups   the tested keys, in the frozen order (= ObservedFamily.tested)
      t        (B, G) each group's replication t; NaN = undefined, failed or
               (N2) not covered
      covered  (G,) bool: the group takes part in this null (always True for
               N1; N2 skips the groups of a series with too few offsets)
      fail     (B, S) bool over `series_names`: a group of that series could
               not reach its n in that replication (N1 only)
      stat     (B,) the replication statistic: the maximum t over the covered
               tested groups, minus infinity if ANY draw failed or nothing
               is defined
      offsets  N2: {series: feasible offsets}; skipped: series without N2
      draws    only with collect_draws: N1 -> [ {key: anchors} per
               replication ]; N2 -> {series: (B,) chosen offsets}
    """
    method: str
    seed: int
    B: int
    groups: Tuple[GroupKey, ...]
    series_names: Tuple[str, ...]
    t: np.ndarray
    covered: np.ndarray
    fail: np.ndarray
    stat: np.ndarray
    offsets: Dict[str, np.ndarray] = field(default_factory=dict)
    skipped: Tuple[str, ...] = ()
    draws: object = None

    @property
    def n_failed(self) -> int:
        return int(self.fail.any(axis=1).sum())

    def statistic(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """The replication statistic restricted to a subset of the covered
        groups (`mask` over the columns of `t`; default all): the maximum t
        over them, minus infinity if a draw of one of THEIR series failed."""
        cols = self.covered if mask is None else (self.covered & np.asarray(mask, dtype=bool))
        used_series = {self.groups[j][0] for j in np.flatnonzero(cols)}
        s_idx = [i for i, nm in enumerate(self.series_names) if nm in used_series]
        failed = self.fail[:, s_idx].any(axis=1) if s_idx else np.zeros(self.B, dtype=bool)
        return _replication_stat(self.t, cols, failed)


def _replication_stat(t: np.ndarray, cols: np.ndarray, failed: np.ndarray) -> np.ndarray:
    B = t.shape[0]
    if not np.any(cols):
        return np.full(B, -np.inf)
    sub = t[:, cols]
    m = np.where(np.isfinite(sub), sub, -np.inf).max(axis=1)
    return np.where(failed, -np.inf, m)


def _tested_by_series(obs: ObservedFamily) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {nm: [] for nm in obs.series_names}
    for j, key in enumerate(obs.tested):
        out[key[0]].append(j)
    return out


# ---------------------------------------------------------------------------
# N1: random dates, matched on exact valid n and on the D-sign strata
# ---------------------------------------------------------------------------

_WALK_CHUNK = 512


def _walk_candidates(cand: np.ndarray, codes: np.ndarray, quota: Sequence[int], n: int,
                     gap: int = sfs.GAP_SESSIONS) -> Optional[np.ndarray]:
    """Greedy walk over already-filtered candidate dates (in permutation
    order): accept a date when its stratum still has quota and it is at least
    `gap` sessions from every date accepted so far. Returns the accepted
    anchors ascending, or None if the candidates run out first."""
    quota = [int(q) for q in quota]
    need = sum(quota)
    if need == 0:
        return np.zeros(0, dtype=np.int64)
    reach = gap - 1
    blocked = bytearray(n)
    acc: List[int] = []
    for start in range(0, len(cand), _WALK_CHUNK):
        chunk_c = cand[start:start + _WALK_CHUNK].tolist()
        chunk_k = codes[start:start + _WALK_CHUNK].tolist()
        for c, k in zip(chunk_c, chunk_k):
            if quota[k] and not blocked[c]:
                acc.append(c)
                quota[k] -= 1
                need -= 1
                if need == 0:
                    return np.sort(np.asarray(acc, dtype=np.int64))
                lo, hi = max(c - reach, 0), min(c + reach + 1, n)
                blocked[lo:hi] = b"\x01" * (hi - lo)
    return None


def draw_matched_anchors(perm: np.ndarray, valid: np.ndarray, stratum: np.ndarray,
                         quota: Sequence[int], n: int,
                         gap: int = sfs.GAP_SESSIONS) -> Optional[np.ndarray]:
    """N1 anchor draw for ONE group from a permutation of candidate dates.

    Walks `perm` in order and accepts a date only if `valid[date]` (a valid
    traded episode for the group side, censoring included), its D stratum
    `stratum[date]` still has quota (`quota` = (n_pos, n_nonpos, n_undefined),
    filled EXACTLY) and it is at least `gap` sessions away from every date
    already accepted (gap20 spacing). Returns the anchors ascending, or None
    if the permutation is exhausted first (a failed replication)."""
    perm = np.asarray(perm, dtype=np.int64)
    cand = perm[valid[perm]]
    return _walk_candidates(cand, stratum[cand], quota, n, gap)


def run_null_n1(obs: ObservedFamily, *, B: int = NULL_B, seed: int = N1_SEED,
                collect_draws: bool = False) -> NullRun:
    """The random-date null (N1), B replications over the FROZEN
    tested set.

    Per series (each with its own stream, `series_rng(seed, name)`) and per
    replication: ONE random permutation of the series censor-eligible dates
    is drawn; every tested group of the series then walks that same
    permutation (`draw_matched_anchors`) to obtain pseudo-episodes whose
    number of valid traded episodes equals the group observed valid n
    exactly and whose D strata equal the observed strata exactly. The group
    t is computed from the pre-evaluated x of those anchors (the core
    engine own values), anchors in ascending order. A group that cannot
    reach n fails the replication (statistic minus infinity)."""
    G = len(obs.tested)
    S = len(obs.series_names)
    t = np.full((B, G), np.nan)
    fail = np.zeros((B, S), dtype=bool)
    draws = [dict() for _ in range(B)] if collect_draws else None
    by_series = _tested_by_series(obs)
    for si, name in enumerate(obs.series_names):
        cols = by_series[name]
        if not cols:
            continue
        tab = obs.tables[name]
        rng = series_rng(seed, name)
        sides = sorted({obs.tested[j][1] for j in cols})
        for b in range(B):
            perm = rng.permutation(tab.elig)
            cand = {}
            for side in sides:
                c = perm[tab.valid[side][perm]]
                cand[side] = (c, tab.stratum[side][c])
            for j in cols:
                key = obs.tested[j]
                g = obs.groups[key]
                if sum(g.strata) != g.n:
                    raise ValueError(f"strata {g.strata} do not add up to n={g.n} for {key}")
                c, codes = cand[g.side]
                anchors = _walk_candidates(c, codes, g.strata, tab.n)
                if anchors is None:
                    fail[b, si] = True
                    continue
                t[b, j] = t_stat(tab.x[g.side][anchors])
                if draws is not None:
                    draws[b][key] = anchors
    covered = np.ones(G, dtype=bool)
    stat = _replication_stat(t, covered, fail.any(axis=1))
    return NullRun(method="N1", seed=int(seed), B=int(B), groups=obs.tested,
                   series_names=obs.series_names, t=t, covered=covered, fail=fail,
                   stat=stat, draws=draws)


# ---------------------------------------------------------------------------
# N2: pure translation of the observed anchors, one common offset per series
# ---------------------------------------------------------------------------

def feasible_offsets(anchor_sets: Iterable[Tuple[Sequence[int], np.ndarray]], n: int,
                     step: int = N2_GRID_STEP) -> np.ndarray:
    """Feasible translations of a series (N2), ascending.

    `anchor_sets` = [(anchors, valid)] one entry per tested group of the
    series: its valid observed anchors and its side validity table (`valid[t]`
    = t is a valid, uncensored traded episode for that group). The grid is
    every NON-ZERO multiple of `step` (21: offsets are at least 21 sessions
    apart; 0 is the observed configuration itself). An offset is feasible
    only if EVERY shifted anchor of EVERY group lands inside the series
    (0 <= a + d < n: a pure translation, NO wrap-around) and on a valid
    date, so uniform censoring and the data-quality rules still hold."""
    m_max = n // step + 1
    deltas = step * np.concatenate([np.arange(-m_max, 0), np.arange(1, m_max + 1)])
    ok = np.ones(len(deltas), dtype=bool)
    for anchors, valid in anchor_sets:
        a = np.asarray(anchors, dtype=np.int64)[:, None] + deltas[None, :]
        inside = (a >= 0) & (a < n)
        good = inside & np.asarray(valid, dtype=bool)[np.clip(a, 0, n - 1)]
        ok &= good.all(axis=0)
    return deltas[ok]


def run_null_n2(obs: ObservedFamily, *, B: int = NULL_B, seed: int = N2_SEED,
                min_offsets: int = N2_MIN_OFFSETS, step: int = N2_GRID_STEP,
                collect_draws: bool = False) -> NullRun:
    """The translation null (N2), B replications over the frozen
    tested set.

    Per series: the feasible offsets (`feasible_offsets`, over the valid
    observed anchors of the series tested groups) are computed once; per
    replication ONE offset is drawn uniformly from them (the series own
    stream, `series_rng(seed, name)`) and added to EVERY tested anchor of the
    series, so the count, the spacing and the clustering of events are
    preserved exactly and nothing wraps. The group t is the primary-cell t of
    the shifted anchors (all valid by construction). D strata are NOT
    re-matched (the different-regime sensitivity check). A series with fewer
    than `min_offsets` feasible offsets is skipped: its groups are not
    covered, the skip is listed in `skipped` and the feasible offsets are
    reported in `offsets` either way."""
    G = len(obs.tested)
    S = len(obs.series_names)
    t = np.full((B, G), np.nan)
    covered = np.zeros(G, dtype=bool)
    offsets: Dict[str, np.ndarray] = {}
    skipped: List[str] = []
    chosen: Dict[str, np.ndarray] = {}
    by_series = _tested_by_series(obs)
    for name in obs.series_names:
        cols = by_series[name]
        if not cols:
            continue
        tab = obs.tables[name]
        sets = [(obs.groups[obs.tested[j]].anchors, tab.valid[obs.tested[j][1]]) for j in cols]
        offs = feasible_offsets(sets, tab.n, step)
        offsets[name] = offs
        if len(offs) < min_offsets or len(offs) == 0:
            skipped.append(name)
            continue
        rng = series_rng(seed, name)
        picks = offs[rng.integers(0, len(offs), size=B)]
        chosen[name] = picks
        for j in cols:
            g = obs.groups[obs.tested[j]]
            covered[j] = True
            xs = tab.x[g.side]
            for b in range(B):
                t[b, j] = t_stat(xs[g.anchors + picks[b]])
    fail = np.zeros((B, S), dtype=bool)
    stat = _replication_stat(t, covered, np.zeros(B, dtype=bool))
    return NullRun(method="N2", seed=int(seed), B=int(B), groups=obs.tested,
                   series_names=obs.series_names, t=t, covered=covered, fail=fail,
                   stat=stat, offsets=offsets, skipped=tuple(skipped),
                   draws=chosen if collect_draws else None)


# ---------------------------------------------------------------------------
# The confirmatory test: T*, both nulls, p-values, headline, adjusted p
# ---------------------------------------------------------------------------

ADJUSTED_P_LABEL = ("descriptive: per-series, per-side, per-pair and per-group maxima are "
                    "compared against the SAME global null-max distribution")


def decision(p: float) -> str:
    """The pre-registered decision language, applied to the HEADLINE
    p, which is the N1 p alone (N2 never enters it)."""
    if p is None or not math.isfinite(p):
        return ("no tested group: no group has the required number of valid traded "
                "episodes at the primary cell")
    if p <= HEADLINE_P_MAX:
        return ("confirmatory evidence that fading extremes beats same-side random entries "
                "at the primary cell, in the named pair")
    return ("inconclusive at the study's power (this is not a finding of no edge); the "
            "minimum detectable effect must be stated beside it")


def _jf(v) -> Optional[float]:
    """JSON-safe float: None for NaN and infinities."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _key_text(key) -> Optional[str]:
    return None if key is None else "|".join(key)


@dataclass(eq=False)
class ConfirmatoryResult:
    """Everything the report shows for the confirmatory test.

      tstar / tstar_key / tstar_n   the observed maximum t over the frozen
                                    tested set and the group that holds it
    HEADLINE -- the N1 p ALONE:
      p_n1 / p_headline             (1 + k) / (1 + B) against the N1 null;
                                    p_headline IS p_n1
      mc_se_n1 / mc_se_headline     its Monte Carlo standard error (equal)
      headline_rule / decision      "N1 alone (spec v3.1)" and the decision
                                    language applied to p_headline

    N2 SENSITIVITY (label "sensitivity, not gating"; never in the headline):
      p_n2 / mc_se_n2               (1 + k) / (1 + B) against the N2 null; None
                                    when N2 does not cover the T* group (its
                                    series was skipped)
      n2_offsets / n2_skipped       feasible-offset counts per series and the
                                    series N2 was not computed for
      n2_coverage / n2_sensitivity  coverage counts; `n2_sensitivity` is the
                                    labelled dict of all of these
      tstar_n2_covered / p_n2_covered
                                    LIKE-FOR-LIKE N2: the maximum t over the
                                    groups N2 covers, against the N2 null
                                    maximum over the same groups (None if N2
                                    covers nothing). Equal to p_n2 when N2
                                    covers every tested group.
      n_failed_n1                   replications where some group could not
                                    reach its n (statistic minus infinity)
      per_series / per_side / per_pair / per_group
                                    adjusted p of each item's maximum against
                                    the SAME global null-max (descriptive):
                                    p_n1 = p_headline, plus the separate
                                    p_n2_sensitivity
    """
    observed: ObservedFamily
    n1: NullRun
    n2: NullRun
    tstar: float
    tstar_key: Optional[GroupKey]
    tstar_n: Optional[int]
    p_n1: float
    p_n2: Optional[float]
    p_headline: float
    headline_rule: str
    mc_se_n1: float
    mc_se_n2: Optional[float]
    mc_se_headline: float
    n2_offsets: Dict[str, int]
    n2_skipped: List[str]
    n2_coverage: dict
    tstar_n2_covered: Optional[float]
    p_n2_covered: Optional[float]
    n_failed_n1: int
    per_series: Dict[str, dict]
    per_side: Dict[str, dict]
    per_pair: Dict[Tuple[str, str], dict]
    per_group: Dict[GroupKey, dict]

    @property
    def decision(self) -> str:
        return decision(self.p_headline)

    @property
    def n2_sensitivity(self) -> dict:
        """Everything about N2, in one labelled block: it is reported beside
        the headline and never enters it."""
        return {"label": N2_SENSITIVITY_LABEL, "gating": False, "seed": self.n2.seed,
                "p_n2": self.p_n2, "mc_se_n2": self.mc_se_n2,
                "available_for_tstar": self.p_n2 is not None,
                "tstar_n2_covered": self.tstar_n2_covered, "p_n2_covered": self.p_n2_covered,
                "offsets": dict(self.n2_offsets), "skipped": list(self.n2_skipped),
                "coverage": dict(self.n2_coverage), "null_max_quantiles": _quantiles(self.n2.stat)}

    def report_lines(self) -> List[str]:
        """Plain report lines for the confirmatory section: T*, the HEADLINE
        (N1 alone) with its Monte Carlo standard error, the
        replication failures, and then the N2 SENSITIVITY lines, each labelled
        "sensitivity, not gating": its p where available, its availability,
        the feasible-offset counts, the skip line for every series with too
        few feasible offsets, and the like-for-like p.
        Every N2 line starts with "N2"; no other line mentions the label."""
        lab = N2_SENSITIVITY_LABEL
        L = []
        if self.tstar_key is None:
            L.append("T*: undefined (no tested group).")
        else:
            L.append(f"T* = {self.tstar:.4f} at {_key_text(self.tstar_key)} (n = {self.tstar_n}); "
                     f"tested groups: {len(self.observed.tested)} of {len(self.observed.groups)}.")
        L.append(f"N1 (random dates): p = {self.p_n1:.4f}, Monte Carlo SE {self.mc_se_n1:.4f}, "
                 f"B = {self.n1.B}, seed {self.n1.seed}, failed replications {self.n_failed_n1}.")
        L.append(f"Headline p = {self.p_headline:.4f} ({self.headline_rule}), Monte Carlo SE "
                 f"{self.mc_se_headline:.4f}: {self.decision}.")
        if self.p_n2 is not None:
            L.append(f"N2 (translation, {lab}): p = {self.p_n2:.4f}, Monte Carlo SE "
                     f"{self.mc_se_n2:.4f}, seed {self.n2.seed}.")
        else:
            L.append(f"N2 (translation, {lab}): not available for the T* series.")
        for name, n in self.n2_offsets.items():
            L.append(f"N2 feasible offsets for {name} ({lab}): {n}.")
        for name in self.n2_skipped:
            L.append(f"N2 skipped for {name} ({lab}): only {self.n2_offsets.get(name, 0)} "
                     f"feasible offsets (fewer than {N2_MIN_OFFSETS}); N2 is not computed for its "
                     f"groups.")
        if self.p_n2_covered is not None and self.p_n2_covered != self.p_n2:
            L.append(f"N2 like-for-like ({lab}) over the {self.n2_coverage['groups_covered']} "
                     f"covered groups: T* = {self.tstar_n2_covered:.4f}, p = {self.p_n2_covered:.4f}.")
        return L

    def to_frame(self) -> pd.DataFrame:
        """One row per group (tested or not): observed statistics and, for a
        tested group, the descriptive adjusted p against the global null-max:
        `adj_p_n1` = `adj_p_headline` (N1 alone) and the separate
        `adj_p_n2_sensitivity` (sensitivity, not gating)."""
        rows = []
        for key, g in self.observed.groups.items():
            item = self.per_group.get(key)
            has_n2 = item is not None and item["p_n2_sensitivity"] is not None
            rows.append({
                "series": g.series, "side": g.side, "trigger": g.trigger,
                "n_kept": g.n_kept, "n": g.n, "mean_x": g.mean_x, "t": g.t,
                "tested": g.tested, "n_d_pos": g.strata[0], "n_d_nonpos": g.strata[1],
                "n_d_undefined": g.strata[2],
                "adj_p_n1": item["p_n1"] if item else float("nan"),
                "adj_p_headline": item["p_headline"] if item else float("nan"),
                "adj_p_n2_sensitivity": item["p_n2_sensitivity"] if has_n2 else float("nan"),
                "is_tstar": key == self.tstar_key})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        """Strict-JSON summary (NaN and infinities become None). The headline
        fields are p_n1 / p_headline (equal) and mc_se_n1 / mc_se_headline
        (equal); ALL N2 fields live under the single key `n2_sensitivity`."""
        def item_d(it):
            return {"t_max": _jf(it["t_max"]), "argmax_key": _key_text(it["argmax_key"]),
                    "n_tested_groups": it["n_tested_groups"], "p_n1": _jf(it["p_n1"]),
                    "p_headline": _jf(it["p_headline"]),
                    "p_n2_sensitivity": _jf(it["p_n2_sensitivity"])}
        n2 = self.n2_sensitivity
        n2_d = {"label": n2["label"], "gating": False, "seed": n2["seed"],
                "p_n2": _jf(n2["p_n2"]), "mc_se_n2": _jf(n2["mc_se_n2"]),
                "available_for_tstar": n2["available_for_tstar"],
                "tstar_n2_covered": _jf(n2["tstar_n2_covered"]),
                "p_n2_covered": _jf(n2["p_n2_covered"]), "offsets": n2["offsets"],
                "skipped": n2["skipped"], "coverage": n2["coverage"],
                "null_max_quantiles": n2["null_max_quantiles"]}
        return {
            "tstar": _jf(self.tstar), "tstar_key": _key_text(self.tstar_key),
            "tstar_n": self.tstar_n,
            "n_groups": len(self.observed.groups), "n_tested": len(self.observed.tested),
            "min_n": self.observed.min_n, "B": self.n1.B, "n1_seed": self.n1.seed,
            "p_n1": _jf(self.p_n1), "p_headline": _jf(self.p_headline),
            "headline_rule": self.headline_rule, "decision": self.decision,
            "mc_se_n1": _jf(self.mc_se_n1), "mc_se_headline": _jf(self.mc_se_headline),
            "n_failed_n1": self.n_failed_n1, "n1_null_max_quantiles": _quantiles(self.n1.stat),
            "n2_sensitivity": n2_d,
            "adjusted_p_label": ADJUSTED_P_LABEL,
            "per_series": {k: item_d(v) for k, v in self.per_series.items()},
            "per_side": {k: item_d(v) for k, v in self.per_side.items()},
            "per_pair": {"|".join(k): item_d(v) for k, v in self.per_pair.items()},
            "per_group": {"|".join(k): item_d(v) for k, v in self.per_group.items()},
            "open_choices": list(OPEN_CHOICES),
        }


def _quantiles(stat) -> Optional[dict]:
    fin = np.asarray(stat, dtype=float)
    fin = fin[np.isfinite(fin)]
    if not len(fin):
        return None
    return {f"q{q}": _jf(np.percentile(fin, q)) for q in (50, 90, 95)}


def _adjusted_item(obs: ObservedFamily, n1: NullRun, n2: NullRun, mask: np.ndarray) -> dict:
    """Adjusted p of the maximum t over the tested groups selected by `mask`,
    against the SAME global null-max distributions. `p_n1` = `p_headline`
    (N1 alone); the N2 p is the separate `p_n2_sensitivity`
    (sensitivity, not gating), present only when every group of the item is
    covered by N2."""
    keys = [k for k, m in zip(obs.tested, mask) if m]
    best = select_tstar((obs.groups[k].t, obs.groups[k].n, k) for k in keys)
    if best is None:
        return {"t_max": float("nan"), "argmax_key": None, "n_tested_groups": len(keys),
                "p_n1": float("nan"), "p_headline": float("nan"), "p_n2_sensitivity": None}
    t_max, key, _ = best
    p1 = p_value(t_max, n1.stat)
    p2 = p_value(t_max, n2.stat) if bool(n2.covered[mask].all()) else None
    return {"t_max": t_max, "argmax_key": key, "n_tested_groups": len(keys),
            "p_n1": p1, "p_headline": headline_p(p1), "p_n2_sensitivity": p2}


def run_confirmatory_test(obs: ObservedFamily, *, B: int = NULL_B, n1_seed: int = N1_SEED,
                          n2_seed: int = N2_SEED, n2_min_offsets: int = N2_MIN_OFFSETS,
                          collect_draws: bool = False) -> ConfirmatoryResult:
    """The confirmatory test end to end on an observed family: T*, the N1 and N2 nulls, both
    p-values, the headline and the adjusted per-series / per-side / per-pair /
    per-group p (against the same global null-max distributions).

    The HEADLINE p is the N1 p ALONE (`p_headline` = `p_n1`), and
    so are its Monte Carlo SE, the decision and every per-item adjusted
    headline. N2 is still computed and reported, as a SENSITIVITY (sensitivity,
    not gating): `p_n2` (None when N2 does not cover the T* group), the
    feasible-offset counts, the skipped series and the like-for-like
    `p_n2_covered`. When N2 covers only some series its null maximum ranges
    over the covered groups only (an open choice, reported in `n2_coverage`).
    Changing the N2 seed, the offset threshold or the N2 code can never move
    the headline."""
    tstar, tkey = obs.tstar()
    tn = obs.groups[tkey].n if tkey is not None else None
    n1 = run_null_n1(obs, B=B, seed=n1_seed, collect_draws=collect_draws)
    n2 = run_null_n2(obs, B=B, seed=n2_seed, min_offsets=n2_min_offsets,
                     collect_draws=collect_draws)
    p1 = p_value(tstar, n1.stat)
    j_star = obs.tested.index(tkey) if tkey is not None else None
    n2_applies = j_star is not None and bool(n2.covered[j_star])
    p2 = p_value(tstar, n2.stat) if n2_applies else None
    ph = headline_p(p1)
    rule = "no tested group" if tkey is None else "N1 alone (spec v3.1)"
    tested_series = sorted({k[0] for k in obs.tested})
    per_group = {k: _adjusted_item(obs, n1, n2, np.array([kk == k for kk in obs.tested]))
                 for k in obs.tested}
    per_series = {nm: _adjusted_item(obs, n1, n2, np.array([k[0] == nm for k in obs.tested]))
                  for nm in tested_series}
    per_side = {sd: _adjusted_item(obs, n1, n2, np.array([k[1] == sd for k in obs.tested]))
                for sd in sfs.SIDES if any(k[1] == sd for k in obs.tested)}
    pairs = sorted({(k[0], k[1]) for k in obs.tested})
    per_pair = {pr: _adjusted_item(obs, n1, n2, np.array([(k[0], k[1]) == pr for k in obs.tested]))
                for pr in pairs}
    cov_best = select_tstar((obs.groups[k].t, obs.groups[k].n, k)
                            for j, k in enumerate(obs.tested) if n2.covered[j])
    tcov = cov_best[0] if cov_best else None
    p2cov = p_value(tcov, n2.stat) if cov_best else None
    coverage = {"groups_tested": len(obs.tested), "groups_covered": int(n2.covered.sum()),
                "series_covered": [nm for nm in tested_series if nm not in n2.skipped],
                "tstar_covered": bool(n2_applies)}
    return ConfirmatoryResult(
        observed=obs, n1=n1, n2=n2, tstar=tstar, tstar_key=tkey, tstar_n=tn,
        p_n1=p1, p_n2=p2, p_headline=ph, headline_rule=rule,
        mc_se_n1=mc_se(p1, B), mc_se_n2=mc_se(p2, B) if p2 is not None else None,
        mc_se_headline=mc_se(ph, B),
        n2_offsets={nm: int(len(v)) for nm, v in n2.offsets.items()},
        n2_skipped=list(n2.skipped), n2_coverage=coverage, tstar_n2_covered=tcov,
        p_n2_covered=p2cov, n_failed_n1=n1.n_failed,
        per_series=per_series, per_side=per_side, per_pair=per_pair, per_group=per_group)


def confirmatory_test(series_list: Sequence[sfs.SpikeSeries], *, B: int = NULL_B,
                      n1_seed: int = N1_SEED, n2_seed: int = N2_SEED,
                      min_n: int = sfs.MIN_EPISODES, n2_min_offsets: int = N2_MIN_OFFSETS,
                      signals: Optional[dict] = None,
                      collect_draws: bool = False) -> ConfirmatoryResult:
    """`observe_family` then `run_confirmatory_test`: the whole confirmatory
    test on a list of prepared series (7 series -> 56 groups in the study;
    the list is a parameter so tests and the calibration can use others)."""
    obs = observe_family(series_list, min_n=min_n, signals=signals)
    return run_confirmatory_test(obs, B=B, n1_seed=n1_seed, n2_seed=n2_seed,
                                 n2_min_offsets=n2_min_offsets, collect_draws=collect_draws)


# ---------------------------------------------------------------------------
# Exploratory-map context: the same-cell N1 null distribution of the cell mean S
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class MapNull:
    """N1 null distribution of the cell mean S, for any of the 72 variants of
    one group (the exploratory map context column).

      cells      the variants, in row order (CELL_GRID by default)
      S          (n_cells, B) mean x over the pseudo-episodes that TRADE in
                 that cell; NaN if none traded or the replication failed
      n_traded   (n_cells, B) traded pseudo-episodes per cell and replication
      q90        (n_cells,) the 90th percentile (numpy linear) of the finite
                 S values of each cell: the context number for the map
      n_target / strata   the observed valid n and D strata the pseudo-
                 episodes are matched to (the PRIMARY cell, as in the confirmatory test)
      n_failed   replications that could not reach n (excluded)
      anchors    (B, n) the drawn anchors (only with collect_draws)
    """
    series: str
    trigger: str
    side: str
    cells: tuple
    B: int
    seed: int
    n_target: int
    strata: Tuple[int, int, int]
    S: np.ndarray
    n_traded: np.ndarray
    q90: np.ndarray
    n_failed: int
    anchors: Optional[np.ndarray] = None

    def q90_of(self, cell: tuple) -> float:
        return float(self.q90[self.cells.index(cell)])


def _nan_percentile_rows(S: np.ndarray, q: float) -> np.ndarray:
    out = np.full(S.shape[0], np.nan)
    for i in range(S.shape[0]):
        row = S[i][np.isfinite(S[i])]
        if len(row):
            out[i] = float(np.percentile(row, q, method="linear"))
    return out


def n1_cell_null(series: sfs.SpikeSeries, trigger: str, side: str, *,
                 cells: Iterable[tuple] = sfs.CELL_GRID, B: int = MAP_B, seed: int = MAP_SEED,
                 signals: Optional[dict] = None, tables: Optional[SeriesTables] = None,
                 observed: Optional[ObservedGroup] = None,
                 collect_draws: bool = False) -> MapNull:
    """The N1 null distribution of the cell mean S of one group, for every
    requested variant (default all 72), and its 90th percentile.

    Pseudo-episodes are N1's: per replication one random permutation of the
    series censor-eligible dates, walked with the SAME matching as the
    confirmatory test (exact observed valid n and D strata of the group's
    PRIMARY cell, gap20 spacing, data rules). ALL pseudo-episodes of all
    replications are then pushed through the core engine in ONE
    `evaluate_variants` call (vectorized over pseudo-episodes and variants);
    S of a cell in a replication is the mean x over the pseudo-episodes that
    trade in that cell (NaN if none). Each group has its own stream
    (`series_rng(seed, name, salt)`), so the result does not depend on which
    other groups are computed. `observed` / `tables` / `signals` may be
    supplied to avoid recomputation."""
    cells = tuple(cells)
    tab = tables if tables is not None else build_tables(series)
    if observed is None:
        sig = (signals or sfs.compute_triggers(series))[trigger]
        observed = _observe_one(series, tab, sig, side, trigger, sfs.MIN_EPISODES)
    n_cells = len(cells)
    n = observed.n
    S = np.full((n_cells, B), np.nan)
    n_traded = np.zeros((n_cells, B), dtype=np.int64)
    if n == 0:
        return MapNull(series=series.name, trigger=trigger, side=side, cells=cells, B=B,
                       seed=int(seed), n_target=0, strata=observed.strata, S=S,
                       n_traded=n_traded, q90=np.full(n_cells, np.nan), n_failed=0,
                       anchors=np.zeros((0, 0), dtype=np.int64) if collect_draws else None)
    salt = zlib.crc32(f"{trigger}|{side}".encode("utf-8")) & 0xFFFFFFFF
    rng = series_rng(seed, series.name, salt)
    ok_reps: List[int] = []
    drawn: List[np.ndarray] = []
    for b in range(B):
        perm = rng.permutation(tab.elig)
        c = perm[tab.valid[side][perm]]
        anchors = _walk_candidates(c, tab.stratum[side][c], observed.strata, tab.n)
        if anchors is not None:
            ok_reps.append(b)
            drawn.append(anchors)
    n_failed = B - len(ok_reps)
    A = np.vstack(drawn) if drawn else np.zeros((0, n), dtype=np.int64)
    if len(ok_reps):
        res = sfs.evaluate_variants(series, A.ravel(), side, cells)
        x = res.x.reshape(n_cells, len(ok_reps), n)
        fin = np.isfinite(x)
        cnt = fin.sum(axis=2)
        tot = np.where(fin, x, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        S[:, ok_reps] = means
        n_traded[:, ok_reps] = cnt
    return MapNull(series=series.name, trigger=trigger, side=side, cells=cells, B=B,
                   seed=int(seed), n_target=n, strata=observed.strata, S=S, n_traded=n_traded,
                   q90=_nan_percentile_rows(S, 90.0), n_failed=n_failed,
                   anchors=A if collect_draws else None)


def n1_cell_null_series(series: sfs.SpikeSeries, *, cells: Iterable[tuple] = sfs.CELL_GRID,
                        B: int = MAP_B, seed: int = MAP_SEED,
                        signals: Optional[dict] = None) -> Dict[Tuple[str, str], MapNull]:
    """`n1_cell_null` for all 8 (trigger, side) groups of one series
    (tables and triggers computed once): {(trigger, side): MapNull}."""
    tab = build_tables(series)
    signals = signals or sfs.compute_triggers(series)
    cells = tuple(cells)
    return {(trig, side): n1_cell_null(series, trig, side, cells=cells, B=B, seed=seed,
                                       signals=signals, tables=tab)
            for trig in sfs.TRIGGERS for side in sfs.SIDES}


# ---------------------------------------------------------------------------
# Subset families (the real family has 7 series; the calibration draws more)
# ---------------------------------------------------------------------------

def family_subset_p(res: ConfirmatoryResult, series_names: Iterable[str]) -> dict:
    """The confirmatory p-values of the sub-family made of `series_names`,
    from the SAME null replications as the full result: T* of the subset,
    and each null maximum restricted to the subset (`NullRun.statistic`).
    Series are independent under the null and own their streams, so this is
    exactly what a run on that sub-family alone would give, at no cost.
    `p_headline` = `p_n1`; `p_n2` and `p_n2_covered` are the N2
    sensitivity (sensitivity, not gating)."""
    names = set(series_names)
    obs = res.observed
    mask = np.array([k[0] in names for k in obs.tested], dtype=bool)
    keys = [k for k, m in zip(obs.tested, mask) if m]
    best = select_tstar((obs.groups[k].t, obs.groups[k].n, k) for k in keys)
    if best is None:
        return {"tstar": float("nan"), "tstar_key": None, "n_tested": 0, "p_n1": float("nan"),
                "p_n2": None, "p_n2_covered": None, "p_headline": float("nan"),
                "n2_covers_tstar": False}
    t, key, _ = best
    p1 = p_value(t, res.n1.statistic(mask))
    covers = bool(res.n2.covered[obs.tested.index(key)])
    p2 = p_value(t, res.n2.statistic(mask)) if covers else None
    cov_best = select_tstar((obs.groups[k].t, obs.groups[k].n, k)
                            for k in keys if res.n2.covered[obs.tested.index(k)])
    p2cov = p_value(cov_best[0], res.n2.statistic(mask)) if cov_best else None
    return {"tstar": t, "tstar_key": key, "n_tested": len(keys), "p_n1": p1, "p_n2": p2,
            "p_n2_covered": p2cov, "p_headline": headline_p(p1), "n2_covers_tstar": covers}


# ---------------------------------------------------------------------------
# Null-calibration acceptance test: the synthetic calibration harness
# ---------------------------------------------------------------------------
# Datasets of independent stochastic-volatility, fat-tailed (Student t, 3.5 to
# 5 degrees of freedom) random-walk series with NO edge; the whole
# confirmatory test is run on each. Under no edge the N1 rejection rate at 0.10
# and the headline rejection rate (= N1) must each land in
# [0.06, 0.14]; N2's rate is REPORTED, not required (sensitivity, not gating).
# With post-spike reversion injected into ONE series the power curve is
# reported and 1.0 sigma21 per trade needs at least 50 percent power. A failed
# window is a finding, never tuned away.

CALIB_SERIES = 13
CALIB_SESSIONS = 5000
CALIB_START = "2000-01-03"
CALIB_ALPHA = 0.10
SIZE_WINDOW = (0.06, 0.14)
CALIB_EFFECTS = (0.25, 0.5, 1.0)
POWER_REQUIRED_AT = (1.0, 0.5)            # (effect in sigma21 per trade, minimum power)
CALIB_DF_RANGE = (3.5, 5.0)
CALIB_VOL_RANGE = (0.007, 0.020)          # daily volatility level
CALIB_PHI_RANGE = (0.95, 0.99)            # persistence of the log-volatility AR(1)
CALIB_ETA_RANGE = (0.10, 0.30)            # innovation sd of h, where volatility = vol * exp(h / 2)
CALIB_ETA_DOC = ("eta is the innovation sd of the AR(1) state h, and the return volatility is "
                 "vol * exp(h / 2); the innovation sd of the LOG-volatility (h / 2) is therefore "
                 "eta / 2, i.e. 0.05 to 0.15 for eta in (0.10, 0.30). The default generator is "
                 "the v3.1 calibration generator, byte for byte (pinned by the tests).")
INJECT_HOLD = sfs.PRIMARY_CELL[1]         # 21: the primary cell horizon
INJECT_ITERATIONS = 6                     # fixed-point iterations of the injection
CALIB_SUBSET_SIZE = 7                     # the real confirmatory family has 7 series


def calibration_params(dataset_seed: int, k: int) -> dict:
    """Stochastic-volatility parameters of series k of a dataset (drawn from
    the documented ranges, deterministically from (dataset_seed, k))."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy=int(dataset_seed),
                                                       spawn_key=(int(k), 0)))
    return {"df": float(rng.uniform(*CALIB_DF_RANGE)),
            "vol": float(rng.uniform(*CALIB_VOL_RANGE)),
            "phi": float(rng.uniform(*CALIB_PHI_RANGE)),
            "eta": float(rng.uniform(*CALIB_ETA_RANGE))}


def log_vol_innovation_sd(params: dict, vol_of_vol_multiplier: float = 1.0) -> float:
    """Innovation sd of the LOG-volatility of the generator: volatility is
    vol * exp(h / 2) and h has innovation sd eta (times the vol-of-vol
    multiplier), so the log-volatility innovation sd is eta * multiplier / 2
    (see `CALIB_ETA_DOC`)."""
    return float(params["eta"]) * float(vol_of_vol_multiplier) / 2.0


def simulate_no_edge_series(n: int, dataset_seed: int, k: int, *,
                            params: Optional[dict] = None,
                            vol_of_vol_multiplier: float = 1.0) -> np.ndarray:
    """One no-edge close series: log-volatility AR(1) (stochastic volatility),
    Student t innovations rescaled to unit variance, returns with zero mean,
    prices 100 * exp(cumulative return). Positive by construction; series k
    of a dataset never depends on how many other series the dataset has.

    `vol_of_vol_multiplier` (default 1: the generator of the v3.1 calibration,
    byte for byte) multiplies eta, the innovation sd of the AR(1) state h
    (volatility = vol * exp(h / 2)): it is the robustness arm that measures
    the size of the test under higher vol-of-vol. The random draws are the
    same for every multiplier, so arms are paired."""
    if not vol_of_vol_multiplier > 0:
        raise ValueError("vol_of_vol_multiplier must be positive")
    p = params or calibration_params(dataset_seed, k)
    eta = p["eta"] * vol_of_vol_multiplier
    rng = np.random.default_rng(np.random.SeedSequence(entropy=int(dataset_seed),
                                                       spawn_key=(int(k), 1)))
    e = rng.standard_normal(n)
    h = np.empty(n)
    h[0] = e[0] * eta / math.sqrt(1.0 - p["phi"] ** 2)           # stationary start
    for i in range(1, n):
        h[i] = p["phi"] * h[i - 1] + eta * e[i]
    z = rng.standard_t(p["df"], size=n) / math.sqrt(p["df"] / (p["df"] - 2.0))
    r = z * p["vol"] * np.exp(h / 2.0)
    return 100.0 * np.exp(np.cumsum(r))


def make_dataset(dataset_seed: int, *, n_series: int = CALIB_SERIES,
                 n_sessions: int = CALIB_SESSIONS, start: str = CALIB_START,
                 vol_of_vol_multiplier: float = 1.0) -> Dict[str, pd.Series]:
    """{name: close Series}: n_series independent no-edge series S00, S01, ...
    on consecutive business days from `start` (all before the sealed cutoff).
    `vol_of_vol_multiplier` is the robustness arm of `simulate_no_edge_series`
    (default 1 = the v3.1 calibration generator, byte for byte)."""
    idx = pd.bdate_range(start, periods=n_sessions)
    if len(idx) and idx[-1] > sfs.DISCOVERY_CUTOFF:
        raise ValueError("synthetic dates must end on or before the discovery cutoff")
    return {f"S{k:02d}": pd.Series(
        simulate_no_edge_series(n_sessions, dataset_seed, k,
                                vol_of_vol_multiplier=vol_of_vol_multiplier), index=idx)
        for k in range(n_series)}


def _series_anchors(series: sfs.SpikeSeries) -> List[Tuple[int, int]]:
    """Distinct (anchor, side sign) over the gap20 episodes of all four
    triggers and both sides, ascending."""
    sets = sfs.build_all_episode_sets(series)
    return sorted({(int(a), sfs.SIDE_SIGN[side]) for (_, side), es in sets.items()
                   for a in es.t0})


def inject_post_spike_reversion(close: pd.Series, delta: float, *, hold: int = INJECT_HOLD,
                                iterations: int = INJECT_ITERATIONS) -> Tuple[pd.Series, dict]:
    """Add a post-spike reversion of `delta` * sigma21 to the primary-cell
    trades of ONE series (the injected-effect calibration datasets).

    For an anchor a on side sign s (+1 up-fade / short, -1 down-fade / long)
    the primary cell holds from the close of a + 1 to the close of a + 22, so
    its P&L is driven by the 21 daily log returns of sessions a + 2 .. a + 22.
    The injection is the minimum-norm vector v of extra daily log returns for
    which the window sum of v over EVERY injected anchor's window equals
    -s * delta * sigma21(a) (one linear equation per anchor): each such trade
    gains exactly delta * sigma21 in log terms even where the windows of
    different triggers overlap, and nothing changes before the earliest
    window. Prices after a change carry the shift forward multiplicatively.

    Injecting reversion changes which sessions flag, so the anchors are found
    by a FIXED-POINT iteration: the anchors of every trigger and side on the
    current path join a growing union, the system is re-solved for the whole
    union on the ORIGINAL series, and this repeats `iterations` times
    (`sigma21(a)` is the value on the path where the anchor first appeared).
    `info` reports how many of the final path's flagged anchors carry an
    injected window (`coincidence`), because the delivered effect is the
    honest quantity: the calibration reports the REALIZED mean x beside the
    nominal delta. delta = 0 returns the series unchanged bit for bit."""
    base = close.to_numpy(dtype=float)
    n = len(base)
    union: Dict[int, Tuple[int, float]] = {}       # anchor -> (side sign, sigma21 target scale)
    v = np.zeros(n)
    cur = close
    max_err = 0.0
    n_iter = int(iterations) if delta else 0
    for _ in range(n_iter):
        s = sfs.build_series(cur, name="inject")
        sigma = s.context().sigma
        for a, sg in _series_anchors(s):
            if a not in union and a + 1 + hold <= n - 1 and np.isfinite(sigma[a]) and sigma[a] > 0:
                union[a] = (sg, float(sigma[a]))
        keep = sorted(union)
        if not keep:
            break
        A = np.zeros((len(keep), n))
        target = np.empty(len(keep))
        for i, a in enumerate(keep):
            sg, scale = union[a]
            A[i, a + 2:a + 2 + hold] = 1.0
            target[i] = -sg * float(delta) * scale
        y = np.linalg.lstsq(A @ A.T, target, rcond=None)[0]
        v = A.T @ y
        max_err = float(np.abs(A @ v - target).max())
        cur = pd.Series(base * np.exp(np.cumsum(v)), index=close.index)
    out = pd.Series(base * np.exp(np.cumsum(v)), index=close.index)
    coincidence = None
    if union:
        final = {a for a, _ in _series_anchors(sfs.build_series(out, name="inject"))}
        coincidence = len(final & set(union)) / len(final) if final else None
    return out, {"delta": float(delta), "hold": int(hold), "iterations": n_iter,
                 "n_injected": len(union),
                 "anchors": [(a, union[a][0]) for a in sorted(union)],
                 "sigma21": {a: union[a][1] for a in sorted(union)},
                 "coincidence": coincidence, "max_error": max_err,
                 "max_abs_daily_extra": float(np.abs(v).max())}


def _derived_seed(base: int, dataset_seed: int) -> int:
    """A null seed per dataset (the frozen study seeds are for the real data;
    the calibration gives every dataset its own deterministic stream)."""
    return int(base) * 1_000_003 + int(dataset_seed)


def run_calibration_dataset(dataset_seed: int, *, n_series: int = CALIB_SERIES,
                            n_sessions: int = CALIB_SESSIONS, B: int = 200,
                            effect: float = 0.0, effect_series: int = 0,
                            subset_size: int = CALIB_SUBSET_SIZE,
                            vol_of_vol_multiplier: float = 1.0) -> dict:
    """One calibration dataset end to end: generate, optionally inject
    `effect` (sigma21 per trade) into series S{effect_series}, run the whole
    confirmatory test with B replications, return a strict-JSON record.

    The record carries the full family's T*, the N1 p and the headline p
    (equal), the N2 sensitivity p (None when N2 does not cover T*)
    and its like-for-like p, the failure and coverage counts and, when the
    dataset has more than `subset_size` series, the same p-values for the
    sub-family of the first `subset_size` series (the study's real family
    has 7) taken from the same null replications. `vol_of_vol_multiplier`
    (default 1) is the robustness arm: the size measured with the log-
    volatility innovation multiplied (a REPORTED caveat, not a re-tuning)."""
    if n_series < 1:
        raise ValueError("n_series must be at least 1")
    closes = make_dataset(dataset_seed, n_series=n_series, n_sessions=n_sessions,
                          vol_of_vol_multiplier=vol_of_vol_multiplier)
    names = list(closes)
    info = None
    if effect:
        nm = names[int(effect_series)]
        closes[nm], info = inject_post_spike_reversion(closes[nm], effect)
    spikes = [sfs.build_series(c, name=nm) for nm, c in closes.items()]
    res = confirmatory_test(spikes, B=B, n1_seed=_derived_seed(N1_SEED, dataset_seed),
                            n2_seed=_derived_seed(N2_SEED, dataset_seed))
    rec = {
        "dataset_seed": int(dataset_seed), "effect": float(effect),
        "effect_series": names[int(effect_series)] if effect else None,
        "n_series": int(n_series), "n_sessions": int(n_sessions), "B": int(B),
        "vol_of_vol_multiplier": float(vol_of_vol_multiplier),
        "n_groups": len(res.observed.groups), "n_tested": len(res.observed.tested),
        "tstar": _jf(res.tstar), "tstar_key": _key_text(res.tstar_key), "tstar_n": res.tstar_n,
        "p_n1": _jf(res.p_n1), "p_n2": _jf(res.p_n2), "p_n2_covered": _jf(res.p_n2_covered),
        "p_headline": _jf(res.p_headline), "n_failed_n1": res.n_failed_n1, "n2_covers_tstar": bool(res.n2_coverage["tstar_covered"]),
        "n2_offsets": dict(res.n2_offsets), "n2_skipped": list(res.n2_skipped),
        "n_injected": info["n_injected"] if info else 0,
        "realized_mean_x": None, "effect_series_t_max": None, "subset": None}
    if effect:
        nm = names[int(effect_series)]
        means = [res.observed.groups[k].mean_x for k in res.observed.tested if k[0] == nm]
        rec["realized_mean_x"] = _jf(np.mean(means)) if means else None
        item = res.per_series.get(nm)
        rec["effect_series_t_max"] = _jf(item["t_max"]) if item else None
    if n_series > subset_size:
        sub = family_subset_p(res, names[:subset_size])
        rec["subset"] = {"n_series": int(subset_size), "n_tested": sub["n_tested"],
                         "tstar": _jf(sub["tstar"]), "tstar_key": _key_text(sub["tstar_key"]),
                         "p_n1": _jf(sub["p_n1"]), "p_n2": _jf(sub["p_n2"]),
                         "p_n2_covered": _jf(sub["p_n2_covered"]),
                         "p_headline": _jf(sub["p_headline"]),
                         "n2_covers_tstar": sub["n2_covers_tstar"]}
    return rec


def calibration_task(task: dict) -> dict:
    """Picklable worker entry for multiprocessing: `task` holds the keyword
    arguments of `run_calibration_dataset` (dataset_seed, n_series, n_sessions,
    B, effect, effect_series, subset_size) plus an optional `kind` label that
    is echoed into the record. An exception is returned as {"error": ...}
    with the dataset seed, never raised, so one bad dataset cannot lose a
    long run; the summaries count such records."""
    kind = task.get("kind")
    try:
        kw = {k: v for k, v in task.items() if k not in ("dataset_seed", "kind")}
        rec = run_calibration_dataset(task["dataset_seed"], **kw)
    except Exception as exc:                       # noqa: BLE001 - recorded, not swallowed
        rec = {"dataset_seed": task.get("dataset_seed"),
               "effect": float(task.get("effect", 0.0)),
               "vol_of_vol_multiplier": float(task.get("vol_of_vol_multiplier", 1.0)),
               "error": f"{type(exc).__name__}: {exc}"}
    if kind is not None:
        rec["kind"] = kind
    return rec


def _rate(records: Sequence[dict], name: str, alpha: float, missing_counts: bool) -> dict:
    vals = [r.get(name) for r in records]
    avail = [v for v in vals if v is not None]
    n = len(vals) if missing_counts else len(avail)
    rej = int(sum(1 for v in avail if v <= alpha))
    rate = rej / n if n else None
    se = math.sqrt(rate * (1.0 - rate) / n) if n else None
    return {"n_available": len(avail), "n_denominator": n, "rejections": rej, "rate": rate,
            "se": se}


N2_MIN_JUDGED = 50      # fewer available datasets than this: an N2 rate is reported, not judged
SIZE_REQUIRED = ("p_n1", "p_headline")            # the size window is REQUIRED here
SIZE_REPORTED_ONLY = ("p_n2", "p_n2_covered")     # ... and only reported here (sensitivity, not gating)


def summarize_size(records: Sequence[dict], *, alpha: float = CALIB_ALPHA,
                   window: Tuple[float, float] = SIZE_WINDOW) -> dict:
    """Rejection rates at `alpha` of N1, the headline and N2 over the no-edge
    datasets, each with its binomial standard error and whether it lies in
    the acceptance window ([0.06, 0.14]).

    The window is REQUIRED for N1 (`p_n1`) and for the headline
    (`p_headline`, which equals N1); both are marked `gating = True` and
    `all_in_window` is True exactly when both are in the window. N2's rates
    (`p_n2`, `p_n2_covered`) are marked `gating = False`: they are reported
    with their own `in_window` flag for information (`n2_label` says
    "sensitivity, not gating") and never enter `all_in_window`.

    N1 and the headline always count a dataset (no test = no rejection). N2
    as specified (`p_n2`) exists only when N2 covers the T* group, and the
    like-for-like N2 (`p_n2_covered`: the covered groups' maximum against the
    N2 null over the same groups) whenever N2 covers anything; a dataset
    without one is left out of that rate's denominator, and a rate over fewer
    than `N2_MIN_JUDGED` datasets has `in_window = None` (not judged)."""
    recs = [r for r in records if "error" not in r]
    out = {"n_datasets": len(recs), "n_errors": len(records) - len(recs), "alpha": alpha,
           "window": list(window), "required": list(SIZE_REQUIRED),
           "n2_label": "N2 (translation null): " + N2_SENSITIVITY_LABEL}
    required_ok = []
    for name, miss, gating in (("p_n1", True, True), ("p_n2", False, False),
                               ("p_n2_covered", False, False), ("p_headline", True, True)):
        d = _rate(recs, name, alpha, miss)
        if not gating and d["n_available"] < N2_MIN_JUDGED:
            d["in_window"] = None
        else:
            d["in_window"] = bool(d["rate"] is not None and window[0] <= d["rate"] <= window[1])
        d["gating"] = gating
        if gating:
            required_ok.append(d["in_window"])
        out[name] = d
    out["n2_judged"] = bool(out["p_n2"]["in_window"] is not None
                            or out["p_n2_covered"]["in_window"] is not None)
    out["all_in_window"] = bool(all(required_ok))
    return out


def min_detectable_effect(points: Sequence[Tuple[float, float]], level: float) -> Optional[float]:
    """Smallest effect with power at least `level`, by linear interpolation of
    the observed (effect, power) points; the first point's effect if it
    already reaches `level`; None if the level is not reached within the
    tested effects."""
    pts = sorted((float(e), float(p)) for e, p in points if p is not None)
    if not pts:
        return None
    if pts[0][1] >= level:
        return pts[0][0]
    for (e0, p0), (e1, p1) in zip(pts, pts[1:]):
        if p0 < level <= p1:
            return e0 + (level - p0) * (e1 - e0) / (p1 - p0)
    return None


def summarize_power(records_by_effect: Dict[float, Sequence[dict]], *,
                    alpha: float = CALIB_ALPHA) -> dict:
    """The observed power curve of the injected-effect datasets: for each
    effect the rejection rate of the headline (= N1; the
    fraction of datasets with p at or below alpha) and, as a reported
    sensitivity (not gating), of N2, the mean realized primary-cell mean x
    of the injected series (the effect actually delivered), the smallest
    effects with 50 and 80 percent headline power, and whether the pre-
    registered requirement (at least 50 percent power at 1.0) holds."""
    curve = {}
    for eff in sorted(records_by_effect):
        recs = [r for r in records_by_effect[eff] if "error" not in r]
        d = {"n_datasets": len(recs)}
        for name, miss in (("p_n1", True), ("p_n2", False), ("p_n2_covered", False),
                           ("p_headline", True)):
            r = _rate(recs, name, alpha, miss)
            d["power_" + name[2:]] = r["rate"]
            d["se_" + name[2:]] = r["se"]
        real = [r["realized_mean_x"] for r in recs if r.get("realized_mean_x") is not None]
        d["mean_realized_x"] = float(np.mean(real)) if real else None
        curve[float(eff)] = d
    pts = [(e, c["power_headline"]) for e, c in curve.items()]
    at, need = POWER_REQUIRED_AT
    passes = None
    if float(at) in curve and curve[float(at)]["power_headline"] is not None:
        passes = bool(curve[float(at)]["power_headline"] >= need)
    return {"alpha": alpha, "curve": curve, "n2_label": N2_SENSITIVITY_LABEL,
            "min_detectable_effect_50": min_detectable_effect(pts, 0.5),
            "min_detectable_effect_80": min_detectable_effect(pts, 0.8),
            "required": {"effect": at, "power": need}, "passes_required_power": passes}
