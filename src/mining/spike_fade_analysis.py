"""Analysis layer of the range-bound spike-fade study, Stage 1 (pre-registered:
the grid, thresholds and rules are fixed constants in the `spike_fade_*`
modules; "the design" below means that fixed rule set).

WHAT THIS MODULE IS. The exploratory plateau map, the tradable-
vehicle test, the put-affordability screen and the
ABI-1 transfer line, built on the deterministic core
(`spike_fade_study`) and the null machinery (`spike_fade_null`). It is a
RESEARCH module: it places no trade, writes no ledger row, calls no ABI-1
evaluate or record function and spends no trial slot. No network access, no
currency-symbol amounts anywhere in this file. Every function is
deterministic: the only randomness is the frozen-seed random-date null used
by the vehicle test, drawn through the null module's own streams.

VOCABULARY. A GROUP is (series, side, trigger); a CELL is a
group plus (entry rule, horizon H, take-profit TP, stop-loss SL); a PAIR is
(series, side). Positions are 0-based offsets into one series' own sealed
date index; `side` follows the core (+1 short the traded thing, -1 long).

PUBLIC ENTRY POINTS, in four groups (plateau, vehicle, put screen, transfer line):
  plateau   plateau_scores(cell_frame)          plateau score + eligible-neighbor count
      rank_cells(scored) / top_cells(scored)      the total order; top 10
                                          per pair, top 30 overall, one line per
                                          (group, entry rule)
      stability_flags(x, dates) / stability_for_cell(res, cell, dates)
                                          the pre-declared stability rule
      null_stability_pass_rate(null_cells)        the same rule on N1 null cells
      null_cells_for_stability(series, trigger, side, cell)   wires the null
                                          module's N1 draws to the core engine
  vehicle   VEHICLE_TABLE / vehicle_for / position_sign / fred_extra_lag /
      is_contaminated                     the vehicle table, sign, lag, flag
      vehicle_cell_test(vehicle, trigger_dates, cell, position, ...)
      vehicle_test_for_pair(series, side, vehicle, trigger_dates, cell, ...)
                                          a cell on the vehicle vs the random-
                                          entry null (N1 on vehicle-era anchors)
      vehicle_test_scope(scored, tstar_group)     which pairs get the test
      candidate_definition(headline_p, vehicle_result)    the candidate rule
  put screen   bs_put / bs_call, put_payoff_share, put_break_even_premium, put_screen,
      put_screen_vs_baseline, put_screen_inputs, put_screen_for_cell,
      realized_vol_series, market_k, market_k_summary, market_k_at_entries
  transfer line   engine_convention_trades(...)       trades under the engine's conventions
      abi1_transfer_line(spy_frame, vehicle_frame, signal_dates, cell, position)
      dsr_bar_annualized(n_sessions)      the exact Normal-moment DSR bar

CHOICES WHERE THE DESIGN IS SILENT OR AMBIGUOUS are made in the most literal way,
documented in the docstring of the function that makes them, and collected in
`OPEN_CHOICES` below so a report can print them; none is made silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import account_relative as _ar
from . import benchmark_relative as _br
from . import constants as _constants
from . import dsr as _dsr
from . import spike_fade_null as _null
from . import spike_fade_study as sfs

OPEN_CHOICES = (
    # plateau map
    "Stability (a): 'largest-contribution episode' = the episode with the largest "
    "signed x (contribution to the mean is x / n); S, S after removing the largest "
    "and S after removing the two largest must ALL be strictly above zero. (The "
    "three conditions imply one another in that order, so the last binds; all are "
    "evaluated.)",
    "Stability (b): the first half is the first n // 2 episodes by date, the second "
    "half the rest; a half mean of exactly zero is not sign agreement; a cell whose "
    "halves have fewer than 4 episodes passes on rule (a) alone (b not applicable).",
    "Null stability pass rate: null cells with fewer than 8 episodes are excluded "
    "from the denominator; the rate is also given among null cells with S above zero "
    "(the observed cell is picked for a high S).",
    "Plateau score: a neighbor with a non-finite S counts as ineligible; a cell that "
    "is itself ineligible gets no plateau score; the per-pair list is taken after the "
    "one-line-per-(group, entry rule) reduction.",
    # vehicle test
    "Vehicle test drops and counts anchors with no vehicle session, fewer than 252 "
    "vehicle sessions of history, a lagged position past the vehicle's end, a path "
    "that fails uniform censoring on the VEHICLE calendar (t0 + 70 sessions), and "
    "duplicates; the design does not spell out the censoring step.",
    "The FRED publication lag L is applied to the ANCHOR position on the vehicle "
    "calendar (the last vehicle session on or before t0, plus the extra "
    "FRED lag; the core's `map_anchors_to_vehicle` does the same): the vehicle test "
    "then runs at the lagged anchor, so the vehicle's median M and sigma21 are read at "
    "t0 + L (the design keeps the baselines at t0 for a FRED series traded on its "
    "own calendar) and the E1 confirmation window is (t0 + L, t0 + L + 5]. The "
    "vehicle is a Yahoo series, so its own TP / stop fills carry no further lag, and "
    "a vehicle SpikeSeries must have extra_lag 0 (otherwise the lag would count twice).",
    "Vehicle null: the null module's matched draw (gap20 spacing, exact valid n, "
    "exact D strata on the vehicle's own D, a third stratum for an undefined "
    "median) over vehicle eligible sessions (position >= 251, a 70-session path, a "
    "valid trade in the cell); a replication that cannot reach the quota is failed "
    "and counted, not padded.",
    "Vehicle excess: pp = 100 x (mean raw P&L minus the mean of the null "
    "replication means); the sigma-unit excess is the same difference of mean "
    "winsorised x; the NULL PERCENTILE is the mid-rank of the observed mean raw P&L "
    "among the replication means, and the candidate rule gates on the pp excess and "
    "that percentile.",
    "Unconditional average return over the same window: the cell's mean P&L entered "
    "on EVERY eligible session of the vehicle, plus the vehicle's mean simple daily "
    "return from its 252nd session to its last.",
    "SVXY era split: by ENTRY date (on or before 2018-02-28 is 'pre'), pooled "
    "headline unchanged; each era's excess is against the pooled null mean.",
    "vehicle_test_scope: the union of the T* pair (primary cell) and the top 3 "
    "vehicle-eligible pairs by best plateau score (each with its top plateau cell); "
    "a pair that is both appears once and runs the primary cell.",
    # put screen
    "Put screen: EXPIRY-ONLY payoff at the close H sessions after the entry close, "
    "read from the (entry rule, H, none, none) cell; realized volatility = sample "
    "(ddof 1) sd of the 21 daily log returns ending at the entry close times sqrt(252); "
    "time to expiry = H / 252 years; strikes are integer percents of the entry price.",
    "Break-even k: bisection on [0.01, 50]; 'never_profitable' (NaN) when the average "
    "put P&L is not positive even at the lower end, 'beyond_bracket' (+inf) when it is "
    "still positive at the upper end. A margin with a non-finite break-even k is NaN "
    "and never counts as exceeding the baseline.",
    "'a clear margin' of the break-even k over the market k has no number in the design: "
    "margins over the market median and 90th percentile are reported, not judged.",
    "Market k uses the implied-volatility index close ON the entry date (exact date, "
    "nothing filled forward) over the study series' own realized volatility at entry.",
    # transfer line
    "ABI-1 line, engine conventions: one position at a time (a signal while a position "
    "is open is lost, a signal on the exit bar's close is usable); take-profit level "
    "D and the stop are referenced to the FILL (the entry open); exit checks start the "
    "bar after entry; flags exit at the flag bar's own close; sigma / median gating "
    "applies only to cells that use them.",
    "ABI-1 line is PRE-TAX, with no financing or borrow cost and no cost unless "
    "`cost_bps` is given; the SPY core is the benchmark's plain buy-and-hold series "
    "(the benchmark's one-off entry cost is common to account and benchmark and "
    "cancels in the excess); sizing is 0.25 of the account's equity at the prior close.",
    "'the two largest-loss episodes': ranked by net return on the account's equity at "
    "sizing; removal drops those trades without re-admitting signals the engine had "
    "skipped. 'passes the drawdown refusals' is read as the drawdown block of "
    "`risk_refusals` (relative and absolute), the other three refusals are reported.",
)

# ---------------------------------------------------------------------------
# Plateau map: plateau score and ranking
# ---------------------------------------------------------------------------

# Ordered axes of the exit-parameter grid. "none" is the
# loosest stop and the loosest take-profit, so index 0 is always "none".
H_AXIS = tuple(sfs.H_GRID)                                  # 10 < 21 < 42 < 63
TP_AXIS = tuple(sfs.TP_GRID)                                # none < half < full
SL_AXIS = tuple(sfs.sl_label(sl) for sl in sfs.SL_GRID)     # none < 1.5 < 3.0
_H_INDEX = {h: i for i, h in enumerate(H_AXIS)}
_TP_INDEX = {t: i for i, t in enumerate(TP_AXIS)}
_SL_INDEX = {s: i for i, s in enumerate(SL_AXIS)}

MIN_ELIGIBLE_NEIGHBORS = sfs.MIN_PLATEAU_NEIGHBORS          # 3
TOP_PER_PAIR = 10
TOP_OVERALL = 30

_GROUP_COLS = ("series", "trigger", "side")
_LINE_COLS = _GROUP_COLS + ("entry",)                       # one report line per this key
_NEIGHBOR_STEPS = tuple((dh, dt, ds) for dh in (-1, 0, 1) for dt in (-1, 0, 1)
                        for ds in (-1, 0, 1) if (dh, dt, ds) != (0, 0, 0))


def _sl_axis_label(value) -> str:
    """The frame's SL value as an axis label: 'none' for None / NaN / 'none',
    the core's `sl_label` string otherwise (so 1.5 and '1.5' agree)."""
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return "none"
    return sfs.sl_label(float(value))


def plateau_scores(cell_frame: pd.DataFrame) -> pd.DataFrame:
    """Plateau score for every cell of a cell frame.

    `cell_frame` has one row per cell with at least `series, trigger, side,
    entry, H, tp, sl, mean_x, eligible` (the core's `series_frame` /
    `GroupResult.to_frame` output, concatenated over groups). Returns a copy
    with two added columns: `n_eligible_neighbors` and `plateau_score`.

    NEIGHBORS of a cell are the cells of the SAME group (series, trigger,
    side) and the SAME entry rule whose (H, TP, SL) axis indices each differ
    from the cell's by at most one step (H: 10 < 21 < 42 < 63; TP: none <
    half < full; SL: none < 1.5 < 3.0) and that differ in at least one axis:
    up to 26 (an interior cell), 7 at a grid corner. A neighbor counts only
    if it is ELIGIBLE (`eligible` true, n at least 8) and has a finite S
    (`mean_x`). The plateau score is the MEDIAN of S over the eligible
    neighbors; a cell with fewer than 3 eligible neighbors, or that is
    itself not eligible ("for each eligible cell"), gets NaN. Ineligible
    neighbors never enter the median, whatever their S.

    Raises ValueError on a missing column, an axis value outside the grid,
    or two rows for the same cell. The input is not modified.
    """
    need = set(_LINE_COLS) | {"H", "tp", "sl", "mean_x", "eligible"}
    missing = need - set(cell_frame.columns)
    if missing:
        raise ValueError(f"cell frame is missing columns {sorted(missing)}")
    out = cell_frame.copy()
    n = len(out)
    h_idx = np.empty(n, dtype=np.int64)
    t_idx = np.empty(n, dtype=np.int64)
    s_idx = np.empty(n, dtype=np.int64)
    for j, (h, t, sl) in enumerate(zip(out["H"], out["tp"], out["sl"])):
        label = _sl_axis_label(sl)
        if h not in _H_INDEX or t not in _TP_INDEX or label not in _SL_INDEX:
            raise ValueError(f"cell (H={h!r}, tp={t!r}, sl={sl!r}) is outside the "
                             "pre-registered grid")
        h_idx[j], t_idx[j], s_idx[j] = _H_INDEX[h], _TP_INDEX[t], _SL_INDEX[label]

    S = out["mean_x"].to_numpy(dtype=float)
    usable = out["eligible"].to_numpy(dtype=bool) & np.isfinite(S)

    lines = {}                                   # line key -> {(h, t, s): row position}
    keys = list(zip(*(out[c].to_numpy() for c in _LINE_COLS)))
    for j, key in enumerate(keys):
        cell = (int(h_idx[j]), int(t_idx[j]), int(s_idx[j]))
        bucket = lines.setdefault(key, {})
        if cell in bucket:
            raise ValueError(f"duplicate cell {key} {cell}")
        bucket[cell] = j

    n_nb = np.zeros(n, dtype=np.int64)
    score = np.full(n, np.nan)
    for bucket in lines.values():
        for (h, t, s), j in bucket.items():
            vals = []
            for dh, dt, ds in _NEIGHBOR_STEPS:
                k = bucket.get((h + dh, t + dt, s + ds))
                if k is not None and usable[k]:
                    vals.append(S[k])
            n_nb[j] = len(vals)
            if usable[j] and len(vals) >= MIN_ELIGIBLE_NEIGHBORS:
                score[j] = float(np.median(vals))
    out["n_eligible_neighbors"] = n_nb
    out["plateau_score"] = score
    return out


def _sort_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_h=df["H"].to_numpy(),
                     _sl=[_SL_INDEX[_sl_axis_label(v)] for v in df["sl"]],
                     _tp=[_TP_INDEX[v] for v in df["tp"]])


def rank_cells(scored: pd.DataFrame) -> pd.DataFrame:
    """Rank cells that HAVE a plateau score, best first, with a `rank` column.

    Total order: plateau score descending, then own S
    descending, then smaller H, then SL with none first (then 1.5, 3.0), then
    TP with none first (then half, full), then lexicographic order of series,
    side, trigger, entry. Cells without a plateau score are left out.
    """
    df = scored[scored["plateau_score"].notna()]
    if df.empty:
        return df.assign(rank=pd.Series(dtype="int64"))
    key = _sort_key_frame(df)
    order = key.sort_values(
        by=["plateau_score", "mean_x", "_h", "_sl", "_tp",
            "series", "side", "trigger", "entry"],
        ascending=[False, False, True, True, True, True, True, True, True],
        kind="mergesort").index
    ranked = df.loc[order].copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def top_cells(scored: pd.DataFrame, *, per_pair: int = TOP_PER_PAIR,
              overall: int = TOP_OVERALL) -> dict:
    """The plateau lists: the top `per_pair` (10) per pair
    (series, side) and the top `overall` (30) overall, ONE LINE PER DISTINCT
    (group, entry rule): among cells of one (series, trigger, side, entry) only
    the best-ranked survives. Returns {"per_pair": frame, "overall": frame},
    both in rank order.
    """
    ranked = rank_cells(scored)
    lines = ranked.drop_duplicates(subset=list(_LINE_COLS), keep="first")
    per = lines.groupby(["series", "side"], sort=False).head(per_pair)
    return {"per_pair": per, "overall": lines.head(overall)}


# ---------------------------------------------------------------------------
# Plateau map: the stability rule (pre-declared) and its null pass rate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StabilityResult:
    """Outcome of the stability rule for ONE cell.

      n, S                    traded episodes (finite x) and their mean x
      S_drop1, S_drop2        mean x after removing the single / the two
                              largest-contribution episodes (NaN if too few left)
      a_pass                  rule (a): S, S_drop1 and S_drop2 are ALL above zero
      n_first, n_second       episodes in the two date halves
      first_half_mean, second_half_mean
      b_applicable, b_pass    rule (b): both halves have at least 4 episodes and
                              their means have the same strict sign; b_pass is
                              None when (b) is not applicable
      passed                  a_pass AND (b_pass is not False)
    """
    n: int
    S: float
    S_drop1: float
    S_drop2: float
    a_pass: bool
    n_first: int
    n_second: int
    first_half_mean: float
    second_half_mean: float
    b_applicable: bool
    b_pass: Optional[bool]
    passed: bool


def _mean_or_nan(values) -> float:
    return math.fsum(values) / len(values) if len(values) else float("nan")


def stability_flags(x, dates) -> StabilityResult:
    """The pre-declared stability rule on one cell's per-episode
    x values (`dates` = the episodes' t0 dates, any order).

    (a) S remains above zero after removing the single largest-contribution
        episode AND after removing the two largest. "Contribution" of an
        episode to S = mean(x) is x / n, so the largest contribution is the
        largest x (signed): the episodes that CARRY a positive result. All
        three means must be strictly above zero (the full-sample S included:
        "remains" presupposes it; a cell with S at or below zero fails). With
        too few episodes to leave one after the removal the mean is undefined
        and the cell fails (a).
    (b) the sign of S agrees across the two halves of the episode list by
        date: episodes are ordered by date (stable for ties), the first half
        is the first n // 2 of them and the second half is the rest (so the
        extra episode of an odd n goes to the second half). Each half needs at
        least 4 episodes, else (b) is NOT APPLICABLE (`b_pass` None). Agreement
        means both half means are strictly positive or both strictly negative;
        a half mean of exactly zero is not agreement.
    A cell passes when (a) holds and (b) is not False; with (b) not
    applicable the cell passes on (a) alone. Non-finite x are dropped first.
    """
    x = np.asarray(x, dtype=float)
    d = pd.DatetimeIndex(dates)
    if len(x) != len(d):
        raise ValueError("x and dates must have the same length")
    keep = np.isfinite(x)
    x, d = x[keep], d[keep]
    order = np.argsort(d.values, kind="stable")
    x = x[order]
    n = int(len(x))
    nan = float("nan")
    ascending = np.sort(x)
    S = _mean_or_nan(list(x))
    S1 = _mean_or_nan(list(ascending[:-1])) if n >= 2 else nan
    S2 = _mean_or_nan(list(ascending[:-2])) if n >= 3 else nan
    a_pass = bool(S > 0 and S1 > 0 and S2 > 0)      # NaN compares False

    n1 = n // 2
    first, second = x[:n1], x[n1:]
    m1, m2 = _mean_or_nan(list(first)), _mean_or_nan(list(second))
    applicable = len(first) >= sfs.MIN_HALF_EPISODES and len(second) >= sfs.MIN_HALF_EPISODES
    b_pass: Optional[bool] = None
    if applicable:
        b_pass = bool((m1 > 0 and m2 > 0) or (m1 < 0 and m2 < 0))
    return StabilityResult(n=n, S=S, S_drop1=S1, S_drop2=S2, a_pass=a_pass,
                           n_first=int(len(first)), n_second=int(len(second)),
                           first_half_mean=m1, second_half_mean=m2,
                           b_applicable=applicable, b_pass=b_pass,
                           passed=bool(a_pass and b_pass is not False))


def stability_for_cell(res: "sfs.VariantResults", cell: tuple,
                       dates: pd.DatetimeIndex) -> StabilityResult:
    """`stability_flags` for one cell of a core `VariantResults`: x of its
    TRADED episodes, dated by their anchors (`dates` = the series' date index)."""
    i = res.row(cell)
    ok = res.status[i] == sfs.STATUS_TRADED
    return stability_flags(res.x[i][ok], pd.DatetimeIndex(dates)[res.t0[ok]])


def null_stability_pass_rate(null_cells: Iterable, *,
                             min_n: int = sfs.MIN_EPISODES) -> dict:
    """The pass rate of the SAME stability rule on null cells, reported beside
    the observed cells' result as the null pass rate of the same rule computed
    on the N1 null cells. `null_cells` is an iterable of `(x, dates)` pairs, one per
    null cell (for instance the primary cell of every N1 replication or the
    same cell as the observed one in each replication). Cells with fewer than
    `min_n` (8) valid episodes are counted as `n_ineligible` and excluded,
    because the observed cells the rate is compared with are eligible ones.

    Returns n_cells, n_ineligible, n_pass, pass_rate, and the same rate among
    the null cells whose S is above zero (`n_positive`,
    `pass_rate_given_positive`): the observed cell being tested was picked
    because its S is high, so the conditional rate is the like-for-like
    comparison. Rates are NaN (not an error) when the denominator is zero.
    """
    n_cells = n_inelig = n_pass = n_pos = n_pos_pass = 0
    for x, dates in null_cells:
        r = stability_flags(x, dates)
        if r.n < min_n:
            n_inelig += 1
            continue
        n_cells += 1
        n_pass += int(r.passed)
        if r.S > 0:
            n_pos += 1
            n_pos_pass += int(r.passed)
    return {
        "n_cells": n_cells, "n_ineligible": n_inelig, "n_pass": n_pass,
        "pass_rate": n_pass / n_cells if n_cells else float("nan"),
        "n_positive": n_pos,
        "pass_rate_given_positive": n_pos_pass / n_pos if n_pos else float("nan"),
    }


# ---------------------------------------------------------------------------
# Vehicle test: the tradable-vehicle table and the vehicle test
# ---------------------------------------------------------------------------

# SVXY changed from -1x to -0.5x exposure at this date; the era split is
# REPORTED (pooled, never a gate) for the VIX up_fade / SVXY pair.
SVXY_ERA_SPLIT = pd.Timestamp("2018-02-28")


@dataclass(frozen=True)
class Vehicle:
    """One row of the vehicle table: the instrument a trader
    of a (series, side) pair would actually hold, and the position taken.

      ticker      the ETF (its OWN auto-adjusted total-return closes are used)
      position    "long" or "short" (a long is +1 exposure to the vehicle,
                  evaluated with the core simulator's side = -1)
      first_date  the vehicle's first date per a data check; the
                  builder re-verifies it
      note        the first-date note recorded with the table row
    """
    ticker: str
    position: str
    first_date: pd.Timestamp
    note: str


def _vehicle(ticker: str, position: str, first: str, extra: str = "") -> Vehicle:
    note = f"{ticker} {position}, from {first}"
    if position == "short":
        note += "; borrow and financing cost ignored in Stage 1 (disclosed)"
    if extra:
        note += "; " + extra
    return Vehicle(ticker=ticker, position=position, first_date=pd.Timestamp(first),
                   note=note)


VEHICLE_TABLE = {
    ("^VIX", "up_fade"): _vehicle(
        "SVXY", "long", "2011-10-04",
        "era split at 2018-02-28 when exposure changed from -1x to -0.5x "
        "(reported pooled, not a gate)"),
    ("^VIX", "down_fade"): _vehicle("VIXY", "long", "2011-01-04"),
    ("WTI spot", "up_fade"): _vehicle("USO", "short", "2006-04-10"),
    ("WTI spot", "down_fade"): _vehicle("USO", "long", "2006-04-10"),
    ("Brent spot", "up_fade"): _vehicle("BNO", "short", "2010-06-02"),
    ("Brent spot", "down_fade"): _vehicle("BNO", "long", "2010-06-02"),
    ("Henry Hub spot", "up_fade"): _vehicle("UNG", "short", "2007-04-18"),
    ("Henry Hub spot", "down_fade"): _vehicle("UNG", "long", "2007-04-18"),
    ("GC=F", "up_fade"): _vehicle("GLD", "short", "2004-11-18"),
    ("GC=F", "down_fade"): _vehicle("GLD", "long", "2004-11-18"),
    ("HG=F", "up_fade"): _vehicle("CPER", "short", "2011-11-15"),
    ("HG=F", "down_fade"): _vehicle("CPER", "long", "2011-11-15"),
    ("DX-Y.NYB", "up_fade"): _vehicle("UUP", "short", "2007-03-01"),
    ("DX-Y.NYB", "down_fade"): _vehicle("UUP", "long", "2007-03-01"),
}
VEHICLES = VEHICLE_TABLE       # alias for the table: (series, side) -> Vehicle

# FRED publication lag: FRED daily spot files are published after the
# observation date, so a signal from one is actionable one valid observation
# later; on the vehicle calendar that is one extra session of anchor lag.
FRED_SERIES = ("WTI spot", "Brent spot", "Henry Hub spot")
# Contaminated-series list: VIX, WTI and Brent (Brent co-moves with WTI) are
# marked CONTAMINATED (some facts about their behavior inside the discovery
# window were already known when the design was fixed).
CONTAMINATED_SERIES = ("^VIX", "WTI spot", "Brent spot")
CONTAMINATION_NOTE = ("confirmatory on the discovery window only; the sealed window "
                      "is mechanically sealed but partly known for this series")

_SERIES_ALIASES = {
    "wti": "WTI spot", "dcoilwtico": "WTI spot", "wti spot": "WTI spot",
    "brent": "Brent spot", "dcoilbrenteu": "Brent spot", "brent spot": "Brent spot",
    "henry hub": "Henry Hub spot", "dhhngsp": "Henry Hub spot",
    "henry hub spot": "Henry Hub spot", "vix": "^VIX",
}


def canonical_series(name: str) -> str:
    """The canonical name of a series: FRED ids and short names
    (DCOILWTICO, WTI, ...) resolve to 'WTI spot' / 'Brent spot' / 'Henry Hub
    spot'; any other string is returned unchanged."""
    return _SERIES_ALIASES.get(str(name).casefold(), name)


def fred_extra_lag(series: str) -> int:
    """1 (extra vehicle session) for a FRED-sourced trigger series, else 0."""
    return 1 if canonical_series(series) in FRED_SERIES else 0


def is_contaminated(series: str) -> bool:
    """True for the series marked contaminated (see CONTAMINATED_SERIES)."""
    return canonical_series(series) in CONTAMINATED_SERIES


def position_sign(position: str) -> int:
    """The core simulator's `side` for a vehicle position: 'long' -> -1 (gains
    when the vehicle rises), 'short' -> +1."""
    if position == "long":
        return -1
    if position == "short":
        return +1
    raise ValueError(f"position must be 'long' or 'short', got {position!r}")


def vehicle_for(series: str, side: str) -> Vehicle:
    """The vehicle table row for a pair (KeyError for a pair with no
    tradable vehicle, e.g. the exploratory-only series)."""
    return VEHICLE_TABLE[(canonical_series(series), side)]


@dataclass(eq=False)
class VehicleResult:
    """A cell run on a vehicle as a trader of that vehicle.

    Identity: series, side, vehicle, position, sign (the core side: long = -1,
    short = +1), cell, extra_lag, contaminated (trigger series flagged in
    the contaminated-series list), B / seed of the null.

    Accounting of the anchors (each input anchor lands in exactly ONE bucket):
    n_anchors_input, dropped_no_session, dropped_past_end, dropped_short_history
    (fewer than 252 vehicle sessions through the anchor), dropped_censored
    (uniform censoring on the vehicle calendar), dropped_duplicate,
    n_anchors_kept. Of the kept anchors `n` are valid trades (finite x); the
    rest are counted in `n_kept_not_traded` (a status other than traded).

    The cell on the vehicle: n, mean_pnl (mean raw P&L per unit notional),
    mean_x (mean winsorised P&L / sigma21 of the vehicle), hit_rate, and the
    D-strata counts n_pos_d / n_nonpos_d / n_undef_d used for null matching.

    The random-entry null (N1 on vehicle-era anchors): null_mean_pnl and
    null_mean_x (means of the per-replication means), null_means_pnl (all of
    them), null_reps_used / null_reps_failed, excess_pp = 100 x (mean_pnl -
    null_mean_pnl) in percentage points, excess_sigma = mean_x - null_mean_x
    in units of the vehicle's sigma21, null_percentile = mid-rank of mean_pnl
    among the null means (null_percentile_x: the same for mean_x).
    `null_anchors` holds the drawn pseudo-anchor positions (one row per
    successful replication) when the test was run with `collect_draws`.

    Context: unconditional_mean_pnl (the same cell entered on EVERY eligible
    session of the vehicle), unconditional_mean_daily_return (mean simple
    daily return of the vehicle over the same window), window_start /
    window_end, `era` (dict pre/post or None), `trades` (one row per valid
    observed trade), anchor_dates (kept anchors on the vehicle calendar, the
    input the ABI-1 transfer line takes as its signal dates) and
    underpowered_on_vehicle (n below 8).
    """
    series: Optional[str]
    side: Optional[str]
    vehicle: Optional[str]
    position: str
    sign: int
    cell: tuple
    extra_lag: int
    contaminated: bool
    B: int
    seed: int
    n_anchors_input: int
    dropped_no_session: int
    dropped_past_end: int
    dropped_short_history: int
    dropped_censored: int
    dropped_duplicate: int
    n_anchors_kept: int
    n_kept_not_traded: int
    n: int
    mean_pnl: float
    mean_x: float
    hit_rate: float
    n_pos_d: int
    n_nonpos_d: int
    n_undef_d: int
    null_mean_pnl: float
    null_mean_x: float
    null_means_pnl: tuple
    null_reps_used: int
    null_reps_failed: int
    excess_pp: float
    excess_sigma: float
    null_percentile: float
    null_percentile_x: float
    unconditional_mean_pnl: float
    unconditional_mean_daily_return: float
    window_start: Optional[pd.Timestamp]
    window_end: Optional[pd.Timestamp]
    era: Optional[dict]
    trades: pd.DataFrame
    anchor_dates: pd.DatetimeIndex
    underpowered_on_vehicle: bool = True
    null_anchors: Optional[np.ndarray] = None


def _midrank(null_values: np.ndarray, observed: float) -> float:
    """Percentile of `observed` among `null_values`: (#below + half the ties)
    over the count; NaN with no null values."""
    if len(null_values) == 0 or not np.isfinite(observed):
        return float("nan")
    return float(((null_values < observed).sum()
                  + 0.5 * (null_values == observed).sum()) / len(null_values))


_TRADE_COLUMNS = ["anchor_date", "entry_date", "exit_date", "pnl", "x", "D",
                  "sigma", "exit_reason", "tp_not_applicable"]


def vehicle_cell_test(vehicle: "sfs.SpikeSeries", trigger_dates, cell: tuple,
                      position: str, *, extra_lag: int = 0,
                      B: int = sfs.NULL_REPLICATIONS, seed: int = sfs.N1_SEED,
                      era_split: Optional[pd.Timestamp] = None,
                      collect_draws: bool = False) -> VehicleResult:
    """Run one cell on a VEHICLE as a trader of that vehicle
    and compare it with a random-entry null on the same vehicle.

    `vehicle` is the vehicle's own sealed `SpikeSeries` (its own total-return
    closes and calendar; it must carry `extra_lag` 0: the publication lag of a
    FRED trigger series is applied to the ANCHOR below, so a lag on the
    vehicle series too would count it twice); `trigger_dates` are the trigger
    series' episode anchors t0 (dates); `cell` is any (entry, H, TP, SL) of the
    grid; `position` is 'long' or 'short' (long = core side -1). D, the median
    M, the take-profit and stop levels, sigma21, the E1 confirmation and H all
    use the VEHICLE's closes (the core simulator reads them from `vehicle`);
    the trigger series only supplies the anchor DATE.

    Anchor mapping is the core's `map_anchors_to_vehicle`: the last vehicle
    session on or before t0 plus `extra_lag` sessions (1 for FRED series);
    entry is the close of the next vehicle session. For a FRED series the lag
    therefore moves the whole test to t0 + L on the vehicle calendar: M and
    sigma21 are read at the lagged anchor and the E1 window is (t0 + L,
    t0 + L + 5] (disclosed in `OPEN_CHOICES`). Anchors are DROPPED AND
    COUNTED when there is no vehicle session, when the vehicle has fewer than
    252 sessions of history up to the anchor, when the lagged position lies
    past the vehicle's end, when the anchor fails uniform censoring
    on the vehicle calendar (a path of t0 + 70 sessions must exist; an OPEN
    CHOICE the design does not spell out, taken so the vehicle's episodes and its
    null draw come from one common eligible set) and when two anchors map to
    the same vehicle session. Anchors dated after the vehicle's cutoff raise
    SealedWindowError.

    NULL (N1 construction, `B` replications, stream `series_rng(seed, vehicle
    name)` of the null module): each replication draws pseudo-episodes from
    the vehicle's own eligible sessions (position >= 251, path of 70 sessions
    available, a VALID trade in this cell) with the null module's
    `draw_matched_anchors`: gap20 spacing and EXACTLY the observed valid n and
    the observed D-sign strata (D = s x (P_entry - M) computed on the vehicle
    at each trade's own entry; a third stratum for an undefined median). A
    replication that cannot reach the quota is failed and counted
    (`null_reps_failed`), not padded. The null mean is the average of the
    replication means; the null percentile is the mid-rank of the observed
    mean P&L among the replication means. The cell is evaluated once at every
    eligible session (vectorized), then replications only index into it.

    `era_split`: for the VIX up_fade / SVXY pair pass `SVXY_ERA_SPLIT`; the
    observed trades are then also summarised before / after it BY ENTRY DATE
    (`era`), pooled headline unchanged, not a gate; each era's excess is its
    own observed mean minus the POOLED null mean (descriptive).
    """
    sign = position_sign(position)
    cell = tuple(cell)
    if vehicle.extra_lag != 0:
        raise ValueError("the vehicle series must have extra_lag 0; the FRED lag of the "
                         "trigger series is passed as `extra_lag` of the test")
    td = pd.DatetimeIndex(trigger_dates).normalize()
    sfs.assert_sealed(vehicle.dates, vehicle.cutoff)
    sfs.assert_sealed(td, vehicle.cutoff)
    n_in = int(len(td))
    va = sfs.map_anchors_to_vehicle(td, vehicle.dates, extra_lag)
    pos, n_cens = sfs.censor_episodes(va.positions, vehicle.n)
    uniq = np.unique(pos)
    n_dup = int(len(pos) - len(uniq))
    pos = uniq

    first_pos = sfs.MEDIAN_WINDOW - 1
    all_pos = np.arange(first_pos, max(first_pos, vehicle.n - sfs.CENSOR_TAIL),
                        dtype=np.int64)
    nan = float("nan")
    empty_trades = pd.DataFrame(columns=_TRADE_COLUMNS)

    def _result(**kw) -> VehicleResult:
        base = dict(
            series=None, side=None, vehicle=None,
            position=position, sign=sign, cell=cell, extra_lag=int(extra_lag),
            contaminated=False, B=int(B), seed=int(seed), n_anchors_input=n_in,
            dropped_no_session=va.dropped_no_session, dropped_past_end=va.dropped_beyond_end,
            dropped_short_history=va.dropped_short_history, dropped_censored=n_cens,
            dropped_duplicate=n_dup, n_anchors_kept=int(len(pos)),
            n_kept_not_traded=0, n=0, mean_pnl=nan, mean_x=nan, hit_rate=nan,
            n_pos_d=0, n_nonpos_d=0, n_undef_d=0, null_mean_pnl=nan, null_mean_x=nan,
            null_means_pnl=(), null_reps_used=0, null_reps_failed=0,
            excess_pp=nan, excess_sigma=nan, null_percentile=nan, null_percentile_x=nan,
            unconditional_mean_pnl=nan, unconditional_mean_daily_return=nan,
            window_start=None, window_end=None, era=None, trades=empty_trades,
            anchor_dates=vehicle.dates[pos], underpowered_on_vehicle=True)
        base.update(kw)
        return VehicleResult(**base)

    if len(all_pos) == 0:                       # vehicle too short for any episode
        return _result()

    res = sfs.evaluate_variants(vehicle, all_pos, sign, [cell])
    ctx = vehicle.context()
    status, x, pnl = res.status[0], res.x[0], res.pnl[0]
    entry_idx, exit_idx = res.entry_idx[0], res.exit_idx[0]
    valid = (status == sfs.STATUS_TRADED) & np.isfinite(x) & np.isfinite(pnl)
    with np.errstate(invalid="ignore"):
        entry_px = np.where(valid, vehicle.p[np.where(valid, entry_idx, 0)], np.nan)
        D = sign * (entry_px - ctx.med[all_pos])
    # D strata, in the null module's coding: 0 = D > 0, 1 = D <= 0, 2 = D undefined
    strata = np.full(len(all_pos), _null.STRATUM_UNDEFINED, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        strata[D > 0] = _null.STRATUM_POS
        strata[D <= 0] = _null.STRATUM_NONPOS

    obs_rows = pos - first_pos
    obs_valid = valid[obs_rows]
    n = int(obs_valid.sum())
    rows = obs_rows[obs_valid]
    unc_pnl = float(pnl[valid].mean()) if valid.any() else nan
    p_win = vehicle.p[first_pos:]
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = p_win[1:] / p_win[:-1] - 1.0
    daily_mean = float(np.nanmean(daily)) if np.isfinite(daily).any() else nan
    w_start, w_end = vehicle.dates[first_pos], vehicle.dates[-1]
    common = dict(n_kept_not_traded=int(len(pos) - n), n=n,
                  unconditional_mean_pnl=unc_pnl,
                  unconditional_mean_daily_return=daily_mean,
                  window_start=w_start, window_end=w_end,
                  underpowered_on_vehicle=n < CANDIDATE_MIN_VEHICLE_N)
    if n == 0:
        return _result(**common)

    obs_pnl, obs_x, obs_D = pnl[rows], x[rows], D[rows]
    obs_strata = strata[rows]
    quota = tuple(int((obs_strata == code).sum()) for code in
                  (_null.STRATUM_POS, _null.STRATUM_NONPOS, _null.STRATUM_UNDEFINED))
    # absolute-position views of the pre-evaluated cell, the shape draw_matched_anchors reads
    n_sess = vehicle.n
    valid_abs = np.zeros(n_sess, dtype=bool)
    valid_abs[all_pos] = valid
    stratum_abs = np.full(n_sess, _null.STRATUM_UNDEFINED, dtype=np.int8)
    stratum_abs[all_pos] = strata
    rng = _null.series_rng(seed, vehicle.name)
    null_pnl, null_x, drawn, failed = [], [], [], 0
    for _ in range(int(B)):
        perm = rng.permutation(all_pos)
        got = _null.draw_matched_anchors(perm, valid_abs, stratum_abs, quota, n_sess)
        if got is None:
            failed += 1
            continue
        r = got - first_pos
        null_pnl.append(float(pnl[r].mean()))
        null_x.append(float(x[r].mean()))
        if collect_draws:
            drawn.append(got)
    null_pnl_arr, null_x_arr = np.array(null_pnl), np.array(null_x)
    mean_pnl, mean_x = float(obs_pnl.mean()), float(obs_x.mean())
    null_mean_pnl = float(null_pnl_arr.mean()) if len(null_pnl_arr) else nan
    null_mean_x = float(null_x_arr.mean()) if len(null_x_arr) else nan

    trades = pd.DataFrame({
        "anchor_date": vehicle.dates[pos[obs_valid]],
        "entry_date": vehicle.dates[entry_idx[rows]],
        "exit_date": vehicle.dates[exit_idx[rows]],
        "pnl": obs_pnl, "x": obs_x, "D": obs_D, "sigma": res.sigma[rows],
        "exit_reason": res.exit_reason[0][rows], "tp_not_applicable": res.tp_na[0][rows],
    })
    era = None
    if era_split is not None:
        split = pd.Timestamp(era_split)
        era = {}
        for name, m in (("pre", trades["entry_date"] <= split),
                        ("post", trades["entry_date"] > split)):
            sub = trades[m]
            em = float(sub["pnl"].mean()) if len(sub) else nan
            era[name] = {"n": int(len(sub)), "mean_pnl": em,
                         "hit_rate": float((sub["pnl"] > 0).mean()) if len(sub) else nan,
                         "excess_pp_vs_pooled_null": 100.0 * (em - null_mean_pnl)}
    return _result(
        **common, mean_pnl=mean_pnl, mean_x=mean_x, hit_rate=float((obs_pnl > 0).mean()),
        n_pos_d=quota[0], n_nonpos_d=quota[1], n_undef_d=quota[2],
        null_mean_pnl=null_mean_pnl, null_mean_x=null_mean_x,
        null_means_pnl=tuple(null_pnl), null_reps_used=int(len(null_pnl)),
        null_reps_failed=int(failed),
        excess_pp=100.0 * (mean_pnl - null_mean_pnl), excess_sigma=mean_x - null_mean_x,
        null_percentile=_midrank(null_pnl_arr, mean_pnl),
        null_percentile_x=_midrank(null_x_arr, mean_x),
        era=era, trades=trades,
        null_anchors=(np.vstack(drawn) if drawn else np.zeros((0, n), dtype=np.int64))
        if collect_draws else None)


def vehicle_test_for_pair(series: str, side: str, vehicle: "sfs.SpikeSeries",
                          trigger_dates, cell: tuple, *,
                          B: int = sfs.NULL_REPLICATIONS, seed: int = sfs.N1_SEED,
                          era_split: Optional[pd.Timestamp] = None,
                          extra_lag: Optional[int] = None,
                          collect_draws: bool = False) -> VehicleResult:
    """`vehicle_cell_test` for a study pair: the position comes from the
    vehicle table, the extra FRED lag from `fred_extra_lag` (override with
    `extra_lag`, e.g. the trigger series' own `SpikeSeries.extra_lag`), the
    SVXY era split is applied automatically to VIX up_fade (override with
    `era_split`), and the result is labelled with the pair, the vehicle and the
    contaminated-series flag."""
    name = canonical_series(series)
    v = vehicle_for(name, side)
    if era_split is None and (name, side) == ("^VIX", "up_fade"):
        era_split = SVXY_ERA_SPLIT
    lag = fred_extra_lag(name) if extra_lag is None else int(extra_lag)
    res = vehicle_cell_test(vehicle, trigger_dates, cell, v.position, extra_lag=lag,
                            B=B, seed=seed, era_split=era_split,
                            collect_draws=collect_draws)
    res.series, res.side, res.vehicle = name, side, v.ticker
    res.contaminated = is_contaminated(name)
    return res


def null_cells_for_stability(series: "sfs.SpikeSeries", trigger: str, side: str,
                             cell: tuple, *, B: int = _null.MAP_B,
                             seed: int = _null.MAP_SEED) -> list:
    """The N1 null cells the stability rule is compared with: for one
    group and one cell, the `(x, dates)` of the pseudo-episodes of every
    successful N1 replication, ready for `null_stability_pass_rate`. It wires
    the null module's `n1_cell_null` (pseudo-episodes matched to the group's
    primary-cell valid n and D strata, gap20 spacing) to the core engine: the
    drawn anchors are evaluated in the requested `cell` and the episodes that
    trade there are kept. Failed replications are simply absent."""
    mn = _null.n1_cell_null(series, trigger, side, cells=[tuple(cell)], B=B, seed=seed,
                            collect_draws=True)
    A = mn.anchors
    if A is None or A.size == 0:
        return []
    res = sfs.evaluate_variants(series, A.ravel(), side, [tuple(cell)])
    xs = res.x[0].reshape(A.shape)
    out = []
    for r in range(A.shape[0]):
        ok = np.isfinite(xs[r])
        out.append((xs[r][ok], series.dates[A[r][ok]]))
    return out


def _cell_of_row(row) -> tuple:
    sl = _sl_axis_label(row.sl)
    return (row.entry, int(row.H), row.tp, None if sl == "none" else float(sl))


def vehicle_test_scope(scored: pd.DataFrame, tstar_group: Optional[tuple] = None,
                       n_top: int = 3) -> list:
    """The vehicle-test scope, exactly: the pair that attains T*
    and each of the top `n_top` (3) pairs by best plateau score among the
    vehicle-eligible pairs (the 7 series x 2 sides of `VEHICLE_TABLE`), and
    nothing else. `scored` is a `plateau_scores` frame; `tstar_group` is
    (series, side, trigger) of the T* group, or None.

    Returns a list of dicts in run order: the T* pair first (cell =
    `PRIMARY_CELL` on its own group), then the plateau pairs in rank order, each
    with its TOP-RANKED plateau cell and the group (trigger) that cell belongs
    to. A pair that is both T* and in the top 3 appears once, role
    'tstar+plateau_top3', with the primary cell (the design runs the primary
    cell for T*'s pair). A T* pair with no tradable vehicle is listed with
    `vehicle` None and `run` False (the report says 'vehicle test not run').
    Keys: series, side, trigger, cell, role, vehicle, position, run,
    plateau_score (None for a T*-only entry)."""
    ranked = rank_cells(scored)
    top, seen = [], set()
    for row in ranked.itertuples():
        pair = (canonical_series(row.series), row.side)
        if pair in seen:
            continue
        seen.add(pair)
        if pair in VEHICLE_TABLE:
            top.append((pair, row))
            if len(top) == n_top:
                break

    def entry(series, side, trigger, cell, role, plateau_score):
        v = VEHICLE_TABLE.get((canonical_series(series), side))
        return {"series": canonical_series(series), "side": side, "trigger": trigger,
                "cell": cell, "role": role, "plateau_score": plateau_score,
                "vehicle": None if v is None else v.ticker,
                "position": None if v is None else v.position, "run": v is not None}

    out = []
    tstar_pair = None
    if tstar_group is not None:
        series, side, trigger = tstar_group
        tstar_pair = (canonical_series(series), side)
        in_top = any(pair == tstar_pair for pair, _ in top)
        out.append(entry(series, side, trigger, sfs.PRIMARY_CELL,
                         "tstar+plateau_top3" if in_top else "tstar", None))
    for pair, row in top:
        if pair == tstar_pair:
            continue
        out.append(entry(row.series, row.side, row.trigger, _cell_of_row(row),
                         "plateau_top3", float(row.plateau_score)))
    return out


# ---------------------------------------------------------------------------
# Vehicle test: the candidate definition
# ---------------------------------------------------------------------------

CANDIDATE_HEADLINE_P_MAX = sfs.HEADLINE_P_MAX          # 0.10
CANDIDATE_MIN_NULL_PERCENTILE = 0.80
CANDIDATE_MIN_VEHICLE_N = sfs.MIN_EPISODES             # 8


@dataclass(frozen=True)
class CandidateVerdict:
    """The candidate rule for the T* pair. `reasons` lists EVERY
    failing condition (empty for a candidate); `status` is 'candidate' or the
    first failing reason. `contaminated` / `contamination_note` carry the
    contamination wording when the pair's series is VIX, WTI or Brent."""
    is_candidate: bool
    status: str
    reasons: tuple
    headline_p: Optional[float]
    underpowered_on_vehicle: bool
    contaminated: bool
    contamination_note: str


def candidate_definition(headline_p: Optional[float], vehicle_result,
                         *, series: Optional[str] = None) -> CandidateVerdict:
    """A candidate exists only if (i) the confirmatory-test headline p is at most 0.10
    AND (ii) for the pair attaining T*, the vehicle-level excess over the
    random-entry null is positive with null percentile at least 0.80 and n at
    least 8 vehicle episodes. With n below 8 the pair is 'underpowered on the
    vehicle' and no candidate can be declared. `vehicle_result` is the T*
    pair's `VehicleResult` (any object with `n`, `excess_pp`, `null_percentile`
    and optionally `series` works), or None when the vehicle test was not run
    (no candidate). "Excess" is the excess mean P&L in percentage points and
    the percentile is that of the mean P&L (both the vehicle-test quantities);
    the sigma-unit excess is reported beside them, not gated. The contaminated
    flag comes from the pair's series (`vehicle_result.series`, or `series`).
    """
    reasons = []
    if headline_p is None or not np.isfinite(headline_p):
        reasons.append("headline_p_unavailable")
    elif not headline_p <= CANDIDATE_HEADLINE_P_MAX:
        reasons.append("headline_p_above_max")
    underpowered = False
    if vehicle_result is None:
        reasons.append("vehicle_test_not_run")
    else:
        if not vehicle_result.n >= CANDIDATE_MIN_VEHICLE_N:
            underpowered = True
            reasons.append("underpowered_on_vehicle")
        if not vehicle_result.excess_pp > 0:
            reasons.append("vehicle_excess_not_positive")
        if not vehicle_result.null_percentile >= CANDIDATE_MIN_NULL_PERCENTILE:
            reasons.append("null_percentile_below_min")
    name = series if series is not None else getattr(vehicle_result, "series", None)
    contaminated = bool(name is not None and is_contaminated(name))
    return CandidateVerdict(
        is_candidate=not reasons, status="candidate" if not reasons else reasons[0],
        reasons=tuple(reasons),
        headline_p=None if headline_p is None else float(headline_p),
        underpowered_on_vehicle=underpowered, contaminated=contaminated,
        contamination_note=CONTAMINATION_NOTE if contaminated else "")


# ---------------------------------------------------------------------------
# Put screen: the put-affordability screen (a screen, not an options backtest)
# ---------------------------------------------------------------------------
# No historical option price series exists in the repo or for free, so the
# screen never pretends to know an option price. It measures (1) the most a
# put buyer could pay and still break even (the mean expiry payoff), (2) what
# a Black-Scholes put would cost at k times the trailing realized volatility
# and the k at which the average put P&L is zero, (3) the same on random dates
# as the baseline, and (4) the MARKET k where an implied-volatility index
# exists. VIX options are priced off same-expiry futures, so the VIX version
# is indicative only.
#
# CONVENTIONS. Realized volatility is the sample (ddof = 1) standard
# deviation of the last 21 daily LOG returns ending at the entry close,
# multiplied by sqrt(252). Implied-volatility indices are quoted in percent,
# so market k = (index / 100) / realized volatility. Time to expiry is the
# horizon in sessions divided by 252 (calendar time is not modelled). Rates
# and dividends are zero (a design choice; American exercise and rates
# change the premium by under 3 percent).

TRADING_DAYS_PER_YEAR = 252
RV_WINDOW = 21
PUT_STRIKE_PCTS = (100, 95, 90)          # strikes as percent of the entry price
PUT_K_GRID = (1.0, 1.25, 1.5, 2.0)       # implied volatility = k x realized volatility
# Bisection bracket for the break-even k. The average put P&L is strictly
# DECREASING in k (a put's vega is positive), so a sign change inside the
# bracket is a unique root. Below the lower end the premium is practically
# zero, so a negative P&L there means the puts lose whatever they cost
# ("never_profitable", NaN); a positive P&L at the upper end means the
# payoffs are too large for any volatility in the bracket to price
# ("beyond_bracket", +inf).
BREAK_EVEN_K_BRACKET = (0.01, 50.0)
_BISECT_TOL = 1e-12
_BISECT_MAX_ITER = 200


def _bs_terms(spot, strike, vol, t):
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    v = np.asarray(vol, dtype=float)
    T = np.asarray(t, dtype=float)
    sd = v * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + 0.5 * sd * sd) / sd
    return S, K, sd, d1, d1 - sd


def _as_scalar_or_array(out):
    out = np.asarray(out, dtype=float)
    return float(out) if out.ndim == 0 else out


def bs_put(spot, strike, vol, t):
    """Black-Scholes European put, ZERO rate and dividend: K N(-d2) - S N(-d1)
    with d1 = (ln(S/K) + vol^2 t / 2) / (vol sqrt t). `t` in years. Broadcasts
    over its arguments. Exactly zero volatility or exactly zero time returns
    the intrinsic value max(K - S, 0); a NaN (or negative) volatility or time
    returns NaN, never a silent intrinsic value. Validated in the tests against
    closed forms, an independent library's values and put-call parity."""
    from scipy.special import ndtr
    S, K, sd, d1, d2 = _bs_terms(spot, strike, vol, t)
    with np.errstate(invalid="ignore"):
        price = K * ndtr(-d2) - S * ndtr(-d1)
    intrinsic = np.maximum(K - S, 0.0)
    return _as_scalar_or_array(
        np.where(sd > 0, price, np.where(sd == 0, intrinsic, np.nan)))


def bs_call(spot, strike, vol, t):
    """Black-Scholes European call, zero rate and dividend (used for the
    put-call parity check): S N(d1) - K N(d2). Same NaN / zero rules as
    `bs_put`."""
    from scipy.special import ndtr
    S, K, sd, d1, d2 = _bs_terms(spot, strike, vol, t)
    with np.errstate(invalid="ignore"):
        price = S * ndtr(d1) - K * ndtr(d2)
    intrinsic = np.maximum(S - K, 0.0)
    return _as_scalar_or_array(
        np.where(sd > 0, price, np.where(sd == 0, intrinsic, np.nan)))


def put_payoff_share(entry_px, expiry_px, strike_pct: float):
    """Expiry payoff of a put bought at the entry close with strike =
    `strike_pct` x entry price, as a SHARE of the entry price:
    max(strike_pct - P_expiry / P_entry, 0). EXPIRY-ONLY: the cell's
    take-profit and stop never enter."""
    entry = np.asarray(entry_px, dtype=float)
    expiry = np.asarray(expiry_px, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.maximum(float(strike_pct) - expiry / entry, 0.0)


def _usable(entry_px, expiry_px, rv=None) -> np.ndarray:
    entry = np.asarray(entry_px, dtype=float)
    expiry = np.asarray(expiry_px, dtype=float)
    ok = np.isfinite(entry) & (entry > 0) & np.isfinite(expiry) & (expiry >= 0)
    if rv is not None:
        r = np.asarray(rv, dtype=float)
        ok &= np.isfinite(r) & (r > 0)
    return ok


def put_break_even_premium(entry_px, expiry_px,
                           strike_pcts: Sequence[int] = PUT_STRIKE_PCTS) -> dict:
    """The break-even premium per strike (percent of entry as the key): the
    mean expiry payoff, as a share of the entry price, over the usable
    episodes. It is the most a buyer could pay per unit of entry price and
    break even on average, before any pricing model."""
    ok = _usable(entry_px, expiry_px)
    entry = np.asarray(entry_px, dtype=float)[ok]
    expiry = np.asarray(expiry_px, dtype=float)[ok]
    return {int(m): (float(put_payoff_share(entry, expiry, m / 100.0).mean())
                     if ok.any() else float("nan")) for m in strike_pcts}


@dataclass(eq=False)
class PutScreen:
    """`put_screen` output. `results[strike_pct]` is a dict with
    break_even_premium, avg_premium {k: value}, avg_pnl {k: value}, break_even_k
    and break_even_k_status ('ok' / 'never_profitable' / 'beyond_bracket'); all
    money quantities are shares of the entry price. n_used / n_dropped count
    the episodes with and without usable inputs (finite positive entry, finite
    non-negative expiry, finite positive realized volatility)."""
    H: int
    strike_pcts: tuple
    ks: tuple
    n_used: int
    n_dropped: int
    results: dict


def _break_even_k(payoff: np.ndarray, m: float, rv: np.ndarray, T: float):
    lo, hi = BREAK_EVEN_K_BRACKET
    mean_payoff = float(payoff.mean())

    def f(k):
        return mean_payoff - float(np.mean(bs_put(1.0, m, k * rv, T)))

    if f(lo) <= 0.0:
        return float("nan"), "never_profitable"
    if f(hi) >= 0.0:
        return float("inf"), "beyond_bracket"
    for _ in range(_BISECT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECT_TOL:
            break
    return 0.5 * (lo + hi), "ok"


def put_screen(entry_px, expiry_px, rv, *, H: int,
               strike_pcts: Sequence[int] = PUT_STRIKE_PCTS,
               ks: Sequence[float] = PUT_K_GRID) -> PutScreen:
    """The put-affordability screen for one set of episodes (the T* pair's / a top plateau
    cell's real episodes, or the caller's random-date anchors as the
    BASELINE): per strike, the break-even premium; the average put premium and
    average put P&L (payoff minus premium, shares of the entry price) at each
    k in `ks` with the Black-Scholes premium at volatility k x `rv` (the
    trailing 21-session realized volatility at each entry, annualized) and
    time H / 252; and the break-even k, the k at which the average P&L is
    zero (bisection on `BREAK_EVEN_K_BRACKET`; not-attainable cases are
    labelled, not guessed). `H` is the expiry horizon in sessions."""
    entry = np.asarray(entry_px, dtype=float)
    expiry = np.asarray(expiry_px, dtype=float)
    vol = np.asarray(rv, dtype=float)
    ok = _usable(entry, expiry, vol)
    e, x, v = entry[ok], expiry[ok], vol[ok]
    T = float(H) / TRADING_DAYS_PER_YEAR
    results = {}
    for pct in strike_pcts:
        m = int(pct) / 100.0
        row = {"break_even_premium": float("nan"), "avg_premium": {}, "avg_pnl": {},
               "break_even_k": float("nan"), "break_even_k_status": "no_episodes"}
        if ok.any():
            payoff = put_payoff_share(e, x, m)
            row["break_even_premium"] = float(payoff.mean())
            for k in ks:
                prem = bs_put(1.0, m, float(k) * v, T)
                row["avg_premium"][float(k)] = float(np.mean(prem))
                row["avg_pnl"][float(k)] = float(np.mean(payoff - prem))
            row["break_even_k"], row["break_even_k_status"] = _break_even_k(payoff, m, v, T)
        results[int(pct)] = row
    return PutScreen(H=int(H), strike_pcts=tuple(int(p) for p in strike_pcts),
                     ks=tuple(float(k) for k in ks), n_used=int(ok.sum()),
                     n_dropped=int((~ok).sum()), results=results)


def put_screen_vs_baseline(observed: PutScreen, baseline: Optional[PutScreen],
                           market_k: Optional[Iterable[float]] = None) -> dict:
    """Per strike: the observed break-even k against the random-date
    baseline's (a model misfit shows up there too) and, where market k values
    are supplied, against their median and 90th percentile. The screen rule: "a put is
    only worth considering where the break-even k exceeds the market k by a
    clear margin AND exceeds the random-date baseline break-even k". The
    baseline condition is decided here (`exceeds_baseline`); "a clear margin"
    over the market is NOT given a number by the design, so the margins are
    reported (over the median and the 90th percentile) and left to the reader.
    A margin is NaN, and `exceeds_baseline` False, whenever a break-even k
    involved is not a finite number (not attainable, +inf beyond the bracket,
    or no baseline was supplied): the baseline margin needs BOTH break-even
    values finite, the market margins need the observed one finite. Read the
    `*_status` fields for why; nothing is imputed.
    """
    summary = market_k_summary(list(market_k)) if market_k is not None else None
    out = {}
    for pct in observed.strike_pcts:
        o = observed.results[pct]["break_even_k"]
        if baseline is not None:
            b = baseline.results[pct]["break_even_k"]
            b_status = baseline.results[pct]["break_even_k_status"]
        else:
            b, b_status = float("nan"), "no_baseline"
        finite_o = math.isfinite(o)
        margin = o - b if (finite_o and math.isfinite(b)) else float("nan")
        row = {"observed_break_even_k": o, "baseline_break_even_k": b,
               "observed_status": observed.results[pct]["break_even_k_status"],
               "baseline_status": b_status,
               "margin_over_baseline": margin,
               "exceeds_baseline": bool(margin > 0),
               "margin_over_market_median": float("nan"),
               "margin_over_market_p90": float("nan")}
        if summary is not None and summary["n"] > 0 and finite_o:
            row["margin_over_market_median"] = o - summary["median"]
            row["margin_over_market_p90"] = o - summary["p90"]
        out[int(pct)] = row
    return out


def realized_vol_series(p) -> np.ndarray:
    """Trailing 21-session realized volatility at every session: the sample
    (ddof = 1) standard deviation of the 21 daily LOG returns ending at
    that session's close, times sqrt(252). NaN until 21 returns exist and for
    any window that contains an invalid close. Causal (uses closes up to and
    including the session)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    if n <= RV_WINDOW:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
    r = np.full(n, np.nan)
    r[1:] = lp[1:] - lp[:-1]
    view = np.lib.stride_tricks.sliding_window_view(r, RV_WINDOW)   # row s = r[s:s+21]
    finite = np.isfinite(view).all(axis=1)
    sd = np.full(len(view), np.nan)
    if finite.any():
        sd[finite] = np.std(view[finite], axis=1, ddof=1)
    out[RV_WINDOW - 1:] = sd * math.sqrt(TRADING_DAYS_PER_YEAR)
    return out


def realized_vol_annualized(closes, window: int = RV_WINDOW) -> float:
    """Annualized realized volatility at the LAST of the given closes: needs
    `window` + 1 closes; `window` must be the study's 21."""
    if window != RV_WINDOW:
        raise ValueError(f"the study fixes the realized-volatility window at {RV_WINDOW}")
    closes = np.asarray(closes, dtype=float)
    if len(closes) < window + 1:
        return float("nan")
    return float(realized_vol_series(closes[-(window + 1):])[-1])


def put_screen_inputs(series: "sfs.SpikeSeries", t0, cell: tuple,
                      side: str = "up_fade") -> dict:
    """Entry price, expiry price and entry-date realized volatility for the
    put screen, from a series' own closes. `t0` are anchor POSITIONS (real
    episodes or random-date anchors); `cell` supplies the entry rule and H
    only (the put is EXPIRY-ONLY: TP and SL are ignored, so the (entry, H,
    none, none) cell is evaluated, expiry = the close H sessions after the
    entry close); `side` is the side the entry confirmation (E1) is judged on
    (a put is bought after an up-spike: 'up_fade'). Episodes that do not
    trade under this cell are left out."""
    entry_rule, H = cell[0], cell[1]
    res = sfs.evaluate_variants(series, np.asarray(t0, dtype=np.int64), side,
                                [(entry_rule, H, "none", None)])
    ok = res.status[0] == sfs.STATUS_TRADED
    entry_idx = res.entry_idx[0][ok]
    exit_idx = res.exit_idx[0][ok]
    rv = realized_vol_series(series.p)
    return {"t0": res.t0[ok], "H": int(H), "entry_dates": series.dates[entry_idx],
            "entry_px": series.p[entry_idx], "expiry_px": series.p[exit_idx],
            "rv": rv[entry_idx]}


# Implied-volatility proxies: IV index -> the underlying
# whose realized volatility it is divided by; and the study series each one
# serves as the market-k proxy for.
IV_PROXY_TABLE = {"^VIX": "SPY", "^OVX": "WTI spot", "^GVZ": "GC=F", "^VVIX": "^VIX"}
STUDY_SERIES_IV_PROXY = {"^VIX": "^VVIX", "WTI spot": "^OVX", "GC=F": "^GVZ"}


def market_k(iv_pct, rv_annual):
    """Market k = implied volatility over realized volatility, with the
    implied-volatility INDEX quoted in percent: (iv_pct / 100) / rv_annual."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (np.asarray(iv_pct, dtype=float) / 100.0) / np.asarray(rv_annual, dtype=float)
    return _as_scalar_or_array(out)


def market_k_summary(values) -> dict:
    """n, median, 10th and 90th percentile (numpy 'linear') of the finite
    market-k values."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        nan = float("nan")
        return {"n": 0, "median": nan, "p10": nan, "p90": nan}
    return {"n": int(len(v)), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def market_k_at_entries(iv_close: pd.Series, entry_dates, rv_at_entry) -> np.ndarray:
    """Market k at each entry: the implied-volatility index close ON THE
    ENTRY DATE (exact date, nothing filled forward: a missing print is NaN
    rather than an invented one) over the underlying's realized volatility at
    that entry."""
    iv = pd.Series(iv_close)
    iv = iv[~iv.index.duplicated(keep="first")]
    vals = iv.reindex(pd.DatetimeIndex(entry_dates)).to_numpy(dtype=float)
    return np.asarray(market_k(vals, rv_at_entry), dtype=float)


def put_screen_for_cell(series: "sfs.SpikeSeries", t0, cell: tuple, *, side: str = "up_fade",
                        baseline_t0=None, iv_close: Optional[pd.Series] = None,
                        strike_pcts: Sequence[int] = PUT_STRIKE_PCTS,
                        ks: Sequence[float] = PUT_K_GRID) -> dict:
    """The whole put-affordability screen for one cell of one series: the episodes'
    put inputs (`put_screen_inputs`), the screen (`put_screen`), the same
    screen on the caller-supplied random-date anchors `baseline_t0` (positions,
    e.g. N1 draws) as the baseline, and the MARKET k at the episodes' entries
    where an implied-volatility index close series `iv_close` (percent, indexed
    by date) is supplied (its realized-volatility denominator is the series'
    own, so VVIX goes with the VIX series, OVX with WTI, GVZ with gold).
    Returns {'inputs', 'observed', 'baseline', 'comparison', 'market_k',
    'market_k_summary'}; parts that were not asked for are None."""
    inp = put_screen_inputs(series, t0, cell, side)
    obs = put_screen(inp["entry_px"], inp["expiry_px"], inp["rv"], H=inp["H"],
                     strike_pcts=strike_pcts, ks=ks)
    base = None
    if baseline_t0 is not None:
        binp = put_screen_inputs(series, baseline_t0, cell, side)
        base = put_screen(binp["entry_px"], binp["expiry_px"], binp["rv"], H=binp["H"],
                          strike_pcts=strike_pcts, ks=ks)
    mk = summary = None
    if iv_close is not None:
        mk = market_k_at_entries(iv_close, inp["entry_dates"], inp["rv"])
        summary = market_k_summary(mk)
    comparison = (put_screen_vs_baseline(obs, base, market_k=mk)
                  if (base is not None or mk is not None) else None)
    return {"inputs": inp, "observed": obs, "baseline": base, "comparison": comparison,
            "market_k": mk, "market_k_summary": summary}


# ---------------------------------------------------------------------------
# Transfer line: the ABI-1 transfer line (a descriptive account build; it scores
# nothing, records nothing and never touches the trial ledger)
# ---------------------------------------------------------------------------
# The account the pipeline would score for one pair and cell: the SPY core
# (held at all times, the identical daily series the evaluator subtracts) plus
# a vehicle overlay at 0.25 of equity. It follows the engine's conventions
# (mining.backtest_engine): a signal fires at a bar's close, the entry fills
# at the NEXT bar's OPEN, exit checks begin the bar AFTER the entry, a
# take-profit / stop / time-stop exit fills at the flag bar's own CLOSE, one
# position at a time (a signal that fires while a position is open is lost, as
# in the engine), and a position still open at the end is closed at the last
# close. Stage 1's own fills differ (entry at the close of the session after
# the signal, TP / stop fills one close after the flag): Stage 2 must re-derive
# the cell under these conventions. The overlay is marked to market every
# session on actual price paths (share counts fixed for the life of a trade),
# as in the repo's other overlay builders (beta_neutral_overlay, long_overlay).
#
# NOT MODELLED (each flatters the strategy, all disclosed): tax (the real
# evaluator nets short-term tax through a realization walk; this line is
# PRE-TAX), margin financing of the overlay, borrow cost on a short vehicle,
# and every cost except the optional round-trip `cost_bps` (default zero: no
# cost is invented).

OVERLAY_NOTIONAL = 0.25
# "About 1.02" annualized (constants.py: the in-sample bar at the standard
# DAILY_SR_VARIANCE / ABI1_N_TRIALS). The exact bar depends on the number of
# sessions; `dsr_bar_annualized` gives it for the account's own count.
IN_SAMPLE_BAR_ANNUALIZED_SHARPE = 1.02
N_DROP_WORST = 2
ABSOLUTE_MAX_DRAWDOWN_CAP = _ar.ABSOLUTE_MAX_DRAWDOWN_CAP     # 0.35, the pipeline's configured cap
MIN_ACCOUNT_SESSIONS = _ar.MIN_ACCOUNT_SESSIONS               # 252

_ENGINE_CONVENTIONS = (
    "signal at a bar's close; entry at the next bar's open; exit checks begin the bar "
    "after entry; take-profit / stop / time-stop fill at the flag bar's own close; one "
    "position at a time; open at the end -> closed at the last close; pre-tax; no "
    "financing or borrow; cost only if cost_bps is given")


def dsr_bar_annualized(n_sessions: int) -> float:
    """The annualized daily-excess Sharpe an account of `n_sessions` sessions
    needs for the pipeline's own deflated Sharpe to reach DSR_PASS_THRESHOLD at
    the standard DAILY_SR_VARIANCE / ABI1_N_TRIALS, at Normal moments (skew 0,
    kurtosis 3). About 1.02 near 5000 sessions; higher for shorter accounts.
    Solved by bisection on `dsr.deflated_sharpe_ratio`; NaN when it is not
    computable (fewer than 3 sessions)."""
    if n_sessions < 3:
        return float("nan")
    lo, hi = 0.0, 1.0
    try:
        def clears(sr):
            return _dsr.deflated_sharpe_ratio(
                sr=sr, n=int(n_sessions), skew=0.0, kurtosis=3.0,
                n_trials=_constants.ABI1_N_TRIALS,
                sr_variance=_constants.DAILY_SR_VARIANCE) >= _constants.DSR_PASS_THRESHOLD
        while not clears(hi):
            hi *= 2.0
            if hi > 1e3:
                return float("nan")
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if clears(mid):
                hi = mid
            else:
                lo = mid
    except ValueError:
        return float("nan")
    return float(hi * math.sqrt(TRADING_DAYS_PER_YEAR))


@dataclass(eq=False)
class EngineTrades:
    """`engine_convention_trades` output: `trades` (dicts with signal_pos,
    entry_pos, exit_pos, reason, entry_px) and the counts of signals that did
    not become trades (each signal lands in exactly one bucket)."""
    trades: list
    n_signals: int
    n_skipped_overlap: int
    n_skipped_no_entry: int
    n_skipped_undefined: int


def engine_convention_trades(open_, close_, signal_positions, cell: tuple, sign: int, *,
                             med=None, sigma=None) -> EngineTrades:
    """Trades of one cell under the ENGINE's conventions (see the section
    comment above), for signal bars `signal_positions` on a price path.

    `open_` / `close_` are the vehicle's opens and closes; `sign` is the core
    side (+1 short, -1 long); `cell` = (entry, H, TP, SL). Entry: E0 fills at
    the open of signal + 1; E1 waits for the first close in (t0, t0 + 5] below
    the previous close (short) / above it (long) and fills at the open of the
    next bar. The take-profit level uses D = s x (P_entry - M) with P_entry the
    FILL (the open) and M the median at the signal bar; it applies only when
    D > 0. The stop flags at the first close k < H with s x ln(P_k / P_entry)
    at least SL x sigma21. The exit is the flag bar's own close; on the close
    k = H it is the time stop (H sessions after the entry bar); when the data
    end first the position closes at the last close ('end_of_data'). A signal
    is accepted only if the entry bar is after the previous trade's exit bar
    (the engine is flat at the next bar's open only then); a signal on the
    exit bar's own close is usable. `med` / `sigma` (arrays over positions) are
    required only for TP / SL cells; an undefined value skips that signal
    (counted), never invents one. Every signal position must lie in
    [0, len(path) - 1]: a position outside it raises ValueError before anything
    is evaluated (a negative one would index from the end of the path, that is
    the future).
    """
    entry_rule, H, tp, sl = cell
    if entry_rule not in sfs.ENTRY_RULES or tp not in sfs.TP_FRACTION:
        raise ValueError(f"unknown entry rule or take-profit in cell {cell!r}")
    if not (isinstance(H, (int, np.integer)) and H >= 1):
        raise ValueError(f"H must be a positive integer, got {H!r}")
    frac = sfs.TP_FRACTION[tp]
    if frac is not None and med is None:
        raise ValueError("a take-profit cell needs the median array `med`")
    if sl is not None and sigma is None:
        raise ValueError("a stop-loss cell needs the sigma21 array `sigma`")
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 (short) or -1 (long), got {sign!r}")
    s = int(sign)
    o = np.asarray(open_, dtype=float)
    c = np.asarray(close_, dtype=float)
    n = len(c)
    signals = np.unique(np.asarray(signal_positions, dtype=np.int64))
    if len(signals) and (signals[0] < 0 or signals[-1] > n - 1):
        raise ValueError(
            f"signal positions must lie in [0, {n - 1}], got {int(signals[0])}..."
            f"{int(signals[-1])} (a negative position would index from the END of "
            "the path, i.e. the future)")
    trades = []
    skipped_overlap = skipped_no_entry = skipped_undefined = 0
    busy_until = -1
    for t0 in signals:
        t0 = int(t0)
        if entry_rule == "E0":
            i0 = t0 + 1
        else:
            i0 = -1
            for u in range(t0 + 1, t0 + sfs.E1_CONFIRM_SESSIONS + 1):
                if u > n - 1:
                    break
                if (s > 0 and c[u] < c[u - 1]) or (s < 0 and c[u] > c[u - 1]):
                    i0 = u + 1
                    break
            if i0 < 0:
                skipped_no_entry += 1
                continue
        if i0 > n - 1:
            skipped_no_entry += 1
            continue
        if i0 <= busy_until:
            skipped_overlap += 1
            continue
        if frac is not None and not np.isfinite(med[t0]):
            skipped_undefined += 1
            continue
        if sl is not None and not np.isfinite(sigma[t0]):
            skipped_undefined += 1
            continue
        pe = float(o[i0])
        D = s * (pe - float(med[t0])) if frac is not None else float("nan")
        use_tp = frac is not None and D > 0
        exit_pos, reason = n - 1, "end_of_data"
        for k in range(1, H + 1):
            j = i0 + k
            if j > n - 1:
                break                                   # data end: keep end_of_data
            if k == H:
                exit_pos, reason = j, "time_stop"
                break
            stop_flag = sl is not None and s * math.log(c[j] / pe) >= sl * float(sigma[t0])
            tp_flag = use_tp and s * (pe - c[j]) >= frac * D
            if stop_flag or tp_flag:
                exit_pos = j
                reason = "stop_loss" if stop_flag else "take_profit"
                break
        trades.append({"signal_pos": t0, "entry_pos": i0, "exit_pos": int(exit_pos),
                       "reason": reason, "entry_px": pe})
        busy_until = int(exit_pos)
    return EngineTrades(trades=trades, n_signals=int(len(signals)),
                        n_skipped_overlap=skipped_overlap,
                        n_skipped_no_entry=skipped_no_entry,
                        n_skipped_undefined=skipped_undefined)


@dataclass(eq=False)
class AccountReport:
    """The ABI-1 transfer-line numbers for one account.

      excess_sharpe_annualized   annualized daily Sharpe of the LOG excess of the
                                 account over its SPY core (the series ABI-1
                                 scores), mean / sd(ddof 1) x sqrt(252)
      in_sample_bar, meets_in_sample_bar   the bar (about 1.02) and 'at or above'
      dsr_bar_annualized         the exact Normal-moment bar for this many sessions
      max_dd_account / max_dd_spy / absolute_cap   maximum drawdowns (non-positive
                                 fractions) and the absolute cap (-0.35)
      dd_relative_passed / dd_absolute_passed / dd_passed   drawdown not deeper
                                 than SPY's; not deeper than the cap; both
      risk_refusals              the pipeline's own four-refusal block for the
                                 account vs SPY (account_relative.risk_refusals)
      terminal_wealth_account / _spy, n_sessions, n_trades, below_min_sessions
    """
    n_sessions: int
    n_trades: int
    excess_sharpe_annualized: float
    in_sample_bar: float
    meets_in_sample_bar: bool
    dsr_bar_annualized: float
    max_dd_account: float
    max_dd_spy: float
    absolute_cap: float
    dd_relative_passed: bool
    dd_absolute_passed: bool
    dd_passed: bool
    risk_refusals: dict
    terminal_wealth_account: float
    terminal_wealth_spy: float
    below_min_sessions: bool


@dataclass(eq=False)
class TransferLine:
    """`abi1_transfer_line` output: `full` and `ex_worst` (the same numbers with
    the `n_drop` largest-loss episodes removed) as `AccountReport`s, the trade
    table (`trades`, every trade of the full account; `dropped_trades` the
    removed ones), the daily series of the full account (`account_daily`,
    `core_daily`, `excess_daily`), the ex-worst account series, the signal
    accounting and the conventions used."""
    full: AccountReport
    ex_worst: AccountReport
    trades: pd.DataFrame
    dropped_trades: pd.DataFrame
    account_daily: pd.Series
    core_daily: pd.Series
    excess_daily: pd.Series
    account_daily_ex_worst: pd.Series
    n_signals: int
    n_skipped_overlap: int
    n_skipped_no_entry: int
    n_skipped_undefined: int
    conventions: str
    overlay_notional: float
    cost_bps: float
    position: str
    cell: tuple


def _validated_price_frame(frame, name: str, need_open: bool) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name}: expected a non-empty price DataFrame")
    cols = ["Close"] + (["Open"] if need_open else [])
    for column in cols:
        if column not in frame.columns:
            raise ValueError(f"{name}: frame is missing {column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: frame needs a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{name}: index must be sorted and unique")
    values = frame[cols].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{name}: prices must be finite and positive")
    return frame


def _build_overlay_account(core: np.ndarray, o: np.ndarray, c: np.ndarray, trades: list,
                           exposure: int, notional: float, cost_frac: float,
                           skip_entries=frozenset()):
    """Daily account returns for SPY core + overlay, and one record per trade.

    Trade sizing: notional x the account's equity at the close BEFORE the entry
    bar (known at decision time), split into fixed share counts for the life of
    the trade. The entry bar earns open-to-close, every later bar close-to-
    close, the exit bar included (exit at its close). The round-trip cost (a
    fraction of the overlay notional) is charged on the exit bar. Daily account
    return = core + overlay dollars / equity at the previous close; the equity
    compounds daily (overlay P&L is added to equity and held in SPY, the
    convention of the repo's other overlay builders)."""
    n = len(core)
    by_entry = {t["entry_pos"]: t for t in trades if t["entry_pos"] not in skip_entries}
    account = np.empty(n)
    rows = []
    E = 1.0
    active = None
    for t in range(n):
        e_prev = E
        overlay = 0.0
        tr = by_entry.get(t)
        if tr is not None:
            size = e_prev
            shares = notional * size / o[t]
            active = {"tr": tr, "size": size, "shares": shares, "pnl": 0.0}
            step = exposure * shares * (c[t] - o[t])
            overlay += step
            active["pnl"] += step
        elif active is not None:
            step = exposure * active["shares"] * (c[t] - c[t - 1])
            overlay += step
            active["pnl"] += step
        if active is not None and t == active["tr"]["exit_pos"]:
            cost = active["size"] * notional * cost_frac
            overlay -= cost
            active["pnl"] -= cost
            a_tr = active["tr"]
            rows.append({
                "signal_pos": a_tr["signal_pos"], "entry_pos": a_tr["entry_pos"],
                "exit_pos": a_tr["exit_pos"], "reason": a_tr["reason"],
                "entry_px": a_tr["entry_px"], "exit_px": float(c[t]),
                "vehicle_ret": float(c[t]) / a_tr["entry_px"] - 1.0,
                "position_ret": exposure * (float(c[t]) / a_tr["entry_px"] - 1.0),
                "size": active["size"], "cost_ret": cost / active["size"],
                "net_ret": active["pnl"] / active["size"]})
            active = None
        daily = core[t] + overlay / e_prev
        if not np.isfinite(daily) or daily <= -1.0:
            raise ValueError(f"account return {daily} at position {t} is not a valid "
                             "simple return (the overlay wiped the account)")
        account[t] = daily
        E = e_prev * (1.0 + daily)
    return account, rows


def _account_report(account: pd.Series, core: pd.Series, n_trades: int,
                    bar: float) -> AccountReport:
    excess = _ar.log_excess(account, core)
    sharpe = float(_br.sharpe_annualized(excess))
    dd_a, dd_b = float(_br.max_drawdown(account)), float(_br.max_drawdown(core))
    refusals = _ar.risk_refusals(account, core)
    cap = -abs(float(ABSOLUTE_MAX_DRAWDOWN_CAP))
    rel, ab = bool(dd_a >= dd_b), bool(dd_a >= cap)
    return AccountReport(
        n_sessions=int(len(account)), n_trades=int(n_trades),
        excess_sharpe_annualized=sharpe, in_sample_bar=float(bar),
        meets_in_sample_bar=bool(sharpe >= bar),
        dsr_bar_annualized=dsr_bar_annualized(len(account)),
        max_dd_account=dd_a, max_dd_spy=dd_b, absolute_cap=cap,
        dd_relative_passed=rel, dd_absolute_passed=ab, dd_passed=bool(rel and ab),
        risk_refusals=refusals,
        terminal_wealth_account=float(np.prod(1.0 + account.to_numpy(dtype=float))),
        terminal_wealth_spy=float(np.prod(1.0 + core.to_numpy(dtype=float))),
        below_min_sessions=bool(len(account) < MIN_ACCOUNT_SESSIONS))


def abi1_transfer_line(spy_frame: pd.DataFrame, vehicle_frame: pd.DataFrame, signal_dates,
                       cell: tuple, position: str, *,
                       overlay_notional: float = OVERLAY_NOTIONAL, cost_bps: float = 0.0,
                       n_drop: int = N_DROP_WORST,
                       in_sample_bar: float = IN_SAMPLE_BAR_ANNUALIZED_SHARPE,
                       vehicle_series: Optional["sfs.SpikeSeries"] = None,
                       cutoff: pd.Timestamp = sfs.DISCOVERY_CUTOFF) -> TransferLine:
    """The ABI-1 transfer line for ONE pair and cell.

    Builds the account the pipeline would score (SPY core + a `position`
    ('long' / 'short') overlay in the vehicle at `overlay_notional` of equity;
    see the section comment for the engine conventions) from plain price
    frames and reports, for the account and again with the `n_drop` (2)
    largest-loss episodes removed: the annualized daily-excess Sharpe against
    the in-sample bar (about 1.02, `at or above`), the maximum drawdown against
    SPY's and against the absolute cap of 0.35, and the pipeline's own
    risk-refusal block. Nothing is scored, recorded or written.

    `spy_frame` / `vehicle_frame` have `Close` (and, for the vehicle, `Open`)
    on DatetimeIndexes that are sorted, unique and NOT after `cutoff`
    (SealedWindowError otherwise: this line only ever runs on the discovery
    window); the SPY sessions inside the vehicle's span must equal the
    vehicle's calendar exactly. The account lives on the vehicle's calendar
    and its SPY core is the benchmark's own buy-and-hold series over it, so the
    excess is zero whenever no overlay is on. `signal_dates` are signal bars
    (vehicle-calendar dates: the mapped anchors of `VehicleResult.anchor_dates`
    plus any FRED lag already applied); `cell` is (entry, H, TP, SL). The
    take-profit median and stop sigma21 (only for such cells) come from
    `vehicle_series` or, if absent, from one built on the vehicle's closes.

    "Largest-loss episodes" are ranked by each trade's net return on the
    account's equity at its sizing point; removal drops those trades from the
    account without re-admitting signals the engine had skipped (a sensitivity
    on the realised trades, not a re-run of the engine).
    """
    if not 0.0 < float(overlay_notional) <= 1.0:
        raise ValueError("overlay_notional must be in (0, 1]")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if int(n_drop) != n_drop or n_drop < 0:
        raise ValueError("n_drop must be a non-negative integer")
    exposure = -position_sign(position)                      # long = +1 exposure
    sign = position_sign(position)                           # the core side
    veh = _validated_price_frame(vehicle_frame, "vehicle_frame", need_open=True)
    spy = _validated_price_frame(spy_frame, "spy_frame", need_open=False)
    sfs.assert_sealed(veh.index, cutoff)
    sfs.assert_sealed(spy.index, cutoff)
    idx = veh.index
    spy_span = spy.loc[idx[0]:idx[-1]].index
    if not spy_span.equals(idx):
        raise ValueError(
            "spy_frame: the SPY sessions inside the vehicle's span must match the "
            "vehicle calendar exactly; refusing to patch a calendar mismatch")
    sig_dates = pd.DatetimeIndex(signal_dates).normalize()
    sig_pos = idx.get_indexer(sig_dates)
    if (sig_pos < 0).any():
        raise ValueError("every signal date must be a session of the vehicle calendar; "
                         f"not found: {list(sig_dates[sig_pos < 0][:3])}")

    entry_rule, H, tp, sl = cell
    med = sigma = None
    if sfs.TP_FRACTION.get(tp) is not None or sl is not None:
        vs = vehicle_series if vehicle_series is not None else sfs.build_series(
            veh["Close"], name="vehicle", cutoff=cutoff)
        if not vs.dates.equals(idx):
            raise ValueError("vehicle_series must be built on the vehicle frame's calendar")
        ctx = vs.context()
        med = np.where(ctx.med_status == sfs.CTX_OK, ctx.med, np.nan)
        sigma = np.where(ctx.sigma_status == sfs.CTX_OK, ctx.sigma, np.nan)

    o = veh["Open"].to_numpy(dtype=float)
    c = veh["Close"].to_numpy(dtype=float)
    core_series = _br.buy_and_hold_daily_returns(spy).reindex(idx)
    core = core_series.to_numpy(dtype=float)
    if not np.isfinite(core).all():
        raise ValueError("the SPY core return must be finite on every vehicle session")

    eng = engine_convention_trades(o, c, sig_pos, cell, sign, med=med, sigma=sigma)
    cost_frac = float(cost_bps) / 1e4
    full_acct, rows = _build_overlay_account(core, o, c, eng.trades, exposure,
                                             float(overlay_notional), cost_frac)
    cols = ["signal_date", "entry_date", "exit_date", "held", "reason", "entry_px",
            "exit_px", "vehicle_ret", "position_ret", "size", "cost_ret", "net_ret"]
    trades = pd.DataFrame(
        [{"signal_date": idx[r["signal_pos"]], "entry_date": idx[r["entry_pos"]],
          "exit_date": idx[r["exit_pos"]], "held": r["exit_pos"] - r["entry_pos"],
          "reason": r["reason"], "entry_px": r["entry_px"], "exit_px": r["exit_px"],
          "vehicle_ret": r["vehicle_ret"], "position_ret": r["position_ret"],
          "size": r["size"], "cost_ret": r["cost_ret"], "net_ret": r["net_ret"]}
         for r in rows], columns=cols)
    # rank by net return on equity, worst first (stable: earlier entry first on ties)
    order = np.argsort(trades["net_ret"].to_numpy(dtype=float), kind="stable")
    drop_idx = order[:int(n_drop)]
    dropped = trades.iloc[np.sort(drop_idx)].reset_index(drop=True)
    skip = frozenset(int(idx.get_loc(d)) for d in dropped["entry_date"])
    ex_acct, _ = _build_overlay_account(core, o, c, eng.trades, exposure,
                                        float(overlay_notional), cost_frac, skip)

    account = pd.Series(full_acct, index=idx, name="account")
    ex_account = pd.Series(ex_acct, index=idx, name="account_ex_worst")
    core_series = core_series.rename("core")
    full = _account_report(account, core_series, len(trades), in_sample_bar)
    ex_worst = _account_report(ex_account, core_series, len(trades) - len(dropped),
                               in_sample_bar)
    return TransferLine(
        full=full, ex_worst=ex_worst, trades=trades, dropped_trades=dropped,
        account_daily=account, core_daily=core_series,
        excess_daily=_ar.log_excess(account, core_series).rename("excess"),
        account_daily_ex_worst=ex_account, n_signals=eng.n_signals,
        n_skipped_overlap=eng.n_skipped_overlap, n_skipped_no_entry=eng.n_skipped_no_entry,
        n_skipped_undefined=eng.n_skipped_undefined, conventions=_ENGINE_CONVENTIONS,
        overlay_notional=float(overlay_notional), cost_bps=float(cost_bps),
        position=position, cell=tuple(cell))
