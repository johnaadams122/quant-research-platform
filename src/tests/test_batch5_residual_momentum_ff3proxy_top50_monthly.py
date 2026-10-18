"""Tests for src/mining/batch5/residual_momentum_ff3proxy_top50_monthly.py.

Everything runs on SYNTHETIC frames built on the REAL exchange session
calendar; nothing here reads a price file, the network or a ledger (the one
file read is the committed ticker LIST, and the family's own catalog entry).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch5_mining as batch5  # noqa: E402
from mining import cohort_overlay as co  # noqa: E402
from mining.batch5 import residual_momentum_ff3proxy_top50_monthly as mod  # noqa: E402
from mining.batch5 import treasury_month_end_index_demand_overlay as cal  # noqa: E402
from mining.long_overlay import CalendarCoverageError  # noqa: E402

_ENTRY_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == mod.FAMILY_ID)


_SNAPSHOT_PATH = os.path.join(REPO_ROOT, "data", "mining", "universe_snapshot_2026-08-29.json")

FIRST_ME = "2016-04-29"          # synthetic worlds start their month-end table here


# ------------------------------------------------------------------ helpers

def _sessions(start="2016-04-01", end="2021-06-30"):
    return pd.DatetimeIndex(cal.exchange_sessions(pd.Timestamp(start), pd.Timestamp(end)))


def _ohlc(close, index):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999, "Close": close}, index=index)


def _walk(index, seed, vol, drift=0.0, start=100.0):
    r = np.random.RandomState(seed).normal(drift, vol, len(index))
    r[0] = 0.0
    return start * np.cumprod(1.0 + r)


NAMES = [f"T{i:02d}" for i in range(60)]


def _world(seed=1, n=60, start="2016-04-01", end="2021-06-30"):
    idx = _sessions(start, end)
    spy = _ohlc(_walk(idx, seed, 0.009, 0.0003), idx)
    factors = {sym: _ohlc(_walk(idx, seed + 10 + k, 0.011, 0.0002), idx)
               for k, sym in enumerate(mod.FACTOR_SYMBOLS)}
    panel = {f"T{i:02d}": _ohlc(_walk(idx, 1000 + 7 * i + seed, 0.014, 0.0003, 40.0 + i), idx)
             for i in range(n)}
    irx_close = np.linspace(0.5, 2.5, len(idx))
    irx = pd.DataFrame({"Open": irx_close, "High": irx_close, "Low": irx_close, "Close": irx_close},
                       index=idx)
    return idx, spy, factors, panel, irx


def _prepare(world=None, **kw):
    idx, spy, factors, panel, irx = world or _world()
    return mod.prepare(spy, factors, panel, first_month_end=FIRST_ME, **kw)


def _cost(panel, base=2.0):
    return {t: base + 0.1 * i for i, t in enumerate(sorted(panel))}


# ------------------------------------------------- contract and pre-registration

def test_the_module_satisfies_the_batch5_contract():
    spec = batch5.validate_module_contract(mod)
    assert spec["family_id"] == "residual_momentum_ff3proxy_top50_monthly"
    assert spec["shape"] == "pooled_single_asset"
    assert spec["benchmark"] == "SPY"
    assert set(spec["signal"]) == {"IWM", "IVE", "IVW"}
    assert set(spec["traded"]) & set(spec["signal"]) == set()


def test_parameters_equal_the_catalog_entry_byte_for_byte():
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        entry = _catalog_entry_from(fh)
    assert entry["family_id"] == mod.FAMILY_ID
    assert batch5.parameter_drift(mod.PARAMETERS, entry["parameters"]) is None
    assert entry["status"] == "eligible"
    assert entry["account_types"] == ["taxable"]


def test_the_literature_constants_are_the_pre_registered_ones_and_match_parameters():
    assert (mod.REGRESSION_MONTHS, mod.FORMATION_MONTHS, mod.SKIP_MONTHS, mod.N_SELECT) == (36, 11, 1, 50)
    assert mod.OVERLAY_NOTIONAL == 0.5 <= co.MAX_COHORT_OVERLAY_NOTIONAL
    assert mod.PLACEBO_DRAWS >= 500 and mod.PLACEBO_DRAWS >= co.MIN_PLACEBO_DRAWS
    p = mod.PARAMETERS
    assert p["regression_window_months"] == 36 and p["formation_months"] == 11
    assert p["skip_months"] == 1 and p["select_top_n"] == 50
    assert p["overlay_notional_fraction"] == mod.OVERLAY_NOTIONAL
    assert p["placebo_draws"] == mod.PLACEBO_DRAWS and p["placebo_seed"] == mod.PLACEBO_SEED
    assert p["benchmark"] == "SPY"


def test_make_signal_fn_always_raises_so_the_generic_dispatch_cannot_mis_score_it():
    with pytest.raises(RuntimeError, match="bespoke driver"):
        mod.make_signal_fn("AAPL", pd.DatetimeIndex([]), {})


def test_the_traded_list_drops_defective_names_and_share_class_duplicates():
    assert len(mod.TRADED) == len(set(mod.TRADED))
    for ticker in ("HUBB", "MNST"):                                   # documented vendor defects
        assert ticker not in mod.TRADED and ticker in mod.DOCUMENTED_DEFECTS
    for dropped, kept in mod.SHARE_CLASS_DUPLICATES.items():          # one class per issuer
        assert dropped not in mod.TRADED and kept in mod.TRADED
    assert set(mod.SHARE_CLASS_DUPLICATES) == {"GOOGL", "FOX", "NWSA"}
    assert set(mod.SHARE_CLASS_DUPLICATES.values()) == {"GOOG", "FOXA", "NWS"}
    assert "SPY" not in mod.TRADED and set(mod.FACTOR_SYMBOLS).isdisjoint(mod.TRADED)


@pytest.mark.skipif(not os.path.exists(_SNAPSHOT_PATH), reason="universe snapshot not present")
def test_the_traded_list_is_exactly_the_snapshot_minus_the_documented_removals():
    with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
        snapshot = json.load(fh)["tickers"]
    removed = set(mod.DOCUMENTED_DEFECTS) | set(mod.SHARE_CLASS_DUPLICATES)
    assert sorted(mod.TRADED) == sorted(t for t in snapshot if t not in removed)
    assert len(snapshot) == 518 and len(mod.TRADED) == 518 - 5


def test_the_expected_minimum_trades_is_below_the_structural_window_count():
    schedule = mod.month_schedule("2003-05-30", "2026-07-31")
    assert 270 <= mod.EXPECTED_MIN_TRADES <= len(schedule)


# ------------------------------------------------------------------ calendar

def test_the_month_schedule_equals_an_independent_derivation_from_the_exchange_sessions():
    sessions = _sessions("2003-01-02", "2026-12-31")
    by_month = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M"))
    last, first = by_month.max(), by_month.min()
    schedule = mod.month_schedule("2003-05-30", "2026-07-31")
    assert len(schedule) == 279                                       # May 2003 .. Jul 2026
    for decision, entry, exit_ in schedule:
        month = decision.to_period("M")
        assert decision == last[month]                                # last session of month M
        assert entry == first[month + 1] and exit_ == last[month + 1]  # first / last session of M+1
        assert entry == sessions[sessions.get_loc(decision) + 1]       # the very next session
    decisions = [d for d, _, _ in schedule]
    assert decisions == sorted(decisions)
    # month M+1's exit precedes month M+2's entry: complete, synchronized, non-overlapping cohorts
    for (_, _, x1), (_, e2, _) in zip(schedule, schedule[1:]):
        assert x1 < e2


def test_the_schedule_uses_actual_sessions_where_a_closure_was_announced_before_the_decision():
    """Jan 2 2007 was closed (day of mourning announced Dec 27 2006, before the
    Dec 29 decision): the entry is the first ACTUAL session, Jan 3."""
    schedule = {d: (e, x) for d, e, x in mod.month_schedule("2006-11-30", "2007-03-30")}
    entry, _ = schedule[pd.Timestamp("2006-12-29")]
    assert entry == pd.Timestamp("2007-01-03")


def test_the_schedule_is_a_pure_function_of_the_calendar_not_of_any_price_data():
    a = mod.month_schedule("2019-04-30", "2020-12-31")
    b = mod.month_schedule("2019-04-30", "2020-12-31")
    assert a == b and len(a) == 21


def test_a_spy_history_missing_an_exchange_session_is_refused_not_patched():
    idx, spy, factors, panel, irx = _world()
    hole = idx[300]
    with pytest.raises(CalendarCoverageError) as err:
        mod.prepare(spy.drop(hole), {k: v.drop(hole) for k, v in factors.items()}, panel,
                    first_month_end=FIRST_ME)
    assert err.value.reason_code == "missing_exchange_session"
    assert pd.Timestamp(hole).strftime("%Y-%m-%d") in err.value.missing_dates


def test_factor_etfs_that_end_before_spy_are_refused_with_unequal_trailing_coverage():
    idx, spy, factors, panel, irx = _world()
    factors = dict(factors)
    factors["IVE"] = factors["IVE"].iloc[:-1]
    with pytest.raises(CalendarCoverageError) as err:
        mod.prepare(spy, factors, panel, first_month_end=FIRST_ME)
    assert err.value.reason_code == "unequal_trailing_coverage"


def test_a_factor_etf_with_an_interior_hole_is_refused_as_a_calendar_mismatch():
    idx, spy, factors, panel, irx = _world()
    factors = dict(factors)
    factors["IWM"] = factors["IWM"].drop(idx[400])
    with pytest.raises(CalendarCoverageError) as err:
        mod.prepare(spy, factors, panel, first_month_end=FIRST_ME)
    assert err.value.reason_code == "calendar_mismatch"


def test_first_month_end_must_be_the_last_exchange_session_of_its_month():
    idx, spy, factors, panel, irx = _world()
    with pytest.raises(ValueError, match="last exchange session of its month"):
        mod.prepare(spy, factors, panel, first_month_end="2016-04-28")      # a Thursday, not the month-end


def test_the_factor_frames_must_be_exactly_the_registered_factor_etfs():
    idx, spy, factors, panel, irx = _world()
    with pytest.raises(ValueError, match="factor"):
        mod.prepare(spy, {k: v for k, v in factors.items() if k != "IVW"}, panel,
                    first_month_end=FIRST_ME)


# ------------------------------------------------------------- signal arithmetic

def _month_end_close_table(frames, sessions, first_me):
    """Independent month-end close lookup (explicit loop, no shared code)."""
    by_month = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max()
    ends = [d for d in by_month.values if d >= pd.Timestamp(first_me)]
    table = {}
    for name, frame in frames.items():
        table[name] = [float(frame.loc[d, "Close"]) if d in frame.index else np.nan for d in ends]
    return ends, table


def _independent_score(name, world, m):
    """S for one name at month-end row m: numpy lstsq on 36 rows, residual sum
    over rows m-11..m-1 divided by the sample std of those residuals."""
    idx, spy, factors, panel, irx = world
    frames = {"SPY": spy, **factors, name: panel[name]}
    ends, table = _month_end_close_table(frames, idx, FIRST_ME)
    ret = {k: np.array(v[1:]) / np.array(v[:-1]) - 1.0 for k, v in table.items()}      # row r = ends[r+1]
    mkt = ret["SPY"]
    smb = ret["IWM"] - ret["SPY"]
    hml = ret["IVE"] - ret["IVW"]
    rows = list(range(m - 36, m))                                                    # returns of months m-35..m
    X = np.column_stack([np.ones(36), mkt[rows], smb[rows], hml[rows]])
    y = ret[name][rows]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    formation = e[36 - 1 - 11:36 - 1]                                                # months M-11..M-1
    return float(formation.sum() / formation.std(ddof=1))


def test_the_score_is_the_three_factor_residual_sum_over_m_minus_11_to_m_minus_1_over_its_std():
    world = _world()
    prepared = _prepare(world)
    scores = prepared["score_fn"](prepared["windows"][2].decision)
    m = prepared["month_ends"].get_loc(prepared["windows"][2].decision)
    for name in ("T00", "T17", "T41"):
        assert scores[name] == pytest.approx(_independent_score(name, world, m), rel=1e-9)


def test_the_just_ended_month_is_skipped_in_the_formation_sum():
    """A variant that included month M (formation M-10..M) or dropped month M-11
    must differ from the pre-registered score on the same data."""
    world = _world()
    prepared = _prepare(world)
    decision = prepared["windows"][1].decision
    m = prepared["month_ends"].get_loc(decision)
    idx, spy, factors, panel, irx = world
    frames = {"SPY": spy, **factors, "T05": panel["T05"]}
    ends, table = _month_end_close_table(frames, idx, FIRST_ME)
    ret = {k: np.array(v[1:]) / np.array(v[:-1]) - 1.0 for k, v in table.items()}
    rows = list(range(m - 36, m))
    X = np.column_stack([np.ones(36), ret["SPY"][rows], (ret["IWM"] - ret["SPY"])[rows],
                         (ret["IVE"] - ret["IVW"])[rows]])
    e = ret["T05"][rows] - X @ np.linalg.lstsq(X, ret["T05"][rows], rcond=None)[0]
    with_m = e[36 - 11:36]                                   # months M-10..M (no skip)
    assert prepared["score_fn"](decision)["T05"] != pytest.approx(with_m.sum() / with_m.std(ddof=1), rel=1e-6)


def test_the_factor_proxies_are_spy_iwm_minus_spy_and_ive_minus_ivw():
    """A name that is EXACTLY a linear combination of the proxies (plus a constant)
    has zero residuals, so its std is zero and it is not scoreable; a name with
    idiosyncratic noise is. This also proves the three regressors are the
    pre-registered proxies (any other definition leaves a nonzero residual)."""
    world = _world()
    idx, spy, factors, panel, irx = world
    ends, table = _month_end_close_table({"SPY": spy, **factors}, idx, FIRST_ME)
    ret = {k: np.array(v[1:]) / np.array(v[:-1]) - 1.0 for k, v in table.items()}
    mkt, smb, hml = ret["SPY"], ret["IWM"] - ret["SPY"], ret["IVE"] - ret["IVW"]
    exact = 0.002 + 1.1 * mkt + 0.4 * smb - 0.3 * hml            # monthly returns, r[k] = ends[k+1]
    # build a close path whose month-end returns are exactly `exact`; daily path linear in between
    closes = [50.0]
    for r in exact:
        closes.append(closes[-1] * (1.0 + r))
    by_month = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    month_ends = [d for d in by_month.values if d >= pd.Timestamp(FIRST_ME)]
    daily = pd.Series(np.nan, index=idx)
    daily.loc[month_ends] = closes
    daily = daily.interpolate().bfill().ffill()
    panel["EXACT"] = _ohlc(daily.to_numpy(), idx)
    prepared = _prepare((idx, spy, factors, panel, irx))
    decision = prepared["windows"][3].decision
    scores = prepared["score_fn"](decision)
    assert not np.isfinite(scores["EXACT"])
    assert np.isfinite(scores["T00"])
    assert "EXACT" not in prepared["windows"][3].eligible and "T00" in prepared["windows"][3].eligible


def _monthly_tables(n_names=40, seed=3):
    """Synthetic month-by-name returns and a month-by-3 factor table with 48 rows
    (decision row 47): a name's returns are its factor loadings times the factors
    plus noise, so the regression residual is the noise and can be planted."""
    rng = np.random.RandomState(seed)
    rows = 48
    factors = pd.DataFrame(rng.normal(0.005, 0.03, (rows, 3)), columns=["MKT", "SMB", "HML"])
    names = [f"S{i:02d}" for i in range(n_names)]
    loadings = rng.normal(1.0, 0.3, (n_names, 3))
    noise = rng.normal(0.0, 0.01, (rows, n_names))
    returns = pd.DataFrame(factors.to_numpy() @ loadings.T + noise, columns=names)
    return returns, factors, names


def test_a_recent_positive_residual_ranks_first_and_a_recent_negative_one_last():
    returns, factors, names = _monthly_tables()
    row = 47
    returns = returns.copy()
    returns["WIN"] = returns["S00"]
    returns["LOSE"] = returns["S00"]
    returns.loc[row - 11:row - 1, "WIN"] += 0.02        # formation months M-11..M-1
    returns.loc[row - 11:row - 1, "LOSE"] -= 0.02
    scores = mod.scores_at(returns, factors, row).dropna().sort_values(ascending=False)
    assert scores.index[0] == "WIN" and scores.index[-1] == "LOSE"
    assert scores["WIN"] > scores.drop(["WIN"]).max()


def test_a_constant_alpha_is_absorbed_by_the_intercept_and_does_not_score():
    """The regression has an intercept, so a persistent drift is not a
    residual: the score rewards RECENT deviations from the 36-month fit."""
    returns, factors, names = _monthly_tables()
    row = 47
    returns = returns.copy()
    returns["DRIFT"] = returns["S00"] + 0.02            # +2 percent EVERY month, all 36 months
    returns["RECENT"] = returns["S00"]
    returns.loc[row - 11:row - 1, "RECENT"] += 0.02
    scores = mod.scores_at(returns, factors, row)
    assert abs(scores["DRIFT"] - scores["S00"]) < 1e-6
    assert scores["RECENT"] > scores["S00"] + 1.0


def test_scores_at_uses_only_the_36_rows_ending_at_the_decision_row():
    returns, factors, names = _monthly_tables()
    row = 40
    base = mod.scores_at(returns, factors, row)
    future = returns.copy()
    future.iloc[row + 1:] = future.iloc[row + 1:] * 9.0
    early = returns.copy()
    early.iloc[:row - 35] = early.iloc[:row - 35] * 9.0        # rows before the window
    fut_factors = factors.copy()
    fut_factors.iloc[row + 1:] = 5.0
    pd.testing.assert_series_equal(mod.scores_at(future, fut_factors, row), base)
    pd.testing.assert_series_equal(mod.scores_at(early, factors, row), base)


def test_collinear_factor_proxies_are_refused_not_silently_solved():
    returns, factors, names = _monthly_tables()
    factors = factors.copy()
    factors["SMB"] = 2.0 * factors["MKT"]
    with pytest.raises(ValueError, match="collinear"):
        mod.scores_at(returns, factors, 47)


def test_scores_at_returns_nan_without_a_full_history_and_for_a_zero_variance_residual():
    returns, factors, names = _monthly_tables()
    assert mod.scores_at(returns, factors, 34).isna().all()      # fewer than 36 returns
    returns = returns.copy()
    returns.loc[20, "S01"] = np.nan                                # a hole inside the window (rows 12..47)
    scores = mod.scores_at(returns, factors, 47)
    assert np.isnan(scores["S01"]) and np.isfinite(scores["S02"])


def test_a_missing_month_end_bar_is_never_forward_filled_it_makes_the_name_ineligible():
    world = _world()
    idx, spy, factors, panel, irx = world
    by_month = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    month_ends = [d for d in by_month.values if d >= pd.Timestamp(FIRST_ME)]
    hole = month_ends[2]                                   # row 2 = 2016-06-30
    panel = dict(panel)
    panel["T07"] = panel["T07"].drop(hole)
    prepared = _prepare((idx, spy, factors, panel, irx))
    returns = prepared["returns"]["T07"]
    # the missing close makes exactly the return INTO it (row 2) and OUT of it (row 3) undefined
    assert returns.iloc[[2, 3]].isna().all() and returns.drop(returns.index[[0, 2, 3]]).notna().all()
    for w in prepared["windows"]:
        row = prepared["month_ends"].get_loc(w.decision)
        # the 36-return window of decision row r covers returns r-35..r: the hole is inside it
        # until r-35 > 3, i.e. from row 39 on the name is eligible again
        assert ("T07" in w.eligible) == (row >= 39), (w.decision, row)
    assert any("T07" not in w.eligible for w in prepared["windows"])
    assert any("T07" in w.eligible for w in prepared["windows"])


def test_a_name_needs_a_full_36_monthly_returns_to_be_eligible():
    world = _world()
    idx, spy, factors, panel, irx = world
    late = idx[idx > pd.Timestamp("2016-08-31")][0]                # listing a few months late
    panel = dict(panel)
    panel["LATE"] = _ohlc(_walk(idx[idx >= late], 5, 0.012, 0.0, 30.0), idx[idx >= late])
    prepared = _prepare((idx, spy, factors, panel, irx))
    # LATE's first month-end is 2016-09-30 => its first full 36 returns end 3 years+ later
    first_ok = min((w for w in prepared["windows"] if "LATE" in w.eligible), key=lambda w: w.decision)
    assert first_ok.decision >= pd.Timestamp("2019-09-30")
    earlier = [w for w in prepared["windows"] if w.decision < first_ok.decision]
    assert earlier and all("LATE" not in w.eligible for w in earlier)


def test_the_score_never_reads_a_bar_after_the_decision_session():
    world = _world()
    idx, spy, factors, panel, irx = world
    base = _prepare(world)
    w = base["windows"][6]
    cut = w.decision
    bumped_panel = {}
    for name, frame in panel.items():
        f = frame.copy()
        f.loc[f.index > cut] = f.loc[f.index > cut] * 1.9
        bumped_panel[name] = f
    bumped_factors = {}
    for name, frame in factors.items():
        f = frame.copy()
        f.loc[f.index > cut] = f.loc[f.index > cut] * 0.5
        bumped_factors[name] = f
    bumped_spy = spy.copy()
    bumped_spy.loc[bumped_spy.index > cut] = bumped_spy.loc[bumped_spy.index > cut] * 0.7
    bumped = mod.prepare(bumped_spy, bumped_factors, bumped_panel, first_month_end=FIRST_ME)
    pd.testing.assert_series_equal(bumped["score_fn"](cut), base["score_fn"](cut))
    same = [x for x in bumped["windows"] if x.decision <= cut]
    assert [(x.selected, x.eligible) for x in same] == [(x.selected, x.eligible)
                                                        for x in base["windows"] if x.decision <= cut]


def test_ties_are_broken_by_ticker_at_the_family_level():
    world = _world()
    idx, spy, factors, panel, irx = world
    panel = dict(panel)
    panel["ZZZ_CLONE"] = panel["T20"].copy()                # identical returns => identical score
    first = _prepare((idx, spy, factors, panel, irx))
    decision = first["windows"][0].decision
    scores = first["score_fn"](decision).dropna()
    # identical returns => the same score up to floating-point noise: np.linalg.lstsq is not column-position
    # invariant and, depending on the BLAS thread count (a single BLAS thread, for example), the clone's score
    # can differ from T20's in the last bit, so the precondition is numerical closeness, not bit equality
    assert np.isclose(scores["T20"], scores["ZZZ_CLONE"], rtol=1e-12, atol=0.0)
    # the tie rule applied to scores at twelve significant digits, exactly as the oracle below and the
    # production selection treat last-bit noise (a raw-float ordering would let noise decide the ranking)
    order = sorted(scores.index, key=lambda t: (-float(f"{scores[t]:.11e}"), t))
    k = order.index("T20") + 1                              # T20 is the k-th name, its clone the (k+1)-th
    assert order[k] == "ZZZ_CLONE"
    cut = mod.prepare(spy, factors, panel, first_month_end=FIRST_ME, n_select=k)
    w = cut["windows"][0]
    assert "T20" in w.selected and "ZZZ_CLONE" not in w.selected      # the smaller ticker takes the last slot
    for window in cut["windows"]:
        # oracle: the tie rule applied to scores at twelve significant digits (the same input, an independent
        # rule). At the LATER windows the clone's raw score can sit a last bit above T20's (lstsq is not column-
        # position invariant), which a raw-float oracle would misread as the clone outranking T20.
        scores = cut["score_fn"](window.decision).dropna()
        assert list(window.selected) == sorted(
            scores.index, key=lambda t: (-float(f"{scores[t]:.11e}"), t))[:k]


def _clone_world_with_a_last_bit_score_difference():
    """A REAL world (the module's own scores, no stubbing) holding a name and a byte-identical clone whose
    UNROUNDED scores at the first decision differ in the last bit, because ``np.linalg.lstsq`` is not column-
    position invariant. Returns ``(panel, source, lo, hi)`` where ``lo`` and ``hi`` are the two twins named so
    that the LARGER ticker ``hi`` holds the strictly higher raw score (the arrangement in which ranking by raw
    float, not by tie class, would let it displace the smaller ticker), or None when this BLAS shows no such
    difference for any name."""
    idx, spy, factors, panel, irx = _world()
    for source in sorted(panel):
        for clone_name in ("ZZZ_CLONE", "AAA_CLONE"):
            probe = dict(panel)
            probe[clone_name] = probe[source].copy()             # appended last: same matrix position either way
            prepared = _prepare((idx, spy, factors, probe, irx))
            scores = prepared["score_fn"](prepared["windows"][0].decision).dropna()
            a, b = scores[source], scores[clone_name]
            if a == b:
                break                                            # this source shows no difference at all
            higher = source if a > b else clone_name
            if higher == max(source, clone_name):                # the larger ticker holds the higher raw score
                return probe, source, min(source, clone_name), max(source, clone_name)
    return None


def test_a_last_bit_score_difference_between_identical_histories_cannot_beat_the_ticker_tie_break():
    """Two byte-identical histories are one tie by the documented rule (ties by ticker
    ascending), but their unrounded scores can differ in the last bit; the larger ticker must still not take
    the last slot. The premise (a real, unrounded last-bit difference with the LARGER ticker higher) is proven
    inside the test on the module's own scores. A BLAS that gives identical histories scores that never
    differ that way cannot build the premise; the test then checks the same tie-break on a plain clone pair
    (never a skip, so it runs on every platform; the nextafter-based unit tests in
    test_mining_cohort_overlay pin the tolerance rule itself)."""
    idx, spy, factors, _panel, irx = _world()
    found = _clone_world_with_a_last_bit_score_difference()
    if found is None:
        source = sorted(_panel)[0]
        panel = dict(_panel)
        panel["ZZZ_CLONE"] = panel[source].copy()             # appended last, as in the probe
        lo, hi = sorted((source, "ZZZ_CLONE"))
    else:
        panel, source, lo, hi = found
    first = _prepare((idx, spy, factors, panel, irx))
    scores = first["score_fn"](first["windows"][0].decision).dropna()
    if found is not None:
        assert scores[lo] != scores[hi] and scores[hi] > scores[lo]    # premise: unrounded, hi strictly higher
    assert abs(scores[hi] - scores[lo]) <= 1e-12 * max(1.0, abs(scores[hi]))    # ... by rounding noise only
    others = scores.drop([lo, hi])
    assert (abs(others - scores[hi]) > 1e-9 * max(1.0, abs(scores[hi]))).all()  # no third name near the pair
    k = int((others > scores[hi]).sum()) + 1                # the pair straddles the cut: one slot, two twins
    cut = mod.prepare(spy, factors, panel, first_month_end=FIRST_ME, n_select=k)
    chosen = set(cut["windows"][0].selected)
    assert lo in chosen and hi not in chosen                # the smaller ticker takes the last slot
    for window in cut["windows"]:                           # and at EVERY window a lone larger twin never wins
        picked = set(window.selected)
        assert not (hi in picked and lo not in picked), window.decision


# ---------------------------------------------------------------------- windows

def test_windows_start_at_the_first_decision_with_36_returns_and_each_holds_the_next_month():
    prepared = _prepare()
    first = prepared["windows"][0]
    assert first.decision == pd.Timestamp("2019-04-30")           # 36 monthly returns from 2016-05
    assert first.entry == pd.Timestamp("2019-05-01") and first.exit == pd.Timestamp("2019-05-31")
    for w in prepared["windows"]:
        assert w.entry > w.decision and w.exit > w.entry
        assert w.entry.month == (w.decision + pd.Timedelta(days=5)).month


def test_a_still_open_final_window_is_excluded_and_reported_never_closed_early():
    idx, spy, factors, panel, irx = _world(end="2021-06-15")
    prepared = mod.prepare(spy, factors, panel, first_month_end=FIRST_ME)
    assert prepared["windows"][-1].exit == pd.Timestamp("2021-05-28")
    assert prepared["open_window"]["entry"] == pd.Timestamp("2021-06-01")
    assert prepared["open_window"]["scheduled_exit"] == pd.Timestamp("2021-06-30")
    assert prepared["last_exit"] == pd.Timestamp("2021-05-28")


def test_a_data_end_exactly_on_a_month_end_completes_that_window():
    idx, spy, factors, panel, irx = _world(end="2021-06-30")
    prepared = mod.prepare(spy, factors, panel, first_month_end=FIRST_ME)
    assert prepared["windows"][-1].exit == pd.Timestamp("2021-06-30")
    assert prepared["open_window"] is None


def test_months_with_fewer_than_50_eligible_names_are_skipped_and_reported():
    world = _world(n=49)
    prepared = _prepare(world)
    assert prepared["windows"] == []
    assert prepared["skipped"] and all(n == 49 for _, n in prepared["skipped"])


# ---------------------------------------------------------------------- hygiene

def _hyg_world():
    idx, spy, factors, panel, irx = _world()
    return idx, spy, panel


def test_hygiene_keeps_a_clean_history_and_reports_nothing():
    idx, spy, panel = _hyg_world()
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert set(result["kept"]) == set(panel) and result["excluded"] == {}


def test_hygiene_excludes_the_documented_defects_even_with_clean_data():
    idx, spy, panel = _hyg_world()
    panel = dict(panel)
    panel["HUBB"] = panel["T00"].copy()
    panel["MNST"] = panel["T01"].copy()
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert "HUBB" not in result["kept"] and "MNST" not in result["kept"]
    assert result["excluded"]["HUBB"]["reason"] == "documented_defect"
    assert result["excluded"]["MNST"]["reason"] == "documented_defect"


def test_hygiene_excludes_an_interior_gap_after_the_cutoff_and_names_its_dates():
    idx, spy, panel = _hyg_world()
    panel = dict(panel)
    panel["T03"] = panel["T03"].drop([idx[200], idx[201]])
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert "T03" not in result["kept"]
    info = result["excluded"]["T03"]
    assert info["reason"] == "interior_gap" and info["n_missing"] == 2
    assert idx[200].strftime("%Y-%m-%d") in info["dates"]


def test_hygiene_ignores_a_gap_before_the_cutoff():
    idx, spy, panel = _hyg_world()
    panel = dict(panel)
    panel["T04"] = panel["T04"].drop([idx[3]])               # before the first needed month-end
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert "T04" in result["kept"]


def test_hygiene_excludes_a_bar_on_a_closed_day():
    idx, spy, panel = _hyg_world()
    panel = dict(panel)
    stray = pd.Timestamp("2017-07-04")                       # Independence Day, exchange closed
    assert stray not in idx
    frame = panel["T05"].copy()
    frame.loc[stray] = frame.iloc[100]
    panel["T05"] = frame.sort_index()
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert "T05" not in result["kept"] and result["excluded"]["T05"]["reason"] == "bar_on_closed_day"


def test_hygiene_excludes_a_history_that_ends_before_the_last_session():
    idx, spy, panel = _hyg_world()
    panel = dict(panel)
    panel["T06"] = panel["T06"].iloc[:-3]
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert "T06" not in result["kept"] and result["excluded"]["T06"]["reason"] == "stale_history"


def test_hygiene_allows_a_late_listing_with_a_clean_history_after_it():
    idx, spy, panel = _hyg_world()
    panel = dict(panel)
    late = idx[idx > pd.Timestamp("2018-01-01")]
    panel["NEW"] = _ohlc(_walk(late, 4, 0.01, 0.0, 20.0), late)
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert "NEW" in result["kept"]


def test_hygiene_returns_frames_untouched_and_never_patches_them():
    idx, spy, panel = _hyg_world()
    panel = dict(panel)
    hole = panel["T08"].drop([idx[250]])
    panel["T08"] = hole
    result = mod.panel_hygiene(panel, calendar_index=idx, first_needed="2016-04-29")
    assert "T08" not in result["kept"]
    assert len(panel["T08"]) == len(idx) - 1                 # the caller's frame is not repaired


# ----------------------------------------------------------------------- curve

def test_the_curve_is_a_benchmark_plus_overlay_account_with_one_window_per_cohort():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    curve = mod.build_curve(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=1.0)
    assert curve["account_kind"] == "benchmark_plus_overlay"
    assert len(curve["trades_df"]) == len(prepared["windows"]) == 26
    assert (curve["trades_df"]["n_selected"] == 50).all() and (curve["trades_df"]["n_eligible"] == 60).all()
    assert curve["strategy_daily"].index[0] == prepared["windows"][0].decision
    assert curve["strategy_daily"].index[-1] == prepared["windows"][-1].exit
    assert curve["window_info"]["n_windows"] == 26


def test_the_overlay_notional_is_the_pinned_parameter_not_a_caller_choice():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    curve = mod.build_curve(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    first = curve["trades_df"].iloc[0]
    w = prepared["windows"][0]
    sessions = idx[(idx >= w.entry) & (idx <= w.exit)]
    rate = curve["financing_rate"]
    expected = sum(0.5 * rate.loc[t] * (1.0 if k == 0 else (sessions[k] - sessions[k - 1]).days) / 365.0
                   for k, t in enumerate(sessions))
    assert first["financing_ret"] == pytest.approx(expected, rel=1e-9)


def test_the_financing_charged_scales_with_the_bill_series():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    low = mod.build_curve(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    hi_irx = irx.copy()
    hi_irx["Close"] = hi_irx["Close"] * 2.0
    high = mod.build_curve(prepared, panel, hi_irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    ratio = high["trades_df"]["financing_ret"] / low["trades_df"]["financing_ret"]
    assert (ratio > 1.9).all()


def test_a_selected_name_without_the_entry_open_fails_closed_through_the_family():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    w = prepared["windows"][4]
    victim = w.selected[0]
    panel = dict(panel)
    frame = panel[victim].copy()
    frame.loc[w.entry, "Open"] = np.nan
    panel[victim] = frame
    with pytest.raises(co.CohortDataError) as err:
        mod.build_curve(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    assert err.value.reason_code == "missing_member_price" and victim in err.value.tickers


def test_the_control_curve_uses_the_same_windows_and_ignores_the_cohort():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    control = mod.build_control_curve(prepared, panel, cost_bps_by_symbol=_cost(panel))
    assert "account_kind" not in control
    assert len(control["trades_df"]) == len(prepared["windows"])
    assert control["strategy_daily"].index[0] == prepared["windows"][0].decision
    assert (control["trades_df"]["n_selected"] == 60).all()          # the whole eligible basket


def test_the_placebo_defaults_are_the_pre_registered_draw_count_and_seed(monkeypatch):
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    seen = {}
    real = co.permutation_placebo

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **{**kwargs, "n_draws": co.MIN_PLACEBO_DRAWS})

    monkeypatch.setattr(co, "permutation_placebo", fake)
    out = mod.run_placebo(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    assert seen["n_draws"] == mod.PLACEBO_DRAWS == 500 and seen["seed"] == mod.PLACEBO_SEED
    assert seen["overlay_notional"] == mod.OVERLAY_NOTIONAL
    assert out["n_draws"] == co.MIN_PLACEBO_DRAWS and 0.0 <= out["real_percentile"] <= 1.0


def test_the_abi1_evaluator_accepts_the_family_curve_on_the_overlay_only_basis():
    from mining import abi1_driver
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    curve = mod.build_curve(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=1.0)
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy, benchmark_entry_cost_bps=1.0, account_type="taxable",
        tax_policy="realization")
    assert result["layer2"]["account_basis"] == "overlay_only_excess"


def test_the_daily_control_uses_the_same_windows_and_is_fully_invested_inside_them():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    daily = mod.build_daily_control_curve(prepared, panel)
    assert "account_kind" not in daily
    assert len(daily["trades_df"]) == len(prepared["windows"]) == 26
    assert daily["strategy_daily"].index[0] == prepared["windows"][0].decision
    assert daily["strategy_daily"].index[-1] == prepared["windows"][-1].exit
    assert (daily["trades_df"]["n_eligible"] == 60).all()
    assert daily["in_market"].sum() == sum(idx.get_loc(w.exit) - idx.get_loc(w.entry) + 1
                                           for w in prepared["windows"])
    with pytest.raises(ValueError, match="no complete window"):
        mod.build_daily_control_curve(_prepare(_world(n=49)), panel)


# ================================================================ look-ahead and disclosure guards
# Each test below pins one look-ahead or disclosure property.

def _bump_bills_from(irx, session, factor=3.0):
    bumped = irx.copy()
    later = bumped.index >= session
    bumped.loc[later, "Close"] = bumped.loc[later, "Close"] * factor
    return bumped


def test_month_ends_come_from_the_exchange_calendar_not_from_the_datas_own_last_row_per_month():
    """A history that stops mid-month has a last ROW for that month, but that row
    is not a month-end: the month-end table is the exchange calendar's last
    session of each month, cut at the last bar (an implementation that read it off the data would fail)."""
    world = _world(end="2021-06-15")
    idx = world[0]
    assert idx[-1] == pd.Timestamp("2021-06-15")
    prepared = _prepare(world)
    calendar = _sessions("2016-04-01", "2021-07-31")              # the FULL exchange calendar
    by_month = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max()
    expected = [d for d in by_month.values if pd.Timestamp(FIRST_ME) <= d <= idx[-1]]
    assert list(prepared["month_ends"]) == expected
    assert prepared["month_ends"][-1] == pd.Timestamp("2021-05-28")
    assert idx[-1] not in prepared["month_ends"]                   # June's last DATA row is not a month-end
    assert len(prepared["returns"]) == len(prepared["factors"]) == len(expected)


def test_a_month_end_session_missing_from_every_market_frame_is_refused_not_replaced_by_the_day_before():
    """The last calendar session of June 2018 is absent from SPY and all three
    factor ETFs: the family refuses (missing_exchange_session, naming the date)
    instead of treating the previous row as the month-end."""
    idx, spy, factors, panel, irx = _world()
    month_end = pd.Timestamp("2018-06-29")
    assert idx[idx.to_period("M") == pd.Period("2018-06")][-1] == month_end
    with pytest.raises(CalendarCoverageError) as err:
        mod.prepare(spy.drop(month_end), {k: v.drop(month_end) for k, v in factors.items()}, panel,
                    first_month_end=FIRST_ME)
    assert err.value.reason_code == "missing_exchange_session"
    assert "2018-06-29" in err.value.missing_dates


def test_the_sample_start_is_pinned_the_first_month_end_and_the_first_decision():
    """FIRST_MONTH_END fixes the first cohort (June 2003), the sample start the
    catalog text states; it is not in PARAMETERS, so it is pinned here."""
    import inspect
    assert mod.FIRST_MONTH_END == "2000-05-31"
    ends = mod.month_end_sessions(mod.FIRST_MONTH_END, "2004-01-30")
    assert ends[0] == pd.Timestamp("2000-05-31")
    first_decision = ends[mod.REGRESSION_MONTHS]                     # the 37th month-end: 36 returns exist
    assert first_decision == pd.Timestamp("2003-05-30")
    assert mod.month_schedule(first_decision, ends[-1])[0] == (
        pd.Timestamp("2003-05-30"), pd.Timestamp("2003-06-02"), pd.Timestamp("2003-06-30"))
    assert inspect.signature(mod.prepare).parameters["first_month_end"].default == mod.FIRST_MONTH_END
    assert inspect.signature(mod.panel_hygiene).parameters["first_needed"].default == mod.FIRST_MONTH_END


# --- financing is strictly causal in the family's own call sites

def _financed_pair(prepared, idx):
    """A session inside a window (not its first) and the session after it."""
    w = prepared["windows"][4]
    sessions = idx[(idx >= w.entry) & (idx <= w.exit)]
    return sessions[5], sessions[6]


def test_the_scored_account_finances_at_the_last_print_strictly_before_each_session():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    s, nxt = _financed_pair(prepared, idx)
    kw = dict(cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    base = mod.build_curve(prepared, panel, irx, **kw)
    bumped = mod.build_curve(prepared, panel, _bump_bills_from(irx, s), **kw)
    # nothing up to and including session s moves: the print OF session s is first read at s+1
    pd.testing.assert_series_equal(bumped["financing_rate"].loc[:s], base["financing_rate"].loc[:s],
                                   check_exact=True)
    pd.testing.assert_series_equal(bumped["strategy_daily"].loc[:s], base["strategy_daily"].loc[:s],
                                   check_exact=True)
    assert bumped["financing_rate"].loc[nxt] > base["financing_rate"].loc[nxt]
    assert bumped["strategy_daily"].loc[nxt] != base["strategy_daily"].loc[nxt]


def test_the_placebo_finances_at_the_last_print_strictly_before_each_session(monkeypatch):
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    s, nxt = _financed_pair(prepared, idx)
    seen = []

    def fake(*args, **kwargs):
        seen.append(kwargs["financing_rate_annual"])
        return {}

    monkeypatch.setattr(co, "permutation_placebo", fake)
    kw = dict(cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    mod.run_placebo(prepared, panel, irx, **kw)
    mod.run_placebo(prepared, panel, _bump_bills_from(irx, s), **kw)
    base, bumped = seen
    pd.testing.assert_series_equal(bumped.loc[:s], base.loc[:s], check_exact=True)
    assert bumped.loc[nxt] > base.loc[nxt]


# --- the bill print used on any financed session must be fresh

def _gap_pair(sessions, days):
    """Positions (a, b) inside a window with sessions[b] - sessions[a] exactly ``days``
    calendar days and at least one session between them."""
    for a in range(len(sessions)):
        for b in range(a + 2, len(sessions)):
            if (sessions[b] - sessions[a]).days == days:
                return a, b
    raise AssertionError(f"no {days}-day session pair in the window")


def _window_sessions(prepared, idx, k=6):
    w = prepared["windows"][k]
    return idx[(idx >= w.entry) & (idx <= w.exit)]


def test_the_curve_reports_the_oldest_bill_print_used_on_any_financed_session():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    curve = mod.build_curve(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    financed = []
    for w in prepared["windows"]:
        financed += list(idx[(idx >= w.entry) & (idx <= w.exit)])
    expected = max((s - idx[idx.get_loc(s) - 1]).days for s in financed)       # a print on every session
    assert curve["window_info"]["bill_print_max_age_days"] == expected
    assert 3 <= expected <= mod.MAX_BILL_PRINT_AGE_DAYS


def test_a_bill_print_ten_calendar_days_old_is_accepted_and_eleven_is_refused():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    sessions = _window_sessions(prepared, idx)
    assert mod.MAX_BILL_PRINT_AGE_DAYS == 10
    kw = dict(cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    a, b = _gap_pair(sessions, 10)
    ok = mod.build_curve(prepared, panel, irx.drop(sessions[a + 1:b]), **kw)
    assert ok["window_info"]["bill_print_max_age_days"] == 10
    a, b = _gap_pair(sessions, 11)
    with pytest.raises(mod.BillPrintStaleError) as err:
        mod.build_curve(prepared, panel, irx.drop(sessions[a + 1:b]), **kw)
    assert err.value.reason_code == "stale_bill_print"
    assert err.value.max_age_days == 11 and err.value.session == sessions[b].strftime("%Y-%m-%d")
    assert "10" in str(err.value) and isinstance(err.value, ValueError)


def test_the_placebo_refuses_a_stale_bill_print_too():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    sessions = _window_sessions(prepared, idx)
    a, b = _gap_pair(sessions, 11)
    with pytest.raises(mod.BillPrintStaleError):
        mod.run_placebo(prepared, panel, irx.drop(sessions[a + 1:b]), cost_bps_by_symbol=_cost(panel),
                        spy_entry_cost_bps=0.0, n_draws=co.MIN_PLACEBO_DRAWS)


def test_a_bill_series_that_starts_after_a_financed_session_is_refused_as_having_no_print():
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    late = irx.loc[irx.index > prepared["windows"][0].entry]
    with pytest.raises(mod.BillPrintStaleError) as err:
        mod.build_curve(prepared, panel, late, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    assert err.value.reason_code == "stale_bill_print" and err.value.max_age_days is None
    assert "no bill print" in str(err.value)


def test_a_bill_series_hole_on_sessions_that_are_never_financed_is_not_refused():
    """Only the sessions of the holding windows are financed: a hole before the
    first decision is irrelevant to the account."""
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    holed = irx.drop(idx[:60])                          # 2016-04 .. mid 2016: long before the first decision
    curve = mod.build_curve(prepared, panel, holed, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    assert curve["window_info"]["bill_print_max_age_days"] <= mod.MAX_BILL_PRINT_AGE_DAYS


# --- the pre-registration text states its limits


def _surfaces():
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        entry = _catalog_entry_from(fh)
    surfaces = {"family module docstring": mod.__doc__, "cohort_overlay docstring": co.__doc__,
                "catalog mechanism": entry["mechanism"]}
    return surfaces


def test_the_family_docstring_and_the_catalog_text_name_the_design_choices():
    """The family docstring and the catalog text name which constants are
    design choices rather than literature values (the survivorship limit
    every surface states is checked by the next test)."""
    for where, text in _surfaces().items():
        if where in ("family module docstring", "catalog mechanism"):
            lowered = " ".join(text.lower().split())
            assert "design choices" in lowered, where


def test_every_surface_states_the_survivorship_limit_the_overlay_cannot_remove():
    """The overlay cancels the LEVEL of the survivorship baseline only; a rank-
    conditional survivorship effect (past winners that later failed or joined the
    index later) is not cancelled and the placebo shares it."""
    for where, text in _surfaces().items():
        lowered = " ".join(text.lower().split())
        assert "rank-conditional survivorship" in lowered, where
        assert "level" in lowered, where
        if where in ("family module docstring", "catalog mechanism"):
            assert "not cancelled" in lowered, where
            assert "inconclusive" in lowered, where


def test_the_hygiene_and_spread_conventions_are_disclosed_where_the_catalog_and_the_docstring_say_they_are():
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        mechanism = " ".join(_catalog_entry_from(fh)["mechanism"].lower().split())
    doc = " ".join(mod.__doc__.lower().split())
    for text in (mechanism, doc):
        assert "2026 quoted spreads" in text and "direction unmeasured" in text
        assert "full-sample" in text or "full sample" in text            # hygiene decided on all-history data
    overlay_doc = " ".join(co.__doc__.lower().split())
    assert "first three flatter" in overlay_doc and "the tax identity" in overlay_doc


# --- the pre-registered pass rule and the tie tolerance are stated where they apply

def test_the_catalog_entry_states_the_pre_registered_pass_rule_and_the_tie_tolerance():
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        mechanism = " ".join(_catalog_entry_from(fh)["mechanism"].split())
    assert "PRE-REGISTERED PASS RULE" in mechanism
    assert "a passing stage counts as a pass of this family only when the real rule's overlay Sharpe percentile" in mechanism
    assert "at least 0.95" in mechanism and "CONTROL CONDITION FAILED" in mechanism
    assert "pre_registered_pass is false" in mechanism
    assert "ties by ticker ascending (scores within a relative 1e-12 of each other count as one tie" in mechanism


def test_the_tie_tolerance_is_stated_in_the_family_and_generic_docstrings_and_matches_the_constant():
    family_doc = " ".join(mod.__doc__.split())
    assert "scores within a relative 1e-12 are one tie" in family_doc
    overlay_doc = " ".join(co.__doc__.split())
    assert "RANK_REL_TOL" in overlay_doc and "1e-12" in overlay_doc
    assert "require_account_is_benchmark_plus_overlay" in overlay_doc and "EVERY session" in overlay_doc
    assert co.RANK_REL_TOL == 1e-12


def test_the_tie_break_parameter_text_is_unchanged_by_the_tolerance_so_the_pre_registration_holds():
    assert mod.PARAMETERS["tie_break"] == "ticker ascending"
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        assert _catalog_entry_from(fh)["parameters"]["tie_break"] == "ticker ascending"


# ================================================================ boundary guards
# Each test below pins one boundary behaviour of the controls, the window report or the bill-print check.

def test_the_zero_signal_monthly_control_pays_the_cost_of_every_eligible_name_on_the_exit_day():
    """A control built with zero costs would still have the right windows, sizes and price path, and the
    only thing that gives it away is a cost line of zero. The control is a whole account holding the
    eligible basket for one round trip a month, so each window pays the eligible names' mean round trip."""
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    costs = _cost(panel)
    control = mod.build_control_curve(prepared, panel, cost_bps_by_symbol=costs)
    trades = control["trades_df"]
    assert len(trades) == len(prepared["windows"]) and (trades["cost_ret"] > 0).all()
    expected = [np.mean([costs[t] for t in w.eligible]) / 1e4 for w in prepared["windows"]]
    np.testing.assert_allclose(trades["cost_ret"].to_numpy(dtype=float), expected, rtol=1e-12)
    zero = mod.build_control_curve(prepared, panel, cost_bps_by_symbol={t: 0.0 for t in panel})
    assert (zero["trades_df"]["cost_ret"] == 0.0).all()
    assert (trades["net_ret"].to_numpy() < zero["trades_df"]["net_ret"].to_numpy()).all()


def test_a_data_end_exactly_on_a_month_end_reports_the_next_decision_as_not_started():
    """``not_started`` is the report of a decision whose entry lies after the data; the data here
    ends on a month-end session, so the next month's window has not started (it is not open and not scored)."""
    idx, spy, factors, panel, irx = _world(end="2021-06-30")
    prepared = mod.prepare(spy, factors, panel, first_month_end=FIRST_ME)
    assert prepared["open_window"] is None
    assert prepared["not_started"] == {"decision": pd.Timestamp("2021-06-30"), "entry": pd.Timestamp("2021-07-01"),
                                       "scheduled_exit": pd.Timestamp("2021-07-30")}
    curve = mod.build_curve(prepared, panel, irx, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    assert curve["window_info"]["not_started"] == prepared["not_started"]


def test_a_data_end_in_the_middle_of_a_window_reports_it_open_and_nothing_not_started():
    idx, spy, factors, panel, irx = _world(end="2021-06-15")
    prepared = mod.prepare(spy, factors, panel, first_month_end=FIRST_ME)
    assert prepared["open_window"] is not None and prepared["not_started"] is None


def _window_boundary_eleven_days_after_a_session(prepared, idx, which):
    """A window whose entry (``which == "entry"``) or exit (``"exit"``) session sits exactly 11 calendar days
    after another session; returns (window, that earlier session)."""
    for w in prepared["windows"][1:]:
        target = getattr(w, which)
        earlier = target - pd.Timedelta(days=11)
        if earlier in idx and earlier > prepared["windows"][0].entry:
            return w, earlier
    raise AssertionError(f"no window {which} sits exactly 11 days after a session")


@pytest.mark.parametrize("which", ["entry", "exit"])
def test_a_bill_print_hole_ending_exactly_on_a_windows_first_or_last_session_is_seen_and_refused(which):
    """The age of the print a financed session reads is measured on EVERY session of every
    window, both boundary sessions included. Here the only stale read of the whole account is the boundary
    session itself (the last print is exactly 11 calendar days old on it; every session inside the hole reads
    the same print but is younger), so a report that skipped either boundary would call the series fresh."""
    world = _world()
    idx, spy, factors, panel, irx = world
    prepared = _prepare(world)
    w, last_print = _window_boundary_eleven_days_after_a_session(prepared, idx, which)
    boundary = getattr(w, which)
    holed = irx.drop(idx[(idx > last_print) & (idx < boundary)])
    assert boundary in holed.index and last_print in holed.index          # prints exist on both ends of the hole
    report = mod.bill_print_age_report(holed, idx, prepared["windows"])
    assert report["max_age_days"] == 11 and report["session"] == boundary.strftime("%Y-%m-%d")
    assert report["n_sessions_over_limit"] == 1
    clean = mod.bill_print_age_report(irx, idx, prepared["windows"])
    assert clean["max_age_days"] <= mod.MAX_BILL_PRINT_AGE_DAYS and clean["n_sessions_over_limit"] == 0
    with pytest.raises(mod.BillPrintStaleError) as err:
        mod.build_curve(prepared, panel, holed, cost_bps_by_symbol=_cost(panel), spy_entry_cost_bps=0.0)
    assert err.value.session == boundary.strftime("%Y-%m-%d") and err.value.max_age_days == 11
