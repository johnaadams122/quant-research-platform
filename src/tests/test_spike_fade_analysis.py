"""Tests for src/mining/spike_fade_analysis.py -- the ANALYSIS layer of the
range-bound spike-fade study. Synthetic
data only -- no network, no real prices, no ledger.

Layout:
  1. plateau score, ranking, top lists
  2. stability flags and the null pass rate
  3. vehicle table, sign, calendar mapping
  4. vehicle cell test against the random-entry null, era split
  5. candidate definition and the contaminated-series flag
  6. Black-Scholes pricer, put payoffs, break-even k, market k
  7. the ABI-1 transfer line (engine conventions, mark to market)
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

from mining import spike_fade_analysis as sfa
from mining import spike_fade_study as sfs


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

H_AXIS = (10, 21, 42, 63)
TP_AXIS = ("none", "half", "full")
SL_AXIS = ("none", "1.5", "3.0")


def _cell_frame(values, *, series="A", trigger="pct756", side="up_fade",
                entry="E0", ineligible=()):
    """One group's cell frame from {(h_idx, tp_idx, sl_idx): S}. Cells listed
    in `ineligible` (index triples) are marked not eligible."""
    rows = []
    for (h, t, s), S in values.items():
        rows.append({"series": series, "trigger": trigger, "side": side,
                     "entry": entry, "H": H_AXIS[h], "tp": TP_AXIS[t],
                     "sl": SL_AXIS[s], "mean_x": float(S),
                     "eligible": (h, t, s) not in set(ineligible),
                     "n": 12})
    return pd.DataFrame(rows)


def _full_grid(fn):
    return {(h, t, s): fn(h, t, s)
            for h in range(4) for t in range(3) for s in range(3)}


def _score(frame, h, t, s, **key):
    """Plateau score of the cell (h, t, s) of a scored frame."""
    m = ((frame["H"] == H_AXIS[h]) & (frame["tp"] == TP_AXIS[t])
         & (frame["sl"] == SL_AXIS[s]))
    for k, v in key.items():
        m &= frame[k] == v
    assert m.sum() == 1
    return float(frame.loc[m, "plateau_score"].iloc[0]), int(
        frame.loc[m, "n_eligible_neighbors"].iloc[0])


# ---------------------------------------------------------------------------
# 1. Plateau score
# ---------------------------------------------------------------------------

def _code_grid():
    """S = 100 h + 10 t + s: every cell distinct, medians hand-computable."""
    return _full_grid(lambda h, t, s: 100 * h + 10 * t + s)


def test_plateau_corner_cell_has_seven_neighbors_and_median_100():
    scored = sfa.plateau_scores(_cell_frame(_code_grid()))
    # corner (H=10, TP none, SL none): neighbors are the 2x2x2 block minus itself:
    # values 1, 10, 11, 100, 101, 110, 111 -> median (4th of 7) = 100.
    score, n_nb = _score(scored, 0, 0, 0)
    assert n_nb == 7
    assert score == 100.0


def test_plateau_edge_cell_has_eleven_neighbors_and_median_100():
    scored = sfa.plateau_scores(_cell_frame(_code_grid()))
    # (H=10, TP none, SL 1.5): h in {0,1} x t in {0,1} x s in {0,1,2} minus self:
    # 0 2 10 11 12 100 101 102 110 111 112 -> median (6th of 11) = 100.
    score, n_nb = _score(scored, 0, 0, 1)
    assert n_nb == 11
    assert score == 100.0


def test_plateau_neighbor_counts_by_position_class():
    scored = sfa.plateau_scores(_cell_frame(_code_grid()))
    # 4 x 3 x 3 axes: (values valid in {i-1, i, i+1}) multiplied, minus one
    assert _score(scored, 0, 1, 1)[1] == 2 * 3 * 3 - 1      # face
    assert _score(scored, 1, 1, 1)[1] == 3 * 3 * 3 - 1      # interior: 26
    assert _score(scored, 3, 2, 2)[1] == 2 * 2 * 2 - 1      # opposite corner
    assert _score(scored, 1, 0, 0)[1] == 3 * 2 * 2 - 1      # H interior, tp/sl at corner


def test_plateau_interior_cell_uses_all_26_neighbors():
    # S = 1 for every cell except one interior cell -> interior median stays 1
    vals = _full_grid(lambda h, t, s: 1.0)
    vals[(1, 1, 1)] = 999.0
    scored = sfa.plateau_scores(_cell_frame(vals))
    score, n_nb = _score(scored, 1, 1, 1)
    assert n_nb == 26 and score == 1.0
    # a neighbor of the odd cell sees it among its 26 (or fewer) neighbors,
    # but the median is still 1
    assert _score(scored, 0, 0, 0)[0] == 1.0


def test_plateau_excludes_cells_two_steps_away():
    # give the h=2 face the value 1e6: a corner at h=0 must not see it
    vals = _code_grid()
    for t in range(3):
        for s in range(3):
            vals[(2, t, s)] = 1e6
    scored = sfa.plateau_scores(_cell_frame(vals))
    # neighbors of (0,0,0): h in {0,1}; values 1,10,11,100,101,110,111 -> 100
    assert _score(scored, 0, 0, 0)[0] == 100.0


def test_plateau_median_uses_only_eligible_neighbors():
    vals = _code_grid()
    # make 4 of the 7 corner neighbors ineligible and absurdly large
    bad = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    for b in bad:
        vals[b] = 1e9
    scored = sfa.plateau_scores(_cell_frame(vals, ineligible=bad))
    # eligible neighbors of (0,0,0): (0,0,1)=1, (0,1,0)=10, (0,1,1)=11 -> median 10
    score, n_nb = _score(scored, 0, 0, 0)
    assert n_nb == 3
    assert score == 10.0


def test_plateau_no_score_with_fewer_than_three_eligible_neighbors():
    vals = _code_grid()
    bad = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (0, 1, 1)]   # leaves 2
    scored = sfa.plateau_scores(_cell_frame(vals, ineligible=bad))
    score, n_nb = _score(scored, 0, 0, 0)
    assert n_nb == 2
    assert math.isnan(score)


def test_plateau_two_eligible_neighbors_boundary_and_three_scored():
    vals = _code_grid()
    three = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]   # leaves 3 eligible
    scored = sfa.plateau_scores(_cell_frame(vals, ineligible=three))
    score, n_nb = _score(scored, 0, 0, 0)
    assert n_nb == 3 and not math.isnan(score)


def test_plateau_ineligible_own_cell_gets_no_score():
    vals = _code_grid()
    scored = sfa.plateau_scores(_cell_frame(vals, ineligible=[(1, 1, 1)]))
    assert math.isnan(_score(scored, 1, 1, 1)[0])
    # its neighbors still score, over 25 eligible neighbors
    assert _score(scored, 2, 1, 1)[1] == 25


def test_plateau_neighbors_stay_inside_group_and_entry_rule():
    base = _code_grid()
    e0 = _cell_frame(base, entry="E0")
    e1 = _cell_frame(_full_grid(lambda h, t, s: 1e6), entry="E1")
    other_trigger = _cell_frame(_full_grid(lambda h, t, s: 1e6), trigger="z20@2.0")
    other_side = _cell_frame(_full_grid(lambda h, t, s: 1e6), side="down_fade")
    other_series = _cell_frame(_full_grid(lambda h, t, s: 1e6), series="B")
    scored = sfa.plateau_scores(pd.concat([e0, e1, other_trigger, other_side,
                                           other_series], ignore_index=True))
    sub = scored[(scored["series"] == "A") & (scored["trigger"] == "pct756")
                 & (scored["side"] == "up_fade") & (scored["entry"] == "E0")]
    assert _score(sub, 0, 0, 0)[0] == 100.0
    assert _score(sub, 1, 1, 1)[1] == 26


def test_plateau_nan_s_is_treated_as_ineligible_neighbor():
    vals = _code_grid()
    vals[(1, 1, 1)] = float("nan")
    scored = sfa.plateau_scores(_cell_frame(vals))
    # the NaN cell still says eligible=True in the frame, but has no usable S
    assert _score(scored, 0, 0, 0)[1] == 6


def test_plateau_rejects_duplicate_cells_and_unknown_axis_values():
    f = _cell_frame(_code_grid())
    with pytest.raises(ValueError):
        sfa.plateau_scores(pd.concat([f, f.iloc[[0]]], ignore_index=True))
    g = f.copy()
    g.loc[0, "H"] = 15
    with pytest.raises(ValueError):
        sfa.plateau_scores(g)


def test_plateau_preserves_row_order_and_input_untouched():
    f = _cell_frame(_code_grid())
    snap = f.copy(deep=True)
    scored = sfa.plateau_scores(f)
    pd.testing.assert_frame_equal(f, snap)                 # input not mutated
    assert list(scored.index) == list(f.index)
    for c in f.columns:
        assert scored[c].equals(f[c])


def test_plateau_accepts_core_frame_sl_labels_and_sl_mult_column():
    # the core's frame uses sl labels 'none' / '1.5' / '3.0' (sfs.sl_label)
    assert [sfs.sl_label(x) for x in sfs.SL_GRID] == list(SL_AXIS)


# ---------------------------------------------------------------------------
# 1b. Plateau map: ranking, tie order, top lists
# ---------------------------------------------------------------------------

def _row(plateau, S, H=21, tp="none", sl="none", series="A", side="up_fade",
         trigger="pct756", entry="E0"):
    return {"series": series, "trigger": trigger, "side": side, "entry": entry,
            "H": H, "tp": tp, "sl": sl, "mean_x": S, "plateau_score": plateau,
            "n_eligible_neighbors": 5, "eligible": True, "n": 12}


def _order(frame):
    return [(r.plateau_score, r.mean_x, r.H, r.sl, r.tp, r.series, r.side,
             r.trigger, r.entry) for r in frame.itertuples()]


def test_rank_orders_by_plateau_score_descending_first():
    f = pd.DataFrame([_row(0.2, 9.0), _row(0.5, 0.0, H=10), _row(0.3, 1.0, H=42)])
    r = sfa.rank_cells(f)
    assert list(r["plateau_score"]) == [0.5, 0.3, 0.2]


def test_rank_tie_plateau_then_own_s_descending():
    f = pd.DataFrame([_row(0.4, 0.1), _row(0.4, 0.9, H=42), _row(0.4, 0.5, H=10)])
    r = sfa.rank_cells(f)
    assert list(r["mean_x"]) == [0.9, 0.5, 0.1]


def test_rank_tie_then_smaller_h_first():
    f = pd.DataFrame([_row(0.4, 0.5, H=63), _row(0.4, 0.5, H=10),
                      _row(0.4, 0.5, H=42)])
    assert list(sfa.rank_cells(f)["H"]) == [10, 42, 63]


def test_rank_tie_then_sl_none_first_then_tighter_before_looser():
    f = pd.DataFrame([_row(0.4, 0.5, sl="3.0"), _row(0.4, 0.5, sl="1.5"),
                      _row(0.4, 0.5, sl="none")])
    assert list(sfa.rank_cells(f)["sl"]) == ["none", "1.5", "3.0"]


def test_rank_tie_then_tp_none_first():
    f = pd.DataFrame([_row(0.4, 0.5, tp="full"), _row(0.4, 0.5, tp="none"),
                      _row(0.4, 0.5, tp="half")])
    assert list(sfa.rank_cells(f)["tp"]) == ["none", "half", "full"]


def test_rank_final_ties_are_lexicographic_series_side_trigger_entry():
    f = pd.DataFrame([
        _row(0.4, 0.5, series="B"),
        _row(0.4, 0.5, series="A", side="up_fade"),
        _row(0.4, 0.5, series="A", side="down_fade", trigger="z20@2.0"),
        _row(0.4, 0.5, series="A", side="down_fade", trigger="pct756", entry="E1"),
        _row(0.4, 0.5, series="A", side="down_fade", trigger="pct756", entry="E0"),
    ])
    r = sfa.rank_cells(f)
    got = [(x.series, x.side, x.trigger, x.entry) for x in r.itertuples()]
    assert got == [("A", "down_fade", "pct756", "E0"),
                   ("A", "down_fade", "pct756", "E1"),
                   ("A", "down_fade", "z20@2.0", "E0"),
                   ("A", "up_fade", "pct756", "E0"),
                   ("B", "up_fade", "pct756", "E0")]


def test_rank_is_permutation_invariant():
    rows = [_row(0.4, 0.5, H=h, sl=sl, tp=tp)
            for h in (10, 21) for sl in ("none", "1.5") for tp in ("none", "half")]
    a = sfa.rank_cells(pd.DataFrame(rows))
    b = sfa.rank_cells(pd.DataFrame(rows[::-1]))
    assert _order(a) == _order(b)


def test_rank_drops_cells_without_a_plateau_score():
    f = pd.DataFrame([_row(0.4, 0.5), _row(float("nan"), 9.0, H=10)])
    assert len(sfa.rank_cells(f)) == 1


def test_top_lists_one_line_per_group_and_entry_rule():
    # same (group, entry): only the best-ranked cell survives; E1 is its own line
    f = pd.DataFrame([_row(0.5, 0.5, H=10), _row(0.4, 0.9, H=21),
                      _row(0.3, 0.1, H=42, entry="E1")])
    top = sfa.top_cells(f, per_pair=10, overall=30)
    assert len(top["overall"]) == 2
    assert list(top["overall"]["entry"]) == ["E0", "E1"]
    assert list(top["overall"]["H"]) == [10, 42]


def test_top_lists_cap_per_pair_and_overall():
    rows = []
    for series in ("A", "B", "C", "D"):
        for i in range(13):                       # 13 distinct triggers-worth of lines
            rows.append(_row(1.0 - 0.001 * i, 0.1, series=series, trigger=f"t{i:02d}"))
    top = sfa.top_cells(pd.DataFrame(rows), per_pair=10, overall=30)
    per_pair = top["per_pair"]
    assert len(per_pair) == 40
    assert per_pair.groupby(["series", "side"]).size().max() == 10
    assert len(top["overall"]) == 30
    # overall is the deduplicated ranking's head, independent of the per-pair cap
    assert list(top["overall"]["plateau_score"]) == sorted(
        top["overall"]["plateau_score"], reverse=True)


def test_top_lists_pair_is_series_and_side():
    rows = [_row(0.9, 0.5, side="up_fade", trigger="a"),
            _row(0.8, 0.5, side="down_fade", trigger="b"),
            _row(0.7, 0.5, side="up_fade", trigger="c")]
    top = sfa.top_cells(pd.DataFrame(rows), per_pair=1, overall=30)
    assert len(top["per_pair"]) == 2
    assert set(top["per_pair"]["trigger"]) == {"a", "b"}


def test_top_lists_default_caps_are_10_and_30():
    assert sfa.TOP_PER_PAIR == 10 and sfa.TOP_OVERALL == 30
    assert sfa.MIN_ELIGIBLE_NEIGHBORS == 3


# ---------------------------------------------------------------------------
# 2. Plateau map: stability flags (pre-declared pass rule) and the null pass rate
# ---------------------------------------------------------------------------

def _dates(n, start="2001-01-01"):
    """n episode dates, one every 40 calendar days, ascending."""
    return pd.DatetimeIndex([pd.Timestamp(start) + pd.Timedelta(days=40 * i)
                             for i in range(n)])


def test_stability_passes_when_result_survives_removals_and_halves_agree():
    x = [3, 1, 1, 1, 1, 1, 1, -1]            # n = 8, S = 8/8 = 1.0
    r = sfa.stability_flags(x, _dates(8))
    assert r.n == 8 and r.S == pytest.approx(1.0)
    # drop the largest (3): sum 5 over 7 ; drop the two largest (3, 1): sum 4 over 6
    assert r.S_drop1 == pytest.approx(5 / 7)
    assert r.S_drop2 == pytest.approx(4 / 6)
    assert r.a_pass is True
    # halves by date: [3,1,1,1] mean 1.5 ; [1,1,1,-1] mean 0.5 -> both positive
    assert r.b_applicable is True and r.b_pass is True
    assert r.first_half_mean == pytest.approx(1.5)
    assert r.second_half_mean == pytest.approx(0.5)
    assert r.passed is True


def test_stability_fails_when_one_episode_carries_the_result():
    x = [10, 0.1, -1, -1, -1, 0.5, 0.2, 0.3]   # S = 8.1/8 > 0 but drop 10 -> negative
    r = sfa.stability_flags(x, _dates(8))
    assert r.S > 0
    assert r.S_drop1 == pytest.approx((8.1 - 10) / 7)
    assert r.S_drop1 < 0
    assert r.a_pass is False and r.passed is False


def test_stability_fails_when_only_the_two_largest_removal_breaks_it():
    x = [6, 5, -1, -1, -1, 0.5, 0.5, 0.5]      # sum 9.5
    r = sfa.stability_flags(x, _dates(8))
    assert r.S_drop1 == pytest.approx(3.5 / 7) and r.S_drop1 > 0
    assert r.S_drop2 == pytest.approx(-1.5 / 6) and r.S_drop2 < 0
    assert r.a_pass is False and r.passed is False


def test_stability_removal_must_stay_strictly_above_zero():
    # drop-1 mean is exactly zero: 'above zero' is strict
    x = [4, 1, 1, 1, -1, -1, -1, 0]
    assert sum(x) == 4
    r = sfa.stability_flags(x, _dates(8))
    assert r.S_drop1 == pytest.approx(0.0)
    assert r.a_pass is False


def test_stability_requires_positive_full_sample_mean():
    r = sfa.stability_flags([-1, -2, -1, -1, -3, -1, -2, -1], _dates(8))
    assert r.S < 0 and r.a_pass is False and r.passed is False


def test_stability_sign_disagreement_across_halves_fails():
    x = [-1, -1, -1, -0.5, 2, 2, 2, 2]         # S = 0.5625 > 0, both drops > 0
    r = sfa.stability_flags(x, _dates(8))
    assert r.a_pass is True
    assert r.b_applicable is True and r.b_pass is False
    assert r.first_half_mean < 0 < r.second_half_mean
    assert r.passed is False


def test_stability_halves_follow_date_order_not_input_order():
    x = np.array([-1, -1, -1, -0.5, 2, 2, 2, 2])
    d = _dates(8)
    order = np.array([7, 3, 5, 1, 0, 6, 2, 4])          # scrambled
    r = sfa.stability_flags(x[order], d[order])
    assert r.first_half_mean == pytest.approx(np.mean(x[:4]))
    assert r.second_half_mean == pytest.approx(np.mean(x[4:]))
    assert r.b_pass is False


def test_stability_halves_below_four_episodes_are_not_applicable():
    r = sfa.stability_flags([1.0] * 7, _dates(7))       # halves of 3 and 4
    assert r.b_applicable is False and r.b_pass is None
    # passes on rule (a) alone when (b) is not applicable
    assert r.a_pass is True and r.passed is True
    assert (r.n_first, r.n_second) == (3, 4)


def test_stability_odd_n_gives_the_extra_episode_to_the_second_half():
    r = sfa.stability_flags([1.0] * 9, _dates(9))
    assert (r.n_first, r.n_second) == (4, 5)
    assert r.b_applicable is True and r.b_pass is True


def test_stability_a_half_with_zero_mean_is_not_agreement():
    x = [1, -1, 1, -1, 2, 2, 2, 2]             # first half mean 0
    r = sfa.stability_flags(x, _dates(8))
    assert r.first_half_mean == pytest.approx(0.0)
    assert r.b_pass is False and r.passed is False


def test_stability_two_episodes_cannot_pass_rule_a():
    # after removing the two largest nothing is left: S_drop2 undefined -> fail
    r = sfa.stability_flags([2.0, 1.0], _dates(2))
    assert math.isnan(r.S_drop2) and r.a_pass is False


def test_stability_drops_nan_x():
    x = [3, 1, 1, 1, 1, 1, 1, -1, float("nan")]
    d = _dates(9)
    r = sfa.stability_flags(x, d)
    assert r.n == 8 and r.S == pytest.approx(1.0)


def test_stability_length_mismatch_raises():
    with pytest.raises(ValueError):
        sfa.stability_flags([1.0, 2.0], _dates(3))


def test_null_stability_pass_rate_hand_computed():
    good = ([3, 1, 1, 1, 1, 1, 1, -1], _dates(8))            # passes
    carried = ([10, 0.1, -1, -1, -1, 0.5, 0.2, 0.3], _dates(8))  # positive S, fails (a)
    negative = ([-1] * 8, _dates(8))                          # negative S, fails
    tiny = ([1.0] * 5, _dates(5))                             # ineligible (n < 8)
    r = sfa.null_stability_pass_rate([good, carried, negative, tiny, good])
    assert r["n_cells"] == 4 and r["n_ineligible"] == 1
    assert r["n_pass"] == 2 and r["pass_rate"] == pytest.approx(0.5)
    # among null cells with a positive S (good, carried, good): 2 of 3 pass
    assert r["n_positive"] == 3
    assert r["pass_rate_given_positive"] == pytest.approx(2 / 3)


def test_null_stability_pass_rate_empty_is_nan_not_an_error():
    r = sfa.null_stability_pass_rate([])
    assert r["n_cells"] == 0 and math.isnan(r["pass_rate"])
    assert math.isnan(r["pass_rate_given_positive"])


def test_stability_for_cell_reads_traded_x_of_a_variant_result():
    rng = np.random.default_rng(3)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900)))
    ser = sfs.build_series(pd.Series(p, index=pd.bdate_range("2000-01-03", periods=900)),
                           name="SYN")
    t0 = np.arange(300, 800, 40)
    cell = sfs.PRIMARY_CELL
    res = sfs.evaluate_variants(ser, t0, "up_fade", [cell])
    got = sfa.stability_for_cell(res, cell, ser.dates)
    ok = res.status[0] == sfs.STATUS_TRADED
    ref = sfa.stability_flags(res.x[0][ok], ser.dates[res.t0[ok]])
    assert got == ref and got.n == int(ok.sum()) and got.n > 0


# ---------------------------------------------------------------------------
# 3. Vehicle test: vehicle table, sign, calendar mapping
# ---------------------------------------------------------------------------

SPEC_TABLE = {
    # (series, side): (vehicle, position, first date)
    ("^VIX", "up_fade"): ("SVXY", "long", "2011-10-04"),
    ("^VIX", "down_fade"): ("VIXY", "long", "2011-01-04"),
    ("WTI spot", "up_fade"): ("USO", "short", "2006-04-10"),
    ("WTI spot", "down_fade"): ("USO", "long", "2006-04-10"),
    ("Brent spot", "up_fade"): ("BNO", "short", "2010-06-02"),
    ("Brent spot", "down_fade"): ("BNO", "long", "2010-06-02"),
    ("Henry Hub spot", "up_fade"): ("UNG", "short", "2007-04-18"),
    ("Henry Hub spot", "down_fade"): ("UNG", "long", "2007-04-18"),
    ("GC=F", "up_fade"): ("GLD", "short", "2004-11-18"),
    ("GC=F", "down_fade"): ("GLD", "long", "2004-11-18"),
    ("HG=F", "up_fade"): ("CPER", "short", "2011-11-15"),
    ("HG=F", "down_fade"): ("CPER", "long", "2011-11-15"),
    ("DX-Y.NYB", "up_fade"): ("UUP", "short", "2007-03-01"),
    ("DX-Y.NYB", "down_fade"): ("UUP", "long", "2007-03-01"),
}


def test_vehicle_table_matches_the_spec_exactly():
    assert set(sfa.VEHICLE_TABLE) == set(SPEC_TABLE)
    for key, (ticker, position, first) in SPEC_TABLE.items():
        v = sfa.VEHICLE_TABLE[key]
        assert (v.ticker, v.position) == (ticker, position), key
        assert v.first_date == pd.Timestamp(first), key
        assert isinstance(v.note, str) and v.note


def test_vehicles_is_the_spec_name_of_the_same_table():
    assert sfa.VEHICLES is sfa.VEHICLE_TABLE
    assert sfa.VEHICLES[("GC=F", "up_fade")].ticker == "GLD"


def test_vehicle_table_first_date_note_mentions_the_svxy_era_split():
    note = sfa.VEHICLE_TABLE[("^VIX", "up_fade")].note
    assert "2018-02-28" in note
    assert sfa.SVXY_ERA_SPLIT == pd.Timestamp("2018-02-28")


def test_vehicle_position_sign_long_is_minus_one_and_short_is_plus_one():
    assert sfa.position_sign("long") == -1
    assert sfa.position_sign("short") == +1
    with pytest.raises(ValueError):
        sfa.position_sign("flat")


def test_vehicle_for_lookup_and_unknown_pair():
    assert sfa.vehicle_for("WTI spot", "down_fade").ticker == "USO"
    with pytest.raises(KeyError):
        sfa.vehicle_for("^OVX", "up_fade")           # exploratory-only: no vehicle


def test_series_aliases_resolve_to_spec_names():
    assert sfa.canonical_series("DCOILWTICO") == "WTI spot"
    assert sfa.canonical_series("DCOILBRENTEU") == "Brent spot"
    assert sfa.canonical_series("DHHNGSP") == "Henry Hub spot"
    assert sfa.canonical_series("^VIX") == "^VIX"


def test_fred_series_get_one_extra_session_lag_others_none():
    assert sfa.fred_extra_lag("WTI spot") == 1
    assert sfa.fred_extra_lag("Brent spot") == 1
    assert sfa.fred_extra_lag("Henry Hub spot") == 1
    assert sfa.fred_extra_lag("DCOILWTICO") == 1
    for s in ("^VIX", "GC=F", "HG=F", "DX-Y.NYB"):
        assert sfa.fred_extra_lag(s) == 0


def test_contaminated_series_are_vix_wti_and_brent_only():
    assert set(sfa.CONTAMINATED_SERIES) == {"^VIX", "WTI spot", "Brent spot"}
    assert sfa.is_contaminated("^VIX") and sfa.is_contaminated("DCOILWTICO")
    assert not sfa.is_contaminated("GC=F") and not sfa.is_contaminated("Henry Hub spot")


# --- calendar mapping acceptance test ------------------------------------------

def _calendars():
    """Trigger calendar T (1400 business days) and a vehicle calendar V that
    starts at T[300] and lacks three trigger sessions (its own holidays)."""
    T = pd.bdate_range("2005-01-03", periods=1400)
    holidays = {400, 700, 701}
    keep = [i for i in range(300, 1400) if i not in holidays]
    V = T[keep]
    return T, V, holidays


def test_map_anchor_is_last_vehicle_session_on_or_before_t0():
    T, V, _ = _calendars()
    # trigger 700 is a vehicle holiday: last vehicle session on or before is T[699]
    # V position of T[699] = 699 - 300 - 1 (only T[400] is missing before it) = 398
    m = sfs.map_anchors_to_vehicle(T[[700, 705, 900]], V)
    assert list(m.positions) == [398, 705 - 300 - 3, 900 - 300 - 3]
    assert V[m.positions[0]] == T[699]           # the holiday anchor moved back one session
    assert V[m.positions[1]] == T[705]           # a shared session maps to itself
    assert list(m.session_dates) == [V[p] for p in m.positions]


def test_map_two_consecutive_vehicle_holidays_both_map_to_the_last_session_before():
    T, V, _ = _calendars()
    m = sfs.map_anchors_to_vehicle(T[[701]], V)            # 700 and 701 both missing
    assert V[m.positions[0]] == T[699]


def test_map_anchor_before_first_vehicle_session_is_dropped_and_counted():
    T, V, _ = _calendars()
    m = sfs.map_anchors_to_vehicle(T[[100, 299, 300]], V, min_history=1)
    assert m.dropped_no_session == 2 and list(m.positions) == [0]
    m2 = sfs.map_anchors_to_vehicle(T[[100, 299, 300]], V)      # the 252-session history rule
    assert (m2.dropped_no_session, m2.dropped_short_history) == (2, 1)
    assert len(m2.positions) == 0


def test_map_extra_fred_lag_adds_one_vehicle_session():
    T, V, _ = _calendars()
    base = sfs.map_anchors_to_vehicle(T[[900]], V)
    lagged = sfs.map_anchors_to_vehicle(T[[900]], V, extra_lag=1)
    assert lagged.positions[0] == base.positions[0] + 1


def test_map_extra_lag_can_run_past_the_vehicle_end_and_is_counted():
    T, V, _ = _calendars()
    m = sfs.map_anchors_to_vehicle(pd.DatetimeIndex([T[-1]]), V, extra_lag=1)
    assert m.dropped_beyond_end == 1 and len(m.positions) == 0


# ---------------------------------------------------------------------------
# 4. Vehicle test: the vehicle cell test against the random-entry null
# ---------------------------------------------------------------------------

def _vehicle_closes(n, seed, vol=0.006, boosts=(), boost=0.0, boost_len=21):
    """Close path with fat-tailed noise; every position in `boosts` (a vehicle
    anchor) gets `boost` added to each log return of the boost_len sessions
    that follow the entry close (anchor + 1)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4.5, size=n) * vol / math.sqrt(4.5 / 2.5)
    for a in boosts:
        r[a + 2: a + 2 + boost_len] += boost
    return 100.0 * np.exp(np.cumsum(r))


def _vehicle_from(V, closes, name="VEH"):
    return sfs.build_series(pd.Series(closes, index=V), name=name)


def test_vehicle_test_hand_computed_entry_exit_and_long_sign():
    T, V, _ = _calendars()
    closes = _vehicle_closes(len(V), 5)
    veh = _vehicle_from(V, closes)
    anchors = T[[700, 900]]                       # 700 is a vehicle holiday -> V pos 398
    res = sfa.vehicle_cell_test(veh, anchors, sfs.PRIMARY_CELL, "long", B=20, seed=1)
    a = [398, 900 - 300 - 3]
    entry = [i + 1 for i in a]
    exit_ = [i + 21 for i in entry]                # time stop, 21 sessions after entry
    want = [closes[x] / closes[e] - 1 for e, x in zip(entry, exit_)]   # long gains when it rises
    assert res.n == 2
    assert res.mean_pnl == pytest.approx(np.mean(want))
    assert res.hit_rate == pytest.approx(np.mean([w > 0 for w in want]))
    tr = res.trades
    assert list(tr["entry_date"]) == [V[e] for e in entry]
    assert list(tr["exit_date"]) == [V[x] for x in exit_]
    assert list(tr["anchor_date"]) == [V[i] for i in a]
    assert tr["pnl"].to_numpy() == pytest.approx(want)


def test_vehicle_short_position_flips_the_sign_of_the_pnl():
    T, V, _ = _calendars()
    closes = _vehicle_closes(len(V), 5)
    veh = _vehicle_from(V, closes)
    anchors = T[[700, 900]]
    lng = sfa.vehicle_cell_test(veh, anchors, sfs.PRIMARY_CELL, "long", B=10, seed=1)
    sht = sfa.vehicle_cell_test(veh, anchors, sfs.PRIMARY_CELL, "short", B=10, seed=1)
    assert sht.mean_pnl == pytest.approx(-lng.mean_pnl)
    assert lng.sign == -1 and sht.sign == +1


def test_vehicle_stop_and_median_use_the_vehicles_own_closes_and_sign():
    # long vehicle: D = -(P_entry - M) = M - P_entry, so D > 0 when the vehicle sits
    # BELOW its own 252-session median at entry; check against the core directly.
    T, V, _ = _calendars()
    closes = _vehicle_closes(len(V), 8)
    veh = _vehicle_from(V, closes)
    a = 500
    res = sfa.vehicle_cell_test(veh, V[[a]], ("E0", 21, "full", None), "long",
                                B=10, seed=1)
    ctx = veh.context()
    entry_px = veh.p[a + 1]
    D_hand = -(entry_px - ctx.med[a])
    med_hand = float(np.median(closes[a - 251:a + 1]))
    assert ctx.med[a] == pytest.approx(med_hand)
    assert res.trades["D"].iloc[0] == pytest.approx(D_hand)


def test_vehicle_drops_and_counts_no_session_short_history_and_censored():
    T, V, _ = _calendars()
    veh = _vehicle_from(V, _vehicle_closes(len(V), 5))
    # trigger positions: 100, 150 (before the vehicle), 350 (V pos 50), 500 (V pos 199),
    # 700 (V 398), 760 (V 457), 1350 (V 1047: beyond n-70 -> censored)
    anchors = T[[100, 150, 350, 500, 700, 760, 1350]]
    res = sfa.vehicle_cell_test(veh, anchors, sfs.PRIMARY_CELL, "long", B=10, seed=1)
    assert res.n_anchors_input == 7
    assert res.dropped_no_session == 2
    assert res.dropped_short_history == 2
    assert res.dropped_censored == 1
    assert res.n_anchors_kept == 2
    assert res.n == 2
    # every input anchor is accounted for exactly once
    assert (res.dropped_no_session + res.dropped_short_history + res.dropped_censored
            + res.dropped_past_end + res.dropped_duplicate + res.n_anchors_kept) == 7


def test_vehicle_extra_lag_shifts_entry_by_one_session():
    T, V, _ = _calendars()
    veh = _vehicle_from(V, _vehicle_closes(len(V), 5))
    a = T[[900]]
    base = sfa.vehicle_cell_test(veh, a, sfs.PRIMARY_CELL, "long", B=10, seed=1)
    lag = sfa.vehicle_cell_test(veh, a, sfs.PRIMARY_CELL, "long", B=10, seed=1,
                                extra_lag=1)
    d0, d1 = base.trades["entry_date"].iloc[0], lag.trades["entry_date"].iloc[0]
    assert V.get_loc(d1) == V.get_loc(d0) + 1


def test_vehicle_pair_helper_applies_table_position_and_fred_lag():
    T, V, _ = _calendars()
    veh = _vehicle_from(V, _vehicle_closes(len(V), 5))
    a = T[[900]]
    wti_up = sfa.vehicle_test_for_pair("WTI spot", "up_fade", veh, a,
                                       sfs.PRIMARY_CELL, B=10, seed=1)
    assert wti_up.position == "short" and wti_up.sign == +1 and wti_up.vehicle == "USO"
    assert wti_up.extra_lag == 1                    # FRED series
    gold = sfa.vehicle_test_for_pair("GC=F", "down_fade", veh, a,
                                     sfs.PRIMARY_CELL, B=10, seed=1)
    assert gold.position == "long" and gold.extra_lag == 0
    assert gold.contaminated is False and wti_up.contaminated is True


def test_vehicle_null_draws_match_observed_n_strata_and_spacing_exactly():
    V, veh, anchors = _vehicle_with_effect(0.0)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=30, seed=5,
                                collect_draws=True)
    A = res.null_anchors
    assert A.shape == (res.null_reps_used, res.n) and res.null_reps_used > 0
    ctx = veh.context()
    for row in A:
        assert list(row) == sorted(row)
        assert (np.diff(row) >= sfs.GAP_SESSIONS).all()             # gap20 spacing
        assert row.min() >= 251 and row.max() + sfs.CENSOR_TAIL <= veh.n - 1
        # D-sign strata equal the observed strata exactly, D on the vehicle's own closes
        D = -(veh.p[row + 1] - ctx.med[row])
        assert int((D > 0).sum()) == res.n_pos_d
        assert int((D <= 0).sum()) == res.n_nonpos_d
        # every pseudo-anchor is a valid trade of this cell on this vehicle
        r = sfs.evaluate_variants(veh, row, -1, [sfs.PRIMARY_CELL])
        assert (r.status[0] == sfs.STATUS_TRADED).all() and np.isfinite(r.x[0]).all()
    # without collect_draws nothing is stored
    none = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=5, seed=5)
    assert none.null_anchors is None


def test_vehicle_null_replication_that_cannot_reach_the_quota_is_failed_not_padded():
    # ten well-spaced anchors on a short vehicle: 10 anchors at 20-session spacing need at
    # least 181 sessions of eligible room; a vehicle with only ~130 leaves no draw feasible
    n = 251 + 70 + 130
    V = pd.bdate_range("2006-01-02", periods=n)
    veh = _vehicle_from(V, _vehicle_closes(n, 3))
    anchors = V[[251 + 12 * i for i in range(10)]]        # observed anchors are not spaced
    res = sfa.vehicle_cell_test(veh, anchors, sfs.PRIMARY_CELL, "long", B=25, seed=1)
    assert res.n >= 8
    assert res.null_reps_failed == 25 and res.null_reps_used == 0
    assert math.isnan(res.null_percentile) and math.isnan(res.excess_pp)


def test_vehicle_series_with_its_own_extra_lag_is_refused_to_avoid_a_double_lag():
    V = pd.bdate_range("2006-01-02", periods=900)
    veh = sfs.build_series(pd.Series(_vehicle_closes(900, 3), index=V), name="X", extra_lag=1)
    with pytest.raises(ValueError):
        sfa.vehicle_cell_test(veh, V[[400]], sfs.PRIMARY_CELL, "long", B=5, seed=1)


def test_vehicle_null_uses_the_null_modules_public_matched_draw():
    from mining import spike_fade_null as nl
    assert callable(nl.draw_matched_anchors) and callable(nl.series_rng)
    # two runs through the wrapper agree with the module's own stream for the same seed
    V, veh, anchors = _vehicle_with_effect(0.0)
    a = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=8, seed=9,
                              collect_draws=True)
    b = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=8, seed=9,
                              collect_draws=True)
    assert np.array_equal(a.null_anchors, b.null_anchors)


def test_null_cells_for_stability_wires_the_n1_null_to_the_core_engine():
    rng = np.random.default_rng(3)
    n = 4000
    p = 100 * np.exp(np.cumsum(rng.standard_t(4.0, size=n) * 0.01 / math.sqrt(2)))
    ser = sfs.build_series(pd.Series(p, index=pd.bdate_range("2000-01-03", periods=n)),
                           name="SYN")
    cells = sfa.null_cells_for_stability(ser, "z20@2.0", "up_fade", sfs.PRIMARY_CELL,
                                         B=12, seed=1)
    assert 0 < len(cells) <= 12
    lengths = {len(x) for x, _ in cells}
    assert len(lengths) == 1 and lengths.pop() >= 8       # exact valid n, every replication
    for x, d in cells:
        assert np.isfinite(x).all() and len(d) == len(x)
        assert list(d) == sorted(d) and d.isin(ser.dates).all()
    rate = sfa.null_stability_pass_rate(cells)
    assert rate["n_cells"] + rate["n_ineligible"] == len(cells)
    # deterministic in the seed
    again = sfa.null_cells_for_stability(ser, "z20@2.0", "up_fade", sfs.PRIMARY_CELL,
                                         B=12, seed=1)
    assert all(np.array_equal(x1, x2) and d1.equals(d2)
               for (x1, d1), (x2, d2) in zip(cells, again))


def _vehicle_with_effect(boost, n=1500, seed=21):
    """Vehicle where 10 chosen anchors are followed by `boost` per session."""
    V = pd.bdate_range("2006-01-02", periods=n)
    anchors = [300 + 100 * i for i in range(10)]
    closes = _vehicle_closes(n, seed, boosts=anchors, boost=boost)
    return V, _vehicle_from(V, closes), anchors


def test_vehicle_excess_positive_when_long_vehicle_rallies_after_anchors():
    V, veh, anchors = _vehicle_with_effect(+0.006)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=200, seed=7)
    assert res.n == 10
    assert res.mean_pnl > 0.10                       # about 21 * 0.006 = 12.6 percent
    assert res.excess_pp > 5.0
    assert res.null_percentile >= 0.95
    assert res.excess_sigma > 1.0
    assert res.excess_pp == pytest.approx(100 * (res.mean_pnl - res.null_mean_pnl))
    assert res.null_reps_used + res.null_reps_failed == 200


def test_vehicle_excess_negative_when_long_vehicle_falls_after_anchors():
    V, veh, anchors = _vehicle_with_effect(-0.006)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=200, seed=7)
    assert res.excess_pp < -5.0
    assert res.null_percentile <= 0.05


def test_vehicle_short_gains_when_vehicle_falls_so_sign_is_not_inverted():
    V, veh, anchors = _vehicle_with_effect(-0.006)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "short", B=200, seed=7)
    assert res.mean_pnl > 0.05 and res.excess_pp > 5.0 and res.null_percentile >= 0.95


def test_vehicle_null_is_deterministic_and_seed_sensitive():
    V, veh, anchors = _vehicle_with_effect(+0.002)
    a = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=60, seed=3)
    b = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=60, seed=3)
    c = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=60, seed=4)
    assert (a.null_mean_pnl, a.null_percentile) == (b.null_mean_pnl, b.null_percentile)
    assert a.null_mean_pnl != c.null_mean_pnl


def test_vehicle_null_percentile_is_midrank_on_the_null_means():
    # zero-effect vehicle: the percentile of the observed mean lies inside (0, 1)
    V, veh, anchors = _vehicle_with_effect(0.0)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=100, seed=5)
    assert 0.0 < res.null_percentile < 1.0
    assert len(res.null_means_pnl) == res.null_reps_used
    obs = res.mean_pnl
    arr = np.asarray(res.null_means_pnl)
    want = ((arr < obs).sum() + 0.5 * (arr == obs).sum()) / len(arr)
    assert res.null_percentile == pytest.approx(want)


def test_vehicle_reports_dsign_strata_counts_and_unconditional_return():
    V, veh, anchors = _vehicle_with_effect(0.0)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=20, seed=5)
    assert res.n_pos_d + res.n_nonpos_d + res.n_undef_d == res.n
    ctx = veh.context()
    exp_pos = 0
    for a in anchors:
        D = -(veh.p[a + 1] - ctx.med[a])
        exp_pos += int(D > 0)
    assert res.n_pos_d == exp_pos
    assert np.isfinite(res.unconditional_mean_pnl)
    assert np.isfinite(res.unconditional_mean_daily_return)
    assert res.window_start < res.window_end


def test_vehicle_underpowered_flag_below_eight_episodes():
    V, veh, anchors = _vehicle_with_effect(0.0)
    r7 = sfa.vehicle_cell_test(veh, V[anchors[:7]], sfs.PRIMARY_CELL, "long", B=10, seed=1)
    r8 = sfa.vehicle_cell_test(veh, V[anchors[:8]], sfs.PRIMARY_CELL, "long", B=10, seed=1)
    assert r7.n == 7 and r7.underpowered_on_vehicle is True
    assert r8.n == 8 and r8.underpowered_on_vehicle is False


def test_vehicle_era_split_is_reported_pooled_by_entry_date():
    V, veh, anchors = _vehicle_with_effect(+0.004)
    split = V[anchors[4]] + pd.Timedelta(days=1)         # anchors 0-4 before, 5-9 after
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=40, seed=5,
                                era_split=split)
    assert res.era["pre"]["n"] == 5 and res.era["post"]["n"] == 5
    tr = res.trades
    pre = tr[tr["entry_date"] <= split]
    assert res.era["pre"]["mean_pnl"] == pytest.approx(pre["pnl"].mean())
    # the pooled headline is untouched by the split
    assert res.n == 10 and res.mean_pnl == pytest.approx(tr["pnl"].mean())
    # no era block without a split
    none = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=10, seed=5)
    assert none.era is None


def test_vehicle_era_split_puts_an_entry_on_the_split_date_in_the_pre_era():
    V, veh, anchors = _vehicle_with_effect(+0.004)
    split = V[anchors[4] + 1]                       # exactly the fifth trade's entry date
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=10, seed=5,
                                era_split=split)
    assert res.era["pre"]["n"] == 5 and res.era["post"]["n"] == 5


def test_midrank_counts_ties_as_half_and_handles_the_extremes():
    arr = np.array([1.0, 2.0, 2.0, 3.0])
    assert sfa._midrank(arr, 2.0) == pytest.approx((1 + 0.5 * 2) / 4)
    assert sfa._midrank(arr, 0.5) == 0.0 and sfa._midrank(arr, 9.0) == 1.0
    assert math.isnan(sfa._midrank(np.array([]), 1.0))
    assert math.isnan(sfa._midrank(arr, float("nan")))


def test_vehicle_result_with_no_valid_anchor_is_nan_not_an_error():
    V, veh, _ = _vehicle_with_effect(0.0)
    res = sfa.vehicle_cell_test(veh, V[[10, 20]], sfs.PRIMARY_CELL, "long", B=10, seed=1)
    assert res.n == 0 and math.isnan(res.mean_pnl) and math.isnan(res.null_percentile)
    assert res.underpowered_on_vehicle is True


def test_vehicle_test_refuses_anchor_dates_after_the_cutoff():
    # the loader truncates a vehicle at the cutoff, and an anchor dated after it
    # would be a look at sealed data: refuse it loudly, never drop it quietly
    V = pd.bdate_range("2021-01-04", periods=800)          # runs past 2023-06-30
    veh = sfs.build_series(pd.Series(_vehicle_closes(800, 3), index=V), name="X")
    assert veh.dates.max() <= sfs.DISCOVERY_CUTOFF
    with pytest.raises(sfs.SealedWindowError):
        sfa.vehicle_cell_test(veh, pd.DatetimeIndex([pd.Timestamp("2023-07-05")]),
                              sfs.PRIMARY_CELL, "long", B=5, seed=1)


# ---------------------------------------------------------------------------
# 5. Vehicle test: the candidate definition
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _vres(**over):
    base = dict(series="GC=F", n=10, excess_pp=1.5, null_percentile=0.90)
    base.update(over)
    return SimpleNamespace(**base)


def test_candidate_when_all_three_conditions_hold():
    v = sfa.candidate_definition(0.08, _vres())
    assert v.is_candidate is True and v.status == "candidate" and v.reasons == ()


def test_candidate_headline_p_boundary_is_at_most_point_ten():
    assert sfa.candidate_definition(0.10, _vres()).is_candidate is True
    just = sfa.candidate_definition(0.1000001, _vres())
    assert just.is_candidate is False and "headline_p_above_max" in just.reasons
    assert just.status == "headline_p_above_max"


def test_candidate_headline_p_missing_or_nan_is_not_a_candidate():
    assert sfa.candidate_definition(None, _vres()).is_candidate is False
    nan = sfa.candidate_definition(float("nan"), _vres())
    assert nan.is_candidate is False and "headline_p_unavailable" in nan.reasons


def test_candidate_requires_a_vehicle_result():
    v = sfa.candidate_definition(0.05, None)
    assert v.is_candidate is False and v.status == "vehicle_test_not_run"


def test_candidate_underpowered_on_vehicle_below_eight_episodes():
    v = sfa.candidate_definition(0.05, _vres(n=7))
    assert v.is_candidate is False and v.underpowered_on_vehicle is True
    assert "underpowered_on_vehicle" in v.reasons
    assert sfa.candidate_definition(0.05, _vres(n=8)).is_candidate is True


def test_candidate_vehicle_excess_must_be_strictly_positive():
    zero = sfa.candidate_definition(0.05, _vres(excess_pp=0.0))
    assert zero.is_candidate is False and "vehicle_excess_not_positive" in zero.reasons
    neg = sfa.candidate_definition(0.05, _vres(excess_pp=-0.2))
    assert neg.is_candidate is False
    nan = sfa.candidate_definition(0.05, _vres(excess_pp=float("nan")))
    assert nan.is_candidate is False


def test_candidate_null_percentile_boundary_is_at_least_point_eighty():
    assert sfa.candidate_definition(0.05, _vres(null_percentile=0.80)).is_candidate is True
    low = sfa.candidate_definition(0.05, _vres(null_percentile=0.7999))
    assert low.is_candidate is False and "null_percentile_below_min" in low.reasons
    assert sfa.candidate_definition(0.05, _vres(null_percentile=float("nan"))
                                    ).is_candidate is False


def test_candidate_lists_every_failing_condition():
    v = sfa.candidate_definition(0.5, _vres(n=3, excess_pp=-1.0, null_percentile=0.1))
    assert set(v.reasons) == {"headline_p_above_max", "underpowered_on_vehicle",
                              "vehicle_excess_not_positive", "null_percentile_below_min"}


def test_contaminated_series_flag_and_wording_on_a_candidate():
    v = sfa.candidate_definition(0.05, _vres(series="^VIX"))
    assert v.is_candidate is True and v.contaminated is True
    assert "confirmatory on the discovery window only" in v.contamination_note
    assert "partly known" in v.contamination_note
    clean = sfa.candidate_definition(0.05, _vres(series="GC=F"))
    assert clean.contaminated is False and clean.contamination_note == ""
    for s in ("WTI spot", "Brent spot"):
        assert sfa.candidate_definition(0.05, _vres(series=s)).contaminated is True


def test_candidate_constants_match_the_spec():
    assert sfa.CANDIDATE_HEADLINE_P_MAX == 0.10
    assert sfa.CANDIDATE_MIN_NULL_PERCENTILE == 0.80
    assert sfa.CANDIDATE_MIN_VEHICLE_N == 8


# ---------------------------------------------------------------------------
# 6. Put screen: the put-affordability screen
# ---------------------------------------------------------------------------

def test_bs_put_atm_zero_rate_matches_closed_form_and_textbook_values():
    # S = K, r = q = 0:  put = call = S * erf(vol sqrt(T) / (2 sqrt 2))
    closed = 100.0 * math.erf(0.2 * 1.0 / (2 * math.sqrt(2)))
    assert sfa.bs_put(100.0, 100.0, 0.2, 1.0) == pytest.approx(closed, abs=1e-12)
    assert sfa.bs_put(100.0, 100.0, 0.2, 1.0) == pytest.approx(7.965_567_455_405_804, abs=1e-9)
    # reference values computed with an independent tool (scipy.stats.norm)
    assert sfa.bs_put(100.0, 110.0, 0.3, 0.5) == pytest.approx(14.745_683_813_458_086, abs=1e-9)
    assert sfa.bs_call(100.0, 110.0, 0.3, 0.5) == pytest.approx(4.745_683_813_458_079, abs=1e-9)
    assert sfa.bs_put(50.0, 45.0, 0.25, 0.25) == pytest.approx(0.659_660_133_684_978_3, abs=1e-9)
    assert sfa.bs_put(1.0, 0.95, 0.32, 21 / 252) == pytest.approx(0.016_312_530_823_824_223,
                                                                  abs=1e-10)


def test_bs_put_call_parity_at_zero_rate_over_a_grid():
    rng = np.random.default_rng(1)
    for _ in range(200):
        S = rng.uniform(20, 200)
        K = S * rng.uniform(0.6, 1.4)
        vol = rng.uniform(0.05, 1.5)
        T = rng.uniform(0.02, 2.0)
        c = sfa.bs_call(S, K, vol, T)
        p = sfa.bs_put(S, K, vol, T)
        assert c - p == pytest.approx(S - K, abs=1e-9)


def test_bs_put_bounds_monotone_in_vol_and_zero_vol_limit():
    S, K, T = 100.0, 95.0, 0.25
    vols = np.linspace(0.05, 2.0, 40)
    prices = sfa.bs_put(S, K, vols, T)
    assert np.all(np.diff(prices) > 0)                    # vega positive
    assert np.all(prices >= max(K - S, 0.0) - 1e-12)
    assert np.all(prices <= K + 1e-12)                     # a put never exceeds the strike
    # zero volatility: the intrinsic value, no error
    assert sfa.bs_put(100.0, 110.0, 0.0, 0.5) == pytest.approx(10.0)
    assert sfa.bs_put(100.0, 90.0, 0.0, 0.5) == pytest.approx(0.0)


def test_bs_vectorises_over_spot_and_vol():
    S = np.array([90.0, 100.0, 110.0])
    out = sfa.bs_put(S, 100.0, np.array([0.2, 0.2, 0.2]), 1.0)
    assert out.shape == (3,)
    assert out[1] == pytest.approx(7.965_567_455_405_804, abs=1e-9)
    assert out[0] > out[1] > out[2]


def test_put_payoff_share_hand_computed_for_three_strikes():
    entry = np.array([100.0, 100.0, 100.0])
    expiry = np.array([90.0, 110.0, 80.0])
    # payoff as a share of the entry price: max(K% - S_H/S_0, 0)
    assert sfa.put_payoff_share(entry, expiry, 1.00).tolist() == pytest.approx([0.10, 0.0, 0.20])
    assert sfa.put_payoff_share(entry, expiry, 0.95).tolist() == pytest.approx([0.05, 0.0, 0.15])
    assert sfa.put_payoff_share(entry, expiry, 0.90).tolist() == pytest.approx([0.0, 0.0, 0.10])


def test_put_break_even_premium_is_the_mean_payoff_at_each_strike():
    entry = np.array([100.0, 100.0, 100.0])
    expiry = np.array([90.0, 110.0, 80.0])
    be = sfa.put_break_even_premium(entry, expiry)
    assert set(be) == {100, 95, 90}
    assert be[100] == pytest.approx(0.30 / 3)
    assert be[95] == pytest.approx(0.20 / 3)
    assert be[90] == pytest.approx(0.10 / 3)


def test_put_screen_constants_match_the_spec():
    assert sfa.PUT_STRIKE_PCTS == (100, 95, 90)
    assert sfa.PUT_K_GRID == (1.0, 1.25, 1.5, 2.0)
    assert sfa.TRADING_DAYS_PER_YEAR == 252
    assert sfa.RV_WINDOW == 21


def test_put_screen_single_episode_hand_computed_pnl_at_each_k():
    # one ATM put (K = 100 percent) held H = 252 sessions (T = 1 year), rv = 0.2,
    # expiry at 90 percent of entry: payoff 0.10 of entry.
    # premium(k) = erf(0.2 k / (2 sqrt 2)) for an ATM zero-rate put.
    r = sfa.put_screen(np.array([100.0]), np.array([90.0]), np.array([0.2]), H=252,
                       strike_pcts=(100,), ks=(1.0, 2.0))
    row = r.results[100]
    p1 = math.erf(0.2 * 1.0 / (2 * math.sqrt(2)))
    p2 = math.erf(0.2 * 2.0 / (2 * math.sqrt(2)))
    assert row["avg_premium"][1.0] == pytest.approx(p1, abs=1e-12)
    assert row["avg_premium"][2.0] == pytest.approx(p2, abs=1e-12)
    assert row["avg_pnl"][1.0] == pytest.approx(0.10 - p1, abs=1e-12)
    assert row["avg_pnl"][2.0] == pytest.approx(0.10 - p2, abs=1e-12)
    assert row["break_even_premium"] == pytest.approx(0.10)
    assert r.n_used == 1 and r.n_dropped == 0


def test_put_break_even_k_solves_the_hand_computed_case():
    # premium(k) = erf(0.2 k / (2 sqrt 2)) = 0.10  ->  k = erfinv(0.1) / (0.2 / (2 sqrt 2))
    r = sfa.put_screen(np.array([100.0]), np.array([90.0]), np.array([0.2]), H=252,
                       strike_pcts=(100,), ks=(1.0,))
    assert r.results[100]["break_even_k_status"] == "ok"
    assert r.results[100]["break_even_k"] == pytest.approx(1.256_613_468_550_740_4, abs=1e-7)


def test_put_break_even_k_is_where_average_pnl_crosses_zero_over_many_episodes():
    rng = np.random.default_rng(4)
    entry = np.full(40, 100.0)
    expiry = 100.0 * np.exp(rng.normal(-0.02, 0.09, 40))
    rv = rng.uniform(0.15, 0.35, 40)
    r = sfa.put_screen(entry, expiry, rv, H=21, strike_pcts=(100, 95, 90), ks=(1.0,))
    for m in (100, 95, 90):
        k = r.results[m]["break_even_k"]
        assert r.results[m]["break_even_k_status"] == "ok"
        pnl = np.mean(sfa.put_payoff_share(entry, expiry, m / 100.0)
                      - sfa.bs_put(1.0, m / 100.0, k * rv, 21 / 252))
        assert pnl == pytest.approx(0.0, abs=1e-9)


def test_put_break_even_k_not_attainable_cases():
    # every episode expires out of the money: no premium can be recovered
    never = sfa.put_screen(np.array([100.0, 100.0]), np.array([105.0, 120.0]),
                           np.array([0.2, 0.2]), H=21, strike_pcts=(100,), ks=(1.0,))
    assert never.results[100]["break_even_k_status"] == "never_profitable"
    assert math.isnan(never.results[100]["break_even_k"])
    # a collapse far beyond anything the bracket can price: break-even above the bracket
    huge = sfa.put_screen(np.array([100.0]), np.array([1.0]), np.array([0.05]), H=10,
                          strike_pcts=(100,), ks=(1.0,))
    assert huge.results[100]["break_even_k_status"] == "beyond_bracket"
    assert huge.results[100]["break_even_k"] == float("inf")
    assert sfa.BREAK_EVEN_K_BRACKET == (0.01, 50.0)


def test_put_screen_drops_and_counts_episodes_with_unusable_inputs():
    entry = np.array([100.0, 100.0, 100.0, np.nan, 100.0])
    expiry = np.array([90.0, 95.0, 80.0, 90.0, 0.0])
    rv = np.array([0.2, np.nan, 0.0, 0.2, 0.2])
    r = sfa.put_screen(entry, expiry, rv, H=21, strike_pcts=(100,), ks=(1.0,))
    # kept: episode 0 (ok) and episode 4 (expiry 0 is a valid total loss, rv fine)
    assert r.n_used == 2 and r.n_dropped == 3


def test_put_screen_versus_random_date_baseline_and_market_k():
    rng = np.random.default_rng(9)
    entry = np.full(30, 100.0)
    rv = np.full(30, 0.25)
    spike_exp = 100.0 * np.exp(rng.normal(-0.06, 0.06, 30))     # fades: puts pay
    base_exp = 100.0 * np.exp(rng.normal(0.0, 0.06, 30))        # random dates
    obs = sfa.put_screen(entry, spike_exp, rv, H=21, strike_pcts=(100,), ks=(1.0,))
    base = sfa.put_screen(entry, base_exp, rv, H=21, strike_pcts=(100,), ks=(1.0,))
    cmp_ = sfa.put_screen_vs_baseline(obs, base, market_k=[1.1, 1.2, 1.3, 1.4, 1.5])
    row = cmp_[100]
    assert row["observed_break_even_k"] == obs.results[100]["break_even_k"]
    assert row["baseline_break_even_k"] == base.results[100]["break_even_k"]
    assert row["margin_over_baseline"] == pytest.approx(
        row["observed_break_even_k"] - row["baseline_break_even_k"])
    assert row["exceeds_baseline"] is (row["margin_over_baseline"] > 0)
    assert row["margin_over_baseline"] > 0                       # the fade pays more
    assert row["margin_over_market_median"] == pytest.approx(
        row["observed_break_even_k"] - 1.3)
    assert row["margin_over_market_p90"] == pytest.approx(
        row["observed_break_even_k"] - np.percentile([1.1, 1.2, 1.3, 1.4, 1.5], 90))


def test_realized_vol_is_daily_log_return_sd_ddof1_times_sqrt_252():
    # 11 up moves and 10 down moves of one percent (log), 21 returns in the window
    a = 0.01
    r = np.array([a, -a] * 10 + [a])
    closes = 100.0 * np.exp(np.r_[0.0, np.cumsum(r)])          # 22 closes
    rv = sfa.realized_vol_annualized(closes, 21)
    var = a * a * (21 / 20) * (1 - (1 / 21) ** 2)
    assert rv == pytest.approx(math.sqrt(var) * math.sqrt(252), abs=1e-12)


def test_realized_vol_series_is_nan_until_a_full_window_and_ignores_the_future():
    rng = np.random.default_rng(2)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
    rv = sfa.realized_vol_series(p)
    assert np.isnan(rv[:21]).all() and np.isfinite(rv[21:]).all()
    # causal: truncating the series leaves earlier values unchanged
    rv_trunc = sfa.realized_vol_series(p[:80])
    assert np.array_equal(rv[:80], rv_trunc, equal_nan=True)
    # any invalid close inside the window blanks it, only that window
    q = p.copy()
    q[60] = np.nan
    rq = sfa.realized_vol_series(q)
    assert np.isnan(rq[60:82]).all() and np.isfinite(rq[59]) and np.isfinite(rq[82])


def test_put_screen_inputs_take_entry_expiry_and_rv_at_the_entry_close():
    rng = np.random.default_rng(6)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900)))
    ser = sfs.build_series(pd.Series(p, index=pd.bdate_range("2000-01-03", periods=900)),
                           name="SYN")
    t0 = np.array([300, 400, 500])
    inp = sfa.put_screen_inputs(ser, t0, ("E0", 10, "half", 1.5))   # TP and SL are ignored
    assert list(inp["entry_px"]) == pytest.approx(list(p[t0 + 1]))
    assert list(inp["expiry_px"]) == pytest.approx(list(p[t0 + 1 + 10]))
    rvs = sfa.realized_vol_series(p)
    assert list(inp["rv"]) == pytest.approx(list(rvs[t0 + 1]))
    assert list(inp["entry_dates"]) == list(ser.dates[t0 + 1])
    # E1 (up side, the side a put is bought on): the first close in (t0, t0 + 5] below
    # the previous close confirms, and the put is bought at the close of the NEXT session
    inp1 = sfa.put_screen_inputs(ser, t0, ("E1", 10, "none", None))
    want_entry = []
    for t in t0:
        for u in range(t + 1, t + 6):
            if p[u] < p[u - 1]:
                want_entry.append(u + 1)
                break
    assert len(inp1["entry_px"]) == len(want_entry) > 0
    assert list(inp1["entry_px"]) == pytest.approx([p[e] for e in want_entry])
    assert list(inp1["expiry_px"]) == pytest.approx([p[e + 10] for e in want_entry])


def test_market_k_is_implied_over_realized_with_iv_quoted_in_percent():
    assert sfa.market_k(20.0, 0.16) == pytest.approx(1.25)
    assert sfa.market_k(np.array([30.0, 15.0]), 0.15).tolist() == pytest.approx([2.0, 1.0])


def test_market_k_summary_median_p10_p90():
    s = sfa.market_k_summary([1, 2, 3, 4, 5])
    assert s["n"] == 5 and s["median"] == pytest.approx(3.0)
    assert s["p10"] == pytest.approx(1.4) and s["p90"] == pytest.approx(4.6)
    empty = sfa.market_k_summary([np.nan])
    assert empty["n"] == 0 and math.isnan(empty["median"])


def test_market_k_at_entries_uses_the_iv_close_on_the_entry_date_only():
    iv = pd.Series([20.0, 30.0, 40.0],
                   index=pd.DatetimeIndex(["2010-01-04", "2010-01-05", "2010-01-07"]))
    dates = pd.DatetimeIndex(["2010-01-05", "2010-01-06", "2010-01-07"])
    rv = np.array([0.20, 0.20, 0.40])
    k = sfa.market_k_at_entries(iv, dates, rv)
    assert k[0] == pytest.approx(1.5)
    assert np.isnan(k[1])                        # no IV close on the 6th: not invented
    assert k[2] == pytest.approx(1.0)


def test_iv_proxy_table_matches_the_spec_pairs():
    assert sfa.IV_PROXY_TABLE == {"^VIX": "SPY", "^OVX": "WTI spot",
                                  "^GVZ": "GC=F", "^VVIX": "^VIX"}
    assert sfa.STUDY_SERIES_IV_PROXY == {"^VIX": "^VVIX", "WTI spot": "^OVX",
                                         "GC=F": "^GVZ"}


# ---------------------------------------------------------------------------
# 7. Transfer line: the ABI-1 transfer line
# ---------------------------------------------------------------------------

from mining import account_relative as _ar
from mining import benchmark_relative as _br


def _idx(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _frame(idx, open_, close):
    return pd.DataFrame({"Open": np.asarray(open_, float), "Close": np.asarray(close, float)},
                        index=idx)


def _flat_spy(idx, level=100.0):
    return _frame(idx, [level] * len(idx), [level] * len(idx))


def test_transfer_constants_reuse_the_pipelines_own_numbers():
    assert sfa.OVERLAY_NOTIONAL == 0.25
    assert sfa.IN_SAMPLE_BAR_ANNUALIZED_SHARPE == 1.02
    assert sfa.N_DROP_WORST == 2
    assert sfa.ABSOLUTE_MAX_DRAWDOWN_CAP == 0.35
    assert sfa.ABSOLUTE_MAX_DRAWDOWN_CAP == _ar.ABSOLUTE_MAX_DRAWDOWN_CAP


# --- engine conventions: entry at the next open, exit at the flag bar's close ---

def _eng(open_, close_, signals, cell, sign, **kw):
    return sfa.engine_convention_trades(np.asarray(open_, float), np.asarray(close_, float),
                                        np.asarray(signals), cell, sign, **kw)


def test_engine_time_stop_enters_next_open_and_exits_at_close_h_sessions_later():
    o = [100, 100, 100, 100, 108, 110, 109, 108, 107, 106]
    c = [100, 100, 100, 104, 108, 110, 109, 108, 107, 106]
    r = _eng(o, c, [2], ("E0", 2, "none", None), -1)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert (t["signal_pos"], t["entry_pos"], t["exit_pos"]) == (2, 3, 5)
    assert t["reason"] == "time_stop" and t["entry_px"] == 100.0


def test_engine_stop_exits_at_the_flag_bars_own_close_not_the_next():
    o = [100] * 10
    c = [100, 100, 100, 99, 98, 96, 95, 95, 95, 95]
    sigma = np.full(10, 0.02)
    r = _eng(o, c, [2], ("E0", 5, "none", 1.5), -1, sigma=sigma)
    t = r.trades[0]
    # long: -ln(P_k / 100) >= 1.5 * 0.02 = 0.03 first at the close of 96 (bar 5, k = 2);
    # bar 4 (98) is only 2.02 percent down.  Stage 1 would fill at bar 6; the engine exits here.
    assert (t["entry_pos"], t["exit_pos"], t["reason"]) == (3, 5, "stop_loss")


def test_engine_stop_on_a_short_position_triggers_on_a_rise():
    o = [100] * 10
    c = [100, 100, 100, 101, 102, 104, 105, 105, 105, 105]
    sigma = np.full(10, 0.02)
    t = _eng(o, c, [2], ("E0", 5, "none", 1.5), +1, sigma=sigma).trades[0]
    # short: ln(P_k / 100) >= 0.03 first at 104 (bar 5)
    assert (t["exit_pos"], t["reason"]) == (5, "stop_loss")


def test_engine_take_profit_half_and_full_on_a_long():
    o = [100] * 10
    c = [100, 100, 100, 101, 104, 106, 107, 108, 109, 109]
    med = np.full(10, 110.0)                       # D = -(100 - 110) = 10 > 0
    half = _eng(o, c, [2], ("E0", 5, "half", None), -1, med=med).trades[0]
    assert (half["exit_pos"], half["reason"]) == (5, "take_profit")     # 106 >= 105
    full = _eng(o, c, [2], ("E0", 5, "full", None), -1, med=med).trades[0]
    assert (full["exit_pos"], full["reason"]) == (8, "time_stop")       # 110 never reached


def test_engine_take_profit_on_a_short_and_when_d_is_not_positive():
    o = [100] * 10
    c = [100, 100, 100, 99, 96, 94, 93, 93, 93, 93]
    short = _eng(o, c, [2], ("E0", 5, "half", None), +1, med=np.full(10, 90.0)).trades[0]
    # D = +(100 - 90) = 10; half flags when 100 - P_k >= 5, i.e. at 94 (bar 5)
    assert (short["exit_pos"], short["reason"]) == (5, "take_profit")
    # D <= 0 (median below the entry for a long): the TP is not applicable -> time stop
    c2 = [100, 100, 100, 101, 104, 106, 107, 108, 109, 109]
    na = _eng(o, c2, [2], ("E0", 5, "half", None), -1, med=np.full(10, 95.0)).trades[0]
    assert (na["exit_pos"], na["reason"]) == (8, "time_stop")


def test_engine_a_flag_on_the_time_stop_close_is_the_time_stop():
    o = [100] * 10
    c = [100, 100, 100, 101, 102, 96, 96, 96, 96, 96]
    # H = 2: the 96 close at k = 2 satisfies the stop but is the scheduled exit
    t = _eng(o, c, [2], ("E0", 2, "none", 1.5), -1, sigma=np.full(10, 0.02)).trades[0]
    assert (t["exit_pos"], t["reason"]) == (5, "time_stop")


def test_engine_e1_waits_for_the_confirming_close_and_enters_the_next_open():
    c = [100, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105, 106]
    o = [x + 0.5 for x in c]
    # long (sign -1): a confirming close is ABOVE the previous close; first is bar 4 (99 > 98)
    r = _eng(o, c, [1], ("E1", 3, "none", None), -1)
    t = r.trades[0]
    assert (t["entry_pos"], t["exit_pos"]) == (5, 8)
    assert t["entry_px"] == o[5]
    # no confirmation inside (t0, t0 + 5]: no trade, counted
    falling = [100 - i for i in range(12)]
    r2 = _eng(falling, falling, [1], ("E1", 3, "none", None), -1)
    assert r2.trades == [] and r2.n_skipped_no_entry == 1


def test_engine_holds_one_position_at_a_time_and_a_signal_on_the_exit_bar_is_usable():
    o = [100] * 12
    c = [100] * 12
    r = _eng(o, c, [2, 4, 5], ("E0", 2, "none", None), -1)
    # signal 2 -> enter bar 3, exit bar 5. signal 4 would enter bar 5 while still held: skipped.
    # signal 5 (the exit bar's own close) -> enter bar 6, exit bar 8.
    assert [(t["entry_pos"], t["exit_pos"]) for t in r.trades] == [(3, 5), (6, 8)]
    assert r.n_skipped_overlap == 1 and r.n_signals == 3


def test_engine_end_of_data_closes_at_the_last_close_and_last_bar_signal_has_no_entry():
    o = [100] * 10
    c = [100] * 10
    r = _eng(o, c, [7, 9], ("E0", 5, "none", None), -1)
    t = r.trades[0]
    assert (t["entry_pos"], t["exit_pos"], t["reason"]) == (8, 9, "end_of_data")
    assert r.n_skipped_no_entry == 1


def test_engine_undefined_median_or_sigma_skips_only_the_cells_that_use_them():
    o = [100] * 10
    c = [100, 100, 100, 101, 102, 103, 104, 105, 106, 107]
    med = np.full(10, np.nan)
    sig = np.full(10, np.nan)
    r = _eng(o, c, [2], ("E0", 3, "half", None), -1, med=med)
    assert r.trades == [] and r.n_skipped_undefined == 1
    r = _eng(o, c, [2], ("E0", 3, "none", 1.5), -1, sigma=sig)
    assert r.trades == [] and r.n_skipped_undefined == 1
    ok = _eng(o, c, [2], ("E0", 3, "none", None), -1)              # neither needed
    assert len(ok.trades) == 1
    with pytest.raises(ValueError):
        _eng(o, c, [2], ("E0", 3, "half", None), -1)                # median required, absent


# --- the account: SPY core + overlay, marked to market every day ------------

_TOY_O = [100, 100, 100, 100, 108, 110, 109, 108, 107, 106]
_TOY_C = [100, 100, 100, 104, 108, 110, 109, 108, 107, 106]


def _toy_line(position="long", spy=None, cost_bps=0.0, **kw):
    idx = _idx(10)
    spy = spy if spy is not None else _flat_spy(idx)
    veh = _frame(idx, _TOY_O, _TOY_C)
    return sfa.abi1_transfer_line(spy, veh, idx[[2]], ("E0", 2, "none", None), position,
                                  cost_bps=cost_bps, **kw), idx


def test_account_marks_to_market_daily_hand_computed_long_on_a_flat_spy():
    line, idx = _toy_line()
    a = line.account_daily
    # size = equity at the prior close = 1.0; shares = 0.25 / 100 = 0.0025; the entry bar earns
    # open-to-close (100 -> 104), then close-to-close (104 -> 108, 108 -> 110 exit bar)
    want = [0, 0, 0, 0.01, 0.01 / 1.01, 0.005 / 1.02, 0, 0, 0, 0]
    assert a.to_numpy() == pytest.approx(want, abs=1e-15)
    assert (1 + a).prod() == pytest.approx(1.025, abs=1e-12)
    assert line.core_daily.abs().max() == 0.0
    assert line.full.terminal_wealth_account == pytest.approx(1.025, abs=1e-12)
    assert line.full.terminal_wealth_spy == pytest.approx(1.0)


def test_account_short_position_loses_when_the_vehicle_rises():
    line, idx = _toy_line(position="short")
    want = [0, 0, 0, -0.01, -0.01 / 0.99, -0.005 / 0.98, 0, 0, 0, 0]
    assert line.account_daily.to_numpy() == pytest.approx(want, abs=1e-15)
    assert line.full.terminal_wealth_account == pytest.approx(0.975, abs=1e-12)


def test_account_trade_record_and_costs():
    line, idx = _toy_line(cost_bps=100.0)        # one percent round trip on the overlay notional
    t = line.trades.iloc[0]
    assert t["signal_date"] == idx[2] and t["entry_date"] == idx[3] and t["exit_date"] == idx[5]
    assert t["reason"] == "time_stop" and t["held"] == 2
    assert t["entry_px"] == 100.0 and t["exit_px"] == 110.0
    assert t["vehicle_ret"] == pytest.approx(0.10)
    # cost = size x notional x 1 percent = 0.0025, charged on the exit bar
    assert t["size"] == pytest.approx(1.0)
    assert t["net_ret"] == pytest.approx(0.025 - 0.0025)
    assert line.account_daily.iloc[5] == pytest.approx((0.005 - 0.0025) / 1.02, abs=1e-15)


def test_account_core_is_spy_and_overlay_sits_on_top_of_it():
    idx = _idx(10)
    spy_close = 100 * 1.01 ** np.arange(10)
    spy = _frame(idx, spy_close, spy_close)
    line, _ = _toy_line(spy=spy)
    core = np.r_[0.0, np.full(9, 0.01)]
    assert line.core_daily.to_numpy() == pytest.approx(core, abs=1e-12)
    a = line.account_daily.to_numpy()
    E2 = 1.01 ** 2
    E3 = E2 * (1 + a[3])
    # entry bar: core + 0.25 * (104 - 100) / 100 = 0.01 + 0.01
    assert a[3] == pytest.approx(0.01 + 0.01, abs=1e-12)
    # next bar: core + overlay / E3, overlay = shares * 4 with shares = 0.25 * E2 / 100
    assert a[4] == pytest.approx(0.01 + 0.25 * E2 * 4 / 100 / E3, abs=1e-12)
    # outside the trade the account is exactly SPY
    assert a[[0, 1, 2, 6, 7, 8, 9]] == pytest.approx(core[[0, 1, 2, 6, 7, 8, 9]], abs=1e-15)


def test_report_excess_sharpe_and_drawdown_are_computed_on_the_right_series():
    line, idx = _toy_line()
    a = line.account_daily.to_numpy()
    ex = np.log1p(a) - np.log1p(line.core_daily.to_numpy())
    want = ex.mean() / ex.std(ddof=1) * math.sqrt(252)
    assert line.full.excess_sharpe_annualized == pytest.approx(want, rel=1e-12)
    # and it is the pipeline's own annualization of the same daily series
    assert line.full.excess_sharpe_annualized == pytest.approx(_br.sharpe_annualized(ex),
                                                               rel=1e-12)
    assert line.excess_daily.to_numpy() == pytest.approx(ex, abs=1e-15)
    assert line.full.max_dd_account == 0.0 and line.full.max_dd_spy == 0.0
    assert line.full.n_sessions == 10 and line.full.n_trades == 1


def test_report_meets_bar_is_at_or_above_and_uses_the_supplied_bar():
    line, _ = _toy_line()
    s = line.full.excess_sharpe_annualized
    assert s > 1.02 and line.full.meets_in_sample_bar is True
    same, _ = _toy_line(in_sample_bar=s)
    assert same.full.meets_in_sample_bar is True and same.full.in_sample_bar == s
    above, _ = _toy_line(in_sample_bar=s + 1e-9)
    assert above.full.meets_in_sample_bar is False


def test_report_drawdown_against_spy_and_the_absolute_cap_hand_computed():
    idx = _idx(10)
    o = [100, 100, 100, 100, 96, 92, 94, 94, 94, 94]
    c = [100, 100, 100, 96, 92, 94, 94, 94, 94, 94]
    veh = _frame(idx, o, c)
    line = sfa.abi1_transfer_line(_flat_spy(idx), veh, idx[[2]], ("E0", 2, "none", None), "long")
    a = line.account_daily.to_numpy()
    # bar 3: -0.01 (E 0.99), bar 4: -0.01/0.99 (E 0.98), bar 5: +0.005/0.98 (E 0.985)
    assert a[3:6] == pytest.approx([-0.01, -0.01 / 0.99, 0.005 / 0.98], abs=1e-15)
    r = line.full
    assert r.max_dd_account == pytest.approx(-0.02, abs=1e-12)
    assert r.max_dd_spy == 0.0
    assert r.absolute_cap == -0.35
    assert r.dd_relative_passed is False              # the account draws down more than SPY
    assert r.dd_absolute_passed is True               # ... but nowhere near 35 percent
    assert r.dd_passed is False
    assert r.risk_refusals["drawdown"]["passed"] is False
    assert r.risk_refusals["drawdown"]["absolute_cap"] == -0.35


def test_report_absolute_cap_fails_a_deep_drawdown():
    idx = _idx(10)
    # a huge overlay on a collapsing vehicle: 100 -> 10 in three sessions
    o = [100, 100, 100, 100, 50, 25, 25, 25, 25, 25]
    c = [100, 100, 100, 50, 25, 10, 10, 10, 10, 10]
    veh = _frame(idx, o, c)
    line = sfa.abi1_transfer_line(_flat_spy(idx), veh, idx[[2]], ("E0", 2, "none", None), "long",
                                  overlay_notional=0.9)
    assert line.full.max_dd_account < -0.35
    assert line.full.dd_absolute_passed is False and line.full.dd_passed is False


def _three_trade_line(n_drop=2):
    """Flat SPY, three 2-session trades of a long vehicle: +10, -10 and -5 percent."""
    n = 30
    idx = _idx(n)
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    # trade A: entry bar 3, exit bar 5 at 110;  B: entry 10 -> exit 12 at 90;  C: 17 -> 19 at 95
    c[3], c[4], c[5] = 104.0, 108.0, 110.0
    c[10], c[11], c[12] = 96.0, 93.0, 90.0
    c[17], c[18], c[19] = 98.0, 96.0, 95.0
    veh = _frame(idx, o, c)
    return sfa.abi1_transfer_line(_flat_spy(idx), veh, idx[[2, 9, 16]],
                                  ("E0", 2, "none", None), "long", n_drop=n_drop), idx


def test_two_largest_loss_episodes_removed_hand_computed():
    line, idx = _three_trade_line()
    # full: A +0.25*0.10 -> 1.025 ; B on 1.025: -0.25*1.025*0.10 -> 0.999375 ;
    #       C on 0.999375: -0.25*0.999375*0.05 -> 0.986883...
    E = 1.0
    E *= 1 + 0.25 * 0.10
    E *= 1 - 0.25 * 0.10
    E *= 1 - 0.25 * 0.05
    assert line.full.terminal_wealth_account == pytest.approx(E, abs=1e-12)
    assert line.full.n_trades == 3
    # the two worst by net return on equity are B (-2.5 percent) and C (-1.25 percent)
    worst = set(pd.DatetimeIndex(line.dropped_trades["entry_date"]))
    assert worst == {idx[10], idx[17]}
    # with them removed only A remains: 1.025
    assert line.ex_worst.n_trades == 1
    assert line.ex_worst.terminal_wealth_account == pytest.approx(1.025, abs=1e-12)
    assert line.ex_worst.excess_sharpe_annualized > line.full.excess_sharpe_annualized


def test_removing_fewer_or_more_worst_episodes():
    one, idx = _three_trade_line(n_drop=1)
    assert set(pd.DatetimeIndex(one.dropped_trades["entry_date"])) == {idx[10]}
    assert one.ex_worst.n_trades == 2
    all_, _ = _three_trade_line(n_drop=3)
    assert all_.ex_worst.n_trades == 0
    assert all_.ex_worst.terminal_wealth_account == pytest.approx(1.0)
    assert all_.ex_worst.excess_sharpe_annualized == 0.0


def test_no_signals_means_the_account_is_spy_and_the_excess_is_exactly_zero():
    idx = _idx(300)
    rng = np.random.default_rng(3)
    spy_close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    spy = _frame(idx, spy_close, spy_close)
    veh = _frame(idx, spy_close, spy_close)
    line = sfa.abi1_transfer_line(spy, veh, idx[:0], ("E0", 21, "none", None), "long")
    assert line.account_daily.equals(line.core_daily)
    assert (line.excess_daily == 0).all() and line.full.excess_sharpe_annualized == 0.0
    assert line.full.n_trades == 0 and line.full.meets_in_sample_bar is False


def test_transfer_line_with_tp_and_stop_on_random_data_is_finite_and_consistent():
    n = 900
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(12)
    spy_close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    v_close = 50 * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
    v_open = v_close * np.exp(rng.normal(0, 0.003, n))
    spy = _frame(idx, spy_close, spy_close)
    veh = _frame(idx, v_open, v_close)
    sig = idx[np.arange(300, 800, 45)]
    for cell in (("E0", 21, "half", 1.5), ("E1", 42, "full", 3.0), sfs.PRIMARY_CELL):
        line = sfa.abi1_transfer_line(spy, veh, sig, cell, "long")
        assert np.isfinite(line.account_daily).all()
        assert np.isfinite(line.full.excess_sharpe_annualized)
        assert line.full.n_trades == len(line.trades) > 0
        # the account is SPY outside the sessions a trade is open
        open_days = np.zeros(n, bool)
        for _, t in line.trades.iterrows():
            open_days[idx.get_loc(t["entry_date"]):idx.get_loc(t["exit_date"]) + 1] = True
        same = ~open_days
        assert line.account_daily.to_numpy()[same] == pytest.approx(
            line.core_daily.to_numpy()[same], abs=1e-15)
        assert line.full.dsr_bar_annualized > 0


def test_dsr_bar_for_a_session_count_matches_the_pipelines_dsr():
    from mining import constants, dsr
    n = 5000
    bar = sfa.dsr_bar_annualized(n)
    assert bar == pytest.approx(1.0188, abs=2e-3)       # "about 1.02" in constants.py
    sr = bar / math.sqrt(252)
    v = dsr.deflated_sharpe_ratio(sr=sr, n=n, skew=0.0, kurtosis=3.0,
                                  n_trials=constants.ABI1_N_TRIALS,
                                  sr_variance=constants.DAILY_SR_VARIANCE)
    assert v == pytest.approx(constants.DSR_PASS_THRESHOLD, abs=1e-6)
    assert sfa.dsr_bar_annualized(756) == pytest.approx(1.6018, abs=2e-3)


def test_transfer_line_input_validation_fails_closed():
    idx = _idx(10)
    veh = _frame(idx, _TOY_O, _TOY_C)
    spy = _flat_spy(idx)
    cell = ("E0", 2, "none", None)
    # a signal date that is not a vehicle session
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(spy, veh, pd.DatetimeIndex(["2020-01-04"]), cell, "long")
    # an SPY calendar that disagrees with the vehicle's inside the span
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(spy.drop(idx[4]), veh, idx[[2]], cell, "long")
    # non-positive or missing prices
    bad = veh.copy()
    bad.loc[idx[5], "Close"] = 0.0
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(spy, bad, idx[[2]], cell, "long")
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(spy, veh.drop(columns=["Open"]), idx[[2]], cell, "long")
    # unsorted / duplicated index
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(spy, veh.iloc[::-1], idx[[2]], cell, "long")
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(spy, veh, idx[[2]], cell, "flat")
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(spy, veh, idx[[2]], cell, "long", overlay_notional=0.0)


def test_transfer_line_refuses_data_after_the_cutoff():
    idx = pd.bdate_range("2023-06-20", periods=15)            # runs past 2023-06-30
    veh = _frame(idx, [100] * 15, [100] * 15)
    with pytest.raises(sfs.SealedWindowError):
        sfa.abi1_transfer_line(_flat_spy(idx), veh, idx[[2]], ("E0", 2, "none", None), "long")


def test_transfer_line_never_touches_the_abi1_evaluator_or_the_ledger():
    import inspect
    src = inspect.getsource(sfa)
    for banned in ("account_relative.evaluate(", "evaluate_family", "record_outcome",
                   "ledger.append", "abi1_driver"):
        assert banned not in src, banned
    # the pure helpers the line reads are used
    assert "risk_refusals" in src and "log_excess" in src


# ---------------------------------------------------------------------------
# 8. Scope helpers the driver uses: which pairs get the vehicle test, and the
#    put screen for one cell (observed vs random-date baseline vs market k)
# ---------------------------------------------------------------------------

def _scope_frame():
    rows = [
        _row(0.90, 0.5, series="^OVX", side="up_fade"),        # no tradable vehicle
        _row(0.80, 0.5, series="GC=F", side="up_fade", trigger="pct756", H=42, sl="1.5"),
        _row(0.70, 0.5, series="WTI spot", side="up_fade", trigger="z20@2.0"),
        _row(0.65, 0.9, series="WTI spot", side="up_fade", trigger="z20@2.5", H=10),
        _row(0.60, 0.5, series="^VIX", side="down_fade", trigger="pctexp", entry="E1"),
        _row(0.55, 0.5, series="^MOVE", side="down_fade"),     # no tradable vehicle
        _row(0.50, 0.5, series="HG=F", side="up_fade"),
        _row(0.40, 0.5, series="DX-Y.NYB", side="down_fade"),
    ]
    return pd.DataFrame(rows)


def test_vehicle_scope_is_the_tstar_pair_plus_the_top_three_vehicle_eligible_pairs():
    scope = sfa.vehicle_test_scope(_scope_frame(), tstar_group=("^VIX", "up_fade", "z20@2.5"))
    got = [(e["series"], e["side"], e["role"]) for e in scope]
    assert got == [("^VIX", "up_fade", "tstar"),
                   ("GC=F", "up_fade", "plateau_top3"),
                   ("WTI spot", "up_fade", "plateau_top3"),
                   ("^VIX", "down_fade", "plateau_top3")]
    # T* pair runs the primary cell on its own group; the others run their top plateau cell
    assert scope[0]["cell"] == sfs.PRIMARY_CELL and scope[0]["trigger"] == "z20@2.5"
    assert scope[1]["cell"] == ("E0", 42, "none", 1.5) and scope[1]["trigger"] == "pct756"
    assert scope[2]["cell"] == ("E0", 21, "none", None) and scope[2]["trigger"] == "z20@2.0"
    assert scope[3]["cell"] == ("E1", 21, "none", None) and scope[3]["trigger"] == "pctexp"
    assert (scope[0]["vehicle"], scope[0]["position"]) == ("SVXY", "long")
    assert (scope[2]["vehicle"], scope[2]["position"]) == ("USO", "short")


def test_vehicle_scope_uses_each_pairs_best_plateau_cell_only_once():
    scope = sfa.vehicle_test_scope(_scope_frame(), tstar_group=("HG=F", "down_fade", "pct756"))
    wti = [e for e in scope if e["series"] == "WTI spot"]
    assert len(wti) == 1 and wti[0]["trigger"] == "z20@2.0"      # 0.70, not the 0.65 cell


def test_vehicle_scope_when_the_tstar_pair_is_also_in_the_top_three_is_not_duplicated():
    scope = sfa.vehicle_test_scope(_scope_frame(), tstar_group=("GC=F", "up_fade", "pct756"))
    assert [(e["series"], e["side"]) for e in scope] == [
        ("GC=F", "up_fade"), ("WTI spot", "up_fade"), ("^VIX", "down_fade")]
    assert scope[0]["role"] == "tstar+plateau_top3"
    assert scope[0]["cell"] == sfs.PRIMARY_CELL           # the T* pair's cell is the primary one


def test_vehicle_scope_without_a_tstar_and_with_fewer_eligible_pairs():
    scope = sfa.vehicle_test_scope(_scope_frame(), tstar_group=None)
    assert len(scope) == 3 and all(e["role"] == "plateau_top3" for e in scope)
    few = sfa.vehicle_test_scope(_scope_frame().iloc[:2], tstar_group=None)
    assert [e["series"] for e in few] == ["GC=F"]         # only one vehicle-eligible pair exists


def test_vehicle_scope_tstar_pair_without_a_vehicle_is_reported_not_run():
    scope = sfa.vehicle_test_scope(_scope_frame(), tstar_group=("^OVX", "up_fade", "pct756"))
    assert scope[0]["role"] == "tstar" and scope[0]["vehicle"] is None
    assert scope[0]["run"] is False and all(e["run"] for e in scope[1:])


def test_put_screen_for_cell_composes_inputs_baseline_and_market_k():
    rng = np.random.default_rng(6)
    n = 900
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ser = sfs.build_series(pd.Series(p, index=pd.bdate_range("2000-01-03", periods=n)),
                           name="SYN")
    t0 = np.array([300, 350, 400, 450, 500, 550])
    base_t0 = np.array([310, 360, 410, 470, 520, 560, 600, 650])
    iv = pd.Series(20.0, index=ser.dates)                       # an index quoted in percent
    out = sfa.put_screen_for_cell(ser, t0, ("E0", 10, "half", 1.5), baseline_t0=base_t0,
                                  iv_close=iv, ks=(1.0, 2.0))
    inp = sfa.put_screen_inputs(ser, t0, ("E0", 10, "half", 1.5))
    ref = sfa.put_screen(inp["entry_px"], inp["expiry_px"], inp["rv"], H=10, ks=(1.0, 2.0))
    for pct in (100, 95, 90):
        assert out["observed"].results[pct]["break_even_k"] == pytest.approx(
            ref.results[pct]["break_even_k"], nan_ok=True)
        assert out["comparison"][pct]["baseline_break_even_k"] == pytest.approx(
            out["baseline"].results[pct]["break_even_k"], nan_ok=True)
    assert out["baseline"].n_used == len(base_t0)
    mk = sfa.market_k(20.0, inp["rv"])
    assert out["market_k_summary"]["median"] == pytest.approx(np.median(mk))
    assert list(out["market_k"]) == pytest.approx(list(mk))
    # no baseline and no IV series: those parts are simply absent
    bare = sfa.put_screen_for_cell(ser, t0, ("E0", 10, "half", 1.5))
    assert bare["baseline"] is None and bare["comparison"] is None
    assert bare["market_k"] is None and bare["market_k_summary"] is None


# ---------------------------------------------------------------------------
# 9. Regression tests (each pins a behaviour the broader tests above would
#    not catch if it broke)
# ---------------------------------------------------------------------------

def test_stability_strictness_binds_on_the_two_largest_removal():
    # after removing the two largest (5 and 3) the remaining mean is exactly zero
    x = [5, 3, 1, 1, 0, 0, -1, -1]
    r = sfa.stability_flags(x, _dates(8))
    assert r.S == pytest.approx(1.0) and r.S_drop1 > 0
    assert r.S_drop2 == 0.0
    assert r.a_pass is False and r.passed is False


def test_put_payoff_share_is_normalised_by_the_entry_price_not_a_constant():
    entry = np.array([50.0, 200.0])
    expiry = np.array([45.0, 190.0])
    got = sfa.put_payoff_share(entry, expiry, 1.0)
    assert got.tolist() == pytest.approx([0.10, 0.05])
    be = sfa.put_break_even_premium(entry, expiry)
    assert be[100] == pytest.approx(0.075) and be[95] == pytest.approx(0.025)
    assert be[90] == pytest.approx(0.0)


def test_vehicle_window_starts_at_the_252nd_session_and_ends_at_the_last():
    V, veh, anchors = _vehicle_with_effect(0.0)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=5, seed=1)
    assert res.window_start == V[251] and res.window_end == V[-1]


def test_vehicle_undefined_median_is_its_own_stratum_and_matched_exactly():
    # a non-positive close at session 300 blanks the median window of the next 252
    # sessions but not sigma21 after 64 of them: valid trades whose D is undefined
    n = 1500
    V = pd.bdate_range("2006-01-02", periods=n)
    closes = _vehicle_closes(n, 21)
    closes[300] = -1.0
    veh = _vehicle_from(V, closes)
    anchors = [400, 450, 500, 700, 800, 900, 1000, 1100, 1200]
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=25, seed=3,
                                collect_draws=True)
    assert res.n == 9 and res.n_undef_d == 3
    assert res.n_pos_d + res.n_nonpos_d == 6
    ctx = veh.context()
    assert int(np.isnan(ctx.med[[400, 450, 500]]).sum()) == 3
    assert res.null_reps_used > 0
    for row in res.null_anchors:
        assert int(np.isnan(ctx.med[row]).sum()) == 3          # undefined-D anchors matched
    # the draws are valid trades: none sits inside the sigma-touched sessions after 300
    assert not ((res.null_anchors >= 300) & (res.null_anchors < 364)).any()


def test_null_cells_for_stability_keeps_only_episodes_that_trade_in_the_cell():
    rng = np.random.default_rng(3)
    n = 4000
    p = 100 * np.exp(np.cumsum(rng.standard_t(4.0, size=n) * 0.01 / math.sqrt(2)))
    ser = sfs.build_series(pd.Series(p, index=pd.bdate_range("2000-01-03", periods=n)),
                           name="SYN")
    e1 = sfa.null_cells_for_stability(ser, "z20@2.0", "up_fade", ("E1", 21, "none", None),
                                      B=40, seed=1)
    prim = sfa.null_cells_for_stability(ser, "z20@2.0", "up_fade", sfs.PRIMARY_CELL,
                                        B=40, seed=1)
    n_prim = len(prim[0][0])
    assert all(np.isfinite(x).all() for x, _ in e1)
    assert all(len(x) <= n_prim for x, _ in e1)
    # an entry rule that needs a confirming close loses some pseudo-episodes
    assert min(len(x) for x, _ in e1) < n_prim


def test_transfer_entry_bar_earns_open_to_close_when_the_open_gaps():
    idx = _idx(10)
    o = [100, 100, 100, 98, 108, 110, 109, 108, 107, 106]      # bar 3 opens 2 percent below
    c = [100, 100, 100, 104, 108, 110, 109, 108, 107, 106]     # the prior close
    line = sfa.abi1_transfer_line(_flat_spy(idx), _frame(idx, o, c), idx[[2]],
                                  ("E0", 2, "none", None), "long")
    assert line.trades.iloc[0]["entry_px"] == 98.0
    # the entry bar earns 104 - 98 (open to close), not 104 - 100 (close to close)
    assert line.account_daily.iloc[3] == pytest.approx(0.25 * (104 - 98) / 98, abs=1e-15)
    assert line.trades.iloc[0]["vehicle_ret"] == pytest.approx(110 / 98 - 1)


def test_transfer_line_cutoff_is_enforced_on_each_frame_separately():
    n = 15
    idx = pd.bdate_range("2023-06-20", periods=n)               # runs past 2023-06-30
    sealed = idx[idx <= sfs.DISCOVERY_CUTOFF]
    cell = ("E0", 2, "none", None)
    # vehicle runs past the cutoff, SPY is sealed
    with pytest.raises(sfs.SealedWindowError):
        sfa.abi1_transfer_line(_flat_spy(sealed), _frame(idx, [100] * n, [100] * n),
                               idx[[2]], cell, "long")
    # SPY runs past the cutoff, the vehicle is sealed
    with pytest.raises(sfs.SealedWindowError):
        sfa.abi1_transfer_line(_flat_spy(idx), _frame(sealed, [100] * len(sealed),
                                                      [100] * len(sealed)),
                               sealed[[2]], cell, "long")


def test_transfer_line_refuses_a_calendar_mismatch_even_when_reindexing_would_work():
    idx = _idx(10)
    veh = _frame(idx, _TOY_O, _TOY_C).drop(idx[4])      # the vehicle lacks a session SPY has
    with pytest.raises(ValueError):
        sfa.abi1_transfer_line(_flat_spy(idx), veh, idx[[2]], ("E0", 2, "none", None), "long")


def test_two_trigger_anchors_mapping_to_one_vehicle_session_are_deduplicated_and_counted():
    T, V, _ = _calendars()
    veh = _vehicle_from(V, _vehicle_closes(len(V), 5))
    # trigger 700 and 701 are both vehicle holidays: both map to V position 398
    res = sfa.vehicle_cell_test(veh, T[[700, 701, 900]], sfs.PRIMARY_CELL, "long", B=10, seed=1)
    assert res.dropped_duplicate == 1 and res.n_anchors_kept == 2 and res.n == 2
    assert (res.dropped_no_session + res.dropped_past_end + res.dropped_short_history
            + res.dropped_censored + res.dropped_duplicate + res.n_anchors_kept) == 3


def test_vehicle_hit_rate_counts_only_strictly_positive_pnl():
    n = 1100
    V = pd.bdate_range("2006-01-02", periods=n)
    closes = _vehicle_closes(n, 5)
    closes[401 + 21] = closes[401]                 # anchor 400: exit close equals entry close
    closes[601 + 21] = closes[601] * 1.05          # anchor 600: a gain
    veh = _vehicle_from(V, closes)
    res = sfa.vehicle_cell_test(veh, V[[400, 600]], sfs.PRIMARY_CELL, "long", B=10, seed=1)
    assert res.n == 2
    assert list(res.trades["pnl"]) == pytest.approx([0.0, 0.05])
    assert res.hit_rate == pytest.approx(0.5)


def test_put_screen_comparison_direction_and_missing_baseline():
    rng = np.random.default_rng(9)
    entry = np.full(30, 100.0)
    rv = np.full(30, 0.25)
    fade = sfa.put_screen(entry, 100.0 * np.exp(rng.normal(-0.06, 0.06, 30)), rv, H=21,
                          strike_pcts=(100,), ks=(1.0,))
    rand = sfa.put_screen(entry, 100.0 * np.exp(rng.normal(0.0, 0.06, 30)), rv, H=21,
                          strike_pcts=(100,), ks=(1.0,))
    fwd = sfa.put_screen_vs_baseline(fade, rand)[100]
    rev = sfa.put_screen_vs_baseline(rand, fade)[100]
    assert fwd["exceeds_baseline"] is True and fwd["margin_over_baseline"] > 0
    assert rev["exceeds_baseline"] is False
    assert rev["margin_over_baseline"] == pytest.approx(-fwd["margin_over_baseline"])
    none = sfa.put_screen_vs_baseline(fade, None)[100]
    assert none["baseline_status"] == "no_baseline" and none["exceeds_baseline"] is False
    assert math.isnan(none["margin_over_baseline"])
    # a not-attainable baseline (NaN) never counts as exceeded either
    never = sfa.put_screen(np.array([100.0, 100.0]), np.array([105.0, 120.0]),
                           np.array([0.2, 0.2]), H=21, strike_pcts=(100,), ks=(1.0,))
    nan_cmp = sfa.put_screen_vs_baseline(fade, never)[100]
    assert nan_cmp["baseline_status"] == "never_profitable"
    assert nan_cmp["exceeds_baseline"] is False and math.isnan(nan_cmp["margin_over_baseline"])


def test_transfer_trade_record_position_return_carries_the_sign_of_the_position():
    long_line, _ = _toy_line("long")
    short_line, _ = _toy_line("short")
    assert long_line.trades.iloc[0]["position_ret"] == pytest.approx(0.10)
    assert short_line.trades.iloc[0]["position_ret"] == pytest.approx(-0.10)
    assert short_line.trades.iloc[0]["vehicle_ret"] == pytest.approx(0.10)     # unsigned
    assert short_line.trades.iloc[0]["net_ret"] == pytest.approx(-0.025)


# ---------------------------------------------------------------------------
# 10. Further edge cases
# ---------------------------------------------------------------------------

# --- 1. engine_convention_trades rejects signal positions outside the path ---

def test_engine_rejects_signal_positions_outside_the_price_path():
    o = c = [100.0] * 10
    for cell in (("E0", 2, "none", None), ("E1", 2, "none", None)):
        for bad in ([-1], [10], [11], [2, -3], [0, 10], [-1, 5], [5, 10 ** 6]):
            with pytest.raises(ValueError):
                _eng(o, c, bad, cell, -1)


def test_engine_e1_negative_signal_never_reads_the_final_close():
    # a signal at -1 would compare c[0] with c[-1], the LAST (future) close, and could
    # confirm at u = 0; the rejection must happen before any signal is evaluated
    c = [100.0, 99, 98, 97, 96, 95, 94, 93, 92, 200.0]        # c[-1] is far above c[0]
    with pytest.raises(ValueError):
        _eng(c, c, [-1], ("E1", 2, "none", None), -1)


def test_engine_accepts_the_first_and_last_positions_and_an_empty_signal_list():
    o = c = [100.0] * 10
    r = _eng(o, c, [0, 9], ("E0", 2, "none", None), -1)
    assert r.n_signals == 2 and [t["signal_pos"] for t in r.trades] == [0]
    assert r.n_skipped_no_entry == 1                    # the last bar has no next open
    e1 = _eng(o, c, [0], ("E1", 2, "none", None), -1)   # position 0 reads c[0], c[1] only
    assert e1.n_signals == 1
    empty = _eng(o, c, [], ("E0", 2, "none", None), -1)
    assert empty.trades == [] and empty.n_signals == 0


# --- 2. put_screen_vs_baseline: only finite break-even values are compared ---

def _ps(k, status="ok", pct=100):
    return sfa.PutScreen(H=21, strike_pcts=(pct,), ks=(1.0,), n_used=5, n_dropped=0,
                         results={pct: {"break_even_premium": 0.01, "avg_premium": {},
                                        "avg_pnl": {}, "break_even_k": k,
                                        "break_even_k_status": status}})


def test_put_comparison_non_finite_break_even_never_exceeds_the_baseline():
    inf, nan = float("inf"), float("nan")
    cases = [(inf, 1.2), (1.2, inf), (inf, inf), (nan, 1.2), (1.2, nan), (nan, nan),
             (-inf, 1.2)]
    for o, b in cases:
        row = sfa.put_screen_vs_baseline(_ps(o), _ps(b))[100]
        assert row["exceeds_baseline"] is False, (o, b)
        assert math.isnan(row["margin_over_baseline"]), (o, b)
    ok = sfa.put_screen_vs_baseline(_ps(1.9), _ps(1.2))[100]
    assert ok["exceeds_baseline"] is True and ok["margin_over_baseline"] == pytest.approx(0.7)
    below = sfa.put_screen_vs_baseline(_ps(1.0), _ps(1.2))[100]
    assert below["exceeds_baseline"] is False and below["margin_over_baseline"] < 0


def test_put_comparison_market_margins_are_also_nan_for_a_non_finite_observed_k():
    row = sfa.put_screen_vs_baseline(_ps(float("inf"), "beyond_bracket"), _ps(1.2),
                                     market_k=[1.1, 1.3])[100]
    assert row["observed_status"] == "beyond_bracket"
    assert math.isnan(row["margin_over_market_median"])
    assert math.isnan(row["margin_over_market_p90"])
    fine = sfa.put_screen_vs_baseline(_ps(1.9), _ps(1.2), market_k=[1.1, 1.3])[100]
    assert fine["margin_over_market_median"] == pytest.approx(0.7)


# --- 3. Black-Scholes with a NaN volatility is NaN, not the intrinsic value ---

def test_bs_nan_volatility_or_time_is_nan_not_intrinsic():
    nan = float("nan")
    assert math.isnan(sfa.bs_put(1.0, 0.95, nan, 0.08))
    assert math.isnan(sfa.bs_call(1.0, 0.95, nan, 0.08))
    assert math.isnan(sfa.bs_put(1.0, 0.95, 0.2, nan))
    assert math.isnan(sfa.bs_call(1.0, 0.95, 0.2, nan))
    out = sfa.bs_put(1.0, 1.05, np.array([0.2, nan, 0.0]), 0.08)
    assert np.isfinite(out[0]) and math.isnan(out[1]) and out[2] == pytest.approx(0.05)
    # zero volatility (exactly) keeps the intrinsic value
    assert sfa.bs_put(100.0, 110.0, 0.0, 0.5) == pytest.approx(10.0)
    assert sfa.bs_call(100.0, 90.0, 0.0, 0.5) == pytest.approx(10.0)
    assert sfa.bs_put(100.0, 110.0, 0.2, 0.0) == pytest.approx(10.0)


def test_put_screen_is_unchanged_by_the_nan_rule():
    entry = np.array([100.0, 100.0, 100.0, 100.0])
    expiry = np.array([90.0, 96.0, 110.0, 85.0])
    rv = np.array([0.2, 0.25, 0.3, 0.22])
    full = sfa.put_screen(entry, expiry, rv, H=21)
    with_nan = sfa.put_screen(np.r_[entry, 100.0], np.r_[expiry, 90.0],
                              np.r_[rv, np.nan], H=21)
    assert with_nan.n_dropped == 1 and with_nan.n_used == 4
    for pct in full.strike_pcts:
        a, b = full.results[pct], with_nan.results[pct]
        assert a["break_even_k"] == b["break_even_k"]
        assert a["avg_pnl"] == b["avg_pnl"] and a["avg_premium"] == b["avg_premium"]
    assert all(np.isfinite(v) for r in full.results.values() for v in r["avg_pnl"].values())


# --- 4. confirmation-window, sigma-unit and tie-order pins ---

def test_engine_e1_confirmation_window_is_t0_plus_1_through_t0_plus_5():
    # long (sign -1): a confirming close is ABOVE the previous close
    on_edge = [100, 100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 102]     # first rise at u = 6
    r = _eng(on_edge, on_edge, [1], ("E1", 3, "none", None), -1)
    assert [(t["entry_pos"], t["exit_pos"]) for t in r.trades] == [(7, 10)]   # u = t0 + 5
    just_outside = [100, 100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100]  # first rise at u = 7
    r2 = _eng(just_outside, just_outside, [1], ("E1", 3, "none", None), -1)
    assert r2.trades == [] and r2.n_skipped_no_entry == 1                     # u = t0 + 6


def test_engine_e1_short_side_window_boundary_and_direction():
    # short (sign +1): a confirming close is BELOW the previous close
    on_edge = [100, 100, 101, 102, 103, 104, 103, 104, 105, 106, 107, 108]   # first fall at u = 6
    r = _eng(on_edge, on_edge, [1], ("E1", 3, "none", None), +1)
    assert [(t["entry_pos"], t["exit_pos"]) for t in r.trades] == [(7, 10)]
    outside = [100, 100, 101, 102, 103, 104, 105, 104, 105, 106, 107, 108]   # first fall at u = 7
    assert _eng(outside, outside, [1], ("E1", 3, "none", None), +1).trades == []


def test_vehicle_excess_in_sigma_units_is_the_difference_of_mean_winsorised_x():
    V, veh, anchors = _vehicle_with_effect(+0.004)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=40, seed=5,
                                collect_draws=True)
    assert res.mean_x == pytest.approx(res.trades["x"].mean())
    means = []
    for row in res.null_anchors:                          # independent recomputation
        r = sfs.evaluate_variants(veh, row, -1, [sfs.PRIMARY_CELL])
        assert np.isfinite(r.x[0]).all()
        means.append(float(r.x[0].mean()))
    assert res.null_mean_x == pytest.approx(np.mean(means), abs=1e-12)
    assert res.excess_sigma == pytest.approx(res.mean_x - np.mean(means), abs=1e-12)
    assert res.excess_sigma > 0
    # x is P&L in units of the vehicle's own sigma21, so it differs from the pp figure
    assert res.excess_sigma != pytest.approx(res.excess_pp)
    # and the sigma-unit percentile is the mid-rank of mean x among the null x means
    assert res.null_percentile_x == pytest.approx(
        ((np.array(means) < res.mean_x).sum() + 0.5 * (np.array(means) == res.mean_x).sum())
        / len(means))


def test_transfer_relative_drawdown_equal_to_spy_passes_with_a_zero_overlay_account():
    idx = _idx(300)
    rng = np.random.default_rng(7)
    spy_close = 100 * np.exp(np.cumsum(rng.normal(-0.0002, 0.006, 300)))    # about -27 percent
    spy = _frame(idx, spy_close, spy_close)
    line = sfa.abi1_transfer_line(spy, _frame(idx, spy_close, spy_close), idx[:0],
                                  ("E0", 21, "none", None), "long")
    r = line.full
    assert r.max_dd_account == r.max_dd_spy and -0.35 < r.max_dd_spy < 0
    assert r.dd_relative_passed is True and r.dd_absolute_passed is True
    assert r.dd_passed is True and r.risk_refusals["drawdown"]["passed"] is True
    # the same holds for the account with every trade removed
    assert line.ex_worst.dd_relative_passed is True


def test_rank_tie_order_puts_sl_before_tp():
    # same plateau, same S, same H: SL 'none' beats SL '1.5' even though its TP is worse
    a = _row(0.4, 0.5, sl="1.5", tp="none", trigger="a")
    b = _row(0.4, 0.5, sl="none", tp="half", trigger="b")
    r = sfa.rank_cells(pd.DataFrame([a, b]))
    assert list(r["trigger"]) == ["b", "a"]
    r2 = sfa.rank_cells(pd.DataFrame([b, a]))
    assert list(r2["trigger"]) == ["b", "a"]


def test_stability_both_halves_negative_is_sign_agreement_but_the_cell_still_fails_a():
    x = [-1, -2, -1, -3, -1, -1, -2, -1]
    r = sfa.stability_flags(x, _dates(8))
    assert r.first_half_mean < 0 and r.second_half_mean < 0
    assert r.b_applicable is True and r.b_pass is True
    assert r.a_pass is False and r.passed is False


def test_null_stability_pass_rate_positive_means_strictly_above_zero():
    zero = ([1, -1, 1, -1, 1, -1, 1, -1], _dates(8))         # S is exactly 0.0
    good = ([3, 1, 1, 1, 1, 1, 1, -1], _dates(8))            # passes
    assert sfa.stability_flags(*zero).S == 0.0
    r = sfa.null_stability_pass_rate([zero, good])
    assert r["n_cells"] == 2 and r["n_pass"] == 1
    assert r["n_positive"] == 1                               # the S = 0 cell is not positive
    assert r["pass_rate_given_positive"] == pytest.approx(1.0)


def test_plateau_accepts_nan_none_and_numeric_sl_labels_like_the_string_labels():
    base = sfa.plateau_scores(_cell_frame(_code_grid()))
    numeric = {"none": float("nan"), "1.5": 1.5, "3.0": 3.0}
    f_nan = _cell_frame(_code_grid())
    f_nan["sl"] = f_nan["sl"].map(numeric)
    assert f_nan["sl"].isna().any()
    got = sfa.plateau_scores(f_nan)
    assert got["plateau_score"].equals(base["plateau_score"])
    assert got["n_eligible_neighbors"].equals(base["n_eligible_neighbors"])
    f_none = _cell_frame(_code_grid())
    f_none["sl"] = pd.Series([None if v == "none" else float(v) for v in f_none["sl"]],
                             dtype=object)
    got2 = sfa.plateau_scores(f_none)
    assert got2["plateau_score"].equals(base["plateau_score"])
    # ranking accepts the same forms
    assert len(sfa.rank_cells(got)) == len(sfa.rank_cells(base))


def test_put_screen_inputs_side_decides_the_e1_confirmation_direction():
    rng = np.random.default_rng(6)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900)))
    ser = sfs.build_series(pd.Series(p, index=pd.bdate_range("2000-01-03", periods=900)),
                           name="SYN")
    t0 = np.arange(300, 700, 25)
    cell = ("E1", 10, "none", None)

    def want(down):
        out = []
        for t in t0:
            for u in range(t + 1, t + 6):
                if (p[u] > p[u - 1]) if down else (p[u] < p[u - 1]):
                    out.append(u + 1)
                    break
        return out

    up = sfa.put_screen_inputs(ser, t0, cell)                     # default: up_fade
    down = sfa.put_screen_inputs(ser, t0, cell, side="down_fade")
    up_e, down_e = want(False), want(True)
    assert up_e != down_e
    assert list(up["entry_px"]) == pytest.approx([p[e] for e in up_e])
    assert list(down["entry_px"]) == pytest.approx([p[e] for e in down_e])
    assert list(sfa.put_screen_inputs(ser, t0, cell, side="up_fade")["entry_px"]) \
        == pytest.approx(list(up["entry_px"]))
    # the E0 entry does not depend on the side
    e0u = sfa.put_screen_inputs(ser, t0, ("E0", 10, "none", None), side="up_fade")
    e0d = sfa.put_screen_inputs(ser, t0, ("E0", 10, "none", None), side="down_fade")
    assert list(e0u["entry_px"]) == pytest.approx(list(e0d["entry_px"]))


# --- 5. the FRED-lag reading is disclosed ---

def test_open_choices_disclose_the_fred_lag_reading():
    text = " ".join(sfa.OPEN_CHOICES)
    assert "t0 + L" in text and "the design keeps the baselines at t0" in text
    assert "(t0 + L, t0 + L + 5]" in text
    assert "the design does not spell out the censoring step" in text


def test_vehicle_x_percentile_is_the_midrank_on_the_x_means_and_differs_from_the_pnl_one():
    V, veh, anchors = _vehicle_with_effect(0.0)
    res = sfa.vehicle_cell_test(veh, V[anchors], sfs.PRIMARY_CELL, "long", B=100, seed=5,
                                collect_draws=True)
    xs = []
    for row in res.null_anchors:
        r = sfs.evaluate_variants(veh, row, -1, [sfs.PRIMARY_CELL])
        xs.append(float(r.x[0].mean()))
    arr = np.array(xs)
    want = ((arr < res.mean_x).sum() + 0.5 * (arr == res.mean_x).sum()) / len(arr)
    assert res.null_percentile_x == pytest.approx(want)
    # the raw-P&L percentile is a different statistic (seed 5: 0.81 against 0.76)
    assert res.null_percentile != pytest.approx(res.null_percentile_x)
    assert res.null_percentile == pytest.approx(
        ((np.array(res.null_means_pnl) < res.mean_pnl).sum()
         + 0.5 * (np.array(res.null_means_pnl) == res.mean_pnl).sum()) / res.null_reps_used)


def test_put_screen_for_cell_passes_the_side_to_both_the_observed_and_baseline_inputs():
    rng = np.random.default_rng(6)
    n = 900
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ser = sfs.build_series(pd.Series(p, index=pd.bdate_range("2000-01-03", periods=n)),
                           name="SYN")
    t0 = np.arange(300, 700, 25)
    base_t0 = np.arange(310, 700, 17)
    cell = ("E1", 10, "none", None)
    for side in ("up_fade", "down_fade"):
        out = sfa.put_screen_for_cell(ser, t0, cell, side=side, baseline_t0=base_t0)
        inp = sfa.put_screen_inputs(ser, t0, cell, side=side)
        binp = sfa.put_screen_inputs(ser, base_t0, cell, side=side)
        assert list(out["inputs"]["entry_px"]) == pytest.approx(list(inp["entry_px"]))
        assert out["observed"].n_used + out["observed"].n_dropped == len(inp["entry_px"])
        assert out["baseline"].n_used + out["baseline"].n_dropped == len(binp["entry_px"])
        want_obs = sfa.put_screen(inp["entry_px"], inp["expiry_px"], inp["rv"], H=inp["H"])
        want_base = sfa.put_screen(binp["entry_px"], binp["expiry_px"], binp["rv"], H=binp["H"])
        for pct in (100, 95, 90):
            for got, want in ((out["observed"], want_obs), (out["baseline"], want_base)):
                assert got.results[pct]["avg_pnl"] == pytest.approx(want.results[pct]["avg_pnl"],
                                                                     nan_ok=True)
                assert got.results[pct]["break_even_premium"] == pytest.approx(
                    want.results[pct]["break_even_premium"], nan_ok=True)
    up = sfa.put_screen_inputs(ser, base_t0, cell, side="up_fade")
    down = sfa.put_screen_inputs(ser, base_t0, cell, side="down_fade")
    assert list(up["entry_px"]) != list(down["entry_px"])         # the two sides really differ
