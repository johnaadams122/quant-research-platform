"""Tests for src/mining/cohort_overlay.py -- the generic benchmark-plus-overlay
account builder whose overlay is a monthly SELECTED COHORT minus the
equal-weight basket of ALL ELIGIBLE names (a research control, not an
investable account), plus the seeded permutation placebo and the zero-signal
equal-weight control curve.

Each acceptance property below is pinned by a test that fails when the
module guard it names is removed. All frames are SYNTHETIC on a small
business-day calendar; nothing here reads a file, the network or a ledger.
"""
import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import account_relative  # noqa: E402
from mining import cohort_overlay as co  # noqa: E402
from mining import long_overlay as lo  # noqa: E402
from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns  # noqa: E402

NAMES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


# ------------------------------------------------------------------ helpers

def _calendar(periods: int, start: str = "2024-03-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _ohlc(close, index, gap=None) -> pd.DataFrame:
    """OHLC from a close path. Gapless by default (Open == previous Close);
    ``gap`` is an array of overnight open-vs-previous-close fractions."""
    close = np.asarray(close, dtype=float)
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev if gap is None else prev * (1.0 + np.asarray(gap, dtype=float))
    return pd.DataFrame({"Open": open_,
                         "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999,
                         "Close": close}, index=index)


def _walk(index, seed, vol, drift=0.0, start=100.0):
    rng = np.random.RandomState(seed)
    r = rng.normal(drift, vol, len(index))
    r[0] = 0.0
    return start * np.cumprod(1.0 + r)


def _panel(index, names=NAMES, vol=0.01, drift=0.0):
    return {name: _ohlc(_walk(index, 100 + k, vol, drift, 50.0 + 10 * k), index)
            for k, name in enumerate(names)}


def _spy(index):
    return _ohlc(_walk(index, 3, 0.008, 0.0003), index)


def _flat_spy(index, level=100.0):
    return _ohlc(np.full(len(index), level), index)


def _rate(index, value=0.05):
    return pd.Series(value, index=index)


def _costs(names=NAMES, base=2.0):
    return {name: base + 3.0 * k for k, name in enumerate(names)}


def _window(index, entry_pos, exit_pos, selected, eligible):
    return co.CohortWindow(entry=index[entry_pos], exit=index[exit_pos],
                           selected=tuple(selected), eligible=tuple(eligible),
                           decision=index[entry_pos - 1])


def _monthlike_windows(index, length=5, gap=1, first=2, selected=("AAA", "BBB"),
                       eligible=NAMES):
    """Consecutive windows of ``length`` sessions separated by ``gap`` off
    sessions, starting at position ``first`` (position first-1 is the decision)."""
    out, pos = [], first
    while pos + length - 1 < len(index):
        out.append(_window(index, pos, pos + length - 1, selected, eligible))
        pos += length + gap
    return out


def _build(spy, panel, windows, *, x=0.5, rate=None, costs=None, entry_cost=0.0):
    idx = spy.index
    return co.build_cohort_overlay_curve(
        spy, panel, windows,
        cost_bps_by_symbol=_costs() if costs is None else costs,
        spy_entry_cost_bps=entry_cost,
        financing_rate_annual=_rate(idx) if rate is None else rate,
        overlay_notional=x)


# ------------------------------------------------------------ top-N selection

def test_select_top_n_ranks_descending_and_breaks_ties_by_ticker():
    scores = pd.Series({"CCC": 1.0, "AAA": 1.0, "BBB": 2.0, "DDD": 0.5})
    assert co.select_top_n(scores, 2) == ("BBB", "AAA")       # tie AAA/CCC -> AAA first
    assert co.select_top_n(scores, 3) == ("BBB", "AAA", "CCC")


def test_select_top_n_drops_non_finite_scores_and_never_pads():
    scores = pd.Series({"AAA": np.nan, "BBB": np.inf, "CCC": 1.0, "DDD": -np.inf, "EEE": 2.0})
    assert co.select_top_n(scores, 4) == ("EEE", "CCC")


def test_select_top_n_is_independent_of_input_order():
    scores = pd.Series({"AAA": 1.0, "BBB": 1.0, "CCC": 1.0, "DDD": 1.0})
    forward = co.select_top_n(scores, 2)
    backward = co.select_top_n(scores.iloc[::-1], 2)
    assert forward == backward == ("AAA", "BBB")


# np.linalg.lstsq scores of BYTE-IDENTICAL histories can differ
# in the last bit depending on the column's position in the matrix, so "ties by ticker ascending" applied to
# the raw floats is not reliable: the larger ticker could take the last slot on a rounding accident. Scores
# within RANK_REL_TOL of each other (relative; a gap of a few thousand ulps) are ONE tie class.

def _next_up(value, steps=1):
    for _ in range(steps):
        value = float(np.nextafter(value, np.inf))
    return value


def test_the_tie_tolerance_is_about_twelve_significant_digits():
    assert co.RANK_REL_TOL == 1e-12


@pytest.mark.parametrize("base", [0.3459, -1.0001, 6.0289, 1e-7])
def test_scores_that_differ_only_in_the_last_bit_are_one_tie_and_the_ticker_decides(base):
    twin = _next_up(base)
    assert twin != base                                    # premise: the UNROUNDED scores differ
    scores = pd.Series({"AAA": base, "ZZZ": twin, "MMM": base - abs(base) * 0.5 - 1.0})
    # ZZZ holds the strictly higher raw score, but the pair is one tie class: the smaller ticker goes first
    assert co.select_top_n(scores, 1) == ("AAA",)
    assert co.select_top_n(scores, 2) == ("AAA", "ZZZ")
    assert co.select_top_n(scores, 3) == ("AAA", "ZZZ", "MMM")


def test_the_tie_class_orders_by_ticker_whichever_twin_holds_the_higher_raw_score():
    base = 0.25
    scores = pd.Series({"BBB": base, "AAA": _next_up(base), "CCC": _next_up(base, 3)})
    assert co.select_top_n(scores, 1) == ("AAA",)
    assert co.select_top_n(scores, 3) == ("AAA", "BBB", "CCC")
    lower_first = pd.Series({"AAA": base, "BBB": _next_up(base), "CCC": _next_up(base, 3)})
    assert co.select_top_n(lower_first, 2) == ("AAA", "BBB")


def test_a_near_tie_class_is_never_split_by_the_cut_while_it_is_ranked_below_a_real_gap():
    top, base = 2.0, 0.7
    scores = pd.Series({"ZZZ": _next_up(base, 2), "AAA": base, "BBB": top, "MMM": _next_up(base)})
    assert co.select_top_n(scores, 2) == ("BBB", "AAA")
    assert co.select_top_n(scores, 4) == ("BBB", "AAA", "MMM", "ZZZ")


def test_the_tie_tolerance_is_relative_so_it_holds_at_any_scale_of_score():
    for scale in (1e6, 1.0, 1e-6):
        tied = pd.Series({"AAA": scale, "ZZZ": scale * (1.0 + 5e-14)})
        assert co.select_top_n(tied, 1) == ("AAA",), scale             # 5e-14 relative: one tie class
        apart = pd.Series({"AAA": scale, "ZZZ": scale * (1.0 + 1e-10)})
        assert co.select_top_n(apart, 1) == ("ZZZ",), scale            # 1e-10 relative: a genuine difference


def test_a_chain_of_scores_each_within_tolerance_of_the_next_is_one_tie_class():
    """The documented semantics: a tie class is a run of neighbours, so a pair of near-equal scores can never be
    split by a third score sitting between them (rounding to significant digits could split it)."""
    base = 2.0
    scores = pd.Series({"MMM": base * (1.0 + 1.6e-12), "ZZZ": base * (1.0 + 0.8e-12), "AAA": base})
    assert co.select_top_n(scores, 1) == ("AAA",)
    assert co.select_top_n(scores, 3) == ("AAA", "MMM", "ZZZ")


def test_the_tie_tolerance_is_a_parameter_of_the_selection():
    pair = pd.Series({"AAA": 0.5, "ZZZ": 0.5001})
    assert co.select_top_n(pair, 1) == ("ZZZ",)                          # the default tolerance: 1e-12
    assert co.select_top_n(pair, 1, rel_tol=1e-3) == ("AAA",)            # widened: now one tie class
    twins = pd.Series({"AAA": 0.25, "ZZZ": _next_up(0.25)})
    assert co.select_top_n(twins, 1, rel_tol=0.0) == ("ZZZ",)            # exact comparison: the raw floats rule
    assert co.select_top_n(twins, 1) == ("AAA",)


def test_genuinely_different_scores_are_ranked_by_score_not_by_ticker():
    scores = pd.Series({"AAA": 0.5, "ZZZ": 0.5001, "MMM": 0.4999})
    assert co.select_top_n(scores, 3) == ("ZZZ", "AAA", "MMM")


def test_tie_classes_are_independent_of_input_order():
    base = 1.5
    scores = pd.Series({"CCC": _next_up(base, 2), "AAA": base, "BBB": _next_up(base), "DDD": 0.1})
    forward = co.select_top_n(scores, 3)
    assert forward == co.select_top_n(scores.iloc[::-1], 3) == ("AAA", "BBB", "CCC")


def test_a_tie_class_around_zero_uses_exact_equality_and_never_pads():
    scores = pd.Series({"AAA": 0.0, "BBB": -0.0, "CCC": 1e-300, "DDD": np.nan})
    assert co.select_top_n(scores, 4) == ("CCC", "AAA", "BBB")


# ---------------------------------------------------------------- planning

def _month_windows(index, count=4, first=2, length=5):
    out, pos = [], first
    for _ in range(count):
        out.append((index[pos - 1], index[pos], index[pos + length - 1]))
        pos += length + 1
    return out


def test_plan_cohort_windows_uses_finite_scores_as_the_eligible_set():
    idx = _calendar(40)
    seen = []

    def score_fn(decision):
        seen.append(decision)
        return pd.Series({"AAA": 3.0, "BBB": 2.0, "CCC": 1.0, "DDD": np.nan, "EEE": 0.5})

    plan = co.plan_cohort_windows(_month_windows(idx), score_fn, 2)
    assert len(plan["windows"]) == 4 and plan["skipped"] == []
    first = plan["windows"][0]
    assert first.selected == ("AAA", "BBB")
    assert first.eligible == ("AAA", "BBB", "CCC", "EEE")     # NaN name is not eligible
    assert first.decision == idx[1] and first.entry == idx[2] and first.exit == idx[6]
    assert seen == [w.decision for w in plan["windows"]]      # the ranking is asked once per decision


def test_plan_cohort_windows_skips_a_month_with_fewer_than_n_eligible_names():
    idx = _calendar(40)
    calls = {"n": 0}

    def score_fn(decision):
        calls["n"] += 1
        if calls["n"] == 2:
            return pd.Series({"AAA": 1.0})                     # only one scoreable name
        return pd.Series({"AAA": 3.0, "BBB": 2.0, "CCC": 1.0})

    plan = co.plan_cohort_windows(_month_windows(idx), score_fn, 2)
    assert len(plan["windows"]) == 3
    assert plan["skipped"] == [(idx[1 + 6], 1)]                # (decision, n_eligible)


def test_a_month_with_exactly_n_select_eligible_names_is_a_valid_cohort_not_a_skipped_month():
    """The skip rule is STRICTLY fewer than ``n_select`` eligible names. Exactly
    ``n_select`` fills the cohort (it holds every eligible name); a ``<=`` slip would silently drop the month."""
    idx = _calendar(40)
    plan = co.plan_cohort_windows(_month_windows(idx), lambda d: pd.Series({"AAA": 3.0, "BBB": 2.0}), 2)
    assert len(plan["windows"]) == 4 and plan["skipped"] == []
    first = plan["windows"][0]
    assert first.selected == ("AAA", "BBB") and first.eligible == ("AAA", "BBB")
    one_short = co.plan_cohort_windows(_month_windows(idx), lambda d: pd.Series({"AAA": 3.0}), 2)
    assert one_short["windows"] == [] and [n for _, n in one_short["skipped"]] == [1, 1, 1, 1]


def test_plan_cohort_windows_rejects_a_bad_n_select():
    idx = _calendar(20)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="n_select"):
            co.plan_cohort_windows(_month_windows(idx, count=1), lambda d: pd.Series({"AAA": 1.0}), bad)


# ----------------------------------------------------------- curve contract

def test_the_curve_has_the_shape_abi1_accepts_and_declares_its_account_kind():
    idx = _calendar(60)
    curve = _build(_spy(idx), _panel(idx), _monthlike_windows(idx))
    for key in ("strategy_daily", "in_market", "trades_df", "account_kind", "core_daily",
                "overlay_daily", "overlay_mask", "financing_rate"):
        assert key in curve
    assert curve["account_kind"] == account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY
    assert curve["account_kind"] == "benchmark_plus_overlay"
    assert list(curve["trades_df"].columns) == list(co.TRADES_COLUMNS)
    assert curve["in_market"].all() and curve["in_market"].index.equals(curve["strategy_daily"].index)


def test_the_account_lives_from_the_first_decision_session_to_the_last_exit():
    idx = _calendar(60)
    windows = _monthlike_windows(idx)
    curve = _build(_spy(idx), _panel(idx), windows)
    assert curve["strategy_daily"].index[0] == windows[0].decision
    assert curve["strategy_daily"].index[-1] == windows[-1].exit


def test_the_account_equals_the_benchmark_series_bit_for_bit_off_the_windows():
    idx = _calendar(60)
    spy = _spy(idx)
    windows = _monthlike_windows(idx)
    curve = _build(spy, _panel(idx), windows, entry_cost=1.5)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=1.5)
    off = ~curve["overlay_mask"].to_numpy()
    assert off.any() and off.sum() < len(off)
    account = curve["strategy_daily"]
    assert (account[off] == bench.reindex(account.index)[off]).all()
    assert int(curve["overlay_mask"].sum()) == sum(
        idx.get_loc(w.exit) - idx.get_loc(w.entry) + 1 for w in windows)


def test_the_core_series_is_the_evaluators_benchmark_series():
    idx = _calendar(60)
    spy = _spy(idx)
    curve = _build(spy, _panel(idx), _monthlike_windows(idx), entry_cost=2.5)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=2.5)
    assert (curve["core_daily"] == bench.reindex(curve["core_daily"].index)).all()


def test_the_benchmark_entry_cost_lands_on_the_first_account_session_when_it_is_the_spy_frames_first_bar():
    """An edge test only: the cost is inert on the real 2000-start SPY history, whose
    first bar precedes the account by years). When the first decision session IS the SPY frame's first bar, the
    benchmark's one entry cost (already halved by the caller) is charged there and the core carries it."""
    idx = _calendar(60)
    spy = _spy(idx)
    windows = _monthlike_windows(idx, first=1)
    assert windows[0].decision == idx[0]
    curve = _build(spy, _panel(idx), windows, entry_cost=2.5)
    assert curve["core_daily"].index[0] == idx[0]
    assert curve["core_daily"].iloc[0] == pytest.approx(-2.5e-4, abs=1e-15)
    assert curve["strategy_daily"].iloc[0] == curve["core_daily"].iloc[0]
    free = _build(spy, _panel(idx), windows, entry_cost=0.0)
    assert free["core_daily"].iloc[0] == 0.0
    assert (curve["core_daily"].iloc[1:] == free["core_daily"].iloc[1:]).all()


# --------------------------------------------------------- overlay arithmetic

def test_the_overlay_price_pnl_financing_and_costs_match_hand_arithmetic():
    idx = _calendar(8)                                   # Mon 2024-03-04 .. Wed 2024-03-13
    spy = _flat_spy(idx)
    open_a = np.array([100, 100, 101, 102, 104, 103, 103, 103], float)
    close_a = np.array([100, 102, 103.02, 104, 103, 103, 103, 103], float)
    open_b = np.array([50, 50, 50.5, 51, 50, 50, 50, 50], float)
    close_b = np.array([50, 50.5, 51, 50, 49, 50, 50, 50], float)
    panel = {"AAA": pd.DataFrame({"Open": open_a, "High": open_a * 1.01, "Low": close_a * 0.99,
                                  "Close": close_a}, index=idx),
             "BBB": pd.DataFrame({"Open": open_b, "High": open_b * 1.01, "Low": close_b * 0.99,
                                  "Close": close_b}, index=idx)}
    window = co.CohortWindow(entry=idx[1], exit=idx[4], selected=("AAA",),
                             eligible=("AAA", "BBB"), decision=idx[0])
    x, rate = 0.5, 0.04
    costs = {"AAA": 6.0, "BBB": 10.0}
    curve = co.build_cohort_overlay_curve(
        spy, panel, [window], cost_bps_by_symbol=costs, spy_entry_cost_bps=0.0,
        financing_rate_annual=_rate(idx, rate), overlay_notional=x)

    # ---- hand arithmetic -------------------------------------------------
    equity = 1.0                                         # flat SPY => core 0
    size = equity                                        # sized at the decision close
    sh_long_a = x * size / open_a[1]                     # one selected name, full weight
    sh_short = {"AAA": x * size * 0.5 / open_a[1], "BBB": x * size * 0.5 / open_b[1]}
    expected_daily, total_price, total_fin = [], 0.0, 0.0
    for t in range(1, 5):
        ref_a = open_a[1] if t == 1 else close_a[t - 1]
        ref_b = open_b[1] if t == 1 else close_b[t - 1]
        price = (sh_long_a * (close_a[t] - ref_a)
                 - sh_short["AAA"] * (close_a[t] - ref_a)
                 - sh_short["BBB"] * (close_b[t] - ref_b))
        days = 1.0 if t == 1 else 1.0                    # consecutive weekdays here
        fin = x * size * rate * days / 365.0
        overlay = price - fin
        total_price += price
        total_fin += fin
        if t == 4:
            cost = size * x * (costs["AAA"] + 0.5 * (costs["AAA"] + costs["BBB"])) / 1e4
            overlay -= cost
        expected_daily.append(overlay / equity)          # e_prev, before compounding
        equity *= 1.0 + overlay / equity
    got = curve["overlay_daily"].loc[idx[1]:idx[4]].to_numpy()
    # the annualised rate is the COUPON-EQUIVALENT input to the accrual, so the
    # builder must use the supplied series verbatim
    assert got == pytest.approx(expected_daily, rel=1e-12, abs=1e-15)
    trade = curve["trades_df"].iloc[0]
    assert trade["gross_ret"] == pytest.approx(total_price / size, rel=1e-12)
    assert trade["financing_ret"] == pytest.approx(total_fin / size, rel=1e-12)
    assert trade["cost_ret"] == pytest.approx(
        x * (costs["AAA"] + 0.5 * (costs["AAA"] + costs["BBB"])) / 1e4, rel=1e-12)
    assert trade["net_ret"] == pytest.approx(
        (total_price - total_fin) / size - trade["cost_ret"], rel=1e-12)
    assert trade["entry_date"] == idx[1] and trade["exit_date"] == idx[4]
    assert int(trade["n_selected"]) == 1 and int(trade["n_eligible"]) == 2


def test_selecting_every_eligible_name_cancels_the_price_pnl_exactly():
    """A cohort equal to the eligible set has no signal: the two baskets cancel
    to arithmetic residue and only financing and costs remain."""
    idx = _calendar(60)
    panel = _panel(idx, vol=0.02)
    windows = _monthlike_windows(idx, selected=NAMES, eligible=NAMES)
    curve = _build(_spy(idx), panel, windows, rate=_rate(idx, 0.03))
    trades = curve["trades_df"]
    assert trades["gross_ret"].abs().max() < 1e-12
    assert (trades["net_ret"] < 0).all()
    assert trades["net_ret"].to_numpy() == pytest.approx(
        (-trades["financing_ret"] - trades["cost_ret"]).to_numpy(), abs=1e-12)


def test_costs_are_charged_on_both_legs_short_leg_included():
    """Changing ONLY a name that is eligible but never selected must change the
    trade cost: the short (all-eligible) leg pays its own round trip."""
    idx = _calendar(40)
    panel = _panel(idx)
    windows = _monthlike_windows(idx, selected=("AAA", "BBB"))
    base = _costs()
    heavier = {**base, "FFF": base["FFF"] + 60.0}                 # eligible, never selected
    a = _build(_spy(idx), panel, windows, costs=base)["trades_df"]
    b = _build(_spy(idx), panel, windows, costs=heavier)["trades_df"]
    assert (b["cost_ret"] > a["cost_ret"]).all()
    per_name = 0.5 * 60.0 / len(NAMES) / 1e4                       # x * dcost / n_eligible
    assert (b["cost_ret"] - a["cost_ret"]).to_numpy() == pytest.approx(per_name, rel=1e-9)
    # ...and the long leg pays too: a selected-only change moves it by x*dcost/n_selected
    selected_heavier = {**base, "AAA": base["AAA"] + 40.0}
    c = _build(_spy(idx), panel, windows, costs=selected_heavier)["trades_df"]
    both = 0.5 * 40.0 * (1 / 2 + 1 / len(NAMES)) / 1e4
    assert (c["cost_ret"] - a["cost_ret"]).to_numpy() == pytest.approx(both, rel=1e-9)


def test_entry_is_at_the_open_so_the_overnight_gap_into_the_entry_session_is_not_earned():
    idx = _calendar(30)
    n = len(idx)
    gap = np.zeros(n)
    gap[6] = 0.05                                    # selected name opens 5% above the prior close
    panel = {name: _ohlc(np.full(n, 100.0), idx) for name in NAMES}
    panel["AAA"] = _ohlc(np.full(n, 100.0), idx, gap=gap)    # open 105, close 100 on the entry session
    window = _window(idx, 6, 9, ("AAA",), NAMES)
    curve = _build(_flat_spy(idx), panel, [window], costs={n_: 0.0 for n_ in NAMES},
                   rate=_rate(idx, 0.0))
    # bought at 105 and marked at 100: the long leg loses 0.5/105 * 5, the AAA slice of
    # the short basket (weight 1/6 of 0.5) earns the mirror image. A close-entry
    # builder would book exactly 0 on this session.
    expected = 0.5 / 105.0 * (100.0 - 105.0) * (1.0 - 1.0 / len(NAMES))
    assert curve["overlay_daily"].loc[idx[6]] == pytest.approx(expected, rel=1e-12)
    assert expected < -0.01


def test_position_weights_drift_with_prices_within_a_window_no_daily_rebalance():
    idx = _calendar(12)
    n = len(idx)
    up = 100.0 * np.cumprod(np.r_[1.0, np.full(n - 1, 1.05)])
    flat = np.full(n, 100.0)
    panel = {"AAA": _ohlc(up, idx), "BBB": _ohlc(flat, idx)}
    window = co.CohortWindow(entry=idx[2], exit=idx[6], selected=("AAA",),
                             eligible=("AAA", "BBB"), decision=idx[1])
    curve = co.build_cohort_overlay_curve(
        _flat_spy(idx), panel, [window], cost_bps_by_symbol={"AAA": 0.0, "BBB": 0.0},
        spy_entry_cost_bps=0.0, financing_rate_annual=_rate(idx, 0.0), overlay_notional=0.5)
    # fixed shares: selected basket P&L on day k is (1.05^k - 1) * 0.5 (relative to entry
    # open), short basket = half of it; the net is 0.25 * (cum growth) summed daily
    daily = curve["overlay_daily"].loc[idx[2]:idx[6]].to_numpy()
    first_open = up[1]                                   # Open == previous Close (gapless)
    shares_long = 0.5 / first_open
    shares_short_a = 0.25 / first_open
    expect_raw = [(shares_long - shares_short_a) * (up[t] - (first_open if t == 2 else up[t - 1]))
                  for t in range(2, 7)]
    equity, expected = 1.0, []
    for raw in expect_raw:
        expected.append(raw / equity)
        equity *= 1.0 + raw / equity
    assert daily == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------- oracle vs long_overlay

def test_a_long_only_cohort_against_a_one_name_flat_eligible_set_reproduces_long_overlay():
    """Independent-implementation oracle. The cohort (AAA, BBB, CCC) is the LONG
    leg; the eligible set adds 3000 flat, free-to-trade members, so the short
    basket is almost entirely flat (its exposure to the three legs is 3/3003 of
    the long leg's). The account must then equal the long-only next-open overlay
    `mining.long_overlay` already builds (equal thirds, financing on the whole
    long notional, same costs and sizing) to that order."""
    idx = _calendar(70)
    legs = {"AAA": _ohlc(_walk(idx, 21, 0.01, 0.0002, 100.0), idx),
            "BBB": _ohlc(_walk(idx, 22, 0.012, 0.0001, 90.0), idx),
            "CCC": _ohlc(_walk(idx, 23, 0.008, 0.0003, 120.0), idx)}
    n_flat = 3000                                        # short basket ~ (3/3003) legs, 3000/3003 flat
    flat = {f"F{i:04d}": None for i in range(n_flat)}
    flat_frame = _ohlc(np.full(len(idx), 77.0), idx)
    members = {**legs, **{name: flat_frame for name in flat}}
    eligible = tuple(legs) + tuple(flat)
    spy = _spy(idx)
    rate = _rate(idx, 0.04)
    costs = {**{"AAA": 3.0, "BBB": 8.0, "CCC": 5.0}, **{name: 0.0 for name in flat}}
    windows = [co.CohortWindow(entry=idx[pos], exit=idx[pos + 6], selected=("AAA", "BBB", "CCC"),
                               eligible=eligible, decision=idx[pos - 1]) for pos in range(1, 60, 10)]
    curve = co.build_cohort_overlay_curve(
        spy, members, windows, cost_bps_by_symbol=costs, spy_entry_cost_bps=1.0,
        financing_rate_annual=rate, overlay_notional=0.5)
    reference = lo.build_next_open_sensitivity_curve(
        spy, legs, [(idx[pos - 1], idx[pos + 6]) for pos in range(1, 60, 10)],
        leg_cost_bps={"AAA": 3.0, "BBB": 8.0, "CCC": 5.0}, spy_entry_cost_bps=1.0,
        financing_rate_annual=rate, overlay_notional=0.5)
    # the short basket's own exposure to the three legs is 3/3003 of the long's: the
    # accounts agree to that order
    got = curve["overlay_daily"]
    want = reference["overlay_daily"].reindex(got.index)          # both accounts start at idx[0]
    assert np.abs(got - want).max() < 0.002 * np.abs(want).max() + 1e-6
    assert curve["trades_df"]["net_ret"].to_numpy() == pytest.approx(
        reference["trades_df"]["net_ret"].to_numpy(), rel=0.01, abs=1e-5)
    assert curve["trades_df"]["financing_ret"].to_numpy() == pytest.approx(
        reference["trades_df"]["financing_ret"].to_numpy(), rel=1e-3)


# ------------------------------------------------------------------ financing

def test_financing_is_charged_on_the_long_notional_only_and_scales_with_the_rate():
    idx = _calendar(40)
    panel = _panel(idx)
    windows = _monthlike_windows(idx)
    lo_ = _build(_spy(idx), panel, windows, rate=_rate(idx, 0.02))["trades_df"]
    hi_ = _build(_spy(idx), panel, windows, rate=_rate(idx, 0.04))["trades_df"]
    assert (lo_["financing_ret"] > 0).all()
    assert (hi_["financing_ret"] / lo_["financing_ret"]).to_numpy() == pytest.approx(2.0, rel=1e-9)
    # long leg only: x * annual * days / 365 per session, ONE calendar day on the
    # first session (intraday borrow, even after a weekend) and elapsed calendar
    # days afterwards; checked on EVERY window (some enter on a Monday)
    assert any(w.entry.weekday() == 0 for w in windows)
    for k, w in enumerate(windows):
        first, last = idx.get_loc(w.entry), idx.get_loc(w.exit)
        days = 1.0 + sum((idx[t] - idx[t - 1]).days for t in range(first + 1, last + 1))
        assert lo_["financing_ret"].iloc[k] == pytest.approx(0.5 * 0.02 * days / 365.0, rel=1e-9)


def test_a_non_finite_financing_rate_on_a_window_session_refuses_to_assume_zero():
    idx = _calendar(40)
    rate = _rate(idx, 0.03)
    windows = _monthlike_windows(idx)
    rate.loc[windows[1].entry] = np.nan
    with pytest.raises(ValueError, match="financing rate is not finite"):
        _build(_spy(idx), _panel(idx), windows, rate=rate)


def test_a_negative_bill_rate_is_applied_with_its_sign():
    idx = _calendar(40)
    windows = _monthlike_windows(idx)
    neg = _build(_spy(idx), _panel(idx), windows, rate=_rate(idx, -0.01))["trades_df"]
    assert (neg["financing_ret"] < 0).all()


# --------------------------------------------------------------- fail closed

def test_a_selected_name_without_an_open_on_the_entry_date_fails_closed():
    idx = _calendar(40)
    panel = _panel(idx)
    windows = _monthlike_windows(idx, selected=("AAA", "BBB"))
    entry = windows[1].entry
    frame = panel["AAA"].copy()
    frame.loc[entry, "Open"] = np.nan
    panel["AAA"] = frame
    with pytest.raises(co.CohortDataError) as err:
        _build(_spy(idx), panel, windows)
    assert err.value.reason_code == "missing_member_price"
    assert "AAA" in err.value.tickers


def test_a_selected_name_without_a_close_on_the_exit_date_fails_closed():
    idx = _calendar(40)
    panel = _panel(idx)
    windows = _monthlike_windows(idx, selected=("AAA", "BBB"))
    exit_ = windows[0].exit
    panel["BBB"] = panel["BBB"].drop(exit_)
    with pytest.raises(co.CohortDataError) as err:
        _build(_spy(idx), panel, windows)
    assert err.value.reason_code == "missing_member_price"
    assert "BBB" in err.value.tickers and pd.Timestamp(exit_).strftime("%Y-%m-%d") in err.value.dates


def test_an_eligible_but_unselected_name_with_a_hole_also_fails_closed():
    """The equal-weight baseline is a leg of the account: a hole in a name that
    is only in the short basket would otherwise be silently skipped."""
    idx = _calendar(40)
    panel = _panel(idx)
    windows = _monthlike_windows(idx, selected=("AAA", "BBB"))
    panel["FFF"] = panel["FFF"].drop(idx[idx.get_loc(windows[0].entry) + 2])     # interior session
    with pytest.raises(co.CohortDataError) as err:
        _build(_spy(idx), panel, windows)
    assert err.value.reason_code == "missing_member_price" and "FFF" in err.value.tickers


def test_a_missing_member_frame_or_cost_is_refused():
    idx = _calendar(40)
    panel = _panel(idx)
    windows = _monthlike_windows(idx)
    short = {k: v for k, v in panel.items() if k != "CCC"}
    with pytest.raises(co.CohortDataError) as err:
        _build(_spy(idx), short, windows)
    assert err.value.reason_code == "missing_member_frame"
    with pytest.raises(co.CohortDataError) as err:
        _build(_spy(idx), panel, windows, costs={k: v for k, v in _costs().items() if k != "DDD"})
    assert err.value.reason_code == "missing_member_cost"


def test_a_selected_name_that_is_not_eligible_is_refused():
    idx = _calendar(40)
    bad = co.CohortWindow(entry=idx[3], exit=idx[6], selected=("AAA", "ZZZ"),
                          eligible=("AAA", "BBB"), decision=idx[2])
    with pytest.raises(ValueError, match="subset"):
        _build(_spy(idx), _panel(idx), [bad])


def test_duplicate_or_empty_cohorts_are_refused():
    idx = _calendar(40)
    dup = co.CohortWindow(entry=idx[3], exit=idx[6], selected=("AAA", "AAA"),
                          eligible=("AAA", "BBB"), decision=idx[2])
    with pytest.raises(ValueError, match="duplicate"):
        _build(_spy(idx), _panel(idx), [dup])
    empty = co.CohortWindow(entry=idx[3], exit=idx[6], selected=(), eligible=("AAA", "BBB"),
                            decision=idx[2])
    with pytest.raises(ValueError, match="empty"):
        _build(_spy(idx), _panel(idx), [empty])


def test_windows_must_be_ascending_disjoint_and_on_the_calendar():
    idx = _calendar(40)
    a = _window(idx, 3, 8, ("AAA",), NAMES)
    overlapping = _window(idx, 8, 12, ("AAA",), NAMES)
    with pytest.raises(ValueError, match="overlap"):
        _build(_spy(idx), _panel(idx), [a, overlapping])
    with pytest.raises(ValueError, match="overlap|ascending"):
        _build(_spy(idx), _panel(idx), [_window(idx, 10, 14, ("AAA",), NAMES), a])
    off_calendar = co.CohortWindow(entry=pd.Timestamp("2024-03-09"), exit=idx[8],
                                   selected=("AAA",), eligible=NAMES, decision=idx[2])
    with pytest.raises(ValueError, match="not a session"):
        _build(_spy(idx), _panel(idx), [off_calendar])
    backwards = co.CohortWindow(entry=idx[8], exit=idx[8], selected=("AAA",), eligible=NAMES,
                                decision=idx[7])
    with pytest.raises(ValueError, match="strictly before"):
        _build(_spy(idx), _panel(idx), [backwards])


def test_the_decision_must_be_the_session_immediately_before_the_entry():
    idx = _calendar(40)
    bad = co.CohortWindow(entry=idx[5], exit=idx[9], selected=("AAA",), eligible=NAMES,
                          decision=idx[3])
    with pytest.raises(ValueError, match="decision"):
        _build(_spy(idx), _panel(idx), [bad])


def test_the_first_window_needs_a_session_before_its_entry():
    idx = _calendar(40)
    first = co.CohortWindow(entry=idx[0], exit=idx[4], selected=("AAA",), eligible=NAMES)
    with pytest.raises(ValueError, match="before"):
        _build(_spy(idx), _panel(idx), [first])


@pytest.mark.parametrize("x", [0.0, -0.1, 0.5000001, 1.0, float("nan"), float("inf")])
def test_the_overlay_notional_is_bounded_by_reg_t_margin_arithmetic(x):
    idx = _calendar(40)
    with pytest.raises(ValueError, match="overlay_notional"):
        _build(_spy(idx), _panel(idx), _monthlike_windows(idx), x=x)


def test_the_maximum_notional_is_the_reg_t_margin_bound():
    # gross exposure 1 + 2x (SPY core + long leg + short leg), half of it initial margin
    assert co.MAX_COHORT_OVERLAY_NOTIONAL == 0.5
    assert 0.5 * (1.0 + 2.0 * co.MAX_COHORT_OVERLAY_NOTIONAL) <= 1.0
    idx = _calendar(40)
    _build(_spy(idx), _panel(idx), _monthlike_windows(idx), x=0.5)          # accepted


def test_a_window_needs_a_calendar_in_the_spy_frame_not_a_shorter_member_calendar():
    idx = _calendar(40)
    spy = _spy(idx)
    panel = _panel(idx)
    windows = _monthlike_windows(idx)
    curve = _build(spy, panel, windows)
    assert curve["strategy_daily"].index.isin(spy.index).all()


# ------------------------------------------------------------- look-ahead

def test_prices_after_a_windows_exit_never_change_the_account_up_to_that_exit():
    idx = _calendar(80)
    spy = _spy(idx)
    panel = _panel(idx, vol=0.015)
    windows = _monthlike_windows(idx)
    base = _build(spy, panel, windows)
    cut = idx.get_loc(windows[2].exit)
    perturbed_panel, perturbed_spy = {}, spy.copy()
    for name, frame in panel.items():
        f = frame.copy()
        f.iloc[cut + 1:] = f.iloc[cut + 1:] * 1.37
        perturbed_panel[name] = f
    perturbed_spy.iloc[cut + 1:] = perturbed_spy.iloc[cut + 1:] * 0.6
    changed = _build(perturbed_spy, perturbed_panel, windows[:3])
    prefix = base["strategy_daily"].loc[:idx[cut]]
    assert (changed["strategy_daily"].loc[:idx[cut]] == prefix).all()
    assert (changed["overlay_daily"].loc[:idx[cut]] == base["overlay_daily"].loc[:idx[cut]]).all()


def test_the_sizing_equity_is_the_prior_close_not_the_entry_session_close():
    idx = _calendar(40)
    spy = _spy(idx)
    spy.iloc[idx.get_loc(_monthlike_windows(idx)[1].entry), spy.columns.get_loc("Close")] *= 1.08
    windows = _monthlike_windows(idx)
    curve = _build(spy, _panel(idx), windows)
    equity = (1.0 + curve["strategy_daily"]).cumprod()
    for k, window in enumerate(windows):
        assert curve["trades_df"]["size"].iloc[k] == pytest.approx(
            float(equity.loc[window.decision]), rel=1e-12)          # the close BEFORE the entry




# ------------------------------------------------- the account identity on EVERY session
# The shared guard (overlay_risk_report.require_excess_is_overlay_only) only verifies the sessions
# the overlay is OFF. A monthly cohort schedule has no gaps, so on the real schedule that is ONE session (the
# first decision session) and an in-window fault reaches the ledger unseen. This check recomputes, on EVERY
# session of the account, account minus benchmark against the overlay series, and the core against the
# benchmark, so the guard no longer depends on one session.

def _identity_fixture(entry_cost=1.5):
    idx = _calendar(60)
    spy = _spy(idx)
    curve = _build(spy, _panel(idx), _monthlike_windows(idx), entry_cost=entry_cost)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=entry_cost)
    return curve, bench


def _with(curve, **replacements):
    return {**curve, **replacements}


def _bumped(series, position, amount):
    out = series.copy()
    out.iloc[position] += amount
    return out


def test_a_genuine_curve_satisfies_the_account_identity_on_every_session():
    curve, bench = _identity_fixture()
    assert int(curve["overlay_mask"].sum()) > 30                       # plenty of in-window sessions are checked
    assert co.require_account_is_benchmark_plus_overlay(curve, bench, family_id="fam") is None


def test_every_in_window_session_is_checked_by_the_identity_guard():
    curve, bench = _identity_fixture()
    on = np.flatnonzero(curve["overlay_mask"].to_numpy())
    assert len(on) > 30
    for position in on:
        for amount in (1e-4, -1e-4):                                   # a gap of either sign is a fault
            faulty = _with(curve, strategy_daily=_bumped(curve["strategy_daily"], position, amount))
            with pytest.raises(RuntimeError, match="not the benchmark plus the overlay"):
                co.require_account_is_benchmark_plus_overlay(faulty, bench, family_id="fam")


def test_the_identity_guard_sees_an_in_window_fault_the_shared_off_window_guard_cannot():
    from mining import overlay_risk_report
    curve, bench = _identity_fixture()
    on = np.flatnonzero(curve["overlay_mask"].to_numpy())
    bumped = _bumped(curve["strategy_daily"], on[3], 1e-4)
    # premise: the shared off-window guard passes on this faulty account (it only reads overlay-OFF sessions)
    assert overlay_risk_report.require_excess_is_overlay_only(
        bumped, bench, curve["overlay_mask"], family_id="fam") is None
    with pytest.raises(RuntimeError,
                       match=r"risk-basis rule.*not the benchmark plus the overlay.*Nothing was recorded"):
        co.require_account_is_benchmark_plus_overlay(_with(curve, strategy_daily=bumped), bench, family_id="fam")


def test_an_off_window_fault_is_also_refused():
    curve, bench = _identity_fixture()
    off = np.flatnonzero(~curve["overlay_mask"].to_numpy())
    assert len(off) > 0
    faulty = _with(curve, strategy_daily=_bumped(curve["strategy_daily"], off[0], 1e-4))
    with pytest.raises(RuntimeError, match="not the benchmark plus the overlay"):
        co.require_account_is_benchmark_plus_overlay(faulty, bench, family_id="fam")


def test_an_overlay_series_that_disagrees_with_the_account_is_refused():
    curve, bench = _identity_fixture()
    on = np.flatnonzero(curve["overlay_mask"].to_numpy())
    faulty = _with(curve, overlay_daily=_bumped(curve["overlay_daily"], on[5], 1e-4))
    with pytest.raises(RuntimeError, match="not the benchmark plus the overlay"):
        co.require_account_is_benchmark_plus_overlay(faulty, bench, family_id="fam")


@pytest.mark.parametrize("amount", [1e-4, -1e-4])
def test_a_reported_core_series_that_is_not_the_benchmark_is_refused_by_its_own_check(amount):
    """The account is right (benchmark plus overlay) but the curve's reported core_daily is not the
    evaluator's benchmark: only the core check sees it. A gap of either sign is a fault."""
    curve, bench = _identity_fixture()
    on = np.flatnonzero(curve["overlay_mask"].to_numpy())
    faulty = _with(curve, core_daily=_bumped(curve["core_daily"], on[7], amount))
    with pytest.raises(RuntimeError, match=r"core differs from the evaluator's benchmark on 1 session\(s\)"):
        co.require_account_is_benchmark_plus_overlay(faulty, bench, family_id="fam")


def test_an_account_built_on_a_wrong_core_is_refused_by_the_account_identity():
    curve, bench = _identity_fixture()
    on = np.flatnonzero(curve["overlay_mask"].to_numpy())
    faulty = _with(curve, core_daily=_bumped(curve["core_daily"], on[7], 1e-4),
                   strategy_daily=_bumped(curve["strategy_daily"], on[7], 1e-4))      # account = core + overlay holds
    with pytest.raises(RuntimeError, match="not the benchmark plus the overlay on 1 session"):
        co.require_account_is_benchmark_plus_overlay(faulty, bench, family_id="fam")


def test_a_different_benchmark_than_the_one_the_account_was_built_on_is_refused():
    curve, bench = _identity_fixture(entry_cost=1.5)
    idx = _calendar(60)
    other_spy = _ohlc(_walk(idx, 99, 0.008, 0.0003), idx)                # a different price history
    other = cost_adjusted_benchmark_daily_returns(other_spy, entry_cost_bps=1.5)
    with pytest.raises(RuntimeError, match="not the benchmark plus the overlay|core"):
        co.require_account_is_benchmark_plus_overlay(curve, other, family_id="fam")


def test_a_benchmark_entry_cost_charged_on_the_first_bar_outside_the_account_span_is_not_visible_to_the_guard():
    """Disclosed limit, shared with the off-window guard: the benchmark's entry cost lands on the benchmark
    series' FIRST bar, which lies before the account starts when SPY lists earlier (production), so a
    different entry cost is not seen on the account's own sessions."""
    curve, _ = _identity_fixture(entry_cost=1.5)
    idx = _calendar(60)
    different_cost = cost_adjusted_benchmark_daily_returns(_spy(idx), entry_cost_bps=9.0)
    assert curve["strategy_daily"].index[0] > idx[0]
    assert co.require_account_is_benchmark_plus_overlay(curve, different_cost, family_id="fam") is None


def test_the_identity_tolerance_is_arithmetic_residue_and_can_be_overridden():
    curve, bench = _identity_fixture()
    on = np.flatnonzero(curve["overlay_mask"].to_numpy())
    residue = _with(curve, strategy_daily=_bumped(curve["strategy_daily"], on[2], 1e-13))
    assert co.require_account_is_benchmark_plus_overlay(residue, bench, family_id="fam") is None
    real = _with(curve, strategy_daily=_bumped(curve["strategy_daily"], on[2], 1e-11))
    with pytest.raises(RuntimeError, match="not the benchmark plus the overlay"):
        co.require_account_is_benchmark_plus_overlay(real, bench, family_id="fam")
    assert co.require_account_is_benchmark_plus_overlay(real, bench, family_id="fam", tolerance=1e-9) is None


def test_the_identity_message_names_the_family_the_count_and_the_largest_gap():
    curve, bench = _identity_fixture()
    on = np.flatnonzero(curve["overlay_mask"].to_numpy())
    faulty = _with(curve, strategy_daily=_bumped(_bumped(curve["strategy_daily"], on[1], 1e-4), on[4], 3e-4))
    with pytest.raises(RuntimeError) as excinfo:
        co.require_account_is_benchmark_plus_overlay(faulty, bench, family_id="my_family")
    text = str(excinfo.value)
    assert "my_family" in text and "2 session(s)" in text and "0.0003" in text


def test_the_identity_guard_refuses_series_that_are_not_on_the_accounts_index():
    curve, bench = _identity_fixture()
    shifted = curve["overlay_daily"].copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="overlay_daily"):
        co.require_account_is_benchmark_plus_overlay(_with(curve, overlay_daily=shifted), bench, family_id="fam")
    short_core = curve["core_daily"].iloc[:-1]
    with pytest.raises(ValueError, match="core_daily"):
        co.require_account_is_benchmark_plus_overlay(_with(curve, core_daily=short_core), bench, family_id="fam")


def test_a_benchmark_missing_on_an_account_session_cannot_be_verified_and_is_refused():
    curve, bench = _identity_fixture()
    holed = bench.drop(curve["strategy_daily"].index[10])
    with pytest.raises(RuntimeError, match="benchmark has 1 missing session"):
        co.require_account_is_benchmark_plus_overlay(curve, holed, family_id="fam")


def test_the_identity_guard_does_not_mutate_its_inputs():
    curve, bench = _identity_fixture()
    before = {k: curve[k].copy() for k in ("strategy_daily", "overlay_daily", "core_daily")}
    bench_before = bench.copy()
    co.require_account_is_benchmark_plus_overlay(curve, bench, family_id="fam")
    for key, series in before.items():
        pd.testing.assert_series_equal(curve[key], series)
    pd.testing.assert_series_equal(bench, bench_before)

# ------------------------------------------------------- permutation placebo

def _planted_panel(n_names=60, periods=160, seed=7):
    """A panel where the first 5 names are given a genuine, persistent positive
    drift; a cohort that picks them has real edge over the eligible set."""
    idx = _calendar(periods)
    names = tuple(f"N{i:03d}" for i in range(n_names))
    rng = np.random.RandomState(seed)
    panel = {}
    for k, name in enumerate(names):
        drift = 0.004 if k < 5 else 0.0
        r = rng.normal(drift, 0.01, periods)
        r[0] = 0.0
        panel[name] = _ohlc(60.0 * np.cumprod(1.0 + r), idx)
    return idx, names, panel


def _placebo_inputs(selected_names):
    idx, names, panel = _planted_panel()
    spy = _spy(idx)
    windows = _monthlike_windows(idx, length=10, gap=1, first=3, selected=selected_names,
                                 eligible=names)
    kwargs = dict(cost_bps_by_symbol={n: 2.0 for n in names}, spy_entry_cost_bps=0.0,
                  financing_rate_annual=_rate(idx, 0.0), overlay_notional=0.5)
    return spy, panel, windows, kwargs


def test_the_placebo_refuses_fewer_than_the_minimum_number_of_draws():
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    assert co.MIN_PLACEBO_DRAWS >= 300
    with pytest.raises(ValueError, match="n_draws"):
        co.permutation_placebo(spy, panel, windows, n_draws=co.MIN_PLACEBO_DRAWS - 1, seed=1, **kwargs)


def test_the_placebo_is_deterministic_for_a_seed_and_differs_between_seeds():
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    a = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=5, **kwargs)
    b = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=5, **kwargs)
    c = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=6, **kwargs)
    assert a["draws"] == b["draws"]
    assert a["draws"] != c["draws"]
    assert a["n_draws"] == 300 and a["seed"] == 5 and len(a["draws"]) == 300


def test_placebo_cohorts_have_the_real_size_and_come_from_the_eligible_set():
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    seen = []
    original = co._simulate

    def spy_simulate(panel_data, sel_cols, *args, **kw):
        seen.append((panel_data, [c.copy() for c in sel_cols]))
        return original(panel_data, sel_cols, *args, **kw)

    co._simulate = spy_simulate
    try:
        co.permutation_placebo(spy, panel, windows, n_draws=300, seed=5, **kwargs)
    finally:
        co._simulate = original
    assert len(seen) == 301                                    # the real rule + 300 placebo draws
    real_cols = seen[0][1]
    real_sets = [set(c.tolist()) for c in real_cols]
    differing = 0
    for panel_data, cols in seen[1:]:
        for k, chosen in enumerate(cols):
            assert len(chosen) == len(real_cols[k]) == 3
            assert len(set(chosen.tolist())) == 3
            assert set(chosen.tolist()) <= set(panel_data.all_cols[k].tolist())
            differing += set(chosen.tolist()) != real_sets[k]
    assert differing > 0.9 * 300 * len(windows)               # they really are random draws


def _placebo_selections(monkeypatch, spy, panel, windows, kwargs, *, n_draws=300, seed=5):
    """Every per-window cohort list the placebo hands to ``_simulate``: the real rule's first, then one list
    per draw (each holding one column array per window)."""
    seen = []
    original = co._simulate

    def spy_simulate(panel_data, sel_cols, *args, **kw):
        seen.append([c.copy() for c in sel_cols])
        return original(panel_data, sel_cols, *args, **kw)

    monkeypatch.setattr(co, "_simulate", spy_simulate)
    result = co.permutation_placebo(spy, panel, windows, n_draws=n_draws, seed=seed, **kwargs)
    return seen, result


def test_every_window_of_a_placebo_draw_gets_its_own_independently_drawn_cohort(monkeypatch):
    """A placebo that draws ONE random cohort per draw and reuses it in every window
    keeps the right cohort size and the right eligible set, passes every per-cohort check, and turns the null
    into a much wider, correlated distribution while the 0.95 reading rule still reads its percentile. Every
    window of a draw must be a fresh pick."""
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    seen, _ = _placebo_selections(monkeypatch, spy, panel, windows, kwargs)
    assert len(seen) == 301 and len(windows) > 10
    draws = seen[1:]
    distinct_per_draw = [len({frozenset(c.tolist()) for c in cols}) for cols in draws]
    assert min(distinct_per_draw) > 1                             # never one cohort across the whole draw
    assert np.mean(distinct_per_draw) >= 0.9 * len(windows)       # about one distinct cohort per window
    consecutive_equal = sum(frozenset(a.tolist()) == frozenset(b.tolist())
                            for cols in draws for a, b in zip(cols, cols[1:]))
    assert consecutive_equal <= 3                                 # 3,900 pairs; a random repeat is a 1-in-34,000 event


def test_the_placebo_draws_are_not_the_same_cohort_shifted_from_window_to_window(monkeypatch):
    """Independence across windows means a window's pick carries no information about the next window's: the
    cohorts of window k and window k+1 overlap only as much as two random 3-of-60 picks do (0.15 names on
    average), not as a copy would (3)."""
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    seen, _ = _placebo_selections(monkeypatch, spy, panel, windows, kwargs)
    overlaps = [len(set(a.tolist()) & set(b.tolist())) for cols in seen[1:] for a, b in zip(cols, cols[1:])]
    assert np.mean(overlaps) < 0.5
    assert max(overlaps) < 3


def test_a_cohort_with_real_edge_ranks_high_against_the_placebo_and_a_random_one_does_not():
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002", "N003", "N004"))
    good = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=11, **kwargs)
    assert good["real_percentile"] >= 0.95
    assert good["p_upper"] <= 0.06
    _, names, _ = _planted_panel()
    spy2, panel2, windows2, kwargs2 = _placebo_inputs(tuple(names[20:25]))    # no planted drift
    plain = co.permutation_placebo(spy2, panel2, windows2, n_draws=300, seed=11, **kwargs2)
    assert 0.02 < plain["real_percentile"] < 0.98


def test_the_placebo_statistic_of_the_real_cohort_is_the_curves_own_overlay_sharpe():
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    curve = co.build_cohort_overlay_curve(spy, panel, windows, **kwargs)
    result = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=2, **kwargs)
    assert result["real_value"] == pytest.approx(co.overlay_sharpe(curve))
    assert result["statistic"] == "annualized_sharpe_of_overlay_daily_arithmetic_excess"
    for key in ("draws_mean", "draws_std", "draws_p05", "draws_p50", "draws_p95",
                "real_percentile", "p_upper"):
        assert key in result


def _numpy_sharpe(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean() / values.std(ddof=1) * np.sqrt(252.0))


@pytest.mark.parametrize("selected", [("N000", "N001", "N002", "N003", "N004"), ("N020", "N021", "N022")])
def test_the_placebo_report_numbers_equal_an_independent_numpy_computation_on_a_fixed_seed(selected):
    """The report-only numbers the ledger note prints (the real rule's
    annualized Sharpe, the upper-tail p with its +1 correction, the quantiles, the percentile) are pinned by
    value against a numpy computation from the draws themselves, not by key presence."""
    spy, panel, windows, kwargs = _placebo_inputs(selected)
    curve = co.build_cohort_overlay_curve(spy, panel, windows, **kwargs)
    result = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=5, **kwargs)
    draws = np.array(result["draws"], dtype=float)
    assert len(draws) == 300 and np.isfinite(draws).all()
    assert result["real_value"] == pytest.approx(_numpy_sharpe(curve["overlay_daily"]), rel=1e-12)
    assert result["draws_mean"] == pytest.approx(draws.mean(), rel=1e-12)
    assert result["draws_std"] == pytest.approx(draws.std(ddof=1), rel=1e-12)
    assert result["draws_p05"] == pytest.approx(np.percentile(draws, 5), rel=1e-12)
    assert result["draws_p50"] == pytest.approx(np.percentile(draws, 50), rel=1e-12)
    assert result["draws_p95"] == pytest.approx(np.percentile(draws, 95), rel=1e-12)
    real = result["real_value"]
    assert result["real_percentile"] == (draws <= real).mean()
    assert result["p_upper"] == pytest.approx((1 + int((draws >= real).sum())) / 301.0, rel=1e-12)
    assert result["p_upper"] >= 1.0 / 301.0                           # never exactly zero: the +1 correction


def test_the_placebo_draws_are_annualized_with_252_sessions_not_calendar_days():
    """Each draw's Sharpe is the daily mean over the daily standard deviation of THAT draw's overlay series,
    times the square root of 252. The percentile is scale-free, so only a value pin sees a wrong constant."""
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    seen = []
    original = co._simulate

    def recording(panel_data, sel_cols, *args, **kw):
        out = original(panel_data, sel_cols, *args, **kw)
        seen.append(np.array(out["overlay_frac"], dtype=float))
        return out

    co._simulate = recording
    try:
        result = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=5, **kwargs)
    finally:
        co._simulate = original
    assert len(seen) == 301
    expected = [_numpy_sharpe(series) for series in seen[1:]]
    np.testing.assert_allclose(result["draws"], expected, rtol=1e-12)
    assert result["real_value"] == pytest.approx(_numpy_sharpe(seen[0]), rel=1e-12)


def test_a_draw_that_ties_the_real_rule_counts_as_at_or_below_it():
    """The percentile is the share of draws AT OR BELOW the real statistic (its docstring says so). With every
    name on the identical price path every random cohort scores exactly the real rule's overlay (no price edge,
    the same cost on both legs), so every draw ties it: at-or-below is 1.0 and the upper-tail p is 1.0."""
    idx = _calendar(160)
    names = tuple(f"N{i:03d}" for i in range(20))
    frame = _ohlc(60.0 * np.cumprod(1.0 + 0.01 * np.sin(np.arange(160) / 3.0)), idx)
    panel = {name: frame.copy() for name in names}
    windows = _monthlike_windows(idx, length=10, gap=1, first=3, selected=names[:3], eligible=names)
    result = co.permutation_placebo(
        _spy(idx), panel, windows, n_draws=300, seed=5, cost_bps_by_symbol={n: 2.0 for n in names},
        spy_entry_cost_bps=0.0, financing_rate_annual=_rate(idx, 0.0), overlay_notional=0.5)
    assert set(result["draws"]) == {result["real_value"]}
    assert result["real_percentile"] == 1.0 and result["p_upper"] == 1.0


def test_the_placebo_never_touches_the_scored_curve_inputs():
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    before = {k: v.copy() for k, v in panel.items()}
    windows_before = list(windows)
    co.permutation_placebo(spy, panel, windows, n_draws=300, seed=2, **kwargs)
    assert windows == windows_before
    for name, frame in panel.items():
        pd.testing.assert_frame_equal(frame, before[name])


def test_the_placebo_docstring_labels_the_short_leg_a_research_control():
    for fn in (co.build_cohort_overlay_curve, co.permutation_placebo,
               co.build_equal_weight_control_curve):
        assert "RESEARCH CONTROL" in " ".join(inspect.getdoc(fn).split())
    assert "RESEARCH CONTROL" in " ".join(inspect.getdoc(co).split())


# ------------------------------------------------------ equal-weight control

def test_the_equal_weight_control_is_a_whole_account_curve_with_no_account_kind():
    idx = _calendar(60)
    panel = _panel(idx)
    windows = _monthlike_windows(idx)
    curve = co.build_equal_weight_control_curve(
        _spy(idx), panel, windows, cost_bps_by_symbol=_costs())
    assert "account_kind" not in curve
    assert set(("strategy_daily", "in_market", "trades_df")) <= set(curve)
    mask = curve["in_market"]
    assert mask.any() and (~mask).any()
    # off-window sessions carry no return (cash is credited by the evaluator)
    assert (curve["strategy_daily"][~mask] == 0.0).all()
    assert curve["strategy_daily"].index[0] == windows[0].decision


def test_the_equal_weight_control_matches_hand_arithmetic_and_pays_one_leg_of_cost():
    idx = _calendar(8)
    open_a = np.array([100, 100, 101, 102, 104, 103, 103, 103], float)
    close_a = np.array([100, 102, 103.02, 104, 103, 103, 103, 103], float)
    open_b = np.array([50, 50, 50.5, 51, 50, 50, 50, 50], float)
    close_b = np.array([50, 50.5, 51, 50, 49, 50, 50, 50], float)
    panel = {"AAA": pd.DataFrame({"Open": open_a, "Close": close_a}, index=idx),
             "BBB": pd.DataFrame({"Open": open_b, "Close": close_b}, index=idx)}
    window = co.CohortWindow(entry=idx[1], exit=idx[4], selected=("AAA",),
                             eligible=("AAA", "BBB"), decision=idx[0])
    costs = {"AAA": 6.0, "BBB": 10.0}
    curve = co.build_equal_weight_control_curve(_flat_spy(idx), panel, [window],
                                                cost_bps_by_symbol=costs)
    equity, expected = 1.0, []
    for t in range(1, 5):
        ref_a = open_a[1] if t == 1 else close_a[t - 1]
        ref_b = open_b[1] if t == 1 else close_b[t - 1]
        raw = 0.5 * (close_a[t] - ref_a) / open_a[1] + 0.5 * (close_b[t] - ref_b) / open_b[1]
        if t == 4:
            raw -= 0.5 * (costs["AAA"] + costs["BBB"]) / 1e4
        expected.append(raw / equity)
        equity *= 1.0 + raw / equity
    got = curve["strategy_daily"].loc[idx[1]:idx[4]].to_numpy()
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert curve["in_market"].loc[idx[1]:idx[4]].all() and not curve["in_market"].loc[idx[0]]


def test_the_equal_weight_control_ignores_the_selected_cohort():
    idx = _calendar(60)
    panel = _panel(idx)
    a = co.build_equal_weight_control_curve(
        _spy(idx), panel, _monthlike_windows(idx, selected=("AAA",)), cost_bps_by_symbol=_costs())
    b = co.build_equal_weight_control_curve(
        _spy(idx), panel, _monthlike_windows(idx, selected=("FFF", "EEE")), cost_bps_by_symbol=_costs())
    assert (a["strategy_daily"] == b["strategy_daily"]).all()


def test_the_equal_weight_control_fails_closed_on_a_hole_like_the_overlay():
    idx = _calendar(40)
    panel = _panel(idx)
    windows = _monthlike_windows(idx)
    panel["FFF"] = panel["FFF"].drop(idx[idx.get_loc(windows[0].entry) + 2])
    with pytest.raises(co.CohortDataError):
        co.build_equal_weight_control_curve(_spy(idx), panel, windows, cost_bps_by_symbol=_costs())


# ----------------------------------------------- the ABI-1 evaluator accepts it

def test_abi1_evaluates_the_overlay_curve_on_the_overlay_only_excess_basis():
    from mining import abi1_driver
    idx = _calendar(1600, "2015-01-05")
    names = tuple(f"S{i:02d}" for i in range(12))
    rng = np.random.RandomState(4)
    panel = {name: _ohlc(60.0 * np.cumprod(1.0 + np.r_[0.0, rng.normal(0.0002, 0.012, len(idx) - 1)]), idx)
             for name in names}
    spy = _spy(idx)
    windows, pos = [], 3
    while pos + 21 < len(idx):
        windows.append(co.CohortWindow(entry=idx[pos], exit=idx[pos + 20], selected=names[:3],
                                       eligible=names, decision=idx[pos - 1]))
        pos += 21
    curve = co.build_cohort_overlay_curve(
        spy, panel, windows, cost_bps_by_symbol={n: 1.0 for n in names}, spy_entry_cost_bps=0.0,
        financing_rate_annual=_rate(idx, 0.01), overlay_notional=0.5)
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy, benchmark_entry_cost_bps=0.0, account_type="taxable",
        tax_policy="realization")
    assert result["layer2"]["account_basis"] == "overlay_only_excess"
    assert result["window"]["account_sessions"] >= 1500


def test_the_control_curve_is_scored_on_the_whole_account_basis():
    from mining import abi1_driver
    idx = _calendar(1600, "2015-01-05")
    names = tuple(f"S{i:02d}" for i in range(12))
    rng = np.random.RandomState(4)
    panel = {name: _ohlc(60.0 * np.cumprod(1.0 + np.r_[0.0, rng.normal(0.0002, 0.012, len(idx) - 1)]), idx)
             for name in names}
    spy = _spy(idx)
    windows, pos = [], 3
    while pos + 21 < len(idx):
        windows.append(co.CohortWindow(entry=idx[pos], exit=idx[pos + 20], selected=names[:3],
                                       eligible=names, decision=idx[pos - 1]))
        pos += 21
    control = co.build_equal_weight_control_curve(spy, panel, windows,
                                                  cost_bps_by_symbol={n: 1.0 for n in names})
    result = abi1_driver.evaluate_family(
        control, benchmark_frame=spy, benchmark_entry_cost_bps=0.0, account_type="taxable",
        tax_policy="none")
    assert result["layer2"]["account_basis"] == "whole_account"


# ------------------------------------------------- ex-ante scheduled month windows

def _sessions_with_holidays():
    days = pd.bdate_range("2024-01-01", "2024-07-31")
    return days.difference(pd.DatetimeIndex(["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29"]))


def test_month_windows_are_the_last_session_then_the_next_months_first_and_last():
    sessions = _sessions_with_holidays()
    out = co.month_windows(sessions, "2024-01-31", "2024-04-30")
    assert [d.strftime("%Y-%m-%d") for d, _, _ in out] == ["2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"]
    # Mar 29 (Good Friday) is not a session, so March's last session is Thursday the 28th
    assert [e.strftime("%Y-%m-%d") for _, e, _ in out] == ["2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
    assert [x.strftime("%Y-%m-%d") for _, _, x in out] == ["2024-02-29", "2024-03-28", "2024-04-30", "2024-05-31"]
    for decision, entry, exit_ in out:
        assert entry == sessions[sessions.get_loc(decision) + 1]          # the very next session
    for (_, _, x1), (_, e2, _) in zip(out, out[1:]):
        assert x1 < e2                                                     # complete, non-overlapping cohorts


def test_month_windows_range_bounds_are_inclusive_and_decisions_outside_are_skipped():
    sessions = _sessions_with_holidays()
    out = co.month_windows(sessions, "2024-02-29", "2024-03-28")
    assert [d.strftime("%Y-%m-%d") for d, _, _ in out] == ["2024-02-29", "2024-03-28"]
    assert co.month_windows(sessions, "2024-02-15", "2024-02-28") == []     # no month-end in range


def test_month_windows_refuse_a_partial_following_month():
    sessions = pd.bdate_range("2024-01-01", "2024-03-15")                   # March is partial
    with pytest.raises(ValueError, match="partial month"):
        co.month_windows(sessions, "2024-01-31", "2024-02-29")
    assert len(co.month_windows(sessions, "2024-01-31", "2024-01-31")) == 1  # Feb is whole and Mar exists
    with pytest.raises(ValueError, match="sorted, unique"):
        co.month_windows(pd.DatetimeIndex(["2024-01-03", "2024-01-02"]), "2024-01-01", "2024-01-31")
    with pytest.raises(ValueError, match="sorted, unique"):
        co.month_windows(pd.DatetimeIndex([]), "2024-01-01", "2024-01-31")


def test_month_windows_are_a_pure_function_of_the_session_list():
    sessions = _sessions_with_holidays()
    assert co.month_windows(sessions, "2024-01-31", "2024-04-30") == co.month_windows(
        list(sessions), "2024-01-31", "2024-04-30")


# ------------------------------------- daily-rebalanced zero-signal control

def test_the_daily_rebalanced_control_matches_hand_arithmetic_with_the_decision_close_as_reference():
    idx = _calendar(8)
    open_a = np.array([100, 101, 110, 102, 104, 103, 103, 103], float)      # the entry session opens 1 percent up
    close_a = np.array([100, 102, 103.02, 104, 103, 103, 103, 103], float)
    open_b = np.array([50, 50.4, 50.5, 51, 50, 50, 50, 50], float)
    close_b = np.array([50, 50.5, 51, 50, 49, 50, 50, 50], float)
    panel = {"AAA": pd.DataFrame({"Open": open_a, "Close": close_a}, index=idx),
             "BBB": pd.DataFrame({"Open": open_b, "Close": close_b}, index=idx)}
    window = co.CohortWindow(entry=idx[1], exit=idx[4], selected=("AAA",), eligible=("AAA", "BBB"),
                             decision=idx[0])
    curve = co.build_daily_rebalanced_control_curve(_flat_spy(idx), panel, [window])
    expected = [0.5 * (close_a[t] / close_a[t - 1] - 1.0) + 0.5 * (close_b[t] / close_b[t - 1] - 1.0)
                for t in range(1, 5)]
    got = curve["strategy_daily"].loc[idx[1]:idx[4]].to_numpy()
    assert got == pytest.approx(expected, rel=1e-12)
    # the FIRST session is measured from the decision-session close, so the overnight gap is earned
    assert got[0] == pytest.approx(0.5 * 0.02 + 0.5 * 0.01, rel=1e-12)
    assert curve["in_market"].loc[idx[1]:idx[4]].all() and not curve["in_market"].loc[idx[0]]
    assert curve["strategy_daily"].loc[idx[0]] == 0.0
    trade = curve["trades_df"].iloc[0]
    assert trade["net_ret"] == pytest.approx(np.prod(1.0 + np.array(expected)) - 1.0, rel=1e-12)
    assert trade["cost_ret"] == 0.0 and trade["financing_ret"] == 0.0 and "account_kind" not in curve


def test_the_daily_rebalanced_control_is_rebalanced_not_buy_and_hold_and_ignores_the_cohort():
    idx = _calendar(12)
    n = len(idx)
    up = 100.0 * np.cumprod(np.r_[1.0, np.full(n - 1, 1.05)])
    flat = np.full(n, 100.0)
    panel = {"AAA": _ohlc(up, idx), "BBB": _ohlc(flat, idx)}
    a = co.build_daily_rebalanced_control_curve(
        _flat_spy(idx), panel, [co.CohortWindow(entry=idx[2], exit=idx[6], selected=("AAA",),
                                                eligible=("AAA", "BBB"), decision=idx[1])])
    b = co.build_daily_rebalanced_control_curve(
        _flat_spy(idx), panel, [co.CohortWindow(entry=idx[2], exit=idx[6], selected=("BBB",),
                                                eligible=("AAA", "BBB"), decision=idx[1])])
    assert (a["strategy_daily"] == b["strategy_daily"]).all()
    # weights are RESET to one half each session: every daily return is exactly half of AAA's 5 percent
    assert a["strategy_daily"].loc[idx[2]:idx[6]].to_numpy() == pytest.approx(np.full(5, 0.025), rel=1e-12)


def test_the_daily_rebalanced_control_fails_closed_on_a_missing_decision_close_or_window_close():
    idx = _calendar(30)
    panel = _panel(idx)
    windows = _monthlike_windows(idx)
    first = windows[0]
    no_decision = dict(panel)
    no_decision["CCC"] = panel["CCC"].drop(first.decision)
    with pytest.raises(co.CohortDataError) as err:
        co.build_daily_rebalanced_control_curve(_spy(idx), no_decision, windows)
    assert err.value.reason_code == "missing_member_price" and "CCC" in err.value.tickers
    assert pd.Timestamp(first.decision).strftime("%Y-%m-%d") in err.value.dates
    no_window = dict(panel)
    no_window["DDD"] = panel["DDD"].drop(idx[idx.get_loc(first.entry) + 2])
    with pytest.raises(co.CohortDataError):
        co.build_daily_rebalanced_control_curve(_spy(idx), no_window, windows)
    with pytest.raises(ValueError, match="at least one"):
        co.build_daily_rebalanced_control_curve(_spy(idx), panel, [])


def test_the_daily_rebalanced_control_docstring_says_gross_and_research_control():
    doc = " ".join(inspect.getdoc(co.build_daily_rebalanced_control_curve).split())
    assert "RESEARCH CONTROL" in doc and "GROSS OF COSTS" in doc


# ================================================================ additional tests

def test_the_placebo_states_that_it_cannot_test_rank_conditional_survivorship():
    """Every draw picks from the SAME eligible set as the real cohort, so a survivorship or
    index-inclusion effect specific to the ranked cohort is present only in the real one."""
    spy, panel, windows, kwargs = _placebo_inputs(("N000", "N001", "N002"))
    result = co.permutation_placebo(spy, panel, windows, n_draws=300, seed=2, **kwargs)
    note = " ".join(result["note"].split())
    assert "RESEARCH CONTROL" in note
    assert "same eligible set" in note and "cannot test rank-conditional survivorship" in note
    assert "rank-conditional survivorship" in " ".join(co.permutation_placebo.__doc__.split())


def test_the_module_docstring_labels_only_the_first_three_limitations_as_flattering():
    doc = " ".join(co.__doc__.split())
    assert "each of the first three FLATTERS" in doc and "first four" not in doc
    assert "the tax identity" in doc and "unmeasured direction" in doc     # the tax item is not a flatterer
    assert "cancel the LEVEL" in doc and "does NOT cancel rank-conditional survivorship" in doc
