"""Tests for src/mining/beta_neutral_overlay.py -- the benchmark-plus-overlay
account builder behind ipo_long_run_underperformance_fade_beta_neutral.

Each test below guards one acceptance property and is built to FAIL when
that property is broken. The tests use small synthetic calendars and a short
beta window unless a test is specifically about the production 252-session
window.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import beta_neutral_overlay as bno  # noqa: E402


# ------------------------------------------------------------------ helpers

def _calendar(periods: int, start: str = "2015-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _returns(index, seed: int, vol: float, drift: float = 0.0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.normal(drift, vol, len(index))


def _close_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(1.0 + np.asarray(returns, dtype=float))


def _ohlc(close, index, gap=None) -> pd.DataFrame:
    """OHLC from a close path. Gapless by default (Open == previous Close);
    ``gap`` is an array of overnight open-vs-previous-close fractions."""
    close = np.asarray(close, dtype=float)
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev if gap is None else prev * (1.0 + np.asarray(gap, dtype=float))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _pair(periods=400, k=None, seed=7, spy_vol=0.01, noise=0.0,
          ipo_drift=0.0, gap_vol=0.0, index=None):
    """(ipo_frame, spy_frame) where IPO daily return = k*SPY return + noise
    + ipo_drift (k None -> independent random IPO; k may be a per-session
    array for a beta regime shift). ``gap_vol`` adds independent overnight
    gaps to each instrument's Open."""
    idx = _calendar(periods) if index is None else index
    r_spy = _returns(idx, seed, spy_vol, 0.0004)
    if k is None:
        # an independent-noise IPO with a genuine positive beta to SPY (the
        # registered range refuses a negative or huge estimate, and a real
        # IPO ETF's beta is ~1.2)
        r_ipo = (1.2 * r_spy + _returns(idx, seed + 1, 0.012, 0.0002))
    else:
        r_ipo = k * r_spy + ipo_drift
        if noise:
            r_ipo = r_ipo + _returns(idx, seed + 2, noise)
    r_spy[0] = 0.0
    r_ipo[0] = 0.0
    gap_spy = _returns(idx, seed + 3, gap_vol) if gap_vol else None
    gap_ipo = _returns(idx, seed + 4, gap_vol) if gap_vol else None
    spy = _ohlc(_close_from_returns(r_spy), idx, gap=gap_spy)
    ipo = _ohlc(_close_from_returns(r_ipo, 50.0), idx, gap=gap_ipo)
    return ipo, spy


def _build(ipo, spy, *, window=40, x=0.25, ipo_cost=0.0, spy_cost=0.0,
           entry_cost=0.0):
    return bno.build_beta_neutral_overlay_curve(
        ipo, spy, ipo_cost_bps=ipo_cost, spy_cost_bps=spy_cost,
        spy_entry_cost_bps=entry_cost, overlay_notional=x, beta_window=window)


# ------------------------------------------------------- rebalance calendar

def test_rebalance_positions_are_the_first_session_of_each_calendar_month():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01",
                            "2024-02-02", "2024-02-29", "2024-03-01",
                            "2024-03-04"])
    positions = bno.rebalance_positions(idx)
    assert list(positions) == [0, 2, 5]


def test_rebalance_positions_on_a_real_business_calendar_hit_every_month_once():
    idx = _calendar(600)
    positions = bno.rebalance_positions(idx)
    months = idx[positions].to_period("M")
    assert months.is_unique
    assert len(months) == len(idx.to_period("M").unique())
    # each is the first session of its month
    for p in positions:
        assert p == 0 or idx[p].month != idx[p - 1].month


# ------------------------------------------------------------- hedge ratio

def test_causal_hedge_ratio_is_the_ols_slope_of_the_prior_window_returns():
    ipo, spy = _pair(200)
    window = 40
    ratios = bno.causal_hedge_ratios(ipo["Close"], spy["Close"], window=window)
    r_ipo = ipo["Close"].pct_change().to_numpy()
    r_spy = spy["Close"].pct_change().to_numpy()
    for t in (window + 1, window + 2, 90, 150, 199):
        expected = np.polyfit(r_spy[t - window:t], r_ipo[t - window:t], 1)[0]
        assert ratios.iloc[t] == pytest.approx(expected, rel=1e-9)


def test_causal_hedge_ratio_is_nan_until_a_full_window_of_prior_returns_exists():
    ipo, spy = _pair(120)
    window = 40
    ratios = bno.causal_hedge_ratios(ipo["Close"], spy["Close"], window=window)
    # returns exist from position 1, so the first full window of PRIOR
    # returns is positions 1..window, first usable at session window + 1
    assert ratios.iloc[:window + 1].isna().all()
    assert np.isfinite(ratios.iloc[window + 1:]).all()


@pytest.mark.parametrize("t", [45, 60, 61, 100, 150, 199])
def test_hedge_ratio_for_session_t_is_unchanged_by_data_from_t_onward(t):
    ipo, spy = _pair(200)
    base = bno.causal_hedge_ratios(ipo["Close"], spy["Close"], window=40)
    rng = np.random.RandomState(99)
    ipo_bad, spy_bad = ipo["Close"].copy(), spy["Close"].copy()
    ipo_bad.iloc[t:] = ipo_bad.iloc[t:] * rng.uniform(0.3, 3.0, len(ipo_bad) - t)
    spy_bad.iloc[t:] = spy_bad.iloc[t:] * rng.uniform(0.3, 3.0, len(spy_bad) - t)
    corrupted = bno.causal_hedge_ratios(ipo_bad, spy_bad, window=40)
    assert corrupted.iloc[t] == base.iloc[t]
    # NaN warm-up entries must match as NaN (NaN != NaN under ==)
    np.testing.assert_array_equal(corrupted.iloc[:t + 1].to_numpy(),
                                  base.iloc[:t + 1].to_numpy())


# ------------------------------------------------ curve structure / warm-up

from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns  # noqa: E402


def _first_expected_entry_position(index, window):
    """First calendar-month start with a full window of PRIOR returns."""
    for p in bno.rebalance_positions(index):
        if p >= window + 1:
            return int(p)
    raise AssertionError("no eligible month start")


def test_curve_is_aligned_fully_invested_and_carries_a_trades_frame():
    ipo, spy = _pair(300)
    curve = _build(ipo, spy)
    assert curve["strategy_daily"].index.equals(ipo.index)
    assert curve["in_market"].index.equals(ipo.index)
    assert curve["in_market"].dtype == bool
    assert curve["in_market"].all()  # the SPY core keeps the account deployed
    trades = curve["trades_df"]
    for column in ("entry_date", "exit_date", "held", "reason", "gross_ret",
                   "net_ret", "hedge_ratio"):
        assert column in trades.columns
    assert len(trades) > 0
    assert np.isfinite(curve["strategy_daily"].to_numpy()).all()


def test_curve_declares_itself_a_benchmark_plus_overlay_account():
    """Risk-basis rule: the risk layer of such an account is measured on the
    overlay's own excess. `abi1_driver.evaluate_family` reads this declaration,
    so a builder that forgets it silently reverts to the whole-account basis."""
    ipo, spy = _pair(300)
    curve = _build(ipo, spy)
    assert curve["account_kind"] == "benchmark_plus_overlay"


def test_account_is_exactly_the_benchmark_series_before_the_first_overlay():
    ipo, spy = _pair(300)
    curve = _build(ipo, spy, window=40, entry_cost=0.3)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.3)
    first = _first_expected_entry_position(ipo.index, 40)
    assert ipo.index.get_loc(curve["trades_df"]["entry_date"].iloc[0]) == first
    np.testing.assert_array_equal(
        curve["strategy_daily"].iloc[:first].to_numpy(),
        bench.iloc[:first].to_numpy())
    # ...and the overlay is genuinely live from the entry session onward
    assert curve["strategy_daily"].iloc[first] != bench.iloc[first]


def test_core_is_the_full_history_benchmark_series_when_spy_history_is_longer():
    """Production case: SPY history starts long before the IPO ETF. The core
    must be the benchmark series computed over the FULL SPY history -- the
    first IPO day carries SPY's true return that day, and the benchmark's
    one-off entry cost sits at SPY's own first bar, outside the account's
    life -- not one recomputed on the IPO-span slice (which would zero the
    first return and charge the entry cost inside the account)."""
    spy_idx = _calendar(500)
    r_spy = _returns(spy_idx, 21, 0.01, 0.0004)
    r_spy[0] = 0.0
    spy = _ohlc(_close_from_returns(r_spy), spy_idx)
    ipo_idx = spy_idx[80:]
    r_ipo = 1.2 * r_spy[80:] + _returns(ipo_idx, 22, 0.008)
    r_ipo[0] = 0.0
    ipo = _ohlc(_close_from_returns(r_ipo, 50.0), ipo_idx)
    curve = _build(ipo, spy, entry_cost=0.4)
    expected = cost_adjusted_benchmark_daily_returns(
        spy, entry_cost_bps=0.4).reindex(ipo_idx)
    first = _first_expected_entry_position(ipo_idx, 40)
    np.testing.assert_array_equal(curve["strategy_daily"].iloc[:first].to_numpy(),
                                  expected.iloc[:first].to_numpy())
    assert curve["core_daily"].iloc[0] != 0.0
    assert curve["core_daily"].iloc[0] == pytest.approx(
        spy["Close"].iloc[80] / spy["Close"].iloc[79] - 1.0, abs=1e-15)


def test_production_warmup_is_252_sessions_of_prior_returns():
    ipo, spy = _pair(700)
    curve = bno.build_beta_neutral_overlay_curve(
        ipo, spy, ipo_cost_bps=0.0, spy_cost_bps=0.0, spy_entry_cost_bps=0.0)
    assert bno.BETA_WINDOW_SESSIONS == 252
    first = ipo.index.get_loc(curve["trades_df"]["entry_date"].iloc[0])
    assert first == _first_expected_entry_position(ipo.index, 252)
    assert first >= 253
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    np.testing.assert_array_equal(curve["strategy_daily"].iloc[:first].to_numpy(),
                                  bench.iloc[:first].to_numpy())


# --------------------------------------------------------------- perfect hedge

def test_ipo_identical_to_spy_earns_exactly_zero_excess_when_costs_are_zero():
    ipo, spy = _pair(300, k=1.0)
    curve = _build(ipo, spy)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    np.testing.assert_allclose(curve["strategy_daily"].to_numpy(),
                               bench.to_numpy(), atol=1e-12, rtol=0)
    assert (curve["trades_df"]["net_ret"].abs() < 1e-12).all()


def test_ipo_identical_to_spy_pays_only_the_round_trip_costs_on_both_legs():
    ipo, spy = _pair(300, k=1.0)
    x, ipo_cost, spy_cost = 0.25, 12.0, 1.5
    curve = _build(ipo, spy, x=x, ipo_cost=ipo_cost, spy_cost=spy_cost)
    trades = curve["trades_df"]
    expected = -(x * ipo_cost + x * trades["hedge_ratio"] * spy_cost) / 1e4
    np.testing.assert_allclose(trades["net_ret"].to_numpy(),
                               expected.to_numpy(), atol=1e-12, rtol=0)
    # gross is the price P&L only: zero for a perfectly hedged pair
    assert (trades["gross_ret"].abs() < 1e-12).all()
    # the account underperforms the benchmark by the summed costs, no more
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    excess = curve["strategy_daily"] - bench
    assert excess.sum() < 0
    assert excess.sum() == pytest.approx(trades["net_ret"].sum(), rel=0.05)


@pytest.mark.parametrize("k", [-0.8, 4.0])
def test_a_hedge_ratio_outside_the_registered_range_fails_closed(k):
    """The registered design is short IPO + a LONG SPY hedge that fits Reg-T
    (x * (beta + 1) <= 1). A negative estimate would silently flip the hedge
    to a short (and a negative beta times a cost would print a rebate); one
    above the cap breaks the feasibility argument. Refuse, never clip."""
    ipo, spy = _pair(300, k=k, spy_vol=0.004)
    with pytest.raises(ValueError, match="hedge ratio"):
        _build(ipo, spy)


def test_the_hedge_ratio_cap_is_the_reg_t_feasibility_limit_for_the_notional():
    assert bno.MAX_HEDGE_RATIO == 3.0
    assert bno.OVERLAY_NOTIONAL * (bno.MAX_HEDGE_RATIO + 1.0) <= 1.0 + 1e-12


@pytest.mark.parametrize("k", [0.5, 1.7])
def test_ipo_proportional_to_spy_is_nearly_perfectly_hedged(k):
    ipo, spy = _pair(400, k=k, spy_vol=0.004)
    curve = _build(ipo, spy)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    excess = curve["strategy_daily"] - bench
    # only second-order (compounding-convexity) leakage survives
    assert excess.abs().max() < 2e-4
    assert abs(excess.sum()) < 5e-4


# ---------------------------------------------------------------- alpha capture

def test_overlay_captures_a_known_daily_underperformance_scaled_by_notional():
    drift = 0.0005
    x = 0.25
    ipo, spy = _pair(500, k=1.3, spy_vol=0.005, ipo_drift=-drift)
    curve = _build(ipo, spy, x=x)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    excess = curve["strategy_daily"] - bench
    first = _first_expected_entry_position(ipo.index, 40)
    active_days = len(ipo.index) - first
    expected = x * drift * active_days
    assert excess.sum() > 0
    assert excess.sum() == pytest.approx(expected, rel=0.03)
    # the pre-overlay stretch contributes nothing
    assert (excess.iloc[:first] == 0).all()


# ----------------------------------------------- fixed share counts / drift

def _pos(frame, stamp) -> int:
    return int(frame.index.get_loc(stamp))


def test_segment_return_is_the_fixed_share_open_to_open_price_move():
    """Independent closed form: with share counts fixed at the fill, a
    segment's overlay gross return on sizing equity is
    x * (beta * spy_move - ipo_move) over the entry-Open to exit-fill move,
    whatever happened in between (weights drift)."""
    ipo, spy = _pair(400, gap_vol=0.004)
    x, ipo_cost, spy_cost = 0.25, 12.0, 1.5
    curve = _build(ipo, spy, x=x, ipo_cost=ipo_cost, spy_cost=spy_cost)
    trades = curve["trades_df"]
    assert len(trades) > 5
    for _, row in trades.iterrows():
        e, xp = _pos(ipo, row["entry_date"]), _pos(ipo, row["exit_date"])
        if row["reason"] == "rebalance":
            ipo_exit, spy_exit = ipo["Open"].iloc[xp], spy["Open"].iloc[xp]
        else:
            ipo_exit, spy_exit = ipo["Close"].iloc[xp], spy["Close"].iloc[xp]
        ipo_move = ipo_exit / ipo["Open"].iloc[e] - 1.0
        spy_move = spy_exit / spy["Open"].iloc[e] - 1.0
        gross = x * (row["hedge_ratio"] * spy_move - ipo_move)
        cost = x * (ipo_cost + row["hedge_ratio"] * spy_cost) / 1e4
        assert row["gross_ret"] == pytest.approx(gross, abs=1e-12)
        assert row["net_ret"] == pytest.approx(gross - cost, abs=1e-12)


def test_daily_account_return_is_core_plus_fixed_share_overlay_over_prior_equity():
    ipo, spy = _pair(400, gap_vol=0.004)
    x, ipo_cost, spy_cost = 0.25, 12.0, 1.5
    curve = _build(ipo, spy, x=x, ipo_cost=ipo_cost, spy_cost=spy_cost,
                   entry_cost=0.4)
    strat = curve["strategy_daily"].to_numpy()
    core = curve["core_daily"].to_numpy()
    equity_before = np.concatenate([[1.0], np.cumprod(1.0 + strat)])[:-1]
    o_i, c_i = ipo["Open"].to_numpy(), ipo["Close"].to_numpy()
    o_s, c_s = spy["Open"].to_numpy(), spy["Close"].to_numpy()
    trades = curve["trades_df"]

    def shares(row):
        e = _pos(ipo, row["entry_date"])
        size = equity_before[e]
        beta = row["hedge_ratio"]
        cost = size * x * (ipo_cost + beta * spy_cost) / 1e4
        return e, x * size / o_i[e], beta * x * size / o_s[e], cost

    prev, cur = trades.iloc[0], trades.iloc[1]
    pe, p_nipo, p_nspy, p_cost = shares(prev)
    e, n_ipo, n_spy, _ = shares(cur)
    xp = _pos(ipo, cur["exit_date"])
    assert _pos(ipo, prev["exit_date"]) == e and e + 3 < xp

    # a mid-segment day: fixed shares, close to close, over prior equity
    t = e + 2
    expected_mid = core[t] + (n_spy * (c_s[t] - c_s[t - 1])
                              - n_ipo * (c_i[t] - c_i[t - 1])) / equity_before[t]
    assert strat[t] == pytest.approx(expected_mid, abs=1e-12)

    # the rebalance day: old legs closed at Open (net of their round-trip
    # cost) + new legs opened at Open and marked to the Close
    expected_entry = core[e] + (
        p_nspy * (o_s[e] - c_s[e - 1]) - p_nipo * (o_i[e] - c_i[e - 1]) - p_cost
        + n_spy * (c_s[e] - o_s[e]) - n_ipo * (c_i[e] - o_i[e])
    ) / equity_before[e]
    assert strat[e] == pytest.approx(expected_entry, abs=1e-12)


# ------------------------------------------------------------ month boundaries

def test_overlay_rebalances_on_month_start_sessions_only_and_segments_chain():
    ipo, spy = _pair(400)
    curve = _build(ipo, spy, window=40)
    trades = curve["trades_df"]
    n = len(ipo)
    entries = [_pos(ipo, d) for d in trades["entry_date"]]
    expected = [int(p) for p in bno.rebalance_positions(ipo.index)
                if p >= 41 and p < n - 1]
    assert entries == expected
    exits = list(trades["exit_date"])
    assert exits[:-1] == list(trades["entry_date"])[1:]        # chained
    assert set(trades["reason"].iloc[:-1]) == {"rebalance"}
    assert trades["reason"].iloc[-1] == "end_of_data"
    assert exits[-1] == ipo.index[-1]
    ratios = bno.causal_hedge_ratios(ipo["Close"], spy["Close"], window=40)
    np.testing.assert_allclose(trades["hedge_ratio"].to_numpy(),
                               ratios.iloc[entries].to_numpy(), atol=1e-15, rtol=0)


def test_hedge_ratio_tracks_a_beta_regime_shift():
    idx = _calendar(500)
    k = np.where(np.arange(len(idx)) < 200, 1.0, 2.0)
    ipo, spy = _pair(k=k, spy_vol=0.008, noise=0.0005, index=idx)
    curve = _build(ipo, spy, window=40)
    trades = curve["trades_df"]
    assert trades["hedge_ratio"].iloc[0] == pytest.approx(1.0, abs=0.08)
    assert trades["hedge_ratio"].iloc[-1] == pytest.approx(2.0, abs=0.08)


def test_a_rebalance_on_the_final_bar_closes_the_old_segment_and_opens_nothing():
    idx = pd.bdate_range("2015-01-02", "2016-04-01")   # 2016-04-01 is a month start
    assert bno.rebalance_positions(idx)[-1] == len(idx) - 1
    ipo, spy = _pair(index=idx, gap_vol=0.003)
    curve = _build(ipo, spy, window=40, ipo_cost=10.0, spy_cost=1.0)
    trades = curve["trades_df"]
    assert (trades["entry_date"] < idx[-1]).all()
    last = trades.iloc[-1]
    assert last["reason"] == "rebalance" and last["exit_date"] == idx[-1]
    assert last["exit_px"] == pytest.approx(ipo["Open"].iloc[-1])
    assert np.isfinite(curve["strategy_daily"].to_numpy()).all()


def test_an_unusable_hedge_ratio_on_a_final_bar_rebalance_is_not_applied_so_not_refused():
    """The final-bar rebalance opens nothing, so its (here invalid) hedge ratio
    is never APPLIED; the range check is about applied hedges only."""
    idx = pd.bdate_range("2015-01-02", "2016-04-01")    # last bar is a month start
    assert bno.rebalance_positions(idx)[-1] == len(idx) - 1
    ipo, spy = _pair(index=idx, k=1.2, spy_vol=0.008, noise=0.003)
    n = len(idx)
    r_spy = spy["Close"].pct_change().to_numpy()
    r_ipo = ipo["Close"].pct_change().to_numpy()
    r_ipo[n - 22:n] = -5.0 * r_spy[n - 22:n]        # flips only the newest window
    r_ipo[0] = 0.0
    ipo = _ohlc(50.0 * np.cumprod(1.0 + r_ipo), idx)
    ratios = bno.causal_hedge_ratios(ipo["Close"], spy["Close"], window=40)
    assert ratios.iloc[-1] < 0.0                    # the unapplied final-bar ratio is invalid
    curve = _build(ipo, spy, window=40)
    trades = curve["trades_df"]
    assert trades["reason"].iloc[-1] == "rebalance"
    assert trades["exit_date"].iloc[-1] == idx[-1]
    assert ((trades["hedge_ratio"] >= 0.0) & (trades["hedge_ratio"] <= bno.MAX_HEDGE_RATIO)).all()


def test_a_mid_month_final_bar_closes_the_last_segment_at_that_close():
    idx = pd.bdate_range("2015-01-02", "2016-03-17")
    ipo, spy = _pair(index=idx, gap_vol=0.003)
    curve = _build(ipo, spy, window=40)
    last = curve["trades_df"].iloc[-1]
    assert last["reason"] == "end_of_data" and last["exit_date"] == idx[-1]
    assert last["exit_px"] == pytest.approx(ipo["Close"].iloc[-1])
    assert last["spy_exit_px"] == pytest.approx(spy["Close"].iloc[-1])


# ------------------------------------------------------------- no look-ahead

def _corrupt_future(ipo, spy, cut, seed):
    """Replace EVERY bar from position ``cut`` on with freshly generated ones
    (different volatility, drift and gaps; still a positive IPO/SPY beta so
    the builder's registered hedge-range check is not what reacts). Only the
    past survives; the level is continued from the last kept close."""
    rng = np.random.RandomState(seed)
    m = len(ipo) - cut
    r_spy = rng.normal(-0.001, 0.02, m)
    r_ipo = 1.3 * r_spy + rng.normal(0.0, 0.012, m)
    gaps = (rng.normal(0.0, 0.004, m), rng.normal(0.0, 0.004, m))

    def replaced(frame, returns, gap):
        out = frame.copy()
        prev_close = frame["Close"].iloc[cut - 1]
        close = prev_close * np.cumprod(1.0 + returns)
        open_ = np.concatenate([[prev_close], close[:-1]]) * (1.0 + gap)
        for name, values in (("Open", open_), ("Close", close),
                             ("High", np.maximum(open_, close) * 1.001),
                             ("Low", np.minimum(open_, close) * 0.999)):
            out.iloc[cut:, out.columns.get_loc(name)] = values
        return out

    return replaced(ipo, r_ipo, gaps[0]), replaced(spy, r_spy, gaps[1])


# Cutoffs sitting 1-4 sessions AFTER a segment's entry are the ones that can
# expose a hedge ratio (or fill) that peeks a few sessions ahead: the segment
# is live before the cutoff but its inputs would reach past it.
_ENTRY_POSITIONS = [int(p) for p in bno.rebalance_positions(_calendar(400))
                    if 41 <= p < 399]
_CUTS = sorted({120, 141, 200, 260, 331}
               | {_ENTRY_POSITIONS[j] + d for j in (2, 5, 8) for d in (1, 2, 4)})


@pytest.mark.parametrize("cut", _CUTS)
def test_curve_days_before_a_cutoff_do_not_depend_on_data_after_it(cut):
    ipo, spy = _pair(400, gap_vol=0.003)
    kwargs = dict(x=0.25, ipo_cost=8.0, spy_cost=1.0, entry_cost=0.3)
    full = _build(ipo, spy, **kwargs)["strategy_daily"]
    # truncation: the run that never saw bars >= cut agrees on every day
    # before its own final bar (the final bar carries the end_of_data close)
    truncated = _build(ipo.iloc[:cut], spy.iloc[:cut], **kwargs)["strategy_daily"]
    np.testing.assert_array_equal(truncated.iloc[:cut - 1].to_numpy(),
                                  full.iloc[:cut - 1].to_numpy())
    # corruption: every bar >= cut replaced leaves days < cut untouched
    ipo_bad, spy_bad = _corrupt_future(ipo, spy, cut, seed=5)
    corrupted = _build(ipo_bad, spy_bad, **kwargs)["strategy_daily"]
    np.testing.assert_array_equal(corrupted.iloc[:cut].to_numpy(),
                                  full.iloc[:cut].to_numpy())


def test_each_segments_hedge_ratio_never_uses_data_from_its_own_entry_session_onward():
    ipo, spy = _pair(400, gap_vol=0.003)
    base = _build(ipo, spy)["trades_df"]
    for j in (0, 2, 5, len(base) - 3):
        entry = _pos(ipo, base["entry_date"].iloc[j])
        ipo_bad, spy_bad = _corrupt_future(ipo, spy, entry, seed=11 + j)
        bad = _build(ipo_bad, spy_bad)["trades_df"]
        np.testing.assert_array_equal(bad["hedge_ratio"].iloc[:j + 1].to_numpy(),
                                      base["hedge_ratio"].iloc[:j + 1].to_numpy())


# --------------------------------------------------------- beta neutrality

def test_scored_excess_is_beta_neutral_while_a_naked_short_carries_the_market_beta():
    """Criterion: the series ABI-1 actually scores (daily log excess over SPY)
    must have ~zero beta to SPY, and the naked-short design must NOT."""
    x = 0.25
    ipo, spy = _pair(900, k=1.3, spy_vol=0.01, noise=0.006, ipo_drift=-0.0002)
    curve = _build(ipo, spy, x=x, window=60)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    beta_bn = bno.scored_excess_spy_beta(curve["strategy_daily"], bench)["beta"]
    naked = bno.naked_short_account_daily(ipo)
    beta_naked = bno.scored_excess_spy_beta(naked, bench)["beta"]
    assert abs(beta_bn) / x < bno.MAX_RESIDUAL_BETA_PER_NOTIONAL
    assert beta_naked < -1.5          # -(1.3) hedge-less short beta, minus SPY itself
    assert abs(beta_bn) < 0.05 * abs(beta_naked)


def test_the_residual_beta_tolerance_is_a_quarter_of_the_unhedged_leg_beta():
    # documented pre-registered constant, not tuned after any score
    assert bno.MAX_RESIDUAL_BETA_PER_NOTIONAL == 0.25


def test_omitting_the_spy_hedge_would_leave_a_large_residual_beta():
    """Power check for the tolerance above: an overlay with NO hedge leg has
    a residual beta per notional near -beta_ipo, far outside the bound."""
    x = 0.25
    ipo, spy = _pair(900, k=1.3, spy_vol=0.01, noise=0.006)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    unhedged = bench - x * ipo["Close"].pct_change().fillna(0.0)
    residual = bno.scored_excess_spy_beta(unhedged, bench)["beta"] / x
    assert abs(residual) > 3 * bno.MAX_RESIDUAL_BETA_PER_NOTIONAL


# ------------------------------------------------- floor / tax / fail closed

def test_a_tripling_ipo_segment_stays_far_above_minus_100_percent():
    idx = _calendar(300)
    ipo, spy = _pair(index=idx)
    # tripling inside one calendar month (positions 160-180)
    closes = ipo["Close"].to_numpy().copy()
    closes[160:181] *= np.linspace(1.0, 3.0, 21)
    closes[181:] *= 3.0
    ipo = _ohlc(closes, idx)
    curve = _build(ipo, spy, x=0.25)
    assert curve["trades_df"]["net_ret"].min() > -0.6   # about -x*2 less the hedge leg
    assert curve["trades_df"]["net_ret"].min() < -0.2   # ...and the loss is real
    assert (curve["trades_df"]["net_ret"] > -1.0).all()


def test_built_curve_passes_the_netted_tax_walk_without_failing_closed():
    from mining import constants
    from mining.account_relative import netted_tax_factors
    ipo, spy = _pair(600, gap_vol=0.004)
    curve = _build(ipo, spy, ipo_cost=12.0, spy_cost=1.5)
    factors = netted_tax_factors(
        pretax_account_daily=curve["strategy_daily"], trades_df=curve["trades_df"],
        short_rate=constants.STCG_TAX_RATE, long_rate=constants.LTCG_TAX_RATE)
    assert len(factors) == len(ipo)
    assert (factors <= 1.0).all() and (factors > 0.0).all()
    assert (factors < 1.0).any()  # some overlay gain was actually taxed


def test_misaligned_calendars_fail_closed():
    ipo, spy = _pair(300)
    with pytest.raises(ValueError, match="calendar"):
        _build(ipo.drop(ipo.index[100]), spy)


def test_spy_history_that_does_not_cover_the_ipo_span_fails_closed():
    ipo, spy = _pair(300)
    with pytest.raises(ValueError, match="calendar"):
        _build(ipo, spy.iloc[:250])


def test_non_finite_or_non_positive_prices_fail_closed():
    ipo, spy = _pair(300)
    bad = ipo.copy()
    bad.iloc[50, bad.columns.get_loc("Close")] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _build(bad, spy)
    bad = ipo.copy()
    bad.iloc[50, bad.columns.get_loc("Open")] = 0.0
    with pytest.raises(ValueError, match="positive"):
        _build(bad, spy)


def test_a_dead_spy_window_after_warmup_fails_closed_instead_of_skipping_a_rebalance():
    ipo, spy = _pair(400)
    dead = spy.copy()
    flat = dead["Close"].iloc[150]
    dead.loc[dead.index[150:], ["Open", "High", "Low", "Close"]] = flat
    with pytest.raises(ValueError, match="hedge ratio is not finite"):
        _build(ipo, dead, window=40)


@pytest.mark.parametrize("kwargs", [
    {"x": 0.0}, {"x": 1.5}, {"x": float("nan")}, {"window": 2},
    {"ipo_cost": -1.0}, {"spy_cost": float("inf")}, {"entry_cost": -0.1},
])
def test_invalid_parameters_are_rejected(kwargs):
    ipo, spy = _pair(200)
    with pytest.raises(ValueError):
        _build(ipo, spy, **kwargs)
