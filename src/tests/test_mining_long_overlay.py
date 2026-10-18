"""Tests for src/mining/long_overlay.py -- the LONG-ONLY benchmark-plus-overlay
account builder behind treasury_month_end_index_demand_overlay (the long
analogue of mining.beta_neutral_overlay).

Small synthetic calendars are used throughout; the production month-end
schedule is tested in
test_batch5_treasury_month_end_index_demand_overlay.py.
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
from mining import long_overlay as lo  # noqa: E402
from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns  # noqa: E402

LEGS = ("AAA", "BBB", "CCC")


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


def _flat_spy(index, level=100.0):
    return _ohlc(np.full(len(index), level), index)


def _frames(index, *, spy=None, seeds=(11, 12, 13), vol=0.004, drift=0.0):
    spy_frame = _ohlc(_walk(index, 3, 0.01, 0.0003), index) if spy is None else spy
    legs = {sym: _ohlc(_walk(index, seed, vol, drift, 100.0 + 5 * k), index)
            for k, (sym, seed) in enumerate(zip(LEGS, seeds))}
    return spy_frame, legs


def _month_end_windows(index, window=3):
    """(entry, exit) pairs = the (window+1)-th-to-last and last session of every
    complete calendar month in a plain business-day index (test-local; the
    production schedule comes from the exchange calendar)."""
    out = []
    months = index.to_period("M")
    for period in months.unique():
        days = index[months == period]
        if len(days) > window:
            out.append((days[-(window + 1)], days[-1]))
    return out


def _rate(index, value=0.05):
    return pd.Series(value, index=index)


def _costs(a=2.0, b=10.0, c=4.0):
    return {"AAA": a, "BBB": b, "CCC": c}


def _build(spy, legs, windows, *, x=0.5, rate=None, costs=None, entry_cost=0.0):
    idx = lo.common_calendar(spy, legs)
    return lo.build_long_overlay_curve(
        spy, legs, windows,
        leg_cost_bps=_costs() if costs is None else costs,
        spy_entry_cost_bps=entry_cost,
        financing_rate_annual=_rate(idx) if rate is None else rate,
        overlay_notional=x)


# --------------------------------------------------------- financing rate

def test_financing_rate_is_the_coupon_equivalent_of_the_last_close_strictly_before():
    idx = _calendar(6)
    irx = pd.Series([3.0, 3.2, 3.4, 3.6, 3.8, 4.0], index=idx)   # percent
    rate = lo.causal_financing_rate(irx, idx)
    assert np.isnan(rate.iloc[0])
    for t in range(1, 6):
        expected = account_relative.coupon_equivalent_yield(irx.iloc[t - 1] / 100.0)
        assert rate.iloc[t] == pytest.approx(float(expected), rel=1e-12)


def test_financing_rate_never_reads_the_session_it_is_applied_to():
    idx = _calendar(8)
    irx = pd.Series(np.linspace(3.0, 3.7, 8), index=idx)
    base = lo.causal_financing_rate(irx, idx)
    for t in range(1, 8):
        bad = irx.copy()
        bad.iloc[t:] = 9.9                      # corrupt today and the future
        corrupted = lo.causal_financing_rate(bad, idx)
        assert corrupted.iloc[t] == base.iloc[t]


def test_financing_rate_carries_the_last_print_across_a_hole():
    idx = _calendar(6)
    irx = pd.Series([3.0, 3.2, 3.4, 3.6, 3.8, 4.0], index=idx).drop(idx[3])
    rate = lo.causal_financing_rate(irx, idx)
    # session 4 (idx[4]): the previous PUBLISHED close is idx[2] (3.4), idx[3] is missing
    assert rate.iloc[3] == pytest.approx(
        float(account_relative.coupon_equivalent_yield(0.034)), rel=1e-12)
    assert rate.iloc[4] == pytest.approx(
        float(account_relative.coupon_equivalent_yield(0.034)), rel=1e-12)


def test_negative_bill_print_passes_through_with_its_sign():
    idx = _calendar(4)
    irx = pd.Series([1.0, -0.105, 0.5, 0.5], index=idx)
    rate = lo.causal_financing_rate(irx, idx)
    assert rate.iloc[2] < 0.0          # the negative print is applied, never clipped


# ---------------------------------------------------------- common calendar

def test_common_calendar_trims_only_the_leading_edge_to_the_legs_common_span():
    """A later-listing leg (AGG lists after IEF and TLT) legitimately shortens the
    account's LEADING edge; SPY's longer lead history is sliced to the span."""
    idx = _calendar(40)
    spy, legs = _frames(idx)
    late = {**legs, "CCC": legs["CCC"].iloc[5:]}            # CCC lists 5 sessions later
    common = lo.common_calendar(spy, late)
    assert common[0] == idx[5] and common[-1] == idx[-1]
    assert common.equals(idx[5:])


def _short_tail(symbols, sessions_short, idx, spy, legs):
    """Cut ``sessions_short`` trailing bars off each named series."""
    frames = {"SPY": spy, **legs}
    for symbol in symbols:
        frames[symbol] = frames[symbol].iloc[:-sessions_short]
    return frames.pop("SPY"), frames


@pytest.mark.parametrize("short_series", [
    ("BBB",),                  # one leg lacks the trailing bar the others have
    ("AAA", "CCC"),            # two legs lack it while the third leg and SPY have it
    ("AAA", "BBB", "CCC"),     # every leg lacks it and only SPY has it
    ("SPY",),                  # SPY lacks it
])
def test_common_calendar_refuses_unequal_trailing_coverage_instead_of_trimming_it(short_series):
    idx = _calendar(40)
    spy, legs = _frames(idx)
    spy, legs = _short_tail(short_series, 2, idx, spy, legs)
    with pytest.raises(lo.CalendarCoverageError) as caught:
        lo.common_calendar(spy, legs)
    error = caught.value
    assert isinstance(error, ValueError)                    # every existing caller's except clause still works
    assert error.reason_code == "unequal_trailing_coverage"
    expected = [d.strftime("%Y-%m-%d") for d in idx[-2:]]   # the sessions the shorter series lack
    assert error.missing_dates == expected
    for date in expected:
        assert date in str(error)
    for symbol in short_series:
        assert symbol in str(error)


def test_common_calendar_accepts_equal_trailing_coverage_with_different_leading_history():
    idx = _calendar(40)
    spy, legs = _frames(idx)
    legs = {**legs, "AAA": legs["AAA"].iloc[7:], "CCC": legs["CCC"].iloc[3:]}
    assert lo.common_calendar(spy, legs).equals(idx[7:])


def test_common_calendar_refuses_an_interior_hole_in_one_leg():
    idx = _calendar(40)
    spy, legs = _frames(idx)
    legs = {**legs, "BBB": legs["BBB"].drop(idx[20])}
    with pytest.raises(ValueError, match="identical calendar"):
        lo.common_calendar(spy, legs)


def test_a_calendar_mismatch_error_names_the_first_differing_dates():
    """A vendor hole is fixed by refetching, so the refusal has to say WHICH
    session differs and which side has it."""
    idx = _calendar(40)
    spy, legs = _frames(idx)
    with pytest.raises(ValueError) as leg_error:
        lo.common_calendar(spy, {**legs, "BBB": legs["BBB"].drop(idx[20])})
    message = str(leg_error.value)
    assert idx[20].strftime("%Y-%m-%d") in message and "BBB" in message and "AAA" in message
    with pytest.raises(ValueError) as spy_error:
        lo.common_calendar(spy.drop(idx[21]), legs)
    assert idx[21].strftime("%Y-%m-%d") in str(spy_error.value) and "SPY" in str(spy_error.value)


def test_an_interior_mismatch_carries_a_reason_code_and_every_differing_date():
    idx = _calendar(40)
    spy, legs = _frames(idx)
    gone = [idx[20], idx[25]]
    with pytest.raises(lo.CalendarCoverageError) as leg_error:
        lo.common_calendar(spy, {**legs, "BBB": legs["BBB"].drop(gone)})
    assert leg_error.value.reason_code == "calendar_mismatch"
    assert leg_error.value.missing_dates == [d.strftime("%Y-%m-%d") for d in gone]
    with pytest.raises(lo.CalendarCoverageError) as spy_error:
        lo.common_calendar(spy.drop(idx[21]), legs)
    assert spy_error.value.reason_code == "calendar_mismatch"
    assert spy_error.value.missing_dates == [idx[21].strftime("%Y-%m-%d")]


def test_common_calendar_refuses_a_spy_calendar_that_differs_inside_the_span():
    idx = _calendar(40)
    spy, legs = _frames(idx)
    with pytest.raises(ValueError, match="SPY"):
        lo.common_calendar(spy.drop(idx[20]), legs)


# ------------------------------------------------- hand-computed one window

def _hand_setup():
    idx = _calendar(12)                    # Mon 2024-03-04 .. Tue 2024-03-19
    spy = _flat_spy(idx)                   # zero core return, so equity path is the overlay alone
    closes = {
        "AAA": [100, 100, 100, 100, 101.0, 99.0, 103.0, 103, 103, 103, 103, 103],
        "BBB": [100, 100, 100, 100, 100.0, 101.0, 102.0, 102, 102, 102, 102, 102],
        "CCC": [100, 100, 100, 100, 102.0, 102.0, 101.0, 101, 101, 101, 101, 101],
    }
    legs = {sym: _ohlc(np.array(vals, dtype=float), idx) for sym, vals in closes.items()}
    # entry = Thu 2024-03-07 (idx[3]); window sessions Fri 8th, Mon 11th, Tue 12th
    windows = [(idx[3], idx[6])]
    return idx, spy, legs, windows, closes


def test_one_window_matches_an_independent_closed_form():
    idx, spy, legs, windows, closes = _hand_setup()
    x = 0.5
    rate_annual = pd.Series(np.linspace(0.03, 0.07, len(idx)), index=idx)   # a DIFFERENT rate every session
    costs = _costs()
    curve = _build(spy, legs, windows, x=x, rate=rate_annual, costs=costs)
    equity = 1.0                              # flat SPY core, nothing earned before entry
    shares = {s: (x / 3.0) * equity / closes[s][3] for s in LEGS}
    daily_expected = np.zeros(len(idx))
    days = {4: 1.0, 5: 3.0, 6: 1.0}           # Fri (1), Mon (3, spans the weekend), Tue (1)
    for t in (4, 5, 6):
        dollars = sum(shares[s] * (closes[s][t] - closes[s][t - 1]) for s in LEGS)
        dollars -= x * 1.0 * rate_annual.iloc[t] * days[t] / 365.0
        if t == 6:
            dollars -= x * (sum(costs.values()) / 3.0) / 1e4
        daily_expected[t] = dollars / equity
        equity += dollars
    np.testing.assert_allclose(curve["strategy_daily"].to_numpy(), daily_expected,
                               rtol=0, atol=1e-15)
    trades = curve["trades_df"]
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_date"] == idx[3] and row["exit_date"] == idx[6]
    assert row["held"] == 3 and row["reason"] == "window"
    gross = sum(shares[s] * (closes[s][6] - closes[s][3]) for s in LEGS)
    fin = x * sum(rate_annual.iloc[t] * days[t] for t in (4, 5, 6)) / 365.0
    cost = x * (sum(costs.values()) / 3.0) / 1e4
    assert row["gross_ret"] == pytest.approx(gross, abs=1e-15)
    assert row["financing_ret"] == pytest.approx(fin, abs=1e-15)
    assert row["cost_ret"] == pytest.approx(cost, abs=1e-15)
    assert row["net_ret"] == pytest.approx(gross - fin - cost, abs=1e-15)
    assert row["size"] == pytest.approx(1.0, abs=1e-15)


def test_overlay_mask_is_exactly_the_sessions_after_entry_through_exit():
    idx, spy, legs, windows, _ = _hand_setup()
    curve = _build(spy, legs, windows)
    mask = curve["overlay_mask"]
    assert mask.dtype == bool and mask.index.equals(idx)
    assert list(np.flatnonzero(mask.to_numpy())) == [4, 5, 6]    # entry day itself is NOT held


def test_account_equals_the_benchmark_series_exactly_outside_the_windows():
    idx = _calendar(200)
    spy, legs = _frames(idx)
    curve = _build(spy, legs, _month_end_windows(idx), entry_cost=0.3)
    core = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.3).reindex(idx)
    outside = ~curve["overlay_mask"].to_numpy()
    assert outside.sum() > 100 and (~outside).sum() >= 9
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy()[outside],
                                  core.to_numpy()[outside])
    # ...and the overlay is genuinely live inside them
    assert (curve["strategy_daily"].to_numpy()[~outside] != core.to_numpy()[~outside]).all()


def test_account_equals_the_evaluators_benchmark_outside_the_windows_when_spy_history_is_longer_than_the_legs():
    """PRODUCTION SHAPE: SPY lists years before the legs (SPY from 1993, AGG from
    2003) and the ABI-1 evaluator computes its benchmark series on SPY's FULL
    history. The account's core must equal that series bit for bit on every
    account session outside the windows. A core computed from SPY SLICED to the
    account span would replace the first account day's real SPY return with
    minus the entry cost, and the account would silently differ from the
    benchmark on a day no window covers."""
    lead = 100
    full_idx = _calendar(300)
    idx = full_idx[lead:]
    spy_full = _ohlc(_walk(full_idx, 3, 0.01, 0.0003), full_idx)
    legs = {sym: _ohlc(_walk(idx, seed, 0.004, 0.0, 100.0 + 5 * k), idx)
            for k, (sym, seed) in enumerate(zip(LEGS, (11, 12, 13)))}
    entry_cost = 0.3
    curve = lo.build_long_overlay_curve(
        spy_full, legs, _month_end_windows(idx), leg_cost_bps=_costs(),
        spy_entry_cost_bps=entry_cost, financing_rate_annual=_rate(idx),
        overlay_notional=0.5)
    evaluator_benchmark = cost_adjusted_benchmark_daily_returns(
        spy_full, entry_cost_bps=entry_cost).reindex(idx)
    outside = ~curve["overlay_mask"].to_numpy()
    assert curve["strategy_daily"].index.equals(idx)
    assert outside.sum() > 100 and outside[0]           # day one is outside every window
    # the fixture has power: a core built from SPY sliced to the span would
    # differ from the evaluator's benchmark on that very first account day
    sliced = cost_adjusted_benchmark_daily_returns(
        spy_full.loc[idx[0]:idx[-1]], entry_cost_bps=entry_cost)
    assert sliced.iloc[0] != evaluator_benchmark.iloc[0]
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy()[outside],
                                  evaluator_benchmark.to_numpy()[outside])
    np.testing.assert_array_equal(curve["core_daily"].to_numpy(),
                                  evaluator_benchmark.to_numpy())


def test_the_overlay_is_held_continuously_through_the_whole_window():
    """A window flag that fired on one bar only would make an evaluation slice
    that starts mid-window blind to the carried-in position."""
    idx = _calendar(120)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    curve = _build(spy, legs, windows)
    mask = curve["overlay_mask"].to_numpy()
    for entry, exit_ in windows:
        e, x_ = idx.get_loc(entry), idx.get_loc(exit_)
        assert mask[e + 1:x_ + 1].all() and not mask[e]
        assert curve["overlay_daily"].to_numpy()[e + 1:x_ + 1].std() > 0.0


def test_sizing_uses_the_account_equity_at_the_entry_close_and_compounds_daily():
    idx = _calendar(160)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    x, rate_annual, entry_cost = 0.5, 0.03, 0.4
    costs = _costs(1.0, 6.0, 3.0)
    curve = _build(spy, legs, windows, x=x, rate=_rate(idx, rate_annual),
                   costs=costs, entry_cost=entry_cost)
    core = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=entry_cost).reindex(idx).to_numpy()
    close = {s: legs[s]["Close"].to_numpy() for s in LEGS}
    elapsed = np.concatenate([[1.0], np.diff(idx.values.astype("datetime64[D]")).astype(float)])
    entries = {idx.get_loc(e): idx.get_loc(x_) for e, x_ in windows}
    equity, daily = 1.0, np.zeros(len(idx))
    live = None
    for t in range(len(idx)):
        e_prev = equity
        dollars = 0.0
        if live is not None:
            size, shares, exit_pos = live
            dollars = sum(shares[s] * (close[s][t] - close[s][t - 1]) for s in LEGS)
            dollars -= x * size * rate_annual * elapsed[t] / 365.0
            if t == exit_pos:
                dollars -= size * x * sum(costs.values()) / 3.0 / 1e4
                live = None
        daily[t] = core[t] + dollars / e_prev
        equity = e_prev * (1.0 + daily[t])
        if t in entries:
            live = (equity, {s: (x / 3.0) * equity / close[s][t] for s in LEGS}, entries[t])
    np.testing.assert_allclose(curve["strategy_daily"].to_numpy(), daily, rtol=0, atol=1e-13)


def test_the_overlay_is_long_only_and_carries_no_hedge_or_short_leg():
    """Rising legs raise the account above the benchmark inside the window and
    falling legs lower it; and the overlay's P&L per unit of sizing equity does
    not depend on what SPY does (no SPY hedge leg)."""
    idx = _calendar(120)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    up = {s: _ohlc(100.0 * np.cumprod(np.full(len(idx), 1.004)), idx) for s in LEGS}
    down = {s: _ohlc(100.0 * np.cumprod(np.full(len(idx), 0.996)), idx) for s in LEGS}
    core = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0).reindex(idx)
    m = _build(spy, up, windows, costs=_costs(0, 0, 0), rate=_rate(idx, 0.0))["overlay_mask"].to_numpy()
    excess_up = _build(spy, up, windows, costs=_costs(0, 0, 0), rate=_rate(idx, 0.0))["strategy_daily"] - core
    excess_dn = _build(spy, down, windows, costs=_costs(0, 0, 0), rate=_rate(idx, 0.0))["strategy_daily"] - core
    assert (excess_up.to_numpy()[m] > 0).all() and (excess_dn.to_numpy()[m] < 0).all()
    spy_b = _ohlc(_walk(idx, 99, 0.02, -0.001), idx)                # a very different SPY path
    a = _build(spy, legs, windows)["trades_df"]["gross_ret"].to_numpy()
    b = _build(spy_b, legs, windows)["trades_df"]["gross_ret"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)


def test_equal_thirds_and_total_notional_scale_the_price_pnl_linearly():
    idx = _calendar(80)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    z = dict(costs=_costs(0, 0, 0), rate=_rate(idx, 0.0))
    g1 = _build(spy, legs, windows, x=0.25, **z)["trades_df"]["gross_ret"].to_numpy()
    g2 = _build(spy, legs, windows, x=0.5, **z)["trades_df"]["gross_ret"].to_numpy()
    np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-12)
    # one leg's move counts exactly one third of the overlay
    flat = {s: _ohlc(np.full(len(idx), 100.0), idx) for s in LEGS}
    moved = {**flat, "BBB": legs["BBB"]}
    only_b = _build(spy, moved, windows, x=0.5, **z)["trades_df"]
    b_ret = [legs["BBB"]["Close"].loc[x_] / legs["BBB"]["Close"].loc[e] - 1.0 for e, x_ in windows]
    np.testing.assert_allclose(only_b["gross_ret"].to_numpy(),
                               np.array(b_ret) * (0.5 / 3.0), rtol=1e-12)


def test_execution_cost_is_a_full_round_trip_per_leg_charged_on_the_exit_day():
    idx, spy, legs, windows, closes = _hand_setup()
    zero = _build(spy, legs, windows, costs=_costs(0, 0, 0), rate=_rate(idx, 0.0))
    costed = _build(spy, legs, windows, costs=_costs(2.0, 10.0, 4.0), rate=_rate(idx, 0.0))
    diff = zero["strategy_daily"].to_numpy() - costed["strategy_daily"].to_numpy()
    assert np.flatnonzero(np.abs(diff) > 0).tolist() == [6]
    charge = 0.5 * (2.0 + 10.0 + 4.0) / 3.0 / 1e4          # x times the mean round-trip cost
    d = zero["strategy_daily"].to_numpy()
    equity_before_exit = (1.0 + d[4]) * (1.0 + d[5])        # flat core, so this is the overlay's equity path
    assert diff[6] == pytest.approx(charge / equity_before_exit, rel=1e-12)
    row = costed["trades_df"].iloc[0]
    assert row["cost_ret"] == pytest.approx(charge, abs=1e-15)


def test_financing_accrues_on_calendar_days_including_the_weekend():
    idx, spy, legs, windows, _ = _hand_setup()
    flat = {s: _ohlc(np.full(len(idx), 100.0), idx) for s in LEGS}
    curve = _build(spy, flat, windows, costs=_costs(0, 0, 0), rate=_rate(idx, 0.073))
    daily = curve["strategy_daily"].to_numpy()
    per_day = 0.5 * 0.073 / 365.0
    assert daily[4] == pytest.approx(-per_day * 1.0, abs=1e-15)     # Fri: 1 day
    assert daily[5] == pytest.approx(-per_day * 3.0 / (1.0 - per_day), abs=1e-15)   # Mon: 3 days
    assert curve["trades_df"]["financing_ret"].iloc[0] == pytest.approx(per_day * 5.0, abs=1e-15)


def test_a_missing_financing_rate_on_a_financed_session_fails_closed_but_not_elsewhere():
    idx, spy, legs, windows, _ = _hand_setup()
    rate = _rate(idx, 0.04)
    ok = rate.copy()
    ok.iloc[0] = np.nan                       # never financed: fine
    _build(spy, legs, windows, rate=ok)
    bad = rate.copy()
    bad.iloc[5] = np.nan                      # a window session: refuse
    with pytest.raises(ValueError, match="financing"):
        _build(spy, legs, windows, rate=bad)


def test_the_final_window_is_closed_at_the_last_close_when_its_exit_is_after_the_data():
    idx, spy, legs, _, closes = _hand_setup()
    windows = [(idx[8], idx[-1] + pd.Timedelta(days=4))]      # scheduled exit beyond the data
    curve = _build(spy, legs, windows, costs=_costs(2, 10, 4), rate=_rate(idx, 0.05))
    row = curve["trades_df"].iloc[0]
    assert row["exit_date"] == idx[-1] and row["reason"] == "end_of_data"
    assert row["held"] == len(idx) - 1 - 8
    assert row["cost_ret"] > 0.0                              # the round trip is charged at the last bar


def test_a_window_entered_on_the_very_last_bar_opens_nothing():
    idx, spy, legs, _, _ = _hand_setup()
    curve = _build(spy, legs, [(idx[-1], idx[-1] + pd.Timedelta(days=4))])
    assert len(curve["trades_df"]) == 0 and not curve["overlay_mask"].any()
    assert list(curve["trades_df"].columns) == list(lo.TRADES_COLUMNS)
    core = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0).reindex(idx)
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy(), core.to_numpy())


def test_no_windows_returns_the_benchmark_series_untouched():
    idx = _calendar(50)
    spy, legs = _frames(idx)
    curve = _build(spy, legs, [], entry_cost=0.25)
    core = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.25).reindex(idx)
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy(), core.to_numpy())
    assert curve["in_market"].all() and len(curve["trades_df"]) == 0


def test_a_non_finite_benchmark_core_fails_closed(monkeypatch):
    """The account core IS the benchmark series; if that series ever came back
    non-finite the builder must refuse rather than carry NaN through equity."""
    idx = _calendar(60)
    spy, legs = _frames(idx)
    monkeypatch.setattr(lo, "cost_adjusted_benchmark_daily_returns",
                        lambda frame, entry_cost_bps=0.0: pd.Series(np.nan, index=frame.index))
    with pytest.raises(ValueError, match="benchmark core"):
        _build(spy, legs, _month_end_windows(idx))


def test_curve_shape_matches_the_abi1_builder_contract():
    idx = _calendar(120)
    spy, legs = _frames(idx)
    curve = _build(spy, legs, _month_end_windows(idx))
    assert set(("strategy_daily", "in_market", "trades_df")) <= set(curve)
    assert curve["strategy_daily"].index.equals(idx) and curve["in_market"].index.equals(idx)
    assert curve["in_market"].dtype == bool and curve["in_market"].all()
    assert np.isfinite(curve["strategy_daily"].to_numpy()).all()
    for column in ("entry_date", "exit_date", "held", "reason", "gross_ret",
                   "financing_ret", "cost_ret", "net_ret", "size"):
        assert column in curve["trades_df"].columns


def test_curve_declares_itself_a_benchmark_plus_overlay_account():
    """Risk-basis rule: the risk layer of such an account is measured on the
    overlay's own excess. `abi1_driver.evaluate_family` reads this declaration,
    so a builder that forgets it silently reverts to the whole-account basis."""
    idx = _calendar(120)
    spy, legs = _frames(idx)
    curve = _build(spy, legs, _month_end_windows(idx))
    assert curve["account_kind"] == account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY
    assert curve["account_kind"] == "benchmark_plus_overlay"


def test_the_next_open_sensitivity_curve_declares_the_same_account_kind():
    idx = _calendar(120)
    spy, legs = _frames(idx)
    curve = lo.build_next_open_sensitivity_curve(
        spy, legs, _month_end_windows(idx), leg_cost_bps=_costs(),
        spy_entry_cost_bps=0.0, financing_rate_annual=_rate(idx),
        overlay_notional=0.5)
    assert curve["account_kind"] == "benchmark_plus_overlay"


# ------------------------------------------------------------- validation

@pytest.mark.parametrize("x", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_overlay_notional_must_be_a_finite_fraction_up_to_regt_gross(x):
    idx, spy, legs, windows, _ = _hand_setup()
    with pytest.raises(ValueError, match="overlay_notional"):
        _build(spy, legs, windows, x=x)


def test_costs_must_be_finite_nonnegative_and_cover_exactly_the_legs():
    idx, spy, legs, windows, _ = _hand_setup()
    with pytest.raises(ValueError, match="leg_cost_bps"):
        _build(spy, legs, windows, costs={"AAA": 1.0, "BBB": 1.0})
    with pytest.raises(ValueError, match="leg_cost_bps"):
        _build(spy, legs, windows, costs={"AAA": 1.0, "BBB": 1.0, "CCC": -1.0})
    with pytest.raises(ValueError, match="leg_cost_bps"):
        _build(spy, legs, windows, costs={"AAA": 1.0, "BBB": 1.0, "CCC": float("nan")})
    with pytest.raises(ValueError, match="spy_entry_cost_bps"):
        _build(spy, legs, windows, entry_cost=-0.1)


def test_window_dates_must_exist_be_ordered_and_not_overlap():
    idx, spy, legs, windows, _ = _hand_setup()
    weekend = pd.Timestamp("2024-03-09")                    # a Saturday, not a session
    with pytest.raises(ValueError, match="not a session"):
        _build(spy, legs, [(weekend, idx[6])])
    with pytest.raises(ValueError, match="not a session"):
        _build(spy, legs, [(idx[3], weekend)])
    with pytest.raises(ValueError, match="before its exit"):
        _build(spy, legs, [(idx[6], idx[3])])
    with pytest.raises(ValueError, match="before its exit"):
        _build(spy, legs, [(idx[3], idx[3])])
    with pytest.raises(ValueError, match="overlap|ascending"):
        _build(spy, legs, [(idx[3], idx[6]), (idx[5], idx[9])])
    with pytest.raises(ValueError, match="overlap|ascending"):
        _build(spy, legs, [(idx[7], idx[9]), (idx[2], idx[5])])


def test_legs_and_spy_must_be_clean_ohlc_frames():
    idx, spy, legs, windows, _ = _hand_setup()
    bad = legs["AAA"].copy()
    bad.iloc[4, bad.columns.get_loc("Close")] = np.nan
    with pytest.raises(ValueError, match="AAA"):
        _build(spy, {**legs, "AAA": bad}, windows)
    neg = legs["BBB"].copy()
    neg.iloc[4, neg.columns.get_loc("Open")] = -1.0
    with pytest.raises(ValueError, match="BBB"):
        _build(spy, {**legs, "BBB": neg}, windows)
    with pytest.raises(ValueError, match="spy"):
        _build(spy.iloc[:0], legs, windows)


# -------------------------------------------------- next-open sensitivity

def test_the_scored_builder_exposes_no_fill_convention_argument():
    """The engine's next-open convention is a DISCLOSED SENSITIVITY, never a
    selectable option of the scored curve."""
    names = set(inspect.signature(lo.build_long_overlay_curve).parameters)
    assert names == {"spy_frame", "leg_frames", "windows", "leg_cost_bps",
                     "spy_entry_cost_bps", "financing_rate_annual", "overlay_notional"}


def test_next_open_sensitivity_buys_at_the_open_of_the_first_window_session():
    idx = _calendar(12)
    spy = _flat_spy(idx)
    gap = np.array([0, 0, 0, 0, 0.01, -0.005, 0.002] + [0] * 5)
    closes = {}
    for k, s in enumerate(LEGS):
        pre = 100.0 + 10.0 * k
        closes[s] = np.concatenate([np.full(4, pre), np.full(3, pre * (1.02 + 0.01 * k)),
                                    np.full(5, pre * 1.03)])
    legs = {s: _ohlc(closes[s], idx, gap=gap) for s in LEGS}
    windows = [(idx[3], idx[6])]
    z = dict(costs=_costs(0, 0, 0), rate=_rate(idx, 0.0))
    scored = _build(spy, legs, windows, **z)
    sens = lo.build_next_open_sensitivity_curve(
        spy, legs, windows, leg_cost_bps=_costs(0, 0, 0), spy_entry_cost_bps=0.0,
        financing_rate_annual=_rate(idx, 0.0), overlay_notional=0.5)
    opens = {s: legs[s]["Open"].to_numpy() for s in LEGS}
    shares = {s: (0.5 / 3.0) / opens[s][4] for s in LEGS}
    d4 = sum(shares[s] * (closes[s][4] - opens[s][4]) for s in LEGS)
    assert d4 != 0.0
    assert sens["strategy_daily"].iloc[4] == pytest.approx(d4, abs=1e-15)
    assert sens["strategy_daily"].iloc[3] == 0.0                   # nothing is held at the entry close
    assert scored["strategy_daily"].iloc[4] != sens["strategy_daily"].iloc[4]
    assert sens["trades_df"]["entry_date"].iloc[0] == idx[4]
    assert list(np.flatnonzero(sens["overlay_mask"].to_numpy())) == [4, 5, 6]
    # both end on the same exit close
    assert sens["trades_df"]["exit_date"].iloc[0] == scored["trades_df"]["exit_date"].iloc[0]


# --------------------------- prefix invariance / look-ahead (with power)

_ZERO = dict(costs=_costs(0, 0, 0))


def _prefix_violations(idx, spy, legs, windows, cutoffs, **build_kw):
    """Curve computed on frames cut at each cutoff must equal the full-history
    curve on EVERY session up to and including the cutoff (costs are zero in
    these runs, so a truncated window's round-trip charge cannot differ)."""
    full = _build(spy, legs, windows, **build_kw)["strategy_daily"]
    bad = []
    for c in cutoffs:
        cut = idx[:c + 1]
        part = _build(spy.loc[cut], {s: f.loc[cut] for s, f in legs.items()},
                      [w for w in windows if w[0] <= cut[-1]], **build_kw)["strategy_daily"]
        if not np.array_equal(part.to_numpy(), full.to_numpy()[:c + 1]):
            bad.append(c)
    return bad


def _peeking(real_build):
    """Faithful copy of the scored builder with the overlay P&L marked one
    session EARLY (session t carries the t -> t+1 overlay move)."""
    def peeking_build(spy_frame, leg_frames, windows_, **kw):
        curve = real_build(spy_frame, leg_frames, windows_, **kw)
        peeked = curve["overlay_daily"].shift(-1).fillna(0.0)
        out = dict(curve)
        out["strategy_daily"] = curve["core_daily"].reindex(curve["strategy_daily"].index) + peeked
        return out
    return peeking_build


def _cutoffs_inside_and_around_windows(idx, windows, n=5):
    cutoffs = []
    for entry, _ in windows[:n]:
        e = idx.get_loc(entry)
        cutoffs += [e, e + 1, e + 2, e + 3, e + 4]      # entry day, all 3 window sessions, first day after
    return cutoffs


def test_curve_is_prefix_invariant_with_cutoffs_one_to_three_sessions_after_each_entry():
    idx = _calendar(150)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    assert _prefix_violations(idx, spy, legs, windows,
                              _cutoffs_inside_and_around_windows(idx, windows), **_ZERO) == []


def test_prefix_invariance_test_has_power_against_a_one_session_lookahead(monkeypatch):
    idx = _calendar(150)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    monkeypatch.setattr(lo, "build_long_overlay_curve", _peeking(lo.build_long_overlay_curve))
    caught = _prefix_violations(idx, spy, legs, windows,
                                _cutoffs_inside_and_around_windows(idx, windows), **_ZERO)
    assert len(caught) >= 5          # every window is caught, not just one lucky cutoff


def test_cutoffs_away_from_the_windows_would_miss_the_lookahead(monkeypatch):
    """Why the cutoffs above sit AT and AFTER each entry: a cut in the quiet
    middle of a month has nothing held on either side of it, so a look-ahead
    in the window's own sessions is invisible to it."""
    idx = _calendar(150)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    monkeypatch.setattr(lo, "build_long_overlay_curve", _peeking(lo.build_long_overlay_curve))
    quiet = []
    for (e0, x0), (e1, _) in zip(windows[:4], windows[1:5]):
        quiet.append((idx.get_loc(x0) + idx.get_loc(e1)) // 2)   # halfway between one exit and the next entry
    assert _prefix_violations(idx, spy, legs, windows, quiet, **_ZERO) == []


def test_the_truncated_final_window_differs_from_the_full_run_only_by_its_round_trip_charge():
    idx = _calendar(150)
    spy, legs = _frames(idx)
    windows = _month_end_windows(idx)
    entry, _ = windows[3]
    e = idx.get_loc(entry)
    cut = idx[:e + 2]
    full = _build(spy, legs, windows)
    part = _build(spy.loc[cut], {s: f.loc[cut] for s, f in legs.items()},
                  [w for w in windows if w[0] <= cut[-1]])
    np.testing.assert_array_equal(part["strategy_daily"].to_numpy()[:e + 1],
                                  full["strategy_daily"].to_numpy()[:e + 1])
    diff = part["strategy_daily"].to_numpy()[e + 1] - full["strategy_daily"].to_numpy()[e + 1]
    # the sizing equity is the entry-close equity, which is also e_prev for the
    # first window session, so the whole difference is x * mean(cost) / 1e4
    assert diff == pytest.approx(-0.5 * (sum(_costs().values()) / 3.0) / 1e4, abs=1e-15)


# ------------------------------ how the ABI-1 risk layer treats such an account
#
# Risk-basis rule: for a curve that declares
# account_kind == "benchmark_plus_overlay", ABI-1 measures the OVERLAY-ONLY
# EXCESS account (the daily arithmetic excess over SPY), not the WHOLE
# account. The whole account holds SPY at all times, so on that basis a
# benchmark that draws down more than the 0.35 absolute cap would cap every
# such account at cut_risk_refusal whatever the overlay earned. The
# whole-account facts below are still TRUE and are pinned by calling
# risk_refusals on the whole account directly: they are what the rule
# deliberately does not gate on.

def _crash_account(*, losing_windows_in_crash: bool):
    """(ABI-1 result, curve, benchmark daily returns) for an SPY-core account
    whose SPY suffers a ~65 percent crash (sessions 300-420). The overlay legs
    rise steadily inside every window and are flat outside it (a consistently
    profitable overlay); with ``losing_windows_in_crash`` the windows that
    ENTER inside the crash fall instead (a few ordinary losing windows landing
    in SPY's worst stretch)."""
    from mining import abi1_driver
    n = 1500
    idx = _calendar(n, "2010-01-04")
    r = np.full(n, 0.0004)
    r[300:420] = -0.0088                     # a ~65 percent crash inside the account's life
    r[0] = 0.0
    spy = _ohlc(100.0 * np.cumprod(1.0 + r), idx)
    windows = _month_end_windows(idx)
    legs = {}
    for k, sym in enumerate(LEGS):
        step = np.zeros(n)
        for entry, exit_ in windows:
            e, x_ = idx.get_loc(entry), idx.get_loc(exit_)
            sign = -1.0 if losing_windows_in_crash and 300 <= e <= 420 else 1.0
            step[e + 1:x_ + 1] = sign * 0.004 * (1 + 0.1 * k)
        step += np.random.RandomState(40 + k).normal(0.0, 0.0002, n)
        step[0] = 0.0
        legs[sym] = _ohlc(100.0 * np.cumprod(1.0 + step), idx)
    curve = _build(spy, legs, windows, costs=_costs(0.5, 0.5, 0.5), rate=_rate(idx, 0.02))
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy, benchmark_entry_cost_bps=0.0,
        account_type="taxable", tax_policy="realization")
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.0)
    return result, curve, bench


def test_an_spy_core_overlay_account_is_judged_on_its_overlay_only_excess_not_the_whole_account():
    result, _, bench = _crash_account(losing_windows_in_crash=False)
    assert result["layer2"]["account_basis"] == "overlay_only_excess"
    assert result["layer2"]["drawdown"]["absolute_passed"] is True
    # the absolute cap still refuses the WHOLE account
    whole = account_relative.risk_refusals(result["account_daily"], bench)
    assert whole["account_basis"] == "whole_account"
    assert whole["drawdown"]["absolute_passed"] is False
    assert whole["drawdown"]["account"] < -0.35
    # ...but it does not cap the verdict: the edge itself decides
    assert result["layer1"]["dsr_value"] >= result["dsr_pass_threshold"]
    assert result["verdict"] == "survived"


def test_an_overlay_that_never_loses_passes_all_four_refusals_on_the_excess_account():
    result, curve, bench = _crash_account(losing_windows_in_crash=False)
    layer2 = result["layer2"]
    assert layer2["drawdown"]["passed"] is True
    assert layer2["red_week_streak"]["passed"] is True
    assert layer2["worst_stretch"]["passed"] is True
    assert layer2["all_refusals_passed"] is True
    # On the WHOLE account the three SPY-relative refusals pass by construction
    # (SPY times a non-decreasing overlay factor can never be worse than SPY);
    # only the absolute cap refuses it.
    whole = account_relative.risk_refusals(result["account_daily"], bench)
    assert whole["drawdown"]["relative_passed"] is True
    assert whole["red_week_streak"]["passed"] is True
    assert whole["worst_stretch"]["passed"] is True
    assert whole["drawdown"]["absolute_passed"] is False
    # Outside the overlay windows the excess account earns EXACTLY zero: float
    # noise there would read as spurious losing weeks in the streak refusal.
    excess = result["account_daily"] - bench
    assert (excess[~curve["overlay_mask"]] == 0.0).all()


def test_a_few_losing_windows_inside_spys_worst_stretch_trip_the_whole_account_refusals_but_not_the_excess_ones():
    """A fact about the whole account: an overlay that earns in every other
    window but loses in the windows that fall inside SPY's own crash is
    refused by the SPY-relative drawdown and worst-stretch comparisons on the
    WHOLE account. On the overlay's own excess the same few small losses are
    nowhere near SPY's drawdown, streak or worst stretch, so the three
    SPY-relative refusals do not bite: this is what the risk-basis rule
    produces."""
    result, _, bench = _crash_account(losing_windows_in_crash=True)
    whole = account_relative.risk_refusals(result["account_daily"], bench)
    assert whole["drawdown"]["relative_passed"] is False
    assert whole["worst_stretch"]["passed"] is False
    assert whole["all_refusals_passed"] is False
    layer2 = result["layer2"]
    assert layer2["account_basis"] == "overlay_only_excess"
    assert layer2["drawdown"]["account"] < 0.0          # the losing windows ARE seen...
    assert layer2["drawdown"]["account"] > -0.35        # ...and are ordinary
    assert layer2["all_refusals_passed"] is True
    assert result["verdict"] == "survived"
