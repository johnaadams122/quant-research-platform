"""Tests for mining.account_relative -- ABI-1 account-relative scoring.

All data is deterministic and synthetic; no network, no clock. The only file
read is a temporary IRX.csv written by the test itself into tmp_path.

Convention note: mining tests build their own module-level `_`-prefixed
frame builders rather than sharing a fixture module (the same convention
test_mining_panel.py's module docstring states).
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from mining import account_relative as ar  # noqa: E402
from mining import constants  # noqa: E402


# ---------------------------------------------------------------- fixtures

def _idx(n=300, start="2020-01-02"):
    return pd.bdate_range(start, periods=n)


def _irx(tmp_path, rate=0.0504, start="2019-01-01", n=2000):
    """Write a minimal IRX.csv and return its directory."""
    dates = pd.bdate_range(start, periods=n)
    pd.DataFrame({"Date": dates,
                  "irx_rate_annual": [rate] * n}).to_csv(
        os.path.join(str(tmp_path), "IRX.csv"), index=False)
    return str(tmp_path)


def _flat_strategy(index, *, deployed_slice=slice(10, 30), ret=0.002):
    strat = pd.Series(0.0, index=index)
    mask = pd.Series(False, index=index)
    strat.iloc[deployed_slice] = ret
    mask.iloc[deployed_slice] = True
    return strat, mask


# ------------------------------------------------------------ daily_cash_rate

def test_cash_rate_uses_prior_published_rate_not_same_day(tmp_path):
    """Lookahead safety: a rate published ON a session must not be used FOR
    that session."""
    d = str(tmp_path)
    pd.DataFrame({"Date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                  "irx_rate_annual": [0.02, 0.99]}).to_csv(
        os.path.join(d, "IRX.csv"), index=False)
    out = ar.daily_cash_rate(pd.DatetimeIndex(["2020-01-03"]), data_dir=d)
    # 0.99 is published on 01-03; using it would be lookahead. Expect the
    # coupon-equivalent of 0.02, accrued over the first session's one day
    # (calendar days, not 252 trading days, and the coupon equivalent, not
    # the raw quote -- see the calendar-accrual and coupon-equivalent tests below).
    assert out.iloc[0] == pytest.approx(5.583784689262422e-05, rel=1e-12)


def test_cash_rate_is_a_simple_return_per_calendar_day(tmp_path):
    """Calendar-day accrual over 365, not a 252-trading-day divisor."""
    d = _irx(tmp_path, rate=0.0504)
    out = ar.daily_cash_rate(_idx(10, "2020-06-01"), data_dir=d)
    # 2020-06-01 is a Monday and the first session in the index, so it
    # accrues one day, not three -- the first session has no predecessor.
    assert out.iloc[0] == pytest.approx(0.051_759_414_946_417_25 / 365, rel=1e-12)
    assert out.name == "cash"


def test_cash_rate_fails_closed_before_series_start(tmp_path):
    """A 0%-idle assumption can flip a verdict; a session with no prior
    rate must RAISE, never silently return 0."""
    d = _irx(tmp_path, start="2021-01-01", n=100)
    with pytest.raises(ValueError, match="no prior published cash rate"):
        ar.daily_cash_rate(_idx(5, "2020-01-02"), data_dir=d)


def test_cash_rate_rejects_missing_file_and_bad_index(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        ar.daily_cash_rate(_idx(5), data_dir=str(tmp_path))
    d = _irx(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        ar.daily_cash_rate(pd.DatetimeIndex(["2020-06-01", "2020-06-01"]),
                           data_dir=d)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        ar.daily_cash_rate(pd.Index([1, 2, 3]), data_dir=d)


def test_cash_rate_annual_total_is_the_coupon_equivalent_not_the_discount(
        tmp_path):
    """DELIBERATE divergence from causal_backtest.cash_log_returns_by_date,
    which spreads the raw IRX quote over 252 trading days and therefore
    totals the BANK DISCOUNT quote over a year.

    The stored quote is converted to its coupon
    equivalent first, and then accrued over calendar days, so
    a full year of sessions must total the coupon equivalent -- which is
    strictly ABOVE the discount quote, because the discount understates the
    idle alternative and therefore flatters the strategy.
    """
    d = _irx(tmp_path, rate=0.0504, start="2018-01-01", n=1200)
    idx = pd.bdate_range("2020-01-02", "2021-01-01")
    total = float(ar.daily_cash_rate(idx, data_dir=d).sum())
    spanned = (idx[-1] - idx[0]).days + 1          # first session accrues 1
    assert total == pytest.approx(
        0.051_759_414_946_417_25 * spanned / 365, rel=1e-12)
    assert total > 0.0504                          # above the discount quote



# --------------------------------------------- idle cash: sweep vs parked
# The idle-cash rule: money actually held as cash earns the flat brokerage
# sweep rate (constants.IDLE_SWEEP_APY); money parked in bills, a bond fund
# or another risk-free-rate holding earns that yield only net of its
# friction costs (the round-trip spread and the tax on interest).


def test_sweep_cash_credits_the_configured_flat_rate_not_the_bill_rate(tmp_path):
    """The DEFAULT idle credit is the brokerage sweep rate -- a flat APY --
    and it is far below the 13-week bill rate of ``daily_cash_rate``."""
    idx = pd.bdate_range("2020-01-02", "2021-01-01")
    spanned = (idx[-1] - idx[0]).days + 1           # first session accrues 1
    swept = ar.sweep_cash_daily(idx)
    assert float(swept.sum()) == pytest.approx(
        constants.IDLE_SWEEP_APY * spanned / ar.CALENDAR_DAYS_PER_YEAR,
        rel=1e-12)
    d = _irx(tmp_path, rate=0.0504, start="2018-01-01", n=1200)
    assert float(swept.sum()) < float(
        ar.daily_cash_rate(idx, data_dir=d).sum())


def test_sweep_cash_accrues_on_calendar_days_so_a_weekend_pays_three():
    """Same calendar-accrual rule as daily_cash_rate (calendar-day accrual): the
    Monday session carries the whole weekend."""
    idx = pd.DatetimeIndex(["2020-06-18", "2020-06-19", "2020-06-22"])
    swept = ar.sweep_cash_daily(idx)
    one = constants.IDLE_SWEEP_APY / ar.CALENDAR_DAYS_PER_YEAR
    assert float(swept.iloc[0]) == pytest.approx(one, rel=1e-12)
    assert float(swept.iloc[1]) == pytest.approx(one, rel=1e-12)
    assert float(swept.iloc[2]) == pytest.approx(3.0 * one, rel=1e-12)


def test_parked_cash_charges_the_round_trip_spread_once_per_idle_run(tmp_path):
    """Parking idle capital in a bill or bond fund is a real round trip per
    idle stretch: buy on going flat, sell on redeploying. Half the spread
    lands on the FIRST idle session of a run and half on the LAST, so both
    halves fall on sessions the account is actually idle -- a charge on the
    redeploy session would be silently discarded by account_daily_returns,
    which reads the strategy's own return there."""
    d = _irx(tmp_path, rate=0.0, start="2019-01-01", n=500)
    idx = pd.bdate_range("2020-06-01", periods=10)
    mask = pd.Series(
        [False, False, True, True, False, False, False, True, False, False],
        index=idx)
    parked = ar.parked_cash_daily(idx, in_market=mask, data_dir=d,
                                  spread_bps=8.0, interest_tax_rate=0.0)
    half = 8.0 / 1e4 / 2.0
    assert float(parked.sum()) == pytest.approx(-3.0 * 2.0 * half, rel=1e-12)
    charged = {i for i, v in enumerate(parked.to_numpy()) if v != 0.0}
    assert charged == {0, 1, 4, 6, 8, 9}


def test_parked_cash_charges_ordinary_income_tax_on_interest(tmp_path):
    """Bill and money-fund interest is ordinary income, so it is charged at
    the short-term rate, not the long-term one."""
    d = _irx(tmp_path, rate=0.04, start="2019-01-01", n=500)
    idx = pd.bdate_range("2020-06-01", periods=5)
    mask = pd.Series(False, index=idx)
    gross = ar.daily_cash_rate(idx, data_dir=d)
    parked = ar.parked_cash_daily(idx, in_market=mask, data_dir=d,
                                  spread_bps=0.0,
                                  interest_tax_rate=constants.STCG_TAX_RATE)
    assert float(parked.sum()) == pytest.approx(
        float(gross.sum()) * (1.0 - constants.STCG_TAX_RATE), rel=1e-12)


def test_parked_cash_passes_a_negative_yield_through_untaxed(tmp_path):
    """A NEGATIVE bill yield is passed through undiminished. Multiplying it
    by (1 - rate) would shrink a real cost toward zero -- flattering the
    strategy in exactly the direction the idle-cash rule exists to stop."""
    d = _irx(tmp_path, rate=-0.001, start="2019-01-01", n=500)
    idx = pd.bdate_range("2020-06-01", periods=5)
    mask = pd.Series(False, index=idx)
    gross = ar.daily_cash_rate(idx, data_dir=d)
    assert float(gross.sum()) < 0.0
    parked = ar.parked_cash_daily(idx, in_market=mask, data_dir=d,
                                  spread_bps=0.0,
                                  interest_tax_rate=constants.STCG_TAX_RATE)
    assert float(parked.sum()) == pytest.approx(float(gross.sum()), rel=1e-12)


def test_parked_cash_refuses_to_run_without_declared_friction(tmp_path):
    """The un-frictioned bill rate must not be reachable by accident: the
    idle-cash rule admits bill/bond parking ONLY with its costs tracked."""
    d = _irx(tmp_path, rate=0.04, start="2019-01-01", n=500)
    idx = pd.bdate_range("2020-06-01", periods=5)
    mask = pd.Series(False, index=idx)
    with pytest.raises(TypeError):
        ar.parked_cash_daily(idx, in_market=mask, data_dir=d)


# ------------------------------------------------------------- tax drag

def test_tax_factor_matches_the_engines_own_post_tax_multiple():
    """THE identity: applying the factor at the exit bar must reproduce
    backtest_engine.run_backtest's own post-tax per-trade multiple exactly.
    A subtractive drag does not -- it over-charges by the account's growth
    over the trade (1.8x on a +100% winner)."""
    from mining import backtest_engine, benchmark_relative as br
    from mining.backtest_engine import SignalResult
    idx = _idx(12)
    close = np.full(12, 100.0)
    close[10] = 200.0                                   # a +100% winner
    df = pd.DataFrame({"Open": close, "High": close, "Low": close,
                       "Close": close}, index=idx)
    trades, mask = backtest_engine.run_backtest(
        df, lambda frame, i: SignalResult(entry=(i == 0), exit=(i == 10)),
        cost_bps=0.0, tax_haircut_rate=0.12)
    post = float(trades["net_ret"].iloc[0])
    pre = trades.copy()
    pre["net_ret"] = post / (1 - 0.12)                  # recover PRE-tax
    acct = ar.account_daily_returns(
        strategy_daily=br.mark_to_market_daily_returns(df, trades, cost_bps=0.0),
        in_market=mask, cash_daily=pd.Series(0.0, index=idx),
        tax_factor=ar.tax_factor_by_exit_day(idx, pre, tax_haircut_rate=0.12))
    assert ar.terminal_wealth(acct) == pytest.approx(1.0 + post, abs=1e-12)


def test_tax_factor_stays_bounded_for_a_huge_winner():
    """A subtractive drag of tax*net goes below -100% once net > 1/tax - 1
    (net 9 at 12% => -1.08), making the account unscoreable. The factor form
    is bounded in (1 - tax, 1]."""
    idx = _idx(20)
    big = pd.DataFrame({"exit_date": [idx[10]], "net_ret": [9.0]})
    f = ar.tax_factor_by_exit_day(idx, big, tax_haircut_rate=0.12)
    assert f.iloc[10] == pytest.approx(1.0 - 0.12 * 9.0 / 10.0)
    assert 1.0 - 0.12 < f.iloc[10] <= 1.0


def test_tax_factor_charges_at_exit_on_positive_net_only():
    idx = _idx(50)
    trades = pd.DataFrame({"exit_date": [idx[10], idx[20]],
                           "net_ret": [0.05, -0.03]})
    f = ar.tax_factor_by_exit_day(idx, trades, tax_haircut_rate=0.12)
    assert f.iloc[10] == pytest.approx(1.0 - 0.12 * 0.05 / 1.05)
    assert f.iloc[20] == 1.0                 # a loss is not taxed
    assert (f.drop(f.index[[10]]) == 1.0).all()


def test_tax_factor_composes_same_day_exits_multiplicatively():
    idx = _idx(50)
    trades = pd.DataFrame({"exit_date": [idx[10], idx[10]],
                           "net_ret": [0.05, 0.03]})
    f = ar.tax_factor_by_exit_day(idx, trades, tax_haircut_rate=0.10)
    expected = (1 - 0.10 * 0.05 / 1.05) * (1 - 0.10 * 0.03 / 1.03)
    assert f.iloc[10] == pytest.approx(expected)


def test_tax_factor_zero_rate_and_empty_trades_are_identity():
    idx = _idx(20)
    trades = pd.DataFrame({"exit_date": [idx[5]], "net_ret": [1.0]})
    assert (ar.tax_factor_by_exit_day(idx, trades,
                                      tax_haircut_rate=0.0) == 1.0).all()
    empty = pd.DataFrame({"exit_date": [], "net_ret": []})
    assert (ar.tax_factor_by_exit_day(idx, empty,
                                      tax_haircut_rate=0.5) == 1.0).all()


def test_tax_factor_fails_closed_on_bad_rate_and_missing_exit():
    idx = _idx(20)
    trades = pd.DataFrame({"exit_date": [idx[5]], "net_ret": [0.1]})
    with pytest.raises(ValueError, match="tax_haircut_rate"):
        ar.tax_factor_by_exit_day(idx, trades, tax_haircut_rate=1.0)
    with pytest.raises(ValueError, match="tax_haircut_rate"):
        ar.tax_factor_by_exit_day(idx, trades, tax_haircut_rate=-0.1)
    off = pd.DataFrame({"exit_date": [pd.Timestamp("1999-01-04")],
                        "net_ret": [0.1]})
    with pytest.raises(ValueError, match="not in index"):
        ar.tax_factor_by_exit_day(idx, off, tax_haircut_rate=0.12)


# -------------------------------------------------- account_daily_returns

def test_account_earns_strategy_when_deployed_and_cash_when_idle():
    idx = _idx(100)
    strat, mask = _flat_strategy(idx)
    cash = pd.Series(0.0001, index=idx)
    acct = ar.account_daily_returns(strategy_daily=strat, in_market=mask,
                                    cash_daily=cash)
    assert acct.iloc[15] == pytest.approx(0.002)     # deployed
    assert acct.iloc[0] == pytest.approx(0.0001)     # idle
    assert acct.name == "account"


def test_idle_is_the_mask_not_a_zero_return():
    """A DEPLOYED session that happens to close exactly flat must still be
    treated as deployed. Deriving idleness from `strategy_daily == 0` would
    badly understate time_in_market."""
    idx = _idx(30)
    strat = pd.Series(0.0, index=idx)          # every session returns 0.0
    mask = pd.Series(True, index=idx)          # ...but all are deployed
    cash = pd.Series(0.01, index=idx)
    acct = ar.account_daily_returns(strategy_daily=strat, in_market=mask,
                                    cash_daily=cash)
    assert (acct == 0.0).all(), "a flat deployed session earned cash by mistake"


def test_account_applies_tax_factor_multiplicatively():
    idx = _idx(30)
    strat, mask = _flat_strategy(idx, deployed_slice=slice(0, 30))
    cash = pd.Series(0.0, index=idx)
    factor = pd.Series(1.0, index=idx)
    factor.iloc[5] = 0.999
    acct = ar.account_daily_returns(strategy_daily=strat, in_market=mask,
                                    cash_daily=cash, tax_factor=factor)
    assert acct.iloc[5] == pytest.approx(1.002 * 0.999 - 1.0)
    assert acct.iloc[6] == pytest.approx(0.002)


def test_account_fails_closed_on_incomplete_cash_and_bad_input():
    idx = _idx(30)
    strat, mask = _flat_strategy(idx)
    short_cash = pd.Series(0.0001, index=idx[:10])
    with pytest.raises(ValueError, match="missing"):
        ar.account_daily_returns(strategy_daily=strat, in_market=mask,
                                 cash_daily=short_cash)
    bad = strat.copy()
    bad.iloc[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ar.account_daily_returns(strategy_daily=bad, in_market=mask,
                                 cash_daily=pd.Series(0.0, index=idx))


def test_missing_in_market_entry_credits_cash_not_exposure():
    """The conservative direction: an absent mask entry means NOT deployed."""
    idx = _idx(20)
    strat = pd.Series(0.5, index=idx)
    mask = pd.Series(True, index=idx[:5])        # only first 5 known
    cash = pd.Series(0.001, index=idx)
    acct = ar.account_daily_returns(strategy_daily=strat, in_market=mask,
                                    cash_daily=cash)
    assert acct.iloc[0] == pytest.approx(0.5)
    assert acct.iloc[10] == pytest.approx(0.001)


# ------------------------------------------------------------- log_excess

def test_log_excess_sums_to_log_of_terminal_wealth_ratio():
    """THE identity that justifies log over arithmetic: the excess terms sum
    to log(terminal_account / terminal_benchmark), so 'positive on average'
    and 'finished ahead' are the same statement."""
    rng = np.random.default_rng(11)
    idx = _idx(400)
    a = pd.Series(0.0003 + 0.011 * rng.standard_normal(400), index=idx)
    b = pd.Series(0.0004 + 0.009 * rng.standard_normal(400), index=idx)
    total = ar.log_excess(a, b).sum()
    ratio = math.log(ar.terminal_wealth(a) / ar.terminal_wealth(b))
    assert total == pytest.approx(ratio, rel=1e-12)


def test_log_excess_inner_joins_and_drops_nan():
    idx = _idx(50)
    a = pd.Series(0.001, index=idx)
    b = pd.Series(0.001, index=idx[:30])
    assert len(ar.log_excess(a, b)) == 30
    b2 = pd.Series(0.001, index=idx).copy()
    b2.iloc[4] = np.nan
    assert len(ar.log_excess(a, b2)) == 49


def test_log_excess_refuses_a_wiped_out_account():
    idx = _idx(10)
    a = pd.Series(0.0, index=idx)
    a.iloc[3] = -1.0
    with pytest.raises(ValueError, match="log1p undefined"):
        ar.log_excess(a, pd.Series(0.0, index=idx))
    # and it must check the BENCHMARK too, not only the account
    b = pd.Series(0.0, index=idx)
    b.iloc[3] = -1.5
    with pytest.raises(ValueError, match="log1p undefined"):
        ar.log_excess(pd.Series(0.0, index=idx), b)


def test_log_excess_empty_join_is_empty_not_an_error():
    a = pd.Series(0.001, index=_idx(10, "2020-01-02"))
    b = pd.Series(0.001, index=_idx(10, "2023-01-02"))
    assert ar.log_excess(a, b).empty


def test_crypto_saturday_return_survives_spy_calendar_alignment():
    idx = pd.date_range("2020-01-03", periods=4)
    account = pd.Series([0.0, 0.10, 0.0, 0.0], index=idx)
    benchmark = pd.Series([0.0, 0.0], index=idx[[0, 3]])
    excess = ar.log_excess(account, benchmark)
    assert excess.sum() == pytest.approx(math.log(1.10))
    pd.testing.assert_index_equal(excess.index, idx[[0, 3]])
    assert excess.iloc[-1] == pytest.approx(math.log(1.10))


def test_crypto_weekend_drawdown_is_retained_by_risk_layer():
    idx = pd.date_range("2020-01-03", periods=4)
    account = pd.Series([0.0, -0.50, 1.0, 0.0], index=idx)
    benchmark = pd.Series([0.0, 0.0], index=idx[[0, 3]])
    result = ar.risk_refusals(account, benchmark)
    assert result["drawdown"]["account"] == pytest.approx(-0.50)
    assert result["n_sessions"] == 4


def test_crypto_evaluate_keeps_same_calendar_through_holdout():
    idx = pd.date_range("2020-01-02", periods=6)
    account = pd.Series([0.0, 0.01, 0.10, -0.03, 0.02, -0.01], index=idx)
    benchmark = pd.Series(0.0, index=idx[[0, 1, 4, 5]])
    result = ar.evaluate(strategy_daily=account, in_market=pd.Series(True, index=idx),
                         cash_daily=pd.Series(0.0, index=idx), benchmark_daily=benchmark,
                         n_trials=1, sr_variance=0.01, tax_policy="none",
                         min_sessions=2, holdout_sessions=2)
    assert result["window"]["scored_sessions"] == 4
    assert result["window"]["holdout_split_applied"] is True
    assert result["window"]["sessions_compounded_to_benchmark"] == 2
    assert result["window"]["sessions_dropped_to_benchmark"] == 0
    assert result["holdout"]["n_holdout"] == 2
    assert result["holdout"]["log_excess_sum"] == pytest.approx(math.log(1.10 * 0.97 * 1.02 * 0.99))


def test_crypto_calendar_preserves_three_year_holdout_boundary():
    idx = pd.date_range("2020-01-01", "2024-12-31")
    benchmark_idx = pd.bdate_range("2020-01-01", "2024-12-31")
    excess = ar.log_excess(pd.Series(0.001, index=idx), pd.Series(0.0, index=benchmark_idx))
    split = ar.split_holdout(excess.index)
    assert len(excess) == 1305
    assert split["holdout"][0] == pd.Timestamp("2022-02-08")
    assert len(split["in_sample"]) == 549


def test_crypto_compounds_a_weekday_holiday_and_benchmark_return():
    idx = pd.date_range("2020-01-03", periods=5)
    a = pd.Series([0, 0.10, 0, 0.20, 0], index=idx)
    b = pd.Series([0, 0.05], index=idx[[0, 4]])
    excess = ar.log_excess(a, b)
    assert excess.sum() == pytest.approx(math.log(1.10 * 1.20 / 1.05))
    assert len(excess) == 2


# ----------------------------------------------------------- score_account

def test_score_refuses_below_the_session_floor():
    x = pd.Series(np.full(251, 0.0001))
    out = ar.score_account(x, n_trials=100.0, sr_variance=0.0002)
    assert out["scored"] is False and out["dsr_value"] is None
    assert "MIN_ACCOUNT_SESSIONS" in out["reason"]
    assert ar.MIN_ACCOUNT_SESSIONS == 252


def test_score_accepts_exactly_the_session_floor():
    rng = np.random.default_rng(3)
    x = pd.Series(0.0002 + 0.004 * rng.standard_normal(252))
    out = ar.score_account(x, n_trials=100.0, sr_variance=0.0002)
    assert out["scored"] is True and out["n_sessions"] == 252
    assert 0.0 <= out["dsr_value"] <= 1.0


def test_score_refuses_when_daily_sr_variance_is_absent():
    """The per-trade vs daily unit-mismatch guard: it must NOT fall back to
    the per-trade DEFAULT_SR_VARIANCE_FLOOR."""
    x = pd.Series(np.full(300, 0.0001))
    out = ar.score_account(x, n_trials=100.0, sr_variance=None)
    assert out["scored"] is False
    assert out["dsr_value"] is None
    assert "per-trade floor" in out["reason"]
    assert out["sr_variance"] is None
    assert out["sr_variance"] != constants.DEFAULT_SR_VARIANCE_FLOOR


def test_score_uses_days_not_trades():
    """n must be the session count. Same series, no trade information
    anywhere in the call."""
    rng = np.random.default_rng(5)
    x = pd.Series(0.0002 + 0.004 * rng.standard_normal(600))
    out = ar.score_account(x, n_trials=100.0, sr_variance=0.0002)
    assert out["n_sessions"] == 600


def test_score_matches_dsr_called_directly():
    """The formula and bar are the EXISTING machinery, unchanged."""
    from mining import dsr
    from mining.benchmark_relative import daily_sharpe_moments
    rng = np.random.default_rng(9)
    x = pd.Series(0.0002 + 0.004 * rng.standard_normal(400))
    m = daily_sharpe_moments(np.asarray(x, dtype=float))
    expected = dsr.deflated_sharpe_ratio(
        sr=m["sr"], n=m["n"], skew=m["skew"], kurtosis=m["kurtosis"],
        n_trials=100.0, sr_variance=0.0002)
    out = ar.score_account(x, n_trials=100.0, sr_variance=0.0002)
    assert out["dsr_value"] == pytest.approx(expected, rel=0, abs=0)


# -------------------------------------------------------- weekly helpers

def test_weekly_returns_compound_within_a_week():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-01-08",
                            "2020-01-09", "2020-01-10"])   # one Mon-Fri
    wk = ar.weekly_returns(pd.Series([0.01] * 5, index=idx))
    assert len(wk) == 1
    assert wk.iloc[0] == pytest.approx(1.01 ** 5 - 1.0)


def test_losing_week_streak_counts_strictly_negative_runs():
    assert ar.longest_losing_week_streak(
        pd.Series([-0.1, -0.1, 0.2, -0.1, -0.1, -0.1, 0.3])) == 3
    assert ar.longest_losing_week_streak(pd.Series([0.1, 0.2])) == 0
    assert ar.longest_losing_week_streak(pd.Series([], dtype=float)) == 0


def test_zero_week_breaks_the_streak():
    """A flat week is not a losing week."""
    assert ar.longest_losing_week_streak(
        pd.Series([-0.1, -0.1, 0.0, -0.1])) == 2


def test_worst_rolling_window_finds_the_deepest_stretch():
    weeks = pd.Series([0.01] * 10 + [-0.02] * 8 + [0.01] * 10)
    got = ar.worst_rolling_window(weeks, window=8)
    assert got == pytest.approx(0.98 ** 8 - 1.0)


def test_worst_rolling_window_is_zero_when_unmeasurable():
    """Fewer weeks than the window: 0.0, so an unmeasurable stretch does not
    read as a catastrophic one."""
    assert ar.worst_rolling_window(pd.Series([-0.5] * 7), window=8) == 0.0
    assert ar.worst_rolling_window(pd.Series([], dtype=float)) == 0.0


def test_terminal_wealth_compounds():
    assert ar.terminal_wealth(pd.Series([0.1, 0.1])) == pytest.approx(1.21)
    assert ar.terminal_wealth(pd.Series([], dtype=float)) == 1.0


# ---------------------------------------------------------- risk_refusals

def _curve(index, per_day):
    return pd.Series(per_day, index=index)


def test_refusals_pass_when_account_dominates_benchmark():
    idx = _idx(400)
    acct = _curve(idx, 0.0006)
    bench = _curve(idx, 0.0002)
    out = ar.risk_refusals(acct, bench)
    assert out["all_refusals_passed"] is True
    assert out["terminal_wealth"]["passed"] is True


def test_absolute_drawdown_cap_rejects_a_deep_account(monkeypatch):
    """A 40% drawdown must fail even when the benchmark drew down MORE --
    that is the whole point of the absolute cap."""
    idx = _idx(300)
    acct = pd.Series(0.0, index=idx)
    acct.iloc[5] = -0.40
    bench = pd.Series(0.0, index=idx)
    bench.iloc[5] = -0.60          # benchmark is worse, so relative passes
    out = ar.risk_refusals(acct, bench)
    assert out["drawdown"]["relative_passed"] is True
    assert out["drawdown"]["absolute_passed"] is False
    assert out["drawdown"]["passed"] is False
    assert out["all_refusals_passed"] is False
    assert ar.ABSOLUTE_MAX_DRAWDOWN_CAP == 0.35


def test_relative_drawdown_rejects_even_inside_the_absolute_cap():
    """Both halves of the drawdown gate must bite independently."""
    idx = _idx(300)
    acct = pd.Series(0.0, index=idx)
    acct.iloc[5] = -0.20           # inside the 35% cap
    bench = pd.Series(0.0, index=idx)
    bench.iloc[5] = -0.05          # but worse than the benchmark
    out = ar.risk_refusals(acct, bench)
    assert out["drawdown"]["absolute_passed"] is True
    assert out["drawdown"]["relative_passed"] is False
    assert out["drawdown"]["passed"] is False


def test_red_week_streak_refusal_bites():
    idx = _idx(200)
    acct = pd.Series(0.0005, index=idx)
    acct.iloc[0:60] = -0.001       # ~12 consecutive losing weeks
    bench = pd.Series(0.0005, index=idx)
    bench.iloc[0:10] = -0.001      # ~2 losing weeks
    out = ar.risk_refusals(acct, bench)
    assert out["red_week_streak"]["account"] > out["red_week_streak"]["benchmark"]
    assert out["red_week_streak"]["passed"] is False


def test_terminal_wealth_is_excluded_from_the_verdict():
    """By design, terminal wealth is reported, never the sole grounds. An account
    that loses on final dollars but passes all three risk gates must still
    show all_refusals_passed True."""
    idx = _idx(400)
    acct = _curve(idx, 0.0001)
    bench = _curve(idx, 0.0009)    # benchmark finishes far ahead
    out = ar.risk_refusals(acct, bench)
    assert out["terminal_wealth"]["passed"] is False
    assert out["drawdown"]["passed"] is True
    assert out["red_week_streak"]["passed"] is True
    assert out["all_refusals_passed"] is True


def test_worst_stretch_is_downside_only_not_a_return_comparison():
    """Guards the downside-only comparison. When neither series ever
    loses money over any 8-week stretch, both "worst" values are POSITIVE and
    a raw >= comparison would refuse the account merely for earning less --
    a return comparison wearing a risk gate's clothes, which would defeat the
    exclusion of terminal wealth from the verdict."""
    idx = _idx(400)
    acct = _curve(idx, 0.0001)     # never loses, earns little
    bench = _curve(idx, 0.0009)    # never loses, earns more
    out = ar.risk_refusals(acct, bench)
    assert out["worst_stretch"]["account"] > 0.0
    assert out["worst_stretch"]["benchmark"] > 0.0
    assert out["worst_stretch"]["account"] < out["worst_stretch"]["benchmark"]
    assert out["worst_stretch"]["passed"] is True, (
        "a never-losing account was refused on a downside gate")
    # ...but a genuine 8-week loss deeper than the benchmark's still refuses
    losing = pd.Series(0.0005, index=idx)
    losing.iloc[0:60] = -0.004
    assert ar.risk_refusals(losing, bench)["worst_stretch"]["passed"] is False


def test_refusal_thresholds_are_the_benchmarks_own_measured_values():
    """No invented constant in three of the four gates: change the benchmark
    and the threshold moves with it."""
    idx = _idx(300)
    acct = pd.Series(0.0, index=idx)
    acct.iloc[5] = -0.10
    mild = pd.Series(0.0, index=idx)
    mild.iloc[5] = -0.05
    harsh = pd.Series(0.0, index=idx)
    harsh.iloc[5] = -0.20
    assert ar.risk_refusals(acct, mild)["drawdown"]["relative_passed"] is False
    assert ar.risk_refusals(acct, harsh)["drawdown"]["relative_passed"] is True


# ---------------------------------- overlay-only excess basis
#
# The risk-basis rule: for a benchmark-plus-overlay family the 0.35 absolute
# drawdown cap and the three SPY-relative refusals apply to the OVERLAY-ONLY
# EXCESS account, not to the whole SPY-core account (which holds SPY through
# 2008 and so can never clear the cap whatever the overlay earns). The excess
# account earns the daily ARITHMETIC excess a - b: for SPY-core-plus-overlay
# that is exactly the overlay's dollars over the prior close's account equity.

def _crash_benchmark(n=400):
    """A benchmark with a deep crash (about -60%): the whole-account curve of an
    SPY-core account cannot clear the 0.35 cap, whatever an overlay earns."""
    idx = _idx(n)
    bench = pd.Series(0.0004, index=idx)
    bench.iloc[100:160] = -0.015
    return idx, bench


def test_overlay_only_basis_measures_the_excess_account_not_the_whole_account():
    idx, bench = _crash_benchmark()
    excess = pd.Series(0.0, index=idx)
    excess.iloc[::20] = 0.001                     # a steady small overlay gain
    acct = bench + excess
    whole = ar.risk_refusals(acct, bench)
    over = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert whole["account_basis"] == "whole_account"
    assert whole["drawdown"]["absolute_passed"] is False      # the SPY-core ceiling
    assert whole["all_refusals_passed"] is False
    assert over["account_basis"] == "overlay_only_excess"
    assert over["drawdown"]["absolute_passed"] is True
    assert over["all_refusals_passed"] is True
    # the measured account IS the excess path, the benchmark is still SPY itself
    assert over["drawdown"]["account"] == pytest.approx(
        ar.max_drawdown(excess), abs=1e-12)
    assert over["drawdown"]["benchmark"] == whole["drawdown"]["benchmark"]
    assert over["n_sessions"] == whole["n_sessions"]


def test_overlay_only_basis_applies_the_absolute_cap_to_the_excess_account():
    """The 0.35 cap follows the risk-basis rule onto
    the excess account: an overlay that itself loses more than 35 percent
    breaches it even when the benchmark drew down more."""
    idx, bench = _crash_benchmark()
    excess = pd.Series(0.0, index=idx)
    excess.iloc[100:150] = -0.010                # 0.99 ** 50 = -39.5 percent
    acct = bench + excess
    out = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert out["drawdown"]["account"] < -ar.ABSOLUTE_MAX_DRAWDOWN_CAP
    assert out["drawdown"]["relative_passed"] is True         # SPY drew down more
    assert out["drawdown"]["absolute_passed"] is False
    assert out["drawdown"]["passed"] is False
    assert out["all_refusals_passed"] is False


def test_overlay_only_basis_applies_the_relative_drawdown_refusal_to_the_excess_account():
    """Inside the cap but worse than SPY's own drawdown: refused. SPY's own
    drawdown is a single -3 percent day; the overlay loses about -18 percent
    while SPY is UP, so the whole account (up on those days) never draws down
    below SPY's -3 percent and passes the same comparison."""
    idx = _idx(300)
    bench = pd.Series(0.03, index=idx)
    bench.iloc[10] = -0.03                        # SPY's own drawdown: -3 percent
    excess = pd.Series(0.0, index=idx)
    excess.iloc[50:60] = -0.02                    # about -18 percent, inside the cap
    acct = bench + excess
    whole = ar.risk_refusals(acct, bench)
    over = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert whole["drawdown"]["relative_passed"] is True
    assert over["drawdown"]["account"] < over["drawdown"]["benchmark"]
    assert over["drawdown"]["absolute_passed"] is True
    assert over["drawdown"]["relative_passed"] is False
    assert over["drawdown"]["passed"] is False


def test_overlay_only_basis_applies_the_red_week_streak_refusal_to_the_excess_account():
    idx = _idx(400)
    bench = pd.Series(0.003, index=idx)           # no losing week at all
    excess = pd.Series(0.0, index=idx)
    excess.iloc[40:100] = -0.001                  # about 12 losing weeks of overlay
    acct = bench + excess
    whole = ar.risk_refusals(acct, bench)
    over = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert whole["red_week_streak"]["account"] == 0
    assert whole["red_week_streak"]["passed"] is True
    assert over["red_week_streak"]["account"] >= 10
    assert over["red_week_streak"]["benchmark"] == 0
    assert over["red_week_streak"]["passed"] is False


def test_overlay_only_basis_applies_the_worst_stretch_refusal_to_the_excess_account():
    idx = _idx(400)
    bench = pd.Series(0.003, index=idx)           # never loses over any stretch
    excess = pd.Series(0.0, index=idx)
    excess.iloc[40:100] = -0.001
    acct = bench + excess
    whole = ar.risk_refusals(acct, bench)
    over = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert whole["worst_stretch"]["passed"] is True
    assert over["worst_stretch"]["account_loss"] < 0.0
    assert over["worst_stretch"]["benchmark_loss"] == 0.0
    assert over["worst_stretch"]["passed"] is False


def test_overlay_only_basis_can_pass_all_three_relative_refusals_and_the_cap_on_a_small_overlay():
    """The other side of the overlay-only basis, checked rather than assumed:
    an overlay that is on a few sessions a month and small never comes near
    SPY's own drawdown, streak or worst stretch, so under this basis the three
    SPY-relative refusals do not bite in practice."""
    idx, bench = _crash_benchmark(600)
    excess = pd.Series(0.0, index=idx)
    excess.iloc[::21] = -0.002                    # a small losing overlay window each month
    acct = bench + excess
    out = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert out["all_refusals_passed"] is True


def test_overlay_only_basis_still_refuses_an_interior_hole():
    """The interior-gap guard runs on the account and benchmark BEFORE the
    excess is formed: the basis must not open a way round it."""
    idx = _idx(30)
    acct = pd.Series(0.0004, index=idx)
    acct.iloc[5] = -0.46
    bench = pd.Series(0.0004, index=idx)
    bench.iloc[5] = np.nan
    with pytest.raises(ValueError, match="interior gap"):
        ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")


def test_overlay_only_basis_is_the_arithmetic_excess_not_the_geometric_one():
    """Threshold tests alone cannot tell the arithmetic excess from the ratio
    series (1+a)/(1+b)-1, because they only compare against a limit, so pin
    the NUMBER. SPY rallies hard on the days the overlay loses, where the two
    definitions differ (0.99**20 - 1 = -0.1821 against about -0.1772)."""
    idx = _idx(200)
    bench = pd.Series(0.0, index=idx)
    bench.iloc[50:70] = 0.03
    excess = pd.Series(0.0, index=idx)
    excess.iloc[50:70] = -0.01
    acct = bench + excess
    arithmetic = float(np.prod(1.0 + excess.to_numpy()) - 1.0)
    geometric = float(np.prod((1.0 + acct.to_numpy()) / (1.0 + bench.to_numpy())) - 1.0)
    assert abs(arithmetic - geometric) > 0.004                 # the fixture really tells them apart
    out = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert out["drawdown"]["account"] == pytest.approx(arithmetic, abs=1e-12)
    assert out["drawdown"]["account"] != pytest.approx(geometric, abs=1e-3)


def test_overlay_only_basis_scores_only_the_sessions_the_account_and_benchmark_share():
    """Pins the alignment of the excess series: sessions the benchmark does
    not cover (a leading gap, i.e. the expected truncation when the
    benchmark's history starts later than the account's) must not enter the
    excess account, whatever the account did on them."""
    idx = _idx(300)
    bench = pd.Series(0.001, index=idx[60:])                  # the benchmark starts 60 sessions late
    excess = pd.Series(0.0, index=idx)
    excess.iloc[100:110] = -0.004
    acct = pd.Series(0.001, index=idx) + excess
    acct.iloc[10] = -0.5                                      # a crash BEFORE the benchmark exists
    trimmed = ar.risk_refusals(acct.iloc[60:], bench, account_basis="overlay_only_excess")
    out = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert out["n_sessions"] == 240
    assert out == trimmed
    assert out["drawdown"]["account"] > -0.05                 # the -50 percent day is not in the excess account


def test_terminal_wealth_is_the_whole_account_on_either_basis_and_never_gates():
    idx, bench = _crash_benchmark()
    excess = pd.Series(0.0, index=idx)
    excess.iloc[::20] = 0.001
    acct = bench + excess
    whole = ar.risk_refusals(acct, bench)
    over = ar.risk_refusals(acct, bench, account_basis="overlay_only_excess")
    assert over["terminal_wealth"] == whole["terminal_wealth"]
    assert over["terminal_wealth"]["account"] == pytest.approx(ar.terminal_wealth(acct))


def test_default_basis_is_the_whole_account_and_equals_the_explicit_setting():
    idx, bench = _crash_benchmark()
    acct = bench + pd.Series(0.0005, index=idx)
    default = ar.risk_refusals(acct, bench)
    explicit = ar.risk_refusals(acct, bench, account_basis="whole_account")
    assert default == explicit
    assert default["account_basis"] == "whole_account"
    assert default["drawdown"]["account"] == pytest.approx(ar.max_drawdown(acct))


def test_an_unknown_account_basis_is_refused_not_defaulted():
    idx, bench = _crash_benchmark(60)
    with pytest.raises(ValueError, match="account_basis"):
        ar.risk_refusals(bench, bench, account_basis="overlay")
    assert ar.RISK_ACCOUNT_BASES == ("whole_account", "overlay_only_excess")


# ---------------------------------------------------------------- evaluate

def _eval_kwargs(idx, strat, mask, bench, cash, **over):
    base = dict(strategy_daily=strat, in_market=mask, benchmark_daily=bench,
                cash_daily=cash, n_trials=100.0, sr_variance=0.0002,
                tax_policy="none")
    base.update(over)
    return base


def test_evaluate_returns_not_scoreable_below_the_session_floor():
    idx = _idx(100)
    strat, mask = _flat_strategy(idx)
    out = ar.evaluate(**_eval_kwargs(
        idx, strat, mask, pd.Series(0.0003, index=idx),
        pd.Series(0.0001, index=idx)))
    assert out["verdict"] == "not_scoreable"
    # layer 2 is still reported so a reader can see why
    assert "drawdown" in out["layer2"]


def test_evaluate_cuts_on_dsr_before_risk():
    idx = _idx(400)
    strat, mask = _flat_strategy(idx, deployed_slice=slice(0, 400),
                                 ret=0.00005)
    out = ar.evaluate(**_eval_kwargs(
        idx, strat, mask, pd.Series(0.0004, index=idx),
        pd.Series(0.0001, index=idx)))
    assert out["verdict"] == "cut_dsr"


def _spy_like(n=1500, seed=99, start="2015-01-02"):
    """A benchmark with REAL volatility. A constant-return benchmark has zero
    drawdown and zero losing weeks, which makes the relative refusals
    structurally unpassable for any realistic account -- a fixture artefact, not a
    property of the gate. This one measures dd -20.4%, longest red-week run 8."""
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(0.00035 + 0.011 * rng.standard_normal(n), index=idx)


def test_evaluate_cuts_on_risk_refusal_when_dsr_clears():
    """A candidate that clears the statistical bar but breaches the absolute
    drawdown cap is cut on risk, not on statistics.

    The fixture is 2400 sessions long because the out-of-sample holdout gate
    carves the last 756 sessions out of the scored window; a much shorter
    fixture would leave too little in-sample history for the DSR to clear,
    and the test would then measure "cut on statistics", the opposite of
    what its name claims. The account's drawdown is -49.1%, measured over
    the full window by the refusals."""
    bench = _spy_like(2400)
    idx = bench.index
    rng = np.random.default_rng(4)
    strat = pd.Series(0.0022 + 0.012 * rng.standard_normal(len(idx)), index=idx)
    strat.iloc[300:360] -= 0.010          # a sustained ~60-session decline
    mask = pd.Series(True, index=idx)
    out = ar.evaluate(**_eval_kwargs(
        idx, strat, mask, bench, pd.Series(0.0, index=idx)))
    assert out["layer1"]["scored"] is True
    assert out["layer1"]["dsr_value"] >= out["dsr_pass_threshold"]
    assert out["layer2"]["drawdown"]["account"] < -ar.ABSOLUTE_MAX_DRAWDOWN_CAP
    assert out["layer2"]["drawdown"]["absolute_passed"] is False
    assert out["verdict"] == "cut_risk_refusal"


def _overlay_account_on_a_crashing_benchmark(n=2400, seed=5):
    """SPY (real volatility, plus a 40-session crash of about -55 percent) with
    a small overlay held on 3 sessions of every 21: an SPY-core-plus-overlay
    account whose whole curve cannot clear the 0.35 cap but whose overlay edge
    is steady enough to clear layer 1 and the holdout."""
    bench = _spy_like(n)
    bench.iloc[600:640] = -0.02
    idx = bench.index
    rng = np.random.default_rng(seed)
    excess = pd.Series(0.0, index=idx)
    for start in range(0, n, 21):
        excess.iloc[start:start + 3] = 0.004 + 0.0005 * rng.standard_normal(
            len(excess.iloc[start:start + 3]))
    return idx, bench, bench + excess


def test_evaluate_defaults_to_the_whole_account_risk_basis():
    idx, bench, acct = _overlay_account_on_a_crashing_benchmark()
    out = ar.evaluate(**_eval_kwargs(
        idx, acct, pd.Series(True, index=idx), bench, pd.Series(0.0, index=idx)))
    assert out["layer2"]["account_basis"] == "whole_account"
    assert out["layer2"]["drawdown"]["absolute_passed"] is False
    assert out["layer1"]["dsr_value"] >= out["dsr_pass_threshold"]
    assert out["verdict"] == "cut_risk_refusal"


def test_evaluate_scores_the_risk_layer_on_the_overlay_only_excess_when_asked():
    """The same account, the same layer 1, the same holdout: only the account
    the four refusals measure changes, and the verdict moves past the risk
    layer -- under this basis an SPY-core overlay is judged on its own edge."""
    idx, bench, acct = _overlay_account_on_a_crashing_benchmark()
    kwargs = _eval_kwargs(idx, acct, pd.Series(True, index=idx), bench,
                          pd.Series(0.0, index=idx))
    whole = ar.evaluate(**kwargs)
    over = ar.evaluate(**kwargs, risk_account_basis="overlay_only_excess")
    assert over["layer2"]["account_basis"] == "overlay_only_excess"
    assert over["layer2"]["all_refusals_passed"] is True
    assert over["layer1"] == whole["layer1"]                  # nothing upstream moves
    assert over["holdout"] == whole["holdout"]
    assert over["verdict"] == "survived"
    assert whole["verdict"] == "cut_risk_refusal"


def test_evaluate_refuses_an_unknown_risk_account_basis():
    idx, bench, acct = _overlay_account_on_a_crashing_benchmark(400)
    with pytest.raises(ValueError, match="account_basis"):
        ar.evaluate(**_eval_kwargs(
            idx, acct, pd.Series(True, index=idx), bench,
            pd.Series(0.0, index=idx)), risk_account_basis="overlay")


def test_evaluate_survived_requires_both_layers():
    bench = _spy_like()
    idx = bench.index
    rng = np.random.default_rng(21)
    strat = pd.Series(0.0016 + 0.009 * rng.standard_normal(len(idx)), index=idx)
    mask = pd.Series(True, index=idx)
    out = ar.evaluate(**_eval_kwargs(
        idx, strat, mask, bench, pd.Series(0.0, index=idx)))
    assert out["verdict"] == "survived"
    assert out["layer1"]["scored"] and out["layer2"]["all_refusals_passed"]
    # a genuine survivor drew down LESS than the benchmark, not merely little
    assert (out["layer2"]["drawdown"]["account"]
            > out["layer2"]["drawdown"]["benchmark"])


def test_evaluate_realization_tax_needs_trades_and_reduces_return():
    idx = _idx(300)
    strat, mask = _flat_strategy(idx, deployed_slice=slice(0, 300))
    cash = pd.Series(0.0, index=idx)
    bench = pd.Series(0.0001, index=idx)
    with pytest.raises(ValueError, match="needs trades_df"):
        ar.evaluate(**_eval_kwargs(idx, strat, mask, bench, cash,
                                   tax_policy="realization"))
    # entry_date is REQUIRED because of the long/short holding-period split --
    # a trade record without one has an unknowable holding period, and
    # defaulting it to short-term would silently mis-rate long-term gains.
    # backtest_engine.run_backtest has always emitted this column.
    trades = pd.DataFrame({"entry_date": [idx[40]], "exit_date": [idx[50]],
                           "net_ret": [0.20]})
    taxed = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench, cash,
                                       tax_policy="realization",
                                       trades_df=trades))
    untaxed = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench, cash))
    assert (ar.terminal_wealth(taxed["account_daily"])
            < ar.terminal_wealth(untaxed["account_daily"]))
    assert taxed["tax_haircut_rate"] == constants.STCG_TAX_RATE
    assert untaxed["tax_haircut_rate"] == 0.0


def test_evaluate_rejects_unknown_tax_policy():
    idx = _idx(50)
    strat, mask = _flat_strategy(idx)
    with pytest.raises(ValueError, match="unknown tax_policy"):
        ar.evaluate(**_eval_kwargs(idx, strat, mask,
                                   pd.Series(0.0, index=idx),
                                   pd.Series(0.0, index=idx),
                                   tax_policy="mark_to_market"))


def test_evaluate_reports_time_in_market_from_the_mask():
    idx = _idx(300)
    strat, mask = _flat_strategy(idx, deployed_slice=slice(0, 75))
    out = ar.evaluate(**_eval_kwargs(
        idx, strat, mask, pd.Series(0.0002, index=idx),
        pd.Series(0.0001, index=idx)))
    assert out["time_in_market"] == pytest.approx(0.25)


def test_evaluate_end_to_end_with_a_real_cash_file(tmp_path):
    """The one test that exercises the file-reading path."""
    d = _irx(tmp_path, rate=0.05, start="2019-01-01", n=1500)
    idx = _idx(400, "2020-01-02")
    strat, mask = _flat_strategy(idx, deployed_slice=slice(0, 200))
    cash = ar.daily_cash_rate(idx, data_dir=d)
    out = ar.evaluate(**_eval_kwargs(
        idx, strat, mask, pd.Series(0.0003, index=idx), cash))
    assert out["construction"] == "ABI-1"
    # Any verdict in the vocabulary is accepted. 400 sessions cannot carry
    # a 756-session holdout, so this fixture lands on
    # "holdout_not_evaluable" -- which is the correct verdict for it, and
    # not what this test is about: its subject is the file-reading path and
    # the cash-accrual arithmetic asserted below.
    assert out["verdict"] in ar.VERDICTS
    # idle sessions really did earn the T-bill rate, not zero -- the
    # COUPON EQUIVALENT of the stored discount quote, accrued over
    # the calendar days since the previous session.
    elapsed = (idx[250] - idx[249]).days
    assert out["account_daily"].iloc[250] == pytest.approx(
        0.051_343_367_562_244_946 * elapsed / 365, rel=1e-12)
    assert out["account_daily"].iloc[250] != pytest.approx(0.05 / 252)


# ------------------------------- input guards

def test_cash_rate_refuses_a_timezone_aware_index(tmp_path):
    """Without this guard a tz-aware index would degrade
    `rate.index.union(index)` to object dtype; ffill/shift would stop
    aligning on time and EVERY session would silently receive IRX.csv's LAST
    row -- a lookahead leak with no error raised. Reachable: intraday_data
    builds tz-aware ET frames."""
    d = _irx(tmp_path)
    aware = _idx(5, "2020-06-01").tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-NAIVE"):
        ar.daily_cash_rate(aware, data_dir=d)


def test_mask_with_nan_is_refused_not_read_as_deployed():
    """`pd.Series([True, nan]).astype(bool)` maps
    NaN to True -- DEPLOYED -- the opposite of the documented conservative
    direction. A 95%-drawdown account would be scored as passing through it."""
    idx = _idx(20)
    bad = pd.Series([True, np.nan] + [False] * 18, index=idx)
    with pytest.raises(ValueError, match="NaN"):
        ar.account_daily_returns(strategy_daily=pd.Series(0.0, index=idx),
                                 in_market=bad,
                                 cash_daily=pd.Series(0.0, index=idx))


def test_wholly_misaligned_mask_is_refused_not_read_as_all_idle():
    """A mask sharing no session with the strategy
    reindexes to all-False and reads as "idle for the account's entire life".
    Its sibling cash_daily already refuses the same mistake."""
    idx = _idx(20)
    elsewhere = pd.Series(True, index=pd.bdate_range("1999-01-04", periods=5))
    with pytest.raises(ValueError, match="shares no session"):
        ar.account_daily_returns(strategy_daily=pd.Series(0.0, index=idx),
                                 in_market=elsewhere,
                                 cash_daily=pd.Series(0.0, index=idx))


def test_a_shorter_mask_is_still_allowed():
    """The conservative default coexists with the misaligned-mask guard: a
    mask covering only part of the index still means NOT deployed on the
    rest."""
    idx = _idx(20)
    partial = pd.Series(True, index=idx[:5])
    acct = ar.account_daily_returns(strategy_daily=pd.Series(0.5, index=idx),
                                    in_market=partial,
                                    cash_daily=pd.Series(0.001, index=idx))
    assert acct.iloc[0] == pytest.approx(0.5)
    assert acct.iloc[10] == pytest.approx(0.001)


def test_interior_benchmark_hole_is_refused_not_spliced():
    """Dropping interior NaN would SPLICE the equity curve, and drawdown is
    a path statistic: a true -46% drawdown would read as 0.0 and pass the
    absolute cap whenever the crash sessions are exactly the ones the
    benchmark is missing."""
    idx = _idx(30)
    acct = pd.Series(0.0004, index=idx)
    acct.iloc[5] = -0.46
    bench = pd.Series(0.0004, index=idx)
    bench.iloc[5] = np.nan                      # hole exactly at the crash
    with pytest.raises(ValueError, match="interior gap"):
        ar.risk_refusals(acct, bench)


def test_leading_benchmark_gap_is_still_allowed():
    """The expected leading-gap truncation (the benchmark's history starts
    later than the account's: SPY starts 1993 while pools trade back to
    1962) must pass the interior-hole guard: truncating a PREFIX splices
    nothing."""
    idx = _idx(30)
    acct = pd.Series(0.0004, index=idx)
    bench = pd.Series(0.0004, index=idx)
    bench.iloc[:6] = np.nan
    out = ar.risk_refusals(acct, bench)
    assert out["n_sessions"] == 24


def test_time_in_market_is_measured_over_the_scored_window():
    """Reporting deployment over the account's FULL span next to a verdict
    computed on the benchmark-truncated window would mix two different
    windows in one report. Here the account is deployed on 100% of scored
    sessions but only 50% overall."""
    idx = _idx(400)
    strat, mask = _flat_strategy(idx, deployed_slice=slice(200, 400))
    bench = pd.Series(0.0003, index=idx[200:])          # covers the 2nd half
    out = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench,
                                     pd.Series(0.0001, index=idx)))
    assert out["time_in_market"] == pytest.approx(1.0)
    assert out["time_in_market_full_window"] == pytest.approx(0.5)
    assert out["window"]["benchmark_truncated"] is True
    assert out["window"]["scored_sessions"] == 200
    assert out["window"]["sessions_dropped_to_benchmark"] == 200


# ---------------------------------------------------------------- boundary

def test_module_never_records_anything():
    """ABI-1 computes; a caller decides what to record. It must not import
    the ledger, read the clock, or WRITE any file.

    NOTE the honest difference from benchmark_relative's own boundary test:
    this module DOES read one file (IRX.csv, in daily_cash_rate). Reads are
    permitted; writes and ledger access are not.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(ar))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    assert not any("trial_ledger" in n for n in imported), imported
    assert not any(n.split(".")[0] in ("datetime", "time") for n in imported), imported
    assert not hasattr(ar, "trial_ledger")
    # no write-shaped call anywhere in the source
    attrs = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for banned in ("to_csv", "to_json", "to_parquet", "write", "writelines",
                   "dump", "mkdir", "makedirs"):
        assert banned not in attrs, f"{banned} is a write"
    names = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in names, "bare open() -- could be a write"


def test_the_two_abi1_constants_are_pinned():
    assert ar.MIN_ACCOUNT_SESSIONS == 252
    assert ar.ABSOLUTE_MAX_DRAWDOWN_CAP == 0.35
    # ABI-1 must NOT quietly inherit the per-trade minimum
    assert ar.MIN_ACCOUNT_SESSIONS != constants.MIN_TRADES_FOR_SCORING


# ------------------- IRX is a BANK DISCOUNT quote --------

def test_coupon_equivalent_conversion_matches_the_published_relationship():
    """The stored ^IRX quote is a BANK DISCOUNT quote, not an investment
    yield. Verified against the U.S. Treasury's published Daily Treasury
    Bill Rates on 7 dates: mean absolute gap to the 13-week DISCOUNT figure
    was 0.0101 pp versus 0.1001 pp to the 13-week COUPON EQUIVALENT, with
    the stored value BELOW the coupon equivalent on 7 of 7 dates -- an
    understatement of the idle alternative, which always flatters the
    strategy.

    Pinned as literals, not recomputed from the formula, so this test
    cannot pass by copying whatever the implementation happens to do.
    """
    assert ar.coupon_equivalent_yield(0.0504) == pytest.approx(
        0.051_759_414_946_417_25, rel=1e-12)
    assert ar.coupon_equivalent_yield(0.02) == pytest.approx(
        0.020_380_814_115_807_84, rel=1e-12)
    assert ar.coupon_equivalent_yield(0.0) == 0.0
    # strictly above the discount quote for every positive rate
    grid = np.linspace(1e-6, 0.15, 40)
    assert (ar.coupon_equivalent_yield(grid) > grid).all()


def test_coupon_equivalent_survives_a_negative_discount_without_inverting():
    """^IRX is negative on 7 post-1993 sessions (minimum about -0.00105 on
    2020-03-26). Then the bill price is ABOVE par and the yield is negative
    -- arithmetically fine, but it must stay finite and must not flip sign."""
    ce = ar.coupon_equivalent_yield(-0.00105)
    assert ce == pytest.approx(-0.001_064_300_850_149_428, rel=1e-12)
    assert ce < 0.0 and math.isfinite(ce)


def test_coupon_equivalent_refuses_a_discount_that_zeroes_the_bill_price():
    """P = 1 - d*91/360 reaches 0 at an absurd discount rate (about 395.6%).
    Dividing by it would return inf and silently corrupt every downstream
    session, so it fails closed with a named error instead."""
    with pytest.raises(ValueError, match="bill price"):
        ar.coupon_equivalent_yield(4.0)
    with pytest.raises(ValueError, match="bill price"):
        ar.coupon_equivalent_yield(np.array([0.05, 4.0]))


def test_cash_rate_applies_the_coupon_equivalent_conversion(tmp_path):
    """The conversion is wired into the real reader, not just available
    beside it. The percent-to-decimal conversion in data_pipeline is a
    separate step and must not be double-applied: IRX.csv stores 0.0504 for
    5.04%, so a doubly-converted value would be ~100x too small."""
    d = _irx(tmp_path, rate=0.0504)
    out = ar.daily_cash_rate(_idx(5, "2020-06-01"), data_dir=d)
    assert out.iloc[0] * 365 == pytest.approx(
        ar.coupon_equivalent_yield(0.0504), rel=1e-12)


# ----------------- CALENDAR-day accrual, not trading-day -

def test_monday_accrues_the_whole_weekend(tmp_path):
    """Brokerage cash sweeps accrue on CALENDAR days: interest accrues daily
    from the day funds are deposited. So the
    return posted on a session covers the calendar days since the PREVIOUS
    session -- three of them on a Monday after a normal weekend."""
    d = _irx(tmp_path, rate=0.05, start="2019-01-01", n=1000)
    idx = pd.DatetimeIndex(["2020-06-16", "2020-06-17", "2020-06-18",
                            "2020-06-19", "2020-06-22"])   # Tue..Fri, Mon
    out = ar.daily_cash_rate(idx, data_dir=d)
    assert out.iloc[-1] == pytest.approx(3.0 * out.iloc[1], rel=1e-12)
    assert out.iloc[2] == pytest.approx(out.iloc[1], rel=1e-12)


def test_first_session_accrues_one_day_and_never_reaches_backwards(tmp_path):
    """The first session has no predecessor IN THE INDEX, so its elapsed-day
    count is unknowable. One day is the documented choice; silently reaching
    outside the index for a prior date is not."""
    d = _irx(tmp_path, rate=0.05, start="2019-01-01", n=1000)
    monday_first = ar.daily_cash_rate(
        pd.DatetimeIndex(["2020-06-08", "2020-06-09"]), data_dir=d)
    assert monday_first.iloc[0] == pytest.approx(monday_first.iloc[1],
                                                 rel=1e-12)
    # ... and it is exactly one day's worth, not three
    assert monday_first.iloc[0] == pytest.approx(
        ar.coupon_equivalent_yield(0.05) / 365, rel=1e-12)


def test_accrual_over_a_year_of_sessions_sums_to_365_days(tmp_path):
    """Summed over a year the elapsed-day counts total 365, which is why
    the aggregate effect of this accrual is small: calendar accrual is about
    correctness (which session carries a weekend), not magnitude."""
    d = _irx(tmp_path, rate=0.05, start="2018-01-01", n=1200)
    idx = pd.bdate_range("2020-01-02", "2021-01-01")
    out = ar.daily_cash_rate(idx, data_dir=d)
    days = float(out.sum()) / (ar.coupon_equivalent_yield(0.05) / 365.0)
    assert days == pytest.approx((idx[-1] - idx[0]).days + 1, rel=1e-9)


def test_divisor_is_365_and_is_the_documented_default(tmp_path):
    """365, not 360: the 365/91 factor in the coupon-equivalent conversion
    already puts the rate on an actual/365 INVESTMENT-yield basis, and 360
    is the money-market convention for the discount quote just converted
    away from."""
    assert ar.CALENDAR_DAYS_PER_YEAR == 365
    d = _irx(tmp_path, rate=0.05, start="2019-01-01", n=1000)
    idx = _idx(5, "2020-06-01")
    base = ar.daily_cash_rate(idx, data_dir=d)
    other = ar.daily_cash_rate(idx, data_dir=d, days_per_year=360)
    assert other.iloc[0] == pytest.approx(base.iloc[0] * 365.0 / 360.0,
                                          rel=1e-12)


def test_periods_parameter_is_rejected_not_silently_reinterpreted(tmp_path):
    """There is no `periods` parameter: `periods=252` cannot coexist with
    calendar-day accrual (a year would credit 365/252 = 1.45x the annual
    rate), so passing it raises TypeError rather than being silently
    reinterpreted."""
    d = _irx(tmp_path, rate=0.05, start="2019-01-01", n=1000)
    with pytest.raises(TypeError):
        ar.daily_cash_rate(_idx(5, "2020-06-01"), data_dir=d, periods=252)


def test_cash_rate_refuses_an_unsorted_or_intraday_index(tmp_path):
    """Calendar accrual needs a strictly ascending DAILY index: an unsorted
    one yields negative elapsed days (a NEGATIVE cash credit), and two
    timestamps inside one day yield zero (a silently uncredited session)."""
    d = _irx(tmp_path, rate=0.05, start="2019-01-01", n=1000)
    with pytest.raises(ValueError, match="ascending"):
        ar.daily_cash_rate(pd.DatetimeIndex(["2020-06-03", "2020-06-02"]),
                           data_dir=d)
    with pytest.raises(ValueError, match="ascending"):
        ar.daily_cash_rate(
            pd.DatetimeIndex(["2020-06-02 09:30", "2020-06-02 16:00"]),
            data_dir=d)


# ------- LOSS NETTING WITH CARRYFORWARD, and the LT/ST split

def _cal(n=800, start="2020-01-01"):
    """A CALENDAR-day index. The holding-period split is measured in calendar
    days, so a business-day index makes the 365/366 boundary awkward to
    state; nothing in the tax path requires trading sessions."""
    return pd.date_range(start, periods=n, freq="D")


def _trade(entry, exit_, net):
    return {"entry_date": pd.Timestamp(entry), "exit_date": pd.Timestamp(exit_),
            "net_ret": float(net)}


def test_netted_factors_reduce_to_the_unnetted_form_when_nothing_is_carried():
    """THE reduction netting must satisfy: with no losing trade there is no
    carryforward, so the netted walk must reproduce
    `tax_factor_by_exit_day`'s closed form -- which is itself pinned against
    backtest_engine.run_backtest's own post-tax per-trade multiple."""
    idx = _idx(60)
    pretax = pd.Series(0.004, index=idx)
    trades = pd.DataFrame([
        _trade(idx[2], idx[10], 0.05),
        _trade(idx[12], idx[25], 0.60),
        _trade(idx[30], idx[40], 9.0),
    ])
    netted = ar.netted_tax_factors(pretax_account_daily=pretax,
                                   trades_df=trades,
                                   short_rate=0.12, long_rate=0.12)
    unnetted = ar.tax_factor_by_exit_day(idx, trades, tax_haircut_rate=0.12)
    assert np.allclose(netted.to_numpy(), unnetted.to_numpy(), rtol=0,
                       atol=1e-15)


def test_carryforward_is_measured_in_dollars_not_in_return_units():
    """THE load-bearing constraint. A trade's loss in dollars is
    `entry_equity * (-net)`, and entry equity differs per trade, so a
    carryforward accumulated in per-trade RETURN units reproduces exactly the
    per-trade-quantity-on-a-compounding-curve error already
    guarded against once in this same function.

    Here the account grows 3.2x between a -50% trade and a +50% trade. In
    DOLLAR units the earlier loss shelters only part of the later gain, so the
    winner is still mostly taxed. In RETURN units the -0.5 would cancel the
    +0.5 outright and the factor would be exactly 1.0 -- flattering, and
    wrong.

    The pretax curve and the trade list are deliberately decoupled here so
    the arithmetic is isolated; the wiring is proved end-to-end by
    `test_evaluate_nets_losses_against_later_gains`.
    """
    idx = _idx(50)
    pretax = pd.Series(0.06, index=idx)
    trades = pd.DataFrame([_trade(idx[0], idx[10], -0.5),
                           _trade(idx[20], idx[30], 0.5)])
    f = ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades,
                              short_rate=0.12, long_rate=0.12)
    assert f.iloc[10] == 1.0                      # a loss is never taxed
    assert f.iloc[30] == pytest.approx(0.997_416_567_226_330_1, rel=1e-12)
    # NOT the return-unit answer, which would cancel outright at exactly 1.0
    assert f.iloc[30] != pytest.approx(1.0, abs=1e-9)


def test_carryforward_shelters_in_full_then_partially_then_is_exhausted():
    """Unused losses carry forward without limit to later gains. The annual
    ordinary-income offset limit and its calendar-year reset are real and
    deliberately NOT modelled (documented in the function)."""
    idx = _idx(60)
    pretax = pd.Series(0.0, index=idx)            # equity pinned at 1.0
    trades = pd.DataFrame([_trade(idx[0], idx[5], -0.4),    # carry 0.666667
                           _trade(idx[6], idx[15], 0.25),   # gain 0.2, sheltered
                           _trade(idx[16], idx[25], 1.5)])  # gain 0.6, partial
    f = ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades,
                              short_rate=0.12, long_rate=0.12)
    assert f.iloc[5] == 1.0
    assert f.iloc[15] == 1.0                      # fully sheltered
    assert f.iloc[25] == pytest.approx(0.984, rel=1e-12)


def test_a_loss_after_the_last_gain_is_carried_not_refunded():
    idx = _idx(60)
    pretax = pd.Series(0.0, index=idx)
    trades = pd.DataFrame([_trade(idx[0], idx[10], 0.5),
                           _trade(idx[20], idx[30], -0.4)])
    f = ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades,
                              short_rate=0.12, long_rate=0.12)
    assert f.iloc[10] == pytest.approx(0.96, rel=1e-12)   # taxed in full
    assert f.iloc[30] == 1.0                              # no refund
    assert float(f.prod()) == pytest.approx(0.96, rel=1e-12)


def test_long_term_boundary_is_strictly_greater_than_the_holding_days():
    """Few trades are held over a year, but those carry a large share of ALL
    gains (especially for trend-following families) and some
    families have zero long-term trades -- so a single flat rate mis-ranks
    family TYPES against each other. A round trip held exactly
    LONG_TERM_HOLDING_DAYS is still SHORT-term."""
    cal = _cal()
    pretax = pd.Series(0.0, index=cal)
    exactly = pd.DataFrame([_trade("2020-01-01", "2020-12-31", 0.5)])
    one_more = pd.DataFrame([_trade("2020-01-01", "2021-01-01", 0.5)])
    assert (pd.Timestamp("2020-12-31") - pd.Timestamp("2020-01-01")).days == 365
    short = ar.netted_tax_factors(pretax_account_daily=pretax,
                                  trades_df=exactly,
                                  short_rate=0.20, long_rate=0.05)
    long_ = ar.netted_tax_factors(pretax_account_daily=pretax,
                                  trades_df=one_more,
                                  short_rate=0.20, long_rate=0.05)
    assert short.loc["2020-12-31"] == pytest.approx(1 - 0.20 / 3, rel=1e-12)
    assert long_.loc["2021-01-01"] == pytest.approx(1 - 0.05 / 3, rel=1e-12)


def test_a_zero_rated_long_gain_leaves_short_term_shelter_intact():
    """CHARACTER-MATCHED netting.

    Guards against spending the carryforward BEFORE the tax rate is
    consulted: then a long-term gain rated at `constants.LTCG_TAX_RATE` =
    0.0 would exhaust the shelter pool while saving nothing, a later
    short-term gain would pay full freight, and the with-loss result would
    be BYTE-IDENTICAL to a no-loss control -- the loss delivering zero
    shelter.

    The control is the discriminating half: adding a losing trade must make
    the account's total tax strictly smaller.
    """
    cal = _cal()
    pretax = pd.Series(0.0, index=cal)
    with_loss = pd.DataFrame([
        _trade("2020-01-02", "2020-03-02", -0.4),      # short-term LOSS
        _trade("2020-04-01", "2021-06-01", 3.0),       # long-term gain, 0%
        _trade("2021-07-01", "2021-09-01", 1.0),       # short-term gain, 12%
    ])
    without_loss = pd.DataFrame([
        _trade("2020-04-01", "2021-06-01", 3.0),
        _trade("2021-07-01", "2021-09-01", 1.0),
    ])
    kw = dict(pretax_account_daily=pretax, short_rate=0.12,
              long_rate=constants.LTCG_TAX_RATE)
    sheltered = float(ar.netted_tax_factors(trades_df=with_loss, **kw).prod())
    control = float(ar.netted_tax_factors(trades_df=without_loss, **kw).prod())
    assert sheltered > control


def test_a_short_term_loss_does_not_shelter_a_long_term_gain():
    """The other half of character matching. A short-term loss is reserved
    for short-term gains, so a long-term gain pays its own rate even with a
    short loss outstanding.

    Exercised at a NON-ZERO long rate deliberately: at the shipped 0.0 both
    sides of this assertion collapse to 1.0 and the test would be vacuous.
    """
    cal = _cal()
    pretax = pd.Series(0.0, index=cal)
    st_loss_then_lt_gain = pd.DataFrame([
        _trade("2020-01-02", "2020-03-02", -0.4),          # short-term loss
        _trade("2020-04-01", "2021-06-01", 3.0),           # long-term gain
    ])
    lt_gain_alone = pd.DataFrame([_trade("2020-04-01", "2021-06-01", 3.0)])
    kw = dict(pretax_account_daily=pretax, short_rate=0.12, long_rate=0.05)
    with_loss = ar.netted_tax_factors(trades_df=st_loss_then_lt_gain, **kw)
    without = ar.netted_tax_factors(trades_df=lt_gain_alone, **kw)
    unsheltered = 1 - 0.05 * 0.75
    assert without.loc["2021-06-01"] == pytest.approx(unsheltered, rel=1e-12)
    assert with_loss.loc["2021-06-01"] == pytest.approx(unsheltered, rel=1e-12)


def test_a_long_term_loss_is_reserved_for_long_term_gains():
    """Character matching runs both ways, and THIS direction can cost the
    account: a single shared pool would let a long-term loss shelter a
    short-term gain at the higher short rate; character matching does not.

    That is the right reading of the real rule -- a carried loss keeps its
    character and offsets its own kind first -- and it is the conservative
    direction, since it can only raise the modelled tax bill.
    """
    cal = _cal()
    pretax = pd.Series(0.0, index=cal)
    trades = pd.DataFrame([
        _trade("2020-01-02", "2021-03-02", -0.4),      # long-term LOSS
        _trade("2021-04-01", "2021-06-01", 1.0),       # short-term gain
        _trade("2020-02-01", "2021-08-01", 1.0),       # long-term gain
    ])
    f = ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades,
                              short_rate=0.12, long_rate=0.05)
    # The short gain pays full freight -- the long loss's shelter is not
    # available to it, and waits for the long gain instead.
    assert f.loc["2021-06-01"] == pytest.approx(1 - 0.12 * 0.5, rel=1e-12)
    assert f.loc["2021-08-01"] > 1 - 0.05 * 0.5


def test_same_session_exits_process_losses_first_regardless_of_row_order():
    """Within one session there is no chronology, so the tie-break is stated
    and deterministic: losses are realised before gains. Otherwise the shelter
    a same-day loss provides would depend on the order rows happen to sit in
    trades_df."""
    idx = _idx(40)
    pretax = pd.Series(0.0, index=idx)
    gain_first = pd.DataFrame([_trade(idx[0], idx[10], 0.5),
                               _trade(idx[1], idx[10], -0.4)])
    f = ar.netted_tax_factors(pretax_account_daily=pretax,
                              trades_df=gain_first,
                              short_rate=0.12, long_rate=0.12)
    assert f.iloc[10] == 1.0            # the same-day loss sheltered the gain
    # and with no loss at all the same gain is taxed in full
    only_gain = pd.DataFrame([_trade(idx[0], idx[10], 0.5)])
    g = ar.netted_tax_factors(pretax_account_daily=pretax,
                              trades_df=only_gain,
                              short_rate=0.12, long_rate=0.12)
    assert g.iloc[10] == pytest.approx(0.96, rel=1e-12)


def test_two_same_session_winners_still_compose_multiplicatively():
    idx = _idx(40)
    pretax = pd.Series(0.0, index=idx)
    trades = pd.DataFrame([_trade(idx[0], idx[10], 0.05),
                           _trade(idx[1], idx[10], 0.03)])
    f = ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades,
                              short_rate=0.10, long_rate=0.10)
    expected = (1 - 0.10 * 0.05 / 1.05) * (1 - 0.10 * 0.03 / 1.03)
    assert f.iloc[10] == pytest.approx(expected, rel=1e-12)


def test_netted_factors_require_entry_dates_and_refuse_a_reversed_holding():
    """The split needs entry dates, which are already on every trade record
    (backtest_engine.run_backtest emits entry_date). Defaulting a missing
    column to short-term would silently mis-rate every long-term gain."""
    idx = _idx(40)
    pretax = pd.Series(0.0, index=idx)
    no_entry = pd.DataFrame({"exit_date": [idx[10]], "net_ret": [0.5]})
    with pytest.raises(ValueError, match="entry_date"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=no_entry)
    backwards = pd.DataFrame([_trade(idx[20], idx[10], 0.5)])
    with pytest.raises(ValueError, match="exit_date before entry_date"):
        ar.netted_tax_factors(pretax_account_daily=pretax,
                              trades_df=backwards)
    missing = pd.DataFrame([_trade(idx[0], idx[10], 0.5)])
    missing.loc[0, "entry_date"] = pd.NaT
    with pytest.raises(ValueError, match="entry_date"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=missing)


def test_netted_factors_refuse_a_total_loss_and_a_nan_net():
    """A net_ret at or below -100% makes the trade's own entry equity
    (exit_equity / (1 + net)) undefined. The un-netted form never divides by
    (1 + net) for a loser because it clips losses away; the netted one
    must, so this division-by-zero path exists only in the netted walk."""
    idx = _idx(40)
    pretax = pd.Series(0.0, index=idx)
    wiped = pd.DataFrame([_trade(idx[0], idx[10], -1.0)])
    with pytest.raises(ValueError, match="-100%"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=wiped)
    nan = pd.DataFrame([_trade(idx[0], idx[10], float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=nan)


def test_netted_factors_refuse_bad_rates_and_an_exit_outside_the_index():
    idx = _idx(40)
    pretax = pd.Series(0.0, index=idx)
    trades = pd.DataFrame([_trade(idx[0], idx[10], 0.5)])
    with pytest.raises(ValueError, match="short_rate"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades,
                              short_rate=1.0)
    with pytest.raises(ValueError, match="long_rate"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades,
                              long_rate=-0.1)
    off = pd.DataFrame([_trade(pd.Timestamp("1999-01-04"),
                               pd.Timestamp("1999-02-04"), 0.5)])
    with pytest.raises(ValueError, match="not in index"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=off)


def test_netted_factors_refuse_an_account_whose_equity_reaches_zero():
    """The walk divides the tax bill by the running equity. A wiped-out
    account makes that undefined; returning inf would silently poison every
    later session."""
    idx = _idx(40)
    pretax = pd.Series(0.0, index=idx)
    pretax.iloc[5] = -1.0
    trades = pd.DataFrame([_trade(idx[0], idx[10], 0.5)])
    with pytest.raises(ValueError, match="equity"):
        ar.netted_tax_factors(pretax_account_daily=pretax, trades_df=trades)


def test_netted_factors_default_to_the_module_constants():
    """The two rates are parameters so a future change alters an argument,
    not the design -- and they default to constants.py, never to a literal
    redefined here."""
    import inspect
    params = inspect.signature(ar.netted_tax_factors).parameters
    assert params["short_rate"].default == constants.STCG_TAX_RATE
    assert params["long_rate"].default == constants.LTCG_TAX_RATE
    assert params["long_term_days"].default == constants.LONG_TERM_HOLDING_DAYS
    assert not hasattr(ar, "LTCG_TAX_RATE")      # imported, not redefined
    assert not hasattr(ar, "LONG_TERM_HOLDING_DAYS")


def test_evaluate_nets_losses_against_later_gains():
    """End-to-end wiring: evaluate must run the NETTED walk, not the
    winners-only clip. Netting's benefit grows with turnover (more realized
    losses to offset), so winners-only is a turnover-dependent penalty that
    distorts RANKING, not a conservative default."""
    idx = _idx(300)
    strat, mask = _flat_strategy(idx, deployed_slice=slice(0, 300),
                                 ret=0.0008)
    cash = pd.Series(0.0, index=idx)
    bench = pd.Series(0.0001, index=idx)
    trades = pd.DataFrame([_trade(idx[10], idx[50], -0.10),
                           _trade(idx[60], idx[150], 0.30)])
    taxed = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench, cash,
                                       tax_policy="realization",
                                       trades_df=trades))
    winners_only = ar.account_daily_returns(
        strategy_daily=strat, in_market=mask, cash_daily=cash,
        tax_factor=ar.tax_factor_by_exit_day(
            idx, trades, tax_haircut_rate=constants.STCG_TAX_RATE))
    assert (ar.terminal_wealth(taxed["account_daily"])
            > ar.terminal_wealth(winners_only))
    assert taxed["ltcg_tax_rate"] == constants.LTCG_TAX_RATE
    assert taxed["tax_haircut_rate"] == constants.STCG_TAX_RATE


def test_evaluate_charges_the_long_term_rate_on_a_long_held_winner():
    """With LTCG_TAX_RATE pinned at 0.0 (derived from the already-pinned 12%
    short-term bracket assumption -- see constants.py), a winner held beyond
    a year carries NO modelled tax, so the taxed curve must equal the untaxed
    one exactly. That is a real consequence of the pinned bracket, called out
    rather than buried."""
    idx = _idx(500)
    strat, mask = _flat_strategy(idx, deployed_slice=slice(0, 500),
                                 ret=0.0008)
    cash = pd.Series(0.0, index=idx)
    bench = pd.Series(0.0001, index=idx)
    long_held = pd.DataFrame([_trade(idx[0], idx[400], 0.30)])
    assert (idx[400] - idx[0]).days > constants.LONG_TERM_HOLDING_DAYS
    taxed = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench, cash,
                                       tax_policy="realization",
                                       trades_df=long_held))
    untaxed = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench, cash))
    assert (ar.terminal_wealth(taxed["account_daily"])
            == pytest.approx(ar.terminal_wealth(untaxed["account_daily"]),
                             rel=1e-12))
    # the same trade held under a year IS taxed, so this is not a no-op path
    short_held = pd.DataFrame([_trade(idx[300], idx[400], 0.30)])
    st = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench, cash,
                                    tax_policy="realization",
                                    trades_df=short_held))
    assert (ar.terminal_wealth(st["account_daily"])
            < ar.terminal_wealth(untaxed["account_daily"]))


# ===========================================================================
# THE OUT-OF-SAMPLE HOLDOUT GATE
#
# Without this gate `evaluate` would have no out-of-sample check under any
# name.
#
# The criterion under test is deliberately NOT "did the account beat the
# index over the holdout": the daily log excess SUMS to the log of the
# terminal-wealth ratio, so that test IS terminal wealth, which may be
# reported but never be sole grounds for a verdict -- and
# `risk_refusals` already excludes it for exactly that reason. Restating it
# as "Sharpe > 0" does not escape it either, because Sharpe divides by a
# strictly positive number and a zero threshold is sign-preserving. Only a
# STRICTLY POSITIVE bar makes the denominator do any work, which is why the
# criterion is PSR >= DSR_PASS_THRESHOLD with sr_benchmark = 0.0.
# ===========================================================================

def _excess_series(index, excess_values, bench=None):
    """Build (account, benchmark) whose `log_excess` is EXACTLY
    `excess_values`, so a test controls the quantity under test directly
    instead of hoping a return construction lands near it."""
    e = np.asarray(excess_values, dtype=float)
    if bench is None:
        rng = np.random.default_rng(3)
        bench = pd.Series(0.0004 + 0.010 * rng.standard_normal(len(index)),
                          index=index)
    b = pd.Series(bench, index=index).astype(float)
    acct = pd.Series(np.expm1(np.log1p(b.to_numpy()) + e), index=index)
    return acct, b


def _holdout_idx(n=None):
    return pd.bdate_range("2016-01-04",
                          periods=constants.ABI1_HOLDOUT_SESSIONS
                          if n is None else n)


def _psr_direct(excess):
    from mining.benchmark_relative import daily_sharpe_moments
    from mining import dsr
    m = daily_sharpe_moments(np.asarray(excess, dtype=float))
    return dsr.probabilistic_sharpe_ratio(m["sr"], 0.0, m["n"], m["skew"],
                                          m["kurtosis"]), m


# ----------------------------------------------------------- the constant

def test_holdout_length_is_three_years():
    assert constants.ABI1_HOLDOUT_SESSIONS == 3 * 252 == 756
    # the module must DEFAULT to it, not carry a private copy
    import inspect
    sig = inspect.signature(ar.split_holdout)
    assert (sig.parameters["holdout_sessions"].default
            is constants.ABI1_HOLDOUT_SESSIONS)
    assert (inspect.signature(ar.evaluate).parameters["holdout_sessions"]
            .default is constants.ABI1_HOLDOUT_SESSIONS)
    # and it must NOT reuse the per-trade pipeline's 6-month holdout
    assert constants.ABI1_HOLDOUT_SESSIONS != constants.HOLDOUT_WINDOW_MONTHS


# --------------------------------------------------------- split_holdout

def test_split_takes_the_last_three_years_and_removes_them_from_in_sample():
    idx = _holdout_idx(1500)
    s = ar.split_holdout(idx)
    assert s["ok"] is True
    assert s["n_holdout"] == 756 and s["n_in_sample"] == 744
    # the holdout is the TAIL, and the two halves are disjoint and complete
    assert list(s["holdout"]) == list(idx[-756:])
    assert list(s["in_sample"]) == list(idx[:744])
    assert not set(s["in_sample"]) & set(s["holdout"])
    assert len(s["in_sample"]) + len(s["holdout"]) == len(idx)


def test_split_refuses_a_series_shorter_than_the_holdout_itself():
    s = ar.split_holdout(_holdout_idx(500))
    assert s["ok"] is False
    assert s["refusal"] == "holdout_too_short"
    assert "500" in s["reason"] and "756" in s["reason"]


def test_split_refuses_when_the_in_sample_remainder_falls_below_the_floor():
    """A holdout long enough on its own can still be illegal: carving it out
    must not leave fewer than MIN_ACCOUNT_SESSIONS to score on."""
    s = ar.split_holdout(_holdout_idx(756 + 251))
    assert s["ok"] is False
    assert s["refusal"] == "in_sample_too_short"
    # the two refusals are DISTINGUISHABLE -- "could not evaluate" has two
    # different causes and conflating them misroutes investigation
    assert s["refusal"] != "holdout_too_short"


def test_split_accepts_exactly_the_boundary():
    s = ar.split_holdout(_holdout_idx(756 + 252))
    assert s["ok"] is True
    assert s["n_in_sample"] == ar.MIN_ACCOUNT_SESSIONS


def test_split_refuses_a_nonsensical_holdout_length():
    with pytest.raises(ValueError, match="holdout_sessions"):
        ar.split_holdout(_holdout_idx(1500), holdout_sessions=1)


# ----------------------------------------------------------- holdout_gate

def test_holdout_gate_is_the_existing_psr_against_zero_with_no_deflation():
    """The bar is `dsr.probabilistic_sharpe_ratio` with sr_benchmark = 0.0
    and NO multiple-testing deflation: the holdout is ONE pre-registered
    confirmation test, not a search over candidates."""
    from mining import dsr
    idx = _holdout_idx()
    rng = np.random.default_rng(5)
    e = 0.0003 + 0.004 * rng.standard_normal(len(idx))
    acct, bench = _excess_series(idx, e)
    out = ar.holdout_gate(acct, bench)
    expected, m = _psr_direct(ar.log_excess(acct, bench))
    assert out["psr"] == expected
    assert out["sr_benchmark"] == 0.0
    # a deflated value would be STRICTLY smaller -- proving no deflation
    deflated = dsr.deflated_sharpe_ratio(
        sr=m["sr"], n=m["n"], skew=m["skew"], kurtosis=m["kurtosis"],
        n_trials=constants.ABI1_N_TRIALS,
        sr_variance=constants.DAILY_SR_VARIANCE)
    assert deflated < out["psr"]
    # moments come from the module's own convention: Pearson, bias=True,
    # which dsr.py requires to be consistent between skew and kurtosis
    assert out["kurtosis"] >= 1.0 + out["skew"] ** 2


def test_holdout_bar_is_strictly_positive_not_a_sign_check():
    """A holdout that is a BARE SIGN CHECK is too weak. Here the account genuinely
    beats the index over the holdout -- positive mean excess, positive sum,
    positive Sharpe -- and still fails, because beating it is not evidence
    of beating it repeatably."""
    idx = _holdout_idx()
    rng = np.random.default_rng(6)
    noise = 0.006 * rng.standard_normal(len(idx))
    noise -= noise.mean()
    e = noise + 0.00008                     # ~0.2 annualized excess Sharpe
    acct, bench = _excess_series(idx, e)
    out = ar.holdout_gate(acct, bench)
    assert out["evaluated"] is True
    assert out["log_excess_sum"] > 0.0           # it BEAT the index
    assert out["sharpe_daily_excess"] > 0.0      # a sign check would pass
    assert ar.terminal_wealth(acct) > ar.terminal_wealth(bench)
    assert out["psr"] < constants.DSR_PASS_THRESHOLD
    assert out["passed"] is False


def test_holdout_gate_passes_a_strong_enough_out_of_sample_edge():
    idx = _holdout_idx()
    rng = np.random.default_rng(6)
    noise = 0.006 * rng.standard_normal(len(idx))
    noise -= noise.mean()
    acct, bench = _excess_series(idx, noise + 0.00045)
    out = ar.holdout_gate(acct, bench)
    assert out["passed"] is True
    assert out["psr"] >= constants.DSR_PASS_THRESHOLD
    assert out["n_holdout"] == 756


def test_holdout_gate_refuses_a_window_shorter_than_required():
    idx = _holdout_idx(755)
    acct, bench = _excess_series(idx, np.full(755, 0.0005))
    out = ar.holdout_gate(acct, bench)
    assert out["evaluated"] is False
    assert out["passed"] is False
    assert out["refusal"] == "holdout_too_short"
    assert out["psr"] is None                 # never a number to misread


def test_holdout_gate_refuses_a_non_finite_excess():
    """A silent False here would read as "evaluated and failed"."""
    idx = _holdout_idx()
    e = np.full(len(idx), 0.0005)
    acct, bench = _excess_series(idx, e)
    acct.iloc[100] = np.inf
    out = ar.holdout_gate(acct, bench)
    assert out["evaluated"] is False
    assert out["refusal"] == "non_finite_excess"
    assert out["psr"] is None


def test_holdout_gate_refuses_a_zero_variance_holdout():
    """An account that tracks the benchmark exactly has an UNDEFINED excess
    Sharpe. `daily_sharpe_moments` would hand back sr 0.0, PSR would be
    exactly 0.5, and that reads as a real, measured failure. Refuse."""
    idx = _holdout_idx()
    rng = np.random.default_rng(12)
    bench = pd.Series(0.0004 + 0.010 * rng.standard_normal(len(idx)),
                      index=idx)
    # the SAME series, not a round-tripped reconstruction of it: expm1(log1p(x))
    # is not exactly x, and the float dust it leaves is real variance
    out = ar.holdout_gate(bench.copy(), bench)
    assert out["evaluated"] is False
    assert out["refusal"] == "zero_variance"
    assert out["psr"] is None
    assert out["passed"] is False


def test_holdout_gate_reports_the_annualized_sharpe_for_a_human():
    idx = _holdout_idx()
    rng = np.random.default_rng(8)
    acct, bench = _excess_series(
        idx, 0.0004 + 0.005 * rng.standard_normal(len(idx)))
    out = ar.holdout_gate(acct, bench)
    assert out["sharpe_annualized"] == pytest.approx(
        out["sharpe_daily_excess"] * math.sqrt(252))


def test_holdout_gate_propagates_a_wiped_out_account():
    idx = _holdout_idx()
    acct, bench = _excess_series(idx, np.full(len(idx), 0.0005))
    acct.iloc[10] = -1.0
    with pytest.raises(ValueError, match="log1p"):
        ar.holdout_gate(acct, bench)


# -------------------------------------------------- wiring into evaluate

def _long_case(seed, drift, *, n=1500, vol=0.006, holdout_drift=None,
               bench_seed=99):
    """A full-length account/benchmark pair with a controlled DAILY LOG
    EXCESS, so a test can set the in-sample and holdout edges separately."""
    idx = pd.bdate_range("2014-01-02", periods=n)
    rng = np.random.default_rng(bench_seed)
    bench = pd.Series(0.00035 + 0.011 * rng.standard_normal(n), index=idx)
    r = np.random.default_rng(seed)
    noise = vol * r.standard_normal(n)
    noise -= noise.mean()
    e = noise + drift
    if holdout_drift is not None:
        h = constants.ABI1_HOLDOUT_SESSIONS
        e[-h:] = noise[-h:] + holdout_drift
    acct, bench = _excess_series(idx, e, bench=bench)
    mask = pd.Series(True, index=idx)
    return idx, acct, mask, bench


def test_evaluate_scores_layer1_only_on_the_in_sample_window():
    """The holdout sessions must be EXCLUDED from the score, or the
    "out-of-sample" confirmation is confirming itself."""
    idx, strat, mask, bench = _long_case(31, 0.0006)
    out = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench,
                                     pd.Series(0.0, index=idx)))
    assert out["layer1"]["n_sessions"] == len(idx) - 756
    assert out["window"]["in_sample_sessions"] == len(idx) - 756
    assert out["window"]["holdout_sessions"] == 756
    assert out["window"]["holdout_split_applied"] is True
    # and the score really is a different number from the full-window one
    full = ar.score_account(out["log_excess"], n_trials=100.0,
                            sr_variance=0.0002)
    assert out["layer1"]["dsr_value"] != pytest.approx(full["dsr_value"])
    # the holdout block is reported whether or not it passes
    assert out["holdout"]["evaluated"] is True
    assert out["holdout"]["psr"] is not None


def test_evaluate_cuts_on_the_holdout_when_both_earlier_layers_clear():
    """The verdict the holdout layer exists to create: strong in sample, and the
    edge is simply not there out of sample."""
    idx, strat, mask, bench = _long_case(71, 0.0011, holdout_drift=0.00002)
    out = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench,
                                     pd.Series(0.0, index=idx)))
    assert out["layer1"]["scored"] is True
    assert out["layer1"]["dsr_value"] >= out["dsr_pass_threshold"]
    assert out["layer2"]["all_refusals_passed"] is True
    assert out["holdout"]["evaluated"] is True
    assert out["holdout"]["passed"] is False
    assert out["verdict"] == "cut_holdout"


def test_survived_is_unreachable_without_passing_the_holdout():
    """Same account, one difference: the holdout edge. Everything upstream
    is identical and clears; only the gate decides."""
    idx, strat, mask, bench = _long_case(39, 0.0011, holdout_drift=0.00002)
    failed = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench,
                                        pd.Series(0.0, index=idx)))
    idx2, strat2, mask2, bench2 = _long_case(39, 0.0011)
    passed = ar.evaluate(**_eval_kwargs(idx2, strat2, mask2, bench2,
                                        pd.Series(0.0, index=idx2)))
    assert failed["verdict"] == "cut_holdout"
    assert passed["verdict"] == "survived"
    assert passed["holdout"]["passed"] is True
    # both cleared layers 1 and 2 -- the gate is the only difference
    for o in (failed, passed):
        assert o["layer1"]["dsr_value"] >= o["dsr_pass_threshold"]
        assert o["layer2"]["all_refusals_passed"] is True


def test_evaluate_reports_not_evaluable_distinctly_from_a_holdout_failure():
    """"Could not evaluate" and "evaluated and failed" must never share a
    verdict -- conflating them would misroute any investigation of why a
    candidate did not survive.

    The candidate here is a GOOD one on every measure that can be taken: it
    clears the in-sample DSR bar and passes all three risk refusals. It
    still cannot be called a survivor, because 400 sessions cannot carry a
    756-session holdout and it was therefore never tested out of sample. An
    untested candidate has not earned the verdict, it has only avoided the
    test."""
    idx, strat, mask, bench = _long_case(57, 0.0011, n=400)
    out = ar.evaluate(**_eval_kwargs(idx, strat, mask, bench,
                                     pd.Series(0.0, index=idx)))
    assert out["layer1"]["dsr_value"] >= out["dsr_pass_threshold"]
    assert out["layer2"]["all_refusals_passed"] is True
    assert out["verdict"] == "holdout_not_evaluable"
    assert out["holdout"]["evaluated"] is False
    assert out["holdout"]["refusal"] == "holdout_too_short"
    assert out["window"]["holdout_split_applied"] is False
    # layer 1 still reports a number, over the FULL window, and says so
    assert out["window"]["in_sample_sessions"] == 400
    assert out["layer1"]["scored"] is True


def test_the_verdict_vocabulary_is_pinned_and_survived_is_last():
    assert ar.VERDICTS == ("not_scoreable", "cut_dsr", "cut_risk_refusal",
                           "holdout_not_evaluable", "cut_holdout",
                           "survived")


# ------------------------------------------------- the adversarial-input tests

def test_a_pure_cash_account_fails_the_holdout_while_passing_every_refusal(
        tmp_path):
    """THE test. A PURE-CASH account -- one that never trades -- passes the
    entire risk half (drawdown 0.0 against the benchmark's, zero red weeks,
    zero worst-8-week loss) while losing over a thousand basis points to the
    index, so a design that relied on the risk layer alone could not refuse
    it. This test uses a synthetic,
    equity-like benchmark (a round daily mean of 0.0008 and daily std of
    0.01) because data/raw/*.csv is gitignored and would not exist on a
    fresh clone."""
    d = _irx(tmp_path, rate=0.05, start="2013-01-01", n=3200)
    idx = pd.bdate_range("2014-01-02", periods=1500)
    rng = np.random.default_rng(77)
    bench = pd.Series(0.0008 + 0.0100 * rng.standard_normal(len(idx)),
                      index=idx)
    cash = ar.daily_cash_rate(idx, data_dir=d)
    never = pd.Series(False, index=idx)
    out = ar.evaluate(**_eval_kwargs(idx, pd.Series(0.0, index=idx), never,
                                     bench, cash))
    # the trap: the risk half alone is clean
    assert out["layer2"]["drawdown"]["passed"] is True
    assert out["layer2"]["red_week_streak"]["account"] == 0
    assert out["layer2"]["worst_stretch"]["account_loss"] == 0.0
    assert out["layer2"]["all_refusals_passed"] is True
    # and it lost badly to the index
    assert out["log_excess"].sum() < -0.10
    # the gate refuses it, on its own, out of sample
    h = ar.holdout_gate(out["account_daily"].iloc[-756:], bench.iloc[-756:])
    assert h["evaluated"] is True
    assert h["passed"] is False
    # not a near miss: far below the 0.95 bar. The account is refused
    # because sitting in cash has a
    # persistently NEGATIVE excess Sharpe, not because it lost a coin flip.
    assert h["psr"] < 0.10
    assert h["sharpe_annualized"] < -0.5
    assert out["verdict"] != "survived"


def test_a_steady_edge_passes_and_the_same_total_won_luckily_does_not():
    """The whole justification for the criterion. Two accounts with the
    IDENTICAL total log excess over the holdout -- so identical terminal
    wealth, which a terminal-wealth criterion could not tell apart. One earns it
    a little at a time; the other earns all of it in ten sessions."""
    idx = _holdout_idx()
    rng = np.random.default_rng(11)
    noise = 0.004 * rng.standard_normal(len(idx))
    noise -= noise.mean()
    total = 0.20
    steady = noise + total / len(idx)
    lucky = noise.copy()
    lucky[200:210] += total / 10
    assert steady.sum() == pytest.approx(lucky.sum(), abs=1e-12)

    a_s, b = _excess_series(idx, steady)
    a_l, _ = _excess_series(idx, lucky)
    g_s = ar.holdout_gate(a_s, b)
    g_l = ar.holdout_gate(a_l, b)

    # identical terminal wealth ratio -- the forbidden criterion cannot
    # separate these two at all
    assert (ar.log_excess(a_s, b).sum()
            == pytest.approx(ar.log_excess(a_l, b).sum(), abs=1e-12))
    assert g_s["passed"] is True, (
        "the steady winner must pass -- if it does not, the bar is wrong")
    assert g_l["passed"] is False, (
        "SAME total, concentrated in ten sessions, and the gate let it "
        "through: the criterion is not doing the job it was chosen for")
    assert g_l["psr"] < g_s["psr"]


def test_split_holdout_refuses_a_non_ascending_index():
    """A DESCENDING session index must be refused, not silently split.

    `split_holdout` slices POSITIONALLY (`idx[:n-h]` / `idx[n-h:]`), so on a
    non-ascending index the "holdout tail" is actually the OLDEST stretch and
    the scored "in-sample head" is the most recent -- a lookahead split that
    reports itself as a clean out-of-sample test. `evaluate` would go on to
    return a passing verdict off the back of it.

    This is the same failure SHAPE as the tz-aware-index leak this module
    already refuses in ``_validated_session_index``: the data is present and complete, every
    count is right, and no missingness check can fire -- the ORDER is wrong.
    A guard on ordering is the only thing that catches it.
    """
    ascending = pd.date_range("2018-01-01", periods=1200, freq="B")
    descending = ascending[::-1]
    with pytest.raises(ValueError, match="(?i)ascending"):
        ar.split_holdout(descending, holdout_sessions=756)
    # A shuffled index is the same defect without the tell-tale reversal.
    shuffled = ascending.to_list()
    shuffled[10], shuffled[900] = shuffled[900], shuffled[10]
    with pytest.raises(ValueError, match="(?i)ascending"):
        ar.split_holdout(pd.DatetimeIndex(shuffled), holdout_sessions=756)
    # The ascending case is untouched and still splits.
    ok = ar.split_holdout(ascending, holdout_sessions=756)
    assert ok["ok"] is True
    assert ok["n_holdout"] == 756
    assert ok["holdout"][0] > ok["in_sample"][-1]
