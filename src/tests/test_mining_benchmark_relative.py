"""Tests for mining.benchmark_relative -- benchmark-relative and
daily-frequency analytics for the strategy-mining pipeline.

All data is deterministic and synthetic; no network, no clock, no files.
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

from mining import backtest_engine, dsr  # noqa: E402
from mining import benchmark_relative as br  # noqa: E402
from mining.backtest_engine import SignalResult  # noqa: E402


# ---------------------------------------------------------------- fixtures

def _closes(n, seed=7, drift=0.0004, vol=0.01):
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    return 100.0 * np.cumprod(1.0 + rets)


def _frame_gapped(n=300, seed=7):
    """OHLC with overnight gaps: Open != previous Close."""
    close = _closes(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1] * (1.0 + 0.003 * rng.standard_normal(n - 1))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    index = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _frame_gap_free(n=300, seed=11):
    """OHLC where every Open equals the previous Close (no gaps), so a
    continuously-invested strategy's mark-to-market equals buy-and-hold
    on every bar."""
    close = _closes(n, seed=seed)
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    index = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _always_entry(frame, i):
    return SignalResult(entry=True, exit=False)


def _never_entry(frame, i):
    return SignalResult(entry=False, exit=False)


def _compound_per_trade(mtm, trades_df):
    out = []
    for _, row in trades_df.iterrows():
        seg = mtm.loc[row["entry_date"]:row["exit_date"]]
        out.append(float(np.prod(1.0 + seg.to_numpy())) - 1.0)
    return np.asarray(out)


# ---------------------------------------------------------------- (a)

def test_long_daily_terms_compound_to_engine_gross_ret_gapped():
    df = _frame_gapped()
    trades, in_market = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=0.0, time_stop=5)
    assert len(trades) > 10
    mtm = br.mark_to_market_daily_returns(df, trades, direction="long")
    assert mtm.name == "mtm"
    assert mtm.index.equals(df.index)
    compounded = _compound_per_trade(mtm, trades)
    np.testing.assert_allclose(
        compounded, trades["gross_ret"].to_numpy(), rtol=0, atol=1e-12)
    # Zero on every flat day, non-zero somewhere in-market.
    assert (mtm[~in_market] == 0.0).all()
    assert (mtm[in_market] != 0.0).any()


def test_always_long_gap_free_matches_buy_and_hold_beta_one_alpha_zero():
    df = _frame_gap_free()
    trades, in_market = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=0.0, time_stop=5)
    mtm = br.mark_to_market_daily_returns(df, trades, direction="long")
    bnh = br.buy_and_hold_daily_returns(df)
    # In market on every bar but the first; with no gaps the daily terms
    # are identical to buy-and-hold everywhere.
    assert not in_market.iloc[0] and in_market.iloc[1:].all()
    np.testing.assert_allclose(mtm.to_numpy(), bnh.to_numpy(),
                               rtol=0, atol=1e-15)
    ols = br.ols_alpha_beta(mtm, bnh)
    assert ols["beta"] == pytest.approx(1.0, abs=1e-9)
    assert ols["alpha_daily"] == pytest.approx(0.0, abs=1e-12)
    assert ols["n"] == len(df)
    assert ols["r2"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- (b)

def test_flat_trades_gives_all_zero_mtm_and_zero_sharpe():
    df = _frame_gapped()
    trades, in_market = backtest_engine.run_backtest(df, _never_entry)
    assert trades.empty and not in_market.any()
    mtm = br.mark_to_market_daily_returns(df, trades)
    assert mtm.index.equals(df.index)
    assert (mtm == 0.0).all()
    assert br.sharpe_annualized(mtm) == 0.0
    summary = br.full_window_summary(
        mtm, br.buy_and_hold_daily_returns(df), in_market=in_market)
    assert summary["strategy_sharpe_ann"] == 0.0
    assert summary["strategy_cagr"] == 0.0
    assert summary["strategy_max_dd"] == 0.0
    assert summary["time_in_market"] == 0.0
    assert summary["n_days"] == len(df)
    assert summary["benchmark_sharpe_ann"] != 0.0
    # OLS still runs (benchmark has variance, n >= 30): beta 0, alpha 0.
    assert summary["ols"] is not None
    assert summary["ols"]["beta"] == 0.0
    assert summary["ols"]["alpha_daily"] == 0.0
    assert summary["ols"]["t_alpha_ols"] == 0.0


# ---------------------------------------------------------------- (c)

def test_two_x_benchmark_gives_beta_two_alpha_zero():
    df = _frame_gapped(n=400, seed=3)
    bnh = br.buy_and_hold_daily_returns(df)
    strat = 2.0 * bnh
    ols = br.ols_alpha_beta(strat, bnh)
    assert ols["beta"] == pytest.approx(2.0, abs=1e-9)
    assert ols["alpha_daily"] == pytest.approx(0.0, abs=1e-9)
    assert ols["alpha_annualized"] == pytest.approx(
        ols["alpha_daily"] * 252, abs=1e-12)
    assert ols["n"] == len(df)


# ---------------------------------------------------------------- (d)

def test_known_positive_alpha_has_large_t_stat():
    n = 600
    rng = np.random.default_rng(42)
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    bench = pd.Series(0.0003 + 0.01 * rng.standard_normal(n), index=index)
    alpha = 0.001
    strat = alpha + 1.2 * bench + 0.001 * rng.standard_normal(n)
    ols = br.ols_alpha_beta(strat, bench)
    assert ols["alpha_daily"] == pytest.approx(alpha, abs=2e-4)
    assert ols["beta"] == pytest.approx(1.2, abs=0.02)
    assert ols["t_alpha_ols"] > 10.0
    assert ols["t_alpha_hac"] > 10.0
    assert 0.0 < ols["r2"] <= 1.0


def _alpha_fixture_iid():
    """The test (d) data: iid residuals, known alpha 0.001, beta 1.2."""
    n = 600
    rng = np.random.default_rng(42)
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    bench = pd.Series(0.0003 + 0.01 * rng.standard_normal(n), index=index)
    strat = 0.001 + 1.2 * bench + 0.001 * rng.standard_normal(n)
    return strat, bench


def _alpha_fixture_ar1(n=600, rho=0.7):
    """Residuals follow an AR(1) with coefficient `rho`, so the HAC and
    classical standard errors are materially different."""
    rng = np.random.default_rng(99)
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    bench = pd.Series(0.0002 + 0.01 * rng.standard_normal(n), index=index)
    eps = np.empty(n)
    eps[0] = rng.standard_normal()
    innov = rng.standard_normal(n)
    for i in range(1, n):
        eps[i] = rho * eps[i - 1] + innov[i]
    strat = 0.0005 + 0.8 * bench + 0.004 * eps
    return strat, bench


def _bartlett_hac_t_alpha(strat, bench, lag):
    """Independent hand-computed Newey-West t-statistic for the intercept:
    OLS via lstsq, then V = (X'X)^-1 S (X'X)^-1 with S the Bartlett-
    weighted sum of autocovariances of the scores, weights 1 - l/(L+1)."""
    y = strat.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), bench.to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ coef
    scores = X * u[:, None]
    S = scores.T @ scores
    for ell in range(1, lag + 1):
        gamma = np.zeros((2, 2))
        for t in range(ell, len(y)):
            gamma += np.outer(scores[t], scores[t - ell])
        S += (1.0 - ell / (lag + 1.0)) * (gamma + gamma.T)
    xtx_inv = np.linalg.inv(X.T @ X)
    V = xtx_inv @ S @ xtx_inv
    return float(coef[0] / math.sqrt(V[0, 0]))


def test_t_alpha_ols_matches_linregress_and_alpha_annualizes_by_252():
    # Pins the residual-variance dof (n-2) and the 252x annualization on a
    # non-zero alpha against an exact reference t-stat: checks only where
    # alpha is 0 or the t-stat is merely "> 10" would let a wrong dof or a
    # 365x multiplier pass.
    import scipy.stats
    strat, bench = _alpha_fixture_iid()
    ols = br.ols_alpha_beta(strat, bench)
    lr = scipy.stats.linregress(bench.to_numpy(), strat.to_numpy())
    assert ols["t_alpha_ols"] == pytest.approx(
        lr.intercept / lr.intercept_stderr, rel=1e-9)
    assert ols["alpha_daily"] != 0.0
    assert ols["alpha_annualized"] == pytest.approx(
        ols["alpha_daily"] * 252, rel=1e-12)
    assert ols["alpha_annualized"] == pytest.approx(0.252, abs=0.06)


def test_hac_t_stat_on_autocorrelated_residuals_matches_bartlett_sandwich():
    # With iid residuals HAC ~ OLS, so dropping the correction, the gamma.T
    # term, or the Bartlett weights would be invisible. AR(1) residuals
    # separate them, and the hand-computed sandwich pins the
    # exact formula (lag 5 at n = 600 per floor(4 * (n/100) ** (2/9))).
    strat, bench = _alpha_fixture_ar1()
    ols = br.ols_alpha_beta(strat, bench)
    assert ols["hac_lag"] == 5
    assert ols["t_alpha_hac"] < 0.7 * ols["t_alpha_ols"]
    expected = _bartlett_hac_t_alpha(strat, bench, lag=5)
    assert ols["t_alpha_hac"] == pytest.approx(expected, rel=1e-9)
    # Sanity: a rectangular kernel or no correction gives a different value,
    # so the assertion above is discriminating, not tautological.
    assert expected != pytest.approx(ols["t_alpha_ols"], rel=1e-3)


def test_hac_lag_follows_the_newey_west_lag_rule():
    # floor(4 * (n / 100) ** (2 / 9)) -> 4, 5, 7; an exponent of 1/4 would
    # give 4, 6, 7 at the same n, so n = 600 discriminates.
    rng = np.random.default_rng(3)
    for n, expected_lag in ((100, 4), (600, 5), (1250, 7)):
        index = pd.date_range("2018-01-02", periods=n, freq="B")
        bench = pd.Series(0.01 * rng.standard_normal(n), index=index)
        strat = 0.0002 + bench + 0.005 * rng.standard_normal(n)
        ols = br.ols_alpha_beta(strat, bench)
        assert ols["hac_lag"] == expected_lag
        assert ols["hac_lag"] == math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))


def test_hac_and_ols_t_stats_match_statsmodels_reference():
    sm = pytest.importorskip("statsmodels.api")
    strat, bench = _alpha_fixture_ar1()
    ols = br.ols_alpha_beta(strat, bench)
    X = sm.add_constant(bench.to_numpy())
    hac = sm.OLS(strat.to_numpy(), X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    classical = sm.OLS(strat.to_numpy(), X).fit()
    assert ols["t_alpha_hac"] == pytest.approx(hac.tvalues[0], rel=1e-9)
    assert ols["t_alpha_ols"] == pytest.approx(classical.tvalues[0],
                                               rel=1e-9)
    assert ols["alpha_daily"] == pytest.approx(classical.params[0],
                                               rel=1e-9)
    assert ols["beta"] == pytest.approx(classical.params[1], rel=1e-9)


def test_ols_inner_joins_on_index():
    n = 100
    rng = np.random.default_rng(5)
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    bench = pd.Series(0.01 * rng.standard_normal(n), index=index)
    strat = (2.0 * bench).iloc[10:]          # 90 rows, subset of index
    ols = br.ols_alpha_beta(strat, bench)
    assert ols["n"] == 90
    assert ols["beta"] == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------- (e)

def test_time_matched_excess_equals_minus_cost_when_benchmark_is_traded_frame():
    df = _frame_gapped(n=200, seed=9)
    cost_bps = 12.5
    trades, _ = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=cost_bps, tax_haircut_rate=0.0,
        time_stop=7)
    excess = br.time_matched_benchmark_excess(trades, df)
    assert excess.index.equals(trades.index)
    np.testing.assert_allclose(
        excess.to_numpy(), np.full(len(trades), -cost_bps / 1e4),
        rtol=0, atol=1e-15)


def test_time_matched_excess_subtracts_long_benchmark_even_for_short():
    df = _frame_gapped(n=120, seed=2)
    trades, _ = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=0.0, time_stop=4, direction="short")
    excess = br.time_matched_benchmark_excess(trades, df)
    bench_long = (df["Close"].loc[trades["exit_date"]].to_numpy()
                  / df["Open"].loc[trades["entry_date"]].to_numpy() - 1.0)
    np.testing.assert_allclose(
        excess.to_numpy(), trades["net_ret"].to_numpy() - bench_long,
        rtol=0, atol=1e-15)


# ---------------------------------------------------------------- (f)

def test_missing_benchmark_date_raises():
    df = _frame_gapped(n=120, seed=2)
    trades, _ = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=0.0, time_stop=4)
    bench_short = df.iloc[:-1]        # last exit_date no longer present
    with pytest.raises(ValueError):
        br.time_matched_benchmark_excess(trades, bench_short)


def test_mtm_missing_date_raises():
    df = _frame_gapped(n=60, seed=2)
    trades = pd.DataFrame({
        "entry_date": [df.index[3]],
        "exit_date": [pd.Timestamp("2030-01-01")],
        "net_ret": [0.0], "gross_ret": [0.0]})
    with pytest.raises(ValueError):
        br.mark_to_market_daily_returns(df, trades)


def test_mtm_dates_absent_from_df_raise_with_membership_message():
    # A missing EXIT date alone is also rejected by the exit >= entry check
    # (get_indexer maps it to -1), so it cannot show the membership check
    # works on its own -- and with BOTH dates absent, code without that
    # check would silently write onto the last bar. Each case must raise
    # the membership error.
    df = _frame_gapped(n=60, seed=2)
    other = _frame_gapped(n=60, seed=5)
    other.index = pd.date_range("2030-01-01", periods=60, freq="B")
    assert not df.index.intersection(other.index).size
    trades_foreign = pd.DataFrame({
        "entry_date": [other.index[3]], "exit_date": [other.index[8]],
        "net_ret": [0.0], "gross_ret": [0.0]})
    with pytest.raises(ValueError, match="not in df.index"):
        br.mark_to_market_daily_returns(df, trades_foreign)
    trades_entry_missing = pd.DataFrame({
        "entry_date": [other.index[3]], "exit_date": [df.index[8]],
        "net_ret": [0.0], "gross_ret": [0.0]})
    with pytest.raises(ValueError, match="not in df.index"):
        br.mark_to_market_daily_returns(df, trades_entry_missing)
    trades_exit_missing = pd.DataFrame({
        "entry_date": [df.index[3]], "exit_date": [other.index[8]],
        "net_ret": [0.0], "gross_ret": [0.0]})
    with pytest.raises(ValueError, match="not in df.index"):
        br.mark_to_market_daily_returns(df, trades_exit_missing)


def test_overlapping_trades_raise():
    df = _frame_gapped(n=60, seed=2)
    trades = pd.DataFrame({
        "entry_date": [df.index[3], df.index[6]],
        "exit_date": [df.index[8], df.index[12]],
        "net_ret": [0.0, 0.0], "gross_ret": [0.0, 0.0]})
    with pytest.raises(ValueError):
        br.mark_to_market_daily_returns(df, trades)
    # Sharing the exit bar as the next entry bar is also an overlap (the
    # engine never re-enters on the bar it exits).
    trades2 = pd.DataFrame({
        "entry_date": [df.index[3], df.index[8]],
        "exit_date": [df.index[8], df.index[12]],
        "net_ret": [0.0, 0.0], "gross_ret": [0.0, 0.0]})
    with pytest.raises(ValueError):
        br.mark_to_market_daily_returns(df, trades2)


def test_exit_before_entry_raises():
    df = _frame_gapped(n=60, seed=2)
    trades = pd.DataFrame({
        "entry_date": [df.index[10]], "exit_date": [df.index[4]],
        "net_ret": [0.0], "gross_ret": [0.0]})
    with pytest.raises(ValueError):
        br.mark_to_market_daily_returns(df, trades)


def test_bad_direction_raises():
    df = _frame_gapped(n=60, seed=2)
    trades, _ = backtest_engine.run_backtest(df, _always_entry, time_stop=4)
    with pytest.raises(ValueError):
        br.mark_to_market_daily_returns(df, trades, direction="sideways")


def test_ols_rejects_small_n_and_zero_variance_benchmark():
    index = pd.date_range("2022-01-03", periods=29, freq="B")
    rng = np.random.default_rng(1)
    s = pd.Series(rng.standard_normal(29), index=index)
    b = pd.Series(rng.standard_normal(29), index=index)
    with pytest.raises(ValueError):
        br.ols_alpha_beta(s, b)
    index = pd.date_range("2022-01-03", periods=50, freq="B")
    s = pd.Series(rng.standard_normal(50), index=index)
    b = pd.Series(0.0, index=index)
    with pytest.raises(ValueError):
        br.ols_alpha_beta(s, b)


def test_full_window_summary_reports_ols_error_instead_of_raising():
    index = pd.date_range("2022-01-03", periods=20, freq="B")
    rng = np.random.default_rng(1)
    s = pd.Series(rng.standard_normal(20) * 0.01, index=index)
    b = pd.Series(rng.standard_normal(20) * 0.01, index=index)
    summary = br.full_window_summary(s, b)
    assert summary["ols"] is None
    assert isinstance(summary["ols_error"], str) and summary["ols_error"]
    assert summary["n_days"] == 20


# ---------------------------------------------------------------- (g)

def test_daily_sharpe_moments_matches_trade_stats_convention():
    import run_batch3_mining as batch3
    rng = np.random.default_rng(21)
    sample = 0.0005 + 0.012 * rng.standard_normal(250)
    sample[::17] += 0.05            # fat right tail
    index = pd.date_range("2023-01-02", periods=250, freq="B")
    got = br.daily_sharpe_moments(pd.Series(sample, index=index))
    ref = batch3._trade_stats(pd.DataFrame({"net_ret": sample}))
    assert got["n"] == 250
    assert got["sr"] == ref["sharpe"]
    assert got["skew"] == ref["skew"]
    assert got["kurtosis"] == ref["kurtosis"]
    assert got["kurtosis"] >= 1.0 + got["skew"] ** 2


def test_daily_sharpe_moments_kurtosis_floor_and_degenerate_cases():
    import run_batch3_mining as batch3
    # Two-point distribution: plug-in Pearson kurtosis equals 1 + skew**2
    # exactly in theory; floating point can land a hair under -- the floor
    # must hold either way, matching _trade_stats.
    sample = np.array([0.02] * 9 + [-0.05])
    got = br.daily_sharpe_moments(pd.Series(sample))
    ref = batch3._trade_stats(pd.DataFrame({"net_ret": sample}))
    assert got["kurtosis"] >= 1.0 + got["skew"] ** 2
    assert got["kurtosis"] == ref["kurtosis"]
    assert got["skew"] == ref["skew"]
    # Constant series: sr 0, skew 0, kurtosis 3 -- the same defaults.
    flat = br.daily_sharpe_moments(pd.Series(np.zeros(40)))
    assert flat == {"sr": 0.0, "n": 40, "skew": 0.0, "kurtosis": 3.0}
    assert br.daily_sharpe_moments(pd.Series([0.01])) == {
        "sr": 0.0, "n": 1, "skew": 0.0, "kurtosis": 3.0}
    assert br.daily_sharpe_moments(pd.Series([], dtype=float)) == {
        "sr": 0.0, "n": 0, "skew": 0.0, "kurtosis": 3.0}
    with pytest.raises(ValueError):
        br.daily_sharpe_moments(pd.Series([0.01, np.nan, 0.02]))


def test_daily_sharpe_moments_applies_kurtosis_floor_when_plugin_is_below(
        monkeypatch):
    # Pearson's inequality means the plug-in moments never
    # violate kurtosis >= 1 + skew**2 except by rounding, so no real sample
    # exercises the floor. Force the kurtosis estimator below the bound and
    # confirm both this module and _trade_stats floor it identically.
    import scipy.stats
    import run_batch3_mining as batch3
    rng = np.random.default_rng(31)
    sample = 0.001 + 0.01 * rng.standard_normal(60)
    sample[:5] += 0.04                                # non-zero skew
    true_skew = float(scipy.stats.skew(sample, bias=True))
    assert abs(true_skew) > 0.1
    monkeypatch.setattr(scipy.stats, "kurtosis", lambda *a, **k: 0.5)
    got = br.daily_sharpe_moments(pd.Series(sample))
    ref = batch3._trade_stats(pd.DataFrame({"net_ret": sample}))
    assert got["skew"] == pytest.approx(true_skew, rel=1e-12)
    assert got["kurtosis"] == 1.0 + got["skew"] ** 2
    assert got["kurtosis"] > 0.5
    assert got["kurtosis"] == ref["kurtosis"]
    assert got["skew"] == ref["skew"]


def test_daily_sharpe_moments_nonfinite_moments_fall_back_to_defaults():
    # Values of magnitude ~1e80 have a finite std (the std guard
    # does not fire) but overflow the plug-in kurtosis to nan; the fallback
    # must be (0.0, 3.0), with sr computed from the data as usual.
    import scipy.stats
    big = pd.Series([1.0, -1.0, 0.5, -0.3, 2.0]) * 1e80
    with np.errstate(over="ignore", invalid="ignore"):
        assert not math.isfinite(float(scipy.stats.kurtosis(
            big.to_numpy(), fisher=False, bias=True)))
        got = br.daily_sharpe_moments(big)
    x = big.to_numpy()
    assert got == {"sr": float(np.mean(x) / np.std(x, ddof=1)), "n": 5,
                   "skew": 0.0, "kurtosis": 3.0}
    assert got["sr"] != 0.0 and math.isfinite(got["sr"])


# ---------------------------------------------------------------- (h)

def test_short_direction_negates_price_terms_and_still_charges_cost():
    n = 12
    index = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100.0 + np.arange(n, dtype=float)        # strictly rising
    open_ = close - 0.5                              # Close > Open each bar
    df = pd.DataFrame({"Open": open_, "High": close + 1, "Low": open_ - 1,
                       "Close": close}, index=index)
    cost_bps = 10.0
    trades = pd.DataFrame({
        "entry_date": [index[3]], "exit_date": [index[6]],
        "net_ret": [0.0], "gross_ret": [0.0]})
    mtm = br.mark_to_market_daily_returns(
        df, trades, direction="short", cost_bps=cost_bps)
    # Entry day: -(Close/Open - 1) < 0 on an up-move.
    assert mtm.iloc[3] == pytest.approx(-(close[3] / open_[3] - 1.0))
    assert mtm.iloc[3] < 0.0
    assert mtm.iloc[4] == pytest.approx(-(close[4] / close[3] - 1.0))
    assert mtm.iloc[5] == pytest.approx(-(close[5] / close[4] - 1.0))
    # Exit day: negated price term MINUS the round-trip cost, charged on
    # entry capital, i.e. cost / G of the exit session's opening equity,
    # where G is the trade's growth over the sessions before the exit.
    cost = cost_bps / 1e4
    growth_short = float(np.prod(1.0 + mtm.iloc[3:6].to_numpy()))
    assert mtm.iloc[6] == pytest.approx(
        -(close[6] / close[5] - 1.0) - cost / growth_short)
    assert (mtm.iloc[:3] == 0.0).all() and (mtm.iloc[7:] == 0.0).all()
    # The long version charges the same entry-capital cost at exit.
    mtm_long = br.mark_to_market_daily_returns(
        df, trades, direction="long", cost_bps=cost_bps)
    growth_long = float(np.prod(1.0 + mtm_long.iloc[3:6].to_numpy()))
    assert mtm_long.iloc[6] == pytest.approx(
        (close[6] / close[5] - 1.0) - cost / growth_long)
    assert mtm_long.iloc[3] > 0.0
    # In compounded terms each direction loses exactly `cost` of entry
    # capital against its own cost-free path.
    free = {d: br.mark_to_market_daily_returns(df, trades, direction=d, cost_bps=0.0)
            for d in ("short", "long")}
    for d, path in (("short", mtm), ("long", mtm_long)):
        with_cost = float(np.prod(1.0 + path.iloc[3:7].to_numpy()))
        without = float(np.prod(1.0 + free[d].iloc[3:7].to_numpy()))
        assert with_cost == pytest.approx(without - cost, abs=1e-12)


def test_long_cost_is_charged_once_at_exit_in_compounded_terms():
    df = _frame_gapped(n=80, seed=4)
    cost_bps = 8.0
    trades, _ = backtest_engine.run_backtest(
        df, _always_entry, cost_bps=cost_bps, time_stop=3)
    mtm_free = br.mark_to_market_daily_returns(df, trades, cost_bps=0.0)
    mtm_cost = br.mark_to_market_daily_returns(df, trades, cost_bps=cost_bps)
    diff = (mtm_free - mtm_cost)
    exits = trades["exit_date"]
    cost = cost_bps / 1e4
    assert (diff.drop(exits) == 0.0).all()
    for _, row in trades.iterrows():
        span_free = mtm_free.loc[row["entry_date"]:row["exit_date"]]
        span_cost = mtm_cost.loc[row["entry_date"]:row["exit_date"]]
        growth_before_exit = float(np.prod(1.0 + span_free.iloc[:-1].to_numpy()))
        # exit session: the entry-capital cost as a share of that session's equity
        assert diff.loc[row["exit_date"]] == pytest.approx(cost / growth_before_exit,
                                                           rel=0, abs=1e-15)
        # compounded over the trade, the daily path equals the engine's net return
        compounded = float(np.prod(1.0 + span_cost.to_numpy()) - 1.0)
        assert compounded == pytest.approx(float(row["net_ret"]), abs=1e-12)


# ---------------------------------------------------------------- (i)

def test_daily_frequency_dsr_matches_direct_call():
    kwargs = dict(sr=0.09, n=1250, skew=-0.4, kurtosis=4.2,
                  n_trials=37.0, sr_variance=0.002)
    assert br.daily_frequency_dsr(**kwargs) == dsr.deflated_sharpe_ratio(
        **kwargs)
    # Guards propagate unchanged (impossible moment pair).
    with pytest.raises(ValueError):
        br.daily_frequency_dsr(sr=0.1, n=100, skew=1.0, kurtosis=1.5,
                               n_trials=5.0, sr_variance=0.01)


# ---------------------------------------------------------------- misc

def test_buy_and_hold_first_bar_zero_and_aligned():
    df = _frame_gapped(n=50, seed=8)
    bnh = br.buy_and_hold_daily_returns(df)
    assert bnh.index.equals(df.index)
    assert bnh.iloc[0] == 0.0
    expected = df["Close"].to_numpy()[1:] / df["Close"].to_numpy()[:-1] - 1
    np.testing.assert_allclose(bnh.to_numpy()[1:], expected, rtol=0,
                               atol=1e-15)
    assert not bnh.isna().any()


def test_cost_adjusted_benchmark_charges_only_the_first_bar():
    df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0], "High": [100.0, 101.0, 102.0],
        "Low": [100.0, 101.0, 102.0], "Close": [100.0, 101.0, 102.0],
    }, index=pd.bdate_range("2026-01-02", periods=3))
    out = br.cost_adjusted_benchmark_daily_returns(df, entry_cost_bps=10.0)
    gross = br.buy_and_hold_daily_returns(df)
    assert out.iloc[0] == pytest.approx(gross.iloc[0] - 10.0 / 1e4)
    assert out.iloc[1] == pytest.approx(gross.iloc[1])
    assert out.iloc[2] == pytest.approx(gross.iloc[2])
    assert list(out.index) == list(df.index)


def test_cost_adjusted_benchmark_at_zero_cost_matches_buy_and_hold_exactly():
    df = pd.DataFrame({
        "Open": [100.0, 101.0], "High": [100.0, 101.0],
        "Low": [100.0, 101.0], "Close": [100.0, 101.0],
    }, index=pd.bdate_range("2026-01-02", periods=2))
    out = br.cost_adjusted_benchmark_daily_returns(df, entry_cost_bps=0.0)
    gross = br.buy_and_hold_daily_returns(df)
    pd.testing.assert_series_equal(out, gross, check_names=False)


def test_cost_adjusted_benchmark_charges_a_full_round_trip_worth_of_cost_only_once():
    """The cost is charged EXACTLY ONCE regardless of series length -- a
    long-history benchmark comparison must not repeatedly re-charge entry
    cost on every bar, which would compound into a material, wrong drag
    over a multi-year account life."""
    df = pd.DataFrame({
        "Open": [100.0] * 5, "High": [100.0] * 5,
        "Low": [100.0] * 5, "Close": [100.0, 101.0, 100.0, 101.0, 100.0],
    }, index=pd.bdate_range("2026-01-02", periods=5))
    out = br.cost_adjusted_benchmark_daily_returns(df, entry_cost_bps=25.0)
    gross = br.buy_and_hold_daily_returns(df)
    diffs = (gross - out).to_numpy()
    assert diffs[0] == pytest.approx(25.0 / 1e4)
    np.testing.assert_allclose(diffs[1:], 0.0, atol=1e-15)


def test_sharpe_cagr_max_drawdown_known_values():
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    daily = pd.Series([0.10, -0.50, 0.25, 0.0], index=index)
    # Equity: 1.10, 0.55, 0.6875, 0.6875 -> max dd = 0.55/1.10 - 1 = -0.5
    assert br.max_drawdown(daily) == pytest.approx(-0.5)
    assert br.cagr(daily, periods=4) == pytest.approx(0.6875 - 1.0)
    mean, std = daily.mean(), daily.std(ddof=1)
    assert br.sharpe_annualized(daily, periods=252) == pytest.approx(
        mean / std * math.sqrt(252))
    assert br.sharpe_annualized(pd.Series([0.01, 0.01, 0.01])) == 0.0
    assert br.sharpe_annualized(pd.Series([0.01])) == 0.0
    assert br.max_drawdown(pd.Series([0.01, 0.02])) == 0.0
    assert br.max_drawdown(pd.Series([], dtype=float)) == 0.0
    assert br.cagr(pd.Series([], dtype=float)) == 0.0


def test_cagr_annualizes_with_periods_over_n():
    # The known-values case above uses periods == n, where the exponent
    # periods / n is 1 and dropping it would be invisible; these series
    # use n != periods.
    daily_504 = pd.Series(np.full(504, 0.001))
    daily_126 = pd.Series(np.full(126, 0.001))
    expected = 1.001 ** 252 - 1.0
    assert br.cagr(daily_504, periods=252) == pytest.approx(expected,
                                                             rel=1e-12)
    assert br.cagr(daily_126, periods=252) == pytest.approx(expected,
                                                             rel=1e-12)
    assert (1.001 ** 504 - 1.0) != pytest.approx(expected, rel=1e-3)


def test_max_drawdown_counts_starting_equity_as_a_peak():
    # A running max that forgot the 1.0 starting equity would report zero
    # drawdown for a strategy that lost 20% from inception; these series
    # fall first.
    assert br.max_drawdown(pd.Series([-0.20, 0.05, 0.05])) == pytest.approx(
        -0.20)
    assert br.max_drawdown(pd.Series([-0.10, -0.10])) == pytest.approx(-0.19)
    # Recovering above 1.0 later does not erase the initial drawdown.
    assert br.max_drawdown(pd.Series([-0.20, 0.5, 0.1])) == pytest.approx(
        -0.20)


def test_full_window_summary_in_market_mask_is_not_the_nonzero_fallback():
    # On the always-long fixture in_market.mean() equals (mtm != 0).mean()
    # exactly, so a summary that ignored the mask would still pass there.
    # Use masks that differ from the non-zero set.
    df = _frame_gap_free(n=120, seed=13)
    trades, in_market = backtest_engine.run_backtest(
        df, _always_entry, time_stop=5)
    mtm = br.mark_to_market_daily_returns(df, trades)
    bnh = br.buy_and_hold_daily_returns(df)
    nonzero_frac = float((mtm != 0.0).mean())
    assert nonzero_frac < 1.0
    all_true = pd.Series(True, index=df.index)
    assert br.full_window_summary(mtm, bnh, in_market=all_true)[
        "time_in_market"] == 1.0
    partial = pd.Series(False, index=df.index)
    partial.iloc[:30] = True
    assert br.full_window_summary(mtm, bnh, in_market=partial)[
        "time_in_market"] == pytest.approx(0.25)
    assert br.full_window_summary(mtm, bnh)[
        "time_in_market"] == pytest.approx(nonzero_frac)
    # A mask missing some joint days counts them as out of market -- and
    # filling those days must not go through pandas' deprecated object-
    # dtype downcast (a FutureWarning today, a dtype change later).
    import warnings
    short_mask = pd.Series(True, index=df.index[:60])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert br.full_window_summary(mtm, bnh, in_market=short_mask)[
            "time_in_market"] == pytest.approx(0.5)


def test_full_window_summary_n_days_is_the_joint_count():
    # With identical indexes, reporting len(strategy) instead of the
    # inner-join length would go unnoticed; these calls join unequal ones.
    n = 100
    rng = np.random.default_rng(17)
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    bench = pd.Series(0.01 * rng.standard_normal(n), index=index)
    strat = 0.0002 + 1.5 * bench + 0.003 * rng.standard_normal(n)
    bench_subset = bench.iloc[10:90]                    # 80 rows
    summary = br.full_window_summary(strat, bench_subset)
    assert summary["n_days"] == 80
    assert summary["ols"] is not None and summary["ols"]["n"] == 80
    assert summary["benchmark_sharpe_ann"] == pytest.approx(
        br.sharpe_annualized(bench_subset))
    assert summary["strategy_sharpe_ann"] == pytest.approx(
        br.sharpe_annualized(strat.iloc[10:90]))
    # Symmetric: a shorter strategy series is also joined, not padded.
    summary2 = br.full_window_summary(strat.iloc[:50], bench)
    assert summary2["n_days"] == 50


def test_full_window_summary_time_in_market_from_series_or_nonzero():
    df = _frame_gap_free(n=120, seed=13)
    trades, in_market = backtest_engine.run_backtest(
        df, _always_entry, time_stop=5)
    mtm = br.mark_to_market_daily_returns(df, trades)
    bnh = br.buy_and_hold_daily_returns(df)
    with_mask = br.full_window_summary(mtm, bnh, in_market=in_market)
    assert with_mask["time_in_market"] == pytest.approx(
        float(in_market.mean()))
    without = br.full_window_summary(mtm, bnh)
    assert without["time_in_market"] == pytest.approx(
        float((mtm != 0.0).mean()))
    for key in ("strategy_sharpe_ann", "benchmark_sharpe_ann",
                "strategy_cagr", "benchmark_cagr", "strategy_max_dd",
                "benchmark_max_dd", "n_days", "time_in_market", "ols"):
        assert key in with_mask
    assert with_mask["strategy_max_dd"] <= 0.0
    assert with_mask["ols"]["beta"] == pytest.approx(1.0, abs=1e-9)


def test_module_is_report_only():
    """The module must not import the ledger, the clock, or file I/O --
    checked on the real import statements and call sites, not on prose."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(br))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    assert not any("trial_ledger" in name for name in imported), imported
    assert not any(name.split(".")[0] in ("datetime", "time")
                   for name in imported), imported
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in called
    assert not hasattr(br, "trial_ledger")
