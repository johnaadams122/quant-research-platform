import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import signals
from mining.backtest_engine import run_backtest


def _make_df(opens, highs, lows, closes, start="2020-01-01", freq="B"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                          "Close": closes}, index=idx)


# ---------------------------------------------------------------------------
# fifty_two_week_high_momentum_signal_fn -- fixed constants, monthly-
# rebalance-only evaluation and a top-decile-of-range test, with no
# per-ticker parameters, so running it on many tickers applies the SAME
# mechanism to each, not a per-ticker re-interpretation.
# ---------------------------------------------------------------------------

def test_fifty_two_week_high_is_a_noop_on_non_rebalance_days():
    # Same calendar month for both rows -> not a rebalance day -> no-op
    # regardless of price level.
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    frame = pd.DataFrame({"Open": [100, 100], "High": [100, 100],
                           "Low": [100, 100], "Close": [100, 100]}, index=idx)
    result = signals.fifty_two_week_high_momentum_signal_fn(frame, 1)
    assert result.entry is False
    assert result.exit is False


def test_fifty_two_week_high_enters_when_price_near_the_trailing_high():
    # Trailing range low=50, high=150 (band=100); price=145 is within the
    # top 10% of that band (threshold = 150 - 0.10*100 = 140) -> entry.
    closes = [100] * 10 + [50] + [100] * 5 + [150] + [100] * 3 + [145]
    idx = list(pd.date_range("2020-01-01", periods=len(closes) - 1, freq="B"))
    idx.append(pd.Timestamp("2020-02-03"))  # forces a month change on the last row
    frame = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                           "Close": closes}, index=pd.DatetimeIndex(idx))
    result = signals.fifty_two_week_high_momentum_signal_fn(frame, len(frame) - 1)
    assert result.entry is True
    assert result.exit is False


def test_fifty_two_week_high_exits_when_price_far_from_the_trailing_high():
    closes = [100] * 10 + [50] + [100] * 5 + [150] + [100] * 3 + [60]
    idx = list(pd.date_range("2020-01-01", periods=len(closes) - 1, freq="B"))
    idx.append(pd.Timestamp("2020-02-03"))
    frame = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                           "Close": closes}, index=pd.DatetimeIndex(idx))
    result = signals.fifty_two_week_high_momentum_signal_fn(frame, len(frame) - 1)
    assert result.entry is False
    assert result.exit is True


def test_fifty_two_week_high_signal_is_reusable_across_tickers_unmodified():
    # The whole point of pooling: this signal function takes no
    # per-ticker parameters at all -- the identical function object
    # (not a per-ticker factory/copy) can be run against any ticker's
    # own OHLC frame.
    import inspect
    sig = inspect.signature(signals.fifty_two_week_high_momentum_signal_fn)
    assert list(sig.parameters) == ["frame", "i"]


# ---------------------------------------------------------------------------
# make_pead_signal_fn -- a factory for the pead_earnings_surprise_drift
# entry trigger taking ANY ticker's own real earnings-date set, so the
# identical entry-trigger logic (earnings-day return > +1 stdev of
# trailing 60d returns) is reused per ticker without a per-ticker copy.
# ---------------------------------------------------------------------------

def _pead_frame(n_days=65, earnings_idx=64, earnings_jump=0.10, start="2020-01-01"):
    """n_days closes: noisy ~0.5%/day-stdev random-walk drift for the
    trailing window (so the trailing-stdev threshold is meaningfully
    positive, not a degenerate ~0), then a jump of `earnings_jump` on
    `earnings_idx`."""
    rng = np.random.default_rng(0)
    daily_rets = rng.normal(loc=0.0, scale=0.005, size=n_days - 1)
    closes = [100.0]
    for r in daily_rets:
        closes.append(closes[-1] * (1 + r))
    closes[earnings_idx] = closes[earnings_idx - 1] * (1 + earnings_jump)
    idx = pd.date_range(start, periods=n_days, freq="B")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                          "Close": closes}, index=idx), idx


def test_pead_factory_enters_on_a_real_earnings_date_with_a_big_move():
    frame, idx = _pead_frame()
    earnings_date = idx[64].strftime("%Y-%m-%d")
    signal_fn = signals.make_pead_signal_fn(frozenset({earnings_date}))
    result = signal_fn(frame, len(frame) - 1)
    assert result.entry is True
    assert result.exit is False


def test_pead_factory_does_not_enter_off_an_earnings_date():
    frame, idx = _pead_frame()
    # Earnings-date set names a DIFFERENT day than the one with the jump.
    signal_fn = signals.make_pead_signal_fn(frozenset({"2019-01-01"}))
    result = signal_fn(frame, len(frame) - 1)
    assert result.entry is False


def test_pead_factory_does_not_enter_on_earnings_date_with_a_small_move():
    frame, idx = _pead_frame(earnings_jump=0.0001)  # far below the 1-stdev bar
    earnings_date = idx[64].strftime("%Y-%m-%d")
    signal_fn = signals.make_pead_signal_fn(frozenset({earnings_date}))
    result = signal_fn(frame, len(frame) - 1)
    assert result.entry is False


def test_pead_factory_requires_enough_trailing_history():
    # Fewer than trailing_window_days + 2 closes -> can't form the
    # trailing-stdev threshold -> no-op rather than a crash.
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    frame = pd.DataFrame({"Open": [100] * 10, "High": [100] * 10,
                           "Low": [100] * 10, "Close": [100] * 10}, index=idx)
    signal_fn = signals.make_pead_signal_fn(frozenset({idx[-1].strftime("%Y-%m-%d")}))
    result = signal_fn(frame, len(frame) - 1)
    assert result.entry is False
    assert result.exit is False


def test_pead_factory_never_sets_exit_true_time_stop_only():
    # Matches the candidate's own mechanism: exit is via the engine's
    # time_stop=hold_days, never a signal-driven exit.
    frame, idx = _pead_frame()
    earnings_date = idx[64].strftime("%Y-%m-%d")
    signal_fn = signals.make_pead_signal_fn(frozenset({earnings_date}))
    for i in range(len(frame)):
        result = signal_fn(frame.iloc[: i + 1], i)
        assert result.exit is False


def test_pead_factory_end_to_end_through_run_backtest_with_time_stop():
    frame, idx = _pead_frame(n_days=140, earnings_idx=64)
    earnings_date = idx[64].strftime("%Y-%m-%d")
    signal_fn = signals.make_pead_signal_fn(frozenset({earnings_date}))
    trades, _ = run_backtest(frame, signal_fn, time_stop=60)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] in ("time_stop", "end_of_data")
    # Entry happens at the NEXT session's open after the earnings date.
    assert trades.iloc[0]["entry_date"] == idx[65]


# ---------------------------------------------------------------------------
# make_index_addition_signal_fn -- single-shot per-ticker (entry_date,
# exit_date) factory for sp500_index_addition_effect.
# ---------------------------------------------------------------------------

def _dated_frame(n_days=30, start="2020-01-01"):
    idx = pd.date_range(start, periods=n_days, freq="B")
    closes = [100.0 + i * 0.1 for i in range(n_days)]
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                          "Close": closes}, index=idx), idx


def test_index_addition_factory_enters_on_the_real_entry_date():
    frame, idx = _dated_frame()
    entry_date = idx[10].strftime("%Y-%m-%d")
    exit_date = idx[20].strftime("%Y-%m-%d")
    signal_fn = signals.make_index_addition_signal_fn(entry_date, exit_date)
    result = signal_fn(frame.iloc[:11], 10)
    assert result.entry is True
    assert result.exit is False


def test_index_addition_factory_is_a_noop_on_any_other_date():
    frame, idx = _dated_frame()
    entry_date = idx[10].strftime("%Y-%m-%d")
    exit_date = idx[20].strftime("%Y-%m-%d")
    signal_fn = signals.make_index_addition_signal_fn(entry_date, exit_date)
    result = signal_fn(frame.iloc[:6], 5)
    assert result.entry is False
    assert result.exit is False


def test_index_addition_factory_exits_on_the_real_exit_date():
    frame, idx = _dated_frame()
    entry_date = idx[10].strftime("%Y-%m-%d")
    exit_date = idx[20].strftime("%Y-%m-%d")
    signal_fn = signals.make_index_addition_signal_fn(entry_date, exit_date)
    result = signal_fn(frame.iloc[:21], 20)
    assert result.exit is True
    assert result.entry is False


def test_index_addition_factory_end_to_end_through_run_backtest():
    frame, idx = _dated_frame(n_days=30)
    entry_date = idx[10].strftime("%Y-%m-%d")
    exit_date = idx[20].strftime("%Y-%m-%d")
    signal_fn = signals.make_index_addition_signal_fn(entry_date, exit_date)
    trades, _ = run_backtest(frame, signal_fn)
    assert len(trades) == 1
    # Next-open execution: entry lands the session AFTER entry_date.
    assert trades.iloc[0]["entry_date"] == idx[11]
    # Exit fires exactly at exit_date's close.
    assert trades.iloc[0]["exit_date"] == idx[20]
    assert trades.iloc[0]["reason"] == "signal_exit"


def test_index_addition_factory_produces_zero_trades_if_entry_date_never_occurs():
    # Ticker's own OHLC history doesn't cover the entry date at all (e.g. a
    # real data gap) -- must produce zero trades, never a crash or a
    # fabricated entry.
    frame, idx = _dated_frame(n_days=15)
    signal_fn = signals.make_index_addition_signal_fn("2099-01-01", "2099-02-01")
    trades, _ = run_backtest(frame, signal_fn)
    assert len(trades) == 0


def test_index_addition_factory_force_closes_at_end_of_data_if_exit_date_never_reached():
    frame, idx = _dated_frame(n_days=15)
    entry_date = idx[10].strftime("%Y-%m-%d")
    signal_fn = signals.make_index_addition_signal_fn(entry_date, "2099-02-01")
    trades, _ = run_backtest(frame, signal_fn)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "end_of_data"
    assert trades.iloc[0]["exit_date"] == idx[-1]
# ---------------------------------------------------------------------------
# sector_dispersion / dispersion_regime / make_sector_dispersion_cyclical_
# stance_signal_fn -- used by sector_dispersion_regime_cyclical_stance.
# ---------------------------------------------------------------------------

def test_sector_dispersion_computes_cross_sectional_std():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    panel = pd.DataFrame({
        "A": [0.01, 0.02],
        "B": [0.02, 0.02],
        "C": [0.03, -0.01],
    }, index=idx)
    result = signals.sector_dispersion(panel)
    expected = panel.std(axis=1, ddof=1)
    assert result.index.equals(idx)
    assert np.allclose(result.values, expected.values)
    # Sanity-check the ddof=1 sample-std formula by hand for row 0
    # (mean=0.02; squared devs 1e-4, 0, 1e-4; /(3-1)=1e-4; sqrt=0.01).
    assert result.iloc[0] == pytest.approx(0.01)


def test_sector_dispersion_raises_on_single_column():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    panel = pd.DataFrame({"A": [0.01, 0.02]}, index=idx)
    with pytest.raises(ValueError, match=">=2 columns"):
        signals.sector_dispersion(panel)


def test_sector_dispersion_is_nan_when_any_column_nan():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    panel = pd.DataFrame({
        "A": [0.01, np.nan],
        "B": [0.02, 0.02],
        "C": [0.03, -0.01],
    }, index=idx)
    result = signals.sector_dispersion(panel)
    assert not pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])


def _dispersion_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_dispersion_regime_is_nan_before_both_windows_fill():
    # smoothing_days=2, lookback_days=3 -> first valid regime needs
    # 2 + 3 - 1 = 4 observations; fewer than that is NaN everywhere.
    dispersion = _dispersion_series([0.01, 0.01, 0.01])
    regime = signals.dispersion_regime(dispersion, smoothing_days=2, lookback_days=3)
    assert regime.isna().all()


def test_dispersion_regime_labels_low_when_smoothed_below_long_run_average():
    # A long flat baseline (dispersion=0.02) followed by a drop to 0.01 --
    # once the drop has been in the smoothing window long enough, the
    # smoothed value should read below the (still baseline-dominated)
    # long-run average and classify 'low'.
    values = [0.02] * 10 + [0.01] * 4
    dispersion = _dispersion_series(values)
    regime = signals.dispersion_regime(dispersion, smoothing_days=2, lookback_days=5)
    assert regime.iloc[-1] == "low"


def test_dispersion_regime_labels_high_when_smoothed_above_long_run_average():
    values = [0.01] * 10 + [0.05] * 4
    dispersion = _dispersion_series(values)
    regime = signals.dispersion_regime(dispersion, smoothing_days=2, lookback_days=5)
    assert regime.iloc[-1] == "high"


def test_dispersion_regime_tie_break_is_high():
    # Constant dispersion -> smoothed always equals its own trailing
    # long-run average -> the documented tie-break goes to 'high'.
    dispersion = _dispersion_series([0.02] * 10)
    regime = signals.dispersion_regime(dispersion, smoothing_days=2, lookback_days=5)
    valid = regime.dropna()
    assert len(valid) > 0
    assert (valid == "high").all()


def _panel_returns_3seg(n1, v1, n2, v2, n3, v3, start="2020-01-01"):
    """A synthetic DISPERSION_PANEL-shaped returns frame (3 columns is
    enough to exercise sector_dispersion -- the mechanism doesn't care
    about panel width): `n1` days of dispersion `v1`, then `n2` days of
    `v2`, then `n3` days of `v3`. Column A holds +d/2, column B holds
    -d/2, column C holds 0, whose sample std (ddof=1, n=3) is exactly d
    for whichever segment a date falls in."""
    vals = [v1] * n1 + [v2] * n2 + [v3] * n3
    idx = pd.date_range(start, periods=len(vals), freq="B")
    d = np.array(vals, dtype=float)
    return pd.DataFrame({"A": d / 2, "B": -d / 2, "C": 0.0}, index=idx)


# Shared 3-segment fixture, hand-verified against
# dispersion_regime(smoothing_days=2, lookback_days=5): a 0.02-dispersion
# baseline (20 days, all NaN-then-'high' regime once warmed up), a
# 0.01-dispersion dip (10 days, regime settles to 'low' partway through),
# then a 0.08-dispersion spike (10 days, regime flips back to 'high').
_BASELINE_N, _BASELINE_V = 20, 0.02
_DIP_N, _DIP_V = 10, 0.01
_SPIKE_N, _SPIKE_V = 10, 0.08
_LOW_REGIME_DATE = pd.Timestamp("2020-01-31")   # inside the dip, regime='low'
_HIGH_REGIME_DATE = pd.Timestamp("2020-02-06")  # inside the spike, regime='high'


def _full_3seg_panel():
    return _panel_returns_3seg(_BASELINE_N, _BASELINE_V, _DIP_N, _DIP_V,
                                _SPIKE_N, _SPIKE_V)


def test_signal_fn_enters_on_low_regime():
    panel = _full_3seg_panel()
    signal_fn = signals.make_sector_dispersion_cyclical_stance_signal_fn(
        panel, smoothing_days=2, lookback_days=5)
    sliced = panel.loc[:_LOW_REGIME_DATE]
    xly_frame = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                               "Close": 100.0}, index=sliced.index)
    result = signal_fn(xly_frame, len(xly_frame) - 1)
    assert result.entry is True
    assert result.exit is False


def test_signal_fn_flat_when_regime_flips_high():
    panel = _full_3seg_panel()
    signal_fn = signals.make_sector_dispersion_cyclical_stance_signal_fn(
        panel, smoothing_days=2, lookback_days=5)
    sliced = panel.loc[:_HIGH_REGIME_DATE]
    xly_frame = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                               "Close": 100.0}, index=sliced.index)
    result = signal_fn(xly_frame, len(xly_frame) - 1)
    assert result.entry is False
    assert result.exit is True


def test_signal_fn_flat_when_date_missing_from_regime():
    panel = _panel_returns_3seg(10, 0.01, 0, 0.0, 0, 0.0)
    signal_fn = signals.make_sector_dispersion_cyclical_stance_signal_fn(
        panel, smoothing_days=2, lookback_days=5)
    # A date entirely outside the panel's own index.
    future_idx = pd.date_range("2030-01-01", periods=1, freq="B")
    xly_frame = pd.DataFrame({"Open": [100.0], "High": [100.0], "Low": [100.0],
                               "Close": [100.0]}, index=future_idx)
    result = signal_fn(xly_frame, 0)
    assert result.entry is False
    assert result.exit is True


def test_signal_fn_end_to_end_through_run_backtest():
    # Baseline 'high'-ish regime, a dip that settles to 'low' (entry),
    # then a spike back to 'high' (exit) -- exactly one round trip.
    panel = _full_3seg_panel()
    signal_fn = signals.make_sector_dispersion_cyclical_stance_signal_fn(
        panel, smoothing_days=2, lookback_days=5)
    rng = np.random.default_rng(1)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, size=len(panel)))
    xly_frame = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                               "Close": closes}, index=panel.index)
    trades, in_market = run_backtest(xly_frame, signal_fn)
    assert len(trades) == 1
    # Every trade's gross_ret must be finite -- entries/exits landed on
    # real prices, not NaN/inf from a misaligned regime lookup.
    assert trades["gross_ret"].apply(np.isfinite).all()
    # After the spike back to 'high', the strategy should be flat again.
    assert bool(in_market.iloc[-1]) is False
