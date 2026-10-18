import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.stats

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.backtest_engine import (SignalResult, run_backtest,
                                     holdout_passes, daily_returns_by_exit_day)


def _make_df(closes):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                          "Close": closes}, index=idx)


def test_simple_entry_and_exit_round_trip():
    df = _make_df([100, 100, 105, 110, 100])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, in_market = run_backtest(df, signal_fn, cost_bps=0.0)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_date"] == df.index[1]
    assert row["exit_date"] == df.index[2]
    assert row["reason"] == "signal_exit"
    assert row["gross_ret"] == pytest.approx(0.05)
    assert list(in_market) == [False, True, True, False, False]


def test_cost_bps_reduces_net_return():
    df = _make_df([100, 100, 110])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_backtest(df, signal_fn, cost_bps=50.0)
    row = trades.iloc[0]
    assert row["gross_ret"] == pytest.approx(0.10)
    assert row["net_ret"] == pytest.approx(0.10 - 0.0050)


def test_tax_haircut_applies_only_to_positive_net_returns():
    df_win = _make_df([100, 100, 110])
    df_loss = _make_df([100, 100, 90])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    win_trades, _ = run_backtest(df_win, signal_fn, tax_haircut_rate=0.12)
    loss_trades, _ = run_backtest(df_loss, signal_fn, tax_haircut_rate=0.12)
    assert win_trades.iloc[0]["net_ret"] == pytest.approx(0.10 * 0.88)
    assert loss_trades.iloc[0]["net_ret"] == pytest.approx(-0.10)


def test_time_stop_forces_exit():
    df = _make_df([100] * 10)

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=False)

    trades, _ = run_backtest(df, signal_fn, time_stop=3)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "time_stop"
    assert trades.iloc[0]["held"] == 3


def test_open_position_at_end_of_data_closes_as_end_of_data():
    df = _make_df([100, 100, 105])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=False)

    trades, in_market = run_backtest(df, signal_fn)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "end_of_data"
    assert in_market.iloc[-1] == True


def test_no_reentry_while_in_position():
    df = _make_df([100, 100, 100, 100, 90])

    def signal_fn(frame, i):
        return SignalResult(entry=(i in (0, 1)), exit=(i == 3))

    trades, _ = run_backtest(df, signal_fn)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == df.index[1]


def test_empty_trades_df_has_expected_columns_when_no_signals_fire():
    df = _make_df([100, 100, 100])

    def signal_fn(frame, i):
        return SignalResult()

    trades, in_market = run_backtest(df, signal_fn)
    assert len(trades) == 0
    assert list(trades.columns) == ["entry_date", "exit_date", "entry_px",
                                     "exit_px", "held", "reason",
                                     "gross_ret", "net_ret"]
    assert not in_market.any()


def test_holdout_passes_requires_positive_holdout_and_positive_in_sample():
    assert holdout_passes(in_sample_net_edge_bps=30.0, holdout_net_edge_bps=10.0) is True
    assert holdout_passes(in_sample_net_edge_bps=30.0, holdout_net_edge_bps=-5.0) is False
    assert holdout_passes(in_sample_net_edge_bps=-10.0, holdout_net_edge_bps=20.0) is False


def test_daily_returns_by_exit_day_places_lump_sum_on_exit_date():
    df = _make_df([100, 100, 105, 110, 100])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_backtest(df, signal_fn)
    daily = daily_returns_by_exit_day(df.index, trades)
    assert daily.loc[df.index[2]] == pytest.approx(0.05)
    assert daily.drop(df.index[2]).eq(0.0).all()


def test_daily_returns_by_exit_day_raises_on_index_mismatch():
    df = _make_df([100, 100, 105])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=False)

    trades, _ = run_backtest(df, signal_fn)
    wrong_index = pd.date_range("2019-01-01", periods=3, freq="B")
    with pytest.raises(ValueError, match="not in the provided index"):
        daily_returns_by_exit_day(wrong_index, trades)


def test_signal_fn_receives_a_frame_truncated_exactly_to_the_current_bar():
    # Structural look-ahead prevention. Asserting that
    # `frame.iloc[len(frame)]` raises IndexError would be hollow: it is true
    # for ANY dataframe regardless of truncation, so deleting the actual
    # slicing (`df.iloc[:i]` -> `df`) would leave it green. This test records
    # the frame length `signal_fn` actually received at each call and
    # asserts it is EXACTLY i+1 (rows 0..i, no more) -- a naive
    # non-truncating implementation fails this immediately.
    df = _make_df([100, 100, 105, 110, 100])
    observed = []

    def recording_signal_fn(frame, i):
        observed.append((i, len(frame)))
        return SignalResult(entry=(i == 0), exit=(i == 2))

    run_backtest(df, recording_signal_fn)
    assert len(observed) > 0
    for i, frame_len in observed:
        assert frame_len == i + 1, (
            f"signal_fn at i={i} received a frame of length {frame_len}, "
            f"expected exactly {i + 1} (rows 0..i only -- no look-ahead)")


def test_time_stop_and_signal_exit_on_same_bar_prefers_signal_exit():
    df = _make_df([100, 100, 100, 100])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_backtest(df, signal_fn, time_stop=1)
    assert trades.iloc[0]["reason"] == "signal_exit"


def test_tax_haircut_rate_out_of_range_raises():
    df = _make_df([100, 100, 110])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    with pytest.raises(ValueError):
        run_backtest(df, signal_fn, tax_haircut_rate=1.0)
    with pytest.raises(ValueError):
        run_backtest(df, signal_fn, tax_haircut_rate=-0.1)


def test_empty_trades_df_columns_have_correct_dtypes_not_object():
    df = _make_df([100, 100, 100])

    def signal_fn(frame, i):
        return SignalResult()

    trades, _ = run_backtest(df, signal_fn)
    assert trades["net_ret"].dtype == np.float64
    assert trades["held"].dtype == np.int64
    # A downstream scipy.stats call on an empty float column must not
    # raise just because pandas inferred object dtype from zero rows.
    assert np.isnan(scipy.stats.skew(trades["net_ret"])) or len(trades) == 0


def test_short_trade_simple_pnl_profits_when_price_falls():
    df = _make_df([100, 100, 80])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_backtest(df, signal_fn, direction="short")
    assert trades.iloc[0]["gross_ret"] == pytest.approx(0.20)


def test_short_trade_simple_pnl_loses_when_price_rises():
    df = _make_df([100, 100, 120])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_backtest(df, signal_fn, direction="short")
    assert trades.iloc[0]["gross_ret"] == pytest.approx(-0.20)


def test_default_long_is_identical_to_explicit_long():
    df = _make_df([100, 100, 105, 110, 100])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    default_trades, default_market = run_backtest(
        df, signal_fn, cost_bps=12.5, tax_haircut_rate=0.12)
    explicit_trades, explicit_market = run_backtest(
        df, signal_fn, cost_bps=12.5, tax_haircut_rate=0.12,
        direction="long")
    pd.testing.assert_frame_equal(
        default_trades, explicit_trades, check_exact=True)
    pd.testing.assert_series_equal(
        default_market, explicit_market, check_exact=True)


def test_invalid_direction_fails_closed():
    with pytest.raises(ValueError, match="direction"):
        run_backtest(_make_df([100, 100]), lambda frame, i: SignalResult(),
                     direction="inverse")


@pytest.mark.parametrize(
    ("terminal_price", "expected_return"),
    [(50.0, 0.5), (300.0, -2.0)],
)
def test_short_fixed_share_pnl_handles_asymmetric_return_tails(
        terminal_price, expected_return):
    """A short return is fixed-share P&L, not the reciprocal return."""
    df = _make_df([100.0, 100.0, terminal_price])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_backtest(df, signal_fn, direction="short")
    assert trades.iloc[0]["gross_ret"] == pytest.approx(expected_return)


def test_invalid_direction_is_rejected_before_signal_invocation():
    calls = []

    def recording_signal_fn(frame, i):
        calls.append((len(frame), i))
        return SignalResult(entry=True)

    with pytest.raises(ValueError, match="direction"):
        run_backtest(_make_df([100.0, 100.0]), recording_signal_fn,
                     direction="inverse")
    assert calls == []


def _long_regression_case(case):
    if case == "no_trade":
        return (_make_df([100.0, 100.0, 100.0]),
                lambda frame, i: SignalResult(), {})
    if case == "signal_win_cost_tax":
        return (
            _make_df([100.0, 100.0, 120.0]),
            lambda frame, i: SignalResult(entry=(i == 0), exit=(i == 2)),
            {"cost_bps": 25.0, "tax_haircut_rate": 0.12},
        )
    if case == "signal_loss":
        return (
            _make_df([100.0, 100.0, 80.0]),
            lambda frame, i: SignalResult(entry=(i == 0), exit=(i == 2)),
            {},
        )
    if case == "time_stop":
        return (
            _make_df([100.0, 100.0, 105.0, 110.0]),
            lambda frame, i: SignalResult(entry=(i == 0), exit=False),
            {"time_stop": 1},
        )
    if case == "end_of_data":
        return (
            _make_df([100.0, 100.0, 105.0]),
            lambda frame, i: SignalResult(entry=(i == 0), exit=False),
            {},
        )
    raise AssertionError(f"unknown test case {case}")


@pytest.mark.parametrize(
    "case",
    ["no_trade", "signal_win_cost_tax", "signal_loss", "time_stop",
     "end_of_data"],
)
def test_default_direction_is_exactly_explicit_long_across_exit_paths(case):
    df, signal_fn, kwargs = _long_regression_case(case)
    default_trades, default_market = run_backtest(df, signal_fn, **kwargs)
    explicit_trades, explicit_market = run_backtest(
        df, signal_fn, direction="long", **kwargs)
    pd.testing.assert_frame_equal(
        default_trades, explicit_trades, check_exact=True,
        check_dtype=True, check_like=False)
    pd.testing.assert_series_equal(
        default_market, explicit_market, check_exact=True,
        check_dtype=True, check_names=True)


@pytest.mark.parametrize(
    ("closes", "time_stop", "expected_reason", "expected_gross",
     "expected_net"),
    [
        ([100.0, 100.0, 50.0], None, "end_of_data", 0.5,
         (0.5 - 0.01) * 0.8),
        ([100.0, 100.0, 300.0], None, "end_of_data", -2.0, -2.01),
        ([100.0, 100.0, 50.0, 40.0], 1, "time_stop", 0.5,
         (0.5 - 0.01) * 0.8),
        ([100.0, 100.0, 300.0, 400.0], 1, "time_stop", -2.0, -2.01),
    ],
)
def test_short_cost_tax_and_forced_exit_paths_use_fixed_share_pnl(
        closes, time_stop, expected_reason, expected_gross, expected_net):
    df = _make_df(closes)

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=False)

    trades, _ = run_backtest(
        df, signal_fn, direction="short", time_stop=time_stop,
        cost_bps=100.0, tax_haircut_rate=0.20)
    row = trades.iloc[0]
    assert row["reason"] == expected_reason
    assert row["gross_ret"] == pytest.approx(expected_gross)
    assert row["net_ret"] == pytest.approx(expected_net)


@pytest.mark.parametrize(
    ("terminal_price", "expected_gross", "expected_net"),
    [
        (50.0, 0.5, (0.5 - 0.01) * 0.8),
        (300.0, -2.0, -2.01),
    ],
)
def test_short_signal_exit_uses_same_cost_and_positive_only_tax_formula(
        terminal_price, expected_gross, expected_net):
    df = _make_df([100.0, 100.0, terminal_price, terminal_price])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_backtest(
        df, signal_fn, direction="short", cost_bps=100.0,
        tax_haircut_rate=0.20)
    row = trades.iloc[0]
    assert row["reason"] == "signal_exit"
    assert row["gross_ret"] == pytest.approx(expected_gross)
    assert row["net_ret"] == pytest.approx(expected_net)


@pytest.mark.parametrize("exit_mode", ["signal_exit", "time_stop", "end_of_data"])
def test_direction_changes_only_return_values_not_execution_contract(exit_mode):
    df = _make_df([100.0, 100.0, 150.0, 175.0])
    long_calls = []
    short_calls = []

    def make_signal(calls):
        def signal_fn(frame, i):
            calls.append((i, len(frame), frame.index[-1]))
            return SignalResult(
                entry=(i == 0), exit=(exit_mode == "signal_exit" and i == 2))
        return signal_fn

    kwargs = {"time_stop": 1} if exit_mode == "time_stop" else {}
    long_trades, long_market = run_backtest(
        df, make_signal(long_calls), direction="long", **kwargs)
    short_trades, short_market = run_backtest(
        df, make_signal(short_calls), direction="short", **kwargs)

    contract_columns = [
        "entry_date", "exit_date", "entry_px", "exit_px", "held", "reason"]
    pd.testing.assert_frame_equal(
        long_trades[contract_columns], short_trades[contract_columns],
        check_exact=True, check_dtype=True)
    pd.testing.assert_series_equal(
        long_market, short_market, check_exact=True, check_dtype=True)
    assert long_trades.dtypes.equals(short_trades.dtypes)
    assert long_calls == short_calls


def test_explicit_short_no_trade_preserves_empty_contract_and_signal_calls():
    df = _make_df([100.0, 101.0, 102.0])
    long_calls = []
    short_calls = []

    def make_no_trade_signal(calls):
        def signal_fn(frame, i):
            calls.append((i, len(frame), frame.index[-1]))
            return SignalResult()
        return signal_fn

    long_trades, long_market = run_backtest(
        df, make_no_trade_signal(long_calls), direction="long")
    short_trades, short_market = run_backtest(
        df, make_no_trade_signal(short_calls), direction="short")

    pd.testing.assert_frame_equal(
        short_trades, long_trades, check_exact=True, check_dtype=True)
    pd.testing.assert_series_equal(
        short_market, long_market, check_exact=True, check_dtype=True)
    assert list(short_trades.columns) == [
        "entry_date", "exit_date", "entry_px", "exit_px", "held",
        "reason", "gross_ret", "net_ret"]
    assert short_trades.dtypes.to_dict() == {
        "entry_date": np.dtype("datetime64[ns]"),
        "exit_date": np.dtype("datetime64[ns]"),
        "entry_px": np.dtype("float64"),
        "exit_px": np.dtype("float64"),
        "held": np.dtype("int64"),
        "reason": np.dtype("object"),
        "gross_ret": np.dtype("float64"),
        "net_ret": np.dtype("float64"),
    }
    assert short_calls == long_calls == [
        (0, 1, df.index[0]), (1, 2, df.index[1])]
    assert short_market.dtype == bool
    assert short_market.name == "in_market"
    assert not short_market.any()
