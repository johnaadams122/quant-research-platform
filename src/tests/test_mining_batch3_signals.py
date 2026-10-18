import os
import sys

import pandas as pd
import numpy as np
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import signals
from mining.backtest_engine import run_backtest


def _frame(index):
    prices = [100.0 + i for i in range(len(index))]
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices},
        index=pd.DatetimeIndex(index),
    )


def _price_frame(index, closes):
    closes = [float(value) for value in closes]
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes},
        index=pd.DatetimeIndex(index),
    )


def test_external_regime_uses_latest_observation_not_a_future_one():
    risk_on = pd.Series(
        [False, True],
        index=pd.to_datetime(["2024-01-02", "2024-01-05"]),
    )
    frame = _frame(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    result = signals.make_external_regime_signal_fn(risk_on)(frame, 2)
    assert result.entry is False
    assert result.exit is True


def test_external_regime_is_noop_before_its_first_observation():
    risk_on = pd.Series([True], index=pd.to_datetime(["2024-01-05"]))
    frame = _frame(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    result = signals.make_external_regime_signal_fn(risk_on)(frame, 1)
    assert result.entry is False
    assert result.exit is False


def test_nonledgerable_metadata_survives_external_signal_callable_boundary():
    risk_on = pd.Series(
        [True], index=pd.to_datetime(["2024-01-03"]), name="risk_on")
    risk_on.attrs.update({
        "ledgerable": False,
        "ledger_block_reason": "missing publication vintage",
        "source_url": "https://example.test/current-history.csv",
        "date_semantics": "reference_date_current_history",
    })
    signal_fn = signals.make_external_regime_signal_fn(risk_on)
    assert signal_fn.ledgerable is False
    assert signal_fn.ledger_block_reason == "missing publication vintage"
    assert signal_fn.source_metadata["source_url"].endswith("current-history.csv")
    assert signal_fn.source_metadata["date_semantics"] == (
        "reference_date_current_history")


def test_nonledgerable_metadata_survives_external_target_callable_boundary():
    risk_on = pd.Series(
        [True], index=pd.to_datetime(["2024-01-03"]), name="risk_on")
    risk_on.attrs.update({
        "ledgerable": False,
        "ledger_block_reason": "missing publication vintage",
        "source_url": "https://example.test/current-history.csv",
    })
    target = signals.make_external_regime_target_fn(
        risk_on, risk_ticker="SPY", defensive_ticker="SHY")
    assert target.ledgerable is False
    assert target.ledger_block_reason == "missing publication vintage"
    assert target.source_metadata["source_url"].endswith("current-history.csv")


def test_monthly_opex_dates_count_actual_sessions_across_a_holiday():
    # Third Friday is 2024-01-19; MLK Day (Jan 15) is absent. Four actual
    # sessions before expiry is therefore Friday Jan 12. Because entry is
    # next-open, the signal must fire one session earlier, Thursday Jan 11.
    index = pd.bdate_range("2024-01-01", "2024-01-31")
    index = index.difference(pd.DatetimeIndex(["2024-01-15"]))
    entries, exits = signals.monthly_opex_dates(
        index, entry_trading_days_before_opex=4)
    assert pd.Timestamp("2024-01-11") in entries
    assert pd.Timestamp("2024-01-19") in exits


def test_opex_signal_runs_end_to_end_on_precomputed_bounded_dates():
    index = pd.bdate_range("2024-01-01", "2024-01-31")
    index = index.difference(pd.DatetimeIndex(["2024-01-15"]))
    entries, exits = signals.monthly_opex_dates(
        index, entry_trading_days_before_opex=4)
    trades, _ = run_backtest(
        _frame(index), signals.make_date_window_signal_fn(entries, exits))
    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == pd.Timestamp("2024-01-12")
    assert trades.iloc[0]["exit_date"] == pd.Timestamp("2024-01-19")


def test_monthly_opex_dates_uses_catalog_entry_offset_not_a_hardcode():
    index = pd.bdate_range("2024-01-01", "2024-01-31")
    index = index.difference(pd.DatetimeIndex(["2024-01-15"]))
    entries, exits = signals.monthly_opex_dates(
        index, entry_trading_days_before_opex=2)
    assert entries == frozenset({pd.Timestamp("2024-01-16")})
    assert exits == frozenset({pd.Timestamp("2024-01-19")})


@pytest.mark.parametrize("offset", [0, -1])
def test_monthly_opex_dates_rejects_nonpositive_entry_offset(offset):
    with pytest.raises(ValueError, match="positive"):
        signals.monthly_opex_dates(
            pd.bdate_range("2024-01-01", "2024-01-31"),
            entry_trading_days_before_opex=offset)


def test_post_opex_window_dates_enters_on_expiry_exits_n_sessions_later():
    # Third Friday of Jan 2024 is 2024-01-19. With hold=5, five actual
    # (weekday) sessions after expiry lands on 2024-01-26 (Mon 22, Tue 23,
    # Wed 24, Thu 25, Fri 26).
    index = pd.bdate_range("2024-01-01", "2024-02-05")
    entries, exits = signals.post_opex_window_dates(
        index, hold_trading_days_after_opex=5)
    assert pd.Timestamp("2024-01-19") in entries
    assert pd.Timestamp("2024-01-26") in exits


def test_post_opex_window_signal_runs_end_to_end_on_precomputed_bounded_dates():
    index = pd.bdate_range("2024-01-01", "2024-02-05")
    entries, exits = signals.post_opex_window_dates(
        index, hold_trading_days_after_opex=5)
    trades, _ = run_backtest(
        _frame(index), signals.make_date_window_signal_fn(entries, exits))
    assert len(trades) == 1
    # Entry signal fires ON expiry Friday (Jan 19); next-open execution
    # fills the following session, Jan 22. Exit fills at the close of the
    # signalled exit session, Jan 26.
    assert trades.iloc[0]["entry_date"] == pd.Timestamp("2024-01-22")
    assert trades.iloc[0]["exit_date"] == pd.Timestamp("2024-01-26")


def test_post_opex_window_dates_uses_catalog_hold_not_a_hardcode():
    index = pd.bdate_range("2024-01-01", "2024-02-05")
    entries, exits = signals.post_opex_window_dates(
        index, hold_trading_days_after_opex=2)
    assert entries == frozenset({pd.Timestamp("2024-01-19")})
    assert exits == frozenset({pd.Timestamp("2024-01-23")})


@pytest.mark.parametrize("offset", [0, -1])
def test_post_opex_window_dates_rejects_nonpositive_hold(offset):
    with pytest.raises(ValueError, match="positive"):
        signals.post_opex_window_dates(
            pd.bdate_range("2024-01-01", "2024-02-05"),
            hold_trading_days_after_opex=offset)


def test_post_opex_window_dates_skips_a_month_whose_exit_falls_off_the_calendar():
    # Calendar ends right at expiry Friday (2024-01-19) -- there is no
    # session for a 5-day post-opex hold to exit on, so the month is
    # skipped entirely rather than inventing a truncated exit.
    index = pd.bdate_range("2024-01-01", "2024-01-19")
    entries, exits = signals.post_opex_window_dates(
        index, hold_trading_days_after_opex=5)
    assert entries == frozenset()
    assert exits == frozenset()


def test_post_opex_window_dates_never_shares_a_date_with_monthly_opex_dates_pre_window():
    # Distinctness sanity check: the pre-opex entry (monthly_opex_dates,
    # entry_trading_days_before_opex=4) and the post-opex entry
    # (post_opex_window_dates) must never fire on the same calendar date --
    # they are opposite-direction windows around the same expiry.
    index = pd.bdate_range("2024-01-01", "2024-03-31")
    pre_entries, pre_exits = signals.monthly_opex_dates(
        index, entry_trading_days_before_opex=4)
    post_entries, post_exits = signals.post_opex_window_dates(
        index, hold_trading_days_after_opex=5)
    assert pre_entries.isdisjoint(post_entries)
    assert pre_exits == post_entries  # both anchor on expiry Friday itself


def test_moving_average_risk_series_is_truncation_invariant():
    series = pd.Series(
        [100, 110, 90, 80, 120, 130],
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
        dtype=float,
    )
    full = signals.moving_average_risk_on_series(series, fast=2, slow=3)
    prefix = signals.moving_average_risk_on_series(series.iloc[:5], fast=2, slow=3)
    pd.testing.assert_series_equal(full.loc[:prefix.index.max()], prefix)


def test_money_market_risk_series_uses_causal_expanding_zscore():
    series = pd.Series(
        [100, 101, 103, 106, 110, 130, 132],
        index=pd.date_range("2024-01-01", periods=7, freq="MS"),
        dtype=float,
    )
    full = signals.money_market_risk_on_series(
        series, roc_months=3, riskoff_z_threshold=1.0)
    prefix = signals.money_market_risk_on_series(
        series.iloc[:6], roc_months=3, riskoff_z_threshold=1.0)
    pd.testing.assert_series_equal(full.loc[:prefix.index.max()], prefix)
    assert bool(full.loc[pd.Timestamp("2024-06-01")]) is False


def test_fed_liquidity_converts_rrp_billions_to_millions():
    index = pd.date_range("2024-01-03", periods=3, freq="7D")
    walcl = pd.Series([10_000, 10_500, 11_000], index=index, dtype=float)
    wtregen = pd.Series([0, 0, 0], index=index, dtype=float)
    rrp_billions = pd.Series([1, 2, 3], index=index, dtype=float)
    risk_on = signals.fed_net_liquidity_risk_on_series(
        walcl, wtregen, rrp_billions, ma_weeks=2)
    # Correct units yield net liquidity 9000, 8500, 8000 (contracting).
    assert bool(risk_on.iloc[-1]) is False


def test_ratio_risk_series_aligns_only_shared_past_dates():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    numerator = pd.Series([20, 21, 25, 26, 27], index=dates, dtype=float)
    denominator = pd.Series([10, 10, 10, 10], index=dates[:4], dtype=float)
    risk_on = signals.ratio_moving_average_risk_on_series(
        numerator, denominator, fast=2, slow=3)
    assert risk_on.index.max() == dates[3]


def test_monthly_absolute_momentum_requires_calendar_lookback_coverage():
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29",
        "2024-03-01",
    ])
    frames = {
        "HYG": _price_frame(dates, [100, 101, 102, 103, 110]),
        "BIL": _price_frame(dates, [100, 100, 100, 100, 100]),
    }
    target = signals.make_monthly_absolute_momentum_target_fn(
        "HYG", "BIL", lookback_months=2)
    assert target(frames) is None


def test_monthly_absolute_momentum_target_is_stable_for_the_month():
    dates = pd.to_datetime([
        "2023-12-29", "2024-01-31", "2024-02-29", "2024-03-01",
        "2024-03-04",
    ])
    target = signals.make_monthly_absolute_momentum_target_fn(
        "HYG", "BIL", lookback_months=2)
    decision_frames = {
        "HYG": _price_frame(dates[:4], [100, 101, 102, 110]),
        "BIL": _price_frame(dates[:4], [100, 100, 100, 100]),
    }
    later_frames = {
        "HYG": _price_frame(dates, [100, 101, 102, 110, 50]),
        "BIL": _price_frame(dates, [100, 100, 100, 100, 101]),
    }
    assert target(decision_frames) == "HYG"
    assert target(later_frames) == "HYG"


def test_monthly_relative_momentum_uses_calendar_anchor_and_stays_stable():
    dates = pd.to_datetime([
        "2023-12-29", "2024-01-31", "2024-02-29", "2024-03-01",
        "2024-03-04",
    ])
    target = signals.make_monthly_relative_momentum_target_fn(
        "TIP", "IEF", lookback_months=2)
    frames = {
        "TIP": _price_frame(dates, [100, 101, 102, 110, 50]),
        "IEF": _price_frame(dates, [100, 100, 101, 105, 200]),
    }
    assert target({ticker: frame.iloc[:4] for ticker, frame in frames.items()}) == "TIP"
    assert target(frames) == "TIP"


def test_monthly_volatility_target_uses_first_session_decision_only():
    dates = pd.bdate_range(end="2025-01-02", periods=291)
    dates = dates[dates != pd.Timestamp("2025-01-01")]
    dates = dates[-290:]
    returns = np.r_[
        np.resize([0.02, -0.02], len(dates) - 21),
        np.repeat(0.001, 20),
    ]
    closes = np.r_[100.0, 100.0 * np.cumprod(1.0 + returns)]
    target = signals.make_monthly_volatility_target_fn(
        "TLT", "SHY", vol_lookback_days=20,
        median_lookback_days=252)
    decision = {
        "TLT": _price_frame(dates, closes),
        "SHY": _price_frame(dates, np.repeat(100.0, len(dates))),
    }
    assert target(decision) == "TLT"

    later_dates = dates.append(pd.DatetimeIndex(["2025-01-03"]))
    later = {
        "TLT": _price_frame(later_dates, np.r_[closes, closes[-1] * 0.5]),
        "SHY": _price_frame(later_dates, np.repeat(100.0, len(later_dates))),
    }
    assert target(later) == "TLT"


def test_skew_target_returns_stable_label_without_future_data():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    target = signals.make_moving_average_regime_target_fn(
        risk_ticker="SPY", defensive_ticker="SHY", signal_ticker="^SKEW",
        fast=2, slow=3)
    frames = {
        "SPY": _price_frame(dates, [100, 100, 100, 100, 100]),
        "SHY": _price_frame(dates, [100, 100, 100, 100, 100]),
        "^SKEW": _price_frame(dates, [100, 100, 100, 200, 220]),
    }
    assert target({ticker: frame.iloc[:4] for ticker, frame in frames.items()}) == "SHY"
    assert target(frames) == "SHY"


def test_vxn_vix_ratio_target_uses_only_aligned_history_through_t():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    target = signals.make_ratio_regime_target_fn(
        risk_ticker="QQQ", defensive_ticker="SHY",
        numerator_ticker="^VXN", denominator_ticker="^VIX",
        fast=2, slow=3)
    frames = {
        "QQQ": _price_frame(dates, [100, 100, 100, 100]),
        "SHY": _price_frame(dates, [100, 100, 100, 100]),
        "^VXN": _price_frame(dates, [20, 20, 20, 40]),
        "^VIX": _price_frame(dates, [10, 10, 10, 10]),
    }
    assert target(frames) == "SHY"


def test_external_regime_target_is_asof_only_and_never_backward_fills():
    state = pd.Series(
        [True, False], index=pd.to_datetime(["2024-01-03", "2024-01-05"]))
    target = signals.make_external_regime_target_fn(
        state, risk_ticker="SPY", defensive_ticker="SHY")
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    frames = {
        "SPY": _price_frame(dates, [100] * 5),
        "SHY": _price_frame(dates, [100] * 5),
    }
    assert target({ticker: frame.iloc[:1] for ticker, frame in frames.items()}) is None
    assert target({ticker: frame.iloc[:3] for ticker, frame in frames.items()}) == "SPY"
    assert target(frames) == "SHY"


def test_finra_growth_uses_exact_prior_calendar_month_and_twentieth_later_session():
    monthly = pd.Series(
        [100.0, 120.0],
        index=pd.to_datetime(["2023-01-31", "2024-01-31"]),
    )
    calendar = pd.bdate_range("2024-01-31", "2024-03-01")
    growth = signals.finra_margin_debt_growth_available_series(
        monthly, calendar, lookback_months=12, lag_sessions=20)
    assert list(growth.index) == [pd.Timestamp("2024-02-28")]
    assert growth.iloc[0] == 0.2


def test_finra_growth_does_not_substitute_previous_observation_for_missing_month():
    monthly = pd.Series(
        [100.0, 120.0],
        index=pd.to_datetime(["2023-01-31", "2024-02-29"]),
    )
    calendar = pd.bdate_range("2024-02-29", "2024-04-30")
    growth = signals.finra_margin_debt_growth_available_series(
        monthly, calendar, lookback_months=12, lag_sessions=20)
    assert growth.empty


def test_lag_boolean_regime_uses_nth_later_session():
    regime = pd.Series(
        [True, False],
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    calendar = pd.bdate_range("2024-01-31", "2024-04-01")
    available = signals.lag_boolean_regime_to_availability(
        regime, calendar, lag_trading_days=3)
    # Business days strictly after 2024-01-31 (Wed): 02-01, 02-02, 02-05 (3rd).
    assert list(available.index) == [
        pd.Timestamp("2024-02-05"), pd.Timestamp("2024-03-05")]
    assert list(available) == [True, False]


def test_lag_boolean_regime_drops_entries_without_enough_later_sessions():
    regime = pd.Series([True], index=pd.to_datetime(["2024-03-28"]))
    calendar = pd.bdate_range("2024-03-28", "2024-04-01")  # only 2 later sessions
    available = signals.lag_boolean_regime_to_availability(
        regime, calendar, lag_trading_days=5)
    assert available.empty


def test_lag_boolean_regime_rejects_duplicate_reference_dates():
    regime = pd.Series(
        [True, False], index=pd.to_datetime(["2024-01-31", "2024-01-31"]))
    calendar = pd.bdate_range("2024-01-31", "2024-04-01")
    with pytest.raises(ValueError, match="duplicate reference dates"):
        signals.lag_boolean_regime_to_availability(
            regime, calendar, lag_trading_days=1)


def test_lag_boolean_regime_keeps_latest_reference_date_on_collision():
    # Both reference dates fall in the SAME calendar gap (01-02, 01-03 have
    # no session of their own), so both see identical later_sessions and
    # collide onto the single next real session, 01-05 -- the LATER
    # reference date (01-03, value False) must win, since a stale reading
    # is superseded by the fresher one before either is ever tradeable
    # (this is the real-world shape of a data series that predates the
    # traded instrument's own inception).
    regime = pd.Series(
        [True, False],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    calendar = pd.to_datetime(["2024-01-01", "2024-01-05"])
    available = signals.lag_boolean_regime_to_availability(
        regime, calendar, lag_trading_days=1)
    assert list(available.index) == [pd.Timestamp("2024-01-05")]
    assert list(available) == [False]


def test_lag_boolean_regime_rejects_nonpositive_lag():
    regime = pd.Series([True], index=pd.to_datetime(["2024-01-31"]))
    with pytest.raises(ValueError, match="lag_trading_days must be positive"):
        signals.lag_boolean_regime_to_availability(
            regime, pd.bdate_range("2024-01-31", "2024-02-05"),
            lag_trading_days=0)


def test_lag_boolean_regime_preserves_source_attrs():
    """Dropping .attrs here breaks the
    ledgerable/point_in_time metadata chain that make_external_regime_
    signal_fn's _attach_source_metadata relies on -- a signal_fn built
    from a non-ledgerable source would silently default to ledgerable=True
    (missing attr) instead of correctly failing run_batch3_mining's
    getattr(signal_fn, "ledgerable", True) is False gate. Not exploitable
    today (the FRED allowlist gate runs upstream, at fetch time), but the
    metadata must still propagate for defense in depth."""
    regime = pd.Series([True], index=pd.to_datetime(["2024-01-31"]))
    regime.attrs.update({
        "ledgerable": False, "point_in_time": True,
        "ledger_block_reason": "test reason"})
    calendar = pd.bdate_range("2024-01-31", "2024-02-10")
    available = signals.lag_boolean_regime_to_availability(
        regime, calendar, lag_trading_days=1)
    assert available.attrs == {
        "ledgerable": False, "point_in_time": True,
        "ledger_block_reason": "test reason"}


def test_lag_boolean_regime_is_truncation_invariant():
    regime = pd.Series(
        [True, False, True],
        index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"]))
    calendar = pd.bdate_range("2024-01-31", "2024-05-01")
    full = signals.lag_boolean_regime_to_availability(
        regime, calendar, lag_trading_days=4)
    prefix = signals.lag_boolean_regime_to_availability(
        regime.iloc[:2], calendar, lag_trading_days=4)
    pd.testing.assert_series_equal(full.loc[:prefix.index.max()], prefix)


def test_cap_pre_inception_history_trims_to_the_cutoff():
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.to_datetime(["1980-01-01", "1985-01-01", "1990-01-01", "1995-01-01"]))
    trimmed = signals.cap_pre_inception_history(
        series, "2000-01-01", max_years_before=10)
    # cutoff = 1990-01-01; 1990-01-01 itself is kept (>=), 1985 and 1980 dropped.
    assert list(trimmed.index) == [
        pd.Timestamp("1990-01-01"), pd.Timestamp("1995-01-01")]
    assert list(trimmed) == [3.0, 4.0]


def test_cap_pre_inception_history_is_a_noop_when_history_is_already_short():
    series = pd.Series([1.0], index=pd.to_datetime(["1999-01-01"]))
    trimmed = signals.cap_pre_inception_history(
        series, "2000-01-01", max_years_before=10)
    assert list(trimmed.index) == [pd.Timestamp("1999-01-01")]


def test_cap_pre_inception_history_rejects_nonpositive_years():
    series = pd.Series([1.0], index=pd.to_datetime(["1999-01-01"]))
    with pytest.raises(ValueError, match="max_years_before must be positive"):
        signals.cap_pre_inception_history(series, "2000-01-01", max_years_before=0)


def test_cap_pre_inception_history_preserves_attrs():
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["1980-01-01", "1995-01-01"]))
    series.attrs.update({"ledgerable": True, "point_in_time": True})
    trimmed = signals.cap_pre_inception_history(
        series, "2000-01-01", max_years_before=10)
    assert trimmed.attrs == {"ledgerable": True, "point_in_time": True}


def test_money_market_roc_requires_exact_calendar_month():
    series = pd.Series(
        [100.0, 110.0, 120.0, 130.0, 140.0],
        index=pd.to_datetime([
            "2023-12-01", "2024-01-01", "2024-03-01", "2024-04-01",
            "2024-05-01",
        ]),
    )
    out = signals.money_market_risk_on_series(
        series, roc_months=3, riskoff_z_threshold=1.0)
    assert pd.Timestamp("2024-03-01") not in out.index


def test_fed_liquidity_requires_consecutive_calendar_weeks():
    dates = pd.to_datetime([
        "2024-01-03", "2024-01-17", "2024-01-24", "2024-01-31",
    ])
    walcl = pd.Series([100, 101, 102, 103], index=dates, dtype=float)
    wtregen = pd.Series([10, 10, 10, 10], index=dates, dtype=float)
    rrp = pd.Series([0, 0, 0, 0], index=dates, dtype=float)
    out = signals.fed_net_liquidity_risk_on_series(
        walcl, wtregen, rrp, ma_weeks=3)
    assert pd.Timestamp("2024-01-24") not in out.index
    assert pd.Timestamp("2024-01-31") in out.index


def test_fed_liquidity_uses_only_prior_values_across_different_release_dates():
    walcl = pd.Series(
        [100.0, 150.0, 200.0],
        index=pd.to_datetime(["2024-01-03", "2024-01-10", "2024-01-17"]),
    )
    wtregen = pd.Series(
        [10.0, 20.0, 999.0],
        index=pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-18"]),
    )
    rrp = pd.Series(
        [0.0, 0.01, 99.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-15", "2024-01-19"]),
    )
    full = signals.fed_net_liquidity_risk_on_series(
        walcl, wtregen, rrp, ma_weeks=2)
    truncated = signals.fed_net_liquidity_risk_on_series(
        walcl, wtregen.loc[:"2024-01-17"], rrp.loc[:"2024-01-17"],
        ma_weeks=2)
    pd.testing.assert_series_equal(full, truncated)
    assert list(full.index) == [pd.Timestamp("2024-01-17")]
    assert bool(full.iloc[0]) is True


def test_clean_series_rejects_infinity_with_actionable_date():
    series = pd.Series(
        [100.0, float("inf"), 102.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    with pytest.raises(ValueError, match="non-finite.*2024-01-02"):
        signals.moving_average_risk_on_series(series, fast=1, slow=2)


def test_finra_risk_on_is_strictly_above_catalog_growth_threshold():
    monthly = pd.Series(
        [100.0, 100.0, 100.0, 101.0],
        index=pd.to_datetime([
            "2023-01-31", "2023-02-28", "2024-01-31", "2024-02-29",
        ]),
    )
    calendar = pd.bdate_range("2024-01-31", "2024-04-01")
    risk_on = signals.finra_margin_debt_risk_on_series(
        monthly, calendar, lookback_months=12, lag_sessions=20,
        growth_threshold=0.0)
    assert list(risk_on.index) == [
        pd.Timestamp("2024-02-28"), pd.Timestamp("2024-03-28")]
    assert list(risk_on) == [False, True]
