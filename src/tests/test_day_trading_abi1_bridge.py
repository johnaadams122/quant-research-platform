"""Tests for src/mining/day_trading_abi1_bridge.py -- the logic this bridge
module exists for: converting a day-trading family's real
`intraday_backtest_engine.run_intraday_backtest` `trades_df` (max_hold_days=1
-- every trade opens AND closes within the same session, by the day-trading
families' own mechanical invariant) into the
`{"strategy_daily", "in_market", "trades_df"}` account-curve dict shape
`mining.abi1_driver.evaluate_family` consumes.

Every case below is hand-computed against a small synthetic `trades_df`.
"""
import os
import sys

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.day_trading_abi1_bridge import (build_day_trading_curve,
                                             build_pooled_day_trading_curve)

_ET = "America/New_York"


def _trades_df(rows):
    """Build a trades_df shaped like run_intraday_backtest's own output --
    only the columns this bridge actually reads (entry_datetime,
    exit_datetime, net_ret) need real values; the rest are filled with
    innocuous placeholders so a caller inspecting the full shape still sees
    every column run_intraday_backtest documents."""
    if not rows:
        return pd.DataFrame(columns=[
            "entry_datetime", "exit_datetime", "entry_px", "exit_px",
            "held_days", "reason", "gross_ret", "net_ret"])
    out = []
    for entry_str, exit_str, net_ret in rows:
        out.append({
            "entry_datetime": pd.Timestamp(entry_str, tz=_ET),
            "exit_datetime": pd.Timestamp(exit_str, tz=_ET),
            "entry_px": 100.0, "exit_px": 100.0 * (1.0 + net_ret),
            "held_days": 1, "reason": "signal_exit",
            "gross_ret": net_ret, "net_ret": net_ret,
        })
    return pd.DataFrame(out)


def test_one_trade_closed_day_carries_its_net_ret():
    """Hand-computed case 1: a day with exactly one closed trade -- that
    day's strategy_daily must equal the trade's own net_ret exactly, and
    in_market must be True that day."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    trades = _trades_df([
        ("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02),
    ])
    curve = build_day_trading_curve(trades, calendar)

    assert curve["strategy_daily"].loc[pd.Timestamp("2024-01-02")] == pytest.approx(0.02)
    assert curve["in_market"].loc[pd.Timestamp("2024-01-02")] is True or \
        bool(curve["in_market"].loc[pd.Timestamp("2024-01-02")]) is True


def test_day_with_zero_trades_is_flat_and_out_of_market():
    """Hand-computed case 2: a calendar day with NO closed trade -- 0.0
    strategy_daily, False in_market."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    trades = _trades_df([
        ("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02),
    ])
    curve = build_day_trading_curve(trades, calendar)

    assert curve["strategy_daily"].loc[pd.Timestamp("2024-01-03")] == 0.0
    assert bool(curve["in_market"].loc[pd.Timestamp("2024-01-03")]) is False


def test_completely_empty_trades_df_is_all_flat_and_out_of_market():
    """No trades at all over the whole calendar -- every day 0.0/False."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    trades = _trades_df([])
    curve = build_day_trading_curve(trades, calendar)

    assert (curve["strategy_daily"] == 0.0).all()
    assert (~curve["in_market"]).all()
    assert len(curve["trades_df"]) == 0


def test_calendar_alignment_matches_a_daily_ohlc_style_benchmark_index():
    """Hand-computed case 3: the returned strategy_daily/in_market indices
    must be tz-naive, midnight-normalized daily Timestamps -- the SAME
    convention run_batch3_mining._normalize_ohlc (and therefore every real
    daily OHLC benchmark frame, e.g. SPY) uses -- so abi1_driver.evaluate_family's
    inner join against benchmark_daily lines up sessions correctly instead
    of silently producing an empty intersection because one side carries a
    timezone and the other does not."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    # Mimics run_batch3_mining._normalize_ohlc's own benchmark index shape
    # exactly: tz-naive, normalized to midnight.
    benchmark_index = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-03"])).normalize()
    trades = _trades_df([("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.01)])

    curve = build_day_trading_curve(trades, calendar)

    assert curve["strategy_daily"].index.tz is None
    assert curve["in_market"].index.tz is None
    assert curve["strategy_daily"].index.equals(benchmark_index)
    # A real inner join (what evaluate_family's log_excess step does under
    # the hood) must actually find both sessions, not silently produce an
    # empty intersection.
    joined = curve["strategy_daily"].index.intersection(benchmark_index)
    assert len(joined) == 2


def test_two_trades_closing_the_same_day_compound_multiplicatively():
    """A day-trading position can flatten and re-enter within one session
    (the day-trading single-position-at-a-time invariant does not forbid
    a second round trip after the first exits), so a real calendar day can
    carry MORE than one closed trade. Combined the same way
    intraday_backtest_engine.py's own module docstring documents for a
    same-day multi-trade day: product(1 + leg) - 1, never naive sum."""
    calendar = pd.DatetimeIndex(["2024-01-02"])
    trades = _trades_df([
        ("2024-01-02 09:31:00", "2024-01-02 11:00:00", 0.01),
        ("2024-01-02 13:00:00", "2024-01-02 15:55:00", -0.02),
    ])
    curve = build_day_trading_curve(trades, calendar)

    expected = (1.01) * (0.98) - 1.0
    assert curve["strategy_daily"].loc[pd.Timestamp("2024-01-02")] == pytest.approx(expected)
    assert bool(curve["in_market"].loc[pd.Timestamp("2024-01-02")]) is True


def test_entry_exit_day_mismatch_raises_instead_of_silently_misfiling():
    """Invariant self-test: this bridge assumes every trade closes on
    the SAME calendar day it opened (max_hold_days=1, the day-trading
    families' own mechanical invariant, enforced elsewhere by
    run_day_trading_mining._validate_day_trading_candidate). A trade that
    violates it (e.g. a bug upstream that let a position survive past
    midnight) must be a loud failure here, not silently filed under
    whichever day happens to match, which would misstate which day the
    account was really exposed on."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    trades = _trades_df([("2024-01-02 15:59:00", "2024-01-03 09:31:00", 0.02)])

    with pytest.raises(ValueError, match="entry/exit calendar day"):
        build_day_trading_curve(trades, calendar)


def test_trade_day_outside_supplied_calendar_raises():
    """A trade dated outside the caller's own calendar_index is a real
    caller bug (bars and trades_df out of sync) -- fail loud rather than
    silently drop the trade's contribution to strategy_daily."""
    calendar = pd.DatetimeIndex(["2024-01-02"])
    trades = _trades_df([("2024-01-03 09:31:00", "2024-01-03 15:55:00", 0.02)])

    with pytest.raises(ValueError, match="not in the supplied calendar_index"):
        build_day_trading_curve(trades, calendar)


def test_trades_df_gains_entry_date_exit_date_columns_required_downstream():
    """`mining.account_relative.netted_tax_factors` (reached through
    `abi1_driver.evaluate_family`) requires `trades_df` to carry
    `entry_date`/`exit_date` columns by name (without them, a non-empty
    trades_df makes `netted_tax_factors` raise a ValueError: "trades_df is
    missing 'entry_date'"). `backtest_engine.run_backtest`'s own (daily)
    trades_df already carries them under exactly these names for the same
    reason; this bridge must add the equivalent, tz-naive/midnight-normalized
    (matching `calendar_index`'s own convention), derived from
    `entry_datetime`/`exit_datetime`'s own calendar day -- which for
    day-trading families are always equal (max_hold_days=1, enforced above)."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    trades = _trades_df([("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02)])
    curve = build_day_trading_curve(trades, calendar)

    out = curve["trades_df"]
    assert list(out["entry_date"]) == [pd.Timestamp("2024-01-02")]
    assert list(out["exit_date"]) == [pd.Timestamp("2024-01-02")]
    assert out["entry_date"].dtype == "datetime64[ns]"
    assert out["exit_date"].dtype == "datetime64[ns]"
    # Every original column must survive untouched.
    assert out["net_ret"].iloc[0] == pytest.approx(0.02)
    assert out["entry_px"].iloc[0] == pytest.approx(100.0)


def test_trades_df_entry_date_exit_date_columns_on_empty_input():
    """An empty trades_df must still come back with entry_date/exit_date
    columns of the right dtype -- a downstream len(trades_df) == 0 check in
    netted_tax_factors returns early without ever reading the columns, but
    this bridge's own contract (every trades_df it returns carries these
    two columns) should not depend on that early return to stay safe."""
    calendar = pd.DatetimeIndex(["2024-01-02"])
    trades = _trades_df([])
    curve = build_day_trading_curve(trades, calendar)
    out = curve["trades_df"]
    assert "entry_date" in out.columns and "exit_date" in out.columns
    assert len(out) == 0


def test_original_trades_df_object_is_not_mutated_in_place():
    """build_day_trading_curve must not mutate the caller's own trades_df
    object -- it returns an augmented COPY, so a caller still holding the
    original reference (e.g. for its own n_trades count/reporting) never
    sees columns appear on it unexpectedly."""
    calendar = pd.DatetimeIndex(["2024-01-02"])
    trades = _trades_df([("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02)])
    build_day_trading_curve(trades, calendar)
    assert "entry_date" not in trades.columns


# ---------------------------------------------------------------------------
# build_pooled_day_trading_curve -- a multi-ticker day-trading family (e.g.
# fvg_reversal_cascade_continuation_daytrade, pooled across QQQ/SPY/IWM)
# combined via the SAME equal-weight-mean / OR-of-in_market / pool_trades
# convention mining.abi1_driver.build_pooled_curve already uses for daily-bar
# `pooled_single_asset` families -- ported here, not re-derived, so a
# pooled day-trading family is scored under the identical
# rule as every other pooled family in this catalog.

def _curve_from(calendar, rows):
    """One member's own build_day_trading_curve output, built the same way
    a real bespoke driver would (one real intraday backtest per ticker)."""
    return build_day_trading_curve(_trades_df(rows), calendar)


def test_pooled_two_members_different_days_is_zero_filled_mean():
    """Hand-computed: QQQ trades day 1 only, SPY trades day 2 only. Neither
    member's own OTHER day is a trade, so each contributes 0.0 there --
    pooled strategy_daily on day 1 = mean(0.02, 0.0) = 0.01, on day 2 =
    mean(0.0, -0.01) = -0.005. in_market is the OR across members: True
    both days (some member traded each day)."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    qqq = _curve_from(calendar, [("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02)])
    spy = _curve_from(calendar, [("2024-01-03 09:31:00", "2024-01-03 15:55:00", -0.01)])

    pooled = build_pooled_day_trading_curve({"QQQ": qqq, "SPY": spy})

    assert pooled["strategy_daily"].loc[pd.Timestamp("2024-01-02")] == pytest.approx(0.01)
    assert pooled["strategy_daily"].loc[pd.Timestamp("2024-01-03")] == pytest.approx(-0.005)
    assert bool(pooled["in_market"].loc[pd.Timestamp("2024-01-02")]) is True
    assert bool(pooled["in_market"].loc[pd.Timestamp("2024-01-03")]) is True


def test_pooled_both_members_trade_the_same_day_averages_their_returns():
    """Hand-computed: QQQ and SPY both trade on the same day -- pooled
    strategy_daily = mean(0.02, 0.04) = 0.03, the equal-weight-sleeve
    convention (never a naive sum, which would overstate the account's
    real capital exposure across two simultaneously-held sleeves)."""
    calendar = pd.DatetimeIndex(["2024-01-02"])
    qqq = _curve_from(calendar, [("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02)])
    spy = _curve_from(calendar, [("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.04)])

    pooled = build_pooled_day_trading_curve({"QQQ": qqq, "SPY": spy})

    assert pooled["strategy_daily"].loc[pd.Timestamp("2024-01-02")] == pytest.approx(0.03)
    assert bool(pooled["in_market"].loc[pd.Timestamp("2024-01-02")]) is True


def test_pooled_day_no_member_trades_is_flat_and_out_of_market():
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    qqq = _curve_from(calendar, [("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02)])
    spy = _curve_from(calendar, [("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.01)])

    pooled = build_pooled_day_trading_curve({"QQQ": qqq, "SPY": spy})

    assert pooled["strategy_daily"].loc[pd.Timestamp("2024-01-03")] == 0.0
    assert bool(pooled["in_market"].loc[pd.Timestamp("2024-01-03")]) is False


def test_pooled_trades_df_is_concatenated_and_ticker_tagged():
    """trades_df must be every member's own real, undiluted trades,
    concatenated and tagged by ticker -- matching build_pooled_curve's own
    `pool_trades` convention exactly, so a ledger row built from this
    output reports a real, correct n_trades count and per-member
    attribution."""
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    qqq = _curve_from(calendar, [("2024-01-02 09:31:00", "2024-01-02 15:55:00", 0.02)])
    spy = _curve_from(calendar, [("2024-01-03 09:31:00", "2024-01-03 15:55:00", -0.01)])
    iwm = _curve_from(calendar, [])

    pooled = build_pooled_day_trading_curve({"QQQ": qqq, "SPY": spy, "IWM": iwm})

    out = pooled["trades_df"]
    assert len(out) == 2
    assert set(out["ticker"]) == {"QQQ", "SPY"}
    assert sorted(out["net_ret"].round(4).tolist()) == [-0.01, 0.02]


def test_pooled_curve_empty_dict_raises():
    with pytest.raises(ValueError, match="no pool members"):
        build_pooled_day_trading_curve({})


def test_pooled_curve_all_members_zero_trades_is_all_flat_with_empty_trades_df():
    calendar = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    qqq = _curve_from(calendar, [])
    spy = _curve_from(calendar, [])

    pooled = build_pooled_day_trading_curve({"QQQ": qqq, "SPY": spy})

    assert (pooled["strategy_daily"] == 0.0).all()
    assert (~pooled["in_market"]).all()
    assert len(pooled["trades_df"]) == 0


def test_pooled_matches_the_daily_pool_on_a_different_calendar_pool():
    """Parity with the rule this bridge claims to port (`abi1_driver.build_pooled_curve`): feed the SAME per-member curves through
    both and require exact agreement of strategy_daily, in_market and trades_df.
    Deliberately hard input: members on DIFFERENT calendars (one starts later and
    runs longer), one member with two trades compounding on the same day, and
    one member with NO trades at all. `build_single_asset_curve` is stubbed to
    hand `build_pooled_curve` the already-built member curves, so what is
    compared is exactly the combination step, not the per-member curve builder.
    (The pooled tests above check hand-computed values; this one checks
    agreement with `build_pooled_curve` itself.)"""
    from mining import abi1_driver

    cal_a = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-01-12"))
    cal_b = pd.DatetimeIndex(pd.bdate_range("2024-01-05", "2024-01-17"))
    curves = {
        "A": build_day_trading_curve(_trades_df([
            ("2024-01-03 10:00:00", "2024-01-03 11:00:00", 0.02),
            ("2024-01-03 13:00:00", "2024-01-03 14:00:00", -0.01),
            ("2024-01-08 10:00:00", "2024-01-08 12:00:00", 0.03),
        ]), cal_a),
        "B": build_day_trading_curve(_trades_df([
            ("2024-01-05 10:00:00", "2024-01-05 11:00:00", 0.05),
            ("2024-01-16 10:00:00", "2024-01-16 11:00:00", -0.02),
        ]), cal_b),
        "C": build_day_trading_curve(_trades_df([]), cal_a),
    }
    pooled = build_pooled_day_trading_curve(curves)

    original = abi1_driver.build_single_asset_curve
    abi1_driver.build_single_asset_curve = lambda frame, fn, cost_bps: curves[frame]
    try:
        ref = abi1_driver.build_pooled_curve(
            {k: None for k in curves}, {k: k for k in curves},
            cost_bps_by_symbol={k: 0 for k in curves})
    finally:
        abi1_driver.build_single_asset_curve = original

    pd.testing.assert_series_equal(pooled["strategy_daily"], ref["strategy_daily"],
                                   check_exact=True, check_names=False)
    pd.testing.assert_series_equal(pooled["in_market"], ref["in_market"],
                                   check_exact=True, check_names=False)
    pd.testing.assert_frame_equal(pooled["trades_df"], ref["trades_df"],
                                  check_exact=True, check_names=False)
    # hand check: day 2024-01-03 -- A compounds 1.02 * 0.99 - 1, averaged over 3 members
    assert pooled["strategy_daily"].loc[pd.Timestamp("2024-01-03")] == pytest.approx(
        (1.02 * 0.99 - 1.0) / 3.0)
