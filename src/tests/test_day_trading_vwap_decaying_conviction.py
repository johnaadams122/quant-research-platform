"""Tests for src/mining/day_trading_vwap_decaying_conviction.py -- the
vwap_cross_decaying_conviction_exit_daytrade family (day-trading track).

Mirrors test_mining_intraday_backtest_engine.py's conventions (the
INTRADAY engine's own test template, not the daily-bar engine's): small
local helper functions building a synthetic tz-aware ("America/New_York")
minute-bar DataFrame, driven through the real `run_intraday_backtest`,
with exact/close-tolerance value assertions rather than mocking the
engine.

Every bar's own "VWAP" column is held at a constant 100.0 throughout each
test's frame. Since the session-VWAP this family accumulates is a
Volume-weighted average of that same per-bar VWAP column (mirrored from
the sibling family, see the module docstring), a CONSTANT per-bar VWAP
of 100.0 makes the running session VWAP exactly 100.0 at every bar
regardless of Volume -- a deliberate simplification that keeps each
test's entry-confirmation geometry (Close vs. session VWAP) fully
explicit and exact, rather than depending on a computed weighted
average.

Three scenarios are covered below (test_..._a/_b/_c); two more
(test_decayed_distance_*) directly unit-test the decay MATH as a pure
function, independent of the signal_fn/backtest-engine plumbing, which
shows the mechanism itself narrows monotonically toward a floor. A fourth
scenario (test_scenario_d_...) exercises the STOP-LOSS exit path, which
(a)/(b)/(c) never reach: they only hit the target or the force-flat
backstop.
"""
import os
import sys

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.intraday_backtest_engine import run_intraday_backtest  # noqa: E402
from mining.day_trading_vwap_decaying_conviction import (  # noqa: E402
    decayed_distance,
    make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _minute_index(date_str: str, start="09:30", end="15:55") -> pd.DatetimeIndex:
    day = pd.Timestamp(date_str, tz="America/New_York")
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    start_ts = day.replace(hour=sh, minute=sm)
    end_ts = day.replace(hour=eh, minute=em)
    return pd.date_range(start_ts, end_ts, freq="1min")


def _flat_frame(date_str: str, start="09:30", end="15:55", price=100.0, vwap=100.0):
    """A whole session of flat OHLC bars at `price`, VWAP held at `vwap`
    on every bar, Volume=100 throughout.
    """
    idx = _minute_index(date_str, start=start, end=end)
    n = len(idx)
    return pd.DataFrame(
        {
            "Open": [price] * n,
            "High": [price] * n,
            "Low": [price] * n,
            "Close": [price] * n,
            "Volume": [100.0] * n,
            "VWAP": [vwap] * n,
        },
        index=idx,
    )


def _set_bar(df, ts_str, date_str, **cols):
    ts = pd.Timestamp(f"{date_str} {ts_str}", tz="America/New_York")
    for col, val in cols.items():
        df.loc[ts, col] = float(val)
    return df


# ---------------------------------------------------------------------
# Pure decay-math tests
# ---------------------------------------------------------------------

def test_decayed_distance_equals_base_at_zero_elapsed():
    assert decayed_distance(0.01, 0.25, 30.0, 0.0) == pytest.approx(0.01)
    assert decayed_distance(0.005, 0.25, 30.0, 0.0) == pytest.approx(0.005)


def test_decayed_distance_reaches_half_life_midpoint():
    # At elapsed == half_life_minutes, decay == 0.5 -> distance is the
    # midpoint between floor_fraction*base and base:
    # 0.25 + 0.75*0.5 = 0.625 of base.
    got = decayed_distance(0.01, 0.25, 30.0, 30.0)
    assert got == pytest.approx(0.01 * 0.625, rel=1e-9)


def test_decayed_distance_approaches_but_never_reaches_floor():
    # 1000 minutes at a 30-minute half-life is ~33.3 half-lives (decay
    # ~= 1.4e-10) -- extreme enough to be visually indistinguishable from
    # the floor, but not so extreme that 0.5**(elapsed/half_life)
    # underflows to exactly 0.0 in IEEE double (which happens well before
    # 100,000 minutes at this half-life, and would make `got == floor`
    # exactly -- a floating-point artifact, not evidence the formula ever
    # truly reaches its floor).
    got = decayed_distance(0.01, 0.25, 30.0, 1000.0)
    floor = 0.01 * 0.25
    assert got > floor
    assert got == pytest.approx(floor, abs=1e-9)


def test_decayed_distance_is_monotonically_non_increasing():
    elapsed_points = [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 1000.0]
    values = [decayed_distance(0.01, 0.25, 30.0, e) for e in elapsed_points]
    for earlier, later in zip(values, values[1:]):
        assert later <= earlier


def test_decayed_distance_negative_elapsed_clamped_to_zero():
    # A defensive clamp -- elapsed_minutes should never legitimately be
    # negative in production, but a caller passing a small negative value
    # (e.g. floating-point noise) must not get a distance ABOVE base_pct.
    assert decayed_distance(0.01, 0.25, 30.0, -1.0) == pytest.approx(0.01)


# ---------------------------------------------------------------------
# (a) VWAP cross, no adverse move -> exit near the (near-)base target
# ---------------------------------------------------------------------

def test_scenario_a_fast_favorable_move_exits_close_to_base_target():
    date_str = "2026-03-02"  # Monday
    df = _flat_frame(date_str, start="09:30", end="09:40", price=99.0, vwap=100.0)
    # Confirmation streak (vwap_confirmation_bars=2 default): two
    # consecutive closes strictly above the constant session VWAP=100.
    _set_bar(df, "09:31", date_str, Open=100.4, High=100.6, Low=100.3, Close=100.5)
    _set_bar(df, "09:32", date_str, Open=100.5, High=100.7, Low=100.4, Close=100.6)
    # Fill bar: engine buys at THIS bar's Open (next-bar-open execution
    # after the 09:32 entry signal) -> entry_price = 101.0, elapsed=0 here.
    _set_bar(df, "09:33", date_str, Open=101.0, High=101.05, Low=100.95, Close=101.0)
    # One minute later: base_target_pct=0.01 has barely decayed (elapsed=1
    # minute, half_life=30) -> decayed target ~= 101 * 1.00983 ~= 102.09.
    # High crosses it; same-bar-close execution fills near that level.
    _set_bar(df, "09:34", date_str, Open=102.0, High=102.20, Low=101.95, Close=102.10)

    signal_fn = make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn()
    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0, tax_haircut_rate=0.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(101.0)
    assert row["reason"] == "signal_exit"
    # Realized gain is CLOSE TO the undecayed base_target_pct (0.01) --
    # not equal (same-bar-close execution + one minute of decay both move
    # it slightly), and decisively closer to the base target than to the
    # floor target (0.0025).
    assert row["gross_ret"] == pytest.approx(0.01, abs=0.003)
    assert row["gross_ret"] > 0.007


# ---------------------------------------------------------------------
# (b) VWAP cross, price stalls for a long time -> narrower TP/SL near
#     the floor by then than the entry distances were.
# ---------------------------------------------------------------------

def test_scenario_b_stalled_trade_exits_at_a_much_narrower_target_than_base():
    date_str = "2026-03-03"  # Tuesday
    # Default every bar in the whole session to the post-fill flat price
    # (101.0) FIRST, so any bar left un-overridden below (before the
    # confirm/fill bars, and after the trigger bar) is a safe, inert
    # price rather than an unrelated cliff back to some other default --
    # only the two confirm bars below are deliberately below VWAP=100
    # (99.0), everything else sits at 101.0. The frame ends one minute
    # after the trigger bar (13:35), deliberately too short a tail for a
    # FRESH vwap_confirmation_bars=2 streak to build and open a second,
    # unintended trade after the first one exits (Close=101.0 is still
    # above the constant VWAP=100 baseline, so a longer flat tail here
    # would legitimately re-arm a second entry -- a real behavior of the
    # signal_fn, just not one this scenario is testing).
    df = _flat_frame(date_str, start="09:30", end="13:35", price=101.0, vwap=100.0)
    _set_bar(df, "09:30", date_str, Open=99.0, High=99.0, Low=99.0, Close=99.0)
    _set_bar(df, "09:31", date_str, Open=100.4, High=100.6, Low=100.3, Close=100.5)
    _set_bar(df, "09:32", date_str, Open=100.5, High=100.7, Low=100.4, Close=100.6)
    _set_bar(df, "09:33", date_str, Open=101.0, High=101.0, Low=101.0, Close=101.0)

    # Every bar from 09:34 through 13:40 defaults to dead flat at
    # entry_price (101.0) except the single tick bar below at 13:34 --
    # 240 minutes elapsed by then. At half_life=30min that is 8
    # half-lives: decay = 0.5**8 ~= 0.0039, so the target distance has
    # shrunk to ~= base_target_pct*(0.25 + 0.75*0.0039) ~= 0.2529% of
    # entry price -- visibly narrower than the ~1.0% (minus one minute of
    # decay) distance scenario (a) exited at.

    # 13:34: a small tick up to 101.30 (+0.297%) -- clearly BELOW what the
    # undecayed base target (102.01, +1.0%) would require, but above the
    # now-decayed (near-floor) target (~101.2555).
    _set_bar(df, "13:34", date_str, Open=101.05, High=101.30, Low=101.0, Close=101.28)

    signal_fn = make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn()
    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0, tax_haircut_rate=0.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(101.0)
    assert row["reason"] == "signal_exit"
    # A move this small (~0.3%) would NEVER have hit the undecayed base
    # target (1.0%) -- proves the exit fired on the SHRUNKEN target, not
    # the entry-time one.
    assert row["gross_ret"] < 0.005
    assert row["gross_ret"] > 0.0


# ---------------------------------------------------------------------
# (c) force-flat at 15:55 ET still fires if TP/SL never triggers.
# ---------------------------------------------------------------------

def test_scenario_c_force_flat_fires_at_1555_when_tp_sl_never_hit():
    date_str = "2026-03-04"  # Wednesday
    df = _flat_frame(date_str, start="09:30", end="15:55", price=99.0, vwap=100.0)
    _set_bar(df, "09:31", date_str, Open=100.4, High=100.6, Low=100.3, Close=100.5)
    _set_bar(df, "09:32", date_str, Open=100.5, High=100.7, Low=100.4, Close=100.6)

    fill_idx = pd.date_range(
        pd.Timestamp(f"{date_str} 09:33", tz="America/New_York"),
        pd.Timestamp(f"{date_str} 15:55", tz="America/New_York"),
        freq="1min",
    )
    for ts in fill_idx:
        # Dead flat at entry_price (101.0) the whole rest of the session
        # -- even the FLOOR-level target/stop distances (0.25% / 0.125%
        # of entry price) are never touched.
        df.loc[ts, ["Open", "High", "Low", "Close"]] = 101.0
        df.loc[ts, "Volume"] = 100.0
        df.loc[ts, "VWAP"] = 100.0

    signal_fn = make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn()
    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0, tax_haircut_rate=0.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(101.0)
    assert row["exit_px"] == pytest.approx(101.0)
    assert row["gross_ret"] == pytest.approx(0.0, abs=1e-9)
    assert row["reason"] == "signal_exit"
    exit_ts = row["exit_datetime"]
    assert (exit_ts.hour, exit_ts.minute) == (15, 55)


# ---------------------------------------------------------------------
# (d) stop-loss fires on an adverse move -- (a)/(b)/(c) never
#     exercise the `low <= stop_price` branch.
# ---------------------------------------------------------------------

def test_scenario_d_stop_loss_fires_on_adverse_move():
    date_str = "2026-03-05"  # Thursday
    df = _flat_frame(date_str, start="09:30", end="09:40", price=99.0, vwap=100.0)
    _set_bar(df, "09:31", date_str, Open=100.4, High=100.6, Low=100.3, Close=100.5)
    _set_bar(df, "09:32", date_str, Open=100.5, High=100.7, Low=100.4, Close=100.6)
    # Fill bar, same as scenario (a): entry_price = 101.0.
    _set_bar(df, "09:33", date_str, Open=101.0, High=101.05, Low=100.95, Close=101.0)
    # One minute later: base_stop_pct=0.005 has barely decayed (elapsed=1
    # minute, half_life=30) -> decayed stop ~= 101 * (1 - 0.00491) ~= 100.50.
    # Low crosses it while High stays well below the (undecayed) target
    # (~102.09) -- an unambiguous stop-loss exit, not a target/tie-break case.
    _set_bar(df, "09:34", date_str, Open=100.60, High=100.65, Low=100.40, Close=100.45)

    signal_fn = make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn()
    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0, tax_haircut_rate=0.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(101.0)
    assert row["reason"] == "signal_exit"
    # Realized loss is CLOSE TO the undecayed base_stop_pct (-0.005), and
    # decisively past the floor stop (-0.00125) -- proves this exit fired
    # on the stop-loss branch specifically, not the target or force-flat.
    assert row["gross_ret"] == pytest.approx(-0.005, abs=0.0015)
    assert row["gross_ret"] < -0.003
