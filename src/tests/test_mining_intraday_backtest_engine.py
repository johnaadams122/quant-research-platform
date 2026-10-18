"""Tests for src/mining/intraday_backtest_engine.py, the intraday/swing
strategy-mining backtest engine.

Mirrors src/tests/test_mining_backtest_engine.py's conventions: small
local helper functions to build synthetic frames, inline closures for
signal logic per test, exact-value assertions.

The two "telescoping" invariant tests (test_gross_legs_telescope_to_...
and test_net_legs_compound_to_trades_own_net_ret) are the most important
tests in this file -- they prove the mark-to-market daily-return
decomposition is mathematically honest, not merely plausible-looking.
See the module docstring in intraday_backtest_engine.py for the exact
algorithm these encode, and the hand-derivation in the comments below
for the worked numeric example these tests assert against.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.intraday_backtest_engine import SignalResult, run_intraday_backtest


def _minute_frame(day_specs):
    """Build a synthetic RTH-shaped, tz-aware ("America/New_York") minute-
    bar DataFrame matching the shape intraday_data.fetch_intraday_bars
    actually produces (Open/High/Low/Close/Volume columns, tz-aware
    DatetimeIndex).

    `day_specs` is a list of (date_str, bars) pairs, in chronological
    order. Each `bars` entry is a list of either:
      - (time_str, close)  -- Open/High/Low default to `close`, Volume
        defaults to 100 (shorthand for tests that only care about Close)
      - (time_str, open, high, low, close, volume) -- full control
    """
    rows = []
    idx = []
    for date_str, bars in day_specs:
        for bar in bars:
            if len(bar) == 2:
                time_str, close = bar
                o = h = l = c = float(close)
                v = 100
            else:
                time_str, o, h, l, c, v = bar
            ts = pd.Timestamp(f"{date_str} {time_str}", tz="America/New_York")
            idx.append(ts)
            rows.append({"Open": float(o), "High": float(h), "Low": float(l),
                         "Close": float(c), "Volume": v})
    index = pd.DatetimeIndex(idx, name="timestamp")
    return pd.DataFrame(rows, index=index,
                         columns=["Open", "High", "Low", "Close", "Volume"])


# ---------------------------------------------------------------------
# 1. Telescoping / reconstruction invariants -- the most important tests.
# ---------------------------------------------------------------------
#
# Hand-derived worked example (3-day long trade). The entry day gets its
# own real leg (see the module docstring's mark-to-market design section):
#
#   Day D0 (entry day): entry_px = 102 (next-bar-open execution);
#     D0's OWN last bar close = 105 (D0 is held from entry through its
#     own close, so it gets a real leg too, not a deferred-to-D1 zero).
#   Day D1 (interior day, held all the way through): last bar close = 108
#   Day D2 (exit day): exit_px = 110 (same-bar-close execution)
#
#   leg_0 (assigned to D0, the ENTRY day's own move) = 105/102 - 1
#   leg_1 (assigned to D1) = 108/105 - 1
#   leg_2 (assigned to D2, the FINAL leg) = 110/108 - 1
#
#   Telescoping check (gross, cost_bps=0 so net==gross):
#     (1 + leg_0) * (1 + leg_1) * (1 + leg_2)
#       = (105/102) * (108/105) * (110/108) = 110/102
#       = exit_px / entry_px   -- EXACT, by construction (the 105 and
#     108 terms cancel algebraically). The product runs over EVERY
#     touched day (0..N), the entry day included.

def _three_day_trade_frame():
    return _minute_frame([
        ("2026-03-02", [  # Monday -- entry day (D0)
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 104, 100, 103, 100),  # entry executes at this bar's OPEN=102...
            ("09:32", 104, 106, 104, 105, 100),  # last bar of D0 -> D0's OWN mark = Close = 105
        ]),
        ("2026-03-03", [  # Tuesday -- interior day (D1)
            ("09:30", 106, 106, 106, 106, 100),
            ("09:31", 107, 109, 107, 108, 100),  # last bar of D1 -> mark(D1) = Close = 108
        ]),
        ("2026-03-04", [  # Wednesday -- exit day (D2)
            ("09:30", 109, 111, 109, 110, 100),  # exit fires here -> exit_px = Close = 110
            ("09:31", 111, 112, 111, 112, 100),
        ]),
    ])


def test_gross_legs_telescope_to_total_gross_return_long():
    # Set the entry bar's Open to exactly 102 to match the hand-derivation.
    df = _three_day_trade_frame()
    df.iloc[1, df.columns.get_loc("Open")] = 102.0

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 5))

    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0, tax_haircut_rate=0.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(102.0)
    assert row["exit_px"] == pytest.approx(110.0)
    assert row["gross_ret"] == pytest.approx(110.0 / 102.0 - 1.0)
    assert row["net_ret"] == pytest.approx(row["gross_ret"])  # zero cost/tax

    d0, d1, d2 = daily.index[0], daily.index[1], daily.index[2]
    assert len(daily) == 3
    # Entry day gets its OWN real leg: D0's last-bar close (105) vs
    # the actual entry price (102) -- NOT deferred to D1, NOT zero.
    assert daily.loc[d0] == pytest.approx(105.0 / 102.0 - 1.0)
    # Interior leg: mark(D1)/mark(D0) - 1 = 108/105 - 1
    assert daily.loc[d1] == pytest.approx(108.0 / 105.0 - 1.0)
    # Final (exit) leg, gross==net since cost_bps=0: exit_px/mark(D1) - 1
    assert daily.loc[d2] == pytest.approx(110.0 / 108.0 - 1.0)

    # THE core invariant: compounding the gross daily legs -- INCLUDING
    # the entry day's own leg -- reconstructs the trade's total gross
    # return EXACTLY (telescoping, no approximation).
    compounded = (1.0 + daily.loc[d0]) * (1.0 + daily.loc[d1]) * (1.0 + daily.loc[d2])
    assert compounded == pytest.approx(110.0 / 102.0, abs=1e-9)
    assert compounded == pytest.approx(1.0 + row["gross_ret"], abs=1e-9)


def test_net_legs_compound_to_trades_own_net_ret():
    # Same trade shape, but with real cost/tax so gross != net -- proves
    # cost/tax attribution lands ONLY on the final (exit) leg (not the
    # entry day's leg either), and that compounding ALL of a trade's
    # daily legs (entry + interior gross, final net) reconstructs the
    # trade's own reported net_ret exactly.
    df = _three_day_trade_frame()
    df.iloc[1, df.columns.get_loc("Open")] = 102.0

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 5))

    trades, daily = run_intraday_backtest(
        df, signal_fn, cost_bps=50.0, tax_haircut_rate=0.15)

    assert len(trades) == 1
    row = trades.iloc[0]
    gross_ret = 110.0 / 102.0 - 1.0
    net_before_tax = gross_ret - 50.0 / 1e4
    expected_net_ret = net_before_tax * (1.0 - 0.15)  # positive net -> haircut applies
    assert row["gross_ret"] == pytest.approx(gross_ret)
    assert row["net_ret"] == pytest.approx(expected_net_ret)

    d0, d1, d2 = daily.index[0], daily.index[1], daily.index[2]
    entry_leg_gross = 105.0 / 102.0 - 1.0
    interior_leg_gross = 108.0 / 105.0 - 1.0

    # Cost/tax must NOT touch the entry-day or interior legs -- both stay
    # pure gross; only the final (exit-day) leg absorbs the adjustment.
    assert daily.loc[d0] == pytest.approx(entry_leg_gross)
    assert daily.loc[d1] == pytest.approx(interior_leg_gross)

    # THE second core invariant: compounding the FULL daily contribution
    # (entry-day gross leg * interior gross leg * final net-adjusted leg)
    # reconstructs the trade's own net_ret exactly -- proves the two
    # outputs (trades_df, daily_returns) are mutually consistent, not
    # independently computed.
    compounded = (1.0 + daily.loc[d0]) * (1.0 + daily.loc[d1]) * (1.0 + daily.loc[d2])
    assert compounded == pytest.approx(1.0 + expected_net_ret, abs=1e-9)
    assert compounded == pytest.approx(1.0 + row["net_ret"], abs=1e-9)


def test_net_legs_compound_correctly_for_short_direction_too():
    # Short uses the ADDITIVE, fixed-basis convention, matching
    # backtest_engine.py's own linear gross_ret EXACTLY (zero deviation;
    # see test_short_legs_sum_to_trades_own_linear_gross_ret's comment for
    # the alternatives and the boundedness argument).
    # The final-leg net-adjustment for short SUMS (not compounds/
    # multiplies) -- matching that additive convention. It only ever
    # references net_ret and the prior legs' SUM, so the reconciliation
    # invariant (all legs sum to net_ret) must hold regardless of what
    # values the non-final (entry+interior) legs took.
    df = _three_day_trade_frame()
    df.iloc[1, df.columns.get_loc("Open")] = 102.0

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 5))

    trades, daily = run_intraday_backtest(
        df, signal_fn, cost_bps=25.0, tax_haircut_rate=0.10, direction="short")

    row = trades.iloc[0]
    long_ret = 110.0 / 102.0 - 1.0
    assert row["gross_ret"] == pytest.approx(-long_ret)  # linear, matching backtest_engine.py exactly

    d0, d1, d2 = daily.index[0], daily.index[1], daily.index[2]
    entry_px, mark_d0, mark_d1 = 102.0, 105.0, 108.0
    # ADDITIVE, fixed-basis leg: every leg divides by the SAME entry_px,
    # not the previous day's mark and not a reciprocal ratio (the
    # reciprocal form produces economically impossible >100% short gains,
    # see the boundedness test below).
    assert daily.loc[d0] == pytest.approx((entry_px - mark_d0) / entry_px)
    assert daily.loc[d1] == pytest.approx((mark_d0 - mark_d1) / entry_px)

    # THE reconciliation invariant for short: legs SUM (not compound) to
    # net_ret exactly.
    summed = daily.loc[d0] + daily.loc[d1] + daily.loc[d2]
    assert summed == pytest.approx(row["net_ret"], abs=1e-9)


def test_short_legs_sum_to_trades_own_linear_gross_ret():
    # Guards the short-leg design: additive, fixed-basis daily legs that SUM
    # (not compound) to the trade's own linear gross_ret, with no gap. Other
    # designs fail:
    #   - Negated long legs combined by product do not telescope at all.
    #     entry=100, interior mark=105, exit=110 -> negated legs compound to
    #     -0.095238, matching neither -0.10 nor anything meaningful.
    #   - A reciprocal ratio (entry_px/exit_px - 1) telescopes but is a
    #     different number from the linear gross_ret, and is economically
    #     wrong for a short: its maximum gain on fixed notional is exactly
    #     100% (the asset falls to zero), yet entry_px/exit_px - 1 is
    #     unbounded as exit_px -> 0. entry=100, exit=50 (price halved) is a
    #     +50% gain; the reciprocal formula gives +100%.
    # The additive design follows backtest_engine.py's own short convention;
    # test_short_gross_ret_is_bounded_at_100_percent_as_exit_approaches_zero
    # guards the boundedness. Mixing an additive short leg with a
    # multiplicative long leg in same-day aggregation is structurally
    # unreachable because direction is a whole-call parameter (see the module
    # docstring's section on the calling-contract invariant). The full leg
    # series, including the final exit-day leg, therefore needs no separate
    # reconciliation target to match gross_ret when cost/tax are zero: the
    # additive telescoping sum IS gross_ret, term for term.
    df = _three_day_trade_frame()
    df.iloc[1, df.columns.get_loc("Open")] = 102.0

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 5))

    trades, daily = run_intraday_backtest(
        df, signal_fn, cost_bps=0.0, tax_haircut_rate=0.0, direction="short")

    entry_px, mark_d0, mark_d1, exit_px = 102.0, 105.0, 108.0, 110.0
    d0, d1, d2 = daily.index[0], daily.index[1], daily.index[2]

    assert daily.loc[d0] == pytest.approx((entry_px - mark_d0) / entry_px)
    assert daily.loc[d1] == pytest.approx((mark_d0 - mark_d1) / entry_px)
    assert daily.loc[d2] == pytest.approx((mark_d1 - exit_px) / entry_px)

    row = trades.iloc[0]
    expected_gross = 1.0 - exit_px / entry_px
    assert row["gross_ret"] == pytest.approx(expected_gross)

    # THE telescoping proof: ALL THREE legs SUM exactly to trades_df's
    # own linear gross_ret -- no gap, no separate "reciprocal" target
    # needed.
    summed = daily.loc[d0] + daily.loc[d1] + daily.loc[d2]
    assert summed == pytest.approx(expected_gross, abs=1e-9)
    assert summed == pytest.approx(row["gross_ret"], abs=1e-9)


def test_short_gross_ret_is_bounded_at_100_percent_as_exit_approaches_zero():
    # Guards the hard economic bound on a short's gain. A short's real,
    # physical maximum possible gain on fixed notional is exactly 100%:
    # the shorted asset falls to zero, you owe nothing to buy it back, you
    # keep the full entry proceeds. Never more than that, by definition. A
    # reciprocal-consistent convention (entry_px/exit_px - 1) would violate
    # this outright (entry=100, exit=50 gives +100% instead of the real
    # +50%); the linear convention (1 - exit_px/entry_px) respects it
    # exactly -- as exit_px -> 0, gross_ret -> 1.0 (100%), never more.
    df = _minute_frame([
        ("2026-03-02", [
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 100, 100, 100, 100),  # entry open = 100
            ("09:32", 1, 1, 1, 1, 100),           # exit close = 1 (price near-zero)
        ]),
    ])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, _ = run_intraday_backtest(df, signal_fn, cost_bps=0.0, direction="short")

    assert len(trades) == 1
    gross_ret = trades.iloc[0]["gross_ret"]
    assert gross_ret == pytest.approx(1.0 - 1.0 / 100.0)  # 0.99, matches the linear formula exactly
    assert gross_ret < 1.0  # THE hard economic bound: never >= 100% on fixed notional
    assert gross_ret > 0.9  # sanity: genuinely close to the bound, not a trivially small number


def test_short_same_day_round_trip_matches_linear_gross_ret():
    # A same-day short round trip's single leg is ALWAYS the "final" leg
    # (entry_day_idx == exit_day_idx), unconditionally overwritten by the
    # net-reconciliation formula regardless of what its pre-override
    # formula produced -- so this degenerate case is unaffected by which
    # per-day convention is used. trades_df's own gross_ret is confirmed
    # directly here to be the linear convention (1 - exit/entry),
    # matching backtest_engine.py exactly.
    df = _minute_frame([
        ("2026-03-02", [
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 100, 100, 100, 100),  # entry open = 100
            ("09:32", 101, 102, 101, 102, 100),  # exit close = 102 (price rose -> short loses)
        ]),
    ])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 2))

    trades, daily = run_intraday_backtest(
        df, signal_fn, cost_bps=10.0, tax_haircut_rate=0.0, direction="short")

    assert len(trades) == 1
    row = trades.iloc[0]
    expected_gross = 1.0 - 102.0 / 100.0  # linear short convention, matching backtest_engine.py
    expected_net = expected_gross - 10.0 / 1e4  # negative net -> no haircut branch taken
    assert row["gross_ret"] == pytest.approx(expected_gross)
    assert row["net_ret"] == pytest.approx(expected_net)

    assert len(daily) == 1
    assert daily.iloc[0] == pytest.approx(expected_net)


def test_multiple_same_day_short_round_trips_compound_not_sum():
    # Same-day multi-TRADE aggregation (already proven for long by
    # test_multiple_same_day_round_trips_compound_not_sum) is a SEPARATE
    # layer from the per-trade leg formula above: once a trade closes,
    # its OWN leg is just a plain percentage number (whether it got there
    # via short's additive-fixed-basis formula or long's multiplicative-
    # ratio one), and stacking MULTIPLE such closed-trade percentages on
    # the same calendar day is a compounding-on-shared-capital operation
    # regardless of direction -- this exercises that REAL aggregation
    # code path with two short trades specifically.
    #
    # Trade A (short): entry=100, exit=95 -> gross = 1-95/100 = 0.05
    # (price fell 5%, short gains 5%). Trade B (short, same day, right
    # after A closes): entry=200, exit=190 -> gross = 1-190/200 = 0.05
    # (also a 5% drop) -- chosen so the two legs are identical, making the
    # correct-vs-naive distinction purely about compounding vs summing,
    # not about different magnitudes.
    df = _minute_frame([
        ("2026-03-02", [
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 100, 100, 100, 100),   # trade A entry open = 100
            ("09:32", 96, 96, 95, 95, 100),        # trade A exit close = 95
            ("09:33", 200, 200, 200, 200, 100),    # trade B entry open = 200
            ("09:34", 191, 191, 190, 190, 100),    # trade B exit close = 190
        ]),
    ])

    def signal_fn(frame, i):
        return SignalResult(entry=(i in (0, 2)), exit=(i in (2, 4)))

    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0, direction="short")

    assert len(trades) == 2
    leg_a = 1.0 - 95.0 / 100.0
    leg_b = 1.0 - 190.0 / 200.0
    assert trades.iloc[0]["gross_ret"] == pytest.approx(leg_a)
    assert trades.iloc[1]["gross_ret"] == pytest.approx(leg_b)

    assert len(daily) == 1
    correct_compounded = (1.0 + leg_a) * (1.0 + leg_b) - 1.0
    naive_sum = leg_a + leg_b
    assert correct_compounded != pytest.approx(naive_sum, abs=1e-6)  # sanity: they differ
    assert daily.iloc[0] == pytest.approx(correct_compounded)
    assert daily.iloc[0] != pytest.approx(naive_sum)


# ---------------------------------------------------------------------
# 2. Boundary-day inclusion/exclusion at the input frame's edge.
# ---------------------------------------------------------------------

def test_daily_returns_covers_every_day_in_df_including_first_and_last():
    df = _minute_frame([
        ("2026-03-02", [("09:30", 100), ("09:31", 100)]),
        ("2026-03-03", [("09:30", 100), ("09:31", 100)]),
        ("2026-03-04", [("09:30", 100), ("09:31", 100)]),
    ])

    def signal_fn(frame, i):
        return SignalResult()  # never trades

    trades, daily = run_intraday_backtest(df, signal_fn)

    assert len(trades) == 0
    assert len(daily) == 3
    expected_days = sorted({ts.date() for ts in df.index})
    assert list(daily.index) == expected_days
    assert (daily == 0.0).all()


# ---------------------------------------------------------------------
# 3. Multiple same-day round trips compound multiplicatively.
# ---------------------------------------------------------------------

def test_multiple_same_day_round_trips_compound_not_sum():
    # Trade A: entry open=100, exit close=102 -> gross +2%.
    # Trade B (same day, right after A closes): entry open=200, exit
    # close=206 -> gross +3%.
    # Naive summation would (WRONGLY) give the day 0.02 + 0.03 = 0.05.
    # Correct compounding gives 1.02 * 1.03 - 1 = 0.0506, which is what
    # this test asserts -- the two values are far enough apart (2% vs
    # nearly 2.06% i.e. absolute values 0.05 vs 0.0506) to discriminate.
    df = _minute_frame([
        ("2026-03-02", [
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 100, 100, 100, 100),   # trade A entry open = 100
            ("09:32", 101, 102, 101, 102, 100),   # trade A exit close = 102
            ("09:33", 200, 200, 200, 200, 100),   # trade B entry open = 200
            ("09:34", 201, 206, 201, 206, 100),   # trade B exit close = 206
        ]),
    ])

    def signal_fn(frame, i):
        return SignalResult(entry=(i in (0, 2)), exit=(i in (2, 4)))

    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0)

    assert len(trades) == 2
    assert trades.iloc[0]["gross_ret"] == pytest.approx(0.02)
    assert trades.iloc[1]["gross_ret"] == pytest.approx(0.03)
    assert trades.iloc[0]["held_days"] == 1
    assert trades.iloc[1]["held_days"] == 1

    assert len(daily) == 1
    naive_sum = 0.02 + 0.03
    correct_compounded = 1.02 * 1.03 - 1.0
    assert correct_compounded != pytest.approx(naive_sum, abs=1e-6)  # sanity: they really differ
    assert daily.iloc[0] == pytest.approx(correct_compounded)
    assert daily.iloc[0] != pytest.approx(naive_sum)


# ---------------------------------------------------------------------
# 4. A trade spanning a real US DST transition.
# ---------------------------------------------------------------------

def test_trade_spanning_spring_forward_dst_transition():
    # 2026-03-08 is a real US spring-forward Sunday (market closed); the
    # first trading day under EDT is Monday 2026-03-09. Entry Friday
    # 2026-03-06 (still EST), exit Monday 2026-03-09 (now EDT) -- proves
    # date-bucketing by tz-aware timestamp genuinely works across the
    # offset change, rather than assuming it. Friday gets a THIRD bar
    # after the entry bar so its own real mark-to-market leg
    # is distinct from the entry price -- otherwise a coincidentally-flat
    # fixture couldn't tell a real leg_0 apart from a leftover zero.
    df = _minute_frame([
        ("2026-03-06", [  # Friday, EST (UTC-5) -- entry day
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 101, 100, 101, 100),  # entry open = 100
            ("09:32", 102, 104, 102, 103, 100),  # LAST bar of Friday -> mark(D0) = 103
        ]),
        ("2026-03-09", [  # Monday, EDT (UTC-4) -- first session after DST, exit day
            ("09:30", 105, 106, 105, 106, 100),  # exit close = 106
        ]),
    ])

    # Confirm the test fixture itself really crosses the DST boundary
    # (UTC offset differs by exactly one hour: EST UTC-5 -> EDT UTC-4)
    # before trusting the result.
    assert df.index[3].utcoffset() - df.index[2].utcoffset() == pd.Timedelta(hours=1)

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 3))

    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(100.0)
    assert row["exit_px"] == pytest.approx(106.0)
    assert row["held_days"] == 2

    assert len(daily) == 2  # exactly two distinct trading days, no phantom third day
    d0, d1 = daily.index
    assert d0 == df.index[0].date()
    assert d1 == df.index[3].date()
    # Entry day (Friday) gets its OWN real leg: mark(D0)/entry_px - 1
    # = 103/100 - 1 -- NOT zero, NOT deferred onto Monday.
    assert daily.loc[d0] == pytest.approx(103.0 / 100.0 - 1.0)
    assert daily.loc[d1] == pytest.approx(106.0 / 103.0 - 1.0)
    # Telescoping still holds across the DST boundary: compounding both
    # legs reconstructs the trade's total gross return exactly.
    compounded = (1.0 + daily.loc[d0]) * (1.0 + daily.loc[d1])
    assert compounded == pytest.approx(106.0 / 100.0, abs=1e-9)


# ---------------------------------------------------------------------
# 5. A trade spanning a half trading session (early close).
# ---------------------------------------------------------------------

def test_interior_day_mark_uses_actual_last_bar_on_a_half_session():
    # Day D1 (interior) is a synthetic early-close day -- its bars stop at
    # 13:00 instead of a normal day's 15:59. The mark for D1 must be
    # whatever bar is ACTUALLY last present (13:00's close), proving the
    # aggregation never assumes/hardcodes a close time. D0 gets a third
    # bar after the entry bar so its own real leg is distinct from the
    # entry price.
    df = _minute_frame([
        ("2026-03-02", [  # D0, entry day, normal
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 101, 100, 101, 100),  # entry open = 100
            ("09:32", 102, 104, 102, 103, 100),  # LAST bar of D0 -> mark(D0) = 103
        ]),
        ("2026-03-03", [  # D1, interior day, half session (early close)
            ("09:30", 104, 104, 104, 104, 100),
            ("13:00", 105, 107, 105, 106, 100),  # LAST bar of the half day
        ]),
        ("2026-03-04", [  # D2, exit day, normal
            ("09:30", 107, 109, 107, 108, 100),  # exit close = 108
        ]),
    ])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=(i == 5))

    trades, daily = run_intraday_backtest(df, signal_fn, cost_bps=0.0)

    assert len(trades) == 1
    d0, d1, d2 = daily.index
    # Entry day's own real leg: mark(D0)/entry_px - 1 = 103/100 - 1.
    assert daily.loc[d0] == pytest.approx(103.0 / 100.0 - 1.0)
    # mark(D1) must be the 13:00 bar's close (106), not any other value.
    assert daily.loc[d1] == pytest.approx(106.0 / 103.0 - 1.0)
    assert daily.loc[d2] == pytest.approx(108.0 / 106.0 - 1.0)
    compounded = (1.0 + daily.loc[d0]) * (1.0 + daily.loc[d1]) * (1.0 + daily.loc[d2])
    assert compounded == pytest.approx(108.0 / 100.0, abs=1e-9)


# ---------------------------------------------------------------------
# 6. max_hold_days cap.
# ---------------------------------------------------------------------

def test_max_hold_days_force_closes_at_last_bar_of_cap_day():
    # 5 trading days, entry on day 0, never a signal exit; max_hold_days=3
    # must force a close at the LAST bar of day index 2 (the 3rd day,
    # 1-indexed) -- not day 0/1, and not day 3/4.
    df = _minute_frame([
        ("2026-03-02", [("09:30", 100), ("09:31", 100)]),   # D0 entry day
        ("2026-03-03", [("09:30", 100), ("09:31", 100)]),   # D1
        ("2026-03-04", [("09:30", 100), ("09:31", 105)]),   # D2 -- forced close here
        ("2026-03-05", [("09:30", 100), ("09:31", 100)]),   # D3 -- must NOT reach here
        ("2026-03-06", [("09:30", 100), ("09:31", 100)]),   # D4
    ])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=False)  # never a signal exit

    trades, daily = run_intraday_backtest(df, signal_fn, max_hold_days=3)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["reason"] == "time_stop"
    assert row["held_days"] == 3
    assert row["exit_datetime"] == df.index[5]  # last bar of D2 (index 5: bars 0-1=D0,2-3=D1,4-5=D2)
    assert row["exit_px"] == pytest.approx(105.0)


def test_signal_exit_wins_over_time_stop_on_same_bar():
    df = _minute_frame([
        ("2026-03-02", [("09:30", 100), ("09:31", 100)]),
        ("2026-03-03", [("09:30", 100), ("09:31", 100)]),
        ("2026-03-04", [("09:30", 100), ("09:31", 100)]),
    ])

    def signal_fn(frame, i):
        # exit fires on i==3, the last bar of D1 -- same bar the
        # max_hold_days=2 cap would also force a close on.
        return SignalResult(entry=(i == 0), exit=(i == 3))

    trades, _ = run_intraday_backtest(df, signal_fn, max_hold_days=2)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "signal_exit"


def test_max_hold_days_one_force_closes_same_bar_as_entry():
    # Guards the time-stop check on a freshly-opened position's own entry
    # bar. If the main loop's `if position is None:` branch skipped
    # straight to the next bar after opening a position, the position
    # would only be checked for hit_time_stop from the NEXT bar on. With
    # max_hold_days=1 and an entry landing on its own day's LAST bar, that
    # would let the position survive into the NEXT day (held_days=2)
    # before the cap fired, violating the documented "force-closed at the
    # LAST bar of its Nth trading day" contract. No other test combines
    # max_hold_days=1 with an entry on a day's final bar.
    #
    # D0 has exactly 2 bars: a pre-entry bar, then the entry bar itself,
    # which is therefore ALSO D0's last bar. D1 exists afterward purely
    # to prove the position does NOT carry into it. The signal function
    # never fires a real exit, so the ONLY thing that can close this
    # trade is the max_hold_days=1 cap -- and it must do so on D0's own
    # last bar (the entry bar), not D1's.
    df = _minute_frame([
        ("2026-03-02", [  # D0 -- entry bar is ALSO D0's last bar
            ("09:30", 99, 99, 99, 99, 100),
            ("09:31", 100, 101, 100, 101, 100),  # entry open=100, this bar's close=101
        ]),
        ("2026-03-03", [  # D1 -- must NEVER be reached with the position still open
            ("09:30", 102, 103, 102, 103, 100),
        ]),
    ])

    def signal_fn(frame, i):
        return SignalResult(entry=(i == 0), exit=False)  # never a real signal exit

    trades, daily = run_intraday_backtest(df, signal_fn, max_hold_days=1)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["reason"] == "time_stop"
    assert row["held_days"] == 1
    # THE sharpest possible proof: entry and exit are the SAME bar --
    # a genuine same-bar round trip, forced by the cap firing on the
    # entry bar itself rather than carrying over to D1.
    assert row["entry_datetime"] == df.index[1]
    assert row["exit_datetime"] == df.index[1]
    assert row["exit_px"] == pytest.approx(101.0)  # D0's own last bar close

    # D1 must show 0.0 -- untouched by this trade, proving it never
    # carried over (a missed entry-bar check would close there instead,
    # with held_days=2).
    d0, d1 = daily.index
    assert d1 == df.index[2].date()
    assert daily.loc[d1] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# 7. Look-ahead-proofness.
# ---------------------------------------------------------------------

def test_signal_fn_never_receives_a_frame_longer_than_i_plus_1():
    # Same "recording" pattern as test_mining_backtest_engine.py's own
    # look-ahead test -- records the frame length actually observed at
    # each call and asserts it is EXACTLY i+1, which a naive
    # non-truncating implementation fails.
    df = _minute_frame([
        ("2026-03-02", [("09:30", 100), ("09:31", 100), ("09:32", 102)]),
        ("2026-03-03", [("09:30", 103), ("09:31", 104)]),
    ])
    observed = []

    def recording_signal_fn(frame, i):
        observed.append((i, len(frame)))
        return SignalResult(entry=(i == 0), exit=(i == 3))

    run_intraday_backtest(df, recording_signal_fn)
    assert len(observed) > 0
    for i, frame_len in observed:
        assert frame_len == i + 1, (
            f"signal_fn at i={i} received a frame of length {frame_len}, "
            f"expected exactly {i + 1} (rows 0..i only -- no look-ahead)")


# ---------------------------------------------------------------------
# 8. Empty-trades case: dtypes and all-zero daily_returns.
# ---------------------------------------------------------------------

def test_empty_trades_df_has_correct_dtypes_and_zero_daily_returns():
    df = _minute_frame([
        ("2026-03-02", [("09:30", 100), ("09:31", 100)]),
        ("2026-03-03", [("09:30", 100), ("09:31", 100)]),
    ])

    def signal_fn(frame, i):
        return SignalResult()

    trades, daily = run_intraday_backtest(df, signal_fn)

    assert len(trades) == 0
    assert list(trades.columns) == [
        "entry_datetime", "exit_datetime", "entry_px", "exit_px",
        "held_days", "reason", "gross_ret", "net_ret"]
    assert trades["entry_datetime"].dtype == "datetime64[ns, America/New_York]"
    assert trades["exit_datetime"].dtype == "datetime64[ns, America/New_York]"
    assert trades["entry_px"].dtype == np.float64
    assert trades["exit_px"].dtype == np.float64
    assert trades["held_days"].dtype == np.int64
    assert trades["reason"].dtype == object
    assert trades["gross_ret"].dtype == np.float64
    assert trades["net_ret"].dtype == np.float64

    assert len(daily) == 2
    assert (daily == 0.0).all()
    assert daily.dtype == np.float64


def test_empty_input_frame_returns_empty_trades_and_empty_daily_returns():
    df = _minute_frame([])  # zero rows, zero days

    def signal_fn(frame, i):
        return SignalResult()

    trades, daily = run_intraday_backtest(df, signal_fn)
    assert len(trades) == 0
    assert len(daily) == 0


# ---------------------------------------------------------------------
# Validation (mirrors backtest_engine.py's own guard conventions).
# ---------------------------------------------------------------------

def test_tax_haircut_rate_out_of_range_raises():
    df = _minute_frame([("2026-03-02", [("09:30", 100), ("09:31", 100)])])

    def signal_fn(frame, i):
        return SignalResult()

    with pytest.raises(ValueError):
        run_intraday_backtest(df, signal_fn, tax_haircut_rate=1.0)
    with pytest.raises(ValueError):
        run_intraday_backtest(df, signal_fn, tax_haircut_rate=-0.1)


def test_invalid_direction_raises():
    df = _minute_frame([("2026-03-02", [("09:30", 100), ("09:31", 100)])])

    def signal_fn(frame, i):
        return SignalResult()

    with pytest.raises(ValueError):
        run_intraday_backtest(df, signal_fn, direction="sideways")
