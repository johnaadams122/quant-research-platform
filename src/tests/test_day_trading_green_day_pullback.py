"""Tests for src/mining/day_trading_green_day_pullback.py -- a day-trading
mining candidate modeled on a common discretionary pattern:
on an SPY session that was green from the open, an early rally is followed
by a shallow, quiet pause, with entry on the resumption of the rally and
flat well before 14:30 ET (ahead of the afternoon
profit-taking flow that tends to show up on days like that). This factory is
that shape, made mechanical: LONG SIDE ONLY (matching this catalog's
established day-trading convention, e.g.
day_trading_signals.make_orb_5min_atr_stop_scalp_signal_fn's own long-only
disclosure).

Every scenario below is hand-derived against small, plain 1-minute OHLC
fixtures.

Unlike cascade_signals.py's signals, this mechanism needs no daily/hourly/
5-minute bucketing -- it operates directly on the raw 1-minute frame the
engine hands it, so fixtures here are one row per REAL minute of interest,
not one landmark row per bucket boundary.

Every test is driven through the real `run_intraday_backtest` engine (never
a hand-rolled loop), matching test_cascade_signals.py's own established
convention for this engine.
"""
import os
import sys

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.intraday_backtest_engine import run_intraday_backtest  # noqa: E402
from mining.day_trading_green_day_pullback import (  # noqa: E402
    make_green_day_morning_pullback_continuation_signal_fn,
)


TZ = "America/New_York"
DAY_1 = "2026-03-04"  # Wednesday
DAY_2 = "2026-03-05"  # Thursday
DAY_3 = "2026-03-06"  # Friday


def _ts(date_str, time_str):
    return pd.Timestamp(f"{date_str} {time_str}", tz=TZ)


def _row(o, h, l, c, v=1000.0):
    return {"Open": float(o), "High": float(h), "Low": float(l), "Close": float(c), "Volume": float(v)}


def _frame(rows):
    """rows: list of (timestamp, dict) pairs. Sorted defensively."""
    index = pd.DatetimeIndex([ts for ts, _ in rows])
    data = {col: [r[col] for _, r in rows] for col in ["Open", "High", "Low", "Close", "Volume"]}
    return pd.DataFrame(data, index=index).sort_index()


# Shared, tractable parameter overrides used by most tests below -- the real
# catalog defaults (min_rally_gain_pct=0.0015, min_consolidation_bars=15,
# pullback_window 09:45-11:30, min_target_r_multiple=1.5,
# force_flat_et=14:30) would need ~20+ real-minute rows per fixture; these
# smaller values exercise the identical logic with a handful of rows.
DEFAULT_KWARGS = dict(
    min_rally_gain_pct=0.01,           # 1% rally required off the open
    min_consolidation_bars=3,          # 3 bars with no new high confirms a pause
    max_pullback_retrace_pct=0.5,      # pullback may give back at most half the leg
    pullback_window_start_et="09:30",
    pullback_window_end_et="10:00",
    min_target_r_multiple=2.0,
    force_flat_et="14:30",
)


def _happy_path_rows(day=DAY_1):
    """Session open 100.00. Rally to 101.20 (+1.2% >= 1% threshold) by 09:31.
    Pause 09:32-09:34 (3 bars, no new high, Low never below 100.90 -- a 0.30
    retrace of the 1.20 leg, 25% <= 50% allowed) confirms the pullback at
    09:34, freezing leg_high=101.20 / pullback_low=100.90. 09:35 stays inside
    the range (no breakout, no invalidation). 09:36 closes at 101.30, above
    the frozen leg_high -- breakout entry fires. Fill at 09:37's Open
    (101.35); stop=100.90, target = 101.35 + 2.0*(101.35-100.90) = 102.25.
    09:38's High (102.30) clears the target -- signal_exit.
    """
    return [
        (_ts(day, "09:30"), _row(100.00, 100.00, 100.00, 100.00)),
        (_ts(day, "09:31"), _row(100.00, 101.20, 100.00, 101.20)),
        (_ts(day, "09:32"), _row(101.20, 101.20, 100.90, 101.00)),
        (_ts(day, "09:33"), _row(101.00, 101.00, 100.95, 100.97)),
        (_ts(day, "09:34"), _row(100.97, 101.00, 100.92, 100.98)),
        (_ts(day, "09:35"), _row(100.98, 101.05, 100.91, 100.95)),
        (_ts(day, "09:36"), _row(100.95, 101.30, 100.95, 101.30)),
        (_ts(day, "09:37"), _row(101.35, 101.40, 101.30, 101.38)),
        (_ts(day, "09:38"), _row(101.38, 102.30, 101.36, 102.20)),
    ]


def test_happy_path_rally_pause_breakout_hits_target():
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(_happy_path_rows()), signal_fn, max_hold_days=1, direction="long")

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["entry_px"] == pytest.approx(101.35)
    assert trade["exit_px"] == pytest.approx(102.20)  # exits at the bar's Close
    assert trade["reason"] == "signal_exit"
    assert trade["gross_ret"] > 0


def test_stop_loss_exit_before_target():
    """Same setup through the breakout fill, but the bar after fill drops
    through the stop (100.90) instead of reaching the target."""
    rows = _happy_path_rows()[:8]  # through the 09:37 fill bar
    rows.append((_ts(DAY_1, "09:38"), _row(101.38, 101.38, 100.80, 100.85)))  # Low breaches stop
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["reason"] == "signal_exit"
    assert trade["exit_px"] == pytest.approx(100.85)
    assert trade["gross_ret"] < 0


def test_force_flat_exit_fires_before_stop_or_target():
    """After a real fill, force_flat_et (overridden to 09:40 for a short
    fixture) closes the position on the wall-clock bar even though neither
    stop nor target has been touched."""
    kwargs = dict(DEFAULT_KWARGS, force_flat_et="09:40")
    rows = _happy_path_rows()[:8]  # through the 09:37 fill bar
    # Neither stop (100.90) nor target (102.25) touched; price just drifts.
    rows.append((_ts(DAY_1, "09:38"), _row(101.38, 101.50, 101.30, 101.45)))
    rows.append((_ts(DAY_1, "09:39"), _row(101.45, 101.55, 101.35, 101.50)))
    rows.append((_ts(DAY_1, "09:40"), _row(101.50, 101.60, 101.40, 101.55)))
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**kwargs)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["reason"] == "signal_exit"
    assert trade["exit_px"] == pytest.approx(101.55)  # the 09:40 bar's Close


def test_invalidation_close_below_session_open_during_rally_phase():
    """A day that closes below its own session open WHILE STILL in
    'awaiting_rally' (before the rally threshold is even met) has already
    broken the 'green day throughout' premise this whole family depends
    on -- it must never later qualify just because a subsequent V-shaped
    recovery happens to clear the rally threshold. This mirrors the
    identical close-below-open invalidation already enforced once a
    pullback is being tracked (see test_invalidation_close_below_
    session_open_before_entry) -- the SAME rule must hold from the very
    first bar of the day, not just from the moment a rally is recognized."""
    rows = [
        (_ts(DAY_1, "09:30"), _row(100.00, 100.00, 100.00, 100.00)),
        (_ts(DAY_1, "09:31"), _row(100.00, 100.00, 99.30, 99.50)),  # closes < open -- invalidated
        (_ts(DAY_1, "09:32"), _row(99.50, 101.20, 99.50, 101.20)),  # V-shaped rally clears threshold
        (_ts(DAY_1, "09:33"), _row(101.20, 101.20, 100.90, 101.00)),
        (_ts(DAY_1, "09:34"), _row(101.00, 101.00, 100.95, 100.97)),
        (_ts(DAY_1, "09:35"), _row(100.97, 101.00, 100.92, 100.98)),  # would confirm pullback here
        (_ts(DAY_1, "09:36"), _row(100.98, 101.05, 100.91, 100.95)),
        (_ts(DAY_1, "09:37"), _row(100.95, 101.30, 100.95, 101.30)),  # would break out here
        (_ts(DAY_1, "09:38"), _row(101.35, 101.40, 101.30, 101.38)),
    ]
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 0


def test_no_rally_never_signals():
    """A flat/choppy day that never gains min_rally_gain_pct off the open
    never leaves 'awaiting_rally' -- no entry, ever."""
    rows = [
        (_ts(DAY_1, "09:30"), _row(100.00, 100.05, 99.95, 100.00)),
        (_ts(DAY_1, "09:31"), _row(100.00, 100.10, 99.90, 100.02)),
        (_ts(DAY_1, "09:32"), _row(100.02, 100.08, 99.92, 99.98)),
        (_ts(DAY_1, "09:33"), _row(99.98, 100.05, 99.95, 100.01)),
    ]
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 0


def test_continuous_new_highs_never_confirm_a_pullback():
    """The rally threshold is met, but price keeps printing a fresh high
    every single bar (bars_since_new_high never accumulates) -- the pullback
    never confirms, so there is never a breakout to trade."""
    rows = [
        (_ts(DAY_1, "09:30"), _row(100.00, 100.00, 100.00, 100.00)),
        (_ts(DAY_1, "09:31"), _row(100.00, 101.20, 100.00, 101.20)),  # rally threshold met
        (_ts(DAY_1, "09:32"), _row(101.20, 101.40, 101.10, 101.35)),  # new high
        (_ts(DAY_1, "09:33"), _row(101.35, 101.60, 101.30, 101.55)),  # new high
        (_ts(DAY_1, "09:34"), _row(101.55, 101.80, 101.50, 101.75)),  # new high
        (_ts(DAY_1, "09:35"), _row(101.75, 102.00, 101.70, 101.95)),  # new high
    ]
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 0


def test_pullback_outside_window_never_confirms():
    """The exact happy-path shape, but shifted so the pause completes at
    10:05 -- after the (09:30-10:00) window override -- so the pullback is
    never confirmed and no breakout entry is ever searched for, even though
    price does eventually clear the (unfrozen) leg_high."""
    rows = [
        (_ts(DAY_1, "09:57"), _row(100.00, 100.00, 100.00, 100.00)),
        (_ts(DAY_1, "09:58"), _row(100.00, 101.20, 100.00, 101.20)),
        (_ts(DAY_1, "09:59"), _row(101.20, 101.20, 100.90, 101.00)),
        (_ts(DAY_1, "10:00"), _row(101.00, 101.00, 100.95, 100.97)),
        (_ts(DAY_1, "10:01"), _row(100.97, 101.00, 100.92, 100.98)),  # would confirm here,
        # but 10:01 > pullback_window_end_et (10:00) -- must NOT confirm.
        (_ts(DAY_1, "10:02"), _row(100.98, 101.30, 100.95, 101.30)),  # would-be breakout bar
        (_ts(DAY_1, "10:03"), _row(101.35, 101.40, 101.30, 101.38)),
    ]
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 0


def test_invalidation_close_below_session_open_before_entry():
    """After the pullback confirms (awaiting_breakout), price closes back
    below the session open -- the 'green day' premise breaks, so the setup
    is abandoned for the day even though price later reclaims and would
    otherwise have broken out above the frozen leg_high."""
    rows = _happy_path_rows()[:5]  # through the 09:34 confirmation bar
    rows.append((_ts(DAY_1, "09:35"), _row(100.98, 100.99, 99.50, 99.80)))  # closes < 100.00 open
    rows.append((_ts(DAY_1, "09:36"), _row(99.80, 101.50, 99.75, 101.40)))  # would break out
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 0


def test_invalidation_undercut_pullback_low_before_breakout():
    """After confirmation, a bar's Low trades below the frozen
    pullback_low (100.90) -- the consolidation structure is broken, so the
    setup is abandoned even though price never closed below the session
    open and later reclaims above leg_high."""
    rows = _happy_path_rows()[:5]  # through the 09:34 confirmation bar
    rows.append((_ts(DAY_1, "09:35"), _row(100.98, 100.99, 100.70, 100.85)))  # Low < 100.90
    rows.append((_ts(DAY_1, "09:36"), _row(100.85, 101.50, 100.80, 101.40)))  # would break out
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 0


def test_one_trade_per_day_no_reentry_after_exit():
    """After the happy-path trade closes on a target hit, a second,
    independently-qualifying rally/pause/breakout shape later the SAME day
    is never taken -- one trade per day maximum, matching this catalog's
    established day-trading convention (e.g. cascade_signals.py)."""
    rows = _happy_path_rows()  # closes via target hit at 09:38
    # A second full rally/pause/breakout cycle, later the same day.
    rows += [
        (_ts(DAY_1, "10:00"), _row(102.20, 102.20, 102.20, 102.20)),
        (_ts(DAY_1, "10:01"), _row(102.20, 103.50, 102.20, 103.50)),
        (_ts(DAY_1, "10:02"), _row(103.50, 103.50, 103.20, 103.30)),
        (_ts(DAY_1, "10:03"), _row(103.30, 103.30, 103.25, 103.28)),
        (_ts(DAY_1, "10:04"), _row(103.28, 103.30, 103.22, 103.27)),
        (_ts(DAY_1, "10:05"), _row(103.27, 104.00, 103.27, 104.00)),  # would break out
        (_ts(DAY_1, "10:06"), _row(104.05, 104.10, 104.00, 104.08)),
    ]
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 1  # only the first trade


def test_state_resets_across_calendar_days():
    """Day 1 never rallies enough to trade; day 2 (a fresh session_open,
    fresh leg_high) runs the identical happy-path shape and trades
    normally -- day 1's state must not leak into day 2."""
    day1_rows = [
        (_ts(DAY_1, "09:30"), _row(100.00, 100.05, 99.95, 100.00)),
        (_ts(DAY_1, "09:31"), _row(100.00, 100.10, 99.90, 100.02)),
    ]
    day2_rows = _happy_path_rows(day=DAY_2)
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(day1_rows + day2_rows), signal_fn, max_hold_days=1, direction="long")
    assert len(trades_df) == 1
    assert trades_df.iloc[0]["entry_datetime"].date() == pd.Timestamp(DAY_2).date()


def test_never_raises_on_insufficient_history():
    """A 1-row and a 2-row frame must never raise -- both far short of
    min_consolidation_bars, and the factory must simply keep returning a
    no-signal result rather than crash on missing lookback. Each case uses
    its own fresh closure so the two checks are independent of each other."""
    one_row = _frame([(_ts(DAY_1, "09:30"), _row(100.00, 100.00, 100.00, 100.00))])
    result = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)(one_row, 0)
    assert result.entry is False and result.exit is False

    two_rows = _frame([
        (_ts(DAY_1, "09:30"), _row(100.00, 100.00, 100.00, 100.00)),
        (_ts(DAY_1, "09:31"), _row(100.00, 101.20, 100.00, 101.20)),
    ])
    result = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)(two_rows, 1)
    assert result.entry is False and result.exit is False


def test_stale_position_recovers_after_a_malformed_cross_day_gap():
    """Structurally, a real entry can only fire strictly before
    force_flat_et, and this factory's own force-flat check runs on every
    bar while a position is open -- so on any REAL, gap-free RTH frame the
    position is always flattened well before the day rolls over. This test
    deliberately feeds a malformed frame that jumps from a fill bar
    straight to the next calendar day (one bar, day 2), skipping every bar
    that would normally trigger the force-flat exit, then gives day 3 a
    completely fresh, independently-tradeable happy-path shape.

    The engine's OWN max_hold_days=1 backstop will force-close the day-1
    position regardless of anything this factory does (it does not need
    the signal_fn's cooperation to do that) -- so "day 1's trade closes on
    day 2" is true either way and proves nothing on its own. The real
    proof is day 3: without this factory's own defensive stale-position
    safety net (mirroring day_trading_signals.
    make_vwap_trend_intraday_scalp_signal_fn's own established pattern),
    the closure's internal `open` flag would never learn the engine force-
    closed its position, and the entry-search step (gated on `not open`)
    would then stay dark for the rest of the run -- day 3 would silently
    never trade, no matter how clean its own setup is.

    Passes `max_hold_days=2` to the ENGINE for this call only (a deliberate
    test-only choice, not this family's own real max_hold_days=1
    day-trading convention): at max_hold_days=1, the fill bar (09:37) is
    immediately followed by day 2 in this truncated frame, which makes it
    the engine's own idea of "the last bar of its own trading day" too --
    intraday_backtest_engine.py's own documented same-bar force-close rule
    would then close the position AT THE FILL BAR ITSELF, on day 1, before
    the malformed gap this test is about ever has a chance to matter.
    max_hold_days=2 lets the position genuinely survive to see day 2's bar,
    which is what this test needs to exercise."""
    rows = _happy_path_rows()[:8]  # through the 09:37 fill bar (position now open)
    rows.append((_ts(DAY_2, "09:30"), _row(102.00, 102.10, 101.90, 102.05)))
    rows += _happy_path_rows(day=DAY_3)
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=2, direction="long")

    assert len(trades_df) == 2, (
        "day 1's stale position must be force-closed AND day 3's fresh "
        "setup must still be tradeable -- if the closure's own `open` "
        "flag never resets, day 3 silently never enters")
    day1_trade, day3_trade = trades_df.iloc[0], trades_df.iloc[1]
    assert day1_trade["exit_datetime"].date() == pd.Timestamp(DAY_2).date()
    assert day3_trade["entry_datetime"].date() == pd.Timestamp(DAY_3).date()
    assert day3_trade["exit_px"] == pytest.approx(102.20)


def test_armed_entry_discarded_rather_than_filled_with_a_stale_cross_day_stop():
    """The failure this guards against: an entry can be ARMED
    (breakout confirmed, decided but not yet filled) on the last bar of a
    day, with `stop_price` captured from THAT day's own frozen pullback
    low. If the fill lands on a DIFFERENT calendar day (only reachable on a
    genuine NYSE early-close session, since force_flat_et normally
    guarantees bars remain), completing the fill would combine a REAL new-
    day entry_px with a STALE, economically unrelated stop_price from the
    day before -- corrupting the computed target and, in this fixture,
    triggering an immediate stale-stop exit that has nothing to do with the
    new day's own price structure.

    Proves the guard: the armed decision is DISCARDED on the new day's first
    bar (this closure never completes the fill, and never fires an exit of
    its own), so the resulting trade is closed by the ENGINE's own honest
    max_hold_days backstop (reason="time_stop", at day 2's own actual last
    bar) rather than this closure's own stale-stop-triggered "signal_exit"
    on day 2's very first bar."""
    rows = _happy_path_rows()[:7]  # through the 09:36 breakout-arm bar (last bar of day 1)
    # Day 2 gaps down well below day 1's frozen_pullback_low (100.90) -- a
    # stale stop check would fire immediately on bar 1; the real day-2
    # price action has nothing to do with day 1's structure.
    rows.append((_ts(DAY_2, "09:30"), _row(99.50, 99.60, 99.40, 99.55)))
    rows.append((_ts(DAY_2, "09:31"), _row(99.55, 99.70, 99.45, 99.65)))  # day 2's actual last bar
    signal_fn = make_green_day_morning_pullback_continuation_signal_fn(**DEFAULT_KWARGS)
    trades_df, _daily = run_intraday_backtest(
        _frame(rows), signal_fn, max_hold_days=1, direction="long")

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["reason"] == "time_stop", (
        "a stale cross-day stop_price must never drive this closure's own "
        "exit -- only the engine's own honest end-of-day backstop should "
        "close a position whose arming decision spanned a day boundary")
    assert trade["exit_datetime"] == _ts(DAY_2, "09:31")
    assert trade["exit_px"] == pytest.approx(99.65)
