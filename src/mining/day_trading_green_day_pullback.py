"""Green-day morning pullback continuation scalp -- a day-trading mining
candidate modeled on a common discretionary pattern: on an SPY session that
is green from the open, recognize an early rally and a brief shallow pause,
enter on the resumption of the rally, and be flat well before 14:30 ET --
ahead of the afternoon profit-taking flow that tends to show up on days like
that.

LONG SIDE ONLY (disclosed scope, matching this catalog's established
day-trading convention, e.g. day_trading_signals.
make_orb_5min_atr_stop_scalp_signal_fn's own long-only disclosure).

Mechanism, evaluated directly on the raw 1-minute OHLCV frame the engine
hands this closure (no daily/hourly/5-minute bucketing needed, unlike
cascade_signals.py's mechanism):

1. RALLY (phase "awaiting_rally", from the session open): `session_open` is
   captured from the FIRST bar seen for each calendar date (the input frame
   is RTH-only per intraday_data.fetch_intraday_bars, so that bar's own Open
   is the real 09:30 open). Checked first, every bar in this phase: if the
   bar's Close drops below `session_open`, the "green day throughout"
   premise this whole family depends on is already broken -- abandoned for
   the day (phase -> "done_today") even if a later bar's own V-shaped
   recovery would otherwise clear the rally threshold (this check must hold
   from the very first bar, not just from the moment a rally is later
   recognized in step 2).
   Otherwise, `leg_high` tracks the running max High since the open. Once
   `(leg_high - session_open) / session_open >= min_rally_gain_pct`, the day
   has rallied enough to look for a pullback -- move to "awaiting_pullback".

2. PULLBACK (phase "awaiting_pullback"): `leg_high` keeps extending on any
   bar whose High exceeds it (a continuing rally resets the pause clock:
   `bars_since_new_high` -> 0, `pullback_low` -> None, so the pause is always
   measured from the MOST RECENT high). On a bar that does NOT make a new
   high, `bars_since_new_high` increments and `pullback_low` tracks the
   running min Low since the last new high. Once `bars_since_new_high >=
   min_consolidation_bars` AND the retracement so far
   (`leg_high - pullback_low`) is at most `max_pullback_retrace_pct` of the
   rally leg (`leg_high - session_open` -- a shallow pause, not a reversal;
   50% is the standard technical-analysis heuristic for "still a
   continuation pattern", the same kind of generalized price-action
   convention `mining.price_structure`'s own primitives document) AND the
   CONFIRMING bar's own wall-clock time falls inside
   [`pullback_window_start_et`, `pullback_window_end_et`] (this family is
   specifically the MORNING version of this shape -- an afternoon rally/pause
   is a different pattern, not this one), the pullback is confirmed:
   `leg_high`/`pullback_low` are FROZEN into `frozen_leg_high`/
   `frozen_pullback_low` and phase moves to "awaiting_breakout". If the
   confirmation conditions are not all met on the bar that first reaches
   `min_consolidation_bars`, the phase stays "awaiting_pullback" and keeps
   re-checking on every later bar (a new high can still reset the cycle; the
   window bound can still be missed permanently if the pause completes too
   late). Checked BEFORE the new-high/no-new-high branching, on every bar
   in this phase: if the bar's Close drops below `session_open`, the
   "still a green day" premise is broken and the setup is abandoned for the
   day (phase -> "done_today") even if a rally/pause shape technically
   exists.

3. BREAKOUT (phase "awaiting_breakout"): `frozen_leg_high`/
   `frozen_pullback_low` no longer move. Checked first, every bar: if Close
   drops below `session_open`, OR Low trades below `frozen_pullback_low`
   (the pause's own structure fails), the setup is abandoned for the day
   (phase -> "done_today") -- a broken pause is not a continuation signal.
   Otherwise, the instant a bar's Close closes back above `frozen_leg_high`
   -- while strictly before `force_flat_et` (the same guard this catalog's
   cascade_signals.py and day_trading_signals.
   make_vwap_trend_intraday_scalp_signal_fn factories already use: without
   it, an entry decided too close to the cutoff can fill on the FOLLOWING
   calendar day, made invisible to this closure's own day-boundary
   bookkeeping) -- entry=True fires, `stop_price` is set to
   `frozen_pullback_low`, and the closure marks itself `armed` (decided,
   not yet filled; the engine fills at the NEXT bar's open).

4. FILL CAPTURE (checked on every call before the setup phases, whenever
   `armed` and not yet `open`): the engine's own next-bar-open
   execution convention means THIS call's `frame.iloc[-1]` is the bar whose
   Open is the real fill price for the entry decided on the PREVIOUS call.
   `target_price = entry_px + min_target_r_multiple * (entry_px -
   stop_price)` -- a plain R-multiple target off the real fill, no extra
   reference level (unlike cascade_signals.py's own target formula, which
   additionally floors at the prior day's high -- that reference has no
   equivalent concept in THIS mechanism, which never looks at daily bars at
   all). Sets `open=True`, `armed=False`, records `entry_date`, and
   deliberately falls through (no early return) to step 6's exit check for
   this SAME bar -- a freshly-filled position must still be checked for an
   immediate stop/target/force-flat hit on its own fill bar (mirrors
   intraday_backtest_engine.py's own documented "no continue" convention,
   and cascade_signals.py's own identical fall-through).

5. DEFENSIVE STALE-POSITION SAFETY NET (checked right after the new-day
   reset, before step 4, unconditionally): if `open` and `entry_date` is a
   calendar day before today, force an exit. On a max_hold_days=1 day-trading run this should
   never actually fire in real operation -- step 6's own force-flat check
   runs on every bar while open and force_flat_et is always well before the
   session's last bar, so this closure's OWN exit logic should always
   flatten the position before the day ever rolls over. It is kept anyway,
   mirroring make_vwap_trend_intraday_scalp_signal_fn's own established
   precedent for this catalog: the ENGINE's own max_hold_days backstop will
   force-close a stale position regardless of what this closure does, but
   only THIS check tells the CLOSURE ITSELF that it is flat again --
   without it, a bug (or a malformed/gapped caller) that let `open` survive
   past a day boundary would permanently block this closure's own entry
   search for the rest of the run, since entry detection is gated on
   `not open`.

6. EXIT (checked whenever `open`, independent of phase): stop checked first
   (Low <= `stop_price`), then target (High >= `target_price`), then
   independently the force-flat cutoff (bar's own wall-clock time at or
   after `force_flat_et`) regardless of stop/target status -- same
   precedence convention as every other factory in this catalog's
   day-trading track. Any exit resets `open`/`armed`/`stop_price`/
   `target_price`/`entry_date` and moves phase to "done_today" -- one trade
   per day maximum, matching this catalog's established "only one position
   tracked at a time" convention (e.g. cascade_signals.py).

`force_flat_et` defaults to 14:30 ET -- a deliberate ~90-minute-earlier
cutoff than every other family in this catalog's day-trading track (which
exit 15:55-15:59), matching the discretionary pattern this family
encodes: exit ahead of the afternoon profit-taking/reversal flow, not at
the ordinary end-of-day flatten time.

Disclosed, unresolved residual (handled only as far as this factory can
without real trading-calendar awareness -- the same class of residual
cascade_signals.py's own module docstring discloses for its identical
late-arm-then-next-day-fill scenario): an
`armed` decision whose fill lands on a DIFFERENT calendar day than the bar
that armed it (only reachable on a genuine NYSE early-close session, since
`force_flat_et` guarantees an ordinary session has bars remaining) is
DISCARDED on the new day's first bar (`armed`/`stop_price` are cleared by
the new-day reset at the top of `signal_fn`)
rather than filled with the prior day's now-stale `stop_price` -- but the
ENGINE has already unconditionally opened that position at the new day's
open regardless of this closure's own bookkeeping, so it rides untouched by
this closure's own stop/target/force-flat logic until the engine's own
`max_hold_days` backstop force-closes it. This factory has no trading-
calendar input to detect an early close in advance.

Never raises on early bars with insufficient history at any step -- simply
stays in the current phase and returns SignalResult(). A fresh factory call
=> a fresh, empty `state` dict; nothing here is shared across signal_fn
instances (matches every other factory in this module's sibling files).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from mining.intraday_backtest_engine import SignalFn, SignalResult


def _parse_et_clock(value: str) -> dt.time:
    return dt.datetime.strptime(value, "%H:%M").time()


def make_green_day_morning_pullback_continuation_signal_fn(
    min_rally_gain_pct: float = 0.0015,
    min_consolidation_bars: int = 15,
    max_pullback_retrace_pct: float = 0.5,
    pullback_window_start_et: str = "09:45",
    pullback_window_end_et: str = "11:30",
    min_target_r_multiple: float = 1.5,
    force_flat_et: str = "14:30",
) -> SignalFn:
    """See module docstring for the full mechanism. Every parameter is used
    directly: `min_rally_gain_pct` gates the rally-leg threshold,
    `min_consolidation_bars`/`max_pullback_retrace_pct` size the pullback
    confirmation, `pullback_window_start_et`/`pullback_window_end_et` bound
    when that confirmation may occur, `min_target_r_multiple` sizes the
    profit target off the real fill/stop, and `force_flat_et` gates both the
    entry cutoff and the unconditional time-based exit."""
    min_gain = float(min_rally_gain_pct)
    min_bars = int(min_consolidation_bars)
    max_retrace = float(max_pullback_retrace_pct)
    window_start_t = _parse_et_clock(pullback_window_start_et)
    window_end_t = _parse_et_clock(pullback_window_end_et)
    min_r = float(min_target_r_multiple)
    force_flat_t = _parse_et_clock(force_flat_et)

    state = {
        "last_date": None,
        "phase": "awaiting_rally",
        "session_open": None,
        "leg_high": None,
        "bars_since_new_high": 0,
        "pullback_low": None,
        "frozen_leg_high": None,
        "frozen_pullback_low": None,
        "armed": False,
        "open": False,
        "entry_date": None,
        "stop_price": None,
        "target_price": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        row = frame.iloc[-1]
        ts = frame.index[-1]
        today = ts.date()

        # --- STEP 1: new trading day -- reset SETUP-ONLY tracking, PLUS any
        # armed-but-unfilled decision. Never touches open/entry_date/
        # target_price for an ALREADY-open position: those stay the sole
        # job of step 2's defensive safety net and step 6's exit, so a
        # believed-open position remains genuinely exit-checkable even
        # across a day boundary (mirrors make_vwap_trend_intraday_scalp_
        # signal_fn's own identical reset-scope discipline for an OPEN
        # position). An ARMED-but-not-yet-open decision is different and is
        # reset here (matching cascade_signals.py's own day-boundary reset,
        # which clears its equivalent "armed" flag too): `stop_price` for an
        # armed entry was captured from the PRIOR day's own frozen price
        # structure, and step 3's fill capture would otherwise combine it
        # with the NEW day's real entry_px to size a target (an early-close
        # session that arms in its final minutes, then fills on the next
        # calendar day, would compute entry_px - stop_price against two
        # unrelated sessions' prices; a large enough overnight gap flips
        # that risk sign entirely). Discarding the stale arm means the ENGINE's own
        # resulting position (it fills unconditionally at the next bar's
        # open regardless of this closure's own bookkeeping) rides
        # untouched by this closure's stop/target logic until the engine's
        # own max_hold_days backstop force-closes it -- a genuine, narrow,
        # DISCLOSED residual, the same shape cascade_signals.py's own module
        # docstring already discloses for the identical late-arm-then-
        # next-day-fill scenario. Neither signal has real trading-calendar
        # awareness to detect an early close in advance.
        if state["last_date"] != today:
            state["last_date"] = today
            state["phase"] = "awaiting_rally"
            state["session_open"] = float(row["Open"])
            state["leg_high"] = float(row["High"])
            state["bars_since_new_high"] = 0
            state["pullback_low"] = None
            state["frozen_leg_high"] = None
            state["frozen_pullback_low"] = None
            state["armed"] = False
            state["stop_price"] = None

        entry_signal = False
        exit_signal = False

        # --- STEP 2: defensive stale-position safety net -- checked first,
        # unconditionally. See module docstring point 5.
        if (
            state["open"]
            and state["entry_date"] is not None
            and (today - state["entry_date"]).days >= 1
        ):
            state["open"] = False
            state["armed"] = False
            state["entry_date"] = None
            state["stop_price"] = None
            state["target_price"] = None
            return SignalResult(exit=True)

        # --- STEP 3: fill capture -- checked before setup-phase logic.
        if state["armed"] and not state["open"]:
            entry_px = float(row["Open"])
            stop = state["stop_price"]
            state["target_price"] = entry_px + min_r * (entry_px - stop)
            state["open"] = True
            state["armed"] = False
            state["entry_date"] = today
            # Deliberately NO return -- fall through to step 6's exit check
            # for this same (fill) bar too.

        # --- STEP 4/5: setup phases -- only while no position is
        # armed/open. `phase == "done_today"` matches none of the branches
        # below and simply no-ops for the rest of the day.
        if not state["armed"] and not state["open"]:
            if state["phase"] == "awaiting_rally":
                high = float(row["High"])
                close = float(row["Close"])
                if close < state["session_open"]:
                    state["phase"] = "done_today"
                else:
                    if high > state["leg_high"]:
                        state["leg_high"] = high
                    gain = (state["leg_high"] - state["session_open"]) / state["session_open"]
                    if gain >= min_gain:
                        state["phase"] = "awaiting_pullback"
                        state["bars_since_new_high"] = 0
                        state["pullback_low"] = None

            elif state["phase"] == "awaiting_pullback":
                high = float(row["High"])
                low = float(row["Low"])
                close = float(row["Close"])
                if close < state["session_open"]:
                    state["phase"] = "done_today"
                elif high > state["leg_high"]:
                    state["leg_high"] = high
                    state["bars_since_new_high"] = 0
                    state["pullback_low"] = None
                else:
                    state["bars_since_new_high"] += 1
                    state["pullback_low"] = (
                        low if state["pullback_low"] is None
                        else min(state["pullback_low"], low)
                    )
                    if state["bars_since_new_high"] >= min_bars:
                        leg = state["leg_high"] - state["session_open"]
                        retrace = state["leg_high"] - state["pullback_low"]
                        in_window = window_start_t <= ts.time() <= window_end_t
                        if retrace <= max_retrace * leg and in_window:
                            state["frozen_leg_high"] = state["leg_high"]
                            state["frozen_pullback_low"] = state["pullback_low"]
                            state["phase"] = "awaiting_breakout"

            elif state["phase"] == "awaiting_breakout":
                low = float(row["Low"])
                close = float(row["Close"])
                if close < state["session_open"] or low < state["frozen_pullback_low"]:
                    state["phase"] = "done_today"
                elif close > state["frozen_leg_high"] and ts.time() < force_flat_t:
                    state["stop_price"] = state["frozen_pullback_low"]
                    state["armed"] = True
                    entry_signal = True

        # --- STEP 6: exit (stop / target / force-flat) -- independent of
        # phase, whenever a position is open (including the fill bar
        # itself, via step 3's fall-through).
        if state["open"]:
            low = float(row["Low"])
            high = float(row["High"])
            bar_time = ts.time()
            if low <= state["stop_price"]:
                exit_signal = True
            elif high >= state["target_price"]:
                exit_signal = True
            elif bar_time >= force_flat_t:
                exit_signal = True

            if exit_signal:
                state["open"] = False
                state["armed"] = False
                state["entry_date"] = None
                state["stop_price"] = None
                state["target_price"] = None
                state["phase"] = "done_today"

        return SignalResult(entry=entry_signal, exit=exit_signal)

    return signal_fn
