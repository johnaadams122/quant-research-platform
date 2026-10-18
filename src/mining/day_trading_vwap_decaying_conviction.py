"""VWAP-cross entry with a time-decaying "conviction" take-profit/stop-loss
exit, day-trading track, LONG ONLY.

Built as a STANDALONE module rather than added to
src/mining/day_trading_signals.py; the `SIGNAL_FACTORIES` registry in that
file does not include this family.

NOTE: this family is scored through the account-relative ABI-1 path
(`mining.abi1_driver.evaluate_family` / `record_abi1_outcome`). The generic
ABI-1 entry point (`scripts/run_abi1_mining.py`) has no intraday-timeframe
support at all (its default OHLC loader only fetches DAILY bars), so this
family's path is the bespoke driver
`scripts/run_vwap_cross_decaying_conviction_abi1.py`, which runs this
module's own unmodified `signal_fn` through
`intraday_backtest_engine.run_intraday_backtest` and bridges the result into
ABI-1 via `mining.day_trading_abi1_bridge.build_day_trading_curve` -- see
that driver's own module docstring for the full mechanism and its intraday
data range. `day_trading_signals.SIGNAL_FACTORIES` is not on this family's
scoring path.

## Hypothesis

Same VWAP-cross entry as the sibling family `vwap_trend_intraday_scalp`
(day_trading_signals.py, `make_vwap_trend_intraday_scalp_signal_fn`) --
MIRRORED here (the running-session-VWAP construction and the
confirmed-crossover streak logic in that factory's `signal_fn`), not
imported, so the two families' modules stay independent. But the exit is
COMPLETELY different: instead of flip-exiting on a confirmed reverse VWAP
cross, this family opens a fixed take-profit/stop-loss pair at entry (as
a fraction of entry price, matching this codebase's existing pct-based
TP/SL convention) and lets BOTH distances shrink monotonically toward a
floor as the trade ages -- modeling the idea that a VWAP-trend thesis
which hasn't played out yet deserves a SMALLER target and a TIGHTER
protective stop, not the same wide levels held indefinitely.

## Entry (mirrored from vwap_trend_intraday_scalp, cited above)

Identical running-session-VWAP construction (bar Volume * that bar's own
per-bar VWAP, accumulated from `session_start_et` onward and divided by
cumulative Volume -- same disclosed approximation the sibling already
carries) and confirmed-crossover detection: `vwap_confirmation_bars`
CONSECUTIVE closes strictly above the running session VWAP. SHORT SIDE
DELIBERATELY NOT BUILT here either, mirroring the sibling's own
long-only scope decision exactly (see this candidate's catalog entry
`distinctness_note`).

## Exit -- the genuinely new part (decaying-conviction TP/SL)

At the fill bar (the engine's own next-bar-open execution -- tracked via
a pending-fill flag exactly like `make_orb_5min_atr_stop_scalp_signal_fn`
captures its own `entry_price` in day_trading_signals.py, in the
`pending_entry_fill` block of its `signal_fn`), two starting
distances are recorded as a fraction of entry price: `base_target_pct`
(default 0.01, 1.0%) and `base_stop_pct` (default 0.005, 0.5%).

These starting magnitudes are BENCHMARKED directly against
`make_selling_climax_capitulation_bounce_daytrade_signal_fn`'s own
`profit_target_pct=0.01` / `trailing_stop_pct=0.005` (day_trading_signals.py,
its factory defaults, disclosed in that factory's own docstring as
"modest intraday defaults") -- the ONLY other pct-of-entry-
price (not ATR-multiple) day-track TP/SL pair already in this codebase,
so this candidate's starting values sit in the SAME established order of
magnitude rather than being invented with no reference point. The 2:1
target:stop ratio those two starting values already carry (1.0% / 0.5%)
is preserved throughout the whole decay, since both distances decay by
the identical formula below.

As bars elapse since the fill (`elapsed_minutes`, real wall-clock minutes
from the fill bar's own timestamp -- matching the sibling's own disclosed
"measured from the bar's own timestamp" convention rather than a bar
count), each distance shrinks toward a FLOOR of `floor_fraction` (default
0.25, i.e. a quarter of the base distance) via exponential decay of the
excess above that floor, with a half-life of `half_life_minutes`
(default 30):

    decay = 0.5 ** (elapsed_minutes / half_life_minutes)
    distance(t) = base_pct * (floor_fraction + (1 - floor_fraction) * decay)

See `decayed_distance()` below (exported at module level specifically so
the decay MATH is directly unit-testable, independent of the signal_fn/
engine plumbing around it). At elapsed_minutes=0 this equals base_pct
exactly; as elapsed_minutes -> infinity it approaches (never reaches)
`floor_fraction * base_pct` -- a floor of exactly 0.0 would make the exit
a hair-trigger indistinguishable from ordinary 1-minute noise, which is
why the floor is a disclosed positive fraction rather than zero.

`half_life_minutes=30` is sized deliberately against this track's own
~385-minute usable session (09:30 through the 15:55 force-flat): by the
one-hour mark (2 half-lives) a stalled trade's distances are already down
to `floor_fraction + 0.25*(1 - floor_fraction)` (~44% of the base gap
above the floor), and by ~2-2.5 hours in (4-5 half-lives) the distances
are within a few percent of the floor -- decisively decayed well before
the session ends, but slow enough that a trade closed in the first few
1-minute bars after entry still sees close to the full base distances,
not an instantly-collapsed one. `floor_fraction=0.25` and
`half_life_minutes=30` are both reasoned, DISCLOSED choices, not derived
from any existing codebase constant (no sibling family has a "decay" or
"floor fraction" parameter to benchmark against).

Every bar with a tracked open position: the bar's own High/Low is checked
against the CURRENT (decayed-as-of-this-bar) target/stop price, stop
checked first as the conservative tie-break (same convention as
`make_orb_5min_atr_stop_scalp_signal_fn`, cited above); independently,
the current bar's own America/New_York clock time at/after
`force_flat_et` (default "15:55", matching every other day-track family)
force-flattens the position even if neither TP nor SL has fired -- this
family's own final backstop, exactly mirroring the sibling's own
force-flat time. `max_hold_days` (default 1) is accepted for
interface/`parameters`-dict parity with this track's other families only,
per the disclosed convention `make_selling_climax_capitulation_bounce_daytrade_signal_fn`
already established (same file, its own docstring) -- the position can
never actually survive past the same session under this same-day-only
mechanism, so `max_hold_days` is not itself compared anywhere in this
closure.

On an exit (whether TP, SL, or force-flat), the VWAP-above streak is
reset to zero -- a disclosed design choice preventing an immediate
whipsaw re-entry on the very next bar off a streak left over from the
trade just closed (a real risk right after a fast target/stop exit while
price is still trending above VWAP); a fresh `vwap_confirmation_bars`
streak must build up again before this closure arms another entry.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .intraday_backtest_engine import SignalFn, SignalResult


def decayed_distance(
    base_pct: float,
    floor_fraction: float,
    half_life_minutes: float,
    elapsed_minutes: float,
) -> float:
    """The CURRENT take-profit or stop-loss distance (as a fraction of
    entry price), `elapsed_minutes` after a trade's own fill, given a
    starting `base_pct` distance, a `floor_fraction` of that base distance
    the decay approaches but never crosses, and a `half_life_minutes`
    decay rate for the excess above that floor. See the module docstring's
    "Exit" section for the full derivation and the parameter choices'
    justification.

    Monotonically non-increasing in `elapsed_minutes`: returns exactly
    `base_pct` at `elapsed_minutes=0` and approaches (never reaches)
    `floor_fraction * base_pct` as `elapsed_minutes -> infinity`. A
    negative `elapsed_minutes` (not expected in production -- defensive
    only) is clamped to 0.0 rather than allowed to push the distance
    above `base_pct`.
    """
    if elapsed_minutes < 0:
        elapsed_minutes = 0.0
    decay = 0.5 ** (float(elapsed_minutes) / float(half_life_minutes))
    return float(base_pct) * (
        float(floor_fraction) + (1.0 - float(floor_fraction)) * decay
    )


def make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn(
    vwap_confirmation_bars: int = 2,
    session_start_et: str = "09:30",
    force_flat_et: str = "15:55",
    max_hold_days: int = 1,
    base_target_pct: float = 0.01,
    base_stop_pct: float = 0.005,
    floor_fraction: float = 0.25,
    half_life_minutes: float = 30.0,
) -> SignalFn:
    """VWAP-Cross Decaying-Conviction Exit (Day Trade), LONG ONLY. See the
    module docstring for the full mechanism and every parameter's
    justification/benchmark. `max_hold_days` is accepted for
    `parameters`-dict/interface parity only (see module docstring) and is
    not itself read anywhere in this closure.
    """
    confirm_n = int(vwap_confirmation_bars)

    _sess_h, _sess_m = (int(x) for x in str(session_start_et).split(":"))
    _flat_h, _flat_m = (int(x) for x in str(force_flat_et).split(":"))
    session_start_t = dt.time(_sess_h, _sess_m)
    force_flat_t = dt.time(_flat_h, _flat_m)

    base_target = float(base_target_pct)
    base_stop = float(base_stop_pct)
    floor_frac = float(floor_fraction)
    half_life = float(half_life_minutes)

    # Closure state -- reset defensively on every detected session/day
    # boundary, mirroring vwap_trend_intraday_scalp's own convention
    # (day_trading_signals.py, cited above): a real position must never
    # survive a day boundary under this same-day-only, max_hold_days=1
    # mechanism, so this reset is a safety net, not the primary
    # flattening path (force_flat_et below is that).
    state = {
        "session_date": None,
        "cum_pv": 0.0,
        "cum_v": 0.0,
        "above_streak": 0,
        "in_position": False,
        "pending_entry_fill": False,
        "entry_price": None,
        "entry_ts": None,
    }

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        row = frame.iloc[-1]
        ts = frame.index[-1]
        cur_date = ts.date()
        cur_time = ts.time()

        if state["session_date"] != cur_date:
            state["session_date"] = cur_date
            state["cum_pv"] = 0.0
            state["cum_v"] = 0.0
            state["above_streak"] = 0
            state["in_position"] = False
            state["pending_entry_fill"] = False
            state["entry_price"] = None
            state["entry_ts"] = None

        # --- consume a pending fill: this bar's own Open is the real
        # --- entry price (next-bar-open execution, mirrored from
        # --- make_orb_5min_atr_stop_scalp_signal_fn, cited above) ---
        if state["pending_entry_fill"]:
            state["pending_entry_fill"] = False
            state["in_position"] = True
            state["entry_price"] = float(row["Open"])
            state["entry_ts"] = ts

        # --- running session VWAP accumulation (mirrored from
        # --- vwap_trend_intraday_scalp, cited above) ---
        vol = row["Volume"]
        bar_vwap = row["VWAP"]
        if cur_time >= session_start_t and pd.notna(vol) and pd.notna(bar_vwap):
            state["cum_pv"] += float(vol) * float(bar_vwap)
            state["cum_v"] += float(vol)

        session_vwap = None
        if state["cum_v"] > 0:
            session_vwap = state["cum_pv"] / state["cum_v"]

        close = row["Close"]
        close = float(close) if pd.notna(close) else None

        if session_vwap is not None and close is not None:
            if close > session_vwap:
                state["above_streak"] += 1
            else:
                state["above_streak"] = 0

        # --- manage an open position: decayed stop / decayed target /
        # --- force-flat, stop checked first as the conservative
        # --- tie-break (mirrored from make_orb_5min_atr_stop_scalp_signal_fn) ---
        if state["in_position"]:
            elapsed_minutes = (ts - state["entry_ts"]).total_seconds() / 60.0
            target_dist = decayed_distance(base_target, floor_frac, half_life, elapsed_minutes)
            stop_dist = decayed_distance(base_stop, floor_frac, half_life, elapsed_minutes)
            entry_price = state["entry_price"]
            target_price = entry_price * (1.0 + target_dist)
            stop_price = entry_price * (1.0 - stop_dist)

            low = float(row["Low"])
            high = float(row["High"])

            exit_signal = False
            if low <= stop_price:
                exit_signal = True
            elif high >= target_price:
                exit_signal = True
            elif cur_time >= force_flat_t:
                exit_signal = True

            if exit_signal:
                state["in_position"] = False
                state["entry_price"] = None
                state["entry_ts"] = None
                # Disclosed whipsaw guard -- see module docstring's final
                # paragraph: force a FRESH confirmation streak before this
                # closure will arm another entry.
                state["above_streak"] = 0
                return SignalResult(exit=True)
            return SignalResult()

        # --- confirmed entry, long only, gated on strictly-before-force-flat ---
        if (
            session_vwap is not None
            and state["above_streak"] >= confirm_n
            and cur_time < force_flat_t
        ):
            state["pending_entry_fill"] = True
            return SignalResult(entry=True)

        return SignalResult()

    return signal_fn


SIGNAL_FACTORIES = {
    "vwap_cross_decaying_conviction_exit_daytrade": (
        make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn
    ),
}
"""A local registry mirroring day_trading_signals.SIGNAL_FACTORIES's own
shape, but scoped to just this one family. Kept for interface parity with
that file's own registry shape (a caller can merge the two via
`{**day_trading_signals.SIGNAL_FACTORIES,
**day_trading_vwap_decaying_conviction.SIGNAL_FACTORIES}`) -- without this
module needing to import or modify day_trading_signals.py itself, and
without day_trading_signals.py needing to import this module either.

NOTE: this registry is not this family's scoring path. That path is
`scripts/run_vwap_cross_decaying_conviction_abi1.py`, a bespoke driver that
reads `make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn` directly
(never through this dict) and bridges it into ABI-1
(`mining.abi1_driver.evaluate_family`) -- see the module docstring's NOTE
paragraph above for the full trace."""
