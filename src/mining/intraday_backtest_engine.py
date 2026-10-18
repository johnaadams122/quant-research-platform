"""Intraday-capable backtest engine for the strategy mining pipeline
(the shared engine of its intraday/swing and day-trading tracks).

A SIBLING to `backtest_engine.py` (the daily-bar engine), not a
modification of it -- mirrors that module's style and conventions
exactly (see its own docstring): the `SignalResult` dataclass, the
look-ahead-proof slicing convention (`signal_fn` receives a slice of the
frame ending EXACTLY at the bar it may see -- `frame.iloc[i + 1]` always
raises IndexError, making look-ahead structurally impossible), next-bar-
open entry / same-bar-close exit execution, and the `_close_trade`
cost/tax formula (`net_ret = gross_ret - cost_bps/1e4`, then a haircut on
positive net returns only). Consumes minute-bar OHLCV frames (the shape
`intraday_data.fetch_intraday_bars` produces: RTH-only, tz-aware
"America/New_York" DatetimeIndex) instead of daily bars, and adds a
day-based `max_hold_days` cap plus a genuine per-day mark-to-market
return decomposition that the daily engine has no need for (a daily-bar
position's every held day already IS a bar; an intraday position can be
held across many bars within one day and many days overall, so "one
return per day" has to be built explicitly rather than falling out of
the bar loop for free).

This module has NO concept of a holdout window -- same as
`backtest_engine.py` and `rotation_engine.py`, neither of which contains
any holdout-window splitting/pinning/padding logic; that concept lives
entirely in the ORCHESTRATING script that calls these engines (not built
here).

Never imports or modifies `backtest_engine.py`, `rotation_engine.py`, or
`intraday_data.py` -- additive only.

## The mark-to-market design (the module's core design problem)

Every day a position is held -- INCLUDING the entry day itself, not just
the days after it -- must get a real daily return, not a zero followed
by a lump-sum spike on the exit day (a zeros-then-spike pattern would
badly distort the volatility/skew/kurtosis inputs `dsr.py` depends on).
This is achieved via an EXACT telescoping decomposition.

(Note: the entry day gets a leg of its OWN. Using entry_px only as a
denominator for leg_1 would silently shift the entry day's real
open-to-close P&L onto the following calendar day, which is not per-day
mark-to-market and gives incorrect daily return dates/values for a
realistic overnight hold. So every touched day, including d_0, gets its
own real leg.)

For a single trade spanning trading days d_0 (entry day) through d_N
(exit day), define a reference price `ref(d)` for each touched day:
  - ref(d) = exit_px if d == d_N (the exit day, whether or not it is
    ALSO the entry day -- see the same-day case below)
  - ref(d) = that day's LAST bar's Close present in the input frame,
    otherwise (a day the position is held all the way through, whether
    the entry day or a later interior day) -- NOT a hardcoded close
    time, so a half trading session (early close) is handled
    automatically: whatever bar is actually last for that day is the
    reference, full stop.

Each touched day d_i (i = 0..N) gets a "leg" return. For LONG:
  - leg_0 = ref(d_0)/entry_px - 1  (the entry day's OWN move: from the
    actual next-bar-open execution price to that day's own reference)
  - leg_i (i = 1..N) = ref(d_i)/ref(d_{i-1}) - 1

This telescopes EXACTLY (as a PRODUCT):
  product(1 + leg_i for i in 0..N) == ref(d_N)/entry_px
                                    == exit_px/entry_px  (long)
with no approximation -- every interior reference cancels algebraically,
and this holds for ANY N >= 0, including N=0.

## SHORT's trade-level convention: LINEAR, matching `backtest_engine.py`
## EXACTLY -- zero deviation (see the boundedness proof below)

`trades_df`'s short `gross_ret` here is `1 - exit_px/entry_px`, IDENTICAL
to `backtest_engine.py`'s own convention (`gross_ret = long_ret if
direction == "long" else -long_ret`) -- no deviation, in either
direction. A deliberate deviation (making short reciprocal-consistent,
`entry_px/exit_px - 1`, to get clean multiplicative telescoping) would be
WRONG, for a decisive, non-stylistic reason: boundedness.

**The boundedness proof (the reason this convention wins, not a
preference):** a short's real, physical maximum possible gain on fixed
notional is exactly 100% -- the shorted asset falls to zero, you owe
nothing to buy it back, you keep the full entry proceeds. Never more
than that, by definition. Check each candidate formula against this
hard limit as `exit_px -> 0`:
  - LINEAR (`1 - exit_px/entry_px`): as exit_px -> 0, this -> 1.0 (100%)
    exactly -- matches the real cap precisely. On the loss side, as
    exit_px -> infinity, this -> -infinity, correctly unbounded (a
    short's LOSS potential genuinely is unlimited -- only the gain side
    is capped).
  - RECIPROCAL (`entry_px/exit_px - 1`, the rejected alternative): as
    exit_px -> 0, this -> +infinity -- an impossible result. Concretely:
    entry=100, exit=50 (price HALVED) should be a real +50% gain (bought
    back at half price, profit = 50 on 100 of entry capital); the
    reciprocal formula gives 100/50-1 = +100%, exactly DOUBLE the real
    number. This is not a rounding error or an edge case -- it is wrong
    at ordinary, realistic magnitudes, not just in the limit.
Only the linear convention is economically sound across the full range
of possible outcomes. This is a hard mathematical constraint the
reciprocal convention violates, not a matter of taste -- see
test_short_gross_ret_is_bounded_at_100_percent_as_exit_approaches_zero
for the concrete, permanent regression guard.

For SHORT, every leg uses the SAME FIXED denominator (`entry_px`), not
the previous day's mark:
  - leg_i (short, i = 0..N) = (ref(d_{i-1}) - ref(d_i)) / entry_px,
    where ref(d_{-1}) = entry_px (the baseline)

This telescopes EXACTLY too, but as a SUM, not a product (consecutive
terms cancel additively):
  sum(leg_i for i in 0..N) == (entry_px - ref(d_N)) / entry_px
                            == (entry_px - exit_px) / entry_px
                            == 1 - exit_px/entry_px  (short)
which is `trades_df`'s own short `gross_ret` EXACTLY, term for term, with
NO residual gap.

**WARNING for any downstream reader of `daily_returns`, stated explicitly
because the naive version of this mistake is easy to make: a
SHORT trade's own entries in `daily_returns` SUM to that trade's
`net_ret` -- they do NOT COMPOUND (multiply) to it.** E.g. a 2-day short
with legs -5% and -5% has net_ret == -10% (sum), NOT -9.75%
((1-0.05)*(1-0.05)-1, the wrong, compounded answer). This is an
unavoidable, disclosed consequence of the boundedness proof above, not
an oversight: the ONLY way to make a per-day SHORT decomposition
compound multiplicatively to a total was the rejected reciprocal
formula (alternative 3 below), which is economically unbounded and wrong.
Given a hard choice between "the daily series compounds cleanly" and
"the daily series is economically bounded and correct," boundedness
wins -- an internal-consistency nicety does not outrank a real number
being wrong. This does NOT corrupt this module's actual downstream
consumer: DSR/Sharpe-style statistics (mean, std, skew, kurtosis) are
computed directly on each `daily_returns` VALUE as its own independent
observation -- never by compounding/reconstructing a multi-day total
from the series -- so this asymmetry has no adverse effect on the real
consumer, only on a hypothetical reader who assumed (incorrectly) that
every entry in this Series is a compoundable percentage return in the
conventional sense. The SAME-DAY multi-trade aggregation step below
still compounds (multiplicatively) when combining MULTIPLE trades' legs
that land on one calendar day -- that is a different operation (combining
independent trades sharing one capital base over time), not "reconstructing
one trade's own total," and it stays correct per the disclosed invariant
two sections below (same-direction-only in practice).

Design alternatives (the first three are rejected), listed so a reader
can see why the fourth is the one used:
  1. Negation of the long leg (`-(curr_ref/prev_ref - 1)`), combined by
     PRODUCT, short `gross_ret` left linear -- does not telescope AT
     ALL. Example: entry=100, one interior mark=105, exit=110 ->
     negated legs compound to -0.095238, matching neither -0.10 nor
     anything economically meaningful.
  2. Reciprocal ratio (`prev_ref/curr_ref - 1`), combined by PRODUCT,
     short `gross_ret` STILL left linear -- telescopes exactly, but to
     `entry_px/exit_px - 1`, a genuinely different number from the
     linear `gross_ret` it was being reconciled against. Mathematically
     sound as a decomposition, but mismatched against the trade-level
     convention it had to answer to.
  3. Reciprocal ratio AGAIN, this time with `gross_ret` ALSO redefined to
     `entry_px/exit_px - 1` (so the mismatch from alternative 2 disappears,
     everything telescopes cleanly, LONG and SHORT become structurally
     identical) -- this looks like the clean resolution, but the
     boundedness check above shows it produces an economically
     IMPOSSIBLE result (>100% gain on a fixed-notional short). Telescoping
     cleanly is necessary but not sufficient; a formula also has to
     respect the real economics it is supposed to represent.
  4. THIS one -- additive, fixed-basis legs
     (`(prev_ref - curr_ref)/entry_px`), combined by SUM, short
     `gross_ret` back to linear -- properly bounded (see the proof
     above), telescopes exactly to `trades_df`'s own `gross_ret` with
     zero gap, and is IDENTICAL to `backtest_engine.py`'s own convention
     (no deviation to justify at all).

A same-day round trip (d_0 == d_N) is NOT special-cased for either
direction -- it falls out of the exact same formula as the N=0 case.
For long: one leg, ref(d_0) == exit_px, so leg_0 == exit_px/entry_px - 1,
the trade's own gross return directly. For short: one leg,
leg_0 == (entry_px - exit_px)/entry_px == 1 - exit_px/entry_px, ALSO the
trade's own gross return directly -- the additive formula reproduces
gross_ret exactly even in this single-leg degenerate case, with no
reconciliation needed. Either way this single leg is ALSO always the
"final" leg (see the cost/tax paragraph below) and gets unconditionally
overwritten by the net-reconciliation formula regardless of what its
pre-override value would have been: the long-vs-short leg-formula choice
only ever affects legs[0:-1] (the non-final legs), and for N=0 that slice
is empty, so a same-day round trip in EITHER direction is completely
unaffected by which per-day formula is used and always lands exactly on
the trade's own `net_ret`. (The test file's hand-derivation comments
cover both directions, including a dedicated same-day short round-trip
test.)

Cost/tax charge ONCE, at realization (the exit leg), matching real
transaction economics -- not distributed across every held day. Every
leg except the final one (leg_0 through leg_{N-1}) contributes its raw
GROSS value (multiplicative-ratio for long, additive-fixed-basis for
short, per the asymmetry above); the final (exit-day) leg is adjusted so
that combining the trade's ENTIRE daily contribution (all legs)
reconstructs the trade's own `net_ret` exactly. The combining operator
matches each direction's own convention:

  LONG:  final_leg_net = (1 + net_ret) / product(1 + leg_i for i in 0..N-1) - 1
  SHORT: final_leg_net = net_ret - sum(leg_i for i in 0..N-1)

Both formulas are structurally the same idea (solve the final term so
the whole series reconstructs `net_ret`), just division/product for a
multiplicative series vs. subtraction/sum for an additive one -- the
same kind of long/short formula asymmetry that already exists at the
trade level, extended consistently. (For N=0, the product is empty == 1
/ the sum is empty == 0, so both formulas reduce to
`final_leg_net == net_ret` directly -- the same-day case, again falling
out of the general formula with no special-casing needed.)

## Same-day multi-trade aggregation and mixed direction: a DISCLOSED,
## VERIFIED invariant of the calling contract, not swept under the rug

Multiple legs from different trades landing on the same calendar day
(e.g. a multi-day hold's exit leg in the morning, then a fresh same-day
round trip that afternoon -- possible because this engine, like
`backtest_engine.py`, only ever holds one position at a time) combine
MULTIPLICATIVELY, in chronological order: product(1 + leg_j) - 1, never
naive summation, REGARDLESS of direction.

Honesty note: this multiplicative combination is only ever applied to
legs sharing the SAME basis in practice (all-long legs, from a
multiplicative/ratio basis, OR all-short legs, from an additive/
fixed-basis measure) -- it is NOT, strictly, a unit-consistent operation
if it were ever handed one leg of each kind (short's additive-fixed-
basis fraction and long's multiplicative-ratio fraction do not mean the
same thing, and compounding them together would produce a number with
no clean economic interpretation). This is SAFE in practice, and
verified as an actual invariant of the calling contract rather than
assumed: `direction` is a WHOLE-CALL parameter (`run_intraday_backtest`
takes exactly one `direction` for its entire invocation, mirroring
`backtest_engine.py`'s own convention) -- there is no per-trade direction
field on `SignalResult`, so a single real call can never produce two
different-direction trades, and therefore never produces two different-
basis legs landing on the same calendar day. Every leg that ever reaches
`per_day_legs[day]` within one call shares the SAME direction and
therefore the SAME basis, by construction of the calling contract, not
by a runtime check in the aggregation code itself. If this module were
ever extended to allow a single call to mix directions (not currently
possible), this aggregation step would need to be
revisited FIRST.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalResult:
    """One bar's signal state, returned by a candidate's signal function."""
    entry: bool = False
    exit: bool = False


SignalFn = Callable[[pd.DataFrame, int], "SignalResult"]
"""A signal function receives a SLICE of the minute-bar OHLCV DataFrame
ending EXACTLY at the bar it may look at, plus that bar's position `i` in
the ORIGINAL (unsliced) frame. `frame` always has exactly `i + 1` rows,
so `frame.iloc[-1]` (equivalently `frame.iloc[i]`) is "the current bar"
and there is no row for "the next bar" to accidentally read --
`frame.iloc[i + 1]` raises IndexError. This makes look-ahead bias
STRUCTURALLY impossible rather than merely forbidden by convention --
same rationale and same convention as `backtest_engine.py`'s SignalFn.

The engine acts on an entry signal at the NEXT bar's open (next-open
execution) and on an exit signal at the SAME bar's close -- identical
execution convention to the daily engine, just operating on minute bars
instead of daily bars."""


_TRADE_COLUMNS = [
    "entry_datetime", "exit_datetime", "entry_px", "exit_px",
    "held_days", "reason", "gross_ret", "net_ret",
]


def run_intraday_backtest(
    df: pd.DataFrame,
    signal_fn: SignalFn,
    *,
    cost_bps: float = 0.0,
    tax_haircut_rate: float = 0.0,
    max_hold_days: int = 20,
    direction: Literal["long", "short"] = "long",
):
    """Run one candidate's signal function over `df` (minute-bar OHLCV,
    columns Open/High/Low/Close at minimum; tz-aware DatetimeIndex,
    ascending, matching intraday_data.fetch_intraday_bars's output
    shape).

    `cost_bps` must be a ROUND-TRIP figure (same convention as
    `backtest_engine.run_backtest`). `tax_haircut_rate` must be in
    [0, 1); raises ValueError otherwise. `max_hold_days` is a
    trading-DAY cap (not a bar-count cap): a position still open at the
    LAST bar of its `max_hold_days`-th trading day (counting the entry
    day as day 1) is force-closed there with reason "time_stop" -- it is
    given the full final day to produce a signal exit first; when both a
    signal exit and the time-stop condition are true on the same bar,
    "signal_exit" wins (same precedence convention as the daily engine).
    This is checked starting on the entry bar ITSELF, not the bar after:
    `max_hold_days=1` with an entry landing on its own day's last bar
    force-closes same-bar, same-day (checking only from the bar after
    entry would let a `max_hold_days=1` position silently survive into the
    next day; see test_max_hold_days_one_force_closes_same_bar_as_entry).
    `direction` is "long" (default) or "short"; short gross return
    mirrors `backtest_engine.py`'s own simple one-share-P&L approximation
    EXACTLY, zero deviation: `gross_ret = long_ret if
    direction == "long" else -long_ret` (i.e. `1 - exit_px/entry_px` for
    short). This is a deliberately BOUNDED, fixed-notional convention --
    see the module docstring's "SHORT's trade-level convention" section
    for the boundedness proof that excludes a cleaner-looking but
    economically impossible alternative (reciprocal-consistent short
    return going unbounded as exit_px -> 0).

    Returns (trades_df, daily_returns):

    - trades_df columns: entry_datetime, exit_datetime, entry_px,
      exit_px, held_days, reason ("signal_exit" | "time_stop" |
      "end_of_data"), gross_ret, net_ret. `held_days` counts distinct
      trading days the position spanned, inclusive of entry and exit day
      (a same-day round trip is held_days=1). Always has correct
      per-column dtypes (datetime64[ns, <df's tz>] for the two datetime
      columns, float64 for prices/returns, int64 for held_days, object
      for reason) even with zero trades -- never falls back to object
      dtype for a numeric/datetime column.

    - daily_returns: a float64 pd.Series with exactly one entry per
      unique trading day present in `df` (derived from `df.index`
      directly -- no external calendar), indexed by `datetime.date`, in
      chronological order. A day untouched by any trade is 0.0. See the
      module docstring for the exact mark-to-market decomposition this
      implements.

    Degenerate edge case, disclosed not fixed (matching the daily
    engine's own disclosed end-of-data edge case): an entry on the FINAL
    bar of `df` is force-closed the same bar via the end_of_data path,
    with held_days=1 -- reachable only with very short input data.
    """
    if not (0.0 <= tax_haircut_rate < 1.0):
        raise ValueError(f"tax_haircut_rate must be in [0, 1), got {tax_haircut_rate}")
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    if not isinstance(max_hold_days, int) or max_hold_days < 1:
        raise ValueError(f"max_hold_days must be a positive int, got {max_hold_days!r}")

    n = len(df)
    tz = df.index.tz

    if n == 0:
        return _empty_trades_df(tz), pd.Series(dtype="float64")

    # --- Precompute per-bar day bookkeeping -------------------------------
    # `.date` on a tz-aware DatetimeIndex returns each timestamp's LOCAL
    # calendar date (i.e. already resolved through the "America/New_York"
    # zone, not a UTC date) -- verified this handles a real DST offset
    # change correctly (see the module's DST test).
    dates = df.index.date
    unique_days = list(pd.unique(dates))  # order-preserving; df is time-ascending
    day_to_idx = {d: idx for idx, d in enumerate(unique_days)}
    day_idx_of_bar = np.array([day_to_idx[d] for d in dates])

    is_last_bar_of_day = np.zeros(n, dtype=bool)
    for i in range(n):
        is_last_bar_of_day[i] = (i == n - 1) or (day_idx_of_bar[i] != day_idx_of_bar[i + 1])

    closes = df["Close"].to_numpy(dtype=float)
    opens = df["Open"].to_numpy(dtype=float)

    last_close_of_day: dict[int, float] = {}
    for i in range(n):
        last_close_of_day[int(day_idx_of_bar[i])] = closes[i]

    trades: list[dict] = []
    # day_idx (int, position in `unique_days`) -> list of leg returns
    # contributed by every trade that touched that day, in chronological
    # order (multiple legs on the same day combine multiplicatively).
    per_day_legs: dict[int, list[float]] = {}

    def _record_legs(entry_px, exit_px, entry_day_idx, exit_day_idx, net_ret):
        # Every touched day (entry_day_idx..exit_day_idx inclusive) gets
        # its own real leg -- including the entry day. `ref(d)` is
        # exit_px for the exit day (whether or not it's ALSO the entry
        # day -- a same-day round trip is d == exit_day_idx on its very
        # first iteration below, which correctly uses exit_px rather than
        # the day's later last-bar close) and that day's last-bar Close
        # otherwise. No special-casing needed for the same-day case: it
        # falls out of this loop as the N=0 case -- see the module
        # docstring's same-day-round-trip paragraph (both directions).
        #
        # Direction (see the module docstring's "SHORT's trade-level
        # convention" section for the full reasoning, the boundedness
        # proof, and the alternatives considered): long uses the multiplicative
        # ratio against the PREVIOUS mark (`prev_ref`, which advances
        # each iteration). Short uses an ADDITIVE, fixed-basis formula
        # against the ORIGINAL `entry_px` (which never advances --
        # deliberately NOT `prev_ref`) -- this matches
        # backtest_engine.py's own fixed-notional short convention
        # EXACTLY (zero deviation), unlike a negated or reciprocal ratio
        # (both rejected -- the reciprocal one specifically because it
        # produces economically impossible >100% short gains). This only
        # matters for legs[0:-1] (the non-final legs) -- the final leg
        # computed here is unconditionally overwritten below regardless
        # of which formula produced its pre-override value.
        legs: list[list] = []  # [day_idx, gross_leg] pairs, mutated in place
        prev_ref = entry_px
        for d in range(entry_day_idx, exit_day_idx + 1):
            curr_ref = exit_px if d == exit_day_idx else last_close_of_day[d]
            if direction == "long":
                gross_leg = curr_ref / prev_ref - 1.0
            else:
                gross_leg = (prev_ref - curr_ref) / entry_px
            legs.append([d, gross_leg])
            prev_ref = curr_ref

        # Cost/tax realize ONCE, on the final (exit-day) leg: adjust it so
        # that combining every leg (all gross except this one, adjusted)
        # reconstructs the trade's own net_ret exactly. LONG combines by
        # PRODUCT (solve via division); SHORT combines by SUM (solve via
        # subtraction) -- matching each direction's own leg convention
        # above. For a same-day trade `legs` has exactly one element, so
        # `legs[:-1]` is empty: interior_product stays 1.0 (long) /
        # interior_sum stays 0.0 (short), and final_leg_net == net_ret
        # directly either way -- again no special case required.
        if direction == "long":
            interior_product = 1.0
            for _, leg in legs[:-1]:
                interior_product *= (1.0 + leg)
            legs[-1][1] = (1.0 + net_ret) / interior_product - 1.0
        else:
            interior_sum = sum(leg for _, leg in legs[:-1])
            legs[-1][1] = net_ret - interior_sum

        for d, leg in legs:
            per_day_legs.setdefault(d, []).append(leg)

    position: Optional[dict] = None

    def _close_trade(i: int, reason: str) -> None:
        exit_px = closes[i]
        entry_px = position["entry_px"]
        # Linear gross_ret, IDENTICAL to backtest_engine.py's own
        # convention for both directions -- see the module docstring's
        # "SHORT's trade-level convention" section for the boundedness
        # proof that excludes a reciprocal-consistent alternative.
        long_ret = exit_px / entry_px - 1.0
        gross_ret = long_ret if direction == "long" else -long_ret
        net_ret = gross_ret - cost_bps / 1e4
        if tax_haircut_rate and net_ret > 0:
            net_ret *= (1.0 - tax_haircut_rate)

        entry_day_idx = int(day_idx_of_bar[position["entry_idx"]])
        exit_day_idx = int(day_idx_of_bar[i])
        held_days = exit_day_idx - entry_day_idx + 1

        trades.append({
            "entry_datetime": df.index[position["entry_idx"]],
            "exit_datetime": df.index[i],
            "entry_px": entry_px,
            "exit_px": exit_px,
            "held_days": held_days,
            "reason": reason,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })
        _record_legs(entry_px, exit_px, entry_day_idx, exit_day_idx, net_ret)

    for i in range(n):
        if position is None:
            if i == 0:
                continue
            sig = signal_fn(df.iloc[:i], i - 1)
            if not sig.entry:
                continue
            position = {"entry_idx": i, "entry_px": opens[i]}
            # Deliberately NO `continue` here: fall through to the same
            # exit/time-stop check every other bar with an open position
            # gets, evaluated for THIS bar too. A freshly-opened position
            # must be checked on its own entry bar -- otherwise a
            # max_hold_days=1 position entered on a day's LAST bar would
            # never get force-closed until the NEXT day's own last bar,
            # silently violating the documented "force-closed at the LAST
            # bar of its Nth trading day" contract (see
            # test_max_hold_days_one_force_closes_same_bar_as_entry).
            # This is general, not a special case for max_hold_days==1: the
            # entry bar goes through IDENTICAL exit-evaluation to every
            # later bar, rather than being skipped once.

        entry_day_idx = int(day_idx_of_bar[position["entry_idx"]])
        days_held_inclusive = int(day_idx_of_bar[i]) - entry_day_idx + 1
        sig = signal_fn(df.iloc[:i + 1], i)
        hit_time_stop = bool(is_last_bar_of_day[i]) and days_held_inclusive >= max_hold_days
        if sig.exit or hit_time_stop:
            _close_trade(i, "time_stop" if hit_time_stop and not sig.exit else "signal_exit")
            position = None

    if position is not None:
        _close_trade(n - 1, "end_of_data")

    if trades:
        trades_df = pd.DataFrame(trades, columns=_TRADE_COLUMNS)
    else:
        trades_df = _empty_trades_df(tz)

    daily_values = np.empty(len(unique_days), dtype=float)
    for idx in range(len(unique_days)):
        legs = per_day_legs.get(idx)
        if not legs:
            daily_values[idx] = 0.0
        else:
            prod = 1.0
            for leg in legs:
                prod *= (1.0 + leg)
            daily_values[idx] = prod - 1.0

    daily_returns = pd.Series(
        daily_values, index=pd.Index(unique_days, name="date"),
        dtype="float64", name="daily_return")

    return trades_df, daily_returns


def _empty_trades_df(tz) -> pd.DataFrame:
    dt_dtype = f"datetime64[ns, {tz}]" if tz is not None else "datetime64[ns]"
    return pd.DataFrame({
        "entry_datetime": pd.Series(dtype=dt_dtype),
        "exit_datetime": pd.Series(dtype=dt_dtype),
        "entry_px": pd.Series(dtype="float64"),
        "exit_px": pd.Series(dtype="float64"),
        "held_days": pd.Series(dtype="int64"),
        "reason": pd.Series(dtype="object"),
        "gross_ret": pd.Series(dtype="float64"),
        "net_ret": pd.Series(dtype="float64"),
    }, columns=_TRADE_COLUMNS)
