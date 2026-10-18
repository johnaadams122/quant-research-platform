"""Options cycle state machines (the options-chain simulator families): run_wheel (CSP <->
covered call), run_pmcc and run_put_ratio_spread.

Each returns {"nav_daily": pd.Series, "in_market": pd.Series,
"realizations": list[dict]} -- options_curve.build_options_curve
translates this into the ABI-1
{strategy_daily, in_market, trades_df} contract. `nav_daily` is a
CONTINUOUS running account-level return-index (starts at 1.0), NOT a
concatenation of each cycle's own absolute NAV path -- see this file's
own _advance_cycle for why (consecutive cycles live on different
absolute bases; concatenating raw values would create a fake jump at
every state transition).

Calendar/DTE-rule-driven, NOT backtest_engine.SignalFn-based -- these
mirror rotation_engine.run_two_asset_rotation's target_fn pattern more
than run_backtest's discrete-signal pattern. Do not expect a SignalFn
here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from options_engine import bs_delta, bs_price
from . import options_positions

# Numerical-stability backstop for _advance_cycle's realized-exit NAV ONLY --
# same KIND of guard as options_positions.PMCC_FLOOR_BASIS_EPSILON (tiny,
# disclosed, not an economic judgment call), but a DIFFERENT quantity: this
# one guards the value AFTER the active-close cost subtraction below, which
# pmcc_nav_path's own floor cannot reach (that floor bounds the "nav" column
# BEFORE this function ever sees it).
#
# Context (see pmcc_nav_path's own docstring for its double-degeneracy
# floor): flooring pmcc_nav_path alone prevents a ZeroDivisionError, but a
# second failure mode remains one level up. `pmcc_nav_path` can legitimately return a
# "nav" column as small as `nav_floor_fraction * PMCC_FLOOR_BASIS_EPSILON`
# (~1e-8 at the 0.01 production default) in the double-degenerate crash
# scenario -- but `active_cost = cost_pct_of_premium * abs(close_row[exit_col])`
# below is computed from a REAL, un-degenerate premium (e.g. short_price
# ~0.037 in a deep-crash path), not from "nav" itself, so nothing
# stops `active_cost` from EXCEEDING that tiny nav on the very cycle where
# the clamp engaged. `close_row["nav"] - active_cost` then goes negative --
# without this guard, a 130-day 100.0->0.01 crash at iv=0.60 through
# run_pmcc can produce an arbitrarily large negative realized NAV.
# A negative NAV here is worse than a crash: it does not merely
# crash, it makes `strategy_daily = nav_daily.pct_change()`
# (options_curve.build_options_curve) cross BELOW -1.0 when the prior day's
# NAV was positive -- exactly the invariant the property test
# (`test_build_options_curve_strategy_daily_never_below_negative_one`)
# exists to guard, and account_relative.log_excess raises on any such day.
# This clamp closes that gap for ALL THREE drivers that share this
# function (run_wheel/run_pmcc/run_put_ratio_spread) -- harmless to the
# other two, whose own NAV-path floors are provably bounded well above any
# realistic active_cost, so this can only ever engage for PMCC's own
# already-disclosed non-provable floor.
_MIN_REALIZED_NAV_EXIT = 1e-6


def _advance_cycle(path: pd.DataFrame, cycle_base: float, account_nav: float,
                    cost_pct_of_premium: float, exit_col: str):
    """Shared by run_wheel/run_pmcc/run_put_ratio_spread: given one
    cycle's NAV path, find the exit day (the FIRST day close_signal is
    True, else the path's last day), convert each day's absolute NAV
    into a within-cycle return factor chained onto the running
    `account_nav` level, and apply the active-close cost adjustment ONLY
    on the realized exit day (never on expiry/assignment, which does not
    cross a spread -- see the NAV-path functions' own cost convention).

    Returns (new_account_nav, exit_pos, active_closed, realized_nav_exit,
    cycle_nav_rows) where cycle_nav_rows is a list of (date, account_nav)
    tuples for every day from cycle entry through the exit day inclusive.

    PRECONDITION on `exit_col`: its value must be a transacted-premium
    magnitude that an active close always makes NAV more expensive to
    exit at (the active-close cost is applied via abs() specifically so
    this holds for both a non-negative price, e.g. put_price/call_price/
    short_price, and a signed net value like ratio_spread_nav_path's own
    `liability` column). A hypothetical future caller whose exit_col
    instead encodes a signed net-credit-on-close (positive = a credit
    favorable to the trader, not a cost) would have abs() silently invert
    that sign into an always-negative NAV adjustment -- not true of any
    of the three current callers, but worth checking explicitly before
    reusing this helper for a fourth driver.
    """
    signal_positions = np.flatnonzero(path["close_signal"].to_numpy())
    active_closed = len(signal_positions) > 0
    exit_pos = int(signal_positions[0]) if active_closed else len(path) - 1

    cycle_nav_rows = []
    for d in range(exit_pos):
        cycle_ret = float(path["nav"].iloc[d]) / cycle_base
        cycle_nav_rows.append((path.index[d], account_nav * cycle_ret))

    close_row = path.iloc[exit_pos]
    if active_closed:
        # abs() so this always makes the active close MORE expensive,
        # regardless of whether close_row[exit_col] is a non-negative raw
        # price (run_wheel's put_price/call_price, run_pmcc's short_price
        # -- all bs_price(...) outputs, provably >= 0 by construction, so
        # abs() is a no-op there) or a signed net value (run_put_ratio_
        # spread's "liability", options_positions.ratio_spread_nav_path's
        # short_multiple*short_price - long_price, which is typically
        # negative at an active close for this pipeline's own catalog
        # deltas). Without abs(), a negative close_row[exit_col] flips the
        # subtraction into an addition -- a "cost" that INCREASES NAV.
        # Same rule as options_positions._cost_adjusted (see that
        # helper's own docstring): a cost formula that only works for a
        # positive magnitude silently flips direction when a caller
        # passes it a signed value.
        active_cost = cost_pct_of_premium * abs(float(close_row[exit_col]))
        # max(..., _MIN_REALIZED_NAV_EXIT): see that constant's own module-
        # level comment -- active_cost is computed from a real premium, not
        # from "nav" itself, so it can exceed an already near-floor "nav"
        # (PMCC's own non-provable floor) and drive this negative without
        # this guard.
        realized_nav_exit = max(float(close_row["nav"]) - active_cost,
                                 _MIN_REALIZED_NAV_EXIT)
    else:
        realized_nav_exit = float(close_row["nav"])
    realized_ret = realized_nav_exit / cycle_base
    new_account_nav = account_nav * realized_ret
    cycle_nav_rows.append((path.index[exit_pos], new_account_nav))

    return new_account_nav, exit_pos, active_closed, realized_nav_exit, cycle_nav_rows


def _first_valid_iv_position(iv_series: pd.Series):
    valid = iv_series.notna().to_numpy()
    return int(np.argmax(valid)) if valid.any() else None


def run_wheel(price_series: pd.Series, iv_series: pd.Series, r_series: pd.Series,
              div_yield: float, *, target_delta: float = 0.15, dte: int = 21,
              profit_target_pct: float = 0.5, cost_pct_of_premium: float
              ) -> dict:
    """CSP -> (assignment) -> covered call -> (called away) -> CSP, over
    the FULL history of `price_series`. See module docstring for the
    return shape and nav_daily's continuous-return-index construction.

    An expired-worthless covered call stays in the covered_call state
    (re-write against the SAME shares: this model always re-writes);
    only a CALLED-AWAY exit (S_T > K_call, not active-closed) returns the
    state to a fresh CSP. Every active close (profit-target hit,
    whichever leg) re-enters the SAME state's next cycle the following
    trading day.
    """
    n = len(price_series)
    i = _first_valid_iv_position(iv_series)
    if i is None:
        return {"nav_daily": pd.Series(dtype=float),
                "in_market": pd.Series(dtype=bool, name="in_market"),
                "realizations": []}

    nav_rows = []
    realizations = []
    account_nav = 1.0
    state = "csp"
    cost_basis = None

    while i <= n - 1:
        S0 = float(price_series.iloc[i])
        iv0 = float(iv_series.iloc[i])
        r0 = float(r_series.iloc[i])

        if state == "csp":
            pos = options_positions.build_short_put(
                S0, iv0, dte, target_delta=target_delta, r=r0, q=div_yield)
            path = options_positions.short_put_nav_path(
                pos["K"], pos["entry_credit"], i, price_series, iv_series, r_series,
                q=div_yield, dte=dte, profit_target_pct=profit_target_pct,
                cost_pct_of_premium=cost_pct_of_premium)
            exit_col = "put_price"
        else:
            pos = options_positions.build_covered_call(
                S0, iv0, dte, target_delta=target_delta, r=r0, q=div_yield,
                cost_basis=cost_basis)
            path = options_positions.covered_call_nav_path(
                pos["K"], pos["entry_credit"], i, price_series, iv_series, r_series,
                q=div_yield, dte=dte, profit_target_pct=profit_target_pct,
                cost_pct_of_premium=cost_pct_of_premium)
            exit_col = "call_price"

        cycle_base = float(path["nav"].iloc[0])
        account_nav, exit_pos, active_closed, realized_nav_exit, cycle_rows = _advance_cycle(
            path, cycle_base, account_nav, cost_pct_of_premium, exit_col)
        nav_rows.extend(cycle_rows)
        exit_i = i + exit_pos
        ran_to_data_end = (not active_closed) and (exit_i == n - 1) and (exit_pos < dte)
        S_T = float(price_series.iloc[exit_i])

        if active_closed:
            reason = "active_close"
        elif ran_to_data_end:
            reason = "end_of_data"
        elif state == "csp" and S_T < pos["K"]:
            reason = "assigned"
        elif state == "covered_call" and S_T > pos["K"]:
            reason = "called_away"
        else:
            reason = "expiry_otm"

        realizations.append({
            "entry_date": path.index[0], "exit_date": path.index[exit_pos],
            "nav_entry": cycle_base, "nav_exit": realized_nav_exit, "reason": reason,
        })

        if ran_to_data_end:
            break
        # `carries_shares_forward`: True exactly when the account continues
        # holding the SAME physical shares from this cycle's exit day
        # (exit_i) into the next cycle's entry day (i = exit_i + 1), with a
        # full untracked trading day of real price risk in between. Two
        # cases:
        #   (a) "assigned": a CSP just got exercised -- the account takes
        #       delivery of shares at the strike and immediately starts the
        #       covered_call state next cycle, still holding those shares.
        #   (b) covered_call rolling over into ANOTHER covered_call cycle
        #       (active_close/expiry_otm while `state` -- captured BEFORE
        #       the reassignment below -- was already "covered_call"): the
        #       short call closed/expired but the shares were never sold,
        #       so the SAME shares carry into the next write.
        # Deliberately NOT true for "called_away" (the shares are SOLD at
        # the call strike on exit -- the account converts to cash exactly
        # at exit_i, so the next CSP cycle correctly starts fresh with
        # nothing held overnight) nor for a CSP->CSP active_close/expiry_otm
        # (cash the whole time, never any shares to carry).
        carries_shares_forward = (reason == "assigned") or (
            state == "covered_call" and reason in ("active_close", "expiry_otm"))
        if reason == "assigned":
            cost_basis = pos["K"] - pos["entry_credit"] * (1.0 - cost_pct_of_premium)
            state = "covered_call"
        elif reason == "called_away":
            state = "csp"
        # active_close/expiry_otm: stay in the same state (re-write next cycle)
        i = exit_i + 1
        if carries_shares_forward and i <= n - 1:
            # Bridge the one connecting trading day (exit_i's close -> i's
            # close) that the next cycle's own NAV chain can never see --
            # `_advance_cycle` always sets `cycle_base = path["nav"].iloc[0]`
            # to the NEW cycle's own entry-day NAV, so its first-day ratio
            # is trivially 1.0 regardless of what happened overnight. That
            # is correct when the account held CASH between cycles (a fresh
            # CSP), but wrong here: real shares were held the whole time, so
            # the real close-to-close SPY move on day `i` must be attributed
            # before the next cycle's own chain starts (mutating
            # `account_nav` itself, not just appending a row, so every later
            # day of the next cycle also chains off the corrected base).
            account_nav *= float(price_series.iloc[i]) / float(price_series.iloc[exit_i])
            # Also append an explicit row for day `i` -- redundant-but-safe
            # with the next cycle's own d=0 row (`_advance_cycle` will
            # append (day i, account_nav * 1.0) for that cycle's first day,
            # identical to this value since `account_nav` was already
            # updated above; `nav_daily`'s own `duplicated(keep="last")`
            # dedup means whichever of the two survives holds the same
            # number either way) -- kept explicit for clarity/robustness.
            nav_rows.append((price_series.index[i], account_nav))

    nav_daily = pd.Series({d: v for d, v in nav_rows}).sort_index()
    nav_daily = nav_daily[~nav_daily.index.duplicated(keep="last")]
    in_market = pd.Series(True, index=nav_daily.index, name="in_market")
    return {"nav_daily": nav_daily, "in_market": in_market, "realizations": realizations}


def run_pmcc(price_series: pd.Series, iv_series: pd.Series, r_series: pd.Series,
             div_yield: float, *, long_target_delta: float = 0.80, long_dte: int = 252,
             short_target_delta: float = 0.25, short_dte: int = 21,
             long_roll_dte_floor: int = 63, roll_trigger_delta: float = 0.60,
             profit_target_pct: float = 0.5, cost_pct_of_premium: float,
             nav_floor_fraction: float, long_reset_delta: float = 0.40) -> dict:
    """Poor Man's Covered Call: one long-dated deep-ITM LEAPS (rolled only
    when its own remaining time drops below `long_roll_dte_floor`, OR when
    its own delta decays below `long_reset_delta` -- see that parameter's
    own note below) financing repeated near-term OTM short-call sub-cycles
    (rolled monthly, or defensively via `roll_trigger_delta` while
    T_short_t > 0 -- the short leg is never allowed to reach assignment
    EXCEPT in one narrow gap: a move entirely within a sub-cycle's LAST
    trading day, after which T_short_t=0 and close_signal can no longer
    fire by construction. That gap is not silently mislabeled here -- see
    the moneyness check below).

    Raises ValueError up front if `long_roll_dte_floor <= short_dte`
    (without this check, a retune could
    silently let a short sub-cycle outlive its own LEAPS -- the LEAPS
    would then sit priced at bare T<=0 intrinsic value via bs_price's own
    T<=0 branch for the remainder of that sub-cycle, financing the short
    call as if it were still a live position when it is economically
    already expired).

    `long_reset_delta` (default 0.40, mirroring the short leg's own
    `roll_trigger_delta=0.60` defensive-delta convention, applied to the
    opposite side of the spread): a value-based reset trigger, needed
    because the time-based trigger alone leaves a gap. A
    LEAPS crushed from deep-ITM toward worthless by a severe, fast crash
    can sit BELOW `long_roll_dte_floor`'s time trigger for months while its
    own value has collapsed to near-zero -- with only the time trigger,
    the model would keep financing new
    short calls against an economically-dead LEAPS, driving
    `entry_net_debit` (pmcc_nav_path's own return basis) to near-zero or
    negative and manufacturing extreme, economically nonsensical
    percentage swings (near-total sub-cycle losses even while the
    underlying is rising). This check runs every loop
    iteration, evaluated on the SAME LEAPS delta `pmcc_nav_path`'s own
    `short_delta_t` machinery already computes one level down (`bs_delta`,
    same signature). Not claimed to be the economically "correct"
    threshold -- a reasoned starting point, open to retuning.

    `long_leg_is_fresh_entry` (pmcc_nav_path's own parameter) is tracked
    here: True only for the FIRST short sub-cycle immediately after a
    genuine LEAPS purchase (the initial one, any DTE-based roll, or
    any delta-based reset), False for every later sub-cycle against the
    same LEAPS -- so entry cost on the long leg is charged only when it is
    actually being transacted.

    Returns the usual `{nav_daily, in_market, realizations}` PLUS two
    informational keys -- `leaps_roll_count` (how many times `_enter_leaps`
    fired via the TIME-based trigger) and `leaps_value_reset_count` (how
    many times it fired via the delta-based trigger) -- not part of the shape every
    other driver returns, harmless to any consumer that reads only the
    three standard keys.
    """
    if long_roll_dte_floor <= short_dte:
        raise ValueError(
            f"run_pmcc: long_roll_dte_floor ({long_roll_dte_floor}) must be strictly "
            f"greater than short_dte ({short_dte}) -- otherwise a short sub-cycle "
            f"could outlive its own LEAPS, letting it silently reach true expiry "
            f"mid-sub-cycle (see this function's own docstring)")
    if long_reset_delta >= long_target_delta:
        raise ValueError(
            f"run_pmcc: long_reset_delta ({long_reset_delta}) must be strictly less "
            f"than long_target_delta ({long_target_delta}) -- otherwise a freshly "
            f"entered/rolled/reset LEAPS' own delta (solved to land at "
            f"long_target_delta) would already sit at or below the reset "
            f"threshold, silently firing the value-based reset on every single "
            f"sub-cycle and defeating PMCC's long-dated-LEAPS premise (see "
            f"long_reset_delta's own docstring paragraph)")

    n = len(price_series)
    i = _first_valid_iv_position(iv_series)
    if i is None:
        return {"nav_daily": pd.Series(dtype=float),
                "in_market": pd.Series(dtype=bool, name="in_market"),
                "realizations": [], "leaps_roll_count": 0,
                "leaps_value_reset_count": 0}

    def _enter_leaps(entry_i):
        S0 = float(price_series.iloc[entry_i])
        iv0 = float(iv_series.iloc[entry_i])
        r0 = float(r_series.iloc[entry_i])
        return options_positions.build_pmcc_leaps(
            S0, iv0, long_dte, target_delta=long_target_delta, r=r0, q=div_yield)

    leaps = _enter_leaps(i)
    leaps_entry_i = i
    leaps_roll_count = 0
    leaps_value_reset_count = 0
    long_leg_is_fresh_entry = True
    nav_rows = []
    realizations = []
    account_nav = 1.0
    # LEAPS' own bs_price re-mark at the PRIOR sub-cycle's exit day, carried
    # across loop iterations -- None whenever there is no continuity to
    # bridge yet (before the first sub-cycle, or immediately after a LEAPS
    # roll bought a genuinely fresh position). See the bridge below for why
    # this exists: the LEAPS persists across nearly every short-call
    # sub-cycle boundary (that is the whole point of PMCC), so this is the
    # PMCC analogue of run_wheel's carries_shares_forward connecting-day bridge
    # -- except re-marked through the LEAPS' own bs_price (S, K, T, r, iv can
    # all differ day to day), not the underlying's raw price ratio.
    prev_leaps_exit_mark = None

    while i <= n - 1:
        T_long_remaining = leaps["T_entry"] - (i - leaps_entry_i) / 252.0
        if T_long_remaining * 252.0 < long_roll_dte_floor:
            leaps = _enter_leaps(i)
            leaps_entry_i = i
            T_long_remaining = leaps["T_entry"]
            leaps_roll_count += 1
            long_leg_is_fresh_entry = True
            # A LEAPS roll buys a genuinely fresh position (the old LEAPS is
            # not carried forward) -- same shape as run_wheel's "called_away"
            # exclusion, so there is no connecting-day gap to bridge here.
            prev_leaps_exit_mark = None

        S0 = float(price_series.iloc[i])
        iv0 = float(iv_series.iloc[i])
        r0 = float(r_series.iloc[i])

        # Value-based reset (see this function's own `long_reset_delta`
        # docstring paragraph): checked AFTER the DTE-based roll above, on
        # whatever LEAPS is current at this point -- if the DTE branch just
        # rolled, `leaps` is already fresh (delta ~= long_target_delta,
        # always above long_reset_delta for any sane pair of the two), so
        # this naturally never double-processes the same reset in one
        # iteration without needing an explicit guard.
        leaps_delta_now = bs_delta(S0, leaps["K"], T_long_remaining, r0, iv0,
                                    "call", q=div_yield)
        if leaps_delta_now < long_reset_delta:
            leaps = _enter_leaps(i)
            leaps_entry_i = i
            # Required: T_long_remaining
            # MUST be reset to the fresh LEAPS' own T_entry here,
            # exactly like the DTE-based roll branch above does -- otherwise
            # long_mark_now below (and every day of this new sub-cycle's own
            # pmcc_nav_path decay) prices the brand-new LEAPS' strike
            # (solved for long_target_delta at a FULL fresh T) against the
            # OLD, stale, much-smaller T_long_remaining left over from the
            # LEAPS just discarded -- mispricing the entire post-reset
            # sub-cycle and very plausibly re-tripping a roll/reset almost
            # immediately (thrashing), the opposite of this reset's intent.
            T_long_remaining = leaps["T_entry"]
            leaps_value_reset_count += 1
            long_leg_is_fresh_entry = True
            # Same reasoning as the DTE-based roll: a reset buys a genuinely
            # fresh position, so there is no connecting-day gap to bridge.
            prev_leaps_exit_mark = None

        short = options_positions.build_pmcc_short_call(
            S0, iv0, short_dte, target_delta=short_target_delta, r=r0, q=div_yield)
        long_mark_now = bs_price(S0, leaps["K"], T_long_remaining, r0, iv0,
                                  "call", q=div_yield)

        if (not long_leg_is_fresh_entry) and (prev_leaps_exit_mark is not None):
            # The SAME LEAPS carried continuously across this sub-cycle
            # boundary (no roll happened above) -- bridge the connecting
            # day's real LEAPS re-mark move before starting this sub-cycle's
            # own chain, mirroring run_wheel's covered-call connecting-day
            # bridge but using the LEAPS' own bs_price re-mark (`long_mark_now`,
            # just computed above, IS the LEAPS' mark at this entry day;
            # `prev_leaps_exit_mark`, captured below at the end of the prior
            # iteration, is the same LEAPS' mark at that cycle's exit day)
            # rather than the underlying's raw price ratio -- the LEAPS'
            # value depends on S, K, T, r, iv, not just S.
            #
            # BOTH sides floored at options_positions.PMCC_FLOOR_BASIS_
            # EPSILON: that module's own docstring documents a deep-enough-
            # OTM LEAPS re-mark can underflow to EXACTLY 0.0 in float64 (the
            # same degeneracy PMCC_FLOOR_BASIS_EPSILON/_MIN_REALIZED_NAV_EXIT
            # already guard elsewhere in this file). Flooring only the
            # denominator is not enough on its own, as
            # test_run_pmcc_stays_positive_through_a_crash_to_zero (a
            # 130-day 100.0->0.01 crash fixture at iv=0.60) exercises: a
            # genuinely-zero `long_mark_now` NUMERATOR would drive
            # `account_nav` to EXACTLY 0.0 on the bridge itself (not merely
            # small), breaking the same "nav must be strictly positive,
            # never exactly 0.0" invariant PMCC_FLOOR_BASIS_EPSILON guards,
            # from a call site pmcc_nav_path's own floor cannot reach (this
            # bridge runs BEFORE pmcc_nav_path's
            # own floor ever sees anything). Flooring both sides guarantees
            # this bridge ratio is always a ratio of two strictly-positive
            # numbers, so it can shrink account_nav arbitrarily close to
            # zero but never multiply it by exactly 0.0 in one step.
            bridge_numer = max(long_mark_now, options_positions.PMCC_FLOOR_BASIS_EPSILON)
            bridge_denom = max(prev_leaps_exit_mark, options_positions.PMCC_FLOOR_BASIS_EPSILON)
            account_nav *= bridge_numer / bridge_denom
            nav_rows.append((price_series.index[i], account_nav))

        path = options_positions.pmcc_nav_path(
            leaps["K"], short["K"], T_long_remaining, long_mark_now,
            short["entry_credit"], i, price_series, iv_series, r_series,
            q=div_yield, short_dte=short_dte, roll_trigger_delta=roll_trigger_delta,
            profit_target_pct=profit_target_pct, cost_pct_of_premium=cost_pct_of_premium,
            nav_floor_fraction=nav_floor_fraction,
            long_leg_is_fresh_entry=long_leg_is_fresh_entry)
        long_leg_is_fresh_entry = False  # only the sub-cycle right after a roll is fresh

        cycle_base = float(path["nav"].iloc[0])
        account_nav, exit_pos, active_closed, realized_nav_exit, cycle_rows = _advance_cycle(
            path, cycle_base, account_nav, cost_pct_of_premium, "short_price")
        nav_rows.extend(cycle_rows)
        exit_i = i + exit_pos
        ran_to_data_end = (not active_closed) and (exit_i == n - 1) and (exit_pos < short_dte)
        S_T = float(price_series.iloc[exit_i])
        if active_closed:
            reason = "active_close"
        elif ran_to_data_end:
            reason = "end_of_data"
        elif S_T > short["K"]:
            # Real ITM-at-true-expiry, despite the roll_trigger_delta
            # defense -- the one narrow gap that defense cannot close
            # (see this function's own docstring). Labeling this exit
            # "short_expiry_otm" would falsify the "never reaches
            # assignment" claim with no detection whenever this gap is
            # hit, so it gets its own reason.
            reason = "short_assigned_at_expiry"
        else:
            reason = "short_expiry_otm"
        realizations.append({
            "entry_date": path.index[0], "exit_date": path.index[exit_pos],
            "nav_entry": cycle_base, "nav_exit": realized_nav_exit, "reason": reason,
        })
        if ran_to_data_end:
            break
        # Capture THIS sub-cycle's own LEAPS re-mark at its exit day, for
        # the next iteration's bridge above -- valid regardless of why this
        # sub-cycle exited (active_close/short_expiry_otm/short_assigned_at_
        # expiry all keep the LEAPS; only a roll, handled at the top of the
        # next iteration, discards it, which is why that branch resets this
        # back to None rather than trusting this value unconditionally).
        prev_leaps_exit_mark = float(path["long_price"].iloc[exit_pos])
        i = exit_i + 1

    nav_daily = pd.Series({d: v for d, v in nav_rows}).sort_index()
    nav_daily = nav_daily[~nav_daily.index.duplicated(keep="last")]
    in_market = pd.Series(True, index=nav_daily.index, name="in_market")
    return {"nav_daily": nav_daily, "in_market": in_market, "realizations": realizations,
            "leaps_roll_count": leaps_roll_count,
            "leaps_value_reset_count": leaps_value_reset_count}


def run_put_ratio_spread(price_series: pd.Series, iv_series: pd.Series,
                          r_series: pd.Series, div_yield: float, *,
                          long_delta: float = 0.30, short_delta: float = 0.15,
                          short_multiple: int = 2, dte: int = 21,
                          profit_target_pct: float = 0.5,
                          cost_pct_of_premium: float) -> dict:
    """Put ratio spread, one cycle at a time -- no state transitions (every
    exit, active or at expiry, re-enters the identical shape the next
    trading day). Cash-settled at expiry via bs_price's own T=0 intrinsic
    collapse; this model does NOT simulate physical assignment of the naked excess
    short legs (disclosed, unmodeled -- see options_positions.
    ratio_spread_nav_path's own docstring)."""
    n = len(price_series)
    i = _first_valid_iv_position(iv_series)
    if i is None:
        return {"nav_daily": pd.Series(dtype=float),
                "in_market": pd.Series(dtype=bool, name="in_market"),
                "realizations": []}

    nav_rows = []
    realizations = []
    account_nav = 1.0

    while i <= n - 1:
        S0 = float(price_series.iloc[i])
        iv0 = float(iv_series.iloc[i])
        r0 = float(r_series.iloc[i])
        pos = options_positions.build_put_ratio_spread(
            S0, iv0, dte, long_delta=long_delta, short_delta=short_delta,
            short_multiple=short_multiple, r=r0, q=div_yield)
        path = options_positions.ratio_spread_nav_path(
            pos["K_long"], pos["K_short"], pos["short_multiple"], pos["entry_credit"],
            i, price_series, iv_series, r_series, q=div_yield, dte=dte,
            profit_target_pct=profit_target_pct, cost_pct_of_premium=cost_pct_of_premium)

        cycle_base = float(path["nav"].iloc[0])
        account_nav, exit_pos, active_closed, realized_nav_exit, cycle_rows = _advance_cycle(
            path, cycle_base, account_nav, cost_pct_of_premium, "liability")
        nav_rows.extend(cycle_rows)
        exit_i = i + exit_pos
        ran_to_data_end = (not active_closed) and (exit_i == n - 1) and (exit_pos < dte)
        reason = ("active_close" if active_closed else
                  "end_of_data" if ran_to_data_end else "expiry")
        realizations.append({
            "entry_date": path.index[0], "exit_date": path.index[exit_pos],
            "nav_entry": cycle_base, "nav_exit": realized_nav_exit, "reason": reason,
        })
        if ran_to_data_end:
            break
        i = exit_i + 1

    nav_daily = pd.Series({d: v for d, v in nav_rows}).sort_index()
    nav_daily = nav_daily[~nav_daily.index.duplicated(keep="last")]
    in_market = pd.Series(True, index=nav_daily.index, name="in_market")
    return {"nav_daily": nav_daily, "in_market": in_market, "realizations": realizations}
