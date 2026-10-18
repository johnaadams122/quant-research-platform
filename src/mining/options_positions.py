"""Options position constructors + NAV revaluation (options-chain simulator).

Position constructors (this section) are pure strike/entry-price builders
via options_engine.solve_strike_for_delta + bs_price -- no simulation
logic, no state. NAV revaluation functions (the second section, below)
are the daily mark-to-market + close/roll-decision layer each cycle
driver consumes one day at a time.

Deliberately does NOT reuse options_engine.build_iron_condor/
simulate_trade -- those are 4-leg, intraday-High/Low, single-DAY (0DTE)
constructs; this module's positions are 1-2-leg, Close-only, MONTHLY-cycle
constructs. Only bs_price/bs_delta/solve_strike_for_delta are shared.
"""
from __future__ import annotations

import pandas as pd

from options_engine import bs_price, bs_delta, solve_strike_for_delta


def build_short_put(S: float, iv: float, dte: int, *, target_delta: float,
                     r: float, q: float) -> dict:
    """Cash-secured short put: one strike at `-target_delta` (e.g. 0.15
    for a 15-delta put), `entry_credit` = its BSM price. `target_delta`
    is passed POSITIVE by convention (the sign flip to `-target_delta` for
    solve_strike_for_delta's put-delta convention happens inside)."""
    T = dte / 252.0
    K = solve_strike_for_delta(S, T, r, iv, -target_delta, "put", q=q)
    entry_credit = bs_price(S, K, T, r, iv, "put", q=q)
    return {"kind": "short_put", "K": K, "entry_credit": entry_credit,
            "T_entry": T, "dte": dte, "iv": iv, "r": r, "q": q}


def build_covered_call(S: float, iv: float, dte: int, *, target_delta: float,
                        r: float, q: float, cost_basis: float) -> dict:
    """Covered call written against shares already held at `cost_basis`
    (from a prior CSP assignment, or a standing position). One strike at
    `+target_delta`, `entry_credit` = its BSM price.

    `cost_basis` is carried for reporting/informational purposes only and
    is NOT read by `covered_call_nav_path` or anywhere downstream --
    the NAV-chaining formula needs no absolute cost basis (it
    mathematically cancels out of the account-
    level return computation). Disclosed here rather than left as an
    unexplained unused parameter."""
    T = dte / 252.0
    K = solve_strike_for_delta(S, T, r, iv, target_delta, "call", q=q)
    entry_credit = bs_price(S, K, T, r, iv, "call", q=q)
    return {"kind": "covered_call", "K": K, "entry_credit": entry_credit,
            "T_entry": T, "dte": dte, "iv": iv, "r": r, "q": q,
            "cost_basis": cost_basis}


def build_pmcc_leaps(S: float, iv: float, dte: int, *, target_delta: float,
                      r: float, q: float) -> dict:
    """PMCC long leg: a deep-ITM LEAPS call (target_delta ~0.75-0.85,
    needs solve_strike_for_delta's [0.05*S, 3S] call bracket, which
    reaches strikes well below S).
    `entry_cost` (a DEBIT, unlike every other constructor's `entry_credit`)
    is its BSM price."""
    T = dte / 252.0
    K = solve_strike_for_delta(S, T, r, iv, target_delta, "call", q=q)
    entry_cost = bs_price(S, K, T, r, iv, "call", q=q)
    return {"kind": "pmcc_leaps", "K": K, "entry_cost": entry_cost,
            "T_entry": T, "dte": dte, "iv": iv, "r": r, "q": q}


def build_pmcc_short_call(S: float, iv: float, dte: int, *, target_delta: float,
                           r: float, q: float) -> dict:
    """PMCC short leg: a near-term OTM call (target_delta ~0.20-0.30,
    reached by the K >= S part of that same call bracket). `entry_credit`
    is its BSM price."""
    T = dte / 252.0
    K = solve_strike_for_delta(S, T, r, iv, target_delta, "call", q=q)
    entry_credit = bs_price(S, K, T, r, iv, "call", q=q)
    return {"kind": "pmcc_short_call", "K": K, "entry_credit": entry_credit,
            "T_entry": T, "dte": dte, "iv": iv, "r": r, "q": q}


def build_put_ratio_spread(S: float, iv: float, dte: int, *, long_delta: float,
                            short_delta: float, short_multiple: int,
                            r: float, q: float) -> dict:
    """Put ratio spread: one long put closer to the money (`long_delta`,
    e.g. 0.30) and `short_multiple` short puts further OTM (`short_delta`,
    e.g. 0.15) -- `short_multiple - 1` of those short puts are "naked"
    relative to the single long hedge. `short_multiple` must be >= 2 (a
    value of 1 degenerates to a plain vertical spread, not a ratio spread).
    `entry_credit = short_multiple * short_put_price - long_put_price`."""
    if short_multiple < 2:
        raise ValueError(
            f"build_put_ratio_spread: short_multiple must be >= 2 to create "
            f"a real ratio spread (naked excess short legs); got {short_multiple}")
    T = dte / 252.0
    K_long = solve_strike_for_delta(S, T, r, iv, -long_delta, "put", q=q)
    K_short = solve_strike_for_delta(S, T, r, iv, -short_delta, "put", q=q)
    long_price = bs_price(S, K_long, T, r, iv, "put", q=q)
    short_price = bs_price(S, K_short, T, r, iv, "put", q=q)
    entry_credit = short_multiple * short_price - long_price
    return {"kind": "put_ratio_spread", "K_long": K_long, "K_short": K_short,
            "short_multiple": short_multiple, "entry_credit": entry_credit,
            "T_entry": T, "dte": dte, "iv": iv, "r": r, "q": q}


# ── NAV revaluation (daily mark-to-market) ────────────────────────────────────


def _cost_adjusted(gross: float, cost_pct: float) -> float:
    """Apply a %-of-premium transaction cost so it ALWAYS makes a
    position more expensive, regardless of whether `gross` is a credit
    (positive, e.g. a sold put's premium) or a net debit (negative --
    e.g. a put ratio spread at THIS pipeline's own catalog deltas, where
    the closer-to-money long put costs more than the further-OTM short
    puts combined). `gross * (1 - cost_pct)` only does this correctly
    for a positive `gross`; for a negative one it computes BACKWARDS
    (making a debit position cheaper as cost rises).
    This formula is correct for both signs: it always subtracts
    `cost_pct * abs(gross)` from `gross`, i.e. always moves `gross`
    toward being a smaller credit or a bigger debit.

    NOT used for PMCC's long_cost_gross/short_credit_gross -- those are
    separate variables with a FIXED sign convention each (long_cost_gross
    is always a positive magnitude representing money PAID; short_credit_
    gross is always a positive magnitude representing money RECEIVED),
    so they use the ordinary `*(1+cost_pct)`/`*(1-cost_pct)` forms
    directly in pmcc_nav_path -- this helper is for a SINGLE variable that
    can itself flip between representing a credit and a debit, which only
    the ratio spread's entry_credit does."""
    return gross - cost_pct * abs(gross)


def short_put_nav_path(K: float, entry_credit_gross: float, entry_i: int,
                        price_series: pd.Series, iv_series: pd.Series,
                        r_series: pd.Series, *, q: float, dte: int,
                        profit_target_pct: float, cost_pct_of_premium: float
                        ) -> pd.DataFrame:
    """Daily NAV path for a cash-secured short put, from `entry_i` through
    expiry (day `dte`) or the end of `price_series`, whichever is first.

    NAV_t = K + entry_credit_net - put_price_t, entry_credit_net =
    entry_credit_gross * (1 - cost_pct_of_premium). `entry_credit_gross`
    from build_short_put is always positive (a real OTM/near-money put
    always has positive premium, unlike the ratio spread's entry_credit
    -- see _cost_adjusted's own docstring for why that one needs the
    abs()-based formula and this one does not).

    PROVABLY BOUNDED BELOW BY entry_credit_net > 0 for the entire path,
    including at expiry (T=0): bs_price's put formula is
    `K*exp(-rT)*N(-d2) - S*exp(-qT)*N(-d1)`, and since N(-d2) <= 1 and the
    subtracted term is >= 0, `put_price_t <= K*exp(-r*T_t)` always. So
    `NAV_t = K + entry_credit_net - put_price_t >= K*(1-exp(-r*T_t)) +
    entry_credit_net >= entry_credit_net` for any S_t>=0, T_t>=0, r>=0.

    PRECONDITION on this proof: it requires
    r>=0 (so that K*(1-exp(-r*T_t)) >= 0); q's non-negativity is
    irrelevant here (q does not appear in the put formula's bound).
    Dividend yields in this pipeline are always >= 0 by construction
    (options_vol.DIVIDEND_YIELD_TABLE), but `r` comes from a live-fetched
    T-bill series (data_pipeline.fetch_risk_free_rate) with no floor --
    this repo's own cached risk-free-rate history contains genuinely
    negative daily rates (e.g. -0.00105 on 2020-03-26, a COVID-era
    flight-to-safety reading), so this bound's precondition is not always
    strictly satisfied by real data. The resulting violation is a small
    fraction of a cent per contract at realistic rate magnitudes -- not a
    functional NAV-non-positivity risk -- but is a real, unclamped gap
    between this proof's stated precondition and the data this pipeline
    actually consumes, disclosed here rather than assumed away.

    Returns a DataFrame (one row per day held) with columns `nav`,
    `put_price`, `close_signal` (True on a day T_t>0 where put_price_t has
    decayed to profit_target_pct of entry_credit_gross -- an EARLY-close
    candidate the caller may act on; false at expiry itself, which the
    caller detects via `len(path) == dte + 1`, or via truncation if the
    price series ran out first, via `len(path) < dte + 1`).
    """
    entry_credit_net = entry_credit_gross * (1.0 - cost_pct_of_premium)
    last_i = min(entry_i + dte, len(price_series) - 1)
    rows, idx_list = [], []
    for i in range(entry_i, last_i + 1):
        d = i - entry_i
        T_t = (dte - d) / 252.0
        S_t = float(price_series.iloc[i])
        iv_t = float(iv_series.iloc[i])
        r_t = float(r_series.iloc[i])
        put_price_t = bs_price(S_t, K, T_t, r_t, iv_t, "put", q=q)
        nav_t = K + entry_credit_net - put_price_t
        close_signal = (T_t > 0) and (
            put_price_t <= entry_credit_gross * (1.0 - profit_target_pct))
        rows.append({"nav": nav_t, "put_price": put_price_t,
                     "close_signal": bool(close_signal)})
        idx_list.append(price_series.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx_list))


def covered_call_nav_path(K: float, entry_credit_gross: float, entry_i: int,
                           price_series: pd.Series, iv_series: pd.Series,
                           r_series: pd.Series, *, q: float, dte: int,
                           profit_target_pct: float, cost_pct_of_premium: float
                           ) -> pd.DataFrame:
    """Daily NAV path for a covered call written against shares already
    held. NAV_t = S_t - (call_price_t - entry_credit_net).

    PROVABLY BOUNDED BELOW BY entry_credit_net > 0, same proof shape as
    short_put_nav_path: bs_price's call formula is `S*exp(-qT)*N(d1) -
    K*exp(-rT)*N(d2)`, so `call_price_t <= S_t*exp(-q*T_t) <= S_t` always
    (N(d1)<=1, q>=0), giving `NAV_t >= S_t*(1-exp(-q*T_t)) +
    entry_credit_net >= entry_credit_net`.

    PRECONDITION on this proof, same caveat shape as short_put_nav_path's
    own docstring: requires q>=0, which holds by construction in this
    pipeline (options_vol.DIVIDEND_YIELD_TABLE has no negative entries),
    so this particular bound's precondition is genuinely always satisfied
    here -- unlike short_put_nav_path's r>=0 precondition, which is not.

    Same column/truncation contract as short_put_nav_path.
    """
    entry_credit_net = entry_credit_gross * (1.0 - cost_pct_of_premium)
    last_i = min(entry_i + dte, len(price_series) - 1)
    rows, idx_list = [], []
    for i in range(entry_i, last_i + 1):
        d = i - entry_i
        T_t = (dte - d) / 252.0
        S_t = float(price_series.iloc[i])
        iv_t = float(iv_series.iloc[i])
        r_t = float(r_series.iloc[i])
        call_price_t = bs_price(S_t, K, T_t, r_t, iv_t, "call", q=q)
        nav_t = S_t - (call_price_t - entry_credit_net)
        close_signal = (T_t > 0) and (
            call_price_t <= entry_credit_gross * (1.0 - profit_target_pct))
        rows.append({"nav": nav_t, "call_price": call_price_t,
                     "close_signal": bool(close_signal)})
        idx_list.append(price_series.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx_list))


# Numerical-stability backstop for pmcc_nav_path's floor computation ONLY --
# NOT an economic judgment call the way nav_floor_fraction (a caller-supplied
# risk-tolerance default) is. In a deep enough crash (e.g. 100.0 -> 0.01
# over 130 days, iv=0.60) a deep-OTM LEAPS re-mark can underflow to EXACTLY
# 0.0 in float64 (bs_price(1.56, 75.20, 0.496, 0.04, 0.60, "call") == 0.0,
# not merely "small"). When that coincides with a non-positive
# entry_net_debit -- an ordinary outcome of the same crash, not a rare
# coincidence -- the floor's `max(entry_net_debit, long_cost_gross)` basis
# alone is exactly 0.0, so the clamp would produce nav_t = 0.0 exactly
# rather than a small positive number, and downstream callers (run_pmcc's
# _advance_cycle divides by this NAV as cycle_base with no zero guard)
# require it to be positive. This constant is a THIRD term in that same max(...): tiny enough
# (1e-6) to be economically negligible against any realistic LEAPS/short-
# call premium, large enough that nav_floor_fraction * PMCC_FLOOR_BASIS_EPSILON
# (~1e-8 at the 0.01 production default) never collapses to a float64 zero
# or produces a NaN/inf on division. Its only job is preventing this exact
# degeneracy -- it does not change what the floor means economically.
PMCC_FLOOR_BASIS_EPSILON = 1e-6


def pmcc_nav_path(K_long: float, K_short: float, T_long_at_entry: float,
                   long_cost_gross: float, short_credit_gross: float, entry_i: int,
                   price_series: pd.Series, iv_series: pd.Series, r_series: pd.Series,
                   *, q: float, short_dte: int, roll_trigger_delta: float,
                   profit_target_pct: float, cost_pct_of_premium: float,
                   nav_floor_fraction: float, long_leg_is_fresh_entry: bool
                   ) -> pd.DataFrame:
    """Daily NAV path for ONE short-call sub-cycle of a PMCC (the LEAPS
    long leg persists across many of these; the short leg rolls monthly).

    `long_leg_is_fresh_entry`: pass True ONLY on the first short sub-cycle
    immediately after a real LEAPS purchase -- entry cost is charged on
    the long leg ONLY then. Every later sub-cycle against the SAME
    already-owned LEAPS passes False, so the long leg's own daily re-mark
    carries no fresh transaction cost. `long_cost_gross`/`short_credit_gross`
    are each a FIXED-sign magnitude, unlike the ratio spread's single
    entry_credit which can itself flip sign, so this function uses the
    ordinary `*(1+cost_pct)`/`*(1-cost_pct)` forms directly rather than
    _cost_adjusted.

    NAV_t = long_call_price_t - short_call_price_t, shifted by a constant
    entry-cost offset so NAV_0 equals the cash-inclusive net debit for
    THIS sub-cycle:

        long_cost_component    = long_cost_gross*(1+cost_pct) if long_leg_is_fresh_entry else long_cost_gross
        short_credit_component = short_credit_gross*(1-cost_pct)
        entry_net_debit  = long_cost_component - short_credit_component
        gross_net_debit  = long_cost_gross - short_credit_gross
        cost_offset      = entry_net_debit - gross_net_debit
        nav_t = max(long_call_price_t - short_call_price_t + cost_offset,
                    nav_floor_fraction * max(entry_net_debit, long_cost_gross,
                                              PMCC_FLOOR_BASIS_EPSILON))

    The offset is ADDED, not subtracted -- required for NAV_0 to equal
    entry_net_debit exactly.

    CAVEAT on this offset mechanism itself: adding a
    positive constant to both the numerator and denominator of a
    within-cycle return ratio (`(X_t+offset)/(X_0+offset)`) provably
    DAMPENS the return toward 1.0 in BOTH directions -- it reduces a gain
    on a WINNING sub-cycle (X_t > X_0, the intended, "conservative"
    direction) but by the identical mechanism also shrinks a LOSS on a
    LOSING sub-cycle (X_t < X_0). Re-marking/re-costing the LEAPS every
    sub-cycle is therefore NOT uniformly conservative: differentiating the
    return with respect to `offset` gives an effect of sign(X_0 - X_t),
    which flips between winning and losing sub-cycles. `long_leg_is_fresh_
    entry=False` makes `cost_offset` for a continuing sub-cycle equal to
    just the SHORT leg's own cost component (the LEAPS contributes
    nothing once it is no longer being freshly transacted), which keeps
    this effect small relative to re-costing the LEAPS on every sub-cycle,
    but does not make it directionally safe in either direction. This is
    disclosed as an accepted, small, NOT uniformly conservative
    simplification.

    `entry_net_debit` is NOT guaranteed positive and must NOT be treated
    as a fatal precondition -- an ordinary decline of the underlying
    within one LEAPS holding window can drive it negative through
    completely ordinary bear-market price action. The floor is computed
    from `max(entry_net_debit, long_cost_gross, PMCC_FLOOR_BASIS_EPSILON)`
    specifically so it stays meaningful (strictly positive) even when
    BOTH `entry_net_debit` and `long_cost_gross` degrade at once.

    `long_cost_gross` is NOT always > 0. For a continuing
    (`long_leg_is_fresh_entry=False`) sub-cycle, `long_cost_gross` is fed
    the LEAPS' own LIVE re-mark, not its original entry cost (see
    `run_pmcc`'s `long_mark_now = bs_price(...)` call) -- and a
    deep-enough-OTM call's BSM value genuinely underflows to EXACTLY
    `0.0` in float64 at extreme moneyness (e.g. a 130-day 100.0 -> 0.01
    crash at iv=0.60 fed through run_pmcc;
    `bs_price(1.56, 75.20, 0.496, 0.04, 0.60, "call")` returns exactly
    `0.0`, not merely "small"). When that `0.0` long re-mark coincides
    with a non-positive `entry_net_debit` (the short leg's own credit
    exceeding the LEAPS' now-worthless value -- an ordinary outcome of
    the SAME crash, not a rare coincidence), `max(entry_net_debit,
    long_cost_gross)` alone is exactly `0.0`, so without a third term the
    floor itself would be exactly `0.0` and the clamp would produce
    `nav_t = 0.0` exactly rather than a small positive number -- breaking
    every caller's assumption that this floor can never be non-positive
    (`run_pmcc`'s `_advance_cycle` divides by this NAV as `cycle_base`
    with no zero guard, so a `0.0` NAV here would raise
    `ZeroDivisionError` and discard the whole ticker's curve).

    `PMCC_FLOOR_BASIS_EPSILON` (module-level constant, defined above this
    function) is a genuinely tiny (1e-6) THIRD term added to that same
    `max(...)` purely to close this exact-zero/negative degeneracy. It is
    NOT an economic judgment call the way `nav_floor_fraction` itself is
    (a real, caller-supplied, disclosed risk-tolerance default) -- it is
    a numerical-stability backstop only, small enough to be economically
    negligible against any realistic LEAPS/short-call premium and large
    enough that `nav_floor_fraction * PMCC_FLOOR_BASIS_EPSILON` (the
    worst-case floor -- about 1e-8 at the 0.01 production default) never
    collapses to a float64 zero or produces a NaN/inf on division.

    NOT provably bounded below by a positive constant the way the
    cash-secured shapes are, and there is no no-arbitrage proof here.
    The epsilon term only guarantees the floor itself,
    and therefore `nav_t`, is always strictly > 0 (never exactly `0.0`
    or negative), so a downstream division by `nav_t` can never raise
    `ZeroDivisionError` on this account. `nav_floor_fraction *
    max(entry_net_debit, long_cost_gross, PMCC_FLOOR_BASIS_EPSILON)` is
    still an explicit, disclosed defensive clamp, not a proof -- the
    `clamped` column records every day the clamp actually engaged.

    `roll_trigger_delta` closes the short leg defensively if its own
    delta drifts too deep ITM while T_short_t > 0. `close_signal` is True
    if EITHER the short call hit its own profit target OR its delta
    breached roll_trigger_delta, both gated on T_short_t > 0.

    Returns columns `nav`, `raw_nav` (pre-clamp, for checks),
    `long_price`, `short_price`, `short_delta`, `close_signal`, `clamped`.

    Raises ValueError up front if `nav_floor_fraction <= 0`: the "always
    strictly positive" guarantee above is `floor =
    nav_floor_fraction * max(..., PMCC_FLOOR_BASIS_EPSILON)`, which is
    only actually positive when `nav_floor_fraction` itself is positive
    -- a zero or negative value makes floor collapse to 0.0 or go
    negative regardless of the epsilon term, reopening the same
    ZeroDivisionError/negative-NAV failure modes the epsilon term closes.
    Like the sibling long_roll_dte_floor <= short_dte guard in run_pmcc,
    it is validated explicitly here rather than left to caller
    discipline.
    """
    if nav_floor_fraction <= 0:
        raise ValueError(
            f"pmcc_nav_path: nav_floor_fraction ({nav_floor_fraction}) must be "
            f"strictly positive -- the floor's 'always > 0' guarantee "
            f"(PMCC_FLOOR_BASIS_EPSILON) depends on nav_floor_fraction itself "
            f"being positive; a zero or negative value defeats it")
    long_cost_component = (long_cost_gross * (1.0 + cost_pct_of_premium)
                           if long_leg_is_fresh_entry else long_cost_gross)
    short_credit_component = short_credit_gross * (1.0 - cost_pct_of_premium)
    entry_net_debit = long_cost_component - short_credit_component
    gross_net_debit = long_cost_gross - short_credit_gross
    cost_offset = entry_net_debit - gross_net_debit
    floor = nav_floor_fraction * max(entry_net_debit, long_cost_gross,
                                      PMCC_FLOOR_BASIS_EPSILON)
    last_i = min(entry_i + short_dte, len(price_series) - 1)
    rows, idx_list = [], []
    for i in range(entry_i, last_i + 1):
        d = i - entry_i
        T_short_t = (short_dte - d) / 252.0
        T_long_t = max(T_long_at_entry - d / 252.0, 0.0)
        S_t = float(price_series.iloc[i])
        iv_t = float(iv_series.iloc[i])
        r_t = float(r_series.iloc[i])
        long_price_t = bs_price(S_t, K_long, T_long_t, r_t, iv_t, "call", q=q)
        short_price_t = bs_price(S_t, K_short, T_short_t, r_t, iv_t, "call", q=q)
        raw_nav_t = long_price_t - short_price_t + cost_offset
        clamped = raw_nav_t < floor
        nav_t = max(raw_nav_t, floor)
        short_delta_t = bs_delta(S_t, K_short, T_short_t, r_t, iv_t, "call", q=q)
        close_signal = (T_short_t > 0) and (
            short_price_t <= short_credit_gross * (1.0 - profit_target_pct)
            or short_delta_t >= roll_trigger_delta)
        rows.append({"nav": nav_t, "raw_nav": raw_nav_t, "long_price": long_price_t,
                     "short_price": short_price_t, "short_delta": short_delta_t,
                     "close_signal": bool(close_signal), "clamped": bool(clamped)})
        idx_list.append(price_series.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx_list))


def ratio_spread_nav_path(K_long: float, K_short: float, short_multiple: int,
                           entry_credit_gross: float, entry_i: int,
                           price_series: pd.Series, iv_series: pd.Series,
                           r_series: pd.Series, *, q: float, dte: int,
                           profit_target_pct: float, cost_pct_of_premium: float
                           ) -> pd.DataFrame:
    """Daily NAV path for a put ratio spread, modeled as FULLY
    CASH-SECURED against `short_multiple * K_short`.

    NAV_t = short_multiple*K_short + entry_credit_net - liability_t,
    liability_t = short_multiple*put_price(K_short,t) - put_price(K_long,t).
    `entry_credit_net = _cost_adjusted(entry_credit_gross, cost_pct)` (NOT
    `entry_credit_gross * (1 - cost_pct)`).

    At THIS PIPELINE'S OWN CATALOG DELTAS (long_delta=0.30, short_delta=0.15,
    short_multiple=2), `entry_credit_gross` is NEGATIVE across the entire
    realistic volatility range: the single closer-to-money long put costs
    MORE than `short_multiple` further-OTM short puts combined -- a real
    property of a 1x2 put ratio spread at these delta spacings.

    (1) The bound: as S_t -> 0, `NAV_t -> K_long + entry_credit_net`
    EXACTLY (not `entry_credit_net` alone) -- BOUNDED BELOW BY
    `K_long + entry_credit_net`, a materially WEAKER guarantee than the
    CSP/covered-call proofs (which are bounded by entry_credit_net ALONE).
    The negative-entry_credit_gross property above was checked across the
    volatility range sigma=0.12-0.60, q in {0, 0.013, 0.02} --
    a future retune of these deltas/short_multiple should re-derive this
    bound rather than assume it holds, since it depends on K_long
    dominating a potentially-negative entry_credit_net, not on
    entry_credit_net being positive on its own.

    (2) The cost formula: `entry_credit_gross * (1 - cost_pct_of_premium)`
    computes cost BACKWARDS when entry_credit_gross is negative.
    `_cost_adjusted` is correct for both signs.

    Does NOT model physical assignment of the naked excess short legs
    into a stock position -- cash-settles at expiry via bs_price's own
    T=0 intrinsic collapse, same as every other position type here.

    Same column/truncation contract as short_put_nav_path, with `liability`
    replacing `put_price`.
    """
    entry_credit_net = _cost_adjusted(entry_credit_gross, cost_pct_of_premium)
    collateral = short_multiple * K_short
    last_i = min(entry_i + dte, len(price_series) - 1)
    rows, idx_list = [], []
    for i in range(entry_i, last_i + 1):
        d = i - entry_i
        T_t = (dte - d) / 252.0
        S_t = float(price_series.iloc[i])
        iv_t = float(iv_series.iloc[i])
        r_t = float(r_series.iloc[i])
        long_price_t = bs_price(S_t, K_long, T_t, r_t, iv_t, "put", q=q)
        short_price_t = bs_price(S_t, K_short, T_t, r_t, iv_t, "put", q=q)
        liability_t = short_multiple * short_price_t - long_price_t
        nav_t = collateral + entry_credit_net - liability_t
        # Profit-target threshold, defined as a MOVE from liability_0
        # (which equals entry_credit_gross exactly, by construction) --
        # NOT `entry_credit_gross * (1 - profit_target_pct)`, which for a
        # negative entry_credit_gross (as at the catalog deltas above) is
        # already satisfied on the first day of every cycle. The form
        # below reduces to `entry_credit_gross * (1 - profit_target_pct)`
        # when entry_credit_gross > 0 and is non-triggering at t=0 for
        # either sign.
        close_signal = (T_t > 0) and (
            liability_t <= entry_credit_gross
                          - profit_target_pct * abs(entry_credit_gross))
        rows.append({"nav": nav_t, "liability": liability_t,
                     "close_signal": bool(close_signal)})
        idx_list.append(price_series.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx_list))
