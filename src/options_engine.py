"""
options_engine.py
------------------
Black-Scholes pricing + iron condor construction/simulation for the
0-2 DTE iron-condor walk-forward backtest.

CONVENTIONS:
  - All times (T) are in YEARS.
  - TRADING_HOURS_PER_DAY = 6.5, TRADING_DAYS = 252.
  - R-multiple convention (one R = the defined maximum loss):
      max_loss   = (wing_width - entry_credit) * 100   per contract = 1R
      pnl_dollars = pnl_per_share * 100
      r_multiple  = pnl_dollars / max_loss

"""

import math

TRADING_DAYS = 252
TRADING_HOURS_PER_DAY = 6.5

# Baseline strategy parameters (mid-points of documented ranges; tunable).
BASELINE_PARAMS = {
    "short_delta_target": 0.125,   # 10-15 delta
    "pt_pct": 0.45,                 # 40-50% profit target
    "stop_pct": 1.75,               # 150-200% stop
    "wing_width_pct": 0.01,         # 1% of underlying
    "entry_hours_remaining": 3.0,
    "r": 0.04,                       # risk-free rate; negligible for 0DTE
}


# ── Black-Scholes core ──────────────────────────────────────────────────────────

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (much faster than scipy.stats.norm.cdf
    for scalar inputs, which matters here -- this is called millions of times
    by Monte Carlo / sensitivity sweep callers)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call", *, q: float = 0.0) -> float:
    """
    European Black-Scholes-Merton price with continuous dividend yield `q`.
    At T <= 0, collapses to intrinsic value (unaffected by q).

    `q` is keyword-only, default 0.0 -- a call site that never passes it
    (build_iron_condor, simulate_trade) is byte-identical to the
    no-dividend-yield formula.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call", *, q: float = 0.0) -> float:
    """
    Black-Scholes-Merton delta with continuous dividend yield `q`. At
    T <= 0, returns the (discontinuous) intrinsic-value delta (unaffected
    by q). `q` is keyword-only, default 0.0 -- byte-identical to the
    no-dividend-yield formula for call sites that omit it.
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "call":
        return math.exp(-q * T) * _norm_cdf(d1)
    return math.exp(-q * T) * (_norm_cdf(d1) - 1.0)


# ── Strike Selection ─────────────────────────────────────────────────────────────

def solve_strike_for_delta(S: float, T: float, r: float, sigma: float,
                            target_delta: float, option_type: str = "call",
                            tol: float = 1e-6, max_iter: int = 100, *,
                            q: float = 0.0) -> float:
    """
    Solve for the strike K such that bs_delta(S, K, T, r, sigma, option_type,
    q=q) == target_delta, via bisection.

    Call delta is monotonically decreasing in K (search K in [0.05*S, 3S] --
    widened from [S, 3S], which could only reach K >= S, i.e.
    delta roughly in (0, 0.5-0.6]; a PMCC long LEAPS leg needs a DEEP-ITM
    call, target delta ~0.75-0.85, requiring K well below S. Monotonicity
    in K still holds over the wider range for a fixed (S,T,r,sigma,q), so
    no other bisection logic changes; the widened bracket is a superset
    and still resolves every narrow-bracket OTM-call target to the SAME
    delta within this function's own tolerance (tol=1e-6 on delta, NOT
    on K), see test_solve_strike_for_delta_q_default_zero_still_resolves_
    pre_widen_target_delta. It is NOT byte-identical on the strike K
    itself: bisection started from a wider [lo, hi] visits a different
    sequence of midpoints, so it can converge to a slightly different
    floating-point K for a target that was already reachable in the old
    bracket too (a tiny drift in K, on the order of 1e-5 to 1e-7, for
    call sites such as build_iron_condor -- immaterial for realistic
    trading, but not byte-identical).
    Put delta is monotonically increasing in K, i.e. less negative as K
    rises (search K in [0.1S, S]) -- unchanged.

    target_delta sign convention: pass positive deltas for calls (e.g. 0.125
    for a 12.5-delta short call) and negative deltas for puts (e.g. -0.125).
    `q` is keyword-only, threaded through to bs_delta; default 0.0 means
    q itself introduces no change to the delta formula for call sites
    that omit it. It does NOT by itself preserve the exact strike K those
    call sites resolved to -- the widened bracket (independent of q) can
    still shift K by the small amount described above.
    """
    if option_type == "call":
        lo, hi = S * 0.05, S * 3.0
    else:
        lo, hi = S * 0.1, S

    def f(K):
        return bs_delta(S, K, T, r, sigma, option_type, q=q) - target_delta

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"solve_strike_for_delta: target_delta={target_delta} not bracketed "
            f"by K in [{lo}, {hi}] (f_lo={f_lo}, f_hi={f_hi})"
        )

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return (lo + hi) / 2.0


# ── Iron Condor Construction ───────────────────────────────────────────────────

def build_iron_condor(S: float, iv: float, dte: int = 0,
                       short_delta_target: float = BASELINE_PARAMS["short_delta_target"],
                       wing_width_pct: float = BASELINE_PARAMS["wing_width_pct"],
                       r: float = BASELINE_PARAMS["r"],
                       entry_hours_remaining: float = BASELINE_PARAMS["entry_hours_remaining"]) -> dict:
    """
    Construct a 4-leg iron condor: short strikes at +/- short_delta_target,
    long strikes wing_width_pct * S further out.

    Returns a dict with the 4 strikes, entry_credit, max_loss (1R in dollars
    per contract), wing_width, T_entry, and the params needed to revalue the
    position later (iv, r, entry_hours_remaining).
    """
    T_entry = (dte * TRADING_HOURS_PER_DAY + entry_hours_remaining) / (TRADING_HOURS_PER_DAY * TRADING_DAYS)

    short_put_K = solve_strike_for_delta(S, T_entry, r, iv, -short_delta_target, "put")
    short_call_K = solve_strike_for_delta(S, T_entry, r, iv, short_delta_target, "call")

    wing_width = wing_width_pct * S
    long_put_K = short_put_K - wing_width
    long_call_K = short_call_K + wing_width

    put_credit = (bs_price(S, short_put_K, T_entry, r, iv, "put")
                   - bs_price(S, long_put_K, T_entry, r, iv, "put"))
    call_credit = (bs_price(S, short_call_K, T_entry, r, iv, "call")
                    - bs_price(S, long_call_K, T_entry, r, iv, "call"))
    entry_credit = put_credit + call_credit

    max_loss = (wing_width - entry_credit) * 100

    return {
        "short_put_K": short_put_K,
        "long_put_K": long_put_K,
        "short_call_K": short_call_K,
        "long_call_K": long_call_K,
        "entry_credit": entry_credit,
        "max_loss": max_loss,
        "wing_width": wing_width,
        "T_entry": T_entry,
        "iv": iv,
        "dte": dte,
        "entry_hours_remaining": entry_hours_remaining,
        "r": r,
    }


# ── Trade Simulation ────────────────────────────────────────────────────────────

def _ic_value(ic: dict, S: float, iv: float, T: float) -> float:
    """Cost to close (buy back) the iron condor at underlying price S, IV iv, time T."""
    r = ic["r"]
    put_value = (bs_price(S, ic["short_put_K"], T, r, iv, "put")
                  - bs_price(S, ic["long_put_K"], T, r, iv, "put"))
    call_value = (bs_price(S, ic["short_call_K"], T, r, iv, "call")
                   - bs_price(S, ic["long_call_K"], T, r, iv, "call"))
    return put_value + call_value


def simulate_trade(ic: dict, day_ohlc: dict, iv_exit: float,
                    pt_pct: float = BASELINE_PARAMS["pt_pct"],
                    stop_pct: float = BASELINE_PARAMS["stop_pct"]) -> dict:
    """
    Simulate one day's outcome for an iron condor built by build_iron_condor().

    Exit logic:
      1. Revalue the condor at the day's High and Low (using iv_exit and
         T_exit = half the entry's remaining time) to check whether the
         profit target or stop would have been crossed intraday.
      2. If both PT and stop are reachable within the day's range, "stop"
         wins (conservative -- assumes the adverse path happens first).
      3. Otherwise, settle at expiry (T=0) using the day's Close ->
         intrinsic value only.

    day_ohlc: dict with "High", "Low", "Close" (and optionally "Open",
    unused here -- Open is the entry price baked into `ic`).

    Returns dict: exit_reason ("profit_target"|"stop"|"expiry"), pnl
    (per-share credit units), pnl_dollars (pnl * 100), r_multiple.
    """
    entry_credit = ic["entry_credit"]
    max_loss = ic["max_loss"]
    T_exit = (ic["entry_hours_remaining"] / 2) / (TRADING_HOURS_PER_DAY * TRADING_DAYS)

    value_high = _ic_value(ic, day_ohlc["High"], iv_exit, T_exit)
    value_low = _ic_value(ic, day_ohlc["Low"], iv_exit, T_exit)

    pnl_high = entry_credit - value_high
    pnl_low = entry_credit - value_low

    pt_threshold = entry_credit * pt_pct
    stop_threshold = -entry_credit * stop_pct

    pt_hit = pnl_high >= pt_threshold or pnl_low >= pt_threshold
    stop_hit = pnl_high <= stop_threshold or pnl_low <= stop_threshold

    if stop_hit:
        exit_reason = "stop"
        pnl = stop_threshold
    elif pt_hit:
        exit_reason = "profit_target"
        pnl = pt_threshold
    else:
        exit_reason = "expiry"
        value_close = _ic_value(ic, day_ohlc["Close"], iv_exit, 0.0)
        pnl = entry_credit - value_close

    pnl_dollars = pnl * 100
    r_multiple = pnl_dollars / max_loss if max_loss > 0 else 0.0

    return {
        "exit_reason": exit_reason,
        "pnl": pnl,
        "pnl_dollars": pnl_dollars,
        "r_multiple": r_multiple,
    }
