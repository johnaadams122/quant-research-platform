"""Benchmark-plus-overlay account builder (beta-neutral short overlay).

Pure functions only: no network, no ledger, no clock. Built for
``ipo_long_run_underperformance_fade_beta_neutral`` -- the beta-neutral variant
of the naked-short ``ipo_long_run_underperformance_fade`` family. A naked short
does not fit the ABI-1 account for two structural reasons (a never-exited short
can have a net return below -100%, and a naked short's excess over SPY
measures the account's negative market beta rather than IPO's RELATIVE
underperformance).

Why this is NOT "long beta*SPY / short IPO"
-------------------------------------------
ABI-1 (``mining.account_relative``) scores the ACCOUNT's daily log excess over
SPY. A market-neutral account (long beta*SPY, short IPO) earns roughly zero
market return, so its excess over SPY is roughly ``-SPY`` -- the same beta
confound as the naked short, in a new form. The account built here instead
holds SPY ITSELF (the benchmark, equity fully invested) PLUS a beta-neutral
overlay: short IPO with notional ``x`` and an additional long SPY with
notional ``beta_hat * x``, both as fractions of account equity.

    account return  = r_spy + overlay
    excess over SPY = overlay = x * (beta_hat * r_spy - r_ipo)   (to first order)

The account's beta to SPY is 1 and the excess series that ABI-1 scores has a
beta of roughly ``x * (beta_hat - beta_true)`` -- close to zero -- so what is
scored is IPO's relative behaviour versus its own hedge, not market direction.

Fixed design constants (chosen BEFORE any score was seen)
------------------------------------------------
* ``BETA_WINDOW_SESSIONS`` = 252 -- trailing OLS window, daily simple
  close-to-close returns, IPO on SPY, WITH intercept (the slope is
  ``cov/var``).
* Hedge ratio for the overlay opened on session ``t`` uses only returns dated
  BEFORE ``t`` (the estimate is available at the prior close; fills are at
  ``t``'s Open, the engine's own next-open convention). It is re-estimated and
  the overlay rebalanced on the FIRST trading session of each calendar month.
* Warm-up: no overlay until a full window of prior returns exists (session
  index ``window + 1``); until then the account is plain SPY and the excess is
  exactly zero.
* ``OVERLAY_NOTIONAL`` = 0.25 of account equity. Chosen for Reg-T feasibility,
  NOT for the score, and the score is NOT invariant to it: the excess series
  scales linearly with x only to first order, and ABI-1 scores the LOG excess,
  whose curvature term (about x * sigma / 2 per session) penalizes a larger
  overlay. A larger x is therefore the CONSERVATIVE direction. x also
  decides the absolute-drawdown risk refusal, which matters only for a
  family that clears the DSR bar. Under Reg-T initial margin the
  overlay is feasible when ``x * (beta + 1) <= 1`` (50% of long value plus 50%
  of short value against equity of 1 with a fully invested SPY core), i.e.
  ``x <= 1 / (beta + 1)``.
  ``x = 0.25`` stays inside that for every hedge ratio up to 3 (measured
  rolling betas of IPO on SPY are far below that); ``x = 0.5`` would only be
  feasible up to a hedge ratio of 1. ``MAX_HEDGE_RATIO`` = 3 turns that limit
  into a refusal: a trailing estimate outside ``[0, 3]`` makes the builder
  raise (never clip -- clipping would be a silent tuning knob -- and never
  flip the hedge to a short SPY), because the account would no longer be the
  registered design.

Accounting conventions (deliberately mirrors ``backtest_engine``)
-----------------------------------------------------------------
* Overlay legs are opened at the OPEN of the rebalance session and closed at
  the OPEN of the next one; the final segment is closed at the last bar's
  Close (the engine's ``end_of_data`` path). A rebalance that lands on the
  very last bar closes the old segment and opens nothing (a one-bar segment
  would be opened and closed inside a single session).
* Between rebalances SHARE COUNTS are fixed (leg weights drift). Daily overlay
  P&L is ``n_spy * dP_spy - n_ipo * dP_ipo`` from the actual price path, never
  from re-normalised weights.
* Account daily return is ``core_t + overlay_dollars_t / E_{t-1}`` where
  ``core_t`` is the benchmark's OWN cost-adjusted daily return (the identical
  ``benchmark_relative.cost_adjusted_benchmark_daily_returns`` series the ABI-1
  evaluator subtracts, so the excess is exactly zero when no overlay is on)
  and ``E`` is the account's pre-tax equity. The core position is implicitly
  reinvested daily (overlay P&L is added to equity and held in SPY), the same
  per-session-return-on-current-equity convention every ``strategy_daily`` in
  this pipeline uses.
* Costs: every monthly segment is charged a full round trip on BOTH overlay
  legs, ``x * ipo_cost_bps + x * beta_hat * spy_cost_bps`` (bps of overlay
  notional, as fractions of the sizing equity), charged on the segment's exit
  day. This deliberately over-counts turnover (a continuing hedge could be
  rebalanced only by the change in its size).

LIMITATION (disclosed, not fixable inside this design)
-------------------------------------------------------
The hedge neutralizes SPY beta ONLY. The IPO ETF also carries size, growth and
momentum tilts relative to the S&P 500 that a single-factor hedge leaves in
place, so the scored excess is IPO-versus-its-SPY-hedge, not IPO-versus-a-
style-matched-control (Ritter's own control sample was matched on size and
industry). An apparent edge here could be a style premium rather than an
IPO-specific effect; a survivor would need a multi-factor or matched-control
re-check before being believed.

NOT MODELLED (disclosed; each of these FLATTERS the strategy)
-------------------------------------------------------------
Short-borrow fees on the IPO leg, margin financing of the net long
(``beta_hat * x`` of SPY, which a retail account cannot fund from the short
proceeds), Reg-T margin calls, and wash-sale disallowance of overlay losses.
Short-rebate interest on the proceeds is income, so omitting it is
conservative. The pipeline has no margin-rate or borrow-fee source; it does
carry a T-bill rate series (``account_relative.daily_cash_rate``), which is
only a LOWER BOUND on the financing rate and was deliberately NOT charged
(inventing a number is not allowed). Charging even that floor on the hedge
notional would lower the score, so any apparent passing result needs a
cost-sensitivity re-check before being believed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import account_relative
from .backtest_engine import SignalResult, run_backtest
from .benchmark_relative import (cost_adjusted_benchmark_daily_returns,
                                 mark_to_market_daily_returns, ols_alpha_beta)

BETA_WINDOW_SESSIONS = 252
OVERLAY_NOTIONAL = 0.25
# The hedge-ratio cap that makes the notional Reg-T feasible: x * (beta + 1) <= 1
# holds for every beta up to 3 at x = 0.25. An estimate outside [0, cap] means
# the IPO/SPY relationship broke (or the data did): the builder REFUSES rather
# than clipping (clipping would be a silent tuning knob) or flipping the hedge.
MAX_HEDGE_RATIO = 3.0
# Pre-registered (before any score): the scored excess series'
# OLS beta to SPY, per unit of overlay notional, must stay below a quarter of
# the unhedged short leg's own beta (~1). A naked short of IPO scores about
# -(beta_ipo + 1) on the same measure (around -2 for a beta near 1), i.e.
# roughly ten times outside it, so the bound separates the two designs cleanly without
# being tuned to pass. A breach is a finding about the design, not something
# to be relaxed.
MAX_RESIDUAL_BETA_PER_NOTIONAL = 0.25


def rebalance_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions of the first trading session of each calendar month
    in ``index`` (position 0 counts as a month start)."""
    idx = pd.DatetimeIndex(index)
    months = idx.to_period("M")
    starts = np.ones(len(idx), dtype=bool)
    if len(idx) > 1:
        starts[1:] = months[1:] != months[:-1]
    return np.flatnonzero(starts)


def causal_hedge_ratios(ipo_close: pd.Series, spy_close: pd.Series, *,
                        window: int = BETA_WINDOW_SESSIONS) -> pd.Series:
    """Hedge ratio available at the start of each session ``t``: the OLS
    slope of IPO daily simple returns on SPY daily simple returns over the
    ``window`` returns ENDING AT ``t - 1``. NaN until a full window of prior
    returns exists. Strictly causal: value ``t`` never reads any price dated
    ``t`` or later."""
    r_ipo = pd.Series(ipo_close).astype(float).pct_change()
    r_spy = pd.Series(spy_close).astype(float).pct_change()
    slope = r_ipo.rolling(window).cov(r_spy) / r_spy.rolling(window).var()
    return slope.shift(1).rename("hedge_ratio")


def scored_excess_spy_beta(account_daily: pd.Series,
                           benchmark_daily: pd.Series) -> dict:
    """OLS regression of the series ABI-1 actually scores -- the account's
    daily LOG excess over the benchmark (``account_relative.log_excess``) --
    on the benchmark's daily returns. Returns ``ols_alpha_beta``'s dict; its
    ``beta`` is the residual market exposure of what gets scored."""
    excess = account_relative.log_excess(account_daily, benchmark_daily)
    return ols_alpha_beta(excess, benchmark_daily)


def naked_short_account_daily(ipo_frame: pd.DataFrame, *,
                              cost_bps: float = 0.0) -> pd.Series:
    """PRE-tax daily account return of the NAKED short that
    ``ipo_long_run_underperformance_fade`` scores (short IPO from the first
    reachable bar, never exited; idle sessions earn the sweep rate) -- the
    exact construction ``scripts/run_ipo_fade_abi1.py`` builds, rebuilt here
    only so the beta confound can be reported next to the beta-neutral
    account. Report only: this series is never scored by the beta-neutral
    family."""
    def _always_short(frame, i):
        return SignalResult(entry=True, exit=False)

    trades_df, in_market = run_backtest(
        ipo_frame, _always_short, cost_bps=cost_bps, tax_haircut_rate=0.0,
        direction="short")
    strategy = mark_to_market_daily_returns(
        ipo_frame, trades_df, cost_bps=cost_bps, direction="short")
    cash = account_relative.sweep_cash_daily(strategy.index)
    return account_relative.account_daily_returns(
        strategy_daily=strategy, in_market=in_market, cash_daily=cash)


_TRADES_COLUMNS = ("entry_date", "exit_date", "entry_px", "exit_px",
                   "spy_entry_px", "spy_exit_px", "held", "reason",
                   "hedge_ratio", "gross_ret", "net_ret")


def _validated_frame(frame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name}: expected a non-empty OHLC DataFrame")
    for column in ("Open", "Close"):
        if column not in frame.columns:
            raise ValueError(f"{name}: frame is missing {column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: frame needs a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{name}: index must be sorted and unique")
    values = frame[["Open", "Close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{name}: Open/Close must be finite and positive")
    return frame


def _validated_scalar(value, name: str, *, low: float, high: float,
                      inclusive_low: bool) -> float:
    value = float(value)
    ok = np.isfinite(value) and (value >= low if inclusive_low else value > low)
    if not ok or value > high:
        raise ValueError(f"{name} must be finite and within "
                         f"{'[' if inclusive_low else '('}{low}, {high}]; "
                         f"got {value}")
    return value


def build_beta_neutral_overlay_curve(ipo_frame: pd.DataFrame,
                                     spy_frame: pd.DataFrame, *,
                                     ipo_cost_bps: float,
                                     spy_cost_bps: float,
                                     spy_entry_cost_bps: float,
                                     overlay_notional: float = OVERLAY_NOTIONAL,
                                     beta_window: int = BETA_WINDOW_SESSIONS
                                     ) -> dict:
    """Account curve for the benchmark-plus-overlay account described in the
    module docstring, in the ``{"strategy_daily", "in_market", "trades_df"}``
    shape ``abi1_driver.evaluate_family`` accepts from any builder.

    ``spy_frame`` is the benchmark's FULL history (a superset of the IPO
    calendar); the account lives on ``ipo_frame``'s calendar and the SPY
    sessions inside that span must match it exactly (a mismatch fails closed
    rather than being silently patched). ``ipo_cost_bps`` / ``spy_cost_bps``
    are ROUND-TRIP costs in bps of notional (``execution_costs_for``'s
    ``cost_bps``); ``spy_entry_cost_bps`` is the benchmark's own already-halved
    entry cost, the identical figure the ABI-1 evaluator charges the
    benchmark, so the account's SPY core equals the benchmark series exactly.

    ``trades_df`` has one row per overlay segment; ``net_ret`` is the
    overlay's net return ON ACCOUNT EQUITY at the segment's sizing point
    (after its round-trip cost), bounded far from -100% because a segment is
    at most one calendar month long and only a fraction ``x`` of equity is
    short.
    """
    ipo = _validated_frame(ipo_frame, "ipo_frame")
    spy_full = _validated_frame(spy_frame, "spy_frame")
    x = _validated_scalar(overlay_notional, "overlay_notional",
                          low=0.0, high=1.0, inclusive_low=False)
    ipo_cost = _validated_scalar(ipo_cost_bps, "ipo_cost_bps", low=0.0,
                                 high=1e4, inclusive_low=True) / 1e4
    spy_cost = _validated_scalar(spy_cost_bps, "spy_cost_bps", low=0.0,
                                 high=1e4, inclusive_low=True) / 1e4
    entry_cost = _validated_scalar(spy_entry_cost_bps, "spy_entry_cost_bps",
                                   low=0.0, high=1e4, inclusive_low=True)
    if isinstance(beta_window, bool) or not isinstance(beta_window, int) \
            or beta_window < 3:
        raise ValueError(f"beta_window must be an int >= 3; got {beta_window!r}")

    idx = ipo.index
    n = len(idx)
    spy = spy_full.loc[idx[0]:idx[-1]]
    if not spy.index.equals(idx):
        raise ValueError(
            "the SPY sessions inside the IPO span must match the IPO calendar "
            "exactly; refusing to patch a calendar mismatch between two "
            "same-exchange instruments")

    core_series = cost_adjusted_benchmark_daily_returns(
        spy_full, entry_cost_bps=entry_cost).reindex(idx)
    core = core_series.to_numpy(dtype=float)
    if not np.isfinite(core).all():
        raise ValueError("benchmark core returns must be finite on every "
                         "IPO session")

    hedge_series = causal_hedge_ratios(ipo["Close"], spy["Close"],
                                       window=beta_window)
    hedge = hedge_series.to_numpy(dtype=float)
    entry_set = {int(p) for p in rebalance_positions(idx) if p > beta_window}
    # A rebalance on the very last bar closes the old segment and opens
    # nothing, so its hedge ratio is never APPLIED and is not checked.
    open_set = {p for p in entry_set if p < n - 1}
    for p in sorted(open_set):
        if not np.isfinite(hedge[p]):
            raise ValueError(
                f"hedge ratio is not finite at {idx[p].date()} even though a "
                "full window of prior returns exists (zero-variance SPY "
                "window?); refusing to skip a rebalance silently")
        if not 0.0 <= hedge[p] <= MAX_HEDGE_RATIO:
            raise ValueError(
                f"hedge ratio {hedge[p]:.4f} at {idx[p].date()} is outside "
                f"[0, {MAX_HEDGE_RATIO}]: the registered design (short IPO + "
                "a LONG SPY hedge that fits Reg-T) no longer describes this "
                "account; refusing to clip or flip the hedge")

    o_ipo = ipo["Open"].to_numpy(dtype=float)
    c_ipo = ipo["Close"].to_numpy(dtype=float)
    o_spy = spy["Open"].to_numpy(dtype=float)
    c_spy = spy["Close"].to_numpy(dtype=float)

    strategy = np.empty(n, dtype=float)
    trades = []
    equity = 1.0
    seg = None

    def _finalize(seg, exit_pos, exit_ipo_px, exit_spy_px, reason):
        trades.append({
            "entry_date": idx[seg["entry"]], "exit_date": idx[exit_pos],
            "entry_px": seg["ipo_entry_px"], "exit_px": float(exit_ipo_px),
            "spy_entry_px": seg["spy_entry_px"],
            "spy_exit_px": float(exit_spy_px),
            "held": int(exit_pos - seg["entry"]), "reason": reason,
            "hedge_ratio": seg["beta"],
            "gross_ret": (seg["pnl"] + seg["cost"]) / seg["size"],
            "net_ret": seg["pnl"] / seg["size"],
        })

    for t in range(n):
        e_prev = equity
        if not np.isfinite(e_prev) or e_prev <= 0.0:
            raise ValueError(f"account equity is {e_prev} before "
                             f"{idx[t].date()}; the overlay wiped the account")
        overlay = 0.0
        if seg is not None and t in entry_set:
            # rebalance: the old legs are closed at this session's Open
            step = (seg["n_spy"] * (o_spy[t] - c_spy[t - 1])
                    - seg["n_ipo"] * (o_ipo[t] - c_ipo[t - 1]) - seg["cost"])
            overlay += step
            seg["pnl"] += step
            _finalize(seg, t, o_ipo[t], o_spy[t], "rebalance")
            seg = None
        if t in open_set:
            beta = float(hedge[t])
            size = e_prev  # equity at the prior close: known at decision time
            seg = {
                "entry": t, "beta": beta, "size": size,
                "n_ipo": x * size / o_ipo[t],
                "n_spy": beta * x * size / o_spy[t],
                "cost": size * x * (ipo_cost + beta * spy_cost),
                "ipo_entry_px": float(o_ipo[t]),
                "spy_entry_px": float(o_spy[t]),
                "pnl": 0.0,
            }
            step = (seg["n_spy"] * (c_spy[t] - o_spy[t])
                    - seg["n_ipo"] * (c_ipo[t] - o_ipo[t]))
            overlay += step
            seg["pnl"] += step
        elif seg is not None:
            step = (seg["n_spy"] * (c_spy[t] - c_spy[t - 1])
                    - seg["n_ipo"] * (c_ipo[t] - c_ipo[t - 1]))
            overlay += step
            seg["pnl"] += step
        if seg is not None and t == n - 1:
            # engine's end_of_data path: closed at the last bar's Close
            overlay -= seg["cost"]
            seg["pnl"] -= seg["cost"]
            _finalize(seg, t, c_ipo[t], c_spy[t], "end_of_data")
            seg = None
        daily = core[t] + overlay / e_prev
        if not np.isfinite(daily) or daily <= -1.0:
            raise ValueError(f"account return {daily} at {idx[t].date()} is "
                             "not a valid simple return")
        strategy[t] = daily
        equity = e_prev * (1.0 + daily)

    if trades:
        trades_df = pd.DataFrame(trades, columns=list(_TRADES_COLUMNS))
    else:
        trades_df = pd.DataFrame({
            "entry_date": pd.Series(dtype="datetime64[ns]"),
            "exit_date": pd.Series(dtype="datetime64[ns]"),
            "entry_px": pd.Series(dtype="float64"),
            "exit_px": pd.Series(dtype="float64"),
            "spy_entry_px": pd.Series(dtype="float64"),
            "spy_exit_px": pd.Series(dtype="float64"),
            "held": pd.Series(dtype="int64"),
            "reason": pd.Series(dtype="object"),
            "hedge_ratio": pd.Series(dtype="float64"),
            "gross_ret": pd.Series(dtype="float64"),
            "net_ret": pd.Series(dtype="float64"),
        })
    return {
        "strategy_daily": pd.Series(strategy, index=idx,
                                    name="beta_neutral_overlay_account"),
        "in_market": pd.Series(True, index=idx, name="in_market"),
        "trades_df": trades_df,
        # ABI-1 measures this account's risk layer on the
        # overlay's own excess over SPY (`abi1_driver.evaluate_family` reads
        # this key), not on the whole SPY-core account.
        "account_kind": account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY,
        "core_daily": core_series.rename("core"),
        "hedge_ratio": hedge_series,
    }
