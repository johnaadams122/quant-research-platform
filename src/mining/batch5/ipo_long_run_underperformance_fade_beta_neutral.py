"""Batch5 candidate: ipo_long_run_underperformance_fade_beta_neutral (bespoke
driver; run only via scripts/run_ipo_fade_beta_neutral_abi1.py, never the
generic scripts/run_abi1_mining.py dispatch -- see "Dispatch safety").

The BETA-NEUTRAL test of the same thesis as
``ipo_long_run_underperformance_fade`` (Ritter, J.R. (1991), "The Long-Run
Performance of Initial Public Offerings," Journal of Finance, 46(1), 3-27:
newly public companies underperform the broad market over the following one
to three years). The naked-short sibling cannot answer that question (see
that module's and ``scripts/run_ipo_fade_abi1.py``'s docstrings): a
never-exited short can reach a net return below -100%, and ABI-1's daily
excess over SPY, applied to a NAKED short of a market-correlated
instrument, measures the account's negative market beta rather than IPO's
RELATIVE underperformance. This module is the beta-neutral construction;
the naked-short module is the simple primitive beside it.

What is built
-------------
A benchmark-plus-overlay account (full derivation, fixed constants and every
accounting convention in ``mining.beta_neutral_overlay``): the account holds
SPY itself (fully invested) PLUS a beta-neutral overlay -- short IPO with
notional ``x`` of equity and an additional long SPY of ``beta_hat * x`` --
where ``beta_hat`` is the trailing 252-session OLS slope of IPO daily returns
on SPY's, estimated from data ending the session BEFORE it is applied and
refreshed on the first trading session of each calendar month. Account excess
over SPY is then the overlay alone; its beta to SPY is near zero, so ABI-1
scores IPO's behaviour relative to its own hedge instead of market direction.
A pure "long beta*SPY / short IPO" account was deliberately NOT built: its
excess over SPY would be roughly -SPY, re-importing the same confound.

Both problems of the naked short are removed structurally: the overlay is
re-sized every month, so each ``trades_df`` segment is at most one calendar
month long and bounded far from -100% (so ``netted_tax_factors`` does not fail
closed); and the scored series is beta-neutral by construction (the tests
check it on synthetic data, and the driver reports the scored excess's beta
to SPY on every run).

Dispatch safety
---------------
``SHAPE = "single_inverse_product"`` is a contract LABEL only -- none of the
six registered shapes describes a benchmark-plus-overlay account, and adding a
seventh would touch shared dispatch code. ``TRADED`` lists only IPO because
single shapes require exactly one traded symbol; SPY (core, hedge and
benchmark) is ``BENCHMARK``. ``make_signal_fn`` exists only to satisfy that
contract check and ALWAYS raises: the generic dispatch would otherwise score a
plain long IPO position (no per-bar signal exists for this family). The
generic dispatch turns that error into a clean ``compute_error`` skip. This
family is also named in no accepted-scope file (pinned by test).

LIMITATION -- SPY beta only
---------------------------
The hedge neutralizes SPY beta only; the IPO ETF's size/growth/momentum tilts
versus the S&P 500 stay unhedged, so an apparent edge could be a style premium
rather than an IPO-specific effect (details in ``mining.beta_neutral_overlay``).

NOT MODELLED -- each of these FLATTERS the strategy
---------------------------------------------------
Short-borrow fees on IPO, margin financing of the net long (``beta_hat * x`` of
SPY), Reg-T margin calls and wash-sale disallowance of overlay losses.
Short-rebate interest is income, so omitting it is conservative. The pipeline
has no margin-rate or borrow-fee source; its T-bill series is only a lower
bound on financing cost and was deliberately not charged by design
(details in ``mining.beta_neutral_overlay``), so any apparent survivor needs a
cost-sensitivity re-check before it is believed.

TAX APPROXIMATION -- direction NOT uniformly conservative
---------------------------------------------------------
``netted_tax_factors`` sizes each realized gain from an entry-equity identity
that assumes the account grows only by the trade; here the SPY core also
moves, so the modelled dollar gain is off by the ratio of the account's growth
to the overlay's over each one-month segment. Where that ratio exceeds 1,
gains are over-taxed and losses over-sheltered; below 1, the reverse. The
driver reports the measured ratio range and a HEURISTIC sensitivity (not a
rigorous bound -- the loss carryforward makes the true effect nonlinear) of
the scored daily-excess Sharpe to it: the whole tax effect scaled by the
ratio's worst deviation from 1. It never gates on it.

HEDGE-RATIO RANGE
-----------------
The builder refuses (never clips) a hedge ratio outside ``[0,
mining.beta_neutral_overlay.MAX_HEDGE_RATIO]``: a negative estimate would flip
the hedge to a short SPY and one above the cap breaks the Reg-T feasibility
argument -- either way the account would no longer be the registered design.
"""
from __future__ import annotations

from mining import beta_neutral_overlay as _overlay

FAMILY_ID = "ipo_long_run_underperformance_fade_beta_neutral"
SHAPE = "single_inverse_product"   # contract label only -- see "Dispatch safety"
TRADED = ["IPO"]
SIGNAL: list = []
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY (scripts/run_abi1_mining.py's
# MARKET_BENCHMARK); here SPY is also the account's core and hedge leg.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" field for
# this family_id (the driver's parameter-drift check compares them; a test
# ties the two numeric values to mining.beta_neutral_overlay's constants).
PARAMETERS = {
    "position": "beta_neutral_short_overlay_on_benchmark",
    "hedge_leg": "SPY",
    "beta_window_sessions": 252,
    "rebalance": "monthly_first_session",
    "overlay_notional_fraction": 0.25,
    "benchmark": "SPY",
}
# The overlay is re-sized monthly, so a full-history run has one segment per
# calendar month from the first month start with 252 prior returns onward.
# A conservative floor well below that count on IPO's listed history (a data
# sanity check, not a tuned value); the count can only grow as history accrues.
EXPECTED_MIN_TRADES = 100


def make_signal_fn(*args, **kwargs):
    """Always raises -- this family has no per-bar signal. See the module
    docstring's "Dispatch safety" section; the only legitimate entry point is
    ``scripts/run_ipo_fade_beta_neutral_abi1.py`` calling ``build_curve``."""
    raise RuntimeError(
        f"{FAMILY_ID}: this family must only be run through its own bespoke "
        "driver (scripts/run_ipo_fade_beta_neutral_abi1.py), never the "
        "generic scripts/run_abi1_mining.py dispatch -- it has no per-bar "
        "signal, and the generic path would score a plain long IPO position "
        "instead of the benchmark-plus-overlay account.")


def build_curve(ipo_frame, spy_frame, *, ipo_cost_bps: float,
                spy_cost_bps: float, spy_entry_cost_bps: float) -> dict:
    """The family's account curve, built with the PINNED ``PARAMETERS`` (so a
    driver cannot accidentally run different constants than the catalog
    entry records)."""
    return _overlay.build_beta_neutral_overlay_curve(
        ipo_frame, spy_frame, ipo_cost_bps=ipo_cost_bps,
        spy_cost_bps=spy_cost_bps, spy_entry_cost_bps=spy_entry_cost_bps,
        overlay_notional=PARAMETERS["overlay_notional_fraction"],
        beta_window=PARAMETERS["beta_window_sessions"])
