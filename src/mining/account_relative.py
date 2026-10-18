"""ABI-1 account-relative scoring: score the ACCOUNT, not the trades.

Adopted because per-trade constructions reward sitting out the market, and
because an absolute-return score can accept a strategy that is only
measuring index drift rather than its claimed edge.

The construction
----------------
Every session the account exists -- INCLUDING sessions it holds nothing --
write down what the account made and what the benchmark made, and subtract.
That one column of numbers is the whole test.

  Layer 1 (is the lead real, or the luckiest of N tries?)
      The EXISTING `dsr.py` machinery, unchanged in formula, unchanged in
      trial count, unchanged in the 0.95 bar. Only the input series changes:
      the full-period daily LOG excess of the account curve over the
      benchmark. `n` is DAYS, never trades -- which is what satisfies the
      requirement that a strategy need not trade every cycle if it has to
      wait. Ten trades and ten thousand are handled identically.

  Layer 2 (would an investor actually hold it?)
      Four pass/fail refusals on the account's own equity curve, applied only
      AFTER layer 1, never ranked, never blended, never a denominator. Three
      thresholds are the benchmark's own measured value over the same
      sessions, so no constant is invented; the fourth is an
      absolute drawdown cap.

  Layer 3 (does it hold up on sessions the search never saw?)
      The last three years of sessions (`constants.ABI1_HOLDOUT_SESSIONS`)
      are carved off BEFORE layer 1 scores anything, and the same daily log
      excess over that stretch must clear `constants.DSR_PASS_THRESHOLD` on
      `dsr.probabilistic_sharpe_ratio` against a ZERO benchmark Sharpe, with
      NO multiple-testing deflation. See `split_holdout`/`holdout_gate` and the section header above
      them for why the criterion is NOT "did it beat the index over the
      holdout" (that question is terminal wealth, which may
      be reported but must never decide) and NOT a Sharpe sign check (the same
      question again, divided by a positive number).

Why the account and not the trades
----------------------------------
A strategy invested a fraction f of the time must earn ~1/f times the
benchmark's rate WHILE INVESTED merely to tie over the account's life. A
strategy can beat the market per invested day and still lose
the account race purely from sitting out. Time-matched excess is not the
gate measure precisely because it hides idle time: for a strategy that sits
out most sessions, full-period excess can be far more negative than
time-matched excess -- the measure that hides idle time shrugs where the
full-period one convicts.

Log, not arithmetic: arithmetic excess can disagree with terminal wealth,
while log excess agrees, and the objective is geometric (maximise terminal
wealth over time).

Boundaries (deliberate, and pinned by `test_module_never_records_anything`)
-------------------------------------------------------------------------
This module never imports `trial_ledger`, never writes a file, and never
reads the clock. It computes and returns; a caller decides what to record.
Mirrors `benchmark_relative.py`'s own boundary for the same reason.

THE LEDGER TRAP THIS MODULE DELIBERATELY AVOIDS
-----------------------------------------------
`run_batch3_mining._score_one` writes `sharpe_in_sample = stats["sharpe"]` --
the PER-TRADE Sharpe -- and `trial_ledger.pooled_sr_variance` reads exactly
that field to build the DSR's dominant `sr_variance` term. The per-trade
sr_variance is in per-TRADE units and far larger than the daily
`constants.DAILY_SR_VARIANCE`, so mixing the two moves the SR0 term
substantially. Writing an ABI-1 daily Sharpe into
`sharpe_in_sample` would silently corrupt the pool for BOTH constructions,
permanently, in an append-only ledger. This module therefore RETURNS its
daily Sharpe and writes nothing; wiring must use a separate field and a
separate pool.

Note also that `constants.DEFAULT_SR_VARIANCE_FLOOR` (0.0004) was chosen for
the per-trade construction. That it may happen to sit near
`constants.DAILY_SR_VARIANCE` is a COINCIDENCE, not a justification, and this
module refuses to score rather than silently borrow it (see `score_account`).

Engine conventions mirrored (see backtest_engine.run_backtest)
--------------------------------------------------------------
* Entry fills at the NEXT bar's Open; `in_market` is True from that bar
  through `exit_date` inclusive, so the entry bar's Close/Open move belongs
  to the account and crediting cash on it would double-count.
* Exit fills at `exit_date`'s Close; round-trip cost is charged once, at exit.
* The engine's tax haircut applies to a POSITIVE round-trip net return only.

Cash and tax conventions this module implements
------------------------------------------------
They do NOT all push the same way. The idle credit IS the account's entire return on every
session it sits out, so RAISING that credit RAISES a patient strategy's
measured excess. Conventions B and C raise it -- they are adopted because
they are arithmetically right, not because they tighten anything. Convention
A cuts it by an order of magnitude and is the one that genuinely tightens
the bar.

  A       Idle cash is credited at the brokerage SWEEP rate
          (`constants.IDLE_SWEEP_APY`, `sweep_cash_daily`) -- what an
          account actually holding cash earns. Parking it in bills or a
          bond fund instead is admitted ONLY through `parked_cash_daily`,
          which requires its spread and interest tax to be declared.
  B       ^IRX is a BANK DISCOUNT quote, not an investment yield. It is
          converted to its coupon-equivalent yield before crediting parked
          cash (`coupon_equivalent_yield`). Reading a discount quote as a
          return under-states what parked cash actually earns.
  C       Cash accrues on CALENDAR days, not trading days, so a Monday
          carries the whole weekend (`daily_cash_rate`). Small in aggregate
          (0-5 bps/yr); a correctness rule, not a magnitude one.
  D       Losses NET against later gains, with an unlimited carryforward
          tracked in DOLLAR units (`netted_tax_factors`). Winners-only is
          not a conservative default: its penalty scales with turnover, so
          it distorts the RANKING between families.
  E       Round trips held beyond `constants.LONG_TERM_HOLDING_DAYS` are
          charged `constants.LTCG_TAX_RATE`, not the short-term rate. A
          long-holding family can realize most of its gains long-term, so a
          flat rate would bias trend-followers against high-turnover families.

`tax_factor_by_exit_day` survives as the un-netted single-rate closed form
that the netted walk must reduce to; it is NOT on the scoring path.
"""
from __future__ import annotations

import math
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd

from . import constants, dsr
from .benchmark_relative import max_drawdown

# Sessions per year. NOT an accrual divisor: cash accrues on CALENDAR days
# (see CALENDAR_DAYS_PER_YEAR below). Its
# one live use is annualizing a per-session Sharpe for human reading in
# `holdout_gate`, which is a sessions-per-year conversion and
# the correct place for it. Do not reach for it to accrue cash.
# Matches benchmark_relative.TRADING_DAYS_PER_YEAR and
# causal_backtest.TRADING_DAYS; kept local so this module has no import
# dependency on either for a bare integer.
TRADING_DAYS_PER_YEAR = 252

# Replaces constants.MIN_TRADES_FOR_SCORING (15) on this path ONLY. A
# minimum-trade rule is a frequency penalty: it refuses a score to a patient,
# low-turnover strategy, which is exactly the penalty this construction is
# meant to remove. One year of account history is the replacement.
MIN_ACCOUNT_SESSIONS = 252

# A fixed absolute cap, with the accepted cost that it can also reject a
# strategy that beats the index but draws down heavily. Applied ON TOP OF the relative drawdown
# gate, because the benchmark's own worst drawdown over 1993-2026 is about
# -55%, so a candidate can pass the relative gate and still halve the account.
ABSOLUTE_MAX_DRAWDOWN_CAP = 0.35

# Weekly resampling anchor. W-FRI buckets a Mon-Fri trading week into one
# label, which is what "consecutive losing weeks" means for a US equity
# account. A partial final week is kept (it is a real week the account lived
# through), which is why the streak counter must tolerate a short last bucket.
_WEEK_RULE = "W-FRI"

TaxPolicy = Literal["realization", "none"]

# Every verdict `evaluate` can return, in the order it decides them. The last
# two stages come from the out-of-sample gate; the first four precede it.
#
# `cut_holdout` is EVALUATED AND FAILED; `holdout_not_evaluable` is COULD
# NOT EVALUATE. They are separate members for the same reason
# `split_holdout` has two refusal codes: one of them means the candidate was
# tested and lost, the other means nothing was tested, and a reader who
# cannot tell them apart will draw the wrong conclusion from a run log.
#
# "survived" is deliberately LAST: nothing may reach it without passing
# every gate before it.
VERDICTS = ("not_scoreable", "cut_dsr", "cut_risk_refusal",
            "holdout_not_evaluable", "cut_holdout", "survived")


# ------------------------------------------------------------ cash / idle

# Calendar days per year -- the accrual divisor. NOT `TRADING_DAYS_PER_YEAR`: a brokerage credits
# interest on calendar days, so a year of sessions accrues 365 days, not 252.
# And NOT 360: the 365/91 factor in `coupon_equivalent_yield` below already
# puts the rate on an actual/365 INVESTMENT-yield basis, while 360 is the
# money-market convention belonging to the DISCOUNT quote that conversion
# moves away from. Brokerage cash-credit disclosures typically state no
# day-count basis for the credit side, so the basis is inherited from the
# conversion rather than invented here.
CALENDAR_DAYS_PER_YEAR = 365

# Bank-discount quoting conventions for a 13-week U.S. Treasury bill.
_BILL_TERM_DAYS = 91
_DISCOUNT_BASIS_DAYS = 360
_INVESTMENT_BASIS_DAYS = 365


def coupon_equivalent_yield(discount):
    """Convert a 13-week T-Bill BANK DISCOUNT quote to its coupon-equivalent
    (investment) yield.

        P  = 1 - d * 91 / 360          price per unit of face
        CE = (1 - P) / P * 365 / 91

    Why this exists. ``^IRX``, and
    therefore ``data/raw/IRX.csv``'s ``irx_rate_annual`` column, is a bank
    DISCOUNT quote: a fraction of FACE value on a 360-day year, not a return
    on the amount actually invested. Reading it as a simple return understates
    the idle alternative. DIRECTION: under-crediting idle cash PENALISES a strategy that sits out, because
    the idle credit is that strategy's whole return on every session it
    holds nothing.

    Checked against the U.S. Treasury's published Daily Treasury Bill Rates:
    the stored value tracks the published 13-week DISCOUNT figure closely and
    sits below the published 13-week COUPON EQUIVALENT. The formula above
    reproduces Treasury's own published coupon equivalent to within a few
    thousandths of a percentage point.

    NOT a percent/decimal conversion. `src/data_pipeline.py`'s
    `fetch_risk_free_rate` already divides by 100 and IRX.csv already stores
    0.03757 for 3.757%; that conversion is correct and must not be applied
    twice.

    `discount` may be a scalar or an array. A NEGATIVE discount is real --
    ^IRX has printed negative on a handful of sessions -- and gives P > 1 and CE < 0, which is arithmetically fine
    and deliberately allowed through with its sign intact.

    Raises ValueError if the implied price reaches zero or below (a discount
    at or beyond 360/91, about 395.6%), rather than returning inf and
    corrupting every downstream session.
    """
    d = np.asarray(discount, dtype=float)
    price = 1.0 - d * (_BILL_TERM_DAYS / _DISCOUNT_BASIS_DAYS)
    if np.any(price <= 0.0):
        raise ValueError(
            "implied bill price is zero or negative for a discount rate of "
            f"{float(np.max(d))}; a 13-week discount quote at or beyond "
            f"{_DISCOUNT_BASIS_DAYS / _BILL_TERM_DAYS:.4f} has no "
            "coupon-equivalent yield. Refusing to return inf.")
    return ((1.0 - price) / price
            * (_INVESTMENT_BASIS_DAYS / _BILL_TERM_DAYS))


def _validated_session_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Preconditions shared by every per-session cash producer.

    One shared copy for all three producers, so a future guard cannot be
    added to the bill path and forgotten on the sweep path.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a pd.DatetimeIndex")
    if not index.is_unique:
        raise ValueError("index must be unique")
    if index.tz is not None:
        # NOT cosmetic, and NOT auto-converted. A tz-AWARE index makes
        # `rate.index.union(index)` (tz-naive union tz-aware) degrade to an
        # object-dtype Index; ffill/shift then no longer align on time, every
        # session silently receives IRX.csv's LAST row, and the isna() guard
        # below never fires -- a lookahead leak that reports no error at all.
        # Reachable: src/mining/intraday_data.py builds tz-aware ET frames.
        # Refusing beats tz_localize(None) here, because a silent conversion
        # would move sessions across a DST boundary without saying so.
        raise ValueError(
            "index must be timezone-NAIVE; a tz-aware index silently yields "
            "a future cash rate for every session. Convert explicitly with "
            "index.tz_convert('America/New_York').tz_localize(None).")
    return index


def _elapsed_calendar_days(index: pd.DatetimeIndex):
    """Calendar days each session accrues over.

    The first session has no predecessor in the index and is credited
    one day; reaching outside the caller's window to find a real previous
    session is deliberately refused.
    """
    # Calendar days since the PREVIOUS session in this index; the
    # first session has no predecessor here and is credited exactly one day.
    elapsed = np.diff(index.to_numpy().astype("datetime64[D]")).astype(float)
    elapsed = np.concatenate(([1.0], elapsed))
    if np.any(elapsed <= 0.0):
        raise ValueError(
            "index must be sorted ascending with sessions at least one "
            "calendar day apart; a non-ascending index credits NEGATIVE cash "
            "and two timestamps inside one day credit none")
    return elapsed


def daily_cash_rate(index: pd.DatetimeIndex, *, data_dir: str,
                    days_per_year: int = CALENDAR_DAYS_PER_YEAR) -> pd.Series:
    """Per-session SIMPLE cash return on `index`, from the ^IRX 13-week
    T-Bill series at ``<data_dir>/IRX.csv``.

    NOT THE DEFAULT IDLE PRODUCER. This
    is the GROSS, pre-friction bill yield: correct arithmetic for a bill,
    but not what an account holding cash earns, and not admissible on the
    scoring path on its own. Feed `sweep_cash_daily` (cash) or
    `parked_cash_daily` (declared bill/bond parking, friction charged) to
    `account_daily_returns`. Kept in the same spirit as
    `tax_factor_by_exit_day`: the honest special case that the admitted
    producers are built from.

    Two conventions shape the arithmetic, and both move the
    number UP. Both are adopted for CORRECTNESS, not because they tighten
    the bar -- a higher idle credit RAISES a sitting-out
    strategy's excess, so these two move results in the flattering
    direction. The sweep-rate producer is what cuts it back.

    Discount conversion -- the stored quote is a BANK DISCOUNT quote and is converted to
    its coupon-equivalent yield first; see `coupon_equivalent_yield`.

    Calendar accrual -- accrual is on CALENDAR days, not trading days.
    Brokerage sweep interest typically accrues daily on funds from the day
    they are deposited. So the return posted on session ``t`` covers ``(t - t_prev).days`` calendar
    days -- three on a Monday after a normal weekend. Over a year those
    counts total 365, which is why the aggregate effect is small: the convention is for CORRECTNESS, not magnitude.

    FIRST SESSION: it has no predecessor in `index`, so its elapsed-day count
    is unknowable. It is credited ONE day. Reaching backwards outside the
    index for a prior date is deliberately refused -- the caller's window is
    the account's life, and one day out of thousands is not worth inventing
    a session boundary the caller did not give.

    The divisor is calendar days, not trading days: 365 days of accrual over
    a 252 divisor would credit 1.45x the annual rate. `days_per_year`
    defaults to `CALENDAR_DAYS_PER_YEAR`.

    Lookahead safety is inherited, not reimplemented: the rate applied to a
    session is the last rate published STRICTLY BEFORE it (forward-filled,
    never backfilled, then shifted one place). This mirrors
    `causal_backtest.cash_log_returns_by_date`, which is the repo's one
    existing implementation of exactly this; the arithmetic is duplicated
    here only to return SIMPLE returns on a `DatetimeIndex` rather than LOG
    returns on a `datetime.date` index, and because `src/mining/` may not
    import from the LETF side of the repo. The two DELIBERATELY disagree
    on level: that reader spreads the raw discount quote over 252 trading
    days, while this one applies the two conventions above.

    FAILS CLOSED (ValueError) if any session has no prior published rate --
    e.g. an index reaching before IRX.csv's first row. Silently zero-filling
    is precisely the idle-cash-at-0% assumption that can reverse a verdict, so it is refused rather than defaulted.
    """
    index = _validated_session_index(index)
    path = os.path.join(data_dir, "IRX.csv")
    if not os.path.exists(path):
        raise ValueError(f"cash rate series not found at {path}")
    irx = pd.read_csv(path, parse_dates=["Date"])
    irx = irx.dropna().drop_duplicates("Date").sort_values("Date")
    if irx.empty:
        raise ValueError(f"cash rate series at {path} has no usable rows")
    rate = pd.Series(irx["irx_rate_annual"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(irx["Date"]))
    # Union-then-shift: the value carried onto a session is the previous
    # entry in the merged calendar, i.e. a rate published before it.
    prior = rate.reindex(rate.index.union(index)).ffill().shift(1).reindex(index)
    if prior.isna().any():
        missing = index[prior.isna()]
        raise ValueError(
            f"no prior published cash rate for {int(prior.isna().sum())} "
            f"session(s) (first: {missing[0].date()}, last: "
            f"{missing[-1].date()}); IRX.csv starts "
            f"{rate.index[0].date()}. Refusing to assume 0% idle return.")
    # The stored quote is a bank discount, not an investment yield.
    annual = coupon_equivalent_yield(prior.to_numpy(dtype=float))
    elapsed = _elapsed_calendar_days(index)
    return pd.Series(annual * elapsed / float(days_per_year),
                     index=index, name="cash")


def sweep_cash_daily(index: pd.DatetimeIndex, *,
                     apy: float = constants.IDLE_SWEEP_APY,
                     days_per_year: int = CALENDAR_DAYS_PER_YEAR
                     ) -> pd.Series:
    """Per-session SIMPLE cash return for an account that is ACTUALLY HOLDING
    CASH -- the default idle credit.

    This is the producer `evaluate` should be fed unless the caller has
    deliberately declared that idle capital is parked in something else.
    `daily_cash_rate` above is not that default: crediting the 13-week
    bill to an account whose cash is sitting in a brokerage sweep overstates
    the idle leg by hundreds of basis points a year, and does so ONLY for
    candidates that sit out -- a bias between strategy types, not a level
    error. See `constants.IDLE_SWEEP_APY` for the rate and its provenance.

    Needs no data file and cannot fail closed on coverage, because the rate
    is a constant rather than a series: that is the whole point of the sweep
    convention. Accrual is on CALENDAR days on exactly the same rule as
    `daily_cash_rate` -- a Monday session carries the
    weekend, and the first session, having no predecessor in `index`, is
    credited one day.

    `apy` is an argument so a rate change is a call-site change rather than
    an edit to this function; production callers should leave the default.
    """
    idx = _validated_session_index(index)
    elapsed = _elapsed_calendar_days(idx)
    return pd.Series(float(apy) * elapsed / float(days_per_year),
                     index=idx, name="cash")


def parked_cash_daily(index: pd.DatetimeIndex, *,
                      in_market: pd.Series,
                      data_dir: str,
                      spread_bps: float,
                      interest_tax_rate: float,
                      days_per_year: int = CALENDAR_DAYS_PER_YEAR
                      ) -> pd.Series:
    """Per-session SIMPLE return for an account that PARKS idle capital in
    bills, a bond fund or a money fund instead of holding cash.

    The principle: if holding bonds or some other risk-free-rate instrument,
    friction costs like spread and taxes must be tracked. So this admits the higher bill yield ONLY with its costs, and
    `spread_bps` and `interest_tax_rate` are REQUIRED -- there is deliberately
    no way to reach an un-frictioned bill rate through this function, because
    that number flatters every candidate that sits out.

    Two frictions, and each is charged in the direction that cannot
    manufacture a survivor:

    INTEREST TAX. Bill, bond-fund and money-fund distributions are ordinary
    income, so `interest_tax_rate` should be the SHORT-term rate, never
    `constants.LTCG_TAX_RATE`. It is charged on POSITIVE interest only: a
    negative bill yield is a real cost and is passed through undiminished,
    where multiplying it by (1 - rate) would shrink it toward zero and
    flatter the strategy. ^IRX has printed negative on some sessions, so
    this path is reachable on real data.

    ROUND-TRIP SPREAD. Every contiguous idle stretch is one real round trip:
    buy the parking instrument on going flat, sell it on redeploying. Half
    the spread is charged on the FIRST idle session of each run and half on
    the LAST, which keeps both halves on sessions the account is actually
    idle -- `account_daily_returns` reads the strategy's own return on
    deployed sessions, so a charge posted to the redeploy session would be
    silently discarded. A one-session idle run therefore carries the whole
    round trip, which is correct: it was bought and sold. A run that reaches
    the end of `index` is still charged both halves; the position has to be
    liquidated to be measured, and over-charging is the safe direction.

    NOT MODELLED, deliberately: the parking instrument's own price risk (a
    bond fund is not cash), its expense ratio, and any settlement lag. This
    function is only honest for a short-duration parking vehicle; anything
    with duration needs its own return series, not a yield.

    `days_per_year` and the lookahead-safe rate lookup are inherited from
    `daily_cash_rate` unchanged.
    """
    idx = _validated_session_index(index)
    if not (0.0 <= interest_tax_rate < 1.0):
        raise ValueError(
            f"interest_tax_rate must be in [0, 1), got {interest_tax_rate}")
    if not spread_bps >= 0.0:
        raise ValueError(
            f"spread_bps must be non-negative, got {spread_bps}")
    gross = daily_cash_rate(idx, data_dir=data_dir,
                            days_per_year=days_per_year).to_numpy(dtype=float)
    # Tax on POSITIVE interest only -- see the docstring on why a negative
    # yield must not be shrunk toward zero.
    net = gross - float(interest_tax_rate) * np.maximum(gross, 0.0)
    idle = ~_validated_mask(in_market, idx).to_numpy().astype(bool)
    half = float(spread_bps) / 1e4 / 2.0
    charge = np.zeros(len(idx), dtype=float)
    if half > 0.0:
        for i in range(len(idx)):
            if not idle[i]:
                continue
            if i == 0 or not idle[i - 1]:
                charge[i] += half           # bought the parking instrument
            if i == len(idx) - 1 or not idle[i + 1]:
                charge[i] += half           # sold it
    return pd.Series(net - charge, index=idx, name="cash")


def tax_factor_by_exit_day(index: pd.DatetimeIndex, trades_df: pd.DataFrame,
                           *, tax_haircut_rate: float) -> pd.Series:
    """The UN-NETTED, single-rate special case. NOT the function `evaluate`
    uses -- see `netted_tax_factors` below, which is what is on the
    scoring path.

    Retained deliberately, for two reasons: it is the closed form that needs
    no equity path, and it is the reference the netted walk must reduce to
    when the carryforward is zero (pinned by
    `test_netted_factors_reduce_to_the_unnetted_form_when_nothing_is_carried`).
    Do not reach for it as a scoring input: it taxes every winner and credits
    no loser, a turnover-dependent penalty, so it distorts the RANKING
    between families rather than merely their level.

    Per-session MULTIPLICATIVE tax factor aligned to `index`: 1.0 on every
    session except each trade's exit bar, which carries

        f = 1 - tax_haircut_rate * net / (1 + net)      for net > 0
        f = 1                                            otherwise

    where `net` is that trade's PRE-TAX round-trip net return.

    Why a factor and not a subtracted drag (a per-session subtraction reproduces
    an error): the account curve COMPOUNDS, so
    a per-session decrement is a fraction of CURRENT equity, while `net_ret`
    is a fraction of the trade's ENTRY equity. Subtracting `tax * net`
    directly therefore over-charges by the account's growth factor over the
    trade -- 1.8x the true tax on a +100% trade, 6.6x on a +900% one -- and
    for net > 1/tax - 1 it drives the session's return below -100% and makes
    the account unscoreable. The factor form is exact and self-normalising:
    over the trade the daily terms compound to (1 + net), so scaling the exit
    bar's growth by f gives (1 + net) * f = 1 + net * (1 - tax), which is
    EXACTLY `backtest_engine.run_backtest`'s own post-tax per-trade multiple,
    independent of prior equity and of the exit bar's own move.

    Why realization and not mark-to-market: the engine haircuts a POSITIVE
    round-trip net return only, and there is no well-defined per-day
    analogue -- haircutting every up-day is a strictly harsher operation that
    would penalise a volatile winner more than a smooth one of identical
    outcome. Charging at the exit bar, where the engine's round-trip cost is
    already charged, keeps the daily and per-trade paths consistent.

    Accepts `net_ret` PRE-tax. Callers whose `trades_df` already carries a
    POST-tax `net_ret` (anything that ran `run_backtest` with a non-zero
    `tax_haircut_rate`) must pass `tax_haircut_rate=0.0` here, or the haircut
    is charged twice.

    Multiple trades exiting the same session compose multiplicatively, which
    is the correct treatment of two independent realizations on one day.

    Raises ValueError on a rate outside [0, 1) or an exit_date absent from
    `index` -- fail closed rather than silently drop a trade's tax.
    """
    if not (0.0 <= tax_haircut_rate < 1.0):
        raise ValueError(
            f"tax_haircut_rate must be in [0, 1), got {tax_haircut_rate}")
    out = pd.Series(1.0, index=index, name="tax_factor")
    if trades_df is None or len(trades_df) == 0 or tax_haircut_rate == 0.0:
        return out
    exits = pd.DatetimeIndex(trades_df["exit_date"])
    pos = index.get_indexer(exits)
    if (pos < 0).any():
        bad = exits[pos < 0]
        raise ValueError(
            f"{len(bad)} trade exit_date(s) not in index (first {bad[0]}); "
            "pass the same index that produced trades_df")
    net = trades_df["net_ret"].to_numpy(dtype=float)
    gains = np.clip(net, 0.0, None)
    # net > 0 implies 1 + net > 1, so the divisor is never near zero and the
    # factor is bounded in (1 - tax, 1] -- it can never reach 0 or go negative.
    factors = 1.0 - tax_haircut_rate * gains / (1.0 + np.clip(net, 0.0, None))
    values = out.to_numpy(dtype=float).copy()
    np.multiply.at(values, pos, factors)
    return pd.Series(values, index=index, name="tax_factor")


def netted_tax_factors(*, pretax_account_daily: pd.Series,
                       trades_df: pd.DataFrame,
                       short_rate: float = constants.STCG_TAX_RATE,
                       long_rate: float = constants.LTCG_TAX_RATE,
                       long_term_days: int = constants.LONG_TERM_HOLDING_DAYS
                       ) -> pd.Series:
    """Per-session multiplicative tax factor with LOSS NETTING and a
    long-term / short-term rate split. This is the function `evaluate` uses; `tax_factor_by_exit_day`
    above is its no-carryforward, single-rate special case.

    WHY NETTING IS NOT OPTIONAL, and not a conservative default
    -----------------------------------------------------------
    Taxing every winner while giving a loser no credit is what real tax law
    does NOT do. Netting's benefit grows with turnover (more round trips, more
    losses to offset), so winners-only is a turnover-dependent penalty that
    distorts the RANKING between families, not merely their level.

    THE DESIGN, and why the obvious one-line version is wrong
    --------------------------------------------------------
    A trade's loss in DOLLARS is `entry_equity * (-net)`, and entry equity
    differs per trade because the account compounds (and moves on idle
    sessions via the cash leg). A carryforward accumulated in per-trade
    RETURN units would therefore be a per-trade-quantity-on-a-compounding-
    curve error (the same shape `tax_factor_by_exit_day` explains for a
    subtracted drag): getting it wrong shifts terminal wealth and the daily
    Sharpe, ALWAYS in the flattering direction.

    So this function takes the PRE-TAX account curve and walks it forward
    ONCE, carrying the unused loss in DOLLAR (equity) units. The PRE-TAX
    curve is the load-bearing input -- it is what sizes each loss in dollars,
    and it is why `evaluate` computes the account twice.

    The tax already charged is also folded into the running equity, so
    `equity` here is the account's real modelled value. That feedback turns
    out to be provably INERT for the factors this function returns, and the
    reason is worth writing down so a later reader neither deletes the line
    as dead nor trusts it to do something it does not: divergence between a
    walk that reinvests the tax and one that does not can only begin at the
    first factor below 1, which requires `gain > offset`, which requires the
    carryforward to have been exhausted at exactly that trade. From there
    both walks' equities differ by one constant factor, every later
    carryforward and gain scales with it, and `1 - rate*(gain - offset)
    / equity` is invariant under that common scaling. The line stays because
    it makes `equity` mean what it says and because the zero-equity guard
    below reads that same value -- not because any returned number moves.

    Entry equity is recovered from the exit, not looked up at the entry bar:
    over a trade the daily terms compound to (1 + net), so with `W` the
    running equity at the exit bar

        entry_equity = W / (1 + net)        gain_dollars = W * net / (1 + net)

    This is exact under the identity `tax_factor_by_exit_day` already relies
    on, and it is what makes the reduction below hold to floating point.
    Reading equity at the entry bar instead would give a different number
    whenever another trade's tax landed inside this one's span.

    REDUCES EXACTLY to `tax_factor_by_exit_day` when the carryforward is
    zero: with no offset the tax is `rate * W * net / (1 + net)`, so
    `f = 1 - rate * net / (1 + net)` independently of equity -- the closed
    form, which is itself pinned against `backtest_engine.run_backtest`'s own
    post-tax per-trade multiple. Pinned by
    `test_netted_factors_reduce_to_the_unnetted_form_when_nothing_is_carried`.

    LONG-TERM vs SHORT-TERM
    -----------------------------------------------
    A round trip whose `(exit_date - entry_date).days` is STRICTLY GREATER
    than `long_term_days` is long-term. Exactly that many days is still
    short-term. Both rates are parameters defaulting to `constants` so a
    rate change is an argument change, not a design change. The split matters
    because a long-holding (e.g. trend-following) family can realize most of
    its gains long-term while a high-turnover family may have no long-term
    trades at all -- a flat rate biases BETWEEN family types, which is worse
    than a level error.

    ORDERING, stated rather than implied
    ------------------------------------
    TWO carryforward pools, matched on CHARACTER. A short-term loss shelters short-term gains; a long-term
    loss shelters long-term gains. This mirrors the real rule that a carried
    capital loss keeps its character and offsets its own kind first.

    WHY TWO POOLS, not one. A single pooled carryforward spent BEFORE the tax
    rate is consulted would let a long-term gain rated at
    `constants.LTCG_TAX_RATE` = 0.0 exhaust the shelter while saving
    nothing, leaving a later short-term gain to pay full freight. Families
    whose gains are mostly long-term (trend-followers) would systematically
    burn their shelter at 0% while high-turnover families kept the full
    benefit -- re-creating, in mirror image, the very between-family bias the
    short/long split exists to remove.

    THE CROSS-OVER IS DEFERRED, NOT DROPPED, and this is forced rather than
    chosen. Real law nets the two characters against each other only at the
    END of a netting period, once it is known that no same-character gain
    remains to use the shelter. Deciding mid-walk to spend short-term shelter
    on a long-term gain would require knowing no future short-term gain needs
    it -- lookahead, which this module refuses everywhere else. Since the
    carryforward is unlimited and never expires within the account's life
    (see SIMPLIFICATIONS), the deferred cross-over would resolve only at the
    end, where an unused loss is carried and never refunded. So it has no
    effect on any returned factor, and is deliberately not implemented as a
    reachable branch that could never be exercised.

    DIRECTION: character matching can only RAISE the modelled tax bill, never
    lower it, because it strictly narrows where each loss may be applied.
    That is the direction that cannot manufacture a false survivor.

    Within a single session there is no chronology, so the deterministic
    tie-break is: losses are realised before gains, and gains in `trades_df`
    row order. Without that, the shelter a same-day loss provides would
    depend on row order.

    SIMPLIFICATIONS, deliberate and load-bearing for a reader
    --------------------------------------------------------
    * Carryforward is UNLIMITED and never expires within the account's life.
      The annual ordinary-income offset limit and its calendar-year reset are
      real and are NOT modelled.
    * WASH SALES are NOT modelled, and the exposure is real: a strategy that
      re-enters the same instrument within 30 days of a losing exit triggers
      the rule. A disallowed loss cannot shelter a gain, so
      the netting modelled here is an UPPER BOUND on the real benefit.
    * A loss remaining at the end of the account's life is carried, never
      refunded.

    Accepts `net_ret` PRE-tax, exactly as `tax_factor_by_exit_day` does: a
    caller whose `trades_df` already carries a POST-tax `net_ret` must pass
    both rates as 0.0, or the haircut is charged twice.

    Fails closed (ValueError) on: a rate outside [0, 1); a missing or NaT
    `entry_date`/`exit_date`; an `exit_date` absent from the curve's index; a
    non-finite `net_ret`; a `net_ret` at or below -100% (the entry equity is
    then undefined -- a division by zero the winners-only form never
    reaches, because it clips losses away before dividing); an `exit_date`
    before its `entry_date`; and an account whose running equity reaches zero
    or goes non-finite.
    """
    for name, rate in (("short_rate", short_rate), ("long_rate", long_rate)):
        if not (0.0 <= float(rate) < 1.0):
            raise ValueError(f"{name} must be in [0, 1), got {rate}")
    idx = pretax_account_daily.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("pretax_account_daily must have a DatetimeIndex")
    if not idx.is_unique:
        raise ValueError("pretax_account_daily.index must be unique")
    out = pd.Series(1.0, index=idx, name="tax_factor")
    if trades_df is None or len(trades_df) == 0:
        return out
    for column in ("entry_date", "exit_date", "net_ret"):
        if column not in trades_df.columns:
            raise ValueError(
                f"trades_df is missing {column!r}; the long-term/short-term "
                "split needs both dates, and defaulting a missing entry_date "
                "to short-term would silently mis-rate every long-term gain")
    entries = pd.DatetimeIndex(trades_df["entry_date"])
    exits = pd.DatetimeIndex(trades_df["exit_date"])
    if entries.isna().any():
        raise ValueError(f"{int(entries.isna().sum())} trade(s) have a null "
                         "entry_date; the holding period is unknowable")
    if exits.isna().any():
        raise ValueError(f"{int(exits.isna().sum())} trade(s) have a null "
                         "exit_date")
    held = (exits - entries).days.to_numpy()
    if (held < 0).any():
        raise ValueError(
            f"{int((held < 0).sum())} trade(s) have an exit_date before "
            "entry_date; refusing to guess the holding period")
    pos = idx.get_indexer(exits)
    if (pos < 0).any():
        bad = exits[pos < 0]
        raise ValueError(
            f"{len(bad)} trade exit_date(s) not in index (first {bad[0]}); "
            "pass the account curve that produced trades_df")
    net = trades_df["net_ret"].to_numpy(dtype=float)
    if not np.isfinite(net).all():
        raise ValueError("net_ret must be finite for every trade")
    if (net <= -1.0).any():
        raise ValueError(
            "a net_ret at or below -100% leaves the trade's entry equity "
            "undefined (exit equity / (1 + net) divides by zero); the "
            "account is wiped out and cannot be taxed")
    daily = np.asarray(pretax_account_daily, dtype=float).ravel()
    if not np.isfinite(daily).all():
        raise ValueError("pretax_account_daily must be finite")
    if short_rate == 0.0 and long_rate == 0.0:
        return out

    is_long = held > int(long_term_days)
    # Losses first within a session (the stated tie-break), then row order --
    # np.lexsort's last key is primary and its sort is stable.
    order = np.lexsort(((net > 0.0).astype(np.int8), pos))
    # Equity BEFORE any tax, session by session; the tax already charged is
    # carried separately in `tax_mult` so the walk stays O(trades).
    growth = np.cumprod(1.0 + daily)
    factors = np.ones(len(idx), dtype=float)
    tax_mult = 1.0
    # TWO carryforward pools, matched on character.
    # A carried loss keeps its character and offsets its own kind; see the
    # ORDERING section above for why the cross-over is deferred rather than
    # applied on contact.
    carryforward = {False: 0.0, True: 0.0}      # keyed by is_long
    for j in order:
        t = int(pos[j])
        equity = growth[t] * tax_mult
        if not np.isfinite(equity) or equity <= 0.0:
            raise ValueError(
                f"account equity is {equity} at {idx[t].date()}, at or below "
                "zero; a wiped-out account has no well-defined tax bill and "
                "dividing the bill by it would return inf")
        n = float(net[j])
        gain = equity * n / (1.0 + n)          # == entry_equity * net
        character = bool(is_long[j])
        if gain <= 0.0:
            carryforward[character] += -gain
            factor = 1.0
        else:
            offset = min(carryforward[character], gain)
            carryforward[character] -= offset
            rate = float(long_rate) if character else float(short_rate)
            factor = 1.0 - rate * (gain - offset) / equity
        factors[t] *= factor
        tax_mult *= factor
    return pd.Series(factors, index=idx, name="tax_factor")


def _validated_mask(in_market, idx: pd.DatetimeIndex) -> pd.Series:
    """`in_market` aligned to `idx` as a real bool Series, failing closed on
    the two mistakes that would otherwise pass silently.

    (a) A NaN in the mask. `pd.Series([True, np.nan]).astype(bool)` maps NaN
        to **True**, i.e. DEPLOYED -- the exact opposite of this module's
        documented conservative direction, and the direction that flatters a
        candidate. A 95%-drawdown account can otherwise be
        scored as passing through this path.
    (b) A wholly misaligned mask (no label in common with `idx`), which
        reindexes to all-False and reads as "idle for the entire life of the
        account" -- a plausible-looking result from a caller bug. Its sibling
        `cash_daily` is refused for the identical mistake, and so is the
        mask.

    A mask that is merely SHORTER than `idx` is still accepted, and its
    missing sessions still mean NOT deployed -- that is the documented
    conservative default, and it is distinguishable from (b) by overlap.
    """
    s = pd.Series(in_market)
    if s.isna().any():
        raise ValueError(
            f"in_market contains {int(s.isna().sum())} NaN value(s); NaN "
            "casts to True (DEPLOYED), the opposite of the conservative "
            "direction. Pass an explicit boolean mask.")
    if len(s) and not s.index.intersection(idx).size:
        raise ValueError(
            "in_market shares no session with strategy_daily; a wholly "
            "misaligned mask would read as idle for the account's entire "
            "life. Pass the mask backtest_engine.run_backtest returned "
            "alongside this strategy series.")
    return s.reindex(idx, fill_value=False).astype(bool)


def account_daily_returns(*, strategy_daily: pd.Series,
                          in_market: pd.Series,
                          cash_daily: pd.Series,
                          tax_factor: Optional[pd.Series] = None) -> pd.Series:
    """The account's own per-session simple return: the strategy's return on
    sessions capital is deployed, the cash rate on sessions it is not, minus
    any tax drag.

    `in_market` is the mask `backtest_engine.run_backtest` returns as its
    second element. Idle sessions are never derived from
    ``strategy_daily == 0`` (a "did the series move today" proxy, which is
    what `full_window_summary` falls back to without a mask): that proxy
    misstates deployment, and a genuinely deployed session can close exactly
    flat.

    All inputs are reindexed onto `strategy_daily.index`. A missing
    `in_market` entry is treated as NOT deployed (the conservative direction:
    it credits cash rather than assuming exposure). Raises ValueError if
    `cash_daily` does not cover every session -- see `daily_cash_rate`.
    """
    idx = strategy_daily.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("strategy_daily must have a DatetimeIndex")
    if not idx.is_unique:
        raise ValueError("strategy_daily.index must be unique")
    mask = _validated_mask(in_market, idx)
    cash = pd.Series(cash_daily).reindex(idx)
    if cash.isna().any():
        raise ValueError(
            f"cash_daily is missing {int(cash.isna().sum())} of "
            f"{len(idx)} sessions; refusing to assume 0% idle return")
    strat = pd.Series(strategy_daily).astype(float)
    if not np.isfinite(strat.to_numpy(dtype=float)).all():
        raise ValueError("strategy_daily must be finite")
    out = np.where(mask.to_numpy(), strat.to_numpy(dtype=float),
                   cash.to_numpy(dtype=float))
    if tax_factor is not None:
        # Multiplicative on the session's GROWTH factor, not a subtraction --
        # see tax_factor_by_exit_day for why the subtractive form over-charged
        # by the account's growth over the trade.
        f = pd.Series(tax_factor).reindex(idx, fill_value=1.0)
        if f.isna().any():
            raise ValueError("tax_factor contains NaN")
        out = (1.0 + out) * f.to_numpy(dtype=float) - 1.0
    return pd.Series(out, index=idx, name="account")


# ------------------------------------------------------------- layer one

def _calendar_aligned_returns(account_daily: pd.Series,
                              benchmark_daily: pd.Series) -> pd.DataFrame:
    """Carry the benchmark's last valuation across its non-trading dates.

    An omitted benchmark date denotes a closed session; its simple return
    is zero, not its previous session's return. Explicit NaNs remain missing
    observations. Only the benchmark's supplied history span is covered:
    never extrapolate an absent leading or trailing price history. Callers
    must retain expected trading dates with NaN for missing observations.
    """
    account = pd.Series(account_daily, name="a")
    benchmark = pd.Series(benchmark_daily, name="b")
    if benchmark.empty:
        return pd.DataFrame({"a": account.iloc[:0], "b": benchmark})
    index = account.index[(account.index >= benchmark.index.min())
                          & (account.index <= benchmark.index.max())]
    return pd.concat([account.reindex(index),
                      benchmark.reindex(index, fill_value=0.0)], axis=1)


def log_excess(account_daily: pd.Series,
               benchmark_daily: pd.Series) -> pd.Series:
    """Per-session LOG excess, ``log1p(account) - log1p(benchmark)``, over
    shared benchmark cutoffs. Account returns between shared cutoffs compound
    into the next cutoff; a weekend cannot disappear or consume extra
    holdout sessions. Risk checks retain the full daily account path.

    Log rather than arithmetic because the objective is geometric: these
    terms SUM to the log of the terminal wealth ratio, so "the excess series
    is positive on average" and "the account finished ahead of the
    benchmark" are the same statement. Arithmetic excess has no such
    identity and can disagree with terminal wealth.

    Raises ValueError if either series has a return <= -100% on a shared
    session (log1p undefined -- a wiped-out account is not scoreable here).
    """
    account, benchmark = pd.Series(account_daily), pd.Series(benchmark_daily)
    cutoffs = account.index.intersection(benchmark.index, sort=False)
    if cutoffs.empty:
        return pd.Series(dtype=float, name="log_excess")
    if not cutoffs.is_monotonic_increasing or not cutoffs.is_unique:
        raise ValueError("scored dates must be unique and ascending")
    a = _log_returns_at_cutoffs(account, cutoffs)
    b = _log_returns_at_cutoffs(benchmark, cutoffs)
    return (a - b).dropna().rename("log_excess")


def _log_returns_at_cutoffs(daily, cutoffs):
    """Sum logs over (previous cutoff, cutoff], keeping every intervening bar."""
    daily = daily.loc[(daily.index >= cutoffs[0]) & (daily.index <= cutoffs[-1])]
    if (daily <= -1.0).any():
        raise ValueError("a return <= -100% makes log1p undefined")
    labels = cutoffs[cutoffs.searchsorted(daily.index, side="left")]
    grouped = np.log1p(daily).groupby(labels, sort=False)
    result = grouped.sum(min_count=1)
    # An explicit unknown bar makes its whole interval unknown; never turn
    # an incomplete interval into a known compounded return by skipping NaN.
    return result.where(grouped.count() == grouped.size()).reindex(cutoffs)


def score_account(excess: pd.Series, *, n_trials: float,
                  sr_variance: Optional[float],
                  min_sessions: int = MIN_ACCOUNT_SESSIONS) -> dict:
    """Layer 1. The EXISTING deflated-Sharpe machinery, fed the daily log
    excess series, with `n` = SESSIONS.

    Returns a dict: `scored` (bool), `dsr_value` (float or None), `reason`,
    `sharpe_daily_excess`, `n_sessions`, `skew`, `kurtosis`, and the
    `n_trials` / `sr_variance` actually used.

    REFUSES to score (`scored=False`) rather than guess when:
      * fewer than `min_sessions` sessions -- the 252-session floor that
        replaces the 15-trade minimum; or
      * `sr_variance` is None. There is deliberately NO fallback to
        `constants.DEFAULT_SR_VARIANCE_FLOOR` here: that constant was chosen
        for the PER-TRADE construction (see this module's docstring), and
        the per-trade vs daily unit mismatch would silently move the bar. A daily
        `sr_variance` must be supplied by a daily pool, and until one exists
        the honest answer is "not scoreable", not a borrowed number.

    Moments come from `benchmark_relative.daily_sharpe_moments`' conventions
    (per-period, NOT annualized -- `dsr.py` requires that).
    """
    from .benchmark_relative import daily_sharpe_moments  # local: cycle-free

    x = np.asarray(excess, dtype=float).ravel()
    n = int(x.size)
    base = {"scored": False, "dsr_value": None, "n_sessions": n,
            "sharpe_daily_excess": None, "skew": None, "kurtosis": None,
            "n_trials": n_trials, "sr_variance": sr_variance}
    if n < int(min_sessions):
        return {**base, "reason": (
            f"only {n} account sessions, below MIN_ACCOUNT_SESSIONS "
            f"({int(min_sessions)})")}
    if sr_variance is None:
        return {**base, "reason": (
            "no daily-construction sr_variance available; refusing to "
            "borrow the per-trade floor (per-trade vs daily unit mismatch)")}
    moments = daily_sharpe_moments(x)
    value = dsr.deflated_sharpe_ratio(
        sr=moments["sr"], n=moments["n"], skew=moments["skew"],
        kurtosis=moments["kurtosis"], n_trials=n_trials,
        sr_variance=sr_variance)
    return {**base, "scored": True, "dsr_value": float(value),
            "sharpe_daily_excess": moments["sr"], "skew": moments["skew"],
            "kurtosis": moments["kurtosis"],
            "reason": f"daily-excess DSR {value:.4f} over {n} sessions"}


# ------------------------------------- the OUT-OF-SAMPLE holdout gate
#
# The holdout stretches to THREE YEARS. Without it there is no out-of-sample
# gate under any name (`evaluate` would take no date, window or split
# parameter, and all four risk refusals would be computed on the same window
# as the score). This section provides it.
#
# THE CRITERION, AND THE TWO CRITERIA IT IS NOT
# ---------------------------------------------
# The obvious test -- "did the account beat the index over the holdout" --
# is forbidden here, and not on taste: the daily log excess terms SUM to the
# log of the terminal-wealth ratio (see `log_excess`), so that test is
# arithmetically IDENTICAL to terminal wealth, which may be
# reported but must never be sole grounds for a verdict. `risk_refusals` already
# excludes terminal wealth from `all_refusals_passed` for exactly this
# reason; admitting it here under another name would be no exclusion.
#
# Restating it in risk-adjusted units does not escape it either. A Sharpe
# ratio is a mean divided by a strictly POSITIVE standard deviation, so
# "Sharpe > 0" and "mean excess > 0" are the same test -- any
# sign-preserving transform (log, annualization, dividing by a positive
# quantity) leaves a ZERO threshold exactly where it started. Only a
# strictly positive threshold makes the denominator do any work at all.
#
# So: the holdout daily log-excess Sharpe must be distinguishable from zero
# at `constants.DSR_PASS_THRESHOLD`, via the EXISTING
# `dsr.probabilistic_sharpe_ratio`, with `sr_benchmark = 0.0` and NO
# multiple-testing deflation. No new constant is introduced -- 0.95 is the
# bar this project already uses, and the length is the three
# years in `constants.ABI1_HOLDOUT_SESSIONS`.
#
# WHY NO DEFLATION HERE, when layer 1 deflates hard
# --------------------------------------------------
# `expected_max_sharpe_ratio` corrects for the luckiest of N tries. The
# holdout is not a search: it is ONE pre-registered confirmation test on a
# candidate already selected, run once, on sessions the selection never saw.
# Deflating it would charge the candidate twice for the same multiplicity --
# the search is charged in layer 1, where it happened.
#
# THE FAILURE THIS GUARDS AGAINST
# -------------------------------
# A holdout gate with no minimum observation count, no benchmark subtracted
# and a zero bar -- the shape of `backtest_engine.holdout_passes`, a BARE
# SIGN CHECK -- can pass a strategy on a handful of trades that only track
# index drift. All three
# holes are closed here: `required_sessions` is a hard floor, the series
# scored IS the benchmark-subtracted excess, and the bar is strictly
# positive.
#
# POWER, MEASURED (against dsr.py at Normal moments)
# --------------------------------------------------
# The annualized daily excess Sharpe a holdout must show to clear 0.95:
# about 1.65 over 1 year, 1.17 over 2, 0.95 over 3, 0.74 over 5. The
# in-sample bar at the configured `DAILY_SR_VARIANCE` / `ABI1_N_TRIALS` is
# about 1.02. Three years therefore sits slightly EASIER than the search
# that produced the candidate, which is the right shape for a confirmation
# test: it must not be a second, harder search.
#
# HONEST LIMIT OF THIS CRITERION, so nobody over-trusts it
# --------------------------------------------------------
# It separates a steady edge from a concentrated windfall only WEAKLY. For a
# total log excess T spread over H sessions on idiosyncratic vol s, moving
# the whole of T into K sessions multiplies the measured Sharpe by roughly
# s / sqrt(s**2 + T**2/(K*H)). Measured over a 756-session holdout at
# s = 0.004: at T = 0.20 a 10-session windfall lands at PSR 0.9484 against
# the steady version's 0.9676 -- it is refused, but by 0.019, and at
# T = 0.36 BOTH pass. `test_a_steady_edge_passes_and_the_same_total_won_
# luckily_does_not` pins the separation at the measured point; do not read
# it as a claim that this gate is a general concentration detector. It is
# not. It is a "is the mean big relative to its own noise" test, and the
# concentration penalty is a side effect of that.
#
# RISK REFUSALS RUN ON THE FULL WINDOW
# ------------------------------------
# The four risk refusals in `risk_refusals` run on the SAME window as
# everything else -- the full joint series, holdout sessions included -- and
# are not evaluated on the holdout alone, so a candidate can pass a 35%
# drawdown cap across 20 years while breaching it inside the last 3. That is
# a choice about what a risk refusal MEANS, not an implementation gap.

def split_holdout(index, *,
                  holdout_sessions: int = constants.ABI1_HOLDOUT_SESSIONS,
                  min_in_sample: int = MIN_ACCOUNT_SESSIONS) -> dict:
    """Split a session index into an in-sample head and a holdout TAIL of
    `holdout_sessions` sessions.

    The holdout is the most recent stretch, and it is REMOVED from the
    in-sample half: an "out-of-sample" confirmation computed on sessions the
    score already saw is confirming itself.

    A consequence a future reader should not misread as a regression:
    removing the tail shrinks the in-sample session count, and `n` enters
    `dsr.probabilistic_sharpe_ratio` as `sqrt(n - 1)`, so the in-sample bar
    RISES slightly. Typically a small percentage for long-history
    families. That is the price of an honest out-of-sample test, not a bug;
    it is large only for short histories, where the split is refused anyway.

    Returns a dict, never raises for a merely-too-short series:
      ok                        bool
      refusal                   None | "holdout_too_short" |
                                "in_sample_too_short"
      reason                    human-readable, names both counts
      in_sample, holdout        the two index objects (on a refusal,
                                `in_sample` is the WHOLE index and `holdout`
                                is empty, so a caller that scores anyway
                                scores the full window rather than nothing)
      n_total, n_in_sample, n_holdout, holdout_sessions_required,
      min_in_sample             the counts, for reporting

    The two refusals are separate codes on purpose. "The series is shorter
    than the holdout" and "the split would starve the score" are different
    problems with different fixes, and a single reason code for both
    misroutes investigation.

    Raises ValueError only for a `holdout_sessions` that cannot define a
    Sharpe at all (< 2) -- a caller error, not a data condition.
    """
    idx = pd.Index(index)
    # NOT cosmetic. Every split below is POSITIONAL (`idx[:n - h]` /
    # `idx[n - h:]`), so on a non-ascending index the "holdout tail" is
    # actually the OLDEST stretch and the scored head is the most recent --
    # a lookahead split that reports itself as a clean out-of-sample test,
    # after which `evaluate` could return a passing verdict. Same failure
    # SHAPE as the tz-aware refusal in `_validated_session_index`: the data is complete, every
    # count is correct, and no missingness check can fire, because it is the
    # ORDER that is wrong. Duplicates are refused in the same breath -- a
    # repeated session would be scored twice and would make `n` a lie.
    if not idx.is_unique:
        raise ValueError(
            "session index must be unique before splitting a holdout; "
            "a duplicated session is scored twice and overstates n")
    if not idx.is_monotonic_increasing:
        raise ValueError(
            "session index must be sorted ASCENDING before splitting a "
            "holdout: the split is positional, so on any other ordering the "
            "'holdout' is not the most recent stretch and the test is a "
            "lookahead. Sort the index (and the series aligned to it) first.")
    h = int(holdout_sessions)
    m = int(min_in_sample)
    if h < 2:
        raise ValueError(
            f"holdout_sessions must be >= 2 for a Sharpe to exist over the "
            f"holdout at all, got {holdout_sessions!r}")
    n = int(len(idx))
    counts = {"n_total": n, "holdout_sessions_required": h,
              "min_in_sample": m}
    if n < h:
        return {"ok": False, "refusal": "holdout_too_short",
                "reason": (f"only {n} joint sessions, fewer than the "
                           f"{h}-session holdout itself"),
                "in_sample": idx, "holdout": idx[:0],
                "n_in_sample": n, "n_holdout": 0, **counts}
    if n - h < m:
        return {"ok": False, "refusal": "in_sample_too_short",
                "reason": (f"carving a {h}-session holdout out of {n} joint "
                           f"sessions leaves {n - h} in sample, below "
                           f"MIN_ACCOUNT_SESSIONS ({m})"),
                "in_sample": idx, "holdout": idx[:0],
                "n_in_sample": n, "n_holdout": 0, **counts}
    return {"ok": True, "refusal": None,
            "reason": (f"{n - h} in-sample sessions scored, {h} sessions "
                       f"held out"),
            "in_sample": idx[:n - h], "holdout": idx[n - h:],
            "n_in_sample": n - h, "n_holdout": h, **counts}


def _holdout_refusal(refusal: str, reason: str, *, n: int,
                     required: int, psr_threshold: float) -> dict:
    """The canonical "could not evaluate" block. Shaped IDENTICALLY to an
    evaluated one so a caller never has to branch on which shape it got --
    but with `evaluated=False`, a named `refusal`, and every measured field
    None rather than a plausible-looking number. A silent `passed=False`
    with a number beside it reads as "evaluated and failed", and conflating
    "could not evaluate" with "evaluated and failed" is an easy, costly
    misreading."""
    return {"evaluated": False, "passed": False, "refusal": refusal,
            "reason": reason, "n_holdout": int(n),
            "required_sessions": int(required), "psr": None,
            "psr_threshold": float(psr_threshold), "sr_benchmark": 0.0,
            "sharpe_daily_excess": None, "sharpe_annualized": None,
            "skew": None, "kurtosis": None, "log_excess_sum": None}


def holdout_gate(account_daily: pd.Series, benchmark_daily: pd.Series, *,
                 required_sessions: int = constants.ABI1_HOLDOUT_SESSIONS,
                 psr_threshold: float = constants.DSR_PASS_THRESHOLD
                 ) -> dict:
    """The out-of-sample confirmation test, on ALREADY-SLICED holdout
    series. Read this section's header comment for why the criterion is a
    PSR against zero and not "did it beat the index".

    Both arguments are the HOLDOUT window only; `evaluate` does the slicing
    via `split_holdout`. The excess is computed with this module's own
    `log_excess` (same inner join, same NaN drop, same wipe-out refusal), so
    the holdout and the score are the same construction on different
    sessions.

    Pass condition, and nothing else::

        psr = dsr.probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurtosis)
        passed = psr >= psr_threshold

    Moments come from `benchmark_relative.daily_sharpe_moments` -- Pearson
    kurtosis, `bias=True` for BOTH moments, which `dsr.py` requires (it
    raises on a mixed convention).

    Three fail-closed refusals, each a NAMED `refusal` code with every
    measured field left None:
      holdout_too_short   fewer than `required_sessions` joint sessions
                          after the join. Without it, a holdout with almost
                          no observations could pass with nothing objecting.
      non_finite_excess   an inf survives `log_excess`'s dropna and would
                          make `daily_sharpe_moments` raise or poison the
                          moments.
      zero_variance       the account tracked the benchmark exactly. The
                          excess Sharpe is UNDEFINED; `daily_sharpe_moments`
                          would hand back sr 0.0 and PSR would come out at
                          exactly 0.5 -- a number that reads like a measured
                          failure and is not one.

    Raises (does not refuse) only for a return <= -100%, which `log_excess`
    rejects: a wiped-out account is a data defect, not a verdict.
    """
    return _holdout_from_excess(
        log_excess(account_daily, benchmark_daily),
        required_sessions=required_sessions, psr_threshold=psr_threshold)


def _holdout_from_excess(excess, *, required_sessions, psr_threshold):
    """Evaluate a prealigned holdout without dropping its leading weekend."""
    from .benchmark_relative import daily_sharpe_moments
    x = np.asarray(excess, dtype=float).ravel()
    n = int(x.size)
    req = int(required_sessions)
    if n < req:
        return _holdout_refusal(
            "holdout_too_short",
            f"holdout has {n} joint sessions, fewer than the {req} required",
            n=n, required=req, psr_threshold=psr_threshold)
    if not np.isfinite(x).all():
        return _holdout_refusal(
            "non_finite_excess",
            f"{int((~np.isfinite(x)).sum())} non-finite value(s) in the "
            f"holdout log-excess series",
            n=n, required=req, psr_threshold=psr_threshold)
    std = float(np.std(x, ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return _holdout_refusal(
            "zero_variance",
            "the holdout log-excess series has zero variance; its Sharpe is "
            "undefined and a PSR built from sr=0.0 would be exactly 0.5, "
            "which reads as a measured failure",
            n=n, required=req, psr_threshold=psr_threshold)

    moments = daily_sharpe_moments(x)
    psr = dsr.probabilistic_sharpe_ratio(
        moments["sr"], 0.0, moments["n"], moments["skew"],
        moments["kurtosis"])
    passed = bool(psr >= float(psr_threshold))
    sr_ann = float(moments["sr"]) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return {"evaluated": True, "passed": passed, "refusal": None,
            "reason": (f"holdout PSR {psr:.4f} against a 0.0 benchmark "
                       f"Sharpe over {n} sessions "
                       f"({'>=' if passed else '<'} {float(psr_threshold)})"),
            "n_holdout": n, "required_sessions": req,
            "psr": psr, "psr_threshold": float(psr_threshold),
            # Pinned in the block so a reader can see the test was against
            # ZERO -- i.e. "is there an edge", not "did it beat the index".
            "sr_benchmark": 0.0,
            "sharpe_daily_excess": float(moments["sr"]),
            "sharpe_annualized": sr_ann,
            "skew": float(moments["skew"]),
            "kurtosis": float(moments["kurtosis"]),
            # Reported, never a criterion: this IS the log terminal-wealth
            # ratio over the holdout (report it, never decide on it).
            "log_excess_sum": float(x.sum())}


# ------------------------------------------------------------- layer two

def weekly_returns(daily: pd.Series, *, rule: str = _WEEK_RULE) -> pd.Series:
    """Compound `daily` into weekly simple returns, ``prod(1 + r) - 1`` per
    bucket. A partial final week is kept -- it is a real week the account
    lived through."""
    s = pd.Series(daily).astype(float)
    if s.empty:
        return pd.Series(dtype=float, name="weekly")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("daily must have a DatetimeIndex to resample")
    out = s.add(1.0).resample(rule).prod().sub(1.0)
    return pd.Series(out.to_numpy(dtype=float), index=out.index,
                     name="weekly")


def longest_losing_week_streak(weekly: pd.Series) -> int:
    """Longest run of consecutive weeks with a strictly negative return.
    A zero week BREAKS the streak (it is not a losing week)."""
    best = run = 0
    for v in np.asarray(weekly, dtype=float).ravel():
        run = run + 1 if v < 0.0 else 0
        best = max(best, run)
    return int(best)


def worst_rolling_window(weekly: pd.Series, *, window: int = 8) -> float:
    """Worst compounded return over any `window` consecutive weeks, as a
    fraction. Returns 0.0 when there are fewer than `window` weeks -- an
    unmeasurable stretch must not read as a catastrophic one."""
    x = np.asarray(weekly, dtype=float).ravel()
    if x.size < int(window):
        return 0.0
    rolled = pd.Series(1.0 + x).rolling(int(window)).apply(
        np.prod, raw=True).dropna()
    if rolled.empty:
        return 0.0
    return float(rolled.min() - 1.0)


def terminal_wealth(daily: pd.Series) -> float:
    """Compounded terminal wealth multiple, ``prod(1 + r)``. 1.0 for an
    empty series (no sessions, no change)."""
    x = np.asarray(daily, dtype=float).ravel()
    if x.size == 0:
        return 1.0
    return float(np.prod(1.0 + x))


# The account whose path the four risk refusals measure. A benchmark-plus-overlay family is judged on the OVERLAY-ONLY
# EXCESS account, not on the whole SPY-core account. See `risk_refusals`.
RISK_ACCOUNT_BASES = ("whole_account", "overlay_only_excess")

# What a curve builder puts under the `account_kind` key of its curve dict to
# declare that its account is a benchmark held at all times PLUS an overlay, so
# the risk layer is measured on the overlay's own excess. Read by
# `abi1_driver.evaluate_family`; a curve without the key is scored on the
# whole account.
ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY = "benchmark_plus_overlay"


def risk_refusals(account_daily: pd.Series, benchmark_daily: pd.Series, *,
                  absolute_dd_cap: float = ABSOLUTE_MAX_DRAWDOWN_CAP,
                  worst_window_weeks: int = 8,
                  account_basis: str = "whole_account") -> dict:
    """Layer 2. Four pass/fail refusals on the account's own curve, measured
    over the sessions the account and benchmark SHARE.

    Applied only AFTER layer 1, never ranked, never blended, never a
    denominator. That is deliberate: a candidate can beat the benchmark's
    risk-adjusted ratio while compounding very slowly on tiny exposure, so
    any ratio-shaped ranking mechanically
    promotes the cash-heaviest candidate. Risk is a constraint; return is the
    objective; never divide one by the other.

    Three thresholds are the BENCHMARK's own value over the same sessions, so
    no constant is invented. The fourth, `absolute_dd_cap`, is a fixed constant.

    `account_basis` names the account the four refusals measure:

      "whole_account" (default): `account_daily` itself.
      "overlay_only_excess": the daily ARITHMETIC excess `account - benchmark`
          over the shared sessions, compounded as its own account. For a
          benchmark-plus-overlay family (SPY
          held at all times plus an overlay): the 0.35 absolute cap and the
          three SPY-relative refusals apply to what the overlay itself earns
          and loses, not to a whole-account curve that holds SPY through 2008,
          where SPY's own drawdown alone is about 55 percent, so the whole
          account clears the cap only if the overlay offsets most of it. For such
          an account the excess is the overlay's dollars (net of costs and
          financing) over the prior close's account equity, less the
          realization-tax drag charged on exit sessions: it is measured in
          units of WHOLE-account equity, not of the overlay's own committed
          capital. The three SPY-relative refusals then compare that excess
          account with SPY ITSELF. Over a window that holds a large SPY
          drawdown a small overlay is expected to be much less constrained by
          them than a whole SPY-core account would be, but they are not inert:
          on a calm benchmark they fire (pinned by tests), and each overlay
          driver writes its run's refusal numbers into its ledger note (see
          `overlay_risk_report.risk_layer_note`).
          Terminal wealth stays on the whole account either way: it is never
          a gate.

    Returns a dict of four `{passed, account, benchmark|cap, ...}` blocks
    plus `all_refusals_passed`, which deliberately EXCLUDES terminal wealth:
    it is reported but never the sole grounds for a verdict.
    `account_basis` is echoed in the result, and "account" in the drawdown,
    streak and stretch blocks is the measured (basis) account.
    """
    if account_basis not in RISK_ACCOUNT_BASES:
        raise ValueError(
            f"account_basis must be one of {RISK_ACCOUNT_BASES}; "
            f"got {account_basis!r}")
    joined = _calendar_aligned_returns(account_daily, benchmark_daily)
    # INTERIOR holes must fail closed, not be dropped. Dropping them SPLICES
    # the account curve, and drawdown/streak/worst-stretch are all path
    # statistics: a deep drawdown can
    # read as 0.0 -- passing the absolute cap -- if the crash
    # sessions are exactly the ones the benchmark is missing. A LEADING
    # gap is different and stays allowed: it is the expected benchmark-start
    # truncation (SPY starts 1993; some pools trade back to 1962), and
    # truncating a prefix does not splice anything.
    if len(joined):
        b_valid = joined["b"]
        first, last = b_valid.first_valid_index(), b_valid.last_valid_index()
        if first is not None:
            span = joined.loc[first:last]
            holes = int(span["b"].isna().sum() + span["a"].isna().sum())
            if holes:
                raise ValueError(
                    f"{holes} interior gap(s) between {first.date()} and "
                    f"{last.date()}; dropping them would splice the equity "
                    "curve and understate drawdown. Fill or trim the series.")
    joined = joined.dropna()
    whole, b = joined["a"], joined["b"]
    # The excess is formed AFTER the interior-gap guard and the dropna above,
    # so this basis cannot open a way round either.
    a = whole if account_basis == "whole_account" else whole - b
    a_wk, b_wk = weekly_returns(a), weekly_returns(b)

    a_dd, b_dd = max_drawdown(a), max_drawdown(b)
    cap = -abs(float(absolute_dd_cap))
    drawdown = {
        "passed": bool(a_dd >= b_dd and a_dd >= cap),
        "account": a_dd, "benchmark": b_dd, "absolute_cap": cap,
        "relative_passed": bool(a_dd >= b_dd),
        "absolute_passed": bool(a_dd >= cap),
    }
    a_streak, b_streak = (longest_losing_week_streak(a_wk),
                          longest_losing_week_streak(b_wk))
    red_week_streak = {"passed": bool(a_streak <= b_streak),
                       "account": a_streak, "benchmark": b_streak}
    a_worst = worst_rolling_window(a_wk, window=worst_window_weeks)
    b_worst = worst_rolling_window(b_wk, window=worst_window_weeks)
    # DOWNSIDE ONLY, clamped at 0.0 -- the same convention `max_drawdown`
    # already applies to itself. Comparing the raw values would let this
    # refusal fire when NEITHER series ever lost money over any 8-week
    # stretch (both "worst" values positive), which is a RETURN comparison
    # wearing a risk gate's clothes. That would defeat the design
    # twice over: risk is a constraint, return is the objective, and
    # terminal wealth is deliberately excluded from the verdict -- excluding
    # it at the front door while admitting it here would be no exclusion.
    # Raw values are still reported, because "worst 8 weeks was +0.4%" is
    # useful to a reader even when it cannot refuse.
    a_loss, b_loss = min(a_worst, 0.0), min(b_worst, 0.0)
    worst_stretch = {"passed": bool(a_loss >= b_loss),
                     "account": a_worst, "benchmark": b_worst,
                     "account_loss": a_loss, "benchmark_loss": b_loss,
                     "window_weeks": int(worst_window_weeks)}
    # Terminal wealth is always the WHOLE account against the benchmark: it is
    # never a gate, and "the excess account's wealth against SPY's wealth"
    # would be a meaningless comparison of two different-sized accounts.
    a_term, b_term = terminal_wealth(whole), terminal_wealth(b)
    terminal = {"passed": bool(a_term > b_term),
                "account": a_term, "benchmark": b_term}

    return {
        "account_basis": account_basis,
        "n_sessions": int(len(joined)), "n_weeks": int(len(a_wk)),
        "drawdown": drawdown, "red_week_streak": red_week_streak,
        "worst_stretch": worst_stretch, "terminal_wealth": terminal,
        # Terminal wealth is EXCLUDED on purpose.
        "all_refusals_passed": bool(drawdown["passed"]
                                    and red_week_streak["passed"]
                                    and worst_stretch["passed"]),
    }


# ----------------------------------------------------------- entry point

def evaluate(*, strategy_daily: pd.Series, in_market: pd.Series,
             benchmark_daily: pd.Series, cash_daily: pd.Series,
             n_trials: float, sr_variance: Optional[float],
             trades_df: Optional[pd.DataFrame] = None,
             tax_policy: TaxPolicy = "realization",
             tax_haircut_rate: float = constants.STCG_TAX_RATE,
             ltcg_tax_rate: float = constants.LTCG_TAX_RATE,
             min_sessions: int = MIN_ACCOUNT_SESSIONS,
             absolute_dd_cap: float = ABSOLUTE_MAX_DRAWDOWN_CAP,
             dsr_pass_threshold: float = constants.DSR_PASS_THRESHOLD,
             holdout_sessions: int = constants.ABI1_HOLDOUT_SESSIONS,
             holdout_psr_threshold: float = constants.DSR_PASS_THRESHOLD,
             risk_account_basis: str = "whole_account"
             ) -> dict:
    """Run ABI-1 end to end and return everything a caller needs to record.

    Order is load-bearing: layer 1 decides, layer 2 refuses, and the
    out-of-sample holdout confirms. `verdict` is "survived" only when the
    account clears the in-sample DSR bar, passes the three risk refusals,
    AND passes `holdout_gate` -- see `VERDICTS` for the full order. Every
    layer is still computed and reported when an earlier one fails, because
    a reader wants to see why.

    THE HOLDOUT CHANGES WHAT LAYER 1 SCORES. `split_holdout` carves the last
    `holdout_sessions` joint sessions off the end, and layer 1 is scored on
    what remains -- the holdout is out of sample or it is not a holdout.
    Fewer in-sample sessions raises the DSR bar slightly; that is the price of the test, not a
    regression. When the split is REFUSED (too little history), layer 1
    falls back to scoring the FULL window -- so its number stays comparable
    with an unsplit score -- but the verdict then caps
    at "holdout_not_evaluable" and cannot reach "survived".
    `window["holdout_split_applied"]` says which of the two happened, and
    `window["in_sample_sessions"]` says what layer 1 actually scored.

    The four risk refusals are NOT re-run on the holdout alone; they
    measure the full joint window -- see the holdout section's header
    comment.

    `risk_account_basis` picks the account the four refusals measure (see
    `risk_refusals`): "whole_account" (default) or "overlay_only_excess" (for a
    benchmark-plus-overlay family). It changes ONLY layer 2: layer 1 and the
    holdout are computed on the same excess series either way.

    `cash_daily` MUST come from one of the two producers
    admitted: `sweep_cash_daily` for an account holding actual cash (the
    default convention), or `parked_cash_daily` for one that parks idle
    capital in bills or a bond fund, which charges that choice's spread and
    interest tax. `daily_cash_rate`'s raw bill yield is NOT admissible on
    its own -- it credits a patient candidate many times the sweep rate for
    doing nothing. Nothing here can VERIFY which producer a caller used, so
    the caller must record the convention alongside the score; this
    signature carries no field for it.

    `tax_policy="realization"` charges tax at each trade's exit bar through
    `netted_tax_factors` (requires `trades_df`): losses NET against later
    gains with an unlimited dollar-unit carryforward, and a round trip held
    beyond `constants.LONG_TERM_HOLDING_DAYS` is charged `ltcg_tax_rate`
    rather than `tax_haircut_rate`. Both
    rates are arguments so a bracket change is a call-site change, not a
    design change; read `netted_tax_factors` for what the model does and does not
    reproduce (no wash sales, no annual offset limit, two character-matched
    short/long carryforward pools with the cross-over deferred).

    `tax_policy="none"` charges nothing -- correct when `trades_df` already
    carries a post-tax `net_ret`; on a pre-tax curve it flatters high-turnover
    candidates. There is no per-day mark-to-market option: haircutting every
    up-day is a strictly harsher and differently-shaped operation, not the
    same rule expressed daily.

    Taxing is a TWO-PASS computation, and it has to be: the netted walk needs
    the account's PRE-TAX curve to size each loss in dollars, and the taxed
    curve is then that same curve with the resulting factors applied.
    """
    if tax_policy not in ("realization", "none"):
        raise ValueError(f"unknown tax_policy {tax_policy!r}")
    account = account_daily_returns(
        strategy_daily=strategy_daily, in_market=in_market,
        cash_daily=cash_daily)
    if tax_policy == "realization" and (tax_haircut_rate or ltcg_tax_rate):
        if trades_df is None:
            raise ValueError(
                'tax_policy="realization" needs trades_df to locate exit '
                'bars; pass tax_policy="none" for an untaxed curve')
        factor = netted_tax_factors(pretax_account_daily=account,
                                    trades_df=trades_df,
                                    short_rate=tax_haircut_rate,
                                    long_rate=ltcg_tax_rate)
        account = account_daily_returns(
            strategy_daily=strategy_daily, in_market=in_market,
            cash_daily=cash_daily, tax_factor=factor)
    excess = log_excess(account, benchmark_daily)
    # The holdout is carved off the END of the SCORED sessions (the joint
    # index), not off the account's own span: sessions the benchmark cannot
    # cover were never scoreable and must not be spent as holdout.
    split = split_holdout(excess.index, holdout_sessions=holdout_sessions,
                          min_in_sample=min_sessions)
    in_sample_excess = excess.loc[split["in_sample"]]
    layer1 = score_account(in_sample_excess, n_trials=n_trials,
                           sr_variance=sr_variance,
                           min_sessions=min_sessions)
    layer2 = risk_refusals(account, benchmark_daily,
                           absolute_dd_cap=absolute_dd_cap,
                           account_basis=risk_account_basis)
    if split["ok"]:
        h_idx = split["holdout"]
        holdout = _holdout_from_excess(
            excess.reindex(h_idx), required_sessions=holdout_sessions,
            psr_threshold=holdout_psr_threshold)
    else:
        holdout = _holdout_refusal(split["refusal"], split["reason"],
                                   n=split["n_holdout"],
                                   required=int(holdout_sessions),
                                   psr_threshold=holdout_psr_threshold)
    if not layer1["scored"]:
        verdict = "not_scoreable"
    elif layer1["dsr_value"] < dsr_pass_threshold:
        verdict = "cut_dsr"
    elif not layer2["all_refusals_passed"]:
        verdict = "cut_risk_refusal"
    elif not holdout["evaluated"]:
        # Could not be tested out of sample. NOT the same as tested and
        # failed, and never a pass: an untested candidate has not
        # earned the verdict, it has only avoided the test.
        verdict = "holdout_not_evaluable"
    elif not holdout["passed"]:
        verdict = "cut_holdout"
    else:
        verdict = "survived"
    # Deployment is reported over the SCORED window -- the sessions the
    # verdict was actually computed on -- not the account's full span. The
    # two differ whenever the benchmark starts later than the strategy (the
    # benchmark-start truncation: SPY from 1993, some pools back to 1962), and either
    # direction of error is possible: a candidate deployed on 100% of scored
    # sessions could report a middling figure, and one deployed on 0% of them
    # likewise. Reporting the wrong-window figure next
    # to a verdict would misdescribe the account the verdict was computed
    # on. The full-window figure is kept beside
    # it, named, rather than silently dropped.
    # `time_in_market` means the JOINT window
    # (in-sample + holdout), not layer 1's in-sample half. Deployment is a
    # description of the account, not of the score, and re-pointing it at
    # the split would silently change a reported figure's meaning. The
    # split's own counts are in `window` below.
    mask = _validated_mask(in_market, account.index)
    scored_idx = excess.index
    covered_idx = (account.index[(account.index >= scored_idx[0])
                                  & (account.index <= scored_idx[-1])]
                   if len(scored_idx) else account.index[:0])
    deployed_scored = (float(mask.reindex(covered_idx, fill_value=False).mean())
                       if len(covered_idx) else 0.0)
    deployed_full = float(mask.mean()) if len(account) else 0.0
    return {
        "construction": "ABI-1",
        "verdict": verdict,
        "layer1": layer1,
        "layer2": layer2,
        # Reported whether or not it passes, and whether or not it could run
        # at all, so a reader always sees the number behind the verdict.
        "holdout": holdout,
        "account_daily": account,
        "log_excess": excess,
        "time_in_market": deployed_scored,
        "time_in_market_full_window": deployed_full,
        "window": {
            "account_sessions": int(len(account)),
            "scored_sessions": int(len(scored_idx)),
            "sessions_dropped_to_benchmark": int(len(account) - len(covered_idx)),
            "sessions_compounded_to_benchmark": int(len(covered_idx) - len(scored_idx)),
            "benchmark_truncated": bool(len(covered_idx) < len(account)),
            # The holdout split. `in_sample_sessions` is what layer 1
            # actually scored -- equal to `scored_sessions` only when the
            # split was refused.
            "holdout_split_applied": bool(split["ok"]),
            "in_sample_sessions": int(split["n_in_sample"]),
            "holdout_sessions": int(split["n_holdout"]),
            "holdout_sessions_required": int(split["holdout_sessions_required"]),
        },
        "tax_policy": tax_policy,
        "tax_haircut_rate": float(tax_haircut_rate)
        if tax_policy == "realization" else 0.0,
        # The short-term rate above and this long-term one are reported
        # separately because they rank family TYPES differently: a
        # high-turnover family may have no long-term trades, while a
        # trend-follower can realize most of its gains long-term.
        "ltcg_tax_rate": float(ltcg_tax_rate)
        if tax_policy == "realization" else 0.0,
        "tax_loss_netting": tax_policy == "realization",
        "dsr_pass_threshold": float(dsr_pass_threshold),
    }
