"""Batch5 candidate: gold_copper_ratio_fear_valuation_spy_overlay (bespoke driver;
run only via scripts/run_gold_copper_overlay_abi1.py, never the generic
scripts/run_abi1_mining.py dispatch -- see "Dispatch safety").

PRE-REGISTERED RULE. Every constant below was fixed before THIS
PIPELINE computed any return of this account, and none is to be tuned after a
result. That is NOT the same as chosen blind: SIGNAL-ONLY descriptive numbers
on these same series were looked at first (see "What was and was not chosen
blind" below)::

    account = 100 percent SPY at all times (the benchmark series itself)
              + a margin-financed LONG overlay of extra SPY held ONLY while the
                gold-to-copper fear-valuation state is HIGH
    overlay = total notional 0.25 of account equity in SPY, borrowed at the
              ^IRX bill rate (no spread)
    x_M     = ln(GC=F close / HG=F close) read at the last ACTUAL NYSE session
              strictly before each scheduled month-end M (the one-session lag)
    z_M     = (x_M - mean of the last 60 monthly reads incl. x_M) / sample
              stdev of the same 60 reads   (a 60-month, i.e. five-year, window)
    state   = HIGH when z_M >= 0
    window  = consecutive HIGH month-ends MERGED into one window: entry at the
              close of the first HIGH scheduled month-end, exit at the close of
              the first later scheduled month-end that is not HIGH

Mechanism and sources. Kim, D., Roh, T.-Y. and Yoon, S.-J. (October 2022 working
paper), with Yu You added in the current SSRN listing, "When Gold Meets Copper:
A Comprehensive Look at the Relative Value of Gold on Global Stock Markets",
SSRN abstract 5677761 (Roh, Kim, Yoon, You; posted October 2025 from the October
2022 draft). Read from the paper PDF: 23
industrialized countries, January 1990 to mid-2019, Datastream monthly SPOT
prices, the predictor is the LOG gold-copper ratio LEVEL in linear predictive
regressions at 6-month and 1-year horizons, coefficients positive (a higher
ratio precedes a higher subsequent equity excess return), out-of-sample
Campbell-Thompson tests January 2000 to June 2019, and the predictability is
stated to come mainly from low-sentiment (bad-sentiment) periods; the authors
test seven gold ratios and report the gold-copper one as strong. Journal
publication status is UNVERIFIED (an SSRN working paper only). The paper reports
mean-variance forecast allocations, NOT a threshold rule: the 60-month z state,
the z >= 0 threshold, the 0.25 notional, the one-session lag, the merged window
and every hole rule below are THIS PIPELINE'S OWN pre-registered constants, not
paper parameters. Parnes (2024), North American Journal of Economics and
Finance 69(A), finds the copper-to-gold ratio is NOT a fast leading indicator
(a lead of only a few days, for the 10-year Treasury yield) -- side evidence
that the ratio is a slow valuation-type state, which is how it is used here.
The paper used spot prices; this rule uses COMEX front-month CONTINUOUS futures
closes from Yahoo (a vendor splice, unadjusted across contract rolls), a real
instrument difference (the 2025 copper tariff episode opened a large COMEX
premium over the London spot).

Amendments applied (each one binding and disclosed here)
--------------------------------------------------------
1. WINDOWS. ``mining.long_overlay`` RAISES on windows that share an endpoint
   (month-to-month HIGH windows touch whenever consecutive month-ends are
   HIGH), so consecutive HIGH month-ends are MERGED into one window. The
   overlay therefore holds a fixed share count, set at the entry equity, for the
   whole merged window (the borrowed principal is fixed too), and its share of account
   equity DRIFTS for as long as the merged window lasts. With the overlay's daily P&L reinvested in the SPY core
   (the builder's own account identity) the share is exactly
   0.25 / (1 + 0.25 * sum of (1 - P_prev / P)) over the window, about
   0.25 / (1 + 0.25 ln x) for SPY at x times its entry price (financing and
   costs aside; a test pins the identity): 0.21 after a doubling, 0.30 after a
   halving, 0.32 after a 60 percent fall, so the drift is mild when SPY rises and
   larger when it falls, and it matters most inside a window entered before a
   crash. A monthly-resizing builder would be new
   engine work and is NOT built; the drift is disclosed, not corrected. It
   applies equally to the main arm and to both control arms.
2. CONTROLS (report-only, no ledger row of their own). (a) The mandatory
   ALWAYS-ON arm: the SAME construction, financing, costs and tax policy with the
   state forced HIGH on every decision month, i.e. one window from the first
   decision month-end to the end of the data. It differs from the main arm in two
   ways, both disclosed limits: it rides through every episode AND the off
   months, and its single window (about 21 years, so long-term) pays no modelled tax (long-term at the 0.0 rate), not merely deferred: the
   pipeline's long-term rate is 0.0 by pipeline design (the comment above
   ``constants.LTCG_TAX_RATE``), and the same zero rate leaves most of the main
   arm's window-days untaxed too. The interpretation rule
   below compares the main arm's TAXED Sharpe with the always-on arm's TAXED
   Sharpe, which is tax-asymmetric (the two arms hold different shares of
   long-term, untaxed window-days); every run therefore prints every arm's UNTAXED
   daily-excess Sharpe, the untaxed-basis comparison (main untaxed versus always-on
   untaxed) beside it, and the count of windows and of window calendar days held
   over 365 days for the main arm and the controls, computed from that run's own
   trades; the pre-declared interpretation rule itself is unchanged. The overlay's share of equity drifts the same way in every arm
   (see 1).
   (b) The RANDOM-WINDOW arm: a seeded, deterministic set of 500 draws that keep
   the main arm's episode count, its exact multiset of episode lengths and so its
   HIGH share, placed at random non-adjacent positions; because the windows
   reset the same way, this is the exposure-matched comparison. The interpretation
   rule is pre-declared: a DSR pass is evidence about the SIGNAL only if the main
   arm's daily-excess Sharpe exceeds the always-on arm's AND sits at or above the
   95th percentile of the random-window draws; otherwise it is attributed to
   equity exposure, not to the signal.
3. CALENDAR. The month-end M is the exchange's SCHEDULED last session (regular
   holidays only, the treasury sibling's calendar, unscheduled closures such as
   Sandy 2012 NOT removed), never the last row in the data. Futures are read
   as-of: a month whose read date has no futures row is tolerated ONLY in the
   warm-up (before the first defined z), with a causal last-row-at-or-before read
   at most 4 calendar days stale; ANY missing read on or after the first defined z
   RAISES (``futures_read_hole``) and a warm-up read staler than the limit raises
   (``futures_asof_too_stale``). Nothing is patched, forward-filled after the
   warm-up, or trimmed.
4. SETTLEMENT VERSUS LATE PRINT (one-session lag). The Yahoo =F daily Close was
   NOT verifiable as the COMEX settlement over the whole history. Over a recent
   sample of a few weeks of sessions the gold Close tracks the 13:30 ET print
   tightly and the copper Close tracks the 13:00 ET print more noisily, consistent
   with the COMEX settlement times and not with the
   16:00 or 17:00 prints; and the Close lies outside the day's High-Low range on
   430 gold and 275 copper rows, all in 2001-2011 and none from 2012, which is the
   pit-era pattern of a settlement price outside a pit range but says nothing
   about the electronic era. Because the earlier history cannot be verified, the
   more conservative option is taken: the ONE-SESSION LAG. The read is the last
   ACTUAL NYSE session strictly before M (Sandy: 26 October 2012, not the 30th),
   so no futures bar on or after M is ever used and a settlement-versus-late-print
   ambiguity cannot leak the month-end session into the entry decision. A frozen
   sha256-pinned cache of the vendor splice is read instead of the network
   (``freeze_futures`` / ``load_frozen_futures``): the vendor revises its latest
   bar after the fact (two fetches a few hours apart can differ on the final
   bar), so the bytes scored must be the bytes hashed.
5. THE 2025 COMEX TARIFF DISLOCATION is DISCLOSE-ONLY: the largest one-day
   change of the log ratio was 0.251 on 31 July 2025, coinciding with the record
   one-day copper drop after refined copper was exempted from the tariff.
   Under the one-session lag the July 2025 read
   already IS the 30 July close, the dislocation day is no month's read, and the
   August 2025 read sits after it inside the 60-read window. No winsorising is
   applied.
6. DISCLOSURES. The state is not a re-labelled VIX-type fear state. The
   evidence is thin: a slow monthly state forms only a handful of HIGH episodes,
   a few long episodes can carry most of the in-sample HIGH months, and a recent
   run of HIGH month-ends can fill most of the ABI-1 holdout, so the holdout is
   read as DESCRIPTIVE only and cannot separate signal from bull-market exposure.
   The daily-iid DSR treats thousands of sessions as independent bets where the
   real evidence is a few independent episodes. The
   frozen splice also carries thin vendor prints (zero-volume rows) at 9 gold and 1 copper read dates (11 and 2 counting the as-of warm-up reads), all
   through 2005-08-30 (detail in the catalog); they are kept, not patched.
7. SINGLE SHOT. The family is scored once (one permanent ledger row), with real
   quoted costs; the driver's ``--dry-run`` path writes no permanent row.

What was and was not chosen blind (pre-registration honesty)
------------------------------------------------------------
SIGNAL-ONLY numbers on these same futures were measured before the constants
were frozen (coverage, the state's HIGH share and switch count, correlation of
the state with a VIX state and with SPY momentum, and agreement between
alternative windows); no signal-conditional SPY return was computed, to keep
the blind pre-registration untainted, and the build computed no return of the
account. An always-on overlay is a strong unconditional baseline, which is why
a pass that does not beat the always-on and random-window controls means
nothing. Not chosen blind: the 60-month window and z >= 0 threshold were
declared after a look at the signal's coverage and state counts (alternative windows were compared for
state agreement only); the one-session lag was chosen AFTER the settlement probe.
The 0.25 notional was chosen for Reg-T slack, not from any look at its effect.
No constant was tuned against an ABI-1 score and none may be after one. Those
looks ran outside the trial ledger, so the multiple-testing count DSR applies is
NOT charged for them.

Risk layer (the risk-basis rule)
-----------------------
ABI-1's four risk refusals (the 0.35 absolute drawdown cap, drawdown no worse
than SPY's, longest losing-week streak no longer than SPY's, worst 8-week stretch
no worse than SPY's) are measured for a benchmark-plus-overlay family on the
overlay-only excess account (the account's daily arithmetic excess over SPY: the
overlay's own dollars, net of costs and financing, over the prior close's account
equity, less the realization-tax drag), not on the whole SPY-core account. The
curve declares ``account_kind == "benchmark_plus_overlay"`` (``mining.long_overlay``
sets it) and the driver refuses to score if that goes missing or if the account
differs from SPY on any session the overlay is off. The expected overlay-only
drawdown is bounded near a quarter of SPY's own drawdown; a dry run shows
whether the guard would false-trip before any permanent row is written.

Interpretation choices
----------------------
* Account calendar: SPY's own sessions from the FIRST DEFINED-z month-end (the
  60th monthly read) to the last session. Before it the account is plain SPY and
  its excess exactly zero; slicing it off keeps a decade of zero-excess sessions
  from diluting the score. This is the pipeline's convention, not a conservative
  choice: its direction is favourable to the DSR (fewer zero-excess sessions in
  the daily-iid count and, when the Sharpe is positive, a higher daily-excess Sharpe than the
  whole-history calendar would give). The core and the benchmark are still built on SPY's
  FULL history (identical to the evaluator's benchmark series).
* Data calendar: validated, never trimmed or patched, with the treasury family's
  validators (``mining.long_overlay.common_calendar`` and that module's
  ``validate_session_coverage``): a session the exchange held that SPY lacks is
  ``missing_exchange_session``; a stray bar is ``bar_on_closed_day``.
* A window still open at the end of the data (the last month-end is HIGH, so the
  exit month-end lies after the last bar) is closed at the last close (the
  engine's ``end_of_data`` path) and reported as ``truncated_final``; a window
  entered on the very last bar opens nothing (``entered_on_last_bar``). Unlike
  the treasury sibling's fixed three-session trade, a state-driven window is
  ordinarily open at the data's end, so the driver REFUSES a real score on a
  truncated final window by default (the closing bar could be a live partial
  bar) and scores it only when the caller explicitly accepts that, in which case
  the ledger note records the acceptance. The note also states whether the last
  SPY bar used may be a live partial bar (its date against the run's tested date;
  a bar dated on or after the run date is treated as possibly partial) and records
  the installed ``exchange_calendars`` version with a sha256 of the decision
  month-ends, read dates and terminal exit (``schedule_provenance``), because every
  one of those dates is a function of that package.
* Account type: taxable only (the overlay is borrowed, which needs a margin
  account).
* Costs: the SPY overlay is charged a full round trip of the REAL quoted SPY
  spread once per window; financing is the bare ^IRX rate with NO spread
  invented, which FLATTERS the strategy.
* Tax: windows held over 365 days are untaxed at the pipeline's 0.0 long-term rate
  (a documented, flagged assumption, the comment above
  ``constants.LTCG_TAX_RATE``), which also FLATTERS the strategy; the count of
  such windows is printed in every ledger note.

Dispatch safety
---------------
``SHAPE = "single_long"`` is a contract LABEL only (none of the six registered
shapes describes a benchmark-plus-overlay account). ``TRADED`` lists the SPY
overlay leg; ``SIGNAL`` lists the two futures and ^IRX, series that are never
traded (the bill rate can print at or below zero and is loaded without the
positivity gate). ``make_signal_fn`` exists only to satisfy the contract and
ALWAYS raises, so the generic dispatch (which would otherwise score a plain long
SPY position) turns it into a clean ``compute_error`` skip.
"""
from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np
import pandas as pd

from mining import long_overlay as _overlay
from mining.long_overlay import CalendarCoverageError
from mining.batch5 import treasury_month_end_index_demand_overlay as _cal

FAMILY_ID = "gold_copper_ratio_fear_valuation_spy_overlay"
SHAPE = "single_long"              # contract label only -- see "Dispatch safety"
TRADED = ["SPY"]
SIGNAL = ["GC=F", "HG=F", "^IRX"]  # signal-only series, never traded
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY; here SPY is also the core.
BENCHMARK = "SPY"

# Pre-registered constants, each this pipeline's own choice:
#   OVERLAY_NOTIONAL -- 0.25 of equity, chosen for Reg-T slack, not for the score
#   Z_WINDOW_MONTHS  -- five years of monthly reads: the ratio is near a unit root
#                       and shifted after 2008, so it is standardised on a fixed window
#   SIGNAL_LAG_SESSIONS -- the one-session lag (amendment 4)
#   MAX_ASOF_STALENESS_DAYS -- one missing session (a Monday read whose Friday-to-
#                       Monday span is three days, plus one more)
OVERLAY_NOTIONAL = 0.25
Z_WINDOW_MONTHS = 60
Z_THRESHOLD = 0.0
SIGNAL_LAG_SESSIONS = 1
MAX_ASOF_STALENESS_DAYS = 4
RANDOM_CONTROL_DRAWS = 500
RANDOM_CONTROL_SEED = 2026_0924
RANDOM_CONTROL_PERCENTILE_BAR = 0.95

# Copied byte-for-byte into the family's catalog entry in data/mining/catalog.json
# (the driver's parameter-drift check refuses a run if the two differ).
PARAMETERS = {
    "position": "long_overlay_on_benchmark",
    "core": "SPY fully invested at all times (the benchmark series itself)",
    "overlay_legs": list(TRADED),
    "overlay_notional_fraction": OVERLAY_NOTIONAL,
    "signal": ("x = ln(GC=F close / HG=F close) read at the last actual NYSE session strictly "
               "before each scheduled month-end; z = (x - mean of the last 60 monthly reads "
               "including this one) / sample stdev of the same 60; state HIGH when z >= 0"),
    "z_window_month_ends": Z_WINDOW_MONTHS,
    "state_threshold_z": Z_THRESHOLD,
    "signal_lag_actual_sessions": SIGNAL_LAG_SESSIONS,
    "max_asof_staleness_calendar_days": MAX_ASOF_STALENESS_DAYS,
    "window": (
        "consecutive HIGH month-ends merged into ONE window: entry at the "
        "close of the first HIGH scheduled month-end, exit at the close of "
        "the first later scheduled month-end that is not HIGH; the overlay "
        "share count is fixed at the entry equity for the whole merged "
        "window, so its share of account equity drifts"
    ),
    "calendar": ("exchange_calendars XNYS scheduled sessions: weekdays minus regular holidays; "
                 "unscheduled closures are not removed; the read date is the last ACTUAL session "
                 "before the month-end"),
    "hole_rule": ("a missing futures read is a causal as-of read (at most 4 calendar days stale) "
                  "only before the first defined z; any missing read from the first defined z on is "
                  "refused, never patched"),
    "financing_rate": ("^IRX 13-week bill coupon-equivalent yield of the last close strictly before "
                       "each session, on the borrowed overlay notional, actual calendar days over 365, "
                       "no spread"),
    "control_arms": ("report-only, no ledger row: an always-on overlay (same construction, state "
                     "forced HIGH) and 500 seeded random-window overlays with the main arm's episode "
                     "count and exact episode-length multiset"),
    "random_window_control_draws": RANDOM_CONTROL_DRAWS,
    "random_window_control_seed": "2026_0924",
    "benchmark": BENCHMARK,
}
# A conservative floor on the number of merged HIGH windows a full-history run
# must produce (a data sanity check, not a tuned value).
EXPECTED_MIN_TRADES = 10


def make_signal_fn(*args, **kwargs):
    """Always raises -- this family has no per-bar signal. See the module
    docstring's "Dispatch safety" section; the only legitimate entry point is
    ``scripts/run_gold_copper_overlay_abi1.py`` calling ``build_curve``."""
    raise RuntimeError(
        f"{FAMILY_ID}: this family must only be run through its own bespoke "
        "driver (scripts/run_gold_copper_overlay_abi1.py), never the "
        "generic scripts/run_abi1_mining.py dispatch -- it has no per-bar "
        "signal, and the generic path would score a plain long SPY position "
        "instead of the benchmark-plus-overlay account.")


# ------------------------------------------------------------------ errors

class FuturesReadError(ValueError):
    """A futures history this family refuses to score. A ``ValueError`` that
    also carries a machine-readable ``reason_code`` and the dates involved, so a
    driver reports a clean refusal instead of a traceback:

    * ``futures_read_hole`` -- a read date on or after the first defined z has no
      futures row (or the history ends before the last read);
    * ``futures_asof_too_stale`` -- a warm-up as-of read is staler than the limit;
    * ``futures_series_invalid`` -- a non-finite, non-positive, unsorted or
      duplicated series;
    * ``futures_history_too_short`` -- fewer than 60 monthly reads exist before
      SPY's last session, so z is never defined;
    * ``futures_window_zero_variance`` -- a 60-read window has no variation, so z is
      undefined (a refusal, never a guess);
    * ``futures_cache_missing`` / ``futures_cache_hash_mismatch`` -- the frozen
      cache is absent or its bytes no longer match the manifest's sha256."""

    def __init__(self, reason_code: str, message: str, *, missing_dates=()):
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.missing_dates = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in missing_dates})


# ----------------------------------------------------------------- calendar

def _normalized(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def month_end_schedule(start, end) -> pd.DatetimeIndex:
    """The SCHEDULED last session of every calendar month touched by
    ``[start, end]`` (whole months: a month-end may lie after ``end``), from the
    exchange's regular-holiday rules only (the treasury sibling's calendar).
    Takes dates only; it cannot read data, so a month-end is never inferred from
    the last row of a data file."""
    start, end = _normalized(start), _normalized(end)
    first = pd.Timestamp(start.year, start.month, 1)
    last = pd.Period(end, freq="M").end_time.normalize()
    sessions = _cal.scheduled_sessions(first, last)
    months = sessions.to_period("M")
    return pd.DatetimeIndex([sessions[months == p][-1] for p in months.unique()])


def read_dates_for(month_ends) -> pd.DatetimeIndex:
    """For each month-end M, the read date: the ``SIGNAL_LAG_SESSIONS``-th ACTUAL
    NYSE session strictly before M (the exchange's real session set, so a closure
    that was not scheduled -- Sandy -- is skipped). A fact about the past at M,
    hence causal."""
    ends = pd.DatetimeIndex([_normalized(m) for m in month_ends])
    if len(ends) == 0:
        return pd.DatetimeIndex([])
    actual = _cal.exchange_sessions(ends.min() - pd.Timedelta(days=30), ends.max())
    position = actual.searchsorted(ends, side="left")          # sessions strictly before M
    return pd.DatetimeIndex(actual[position - SIGNAL_LAG_SESSIONS])


def read_date_for(month_end) -> pd.Timestamp:
    return read_dates_for([month_end])[0]


# What the schedule hash covers, written into the ledger note verbatim.
SCHEDULE_HASH_FORMAT = ("sha256 (hex) of UTF-8 text, one line per decision 'month-end|read-date' then one "
                        "'terminal|exit-date' line, dates YYYY-MM-DD, every line newline-terminated")


def schedule_digest(decisions, read_dates, terminal) -> str:
    """The sha256 (hex) of the calendar-derived schedule of a run: every decision
    month-end with its read date (``SCHEDULE_HASH_FORMAT``), then the terminal exit.
    Every one of these dates is a function of the installed ``exchange_calendars``
    package, so a package upgrade that edited historical exchange holiday data would
    silently move them for a one-shot permanent row; the digest makes that visible."""
    decisions = [_normalized(d) for d in decisions]
    read_dates = [_normalized(r) for r in read_dates]
    if len(decisions) != len(read_dates):
        raise ValueError("schedule_digest needs one read date per decision month-end")
    lines = [f"{m:%Y-%m-%d}|{r:%Y-%m-%d}" for m, r in zip(decisions, read_dates)]
    lines.append(f"terminal|{_normalized(terminal):%Y-%m-%d}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def schedule_provenance(ctx: dict) -> dict:
    """The provenance of a prepared run's calendar: the installed
    ``exchange_calendars`` version and ``schedule_digest`` of its decision
    month-ends, read dates and terminal exit. Recorded in the ledger note."""
    import importlib.metadata
    decisions = list(ctx["decisions"])
    reads = [ctx["table"].loc[m, "read_date"] for m in decisions]
    terminal = _normalized(ctx["terminal_exit"])
    return {
        "exchange_calendars_version": importlib.metadata.version("exchange_calendars"),
        "n_decisions": len(decisions),
        "first_decision": decisions[0].strftime("%Y-%m-%d"), "last_decision": decisions[-1].strftime("%Y-%m-%d"),
        "terminal_exit": terminal.strftime("%Y-%m-%d"),
        "schedule_sha256": schedule_digest(decisions, reads, terminal),
        "hash_format": SCHEDULE_HASH_FORMAT,
    }


# ------------------------------------------------------- rolling z / state

def rolling_z(x) -> pd.Series:
    """z of each monthly read against the mean and sample stdev (ddof=1) of the
    last ``Z_WINDOW_MONTHS`` reads INCLUDING itself; NaN until the 60th read. A
    window with no variation has no defined z and raises rather than guessing."""
    x = pd.Series(x, dtype=float)
    mean = x.rolling(Z_WINDOW_MONTHS).mean()
    std = x.rolling(Z_WINDOW_MONTHS).std(ddof=1)
    if (std.dropna() <= 1e-12).any():
        raise FuturesReadError("futures_window_zero_variance",
                               "a 60-read window has zero variance, so z is undefined; "
                               "refusing to classify a state from it")
    return (x - mean) / std


def state_from_z(z) -> pd.Series:
    """HIGH (True) when z >= the pre-registered threshold; NA where z is undefined."""
    z = pd.Series(z, dtype=float)
    return (z >= Z_THRESHOLD).astype("boolean").mask(z.isna())


# ------------------------------------------------------------ futures reads

def _validated_close(series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        raise FuturesReadError("futures_series_invalid", f"{name}: expected a non-empty close Series")
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    series = pd.Series(series.to_numpy(dtype=float), index=index.normalize())
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise FuturesReadError("futures_series_invalid", f"{name}: index must be sorted and unique")
    values = series.to_numpy()
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise FuturesReadError("futures_series_invalid", f"{name}: closes must be finite and positive")
    return series


def _read_asof(series: pd.Series, name: str, read_date: pd.Timestamp, *, warm_up: bool, asof_log: list,
               missing: list):
    """The close of ``series`` on ``read_date``; ``(value, row_date)``. A missing
    row is a causal as-of read (the last row at or before the read date) only in
    warm-up and within ``MAX_ASOF_STALENESS_DAYS``; otherwise it is collected as a
    hole. Returns ``None`` when a hole was collected."""
    if read_date in series.index:
        return float(series.loc[read_date]), read_date
    if not warm_up:
        missing.append(read_date)
        return None
    prior = series.loc[:read_date]
    if prior.empty or (read_date - prior.index[-1]).days > MAX_ASOF_STALENESS_DAYS:
        have = "no earlier row" if prior.empty else f"the last row is {prior.index[-1]:%Y-%m-%d}"
        raise FuturesReadError(
            "futures_asof_too_stale",
            f"{name}: no row on the warm-up read date {read_date:%Y-%m-%d} and {have}, more than "
            f"{MAX_ASOF_STALENESS_DAYS} calendar days before it; refusing to read that far back",
            missing_dates=[read_date])
    row_date = prior.index[-1]
    asof_log.append({"series": name, "read_date": read_date.strftime("%Y-%m-%d"),
                     "row_date": row_date.strftime("%Y-%m-%d"),
                     "staleness_days": int((read_date - row_date).days)})
    return float(prior.iloc[-1]), row_date


def monthly_signal(gc_close, hg_close, *, last_session) -> pd.DataFrame:
    """The monthly signal table, indexed by SCHEDULED month-end M (every month
    from the first data month whose read date has data, through the last
    month-end at or before ``last_session``): ``read_date``, ``gc_row_date``,
    ``hg_row_date``, ``x`` (log gold/copper of the read-date closes), ``z``,
    ``state`` (HIGH True, NA until defined) and ``defined``.

    Reads are strictly causal (the last actual session before M; nothing on or
    after M is touched). ``table.attrs["asof_reads"]`` lists every warm-up as-of
    read. Raises ``FuturesReadError`` (see its docstring) on any hole from the
    first defined z on."""
    gc = _validated_close(gc_close, "GC=F")
    hg = _validated_close(hg_close, "HG=F")
    last_session = _normalized(last_session)
    first_data = max(gc.index[0], hg.index[0])
    ends = month_end_schedule(first_data, last_session)
    ends = ends[ends <= last_session]
    reads = read_dates_for(ends)
    keep = reads >= first_data
    ends, reads = ends[keep], reads[keep]
    records, asof_log, missing = [], [], []
    for k, (month_end, read_date) in enumerate(zip(ends, reads)):
        warm_up = k < Z_WINDOW_MONTHS - 1
        gc_hit = _read_asof(gc, "GC=F", read_date, warm_up=warm_up, asof_log=asof_log, missing=missing)
        hg_hit = _read_asof(hg, "HG=F", read_date, warm_up=warm_up, asof_log=asof_log, missing=missing)
        if gc_hit is None or hg_hit is None:
            continue
        records.append({"month_end": month_end, "read_date": read_date, "gc_row_date": gc_hit[1],
                        "hg_row_date": hg_hit[1], "x": float(np.log(gc_hit[0] / hg_hit[0]))})
    if missing:
        raise FuturesReadError(
            "futures_read_hole",
            f"{len(set(missing))} futures read date(s) on or after the first defined z have no row "
            f"({', '.join(sorted({d.strftime('%Y-%m-%d') for d in missing})[:20])}); refusing to "
            "patch, forward-fill or trim (refreeze the futures history, or the vendor's series has a hole)",
            missing_dates=missing)
    table = pd.DataFrame(records).set_index("month_end")
    table["z"] = rolling_z(table["x"])
    table["state"] = state_from_z(table["z"])
    table["defined"] = table["z"].notna()
    table.attrs["asof_reads"] = asof_log
    return table


# ---------------------------------------------------------- merged windows

def high_windows(month_ends, states) -> list:
    """``(entry, exit)`` per run of consecutive HIGH months. ``month_ends`` holds
    one date per decision PLUS a terminal exit date (the next scheduled month-end
    after the last decision), so it is one longer than ``states``. A run's entry
    is the close of its first HIGH month-end and its exit the close of the next
    month-end (the first that is not HIGH, or the terminal one); runs are
    separated by at least one non-HIGH month, so windows never touch."""
    states = [bool(s) for s in states]
    ends = list(month_ends)
    if len(ends) != len(states) + 1:
        raise ValueError("month_ends must hold one terminal exit date more than states")
    windows, start = [], None
    for i, high in enumerate(states):
        if high and start is None:
            start = i
        elif not high and start is not None:
            windows.append((ends[start], ends[i]))
            start = None
    if start is not None:
        windows.append((ends[start], ends[len(states)]))
    return windows


def episode_lengths(states) -> list:
    """Lengths, in months, of the runs of consecutive HIGH states."""
    lengths, run = [], 0
    for high in [bool(s) for s in states]:
        if high:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


# ------------------------------------------------------ frozen futures cache

FUTURES_FILES = {"GC=F": "yahoo_GC_F.csv", "HG=F": "yahoo_HG_F.csv"}
MANIFEST_NAME = "manifest.json"
_CSV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def real_yahoo_fetcher(symbol: str) -> pd.DataFrame:
    """The real network layer, used only by an explicit freeze."""
    import yfinance as yf
    raw = yf.Ticker(symbol).history(period="max", auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    return raw


def _serialize(symbol: str, frame, through: pd.Timestamp) -> tuple:
    """``(csv_bytes, cleaned_frame)`` for one vendor payload: tz-naive dates,
    rows after ``through`` dropped (the live partial bar), Close validated."""
    if not isinstance(frame, pd.DataFrame) or "Close" not in frame.columns:
        raise FuturesReadError("futures_series_invalid", f"{symbol}: vendor payload has no Close column")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    clean = frame.copy()
    clean.index = index.normalize()
    clean = clean.loc[clean.index <= through]
    for column in _CSV_COLUMNS:
        if column not in clean.columns:
            clean[column] = 0 if column == "Volume" else clean["Close"]
    clean = clean[_CSV_COLUMNS]
    _validated_close(clean["Close"], symbol)
    text = clean.to_csv(index_label="Date", date_format="%Y-%m-%d", lineterminator="\n")
    return text.encode("utf-8"), clean


def freeze_futures(data_dir, *, fetcher, through, retrieved_utc=None, overwrite: bool = False) -> dict:
    """Fetch GC=F and HG=F through ``fetcher(symbol)`` and FREEZE them under
    ``data_dir`` with a sha256 manifest: the
    vendor's continuous splice is unadjusted across rolls and can differ from one
    fetch to the next, so the bytes scored are the bytes hashed here. Bars dated
    after ``through`` (the live partial bar) are dropped. An existing frozen file is
    NEVER overwritten unless ``overwrite`` is set. Nothing is written unless both
    payloads validate."""
    data_dir = os.fspath(data_dir)
    through = _normalized(through)
    if not overwrite:
        existing = [n for n in [*FUTURES_FILES.values(), MANIFEST_NAME]
                    if os.path.exists(os.path.join(data_dir, n))]
        if existing:
            raise FileExistsError(f"refusing to overwrite the frozen futures cache: {', '.join(existing)}")
    stamp = retrieved_utc or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    payloads, files = {}, {}
    for symbol, name in FUTURES_FILES.items():
        raw, clean = _serialize(symbol, fetcher(symbol), through)
        payloads[name] = raw
        files[name] = {"symbol": symbol, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                       "rows": int(len(clean)), "first_date": clean.index[0].strftime("%Y-%m-%d"),
                       "last_date": clean.index[-1].strftime("%Y-%m-%d"), "retrieved_utc": stamp}
    manifest = {"schema_version": 1, "source": "yahoo", "through": through.strftime("%Y-%m-%d"),
                "note": ("frozen vendor continuous front-month splice, unadjusted across contract rolls; "
                         "auto_adjust True requested (no effect on futures); bars after `through` dropped"),
                "files": files}
    os.makedirs(data_dir, exist_ok=True)
    for name, raw in payloads.items():
        _atomic_write_bytes(os.path.join(data_dir, name), raw)
    _atomic_write_bytes(os.path.join(data_dir, MANIFEST_NAME),
                        (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    return manifest


def _atomic_write_bytes(path: str, raw: bytes) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(raw)
    os.replace(temporary, path)


def load_frozen_futures(data_dir) -> tuple:
    """``(gc_close, hg_close, manifest)`` from the frozen cache, each file's bytes
    verified against the manifest's sha256 BEFORE it is parsed. Raises
    ``FuturesReadError`` (``futures_cache_missing`` / ``futures_cache_hash_mismatch``
    / ``futures_series_invalid``)."""
    data_dir = os.fspath(data_dir)
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise FuturesReadError("futures_cache_missing",
                               "the frozen futures manifest is absent; run the explicit freeze first")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    closes = {}
    for symbol, name in FUTURES_FILES.items():
        path = os.path.join(data_dir, name)
        entry = manifest.get("files", {}).get(name)
        if entry is None or not os.path.exists(path):
            raise FuturesReadError("futures_cache_missing", f"the frozen futures file {name} is absent")
        with open(path, "rb") as handle:
            raw = handle.read()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise FuturesReadError(
                "futures_cache_hash_mismatch",
                f"{name} no longer matches the sha256 frozen in its manifest; refusing to score a "
                "history that is not the frozen splice")
        frame = pd.read_csv(io.BytesIO(raw))
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("Date")))
        closes[symbol] = _validated_close(frame["Close"], symbol)
    return closes["GC=F"], closes["HG=F"], manifest


# ------------------------------------------------------------------ curves

def prepare(spy_frame, irx_frame, gc_close, hg_close) -> dict:
    """Everything the curve builders share, computed ONCE: the monthly signal
    table, the decision month-ends, the terminal exit date, the state vector, the
    account calendar and the causal financing rate.

    The account lives on SPY's own sessions from the FIRST DEFINED-z month-end
    (the 60th monthly read) to SPY's last session: before it the account is plain
    SPY and its excess exactly zero, and slicing that off keeps a decade of
    zero-excess sessions from diluting the score. A month-end is a DECISION only
    if it is at or before SPY's last session. The data calendar is validated with
    the treasury sibling's validators and refused, never trimmed or patched
    (``mining.long_overlay.CalendarCoverageError``); a futures hole raises
    ``FuturesReadError``. Returns a dict (see the keys below)."""
    if not isinstance(irx_frame, pd.DataFrame) or "Close" not in irx_frame.columns:
        raise ValueError("irx_frame must be a DataFrame with a Close column")
    spy_index = pd.DatetimeIndex(spy_frame.index)
    table = monthly_signal(gc_close, hg_close, last_session=spy_index[-1])
    decided = table[table["defined"]]
    if decided.empty:
        raise FuturesReadError(
            "futures_history_too_short",
            f"no decision month: fewer than {Z_WINDOW_MONTHS} monthly futures reads exist before SPY's last "
            f"session ({spy_index[-1]:%Y-%m-%d}), so z is never defined; refusing to score")
    decisions = pd.DatetimeIndex(decided.index)
    first = decisions[0]
    if first < spy_index[0]:
        raise CalendarCoverageError(
            "spy_history_starts_after_first_decision",
            f"SPY's history starts {spy_index[0]:%Y-%m-%d}, after the first decision month-end "
            f"{first:%Y-%m-%d}; refusing to shift the first window (refetch a longer SPY history)")
    spy_leg = spy_frame.loc[first:]
    idx = _overlay.common_calendar(spy_frame, {"SPY": spy_leg})
    _cal.validate_session_coverage(idx)
    present = set(idx)
    absent = pd.DatetimeIndex([d for d in decisions if d not in present])
    if len(absent):
        raise CalendarCoverageError(
            "missing_month_end_session",
            f"SPY lacks the decision month-end session(s) {_overlay._listed(absent)}; refusing to shift "
            "a window to the next available bar", missing_dates=absent)
    # the terminal exit: the NEXT calendar month's scheduled month-end (not looked up from the day
    # after the last decision, which is still in the same month when that session is not the
    # last calendar day)
    next_month = (decisions[-1].to_period("M") + 1).start_time.normalize()
    terminal = month_end_schedule(next_month, next_month)[0]
    return {
        "table": table, "decisions": decisions, "month_ends": decisions.append(pd.DatetimeIndex([terminal])),
        "terminal_exit": terminal, "first_decision": first,
        "states": [bool(s) for s in decided["state"].astype(bool)], "spy_leg": spy_leg, "idx": idx,
        "financing": _overlay.causal_financing_rate(irx_frame["Close"], idx),
    }


def _window_info(ctx: dict, windows: list, states: list) -> dict:
    idx = ctx["idx"]
    truncated = bool(windows and windows[-1][0] < idx[-1] < windows[-1][1])
    entered_on_last_bar = bool(windows and windows[-1][0] == idx[-1])
    return {
        "windows": windows,
        "n_decisions": len(states), "n_high": int(sum(states)),
        "high_share": float(np.mean(states)) if states else None,
        "n_state_switches": int(sum(a != b for a, b in zip(states, states[1:]))),
        "episode_months": episode_lengths(states),
        "first_decision": ctx["first_decision"].strftime("%Y-%m-%d"),
        "last_decision": ctx["decisions"][-1].strftime("%Y-%m-%d"),
        "terminal_exit": ctx["terminal_exit"].strftime("%Y-%m-%d"),
        # a real trade closed early at the last close: the data ends INSIDE the window
        "truncated_final": truncated,
        # entered on the very last bar: nothing left to earn, no trade
        "entered_on_last_bar": entered_on_last_bar,
        "asof_reads": list(ctx["table"].attrs.get("asof_reads", [])),
    }


def curve_from_states(ctx: dict, spy_frame, states, *, spy_cost_bps, spy_entry_cost_bps) -> dict:
    """An account curve for ANY state vector (one bool per decision month), built
    with the PINNED constants so the main arm and every control arm share exactly
    one construction. ``spy_cost_bps`` is the SPY overlay leg's ROUND-TRIP cost;
    ``spy_entry_cost_bps`` the benchmark's already-halved entry cost (the
    identical figure the ABI-1 evaluator charges the benchmark). The returned
    dict is ``mining.long_overlay.build_long_overlay_curve``'s, plus
    ``window_info`` and the ``signal_table``."""
    states = [bool(s) for s in states]
    if len(states) != len(ctx["states"]):
        raise ValueError(f"curve_from_states needs one state per decision month "
                         f"({len(ctx['states'])}); got {len(states)}")
    windows = high_windows(ctx["month_ends"], states)
    curve = _overlay.build_long_overlay_curve(
        spy_frame, {"SPY": ctx["spy_leg"]}, windows, leg_cost_bps={"SPY": spy_cost_bps},
        spy_entry_cost_bps=spy_entry_cost_bps, financing_rate_annual=ctx["financing"],
        overlay_notional=OVERLAY_NOTIONAL)
    curve["window_info"] = _window_info(ctx, windows, states)
    curve["signal_table"] = ctx["table"]
    return curve


def build_curve(spy_frame, irx_frame, gc_close, hg_close, *, spy_cost_bps, spy_entry_cost_bps,
                context=None) -> dict:
    """The family's SCORED account curve (the merged HIGH windows). ``context`` is
    an already-computed ``prepare(...)`` dict for the same inputs (a driver that
    also builds the control arms prepares once and passes it in)."""
    ctx = context if context is not None else prepare(spy_frame, irx_frame, gc_close, hg_close)
    return curve_from_states(ctx, spy_frame, ctx["states"], spy_cost_bps=spy_cost_bps,
                             spy_entry_cost_bps=spy_entry_cost_bps)


def build_always_on_curve(ctx: dict, spy_frame, *, spy_cost_bps, spy_entry_cost_bps) -> dict:
    """REPORT-ONLY always-on control: the identical construction, financing, costs
    and tax policy with the state forced HIGH on every decision month, i.e. ONE
    window from the first decision month-end to the end of the data. Never scored,
    never written to a ledger."""
    return curve_from_states(ctx, spy_frame, [True] * len(ctx["states"]), spy_cost_bps=spy_cost_bps,
                             spy_entry_cost_bps=spy_entry_cost_bps)


def random_window_state_draws(states, n_draws: int, *, seed: int) -> list:
    """``n_draws`` deterministic random state vectors for the REPORT-ONLY
    random-window control. Each keeps the main arm's episode COUNT and exact
    multiset of episode LENGTHS (so its HIGH share), permutes the lengths, and
    places the episodes at a uniformly random non-adjacent arrangement (at least
    one non-HIGH month between two episodes, so windows never touch). A local
    ``numpy`` Generator is seeded per call: the global random state is never
    read or written."""
    states = [bool(s) for s in states]
    n, lengths = len(states), episode_lengths(states)
    if not lengths:
        return [[False] * n for _ in range(n_draws)]
    m = len(lengths)
    extra = n - sum(lengths) - (m - 1)                  # non-HIGH months beyond the mandatory single gaps
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        order = rng.permutation(m)
        bars = np.sort(rng.choice(extra + m, size=m, replace=False))     # stars and bars: uniform composition
        bins = np.diff(np.concatenate([[-1], bars, [extra + m]])) - 1    # m + 1 non-negative gap sizes
        draw = [False] * int(bins[0])
        for i, k in enumerate(order):
            if i:
                draw += [False] * (1 + int(bins[i]))
            draw += [True] * lengths[k]
        draw += [False] * int(bins[m])
        draws.append(draw)
    return draws
