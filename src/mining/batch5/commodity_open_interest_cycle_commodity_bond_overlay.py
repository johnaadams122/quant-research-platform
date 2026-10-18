"""Batch5 candidate: commodity_open_interest_cycle_commodity_bond_overlay (bespoke
driver; run only via scripts/run_commodity_oi_cycle_overlay_abi1.py, never the
generic scripts/run_abi1_mining.py dispatch -- see "Dispatch safety").

PRE-REGISTERED RULE. Every constant below was fixed from the catalog entry and
the literature before this pipeline computed any return of this account, and none
of them is tuned after a result (not tuned, not re-fit). This is one pooled trial
(one family, one permanent ledger row)::

    account = 100 percent SPY at all times (the benchmark series itself)
              + a margin-financed LONG overlay of 0.25 of account equity, holding
                ONE leg chosen by the state below, DBC when HIGH and TLT when LOW
    T(m)    = the final SCHEDULED NYSE session of calendar month m
    state   = decided at the close of T(m) from CFTC reports usable by then
    leg     = held from the close of T(m) to the close of T(m+1)
    window  = ONE month per window (the SCORED convention, ``SCORED_MERGE_SAME_LEG`` is False): the
              overlay is re-sized to 0.25 of equity and charged a full round trip every month, even
              when the leg continues; merging consecutive same-leg months into one window is the
              report-only sensitivity

Signal (Hong and Yogo "commodity market interest", replicated on contract COUNTS)
--------------------------------------------------------------------------------
Data: CFTC Legacy Futures-Only ``open_interest_all`` for 18 contracts in four
sectors (``mining.cftc_data.CONTRACTS``). Per month-end T(m), each contract's open
interest is its LAST AVAILABLE report (``mining.cftc_data.month_end_open_interest``:
a report as of Tuesday D is usable only from D+6 calendar days, with the explicit
publication-blackout handling below). Then:

  1. per contract, monthly log growth  g = ln(OI_m / OI_(m-1));
  2. per sector, the mean of its contracts' g; a contract with no value that month
     is left out, never zero-filled;
  3. the aggregate is the EQUAL-WEIGHT mean of the sectors that have a value;
  4. CMI_m is the mean of the last 12 aggregate values (all 12 are required);
  5. state HIGH when CMI_m is STRICTLY greater than the expanding median of CMI over
     every prior evaluated month AND the current month (pandas expanding-median
     semantics), with a minimum of 36 monthly observations; otherwise LOW.
     2019-11-29 is an EXACT TIE (CMI equals its expanding median to the last digit: it
     is the middle observation of an odd count), resolved to LOW by the pre-registered
     strict rule; the tie arises from the blackout exclusion (blackout months left out
     of the median; without that exclusion the state there is HIGH), and it is also
     sensitive to the unverified pre-2000 release timing (an extra 7 or 14 days of lag on the
     pre-2000 reports alone flips it; no other scored state moves).

Deviations from the published paper (each pre-declared; the citation gives the idea
and the two signs, not this exact rule): Hong, H. and Yogo, M. (2012), "What does
futures market interest tell us about the macroeconomy and asset prices?", Journal of
Financial Economics 105(3):473-490 (1965-2008 sample, no out-of-sample test) uses
DOLLAR open interest (spot price times contracts), about 30 commodities, growth taken
on the aggregate, a 12-month geometric average, a linear regression on the continuous
variable and the 10-year Treasury note. This family differs in FIVE declared ways:
(1) contract counts, not dollar open interest (dollar open interest would mostly
re-encode commodity price momentum, a different signal); (2) contract-level growth
averaged within each sector, then the sectors equal-weighted and averaged arithmetically
over 12 months (it telescopes to a year-over-year count change), not the paper's
aggregate-then-growth with a geometric average; (3) 18 contracts, not about 30
commodities; (4) a median split, not a
regression on the continuous variable; (5) TLT (a 20-plus-year fund), not the 10-year
note. The paper's commodity-return sign is positive (momentum from underreaction) and
its bond-return sign is negative (a one standard deviation rise cuts expected bond
returns), so HIGH holds commodities (DBC) and LOW holds long bonds (TLT).

Point-in-time status and the three publication blackouts
--------------------------------------------------------
The CFTC series is NOT verified never-revised: corrections are documented for the
reports as of 2017-03-28, 2018-09-18 and 2019-10-15 and Socrata serves the corrected
values (see ``mining.cftc_data``). It is never labelled never_revised anywhere, and
the ledger note says so. The fixed D+6 lag is false in the three publication
blackouts (the 2013, 2018-19 and 2025-26 federal shutdowns), when a decision date
would consume reports that were not yet public. At each such month-end (2013-10-31,
2018-12-31, 2019-01-31, 2019-02-28, 2025-10-31, 2025-11-28 and 2025-12-31 in the
history; the last three lie inside the ABI-1 holdout) this family REFUSES to update its
state and carry the prior state (the more conservative of the two available options), leaves that month's CMI out of the expanding median, and reads the open
interest of every such month-end from the last report that was really public. Per the
CFTC's own notice, the first delayed 2018-19 report is
the Christmas-shifted one as of Monday 2018-12-24, so 2019-02-28 is a blackout
month-end too (on the announced cadence its plain-lag report was published after it). A CMI whose 12-month base
month is a blackout month-end measures a window longer than 12 months (2014-10-31,
2019-12-31, 2020-01-31 and 2020-02-28 in the scored span): a disclosed, not repaired,
consequence of the conservative carry, with no look-ahead. The 2025-26 table keeps the
CFTC's first plan's outer bound although its accelerated plan cleared the backlog on
2025-12-29, so 2025-12-31 is carried though its report was already public
(conservative). The D+6 lag has three days of margin in a normal week and none in a
holiday-shifted week, immaterial to the state (see ``mining.cftc_data``). Five scored month-end
decisions (2009-11-30, 2011-02-28, 2015-11-30, 2020-11-30 and 2022-02-28) sit in that no-margin case:
their report is from a US-federal-holiday week and its plain D+6 date falls on the decision day
itself. The CFTC usually moves such a release to the Monday afternoon, about 30 minutes before the
close on the decision day (NOT verified against a CFTC notice); withholding those five reports on
their D+6 date changes no state, so no state depends on them.

The expanding median suffers composition drift: grains start 1998, lean hogs 1996, natural
gas 1990, copper 1989 and RBOB 2006, so the early CMI, and therefore the early median
baseline, is built from fewer contracts (the sector average and the equal-weight of
sectors keep every sector represented from 1986). On the cached CFTC counts: 241 of
the 476 CMI values (50.6 percent) have a 12-month window in which fewer than 18 contracts
carry a growth value, all 18 first carry one at 2006-03-31, and 12 of the 247 evaluation
months since 2006-02 are still partial-composition, so the median baseline is not
like-for-like with the later months it is applied to; disclosed, not repaired. The
cached series has no staleness rule before 2005 (from 2005 on ``require_recent_reads`` refuses a
non-blackout month-end that reads a report older than 14 days; every real read is 6 to 10 days old):
palladium reports have real gaps of 84, 168 and
259 days in 2000-2002 and its month-end value is carried stale (up to 247 days) with zero
growth, then jumps; treating any contract older than 30 days as missing changes no state.
When the state is mostly LOW for long stretches the account is close to a static long-TLT
overlay and the effective sample is small; the report-only checks print state shares and switch
counts by sub-period.

SIXTH DISCLOSED LIMITATION (a limitation, not a deviation from the paper, so the deviations stay
FIVE): the 18-contract universe is a present-day survivor set, the contracts that exist and report
cleanly today, with no membership rule applied at each historical date; a contract that stopped
trading before 2026 cannot be in it.

Data-quality refusals (they can only refuse a run, never change a state): the cache must reach the
last decision's report for EVERY contract (``require_fresh_cache``, reason ``cftc_contract_stale``
for a contract that ends early, ``cftc_cache_stale`` for the whole cache), ``fetch_and_cache_all``
refuses contracts that do not share one last as-of date, and ``require_recent_reads`` refuses any
non-blackout month-end from 2005 on that reads a report older than ``MAX_REPORT_AGE_DAYS`` for any
contract. ``STATE_HISTORY_START`` (the first decision date, so how much history feeds the
expanding-median baseline) is a pre-registered parameter, because a later start moves scored states.

The account (why an overlay, and the scored window convention)
--------------------------------------------------------------
ABI-1 scores the account's daily excess over SPY, so the premium is isolated by holding
SPY at all times and adding a small financed long overlay (``mining.long_overlay``).
The documented builder takes a fixed leg set for every window, equal-weights the legs
and rejects back-to-back windows, so this module composes it WITHOUT editing it: one
``build_long_overlay_curve`` call per leg (and per non-adjacent group of windows), each
with its own windows, whose overlay returns are summed on the shared calendar. Within a
window the overlay's return on account equity depends only on that window's own leg
prices, the financing rate and the SPY core (the sizing equity cancels), and the
windows of different legs never overlap, so the sum is exact; a test compares the
result with an independent brute-force account and, for a single leg, bit for bit with
the documented builder. Costs are the builder's: a full round trip per window on the
exit session.

SCORED CONVENTION (design choice, one switch: ``SCORED_MERGE_SAME_LEG``, False):
every calendar month is its own window. The mechanism is a one-month hold per decision,
so each month re-sizes the overlay to 0.25 of equity and charges a full round trip even when the
leg continues (conservative on cost). Merging consecutive same-leg months into one window is
the REPORT-ONLY merged sensitivity. The price of merging,
disclosed: a merged window keeps a fixed share count for its whole life, so its notional drifts
from 0.25 of equity (a long window in a rising account shrinks in relative terms), and this
pipeline's realization tax walk (``mining.account_relative``) charges the short-term rate on a
round trip held 365 days or less and the long-term rate, which the pipeline sets to zero, on one
held longer, so a merged window held beyond a year realizes its gain untaxed while the same gain
earned month by month is taxed as short-term. The merged convention therefore
flatters the strategy through the tax walk as well as through costs (the scored monthly
convention does not): in the scored convention no window is held beyond a year, while merged
same-leg blocks can be. The driver reports how many windows were held
beyond a year and how much of the positive window return they carry, for the scored account and
for the sensitivity. A truncated final window is closed at the last close and flagged; the driver
refuses to record it.

Control arm (report-only, no ledger row)
----------------------------------------
``build_control_curve`` is an always-TLT 0.25x overlay: the same account, costs,
financing and first entry, holding TLT every month regardless of the state (merged: one
long position; unmerged: re-sized monthly). A pass is only meaningful if the strategy
beats it: the state rule has to add something over plain long-duration exposure. The
driver prints both variants beside the main result and writes the pre-declared
interpretation rule into the ledger note. The recorded stage stays the ABI-1 verdict (the
stage vocabulary is closed and every tested family is recorded), so a ``survived`` stage
whose strategy did not beat BOTH controls is flagged in the note's first sentence and is
not a pre-registered pass (the driver's ``pre_registered_pass`` field, True only when the
verdict is ``survived`` AND both controls were beaten).

Disclosed flattering assumptions (each flatters the strategy)
------------------------------------------------------------
There is no margin spread: financing is the bare ^IRX bill rate on the whole borrowed notional
(the pipeline has no margin-rate source and inventing one is not allowed). DBC is
energy-heavy (crude, heating oil and gasoline dominate the index) while the signal is
equal-weight across four sectors. Reg-T margin calls and borrow availability are not
modelled, and ABI-1's realization tax walk treats financing interest as deductible.

Risk layer (the risk-basis rule)
-----------------------
The curve declares ``account_kind == "benchmark_plus_overlay"`` so ABI-1's four risk
refusals are measured on the OVERLAY-ONLY EXCESS account, not the whole SPY-core account
(``mining.overlay_risk_report``). Because the overlay is on in essentially every month
after the first entry, the guard that the excess is overlay-only can only verify the
sessions before the first window; that is stated, not hidden.

Look-ahead discipline: the state at T(m) uses only reports whose availability date is at
or before T(m), the month-ends come from the exchange's scheduled calendar (never the
last row of the data), a state-level prefix-invariance test truncates the CFTC data and shows every
earlier state unchanged, a second test perturbs every report a decision date cannot yet use,
and the curve-level prefix-invariance test cuts the prices 1 to 4 sessions after each switch
(prices only: a leg chosen from the next month's state is caught by the state-flip
cannot-move test, not by that price prefix test). The state at T(m) moves no
price return on or before T(m); the only thing it can change on session T(m) itself is the
round-trip cost of a switch executed at that close, which the builder books on the exit
session (the state depends on no price, so a market-on-close order is feasible).

Dispatch safety
---------------
``SHAPE = "pooled_single_asset"`` is a contract LABEL only. ``TRADED`` lists the two overlay
legs; ``SIGNAL`` lists ^IRX, a bill-rate series that is never traded and can print at or
below zero (a bespoke driver loads it WITHOUT the positivity gate meant for traded prices).
The CFTC open interest is not a price series; the driver loads it from
``data/mining/cftc_oi``. SPY is ``BENCHMARK``. ``make_signal_fn`` exists only to satisfy the
contract and ALWAYS raises, so the generic dispatch turns it into a clean ``compute_error``.
The holdout is the last 756 account sessions (``constants.ABI1_HOLDOUT_SESSIONS``); the
2025-26 blackout falls inside it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining import cftc_data as _cftc
from mining import long_overlay as _overlay
from mining.long_overlay import CalendarCoverageError  # noqa: F401  (re-exported for the driver)
from mining.batch5 import treasury_month_end_index_demand_overlay as _calendar
from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns

FAMILY_ID = "commodity_open_interest_cycle_commodity_bond_overlay"
SHAPE = "pooled_single_asset"      # contract label only -- see "Dispatch safety"
TRADED = ["DBC", "TLT"]
SIGNAL = ["^IRX"]                  # financing-rate series, never traded
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"

# Pre-registered constants, fixed in the catalog entry and from the paper, not tuned:
#   OVERLAY_NOTIONAL  -- 0.25 of equity (gross 1.25 of the 2.0 Reg-T ceiling)
#   CMI_MONTHS        -- the paper's 12-month average
#   MIN_MEDIAN_MONTHS -- minimum monthly observations before a state exists
OVERLAY_NOTIONAL = 0.25
CMI_MONTHS = 12
MIN_MEDIAN_MONTHS = 36
# STATE_HISTORY_START -- the first decision date of the state history: how much history feeds the expanding-median
# baseline (1986-01-02 is the first CFTC report in the cache, so the baseline is the WHOLE history; a later start
# moves scored states, so it is a pre-registered parameter and the driver refuses any other value).
STATE_HISTORY_START = "1986-01-02"
LEG_BY_STATE = {"HIGH": "DBC", "LOW": "TLT"}
CONTROL_LEG = "TLT"
# Data-quality refusals (they can only REFUSE a run, never change a state, so they are not strategy parameters).
# A non-blackout month-end from 2005 on must read, for every contract that has started reporting, a report no
# older than MAX_REPORT_AGE_DAYS. Measured on the real cache (signal-only, no price): every such read is 6 to 10
# days old (a Friday month-end reads the Tuesday ten days back), and one missed weekly report pushes a read to 13
# to 17 days, so 14 clears the real history with four days of slack and refuses a contract that stopped early or
# lost weeks mid-history. The check starts in 2005: palladium's real reporting gaps (84, 168 and 259 days) lie
# wholly in 2000-2002 and change no state.
MAX_REPORT_AGE_DAYS = 14
STALE_READ_CHECK_START = pd.Timestamp("2005-01-01")
# THE ONE SWITCH for the scored window convention (design choice). False: every calendar
# month is its own window (the one-month hold; a round trip and a re-size to 0.25 of equity each
# month, even when the leg continues). True: consecutive same-leg months merge into one window. The other
# convention is always computed by the driver as a report-only sensitivity. It is part of PARAMETERS, so
# flipping it makes the driver refuse with ``parameter_drift`` until the catalog entry is re-registered.
SCORED_MERGE_SAME_LEG = False

# Copied byte-for-byte into the catalog entry (a test compares canonical JSON).
PARAMETERS = {
    "position": "long_overlay_on_benchmark",
    "core": "SPY fully invested at all times (the benchmark series itself)",
    "overlay_legs": list(TRADED),
    "overlay_notional_fraction": OVERLAY_NOTIONAL,
    "state_to_leg": dict(LEG_BY_STATE),
    "data": "CFTC Legacy Futures-Only open_interest_all (Socrata dataset 6dca-aqww), contract COUNTS not dollar open interest",
    "contracts_by_sector": _cftc.contracts_by_sector(),
    "availability_lag_calendar_days": _cftc.AVAILABILITY_LAG_DAYS,
    "publication_blackouts": "month-ends inside the 2013, 2018-19 and 2025-26 publication blackouts carry the prior state and are left out of the median; delayed reports count as available only from their conservative public-by date",
    "monthly_growth": "per contract ln(OI_m / OI_(m-1)) of the last available report at each scheduled month-end; contracts without a value are left out",
    "aggregation": "mean within each sector, then the equal-weight mean of the sectors that have a value",
    "cmi_months": CMI_MONTHS,
    "state_history_start": STATE_HISTORY_START,
    "state_rule": "HIGH when CMI is strictly above the expanding median of evaluated CMI including the current month, else LOW",
    "min_median_months": MIN_MEDIAN_MONTHS,
    "decision": "state fixed at the close of each scheduled month-end session T(m), held for the following month",
    "merge": "scored convention: every calendar month is its own window, so each month re-sizes the overlay to 0.25 of equity and pays a full round trip even when the leg continues; merging consecutive same-leg months into one window (one round trip, fixed share count) is a report-only sensitivity",
    "scored_merge_same_leg": SCORED_MERGE_SAME_LEG,
    "entry": "close of T(m) for every month",
    "exit": "close of T(m+1), the next scheduled month-end session",
    "calendar": "exchange_calendars XNYS scheduled sessions: weekdays minus regular holidays; unscheduled closures are not removed",
    "financing_rate": "^IRX 13-week bill coupon-equivalent yield of the last close strictly before each session, on the borrowed overlay notional, actual calendar days over 365, no margin spread",
    "control_arm": "report-only always-TLT 0.25x overlay, same costs and financing, merged and monthly-resized variants",
    "benchmark": BENCHMARK,
}

# A conservative floor, set well below the window count of both the scored monthly convention and
# the merged sensitivity over the cached open-interest history (signal only, no price read), the
# sibling convention, and not tuned.
EXPECTED_MIN_TRADES = 20


def make_signal_fn(*args, **kwargs):
    """Always raises -- this family has no per-bar signal. See "Dispatch safety"."""
    raise RuntimeError(
        f"{FAMILY_ID}: this family must only be run through its own bespoke driver "
        "(scripts/run_commodity_oi_cycle_overlay_abi1.py), never the generic "
        "scripts/run_abi1_mining.py dispatch -- it has no per-bar signal, and the generic "
        "path would score two plain long ETF positions instead of the benchmark-plus-overlay "
        "account.")


class SignalDataError(ValueError):
    """The open-interest state cannot support the requested account; ``reason_code`` says why."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


# ----------------------------------------------------------------- calendar

def _normalized(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def month_end_sessions(start, end) -> pd.DatetimeIndex:
    """The final SCHEDULED NYSE session of every calendar month touched by
    ``[start, end]`` (the exchange's regular-holiday schedule, fixed in advance; it
    takes dates only and cannot read data)."""
    start, end = _normalized(start), _normalized(end)
    if end < start:
        raise ValueError("end must not precede start")
    first = pd.Timestamp(start.year, start.month, 1)
    last = pd.Period(end, freq="M").end_time.normalize()
    sessions = _calendar.scheduled_sessions(first, last)
    ends = sessions.to_series().groupby(sessions.to_period("M")).max()
    return pd.DatetimeIndex(ends.to_numpy())


# ------------------------------------------------------------------ signal

def aggregate_growth(oi_month_end: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregate log growth of open interest. ``oi_month_end`` has one row
    per decision date and one column per contract code. Returns one column per
    sector (mean of that sector's contract log growth, NaN when no contract has a
    value) and ``aggregate`` (the equal-weight mean of the sectors that have one)."""
    oi = oi_month_end.astype(float)
    if (oi.to_numpy(dtype=float) <= 0.0).any():
        raise ValueError("open interest must be positive wherever it is present")
    growth = np.log(oi / oi.shift(1))
    table = pd.DataFrame(index=oi.index)
    for sector, codes in _cftc.contracts_by_sector().items():
        present = [c for c in codes if c in growth.columns]
        table[sector] = growth[present].mean(axis=1, skipna=True) if present else np.nan
    table["aggregate"] = table[list(_cftc.SECTORS)].mean(axis=1, skipna=True)
    return table


def cmi_from_aggregate(aggregate: pd.Series) -> pd.Series:
    """CMI: the mean of the last ``CMI_MONTHS`` aggregate values (all required)."""
    return aggregate.rolling(CMI_MONTHS, min_periods=CMI_MONTHS).mean()


def classify_states(cmi: pd.Series, blackout: pd.Series) -> pd.DataFrame:
    """HIGH / LOW per month. A blackout month is not evaluated: its CMI is left out
    of the expanding median and its state CARRIES the prior month's (``carried``).
    A missing CMI outside a blackout leaves the state undefined (never
    forward-filled)."""
    blackout = blackout.reindex(cmi.index).fillna(False).astype(bool)
    evaluable = cmi.where(~blackout)
    median = evaluable.expanding(min_periods=MIN_MEDIAN_MONTHS).median()
    raw, state, carried = [], [], []
    previous = None
    for month in cmi.index:
        if blackout[month]:
            raw.append(None)
            state.append(previous)
            carried.append(previous is not None)
            continue
        value, cutoff = evaluable[month], median[month]
        if pd.isna(value) or pd.isna(cutoff):
            current = None
        else:
            current = "HIGH" if value > cutoff else "LOW"
        raw.append(current)
        state.append(current)
        carried.append(False)
        previous = current
    return pd.DataFrame({
        "cmi": cmi, "median": median,
        "raw_state": pd.Series(raw, index=cmi.index, dtype=object),
        "blackout": blackout,
        "state": pd.Series(state, index=cmi.index, dtype=object),
        "carried": pd.Series(carried, index=cmi.index, dtype=bool),
    })


def signal_state_table(series_by_code, decision_dates) -> pd.DataFrame:
    """The open-interest state at every decision date: per-sector and aggregate
    growth, CMI, the expanding median, the raw and the carried state, and the
    blackout flag. ``series_by_code`` maps contract code to its as-of-date series
    (``CftcSnapshot.series``); ``decision_dates`` are scheduled month-end sessions."""
    decisions = pd.DatetimeIndex(decision_dates)
    values, _ = _cftc.month_end_open_interest(series_by_code, decisions)
    growth = aggregate_growth(values)
    cmi = cmi_from_aggregate(growth["aggregate"])
    blackout = pd.Series([_cftc.blackout_at(d) is not None for d in decisions], index=decisions)
    return growth.join(classify_states(cmi, blackout))


def require_fresh_cache(snapshot, decision_dates) -> None:
    """Refuse a cache too old to decide the LAST non-blackout decision date: the report
    that the D+6 rule would name at that date must already be in the cache, for EVERY
    contract (``cftc_contract_stale`` names each one that ends earlier: the newest
    contract alone must not vouch for a cache whose other contracts stopped early). A
    blackout decision date needs no new report (its state is carried)."""
    live = [d for d in pd.DatetimeIndex(decision_dates) if _cftc.blackout_at(d) is None]
    if not live:
        return
    last = max(live)
    cutoff = last - pd.Timedelta(days=_cftc.AVAILABILITY_LAG_DAYS)
    expected = cutoff - pd.Timedelta(days=(cutoff.dayofweek - 1) % 7)          # the latest Tuesday
    refresh = "refresh it with `python -m mining.cftc_data --default-cache-dir` (network)"
    if snapshot.latest_asof < expected:
        raise SignalDataError(
            "cftc_cache_stale",
            f"the decision at {last:%Y-%m-%d} needs the report as of {expected:%Y-%m-%d} "
            f"(usable from {_cftc.availability_date(expected):%Y-%m-%d}) but the cache's newest report "
            f"is as of {snapshot.latest_asof:%Y-%m-%d}; {refresh}")
    lagging = {code: series.index[-1] for code, series in snapshot.series.items() if series.index[-1] < expected}
    if lagging:
        detail = ", ".join(f"{code} ends {stamp:%Y-%m-%d}" for code, stamp in sorted(lagging.items()))
        raise SignalDataError(
            "cftc_contract_stale",
            f"the decision at {last:%Y-%m-%d} needs the report as of {expected:%Y-%m-%d} for every contract, but "
            f"{len(lagging)} contract(s) end earlier ({detail}); the newest report in the cache is as of "
            f"{snapshot.latest_asof:%Y-%m-%d}; {refresh}")


def require_recent_reads(series_by_code, decision_dates) -> None:
    """Refuse a history in which a contract's read at a scored month-end is stale: at every
    non-blackout decision date from ``STALE_READ_CHECK_START`` on, each contract that has
    started reporting must read a report no older than ``MAX_REPORT_AGE_DAYS`` (a
    contract that stopped early, or lost weekly reports mid-history, would otherwise be
    carried forward with zero growth). Blackout month-ends read older reports by design
    and are skipped. Reason code ``cftc_report_stale_at_month_end``. Pass only decision
    dates that have happened: a future month-end has no report yet by construction."""
    live = pd.DatetimeIndex([d for d in pd.DatetimeIndex(decision_dates)
                             if d >= STALE_READ_CHECK_START and _cftc.blackout_at(d) is None])
    if not len(live):
        return
    _, used = _cftc.month_end_open_interest(series_by_code, live)
    stale = []
    for decision in live:
        for code in used.columns:
            asof = used.loc[decision, code]
            if pd.isna(asof):
                continue                                   # no report is usable yet: the contract has not started
            age = int((decision - pd.Timestamp(asof)).days)
            if age > MAX_REPORT_AGE_DAYS:
                stale.append((decision, code, pd.Timestamp(asof), age))
    if stale:
        shown = "; ".join(f"{code} at the {decision:%Y-%m-%d} month-end reads the report as of {asof:%Y-%m-%d} "
                          f"({age} days old)" for decision, code, asof, age in stale[:5])
        raise SignalDataError(
            "cftc_report_stale_at_month_end",
            f"{len(stale)} contract read(s) at scored month-ends are older than {MAX_REPORT_AGE_DAYS} days "
            f"(first {min(5, len(stale))}: {shown}); a contract that stopped early or lost reports would be "
            "carried forward stale, so the history is refused rather than patched")


# --------------------------------------------------------------------- plan

def _leg_for(state) -> str:
    return LEG_BY_STATE[state]


def plan_windows(index, table, *, merge_same_leg: bool = True, force_leg=None) -> dict:
    """The windows for a data calendar. Every calendar month whose scheduled last
    session ``T(m)`` lies in the data opens a month-long window to ``T(m+1)`` holding
    the leg of the state decided at ``T(m)``; a month whose exit lies after the last
    data date is kept (the builder closes it at the last close); a window entered on
    the very last bar opens nothing and is dropped before merging. With
    ``merge_same_leg`` consecutive same-leg months become one window. ``force_leg``
    (the control arm) holds one leg every month and needs no state. A scheduled entry
    or an in-data exit missing from the data is a hole and raises (never shifted).
    Returns ``windows`` (merged unless asked otherwise, ``(entry, exit, leg)``),
    ``monthly``, ``windows_by_leg`` and the end-of-data flags."""
    if force_leg is not None and force_leg not in TRADED:
        raise ValueError(f"force_leg must be one of {TRADED}; got {force_leg!r}")
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if len(idx) == 0:
        raise ValueError("plan_windows needs a non-empty data calendar")
    present = set(idx)
    ends = month_end_sessions(idx[0], idx[-1] + pd.Timedelta(days=45))
    last_month = idx[-1].to_period("M")
    monthly, skipped, not_started = [], 0, 0
    for k in range(len(ends) - 1):
        entry, exit_ = ends[k], ends[k + 1]
        if entry.to_period("M") > last_month:
            break
        if entry < idx[0]:
            skipped += 1
            continue
        if entry > idx[-1]:
            not_started += 1
            continue
        if entry not in present:
            raise ValueError(
                f"the scheduled window for {entry:%Y-%m} needs its entry session {entry.date()} in the "
                "data, but it is missing (vendor hole or unscheduled closure); refusing to shift or "
                "drop the window")
        if exit_ <= idx[-1] and exit_ not in present:
            raise ValueError(
                f"the scheduled window for {entry:%Y-%m} needs its exit session {exit_.date()} in the "
                "data, but it is missing (vendor hole or unscheduled closure); refusing to shift or "
                "drop the window")
        if force_leg is not None:
            leg = force_leg
        else:
            if entry not in table.index:
                raise SignalDataError(
                    "state_table_does_not_cover_the_data",
                    f"the state table has no row for the decision date {entry:%Y-%m-%d}")
            state = table.loc[entry, "state"]
            if pd.isna(state):
                raise SignalDataError(
                    "state_unavailable",
                    f"the open-interest state at the decision date {entry:%Y-%m-%d} is undefined; "
                    "refusing to guess a leg")
            leg = _leg_for(state)
        monthly.append((entry, exit_, leg))
    entered_on_last_bar = bool(monthly and monthly[-1][0] == idx[-1])
    if entered_on_last_bar:
        monthly.pop()                     # entry close is the last close: nothing to earn, nothing opens
    truncated_final = bool(monthly and monthly[-1][0] < idx[-1] < monthly[-1][1])
    windows = []
    for window in monthly:
        if merge_same_leg and windows and windows[-1][2] == window[2] and windows[-1][1] == window[0]:
            windows[-1] = (windows[-1][0], window[1], window[2])
        else:
            windows.append(window)
    return {
        "windows": windows,
        "monthly": monthly,
        "windows_by_leg": {leg: [(w[0], w[1]) for w in windows if w[2] == leg] for leg in TRADED},
        "skipped_before_data": skipped,
        "not_started": not_started,
        "truncated_final": truncated_final,
        "entered_on_last_bar": entered_on_last_bar,
        "merge_same_leg": bool(merge_same_leg),
        "force_leg": force_leg,
    }


def _split_adjacent(windows: list) -> list:
    """Split one leg's windows into groups in which no two windows share an endpoint
    (the documented builder rejects back-to-back windows): consecutive months of one
    leg alternate between two groups."""
    groups = [[], []]
    for window in windows:
        for group in groups:
            if not group or group[-1][1] != window[0]:
                group.append(window)
                break
    return [g for g in groups if g]


# -------------------------------------------------------------------- curves

def _build(spy_frame, leg_frames, irx_frame, table, *, leg_cost_bps, spy_entry_cost_bps,
           merge_same_leg, force_leg, entry_at_next_open: bool) -> dict:
    if set(leg_frames) != set(TRADED):
        raise ValueError(
            f"leg_frames must be exactly the registered legs {', '.join(TRADED)}; "
            f"got {', '.join(sorted(map(str, leg_frames)))}")
    if not isinstance(leg_cost_bps, dict) or set(leg_cost_bps) != set(TRADED):
        raise ValueError(f"leg_cost_bps must map exactly {', '.join(TRADED)} to a round-trip cost in bps")
    if not isinstance(irx_frame, pd.DataFrame) or "Close" not in irx_frame.columns:
        raise ValueError("irx_frame must be a DataFrame with a Close column")
    legs = {symbol: leg_frames[symbol] for symbol in TRADED}
    idx = _overlay.common_calendar(spy_frame, legs)
    _calendar.validate_session_coverage(idx)
    plan = plan_windows(idx, table, merge_same_leg=merge_same_leg, force_leg=force_leg)
    financing = _overlay.causal_financing_rate(irx_frame["Close"], idx)
    builder = (_overlay.build_next_open_sensitivity_curve if entry_at_next_open
               else _overlay.build_long_overlay_curve)
    curves, curve_legs = [], []
    for leg in TRADED:
        sliced = legs[leg].loc[idx[0]:idx[-1]]
        for group in _split_adjacent(plan["windows_by_leg"][leg]):
            curves.append(builder(
                spy_frame, {leg: sliced}, group, leg_cost_bps={leg: leg_cost_bps[leg]},
                spy_entry_cost_bps=spy_entry_cost_bps, financing_rate_annual=financing,
                overlay_notional=OVERLAY_NOTIONAL))
            curve_legs.append(leg)
    if not curves:                                     # no window at all: the account is the benchmark
        leg = TRADED[0]
        curves.append(builder(
            spy_frame, {leg: legs[leg].loc[idx[0]:idx[-1]]}, [], leg_cost_bps={leg: leg_cost_bps[leg]},
            spy_entry_cost_bps=spy_entry_cost_bps, financing_rate_annual=financing,
            overlay_notional=OVERLAY_NOTIONAL))
        curve_legs.append(leg)
    core = curves[0]["core_daily"]
    overlay = np.zeros(len(idx), dtype=float)
    mask = np.zeros(len(idx), dtype=bool)
    for curve in curves:
        overlay = overlay + curve["overlay_daily"].to_numpy(dtype=float)
        mask = mask | curve["overlay_mask"].to_numpy(dtype=bool)
    strategy = core.to_numpy(dtype=float) + overlay
    growth = np.cumprod(1.0 + strategy)
    frames = []
    for curve, leg in zip(curves, curve_legs):
        trades = curve["trades_df"].copy()
        trades["_leg"] = leg
        frames.append(trades)
    trades_df = pd.concat(frames, ignore_index=True)
    trades_df = trades_df.sort_values("entry_date", kind="stable").reset_index(drop=True)
    legs_by_trade = list(trades_df["_leg"])
    trades_df = trades_df[list(_overlay.TRADES_COLUMNS)]
    if len(trades_df):
        # the builder sizes each call on its OWN partial account; the trade record carries the
        # true combined-account equity at the entry close (informational: ABI-1 does not read it)
        shift = 1 if entry_at_next_open else 0
        positions = idx.get_indexer(pd.DatetimeIndex(trades_df["entry_date"])) - shift
        trades_df = trades_df.assign(size=growth[positions])
    curve = curves[0]
    return {
        "strategy_daily": pd.Series(strategy, index=idx, name="long_overlay_account"),
        "in_market": pd.Series(True, index=idx, name="in_market"),
        "trades_df": trades_df,
        "account_kind": curve["account_kind"],
        "core_daily": core,
        "overlay_daily": pd.Series(overlay, index=idx, name="overlay"),
        "overlay_mask": pd.Series(mask, index=idx, name="overlay_mask"),
        "financing_rate": curve["financing_rate"],
        "leg_by_trade": legs_by_trade,
        "window_info": plan,
    }


def build_curve(spy_frame, leg_frames, irx_frame, table, *, leg_cost_bps, spy_entry_cost_bps,
                merge_same_leg: bool = SCORED_MERGE_SAME_LEG, force_leg=None) -> dict:
    """The family's SCORED account curve (entry at the close of each window's entry
    session, exit at the close of its exit session), built with the PINNED constants.
    ``table`` is ``signal_state_table``'s output; ``irx_frame`` is the raw ^IRX frame
    (Close in percent) loaded WITHOUT the positivity gate; ``leg_cost_bps`` maps DBC and
    TLT to a round-trip cost. ``merge_same_leg`` defaults to the ONE switch
    ``SCORED_MERGE_SAME_LEG`` (monthly windows); the driver passes the opposite value for
    the report-only merged sensitivity. ``force_leg`` exists for the report-only control
    arms; the scored call does not use it. The result has
    ``mining.long_overlay.build_long_overlay_curve``'s shape plus ``leg_by_trade``
    and ``window_info`` (``plan_windows``'s report)."""
    return _build(spy_frame, leg_frames, irx_frame, table, leg_cost_bps=leg_cost_bps,
                  spy_entry_cost_bps=spy_entry_cost_bps, merge_same_leg=merge_same_leg,
                  force_leg=force_leg, entry_at_next_open=False)


def build_control_curve(spy_frame, leg_frames, irx_frame, table, *, leg_cost_bps, spy_entry_cost_bps,
                        merge_same_leg: bool = True) -> dict:
    """REPORT-ONLY control arm: an always-TLT 0.25x overlay with the same account,
    costs, financing and first entry, ignoring the state. Never scored, never recorded."""
    return _build(spy_frame, leg_frames, irx_frame, table, leg_cost_bps=leg_cost_bps,
                  spy_entry_cost_bps=spy_entry_cost_bps, merge_same_leg=merge_same_leg,
                  force_leg=CONTROL_LEG, entry_at_next_open=False)


def build_next_open_sensitivity(spy_frame, leg_frames, irx_frame, table, *, leg_cost_bps,
                                spy_entry_cost_bps) -> dict:
    """REPORT-ONLY disclosed sensitivity: the identical account (the scored window
    convention, ``SCORED_MERGE_SAME_LEG``) with each window's leg bought at the OPEN of
    the first window session (the engine's next-open fill). Never scored."""
    return _build(spy_frame, leg_frames, irx_frame, table, leg_cost_bps=leg_cost_bps,
                  spy_entry_cost_bps=spy_entry_cost_bps, merge_same_leg=SCORED_MERGE_SAME_LEG, force_leg=None,
                  entry_at_next_open=True)


# --------------------------------------------------------------- checks

_SUB_PERIODS = (
    ("2006-2008", pd.Timestamp("2006-01-01"), pd.Timestamp("2008-12-31")),
    ("2009-2016", pd.Timestamp("2009-01-01"), pd.Timestamp("2016-12-31")),
    ("2017-2022", pd.Timestamp("2017-01-01"), pd.Timestamp("2022-12-31")),
    ("2023-onward", pd.Timestamp("2023-01-01"), pd.Timestamp("2100-12-31")),
)


def _shares(states: pd.Series, previous=None) -> dict:
    states = states.dropna()
    n = int(len(states))
    if n == 0:
        return {"evaluations": 0, "high_share": None, "low_share": None, "switches": 0,
                "switches_including_entering": 0}
    within = int((states.to_numpy()[1:] != states.to_numpy()[:-1]).sum())
    entering = int(previous is not None and states.iloc[0] != previous)
    high = float((states == "HIGH").mean())
    return {"evaluations": n, "high_share": high, "low_share": 1.0 - high, "switches": within,
            "switches_including_entering": within + entering}


def state_diagnostics(table: pd.DataFrame, *, start, end, holdout_start=None) -> dict:
    """REPORT-ONLY: HIGH share and regime-switch counts, overall, by pre-declared
    sub-period, by calendar year and inside the ABI-1 holdout, so a reader can see
    how close the account is to a static long-TLT overlay. No return is used."""
    start, end = _normalized(start), _normalized(end)
    rows = table.loc[(table.index >= start) & (table.index <= end)]
    states = rows["state"]
    overall = _shares(states)
    valid = states.dropna()

    def _previous(first_date):
        earlier = valid[valid.index < first_date]
        return earlier.iloc[-1] if len(earlier) else None

    by_period = {}
    for name, lo_date, hi_date in _SUB_PERIODS:
        chunk = states[(states.index >= lo_date) & (states.index <= hi_date)]
        first = chunk.dropna().index[0] if chunk.notna().any() else None
        by_period[name] = _shares(chunk, _previous(first) if first is not None else None)
    by_year = {}
    for year in sorted({d.year for d in states.dropna().index}):
        chunk = states[states.index.year == year]
        by_year[str(year)] = _shares(chunk)
    runs = {"HIGH": 0, "LOW": 0}
    length, label = 0, None
    for state in valid:
        length = length + 1 if state == label else 1
        label = state
        runs[state] = max(runs[state], length)
    holdout = None
    if holdout_start is not None:
        chunk = states[states.index >= _normalized(holdout_start)]
        holdout = _shares(chunk)
        holdout["carried_blackout_month_ends"] = int(rows.loc[chunk.index, "carried"].sum()) if len(chunk) else 0
    return {
        **overall,
        "by_period": by_period,
        "by_year": by_year,
        "longest_run_months": runs,
        "holdout": holdout,
        "blackout_month_ends_carried": int(rows["carried"].sum()),
    }
