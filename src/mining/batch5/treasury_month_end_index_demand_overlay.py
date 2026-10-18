"""Batch5 candidate: treasury_month_end_index_demand_overlay (bespoke driver;
run only via scripts/run_treasury_month_end_overlay_abi1.py, never the generic
scripts/run_abi1_mining.py dispatch -- see "Dispatch safety").

PRE-REGISTERED RULE. Every constant below was fixed before THIS
PIPELINE computed any return of this account, and none of them is to be tuned
after a result. That is NOT the same as chosen blind: some were fixed after
descriptive looks at these same instruments (see "What was and was not chosen
blind" below)::

    account = 100 percent SPY at all times (the benchmark series itself)
              + a margin-financed LONG overlay held ONLY over the last three
                scheduled sessions of each calendar month
    overlay = total notional 0.5 of account equity, split equally across
              IEF, TLT and AGG, borrowed at the ^IRX bill rate (no spread)
    T(m)    = the final SCHEDULED NYSE session of calendar month m
    entry   = the close of T-3   (the session before the 3-session window)
    exit    = the close of T
    window  = sessions T-2, T-1, T  (three sessions; the headline case)

Mechanism and sources (the primary paper is a working paper and its
journal-publication status was NOT verified): fixed-income
benchmark indices rebalance on the last business day of each month and life
insurers and passive bond funds are large net buyers of Treasuries around those
dates, a temporary, predictable demand shock. Hartley, J. and Schwarz, K.
(2019), "Predictable End-of-Month Treasury Returns", SSRN 3440417 (working
paper dated 22 Nov 2019): positive, highly significant excess returns on coupon
Treasuries only in the last few days of the month (about a quarter of a
percent for the 10-year over the last 3 days), roughly zero on other days,
annualized Sharpe near 1, positive in every 5-year block 1990-2018, no
structural break, robust to controlling for the auction cycle; built on an
off-the-run zero-coupon curve, NOT ETF total returns. Federal Reserve Bank of
New York, Liberty Street Economics, "End-of-Month Liquidity in the Treasury
Market" (Sept 2024): Treasury volume is about 46 percent higher and price
impact about 28 percent lower on the last trading day of the month since 2020,
conjectured to reflect passive Treasury-ETF rebalancing. Bloomberg US Aggregate
Index factsheet (12 Jun 2024): "Rebalance Date: The last business day of each
month". Third-party ETF replication (Allocate Smartly) is unverified against
its source data and reports the peak drifting earlier from about 2014
(front-running), which is why the window length is fixed ONCE at the paper's
headline three sessions rather than tuned within this pipeline. None of these
sources tests THIS rule (a 3-session, three-ETF, margin-financed overlay on an
SPY core, scored as ABI-1 daily excess): the window, the notional and the
financing rule are this pipeline's own pre-registered choices.

What was and was not chosen blind (pre-registration honesty)
------------------------------------------------------------
Some of the choices below were made after descriptive looks at these same
three ETFs (close of T-3 to close of T, per ETF and pooled, over a T-bill
proxy), using only data inside ABI-1's in-sample window; the sealed holdout
was not looked at. The looks covered the window length, the close-fill entry
against the engine's next-open fill, and the rotation expression against the
overlay expression. They ran outside the trial ledger, so the multiple-testing
count DSR applies is NOT charged for them, and a DSR pass must be read with
them in mind. Precisely:

  * Window of 3 sessions: the paper's headline example, fixed after the
    descriptive looks. Pre-declared, not blind.
  * Entry at the close of T-3 and exit at the close of T: fixed after the
    comparison with the engine's next-open fill (it is load-bearing and
    must be pre-declared). Not blind.
  * Overlay expression (rather than rotation) and the choice of IEF, TLT and
    AGG in equal thirds: made after those looks; the equal weights were not
    tuned.
  * NOT chosen from any look at its effect: the 0.5 notional (chosen for
    Reg-T slack), the bare bill-rate financing rule with no spread, the
    regular-holiday calendar rule and the real-quote cost model. (The
    descriptive looks used a T-bill proxy, not this account's financed cost.)
  * No constant was tuned against an ABI-1 score, and none may be after one.

Why an overlay and not a rotation
---------------------------------
The score is the account's daily excess over SPY, so a bond fund held IN PLACE
of SPY for the window scores roughly minus SPY's return on those days and
buries the premium; the premium is isolated only by holding SPY as the core
and adding a small financed long overlay (``mining.long_overlay``, the long
analogue of ``mining.beta_neutral_overlay``). Excess over SPY then equals the
overlay's financed, cost-charged P&L. No hedge leg, no short leg.

Interpretation choices (each the most literal reading of the rule)
---------------------------------------------------------------
1. Calendar. ``exchange_calendars`` XNYS is the repo's calendar (the same
   package ``pre_holiday_small_cap_effect`` uses). The window dates must be
   knowable in advance, so the schedule is the exchange's REGULAR holiday
   rules only (weekdays minus ``regular_holidays``): the unscheduled closures
   ``exchange_calendars`` lists as ``adhoc_holidays`` (9/11, the Reagan, Ford,
   Bush and Carter days of mourning, Hurricane Sandy) are KEPT as scheduled
   sessions, because nobody knew about them in advance. The only affected
   window in the history is October 2012 (entry 2012-10-26, exit 2012-10-31,
   the 29th and 30th being scheduled sessions the exchange never opened), and
   its entry and exit are ordinary trading days. A whole-set test proves the
   schedule equals the exchange calendar plus exactly those ad hoc days over
   2001-2028, and a second, independent derivation (the Easter / Memorial Day
   rule in ``short_rate_easing_smallcap_rotation``) agrees on every
   month-end. A month-end is never derived from the data ("the next row falls
   in a new month" would be a look-ahead).
2. Entry at the close of T-3, exit at the close of T (ex-ante-close
   convention). The engine's own convention (a signal fills at the NEXT
   session's Open, i.e. the Open of T-2) is reported only as a disclosed
   sensitivity (``build_next_open_sensitivity``); the scored builder has no
   fill-convention argument.
3. Financing: the whole overlay notional is borrowed at the ^IRX bill rate
   (coupon-equivalent yield of the last close strictly before each session,
   calendar-day accrual), with NO spread invented: the pre-registered
   ``financing_rate`` parameter fixes the financing at the bare bill rate
   with no spread, so none is charged and a real margin rate (a spread over
   the bill rate) is UNMODELLED, which FLATTERS the strategy.
4. Account calendar: the common span of IEF, TLT and AGG (AGG's inception,
   2003-09-29, is the latest), so the first window is October 2003. Before it
   the account is plain SPY and its excess is exactly zero. A month whose
   entry (T-3) precedes the data has no window (warm-up, never back-filled).
5. Pool: IEF, TLT and AGG are ONE pooled hypothesis (one trial), always all
   three; a leg that cannot be priced makes the whole family unscoreable
   (a three-member pool needs all three, the batch5 pooled-coverage rule).
6. A window still open at the end of the data (the data ends at T-2 or T-1) is
   closed at the last close (the engine's ``end_of_data`` path) and reported
   as ``truncated_final``. A window entered on the very last bar opens nothing
   and is reported as ``entered_on_last_bar`` (so trades = windows minus that
   flag). Both states are recorded in the driver's report AND in the permanent
   ledger note, and the driver REFUSES a real (non-dry-run) score on a
   truncated final window: that trade would not exit at the pre-registered T
   and could bake a partial last bar into its exit price.
7. Account type: taxable only. The overlay is borrowed, which needs a margin
   account (a Roth IRA cannot borrow), exactly as the beta-neutral IPO fade's
   short leg needs one. Listing a type the strategy cannot be held in would be
   a false statement in a permanent catalog record.
8. Data calendar: validated, never trimmed or patched. The account calendar is
   the legs' common span (its leading edge follows the latest-listing leg,
   AGG). The legs and SPY must END on the same session
   (``unequal_trailing_coverage``: a vendor missing one series' last bar would
   otherwise silently end the account a session early), must share identical
   sessions inside the span (``calendar_mismatch``), and the shared calendar
   must contain EXACTLY the exchange's actual session set between its first and
   last date (``missing_exchange_session`` for a session EVERY series lacks,
   which pairwise checks cannot see and which would be booked as one multi-day
   bar; ``bar_on_closed_day`` for a stray bar). The known unscheduled closures
   (Hurricane Sandy 2012 and the like) are absent from the actual session set,
   so their absence from the data is correct, not a gap. Every refusal is a
   ``mining.long_overlay.CalendarCoverageError`` (a ``ValueError``) carrying a
   reason code and the dates.

Structural facts (calendar only; no price was read to choose anything here)
---------------------------------------------------------------------------
One window per calendar month from October 2003 onward; the overlay is on for
3 of about 21 sessions a month (about 14 percent). ``EXPECTED_MIN_TRADES`` is
set below the window count, the sibling convention.

Cost source (human-process safeguard)
-------------------------------------
Every leg's cost is a full round trip of the REAL quoted spread, read by the
driver from a dated bid/ask capture file. The shared guard
``refuse_synthetic_quotes`` (scripts/run_batch5_mining.py) refuses a real run
only when the file carries the word SYNTHETIC somewhere in it; it cannot tell a
measured book from an invented bid/ask file that simply omits the marker. That
is a pipeline-wide limit in shared code (deliberately not changed here), so the
provenance of a real run's quotes is a human-process safeguard, not a code
check: real quotes must come from a live broker quote capture, never typed or
estimated by hand. A test pins the limit
so this disclosure is revisited if the shared guard ever verifies provenance.

Dispatch safety
---------------
``SHAPE = "pooled_single_asset"`` is a contract LABEL only (none of the six
registered shapes describes a benchmark-plus-overlay account). ``TRADED`` lists
the three overlay legs; ``SIGNAL`` lists ^IRX, a bill-rate series that is never
traded and can print at or below zero (a bespoke driver loads it WITHOUT the
positivity gate meant for traded prices). SPY (core and benchmark) is
``BENCHMARK``. ``make_signal_fn`` exists
only to satisfy the contract and ALWAYS raises, so the generic dispatch (which
would otherwise score three plain long ETF positions) turns it into a clean
``compute_error`` skip.

Risk layer (the risk-basis rule, benchmark-plus-overlay accounts)
--------------------------------------------------------
ABI-1's layer-2 refusals (``account_relative.risk_refusals``) are FOUR:

  1. the ABSOLUTE drawdown cap (``account_relative.ABSOLUTE_MAX_DRAWDOWN_CAP``,
     0.35);
  2. drawdown no worse than SPY's;
  3. longest losing-week streak no longer than SPY's;
  4. worst 8-week stretch no worse than SPY's (downside only).

Measured on the WHOLE account they would be the wrong test for this family:
the account holds SPY at all times, so over any window containing 2008 (SPY's
own drawdown from 2003 is about 55 percent) the whole account clears the cap
only if the overlay offsets most of it, and the SPY-relative three compare an
account that IS SPY plus a small overlay with SPY itself. The rule is that for
a benchmark-plus-overlay family the cap and the three SPY-relative refusals
apply to the OVERLAY-ONLY EXCESS ACCOUNT,
not the whole SPY-core account. That account earns the daily ARITHMETIC excess
``account - SPY``: for SPY core plus overlay, the overlay's own dollars (net of
costs and financing) over the prior close's account equity, less the
realization-tax drag charged on exit sessions, all in
units of whole-account equity (not of the overlay's own committed capital,
which is borrowed here).

The curve declares this with ``account_kind == "benchmark_plus_overlay"``
(``mining.long_overlay`` sets it; ``abi1_driver.evaluate_family`` reads it) and
the driver refuses to score if the declaration goes missing. Consequences, all
disclosed in the ledger note:

  * the family can reach ``survived``: the risk layer does not cap it;
  * the three SPY-relative refusals compare the overlay's own excess with SPY
    itself and, for a small overlay measured over a window that holds a large
    SPY drawdown, are expected to be much less binding than on the whole
    account; the risk layer is mainly the absolute cap on the overlay's own
    drawdown (overlay-only excess account); the driver writes each run's
    measured numbers into that run's ledger note;
  * the driver refuses to score if the account differs from SPY on any
    session the overlay is off (the excess would then not be overlay-only);
  * the whole-account numbers are still computed and reported beside them
    (``whole_account_reference``), gating nothing, so the ceiling this rule
    removes stays visible.

The driver reports which refusal failed (``failed_refusals``) and writes it
into the ledger note.

Other disclosed risks: window migration and front-running of the premium,
crowding by the passive ETF assets it relies on, the stock-bond correlation
turning positive in 2022, a sparse excess series (the overlay is on about 14
percent of sessions), and short-term tax on the overlay's gains modelled by
ABI-1's realization walk net of financing (interest treated as deductible),
whose entry-equity identity is approximate here because the SPY core moves too.
"""
from __future__ import annotations

import functools

import pandas as pd

from mining import long_overlay as _overlay
from mining.long_overlay import CalendarCoverageError

FAMILY_ID = "treasury_month_end_index_demand_overlay"
SHAPE = "pooled_single_asset"      # contract label only -- see "Dispatch safety"
TRADED = ["IEF", "TLT", "AGG"]
SIGNAL = ["^IRX"]                  # financing-rate series, never traded
SNAPSHOTS: dict = {}
# ABI-1 ALWAYS benchmarks the account against SPY; here SPY is also the core.
BENCHMARK = "SPY"

# Pre-registered constants, each with its source (see the module
# docstring's "What was and was not chosen blind" for which were and were not
# chosen without a prior descriptive look):
#   WINDOW_SESSIONS  -- the paper's headline "last 3 days" example, not tuned
#                       within this pipeline, but fixed after the
#                       descriptive looks
#   OVERLAY_NOTIONAL -- 0.5 of equity, chosen for Reg-T slack (gross 1.5 of a
#                       2.0 ceiling), not for the score; no look at its effect
WINDOW_SESSIONS = 3
OVERLAY_NOTIONAL = 0.5

# Copied byte-for-byte into the catalog entry (data/mining/catalog.json).
PARAMETERS = {
    "position": "long_overlay_on_benchmark",
    "core": "SPY fully invested at all times (the benchmark series itself)",
    "overlay_legs": list(TRADED),
    "overlay_weights": "equal thirds",
    "overlay_notional_fraction": OVERLAY_NOTIONAL,
    "window_sessions": WINDOW_SESSIONS,
    "entry": "close of the last scheduled session before the window (T-3)",
    "exit": "close of the last scheduled session of the calendar month (T)",
    "calendar": "exchange_calendars XNYS scheduled sessions: weekdays minus regular holidays; unscheduled closures are not removed",
    "financing_rate": "^IRX 13-week bill coupon-equivalent yield of the last close strictly before each session, on the borrowed overlay notional, actual calendar days over 365, no spread",
    "benchmark": BENCHMARK,
}
# Counted from the exchange schedule alone (no price read): one window per
# calendar month from October 2003 (AGG's first full month) onward. Set
# below that count, the sibling convention.
EXPECTED_MIN_TRADES = 270


def make_signal_fn(*args, **kwargs):
    """Always raises -- this family has no per-bar signal. See the module
    docstring's "Dispatch safety" section; the only legitimate entry point is
    ``scripts/run_treasury_month_end_overlay_abi1.py`` calling ``build_curve``."""
    raise RuntimeError(
        f"{FAMILY_ID}: this family must only be run through its own bespoke "
        "driver (scripts/run_treasury_month_end_overlay_abi1.py), never the "
        "generic scripts/run_abi1_mining.py dispatch -- it has no per-bar "
        "signal, and the generic path would score three plain long ETF "
        "positions instead of the benchmark-plus-overlay account.")


# ----------------------------------------------------------------- calendar

XNYS = "XNYS"


def _normalized(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


@functools.lru_cache(maxsize=32)
def _regular_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    import exchange_calendars as xc      # local import: heavy, only needed here
    calendar = xc.get_calendar(XNYS, start=start, end=end)
    holidays = pd.DatetimeIndex(calendar.regular_holidays.holidays(start, end))
    if holidays.tz is not None:
        holidays = holidays.tz_localize(None)
    return holidays.normalize()


def scheduled_sessions(start, end) -> pd.DatetimeIndex:
    """The exchange's SCHEDULED sessions in ``[start, end]``: every weekday
    except the regular (rule-based) holidays ``exchange_calendars`` publishes
    for XNYS. Unscheduled closures (its ``adhoc_holidays``) are NOT removed --
    they were not known in advance. Takes dates only; it cannot read data."""
    start, end = _normalized(start), _normalized(end)
    if end < start:
        raise ValueError("end must not precede start")
    weekdays = pd.bdate_range(start, end)
    return weekdays.difference(_regular_holidays(start, end))


@functools.lru_cache(maxsize=32)
def _exchange_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    import exchange_calendars as xc      # local import: heavy, only needed here
    calendar = xc.get_calendar(XNYS, start=start, end=end)
    # ``calendar.sessions`` (not ``sessions_in_range``): the latter raises when
    # ``start`` or ``end`` is not itself a session, which is exactly the case a
    # stray vendor bar on a closed day would create
    sessions = pd.DatetimeIndex(calendar.sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    sessions = sessions.normalize()
    return sessions[(sessions >= start) & (sessions <= end)]


def exchange_sessions(start, end) -> pd.DatetimeIndex:
    """The sessions the exchange ACTUALLY held in ``[start, end]``: unlike
    ``scheduled_sessions`` (which keeps the unscheduled closures because they
    were not known in advance) this OMITS them (9/11, the days of mourning,
    Hurricane Sandy). It describes what the DATA must contain; it is never used
    to place a window."""
    start, end = _normalized(start), _normalized(end)
    if end < start:
        raise ValueError("end must not precede start")
    return _exchange_sessions(start, end)


def validate_session_coverage(index) -> None:
    """Refuse a data calendar that is not EXACTLY the exchange's actual session
    set between its first and last dates. Pairwise agreement between the series
    is not enough: a session that EVERY series lacks (a shared vendor hole)
    passes every pairwise check, its multi-day close-to-close move would be
    booked as one bar, and it can fall inside a window without touching its
    entry or exit dates. A known unscheduled closure (Hurricane Sandy 2012 and
    the like) is no gap: the actual session set omits it, so the data must too.

    Raises ``mining.long_overlay.CalendarCoverageError`` -- a ``ValueError``
    with a ``reason_code`` and the dates: ``missing_exchange_session`` (an
    actual session is absent; ``missing_dates`` lists every one) or
    ``bar_on_closed_day`` (a bar exists on a day the exchange was closed;
    ``extra_dates``). Nothing is trimmed, shifted or patched."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if len(idx) == 0:
        raise ValueError("validate_session_coverage needs a non-empty data calendar")
    actual = exchange_sessions(idx[0], idx[-1])
    missing, extra = actual.difference(idx), idx.difference(actual)
    span = f"{idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}"
    if len(missing):
        raise CalendarCoverageError(
            "missing_exchange_session",
            f"the scored span {span} lacks {len(missing)} actual NYSE session(s) "
            f"in EVERY series: {_overlay._listed(missing)}; a session every series "
            "lacks would be booked as one multi-day bar (and can fall inside a "
            "window); refusing to trim, shift or patch it (refetch the history)",
            missing_dates=missing, extra_dates=extra)
    if len(extra):
        raise CalendarCoverageError(
            "bar_on_closed_day",
            f"the scored span {span} has {len(extra)} bar(s) on day(s) the exchange "
            f"was closed: {_overlay._listed(extra)}; refusing to treat a stray "
            "vendor bar as a session (refetch the history)",
            extra_dates=extra)


def month_end_windows(start, end) -> list:
    """``(entry, exit)`` = ``(T-3, T)`` for EVERY calendar month touched by
    ``[start, end]`` (whole months: a month's scheduled last session may lie
    after ``end``), where ``T`` is the month's final scheduled session and
    ``T-3`` the session ``WINDOW_SESSIONS`` sessions before it."""
    start, end = _normalized(start), _normalized(end)
    first = pd.Timestamp(start.year, start.month, 1)
    last = pd.Period(end, freq="M").end_time.normalize()
    sessions = scheduled_sessions(first, last)
    months = sessions.to_period("M")
    windows = []
    for period in months.unique():
        days = sessions[months == period]
        windows.append((days[-(WINDOW_SESSIONS + 1)], days[-1]))
    return windows


def windows_in_data(index) -> dict:
    """The scheduled windows that fall in a data calendar.

    ``windows`` keeps ``(entry, scheduled_exit)`` for every month whose entry
    session lies in the data (an entry ON the first data date is kept; only an
    entry strictly before it is warm-up); a window whose scheduled exit is
    after the last data date is kept as it is (the builder closes it at the
    last close). Months whose entry precedes the first data date are skipped
    (warm-up) and a month whose entry is after the last data date has not
    started. A scheduled entry, or an exit inside the data, that is missing
    from the data calendar is a hole and raises: this score is one-shot and
    permanent, and a silently dropped or shifted window must never reach it.

    The state of the LAST window is reported exactly, in three mutually
    exclusive ways (all false when it completes at or before the last bar):
    ``truncated_final`` is true only when the data ends INSIDE the window (T-2 or
    T-1), i.e. a real trade will be closed early at the last close;
    ``entered_on_last_bar`` is true only when the data ends exactly on the
    entry date, so the window has entered but the builder opens nothing (its
    entry close is the last close) and it produces no trade. The number of
    trades therefore equals ``len(windows)`` minus ``entered_on_last_bar``."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if len(idx) == 0:
        raise ValueError("windows_in_data needs a non-empty data calendar")
    present = set(idx)
    windows, skipped, not_started = [], 0, 0
    for entry, exit_ in month_end_windows(idx[0], idx[-1]):
        if entry < idx[0]:
            skipped += 1
            continue
        if entry > idx[-1]:
            not_started += 1
            continue
        if entry not in present:
            raise ValueError(
                f"the scheduled window for {entry:%Y-%m} needs its entry "
                f"session {entry.date()} in the data, but it is missing "
                "(vendor hole or unscheduled closure); refusing to shift "
                "or drop the window")
        if exit_ <= idx[-1] and exit_ not in present:
            raise ValueError(
                f"the scheduled window for {entry:%Y-%m} needs its exit "
                f"session {exit_.date()} in the data, but it is missing "
                "(vendor hole or unscheduled closure); refusing to shift "
                "or drop the window")
        windows.append((entry, exit_))
    last_entry, last_exit = windows[-1] if windows else (None, None)
    return {
        "windows": windows,
        "skipped_before_data": skipped,
        "not_started": not_started,
        # a real trade closed early at the last close: the data ends INSIDE
        # the window (after its entry date, before its scheduled exit)
        "truncated_final": bool(windows and last_entry < idx[-1] < last_exit),
        # entered on the very last bar: nothing left to earn, no trade
        "entered_on_last_bar": bool(windows and last_entry == idx[-1]),
    }


# ------------------------------------------------------------------ curves

def _prepare(spy_frame, leg_frames, irx_frame):
    if set(leg_frames) != set(TRADED):
        raise ValueError(
            f"leg_frames must be exactly the registered legs {', '.join(TRADED)}; "
            f"got {', '.join(sorted(map(str, leg_frames)))}")
    if not isinstance(irx_frame, pd.DataFrame) or "Close" not in irx_frame.columns:
        raise ValueError("irx_frame must be a DataFrame with a Close column")
    legs = {symbol: leg_frames[symbol] for symbol in TRADED}
    idx = _overlay.common_calendar(spy_frame, legs)
    validate_session_coverage(idx)
    info = windows_in_data(idx)
    financing = _overlay.causal_financing_rate(irx_frame["Close"], idx)
    return legs, info, financing


def build_curve(spy_frame, leg_frames, irx_frame, *, leg_cost_bps,
                spy_entry_cost_bps) -> dict:
    """The family's SCORED account curve, built with the PINNED
    ``PARAMETERS`` (so a driver cannot accidentally run different constants
    than the catalog entry records). ``irx_frame`` is the raw ^IRX frame (Close
    in percent), loaded WITHOUT the positivity gate. The returned dict is
    ``mining.long_overlay.build_long_overlay_curve``'s, plus ``window_info``
    (``windows_in_data``'s report)."""
    legs, info, financing = _prepare(spy_frame, leg_frames, irx_frame)
    curve = _overlay.build_long_overlay_curve(
        spy_frame, legs, info["windows"], leg_cost_bps=leg_cost_bps,
        spy_entry_cost_bps=spy_entry_cost_bps, financing_rate_annual=financing,
        overlay_notional=PARAMETERS["overlay_notional_fraction"])
    curve["window_info"] = info
    return curve


def build_next_open_sensitivity(spy_frame, leg_frames, irx_frame, *,
                                leg_cost_bps, spy_entry_cost_bps) -> dict:
    """REPORT-ONLY disclosed sensitivity: the identical account with the
    overlay bought at the OPEN of the first window session (the engine's
    next-open convention) instead of the entry close. Never scored."""
    legs, info, financing = _prepare(spy_frame, leg_frames, irx_frame)
    curve = _overlay.build_next_open_sensitivity_curve(
        spy_frame, legs, info["windows"], leg_cost_bps=leg_cost_bps,
        spy_entry_cost_bps=spy_entry_cost_bps, financing_rate_annual=financing,
        overlay_notional=PARAMETERS["overlay_notional_fraction"])
    curve["window_info"] = info
    return curve
