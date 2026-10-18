"""Batch5 candidate: long_treasury_auction_cycle_3x_inverse_rotation (rotation_two_asset).

30-year Treasury bond auction cycle (Lou, Yan & Zhang, RFS 2013; Herb, JBF
2025; Fleming, Liu & Nguyen, FRBNY SR 1188, 2026): secondary-market Treasury
prices are pushed down over the ~5 sessions into a scheduled auction and
recover over the ~5 sessions after it. Expressed two-sided through real
leveraged products: hold TMV (-3x) on execution sessions a-4..a (auction day
included), hold TMF (3x) on execution sessions a+1..a+5, otherwise the cash
leg. Daily decision cadence; the decision on session t is evaluated for the
EXECUTION session x = t+1 (the rotation engine fills a target chosen on t at
t+1's open). Post-leg of an earlier auction takes precedence over the pre-leg
of a later one.

Implemented exactly per the catalog entry's ``rule_pseudocode`` with its
``parameters``; no tuning, no extra filters. The decision uses
ONLY the auction calendar and the session calendar -- no price is read.

Interpretation choices
----------------------
1. Cash leg. ``rotation_engine.run_two_asset_rotation`` holds exactly two
   traded assets, so the candidate's three-ticker rule is expressed as
   TRADED = [TMF, TMV] with the cash leg (parameters.cash_asset = BIL)
   mapped to the engine's ``None`` target (flat, zero return). Both
   directional legs -- the pre-auction concession and the post-auction
   recovery, which ARE the documented mechanism -- survive intact, so the
   candidate's ``pre_registered_fallback`` (post-leg only as single_long
   TMF) is NOT invoked: the engine can take a daily calendar target once
   the cash leg is ``None``. Consequence: the T-bill carry BIL would earn on
   the ~11 of every ~21 sessions spent flat is forgone, which understates
   the rule's return slightly and never flatters it. PARAMETERS is still
   the byte-identical candidate dict (it names BIL); this note is the
   record of how BIL is realised.
2. Session index of an auction day. ``s(a)`` is the position of the LAST
   session on or before ``a`` in the session calendar. Treasury auctions are
   held on business days of the bond-market calendar, which can be an NYSE
   holiday (rare); in that case the pre-leg ends at the prior session's
   close and the post-leg begins on the next session, i.e. after the
   auction has actually happened.
3. Session calendar. Built ONCE in ``make_target_fn`` from the DATES of the
   frames handed to the factory (intersection of the traded frames' session
   dates = the calendar the rotation engine actually iterates), extended
   past the last observed bar with the vendored, sha-pinned XNYS calendar
   (``trading_calendar``, 2026-2028) so a trailing leg keeps its true
   session count. Dates only; no price is read here or per bar.
4. Point-in-time, and its UNCLOSED residual. Auction dates are public via
   the quarterly tentative auction schedule (the candidate's named
   ``event_source``) months ahead, so the a-5 decision is causal and no
   price is ever read from the future. The formal offering ANNOUNCEMENT is
   NOT the point-in-time source, and it comes far too late to be one.
   Measured on the NYSE session calendar this rule actually counts -- not
   on plain weekdays -- the a-5 decision precedes the announcement for 46
   of the 208 auctions inside the traded frames (22.1%), and the first
   pre-leg EXECUTION session precedes it for 13 of 208 (6.2%): e.g.
   auction 2016-09-13, announced 2016-09-08, first TMV execution
   2016-09-07; most recent 2026-01-13. (On the PRODUCTION calendar -- the
   same sessions extended with the vendored XNYS sessions, which also
   admits the not-yet-traded 2026-09-10 auction -- it is 47 of 209 and 13
   of 209; provenance.json carries both.) A weekday-basis count (30 of 211
   auctions under five business days) is arithmetically correct but
   understates the count that matters, because five WEEKDAYS can be four
   SESSIONS across a holiday. Both bases are carried in provenance.json and
   both are DERIVED by ``announcement_lead_report`` from the events file,
   not hand-measured.
   Consequence, stated plainly: the rule leans entirely on the tentative
   schedule, and that leg is verified for only the 2 most recent realized
   auctions (2026-08-13 and 2026-09-10, both matching the schedule
   published at the time). Treasury's API serves no historical tentative
   schedules, so the match for the other 206 is NOT re-verified here and
   stays an OPEN, disclosed limitation rather than a closed check. This is
   a causality-EVIDENCE gap, not a price look-ahead: every stored date is a
   realized auction date, and the rule reads no price whatsoever (pinned by
   ``test_target_reads_no_prices_at_all``).
5. Auctions dated before the first session of the calendar are ignored
   (a post-leg that would straddle the data's first bar is dropped rather
   than partially applied); an auction after the last session contributes
   only the pre-leg sessions that exist. A leg open at the last bar is
   liquidated by the engine at that close (``end_of_data``).
6. Both new issues and reopenings count (parameters.event), TIPS and
   20-year bonds do not; see data/mining/batch5_events/<family_id>/
   provenance.json for the exact filter and completeness checks.
7. Odd long-bond original tenors ARE the 30-year bond and are admitted.
   A filter keeping only ``originalSecurityTerm == "30-Year"`` would
   silently drop SEVEN genuine long-bond auctions Treasury
   records under an odd original tenor: four at ``29-Year 9-Month``
   (2007-08-09, 2007-11-08, 2008-08-07, 2008-11-13) and three at
   ``30-Year 3-Month`` (1994-08-11, 2000-02-10, 2000-08-10). Each is the
   long bond -- a February-dated bond first auctioned in August is
   29y9m by construction, and the 1994/2000 pair are 30y3m new issues --
   so dropping them would be a data defect, not a filter. They are kept
   (``ADMITTED_ORIGINAL_TERMS``), and the fetcher REFUSES an unclassified
   non-TIPS Bond tenor rather than dropping it, so an odd tenor can never
   be dropped silently. 15-Year/14-Year bonds and 20-year bonds stay
   excluded: those are different instruments, and 20-year exclusion is
   pre-registered. Effect on the decision series: NONE, from the dates
   alone. All seven auctions fall in 1994-2008, the latest on 2008-11-13,
   and the traded session calendar begins 2009-04-16 (TMF/TMV inception),
   so every one of them is dropped again by interpretation choice 5 before
   any target is written. The execution-target series over the TMF/TMV
   calendar is therefore IDENTICAL with and without them, and every
   statistic provenance publishes is computed from 2009-01-01 or from the
   traded calendar, so none of them moves either. What DOES
   change is ``rows_after_filter``/``unique_auction_dates`` (+7) and the
   events digest. Inclusion was chosen over exclusion because the
   hypothesis is about long-bond supply, and a rule that silently ignores
   a real long-bond auction is wrong even where the omission happens to
   be unobservable.

Cost caveat (recorded, deliberately NOT acted on)
--------------------------------------------------
The inverse leg's trading cost can be as large as the effect the catalog
entry cites. Priced through ``cost_tax_inputs.cost_bps_from_quotes`` (the
driver's round trip is half the entry spread plus half the exit spread),
TMV's quoted spread is typically several times TMF's and far above BIL's.
The catalog entry's cited effect size is the Lou-Yan-Zhang 5-session
pre/post return differential -- 23.84 bps on the 10-year (t = 1.78) and
22.54 bps on the 5-year -- so a wide TMV spread can exceed it before any
leverage adjustment. The pre-leg runs once per in-frame auction (roughly
200 of them since the TMF/TMV inception), so that cost is paid that many
times. This is an economic fact for the pre-registered verdict to weigh,
exactly the risk the catalog entry's honest weakness (c) names ("the
effect is a few basis points of yield per leg, so TMF/TMV spreads
matter"). It is NOT a reason to change the rule here: dropping the
pre-leg, swapping the inverse product or adding a cost filter would all be
tuning after seeing a cost number, and the pre-registered rule is what
must be scored.

Data hygiene
------------
``load_auction_records`` is the single load path and FAILS CLOSED on every
breach it can see, naming the offending row, so a hand-edit or a bad refetch
cannot reach the rule silently: each record must carry EXACTLY the seven
date/identity fields and nothing else (a price, yield, amount or bid-to-cover
field is a refusal -- this file is dates and identity only);
``announcement_date`` must parse and must not fall after its own
``auction_date``; ``original_security_term`` must be one of
``ADMITTED_ORIGINAL_TERMS`` on every row (the long bond and its two odd
original tenors, interpretation choice 7 -- so a 20-year, 15-year or TIPS row
leaking in is an error, never a silent extra event); ``auction_date`` must be
unique; and the convenience ``auction_dates`` list must equal the sorted unique
``auction_date`` values of ``records`` exactly, so the two views of one file
cannot drift apart.

Finally, the events file is BOUND to its provenance file by digest.
``events_json_sha256`` in the sibling provenance.json must equal the sha256 of
the events payload re-serialized canonically (``indent=1, sort_keys=True``
plus a trailing newline -- byte-for-byte what the fetcher writes), and
``load_auction_records`` verifies that on every load, AFTER the field checks
so a structural breach still reports its own specific cause. A missing
provenance file, a missing or malformed digest, and a digest that does not
match are all refusals. This is what stops the two files drifting apart:
regenerating events without regenerating provenance, or hand-editing
either one, fails closed at load instead of quietly serving an events
file no provenance describes. Canonicalization is deliberate -- reformatting
the JSON is not a content change and must not trip the check, while any
change to a date, a cusip or the record set must.

``announcement_lead_report`` is REPORT ONLY. It is never called by
``make_target_fn`` and never by a target lookup, it cannot move a target, and
it exists so the point-in-time disclosure in provenance.json is derived from
the events file instead of typed in by hand. Nothing in this module adds,
removes or tunes a parameter or a filter: the point-in-time gap in
interpretation choice 4 is a disclosure and evidence matter, handled by
disclosing the right statistic on the right calendar, not by dropping an
auction.

The events file is re-derivable from the TreasuryDirect API by
``scripts/fetch_batch5_events_long_treasury_auction_cycle_3x_inverse_rotation.py``
(``--check``, the default, re-derives and diffs without writing anything).

Performance: every execution-session target is precomputed once in the
factory; each ``target_fn`` call is ONE dict lookup keyed by the decision
date read from the truncated frame's last index label. No rolling or
whole-slice work per bar.
"""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

import trading_calendar

FAMILY_ID = "long_treasury_auction_cycle_3x_inverse_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["TMF", "TMV"]          # [3x long leg, -3x leg]; cash leg = None (see note 1)
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "TMF"
# Copied verbatim from the candidate's pre-registered "parameters"
# (byte-identical under json.dumps(sort_keys=True)).
PARAMETERS = {
    "event": "30-year Treasury bond auction date (new issues and reopenings)",
    "event_source": "TreasuryDirect auction query (treasurydirect.gov/auctions/auction-query/), CSV/JSON export, security type Bond, security term 30-Year; non-TIPS history available from 1998-07-27 per the page; dates are known in advance via the tentative auction schedule (free, public, point-in-time)",
    "pre_leg_asset": "TMV",
    "pre_leg_sessions": 5,
    "pre_leg_window": "sessions a-4 through a inclusive (auction day included; auctions close 13:00 ET)",
    "post_leg_asset": "TMF",
    "post_leg_sessions": 5,
    "post_leg_window": "sessions a+1 through a+5 inclusive",
    "cash_asset": "BIL",
    "decision_cadence": "daily",
    "overlap_rule": "post-leg of the earlier auction takes precedence",
    "pre_registered_fallback": "if run_two_asset_rotation cannot take a daily calendar target with three assets, run the post-leg alone as single_long TMF (the leg with the larger documented effect) with the same 5-session window",
}
EXPECTED_MIN_TRADES = 500

_PRE_ASSET = PARAMETERS["pre_leg_asset"]          # TMV
_POST_ASSET = PARAMETERS["post_leg_asset"]        # TMF
_PRE_SESSIONS = int(PARAMETERS["pre_leg_sessions"])    # 5: a-4 .. a
_POST_SESSIONS = int(PARAMETERS["post_leg_sessions"])  # 5: a+1 .. a+5

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
EVENTS_DIR = os.path.join(_REPO_ROOT, "data", "mining", "batch5_events",
                          FAMILY_ID)
EVENTS_PATH = os.path.join(EVENTS_DIR, "events.json")
PROVENANCE_BASENAME = "provenance.json"

# The ONLY fields an events record may carry: dates and identity, never a
# price, yield, amount or bid-to-cover (see "Data hygiene" above).
_RECORD_FIELDS = ("announcement_date", "auction_date", "cusip", "issue_date",
                  "original_security_term", "reopening", "security_term")
# The long bond, under every original tenor Treasury has recorded it as
# (interpretation choice 7). "30-Year" is the modern term; the other two are
# the same instrument sold at an odd original tenor. 20-Year/15-Year/14-Year
# bonds and TIPS are NOT the long bond and stay excluded -- the fetcher
# refuses any non-TIPS Bond tenor that is in neither list.
ADMITTED_ORIGINAL_TERMS = ("29-Year 9-Month", "30-Year", "30-Year 3-Month")


def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _parse_date(value, *, what: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not a parseable date: {value!r}") from exc
    if stamp is None or pd.isna(stamp):
        raise ValueError(f"{what} is not a parseable date: {value!r}")
    return stamp.normalize()


def _first_difference(left, right) -> str:
    for position, (ours, theirs) in enumerate(zip(left, right)):
        if ours != theirs:
            return f"index {position}: {ours!r} != {theirs!r}"
    return f"index {min(len(left), len(right))}: one list is a prefix of the other"


def canonical_events_bytes(payload) -> bytes:
    """The events payload re-serialized in the ONE canonical form the fetcher
    writes (``indent=1, sort_keys=True`` plus a trailing newline). Hashing
    this rather than the file's raw bytes means reformatting the JSON is not
    a content change, while any change to a date, cusip or record is."""
    return (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")


def events_digest(payload) -> str:
    """sha256 of ``canonical_events_bytes(payload)`` -- the value provenance
    records as ``events_json_sha256``. The fetcher and this module compute it
    through this one function so the binding cannot be defined twice."""
    return hashlib.sha256(canonical_events_bytes(payload)).hexdigest()


def provenance_path_for(events_path: str | None = None) -> str:
    """The provenance file bound to ``events_path`` (its sibling)."""
    target = os.path.abspath(events_path or EVENTS_PATH)
    return os.path.join(os.path.dirname(target), PROVENANCE_BASENAME)


def verify_events_provenance_binding(payload, events_path: str | None = None) -> str:
    """FAIL CLOSED unless the sibling provenance.json's ``events_json_sha256``
    is the digest of ``payload``. Without this the two files can
    drift silently -- a regenerated events file described by a stale
    provenance, or a hand-edit to either one. Returns the verified digest."""
    path = provenance_path_for(events_path)
    if not os.path.exists(path):
        raise ValueError(
            f"provenance file is missing next to the events file: {path}; the "
            "events file is only usable when the provenance that describes it "
            "is present and binds it by digest")
    with open(path, "r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not isinstance(provenance, dict):
        raise ValueError(f"{path} must be a JSON object")
    recorded = provenance.get("events_json_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(
            f"{path} carries no usable events_json_sha256 (got {recorded!r}); "
            "the events file cannot be bound to its provenance")
    actual = events_digest(payload)
    if recorded != actual:
        raise ValueError(
            "events.json does not match the events_json_sha256 recorded in "
            f"its provenance: provenance says {recorded}, the events payload "
            f"digests to {actual}. One of the two files changed without the "
            "other -- regenerate both with "
            "scripts/fetch_batch5_events_"
            "long_treasury_auction_cycle_3x_inverse_rotation.py --write")
    return actual


def load_auction_records(path: str | None = None) -> list:
    """Validated auction records from the family's events file, sorted by
    auction date. Dates and identity fields only -- no prices, yields or
    amounts. FAILS CLOSED on every breach it can see, naming the offending
    row, and then on a provenance digest that does not bind this exact
    payload (see the module's "Data hygiene" note). No network."""
    with open(path or EVENTS_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("events.json must be a JSON object")
    if payload.get("family_id") != FAMILY_ID:
        raise ValueError(f"events.json family_id {payload.get('family_id')!r} "
                         f"is not {FAMILY_ID!r}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("events.json must carry a non-empty records list")
    by_date: dict = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("every record must be a JSON object, got "
                             f"{type(row).__name__}")
        if set(row) != set(_RECORD_FIELDS):
            unexpected = sorted(set(row) - set(_RECORD_FIELDS))
            missing = sorted(set(_RECORD_FIELDS) - set(row))
            raise ValueError(
                f"record {row.get('auction_date')!r} has the wrong field set "
                f"(unexpected={unexpected}, missing={missing}); this file "
                "carries dates and identity only")
        auction = _parse_date(row["auction_date"], what="auction_date")
        announced = _parse_date(
            row["announcement_date"],
            what=f"announcement_date of auction {row['auction_date']}")
        if announced > auction:
            raise ValueError(
                f"announcement_date {row['announcement_date']} falls after its "
                f"own auction_date {row['auction_date']}")
        _parse_date(row["issue_date"],
                    what=f"issue_date of auction {row['auction_date']}")
        if row["original_security_term"] not in ADMITTED_ORIGINAL_TERMS:
            raise ValueError(
                f"record {row['auction_date']} original_security_term "
                f"{row['original_security_term']!r} is not one of "
                f"{list(ADMITTED_ORIGINAL_TERMS)}; the filter admits 30-year "
                "bonds only (incl. their odd original tenors)")
        if not isinstance(row["reopening"], bool):
            raise ValueError(f"record {row['auction_date']} reopening must be a "
                             f"bool, got {row['reopening']!r}")
        if auction in by_date:
            raise ValueError(f"duplicate auction_date {row['auction_date']}")
        by_date[auction] = row
    dates = payload.get("auction_dates")
    if not isinstance(dates, list) or not dates:
        raise ValueError("events.json must carry a non-empty auction_dates list")
    expected = [stamp.strftime("%Y-%m-%d") for stamp in sorted(by_date)]
    if list(dates) != expected:
        raise ValueError(
            "events.json auction_dates disagrees with records "
            f"({len(dates)} vs {len(expected)} entries; first difference at "
            f"{_first_difference(list(dates), expected)})")
    # Last, so a structural breach above still reports its own cause rather
    # than a generic digest mismatch.
    verify_events_provenance_binding(payload, path or EVENTS_PATH)
    return [by_date[stamp] for stamp in sorted(by_date)]


def load_auction_dates(path: str | None = None) -> pd.DatetimeIndex:
    """Sorted unique auction dates from the family's events file (dates
    only; the file is re-derivable from the TreasuryDirect query by the
    sibling fetch script, see provenance.json). Validated through
    ``load_auction_records``, so a malformed or self-inconsistent file
    raises here rather than silently changing the calendar. No network."""
    records = load_auction_records(path)
    return _normalize_index(
        [row["auction_date"] for row in records]).sort_values().unique()


def announcement_lead_report(calendar, records) -> dict:
    """REPORT ONLY: how the formal offering announcement sits relative to the
    sessions this rule actually acts on, for the auctions inside
    ``calendar``. Never called by ``make_target_fn`` or by a target lookup --
    it cannot move a target. It exists so provenance.json's point-in-time
    disclosure is DERIVED from the events file.

    ``session_lead_distribution`` counts sessions from the first session on
    or after the announcement to ``s(a)``; ``weekday_lead_distribution``
    counts plain business days (the weekday basis).
    ``decision_precedes_announcement`` counts auctions whose ``s(a)-5``
    decision session (the close that commits the first pre-leg TMV
    execution) falls BEFORE the announcement date;
    ``first_execution_precedes_announcement`` counts the stricter case where
    the ``s(a)-4`` execution session itself does."""
    calendar = _normalize_index(calendar)
    if len(calendar) == 0:
        raise ValueError("announcement_lead_report needs a non-empty calendar")
    first, last = calendar[0], calendar[-1]
    session_lead: dict = {}
    weekday_lead: dict = {}
    in_frame = decided_before = executed_before = 0
    examples = []
    for row in records:
        auction = _parse_date(row["auction_date"], what="auction_date")
        if auction < first or auction > last:
            continue
        in_frame += 1
        announced = _parse_date(
            row["announcement_date"],
            what=f"announcement_date of auction {row['auction_date']}")
        position = int(calendar.searchsorted(auction, side="right")) - 1
        announced_position = int(calendar.searchsorted(announced, side="left"))
        lead = position - announced_position
        session_lead[lead] = session_lead.get(lead, 0) + 1
        weekdays = len(pd.bdate_range(announced, auction)) - 1
        weekday_lead[weekdays] = weekday_lead.get(weekdays, 0) + 1
        decision = position - _PRE_SESSIONS            # commits s(a)-4
        execution = position - (_PRE_SESSIONS - 1)     # s(a)-4 itself
        if decision >= 0 and calendar[decision] < announced:
            decided_before += 1
        if execution >= 0 and calendar[execution] < announced:
            executed_before += 1
            examples.append({
                "auction_date": auction.strftime("%Y-%m-%d"),
                "announcement_date": announced.strftime("%Y-%m-%d"),
                "first_pre_leg_execution_session":
                    calendar[execution].strftime("%Y-%m-%d")})
    return {
        "sessions": int(len(calendar)),
        "calendar_first_session": first.strftime("%Y-%m-%d"),
        "calendar_last_session": last.strftime("%Y-%m-%d"),
        "in_frame_auctions": in_frame,
        "session_lead_distribution":
            {str(key): value for key, value in sorted(session_lead.items())},
        "weekday_lead_distribution":
            {str(key): value for key, value in sorted(weekday_lead.items())},
        "decision_precedes_announcement": decided_before,
        "first_execution_precedes_announcement": executed_before,
        "first_execution_precedes_announcement_examples": examples,
    }


def build_session_calendar(frames: dict, *,
                           extra_sessions=None) -> pd.DatetimeIndex:
    """Intersection of the traded frames' session dates (the engine's own
    calendar), extended past the last observed bar with the vendored XNYS
    calendar (or ``extra_sessions`` when given). Dates only."""
    calendar = None
    for key in TRADED:
        if key not in frames:
            raise ValueError(f"make_target_fn requires a frame for {key}")
        dates = _normalize_index(frames[key].index)
        calendar = dates if calendar is None else calendar.intersection(dates)
    if calendar is None or calendar.empty:
        raise ValueError("cannot build a session calendar from empty frames")
    last_observed = calendar.max()
    if extra_sessions is None:
        extra = pd.DatetimeIndex(
            [pd.Timestamp(s) for s in trading_calendar.load_default_sessions()])
    else:
        extra = _normalize_index(extra_sessions)
    extra = extra[extra > last_observed]
    return pd.DatetimeIndex(calendar.union(extra)).sort_values().unique()


def execution_targets(calendar, auction_dates) -> list:
    """Target held on EACH execution session of ``calendar`` (position i):
    TMV on s(a)-4..s(a), TMF on s(a)+1..s(a)+5, else None; post-legs are
    written after pre-legs so the post-leg of an earlier auction wins any
    overlap (parameters.overlap_rule)."""
    calendar = _normalize_index(calendar)
    auctions = _normalize_index(auction_dates).sort_values().unique()
    n = len(calendar)
    targets: list = [None] * n
    # s(a) = last session on or before a (interpretation choice 2); an
    # auction before the first session is ignored (choice 5).
    positions = calendar.searchsorted(auctions, side="right") - 1
    positions = [int(p) for p in positions if p >= 0]
    for pos in positions:
        for s in range(pos - (_PRE_SESSIONS - 1), pos + 1):
            if 0 <= s < n:
                targets[s] = _PRE_ASSET
    for pos in positions:
        for s in range(pos + 1, pos + 1 + _POST_SESSIONS):
            if 0 <= s < n:
                targets[s] = _POST_ASSET
    return targets


def decision_targets(calendar, auction_dates) -> dict:
    """{decision date t: target for execution session t+1}. The last
    session has no t+1 inside the calendar and maps to None."""
    calendar = _normalize_index(calendar)
    targets = execution_targets(calendar, auction_dates)
    return {calendar[i]: targets[i + 1] for i in range(len(calendar) - 1)}


def make_target_fn(frames: dict, *, auction_dates=None, extra_sessions=None):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history OHLC frames for TRADED; only their DATES are used, to fix
    the session calendar. ``auction_dates`` defaults to the family's
    events file. Per call: one dict lookup, no prices read."""
    calendar = build_session_calendar(frames, extra_sessions=extra_sessions)
    if auction_dates is None:
        auction_dates = load_auction_dates()
    lookup = decision_targets(calendar, auction_dates)

    def target(truncated: dict):
        frame = truncated.get(_POST_ASSET)
        if frame is None:
            frame = next(iter(truncated.values()))
        today = pd.Timestamp(frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        return lookup.get(today.normalize())

    return target
