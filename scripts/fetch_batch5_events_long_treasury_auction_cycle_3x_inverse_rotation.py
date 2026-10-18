"""Re-derive the committed event file for batch5 family
``long_treasury_auction_cycle_3x_inverse_rotation`` from the TreasuryDirect
auction query web service, and regenerate its provenance.

    # default: re-derive and DIFF, writing nothing (exit 1 on any mismatch)
    python scripts/fetch_batch5_events_long_treasury_auction_cycle_3x_inverse_rotation.py

    # rewrite data/mining/batch5_events/<family_id>/{events,provenance}.json
    python scripts/fetch_batch5_events_long_treasury_auction_cycle_3x_inverse_rotation.py --write

WHY THIS SCRIPT EXISTS
----------------------
1. Re-derivability. The events file must be reproducible from the repo on
   demand, not a one-shot fetch. The filter is a few lines
   (``originalSecurityTerm`` in the admitted long-bond tenors,
   ``tips == "No"``, ``type == "Bond"``) and the endpoint is public and
   unauthenticated.

2. The right disclosure statistic on the right calendar. A plain WEEKDAY
   count of the announcement-to-auction lead understates the exposure that
   matters, because the rule counts NYSE SESSIONS, not weekdays, and five
   weekdays can be four sessions across a holiday. On the session calendar
   the rule actually uses, the a-5 decision that commits the first pre-leg
   TMV execution can fall BEFORE the formal offering announcement for a
   material share of in-frame auctions, and the first pre-leg execution
   session itself occasionally does. Provenance publishes both bases,
   derived rather than hand-measured.

   That is a causality-EVIDENCE gap, not a price look-ahead: every stored
   date is a realized auction date and the rule reads no price at all. The
   point-in-time source is the quarterly tentative auction schedule, which
   publishes 30-year auction DATES months ahead of their announcements --
   real, and verified live only for the most recent realized auctions. But
   Treasury's API serves no historical tentative schedules, so the match
   cannot be re-verified here for the rest. This script therefore DISCLOSES
   the right statistic on the right calendar and records the residual as
   open, rather than pretending to close it.

3. Integrity guards.
   (a) THE TWO COMMITTED FILES ARE BOUND. The provenance's
       ``events_json_sha256`` is verified against the committed events
       payload on every LOAD (the candidate module's
       ``verify_events_provenance_binding``, one shared definition) and again
       here in check mode, and check mode also diffs the whole re-derived
       provenance, not just the events.
   (b) THE COMPLETENESS CHECKS CARRY THEIR NUMBERS. Both are performed LIVE
       against the API on every run and published WITH their derived
       counts, so a reader can check them.
   (c) A HAND-MEASURED VERIFICATION CLAIM CANNOT GO STALE SILENTLY. The
       tentative-schedule block names the auctions verified by eye. If a
       refetch adds a newer auction, that claim would stop covering the
       most recent data -- so the script derives whether the named
       auctions are still the tail of the record set and REFUSES ``--write``
       when they are not, rather than republishing a stale claim.
   (d) ODD LONG-BOND TENORS ARE ADMITTED. An exact
       ``originalSecurityTerm == "30-Year"`` match would exclude the
       ``29-Year 9-Month`` and ``30-Year 3-Month`` auctions, which ARE the
       long bond. They are admitted (see ``ADMITTED_ORIGINAL_TERMS`` and
       the module's interpretation choice 7), and an unclassified non-TIPS
       Bond tenor is a hard error instead of a silent drop. These odd tenors
       all predate the traded calendar, and provenance publishes the
       measured effect on the decision series under
       ``odd_long_tenor_inclusion``.

Nothing here changes the rule: PARAMETERS, the pre-leg/post-leg windows and
the decision series are untouched, and the derived events.json is
byte-identical to the committed one (that identity is what ``--check``
asserts).

WHAT IT DOES
------------
1. Three GETs of the auction query. The first is the data itself (all
   ``type=Bond`` rows, JSON) and its sha256 is recorded. The other two are
   the completeness checks, run live rather than narrated: a
   ``pagesize=1000`` fetch that must return the identical row set, and an
   ``auctionDate``-restricted fetch that must return exactly the default
   response's rows in that window. Both publish their derived row counts.
2. Applies the pre-registered filter and keeps ONLY the seven date/identity
   fields. No price, yield, amount or bid-to-cover value is ever stored, so
   this file cannot become a back-door price source.
3. Fails closed on structural surprises: an empty response, a missing or
   unparseable date, an announcement dated after its own auction, a duplicate
   auction date, and any non-TIPS Bond ``originalSecurityTerm`` that is in
   neither the admitted nor the excluded list.
4. Derives every statistic provenance publishes -- the weekday-basis lead
   (under its own key, published alongside the session-basis figures rather
   than replaced by them), the NYSE-session-basis lead and the two precedence
   counts (via the module's own report-only ``announcement_lead_report``, so
   the module and the provenance cannot drift), the schedule-regularity
   invariants that make the tentative-schedule argument checkable at all, and
   the odd-tenor inclusion's measured effect on the decision series.
5. Writes events.json and provenance.json atomically, byte-stable
   (``json.dumps(..., indent=1, sort_keys=True)`` plus a trailing newline),
   with the events digest recorded in the provenance so the pair is bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
for _path in (os.path.join(REPO_ROOT, "src"),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining.batch5 import (                                  # noqa: E402
    long_treasury_auction_cycle_3x_inverse_rotation as mod)

FAMILY_ID = mod.FAMILY_ID
EVENTS_DIR = os.path.join(REPO_ROOT, "data", "mining", "batch5_events", FAMILY_ID)
EVENTS_PATH = os.path.join(EVENTS_DIR, "events.json")
PROVENANCE_PATH = os.path.join(EVENTS_DIR, "provenance.json")

SOURCE_URL = ("https://www.treasurydirect.gov/TA_WS/securities/search"
              "?type=Bond&format=json")
USER_AGENT = "batch5-auction-fetch/1.0"
TIMEOUT_SECONDS = 120

# The pre-registered filter (PARAMETERS.event / event_source): 30-year bonds,
# new issues AND reopenings, never TIPS, never 20-year bonds. The admitted
# tenor list is the candidate module's, not a second copy: the loader and the
# fetcher must never disagree about what the long bond is.
ADMITTED_ORIGINAL_TERMS = tuple(mod.ADMITTED_ORIGINAL_TERMS)
# Non-TIPS Bond tenors that are deliberately NOT the long bond. 20-year is
# pre-registered as excluded; the 1980 15-year/14-year bonds are a different
# instrument. Listed rather than implied so a NEW tenor lands in neither list
# and raises, instead of being dropped silently.
EXCLUDED_ORIGINAL_TERMS = ("14-Year 10-Month", "15-Year 1-Month", "20-Year",
                           "20-Year 1-Month")
KEEP_TIPS = "No"
KEEP_TYPE = "Bond"
# The odd long tenors admitted by interpretation choice 7 -- kept as a named
# set purely so provenance can publish their measured effect separately.
ODD_LONG_TENORS = tuple(term for term in ADMITTED_ORIGINAL_TERMS
                        if term != "30-Year")

# Raw field -> stored field. Dates and identity only.
FIELD_MAP = {
    "auctionDate": "auction_date",
    "announcementDate": "announcement_date",
    "issueDate": "issue_date",
    "cusip": "cusip",
    "securityTerm": "security_term",
    "originalSecurityTerm": "original_security_term",
}

# The window over which the disclosure statistics are computed: the monthly
# 30-year cadence, and the only era the traded products exist in (TMF/TMV
# inception 2009-04-16).
STATS_FROM = "2009-01-01"

# The window the live restricted-query completeness check uses. Fixed (not
# derived from the data) so the published count means the same thing run to
# run; it spans every auction of the traded era with room to spare.
RESTRICTED_QUERY_WINDOW = ("2009-01-01", "2026-12-31")

# Default session-calendar source for the session-basis disclosure: the same
# per-symbol OHLC cache the mining driver builds. Gitignored and regenerable,
# so its absence is a hard stop for --write (the disclosure would be missing)
# but only a warning for --check.
DEFAULT_OHLC_DIR = os.path.join(REPO_ROOT, "runs", "mining", "batch5",
                                "artifacts", "ohlc")

# What the tentative-schedule leg of the point-in-time argument was actually
# verified against, and what it was NOT. Hand-recorded because it is a live
# observation of a PDF, not an API result -- and deliberately narrow.
TENTATIVE_SCHEDULE_VERIFICATION = {
    "source": ("Treasury Tentative Auction Schedule of U.S. Treasury "
               "Securities (home.treasury.gov/system/files/221/"
               "Tentative-Auction-Schedule.pdf)"),
    "observed_on": "2026-09-05",
    "what_it_shows": ("30-Year BOND auction dates are published months ahead "
                      "of their own offering announcements (the schedule read "
                      "on 2026-09-05 carried 30-year auction dates through "
                      "2027-01-13, each paired with a still-future "
                      "announcement date), so the auction DATE is public "
                      "before the announcement and the a-5 decision is "
                      "causal on that basis"),
    "realized_auctions_verified_against_it": ["2026-08-13", "2026-09-10"],
    "realized_auctions_not_verified": ("every other auction in this file; "
                                       "Treasury's API serves no historical "
                                       "tentative schedules, so the "
                                       "contemporaneous match cannot be "
                                       "re-derived here"),
    "status": ("OPEN, disclosed limitation -- NOT a closed check. Closing it "
               "would need the archived quarterly refunding statements, one "
               "document per quarter, which are outside this script's scope."),
}


def tentative_schedule_verification(records) -> dict:
    """The hand-recorded observation above, plus the ONE thing about it that
    can be derived: whether the auctions verified by eye are still the most
    recent auctions in the file. A hand observation cannot be re-derived, but
    it CAN go stale -- if a refetch adds a newer auction, the block would
    keep claiming verification of a tail it no longer covers. So the staleness
    is derived and published, and ``main`` refuses to --write when it is
    True."""
    verified = list(TENTATIVE_SCHEDULE_VERIFICATION[
        "realized_auctions_verified_against_it"])
    tail = [record["auction_date"] for record in records][-len(verified):]
    block = dict(TENTATIVE_SCHEDULE_VERIFICATION)
    block["verified_auctions_are_the_most_recent_in_the_file"] = (
        verified == tail)
    block["most_recent_auctions_in_the_file"] = tail
    block["staleness_note"] = (
        "verified_auctions_are_the_most_recent_in_the_file is DERIVED: the "
        "hand-verified dates compared against the tail of the record set. "
        "False means a newer auction has appeared and the observation needs "
        "redoing by eye; --write refuses in that case rather than "
        "republishing a claim that no longer reaches the newest data.")
    return block


# ------------------------------------------------------------------ fetching

def fetch_raw(url: str = SOURCE_URL) -> tuple:
    """``(rows, raw_bytes)`` from one GET. Network -- this script only; the
    candidate module never touches the network."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        raw = response.read()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"auction query returned no rows: {url}")
    return rows, raw


# ------------------------------------------------------------------ shaping

def _iso_date(value, *, what: str) -> str:
    stamp = pd.Timestamp(value)
    if stamp is None or pd.isna(stamp):
        raise RuntimeError(f"{what} is not a parseable date: {value!r}")
    return stamp.normalize().strftime("%Y-%m-%d")


def select_records(rows) -> list:
    """Filtered, field-restricted, date-sorted records. Fails closed on any
    structural surprise rather than dropping a row silently."""
    kept = []
    for row in rows:
        if row.get("tips") != KEEP_TIPS or row.get("type") != KEEP_TYPE:
            continue
        term = row.get("originalSecurityTerm")
        if term in EXCLUDED_ORIGINAL_TERMS:
            continue
        if term not in ADMITTED_ORIGINAL_TERMS:
            raise RuntimeError(
                f"row {row.get('cusip')!r} auctioned "
                f"{row.get('auctionDate')!r} has originalSecurityTerm "
                f"{term!r}, which is in neither ADMITTED_ORIGINAL_TERMS "
                f"{list(ADMITTED_ORIGINAL_TERMS)} nor "
                f"EXCLUDED_ORIGINAL_TERMS {list(EXCLUDED_ORIGINAL_TERMS)}. "
                "Classify it deliberately -- silently dropping an "
                "unrecognized long tenor is exactly the defect this listing "
                "exists to prevent (see the module's interpretation "
                "choice 7)")
        record = {}
        for raw_field, stored in FIELD_MAP.items():
            value = row.get(raw_field)
            if value in (None, ""):
                raise RuntimeError(
                    f"row {row.get('cusip')!r} auctioned "
                    f"{row.get('auctionDate')!r} is missing {raw_field}")
            record[stored] = (_iso_date(value, what=f"{raw_field} of "
                                                    f"{row.get('cusip')!r}")
                              if raw_field.endswith("Date") else str(value))
        reopening = row.get("reopening")
        if reopening not in ("Yes", "No"):
            raise RuntimeError(
                f"row {record['auction_date']} reopening is {reopening!r}, "
                "expected 'Yes' or 'No'")
        record["reopening"] = reopening == "Yes"
        if record["announcement_date"] > record["auction_date"]:
            raise RuntimeError(
                f"announcement_date {record['announcement_date']} falls after "
                f"its own auction_date {record['auction_date']}")
        kept.append(record)
    if not kept:
        raise RuntimeError("the pre-registered filter kept no rows")
    seen = set()
    for record in kept:
        if record["auction_date"] in seen:
            raise RuntimeError(f"duplicate auction_date {record['auction_date']}")
        seen.add(record["auction_date"])
    return sorted(kept, key=lambda record: record["auction_date"])


def build_events(records) -> dict:
    return {
        "auction_dates": [record["auction_date"] for record in records],
        "event": ("30-year Treasury bond auction date (new issues and "
                  "reopenings, non-TIPS)"),
        "family_id": FAMILY_ID,
        "records": records,
    }


# --------------------------------------------------- completeness (live)

def _row_identity(row) -> tuple:
    """The identity of a raw API row, for set comparison across queries."""
    return (str(row.get("cusip")), str(row.get("auctionDate"))[:10],
            str(row.get("issueDate"))[:10],
            str(row.get("originalSecurityTerm")), str(row.get("securityTerm")))


def completeness_checks(rows) -> tuple:
    """``(checks, counts)``: the two page/window completeness checks,
    PERFORMED against the API on every run and reported WITH their derived
    counts -- as readable prose in ``checks`` and as integers in ``counts``,
    so the numbers can be bound mechanically instead of only read.

    Narrating them as one-off observations of an earlier fetch would be
    unfalsifiable prose with the numbers stripped out. Running them live costs two extra GETs and makes the counts checkable.

    Raises rather than reporting a soft failure: a page cap or a window query
    that disagrees with the default response means the row set this file is
    built from may be incomplete, which is not a warning."""
    default_identities = {_row_identity(row) for row in rows}

    paged, _ = fetch_raw(f"{SOURCE_URL}&pagesize=1000")
    paged_identities = {_row_identity(row) for row in paged}
    if paged_identities != default_identities:
        raise RuntimeError(
            f"pagesize=1000 returned {len(paged)} rows whose identity set "
            f"differs from the default response's {len(rows)} rows "
            f"(only-paged={len(paged_identities - default_identities)}, "
            f"only-default={len(default_identities - paged_identities)}); a "
            "page cap may be truncating the fetch")

    start, end = RESTRICTED_QUERY_WINDOW
    restricted_url = (f"{SOURCE_URL}&startDate={start}&endDate={end}"
                      "&dateFieldName=auctionDate")
    restricted, _ = fetch_raw(restricted_url)
    restricted_identities = {_row_identity(row) for row in restricted}
    expected = {identity for identity in default_identities
                if start <= identity[1] <= end}
    if restricted_identities != expected:
        raise RuntimeError(
            f"the auctionDate-restricted query {start}..{end} returned "
            f"{len(restricted)} rows but the default response holds "
            f"{len(expected)} rows in that window "
            f"(only-restricted={len(restricted_identities - expected)}, "
            f"only-default={len(expected - restricted_identities)}); the two "
            "views of the same data disagree")

    counts = {
        "default_response_rows": len(rows),
        "default_response_rows_in_restricted_window": len(expected),
        "pagesize_1000_rows": len(paged),
        "restricted_query_rows": len(restricted),
        "restricted_query_window": [start, end],
    }
    checks = [
        f"pagesize=1000 returned the identical row set on this run: "
        f"{len(paged)} rows, matching the default response's {len(rows)} "
        "row-for-row by (cusip, auctionDate, issueDate, "
        "originalSecurityTerm, securityTerm) -- no page cap in effect at "
        "this size. VERIFIED LIVE by completeness_checks on every run, not "
        "carried over from an earlier fetch.",
        f"startDate={start}&endDate={end}&dateFieldName=auctionDate returned "
        f"{len(restricted)} rows, exactly the {len(expected)} rows of the "
        "default response whose auctionDate falls in that window -- the "
        "windowed and unwindowed views of the same data agree. VERIFIED LIVE "
        "by completeness_checks on every run.",
        "no auction date carries two 30-year securities (rows == unique "
        "dates; enforced fail-closed by select_records)",
        "announcement_date <= auction_date on every kept row (enforced "
        "fail-closed by select_records AND by the module's "
        "load_auction_records)",
        "every non-TIPS Bond originalSecurityTerm in the response is "
        f"classified: admitted {list(ADMITTED_ORIGINAL_TERMS)}, excluded "
        f"{list(EXCLUDED_ORIGINAL_TERMS)}, anything else raises in "
        "select_records rather than being dropped silently",
        "events.json is bound to this file by events_json_sha256, verified "
        "on every load by the module's verify_events_provenance_binding and "
        "again by this script's --check",
        "the two counted checks above are also published as integers under "
        "completeness_check_counts, so the prose and the numbers cannot "
        "drift apart",
    ]
    return checks, counts


# ------------------------------------------------------------- disclosure

def odd_long_tenor_inclusion(records, calendars) -> dict:
    """The measured effect of admitting the odd long tenors (interpretation
    choice 7). Derived, including the decision-series comparison itself: the
    execution-target series is recomputed over the traded calendar with and
    without those auctions and the two are compared, so "no effect" is a
    measurement published here rather than an assertion."""
    odd = [record for record in records
           if record["original_security_term"] in ODD_LONG_TENORS]
    all_dates = [record["auction_date"] for record in records]
    thirty_only = [record["auction_date"] for record in records
                   if record["original_security_term"] == "30-Year"]
    block = {
        "admitted_odd_tenors": list(ODD_LONG_TENORS),
        "auctions_admitted_by_the_widened_filter": [
            {"auction_date": record["auction_date"],
             "original_security_term": record["original_security_term"],
             "security_term": record["security_term"],
             "reopening": record["reopening"]}
            for record in odd],
        "count": len(odd),
        "latest_odd_tenor_auction": (max(r["auction_date"] for r in odd)
                                     if odd else None),
        "odd_tenor_auctions_on_or_after_2009_01_01":
            sum(1 for record in odd if record["auction_date"] >= STATS_FROM),
        "rationale": (
            "A 30-year bond first auctioned off-cycle carries an odd original "
            "tenor (a February-dated bond sold in August is 29-Year 9-Month) "
            "and the 1994/2000 pair are 30-Year 3-Month new issues. All are "
            "the long bond, so an exact-match filter would drop real "
            "events. 20-Year/15-Year/14-Year bonds remain excluded -- "
            "different instruments, and 20-year exclusion is pre-registered."),
    }
    if calendars is None:
        block["decision_series_effect"] = {
            "status": "NOT MEASURED -- no session-calendar source was given"}
        return block
    effect = {}
    for name, calendar in calendars.items():
        with_odd = mod.execution_targets(calendar, all_dates)
        without_odd = mod.execution_targets(calendar, thirty_only)
        differing = sum(1 for a, b in zip(with_odd, without_odd) if a != b)
        effect[name] = {
            "calendar_first_session": calendar[0].strftime("%Y-%m-%d"),
            "execution_sessions_whose_target_changes": differing,
            "identical_decision_series": with_odd == without_odd,
        }
    block["decision_series_effect"] = effect
    block["decision_series_note"] = (
        "Derived by recomputing mod.execution_targets over each calendar "
        "with and without the odd-tenor auctions. Zero changed targets is "
        "expected and is the reason inclusion is safe here: every odd-tenor "
        "auction predates the traded calendar's first session, so "
        "interpretation choice 5 drops it before any target is written. The "
        "counts that DO move are rows_after_filter and unique_auction_dates "
        "(+{}), and the events digest.".format(len(odd)))
    return block

def weekday_lead_stats(records, *, since: str = STATS_FROM) -> dict:
    """The weekday-basis lead statistic, recomputed rather than copied: lead
    in plain business days over every auction dated ``since`` or later.
    Published alongside the session-basis counts, and labelled with why it
    understates the exposure."""
    distribution: dict = {}
    for record in records:
        if record["auction_date"] < since:
            continue
        lead = len(pd.bdate_range(record["announcement_date"],
                                  record["auction_date"])) - 1
        distribution[lead] = distribution.get(lead, 0) + 1
    if not distribution:
        raise RuntimeError(f"no auctions dated {since} or later")
    return {
        "count_ge_5": sum(v for k, v in distribution.items() if k >= 5),
        "count_lt_5": sum(v for k, v in distribution.items() if k < 5),
        "distribution": {str(k): v for k, v in sorted(distribution.items())},
        "max": max(distribution),
        "min": min(distribution),
        "basis_note": ("PLAIN WEEKDAYS, which is NOT the basis the rule acts "
                       "on and understates the exposure: five weekdays can be "
                       "four NYSE sessions across a holiday. See "
                       "announcement_lead_nyse_sessions for the counts that "
                       "matter."),
    }


def schedule_regularity(records, *, since: str = STATS_FROM) -> dict:
    """Derived, checkable regularity of the auction calendar itself. This is
    the part of the point-in-time argument that IS verifiable for every
    auction: a slot this regular is scheduled far ahead, not conjured by the
    announcement. It does NOT establish that the exact day was published in
    advance -- only the tentative schedule does that, and see
    ``tentative_schedule_verification`` for how little of that is verified."""
    stamps = sorted(pd.Timestamp(record["auction_date"]) for record in records
                    if record["auction_date"] >= since)
    if not stamps:
        raise RuntimeError(f"no auctions dated {since} or later")
    months = [stamp.strftime("%Y-%m") for stamp in stamps]
    expected = [str(period) for period in pd.period_range(
        months[0], months[-1], freq="M")]
    return {
        "auctions": len(stamps),
        "first_auction": stamps[0].strftime("%Y-%m-%d"),
        "last_auction": stamps[-1].strftime("%Y-%m-%d"),
        "calendar_months_spanned": len(expected),
        "months_with_an_auction": len(set(months)),
        "months_with_two_or_more_auctions":
            sorted({month for month in months if months.count(month) > 1}),
        "months_with_no_auction": [month for month in expected
                                   if month not in set(months)],
        "day_of_month_min": min(stamp.day for stamp in stamps),
        "day_of_month_max": max(stamp.day for stamp in stamps),
        "note": ("At most one 30-year auction per calendar month, always in a "
                 "narrow first-half-of-month window, across every month "
                 "spanned bar the listed gap(s) -- i.e. the slot is a "
                 "standing schedule, which is the premise the point-in-time "
                 "claim rests on."),
    }


def _session_index(ohlc_dir: str, symbol: str) -> pd.DatetimeIndex:
    path = os.path.join(ohlc_dir, f"{symbol}.csv")
    frame = pd.read_csv(path)
    column = "Date" if "Date" in frame.columns else frame.columns[0]
    return pd.DatetimeIndex(pd.to_datetime(frame[column])).normalize()


def session_calendars(ohlc_dir: str) -> dict:
    """``{"traded_frames": idx, "production": idx}`` -- the intersection of the
    TRADED frames' sessions (the calendar the rotation engine iterates over
    the real data), and that calendar extended with the vendored XNYS
    sessions exactly as ``make_target_fn`` does in production."""
    calendar = None
    for symbol in mod.TRADED:
        dates = _session_index(ohlc_dir, symbol)
        calendar = dates if calendar is None else calendar.intersection(dates)
    calendar = pd.DatetimeIndex(calendar.sort_values().unique())
    if calendar.empty:
        raise RuntimeError(f"no shared sessions across {mod.TRADED} in {ohlc_dir}")
    frames = {symbol: pd.DataFrame(index=_session_index(ohlc_dir, symbol))
              for symbol in mod.TRADED}
    return {"traded_frames": calendar,
            "production": mod.build_session_calendar(frames)}


def build_provenance(records, raw_rows, raw_bytes, events_payload, *,
                     ohlc_dir: str | None, reason: str,
                     checks, counts) -> dict:
    """Every number here is derived from the data just fetched, from a live
    completeness query, or from the module's own report-only helper -- none is
    hand-typed. The one hand-recorded block (the tentative-schedule
    observation) carries a DERIVED staleness flag."""
    calendars = session_calendars(ohlc_dir) if ohlc_dir is not None else None
    provenance = {
        "announcement_lead_business_days_2009_plus":
            weekday_lead_stats(records),
        "auction_dates_on_or_after_2009_01_01":
            sum(1 for r in records if r["auction_date"] >= STATS_FROM),
        "completeness_check_counts": counts,
        "completeness_checks": checks,
        # The binding between this file and events.json. Computed through the
        # candidate module's own helper so the writer and the loader cannot
        # define the digest differently.
        "events_json_sha256": mod.events_digest(events_payload),
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetched_by": (f"{os.path.basename(__file__)} via urllib.request "
                       "(standard library only), one GET; the raw response is "
                       "hashed, not committed"),
        "fields_kept": sorted(set(FIELD_MAP.values()) | {"reopening"}),
        "filter": (f'originalSecurityTerm in {list(ADMITTED_ORIGINAL_TERMS)} '
                   f'and tips == "{KEEP_TIPS}" and type == "{KEEP_TYPE}" '
                   "(keeps new issues AND reopenings, and the long bond under "
                   "every original tenor Treasury records it as -- see "
                   "odd_long_tenor_inclusion; drops 20-year bonds and the "
                   "1980 15-Year/14-Year bonds by explicit exclusion list, "
                   "and raises on any unclassified non-TIPS Bond tenor; TIPS "
                   "are a separate type)"),
        "first_auction_date": records[0]["auction_date"],
        "last_auction_date": records[-1]["auction_date"],
        "note": "Dates only; no prices, yields or amounts are stored.",
        "odd_long_tenor_inclusion": odd_long_tenor_inclusion(records,
                                                             calendars),
        "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_rows": raw_rows,
        "regenerated_by": {
            "script": (f"scripts/{os.path.basename(__file__)}"),
            "reason": reason,
        },
        "rows_after_filter": len(records),
        "schedule_regularity_2009_plus": schedule_regularity(records),
        "source": "TreasuryDirect auction query web service",
        "source_url": SOURCE_URL,
        "tentative_schedule_verification":
            tentative_schedule_verification(records),
        "unique_auction_dates": len({r["auction_date"] for r in records}),
    }
    if calendars is None:
        provenance["announcement_lead_nyse_sessions"] = {
            "status": "NOT COMPUTED -- no session-calendar source was given",
        }
        return provenance
    provenance["announcement_lead_nyse_sessions"] = {
        "basis_note": (
            "The basis the rule ACTUALLY acts on. decision_precedes_"
            "announcement counts auctions whose s(a)-5 close -- the decision "
            "that commits the first pre-leg TMV execution -- falls before the "
            "formal offering announcement; first_execution_precedes_"
            "announcement counts the stricter case where the s(a)-4 execution "
            "session itself does. Derived by the candidate module's "
            "report-only announcement_lead_report, so module and provenance "
            "cannot drift."),
        "calendar_source": (
            "traded_frames = intersection of the TMF/TMV session dates in the "
            "mining run's OHLC cache; production = that calendar extended "
            "past the last observed bar with the vendored sha-pinned XNYS "
            "sessions, exactly as make_target_fn builds it"),
        "traded_frames": mod.announcement_lead_report(
            calendars["traded_frames"], records),
        "production": mod.announcement_lead_report(
            calendars["production"], records),
    }
    return provenance


# ------------------------------------------------------------------- output

def dumps(payload) -> bytes:
    return (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")


VOLATILE_PROVENANCE_KEYS = ("fetched_at", "raw_response_sha256")


def provenance_differences(committed, derived) -> list:
    """Top-level keys where the committed provenance disagrees with a fresh
    derivation, ignoring the two keys that legitimately move on every run
    (``fetched_at`` always; ``raw_response_sha256`` whenever Treasury
    reserializes an unchanged row set). Every other key is derived from data
    or from the module, so a difference is a real drift."""
    keys = sorted(set(committed) | set(derived))
    differences = []
    for key in keys:
        if key in VOLATILE_PROVENANCE_KEYS:
            continue
        if committed.get(key) != derived.get(key):
            differences.append(key)
    return differences


def _write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite events.json and provenance.json "
                             "(default: re-derive and diff only)")
    parser.add_argument("--ohlc-dir", default=DEFAULT_OHLC_DIR,
                        help="directory holding <SYMBOL>.csv session dates "
                             "for the session-basis disclosure")
    parser.add_argument("--reason", default=(
        "Re-derived from the TreasuryDirect auction query web service: "
        "events and provenance regenerated together and bound by "
        "events_json_sha256; the announcement lead published on both the "
        "weekday and the NYSE-session basis; completeness checks performed "
        "live with their derived row counts; the tentative-schedule leg of "
        "the point-in-time claim recorded as an open limitation. PARAMETERS "
        "and the rule are unchanged."),
        help="the 'regenerated_by.reason' string")
    args = parser.parse_args(argv)

    ohlc_dir = args.ohlc_dir if os.path.isdir(args.ohlc_dir) else None
    if ohlc_dir is None:
        message = (f"session-calendar source not found: {args.ohlc_dir} "
                   "(regenerable per-symbol OHLC cache)")
        if args.write:
            print(f"REFUSING TO WRITE: {message}; the session-basis "
                  "disclosure would be missing", file=sys.stderr)
            return 2
        print(f"WARNING: {message}; session-basis disclosure not compared")

    rows, raw = fetch_raw()
    records = select_records(rows)
    checks, counts = completeness_checks(rows)
    events_payload = build_events(records)
    events_bytes = dumps(events_payload)
    provenance = build_provenance(
        records, len(rows), raw, events_payload,
        ohlc_dir=ohlc_dir, reason=args.reason, checks=checks, counts=counts)
    provenance_bytes = dumps(provenance)

    if args.write:
        stale = not provenance["tentative_schedule_verification"][
            "verified_auctions_are_the_most_recent_in_the_file"]
        if stale:
            print("REFUSING TO WRITE: the hand-verified tentative-schedule "
                  "auctions "
                  f"{TENTATIVE_SCHEDULE_VERIFICATION['realized_auctions_verified_against_it']} "
                  "are no longer the most recent auctions in the file "
                  f"({provenance['tentative_schedule_verification']['most_recent_auctions_in_the_file']}). "
                  "Re-observe the tentative schedule against the new tail and "
                  "update TENTATIVE_SCHEDULE_VERIFICATION; republishing the "
                  "old claim would overstate what has been checked.",
                  file=sys.stderr)
            return 2
        _write_atomic(EVENTS_PATH, events_bytes)
        _write_atomic(PROVENANCE_PATH, provenance_bytes)
        print(f"wrote {EVENTS_PATH}\nwrote {PROVENANCE_PATH}")
        print(f"records={len(records)} raw_rows={len(rows)}")
        print(f"events_json_sha256={provenance['events_json_sha256']}")
        return 0

    status = 0
    committed = (open(EVENTS_PATH, "rb").read()
                 if os.path.exists(EVENTS_PATH) else b"")
    if committed == events_bytes:
        print(f"events.json MATCHES the API re-derivation "
              f"({len(records)} records, raw_rows={len(rows)})")
    else:
        status = 1
        print("events.json DIFFERS from the API re-derivation", file=sys.stderr)
        derived_dates = {r["auction_date"] for r in records}
        try:
            committed_dates = set(json.loads(committed)["auction_dates"])
        except Exception:                               # unreadable committed file
            committed_dates = set()
        print(f"  only in API:       {sorted(derived_dates - committed_dates)}",
              file=sys.stderr)
        print(f"  only in committed: {sorted(committed_dates - derived_dates)}",
              file=sys.stderr)
        print("  committed_bytes="
              f"{len(committed)} derived_bytes={len(events_bytes)}",
              file=sys.stderr)

    # The binding, checked on the COMMITTED pair as a reader would meet it.
    # Diffing the events alone would leave provenance free to describe a
    # different file forever.
    try:
        committed_payload = json.loads(committed) if committed else None
        if committed_payload is None:
            raise ValueError(f"no committed events file at {EVENTS_PATH}")
        digest = mod.verify_events_provenance_binding(
            committed_payload, EVENTS_PATH)
        print(f"provenance BINDS the committed events.json (sha256={digest})")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        status = 1
        print(f"provenance does NOT bind the committed events.json: {exc}",
              file=sys.stderr)

    if not os.path.exists(PROVENANCE_PATH):
        print(f"provenance.json is missing at {PROVENANCE_PATH}",
              file=sys.stderr)
        return 1
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as handle:
        committed_provenance = json.load(handle)
    differences = provenance_differences(committed_provenance, provenance)
    if differences:
        status = 1
        print("provenance.json DIFFERS from the re-derivation at "
              f"{differences} (ignoring {list(VOLATILE_PROVENANCE_KEYS)})",
              file=sys.stderr)
    else:
        print("provenance.json MATCHES the re-derivation on every "
              f"non-volatile key (ignored {list(VOLATILE_PROVENANCE_KEYS)})")
    if (committed_provenance.get("raw_response_sha256")
            != provenance["raw_response_sha256"]):
        print("  note: raw_response_sha256 moved (Treasury reserialized the "
              "response); not a failure on its own -- the events comparison "
              "above is what says whether the DATA moved.")
    return status


if __name__ == "__main__":            # pragma: no cover - CLI entry point
    raise SystemExit(main())
