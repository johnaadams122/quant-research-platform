"""CFTC Legacy Futures-Only open interest: fetcher, raw-response cache with a
sha256 manifest, and the publication-availability map.

Used by ``mining.batch5.commodity_open_interest_cycle_commodity_bond_overlay``.
Nothing here computes a signal or touches a price; it only answers "which
open-interest report could a trader have READ by this date".

Source
------
CFTC Commitments of Traders, Legacy report, Futures-Only, on the CFTC public
reporting Socrata dataset ``6dca-aqww``. Field used: ``open_interest_all``
(total contracts outstanding, all months, all traders) for the 18 contract
codes in ``CONTRACTS`` (four sectors). Keyless and free.

Contract universe: a present-day survivor set (a limitation, not a deviation)
------------------------------------------------------------------------------
The 18 contracts are the ones that exist and report cleanly today. No membership rule was
applied at each historical date, so a contract that stopped trading before 2026 cannot be in
the set and the growth series is built only from contracts that survived.

Point-in-time status: NOT verified never-revised
------------------------------------------------
The CFTC exposes no vintage archive and documents corrections to Legacy
Futures-Only reports: the report as of 2017-03-28 was re-issued after account
transfer errors, concentration ratios were corrected for 2018-09-18, and a
reporting-firm problem was corrected for 2019-10-15.
Socrata serves the corrected values, not the values as first published.
Whether TOTAL open interest was touched by each correction is unverified; on a 12-month average
across 18 contracts the effect is likely small, but that is a judgement, not a
measurement. This series must therefore never be labelled never-revised
anywhere (module, catalog entry, ledger note): every artifact built from it
carries ``POINT_IN_TIME_STATUS``.

Availability: the D+6 lag and the three publication blackouts
-------------------------------------------------------------
A report dated (as of) Tuesday D is released the Friday after (3:30 pm ET), so
it is treated as usable only from D+6 CALENDAR days, which in the weekly era is a
Monday: three days of margin in a normal week over the normal release, none in a
holiday-shifted week (the report is then dated a Monday, not a Tuesday). The
cache is semi-monthly in its first years (a median gap of 15 days in 1986 and in
1990, weekly by 1994) and the release timing of that era is not verified. The
margin is immaterial to the state: lengthening the lag to 7 days changes one
state in the scored span (the exact-tie month 2019-11-29) and one before it
(1999-11-30), and shortening it to 5 days changes none. Five scored month-end
decisions (2009-11-30, 2011-02-28, 2015-11-30, 2020-11-30 and 2022-02-28) read a report from a US
federal-holiday week whose plain D+6 date falls on the decision day itself (a Monday
month-end; measured on the cached vintage, the other 25 month-ends in the scored span whose
read is exactly six days old are ordinary weeks with the Friday release). The CFTC usually
moves such a release to the Monday afternoon, about 30 minutes before the close on the
decision day (NOT verified against a CFTC notice); withholding those five reports on their
D+6 date changes no state, so no state depends on them. The 2019-11-29 exact tie is also
sensitive to the unverified pre-2000 release timing: an extra 7 or 14 days of lag on the
pre-2000 reports alone flips it (measured; no other scored state moves). That fixed lag is FALSE
during the three periods in which the CFTC suspended or delayed publication,
when the plain D+6 rule would consume reports that were not yet public (weeks
early):

* 2013 federal shutdown: reports as of 2013-10-01 on were first published
  2013-10-25 (press release 6745-13) and the regular schedule resumed 2013-11-08;
* 2018-19 federal shutdown: publication suspended 2018-12-22. The last report
  before it was published 2018-12-21 and publication resumed with the report as
  of Monday 2018-12-24 (the Christmas-shifted report), expected 2019-02-01 (press
  release 7864-19), then one report each Tuesday and Friday until current. The
  per-report dates after 2019-02-01 are inferred from that cadence (the notice
  lists no table) and press coverage puts the backlog as cleared 2019-03-08, so
  ``public_by`` is a conservative outer bound. Three month-ends fall inside it:
  2018-12-31, 2019-01-31 and 2019-02-28;
* 2025-26 federal shutdown: publication interrupted from 2025-10-01 and resumed
  2025-11-19. The CFTC's first plan cleared the backlog with the report
  published 2026-01-23; its accelerated plan (press release 9147-25, 2025-12-09)
  cleared it 2025-12-29 with the report as of 2025-12-23 and returned to the
  normal schedule after that. The table keeps the first plan's outer bound
  (conservative, never earlier than the truth), so the month-end 2025-12-31,
  whose plain-lag report was in fact public, is still carried. Three of its
  month-ends fall inside the ABI-1 holdout.

``PUBLICATION_BLACKOUTS`` records each interval of delayed as-of dates and the
conservative date by which every one of them was public (``public_by``). A
delayed report is treated as available only from ``public_by`` (never earlier
than it really was), and a decision date on which the plain rule would have
used such a report is a BLACKOUT decision date (``blackout_at``): the family
refuses to update its state there and carries the prior state (the more
conservative of the two options considered). The first
and last delayed reports and the day the regular schedule resumed were read
from the CFTC's own notices (press releases 6745-13, 7864-19, 9138-25 and
9147-25); the publication dates of the individual catch-up
reports in between are not published there and are not used. A blackout month-end
reads the last report that was really public, so its growth is zero and the next
month carries the whole catch-up; a CMI whose 12-month base month is a blackout
month-end therefore measures a window longer than 12 months (2014-10-31,
2019-12-31, 2020-01-31 and 2020-02-28 in the scored span; disclosed, not
repaired). Historical (pre-2000s) release cadence is also unverified; it affects
the early threshold baseline and, through it, one scored state (the 2019-11-29 exact tie,
see above), not the causality of any 2006+ decision. The
series has no staleness rule before 2005 (from 2005 on the family's ``require_recent_reads``
refuses a non-blackout month-end that reads a report older than 14 days; real reads are 6 to 10
days old): palladium has real reporting gaps of 84, 168 and
259 days in 2000-2002, so its month-end value is carried stale (up to 247 days,
2002-03-28) with zero growth, then jumps; that lies wholly before 2006 and
treating any contract older than 30 days as missing at a non-blackout month-end
was measured to change no state anywhere in the history (the first state date,
1989-12-29, is unchanged).

Cache
-----
``fetch_and_cache_all`` validates every contract's response BEFORE writing any
file, stores the RAW response bytes exactly as received (one file per code)
and a ``manifest.json`` with each file's sha256, byte length, row count,
first/last as-of date, the query, and the retrieval time.
``load_cached_open_interest`` re-verifies every hash and re-parses every file
(strict schema) and refuses on any mismatch. Reading the cache needs no
network; the write path has no default directory so a test cannot reach the
real cache by omission.

``fetch_and_cache_all`` also refuses a fetch in which the 18 contracts do not all end on
the same as-of date (``contracts_disagree_on_last_asof``), before writing anything: one
contract that stopped early would otherwise be carried forward with zero growth.

Failure modes are reason-coded (``CftcPayloadError.reason_code`` for a bad
response, ``CftcCacheError.reason_code`` for a bad cache) and never patched:
nothing is trimmed, forward-filled or repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
import requests

SOCRATA_DATASET_ID = "6dca-aqww"
SOCRATA_URL = f"https://publicreporting.cftc.gov/resource/{SOCRATA_DATASET_ID}.json"
DATE_FIELD = "report_date_as_yyyy_mm_dd"
CODE_FIELD = "cftc_contract_market_code"
OI_FIELD = "open_interest_all"
NAME_FIELD = "market_and_exchange_names"
SELECT_FIELDS = (DATE_FIELD, CODE_FIELD, OI_FIELD, NAME_FIELD)
# Socrata silently truncates at $limit; the longest real history is under 2,000
# rows, so a response AT the limit can only be a truncated one.
PAGE_LIMIT = 50000

AVAILABILITY_LAG_DAYS = 6
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "mining", "cftc_oi")

POINT_IN_TIME_STATUS = (
    "not_verified_never_revised: the CFTC exposes no vintage archive and documents corrections to "
    "Legacy Futures-Only reports (2017-03-28, 2018-09-18, 2019-10-15); Socrata serves the corrected "
    "values, not as first published. Release timing is handled by the D+6 lag and three publication "
    "blackouts, not by a vintage history")
DATE_SEMANTICS = "position_as_of_tuesday_not_release_date"

SECTORS = ("agriculture", "energy", "livestock", "metals")


@dataclass(frozen=True)
class Contract:
    code: str
    label: str
    sector: str
    # any one of these (upper case) must appear in every row's market name
    name_keywords: tuple


CONTRACTS = (
    Contract("002602", "corn", "agriculture", ("CORN",)),
    Contract("005602", "soybeans", "agriculture", ("SOYBEANS",)),
    Contract("001602", "wheat_srw", "agriculture", ("WHEAT",)),
    Contract("080732", "sugar", "agriculture", ("SUGAR",)),
    Contract("083731", "coffee", "agriculture", ("COFFEE",)),
    Contract("033661", "cotton", "agriculture", ("COTTON",)),
    Contract("073732", "cocoa", "agriculture", ("COCOA",)),
    Contract("067651", "wti_crude", "energy", ("CRUDE OIL", "WTI")),
    Contract("023651", "natural_gas", "energy", ("NATURAL GAS", "NAT GAS")),
    Contract("022651", "ny_harbor_ulsd", "energy", ("HEATING OIL", "ULSD", "USLD")),
    Contract("111659", "rbob_gasoline", "energy", ("RBOB",)),
    Contract("057642", "live_cattle", "livestock", ("LIVE CATTLE",)),
    Contract("054642", "lean_hogs", "livestock", ("LEAN HOGS",)),
    Contract("088691", "gold", "metals", ("GOLD",)),
    Contract("084691", "silver", "metals", ("SILVER",)),
    Contract("085692", "copper", "metals", ("COPPER",)),
    Contract("076651", "platinum", "metals", ("PLATINUM",)),
    Contract("075651", "palladium", "metals", ("PALLADIUM",)),
)
CONTRACT_CODES = tuple(c.code for c in CONTRACTS)
_CONTRACT_BY_CODE = {c.code: c for c in CONTRACTS}


def contracts_by_sector() -> dict:
    """``{sector: [codes]}`` in the pre-registered order."""
    out = {sector: [] for sector in SECTORS}
    for contract in CONTRACTS:
        out[contract.sector].append(contract.code)
    return out


class CftcPayloadError(ValueError):
    """A response the parser refuses (never repaired). ``reason_code`` says why."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


class CftcCacheError(ValueError):
    """A cache the loader refuses: ``cftc_cache_missing``, ``cftc_cache_incomplete``
    or ``cftc_cache_corrupt``."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


# ------------------------------------------------------------- query / parser

def query_params(code: str) -> dict:
    if code not in _CONTRACT_BY_CODE:
        raise ValueError(f"contract code {code!r} is not one of the 18 pre-registered codes")
    return {
        "$select": ",".join(SELECT_FIELDS),
        "$where": f"{CODE_FIELD}='{code}'",
        "$order": DATE_FIELD,
        "$limit": str(PAGE_LIMIT),
    }


def _fail(code: str, reason: str, message: str):
    raise CftcPayloadError(reason, f"CFTC {code}: {message}")


def parse_open_interest(code: str, raw: bytes) -> pd.Series:
    """Parse one contract's raw Socrata response into a sorted float Series of
    ``open_interest_all`` indexed by the AS-OF (report) date. Strict: any
    deviation raises ``CftcPayloadError`` with a ``reason_code``."""
    contract = _CONTRACT_BY_CODE.get(code)
    if contract is None:
        raise ValueError(f"contract code {code!r} is not one of the 18 pre-registered codes")
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError):
        _fail(code, "payload_not_json", "the response is not JSON")
    if not isinstance(rows, list):
        _fail(code, "payload_not_a_list", f"expected a JSON list of rows, got {type(rows).__name__}")
    if not rows:
        _fail(code, "payload_empty", "the response has no rows")
    if len(rows) >= PAGE_LIMIT:
        _fail(code, "page_limit_reached",
              f"{len(rows)} rows reached the {PAGE_LIMIT} row page limit, so the response may be "
              "truncated; refusing to use it")
    dates, values = [], []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            _fail(code, "row_not_an_object", f"row {i} is {type(row).__name__}, not an object")
        missing = [f for f in SELECT_FIELDS if f not in row]
        if missing:
            _fail(code, "row_missing_field", f"row {i} lacks {missing}")
        if row[CODE_FIELD] != code:
            _fail(code, "wrong_contract_code", f"row {i} is contract {row[CODE_FIELD]!r}")
        name = str(row[NAME_FIELD]).upper()
        if not any(keyword in name for keyword in contract.name_keywords):
            _fail(code, "unexpected_contract_name",
                  f"row {i} names {row[NAME_FIELD]!r}, which contains none of {list(contract.name_keywords)}")
        try:
            stamp = pd.Timestamp(row[DATE_FIELD])
        except (ValueError, TypeError):
            stamp = pd.NaT
        if pd.isna(stamp) or stamp.tzinfo is not None:
            _fail(code, "unparseable_date", f"row {i} date {row[DATE_FIELD]!r} is not a plain date")
        try:
            value = float(row[OI_FIELD])
        except (ValueError, TypeError):
            value = float("nan")
        if not math.isfinite(value):
            _fail(code, "non_numeric_open_interest", f"row {i} open interest {row[OI_FIELD]!r} is not finite")
        if value <= 0.0:
            _fail(code, "non_positive_open_interest", f"row {i} open interest {value} is not positive")
        dates.append(stamp.normalize())
        values.append(value)
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=code, dtype=float)
    if series.index.has_duplicates:
        dupes = sorted({d.strftime("%Y-%m-%d") for d in series.index[series.index.duplicated()]})
        _fail(code, "duplicate_as_of_date", f"duplicate as-of dates {dupes[:5]}")
    series = series.sort_index()
    series.attrs.update({
        "source_url": SOCRATA_URL,
        "contract_code": code,
        "contract_label": contract.label,
        "sector": contract.sector,
        "date_semantics": DATE_SEMANTICS,
        "point_in_time_status": POINT_IN_TIME_STATUS,
    })
    return series


# --------------------------------------------------------------- availability

@dataclass(frozen=True)
class PublicationBlackout:
    name: str
    first_delayed_asof: pd.Timestamp
    last_delayed_asof: pd.Timestamp
    # a date by which EVERY delayed report was public (conservative: never earlier
    # than the truth)
    public_by: pd.Timestamp
    evidence: str


PUBLICATION_BLACKOUTS = (
    PublicationBlackout(
        "2013_federal_shutdown", pd.Timestamp("2013-10-01"), pd.Timestamp("2013-10-29"),
        pd.Timestamp("2013-11-08"),
        "CFTC press release 6745-13 (2013-10-23): the first delayed report, as of 2013-10-01 "
        "(originally due 2013-10-04), was published 2013-10-25, two more followed the week of "
        "2013-10-28, at least two more the week of 2013-11-04, and the regular schedule resumed "
        "2013-11-08; the exact publication date of each report in between is not published there, so "
        "public_by is the date the regular schedule resumed, a conservative outer bound"),
    PublicationBlackout(
        "2018_19_federal_shutdown", pd.Timestamp("2018-12-24"), pd.Timestamp("2019-02-26"),
        pd.Timestamp("2019-03-08"),
        "CFTC press release 7864-19 (2019-01-29): the last report before the suspension was published "
        "2018-12-21; publication resumed in chronological order with the report as of Monday 2018-12-24 "
        "(the Christmas-shifted report originally due 2018-12-28), expected 2019-02-01, then one report "
        "each Tuesday and Friday until current. The per-report dates after 2019-02-01 are inferred from "
        "that announced cadence (the notice lists no table), which puts the last delayed report, as of "
        "2019-02-26, on 2019-03-05 and the schedule current again on 2019-03-08, the date press "
        "coverage gives for the backlog being cleared; public_by is therefore a conservative outer "
        "bound, never earlier than any inferred date"),
    PublicationBlackout(
        "2025_26_federal_shutdown", pd.Timestamp("2025-09-30"), pd.Timestamp("2026-01-13"),
        pd.Timestamp("2026-01-23"),
        "publication interrupted from 2025-10-01 and resumed 2025-11-19 with the report as of "
        "2025-09-30; the CFTC's first plan (press release 9138-25) cleared the backlog with the report "
        "published 2026-01-23, and its accelerated plan (press release 9147-25, 2025-12-09) cleared it "
        "2025-12-29 with the report as of 2025-12-23 and returned to the normal schedule after that. "
        "The delayed as-of dates through 2026-01-13 and public_by 2026-01-23 keep the FIRST plan's outer "
        "bound on purpose: that is conservative (never earlier than the truth), and its one consequence "
        "is that the month-end 2025-12-31, whose plain-lag report (as of 2025-12-23) was in fact "
        "public from 2025-12-29, is still carried as a blackout month-end and left out of the median"),
)


def availability_dates(asof_index) -> pd.DatetimeIndex:
    """The first date each report (by as-of date) may be used: as-of plus
    ``AVAILABILITY_LAG_DAYS``, except that a report whose as-of date falls in a
    publication blackout's delayed interval is available only from that
    blackout's ``public_by`` date."""
    asof = pd.DatetimeIndex(asof_index)
    available = asof + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
    values = available.to_numpy(dtype="datetime64[ns]").copy()
    for blackout in PUBLICATION_BLACKOUTS:
        delayed = (asof >= blackout.first_delayed_asof) & (asof <= blackout.last_delayed_asof)
        public_by = np.datetime64(blackout.public_by.to_datetime64(), "ns")
        values = np.where(delayed, np.maximum(values, public_by), values)
    return pd.DatetimeIndex(values)


def availability_date(asof) -> pd.Timestamp:
    return availability_dates(pd.DatetimeIndex([pd.Timestamp(asof)]))[0]


def blackout_at(decision_date) -> Optional[PublicationBlackout]:
    """The blackout, if any, in which the PLAIN D+6 rule would read a report that
    was not yet public on ``decision_date`` (the delayed interval's first report
    is nominally usable from its as-of date plus the lag, and everything is
    public again from ``public_by``). Table-driven: reads no data."""
    date = pd.Timestamp(decision_date)
    for blackout in PUBLICATION_BLACKOUTS:
        start = blackout.first_delayed_asof + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
        if start <= date < blackout.public_by:
            return blackout
    return None


def month_end_open_interest(series_by_code: Mapping, decision_dates) -> tuple:
    """For every decision date, each contract's open interest from the LAST
    report (latest as-of date) already available on that date under
    ``availability_dates``. Returns ``(values, asof_used)``: two DataFrames indexed
    by the decision dates with one column per contract; NaN / NaT before the first
    report is available. At a blackout decision date this is the last report that
    was really public, NOT the one the plain lag would name."""
    decisions = pd.DatetimeIndex(decision_dates)
    values = pd.DataFrame(np.nan, index=decisions, columns=list(series_by_code), dtype=float)
    used = pd.DataFrame(pd.NaT, index=decisions, columns=list(series_by_code), dtype="datetime64[ns]")
    for code, series in series_by_code.items():
        ordered = series.sort_index()
        asof = pd.DatetimeIndex(ordered.index)
        available = availability_dates(asof).to_numpy(dtype="datetime64[ns]")
        asof_values = asof.to_numpy(dtype="datetime64[ns]")
        level = ordered.to_numpy(dtype=float)
        for row, decision in enumerate(decisions):
            usable = available <= np.datetime64(decision.to_datetime64(), "ns")
            if not usable.any():
                continue
            pick = int(np.flatnonzero(usable)[-1])      # ascending as-of order: the last usable is the latest
            values.iat[row, values.columns.get_loc(code)] = level[pick]
            used.iat[row, used.columns.get_loc(code)] = pd.Timestamp(asof_values[pick])
    return values, used


# -------------------------------------------------------------- fetch / cache

def _iso_utc(moment: datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_raw(code: str, *, http_get: Callable = requests.get, timeout: float = 60.0) -> bytes:
    """One contract's raw response bytes (HTTP errors are raised, never swallowed)."""
    response = http_get(SOCRATA_URL, params=query_params(code), timeout=timeout)
    response.raise_for_status()
    return bytes(response.content)


def _atomic_write_bytes(path: str, payload: bytes) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def require_one_last_asof(last_asof_by_code: Mapping) -> None:
    """Refuse a fetch in which the 18 contracts do not all END on the same as-of date: one contract that
    stopped early would otherwise be carried forward (zero growth) at every later month-end while the other
    17 are current. ``last_asof_by_code`` maps each contract code to its newest as-of date."""
    newest = max(last_asof_by_code.values())
    lagging = {code: last for code, last in last_asof_by_code.items() if last != newest}
    if lagging:
        detail = ", ".join(f"{code} ({_CONTRACT_BY_CODE[code].label}) ends {last:%Y-%m-%d}"
                           for code, last in sorted(lagging.items()))
        raise CftcPayloadError(
            "contracts_disagree_on_last_asof",
            f"CFTC: the newest report is as of {newest:%Y-%m-%d} but {len(lagging)} contract(s) end earlier: "
            f"{detail}; every contract must end on the same as-of date, so nothing was written")


def fetch_and_cache_all(cache_dir: str, *, http_get: Callable = requests.get, timeout: float = 60.0,
                        clock: Optional[Callable[[], datetime]] = None) -> dict:
    """Fetch all 18 contracts, validate every response, and only then write the
    raw files and the manifest into ``cache_dir`` (required; no default). A bad
    response for ANY contract, or contracts that do not all end on the same as-of
    date (``contracts_disagree_on_last_asof``), raises before anything is written."""
    now = clock() if clock is not None else datetime.now(timezone.utc)
    staged = {}
    for code in CONTRACT_CODES:
        raw = fetch_raw(code, http_get=http_get, timeout=timeout)
        series = parse_open_interest(code, raw)          # raises CftcPayloadError naming the code
        staged[code] = (raw, series)
    require_one_last_asof({code: series.index[-1] for code, (_, series) in staged.items()})
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "CFTC Commitments of Traders, Legacy report, Futures-Only (public reporting Socrata)",
        "dataset": SOCRATA_DATASET_ID,
        "url": SOCRATA_URL,
        "field": OI_FIELD,
        "retrieved_at": _iso_utc(now),
        "point_in_time_status": POINT_IN_TIME_STATUS,
        "availability": f"as-of date plus {AVAILABILITY_LAG_DAYS} calendar days; three publication "
                        "blackouts handled in mining.cftc_data.PUBLICATION_BLACKOUTS",
        "contracts": {},
    }
    os.makedirs(cache_dir, exist_ok=True)
    for code in CONTRACT_CODES:
        raw, series = staged[code]
        file_name = f"{code}.json"
        _atomic_write_bytes(os.path.join(cache_dir, file_name), raw)
        manifest["contracts"][code] = {
            "file": file_name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": int(len(series)),
            "first_asof": series.index[0].strftime("%Y-%m-%d"),
            "last_asof": series.index[-1].strftime("%Y-%m-%d"),
            "query": query_params(code),
        }
    _atomic_write_bytes(os.path.join(cache_dir, MANIFEST_NAME),
                        (json.dumps(manifest, indent=1, sort_keys=False) + "\n").encode("utf-8"))
    return manifest


@dataclass
class CftcSnapshot:
    series: dict
    manifest: dict
    latest_asof: pd.Timestamp
    point_in_time_status: str


def _cache_fail(reason: str, message: str):
    raise CftcCacheError(reason, message)


def load_cached_open_interest(cache_dir: str = DEFAULT_CACHE_DIR) -> CftcSnapshot:
    """Read the cache, re-verify every sha256 and re-parse every file. Refuses a
    missing cache, an incomplete manifest, a raw file that does not match its
    hash or row count, and a manifest that claims the series is never revised."""
    manifest_path = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        _cache_fail("cftc_cache_missing", f"no {MANIFEST_NAME} under the CFTC cache directory; run "
                    "`python -m mining.cftc_data --default-cache-dir` (network) to create it")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        _cache_fail("cftc_cache_corrupt", f"the CFTC cache manifest is unreadable: {exc}")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        _cache_fail("cftc_cache_corrupt", "the CFTC cache manifest has an unexpected schema version")
    if manifest.get("dataset") != SOCRATA_DATASET_ID:
        _cache_fail("cftc_cache_corrupt", f"the CFTC cache manifest is for dataset {manifest.get('dataset')!r}")
    status = manifest.get("point_in_time_status")
    if not isinstance(status, str) or not status.startswith("not_verified_never_revised"):
        _cache_fail("cftc_cache_corrupt",
                    "the CFTC cache manifest does not carry the not-verified-never-revised point-in-time "
                    "status (a manifest that labels this series never_revised is refused)")
    entries = manifest.get("contracts")
    if not isinstance(entries, dict):
        _cache_fail("cftc_cache_corrupt", "the CFTC cache manifest has no contracts table")
    absent = [code for code in CONTRACT_CODES if code not in entries]
    if absent:
        _cache_fail("cftc_cache_incomplete", f"the CFTC cache manifest lacks contract(s) {absent}")
    series = {}
    for code in CONTRACT_CODES:
        entry = entries[code]
        path = os.path.join(cache_dir, str(entry.get("file", f"{code}.json")))
        if not os.path.isfile(path):
            _cache_fail("cftc_cache_corrupt", f"raw file for contract {code} is missing")
        with open(path, "rb") as handle:
            raw = handle.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry.get("sha256"):
            _cache_fail("cftc_cache_corrupt",
                        f"raw file for contract {code} no longer matches its manifest sha256")
        parsed = parse_open_interest(code, raw)
        if int(entry.get("rows", -1)) != len(parsed):
            _cache_fail("cftc_cache_corrupt",
                        f"contract {code}: manifest row count {entry.get('rows')} != parsed {len(parsed)}")
        series[code] = parsed
    latest = max(s.index[-1] for s in series.values())
    return CftcSnapshot(series=series, manifest=manifest, latest_asof=latest,
                        point_in_time_status=POINT_IN_TIME_STATUS)


def main(argv=None, *, http_get: Callable = requests.get,
         clock: Optional[Callable[[], datetime]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache the 18 CFTC open-interest series.")
    parser.add_argument("--cache-dir", default=None, help="directory to write the raw cache into")
    parser.add_argument("--default-cache-dir", action="store_true",
                        help="write into data/mining/cftc_oi (the real cache)")
    args = parser.parse_args(argv)
    if bool(args.cache_dir) == bool(args.default_cache_dir):
        parser.error("give exactly one of --cache-dir PATH or --default-cache-dir")
    target = DEFAULT_CACHE_DIR if args.default_cache_dir else args.cache_dir
    manifest = fetch_and_cache_all(target, http_get=http_get, clock=clock)
    print(f"cached {len(manifest['contracts'])} contracts, retrieved_at {manifest['retrieved_at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
