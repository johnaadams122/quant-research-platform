"""SEC EDGAR XBRL company-concept client (capex-cycle
blue-chip dip-buy). Matches ``external_data.py``'s style: raw fetch kept
separate from pure parsing, fail-closed schema validation, injectable
``http_get``/``timeout`` for testing without network.

Endpoint: ``companyconcept`` (one tag, one company, full history) --
verified against real filings. No API
key; SEC REQUIRES a descriptive ``User-Agent`` naming an organization and
a real contact email, or the request 403s -- this is an external
compliance requirement, never invented here. The exact contact string
is passed by every caller explicitly, never hardcoded as a module
default.

Point-in-time correctness (the reason this module exists rather than a
one-line ``requests.get``): every fact carries an ``end`` (fiscal PERIOD
end -- never usable as an as-of date: a period end precedes the filing
that makes the number public, so treating it as an as-of date is a
lookahead error) and a ``filed`` (the SEC-assigned public filing date) -- ONLY ``filed`` may
ever be treated as an as-of date. Verified against SEC's own filing-
acceptance rule: a submission is dated/disseminated the SAME calendar
day if transmitted by 5:30pm ET, else the NEXT business day (up to a
10:00pm hard cutoff) -- so a fact carrying ``filed = T`` was public no
later than ~10:00pm ET on T, comfortably inside this repo's own
next-session-open entry convention with no extra shift needed.

Residual, disclosed-not-fixed risk (matching this repo's own convention
for disclosed edge cases, e.g. ``backtest_engine.py``'s end-of-data
path): a Post-Acceptance Correction can retroactively change a filing's
date-of-record; the ``companyconcept`` endpoint does not expose PAC
history, so this is a known, low-probability, unmitigated gap.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

from . import sec_contact

EDGAR_COMPANY_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json")

# SEC's documented rate limit is 10 req/s per IP, enforced server-side with
# temporary IP throttling on violation -- self-throttle below it, not at it.
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.15  # ~6.7 req/s

REQUIRED_FACT_FIELDS = ("start", "end", "val", "accn", "fy", "fp", "form", "filed")

# For INSTANT (balance-sheet-as-of-a-date) concepts, e.g. InventoryNet --
# no "start". See parse_instant_concept_payload.
REQUIRED_INSTANT_FACT_FIELDS = ("end", "val", "accn", "fy", "fp", "form", "filed")

# Observed in SEC data: a company can and does switch tags
# mid-history (AMZN's PaymentsToAcquirePropertyPlantAndEquipment stops after
# a 2017-04-28-filed 10-Q; AMZN reports capex under PaymentsToAcquire
# ProductiveAssets from FY2016 onward). A fixed single tag per company is
# wrong -- fetch every tag in this list and keep whichever return data.
CAPEX_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquirePropertyPlantAndEquipmentAndCapitalizedInterest",
    "PaymentsForCapitalImprovements",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
)
REVENUE_TAGS = (
    # ASC 606, ~2018+, the REAL dominant modern tag: the entry below it, "...ExcludingTax" (no
    # "Assessed"), was never observed reporting live data across a real
    # 19-company retail sample (WMT, TGT, DG, SBH, CHWY, REAL, KSS, BJ,
    # RVLV, ASO, VSXY, WRBY, CPNG, OLLI, SFM, AEO, PTRN, SVV, TDUP --
    # every one of them reports under "...ExcludingAssessedTax" instead).
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerExcludingTax",  # kept for a filer that
                                                    # might genuinely use it;
                                                    # never observed in practice
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",                                       # legacy, pre-2018
    "SalesRevenueNet",                                 # older legacy tag
    "SalesRevenueGoodsNet",  # legacy, goods-only revenue --
                             # used by e.g. DG, OLLI, SFM, AEO
)

# Observed in SEC data across a real cross-section of XRT's own
# holdings (WMT, TGT, KMX, ANF, CPNG, ARKO, REAL, GRPN, AN, COST, GME, M):
# InventoryNet is the dominant, near-universal tag among companies that
# report inventory at all. Abercrombie & Fitch (CIK 1018840) is a real,
# real example of a company co-reporting InventoryGross
# alongside InventoryNet for overlapping periods (155 vs 28 facts).
#
# A coverage check across all 75 XRT holdings
# found InventoryNet/InventoryGross alone under-covers real, genuine
# inventory-holding retailers that just use a DIFFERENT balance-sheet
# concept -- verified for real companies via their own companyfacts:
# GAP/AutoZone (AZO) report "InventoryFinishedGoodsNetOfReserves" (each
# company's latest reported value checked and found consistent with
# its known real inventory scale, not a small subcomponent -- exact
# figures are deliberately omitted from source); GAP's OLDER "InventoryFinishedGoods"
# stops in 2012 when it switched to the NetOfReserves variant (ordered
# below it so a company using both picks the current, more complete
# series); Ulta Beauty (ULTA) reports under the bare
# "InventoryFinishedGoods" with no NetOfReserves variant at all -- a
# genuine, still-current user of it; Dillard's (DDS)/Kroger (KR)/Murphy
# USA (MUSA)/Dollar Tree (DLTR)/Upbound Group (UPBD) report under
# "RetailRelatedInventoryMerchandise" (DDS's latest reported value also
# checked live and found sane). Kroger ALSO has an InventoryNet series,
# but it is a sparse note-disclosure tag (only 4 facts total, confirmed
# live) rather than its primary balance-sheet line -- real
# tag-naming/scoping inconsistency, exactly this candidate's own
# disclosed risk, not a data error.
#
# Ordered so InventoryNet always wins via select_priority_series below,
# which picks ONE tag rather than merging across tags (see that
# function's own docstring for why a capex-style largest-value merge
# would be wrong here). A genuine marketplace/platform business (ETSY,
# EBAY, Maplebear/CART all observed) legitimately
# reports NONE of these tags -- correctly read as "no inventory", not a
# tagging gap, by the same downstream "no data for this company" path a
# 404 or an empty units.USD already produces.
INVENTORY_TAGS = (
    "InventoryNet",
    "InventoryFinishedGoodsNetOfReserves",
    "RetailRelatedInventoryMerchandise",
    "InventoryFinishedGoods",
    "InventoryGross",
)

# Accrual-anomaly candidate (accrual_anomaly_sloan, Sloan 1996), confirmed
# observed against real MSFT (CIK 789019) and CAT (CIK 18230)
# companyfacts: MSFT reports net income under NetIncomeLoss (340 real
# facts); CAT reports predominantly under the older ProfitLoss tag (220
# facts) with only a partial NetIncomeLoss history (58 facts) -- both
# tags are real and load-bearing, not a theoretical fallback. Order
# matters (edgar_data.select_priority_series picks the first present):
# NetIncomeLoss is the modern/dominant tag, ProfitLoss the legacy one a
# real S&P 500 filer (CAT) still uses for its longer history.
NET_INCOME_TAGS = (
    "NetIncomeLoss",
    "ProfitLoss",
)

# Observed in SEC data: NetCashProvidedByUsedInOperatingActivities
# returns real data for both MSFT (170 facts) and CAT (86 facts); the
# ContinuingOperations variant also returns real, distinct data for both
# (62 / 75 facts) -- kept as a genuine fallback for a filer that reports
# only the continuing-operations cut, not a theoretical placeholder.
CFO_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)

# Observed in SEC data: "Assets" is an INSTANT (balance-sheet-as-of-
# a-date) concept -- like InventoryNet, it has no "start" field (raises
# via parse_company_concept_payload's duration-only validation; must be
# fetched/parsed via the *_instant_* functions) -- and is essentially
# universal for a GAAP filer (142 real facts for MSFT, 156 for CAT); no
# fallback tag is listed because none was needed for the filers sampled.
TOTAL_ASSETS_TAGS = (
    "Assets",
)


def select_priority_series(by_tag: dict, priority: tuple) -> "tuple[Optional[pd.DataFrame], Optional[str]]":
    """Pick ONE tag's raw facts frame out of ``by_tag`` (``fetch_all_tags_
    for_company``'s own output shape: ``{tag: DataFrame}``) by ``priority``
    order -- the first tag in ``priority`` that is present in ``by_tag``
    (and non-empty) wins; every other tag's facts are never read. Returns
    ``(None, None)`` if no tag in ``priority`` is present.

    Deliberately NOT a per-period merge like ``capex_watchlist.resolve_
    tag_conflicts`` (which prefers the LARGER value on a same-period
    conflict -- correct for capex, where a narrower capex sub-component
    tag can understate the real spend). That rule is WRONG for
    InventoryNet vs InventoryGross: Gross is structurally >= Net by
    definition (gross before a valuation/obsolescence reserve), so
    "prefer the larger value" would systematically pick Gross over Net
    every time a company reports both -- exactly backwards from the
    intended "prefer the standard net concept, fall back only when it's
    absent" behavior. A single company's series is therefore built from
    exactly ONE tag's own facts here, never a mix, so the unit/scope of
    what is being measured never silently changes mid-series. General
    enough to reuse for REVENUE_TAGS too (a company reporting more than
    one revenue tag picks by REVENUE_TAGS's own declared modern-to-legacy
    order, same convention)."""
    for tag in priority:
        frame = by_tag.get(tag)
        if frame is not None and not frame.empty:
            return frame, tag
    return None, None


def fetch_company_concept(cik: int, tag: str, *, user_agent: Optional[str] = None,
                          http_get: Callable = requests.get,
                          timeout: float = 30.0) -> Optional[dict]:
    """Raw fetch of one XBRL us-gaap concept for one company. Returns the
    parsed JSON payload, or ``None`` if this company has never reported
    this tag (SEC returns 404 for that case, which is expected and
    routine for a fallback tag list -- distinguished here from every
    other HTTP error, which still raises). ``user_agent`` defaults to the
    SEC_EDGAR_USER_AGENT setting, read at request time; an explicit blank
    value is refused."""
    if user_agent is None:
        user_agent = sec_contact.sec_contact_user_agent()
    if not user_agent or not user_agent.strip():
        raise ValueError(
            "user_agent must be a non-empty, real organization + contact "
            "string -- SEC requires this or the request 403s")
    url = EDGAR_COMPANY_CONCEPT_URL.format(cik=cik, tag=tag)
    response = http_get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    # Observed when screening the real S&P 500 (e.g. BIIB CIK
    # 875045, PaymentsToAcquirePropertyPlantAndEquipment): SEC signals "this
    # company has never disclosed a USD fact under this tag" via a 200 with
    # units.USD == [] just as often as via a 404. A fallback-tag caller
    # needs both to mean the same thing -- "not applicable, try the next
    # tag" -- never a raised error that aborts before later tags are even
    # tried. parse_company_concept_payload's own fail-closed contract for a
    # direct single-tag call is untouched; this only widens what counts as
    # "no data" at the raw-fetch layer, same tier as the 404 branch above.
    units = payload.get("units") if isinstance(payload, dict) else None
    if isinstance(units, dict) and not units.get("USD"):
        return None
    return payload


def _parse_concept_payload(payload: dict, *, cik: int, tag: str,
                           required_fields: tuple, date_columns: tuple) -> pd.DataFrame:
    """Shared fail-closed parse core for both duration facts (``start``+
    ``end``, e.g. capex/revenue) and instant facts (``end`` only, e.g.
    balance-sheet concepts like inventory) -- the payload-shape/cik/tag
    validation and the val/fy/fp/form/accn coercions are identical either
    way; only which date columns exist differs. Any schema drift (missing
    required field, non-numeric val, malformed date) raises rather than
    silently dropping/coercing -- matching ``external_data.
    fetch_fred_series``'s own "an empty/malformed usable result fails
    closed" convention."""
    if not isinstance(payload, dict):
        raise ValueError(f"CIK{cik} {tag}: payload must be a dict")
    if payload.get("cik") != cik:
        raise ValueError(
            f"CIK{cik} {tag}: payload cik {payload.get('cik')!r} != requested {cik}")
    if payload.get("tag") != tag:
        raise ValueError(
            f"CIK{cik} {tag}: payload tag {payload.get('tag')!r} != requested {tag!r}")
    units = payload.get("units")
    if not isinstance(units, dict) or "USD" not in units:
        raise ValueError(f"CIK{cik} {tag}: payload missing units.USD")
    facts = units["USD"]
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"CIK{cik} {tag}: units.USD must be a non-empty list")

    rows = []
    for i, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise ValueError(f"CIK{cik} {tag}: fact[{i}] is not an object")
        missing = [field for field in required_fields if field not in fact]
        if missing:
            raise ValueError(f"CIK{cik} {tag}: fact[{i}] missing field(s) {missing}")
        rows.append({field: fact[field] for field in required_fields})

    frame = pd.DataFrame(rows)
    for date_column in date_columns:
        parsed = pd.to_datetime(frame[date_column], errors="coerce")
        if parsed.isna().any():
            bad = frame.index[parsed.isna()].tolist()
            raise ValueError(
                f"CIK{cik} {tag}: unparseable {date_column!r} at rows {bad}")
        frame[date_column] = parsed
    values = pd.to_numeric(frame["val"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        bad = frame.index[values.isna() | ~np.isfinite(values.fillna(0.0))].tolist()
        raise ValueError(f"CIK{cik} {tag}: non-numeric/non-finite val at rows {bad}")
    frame["val"] = values.astype(float)
    # fy/fp can be genuinely null on facts SEC discloses via an 8-K
    # comparative-year restatement rather than a periodic 10-K/10-Q --
    # for example an AAPL PaymentsToAcquireProductiveAssets fact from a
    # 2015 8-K. A nullable Int64 tolerates that without
    # losing the "a non-numeric fy is malformed" check (errors="raise"
    # still rejects an unparseable value like "abc"); annual_facts_only's
    # own fp=="FY"/form-in-10-K filter is what excludes these rows later.
    frame["fy"] = pd.to_numeric(frame["fy"], errors="raise").astype("Int64")
    frame["fp"] = frame["fp"].astype(object).where(frame["fp"].notna(), None)
    frame["form"] = frame["form"].astype(str)
    frame["accn"] = frame["accn"].astype(str)
    return frame.sort_values("filed", kind="stable").reset_index(drop=True)


def parse_company_concept_payload(payload: dict, *, cik: int, tag: str) -> pd.DataFrame:
    """Fail-closed parse of one ``companyconcept`` JSON payload into a
    DataFrame of raw USD facts (one row per fact, UNFILTERED by form/fp --
    a caller decides which subset it needs). Columns: ``start``, ``end``,
    ``filed`` (all ``pd.Timestamp``), ``val`` (float), ``accn``, ``fy``,
    ``fp``, ``form`` (str). For DURATION concepts only (a period with both
    a start and end date, e.g. capex payments, revenue) -- a fact missing
    ``start`` raises here by design (see
    ``test_fetch_all_tags_skips_a_tag_whose_parse_raises_and_keeps_the_rest``:
    a stray INSTANT-typed fact appearing under a nominally duration-typed
    concept is a real filer tagging anomaly, not something to silently
    accept). Use ``parse_instant_concept_payload`` for a genuine
    balance-sheet ("instant") concept such as inventory, which has no
    ``start`` at all."""
    return _parse_concept_payload(
        payload, cik=cik, tag=tag,
        required_fields=REQUIRED_FACT_FIELDS, date_columns=("start", "end", "filed"))


def parse_instant_concept_payload(payload: dict, *, cik: int, tag: str) -> pd.DataFrame:
    """Fail-closed parse of one ``companyconcept`` JSON payload for an
    INSTANT (balance-sheet-as-of-a-date) concept -- e.g. ``InventoryNet``,
    observed against real SEC data (WMT CIK 104169: every
    one of 193 InventoryNet facts carries ``end`` but no ``start``, unlike
    a duration concept). Columns:
    ``end``, ``filed`` (``pd.Timestamp``), ``val`` (float), ``accn``,
    ``fy``, ``fp``, ``form`` (str) -- no ``start`` column at all, never a
    fabricated one. Same fail-closed schema-drift discipline as
    ``parse_company_concept_payload``, sharing its validation core via
    ``_parse_concept_payload``; the two functions differ only in which
    fields/date-columns they require."""
    return _parse_concept_payload(
        payload, cik=cik, tag=tag,
        required_fields=REQUIRED_INSTANT_FACT_FIELDS, date_columns=("end", "filed"))


def fetch_company_concept_series(cik: int, tag: str, *, user_agent: Optional[str] = None,
                                 http_get: Callable = requests.get,
                                 timeout: float = 30.0) -> Optional[pd.DataFrame]:
    """Fetch + parse in one call. Returns ``None`` (not an empty frame,
    not an exception) when the company has never reported this tag --
    the routine, expected case for a fallback tag list, distinguished
    from a genuine schema/network failure, which still raises."""
    payload = fetch_company_concept(
        cik, tag, user_agent=user_agent, http_get=http_get, timeout=timeout)
    if payload is None:
        return None
    return parse_company_concept_payload(payload, cik=cik, tag=tag)


def fetch_all_tags_for_company(cik: int, tags, *, user_agent: Optional[str] = None,
                               http_get: Callable = requests.get,
                               timeout: float = 30.0,
                               sleep: Callable[[float], None] = time.sleep,
                               min_interval: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
                               ) -> dict:
    """Fetch every tag in ``tags`` for one company, self-throttled to
    ``min_interval`` seconds between requests (SEC's documented 10 req/s
    limit; default targets ~6.7 req/s, comfortably under it). Returns
    ``{tag: DataFrame}`` for every tag that returned data -- a tag this
    company has never reported is simply ABSENT from the result (no
    error, no None placeholder), so ``len(result)`` is directly the
    "how many of the fallback tags actually apply here" count a manifest
    can record. ``sleep``/``min_interval`` are injectable exactly like
    ``external_data.fetch_finra_margin_debt``'s own ``clock`` parameter,
    so a test can assert throttling happened without a real wall-clock
    wait."""
    result = {}
    for i, tag in enumerate(tags):
        if i > 0:
            sleep(min_interval)
        try:
            series = fetch_company_concept_series(
                cik, tag, user_agent=user_agent, http_get=http_get, timeout=timeout)
        except ValueError:
            # A single malformed/anomalous fact under THIS tag (e.g. a
            # filer mistakenly tagging an instant balance under a
            # duration concept -- e.g. TXN CIK
            # 97476 PaymentsToAcquireProductiveAssets, an instant fact
            # with no "start") must not discard data already fetched
            # from this same company's OTHER, cleanly-parsing fallback
            # tags. Skip this tag only, same "not applicable" tier as a
            # 404 or an empty units.USD -- parse_company_concept_payload
            # and fetch_company_concept_series still raise for a DIRECT
            # single-tag caller; only this fallback-tag loop absorbs it.
            series = None
        if series is not None:
            result[tag] = series
    return result


def fetch_instant_concept_series(cik: int, tag: str, *, user_agent: Optional[str] = None,
                                 http_get: Callable = requests.get,
                                 timeout: float = 30.0) -> Optional[pd.DataFrame]:
    """Fetch + parse in one call for an INSTANT concept (e.g.
    ``InventoryNet``). Mirrors ``fetch_company_concept_series`` exactly,
    substituting ``parse_instant_concept_payload`` -- the raw fetch
    (``fetch_company_concept``, including its 404/empty-units.USD "no
    data" handling) is identical for either concept kind, only the parse
    differs."""
    payload = fetch_company_concept(
        cik, tag, user_agent=user_agent, http_get=http_get, timeout=timeout)
    if payload is None:
        return None
    return parse_instant_concept_payload(payload, cik=cik, tag=tag)


def fetch_all_instant_tags_for_company(cik: int, tags, *, user_agent: Optional[str] = None,
                                       http_get: Callable = requests.get,
                                       timeout: float = 30.0,
                                       sleep: Callable[[float], None] = time.sleep,
                                       min_interval: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
                                       ) -> dict:
    """Fetch every tag in ``tags`` for one company, INSTANT-concept version
    of ``fetch_all_tags_for_company`` -- same throttling, same "keep
    whichever tags return data, skip a tag whose single fact is malformed
    without discarding the others" behavior, built on
    ``fetch_instant_concept_series``/``parse_instant_concept_payload``
    instead of the duration-concept pair."""
    result = {}
    for i, tag in enumerate(tags):
        if i > 0:
            sleep(min_interval)
        try:
            series = fetch_instant_concept_series(
                cik, tag, user_agent=user_agent, http_get=http_get, timeout=timeout)
        except ValueError:
            series = None
        if series is not None:
            result[tag] = series
    return result
