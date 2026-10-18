"""Point-in-time annual capex/revenue series construction from raw EDGAR
XBRL facts (capex-cycle family -- the highest-risk, heaviest-tested piece of
this design, isolated deliberately in its own module).

THE CORE HAZARD, verified against real data (AAPL capex): a single fiscal year's figure is NOT disclosed once. A 10-K's
cash-flow statement shows the current year PLUS two prior comparative
years, so the SAME (``start``, ``end``, ``val``) fact recurs across
MULTIPLE filings with different ``filed``/``accn`` -- e.g. AAPL's FY2014
capex (``9,571`` million USD) first appears in the FY2015 10-K (filed 2015-10-28)
as a comparative year, then AGAIN in the FY2016 10-K (filed 2016-10-26).
Naively taking "the latest fact per fiscal year" would push that year's
KNOWN-since date a full year later than when it was actually first public
-- the opposite of a lookahead leak, but still wrong. Naively taking "the
first fact per fiscal year and freezing it" would miss a genuine later
RESTATEMENT (a differing ``val`` for the same fiscal year, filed later).

The correct model: walk every filing chronologically by ``filed`` date as
a REVISION EVENT. At each event, update this company's "currently known"
value for that fiscal year (creating it on first disclosure, overwriting
it on a later restatement, a no-op on an identical re-disclosure). After
each event, the two MOST RECENT fiscal years (by ``end`` date) among
everything currently known determine ``capex_growth_yoy`` as of that
event's ``filed`` date -- never by row-lag (which breaks the moment a
filing discloses 3 comparative years out of chronological encounter
order, or a restatement reorders which value is "latest").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL_FORMS = ("10-K", "10-K/A")
ANNUAL_FP = "FY"


def annual_facts_only(facts: pd.DataFrame) -> pd.DataFrame:
    """Filter a parsed facts frame (``edgar_data.parse_company_concept_
    payload``'s output) to annual (``fp == "FY"``, ``form`` a 10-K/10-K/A)
    rows only -- a deliberate simplification (10-Q cash-flow figures
    are YTD-cumulative and would need duration-differencing to derive a
    standalone period, an extra error surface this module does not take
    on)."""
    if facts.empty:
        return facts
    mask = (facts["fp"] == ANNUAL_FP) & facts["form"].isin(ANNUAL_FORMS)
    return facts.loc[mask].sort_values("filed", kind="stable").reset_index(drop=True)


def build_point_in_time_annual_series(annual_facts: pd.DataFrame) -> pd.DataFrame:
    """Walk annual facts (already filtered via ``annual_facts_only``,
    sorted by ``filed``) as a chronological revision stream and emit one
    row per event that actually CHANGES the two-most-recent-fiscal-years
    view -- ``usable_asof`` (== that event's ``filed``), the latest and
    prior fiscal year ends and values, and ``capex_growth_yoy`` (NaN
    until a second distinct fiscal year is known).

    Consecutive identical (latest_val, prior_val) events are DROPPED (not
    just no-op'd) -- an unchanged comparative re-disclosure in a later
    filing must not manufacture a new, redundant usable_asof row; only a
    genuine restatement or a newly-disclosed fiscal year advances the
    series. Returns an EMPTY frame (not an error) if ``annual_facts`` is
    empty -- a company with no annual disclosures yet is a data fact, not
    a schema violation.

    Grouped by ``filed`` DATE, not processed one fact at a time: a single filing routinely discloses
    MULTIPLE fiscal years at once -- a company's own FIRST 10-K in the
    XBRL record typically does, and it recurs whenever >= 2 comparative
    years appear in one filing (the AAPL-shaped synthetic fixture this
    module's test suite uses has that shape: its first 10-K discloses 3
    distinct fiscal years under one ``filed`` date). The whole filing becomes public ATOMICALLY, so
    processing its facts one at a time and emitting a row after each
    individual fact fabricates intermediate ``capex_growth_yoy`` values
    from partial fold states that were never independently observable --
    e.g. the out-of-order 3-years-in-one-filing case in this module's own
    test suite would produce a spurious 200% intermediate growth reading under
    per-fact processing, with only the LAST of the 3 rows (the real,
    fully-folded value) being genuine. Every fact
    sharing one ``filed`` date is folded into ``known`` together before
    a single row is ever considered.
    """
    columns = ["usable_asof", "latest_fiscal_year_end", "latest_val",
              "prior_fiscal_year_end", "prior_val", "capex_growth_yoy"]
    if annual_facts.empty:
        return pd.DataFrame(columns=columns)

    known: dict = {}  # (start, end) -> val, "currently known" per fiscal year
    rows = []
    last_emitted = None  # (latest_end, latest_val, prior_end, prior_val)

    for filed_date, group in annual_facts.groupby("filed", sort=True):
        for _, fact in group.iterrows():
            key = (fact["start"], fact["end"])
            known[key] = float(fact["val"])

        ordered = sorted(known.items(), key=lambda kv: kv[0][1])  # by end date
        latest_key, latest_val = ordered[-1]
        if len(ordered) >= 2:
            prior_key, prior_val = ordered[-2]
            growth = latest_val / prior_val - 1.0 if prior_val != 0 else np.nan
            current = (latest_key[1], latest_val, prior_key[1], prior_val)
        else:
            prior_key, prior_val, growth = None, None, np.nan
            current = (latest_key[1], latest_val, None, None)

        if current == last_emitted:
            continue
        last_emitted = current
        rows.append({
            "usable_asof": filed_date,
            "latest_fiscal_year_end": latest_key[1],
            "latest_val": latest_val,
            "prior_fiscal_year_end": prior_key[1] if prior_key else pd.NaT,
            "prior_val": prior_val,
            "capex_growth_yoy": growth,
        })

    return pd.DataFrame(rows, columns=columns)


STALENESS_CAP_DAYS = 400  # wide enough to survive a late large-accelerated-
# filer 10-K (~60-75 days after fiscal year end) with real margin; beyond
# this, a fact is treated as too stale to still represent "current".


def point_in_time_asof(series: pd.DataFrame, dates: pd.DatetimeIndex, *,
                       staleness_cap_days: int = STALENESS_CAP_DAYS) -> pd.DataFrame:
    """As-of join: for each date in ``dates``, the most recent ``series``
    row with ``usable_asof <= date`` -- via ``pd.merge_asof`` with
    ``direction="backward"``, the only direction that can structurally
    never pull a future-dated fact. A date whose nearest prior row is
    more than ``staleness_cap_days`` old is treated as UNKNOWN (all value
    columns become NaN, not silently forward-filled forever) -- the same
    "never let missing data masquerade as a value" doctrine
    ``mining.panel``'s FETCH_FAILED/NO_PRINT convention already
    establishes elsewhere in this pipeline. Returns a frame indexed by
    ``dates`` with the same value columns as ``series`` (minus
    ``usable_asof`` itself)."""
    value_columns = [c for c in series.columns if c != "usable_asof"]
    if series.empty:
        return pd.DataFrame(
            {col: np.nan for col in value_columns}, index=pd.DatetimeIndex(dates))

    left = pd.DataFrame({"Date": pd.DatetimeIndex(dates)})
    # kind="stable" defensively (matches annual_facts_only's own stable
    # sort): build_point_in_time_annual_series already guarantees at most
    # one row per usable_asof (it groups by usable_asof), so this
    # should never need to break a tie -- but a stable sort costs nothing
    # here and removes any dependence on pandas' default sort algorithm
    # for correctness.
    right = series.sort_values("usable_asof", kind="stable").reset_index(drop=True)
    merged = pd.merge_asof(
        left, right, left_on="Date", right_on="usable_asof", direction="backward")
    age_days = (merged["Date"] - merged["usable_asof"]).dt.days
    stale = age_days.isna() | (age_days > staleness_cap_days)
    for col in value_columns:
        merged.loc[stale, col] = np.nan
    return merged.set_index("Date")[value_columns]
