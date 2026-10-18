"""Data-coverage check for the ``retail_inventory_sales_breadth_short``
candidate (short XRT on sector-wide breadth of retailers whose inventory
growth outpaces sales growth -- Bernard & Thomas 1989 inventory/margin-
compression anomaly). This module is the mandatory, disclosed-up-front
gate: inventory/sales tag naming is inconsistent across retail filers in
SEC XBRL data (some report ``InventoryNet``, others report a different or
differently-scoped tag; genuine marketplace/platform retailers correctly
report neither; fiscal quarter alignment varies) -- this module measures
what fraction of a declared retail universe actually has usable,
consistently-tagged inventory AND sales data, per the mining pipeline's
candidate-screening rule: never trust the derived signal on a
degraded/biased subset without disclosing it, and never invent a coverage number.

Reuses, never reimplements: ``edgar_data.select_priority_series`` (one
tag per company, never a per-period merge -- see that function's own
docstring for why a capex-style largest-value merge is wrong for
inventory), ``edgar_point_in_time.annual_facts_only`` (generic, reused
unchanged for either fundamental) and, for revenue (a DURATION concept,
same shape as capex), ``edgar_point_in_time.build_point_in_time_annual_
series`` unchanged. Inventory needs its own plumbing: SEC's
``InventoryNet`` (and its real
retail-specific siblings, see ``edgar_data.INVENTORY_TAGS``) is an
INSTANT (balance-sheet-as-of-a-date) concept -- no ``start`` field at
all, unlike the duration concepts (capex, revenue) this pipeline parses
elsewhere. This
module's own ``build_point_in_time_instant_series`` mirrors the duration
builder's algorithm exactly, keyed on ``end`` alone; ``edgar_data`` has
matching instant-concept fetch/parse functions (``parse_instant_concept_
payload``, ``fetch_instant_concept_series``,
``fetch_all_instant_tags_for_company``). Both builders' growth column is
kept generically named (``capex_growth_yoy`` for duration,
``growth_yoy`` for instant) -- neither has an opinion on which
fundamental it was built from; this module reads whichever name its own
``kind`` argument implies.

Usable per company = a point-in-time series with at least one row where
the YoY growth ratio is defined, i.e. at least two distinct fiscal years
have ever been disclosed under the chosen tag. A company is "coverage-ok"
only when BOTH its inventory series and its revenue series clear that
bar independently -- the combined inventory-growth-vs-sales-growth signal
this candidate needs is undefined for anything less, matching the
pipeline's own pooled-candidate convention of dropping the MEMBER (not
the family) when a precondition fails, and recording exactly why.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import edgar_data, sec_contact
from .edgar_point_in_time import annual_facts_only, build_point_in_time_annual_series

FetchTagsFn = Callable[[int, tuple], dict]
"""Same shape as ``edgar_data.fetch_all_tags_for_company``'s own return
value keyed by its own ``(cik, tags)`` positional inputs -- injected so
every function in this module stays network-free and unit-testable;
production wiring partial-applies the real fetcher with a real
user_agent/sleep/min_interval, exactly like
``run_capex_cycle_mining.build_fundamentals_map`` already does for capex."""


def build_point_in_time_instant_series(instant_facts: pd.DataFrame) -> pd.DataFrame:
    """INSTANT-concept sibling of ``edgar_point_in_time.
    build_point_in_time_annual_series`` -- same chronological-revision-
    stream algorithm (walk facts sharing one ``filed`` date together,
    emit a row only when the two-most-recent-fiscal-years view actually
    changes, drop a consecutive identical re-disclosure), but keyed on
    ``end`` ALONE rather than ``(start, end)``: an instant fact (e.g.
    ``InventoryNet``) has no ``start`` at all -- confirmed against real
    SEC data (e.g. WMT's InventoryNet facts). Deliberately a separate function rather than a
    branch inside the existing duration-concept builder, which this
    pipeline treats as its highest-risk, heaviest-tested piece and never
    touches when a genuinely different fact shape needs new handling
    (same discipline ``edgar_data.parse_instant_concept_payload`` already
    follows for the parse layer).

    ``instant_facts`` must already be ``annual_facts_only``-filtered
    (that function reads only ``fp``/``form``/``filed``, none of them
    start/end-shaped, so it is reused UNCHANGED for either concept kind).
    Output columns: ``usable_asof``, ``latest_fiscal_year_end``,
    ``latest_val``, ``prior_fiscal_year_end``, ``prior_val``,
    ``growth_yoy`` (named generically -- this function has no opinion on
    which fundamental it was built from; a caller decides what to call
    it, e.g. ``inventory_growth_yoy``/``sales_growth_yoy``)."""
    columns = ["usable_asof", "latest_fiscal_year_end", "latest_val",
              "prior_fiscal_year_end", "prior_val", "growth_yoy"]
    if instant_facts.empty:
        return pd.DataFrame(columns=columns)

    known: dict = {}  # end -> val, "currently known" per fiscal year
    rows = []
    last_emitted = None

    for filed_date, group in instant_facts.groupby("filed", sort=True):
        for _, fact in group.iterrows():
            known[fact["end"]] = float(fact["val"])

        ordered = sorted(known.items(), key=lambda kv: kv[0])  # by end date
        latest_end, latest_val = ordered[-1]
        if len(ordered) >= 2:
            prior_end, prior_val = ordered[-2]
            growth = latest_val / prior_val - 1.0 if prior_val != 0 else np.nan
            current = (latest_end, latest_val, prior_end, prior_val)
        else:
            prior_end, prior_val, growth = None, None, np.nan
            current = (latest_end, latest_val, None, None)

        if current == last_emitted:
            continue
        last_emitted = current
        rows.append({
            "usable_asof": filed_date,
            "latest_fiscal_year_end": latest_end,
            "latest_val": latest_val,
            "prior_fiscal_year_end": prior_end if prior_end is not None else pd.NaT,
            "prior_val": prior_val,
            "growth_yoy": growth,
        })

    return pd.DataFrame(rows, columns=columns)


_GROWTH_COLUMN_BY_KIND = {
    "duration": "capex_growth_yoy",  # edgar_point_in_time's own generic name
    "instant": "growth_yoy",
}


def select_and_build_annual_series(by_tag: dict, priority: tuple, *,
                                   kind: str = "duration"
                                   ) -> "tuple[Optional[pd.DataFrame], Optional[str]]":
    """One company's point-in-time annual growth series for one fundamental
    (inventory or revenue), or ``(None, None)`` if no tag in ``priority``
    has a usable multi-year annual series. "Usable" requires at least two
    DISTINCT annual (10-K/10-K/A, fp="FY") fiscal years under the chosen
    tag -- a single disclosed year has no YoY ratio to compute, which the
    underlying series builder already represents correctly as an all-NaN
    growth column rather than raising; this function is what turns that
    into the coverage check's binary usable/not-usable read.

    ``kind`` selects which point-in-time builder applies: ``"duration"``
    (default) for a start+end period concept (capex, revenue -- reuses
    ``edgar_point_in_time.build_point_in_time_annual_series`` unchanged),
    ``"instant"`` for a balance-sheet-as-of-a-date concept like
    ``InventoryNet`` (this module's own ``build_point_in_time_instant_
    series`` -- see that function's docstring for why it must be a
    separate builder, not a branch inside the existing one)."""
    if kind not in _GROWTH_COLUMN_BY_KIND:
        raise ValueError(f"kind must be one of {tuple(_GROWTH_COLUMN_BY_KIND)}, got {kind!r}")
    raw, tag = edgar_data.select_priority_series(by_tag, priority)
    if raw is None:
        return None, None
    annual = annual_facts_only(raw)
    if annual.empty:
        return None, None
    if kind == "instant":
        series = build_point_in_time_instant_series(annual)
    else:
        series = build_point_in_time_annual_series(annual)
    growth_column = _GROWTH_COLUMN_BY_KIND[kind]
    if series.empty or series[growth_column].isna().all():
        return None, None
    return series, tag


def check_ticker_coverage(ticker: str, cik: "Optional[int]", *,
                          fetch_tags_fn: FetchTagsFn) -> dict:
    """Coverage outcome for one ticker. Raises only when the SEC contact
    setting is missing; any other fetch failure
    (network error, malformed payload) is recorded as its own status,
    same "drop the member, log why" convention as every other pooled
    precondition in this pipeline.

    Status values: ``no_cik`` (never reaches the network), ``fetch_error``,
    ``no_usable_inventory``, ``no_usable_revenue``, ``ok``. A ticker
    reaching ``ok`` carries both point-in-time series under
    ``inventory_series``/``revenue_series`` for the downstream signal
    driver (scripts/run_retail_inventory_breadth_mining.py) to consume;
    every other status carries neither."""
    if cik is None:
        return {"ticker": ticker, "cik": None, "status": "no_cik",
                "inventory_tag_used": None, "revenue_tag_used": None}
    try:
        inv_by_tag = fetch_tags_fn(cik, edgar_data.INVENTORY_TAGS)
        inv_series, inv_tag = select_and_build_annual_series(
            inv_by_tag, edgar_data.INVENTORY_TAGS, kind="instant")
    except sec_contact.SecContactMissing:
        raise
    except Exception as exc:  # noqa: BLE001 -- genuinely any fetch/parse failure
        return {"ticker": ticker, "cik": cik, "status": "fetch_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "inventory_tag_used": None, "revenue_tag_used": None}
    if inv_series is None:
        return {"ticker": ticker, "cik": cik, "status": "no_usable_inventory",
                "inventory_tag_used": None, "revenue_tag_used": None}

    try:
        rev_by_tag = fetch_tags_fn(cik, edgar_data.REVENUE_TAGS)
        rev_series, rev_tag = select_and_build_annual_series(
            rev_by_tag, edgar_data.REVENUE_TAGS)
    except sec_contact.SecContactMissing:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "cik": cik, "status": "fetch_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "inventory_tag_used": inv_tag, "revenue_tag_used": None}
    if rev_series is None:
        return {"ticker": ticker, "cik": cik, "status": "no_usable_revenue",
                "inventory_tag_used": inv_tag, "revenue_tag_used": None}

    return {"ticker": ticker, "cik": cik, "status": "ok",
            "inventory_tag_used": inv_tag, "revenue_tag_used": rev_tag,
            "inventory_series": inv_series, "revenue_series": rev_series}


def run_coverage_check(tickers: list, cik_by_ticker: dict, *,
                       fetch_tags_fn: FetchTagsFn) -> dict:
    """The coverage-check headline result for a declared retail
    universe. Every ticker is checked and recorded -- there is no silent
    drop anywhere in this function; ``coverage_fraction`` is always
    ``usable_n / declared_n`` over the FULL declared list (an unresolved
    CIK counts against coverage, it does not shrink the denominator),
    matching this pipeline's own ``MIN_POOL_COVERAGE`` convention
    (``scripts/run_batch5_mining.py``) of measuring against the declared
    pool, never a pre-filtered one."""
    results = [
        check_ticker_coverage(ticker, cik_by_ticker.get(ticker), fetch_tags_fn=fetch_tags_fn)
        for ticker in tickers
    ]
    usable = [r for r in results if r["status"] == "ok"]
    declared_n = len(tickers)
    coverage_fraction = (len(usable) / declared_n) if declared_n else 0.0
    return {
        "declared_n": declared_n,
        "usable_n": len(usable),
        "coverage_fraction": coverage_fraction,
        "by_status": dict(Counter(r["status"] for r in results)),
        "results": results,
    }
