"""Data-coverage check for the ``accrual_anomaly_sloan`` candidate
(Sloan 1996 accrual anomaly: rank equities by total accruals scaled by
average total assets, long the lowest-accrual/highest-cash-flow-quality
subset). Mirrors ``retail_inventory_coverage.py`` exactly -- same
"declared pool -> per-member fetch -> usable/not-usable status -> a
disclosed coverage_fraction, never a guessed one" shape -- swapping that
module's two fundamentals (inventory, revenue) for this candidate's
three (net income, cash flow from operations, total assets).

Accrual formula (default, matching Sloan's own
1996 methodology re-derived via ``mining.batch5.accrual_anomaly_sloan``'s
own module docstring): ``(NetIncomeLoss - CFO) / average(total assets)``,
the standard cash-flow-statement-based accrual proxy (Hribar & Collins
2002) computed directly from three well-tagged XBRL concepts
(``edgar_data.NET_INCOME_TAGS`` / ``CFO_TAGS`` / ``TOTAL_ASSETS_TAGS``),
rather than Sloan's original balance-sheet-change formula (which needs
several additional tags -- current assets, cash, current liabilities,
short-term debt, depreciation -- and is kept out of scope here; see that module's own docstring for the full
citation).

A company is coverage-ok only when all THREE legs each have at least one
disclosed annual (10-K/10-K-A, fp="FY") fact under one of their fallback
tags. Unlike the sibling capex/inventory/revenue coverage checks (which
need a YoY GROWTH ratio and so require >= 2 distinct fiscal years), this
candidate reads each leg's own LATEST known LEVEL directly -- a single
disclosed fiscal year is already usable for net income/CFO. The total-
assets leg still benefits from a SECOND fiscal year (Sloan's own
"average total assets" scaling convention, mean of beginning/ending
assets), but a single year is not treated as unusable either -- the
signal module's own ``accrual_ratio_at`` falls back to the latest known
assets alone when no prior year is known yet, a disclosed approximation
rather than a coverage failure.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import pandas as pd

from . import edgar_data
from . import sec_contact
from .edgar_point_in_time import annual_facts_only, build_point_in_time_annual_series
from .retail_inventory_coverage import build_point_in_time_instant_series

FetchTagsFn = Callable[[int, tuple], dict]
"""Same shape as ``edgar_data.fetch_all_tags_for_company``'s own return
value -- used for the two DURATION legs (net income, CFO)."""

FetchInstantTagsFn = Callable[[int, tuple], dict]
"""Same shape as ``edgar_data.fetch_all_instant_tags_for_company``'s own
return value -- used for the one INSTANT leg (total assets)."""


def _select_and_build_duration_series(by_tag: dict, priority: tuple
                                      ) -> "tuple[Optional[pd.DataFrame], Optional[str]]":
    """One company's point-in-time annual series for a DURATION concept
    (net income or CFO). Returns the full ``build_point_in_time_annual_
    series`` output (``usable_asof``, ``latest_val``, plus its other
    columns, e.g. ``capex_growth_yoy``/``prior_fiscal_year_end``,
    unchanged) -- callers of THIS function only ever read ``latest_val``
    (accrual needs each period's LEVEL, not its own YoY growth), but the
    function itself does not drop the unused columns. Returns
    ``(None, None)`` if no tag in ``priority`` has a usable annual
    series.

    Deliberately does NOT call ``edgar_data.select_priority_series`` (used
    by every sibling coverage check) -- that function picks the first tag
    in ``priority`` that is merely NON-EMPTY, with no concept of "annual-
    usable". Example from real filings data: for some companies
    ``NetIncomeLoss`` returns only non-annual (10-Q) disclosures --
    ``annual_facts_only`` correctly reduces it to zero rows -- while
    ``ProfitLoss`` (the fallback) has many annual rows for the SAME company. Delegating tag choice to
    ``select_priority_series`` would commit to ``NetIncomeLoss``, find it
    annual-empty, and give up instead of trying ``ProfitLoss`` --
    silently under-covering a real, disclosed filer. This function walks
    EVERY tag in ``priority`` order itself and only commits to the first
    one that is genuinely annual-usable."""
    for tag in priority:
        raw = by_tag.get(tag)
        if raw is None or raw.empty:
            continue
        annual = annual_facts_only(raw)
        if annual.empty:
            continue
        series = build_point_in_time_annual_series(annual)
        if series.empty or series["latest_val"].isna().all():
            continue
        return series, tag
    return None, None


def _select_and_build_instant_series(by_tag: dict, priority: tuple
                                     ) -> "tuple[Optional[pd.DataFrame], Optional[str]]":
    """INSTANT-concept sibling of ``_select_and_build_duration_series``
    (total assets) -- built on ``retail_inventory_coverage.
    build_point_in_time_instant_series`` unchanged, same usability bar,
    and the SAME reason for not delegating tag choice to
    ``edgar_data.select_priority_series`` (see that function's own
    docstring above)."""
    for tag in priority:
        raw = by_tag.get(tag)
        if raw is None or raw.empty:
            continue
        annual = annual_facts_only(raw)
        if annual.empty:
            continue
        series = build_point_in_time_instant_series(annual)
        if series.empty or series["latest_val"].isna().all():
            continue
        return series, tag
    return None, None


def check_ticker_coverage(ticker: str, cik: "Optional[int]", *,
                          fetch_tags_fn: FetchTagsFn,
                          fetch_instant_tags_fn: FetchInstantTagsFn) -> dict:
    """Coverage outcome for one ticker. Raises only when the SEC contact
    setting is missing; any other fetch failure is
    recorded as its own status, same "drop the member, log why"
    convention as ``retail_inventory_coverage.check_ticker_coverage``.

    Status values: ``no_cik``, ``fetch_error``, ``no_usable_net_income``,
    ``no_usable_cfo``, ``no_usable_assets``, ``ok``. A ticker reaching
    ``ok`` carries all three point-in-time series under
    ``ni_series``/``cfo_series``/``assets_series`` for the driver's own
    fundamentals_map; every other status carries none of them."""
    empty_tags = {"ni_tag_used": None, "cfo_tag_used": None, "assets_tag_used": None}
    if cik is None:
        return {"ticker": ticker, "cik": None, "status": "no_cik", **empty_tags}

    try:
        ni_by_tag = fetch_tags_fn(cik, edgar_data.NET_INCOME_TAGS)
        ni_series, ni_tag = _select_and_build_duration_series(
            ni_by_tag, edgar_data.NET_INCOME_TAGS)
    except sec_contact.SecContactMissing:
        raise
    except Exception as exc:  # noqa: BLE001 -- genuinely any fetch/parse failure
        return {"ticker": ticker, "cik": cik, "status": "fetch_error",
                "detail": f"{type(exc).__name__}: {exc}", **empty_tags}
    if ni_series is None:
        return {"ticker": ticker, "cik": cik, "status": "no_usable_net_income", **empty_tags}

    try:
        cfo_by_tag = fetch_tags_fn(cik, edgar_data.CFO_TAGS)
        cfo_series, cfo_tag = _select_and_build_duration_series(
            cfo_by_tag, edgar_data.CFO_TAGS)
    except sec_contact.SecContactMissing:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "cik": cik, "status": "fetch_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "ni_tag_used": ni_tag, "cfo_tag_used": None, "assets_tag_used": None}
    if cfo_series is None:
        return {"ticker": ticker, "cik": cik, "status": "no_usable_cfo",
                "ni_tag_used": ni_tag, "cfo_tag_used": None, "assets_tag_used": None}

    try:
        assets_by_tag = fetch_instant_tags_fn(cik, edgar_data.TOTAL_ASSETS_TAGS)
        assets_series, assets_tag = _select_and_build_instant_series(
            assets_by_tag, edgar_data.TOTAL_ASSETS_TAGS)
    except sec_contact.SecContactMissing:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "cik": cik, "status": "fetch_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "ni_tag_used": ni_tag, "cfo_tag_used": cfo_tag, "assets_tag_used": None}
    if assets_series is None:
        return {"ticker": ticker, "cik": cik, "status": "no_usable_assets",
                "ni_tag_used": ni_tag, "cfo_tag_used": cfo_tag, "assets_tag_used": None}

    return {"ticker": ticker, "cik": cik, "status": "ok",
            "ni_tag_used": ni_tag, "cfo_tag_used": cfo_tag, "assets_tag_used": assets_tag,
            "ni_series": ni_series, "cfo_series": cfo_series, "assets_series": assets_series}


def run_coverage_check(tickers: list, cik_by_ticker: dict, *,
                       fetch_tags_fn: FetchTagsFn,
                       fetch_instant_tags_fn: FetchInstantTagsFn) -> dict:
    """The coverage-check headline result for a declared universe.
    Every ticker is checked and recorded; ``coverage_fraction`` is always
    ``usable_n / declared_n`` over the FULL declared list (matches
    ``retail_inventory_coverage.run_coverage_check``'s own convention --
    an unresolved CIK counts against coverage, it never shrinks the
    denominator)."""
    results = [
        check_ticker_coverage(
            ticker, cik_by_ticker.get(ticker),
            fetch_tags_fn=fetch_tags_fn, fetch_instant_tags_fn=fetch_instant_tags_fn)
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
