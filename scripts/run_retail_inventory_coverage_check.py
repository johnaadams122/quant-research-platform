"""Coverage check (real network run) for the
``retail_inventory_sales_breadth_short`` candidate: for each ticker in
XRT's own real, current holdings (a dated JSON snapshot of State Street's
own daily holdings file, passed via ``--holdings``),
resolve its SEC EDGAR CIK and measure whether it has a usable, point-in-time
inventory series AND a usable revenue series (``mining.retail_inventory_
coverage``). This is the candidate's disclosed real risk:
inventory/sales tag naming is inconsistent across
retail filers -- this script measures the real fraction, it does not
assume one.

Usage (real network calls -- SEC only; takes a few minutes, self-throttled
to SEC's documented rate limit):
    python scripts/run_retail_inventory_coverage_check.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import requests  # noqa: E402

from mining import edgar_cik_resolution, edgar_data, retail_inventory_coverage as ric  # noqa: E402
from mining import sec_contact  # noqa: E402
from mining.edgar_data import DEFAULT_MIN_REQUEST_INTERVAL_SECONDS  # noqa: E402

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_HOLDINGS_PATH = os.path.join(
    REPO_ROOT, "data", "mining", "xrt_holdings.json")
DEFAULT_OUTPUT_PATH = os.path.join(
    REPO_ROOT, "data", "mining", "retail_inventory_coverage.json")


def load_universe(holdings_path: str) -> list:
    with open(holdings_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [h["ticker"] for h in payload["holdings"]]


def fetch_company_tickers(http_get=requests.get) -> dict:
    resp = http_get(COMPANY_TICKERS_URL,
                    headers={"User-Agent": sec_contact.sec_contact_user_agent()},
                    timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_cik_map(tickers: list) -> dict:
    payload = fetch_company_tickers()
    table = edgar_cik_resolution.parse_company_tickers_payload(payload)
    return edgar_cik_resolution.resolve_cik_for_tickers(table, tickers)


def make_live_fetch_tags_fn(*, user_agent: str | None = None,
                            sleep=time.sleep,
                            min_interval: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS):
    """Real-network fetch_tags_fn matching ric.FetchTagsFn's shape,
    self-throttled ACROSS the whole run (a single shared counter, not
    per-company) -- two separate per-fundamental fetches (one for
    inventory, one for revenue) both reset their OWN internal i==0 "no
    sleep before the first request" rule if built independently, which
    would otherwise fire back-to-back requests with zero delay between a
    ticker's last inventory tag and its first revenue tag.

    Dispatches to the INSTANT-concept fetch (``fetch_instant_concept_
    series``) when ``tags is edgar_data.INVENTORY_TAGS`` and the
    DURATION-concept fetch (``fetch_company_concept_series``) otherwise
    -- InventoryNet/InventoryGross are balance-sheet ("instant", no
    ``start``) concepts in SEC's own data, unlike the duration-shaped tags
    (capex, revenue). See ``edgar_data.parse_instant_concept_payload``'s own
    docstring."""
    state = {"count": 0}

    def fetch_tags_fn(cik: int, tags: tuple) -> dict:
        series_fetcher = (edgar_data.fetch_instant_concept_series
                          if tags == edgar_data.INVENTORY_TAGS
                          else edgar_data.fetch_company_concept_series)
        by_tag = {}
        for tag in tags:
            if state["count"] > 0:
                sleep(min_interval)
            state["count"] += 1
            series = None
            try:
                series = series_fetcher(cik, tag, user_agent=user_agent, timeout=30.0)
            except ValueError:
                series = None  # one bad tag must not abort the others
            if series is not None:
                by_tag[tag] = series
        return by_tag
    return fetch_tags_fn


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--holdings", default=DEFAULT_HOLDINGS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    tickers = load_universe(args.holdings)
    print(f"[coverage] universe: {len(tickers)} real XRT holdings from {args.holdings}")

    cik_by_ticker = build_cik_map(tickers)
    n_resolved = sum(1 for v in cik_by_ticker.values() if v is not None)
    print(f"[coverage] CIK resolution: {n_resolved}/{len(tickers)} tickers resolved")

    fetch_tags_fn = make_live_fetch_tags_fn()
    t0 = time.time()
    result = ric.run_coverage_check(tickers, cik_by_ticker, fetch_tags_fn=fetch_tags_fn)
    elapsed = time.time() - t0
    print(f"[coverage] real EDGAR fetch complete in {elapsed:.0f}s")

    # Strip the (non-JSON-serializable, and not needed in the saved
    # report) DataFrame payloads before writing -- the coverage report is
    # a usability census, not a data cache.
    serializable_results = []
    for row in result["results"]:
        clean = {k: v for k, v in row.items()
                if k not in ("inventory_series", "revenue_series")}
        serializable_results.append(clean)

    universe_source_rel = os.path.relpath(args.holdings, REPO_ROOT).replace(os.sep, "/")
    payload = {
        "schema_version": 1,
        "candidate_family_id": "retail_inventory_sales_breadth_short",
        "run_date": date.today().isoformat(),
        "universe_source": universe_source_rel,
        "sec_contact_source": "env:" + sec_contact.ENV_VAR,
        "declared_n": result["declared_n"],
        "usable_n": result["usable_n"],
        "coverage_fraction": result["coverage_fraction"],
        "by_status": result["by_status"],
        "results": serializable_results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, args.output)

    print(json.dumps({
        "output": args.output.replace(os.sep, "/"),
        "declared_n": payload["declared_n"],
        "usable_n": payload["usable_n"],
        "coverage_fraction": round(payload["coverage_fraction"], 4),
        "by_status": payload["by_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
