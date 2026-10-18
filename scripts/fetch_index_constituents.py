"""Fetches the REAL, CURRENT S&P 500 and NASDAQ-100 constituent lists from
Wikipedia (the standard free source for this) and writes a de-duplicated
universe snapshot to data/mining/universe_snapshot_<date>.json.

The snapshot is the universe scripts/run_expanded_universe_mining.py pools
over (every ticker in the S&P 500 and NASDAQ 100) for price/earnings-only
catalog candidates such as fifty_two_week_high_momentum and
pead_earnings_surprise_drift. Pure normalization/dedup logic lives in
mining.universe (unit-tested); this script is the live-fetch orchestration
layer only.

Wikipedia blocks pd.read_html's default urllib request (HTTP 403, no
User-Agent) -- fetched via requests with an explicit User-Agent instead,
then parsed with pd.read_html(StringIO(html)).

Usage:
    python scripts/fetch_index_constituents.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from io import StringIO

import pandas as pd
import requests

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import universe  # noqa: E402

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "quant-research-platform-script/1.0"}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# The plain "Nasdaq-100" article no longer carries the constituent table
# (it only has History/Selection
# criteria/Performance/Record-values sections); the constituent table
# lives on the separate "List of NASDAQ-100 companies" article (found via
# the Wikipedia search API), Symbol/Ticker column, 102 rows (includes
# dual-class share tickers like GOOGL/GOOG, so > 100).
NASDAQ100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


def _fetch_table(url: str, ticker_column_candidates=("Symbol", "Ticker")) -> list:
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    for t in tables:
        for col in ticker_column_candidates:
            if col in t.columns and len(t) > 50:
                return t[col].astype(str).tolist()
    raise ValueError(f"fetch_index_constituents: no table at {url} matched "
                      f"columns {ticker_column_candidates} with >50 rows -- "
                      f"page structure may have changed, found tables: "
                      f"{[(t.shape, list(t.columns)[:4]) for t in tables]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "mining"))
    parser.add_argument("--date", default=None,
                         help="Fetch-date stamp for the output filename/"
                              "metadata (default: today, UTC date).")
    args = parser.parse_args()

    fetch_date = args.date or datetime.date.today().isoformat()

    print(f"Fetching S&P 500 constituents from {SP500_URL} ...")
    sp500_raw = _fetch_table(SP500_URL, ("Symbol",))
    print(f"  {len(sp500_raw)} raw rows")

    print(f"Fetching NASDAQ-100 constituents from {NASDAQ100_URL} ...")
    nasdaq100_raw = _fetch_table(NASDAQ100_URL, ("Ticker", "Symbol"))
    print(f"  {len(nasdaq100_raw)} raw rows")

    result = universe.build_universe(sp500_raw, nasdaq100_raw)
    result["fetch_date"] = fetch_date
    result["sp500_source_url"] = SP500_URL
    result["nasdaq100_source_url"] = NASDAQ100_URL

    out_path = os.path.join(args.out_dir, f"universe_snapshot_{fetch_date}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"S&P 500: {result['sp500_count']} raw, NASDAQ-100: "
          f"{result['nasdaq100_count']} raw, overlap: {result['overlap_count']}, "
          f"UNIQUE UNIVERSE: {result['unique_count']}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
