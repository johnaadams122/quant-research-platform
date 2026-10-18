"""Bulk-fetches daily OHLC history for every ticker in a universe snapshot
(scripts/fetch_index_constituents.py output) and caches each ticker's
frame to its own CSV under data/mining/universe_cache/ohlc_<date>/ --
resumable (skips tickers already cached) and chunked (yf.download's
multi-ticker batch mode, roughly 10-50x faster than one ticker at a time
via yf.Ticker(...).history()).

Never silently drops a ticker: every ticker ends up in exactly one of
cached-ok / failed (with a reason), and the failure log is written
alongside the cache so a downstream consumer can report real coverage
honestly (a "no silent caps" rule).

Usage:
    python scripts/fetch_universe_ohlc.py \\
        --snapshot data/mining/universe_snapshot_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
import yfinance as yf

CHUNK_SIZE = 40
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch_and_cache(tickers: list, cache_dir: str) -> dict:
    """Returns {'cached': [...], 'failed': {ticker: reason}}. Skips any
    ticker whose CSV already exists in cache_dir (resumable)."""
    os.makedirs(cache_dir, exist_ok=True)
    already = {f[:-4] for f in os.listdir(cache_dir) if f.endswith(".csv")}
    todo = [t for t in tickers if t not in already]
    cached = list(already & set(tickers))
    failed = {}

    print(f"{len(already)} already cached, {len(todo)} to fetch this run")

    for batch_num, batch in enumerate(_chunks(todo, CHUNK_SIZE), start=1):
        attempt = 0
        remaining = list(batch)
        while remaining and attempt < MAX_RETRIES:
            attempt += 1
            try:
                data = yf.download(tickers=remaining, period="max",
                                    auto_adjust=True, group_by="ticker",
                                    threads=True, progress=False)
            except Exception as e:
                print(f"  batch {batch_num} attempt {attempt}: download() "
                      f"raised {e!r}, retrying remaining {len(remaining)}")
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            got_this_attempt = []
            if isinstance(data.columns, pd.MultiIndex):
                available = set(data.columns.get_level_values(0))
            else:
                # yf.download collapses to a flat frame when only ONE
                # ticker in the batch actually returned data.
                available = set(remaining) if not data.empty else set()

            for t in list(remaining):
                if t not in available:
                    continue
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        df = data[t][["Open", "High", "Low", "Close"]].dropna(how="all")
                    else:
                        df = data[["Open", "High", "Low", "Close"]].dropna(how="all")
                except KeyError:
                    continue
                if df.empty:
                    continue
                df.index.name = "Date"
                df.to_csv(os.path.join(cache_dir, f"{t}.csv"))
                cached.append(t)
                got_this_attempt.append(t)

            remaining = [t for t in remaining if t not in got_this_attempt]
            if remaining and attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

        for t in remaining:
            failed[t] = f"no data returned after {attempt} attempt(s)"

        print(f"  batch {batch_num}: {len(batch) - len(remaining)}/{len(batch)} ok "
              f"({len(cached)} cached total, {len(failed)} failed total)")

    return {"cached": sorted(set(cached)), "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--log-out", default=None)
    args = parser.parse_args()

    with open(args.snapshot, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    tickers = snapshot["tickers"]

    fetch_date = snapshot.get("fetch_date", "unknown")
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(args.snapshot), "universe_cache", f"ohlc_{fetch_date}")
    log_out = args.log_out or os.path.join(
        os.path.dirname(args.snapshot), "universe_cache", f"ohlc_fetch_log_{fetch_date}.json")

    print(f"Universe: {len(tickers)} tickers, cache dir: {cache_dir}")
    result = fetch_and_cache(tickers, cache_dir)

    os.makedirs(os.path.dirname(log_out), exist_ok=True)
    with open(log_out, "w", encoding="utf-8") as f:
        json.dump({"universe_size": len(tickers),
                    "cached_count": len(result["cached"]),
                    "failed_count": len(result["failed"]),
                    "failed": result["failed"]}, f, indent=2)

    print(f"DONE: {len(result['cached'])}/{len(tickers)} cached, "
          f"{len(result['failed'])} failed. Log: {log_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
