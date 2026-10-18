"""Fetches real historical earnings-announcement dates (yfinance
Ticker.get_earnings_dates(), needs an interpreter with lxml installed --
lxml may be missing under a minimal interpreter) for every ticker in a
universe snapshot, needed ONLY by pead_earnings_surprise_drift (the
fifty_two_week_high_momentum family needs OHLC only, already fetched by
scripts/fetch_universe_ohlc.py).

Separate, slower, flakier fetch than plain OHLC -- so this script is deliberately independent and
resumable, caching each ticker's earnings-date list to its own JSON file
under data/mining/universe_cache/earnings_<date>/. Threaded (moderate
concurrency) for throughput; every ticker ends up cached-ok or logged as
failed with a reason -- never silently dropped.

Usage:
    python scripts/fetch_universe_earnings_dates.py \\
        --snapshot data/mining/universe_snapshot_<date>.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time

import pandas as pd
import yfinance as yf

MAX_WORKERS = 6
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3


def _fetch_one(ticker: str) -> dict:
    """Returns {'ticker', 'ok', 'dates': [...], 'reason'}. `dates` is
    every REPORTED (non-NaN Reported EPS) earnings date, ISO strings,
    ascending -- real, already-happened earnings only, never a
    future-scheduled estimate row."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=100)
            if df is None or df.empty:
                return {"ticker": ticker, "ok": True, "dates": [],
                        "reason": "get_earnings_dates returned empty/None "
                                  "(no earnings history on file)"}
            reported = df[df["Reported EPS"].notna()]
            dates = sorted(d.strftime("%Y-%m-%d") for d in reported.index)
            return {"ticker": ticker, "ok": True, "dates": dates, "reason": ""}
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
    return {"ticker": ticker, "ok": False, "dates": [],
            "reason": f"{type(last_err).__name__}: {last_err}"}


def fetch_and_cache(tickers: list, cache_dir: str) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    already = {f[:-5] for f in os.listdir(cache_dir) if f.endswith(".json")}
    todo = [t for t in tickers if t not in already]
    print(f"{len(already)} already cached, {len(todo)} to fetch this run")

    ok_count = 0
    failed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in todo}
        done_count = 0
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            t = result["ticker"]
            done_count += 1
            with open(os.path.join(cache_dir, f"{t}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f)
            if result["ok"]:
                ok_count += 1
            else:
                failed[t] = result["reason"]
            if done_count % 25 == 0 or done_count == len(todo):
                print(f"  {done_count}/{len(todo)} done ({ok_count} ok, "
                      f"{len(failed)} failed so far)")

    return {"ok_count": ok_count + len(already), "failed": failed}


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
        os.path.dirname(args.snapshot), "universe_cache", f"earnings_{fetch_date}")
    log_out = args.log_out or os.path.join(
        os.path.dirname(args.snapshot), "universe_cache", f"earnings_fetch_log_{fetch_date}.json")

    print(f"Universe: {len(tickers)} tickers, cache dir: {cache_dir}")
    result = fetch_and_cache(tickers, cache_dir)

    os.makedirs(os.path.dirname(log_out), exist_ok=True)
    with open(log_out, "w", encoding="utf-8") as f:
        json.dump({"universe_size": len(tickers),
                    "ok_count": result["ok_count"],
                    "failed_count": len(result["failed"]),
                    "failed": result["failed"]}, f, indent=2)

    print(f"DONE: {result['ok_count']}/{len(tickers)} ok, "
          f"{len(result['failed'])} failed. Log: {log_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
