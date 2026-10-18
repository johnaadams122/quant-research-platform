"""Rough-backtest driver for retail_inventory_sales_breadth_short
(NOT an ABI-1 scoring run -- see the hard stop at the bottom of ``main``).

Reuses the coverage-checked universe
(``data/mining/retail_inventory_coverage.json``, written by
``run_retail_inventory_coverage_check.py``: the XRT holdings with a usable
point-in-time inventory AND revenue series),
re-fetches those companies' real point-in-time series live from SEC
EDGAR (the coverage check itself did not persist the series, only the
usability verdict), loads XRT's + SPY's real OHLC, builds the breadth
signal via ``mining.batch5.retail_inventory_sales_breadth_short``, and
runs the REAL generic backtest engine (``mining.backtest_engine.
run_backtest``, direction="short") on in-sample data (everything before
the standard ``mining.constants.HOLDOUT_WINDOW_MONTHS``-month holdout).

COST DISCLOSURE (never invented): a bid/ask quote taken outside NYSE
regular trading hours is unusable as a cost basis for a thinner-volume
fund like XRT (its extended-hours spread widens far beyond its regular-
session spread while SPY/QQQ stay tight), the same pattern other thinner
funds show. Under the pipeline's
rule "never invent a cost number", this rough backtest runs GROSS
(``cost_bps=0.0``), explicitly and loudly disclosed as such in every
output this script produces -- it is NOT a claim that this candidate's
real execution cost is zero.

Usage (real network calls -- SEC + yfinance; takes a few minutes):
    python scripts/run_retail_inventory_breadth_mining.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd  # noqa: E402

import run_batch5_mining as batch5  # noqa: E402
import run_retail_inventory_coverage_check as coverage_script  # noqa: E402
from mining import constants, edgar_cik_resolution, edgar_data  # noqa: E402
from mining import retail_inventory_coverage as ric  # noqa: E402
from mining.backtest_engine import run_backtest  # noqa: E402
from mining.batch5 import retail_inventory_sales_breadth_short as mod  # noqa: E402

DEFAULT_COVERAGE_PATH = coverage_script.DEFAULT_OUTPUT_PATH
RUN_DIR = os.path.join(REPO_ROOT, "runs", "run_retail_inventory_breadth_mining")
DEFAULT_OUTPUT_PATH = os.path.join(RUN_DIR, "rough-backtest-report.json")

COST_DISCLOSURE = (
    "cost_bps=0.0 (GROSS backtest, NOT a claim of zero real cost) -- no "
    "regular-session XRT spread sample is used by this rough pass, and an "
    "after-close/extended-hours sample is unusable as a cost basis "
    "(its spread widens far beyond the regular session's). No cost number is "
    "invented instead.")


def load_ok_tickers(coverage_path: str) -> list:
    with open(coverage_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return sorted(r["ticker"] for r in payload["results"] if r["status"] == "ok")


def build_fundamentals_map(tickers: list, *, sleep=time.sleep) -> dict:
    """Re-fetch the real point-in-time inventory+revenue series for each
    already-coverage-checked ticker, in the shape
    ``retail_inventory_sales_breadth_short.breadth_series`` consumes.
    Every ticker here already cleared the coverage check, so a
    fetch failure here would be a genuine new anomaly (e.g. a transient
    network flake) -- logged and excluded from this run rather than aborting it, same
    "drop the member, log why" convention as every other pooled
    precondition in this pipeline."""
    payload = coverage_script.fetch_company_tickers()
    table = edgar_cik_resolution.parse_company_tickers_payload(payload)
    cik_by_ticker = edgar_cik_resolution.resolve_cik_for_tickers(table, tickers)

    fetch_tags_fn = coverage_script.make_live_fetch_tags_fn(sleep=sleep)
    fundamentals_map = {}
    excluded = {}
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        result = ric.check_ticker_coverage(ticker, cik, fetch_tags_fn=fetch_tags_fn)
        if result["status"] != "ok":
            excluded[ticker] = result["status"]
            continue
        fundamentals_map[ticker] = {
            "inventory": result["inventory_series"],
            "revenue": result["revenue_series"],
        }
    return fundamentals_map, excluded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coverage", default=DEFAULT_COVERAGE_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    tickers = load_ok_tickers(args.coverage)
    print(f"[breadth] re-fetching real point-in-time series for {len(tickers)} "
         f"coverage-checked tickers (coverage check result: {args.coverage})")
    t0 = time.time()
    fundamentals_map, excluded = build_fundamentals_map(tickers)
    print(f"[breadth] re-fetch complete in {time.time() - t0:.0f}s: "
         f"{len(fundamentals_map)} usable, {len(excluded)} excluded this run "
         f"({excluded if excluded else 'none'})")

    cache_dir = os.path.join(RUN_DIR, "artifacts", "ohlc")
    xrt_frame, xrt_source = batch5.load_frame(
        "XRT", ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    print(f"[breadth] XRT OHLC: {len(xrt_frame)} bars "
         f"({xrt_frame.index[0].date()}..{xrt_frame.index[-1].date()}), {xrt_source}")

    holdout_months = constants.HOLDOUT_WINDOW_MONTHS
    holdout_start = xrt_frame.index[-1] - pd.DateOffset(months=holdout_months)
    in_sample = xrt_frame.loc[xrt_frame.index < holdout_start]
    print(f"[breadth] in-sample window: {in_sample.index[0].date()}.."
         f"{in_sample.index[-1].date()} ({len(in_sample)} bars; "
         f"holding out the trailing {holdout_months} months, "
         f"{holdout_start.date()}..{xrt_frame.index[-1].date()}, "
         f"{len(xrt_frame) - len(in_sample)} bars, NOT read by this run)")

    signal_fn = mod.make_signal_fn(
        "XRT", in_sample.index, {"XRT": in_sample}, fundamentals_map=fundamentals_map)

    print(f"[breadth] {COST_DISCLOSURE}")
    trades_df, in_market = run_backtest(
        in_sample, signal_fn, cost_bps=0.0, direction="short")

    n_trades = len(trades_df)
    summary = {
        "n_trades": n_trades,
        "time_in_market_fraction": float(in_market.mean()) if len(in_market) else 0.0,
        "mean_gross_ret": float(trades_df["gross_ret"].mean()) if n_trades else None,
        "median_gross_ret": float(trades_df["gross_ret"].median()) if n_trades else None,
        "win_rate": float((trades_df["gross_ret"] > 0).mean()) if n_trades else None,
        "mean_held_sessions": float(trades_df["held"].mean()) if n_trades else None,
        "reason_counts": (trades_df["reason"].value_counts().to_dict()
                          if n_trades else {}),
    }
    print(f"[breadth] REAL rough-backtest result: {json.dumps(summary, indent=2)}")

    coverage_source_rel = os.path.relpath(args.coverage, REPO_ROOT).replace(os.sep, "/")
    payload = {
        "schema_version": 1,
        "candidate_family_id": mod.FAMILY_ID,
        "construction": "rough_backtest_step3_only",
        "run_date": time.strftime("%Y-%m-%d"),
        "coverage_source": coverage_source_rel,
        "universe_n_ok": len(tickers),
        "fundamentals_refetch_excluded": excluded,
        "xrt_ohlc_bars_total": len(xrt_frame),
        "in_sample_bars": len(in_sample),
        "holdout_window_months": holdout_months,
        "holdout_start": str(holdout_start.date()),
        "cost_disclosure": COST_DISCLOSURE,
        "parameters": mod.PARAMETERS,
        "summary": summary,
        "trades": json.loads(trades_df.to_json(orient="records", date_format="iso"))
                  if n_trades else [],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(tmp, args.output)
    print(f"[breadth] report written: {args.output}")

    print("[breadth] STOPPING BEFORE ABI-1 SCORING (and its trial_ledger.jsonl "
         "write) by design -- this run touched no ledger and no catalog "
         "status.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
