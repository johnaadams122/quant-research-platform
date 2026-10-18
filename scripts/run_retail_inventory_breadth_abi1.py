"""Bespoke ABI-1 driver for retail_inventory_sales_breadth_short
(XRT short, driven by an SEC-EDGAR-XBRL inventory/sales breadth panel) --
the ABI-1 scoring pass. The rough backtest for this family is
``run_retail_inventory_breadth_mining.py``, which writes no ledger record.

The universe is a current-holdings proxy with a disclosed survivorship-bias
caveat (the proxy is CONSERVATIVE, not inflationary).
``data/mining/catalog.json``'s own ``universe_source`` field and this
family's own module ``PARAMETERS``
(``src/mining/batch5/retail_inventory_sales_breadth_short.py``) carry the
identical text (``batch5.parameter_drift`` checks it).

Imports ``run_retail_inventory_breadth_mining.py`` (as ``rough``) for two
pieces: ``load_ok_tickers`` (the tickers the coverage check marked usable)
and ``build_fundamentals_map`` (re-fetches each ticker's real point-in-time
inventory + revenue series live from SEC EDGAR, because the coverage check
stores only the usability verdict, not the series).

DIRECTION: this family shorts XRT DIRECTLY (not via holding a real
inverse ETF long). The driver calls ``backtest_engine.run_backtest`` and
``benchmark_relative.mark_to_market_daily_returns`` with
``direction="short"`` and passes their output to
``abi1_driver.evaluate_family`` as a ``{"strategy_daily", "in_market",
"trades_df"}`` dict, the curve shape ``evaluate_family`` accepts from any
caller; ``evaluate_family`` (through ``account_relative.evaluate``) turns
it into the account curve it scores against SPY.

COST -- never invented. A quote taken just after the 16:00 ET close can
show a post-close XRT-specific spread-widening artifact (XRT far wider
than during the session while SPY and QQQ stay tight at the same moment),
so this driver must be given a REGULAR-SESSION quote file (``--quotes``,
required) that includes XRT and SPY. Checking that XRT's spread is tight
against SPY's simultaneous spread is the operator's job before the run;
this driver applies only the shared quote checks
(``run_batch5_mining.refuse_synthetic_quotes`` and ``execution_costs_for``).

Usage (real network calls -- SEC EDGAR + yfinance; the fundamentals
re-fetch alone takes a few minutes, matching the rough-backtest
driver's own timing):
    python scripts/run_retail_inventory_breadth_abi1.py --quotes QUOTES.json --tested-date YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch3_mining as batch3
import run_batch5_mining as batch5
import run_retail_inventory_breadth_mining as rough
from mining import abi1_driver, catalog as catalog_mod, evidence, trial_ledger
from mining.backtest_engine import run_backtest
from mining.batch5 import retail_inventory_sales_breadth_short as mod
from mining.benchmark_relative import mark_to_market_daily_returns

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
TAX_POLICY = "realization"  # panel_breadth is a SINGLE_SHAPE (abi1_driver's own
                            # module docstring): 100% of capital in one position
                            # at a time, so netted_tax_factors' dollar-gain
                            # identity holds -- matches every sibling single-
                            # instrument bespoke driver's own convention.

DEFAULT_RUN_ID = "retail-inventory-breadth-abi1-v1"
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_ACCEPTED = batch5.DEFAULT_ACCEPTED
DEFAULT_COVERAGE = rough.DEFAULT_COVERAGE_PATH

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def _load_accepted_entry(accepted_path: str, family_id: str) -> dict:
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload["accepted"] if c["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 accepted candidate {family_id!r} in "
            f"{accepted_path}, found {len(matches)}")
    return matches[0]


def compute_and_record(*, catalog_path: str, accepted_path: str,
                       quotes_path: str, coverage_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT
                       ) -> dict:
    family_id = mod.FAMILY_ID
    try:
        _load_accepted_entry(accepted_path, family_id)
    except ValueError as exc:
        return {"family_id": family_id, "reason_code": "not_in_accepted", "reason": str(exc)}

    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    entry = catalog_entries.get(family_id)
    if entry is None:
        return {"family_id": family_id, "reason_code": "not_in_catalog",
                "reason": f"{family_id} is not in the catalog"}
    if entry.get("status") != "eligible":
        return {"family_id": family_id, "reason_code": "catalog_status",
                "reason": f"catalog status is {entry.get('status')!r}, not eligible"}
    drift = batch5.parameter_drift(mod.PARAMETERS, entry.get("parameters"))
    if drift is not None:
        return {"family_id": family_id, "reason_code": "parameter_drift", "reason": drift}

    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"}
    if family_id in already_scored:
        return {"family_id": family_id, "reason_code": "already_scored_abi1",
                "reason": "a prior run already recorded a construction=\"ABI-1\" outcome"}

    quotes_payload = batch5.load_quotes(quotes_path)
    batch5.refuse_synthetic_quotes(quotes_payload, dry_run=dry_run)
    traded_ticker = mod.TRADED[0]
    costs = batch5.execution_costs_for([traded_ticker, MARKET_BENCHMARK], quotes_payload)
    cost_bps = costs[traded_ticker]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    xrt_frame, xrt_source = batch5.load_frame(
        traded_ticker, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    spy_frame, _spy_source = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)

    tickers = rough.load_ok_tickers(coverage_path)
    fundamentals_map, excluded = rough.build_fundamentals_map(tickers)

    signal_fn = mod.make_signal_fn(
        traded_ticker, xrt_frame.index, {traded_ticker: xrt_frame},
        fundamentals_map=fundamentals_map)

    # Short direction -- see module docstring: run_backtest and
    # mark_to_market_daily_returns are called with direction="short", and
    # their output forms the {"strategy_daily", "in_market", "trades_df"}
    # curve dict evaluate_family takes.
    trades_df, in_market = run_backtest(
        xrt_frame, signal_fn, cost_bps=cost_bps, tax_haircut_rate=0.0,
        direction="short")
    strategy_daily = mark_to_market_daily_returns(
        xrt_frame, trades_df, cost_bps=cost_bps, direction="short")
    curve = {"strategy_daily": strategy_daily, "in_market": in_market,
             "trades_df": trades_df}

    account_type = entry["account_types"][0]
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)

    extra_note = (
        f"cost model: {traded_ticker}={cost_bps:.2f}bps "
        f"({batch5.quotes_cost_source(quotes_payload)}; regular-session "
        "capture -- a post-close quote shows an XRT-specific spread-widening "
        "artifact, so none is used and no cost number is invented); direction=short "
        "(run_backtest/mark_to_market_daily_returns called with "
        "direction=\"short\"; see module docstring); "
        f"fundamentals_refetch_excluded={excluded!r}")

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result,
            catalog_entry=entry, instrument_set=list(mod.TRADED),
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
            signal_source_path=signal_source_path,
            signal_source_sha256=signal_source_sha256)

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    account_index = result["account_daily"].index
    return {
        "family_id": family_id, "shape": mod.SHAPE, "tickers": list(mod.TRADED),
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "n_trades": int(len(trades_df)),
        "account_start": account_index[0].strftime("%Y-%m-%d") if len(account_index) else None,
        "account_end": account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None,
        "time_in_market": result.get("time_in_market"),
        "tax_policy": TAX_POLICY, "cash_convention": CASH_CONVENTION,
        "benchmark": MARKET_BENCHMARK, "cost_model_note": extra_note,
        "holdout_evaluated": holdout.get("evaluated"),
        "holdout_passed": holdout.get("passed"),
        "holdout_psr": holdout.get("psr"),
        "window": {
            "account_sessions": window.get("account_sessions"),
            "scored_sessions": window.get("scored_sessions"),
            "in_sample_sessions": window.get("in_sample_sessions"),
            "holdout_sessions": window.get("holdout_sessions"),
        },
        "fundamentals_refetch_excluded": excluded,
        "xrt_ohlc_source": xrt_source,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--quotes", required=True,
                        help="regular-session quote file covering XRT and SPY")
    parser.add_argument("--coverage", default=DEFAULT_COVERAGE)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", required=True,
                        help="ISO date recorded on the ledger outcome (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)

    result = compute_and_record(
        catalog_path=args.catalog, accepted_path=args.accepted,
        quotes_path=args.quotes, coverage_path=args.coverage,
        ledger_path=args.ledger, run_id=args.run_id, tested_date=args.tested_date,
        dry_run=args.dry_run, signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__),
        research_root=args.research_root)

    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    safe_result = batch5._json_safe(result)
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": _SIGNAL_SOURCE_PATH, "result": safe_result,
    }
    output_path = os.path.join(
        run_dir,
        "retail-inventory-breadth-dry-run.json" if args.dry_run
        else "retail-inventory-breadth-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
