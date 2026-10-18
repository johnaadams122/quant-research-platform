"""Bespoke ABI-1 driver for margin_debt_deleverage_synthetic_
leverage_flip -- the ABI-1 scoring pass (`run_margin_debt_deleverage_mining.
py` is the GROSS rough backtest and writes no ledger record).

This driver is the ABI-1 scoring stage: it calls `run_margin_debt_deleverage_mining.py`'s
own `rough_backtest()` (the same `abi1_driver.build_rotation_curve` call,
over the SAME full history) with a REAL cost sample, then calls
`abi1_driver.evaluate_family` + `record_abi1_outcome`, as the other
bespoke drivers do (`run_margin_debt_mining.py`, `run_money_market_mining.py`).

The driver scores the family only when it has a catalog entry (status
"eligible", parameters matching the module) and a candidate record in
`accepted.json`; otherwise it returns a structured skip.
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
import run_margin_debt_deleverage_mining as rough
from mining import abi1_driver, catalog as catalog_mod, evidence, trial_ledger
from mining.batch5 import margin_debt_deleverage_synthetic_leverage_flip as mod

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
ACCOUNT_TYPE = rough.ACCOUNT_TYPE  # "taxable"
TAX_POLICY = rough.TAX_POLICY      # "realization"

DEFAULT_RUN_ID = "margin-debt-deleverage-abi1"
DEFAULT_TESTED_DATE = batch5.DEFAULT_TESTED_DATE
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_ACCEPTED = batch5.DEFAULT_ACCEPTED
DEFAULT_QUOTES = batch5.DEFAULT_QUOTES

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def _load_accepted_entry(accepted_path: str, family_id: str) -> dict:
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload["accepted"] if c["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 accepted candidate {family_id!r}, found {len(matches)}")
    return matches[0]


def compute_and_record(*, catalog_path: str, accepted_path: str,
                       quotes_path: str, ledger_path: str, run_id: str,
                       tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str
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
    costs = batch5.execution_costs_for([rough.UNDERLYING, rough.DEFENSIVE], quotes_payload)
    cost_bps_by_asset = {sym: costs[sym]["cost_bps"] for sym in (rough.UNDERLYING, rough.DEFENSIVE)}
    benchmark_entry_cost_bps = costs[rough.UNDERLYING]["spread_bps"] / 2.0

    spy_frame = rough._fetch_spy_ohlc()
    sh_frame = rough._fetch_sh_ohlc()
    irx_series = rough._fetch_irx()
    margin_debt, finra_metadata = rough.fetch_margin_debt_series()

    if not dry_run:
        os.makedirs(os.path.dirname(rough.PROVENANCE_PATH), exist_ok=True)
        with open(rough.PROVENANCE_PATH, "w", encoding="utf-8") as handle:
            json.dump({"fetched_for_run_id": run_id, **finra_metadata}, handle, indent=2)
            handle.write("\n")

    # holdout_months left at rough_backtest's own default: it only feeds
    # built["stats"] (in-sample descriptive stats this driver never reads),
    # never the real ABI-1 holdout window -- that is fixed inside
    # account_relative.evaluate at constants.HOLDOUT_WINDOW_MONTHS and is
    # not a parameter this driver can or should override (so this driver
    # offers no --holdout-months option).
    built = rough.rough_backtest(
        spy_frame=spy_frame, sh_frame=sh_frame, margin_debt=margin_debt,
        irx_series=irx_series, cost_bps_by_asset=cost_bps_by_asset)
    curve = built["curve"]

    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY)

    extra_note = (
        f"cost model: SPY={cost_bps_by_asset[rough.UNDERLYING]:.2f}bps, "
        f"SH={cost_bps_by_asset[rough.DEFENSIVE]:.2f}bps "
        f"({batch5.quotes_cost_source(quotes_payload)}); "
        f"lag_trading_days={mod.LAG_TRADING_DAYS} (reused verbatim from "
        "margin_debt_growth_regime's own pre-registered constant, "
        "a catalog flag marking this parameter as not yet independently "
        "confirmed carries over unchanged)")

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
    return {
        "family_id": family_id, "shape": mod.SHAPE, "tickers": list(mod.TRADED),
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
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
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--quotes", default=DEFAULT_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)

    result = compute_and_record(
        catalog_path=args.catalog, accepted_path=args.accepted,
        quotes_path=args.quotes, ledger_path=args.ledger,
        run_id=args.run_id, tested_date=args.tested_date, dry_run=args.dry_run,
        signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__))

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
        "margin-debt-deleverage-dry-run.json" if args.dry_run
        else "margin-debt-deleverage-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
