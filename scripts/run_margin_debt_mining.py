"""Bespoke ABI-1 driver for margin_debt_growth_regime.

``scripts/run_batch5_mining.py``'s shared dispatch calls
``module.make_signal_fn(ticker, index, frames)`` with exactly three
positional args, never a keyword the FINRA-sourced ``margin_signal`` this
family's own ``make_signal_fn`` needs (the shared driver never fetches
non-yfinance external data). This driver fetches the FINRA workbook
explicitly (via ``mining.external_data.fetch_finra_margin_debt``, never
re-fetched per call) and threads it through the family's own
``margin_signal`` keyword.

Uses the shared ABI-1 v1 primitives
(``mining.abi1_driver.build_single_asset_curve`` / ``evaluate_family`` /
``record_abi1_outcome``), as the other bespoke drivers do
(``run_cboe_rotation_mining.py``, ``run_capex_cycle_mining.py``).

Only a ``construction="ABI-1"`` outcome trips this driver's own
``already_scored_abi1`` guard below: an outcome row from the per-trade-count
construction (``run_batch3_mining.py``) carries no ``construction`` field,
so it never blocks an ABI-1 run.

Provenance: the real FINRA fetch's ``ExternalSeriesSnapshot.metadata``
(source URL, retrieval timestamp, raw workbook sha256) is written to
``data/mining/margin_debt/provenance.json`` on every REAL (non-dry-run)
invocation -- the same "a fetched external artifact's provenance is
recorded, never silently absorbed" convention this pipeline already
applies to EDGAR and batch5-events fetches.
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
from mining import abi1_driver, catalog as catalog_mod, evidence, external_data, trial_ledger
from mining.batch5 import margin_debt_growth_regime as margin_mod

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
TAX_POLICY = "realization"  # single_long shape, matches abi1_driver.evaluate_family's
                            # own documented rule for SINGLE_SHAPES

DEFAULT_RUN_ID = "margin-debt"
DEFAULT_TESTED_DATE = batch5.DEFAULT_TESTED_DATE
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_ACCEPTED = batch5.DEFAULT_ACCEPTED
DEFAULT_QUOTES = batch5.DEFAULT_QUOTES
PROVENANCE_PATH = os.path.join(REPO_ROOT, "data", "mining", "margin_debt", "provenance.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def fetch_margin_debt_series():
    """Real (or injected-for-test) fetch of the FINRA workbook. Returns
    ``(series, metadata)`` -- never a bare series, so provenance is always
    available to the caller."""
    snapshot = external_data.fetch_finra_margin_debt()
    return snapshot.series, snapshot.metadata


def _load_accepted_entry(accepted_path: str, family_id: str) -> dict:
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload["accepted"] if c["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 accepted candidate {family_id!r}, found {len(matches)}")
    return matches[0]


def compute_and_record(*, spy_frame, monthly_debt, catalog_path: str,
                       accepted_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str) -> dict:
    """Preflight (accepted/catalog/drift/re-run guard, matching every
    sibling bespoke driver) then score+record ONE family:
    ``margin_debt_growth_regime``. Returns a single result/skip dict, not a
    list -- this driver only ever handles this one family."""
    family_id = margin_mod.FAMILY_ID
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
    drift = batch5.parameter_drift(margin_mod.PARAMETERS, entry.get("parameters"))
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
    costs = batch5.execution_costs_for([MARKET_BENCHMARK], quotes_payload)
    cost_bps = costs[MARKET_BENCHMARK]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    signal_fn = margin_mod.make_signal_fn(
        "SPY", spy_frame.index, {"SPY": spy_frame}, margin_signal=monthly_debt)
    curve = abi1_driver.build_single_asset_curve(spy_frame, signal_fn, cost_bps=cost_bps)
    account_type = entry["account_types"][0]
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)

    extra_note = (f"cost model: SPY={cost_bps:.2f}bps ({costs[MARKET_BENCHMARK]['source']}); "
                 f"signal_lag_trading_days={margin_mod.LAG_TRADING_DAYS} "
                 "(pre-registered catalog parameter, flagged in the catalog as not yet "
                 "independently confirmed -- measured real-world FINRA publication lag can run up to ~25+ "
                 "trading sessions, see the mining.batch5.margin_debt_growth_regime "
                 "module docstring); scored under ABI-1, the account-relative "
                 "construction")

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result,
            catalog_entry=entry, instrument_set=["SPY"],
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
            signal_source_path=signal_source_path,
            signal_source_sha256=signal_source_sha256)

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    return {
        "family_id": family_id, "shape": "single_long", "tickers": ["SPY"],
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

    cache_dir = os.path.join(args.research_root, args.run_id, "artifacts", "ohlc")
    spy_frame, _origin = batch5.load_frame(
        "SPY", ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    monthly_debt, metadata = fetch_margin_debt_series()

    if not args.dry_run:
        os.makedirs(os.path.dirname(PROVENANCE_PATH), exist_ok=True)
        with open(PROVENANCE_PATH, "w", encoding="utf-8") as handle:
            json.dump({"fetched_for_run_id": args.run_id, **metadata}, handle, indent=2)
            handle.write("\n")

    result = compute_and_record(
        spy_frame=spy_frame, monthly_debt=monthly_debt,
        catalog_path=args.catalog, accepted_path=args.accepted,
        quotes_path=args.quotes, ledger_path=args.ledger,
        run_id=args.run_id, tested_date=args.tested_date, dry_run=args.dry_run,
        signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__))

    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    # batch5._json_safe (not a bare `default=`) converts non-finite floats to
    # their repr() -- required because allow_nan=False already recognizes a
    # NaN/Infinity float (native or numpy.float64) as serializable and raises
    # before `default` would ever run. Every sibling bespoke driver
    # (run_cboe_rotation_mining.py, run_capex_cycle_mining.py) does the same.
    safe_result = batch5._json_safe(result)
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": _SIGNAL_SOURCE_PATH, "result": safe_result,
    }
    output_path = os.path.join(
        run_dir, "margin-debt-dry-run.json" if args.dry_run else "margin-debt-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
