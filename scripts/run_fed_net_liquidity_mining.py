"""Bespoke ABI-1 driver for fed_net_liquidity_regime.

``scripts/run_batch5_mining.py``'s shared dispatch calls
``module.make_signal_fn(ticker, index, frames)`` with exactly three
positional args, never the FRED-sourced ``walcl``/``wtregen``/``rrpontsyd``
keywords this family's own ``make_signal_fn`` needs (the shared driver
never fetches non-yfinance external data). This driver fetches all three
real, point-in-time FRED series explicitly (via
``mining.external_data.fetch_fred_series_point_in_time``, never re-fetched
per call) and threads them through the family's own keywords.

Uses the shared ABI-1 v1 primitives
(``mining.abi1_driver.build_single_asset_curve`` / ``evaluate_family`` /
``record_abi1_outcome``), as the other bespoke drivers do
(``run_margin_debt_mining.py``, ``run_money_market_mining.py``).

The public FRED graph CSV carries no point-in-time/vintage evidence, so
this driver uses the authenticated point-in-time fetch instead.

Provenance: the real FRED fetch's provenance (source URLs, series ids) is
written to ``data/mining/fed_net_liquidity/provenance.json`` on every REAL
(non-dry-run) invocation -- the same convention this pipeline already
applies to FINRA, EDGAR, and money-market fetches.
"""
from __future__ import annotations

import argparse
import datetime
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
from mining.batch5 import fed_net_liquidity_regime as fnl_mod

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
TAX_POLICY = "realization"  # single_long shape, matches abi1_driver.evaluate_family's
                            # own documented rule for SINGLE_SHAPES

DEFAULT_RUN_ID = "fed-net-liquidity"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()  # today, at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
# Default inputs live in this driver's own run folder; pass --accepted /
# --quotes to point at a curated candidate file and a real quote capture.
DEFAULT_ACCEPTED = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "accepted.json")
DEFAULT_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes.json")
PROVENANCE_PATH = os.path.join(
    REPO_ROOT, "data", "mining", "fed_net_liquidity", "provenance.json")
FRED_SERIES_IDS = ("WALCL", "WTREGEN", "RRPONTSYD")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def fetch_fed_liquidity_series():
    """Real (or injected-for-test) point-in-time fetch of all three FRED
    inputs. Returns ``(walcl, wtregen, rrpontsyd)``."""
    return tuple(
        external_data.fetch_fred_series_point_in_time(series_id)
        for series_id in FRED_SERIES_IDS)


def _load_accepted_entry(accepted_path: str, family_id: str) -> dict:
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload["accepted"] if c["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 accepted candidate {family_id!r}, found {len(matches)}")
    return matches[0]


def compute_and_record(*, spy_frame, walcl, wtregen, rrpontsyd, catalog_path: str,
                       accepted_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str) -> dict:
    """Preflight (accepted/catalog/drift/re-run guard, matching every
    sibling bespoke driver) then score+record ONE family:
    ``fed_net_liquidity_regime``. Returns a single result/skip dict, not a
    list -- this driver only ever handles this one family, on SPY only
    (the ``mining.batch5.fed_net_liquidity_regime`` module docstring
    explains why QQQ, also in the catalog's tickers list, is not built
    here)."""
    family_id = fnl_mod.FAMILY_ID
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
    drift = batch5.parameter_drift(fnl_mod.PARAMETERS, entry.get("parameters"))
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

    signal_fn = fnl_mod.make_signal_fn(
        "SPY", spy_frame.index, {"SPY": spy_frame},
        walcl=walcl, wtregen=wtregen, rrpontsyd=rrpontsyd)
    curve = abi1_driver.build_single_asset_curve(spy_frame, signal_fn, cost_bps=cost_bps)
    account_type = entry["account_types"][0]
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)

    extra_note = (f"cost model: SPY={cost_bps:.2f}bps ({costs[MARKET_BENCHMARK]['source']}); "
                 f"lag_trading_days={fnl_mod.LAG_TRADING_DAYS} "
                 "(MEASURED via FRED's own ALFRED vintage history -- WALCL's "
                 "own 1-trading-day H.4.1 publication lag is the binding constraint, "
                 "WTREGEN/RRPONTSYD verified faster; see mining.batch5."
                 "fed_net_liquidity_regime module docstring; not a catalog-pre-registered "
                 "parameter); SPY ONLY -- catalog also lists QQQ but the mechanism text "
                 "never exercises it, see the same module docstring")

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
    walcl, wtregen, rrpontsyd = fetch_fed_liquidity_series()

    if not args.dry_run:
        os.makedirs(os.path.dirname(PROVENANCE_PATH), exist_ok=True)
        with open(PROVENANCE_PATH, "w", encoding="utf-8") as handle:
            json.dump({
                "fetched_for_run_id": args.run_id,
                "source_url": external_data.FRED_API_OBSERVATIONS_URL,
                "point_in_time": True,
                "series": {
                    series_id: {
                        "retrieved_at": series.attrs.get("retrieved_at"),
                        "raw_sha256": series.attrs.get("raw_sha256"),
                    }
                    for series_id, series in zip(
                        FRED_SERIES_IDS, (walcl, wtregen, rrpontsyd))
                },
            }, handle, indent=2)
            handle.write("\n")

    result = compute_and_record(
        spy_frame=spy_frame, walcl=walcl, wtregen=wtregen, rrpontsyd=rrpontsyd,
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
        "fed-net-liquidity-dry-run.json" if args.dry_run else "fed-net-liquidity-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
