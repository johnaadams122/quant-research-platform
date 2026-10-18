"""Bespoke ABI-1 driver for fed_net_liquidity_regime_qqq.

Identical shape to ``run_fed_net_liquidity_mining.py`` -- see that
script's own docstring for the shared-dispatch-can't-supply-external-data
rationale. This driver differs only in which module/family/traded ticker
it scores: the QQQ-traded sibling (see that module's own docstring for
why it is a separate family_id rather than pooling with the SPY family).

Cost basis: the traded QQQ leg and the SPY benchmark are priced from two
separate quote files (``--qqq-quotes`` and ``--spy-quotes``), each a real
bid/ask capture; the capture time is disclosed in the recorded cost-model
note, the same "never invent a cost" discipline as every other real run in
this pipeline.

Provenance: written to
``data/mining/fed_net_liquidity_qqq/provenance.json`` on every REAL
(non-dry-run) invocation, same convention as every sibling driver.
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
from mining.batch5 import fed_net_liquidity_regime_qqq as fnl_mod

CASH_CONVENTION = "sweep_cash_daily"
TRADED_TICKER = "QQQ"
# The ABI-1 benchmark is SPY unconditionally, system-wide -- NOT whatever
# is traded. abi1_driver.evaluate_family's own docstring: "passing the
# full SPY history even for a family trading back to the 1960s is
# correct, not a bug to work around." Every candidate's DSR is "excess
# over SPY"; benchmarking this one against QQQ instead would make its
# DSR incomparable to the rest of the pipeline's calibration
# (DAILY_SR_VARIANCE/ABI1_N_TRIALS), defeating the whole point of a
# shared multiple-testing correction.
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
TAX_POLICY = "realization"

DEFAULT_RUN_ID = "fed-net-liquidity-qqq"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()  # today, at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
# Default inputs live in this driver's own run folder; pass the CLI flags
# to point at a curated candidate file and real quote captures.
DEFAULT_ACCEPTED = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "accepted.json")
# QQQ's own traded-instrument cost.
DEFAULT_QQQ_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes-qqq.json")
# SPY's benchmark cost.
DEFAULT_SPY_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes-spy.json")
PROVENANCE_PATH = os.path.join(
    REPO_ROOT, "data", "mining", "fed_net_liquidity_qqq", "provenance.json")
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


def compute_and_record(*, qqq_frame, spy_frame, walcl, wtregen, rrpontsyd, catalog_path: str,
                       accepted_path: str, qqq_quotes_path: str, spy_quotes_path: str,
                       ledger_path: str, run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str) -> dict:
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

    # QQQ's own traded-instrument cost, and SPY's own SEPARATE benchmark
    # cost -- two different quote files, since QQQ has no capture in the
    # shared batch5 file and SPY's benchmark cost must stay the same real
    # sample every other family uses.
    qqq_quotes_payload = batch5.load_quotes(qqq_quotes_path)
    batch5.refuse_synthetic_quotes(qqq_quotes_payload, dry_run=dry_run)
    qqq_costs = batch5.execution_costs_for([TRADED_TICKER], qqq_quotes_payload)
    cost_bps = qqq_costs[TRADED_TICKER]["cost_bps"]

    spy_quotes_payload = batch5.load_quotes(spy_quotes_path)
    batch5.refuse_synthetic_quotes(spy_quotes_payload, dry_run=dry_run)
    spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK], spy_quotes_payload)
    benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    signal_fn = fnl_mod.make_signal_fn(
        "QQQ", qqq_frame.index, {"QQQ": qqq_frame},
        walcl=walcl, wtregen=wtregen, rrpontsyd=rrpontsyd)
    curve = abi1_driver.build_single_asset_curve(qqq_frame, signal_fn, cost_bps=cost_bps)
    account_type = entry["account_types"][0]
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)

    extra_note = (f"cost model: QQQ (traded)={cost_bps:.3f}bps "
                 f"({qqq_costs[TRADED_TICKER]['source']}, real bid/ask sample from the "
                 f"--qqq-quotes capture); SPY (benchmark)={spy_costs[MARKET_BENCHMARK]['cost_bps']:.3f}bps "
                 f"({spy_costs[MARKET_BENCHMARK]['source']}) -- ABI-1's benchmark is SPY "
                 "unconditionally per abi1_driver.evaluate_family's own documented contract, "
                 "not whatever is traded, so every family's DSR stays comparable under the "
                 "shared DAILY_SR_VARIANCE/ABI1_N_TRIALS calibration; "
                 f"lag_trading_days={fnl_mod.LAG_TRADING_DAYS} (identical to the SPY "
                 "sibling's own measured WALCL publication lag); QQQ VARIANT of "
                 "fed_net_liquidity_regime (SPY-traded sibling) -- "
                 "same mechanism, different traded instrument, its own trial per this "
                 "pipeline's own never-pool-a-family-across-tickers-without-a-rule "
                 "discipline")

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result,
            catalog_entry=entry, instrument_set=["QQQ"],
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
            signal_source_path=signal_source_path,
            signal_source_sha256=signal_source_sha256)

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    return {
        "family_id": family_id, "shape": "single_long", "tickers": ["QQQ"],
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
    parser.add_argument("--qqq-quotes", default=DEFAULT_QQQ_QUOTES)
    parser.add_argument("--spy-quotes", default=DEFAULT_SPY_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)

    cache_dir = os.path.join(args.research_root, args.run_id, "artifacts", "ohlc")
    qqq_frame, _origin = batch5.load_frame(
        "QQQ", ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    spy_frame, _origin2 = batch5.load_frame(
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
        qqq_frame=qqq_frame, spy_frame=spy_frame,
        walcl=walcl, wtregen=wtregen, rrpontsyd=rrpontsyd,
        catalog_path=args.catalog, accepted_path=args.accepted,
        qqq_quotes_path=args.qqq_quotes, spy_quotes_path=args.spy_quotes,
        ledger_path=args.ledger,
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
        "fed-net-liquidity-qqq-dry-run.json" if args.dry_run
        else "fed-net-liquidity-qqq-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
