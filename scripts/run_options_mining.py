"""Bespoke ABI-1 driver for all three options-chain simulator families
(options_wheel_pooled_liquid, options_pmcc_pooled_liquid,
options_put_ratio_spread_pooled_liquid).

One script handling all three (not one script per family, unlike the
single-family bespoke drivers): they share the same pool, cost model, and
translation layer -- closer in shape to run_batch5_mining.py's own
"one script, many families" convention than to a single-family bespoke
driver.

Unlike the CBOE rotation and synthetic-leverage drivers, this one needs NO real captured options
quote at all -- option cost is a disclosed %-of-premium CONSTANT, not
measured from a captured bid/ask file. The benchmark (SPY) still needs a
real captured EQUITY quote for its own entry cost, reused from batch5's
own capture (execution_costs_for/refuse_synthetic_quotes).
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

import data_pipeline
import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import (abi1_driver, catalog as catalog_mod, constants, evidence,
                    options_curve, options_vol, trial_ledger, wheel_engine)
from mining.batch6 import options_pmcc, options_put_ratio_spread, options_wheel

FAMILIES = {
    options_wheel.FAMILY_ID: (options_wheel, wheel_engine.run_wheel),
    options_pmcc.FAMILY_ID: (options_pmcc, wheel_engine.run_pmcc),
    options_put_ratio_spread.FAMILY_ID: (
        options_put_ratio_spread, wheel_engine.run_put_ratio_spread),
}
CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
ACCOUNT_TYPE = "taxable"

DEFAULT_RUN_ID = "options-simulator-v1"
# The date a run is recorded as tested: today, unless --tested-date is passed.
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT

_SIGNAL_SOURCE_PATH = os.path.relpath(
    os.path.join(SRC_ROOT, "mining", "wheel_engine.py"), REPO_ROOT).replace(os.sep, "/")


def _load_accepted_entry(accepted_path: str, family_id: str) -> dict:
    """Same shape as scripts/run_synthetic_leveraged_dip_mining.py's own
    _load_accepted_entry -- every other real bespoke ABI-1 driver in this
    repo treats accepted.json as load-bearing, not merely catalog.json."""
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload["accepted"] if c["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 accepted candidate {family_id!r} in "
            f"{accepted_path}, found {len(matches)}")
    return matches[0]


def _run_cycle(mod, driver_fn, ticker: str, cache_dir: str, r_series):
    """`r_series` is fetched ONCE by the caller (compute_family) and
    threaded through here -- fetching it live per-ticker (10 x 3 = 30
    redundant fetches per run) with no guard on its None-on-failure
    return value would non-deterministically exclude whichever ticker's
    redundant fetch happened to fail that run. Matches the synthetic-leverage driver's own
    (run_synthetic_leveraged_dip_mining.py) precedent of fetching once
    before the per-ticker loop.

    `driver_fn` is called with the SAME 4 leading positional args
    (price_series, iv_series, r_series, div_yield) for all three
    families: wheel_engine.run_wheel/run_pmcc/run_put_ratio_spread all
    share the identical leading positional signature
    `(price_series, iv_series, r_series, div_yield, *, ...)`, with every
    other parameter keyword-only, so no per-family call-convention branch
    is needed here."""
    real_frame, _ = batch5.load_frame(
        ticker, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    price = real_frame["Close"]
    iv = options_vol.implied_vol_for(ticker, real_frame)
    r = r_series.reindex(price.index, method="ffill")
    div_yield = options_vol.dividend_yield_for(ticker)
    tier = options_vol.options_underlying_tier(ticker)
    cost_pct = constants.OPTIONS_COST_PCT_OF_PREMIUM[tier]

    kwargs = dict(mod.PARAMETERS)
    kwargs["cost_pct_of_premium"] = cost_pct
    if mod is options_pmcc:
        kwargs["nav_floor_fraction"] = constants.PMCC_NAV_FLOOR_FRACTION

    cycle = driver_fn(price, iv, r, div_yield, **kwargs)
    return options_curve.build_options_curve(cycle)


def compute_family(family_id: str, *, accepted_path: str, quotes_path: str,
                   catalog_path: str, ledger_path: str, run_dir: str,
                   tested_date: str, run_id: str, dry_run: bool) -> dict:
    mod, driver_fn = FAMILIES[family_id]
    try:
        _load_accepted_entry(accepted_path, family_id)  # validation only: must exist
    except (FileNotFoundError, ValueError) as exc:
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "not_accepted",
            "reason": f"{type(exc).__name__}: {exc}"}]}
    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    catalog_entry = catalog_entries.get(family_id)
    if catalog_entry is None:
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "not_curated",
            "reason": f"{family_id} is not in the catalog -- curate it first"}]}
    if catalog_entry.get("status") != "eligible":
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "catalog_status",
            "reason": f"catalog status is {catalog_entry.get('status')!r}"}]}
    drift = batch5.parameter_drift(mod.PARAMETERS, catalog_entry.get("parameters"))
    if drift is not None:
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "parameter_drift", "reason": drift}]}

    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"}
    if family_id in already_scored:
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "already_scored_abi1",
            "reason": "a prior run already recorded a construction=\"ABI-1\" "
                     "outcome for this family"}]}

    cache_dir = os.path.join(run_dir, "artifacts", "ohlc")
    spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK],
                                            batch5.load_quotes(quotes_path))
    benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0
    spy_frame, _ = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    r_series = data_pipeline.fetch_risk_free_rate(start="max")
    if r_series is None:
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "compute_error",
            "reason": "fetch_risk_free_rate returned None -- ^IRX fetch failed"}]}

    underlyings = list(mod.TRADED)
    excluded: dict = {}
    curves: dict = {}
    for ticker in underlyings:
        try:
            curves[ticker] = _run_cycle(mod, driver_fn, ticker, cache_dir, r_series)
        except Exception as exc:
            excluded[ticker] = f"{type(exc).__name__}: {exc}"
            continue

    if not batch5._pool_coverage_ok(list(curves), underlyings):
        kept, declared_n = len(curves), len(underlyings)
        fraction = (kept / declared_n) if declared_n else 0.0
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "pool_coverage",
            "reason": f"only {kept} of {declared_n} declared pool members "
                     f"are usable ({fraction:.1%}), below the "
                     f"{batch5.MIN_POOL_COVERAGE:.0%} floor. Excluded: "
                     + batch5._render_exclusions(excluded)}]}

    try:
        pooled_curve = options_curve.pool_options_curves(curves)
        result = abi1_driver.evaluate_family(
            pooled_curve, benchmark_frame=spy_frame,
            benchmark_entry_cost_bps=benchmark_entry_cost_bps,
            account_type=ACCOUNT_TYPE, tax_policy="none")
    except Exception as exc:
        return {"computed": [], "skipped": [{
            "family_id": family_id, "reason_code": "compute_error",
            "reason": f"{type(exc).__name__}: {exc}"}]}

    ticker_label = batch5.pool_ticker_label(sorted(curves))
    account_index = result["account_daily"].index
    account_start = account_index[0].strftime("%Y-%m-%d") if len(account_index) else None
    account_end = account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result,
            catalog_entry=catalog_entry, instrument_set=sorted(curves),
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy="none", ticker_label=ticker_label,
            extra_note=(f"pool: {len(curves)} of {len(underlyings)} declared "
                       f"members usable; option cost is a disclosed "
                       f"%-of-premium constant, not a captured quote; "
                       f"excluded=" + batch5._render_exclusions(excluded)
                       if excluded else
                       f"pool: {len(curves)} of {len(underlyings)} declared "
                       f"members usable; option cost is a disclosed "
                       f"%-of-premium constant, not a captured quote"),
            signal_source_path=_SIGNAL_SOURCE_PATH,
            signal_source_sha256=evidence.source_sha256(
                os.path.join(SRC_ROOT, "mining", "wheel_engine.py")))

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    item = {
        "family_id": family_id, "shape": mod.SHAPE,
        "pool_declared": underlyings, "pool_usable": sorted(curves),
        "pool_excluded": excluded, "ticker_label": ticker_label,
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "account_start": account_start, "account_end": account_end,
        "time_in_market": result.get("time_in_market"),
        "n_trades": len(pooled_curve["trades_df"]),
        "tax_policy": "none", "cash_convention": CASH_CONVENTION,
        "benchmark": MARKET_BENCHMARK,
        "holdout_evaluated": holdout.get("evaluated"),
        "holdout_passed": holdout.get("passed"),
        "holdout_refusal": holdout.get("refusal"),
        "window": {
            "account_sessions": window.get("account_sessions"),
            "scored_sessions": window.get("scored_sessions"),
            "in_sample_sessions": window.get("in_sample_sessions"),
            "holdout_sessions": window.get("holdout_sessions"),
            "holdout_split_applied": window.get("holdout_split_applied"),
        },
    }
    return {"computed": [item], "skipped": []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", required=True,
                        help="curation output accepted.json holding the three families")
    parser.add_argument("--quotes", required=True,
                        help="a dated, real captured equity quotes JSON file (SPY bid/ask)")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)
    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    all_computed, all_skipped = [], []
    for family_id in FAMILIES:
        result = compute_family(
            family_id, accepted_path=args.accepted, quotes_path=args.quotes,
            catalog_path=args.catalog, ledger_path=args.ledger, run_dir=run_dir,
            tested_date=args.tested_date, run_id=args.run_id, dry_run=args.dry_run)
        all_computed.extend(result["computed"])
        all_skipped.extend(result["skipped"])

    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/"),
        "results": [batch5._json_safe(item) for item in all_computed],
        "skipped": all_skipped,
    }
    output_path = os.path.join(
        run_dir, "options-simulator-dry-run.json" if args.dry_run
        else "options-simulator-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": output_path.replace(os.sep, "/"),
        "tested": len(payload["results"]), "skipped": len(payload["skipped"]),
        "survivors": sum(item["stage"] == "survived" for item in payload["results"]),
        "dry_run": args.dry_run,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
