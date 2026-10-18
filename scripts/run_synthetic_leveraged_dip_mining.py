"""Bespoke ABI-1 driver for synthetic_leveraged_dip_buy_pooled_largecap
(the synthetic 2x leveraged dip-buy family).

This family trades a CONSTRUCTED price series (a synthetic 2x
margin-financed position) rather than one fetched directly from a vendor.
``run_abi1_mining.py``'s shared dispatch assumes every TRADED symbol is a
literal string a quote file and an OHLC fetcher both recognize -- false
here (a synthetic id like ``"TSLA_2x_synthetic"`` has no quote and no
yfinance ticker).

Mirrors the CBOE rotation driver's own ``scripts/run_cboe_rotation_mining.py`` precedent
(a bespoke assembly point reusing every existing ABI-1 primitive
unchanged) rather than the fully-generic multi-shape dispatcher, since
this driver only ever needs to support one shape
(``pooled_single_asset``) for one family.

Per-underlying flow: load the real OHLC + a REAL captured quote (the
underlying stock's own spread is the trading-cost basis, by design
-- there is no separate "leveraged product" to quote), build that
underlying's synthetic 2x frame via ``data_pipeline.
build_synthetic_leveraged`` (financed off the real, once-fetched ^IRX
series) and wrap it into OHLC via the family module's own
``synthetic_ohlc_from_close``, store it under its OWN synthetic id (never
overwriting the real ticker's own frame -- this driver's ``frames`` dict
is process-local to this one run, so no other family's data is at risk
regardless). A member missing OHLC or a real quote is EXCLUDED (not
fatal) subject to the same ``MIN_POOL_COVERAGE`` floor every other pooled
family already enforces.
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

import pandas as pd

import data_pipeline
import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import abi1_driver, catalog as catalog_mod, evidence, trial_ledger
from mining.batch5 import synthetic_leveraged_dip_buy_pooled_largecap as mod

FAMILY_ID = mod.FAMILY_ID
CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
ACCOUNT_TYPE = "taxable"  # Reg-T 2x margin is not generally meaningful in
                          # a tax-advantaged account, so this family is
                          # modelled in a taxable one.

DEFAULT_RUN_ID = "synthetic-leveraged-dip-v1"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()  # today, at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
# Default inputs live in this driver's own run folder; pass --accepted /
# --quotes to point at a curated candidate file and a real quote capture.
DEFAULT_ACCEPTED = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "accepted.json")
DEFAULT_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes.json")

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


def _build_underlying_and_synthetic(underlying_ticker: str, real_frame: pd.DataFrame,
                                    irx_series: pd.Series) -> pd.DataFrame:
    """One underlying's synthetic 2x OHLC frame, built from ITS OWN real
    OHLC frame (already indexed by date, columns Open/High/Low/Close) --
    ``build_synthetic_leveraged`` needs a ``Date`` COLUMN plus a column
    literally named ``underlying_ticker`` (build_synthetic_3x's own
    calling convention, unchanged), so this adapts the shape at the call
    site rather than modifying that function."""
    df_for_synth = pd.DataFrame({
        "Date": real_frame.index, underlying_ticker: real_frame["Close"].to_numpy(),
    })
    synth = data_pipeline.build_synthetic_leveraged(
        df_for_synth, underlying_ticker, annual_financing_rate=irx_series)
    synthetic_id = mod.synthetic_id_for(underlying_ticker)
    close = synth[synthetic_id].to_numpy()
    return mod.synthetic_ohlc_from_close(real_frame.index, close, name=synthetic_id)


def compute_batch_synthetic_leveraged(*, accepted_path: str, quotes_path: str,
                                      catalog_path: str, ledger_path: str,
                                      run_dir: str, tested_date: str,
                                      run_id: str, dry_run: bool) -> dict:
    _load_accepted_entry(accepted_path, FAMILY_ID)  # validation only: must exist
    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    catalog_entry = catalog_entries.get(FAMILY_ID)
    if catalog_entry is None:
        raise ValueError(f"{FAMILY_ID} is not in the catalog -- curate it first")
    if catalog_entry.get("status") != "eligible":
        return {"computed": [], "skipped": [{
            "family_id": FAMILY_ID, "reason_code": "catalog_status",
            "reason": f"catalog status is {catalog_entry.get('status')!r}"}]}
    drift = batch5.parameter_drift(mod.PARAMETERS, catalog_entry.get("parameters"))
    if drift is not None:
        return {"computed": [], "skipped": [{
            "family_id": FAMILY_ID, "reason_code": "parameter_drift", "reason": drift}]}

    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"}
    if FAMILY_ID in already_scored:
        return {"computed": [], "skipped": [{
            "family_id": FAMILY_ID, "reason_code": "already_scored_abi1",
            "reason": "a prior run already recorded a construction=\"ABI-1\" "
                     "outcome for this family"}]}

    quotes_payload = batch5.load_quotes(quotes_path)
    batch5.refuse_synthetic_quotes(quotes_payload, dry_run=dry_run)

    cache_dir = os.path.join(run_dir, "artifacts", "ohlc")
    spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK], quotes_payload)
    benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0
    spy_frame, _ = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)

    irx_series = data_pipeline.fetch_risk_free_rate(start="max")

    underlyings = list(mod.SIGNAL)
    excluded: dict = {}
    per_ticker_signal_fn: dict = {}
    full_frames: dict = {}
    cost_bps_by_symbol: dict = {}
    for underlying_ticker in underlyings:
        synthetic_id = mod.synthetic_id_for(underlying_ticker)
        try:
            real_frame, _ = batch5.load_frame(
                underlying_ticker, ohlc_loader=batch5.default_ohlc_loader,
                cache_dir=cache_dir)
        except Exception as exc:
            excluded[underlying_ticker] = f"ohlc: {type(exc).__name__}: {exc}"
            continue
        try:
            cost = batch5.execution_costs_for([underlying_ticker], quotes_payload)
        except ValueError as exc:
            excluded[underlying_ticker] = f"quote gate: {exc}"
            continue

        synthetic_frame = _build_underlying_and_synthetic(
            underlying_ticker, real_frame, irx_series)
        signal_fn = mod.make_signal_fn(
            synthetic_id, synthetic_frame.index,
            {synthetic_id: synthetic_frame, underlying_ticker: real_frame})
        per_ticker_signal_fn[synthetic_id] = signal_fn
        full_frames[synthetic_id] = synthetic_frame
        cost_bps_by_symbol[synthetic_id] = float(
            cost[underlying_ticker]["cost_bps"])

    if not batch5._pool_coverage_ok(list(full_frames), underlyings):
        kept, declared_n = len(full_frames), len(underlyings)
        fraction = (kept / declared_n) if declared_n else 0.0
        return {"computed": [], "skipped": [{
            "family_id": FAMILY_ID, "reason_code": "pool_coverage",
            "reason": f"only {kept} of {declared_n} declared pool members "
                     f"are usable ({fraction:.1%}), below the "
                     f"{batch5.MIN_POOL_COVERAGE:.0%} floor. Excluded: "
                     + batch5._render_exclusions(excluded)}]}

    try:
        curve = abi1_driver.build_pooled_curve(
            per_ticker_signal_fn, full_frames, cost_bps_by_symbol=cost_bps_by_symbol)
        result = abi1_driver.evaluate_family(
            curve, benchmark_frame=spy_frame,
            benchmark_entry_cost_bps=benchmark_entry_cost_bps,
            account_type=ACCOUNT_TYPE, tax_policy="none")
    except Exception as exc:
        return {"computed": [], "skipped": [{
            "family_id": FAMILY_ID, "reason_code": "compute_error",
            "reason": f"{type(exc).__name__}: {exc}"}]}

    ticker_label = batch5.pool_ticker_label(sorted(full_frames))
    account_index = result["account_daily"].index
    account_start = account_index[0].strftime("%Y-%m-%d") if len(account_index) else None
    account_end = account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, FAMILY_ID, run_id, tested_date, result,
            catalog_entry=catalog_entry, instrument_set=sorted(full_frames),
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy="none", ticker_label=ticker_label,
            extra_note=(f"pool: {len(full_frames)} of {len(underlyings)} "
                       f"declared members usable; excluded="
                       + batch5._render_exclusions(excluded) if excluded
                       else f"pool: {len(full_frames)} of {len(underlyings)} "
                            "declared members usable"),
            signal_source_path=_SIGNAL_SOURCE_PATH,
            signal_source_sha256=evidence.source_sha256(__file__))

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    item = {
        "family_id": FAMILY_ID, "shape": mod.SHAPE,
        "pool_declared": underlyings, "pool_usable": sorted(full_frames),
        "pool_excluded": excluded, "ticker_label": ticker_label,
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "account_start": account_start, "account_end": account_end,
        "time_in_market": result.get("time_in_market"),
        "n_trades": len(curve["trades_df"]),
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
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--quotes", default=DEFAULT_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)
    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    result = compute_batch_synthetic_leveraged(
        accepted_path=args.accepted, quotes_path=args.quotes,
        catalog_path=args.catalog, ledger_path=args.ledger, run_dir=run_dir,
        tested_date=args.tested_date, run_id=args.run_id, dry_run=args.dry_run)

    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": _SIGNAL_SOURCE_PATH,
        "results": [batch5._json_safe(item) for item in result["computed"]],
        "skipped": result["skipped"],
    }
    output_path = os.path.join(
        run_dir, "synthetic-leveraged-dip-dry-run.json" if args.dry_run
        else "synthetic-leveraged-dip-results.json")
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
