"""Bespoke ABI-1 driver for sector_dispersion_regime_cyclical_
stance -- a candidate with no ``src/mining/batch5/*.py`` module: its
signal machinery (``sector_dispersion``/``dispersion_regime``/
``make_sector_dispersion_cyclical_stance_signal_fn``) lives directly in
``src/mining/signals.py``. There is no separate rough-backtest driver for
this family -- this driver's own ``--dry-run`` is the sanity check.

``make_sector_dispersion_cyclical_stance_signal_fn`` returns a plain
``mining.backtest_engine.SignalFn`` (``(frame, i) -> SignalResult``), the
SAME calling convention every batch5 module's own ``make_signal_fn``
factory returns -- so it plugs directly into
``abi1_driver.build_single_asset_curve`` unchanged, single_long shape,
same as every non-pooled bespoke driver in this pipeline.

Because there is no module to read ``PARAMETERS``/``SHAPE``/etc. from,
this driver defines its own local constants and checks them against BOTH
the catalog entry (parameter drift, matching every sibling driver) AND
``mining.signals``' own module-level defaults (``DISPERSION_PANEL``,
``DISPERSION_SMOOTHING_DAYS``, ``REGIME_LOOKBACK_DAYS``) -- catching drift
against the production signal math itself, not just against a duplicate
number typed into this file.
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

import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import abi1_driver, catalog as catalog_mod, evidence, signals, trial_ledger

FAMILY_ID = "sector_dispersion_regime_cyclical_stance"
TRADED_TICKER = "XLY"
DISPERSION_PANEL = list(signals.DISPERSION_PANEL)  # 9 original Select Sector SPDRs
DISPERSION_SMOOTHING_DAYS = signals.DISPERSION_SMOOTHING_DAYS
REGIME_LOOKBACK_DAYS = signals.REGIME_LOOKBACK_DAYS

# Byte-for-byte the catalog's own "parameters" field -- checked below
# against BOTH the catalog entry (parameter_drift) and the signals module's
# own live defaults (assert), so a change to either surfaces as a loud
# failure rather than a silent mismatch.
PARAMETERS = {
    "traded_ticker": TRADED_TICKER,
    "dispersion_panel": DISPERSION_PANEL,
    "dispersion_smoothing_days": DISPERSION_SMOOTHING_DAYS,
    "regime_lookback_days": REGIME_LOOKBACK_DAYS,
    "rebalance": "on_regime_change",
}
assert DISPERSION_SMOOTHING_DAYS == 20 and REGIME_LOOKBACK_DAYS == 252, (
    "sector_dispersion driver's local constants no longer match "
    "mining.signals' own live defaults -- re-check before scoring")

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
ACCOUNT_TYPE = "taxable"
TAX_POLICY = "realization"  # single_long shape

DEFAULT_RUN_ID = "sector-dispersion-abi1-v1"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()  # today, at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
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


def _build_panel_returns(frames: dict) -> pd.DataFrame:
    """Daily simple returns for every DISPERSION_PANEL member, pandas-
    aligned by date (an outer join via dict->DataFrame construction).
    Real Select Sector SPDR daily bars share one NYSE calendar, so no
    genuine cross-sectional gap is expected beyond each series' own
    leading pct_change() NaN -- sector_dispersion's own skipna=False
    contract surfaces any real misalignment as NaN rather than masking
    it."""
    returns = {ticker: frames[ticker]["Close"].pct_change() for ticker in DISPERSION_PANEL}
    return pd.DataFrame(returns)


def compute_and_record(*, accepted_path: str, quotes_path: str,
                       catalog_path: str, ledger_path: str, run_dir: str,
                       tested_date: str, run_id: str, dry_run: bool) -> dict:
    _load_accepted_entry(accepted_path, FAMILY_ID)  # validation only: must exist
    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    catalog_entry = catalog_entries.get(FAMILY_ID)
    if catalog_entry is None:
        raise ValueError(f"{FAMILY_ID} is not in the catalog -- curate it first")
    if catalog_entry.get("status") != "eligible":
        return {"family_id": FAMILY_ID, "reason_code": "catalog_status",
                "reason": f"catalog status is {catalog_entry.get('status')!r}"}
    drift = batch5.parameter_drift(PARAMETERS, catalog_entry.get("parameters"))
    if drift is not None:
        return {"family_id": FAMILY_ID, "reason_code": "parameter_drift", "reason": drift}

    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"}
    if FAMILY_ID in already_scored:
        return {"family_id": FAMILY_ID, "reason_code": "already_scored_abi1",
                "reason": "a prior run already recorded a construction=\"ABI-1\" outcome"}

    quotes_payload = batch5.load_quotes(quotes_path)
    batch5.refuse_synthetic_quotes(quotes_payload, dry_run=dry_run)
    costs = batch5.execution_costs_for([TRADED_TICKER, MARKET_BENCHMARK], quotes_payload)
    cost_bps = costs[TRADED_TICKER]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    cache_dir = os.path.join(run_dir, "artifacts", "ohlc")
    spy_frame, _ = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    xly_frame, _ = batch5.load_frame(
        TRADED_TICKER, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    panel_frames = {}
    for ticker in DISPERSION_PANEL:
        frame, _ = batch5.load_frame(
            ticker, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
        panel_frames[ticker] = frame

    panel_returns = _build_panel_returns(panel_frames)
    signal_fn = signals.make_sector_dispersion_cyclical_stance_signal_fn(
        panel_returns, smoothing_days=DISPERSION_SMOOTHING_DAYS,
        lookback_days=REGIME_LOOKBACK_DAYS)

    curve = abi1_driver.build_single_asset_curve(xly_frame, signal_fn, cost_bps=cost_bps)
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY)

    extra_note = (
        f"cost model: {TRADED_TICKER}={cost_bps:.2f}bps "
        f"({batch5.quotes_cost_source(quotes_payload)}); panel={len(DISPERSION_PANEL)} "
        "members, all 9 original Select Sector SPDRs, real OHLC, no synthetic input; "
        "no separate rough-backtest driver (this driver's own --dry-run serves that "
        "role)")

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, FAMILY_ID, run_id, tested_date, result,
            catalog_entry=catalog_entry, instrument_set=[TRADED_TICKER],
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
            signal_source_path=_SIGNAL_SOURCE_PATH,
            signal_source_sha256=evidence.source_sha256(__file__))

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    account_index = result["account_daily"].index
    account_start = account_index[0].strftime("%Y-%m-%d") if len(account_index) else None
    account_end = account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None
    return {
        "family_id": FAMILY_ID, "shape": "single_long", "tickers": [TRADED_TICKER],
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "n_trades": len(curve["trades_df"]),
        "account_start": account_start, "account_end": account_end,
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
    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    result = compute_and_record(
        accepted_path=args.accepted, quotes_path=args.quotes,
        catalog_path=args.catalog, ledger_path=args.ledger, run_dir=run_dir,
        tested_date=args.tested_date, run_id=args.run_id, dry_run=args.dry_run)

    safe_result = batch5._json_safe(result)
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": _SIGNAL_SOURCE_PATH, "result": safe_result,
    }
    output_path = os.path.join(
        run_dir, "sector-dispersion-dry-run.json" if args.dry_run
        else "sector-dispersion-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
