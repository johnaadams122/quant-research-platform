"""Bespoke ABI-1 driver for accrual_anomaly_sloan (Sloan 1996 accrual
anomaly).

Needs a bespoke entry point for the same reason the CBOE rotation, capex-cycle and synthetic-leverage drivers did
(scripts/run_cboe_rotation_mining.py, scripts/run_capex_cycle_mining.py,
scripts/run_synthetic_leveraged_dip_mining.py): the shared
``run_abi1_mining.py``/``run_batch5_mining.py`` dispatch calls
``module.make_signal_fn(ticker, index, frames)`` with exactly three
positional args, never the ``fundamentals_map`` keyword this candidate's
cross-sectional ranking needs (point-in-time SEC EDGAR net income/CFO/
assets per ticker across the WHOLE pool, never fetched by the shared
driver). This driver fetches those per-ticker EDGAR series explicitly
(via ``mining.accrual_coverage``, the fundamentals coverage check) and threads
the resulting ``fundamentals_map`` through the family's own optional
keyword.

UNLIKE ``run_capex_cycle_mining.py``, this family's signal never reads
SPY price data (the ranking is purely cross-sectional on ACCRUAL, not on
a price condition), so there is no per-ticker SPY-alignment step here --
``mining.abi1_driver.build_pooled_curve`` only needs each ticker's own
OHLC frame. SPY is still needed once, at the ACCOUNT level, for
``abi1_driver.evaluate_family``'s own benchmark-relative check (ABI-1
scores the account's daily excess over SPY, so no family is called a
survivor without a benchmark-relative check; see
``mining.benchmark_relative`` for the mechanism).

Pool-inclusion policy: a ticker missing OHLC or a real quote is excluded
from the pool entirely (matches every sibling pooled driver's own
``MIN_POOL_COVERAGE`` gate). A ticker whose EDGAR fetch fails or has no
usable disclosure stays IN the price pool -- its accrual ratio simply
degrades to NaN (the signal module's own documented graceful-
degradation path: it can never be selected, but its presence never
lowers ``min_rank_members``'s bar for everyone else). Fundamentals
coverage (how many of the pool's tickers got a usable 3-leg EDGAR
series) is measured via ``mining.accrual_coverage.run_coverage_check``
and gated against ``MIN_FUNDAMENTALS_COVERAGE`` -- below that floor the
run refuses rather than scoring a cross-section too thin to trust, same
"a materially narrower pool is a different hypothesis" discipline
``run_batch5_mining.MIN_POOL_COVERAGE`` already established for price
coverage.
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

import build_capex_watchlist as watchlist_script
import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import (abi1_driver, accrual_coverage, catalog as catalog_mod,
                    edgar_cik_resolution, evidence, trial_ledger)
from mining.batch5 import accrual_anomaly_sloan as main_mod

FAMILY_ID = main_mod.FAMILY_ID
CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
ACCOUNT_TYPE = "taxable"  # matches the catalog entry's account_types (taxable, roth);
                          # taxable is this pipeline's own ABI-1 default for a
                          # pooled_single_asset family (matches capex_cycle's own choice)
TAX_POLICY = "none"  # pooled_single_asset shape (pooled shapes are scored untaxed), matches every
                     # other sibling pooled family's own ABI-1 policy

# Same floor and rationale as batch5.MIN_POOL_COVERAGE (0.80), applied to
# the FUNDAMENTALS side rather than the price side -- see module docstring.
# PROVISIONAL_DEFAULT, same convention as every other invented
# floor in this pipeline.
MIN_FUNDAMENTALS_COVERAGE = 0.80

DEFAULT_RUN_ID = "accrual-anomaly"
DEFAULT_TESTED_DATE = None  # None -> today's local date at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
# Neutral default locations for the operator-supplied inputs (the accepted
# candidate record and a captured bid/ask quote file); pass --accepted and
# --quotes to point elsewhere.
DEFAULT_ACCEPTED = os.path.join(REPO_ROOT, "runs", "accrual-anomaly", "accepted.json")
DEFAULT_QUOTES = os.path.join(REPO_ROOT, "runs", "accrual-anomaly", "quotes.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def _load_accepted_entry(accepted_path: str, family_id: str):
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload["accepted"] if c["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 accepted candidate {family_id!r}, found {len(matches)}")
    return matches[0]


# -------------------------------------------------------------- fundamentals

def default_cik_by_ticker(tickers) -> dict:
    payload = watchlist_script._fetch_company_tickers()
    table = edgar_cik_resolution.parse_company_tickers_payload(payload)
    return edgar_cik_resolution.resolve_cik_for_tickers(table, tickers)


def build_fundamentals_map(tickers, *, user_agent: str | None = None,
                           sleep=time.sleep, min_interval: float | None = None,
                           cik_by_ticker: dict | None = None) -> tuple:
    """Fetch + fold a point-in-time ``{"ni": df, "cfo": df, "assets": df}``
    triple per ticker via ``mining.accrual_coverage.check_ticker_coverage``
    (never re-implemented here -- a drift between this driver's own
    fetching and that module's coverage-check accounting would be a real
    correctness bug). Returns ``(fundamentals_map, coverage_results)``:
    ``fundamentals_map`` omits any ticker whose coverage status is not
    ``"ok"`` (graceful degradation is the signal module's own job, not
    this function's), while ``coverage_results`` records the FULL
    ``check_ticker_coverage`` outcome for EVERY ticker, present or not --
    reported, never silently absorbed. A ticker with no resolvable CIK
    never reaches the network at all (``check_ticker_coverage``'s own
    ``no_cik`` short-circuit)."""
    from mining.edgar_data import (DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
                                   fetch_all_instant_tags_for_company,
                                   fetch_all_tags_for_company)
    interval = (min_interval if min_interval is not None
               else DEFAULT_MIN_REQUEST_INTERVAL_SECONDS)
    if cik_by_ticker is None:
        cik_by_ticker = default_cik_by_ticker(tickers)

    state = {"called": False}

    def _throttle():
        # Sleeps before every real HTTP request except the very first one
        # overall -- self-throttles the boundary BETWEEN the NI/CFO/Assets
        # legs too, not just within one leg's own fallback-tag loop
        # (fetch_all_tags_for_company already throttles those internally).
        if state["called"]:
            sleep(interval)
        state["called"] = True

    def fetch_tags_fn(cik, tags):
        _throttle()
        return fetch_all_tags_for_company(
            cik, tags, user_agent=user_agent, sleep=sleep, min_interval=interval)

    def fetch_instant_tags_fn(cik, tags):
        _throttle()
        return fetch_all_instant_tags_for_company(
            cik, tags, user_agent=user_agent, sleep=sleep, min_interval=interval)

    fundamentals_map: dict = {}
    coverage_results: dict = {}
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        result = accrual_coverage.check_ticker_coverage(
            ticker, cik, fetch_tags_fn=fetch_tags_fn,
            fetch_instant_tags_fn=fetch_instant_tags_fn)
        coverage_results[ticker] = result
        if result["status"] == "ok":
            fundamentals_map[ticker] = {
                "ni": result["ni_series"], "cfo": result["cfo_series"],
                "assets": result["assets_series"]}
    return fundamentals_map, coverage_results


# -------------------------------------------------------------------- compute

def _evaluate_family(*, pool_frames: dict, fundamentals_map: dict,
                     cost_bps_by_symbol: dict, benchmark_frame,
                     benchmark_entry_cost_bps: float):
    """Build every ticker's signal_fn -- each call gets the SAME, WHOLE
    ``fundamentals_map`` (every usable pool member's own fundamentals),
    never just its own -- the cross-sectional ranking this family's
    mechanism needs closes over every member, matching ``abi1_driver.
    build_pooled_curve``'s own documented precedent (see
    ``accrual_anomaly_sloan.make_signal_fn``'s own docstring)."""
    per_ticker_signal_fn = {
        ticker: main_mod.make_signal_fn(
            ticker, frame.index, {ticker: frame}, fundamentals_map=fundamentals_map)
        for ticker, frame in pool_frames.items()
    }
    curve = abi1_driver.build_pooled_curve(
        per_ticker_signal_fn, pool_frames, cost_bps_by_symbol=cost_bps_by_symbol)
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=benchmark_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY)
    return curve, result


def compute_batch_accrual_anomaly(*, accepted_path: str, quotes_path: str,
                                  catalog_path: str, ledger_path: str,
                                  run_dir: str, tested_date: str, run_id: str,
                                  dry_run: bool, fundamentals_map: dict | None = None,
                                  cik_by_ticker: dict | None = None) -> dict:
    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"}

    quotes_payload = batch5.load_quotes(quotes_path)
    batch5.refuse_synthetic_quotes(quotes_payload, dry_run=dry_run)

    computed: list = []
    skipped: list = []

    try:
        _load_accepted_entry(accepted_path, FAMILY_ID)
    except ValueError as exc:
        skipped.append({"family_id": FAMILY_ID, "reason_code": "not_in_accepted",
                        "reason": str(exc)})
        return {"computed": computed, "skipped": skipped}
    entry = catalog_entries.get(FAMILY_ID)
    if entry is None:
        skipped.append({"family_id": FAMILY_ID, "reason_code": "not_in_catalog",
                        "reason": f"{FAMILY_ID} is not in the catalog -- curate it first"})
        return {"computed": computed, "skipped": skipped}
    if entry.get("status") != "eligible":
        skipped.append({"family_id": FAMILY_ID, "reason_code": "catalog_status",
                        "reason": f"catalog status is {entry.get('status')!r}"})
        return {"computed": computed, "skipped": skipped}
    drift = batch5.parameter_drift(main_mod.PARAMETERS, entry.get("parameters"))
    if drift is not None:
        skipped.append({"family_id": FAMILY_ID, "reason_code": "parameter_drift",
                        "reason": drift})
        return {"computed": computed, "skipped": skipped}
    if FAMILY_ID in already_scored:
        skipped.append({"family_id": FAMILY_ID, "reason_code": "already_scored_abi1",
                        "reason": "a prior run already recorded a construction="
                                 "\"ABI-1\" outcome for this family"})
        return {"computed": computed, "skipped": skipped}

    cache_dir = os.path.join(run_dir, "artifacts", "ohlc")
    spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK], quotes_payload)
    benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0
    benchmark_frame, _ = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)

    declared_pool = list(main_mod.TRADED)
    pool_frames: dict = {}
    cost_bps_by_symbol: dict = {}
    excluded: dict = {}
    for ticker in declared_pool:
        try:
            frame, _ = batch5.load_frame(
                ticker, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
        except Exception as exc:
            excluded[ticker] = f"ohlc: {type(exc).__name__}: {exc}"
            continue
        try:
            cost = batch5.execution_costs_for([ticker], quotes_payload)
        except ValueError as exc:
            excluded[ticker] = f"quote gate: {exc}"
            continue
        pool_frames[ticker] = frame
        cost_bps_by_symbol[ticker] = float(cost[ticker]["cost_bps"])

    if not batch5._pool_coverage_ok(list(pool_frames), declared_pool):
        kept, declared_n = len(pool_frames), len(declared_pool)
        fraction = (kept / declared_n) if declared_n else 0.0
        reason = (f"only {kept} of {declared_n} declared pool members are usable "
                 f"({fraction:.1%}), below the {batch5.MIN_POOL_COVERAGE:.0%} price/"
                 f"quote floor. Excluded: " + batch5._render_exclusions(excluded))
        skipped.append({"family_id": FAMILY_ID, "reason_code": "pool_coverage",
                        "reason": reason})
        return {"computed": computed, "skipped": skipped}

    coverage_results: "dict | None" = None
    if fundamentals_map is None:
        fundamentals_map, coverage_results = build_fundamentals_map(
            sorted(pool_frames), cik_by_ticker=cik_by_ticker)
    # Restrict to price/quote-usable pool members ONLY, even when a caller
    # supplies a broader (e.g. cached full-declared-pool) fundamentals_map --
    # the CROSS-SECTIONAL
    # ranking closes over every key in fundamentals_map (mining.batch5.
    # accrual_anomaly_sloan.selection_panel), so a ticker present there but
    # ALREADY excluded from pool_frames (a bad OHLC fetch or a failed quote
    # gate) would still occupy a rank slot and could crowd a real tradable
    # member out of the bottom-fraction selected set, even though no
    # signal_fn is ever built for it and its own "selection" trades nothing.
    # The default fetch path above never triggers this (it only ever fetches
    # for sorted(pool_frames)), but an explicitly-passed fundamentals_map
    # (as the driver's tests pass, or a re-run reusing a cached full-pool
    # fetch) must not be trusted to already be scoped correctly.
    fundamentals_map = {t: v for t, v in fundamentals_map.items() if t in pool_frames}
    fundamentals_missing = sorted(t for t in pool_frames if t not in fundamentals_map)

    fundamentals_coverage_fraction = (
        (len(pool_frames) - len(fundamentals_missing)) / len(pool_frames)
        if pool_frames else 0.0)
    if fundamentals_coverage_fraction < MIN_FUNDAMENTALS_COVERAGE:
        reason = (f"only {len(pool_frames) - len(fundamentals_missing)} of "
                 f"{len(pool_frames)} price/quote-usable pool members have a usable "
                 f"3-leg EDGAR accrual series ({fundamentals_coverage_fraction:.1%}), "
                 f"below the {MIN_FUNDAMENTALS_COVERAGE:.0%} fundamentals-coverage "
                 f"floor. Missing: {', '.join(fundamentals_missing)}")
        skipped.append({"family_id": FAMILY_ID, "reason_code": "fundamentals_coverage",
                        "reason": reason})
        return {"computed": computed, "skipped": skipped}

    ticker_label = batch5.pool_ticker_label(sorted(pool_frames))

    try:
        curve, result = _evaluate_family(
            pool_frames=pool_frames, fundamentals_map=fundamentals_map,
            cost_bps_by_symbol=cost_bps_by_symbol, benchmark_frame=benchmark_frame,
            benchmark_entry_cost_bps=benchmark_entry_cost_bps)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if excluded:
            detail += "; already excluded: " + batch5._render_exclusions(excluded)
        skipped.append({"family_id": FAMILY_ID, "reason_code": "compute_error",
                        "reason": detail})
        return {"computed": computed, "skipped": skipped}

    account_index = result["account_daily"].index
    account_start = account_index[0].strftime("%Y-%m-%d") if len(account_index) else None
    account_end = account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None
    note = (f"pool: {len(pool_frames)} of {len(declared_pool)} declared members usable"
           + (f"; excluded=" + batch5._render_exclusions(excluded) if excluded else "")
           + f"; fundamentals: {len(pool_frames) - len(fundamentals_missing)} of "
             f"{len(pool_frames)} pool members had a usable 3-leg EDGAR accrual series"
           + (f"; fundamentals_missing=" + ",".join(fundamentals_missing)
              if fundamentals_missing else ""))

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        try:
            stage = abi1_driver.record_abi1_outcome(
                ledger_path, FAMILY_ID, run_id, tested_date, result,
                catalog_entry=entry, instrument_set=sorted(pool_frames),
                cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
                tax_policy=TAX_POLICY, ticker_label=ticker_label,
                extra_note=note, signal_source_path=_SIGNAL_SOURCE_PATH,
                signal_source_sha256=evidence.source_sha256(__file__))
        except Exception as exc:
            skipped.append({"family_id": FAMILY_ID, "reason_code": "record_error",
                            "reason": f"{type(exc).__name__}: {exc}"})
            return {"computed": computed, "skipped": skipped}

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    computed.append({
        "family_id": FAMILY_ID, "shape": main_mod.SHAPE,
        "pool_declared": declared_pool, "pool_usable": sorted(pool_frames),
        "pool_excluded": dict(excluded), "fundamentals_missing": fundamentals_missing,
        "fundamentals_coverage_fraction": fundamentals_coverage_fraction,
        "ticker_label": ticker_label,
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "account_start": account_start, "account_end": account_end,
        "time_in_market": result.get("time_in_market"),
        "n_trades": len(curve["trades_df"]),
        "tax_policy": TAX_POLICY, "cash_convention": CASH_CONVENTION,
        "benchmark": MARKET_BENCHMARK, "note": note,
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
    })

    return {"computed": computed, "skipped": skipped}


# ------------------------------------------------------------------------ run

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
    if args.tested_date is None:
        args.tested_date = time.strftime("%Y-%m-%d")
    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    result = compute_batch_accrual_anomaly(
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
        run_dir, "accrual-anomaly-dry-run.json" if args.dry_run else "accrual-anomaly-results.json")
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
