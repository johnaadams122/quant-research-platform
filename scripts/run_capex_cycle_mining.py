"""Bespoke ABI-1 driver for the capex-cycle blue-chip dip-buy family (capex_cycle_blue_chip_dip_pooled) and
its mandatory control arm (blue_chip_reversal_no_capex_control).

Both modules need a bespoke entry point for the same reason the CBOE rotation and synthetic-leverage drivers did
(scripts/run_cboe_rotation_mining.py, scripts/run_synthetic_leveraged_dip_
mining.py): ``run_abi1_mining.py``'s shared dispatch calls
``module.make_signal_fn(ticker, index, frames)`` with exactly three
positional args, never the ``fundamentals`` keyword the main family's own
``capex_condition_series`` needs (point-in-time SEC EDGAR capex disclosures
per ticker, never fetched by the shared driver). Routed through the generic
driver, ``capex_condition_series``'s own documented "no fundamentals known
yet" graceful-degradation path (``fundamentals is None`` -> always False)
would silently score every candidate month as "capex condition not met" --
a plausible-looking, zero-trade, economically MEANINGLESS result with no
error at all. This driver fetches those per-ticker EDGAR series explicitly
(reusing ``scripts/build_capex_watchlist.py``'s own fetch pipeline, never
re-implemented) and threads them through the family's own optional
``fundamentals`` keyword.

Both families share ONE pool (the same 25-ticker watchlist, same OHLC, same
quotes) and are evaluated as two SEPARATE ABI-1 trials in one run -- the
control arm is a mandatory falsification check for the main family (see
that module's own docstring), not a second candidacy decision, but it still
gets its own catalog/ledger entry and its own DSR score.

Pool-inclusion asymmetry, deliberate: a ticker missing OHLC or a real quote
is excluded from BOTH families' pools identically -- the watchlist must
stay IDENTICAL between them by construction (mirrors blue_chip_reversal_no_
capex_control.py's own stated purpose). A ticker whose EDGAR fetch fails or
has no usable disclosure is NOT excluded from the pool: it stays tradeable
for the control arm, and the main family's own capex_ok simply degrades to
False for it (the signal module's own documented, graceful behavior) --
excluding it from the pool entirely would silently shrink the main family's
watchlist without shrinking the control's, breaking the very invariant the
control arm exists to protect. Fundamentals coverage (how many of the
pool's tickers got a usable EDGAR series) is reported explicitly in every
run's output and ledger note, never silently absorbed.
"""
from __future__ import annotations

import argparse
import datetime
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
import run_cboe_rotation_mining as cboe_driver
from mining import abi1_driver, catalog as catalog_mod, edgar_cik_resolution, evidence, trial_ledger
from mining import sec_contact
from mining.batch5 import blue_chip_reversal_no_capex_control as control_mod
from mining.batch5 import capex_cycle_blue_chip_dip_pooled as main_mod

FAMILIES = (main_mod, control_mod)
CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
ACCOUNT_TYPE = "taxable"  # matches both modules'
                          # own catalog account_types entry
TAX_POLICY = "none"  # pooled_single_asset shape (pooled shapes are scored untaxed), matches
                     # every other sibling pooled family's own ABI-1 policy

DEFAULT_RUN_ID = "capex-cycle-v1"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()  # today, at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_ACCEPTED = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "accepted.json")
DEFAULT_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes.json")

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
    """Fetch + fold a point-in-time capex_growth_yoy series per ticker via
    the SAME pipeline ``scripts/build_capex_watchlist.py`` uses (imported,
    never re-implemented -- a drift between the watchlist's own screening
    and this driver's scoring would be a real correctness bug). Returns
    ``(fundamentals_map, coverage)``: ``fundamentals_map`` omits any ticker
    with no usable series (graceful degradation is the signal module's own
    job, not this function's), while ``coverage`` records the reason for
    EVERY ticker, present or not -- reported, never silently absorbed. A
    ticker with no resolvable CIK never reaches the network fetch at all."""
    from mining.edgar_data import DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
    interval = (min_interval if min_interval is not None
               else DEFAULT_MIN_REQUEST_INTERVAL_SECONDS)
    if cik_by_ticker is None:
        cik_by_ticker = default_cik_by_ticker(tickers)
    fundamentals_map: dict = {}
    coverage: dict = {}
    fetch_count = 0
    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        if cik is None:
            coverage[ticker] = "no_cik_resolved"
            continue
        if fetch_count > 0:
            sleep(interval)
        fetch_count += 1
        try:
            series, _tags_used = watchlist_script._build_point_in_time_series_for_ticker(
                cik, user_agent=user_agent, sleep=sleep, min_interval=interval)
        except sec_contact.SecContactMissing:
            raise
        except Exception as exc:
            coverage[ticker] = f"fetch_error:{type(exc).__name__}: {exc}"
            continue
        if series is None or series.empty:
            coverage[ticker] = "no_usable_disclosure"
            continue
        fundamentals_map[ticker] = series
        coverage[ticker] = "ok"
    return fundamentals_map, coverage


# -------------------------------------------------------------------- compute

def align_pool_with_benchmark(pool_frames: dict, benchmark_frame) -> tuple:
    """Both signal modules' ``_make_signal_fn`` require ``own.index.equals(
    bench.index)`` -- a real individual stock's inception routinely differs
    from SPY's own (1993-01-29; two of this watchlist's own members, CAT
    and CNP, start in 1962), so passing raw full-history frames unmodified
    raises for essentially every real ticker (synthetic fixtures with a
    shared inception date would never expose this).
    Reuses ``run_cboe_rotation_mining._intersect_traded_calendars`` UNCHANGED
    (never re-implemented) -- the CBOE rotation driver's own SPY/^PUT alignment needs the
    identical "trim to common inception, then intersect, fail closed above
    a sparse-gap cap" mechanism. Computed ONCE here and shared by both
    families so their pools can never diverge on calendar grounds alone.
    Returns ``(pool_frames, spy_frame_by_ticker, excluded)`` -- a ticker
    whose alignment fails (a real cross-vendor gap exceeding the cap, not
    merely a later inception) is dropped from ``pool_frames`` and recorded
    in ``excluded``, the SAME dict OHLC/quote failures already populate --
    a calendar problem is a POOL-level exclusion, not a per-family one."""
    aligned_pool: dict = {}
    spy_frame_by_ticker: dict = {}
    excluded: dict = {}
    for ticker, frame in pool_frames.items():
        try:
            pair, _dropped = cboe_driver._intersect_traded_calendars(
                {ticker: frame, MARKET_BENCHMARK: benchmark_frame})
        except ValueError as exc:
            excluded[ticker] = f"calendar: {exc}"
            continue
        aligned_pool[ticker] = pair[ticker]
        spy_frame_by_ticker[ticker] = pair[MARKET_BENCHMARK]
    return aligned_pool, spy_frame_by_ticker, excluded


def _evaluate_one_family(module, *, pool_frames: dict, spy_frame_by_ticker: dict,
                         fundamentals_map: dict, cost_bps_by_symbol: dict,
                         benchmark_frame, benchmark_entry_cost_bps: float):
    """Build every ticker's signal_fn against ITS OWN pre-aligned SPY frame
    (never a single frame shared verbatim across the whole pool -- each
    ticker was trimmed to its own common-inception-with-SPY window by
    ``align_pool_with_benchmark``), then the pooled curve + ABI-1
    evaluation. Any exception here is a genuine, unexpected data problem
    (calendar mismatches are already resolved before this is called) and
    is left to the caller's own per-family try/except, matching every
    sibling bespoke driver's convention of never individually guarding a
    single ``make_signal_fn`` call."""
    per_ticker_signal_fn = {
        ticker: module.make_signal_fn(
            ticker, frame.index, {ticker: frame, MARKET_BENCHMARK: spy_frame_by_ticker[ticker]},
            fundamentals=fundamentals_map)
        for ticker, frame in pool_frames.items()
    }
    curve = abi1_driver.build_pooled_curve(
        per_ticker_signal_fn, pool_frames, cost_bps_by_symbol=cost_bps_by_symbol)
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=benchmark_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY)
    return curve, result


def compute_batch_capex_cycle(*, accepted_path: str, quotes_path: str,
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

    # Preflight both families independently (accepted/catalog/drift/re-run
    # guard) -- a problem with one never blocks the other.
    plans: dict = {}
    for module in FAMILIES:
        family_id = module.FAMILY_ID
        try:
            _load_accepted_entry(accepted_path, family_id)
        except ValueError as exc:
            skipped.append({"family_id": family_id, "reason_code": "not_in_accepted",
                            "reason": str(exc)})
            continue
        entry = catalog_entries.get(family_id)
        if entry is None:
            skipped.append({"family_id": family_id, "reason_code": "not_in_catalog",
                            "reason": f"{family_id} is not in the catalog -- curate it first"})
            continue
        if entry.get("status") != "eligible":
            skipped.append({"family_id": family_id, "reason_code": "catalog_status",
                            "reason": f"catalog status is {entry.get('status')!r}"})
            continue
        drift = batch5.parameter_drift(module.PARAMETERS, entry.get("parameters"))
        if drift is not None:
            skipped.append({"family_id": family_id, "reason_code": "parameter_drift",
                            "reason": drift})
            continue
        if family_id in already_scored:
            skipped.append({"family_id": family_id, "reason_code": "already_scored_abi1",
                            "reason": "a prior run already recorded a construction="
                                     "\"ABI-1\" outcome for this family"})
            continue
        plans[family_id] = entry

    if not plans:
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

    pool_frames, spy_frame_by_ticker, calendar_excluded = align_pool_with_benchmark(
        pool_frames, benchmark_frame)
    excluded.update(calendar_excluded)
    cost_bps_by_symbol = {t: c for t, c in cost_bps_by_symbol.items() if t in pool_frames}

    if not batch5._pool_coverage_ok(list(pool_frames), declared_pool):
        kept, declared_n = len(pool_frames), len(declared_pool)
        fraction = (kept / declared_n) if declared_n else 0.0
        reason = (f"only {kept} of {declared_n} declared pool members are usable "
                 f"({fraction:.1%}), below the {batch5.MIN_POOL_COVERAGE:.0%} floor. "
                 f"Excluded: " + batch5._render_exclusions(excluded))
        for family_id in plans:
            skipped.append({"family_id": family_id, "reason_code": "pool_coverage",
                            "reason": reason})
        return {"computed": computed, "skipped": skipped}

    if fundamentals_map is None:
        fundamentals_map, _fundamentals_coverage = build_fundamentals_map(
            sorted(pool_frames), cik_by_ticker=cik_by_ticker)
    fundamentals_missing = sorted(t for t in pool_frames if t not in fundamentals_map)

    ticker_label = batch5.pool_ticker_label(sorted(pool_frames))

    for module in FAMILIES:
        family_id = module.FAMILY_ID
        if family_id not in plans:
            continue
        entry = plans[family_id]
        try:
            curve, result = _evaluate_one_family(
                module, pool_frames=pool_frames, spy_frame_by_ticker=spy_frame_by_ticker,
                fundamentals_map=fundamentals_map, cost_bps_by_symbol=cost_bps_by_symbol,
                benchmark_frame=benchmark_frame,
                benchmark_entry_cost_bps=benchmark_entry_cost_bps)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if excluded:
                detail += "; already excluded: " + batch5._render_exclusions(excluded)
            skipped.append({"family_id": family_id, "reason_code": "compute_error",
                            "reason": detail})
            continue

        account_index = result["account_daily"].index
        account_start = account_index[0].strftime("%Y-%m-%d") if len(account_index) else None
        account_end = account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None
        note = (f"pool: {len(pool_frames)} of {len(declared_pool)} declared members usable"
               + (f"; excluded=" + batch5._render_exclusions(excluded) if excluded else "")
               + f"; fundamentals: {len(pool_frames) - len(fundamentals_missing)} of "
                 f"{len(pool_frames)} pool members had a usable EDGAR capex series"
               + (f"; fundamentals_missing=" + ",".join(fundamentals_missing)
                  if fundamentals_missing else ""))

        if dry_run:
            verdict = result["verdict"]
            stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
        else:
            try:
                stage = abi1_driver.record_abi1_outcome(
                    ledger_path, family_id, run_id, tested_date, result,
                    catalog_entry=entry, instrument_set=sorted(pool_frames),
                    cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
                    tax_policy=TAX_POLICY, ticker_label=ticker_label,
                    extra_note=note, signal_source_path=_SIGNAL_SOURCE_PATH,
                    signal_source_sha256=evidence.source_sha256(__file__))
            except Exception as exc:
                skipped.append({"family_id": family_id, "reason_code": "record_error",
                                "reason": f"{type(exc).__name__}: {exc}"})
                continue

        layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
        computed.append({
            "family_id": family_id, "shape": module.SHAPE,
            "pool_declared": declared_pool, "pool_usable": sorted(pool_frames),
            "pool_excluded": dict(excluded), "fundamentals_missing": fundamentals_missing,
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
    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    result = compute_batch_capex_cycle(
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
        run_dir, "capex-cycle-dry-run.json" if args.dry_run else "capex-cycle-results.json")
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
