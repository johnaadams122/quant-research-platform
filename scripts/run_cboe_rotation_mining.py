"""Bespoke ABI-1 driver for Cboe index-overlay rotation families whose
non-tradable leg has no real captured quote.

``scripts/run_abi1_mining.py`` (the shared ABI-1 dispatch) prices every
TRADED symbol off a real captured bid/ask via ``run_batch5_mining.
execution_costs_for`` -- correct for every existing family, all of which
trade real ETFs/stocks/futures proxies, but ``^PUT``/``^BXM`` are
non-tradable CALCULATED indices with no bid/ask to capture. Routing them
through the shared driver would either invent a fake quote (which
``refuse_synthetic_quotes`` and this repo's own "a recorded trial must be
priced from a measured quote sample" convention both argue against) or
skip the whole family with a "missing_data" reason regardless of merit.

The synthetic 2x leveraged dip-buy family has the identical shape of
problem -- it trades a constructed price series rather than a fetched
one, so it needs a small bespoke driver script -- and uses the same fix: reuse every existing ABI-1
primitive (``abi1_driver.build_rotation_curve`` / ``evaluate_family`` /
``record_abi1_outcome``) unchanged, just assemble the inputs differently
at one bespoke entry point. This module is that entry point for the
"real series, no real quote" case.

Every OTHER TRADED symbol in a registered family (e.g. the SPY leg here)
still prices off a REAL captured quote via the ordinary
``execution_costs_for`` path -- only the symbol(s) named in
``ASSUMED_COST_REGISTRY`` bypass the quote gate, using that family's own
disclosed module-level constant instead. The ledger's own ``note`` field
and this script's run report both say so explicitly, so a reader can
never mistake an assumed cost for a measured one.

v1 registry: only ``cboe_putwrite_vol_regime_rotation`` (clean, gap-free
``^PUT`` data). Its sibling
``cboe_buywrite_vol_regime_rotation`` (``^BXM``) is NOT registered --
its vendor feed has carried a multi-week contiguous gap in recent data.
Add it here once that gap is understood/resolved.
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
from mining import abi1_driver, constants, evidence, reporting, trial_ledger
from mining.batch5 import cboe_putwrite_vol_regime_rotation as putwrite_mod

# {family_id: {symbol: assumed_cost_bps}} -- a symbol named here bypasses
# the real quote gate entirely for that family; every other TRADED symbol
# in the same family still requires a real captured quote.
ASSUMED_COST_REGISTRY = {
    putwrite_mod.FAMILY_ID: {"PUT": putwrite_mod.ASSUMED_COST_BPS},
}

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"

DEFAULT_RUN_ID = "cboe-rotation-v1"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_MODULES_DIR = batch5.DEFAULT_MODULES_DIR
DEFAULT_ACCEPTED = os.path.join(
    RESEARCH_ROOT, "batch6", "curation", "accepted.json")
DEFAULT_QUOTES = batch5.DEFAULT_QUOTES

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


# --------------------------------------------------------------- costs

def costs_for_family(family_id: str, declared: list, quotes_payload) -> dict:
    """One ``{symbol: {..., cost_bps, source}}`` entry per symbol in
    ``declared``. A symbol named in ``ASSUMED_COST_REGISTRY[family_id]``
    bypasses the real quote gate entirely and uses that family's own
    disclosed constant (``source="assumed_constant"``, no bid/ask -- there
    is none to report). Every other symbol prices off a REAL captured
    quote via ``run_batch5_mining.execution_costs_for``, unchanged --
    raises ``ValueError`` (same as that function) naming the first symbol
    that fails the real-quote gate."""
    assumed = ASSUMED_COST_REGISTRY.get(family_id, {})
    quoted_symbols = [symbol for symbol in declared if symbol not in assumed]
    costs: dict = {}
    if quoted_symbols:
        costs.update(batch5.execution_costs_for(quoted_symbols, quotes_payload))
    for symbol in declared:
        if symbol in assumed:
            costs[symbol] = {
                "bid": None, "ask": None, "spread_bps": None,
                "cost_bps": float(assumed[symbol]),
                "security_status": None, "source": "assumed_constant",
            }
    return costs


# A guard, not a convenience: this bounds how much _intersect_traded_
# calendars is ALLOWED to silently absorb, so a future large, contiguous
# vendor outage (the shape of the sibling ^BXM's own multi-week gap)
# still fails closed here rather than being silently traded through --
# only genuinely sparse cross-vendor print gaps (a handful of isolated
# days for ^PUT over decades of history) are meant to pass. PROVISIONAL_DEFAULT, same disclosed-default convention as
# every other invented constant in this pipeline.
MAX_SPARSE_CALENDAR_GAP_DAYS = 20


def _intersect_traded_calendars(asset_frames: dict) -> tuple:
    """Cboe's ``^PUT``/``^BXM`` index feed occasionally misses a single
    day's print that SPY's own feed (a real traded ETF) has -- a handful
    of sparse, isolated days across the history, NOT a contiguous gap
    (unlike the sibling ``^BXM``'s own multi-week block, which this driver
    does not register or attempt to paper over -- see
    ``cboe_putwrite_vol_regime_rotation``'s own docstring).

    Two independently published vendor feeds disagreeing on a handful of
    sparse print days is expected and benign, unlike
    ``abi1_driver._align_rotation_frames``'s own "same-exchange
    instrument" interior-mismatch guard, which deliberately fails closed
    for two ETFs that SHOULD trade on identical days. Intersects every
    traded leg onto their common calendar HERE, before that guard ever
    runs, so a genuine mismatch between two same-exchange instruments would
    still be caught (it only ever sees the post-intersection, exactly
    matching frames) while this benign cross-vendor gap is not mistaken
    for one.

    First trims every leg to their COMMON INCEPTION (the max of each
    leg's own first date) -- exactly ``abi1_driver._align_rotation_
    frames``'s own first step -- so a leg simply starting later (SPY
    1993 vs PUT 1996) is never counted as a "dropped" day; only a gap
    INTERIOR to the common, post-trim window counts.

    Note this drops the missing-print day from EVERY leg, not only the one
    that lacked the print -- e.g. a day SPY has and PUT does not removes
    that day's real SPY move from the scored account too, rather than
    forward-filling PUT's stale price into it. At a magnitude of a small
    fraction of one percent of sessions this is very unlikely to move a
    verdict, and it is disclosed downstream via `account_relative.log_excess`'s own
    `benchmark_truncated` flag -- named here explicitly rather than left
    implicit, since "drop the day everywhere" was a deliberate choice
    (favors removing an ambiguous day over inventing a price for it), not
    an oversight.

    Raises ``ValueError`` if any leg would lose more than
    ``MAX_SPARSE_CALENDAR_GAP_DAYS`` interior sessions -- a large drop is
    exactly the ^BXM-style live feed outage this function must NOT paper
    over, so it fails closed rather than silently trading through it.
    Returns ``(aligned_frames, dropped_by_symbol)`` so the caller reports
    even a small, allowed drop loudly rather than hiding it."""
    common_inception = max(frame.index.min() for frame in asset_frames.values())
    trimmed = {symbol: frame.loc[frame.index >= common_inception]
              for symbol, frame in asset_frames.items()}
    common = None
    for frame in trimmed.values():
        common = frame.index if common is None else common.intersection(frame.index)
    dropped_by_symbol = {symbol: len(frame.index) - len(common)
                         for symbol, frame in trimmed.items()}
    oversized = {symbol: n for symbol, n in dropped_by_symbol.items()
                if n > MAX_SPARSE_CALENDAR_GAP_DAYS}
    if oversized:
        raise ValueError(
            "cross-vendor calendar mismatch exceeds the "
            f"{MAX_SPARSE_CALENDAR_GAP_DAYS}-session sparse-gap allowance "
            f"(likely a real feed outage, not a sparse print gap): {oversized}")
    aligned = {symbol: frame.loc[frame.index.isin(common)].copy()
              for symbol, frame in trimmed.items()}
    return aligned, dropped_by_symbol


# ------------------------------------------------------------- compute

def compute_batch_cboe_rotation(scope: dict, *, tested_date: str, run_id: str,
                                ledger_path: str, dry_run: bool) -> dict:
    """Same overall shape as ``run_abi1_mining.compute_batch_abi1`` (module
    discovery -> catalog/drift/re-run checks -> quote gate -> OHLC load ->
    curve -> evaluate -> record), narrowed to registered
    ``rotation_two_asset`` families whose cost gate is
    ``costs_for_family`` instead of a pure quote lookup. Every candidate
    NOT in ``ASSUMED_COST_REGISTRY`` is skipped with reason code
    ``not_registered`` -- this driver never silently scores an
    unregistered family."""
    skipped: list = []
    computed: list = []
    quotes_payload = scope["quotes"]
    problem_by_module = {p["module"]: p["reason"] for p in scope["module_problems"]}

    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"
    }

    try:
        spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK], quotes_payload)
        benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0
    except ValueError as exc:
        for candidate in scope["accepted"]:
            batch5._skip(skipped, candidate["family_id"], "missing_data",
                        f"benchmark quote gate ({MARKET_BENCHMARK}): {exc}")
        return {"computed": computed, "skipped": skipped,
                "input_coverage": {}, "input_errors": {},
                "benchmark_entry_cost_bps": None}

    plans = []
    for candidate in scope["accepted"]:
        family_id = candidate["family_id"]
        if family_id not in ASSUMED_COST_REGISTRY:
            batch5._skip(skipped, family_id, "not_registered",
                        "family_id is not in this bespoke driver's "
                        "ASSUMED_COST_REGISTRY")
            continue
        found = scope["modules"].get(family_id)
        if found is None:
            reason = problem_by_module.get(family_id)
            batch5._skip(skipped, family_id, "invalid_module" if reason else "no_module",
                        reason or f"no valid module for {family_id} under "
                                  f"{scope['modules_dir']}")
            continue
        spec, module = found["spec"], found["module"]
        if spec["shape"] != batch5.ROTATION_SHAPE:
            batch5._skip(skipped, family_id, "unsupported_shape",
                        f"this driver only supports {batch5.ROTATION_SHAPE!r}, "
                        f"got {spec['shape']!r}")
            continue
        entry = scope["catalog_by_id"].get(family_id)
        if entry is None:
            batch5._skip(skipped, family_id, "not_in_catalog",
                        "family_id absent from the catalog")
            continue
        if entry.get("status") != "eligible":
            batch5._skip(skipped, family_id, "catalog_status",
                        f"catalog status is {entry.get('status')!r}, not eligible")
            continue
        drift = batch5.parameter_drift(spec["parameters"], entry.get("parameters"))
        if drift is not None:
            batch5._skip(skipped, family_id, "parameter_drift", drift)
            continue
        if family_id in already_scored:
            batch5._skip(skipped, family_id, "already_scored_abi1",
                        "a prior run already recorded a construction="
                        "\"ABI-1\" outcome for this family")
            continue

        declared = list(spec["traded"])
        try:
            costs = costs_for_family(family_id, declared, quotes_payload)
        except ValueError as exc:
            batch5._skip(skipped, family_id, "missing_data", f"quote gate: {exc}")
            continue

        plans.append({
            "family_id": family_id, "spec": spec, "module": module,
            "entry": entry, "costs": costs, "effective": declared,
        })

    symbols = {MARKET_BENCHMARK}
    for plan in plans:
        symbols.update(plan["effective"])
        symbols.update(plan["spec"]["signal"])
    frames: dict = {}
    load_errors: dict = {}
    coverage: dict = {}
    cache_dir = os.path.join(scope["run_dir"], "artifacts", "ohlc")
    for symbol in sorted(symbols):
        try:
            report: dict = {}
            frame, origin = batch5.load_frame(
                symbol, ohlc_loader=scope["ohlc_loader"], cache_dir=cache_dir,
                report=report)
            frames[symbol] = frame
            coverage[symbol] = {
                "first": frame.index.min().strftime("%Y-%m-%d"),
                "last": frame.index.max().strftime("%Y-%m-%d"),
                "origin": origin, **report,
            }
        except Exception as exc:
            load_errors[symbol] = f"{type(exc).__name__}: {exc}"

    benchmark_frame = frames.get(MARKET_BENCHMARK)
    if benchmark_frame is None:
        for plan in plans:
            batch5._skip(skipped, plan["family_id"], "missing_data",
                        f"benchmark OHLC ({MARKET_BENCHMARK}): "
                        f"{load_errors.get(MARKET_BENCHMARK, 'load failed')}")
        return {"computed": computed, "skipped": skipped,
                "input_coverage": coverage, "input_errors": load_errors,
                "benchmark_entry_cost_bps": benchmark_entry_cost_bps}

    for plan in plans:
        family_id, spec, effective = plan["family_id"], plan["spec"], plan["effective"]
        member_problems = [f"{symbol}: {load_errors[symbol]}"
                           for symbol in effective if symbol in load_errors]
        if member_problems:
            batch5._skip(skipped, family_id, "missing_data",
                        "; ".join(member_problems))
            continue
        signal_symbols = list(spec["signal"])
        support_problems = [f"{symbol}: {load_errors[symbol]}"
                            for symbol in signal_symbols if symbol in load_errors]
        if support_problems:
            batch5._skip(skipped, family_id, "missing_data",
                        "; ".join(support_problems))
            continue

        try:
            raw_asset_frames = {symbol: frames[symbol] for symbol in effective}
            asset_frames, dropped_by_symbol = _intersect_traded_calendars(raw_asset_frames)
            # make_target_fn's own precomputation (the expanding-median
            # threshold) intentionally sees the FULL, unaligned signal
            # source history -- only the frames handed to the ENGINE are
            # intersected, matching every rotation module's own documented
            # "ALL available history" interpretation choice.
            module_frames = {symbol: frames[symbol]
                             for symbol in effective + signal_symbols}
            target_fn = plan["module"].make_target_fn(module_frames)
            cost_bps_by_asset = {symbol: float(plan["costs"][symbol]["cost_bps"])
                                 for symbol in effective}
            signal_frames = ({symbol: frames[symbol] for symbol in signal_symbols}
                             if signal_symbols else None)
            curve = abi1_driver.build_rotation_curve(
                asset_frames, target_fn, cost_bps_by_asset=cost_bps_by_asset,
                signal_frames=signal_frames)
            account_type = plan["entry"]["account_types"][0]
            result = abi1_driver.evaluate_family(
                curve, benchmark_frame=benchmark_frame,
                benchmark_entry_cost_bps=benchmark_entry_cost_bps,
                account_type=account_type, tax_policy="realization")
        except Exception as exc:
            batch5._skip(skipped, family_id, "compute_error",
                        f"{type(exc).__name__}: {exc}")
            continue

        assumed_note = ("cost model: " + "; ".join(
            f"{symbol}={plan['costs'][symbol]['cost_bps']:.2f}bps "
            f"({plan['costs'][symbol]['source']})" for symbol in effective))
        if any(dropped_by_symbol.values()):
            assumed_note += ("; cross-vendor calendar alignment dropped "
                             + "; ".join(f"{symbol}={n}" for symbol, n in
                                        sorted(dropped_by_symbol.items()) if n)
                             + " sparse day(s) not common to every traded leg")

        account_index = result["account_daily"].index
        account_start = (account_index[0].strftime("%Y-%m-%d")
                         if len(account_index) else None)
        account_end = (account_index[-1].strftime("%Y-%m-%d")
                       if len(account_index) else None)

        if dry_run:
            verdict = result["verdict"]
            stage = (verdict if verdict in evidence.SCORED_STAGES
                     else "insufficient_evidence")
        else:
            try:
                stage = abi1_driver.record_abi1_outcome(
                    ledger_path, family_id, run_id, tested_date, result,
                    catalog_entry=plan["entry"], instrument_set=effective,
                    cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
                    tax_policy="realization", ticker_label=None,
                    extra_note=assumed_note,
                    signal_source_path=_SIGNAL_SOURCE_PATH,
                    signal_source_sha256=evidence.source_sha256(__file__))
            except Exception as exc:
                batch5._skip(skipped, family_id, "record_error",
                            f"{type(exc).__name__}: {exc}")
                continue

        layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
        computed.append({
            "family_id": family_id, "shape": spec["shape"], "tickers": effective,
            "verdict": result["verdict"], "stage": stage,
            "dsr_value": layer1.get("dsr_value"),
            "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
            "n_sessions": layer1.get("n_sessions"),
            "account_start": account_start, "account_end": account_end,
            "time_in_market": result.get("time_in_market"),
            "tax_policy": "realization", "cash_convention": CASH_CONVENTION,
            "benchmark": MARKET_BENCHMARK, "cost_model_note": assumed_note,
            "holdout_evaluated": holdout.get("evaluated"),
            "holdout_passed": holdout.get("passed"),
            "holdout_refusal": holdout.get("refusal"),
            "holdout_psr": holdout.get("psr"),
            "window": {
                "account_sessions": window.get("account_sessions"),
                "scored_sessions": window.get("scored_sessions"),
                "in_sample_sessions": window.get("in_sample_sessions"),
                "holdout_sessions": window.get("holdout_sessions"),
                "holdout_split_applied": window.get("holdout_split_applied"),
            },
        })

    return {"computed": computed, "skipped": skipped,
            "input_coverage": coverage, "input_errors": load_errors,
            "benchmark_entry_cost_bps": benchmark_entry_cost_bps}


# ------------------------------------------------------------------- run

def run_batch_cboe_rotation(scope: dict, *, tested_date: str, run_id: str,
                            ledger_path: str, dry_run: bool) -> dict:
    batch5.refuse_synthetic_quotes(scope["quotes"], dry_run=dry_run)
    result = compute_batch_cboe_rotation(
        scope, tested_date=tested_date, run_id=run_id,
        ledger_path=ledger_path, dry_run=dry_run)
    run_dir = scope["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": run_id,
        "tested_date": tested_date, "dry_run": dry_run,
        "driver": _SIGNAL_SOURCE_PATH,
        "results": [batch5._json_safe(item) for item in result["computed"]],
        "skipped": result["skipped"],
        "input_coverage": batch5._json_safe(result["input_coverage"]),
        "input_errors": result["input_errors"],
        "benchmark_entry_cost_bps": result.get("benchmark_entry_cost_bps"),
    }
    output_path = os.path.join(
        run_dir, "cboe-rotation-dry-run.json" if dry_run else "cboe-rotation-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    payload["output_path"] = output_path.replace(os.sep, "/")
    return payload


def _split_ids(raw) -> list:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--quotes", default=DEFAULT_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--families", default=None,
                        help="comma-separated subset of ASSUMED_COST_REGISTRY ids")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--modules-dir", default=DEFAULT_MODULES_DIR)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)
    run_dir = os.path.join(args.research_root, args.run_id)
    families = _split_ids(args.families) or list(ASSUMED_COST_REGISTRY)
    scope = batch5.build_scope(
        accepted_path=args.accepted, quotes_path=args.quotes,
        catalog_path=args.catalog, ledger_path=args.ledger,
        modules_dir=args.modules_dir, run_dir=run_dir, families=families)
    payload = run_batch_cboe_rotation(
        scope, tested_date=args.tested_date, run_id=args.run_id,
        ledger_path=args.ledger, dry_run=args.dry_run)
    print(json.dumps({
        "output": payload["output_path"],
        "tested": len(payload["results"]),
        "skipped": len(payload["skipped"]),
        "survivors": sum(item["stage"] == "survived" for item in payload["results"]),
        "dry_run": args.dry_run,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
