"""Bespoke ABI-1 driver for ipo_long_run_underperformance_fade
(IPO short). The family uses only IPO's own real OHLC (no EDGAR
fundamentals, coverage file or fundamentals_map): a short-and-hold with no
per-bar signal decision (a constant entry=True/exit=False signal, run
under direction="short").

POINTER: both problems documented below (the -100% floor and
the SPY-beta confound) are addressed by the DIFFERENT strategy definition
``ipo_long_run_underperformance_fade_beta_neutral`` -- see
``scripts/run_ipo_fade_beta_neutral_abi1.py`` and ``mining.beta_neutral_overlay``.
This driver runs the naked-short primitive through ABI-1 and documents
both problems.

SHORT DIRECTION: this driver calls ``mining.backtest_engine.run_backtest``
and ``mining.benchmark_relative.mark_to_market_daily_returns`` with
``direction="short"`` and passes their output to
``abi1_driver.evaluate_family`` as a ``{"strategy_daily", "in_market",
"trades_df"}`` dict, the curve shape ``evaluate_family`` accepts from any
caller; ``evaluate_family`` turns it into the account curve it scores
against SPY. The family module's ``make_signal_fn`` raises unless the
caller passes ``driver_acknowledges_short_direction=True``, which this driver
and the module's own tests do, so a call from the generic
``scripts/run_abi1_mining.py`` dispatch raises instead of scoring a long
position.

A scoring run fails closed (a structured skip) when the catalog entry is
missing or ineligible, when the module's parameters drift from it, or when
an ABI-1 outcome is already recorded for the family. This driver reads no
curation-stage acceptance record: the catalog entry is this family's
registration.

A scoring run needs live quotes (it cannot be run with markets
closed), because a real run must price its verdict off a MEASURED bid/ask book
(``batch5.refuse_synthetic_quotes``) -- inventing one would record a
trial against a cost nobody observed. This script can be exercised
locally with ``--dry-run`` against a SYNTHETIC quotes file
the caller constructs (see ``--quotes``) -- that only proves the wiring
runs end to end; it is never a real quote-cost measurement and must
never be reported as one. A scoring run (a real quotes capture, no
``--dry-run``) must be run during market hours.

Usage (real network calls -- yfinance historical OHLC only; no live
quotes needed for a --dry-run wiring check against a synthetic quotes
file):
    python scripts/run_ipo_fade_abi1.py \\
        --dry-run --quotes <path-to-a-synthetic-quotes-file.json>

Scoring run (market hours only, a real dated quotes capture, no
--dry-run):
    python scripts/run_ipo_fade_abi1.py \\
        --quotes <path-to-a-real-dated-quotes-capture.json> \\
        --run-id <run-id> --tested-date <YYYY-MM-DD>

KNOWN LIMITATION, THE -100% FLOOR (read this before attempting a
scoring run): a run (a --dry-run against a SYNTHETIC placeholder quotes
file included) passes through fetch ->
signal -> `run_backtest(..., direction="short")` -> `mark_to_market_
daily_returns(..., direction="short")` -> curve assembly, then
can raise INSIDE the shared ``mining.account_relative.netted_tax_factors``: "a net_ret at or
below -100% leaves the trade's entry equity undefined ... the account is
wiped out and cannot be taxed". This is NOT a defect in this driver, in
``ipo_long_run_underperformance_fade.py``, or in the engine primitives it
calls directly -- it is a real, structural consequence of this family's
own DISCLOSED SIMPLIFICATION (see the family module's docstring, "DISCLOSED
SIMPLIFICATION" section): collapsing the catalog's literal ~2-year
rolling short into ONE continuous short-and-hold-forever position means
the single resulting trade spans the instrument's entire fetched history,
and if its price more than doubles over that span, the engine's simple
(uncompounded) one-share short P&L convention produces a net_ret BELOW
the -100% floor
``netted_tax_factors`` requires (it fails closed on this REGARDLESS of
account_type / tax rate, including a rate of 0.0 -- the check runs
before the rate is even consulted). Other single-instrument SHORT families
in this pipeline avoid this either by re-evaluating their position
periodically within a bounded window (retail_inventory_sales_breadth_short's
panel_breadth: quarterly re-evaluation, so no one trade spans the whole
history) or by holding a real inverse product LONG (bounded at
-100% like any long position) -- this family, as reduced, combines
"short direction" with "buy-and-hold-forever", and that combination is
what can break.

FAILURE HANDLING: every other family in this pipeline turns a failed
compute into a structured skip --
scripts/run_abi1_mining.py's own generic dispatch wraps exactly this
shape of failure in a broad except-Exception and returns a clean,
structured skip (reason_code="compute_error") instead of letting it
propagate uncaught. ``compute_and_record`` below catches ONLY this
one, precisely-identified ValueError (matched on its exact message
substring, "wiped out and cannot be taxed" -- any OTHER ValueError still
propagates uncaught, on purpose) and returns the same
reason_code="compute_error" shape. The catch makes the FAILURE MODE a
clean skip instead of an uncaught crash; it does not remove the -100%
floor itself. A periodic reset ALONE (matching the catalog's own literal
`holding_period_years: 2` parameter) is NOT guaranteed to avoid this: any
single two-year segment in which the price more than doubles (as can
happen in an IPO boom) would STILL breach the -100% floor within that
bounded segment. Avoiding it needs an explicit risk bound (e.g. a
stop-loss threshold); the catalog's PARAMETERS specify none, and this
driver adds none. CONSEQUENCE: when the floor is breached, a
(non-dry-run) scoring attempt returns a clean
`reason_code="compute_error"` skip, not an ABI-1 outcome.

SECOND, DEEPER PROBLEM (see the family module's own docstring for the full
account): even setting the
-100%-floor problem aside entirely, this driver's construction -- a
naked, unhedged short of IPO alone, scored via ABI-1's own SPY-excess
methodology -- does not actually isolate the catalog's real economic
claim (IPO underperforms SPY on a RELATIVE basis). ABI-1 scores an
account's daily excess return over SPY; for a naked short of a single
market-correlated instrument, that measure is dominated by the account's
own NEGATIVE overall equity-market beta exposure, not by IPO's own
relative underperformance specifically. A test of Ritter (1991)'s
relative finding needs a beta-neutral construction (e.g. simultaneously
long SPY and short IPO), which is the separate family
``ipo_long_run_underperformance_fade_beta_neutral`` (see POINTER above).
Under either problem, a score from this driver measures the naked-short
primitive, not the relative-underperformance claim.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
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
from mining import abi1_driver, catalog as catalog_mod, evidence, trial_ledger
from mining.backtest_engine import run_backtest
from mining.batch5 import ipo_long_run_underperformance_fade as mod
from mining.benchmark_relative import mark_to_market_daily_returns

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
TAX_POLICY = "realization"  # single_inverse_product is a SINGLE_SHAPE (abi1_
                            # driver's own module docstring): 100% of capital
                            # in one position at a time, so netted_tax_
                            # factors' dollar-gain identity holds -- the
                            # single-instrument bespoke drivers' shared
                            # convention.

DEFAULT_RUN_ID = "ipo-fade-abi1-v1"
DEFAULT_TESTED_DATE = date.today().isoformat()
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT
                       ) -> dict:
    family_id = mod.FAMILY_ID

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
    traded_ticker = mod.TRADED[0]
    costs = batch5.execution_costs_for([traded_ticker, MARKET_BENCHMARK], quotes_payload)
    cost_bps = costs[traded_ticker]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    ipo_frame, ipo_source = batch5.load_frame(
        traded_ticker, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    spy_frame, _spy_source = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)

    signal_fn = mod.make_signal_fn(
        traded_ticker, ipo_frame.index, {traded_ticker: ipo_frame},
        driver_acknowledges_short_direction=True)

    # Short direction -- see this module's docstring, "SHORT DIRECTION":
    # run_backtest and mark_to_market_daily_returns are called with
    # direction="short", and their output forms the {"strategy_daily",
    # "in_market", "trades_df"} curve dict evaluate_family takes.
    trades_df, in_market = run_backtest(
        ipo_frame, signal_fn, cost_bps=cost_bps, tax_haircut_rate=0.0,
        direction="short")
    strategy_daily = mark_to_market_daily_returns(
        ipo_frame, trades_df, cost_bps=cost_bps, direction="short")
    curve = {"strategy_daily": strategy_daily, "in_market": in_market,
             "trades_df": trades_df}

    account_type = entry["account_types"][0]
    # KNOWN, DISCLOSED LIMITATION (see module docstring): this family's own
    # DISCLOSED SIMPLIFICATION (one continuous short trade for the full
    # fetched history) can produce a real net_ret below -100%, which makes
    # account_relative.netted_tax_factors (called inside evaluate_family)
    # fail closed regardless of account_type/tax rate. Every other family
    # in this pipeline turns a failed compute into a structured skip --
    # scripts/run_abi1_mining.py's own generic dispatch
    # wraps this exact kind of failure in a broad except-Exception and
    # returns a clean, structured skip (reason_code="compute_error")
    # rather than letting it propagate uncaught. This driver follows
    # that convention for this ONE precisely-identified, disclosed
    # failure mode -- it does NOT swallow any OTHER exception, and it adds
    # no risk bound (a stop-loss/reset threshold), since the catalog's
    # PARAMETERS specify none (see module docstring).
    _WIPED_OUT_MARKER = "wiped out and cannot be taxed"
    try:
        result = abi1_driver.evaluate_family(
            curve, benchmark_frame=spy_frame,
            benchmark_entry_cost_bps=benchmark_entry_cost_bps,
            account_type=account_type, tax_policy=TAX_POLICY)
    except ValueError as exc:
        if _WIPED_OUT_MARKER not in str(exc):
            raise  # fail loud on anything NOT this one precisely-identified case
        return {
            "family_id": family_id, "reason_code": "compute_error",
            "reason": (
                f"ValueError: {exc}; this family's own DISCLOSED "
                "SIMPLIFICATION (one continuous whole-history short) "
                "produced a net_ret below the engine's -100% floor -- see "
                "the module docstring. The catalog's PARAMETERS define no "
                "risk bound (e.g. a stop-loss; a plain periodic reset "
                "matching the catalog's own holding_period_years=2 is not "
                "by itself sufficient, since one bounded segment can still "
                "more than double), and this driver adds none."),
        }

    extra_note = (
        f"cost model: {traded_ticker}={cost_bps:.2f}bps "
        f"({batch5.quotes_cost_source(quotes_payload)}); "
        "direction=short (run_backtest/mark_to_market_daily_returns called "
        "with direction=\"short\"; see module docstring); "
        "no EDGAR fundamentals, no coverage file, no fundamentals_map -- "
        "this family is pure buy-and-hold-short OHLC; "
        "DISCLOSED CAVEAT (see module docstring): a naive un-hedged "
        "short of IPO's own absolute price can lose even when IPO "
        "underperforms SPY over the same window; "
        "METHODOLOGY LIMIT (see module docstring's "
        "'SECOND, DEEPER PROBLEM' section): ABI-1's own SPY-excess "
        "scoring is NOT the right measure for the relative question -- "
        "a naked short's SPY-excess is dominated by the "
        "account's own negative market-beta exposure, not IPO's relative "
        "underperformance specifically; a verdict recorded with this note "
        "does not answer the catalog's relative question -- the beta-neutral "
        "(long SPY + short IPO) construction is the separate family "
        "ipo_long_run_underperformance_fade_beta_neutral")

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
    account_index = result["account_daily"].index
    return {
        "family_id": family_id, "shape": mod.SHAPE, "tickers": list(mod.TRADED),
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "n_trades": int(len(trades_df)),
        "account_start": account_index[0].strftime("%Y-%m-%d") if len(account_index) else None,
        "account_end": account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None,
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
        "ipo_ohlc_source": ipo_source,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", required=True,
                        help="Path to a dated quotes JSON file (must be a "
                             "REAL captured book unless --dry-run is also "
                             "passed, in which case a SYNTHETIC placeholder "
                             "file is expected and required -- see module "
                             "docstring).")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes,
        ledger_path=args.ledger, run_id=args.run_id, tested_date=args.tested_date,
        dry_run=args.dry_run, signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__),
        research_root=args.research_root)

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
        "ipo-fade-dry-run.json" if args.dry_run else "ipo-fade-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
