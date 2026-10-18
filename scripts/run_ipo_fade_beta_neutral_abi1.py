"""Bespoke ABI-1 driver for ipo_long_run_underperformance_fade_beta_neutral
-- the beta-neutral counterpart of the naked-short ipo_long_run_underperformance_fade.
Uses the same quotes / dry-run / refuse-synthetic / parameter-drift /
already-scored checks as scripts/run_ipo_fade_abi1.py.

WHY A BESPOKE DRIVER: the generic scripts/run_abi1_mining.py dispatch builds a
single-position curve from a per-bar signal function. This family has no
per-bar signal -- its account is a benchmark-plus-overlay curve (SPY core PLUS
a short-IPO / long-beta*SPY overlay) built by
``mining.batch5.ipo_long_run_underperformance_fade_beta_neutral.build_curve``
(math and conventions in ``mining.beta_neutral_overlay``) -- and that module's
``make_signal_fn`` always raises so the generic path cannot mis-score it. The
curve is fed to ``abi1_driver.evaluate_family`` exactly as any builder's dict
is.

WHAT THE REPORT ADDS (report-only, never gating): a ``beta_neutral_diagnostics``
block with the segment count and worst segment ``net_ret`` (the -100% floor a
single whole-history naked short can breach), the hedge-ratio range, the warm-up length, the OLS beta of
the SCORED excess series on SPY next to the naked short's (the confound this
design removes), and the range of the dollar-attribution ratio
``netted_tax_factors`` relies on (its entry-equity identity assumes the
account grows only by the trade; here the SPY core moves too).

NOT MODELLED (each of these FLATTERS the strategy; written into the ledger
note): short-borrow fees, margin financing of the net long, wash-sale
disallowance. Short-rebate interest is income, so omitting it is
conservative. The pipeline has no margin-rate or borrow-fee source; it does
carry a T-bill rate series, which is only a LOWER BOUND on financing cost and
was deliberately NOT charged (inventing a number is not
allowed) -- any apparent survivor needs a cost-sensitivity re-check before it
is believed.

Usage (network: yfinance historical OHLC only). A --dry-run proves the wiring
against a SYNTHETIC quotes file the caller constructs (its quotes must NOT be
reported as a cost measurement); a scoring run needs a real dated quotes
capture (``batch5.refuse_synthetic_quotes``) and writes ONE permanent ledger
row:
    python scripts/run_ipo_fade_beta_neutral_abi1.py \\
        --dry-run --quotes <synthetic-quotes.json> --research-root <scratch dir>

    python scripts/run_ipo_fade_beta_neutral_abi1.py \\
        --quotes <real-dated-capture.json> --run-id <run-id> --tested-date <YYYY-MM-DD>

The quotes file needs bid/ask for BOTH IPO and SPY (the overlay trades both).
A real run needs an explicit --tested-date and a FRESH --run-id (or an emptied
cache dir): OHLC histories cached under research-root/run-id by an earlier dry
run are re-used silently, and the result's *_ohlc_source / *_last_bar_date
fields say which happened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import (abi1_driver, benchmark_relative, catalog as catalog_mod,
                    evidence, overlay_risk_report, trial_ledger)
from mining import beta_neutral_overlay as overlay
from mining.batch5 import ipo_long_run_underperformance_fade_beta_neutral as mod

CASH_CONVENTION = "sweep_cash_daily"   # recorded; the account is always fully
                                       # deployed (SPY core), so idle cash is
                                       # never credited
MARKET_BENCHMARK = "SPY"
# "realization": overlay segments are short-term round trips whose net gains are
# taxed at exit with loss netting; see the module docstring for the disclosed
# dollar-attribution approximation (measured and reported below).
TAX_POLICY = "realization"

DEFAULT_RUN_ID = "ipo-fade-beta-neutral-abi1-v1"
# Dry-run placeholder for the tested date (today); a real run must pass --tested-date.
DEFAULT_TESTED_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

# This family's statement of how the overlay-only risk-basis rule relates to its ledger
# rows (mining.overlay_risk_report.risk_layer_note). A recorded ABI-1 row cannot be
# edited, and this driver refuses a second one (already_scored_abi1, dry runs included).
ORDERING_CLAUSE = (
    "a recorded ABI-1 outcome of this family is never edited, and every record this driver "
    "writes is scored on the overlay-only excess account")
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def overlay_mask(curve: dict) -> pd.Series:
    """Sessions on which the overlay is ON: from the first monthly rebalance entry
    onward (the overlay then rolls continuously), so every session before it -- the
    252-session beta warm-up -- is overlay-off and the account must equal SPY there.
    With no trades the overlay is never on."""
    index = curve["strategy_daily"].index
    trades = curve["trades_df"]
    if not len(trades):
        return pd.Series(False, index=index)
    return pd.Series(index >= pd.Timestamp(trades["entry_date"].min()), index=index)


def beta_neutral_diagnostics(curve: dict, result: dict, *, untaxed_result: dict,
                             ipo_frame, benchmark_daily, ipo_cost_bps: float,
                             spy_cost_bps: float,
                             benchmark_entry_cost_bps: float) -> dict:
    """Report-only structural details for the built curve (see module
    docstring). Nothing here gates a verdict."""
    trades = curve["trades_df"]
    x = float(mod.PARAMETERS["overlay_notional_fraction"])
    index = curve["strategy_daily"].index
    diag = {
        "overlay_notional_fraction": x,
        "beta_window_sessions": int(mod.PARAMETERS["beta_window_sessions"]),
        "ipo_cost_bps": float(ipo_cost_bps),
        "spy_cost_bps": float(spy_cost_bps),
        "benchmark_entry_cost_bps": float(benchmark_entry_cost_bps),
        "n_segments": int(len(trades)),
        "residual_beta_tolerance_per_notional": overlay.MAX_RESIDUAL_BETA_PER_NOTIONAL,
    }
    if len(trades):
        net = trades["net_ret"].to_numpy(dtype=float)
        hedge = trades["hedge_ratio"].to_numpy(dtype=float)
        first = int(index.get_loc(trades["entry_date"].iloc[0]))
        diag.update({
            "worst_segment_net_ret": _finite(net.min()),
            "best_segment_net_ret": _finite(net.max()),
            "segments_at_or_below_minus_100pct": int((net <= -1.0).sum()),
            "hedge_ratio_min": _finite(hedge.min()),
            "hedge_ratio_median": _finite(np.median(hedge)),
            "hedge_ratio_max": _finite(hedge.max()),
            "first_overlay_date": trades["entry_date"].iloc[0].strftime("%Y-%m-%d"),
            "warmup_sessions": first,
            "overlay_sessions": int(len(index) - first),
        })
        # Dollar-attribution ratio netted_tax_factors implicitly assumes is 1:
        # the account's growth over each segment (entry..exit inclusive) over
        # (1 + the overlay's own net return).
        growth = np.cumprod(1.0 + curve["strategy_daily"].to_numpy(dtype=float))
        entry_pos = index.get_indexer(pd.DatetimeIndex(trades["entry_date"]))
        exit_pos = index.get_indexer(pd.DatetimeIndex(trades["exit_date"]))
        before = np.where(entry_pos > 0, growth[np.maximum(entry_pos - 1, 0)], 1.0)
        ratio = (growth[exit_pos] / before) / (1.0 + net)
        diag["tax_identity_ratio_min"] = _finite(ratio.min())
        diag["tax_identity_ratio_max"] = _finite(ratio.max())
        # The identity's error is NOT uniformly conservative (gains over-taxed
        # and losses over-sheltered where the ratio exceeds 1, the reverse
        # below). Give the reader a scale for how much it could matter: the
        # whole tax effect on the scored daily-excess Sharpe (taxed minus
        # untaxed evaluation of the same curve) times the worst deviation of
        # the ratio from 1. A HEURISTIC SENSITIVITY, not a rigorous bound (the
        # loss carryforward makes the true effect nonlinear, and no
        # corrected-attribution curve is evaluated); reported, never gating.
        taxed = result["layer1"].get("sharpe_daily_excess")
        untaxed = untaxed_result["layer1"].get("sharpe_daily_excess")
        if taxed is not None and untaxed is not None:
            effect = float(taxed) - float(untaxed)
            worst = max(abs(diag["tax_identity_ratio_min"] - 1.0),
                        abs(diag["tax_identity_ratio_max"] - 1.0))
            diag.update({
                "sharpe_daily_excess_taxed": _finite(taxed),
                "sharpe_daily_excess_untaxed": _finite(untaxed),
                "tax_effect_on_daily_excess_sharpe": _finite(effect),
                "tax_identity_sensitivity_on_sharpe": _finite(worst * abs(effect)),
            })
    beta = benchmark_relative.ols_alpha_beta(result["log_excess"], benchmark_daily)
    naked = overlay.scored_excess_spy_beta(
        overlay.naked_short_account_daily(ipo_frame, cost_bps=ipo_cost_bps),
        benchmark_daily)
    diag.update({
        "scored_excess_beta_to_spy": _finite(beta["beta"]),
        "scored_excess_beta_per_notional": _finite(beta["beta"] / x),
        "beta_neutral_within_tolerance": bool(
            abs(beta["beta"]) / x < overlay.MAX_RESIDUAL_BETA_PER_NOTIONAL),
        "naked_short_excess_beta_to_spy": _finite(naked["beta"]),
    })
    # Risk-basis rule: the four refusals gate on the OVERLAY-ONLY EXCESS account;
    # the whole-account numbers ride along, report-only (mining.overlay_risk_report).
    diag["risk_layer"] = overlay_risk_report.risk_layer_report(result, benchmark_daily)
    return diag


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT,
                       ohlc_loader=None) -> dict:
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
    ipo_cost_bps = costs[traded_ticker]["cost_bps"]
    spy_cost_bps = costs[MARKET_BENCHMARK]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    loader = ohlc_loader or batch5.default_ohlc_loader
    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    ipo_frame, ipo_source = batch5.load_frame(
        traded_ticker, ohlc_loader=loader, cache_dir=cache_dir)
    spy_frame, spy_source = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=loader, cache_dir=cache_dir)

    curve = mod.build_curve(
        ipo_frame, spy_frame, ipo_cost_bps=ipo_cost_bps, spy_cost_bps=spy_cost_bps,
        spy_entry_cost_bps=benchmark_entry_cost_bps)

    account_type = entry["account_types"][0]
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)
    # Risk-basis rule: the ledger note states the risk basis; refuse to record
    # (or report) a score the basis was not applied to.
    benchmark_daily = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        spy_frame, entry_cost_bps=benchmark_entry_cost_bps)
    overlay_risk_report.require_overlay_basis(result, family_id=family_id)
    overlay_risk_report.require_excess_is_overlay_only(
        result["account_daily"], benchmark_daily, overlay_mask(curve), family_id=family_id)
    # report-only counterfactual for the tax-identity sensitivity (never scored)
    untaxed_result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy="none")

    diag = beta_neutral_diagnostics(
        curve, result, untaxed_result=untaxed_result, ipo_frame=ipo_frame,
        benchmark_daily=benchmark_daily,
        ipo_cost_bps=ipo_cost_bps, spy_cost_bps=spy_cost_bps,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps)

    extra_note = (
        f"cost model: IPO={ipo_cost_bps:.2f}bps SPY={spy_cost_bps:.2f}bps round trip "
        f"({batch5.quotes_cost_source(quotes_payload)}); design=beta-neutral "
        "benchmark-plus-overlay account (SPY core fully invested PLUS short IPO "
        f"x={diag['overlay_notional_fraction']} of equity and long beta_hat*x SPY; "
        f"beta_hat = trailing {diag['beta_window_sessions']}-session OLS slope from "
        "data ending the session BEFORE it is applied; re-estimated and rebalanced "
        "on the first session of each month; fills at that session's Open; fixed "
        "share counts between rebalances) -- see mining.beta_neutral_overlay; "
        "every monthly segment is charged a full round trip on BOTH overlay legs "
        "(over-counts turnover); NOT MODELLED, and each of these flatters the "
        "strategy: short-borrow fees, margin financing of the net long, "
        "wash-sale disallowance (short-rebate interest is income, so omitting "
        "it is conservative); the pipeline has no margin-rate or borrow-fee "
        "source -- a T-bill series exists but is only a lower bound on "
        "financing cost and was deliberately not charged "
        "-- so any apparent survivor needs a cost-sensitivity re-check "
        "before it is believed; LIMITATION: "
        "the hedge neutralizes SPY beta only -- residual style exposure (the "
        "IPO ETF's size/growth/momentum tilts versus the S&P 500) is NOT "
        "hedged, so an apparent edge could be a style premium rather than an "
        "IPO-specific effect; tax: "
        "netted_tax_factors dollar-attribution approximation (not uniformly "
        "conservative), measured ratio "
        f"range [{diag.get('tax_identity_ratio_min')}, {diag.get('tax_identity_ratio_max')}], "
        f"whole tax effect on the daily-excess Sharpe {diag.get('tax_effect_on_daily_excess_sharpe')}, "
        f"heuristic sensitivity (not a rigorous bound) {diag.get('tax_identity_sensitivity_on_sharpe')}; "
        f"measured: {diag['n_segments']} overlay segments, worst segment net_ret "
        f"{diag.get('worst_segment_net_ret')}, scored-excess OLS beta to SPY "
        f"{diag['scored_excess_beta_to_spy']} ({diag['scored_excess_beta_per_notional']} "
        "per unit of overlay notional; pre-registered tolerance "
        f"{diag['residual_beta_tolerance_per_notional']}) vs the naked-short "
        f"design's {diag['naked_short_excess_beta_to_spy']}; a DIFFERENT strategy "
        "definition from ipo_long_run_underperformance_fade (own family_id, own "
        "trial); "
        f"{overlay_risk_report.risk_layer_note(diag['risk_layer'], ordering_clause=ORDERING_CLAUSE)}; "
        f"price data: IPO {ipo_source} through {ipo_frame.index[-1]:%Y-%m-%d}, "
        f"SPY {spy_source} through {spy_frame.index[-1]:%Y-%m-%d}")

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result,
            catalog_entry=entry, instrument_set=list(mod.TRADED) + [MARKET_BENCHMARK],
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
        "n_trades": int(len(curve["trades_df"])),
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
        # provenance: "cache" means the history was re-used from an earlier
        # run under the SAME run_id (a real run must use a fresh run_id or an
        # emptied cache dir, or it silently re-uses stale dry-run bars)
        "ipo_ohlc_source": ipo_source, "spy_ohlc_source": spy_source,
        "ipo_last_bar_date": ipo_frame.index[-1].strftime("%Y-%m-%d"),
        "spy_last_bar_date": spy_frame.index[-1].strftime("%Y-%m-%d"),
        "beta_neutral_diagnostics": diag,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", required=True,
                        help="Path to a dated quotes JSON file with bid/ask for "
                             "IPO AND SPY (must be a REAL captured book unless "
                             "--dry-run is also passed, in which case a "
                             "SYNTHETIC placeholder file is expected).")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=None,
                        help="YYYY-MM-DD; REQUIRED for a real run (it is written "
                             "into a permanent ledger row); a --dry-run falls "
                             "back to a placeholder.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)
    if args.tested_date is None:
        if not args.dry_run:
            parser.error("--tested-date is required for a real (non --dry-run) "
                         "run: it is written into a permanent ledger row and "
                         "must never fall back to a stale default")
        args.tested_date = DEFAULT_TESTED_DATE

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
        "ipo-fade-beta-neutral-dry-run.json" if args.dry_run
        else "ipo-fade-beta-neutral-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
