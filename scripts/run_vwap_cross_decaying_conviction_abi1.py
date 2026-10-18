"""Bespoke ABI-1 scoring driver for vwap_cross_decaying_conviction_exit_
daytrade -- a family in this pipeline's INTRADAY (minute-bar) day-trading
track scored on the ABI-1 account-relative path. The bridge from that
track into ABI-1 is ``mining.day_trading_abi1_bridge`` (imported below as
``bridge``), a small, dedicated, REUSABLE module, not a one-off embedded in
this script.

## Why this needs a bespoke driver, not the generic discover_modules path

``scripts/run_abi1_mining.py`` (the ABI-1 scoring entry point) discovers
strategy modules from ``src/mining/batch5/*.py`` via
``scripts/run_batch5_mining.py``'s ``discover_modules``/
``validate_module_contract``, whose default OHLC loader
(``run_batch5_mining.default_ohlc_loader`` -> ``data_pipeline.fetch_ohlc``)
only ever fetches DAILY bars -- there is no intraday-timeframe path
anywhere in that generic discovery/loading mechanism.

``day_trading_signals.SIGNAL_FACTORIES`` feeds the per-trade
correlation-batch pipeline (``run_day_trading_mining.
execute_day_trading_run`` -> ``trial_ledger.record_first_test``/
``score_batch``) -- a DIFFERENT scoring mechanism from ABI-1's
full-history account-relative one. Registering this family there would
therefore not route it into ABI-1.

This driver passes this family's ``signal_fn`` (written for
``intraday_backtest_engine.py``, the intraday engine; this driver does not
modify it) to ``abi1_driver.evaluate_family`` through a
``{"strategy_daily", "in_market", "trades_df"}`` dict, the curve shape
``evaluate_family`` accepts from any caller. The dict is built by
``mining.day_trading_abi1_bridge.build_day_trading_curve``, a shared module,
so a bespoke driver for any intraday-timeframe family whose signal_fn
follows ``intraday_backtest_engine.py``'s calling convention can use the
same bridge.

## Intraday-data range (vendor coverage)

The intraday vendor's minute-bar coverage for this ticker begins on the
first trading day of 2016: single-day ``intraday_data.fetch_intraday_bars``
calls for earlier dates raise ``IntradayDataGapError`` (a vendor coverage
gap, not a code defect), while 2016-01-04 and later sessions return a full
390-bar RTH session. ``REAL_DATA_START`` below is pinned to that boundary,
not a round-number guess.

At the NEWER end, a same-day fetch can be refused by the vendor (an
access-tier/entitlement restriction on same-day data, not a coverage
gap). ``_most_recent_available_session`` below therefore defaults every
fetch's end date to the most recent NYSE session STRICTLY BEFORE the day
it runs, sidestepping that restriction without asserting anything about
its cause.

Net: this family's intraday history is roughly a DECADE, not the
multi-decade daily-bar corpus this pipeline's other families are scored
against -- a disclosed statistical-power caveat (fewer independent
sessions than a typical daily-bar family).

## Dry run vs real score

The driver runs a backtest over historical QQQ minute bars through the
existing, unmodified
``make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn`` via
``intraday_backtest_engine.run_intraday_backtest``, matching ``scripts/
run_day_trading_mining.py``'s own ``_backtest_tickers`` call pattern
exactly (``cost_bps``/``tax_haircut_rate``/``max_hold_days``/``direction``
kwargs). The pipeline's convention (``scripts/run_abi1_mining.py``'s own
re-run guard) is that a family gets exactly ONE real scoring attempt, with
no retry, so a real scoring run needs a regular-session QQQ+SPY quote
capture. ``--dry-run`` plus a quotes file carrying the literal
``SYNTHETIC`` marker (refused outside ``--dry-run`` by
``run_batch5_mining.refuse_synthetic_quotes``, the SAME guard every
sibling driver relies on) exercises the full path against real price data
without spending that one real attempt.

Imports but never modifies ``day_trading_vwap_decaying_conviction.py``,
``intraday_backtest_engine.py``, ``intraday_data.py``, or ``abi1_driver.py``
-- it only consumes their outputs (and the bridge module's).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

import pandas as pd
import pandas_market_calendars as mcal

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import abi1_driver, catalog as catalog_mod, evidence, intraday_data, trial_ledger
from mining import day_trading_abi1_bridge as bridge
from mining import day_trading_vwap_decaying_conviction as mod
from mining.intraday_backtest_engine import run_intraday_backtest

FAMILY_ID = "vwap_cross_decaying_conviction_exit_daytrade"
TRADED_TICKER = "QQQ"
TRADED = (TRADED_TICKER,)
SHAPE = "single_long"

# Byte-for-byte the catalog's own "parameters" field (data/mining/catalog.json)
# -- checked below against the catalog entry (parameter_drift) AND, via the
# assert further down, against the signal_fn factory's own live defaults, so
# a change to either surfaces as a loud failure rather than a silent
# mismatch. There is no src/mining/batch5/*.py module for this family (it
# was deliberately built standalone, see day_trading_vwap_decaying_conviction.py's
# own module docstring), so -- same as scripts/run_sector_dispersion_abi1.py's
# own precedent for the same situation -- this driver owns these constants
# locally rather than reading them off a module that doesn't define them.
PARAMETERS = {
    "vwap_confirmation_bars": 2,
    "session_start_et": "09:30",
    "force_flat_et": "15:55",
    "max_hold_days": 1,
    "base_target_pct": 0.01,
    "base_stop_pct": 0.005,
    "floor_fraction": 0.25,
    "half_life_minutes": 30,
}
_factory_defaults = {
    name: param.default
    for name, param in inspect.signature(
        mod.make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn
    ).parameters.items()
}
assert PARAMETERS == _factory_defaults, (
    "vwap_cross_decaying_conviction_exit_daytrade driver's local PARAMETERS "
    "no longer match make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn's "
    "own live defaults -- re-check before scoring")

# The vendor's minute-bar coverage boundary -- see the module docstring's
# "Intraday-data range" section.
REAL_DATA_START = "2016-01-04"

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
ACCOUNT_TYPE = "taxable"
TAX_POLICY = "realization"  # single_long shape: 100% of capital in one
                            # position at a time, matching every other
                            # non-pooled bespoke driver's own convention.

DEFAULT_RUN_ID = "vwap-cross-decaying-conviction-abi1"
DEFAULT_TESTED_DATE = None  # None -> today's local date at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
# A real run needs a fresh quote capture that includes QQQ and SPY, saved to
# this driver's own run_id directory.
DEFAULT_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")

_NYSE_CALENDAR = mcal.get_calendar("NYSE")


def _most_recent_available_session(before_date: str) -> str:
    """The most recent real NYSE trading session STRICTLY BEFORE
    `before_date` (YYYY-MM-DD) -- this driver's own default `end_date` for
    the full-history intraday fetch. See module docstring's "Intraday-data
    range" section for why same-day data is avoided (a same-day fetch can
    be refused under the vendor's access tier)."""
    before = pd.Timestamp(before_date).date()
    schedule = _NYSE_CALENDAR.schedule(
        start_date=(pd.Timestamp(before_date) - pd.Timedelta(days=14)).date(),
        end_date=before_date)
    sessions = [ts.date() for ts in schedule.index if ts.date() < before]
    if not sessions:
        raise ValueError(
            f"_most_recent_available_session: no real NYSE session found "
            f"in the 14 days before {before_date}")
    return sessions[-1].isoformat()


def fetch_full_history_bars(*, end_date: str | None = None) -> tuple:
    """Fetch the FULL real available QQQ 1-minute RTH-bar history via
    `intraday_data.fetch_intraday_bars`, from `REAL_DATA_START` (the
    vendor's first covered trading day, see module docstring) through
    `end_date` (default: the most recent real session strictly before
    today -- see `_most_recent_available_session`). `end_date` is a
    dry-run-only override (`compute_and_record` refuses it outside
    `--dry-run`). Returns `(bars, resolved_end_date)`."""
    resolved_end = end_date or _most_recent_available_session(
        pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"))
    bars = intraday_data.fetch_intraday_bars(TRADED_TICKER, REAL_DATA_START, resolved_end)
    return bars, resolved_end


def run_real_intraday_backtest(bars: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Runs the EXISTING, UNMODIFIED
    `make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn` through
    `intraday_backtest_engine.run_intraday_backtest` over `bars`, matching
    `scripts/run_day_trading_mining.py`'s own `_backtest_tickers` call
    pattern exactly (`cost_bps`/`max_hold_days`/`direction`). A FRESH
    `signal_fn` instance is built here on every call -- its closure carries
    real per-session state (VWAP accumulators, a confirmation streak, an
    open position) that must start clean, never be reused across a second
    backtest run.

    `tax_haircut_rate` is ALWAYS `0.0` here, never a parameter this
    function accepts -- deliberately, not an oversight. `mining.
    abi1_driver`'s own module docstring is explicit about why:
    `build_single_asset_curve`/`build_rotation_curve` run their underlying
    engine with `tax_haircut_rate=0.0` because ABI-1's own tax model
    (`account_relative.evaluate(tax_policy="realization", ...)` ->
    `netted_tax_factors`) applies tax itself, working from the PRE-tax
    `trades_df["net_ret"]` -- passing a nonzero `tax_haircut_rate` into the
    ENGINE here would double-tax every winning trade (charged once inside
    `run_intraday_backtest`'s own `_close_trade`, then AGAIN by
    `netted_tax_factors` on the already-haircut value). Passing
    `cost_tax_inputs.tax_haircut_rate_for(ACCOUNT_TYPE)` straight through
    here would do exactly that, so the rate is not a parameter at all
    rather than trusting every caller to remember to pass `0.0`.
    `ACCOUNT_TYPE` still feeds tax correctly, exactly once, via
    `evaluate_family`'s own `account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY`
    arguments below.

    Returns only `trades_df` -- this driver deliberately does NOT reuse
    the engine's own second return value (`daily_returns`, its internal
    per-day mark-to-market decomposition) as this family's ABI-1
    `strategy_daily`: `mining.day_trading_abi1_bridge.build_day_trading_curve`
    builds that directly from `trades_df` instead (see that module's own
    docstring for why -- a small, independently-testable, and more
    generally reusable aggregation, not a re-derivation of the same
    number)."""
    signal_fn = mod.make_vwap_cross_decaying_conviction_exit_daytrade_signal_fn(**PARAMETERS)
    trades_df, _engine_daily_returns = run_intraday_backtest(
        bars, signal_fn, cost_bps=cost_bps, tax_haircut_rate=0.0,
        max_hold_days=PARAMETERS["max_hold_days"], direction="long")
    return trades_df


def _registration_drift(entry: dict) -> str | None:
    """None when this driver still matches the catalog entry it was
    registered against; otherwise the reason. `batch5.parameter_drift`
    compares only the `parameters` dict, so the catalog's `tickers` (the
    traded instruments) is checked here too. Ticker order is not
    significant."""
    drift = batch5.parameter_drift(PARAMETERS, entry.get("parameters"))
    if drift is not None:
        return drift
    catalog_tickers = sorted(entry.get("tickers") or [])
    if catalog_tickers != sorted(TRADED):
        return (f"driver traded tickers {sorted(TRADED)} != catalog tickers "
                f"{catalog_tickers}")
    return None


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       end_date: str | None, signal_source_path: str,
                       signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT) -> dict:
    if end_date is not None and not dry_run:
        raise ValueError(
            "compute_and_record: end_date override is dry-run-only -- a "
            "real (non-dry-run) scoring attempt must always run through the "
            "most recent real session, never a truncated window that would "
            "become this family's permanent one-shot outcome")
    family_id = FAMILY_ID
    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    entry = catalog_entries.get(family_id)
    if entry is None:
        return {"family_id": family_id, "reason_code": "not_in_catalog",
                "reason": f"{family_id} is not in the catalog"}
    if entry.get("status") != "eligible":
        return {"family_id": family_id, "reason_code": "catalog_status",
                "reason": f"catalog status is {entry.get('status')!r}, not eligible"}
    drift = _registration_drift(entry)
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
    costs = batch5.execution_costs_for([TRADED_TICKER, MARKET_BENCHMARK], quotes_payload)
    cost_bps = costs[TRADED_TICKER]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    bars, resolved_end = fetch_full_history_bars(end_date=end_date)

    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    spy_frame, _spy_source = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)

    # tax_haircut_rate is deliberately NOT passed here -- see
    # run_real_intraday_backtest's own docstring: the engine always runs
    # untaxed (0.0), and ACCOUNT_TYPE's real tax rate is applied exactly
    # once, correctly, by evaluate_family below (tax_policy="realization").
    trades_df = run_real_intraday_backtest(bars, cost_bps=cost_bps)

    calendar_index = pd.DatetimeIndex(sorted(set(bars.index.date)))
    curve = bridge.build_day_trading_curve(trades_df, calendar_index)

    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY)

    extra_note = (
        f"cost model: {TRADED_TICKER}={cost_bps:.2f}bps "
        f"({batch5.quotes_cost_source(quotes_payload)}); "
        f"real QQQ intraday history {REAL_DATA_START}..{resolved_end} "
        "(Alpaca SIP 1-minute RTH-only bars, real market data -- see driver "
        "module docstring's intraday-data range section); daily curve "
        "assembled via mining.day_trading_abi1_bridge."
        "build_day_trading_curve")

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        # A concurrent invocation can pass the already_scored prefilter
        # above (unlocked, cheap, run before this expensive compute) and
        # still lose the atomic check-and-append inside record_abi1_outcome
        # -- trial_ledger.OutcomeAlreadyRecorded means this attempt's own
        # freshly computed result must be discarded, same as the early
        # check's own already_scored_abi1 skip.
        try:
            stage = abi1_driver.record_abi1_outcome(
                ledger_path, family_id, run_id, tested_date, result,
                catalog_entry=entry, instrument_set=list(TRADED),
                cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
                tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
                signal_source_path=signal_source_path,
                signal_source_sha256=signal_source_sha256)
        except trial_ledger.OutcomeAlreadyRecorded:
            return {"family_id": family_id, "reason_code": "already_scored_abi1",
                    "reason": "a concurrent run recorded a construction=\"ABI-1\" "
                              "outcome first -- this attempt's freshly computed "
                              "result is discarded"}

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    account_index = result["account_daily"].index
    return {
        "family_id": family_id, "shape": SHAPE, "tickers": list(TRADED),
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
        "real_intraday_data_start": REAL_DATA_START,
        "real_intraday_data_end": resolved_end,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", default=DEFAULT_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument(
        "--end-date", default=None,
        help="DRY-RUN-ONLY override for the intraday fetch's end date "
             "(YYYY-MM-DD). Default is the most recent real NYSE session "
             "strictly before today. Rejected outside --dry-run "
             "(compute_and_record raises).")
    args = parser.parse_args(argv)
    if args.tested_date is None:
        args.tested_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes,
        ledger_path=args.ledger, run_id=args.run_id, tested_date=args.tested_date,
        dry_run=args.dry_run, end_date=args.end_date,
        signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__),
        research_root=args.research_root)

    safe_result = batch5._json_safe(result)
    if "reason_code" in result:
        # A skip dict (not_in_catalog / catalog_status / parameter_drift /
        # already_scored_abi1) carries no scoring output. Writing it would
        # overwrite a prior real run's results file with a reason-only
        # payload, so it is printed only.
        print(json.dumps({"output": None, "result": safe_result}))
        return 0

    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": _SIGNAL_SOURCE_PATH, "result": safe_result,
    }
    output_path = os.path.join(
        run_dir,
        "vwap-cross-decaying-conviction-dry-run.json" if args.dry_run
        else "vwap-cross-decaying-conviction-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
