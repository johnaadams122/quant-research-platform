"""Bespoke ABI-1 driver for cross_asset_divergence_fvg_continuation_
daytrade -- a family in this pipeline's intraday (minute-bar) day-trading
track, scored on the ABI-1 account-relative path. See
``scripts/run_vwap_cross_decaying_conviction_abi1.py``'s module docstring
for why the generic ``scripts/run_abi1_mining.py`` dispatch cannot reach
any intraday-timeframe family at all (the same reason applies here).

## Why this needs a bespoke driver

``mining.cascade_signals.make_cross_asset_divergence_fvg_cascade_signal_fn``
is NOT a standard three-positional-argument ``make_signal_fn(ticker, index,
frames)`` factory -- its first argument is ``secondary_bars``, a full
pre-fetched SPY minute-bar OHLCV frame the cascade's divergence gate reads
at every hourly-confirmation check. The generic dispatch has no path to
supply that, and even if it did, only QQQ (never SPY) is TRADED by this
family -- SPY is read-only reference data, the identical shape the
retail-inventory-breadth driver's own ``direction="short"`` passthrough and
the accrual-anomaly driver's own ``fundamentals_map`` keyword already
established a bespoke-driver precedent for (see either driver's own module
docstring).

## Real observed intraday-data range (measured, not assumed)

QQQ's real Alpaca SIP 1-minute coverage starts 2016-01-04 (measured for
``vwap_cross_decaying_conviction_exit_daytrade``'s driver). This
family also needs SPY over the identical window (secondary reference,
every hourly bucket). A live single-day probe (
``mining.intraday_data.fetch_intraday_bars`` for 2016-01-04) confirmed SPY
returns a full real 390-bar RTH session on that exact date too -- the same
vendor coverage floor, not assumed to carry over from QQQ's own measurement.
``REAL_DATA_START`` below is pinned to that measured, shared floor.

## No look-ahead in the divergence gate

``secondary_bars`` is passed as this driver's own FULL real SPY history
fetched up front -- correctness against look-ahead is NOT this driver's
own job to prove (already proven inside ``cascade_signals.
make_cross_asset_divergence_fvg_cascade_signal_fn``'s own test suite,
``src/tests/test_cascade_signals.py``, via its own ``divergence_check``
closure, which slices ``secondary_bars`` to ``current_ts`` before ever
reading it). This driver only fetches and threads the frame through.

## Single-shot real-scoring discipline

Matches every sibling bespoke driver's own established convention: a
family gets exactly ONE real ABI-1 scoring attempt, ever (enforced by
``already_scored_abi1`` in the driver logic below, mirroring
``scripts/run_abi1_mining.py``'s own re-run guard). ``--dry-run`` plus a
quotes file carrying the literal ``SYNTHETIC`` marker (refused outside
``--dry-run`` by ``run_batch5_mining.refuse_synthetic_quotes``) proves this
driver correct against real fetched data without spending that attempt.
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
from mining import cascade_signals as mod
from mining import day_trading_abi1_bridge as bridge
from mining.intraday_backtest_engine import run_intraday_backtest

FAMILY_ID = "cross_asset_divergence_fvg_continuation_daytrade"
TRADED_TICKER = "QQQ"
SECONDARY_TICKER = "SPY"
TRADED = (TRADED_TICKER,)
SHAPE = "single_long"

# Byte-for-byte the catalog's own "parameters" field (data/mining/catalog.json)
# -- checked below against the catalog entry (parameter_drift), same
# convention as every sibling bespoke driver. Only two of these four keys
# are the signal FACTORY's own live parameters (asserted separately below,
# against inspect.signature) -- max_hold_days/direction describe the
# family's engine-level dispatch, not the cascade's own signal logic
# (mirrors cascade_signals.make_fvg_reversal_cascade_signal_fn's identical
# 2-parameter factory shape for the sibling family).
PARAMETERS = {
    "secondary_ticker": SECONDARY_TICKER,
    "min_target_r_multiple": 2.0,
    "force_flat_et": "15:55",
    "max_hold_days": 1,
    "direction": "long",
}
_factory_defaults = {
    name: param.default
    for name, param in inspect.signature(
        mod.make_cross_asset_divergence_fvg_cascade_signal_fn
    ).parameters.items()
    if name != "secondary_bars"
}
assert _factory_defaults == {
    "min_target_r_multiple": PARAMETERS["min_target_r_multiple"],
    "force_flat_et": PARAMETERS["force_flat_et"],
}, (
    "cross_asset_divergence_fvg_continuation_daytrade driver's local "
    "PARAMETERS no longer match make_cross_asset_divergence_fvg_cascade_"
    "signal_fn's own live defaults -- re-check before scoring")

# Empirically measured, not assumed -- see module
# docstring's "Real observed intraday-data range" section.
REAL_DATA_START = "2016-01-04"

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
ACCOUNT_TYPE = "taxable"
TAX_POLICY = "realization"  # single_long shape, matches every other
                            # non-pooled bespoke driver's own convention.

DEFAULT_RUN_ID = "cross-asset-divergence-fvg-abi1-v1"
# The date a run is recorded as tested: today, unless --tested-date is passed.
DEFAULT_TESTED_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")

_NYSE_CALENDAR = mcal.get_calendar("NYSE")


def _most_recent_available_session(before_date: str) -> str:
    """The most recent real NYSE trading session STRICTLY BEFORE
    `before_date` (YYYY-MM-DD) -- mirrors run_vwap_cross_decaying_
    conviction_abi1.py's own identically-named helper (not shared/
    imported -- that driver's own copy is the established precedent for
    this NOT being extracted into a shared module yet)."""
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


def fetch_full_history_bars(ticker: str, *, start_date: str | None = None,
                            end_date: str | None = None) -> tuple:
    """Fetch the FULL real available 1-minute RTH-bar history for `ticker`
    via `intraday_data.fetch_intraday_bars`, from `start_date` (default:
    `REAL_DATA_START`) through `end_date` (default: the most recent real
    session strictly before today). Returns `(bars, resolved_end_date)`.
    `start_date` and `end_date` are dry-run-only overrides (see
    `--start-date`/`--end-date`; `compute_and_record` refuses both outside
    `--dry-run`) -- a REAL (non-dry-run) invocation always omits them, so
    the real scored window always starts at the measured `REAL_DATA_START`
    floor and runs through the most recent real session."""
    resolved_end = end_date or _most_recent_available_session(
        pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"))
    bars = intraday_data.fetch_intraday_bars(
        ticker, start_date or REAL_DATA_START, resolved_end)
    return bars, resolved_end


def run_real_intraday_backtest(qqq_bars: pd.DataFrame, spy_bars: pd.DataFrame,
                               *, cost_bps: float) -> pd.DataFrame:
    """Runs the EXISTING, UNMODIFIED
    `make_cross_asset_divergence_fvg_cascade_signal_fn` through
    `intraday_backtest_engine.run_intraday_backtest` over `qqq_bars` (the
    only TRADED instrument), with `spy_bars` threaded through as the
    signal's own read-only `secondary_bars` reference. A FRESH `signal_fn`
    instance is built here on every call -- its closure carries real
    per-day state that must start clean.

    `tax_haircut_rate` is ALWAYS `0.0` here -- see
    `run_vwap_cross_decaying_conviction_abi1.run_real_intraday_backtest`'s
    own docstring for why (ABI-1's own tax model applies tax exactly once,
    via `evaluate_family`'s `tax_policy=TAX_POLICY`; passing a nonzero rate
    into the engine here would double-tax every winning trade)."""
    signal_fn = mod.make_cross_asset_divergence_fvg_cascade_signal_fn(
        secondary_bars=spy_bars,
        min_target_r_multiple=PARAMETERS["min_target_r_multiple"],
        force_flat_et=PARAMETERS["force_flat_et"])
    trades_df, _engine_daily_returns = run_intraday_backtest(
        qqq_bars, signal_fn, cost_bps=cost_bps, tax_haircut_rate=0.0,
        max_hold_days=PARAMETERS["max_hold_days"], direction="long")
    return trades_df


def _registration_drift(entry: dict) -> str | None:
    """None when this driver still matches the catalog entry it was
    registered against; otherwise the reason. `batch5.parameter_drift`
    compares only the `parameters` dict, so the catalog's `tickers` (the
    traded instruments) and this driver's own `secondary_ticker` pin --
    `PARAMETERS["secondary_ticker"]` is what the catalog check sees, but
    `SECONDARY_TICKER` is what is actually fetched and fed to the signal --
    are checked here too. Ticker order is not significant."""
    drift = batch5.parameter_drift(PARAMETERS, entry.get("parameters"))
    if drift is not None:
        return drift
    if PARAMETERS["secondary_ticker"] != SECONDARY_TICKER:
        return (f"PARAMETERS['secondary_ticker'] "
                f"{PARAMETERS['secondary_ticker']!r} != SECONDARY_TICKER "
                f"{SECONDARY_TICKER!r}")
    catalog_tickers = sorted(entry.get("tickers") or [])
    if catalog_tickers != sorted(TRADED):
        return (f"driver traded tickers {sorted(TRADED)} != catalog tickers "
                f"{catalog_tickers}")
    return None


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       end_date: str | None, signal_source_path: str,
                       signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT,
                       start_date: str | None = None) -> dict:
    if start_date is not None and not dry_run:
        raise ValueError(
            "compute_and_record: start_date override is dry-run-only -- a "
            "real (non-dry-run) scoring attempt must always use the "
            "measured REAL_DATA_START floor, never a shortened window")
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

    qqq_bars, resolved_end = fetch_full_history_bars(
        TRADED_TICKER, start_date=start_date, end_date=end_date)
    spy_bars, spy_resolved_end = fetch_full_history_bars(
        SECONDARY_TICKER, start_date=start_date, end_date=end_date)

    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    spy_daily_frame, _spy_source = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)

    trades_df = run_real_intraday_backtest(qqq_bars, spy_bars, cost_bps=cost_bps)

    calendar_index = pd.DatetimeIndex(sorted(set(qqq_bars.index.date)))
    curve = bridge.build_day_trading_curve(trades_df, calendar_index)

    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_daily_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY)

    resolved_start = start_date or REAL_DATA_START
    extra_note = (
        f"cost model: {TRADED_TICKER}={cost_bps:.2f}bps "
        f"({batch5.quotes_cost_source(quotes_payload)}); real QQQ intraday "
        f"history {resolved_start}..{resolved_end}; real SPY (secondary, "
        f"read-only reference, never traded) intraday history "
        f"{resolved_start}..{spy_resolved_end} (both Alpaca SIP 1-minute "
        "RTH-only bars, real market data); daily curve assembled via "
        "mining.day_trading_abi1_bridge.build_day_trading_curve")

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
        "real_intraday_data_start": resolved_start,
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
    parser.add_argument(
        "--start-date", default=None,
        help="DRY-RUN-ONLY override for the intraday fetch's start date "
             "(YYYY-MM-DD), to prove this driver against a short real "
             "window without a full REAL_DATA_START..today fetch. Rejected "
             "outside --dry-run (compute_and_record raises).")
    args = parser.parse_args(argv)

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes,
        ledger_path=args.ledger, run_id=args.run_id, tested_date=args.tested_date,
        dry_run=args.dry_run, end_date=args.end_date, start_date=args.start_date,
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
        "cross-asset-divergence-fvg-dry-run.json" if args.dry_run
        else "cross-asset-divergence-fvg-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
