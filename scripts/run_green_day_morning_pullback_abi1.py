"""Bespoke ABI-1 driver for
green_day_morning_pullback_continuation_daytrade -- pooled across SPY/QQQ,
this catalog's two most liquid index ETFs (the mechanism is general and
applies to any liquid index ETF; SPY/QQQ are chosen as the two most
liquid, most-traded index ETFs).

## Why this needs a bespoke driver

Same reason as every sibling intraday-track driver (see
``run_vwap_cross_decaying_conviction_abi1.py``'s own module docstring, not
re-derived here): the generic ``scripts/run_abi1_mining.py`` dispatch has no
intraday-timeframe path at all.

## Pooling

Reuses ``mining.day_trading_abi1_bridge.build_pooled_day_trading_curve``
directly -- the identical combination rule used by the other
pooled drivers: equal-weight
MEAN of every member's own daily sleeve, OR-of-in_market masks, every
member's own undiluted trades concatenated and ticker-tagged.

## tax_policy / account_type

Matches every sibling ``POOLED_SHAPES`` driver: scores UNTAXED in v1
(``tax_policy="none"``), the same simplification applied
verbatim.

## Observed intraday-data range

The intraday vendor behind ``mining.intraday_data.fetch_intraday_bars``
returns a full 390-bar RTH session for both SPY and QQQ from 2016-01-04
onward -- the same data floor ``run_fvg_reversal_cascade_pooled_abi1.py``
and ``run_vwap_cross_decaying_conviction_abi1.py`` use for these two
tickers.

## Single-shot scoring discipline

Same convention as every sibling bespoke driver: exactly ONE ABI-1
scoring attempt, ever (``already_scored_abi1`` guard below, backed by
``trial_ledger.record_outcome_if_not_scored``'s atomic check-and-append).
``--dry-run`` plus a ``SYNTHETIC``-marked quotes file proves this driver
correct against real fetched data without spending that attempt.
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
from mining import day_trading_green_day_pullback as mod
from mining import day_trading_abi1_bridge as bridge
from mining.intraday_backtest_engine import run_intraday_backtest

FAMILY_ID = "green_day_morning_pullback_continuation_daytrade"
POOL_TICKERS = ("SPY", "QQQ")
SHAPE = "pooled_single_asset"

# Byte-for-byte the catalog's own "parameters" field. Only seven of these
# nine keys are the signal FACTORY's own live parameters (asserted
# separately below) -- max_hold_days/direction describe the family's
# engine-level dispatch, not the signal's own logic.
PARAMETERS = {
    "min_rally_gain_pct": 0.0015,
    "min_consolidation_bars": 15,
    "max_pullback_retrace_pct": 0.5,
    "pullback_window_start_et": "09:45",
    "pullback_window_end_et": "11:30",
    "min_target_r_multiple": 1.5,
    "force_flat_et": "14:30",
    "max_hold_days": 1,
    "direction": "long",
}
_factory_defaults = {
    name: param.default
    for name, param in inspect.signature(
        mod.make_green_day_morning_pullback_continuation_signal_fn).parameters.items()
}
assert _factory_defaults == {
    "min_rally_gain_pct": PARAMETERS["min_rally_gain_pct"],
    "min_consolidation_bars": PARAMETERS["min_consolidation_bars"],
    "max_pullback_retrace_pct": PARAMETERS["max_pullback_retrace_pct"],
    "pullback_window_start_et": PARAMETERS["pullback_window_start_et"],
    "pullback_window_end_et": PARAMETERS["pullback_window_end_et"],
    "min_target_r_multiple": PARAMETERS["min_target_r_multiple"],
    "force_flat_et": PARAMETERS["force_flat_et"],
}, (
    "green_day_morning_pullback_continuation_daytrade driver's local "
    "PARAMETERS no longer match make_green_day_morning_pullback_"
    "continuation_signal_fn's own live defaults -- re-check before scoring")

# First full intraday session available for both pool members -- see
# the module docstring's "Observed intraday-data range".
REAL_DATA_START = "2016-01-04"

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
TAX_POLICY = "none"  # pooled_single_asset shape -- see
                     # module docstring.

DEFAULT_RUN_ID = "green-day-morning-pullback-pooled-abi1-v1"
DEFAULT_TESTED_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")

_NYSE_CALENDAR = mcal.get_calendar("NYSE")


def _most_recent_available_session(before_date: str) -> str:
    """The most recent real NYSE session STRICTLY BEFORE `before_date`.
    Mirrors run_vwap_cross_decaying_conviction_abi1.py's own
    identically-named helper as a per-driver local copy (not
    shared/imported); see that driver's module docstring for why same-day
    data is avoided."""
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
    """Fetch the FULL real available 1-minute RTH-bar history for `ticker`,
    from `start_date` (default `REAL_DATA_START`) through `end_date`
    (default: the most recent real session strictly before today).
    `start_date` and `end_date` are dry-run-only overrides (see
    `--start-date`/`--end-date`; `compute_and_record` refuses both outside
    `--dry-run`)."""
    resolved_end = end_date or _most_recent_available_session(
        pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"))
    bars = intraday_data.fetch_intraday_bars(
        ticker, start_date or REAL_DATA_START, resolved_end)
    return bars, resolved_end


def run_real_intraday_backtest(bars: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Runs the EXISTING, UNMODIFIED
    `make_green_day_morning_pullback_continuation_signal_fn` for ONE pool
    member's own bars. A FRESH `signal_fn` instance per ticker -- its
    closure carries real per-day state that must start clean and must never
    be shared across pool members.

    `tax_haircut_rate` is ALWAYS `0.0` -- same reasoning as every sibling
    intraday driver's own `run_real_intraday_backtest` (ABI-1's own tax
    model applies tax exactly once, and for this pooled family that model
    is `tax_policy="none"` -- see module docstring)."""
    signal_fn = mod.make_green_day_morning_pullback_continuation_signal_fn(
        min_rally_gain_pct=PARAMETERS["min_rally_gain_pct"],
        min_consolidation_bars=PARAMETERS["min_consolidation_bars"],
        max_pullback_retrace_pct=PARAMETERS["max_pullback_retrace_pct"],
        pullback_window_start_et=PARAMETERS["pullback_window_start_et"],
        pullback_window_end_et=PARAMETERS["pullback_window_end_et"],
        min_target_r_multiple=PARAMETERS["min_target_r_multiple"],
        force_flat_et=PARAMETERS["force_flat_et"])
    trades_df, _engine_daily_returns = run_intraday_backtest(
        bars, signal_fn, cost_bps=cost_bps, tax_haircut_rate=0.0,
        max_hold_days=PARAMETERS["max_hold_days"], direction="long")
    return trades_df


def _registration_drift(entry: dict) -> str | None:
    """None when this driver still matches the catalog entry it was
    registered against; otherwise the reason. `batch5.parameter_drift`
    compares only the `parameters` dict, so the catalog's `tickers` (the
    pool) is checked here too. Ticker order is not significant."""
    drift = batch5.parameter_drift(PARAMETERS, entry.get("parameters"))
    if drift is not None:
        return drift
    catalog_tickers = sorted(entry.get("tickers") or [])
    if catalog_tickers != sorted(POOL_TICKERS):
        return (f"driver pool tickers {sorted(POOL_TICKERS)} != catalog "
                f"tickers {catalog_tickers}")
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
    costs = batch5.execution_costs_for(list(POOL_TICKERS), quotes_payload)
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    spy_daily_frame, _spy_source = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)

    curves_by_ticker = {}
    trades_by_ticker = {}
    data_windows = {}
    for ticker in POOL_TICKERS:
        bars, resolved_end = fetch_full_history_bars(
            ticker, start_date=start_date, end_date=end_date)
        trades_df = run_real_intraday_backtest(bars, cost_bps=costs[ticker]["cost_bps"])
        trades_by_ticker[ticker] = trades_df
        calendar_index = pd.DatetimeIndex(sorted(set(bars.index.date)))
        curves_by_ticker[ticker] = bridge.build_day_trading_curve(trades_df, calendar_index)
        data_windows[ticker] = (start_date or REAL_DATA_START, resolved_end)

    curve = bridge.build_pooled_day_trading_curve(curves_by_ticker)
    account_type = entry["account_types"][0]
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_daily_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)

    cost_summary = "; ".join(
        f"{t}={costs[t]['cost_bps']:.2f}bps" for t in POOL_TICKERS)
    window_summary = "; ".join(
        f"{t} {w[0]}..{w[1]}" for t, w in data_windows.items())
    extra_note = (
        f"cost model (pool, {batch5.quotes_cost_source(quotes_payload)}): "
        f"{cost_summary}; real intraday data windows: {window_summary} "
        "(Alpaca SIP 1-minute RTH-only bars, real market data); pooled via "
        "mining.day_trading_abi1_bridge.build_pooled_day_trading_curve "
        "(equal-weight mean / OR-in_market / pool_trades, the day-trading "
        "port of build_pooled_curve's own combination rule); "
        "tax_policy=none per the same convention")

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
                catalog_entry=entry, instrument_set=list(POOL_TICKERS),
                cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
                tax_policy=TAX_POLICY, ticker_label="/".join(POOL_TICKERS),
                extra_note=extra_note, signal_source_path=signal_source_path,
                signal_source_sha256=signal_source_sha256)
        except trial_ledger.OutcomeAlreadyRecorded:
            return {"family_id": family_id, "reason_code": "already_scored_abi1",
                    "reason": "a concurrent run recorded a construction=\"ABI-1\" "
                              "outcome first -- this attempt's freshly computed "
                              "result is discarded"}

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    account_index = result["account_daily"].index
    return {
        "family_id": family_id, "shape": SHAPE, "tickers": list(POOL_TICKERS),
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "n_trades": int(len(curve["trades_df"])),
        "n_trades_by_ticker": {t: int(len(trades_by_ticker[t])) for t in POOL_TICKERS},
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
        "real_intraday_data_windows": data_windows,
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
             "strictly before today. Rejected outside --dry-run.")
    parser.add_argument(
        "--start-date", default=None,
        help="DRY-RUN-ONLY override for the intraday fetch's start date "
             "(YYYY-MM-DD). Rejected outside --dry-run.")
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
        "green-day-morning-pullback-pooled-dry-run.json" if args.dry_run
        else "green-day-morning-pullback-pooled-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
