"""Bespoke ABI-1 driver for bitcoin_extreme_decline_spillover_riskoff.

## Why this needs a bespoke driver, not the generic discover_modules path

``bitcoin_extreme_decline_spillover_riskoff.make_signal_fn``'s own module
docstring documents that
it is NOT reachable through ``scripts/run_abi1_mining.py``'s generic
dispatch: that driver's ``_build_curve`` calls
``module.make_signal_fn(symbol, index, frames)`` positionally, with no
``btc_close=`` argument, so the generic path silently returns a no-op
(``_empty_signal_fn``, zero trades) rather than raising. This is the SAME
shape every other external-data-driven ``single_long`` family in this
catalog already needed its own bespoke driver for (``money_market_fund_
flow_riskoff*``, ``fed_net_liquidity_regime*``) -- see this driver's own
sibling ``run_fed_net_liquidity_qqq_mining.py`` for the identical rationale
applied to a FRED input instead of a Kraken one.

## Dual-quote pattern (mirrors run_fed_net_liquidity_qqq_mining.py)

QQQ (the traded instrument) may have no capture in the shared batch5
quotes file, so this driver takes its OWN QQQ quotes file (default:
``<run dir>/quotes-qqq.json``) SEPARATE from the SPY quotes file used for
the benchmark cost (default: the SAME shared batch5 quotes file every
sibling driver already uses). BTC-USD is signal-only and is NEVER traded by this
family -- it therefore has no quote, no cost, and no execution_costs_for
call; only its real fetched price HISTORY (via
``mining.crypto_data.fetch_kraken_spot_ohlcv``) feeds the signal.

## --dry-run proof convention (mirrors run_vwap_cross_decaying_conviction_abi1.py)

The driver can be PROVED via ``--dry-run`` alone: this pipeline's
established convention (see that driver's own docstring) is that a family
is scored under ABI-1 exactly ONCE, ever. ``--dry-run`` plus a quotes file carrying the
literal ``SYNTHETIC`` marker (refused outside ``--dry-run`` by
``run_batch5_mining.refuse_synthetic_quotes``, the SAME guard every
sibling driver relies on) proves this driver correct against REAL fetched
BTC-USD data (real Kraken network call) and REAL QQQ/SPY OHLC
(``batch5.load_frame``'s real cache-or-fetch) without spending that one
real attempt.

## Provenance

Written to ``data/mining/bitcoin_spillover/provenance.json`` on every REAL
(non-dry-run) invocation, recording the fetch timestamp and the raw
response's sha256 -- both taken from whatever ``crypto_data.
fetch_kraken_spot_ohlcv`` actually attaches to its return value's
``.attrs`` (``retrieved_at``/``raw_sha256``), same convention as every
sibling driver's own provenance write (never an invented field).
"""
from __future__ import annotations

import argparse
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
from mining import abi1_driver, catalog as catalog_mod, crypto_data, evidence, trial_ledger
from mining.batch5 import bitcoin_extreme_decline_spillover_riskoff as btc_mod

CASH_CONVENTION = "sweep_cash_daily"
TRADED_TICKER = "QQQ"
# The ABI-1 benchmark is SPY unconditionally, system-wide -- NOT whatever is
# traded. abi1_driver.evaluate_family's own docstring: "passing the full
# SPY history even for a family trading back to the 1960s is correct, not a
# bug to work around." Every candidate's DSR is "excess over SPY";
# benchmarking this one against QQQ instead would make its DSR incomparable
# to the rest of the pipeline's calibration (DAILY_SR_VARIANCE/
# ABI1_N_TRIALS), defeating the whole point of a shared multiple-testing
# correction. (Identical reasoning to run_fed_net_liquidity_qqq_mining.py's
# own comment on this -- copied here deliberately, not re-derived.)
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"
TAX_POLICY = "realization"

DEFAULT_RUN_ID = "bitcoin-spillover-abi1"
DEFAULT_TESTED_DATE = batch5.DEFAULT_TESTED_DATE
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_ACCEPTED = batch5.DEFAULT_ACCEPTED
# QQQ's own traded-instrument cost (no capture in the shared batch5 quote
# file -- the run's own quotes-qqq.json, same shape as the fed-net-
# liquidity-qqq sibling driver's own DEFAULT_QQQ_QUOTES).
DEFAULT_QQQ_QUOTES = os.path.join(
    RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes-qqq.json")
# SPY's benchmark cost -- the SAME shared quote file every other family
# uses for its own SPY benchmark comparison.
DEFAULT_SPY_QUOTES = batch5.DEFAULT_QUOTES
PROVENANCE_PATH = os.path.join(
    REPO_ROOT, "data", "mining", "bitcoin_spillover", "provenance.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def fetch_btc_close(*, http_get=None, clock=None):
    """Real (or injected-for-test) fetch of BTC-USD's RAW daily Close price
    series via ``crypto_data.fetch_kraken_spot_ohlcv(interval=1440)``.

    If the fetched frame's ``.attrs.get("last_row_closed")`` is ``False``
    (the currently-forming, not-yet-committed candle -- see crypto_data.py's
    own module docstring), the last row is dropped before the Close series
    is built, so no-look-ahead correctness never depends on a still-mutable
    price. The returned Series carries the source frame's own ``.attrs``
    (``retrieved_at``/``raw_sha256``/``last_row_closed``/etc.) forward
    unchanged, so a caller (``main`` below) can write real provenance from
    it without a second fetch.

    ``http_get``/``clock`` are forwarded to ``fetch_kraken_spot_ohlcv``
    only when provided, matching that function's own optional-injection
    contract (production omits both and gets the real ``requests.get`` /
    real wall clock; tests inject a stub, same pattern as
    ``test_mining_crypto_data.py``'s own ``http_get`` stubbing)."""
    kwargs = {}
    if http_get is not None:
        kwargs["http_get"] = http_get
    if clock is not None:
        kwargs["clock"] = clock
    frame = crypto_data.fetch_kraken_spot_ohlcv("BTC-USD", interval=1440, **kwargs)
    if frame.attrs.get("last_row_closed") is False:
        frame = frame.iloc[:-1]
    close = frame["close"].astype(float).copy()
    close.attrs.update(frame.attrs)
    return close


def _load_accepted_entry(accepted_path: str, family_id: str) -> dict:
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [c for c in payload["accepted"] if c["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 accepted candidate {family_id!r}, found {len(matches)}")
    return matches[0]


def compute_and_record(*, qqq_frame, spy_frame, btc_close, catalog_path: str,
                       accepted_path: str, qqq_quotes_path: str, spy_quotes_path: str,
                       ledger_path: str, run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str) -> dict:
    family_id = btc_mod.FAMILY_ID
    try:
        _load_accepted_entry(accepted_path, family_id)
    except ValueError as exc:
        return {"family_id": family_id, "reason_code": "not_in_accepted", "reason": str(exc)}

    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    entry = catalog_entries.get(family_id)
    if entry is None:
        return {"family_id": family_id, "reason_code": "not_in_catalog",
                "reason": f"{family_id} is not in the catalog"}
    if entry.get("status") != "eligible":
        return {"family_id": family_id, "reason_code": "catalog_status",
                "reason": f"catalog status is {entry.get('status')!r}, not eligible"}
    drift = batch5.parameter_drift(btc_mod.PARAMETERS, entry.get("parameters"))
    if drift is not None:
        return {"family_id": family_id, "reason_code": "parameter_drift", "reason": drift}

    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"}
    if family_id in already_scored:
        return {"family_id": family_id, "reason_code": "already_scored_abi1",
                "reason": "a prior run already recorded a construction=\"ABI-1\" outcome"}

    # QQQ's own traded-instrument cost, and SPY's own SEPARATE benchmark
    # cost -- two different quote files, since QQQ has no capture in the
    # shared batch5 file and SPY's benchmark cost must stay the same real
    # sample every other family uses. BTC-USD is signal-only and is NEVER
    # traded, so it has no quote and no cost -- only its price history
    # (btc_close, already fetched by the caller) feeds the signal.
    qqq_quotes_payload = batch5.load_quotes(qqq_quotes_path)
    batch5.refuse_synthetic_quotes(qqq_quotes_payload, dry_run=dry_run)
    qqq_costs = batch5.execution_costs_for([TRADED_TICKER], qqq_quotes_payload)
    cost_bps = qqq_costs[TRADED_TICKER]["cost_bps"]

    spy_quotes_payload = batch5.load_quotes(spy_quotes_path)
    batch5.refuse_synthetic_quotes(spy_quotes_payload, dry_run=dry_run)
    spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK], spy_quotes_payload)
    benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    signal_fn = btc_mod.make_signal_fn(
        TRADED_TICKER, qqq_frame.index, {TRADED_TICKER: qqq_frame}, btc_close=btc_close)
    curve = abi1_driver.build_single_asset_curve(qqq_frame, signal_fn, cost_bps=cost_bps)
    account_type = entry["account_types"][0]
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)

    btc_window = (f"{btc_close.index.min().strftime('%Y-%m-%d')}.."
                 f"{btc_close.index.max().strftime('%Y-%m-%d')}"
                 if len(btc_close) else "empty")
    extra_note = (f"cost model: QQQ (traded)={cost_bps:.3f}bps "
                 f"({qqq_costs[TRADED_TICKER]['source']}) -- QQQ has no regular-"
                 "session capture in the shared batch5 quote file, hence its own "
                 f"quotes file; SPY (benchmark)={spy_costs[MARKET_BENCHMARK]['cost_bps']:.3f}bps "
                 f"({spy_costs[MARKET_BENCHMARK]['source']}) -- ABI-1's benchmark is "
                 "SPY unconditionally per abi1_driver.evaluate_family's own documented "
                 "contract, not whatever is traded, so every family's DSR stays "
                 "comparable under the shared DAILY_SR_VARIANCE/ABI1_N_TRIALS "
                 f"calibration; BTC-USD (signal-only, NEVER traded, no quote/cost) "
                 f"real fetched window {btc_window} (Kraken public spot OHLC, "
                 "mining.crypto_data.fetch_kraken_spot_ohlcv, interval=1440, "
                 "still-forming last candle dropped via .attrs['last_row_closed']); "
                 f"lag_trading_days={btc_mod.LAG_TRADING_DAYS}")

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result,
            catalog_entry=entry, instrument_set=[TRADED_TICKER],
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
            signal_source_path=signal_source_path,
            signal_source_sha256=signal_source_sha256)

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    return {
        "family_id": family_id, "shape": "single_long", "tickers": [TRADED_TICKER],
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "n_trades": int(len(curve["trades_df"])),
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
        "btc_data_window": btc_window,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--qqq-quotes", default=DEFAULT_QQQ_QUOTES)
    parser.add_argument("--spy-quotes", default=DEFAULT_SPY_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    args = parser.parse_args(argv)

    cache_dir = os.path.join(args.research_root, args.run_id, "artifacts", "ohlc")
    qqq_frame, _origin = batch5.load_frame(
        "QQQ", ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    spy_frame, _origin2 = batch5.load_frame(
        "SPY", ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    btc_close = fetch_btc_close()

    if not args.dry_run:
        os.makedirs(os.path.dirname(PROVENANCE_PATH), exist_ok=True)
        with open(PROVENANCE_PATH, "w", encoding="utf-8") as handle:
            json.dump({
                "fetched_for_run_id": args.run_id,
                "source_url": crypto_data.KRAKEN_SPOT_OHLC_URL,
                "point_in_time": True,
                "symbol": "BTC-USD",
                "retrieved_at": btc_close.attrs.get("retrieved_at"),
                "raw_sha256": btc_close.attrs.get("raw_sha256"),
            }, handle, indent=2)
            handle.write("\n")

    result = compute_and_record(
        qqq_frame=qqq_frame, spy_frame=spy_frame, btc_close=btc_close,
        catalog_path=args.catalog, accepted_path=args.accepted,
        qqq_quotes_path=args.qqq_quotes, spy_quotes_path=args.spy_quotes,
        ledger_path=args.ledger,
        run_id=args.run_id, tested_date=args.tested_date, dry_run=args.dry_run,
        signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__))

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
        "bitcoin-spillover-dry-run.json" if args.dry_run
        else "bitcoin-spillover-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
