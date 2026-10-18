"""Runs the first test of the sp500_index_addition_effect mining
catalog candidate (data/mining/catalog.json) -- its catalog entry carries
a NOT-TESTABLE-AS-SINGLE-TICKER note because the mechanism is a many-discrete-events-
across-many-tickers event study, which this pipeline's single-instrument
backtest engine has no slot for as a bare single-ticker candidate. This
script reformulates it as a pooled trade-level test (the same pooling
approach as run_expanded_universe_mining.py): pool real per-ticker trades
from REAL, individually-verified historical events into one trade-level
sample, instead of forcing the mechanism onto one placeholder ticker.

Each "trial" here is one REAL S&P 500 index-addition event: buy the added
company's stock the session after S&P Dow Jones Indices' real announcement
date (next-open execution, this engine's only validated execution style),
hold through the announcement, the effective inclusion date, and
hold_days_post_inclusion (catalog parameter, =10) TRADING days past the
effective date, then exit. Every event's ticker, announcement_date,
effective_date, and source_url were independently researched and cited
from public sources (data/mining/index_addition_cache/events.json) --
never invented or estimated.

Proxy note (documented, not silently glossed over): entry is at the
announcement date, matching the catalog's own parameters.entry=
'announcement_date' -- but next-open execution means the position is
opened the FOLLOWING session, missing that first day's own price reaction
(the same disclosed limitation as every other next-open signal in this
engine, e.g. signals.make_pead_signal_fn's earnings-day entry).

CRITICAL INVARIANTS (mirrors run_expanded_universe_mining.py's own):
  - family_id stays sp500_index_addition_effect -- a genuinely new
    family_id per ticker would permanently inflate the trial ledger's
    cumulative_family_count floor by dozens; this is ONE family tested
    across many real tickers, pooled, exactly like the expanded-universe
    precedent.
  - A first test computes a fresh holdout window here (the standard
    "most recent HOLDOUT_WINDOW_MONTHS ending run_date" convention) and pinned via record_first_test. Refuses
    to run if a pinned window already exists and disagrees with the
    freshly computed one (defensive -- should never happen for a
    genuinely first test, but a silent overwrite would be the ledger's
    core invariant broken).
  - Exactly ONE outcome event is recorded for this family this run.
  - Every ticker excluded at any stage is logged with a reason -- no
    silent caps.

Usage:
    python scripts/run_index_addition_mining.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import backtest_engine, catalog as catalog_mod, constants, cost_tax_inputs, dsr, evidence, pooling, signals, trial_ledger  # noqa: E402
import data_pipeline  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LEDGER_PATH = os.path.join(REPO_ROOT, "data", "mining", "trial_ledger.jsonl")
FAMILY_ID = "sp500_index_addition_effect"
ACCOUNT_TYPE = "taxable"  # candidate['account_types'][0]


def _log(msg):
    print(msg, flush=True)


def _fetch_ohlc(ticker: str):
    """Wraps data_pipeline.fetch_ohlc's raw shape (a 'Date' column, not
    yet the index) into the (Open,High,Low,Close, DatetimeIndex) shape
    every mining.* module expects -- same normalization the standard
    mining pipeline and run_expanded_universe_mining.py both do inline."""
    raw = data_pipeline.fetch_ohlc(ticker, start="max")
    if raw is None or len(raw) == 0:
        return None
    df = raw.set_index(pd.to_datetime(raw["Date"])).drop(columns=["Date"])
    df = df[["Open", "High", "Low", "Close"]].dropna(how="any")
    return df if len(df) > 0 else None


def _snap_forward(ohlc_index: pd.DatetimeIndex, date_str: str):
    """Finds `date_str` (or the first trading day AT OR AFTER it, in case
    the real calendar date itself is a weekend/holiday and so is absent
    from this ticker's OHLC index) in `ohlc_index`. Returns None if
    `date_str` is entirely unreachable (after every date in the index).
    Used for the entry date, and `_compute_exit_date` applies the same
    snap-forward rule to the effective date, so a real announcement date
    that falls on a weekend/holiday (rare, but not structurally impossible)
    enters the next available session, exactly as the effective/exit side
    does, instead of silently dropping that event's trade (which an exact
    string match with no fallback would do)."""
    ts = pd.Timestamp(date_str)
    pos = ohlc_index.searchsorted(ts)
    if pos >= len(ohlc_index):
        return None
    return ohlc_index[pos].strftime("%Y-%m-%d")


def _compute_exit_date(ohlc_index: pd.DatetimeIndex, effective_date: str,
                        hold_days_post_inclusion: int):
    """Finds `effective_date` (with the same snap-forward rule as
    _snap_forward) in `ohlc_index`, then
    returns the date `hold_days_post_inclusion` TRADING days later.
    Returns None if `effective_date` is not reachable at all, or the LAST
    available date if the hold would run past the end of available
    history (the position force-closes via run_backtest's own
    end_of_data path in that case -- disclosed, not fixed)."""
    eff_ts = pd.Timestamp(effective_date)
    pos_arr = ohlc_index.searchsorted(eff_ts)
    if pos_arr >= len(ohlc_index):
        return None
    exit_pos = pos_arr + hold_days_post_inclusion
    if exit_pos >= len(ohlc_index):
        exit_pos = len(ohlc_index) - 1
    return ohlc_index[exit_pos].strftime("%Y-%m-%d")


def run_events(events: list, hold_days_post_inclusion: int, tax_rate: float, *,
               bucket: str, holdout_start: str, holdout_end: str):
    """For each event dict ({ticker, announcement_date, effective_date,
    cost_status, bid, ask, ...}), fetch that ticker's full OHLC history,
    then BOUND it to the slice appropriate for `bucket` before building the
    signal_fn or running the backtest. ONE unbounded full-history backtest
    per ticker, bucketed post-hoc by each trade's entry_date, would let a
    HOLDOUT trade's exit read real price data from AFTER holdout_end
    (future-leakage into the out-of-sample result), and symmetrically let
    an IN-SAMPLE trade's exit read real price data from INSIDE the holdout
    window (contaminating the very screen the holdout window exists to keep
    clean) -- the reference pipeline's own holdout re-run avoids both by
    slicing df to `holdout_start:holdout_end` before backtesting at all,
    and this function does the same: `bucket='in_sample'` slices to strictly BEFORE
    holdout_start; `bucket='holdout'` slices to
    [holdout_start, holdout_end] inclusive. `_compute_exit_date` is called
    on THAT bounded slice's own index, so a hold that would run past the
    slice's own last bar force-closes at that last bar (run_backtest's
    documented end_of_data path) -- it can structurally never read a price
    outside the bucket's own window.

    Returns (trades_by_ticker, index_by_ticker, coverage) -- same shape
    convention as run_expanded_universe_mining.run_family_in_sample, so a
    ticker producing zero trades is not an error (pooling.py's own
    documented convention)."""
    if bucket not in ("in_sample", "holdout"):
        raise ValueError(f"bucket must be 'in_sample' or 'holdout', got {bucket!r}")
    holdout_start_ts = pd.Timestamp(holdout_start)
    holdout_end_ts = pd.Timestamp(holdout_end)

    trades_by_ticker = {}
    index_by_ticker = {}
    coverage = {"attempted": 0, "ok": 0, "skipped": {}}

    for e in events:
        t = e["ticker"]
        if e.get("cost_status") != "ok":
            coverage["skipped"][t] = (
                f"no usable live cost quote available -- {e.get('cost_note', 'unspecified reason')}")
            continue
        full_ohlc = _fetch_ohlc(t)
        if full_ohlc is None:
            coverage["skipped"][t] = "no OHLC data available"
            continue
        if bucket == "in_sample":
            ohlc = full_ohlc[full_ohlc.index < holdout_start_ts]
        else:
            ohlc = full_ohlc[(full_ohlc.index >= holdout_start_ts) & (full_ohlc.index <= holdout_end_ts)]
        if len(ohlc) == 0:
            coverage["skipped"][t] = f"no OHLC bars within this event's own {bucket} window"
            continue
        entry_date = _snap_forward(ohlc.index, e["announcement_date"])
        if entry_date is None:
            coverage["skipped"][t] = (
                f"announcement_date {e['announcement_date']} is after this ticker's "
                f"{bucket}-window OHLC ends ({ohlc.index[-1].date()})")
            continue
        exit_date = _compute_exit_date(ohlc.index, e["effective_date"], hold_days_post_inclusion)
        if exit_date is None:
            coverage["skipped"][t] = (
                f"effective_date {e['effective_date']} is after this ticker's "
                f"{bucket}-window OHLC ends ({ohlc.index[-1].date()})")
            continue
        if entry_date >= exit_date:
            coverage["skipped"][t] = (
                f"snapped entry_date {entry_date} is not before exit_date {exit_date} -- "
                f"degenerate window, refusing to build a signal")
            continue

        coverage["attempted"] += 1
        cost_bps = cost_tax_inputs.compute_spread_bps(e["bid"], e["ask"])
        signal_fn = signals.make_index_addition_signal_fn(entry_date, exit_date)
        trades, _ = backtest_engine.run_backtest(
            ohlc, signal_fn, cost_bps=cost_bps, tax_haircut_rate=tax_rate)
        if len(trades) > 1:
            raise RuntimeError(
                f"{t}: single-shot signal produced {len(trades)} trades, expected 0 or 1 -- "
                f"signal_fn defect, refusing to silently pool")
        trades_by_ticker[t] = trades
        index_by_ticker[t] = ohlc.index
        coverage["ok"] += 1
        if len(trades) == 0:
            coverage["skipped"][t] = (
                f"signal built (entry={entry_date}, exit={exit_date}) but produced "
                f"zero trades -- entry date not present in this ticker's {bucket}-window OHLC")

    return trades_by_ticker, index_by_ticker, coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=os.path.join(
        REPO_ROOT, "data", "mining", "index_addition_cache", "events.json"))
    parser.add_argument("--run-id", default="index-addition")
    parser.add_argument("--tested-date",
                        default=datetime.date.today().isoformat())
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "runs", "mining", "index-addition", "artifacts",
        "index_addition_results.json"))
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute everything but do not write to the trial ledger.")
    args = parser.parse_args()

    with open(args.events, "r", encoding="utf-8") as f:
        events_payload = json.load(f)
    holdout_start = events_payload["holdout_start"]
    holdout_end = events_payload["holdout_end"]
    in_sample_events = events_payload["in_sample"]
    holdout_events = events_payload["holdout"]
    total_event_count = len(in_sample_events) + len(holdout_events)
    _log(f"Loaded {total_event_count} real, cited addition events "
         f"({len(in_sample_events)} pre-holdout, {len(holdout_events)} in the "
         f"{holdout_start}..{holdout_end} window) from {events_payload['source']}")

    catalog = catalog_mod.load_catalog(os.path.join(REPO_ROOT, "data", "mining", "catalog.json"))
    entry = next(e for e in catalog if e["family_id"] == FAMILY_ID)
    hold_days = int(entry["parameters"]["hold_days_post_inclusion"])
    _log(f"hold_days_post_inclusion (catalog): {hold_days}")

    pinned = trial_ledger.get_pinned_holdout_window(LEDGER_PATH, FAMILY_ID)
    is_retry = False
    if pinned is not None:
        # A not_evaluated_error family is a caught DEFECT, not evidence
        # about the market (evidence.build_engine_error's own docstring)
        # -- it stays eligible to re-enter this script once the defect is
        # fixed, matching ruled_out.filter_eligible's own treatment at the
        # catalog-filter level. Every OTHER outcome (a scored result) means
        # this family genuinely already has a result, and the
        # "first-test only" refusal applies.
        prior_outcome = trial_ledger.latest_outcome(LEDGER_PATH, FAMILY_ID)
        if prior_outcome != "not_evaluated_error":
            raise RuntimeError(
                f"{FAMILY_ID} already has a pinned holdout window ({pinned}) and a "
                f"real recorded outcome ({prior_outcome!r}) -- this script is only "
                f"for a genuinely FIRST test; a re-test script must reuse the pinned "
                f"window instead of computing a fresh one (see trial_ledger."
                f"get_pinned_holdout_window's own contract).")
        # A not_evaluated_error retry is safe ONLY against the SAME pinned
        # window the earlier failed attempt used -- record_first_test
        # below is idempotent and will NOT re-pin, so scoring against a
        # DIFFERENT window here would silently record an outcome that
        # does not match what the ledger says was tested. This script's
        # own holdout_start/holdout_end come from the events cache file
        # (events_payload), not a date-derived computation, so they are
        # deterministic across runs UNLESS that file itself changes --
        # fail loud rather than risk a silent mismatch either way.
        if (pinned["holdout_start"] != holdout_start or
                pinned["holdout_end"] != holdout_end):
            raise RuntimeError(
                f"{FAMILY_ID}: not_evaluated_error retry window mismatch -- "
                f"pinned {pinned['holdout_start']}..{pinned['holdout_end']}, "
                f"this run's events source computed {holdout_start}.."
                f"{holdout_end}. Refusing to score against a different "
                f"window than the one the ledger says was tested.")
        is_retry = True
        _log(f"Retrying {FAMILY_ID} after a prior not_evaluated_error, "
             f"same pinned window {holdout_start} to {holdout_end}")

    # NOTE: record_first_test is deliberately
    # NOT called here. In the standard mining pipeline's Score phase,
    # record_first_test runs immediately before scoring, AFTER a prior
    # Backtest phase has already fetched every candidate's OHLC/cost data and
    # computed its trades; here, this script's own OHLC-fetch-and-backtest loop
    # (run_events, one network fetch per event ticker) has NOT YET RUN at
    # this point. Pinning the ledger before that loop would mean any
    # fetch/backtest exception partway through leaves a first_test
    # event with no matching outcome, and the pinned-window guard above would
    # then refuse every future retry -- an unrecoverable ledger state without
    # manual repair. So record_first_test runs only after
    # every ticker has been fetched, backtested, and pooled (mirroring where
    # the reference pipeline's Score phase actually sits: strictly after its
    # own Backtest phase, not before it) -- the only remaining risk window
    # (the DSR math and, if passed_dsr, none here since holdout trades are
    # already pooled too) is the same kind of narrow, accepted residual risk
    # the reference pipeline's own Score phase carries.
    tax_rate = cost_tax_inputs.tax_haircut_rate_for(ACCOUNT_TYPE)
    # Two SEPARATE, independently-bounded passes -- see run_events' own
    # docstring for why an event's bucket
    # must be decided BEFORE backtesting (via which OHLC slice it runs
    # against), not after.
    in_sample_trades, in_sample_index, in_sample_cov = run_events(
        in_sample_events, hold_days, tax_rate,
        bucket="in_sample", holdout_start=holdout_start, holdout_end=holdout_end)
    holdout_trades, holdout_index, holdout_cov = run_events(
        holdout_events, hold_days, tax_rate,
        bucket="holdout", holdout_start=holdout_start, holdout_end=holdout_end)
    coverage = {
        "attempted": in_sample_cov["attempted"] + holdout_cov["attempted"],
        "ok": in_sample_cov["ok"] + holdout_cov["ok"],
        "skipped": {**in_sample_cov["skipped"], **holdout_cov["skipped"]},
        "in_sample": in_sample_cov, "holdout": holdout_cov,
    }
    _log(f"Backtested {coverage['ok']}/{total_event_count} tickers "
         f"({len(coverage['skipped'])} skipped -- see coverage in output JSON)")

    pooled_in_sample = pooling.pool_trades(in_sample_trades)
    stats_in_sample = pooling.pooled_trade_stats(pooled_in_sample)
    _log(f"Pooled in-sample: {stats_in_sample}")

    # UNCONDITIONAL and FIRST among the remaining scoring steps, matching the
    # reference pipeline's Score-phase convention exactly (see that script's
    # own comment): every family tested must be recorded in the ledger
    # whether it passes or not, and recording before the DSR/holdout math
    # (rather than after) is what stops a data-shape edge case in that math
    # from silently skipping the ledger write.
    if not args.dry_run:
        recorded = trial_ledger.record_first_test(
            LEDGER_PATH, FAMILY_ID, args.tested_date, holdout_start, holdout_end, [], args.run_id)
        # `recorded is False` is expected, not an error, for a verified
        # not_evaluated_error retry (`is_retry`, set above only after
        # confirming the window matches what is already pinned) -- the
        # window was already pinned by the earlier failed attempt, and
        # record_first_test correctly no-ops rather than re-pinning.
        if not recorded and not is_retry:
            raise AssertionError(
                "record_first_test returned False on a family with no prior "
                "pinned window -- unexpected")
        _log(f"Pinned first_test holdout window: {holdout_start} to {holdout_end}"
             if recorded else
             f"Reusing existing pinned window (not_evaluated_error retry): "
             f"{holdout_start} to {holdout_end}")

    # Everything from here through
    # the record_outcome call below is wrapped in try/except so that ANY
    # exception in this stretch (a non-finite value reaching dsr's own
    # guards, a filesystem failure, an interrupted process) still leaves a
    # matching outcome event in the ledger -- a family pinned via
    # record_first_test with NO outcome event would permanently trip this
    # script's own "already pinned, refusing to recompute" guard on every
    # future attempt, an unrecoverable state without manual ledger repair.
    # The except branch records a not_evaluated_error outcome describing the
    # failure, then re-raises so the run still fails loudly and exits
    # non-zero -- it does not paper over the error, only ensures the ledger
    # itself is never left in a permanently-blocked state because of it.
    try:
        if stats_in_sample["trade_count"] < constants.MIN_TRADES_FOR_SCORING:
            # insufficient_evidence, not a scored DSR cut: no real DSR is
            # computed on this
            # branch, so filing it as a statistical rejection would be
            # indistinguishable from a real one.
            stage = "insufficient_evidence"
            dsr_value = None
            reason = (f"only {stats_in_sample['trade_count']} pooled in-sample trades, below "
                       f"MIN_TRADES_FOR_SCORING ({constants.MIN_TRADES_FOR_SCORING})")
            passed_dsr = False
            passed_holdout = False
            holdout_result = None
            shared_n_trials = None
            shared_sr_variance = None
            attempted_tickers = sorted(in_sample_trades.keys())
            in_sample_indices = [idx for idx in in_sample_index.values() if len(idx)]
            evidence_in_sample_start = (
                min(idx.min() for idx in in_sample_indices).strftime("%Y-%m-%d")
                if in_sample_indices else holdout_start)
            evidence_in_sample_end = (
                max(idx.max() for idx in in_sample_indices).strftime("%Y-%m-%d")
                if in_sample_indices else holdout_start)
            # evidence.build_insufficient_trades reads stats['sharpe_in_
            # sample']; this pipeline's own stats dict (pooling.
            # pooled_trade_stats) keys the identical per-trade value as
            # 'sharpe' -- translated here so a real measured value is
            # recorded instead of silently defaulting to None.
            stats_for_evidence = {**stats_in_sample,
                                  "sharpe_in_sample": stats_in_sample.get("sharpe")}
            outcome_evidence = evidence.build_insufficient_trades(
                stats=stats_for_evidence,
                catalog_entry={"family_id": FAMILY_ID,
                               "parameters": dict(entry.get("parameters") or {})},
                instrument_set=attempted_tickers,
                in_sample_start=evidence_in_sample_start,
                in_sample_end=evidence_in_sample_end,
                pinned_window={"holdout_start": holdout_start,
                               "holdout_end": holdout_end},
                # Both buckets are fetched UNCONDITIONALLY above (the two
                # run_events passes), before this trade-count gate -- holdout_trades
                # already exists for this event set, it was just never
                # scored.
                holdout_look="computed_not_scored",
                signal_source_path="src/mining/signals.py",
                signal_source_sha256=evidence.source_sha256(
                    os.path.join(_SRC, "mining", "signals.py")),
                evidence_source="measured",
                reactivation={
                    "kind": "none_available",
                    "rationale": "real, individually-verified S&P 500 "
                                 "index-addition events already exhaust the "
                                 "cited source for this window; no "
                                 "pre-registered widening predicate. The "
                                 "family stays open to a fresh look via a "
                                 "new family_id",
                })
        else:
            outcome_evidence = None
            # Single-candidate run (this family is the only trial this run) --
            # num_independent_trials needs >=2 series to estimate correlation
            # meaningfully; with exactly one, it returns 1.0 by its own
            # documented m==1 branch, so this run's correction is entirely the
            # cumulative ledger floor -- same convention dsr.py's own module
            # docstring and constants.DEFAULT_SR_VARIANCE_FLOOR's comment both
            # describe for "a single-candidate run".
            series = pooling.pooled_daily_returns_by_exit_day(
                {t: (in_sample_trades[t], in_sample_index[t]) for t in in_sample_trades if t in in_sample_index},
                window_days=constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS)
            arr = series.values
            if arr.size > 0 and np.std(arr) != 0.0 and np.all(np.isfinite(arr)):
                n_this_run = dsr.num_independent_trials([arr])
            else:
                n_this_run = 1.0
            ledger_floor = trial_ledger.cumulative_family_count(LEDGER_PATH)
            shared_n_trials = max(n_this_run, ledger_floor, 1.0)
            # sr_variance, pooled across trials (the most principled
            # estimate). This script always tests exactly ONE family, so
            # it can never estimate a per-run sample variance -- without the
            # ledger pool it would fall back to DEFAULT_SR_VARIANCE_FLOOR every
            # time, an ungrounded placeholder. It is the ONE script
            # in this pipeline structurally guaranteed to always need this
            # tier; see trial_ledger.pooled_sr_variance's own docstring.
            pooled_variance = trial_ledger.pooled_sr_variance(LEDGER_PATH)
            shared_sr_variance = (
                pooled_variance if pooled_variance is not None
                else constants.DEFAULT_SR_VARIANCE_FLOOR)
            _log(f"Shared correction N: this-run estimate={n_this_run:.4f}, "
                 f"ledger cumulative floor={ledger_floor}, USED={shared_n_trials:.4f}")
            _log(f"Shared sr_variance: {shared_sr_variance} "
                 f"({'pooled across the ledger' if pooled_variance is not None else 'pre-registered floor -- pool not yet seeded'})")

            dsr_value = dsr.deflated_sharpe_ratio(
                sr=stats_in_sample["sharpe"], n=stats_in_sample["trade_count"],
                skew=stats_in_sample["skew"], kurtosis=stats_in_sample["kurtosis"],
                n_trials=shared_n_trials, sr_variance=shared_sr_variance)
            passed_dsr = dsr_value >= constants.DSR_PASS_THRESHOLD
            passed_holdout = False
            holdout_result = None

            if passed_dsr:
                pooled_holdout = pooling.pool_trades(holdout_trades)
                holdout_net_edge_bps = float(pooled_holdout["net_ret"].mean() * 1e4) if len(pooled_holdout) else 0.0
                passed_holdout = backtest_engine.holdout_passes(stats_in_sample["net_edge_bps"], holdout_net_edge_bps)
                holdout_result = {
                    "holdout_trade_count": int(len(pooled_holdout)),
                    "holdout_ticker_count": int(pooled_holdout["ticker"].nunique()) if len(pooled_holdout) else 0,
                    "holdout_net_edge_bps": holdout_net_edge_bps,
                    "in_sample_net_edge_bps": stats_in_sample["net_edge_bps"],
                    "passed_holdout": passed_holdout,
                }

            stage = "survived" if (passed_dsr and passed_holdout) else ("cut_holdout" if passed_dsr else "cut_dsr")
            if stage == "cut_dsr":
                reason = f"DSR {dsr_value:.4f} below the {constants.DSR_PASS_THRESHOLD} bar (n_trials={shared_n_trials:.2f})"
            elif stage == "cut_holdout":
                reason = (f"passed DSR ({dsr_value:.4f}) but holdout net edge "
                           f"{holdout_result['holdout_net_edge_bps']:.1f} bps did not confirm the "
                           f"in-sample {holdout_result['in_sample_net_edge_bps']:.1f} bps edge")
            else:
                reason = None

        _log(f"RESULT: stage={stage} dsr_value={dsr_value} passed_dsr={passed_dsr} passed_holdout={passed_holdout}")

        pooled_holdout_for_output = pooling.pool_trades(holdout_trades) if holdout_trades else pd.DataFrame()
        ticker_label = (
            f"{coverage['ok']} real, individually-verified S&P 500 index-addition events "
            f"(2024-2026, press.spglobal.com-cited), pooled trade-level test")
        note = (
            f"Event-study test of {FAMILY_ID} (not testable as a single-ticker candidate -- "
            f"see catalog NOT-TESTABLE-AS-SINGLE-TICKER note). Reformulated as a real event study: "
            f"{coverage['ok']} verified S&P 500 addition events backtested individually (entry = "
            f"session after the real announcement date, exit = effective inclusion date + "
            f"{hold_days} trading days), pooled trade-level per mining.pooling's methodology. "
            f"{stats_in_sample['trade_count']} pooled in-sample trades from "
            f"{stats_in_sample['ticker_count']} tickers, {len(coverage['skipped'])} tickers skipped "
            f"(missing/unusable live cost quote or data). Reason if cut: {reason}. "
            "Entry timing is a next-open proxy for 'buy upon announcement' (this engine's only "
            "validated execution style), and cost/tax are real per-ticker live quotes/rates, never "
            "invented.")

        # sharpe_in_sample carried only when a real DSR was computed (the
        # too-few-trades branch above leaves dsr_value None) -- see
        # record_outcome's own docstring on why a fabricated sharpe on an
        # unscored outcome would corrupt trial_ledger.pooled_sr_variance.
        sharpe_in_sample = (stats_in_sample["sharpe"]
                            if dsr_value is not None else None)
        if not args.dry_run:
            trial_ledger.record_outcome(
                LEDGER_PATH, FAMILY_ID, args.run_id, args.tested_date, stage,
                dsr_value=dsr_value, note=note, ticker_label=ticker_label,
                sharpe_in_sample=sharpe_in_sample, evidence=outcome_evidence)
            _log(f"Recorded outcome for {FAMILY_ID}: stage={stage}")
        else:
            _log(f"[DRY RUN] would record outcome for {FAMILY_ID}: stage={stage}")
    except Exception as exc:
        if not args.dry_run:
            # not_evaluated_error, not a scored DSR cut: a caught exception
            # is a defect,
            # not a statistical finding about the market.
            # Sourced from the ORIGINAL event list (populated before this
            # try block even starts), not in_sample_trades -- the latter
            # could be empty in a total-failure run (every ticker's fetch
            # failed), and an empty instrument_set would make this
            # defensive record itself raise inside the exception handler.
            defensive_instrument_set = sorted(
                {e["ticker"] for e in in_sample_events + holdout_events})
            defensive_evidence = evidence.build_engine_error(
                error=exc, instrument_set=defensive_instrument_set,
                pinned_window={"holdout_start": holdout_start,
                               "holdout_end": holdout_end},
                # Both buckets are fetched unconditionally before this
                # try block (the two run_events passes), so holdout was already
                # computed by the time any exception in this stretch fires.
                holdout_look="computed_not_scored")
            trial_ledger.record_outcome(
                LEDGER_PATH, FAMILY_ID, args.run_id, args.tested_date,
                "not_evaluated_error", dsr_value=None, evidence=defensive_evidence,
                note=(f"Scoring/outcome step raised {type(exc).__name__}: {exc}. Recorded defensively "
                      f"so this family_id's pinned first_test window is never left without a matching "
                      f"outcome event -- investigate and, if the underlying bug is fixed, this family "
                      f"remains eligible for a future re-test against this SAME pinned window "
                      f"(cut families stay eligible forever per this pipeline's own convention)."),
                ticker_label=f"{coverage['ok']} tickers backtested before the failure")
            _log(f"Recorded a defensive not_evaluated_error outcome for {FAMILY_ID} after an exception: {exc}")
        raise

    def _serialize(obj):
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.to_json(orient="records", date_format="iso"))
        return obj

    out_payload = {
        "run_id": args.run_id, "tested_date": args.tested_date, "family_id": FAMILY_ID,
        "holdout_start": holdout_start, "holdout_end": holdout_end,
        "hold_days_post_inclusion": hold_days,
        "events_source": events_payload["source"],
        "coverage": coverage,
        "stats_in_sample": stats_in_sample,
        "shared_n_trials": shared_n_trials, "shared_sr_variance": shared_sr_variance,
        "stage": stage, "dsr_value": dsr_value, "passed_dsr": passed_dsr,
        "passed_holdout": passed_holdout, "reason": reason, "holdout_result": holdout_result,
        "pooled_in_sample_trades": pooled_in_sample,
        "pooled_holdout_trades": pooled_holdout_for_output,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_serialize(out_payload), f, indent=2, default=str)
    _log(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
