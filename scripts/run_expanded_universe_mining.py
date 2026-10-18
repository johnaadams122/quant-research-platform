"""Runs a POOLED-UNIVERSE test of two mining catalog candidates whose
signal is derived purely from price/earnings data
(fifty_two_week_high_momentum and pead_earnings_surprise_drift) across
every ticker in the S&P 500 and NASDAQ 100, against each family's
already-pinned holdout window.

Each family stays ONE family_id, never one per ticker -- see
mining.pooling's module docstring for the full statistical methodology
(why trade-level pooling across tickers, not daily-return averaging or
per-ticker-Sharpe averaging), and this script's own inline comments for
the exact procedure, which mirrors the batch runner's scoring
(scripts/run_batch3_mining.py) as closely as a single-run, two-family,
many-ticker case allows.

CRITICAL INVARIANTS:
  - family_id stays fifty_two_week_high_momentum / pead_earnings_surprise_drift
    -- never a per-ticker family_id (that would permanently inflate the
    trial ledger's cumulative_family_count floor by hundreds).
  - The PINNED holdout window is READ from the ledger
    (trial_ledger.get_pinned_holdout_window), never recomputed.
  - Exactly ONE new outcome event is recorded per family this run.
  - Every ticker that fails at any stage is logged with a reason, never
    silently dropped -- this script's own coverage report is the "no
    silent caps" evidence trail.

Usage:
    python scripts/run_expanded_universe_mining.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import backtest_engine, catalog as catalog_mod, cost_tax_inputs, constants, dsr, evidence, pooling, signals, trial_ledger  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LEDGER_PATH = os.path.join(REPO_ROOT, "data", "mining", "trial_ledger.jsonl")
ACCOUNT_TYPE = "taxable"  # candidate['account_types'][0] for both families

# Two tickers CONFIRMED (not merely suspected) to carry a genuine yfinance/
# Yahoo raw-data defect in their price history, excluded from BOTH
# families for data-quality reasons,
# not to flatter the result -- each is corroborated by two independent
# signals, not just "this number looks big":
#   HUBB: an isolated, implausibly large single-day return in its early
#     history (present in BOTH the bulk yf.download() path and the single-
#     ticker yf.Ticker(...).history() path used by data_pipeline.fetch_ohlc
#     elsewhere in this repo -- so it is a defect in the underlying feed
#     itself, not a fetch-method artifact), preceded by an extended run of
#     perfectly flat/stale pricing.
#   MNST: the price repeatedly, exactly round-trips to ~2x/~0.5x its prior
#     close over a few weeks -- no real security oscillates between two
#     price levels like this; it is the signature of a stock-split
#     adjustment race in the data feed, and it sits INSIDE the pinned
#     holdout window, so it would corrupt the holdout confirmation
#     directly, not just an old historical statistic.
# A scan for the weaker "single day near-exactly 2x/0.5x" signature finds
# other isolated historical dates; those are left IN -- most correspond to
# real, identifiable market events (profit warnings, short-seller reports,
# market-wide crash days, financial-crisis volatility) or are old, isolated,
# single-ticker events immaterial at a pooled universe's scale. Excluding
# them would be removing real risk to flatter the result. Only HUBB and MNST
# meet the higher bar of a CONFIRMED, corroborated defect.
DATA_QUALITY_EXCLUDED_TICKERS = {
    "HUBB": "Isolated, implausibly large single-day return in early history "
            "(present in both the bulk and single-ticker yfinance fetch paths) "
            "preceded by stale/flat pricing -- a feed defect.",
    "MNST": "Repeated near-exact 2x/0.5x single-day price round-trips "
            "inside the pinned holdout window -- a stock-split adjustment "
            "defect, not real price behavior.",
}


def _log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlc_cache(cache_dir: str, tickers: list) -> dict:
    """Returns {ticker: DataFrame(Open,High,Low,Close, DatetimeIndex)}.
    Every ticker not found in the cache is simply absent from the
    returned dict -- caller is responsible for diffing against the full
    universe list to log what's missing."""
    ohlc = {}
    for t in tickers:
        path = os.path.join(cache_dir, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[["Open", "High", "Low", "Close"]].dropna(how="any")
        if len(df) == 0:
            continue
        ohlc[t] = df
    return ohlc


def load_cost_quotes(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_earnings_cache(cache_dir: str, tickers: list) -> dict:
    """Returns {ticker: {'ok': bool, 'dates': [...], 'reason': str}}."""
    out = {}
    for t in tickers:
        path = os.path.join(cache_dir, f"{t}.json")
        if not os.path.exists(path):
            out[t] = {"ok": False, "dates": [], "reason": "not fetched (cache file missing)"}
            continue
        with open(path, "r", encoding="utf-8") as f:
            out[t] = json.load(f)
    return out


def cost_bps_for(ticker: str, quotes: dict) -> float:
    """Round-trip cost in bps from a real live bid/ask
    (cost_tax_inputs.cost_bps_from_quotes(bid, ask, bid, ask) -- same
    quote used for both the entry and exit leg's spread estimate)."""
    q = quotes[ticker]
    return cost_tax_inputs.cost_bps_from_quotes(q["bid"], q["ask"], q["bid"], q["ask"])


# ---------------------------------------------------------------------------
# Per-family, per-ticker backtest runner
# ---------------------------------------------------------------------------

def run_family_in_sample(tickers, ohlc, quotes, holdout_start, signal_fn_for_ticker,
                          min_bars=5, time_stop=None):
    """Runs the in-sample backtest for one family across every ticker
    with OHLC data. `signal_fn_for_ticker(ticker) -> SignalFn or None`
    (None means "skip this ticker for this family" -- used by PEAD to
    skip a ticker with zero usable earnings dates, or missing earnings
    data). `time_stop` is passed straight through to run_backtest --
    REQUIRED for pead_earnings_surprise_drift (catalog parameters.
    hold_days=60): that signal_fn never sets exit=True by design (see
    signals.make_pead_signal_fn's own docstring -- "exit is via the
    engine's time_stop=hold_days mechanism only"), so omitting time_stop
    here would hold a position open indefinitely (to end_of_data)
    instead of exiting after hold_days sessions (a single inflated
    "trade" per ticker, holding from the FIRST qualifying earnings date
    all the way to the last in-sample bar, instead of 60-day round
    trips). fifty_two_week_high_momentum manages its own entries/exits
    every rebalance and passes time_stop=None. Returns (trades_by_ticker,
    index_by_ticker, coverage) where coverage logs exactly why any
    ticker was excluded."""
    trades_by_ticker = {}
    index_by_ticker = {}
    coverage = {"attempted": 0, "ok": 0, "skipped": {}}
    holdout_start_ts = pd.Timestamp(holdout_start)

    for t in tickers:
        if t in DATA_QUALITY_EXCLUDED_TICKERS:
            coverage["skipped"][t] = f"data-quality exclusion: {DATA_QUALITY_EXCLUDED_TICKERS[t]}"
            continue
        if t not in ohlc:
            coverage["skipped"][t] = "no OHLC data cached"
            continue
        in_sample_df = ohlc[t][ohlc[t].index < holdout_start_ts]
        if len(in_sample_df) < min_bars:
            coverage["skipped"][t] = f"fewer than {min_bars} in-sample bars ({len(in_sample_df)})"
            continue
        if t not in quotes:
            coverage["skipped"][t] = "no live cost quote available"
            continue

        signal_fn = signal_fn_for_ticker(t)
        if signal_fn is None:
            coverage["skipped"][t] = "signal_fn_for_ticker returned None (see family-specific reason)"
            continue

        coverage["attempted"] += 1
        cost_bps = cost_bps_for(t, quotes)
        tax_rate = cost_tax_inputs.tax_haircut_rate_for(ACCOUNT_TYPE)
        trades, _ = backtest_engine.run_backtest(
            in_sample_df, signal_fn, cost_bps=cost_bps, tax_haircut_rate=tax_rate,
            time_stop=time_stop)
        trades_by_ticker[t] = trades
        index_by_ticker[t] = in_sample_df.index
        coverage["ok"] += 1

    return trades_by_ticker, index_by_ticker, coverage


# Calendar-day buffer prepended to the holdout slice so both signal functions
# see a REAL trailing lookback window instead of falling back to "whatever's
# available" (the `len(frame) >= LOOKBACK_DAYS` branch of
# signals.fifty_two_week_high_momentum_signal_fn silently degrades
# otherwise: with the
# bare holdout_start:holdout_end slice, the ~126-trading-day holdout window is
# always shorter than fifty_two_week_high_momentum's 252-day lookback, so the
# fallback fires for the ENTIRE holdout period, computing "52-week high" over
# at most half a year of holdout-only data). 400 calendar days safely covers
# both 252 trading days (~365 calendar days incl. weekends/holidays) and PEAD's
# 60-day trailing window with margin.
HOLDOUT_LOOKBACK_BUFFER_DAYS = 400


def run_family_holdout(tickers_with_trades, ohlc, quotes, holdout_start, holdout_end,
                        signal_fn_for_ticker_holdout, time_stop=None):
    """Re-runs the SAME mechanism on the pinned holdout window, for
    every ticker that had at least one in-sample trade OR that the
    family wants tested regardless (PEAD passes a possibly-different
    signal_fn restricted to holdout-window earnings dates). Only called
    for a family that already passed the DSR bar. See
    run_family_in_sample's docstring for why `time_stop` matters.

    The OHLC slice handed to run_backtest is padded BACKWARD by
    HOLDOUT_LOOKBACK_BUFFER_DAYS so the signal function has real trailing
    history to compute its lookback window from -- but only trades whose
    ENTRY falls on or after holdout_start are counted as real holdout
    trades; anything the padding period itself generates is a warm-up
    artifact and dropped before returning."""
    holdout_start_ts = pd.Timestamp(holdout_start)
    holdout_end_ts = pd.Timestamp(holdout_end)
    padded_start_ts = holdout_start_ts - pd.Timedelta(days=HOLDOUT_LOOKBACK_BUFFER_DAYS)
    trades_by_ticker = {}
    index_by_ticker = {}
    for t in tickers_with_trades:
        if t not in ohlc or t not in quotes:
            continue
        padded_df = ohlc[t].loc[padded_start_ts:holdout_end_ts]
        holdout_df = ohlc[t].loc[holdout_start_ts:holdout_end_ts]
        if len(holdout_df) == 0:
            continue
        signal_fn = signal_fn_for_ticker_holdout(t)
        if signal_fn is None:
            continue
        cost_bps = cost_bps_for(t, quotes)
        tax_rate = cost_tax_inputs.tax_haircut_rate_for(ACCOUNT_TYPE)
        trades, _ = backtest_engine.run_backtest(
            padded_df, signal_fn, cost_bps=cost_bps, tax_haircut_rate=tax_rate,
            time_stop=time_stop)
        if len(trades) > 0:
            trades = trades[pd.to_datetime(trades["entry_date"]) >= holdout_start_ts].reset_index(drop=True)
        trades_by_ticker[t] = trades
        index_by_ticker[t] = holdout_df.index
    return trades_by_ticker, index_by_ticker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=os.path.join(
        REPO_ROOT, "data", "mining", "universe_snapshot_2026-08-29.json"))
    parser.add_argument("--ohlc-cache", default=os.path.join(
        REPO_ROOT, "data", "mining", "universe_cache", "ohlc_2026-08-29"))
    parser.add_argument("--cost-quotes", default=os.path.join(
        REPO_ROOT, "data", "mining", "universe_cache", "cost_quotes_merged_2026-08-29.json"))
    parser.add_argument("--earnings-cache", default=os.path.join(
        REPO_ROOT, "data", "mining", "universe_cache", "earnings_2026-08-29"))
    parser.add_argument("--run-id", default="expanded-universe")
    parser.add_argument("--tested-date", required=True,
                         help="ISO date recorded on the ledger outcome (YYYY-MM-DD).")
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "runs", "run_expanded_universe_mining",
        "expanded_universe_results.json"))
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute everything but do not write to the trial ledger.")
    args = parser.parse_args()

    with open(args.snapshot, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    universe = snapshot["tickers"]
    _log(f"Universe: {len(universe)} tickers (fetched {snapshot['fetch_date']}, "
         f"S&P 500 {snapshot['sp500_count']} raw + NASDAQ-100 {snapshot['nasdaq100_count']} "
         f"raw, {snapshot['overlap_count']} overlap)")

    ohlc = load_ohlc_cache(args.ohlc_cache, universe)
    quotes = load_cost_quotes(args.cost_quotes)
    earnings = load_earnings_cache(args.earnings_cache, universe)
    _log(f"OHLC cached: {len(ohlc)}/{len(universe)}. Cost quotes: {len(quotes)}/{len(universe)}. "
         f"Earnings-date fetch ok: {sum(1 for v in earnings.values() if v['ok'])}/{len(universe)}")

    fam_momentum = "fifty_two_week_high_momentum"
    fam_pead = "pead_earnings_surprise_drift"

    catalog = catalog_mod.load_catalog(os.path.join(REPO_ROOT, "data", "mining", "catalog.json"))
    pead_entry = next(e for e in catalog if e["family_id"] == fam_pead)
    momentum_entry = next(e for e in catalog if e["family_id"] == fam_momentum)
    pead_hold_days = int(pead_entry["parameters"]["hold_days"])
    _log(f"pead hold_days (time_stop), read from catalog.json: {pead_hold_days}")
    catalog_entry_by_family = {fam_pead: pead_entry, fam_momentum: momentum_entry}

    pinned_momentum = trial_ledger.get_pinned_holdout_window(LEDGER_PATH, fam_momentum)
    pinned_pead = trial_ledger.get_pinned_holdout_window(LEDGER_PATH, fam_pead)
    if pinned_momentum is None or pinned_pead is None:
        raise RuntimeError("Both families must already have a pinned holdout window in the "
                            "ledger -- refusing to "
                            "compute a fresh one; the pinned window must be reused.")
    _log(f"Pinned holdout window (momentum): {pinned_momentum}")
    _log(f"Pinned holdout window (pead): {pinned_pead}")
    assert pinned_momentum == pinned_pead, "unexpected: both families' pinned windows differ"
    holdout_start = pinned_momentum["holdout_start"]
    holdout_end = pinned_momentum["holdout_end"]

    # -- record_first_test, UNCONDITIONAL and FIRST (idempotent: a no-op
    # for a family that already has a first_test event, as the pinned-window
    # check above requires here; the call keeps the ordering invariant if
    # this script is reused for a brand-new family). --
    if not args.dry_run:
        trial_ledger.record_first_test(
            LEDGER_PATH, fam_momentum, args.tested_date,
            holdout_start, holdout_end, [], args.run_id)
        trial_ledger.record_first_test(
            LEDGER_PATH, fam_pead, args.tested_date,
            holdout_start, holdout_end, [], args.run_id)

    # ---- fifty_two_week_high_momentum: in-sample ----
    momentum_trades, momentum_index, momentum_cov = run_family_in_sample(
        universe, ohlc, quotes, holdout_start,
        signal_fn_for_ticker=lambda t: signals.fifty_two_week_high_momentum_signal_fn)
    _log(f"[momentum] in-sample: attempted {momentum_cov['attempted']}, "
         f"ok {momentum_cov['ok']}, skipped {len(momentum_cov['skipped'])}")

    # ---- pead_earnings_surprise_drift: in-sample ----
    holdout_start_ts = pd.Timestamp(holdout_start)
    holdout_end_ts = pd.Timestamp(holdout_end)
    pead_earnings_coverage = {"ok_fetch": 0, "no_insample_dates": 0, "fetch_failed": {}}

    def pead_signal_fn_in_sample(t):
        rec = earnings.get(t, {"ok": False, "dates": [], "reason": "not fetched"})
        if not rec["ok"]:
            pead_earnings_coverage["fetch_failed"][t] = rec["reason"]
            return None
        pead_earnings_coverage["ok_fetch"] += 1
        in_sample_dates = [d for d in rec["dates"] if pd.Timestamp(d) < holdout_start_ts]
        if not in_sample_dates:
            pead_earnings_coverage["no_insample_dates"] += 1
            return None
        return signals.make_pead_signal_fn(frozenset(in_sample_dates))

    pead_trades, pead_index, pead_cov = run_family_in_sample(
        universe, ohlc, quotes, holdout_start, signal_fn_for_ticker=pead_signal_fn_in_sample,
        time_stop=pead_hold_days)
    _log(f"[pead] in-sample: attempted {pead_cov['attempted']}, ok {pead_cov['ok']}, "
         f"skipped {len(pead_cov['skipped'])}; earnings-date coverage: "
         f"{pead_earnings_coverage['ok_fetch']} fetched ok, "
         f"{len(pead_earnings_coverage['fetch_failed'])} fetch failed, "
         f"{pead_earnings_coverage['no_insample_dates']} had zero usable in-sample dates")

    # ---- Pool trades, compute per-family in-sample stats ----
    pooled_momentum = pooling.pool_trades(momentum_trades)
    pooled_pead = pooling.pool_trades(pead_trades)
    stats_momentum = pooling.pooled_trade_stats(pooled_momentum)
    stats_pead = pooling.pooled_trade_stats(pooled_pead)
    _log(f"[momentum] pooled in-sample: {stats_momentum}")
    _log(f"[pead] pooled in-sample: {stats_pead}")

    # ---- Correlation-purpose pooled daily-return series (per family),
    # windowed, then aligned to a shared calendar for this 2-family
    # "run"'s num_independent_trials estimate -- the shared-N step
    # run_batch3_mining.compute_shared_correction makes, scoped to these two
    # families instead of a whole catalog run, with the ledger's cumulative
    # family count as the floor (both families are already in the ledger). ----
    series_momentum = pooling.pooled_daily_returns_by_exit_day(
        {t: (momentum_trades[t], momentum_index[t]) for t in momentum_trades},
        window_days=constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS
    ) if momentum_trades else pd.Series(dtype=float)
    series_pead = pooling.pooled_daily_returns_by_exit_day(
        {t: (pead_trades[t], pead_index[t]) for t in pead_trades},
        window_days=constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS
    ) if pead_trades else pd.Series(dtype=float)

    shared_index = series_momentum.index.union(series_pead.index)
    aligned_momentum = series_momentum.reindex(shared_index, fill_value=0.0).values
    aligned_pead = series_pead.reindex(shared_index, fill_value=0.0).values

    candidate_arrays = [aligned_momentum, aligned_pead]
    non_degenerate = [a for a in candidate_arrays
                       if a.size > 0 and np.std(a) != 0.0 and np.all(np.isfinite(a))]
    if non_degenerate:
        n_this_run = dsr.num_independent_trials(non_degenerate)
    else:
        n_this_run = float(len(candidate_arrays))
    ledger_floor = trial_ledger.cumulative_family_count(LEDGER_PATH)
    shared_n_trials = max(n_this_run, ledger_floor, 1.0)
    _log(f"Shared correction N: this-run correlation estimate={n_this_run:.4f}, "
         f"ledger cumulative floor={ledger_floor}, USED={shared_n_trials:.4f}")

    # sr_variance tiers (pooling is the most
    # principled choice), matching run_batch3_mining.compute_shared_correction's
    # own tier order: pooled across the whole ledger first, this run's own
    # 2-candidate sample second, the pre-registered floor last.
    pooled_variance = trial_ledger.pooled_sr_variance(LEDGER_PATH)
    if pooled_variance is not None:
        shared_sr_variance = pooled_variance
        _log(f"Shared sr_variance: {shared_sr_variance} (pooled across the ledger)")
    else:
        sharpe_values = [stats_momentum["sharpe"], stats_pead["sharpe"]]
        if len(sharpe_values) >= 2 and np.var(sharpe_values, ddof=1) > 0:
            shared_sr_variance = float(np.var(sharpe_values, ddof=1))
        else:
            shared_sr_variance = constants.DEFAULT_SR_VARIANCE_FLOOR
        _log(f"Shared sr_variance: {shared_sr_variance} (pool not yet seeded)")

    # ---- Score each family ----
    results = {}
    for family_id, stats, trades_by_ticker, index_by_ticker, cov in (
        (fam_momentum, stats_momentum, momentum_trades, momentum_index, momentum_cov),
        (fam_pead, stats_pead, pead_trades, pead_index, pead_cov),
    ):
        if stats["trade_count"] < constants.MIN_TRADES_FOR_SCORING:
            # insufficient_evidence, not a scored cut_dsr
            # (evidence lifecycle): no real DSR was ever computed on
            # this branch (it `continue`s before reaching the DSR call
            # below), so filing it as a statistical rejection would be
            # indistinguishable from a real one.
            #
            # Failure class guarded here: both families' record_first_test
            # calls already ran, UNCONDITIONALLY, before this loop even
            # starts (see the block above at "record_first_test,
            # UNCONDITIONAL and FIRST"). Every step below -- ticker/date
            # arithmetic, a real file hash, evidence.validate -- can raise,
            # and this whole `main()` carries no other exception handler,
            # so an unhandled raise here would leave this family PINNED
            # with NO outcome event at all -- exactly the gap the evidence
            # lifecycle exists to close, just relocated. Wrapped in
            # try/except below so that can never happen.
            try:
                # Not defended with a placeholder if empty: a pooled
                # universe-wide test attempting literally zero tickers has
                # no honest ticker to report, and evidence.py's own
                # validator correctly refuses an empty instrument_set
                # rather than accept an invented one -- if THIS is what's
                # empty, the ValueError below is caught by the except
                # clause and correctly recorded as not_evaluated_error.
                attempted_tickers = sorted(trades_by_ticker.keys())
                in_sample_indices = [idx for idx in index_by_ticker.values() if len(idx)]
                in_sample_start = (min(idx.min() for idx in in_sample_indices)
                                   .strftime("%Y-%m-%d") if in_sample_indices else holdout_start)
                in_sample_end = (max(idx.max() for idx in in_sample_indices)
                                 .strftime("%Y-%m-%d") if in_sample_indices else holdout_start)
                # evidence.build_insufficient_trades reads stats['sharpe_in_
                # sample']; this pipeline's own stats dicts (pooling.
                # pooled_trade_stats) key the identical per-trade value as
                # 'sharpe' -- translated here so a real measured value is
                # recorded instead of silently defaulting to None.
                stats_for_evidence = {**stats, "sharpe_in_sample": stats.get("sharpe")}
                payload = evidence.build_insufficient_trades(
                    stats=stats_for_evidence, catalog_entry=catalog_entry_by_family[family_id],
                    # A 500+-ticker pooled universe test has no realistic further
                    # widening -- "none_available" below is the honest terminal
                    # state, so the full attempted-ticker list is disclosure only.
                    instrument_set=attempted_tickers,
                    in_sample_start=in_sample_start, in_sample_end=in_sample_end,
                    pinned_window={"holdout_start": holdout_start,
                                   "holdout_end": holdout_end},
                    # This branch `continue`s before the holdout section below
                    # ever runs (it is gated `if passed_dsr:`), so the holdout
                    # was never looked at for this outcome.
                    holdout_look="unspent",
                    signal_source_path="src/mining/signals.py",
                    signal_source_sha256=evidence.source_sha256(
                        os.path.join(_SRC, "mining", "signals.py")),
                    evidence_source="measured",
                    reactivation={
                        "kind": "none_available",
                        "rationale": "pooled across the full S&P 500 + NASDAQ "
                                     "100 universe already -- no realistic "
                                     "further widening; the family stays open "
                                     "to a fresh look via a new family_id",
                    })
                results[family_id] = {
                    "stage": "insufficient_evidence", "dsr_value": None,
                    "reason": f"only {stats['trade_count']} pooled in-sample trades, below "
                              f"MIN_TRADES_FOR_SCORING ({constants.MIN_TRADES_FOR_SCORING})",
                    "stats": stats, "holdout": None, "evidence": payload,
                }
            except Exception as exc:
                # Sourced from the FULL universe list (loaded unconditionally
                # at the top of main(), never empty in real usage), not
                # trades_by_ticker -- the latter can itself be the empty
                # collection that caused the ValueError being caught here,
                # and an empty instrument_set would make this defensive
                # record raise a SECOND time (the same failure mode,
                # applied to this catch instead of a helper function).
                error_payload = evidence.build_engine_error(
                    error=exc, instrument_set=sorted(universe),
                    pinned_window={"holdout_start": holdout_start,
                                   "holdout_end": holdout_end},
                    holdout_look="unspent")
                results[family_id] = {
                    "stage": "not_evaluated_error", "dsr_value": None,
                    "reason": f"{type(exc).__name__} while building the "
                              f"insufficient-evidence record: {exc}",
                    "stats": stats, "holdout": None, "evidence": error_payload,
                }
            continue

        dsr_value = dsr.deflated_sharpe_ratio(
            sr=stats["sharpe"], n=stats["trade_count"], skew=stats["skew"],
            kurtosis=stats["kurtosis"], n_trials=shared_n_trials,
            sr_variance=shared_sr_variance)
        passed_dsr = dsr_value >= constants.DSR_PASS_THRESHOLD

        holdout_result = None
        passed_holdout = False
        if passed_dsr:
            if family_id == fam_momentum:
                holdout_fn = lambda t: signals.fifty_two_week_high_momentum_signal_fn
            else:
                def holdout_fn(t, _earnings=earnings):
                    rec = _earnings.get(t, {"ok": False, "dates": []})
                    if not rec["ok"]:
                        return None
                    hdates = [d for d in rec["dates"]
                              if holdout_start_ts <= pd.Timestamp(d) <= holdout_end_ts]
                    if not hdates:
                        return None
                    return signals.make_pead_signal_fn(frozenset(hdates))
            ho_time_stop = None if family_id == fam_momentum else pead_hold_days
            ho_trades, ho_index = run_family_holdout(
                list(trades_by_ticker.keys()), ohlc, quotes, holdout_start, holdout_end, holdout_fn,
                time_stop=ho_time_stop)
            if ho_trades:
                pooled_ho = pooling.pool_trades(ho_trades)
            else:
                pooled_ho = pd.DataFrame({"net_ret": pd.Series(dtype="float64"),
                                           "ticker": pd.Series(dtype="object")})
            holdout_net_edge_bps = float(pooled_ho["net_ret"].mean() * 1e4) if len(pooled_ho) else 0.0
            passed_holdout = backtest_engine.holdout_passes(stats["net_edge_bps"], holdout_net_edge_bps)
            holdout_result = {
                "holdout_trade_count": int(len(pooled_ho)),
                "holdout_ticker_count": int(pooled_ho["ticker"].nunique()) if len(pooled_ho) else 0,
                "holdout_net_edge_bps": holdout_net_edge_bps,
                "in_sample_net_edge_bps": stats["net_edge_bps"],
                "passed_holdout": passed_holdout,
            }

        stage = "survived" if (passed_dsr and passed_holdout) else (
            "cut_holdout" if passed_dsr else "cut_dsr")
        if stage == "cut_dsr":
            reason = (f"DSR {dsr_value:.4f} below the {constants.DSR_PASS_THRESHOLD} bar "
                      f"(n_trials={shared_n_trials:.2f})")
        elif stage == "cut_holdout":
            reason = (f"passed DSR ({dsr_value:.4f}) but holdout net edge "
                      f"{holdout_result['holdout_net_edge_bps']:.1f} bps did not confirm the "
                      f"in-sample {holdout_result['in_sample_net_edge_bps']:.1f} bps edge")
        else:
            reason = None
        results[family_id] = {
            "stage": stage, "dsr_value": dsr_value, "passed_dsr": passed_dsr,
            "passed_holdout": passed_holdout, "stats": stats,
            "holdout": holdout_result, "coverage": cov, "reason": reason,
        }
        _log(f"[{family_id}] DSR={dsr_value:.4f} passed_dsr={passed_dsr} "
             f"passed_holdout={passed_holdout} stage={stage}")

    # ---- Record outcomes (exactly one new event per family) ----
    universe_label = f"{len(universe)} tickers (S&P 500 + NASDAQ 100 pool, deduped, fetched {snapshot['fetch_date']})"
    for family_id in (fam_momentum, fam_pead):
        r = results[family_id]
        note = (
            f"Pooled-universe test ({args.tested_date}): pooled trade-level test across the "
            f"S&P 500 + NASDAQ 100 universe. Pooled {r['stats']['trade_count']} trades from "
            f"{r['stats']['ticker_count']} tickers with >=1 trade "
            f"(pooling methodology: mining.pooling module docstring). "
            f"Coverage: {r.get('coverage', {}).get('ok', 'n/a')} tickers scored. "
            + (f"Holdout: {r['holdout']['holdout_trade_count']} pooled holdout trades, "
               f"net edge {r['holdout']['holdout_net_edge_bps']:.1f} bps vs in-sample "
               f"{r['holdout']['in_sample_net_edge_bps']:.1f} bps. "
               if r.get("holdout") else "Holdout not reached (cut at DSR bar). ")
            + f"Reason if cut: {r.get('reason', 'n/a')}. "
            "This is a narrower single-signal timing test per ticker, not a replication of the "
            "literature's cross-sectional decile-ranking construction -- see the catalog's PROXY/"
            "REFORMULATION notes."
        )
        # sharpe_in_sample carried only when a real DSR was computed -- see
        # record_outcome's own docstring on why a fabricated sharpe on an
        # unscored outcome would corrupt trial_ledger.pooled_sr_variance.
        sharpe_in_sample = (r["stats"]["sharpe"]
                            if r["dsr_value"] is not None else None)
        if not args.dry_run:
            trial_ledger.record_outcome(
                LEDGER_PATH, family_id, args.run_id, args.tested_date, r["stage"],
                dsr_value=r["dsr_value"], note=note, ticker_label=universe_label,
                sharpe_in_sample=sharpe_in_sample, evidence=r.get("evidence"))
            _log(f"Recorded outcome for {family_id}: stage={r['stage']}")
        else:
            _log(f"[DRY RUN] would record outcome for {family_id}: stage={r['stage']}")

    # ---- Write full results JSON (evidence artifact) ----
    def _serialize(obj):
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    out_payload = {
        "run_id": args.run_id, "tested_date": args.tested_date,
        "universe_size": len(universe), "universe_source": snapshot,
        "holdout_start": holdout_start, "holdout_end": holdout_end,
        "shared_n_trials": shared_n_trials, "shared_sr_variance": shared_sr_variance,
        "coverage": {
            "ohlc_cached": len(ohlc), "cost_quotes": len(quotes),
            "earnings_fetch_ok": sum(1 for v in earnings.values() if v["ok"]),
            "momentum_coverage": momentum_cov,
            "pead_coverage": pead_cov,
            "pead_earnings_coverage": pead_earnings_coverage,
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "coverage"} for k, v in results.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_serialize(out_payload), f, indent=2, default=str)
    _log(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
