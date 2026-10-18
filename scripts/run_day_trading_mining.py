"""Orchestrating mining-run script for the DAY-TRADING track of the
intraday/swing strategy-mining extension.

This is the day-trading sibling of scripts/run_swing_mining.py, itself the
intraday-track analogue of scripts/run_batch3_mining.py. It reuses the
intraday library modules (intraday_data.py,
intraday_backtest_engine.py, intraday_cost_gate.py, intraday_holdout.py,
plus the shared catalog/ledger/DSR/results-log machinery) exactly as
run_swing_mining.py does -- see that script's own module docstring for the
"what is reused, unmodified, via import -- and why it is safe" analysis,
which applies here byte-for-byte (this script imports the SAME functions
from the SAME modules).

## End-of-day flat needs no separate engine module

A hard end-of-day flat constraint needs no day-trading engine variant: it
follows from `run_intraday_backtest`'s own force-close logic
(`src/mining/intraday_backtest_engine.py`):

    hit_time_stop = is_last_bar_of_day[i] and days_held_inclusive >= max_hold_days

With `max_hold_days=1`, `days_held_inclusive >= 1` is true from the entry
bar onward, so the ONLY bar this can fire on is the LAST bar of the entry
day itself (`is_last_bar_of_day[i]` is only True there) -- a position
entered at ANY time during a session, with no earlier signal exit, is
force-closed at that SAME session's own last bar. This is exactly the
DAY-order-only broker constraint (flat by the close, no overnight
exposure), covered by the engine's general `max_hold_days` cap and its
regression test (`test_max_hold_days_one_force_closes_same_bar_as_entry`)
-- not a special case added for this
script. So: **every day-trading candidate simply sets `max_hold_days=1`**;
`_validate_day_trading_candidate` below enforces this mechanically (fails
loud on any candidate that doesn't) rather than trusting curation to get
it right.

## Batching rule: one trade-frequency track per correlation call -- load-bearing

`compute_shared_correction` estimates the correlation among a batch's
candidates from a SINGLE run's own aligned daily-return series. Mixing
trade-frequency tracks (daily vs. swing vs. day) in one correlation call
would read a density mismatch as spurious near-independence and understate
how stringent the DSR bar should be. Every entry point
below (`compute_day_trading_candidates`, `run_day_trading_batch`,
`execute_day_trading_run`) calls `compute_shared_correction`/`score_batch`
EXACTLY ONCE per invocation, over ONLY the day-trading candidates passed
into THAT SAME call -- the identical guarantee run_swing_mining.py makes
for its own track. A caller who wants to score a daily-track batch, a
swing-track batch, AND a day-trading-track batch for the same
`tested_date`/`run_id` MUST call all three scripts' functions SEPARATELY
-- never by concatenating their `computed` lists before any one call.

## Cost-gate sequencing -- identical to run_swing_mining.py

Same "run in-sample first, gate on that, only then pay for the padded
holdout fetch" sequencing as the swing script, for the same reasons (see
that script's own docstring section "Cost-gate sequencing" -- this
script's `compute_day_trading_candidates` implements the identical
sequence).

## Shared, timeframe-agnostic helpers reused UNRENAMED from intraday_holdout.py

`intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS` and
`intraday_holdout.pool_swing_daily_returns` are imported here under their
existing names despite saying "swing" -- both are timeframe-agnostic in
behavior (a generic lookback-day count and a generic per-ticker
daily_returns union+sum respectively, with no swing-specific logic in
either body). The reuse is deliberate, not an oversight.

## Scope note

This script is built to run for real.
`main()` has no default candidate slate -- the real day-trading run
supplies its own real `candidates`/`quotes` and calls
`execute_day_trading_run()` directly, matching both sibling scripts' CLI
shape without a static default to accidentally run.

Additive only -- never imports or modifies `run_batch3_mining.py`,
`run_swing_mining.py`, `catalog.py`, `trial_ledger.py`, `results_log.py`,
`dsr.py`, `constants.py`, `pooling.py`, or any `intraday_*` module.
`trial_ledger.py`'s cross-process `_LedgerLock` (see that module's own
docstring) already covers this script's `record_first_test`/
`record_outcome` calls -- no locking work needed here, same as the swing
script.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import trading_calendar  # noqa: E402  (sys.path must be set first)
import run_batch3_mining as batch3  # noqa: E402
from mining import constants  # noqa: E402
from mining import intraday_cost_gate  # noqa: E402
from mining import intraday_data  # noqa: E402
from mining import intraday_holdout  # noqa: E402
from mining import trial_ledger  # noqa: E402
from mining.intraday_backtest_engine import run_intraday_backtest  # noqa: E402

# The SAME shared ledger/catalog/results-log paths as the daily and swing
# pipelines (one statistical pool, not three) -- re-exported
# from batch3 rather than redefined, so all three scripts can never
# silently drift apart on which files they consider authoritative.
LEDGER_PATH = batch3.LEDGER_PATH
CATALOG_PATH = batch3.CATALOG_PATH
RESEARCH_ROOT = batch3.RESEARCH_ROOT
RESULTS_LOG_PATH = batch3.RESULTS_LOG_PATH
NEAR_MISS_LOG_PATH = batch3.NEAR_MISS_LOG_PATH

_REQUIRED_CANDIDATE_FIELDS = (
    "family_id", "tickers", "signal_fn", "cost_bps", "tax_haircut_rate",
    "max_hold_days", "direction", "account_type", "in_sample_start",
    "in_sample_end",
)


def _validate_day_trading_candidate(candidate: dict) -> None:
    missing = [f for f in _REQUIRED_CANDIDATE_FIELDS if f not in candidate]
    if missing:
        raise ValueError(
            f"candidate {candidate.get('family_id', '?')!r} missing "
            f"required field(s): {missing}")
    if not candidate["tickers"]:
        raise ValueError(
            f"candidate {candidate['family_id']!r}: tickers must be a "
            f"non-empty list")
    # The mechanical enforcement of "flat by the close, every trade" --
    # see the module docstring's "End-of-day flat" section for
    # why max_hold_days=1 on the EXISTING engine already IS the EOD-flat
    # constraint. A day-trading candidate with any other max_hold_days
    # would silently become a (mis-tracked) swing candidate on this
    # track's own ledger pool -- fail loud rather than let that happen.
    if candidate["max_hold_days"] != 1:
        raise ValueError(
            f"candidate {candidate['family_id']!r}: day-trading candidates "
            f"must set max_hold_days=1 (this is what forces the position "
            f"flat at the entry session's own last bar -- see module "
            f"docstring), got {candidate['max_hold_days']!r}")


def _ticker_label(candidate: dict) -> str:
    return candidate.get("ticker_label") or "/".join(candidate["tickers"])


def _backtest_tickers(
    tickers: list, start_date: str, end_date: str, *, signal_fn, cost_bps,
    tax_haircut_rate, max_hold_days, direction, fetch_fn: Callable,
) -> dict:
    """Fetch + backtest every ticker in `tickers` over
    `[start_date, end_date]`. Returns `{ticker: (trades_df, daily_returns)}`.
    A ticker with a real data gap (fetch_fn raising) is NOT caught here --
    matching intraday_data.py's own "fail-closed, caller decides the
    skip-and-log policy" contract -- it propagates up to the caller of
    compute_day_trading_candidates."""
    out = {}
    for ticker in tickers:
        bars = fetch_fn(ticker, start_date, end_date)
        trades, daily = run_intraday_backtest(
            bars, signal_fn, cost_bps=cost_bps,
            tax_haircut_rate=tax_haircut_rate, max_hold_days=max_hold_days,
            direction=direction)
        out[ticker] = (trades, daily)
    return out


def _expected_holdout_sessions(holdout_start: str, holdout_end: str) -> list:
    """Every real NYSE trading session (`datetime.date`, ascending) within
    `[holdout_start, holdout_end]` -- identical to run_swing_mining.py's
    own helper of the same name (reuses the SAME vendored NYSE calendar,
    `trading_calendar.load_default_sessions()`); duplicated here rather than
    imported cross-script to keep this script's own dependency surface
    limited to library modules, matching run_swing_mining.py's own
    "additive only" convention against run_batch3_mining.py.

    Raises ValueError if no real NYSE session falls within
    `[holdout_start, holdout_end]`.
    """
    start = pd.Timestamp(holdout_start).date()
    end = pd.Timestamp(holdout_end).date()
    sessions = trading_calendar.load_default_sessions()
    covered = [session for session in sessions if start <= session <= end]
    if not covered:
        raise ValueError(
            f"_expected_holdout_sessions: no real NYSE trading session "
            f"between {start} and {end}")
    return covered


def compute_day_trading_candidates(
    candidates: list,
    *,
    quotes: dict,
    tested_date: str,
    ledger_path: str,
    fetch_fn: Callable = intraday_data.fetch_intraday_bars,
    lookback_buffer_days: int = intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS,
    correlation_window_days: int = constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS,
) -> dict:
    """Pure compute phase -- READS the ledger (only `trial_ledger.
    has_been_tested`, to exclude already-pinned families before doing any
    work for them) but never WRITES to it. For every not-already-tested
    candidate: run its in-sample backtest (pooled across its own tickers),
    gate on the measured net edge, and -- only for a gate PASS -- fetch a
    lookback-padded holdout window, backtest it, reconcile down to the
    real holdout window (`intraday_holdout.reconcile_holdout_window`), and
    pool that too. Identical control flow to run_swing_mining.py's own
    `compute_swing_candidates` (see that function's docstring for the full
    "why" behind each guard below, all of which apply unchanged to this
    track) -- the only behavioral difference between the two tracks is
    that every candidate here is mechanically validated to carry
    `max_hold_days=1` (`_validate_day_trading_candidate`), which is what
    makes the underlying `run_intraday_backtest` calls force-close flat by
    each entry session's own close.

    Returns `{"window": ..., "computed": [...], "gate_exclusions": [...]}`.
    """
    window = batch3.derive_holdout_window(tested_date)
    expected_holdout_sessions = _expected_holdout_sessions(
        window["holdout_start"], window["holdout_end"])
    computed: list = []
    gate_exclusions: list = []

    # Uniqueness guard -- identical rationale to run_swing_mining.py's own
    # (a duplicate family_id in one candidates list would let score_batch's
    # pin loop abort the whole batch and defensively cut every
    # already-pinned family, including unrelated ones).
    family_ids = [c.get("family_id") for c in candidates]
    if len(set(family_ids)) != len(family_ids):
        seen, duplicates = set(), set()
        for fid in family_ids:
            (duplicates if fid in seen else seen).add(fid)
        raise ValueError(
            f"compute_day_trading_candidates: duplicate family_id(s) in "
            f"candidates list: {sorted(duplicates)} -- score_batch's pin "
            f"loop cannot handle two candidates sharing one family_id in "
            f"the same batch (a second pin aborts the whole batch and "
            f"defensively cuts every already-pinned family, including "
            f"unrelated ones)")

    for candidate in candidates:
        _validate_day_trading_candidate(candidate)
        family_id = candidate["family_id"]
        tickers = candidate["tickers"]
        signal_fn = candidate["signal_fn"]
        cost_bps = candidate["cost_bps"]
        tax_haircut_rate = candidate["tax_haircut_rate"]
        max_hold_days = candidate["max_hold_days"]
        direction = candidate["direction"]

        # Mirrors run_swing_mining.py's own identical rule: a not_evaluated_error
        # family is a caught DEFECT, not evidence about the market
        # (evidence.build_engine_error's own docstring) -- it stays
        # eligible to re-enter scoring once the defect is fixed, matching
        # ruled_out.filter_eligible's own treatment. Every OTHER outcome
        # (a real score, or insufficient_evidence) still excludes
        # unconditionally.
        if trial_ledger.has_been_tested(ledger_path, family_id):
            outcome = trial_ledger.latest_outcome(ledger_path, family_id)
            if outcome != "not_evaluated_error":
                gate_exclusions.append({
                    "family_id": family_id, "ticker": _ticker_label(candidate),
                    "reason": (
                        f"{family_id!r} is already pinned in the trial ledger "
                        f"-- its holdout window is fixed from its first test "
                        f"and it is not re-scored (outcome: {outcome!r})"),
                    "spread_bps": None, "edge_over_cost_ratio": None,
                    "stats_in_sample": None,
                })
                continue
            # A not_evaluated_error retry is safe ONLY against the SAME
            # pinned window the earlier failed attempt used -- see
            # run_swing_mining.py's identical guard for the full "why".
            pinned = trial_ledger.get_pinned_holdout_window(ledger_path, family_id)
            if (pinned["holdout_start"] != window["holdout_start"] or
                    pinned["holdout_end"] != window["holdout_end"]):
                gate_exclusions.append({
                    "family_id": family_id, "ticker": _ticker_label(candidate),
                    "reason": (
                        f"{family_id!r} previously recorded not_evaluated_error, "
                        f"but its pinned window ({pinned['holdout_start']}.."
                        f"{pinned['holdout_end']}) is a window mismatch against "
                        f"this run's window ({window['holdout_start']}.."
                        f"{window['holdout_end']}) -- refusing to score against "
                        f"a different window than the one the ledger says was "
                        f"tested"),
                    "spread_bps": None, "edge_over_cost_ratio": None,
                    "stats_in_sample": None,
                })
                continue
            # Falls through: same window, safe to retry.

        if pd.Timestamp(candidate["in_sample_end"]) >= pd.Timestamp(window["holdout_start"]):
            raise ValueError(
                f"candidate {family_id!r}: in_sample_end "
                f"{candidate['in_sample_end']} must be strictly before "
                f"holdout_start {window['holdout_start']} -- an in-sample "
                f"window that reaches into or past the holdout window is a "
                f"look-ahead leak, not a valid train/holdout split")

        if family_id not in quotes:
            gate_exclusions.append({
                "family_id": family_id, "ticker": _ticker_label(candidate),
                "reason": (
                    f"no live quote captured for {family_id!r} -- the "
                    f"cost-realism gate cannot be evaluated without one"),
                "spread_bps": None, "edge_over_cost_ratio": None,
                "stats_in_sample": None,
            })
            continue

        try:
            in_sample = _backtest_tickers(
                tickers, candidate["in_sample_start"], candidate["in_sample_end"],
                signal_fn=signal_fn, cost_bps=cost_bps,
                tax_haircut_rate=tax_haircut_rate, max_hold_days=max_hold_days,
                direction=direction, fetch_fn=fetch_fn)
            in_sample_trades_by_ticker = {t: pair[0] for t, pair in in_sample.items()}
            in_sample_daily_by_ticker = {t: pair[1] for t, pair in in_sample.items()}

            pooled_in_sample_trades = intraday_holdout.pool_intraday_trades(
                in_sample_trades_by_ticker)
            stats_in_sample = intraday_holdout.intraday_trade_stats(
                pooled_in_sample_trades)
            pooled_in_sample_daily = intraday_holdout.pool_swing_daily_returns(
                in_sample_daily_by_ticker)
            windowed_daily_returns = pooled_in_sample_daily.tail(correlation_window_days)

            quote = quotes[family_id]
            gate_result = intraday_cost_gate.evaluate_cost_realism_gate(
                entry_bid=quote["entry_bid"], entry_ask=quote["entry_ask"],
                exit_bid=quote["exit_bid"], exit_ask=quote["exit_ask"],
                net_edge_bps=stats_in_sample["net_edge_bps"])

            if not gate_result["proceed"]:
                gate_exclusions.append({
                    "family_id": family_id, "ticker": _ticker_label(candidate),
                    "reason": gate_result["reason"],
                    "spread_bps": gate_result["spread_bps"],
                    "edge_over_cost_ratio": gate_result["edge_over_cost_ratio"],
                    "stats_in_sample": stats_in_sample,
                })
                continue

            padded_start = (
                pd.Timestamp(window["holdout_start"])
                - pd.Timedelta(days=lookback_buffer_days)
            ).strftime("%Y-%m-%d")

            holdout_trades_by_ticker = {}
            incomplete_reason = None
            holdout_start_date = pd.Timestamp(window["holdout_start"]).date()
            holdout_end_date = pd.Timestamp(window["holdout_end"]).date()
            for ticker in tickers:
                padded_bars = fetch_fn(ticker, padded_start, window["holdout_end"])
                observed_sessions = set(
                    d for d in pd.unique(padded_bars.index.date)
                    if holdout_start_date <= d <= holdout_end_date
                ) if len(padded_bars) else set()
                missing = sorted(set(expected_holdout_sessions) - observed_sessions)
                if missing:
                    preview = ", ".join(d.isoformat() for d in missing[:3])
                    if len(missing) > 3:
                        preview += f", ... ({len(missing)} total)"
                    incomplete_reason = (
                        f"{ticker}: fetched holdout data is missing "
                        f"{len(missing)} expected real trading session(s) "
                        f"within [{window['holdout_start']}, "
                        f"{window['holdout_end']}]: {preview} -- "
                        f"incomplete/truncated data (a real gap, whether at "
                        f"the tail or in the interior), not scored as a "
                        f"complete holdout evaluation")
                    break
                padded_trades, padded_daily = run_intraday_backtest(
                    padded_bars, signal_fn, cost_bps=cost_bps,
                    tax_haircut_rate=tax_haircut_rate,
                    max_hold_days=max_hold_days, direction=direction)
                real_trades, _real_daily = intraday_holdout.reconcile_holdout_window(
                    padded_trades, padded_daily, padded_bars,
                    window["holdout_start"], window["holdout_end"],
                    cost_bps=cost_bps, tax_haircut_rate=tax_haircut_rate,
                    direction=direction)
                holdout_trades_by_ticker[ticker] = real_trades

            if incomplete_reason is not None:
                gate_exclusions.append({
                    "family_id": family_id, "ticker": _ticker_label(candidate),
                    "reason": incomplete_reason,
                    "spread_bps": None, "edge_over_cost_ratio": None,
                    "stats_in_sample": stats_in_sample,
                })
                continue

            pooled_holdout_trades = intraday_holdout.pool_intraday_trades(
                holdout_trades_by_ticker)
            stats_holdout = intraday_holdout.intraday_trade_stats(pooled_holdout_trades)
        except Exception as exc:
            # Deliberately record ONLY the exception's TYPE name, never
            # `str(exc)` -- same PII-safety rationale as
            # run_swing_mining.py's identical guard (see that script's own
            # comment at this exact point for the full "why").
            gate_exclusions.append({
                "family_id": family_id, "ticker": _ticker_label(candidate),
                "reason": (
                    f"{type(exc).__name__} raised during fetch/backtest "
                    f"for {family_id!r} -- message text withheld from "
                    f"this durable record; re-run interactively against "
                    f"the same inputs for full detail"),
                "spread_bps": None, "edge_over_cost_ratio": None,
                "stats_in_sample": None,
            })
            continue

        computed.append({
            "family_id": family_id,
            "ticker": _ticker_label(candidate),
            # `tickers` is the REAL
            # instrument list, independent of `ticker` above -- mirrors
            # run_swing_mining.py's own identical field.
            "tickers": list(tickers),
            "holdout_start": window["holdout_start"],
            "holdout_end": window["holdout_end"],
            "holdout_complete": True,
            "stats_in_sample": stats_in_sample,
            "stats_holdout": stats_holdout,
            "holdout_net_edge_bps": stats_holdout["net_edge_bps"],
            "daily_returns": windowed_daily_returns,
            "account_type": candidate["account_type"],
            "tax_haircut_rate": tax_haircut_rate,
            "cost_bps": cost_bps,
            "cost_gate": gate_result,
            "note": candidate.get("note", ""),
            # `raw_catalog_parameters` -- mirrors run_swing_mining.py's own
            # identical field.
            "raw_catalog_parameters": dict(candidate.get("parameters") or {}),
            # Passed through from the candidate's own already-validated,
            # already-REQUIRED fields -- same rationale as run_swing_mining.py's identical
            # fields (this script also imports batch3._score_one
            # UNMODIFIED).
            "in_sample_start": candidate["in_sample_start"],
            "in_sample_end": candidate["in_sample_end"],
        })

    return {"window": window, "computed": computed,
            "gate_exclusions": gate_exclusions}


def run_day_trading_batch(
    candidates: list,
    *,
    quotes: dict,
    tested_date: str,
    run_id: str,
    ledger_path: str,
    dry_run: bool,
    fetch_fn: Callable = intraday_data.fetch_intraday_bars,
    lookback_buffer_days: int = intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS,
    correlation_window_days: int = constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS,
) -> dict:
    """Compute + gate + score ONE day-trading-track run: calls
    `compute_shared_correction`/`score_batch` (imported unmodified from
    `run_batch3_mining`) EXACTLY ONCE, over ONLY this call's own
    gate-passing day-trading candidates (one trade-frequency track per
    correlation call -- see module docstring). Identical contract to run_swing_mining.py's own
    `run_swing_batch`, stamped `"timeframe": "day"` instead of `"swing"`."""
    batch = compute_day_trading_candidates(
        candidates, quotes=quotes, tested_date=tested_date,
        ledger_path=ledger_path, fetch_fn=fetch_fn,
        lookback_buffer_days=lookback_buffer_days,
        correlation_window_days=correlation_window_days)
    correction = batch3.compute_shared_correction(batch["computed"], ledger_path)
    scored = batch3.score_batch(
        batch["computed"], ledger_path=ledger_path, run_id=run_id,
        tested_date=tested_date, dry_run=dry_run, correction=correction)
    return {
        "schema_version": 1, "run_id": run_id, "tested_date": tested_date,
        "dry_run": dry_run, "timeframe": "day", **batch["window"],
        "shared_correction": correction,
        "results": [batch3.public_result(item) for item in scored],
        "gate_exclusions": batch["gate_exclusions"],
    }


def _file_sha256(path: str) -> str | None:
    """Identical to run_swing_mining.py's own private helper of the same
    name -- reimplemented locally rather than imported cross-script, same
    "additive only" convention."""
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    """Identical to run_swing_mining.py's own private helper of the same
    name -- see that script's docstring for the full "why" (non-finite
    floats from a zero-measured spread's edge_over_cost_ratio must not
    crash json.dumps(allow_nan=False))."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _atomic_write_text(path: str, text: str) -> None:
    """Identical to run_swing_mining.py's own private helper of the same
    name (same atomic-write convention as run_batch3_mining._atomic_write_text)."""
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def execute_day_trading_run(
    candidates: list,
    *,
    quotes: dict,
    tested_date: str,
    run_id: str,
    ledger_path: str,
    dry_run: bool,
    output_path: str,
    catalog_path: str,
    research_root: str,
    results_log_path: str,
    near_miss_log_path: str | None = None,
    fetch_fn: Callable = intraday_data.fetch_intraday_bars,
    lookback_buffer_days: int = intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS,
    correlation_window_days: int = constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS,
    batch_runner: Callable | None = None,
    results_renderer: Callable | None = None,
    near_miss_renderer: Callable | None = None,
) -> dict:
    """Run once and refresh durable evidence after every real ledger
    change -- identical contract to run_swing_mining.py's own
    `execute_swing_run` (see that function's docstring for the full "why"):
    regenerate `RESULTS-LOG.md` AND `NEAR-MISS-LOG.md` (via batch3's own
    `regenerate_near_miss_log`) if and only if `not dry_run`
    AND the ledger file's sha256 actually changed, in a `finally`-shaped
    block that ALSO regenerates on a run-time exception, raising an
    `ExceptionGroup` if BOTH the run and the regeneration fail.
    `near_miss_log_path` defaults to a sibling of `results_log_path` when
    omitted, so a caller that sets only `results_renderer`/
    `results_log_path` still gets the near-miss log refreshed.
    """
    runner = batch_runner or run_day_trading_batch
    renderer = results_renderer or batch3.regenerate_results_log
    nm_renderer = near_miss_renderer or batch3.regenerate_near_miss_log
    nm_log_path = near_miss_log_path or os.path.join(
        os.path.dirname(results_log_path) or ".", "NEAR-MISS-LOG.md")
    ledger_before = _file_sha256(ledger_path)

    def refresh_durable_evidence() -> None:
        ledger_after = _file_sha256(ledger_path)
        if not dry_run and ledger_after != ledger_before:
            renderer(catalog_path=catalog_path, ledger_path=ledger_path,
                      research_root=research_root, out_path=results_log_path)
            nm_renderer(ledger_path=ledger_path, out_path=nm_log_path)

    try:
        payload = runner(
            candidates, quotes=quotes, tested_date=tested_date, run_id=run_id,
            ledger_path=ledger_path, dry_run=dry_run, fetch_fn=fetch_fn,
            lookback_buffer_days=lookback_buffer_days,
            correlation_window_days=correlation_window_days)
        _atomic_write_text(
            output_path,
            json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    except BaseException as run_error:
        try:
            refresh_durable_evidence()
        except BaseException as regeneration_error:
            group_type = (
                ExceptionGroup
                if isinstance(run_error, Exception)
                and isinstance(regeneration_error, Exception)
                else BaseExceptionGroup)
            raise group_type(
                "day-trading batch run and results-log regeneration both failed",
                [run_error, regeneration_error]) from None
        raise
    refresh_durable_evidence()
    return payload


def main() -> int:
    """CLI entry point, mirroring run_batch3_mining.main()'s and
    run_swing_mining.main()'s own shape. The real day-trading mining run
    supplies its own real `candidates`/`quotes` and calls
    `execute_day_trading_run()` directly -- there is no static default
    candidate list here to accidentally run."""
    raise SystemExit(
        "run_day_trading_mining.main() has no default candidate slate to "
        "run -- the real day-trading mining run supplies its own "
        "candidates/quotes and calls execute_day_trading_run() directly. "
        "This entry point exists for structural parity with "
        "run_batch3_mining.py's and run_swing_mining.py's CLI shape, not "
        "for standalone invocation.")


if __name__ == "__main__":
    raise SystemExit(main())
