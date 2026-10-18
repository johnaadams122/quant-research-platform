"""Orchestrating mining-run script for the intraday/swing strategy-mining
extension.

This is the intraday-track analogue of scripts/run_batch3_mining.py: the
intraday pieces are isolated library modules (intraday_data.py,
intraday_backtest_engine.py, intraday_cost_gate.py, intraday_holdout.py,
plus the shared catalog/ledger/DSR/results-log machinery); this script is
what runs them end-to-end against the ledger.

## What is reused, unmodified, via import -- and why it is safe

`derive_holdout_window`, `compute_shared_correction`, `score_batch` (and,
transitively, `_score_one`/`_record_defensive_outcomes`) are imported
directly from `run_batch3_mining` and called exactly as that script calls
them. None of them contains a daily-bar-specific or rotation-engine-specific
assumption:

- `derive_holdout_window(tested_date)` only touches `tested_date` and
  `constants.HOLDOUT_WINDOW_MONTHS`/the vendored NYSE calendar -- nothing
  about bar frequency or asset class.
- `compute_shared_correction(computed, ledger_path)` operates purely on the
  generic `computed` item shape (`family_id`, `daily_returns`,
  `stats_in_sample`) and `trial_ledger.cumulative_family_count` -- no
  reference to `rotation_engine`/`backtest_engine`/OHLC frames anywhere in
  its body.
- `score_batch`/`_score_one`/`_record_defensive_outcomes` operate on the
  same generic item shape plus `holdout_complete`/`holdout_start`/
  `holdout_end`/`holdout_net_edge_bps` -- again no daily-bar assumption.

This script is therefore responsible ONLY for producing `computed` items in
the exact shape those functions already expect, using the intraday engine
(`mining.intraday_backtest_engine`), the cost-realism gate
(`mining.intraday_cost_gate`), and the holdout-boundary reconciliation
(`mining.intraday_holdout`) instead of the daily pipeline's OHLC loader /
`rotation_engine`/`backtest_engine` / `_mark_to_market_spanning_trades`.

## Batching rule: one trade-frequency track per correlation call

`compute_shared_correction` estimates the correlation among a batch's
candidates from a SINGLE run's own aligned daily-return series. Mixing
trade-frequency tracks (daily vs. swing vs. day) in one correlation call
would read a density mismatch as spurious near-independence and understate
how stringent the DSR bar should be. Every scoring entry point
below (`run_swing_batch`, and `execute_swing_run` through it) calls
`compute_shared_correction`/`score_batch` EXACTLY ONCE per invocation,
over ONLY the swing candidates passed into THAT SAME call
(`compute_swing_candidates` only builds the `computed` items and calls
neither). There is no
parameter anywhere in this module that accepts a pre-built `computed` list
from another track, and no code path here ever reads
`run_batch3_mining`'s own ledger-write functions with a MIXED item list --
a caller cannot accidentally violate this rule through this module's own API. A
caller who wants to score a daily-track batch and a swing-track batch for
the same `tested_date`/`run_id` MUST call this module's functions and
`run_batch3_mining.run_batch`/`execute_batch_run` SEPARATELY -- never by
concatenating their `computed` lists before either call.

## Cost-gate sequencing

The cost-realism gate (`intraday_cost_gate.evaluate_cost_realism_gate`) needs
a candidate's own measured `net_edge_bps` BEFORE it can decide whether that
candidate may proceed to the (expensive) holdout evaluation -- but that
number has to come from SOMEWHERE. This script's answer: run the candidate's
FULL in-sample backtest FIRST (over `in_sample_start`..`in_sample_end`, which
never touches the holdout window -- no look-ahead), pool it across the
candidate's tickers, and gate on THAT pooled `net_edge_bps` against the
pre-captured live quote. Only a gate PASS proceeds to the padded-window
holdout fetch + backtest + reconciliation.

Why this, and not a separate "quick" approximation:
1. **No invented number.** The in-sample backtest is the SAME real engine,
   SAME cost/tax formula, SAME trade sample that `stats_in_sample` needs
   regardless of the gate's outcome (`score_batch` requires a computed
   item's `stats_in_sample` unconditionally). There is no second, cheaper
   "quick" estimate to build, maintain, or risk drifting from the real one:
   the in-sample run is the comparatively quick half (it never fetches the
   lookback-padded holdout window) and the holdout run is the genuinely
   "full", expensive evaluation with the wide padded fetch -- there is no
   THIRD, separate lightweight backtest.
2. **Real savings where it matters.** Gating BEFORE the holdout side means a
   candidate that fails the gate never pays for the padded-window fetch (a
   materially larger Alpaca request than the in-sample window, per
   `intraday_holdout.SWING_LOOKBACK_BUFFER_DAYS`'s own citation on why
   minute-bar data is expensive) or the second backtest run: the gate stops
   a candidate before the genuinely costly half of this script's work.
3. **`net_edge_bps` gated on is already net of the backtest's own assumed
   `cost_bps`/`tax_haircut_rate`** (baked into `net_ret` by
   `run_intraday_backtest` itself) -- the gate is asking "after already
   modeling realistic costs, is the remaining edge still convincingly larger
   than what a FRESH live quote says the real spread costs right now",
   which targets the failure mode where the edge is barely bigger than the
   cost, the classic way short-hold strategies fail.
   `cost_bps` (the backtest's own modeled transaction cost, a
   candidate-spec field) and the gate's live-captured `quotes[family_id]`
   (a fresh, separate measurement) are deliberately two independent inputs,
   mirroring how the daily pipeline already keeps `execution_costs` (backtest
   cost input) and its own quote-plausibility check as parallel, not
   merged, concerns -- reconciling them into one number is a curation-time
   concern (choosing a realistic `cost_bps` per candidate), not this script's.
   The in-sample net_edge_bps used for gating IS reused as-is for
   `stats_in_sample` in the eventual `computed` item -- computed once, used
   twice, never recomputed differently for the two purposes.

A gate FAILURE is logged with the real measured numbers
(`intraday_cost_gate`'s own `reason` string) and the candidate is SKIPPED --
`record_first_test` is never called for it (see `run_swing_batch`): a gate
failure is not a "tested" outcome, matching how the daily pipeline's own
missing-data exclusions never reach the ledger either.

## Scope note

Every test exercising these functions injects a fake `fetch_fn` and writes
to a temp-file ledger path. `main()` mirrors `run_batch3_mining.main()`'s
own shape but has no default candidate slate: a real run supplies its own
`candidates`/`quotes` and calls `execute_swing_run()` directly.

This module reuses the shared machinery (`run_batch3_mining.py`,
`trial_ledger.py`, `constants.py` and the `intraday_*` modules) by import
only; it never modifies any of them.

## Cross-process ledger locking

A "check-then-act" race in `trial_ledger.py`'s `record_first_test`
(`has_been_tested` read, then `_append` write) would matter for this
script's own "already-tested" early-exclusion check
(`compute_swing_candidates`). It is closed inside `trial_ledger.py`
itself (a Windows OS byte-range mutex, `_LedgerLock`, holds
`record_first_test`'s read+write as one atomic critical section, and
`record_outcome`'s append routes through the same lock), so this module
needs no locking of its own. See `trial_ledger.py`'s module docstring for
the mechanism.
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

# The SAME shared ledger/catalog/results-log paths as the daily pipeline
# (one statistical pool, not two) -- re-exported from batch3
# rather than redefined, so the two scripts can never silently drift apart
# on which files they consider authoritative.
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


def _validate_candidate(candidate: dict) -> None:
    missing = [f for f in _REQUIRED_CANDIDATE_FIELDS if f not in candidate]
    if missing:
        raise ValueError(
            f"candidate {candidate.get('family_id', '?')!r} missing "
            f"required field(s): {missing}")
    if not candidate["tickers"]:
        raise ValueError(
            f"candidate {candidate['family_id']!r}: tickers must be a "
            f"non-empty list")


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
    compute_swing_candidates."""
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
    `[holdout_start, holdout_end]` -- reuses the SAME vendored NYSE
    calendar `run_batch3_mining.derive_holdout_window`/
    `_expected_holdout_calendar` already use (`trading_calendar.
    load_default_sessions()`).

    `compute_swing_candidates` marks a candidate `holdout_complete=True`,
    so the FETCHED padded-holdout data must actually cover the real
    holdout window. Checking only the LAST expected session would catch a
    truncated-at-the-end fetch, but an interior gap (a real outage for, say,
    a week in the MIDDLE of the window, with real data present before and
    after it) would still pass a last-session-only check: the ticker's
    last fetched date still reaches the expected last session, but the
    backtest would silently run "across" the gap as if it were
    continuous. This function returns the FULL expected session list so
    the completeness check at its call site can require every one of them
    to be present, not just the endpoint -- mirroring `run_batch3_mining.
    run_rotation_windows`'s own `expected_holdout_calendar` parameter
    (which checks the observed daily-bar index against the FULL expected
    calendar, not just its last entry).

    Unlike `run_rotation_windows`'s exact `observed.equals(expected)`
    check, the call site here only requires every EXPECTED session to be
    PRESENT (a subset check, not an exact match) -- see that call site's
    own comment for why: an intraday candidate legitimately CAN have zero
    trades on some real, data-covered trading day (the signal simply
    never fired), and this function's own callers check the RAW fetched
    bars' date coverage, never `daily_returns`/`trades_df` content, so
    that distinction is already handled upstream, not by allowing "extra"
    observed days here.

    Raises ValueError if no real NYSE session falls within
    `[holdout_start, holdout_end]` -- mirrors `_expected_holdout_calendar`'s
    own fail-closed behavior for the same condition (should be unreachable
    in practice: `derive_holdout_window` already validates `tested_date`,
    i.e. `holdout_end`, against this same calendar's coverage).
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


def compute_swing_candidates(
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
    work for them -- see below) but never WRITES to it. For every
    not-already-tested candidate: run its in-sample backtest (pooled
    across its own tickers), gate on the measured net edge (see module
    docstring, "Cost-gate sequencing decision"), and -- only for a gate
    PASS -- fetch a lookback-padded holdout window, backtest it, reconcile
    down to the real holdout window (`intraday_holdout.
    reconcile_holdout_window`), and pool that too.

    **Already-tested filtering:** a family
    whose `family_id` is already pinned in `ledger_path` (from ANY prior
    run, not just this batch's own duplicates -- see the separate
    in-batch uniqueness guard in the body below) is excluded immediately, before any
    fetch/backtest work, with a clear reason -- mirroring
    `run_batch3_mining`'s own `ruled_out.filter_eligible`, which filters
    already-tested families out BEFORE `compute_batch`/`score_batch` ever
    see them. Without this, `score_batch`'s own pre-pin-loop check
    (`already_pinned = [... has_been_tested(...) ...]; if already_pinned:
    raise RuntimeError(...)`) would abort the ENTIRE batch -- discarding
    already-completed compute work for every OTHER, perfectly valid
    candidate in the same call -- the instant a candidate list included
    even one family the ledger already has an outcome for (a realistic
    scenario: a re-run of a similar candidate slate, or overlapping
    candidate lists across runs). Applied unconditionally, including in
    `dry_run` mode: `score_batch` itself only checks `already_pinned` when
    `not dry_run`, but a family the pipeline already has a permanent
    outcome for should never be re-scored even as a preview -- the pinned
    holdout window is authoritative from a family's FIRST test, in both
    modes, matching `run_batch3_mining`'s own always-filter convention
    (`load_run_scope` does not special-case dry runs either).

    Returns `{"window": ..., "computed": [...], "gate_exclusions": [...]}`.
    `computed` items are shaped exactly as `run_batch3_mining.
    compute_shared_correction`/`score_batch` require. `gate_exclusions`
    entries never reach `computed` at all -- see `run_swing_batch` for why
    that means they never reach `record_first_test` either.

    `fetch_fn(symbol, start_date, end_date) -> pd.DataFrame` is injectable
    (default: `intraday_data.fetch_intraday_bars`) so tests never hit the
    network -- mirrors `signal_fn`/`session`/`credentials_fn`/`sleep_fn`'s
    own injection-point convention used throughout the intraday modules.

    **A candidate only ever reaches `holdout_complete=True` if its fetched
    holdout-side data genuinely covers EVERY expected real trading session
    in `[holdout_start, holdout_end]`** (see `_expected_holdout_sessions`)
    -- not just the last one: a truncated-at-the-end fetch AND an interior
    gap (a real outage in the MIDDLE of the window, with data present
    before and after it) are both excluded the same way a missing quote
    or a fetch exception already is, with a real, specific reason, never
    silently scored as if it were a complete evaluation.
    """
    window = batch3.derive_holdout_window(tested_date)
    expected_holdout_sessions = _expected_holdout_sessions(
        window["holdout_start"], window["holdout_end"])
    computed: list = []
    gate_exclusions: list = []

    # Uniqueness guard, checked BEFORE any candidate is touched:
    # `score_batch`'s pin
    # loop (in run_batch3_mining.py) treats a SECOND
    # `record_first_test` call for an already-pinned family_id as a
    # pin_error, which aborts the ENTIRE pin loop and defensively
    # records a not_evaluated_error outcome for EVERY family already pinned in that batch --
    # including unrelated, otherwise-valid candidates that happened to be
    # pinned earlier in iteration order. Two candidates sharing a
    # family_id in one `candidates` list would trigger exactly this,
    # silently mis-recording an unrelated candidate's ledger outcome.
    # `run_batch3_mining.load_run_scope` guards against this on its own
    # static family matrix (`len(set(ids)) != len(ids)`) before
    # `score_batch` is ever reached; this is the equivalent guard for a
    # caller-supplied candidate list. A duplicate family_id is a caller
    # bug (malformed candidate list), not a runtime data condition -- fail
    # loud here, same as _validate_candidate's own missing-field checks,
    # rather than silently excluding one copy.
    family_ids = [c.get("family_id") for c in candidates]
    if len(set(family_ids)) != len(family_ids):
        seen, duplicates = set(), set()
        for fid in family_ids:
            (duplicates if fid in seen else seen).add(fid)
        raise ValueError(
            f"compute_swing_candidates: duplicate family_id(s) in "
            f"candidates list: {sorted(duplicates)} -- score_batch's pin "
            f"loop cannot handle two candidates sharing one family_id in "
            f"the same batch (a second pin aborts the whole batch and "
            f"defensively cuts every already-pinned family, including "
            f"unrelated ones)")

    for candidate in candidates:
        _validate_candidate(candidate)
        family_id = candidate["family_id"]
        tickers = candidate["tickers"]
        signal_fn = candidate["signal_fn"]
        cost_bps = candidate["cost_bps"]
        tax_haircut_rate = candidate["tax_haircut_rate"]
        max_hold_days = candidate["max_hold_days"]
        direction = candidate["direction"]

        # Already-tested filtering (see this
        # function's own docstring for the full "why", including why it
        # applies unconditionally, not just when not dry_run). Checked
        # before any fetch/backtest work, same reasoning as the quote
        # presence check below: cheap, and matches "we already have a
        # permanent answer for this family" being a differently-shaped but
        # equally real reason not to redo the work.
        #
        # A not_evaluated_error family is a caught DEFECT, not evidence
        # about the market (evidence.build_engine_error's own docstring)
        # -- it stays eligible to re-enter scoring once the defect is
        # fixed, matching ruled_out.filter_eligible's own treatment at the
        # catalog-filter level. Every OTHER outcome (a scored result, or
        # insufficient_evidence) still excludes unconditionally.
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
            # pinned window the earlier failed attempt used -- record_
            # first_test (called later, by score_batch) is idempotent and
            # will NOT re-pin, so scoring against a DIFFERENT window here
            # would silently record an outcome that does not match what
            # the ledger says was tested. Exclude with a clear reason
            # instead of risking that; this is NOT the "never re-scored"
            # exclusion above, so it is worded distinctly.
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

        # Look-ahead guard: unlike the daily pipeline (where in-sample is
        # DERIVED as "everything before holdout_start" by construction --
        # see run_batch3_mining._window_masks), this candidate spec's
        # in_sample_start/in_sample_end are separate, caller-supplied
        # values (a deliberate design choice: Alpaca's retention depth is
        # not a known fixed bound, so curation picks its own
        # in-sample window rather than "everything available"). That
        # separation only stays a genuine train/holdout split if
        # in_sample_end actually precedes holdout_start -- fail closed
        # rather than silently score a candidate whose "in-sample" data
        # overlaps or postdates the window it is supposed to be held out
        # against.
        if pd.Timestamp(candidate["in_sample_end"]) >= pd.Timestamp(window["holdout_start"]):
            raise ValueError(
                f"candidate {family_id!r}: in_sample_end "
                f"{candidate['in_sample_end']} must be strictly before "
                f"holdout_start {window['holdout_start']} -- an in-sample "
                f"window that reaches into or past the holdout window is a "
                f"look-ahead leak, not a valid train/holdout split")

        # --- quote presence check (fail-closed as a logged exclusion, NOT
        # a crash) -- checked BEFORE any fetch/backtest work for this
        # candidate, both because it is cheap to check first and because a
        # missing quote is the SAME kind of "we cannot proceed without a
        # real input" exclusion as an Alpaca data gap (the pipeline's "never
        # invent a cost or a candidate's data" rule): there is no way to
        # evaluate the cost-realism gate without one, and live quote
        # capture realistically CAN omit a family (a quote is captured
        # OUTSIDE this function, by the caller, before `quotes` is ever
        # passed in -- the caller may simply not have gotten to every
        # candidate).
        # Indexing `quotes[family_id]` unguarded would let ONE candidate
        # missing a quote raise an unhandled KeyError and crash the
        # ENTIRE batch (including candidates that had already computed
        # perfectly good stats) before `score_batch` ever ran -- exactly
        # the kind of single-candidate failure this pipeline's "every skip
        # carries a real reason, no candidate's failure blocks another's
        # scoring" discipline exists to prevent.
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

        # Everything below is real fetch/backtest work -- wrapped in one
        # broad try/except, mirroring run_batch3_mining.compute_batch's own
        # convention (its per-candidate `except Exception` around the whole
        # `compute_candidate(...)` call): ANY failure here (a
        # real Alpaca data gap/pagination/credential error --
        # intraday_data.py's own module docstring: "It does NOT catch its
        # own gap/pagination errors -- a future orchestrating caller ...
        # decides the skip-and-log policy," which is exactly this script --
        # or any other exception) converts to a per-candidate skip with the
        # real exception recorded as the reason, rather than aborting
        # compute for every OTHER candidate in the batch. Letting a
        # data-gap exception from ONE candidate propagate out of
        # compute_swing_candidates entirely would discard already-completed
        # compute work for unrelated candidates earlier in the loop.
        try:
            # --- in-sample backtest (also doubles as the cost gate's
            # prior-edge measurement -- see module docstring). ---
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

            # --- cost-realism gate ---
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

            # --- holdout evaluation (padded fetch, reconcile, pool) ---
            padded_start = (
                pd.Timestamp(window["holdout_start"])
                - pd.Timedelta(days=lookback_buffer_days)
            ).strftime("%Y-%m-%d")

            # Completeness check (see
            # _expected_holdout_sessions's own docstring for the full
            # "why": a check of only the LAST expected session would still
            # pass an INTERIOR gap -- real data before and after a real
            # outage in the middle of the window). A candidate is marked holdout_complete=True
            # below UNCONDITIONALLY once this loop finishes -- that must
            # never happen for data with ANY missing expected session, not
            # just a short tail. Checked against the RAW fetched
            # `padded_bars` for each ticker (never against daily_returns/
            # trades_df content, which is legitimately zero on a real,
            # covered, but inactive day) -- incomplete data breaks out of
            # the per-ticker loop immediately and the WHOLE candidate is
            # excluded (pooling a genuinely incomplete ticker's holdout
            # stats with other tickers' would still corrupt the pooled
            # result; also, silently backtesting "across" an interior gap
            # as if the missing days simply didn't exist would treat a
            # real outage as if it were continuous history).
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
            # `str(exc)`: this
            # `reason` string flows into `gate_exclusions`, which
            # `execute_swing_run` writes verbatim into a durable JSON file
            # (`output_path`). `fetch_fn` is caller-injectable (this
            # module's own dependency-injection contract), so its raised
            # exceptions are NOT guaranteed secret-free the way
            # `intraday_data.py`'s own documented exception classes are
            # (that module's own docstring: "Never print, log,
            # write-to-file, or embed the actual key/secret values
            # anywhere (including in exception messages)") -- a different,
            # future fetch_fn (or a lower-level HTTP/auth library it
            # wraps) could raise something whose `str()` embeds a header,
            # token, account identifier, or response body. The exception
            # class name alone (e.g. IntradayDataGapError, CredentialError,
            # AlpacaRequestError) is still a real, specific, useful reason
            # for the documented intraday_data.py failure modes -- this
            # pipeline deliberately does NOT attempt its own ad hoc
            # message-scrubbing heuristic here (an improvised inline
            # scrubber can miss a secret; withholding the message text
            # entirely cannot).
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
            # instrument list, independent of `ticker` above -- which can
            # be an arbitrary display LABEL (`_ticker_label` falls back to
            # `candidate.get('ticker_label')`), not a parseable ticker.
            # batch3._instrument_set_for must read this, never `ticker`.
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
            # Passed through from the candidate's own already-validated,
            # already-REQUIRED fields -- this script imports
            # batch3._score_one UNMODIFIED, and it needs these two to build
            # an insufficient_evidence payload for a
            # below-MIN_TRADES_FOR_SCORING candidate.
            "in_sample_start": candidate["in_sample_start"],
            "in_sample_end": candidate["in_sample_end"],
            # `raw_catalog_parameters`: this pipeline's candidates carry a pre-built signal_fn
            # rather than catalog-driven parameters, so there is no raw
            # dict to translate the way run_batch3_mining.py's own daily
            # candidates have one -- an OPTIONAL `parameters` key on the
            # candidate is threaded through if a caller supplies one
            # (an empty dict otherwise).
            "raw_catalog_parameters": dict(candidate.get("parameters") or {}),
        })

    return {"window": window, "computed": computed,
            "gate_exclusions": gate_exclusions}


def run_swing_batch(
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
    """Compute + gate + score ONE swing-track run: calls
    `compute_shared_correction`/`score_batch` (imported unmodified from
    `run_batch3_mining`) EXACTLY ONCE, over ONLY this call's own
    gate-passing swing candidates (the batching rule -- see module
    docstring). A gate-excluded candidate never appears in `computed`, so
    `score_batch`'s unconditional-first `record_first_test` loop never sees
    it -- a cost-gate failure never reaches the ledger, because a gate
    failure is not a tested outcome.

    `score_batch` itself guarantees: `record_first_test` is called
    UNCONDITIONALLY FIRST for every
    item that DOES reach `computed`, and any exception raised anywhere
    after pinning begins -- including inside `_score_one`'s own DSR/holdout
    scoring -- is caught by `score_batch`'s own `try/except`, which calls
    `_record_defensive_outcomes` to write a not_evaluated_error outcome for every
    newly-pinned-but-not-yet-outcomed family before re-raising. This
    function relies on that guarantee rather than re-implementing it."""
    batch = compute_swing_candidates(
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
        "dry_run": dry_run, "timeframe": "swing", **batch["window"],
        "shared_correction": correction,
        "results": [batch3.public_result(item) for item in scored],
        "gate_exclusions": batch["gate_exclusions"],
    }


def _file_sha256(path: str) -> str | None:
    """Same hashing convention as run_batch3_mining._file_sha256 --
    reimplemented locally (a ~10-line stdlib utility) rather
    than imported, to keep this module's coupling to batch3's private
    surface limited to the three statistically load-bearing functions named
    in the module docstring."""
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    """Recursively replace non-finite floats (inf/-inf/nan) with `None` so
    a payload can be serialized with `allow_nan=False` (this pipeline's own
    strict JSON contract, matching `run_batch3_mining.execute_batch_run`'s
    identical `json.dumps(..., allow_nan=False)` call).

    A non-finite float here is a genuine, EXPECTED output of
    `intraday_cost_gate.evaluate_cost_realism_gate`: a zero-measured
    round-trip spread makes `edge_over_cost_ratio` mathematically infinite
    (see that module's own docstring/code, `spread_bps == 0` branch) --
    not corrupted data. Replacing it with `None` matches this pipeline's
    existing "not computed / not meaningful" convention (e.g.
    `trial_ledger.record_outcome`'s `dsr_value=None`), rather than letting
    `json.dumps`'s `ValueError: Out of range float values are not JSON
    compliant` propagate as a hard crash.

    Without this, a zero-spread captured quote would make
    `execute_swing_run` raise AFTER the ledger had already been correctly
    updated (pinned, scored, outcome recorded) and the results log
    correctly regenerated -- silently discarding only this run's own JSON
    evidence artifact for an otherwise fully-successful, already-recorded
    result. Applied once, right before serialization -- never mutates the
    in-memory `payload` callers see.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _atomic_write_text(path: str, text: str) -> None:
    """Same atomic-write convention as run_batch3_mining._atomic_write_text:
    write to a unique temp file in the destination
    directory, then os.replace() it onto the final path (atomic on both
    POSIX and Windows) so a reader never observes a partial write."""
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


def execute_swing_run(
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
    change -- mirrors `run_batch3_mining.execute_batch_run`'s own contract
    exactly: regenerate `RESULTS-LOG.md`
    (via `results_log.render_results_log`, through batch3's own
    `regenerate_results_log` convenience wrapper) AND `NEAR-MISS-LOG.md`
    (via batch3's own `regenerate_near_miss_log`) if and only
    if `not dry_run` AND the ledger file's sha256 actually changed, in a
    `finally`-shaped block that ALSO regenerates on a run-time exception
    (so a defensively-outcomed pin from `score_batch` is never left
    unreflected in either durable log), raising an `ExceptionGroup` if BOTH
    the run and the regeneration fail rather than swallowing either.
    `near_miss_log_path` defaults to a sibling of `results_log_path` when
    omitted, so a caller that only sets `results_renderer`/
    `results_log_path` still gets the near-miss log refreshed.
    """
    runner = batch_runner or run_swing_batch
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
                "swing batch run and results-log regeneration both failed",
                [run_error, regeneration_error]) from None
        raise
    refresh_durable_evidence()
    return payload


def main() -> int:
    """CLI entry point, mirroring run_batch3_mining.main()'s own shape.
    It runs nothing: a real swing-track run supplies its own real
    `candidates`/`quotes` and calls `execute_swing_run()` directly (there is
    no static default candidate list here to accidentally run)."""
    raise SystemExit(
        "run_swing_mining.main() has no default candidate slate to run -- "
        "a swing-track mining run supplies its own "
        "candidates/quotes and calls execute_swing_run() directly. This "
        "entry point exists for structural parity with "
        "run_batch3_mining.py's CLI shape, not for standalone invocation.")


if __name__ == "__main__":
    raise SystemExit(main())
