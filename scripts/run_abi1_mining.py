"""ABI-1 full-history account-relative mining driver.

Runs every accepted candidate (the `--accepted` curation file) through its FULL available price history (not the per-trade
pipeline's pinned 6-month in-sample/holdout split) and records a real
ABI-1 verdict via `mining.account_relative.evaluate()`
(`mining.abi1_driver.evaluate_family`/`record_abi1_outcome`).

Reuses, imported UNCHANGED from `run_batch5_mining.py`: `discover_modules`,
`build_scope`, `execution_costs_for`, `load_frame`/`default_ohlc_loader`,
`refuse_synthetic_quotes`, `load_accepted`, `load_quotes`, `parameter_drift`,
`pool_ticker_label`, `_pool_coverage_ok`/`MIN_POOL_COVERAGE`,
`_render_exclusions`, `_skip`, `_json_safe`. Does NOT import
`compute_candidate`/`compute_batch` (batch5's own per-trade backtest) --
`compute_batch_abi1` below replaces them with a full-history ABI-1 path.

Per-candidate flow: discover module -> validate contract/parameter drift ->
gate on quotes -> load full OHLC for every traded/signal symbol (plus the
fixed SPY benchmark, always) -> build the account curve via the matching
curve builder for `spec["shape"]` -> `evaluate_family`, passing
`tax_policy="realization"` for `single_long`/`single_inverse_product`/
`panel_breadth`/`rotation_two_asset` and `tax_policy="none"` for
`pooled_single_asset`/`event_pooled` (pooled shapes are scored untaxed) ->
`record_abi1_outcome`, UNLESS `--dry-run`, in which case the verdict
is computed and reported but no ledger write ever happens.

`--families` filters the accepted scope (reuses `build_scope`'s own
parameter). There is deliberately no `--retest` flag: ABI-1 scoring is
one-shot per family -- the re-run guard below reads the ledger directly (`trial_ledger.read_all`) and skips (reason code
`already_scored_abi1`) any family carrying a prior `construction == "ABI-1"`
outcome row, regardless of stage.

`--history-end YYYY-MM-DD` (opt-in; default off and then
nothing below changes) ends the scored history at that NYSE session. It only
DROPS later sessions, in memory, once every series has been loaded and cached
(the per-run OHLC cache keeps the full vendor history); it never fills, patches
or shifts anything. Fail closed, for the WHOLE batch: the date must be a real
NYSE session, and EVERY loaded series must hold a bar dated exactly on it. If
any one does not, the whole batch is refused, not just the plans that use that
series (scoring the covered siblings would write permanent rows for a batch that
was not fully covered, and a pooled family would be scored after silently
dropping a member). A series that fails to LOAD for another reason is handled as
with no history end: only the plans using it are skipped (but if NO series loads at all the
setting is refused too, `history_end_not_covered`, rather than silently ignored).
Narrow the batch with `--families` instead.

Every history-end refusal has one shape (`history_cutoff.refusal_payload`:
`reason_code`, `reason`, `history_end_requested`, `symbols`) and the CLI exits 2.
A `history_end_*` refusal writes NO ledger row, NO results file and NO run report.
Refusals raised at the DOOR, before anything is loaded, leave nothing at all (no
OHLC cache either): `history_end_invalid` / `history_end_not_a_session` (a
malformed or non-session date) and, whenever a history end is set,
`history_end_invalid` for a malformed `--tested-date` or a bad
`--max-history-drop` given programmatically (the real-run guard's two inputs,
checked at the door for real and dry runs alike, so neither can fail after the
load step). Refusals raised after the load step: `history_end_not_covered` /
`history_end_empty` (a loaded series lacks the cutoff bar, or no series loaded at
all) and, on a REAL run, the real-run guard `history_end_after_tested_date` (the
cutoff is later than `--tested-date`) and `history_end_drop_too_large` (a series
would drop more than `--max-history-drop` sessions, default
`history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS`; pass a larger N to accept it). The
per-run OHLC cache the load step wrote before such a refusal is kept, so a real
retry needs a fresh --run-id: this dispatch has NO stale-cache refusal (the
bespoke overlay drivers refuse a reused run-id as `stale_ohlc_cache`), a reused
--run-id silently reads the cache the refused attempt wrote. Always use a fresh
--run-id. A DRY run never refuses on the guard's two comparisons (a cutoff after
the tested date, a drop over the limit): it reports the guard's message (or None)
as `history_end_guard_would_refuse` so a rehearsal predicts the real run. Give a
dry run the real --tested-date: the `--tested-date` default is today's
date at run time, so a dry run without it rehearses the guard against
today.

The ledger note, the results JSON (`history_end` = the provenance,
`history_end_requested` = the string typed) and the run report all state the
history end. A scored history end is PERMANENT for the account it scores: the
outcome row's account ends at the cutoff, and a NEW family's first_test pin (its
ABI-1 holdout is the tail of that account) ends there too, so the family cannot
be re-scored on longer history. The pin is not unconditional:
`trial_ledger.record_first_test` is a no-op for a family that already has a
first_test row, and a not_scoreable row pins none.
The same statement is written on an UNSCORED row too (`not_scoreable` /
`holdout_not_evaluable`, recorded as `insufficient_evidence`): an unscored run
still spends the family's one permanent ABI-1 slot, so it is processed like any
other verdict. The driver passes the clause to `record_abi1_outcome` both as
`extra_note` (read only by a scored row) and as `unscored_note` (read only by an
unscored row). With no history end neither keyword is passed at all.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections.abc import Mapping

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import (abi1_driver, constants, evidence, history_cutoff, reporting,
                    trial_ledger)

# The ONLY cash convention this v1 driver uses -- no accepted family
# declares a `parked_cash_daily` alternative (see abi1_driver.evaluate_family's
# own docstring). Recorded literally on every ledger row's `note`, so the
# cash convention is fixed before scoring and recorded with the run, without
# a declaration mechanism no family yet uses.
CASH_CONVENTION = "sweep_cash_daily"

# ABI-1 always compares the account to SPY -- NOT a family's own catalog
# `benchmark`/`BENCHMARK` field (that field is the batch5 per-trade panel's
# "own benchmark" concept and plays no role in ABI-1's account-vs-index test).
MARKET_BENCHMARK = batch5.MARKET_BENCHMARK  # "SPY"

# Pooled/event_pooled families are scored
# UNTAXED in v1 (netted_tax_factors' dollar-gain identity does not hold on a
# diluted pooled curve); every other shape keeps full realization tax.
_UNTAXED_SHAPES = batch5.POOLED_SHAPES

DEFAULT_RUN_ID = "abi1-v1"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()  # today, at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_MODULES_DIR = batch5.DEFAULT_MODULES_DIR
DEFAULT_ACCEPTED = batch5.DEFAULT_ACCEPTED
DEFAULT_QUOTES = batch5.DEFAULT_QUOTES

# Same "the module implementing the score-decision logic" convention
# run_batch3_mining._SIGNAL_SOURCE_PATH established (its own comment: hashing
# the DRIVER script, not each family's own signal module, is the more
# conservative choice when one script dispatches many families' worth of
# scoring decisions) -- matches the literal string the record_abi1_outcome
# test fixtures already use for `signal_source_path`.
_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


# --------------------------------------------------------------- dispatch

def _build_curve(plan: dict, frames: Mapping) -> dict:
    """Build the account curve for one candidate via the shape's curve builder
    matching `plan['spec']['shape']`. `plan['effective']` is the FINAL
    (post cost-gate, post OHLC-load, post pool-coverage) traded set."""
    spec, module = plan["spec"], plan["module"]
    shape = spec["shape"]
    effective = list(plan["effective"])
    signal_symbols = list(spec["signal"])

    if shape in batch5.SINGLE_SHAPES:
        symbol = effective[0]
        full_frame = frames[symbol]
        module_frames = {s: frames[s] for s in [symbol] + signal_symbols}
        signal_fn = module.make_signal_fn(symbol, full_frame.index, module_frames)
        cost_bps = float(plan["costs"][symbol]["cost_bps"])
        return abi1_driver.build_single_asset_curve(
            full_frame, signal_fn, cost_bps=cost_bps)

    if shape == batch5.ROTATION_SHAPE:
        asset_frames = {s: frames[s] for s in effective}
        module_frames = {s: frames[s] for s in effective + signal_symbols}
        target_fn = module.make_target_fn(module_frames)
        cost_bps_by_asset = {s: float(plan["costs"][s]["cost_bps"])
                             for s in effective}
        signal_frames = ({s: frames[s] for s in signal_symbols}
                         if signal_symbols else None)
        return abi1_driver.build_rotation_curve(
            asset_frames, target_fn, cost_bps_by_asset=cost_bps_by_asset,
            signal_frames=signal_frames)

    if shape in batch5.POOLED_SHAPES:
        module_frames = {s: frames[s] for s in effective + signal_symbols}
        per_ticker_signal_fn = {
            symbol: module.make_signal_fn(symbol, frames[symbol].index, module_frames)
            for symbol in effective
        }
        full_frames = {symbol: frames[symbol] for symbol in effective}
        cost_bps_by_symbol = {symbol: float(plan["costs"][symbol]["cost_bps"])
                              for symbol in effective}
        return abi1_driver.build_pooled_curve(
            per_ticker_signal_fn, full_frames,
            cost_bps_by_symbol=cost_bps_by_symbol,
            selected_count=getattr(module, "HOLD_TOP_N", None))

    # Unreachable given validate_module_contract's own SHAPE in SHAPES check,
    # but fail loudly rather than silently misrouting a shape this dispatch
    # was never taught about (the same "no guessing" discipline
    # record_abi1_outcome's own unhandled-verdict branch uses).
    raise AssertionError(f"unhandled ABI-1 shape {shape!r}")


def _tax_policy_for(shape: str) -> str:
    return "none" if shape in _UNTAXED_SHAPES else "realization"


# ----------------------------------------------------------------- compute

def _history_end_refused(exc, history_requested) -> dict:
    """The compute result for a refused history end: no candidate computed or
    skipped, and the standard refusal (`history_cutoff.refusal_payload`) under
    `history_end_refusal`. Nothing has been written to the ledger at this point."""
    return {"computed": [], "skipped": [], "input_coverage": {},
            "input_errors": {}, "benchmark_entry_cost_bps": None,
            "history_end_refusal": history_cutoff.refusal_payload(
                exc, history_end_requested=history_requested)}


def _history_end_door(history_end, tested_date, max_history_drop):
    """The door checks of a history end, shared by main(), run_batch_abi1() and
    compute_batch_abi1() so a malformed cutoff, tested date or max-history-drop
    is refused before ANYTHING else is read or loaded (catalog, quotes, modules,
    OHLC). Returns the parsed cutoff, None when the setting is off; raises
    `history_cutoff.HistoryCutoffError`."""
    if history_end is None:
        return None
    cutoff = history_cutoff.parse_history_end(history_end)
    history_cutoff.validate_guard_arguments(
        tested_date=tested_date, max_drop=max_history_drop)
    return cutoff


def _print_history_end_refusal(refusal: dict, dry_run: bool) -> None:
    """The one printed shape of a refused history end (main() then exits 2)."""
    print(json.dumps({
        "refused": True,
        **{key: refusal[key] for key in
           ("reason_code", "reason", "history_end_requested", "symbols")},
        "dry_run": dry_run}))


def compute_batch_abi1(scope: dict, *, tested_date: str, run_id: str,
                       ledger_path: str, dry_run: bool,
                       history_end=None,
                       max_history_drop: int = history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS
                       ) -> dict:
    """Preflight every accepted candidate (module, catalog, drift, re-run
    guard, quote gate), load every OHLC frame once (including the fixed SPY
    benchmark), then evaluate every feasible candidate under ABI-1. No
    ledger write happens before every earlier candidate's checks are done,
    and (`dry_run=True` aside) no candidate's ledger write happens before its
    own curve/evaluate step succeeds.

    `history_end` (default None = off, everything below unchanged) is the
    opt-in end-of-history setting, see the module docstring. It is validated at
    the door, before anything is loaded, together with the real-run guard's two
    inputs (`tested_date`, `max_history_drop`, via
    `history_cutoff.validate_guard_arguments`), for real and dry runs alike; a
    malformed or non-session value, or a malformed guard input, returns a
    result carrying only `history_end_refusal` (the standard refusal shape) and
    touches nothing (no OHLC cache either). After the load step the WHOLE dict
    of loaded frames is trimmed together: if any loaded series lacks a bar on the
    cutoff, or no series loaded at all, the whole batch is refused
    (`history_end_refusal`, no candidate computed or skipped, no ledger row; the
    per-run OHLC cache stays). On a REAL run `history_cutoff.check_real_run_guard`
    then applies (`tested_date`, `max_history_drop`) and can refuse the same way;
    on a DRY run the guard's message (or None) is reported as
    `history_end_guard_would_refuse` instead."""
    cutoff = None
    history_requested = None if history_end is None else str(history_end)
    if history_end is not None:
        try:
            # The real-run guard's two inputs are checked with the cutoff, for
            # real and dry runs alike (a dry run rehearses the real one), so
            # neither can fail AFTER the load step below has written the per-run
            # OHLC cache; run_batch_abi1() and main() call the same door earlier.
            cutoff = _history_end_door(history_end, tested_date, max_history_drop)
        except history_cutoff.HistoryCutoffError as exc:
            return _history_end_refused(exc, history_requested)
    history_extra = ({} if cutoff is None
                     else {"history_end_requested": history_requested})
    skipped: list = []
    computed: list = []
    quotes_payload = scope["quotes"]
    problem_by_module = {p["module"]: p["reason"] for p in scope["module_problems"]}

    # Re-run guard (no --retest in v1): a family carrying ANY prior
    # construction="ABI-1" outcome row, in any stage, is permanently done.
    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"
    }

    # The benchmark cost is shared by EVERY family (evaluate_family always
    # compares the account to SPY) -- computed once, up front. A missing/bad
    # SPY quote makes every candidate unscoreable, not just one.
    try:
        spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK], quotes_payload)
        benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0
    except ValueError as exc:
        for candidate in scope["accepted"]:
            batch5._skip(skipped, candidate["family_id"], "missing_data",
                        f"benchmark quote gate ({MARKET_BENCHMARK}): {exc}")
        return {"computed": computed, "skipped": skipped,
                "input_coverage": {}, "input_errors": {},
                "benchmark_entry_cost_bps": None, **history_extra}

    plans = []
    for candidate in scope["accepted"]:
        family_id = candidate["family_id"]
        found = scope["modules"].get(family_id)
        if found is None:
            reason = problem_by_module.get(family_id)
            if reason:
                batch5._skip(skipped, family_id, "invalid_module", reason)
            else:
                batch5._skip(skipped, family_id, "no_module",
                            f"no valid module for {family_id} under "
                            f"{batch5.portable_path(scope['modules_dir'])}")
            continue
        spec, module = found["spec"], found["module"]
        entry = scope["catalog_by_id"].get(family_id)
        if entry is None:
            batch5._skip(skipped, family_id, "not_in_catalog",
                        "family_id absent from the catalog (run "
                        "add_batch5_to_catalog.py first)")
            continue
        if entry.get("status") != "eligible":
            batch5._skip(skipped, family_id, "catalog_status",
                        f"catalog status is {entry.get('status')!r}, not eligible")
            continue
        drift = batch5.parameter_drift(spec["parameters"], entry.get("parameters"))
        if drift is not None:
            batch5._skip(skipped, family_id, "parameter_drift", drift)
            continue
        if family_id in already_scored:
            batch5._skip(skipped, family_id, "already_scored_abi1",
                        "a prior run already recorded a construction="
                        "\"ABI-1\" outcome for this family; v1 has no "
                        "retry mechanism")
            continue

        pooled = spec["shape"] in batch5.POOLED_SHAPES
        declared = list(spec["traded"])
        excluded: dict = {}
        if pooled:
            costs = {}
            for symbol in declared:
                try:
                    costs.update(batch5.execution_costs_for([symbol], quotes_payload))
                except ValueError as exc:
                    excluded[symbol] = f"quote gate: {exc}"
            if not costs:
                batch5._skip(skipped, family_id, "missing_data",
                            "every pool member excluded at the quote gate: "
                            + batch5._render_exclusions(excluded))
                continue
        else:
            try:
                costs = batch5.execution_costs_for(declared, quotes_payload)
            except ValueError as exc:
                batch5._skip(skipped, family_id, "missing_data",
                            f"quote gate: {exc}")
                continue

        plans.append({
            "family_id": family_id, "spec": spec, "module": module,
            "entry": entry, "costs": costs, "pooled": pooled,
            "pool_declared": declared, "pool_excluded": excluded,
            "effective": [s for s in declared if s in costs],
        })

    # -------------------------------------------------- OHLC, once, globally
    symbols = {MARKET_BENCHMARK}
    traded_symbols = {MARKET_BENCHMARK}
    for plan in plans:
        symbols.update(plan["effective"])
        symbols.update(plan["spec"]["signal"])
        traded_symbols.update(plan["effective"])
    # A symbol this run NEVER trades anywhere (signal-only for every plan
    # that references it, e.g. ^TNX/^IRX/^VIX) is a decision input, never a
    # traded/priced instrument -- it may legitimately carry a real negative
    # value (a Treasury-bill yield went negative in March 2020) and must not
    # fail the strict positivity gate meant for actual traded prices. A
    # symbol that is TRADED by even one plan in this run keeps the strict
    # gate everywhere, including any other plan that only uses it as SIGNAL.
    signal_only_symbols = symbols - traded_symbols
    frames: dict = {}
    load_errors: dict = {}
    coverage: dict = {}
    cache_dir = os.path.join(scope["run_dir"], "artifacts", "ohlc")
    for symbol in sorted(symbols):
        try:
            report: dict = {}
            frame, origin = batch5.load_frame(
                symbol, ohlc_loader=scope["ohlc_loader"], cache_dir=cache_dir,
                report=report, require_positive=symbol not in signal_only_symbols)
            frames[symbol] = frame
            coverage[symbol] = {
                "first": frame.index.min().strftime("%Y-%m-%d"),
                "last": frame.index.max().strftime("%Y-%m-%d"),
                "origin": origin, **report,
            }
        except Exception as exc:
            load_errors[symbol] = f"{type(exc).__name__}: {exc}"

    history_provenance = None
    if cutoff is not None:
        # In memory only, AFTER every load_frame call: the per-run cache CSVs keep
        # the full vendor history. The WHOLE dict of loaded frames is trimmed in
        # one call, so a series lacking the cutoff bar refuses the whole batch,
        # never just the plans that use it. No ledger row has been written yet,
        # and the caller writes no results file or run report for a refusal (the
        # cache written above stays). A series that failed to load above is not
        # part of `frames` and keeps the usual per-plan skip. An EMPTY `frames`
        # (every series failed to load) is trimmed too, on purpose: the shared
        # trim refuses an empty universe ("no series were supplied"), so the
        # setting is never silently ignored and no report or results file is
        # written for a run that had nothing to end at the cutoff. `frames` is
        # rebound: every later use of a symbol's data sees the trimmed frame.
        try:
            frames, history_provenance = history_cutoff.trim_frames_to_history_end(
                frames, history_end)
        except history_cutoff.HistoryCutoffError as exc:
            return _history_end_refused(exc, history_requested)
        guard_refusal = None
        try:
            history_cutoff.check_real_run_guard(
                history_provenance, tested_date=tested_date,
                max_drop=max_history_drop)
        except history_cutoff.HistoryCutoffError as exc:
            guard_refusal = exc
        if dry_run:
            # A dry run never refuses on the guard: it says what a real run of
            # this same setting would do.
            history_extra["history_end_guard_would_refuse"] = (
                None if guard_refusal is None else str(guard_refusal))
        elif guard_refusal is not None:
            return _history_end_refused(guard_refusal, history_requested)
        for symbol, record in history_provenance["by_symbol"].items():
            coverage[symbol].update(
                first=frames[symbol].index.min().strftime("%Y-%m-%d"),
                last=frames[symbol].index.max().strftime("%Y-%m-%d"),
                raw_last=record["raw_last"], kept_last=record["kept_last"],
                dropped_sessions=record["dropped_sessions"])
        # Run-level statement of what the history end did (every loaded series);
        # each plan below also gets its own, limited to the series it scores.
        history_extra["history_end_provenance"] = history_provenance

    benchmark_frame = frames.get(MARKET_BENCHMARK)
    if benchmark_frame is None:
        for plan in plans:
            batch5._skip(skipped, plan["family_id"], "missing_data",
                        f"benchmark OHLC ({MARKET_BENCHMARK}): "
                        f"{load_errors.get(MARKET_BENCHMARK, 'load failed')}")
        return {"computed": computed, "skipped": skipped,
                "input_coverage": coverage, "input_errors": load_errors,
                "benchmark_entry_cost_bps": benchmark_entry_cost_bps,
                **history_extra}

    # -------------------------------------------------------- per-candidate
    for plan in plans:
        family_id, spec = plan["family_id"], plan["spec"]
        pooled, excluded = plan["pooled"], plan["pool_excluded"]
        effective = list(plan["effective"])

        # Signal-only symbols carry the whole family: a load failure there
        # is always fatal, pooled or not.
        support = dict.fromkeys(spec["signal"])
        support_problems = [f"{symbol}: {load_errors[symbol]}"
                            for symbol in support if symbol in load_errors]
        if support_problems:
            batch5._skip(skipped, family_id, "missing_data",
                        "; ".join(support_problems))
            continue

        if pooled:
            for symbol in list(effective):
                if symbol in load_errors:
                    excluded[symbol] = f"ohlc: {load_errors[symbol]}"
                    effective.remove(symbol)
            if not batch5._pool_coverage_ok(effective, plan["pool_declared"]):
                kept, declared_n = len(effective), len(plan["pool_declared"])
                fraction = (kept / declared_n) if declared_n else 0.0
                batch5._skip(skipped, family_id, "pool_coverage",
                            f"only {kept} of {declared_n} declared pool "
                            f"members are usable ({fraction:.1%}), below the "
                            f"{batch5.MIN_POOL_COVERAGE:.0%} floor. Excluded: "
                            + batch5._render_exclusions(excluded))
                continue
        else:
            member_problems = [f"{symbol}: {load_errors[symbol]}"
                               for symbol in effective if symbol in load_errors]
            if member_problems:
                batch5._skip(skipped, family_id, "missing_data",
                            "; ".join(member_problems))
                continue

        tax_policy = _tax_policy_for(spec["shape"])
        plan_history = None
        try:
            if history_provenance is not None:
                # Only the series THIS plan scores (its traded set, its signal
                # series and the SPY benchmark), so its ledger note is exact.
                plan_symbols = {MARKET_BENCHMARK, *effective, *spec["signal"]}
                plan_history = history_cutoff.build_provenance(
                    cutoff, {s: history_provenance["by_symbol"][s]
                             for s in plan_symbols})
            curve = _build_curve({**plan, "effective": effective}, frames)
            account_type = plan["entry"]["account_types"][0]
            result = abi1_driver.evaluate_family(
                curve, benchmark_frame=benchmark_frame,
                benchmark_entry_cost_bps=benchmark_entry_cost_bps,
                account_type=account_type, tax_policy=tax_policy)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if excluded:
                detail += "; already excluded: " + batch5._render_exclusions(excluded)
            batch5._skip(skipped, family_id, "compute_error", detail)
            continue

        record_kwargs: dict = {}
        if plan_history is not None:
            clause = history_cutoff.history_end_clause(plan_history)
            history_block = {**plan_history, "clause": clause}
            # Same clause on either kind of row: a scored row reads extra_note,
            # an unscored one reads unscored_note (each ignores the other).
            record_kwargs["extra_note"] = clause
            record_kwargs["unscored_note"] = clause

        ticker_label = batch5.pool_ticker_label(effective) if pooled else None
        account_index = result["account_daily"].index
        account_start = (account_index[0].strftime("%Y-%m-%d")
                         if len(account_index) else None)
        account_end = (account_index[-1].strftime("%Y-%m-%d")
                       if len(account_index) else None)

        if dry_run:
            verdict = result["verdict"]
            stage = (verdict if verdict in evidence.SCORED_STAGES
                     else "insufficient_evidence")
        else:
            # Guarded: unguarded, any failure here (an evidence-schema edge
            # case, an I/O hiccup, anything record_abi1_outcome's own contract
            # didn't anticipate) would propagate out of compute_batch_abi1
            # and abandon EVERY not-yet-processed candidate with a bare
            # traceback -- no _skip entry, no per-family detail, and no
            # score for any sibling still waiting in the loop. Nothing is
            # written to the ledger before this call succeeds, so a re-run
            # just retries the family cleanly; the guard isolates the
            # failure to this one family, matching the curve/evaluate
            # try/except immediately above it.
            try:
                stage = abi1_driver.record_abi1_outcome(
                    ledger_path, family_id, run_id, tested_date, result,
                    catalog_entry=plan["entry"], instrument_set=effective,
                    cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
                    tax_policy=tax_policy, ticker_label=ticker_label,
                    signal_source_path=_SIGNAL_SOURCE_PATH,
                    signal_source_sha256=evidence.source_sha256(__file__),
                    **record_kwargs)
            except Exception as exc:
                batch5._skip(skipped, family_id, "record_error",
                            f"{type(exc).__name__}: {exc}")
                continue

        item = _public_item(
            family_id=family_id, shape=spec["shape"], tickers=effective,
            ticker_label=ticker_label, result=result, stage=stage,
            tax_policy=tax_policy, account_start=account_start,
            account_end=account_end)
        if pooled:
            item["pool_declared"] = plan["pool_declared"]
            item["pool_excluded"] = dict(excluded)
        if plan_history is not None:
            item["history_end"] = history_block
        computed.append(item)

    return {"computed": computed, "skipped": skipped,
            "input_coverage": coverage, "input_errors": load_errors,
            "benchmark_entry_cost_bps": benchmark_entry_cost_bps,
            **history_extra}


def _public_item(*, family_id: str, shape: str, tickers: list,
                 ticker_label, result: dict, stage: str, tax_policy: str,
                 account_start, account_end) -> dict:
    """Every field a caller (report/JSON output) needs, stripped of any
    pandas object -- the raw account/trade series stay inside `result` and
    are never carried into this JSON-safe item."""
    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    return {
        "family_id": family_id, "shape": shape, "tickers": list(tickers),
        "ticker_label": ticker_label,
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "account_start": account_start, "account_end": account_end,
        "time_in_market": result.get("time_in_market"),
        "time_in_market_full_window": result.get("time_in_market_full_window"),
        "tax_policy": tax_policy, "cash_convention": CASH_CONVENTION,
        "benchmark": MARKET_BENCHMARK,
        "holdout_evaluated": holdout.get("evaluated"),
        "holdout_passed": holdout.get("passed"),
        "holdout_refusal": holdout.get("refusal"),
        "holdout_psr": holdout.get("psr"),
        "window": {
            "account_sessions": window.get("account_sessions"),
            "scored_sessions": window.get("scored_sessions"),
            "in_sample_sessions": window.get("in_sample_sessions"),
            "holdout_sessions": window.get("holdout_sessions"),
            "holdout_split_applied": window.get("holdout_split_applied"),
        },
    }


# ---------------------------------------------------------------- reports

def _fmt(value, digits=4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_candidate_table(computed: list) -> str:
    header = ("| family | shape | tickers | stage | dsr_value | sessions | "
              "time in market | holdout |")
    rule = "|" + "---|" * 8
    rows = [header, rule]
    for item in sorted(computed, key=lambda i: i["family_id"]):
        label = item.get("ticker_label") or "/".join(item["tickers"])
        if item["holdout_evaluated"] is True:
            holdout = "pass" if item["holdout_passed"] else "fail"
        elif item.get("holdout_refusal"):
            holdout = item["holdout_refusal"]
        else:
            holdout = "n/a"
        rows.append(
            f"| {item['family_id']} | {item['shape']} | {label} | "
            f"{item['stage']} | {_fmt(item['dsr_value'])} | "
            f"{item['n_sessions'] if item['n_sessions'] is not None else 'n/a'} | "
            f"{_fmt(item['time_in_market'])} | {holdout} |")
    return "\n".join(rows) + "\n"


def _history_end_block(result: dict):
    """The run-level history-end statement for the results JSON / report, or
    None when the setting was off. `clause` is the ledger-note wording."""
    provenance = result.get("history_end_provenance")
    if provenance is not None:
        return {**provenance,
                "clause": history_cutoff.history_end_clause(provenance)}
    requested = result.get("history_end_requested")
    if requested is None:
        return None
    return {"history_end": requested, "opt_in": True,
            "decision": history_cutoff.DECISION, "clause": None,
            "by_symbol": {}}


def render_report(result: dict, *, tested_date: str, n_considered: int,
                  dry_run: bool) -> str:
    computed, skipped = result["computed"], result["skipped"]
    survivors = [item for item in computed if item["stage"] == "survived"]
    passed_dsr = [item for item in computed
                  if item["stage"] in ("survived", "cut_holdout", "cut_risk_refusal")]
    missing = [f"{row['family_id']}: {row['reason']}" for row in skipped]
    excluded_ruled_out = [row["family_id"] for row in skipped
                          if row["reason_code"] == "catalog_status"]
    text = reporting.render_run_report(
        date=tested_date, n_considered=n_considered,
        excluded_ruled_out=excluded_ruled_out, excluded_missing_data=missing,
        not_attempted_budget=[], n_passed_dsr=len(passed_dsr),
        n_survivors=len(survivors),
        survivor_briefs=[
            f"{item['family_id']} ({item.get('ticker_label') or '/'.join(item['tickers'])})"
            for item in survivors],
        budget_used="n/a (script-driven)", budget_ceiling="n/a",
        n_families_tested_this_run=len(computed),
        n_families_correction=float(constants.ABI1_N_TRIALS))
    history_block = _history_end_block(result)
    parts = [text]
    if history_block is not None:
        # The section comes BEFORE the candidate table heading, not between the
        # heading and its table.
        parts.append("\n## History end (opt-in)\n\n")
        parts.append((history_block.get("clause") or
                      f"History end {history_block['history_end']} was requested "
                      "(opt-in) but no series were loaded.")
                     + "\n")
        if dry_run and "history_end_guard_would_refuse" in result:
            refusal = result["history_end_guard_would_refuse"]
            parts.append("\nReal-run guard: " + (
                "a real run of this setting would be accepted." if refusal is None
                else f"a real run of this setting would be REFUSED: {refusal}")
                + "\n")
    parts.append("\n## ABI-1 candidate table\n\n")
    if dry_run:
        parts.append("DRY RUN -- no ledger record was written.\n\n")
    if history_block is None:
        parts.append("Every family here was scored on its FULL available price "
                     "history under the ABI-1 (account-relative) construction, "
                     "not a windowed per-trade backtest. `dsr_value` and "
                     "`sessions` are the in-sample layer; `holdout` reports the "
                     "separate out-of-sample confirmation.\n\n")
    else:
        parts.append(f"Every family here was scored on its price history ended at "
                     f"{history_block['history_end']} by the opt-in history-end "
                     "setting (the History end section above says what it "
                     "dropped), under the ABI-1 "
                     "(account-relative) construction, not a windowed per-trade "
                     "backtest. `dsr_value` and `sessions` are the in-sample "
                     "layer; `holdout` reports the separate out-of-sample "
                     "confirmation.\n\n")
    parts.append(render_candidate_table(computed))
    parts.append("\n## Skipped candidates\n\n")
    if skipped:
        for row in sorted(skipped, key=lambda r: r["family_id"]):
            parts.append(f"- {row['family_id']} [{row['reason_code']}]: "
                         f"{row['reason']}\n")
    else:
        parts.append("- none\n")
    return "".join(parts)


# ------------------------------------------------------------------- run

def run_batch_abi1(scope: dict, *, tested_date: str, run_id: str,
                   ledger_path: str, dry_run: bool, history_end=None,
                   max_history_drop: int = history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS
                   ) -> dict:
    # The history-end door comes BEFORE the synthetic-quote refusal, so a bad
    # cutoff / tested date / max-history-drop is reported as the standard
    # refusal instead of being preempted by an unrelated quote problem.
    try:
        _history_end_door(history_end, tested_date, max_history_drop)
    except history_cutoff.HistoryCutoffError as exc:
        result = _history_end_refused(exc, str(history_end))
    else:
        batch5.refuse_synthetic_quotes(scope["quotes"], dry_run=dry_run)
        result = compute_batch_abi1(
            scope, tested_date=tested_date, run_id=run_id,
            ledger_path=ledger_path, dry_run=dry_run, history_end=history_end,
            max_history_drop=max_history_drop)
    refusal = result.get("history_end_refusal")
    if refusal is not None:
        # A refused history end refuses the whole batch: no ledger row, no
        # results file and no run report (the caller writes none). The per-run
        # OHLC cache, if the load step already wrote it, is kept.
        # The standard refusal keys lead; this driver's own identifying keys
        # (a batch has no single family_id) follow.
        return {**refusal, "schema_version": 1, "construction": "ABI-1",
                "run_id": run_id, "tested_date": tested_date,
                "dry_run": dry_run, "refused": True,
                "results": [], "skipped": [], "outputs_written": []}
    run_dir = scope["run_dir"]
    report = render_report(
        result, tested_date=tested_date, n_considered=len(scope["accepted"]),
        dry_run=dry_run)
    if dry_run:
        # Matches run_batch5_mining.py's own convention: a dry run's report
        # is never named run-report.md, since results_log.py globs
        # <research_root>/*/run-report.md as the record of REAL runs.
        report_path = os.path.join(run_dir, "run-report-dry-run.md")
        batch3._atomic_write_text(report_path, report)
        written = [report_path]
    else:
        written = reporting.write_run_outputs(run_dir, report, {})
    spy_quote = scope["quotes"]["quotes"].get(MARKET_BENCHMARK)
    provenance = {
        "accepted_path": batch5.portable_path(scope["accepted_path"]),
        "accepted_sha256": scope["accepted_sha256"],
        "quotes_path": batch5.portable_path(scope["quotes_path"]),
        "quotes_sampled_at_local": scope["quotes"]["sampled_at_local"],
        "quotes_market_state": scope["quotes"]["market_state"],
        "quotes_source": scope["quotes"].get("source"),
        "spy_quote_cross_check": batch5._json_safe(spy_quote) if spy_quote else None,
        "modules_dir": batch5.portable_path(scope["modules_dir"]),
        "catalog_path": batch5.portable_path(scope["catalog_path"]),
        "ledger_path": batch5.portable_path(ledger_path),
        "families_filter": scope["families_filter"],
        "cash_convention": CASH_CONVENTION, "benchmark": MARKET_BENCHMARK,
    }
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": run_id,
        "tested_date": tested_date, "dry_run": dry_run,
        "results": [batch5._json_safe(item) for item in result["computed"]],
        "skipped": result["skipped"],
        "module_problems": scope["module_problems"],
        "unmatched_modules": scope["unmatched_modules"],
        "input_coverage": batch5._json_safe(result["input_coverage"]),
        "input_errors": result["input_errors"],
        "benchmark_entry_cost_bps": result.get("benchmark_entry_cost_bps"),
        "provenance": provenance,
        "outputs_written": [batch5.portable_path(path) for path in written],
    }
    history_block = _history_end_block(result)
    if history_block is not None:
        provenance["history_end"] = history_block
        payload["history_end"] = history_block
        # The string the operator typed, beside the provenance, under the same
        # key names every history-end driver uses.
        payload["history_end_requested"] = result["history_end_requested"]
        if "history_end_guard_would_refuse" in result:       # dry runs only
            payload["history_end_guard_would_refuse"] = (
                result["history_end_guard_would_refuse"])
    return payload


def _split_ids(raw) -> list:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _whole_number_of_sessions(text: str) -> int:
    """argparse type for --max-history-drop: a whole number of sessions >= 0."""
    if not (text.isascii() and text.isdigit()):
        raise argparse.ArgumentTypeError(
            f"must be a whole number of sessions >= 0, got {text!r}")
    return int(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--quotes", default=DEFAULT_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--families", default=None,
                        help="comma-separated subset of accepted family_ids")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--modules-dir", default=DEFAULT_MODULES_DIR)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument("--out", default=None,
                        help="results JSON path (default <research-root>/"
                             "<run-id>/abi1-results.json, or abi1-dry-run.json)")
    parser.add_argument("--history-end", default=None, metavar="YYYY-MM-DD",
                        help="opt-in: end the scored history at this NYSE "
                             "session (drops later sessions in memory, never "
                             "fills or patches). If any loaded series lacks a "
                             "bar on that date (or no series loads) the whole "
                             "batch is refused (exit 2; no ledger row, results "
                             "file or report; the per-run OHLC cache is kept, "
                             "so a real retry needs a fresh --run-id: this "
                             "dispatch has NO stale-cache refusal, a reused "
                             "--run-id silently reads the cache the refused "
                             "attempt wrote, so always use a fresh --run-id). "
                             "Give a dry run the real --tested-date: its "
                             "default is today's date, so a dry run "
                             "without it rehearses the guard against "
                             "today. PERMANENT for a "
                             "scored family: its account ends here (a new "
                             "family's ABI-1 holdout is the tail) and it cannot "
                             "be re-scored on longer history.")
    parser.add_argument("--max-history-drop", type=_whole_number_of_sessions,
                        default=history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS,
                        metavar="N",
                        help="with --history-end: refuse a real run if any "
                             "series would drop more than N sessions (default "
                             f"{history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS}). "
                             "A dry run only reports what the real run would "
                             "do (history_end_guard_would_refuse)")
    args = parser.parse_args(argv)
    # The history-end door runs before the scope is built (which reads the
    # accepted file, quotes, catalog and modules): a malformed cutoff, tested
    # date or max-history-drop is the refusal, whatever else is wrong.
    try:
        _history_end_door(args.history_end, args.tested_date, args.max_history_drop)
    except history_cutoff.HistoryCutoffError as exc:
        _print_history_end_refusal(history_cutoff.refusal_payload(
            exc, history_end_requested=str(args.history_end)), args.dry_run)
        return 2
    run_dir = os.path.join(args.research_root, args.run_id)
    output_path = args.out or os.path.join(
        run_dir, "abi1-dry-run.json" if args.dry_run else "abi1-results.json")
    scope = batch5.build_scope(
        accepted_path=args.accepted, quotes_path=args.quotes,
        catalog_path=args.catalog, ledger_path=args.ledger,
        modules_dir=args.modules_dir, run_dir=run_dir,
        families=_split_ids(args.families))
    payload = run_batch_abi1(
        scope, tested_date=args.tested_date, run_id=args.run_id,
        ledger_path=args.ledger, dry_run=args.dry_run,
        history_end=args.history_end, max_history_drop=args.max_history_drop)
    if payload.get("reason_code") in history_cutoff.HISTORY_END_REASON_CODES:
        # Every history-end refusal: the standard shape on stdout, exit 2, and
        # no results file, no report and no ledger row (a refusal raised after
        # the load step leaves the per-run OHLC cache that step wrote).
        _print_history_end_refusal(payload, args.dry_run)
        return 2
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    summary = {
        "output": output_path.replace(os.sep, "/"),
        "tested": len(payload["results"]),
        "skipped": len(payload["skipped"]),
        "survivors": sum(item["stage"] == "survived"
                         for item in payload["results"]),
        "dry_run": args.dry_run,
    }
    if "history_end_requested" in payload:      # only when the setting was given
        summary["history_end"] = payload["history_end_requested"]
        if "history_end_guard_would_refuse" in payload:      # dry runs only
            summary["history_end_guard_would_refuse"] = (
                payload["history_end_guard_would_refuse"])
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
