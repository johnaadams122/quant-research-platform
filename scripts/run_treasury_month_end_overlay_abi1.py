"""Bespoke ABI-1 scoring driver for treasury_month_end_index_demand_overlay --
an SPY core plus a margin-financed LONG overlay in IEF, TLT and AGG over the
last three scheduled sessions of each month. Uses the same quotes / dry-run /
refuse-synthetic / parameter-drift / already-scored checks as
scripts/run_ipo_fade_beta_neutral_abi1.py.

WHY A BESPOKE DRIVER: the generic scripts/run_abi1_mining.py dispatch builds a
single-position or rotation curve from a per-bar signal function. This family
has no per-bar signal -- its account is a benchmark-plus-overlay curve built by
``mining.batch5.treasury_month_end_index_demand_overlay.build_curve`` (math and
conventions in ``mining.long_overlay``, the calendar in the candidate module) --
and that module's ``make_signal_fn`` always raises so the generic path cannot
mis-score it. The curve is fed to ``abi1_driver.evaluate_family`` exactly as any
builder's dict is.

WHAT THE REPORT ADDS (report-only, never gating): an ``overlay_diagnostics``
block with the window count and calendar coverage, the share of sessions the
overlay is on, the worst and best window return, the financing actually charged,
the tax-identity approximation (measured and bounded heuristically, as for the
beta-neutral sibling), the OLS beta of the SCORED excess series on SPY (the
overlay is NOT designed to be beta-neutral; this is context, not a gate), the
ABI-1 risk-layer numbers (all FOUR refusals, measured on the OVERLAY-ONLY EXCESS
account (the overlay risk-basis rule), with ``failed_refusals`` naming
which fired, plus a report-only ``whole_account_reference`` block: the same four
on the whole SPY-core account, which holds SPY through 2008 (SPY's own drawdown
alone is about 55 percent), so the whole account clears the absolute cap only
if the overlay offsets most of it; kept so a reader sees the whole-account
number the rule does not gate on), and the disclosed NEXT-OPEN SENSITIVITY (the engine's own fill convention,
evaluated the same way but never recorded; the scored curve buys at the entry
close).

NOT MODELLED (each of these FLATTERS the strategy; written into the ledger
note): a real margin rate is the bill rate PLUS a spread and only the bare
^IRX-based rate is charged (the pipeline has no margin-rate source and
inventing one is not allowed); Reg-T margin calls and borrow availability; and
the tax walk treats financing interest as fully deductible against the
overlay's gain.

Usage (network: yfinance historical OHLC only). A --dry-run proves the wiring
against a SYNTHETIC quotes file the caller constructs (its quotes must NOT be
reported as a cost measurement); a scoring run needs a real dated quotes
capture (``batch5.refuse_synthetic_quotes``) and writes ONE permanent ledger
row:
    python scripts/run_treasury_month_end_overlay_abi1.py \\
        --dry-run --quotes <synthetic-quotes.json> --research-root <scratch dir> \\
        --catalog <a catalog json containing the family's entry>

    python scripts/run_treasury_month_end_overlay_abi1.py \\
        --quotes <real-dated-capture.json> --run-id <run-id> --tested-date <YYYY-MM-DD>

The quotes file needs bid/ask for IEF, TLT, AGG and SPY. A real run needs an
explicit --tested-date AND an explicit --run-id, and it REFUSES (reason_code
``stale_ohlc_cache``: no ledger row is written; main() still writes a small
results JSON carrying the refusal and exits 0; the per-run OHLC cache is kept)
to score any history that an earlier run left cached under research-root/run-id
(a dry run caches its bars there): use a
fresh --run-id or empty that run's cache dir. A dry run may re-use the cache,
and its result's ohlc_sources / last_bar_dates fields say which happened. A real
run also REFUSES (reason_code ``truncated_final_window``) when the data ends
inside an open window (T-2 or T-1), because that trade would be closed early at
the last close instead of at the pre-registered T: run it after the close of
that window's exit session. A dry run only reports ``final_window_truncated``.
That flag, ``window_entered_on_last_bar`` (data ending exactly on an entry date
opens nothing) and the skipped / not-started window counts are written into the
permanent ledger note as well as the JSON report.

DATA-CALENDAR GAPS ARE REFUSED, NEVER TRIMMED OR PATCHED (reason-coded, dates
named; no ledger row is written; main() still writes a small results JSON
carrying the refusal and exits 0; the per-run OHLC cache the load step already
wrote is kept, so a real retry needs a fresh --run-id; refetch, or wait for the
vendor to repair the history; the one opt-in exception, --history-end, is
described after this list):
  * ``unequal_trailing_coverage``: the legs and SPY must all END on the same
    session (otherwise a vendor missing one series' last bar would end the
    account a session early), e.g. one leg's vendor history missing the latest session
    while the other series have it;
  * ``calendar_mismatch``: the series must have identical sessions inside the
    legs' common span (SPY's longer lead history is sliced);
  * ``missing_exchange_session``: the shared calendar must contain EVERY actual
    NYSE session between its first and last date, so a session that every
    series lacks (invisible to the pairwise checks) is named and refused;
  * ``bar_on_closed_day``: a stray bar on a day the exchange was closed.
Known unscheduled closures (Hurricane Sandy 2012 and the like) are no gap. The
bill-rate series may lack bond-market holidays, which is expected (its last
print is carried).

THE ONE EXCEPTION IS AN EXPLICIT OPT-IN, ``--history-end YYYY-MM-DD`` (``compute_and_record(history_end=...)``, default unset = all of
the above unchanged). It ends the scored history at that NYSE session by
DROPPING every later session from every loaded series, in memory, after the
OHLC cache write (the cached CSVs keep the whole vendor history; the source
labels and the ``stale_ohlc_cache`` refusal are untouched). It adds, moves or
repairs no bar, and every check above, including ``truncated_final_window``,
still runs on the trimmed frames. A malformed or non-session cutoff is refused at
the very door, before the catalog, already-scored and quote checks and before any
history is loaded; so are a malformed tested date and an unusable
max_history_drop (both checked by ``history_cutoff.validate_guard_arguments``, on
a dry run too, because a dry run rehearses the real run), so neither can fail
after the load step wrote the per-run OHLC cache. The cutoff must be a real NYSE
session and EVERY loaded series (the bill series too, so a cutoff on a
bond-market holiday is refused: pick the session before) must hold a bar dated
exactly on it. When the history ends inside an open window the
``truncated_final_window`` refusal (the same reason_code, exit status and
results-file behaviour as without the setting) names the cutoff and says to choose a scheduled window exit session or wait.
Every refusal of the setting
has the one shape ``history_cutoff.refusal_payload`` builds (reason_code, reason,
history_end_requested, symbols, family_id) and exit status 2: reason_code
``history_end_invalid``, ``history_end_not_a_session``, ``history_end_not_covered``,
``history_end_empty`` and, on a REAL run only, the two guard refusals
``history_end_after_tested_date`` (the cutoff is later than the run's
--tested-date) and ``history_end_drop_too_large`` (some series would lose more
than --max-history-drop sessions, default
``history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS``; pass a larger N to accept a
larger drop). A history_end_* refusal writes no ledger row, no result file and
no report and main() exits 2; what it can leave behind is the per-run OHLC cache
the load step wrote before the check (every such refusal except the door ones,
``history_end_invalid`` and ``history_end_not_a_session``, which come before any
load): that cache is kept, and a real retry needs a fresh --run-id, because the
same run-id refuses ``stale_ohlc_cache``. A --dry-run never refuses on those two
guard refusals: it reports the message a real run would refuse with as
``history_end_guard_would_refuse`` (None when the real run would pass), so a
rehearsal predicts the real run; give the rehearsal the real --tested-date, or
the prediction uses the placeholder. The setting is stated in the ledger note
(the clause LEADS the note this driver supplies, ahead of the cost model and the
risk-layer and sensitivity text; the writer's own fixed metadata prefix still
comes first), the result (``history_end`` provenance,
``history_end_requested`` and ``raw_last_bar_dates``; ``last_bar_dates`` then
describes the trimmed history) and the printed report. It is PERMANENT for the
account it scores: the outcome row's account ends at the cutoff and a NEW family's
first_test pin (its ABI-1 holdout is the tail of that account) ends there too, so
the family cannot be re-scored on longer history. The pin is not unconditional:
``trial_ledger.record_first_test`` is a no-op for a family that already has a
first_test row, and a not_scoreable row pins none. The statement (the
``history_cutoff.history_end_clause`` text) is written on EVERY ledger row this
run writes, scored or not: ``abi1_driver.record_abi1_outcome`` writes ``extra_note``
on the four scored verdicts only, so the driver also passes the same clause as its
``unscored_note`` keyword, which becomes the note of the two unscored rows
(``not_scoreable`` and ``holdout_not_evaluable``); an unscored run still spends the
family's one permanent ABI-1 slot. With the setting off the keyword is not passed
and an unscored row keeps no note.

QUOTE PROVENANCE IS A HUMAN-PROCESS SAFEGUARD. ``batch5.refuse_synthetic_quotes``
only refuses a real run when the quotes file carries the word SYNTHETIC; it
cannot tell a measured book from an invented bid/ask file that omits the
marker (a pipeline-wide limit in shared code). Real quotes
must come from a measured broker quote capture, never typed
or estimated by hand.
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
                    evidence, history_cutoff, overlay_risk_report, trial_ledger)
from mining.batch5 import treasury_month_end_index_demand_overlay as mod

CASH_CONVENTION = "sweep_cash_daily"   # recorded; the account is always fully
                                       # deployed (SPY core), so idle cash is
                                       # never credited
MARKET_BENCHMARK = "SPY"
FINANCING_SERIES = "^IRX"
# "realization": overlay windows are short-term round trips whose net gains are
# taxed at exit with loss netting; see the module docstring for the disclosed
# dollar-attribution approximation (measured and reported below).
TAX_POLICY = "realization"

DEFAULT_RUN_ID = "treasury-month-end-overlay-abi1"   # dry runs only: a real run must pass its own --run-id
DEFAULT_TESTED_DATE = None   # dry runs only: None -> today's local date
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT

# The family's OWN, accurate statement of how the overlay-only risk-basis rule relates
# to its ledger rows (mining.overlay_risk_report.risk_layer_note).
ORDERING_CLAUSE = (
    "every ledger row of this family is scored under the overlay-only risk-basis "
    "rule, but the rule is not certified blind to this family: this driver's "
    "dry-run path can also report whole-account risk numbers")

_MAX_LISTED_DATES = 50     # dates listed in a calendar-gap refusal (the count is always given)

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


def _history_end_refusal(family_id: str, exc, history_end) -> dict:
    """The one standard refusal payload (``history_cutoff.refusal_payload``) for a
    history end this run will not use. main() exits 2 for every reason code in
    ``history_cutoff.HISTORY_END_REASON_CODES`` and writes no result file."""
    return history_cutoff.refusal_payload(
        exc, history_end_requested=history_end, family_id=family_id)


# the reason a truncated_final_window refusal gives with the setting OFF
_TRUNCATED_FINAL_WINDOW_REASON = (
    "the data ends inside an open window (its scheduled exit "
    "lies after the last bar), so the final trade would be "
    "closed early at the last close instead of at the "
    "pre-registered exit; run a real score after the close "
    "of that window's exit session T")


def _truncated_final_window_reason(history_provenance) -> str:
    """The ``reason`` of a ``truncated_final_window`` refusal (the reason_code, the
    exit status 0 and the results-file behaviour do not depend on it). With no
    history end it is the fixed text above. Under a history end the data can end
    inside an open window because the SETTING ended it there, and "run a real score
    after the close of that window's exit session" is then the wrong advice (the
    operator can also choose another cutoff), so the reason names the cutoff, says
    whether the setting dropped later sessions or the vendor's own last bar is that
    date, and says to choose a cutoff that is a scheduled window exit session or
    wait."""
    if history_provenance is None:
        return _TRUNCATED_FINAL_WINDOW_REASON
    cutoff = history_provenance["history_end"]
    dropped = history_provenance["dropped_dates"]
    if dropped:
        n = len(dropped)
        how = (f"--history-end {cutoff} ended the scored history inside that window "
               f"({n} later date{' was' if n == 1 else 's were'} dropped)")
    else:
        how = (f"the --history-end date {cutoff} is also the vendor's own last bar, "
               "so the setting dropped nothing")
    return ("the data ends inside an open window (its scheduled exit lies after the "
            f"last bar): {how}; the final trade would be closed early at the last "
            "close instead of at the pre-registered exit; choose a --history-end that "
            "is a scheduled window exit session (a month's last NYSE session), or "
            "wait until that window's exit session has closed and run again")


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def overlay_diagnostics(curve: dict, result: dict, *, untaxed_result: dict,
                        sensitivity_result: dict, benchmark_daily,
                        leg_cost_bps: dict, benchmark_entry_cost_bps: float) -> dict:
    """Report-only structural details for the built curve (see module
    docstring). Nothing here gates a verdict."""
    trades = curve["trades_df"]
    info = curve["window_info"]
    index = curve["strategy_daily"].index
    mask = curve["overlay_mask"].to_numpy(dtype=bool)
    diag = {
        "overlay_notional_fraction": float(mod.PARAMETERS["overlay_notional_fraction"]),
        "window_sessions": int(mod.PARAMETERS["window_sessions"]),
        "leg_cost_bps": {s: float(leg_cost_bps[s]) for s in mod.TRADED},
        "benchmark_entry_cost_bps": float(benchmark_entry_cost_bps),
        "n_windows": int(len(trades)),
        "windows_skipped_before_data": int(info["skipped_before_data"]),
        "windows_not_started": int(info["not_started"]),
        "final_window_truncated": bool(info["truncated_final"]),
        "window_entered_on_last_bar": bool(info["entered_on_last_bar"]),
        "overlay_sessions": int(mask.sum()),
        "account_sessions": int(len(index)),
        "overlay_session_share": _finite(mask.mean()) if len(index) else None,
        "mean_financing_rate_annual_on_overlay_sessions": (
            _finite(curve["financing_rate"].to_numpy(dtype=float)[mask].mean()) if mask.any() else None),
    }
    if len(trades):
        net = trades["net_ret"].to_numpy(dtype=float)
        diag.update({
            "first_window_entry": trades["entry_date"].iloc[0].strftime("%Y-%m-%d"),
            "last_window_exit": trades["exit_date"].iloc[-1].strftime("%Y-%m-%d"),
            "worst_window_net_ret": _finite(net.min()),
            "best_window_net_ret": _finite(net.max()),
            "sum_window_financing_ret": _finite(trades["financing_ret"].sum()),
            "sum_window_cost_ret": _finite(trades["cost_ret"].sum()),
        })
        # Dollar-attribution ratio netted_tax_factors implicitly assumes is 1:
        # the account's growth from the sizing point (the entry close) to the
        # exit close over (1 + the overlay's own net return).
        growth = np.cumprod(1.0 + curve["strategy_daily"].to_numpy(dtype=float))
        entry_pos = index.get_indexer(pd.DatetimeIndex(trades["entry_date"]))
        exit_pos = index.get_indexer(pd.DatetimeIndex(trades["exit_date"]))
        ratio = (growth[exit_pos] / growth[entry_pos]) / (1.0 + net)
        diag["tax_identity_ratio_min"] = _finite(ratio.min())
        diag["tax_identity_ratio_max"] = _finite(ratio.max())
        # The identity's error is NOT uniformly conservative (gains over-taxed
        # and losses over-sheltered where the ratio exceeds 1, the reverse
        # below). Give the reader a scale: the whole tax effect on the scored
        # daily-excess Sharpe (taxed minus untaxed evaluation of the same
        # curve) times the worst deviation of the ratio from 1. A HEURISTIC
        # SENSITIVITY, not a rigorous bound; reported, never gating.
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
    diag["scored_excess_beta_to_spy"] = _finite(beta["beta"])
    # Risk-basis rule: the four refusals gate on the OVERLAY-ONLY
    # EXCESS account (`layer2["account_basis"]`), not on the whole SPY-core
    # account. The whole-account numbers are kept beside them, report-only, so the
    # whole-account ceiling the rule does not gate on stays visible (see
    # mining.overlay_risk_report).
    diag["risk_layer"] = overlay_risk_report.risk_layer_report(result, benchmark_daily)
    sens_layer1 = sensitivity_result["layer1"]
    diag["next_open_sensitivity"] = {
        "convention": "overlay bought at the Open of the first window session "
                      "(the engine's next-open fill); report-only, never recorded",
        "sharpe_daily_excess": _finite(sens_layer1["sharpe_daily_excess"])
        if sens_layer1.get("sharpe_daily_excess") is not None else None,
        "dsr_value": _finite(sens_layer1["dsr_value"])
        if sens_layer1.get("dsr_value") is not None else None,
        "verdict": sensitivity_result["verdict"],
    }
    return diag


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT,
                       ohlc_loader=None, history_end=None,
                       max_history_drop=history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS) -> dict:
    family_id = mod.FAMILY_ID

    cutoff = None
    if history_end is not None:
        # Opt-in. Validate the setting at the very door,
        # before the catalog, already-scored and quote checks and before any history
        # load: an unusable value must never cost a fetch or hide behind another
        # refusal. The two inputs of the real-run guard below (the run's tested date
        # and max_history_drop) are validated here too, on a dry run as well (a dry
        # run rehearses the real run): the guard runs only AFTER the load step wrote
        # the per-run OHLC cache, so a malformed value must never be able to fail
        # there and leave a cache behind.
        try:
            cutoff = history_cutoff.parse_history_end(history_end)
            history_cutoff.validate_guard_arguments(
                tested_date=tested_date, max_drop=max_history_drop)
        except history_cutoff.HistoryCutoffError as exc:
            return _history_end_refusal(family_id, exc, history_end)

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
    priced = list(mod.TRADED) + [MARKET_BENCHMARK]
    costs = batch5.execution_costs_for(priced, quotes_payload)
    leg_cost_bps = {s: costs[s]["cost_bps"] for s in mod.TRADED}
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    loader = ohlc_loader or batch5.default_ohlc_loader
    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    frames, sources = {}, {}
    for symbol in priced:
        frames[symbol], sources[symbol] = batch5.load_frame(
            symbol, ohlc_loader=loader, cache_dir=cache_dir)
    # ^IRX is a signal-only series that can print at or below zero: it is loaded
    # WITHOUT the positivity gate meant for traded prices (never for a leg).
    irx_frame, irx_source = batch5.load_frame(
        FINANCING_SERIES, ohlc_loader=loader, cache_dir=cache_dir,
        require_positive=False)
    all_sources = {**sources, FINANCING_SERIES: irx_source}
    cached_symbols = [s for s, source in all_sources.items() if source == "cache"]
    if cached_symbols and not dry_run:
        # A permanent one-shot row must be scored from histories fetched by THIS
        # run: bars cached under research-root/run-id by an earlier run (a dry
        # run, or an aborted real one) can be stale or truncated.
        return {"family_id": family_id, "reason_code": "stale_ohlc_cache",
                "cached_symbols": cached_symbols,
                "reason": ("a real run must score freshly fetched histories, but "
                           f"{', '.join(cached_symbols)} came from an OHLC cache left "
                           f"under run id {run_id!r} by an earlier run; use a fresh "
                           "--run-id (or empty that run's cache dir) and run again")}
    history_provenance = None
    guard_would_refuse = None
    if cutoff is not None:
        # Drop every session after the cutoff from EVERY loaded series, in memory
        # only. This is after the cache write and the stale-cache refusal above, so
        # the cached CSVs keep the whole vendor history, and it reassigns the
        # names every later line reads: nothing below can see a post-cutoff bar.
        try:
            trimmed, history_provenance = history_cutoff.trim_frames_to_history_end(
                {**frames, FINANCING_SERIES: irx_frame}, cutoff)
        except history_cutoff.HistoryCutoffError as exc:
            return _history_end_refusal(family_id, exc, history_end)
        # The real-run guard: a real run permanently spends the family's one-shot
        # slot, so a cutoff later than the run's tested date, or one that drops more
        # sessions from some series than max_history_drop allows, is refused before
        # anything is scored or written. A dry run is never refused: it reports the
        # message a real run would refuse with, so a rehearsal predicts the real run.
        try:
            history_cutoff.check_real_run_guard(
                history_provenance, tested_date=tested_date, max_drop=max_history_drop)
        except history_cutoff.HistoryCutoffError as exc:
            if not dry_run:
                return _history_end_refusal(family_id, exc, history_end)
            guard_would_refuse = str(exc)
        irx_frame = trimmed.pop(FINANCING_SERIES)
        frames = trimmed
    spy_frame = frames[MARKET_BENCHMARK]
    leg_frames = {s: frames[s] for s in mod.TRADED}

    try:
        curve = mod.build_curve(
            spy_frame, leg_frames, irx_frame, leg_cost_bps=leg_cost_bps,
            spy_entry_cost_bps=benchmark_entry_cost_bps)
    except mod.CalendarCoverageError as exc:
        # A data-calendar gap is refused, never trimmed or patched: a clean
        # reason-coded refusal that names the dates (no ledger row is written;
        # main() still writes a small results JSON carrying the refusal and
        # exits 0; the per-run OHLC cache is kept).
        return {"family_id": family_id, "reason_code": exc.reason_code,
                "reason": str(exc),
                "missing_dates": exc.missing_dates[:_MAX_LISTED_DATES],
                "n_missing_dates": len(exc.missing_dates),
                "extra_dates": exc.extra_dates[:_MAX_LISTED_DATES],
                "n_extra_dates": len(exc.extra_dates)}
    if curve["window_info"]["truncated_final"] and not dry_run:
        # The data ends INSIDE a window (T-2 or T-1): the trade would be closed
        # early at the last close (possibly a partial bar) instead of at the
        # pre-registered T. Never bake that into a permanent one-shot row.
        return {"family_id": family_id, "reason_code": "truncated_final_window",
                "reason": _truncated_final_window_reason(history_provenance)}

    account_type = entry["account_types"][0]
    eval_kwargs = dict(benchmark_frame=spy_frame,
                       benchmark_entry_cost_bps=benchmark_entry_cost_bps,
                       account_type=account_type)
    result = abi1_driver.evaluate_family(curve, tax_policy=TAX_POLICY, **eval_kwargs)
    benchmark_daily = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        spy_frame, entry_cost_bps=benchmark_entry_cost_bps)
    # Risk-basis rule: the ledger note below states the risk basis and that the
    # excess is the overlay's own; refuse to record anything either claim is false for.
    overlay_risk_report.require_overlay_basis(result, family_id=family_id)
    overlay_risk_report.require_excess_is_overlay_only(
        result["account_daily"], benchmark_daily, curve["overlay_mask"],
        family_id=family_id)
    # report-only counterfactuals (never scored, never recorded)
    untaxed_result = abi1_driver.evaluate_family(curve, tax_policy="none", **eval_kwargs)
    sensitivity_curve = mod.build_next_open_sensitivity(
        spy_frame, leg_frames, irx_frame, leg_cost_bps=leg_cost_bps,
        spy_entry_cost_bps=benchmark_entry_cost_bps)
    sensitivity_result = abi1_driver.evaluate_family(
        sensitivity_curve, tax_policy=TAX_POLICY, **eval_kwargs)

    diag = overlay_diagnostics(
        curve, result, untaxed_result=untaxed_result,
        sensitivity_result=sensitivity_result, benchmark_daily=benchmark_daily,
        leg_cost_bps=leg_cost_bps, benchmark_entry_cost_bps=benchmark_entry_cost_bps)

    cost_text = ", ".join(f"{s}={leg_cost_bps[s]:.2f}bps" for s in mod.TRADED)
    risk_note = overlay_risk_report.risk_layer_note(
        diag["risk_layer"], ordering_clause=ORDERING_CLAUSE)
    extra_note = (
        f"cost model: {cost_text} round trip per leg per window "
        f"({batch5.quotes_cost_source(quotes_payload)}); design=benchmark-plus-overlay "
        "account (SPY core fully invested at all times PLUS a margin-financed "
        f"LONG-ONLY overlay of total notional x={diag['overlay_notional_fraction']} of equity in "
        "IEF, TLT and AGG, equal thirds, no hedge leg and no short leg) over the last "
        f"{diag['window_sessions']} scheduled NYSE sessions of each month; entry at the close of "
        "the session before the window (T-3) and exit at the close of the month's last "
        "scheduled session (T); the month-end schedule is the exchange's regular-holiday "
        "schedule fixed in advance (exchange_calendars XNYS, unscheduled closures NOT "
        "removed), never read off the data -- see mining.long_overlay and the candidate "
        "module; shares fixed at entry, sized on the account equity at the entry close; "
        "every window is charged a full round trip on every leg; financing = the whole "
        "overlay notional borrowed at the ^IRX 13-week bill coupon-equivalent yield of the "
        "last close strictly before each session, on actual calendar days over 365, mean "
        f"{diag['mean_financing_rate_annual_on_overlay_sessions']} on overlay sessions; "
        "NOT MODELLED, and each of these flatters the strategy: a real margin rate is the "
        "bill rate PLUS a spread and no spread is charged (the pipeline has no margin-rate "
        "source and inventing one is not allowed), Reg-T margin calls and borrow "
        "availability; the tax walk (ABI-1 realization netting) deducts financing interest "
        "against the overlay's gain; "
        f"{risk_note}; tax: netted_tax_factors "
        "dollar-attribution approximation (not uniformly conservative), measured ratio "
        f"range [{diag.get('tax_identity_ratio_min')}, {diag.get('tax_identity_ratio_max')}], "
        f"whole tax effect on the daily-excess Sharpe {diag.get('tax_effect_on_daily_excess_sharpe')}, "
        f"heuristic sensitivity (not a rigorous bound) {diag.get('tax_identity_sensitivity_on_sharpe')}; "
        "DISCLOSED SENSITIVITY (report-only, not the scored convention): the engine's "
        "next-open fill (overlay bought at the Open of the first window session) gives "
        f"daily-excess Sharpe {diag['next_open_sensitivity']['sharpe_daily_excess']} "
        f"against {diag.get('sharpe_daily_excess_taxed')} for the scored close-fill "
        "convention; "
        f"measured: {diag['n_windows']} windows over {diag['overlay_sessions']} of "
        f"{diag['account_sessions']} sessions, scored-excess OLS beta to SPY "
        f"{diag['scored_excess_beta_to_spy']}; account span "
        f"{curve['strategy_daily'].index[0]:%Y-%m-%d}..{curve['strategy_daily'].index[-1]:%Y-%m-%d} "
        "(the legs' common span: it starts at the latest-listing leg's first session and "
        "every series must end on the same session); window coverage: "
        f"windows_skipped_before_data={diag['windows_skipped_before_data']}; "
        f"windows_not_started={diag['windows_not_started']}; "
        f"final_window_truncated={diag['final_window_truncated']}; "
        f"window_entered_on_last_bar={diag['window_entered_on_last_bar']}; "
        "a DIFFERENT strategy definition from every "
        "other month-end or Treasury family (own family_id, own trial); "
        "PRE-REGISTRATION LIMITS: no constant was tuned against any ABI-1 score and all were "
        "fixed before this pipeline computed any return of this account, but they were NOT "
        "all chosen blind: some were fixed after descriptive looks at these same ETFs "
        "(in-sample data only; never the holdout), covering the window length and the "
        "fill convention; those looks ran outside the trial ledger, so DSR's "
        "trial count does not charge them; the 3-session window is the paper's headline "
        "example; "
        f"price data: IEF {sources['IEF']}, TLT {sources['TLT']}, AGG {sources['AGG']}, "
        f"SPY {sources['SPY']}, ^IRX {irx_source}, through "
        f"{leg_frames['IEF'].index[-1]:%Y-%m-%d} (legs) / {spy_frame.index[-1]:%Y-%m-%d} (SPY)")
    history_clause = None
    if history_provenance is not None:
        # The clause LEADS the driver-supplied note (as in the gold and commodity
        # drivers), so a reader learns the account was ended early before the long
        # risk-layer and sensitivity text. This note has no other mandatory leading
        # text; the writer's own fixed metadata prefix still comes first.
        history_clause = history_cutoff.history_end_clause(history_provenance)
        extra_note = f"{history_clause} {extra_note}"
    # record_abi1_outcome writes extra_note on the four SCORED verdicts only, and an
    # unscored run still spends the family's one permanent ABI-1 slot, so with a
    # history end the same clause is also handed over as unscored_note (the note of
    # the two unscored rows). With the setting off the keyword is not passed at
    # all.
    unscored_kwargs = {} if history_clause is None else {"unscored_note": history_clause}

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
            signal_source_sha256=signal_source_sha256, **unscored_kwargs)

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    account_index = result["account_daily"].index
    report = {
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
        # run under the SAME run_id. Only a dry run can get here with a cache
        # hit: a real run REFUSES it (reason_code stale_ohlc_cache) so a
        # permanent row is never scored from stale dry-run bars.
        "ohlc_sources": all_sources,
        "last_bar_dates": {
            **{s: frames[s].index[-1].strftime("%Y-%m-%d") for s in priced},
            FINANCING_SERIES: irx_frame.index[-1].strftime("%Y-%m-%d")},
        "overlay_diagnostics": diag,
    }
    if history_provenance is not None:
        # last_bar_dates above describes the TRIMMED history; the vendor's own
        # last bars sit beside it, and the full statement of the setting rides along
        # (history_end and history_end_requested are the same key names in every
        # driver)
        report["history_end"] = history_provenance
        report["history_end_requested"] = str(history_end)
        report["raw_last_bar_dates"] = {
            s: history_provenance["by_symbol"][s]["raw_last"]
            for s in [*priced, FINANCING_SERIES]}
        if dry_run:
            # what the real-run guard would say (None: it would pass); a real run
            # that got this far passed the guard, so it has nothing to predict
            report["history_end_guard_would_refuse"] = guard_would_refuse
    return report


def _non_negative_int(text: str) -> int:
    """argparse type for --max-history-drop: a whole number of sessions >= 0."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a whole number of sessions >= 0, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"must be a whole number of sessions >= 0, got {text!r}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", required=True,
                        help="Path to a dated quotes JSON file with bid/ask for "
                             "IEF, TLT, AGG AND SPY (must be a REAL captured book "
                             "unless --dry-run is also passed, in which case a "
                             "SYNTHETIC placeholder file is expected).")
    parser.add_argument("--run-id", default=None,
                        help="REQUIRED for a real run (it names the OHLC cache dir, "
                             "and a real run refuses any history cached there by an "
                             "earlier run, so it must be a fresh id); a --dry-run "
                             "falls back to a shared default.")
    parser.add_argument("--tested-date", default=None,
                        help="YYYY-MM-DD; REQUIRED for a real run (it is written "
                             "into a permanent ledger row); a --dry-run falls "
                             "back to a placeholder. With --history-end a malformed "
                             "value is refused at the door (exit status 2, dry runs "
                             "too), before anything is loaded or written.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument("--history-end", default=None, metavar="YYYY-MM-DD",
                        help="opt-in history-end setting: end the scored history at "
                             "this NYSE session and drop every later session from every "
                             "series (it adds, moves or repairs no bar). It must be a real "
                             "session with a bar in EVERY series, or the run is refused: "
                             "exit status 2, no ledger row, result file or report (the "
                             "per-run OHLC cache the load step wrote is kept, so a real "
                             "retry needs a fresh --run-id). permanent for the account it "
                             "scores: the outcome row's account ends at this date and a "
                             "NEW family's ABI-1 holdout pin ends there too, so the family "
                             "cannot be re-scored on longer history. A REAL run also "
                             "refuses a date later than --tested-date, or a drop above "
                             "--max-history-drop; a --dry-run reports that as "
                             "history_end_guard_would_refuse instead (give it the real "
                             "--tested-date). Unset (the default) changes nothing.")
    parser.add_argument("--max-history-drop", type=_non_negative_int,
                        default=history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS, metavar="N",
                        help="refuse a real run if any series would drop more than N "
                             "sessions (a whole number >= 0, default "
                             f"{history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS}); pass a "
                             "larger N to accept a larger drop. Only used with "
                             "--history-end.")
    args = parser.parse_args(argv)
    if args.tested_date is None:
        if not args.dry_run:
            parser.error("--tested-date is required for a real (non --dry-run) "
                         "run: it is written into a permanent ledger row and "
                         "must never fall back to a stale default")
        args.tested_date = DEFAULT_TESTED_DATE or pd.Timestamp.now().strftime("%Y-%m-%d")
    if args.run_id is None:
        if not args.dry_run:
            parser.error("--run-id is required for a real (non --dry-run) run: it "
                         "names the OHLC cache dir, and a real run refuses history "
                         "cached there by an earlier run (a dry run's bars), so it "
                         "must be a fresh id")
        args.run_id = DEFAULT_RUN_ID

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes,
        ledger_path=args.ledger, run_id=args.run_id, tested_date=args.tested_date,
        dry_run=args.dry_run, signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__),
        research_root=args.research_root, history_end=args.history_end,
        max_history_drop=args.max_history_drop)

    if result.get("reason_code") in history_cutoff.HISTORY_END_REASON_CODES:
        # A history end this run refuses (any of the codes in
        # history_cutoff.HISTORY_END_REASON_CODES): no ledger row, no result file and
        # no report; the refusal is printed and the exit status is 2. The per-run
        # OHLC cache a refusal after the load leaves behind is kept.
        print(json.dumps({"output": None, "result": batch5._json_safe(result)}))
        return 2
    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    safe_result = batch5._json_safe(result)
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": _SIGNAL_SOURCE_PATH, "result": safe_result,
    }
    if args.history_end is not None:
        payload["history_end_requested"] = args.history_end
    output_path = os.path.join(
        run_dir,
        "treasury-month-end-overlay-dry-run.json" if args.dry_run
        else "treasury-month-end-overlay-results.json")
    batch3._atomic_write_text(
        output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
