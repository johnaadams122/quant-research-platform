"""Bespoke ABI-1 driver for commodity_open_interest_cycle_commodity_bond_overlay --
an SPY core plus a margin-financed LONG overlay held in ONE leg chosen by the CFTC
open-interest state (DBC when HIGH, TLT when LOW). Uses the same quotes /
dry-run / refuse-synthetic / parameter-drift / already-scored / stale-cache /
truncated-final-window / calendar-gap checks and the same overlay_risk_report
usage as scripts/run_treasury_month_end_overlay_abi1.py.

WHY A BESPOKE DRIVER: the generic scripts/run_abi1_mining.py dispatch builds a
single-position or rotation curve from a per-bar price signal. This family's
signal is a monthly open-interest state read from the cached CFTC series
(``mining.cftc_data``), and its account is a benchmark-plus-overlay curve with a
per-window leg set that the documented ``mining.long_overlay`` builder does not take
directly; ``mining.batch5.commodity_open_interest_cycle_commodity_bond_overlay``
composes it (one builder call per leg, summed) without editing it, and its
``make_signal_fn`` always raises so the generic path cannot mis-score it. The curve
goes to ``abi1_driver.evaluate_family`` like any builder's dict.

WHAT THE REPORT ADDS (report-only, never gating):
  * state checks: HIGH share and regime-switch counts overall, by pre-declared
    sub-period, by calendar year and inside the ABI-1 holdout, plus the number of
    carried blackout month-ends (so a reader can see how close the account is to a
    static long-TLT overlay);
  * a CONTROL ARM: an always-TLT 0.25x overlay (same account, costs, financing, first
    entry) evaluated the same way, merged (one long position) and re-sized monthly
    (the two conventions bracket a continuing position), printed beside the main
    result. PRE-DECLARED INTERPRETATION RULE: a pass of the strategy is only
    meaningful if its daily-excess Sharpe (same taxed evaluation) exceeds that of BOTH
    always-TLT controls, i.e. the state rule adds something over plain long-duration
    exposure; the ledger note states this rule and this run's numbers. Controls have no
    ledger row of their own;
  * the WINDOW CONVENTION (one switch:
    ``mining.batch5.<family>.SCORED_MERGE_SAME_LEG``, False): the scored account is MONTHLY-RESIZED
    (every calendar month is its own window, a full round trip and a re-size to 0.25 of equity every
    month even when the leg continues, so every window is short-term taxed); the OTHER convention (merged:
    consecutive same-leg months are one window with a fixed share count) is computed beside it as a
    report-only sensitivity under ``window_convention_sensitivity``;
  * a TAX-HOLDING check: this pipeline's realization tax walk charges the short-term
    rate on a round trip held 365 days or less and a zero long-term rate on one held longer,
    so a merged window held beyond a year realizes its gain untaxed (the merged convention
    flatters the strategy through the tax walk, which is why it is not the scored one); the report
    and the ledger note give the number of windows held beyond a year and the share of positive
    window return they carry, for the scored account and for the sensitivity;
  * the next-open-fill sensitivity, the taxed-versus-untaxed difference, the tax-identity
    approximation and the OLS beta of the scored excess on SPY, as for the treasury sibling;
  * the ABI-1 risk-layer numbers (all FOUR refusals on the OVERLAY-ONLY EXCESS account per
    the overlay-only-excess risk basis, with ``failed_refusals`` and a report-only whole-account reference).

NOT MODELLED (each of the first two FLATTERS the strategy; written into the ledger note):
no margin spread (financing is the bare ^IRX bill rate; the pipeline has no margin-rate
source and inventing one is not allowed); DBC is energy-heavy while the signal is
equal-weight across four sectors; Reg-T margin calls and borrow availability; the tax
walk treats financing interest as deductible against the overlay's gain.

Usage (network: yfinance historical OHLC only; the CFTC data is read from the cache
``data/mining/cftc_oi`` written by ``python -m mining.cftc_data --default-cache-dir``):
    python scripts/run_commodity_oi_cycle_overlay_abi1.py \\
        --dry-run --quotes <synthetic-quotes.json> --research-root <scratch dir> \\
        --catalog <a catalog json containing the family's entry>

    python scripts/run_commodity_oi_cycle_overlay_abi1.py \\
        --quotes <real-dated-capture.json> --run-id <run-id> --tested-date <YYYY-MM-DD>

A real run needs an explicit --tested-date AND --run-id, real dated bid/ask quotes for DBC,
TLT and SPY (``batch5.refuse_synthetic_quotes``; provenance is a human-process safeguard,
see the treasury sibling), freshly fetched prices (``stale_ohlc_cache``), a data end after
the close of the last window's exit (``truncated_final_window``), and a CFTC cache in which EVERY
contract reaches the report the last decision needs (``cftc_cache_stale``; ``cftc_contract_stale`` names a
contract that ends earlier than the others) and no scored month-end reads a report older than 14 days for
any contract (``cftc_report_stale_at_month_end``; a mid-month dry run checks only the month-ends that have happened). DATA-CALENDAR GAPS ARE
REFUSED, NEVER TRIMMED OR PATCHED, with a reason code and the dates (``unequal_trailing_
coverage``, ``calendar_mismatch``, ``missing_exchange_session``, ``bar_on_closed_day``):
a vendor's missing session in several ETFs is exactly such a refusal, unless the opt-in history end
below is given. A missing,
corrupt or incomplete CFTC cache is refused (``cftc_cache_missing`` / ``cftc_cache_corrupt`` /
``cftc_cache_incomplete``) and an undefined state is refused (``state_unavailable``).

OPT-IN HISTORY END (the opt-in history-end setting; ``--history-end YYYY-MM-DD``, default off). Instead of refusing a calendar
hole, a run can be told to END the scored history at a chosen NYSE session: every loaded series (DBC, TLT, SPY and
^IRX) is cut to the sessions dated on or before that date, in memory only and AFTER its OHLC cache was written and the
``stale_ohlc_cache`` check ran, so the per-run OHLC cache keeps the full vendor history. Later sessions are only
DROPPED, never filled, patched or shifted, and every calendar, hole, CFTC-freshness and final-window check above runs
unchanged on the trimmed frames: a REAL run still needs the last bar to be exactly a scheduled month-end session
(``truncated_final_window``), so a cutoff inside a window scores only as a dry run. The CFTC cache is not trimmed (it
is not a price history), but the open-interest state table is built only from the month-end decisions on or before the
cutoff, so no report usable only after the history end is read; the ledger note says so.

The setting fails closed: it must be a strict YYYY-MM-DD date that is a real NYSE session (``history_end_invalid`` /
``history_end_not_a_session``, refused before anything is fetched, so nothing at all is written) and EVERY series must
hold a bar dated exactly on it (``history_end_not_covered``, refused after the load and naming the series). With a history
end set, a malformed ``--tested-date`` or ``max_history_drop`` is refused at the same door (``history_end_invalid``, real and
dry runs alike, before anything is fetched or written, so no OHLC cache is written either), because the guard below would
otherwise fail only after the load step had written that cache. A REAL run is
also guarded against spending its one-shot slot by mistake: it is refused when the cutoff is later than the run's
``--tested-date`` (``history_end_after_tested_date``) or when any series would lose more than ``--max-history-drop N``
sessions (default ``mining.history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS``; ``history_end_drop_too_large``). A DRY run
never refuses on either count; it says what a real run would do in ``history_end_guard_would_refuse`` (the refusal text,
or None), judged against the dry run's tested date (today's date when ``--tested-date`` is omitted). Every
history-end refusal has the one standard shape of ``mining.history_cutoff.refusal_payload`` (``reason_code``, ``reason``,
``history_end_requested`` = the raw setting, ``symbols``, ``family_id``). A refusal after the load step (not covered,
after the tested date, drop too large) writes no ledger row, no result file and no report; the per-run OHLC cache that
step wrote is kept, so a real retry needs a fresh --run-id (the same run id refuses ``stale_ohlc_cache``). The command
ends with exit status 2 for every history-end refusal (the status the treasury, gold/copper and generic drivers use for
the same refusal; for every refusal unrelated to the history end ``main()`` writes its small results JSON and
exits 0). A ``truncated_final_window`` refusal under a history end says so in its reason: it names the cutoff, says the setting
ended the data inside an open window, and says a real score needs a cutoff that is exactly a scheduled month-end session (or a
run on the evening of the month-end).

The setting is stated in the ledger note (a leading HISTORY END clause from
``mining.history_cutoff.history_end_clause``), in the result (``history_end`` provenance, ``history_end_requested`` and
``raw_last_bar_dates``; ``last_bar_dates`` describes the trimmed history) and in the payload file, and it is PERMANENT
for the account it scores: the outcome row's account ends at the cutoff and a NEW family's first_test pin (its ABI-1
holdout is the tail of that account) ends there too, so the family cannot be re-scored on longer history. The pin is
not unconditional: ``trial_ledger.record_first_test`` does nothing for a family that already has a first_test row, and a
not_scoreable row pins none. The shared recorder ``abi1_driver.record_abi1_outcome`` writes the
driver's ``extra_note`` on the scored stages alone, and an unscored run (``not_scoreable`` or
``holdout_not_evaluable``) still spends the family's one permanent slot, so the same clause is also passed as the
recorder's ``unscored_note`` keyword, which it writes verbatim as the note of those two rows. The keyword is passed ONLY
when the setting is on: with it off the recorder call passes no ``unscored_note`` and an unscored row carries no note.

QUOTE CAPTURE HOURS. ``batch5.refuse_synthetic_quotes`` only refuses a real run when the
quotes file carries the word SYNTHETIC; it cannot tell a live book from a closed-market or old one. On top
of it a REAL run refuses (reason_code ``quotes_not_market_hours``; nothing is fetched, cached or added to the ledger, but
``main()`` still writes its small results JSON and exits 0; dry runs
exempt) unless the capture's ``market_state`` starts with ``open`` (any case, whitespace ignored), its
``sampled_at_local`` is an explicit ISO date and time (naive means US Eastern, an offset-aware time is
converted to Eastern; a bare clock time or the word now, which the loose parser would read off this
machine's clock, is refused), its Eastern date is the run's ``--tested-date`` and its Eastern time of day
is within 09:45 to 15:45 Eastern inclusive (``quotes_market_hours_refusal``; it never raises).

THE CONTROLS GATE THE PRE-REGISTERED PASS, NOT THE RECORDED STAGE. The stage vocabulary
is closed in shared code (``evidence.VALID_STAGES``) and every tested family is recorded unconditionally, so
a main ABI-1 ``survived`` is recorded as such. The result's ``pre_registered_pass`` is True only when the
ABI-1 verdict is ``survived`` AND this family's own control condition (the strategy's daily-excess Sharpe
exceeds that of BOTH always-TLT controls) is exactly True; when the verdict is ``survived`` and the
condition is False or missing, the driver-supplied ledger note LEADS with ``CONTROL_CONDITION_FAILED_FLAG``
and the printed summary says the same. A stage that did not survive carries no clause.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
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
from mining import (abi1_driver, benchmark_relative, catalog as catalog_mod, cftc_data,
                    constants, evidence, history_cutoff, overlay_risk_report, trial_ledger)
from mining.batch5 import commodity_open_interest_cycle_commodity_bond_overlay as mod

CASH_CONVENTION = "sweep_cash_daily"   # recorded; the account is always fully deployed (SPY core)
MARKET_BENCHMARK = "SPY"
FINANCING_SERIES = "^IRX"
TAX_POLICY = "realization"

DEFAULT_RUN_ID = "commodity-oi-cycle-overlay-abi1-v1"   # dry runs only: a real run must pass its own --run-id
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()  # today, at run time
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
STATE_HISTORY_START = "1986-01-02"        # the first CFTC report in the cache

ORDERING_CLAUSE = (
    "this rule was fixed before this family's driver or any dry-run path of it "
    "existed, so no look at this family's numbers preceded the rule")

_MAX_LISTED_DATES = 50

# the two window conventions, keyed by ``merge_same_leg`` (the scored one is ``mod.SCORED_MERGE_SAME_LEG``)
_CONVENTION = {
    False: ("monthly-resized",
            "every calendar month is its own window, a full round trip and a re-size to 0.25 of equity every "
            "month even when the leg continues (conservative on cost), each window short-term taxed"),
    True: ("merged",
           "consecutive months with the same leg are one window (one round trip) with a fixed share count, so "
           "its notional drifts from 0.25 of equity"),
}
_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


# A REAL run prices its one-shot verdict off a quote capture, so it also refuses a capture that was not taken
# during the regular session of the run's own date. After-hours cost samples are
# unusable; the shared synthetic-quote guard only looks for the word SYNTHETIC and cannot tell
# a closed-market book from a live one.
MARKET_TIMEZONE = "America/New_York"
CAPTURE_WINDOW_START = (9, 45)     # Eastern, inclusive
CAPTURE_WINDOW_END = (15, 45)      # Eastern, inclusive
# an explicit ISO date and time: the loose pandas parser would read "now" or a bare "10:00" off THIS machine's clock
_ISO_DATE_TIME = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}")
_ISO_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}")


def quotes_market_hours_refusal(quotes_payload, tested_date):
    """``None`` when the quote capture is acceptable for a REAL run, else the text of why not. Accepted only
    when (i) ``market_state`` (case-insensitive, surrounding whitespace ignored) starts with ``open``, (ii)
    ``sampled_at_local`` is a string that starts with an explicit ISO date and time and parses (a naive time is
    read as US Eastern, an offset-aware one is converted to it), (iii) its Eastern calendar date equals
    ``tested_date`` and (iv) its Eastern time of day is between 09:45:00 and 15:45:00 inclusive. Never raises on
    a bad payload: a missing, non-string or unparseable field, a non-dict payload and a daylight-saving gap or
    overlap are all refusals."""
    payload = quotes_payload if isinstance(quotes_payload, dict) else {}
    problems = []
    state = payload.get("market_state")
    if not isinstance(state, str) or not state.strip().lower().startswith("open"):
        problems.append(f"market_state {state!r} does not start with 'open'")
    sampled = payload.get("sampled_at_local")
    stamp = None
    if isinstance(sampled, str) and _ISO_DATE_TIME.match(sampled):
        try:
            parsed = pd.Timestamp(sampled.strip())
            if not pd.isna(parsed):
                stamp = (parsed.tz_localize(MARKET_TIMEZONE) if parsed.tzinfo is None
                         else parsed.tz_convert(MARKET_TIMEZONE))
        except Exception:      # noqa: BLE001 -- deliberate: a bad capture must be a refusal, never a crash
            stamp = None
    if stamp is None:
        problems.append(f"sampled_at_local {sampled!r} is missing or is not an explicit ISO date and time "
                        "that parses")
    tested = None
    if isinstance(tested_date, datetime.date) or (isinstance(tested_date, str) and _ISO_DATE.match(tested_date)):
        try:
            parsed_tested = pd.Timestamp(tested_date)
            tested = None if pd.isna(parsed_tested) else parsed_tested.date()
        except Exception:      # noqa: BLE001
            tested = None
    if tested is None:
        problems.append(f"the run's tested date {tested_date!r} does not parse as a date")
    if stamp is not None:
        if tested is not None and stamp.date() != tested:
            problems.append(f"sampled_at_local date {stamp.date():%Y-%m-%d} (Eastern) is not the run's tested "
                            f"date {tested:%Y-%m-%d}")
        clock = (stamp.hour, stamp.minute, stamp.second, stamp.microsecond)
        if not ((*CAPTURE_WINDOW_START, 0, 0) <= clock <= (*CAPTURE_WINDOW_END, 0, 0)):
            problems.append(f"sampled_at_local time {stamp:%H:%M:%S} Eastern is outside the "
                            f"{CAPTURE_WINDOW_START[0]:02d}:{CAPTURE_WINDOW_START[1]:02d} to "
                            f"{CAPTURE_WINDOW_END[0]:02d}:{CAPTURE_WINDOW_END[1]:02d} Eastern capture window")
    return "; ".join(problems) if problems else None


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def cache_fingerprint(snapshot) -> str:
    """One sha256 over the per-contract raw-file hashes in the pre-registered order."""
    entries = snapshot.manifest["contracts"]
    joined = "|".join(f"{code}:{entries[code]['sha256']}" for code in cftc_data.CONTRACT_CODES)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def tax_holding_diagnostics(trades_df: pd.DataFrame) -> dict:
    """How much of a curve's window return the tax walk treats as LONG-TERM. The pipeline charges the
    short-term rate on a round trip held ``LONG_TERM_HOLDING_DAYS`` or fewer and the long-term rate,
    which is zero, on one held longer, so a merged window held beyond a year is realized untaxed."""
    out = {"long_term_holding_days": int(constants.LONG_TERM_HOLDING_DAYS),
           "ltcg_tax_rate": float(constants.LTCG_TAX_RATE), "stcg_tax_rate": float(constants.STCG_TAX_RATE),
           "n_trades": int(len(trades_df)), "n_long_term": 0, "positive_net_return_share_long_term": None}
    if not len(trades_df):
        return out
    held = (pd.DatetimeIndex(trades_df["exit_date"]) - pd.DatetimeIndex(trades_df["entry_date"])).days.to_numpy()
    long_term = held > int(constants.LONG_TERM_HOLDING_DAYS)
    net = trades_df["net_ret"].to_numpy(dtype=float)
    positive = net > 0.0
    out["n_long_term"] = int(long_term.sum())
    if positive.any():
        out["positive_net_return_share_long_term"] = _finite(net[positive & long_term].sum() / net[positive].sum())
    return out


def _control_report(name: str, curve: dict, result: dict, benchmark_daily) -> dict:
    risk = overlay_risk_report.risk_layer_report(result, benchmark_daily)
    return {
        "name": name,
        "tax_holding": tax_holding_diagnostics(curve["trades_df"]),
        "n_trades": int(len(curve["trades_df"])),
        "sharpe_daily_excess": _finite(result["layer1"]["sharpe_daily_excess"])
        if result["layer1"].get("sharpe_daily_excess") is not None else None,
        "dsr_value": _finite(result["layer1"]["dsr_value"]) if result["layer1"].get("dsr_value") is not None else None,
        "verdict": result["verdict"],
        "excess_account_max_drawdown": risk["account_max_drawdown"],
        "failed_refusals": risk["failed_refusals"],
    }


# ------------------------------------------------------ interpretation rule

CONTROL_NAMES = ("always_tlt_merged", "always_tlt_monthly_resized")

# The stage vocabulary is CLOSED in shared code (evidence.VALID_STAGES) and every tested family is recorded
# unconditionally, so the controls cannot change the recorded stage. They gate the PRE-REGISTERED PASS instead:
# a main ABI-1 'survived' whose strategy did not beat BOTH always-TLT controls is recorded as
# 'survived' but is NOT a survivor under this family's own rule, and the driver-supplied note and the printed
# summary say so in their first sentence.
CONTROL_CONDITION_FAILED_FLAG = (
    "CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under this family's own "
    "pre-registered rule (the strategy's taxed daily-excess Sharpe must exceed that of BOTH always-TLT "
    "controls, the merged and the monthly-resized one). This is not a pass; do not promote it without review.")
PRE_REGISTERED_RULE_MET_CLAUSE = (
    "PRE-REGISTERED RULE MET: the ABI-1 stage survived AND the control condition holds (the strategy's "
    "taxed daily-excess Sharpe exceeds that of BOTH always-TLT controls); pre_registered_pass=True.")


def interpret_controls(strategy_sharpe, controls: dict) -> dict:
    """The PRE-DECLARED control condition: per always-TLT control whether the strategy's taxed daily-excess
    Sharpe STRICTLY exceeds it (``None`` when either Sharpe is missing), and ``beats_both``: True only when both
    controls are present and beaten, ``None`` when any comparison is missing, else False."""
    beats = {}
    for name in CONTROL_NAMES:
        control = controls.get(name)
        control_sharpe = None if control is None else control.get("sharpe_daily_excess")
        beats[name] = (None if strategy_sharpe is None or control_sharpe is None
                       else bool(float(strategy_sharpe) > float(control_sharpe)))
    beats_both = None if any(v is None for v in beats.values()) else bool(all(beats.values()))
    return {**beats, "beats_both": beats_both}


def pre_registered_pass(verdict, control_condition) -> bool:
    """True only when the ABI-1 verdict is ``survived`` AND the control condition is exactly True (a missing or
    False condition is not a pass)."""
    return bool(verdict == "survived" and control_condition is True)


def pre_registered_clause(verdict, control_condition) -> str:
    """The sentence a survived stage carries: the unmissable flag when the control condition did not hold
    (False or missing), the rule-met statement when it did, and nothing for any other verdict."""
    if verdict != "survived":
        return ""
    return PRE_REGISTERED_RULE_MET_CLAUSE if control_condition is True else CONTROL_CONDITION_FAILED_FLAG


def note_with_pre_registered_lead(body: str, verdict, control_condition) -> str:
    """The driver-supplied ledger note: the pre-registered clause FIRST (the shared writer puts its own fixed
    metadata prefix ahead of it, which is not editable from here), then the body."""
    clause = pre_registered_clause(verdict, control_condition)
    return f"{clause} {body}" if clause else body


def overlay_diagnostics(curve: dict, result: dict, *, untaxed_result: dict, sensitivity_result: dict,
                        benchmark_daily, leg_cost_bps: dict, benchmark_entry_cost_bps: float) -> dict:
    """Report-only structural checks for the built curve. Nothing here gates a verdict."""
    trades = curve["trades_df"]
    info = curve["window_info"]
    index = curve["strategy_daily"].index
    mask = curve["overlay_mask"].to_numpy(dtype=bool)
    legs = curve["leg_by_trade"]
    diag = {
        "overlay_notional_fraction": float(mod.OVERLAY_NOTIONAL),
        "leg_cost_bps": {s: float(leg_cost_bps[s]) for s in mod.TRADED},
        "benchmark_entry_cost_bps": float(benchmark_entry_cost_bps),
        "n_windows": int(len(trades)),
        "n_windows_by_leg": {leg: int(sum(1 for x in legs if x == leg)) for leg in mod.TRADED},
        "n_monthly_decisions": int(len(info["monthly"])),
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
        growth = np.cumprod(1.0 + curve["strategy_daily"].to_numpy(dtype=float))
        entry_pos = index.get_indexer(pd.DatetimeIndex(trades["entry_date"]))
        exit_pos = index.get_indexer(pd.DatetimeIndex(trades["exit_date"]))
        ratio = (growth[exit_pos] / growth[entry_pos]) / (1.0 + net)
        diag["tax_identity_ratio_min"] = _finite(ratio.min())
        diag["tax_identity_ratio_max"] = _finite(ratio.max())
        taxed = result["layer1"].get("sharpe_daily_excess")
        untaxed = untaxed_result["layer1"].get("sharpe_daily_excess")
        if taxed is not None and untaxed is not None:
            effect = float(taxed) - float(untaxed)
            worst = max(abs(diag["tax_identity_ratio_min"] - 1.0), abs(diag["tax_identity_ratio_max"] - 1.0))
            diag.update({
                "sharpe_daily_excess_taxed": _finite(taxed),
                "sharpe_daily_excess_untaxed": _finite(untaxed),
                "tax_effect_on_daily_excess_sharpe": _finite(effect),
                "tax_identity_sensitivity_on_sharpe": _finite(worst * abs(effect)),
            })
    beta = benchmark_relative.ols_alpha_beta(result["log_excess"], benchmark_daily)
    diag["scored_excess_beta_to_spy"] = _finite(beta["beta"])
    diag["risk_layer"] = overlay_risk_report.risk_layer_report(result, benchmark_daily)
    sens_layer1 = sensitivity_result["layer1"]
    diag["next_open_sensitivity"] = {
        "convention": "each window's leg bought at the Open of the first window session "
                      "(the engine's next-open fill); report-only, never recorded",
        "sharpe_daily_excess": _finite(sens_layer1["sharpe_daily_excess"])
        if sens_layer1.get("sharpe_daily_excess") is not None else None,
        "dsr_value": _finite(sens_layer1["dsr_value"]) if sens_layer1.get("dsr_value") is not None else None,
        "verdict": sensitivity_result["verdict"],
    }
    return diag


def _state_summary_text(state_diag: dict) -> str:
    by = state_diag["by_period"]
    parts = ", ".join(
        f"{name}: {v['evaluations']} months, HIGH {round(v['high_share'], 3)}, {v['switches']} switches"
        for name, v in by.items() if v["evaluations"])
    holdout = state_diag.get("holdout") or {}
    holdout_text = (f"holdout months {holdout.get('evaluations')}, HIGH share "
                    f"{None if holdout.get('high_share') is None else round(holdout['high_share'], 3)}, "
                    f"{holdout.get('carried_blackout_month_ends')} carried blackout month-ends" if holdout else "no holdout split")
    return (f"{state_diag['evaluations']} monthly evaluations, HIGH share {round(state_diag['high_share'], 3)}, "
            f"{state_diag['switches']} regime switches, longest HIGH run {state_diag['longest_run_months']['HIGH']} "
            f"and LOW run {state_diag['longest_run_months']['LOW']} months; by sub-period ({parts}); {holdout_text}")


_HISTORY_END_REASONS = history_cutoff.HISTORY_END_REASON_CODES      # the six codes main() ends with exit status 2


def cftc_history_end_sentence(cutoff: str) -> str:
    """The sentence a scored ledger note carries under a history end about the open-interest data (the shared clause is
    about the price histories only): the CFTC cache was not trimmed, yet the state table read nothing usable only later."""
    return (f"CFTC CACHE UNDER THE HISTORY END: the CFTC open-interest cache was not trimmed (only the price histories "
            f"were, so its newest-report date and fingerprint above describe the whole cache), but the state table was "
            f"built only from the month-end decisions on or before {cutoff}, so no report usable only after the history "
            f"end {cutoff} was read")


def _history_end_refusal(family_id: str, exc, setting) -> dict:
    """The refusal payload for a history-end setting this run cannot honor (the opt-in history end), in the one
    standard shape every driver uses (``mining.history_cutoff.refusal_payload``): the exception's machine-readable
    reason code and text, the raw setting as ``history_end_requested``, the series it names as ``symbols`` and this
    family's id."""
    return history_cutoff.refusal_payload(exc, history_end_requested=setting, family_id=family_id)


def truncated_final_window_reason(history_provenance, last_bar) -> str:
    """The reason text of the ``truncated_final_window`` refusal. With no history end (``history_provenance`` None) it
    is the standard text. Under a history end the data was ENDED by the setting inside an open window,
    so the text names the cutoff and says so, and says what a real score needs instead: a cutoff that is exactly a
    scheduled month-end session (the example is the last scheduled month-end session before the cutoff) or a run on
    the evening of the month-end. ``last_bar`` is the last bar of the (trimmed) history."""
    if history_provenance is None:
        return ("the data ends inside an open month window (its scheduled exit lies "
                "after the last bar), so the final block would be closed early at the "
                "last close instead of at the pre-registered month-end; run a real "
                "score after the close of that month's last scheduled session")
    cutoff = history_provenance["history_end"]
    earlier = [d for d in mod.month_end_sessions(last_bar - pd.Timedelta(days=45), last_bar) if d < last_bar]
    example = f" (for example {earlier[-1]:%Y-%m-%d})" if earlier else ""
    return (f"the history end {cutoff} (--history-end) ended the data inside an open month "
            "window (its scheduled exit lies after the last bar), so the final block would be closed early at the "
            "last close instead of at the pre-registered month-end; the setting, not the vendor data, cut the "
            f"history there; a real score needs a cutoff that is exactly a scheduled month-end session{example}, or "
            "a run on the evening of the month-end, after that month's last scheduled session, without the setting")


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str, run_id: str,
                       tested_date: str, dry_run: bool, signal_source_path: str,
                       signal_source_sha256: str, research_root: str = RESEARCH_ROOT,
                       ohlc_loader=None, cftc_cache_dir: str = cftc_data.DEFAULT_CACHE_DIR,
                       history_end=None,
                       max_history_drop=history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS) -> dict:
    family_id = mod.FAMILY_ID
    if history_end is not None:
        # opt-in setting: a malformed or non-session cutoff is refused before anything is
        # read, fetched or written. Whether every series covers it can only be known after the load, below.
        # The real-run guard's two inputs (the run's tested date and max_history_drop) are checked HERE too, for real
        # and dry runs alike (a dry run rehearses the real run), so neither can fail only AFTER the load step wrote
        # the per-run OHLC cache. Nothing of this runs with the setting off.
        try:
            history_cutoff.parse_history_end(history_end)
            history_cutoff.validate_guard_arguments(tested_date=tested_date, max_drop=max_history_drop)
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

    try:
        quotes_payload = batch5.load_quotes(quotes_path)
    except ValueError as exc:
        # A REAL run whose capture lacks its own metadata (sampled_at_local / market_state as a non-empty
        # string) is refused with the same reason code as an off-hours capture
        # instead of surfacing the shared loader's bare ValueError. Any other loader error, and every
        # dry-run error, still raises. The reason names the fields, never the file path.
        # match the shared loader's exact TERMINAL message shape (its message begins with the file path, so a
        # substring test would misread a path that happens to contain a field name)
        missing = [f for f in ("sampled_at_local", "market_state")
                   if str(exc).endswith(f": '{f}' must be a non-empty string")]
        if dry_run or not missing:
            raise
        return {"family_id": family_id, "reason_code": "quotes_not_market_hours",
                "reason": "the quote capture lacks a usable " + " / ".join(missing)
                          + " (each must be a non-empty string)"}
    batch5.refuse_synthetic_quotes(quotes_payload, dry_run=dry_run)
    if not dry_run:
        # a real run's quote capture must be an open-market sample dated the run date, taken
        # 09:45 to 15:45 Eastern. Dry runs (synthetic quotes by design) are exempt. Nothing has been
        # fetched, read from the CFTC cache or written when this refuses.
        hours_refusal = quotes_market_hours_refusal(quotes_payload, tested_date)
        if hours_refusal is not None:
            return {"family_id": family_id, "reason_code": "quotes_not_market_hours", "reason": hours_refusal}
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
    irx_frame, irx_source = batch5.load_frame(
        FINANCING_SERIES, ohlc_loader=loader, cache_dir=cache_dir, require_positive=False)
    all_sources = {**sources, FINANCING_SERIES: irx_source}
    cached_symbols = [s for s, source in all_sources.items() if source == "cache"]
    if cached_symbols and not dry_run:
        return {"family_id": family_id, "reason_code": "stale_ohlc_cache",
                "cached_symbols": cached_symbols,
                "reason": ("a real run must score freshly fetched histories, but "
                           f"{', '.join(cached_symbols)} came from an OHLC cache left "
                           f"under run id {run_id!r} by an earlier run; use a fresh "
                           "--run-id (or empty that run's cache dir) and run again")}
    history_provenance = None
    history_guard_would_refuse = None
    if history_end is not None:
        # opt-in: END the scored history at the cutoff by dropping every later session from
        # EVERY loaded series, in memory only. It runs AFTER the frames were loaded and cached (the per-run cache
        # keeps the full vendor history) and after the stale-cache refusal above; every line below, the calendar,
        # hole, CFTC and final-window checks included, reads only the trimmed frames (frames and irx_frame are
        # rebound here, and nothing keeps a reference to the untrimmed ones).
        try:
            trimmed, history_provenance = history_cutoff.trim_frames_to_history_end(
                {**frames, FINANCING_SERIES: irx_frame}, history_end)
        except history_cutoff.HistoryCutoffError as exc:
            return _history_end_refusal(family_id, exc, history_end)
        # the real-run guard: a real run permanently spends the family's one-shot slot, so a cutoff later than the
        # run's tested date or an unexpectedly large drop is refused (nothing is written); a dry run never refuses but
        # reports the same verdict, so a rehearsal predicts the real run
        try:
            history_cutoff.check_real_run_guard(
                history_provenance, tested_date=tested_date, max_drop=max_history_drop)
        except history_cutoff.HistoryCutoffError as exc:
            if not dry_run:
                return _history_end_refusal(family_id, exc, history_end)
            history_guard_would_refuse = str(exc)
        irx_frame = trimmed.pop(FINANCING_SERIES)
        frames = trimmed
    spy_frame = frames[MARKET_BENCHMARK]
    leg_frames = {s: frames[s] for s in mod.TRADED}

    # the open-interest state: read from the verified cache, refuse anything stale or undefined
    try:
        snapshot = cftc_data.load_cached_open_interest(cftc_cache_dir)
    except cftc_data.CftcCacheError as exc:
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc)}
    decisions = mod.month_end_sessions(pd.Timestamp(STATE_HISTORY_START), spy_frame.index[-1])
    happened = [d for d in decisions if d <= spy_frame.index[-1]]     # a mid-month dry run has a coming month-end with no report yet
    if history_end is not None:
        # under a history end the account ends at the cutoff, so the state table is built from the month-end decisions
        # that have HAPPENED (on or before it) only: the COMING month-end lies after the cutoff and its state would be
        # read from reports usable only after the history end, which no scored window needs and which must not be able
        # to crash the run (the CFTC cache itself is not trimmed). Rows up to the cutoff are identical either way, and
        # with the setting off nothing changes.
        decisions = pd.DatetimeIndex(happened)
    try:
        mod.require_fresh_cache(snapshot, happened)                     # EVERY contract must reach the last decision's report
        mod.require_recent_reads(snapshot.series, happened)             # and no scored month-end may read a stale report
    except mod.SignalDataError as exc:
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc)}
    table = mod.signal_state_table(snapshot.series, decisions)

    curve_kwargs = dict(leg_cost_bps=leg_cost_bps, spy_entry_cost_bps=benchmark_entry_cost_bps)
    scored_merge = bool(mod.SCORED_MERGE_SAME_LEG)                # THE one switch
    alt_merge = not scored_merge
    scored_name, scored_desc = _CONVENTION[scored_merge]
    alt_name, alt_desc = _CONVENTION[alt_merge]
    try:
        curve = mod.build_curve(spy_frame, leg_frames, irx_frame, table, merge_same_leg=scored_merge,
                                **curve_kwargs)
    except mod.CalendarCoverageError as exc:
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc),
                "missing_dates": exc.missing_dates[:_MAX_LISTED_DATES],
                "n_missing_dates": len(exc.missing_dates),
                "extra_dates": exc.extra_dates[:_MAX_LISTED_DATES],
                "n_extra_dates": len(exc.extra_dates)}
    except mod.SignalDataError as exc:
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc)}
    if curve["window_info"]["truncated_final"] and not dry_run:
        return {"family_id": family_id, "reason_code": "truncated_final_window",
                "reason": truncated_final_window_reason(history_provenance, spy_frame.index[-1])}

    account_type = entry["account_types"][0]
    eval_kwargs = dict(benchmark_frame=spy_frame, benchmark_entry_cost_bps=benchmark_entry_cost_bps,
                       account_type=account_type)
    result = abi1_driver.evaluate_family(curve, tax_policy=TAX_POLICY, **eval_kwargs)
    benchmark_daily = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        spy_frame, entry_cost_bps=benchmark_entry_cost_bps)
    overlay_risk_report.require_overlay_basis(result, family_id=family_id)
    overlay_risk_report.require_excess_is_overlay_only(
        result["account_daily"], benchmark_daily, curve["overlay_mask"], family_id=family_id)
    # report-only counterfactuals (never scored, never recorded)
    untaxed_result = abi1_driver.evaluate_family(curve, tax_policy="none", **eval_kwargs)
    sensitivity_curve = mod.build_next_open_sensitivity(spy_frame, leg_frames, irx_frame, table, **curve_kwargs)
    sensitivity_result = abi1_driver.evaluate_family(sensitivity_curve, tax_policy=TAX_POLICY, **eval_kwargs)
    alt_curve = mod.build_curve(spy_frame, leg_frames, irx_frame, table, merge_same_leg=alt_merge, **curve_kwargs)
    alt_result = abi1_driver.evaluate_family(alt_curve, tax_policy=TAX_POLICY, **eval_kwargs)
    controls = {}
    for name, merge in (("always_tlt_merged", True), ("always_tlt_monthly_resized", False)):
        control_curve = mod.build_control_curve(spy_frame, leg_frames, irx_frame, table, merge_same_leg=merge,
                                                **curve_kwargs)
        control_result = abi1_driver.evaluate_family(control_curve, tax_policy=TAX_POLICY, **eval_kwargs)
        controls[name] = _control_report(name, control_curve, control_result, benchmark_daily)
    strategy_sharpe = result["layer1"].get("sharpe_daily_excess")
    beats = interpret_controls(strategy_sharpe, controls)
    control_condition = beats["beats_both"]

    diag = overlay_diagnostics(
        curve, result, untaxed_result=untaxed_result, sensitivity_result=sensitivity_result,
        benchmark_daily=benchmark_daily, leg_cost_bps=leg_cost_bps,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps)
    diag["tax_holding"] = tax_holding_diagnostics(curve["trades_df"])
    diag["window_convention"] = {"scored": scored_name, "sensitivity": alt_name,
                                 "scored_merge_same_leg": scored_merge}
    diag["window_convention_sensitivity"] = _control_report(f"strategy_{alt_name}", alt_curve, alt_result,
                                                            benchmark_daily)
    diag["controls"] = controls
    diag["strategy_beats_control"] = beats
    window = result["window"]
    account_index = result["account_daily"].index
    holdout_start = (account_index[-window["holdout_sessions"]]
                     if window.get("holdout_split_applied") and window.get("holdout_sessions") else None)
    first_entry = curve["window_info"]["monthly"][0][0] if curve["window_info"]["monthly"] else account_index[0]
    state_diag = mod.state_diagnostics(table, start=first_entry, end=account_index[-1], holdout_start=holdout_start)
    diag["state_diagnostics"] = state_diag
    diag["cftc_cache"] = {
        "retrieved_at": snapshot.manifest.get("retrieved_at"),
        "latest_asof": snapshot.latest_asof.strftime("%Y-%m-%d"),
        "n_contracts": len(snapshot.series),
        "fingerprint_sha256": cache_fingerprint(snapshot),
        "point_in_time_status": snapshot.point_in_time_status,
    }
    used_states = table.loc[(table.index >= first_entry) & (table.index <= account_index[-1])]
    blackouts = [d.strftime("%Y-%m-%d") for d in used_states.index[used_states["blackout"]]]

    cost_text = ", ".join(f"{s}={leg_cost_bps[s]:.2f}bps" for s in mod.TRADED)
    risk_note = overlay_risk_report.risk_layer_note(diag["risk_layer"], ordering_clause=ORDERING_CLAUSE)

    def _fmt(value, digits=4):
        return None if value is None else round(float(value), digits)

    control_text = "; ".join(
        f"{name} daily-excess Sharpe {_fmt(c['sharpe_daily_excess'])} (DSR {_fmt(c['dsr_value'], 6)}, "
        f"verdict {c['verdict']}, {c['n_trades']} trades)" for name, c in controls.items())
    alt = diag["window_convention_sensitivity"]
    cache_info = diag["cftc_cache"]
    note_body = (
        f"cost model: {cost_text} round trip per leg per window "
        f"({batch5.quotes_cost_source(quotes_payload)}); design=benchmark-plus-overlay account "
        "(SPY core fully invested at all times PLUS a margin-financed LONG-ONLY overlay of total "
        f"notional x={diag['overlay_notional_fraction']} of equity in ONE leg chosen by the CFTC "
        "open-interest state, DBC when the state is HIGH and TLT when it is LOW, no hedge leg and no "
        "short leg); one pooled trial (one family, one permanent ledger row); signal: CFTC Legacy "
        "Futures-Only open_interest_all (Socrata dataset 6dca-aqww), contract counts not dollar open "
        "interest, 18 contracts in four sectors (agriculture, energy, livestock, metals), per-contract "
        "monthly log growth at each scheduled month-end from the last report usable by then, averaged "
        "within each sector, the sectors equal-weighted, CMI = the mean of the last 12 monthly values, "
        "state HIGH when CMI is strictly above the expanding median of CMI (current month included, "
        "minimum 36 monthly observations) else LOW, decided at the close of each scheduled month-end "
        "session from the exchange's regular-holiday schedule fixed in advance and never read off the "
        "data; DEVIATIONS FROM Hong and Yogo (2012), FIVE in all: contract counts, not dollar open "
        "interest; contract-level growth averaged within each sector, then the sectors equal-weighted and "
        "averaged arithmetically over 12 months, not the paper's aggregate-then-growth with a geometric "
        "average; 18 contracts, not about 30 commodities; a median split, not a regression; TLT, not the "
        "10-year note; POINT-IN-TIME STATUS: the CFTC series is NOT verified never-revised (documented "
        "corrections for the reports as of 2017-03-28, 2018-09-18 and 2019-10-15, and the Socrata "
        "dataset serves the corrected values, not as first published) and is never labelled "
        f"never-revised; cache retrieved {cache_info['retrieved_at']}, newest report as of "
        f"{cache_info['latest_asof']}, cache fingerprint {cache_info['fingerprint_sha256']}; "
        "AVAILABILITY: a report as of Tuesday D is usable from D+6 calendar days (three days of margin in "
        "a normal week of the weekly era and none in a holiday-shifted week, immaterial to the state: a 7-day "
        "lag changes only the tie month 2019-11-29 within the scored span); five scored month-ends (2009-11-30, 2011-02-28, 2015-11-30, 2020-11-30 and 2022-02-28) "
        "read a report from a US-federal-holiday week whose plain D+6 date falls on the decision day itself, the CFTC "
        "usually moves such a release to the Monday afternoon, about 30 minutes before the close on the decision day "
        "(not verified against a CFTC notice), and withholding those five reports on their D+6 date changes no "
        "state; in the three CFTC "
        "publication blackouts (the 2013, 2018-19 and 2025-26 federal shutdowns) that lag is false, so "
        "a publication blackout month-end refuses to update the state and carries the prior state, is "
        "left out of the expanding median and reads the last report that was really public; the 2018-19 "
        "blackout starts with the Christmas-shifted report as of Monday 2018-12-24 (CFTC press release "
        "7864-19), so 2019-02-28 is a blackout month-end too, and the 2025-26 table keeps the CFTC's "
        "first plan's outer bound although its accelerated plan cleared the backlog on 2025-12-29, so "
        "2025-12-31 is carried though its report was already public (conservative); "
        f"blackout month-ends in this account's span: {', '.join(blackouts) or 'none'} "
        f"({diag['state_diagnostics']['holdout']['carried_blackout_month_ends'] if diag['state_diagnostics']['holdout'] else 0} "
        "of them inside the ABI-1 holdout); a CMI whose 12-month base month is a blackout month-end "
        "measures a window longer than 12 months (2014-10-31, 2019-12-31, 2020-01-31 and 2020-02-28 in "
        "the scored span), disclosed and not repaired; TIE MONTH: at 2019-11-29 CMI equals its expanding "
        "median exactly and the pre-registered strict rule resolves that exact tie to LOW, and the tie "
        "exists only because blackout month-ends are left out of the expanding median (without that exclusion "
        "the rule gives HIGH there) and is also "
        "sensitive to the unverified pre-2000 release timing (an extra 7 or 14 days of lag on the pre-2000 reports "
        "alone flips it, and no other scored state moves); "
        "COMPOSITION DRIFT of the expanding-median baseline: grains "
        "start 1998, lean hogs 1996, natural gas 1990, copper 1989 and RBOB 2006, so the early CMI is "
        "built from fewer contracts (measured on the cached counts, 241 of the 476 CMI values, 50.6 "
        "percent, have a 12-month window with fewer than 18 contributing contracts, all 18 contribute "
        "from 2006-03-31 and 12 of the 247 evaluation months since 2006-02 are still partial), and "
        "palladium has real reporting gaps of 84, 168 and 259 days in 2000-2002 whose stale carried value "
        "(up to 247 days) changes no state when treated as missing; SIXTH DISCLOSED LIMITATION (a limitation, not a "
        "deviation from the paper, so the deviations stay FIVE): the 18-contract universe is a present-day survivor "
        "set, the contracts that exist and report cleanly today, with no membership rule applied at each historical "
        "date, so a contract that stopped trading before 2026 cannot be in it; STATE (report-only checks measured this run, no return "
        f"involved): {_state_summary_text(state_diag)}; an account whose state is mostly LOW after 2008 "
        "is close to a static long-TLT overlay, so the effective number of independent regime blocks "
        f"({diag['n_windows']} scored windows here, {alt['n_trades']} in the {alt_name} sensitivity) is small; "
        "WINDOW CONVENTION "
        "(one switch): the scored convention is "
        f"{scored_name}: {scored_desc}; the report-only {alt_name} sensitivity ({alt_desc}) has "
        f"daily-excess Sharpe {_fmt(alt['sharpe_daily_excess'])} "
        f"against {_fmt(diag.get('sharpe_daily_excess_taxed'))} for the scored convention; "
        "TAX-HOLDING EFFECT (why merging is not the scored convention: it flatters the strategy through the "
        "tax walk, and that is not the cost effect merging was made for): "
        "this pipeline's realization tax walk charges the short-term rate on a round trip held "
        f"{diag['tax_holding']['long_term_holding_days']} days or fewer and the long-term rate, "
        f"{diag['tax_holding']['ltcg_tax_rate']} here, on one held longer, so a merged window held beyond a year "
        f"realizes its gain untaxed; {diag['tax_holding']['n_long_term']} of {diag['tax_holding']['n_trades']} "
        "scored windows in this run were held beyond a year (carrying "
        f"{diag['tax_holding']['positive_net_return_share_long_term']} of the positive net window return) and "
        f"{alt['tax_holding']['n_long_term']} of {alt['tax_holding']['n_trades']} windows of the {alt_name} "
        "sensitivity (carrying "
        f"{alt['tax_holding']['positive_net_return_share_long_term']} of its positive net window return); "
        f"CONTROL ARM (report-only, no ledger row of its own): an always-TLT 0.25x overlay with the same "
        f"account, costs, financing and first entry, measured this run: {control_text}; PRE-DECLARED "
        "INTERPRETATION RULE: a pass is only meaningful if the strategy's daily-excess Sharpe exceeds "
        "that of both always-TLT controls (the state rule must add something over plain long-duration "
        f"exposure); strategy beats the controls this run: {json.dumps(diag['strategy_beats_control'])}; "
        f"pre_registered_pass={pre_registered_pass(result['verdict'], control_condition)} (True only when the "
        "ABI-1 verdict is survived AND the strategy beats both always-TLT controls; this run's ABI-1 verdict "
        f"is {result['verdict']}); "
        "NOT MODELLED, and each of the first two flatters the strategy: no margin spread (financing is "
        "the bare ^IRX bill rate on the whole borrowed notional; a real margin rate adds a spread and "
        "the pipeline has no margin-rate source), DBC is energy-heavy while the signal is equal-weight "
        "across four sectors, Reg-T margin calls and borrow availability, and the tax walk (ABI-1 "
        "realization netting) deducts financing interest against the overlay's gain; "
        f"{risk_note}; the guard that the excess is overlay-only can only verify the sessions before "
        "the first window because the overlay is on in essentially every month after it; tax: "
        "netted_tax_factors dollar-attribution approximation (not uniformly conservative), measured "
        f"ratio range [{diag.get('tax_identity_ratio_min')}, {diag.get('tax_identity_ratio_max')}], whole "
        f"tax effect on the daily-excess Sharpe {diag.get('tax_effect_on_daily_excess_sharpe')}, heuristic "
        f"sensitivity (not a rigorous bound) {diag.get('tax_identity_sensitivity_on_sharpe')}; "
        "DISCLOSED SENSITIVITY (report-only, not the scored convention): the engine's next-open fill "
        f"gives daily-excess Sharpe {diag['next_open_sensitivity']['sharpe_daily_excess']} against "
        f"{diag.get('sharpe_daily_excess_taxed')} for the scored close-fill convention; measured: "
        f"{diag['n_windows']} windows ({diag['n_windows_by_leg']}) from {diag['n_monthly_decisions']} monthly "
        f"decisions over {diag['overlay_sessions']} of {diag['account_sessions']} sessions, scored-excess "
        f"OLS beta to SPY {diag['scored_excess_beta_to_spy']}; account span "
        f"{curve['strategy_daily'].index[0]:%Y-%m-%d}..{curve['strategy_daily'].index[-1]:%Y-%m-%d} "
        "(the legs' common span: it starts at the latest-listing leg's first session and every series "
        "must end on the same session); window coverage: "
        f"windows_skipped_before_data={diag['windows_skipped_before_data']}; "
        f"windows_not_started={diag['windows_not_started']}; "
        f"final_window_truncated={diag['final_window_truncated']}; "
        f"window_entered_on_last_bar={diag['window_entered_on_last_bar']}; a DIFFERENT strategy "
        "definition from every prior commodity, duration or trend family (own family_id, own trial); "
        "PRE-REGISTRATION LIMITS: every constant was fixed in advance (from the paper and the family's "
        "pre-registered design) before this "
        "pipeline computed any return of this account and none was tuned; the build step ran only "
        "signal-only probes (open-interest coverage, state shares, switch counts, window counts), so the "
        "build step computed no return, Sharpe or excess of this account or of the always-TLT control "
        "on real history; any dry run or scratch-ledger rehearsal before the scored run ran the identical "
        f"code and data as this row and tuned nothing; price data: DBC {sources['DBC']}, TLT {sources['TLT']}, "
        f"SPY {sources['SPY']}, ^IRX {irx_source}, through {leg_frames['DBC'].index[-1]:%Y-%m-%d} "
        f"(legs) / {spy_frame.index[-1]:%Y-%m-%d} (SPY)")
    unscored_kwargs = {}
    if history_provenance is not None:
        # the note LEADS with the history-end clause (any pre-registered flag still goes ahead of
        # it), so a reader of the permanent row sees at once that a setting, not the data, ended the history
        history_clause = history_cutoff.history_end_clause(history_provenance)
        note_body = f"{history_clause} {note_body}; {cftc_history_end_sentence(history_provenance['history_end'])}"
        # the shared recorder writes the note above on the four scored stages only; an UNSCORED stage still spends the
        # family's one permanent slot, so it gets the same clause through the recorder's separate unscored_note
        # keyword. Passed ONLY when the setting is on: with it off the recorder call passes no unscored_note.
        unscored_kwargs["unscored_note"] = history_clause
    # a survived stage that did not meet the family's own control condition is flagged in the FIRST
    # sentence of this driver-supplied note; the recorded stage itself stays the ABI-1 verdict
    extra_note = note_with_pre_registered_lead(note_body, result["verdict"], control_condition)

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

    layer1, holdout = result["layer1"], result["holdout"]
    report = {
        "family_id": family_id, "shape": mod.SHAPE, "tickers": list(mod.TRADED),
        "verdict": result["verdict"], "stage": stage,
        # the recorded stage is unconditional; the control condition gates THIS field
        "pre_registered_pass": pre_registered_pass(result["verdict"], control_condition),
        "pre_registered_clause": pre_registered_clause(result["verdict"], control_condition),
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
        "ohlc_sources": all_sources,
        "last_bar_dates": {
            **{s: frames[s].index[-1].strftime("%Y-%m-%d") for s in priced},
            FINANCING_SERIES: irx_frame.index[-1].strftime("%Y-%m-%d")},
        "overlay_diagnostics": diag,
    }
    if history_provenance is not None:
        # last_bar_dates above describe the TRIMMED history; the raw vendor last bars sit beside them. The two key names
        # below (history_end = the provenance, history_end_requested = the raw setting) are the same in every driver.
        report["history_end"] = history_provenance
        report["history_end_requested"] = str(history_end)
        report["raw_last_bar_dates"] = {symbol: history_provenance["by_symbol"][symbol]["raw_last"]
                                        for symbol in [*priced, FINANCING_SERIES]}
        if dry_run:
            # None when a real run with these arguments would pass the guard, else the text it would be refused with
            report["history_end_guard_would_refuse"] = history_guard_would_refuse
    return report


def _whole_sessions(text: str) -> int:
    """argparse type for --max-history-drop: a whole number of sessions, zero or more."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of sessions") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is negative; a drop limit is a whole number of sessions >= 0")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", required=True,
                        help="Path to a dated quotes JSON file with bid/ask for DBC, TLT AND SPY "
                             "(a REAL captured book unless --dry-run is also passed, in which case "
                             "a SYNTHETIC placeholder file is expected).")
    parser.add_argument("--run-id", default=None,
                        help="REQUIRED for a real run (it names the OHLC cache dir, and a real run "
                             "refuses any history cached there by an earlier run); a --dry-run falls "
                             "back to a shared default.")
    parser.add_argument("--tested-date", default=None,
                        help="YYYY-MM-DD; REQUIRED for a real run (it is written into a permanent "
                             "ledger row); a --dry-run falls back to today's date.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument("--cftc-cache-dir", default=cftc_data.DEFAULT_CACHE_DIR,
                        help="the verified CFTC open-interest cache (read only)")
    parser.add_argument("--history-end", default=None, metavar="YYYY-MM-DD",
                        help="opt-in history-end setting: end the scored history at this NYSE session and drop "
                             "all later sessions from every series (nothing is filled or patched, and every calendar, "
                             "hole and final-window check runs on the trimmed history); permanent for the scored "
                             "account: it ends at this date (a new family's holdout pin too), so the family cannot be "
                             "re-scored on longer history. A real run is refused when this date is later than "
                             "--tested-date or when --max-history-drop is exceeded; any refusal of this setting writes "
                             "no ledger row, result file or report, and a refusal after the load keeps the per-run OHLC "
                             "cache (a real retry needs a fresh --run-id); a malformed --tested-date is refused up "
                             "front, before anything is fetched. Default: off.")
    parser.add_argument("--max-history-drop", type=_whole_sessions, default=history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS,
                        metavar="N",
                        help="with --history-end (ignored without it): refuse a real run if any series would drop "
                             f"more than N sessions (a whole number, default "
                             f"{history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS}); a dry run reports what a real run "
                             "would do in history_end_guard_would_refuse instead of refusing.")
    args = parser.parse_args(argv)
    if args.tested_date is None:
        if not args.dry_run:
            parser.error("--tested-date is required for a real (non --dry-run) run: it is written "
                         "into a permanent ledger row and must never fall back to a stale default")
        args.tested_date = DEFAULT_TESTED_DATE
    if args.run_id is None:
        if not args.dry_run:
            parser.error("--run-id is required for a real (non --dry-run) run: it names the OHLC "
                         "cache dir, and a real run refuses history cached there by an earlier run "
                         "(a dry run's bars), so it must be a fresh id")
        args.run_id = DEFAULT_RUN_ID

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes, ledger_path=args.ledger,
        run_id=args.run_id, tested_date=args.tested_date, dry_run=args.dry_run,
        signal_source_path=_SIGNAL_SOURCE_PATH, signal_source_sha256=evidence.source_sha256(__file__),
        research_root=args.research_root, cftc_cache_dir=args.cftc_cache_dir, history_end=args.history_end,
        max_history_drop=args.max_history_drop)

    safe_result = batch5._json_safe(result)
    if safe_result.get("reason_code") in _HISTORY_END_REASONS:
        # a refused history end writes no ledger row, no result file and no report (the per-run OHLC cache the load
        # step already wrote is kept, so a real retry needs a fresh --run-id): print the refusal only and exit 2, the
        # status the treasury, gold/copper and generic drivers give the same refusal. Every other refusal (a stale
        # cache, a truncated final window, off-hours quotes) exits 0 below.
        print(json.dumps({"result": safe_result}))
        return 2
    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    if safe_result.get("pre_registered_clause"):
        print(safe_result["pre_registered_clause"])      # the same sentence the note leads with; JSON stays the last line
    payload = {"schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
               "tested_date": args.tested_date, "dry_run": args.dry_run,
               "driver": _SIGNAL_SOURCE_PATH, "result": safe_result}
    if args.history_end is not None:
        # the same two key names as the result inside it: the raw setting and (when the run got that far) the provenance
        payload["history_end_requested"] = args.history_end
        if "history_end" in safe_result:
            payload["history_end"] = safe_result["history_end"]
    output_path = os.path.join(
        run_dir, "commodity-oi-cycle-overlay-dry-run.json" if args.dry_run
        else "commodity-oi-cycle-overlay-results.json")
    batch3._atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
