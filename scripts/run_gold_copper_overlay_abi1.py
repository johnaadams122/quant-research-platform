"""Bespoke ABI-1 driver for gold_copper_ratio_fear_valuation_spy_overlay --
an SPY core plus a margin-financed LONG overlay of extra SPY (0.25 of equity)
held over the merged HIGH windows of a slow gold-to-copper fear-valuation state.
Uses the same quotes / dry-run / refuse-synthetic / parameter-drift / already-scored /
stale-cache / truncated-final checks and the same overlay-only-excess risk basis as
scripts/run_treasury_month_end_overlay_abi1.py.

WHY A BESPOKE DRIVER: the generic scripts/run_abi1_mining.py dispatch builds a
single-position or rotation curve from a per-bar signal function. This family has
no per-bar signal -- its account is a benchmark-plus-overlay curve built by
``mining.batch5.gold_copper_ratio_fear_valuation_spy_overlay`` (math and conventions
in ``mining.long_overlay``) -- and that module's ``make_signal_fn`` always raises
so the generic path cannot mis-score it.

WHAT THE REPORT ADDS (report-only, never gating):
  * ``overlay_diagnostics``: the signal's decision counts, HIGH share, state
    switches and merged episode lengths, the warm-up as-of reads, the overlay's
    measured share of account equity (it drifts, because merged windows hold a
    fixed share count), the financing charged, the tax-identity approximation, the
    OLS beta of the scored excess on SPY, the four ABI-1 risk refusals on the
    overlay-only excess account and a whole-account
    reference;
  * ``controls`` (REPORT-ONLY, no ledger row of their
    own): the always-on arm (the same construction with the state forced HIGH) and
    a seeded set of random-window arms that keep the main arm's episode count and
    exact episode-length multiset, each scored on the same basis (taxed, plus
    untaxed: the always-on arm's single window is long-term and pays no modelled
    tax (long-term at the 0.0 rate), not merely deferred, so the taxed comparison
    with it is tax-asymmetric and the untaxed-basis comparison is printed beside it,
    with the count of windows and of window calendar days held over 365 days for
    every arm), with the PRE-DECLARED interpretation rule applied and unchanged: a
    DSR pass is evidence about the
    signal only if the main arm's daily-excess Sharpe exceeds the always-on arm's
    AND sits at or above the 95th percentile of the random-window draws.

NOT MODELLED (each of these FLATTERS the strategy; written into the ledger note):
a real margin rate is the bill rate PLUS a spread and only the bare ^IRX-based rate
is charged (no margin-rate source exists and inventing one is not allowed); Reg-T
margin calls and borrow availability; windows held over 365 days are untaxed at the
pipeline's 0.0 long-term rate (a documented, flagged assumption, the comment
above ``constants.LTCG_TAX_RATE``); and the tax walk treats financing interest as
fully deductible against the overlay's gain.

FUTURES ARE READ FROM A FROZEN, sha256-PINNED CACHE (data/mining/gold_copper),
never from the network during a score: the vendor's continuous splice is
revised after the fact (its latest bar can differ between two fetches hours
apart). ``--freeze-futures --through YYYY-MM-DD`` writes a new cache (and refuses
to overwrite an existing one unless ``--overwrite-frozen``).

Usage (network: yfinance historical OHLC for SPY and ^IRX only). A --dry-run
proves the wiring against a SYNTHETIC quotes file the caller constructs (its
quotes must NOT be reported as a cost measurement); a real scoring run needs a
real dated quotes capture (``batch5.refuse_synthetic_quotes``) with a bid/ask for
SPY and writes ONE permanent ledger row:
    python scripts/run_gold_copper_overlay_abi1.py \\
        --dry-run --quotes <synthetic-quotes.json> --research-root <scratch dir> \\
        --catalog <a catalog json containing the family's entry> --random-draws 20

    python scripts/run_gold_copper_overlay_abi1.py \\
        --quotes <real-dated-capture.json> --run-id <run-id> --tested-date <YYYY-MM-DD>

A real run needs an explicit --tested-date AND --run-id, uses the pre-declared
control configuration only (``--random-draws`` may differ on a dry run), and
REFUSES (reason_code ``stale_ohlc_cache``: no ledger row is written, main() still
writes its small results JSON and exits 0, and the earlier run's cache is kept) any
history that an earlier run left cached under research-root/run-id. Every refusal
named in this docstring that is not a ``history_end_*`` one behaves the same way.
Because a state-driven window is ordinarily still open at the data's end, a real run REFUSES (reason_code
``truncated_final_window``) unless ``--accept-open-final-window`` is passed, in
which case the window is closed at the last close (the engine's ``end_of_data``
path) and the ledger note says so; a window entered on the very last bar opens
nothing and needs no acceptance. Every calendar or futures hole is REFUSED with a
reason code, never trimmed or patched (``missing_exchange_session``,
``bar_on_closed_day``, ``spy_history_starts_after_first_decision``,
``futures_read_hole``, ``futures_asof_too_stale``, ``futures_cache_missing``,
``futures_cache_hash_mismatch``).

OPT-IN HISTORY END (the opt-in history-end setting; ``--history-end YYYY-MM-DD``, off by default, and everything
above is unchanged without it). The hole refusals above are correct when the vendor lacks one recent
session while holding later ones, but they leave no way to score the history up to the hole. The setting
ends the scored history at a chosen last session instead: AFTER the frames were loaded and cached (so the
per-run OHLC cache keeps the full vendor history) it DROPS every SPY and ^IRX session dated after the
cutoff, in memory only. It never fills, patches, shifts or repairs anything, and every calendar, hole and
final-window check above still runs on the trimmed frames (a history end inside an open window still
needs ``--accept-open-final-window`` for a real run). It fails closed: a ``history_end_*`` refusal writes no
ledger row, no result file and no report (main() prints the refusal and exits with status 2). A refusal that
is not a ``history_end_*`` one (``stale_ohlc_cache``, ``quotes_not_market_hours``, a calendar, futures or
open-window refusal) is unchanged even under the setting: no ledger row, but main() still writes its small
results JSON and exits 0. A malformed or non-session value is refused first, before anything is fetched or
cached (``history_end_invalid``: not a real YYYY-MM-DD date; ``history_end_not_a_session``: not an NYSE
session), and the same door validates the real-run guard's two inputs for real and dry runs alike: an
unusable tested date (not a real YYYY-MM-DD date) or an unusable ``max_history_drop`` (not a whole number of
sessions, zero or more) is a ``history_end_invalid`` refusal there too, never a plain ``ValueError`` and
never a refusal that comes after the load step. The refusals that
come after the load step (``history_end_not_covered``: some series has no bar exactly on
the cutoff, named under ``symbols``; ``history_end_empty``: a series has no rows; and the two real-run
refusals below) keep the per-run OHLC cache the load step already wrote, so a real retry needs a fresh
--run-id (the same run id refuses ``stale_ohlc_cache``). A REAL run also refuses a history end that would
spend the family's one-shot slot by mistake: ``history_end_after_tested_date`` (the cutoff is later than
--tested-date) and ``history_end_drop_too_large`` (some series would lose more than ``--max-history-drop``
sessions, 10 by default; pass a larger N to accept a larger drop). A dry run never refuses on those two:
its result's ``history_end_guard_would_refuse`` holds the message the real run would refuse with, or None.
Every refusal of the setting has one shape: ``reason_code``, ``reason``, ``history_end_requested``,
``symbols`` and ``family_id``; a refusal that comes after the trim (a calendar, futures or open-window
refusal) also carries ``history_end_requested`` and the provenance. The setting is written into the ledger note, the
result (``history_end``: each series' raw and kept last bar and every dropped date, plus
``raw_last_bar_dates``; ``last_bar_dates`` then describe the trimmed history) and the result file. A run
that ends UNSCORED (``not_scoreable`` or ``holdout_not_evaluable``, recorded as an
``insufficient_evidence`` row) spends the same permanent slot, so the shared recorder's ``unscored_note``
keyword is passed the same history-end clause, written verbatim as the note of that row (a scored row's note
already carries it); without the setting the keyword is not passed and that row carries no note.
A scored history end is PERMANENT for the account it scores: the outcome row's account ends at the cutoff,
and a NEW family's first_test pin (its ABI-1 holdout is the tail of that account) ends there too, so the
family cannot be re-scored on longer history; ``record_first_test`` is a no-op for a family that already
has a first_test row, and a not_scoreable row pins none. The frozen futures are not trimmed: the files are
loaded whole and validated (so a malformed row dated after the cutoff could still refuse the run); each
decision reads only the last futures row on or before its own read date (the last NYSE session strictly
before its month-end), and no decision is later than the cutoff, so no futures row dated after the cutoff
can influence a decision. The result's ``futures_provenance`` and the ledger note report each frozen series'
last bar date (``last_bar_dates``) and say they were not trimmed; no gate is added on them, because
``futures_read_hole`` already refuses a cache that ends before a read it needs. ``--history-end`` cannot be
combined with ``--freeze-futures`` (a usage error).

QUOTE PROVENANCE IS A HUMAN-PROCESS SAFEGUARD. ``batch5.refuse_synthetic_quotes``
only refuses a real run when the quotes file carries the word SYNTHETIC; it cannot
tell a measured book from an invented bid/ask file that omits the marker (a
pipeline-wide limit in shared code). Real quotes must come from
a live bid/ask capture, never typed or estimated by hand.
On top of it a REAL run refuses (reason_code ``quotes_not_market_hours``: nothing
fetched, no ledger row, main() still writes its small results JSON and exits 0; dry
runs exempt) unless the capture's ``market_state`` starts
with ``open`` (any case), ``sampled_at_local`` parses (naive means US Eastern), its
Eastern date is the run's ``--tested-date`` and its Eastern time is within
09:45 to 15:45 Eastern inclusive (``quotes_market_hours_refusal``).

THE CONTROLS GATE THE PRE-REGISTERED PASS, NOT THE RECORDED STAGE. The stage
vocabulary is closed in shared code and every tested family is recorded, so a main
ABI-1 ``survived`` is recorded as such; the result's ``pre_registered_pass`` is True
only when the control condition also held, and when it did not the driver-supplied
note (and the printed summary) LEADS with ``CONTROL_CONDITION_FAILED_FLAG``.
"""
from __future__ import annotations

import argparse
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
from mining import (abi1_driver, benchmark_relative, catalog as catalog_mod, constants,
                    evidence, history_cutoff, overlay_risk_report, trial_ledger)
from mining.batch5 import gold_copper_ratio_fear_valuation_spy_overlay as mod

CASH_CONVENTION = "sweep_cash_daily"   # recorded; the account is always fully
                                       # deployed (SPY core), so idle cash is never credited
MARKET_BENCHMARK = "SPY"
FINANCING_SERIES = "^IRX"
# "realization": overlay windows are round trips whose net gains are taxed at exit
# with loss netting (ABI-1's walk); the dollar-attribution approximation is measured
# and reported below.
TAX_POLICY = "realization"

DEFAULT_RUN_ID = "gold-copper-overlay-abi1-v1"   # dry runs only: a real run must pass its own --run-id
DEFAULT_TESTED_DATE = None   # a dry run falls back to today's date; a real run must pass --tested-date
DEFAULT_RANDOM_DRAWS = mod.RANDOM_CONTROL_DRAWS
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
FUTURES_DIR = os.path.join(REPO_ROOT, "data", "mining", "gold_copper")

# The family's OWN statement of how the overlay-only risk-basis rule is ordered relative to
# this family's account numbers (mining.overlay_risk_report.risk_layer_note).
ORDERING_CLAUSE = (
    "the rule is built into this driver, so every number of this family's "
    "account is computed under it and none precedes it")

_MAX_LISTED_DATES = 50     # dates listed in a refusal (the count is always given)

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


# A REAL run prices its one-shot verdict off a quote capture, so it also refuses a capture that was not
# taken during the regular session of the run's own date. After-hours
# cost samples are unusable; the shared synthetic-quote guard only looks for
# the word SYNTHETIC and cannot tell a closed-market book from a live one.
MARKET_TIMEZONE = "America/New_York"
CAPTURE_WINDOW_START = (9, 45)     # Eastern, inclusive
CAPTURE_WINDOW_END = (15, 45)      # Eastern, inclusive


_ISO_DATE_PREFIX = re.compile(r"\s*\d{4}-\d{2}-\d{2}")


def _eastern_stamp(value):
    """A US Eastern ``pd.Timestamp`` for a string that carries an ISO date and parses (a naive time is read
    as Eastern, an offset-aware one is converted to it), else None. Never raises. A date-less string is
    refused because pandas reads "10:00" (or "now") as the CURRENT date, which would let a time-only capture
    pass the date check on the run date; a wall-clock time that occurs twice (DST fall back) or never
    (spring forward) is a refusal too, not an exception."""
    if not isinstance(value, str) or not _ISO_DATE_PREFIX.match(value):
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        return stamp.tz_localize(MARKET_TIMEZONE) if stamp.tzinfo is None else stamp.tz_convert(MARKET_TIMEZONE)
    except Exception:                    # a parse failure, an ambiguous or nonexistent local time, an overflow
        return None


def quotes_market_hours_refusal(quotes_payload, tested_date):
    """``None`` when the quote capture is acceptable for a REAL run, else the text of why not. Accepted only
    when (i) ``market_state`` (case-insensitive, surrounding whitespace ignored) starts with ``open``, (ii)
    ``sampled_at_local`` is a string carrying an ISO date that parses as a time (a naive time is read as US
    Eastern, an offset-aware one is converted to it), (iii) its Eastern calendar date equals ``tested_date``
    and (iv) its Eastern time of day is between 09:45:00 and 15:45:00 inclusive. Never raises on a bad
    payload or run date: a missing, non-string, date-less, ambiguous or unparseable field is a refusal."""
    problems = []
    fields = quotes_payload if isinstance(quotes_payload, dict) else {}
    state = fields.get("market_state")
    if not isinstance(state, str) or not state.strip().lower().startswith("open"):
        problems.append(f"market_state {state!r} does not start with 'open'")
    sampled = fields.get("sampled_at_local")
    stamp = _eastern_stamp(sampled)
    tested = _eastern_stamp(tested_date)
    if stamp is None:
        problems.append(f"sampled_at_local {sampled!r} is missing, carries no ISO date or does not parse as a time")
    if tested is None:
        problems.append(f"the run's tested date {tested_date!r} is not a YYYY-MM-DD date")
    if stamp is not None:
        if tested is not None and stamp.date() != tested.date():
            problems.append(f"sampled_at_local date {stamp.date():%Y-%m-%d} (Eastern) is not the run's tested "
                            f"date {tested.date():%Y-%m-%d}")
        clock = (stamp.hour, stamp.minute, stamp.second, stamp.microsecond)
        if not ((*CAPTURE_WINDOW_START, 0, 0) <= clock <= (*CAPTURE_WINDOW_END, 0, 0)):
            problems.append(f"sampled_at_local time {stamp:%H:%M:%S} Eastern is outside the "
                            f"{CAPTURE_WINDOW_START[0]:02d}:{CAPTURE_WINDOW_START[1]:02d} to "
                            f"{CAPTURE_WINDOW_END[0]:02d}:{CAPTURE_WINDOW_END[1]:02d} Eastern capture window")
    return "; ".join(problems) if problems else None


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _r4(value):
    value = _finite(value)
    return None if value is None else round(value, 4)


def _sharpe(result: dict):
    return _finite(result["layer1"].get("sharpe_daily_excess"))


def long_term_window_stats(trades_df) -> dict:
    """The count of windows, and of window calendar days, held STRICTLY MORE than
    ``constants.LONG_TERM_HOLDING_DAYS`` calendar days (the tax walk's own long-term rule,
    ``(exit_date - entry_date).days > 365``), computed from a run's actual trades. A window
    that is long-term is taxed at ``constants.LTCG_TAX_RATE``, which is 0.0 (a
    documented, flagged assumption), so it realises no modelled tax at all."""
    threshold = int(constants.LONG_TERM_HOLDING_DAYS)
    if len(trades_df):
        held = (pd.DatetimeIndex(trades_df["exit_date"]) - pd.DatetimeIndex(trades_df["entry_date"])).days.to_numpy(dtype=int)
    else:
        held = np.array([], dtype=int)
    long_term = held > threshold
    total, long_days = int(held.sum()), int(held[long_term].sum())
    return {"threshold_days": threshold, "n_windows": int(len(held)), "n_long_term": int(long_term.sum()),
            "window_days": total, "long_term_window_days": long_days,
            "long_term_day_share": (long_days / total) if total else None}


def holdout_window_high(decisions, states, account_index, holdout_sessions) -> dict:
    """The share of decision month-ends inside the ABI-1 holdout window (the last ``holdout_sessions``
    sessions of the scored account) whose state is HIGH, computed from THIS run's own states; ``None``
    when the run has no evaluated holdout."""
    if not holdout_sessions or int(holdout_sessions) > len(account_index):
        return None
    start = pd.DatetimeIndex(account_index)[-int(holdout_sessions)]
    inside = [i for i, d in enumerate(decisions) if d >= start]
    high = sum(bool(states[i]) for i in inside)
    return {"start": start.strftime("%Y-%m-%d"), "n_month_ends": int(len(inside)), "n_high": int(high),
            "high_share": (high / len(inside)) if inside else None}


# ------------------------------------------------------ interpretation rule

def interpret_controls(*, main_sharpe, always_on_sharpe, main_percentile) -> dict:
    """The PRE-DECLARED, report-only interpretation rule. ``signal_evidence`` is
    True only when the main arm's taxed daily-excess Sharpe strictly exceeds the
    always-on arm's AND its percentile among the random-window draws is at or above
    ``mod.RANDOM_CONTROL_PERCENTILE_BAR``; None when any input is missing."""
    rule = ("pre-declared, report-only: a DSR pass is evidence about the signal only if the main "
            "arm's taxed daily-excess Sharpe exceeds the always-on arm's AND sits at or above the "
            "95th percentile of the random-window draws; otherwise the pass is attributed to equity "
            "exposure, not to the signal")
    if main_sharpe is None or always_on_sharpe is None or main_percentile is None:
        evidence_flag = None
    else:
        evidence_flag = bool(main_sharpe > always_on_sharpe
                             and main_percentile >= mod.RANDOM_CONTROL_PERCENTILE_BAR)
    return {"signal_evidence": evidence_flag, "percentile_bar": mod.RANDOM_CONTROL_PERCENTILE_BAR,
            "rule": rule}


# The stage vocabulary is CLOSED in shared code (evidence.VALID_STAGES) and every tested family is recorded
# unconditionally, so the controls cannot change the recorded stage. They gate the PRE-REGISTERED PASS
# instead: a main ABI-1 'survived' that lost to the always-on control or fell below the random-window
# control's 95th percentile is recorded as 'survived' but is NOT a survivor under this family's own rule,
# and the note and the printed summary say so in their first sentence.
CONTROL_CONDITION_FAILED_FLAG = (
    "CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under this family's own "
    "pre-registered rule (beat the always-on control AND the 95th percentile of the random-window "
    "control). This is not a pass; do not promote it without review.")
PRE_REGISTERED_RULE_MET_CLAUSE = (
    "PRE-REGISTERED RULE MET: the ABI-1 stage survived AND the control condition holds (beat the "
    "always-on control AND the 95th percentile of the random-window control); pre_registered_pass=True.")


def pre_registered_pass(verdict, signal_evidence) -> bool:
    """True only when the ABI-1 verdict is ``survived`` AND ``signal_evidence`` is exactly True (a missing
    or False control condition is not a pass)."""
    return bool(verdict == "survived" and signal_evidence is True)


def pre_registered_clause(verdict, signal_evidence) -> str:
    """The sentence a survived stage carries: the unmissable flag when the control condition did not hold
    (False or missing), the rule-met statement when it did, and nothing for any other verdict."""
    if verdict != "survived":
        return ""
    return PRE_REGISTERED_RULE_MET_CLAUSE if signal_evidence is True else CONTROL_CONDITION_FAILED_FLAG


# ------------------------------------------------------------- checks

def overlay_share_of_equity(curve: dict, spy_frame) -> dict:
    """Min, max and mean of the overlay's share of account equity over the
    sessions after each window's entry through its exit: merged windows hold a
    FIXED share count, so this drifts away from the 0.25 sized at entry. Uses the
    builder's own pre-tax equity path."""
    trades = curve["trades_df"]
    if not len(trades):
        return {"min": None, "max": None, "mean": None}
    idx = curve["strategy_daily"].index
    equity = np.cumprod(1.0 + curve["strategy_daily"].to_numpy(dtype=float))
    close = spy_frame["Close"].reindex(idx).to_numpy(dtype=float)
    shares = []
    for _, trade in trades.iterrows():
        a, b = idx.get_loc(trade["entry_date"]), idx.get_loc(trade["exit_date"])
        overlay_shares = mod.OVERLAY_NOTIONAL * equity[a] / close[a]
        shares.append(overlay_shares * close[a + 1:b + 1] / equity[a + 1:b + 1])
    values = np.concatenate(shares) if shares else np.array([])
    if not len(values):
        return {"min": None, "max": None, "mean": None}
    return {"min": _finite(values.min()), "max": _finite(values.max()), "mean": _finite(values.mean())}


def overlay_diagnostics(curve: dict, result: dict, *, untaxed_result: dict, benchmark_daily, spy_frame,
                        spy_cost_bps: float, benchmark_entry_cost_bps: float) -> dict:
    """Report-only structural checks for the built curve (see module
    docstring). Nothing here gates a verdict."""
    trades = curve["trades_df"]
    info = curve["window_info"]
    index = curve["strategy_daily"].index
    mask = curve["overlay_mask"].to_numpy(dtype=bool)
    drift = overlay_share_of_equity(curve, spy_frame)
    episodes = list(info["episode_months"])
    diag = {
        "overlay_notional_fraction": float(mod.PARAMETERS["overlay_notional_fraction"]),
        "spy_leg_cost_bps": float(spy_cost_bps),
        "benchmark_entry_cost_bps": float(benchmark_entry_cost_bps),
        "n_windows": int(len(trades)),
        "n_decisions": int(info["n_decisions"]), "n_high": int(info["n_high"]),
        "high_share": _finite(info["high_share"]), "n_state_switches": int(info["n_state_switches"]),
        "episode_months": episodes, "longest_episode_months": int(max(episodes)) if episodes else 0,
        "first_decision": info["first_decision"], "last_decision": info["last_decision"],
        "terminal_exit": info["terminal_exit"],
        "windows": [[a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")] for a, b in info["windows"]],
        "asof_reads": list(info["asof_reads"]),
        "final_window_truncated": bool(info["truncated_final"]),
        "window_entered_on_last_bar": bool(info["entered_on_last_bar"]),
        "long_term_windows": long_term_window_stats(trades),
        "overlay_sessions": int(mask.sum()), "account_sessions": int(len(index)),
        "overlay_session_share": _finite(mask.mean()) if len(index) else None,
        "overlay_share_of_equity_min": drift["min"], "overlay_share_of_equity_max": drift["max"],
        "overlay_share_of_equity_mean": drift["mean"],
        "mean_financing_rate_annual_on_overlay_sessions": (
            _finite(curve["financing_rate"].to_numpy(dtype=float)[mask].mean()) if mask.any() else None),
    }
    if len(trades):
        net = trades["net_ret"].to_numpy(dtype=float)
        diag.update({
            "first_window_entry": trades["entry_date"].iloc[0].strftime("%Y-%m-%d"),
            "last_window_exit": trades["exit_date"].iloc[-1].strftime("%Y-%m-%d"),
            "worst_window_net_ret": _finite(net.min()), "best_window_net_ret": _finite(net.max()),
            "sum_window_financing_ret": _finite(trades["financing_ret"].sum()),
            "sum_window_cost_ret": _finite(trades["cost_ret"].sum()),
        })
        # Dollar-attribution ratio netted_tax_factors implicitly assumes is 1: the
        # account's growth from the sizing point (the entry close) to the exit close
        # over (1 + the overlay's own net return). With merged windows of many
        # months the SPY core's own growth inside the window moves it far from 1.
        growth = np.cumprod(1.0 + curve["strategy_daily"].to_numpy(dtype=float))
        entry_pos = index.get_indexer(pd.DatetimeIndex(trades["entry_date"]))
        exit_pos = index.get_indexer(pd.DatetimeIndex(trades["exit_date"]))
        ratio = (growth[exit_pos] / growth[entry_pos]) / (1.0 + net)
        diag["tax_identity_ratio_min"] = _finite(ratio.min())
        diag["tax_identity_ratio_max"] = _finite(ratio.max())
        taxed, untaxed = _sharpe(result), _sharpe(untaxed_result)
        if taxed is not None and untaxed is not None:
            effect = taxed - untaxed
            worst = max(abs(diag["tax_identity_ratio_min"] - 1.0), abs(diag["tax_identity_ratio_max"] - 1.0))
            diag.update({
                "sharpe_daily_excess_taxed": _finite(taxed), "sharpe_daily_excess_untaxed": _finite(untaxed),
                "tax_effect_on_daily_excess_sharpe": _finite(effect),
                "tax_identity_sensitivity_on_sharpe": _finite(worst * abs(effect)),
            })
    beta = benchmark_relative.ols_alpha_beta(result["log_excess"], benchmark_daily)
    diag["scored_excess_beta_to_spy"] = _finite(beta["beta"])
    # The four refusals gate on the OVERLAY-ONLY EXCESS account
    # (`layer2["account_basis"]`); the whole-account numbers stay beside them.
    diag["risk_layer"] = overlay_risk_report.risk_layer_report(result, benchmark_daily)
    return diag


# ----------------------------------------------------------------- controls

def _score_arm(curve: dict, eval_kwargs: dict) -> tuple:
    taxed = abi1_driver.evaluate_family(curve, tax_policy=TAX_POLICY, **eval_kwargs)
    untaxed = abi1_driver.evaluate_family(curve, tax_policy="none", **eval_kwargs)
    return taxed, untaxed


def _percentile(main, draws) -> tuple:
    """Mid-rank percentile of ``main`` among the scored ``draws`` and the number of
    draws at or above it."""
    scored = [d for d in draws if d is not None]
    if main is None or not scored:
        return None, None
    below = sum(d < main for d in scored) + 0.5 * sum(d == main for d in scored)
    return below / len(scored), int(sum(d >= main for d in scored))


def control_arms(ctx: dict, spy_frame, eval_kwargs: dict, *, spy_cost_bps: float, benchmark_entry_cost_bps: float,
                 n_draws: int, main_sharpe, main_sharpe_untaxed) -> dict:
    """Score the two report-only control arms on the SAME basis as the main arm.
    Nothing is written anywhere; a failure here propagates (fail closed) so a
    permanent row is never recorded without the controls its note refers to."""
    costs = dict(spy_cost_bps=spy_cost_bps, spy_entry_cost_bps=benchmark_entry_cost_bps)
    always_curve = mod.build_always_on_curve(ctx, spy_frame, **costs)
    a_taxed, a_untaxed = _score_arm(always_curve, eval_kwargs)
    always_on = {
        "n_trades": int(len(always_curve["trades_df"])),
        "sharpe_daily_excess": _sharpe(a_taxed), "sharpe_daily_excess_untaxed": _sharpe(a_untaxed),
        "dsr_value": _finite(a_taxed["layer1"].get("dsr_value")), "verdict": a_taxed["verdict"],
        "holdout_passed": a_taxed["holdout"].get("passed"),
        "failed_refusals": overlay_risk_report.failed_refusals(a_taxed["layer2"])
        if a_taxed["layer2"].get("drawdown") else [],
        "overlay_session_share": _finite(always_curve["overlay_mask"].mean()),
        "long_term_windows": long_term_window_stats(always_curve["trades_df"]),
        "note": "the same construction with the state forced HIGH: one window from the first decision "
                "month-end to the end of the data; that window is long-term, so it pays no modelled tax "
                "(long-term at the 0.0 rate), not merely deferred",
    }
    draws = mod.random_window_state_draws(ctx["states"], n_draws, seed=mod.RANDOM_CONTROL_SEED)
    taxed_sharpes, untaxed_sharpes, draw_long_term = [], [], []
    for draw in draws:
        curve = mod.curve_from_states(ctx, spy_frame, draw, **costs)
        d_taxed, d_untaxed = _score_arm(curve, eval_kwargs)
        taxed_sharpes.append(_sharpe(d_taxed))
        untaxed_sharpes.append(_sharpe(d_untaxed))
        draw_long_term.append(long_term_window_stats(curve["trades_df"]))
    percentile, at_or_above = _percentile(main_sharpe, taxed_sharpes)
    percentile_untaxed, _ = _percentile(main_sharpe_untaxed, untaxed_sharpes)
    scored = [d for d in taxed_sharpes if d is not None]
    random_window = {
        # the seed is written as the underscore string the catalog parameters carry, so the ledger note
        # and the results file show it exactly as the catalog does
        "n_draws": int(n_draws), "n_scored": len(scored), "seed": mod.PARAMETERS["random_window_control_seed"],
        "episode_months": mod.episode_lengths(ctx["states"]),
        "high_share": _finite(np.mean(ctx["states"])),
        "draw_sharpes": taxed_sharpes, "draw_sharpes_untaxed": untaxed_sharpes,
        "main_percentile": percentile, "main_percentile_untaxed": percentile_untaxed,
        "draws_at_or_above_main": at_or_above,
        "long_term_windows": {key: [stats[key] for stats in draw_long_term]
                              for key in ("n_windows", "n_long_term", "window_days", "long_term_window_days")},
        "quantiles": ({"p05": _finite(np.quantile(scored, 0.05)), "p50": _finite(np.quantile(scored, 0.5)),
                       "p95": _finite(np.quantile(scored, 0.95))} if scored else None),
    }
    always_untaxed = always_on["sharpe_daily_excess_untaxed"]
    untaxed_basis = {
        "main_sharpe_untaxed": main_sharpe_untaxed, "always_on_sharpe_untaxed": always_untaxed,
        "main_exceeds_always_on_untaxed": (None if main_sharpe_untaxed is None or always_untaxed is None
                                           else bool(main_sharpe_untaxed > always_untaxed)),
        "note": ("report-only: the pre-declared interpretation rule compares the main arm's TAXED Sharpe with the "
                 "always-on arm's TAXED Sharpe, which is tax-asymmetric (windows held over 365 days are untaxed at "
                 "the pipeline's 0.0 long-term rate and the two arms hold different shares of them); this is the "
                 "same comparison with both arms untaxed. The rule itself is unchanged."),
    }
    return {
        "always_on": always_on, "random_window": random_window,
        "interpretation": interpret_controls(main_sharpe=main_sharpe, always_on_sharpe=always_on["sharpe_daily_excess"],
                                             main_percentile=percentile),
        "untaxed_basis_comparison": untaxed_basis,
        "ledger_rows_written_for_controls": 0,
    }


# --------------------------------------------------------------- ledger note

def _history_end_last_bar_sentence(history_end_provenance) -> str:
    """What the last-bar sentence adds under an opt-in history end: the last bar used is the setting's end,
    and which later vendor bars the setting dropped (or that it dropped none)."""
    spy = history_end_provenance["by_symbol"][MARKET_BENCHMARK]
    end = history_end_provenance["history_end"]
    if spy["dropped_sessions"]:
        return (f"; history end: this last bar is the opt-in history end ({end}), not the vendor's final bar: the "
                f"raw last {MARKET_BENCHMARK} bar is {spy['raw_last']} and {spy['dropped_sessions']} later session(s) "
                f"were dropped by the setting ({history_end_provenance['decision']})")
    return (f"; history end: the opt-in history end ({end}) is the vendor's own last {MARKET_BENCHMARK} bar, so the "
            f"setting dropped no {MARKET_BENCHMARK} bar")


def _history_end_futures_sentence(futures: dict) -> str:
    """The note's statement about the frozen futures under an opt-in history end: they were not trimmed (with
    each cached series' last bar date when known); the files are loaded whole and validated (so a malformed
    later row could still refuse the run); each decision reads only the last futures row on or before its own
    read date and no decision is later than the history end, so no futures row dated after the cutoff can
    influence a decision. It deliberately does not say that no later row is read: the whole files are loaded."""
    last_bars = futures.get("last_bar_dates")
    listed = (" (last bars " + ", ".join(f"{series} {day}" for series, day in sorted(last_bars.items())) + ")"
              if last_bars else "")
    return (f"; history end and the futures: the frozen futures files were not trimmed{listed}; they are loaded "
            "whole and validated, each decision reads only the last futures row on or before its own read date "
            "(the last NYSE session strictly before its month-end), and no decision is later than the history end "
            "(the cutoff), so no futures row dated after the cutoff can influence a decision")


def last_bar_partial_clause(last_bar, tested_date, history_end_provenance=None) -> tuple:
    """``(may_be_partial, text)``: compares the date of the last SPY bar the run
    used with the run's tested date and STATES the result for the permanent note.
    The vendor returns the in-progress session as an ordinary row, so a last bar
    dated on or after the run date cannot be shown complete from dates alone (a run
    after the close may still be fine, which this conservative rule cannot tell); a
    bar that predates the run date is a completed session. The last bar sets the exit
    price of a window still open at the data's end, that window's realization tax and
    the last account return. Under an opt-in history end (``history_end_provenance``)
    the last bar is the setting's end, so the text adds that later vendor bars were
    dropped by the setting and are not simply absent."""
    last_bar = pd.Timestamp(last_bar).normalize()
    tested = pd.Timestamp(tested_date).normalize()
    partial = bool(last_bar >= tested)
    head = f"last-bar status: the last SPY bar used is dated {last_bar:%Y-%m-%d} and the run tested date is {tested:%Y-%m-%d}; "
    setting = "" if history_end_provenance is None else _history_end_last_bar_sentence(history_end_provenance)
    if partial:
        return True, (head + "a bar dated on or after the run date may be the vendor's live partial bar (the vendor "
                      "returns the in-progress session as a row), so the exit price of a window still open at the "
                      "data's end, its realization tax and the last account return may rest on a partial bar: "
                      "last_spy_bar_may_be_live_partial=True" + setting)
    return False, (head + "the last bar predates the run date, so it is a completed session: "
                          "last_spy_bar_may_be_live_partial=False" + setting)


def always_on_tax_clause(stats: dict) -> str:
    """The data-driven statement of what tax the always-on arm pays, from its own trades: a window held more
    than 365 days is long-term at the pipeline's 0.0 rate and pays no modelled tax at all, not merely a
    deferred one."""
    n, k = stats["n_windows"], stats["n_long_term"]
    if n and k == n:
        return (f"each of its {n} window(s) is held more than {stats['threshold_days']} calendar days, so the "
                "always-on arm pays no modelled tax (long-term at the 0.0 rate), not merely deferred")
    return (f"{k} of its {n} window(s) are held more than {stats['threshold_days']} calendar days and each of "
            "those pays no modelled tax (long-term at the 0.0 rate), not merely deferred")


def _mean_min_max(values, fmt) -> str:
    return f"{fmt(float(np.mean(values)))} [{min(values)}, {max(values)}]"


def long_term_windows_sentence(diag: dict, controls: dict) -> str:
    """The count of windows and of window calendar days held more than 365 days for the main arm, the
    always-on arm and the random-window draws, counted from THIS run's own trades."""
    main = diag["long_term_windows"]
    always = controls["always_on"]["long_term_windows"]
    rnd = controls["random_window"]["long_term_windows"]
    n_draws = len(rnd["n_long_term"])
    two_places = lambda v: f"{v:.2f}"      # noqa: E731
    return (
        f"WINDOWS HELD OVER {main['threshold_days']} DAYS (untaxed at the pipeline's 0.0 long-term rate; a window is "
        f"long-term when held more than {main['threshold_days']} calendar days, counted from this run's own trades): "
        f"main arm {main['n_long_term']} of {main['n_windows']} windows and {main['long_term_window_days']} of "
        f"{main['window_days']} window calendar days; always-on arm {always['n_long_term']} of {always['n_windows']} "
        f"windows and {always['long_term_window_days']} of {always['window_days']} window calendar days; "
        f"random-window arms (mean [min, max] over {n_draws} draws) "
        f"{_mean_min_max(rnd['n_long_term'], two_places)} windows over {main['threshold_days']} days of "
        f"{float(np.mean(rnd['n_windows'])):g} and {_mean_min_max(rnd['long_term_window_days'], two_places)} window "
        f"calendar days over {main['threshold_days']} of a mean {float(np.mean(rnd['window_days'])):.2f}; ")


def holdout_sentence(holdout) -> str:
    """The holdout window's HIGH share, computed from this run's own states (never a static adjective)."""
    if not holdout:
        return "no ABI-1 holdout window was evaluated for this run"
    share = _r4(holdout["high_share"])
    return (f"measured on this run's own states, the holdout window (from {holdout['start']}) holds "
            f"{holdout['n_high']} HIGH of {holdout['n_month_ends']} month-ends (share {share}); a holdout "
            "window dominated by one state cannot separate signal from bull-market exposure")


def build_ledger_note(*, diag: dict, controls: dict, risk_note: str, quotes_source: str, spy_cost_bps: float,
                      sources: dict, futures: dict, curve: dict, spy_frame, main_sharpe, accepted_open: bool,
                      tested_date: str, provenance: dict, abi1_verdict: str, module_sha256: str, driver_sha256: str,
                      history_end_provenance: dict | None = None) -> str:
    """The permanent ledger note: WHAT the score was computed against, every
    amendment and disclosure, the pre-declared control interpretation with this
    run's measured numbers, and the coverage flags. Under an opt-in history end
    (``history_end_provenance``) the note also opens with the shared history-end
    clause (right after any control-failed flag) and says the futures were not
    trimmed (with each cached futures series' last bar date when the caller gives them)."""
    always, rnd = controls["always_on"], controls["random_window"]
    verdict = controls["interpretation"]
    untaxed = controls["untaxed_basis_comparison"]
    span = f"{curve['strategy_daily'].index[0]:%Y-%m-%d}..{curve['strategy_daily'].index[-1]:%Y-%m-%d}"
    quantiles = rnd["quantiles"] or {}
    # A survived stage that did not meet the family's own control condition is flagged
    # in the FIRST sentence of this driver-supplied note (the shared writer puts its own fixed
    # metadata prefix ahead of it, which is not editable from here).
    clause = pre_registered_clause(abi1_verdict, verdict["signal_evidence"])
    lead = f"{clause} " if clause else ""
    history_lead = (f"{history_cutoff.history_end_clause(history_end_provenance)} "
                    if history_end_provenance is not None else "")
    futures_tail = _history_end_futures_sentence(futures) if history_end_provenance is not None else ""
    return lead + history_lead + (
        f"cost model: spy={spy_cost_bps:.2f}bps full round trip per window ({quotes_source}); "
        "design=benchmark-plus-overlay account (SPY core fully invested at all times PLUS a margin-financed "
        f"LONG-ONLY overlay of extra SPY, total notional {diag['overlay_notional_fraction']} of equity, "
        "no hedge leg and no short leg); signal: x = ln(GC=F close / HG=F close) read at the last ACTUAL "
        "NYSE session strictly before each scheduled month-end (the one-session lag: no futures bar on or "
        "after the month-end is used, because the Yahoo =F daily Close cannot be verified as the COMEX "
        "settlement over the whole history; it is consistent with the COMEX settlement time (gold tight, copper "
        "noisier) only over a short recent stretch), "
        "z against the last 60 monthly reads including itself (sample stdev), state HIGH when z >= 0; "
        "month-ends are the exchange's regular-holiday schedule fixed in advance (exchange_calendars XNYS, "
        "unscheduled closures NOT removed), never read off the data; consecutive HIGH month-ends are merged "
        "into one window (entry at the close of the first HIGH month-end, exit at the close of the first "
        "later month-end that is not HIGH), so the overlay holds a fixed share count, set at the entry "
        f"equity, for the whole window and its share of equity drifted between "
        f"{_r4(diag['overlay_share_of_equity_min'])} and {_r4(diag['overlay_share_of_equity_max'])} "
        f"(mean {_r4(diag['overlay_share_of_equity_mean'])}); "
        f"decisions={diag['n_decisions']} (high {diag['n_high']}, state_switches={diag['n_state_switches']}); "
        f"episode_months={diag['episode_months']}; longest_episode_months={diag['longest_episode_months']}; "
        "hole rule: a missing futures read is a causal as-of read (at most 4 calendar days stale) only "
        f"during the warm-up before the first defined z ({len(diag['asof_reads'])} such reads here), and any "
        "missing read from the first defined z on is refused, never patched; "
        "financing = the whole overlay notional borrowed at the ^IRX 13-week bill coupon-equivalent yield "
        "of the last close strictly before each session, on actual calendar days over 365, mean "
        f"{diag['mean_financing_rate_annual_on_overlay_sessions']} on overlay sessions; "
        "NOT MODELLED, and each of these flatters the strategy: a real margin rate is the bill rate PLUS a "
        "spread and no spread is charged (the pipeline has no margin-rate source and inventing one is not "
        "allowed), Reg-T margin calls and borrow availability; windows held over 365 days are untaxed at the "
        "pipeline's 0.0 long-term rate (a documented, flagged assumption); the tax walk (ABI-1 realization netting) "
        "deducts financing interest against the overlay's gain; "
        f"{risk_note}; tax: netted_tax_factors dollar-attribution approximation (not uniformly "
        "conservative; with merged multi-month windows the SPY core's growth inside a window moves the "
        f"identity ratio far from 1), measured ratio range [{diag.get('tax_identity_ratio_min')}, "
        f"{diag.get('tax_identity_ratio_max')}], whole tax effect on the daily-excess Sharpe "
        f"{diag.get('tax_effect_on_daily_excess_sharpe')}, heuristic sensitivity (not a rigorous bound) "
        f"{diag.get('tax_identity_sensitivity_on_sharpe')}; "
        "CONTROL ARMS (report-only, no ledger row of their own, scored on the same basis: financing, costs, "
        f"tax policy): main_sharpe={_r4(main_sharpe)}; always_on_sharpe={_r4(always['sharpe_daily_excess'])} "
        f"(untaxed {_r4(always['sharpe_daily_excess_untaxed'])}, verdict {always['verdict']}; the always-on "
        "arm is the same construction with the state forced HIGH, one window to the end of the data, and "
        f"{always_on_tax_clause(controls['always_on']['long_term_windows'])}); random-window arm: "
        f"n_draws={rnd['n_draws']}, seed={rnd['seed']}, same episode count and episode-length multiset as the "
        f"main arm, main_percentile={_r4(rnd['main_percentile'])} (untaxed {_r4(rnd['main_percentile_untaxed'])}), "
        f"draws_at_or_above_main={rnd['draws_at_or_above_main']}, draw quantiles p05={_r4(quantiles.get('p05'))} "
        f"p50={_r4(quantiles.get('p50'))} p95={_r4(quantiles.get('p95'))}; "
        f"INTERPRETATION RULE (PRE-DECLARED): {verdict['rule']}; signal_evidence={verdict['signal_evidence']} "
        "(True only means both control conditions hold; it is evidence about the signal only if the verdict "
        "is also a pass); "
        f"pre_registered_pass={pre_registered_pass(abi1_verdict, verdict['signal_evidence'])} "
        f"(True only when the ABI-1 verdict is survived AND signal_evidence is True; this run's ABI-1 verdict "
        f"is {abi1_verdict}); "
        "TAX ASYMMETRY (report-only): the rule compares the main arm's TAXED Sharpe with the always-on arm's "
        "TAXED Sharpe, which is tax-asymmetric because windows held over 365 days are untaxed at the pipeline's "
        "0.0 long-term rate and the two arms hold different shares of them; the UNTAXED-BASIS COMPARISON beside "
        f"it: main_sharpe_untaxed={_r4(untaxed['main_sharpe_untaxed'])}; "
        f"always_on_sharpe_untaxed={_r4(untaxed['always_on_sharpe_untaxed'])}; "
        f"main_exceeds_always_on_untaxed={untaxed['main_exceeds_always_on_untaxed']}; the rule itself is "
        "unchanged; "
        f"{long_term_windows_sentence(diag, controls)}"
        f"the ABI-1 holdout is descriptive only here ({holdout_sentence(diag.get('holdout_window_high'))}) "
        "and the daily-iid DSR counts "
        "thousands of sessions as independent bets where the real evidence is a few independent episodes "
        "(see episode_months); "
        f"measured: {diag['n_windows']} windows over {diag['overlay_sessions']} of {diag['account_sessions']} "
        f"sessions, scored-excess OLS beta to SPY {diag['scored_excess_beta_to_spy']}; account span {span} "
        "(SPY's own sessions from the first month-end at which z is defined; the pipeline's convention, "
        "favourable to the DSR because it keeps the zero-excess prefix out of the daily-iid count); window coverage: "
        f"final_window_truncated={diag['final_window_truncated']} (a truncated window is closed at the last "
        "close, the engine's end_of_data path); "
        f"window_entered_on_last_bar={diag['window_entered_on_last_bar']}; "
        f"accepted_open_final_window={accepted_open}; "
        "a DIFFERENT strategy definition from every other commodity, fear or overlay family (own family_id, "
        "own trial); PRE-REGISTRATION LIMITS: no constant is tuned against any ABI-1 score and all are fixed "
        "before any return of this account is computed, but they are NOT all chosen blind: "
        "signal-only descriptive statistics on these futures "
        "(state counts, HIGH share, alternative-window agreement; no returns) sit outside the trial ledger, so "
        "DSR's trial count does not charge them; the 60-month window, the z >= 0 threshold and the one-session "
        "lag are this pipeline's own choices, not paper parameters; "
        "scored as a single pre-declared shot; "
        f"price data: SPY {sources['SPY']}, ^IRX {sources[FINANCING_SERIES]}, through "
        f"{spy_frame.index[-1]:%Y-%m-%d}; "
        f"{last_bar_partial_clause(spy_frame.index[-1], tested_date, history_end_provenance)[1]}; "
        f"signal_module_sha256={module_sha256}; driver_sha256={driver_sha256}; "
        f"calendar provenance: exchange_calendars_version={provenance['exchange_calendars_version']}, "
        f"schedule_decisions={provenance['n_decisions']} ({provenance['first_decision']} to "
        f"{provenance['last_decision']}), schedule_terminal_exit={provenance['terminal_exit']}, "
        f"schedule_sha256={provenance['schedule_sha256']} ({provenance['hash_format']}; every month-end and "
        "read date is a function of that package version); "
        f"futures: frozen sha256-verified splice of the vendor's continuous "
        f"front-month series (GC=F and HG=F, unadjusted across contract rolls) through {futures['through']}, "
        f"futures rows={futures['files']['yahoo_GC_F.csv']['rows']}/{futures['files']['yahoo_HG_F.csv']['rows']}"
        + futures_tail)


# ----------------------------------------------------------------- the score

def _history_end_refusal(family_id, exc, history_end) -> dict:
    """The one standard refusal payload for an opt-in history end this driver refuses
    (``history_cutoff.refusal_payload``: ``reason_code``, ``reason``, ``history_end_requested``, ``symbols``,
    then ``family_id``; the reason code is the exception's own and ``symbols`` names the series at fault, empty
    when the refusal is not about a series)."""
    return history_cutoff.refusal_payload(exc, history_end_requested=history_end, family_id=family_id)


def _history_end_fields(history_end, history_end_provenance) -> dict:
    """What a refusal that comes AFTER the trim still says about the setting: nothing when it is off, else the
    raw ``history_end_requested`` and, once the trim ran, the ``history_end`` provenance."""
    if history_end is None:
        return {}
    fields = {"history_end_requested": str(history_end)}
    if history_end_provenance is not None:
        fields["history_end"] = history_end_provenance
    return fields


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str, dry_run: bool,
                       signal_source_path: str, signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT, ohlc_loader=None,
                       futures_dir: str = FUTURES_DIR, n_random_draws: int = DEFAULT_RANDOM_DRAWS,
                       accept_open_final_window: bool = False, history_end=None,
                       max_history_drop=history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS) -> dict:
    family_id = mod.FAMILY_ID
    if history_end is not None:
        # Under an opt-in history end the real-run guard's own two inputs, the tested
        # date and the drop limit, are checked HERE, at the very door and for real and dry runs alike (a dry run
        # rehearses the real run), so an unusable one is the standard history_end_invalid refusal before
        # anything is loaded or cached: never a plain ValueError from the tested-date parse just below, and
        # never a refusal that first lets the load step write the per-run OHLC cache. Off, this does not run.
        try:
            history_cutoff.validate_guard_arguments(tested_date=tested_date, max_drop=max_history_drop)
        except history_cutoff.HistoryCutoffError as exc:
            return _history_end_refusal(family_id, exc, history_end)
    # The tested date drives the last-bar sentence of the permanent note: a malformed one
    # must fail here, at the door, not after the control arms have been scored.
    try:
        parsed_tested_date = pd.Timestamp(tested_date)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"tested_date must be a YYYY-MM-DD date, got {tested_date!r}") from exc
    if pd.isna(parsed_tested_date):
        raise ValueError(f"tested_date must be a YYYY-MM-DD date, got {tested_date!r}")
    # The opt-in history end: a malformed or non-session value is refused here, at
    # the door, before anything is fetched, cached or recorded. None is the only off value.
    try:
        history_end_cutoff = history_cutoff.parse_history_end(history_end) if history_end is not None else None
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
    if not dry_run and n_random_draws != DEFAULT_RANDOM_DRAWS:
        return {"family_id": family_id, "reason_code": "control_configuration_drift",
                "reason": (f"a real run must use the pre-declared control configuration "
                           f"({DEFAULT_RANDOM_DRAWS} random-window draws), not {n_random_draws}")}

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
        # A real run's quote capture must be an open-market sample dated the run
        # date, taken 09:45 to 15:45 Eastern. Dry runs (synthetic quotes by design) are exempt. Nothing has
        # been fetched or written when this refuses.
        hours_refusal = quotes_market_hours_refusal(quotes_payload, tested_date)
        if hours_refusal is not None:
            return {"family_id": family_id, "reason_code": "quotes_not_market_hours", "reason": hours_refusal}
    costs = batch5.execution_costs_for([MARKET_BENCHMARK], quotes_payload)
    spy_cost_bps = costs[MARKET_BENCHMARK]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    loader = ohlc_loader or batch5.default_ohlc_loader
    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    spy_frame, spy_source = batch5.load_frame(MARKET_BENCHMARK, ohlc_loader=loader, cache_dir=cache_dir)
    # ^IRX is a signal-only series that can print at or below zero: it is loaded
    # WITHOUT the positivity gate meant for traded prices (never for SPY).
    irx_frame, irx_source = batch5.load_frame(FINANCING_SERIES, ohlc_loader=loader, cache_dir=cache_dir,
                                              require_positive=False)
    sources = {MARKET_BENCHMARK: spy_source, FINANCING_SERIES: irx_source}
    cached_symbols = [s for s, source in sources.items() if source == "cache"]
    if cached_symbols and not dry_run:
        # A permanent one-shot row must be scored from histories fetched by THIS
        # run: bars cached under research-root/run-id by an earlier run can be
        # stale or truncated.
        return {"family_id": family_id, "reason_code": "stale_ohlc_cache", "cached_symbols": cached_symbols,
                "reason": ("a real run must score freshly fetched histories, but "
                           f"{', '.join(cached_symbols)} came from an OHLC cache left under run id "
                           f"{run_id!r} by an earlier run; use a fresh --run-id (or empty that run's "
                           "cache dir) and run again")}

    history_end_provenance = None
    history_end_guard_message = None
    if history_end_cutoff is not None:
        # Opt-in history end: trim IN MEMORY, after the frames were loaded and cached above (the cache keeps
        # the full vendor history) and after the stale-cache refusal. Every later use reads the trimmed
        # variables, and every calendar / hole / final-window check below runs on them.
        try:
            trimmed, history_end_provenance = history_cutoff.trim_frames_to_history_end(
                {MARKET_BENCHMARK: spy_frame, FINANCING_SERIES: irx_frame}, history_end_cutoff)
        except history_cutoff.HistoryCutoffError as exc:
            return _history_end_refusal(family_id, exc, history_end)
        spy_frame, irx_frame = trimmed[MARKET_BENCHMARK], trimmed[FINANCING_SERIES]
        # The real-run guard: a one-shot ledger row must not spend its slot on a history end later than the
        # run's own tested date or on an unexpectedly large drop. A dry run never refuses on it; it reports
        # what the real run would do, so a rehearsal predicts the real run.
        try:
            history_cutoff.check_real_run_guard(
                history_end_provenance, tested_date=tested_date, max_drop=max_history_drop)
        except history_cutoff.HistoryCutoffError as exc:
            if not dry_run:
                return _history_end_refusal(family_id, exc, history_end)
            history_end_guard_message = str(exc)
    setting = _history_end_fields(history_end, history_end_provenance)

    try:
        gc_close, hg_close, manifest = mod.load_frozen_futures(futures_dir)
        ctx = mod.prepare(spy_frame, irx_frame, gc_close, hg_close)
    except mod.FuturesReadError as exc:
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc),
                "missing_dates": exc.missing_dates[:_MAX_LISTED_DATES],
                "n_missing_dates": len(exc.missing_dates), **setting}
    except mod.CalendarCoverageError as exc:
        # A data-calendar gap is refused, never trimmed or patched.
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc),
                "missing_dates": exc.missing_dates[:_MAX_LISTED_DATES],
                "n_missing_dates": len(exc.missing_dates),
                "extra_dates": exc.extra_dates[:_MAX_LISTED_DATES], "n_extra_dates": len(exc.extra_dates),
                **setting}
    futures = {"through": manifest["through"], "files": manifest["files"], "sha256_verified": True,
               "directory": batch5.portable_path(futures_dir)}
    if history_end_provenance is not None:
        # The setting never trims the frozen futures: report each cached series' own last bar (read from the
        # loaded series, not the manifest) and that no later row can influence a decision. No gate is added on them:
        # a cache that ends before a needed read is already refused as futures_read_hole.
        futures["last_bar_dates"] = {"GC=F": gc_close.index[-1].strftime("%Y-%m-%d"),
                                     "HG=F": hg_close.index[-1].strftime("%Y-%m-%d")}
        futures["trimmed_by_history_end"] = False
        futures["history_end_note"] = ("the frozen futures were not trimmed by the history end; the files are "
                                       "loaded whole and validated, and each decision reads only the last futures "
                                       "row on or before its own read date (the last NYSE session strictly before "
                                       "its month-end, at or before the history end)")

    curve = mod.build_curve(spy_frame, irx_frame, gc_close, hg_close, context=ctx,
                            spy_cost_bps=spy_cost_bps, spy_entry_cost_bps=benchmark_entry_cost_bps)
    info = curve["window_info"]
    if info["truncated_final"] and not dry_run and not accept_open_final_window:
        # A state-driven window is ordinarily open at the data's end. Closing it at
        # the last close could bake a live partial bar into the exit price, so it is
        # refused unless the caller accepts that explicitly (and it is then recorded).
        reason = ("the data ends inside an open window (its scheduled exit month-end lies after "
                  "the last bar), so the final trade would be closed early at the last close; "
                  "run after the close of that exit month-end, or pass --accept-open-final-window "
                  "to score it closed at the last close and record that in the note")
        if history_end_provenance is not None:
            reason = (f"the scored history was ended at {history_end_provenance['history_end']} by the opt-in "
                      "--history-end setting (later vendor sessions were dropped), and that last bar lies "
                      "inside an open window (its scheduled exit month-end lies after it), so the final trade "
                      "would be closed early at the last close; end the history at a later session, or pass "
                      "--accept-open-final-window to score it closed at the last close and record that in "
                      "the note")
        return {"family_id": family_id, "reason_code": "truncated_final_window", "reason": reason, **setting}

    account_type = entry["account_types"][0]
    eval_kwargs = dict(benchmark_frame=spy_frame, benchmark_entry_cost_bps=benchmark_entry_cost_bps,
                       account_type=account_type)
    result = abi1_driver.evaluate_family(curve, tax_policy=TAX_POLICY, **eval_kwargs)
    benchmark_daily = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        spy_frame, entry_cost_bps=benchmark_entry_cost_bps)
    # The ledger note states the risk basis and that the excess
    # is the overlay's own; refuse to record anything either claim is false for.
    overlay_risk_report.require_overlay_basis(result, family_id=family_id)
    overlay_risk_report.require_excess_is_overlay_only(
        result["account_daily"], benchmark_daily, curve["overlay_mask"], family_id=family_id)
    untaxed_result = abi1_driver.evaluate_family(curve, tax_policy="none", **eval_kwargs)

    diag = overlay_diagnostics(
        curve, result, untaxed_result=untaxed_result, benchmark_daily=benchmark_daily, spy_frame=spy_frame,
        spy_cost_bps=spy_cost_bps, benchmark_entry_cost_bps=benchmark_entry_cost_bps)
    diag["holdout_window_high"] = holdout_window_high(
        ctx["decisions"], ctx["states"], result["account_daily"].index, result["window"].get("holdout_sessions"))
    controls = control_arms(
        ctx, spy_frame, eval_kwargs, spy_cost_bps=spy_cost_bps, benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        n_draws=n_random_draws, main_sharpe=_sharpe(result), main_sharpe_untaxed=_sharpe(untaxed_result))

    risk_note = overlay_risk_report.risk_layer_note(diag["risk_layer"], ordering_clause=ORDERING_CLAUSE)
    extra_note = build_ledger_note(
        diag=diag, controls=controls, risk_note=risk_note, quotes_source=batch5.quotes_cost_source(quotes_payload),
        spy_cost_bps=spy_cost_bps, sources=sources, futures=futures, curve=curve, spy_frame=spy_frame,
        main_sharpe=_sharpe(result), accepted_open=bool(accept_open_final_window),
        tested_date=tested_date, provenance=mod.schedule_provenance(ctx), abi1_verdict=result["verdict"],
        module_sha256=evidence.source_sha256(mod.__file__), driver_sha256=signal_source_sha256,
        history_end_provenance=history_end_provenance)

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        record_kwargs = {}
        if history_end_provenance is not None:
            # An UNSCORED verdict (not_scoreable / holdout_not_evaluable) writes an insufficient_evidence row
            # that would otherwise carry no note, yet it spends the same permanent ABI-1 slot: hand the shared
            # recorder the same clause the note above leads with, for that row. A scored row never reads it.
            # Off, the keyword is not passed at all.
            record_kwargs["unscored_note"] = history_cutoff.history_end_clause(history_end_provenance)
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result,
            catalog_entry=entry, instrument_set=list(mod.TRADED),
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
            tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
            signal_source_path=signal_source_path, signal_source_sha256=signal_source_sha256,
            **record_kwargs)

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    account_index = result["account_daily"].index
    signal_evidence = controls["interpretation"]["signal_evidence"]
    output = {
        "family_id": family_id, "shape": mod.SHAPE, "tickers": list(mod.TRADED),
        "verdict": result["verdict"], "stage": stage,
        # The recorded stage is unconditional; the controls gate THIS field
        "pre_registered_pass": pre_registered_pass(result["verdict"], signal_evidence),
        "pre_registered_clause": pre_registered_clause(result["verdict"], signal_evidence),
        "dsr_value": layer1.get("dsr_value"), "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"), "n_trades": int(len(curve["trades_df"])),
        "account_start": account_index[0].strftime("%Y-%m-%d") if len(account_index) else None,
        "account_end": account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None,
        "time_in_market": result.get("time_in_market"),
        "tax_policy": TAX_POLICY, "cash_convention": CASH_CONVENTION, "benchmark": MARKET_BENCHMARK,
        "cost_model_note": extra_note,
        "holdout_evaluated": holdout.get("evaluated"), "holdout_passed": holdout.get("passed"),
        "holdout_psr": holdout.get("psr"),
        "window": {
            "account_sessions": window.get("account_sessions"), "scored_sessions": window.get("scored_sessions"),
            "in_sample_sessions": window.get("in_sample_sessions"),
            "holdout_sessions": window.get("holdout_sessions"),
        },
        # provenance: "cache" means the history was re-used from an earlier run under
        # the SAME run_id. Only a dry run can get here with a cache hit: a real run
        # REFUSES it (reason_code stale_ohlc_cache).
        "ohlc_sources": sources,
        "last_bar_dates": {MARKET_BENCHMARK: spy_frame.index[-1].strftime("%Y-%m-%d"),
                           FINANCING_SERIES: irx_frame.index[-1].strftime("%Y-%m-%d")},
        "futures_provenance": futures,
        "overlay_diagnostics": diag,
        "controls": controls,
        "accepted_open_final_window": bool(accept_open_final_window),
    }
    if history_end_provenance is not None:
        # `last_bar_dates` above describe the TRIMMED history; the vendor's raw last bars sit beside them.
        output.update(setting)                       # history_end_requested (the raw string) and history_end
        output["raw_last_bar_dates"] = {
            MARKET_BENCHMARK: history_end_provenance["by_symbol"][MARKET_BENCHMARK]["raw_last"],
            FINANCING_SERIES: history_end_provenance["by_symbol"][FINANCING_SERIES]["raw_last"]}
        if dry_run:
            # A dry run never refuses on the real-run guard: it says what a real run would do (None = accept).
            output["history_end_guard_would_refuse"] = history_end_guard_message
    return output


# ----------------------------------------------------------------------- CLI

def _non_negative_int(text) -> int:
    """argparse type for ``--max-history-drop``: a whole number of sessions, zero or more."""
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of sessions") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative; a drop limit is a number of sessions, zero or more")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", default=None,
                        help="Path to a dated quotes JSON file with bid/ask for SPY (must be a REAL captured "
                             "book unless --dry-run is also passed, in which case a SYNTHETIC placeholder "
                             "file is expected). Not needed with --freeze-futures.")
    parser.add_argument("--run-id", default=None,
                        help="REQUIRED for a real run (it names the OHLC cache dir, and a real run refuses "
                             "any history cached there by an earlier run, so it must be a fresh id); a "
                             "--dry-run falls back to a shared default.")
    parser.add_argument("--tested-date", default=None,
                        help="YYYY-MM-DD; REQUIRED for a real run (it is written into a permanent ledger "
                             "row); a --dry-run falls back to a placeholder.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument("--futures-dir", default=FUTURES_DIR,
                        help="The frozen futures cache (yahoo_GC_F.csv, yahoo_HG_F.csv, manifest.json).")
    parser.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS,
                        help="Random-window control draws; a real run must use the pre-declared count.")
    parser.add_argument("--accept-open-final-window", action="store_true",
                        help="Score a window still open at the data's end (closed at the last close) and "
                             "record that acceptance in the ledger note.")
    parser.add_argument("--history-end", default=None, metavar="YYYY-MM-DD",
                        help="Opt-in: end the scored history at this NYSE session (every SPY and ^IRX series "
                             "must have a bar on it); later sessions are dropped, never filled or patched. "
                             "PERMANENT for the scored account: the outcome row's account ends here, a new "
                             "family's holdout window ends here too, and the family cannot be re-scored on "
                             "longer history. A refusal of the setting (a history_end_* "
                             "code, exit status 2) writes no ledger row, result file or report; when it comes "
                             "after the load step the per-run OHLC cache is kept, so a real retry needs a "
                             "fresh --run-id. Cannot be combined with --freeze-futures. Off by default.")
    parser.add_argument("--max-history-drop", type=_non_negative_int, metavar="N",
                        default=history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS,
                        help="Real runs only, with --history-end: refuse the run (history_end_drop_too_large) if "
                             "any series would lose more than N sessions to the history end (default "
                             f"{history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS}). A dry run never refuses on it; its "
                             "result's history_end_guard_would_refuse says what a real run would do.")
    parser.add_argument("--freeze-futures", action="store_true",
                        help="Fetch GC=F and HG=F and freeze them under --futures-dir with a sha256 "
                             "manifest, then exit (never overwrites unless --overwrite-frozen).")
    parser.add_argument("--through", default=None,
                        help="With --freeze-futures: the last COMPLETED session to keep (later bars, the "
                             "live partial bar, are dropped).")
    parser.add_argument("--overwrite-frozen", action="store_true")
    args = parser.parse_args(argv)

    # One usage trap, opt-in like the setting itself, before anything is fetched or written: the freeze path
    # returns before --history-end is ever read, so the combination would silently drop the setting.
    # --through without --freeze-futures is accepted (a score ignores it).
    if args.freeze_futures and args.history_end is not None:
        parser.error("--history-end cannot be combined with --freeze-futures (a freeze fetches and writes the "
                     "futures cache and never scores, so the setting would be silently ignored)")

    if args.freeze_futures:
        if not args.through:
            parser.error("--freeze-futures requires an explicit --through YYYY-MM-DD (the last completed "
                         "session: a live partial bar must never be frozen)")
        manifest = mod.freeze_futures(args.futures_dir, fetcher=mod.real_yahoo_fetcher, through=args.through,
                                      overwrite=args.overwrite_frozen)
        print(json.dumps({"frozen": batch5.portable_path(args.futures_dir), "manifest": manifest}))
        return 0

    if not args.quotes:
        parser.error("--quotes is required")
    if args.tested_date is None:
        if not args.dry_run:
            parser.error("--tested-date is required for a real (non --dry-run) run: it is written into a "
                         "permanent ledger row and must never fall back to a stale default")
        args.tested_date = DEFAULT_TESTED_DATE or pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.run_id is None:
        if not args.dry_run:
            parser.error("--run-id is required for a real (non --dry-run) run: it names the OHLC cache dir, "
                         "and a real run refuses history cached there by an earlier run (a dry run's bars), "
                         "so it must be a fresh id")
        args.run_id = DEFAULT_RUN_ID
    if not args.dry_run and args.random_draws != DEFAULT_RANDOM_DRAWS:
        parser.error(f"a real run must use the pre-declared {DEFAULT_RANDOM_DRAWS} random-window draws; "
                     "--random-draws may differ only on a --dry-run")

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes, ledger_path=args.ledger, run_id=args.run_id,
        tested_date=args.tested_date, dry_run=args.dry_run, signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__), research_root=args.research_root,
        futures_dir=args.futures_dir, n_random_draws=args.random_draws,
        accept_open_final_window=args.accept_open_final_window, history_end=args.history_end,
        max_history_drop=args.max_history_drop)

    if result.get("reason_code") in history_cutoff.HISTORY_END_REASON_CODES:
        # A refused history end writes no ledger row, no result file and no report (the per-run OHLC cache the
        # load step wrote is kept: a real retry needs a fresh --run-id). The refusal payload is the last
        # printed line and the exit code is 2.
        print(json.dumps({"output": None, "result": batch5._json_safe(result)}))
        return 2

    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    safe_result = batch5._json_safe(result)
    if safe_result.get("pre_registered_clause"):
        print(safe_result["pre_registered_clause"])      # the same sentence the note leads with; JSON stays the last line
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id, "tested_date": args.tested_date,
        "dry_run": args.dry_run, "driver": _SIGNAL_SOURCE_PATH, "result": safe_result,
    }
    if args.history_end is not None:
        # the opt-in setting, at the top of the file too: the raw request whatever the result is (a refusal
        # that came before the trim included), and the provenance whenever the trim ran
        payload["history_end_requested"] = args.history_end
    if "history_end" in safe_result:
        payload["history_end"] = safe_result["history_end"]
    output_path = os.path.join(
        run_dir, "gold-copper-overlay-dry-run.json" if args.dry_run else "gold-copper-overlay-results.json")
    batch3._atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
