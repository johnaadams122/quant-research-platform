"""Bespoke ABI-1 driver for residual_momentum_ff3proxy_top50_monthly --
an SPY core plus a dollar-neutral cohort overlay: long the equal-weight top-50
residual-momentum cohort of a large-cap panel, short the equal-weight basket of
ALL eligible names (a RESEARCH CONTROL, not an investable account), over each
month's holding window. Uses the same quotes / dry-run / refuse-synthetic /
parameter-drift / already-scored / stale-cache / calendar-refusal checks as
scripts/run_treasury_month_end_overlay_abi1.py.

WHY A BESPOKE DRIVER: the generic scripts/run_abi1_mining.py dispatch builds a
single-position, rotation or fixed-sleeve pooled curve from per-bar signal
functions. This family's account is a benchmark-plus-overlay curve built by
``mining.batch5.residual_momentum_ff3proxy_top50_monthly.build_curve`` (math and
conventions in ``mining.cohort_overlay``), and that module's ``make_signal_fn``
always raises so the generic path cannot mis-score it. The curve is fed to
``abi1_driver.evaluate_family`` exactly as any builder's dict is.

WHAT IS PRINTED BESIDE THE MAIN RESULT (report-only, never gating, never a
ledger row of their own; the ledger note states the pre-declared reading rule):

  * the ZERO-SIGNAL EQUAL-WEIGHT CONTROL: the whole account holding the
    equal-weight basket of the same eligible names, scored by ABI-1 on its own
    (verdict, DSR, daily excess Sharpe, holdout). It shows what the panel alone
    scores against SPY with no signal -- the survivorship baseline the overlay
    cancels;
  * the PERMUTATION PLACEBO: 500 seeded random cohorts of the real size from the
    same eligible set (same schedule, costs, financing), and the real rule's
    percentile among them;
  * the four ABI-1 risk refusals on the OVERLAY-ONLY EXCESS account (the
    risk-basis rule) with a whole-account reference block; the claim that the scored
    excess is the overlay's own is checked on EVERY session
    (``cohort_overlay.require_account_is_benchmark_plus_overlay``), because the
    shared off-window guard reads only the one session the overlay is off on
    this contiguous monthly schedule;
  * details: windows, eligible counts, hygiene and quote exclusions, the
    financing charged, the tax-identity approximation, the OLS beta of the scored
    excess on SPY.

PRE-DECLARED READING RULE (written into the ledger note before any real run): a
pass is meaningful only if the real rule's overlay Sharpe percentile among the
permutation placebo draws is at least 0.95; a DSR pass below that is read as NOT
distinguished from random selection of the same names. The main score is
already net of the equal-weight eligible basket (its short leg), which cancels
the LEVEL of the survivorship baseline only. Rank-conditional survivorship (past
residual-momentum winners that later failed, or were added to the index later)
is NOT cancelled, and the placebo, which draws its random cohorts from the same
eligible set, shares that limitation and cannot test it: a pass is therefore
INCONCLUSIVE, not confirmatory, and a cut is the informative outcome. The
control's own numbers are printed to show how large the level baseline is.

NOT MODELLED (written into the ledger note). Each of these flatters the
strategy: no short borrow fee; no margin spread over the bill rate; Reg-T margin
calls and borrow availability; the tax walk treats financing interest as
deductible. A short rebate is not credited either (conservative). Direction
unmeasured: today's-index-members survivorship (the rank-conditional part is not
cancelled, see above); 2026 quoted spreads applied to every year (plausibly
understating early-year costs); the hygiene and quote exclusions, decided on
full-sample or 2026 data and applied to all history (a data-quality rule, not a
return-based one, applied equally to both legs).

Usage (network: yfinance historical OHLC for SPY, IWM, IVE, IVW and ^IRX only;
the 500-name panel is read from a frozen dated cache). A --dry-run proves the
wiring against a SYNTHETIC quotes file (its costs must NOT be reported as a cost
measurement); a scoring run needs a real dated quotes capture and writes
ONE permanent ledger row:
    python scripts/run_residual_momentum_overlay_abi1.py \\
        --dry-run --quotes <synthetic-quotes.json> --research-root <scratch dir> \\
        --catalog <a catalog json containing the family's entry>

    python scripts/run_residual_momentum_overlay_abi1.py \\
        --quotes <real-dated-capture.json> --run-id <run-id> --tested-date <YYYY-MM-DD>

The quotes file needs bid/ask for SPY and for the panel names (a name whose
quote is missing, suspect or wider than the gate is excluded and listed; fewer
than 80 percent of the panel priced refuses the run). A real run needs an
explicit --tested-date AND --run-id and REFUSES (``stale_ohlc_cache``, nothing
written) any market history an earlier run left cached under that run id.

DATA REFUSALS (reason-coded, nothing written; refetch or wait for the vendor):
``unequal_trailing_coverage`` / ``calendar_mismatch`` /
``missing_exchange_session`` / ``bar_on_closed_day`` for SPY and the factor ETFs
(the pipeline's rule: refuse a history with a calendar hole, never patch or
fill it -- a vendor daily-bar hole missing one session for many ETFs at once is
such a case), ``missing_member_price`` / ``missing_member_frame`` /
``missing_member_cost`` (a panel name lacking the entry Open or a window Close),
``ohlc_load_failed`` (a factor ETF that could not be loaded after retries),
``stale_bill_print`` (a financed session whose ^IRX print is more than
``mod.MAX_BILL_PRINT_AGE_DAYS`` calendar days old, or absent: a hole the shared
causal financing rate would carry across silently), ``no_windows`` and
``insufficient_panel_coverage``. Panel names failing hygiene (a documented
defect, an interior gap since the first needed month-end, a stale history, a bar
on a closed day) are EXCLUDED and listed, never repaired.

AS-OF CUTOFF: the market frames are cut to the panel's own last session (the
frozen panel is a dated snapshot), and only COMPLETE monthly windows are scored:
a final month whose scheduled exit lies after the data is excluded and reported
(``open_window_excluded``), never closed early at the last close.

QUOTE PROVENANCE IS A HUMAN-PROCESS SAFEGUARD. ``batch5.refuse_synthetic_quotes``
only refuses a real run when the quotes file carries the word SYNTHETIC; it
cannot tell a measured book from an invented bid/ask file that omits the marker
(a pipeline-wide limit in shared code). Real quotes must come from a measured
broker quote capture, never typed or estimated by hand.
On top of it a REAL run refuses (reason code ``quotes_not_market_hours``, nothing
fetched or written, dry runs exempt) unless the capture's ``market_state`` starts
with ``open`` (any case, surrounding whitespace ignored), ``sampled_at_local``
carries an ISO date and parses (a naive time is read as US Eastern, an offset-aware
one is converted to it), its Eastern calendar date is the run's ``--tested-date``
and its Eastern time of day is within 09:45 to 15:45 Eastern inclusive
(``quotes_market_hours_refusal``). After-hours cost samples are unusable,
and the shared guard cannot tell a closed market's book
from a live one.

THE PLACEBO GATES THE PRE-REGISTERED PASS, NOT THE RECORDED STAGE. The stage
vocabulary is closed in shared code (``evidence.VALID_STAGES``) and every tested
family is recorded, so a main ABI-1 ``survived`` is recorded as such; the result's
``pre_registered_pass`` is True only when the family's own control condition also
holds (the real rule's overlay Sharpe percentile among the seeded random same-size
cohorts is at least 0.95), and when it does not the driver-supplied note (and the
printed summary) LEADS with ``CONTROL_CONDITION_FAILED_FLAG`` (CONTROL CONDITION
FAILED: not a pass; do not promote it without review).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

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
                    cohort_overlay, evidence, overlay_risk_report, trial_ledger)
from mining.batch5 import residual_momentum_ff3proxy_top50_monthly as mod

CASH_CONVENTION = "sweep_cash_daily"   # recorded; the account is always fully
                                       # deployed (SPY core), so idle cash is never credited
MARKET_BENCHMARK = "SPY"
FINANCING_SERIES = "^IRX"
# "realization": the overlay's monthly round trips are short-term, taxed at exit
# with loss netting; the dollar-attribution approximation is measured and reported.
TAX_POLICY = "realization"
CONTROL_TAX_POLICY = "none"            # the control is a pooled whole-account basket

DEFAULT_RUN_ID = "residual-momentum-overlay-abi1-v1"   # dry runs only: a real run must pass its own --run-id
DEFAULT_TESTED_DATE = batch5.DEFAULT_TESTED_DATE
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
PANEL_CACHE_DIR = os.path.join(REPO_ROOT, "data", "mining", "universe_cache", "ohlc_2026-08-29")

PLACEBO_PERCENTILE_BAR = 0.95          # the pre-declared reading threshold (see docstring)
FACTOR_LOAD_ATTEMPTS = 3
_MAX_LISTED = 50

# The family's OWN statement of how the overlay-only risk-basis rule is ordered relative
# to this family's ledger rows (mining.overlay_risk_report.risk_layer_note).
ORDERING_CLAUSE = (
    "the risk-basis rule was set before this family was built and before any of "
    "its ledger rows, so no whole-account risk number of this family preceded it")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")

# A REAL run prices its one-shot verdict off a quote capture, so it also refuses a capture that was not taken
# during the regular session of the run's own date. The shared synthetic-quote
# guard only looks for the word SYNTHETIC and cannot tell a closed-market book from a live one.
MARKET_TIMEZONE = "America/New_York"
CAPTURE_WINDOW_START = (9, 45)     # Eastern, inclusive
CAPTURE_WINDOW_END = (15, 45)      # Eastern, inclusive
_ISO_DATE_PREFIX = re.compile(r"\s*\d{4}-\d{2}-\d{2}")


def _eastern_stamp(value):
    """A US Eastern ``pd.Timestamp`` for a string that carries an ISO date and parses (a naive time is read
    as Eastern, an offset-aware one is converted to it), else None. Never raises. A date-less string is
    refused because pandas reads "10:00" as the CURRENT date, which would let a time-only capture pass the
    date check on the run date."""
    if not isinstance(value, str) or not _ISO_DATE_PREFIX.match(value):
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        return stamp.tz_localize(MARKET_TIMEZONE) if stamp.tzinfo is None else stamp.tz_convert(MARKET_TIMEZONE)
    except Exception:                    # a parse failure, a nonexistent local time, an overflow: a refusal
        return None


def quotes_market_hours_refusal(quotes_payload, tested_date):
    """``None`` when the quote capture is acceptable for a REAL run, else the text of why not. Accepted only
    when (i) ``market_state`` (case-insensitive, surrounding whitespace ignored) starts with ``open``, (ii)
    ``sampled_at_local`` is a string carrying an ISO date that parses as a time (a naive time is read as US
    Eastern, an offset-aware one is converted to it), (iii) its Eastern calendar date equals ``tested_date``
    and (iv) its Eastern time of day is between 09:45:00 and 15:45:00 inclusive. Never raises on a bad
    payload or run date: a missing, non-string or unparseable field is a refusal."""
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
    value = float(value)
    return value if np.isfinite(value) else None


# ------------------------------------------------------------------ loading

def retrying_loader(loader, *, attempts: int = FACTOR_LOAD_ATTEMPTS, sleep=time.sleep):
    """Wrap an OHLC loader so a transient vendor failure (an empty frame or an
    exception, as a single symbol inside a bulk download can return) is
    retried, then FAILS CLOSED with a RuntimeError naming the symbol. Never
    returns an empty or partial history."""
    def load(symbol):
        last = "no attempt made"
        for attempt in range(1, int(attempts) + 1):
            try:
                raw = loader(symbol)
                if raw is not None and not (isinstance(raw, pd.DataFrame) and raw.empty):
                    return raw
                last = "the loader returned no data"
            except Exception as exc:                       # any vendor failure is retried
                last = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                sleep(float(attempt))
        raise RuntimeError(f"{symbol}: OHLC load failed after {attempts} attempt(s) ({last})")
    return load


def _default_panel_loader(panel_cache_dir: str):
    def load(symbol):
        path = os.path.join(panel_cache_dir, batch5._cache_name(symbol))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{symbol}: no frozen panel file")
        return pd.read_csv(path)
    return load


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_panel(traded, panel_loader) -> tuple:
    """``(frames, excluded)``: every panel name loaded through the same
    normalization the pooled drivers use (all-blank rows dropped, prices finite
    and positive). A name that cannot be loaded is EXCLUDED with its reason,
    never repaired."""
    frames, excluded = {}, {}
    for symbol in traded:
        try:
            raw = panel_loader(symbol)
            raw, dropped = batch5._drop_all_nan_rows(raw)
            frames[symbol] = batch3._normalize_ohlc(raw, symbol)
        except FileNotFoundError as exc:
            excluded[symbol] = {"reason": "no_panel_file", "detail": str(exc)}
        except Exception as exc:
            excluded[symbol] = {"reason": "panel_load_failed", "detail": f"{type(exc).__name__}: {exc}"}
    return frames, excluded


def _panel_as_of(frames) -> pd.Timestamp:
    """The panel's own last session: the most common last-bar date (a later
    date wins a tie)."""
    counts = collections.Counter(frame.index[-1] for frame in frames.values())
    return max(counts, key=lambda d: (counts[d], d))


def _quote_costs(traded, quotes_payload) -> tuple:
    costs, excluded = {}, {}
    for symbol in traded:
        try:
            costs[symbol] = batch5.execution_costs_for([symbol], quotes_payload)[symbol]
        except ValueError as exc:
            excluded[symbol] = str(exc)
    return costs, excluded


# ------------------------------------------------------------- details

def _annualized_blocks(result: dict) -> dict:
    """Annualized mean log excess and excess Sharpe (252 sessions) of the scored
    excess series in-sample and over the holdout tail, computed independently of
    the DSR machinery so a reader can compare them with a published headline."""
    excess = np.asarray(result["log_excess"], dtype=float)
    window = result["window"]
    n_hold = int(window.get("holdout_sessions") or 0) if window.get("holdout_split_applied") else 0

    def block(values):
        if len(values) < 2:
            return None
        std = float(np.std(values, ddof=1))
        return {"n_sessions": int(len(values)),
                "annualized_mean_log_excess": _finite(values.mean() * 252.0),
                "annualized_excess_sharpe": (_finite(values.mean() / std * np.sqrt(252.0)) if std > 0 else None)}

    return {"in_sample": block(excess[:-n_hold] if n_hold else excess),
            "holdout": block(excess[-n_hold:]) if n_hold else None}


def _control_summary(result: dict, *, scoring: str) -> dict:
    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    layer2 = result.get("layer2") or {}
    return {
        "verdict": result["verdict"],
        "dsr_value": _finite(layer1["dsr_value"]) if layer1.get("dsr_value") is not None else None,
        "sharpe_daily_excess": (_finite(layer1["sharpe_daily_excess"])
                                if layer1.get("sharpe_daily_excess") is not None else None),
        "n_sessions": layer1.get("n_sessions"),
        "holdout_evaluated": holdout.get("evaluated"), "holdout_passed": holdout.get("passed"),
        "holdout_psr": holdout.get("psr"),
        "account_sessions": window.get("account_sessions"),
        "risk_account_basis": layer2.get("account_basis"),
        "annualized": _annualized_blocks(result),
        "scoring": scoring,
    }


def _placebo_summary(placebo: dict) -> dict:
    keep = ("statistic", "n_draws", "seed", "real_value", "draws_mean", "draws_std", "draws_p05",
            "draws_p50", "draws_p95", "real_percentile", "p_upper", "note")
    out = {k: placebo[k] for k in keep}
    out["percentile_bar"] = PLACEBO_PERCENTILE_BAR
    out["clears_bar"] = bool(placebo["real_percentile"] >= PLACEBO_PERCENTILE_BAR)
    return out


# The stage vocabulary is CLOSED in shared code (evidence.VALID_STAGES) and every tested family is recorded
# unconditionally, so the placebo cannot change the recorded stage. It gates the PRE-REGISTERED PASS instead:
# a main ABI-1 'survived' whose overlay Sharpe percentile among the random-cohort placebo draws is below the
# bar is recorded as 'survived' but is NOT a survivor under this family's own rule, and the note and the
# printed summary say so in their first sentence.
CONTROL_RULE = ("the real rule's overlay Sharpe percentile among the seeded random same-size cohort placebo "
                f"draws must be at least {PLACEBO_PERCENTILE_BAR}")
CONTROL_CONDITION_FAILED_FLAG = (
    "CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under this family's own "
    f"pre-registered rule ({CONTROL_RULE}). This is not a pass; do not promote it without review.")
PRE_REGISTERED_RULE_MET_CLAUSE = (
    f"PRE-REGISTERED RULE MET: the ABI-1 stage survived AND the control condition holds ({CONTROL_RULE}); "
    "pre_registered_pass=True.")


def control_condition_holds(placebo_summary) -> bool:
    """The family's pre-declared control condition: True only when the placebo summary's ``clears_bar`` is
    exactly True. A missing summary, a missing key or any other value means the condition did not hold."""
    return bool(isinstance(placebo_summary, dict) and placebo_summary.get("clears_bar") is True)


def pre_registered_pass(verdict, control_condition) -> bool:
    """True only when the ABI-1 verdict is ``survived`` AND the control condition is exactly True (a missing
    or False control condition is not a pass)."""
    return bool(verdict == "survived" and control_condition is True)


def pre_registered_clause(verdict, control_condition) -> str:
    """The sentence a survived stage carries: the unmissable flag when the control condition did not hold
    (False or missing), the rule-met statement when it did, and nothing for any other verdict."""
    if verdict != "survived":
        return ""
    return PRE_REGISTERED_RULE_MET_CLAUSE if control_condition is True else CONTROL_CONDITION_FAILED_FLAG


def ledger_note_lead(verdict, control_condition) -> str:
    """What the driver-supplied ledger note is prefixed with: the clause and one space, or nothing. The
    shared writer puts its own fixed metadata prefix ahead of the driver's note, which is not editable
    from here, so this is the FIRST sentence of the note the driver controls."""
    clause = pre_registered_clause(verdict, control_condition)
    return f"{clause} " if clause else ""


def _iso(stamp):
    return None if stamp is None else pd.Timestamp(stamp).strftime("%Y-%m-%d")


def _iso_window(window):
    return None if window is None else {k: _iso(v) for k, v in window.items()}


def overlay_diagnostics(curve: dict, result: dict, *, untaxed_result: dict, benchmark_daily,
                        hygiene_excluded: dict, quote_excluded: dict, panel_excluded: dict,
                        n_traded: int, n_used: int, cost_bps: dict, benchmark_entry_cost_bps: float) -> dict:
    """Report-only structural details for the built curve. Nothing here
    gates a verdict."""
    trades = curve["trades_df"]
    info = curve["window_info"]
    index = curve["strategy_daily"].index
    mask = curve["overlay_mask"].to_numpy(dtype=bool)
    by_reason = collections.Counter(v["reason"] for v in hygiene_excluded.values())
    costs = np.array(list(cost_bps.values()), dtype=float)
    diag = {
        "overlay_notional_fraction": float(mod.OVERLAY_NOTIONAL),
        "n_select": int(mod.N_SELECT),
        "n_windows": int(len(trades)),
        "n_months_skipped": int(info["n_months_skipped"]),
        "open_window_excluded": _iso_window(info["open_window_excluded"]),
        "not_started": _iso_window(info["not_started"]),
        "first_decision": _iso(info["first_decision"]), "last_exit": _iso(info["last_exit"]),
        "eligible_min": info["eligible_min"], "eligible_median": info["eligible_median"],
        "eligible_max": info["eligible_max"],
        "bill_print_max_age_days": info.get("bill_print_max_age_days"),
        "panel": {"traded_declared": int(n_traded), "used": int(n_used),
                  "hygiene_excluded_by_reason": dict(by_reason),
                  "hygiene_excluded": {k: v for k, v in list(hygiene_excluded.items())[:_MAX_LISTED]},
                  "quote_excluded": {k: v for k, v in list(quote_excluded.items())[:_MAX_LISTED]},
                  "n_quote_excluded": len(quote_excluded),
                  "load_excluded": {k: v for k, v in list(panel_excluded.items())[:_MAX_LISTED]},
                  "n_load_excluded": len(panel_excluded)},
        "cost_bps_round_trip": {"n": int(len(costs)), "mean": _finite(costs.mean()),
                                "min": _finite(costs.min()), "max": _finite(costs.max())},
        "benchmark_entry_cost_bps": float(benchmark_entry_cost_bps),
        "overlay_sessions": int(mask.sum()), "account_sessions": int(len(index)),
        "overlay_session_share": _finite(mask.mean()) if len(index) else None,
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
        growth = np.cumprod(1.0 + curve["strategy_daily"].to_numpy(dtype=float))
        entry_pos = index.get_indexer(pd.DatetimeIndex(trades["entry_date"]))
        exit_pos = index.get_indexer(pd.DatetimeIndex(trades["exit_date"]))
        # the growth from the sizing close (the session BEFORE the entry) to the exit close
        ratio = (growth[exit_pos] / growth[entry_pos - 1]) / (1.0 + net)
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
    return diag


def reading_rule_text(placebo: dict, control: dict, daily_control: dict) -> str:
    """The pre-declared reading rule with THIS run's measured control and
    placebo numbers (ledger note)."""
    return (
        "PRE-DECLARED READING RULE (fixed before any real run): the scored account is net of the "
        "equal-weight basket of ALL eligible names (its short leg, a RESEARCH CONTROL, not an "
        "investable account), which cancels the LEVEL of the survivorship baseline only (a panel of "
        "today's index members can score well against SPY with no signal, through survivorship "
        "alone); rank-conditional "
        "survivorship (past residual-momentum winners that later failed, or were added to the index "
        "later) is NOT cancelled, and the placebo, which draws its random cohorts from the same "
        "eligible set, shares that limitation and cannot test it, so a pass is INCONCLUSIVE, not "
        "confirmatory, and a cut is the informative outcome; a pass additionally requires that the real "
        f"rule's overlay Sharpe percentile among {placebo['n_draws']} seeded random same-size cohorts "
        f"drawn from the same eligible set (same schedule, costs and financing) is at least "
        f"{PLACEBO_PERCENTILE_BAR}; a DSR pass below that is read as NOT distinguished from random "
        f"selection of the same names. Measured: placebo percentile {_round(placebo['real_percentile'])} "
        f"(p_upper {_round(placebo['p_upper'])}, real overlay Sharpe {_round(placebo['real_value'])} "
        f"against placebo median {_round(placebo['draws_p50'])}, 95th percentile "
        f"{_round(placebo['draws_p95'])}), clears the bar: {placebo['clears_bar']}. Zero-signal "
        "equal-weight control, scored by ABI-1 on its own (report-only): "
        f"verdict {control['verdict']}, DSR {_round(control['dsr_value'])}, daily excess Sharpe "
        f"{_round(control['sharpe_daily_excess'])}, holdout passed {control['holdout_passed']}; the "
        "DAILY-REBALANCED zero-signal control (fully invested, gross of costs, an upper bound): "
        f"verdict {daily_control['verdict']}, DSR {_round(daily_control['dsr_value'])}, in-sample annualized "
        f"excess Sharpe {_round(_block_value(daily_control, 'in_sample'))}, holdout annualized mean log excess "
        f"{_round(_block_value(daily_control, 'holdout', 'annualized_mean_log_excess'))}; a control that "
        "clears the DSR bar means a whole-account test of these names would pass with no "
        "signal at all, and the overlay account is scored so that this level baseline cancels")


def _block_value(summary: dict, which: str, key: str = "annualized_excess_sharpe"):
    block = (summary.get("annualized") or {}).get(which)
    return None if block is None else block.get(key)


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


# ------------------------------------------------------------------ compute

def _calendar_refusal(family_id: str, exc) -> dict:
    return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc),
            "missing_dates": exc.missing_dates[:_MAX_LISTED], "n_missing_dates": len(exc.missing_dates),
            "extra_dates": exc.extra_dates[:_MAX_LISTED], "n_extra_dates": len(exc.extra_dates)}


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str, run_id: str,
                       tested_date: str, dry_run: bool, signal_source_path: str,
                       signal_source_sha256: str, research_root: str = RESEARCH_ROOT,
                       ohlc_loader=None, panel_loader=None, panel_cache_dir: str = PANEL_CACHE_DIR,
                       traded=None, first_month_end: str = mod.FIRST_MONTH_END,
                       placebo_draws: int = mod.PLACEBO_DRAWS, sleep=time.sleep) -> dict:
    family_id = mod.FAMILY_ID
    if not dry_run and int(placebo_draws) != mod.PLACEBO_DRAWS:
        raise ValueError(f"a real run must use the pre-registered {mod.PLACEBO_DRAWS} placebo draws; "
                         f"got {placebo_draws}")
    traded = list(mod.TRADED if traded is None else traded)

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

    already_scored = {row["family_id"] for row in trial_ledger.read_all(ledger_path)
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
    try:
        spy_costs = batch5.execution_costs_for([MARKET_BENCHMARK], quotes_payload)
    except ValueError as exc:
        return {"family_id": family_id, "reason_code": "missing_data",
                "reason": f"benchmark quote gate ({MARKET_BENCHMARK}): {exc}"}
    benchmark_entry_cost_bps = spy_costs[MARKET_BENCHMARK]["spread_bps"] / 2.0
    costs, quote_excluded = _quote_costs(traded, quotes_payload)
    if len(costs) < batch5.MIN_POOL_COVERAGE * len(traded):
        return {"family_id": family_id, "reason_code": "insufficient_panel_coverage",
                "reason": (f"only {len(costs)} of {len(traded)} panel names have a passing quote "
                           f"(minimum {batch5.MIN_POOL_COVERAGE:.0%}); excluded: "
                           + batch5._render_exclusions(quote_excluded))}

    # ---- market frames (fetched with retry, fail closed, frozen to a dated cache)
    loader = retrying_loader(ohlc_loader or batch5.default_ohlc_loader, sleep=sleep)
    cache_dir = os.path.join(research_root, run_id, "artifacts", "ohlc")
    market_symbols = [MARKET_BENCHMARK] + list(mod.FACTOR_SYMBOLS)
    frames, sources = {}, {}
    try:
        for symbol in market_symbols:
            frames[symbol], sources[symbol] = batch5.load_frame(symbol, ohlc_loader=loader, cache_dir=cache_dir)
        # ^IRX is a signal-only series that can print at or below zero: loaded WITHOUT
        # the positivity gate meant for traded prices (never for a panel name).
        irx_frame, irx_source = batch5.load_frame(FINANCING_SERIES, ohlc_loader=loader, cache_dir=cache_dir,
                                                  require_positive=False)
    except RuntimeError as exc:
        return {"family_id": family_id, "reason_code": "ohlc_load_failed", "reason": str(exc)}
    all_sources = {**sources, FINANCING_SERIES: irx_source}
    cached_symbols = [s for s, source in all_sources.items() if source == "cache"]
    if cached_symbols and not dry_run:
        return {"family_id": family_id, "reason_code": "stale_ohlc_cache", "cached_symbols": cached_symbols,
                "reason": ("a real run must score freshly fetched histories, but "
                           f"{', '.join(cached_symbols)} came from an OHLC cache left under run id "
                           f"{run_id!r} by an earlier run; use a fresh --run-id (or empty that run's "
                           "cache dir) and run again")}
    manifest = {}
    for symbol in market_symbols + [FINANCING_SERIES]:
        path = os.path.join(cache_dir, batch5._cache_name(symbol))
        frame = irx_frame if symbol == FINANCING_SERIES else frames[symbol]
        manifest[symbol] = {"source": all_sources[symbol], "rows": int(len(frame)),
                            "first_bar": frame.index[0].strftime("%Y-%m-%d"),
                            "last_bar": frame.index[-1].strftime("%Y-%m-%d"),
                            "sha256": _sha256_of(path)}

    # ---- the frozen panel (read-only), hygiene, as-of cutoff
    panel_frames, panel_excluded = load_panel(sorted(costs), panel_loader or _default_panel_loader(panel_cache_dir))
    if not panel_frames:
        return {"family_id": family_id, "reason_code": "insufficient_panel_coverage",
                "reason": "no panel history could be loaded", "load_excluded": panel_excluded}
    as_of = _panel_as_of(panel_frames)
    cut = {s: f.loc[:as_of] for s, f in frames.items()}
    spy_frame = cut[MARKET_BENCHMARK]
    try:
        mod.validate_market_calendar(spy_frame, {s: cut[s] for s in mod.FACTOR_SYMBOLS})
    except mod.CalendarCoverageError as exc:
        # A data-calendar gap is refused, never trimmed or patched: a clean
        # reason-coded refusal that names the dates (nothing is written). Checked
        # BEFORE panel hygiene: hygiene reads SPY's calendar as the reference, so a
        # vendor hole in SPY would otherwise look like a defect in every panel name.
        return _calendar_refusal(family_id, exc)
    hygiene = mod.panel_hygiene(panel_frames, calendar_index=spy_frame.index, first_needed=first_month_end)
    kept, hygiene_excluded = hygiene["kept"], hygiene["excluded"]
    n_declared = len(traded)
    if len(kept) < batch5.MIN_POOL_COVERAGE * n_declared:
        return {"family_id": family_id, "reason_code": "insufficient_panel_coverage",
                "reason": (f"only {len(kept)} of {n_declared} panel names survive the quote, load and "
                           f"hygiene checks (minimum {batch5.MIN_POOL_COVERAGE:.0%})"),
                "hygiene_excluded": dict(list(hygiene_excluded.items())[:_MAX_LISTED])}

    try:
        prepared = mod.prepare(spy_frame, {s: cut[s] for s in mod.FACTOR_SYMBOLS}, kept,
                               first_month_end=first_month_end)
    except mod.CalendarCoverageError as exc:
        return _calendar_refusal(family_id, exc)
    if not prepared["windows"]:
        return {"family_id": family_id, "reason_code": "no_windows",
                "reason": "no complete monthly window with enough eligible names in this data"}
    cost_bps = {s: costs[s]["cost_bps"] for s in kept}
    try:
        curve = mod.build_curve(prepared, kept, irx_frame, cost_bps_by_symbol=cost_bps,
                                spy_entry_cost_bps=benchmark_entry_cost_bps)
    except cohort_overlay.CohortDataError as exc:
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc),
                "tickers": exc.tickers[:_MAX_LISTED], "n_tickers": len(exc.tickers),
                "dates": exc.dates[:_MAX_LISTED]}
    except mod.BillPrintStaleError as exc:
        # A hole in the bill series is refused, never carried across silently (nothing is written).
        return {"family_id": family_id, "reason_code": exc.reason_code, "reason": str(exc),
                "max_age_days": exc.max_age_days, "session": exc.session,
                "n_sessions_over_limit": exc.n_sessions_over_limit}

    account_type = entry["account_types"][0]
    eval_kwargs = dict(benchmark_frame=spy_frame, benchmark_entry_cost_bps=benchmark_entry_cost_bps,
                       account_type=account_type)
    result = abi1_driver.evaluate_family(curve, tax_policy=TAX_POLICY, **eval_kwargs)
    benchmark_daily = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        spy_frame, entry_cost_bps=benchmark_entry_cost_bps)
    # Risk-basis rule: the ledger note states the risk basis and that the excess is
    # the overlay's own; refuse to record anything either claim is false for.
    overlay_risk_report.require_overlay_basis(result, family_id=family_id)
    overlay_risk_report.require_excess_is_overlay_only(
        result["account_daily"], benchmark_daily, curve["overlay_mask"], family_id=family_id)
    # The shared guard above reads only the sessions the overlay is OFF, which on this
    # contiguous monthly schedule is ONE session (the first decision session). Verify the identity
    # account = benchmark + overlay (and core = benchmark) on EVERY session so the claim does not rest on one.
    cohort_overlay.require_account_is_benchmark_plus_overlay(curve, benchmark_daily, family_id=family_id)

    # ---- report-only counterfactuals and controls (never scored, never recorded)
    untaxed_result = abi1_driver.evaluate_family(curve, tax_policy="none", **eval_kwargs)
    control_curve = mod.build_control_curve(prepared, kept, cost_bps_by_symbol=cost_bps)
    control = _control_summary(
        abi1_driver.evaluate_family(control_curve, tax_policy=CONTROL_TAX_POLICY, **eval_kwargs),
        scoring="ABI-1 daily account-relative, tax_policy=none, whole-account risk basis; a RESEARCH "
                "CONTROL (zero-signal equal-weight basket of the eligible names on the SAME monthly windows "
                "as the overlay, with costs), report-only")
    daily_control = _control_summary(
        abi1_driver.evaluate_family(mod.build_daily_control_curve(prepared, kept),
                                    tax_policy=CONTROL_TAX_POLICY, **eval_kwargs),
        scoring="ABI-1 daily account-relative, tax_policy=none, whole-account risk basis; a RESEARCH "
                "CONTROL (zero-signal equal-weight basket of the eligible names REBALANCED EVERY SESSION, "
                "fully invested, gross of costs: an upper bound), report-only")
    placebo = _placebo_summary(mod.run_placebo(
        prepared, kept, irx_frame, cost_bps_by_symbol=cost_bps,
        spy_entry_cost_bps=benchmark_entry_cost_bps, n_draws=int(placebo_draws)))

    diag = overlay_diagnostics(
        curve, result, untaxed_result=untaxed_result, benchmark_daily=benchmark_daily,
        hygiene_excluded=hygiene_excluded, quote_excluded=quote_excluded, panel_excluded=panel_excluded,
        n_traded=n_declared, n_used=len(kept), cost_bps=cost_bps,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps)

    risk_note = overlay_risk_report.risk_layer_note(diag["risk_layer"], ordering_clause=ORDERING_CLAUSE)
    reading = reading_rule_text(placebo, control, daily_control)
    # The recorded stage is unconditional; the placebo gates THIS field, and a
    # survived stage that did not clear it is flagged in the FIRST sentence of the driver-supplied note (the
    # shared writer puts its own fixed metadata prefix ahead of it, which is not editable from here).
    control_holds = control_condition_holds(placebo)
    passed = pre_registered_pass(result["verdict"], control_holds)
    clause = pre_registered_clause(result["verdict"], control_holds)
    lead = ledger_note_lead(result["verdict"], control_holds)
    account_index = curve["strategy_daily"].index
    extra_note = lead + (
        f"cost model: full round trip of each name's quoted spread on BOTH legs, charged on the exit day "
        f"({batch5.quotes_cost_source(quotes_payload)}); mean {diag['cost_bps_round_trip']['mean']} bps over "
        f"{diag['cost_bps_round_trip']['n']} priced names, and 2026 quoted spreads are applied to every year "
        "(direction unmeasured; plausibly understating early-year costs); "
        "design=benchmark-plus-overlay account (SPY core fully invested at all times PLUS an overlay of "
        f"notional x={diag['overlay_notional_fraction']} of equity: long the equal-weight top-{diag['n_select']} "
        "residual-momentum cohort, short the equal-weight basket of ALL eligible names, the short leg being a "
        "RESEARCH CONTROL and not an investable account); the score of this row does not certify an investable "
        "long-only strategy; monthly windows: decision at the close of the last exchange session of month M, "
        "entry at the OPEN of the first session of M+1 (the overnight gap into it is not earned), exit at the "
        "close of the last session of M+1, dates from the exchange session calendar cross-checked against SPY "
        "(exchange_calendars XNYS actual sessions), never read off a ticker's own bars; signal = 36-month "
        "three-factor regression on the ETF proxies SPY, IWM-SPY and IVE-IVW (raw monthly returns, month-end "
        "adjusted closes with no forward fill), score = sum of residuals over months M-11..M-1 (M skipped) over "
        "their sample standard deviation; shares fixed at entry, sized on the account equity at the decision close; "
        "financing = the LONG leg's notional borrowed at the ^IRX 13-week bill coupon-equivalent yield of the last "
        "close strictly before each session, actual calendar days over 365, mean "
        f"{diag['mean_financing_rate_annual_on_overlay_sessions']} on overlay sessions, maximum bill-print age on "
        f"any financed session {diag['bill_print_max_age_days']} calendar days (refused above "
        f"{mod.MAX_BILL_PRINT_AGE_DAYS}); NOT MODELLED, and each of these flatters the strategy: no short borrow "
        "fee, no margin spread over the bill rate (the pipeline has no margin-rate source and inventing one is not "
        "allowed), Reg-T margin calls and borrow availability, and the tax walk (ABI-1 realization netting) "
        "deducts financing interest against the overlay's gain; no short rebate (conservative) is credited "
        f"either; UNIVERSE: today's S&P 500 and Nasdaq-100 members (survivorship; {diag['panel']['used']} of "
        f"{diag['panel']['traded_declared']} declared names used, hygiene exclusions "
        f"{diag['panel']['hygiene_excluded_by_reason']}, quote exclusions {diag['panel']['n_quote_excluded']}, "
        f"load exclusions {diag['panel']['n_load_excluded']}; GOOGL, FOX, NWSA dropped as duplicate share classes, "
        "HUBB and MNST as documented vendor defects; the hygiene and quote exclusions are decided on "
        "full-sample or 2026 data and applied to all history, a data-quality rule and not a return-based "
        "one, applied equally to both legs); the cited evidence is a long-short decile result and a "
        "long-only cohort against a cap-weighted benchmark is untested in the sources; HML-proxy fidelity: "
        "IVE minus IVW is an imperfect proxy for the Ken French HML factor (as are the SMB and MKT "
        "proxies), so value exposure is only partly removed; "
        f"{risk_note}; {reading}; pre_registered_pass={passed} (True only when the ABI-1 verdict is survived "
        f"AND the placebo percentile bar is cleared; this run's ABI-1 verdict is {result['verdict']} and the "
        f"placebo clears the bar: {placebo['clears_bar']}); tax: netted_tax_factors dollar-attribution approximation (not uniformly "
        "conservative), measured ratio range "
        f"[{diag.get('tax_identity_ratio_min')}, {diag.get('tax_identity_ratio_max')}], whole tax effect on the "
        f"daily-excess Sharpe {diag.get('tax_effect_on_daily_excess_sharpe')}, heuristic sensitivity (not a rigorous "
        f"bound) {diag.get('tax_identity_sensitivity_on_sharpe')}; measured: {diag['n_windows']} windows "
        f"({diag['eligible_min']} to {diag['eligible_max']} eligible names, median {diag['eligible_median']}) over "
        f"{diag['overlay_sessions']} of {diag['account_sessions']} sessions, scored-excess OLS beta to SPY "
        f"{diag['scored_excess_beta_to_spy']}; account span {account_index[0]:%Y-%m-%d}..{account_index[-1]:%Y-%m-%d} "
        f"(first decision session to the last COMPLETE window's exit; the market frames were cut to the panel's "
        f"last session {as_of:%Y-%m-%d}; open window excluded, never closed early: "
        f"{diag['open_window_excluded']}; decision not started: {diag['not_started']}); "
        "PRE-REGISTRATION LIMITS: every constant was fixed ex-ante and none was tuned (the regression, "
        "formation and skip lengths and the score are from the literature; N, the overlay notional, the seed, "
        "the draw count and the 0.95 bar are design choices) before this pipeline computed any return of "
        "this rule; the zero-signal equal-weight controls are measurements of the universe, not of the "
        "signal, report-only and outside the trial "
        f"ledger; price data: SPY {sources['SPY']}, IWM {sources['IWM']}, IVE {sources['IVE']}, IVW "
        f"{sources['IVW']}, ^IRX {irx_source}, factor-ETF freeze sha256 "
        + ", ".join(f"{s}={m['sha256'][:12]}" for s, m in manifest.items()))

    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        stage = abi1_driver.record_abi1_outcome(
            ledger_path, family_id, run_id, tested_date, result, catalog_entry=entry,
            instrument_set=sorted(kept) + [MARKET_BENCHMARK] + list(mod.FACTOR_SYMBOLS),
            cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK, tax_policy=TAX_POLICY,
            ticker_label=None, extra_note=extra_note, signal_source_path=signal_source_path,
            signal_source_sha256=signal_source_sha256)

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    return {
        "family_id": family_id, "shape": mod.SHAPE, "n_panel_names": len(kept),
        "verdict": result["verdict"], "stage": stage,
        # The recorded stage is unconditional; the placebo gates THESE fields
        "pre_registered_pass": passed, "pre_registered_clause": clause,
        "dsr_value": layer1.get("dsr_value"), "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"), "n_trades": int(len(curve["trades_df"])),
        "account_start": account_index[0].strftime("%Y-%m-%d"),
        "account_end": account_index[-1].strftime("%Y-%m-%d"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "time_in_market": result.get("time_in_market"),
        "tax_policy": TAX_POLICY, "cash_convention": CASH_CONVENTION, "benchmark": MARKET_BENCHMARK,
        "cost_model_note": extra_note,
        "holdout_evaluated": holdout.get("evaluated"), "holdout_passed": holdout.get("passed"),
        "holdout_psr": holdout.get("psr"),
        "window": {"account_sessions": window.get("account_sessions"),
                   "scored_sessions": window.get("scored_sessions"),
                   "in_sample_sessions": window.get("in_sample_sessions"),
                   "holdout_sessions": window.get("holdout_sessions")},
        # provenance: "cache" means the history was re-used from an earlier run under
        # the SAME run_id. Only a dry run can get here with a cache hit.
        "ohlc_sources": all_sources, "factor_freeze": manifest,
        "last_bar_dates": {**{s: frames[s].index[-1].strftime("%Y-%m-%d") for s in market_symbols},
                           FINANCING_SERIES: irx_frame.index[-1].strftime("%Y-%m-%d")},
        "zero_signal_control": control, "zero_signal_daily_rebalanced_control": daily_control,
        "permutation_placebo": placebo,
        "reading_rule": reading, "overlay_diagnostics": diag,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", required=True,
                        help="Path to a dated quotes JSON file with bid/ask for SPY and the panel names "
                             "(must be a REAL captured book unless --dry-run is also passed, in which case "
                             "a SYNTHETIC placeholder file is expected).")
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
    parser.add_argument("--panel-cache-dir", default=PANEL_CACHE_DIR)
    args = parser.parse_args(argv)
    if args.tested_date is None:
        if not args.dry_run:
            parser.error("--tested-date is required for a real (non --dry-run) run: it is written into a "
                         "permanent ledger row and must never fall back to a stale default")
        args.tested_date = DEFAULT_TESTED_DATE
    if args.run_id is None:
        if not args.dry_run:
            parser.error("--run-id is required for a real (non --dry-run) run: it names the OHLC cache dir, "
                         "and a real run refuses history cached there by an earlier run (a dry run's bars), "
                         "so it must be a fresh id")
        args.run_id = DEFAULT_RUN_ID

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes, ledger_path=args.ledger,
        run_id=args.run_id, tested_date=args.tested_date, dry_run=args.dry_run,
        signal_source_path=_SIGNAL_SOURCE_PATH, signal_source_sha256=evidence.source_sha256(__file__),
        research_root=args.research_root, panel_cache_dir=args.panel_cache_dir)

    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    safe_result = batch5._json_safe(result)
    if safe_result.get("pre_registered_clause"):
        print(safe_result["pre_registered_clause"])      # the same sentence the note leads with; JSON stays the last line
    payload = {"schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
               "tested_date": args.tested_date, "dry_run": args.dry_run,
               "driver": _SIGNAL_SOURCE_PATH, "result": safe_result}
    output_path = os.path.join(
        run_dir, "residual-momentum-overlay-dry-run.json" if args.dry_run
        else "residual-momentum-overlay-results.json")
    batch3._atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
