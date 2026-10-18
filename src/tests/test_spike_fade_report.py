"""Tests for src/mining/spike_fade_report.py -- the REPORT.md writer of the
range-bound spike-fade study, Stage 1 (the pre-registered study; the
pre-registration document is not included in this repository).

Tests are hand-derived on small synthetic inputs. The
writer is a set of PURE
functions of the result dict (the same dict the driver saves as results.json),
so every test below feeds a hand-built synthetic dict. No network, no files,
no real prices.

What is pinned:
  * the plain-language block comes FIRST (3 to 5 bullets, no jargon) and
    states what was tested, the headline p in words, whether the tradable fund
    shows an excess, the best-looking rules with the luck caveat, and the
    recommendation;
  * every required report section (SECTION_TITLES) is present and in order;
  * the report never contains a dollar sign followed by a digit (money
    figures stay out of generated text) and the writer fails closed rather than emit one;
  * the power statement uses the calibration file when present and says
    calibration is pending when it is not;
  * every count (dropped, censored, undefined, not applicable, externally
    corrected) reaches the report;
  * a failed-closed run still produces a report that lists the missing series.
"""
import copy
import os
import re
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import spike_fade_report as rep  # noqa: E402

DOLLAR = chr(36)      # built at run time: this file must never contain the pattern itself
DOLLAR_DIGIT = re.compile(r"\$\s*\d")

# Words a reader without a statistics course would not know. The summary block
# must not contain them (a term that cannot be avoided is defined in the same
# sentence, which the wording tests check separately).
JARGON = ("sigma", "null", "t-stat", "z-score", "plateau", "winsor", "abi-1", "dsr",
          "percentile", "bootstrap", "quantile", "heteroskedastic", "vehicle test")


# ---------------------------------------------------------------------------
# A complete synthetic results dict (the schema the driver writes)
# ---------------------------------------------------------------------------

def _cell_row(rank, series="^VIX", side="up_fade", trigger="z20@2.0", entry="E0", H=21,
              tp="none", sl="1.5", **kw):
    row = {"rank": rank, "series": series, "side": side, "trigger": trigger, "entry": entry,
           "H": H, "tp": tp, "sl": sl, "n": 14, "mean_x": 0.31, "t": 1.9,
           "plateau_score": 0.27, "n_eligible_neighbors": 11, "hit_rate": 0.64,
           "mean_pnl": 0.031, "break_even_bps": 310.0, "ci_lo": -0.05, "ci_hi": 0.62,
           "null_q90": 0.22, "stab_passed": True, "stab_a": True, "stab_b": True,
           "null_stab_pass_rate": 0.31, "null_stab_pass_rate_given_positive": 0.44}
    row.update(kw)
    return row


def _series_info(role="confirmatory", source="yahoo", symbol="X", lag=0, **kw):
    groups = []
    for trig in ("z20@2.0", "z20@2.5", "pct756", "pctexp"):
        for side in ("up_fade", "down_fade"):
            groups.append({"trigger": trig, "side": side, "n_flagged_days": 40,
                           "n_episodes": 12, "n_kept": 11, "dropped_censored": 1})
    diag = {}
    for trig in ("z20@2.0", "z20@2.5", "pct756", "pctexp"):
        diag[trig] = {"first_usable_date": "1991-01-15", "n_undefined": 300, "n_warmup": 273,
                      "n_undefined_after_first_usable": 27,
                      "up_fade": {"n_raw": 50, "n_flagged": 40,
                                  "first_flag_date": "1992-03-02", "rolloff_share": 0.2},
                      "down_fade": {"n_raw": 44, "n_flagged": 30,
                                    "first_flag_date": "1993-05-04", "rolloff_share": 0.1}}
    info = {
        "role": role, "source": source, "symbol": symbol, "extra_lag": lag,
        "table_first_date": "1990-01-02", "first_date": "1990-01-02", "last_date": "2023-06-30",
        "n_rows": 8500,
        "counts": {"rows": 8500, "post_cutoff_dropped": 800, "pre_start_dropped": 0,
                   "missing_rows_dropped": 12, "external_corrections": 0,
                   "reference_unchecked": 0, "nonpositive": 0, "missing": 0},
        "external_reference": None, "external_correction_dates": [], "nonpositive_dates": [],
        "largest_returns": [{"date": "2020-03-16", "prev_date": "2020-03-13",
                             "log_return": 0.29, "abs_log_return": 0.29}] * 3,
        "ar1": {"phi": 0.997, "half_life_sessions": 230.0, "n_pairs": 8400, "note": ""},
        "trigger_diagnostics": diag, "groups": groups,
        "cell_status_totals": {"traded": 5000, "undefined_sigma": 3, "sigma_touch_nonpositive": 0,
                               "no_e1_confirmation": 210, "touch_nonpositive": 0,
                               "undefined_median": 0, "median_touch_nonpositive": 0,
                               "censored_data_end": 0, "touch_missing": 0,
                               "tp_not_applicable": 40},
        "affected_dates": [],
    }
    info.update(kw)
    return info


def _vehicle_result(**kw):
    r = {"n_anchors_input": 16, "dropped_no_session": 1, "dropped_past_end": 0,
         "dropped_short_history": 2, "dropped_censored": 1, "dropped_duplicate": 0,
         "n_anchors_kept": 12, "n_kept_not_traded": 0, "n": 12, "mean_pnl_pp": 2.4,
         "hit_rate": 0.58, "null_mean_pnl_pp": 1.7, "null_reps_used": 500,
         "null_reps_failed": 0, "excess_pp": 0.7, "excess_sigma": 0.11,
         "null_percentile": 0.58, "unconditional_mean_pnl_pp": 1.6,
         "unconditional_mean_daily_return_pp": 0.02, "window_start": "2012-10-01",
         "window_end": "2023-06-30", "era": None, "underpowered_on_vehicle": False,
         "contaminated": True, "B": 500, "seed": 7101, "trades": []}
    r.update(kw)
    return r


def _abi1(status="computed", **kw):
    acct = {"n_sessions": 2700, "n_trades": 12, "excess_sharpe_annualized": 0.31,
            "in_sample_bar": 1.02, "meets_in_sample_bar": False, "dsr_bar_annualized": 1.05,
            "max_dd_account": -0.30, "max_dd_spy": -0.34, "absolute_cap": -0.35,
            "dd_relative_passed": True, "dd_absolute_passed": True, "dd_passed": True,
            "terminal_wealth_account": 4.1, "terminal_wealth_spy": 3.9,
            "below_min_sessions": False,
            "risk_refusals": {"all_refusals_passed": True}}
    out = {"status": status, "reason": None, "series": "^VIX", "side": "up_fade",
           "trigger": "z20@2.0", "cell": "E0|21|none|none", "vehicle": "SVXY",
           "position": "long", "n_signals": 14, "n_skipped_overlap": 0,
           "n_skipped_no_entry": 0, "n_skipped_undefined": 0,
           "full": acct, "ex_worst": dict(acct, excess_sharpe_annualized=0.1),
           "conventions": "signal at a bar's close; entry at the next bar's open",
           "overlay_notional": 0.25, "cost_bps": 0.0, "trades": [], "dropped_trades": []}
    out.update(kw)
    return out


def _put(series="^VIX", role="tstar", **kw):
    ks = {"1.0": 0.02, "1.25": 0.0, "1.5": -0.01, "2.0": -0.03}
    res = {"100": {"break_even_premium": 0.05, "avg_pnl": ks, "avg_premium": ks,
                   "break_even_k": 1.3, "break_even_k_status": "ok"}}
    out = {"series": series, "side": "up_fade", "trigger": "z20@2.0",
           "cell": "E0|21|none|none", "role": role, "H": 21, "status": "computed",
           "reason": None, "n_used": 12, "n_dropped": 0, "observed": res, "baseline": res,
           "comparison": {"100": {"observed_break_even_k": 1.3, "baseline_break_even_k": 1.1,
                                  "exceeds_baseline": True, "margin_over_baseline": 0.2,
                                  "margin_over_market_median": 0.1,
                                  "margin_over_market_p90": -0.2,
                                  "observed_status": "ok", "baseline_status": "ok"}},
           "market_k_proxy": "^VVIX",
           "market_k_summary": {"n": 10, "median": 1.2, "p10": 0.9, "p90": 1.5},
           "market_k_vix_over_spy": {"n": 10, "median": 1.4, "p10": 1.0, "p90": 1.9}}
    out.update(kw)
    return out


def make_results(**over):
    """A complete, plausible results dict for a completed run (synthetic)."""
    cands = ("^VIX", "WTI spot", "Brent spot", "Henry Hub spot", "GC=F", "HG=F", "DX-Y.NYB")
    series = {name: _series_info(symbol=name, source=("fred" if "spot" in name else "yahoo"),
                                 lag=(1 if "spot" in name else 0))
              for name in cands}
    for name in ("^OVX", "^VVIX", "^GVZ", "^MOVE", "^TNX", "JPY=X"):
        series[name] = _series_info(role="exploratory", symbol=name)
    series["JPY=X"]["external_reference"] = "DEXJPUS"
    series["JPY=X"]["counts"]["external_corrections"] = 2
    series["JPY=X"]["external_correction_dates"] = ["1998-10-08", "1999-01-04"]
    top = [_cell_row(i + 1, H=h) for i, h in enumerate((21, 42, 63, 10))]
    res = {
        "schema": "spike-fade-results-v1", "status": "completed",
        "spec": {"file": "prereg-spec.md",
                 "version": 3},
        "cutoff": "2023-06-30",
        "params": {"B": 500, "B_map": 200,
                   "seeds": {"n1": 7101, "n2": 7202, "map": 7303,
                             "cluster_bootstrap": 7404, "vehicle_null": 7101,
                             "put_baseline": 7303},
                   "universe": list(series), "confirmatory_family": list(cands),
                   "skip_fetch": False},
        "data": {"files": {"^VIX": {"kind": "trigger", "source": "yahoo", "symbol": "^VIX",
                                    "sha256": "a" * 64, "rows_within_cutoff": 8739,
                                    "first_date": "1990-01-02", "last_date": "2023-06-30"}},
                 "labels": {"fred": "current revised history",
                            "yahoo": "vendor-adjusted history as retrieved on the run date"}},
        "series": series,
        "roll_gap": [{"series": "GC=F", "etf": "GLD", "n": 4500, "corr": 0.93,
                      "max_abs_diff": 0.02, "max_abs_diff_date": "2008-10-13",
                      "sd_daily": 0.011, "ratio": 1.8, "flagged": False, "status": "ok"}],
        "vehicle_first_dates": {"SVXY": {"table": "2011-10-04", "observed": "2011-10-04",
                                         "matches": True}},
        "confirmatory": {
            "tstar": 2.31, "tstar_key": "^VIX|up_fade|z20@2.0", "tstar_n": 14,
            "n_groups": 56, "n_tested": 40, "min_n": 8, "B": 500, "n1_seed": 7101,
            "n2_seed": 7202, "p_n1": 0.062, "p_headline": 0.062,
            "headline_rule": "N1 alone (spec v3.1)",
            "decision": "confirmatory evidence that fading extremes beats same-side random "
                        "entries at the primary cell, in the named pair",
            "mc_se_n1": 0.011, "mc_se_headline": 0.011, "n_failed_n1": 0,
            "n1_null_max_quantiles": {"q50": 1.9, "q90": 2.5, "q95": 2.8},
            "n2_sensitivity": {
                "label": "sensitivity, not gating", "gating": False, "seed": 7202,
                "p_n2": 0.041, "mc_se_n2": 0.009, "available_for_tstar": True,
                "tstar_n2_covered": 2.31, "p_n2_covered": 0.041,
                "offsets": {"^VIX": 120}, "skipped": ["Brent spot"],
                "coverage": {"groups_tested": 40, "groups_covered": 30,
                             "series_covered": ["^VIX"], "tstar_covered": True},
                "null_max_quantiles": {"q50": 1.8, "q90": 2.4, "q95": 2.7}},
            "report_lines": ["T* = 2.3100 at ^VIX|up_fade|z20@2.0 (n = 14)."],
            "groups": [{"series": "^VIX", "side": "up_fade", "trigger": "z20@2.0",
                        "n": 14, "mean_x": 0.31, "t": 2.31, "tested": True,
                        "adj_p_headline": 0.062, "is_tstar": True}],
            "per_pair": {"^VIX|up_fade": {"t_max": 2.31, "argmax_key": "^VIX|up_fade|z20@2.0",
                                          "n_tested_groups": 4, "p_n1": 0.062,
                                          "p_n2_sensitivity": 0.041, "p_headline": 0.062}},
        },
        "family_table": [
            {"series": "^VIX", "side": "up_fade", "trigger": "z20@2.0", "n": 14,
             "mean_x": 0.31, "t": 2.31, "is_tstar": True, "adj_p_headline": 0.062,
             "vehicle": "SVXY", "vehicle_excess_pp": 0.7, "vehicle_null_percentile": 0.58,
             "vehicle_n": 12, "vehicle_status": "run",
             "best_plateau_cell": "E0 | H 21 | TP none | SL 1.5 | z20@2.0",
             "best_plateau_score": 0.27},
            {"series": "GC=F", "side": "down_fade", "trigger": "pct756", "n": 20,
             "mean_x": 0.01, "t": 0.2, "is_tstar": False, "adj_p_headline": 0.99,
             "vehicle": "GLD", "vehicle_excess_pp": None, "vehicle_null_percentile": None,
             "vehicle_n": None, "vehicle_status": "not_run",
             "best_plateau_cell": None, "best_plateau_score": None},
        ],
        "map": {"n_series": 13, "n_groups": 104, "n_cells": 7488, "n_eligible_cells": 4100,
                "n_with_plateau": 3900, "top_overall": top,
                "top_per_pair": {"^VIX|up_fade": top[:2]},
                "null_column_note": "context only"},
        "vehicle": {"scope": [
            {"role": "tstar", "series": "^VIX", "side": "up_fade", "trigger": "z20@2.0",
             "cell": "E0|21|none|none", "vehicle": "SVXY", "position": "long", "run": True,
             "plateau_score": None, "result": _vehicle_result(), "reason": None},
            {"role": "plateau_top3", "series": "GC=F", "side": "down_fade",
             "trigger": "pct756", "cell": "E0|42|none|1.5", "vehicle": "GLD",
             "position": "long", "run": True, "plateau_score": 0.2,
             "result": _vehicle_result(n=9, excess_pp=-0.4, null_percentile=0.4,
                                       contaminated=False),
             "reason": None}],
            "candidate": {"is_candidate": False, "status": "null_percentile_below_min",
                          "reasons": ["null_percentile_below_min"], "headline_p": 0.062,
                          "underpowered_on_vehicle": False, "contaminated": True,
                          "contamination_note": "confirmatory on the discovery window only; "
                                                "the sealed window is mechanically sealed but "
                                                "partly known for this series"}},
        "put_screen": [_put()],
        "abi1": {"tstar": _abi1(), "top_plateau": _abi1(series="GC=F", side="down_fade")},
        "counts": {"totals": {"rows": 8500 * 13, "missing_rows_dropped": 156,
                              "pre_start_dropped": 0, "post_cutoff_dropped": 10400,
                              "nonpositive": 1, "external_corrections": 2,
                              "flagged_days": 5000, "episodes": 900, "kept": 880,
                              "dropped_censored": 20, "undefined_sigma": 39,
                              "tp_not_applicable": 520, "no_e1_confirmation": 2730}},
        "headline": {
            "p_headline": 0.062, "p_n1": 0.062,
            "n2_sensitivity": {"label": "sensitivity, not gating", "gating": False,
                               "p_n2": 0.041, "available_for_tstar": True,
                               "offsets": {"^VIX": 120}, "skipped": ["Brent spot"]},
            "decision": "confirmatory evidence that fading extremes beats same-side random "
                        "entries at the primary cell, in the named pair",
            "mc_se_headline": 0.011, "B": 500,
            "tstar": 2.31, "tstar_key": "^VIX|up_fade|z20@2.0", "tstar_n": 14,
            "tstar_series": "^VIX", "tstar_side": "up_fade", "tstar_trigger": "z20@2.0",
            "vehicle": {"ticker": "SVXY", "position": "long", "n": 12, "excess_pp": 0.7,
                        "excess_sigma": 0.11, "null_percentile": 0.58, "hit_rate": 0.58,
                        "mean_pnl_pp": 2.4, "unconditional_mean_pnl_pp": 1.6,
                        "underpowered": False, "reps": 500},
            "candidate": {"is_candidate": False, "status": "null_percentile_below_min",
                          "reasons": ["null_percentile_below_min"], "contaminated": True,
                          "contamination_note": "confirmatory on the discovery window only"},
            "top5_plateau": top[:4],
            "abi1_tstar": {"status": "computed", "excess_sharpe": 0.31, "bar": 1.02,
                           "meets": False, "dd_passed": True, "max_dd_account": -0.30,
                           "max_dd_spy": -0.34, "ex_worst_excess_sharpe": 0.1},
            "abi1_top_plateau": {"status": "computed", "excess_sharpe": 0.2, "bar": 1.02,
                                 "meets": False, "dd_passed": True, "max_dd_account": -0.3,
                                 "max_dd_spy": -0.34, "ex_worst_excess_sharpe": 0.05},
            "calibration_status": "complete"},
        "recommendation": {
            "register_stage2": False,
            "conditions": [
                {"id": "a", "text": "a candidate exists under the tradable-vehicle test", "met": False,
                 "detail": "vehicle excess over random entries is not clearly positive"},
                {"id": "b", "text": "ABI-1 transfer line at or above the in-sample bar and "
                                    "drawdown refusals passed", "met": False, "detail": ""},
                {"id": "c", "text": "the reader rules on how the search is counted",
                 "met": None, "detail": "a judgment call, not decidable by the study"},
                {"id": "d", "text": "the T* pair's series is not contaminated", "met": False,
                 "detail": "^VIX is marked contaminated"},
                {"id": "e", "text": "a market-hours quote sample exists", "met": None,
                 "detail": "not part of Stage 1"}],
            "reasons_not": ["the fund you could trade does not clearly beat random entries"]},
        "open_choices": ["example open choice"],
        "failure": None,
    }
    res.update(over)
    return res


def make_calibration():
    curve = {"0.25": {"power_headline": 0.17, "n_datasets": 60},
             "0.5": {"power_headline": 0.43, "n_datasets": 60},
             "1.0": {"power_headline": 0.97, "n_datasets": 60}}
    return {"finished": True, "args": {"datasets": 300, "power_datasets": 60, "B": 200,
                                       "n_series": 13, "seed": 7505},
            "size": {"n_datasets": 300, "p_headline": {"rate": 0.077, "in_window": True},
                     "p_n1": {"rate": 0.087, "in_window": True},
                     "p_n2": {"rate": 0.131, "in_window": True},
                     "p_n2_covered": {"rate": 0.04, "in_window": False}},
            "power": {"curve": curve, "min_detectable_effect_50": 0.56,
                      "min_detectable_effect_80": 0.84, "passes_required_power": True},
            "power_subset_first_7_series": {
                "curve": curve, "min_detectable_effect_50": 0.52,
                "min_detectable_effect_80": 0.83, "passes_required_power": True},
            "verdict": {"size_headline_in_window": True, "power_at_1_passes": True}}


def make_failed(**over):
    res = {"schema": "spike-fade-results-v1", "status": "failed_closed",
           "spec": {"file": "x", "version": 3}, "cutoff": "2023-06-30",
           "params": {"B": 500, "B_map": 200, "seeds": {}, "universe": ["^VIX", "GC=F"],
                      "confirmatory_family": ["^VIX", "GC=F"], "skip_fetch": False},
           "data": {"files": {}, "labels": {}},
           "failure": {"reason": "required data could not be obtained",
                       "missing": [{"name": "GLD", "kind": "vehicle", "source": "yahoo",
                                    "symbol": "GLD", "reason": "download failed twice"},
                                   {"name": "JPY=X", "kind": "trigger", "source": "yahoo",
                                    "symbol": "JPY=X",
                                    "reason": "requires the DEXJPUS external reference"}]}}
    res.update(over)
    return res


def _sections(text):
    return [m.group(1).strip() for m in re.finditer(r"^## (.+)$", text, flags=re.M)]


def _summary_block(text):
    m = re.search(r"^## Plain-language summary\n(.*?)(?=^## )", text, flags=re.M | re.S)
    assert m, "no plain-language summary section"
    return m.group(1)


# ---------------------------------------------------------------------------
# 1. Structure: the plain-language block is FIRST, all sections present
# ---------------------------------------------------------------------------

def test_plain_language_summary_is_the_first_section():
    text = rep.render_report(make_results(), calibration=make_calibration())
    assert text.startswith("# ")
    assert _sections(text)[0] == "Plain-language summary"


def test_summary_has_three_to_five_bullets_and_nothing_else_but_bullets():
    text = rep.render_report(make_results(), calibration=make_calibration())
    block = _summary_block(text)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    assert 3 <= len(lines) <= 5
    assert all(ln.startswith("- ") for ln in lines)


def test_summary_has_no_jargon_and_defines_p():
    text = rep.render_report(make_results(), calibration=make_calibration())
    block = _summary_block(text).lower()
    for word in JARGON:
        assert word not in block, f"jargon {word!r} in the plain-language block"
    # the one unavoidable term (p) is defined in the same sentence it is used
    assert "by luck" in block or "by chance" in block


def test_all_required_sections_present_and_in_order():
    text = rep.render_report(make_results(), calibration=make_calibration())
    got = _sections(text)
    want = list(rep.SECTION_TITLES)
    positions = [got.index(t) for t in want]        # raises if a section is missing
    assert positions == sorted(positions)
    assert want[0] == "Plain-language summary"
    assert want[1].startswith("Decision table")


def test_summary_states_the_five_required_things():
    text = rep.render_report(make_results(), calibration=make_calibration())
    block = _summary_block(text).lower()
    assert "2023-06-30" in block                        # what was tested and the seal
    assert "0.062" in block                             # the headline p, exactly
    assert "svxy" in block or "fund" in block           # the tradable fund
    assert "luck" in block                              # the exit rules are partly luck
    assert "recommend" in block                         # the recommendation


# ---------------------------------------------------------------------------
# 2. Wording follows the numbers
# ---------------------------------------------------------------------------

def test_headline_p_above_threshold_reads_inconclusive_not_no_edge():
    res = make_results()
    res["headline"].update(p_headline=0.31, decision="inconclusive at the study's power")
    res["confirmatory"]["p_headline"] = 0.31
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "0.310" in block
    assert "not convincing" in block or "inconclusive" in block
    assert "not proof" in block or "does not mean" in block or "not a finding of no" in block


def test_no_tested_group_is_explained_in_plain_words():
    res = make_results()
    res["headline"].update(p_headline=None, tstar=None, tstar_key=None, tstar_n=None,
                           tstar_series=None, tstar_side=None, tstar_trigger=None,
                           vehicle=None, abi1_tstar=None,
                           decision="no tested group")
    res["confirmatory"].update(p_headline=None, tstar=None, tstar_key=None, tstar_n=None)
    text = rep.render_report(res, calibration=make_calibration())
    block = _summary_block(text).lower()
    assert "enough" in block                            # not enough events for the test
    assert "not run" in text.lower() or "no group" in text.lower()


def test_vehicle_excess_wording_positive_and_negative():
    res = make_results()
    res["headline"]["vehicle"].update(excess_pp=-1.2, null_percentile=0.21)
    neg = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "no clear extra gain" in neg or "does not show" in neg
    res2 = make_results()
    res2["headline"]["vehicle"].update(excess_pp=2.5, null_percentile=0.93)
    pos = _summary_block(rep.render_report(res2, calibration=make_calibration())).lower()
    assert "extra gain" in pos


def test_underpowered_vehicle_is_said_plainly():
    res = make_results()
    res["headline"]["vehicle"].update(n=5, underpowered=True)
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "too few" in block or "fewer than 8" in block


def test_recommendation_do_not_register_when_not_a_candidate():
    block = _summary_block(rep.render_report(make_results(),
                                             calibration=make_calibration())).lower()
    assert "do not" in block and "slot" in block


def test_recommendation_conditional_language_when_a_candidate_exists():
    res = make_results()
    res["recommendation"]["register_stage2"] = True
    res["recommendation"]["reasons_not"] = []
    res["headline"]["candidate"].update(is_candidate=True, status="candidate", reasons=[])
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "consider" in block
    assert "owner" in block or "still" in block         # unmet conditions remain


def test_top_exit_rule_is_described_in_words_with_the_luck_caveat():
    res = make_results()
    res["headline"]["top5_plateau"][0].update(H=42, tp="half", sl="3.0", entry="E1")
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "42 trading days" in block
    assert "luck" in block


# ---------------------------------------------------------------------------
# 2b. Headline p and the N2 sensitivity are SEPARATE fields
# ---------------------------------------------------------------------------

def test_headline_p_and_n2_sensitivity_are_separate_fields():
    sec = _section_text(rep.render_report(make_results(), calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    low = sec.lower()
    assert "Headline p = 0.062" in sec
    assert "sensitivity, not gating" in low
    assert "0.041" in sec                                 # the N2 sensitivity p is shown
    assert "^VIX 120" in sec                              # feasible-offset counts per series
    assert "Brent spot" in sec                            # a skipped series is named


def test_n2_like_for_like_size_from_the_calibration_is_printed_beside_the_headline():
    sec = _section_text(rep.render_report(make_results(), calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    assert "like-for-like size" in sec.lower() and "0.040" in sec
    sec_none = _section_text(rep.render_report(make_results(), calibration=None),
                             rep.SECTION_TITLES[1])
    assert "like-for-like size" in sec_none.lower() and "pending" in sec_none.lower()


def test_changing_only_the_n2_field_never_changes_the_headline_text():
    res = make_results()
    res["confirmatory"]["n2_sensitivity"]["p_n2"] = 0.5
    res["headline"]["n2_sensitivity"]["p_n2"] = 0.5
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    assert "Headline p = 0.062" in sec
    assert "0.500" in sec


def test_n2_unavailable_is_stated_as_unavailable_not_as_a_number():
    res = make_results()
    res["confirmatory"]["n2_sensitivity"].update(
        p_n2=None, mc_se_n2=None, p_n2_covered=None, tstar_n2_covered=None,
        available_for_tstar=False, offsets={}, skipped=["^VIX", "Brent spot"],
        coverage={"groups_tested": 40, "groups_covered": 0, "series_covered": [],
                  "tstar_covered": False})
    res["headline"]["n2_sensitivity"].update(p_n2=None, available_for_tstar=False)
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    assert "Headline p = 0.062" in sec
    assert "not available" in sec.lower()
    assert "sensitivity, not gating" in sec.lower()


def test_an_older_flat_n2_layout_still_renders_as_a_separate_sensitivity():
    res = make_results()
    n2 = res["confirmatory"].pop("n2_sensitivity")
    res["confirmatory"].update(p_n2=n2["p_n2"], mc_se_n2=n2["mc_se_n2"],
                               n2_offsets=n2["offsets"], n2_skipped=n2["skipped"],
                               n2_coverage=n2["coverage"], p_n2_covered=n2["p_n2_covered"])
    res["headline"].pop("n2_sensitivity")
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    assert "Headline p = 0.062" in sec and "0.041" in sec
    assert "sensitivity, not gating" in sec.lower()


# ---------------------------------------------------------------------------
# 3. The dollar guard
# ---------------------------------------------------------------------------

def test_report_has_no_dollar_sign_followed_by_a_digit():
    text = rep.render_report(make_results(), calibration=make_calibration(),
                             meta={"command": "python x", "python": "3.13"},
                             manifest={"^VIX": {"sha256": "a" * 64,
                                                "retrieved_utc": "2026-09-23T20:00:00Z"}})
    assert not DOLLAR_DIGIT.search(text)


def test_writer_fails_closed_on_an_injected_dollar_digit():
    res = make_results()
    res["open_choices"] = [f"costs assumed {DOLLAR}5 per trade"]
    with pytest.raises(ValueError):
        rep.render_report(res, calibration=make_calibration())


def test_assert_no_dollar_digit_helper():
    rep.assert_no_dollar_digit("a $ sign alone is fine, and so is 5 " "dollars")
    with pytest.raises(ValueError):
        rep.assert_no_dollar_digit(f"price {DOLLAR}12")
    with pytest.raises(ValueError):
        rep.assert_no_dollar_digit(f"price {DOLLAR} 12")


# ---------------------------------------------------------------------------
# 4. Power statement
# ---------------------------------------------------------------------------

def test_power_statement_uses_the_calibration_numbers():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "Power statement")
    assert "0.56" in sec and "0.84" in sec              # 50 and 80 percent effects
    assert "0.52" in sec or "0.83" in sec               # the 7-series sub-family
    assert "pending" not in sec.lower()
    assert "0.25" in sec and "0.5" in sec and "1.0" in sec   # the power curve rows


def test_power_statement_says_pending_without_a_calibration_file():
    text = rep.render_report(make_results(), calibration=None)
    sec = _section_text(text, "Power statement")
    assert "pending" in sec.lower()
    block = _summary_block(text)
    assert not DOLLAR_DIGIT.search(block)


def _section_text(text, title):
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    assert m, f"missing section {title}"
    return m.group(1)


# ---------------------------------------------------------------------------
# 5. Section contents
# ---------------------------------------------------------------------------

def test_decision_table_lists_every_family_pair_and_says_vehicle_not_run_where_not_run():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, rep.SECTION_TITLES[1])
    assert "^VIX" in sec and "GC=F" in sec
    assert "vehicle test not run" in sec
    assert "SVXY" in sec
    assert "recommend" in sec.lower()                    # the section ends with the recommendation


def test_exploratory_map_labels_the_null_column_context_only():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "Exploratory map")
    assert "context only" in sec.lower()
    assert "partly luck" in sec.lower() or "partly by luck" in sec.lower()
    assert "top 30" in sec.lower()


def test_vehicle_section_reports_excess_percentile_n_hit_rate_and_unconditional_return():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "Tradable-vehicle results")
    for needle in ("SVXY", "0.7", "0.58", "12", "unconditional", "candidate"):
        assert needle in sec
    assert "not a candidate" in sec.lower() or "no candidate" in sec.lower()
    assert "contaminated" in sec.lower()


def test_put_screen_section_states_it_is_a_screen_not_an_options_backtest():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "Put-affordability screen")
    assert "no historical option price" in sec.lower()
    assert "^VVIX" in sec
    assert "break-even" in sec.lower()


def test_put_screen_not_applicable_is_listed_with_its_reason():
    res = make_results()
    res["put_screen"] = [_put(status="not_applicable",
                              reason="T* side is down_fade: a put is bought after an up-spike")]
    text = rep.render_report(res, calibration=make_calibration())
    assert "a put is bought after an up-spike" in _section_text(text, "Put-affordability screen")


def test_abi1_section_reports_sharpe_bar_drawdowns_and_ex_worst():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "ABI-1 transfer line")
    for needle in ("0.31", "1.02", "-0.30", "-0.34", "0.35", "ex-worst", "re-derive"):
        assert needle in sec
    assert "descriptive" in sec.lower()


def test_abi1_not_computed_prints_the_reason():
    res = make_results()
    res["abi1"]["tstar"] = _abi1(status="not_computed",
                                 reason="vehicle_frame: prices must be finite and positive")
    text = rep.render_report(res, calibration=make_calibration())
    assert "prices must be finite and positive" in _section_text(text, "ABI-1 transfer line")


def test_counts_section_carries_every_kind_of_count_and_every_external_date():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "Data, counts and checks")
    for needle in ("dropped", "censored", "undefined", "not applicable", "externally corrected",
                   "1998-10-08", "1999-01-04", "roll-off", "half-life", "GLD"):
        assert needle.lower() in sec.lower(), needle
    # the 20 largest one-session moves are listed per series
    assert "2020-03-16" in sec


def test_contamination_and_limits_are_restated_in_the_report():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "Contamination and limits")
    for needle in ("^VIX", "WTI", "Brent", "discovery window only", "12 independent episodes",
                   "does not prove"):
        assert needle.lower() in sec.lower(), needle


def test_reproduction_section_has_seeds_digests_and_the_command():
    meta = {"command": "python scripts/run_spike_fade_study.py --B 500 --B-map 200",
            "command_to_reuse_cached_data":
                "python scripts/run_spike_fade_study.py --B 500 --B-map 200 --skip-fetch",
            "python": "3.13.5", "started_utc": "2026-09-24T00:00:00Z",
            "finished_utc": "2026-09-24T01:00:00Z", "workers": 4}
    manifest = {"^VIX": {"sha256": "a" * 64, "retrieved_utc": "2026-09-23T20:00:00Z",
                         "file": "yahoo__VIX.csv", "rows": 9000}}
    text = rep.render_report(make_results(), calibration=make_calibration(), meta=meta,
                             manifest=manifest)
    sec = _section_text(text, "Seeds, data digests and reproduction")
    for needle in ("7101", "7202", "7303", "7404", "a" * 64, "2026-09-23T20:00:00Z",
                   "python scripts/run_spike_fade_study.py --B 500 --B-map 200",
                   "current revised history"):
        assert needle in sec, needle
    assert "bulk" in sec.lower()                          # the raw data folder is not tracked
    assert "rows within the cutoff" in sec.lower()
    assert "--B-map 200 --skip-fetch" in sec              # the cached-data command is shown too
    assert "command_to_reuse_cached_data" not in sec      # ... as prose, not as a raw key


def test_open_choices_are_printed_not_hidden():
    res = make_results()
    res["open_choices"] = ["a choice the spec left open, made in the most literal way"]
    text = rep.render_report(res, calibration=make_calibration())
    assert "a choice the spec left open" in text


# ---------------------------------------------------------------------------
# 6. Fail closed: the report still lists the missing series
# ---------------------------------------------------------------------------

def test_failed_closed_report_lists_missing_series_and_says_not_run():
    text = rep.render_report(make_failed())
    assert _sections(text)[0] == "Plain-language summary"
    assert "GLD" in text and "JPY=X" in text
    assert "DEXJPUS" in text
    block = _summary_block(text).lower()
    assert "did not run" in block or "not run" in block
    assert not DOLLAR_DIGIT.search(text)


def test_failed_closed_report_has_a_missing_series_section():
    text = rep.render_report(make_failed())
    assert "Missing required series" in _sections(text)


def test_report_writer_does_not_mutate_its_input():
    res = make_results()
    snap = copy.deepcopy(res)
    rep.render_report(res, calibration=make_calibration())
    assert res == snap


def test_trigger_words_follow_the_side_of_the_trade():
    up = rep.trigger_words("pct756", "up_fade")
    down = rep.trigger_words("pct756", "down_fade")
    assert "top 5 percent" in up and "bottom" not in up
    assert "bottom 5 percent" in down and "top" not in down
    assert "jump" in rep.trigger_words("z20@2.0", "up_fade")
    assert "drop" in rep.trigger_words("z20@2.5", "down_fade")
    assert "all earlier history" in rep.trigger_words("pctexp", "up_fade")
    res = make_results()
    res["headline"]["top5_plateau"][0].update(trigger="pct756", side="down_fade")
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "bottom 5 percent" in block


def test_vehicle_results_do_not_print_a_bare_seed_number():
    text = rep.render_report(make_results(), calibration=make_calibration())
    sec = _section_text(text, "Tradable-vehicle results")
    assert "seed 7101" not in sec


# ---------------------------------------------------------------------------
# 7. Honesty of the plain-language block and completeness
# ---------------------------------------------------------------------------

def test_first_bullet_admits_the_sealed_window_is_not_fully_unseen():
    block = _summary_block(rep.render_report(make_results(),
                                             calibration=make_calibration())).lower()
    assert "not fully unseen" in block or "already knew" in block
    assert "limits" in block


def test_p_is_defined_against_random_dates_not_against_nothing_happening():
    block = _summary_block(rep.render_report(make_results(),
                                             calibration=make_calibration())).lower()
    assert "meant nothing" not in block
    assert "random" in block and "by luck" in block


def test_vehicle_bullet_says_so_when_the_comparison_could_not_be_run():
    res = make_results()
    res["headline"]["vehicle"].update(excess_pp=None, null_percentile=None)
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "could not be run" in block
    assert "no clear extra gain" not in block and "n/a percentage" not in block


def test_best_rule_bullet_states_how_many_trades_it_rests_on_and_its_steadiness():
    res = make_results()
    res["headline"]["top5_plateau"][0].update(n=9, stab_passed=False)
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "9 trades" in block or "9 episodes" in block
    assert "did not pass" in block
    res2 = make_results()
    res2["headline"]["top5_plateau"][0].update(n=14, stab_passed=True)
    block2 = _summary_block(rep.render_report(res2, calibration=make_calibration())).lower()
    assert "passed" in block2 and "14 " in block2


def test_decision_table_says_which_cell_the_vehicle_result_is_for():
    res = make_results()
    res["family_table"][0]["vehicle_cell"] = "E0|42|none|1.5"
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    assert "E0 | 42 | none | 1.5" in sec.replace(chr(92) + "|", "|")     # md_table escapes the pipes


def test_every_affected_date_is_listed_in_the_report_not_just_forty():
    res = make_results()
    dates = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(60)]
    res["series"]["^VIX"]["affected_dates"] = [
        {"date": d, "statuses": ["undefined_sigma"], "n_cells": 3} for d in dates]
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        "Data, counts and checks")
    for d in dates:
        assert d in sec, d


def test_the_power_statement_prints_effect_figures_only_from_a_calibration_file():
    with_cal = _section_text(rep.render_report(make_results(), calibration=make_calibration()),
                             "Power statement")
    without = _section_text(rep.render_report(make_results(), calibration=None),
                            "Power statement")
    assert "0.56" in with_cal and "0.84" in with_cal
    assert "0.56" not in without and "0.84" not in without


def test_stage_failures_are_flagged_in_the_summary_and_listed():
    res = make_results()
    res["stage_failures"] = [{"stage": "put screen", "item": "^VIX up_fade",
                              "reason": "ValueError: boom"}]
    text = rep.render_report(res, calibration=make_calibration())
    assert "incomplete" in _summary_block(text).lower()
    counts = _section_text(text, "Data, counts and checks")
    assert "put screen" in counts and "boom" in counts


def test_a_run_that_is_not_the_preregistered_configuration_says_so_first():
    res = make_results()
    res["params"]["pre_registered_configuration"] = False
    res["params"]["configuration_notes"] = ["restricted universe", "B = 20"]
    block = _summary_block(rep.render_report(res, calibration=make_calibration()))
    assert "NOT the pre-registered configuration" in block
    assert "restricted universe" in block
    ok = make_results()
    ok["params"]["pre_registered_configuration"] = True
    assert "NOT the pre-registered" not in _summary_block(
        rep.render_report(ok, calibration=make_calibration()))


# ---------------------------------------------------------------------------
# 8. Coverage items: the external same-day reference check and the confirmatory test
# ---------------------------------------------------------------------------

def test_the_largest_moves_of_every_vehicle_and_the_core_are_listed_too():
    res = make_results()
    res["vehicle_series"] = {
        "GLD": {"role": "vehicle", "n_rows": 4600, "first_date": "2004-11-18",
                "last_date": "2023-06-30",
                "largest_returns": [{"date": "2008-09-17", "prev_date": "2008-09-16",
                                     "log_return": 0.107, "abs_log_return": 0.107}]},
        "CPER": {"role": "vehicle", "n_rows": 2900, "first_date": "2011-11-15",
                 "last_date": "2023-06-30",
                 "largest_returns": [{"date": "2014-12-05", "prev_date": "2014-12-04",
                                      "log_return": 0.405, "abs_log_return": 0.405}]},
        "SPY": {"role": "core", "n_rows": 7600, "first_date": "1993-01-29",
                "last_date": "2023-06-30", "largest_returns": []}}
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        "Data, counts and checks")
    assert "2014-12-05" in sec and "2008-09-17" in sec
    assert "vehicles" in sec.lower() and "core" in sec.lower()


def test_adjusted_p_by_series_and_side_is_reported_and_labelled_descriptive():
    res = make_results()
    item = {"t_max": 2.31, "argmax_key": "^VIX|up_fade|z20@2.0", "n_tested_groups": 4,
            "p_n1": 0.062, "p_headline": 0.062, "p_n2_sensitivity": None}
    res["confirmatory"]["per_series"] = {"^VIX": item, "GC=F": dict(item, p_headline=0.9,
                                                                    p_n1=0.9,
                                                                    argmax_key="GC=F|up_fade|pct756")}
    res["confirmatory"]["per_side"] = {"up_fade": item, "down_fade": dict(item, p_headline=0.5,
                                                                          p_n1=0.5)}
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    assert "descriptive" in sec.lower()
    assert "GC=F|up_fade|pct756" in sec.replace(" ", "") or "GC=F up_fade pct756" in sec
    assert "0.900" in sec and "0.500" in sec


def test_a_positive_fund_result_on_few_trades_is_called_suggestive_not_conclusive():
    res = make_results()
    res["headline"]["vehicle"].update(excess_pp=4.5, null_percentile=0.9, n=13,
                                      underpowered=False)
    block = _summary_block(rep.render_report(res, calibration=make_calibration())).lower()
    assert "suggestive" in block and "13 trades" in block


def test_decision_table_marks_an_underpowered_vehicle_line():
    res = make_results()
    res["family_table"][0].update(vehicle_n=1)
    sec = _section_text(rep.render_report(res, calibration=make_calibration()),
                        rep.SECTION_TITLES[1])
    assert "underpowered" in sec.lower()


def test_the_power_source_line_marks_the_calibration_seed_as_a_date():
    text = rep._power_section(make_calibration())
    first = text.splitlines()[0]
    assert "date" in first.lower() and "7505" in first


# ---------------------------------------------------------------------------
# 9. The null-size robustness caveat read from the calibration JSON
# ---------------------------------------------------------------------------

def make_robustness(rates=(0.08, 0.13, 0.18), ses=(0.0157, 0.0194, 0.0222)):
    arms = {}
    for mult, rate, se in zip(("1.0", "2.0", "3.0"), rates, ses):
        arms[mult] = {"n_datasets": 300, "n_errors": 0, "alpha": 0.1,
                      "p_n1": {"n_available": 300, "n_denominator": 300,
                               "rejections": round(rate * 300), "rate": rate, "se": se,
                               "in_window": 0.06 <= rate <= 0.14, "gating": mult == "1.0"}}
    return {"finished": True, "spec_version": "3.1", "claim_run": True,
            "args": {"datasets": 300, "B": 200, "n_series": 7, "sessions": 5000,
                     "vol_of_vol_multipliers": [1.0, 2.0, 3.0], "seed": 7505},
            "size_by_vol_of_vol_multiplier": arms, "power": None,
            "robustness_caveat": "REPORTED caveat, not a re-tuning"}


def test_size_caveat_sentence_sits_right_under_the_headline_p():
    sec = _section_text(rep.render_report(make_results(), calibration=make_calibration(),
                                          robustness=make_robustness()),
                        rep.SECTION_TITLES[1])
    i_head = sec.index("Headline p =")
    i_cav = sec.index("Size caveat")
    i_n2 = sec.index("N2 (translation null)")
    assert i_head < i_cav < i_n2
    cav = sec[i_cav:i_n2]
    for needle in ("0.080", "0.130", "0.180", "0.016", "0.019", "0.022", "8 to 18 percent",
                   "double", "triple", "7-series", "non-rejection", "very volatile"):
        assert needle in cav, needle


def test_size_caveat_numbers_come_from_the_json_not_from_the_writer():
    rob = make_robustness(rates=(0.101, 0.303, 0.404), ses=(0.011, 0.022, 0.033))
    text = rep.render_report(make_results(), calibration=make_calibration(), robustness=rob)
    sec = _section_text(text, rep.SECTION_TITLES[1])
    assert "0.303" in sec and "0.404" in sec and "10 to 40 percent" in sec
    assert "8 to 18 percent" not in sec and "0.180" not in sec
    assert "0.404" in _section_text(text, "Contamination and limits")


def test_size_caveat_is_pending_text_when_the_robustness_file_is_absent():
    text = rep.render_report(make_results(), calibration=make_calibration(), robustness=None)
    sec = _section_text(text, rep.SECTION_TITLES[1])
    lim = _section_text(text, "Contamination and limits")
    assert "Size caveat" in sec and "pending" in sec[sec.index("Size caveat"):
                                                    sec.index("N2 (translation null)")].lower()
    assert "null-size robustness caveat" in lim.lower() and "pending" in lim.lower()
    assert "8 to 18" not in text


def test_a_malformed_robustness_file_fails_closed_to_pending_not_to_a_crash():
    for bad in ({}, {"finished": True, "size_by_vol_of_vol_multipliers": {}},
                {"finished": True, "size_by_vol_of_vol_multiplier": {"1.0": {}}},
                {"finished": False, "size_by_vol_of_vol_multiplier": {}}):
        text = rep.render_report(make_results(), calibration=make_calibration(), robustness=bad)
        assert "pending" in _section_text(text, "Contamination and limits").lower()


def test_limits_section_states_the_mechanism_and_the_consequence():
    text = rep.render_report(make_results(), calibration=make_calibration(),
                             robustness=make_robustness())
    lim = _section_text(text, "Contamination and limits")
    low = lim.lower()
    assert "null-size robustness caveat" in low
    for needle in ("convexity", "sign of d", "spike dates", "non-rejection",
                   "no definition was changed", "0.080", "0.130", "0.180", "300 "):
        assert needle in low, needle
    assert "approximately an 8 to 18 percent test" in low


def test_size_caveat_never_changes_a_statistic_or_the_decision_text():
    res = make_results()
    a = rep.render_report(res, calibration=make_calibration(), robustness=None)
    b = rep.render_report(res, calibration=make_calibration(), robustness=make_robustness())
    assert _section_text(a, "Power statement") == _section_text(b, "Power statement")
    assert _section_text(a, "Exploratory map") == _section_text(b, "Exploratory map")
    assert _summary_block(a).count("p = 0.062") == _summary_block(b).count("p = 0.062") == 1


def test_a_convincing_p_carries_the_size_caveat_in_plain_words():
    res = make_results()
    res["headline"].update(p_headline=0.04)
    block = _summary_block(rep.render_report(res, calibration=make_calibration(),
                                             robustness=make_robustness())).lower()
    assert "false alarm" in block and "18" in block
    for word in JARGON:
        assert word not in block, word
    res2 = make_results()
    res2["headline"].update(p_headline=0.3)
    block2 = _summary_block(rep.render_report(res2, calibration=make_calibration(),
                                              robustness=make_robustness())).lower()
    assert "false alarm" not in block2


def test_the_command_is_labelled_as_run_and_the_cached_variant_is_not_duplicated():
    same = {"command": "python x.py --skip-fetch", "command_to_reuse_cached_data":
            "python x.py --skip-fetch", "python": "3.13"}
    sec = _section_text(rep.render_report(make_results(), calibration=make_calibration(),
                                          meta=same), "Seeds, data digests and reproduction")
    assert "Exact command as run" in sec and "fetches fresh data" not in sec
    assert sec.count("python x.py --skip-fetch") == 1
    diff = {"command": "python x.py", "command_to_reuse_cached_data": "python x.py --skip-fetch",
            "python": "3.13"}
    sec2 = _section_text(rep.render_report(make_results(), calibration=make_calibration(),
                                           meta=diff), "Seeds, data digests and reproduction")
    assert "reusing the cached raw data" in sec2 and "python x.py --skip-fetch" in sec2


def test_reproduction_section_says_a_fresh_download_will_have_different_digests():
    sec = _section_text(rep.render_report(make_results(), calibration=make_calibration(),
                                          meta={"command": "python x.py --skip-fetch"}),
                        "Seeds, data digests and reproduction")
    assert "without --skip-fetch" in sec.lower() and "digests will differ" in sec.lower()


# ---------------------------------------------------------------------------
# 10. Seeds are printed with an underscore group separator (no bare long digit runs)
# ---------------------------------------------------------------------------

BARE_RUN = re.compile(r"(?<!\d\.)\b\d{8,17}\b")      # an account-number-like shape


def test_fmt_seed_groups_long_digit_runs_and_leaves_everything_else_alone():
    assert rep.fmt_seed(2026_0923) == "2026_0923"
    assert rep.fmt_seed("2026_0924") == "2026_0924"
    assert rep.fmt_seed(7101) == "7101"
    assert rep.fmt_seed(None) == "n/a"
    assert rep.group_long_digits("seed 2026_0923 and 2023-06-30") == "seed 2026_0923 and 2023-06-30"
    assert rep.group_long_digits("seed " + str(2026_0923) + ".") == "seed 2026_0923."
    assert rep.group_long_digits("n " + str(1234_5678_9012)) == "n 1234_5678_9012"
    assert rep.group_long_digits("x " + str(1234567)) == "x 1234567"          # 7 digits: unchanged
    assert rep.group_long_digits("frac 0." + str(1234_5678_901)) == "frac 0." + str(1234_5678_901)
    digest = "ab" + str(1234_5678) + "cd" + "e" * 52
    assert rep.group_long_digits(digest) == digest                            # inside a hex digest
    assert rep.group_long_digits("a" * 64) == "a" * 64


def test_the_rendered_report_has_no_bare_long_digit_run_with_the_real_seed_values():
    res = make_results()
    res["params"]["seeds"] = {"n1": 2026_0923, "n2": 2026_0924, "map": 2026_0926,
                              "cluster_bootstrap": 2026_0925, "vehicle_null": 2026_0923,
                              "put_baseline": 2026_0926}
    cal = make_calibration()
    cal["args"]["seed"] = 2026_0927
    meta = {"command": "python x.py --B 500", "python": "3.13"}
    manifest = {"^VIX": {"sha256": "a" * 64, "retrieved_utc": "2026-09-23T20:00:00Z"}}
    text = rep.render_report(res, calibration=cal, meta=meta, manifest=manifest,
                             robustness=make_robustness())
    assert BARE_RUN.findall(text) == []
    sec = _section_text(text, "Seeds, data digests and reproduction")
    for shown in ("2026_0923", "2026_0924", "2026_0925", "2026_0926"):
        assert shown in sec
    assert "2026_0927" in _section_text(text, "Power statement")
    failed = rep.render_report(make_failed(params={"B": 500, "B_map": 200,
                                                   "seeds": {"n1": 2026_0923},
                                                   "universe": [], "confirmatory_family": []}))
    assert BARE_RUN.findall(failed) == [] and "2026_0923" in failed
