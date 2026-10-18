"""Renders the mining pipeline's run report and candidate briefs from
fixed text templates.
"""
from __future__ import annotations

import os


def _fmt_list(items):
    return ", ".join(items) if items else "none"


def render_run_report(*, date, n_considered, excluded_ruled_out,
                       excluded_missing_data, not_attempted_budget,
                       n_passed_dsr, n_survivors, survivor_briefs,
                       budget_used, budget_ceiling, n_families_tested_this_run,
                       n_families_correction):
    survivors_line = (", ".join(survivor_briefs) if survivor_briefs else
                       f"zero — none of the {n_considered} candidates "
                       f"tested this run cleared both the in-sample and "
                       f"holdout bars")
    return f"""# Mining run — {date}

- Catalog entries considered: {n_considered}
- Excluded as already ruled-out (catalog status / past gate check): {len(excluded_ruled_out)} — {_fmt_list(excluded_ruled_out)}
- Excluded for missing data: {len(excluded_missing_data)} — {_fmt_list(excluded_missing_data)}
- Not attempted (budget/time ceiling reached): {len(not_attempted_budget)} — {_fmt_list(not_attempted_budget)}
- Passed in-sample deflated-Sharpe bar: {n_passed_dsr}
- Passed holdout confirmation (= survivors): {n_survivors}

**Survivors: {survivors_line}**

Budget used: {budget_used} of {budget_ceiling}
Distinct signal families tested this run: {n_families_tested_this_run}
Multiple-testing correction N (cumulative trial-ledger floor, not this run's count — see candidate briefs): {n_families_correction:.2f}
"""


def render_candidate_brief(*, family_name, ticker, mechanism_hypothesis,
                            parameters_tested, cost_source, tax_treatment,
                            holdout_start_date, dsr_value,
                            n_families_correction, trade_count,
                            other_summary_stats, holdout_window,
                            holdout_result, survival_case,
                            account_eligibility_notes):
    return f"""> **NOT a gate result — requires its own fresh pre-registered gate before any capital.**

# {family_name} — {ticker}

**Mechanism hypothesis:** {mechanism_hypothesis}

**Parameters tested:** {parameters_tested}

**Cost source:** {cost_source}

**Tax treatment:** {tax_treatment}

## In-sample results (through {holdout_start_date})
- Deflated Sharpe Ratio: {dsr_value:.4f} (corrected for an estimated {n_families_correction:.2f} independent trials — correlation-based, floored by the cumulative trial-ledger count, not a raw signal-family count; non-integer is expected, see Bailey & Lopez de Prado 2014 Eq. 9)
- Trade count: {trade_count}
- {other_summary_stats}

## Holdout results ({holdout_window})
- {holdout_result}

## Why this might survive real costs/taxes
{survival_case}

## Account eligibility notes
{account_eligibility_notes}
"""


def brief_filename(family_id: str, ticker: str) -> str:
    return f"candidate-brief-{family_id}-{ticker}.md"


def write_run_outputs(output_dir: str, run_report_text: str, briefs: dict) -> list:
    """Writes run-report.md + each brief to `output_dir` (created if
    missing). `briefs` is {filename: text}. Returns the list of paths
    written, so the caller can report exact locations."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    report_path = os.path.join(output_dir, "run-report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(run_report_text)
    paths.append(report_path)
    for filename, text in briefs.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(path)
    return paths
