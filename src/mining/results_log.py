"""Renders the mining pipeline's cumulative, human-readable results log
(every candidate's result is logged somewhere durable so it can be
reviewed later).

A survivor gets a full candidate brief (reporting.py), but a CUT
candidate's real numbers (DSR value, outcome stage) otherwise live only
in the raw JSONL trial ledger, or in a per-run-date run-report.md that
gets OVERWRITTEN every time a second run happens on the same date
(reporting.write_run_outputs opens run-report.md with mode "w"). This
module reads all three sources of record --
data/mining/trial_ledger.jsonl (every real test, append-only),
data/mining/catalog.json (mechanism/ticker context, and the
catalog-curation-time ruled_out entries that never reach the ledger at
all), and every dated <research_root>/*/run-report.md (the mining run
folders, runs/mining by default; the only
place a MISSING-DATA exclusion is recorded, since a candidate that fails
at fetch_ohlc in the Backtest phase never reaches trial_ledger.record_
first_test) -- and folds them into ONE cumulative markdown table covering
every candidate ever considered, not just survivors.

Regenerate via scripts/render_mining_results_log.py after every mining
run. This module never hand-edits or is hand-edited; it is a pure read-
and-render step over the three sources of record above.
"""
from __future__ import annotations

import glob
import os
import re

from . import trial_ledger

_RUN_DATE_RE = re.compile(r"^# Mining run — (\S+)", re.MULTILINE)
_N_CONSIDERED_RE = re.compile(r"^- Catalog entries considered: (\d+)", re.MULTILINE)
_RULED_OUT_RE = re.compile(
    r"^- Excluded as already ruled-out[^:]*: \d+ — (.+)$", re.MULTILINE)
_MISSING_DATA_RE = re.compile(
    r"^- Excluded for missing data: \d+ — (.+)$", re.MULTILINE)
_NOT_ATTEMPTED_RE = re.compile(
    r"^- Not attempted \(budget/time ceiling reached\): \d+ — (.+)$", re.MULTILINE)

# Splits a comma-joined list of "family_id: reason" items WITHOUT breaking
# on a comma that appears inside a reason's own text (e.g. "...skipped
# rather than guessed, since no cost source was found..."). Only
# splits at a ", " that is immediately followed by a (possibly empty --
# see the blank-family_id note below) family_id slug and ": ".
_MISSING_ITEM_SPLIT_RE = re.compile(r", (?=[a-z0-9_]*: )")

OUTCOME_LABELS = {
    "survived": "survived",
    "cut_dsr": "cut at DSR bar",
    "cut_holdout": "cut at holdout",
    "cut_risk_refusal": "cut at risk refusal (ABI-1)",
    # The two evidence.py UNSCORED_STAGES: a deferral or
    # a defensive error record must never render indistinguishably from a
    # real statistical rejection -- the ambiguity ("cut_dsr" +
    # dsr_value=null) this lifecycle exists to remove.
    "insufficient_evidence": "insufficient evidence (deferred)",
    "not_evaluated_error": "not evaluated (engine error)",
}


def _split_list(raw: str) -> list:
    raw = raw.strip()
    if not raw or raw == "none":
        return []
    return [x.strip() for x in raw.split(", ") if x.strip()]


def parse_run_report(text: str) -> dict:
    """Extract one run-report.md's own date line + its three exclusion
    lists (reporting.render_run_report's template). Returns
    {'run_date', 'n_considered', 'ruled_out': [family_id, ...],
    'missing_data': [{'family_id', 'reason'}, ...],
    'not_attempted': [family_id, ...]}.

    A missing_data entry's family_id may be '' -- a run report can carry
    a blank family_id for that item. This function
    surfaces it as-is; see resolve_missing_data_attribution for the
    (best-effort, honestly-labeled) recovery."""
    date_m = _RUN_DATE_RE.search(text)
    run_date = date_m.group(1) if date_m else None

    n_m = _N_CONSIDERED_RE.search(text)
    n_considered = int(n_m.group(1)) if n_m else None

    ruled_out_m = _RULED_OUT_RE.search(text)
    ruled_out = _split_list(ruled_out_m.group(1)) if ruled_out_m else []

    missing_data = []
    missing_m = _MISSING_DATA_RE.search(text)
    if missing_m:
        raw = missing_m.group(1).strip()
        if raw and raw != "none":
            for item in _MISSING_ITEM_SPLIT_RE.split(raw):
                family_id, _, reason = item.partition(": ")
                missing_data.append({"family_id": family_id.strip(),
                                      "reason": reason.strip()})

    not_attempted_m = _NOT_ATTEMPTED_RE.search(text)
    not_attempted = _split_list(not_attempted_m.group(1)) if not_attempted_m else []

    return {"run_date": run_date, "n_considered": n_considered,
            "ruled_out": ruled_out, "missing_data": missing_data,
            "not_attempted": not_attempted}


def load_all_run_reports(research_root: str) -> list:
    """Glob `research_root`/*/run-report.md, parse each, sorted by the
    report's own date line (falling back to its parent directory name if
    that line is missing/malformed). Each parsed dict also carries
    'path'."""
    paths = sorted(glob.glob(os.path.join(research_root, "*", "run-report.md")))
    reports = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        parsed = parse_run_report(text)
        if not parsed["run_date"]:
            parsed["run_date"] = os.path.basename(os.path.dirname(path))
        parsed["path"] = path
        reports.append(parsed)
    reports.sort(key=lambda r: (r["run_date"], r["path"]))
    return reports


def resolve_missing_data_attribution(reports: list, catalog: list,
                                      ledger_path: str) -> list:
    """Best-effort recovery for a blank-family_id missing_data entry
    (see parse_run_report). For each report, identify which of the
    catalog-eligible family_ids IN THAT REPORT'S OWN SCOPE (the first
    n_considered entries of `catalog`, in catalog order -- this assumes
    the run considered exactly those entries, so that its ruled-out list
    is the ruled_out-status entries among the first n_considered catalog
    entries) are otherwise unaccounted for by this
    report's own ruled_out/named-missing/not_attempted lists or by the
    ledger ever recording a test for them. If the count of such
    "leftover" family_ids exactly matches the count of blank entries in
    this report, attribute them (sorted, for a deterministic order) and
    say so in the reason text. If the counts don't match, leave the
    blank entries unattributed with an explicit placeholder rather than
    guess wrong -- never silently drop them (the run's own exclusion
    count would then no longer reconcile).

    Returns a NEW list of report dicts; does not mutate the inputs."""
    ledger_events = trial_ledger.read_all(ledger_path)
    ever_tested = {ev["family_id"] for ev in ledger_events
                   if ev["event"] == "first_test"}

    resolved = []
    for report in reports:
        blanks = [m for m in report["missing_data"] if not m["family_id"]]
        if not blanks:
            resolved.append(report)
            continue

        n = report["n_considered"]
        scope = catalog[:n] if n is not None else catalog
        eligible_scope = {e["family_id"] for e in scope if e["status"] == "eligible"}

        accounted = (set(report["ruled_out"])
                     | {m["family_id"] for m in report["missing_data"] if m["family_id"]}
                     | set(report["not_attempted"])
                     | ever_tested)
        leftover = sorted(eligible_scope - accounted)

        named = [m for m in report["missing_data"] if m["family_id"]]
        if len(leftover) == len(blanks):
            for family_id, blank in zip(leftover, blanks):
                named.append({
                    "family_id": family_id,
                    "reason": blank["reason"] + " (family_id inferred by elimination "
                              "against this report's own considered-scope + exclusion "
                              "lists -- the run-report itself carries a blank family_id "
                              "for this item)",
                })
        else:
            for i, blank in enumerate(blanks):
                named.append({
                    "family_id": f"(unidentified candidate #{i + 1} in "
                                  f"{report['run_date']} report -- blank family_id, "
                                  f"{len(blanks)} blank item(s) vs {len(leftover)} "
                                  "unaccounted eligible candidate(s), counts didn't "
                                  "match so not attributed)",
                    "reason": blank["reason"],
                })
        new_report = dict(report)
        new_report["missing_data"] = named
        resolved.append(new_report)
    return resolved


def _one_line(text: str, max_len: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "\u2026"
    return text


def build_rows(catalog: list, ledger_path: str, resolved_reports: list) -> list:
    """One row per family_id ever mentioned in the catalog, the ledger,
    or any resolved run-report's exclusion lists (catalog order first,
    then any stragglers the catalog itself doesn't know about). Each row:
    {family_id, ticker, timeframe, dates_tested, dsr_value, outcome,
     mechanism, note}."""
    events = trial_ledger.read_all(ledger_path)
    tested_dates = {}
    latest_stage = {}
    latest_dsr = {}
    latest_note = {}
    latest_ticker_label = {}
    for ev in events:
        fam = ev["family_id"]
        if ev["event"] == "first_test":
            tested_dates.setdefault(fam, set()).add(ev["tested_date"])
        elif ev["event"] == "outcome":
            tested_dates.setdefault(fam, set()).add(ev["tested_date"])
            latest_stage[fam] = ev["stage"]
            latest_dsr[fam] = ev["dsr_value"]
            # .get(), not [...] -- note/ticker_label are optional fields
            # that older outcome events do not carry. "last one wins"
            # matches latest_stage/
            # latest_dsr's own convention: a later outcome WITHOUT these
            # fields correctly clears an earlier one's label/note rather
            # than leaving a stale pooled-test label on a family that
            # was since re-tested normally.
            latest_note[fam] = ev.get("note")
            latest_ticker_label[fam] = ev.get("ticker_label")

    missing_info = {}
    not_attempted_info = {}
    for report in resolved_reports:
        for m in report["missing_data"]:
            entry = missing_info.setdefault(m["family_id"], {"reason": m["reason"], "dates": set()})
            entry["reason"] = m["reason"]
            entry["dates"].add(report["run_date"])
        for fam in report["not_attempted"]:
            not_attempted_info.setdefault(fam, set()).add(report["run_date"])

    by_family = {e["family_id"]: e for e in catalog}
    seen = set()
    rows = []

    def _row_for(family_id: str) -> dict:
        entry = by_family.get(family_id)
        name = entry["name"] if entry else "(not in catalog)"
        ticker = entry["tickers"][0] if entry and entry.get("tickers") else "n/a"
        mechanism = _one_line(entry["mechanism"]) if entry else "n/a"
        # timeframe is sourced from the catalog entry, not the ledger --
        # a family's mechanism/timeframe doesn't change between tests.
        # Absent on older entries (all daily-track), so
        # defaults to "daily"; a straggler with no catalog entry at all
        # gets "n/a", matching name's own "(not in catalog)" convention.
        timeframe = entry.get("timeframe", "daily") if entry else "n/a"

        if family_id in tested_dates:
            stage = latest_stage.get(family_id)
            dsr = latest_dsr.get(family_id)
            return {
                "family_id": family_id, "name": name,
                "ticker": latest_ticker_label.get(family_id) or ticker,
                "timeframe": timeframe,
                "dates_tested": sorted(tested_dates[family_id]),
                "dsr_value": dsr,
                "outcome": OUTCOME_LABELS.get(stage, stage or "tested, outcome unknown"),
                "mechanism": mechanism,
                "note": latest_note.get(family_id) or "",
                # Carried alongside the human-readable "outcome" label so
                # render_markdown's DSR column can tell an UNSCORED
                # deferral apart from a legacy null with no stage at all
                # (see _fmt_dsr) -- the label alone is not enough, since a
                # reader comparing two "n/a" cells side by side could not
                # otherwise tell them apart.
                "stage": stage,
            }
        if family_id in missing_info:
            info = missing_info[family_id]
            return {
                "family_id": family_id, "name": name, "ticker": ticker,
                "timeframe": timeframe,
                "dates_tested": sorted(info["dates"]),
                "dsr_value": None,
                "outcome": "no price data available",
                "mechanism": mechanism,
                "note": info["reason"],
            }
        if family_id in not_attempted_info:
            return {
                "family_id": family_id, "name": name, "ticker": ticker,
                "timeframe": timeframe,
                "dates_tested": sorted(not_attempted_info[family_id]),
                "dsr_value": None,
                "outcome": "not attempted (budget/time ceiling)",
                "mechanism": mechanism,
                "note": "",
            }
        if entry and entry["status"] == "ruled_out":
            return {
                "family_id": family_id, "name": name, "ticker": ticker,
                "timeframe": timeframe,
                "dates_tested": [],
                "dsr_value": None,
                "outcome": "ruled out before catalog curation",
                "mechanism": mechanism,
                "note": entry.get("ruled_out_reason", ""),
            }
        return {
            "family_id": family_id, "name": name, "ticker": ticker,
            "timeframe": timeframe,
            "dates_tested": [],
            "dsr_value": None,
            "outcome": "eligible -- not yet tested",
            "mechanism": mechanism,
            "note": "",
        }

    for entry in catalog:
        fam = entry["family_id"]
        seen.add(fam)
        rows.append(_row_for(fam))

    # Any family_id the ledger/reports mention that the catalog doesn't
    # (defensive -- should not happen in practice, but never silently drop).
    stragglers = ((set(tested_dates) | set(missing_info) | set(not_attempted_info))
                  - seen)
    for fam in sorted(stragglers):
        rows.append(_row_for(fam))

    return rows


# Rendered in the DSR column for the two UNSCORED stages instead of the bare
# "n/a" a legacy stage-less null already uses -- see OUTCOME_LABELS' own
# comment for why: a deferral must never be visually indistinguishable from
# "scored, and the value happened to be missing."
_UNSCORED_DSR_TEXT = {
    "insufficient_evidence": "deferred (insufficient evidence)",
    "not_evaluated_error": "not evaluated (error)",
}


def _fmt_dsr(v, stage=None) -> str:
    if v is not None:
        return f"{v:.4f}"
    return _UNSCORED_DSR_TEXT.get(stage, "n/a")


def _fmt_dates(dates: list) -> str:
    return ", ".join(dates) if dates else "n/a"


def render_markdown(rows: list, *, sources: list) -> str:
    """Render `rows` (from build_rows) as one markdown table, preceded by
    a short generated-from header. `sources` are the file paths this
    render actually read, printed so a reader can see exactly what's
    behind the numbers."""
    lines = [
        "# Mining pipeline -- cumulative results log",
        "",
        "Every candidate ever considered by a real mining run, not just "
        "survivors. Regenerated from source of record every time -- "
        "**never hand-edit this file**; re-run "
        "`scripts/render_mining_results_log.py` after any mining run.",
        "",
        "Sources read for this render:",
    ]
    for s in sources:
        lines.append(f"- `{s}`")
    lines += [
        "",
        "| family_id | ticker | timeframe | date(s) tested | DSR value | outcome | mechanism | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        note = row["note"].replace("|", "\\|").replace("\n", " ") if row["note"] else ""
        mechanism = row["mechanism"].replace("|", "\\|")
        lines.append(
            f"| {row['family_id']} | {row['ticker']} | {row['timeframe']} | "
            f"{_fmt_dates(row['dates_tested'])} | "
            f"{_fmt_dsr(row['dsr_value'], row.get('stage'))} | "
            f"{row['outcome']} | {mechanism} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_results_log(catalog: list, ledger_path: str, research_root: str) -> str:
    """End-to-end convenience: load every run-report under `research_root`,
    resolve blank-family_id missing_data entries against `catalog` +
    `ledger_path`, build one row per family_id, and render the markdown
    table. This is what scripts/render_mining_results_log.py calls."""
    reports = load_all_run_reports(research_root)
    resolved = resolve_missing_data_attribution(reports, catalog, ledger_path)
    rows = build_rows(catalog, ledger_path, resolved)
    sources = [ledger_path] + [r["path"] for r in reports]
    sources = [s.replace(os.sep, "/") for s in sources]
    return render_markdown(rows, sources=sources)
