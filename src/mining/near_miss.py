"""Near-miss candidate tracking for the strategy mining pipeline.

The pipeline holds candidates to a strict DSR (Deflated Sharpe Ratio) bar --
`constants.DSR_PASS_THRESHOLD` -- and the bar is deliberately held fixed
rather than loosened to admit more candidates. To compensate for how strict
that is, the pipeline relies on (a) testing more
candidate ideas over time and (b) tracking "near-miss" candidates -- families that scored close
to the bar but did not clear it -- so they can be revisited later (e.g. once
`trial_ledger.pooled_sr_variance`'s pool has more samples, once more
historical data exists, or once the instrument universe widens) instead of
being silently forgotten in the raw ledger. This module is part (b); part
(a) is the pipeline continuing to run, not a mechanism.

This is PURE, READ-ONLY analysis over `data/mining/trial_ledger.jsonl`. It
never re-scores, re-tests, promotes, or spends any new trial budget on
anything, and it never writes to the ledger -- the ledger stays append-only
by convention, and this module only ever reads it (via
`trial_ledger.read_all`, the same lock-free read path every other read-only
consumer in this codebase uses). A cut family stays eligible forever
regardless of near-miss status; this module only surfaces information, it
never changes eligibility or exclusion rules.

## What counts as a "near miss"

Only `outcome` events with `stage == "cut_dsr"` and a real (non-null,
finite) `dsr_value` are ever candidates. This deliberately excludes:

  - a null/missing `dsr_value` (e.g. a `cut_dsr` cut for insufficient trades
    before a real DSR was ever computed, `constants.MIN_TRADES_FOR_SCORING`)
    -- these were never actually SCORED, so there is no meaningful
    "distance from the bar" to report. This also keeps this module
    independent of how never-scored families are labeled elsewhere: this
    module simply never looks at them.
  - `stage == "survived"` -- already cleared the bar, not a miss.
  - `stage == "cut_holdout"` -- this stage means the family CLEARED the DSR
    bar and failed a later, different (out-of-sample holdout sign) check;
    it is not a DSR near-miss at all.

A family may be tested more than once (retests, the evidence-retry path,
`constants.MAX_EVIDENCE_RETRIES_PER_FAMILY`). Only the family's MOST RECENT
outcome that carries a real `dsr_value` determines its current near-miss
status -- mirroring `trial_ledger.pooled_sr_variance`'s own
dedup-by-real-value, latest-wins convention. Two consequences worth being
explicit about: (1) if a family's latest real-valued outcome is
`survived` or `cut_holdout`, it is excluded even if an EARLIER outcome was
a near miss -- its status has moved on. (2) if a family's latest outcome
event is a null-`dsr_value` cut (e.g. a retest that hit the trade-count
gate) but an EARLIER outcome carried a real near-miss score, that earlier
real score still stands -- a null-dsr record is not evidence that the
family it retested moved away from the bar, so it must not silently erase
a real prior measurement.

## The one invented number

`constants.NEAR_MISS_DSR_MARGIN` is the only judgment call this module
makes -- see its comment for the reasoning. It is labeled a judgment
call there rather than presented as derived, the same convention the other
judgment-call constants in `constants` follow.
"""
from __future__ import annotations

import math
import os
from typing import Optional

from . import constants, evidence, trial_ledger


def find_near_misses(ledger_path: str, margin: Optional[float] = None,
                      threshold: Optional[float] = None) -> list:
    """Scan `ledger_path` for `cut_dsr` outcomes within `margin` of
    `threshold` (defaults: `constants.NEAR_MISS_DSR_MARGIN` /
    `constants.DSR_PASS_THRESHOLD`), deduplicated to each family's most
    recent REAL-dsr_value outcome (see module docstring).

    Returns a list of dicts, sorted closest-to-the-bar first:
        {family_id, dsr_value, distance_from_bar, threshold, margin,
         run_id, tested_date, first_tested_date, first_run_id,
         ticker_label, note}

    `run_id`/`tested_date` describe the outcome event this row is built
    from (the family's latest real score); `first_tested_date`/
    `first_run_id` describe the family's FIRST-ever test (from its
    `first_test` event, i.e. `trial_ledger.get_pinned_holdout_window`'s own
    source event) -- None if no `first_test` event exists for it (should
    not happen in production, but read-only code must not crash on it).

    Never mutates `ledger_path`. Read-only, lock-free (matches every other
    read accessor in `trial_ledger`) -- safe to call at any time, including
    while a mining run is writing to the same ledger."""
    if margin is None:
        margin = constants.NEAR_MISS_DSR_MARGIN
    if threshold is None:
        threshold = constants.DSR_PASS_THRESHOLD
    # Defense-in-depth against ANY caller (not just the CLI script) passing
    # a non-finite or out-of-scale override: a NaN margin/threshold makes
    # every comparison against it False in Python, so an unvalidated NaN
    # would silently flag every single scored cut as a "near miss" instead
    # of erroring. scripts/render_near_miss_log.py validates its own CLI
    # values too, but the check lives here as well because the
    # vulnerability is this function's own parameters, not one entry point.
    if not math.isfinite(margin) or not (0.0 < margin <= 1.0):
        raise ValueError(
            f"find_near_misses: margin must be a finite number in (0, 1] "
            f"(DSR's own probability scale), got {margin!r}")
    if not math.isfinite(threshold) or not (0.0 < threshold < 1.0):
        raise ValueError(
            f"find_near_misses: threshold must be a finite DSR probability "
            f"in (0, 1), got {threshold!r}")

    events = trial_ledger.read_all(ledger_path)

    first_tested: dict = {}
    for ev in events:
        if ev["event"] == "first_test" and ev["family_id"] not in first_tested:
            first_tested[ev["family_id"]] = {
                "tested_date": ev["tested_date"],
                "run_id": ev["run_id"],
            }

    # Latest-wins, but ONLY among outcome events carrying a real (non-null,
    # finite) dsr_value -- a later null-dsr retest must not erase an
    # earlier real score (see module docstring). Mirrors
    # trial_ledger.pooled_sr_variance's own dedup convention exactly.
    latest_real_outcome: dict = {}
    for ev in events:
        if ev["event"] != "outcome":
            continue
        dsr = ev.get("dsr_value")
        if dsr is None:
            continue
        dsr = float(dsr)
        if not math.isfinite(dsr):
            continue
        latest_real_outcome[ev["family_id"]] = ev

    rows = []
    for family_id, ev in latest_real_outcome.items():
        if ev["stage"] != "cut_dsr":
            continue
        dsr = float(ev["dsr_value"])
        distance = threshold - dsr
        if distance <= 0:
            # Defensive: a cut_dsr record should never carry a dsr_value
            # >= threshold (that combination would mean the pipeline cut a
            # candidate that actually cleared the bar -- the pipeline's own
            # pass convention elsewhere, e.g. constants.DSR_PASS_THRESHOLD's
            # docstring, is ">=") -- but this module only ever READS the
            # ledger, so it must not assume that invariant holds elsewhere.
            # A dsr_value == threshold exactly (distance == 0) is a PASSING
            # score under that convention, not a near miss, so it is
            # excluded alongside the (already-impossible-if-upstream-is-
            # correct) distance < 0 case.
            continue
        if distance > margin:
            continue
        ft = first_tested.get(family_id, {})
        rows.append({
            "family_id": family_id,
            "dsr_value": dsr,
            "distance_from_bar": distance,
            "threshold": threshold,
            "margin": margin,
            "run_id": ev["run_id"],
            "tested_date": ev["tested_date"],
            "first_tested_date": ft.get("tested_date"),
            "first_run_id": ft.get("run_id"),
            "ticker_label": ev.get("ticker_label"),
            "note": ev.get("note"),
        })

    rows.sort(key=lambda r: (r["distance_from_bar"], r["family_id"]))
    return rows


def find_cost_gate_near_misses(ledger_path: str) -> list:
    """Scan `ledger_path` for families whose CURRENT status is a
    cost-realism-gate exclusion (`stage == "insufficient_evidence"`,
    `evidence.reason_code == "cost_realism_gate_excluded"`) with a positive
    measured `edge_over_cost_ratio`.

    A SEPARATE category from `find_near_misses` above: a cost-gate exclusion happens BEFORE the DSR/holdout step
    ever runs, so it is recorded with `stage == "insufficient_evidence"`,
    never `stage == "cut_dsr"` -- structurally invisible to
    `find_near_misses`, which only ever scans `cut_dsr` rows, regardless of
    how close the measured edge came to clearing the cost gate.

    Unlike `find_near_misses`, this deliberately uses NO invented DISTANCE
    margin. The natural, non-arbitrary split is `edge_over_cost_ratio > 0`
    (a real, measured, directional in-sample edge that just wasn't large
    enough to survive the spread by the required multiple) vs. `<= 0` (no
    real edge at all -- not "close" to clearing the gate in any meaningful
    sense, regardless of the numeric distance).
    DISCLOSED CONSEQUENCE (flagged, not silently chosen, matching this
    module's own convention for judgment calls like `NEAR_MISS_DSR_MARGIN`):
    unlike the DSR table, this list has no upper bound on DEGREE -- a ratio
    of 0.01 against a required 3.0 (essentially no real edge) is listed
    exactly like a ratio of 2.99 (a genuine near miss). Rows are sorted
    closest-to-clearing-the-gate first specifically so a reader can judge
    degree visually rather than trusting an invented cutoff.

    Dedup does NOT simply take each family's latest `outcome` event of any
    kind: a later UNRELATED retry failure -- e.g. `not_evaluated_error`, or
    a DIFFERENT `insufficient_evidence` reason code such as
    `insufficient_trades` -- carries no information about cost-gate status
    at all, so letting it win would silently erase a real, still-current
    cost-gate exclusion. Only two kinds of event
    update a family's tracked status here: (a) a fresh
    `cost_realism_gate_excluded` exclusion (latest such event wins, so a
    re-measured spread/edge always supersedes an older one), and (b) a
    SCORED outcome (`evidence.SCORED_STAGES` -- survived/cut_dsr/
    cut_holdout/cut_risk_refusal), meaning a later run actually cleared the
    cost gate and reached real scoring, so the family's cost-gate-near-miss
    status has genuinely moved on. Any other event is ignored for this
    function's purposes.

    Returns a list of dicts, sorted closest-to-clearing-the-gate first:
        {family_id, edge_over_cost_ratio, edge_over_cost_multiple_required,
         distance_from_required, spread_bps, net_edge_bps, run_id,
         tested_date, ticker_label, note}

    Never mutates `ledger_path`. Read-only, lock-free, same convention as
    `find_near_misses`."""
    events = trial_ledger.read_all(ledger_path)

    latest_relevant: dict = {}
    for ev in events:
        if ev.get("event") != "outcome":
            continue
        stage = ev.get("stage")
        if stage in evidence.SCORED_STAGES:
            latest_relevant[ev["family_id"]] = ev
            continue
        if stage != "insufficient_evidence":
            continue
        payload = ev.get("evidence") or {}
        if payload.get("reason_code") == "cost_realism_gate_excluded":
            latest_relevant[ev["family_id"]] = ev

    rows = []
    for family_id, ev in latest_relevant.items():
        if ev.get("stage") != "insufficient_evidence":
            continue  # latest relevant event was a SCORED outcome -- status moved on
        payload = ev.get("evidence") or {}
        observed = payload.get("observed") or {}
        ratio = observed.get("edge_over_cost_ratio")
        if ratio is None:
            continue
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio <= 0.0:
            continue
        required = observed.get("edge_over_cost_multiple_required")
        required = float(required) if required is not None else None
        distance = (required - ratio) if required is not None else None
        rows.append({
            "family_id": family_id,
            "edge_over_cost_ratio": ratio,
            "edge_over_cost_multiple_required": required,
            "distance_from_required": distance,
            "spread_bps": observed.get("spread_bps"),
            "net_edge_bps": observed.get("net_edge_bps"),
            "run_id": ev["run_id"],
            "tested_date": ev["tested_date"],
            "ticker_label": ev.get("ticker_label"),
            "note": ev.get("note"),
        })

    rows.sort(key=lambda r: (
        r["distance_from_required"] if r["distance_from_required"] is not None
        else float("inf"), r["family_id"]))
    return rows


def _fmt(v) -> str:
    return "n/a" if v is None else str(v)


def render_markdown(rows: list, *, threshold: float, margin: float,
                     sources: list, cost_gate_rows: Optional[list] = None) -> str:
    """Render `rows` (from `find_near_misses`) as one markdown report, in
    the same generated-from-header + table style as
    `results_log.render_markdown`. `cost_gate_rows` (from
    `find_cost_gate_near_misses`, defaults to none rendered when omitted)
    is appended as a second, separately-explained section -- a different
    gate, a different vocabulary, never merged into the DSR table above."""
    lines = [
        "# Mining pipeline -- near-miss candidate log",
        "",
        "Families cut at the DSR bar (`stage == \"cut_dsr\"`) whose most "
        "recent REAL (non-null, finite) recorded DSR came within "
        f"{margin:.4f} of the {threshold:.4f} pass bar "
        "(`constants.DSR_PASS_THRESHOLD`), using "
        "`constants.NEAR_MISS_DSR_MARGIN` unless overridden. The bar itself is held as-is; "
        "this is where a near-miss family waits to be revisited (e.g. once "
        "the pooled sr_variance estimate has more samples, once more "
        "historical data exists, or once the instrument universe widens), "
        "instead of being forgotten inside the raw trial ledger.",
        "",
        "This is a read-only report over the trial ledger. It never "
        "re-scores, re-tests, promotes, or changes the eligibility of any "
        "family -- a cut family stays eligible forever regardless of "
        "near-miss status. It never re-scores a family and never mutates "
        "the ledger; the ledger stays append-only.",
        "",
        "**Never hand-edit this file** -- regenerate via "
        "`scripts/render_near_miss_log.py` any time the trial ledger "
        "changes.",
        "",
        "Sources read for this render:",
    ]
    for s in sources:
        lines.append(f"- `{s.replace(os.sep, '/')}`")
    lines.append("")

    if not rows:
        lines.append(
            "No near-miss families found at the current margin/threshold.")
        lines.append("")
    else:
        lines += [
            "| family_id | recorded DSR | distance from bar | first tested | most recent score | ticker/label | note |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            first = (f"{row['first_tested_date']} ({row['first_run_id']})"
                     if row["first_tested_date"] else "n/a")
            latest = f"{row['tested_date']} ({row['run_id']})"
            note = (row["note"] or "").replace("|", "\\|").replace("\n", " ")
            ticker = _fmt(row["ticker_label"])
            lines.append(
                f"| {row['family_id']} | {row['dsr_value']:.4f} | "
                f"{row['distance_from_bar']:.4f} | {first} | {latest} | "
                f"{ticker} | {note} |"
            )
        lines.append("")

    if cost_gate_rows is None:
        return "\n".join(lines)

    lines += [
        "## Cost-realism-gate near misses",
        "",
        "A SEPARATE category: a family rejected "
        "by the cost-realism gate (`mining.intraday_cost_gate."
        "evaluate_cost_realism_gate`) is cut BEFORE the DSR step ever runs, "
        "so it is recorded with `stage == \"insufficient_evidence\"` / "
        "`reason_code == \"cost_realism_gate_excluded\"`, never "
        "`stage == \"cut_dsr\"` -- invisible to the DSR-bar table above "
        "regardless of how close it came. No invented margin constant is "
        "used here: a family is listed if its measured in-sample "
        "`edge_over_cost_ratio` is POSITIVE (a real, measured, directional "
        "edge that just wasn't large enough to survive the spread by the "
        "required multiple) -- a non-positive ratio means no real edge at "
        "all, not a near miss in any sense.",
        "",
    ]
    if not cost_gate_rows:
        lines.append(
            "No cost-gate near-miss families found (every recorded "
            "cost-realism-gate exclusion had a non-positive measured edge).")
        lines.append("")
        return "\n".join(lines)

    lines += [
        "| family_id | edge/cost ratio | required multiple | distance from required | spread (bps) | net edge (bps) | most recent test | ticker/label | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in cost_gate_rows:
        required = _fmt(row["edge_over_cost_multiple_required"])
        distance = (f"{row['distance_from_required']:.4f}"
                    if row["distance_from_required"] is not None else "n/a")
        latest = f"{row['tested_date']} ({row['run_id']})"
        note = (row["note"] or "").replace("|", "\\|").replace("\n", " ")
        ticker = _fmt(row["ticker_label"])
        lines.append(
            f"| {row['family_id']} | {row['edge_over_cost_ratio']:.4f} | "
            f"{required} | {distance} | {_fmt(row['spread_bps'])} | "
            f"{_fmt(row['net_edge_bps'])} | {latest} | {ticker} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_near_miss_log(ledger_path: str, margin: Optional[float] = None,
                          threshold: Optional[float] = None) -> str:
    """End-to-end convenience: `find_near_misses` + `find_cost_gate_near_misses`
    + `render_markdown`. What `scripts/render_near_miss_log.py` calls."""
    if margin is None:
        margin = constants.NEAR_MISS_DSR_MARGIN
    if threshold is None:
        threshold = constants.DSR_PASS_THRESHOLD
    rows = find_near_misses(ledger_path, margin=margin, threshold=threshold)
    cost_gate_rows = find_cost_gate_near_misses(ledger_path)
    return render_markdown(rows, threshold=threshold, margin=margin,
                            sources=[ledger_path], cost_gate_rows=cost_gate_rows)
