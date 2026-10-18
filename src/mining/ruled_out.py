"""Catalog load & ruled-out check: the RUN-TIME half of the
ruled-out check. Catalog-curation-time judgment (matching a family by
MECHANISM, not name, against gate verdict docs) happens offline when the
catalog is curated, outside this code. This
module does the cheap, mechanical part that runs on every invocation:
filter by the catalog's own recorded status, then cross-check the trial
ledger so a family a PRIOR mining run already PROMOTED doesn't get
re-mined. A merely CUT family may retry against fresh data.
"""
from __future__ import annotations

from . import trial_ledger


def filter_eligible(catalog: list, ledger_path: str) -> tuple:
    """Returns (eligible, excluded) where `excluded` is a list of
    {family_id, reason} dicts logging every entry this run will NOT mine
    and why -- every exclusion is logged with its reason.
    Never silently drops an entry."""
    excluded = []
    eligible = []
    for entry in catalog:
        family_id = entry["family_id"]
        if entry["status"] != "eligible":
            excluded.append({
                "family_id": family_id,
                "reason": f"catalog status is {entry['status']!r}: "
                          f"{entry.get('ruled_out_reason', 'no reason recorded')}",
            })
            continue
        outcome = trial_ledger.latest_outcome(ledger_path, family_id)
        if outcome == "survived":
            excluded.append({
                "family_id": family_id,
                "reason": "already survived a prior mining run and was "
                          "promoted -- re-mining adds no value",
            })
            continue
        if outcome == "insufficient_evidence":
            # A THIRD exclusion category (evidence lifecycle),
            # alongside catalog status and a survived outcome: this
            # family was already recorded as never having produced enough
            # in-sample trades to judge at all (evidence.py's own
            # insufficient_evidence stage). Re-mining it in an ORDINARY run
            # would just reproduce the identical never-scored result against
            # the SAME pinned holdout window (trial_ledger.record_first_test
            # pins it permanently) -- evidence.py's own module docstring:
            # "waiting for calendar time can never add an in-sample trade to
            # an already-pinned family." A deliberate RETRY against a
            # registered reactivation predicate (widened instrument set,
            # extended history) is a separate, explicit mechanism, not
            # ordinary re-mining, and is not built by this exclusion.
            #
            # Deliberately NOT excluding not_evaluated_error here: an engine
            # error is a defect, not evidence about the market, so that
            # family must stay eligible and re-enter ordinary runs once the
            # defect is fixed (evidence.build_engine_error's own docstring).
            excluded.append({
                "family_id": family_id,
                "reason": "recorded insufficient_evidence by a prior mining "
                          "run -- never produced enough in-sample trades to "
                          "judge against its pinned holdout window; a fresh "
                          "ordinary run cannot add in-sample data. Retry "
                          "only via a registered reactivation predicate, not "
                          "ordinary re-mining",
            })
            continue
        eligible.append(entry)
    return eligible, excluded
