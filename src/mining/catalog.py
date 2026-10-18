"""Strategy catalog for the mining pipeline: documented signal families,
each tagged with a family_id, loaded from a JSON file. Catalog CURATION
(adding new families) is an offline step, explicitly out of scope for this
module, which only loads/filters an existing catalog file.
"""
from __future__ import annotations

import json

REQUIRED_FIELDS = ("family_id", "name", "mechanism", "tickers", "parameters",
                    "account_types", "status", "source", "added_date")


def load_catalog(path: str) -> list:
    """Load and validate the catalog file. Raises ValueError naming the
    entry index and missing field if any required field is absent --
    fail closed rather than silently mining an incomplete entry."""
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for i, entry in enumerate(entries):
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"catalog entry {i} ({entry.get('family_id', '?')}) "
                              f"missing required field(s): {missing}")
        if entry["status"] not in ("eligible", "ruled_out"):
            raise ValueError(f"catalog entry {i} ({entry['family_id']}) has "
                              f"invalid status: {entry['status']!r}")
    return entries


def eligible_entries(catalog: list) -> list:
    """Entries whose catalog-curation-time status is 'eligible'. Does NOT
    check the trial ledger -- that is `ruled_out.filter_eligible` (the
    run-time eligibility filter), which applies the same status rule and
    then consults the ledger."""
    return [e for e in catalog if e["status"] == "eligible"]
