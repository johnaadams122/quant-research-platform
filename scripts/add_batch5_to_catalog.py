"""Append the batch5 accepted candidates to data/mining/catalog.json.

Reads a curation output file (an ``accepted.json`` list of accepted
candidates, passed with --accepted) and appends one catalog entry per
accepted candidate.
Idempotent by family_id: a second run changes nothing; an existing
family_id whose parameters DIFFER raises instead of overwriting (a silent
overwrite would move the tested hypothesis under an already-pinned
family). The merged catalog is validated through mining.catalog.
load_catalog BEFORE anything is written, and the write is atomic.

Catalog entry shape (required fields per mining.catalog.REQUIRED_FIELDS,
plus the same optional keys batch4 used): family_id, name, mechanism,
tickers, parameters, account_types, status, source, added_date +
timeframe, distinctness_note, pooling_note (from the curation's
``resolved_pool``) + shape, benchmark_ticker, signal_tickers,
directional_rationale, rule_pseudocode kept as extra keys so the catalog
row is self-describing.

src/tests/test_mining_catalog.py pins the catalog size and the exact
eligible id set, so appending entries needs a deliberate update of those
pins to the new counts and ids.

Usage:
    python scripts/add_batch5_to_catalog.py --accepted PATH [--catalog PATH]
                                            [--dry-run]
``--dry-run`` prints how many entries WOULD be appended and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mining import catalog as catalog_mod

CATALOG_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")

# accepted.json key -> catalog key, for the fields carried verbatim.
_REQUIRED_FROM_ACCEPTED = ("family_id", "name", "mechanism", "tickers",
                           "parameters", "account_types", "source", "added_date")
_OPTIONAL_VERBATIM = ("timeframe", "distinctness_note", "shape",
                      "benchmark_ticker", "signal_tickers",
                      "directional_rationale", "rule_pseudocode")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def catalog_entry_from_accepted(candidate: dict) -> dict:
    """One catalog row from one accepted candidate. Raises ValueError on a
    missing required field rather than inventing a value."""
    missing = [key for key in _REQUIRED_FROM_ACCEPTED
               if key not in candidate or candidate[key] in (None, "")]
    if missing:
        raise ValueError(
            f"accepted candidate {candidate.get('family_id', '?')} lacks "
            f"required field(s) {missing}")
    if not isinstance(candidate["parameters"], dict):
        raise ValueError(f"{candidate['family_id']}: parameters must be an object")
    if not isinstance(candidate["tickers"], list) or not candidate["tickers"]:
        raise ValueError(f"{candidate['family_id']}: tickers must be a non-empty list")
    entry = {key: candidate[key] for key in _REQUIRED_FROM_ACCEPTED}
    status = candidate.get("status") or "eligible"
    if status not in ("eligible", "ruled_out"):
        raise ValueError(f"{candidate['family_id']}: invalid status {status!r}")
    entry["status"] = status
    for key in _OPTIONAL_VERBATIM:
        if key in candidate and candidate[key] is not None:
            entry[key] = candidate[key]
    pool = candidate.get("resolved_pool")
    if pool:
        entry["pooling_note"] = (
            pool if isinstance(pool, str)
            else f"resolved_pool: {_canonical(pool)}")
    return entry


def merge_entries(existing: list, candidates: list) -> tuple[list, list]:
    """Return ``(merged_catalog, appended_ids)``. An existing family_id with
    identical parameters is left untouched; one with DIFFERENT parameters
    raises ValueError -- never overwritten."""
    by_id = {entry["family_id"]: entry for entry in existing}
    merged = list(existing)
    appended = []
    for candidate in candidates:
        new_entry = catalog_entry_from_accepted(candidate)
        family_id = new_entry["family_id"]
        current = by_id.get(family_id)
        if current is not None:
            if _canonical(current.get("parameters")) != _canonical(new_entry["parameters"]):
                raise ValueError(
                    f"{family_id}: already in the catalog with DIFFERENT "
                    f"parameters {_canonical(current.get('parameters'))} vs "
                    f"accepted {_canonical(new_entry['parameters'])} -- refusing "
                    "to overwrite; curate a new family_id instead")
            continue
        merged.append(new_entry)
        by_id[family_id] = new_entry
        appended.append(family_id)
    return merged, appended


def _validate_via_loader(entries: list) -> None:
    """mining.catalog.load_catalog only reads a file, so validate the merged
    list by writing it to a temporary file and loading THAT."""
    descriptor, temporary = tempfile.mkstemp(prefix="catalog-validate-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2, ensure_ascii=False)
        catalog_mod.load_catalog(temporary)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def _atomic_write_json(path: str, entries: list) -> None:
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(destination)}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(entries, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def add_batch5(accepted_path: str, catalog_path: str, *, dry_run: bool) -> dict:
    with open(accepted_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload.get("accepted")
    if not isinstance(candidates, list):
        raise ValueError(f"{accepted_path}: 'accepted' must be a list")
    existing = catalog_mod.load_catalog(catalog_path)
    merged, appended = merge_entries(existing, candidates)
    _validate_via_loader(merged)
    if appended and not dry_run:
        _atomic_write_json(catalog_path, merged)
    return {"catalog": catalog_path.replace(os.sep, "/"),
            "before": len(existing), "after": len(merged),
            "appended": appended, "already_present": len(candidates) - len(appended),
            "dry_run": dry_run, "written": bool(appended and not dry_run)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", required=True,
                        help="path to the curation output accepted.json")
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = add_batch5(args.accepted, args.catalog, dry_run=args.dry_run)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
