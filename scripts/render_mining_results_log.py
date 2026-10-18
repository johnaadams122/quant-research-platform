"""Regenerates runs/mining/RESULTS-LOG.md, the durable cumulative
record of every strategy-mining candidate ever tested, so each result
is remembered and can be reviewed later.

Reads three sources of record -- never hand-edits, never is hand-edited:
  - data/mining/trial_ledger.jsonl (every test, append-only)
  - data/mining/catalog.json (mechanism/ticker context)
  - runs/mining/*/run-report.md (missing-data exclusions -- the
    only place a candidate that never reached the trial ledger at all is
    recorded)

Idempotent: safe to re-run any time, including after every future mining
run -- it always regenerates the full table from scratch rather than
appending, so the output never drifts from the source files.

Usage (stdlib only -- runs fine under either interpreter, `py` matches
this project's other scripts/ convention):
    py scripts/render_mining_results_log.py
    py scripts/render_mining_results_log.py --out some/other/path.md
"""
from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import catalog as catalog_mod
from mining import results_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/mining/catalog.json")
    parser.add_argument("--ledger", default="data/mining/trial_ledger.jsonl")
    parser.add_argument("--research-root", default="runs/mining")
    parser.add_argument("--out", default="runs/mining/RESULTS-LOG.md")
    args = parser.parse_args()

    catalog = catalog_mod.load_catalog(args.catalog)
    text = results_log.render_results_log(catalog, args.ledger, args.research_root)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    n_reports = len(results_log.load_all_run_reports(args.research_root))
    print(f"Wrote {args.out} ({len(catalog)} catalog entries, "
          f"{n_reports} run-report(s) read)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
