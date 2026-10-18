"""Regenerates runs/mining/NEAR-MISS-LOG.md, the durable record of
strategy-mining candidates that were CUT at the DSR bar but came close
(see src/mining/near_miss.py for the full design rationale), plus a second
section for candidates rejected by the cost-realism gate with a
positive-but-insufficient measured edge (see
near_miss.find_cost_gate_near_misses' own docstring).

Reads one source of record -- never hand-edits, never is hand-edited:
  - data/mining/trial_ledger.jsonl (every real test, append-only)

Pure read-only analysis: never re-scores, re-tests, promotes, or spends any
new trial budget, and never mutates the ledger.

Idempotent: safe to re-run any time, including after every future mining
run -- it always regenerates the full report from scratch rather than
appending, so the output never drifts from the ledger.

This is a SEPARATE artifact from the results log (rendered by
scripts/render_mining_results_log.py) -- deliberately not folded into that
file or its renderer, so each report keeps a single responsibility.

Usage (stdlib only):
    py scripts/render_near_miss_log.py
    py scripts/render_near_miss_log.py --out some/other/path.md
    py scripts/render_near_miss_log.py --margin 0.10 --threshold 0.95
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import near_miss


def _same_file(path_a: str, path_b: str) -> bool:
    """True if `path_a` and `path_b` resolve to the same file on disk --
    catches exact-string identity AND relative-path aliasing (two strings
    that differ but normalize to the same real path, e.g. one containing a
    '..' segment, or a relative vs. absolute form of the same path). Uses
    realpath(abspath(...)) rather than plain string equality specifically
    because a naive string check misses aliasing. Does not require either
    path to already exist -- realpath normalizes purely lexically for a
    path that isn't there yet, and also resolves symlinks for one that is.
    """
    return (os.path.realpath(os.path.abspath(path_a))
            == os.path.realpath(os.path.abspath(path_b)))


def _reject_ledger_aliasing_output(out_path: str, ledger_path: str) -> None:
    """Refuse to proceed if `out_path` resolves to the SAME file as
    `ledger_path` -- a real, plausible CLI mistake (a typo or bad flag
    order, e.g. `py scripts/render_near_miss_log.py --out
    data/mining/trial_ledger.jsonl`), not a contrived edge case. Without
    this guard, `open(out_path, "w")` later would TRUNCATE the append-only
    trial ledger it just read from and overwrite it with Markdown -- real,
    serious ledger corruption. Must run BEFORE any file is opened for
    writing."""
    if _same_file(out_path, ledger_path):
        raise SystemExit(
            f"render_near_miss_log: --out ({out_path!r}) resolves to the "
            f"SAME file as --ledger ({ledger_path!r}) -- refusing to open "
            f"the append-only trial ledger for writing. Opening --out in "
            f"write mode would TRUNCATE the ledger and overwrite it with "
            f"Markdown. Pass a different --out path.")


def _validate_margin(value) -> None:
    """Reject a non-finite or out-of-scale --margin override BEFORE it
    reaches near_miss.render_near_miss_log. NaN is the dangerous case: a
    NaN comparison is always False in Python, so an unvalidated NaN margin
    makes `distance > margin` never exclude a row -- this silently flips
    the report to every single scored cut, reported as a "near miss" with
    no error at all."""
    if value is None:
        return
    if not math.isfinite(value) or not (0.0 < value <= 1.0):
        raise SystemExit(
            f"render_near_miss_log: --margin must be a finite number in "
            f"(0, 1] (DSR's own probability scale), got {value!r}.")


def _validate_threshold(value) -> None:
    """Reject a non-finite or out-of-probability-range --threshold
    override, for the same reason as _validate_margin above."""
    if value is None:
        return
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise SystemExit(
            f"render_near_miss_log: --threshold must be a finite DSR "
            f"probability in (0, 1), got {value!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="data/mining/trial_ledger.jsonl")
    parser.add_argument("--out", default="runs/mining/NEAR-MISS-LOG.md")
    parser.add_argument("--margin", type=float, default=None,
                         help="Override constants.NEAR_MISS_DSR_MARGIN")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override constants.DSR_PASS_THRESHOLD")
    args = parser.parse_args()

    _reject_ledger_aliasing_output(args.out, args.ledger)
    _validate_margin(args.margin)
    _validate_threshold(args.threshold)

    text = near_miss.render_near_miss_log(
        args.ledger, margin=args.margin, threshold=args.threshold)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    rows = near_miss.find_near_misses(args.ledger, margin=args.margin,
                                       threshold=args.threshold)
    cost_gate_rows = near_miss.find_cost_gate_near_misses(args.ledger)
    print(f"Wrote {args.out} ({len(rows)} DSR-bar near-miss famil"
          f"{'y' if len(rows) == 1 else 'ies'}, "
          f"{len(cost_gate_rows)} cost-gate near-miss famil"
          f"{'y' if len(cost_gate_rows) == 1 else 'ies'} found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
