"""Seal the holdout section of a sweep report so a TRAIN-ONLY read is mechanical.

WHY
---
A pre-registered practice-side experiment (for example, re-training on a deeper
corpus) may look at exactly one thing -- does the train-selected config change?
If the holdout has already been spent once, re-reading it against a newly-selected
config
would be a second look at a spent test set ("re-shopping the max"), which the
beat-the-index gate forbids.

But `strategy_sweep.py` computes and writes the holdout section as a normal part of
its run. So "we didn't look" must be enforced by a MECHANISM, not by a promise:
this script splits the report BEFORE anyone reads it.

  TRAIN_ONLY_<name>.md   <- every section EXCEPT report section 8 (holdout). This is the readable one.
  SEALED_<name>.md       <- report section 8 alone, behind a banner. Do not open without an
                            explicit decision to spend the holdout again.

The original report is left untouched on disk (it is the reproducible artifact); the
discipline is that only the TRAIN_ONLY file is ever read.

USAGE
    py scripts/seal_holdout.py path/to/leveraged_gate_sweep_report.md
"""
import argparse
import os
import re
import sys

HOLDOUT_RE = re.compile(r"^##\s*8\.\s", re.M)
NEXT_SECTION_RE = re.compile(r"^##\s+", re.M)

BANNER = """<!-- ============================================================ -->
<!--  SEALED HOLDOUT -- DO NOT OPEN                                -->
<!--                                                               -->
<!--  Sealing is for a holdout already spent on a config selection, -->
<!--  so this run is PRACTICE-SIDE ONLY.                            -->
<!--  Reading this file evaluates a second config against an        -->
<!--  already-opened test set -- re-shopping the max, which the     -->
<!--  pre-registered beat-the-index gate forbids.                   -->
<!--                                                               -->
<!--  Open ONLY on an explicit, recorded decision to spend    -->
<!--  the holdout again.                                            -->
<!-- ============================================================ -->

"""


def split_report(text):
    """-> (train_only_text, holdout_text). Raises if the holdout section is absent."""
    m = HOLDOUT_RE.search(text)
    if not m:
        raise ValueError("no '## 8.' holdout section found -- refusing to guess")
    start = m.start()
    nxt = NEXT_SECTION_RE.search(text, m.end())
    end = nxt.start() if nxt else len(text)
    holdout = text[start:end]
    train_only = (
        text[:start]
        + "## 8. Holdout — SEALED\n\n"
          "This section was removed before reading. See the sealed sibling file.\n"
          "This run is practice-side only: a sealed holdout is treated as spent, and\n"
          "re-reading it would be re-shopping the max. See `scripts/seal_holdout.py`.\n\n"
        + text[end:]
    )
    return train_only, holdout


def main(argv=None):
    ap = argparse.ArgumentParser(description="Split a sweep report; seal the holdout.")
    ap.add_argument("report")
    ap.add_argument("--out-dir", default=None,
                    help="default: alongside the report")
    args = ap.parse_args(argv)

    if not os.path.exists(args.report):
        sys.exit(f"FAILED: no such report: {args.report}")
    text = open(args.report, encoding="utf-8").read()
    try:
        train_only, holdout = split_report(text)
    except ValueError as e:
        sys.exit(f"FAILED: {e}")

    out = args.out_dir or os.path.dirname(os.path.abspath(args.report))
    os.makedirs(out, exist_ok=True)
    base = os.path.basename(args.report)
    train_fp = os.path.join(out, f"TRAIN_ONLY_{base}")
    seal_fp = os.path.join(out, f"SEALED_{base}")

    with open(train_fp, "w", encoding="utf-8") as f:
        f.write(train_only)
    with open(seal_fp, "w", encoding="utf-8") as f:
        f.write(BANNER + holdout)

    # Fail loudly if any holdout content leaked into the readable file.
    leaked = "Holdout (post-train_end) results vs baselines" in train_only
    print(f"  train-only -> {train_fp}  ({len(train_only)} chars)")
    print(f"  sealed     -> {seal_fp}  ({len(holdout)} chars withheld)")
    if leaked:
        sys.exit("FAILED: holdout heading leaked into the train-only file")
    print("VERIFIED: holdout sealed; train-only file is safe to read.")


if __name__ == "__main__":
    main()
