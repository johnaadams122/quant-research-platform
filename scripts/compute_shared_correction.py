"""Compute a mining run's SHARED multiple-testing correction: n_trials AND
sr_variance.

This is the barrier step of a mining run's scoring phase: one correction,
computed once, applied identically to every candidate scored in the run.

    py scripts/compute_shared_correction.py --manifest <path> --ledger <path>

`--manifest` is a JSON array of {"family_id", "daily_returns_path",
"sharpe_in_sample"} objects -- one per candidate that reached scoring;
`sharpe_in_sample` is OPTIONAL (feeds the sr_variance this-run fallback tier
only -- see below). Prints ONE JSON object to stdout and exits 0, or prints a
message to stderr and exits non-zero. It never prints a partial or fallback
result: a caller must not be able to read a usable `n_trials`/`sr_variance`
out of a failed run.

## sr_variance (pooling is the most principled choice)

sr_variance is the DOMINANT term in the DSR bar -- a straight multiplier on
`expected_max_sharpe_ratio`, unlike n_trials which sits inside a
slowly-growing factor, so a realistic Sharpe spread moves the bar far more
than the trial count does. Estimating it from one run's own 1-3
candidates, or falling back to an invented floor constant every time (a
structurally always-single-candidate script would otherwise always do so),
is unstable. Three tiers, each used only if the one above is unavailable:
(1) pooled across every DSR score this ledger has ever recorded
(`trial_ledger.pooled_sr_variance`); (2) this run's own candidate Sharpes
(used when the pool is not yet seeded); (3) the pre-registered floor, last
resort. `sr_variance_source` in the result names which tier fired.

## Why this is a tested script

The correction is a standalone, testable file so that two failure modes
are pinned by tests rather than left to an untested caller:

  * Post-batch floor -- the trial count must include the families this run
    is itself first-testing, not only the PRE-run distinct family count
    (`trial_ledger.cumulative_family_count`). The barrier strictly precedes
    the per-candidate loop that holds the only `record_first_test` call, so
    at floor-read time zero of the run's own families are on disk; reading
    the pre-run count alone would under-correct every run.

  * No silent fallback -- a caller must never fall back to
    `n_trials = 1.0` (NO multiple-testing correction at all, the easiest
    possible bar) when this step fails. The caller hard-stops instead.

The floor arithmetic itself now lives in `mining.trial_ledger.post_batch_floor`,
shared with `scripts/run_batch3_mining.compute_shared_correction`, so the two
call sites cannot drift apart.

## Direction of error

A correction that is too LOW is unsafe: it makes the bar easier and can turn a
rejection into a pass. A correction that is too HIGH is merely conservative.
Every fallback here therefore rounds toward MORE correction, and anything that
cannot be computed honestly raises instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd

from mining import constants, dsr, trial_ledger


class CorrectionError(RuntimeError):
    """The correction could not be computed honestly. Always fatal -- the run
    must stop rather than score against a guessed N."""


def _load_manifest(manifest_path: str) -> list:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        raise CorrectionError("manifest not found: %s" % manifest_path)
    except json.JSONDecodeError as exc:
        raise CorrectionError("manifest is not valid JSON (%s): %s"
                              % (manifest_path, exc))
    if not isinstance(entries, list):
        raise CorrectionError("manifest must be a JSON array, got %s"
                              % type(entries).__name__)
    if not entries:
        # An empty manifest means nothing reached scoring. There is no honest
        # correction for zero candidates, and returning 1.0 would be the exact
        # silent-no-correction failure this script exists to prevent.
        raise CorrectionError("manifest is empty -- no candidates to correct")
    cleaned = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CorrectionError("manifest entry %d is not an object" % i)
        family_id = entry.get("family_id")
        path = entry.get("daily_returns_path")
        if not isinstance(family_id, str) or not family_id:
            raise CorrectionError(
                "manifest entry %d has no usable family_id" % i)
        if not isinstance(path, str) or not path:
            raise CorrectionError(
                "manifest entry %d (%s) has no usable daily_returns_path"
                % (i, family_id))
        # sharpe_in_sample is OPTIONAL: feeds the this-run fallback tier only.
        # An absent or non-numeric value is not an error -- that candidate is
        # simply excluded from the this-run sample, same as
        # trial_ledger.pooled_sr_variance excludes ledger records with none.
        sharpe = entry.get("sharpe_in_sample")
        sharpe = float(sharpe) if isinstance(sharpe, (int, float)) else None
        cleaned.append({"family_id": family_id, "daily_returns_path": path,
                        "sharpe_in_sample": sharpe})
    return cleaned


def _load_series(entry: dict) -> pd.Series:
    path = entry["daily_returns_path"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        raise CorrectionError("daily-returns artifact not found for %s: %s"
                              % (entry["family_id"], path))
    except json.JSONDecodeError as exc:
        raise CorrectionError("daily-returns artifact for %s is not valid "
                              "JSON (%s): %s" % (entry["family_id"], path, exc))
    if not isinstance(payload, dict):
        raise CorrectionError("daily-returns artifact for %s is not an object"
                              % entry["family_id"])
    returns = payload.get("returns")
    dates = payload.get("dates")
    if not isinstance(returns, list) or not isinstance(dates, list):
        raise CorrectionError(
            "daily-returns artifact for %s needs list 'returns' and 'dates'"
            % entry["family_id"])
    if len(returns) != len(dates):
        raise CorrectionError(
            "daily-returns artifact for %s has %d returns but %d dates"
            % (entry["family_id"], len(returns), len(dates)))
    try:
        index = pd.to_datetime(dates)
    except (ValueError, TypeError) as exc:
        raise CorrectionError("daily-returns artifact for %s has unparseable "
                              "dates: %s" % (entry["family_id"], exc))
    return pd.Series(returns, index=index, dtype=float)


def compute(manifest_path: str, ledger_path: str) -> dict:
    """The whole barrier. Read-only with respect to the ledger."""
    entries = _load_manifest(manifest_path)
    series = [_load_series(e) for e in entries]

    union = pd.DatetimeIndex([])
    for s in series:
        union = union.union(pd.DatetimeIndex(s.index))
    union = union.sort_values()

    aligned = [s.reindex(union, fill_value=0.0).to_numpy(dtype=float)
               for s in series]

    # A zero-trade candidate's daily returns are all zeros, and Pearson
    # correlation is undefined for a constant series -- num_independent_trials
    # raises on those BY DESIGN and tells callers to exclude them first. They
    # are excluded from the correlation only; they still consume a trial in the
    # ledger floor below, which is the half that actually matters for the bar.
    non_degenerate = [a for a in aligned
                      if a.size and np.all(np.isfinite(a)) and np.std(a) != 0.0]
    degenerate_count = len(aligned) - len(non_degenerate)

    if non_degenerate:
        try:
            n_this_run = float(dsr.num_independent_trials(non_degenerate))
        except ValueError as exc:
            raise CorrectionError("correlation estimate failed: %s" % exc)
    else:
        # Every candidate is degenerate: fall back to the raw count, i.e. the
        # fully-independent assumption. That is the CONSERVATIVE direction --
        # it never under-corrects.
        n_this_run = float(len(aligned))

    family_ids = [e["family_id"] for e in entries]
    prior_family_count = trial_ledger.cumulative_family_count(ledger_path)
    ledger_floor = trial_ledger.post_batch_floor(ledger_path, family_ids)
    n_new_families = ledger_floor - prior_family_count

    n_trials = float(max(n_this_run, float(ledger_floor), 1.0))
    if not np.isfinite(n_trials) or n_trials < 1.0:
        raise CorrectionError("computed a non-usable n_trials: %r" % n_trials)

    # sr_variance tiers (pooling is most principled, since
    # sr_variance is the DOMINANT term in the DSR bar -- a straight
    # multiplier, unlike n_trials's slowly-growing factor). Mirrors
    # run_batch3_mining.compute_shared_correction's own tier order exactly,
    # so the two scoring paths cannot silently disagree: (1) pooled across
    # every score the ledger has ever recorded; (2) this run's own
    # candidate Sharpes (only those manifest entries that carried one); (3)
    # the pre-registered floor, last resort.
    pooled_variance = trial_ledger.pooled_sr_variance(ledger_path)
    if pooled_variance is not None:
        sr_variance = pooled_variance
        sr_variance_source = "pooled"
    else:
        this_run_sharpes = [e["sharpe_in_sample"] for e in entries
                            if e["sharpe_in_sample"] is not None
                            and math.isfinite(e["sharpe_in_sample"])]
        this_run_variance = (float(np.var(this_run_sharpes, ddof=1))
                             if len(this_run_sharpes) >= 2 else math.nan)
        if math.isfinite(this_run_variance) and this_run_variance > 0:
            sr_variance = this_run_variance
            sr_variance_source = "this_run"
        else:
            sr_variance = float(constants.DEFAULT_SR_VARIANCE_FLOOR)
            sr_variance_source = "floor"

    return {
        "n_this_run": n_this_run,
        "sr_variance": sr_variance,
        "sr_variance_source": sr_variance_source,
        # Reported on EVERY call (not just floor-sourced ones): an external
        # caller that cannot read this repo's constants.py could not otherwise
        # independently verify a floor-sourced sr_variance -- or sanity-bound
        # the constant itself against a corrupted value -- without being told
        # what the floor actually is.
        "sr_variance_floor_constant": float(constants.DEFAULT_SR_VARIANCE_FLOOR),
        "prior_family_count": int(prior_family_count),
        "n_new_families": int(n_new_families),
        "ledger_floor": int(ledger_floor),
        "n_trials": n_trials,
        "n_candidates": len(entries),
        # Sorted so the caller can compare it exactly against its own candidate
        # list and detect a dropped, duplicated or mangled manifest.
        "family_ids": sorted(family_ids),
        "degenerate_series_count": degenerate_count,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute a mining run's shared multiple-testing "
                    "correction N (fails closed).")
    parser.add_argument("--manifest", required=True,
                        help="JSON array of {family_id, daily_returns_path}")
    parser.add_argument("--ledger", required=True,
                        help="path to data/mining/trial_ledger.jsonl")
    args = parser.parse_args(argv)
    try:
        result = compute(args.manifest, args.ledger)
    except CorrectionError as exc:
        sys.stderr.write("correction failed: %s\n" % exc)
        return 2
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
