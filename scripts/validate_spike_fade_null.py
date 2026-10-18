"""Null-calibration acceptance test of the range-bound spike-fade study:
calibrate the null machinery on synthetic data (the size check and the power
statement of the confirmatory max-t test; the definitions are frozen in
`mining.spike_fade_null`).

WHAT IT DOES. Generates datasets of independent stochastic-volatility,
fat-tailed (Student t, 3.5 to 5 degrees of freedom) random-walk series with NO
edge (13 series of 5000 sessions by default), runs the WHOLE confirmatory test
(frozen tested set, N1 and N2 nulls, B replications) on each, and reports:

  * SIZE: the rejection rate at 0.10 of N1, of the headline and of N2 over the
    no-edge datasets. SPEC v3.1: the headline p is the N1 p ALONE, so the
    pre-registered window [0.06, 0.14] is REQUIRED for N1 and for the headline
    (= N1); N2's rate (as specified, and like-for-like) is REPORTED, labelled
    "sensitivity, not gating", and never required. At least 300 datasets are
    required for the claim. The same numbers are reported for the sub-family
    of the first 7 series, taken from the same null replications (the real
    confirmatory family has 7 series).
  * POWER: with post-spike reversion of 0.25, 0.5 and 1.0 sigma21 per trade
    injected into ONE series, the observed power curve, the realized effect,
    the smallest effects with 50 and 80 percent headline power, and whether
    the requirement (at least 50 percent power at 1.0) holds.

A failed size window is a FINDING: the report carries the exact numbers and
nothing here tunes a definition to make it pass.

CLAIM RUNS AND FAILURES. A claim run needs B >= 200 replications per dataset
(the synthetic calibration test): `--B` below 200 is refused unless `--smoke` (tiny test runs; the
report says `claim_run: false`). A dataset that raises is recorded as an error,
and ANY error fails the run: a loud "FAILED" line, `verdict.run_ok = false`,
`size_datasets_ok = false` and exit code 1 (errors are never silently dropped
from the size denominator).

ROBUSTNESS ARM. `--vol-of-vol-multipliers 1 2 3` (default 1) runs the size
datasets once per multiplier of the log-volatility innovation of the generator
(volatility = vol * exp(h / 2); h has innovation sd eta, so the log-volatility
innovation sd is eta / 2, times the multiplier), on the SAME dataset seeds
(paired arms). The default arm (multiplier 1, the generator of the v3.1
calibration byte for byte) is THE size claim; the other arms are a REPORTED
caveat about how the N1 size behaves under higher vol-of-vol, not a re-tuning,
and never gate. The power run uses the default arm only.

RE-SUMMARIZING WITHOUT RE-RUNNING. `--resummarize <calibration.json>` reads the
raw p-values stored in an existing calibration JSON, restates every record's
headline as its N1 p (spec v3.1; a v3 file stored max(N1, N2)), recomputes all
summaries and writes `null_calibration_v3_1_summary.json` next to the input
(or `--summary-out`), keeping the v3-stored headline size as provenance. The
raw records are left out of the summary unless `--include-records` is given. Synthetic data only, no
network. Uses multiprocessing with at most 6 workers (4 is a sensible
default); results are written atomically to --out as strict JSON and
re-written every --checkpoint-every finished datasets, so a killed run keeps
its progress on disk.

Usage (Windows, from the repository root):
    python scripts/validate_spike_fade_null.py --workers 4
    ... --datasets 300 --B 200 --power-datasets 60 --seed 2026_0927 --out <path>
Small smoke run:
    ... --datasets 6 --power-datasets 0 --n-series 3 --sessions 2500 --B 30 --out <path>
Robustness measurement (7-series family, 3 arms, at most 4 workers):
    ... --n-series 7 --datasets 300 --power-datasets 0 --vol-of-vol-multipliers 1 2 3 \
        --workers 4 --out runs/spike-fade-study/calibration/null_size_robustness.json
Re-summarize an existing run under spec v3.1:
    ... --resummarize runs/spike-fade-study/calibration/null_calibration.json
"""
import os

# Keep BLAS single-threaded per worker BEFORE numpy loads: the work is many
# tiny arrays, and oversubscribing threads only makes the machine unresponsive.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mining import spike_fade_null as sfn  # noqa: E402

MAX_WORKERS = 6
MIN_CLAIM_B = 200          # acceptance test: B >= 200 replications per dataset
DEFAULT_SEED = 2026_0927
DEFAULT_OUT = os.path.join(REPO_ROOT, "runs", "spike-fade-study", "calibration", "null_calibration.json")


def clamp_workers(n):
    """At least 1, at most MAX_WORKERS (6): the machine stays responsive."""
    return max(1, min(int(n), MAX_WORKERS))


def validate_args(args):
    """An error message, or None when the arguments are acceptable. A claim
    run needs B >= 200 (the acceptance test); `--smoke` allows smaller B for tiny
    test runs. Every vol-of-vol multiplier must be positive."""
    if int(args.B) < MIN_CLAIM_B and not args.smoke:
        return (f"--B {args.B} is below {MIN_CLAIM_B}: a calibration claim needs B >= {MIN_CLAIM_B} "
                f"replications per dataset (the synthetic calibration test); pass --smoke for a tiny test run")
    if any(not float(v) > 0 for v in args.vol_of_vol_multipliers):
        return "every vol-of-vol multiplier must be positive"
    return None


def clean(obj):
    """Recursively make a structure strict-JSON safe: non-finite floats become
    None, mapping keys become strings."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(clean(payload), fh, indent=1, allow_nan=False)
    os.replace(tmp, path)


def build_tasks(args):
    """Size tasks (no effect; one block per vol-of-vol arm, all arms on the
    same dataset seeds) then power tasks (one block per effect, default arm).
    Every dataset has its own seed: master + k for size, master + block
    offset + k for power, so no two datasets of one arm share a stream."""
    common = {"n_series": args.n_series, "n_sessions": args.sessions, "B": args.B}
    arms = [float(v) for v in getattr(args, "vol_of_vol_multipliers", [1.0])]
    # one size block per vol-of-vol arm on the SAME dataset seeds (paired arms)
    tasks = [dict(common, dataset_seed=args.seed + k, effect=0.0, kind="size",
                  vol_of_vol_multiplier=mult)
             for mult in arms for k in range(args.datasets)]
    for i, eff in enumerate(args.effects):                   # power: the default arm only
        base = args.seed + 1_000_000 * (i + 1)
        tasks += [dict(common, dataset_seed=base + k, effect=float(eff), kind="power",
                       vol_of_vol_multiplier=1.0)
                  for k in range(args.power_datasets)]
    return tasks


def subset_view(records):
    """Records restated on the sub-family of the first 7 series (same null
    replications), so the same summaries apply."""
    out = []
    for r in records:
        s = r.get("subset")
        if s is None:
            continue
        out.append({"p_n1": s["p_n1"], "p_n2": s["p_n2"], "p_n2_covered": s.get("p_n2_covered"),
                    "p_headline": s["p_headline"],
                    "realized_mean_x": r.get("realized_mean_x")})
    return out


def n2_coverage(records):
    """How often N2 was available at all: the share of datasets where it
    covers the T* group, and the mean number of series with at least
    N2_MIN_OFFSETS feasible offsets (a series with fewer is skipped)."""
    recs = [r for r in records if "error" not in r]
    if not recs:
        return {}
    cov = [bool(r["n2_covers_tstar"]) for r in recs]
    per = []
    for r in recs:
        offs = r.get("n2_offsets") or {}
        per.append(sum(1 for v in offs.values() if v >= sfn.N2_MIN_OFFSETS))
    return {"share_tstar_covered": sum(cov) / len(cov),
            "mean_series_with_n2": sum(per) / len(per), "n_series": recs[0]["n_series"]}


SPEC_VERSION = "3.1"
HEADLINE_DEFINITION = "N1 alone (p_headline = p_n1)"
N2_LABEL = sfn.N2_SENSITIVITY_LABEL
SUMMARY_NAME = "null_calibration_v3_1_summary.json"


def restate_headline(rec):
    """A copy of a record with the headline restated as the N1 p (spec v3.1).
    A v3 file stored the headline as max(N1, N2); the stored value is kept as
    `p_headline_v3_stored`. A record already on v3.1 is unchanged in value."""
    out = dict(rec)
    if "p_n1" in out:
        out["p_headline_v3_stored"] = out.get("p_headline")
        out["p_headline"] = out["p_n1"]
    sub = out.get("subset")
    if sub:
        sub = dict(sub)
        sub["p_headline_v3_stored"] = sub.get("p_headline")
        sub["p_headline"] = sub["p_n1"]
        out["subset"] = sub
    return out


ROBUSTNESS_CAVEAT = (
    "REPORTED caveat, not a re-tuning: the size at vol-of-vol multipliers above 1 measures how the "
    "N1 test behaves when the log-volatility innovation is larger than in the calibration "
    "generator (a convexity drift can make the simple-return P&L over sigma21 larger on spike "
    "dates than on random dates for long positions, so rejections are anti-conservative there; a "
    "non-rejection is unaffected). Only the default arm (multiplier 1) is the size claim; the "
    "other arms never gate and no definition is changed in response to them.")


def _arm(rec):
    return float(rec.get("vol_of_vol_multiplier", 1.0))


def report_from_records(cfg, records, elapsed, finished):
    """The whole report from a config dict (datasets, power_datasets, effects,
    B, n_series, sessions, seed, workers, vol_of_vol_multipliers, smoke) and the
    raw dataset records. Every record's headline is (re)stated as its N1 p
    first (spec v3.1). The size claim is the DEFAULT arm (vol-of-vol multiplier
    1, else the smallest present); further arms are reported under
    `size_by_vol_of_vol_multiplier` and never gate. ANY errored dataset (in any
    arm or in the power run) makes `verdict.run_ok` false."""
    records = [restate_headline(r) if "error" not in r else r for r in records]
    size_all = [r for r in records if r.get("kind") == "size"]
    arms = sorted({_arm(r) for r in size_all})
    default_arm = 1.0 if (1.0 in arms or not arms) else arms[0]
    size = [r for r in size_all if _arm(r) == default_arm]
    power = {}
    for r in records:
        if r.get("kind") == "power":
            power.setdefault(float(r["effect"]), []).append(r)
    errors = [r for r in records if "error" in r]
    size_sum = sfn.summarize_size(size)
    power_sum = sfn.summarize_power(power) if power else None
    sub_size = subset_view([r for r in size if "error" not in r])
    sub_power = {e: subset_view([r for r in rs if "error" not in r]) for e, rs in power.items()}
    n2_row, n2_l4l = size_sum["p_n2"], size_sum["p_n2_covered"]
    report = {
        "spec_version": SPEC_VERSION, "headline_definition": HEADLINE_DEFINITION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claim_run": not bool(cfg.get("smoke", False)),
        "finished": bool(finished), "elapsed_seconds": round(elapsed, 1),
        "args": {"datasets": cfg.get("datasets"), "power_datasets": cfg.get("power_datasets"),
                 "effects": list(cfg.get("effects", [])), "B": cfg.get("B"),
                 "n_series": cfg.get("n_series"), "sessions": cfg.get("sessions"),
                 "seed": cfg.get("seed"), "workers": cfg.get("workers"),
                 "vol_of_vol_multipliers": list(cfg.get("vol_of_vol_multipliers", [1.0])),
                 "smoke": bool(cfg.get("smoke", False))},
        "spec": {"size_window": list(sfn.SIZE_WINDOW), "alpha": sfn.CALIB_ALPHA,
                 "min_size_datasets": 300, "min_power_datasets": 50, "min_claim_B": MIN_CLAIM_B,
                 "size_required_for": list(sfn.SIZE_REQUIRED),
                 "size_reported_only": list(sfn.SIZE_REPORTED_ONLY), "n2_label": N2_LABEL,
                 "power_required": {"effect": sfn.POWER_REQUIRED_AT[0],
                                    "power": sfn.POWER_REQUIRED_AT[1]}},
        "generator_note": sfn.CALIB_ETA_DOC,
        "size": size_sum, "size_default_vol_of_vol_multiplier": default_arm,
        "size_subset_first_7_series": sfn.summarize_size(sub_size) if sub_size else None,
        "n2_coverage_size_datasets": dict(n2_coverage(size), label=N2_LABEL),
        "power": power_sum,
        "power_subset_first_7_series": (
            sfn.summarize_power(sub_power) if sub_power and all(sub_power.values()) else None),
        "verdict": {
            "run_ok": len(errors) == 0,
            "size_datasets_ok": size_sum["n_datasets"] >= 300 and len(errors) == 0,
            "size_n1_in_window": size_sum["p_n1"]["in_window"],
            "size_headline_in_window": size_sum["p_headline"]["in_window"],
            "size_required_in_window": size_sum["all_in_window"],
            "size_n2_sensitivity": {
                "label": N2_LABEL, "gating": False,
                "rate": n2_row["rate"], "n_available": n2_row["n_available"],
                "in_window_informational": n2_row["in_window"],
                "like_for_like_rate": n2_l4l["rate"],
                "like_for_like_n_available": n2_l4l["n_available"],
                "like_for_like_in_window_informational": n2_l4l["in_window"]},
            "power_at_1_passes": None if power_sum is None else power_sum["passes_required_power"],
            "n_errors": len(errors)},
        "open_choices": list(sfn.OPEN_CHOICES),
        "errors": errors,
        "records": {"size": size_all, "power": {str(e): rs for e, rs in sorted(power.items())}},
    }
    if len(arms) > 1:
        report["size_by_vol_of_vol_multiplier"] = {
            str(a): sfn.summarize_size([r for r in size_all if _arm(r) == a]) for a in arms}
        report["robustness_caveat"] = ROBUSTNESS_CAVEAT
        report["robustness_summary_lines"] = robustness_lines(report["size_by_vol_of_vol_multiplier"])
    return report


def robustness_lines(by_arm):
    """Short plain-text summary of the size at each vol-of-vol arm: N1 (= the
    headline) rate with its standard error, rejections and window flag."""
    lines = ["Size of the N1 test (rate at 0.10 over no-edge datasets) by vol-of-vol multiplier; "
             "window [0.06, 0.14]; the default arm is the size claim, the others a reported caveat:"]
    for a, sz in by_arm.items():
        d = sz["p_n1"]
        rate = "n/a" if d["rate"] is None else f"{d['rate']:.4f}"
        se = "n/a" if d["se"] is None else f"{d['se']:.4f}"
        lines.append(f"  multiplier {a}: N1 size {rate} (SE {se}), {d['rejections']} of "
                     f"{sz['n_datasets']} datasets, in window: {d['in_window']}")
    lines.append(ROBUSTNESS_CAVEAT)
    return lines


def build_report(args, records, elapsed, finished):
    """`report_from_records` from an argparse namespace of a fresh run."""
    cfg = {"datasets": args.datasets, "power_datasets": args.power_datasets,
           "effects": list(args.effects), "B": args.B, "n_series": args.n_series,
           "sessions": args.sessions, "seed": args.seed, "workers": args.workers,
           "vol_of_vol_multipliers": [float(v) for v in getattr(args, "vol_of_vol_multipliers", [1.0])],
           "smoke": bool(getattr(args, "smoke", False))}
    return report_from_records(cfg, records, elapsed, finished)


def resummarize(path, summary_out=None, include_records=False):
    """Re-summarize an existing calibration JSON under spec v3.1 from its raw
    p-values (no re-run): the headline is restated as N1, all summaries are
    recomputed, the v3-stored headline size is kept as provenance, and the
    summary is written to `summary_out` (default: null_calibration_v3_1_summary.json
    next to `path`). Returns the summary dict."""
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    size = list(stored["records"]["size"])
    power = stored["records"].get("power", {})
    records = size + [r for rs in power.values() for r in rs]
    v3 = sfn.summarize_size([r for r in size if "error" not in r])["p_headline"]
    v3 = {k: v for k, v in v3.items() if k != "gating"}       # a superseded definition gates nothing
    report = report_from_records(stored.get("args", {}), records, stored.get("elapsed_seconds", 0.0),
                                 stored.get("finished", True))
    report["resummarized_from"] = os.path.abspath(path)
    report["v3_stored_headline_size"] = dict(v3, note="headline as stored in the input file "
                                             "(spec v3 took max(N1, N2)); superseded by v3.1")
    if not include_records:
        del report["records"]
    out = summary_out or os.path.join(os.path.dirname(os.path.abspath(path)), SUMMARY_NAME)
    write_json_atomic(out, report)
    return report


def print_summary(report):
    s = report["size"]
    print(f"[size] spec v{report.get('spec_version', '?')}; headline = {report.get('headline_definition')}")
    print(f"[size] datasets={s['n_datasets']} errors={s['n_errors']} window={s['window']} "
          f"(required for {s['required']}; N2 rows are {N2_LABEL})")
    for k in ("p_n1", "p_headline", "p_n2", "p_n2_covered"):
        d = s[k]
        rate = "n/a" if d["rate"] is None else f"{d['rate']:.4f}"
        se = "n/a" if d["se"] is None else f"{d['se']:.4f}"
        tag = "REQUIRED" if d["gating"] else N2_LABEL
        print(f"  {k:13s} [{tag}] available={d['n_available']:4d} rejections={d['rejections']:4d} "
              f"rate={rate} se={se} in_window={d['in_window']}")
    if report.get("size_subset_first_7_series"):
        ss = report["size_subset_first_7_series"]
        print("[size, first 7 series] " + "  ".join(
            f"{k}={ss[k]['rate']:.4f}" if ss[k]["rate"] is not None else f"{k}=n/a"
            for k in ("p_n1", "p_headline", "p_n2", "p_n2_covered")))
    if report.get("n2_coverage_size_datasets"):
        print(f"[n2 coverage, {N2_LABEL}] {report['n2_coverage_size_datasets']}")
    p = report["power"]
    if p:
        print("[power] effect -> realized mean x, headline (= N1) power; N2 shown as sensitivity")
        for e, c in p["curve"].items():
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"  {e:5.2f}: n={c['n_datasets']:3d} realized={fmt(c['mean_realized_x'])} "
                  f"headline={fmt(c['power_headline'])} n1={fmt(c['power_n1'])} "
                  f"n2({N2_LABEL})={fmt(c['power_n2'])}")
        print(f"  min detectable effect: 50% -> {p['min_detectable_effect_50']}, "
              f"80% -> {p['min_detectable_effect_80']}; "
              f"requirement (>= {p['required']['power']} at {p['required']['effect']}) "
              f"passes={p['passes_required_power']}")
    for line in report.get("robustness_summary_lines", []):
        print(f"[robustness] {line}")
    print(f"[claim_run] {report.get('claim_run')}")
    print(f"[verdict] {report['verdict']}")


def failure_line(report, total):
    """The loud line for a run with errored datasets, or None."""
    n = report["verdict"]["n_errors"]
    if not n:
        return None
    first = report["errors"][0].get("error", "?")
    return (f"[FAILED] {n} of {total} datasets errored (recorded as failures, not dropped; exit "
            f"code 1); first error: {first}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", type=int, default=300, help="no-edge datasets for the size run")
    ap.add_argument("--power-datasets", type=int, default=60,
                    help="datasets per injected effect (0 skips the power run)")
    ap.add_argument("--effects", type=float, nargs="+", default=list(sfn.CALIB_EFFECTS),
                    help="injected reversion, in sigma21 per trade")
    ap.add_argument("--B", type=int, default=200, help="null replications per dataset (>= 200)")
    ap.add_argument("--workers", type=int, default=4, help=f"processes (at most {MAX_WORKERS})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master dataset seed")
    ap.add_argument("--n-series", type=int, default=sfn.CALIB_SERIES)
    ap.add_argument("--sessions", type=int, default=sfn.CALIB_SESSIONS)
    ap.add_argument("--out", default=DEFAULT_OUT, help="report JSON path")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--resummarize", metavar="JSON", default=None,
                    help="re-summarize an existing calibration JSON under spec v3.1 (no re-run)")
    ap.add_argument("--summary-out", default=None,
                    help=f"where --resummarize writes (default: {SUMMARY_NAME} next to the input)")
    ap.add_argument("--include-records", action="store_true",
                    help="keep the raw records in the --resummarize summary")
    ap.add_argument("--vol-of-vol-multipliers", type=float, nargs="+", default=[1.0],
                    help="robustness arms: multiplier(s) of the log-volatility innovation of the "
                         "generator (default 1 = the v3.1 calibration generator)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"tiny test run: allows --B below {MIN_CLAIM_B}; the report says claim_run false")
    args = ap.parse_args(argv)
    args.workers = clamp_workers(args.workers)
    if not args.resummarize:
        problem = validate_args(args)
        if problem:
            ap.error(problem)

    if args.resummarize:
        report = resummarize(args.resummarize, args.summary_out, args.include_records)
        print_summary(report)
        print(f"[done] wrote {args.summary_out or os.path.join(os.path.dirname(os.path.abspath(args.resummarize)), SUMMARY_NAME)}",
              flush=True)
        fail = failure_line(report, len(report["errors"]) + report["size"]["n_datasets"])
        if fail:
            print(fail, flush=True)
            return 1
        return 0

    tasks = build_tasks(args)
    print(f"[start] {len(tasks)} datasets ({args.datasets} size x "
          f"{len(args.vol_of_vol_multipliers)} arm(s) + "
          f"{args.power_datasets} x {len(args.effects)} power), B={args.B}, "
          f"series={args.n_series}, sessions={args.sessions}, workers={args.workers}, "
          f"out={args.out}", flush=True)
    t0 = time.time()
    records = []
    with mp.Pool(processes=args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(sfn.calibration_task, tasks, chunksize=1), 1):
            records.append(rec)
            if i % max(1, args.checkpoint_every) == 0 or i == len(tasks):
                write_json_atomic(args.out, build_report(args, records, time.time() - t0,
                                                         finished=(i == len(tasks))))
            if i % 10 == 0 or i == len(tasks):
                print(f"[progress] {i}/{len(tasks)} datasets, {time.time() - t0:.0f}s", flush=True)
    report = build_report(args, records, time.time() - t0, finished=True)
    write_json_atomic(args.out, report)
    print_summary(report)
    if report.get("robustness_summary_lines"):
        txt = os.path.splitext(args.out)[0] + "_summary.txt"
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report["robustness_summary_lines"]) + "\n")
        print(f"[done] wrote {txt}", flush=True)
    print(f"[done] wrote {args.out}", flush=True)
    fail = failure_line(report, len(tasks))
    if fail:
        print(fail, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
