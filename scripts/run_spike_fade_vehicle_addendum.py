"""An optional POST-HOC vehicle-test addendum to the range-bound spike-fade study
(pre-registration: the study's specification,
version 3.1; the study's output folder is the --out folder).

WHAT IT IS. The pre-registered design runs the tradable-vehicle test only for
the pair that attains T* and the top three pairs by plateau score. This
optional addendum runs the same vehicle test on a FIXED list, whatever
the study's own selection: VIX up_fade (short a VIX spike, expressed as LONG
SVXY) and oil up_fade (short an oil spike, expressed as SHORT USO on WTI spot
and SHORT BNO on Brent spot). It is outside the pre-registered design, so it
labels everything POST-HOC.

THE FIXED LIST (nothing else is ever run; the guard refuses anything outside it,
before any data is touched and again in the lowest-level helper):
  pairs     VIX up_fade on SVXY (long); WTI spot up_fade on USO (short);
            Brent spot up_fade on BNO (short)
  triggers  z20@2.0, z20@2.5, pct756, pctexp
  cells     E0 | H = 21 | TP none | SL none  (the pre-registered primary cell)
            E0 | H = 63 | TP none | SL none  (the long-horizon cell)
  = 3 x 4 x 2 = 24 vehicle tests. No tuning, no other cell, no other statistic.

HOW. It reuses the study's own code WITHOUT editing it: the driver's cached-data
helpers (SHA-256 verified raw files, the sealed loader at the discovery cutoff
2023-06-30, the series and vehicle builders), the core's trigger and episode
machinery for the series-level cell numbers (episodes n, mean x, t: the same
numbers as cells.csv, which the script reconciles against), and the analysis
module's `vehicle_test_for_pair` for the vehicle test, with the null module's
frozen constants: B = 500 replications, the vehicle-null seed 2026-09-23 (the
core's N1 seed, that date written as one number without dashes; the stream is
keyed by the vehicle's name), exactly as the
study's own vehicle tests ran. NO network: cached raw data only. Nothing after
the cutoff is loaded (the sealed loader enforces it; this script has no way to
move the cutoff).

OUTPUTS (in the study folder, or --out): ADDENDUM-post-hoc-vehicle-tests.md
and addendum_results.json. An existing output is not overwritten without --force.
`--render-only` rebuilds the markdown from addendum_results.json, recomputing
nothing.

Exit codes: 0 done, 1 unexpected error, 2 refused (outputs already exist, or
nothing to render), 3 failed closed (cached data missing, altered or unusable),
4 a series-level number does not reconcile with cells.csv (or a vehicle series
with the study's results.json).

Short positions ignore borrow costs; no trading cost of any kind is charged. No
trade is placed, no ledger row is written, no trial slot is spent, and no dollar
figure is written anywhere (the writer refuses a dollar sign followed by a digit).

USAGE (Windows, from the repository root):
  python scripts/run_spike_fade_vehicle_addendum.py
"""
import os

# Keep BLAS single-threaded per process BEFORE numpy loads (same as the driver).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import math
import re
import sys
import traceback
from collections import namedtuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _p in (SRC_ROOT, SCRIPTS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_spike_fade_study as drv  # noqa: E402
from mining import spike_fade_analysis as sfa  # noqa: E402
from mining import spike_fade_null as sfn  # noqa: E402
from mining import spike_fade_study as sfs  # noqa: E402

DEFAULT_OUT = drv.DEFAULT_OUT
OUT_MD = "ADDENDUM-post-hoc-vehicle-tests.md"
OUT_JSON = "addendum_results.json"
SCHEMA = "spike-fade-vehicle-addendum-v1"
SPEC_FILE = drv.SPEC_FILE
SPEC_VERSION = drv.SPEC_VERSION

NULL_B = sfn.NULL_B                              # 500, the study's own B
VEHICLE_SEED = sfs.N1_SEED                       # 2026_09_23, the frozen N1 seed
# No headline p is pinned in this file: it is read from the study's own results.json,
# and reported as not available when that file is missing or unreadable.
CONFIRMATORY_HEADLINE_P = None
MIN_VEHICLE_N = sfa.CANDIDATE_MIN_VEHICLE_N      # 8: fewer vehicle trades = underpowered
LOOKS_GOOD_PERCENTILE = 0.90                     # "beat 9 of 10 random tries"
RECONCILE_REL_TOL = 1e-9
RECONCILE_ABS_TOL = 1e-12

# ---------------------------------------------------------------------------
# The fixed list
# ---------------------------------------------------------------------------

FIXED_PAIRS = (
    ("^VIX", "up_fade", "SVXY", "long"),
    ("WTI spot", "up_fade", "USO", "short"),
    ("Brent spot", "up_fade", "BNO", "short"),
)
FIXED_TRIGGERS = ("z20@2.0", "z20@2.5", "pct756", "pctexp")
FIXED_CELLS = (("E0", 21, "none", None), ("E0", 63, "none", None))
CELL_NOTES = {21: "the pre-registered primary cell", 63: "the long-horizon cell"}

PlanItem = namedtuple("PlanItem", "series side ticker position trigger cell")


class OutsideFixedList(Exception):
    """A pair, side, vehicle, trigger or cell outside the fixed list of this
    addendum was requested."""


class ReconciliationError(Exception):
    """A number computed here does not agree with the finished study's files."""


def seed_text(seed: int) -> str:
    """A seed that is a date written as one number (yyyymmdd), shown as that date
    (2026-09-23) so it reads as the date it encodes; the bare number itself is
    never written to an output."""
    seed = int(seed)
    return f"{seed // 10000:04d}-{seed // 100 % 100:02d}-{seed % 100:02d}"


def check_against_study_tables() -> None:
    """The fixed list must agree with the study's own vehicle table and primary
    cell (a drift in either would silently change what these tests mean)."""
    for series, side, ticker, position in FIXED_PAIRS:
        v = sfa.VEHICLE_TABLE[(series, side)]
        if (v.ticker, v.position) != (ticker, position):
            raise RuntimeError(f"the fixed list disagrees with the study's vehicle table for "
                               f"{series} {side}: {(ticker, position)} vs {(v.ticker, v.position)}")
    if FIXED_CELLS[0] != sfs.PRIMARY_CELL:
        raise RuntimeError("the first fixed cell is not the study's pre-registered primary cell")
    if tuple(FIXED_TRIGGERS) != tuple(sfs.TRIGGERS):
        raise RuntimeError("the fixed triggers are not the study's four triggers")
    if NULL_B != 500 or VEHICLE_SEED != 2026_09_23:
        raise RuntimeError("the frozen B or seed of the null module changed")


def build_plan() -> list:
    """The 24 fixed tests, in a stable order: pair, then trigger, then cell."""
    check_against_study_tables()
    return [PlanItem(series, side, ticker, position, trigger, cell)
            for (series, side, ticker, position) in FIXED_PAIRS
            for trigger in FIXED_TRIGGERS
            for cell in FIXED_CELLS]


def check_allowed(series, side, ticker, trigger, cell, position=None) -> None:
    """Refuse anything outside the fixed list (OutsideFixedList)."""
    pair = next((p for p in FIXED_PAIRS if p[0] == series and p[1] == side and p[2] == ticker),
                None)
    if pair is None:
        raise OutsideFixedList(f"pair ({series!r}, {side!r}, vehicle {ticker!r}) is not in the "
                               "fixed list of this addendum")
    if position is not None and position != pair[3]:
        raise OutsideFixedList(f"position {position!r} is not the fixed position of {series} "
                               f"{side} on {ticker} ({pair[3]!r})")
    if trigger not in FIXED_TRIGGERS:
        raise OutsideFixedList(f"trigger {trigger!r} is not in the fixed list {FIXED_TRIGGERS}")
    try:
        cell_t = tuple(cell)
    except TypeError:
        raise OutsideFixedList(f"cell {cell!r} is not in the fixed list") from None
    if cell_t not in FIXED_CELLS:
        raise OutsideFixedList(f"cell {cell!r} is not in the fixed list {FIXED_CELLS}")


def validate_plan(plan) -> None:
    """Every item must be inside the fixed list (a subset is fine)."""
    for item in plan:
        check_allowed(item.series, item.side, item.ticker, item.trigger, item.cell, item.position)


def sl_text(sl) -> str:
    return sfs.sl_label(sl)


def cell_dict(cell) -> dict:
    entry, H, tp, sl = cell
    return {"entry": entry, "H": int(H), "tp": tp, "sl": sl_text(sl)}


def cell_label(cell) -> str:
    entry, H, tp, sl = cell
    return f"{entry} | H = {int(H)} | TP {tp} | SL {sl_text(sl)}"


# ---------------------------------------------------------------------------
# Running the tests
# ---------------------------------------------------------------------------

def run_one_test(series, vehicle, item, group, *, B, seed=VEHICLE_SEED) -> dict:
    """One fixed test. `series` is the trigger series, `vehicle` the vehicle's
    sealed series, `group` the series' GroupResult for this item's trigger and
    side (its episodes and its series-level cell statistics). The guard runs
    first, so this helper refuses an out-of-list item on its own."""
    check_allowed(item.series, item.side, item.ticker, item.trigger, item.cell, item.position)
    cell = tuple(item.cell)
    cs = group.stats[cell]
    series_level = {"n": int(cs.n), "eligible": bool(cs.eligible), "mean_x": cs.mean_x, "t": cs.t,
                    "n_episodes_kept": int(group.episodes.n_kept),
                    "n_episodes_before_censoring": int(group.episodes.n_episodes),
                    "dropped_censored": int(group.episodes.dropped_censored)}
    vres = sfa.vehicle_test_for_pair(item.series, item.side, vehicle, group.episodes.dates, cell,
                                     B=B, seed=seed, extra_lag=series.extra_lag)
    return {"pair": f"{item.series} {item.side} on {item.ticker} ({item.position})",
            "series": item.series, "side": item.side, "vehicle": item.ticker,
            "position": item.position, "trigger": item.trigger, "cell": cell_dict(cell),
            "cell_label": cell_label(cell), "series_extra_lag": int(series.extra_lag),
            "series_level": series_level, "vehicle_result": drv._vehicle_result_dict(vres)}


def run_fixed_tests(series_by_name, vehicles_by_ticker, plan=None, *, B=NULL_B, seed=VEHICLE_SEED,
                    log=None) -> list:
    """Run the plan (default: all 24). The plan is validated BEFORE any data is
    touched. Triggers are computed once per series and each (series, trigger)
    group is evaluated once, only in the fixed cells."""
    log = log or (lambda m: None)
    plan = build_plan() if plan is None else list(plan)
    validate_plan(plan)
    signals, groups, out = {}, {}, []
    for item in plan:
        series = series_by_name[item.series]
        key = (item.series, item.trigger)
        if key not in groups:
            if item.series not in signals:
                signals[item.series] = sfs.compute_triggers(series)
            groups[key] = sfs.evaluate_group(series, item.trigger, item.side, cells=FIXED_CELLS,
                                             signals=signals[item.series], bootstrap=False)
        row = run_one_test(series, vehicles_by_ticker[item.ticker], item, groups[key], B=B,
                           seed=seed)
        vr = row["vehicle_result"]
        log(f"{item.series} {item.trigger} {cell_label(item.cell)} on {item.ticker}: series n = "
            f"{row['series_level']['n']}, vehicle n = {vr['n']}, excess "
            f"{fmt_pp(vr['excess_pp'])} pp, beat share {fmt_share(vr['null_percentile'])}")
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_DOLLAR_DIGIT = re.compile(r"\$\s*\d")


def _isnum(x) -> bool:
    return (isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def fmt_num(x, nd: int = 2) -> str:
    return f"{float(x):.{nd}f}" if _isnum(x) else "n/a"


def fmt_pp(x) -> str:
    return f"{float(x):+.2f}" if _isnum(x) else "n/a"


def fmt_signed(x, nd: int = 3) -> str:
    return f"{float(x):+.{nd}f}" if _isnum(x) else "n/a"


def fmt_share(x) -> str:
    return f"{float(x):.2f}" if _isnum(x) else "n/a"


def fmt_pct(x) -> str:
    return f"{100.0 * float(x):.0f}%" if _isnum(x) else "n/a"


def assert_no_dollar_digit(text: str) -> None:
    """Never write a dollar sign followed by a digit: refuse to write one."""
    m = _DOLLAR_DIGIT.search(text)
    if m:
        raise ValueError("refusing to write a dollar sign followed by a digit: "
                         f"{text[max(0, m.start() - 20): m.end() + 20]!r}")


def headline_text(p) -> str:
    """The study's confirmatory headline as a phrase: the p value and what the
    study's own decision rule says about it, or "not available" when no value
    was read from the study's results.json."""
    if not _isnum(p):
        return "not available (the study's results.json was not read)"
    verdict = "not convincing" if float(p) > sfn.HEADLINE_P_MAX else "meets the study's bar"
    return f"p = {float(p):.3f} ({verdict})"


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary counts and the plain-language bullets
# ---------------------------------------------------------------------------

def _powered(vr) -> bool:
    return not bool(vr.get("underpowered_on_vehicle"))


def _looks_good(vr) -> bool:
    return bool(_powered(vr) and _isnum(vr.get("excess_pp")) and float(vr["excess_pp"]) > 0
                and _isnum(vr.get("null_percentile"))
                and float(vr["null_percentile"]) >= LOOKS_GOOD_PERCENTILE)


def summarize(res) -> dict:
    """Counts only, over the fixed list's own results (no new statistic): how many
    tests are underpowered, how many look good (positive excess, at least 8
    vehicle trades, beat at least 90 percent of the random tries), how many
    would be expected to look good by luck alone, and the same per pair."""
    tests = res["tests"]
    n = len(tests)
    per_pair = {}
    for t in tests:
        vr = t["vehicle_result"]
        st = per_pair.setdefault(t["series"], {
            "vehicle": t["vehicle"], "position": t["position"], "n_tests": 0, "n_powered": 0,
            "n_underpowered": 0, "n_positive": 0, "n_looks_good": 0, "excess_min": None,
            "excess_max": None})
        st["n_tests"] += 1
        if _powered(vr):
            st["n_powered"] += 1
            if _isnum(vr.get("excess_pp")):
                x = float(vr["excess_pp"])
                st["n_positive"] += int(x > 0)
                st["excess_min"] = x if st["excess_min"] is None else min(st["excess_min"], x)
                st["excess_max"] = x if st["excess_max"] is None else max(st["excess_max"], x)
        else:
            st["n_underpowered"] += 1
        st["n_looks_good"] += int(_looks_good(vr))
    return {"n_tests": n,
            "n_powered": sum(s["n_powered"] for s in per_pair.values()),
            "n_underpowered": sum(s["n_underpowered"] for s in per_pair.values()),
            "n_positive": sum(s["n_positive"] for s in per_pair.values()),
            "n_looks_good": sum(s["n_looks_good"] for s in per_pair.values()),
            "looks_good_percentile": LOOKS_GOOD_PERCENTILE,
            "expected_by_luck": round((1.0 - LOOKS_GOOD_PERCENTILE) * n, 6),
            "per_pair": per_pair}


_PAIR_LEAD = {
    "^VIX": "After VIX spikes, buying SVXY (a fund that rises when the VIX falls back)",
    "WTI spot": "After WTI oil spikes, shorting USO (a fund that follows oil; a short bets the "
                "price will fall)",
    "Brent spot": "After Brent oil spikes, shorting BNO (a fund that follows Brent oil)",
}


def _of_them(k: int, n: int) -> str:
    """'all 8 of them', 'none of them' or '3 of them'."""
    if n > 0 and k == n:
        return f"all {n} of them"
    return "none of them" if k == 0 else f"{k} of them"


def _pair_sentence(lead: str, st: dict) -> str:
    if st["n_powered"] == 0:
        return (f"{lead}: none of the {st['n_tests']} tests had 8 or more trades, which is too "
                "few to judge anything.")
    have = (f"all {st['n_tests']} tests" if st["n_powered"] == st["n_tests"]
            else f"{st['n_powered']} of the {st['n_tests']} tests")
    s = (f"{lead}: {have} had enough trades (8 or more) to judge. The fund earned more per "
         f"trade than the average of random entry dates in "
         f"{_of_them(st['n_positive'], st['n_powered'])}, and beat at least 9 out of 10 random "
         f"tries in {_of_them(st['n_looks_good'], st['n_powered'])}. The gap against random dates "
         f"ran from {fmt_pp(st['excess_min'])} to {fmt_pp(st['excess_max'])} percentage points "
         "per trade.")
    if st["n_looks_good"] == 0 and st["n_positive"] == 0:
        s += (" In every one of these tests the position did worse than random entry dates, so "
              "the answer here is no.")
    elif st["n_looks_good"] == 0 and st["n_positive"] * 2 <= st["n_powered"]:
        s += (" None of them clearly beat random dates, so the answer here is no (that is not "
              "proof that no edge exists).")
    elif st["n_looks_good"] == 0:
        s += (" It leans better than random dates but never by a clear margin, so the answer "
              "here is not convincing.")
    else:
        s += " Some of these tests look good (see the luck warning below)."
    return s


def plain_summary(res) -> list:
    """Three to five bullets, no jargon (unavoidable terms defined in place)."""
    s = summarize(res)
    pp = s["per_pair"]
    n, K, U = s["n_tests"], s["n_looks_good"], s["n_underpowered"]
    B = res["params"]["B"]
    p_head = res.get("confirmatory_headline", {}).get("p_headline", CONFIRMATORY_HEADLINE_P)
    if s["n_powered"] == 0:
        answer = "no test had enough trades (8 or more) to judge"
    elif K == 0:
        answer = (f"no. In none of the {s['n_powered']} tests with enough trades (8 or more) did "
                  "the fund clearly beat random entry dates (it never beat 9 out of 10 random "
                  "tries)")
    else:
        answer = (f"not clearly. {K} of the {s['n_powered']} tests with enough trades (8 or more) "
                  "beat at least 9 out of 10 random tries, but read the luck warning below")
    bullets = [
        (f"Short answer: {answer}. The question was: after a big jump in the VIX (the stock "
         "market's fear gauge) or in the "
         "price of oil, does the fund you could actually trade earn more than the same fund "
         f"entered on random dates? We ran {n} tests to find out: three funds, four ways of "
         "defining a spike, and two holding times (21 and 63 trading days, about one and about "
         f"three months). Random dates means {B} sets of randomly chosen entry dates in the same "
         "fund, with the same holding rule and the same number of trades."),
        _pair_sentence(_PAIR_LEAD["^VIX"], pp["^VIX"]) if "^VIX" in pp else "",
        " ".join(_pair_sentence(_PAIR_LEAD[k], pp[k]) for k in ("WTI spot", "Brent spot")
                 if k in pp),
        (f"The luck warning: with {n} tests, about {s['expected_by_luck']:.1f} of them (roughly 2 "
         "or 3) would land in the top tenth of the random tries by luck alone, even if nothing "
         f"real were going on. {K} of the {s['n_powered']} tests with enough trades did"
         + (", which is no more than luck alone would give."
            if K <= 3 else
            ", which is more than luck alone usually gives, but the tests reuse the same spikes "
            "and are not independent chances, so it is still not proof.")),
        ("Status and limits: this is an optional post-hoc addendum outside the study's "
         f"pre-registered design, so it cannot change the study's official result (its one pre-planned test: "
         f"{headline_text(p_head)}; p is the chance of a result this "
         "strong by luck alone if random entry dates did just as well). A short position bets "
         "on a price falling, and here it is counted without any borrowing fee. "
         + (f"{U} of the {n} tests had fewer than 8 trades and are marked underpowered (too few "
            "trades to trust)." if U else "Every test had at least 8 trades.")),
    ]
    return [b for b in bullets if b]


# ---------------------------------------------------------------------------
# Reconciliation against the finished study's own files
# ---------------------------------------------------------------------------

def _parse_float(text) -> float:
    text = "" if text is None else str(text).strip()
    return float("nan") if text == "" else float(text)


def _close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=RECONCILE_REL_TOL, abs_tol=RECONCILE_ABS_TOL)


def reconcile_series_level(tests, cells_csv_path) -> dict:
    """Check every test's series-level n, mean x and t against the same cell in
    the study's cells.csv (skipped, and said so, when there is no such file).
    Any difference raises ReconciliationError."""
    if not os.path.exists(cells_csv_path):
        return {"status": "skipped", "n_checked": 0, "mismatches": [],
                "reason": f"{os.path.basename(cells_csv_path)} not found"}
    index = {}
    with open(cells_csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["series"], r["trigger"], r["side"], r["entry"], int(float(r["H"])),
                   r["tp"], r["sl"])
            index[key] = r
    mismatches = []
    for t in tests:
        c, sl = t["cell"], t["series_level"]
        key = (t["series"], t["trigger"], t["side"], c["entry"], int(c["H"]), c["tp"], c["sl"])
        row = index.get(key)
        if row is None:
            mismatches.append({"key": list(key), "problem": "no such row in cells.csv"})
            continue
        got = {"n": int(sl["n"]), "mean_x": float("nan") if sl["mean_x"] is None
               else float(sl["mean_x"]), "t": float("nan") if sl["t"] is None else float(sl["t"])}
        want = {"n": int(float(row["n"])), "mean_x": _parse_float(row["mean_x"]),
                "t": _parse_float(row["t"])}
        for field in ("n", "mean_x", "t"):
            ok = got[field] == want[field] if field == "n" else _close(got[field], want[field])
            if not ok:
                mismatches.append({"key": list(key), "field": field, "computed": got[field],
                                   "cells_csv": want[field]})
    if mismatches:
        raise ReconciliationError(
            f"{len(mismatches)} series-level number(s) do not match cells.csv: "
            + json.dumps(drv.json_safe(mismatches[:5])))
    return {"status": "checked", "n_checked": len(tests), "mismatches": []}


def reconcile_vehicle_series(vehicle_series_out, study_results_path) -> dict:
    """Check each vehicle series' row count and first and last date against the
    study's own results.json (skipped, and said so, when there is none)."""
    if not os.path.exists(study_results_path):
        return {"status": "skipped", "n_checked": 0, "mismatches": [],
                "reason": "results.json not found"}
    with open(study_results_path, encoding="utf-8") as f:
        study = json.load(f)
    ref = study.get("vehicle_series") or {}
    mismatches, checked = [], 0
    for ticker, info in vehicle_series_out.items():
        want = ref.get(ticker)
        if want is None:
            mismatches.append({"vehicle": ticker, "problem": "not in the study's results.json"})
            continue
        checked += 1
        for field in ("n_rows", "first_date", "last_date"):
            if info[field] != want[field]:
                mismatches.append({"vehicle": ticker, "field": field, "computed": info[field],
                                   "study": want[field]})
    if mismatches:
        raise ReconciliationError("vehicle series do not match the study's results.json: "
                                  + json.dumps(drv.json_safe(mismatches[:5])))
    return {"status": "checked", "n_checked": checked, "mismatches": []}


def read_study_headline(study_results_path) -> dict:
    """The finished study's confirmatory headline p, read from its results.json
    (reported as not available, and said so, when the file is not there)."""
    if os.path.exists(study_results_path):
        try:
            with open(study_results_path, encoding="utf-8") as f:
                p = json.load(f)["headline"]["p_headline"]
            if _isnum(p):
                return {"p_headline": float(p), "source": "the study's results.json"}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return {"p_headline": CONFIRMATORY_HEADLINE_P,
            "source": "not available (the study's results.json was not read)"}


# ---------------------------------------------------------------------------
# Data: cached raw files only, sealed at the cutoff the moment they are parsed
# ---------------------------------------------------------------------------

def addendum_needs() -> list:
    """The raw files this addendum needs: the three trigger series and the
    three vehicles. Nothing else is read."""
    by = {s.name: s for s in drv.UNIVERSE}
    needs = [drv.Need("trigger", name, by[name].source, by[name].symbol)
             for name, _, _, _ in FIXED_PAIRS]
    needs += [drv.Need("vehicle", ticker, "yahoo", ticker) for _, _, ticker, _ in FIXED_PAIRS]
    return needs


class DataUnusable(Exception):
    """A cached file is missing, altered, unparseable or has no rows in the window."""


def load_universe(raw_dir: str, cutoff, log=None) -> dict:
    log = log or (lambda m: None)
    manifest_files = drv.load_manifest(os.path.join(raw_dir, "manifest.json"))
    needs = addendum_needs()
    frames, n_post = {}, {}
    for need in needs:
        try:
            raw = drv.load_cached(raw_dir, manifest_files, need)
            df = drv.parse_raw(need.source, raw)
        except (drv.CacheError, ValueError) as exc:
            raise DataUnusable(f"{need.source}:{need.symbol}: {exc}") from exc
        sealed, n = drv.seal_frame(df, cutoff)
        if len(sealed) == 0:
            raise DataUnusable(f"{need.source}:{need.symbol}: no rows on or before the cutoff")
        frames[(need.source, need.symbol)], n_post[(need.source, need.symbol)] = sealed, n
        log(f"loaded {need.source}:{need.symbol}: {len(sealed)} rows within the cutoff "
            f"({n} later rows removed by the seal)")
    by = {s.name: s for s in drv.UNIVERSE}
    series, vehicles = {}, {}
    try:
        for name, _, _, _ in FIXED_PAIRS:
            series[name] = drv.build_trigger_series(by[name], frames, cutoff, n_post)
        for _, _, ticker, _ in FIXED_PAIRS:
            vehicles[ticker] = drv.build_vehicle_series(ticker, frames, cutoff, n_post)
    except ValueError as exc:
        raise DataUnusable(f"unusable data: {exc}") from exc
    files_out = {}
    for need in needs:
        fname = drv.raw_filename(need.source, need.symbol)
        fr = frames[(need.source, need.symbol)]
        ent = manifest_files[fname]
        files_out[need.name] = {
            "kind": need.kind, "source": need.source, "symbol": need.symbol, "file": fname,
            "sha256": ent["sha256"], "retrieved_utc": ent.get("retrieved_utc"),
            "rows_within_cutoff": int(len(fr)),
            "rows_removed_by_the_seal": int(n_post[(need.source, need.symbol)]),
            "first_date": fr.index.min().strftime("%Y-%m-%d"),
            "last_date": fr.index.max().strftime("%Y-%m-%d")}
    vehicle_series_out, first_dates = {}, {}
    for ticker, vs in vehicles.items():
        vehicle_series_out[ticker] = {
            "n_rows": int(vs.n), "first_date": vs.dates[0].strftime("%Y-%m-%d"),
            "last_date": vs.dates[-1].strftime("%Y-%m-%d"), "counts": vs.counts()}
        table = next(v for (s, sd), v in sfa.VEHICLE_TABLE.items() if v.ticker == ticker)
        obs = frames[("yahoo", ticker)].index.min()
        first_dates[ticker] = {"table": table.first_date.strftime("%Y-%m-%d"),
                               "observed": obs.strftime("%Y-%m-%d"),
                               "matches": bool(obs == table.first_date)}
    return {"series": series, "vehicles": vehicles, "files": files_out,
            "vehicle_series": vehicle_series_out, "vehicle_first_dates": first_dates}


# ---------------------------------------------------------------------------
# The results dict and the markdown
# ---------------------------------------------------------------------------

def build_results(uni: dict, tests: list, *, B: int, reconciliation: dict,
                  vehicle_reconciliation: dict, headline: dict) -> dict:
    underpowered = [{"series": t["series"], "vehicle": t["vehicle"], "trigger": t["trigger"],
                     "H": t["cell"]["H"], "n": t["vehicle_result"]["n"]}
                    for t in tests if t["vehicle_result"]["n"] < MIN_VEHICLE_N]
    res = {
        "schema": SCHEMA, "status": "completed", "post_hoc": True,
        "spec": {"file": SPEC_FILE, "version": SPEC_VERSION},
        "cutoff": sfs.DISCOVERY_CUTOFF.strftime("%Y-%m-%d"),
        "params": {
            "B": int(B), "vehicle_null_seed": seed_text(VEHICLE_SEED),
            "vehicle_null_seed_name": "N1_SEED of the core and null modules, the date shown "
                                      "written as one number without dashes (the stream is keyed "
                                      "by the vehicle's name)",
            "n_tests": len(tests), "min_vehicle_n": MIN_VEHICLE_N,
            "looks_good_percentile": LOOKS_GOOD_PERCENTILE,
            "era_split": sfa.SVXY_ERA_SPLIT.strftime("%Y-%m-%d"),
            "fixed_pairs": [list(p) for p in FIXED_PAIRS],
            "fixed_triggers": list(FIXED_TRIGGERS),
            "fixed_cells": [cell_label(c) for c in FIXED_CELLS]},
        "confirmatory_headline": headline,
        "data": {"files": uni["files"], "labels": drv.DATA_LABELS},
        "vehicle_series": uni["vehicle_series"],
        "vehicle_first_dates": uni["vehicle_first_dates"],
        "reconciliation": reconciliation, "vehicle_reconciliation": vehicle_reconciliation,
        "tests": tests, "underpowered": underpowered}
    return res


_SERIES_WORD = {"^VIX": "VIX", "WTI spot": "WTI", "Brent spot": "Brent"}


def fund_history_bullet(res) -> str:
    """How much of each series' spike history predates its fund (a count taken from
    the anchor accounting of the first fixed trigger and cell), or an empty string
    when the accounting is not there."""
    first, fdates = {}, res.get("vehicle_first_dates") or {}
    for t in res["tests"]:
        if (t["trigger"] == FIXED_TRIGGERS[0] and t["cell"]["H"] == FIXED_CELLS[0][1]
                and t["series"] not in first):
            first[t["series"]] = t
    parts, starts = [], []
    for series, _side, ticker, _pos in FIXED_PAIRS:
        t = first.get(series)
        if t is None:
            continue
        vr = t["vehicle_result"]
        parts.append(f"{vr['dropped_no_session']} of {vr['n_anchors_input']} "
                     f"{_SERIES_WORD.get(series, series)} spikes")
        if ticker in fdates:
            starts.append(f"{ticker} on {fdates[ticker]['observed']}")
    if not parts:
        return ""
    return ("- **The funds are young, so the sample shrinks.** " +
            (f"They start trading: {', '.join(starts)}. " if starts else "") +
            f"With the {FIXED_TRIGGERS[0]} trigger, {', '.join(parts)} happened before the fund "
            "had any trading history, so they cannot be tested in the fund (see the anchor "
            "table). Vehicle n is therefore smaller than the series n.")


def _pair_rows(tests):
    by = {}
    for t in tests:
        by.setdefault(t["series"], []).append(t)
    return by


def _note(vr) -> str:
    if vr["n"] == 0:
        return "no trades"
    return f"UNDERPOWERED (n < {MIN_VEHICLE_N})" if vr["n"] < MIN_VEHICLE_N else ""


def render_markdown(res) -> str:
    p = res["params"]
    B, seed, n_tests = p["B"], p["vehicle_null_seed"], p["n_tests"]
    p_head = res.get("confirmatory_headline", {}).get("p_headline", CONFIRMATORY_HEADLINE_P)
    era = p.get("era_split", "2018-02-28")
    s = summarize(res)
    tests = res["tests"]
    by_pair = _pair_rows(tests)
    L = []
    L.append("# Addendum: optional post-hoc tradable-fund tests for the VIX and oil spike-fade pairs")
    L.append("")
    L.append("> **OPTIONAL POST-HOC VEHICLE TEST.** Not part of the pre-registered study. "
             f"{n_tests} tests are run outside the study's design, so about 2 or 3 would look good "
             "by luck alone even with no real effect. The study's confirmatory headline stays at "
             f"**{headline_text(p_head)}** and **this addendum cannot change that "
             "verdict.** Short positions ignore borrow costs.")
    if B != NULL_B:
        L.append(">")
        L.append(f"> **THIS IS A MACHINERY TEST, NOT THE REAL RUN: B = {B} instead of {NULL_B}.**")
    L.append("")
    L.append("## Plain-language summary")
    L.append("")
    for b in plain_summary(res):
        L.append(f"- {b}")
    L.append("")
    L.append("## Read this first")
    L.append("")
    L.append("- **POST-HOC and optional.** The study's design runs the "
             "tradable-fund test only for the pair that attains T* and the top three pairs by "
             "plateau score. VIX up_fade on SVXY, WTI up_fade on USO and Brent up_fade on BNO "
             "are a fixed list run as a separate addendum, whatever the study's own selection, "
             "and are not part of the study's pre-registered design.")
    L.append(f"- **{n_tests} tests were run** (3 pairs x 4 triggers x 2 cells). If none of them "
             "had any real effect, about 2 or 3 (a tenth of them) would still look good by luck "
             "alone. They are also not independent: the four triggers and the two holding times "
             "reuse the same spikes.")
    L.append(f"- **The confirmatory headline is {headline_text(p_head)}** (source: "
             f"{res.get('confirmatory_headline', {}).get('source', 'the finished study')}). "
             "This addendum cannot change that verdict, promote a candidate or spend a scoring "
             "slot.")
    L.append("- **Short positions ignore borrow costs.** USO and BNO are shorted here with no "
             "borrowing fee, no financing charge and no limit on shares available to short. No "
             "trading cost of any kind is charged on any test: every P&L figure is raw, before "
             "costs, so it is an upper bound.")
    L.append("- **Contaminated series.** The study marks VIX, WTI and Brent as contaminated: "
             "some discovery-window facts about them are already partly known, so these tests are "
             "confirmatory on the discovery window only.")
    hb = fund_history_bullet(res)
    if hb:
        L.append(hb)
    L.append("- **The funds are futures-based.** USO and BNO hold oil futures and SVXY holds "
             "short-VIX-futures exposure, so their returns include the cost or gain of rolling "
             "futures and differ from the spot or index series that trigger the tests. The tests "
             "use the fund's own prices, as the study's design says.")
    if res["underpowered"]:
        L.append(f"- **{len(res['underpowered'])} of the {n_tests} tests are underpowered** "
                 f"(fewer than {MIN_VEHICLE_N} vehicle trades); their numbers are listed but "
                 "should not be trusted. See the underpowered list below.")
    else:
        L.append(f"- No test had fewer than {MIN_VEHICLE_N} vehicle trades.")
    L.append("")
    L.append("## What was run")
    L.append("")
    L.append("Nothing outside this fixed list was run: no other pair, trigger, cell or statistic. "
             "The script refuses anything else.")
    L.append("")
    L.append(md_table(["pair (series, side)", "fund traded", "position"],
                      [[f"{a} {b}", c, ("long: gains if the fund rises" if d == "long"
                                        else "short: gains if the fund falls")]
                       for a, b, c, d in FIXED_PAIRS]))
    L.append("")
    L.append("- Triggers: " + ", ".join(FIXED_TRIGGERS) + " (the study's four).")
    L.append("- Cells (both enter at the close of the fund's next session after the signal, no "
             "profit target, no stop, and exit at the close H sessions later; for the two FRED "
             "oil series one extra session of publication lag is added, as in the study): "
             + "; ".join(f"{cell_label(c)} ({CELL_NOTES[c[1]]})" for c in FIXED_CELLS) + ".")
    L.append("- The vehicle test is the study's own `vehicle_test_for_pair` on the fund's own "
             "auto-adjusted closes and calendar; the trigger series only supplies the spike date.")
    L.append(f"- Random-date null: B = {B} replications (the null module's frozen B is "
             f"{NULL_B}), vehicle-null seed {seed} (the core's N1 seed, which is that date written as one "
             "number without dashes; the same seed the study's own vehicle tests used; the "
             "stream is keyed by the fund's name). Each replication "
             "draws the same number of trades from the fund's own eligible sessions (at least 20 "
             "sessions apart, matched on whether the entry price is above or below the fund's "
             "one-year median), evaluates the same cell and averages. The observed average is then "
             "ranked among the replication averages (mid-rank).")
    L.append("- Sealed window: every file is cut at the discovery cutoff "
             f"{res['cutoff']} the moment it is read; nothing later is used. Cached files only, "
             "no network.")
    L.append("")
    L.append("How to read the tables. **Series n / S / t** are the trigger series' own numbers at "
             "that cell (traded episodes; S = average result per trade in units of the market's "
             "own normal 21-day price swing, sigma21; t = S over its standard error), identical "
             "to cells.csv. **Vehicle n** is the number of trades in the fund after mapping the "
             "spike dates onto the fund's own calendar. **Mean P&L** is the average raw result "
             "per trade in percentage points of the money put in (before any cost). **Random-date "
             "avg** is the average over the random-date replications. **Excess** = Mean P&L minus "
             "Random-date avg (percentage points; and in sigma21 units of the fund). **Beat "
             "share** = the share of random-date averages the observed average beat (0.50 is a "
             "coin flip). **Every-day avg** = the same cell entered on EVERY eligible session of "
             "the fund (its unconditional average result at this cell).")
    L.append("")
    L.append("## Results")
    L.append("")
    for series, rows in by_pair.items():
        first = rows[0]
        L.append(f"### {series} {first['side']} on {first['vehicle']} ({first['position']})")
        L.append("")
        table = []
        for t in rows:
            vr, sl = t["vehicle_result"], t["series_level"]
            table.append([t["trigger"], f"H = {t['cell']['H']}", sl["n"], fmt_num(sl["mean_x"], 3),
                          fmt_num(sl["t"], 2), vr["n"], fmt_pp(vr["mean_pnl_pp"]),
                          fmt_pct(vr["hit_rate"]), fmt_pp(vr["null_mean_pnl_pp"]),
                          fmt_pp(vr["excess_pp"]), fmt_signed(vr["excess_sigma"], 3),
                          fmt_share(vr["null_percentile"]),
                          fmt_pp(vr["unconditional_mean_pnl_pp"]), _note(vr)])
        L.append(md_table(["trigger", "hold", "series n", "series S", "series t", "vehicle n",
                           "mean P&L (pp)", "hit rate", "random-date avg (pp)", "excess (pp)",
                           "excess (sigma21)", "beat share", "every-day avg (pp)", "note"], table))
        L.append("")
    L.append("Count over the whole list (a count only, not a new statistic): "
             f"{s['n_powered']} of {s['n_tests']} tests have at least {MIN_VEHICLE_N} vehicle "
             f"trades; {s['n_positive']} of those have a positive excess; "
             f"{s['n_looks_good']} of those beat at least {int(100 * LOOKS_GOOD_PERCENTILE)}% of the "
             f"random-date averages (about {s['expected_by_luck']:.1f} would be expected by luck "
             "alone if nothing were real).")
    L.append("")
    L.append("## Underpowered tests (fewer than 8 vehicle trades)")
    L.append("")
    if res["underpowered"]:
        for u in res["underpowered"]:
            L.append(f"- {u['series']} {u['trigger']}, H = {u['H']}, {u['vehicle']}: "
                     f"n = {u['n']}")
    else:
        L.append("None.")
    L.append("")
    L.append(f"## SVXY exposure eras (split by entry date at {era})")
    L.append("")
    L.append("SVXY changed from -1x to -0.5x exposure to the VIX futures index on this date. The "
             "pooled result above is the headline; the split is descriptive and not a gate. Each "
             "era's excess is its own average minus the POOLED random-date average.")
    L.append("")
    era_rows = []
    for t in by_pair.get("^VIX", []):
        e = t["vehicle_result"].get("era")
        if not e:
            continue
        pre, post = e["pre"], e["post"]
        era_rows.append([t["trigger"], f"H = {t['cell']['H']}",
                         pre["n"], fmt_pp(100.0 * pre["mean_pnl"] if _isnum(pre["mean_pnl"])
                                          else None), fmt_pct(pre["hit_rate"]),
                         fmt_pp(pre["excess_pp_vs_pooled_null"]),
                         post["n"], fmt_pp(100.0 * post["mean_pnl"] if _isnum(post["mean_pnl"])
                                           else None), fmt_pct(post["hit_rate"]),
                         fmt_pp(post["excess_pp_vs_pooled_null"])])
    if era_rows:
        L.append(md_table(["trigger", "hold", f"n (to {era})", "mean P&L (pp)", "hit rate",
                           "excess vs pooled (pp)", f"n (after {era})", "mean P&L (pp)",
                           "hit rate", "excess vs pooled (pp)"], era_rows))
    else:
        L.append("No era split available.")
    L.append("")
    L.append("## Where the spike dates went (anchor accounting)")
    L.append("")
    acc = []
    for t in tests:
        vr = t["vehicle_result"]
        acc.append([t["series"], t["trigger"], f"H = {t['cell']['H']}", vr["n_anchors_input"],
                    vr["dropped_no_session"], vr["dropped_short_history"], vr["dropped_past_end"],
                    vr["dropped_censored"], vr["dropped_duplicate"], vr["n_anchors_kept"],
                    vr["n_kept_not_traded"], vr["n"]])
    L.append(md_table(["series", "trigger", "hold", "spikes in", "no fund session",
                       "short fund history", "past fund end", "censored", "duplicate", "kept",
                       "kept, not traded", "vehicle n"], acc))
    L.append("")
    L.append("- **no fund session**: no trading day of the fund on or before the spike date (the "
             "fund did not exist yet).")
    L.append("- **short fund history**: fewer than 252 sessions of the fund up to the spike, so "
             "its one-year baselines cannot be measured.")
    L.append("- **past fund end**: after the publication lag the spike would fall beyond the "
             "fund's last session.")
    L.append("- **censored**: fewer than 70 sessions left after the spike inside the sealed "
             "window (the study's uniform censoring rule, applied on the fund's calendar).")
    L.append("- **duplicate**: two spikes landing on the same fund session, merged.")
    L.append("- **kept, not traded**: a kept spike whose trade could not be formed (an undefined "
             "volatility or a missing or non-positive price on its path).")
    L.append("")
    L.append("## Fund context")
    L.append("")
    ctx = {}
    for t in tests:
        vr = t["vehicle_result"]
        ctx.setdefault(t["vehicle"], (t["position"], vr))
    fd = res.get("vehicle_first_dates", {})
    L.append(md_table(["fund", "position", "first session in the data", "table first date",
                       "first eligible session", "last session",
                       "fund's average daily return (pp; the fund's own, not the position's)"],
                      [[tk, pos, fd.get(tk, {}).get("observed", "n/a"),
                        fd.get(tk, {}).get("table", "n/a"), vr.get("window_start") or "n/a",
                        vr.get("window_end") or "n/a",
                        fmt_num(vr.get("unconditional_mean_daily_return_pp"), 4)]
                       for tk, (pos, vr) in ctx.items()]))
    L.append("")
    L.append("The average daily return is the fund's unconditional simple daily return over its "
             "eligible window (the same window as the every-day averages).")
    L.append("")
    L.append("## Data, reconciliation and reproduction")
    L.append("")
    files = res["data"]["files"]
    L.append(md_table(["series or fund", "file", "SHA-256 (first 12)", "retrieved (UTC)",
                       "rows used", "first date", "last date", "rows removed by the seal"],
                      [[name, f["file"], f["sha256"][:12], f.get("retrieved_utc") or "n/a",
                        f["rows_within_cutoff"], f["first_date"], f["last_date"],
                        f["rows_removed_by_the_seal"]] for name, f in files.items()]))
    L.append("")
    L.append("- Data labels: " + "; ".join(f"{k}: {v}" for k, v in
                                            res["data"].get("labels", {}).items()) + ".")
    rc, vrc = res["reconciliation"], res["vehicle_reconciliation"]
    L.append(f"- Series-level n, S and t against the study's cells.csv: {rc['status']} "
             f"({rc['n_checked']} rows checked, {len(rc['mismatches'])} mismatches).")
    L.append(f"- Fund series (rows, first and last date) against the study's results.json: "
             f"{vrc['status']} ({vrc['n_checked']} checked, {len(vrc['mismatches'])} mismatches).")
    L.append("- Reproduce: `python "
             "scripts/run_spike_fade_vehicle_addendum.py --force` (deterministic: frozen seed, "
             "cached data).")
    spec = res.get("spec") or {"file": SPEC_FILE, "version": SPEC_VERSION}
    L.append(f"- Spec: {spec['file']} version {spec['version']}.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Running, writing, command line
# ---------------------------------------------------------------------------

def _write_text(path: str, text: str) -> None:
    assert_no_dollar_digit(text)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _json_text(payload) -> str:
    return json.dumps(drv.json_safe(payload), indent=1, allow_nan=False) + "\n"


def run_addendum(out: str, *, B: int = NULL_B, log=None) -> int:
    log = log or drv._default_log
    check_against_study_tables()
    cutoff = drv.check_cutoff(sfs.DISCOVERY_CUTOFF)
    raw_dir = os.path.join(out, "data", "raw")
    log(f"addendum start: 24 fixed tests, B = {B}, seed {seed_text(VEHICLE_SEED)}, cutoff {cutoff.date()}, "
        f"cached data in {raw_dir}")
    try:
        uni = load_universe(raw_dir, cutoff, log=log)
    except DataUnusable as exc:
        print(f"FAILED CLOSED: {exc}", flush=True)
        return 3
    plan = build_plan()
    tests = run_fixed_tests(uni["series"], uni["vehicles"], plan, B=B, log=log)
    if len(tests) != 24:
        raise RuntimeError(f"expected 24 tests, ran {len(tests)}")
    study_results = os.path.join(out, "results.json")
    try:
        reconciliation = reconcile_series_level(tests, os.path.join(out, "cells.csv"))
        vehicle_reconciliation = reconcile_vehicle_series(uni["vehicle_series"], study_results)
    except ReconciliationError as exc:
        print(f"RECONCILIATION FAILED: {exc}", flush=True)
        return 4
    log(f"reconciliation: series-level {reconciliation['status']}, vehicle series "
        f"{vehicle_reconciliation['status']}")
    res = build_results(uni, tests, B=B, reconciliation=reconciliation,
                        vehicle_reconciliation=vehicle_reconciliation,
                        headline=read_study_headline(study_results))
    text = _json_text(res)
    res_j = json.loads(text)                      # render from exactly what is written
    md = render_markdown(res_j)
    _write_text(os.path.join(out, OUT_JSON), text)
    _write_text(os.path.join(out, OUT_MD), md)
    log(f"wrote {OUT_JSON} and {OUT_MD} in {out}")
    return 0


def render_only(out: str) -> int:
    path = os.path.join(out, OUT_JSON)
    if not os.path.exists(path):
        print(f"REFUSED: {path} does not exist; there is nothing to render", flush=True)
        return 2
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    _write_text(os.path.join(out, OUT_MD), render_markdown(res))
    print(f"{OUT_MD} re-rendered from {path}", flush=True)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="the study folder: holds data/raw, results.json, cells.csv; receives "
                         "the two outputs (default: the spike-fade study driver's output folder)")
    ap.add_argument("--force", action="store_true",
                    help="replace existing outputs (a re-run after a defect must be deliberate)")
    ap.add_argument("--render-only", action="store_true",
                    help="rebuild the markdown from addendum_results.json; recompute nothing")
    return ap.parse_args(argv)


def main(argv=None, *, B: int = NULL_B) -> int:
    """`B` is a keyword-only argument for the tests of the machinery; the command
    line has no way to change it (the real run always uses the null module's 500)."""
    args = parse_args(argv)
    out = os.path.abspath(args.out)
    if args.render_only:
        return render_only(out)
    existing = [n for n in (OUT_MD, OUT_JSON) if os.path.exists(os.path.join(out, n))]
    if existing and not args.force:
        print(f"REFUSED: {', '.join(existing)} already exist in {out}; pass --force to replace "
              "them (a re-run after a defect must be deliberate)", flush=True)
        return 2
    try:
        return run_addendum(out, B=B)
    except Exception:                                    # an unexpected error is a crash, loudly
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
