"""REPORT.md writer of the range-bound spike-fade study, Stage 1.

WHAT THIS MODULE IS. A set of PURE functions that turn the study's result
dict (the same dict the driver saves as results.json; its schema is the
`make_results` builder in src/tests/test_spike_fade_report.py) into the
markdown report. It reads no file, computes no statistic, touches no network
and places no trade: every number in the report is a number the driver already
computed and stored. The driver decides WHAT was found; this module decides
only HOW it is said.

REPORT STRUCTURE (the decision table plus the reader-facing requirements):
  1. Plain-language summary FIRST: 3 to 5 bullets, high-school reading level,
     no jargon (a term that cannot be avoided is defined in the same
     sentence): what was tested, the headline p and what it means, whether the
     fund you could actually trade shows any excess over random entries, what
     the best-looking exit rules were and that they are partly luck, and the
     recommendation on spending a permanent scoring slot.
  2. The decision table and the recommendation.
  3. Power statement (from the calibration file when it exists, else "pending").
  4. Exploratory map summary (plateau ranking, stability, the same-cell null
     90th percentile labeled 'context only').
  5. Tradable-vehicle results, 6. put-affordability screen, 7. ABI-1 transfer
     line, 8. every count and check, 9. contamination and limits (the
     contaminated-series list, restated), 10. seeds, data digests, the exact command.

MONEY-FREE RULE. The report never contains a dollar sign followed by a digit
(generated text stays free of money figures by policy): `render_report` runs
`assert_no_dollar_digit` over its whole output and FAILS CLOSED (ValueError)
rather than return such text.

A failed-closed run (a required series could not be obtained) renders through
`render_failed_closed`: the summary says the study did not run, and a section
lists every missing series with the reason.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence

DOLLAR_DIGIT = re.compile(r"\$\s*\d")
# An account-number-like shape: a bare run of 8 to 17 digits (not the
# fractional tail of a decimal). Frozen seeds are 8-digit dates, so they are
# printed with an underscore group separator (2026_0923), same integer value.
_BARE_LONG_RUN = re.compile(r"(?<!\d\.)\b(\d{8,17})\b")

SECTION_TITLES = (
    "Plain-language summary",
    "Decision table and recommendation",
    "Power statement",
    "Exploratory map",
    "Tradable-vehicle results",
    "Put-affordability screen",
    "ABI-1 transfer line",
    "Data, counts and checks",
    "Contamination and limits",
    "Seeds, data digests and reproduction",
)
FAILED_SECTION_TITLES = (
    "Plain-language summary",
    "Missing required series",
    "Seeds, data digests and reproduction",
)

HEADLINE_P_MAX = 0.10
CANDIDATE_MIN_NULL_PERCENTILE = 0.80
MIN_VEHICLE_N = 8
IN_SAMPLE_BAR_TEXT = "1.02"

_SIDE_WORDS = {
    "up_fade": "betting on a slide back down after an upward spike or unusually high level",
    "down_fade": "betting on a bounce back up after a crash or unusually low level",
}
_SIDE_SHORT = {"up_fade": "short after a spike up", "down_fade": "long after a crash down"}
def trigger_words(trigger: str, side: str) -> str:
    """A trigger in plain words, for the side of the trade (up_fade = the market
    jumped or reached a high level; down_fade = it crashed or reached a low)."""
    up = side == "up_fade"
    if trigger.startswith("z20@"):
        size = trigger.split("@", 1)[1]
        return (f"a very big 20-day {'jump up' if up else 'drop'} "
                f"(about {size} times its usual size)")
    if trigger == "pct756":
        return (f"a level in the {'top' if up else 'bottom'} 5 percent of the last 3 years")
    if trigger == "pctexp":
        return (f"a level in the {'top' if up else 'bottom'} 5 percent of all earlier history")
    return trigger
_STATUS_LABELS = (
    ("traded", "traded"),
    ("undefined_sigma", "undefined swing size"),
    ("sigma_touch_nonpositive", "swing window touches a non-positive close"),
    ("no_e1_confirmation", "no confirming session (entry E1)"),
    ("touch_nonpositive", "entry or path touches a non-positive close"),
    ("undefined_median", "undefined median (profit-target cells)"),
    ("median_touch_nonpositive", "median window touches a non-positive close"),
    ("censored_data_end", "censored at the end of the data"),
    ("touch_missing", "entry, exit or path touches a missing close"),
    ("tp_not_applicable", "profit target not applicable (treated as none)"),
)


# ---------------------------------------------------------------------------
# The money-free guard
# ---------------------------------------------------------------------------

def assert_no_dollar_digit(text: str) -> None:
    """Raise ValueError if `text` contains a dollar sign followed by a digit
    (optionally after spaces): generated text must stay free of money figures,
    so the writer refuses to produce it."""
    m = DOLLAR_DIGIT.search(text)
    if m:
        lo = max(0, m.start() - 30)
        raise ValueError("report text contains a dollar sign followed by a digit near "
                         f"{text[lo:m.end() + 10]!r}; write money-free text only")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _group4(digits: str) -> str:
    return "_".join(digits[i:i + 4] for i in range(0, len(digits), 4))


def group_long_digits(text: str) -> str:
    """Rewrite every bare run of 8 to 17 digits in `text` with an underscore
    every four digits from the left (an 8-digit date becomes 2026_0923). Digits inside a hex
    digest, the fractional tail of a decimal and shorter numbers are untouched.
    This is the one helper every seed printed into a report or a log goes
    through."""
    return _BARE_LONG_RUN.sub(lambda m: _group4(m.group(1)), text)


def fmt_seed(value) -> str:
    """A seed for printing: 'n/a' for None, otherwise its digits with the
    underscore group separator when it has 8 or more (2026_0923)."""
    if value is None:
        return "n/a"
    return group_long_digits(str(value))


def _num(x, nd: int = 2, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _p(x) -> str:
    return _num(x, 3)


def _pct(x, nd: int = 0) -> str:
    return "n/a" if x is None else f"{100.0 * float(x):.{nd}f}%"


def _int(x) -> str:
    return "n/a" if x is None else str(int(x))


def _bool(x) -> str:
    return "n/a" if x is None else ("yes" if x else "no")


def _txt(x) -> str:
    return "" if x is None else str(x)


def md_table(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    """A markdown table; every cell is stringified, pipes are escaped."""
    def cell(v):
        return _txt(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines)


def cell_text(row: dict) -> str:
    """One-line description of an exit-rule cell from a row with entry, H, tp, sl."""
    return f"{row['entry']} | H {row['H']} | TP {row['tp']} | SL {row['sl']}"


def rule_words(entry: str, H, tp: str, sl) -> str:
    """The exit rules of a cell in plain words."""
    enter = ("enter the day after the signal" if entry == "E0"
             else "enter after the first day the move starts to reverse")
    target = {"none": "no profit target",
              "half": "take profit when the move has reversed half way to its usual level",
              "full": "take profit when the move has fully reversed to its usual level"}.get(
                  tp, f"profit target {tp}")
    sls = str(sl)
    stop = ("no stop" if sls in ("none", "None", "nan")
            else f"a safety stop at {sls} times the market's normal 21-day price swing")
    return f"{enter}, hold for {H} trading days, {target}, {stop}"


def _cell_str(cell) -> str:
    return _txt(cell).replace("|", " | ")


# ---------------------------------------------------------------------------
# 1. Plain-language summary
# ---------------------------------------------------------------------------

def _first(seq):
    return seq[0] if seq else None


def _vehicle_bullet(h: dict) -> str:
    v = h.get("vehicle")
    if not v:
        return ("The fund you could actually trade was not checked, because no group of "
                "events qualified for the yes-or-no test (so there is no best group to "
                "follow through to a fund).")
    ticker = v["ticker"]
    verb = "buy" if v.get("position") == "long" else "short"
    pair = (f"{h.get('tstar_series')} {_SIDE_SHORT.get(h.get('tstar_side'), '')}"
            if h.get("tstar_series") else "the best group")
    if v.get("underpowered"):
        return (f"For the fund you could actually {verb} for the best group ({pair}, through "
                f"{ticker}), there were too few trades ({_int(v.get('n'))}, fewer than "
                f"{MIN_VEHICLE_N}) to say anything, so no candidate can be declared.")
    ex = v.get("excess_pp")
    pct = v.get("null_percentile")
    if ex is None or pct is None:
        return (f"For the fund you could actually {verb} for the best group ({pair}, through "
                f"{ticker}), the comparison with random entry dates could not be run (the random "
                "draws could not reach the required number of trades), so nothing is claimed "
                "about it.")
    good = (ex is not None and ex > 0 and pct is not None
            and pct >= CANDIDATE_MIN_NULL_PERCENTILE)
    reps = v.get("reps")
    beat = (f"; it did better than {_pct(pct)} of {reps} random tries" if reps
            else f"; it did better than {_pct(pct)} of random tries")
    verdict = ("shows some extra gain over random entry dates" if good
               else "shows no clear extra gain over random entry dates")
    caution = (f" With only {_int(v.get('n'))} trades this is suggestive, not conclusive."
               if good and (v.get("n") or 0) < 30 else "")
    return (f"For the fund you could actually {verb} for the best group ({pair}, through "
            f"{ticker}), the same rule earned {_num(ex, 1, signed=True)} percentage points per "
            f"trade compared with entering on random dates ({_int(v.get('n'))} trades{beat}). "
            f"So the fund you could really trade {verdict}.{caution}")


def plain_language_summary(results: dict, calibration: Optional[dict] = None,
                           robustness: Optional[dict] = None) -> list:
    """The 3 to 5 plain-language bullets (strings without the leading dash)."""
    if results.get("status") != "completed":
        return _failed_summary(results)
    h = results.get("headline") or {}
    cutoff = results.get("cutoff", "the cutoff date")
    bullets = []

    params = results.get("params") or {}
    flag = ""
    if params.get("pre_registered_configuration") is False:
        notes = "; ".join(params.get("configuration_notes") or [])
        flag = ("WARNING: this run is NOT the pre-registered configuration"
                + (f" ({notes})" if notes else "")
                + ", so it is a test of the machinery, not the study's answer. ")
    bullets.append(
        flag +
        f"What we tested: using market history up to {cutoff}, we asked one question. When "
        "something that normally stays in a calm range (the VIX fear gauge, oil, natural gas, "
        "gold, copper, the US dollar index and a few similar gauges) makes a very big jump, or "
        "reaches an unusually high or low level, does betting that it slides back toward its "
        f"usual range make money? Everything after {cutoff} was locked away on purpose, but it "
        "is not fully unseen: some facts about it were already known when the rules were fixed "
        "(see the limits section), so a later official test is only partly independent.")

    p = h.get("p_headline")
    if p is None:
        bullets.append(
            "The one pre-planned yes-or-no test could not be run: no group of events had "
            f"enough past events (at least {MIN_VEHICLE_N}) to be tested, so there is no p to "
            "report (p is the chance of seeing a result this strong by luck alone).")
    else:
        convincing = p <= HEADLINE_P_MAX
        reps = h.get("B")
        how = (f", found by re-running the same test on {reps} sets of randomly chosen dates"
               if reps else "")
        s = (f"The one pre-planned yes-or-no test gave p = {_p(p)} (p is the chance of seeing a "
             f"result at least this strong by luck alone if entering on randomly chosen dates, "
             f"with the same holding rule, did just as well{how}). "
             f"We decided in advance that 0.10 or lower counts as convincing, so this result "
             f"is {'convincing' if convincing else 'not convincing'}.")
        if convincing:
            cand = h.get("candidate") or {}
            if cand.get("contaminated"):
                s += (" That series is one whose earlier history was already partly known when "
                      "the rules were fixed, so this counts as evidence about the past window only.")
            rows = size_robustness_rows(robustness)
            if rows:
                s += (" One more caution: in a fake-data check with no real effect, this test "
                      "raised a false alarm "
                      f"{_size_range_text(rows)} of the time when the fake series were very "
                      "jumpy, not the 10 percent the label suggests, so treat a result near the "
                      "line with care.")
        else:
            s += (" Not convincing means inconclusive, not proof that there is no effect: with "
                  "this little history the test can only reliably catch fairly large effects.")
        bullets.append(s)

    bullets.append(_vehicle_bullet(h))

    top = _first(h.get("top5_plateau") or [])
    if top:
        bullets.append(
            "The best-looking exit rules: we tried about 288 combinations of entry timing, "
            "holding time, profit target and safety stop for every market. The best-looking one "
            "that also held up among the combinations next to it was "
            f"{top['series']}, {_SIDE_SHORT.get(top['side'], top['side'])}, triggered by "
            f"{trigger_words(top['trigger'], top['side'])}: "
            f"{rule_words(top['entry'], top['H'], top['tp'], top['sl'])} (its neighbour score, "
            f"the typical result of the combinations next to it, was {_num(top.get('plateau_score'))} "
            "in units of that market's normal 21-day price swing). It rests on "
            f"{_int(top.get('n'))} trades and its steadiness check (does the result survive "
            "dropping its biggest winners, and do the two halves of the history agree) "
            f"{'passed' if top.get('stab_passed') else 'did not pass'}. With hundreds of "
            "combinations tried, some always look good by luck alone, so this is a list of "
            "ideas, not proof.")
    else:
        bullets.append(
            "The best-looking exit rules: we tried about 288 combinations for every market, but "
            "none had enough neighbours with enough events to rank. With that many combinations, "
            "any winner would be partly luck anyway, so treat any list as ideas, not proof.")

    rec = results.get("recommendation") or {}
    if not rec.get("register_stage2"):
        reasons = rec.get("reasons_not") or ["the conditions for registering were not met"]
        bullets.append(
            "Recommendation: do not spend a permanent scoring slot (the one-shot official test "
            "that can never be rerun) on this idea, because " + "; ".join(reasons) + ".")
    else:
        open_ = [c["text"] for c in rec.get("conditions", []) if c.get("met") is None]
        tail = (" A reader must still decide: " + "; ".join(open_) + "." if open_
                else " A reader must still confirm the last conditions.")
        bullets.append(
            "Recommendation: consider spending a permanent scoring slot (the one-shot official "
            "test that can never be rerun), but it is not automatic." + tail)
    failures = results.get("stage_failures") or []
    if failures:
        bullets[-1] += (f" Caution: this run is incomplete, {len(failures)} later step(s) failed "
                        "and are listed in the data section, so do not rely on the parts that "
                        "depend on them.")
    return bullets


def _failed_summary(results: dict) -> list:
    fail = results.get("failure") or {}
    missing = [m.get("name", "?") for m in fail.get("missing", [])]
    listing = ", ".join(missing) if missing else "see the list below"
    return [
        "The study did not run: some required price series could not be obtained, so nothing "
        f"was computed (the study stops rather than guess). Missing: {listing}.",
        "Because nothing ran, there is no p, no best exit rule and no recommendation from this "
        "attempt; treat the study as not done.",
        "What to do next: rerun the driver when the data can be fetched (it asks politely, one "
        "series at a time, and retries once); the missing series and the reason for each are "
        "listed below.",
    ]


def _summary_section(results: dict, calibration: Optional[dict],
                     robustness: Optional[dict] = None) -> str:
    bullets = plain_language_summary(results, calibration, robustness)
    return "\n".join("- " + b for b in bullets)


# ---------------------------------------------------------------------------
# 2. Decision table and the recommendation
# ---------------------------------------------------------------------------

def _n2_block(h: dict, conf: dict) -> dict:
    """All N2 fields in one dict. The null module keeps them under
    the single key `n2_sensitivity`; a flat layout (`p_n2`, `n2_offsets`,
    `n2_skipped`, `n2_coverage` beside the headline) is accepted too."""
    blk = conf.get("n2_sensitivity")
    if isinstance(blk, dict):
        out = dict(blk)
        out.setdefault("offsets", {})
        out.setdefault("skipped", [])
        out.setdefault("coverage", {})
        return out
    hb = h.get("n2_sensitivity") if isinstance(h.get("n2_sensitivity"), dict) else {}
    return {"p_n2": conf.get("p_n2", h.get("p_n2", hb.get("p_n2"))),
            "mc_se_n2": conf.get("mc_se_n2", hb.get("mc_se_n2")),
            "offsets": conf.get("n2_offsets") or hb.get("offsets") or {},
            "skipped": conf.get("n2_skipped") or hb.get("skipped") or [],
            "coverage": conf.get("n2_coverage") or {},
            "p_n2_covered": conf.get("p_n2_covered", hb.get("p_n2_covered"))}


def _n2_sensitivity_lines(h: dict, conf: dict, calibration: Optional[dict] = None) -> list:
    """The N2 translation null as a SENSITIVITY, not gating: shown
    beside the headline, never folded into it, with its availability and the
    feasible-offset counts per series."""
    n2 = _n2_block(h, conf)
    p2 = n2.get("p_n2")
    offsets = n2.get("offsets") or {}
    skipped = n2.get("skipped") or []
    cov = n2.get("coverage") or {}
    lines = []
    if p2 is None:
        lines.append("**N2 (translation null), sensitivity, not gating: not available** for the "
                     "T* series (fewer than 10 feasible offsets, or no tested group); the "
                     "headline does not depend on it.")
    else:
        lines.append(f"**N2 (translation null), sensitivity, not gating: p = {_p(p2)}** "
                     f"(Monte Carlo SE {_p(n2.get('mc_se_n2'))}). It applies to the T* "
                     "series; the headline does not depend on it.")
    if offsets:
        lines.append("Feasible N2 offsets per series: "
                     + ", ".join(f"{k} {v}" for k, v in offsets.items()) + ".")
    if skipped:
        lines.append("N2 skipped (fewer than 10 feasible offsets): " + ", ".join(skipped) + ".")
    if cov:
        lines.append(f"N2 covers {_int(cov.get('groups_covered'))} of "
                     f"{_int(cov.get('groups_tested'))} tested groups; like-for-like N2 p over "
                     f"the covered groups: {_p(n2.get('p_n2_covered'))}.")
    lfl = ((calibration or {}).get("size") or {}).get("p_n2_covered")
    if lfl:
        lines.append(f"N2 like-for-like size in the synthetic no-edge calibration: "
                     f"{_num(lfl.get('rate'), 3)} (reported, not required to lie in the "
                     "0.06 to 0.14 window; a shifted holding window lands on the spike's own "
                     "run-up, which is why N2 is a sensitivity and not the headline).")
    else:
        lines.append("N2 like-for-like size: pending (no calibration file was found).")
    return lines


def _adjusted_p_tables(conf: dict) -> list:
    """Per-series and per-side maxima compared against the SAME
    global null-max distribution (adjusted p), labelled descriptive."""
    out = []
    label = ("descriptive: each maximum is compared with the same global null-maximum "
             "distribution as the headline (N1 alone)")
    for title, key in (("Adjusted p by series", "per_series"), ("Adjusted p by side", "per_side")):
        items = conf.get(key) or {}
        if not items:
            continue
        out.append(f"### {title} ({label})")
        out.append("")
        rows = [[name, _txt(it.get("argmax_key")).replace("|", " "), _num(it.get("t_max"), 2),
                 _int(it.get("n_tested_groups")), _p(it.get("p_headline"))]
                for name, it in items.items()]
        out.append(md_table([key.replace("per_", ""), "best group", "max t", "tested groups",
                             "adjusted p"], rows))
        out.append("")
    return out


def _owner_section(results: dict, calibration: Optional[dict] = None,
                   robustness: Optional[dict] = None) -> str:
    h = results.get("headline") or {}
    conf = results.get("confirmatory") or {}
    out = []
    out.append("The table has one line per pair (series and side) of the confirmatory family; "
               "a pair's line shows its best-scoring group at the pre-declared primary cell "
               "(enter after the signal, hold 21 sessions, no target, no stop). "
               "S is the average result per trade in units of the market's normal 21-day price "
               "swing (sigma21); t is S divided by its own standard error. The adjusted p "
               "compares the pair's best t with the study-wide maximum and is descriptive only.")
    out.append("")
    if h.get("p_headline") is None:
        out.append("**Headline p: not available** (no tested group).")
    else:
        out.append(f"**Headline p = {_p(h['p_headline'])}** (rule as stated by the null "
                   f"module: {conf.get('headline_rule', '')}; Monte Carlo SE "
                   f"{_p(h.get('mc_se_headline'))}). N1 (random dates) p = "
                   f"{_p(h.get('p_n1', conf.get('p_n1')))}. T* = {_num(h.get('tstar'), 4)} at "
                   f"{_txt(h.get('tstar_key'))} (n = {_int(h.get('tstar_n'))}). "
                   f"Decision language: {h.get('decision', '')}.")
    out.append("")
    out.append(_size_caveat_sentence(robustness))
    out.append("")
    out.extend(_n2_sensitivity_lines(h, conf, calibration))
    out.append("")
    rows = []
    for r in results.get("family_table", []):
        if r.get("vehicle_status") == "run":
            veh = (f"{r.get('vehicle')}: excess {_num(r.get('vehicle_excess_pp'), 2, True)} pp, "
                   f"beat {_num(r.get('vehicle_null_percentile'), 2)} of random, "
                   f"n = {_int(r.get('vehicle_n'))}"
                   + (f", for cell {_cell_str(r['vehicle_cell'])}" if r.get("vehicle_cell")
                      else "")
                   + (" (underpowered: fewer than 8 trades)"
                      if (r.get("vehicle_n") or 0) < MIN_VEHICLE_N else ""))
        else:
            veh = "vehicle test not run"
        best = _txt(r.get("best_plateau_cell")) or "none (fewer than 3 eligible neighbours)"
        if r.get("best_plateau_score") is not None:
            best += f" (score {_num(r['best_plateau_score'])})"
        rows.append([f"{r['series']} {r['side']}" + (" **T\\***" if r.get("is_tstar") else ""),
                     r.get("trigger"), _int(r.get("n")), _num(r.get("mean_x"), 3),
                     _num(r.get("t"), 2), _p(r.get("adj_p_headline")), veh, best])
    out.append(md_table(["pair", "best group (trigger)", "n", "S", "t", "adjusted p (descriptive)",
                         "vehicle test vs random entries", "best exploratory cell"], rows))
    out.append("")
    out.extend(_adjusted_p_tables(conf))
    out.append("### Recommendation")
    rec = results.get("recommendation") or {}
    out.append("")
    verdict = ("consider registering a Stage 2 family, subject to the open conditions below"
               if rec.get("register_stage2")
               else "do not register a Stage 2 family or spend a permanent scoring slot")
    out.append(f"Recommendation: **{verdict}**.")
    out.append("")
    cond_rows = []
    for c in rec.get("conditions", []):
        met = {True: "met", False: "not met", None: "open / not decidable by this study"}[c.get("met")]
        cond_rows.append([c.get("id"), c.get("text"), met, c.get("detail")])
    if cond_rows:
        out.append(md_table(["", "condition", "status", "detail"], cond_rows))
        out.append("")
    for reason in rec.get("reasons_not", []) or []:
        out.append(f"- {reason}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. Power statement
# ---------------------------------------------------------------------------

def _curve_rows(curve: dict) -> list:
    rows = []
    for eff in sorted(curve, key=float):
        c = curve[eff]
        rows.append([eff, _num(c.get("power_headline"), 2), _int(c.get("n_datasets"))])
    return rows


def _power_section(calibration: Optional[dict]) -> str:
    if not calibration:
        return ("Calibration is pending: the synthetic no-edge calibration study "
                "(calibration/*.json under the study output folder) has not been found, so the "
                "power curve and the smallest detectable effects cannot be stated yet.")
    out = []
    args = calibration.get("args", {})
    size = calibration.get("size", {})
    ver = calibration.get("spec_version")
    hd = calibration.get("headline_definition")
    out.append(f"Source: the synthetic calibration file"
               + (f" (spec version {ver}" + (f", headline = {hd}" if hd else "") + ")" if ver else "")
               + f" ({_txt(args.get('datasets'))} no-edge "
               f"datasets, {_txt(args.get('power_datasets'))} power datasets per effect, "
               f"B = {_txt(args.get('B'))}, {_txt(args.get('n_series'))} series, seed "
               f"(a YYYYMMDD date) {fmt_seed(args.get('seed'))}). Effects are in units of sigma21 "
               "per trade.")
    out.append("")
    if size:
        rows = []
        labels = {"p_n1": "N1 (random dates)",
                  "p_n2": "N2 (translation), sensitivity, not gating",
                  "p_n2_covered": "N2 like-for-like over covered groups, sensitivity, not gating",
                  "p_headline": "headline (as defined by the null module)"}
        for name in ("p_n1", "p_n2", "p_n2_covered", "p_headline"):
            s = size.get(name)
            if s:
                rows.append([labels[name], _num(s.get("rate"), 3), _bool(s.get("in_window"))])
        out.append("No-edge rejection rate at 0.10 (must lie in 0.06 to 0.14):")
        out.append("")
        out.append(md_table(["test", "rate", "inside the window"], rows))
        out.append("")
    for label, key in (("the 7-series confirmatory family size (from the same replications)",
                        "power_subset_first_7_series"),
                       ("the 13-series calibration family", "power")):
        p = calibration.get(key)
        if not p:
            continue
        out.append(f"Headline power curve, {label}:")
        out.append("")
        out.append(md_table(["true effect (sigma21 per trade)", "headline power",
                             "datasets"], _curve_rows(p.get("curve", {}))))
        out.append("")
        out.append(f"Smallest true effect detected with 50 percent power: "
                   f"{_num(p.get('min_detectable_effect_50'))}; with 80 percent power: "
                   f"{_num(p.get('min_detectable_effect_80'))} "
                   f"(at least 50 percent power at 1.0 required: "
                   f"{_bool(p.get('passes_required_power'))}).")
        out.append("")
    out.append("A headline p above 0.10 therefore reads as inconclusive at the study's power, "
               "never as no edge.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. Exploratory map
# ---------------------------------------------------------------------------

def _stab_text(r: dict) -> str:
    if r.get("stab_passed") is None:
        return "n/a"
    s = "pass" if r["stab_passed"] else "fail"
    b = r.get("stab_b")
    return f"{s} (a {'ok' if r.get('stab_a') else 'no'}, b {'n/a' if b is None else ('ok' if b else 'no')})"


_MAP_HEADERS = ["rank", "series", "side", "trigger", "entry", "H", "TP", "SL", "n", "S", "t",
                "plateau score", "hit rate", "stability", "null 90th pct (context only)",
                "null stability pass rate"]


def _map_row(r: dict) -> list:
    return [r.get("rank"), r.get("series"), r.get("side"), r.get("trigger"), r.get("entry"),
            r.get("H"), r.get("tp"), r.get("sl"), _int(r.get("n")), _num(r.get("mean_x"), 3),
            _num(r.get("t"), 2), _num(r.get("plateau_score"), 3), _num(r.get("hit_rate"), 2),
            _stab_text(r), _num(r.get("null_q90"), 3),
            _num(r.get("null_stab_pass_rate_given_positive"), 2)]


def _map_section(results: dict) -> str:
    m = results.get("map") or {}
    out = []
    out.append(f"All {_int(m.get('n_cells'))} cells (every series, side, trigger, entry rule, "
               f"horizon, target and stop; {_int(m.get('n_eligible_cells'))} eligible with at "
               f"least 8 traded episodes, {_int(m.get('n_with_plateau'))} with a plateau score) "
               "are in cells.csv. The map has NO verdict and NO p-value: trying hundreds of "
               "rule combinations always finds a winner by luck, so its winners are partly luck "
               "by construction. It answers 'which situations, time frames, targets and stops "
               "look best', nothing more.")
    out.append("")
    out.append("Plateau score = the median S of a cell's eligible neighbours (cells one step "
               "away in horizon, target and stop); the ranking rewards results that hold up "
               "when the rule is nudged. Stability = S stays above zero without the largest "
               "and the two largest episodes (a) and the two date halves agree in sign (b); the "
               "last column is the pass rate of the same rule on random-date null cells "
               "(among null cells with S above zero). The 'null 90th pct' column is the 90th "
               "percentile of the same cell's random-date S distribution, **context only**: "
               "it is not a test and there is no multiplicity correction.")
    out.append("")
    out.append("### Top 30 overall (one line per group and entry rule)")
    out.append("")
    out.append(md_table(_MAP_HEADERS, [_map_row(r) for r in m.get("top_overall", [])]))
    out.append("")
    out.append("### Top 10 per pair")
    for pair, rows in (m.get("top_per_pair") or {}).items():
        out.append("")
        out.append(f"#### {pair.replace('|', ' ')}")
        out.append("")
        out.append(md_table(_MAP_HEADERS, [_map_row(r) for r in rows]))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. Tradable-vehicle results
# ---------------------------------------------------------------------------

def _vehicle_section(results: dict) -> str:
    v = results.get("vehicle") or {}
    out = []
    out.append("The vehicle test runs a pair's cell on the fund a trader could actually hold, "
               "using the fund's own total-return closes, against random-entry dates on the "
               "same fund (same rule, same number of trades, matched on which side of the "
               "median the entry price sits). Excess = the observed mean P&L minus the mean of "
               "the random-entry means; short positions ignore borrow and financing cost "
               "(disclosed). Scope: the pair that attains T* and the top 3 vehicle-eligible "
               "pairs by best plateau score, and nothing else. A ratio of tradable to series "
               "result is not reported.")
    out.append("")
    rows = []
    for e in v.get("scope", []):
        r = e.get("result")
        if not e.get("run") or r is None:
            rows.append([e.get("role"), f"{e.get('series')} {e.get('side')}", e.get("trigger"),
                         _cell_str(e.get("cell")), e.get("vehicle") or "none", "", "", "", "", "",
                         "", "vehicle test not run" + (f": {e['reason']}" if e.get("reason") else "")])
            continue
        rows.append([e.get("role"), f"{e.get('series')} {e.get('side')}", e.get("trigger"),
                     _cell_str(e.get("cell")), f"{e.get('vehicle')} ({e.get('position')})",
                     _int(r.get("n")), _num(r.get("mean_pnl_pp"), 2),
                     _num(r.get("null_mean_pnl_pp"), 2), _num(r.get("excess_pp"), 2, True),
                     _num(r.get("null_percentile"), 2), _num(r.get("hit_rate"), 2),
                     f"unconditional average return over the same window: "
                     f"{_num(r.get('unconditional_mean_pnl_pp'), 2)} pp per trade"
                     + ("; contaminated series" if r.get("contaminated") else "")])
    out.append(md_table(["role", "pair", "group", "cell", "vehicle", "n", "mean P&L (pp)",
                         "random-entry mean (pp)", "excess (pp)", "share of random tries beaten",
                         "hit rate", "notes"], rows))
    for e in v.get("scope", []):
        r = e.get("result")
        if not r:
            continue
        out.append("")
        out.append(f"### {e.get('series')} {e.get('side')} on {e.get('vehicle')} "
                   f"({e.get('role')})")
        out.append("")
        out.append(f"- Excess over random entries: {_num(r.get('excess_pp'), 2, True)} percentage "
                   f"points, {_num(r.get('excess_sigma'), 3, True)} in units of the fund's own "
                   f"21-day swing; the observed mean beat {_num(r.get('null_percentile'), 2)} of "
                   f"{_int(r.get('null_reps_used'))} random-entry means "
                   f"({_int(r.get('null_reps_failed'))} failed replications; seeds are "
                   "listed at the end).")
        out.append(f"- Anchors: {_int(r.get('n_anchors_input'))} offered, dropped "
                   f"no session {_int(r.get('dropped_no_session'))}, past the end "
                   f"{_int(r.get('dropped_past_end'))}, short history "
                   f"{_int(r.get('dropped_short_history'))}, censored "
                   f"{_int(r.get('dropped_censored'))}, duplicates "
                   f"{_int(r.get('dropped_duplicate'))}; kept {_int(r.get('n_anchors_kept'))}, of "
                   f"which {_int(r.get('n'))} traded ({_int(r.get('n_kept_not_traded'))} kept "
                   "anchors did not trade).")
        out.append(f"- Window {_txt(r.get('window_start'))} to {_txt(r.get('window_end'))}; the "
                   f"fund's unconditional average return there: "
                   f"{_num(r.get('unconditional_mean_daily_return_pp'), 3)} pp per day, "
                   f"{_num(r.get('unconditional_mean_pnl_pp'), 2)} pp per trade for the same "
                   "rule entered on every eligible day.")
        if r.get("underpowered_on_vehicle"):
            out.append(f"- Underpowered on the vehicle (fewer than {MIN_VEHICLE_N} trades): no "
                       "candidate can be declared from this pair.")
        era = r.get("era")
        if era:
            out.append("- Era split (reported pooled, not a gate): " + "; ".join(
                f"{k}: n {_int(x.get('n'))}, mean {_num(x.get('mean_pnl'), 4)}, excess vs "
                f"pooled random {_num(x.get('excess_pp_vs_pooled_null'), 2, True)} pp"
                for k, x in era.items()))
    out.append("")
    out.append("### Candidate definition")
    out.append("")
    c = v.get("candidate") or {}
    if c.get("is_candidate"):
        out.append("A candidate exists: the headline p is at most 0.10 and the T* pair's fund "
                   "beats random entries (positive excess, at least 0.80 of random tries beaten, "
                   "at least 8 trades).")
    else:
        why = ", ".join(c.get("reasons") or []) or "not evaluated"
        out.append(f"Not a candidate: {why}. (A candidate needs the headline p at most 0.10 AND, "
                   "for the T* pair's fund, a positive excess over random entries with at least "
                   "0.80 of random tries beaten and at least 8 fund trades.)")
    if c.get("contaminated"):
        out.append("")
        out.append(f"The T* series is marked contaminated: {_txt(c.get('contamination_note'))}.")
    out.append("")
    fd = results.get("vehicle_first_dates") or {}
    if fd:
        out.append("Vehicle first dates checked against the vehicle table's dates:")
        out.append("")
        out.append(md_table(["vehicle", "table date", "observed first date", "matches"],
                            [[k, x.get("table"), x.get("observed"), _bool(x.get("matches"))]
                             for k, x in fd.items()]))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 6. Put-affordability screen
# ---------------------------------------------------------------------------

def _put_section(results: dict) -> str:
    out = []
    out.append("This is a SCREEN, not a backtest of options: no historical option price series "
               "exists in the repository or for free. It measures the most a put buyer could pay "
               "and still break even (the mean expiry payoff), prices puts with Black-Scholes "
               "(zero rate and dividend) at volatility = k times the trailing 21-session "
               "realized volatility, and gives the break-even k, the same on random dates as the "
               "baseline (a model misfit shows up there too), and the MARKET k where an "
               "implied-volatility index is cached (^VIX over SPY realized volatility, ^OVX over "
               "WTI, ^GVZ over gold, ^VVIX over the VIX index's own realized volatility). "
               "A put is only worth considering where the break-even k exceeds the market k by "
               "a clear margin AND exceeds the random-date baseline break-even k. VIX options "
               "are priced off same-expiry futures, so the VIX version is indicative only.")
    for e in results.get("put_screen", []):
        out.append("")
        out.append(f"### {e.get('series')} {e.get('side')} {e.get('trigger')} "
                   f"{_cell_str(e.get('cell'))} ({e.get('role')})")
        out.append("")
        if e.get("status") != "computed":
            out.append(f"Not applicable: {_txt(e.get('reason'))}.")
            continue
        out.append(f"Episodes used {_int(e.get('n_used'))}, dropped {_int(e.get('n_dropped'))}; "
                   f"expiry horizon H = {_int(e.get('H'))} sessions.")
        out.append("")
        obs = e.get("observed") or {}
        base = e.get("baseline") or {}
        comp = e.get("comparison") or {}
        rows = []
        ks = []
        for pct in sorted(obs, key=int, reverse=True):
            o = obs[pct]
            ks = sorted(o.get("avg_pnl", {}), key=float)
            b = base.get(pct, {})
            cm = comp.get(pct, {})
            rows.append([f"{pct}%", _num(o.get("break_even_premium"), 4)]
                        + [_num(o["avg_pnl"].get(k), 4, True) for k in ks]
                        + [_num(o.get("break_even_k"), 2) + f" ({o.get('break_even_k_status')})",
                           _num(b.get("break_even_k"), 2) + f" ({b.get('break_even_k_status')})",
                           _num(cm.get("margin_over_baseline"), 2, True),
                           _bool(cm.get("exceeds_baseline")),
                           _num(cm.get("margin_over_market_median"), 2, True),
                           _num(cm.get("margin_over_market_p90"), 2, True)])
        out.append(md_table(["strike", "break-even premium (share of entry price)"]
                            + [f"avg put P&L at k={k}" for k in ks]
                            + ["break-even k (observed)", "break-even k (random dates)",
                               "margin over random", "exceeds random", "margin over market median",
                               "margin over market 90th pct"], rows))
        out.append("")
        mk = e.get("market_k_summary")
        proxy = e.get("market_k_proxy")
        if mk and mk.get("n"):
            out.append(f"Market k at the entries ({_txt(proxy)} over the series' own realized "
                       f"volatility): median {_num(mk.get('median'))}, 10th percentile "
                       f"{_num(mk.get('p10'))}, 90th percentile {_num(mk.get('p90'))} "
                       f"(n = {_int(mk.get('n'))}).")
        else:
            out.append("Market k: no implied-volatility series is cached for this series "
                       f"(proxy {_txt(proxy) or 'none'}), so it is not reported.")
        vs = e.get("market_k_vix_over_spy")
        if vs and vs.get("n"):
            out.append(f"Descriptive extra: ^VIX over SPY realized volatility at the same entry "
                       f"dates: median {_num(vs.get('median'))}, 10th percentile "
                       f"{_num(vs.get('p10'))}, 90th percentile {_num(vs.get('p90'))} "
                       f"(n = {_int(vs.get('n'))}).")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 7. ABI-1 transfer line
# ---------------------------------------------------------------------------

def _acct_rows(full: dict, exw: dict) -> list:
    def row(name, f):
        cap = f.get("absolute_cap")
        return [name, _int(f.get("n_trades")), _num(f.get("excess_sharpe_annualized"), 2),
                f"{_num(f.get('in_sample_bar'), 2)} (exact bar for this length "
                f"{_num(f.get('dsr_bar_annualized'), 2)})", _bool(f.get("meets_in_sample_bar")),
                _num(f.get("max_dd_account"), 2), _num(f.get("max_dd_spy"), 2),
                _num(abs(cap) if cap is not None else None, 2), _bool(f.get("dd_passed")),
                _num(f.get("terminal_wealth_account"), 2), _num(f.get("terminal_wealth_spy"), 2)]
    return [row("full", full), row("ex-worst (two largest-loss episodes removed)", exw)]


def _abi_block(title: str, e: Optional[dict]) -> list:
    out = [f"### {title}", ""]
    if not e:
        out.append("Not run (no T* group, or no vehicle-eligible plateau cell).")
        return out
    if e.get("status") != "computed":
        out.append(f"Not computed ({_txt(e.get('status'))}): {_txt(e.get('reason'))}.")
        return out
    out.append(f"{e.get('series')} {e.get('side')} {e.get('trigger')}, cell "
               f"{_cell_str(e.get('cell'))}, vehicle {e.get('vehicle')} ({e.get('position')}), "
               f"overlay at {_num(e.get('overlay_notional'), 2)} of equity, cost "
               f"{_num(e.get('cost_bps'), 1)} bps round trip. Signals {_int(e.get('n_signals'))}: "
               f"skipped for overlap {_int(e.get('n_skipped_overlap'))}, no entry "
               f"{_int(e.get('n_skipped_no_entry'))}, undefined {_int(e.get('n_skipped_undefined'))}.")
    out.append("")
    out.append(md_table(["account", "trades", "annualized excess Sharpe",
                         "in-sample bar", "meets bar", "max drawdown account",
                         "max drawdown SPY", "absolute cap", "drawdown refusals passed",
                         "terminal wealth account", "terminal wealth SPY"],
                        _acct_rows(e.get("full") or {}, e.get("ex_worst") or {})))
    return out


def _abi_section(results: dict) -> str:
    a = results.get("abi1") or {}
    out = []
    out.append("The account the pipeline would score: an SPY core plus the vehicle overlay at "
               "0.25 of equity, following the engine's conventions (entry at the next open after "
               "the signal, exit at the flag bar's close, daily mark to market, pre-tax, no "
               "borrow or financing). Stage 1's own fills differ by one session, so Stage 2 "
               "must re-derive the cell under the engine's conventions. Reported: the annualized "
               "daily excess Sharpe against the in-sample bar of about 1.02, and the account's "
               "maximum drawdown against SPY's and against the absolute cap of 0.35; then the "
               "same with the two largest-loss episodes removed (ex-worst).")
    out.append("")
    out.extend(_abi_block("T* pair, primary cell (the only registrable rule)", a.get("tstar")))
    out.append("")
    out.extend(_abi_block("Top plateau cell among vehicle-eligible pairs (DESCRIPTIVE only; "
                          "can never be registered by this study)", a.get("top_plateau")))
    conv = (a.get("tstar") or a.get("top_plateau") or {}).get("conventions")
    if conv:
        out.append("")
        out.append(f"Engine conventions used: {conv}.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 8. Data, counts and checks
# ---------------------------------------------------------------------------

def _counts_section(results: dict) -> str:
    series = results.get("series") or {}
    out = []
    out.append("Every dropped, censored, undefined, not applicable and externally corrected "
               "observation is counted here or in cells.csv (per cell). Nothing is altered "
               "except the external same-day cross-check of JPY=X against FRED DEXJPUS.")
    out.append("")
    failures = results.get("stage_failures") or []
    if failures:
        out.append("### Stage failures (this run is INCOMPLETE)")
        out.append("")
        for f in failures:
            out.append(f"- {f.get('stage')} ({f.get('item')}): {f.get('reason')}")
        out.append("")
    out.append("### Data quality per series")
    out.append("")
    rows = []
    for name, s in series.items():
        c = s.get("counts", {})
        rows.append([name, s.get("role"), s.get("source"), s.get("first_date"), s.get("last_date"),
                     _int(c.get("rows")), _int(c.get("missing_rows_dropped")),
                     _int(c.get("pre_start_dropped")), _int(c.get("post_cutoff_dropped")),
                     _int(c.get("nonpositive")), _int(c.get("external_corrections")),
                     _int(c.get("reference_unchecked")), s.get("extra_lag")])
    out.append(md_table(["series", "role", "source", "first usable date", "last date", "rows kept",
                         "missing rows dropped", "dropped before first usable date",
                         "dropped after the cutoff (sealed)", "non-positive closes",
                         "externally corrected", "unchecked vs reference",
                         "extra publication lag (sessions)"], rows))
    out.append("")
    out.append("### Externally corrected observations (listed in full)")
    out.append("")
    any_ext = False
    for name, s in series.items():
        ds = s.get("external_correction_dates") or []
        if ds:
            any_ext = True
            out.append(f"- {name} (against {_txt(s.get('external_reference'))}): "
                       f"{len(ds)} closes set missing, dates: {', '.join(ds)}")
    if not any_ext:
        out.append("None: no close was externally corrected in this run.")
    out.append("")
    out.append("Non-positive closes (their logarithm is undefined; only the quantities that use "
               "them are undefined):")
    out.append("")
    any_np = False
    for name, s in series.items():
        ds = s.get("nonpositive_dates") or []
        if ds:
            any_np = True
            out.append(f"- {name}: {', '.join(ds)}")
    if not any_np:
        out.append("- none")
    out.append("")
    out.append("### Triggers per series (first usable signal date, undefined sessions, roll-off share)")
    out.append("")
    rows = []
    for name, s in series.items():
        for trig, d in (s.get("trigger_diagnostics") or {}).items():
            up, dn = d.get("up_fade", {}), d.get("down_fade", {})
            rows.append([name, trig, d.get("first_usable_date"), _int(d.get("n_warmup")),
                         _int(d.get("n_undefined_after_first_usable")), _int(d.get("n_undefined")),
                         _num(up.get("rolloff_share"), 3), _num(dn.get("rolloff_share"), 3)])
    out.append(md_table(["series", "trigger", "first usable signal date",
                         "warm-up sessions", "undefined sessions after that", "undefined sessions total",
                         "roll-off share up", "roll-off share down"], rows))
    out.append("")
    out.append("### Episodes per group (gap20 de-clustering, uniform censoring)")
    out.append("")
    rows = []
    for name, s in series.items():
        for g in s.get("groups", []):
            rows.append([name, g.get("trigger"), g.get("side"), _int(g.get("n_flagged_days")),
                         _int(g.get("n_episodes")), _int(g.get("n_kept")),
                         _int(g.get("dropped_censored"))])
    out.append(md_table(["series", "trigger", "side", "flagged days", "episodes",
                         "kept", "dropped (censored)"], rows))
    out.append("")
    out.append("### Cell-level accounting per series (all 576 cells: every observation by name)")
    out.append("")
    rows = []
    for name, s in series.items():
        t = s.get("cell_status_totals", {})
        rows.append([name] + [_int(t.get(k)) for k, _ in _STATUS_LABELS])
    out.append(md_table(["series"] + [lab for _, lab in _STATUS_LABELS], rows))
    tot = (results.get("counts") or {}).get("totals")
    if tot:
        out.append("")
        names = dict(_STATUS_LABELS)
        names.update({"rows": "rows kept", "missing_rows_dropped": "missing rows dropped",
                      "pre_start_dropped": "dropped before the first usable date",
                      "post_cutoff_dropped": "dropped after the cutoff (sealed)",
                      "nonpositive": "non-positive closes",
                      "external_corrections": "externally corrected",
                      "flagged_days": "flagged days", "episodes": "episodes",
                      "kept": "episodes kept", "dropped_censored": "dropped (censored)"})
        out.append("Totals over all series (cell-level counts are per cell and episode): "
                   + ", ".join(f"{names.get(k, k)} = {_txt(v)}" for k, v in tot.items()) + ".")
    out.append("")
    out.append("### Affected anchor dates (data-rule removals, listed)")
    out.append("")
    any_aff = False
    for name, s in series.items():
        aff = s.get("affected_dates") or []
        if aff:
            any_aff = True
            out.append(f"- {name}: {len(aff)} anchor dates: "
                       + "; ".join(f"{a['date']} [{', '.join(a['statuses'])}]" for a in aff))
    if not any_aff:
        out.append("- none")
    out.append("")
    out.append("### Roll-gap check (continuous futures against their ETFs)")
    out.append("")
    rg = results.get("roll_gap") or []
    if rg:
        out.append(md_table(["series", "ETF", "overlap sessions", "daily-return correlation",
                             "largest absolute difference", "date", "series daily sd",
                             "ratio", "flagged (over 5 x own sd)"],
                            [[r.get("series"), r.get("etf"), _int(r.get("n")), _num(r.get("corr"), 3),
                              _num(r.get("max_abs_diff"), 4), r.get("max_abs_diff_date"),
                              _num(r.get("sd_daily"), 4), _num(r.get("ratio"), 2),
                              _bool(r.get("flagged"))] for r in rg]))
    else:
        out.append("Not computed (the futures or ETF series was not part of this run).")
    out.append("")
    out.append("### AR(1) half-life of each series' log level (descriptive only, not gating)")
    out.append("")
    out.append(md_table(["series", "phi", "half-life (sessions)", "pairs", "note"],
                        [[n, _num(s.get("ar1", {}).get("phi"), 4),
                          _num(s.get("ar1", {}).get("half_life_sessions"), 1),
                          _int(s.get("ar1", {}).get("n_pairs")), s.get("ar1", {}).get("note")]
                         for n, s in series.items()]))
    out.append("")
    out.append("### The 20 largest absolute one-session log returns per series (for the reader "
               "to inspect; nothing is altered)")
    for name, s in series.items():
        lr = s.get("largest_returns") or []
        out.append("")
        out.append(f"- **{name}**: " + "; ".join(
            f"{r.get('date')} ({_num(r.get('log_return'), 3, True)})" for r in lr))
    vser = results.get("vehicle_series") or {}
    if vser:
        out.append("")
        out.append("### The same listing for the vehicles and the SPY core (a vendor bad print "
                   "in a fund would show here; nothing is altered)")
        for name, s in vser.items():
            lr = s.get("largest_returns") or []
            out.append("")
            out.append(f"- **{name}** ({s.get('role')}, {_int(s.get('n_rows'))} sessions, "
                       f"{s.get('first_date')} to {s.get('last_date')}): " + "; ".join(
                           f"{r.get('date')} ({_num(r.get('log_return'), 3, True)})" for r in lr))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Null-size robustness caveat: read from the calibration JSON
# ---------------------------------------------------------------------------

ROBUSTNESS_FILE = "calibration/null_size_robustness.json under the study output folder"
_ARM_WORDS = {1.0: "the calibration generator", 2.0: "double the volatility of volatility",
              3.0: "triple the volatility of volatility"}


def size_robustness_rows(robustness: Optional[dict]) -> Optional[list]:
    """The N1 no-edge rejection rate at 0.10 per volatility-of-volatility
    multiplier, read from the robustness JSON (`size_by_vol_of_vol_multiplier`):
    a list of dicts (mult, rate, se, rejections, n) sorted by multiplier, or
    None when the file is absent, unfinished or unreadable (the caller prints
    'pending', never a guess)."""
    if not isinstance(robustness, dict) or not robustness.get("finished"):
        return None
    arms = robustness.get("size_by_vol_of_vol_multiplier")
    if not isinstance(arms, dict) or not arms:
        return None
    rows = []
    try:
        for key, arm in arms.items():
            p1 = arm["p_n1"]
            rows.append({"mult": float(key), "rate": float(p1["rate"]), "se": float(p1["se"]),
                         "rejections": p1.get("rejections"),
                         "n": int(p1.get("n_denominator", arm.get("n_datasets", 0)))})
    except (KeyError, TypeError, ValueError):
        return None
    rows.sort(key=lambda r: r["mult"])
    return rows


def _arm_text(row: dict) -> str:
    who = _ARM_WORDS.get(row["mult"], f"{row['mult']:g} times the volatility of volatility")
    return f"{_num(row['rate'], 3)} (SE {_num(row['se'], 3)}) at {who}"


def _size_range_text(rows: list) -> str:
    lo = min(r["rate"] for r in rows)
    hi = max(r["rate"] for r in rows)
    return f"{100.0 * lo:.0f} to {100.0 * hi:.0f} percent"


def _size_caveat_sentence(robustness: Optional[dict]) -> str:
    """The one sentence printed right under the headline p (decision-table block)."""
    rows = size_robustness_rows(robustness)
    if rows is None:
        return ("**Size caveat: pending.** The synthetic null-size robustness check "
                f"({ROBUSTNESS_FILE}) was not found or could not be read, so how often this test "
                "would fire on very volatile series with no edge cannot be stated yet.")
    nser = ((robustness.get("args") or {}).get("n_series"))
    fam = f", on a {_int(nser)}-series family" if nser else ""
    parts = ", ".join(_arm_text(r) for r in rows)
    return (f"**Size caveat** (synthetic no-edge check, N1 rejection rate at 0.10: {parts}{fam}): "
            f"a p at or below 0.10 is therefore approximately an {_size_range_text(rows)} test "
            "for very volatile series; a non-rejection is unaffected in direction (see the "
            "null-size robustness caveat in the limits section).")


def _size_caveat_paragraph(robustness: Optional[dict]) -> list:
    rows = size_robustness_rows(robustness)
    if rows is None:
        return ["**Null-size robustness caveat: pending.** The synthetic robustness file "
                f"({ROBUSTNESS_FILE}) was not found or could not be read, so the size of the "
                "N1 test on very volatile series is not stated here."]
    args = robustness.get("args") or {}
    out = ["**Null-size robustness caveat.** In a synthetic no-edge check "
           f"({_txt(args.get('datasets'))} datasets per arm, B = {_txt(args.get('B'))}, "
           f"{_txt(args.get('n_series'))} series of {_txt(args.get('sessions'))} sessions, "
           "identical seeds across arms) the N1 headline test's no-edge rejection rate at 0.10 "
           "was measured while the volatility of volatility of the simulated series was raised. "
           "Only the first arm (the calibration generator) is the size claim; the others are a "
           "reported caveat, never a re-tuning, and no definition was changed in response.",
           ""]
    out.append(md_table(["volatility of volatility", "N1 no-edge rejection rate at 0.10",
                         "SE", "rejections", "datasets", "inside 0.06 to 0.14"],
                        [[("1x (the calibration generator)" if r["mult"] == 1.0
                           else f"{r['mult']:g}x"), _num(r["rate"], 3), _num(r["se"], 3),
                          _int(r["rejections"]), _int(r["n"]),
                          _bool(0.06 <= r["rate"] <= 0.14)] for r in rows]))
    out.append("")
    out.append("Mechanism: N1 matches the random-date pseudo-episodes to the observed ones only on "
               "the sign of D (which side of the median the entry sits), not on the local "
               "volatility level. The convexity drift of simple returns (for a long position a "
               "given log move gains more than it loses) can therefore be larger at spike dates "
               "than at the random dates N1 draws, so the rejections are anti-conservative "
               "there and the test rejects more often than its nominal 0.10 when there is no "
               "edge. "
               f"Consequence: a p at or below 0.10 is approximately an {_size_range_text(rows)} "
               "test for very volatile series (the rate rises with the volatility of volatility "
               "of the series); a non-rejection is unaffected in direction, so a p above 0.10 "
               "keeps its reading of inconclusive at the study's power.")
    return out


# ---------------------------------------------------------------------------
# 9. Contamination and limits
# ---------------------------------------------------------------------------

def _episode_range_phrase(results: dict) -> str:
    """Episode-count range over the (series, trigger, side) groups, derived from
    the results this report is written from; qualitative when none are recorded."""
    counts = []
    for s in (results.get("series") or {}).values():
        for g in s.get("groups", []):
            n = g.get("n_episodes")
            if isinstance(n, (int, float)) and not isinstance(n, bool) and n >= 0:
                counts.append(int(n))
    if not counts:
        return "only a limited number of independent episodes per series;"
    lo, hi = min(counts), max(counts)
    if lo == hi:
        return f"{lo} independent episodes per group (see the episodes table);"
    return f"{lo} to {hi} independent episodes per group (see the episodes table);"


def _limits_section(results: dict, robustness: Optional[dict] = None) -> str:
    return "\n".join(_size_caveat_paragraph(robustness) + [""] + [
        "Contamination and scope:",
        "",
        "- Sealed-window (after 2023-06-30) scope: no per-episode return after the cutoff is "
        "an input to any definition; general market memory of recent volatility spikes and oil "
        "moves is acknowledged (not an input).",
        "- Discovery-window scope (allowed by design, but NOT blind): the VIX spike effect and "
        "the inverse-volatility vehicle's excess are treated as already partly known inside the "
        "discovery window.",
        "- Consequence: ^VIX, WTI spot and Brent spot (Brent co-moves with WTI) are marked "
        "CONTAMINATED. A result attained by one of them is reported as confirmatory on the "
        "discovery window only; the sealed window is mechanically sealed but partly known for "
        "this series, and Stage 2 would need explicit acceptance of that. The primary "
        "cell (enter after the signal, hold 21 sessions, no target, no stop) was chosen "
        "without looking at observed spike-conditional means.",
        "- Everything after 2023-06-30 is mechanically sealed: the study loader truncates every "
        "series at the cutoff before ANY computation and asserts the maximum date; about twelve "
        "weeks of buffer separate the cutoff from the official ABI-1 holdout, so the official "
        "score later has three sealed but partly known years.",
        f"- Limits: {_episode_range_phrase(results)} overlapping holds and crisis "
        "clustering make episodes less independent than counted; a one-sided synthetic power "
        "study cannot fix that; a null finding does not prove the mechanism absent. FRED daily "
        "spot files are research-only (not on the repository's verified-unrevised allowlist) and "
        "are labeled current revised history; Yahoo series are vendor-adjusted history as "
        "retrieved on the run date. Short positions assume shares can be borrowed and ignore "
        "borrow and financing cost. Stage 1 spends no trial slot and writes no ledger row.",
    ])


# ---------------------------------------------------------------------------
# 10. Seeds, digests, reproduction
# ---------------------------------------------------------------------------

def _repro_section(results: dict, meta: Optional[dict], manifest: Optional[dict]) -> str:
    params = results.get("params") or {}
    seeds = params.get("seeds") or {}
    out = []
    out.append("Seeds (each one a YYYYMMDD date, frozen in the pre-registration or, where it "
               "is silent, in the modules): "
               + (", ".join(f"{k} {fmt_seed(v)}" for k, v in seeds.items()) or "none recorded") + ". "
               f"B = {_txt(params.get('B'))} replications per null, B_map = "
               f"{_txt(params.get('B_map'))} for the map's same-cell null.")
    out.append("")
    labels = (results.get("data") or {}).get("labels") or {}
    if labels:
        out.append("Data labels: " + "; ".join(f"{k}: {v}" for k, v in labels.items()) + ".")
        out.append("")
    files = (results.get("data") or {}).get("files") or {}
    man = manifest or {}
    rows = []
    for name, f in files.items():
        m = man.get(name, {})
        rows.append([name, f.get("kind"), m.get("file") or f.get("file"),
                     f.get("rows_within_cutoff", f.get("rows")), m.get("rows"),
                     f.get("first_date"), f.get("last_date"),
                     m.get("sha256") or f.get("sha256"), m.get("retrieved_utc") or ""])
    for name, m in man.items():
        if name not in files:
            rows.append([name, "", m.get("file"), "", m.get("rows"), "", "", m.get("sha256"),
                         m.get("retrieved_utc") or ""])
    if rows:
        out.append(md_table(["series", "kind", "raw file", "rows within the cutoff",
                             "rows in the whole raw file", "first date", "last date within the "
                             "cutoff", "SHA-256", "retrieved (UTC)"], rows))
        out.append("")
    out.append("The raw data folder (data/raw/ under the study output folder) is bulk data and is "
               "not part of the tracked evidence; the digests above identify exactly which bytes "
               "were used. results.json and cells.csv hold every headline number and every cell.")
    out.append("")
    if meta:
        skip = ("command", "command_to_reuse_cached_data")
        out.append("Run metadata: " + ", ".join(
            f"{k} = {_txt(v)}" for k, v in meta.items() if k not in skip) + ".")
        out.append("")
        out.append("The command below runs the study on the raw files listed above. Without "
                   "--skip-fetch it would download fresh data instead, and the digests will "
                   "differ from those above (vendor history and FRED revisions move).")
        out.append("")
        out.append("Exact command as run:")
        out.append("")
        out.append("    " + _txt(meta.get("command")))
        if (meta.get("command_to_reuse_cached_data")
                and meta["command_to_reuse_cached_data"] != meta.get("command")):
            out.append("")
            out.append("Exact command reusing the cached raw data (digests are verified first):")
            out.append("")
            out.append("    " + _txt(meta["command_to_reuse_cached_data"]))
    else:
        out.append("Run metadata was not supplied to the writer.")
    out.append("")
    oc = results.get("open_choices") or []
    if oc:
        out.append("### Open choices (made where the study's fixed rules are silent; never made silently)")
        out.append("")
        for c in oc:
            out.append(f"- {c}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Failed-closed report
# ---------------------------------------------------------------------------

def _missing_section(results: dict) -> str:
    fail = results.get("failure") or {}
    out = [f"Reason: {_txt(fail.get('reason'))}. The study fails closed: it does not run on a "
           "partial universe, because the pre-registration fixes the universe and dropping a "
           "series after seeing what is available would be a silent change of definition.", ""]
    rows = [[m.get("name"), m.get("kind"), m.get("source"), m.get("symbol"), m.get("reason")]
            for m in fail.get("missing", [])]
    out.append(md_table(["series", "kind", "source", "symbol", "why it is missing"], rows))
    for d in fail.get("detail", []) or []:
        out.append("")
        out.append(f"- {d}")
    return "\n".join(out)


def render_failed_closed(results: dict, *, meta: Optional[dict] = None,
                         manifest: Optional[dict] = None) -> str:
    """The report of a run that failed closed: the summary says the study did
    not run, and the missing series are listed with the reason for each."""
    body = ["# Range-bound spike-fade study v1: report (FAILED CLOSED, nothing was run)", ""]
    sections = {
        "Plain-language summary": _summary_section(results, None),
        "Missing required series": _missing_section(results),
        "Seeds, data digests and reproduction": _repro_section(results, meta, manifest),
    }
    for title in FAILED_SECTION_TITLES:
        body.append(f"## {title}")
        body.append("")
        body.append(sections[title])
        body.append("")
    text = "\n".join(body).rstrip() + "\n"
    assert_no_dollar_digit(text)
    return text


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------

def render_report(results: dict, *, meta: Optional[dict] = None,
                  calibration: Optional[dict] = None,
                  manifest: Optional[dict] = None,
                  robustness: Optional[dict] = None) -> str:
    """The full REPORT.md text for a results dict (see the module docstring).

    `meta` = run metadata that varies between runs (command, timestamps,
    python, workers); `calibration` = the parsed calibration JSON or None
    (then the power statement says calibration is pending); `manifest` =
    {series: {sha256, retrieved_utc, file, rows}} from the raw-data manifest;
    `robustness` = the parsed null_size_robustness.json or None (then the size
    caveat prints as pending).
    Pure: nothing is read or written and the inputs are not modified. Raises
    ValueError if the finished text contains a dollar sign followed by a digit.
    """
    if results.get("status") != "completed":
        return render_failed_closed(results, meta=meta, manifest=manifest)
    sections = {
        SECTION_TITLES[0]: _summary_section(results, calibration, robustness),
        SECTION_TITLES[1]: _owner_section(results, calibration, robustness),
        SECTION_TITLES[2]: _power_section(calibration),
        SECTION_TITLES[3]: _map_section(results),
        SECTION_TITLES[4]: _vehicle_section(results),
        SECTION_TITLES[5]: _put_section(results),
        SECTION_TITLES[6]: _abi_section(results),
        SECTION_TITLES[7]: _counts_section(results),
        SECTION_TITLES[8]: _limits_section(results, robustness),
        SECTION_TITLES[9]: _repro_section(results, meta, manifest),
    }
    body = ["# Range-bound spike-fade study v1: report", ""]
    for title in SECTION_TITLES:
        body.append(f"## {title}")
        body.append("")
        body.append(sections[title])
        body.append("")
    text = "\n".join(body).rstrip() + "\n"
    assert_no_dollar_digit(text)
    return text
