"""Tests for scripts/run_spike_fade_vehicle_addendum.py -- the optional
POST-HOC addendum to the range-bound spike-fade study (pre-registration spec,
version 3.1). The addendum runs a FIXED list of 24 tradable-vehicle tests (VIX
up_fade on SVXY long; WTI spot up_fade on USO short; Brent spot up_fade on BNO
short; each for the four triggers and for exactly two cells) and nothing else.

Tests are hand-derived on small synthetic inputs.

Everything runs on SYNTHETIC data: a tiny fake universe (spiky mean-reverting
series named like the real ones, with a raw-file cache and a SHA-256 manifest
built in a temp folder). There is no network, no real price, no ledger and no
trial slot in this file.

What is pinned:
  * the fixed-list guard: any pair, side, vehicle, trigger or cell outside the
    fixed list is refused BEFORE any data is touched, at the runner and at the
    lowest-level helper, and the fixed list is exactly 24 tests;
  * an end-to-end run on the fake universe: exit 0, both output files, 24
    tests, the POST-HOC / borrow-cost / multiplicity / headline statements,
    3 to 5 plain-language bullets, no dollar sign followed by a digit;
  * the sealed window: results are IDENTICAL whatever the data after the
    cutoff say;
  * fail closed: a tampered cache file stops the run (exit 3), an existing
    output is not overwritten without --force (exit 2);
  * the reconciliation of the series-level numbers against cells.csv detects a
    changed value; and the formatting / underpowered / no-trade edge cases.
"""
import csv
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _p in (SRC_ROOT, SCRIPTS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_spike_fade_study as drv  # noqa: E402
import run_spike_fade_vehicle_addendum as add  # noqa: E402
from mining import spike_fade_analysis as sfa  # noqa: E402
from mining import spike_fade_null as sfn  # noqa: E402
from mining import spike_fade_study as sfs  # noqa: E402

DOLLAR_DIGIT = re.compile(r"\$\s*\d")
CUTOFF = pd.Timestamp("2023-06-30")


# ---------------------------------------------------------------------------
# The synthetic world (a fake raw-file cache with a manifest)
# ---------------------------------------------------------------------------

def _index():
    return pd.bdate_range("2008-01-02", "2024-03-29")     # runs past the cutoff


def _spiky(n, seed, base, vol=0.015, jump_every=110, rev=0.96):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    days = set(rng.choice(np.arange(300, n - 80), size=n // jump_every, replace=False).tolist())
    for t in range(1, n):
        x[t] = rev * x[t - 1] + rng.normal(0.0, vol)
        if t in days:
            x[t] += rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.5)
    return base * np.exp(x), x


def _ohlc_bytes(close, index, seed):
    rng = np.random.default_rng(seed)
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.002, len(close)))
    df = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.003,
                       "Low": np.minimum(open_, close) * 0.997, "Close": close,
                       "Volume": 1000.0}, index=index)
    df.index.name = "Date"
    return df.to_csv(date_format="%Y-%m-%d").encode()


def _fred_bytes(values, index, symbol):
    lines = [f"observation_date,{symbol}"]
    for d, v in zip(index, values):
        lines.append(f"{d:%Y-%m-%d},{v:.6f}")
    return ("\n".join(lines) + "\n").encode()


def make_world(seed=1, post_shift=False, short_brent=False):
    """{(source, symbol): csv bytes} for the six series the addendum needs.
    `post_shift` rewrites everything AFTER the cutoff, leaving the sealed
    window untouched. `short_brent` gives the Brent trigger series only 200
    sessions of history: too few for any trigger to be defined, so zero
    episodes (the vehicle keeps its full history)."""
    idx = _index()
    n = len(idx)
    world = {}

    def shifted(c):
        if not post_shift:
            return c
        c = c.copy()
        c[idx > CUTOFF] = c[idx > CUTOFF] * 7.0
        c[np.flatnonzero(idx > CUTOFF)[10]] *= 3.0
        return c

    vix, xv = _spiky(n, seed, 20.0)
    wti, xw = _spiky(n, seed + 1, 70.0, vol=0.012)
    brent, xb = _spiky(n, seed + 2, 75.0, vol=0.012)
    world[("yahoo", "^VIX")] = _ohlc_bytes(shifted(vix), idx, seed + 10)
    world[("fred", "DCOILWTICO")] = _fred_bytes(shifted(wti), idx, "DCOILWTICO")
    if short_brent:
        world[("fred", "DCOILBRENTEU")] = _fred_bytes(brent[-200:], idx[-200:] - (idx[-1] - pd.Timestamp("2022-12-30")),
                                                      "DCOILBRENTEU")
    else:
        world[("fred", "DCOILBRENTEU")] = _fred_bytes(shifted(brent), idx, "DCOILBRENTEU")
    rng = np.random.default_rng(seed + 12)
    for tick, x, base, k in (("SVXY", xv, 60.0, -0.6), ("USO", xw, 40.0, 0.8),
                             ("BNO", xb, 25.0, 0.8)):
        v = base * np.exp(k * x + np.cumsum(rng.normal(0, 0.004, n)))
        world[("yahoo", tick)] = _ohlc_bytes(shifted(v), idx, seed + 20)
    return world


def write_cache(out, world):
    """Write the world's raw files and a manifest under <out>/data/raw."""
    raw_dir = os.path.join(str(out), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    files = {}
    for (source, symbol), raw in world.items():
        name = drv.raw_filename(source, symbol)
        with open(os.path.join(raw_dir, name), "wb") as f:
            f.write(raw)
        df = drv.parse_raw(source, raw)
        files[name] = {"kind": "x", "name": symbol, "source": source, "symbol": symbol,
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                       "rows": int(len(df)), "first_date": df.index.min().strftime("%Y-%m-%d"),
                       "last_date": df.index.max().strftime("%Y-%m-%d"),
                       "retrieved_utc": "2026-01-01T00:00:00Z"}
    drv.write_manifest(os.path.join(raw_dir, "manifest.json"), files)
    return raw_dir


def run_main(out, *extra, B=20):
    return add.main(["--out", str(out), *extra], B=B)


def load_json(out):
    with open(os.path.join(str(out), add.OUT_JSON), encoding="utf-8") as f:
        return json.load(f)


def read_md(out):
    with open(os.path.join(str(out), add.OUT_MD), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("addendum_e2e")
    write_cache(out, make_world())
    code = run_main(out)
    return out, code


# ---------------------------------------------------------------------------
# A) The fixed list and its guard
# ---------------------------------------------------------------------------

def test_the_fixed_list_is_exactly_24_tests_in_a_stable_order():
    plan = add.build_plan()
    assert len(plan) == 24
    assert len({(p.series, p.trigger, p.cell) for p in plan}) == 24
    assert {p.series for p in plan} == {"^VIX", "WTI spot", "Brent spot"}
    assert {p.side for p in plan} == {"up_fade"}
    assert {p.trigger for p in plan} == {"z20@2.0", "z20@2.5", "pct756", "pctexp"}
    assert {p.cell for p in plan} == {("E0", 21, "none", None), ("E0", 63, "none", None)}
    assert add.build_plan() == plan                       # deterministic order


def test_the_plan_agrees_with_the_studys_own_vehicle_table():
    for item in add.build_plan():
        v = sfa.VEHICLE_TABLE[(item.series, item.side)]
        assert (item.ticker, item.position) == (v.ticker, v.position)
    by = {p.series: (p.ticker, p.position) for p in add.build_plan()}
    assert by == {"^VIX": ("SVXY", "long"), "WTI spot": ("USO", "short"),
                  "Brent spot": ("BNO", "short")}


def test_the_primary_cell_and_the_frozen_constants_are_the_studys_own():
    assert ("E0", 21, "none", None) == sfs.PRIMARY_CELL
    assert add.NULL_B == 500 == sfn.NULL_B
    assert add.VEHICLE_SEED == sfs.N1_SEED == sfn.N1_SEED == 2026_09_23
    assert add.FIXED_TRIGGERS == tuple(sfs.TRIGGERS)


@pytest.mark.parametrize("series,side,ticker,trigger,cell", [
    ("GC=F", "up_fade", "GLD", "z20@2.0", ("E0", 21, "none", None)),          # pair outside
    ("Henry Hub spot", "up_fade", "UNG", "z20@2.5", ("E0", 21, "none", None)),
    ("^VIX", "down_fade", "VIXY", "z20@2.0", ("E0", 21, "none", None)),       # side outside
    ("^VIX", "up_fade", "VIXY", "z20@2.0", ("E0", 21, "none", None)),         # wrong vehicle
    ("WTI spot", "up_fade", "USO", "z20@3.0", ("E0", 21, "none", None)),      # trigger outside
    ("WTI spot", "up_fade", "USO", "pct5", ("E0", 21, "none", None)),
    ("Brent spot", "up_fade", "BNO", "pctexp", ("E0", 42, "none", None)),     # cell outside
    ("Brent spot", "up_fade", "BNO", "pctexp", ("E1", 21, "none", None)),
    ("Brent spot", "up_fade", "BNO", "pctexp", ("E0", 21, "half", None)),
    ("Brent spot", "up_fade", "BNO", "pctexp", ("E0", 63, "none", 1.5)),
])
def test_anything_outside_the_fixed_list_is_refused(series, side, ticker, trigger, cell):
    with pytest.raises(add.OutsideFixedList):
        add.check_allowed(series, side, ticker, trigger, cell)


def test_a_valid_combination_passes_the_guard():
    add.check_allowed("^VIX", "up_fade", "SVXY", "pct756", ("E0", 63, "none", None))


def test_validate_plan_refuses_a_plan_with_one_bad_item_and_touches_no_data():
    plan = list(add.build_plan())
    bad = plan[0]._replace(cell=("E0", 42, "none", None))
    with pytest.raises(add.OutsideFixedList):
        add.validate_plan(plan[:3] + [bad])
    # the runner refuses BEFORE looking at any data: empty dicts would KeyError otherwise
    with pytest.raises(add.OutsideFixedList):
        add.run_fixed_tests({}, {}, plan=plan[:3] + [bad], B=5)


def test_the_low_level_helper_carries_its_own_guard():
    with pytest.raises(add.OutsideFixedList):
        add.run_one_test(None, None, add.PlanItem("^VIX", "up_fade", "SVXY", "long", "z20@2.0",
                                                  ("E0", 10, "none", None)), None, B=5)


def test_validate_plan_accepts_a_subset_of_the_fixed_list():
    add.validate_plan(add.build_plan()[:5])


# ---------------------------------------------------------------------------
# B) End to end on the fake universe
# ---------------------------------------------------------------------------

def test_end_to_end_writes_both_outputs_and_24_results(finished_run):
    out, code = finished_run
    assert code == 0
    res = load_json(out)
    assert res["status"] == "completed"
    assert res["schema"] == add.SCHEMA
    assert len(res["tests"]) == 24
    assert res["params"]["B"] == 20 and res["params"]["vehicle_null_seed"] == "2026-09-23"
    assert res["post_hoc"] is True
    for t in res["tests"]:
        assert t["side"] == "up_fade"
        assert set(t["series_level"]) >= {"n", "mean_x", "t"}
        assert "n" in t["vehicle_result"] and "null_percentile" in t["vehicle_result"]
    assert os.path.exists(os.path.join(str(out), add.OUT_MD))


def test_every_vehicle_result_is_what_the_studys_own_function_returns(finished_run):
    """Re-run three tests directly with the study's own vehicle_test_for_pair and
    compare the headline numbers exactly."""
    out, _ = finished_run
    res = load_json(out)
    raw_dir = os.path.join(str(out), "data", "raw")
    manifest = drv.load_manifest(os.path.join(raw_dir, "manifest.json"))
    frames, n_post = {}, {}
    for source, symbol in (("yahoo", "^VIX"), ("yahoo", "SVXY")):
        need = drv.Need("x", symbol, source, symbol)
        df = drv.parse_raw(source, drv.load_cached(raw_dir, manifest, need))
        frames[(source, symbol)], n_post[(source, symbol)] = drv.seal_frame(df, CUTOFF)
    spec = next(s for s in drv.UNIVERSE if s.name == "^VIX")
    ser = drv.build_trigger_series(spec, frames, CUTOFF, n_post)
    veh = drv.build_vehicle_series("SVXY", frames, CUTOFF, n_post)
    sig = sfs.compute_triggers(ser)
    for trig, cell in (("z20@2.0", ("E0", 21, "none", None)), ("pctexp", ("E0", 63, "none", None))):
        g = sfs.evaluate_group(ser, trig, "up_fade", cells=[cell], signals=sig, bootstrap=False)
        direct = sfa.vehicle_test_for_pair("^VIX", "up_fade", veh, g.episodes.dates, cell, B=20,
                                           seed=sfs.N1_SEED, extra_lag=ser.extra_lag)
        row = next(t for t in res["tests"] if t["series"] == "^VIX" and t["trigger"] == trig
                   and t["cell"]["H"] == cell[1])
        vr = row["vehicle_result"]
        assert vr["n"] == direct.n
        for key, val in (("mean_pnl_pp", 100 * direct.mean_pnl), ("excess_pp", direct.excess_pp),
                         ("null_percentile", direct.null_percentile)):
            if math.isnan(val):
                assert vr[key] is None
            else:
                assert vr[key] == pytest.approx(val, rel=0, abs=1e-12)
        assert row["series_level"]["n"] == g.stats[cell].n
        if row["series_level"]["mean_x"] is not None:
            assert row["series_level"]["mean_x"] == pytest.approx(g.stats[cell].mean_x, abs=1e-12)


def test_the_svxy_era_split_is_reported_for_the_vix_pair_only(finished_run):
    out, _ = finished_run
    res = load_json(out)
    for t in res["tests"]:
        era = t["vehicle_result"]["era"]
        if t["series"] == "^VIX":
            assert set(era) == {"pre", "post"}
        else:
            assert era is None
    assert "2018-02-28" in read_md(out)


def test_the_markdown_states_the_required_warnings(finished_run):
    out, _ = finished_run
    md = read_md(out)
    low = md.lower()
    assert "post-hoc" in low and "optional post-hoc vehicle test" in low
    assert "24 tests" in low
    # The fake study folder has no results.json, so no headline p is invented.
    assert "headline stays at **not available" in md
    assert "cannot change" in low
    assert "borrow" in low
    assert "underpowered" in low
    assert "500" in md and "2026-09-23" in md
    assert md.index("Plain-language summary") < md.index("Results")


def test_the_plain_language_summary_has_three_to_five_bullets(finished_run):
    out, _ = finished_run
    res = load_json(out)
    bullets = add.plain_summary(res)
    assert 3 <= len(bullets) <= 5
    md = read_md(out)
    section = md.split("## Plain-language summary", 1)[1].split("\n## ", 1)[0]
    assert len([ln for ln in section.splitlines() if ln.startswith("- ")]) == len(bullets)


def test_no_output_contains_a_dollar_sign_followed_by_a_digit(finished_run):
    out, _ = finished_run
    for name in (add.OUT_MD, add.OUT_JSON):
        with open(os.path.join(str(out), name), encoding="utf-8") as f:
            assert not DOLLAR_DIGIT.search(f.read()), name


def test_underpowered_tests_are_listed_when_fewer_than_8_vehicle_trades(finished_run):
    out, _ = finished_run
    res = load_json(out)
    small = [t for t in res["tests"] if t["vehicle_result"]["n"] < 8]
    listed = res["underpowered"]
    assert len(listed) == len(small)
    assert all(t["vehicle_result"]["underpowered_on_vehicle"] == (t["vehicle_result"]["n"] < 8)
               for t in res["tests"])


# ---------------------------------------------------------------------------
# C) Sealed window, fail closed, no overwrite
# ---------------------------------------------------------------------------

def _stable(res):
    return {"tests": res["tests"], "underpowered": res["underpowered"],
            "vehicle_series": res["vehicle_series"]}


def test_results_are_identical_whatever_the_data_after_the_cutoff_say(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_cache(a, make_world())
    write_cache(b, make_world(post_shift=True))
    assert run_main(a) == 0 and run_main(b) == 0
    ra, rb = load_json(a), load_json(b)
    assert json.dumps(_stable(ra), sort_keys=True) == json.dumps(_stable(rb), sort_keys=True)
    for name, info in ra["data"]["files"].items():
        assert info["last_date"] <= "2023-06-30"


def test_no_series_or_vehicle_reaches_past_the_cutoff(finished_run):
    out, _ = finished_run
    res = load_json(out)
    for info in res["vehicle_series"].values():
        assert info["last_date"] <= "2023-06-30"
    for info in res["data"]["files"].values():
        assert info["last_date"] <= "2023-06-30"


def test_a_tampered_cache_file_fails_closed(tmp_path):
    write_cache(tmp_path, make_world())
    victim = os.path.join(str(tmp_path), "data", "raw", drv.raw_filename("yahoo", "SVXY"))
    with open(victim, "ab") as f:
        f.write(b"\n")
    assert run_main(tmp_path) == 3
    assert not os.path.exists(os.path.join(str(tmp_path), add.OUT_MD))
    assert not os.path.exists(os.path.join(str(tmp_path), add.OUT_JSON))


def test_a_missing_cache_file_fails_closed(tmp_path):
    write_cache(tmp_path, make_world())
    os.remove(os.path.join(str(tmp_path), "data", "raw", drv.raw_filename("yahoo", "BNO")))
    assert run_main(tmp_path) == 3


def test_existing_outputs_are_not_overwritten_without_force(tmp_path):
    write_cache(tmp_path, make_world())
    assert run_main(tmp_path) == 0
    before = read_md(tmp_path)
    assert run_main(tmp_path) == 2
    assert read_md(tmp_path) == before
    assert run_main(tmp_path, "--force") == 0


def test_the_cli_has_no_way_to_add_a_test_or_change_b():
    for flag in ("--pair", "--trigger", "--cell", "--B", "--universe", "--cutoff"):
        with pytest.raises(SystemExit):
            add.parse_args([flag, "x"])


def test_a_series_with_no_episodes_reports_not_available_without_crashing(tmp_path):
    write_cache(tmp_path, make_world(short_brent=True))
    assert run_main(tmp_path) == 0
    res = load_json(tmp_path)
    brent = [t for t in res["tests"] if t["series"] == "Brent spot"]
    assert len(brent) == 8
    assert all(t["vehicle_result"]["n"] == 0 for t in brent)
    assert all(t["vehicle_result"]["excess_pp"] is None for t in brent)
    md = read_md(tmp_path)
    assert "n/a" in md
    assert len(res["underpowered"]) >= 8


# ---------------------------------------------------------------------------
# D) Reconciliation against cells.csv, formatting, the dollar guard
# ---------------------------------------------------------------------------

def _cells_csv(path, rows):
    cols = ["series", "trigger", "side", "entry", "H", "tp", "sl", "n", "mean_x", "t"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def _fake_test_row(n=12, mean_x=0.25, t=1.5):
    return {"series": "^VIX", "side": "up_fade", "trigger": "z20@2.0",
            "cell": {"entry": "E0", "H": 21, "tp": "none", "sl": "none"},
            "series_level": {"n": n, "mean_x": mean_x, "t": t}}


def test_reconciliation_passes_when_the_numbers_match_cells_csv(tmp_path):
    p = str(tmp_path / "cells.csv")
    _cells_csv(p, [{"series": "^VIX", "trigger": "z20@2.0", "side": "up_fade", "entry": "E0",
                    "H": 21, "tp": "none", "sl": "none", "n": 12, "mean_x": repr(0.25), "t": repr(1.5)}])
    out = add.reconcile_series_level([_fake_test_row()], p)
    assert out["status"] == "checked" and out["n_checked"] == 1 and out["mismatches"] == []


@pytest.mark.parametrize("field,value", [("n", 13), ("mean_x", repr(0.2500001)), ("t", repr(1.6))])
def test_reconciliation_detects_a_changed_value(tmp_path, field, value):
    p = str(tmp_path / "cells.csv")
    row = {"series": "^VIX", "trigger": "z20@2.0", "side": "up_fade", "entry": "E0", "H": 21,
           "tp": "none", "sl": "none", "n": 12, "mean_x": repr(0.25), "t": repr(1.5)}
    row[field] = value
    _cells_csv(p, [row])
    with pytest.raises(add.ReconciliationError):
        add.reconcile_series_level([_fake_test_row()], p)


def test_reconciliation_detects_a_missing_cells_row(tmp_path):
    p = str(tmp_path / "cells.csv")
    _cells_csv(p, [])
    with pytest.raises(add.ReconciliationError):
        add.reconcile_series_level([_fake_test_row()], p)


def test_reconciliation_is_skipped_when_there_is_no_cells_csv(tmp_path):
    out = add.reconcile_series_level([_fake_test_row()], str(tmp_path / "nope.csv"))
    assert out["status"] == "skipped" and out["n_checked"] == 0


def test_number_formatting():
    assert add.fmt_pp(4.4849) == "+4.48"
    assert add.fmt_pp(-0.5) == "-0.50"
    assert add.fmt_pp(0.0) == "+0.00"
    assert add.fmt_pp(float("nan")) == "n/a" and add.fmt_pp(None) == "n/a"
    assert add.fmt_num(0.123456, 3) == "0.123" and add.fmt_num(None, 3) == "n/a"
    assert add.fmt_share(0.8449) == "0.84" and add.fmt_share(None) == "n/a"
    assert add.fmt_pct(0.5714) == "57%" and add.fmt_pct(None) == "n/a"


def test_the_dollar_guard():
    add.assert_no_dollar_digit("a price of five dollars and 50 percent")
    with pytest.raises(ValueError):
        add.assert_no_dollar_digit("costs " + "$" + "5")
    with pytest.raises(ValueError):
        add.assert_no_dollar_digit("costs " + "$ " + "12")


def _results_with(vehicle_by_test):
    """A results dict with the given per-test vehicle overrides, built from the
    fixed plan (everything else neutral)."""
    tests = []
    for i, p in enumerate(add.build_plan()):
        vr = {"n": 20, "mean_pnl_pp": 1.0, "hit_rate": 0.55, "null_mean_pnl_pp": 0.5,
              "excess_pp": 0.5, "excess_sigma": 0.05, "null_percentile": 0.6,
              "unconditional_mean_pnl_pp": 0.4, "unconditional_mean_daily_return_pp": 0.03,
              "window_start": "2012-01-03", "window_end": "2023-06-30", "era": None,
              "underpowered_on_vehicle": False, "contaminated": True,
              "n_anchors_input": 30, "dropped_no_session": 2, "dropped_past_end": 0,
              "dropped_short_history": 0, "dropped_censored": 1, "dropped_duplicate": 0,
              "n_anchors_kept": 27, "n_kept_not_traded": 7, "null_reps_used": 500,
              "null_reps_failed": 0, "trades": []}
        vr.update(vehicle_by_test.get(i, {}))
        tests.append({"series": p.series, "side": p.side, "vehicle": p.ticker,
                      "position": p.position, "trigger": p.trigger,
                      "cell": {"entry": p.cell[0], "H": p.cell[1], "tp": p.cell[2], "sl": "none"},
                      "series_level": {"n": 30, "mean_x": 0.2, "t": 1.1, "eligible": True},
                      "vehicle_result": vr})
    return {"schema": add.SCHEMA, "status": "completed", "post_hoc": True,
            "cutoff": "2023-06-30",
            "params": {"B": 500, "vehicle_null_seed": "2026-09-23", "n_tests": 24,
                       "era_split": "2018-02-28"},
            "tests": tests, "underpowered": [], "data": {"files": {}}, "vehicle_series": {},
            "vehicle_first_dates": {}, "reconciliation": {"status": "skipped", "n_checked": 0,
                                                          "mismatches": []},
            "vehicle_reconciliation": {"status": "skipped", "n_checked": 0, "mismatches": []},
            "confirmatory_headline": {"p_headline": 0.6, "source": "test"}}


def test_the_headline_is_read_from_the_results_and_judged_by_the_studys_rule():
    """Synthetic p values only: a value above the study's bar reads as not
    convincing, one at or below it as meeting the bar, a missing one as not
    available."""
    res = _results_with({})
    assert "p = 0.600 (not convincing)" in add.render_markdown(res)
    res["confirmatory_headline"] = {"p_headline": 0.05, "source": "test"}
    assert "p = 0.050 (meets the study's bar)" in add.render_markdown(res)
    res["confirmatory_headline"] = {"p_headline": None, "source": "test"}
    assert "not available" in add.render_markdown(res)
    assert add.read_study_headline(os.path.join("no", "such", "results.json"))["p_headline"] is None


def test_the_summary_counts_look_good_tests_only_among_powered_ones():
    res = _results_with({
        0: {"null_percentile": 0.97, "excess_pp": 3.0},                              # looks good
        1: {"null_percentile": 0.99, "excess_pp": 9.0, "n": 3,
            "underpowered_on_vehicle": True},                                        # underpowered
        2: {"null_percentile": 0.85, "excess_pp": 1.0}})                             # good but < 0.90
    s = add.summarize(res)
    assert s["n_tests"] == 24
    assert s["n_underpowered"] == 1
    assert s["n_looks_good"] == 1
    assert s["expected_by_luck"] == pytest.approx(2.4)


def test_the_summary_text_never_claims_an_edge_from_underpowered_tests():
    res = _results_with({i: {"n": 2, "underpowered_on_vehicle": True, "null_percentile": 0.99,
                             "excess_pp": 8.0} for i in range(0, 8)})
    text = " ".join(add.plain_summary(res)).lower()
    assert "too few" in text or "underpowered" in text
    assert not DOLLAR_DIGIT.search(text)


def test_the_markdown_lists_underpowered_tests_with_their_n():
    res = _results_with({3: {"n": 5, "underpowered_on_vehicle": True}})
    res["underpowered"] = [{"series": "^VIX", "trigger": "z20@2.5", "H": 63, "vehicle": "SVXY", "n": 5}]
    md = add.render_markdown(res)
    assert "Underpowered" in md and "n = 5" in md
    assert not DOLLAR_DIGIT.search(md)


def test_the_summary_words_all_none_and_partial_counts_correctly():
    all_good = _results_with({i: {"null_percentile": 0.97, "excess_pp": 3.0} for i in range(8)})
    text = add.plain_summary(all_good)[1]                  # the VIX bullet
    assert "all 8 tests had enough trades" in text
    assert "in all 8 of them" in text
    assert "Some of these tests look good" in text
    none_good = _results_with({i: {"null_percentile": 0.30, "excess_pp": -2.0} for i in range(8)})
    text = add.plain_summary(none_good)[1]
    assert "in none of them" in text and "did worse than random entry dates" in text
    partial = _results_with({0: {"null_percentile": 0.99, "excess_pp": 5.0},
                             1: {"null_percentile": 0.40, "excess_pp": -1.0}})
    text = add.plain_summary(partial)[1]
    assert "in 7 of them" in text


def test_the_luck_bullet_compares_the_count_with_the_expected_2_or_3():
    few = _results_with({0: {"null_percentile": 0.97, "excess_pp": 3.0}})
    assert "no more than luck alone would give" in add.plain_summary(few)[3]
    many = _results_with({i: {"null_percentile": 0.97, "excess_pp": 3.0} for i in range(6)})
    assert "not independent chances" in add.plain_summary(many)[3]


def test_render_only_rebuilds_the_same_markdown_and_refuses_without_results(finished_run,
                                                                            tmp_path):
    out, _ = finished_run
    before = read_md(out)
    assert add.main(["--out", str(out), "--render-only"]) == 0
    assert read_md(out) == before
    assert add.main(["--out", str(tmp_path), "--render-only"]) == 2


def test_a_run_creates_only_the_two_outputs_and_leaves_the_cache_untouched(tmp_path):
    write_cache(tmp_path, make_world())

    def snapshot():
        snap = {}
        for root, _dirs, files in os.walk(str(tmp_path)):
            for name in files:
                path = os.path.join(root, name)
                with open(path, "rb") as f:
                    snap[os.path.relpath(path, str(tmp_path))] = hashlib.sha256(f.read()).hexdigest()
        return snap

    before = snapshot()
    assert run_main(tmp_path) == 0
    after = snapshot()
    assert set(after) - set(before) == {add.OUT_MD, add.OUT_JSON}
    assert all(after[k] == v for k, v in before.items())


def test_the_first_bullet_gives_the_short_answer_up_front():
    none_good = _results_with({i: {"null_percentile": 0.30, "excess_pp": -2.0} for i in range(24)})
    first = add.plain_summary(none_good)[0]
    assert first.startswith("Short answer: no.")
    good = _results_with({i: {"null_percentile": 0.97, "excess_pp": 3.0} for i in range(24)})
    assert add.plain_summary(good)[0].startswith("Short answer: not clearly.")
    tiny = _results_with({i: {"n": 2, "underpowered_on_vehicle": True} for i in range(24)})
    assert "no test had enough trades" in add.plain_summary(tiny)[0]


def test_the_fund_history_bullet_counts_spikes_that_predate_the_fund():
    res = _results_with({0: {"n_anchors_input": 57, "dropped_no_session": 35}})
    res["vehicle_first_dates"] = {"SVXY": {"observed": "2011-10-04"}}
    b = add.fund_history_bullet(res)
    assert "35 of 57 VIX spikes" in b and "SVXY on 2011-10-04" in b
    md = add.render_markdown(res)
    assert "The funds are young" in md and "The funds are futures-based" in md
    assert not DOLLAR_DIGIT.search(md)


_NUM_SHAPE = re.compile(r"(?<!\d\.)\b\d{8,17}\b")     # a bare run of 8 to 17 digits


def test_no_output_or_source_contains_a_bare_8_to_17_digit_number(finished_run):
    """A bare run of 8 to 17 digits reads like an account number, so the seed
    is written as the date it is (2026-09-23), never as one number, in both
    outputs, the script and this test file."""
    out, _ = finished_run
    paths = [os.path.join(str(out), add.OUT_MD), os.path.join(str(out), add.OUT_JSON),
             os.path.join(SCRIPTS_ROOT, "run_spike_fade_vehicle_addendum.py"),
             os.path.abspath(__file__)]
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                assert not _NUM_SHAPE.search(line), f"{os.path.basename(path)}:{n}: {line.strip()}"


def test_the_seed_is_shown_as_its_date_form():
    assert add.seed_text(2026_09_23) == "2026-09-23"
    assert add.seed_text(2026_12_01) == "2026-12-01"
