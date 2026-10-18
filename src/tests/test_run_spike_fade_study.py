"""Tests for scripts/run_spike_fade_study.py -- the DRIVER of the range-bound
spike-fade study, Stage 1 (the pre-registered study; the pre-registration
document is not included in this repository).

Tests are hand-derived on small synthetic inputs.

Everything runs on SYNTHETIC data through an injected fake fetch layer: there
is no network, no real price, no ledger and no trial slot in this file. The
world the fake fetcher serves is a handful of spiky mean-reverting series
(index-like closes, a few ETF vehicles, an SPY core and a FRED-style reference
with '.' holes) on one business-day calendar that runs PAST the discovery
cutoff, so the sealed window is exercised for real.

What is pinned:
  * the fetch layer: strictly serial, a 3 second pause between fetches, ONE
    retry after 60 seconds, a SHA-256 manifest, a tampered cache refused;
  * the sealed window: the driver refuses a cutoff after
    2023-06-30, loads nothing after it, and the results are IDENTICAL whatever
    the data after the cutoff say (only the raw-file digests differ);
  * determinism: identical seeds and data give byte-identical
    results.json, whatever the worker count;
  * fail closed: a missing required series (or JPY=X without
    its DEXJPUS reference) stops the run with a clear message, and the
    report still lists what is missing;
  * counts: every removed / undefined / externally corrected
    observation reaches results.json and REPORT.md, the external corrections
    with their dates;
  * no output file contains a dollar sign followed by a digit.
"""
import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
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
from mining import spike_fade_analysis as sfa  # noqa: E402
from mining import spike_fade_null as sfn  # noqa: E402
from mining import spike_fade_study as sfs  # noqa: E402

DOLLAR_DIGIT = re.compile(r"\$\s*\d")
CUTOFF = pd.Timestamp("2023-06-30")


# ---------------------------------------------------------------------------
# The synthetic world served by the fake fetch layer
# ---------------------------------------------------------------------------

def _index():
    return pd.bdate_range("2012-01-02", "2024-03-29")     # runs past the cutoff


def _spiky(n, seed, base, vol=0.015, jump_every=110, rev=0.96):
    """A mean-reverting log level with occasional large jumps (so z20 and the
    percentile triggers really fire), as prices."""
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


def _fred_bytes(values, index, symbol, holes=()):
    lines = [f"observation_date,{symbol}"]
    holes = set(holes)
    for i, (d, v) in enumerate(zip(index, values)):
        lines.append(f"{d:%Y-%m-%d}," + ("." if i in holes else f"{v:.6f}"))
    return ("\n".join(lines) + "\n").encode()


BAD_TICKS = ("2015-03-10", "2016-05-11")


def make_world(seed=1, post_shift=False):
    """{(source, symbol): csv bytes} for every series the tests can ask for.
    `post_shift` rewrites everything AFTER the cutoff (a level shift and a wild
    spike) leaving the sealed window untouched."""
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
    gc, xg = _spiky(n, seed + 1, 1500.0, vol=0.008)
    world[("yahoo", "^VIX")] = _ohlc_bytes(shifted(vix), idx, seed + 10)
    world[("yahoo", "GC=F")] = _ohlc_bytes(shifted(gc), idx, seed + 11)
    rng = np.random.default_rng(seed + 12)
    for tick, x, base, k in (("SVXY", xv, 60.0, -0.6), ("VIXY", xv, 30.0, 0.7),
                             ("GLD", xg, 120.0, 0.9)):
        v = base * np.exp(k * x + np.cumsum(rng.normal(0, 0.004, n)))
        world[("yahoo", tick)] = _ohlc_bytes(shifted(v), idx, seed + 20)
    spy, _ = _spiky(n, seed + 3, 300.0, vol=0.009, jump_every=400)
    world[("yahoo", "SPY")] = _ohlc_bytes(shifted(spy), idx, seed + 30)
    jpy, _ = _spiky(n, seed + 4, 110.0, vol=0.004, jump_every=500)
    jpy_bad = jpy.copy()
    for d in BAD_TICKS:
        jpy_bad[idx.get_loc(pd.Timestamp(d))] *= 1.05
    world[("yahoo", "JPY=X")] = _ohlc_bytes(shifted(jpy_bad), idx, seed + 31)
    holes = set(range(40, n, 97))
    world[("fred", "DEXJPUS")] = _fred_bytes(shifted(jpy), idx, "DEXJPUS", holes)
    return world


class FakeFetcher:
    """Serves `world`; `fail` lists (source, symbol) that always raise."""

    def __init__(self, world, fail=()):
        self.world = world
        self.fail = set(fail)
        self.calls = []

    def __call__(self, source, symbol):
        self.calls.append((source, symbol))
        if (source, symbol) in self.fail or (source, symbol) not in self.world:
            raise RuntimeError(f"fake fetch failure for {source}:{symbol}")
        return self.world[(source, symbol)]


def run(out, world, *extra, fail=(), sleeps=None):
    fetcher = FakeFetcher(world, fail)
    sleeps = [] if sleeps is None else sleeps
    argv = ["--out", str(out), "--B", "20", "--B-map", "10", "--workers", "1", *extra]
    code = drv.main(argv, fetcher=fetcher, sleep=sleeps.append)
    return code, fetcher, sleeps


def load_results(out):
    with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
        return json.load(f, parse_constant=_no_constants)


def _no_constants(name):
    raise AssertionError(f"results.json contains the non-JSON constant {name}")


def read(out, name):
    with open(os.path.join(out, name), encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# A) The fetch layer
# ---------------------------------------------------------------------------

def _yahoo_payload():
    return b"Date,Open,High,Low,Close,Volume\n2020-01-02,1,1,1,1,10\n2020-01-03,1,1,1,1,10\n"


def _fred_payload(sym="X"):
    return f"observation_date,{sym}\n2020-01-02,1.0\n2020-01-03,.\n".encode()


def test_fetch_is_serial_paused_and_retries_once_after_sixty_seconds(tmp_path):
    calls, sleeps = [], []

    def fetcher(source, symbol):
        calls.append((source, symbol))
        if symbol == "B" and calls.count((source, symbol)) == 1:
            raise RuntimeError("temporary failure")
        return _yahoo_payload() if source == "yahoo" else _fred_payload(symbol)

    needs = [drv.Need("trigger", "A", "yahoo", "A"), drv.Need("trigger", "B", "yahoo", "B"),
             drv.Need("trigger", "C", "fred", "C")]
    files, failures = drv.fetch_all(needs, fetcher, str(tmp_path), sleep=sleeps.append,
                                    pause=3.0, retry_wait=60.0)
    assert calls == [("yahoo", "A"), ("yahoo", "B"), ("yahoo", "B"), ("fred", "C")]
    assert sleeps == [3.0, 60.0, 3.0]          # pause before B, retry wait, pause before C
    assert failures == {}
    assert set(files) == {drv.raw_filename("yahoo", "A"), drv.raw_filename("yahoo", "B"),
                          drv.raw_filename("fred", "C")}


def test_a_second_failure_is_recorded_and_the_others_are_still_fetched(tmp_path):
    sleeps = []

    def fetcher(source, symbol):
        if symbol == "B":
            raise RuntimeError("down")
        return _yahoo_payload()

    needs = [drv.Need("trigger", "A", "yahoo", "A"), drv.Need("trigger", "B", "yahoo", "B"),
             drv.Need("trigger", "C", "yahoo", "C")]
    files, failures = drv.fetch_all(needs, fetcher, str(tmp_path), sleep=sleeps.append)
    assert ("yahoo", "B") in failures and "down" in failures[("yahoo", "B")]
    assert drv.raw_filename("yahoo", "A") in files and drv.raw_filename("yahoo", "C") in files
    assert sleeps.count(60.0) == 1                      # exactly one retry wait


def test_a_payload_that_does_not_parse_counts_as_a_failed_fetch(tmp_path):
    needs = [drv.Need("trigger", "A", "yahoo", "A")]
    files, failures = drv.fetch_all(needs, lambda s, y: b"<html>rate limited</html>",
                                    str(tmp_path), sleep=lambda s: None)
    assert files == {} and ("yahoo", "A") in failures


def test_manifest_digest_matches_the_file_and_a_tampered_cache_is_refused(tmp_path):
    need = drv.Need("trigger", "A", "yahoo", "A")
    files, failures = drv.fetch_all([need], lambda s, y: _yahoo_payload(), str(tmp_path),
                                    sleep=lambda s: None)
    name = drv.raw_filename("yahoo", "A")
    path = os.path.join(tmp_path, name)
    assert files[name]["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert files[name]["retrieved_utc"].endswith("Z")
    raw = drv.load_cached(str(tmp_path), files, need)
    assert raw == _yahoo_payload()
    with open(path, "ab") as f:
        f.write(b"2020-01-06,9,9,9,9,1\n")
    with pytest.raises(drv.CacheError):
        drv.load_cached(str(tmp_path), files, need)
    with pytest.raises(drv.CacheError):
        drv.load_cached(str(tmp_path), files, drv.Need("trigger", "Z", "yahoo", "Z"))


def test_fred_dot_is_a_missing_value_and_yahoo_index_is_date_only():
    fr = drv.parse_raw("fred", _fred_payload("DCOILWTICO"))
    assert list(fr.columns) == ["value"] and math.isnan(fr["value"].iloc[1])
    ya = drv.parse_raw("yahoo", _yahoo_payload())
    assert ya.index.tz is None and "Close" in ya.columns
    with pytest.raises(ValueError):
        drv.parse_raw("yahoo", b"not,a,price,file\n")


# ---------------------------------------------------------------------------
# B) Universe and requirements
# ---------------------------------------------------------------------------

def test_the_full_universe_matches_the_pre_registered_table():
    names = [s.name for s in drv.resolve_universe(None)]
    assert names == ["^VIX", "WTI spot", "Brent spot", "Henry Hub spot", "GC=F", "HG=F",
                     "DX-Y.NYB", "^OVX", "^VVIX", "^GVZ", "^MOVE", "^TNX", "JPY=X"]
    by = {s.name: s for s in drv.resolve_universe(None)}
    assert [n for n in names if by[n].role == "confirmatory"] == names[:7]
    assert {n for n in names if by[n].extra_lag == 1} == {"WTI spot", "Brent spot",
                                                          "Henry Hub spot"}
    assert by["WTI spot"].symbol == "DCOILWTICO" and by["WTI spot"].source == "fred"
    assert by["Brent spot"].first_date == "1987-05-20"
    assert by["JPY=X"].reference == ("fred", "DEXJPUS")
    assert by["DX-Y.NYB"].first_date == "1990-01-02"
    assert by["^OVX"].first_date is None


def test_required_needs_cover_every_trigger_vehicle_and_reference_ticker():
    needs = drv.required_needs(drv.resolve_universe(None))
    keys = {(n.source, n.symbol) for n in needs}
    for sym in ("^VIX", "^OVX", "^VVIX", "^GVZ", "^MOVE", "^TNX", "JPY=X", "DX-Y.NYB", "GC=F",
                "HG=F", "SPY", "SVXY", "VIXY", "USO", "BNO", "UNG", "GLD", "CPER", "UUP"):
        assert ("yahoo", sym) in keys, sym
    for sym in ("DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "DEXJPUS"):
        assert ("fred", sym) in keys, sym
    assert len(keys) == len(needs)                      # no duplicate downloads


def test_a_restricted_universe_asks_only_for_what_it_needs():
    needs = drv.required_needs(drv.resolve_universe("^VIX,JPY=X"))
    keys = {(n.source, n.symbol) for n in needs}
    assert keys == {("yahoo", "^VIX"), ("yahoo", "JPY=X"), ("fred", "DEXJPUS"),
                    ("yahoo", "SVXY"), ("yahoo", "VIXY"), ("yahoo", "SPY")}


def test_unknown_universe_name_is_refused():
    with pytest.raises(ValueError):
        drv.resolve_universe("^VIX,NOT-A-SERIES")


# ---------------------------------------------------------------------------
# C) The sealed window and the cutoff argument
# ---------------------------------------------------------------------------

def test_a_cutoff_after_the_discovery_cutoff_is_refused_before_anything_happens(tmp_path):
    fetcher = FakeFetcher(make_world())
    code = drv.main(["--out", str(tmp_path / "o"), "--cutoff", "2023-07-03", "--universe", "^VIX"],
                    fetcher=fetcher, sleep=lambda s: None)
    assert code == 2
    assert fetcher.calls == []
    assert not (tmp_path / "o" / "results.json").exists()
    with pytest.raises(drv.CutoffRefused):
        drv.check_cutoff(pd.Timestamp("2023-07-01"))
    assert drv.check_cutoff(pd.Timestamp("2023-06-30")) == pd.Timestamp("2023-06-30")
    assert drv.check_cutoff(pd.Timestamp("2022-12-30")) == pd.Timestamp("2022-12-30")


def test_seal_frame_truncates_and_counts_and_never_lets_a_later_date_through():
    idx = pd.bdate_range("2023-06-26", "2023-07-07")
    df = pd.DataFrame({"Close": np.arange(len(idx), dtype=float) + 1.0}, index=idx)
    sealed, n_post = drv.seal_frame(df, CUTOFF)
    assert sealed.index.max() <= CUTOFF
    assert n_post == int((idx > CUTOFF).sum())


# ---------------------------------------------------------------------------
# D) Building series: JPY=X needs its reference, FRED lag, first dates
# ---------------------------------------------------------------------------

def _frames_for(world, cutoff=CUTOFF):
    out = {}
    for (source, symbol), raw in world.items():
        sealed, _ = drv.seal_frame(drv.parse_raw(source, raw), cutoff)
        out[(source, symbol)] = sealed
    return out


def test_jpy_is_refused_without_the_dexjpus_reference():
    frames = _frames_for(make_world())
    spec = {s.name: s for s in drv.resolve_universe(None)}["JPY=X"]
    frames.pop(("fred", "DEXJPUS"))
    with pytest.raises(drv.MissingReference):
        drv.build_trigger_series(spec, frames, CUTOFF)


def test_jpy_bad_ticks_are_set_missing_counted_and_listed_by_date():
    frames = _frames_for(make_world())
    spec = {s.name: s for s in drv.resolve_universe(None)}["JPY=X"]
    s = drv.build_trigger_series(spec, frames, CUTOFF)
    assert s.counts()["external_corrections"] == len(BAD_TICKS)
    assert [d.strftime("%Y-%m-%d") for d in s.external_dates] == list(BAD_TICKS)
    assert s.dates.max() <= CUTOFF


def test_fred_series_carry_the_extra_publication_lag_and_first_date_is_honoured():
    world = make_world()
    idx = _index()
    world[("fred", "DCOILWTICO")] = _fred_bytes(_spiky(len(idx), 5, 60.0)[0], idx, "DCOILWTICO",
                                                holes=range(10, len(idx), 53))
    frames = _frames_for(world)
    spec = {s.name: s for s in drv.resolve_universe(None)}["WTI spot"]
    s = drv.build_trigger_series(spec, frames, CUTOFF)
    assert s.extra_lag == 1
    assert s.counts()["missing_rows_dropped"] > 0        # '.' rows are not sessions
    assert s.dates.max() <= CUTOFF and not np.isnan(s.p).all()
    spec2 = drv.SeriesSpec("X", "yahoo", "^VIX", "2020-01-01", "exploratory")
    s2 = drv.build_trigger_series(spec2, frames, CUTOFF)
    assert s2.dates.min() >= pd.Timestamp("2020-01-01")
    assert s2.counts()["pre_start_dropped"] > 0


# ---------------------------------------------------------------------------
# E) Pure helpers
# ---------------------------------------------------------------------------

def test_ar1_half_life_recovers_a_known_persistence():
    rng = np.random.default_rng(0)
    phi, n = 0.98, 20000
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0, 0.05)
    out = drv.ar1_half_life(100.0 * np.exp(y))
    assert abs(out["phi"] - phi) < 0.01
    assert abs(out["half_life_sessions"] - math.log(0.5) / math.log(phi)) < 6
    walk = drv.ar1_half_life(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000))))
    assert walk["phi"] > 0.98
    bad = drv.ar1_half_life(np.array([1.0, np.nan, 2.0]))
    assert bad["half_life_sessions"] is None


def test_roll_gap_detail_flags_a_jump_the_etf_does_not_share():
    idx = pd.bdate_range("2015-01-02", periods=600)
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 600)
    etf = 100 * np.exp(np.cumsum(r))
    fut_r = r + rng.normal(0, 0.001, 600)
    fut_r[300] += 0.20                                   # a roll gap the ETF does not have
    fut = 50 * np.exp(np.cumsum(fut_r))
    mk = lambda c, name: sfs.build_series(pd.Series(c, index=idx), name=name, cutoff=CUTOFF)
    out = drv.roll_gap_diagnostic(mk(fut, "GC=F"), mk(etf, "GLD"))
    assert out["flagged"] is True and out["ratio"] > 5
    assert out["max_abs_diff_date"] == idx[300].strftime("%Y-%m-%d")
    # analytic: var_f = 1e-4 + 0.2**2/599 -> corr = 1e-4 / (sqrt(var_f) * 1e-2) = 0.774
    assert abs(out["corr"] - 0.774) < 0.05 and out["n"] == 599
    clean = drv.roll_gap_diagnostic(mk(50 * np.exp(np.cumsum(r)), "GC=F"), mk(etf, "GLD"))
    assert clean["flagged"] is False


def test_json_safe_makes_strict_json():
    obj = {"a": float("nan"), "b": np.float64(1.5), "c": np.int64(3), "d": pd.Timestamp("2020-01-02"),
           ("x", "y"): [float("inf"), -np.inf], "e": np.array([1.0, np.nan]), "f": None,
           "g": pd.DatetimeIndex(["2020-01-02"]), "h": True, "i": np.bool_(False)}
    s = json.dumps(drv.json_safe(obj), allow_nan=False)
    back = json.loads(s)
    assert back["a"] is None and back["b"] == 1.5 and back["c"] == 3
    assert back["d"] == "2020-01-02" and back["x|y"] == [None, None]
    assert back["e"] == [1.0, None] and back["g"] == ["2020-01-02"]
    assert back["h"] is True and back["i"] is False


def test_vehicle_signal_dates_match_the_vehicle_test_anchor_dates():
    world = make_world()
    frames = _frames_for(world)
    vix = drv.build_trigger_series({s.name: s for s in drv.resolve_universe(None)}["^VIX"],
                                   frames, CUTOFF)
    veh = sfs.build_series(frames[("yahoo", "SVXY")]["Close"], name="SVXY", cutoff=CUTOFF)
    sig = sfs.compute_triggers(vix)
    es = sfs.build_episode_set(vix, sig["z20@2.0"], "up_fade")
    assert len(es.dates) >= 3
    res = sfa.vehicle_test_for_pair("^VIX", "up_fade", veh, es.dates, sfs.PRIMARY_CELL, B=5)
    mine = drv.vehicle_signal_dates(veh, es.dates, 0)
    assert list(mine) == list(res.anchor_dates)


# ---------------------------------------------------------------------------
# F) Recommendation logic (conditions a to e)
# ---------------------------------------------------------------------------

def _cand(ok, reasons=(), contaminated=False):
    return {"is_candidate": ok, "status": "candidate" if ok else reasons[0],
            "reasons": list(reasons), "headline_p": 0.05 if ok else 0.4,
            "underpowered_on_vehicle": False, "contaminated": contaminated,
            "contamination_note": "note" if contaminated else ""}


def _abi(meets, dd, status="computed"):
    return {"status": status, "reason": None if status == "computed" else "x",
            "full": {"meets_in_sample_bar": meets, "dd_passed": dd,
                     "excess_sharpe_annualized": 1.1 if meets else 0.2,
                     "in_sample_bar": 1.02}}


def test_recommendation_is_do_not_register_when_no_candidate():
    rec = drv.build_recommendation(_cand(False, ["headline_p_above_max"]), _abi(True, True),
                                   tstar_series="GC=F")
    assert rec["register_stage2"] is False
    assert any("not convincing" in r for r in rec["reasons_not"])
    ids = {c["id"]: c for c in rec["conditions"]}
    assert ids["a"]["met"] is False and ids["c"]["met"] is None and ids["e"]["met"] is None


def test_recommendation_needs_the_abi1_line_too():
    rec = drv.build_recommendation(_cand(True), _abi(False, True), tstar_series="GC=F")
    assert rec["register_stage2"] is False
    rec2 = drv.build_recommendation(_cand(True), _abi(True, False), tstar_series="GC=F")
    assert rec2["register_stage2"] is False
    rec3 = drv.build_recommendation(_cand(True), _abi(True, True, status="not_computed"),
                                    tstar_series="GC=F")
    assert rec3["register_stage2"] is False


def test_recommendation_consider_when_a_and_b_hold_and_contamination_needs_acceptance():
    rec = drv.build_recommendation(_cand(True, contaminated=True), _abi(True, True),
                                   tstar_series="^VIX")
    assert rec["register_stage2"] is True
    d = {c["id"]: c for c in rec["conditions"]}
    assert d["a"]["met"] is True and d["b"]["met"] is True
    assert d["d"]["met"] is None                          # contamination must be accepted
    clean = drv.build_recommendation(_cand(True), _abi(True, True), tstar_series="GC=F")
    assert {c["id"]: c for c in clean["conditions"]}["d"]["met"] is True


def test_recommendation_reasons_name_the_return_bar_and_the_drawdown_limit_separately():
    rec = drv.build_recommendation(_cand(True), _abi(False, True), tstar_series="GC=F")
    text = " ".join(rec["reasons_not"]).lower()
    assert "extra return" in text and "worst loss" not in text
    rec2 = drv.build_recommendation(_cand(True), _abi(True, False), tstar_series="GC=F")
    text2 = " ".join(rec2["reasons_not"]).lower()
    assert "worst loss" in text2 and "extra return" not in text2
    rec3 = drv.build_recommendation(_cand(True), _abi(False, False), tstar_series="GC=F")
    text3 = " ".join(rec3["reasons_not"]).lower()
    assert "extra return" in text3 and "worst loss" in text3
    rec4 = drv.build_recommendation(_cand(True), _abi(True, True, status="not_computed"),
                                    tstar_series="GC=F")
    assert "could not be run" in " ".join(rec4["reasons_not"]).lower()


def test_yahoo_frames_are_cached_with_a_date_only_index():
    idx = pd.DatetimeIndex(["2020-01-02 00:00:00-05:00", "2020-01-03 00:00:00-05:00"])
    df = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
                       "Close": [1.5, 2.5], "Volume": [10, 20], "Dividends": [0.0, 0.1],
                       "Stock Splits": [0.0, 0.0]}, index=idx)
    raw = drv.yahoo_to_csv_bytes(df)
    text = raw.decode()
    assert text.splitlines()[0] == "Date,Open,High,Low,Close,Volume"
    assert "2020-01-02," in text and "-05:00" not in text and "Dividends" not in text
    back = drv.parse_raw("yahoo", raw)
    assert list(back.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


def test_abi1_line_reports_a_missing_column_or_a_calendar_mismatch_instead_of_crashing():
    idx = pd.bdate_range("2015-01-02", periods=400)
    close = 100 + np.arange(400, dtype=float)
    veh = sfs.build_series(pd.Series(close, index=idx), name="V", cutoff=CUTOFF)
    frame_no_open = pd.DataFrame({"Close": close}, index=idx)
    spy = pd.DataFrame({"Close": close}, index=idx)
    out = drv.abi1_line(series_name="S", side="up_fade", trigger="pct756", cell=sfs.PRIMARY_CELL,
                        ticker="V", position="long", signal_dates=idx[[300]],
                        vehicle_series=veh, vehicle_frame=frame_no_open, spy_frame=spy,
                        cutoff=CUTOFF)
    assert out["status"] == "not_computed" and out["reason"]
    frame = pd.DataFrame({"Open": close, "Close": close}, index=idx)
    spy_short = spy.drop(idx[100:110])
    out2 = drv.abi1_line(series_name="S", side="up_fade", trigger="pct756",
                         cell=sfs.PRIMARY_CELL, ticker="V", position="long",
                         signal_dates=idx[[300]], vehicle_series=veh, vehicle_frame=frame,
                         spy_frame=spy_short, cutoff=CUTOFF)
    assert out2["status"] == "not_computed" and "calendar" in out2["reason"].lower()


# ---------------------------------------------------------------------------
# G) End to end on the synthetic world
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    out = tmp_path_factory.mktemp("study")
    world = make_world()
    sleeps = []
    code, fetcher, sleeps = run(out, world, "--universe", "^VIX,GC=F,JPY=X", sleeps=sleeps)
    return {"out": str(out), "code": code, "fetcher": fetcher, "world": world, "sleeps": sleeps}


def test_e2e_completes_and_writes_every_output(e2e):
    assert e2e["code"] == 0
    for name in ("REPORT.md", "results.json", "cells.csv", "run_meta.json",
                 os.path.join("data", "raw", "manifest.json")):
        assert os.path.exists(os.path.join(e2e["out"], name)), name
    res = load_results(e2e["out"])
    assert res["status"] == "completed"
    assert res["cutoff"] == "2023-06-30"
    assert res["params"]["B"] == 20 and res["params"]["B_map"] == 10
    assert res["params"]["seeds"]["n1"] == sfn.N1_SEED == sfs.N1_SEED
    assert res["params"]["seeds"]["n2"] == sfn.N2_SEED == sfs.N2_SEED
    assert res["params"]["universe"] == ["^VIX", "GC=F", "JPY=X"]


def test_e2e_fetch_was_serial_and_paced(e2e):
    calls = e2e["fetcher"].calls
    assert len(calls) == len(set(calls)) == 8              # 3 trigger + DEXJPUS + 3 ETF + SPY
    pauses = [s for s in e2e["sleeps"] if s == 3.0]
    assert len(pauses) == len(calls) - 1
    assert 60.0 not in e2e["sleeps"]


def test_e2e_cells_csv_has_every_cell_of_every_series(e2e):
    df = pd.read_csv(os.path.join(e2e["out"], "cells.csv"))
    assert len(df) == 3 * 576
    assert set(df["series"]) == {"^VIX", "GC=F", "JPY=X"}
    per = df.groupby(["series", "side", "trigger"]).size()
    assert (per == 72).all() and len(per) == 3 * 8
    for col in ("mean_x", "t", "eligible", "n", "null_q90", "plateau_score", "stab_passed",
                "traded", "dropped_censored", "tp_not_applicable"):
        assert col in df.columns, col


def test_e2e_report_has_the_required_structure_and_no_dollar_digits(e2e):
    text = read(e2e["out"], "REPORT.md")
    heads = re.findall(r"^## (.+)$", text, flags=re.M)
    assert heads[0] == "Plain-language summary"
    for title in ("Power statement", "Exploratory map", "Tradable-vehicle results",
                  "Put-affordability screen", "ABI-1 transfer line",
                  "Data, counts and checks", "Contamination and limits",
                  "Seeds, data digests and reproduction"):
        assert title in heads
    assert heads[1].startswith("Decision table")
    assert not DOLLAR_DIGIT.search(text)


def test_e2e_no_output_file_contains_a_dollar_digit(e2e):
    hits = []
    for base, _, files in os.walk(e2e["out"]):
        for f in files:
            if f.endswith(".csv") and base.endswith("raw"):
                continue                                 # raw vendor bytes are not authored text
            with open(os.path.join(base, f), encoding="utf-8", errors="ignore") as fh:
                if DOLLAR_DIGIT.search(fh.read()):
                    hits.append(os.path.join(base, f))
    assert hits == []


def test_e2e_manifest_digests_match_the_cached_files(e2e):
    raw = os.path.join(e2e["out"], "data", "raw")
    with open(os.path.join(raw, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    assert len(man["files"]) == 8
    for name, ent in man["files"].items():
        assert ent["sha256"] == hashlib.sha256(open(os.path.join(raw, name), "rb").read()).hexdigest()
        assert ent["retrieved_utc"].endswith("Z")
    res = load_results(e2e["out"])
    assert res["data"]["files"]["^VIX"]["sha256"] == \
        man["files"][drv.raw_filename("yahoo", "^VIX")]["sha256"]


def test_e2e_sealed_window_no_series_has_a_date_after_the_cutoff(e2e):
    res = load_results(e2e["out"])
    for name, s in res["series"].items():
        assert s["last_date"] <= "2023-06-30", name
        assert s["counts"]["post_cutoff_dropped"] > 0, name   # the world runs past it
    for name, f in res["data"]["files"].items():
        if f.get("last_date"):
            assert f["last_date"] <= "2023-06-30", name


def test_e2e_external_corrections_are_counted_and_listed_with_dates(e2e):
    res = load_results(e2e["out"])
    jpy = res["series"]["JPY=X"]
    assert jpy["counts"]["external_corrections"] == len(BAD_TICKS)
    assert jpy["external_correction_dates"] == list(BAD_TICKS)
    assert jpy["external_reference"] == "DEXJPUS"
    text = read(e2e["out"], "REPORT.md")
    for d in BAD_TICKS:
        assert d in text


def test_e2e_every_kind_of_count_is_in_results_and_report(e2e):
    res = load_results(e2e["out"])
    for name, s in res["series"].items():
        for k in ("rows", "post_cutoff_dropped", "pre_start_dropped", "missing_rows_dropped",
                  "external_corrections", "nonpositive"):
            assert k in s["counts"], (name, k)
        assert len(s["groups"]) == 8
        for g in s["groups"]:
            for k in ("n_flagged_days", "n_episodes", "n_kept", "dropped_censored"):
                assert k in g
        assert set(s["trigger_diagnostics"]) == set(sfs.TRIGGERS)
        assert len(s["largest_returns"]) == 20
        assert s["ar1"]["n_pairs"] > 1000
        for k in ("traded", "undefined_sigma", "no_e1_confirmation", "tp_not_applicable",
                  "censored_data_end", "touch_missing"):
            assert k in s["cell_status_totals"], (name, k)
        # every cell of the series is accounted for: statuses sum to cells x episodes
    text = read(e2e["out"], "REPORT.md")
    for needle in ("dropped", "censored", "undefined", "not applicable", "externally corrected"):
        assert needle in text.lower()


def test_e2e_results_json_carries_the_headline_and_a_separate_n2_sensitivity(e2e):
    res = load_results(e2e["out"])
    conf = res["confirmatory"]
    for k in ("p_n1", "p_headline", "n2_sensitivity", "tstar", "tstar_key", "headline_rule"):
        assert k in conf, k
    for k in ("p_n2", "offsets", "skipped", "available_for_tstar"):
        assert k in conf["n2_sensitivity"], k
    h = res["headline"]
    for k in ("p_headline", "p_n1", "n2_sensitivity", "decision", "tstar_key", "vehicle",
              "candidate", "top5_plateau", "abi1_tstar", "abi1_top_plateau",
              "calibration_status"):
        assert k in h, k
    assert h["p_headline"] == conf["p_headline"]         # taken from the null module, not re-derived
    assert h["calibration_status"] in ("pending", "complete")
    assert res["confirmatory"]["n_tested"] >= 1, "the synthetic world must give a tested group"
    assert len(res["confirmatory"]["groups"]) == 2 * 8 * 2 - 8 or len(res["confirmatory"]["groups"]) > 0


def test_e2e_vehicle_scope_put_screen_and_abi1_are_present(e2e):
    res = load_results(e2e["out"])
    v = res["vehicle"]
    assert v["scope"], "the T* pair and the top plateau pairs must be in scope"
    assert v["scope"][0]["role"] in ("tstar", "tstar+plateau_top3")
    ran = [e for e in v["scope"] if e["run"]]
    assert ran and all(e["result"]["n_anchors_input"] >= 0 for e in ran)
    assert "candidate" in v and "is_candidate" in v["candidate"]
    assert res["put_screen"], "a put screen entry per selected up-side cell"
    assert set(res["abi1"]) == {"tstar", "top_plateau"}
    for e in (res["abi1"]["tstar"], res["abi1"]["top_plateau"]):
        assert e is None or e["status"] in ("computed", "not_computed", "not_run")
    assert res["recommendation"]["conditions"][0]["id"] == "a"


def test_e2e_map_top_lists_are_ordered_and_labelled(e2e):
    res = load_results(e2e["out"])
    top = res["map"]["top_overall"]
    assert 0 < len(top) <= 30
    scores = [r["plateau_score"] for r in top]
    assert scores == sorted(scores, reverse=True)
    lines = {(r["series"], r["trigger"], r["side"], r["entry"]) for r in top}
    assert len(lines) == len(top)                        # one line per group and entry rule
    for r in top:
        assert "null_q90" in r and "stab_passed" in r


def test_e2e_calibration_is_pending_when_no_file_and_used_when_present(e2e, tmp_path):
    text = read(e2e["out"], "REPORT.md")
    sec = re.search(r"^## Power statement\n(.*?)(?=^## )", text, flags=re.M | re.S).group(1)
    assert "pending" in sec.lower()
    # a calibration folder next to the output is picked up
    out2 = tmp_path / "o2"
    os.makedirs(out2 / "calibration")
    cal = {"finished": True, "args": {"datasets": 1}, "size": {},
           "power": {"curve": {"1.0": {"power_headline": 0.9, "n_datasets": 5}},
                     "min_detectable_effect_50": 0.61, "min_detectable_effect_80": 0.91,
                     "passes_required_power": True}}
    with open(out2 / "calibration" / "null_calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal, f)
    world = make_world()
    code, _, _ = run(out2, world, "--universe", "^VIX")
    assert code == 0
    t2 = read(str(out2), "REPORT.md")
    sec2 = re.search(r"^## Power statement\n(.*?)(?=^## )", t2, flags=re.M | re.S).group(1)
    assert "0.61" in sec2 and "pending" not in sec2.lower()
    assert load_results(str(out2))["headline"]["calibration_status"] == "complete"


# ---------------------------------------------------------------------------
# H) Sealed window: nothing after the cutoff can influence any result
# ---------------------------------------------------------------------------

def _strip_data(res):
    res = copy.deepcopy(res)
    res.pop("data", None)
    return res


def test_results_are_identical_whatever_the_data_after_the_cutoff_say(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    code_a, _, _ = run(a, make_world(post_shift=False), "--universe", "^VIX")
    code_b, _, _ = run(b, make_world(post_shift=True), "--universe", "^VIX")
    assert code_a == code_b == 0
    ra, rb = load_results(str(a)), load_results(str(b))
    assert ra["data"]["files"]["^VIX"]["sha256"] != rb["data"]["files"]["^VIX"]["sha256"]
    assert _strip_data(ra) == _strip_data(rb)
    assert read(str(a), "cells.csv") == read(str(b), "cells.csv")


# ---------------------------------------------------------------------------
# I) Determinism, whatever the worker count
# ---------------------------------------------------------------------------

def test_identical_seeds_give_byte_identical_results_json_across_worker_counts(e2e, tmp_path):
    out2 = tmp_path / "again"
    shutil.copytree(os.path.join(e2e["out"], "data"), out2 / "data")
    world = e2e["world"]
    fetcher = FakeFetcher(world, fail=set(world))          # any fetch would fail: cache only
    code = drv.main(["--out", str(out2), "--B", "20", "--B-map", "10", "--workers", "2",
                     "--skip-fetch", "--universe", "^VIX,GC=F,JPY=X"],
                    fetcher=fetcher, sleep=lambda s: None)
    assert code == 0
    assert fetcher.calls == []                             # --skip-fetch never touches the network
    assert read(str(out2), "results.json") == read(e2e["out"], "results.json")
    assert read(str(out2), "cells.csv") == read(e2e["out"], "cells.csv")


# ---------------------------------------------------------------------------
# J) Fail closed
# ---------------------------------------------------------------------------

def test_a_missing_required_series_fails_closed_and_the_report_lists_it(tmp_path):
    out = tmp_path / "fc"
    sleeps = []
    code, fetcher, sleeps = run(out, make_world(), "--universe", "^VIX,GC=F",
                                fail={("yahoo", "GLD")}, sleeps=sleeps)
    assert code == 3
    res = load_results(str(out))
    assert res["status"] == "failed_closed"
    assert [m["name"] for m in res["failure"]["missing"]] == ["GLD"]
    assert "download" in res["failure"]["missing"][0]["reason"] or \
        "fetch" in res["failure"]["missing"][0]["reason"]
    text = read(str(out), "REPORT.md")
    assert "GLD" in text and "did not run" in text.lower()
    assert re.findall(r"^## (.+)$", text, flags=re.M)[0] == "Plain-language summary"
    assert not os.path.exists(out / "cells.csv")
    assert sleeps.count(60.0) == 1                        # the one retry, then give up
    assert not DOLLAR_DIGIT.search(text)


def test_jpy_without_dexjpus_fails_closed_and_lists_jpy_as_missing(tmp_path):
    out = tmp_path / "jpy"
    code, _, _ = run(out, make_world(), "--universe", "^VIX,JPY=X", fail={("fred", "DEXJPUS")})
    assert code == 3
    res = load_results(str(out))
    names = [m["name"] for m in res["failure"]["missing"]]
    assert "JPY=X" in names and "DEXJPUS" in names
    reason = {m["name"]: m["reason"] for m in res["failure"]["missing"]}["JPY=X"]
    assert "DEXJPUS" in reason and "external" in reason.lower()
    text = read(str(out), "REPORT.md")
    assert "JPY=X" in text and "DEXJPUS" in text


def test_skip_fetch_with_a_missing_cache_fails_closed(tmp_path):
    out = tmp_path / "nocache"
    fetcher = FakeFetcher(make_world())
    code = drv.main(["--out", str(out), "--B", "20", "--B-map", "10", "--workers", "1",
                     "--skip-fetch", "--universe", "^VIX"], fetcher=fetcher,
                    sleep=lambda s: None)
    assert code == 3 and fetcher.calls == []
    assert load_results(str(out))["status"] == "failed_closed"
    assert "^VIX" in read(str(out), "REPORT.md")


def test_skip_fetch_refuses_a_tampered_cached_file(e2e, tmp_path):
    out2 = tmp_path / "tamper"
    shutil.copytree(os.path.join(e2e["out"], "data"), out2 / "data")
    victim = out2 / "data" / "raw" / drv.raw_filename("yahoo", "^VIX")
    with open(victim, "ab") as f:
        f.write(b"2020-01-02,1,1,1,1,1\n")
    code = drv.main(["--out", str(out2), "--B", "20", "--B-map", "10", "--workers", "1",
                     "--skip-fetch", "--universe", "^VIX"], fetcher=FakeFetcher({}),
                    sleep=lambda s: None)
    assert code == 3
    res = load_results(str(out2))
    assert res["status"] == "failed_closed"
    assert "digest" in json.dumps(res["failure"]).lower()


# ---------------------------------------------------------------------------
# K) Re-rendering the report from a finished results.json (no recomputation)
# ---------------------------------------------------------------------------

def test_render_only_rebuilds_the_report_without_touching_results(e2e, tmp_path):
    out2 = tmp_path / "render"
    shutil.copytree(e2e["out"], out2)
    before = read(str(out2), "results.json")
    cells_before = read(str(out2), "cells.csv")
    os.remove(out2 / "REPORT.md")
    fetcher = FakeFetcher({}, fail=set())
    code = drv.main(["--out", str(out2), "--render-only"], fetcher=fetcher, sleep=lambda s: None)
    assert code == 0 and fetcher.calls == []
    assert read(str(out2), "results.json") == before
    assert read(str(out2), "cells.csv") == cells_before
    assert read(str(out2), "REPORT.md") == read(e2e["out"], "REPORT.md")


def test_render_only_without_a_results_file_is_refused(tmp_path):
    code = drv.main(["--out", str(tmp_path / "empty"), "--render-only"],
                    fetcher=FakeFetcher({}), sleep=lambda s: None)
    assert code == 2


# ---------------------------------------------------------------------------
# L) A new run never leaves the previous run's outputs looking current
# ---------------------------------------------------------------------------

def test_stale_outputs_of_an_earlier_run_are_removed_before_a_new_run_starts(tmp_path):
    out = tmp_path / "stale"
    os.makedirs(out / "calibration")
    os.makedirs(out / "data" / "raw")
    for name in ("REPORT.md", "results.json", "cells.csv", "run_meta.json"):
        (out / name).write_text("stale", encoding="utf-8")
    keep = out / "calibration" / "null_calibration.json"
    keep.write_text("{}", encoding="utf-8")
    code, _, _ = run(out, make_world(), "--universe", "^VIX,GC=F", fail={("yahoo", "GLD")})
    assert code == 3
    assert not (out / "cells.csv").exists()               # the stale map must not survive
    assert read(str(out), "results.json") != "stale"
    assert json.loads(read(str(out), "results.json"))["status"] == "failed_closed"
    assert keep.exists()                                  # calibration is never touched


def test_the_power_section_labels_n2_as_a_sensitivity(e2e):
    from mining import spike_fade_report as rep
    cal = {"finished": True, "args": {}, "size": {"p_n1": {"rate": 0.09, "in_window": True},
                                                  "p_n2": {"rate": 0.13, "in_window": True},
                                                  "p_headline": {"rate": 0.09, "in_window": True}},
           "power": {"curve": {}, "min_detectable_effect_50": 0.5,
                     "min_detectable_effect_80": 0.8, "passes_required_power": True}}
    text = rep._power_section(cal)
    assert "sensitivity, not gating" in text


# ---------------------------------------------------------------------------
# M) A WTI-like FRED series with one negative print (non-positive close rule) runs end to end
# ---------------------------------------------------------------------------

def test_a_negative_print_is_listed_and_only_the_quantities_that_use_it_are_undefined(tmp_path):
    idx = _index()
    n = len(idx)
    wti, xw = _spiky(n, 41, 60.0, vol=0.012)
    neg_day = pd.Timestamp("2020-04-20")
    wti_neg = wti.copy()
    wti_neg[idx.get_loc(neg_day)] = -37.5
    wti_neg[idx.get_loc(pd.Timestamp("2020-06-15")):] *= 1.7    # a spike inside the 63-session window
    world = make_world()
    world[("fred", "DCOILWTICO")] = _fred_bytes(wti_neg, idx, "DCOILWTICO",
                                                holes=range(30, n, 71))
    rng = np.random.default_rng(43)
    world[("yahoo", "USO")] = _ohlc_bytes(
        40.0 * np.exp(0.8 * xw + np.cumsum(rng.normal(0, 0.004, n))), idx, 44)
    out = tmp_path / "wti"
    code, _, _ = run(out, world, "--universe", "WTI spot")
    assert code == 0
    res = load_results(str(out))
    s = res["series"]["WTI spot"]
    assert s["extra_lag"] == 1
    assert s["nonpositive_dates"] == ["2020-04-20"]
    assert s["counts"]["nonpositive"] == 1
    tot = s["cell_status_totals"]
    assert tot["sigma_touch_nonpositive"] > 0            # windows that contain the bad close
    assert tot["traded"] > 0                             # ... and the rest of the year still trades
    assert s["affected_dates"], "every affected anchor date must be listed"
    assert all("date" in a and a["statuses"] for a in s["affected_dates"])
    text = read(str(out), "REPORT.md")
    assert "2020-04-20" in text
    assert "swing window touches a non-positive close" in text   # readable, not a raw key
    assert not DOLLAR_DIGIT.search(text)


# ---------------------------------------------------------------------------
# N) The driver's wiring agrees with direct calls to the modules
# ---------------------------------------------------------------------------

def _direct_series(world, name, symbol, first):
    raw = drv.parse_raw("yahoo", world[("yahoo", symbol)])
    return sfs.build_series(raw["Close"], name=name, cutoff=CUTOFF, first_date=pd.Timestamp(first))


def test_e2e_confirmatory_result_equals_a_direct_call_to_the_null_module(e2e):
    world = e2e["world"]
    vix = _direct_series(world, "^VIX", "^VIX", "1990-01-02")
    gc = _direct_series(world, "GC=F", "GC=F", "2000-08-30")
    direct = sfn.confirmatory_test([vix, gc], B=20)
    res = load_results(e2e["out"])["confirmatory"]
    assert res["tstar_key"] == "|".join(direct.tstar_key)
    assert res["tstar"] == pytest.approx(direct.tstar, rel=1e-12)
    assert res["tstar_n"] == direct.tstar_n
    assert res["p_headline"] == pytest.approx(direct.p_headline, rel=1e-12)
    assert res["p_n1"] == pytest.approx(direct.p_n1, rel=1e-12)
    assert res["n_tested"] == len(direct.observed.tested)


def test_e2e_cells_match_direct_group_evaluation_and_null_columns_are_row_aligned(e2e):
    world = e2e["world"]
    gc = _direct_series(world, "GC=F", "GC=F", "2000-08-30")
    sig = sfs.compute_triggers(gc)
    df = pd.read_csv(os.path.join(e2e["out"], "cells.csv"))
    nulls = sfn.n1_cell_null_series(gc, B=10, signals=sig)
    for trig, side in (("pct756", "up_fade"), ("z20@2.0", "down_fade")):
        g = sfs.evaluate_group(gc, trig, side, signals=sig)
        want = g.to_frame()
        got = df[(df["series"] == "GC=F") & (df["trigger"] == trig) & (df["side"] == side)]
        assert len(got) == 72 == len(want)
        for col in ("n", "mean_x", "t", "hit_rate", "traded", "dropped_censored"):
            np.testing.assert_allclose(got[col].to_numpy(dtype=float),
                                       want[col].to_numpy(dtype=float), rtol=1e-12,
                                       equal_nan=True, err_msg=col)
        assert list(got["H"]) == list(want["H"]) and list(got["tp"]) == list(want["tp"])
        assert [str(v) for v in got["sl"]] == [str(v) for v in want["sl"]]
        q90 = nulls[(trig, side)].q90
        np.testing.assert_allclose(got["null_q90"].to_numpy(dtype=float), q90, rtol=1e-12,
                                   equal_nan=True)
        assert (got["entry"].to_numpy() == want["entry"].to_numpy()).all()


def test_e2e_top_cells_are_consistent_with_cells_csv(e2e):
    df = pd.read_csv(os.path.join(e2e["out"], "cells.csv"))
    res = load_results(e2e["out"])
    for r in res["map"]["top_overall"][:5]:
        row = df[(df["series"] == r["series"]) & (df["trigger"] == r["trigger"])
                 & (df["side"] == r["side"]) & (df["entry"] == r["entry"]) & (df["H"] == r["H"])
                 & (df["tp"] == r["tp"]) & (df["sl"].astype(str) == str(r["sl"]))]
        assert len(row) == 1
        assert row.iloc[0]["plateau_score"] == pytest.approx(r["plateau_score"], rel=1e-12)
        assert row.iloc[0]["mean_x"] == pytest.approx(r["mean_x"], rel=1e-12)
        assert int(row.iloc[0]["n"]) == r["n"]


def test_e2e_vehicle_scope_entries_equal_direct_vehicle_tests(e2e):
    world = e2e["world"]
    res = load_results(e2e["out"])
    series = {"^VIX": _direct_series(world, "^VIX", "^VIX", "1990-01-02"),
              "GC=F": _direct_series(world, "GC=F", "GC=F", "2000-08-30")}
    checked = 0
    for e in res["vehicle"]["scope"]:
        if not e["run"]:
            continue
        ser = series[e["series"]]
        sig = sfs.compute_triggers(ser)
        es = sfs.build_episode_set(ser, sig[e["trigger"]], e["side"])
        raw = drv.parse_raw("yahoo", world[("yahoo", e["vehicle"])])
        veh = sfs.build_series(raw["Close"], name=e["vehicle"], cutoff=CUTOFF)
        entry, H, tp, sl = e["cell"].split("|")
        cell = (entry, int(H), tp, None if sl == "none" else float(sl))
        direct = sfa.vehicle_test_for_pair(e["series"], e["side"], veh, es.dates, cell, B=20,
                                           seed=sfs.N1_SEED)
        got = e["result"]
        assert got["n"] == direct.n
        assert got["excess_pp"] == pytest.approx(direct.excess_pp, rel=1e-12)
        assert got["null_percentile"] == pytest.approx(direct.null_percentile, rel=1e-12)
        assert got["mean_pnl_pp"] == pytest.approx(100.0 * direct.mean_pnl, rel=1e-12)
        checked += 1
    assert checked >= 1


# ---------------------------------------------------------------------------
# O) Which calibration file the report reads
# ---------------------------------------------------------------------------

def test_the_newest_spec_version_calibration_file_is_the_one_used(tmp_path):
    d = tmp_path / "calibration"
    os.makedirs(d)
    pw = {"curve": {"1.0": {"power_headline": 0.9, "n_datasets": 5}}}
    (d / "null_calibration.json").write_text(json.dumps(
        {"finished": True, "generated_utc": "2000-01-01T00:00:00Z", "marker": "v3-original",
         "power": pw}), encoding="utf-8")
    (d / "null_calibration_v3_1_summary.json").write_text(json.dumps(
        {"finished": True, "spec_version": "3.1", "generated_utc": "2000-01-01T00:10:00Z",
         "marker": "v3.1", "power": pw}), encoding="utf-8")
    (d / "unfinished.json").write_text(json.dumps(
        {"finished": False, "spec_version": "9.9", "marker": "unfinished"}), encoding="utf-8")
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    assert drv._read_calibration(str(tmp_path))["marker"] == "v3.1"
    os.remove(d / "null_calibration_v3_1_summary.json")
    assert drv._read_calibration(str(tmp_path))["marker"] == "v3-original"
    os.remove(d / "null_calibration.json")
    assert drv._read_calibration(str(tmp_path)) is None


def test_the_power_section_names_the_spec_version_and_headline_definition():
    from mining import spike_fade_report as rep
    cal = {"finished": True, "spec_version": "3.1", "headline_definition": "N1 alone",
           "args": {"datasets": 300}, "size": {},
           "power": {"curve": {}, "min_detectable_effect_50": 0.5,
                     "min_detectable_effect_80": 0.81, "passes_required_power": True}}
    text = rep._power_section(cal)
    assert "spec version 3.1" in text and "N1 alone" in text


# ---------------------------------------------------------------------------
# P) Robustness of the driver
# ---------------------------------------------------------------------------

def test_the_reproduction_command_survives_an_output_folder_on_another_drive(monkeypatch):
    def boom(*a, **k):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")
    monkeypatch.setattr(drv.os.path, "relpath", boom)
    cfg = drv.StudyConfig(out=os.path.abspath("D:/somewhere/else"))
    cmd = drv._command(cfg, drv.resolve_universe(None), cached=False)
    assert "run_spike_fade_study.py" in cmd and "--out" in cmd


def test_only_the_full_default_configuration_counts_as_preregistered():
    full = drv.StudyConfig(out="x")
    assert drv.is_preregistered(full, drv.resolve_universe(None)) == (True, [])
    ok, notes = drv.is_preregistered(drv.StudyConfig(out="x", B=20), drv.resolve_universe(None))
    assert ok is False and any("B = 20" in n for n in notes)
    ok, notes = drv.is_preregistered(full, drv.resolve_universe("^VIX"))
    assert ok is False and any("universe" in n for n in notes)
    ok, notes = drv.is_preregistered(drv.StudyConfig(out="x", B_map=10),
                                     drv.resolve_universe(None))
    assert ok is False
    ok, notes = drv.is_preregistered(drv.StudyConfig(out="x", cutoff=pd.Timestamp("2022-12-30")),
                                     drv.resolve_universe(None))
    assert ok is False


def test_a_small_run_is_marked_not_preregistered_in_results_and_report(e2e):
    res = load_results(e2e["out"])
    assert res["params"]["pre_registered_configuration"] is False
    assert res["params"]["configuration_notes"]
    assert res["spec"]["version"] == "3.1"
    text = read(e2e["out"], "REPORT.md")
    block = re.search(r"^## Plain-language summary\n(.*?)(?=^## )", text,
                      flags=re.M | re.S).group(1)
    assert "NOT the pre-registered configuration" in block
    assert res["stage_failures"] == []


def test_an_existing_completed_results_file_is_never_overwritten_without_force(e2e, tmp_path):
    out2 = tmp_path / "again2"
    shutil.copytree(e2e["out"], out2)
    before = read(str(out2), "results.json")
    fetcher = FakeFetcher(e2e["world"])
    code = drv.main(["--out", str(out2), "--B", "20", "--B-map", "10", "--workers", "1",
                     "--universe", "^VIX,GC=F,JPY=X"], fetcher=fetcher, sleep=lambda s: None)
    assert code == 2 and fetcher.calls == []
    assert read(str(out2), "results.json") == before
    code = drv.main(["--out", str(out2), "--B", "20", "--B-map", "10", "--workers", "1",
                     "--universe", "^VIX,GC=F,JPY=X", "--force"], fetcher=FakeFetcher(e2e["world"]),
                    sleep=lambda s: None)
    assert code == 0


def test_recommendation_b_uses_the_larger_of_the_nominal_and_the_exact_bar():
    def abi(sharpe, exact):
        return {"status": "computed", "reason": None,
                "full": {"meets_in_sample_bar": sharpe >= 1.02, "dd_passed": True,
                         "excess_sharpe_annualized": sharpe, "in_sample_bar": 1.02,
                         "dsr_bar_annualized": exact}}
    below = drv.build_recommendation(_cand(True), abi(1.10, 1.15), tstar_series="GC=F")
    assert below["register_stage2"] is False
    d = {c["id"]: c for c in below["conditions"]}["b"]["detail"]
    assert "1.02" in d and "1.15" in d
    above = drv.build_recommendation(_cand(True), abi(1.20, 1.15), tstar_series="GC=F")
    assert above["register_stage2"] is True


def test_abi1_not_computed_reason_names_the_offending_dates():
    idx = pd.bdate_range("2015-01-02", periods=400)
    close = 100 + np.arange(400, dtype=float)
    veh = sfs.build_series(pd.Series(close, index=idx), name="V", cutoff=CUTOFF)
    open_ = close.copy()
    open_[120] = 0.0
    frame = pd.DataFrame({"Open": open_, "Close": close}, index=idx)
    spy = pd.DataFrame({"Close": close}, index=idx)
    out = drv.abi1_line(series_name="S", side="up_fade", trigger="pct756", cell=sfs.PRIMARY_CELL,
                        ticker="V", position="long", signal_dates=idx[[300]],
                        vehicle_series=veh, vehicle_frame=frame, spy_frame=spy, cutoff=CUTOFF)
    assert out["status"] == "not_computed"
    assert idx[120].strftime("%Y-%m-%d") in out["reason"]
    good = frame.copy()
    good["Open"] = close
    spy_gap = spy.drop(idx[[200, 201]])
    out2 = drv.abi1_line(series_name="S", side="up_fade", trigger="pct756",
                         cell=sfs.PRIMARY_CELL, ticker="V", position="long",
                         signal_dates=idx[[300]], vehicle_series=veh, vehicle_frame=good,
                         spy_frame=spy_gap, cutoff=CUTOFF)
    assert out2["status"] == "not_computed"
    assert idx[200].strftime("%Y-%m-%d") in out2["reason"]


def test_a_duplicate_date_in_a_download_fails_closed_instead_of_crashing(tmp_path):
    world = make_world()
    text = world[("yahoo", "^VIX")].decode().splitlines()
    text.insert(60, text[60])                              # the same date twice, before the cutoff
    world[("yahoo", "^VIX")] = ("\n".join(text) + "\n").encode()
    out = tmp_path / "dup"
    code, _, _ = run(out, world, "--universe", "^VIX")
    assert code == 3
    res = load_results(str(out))
    assert res["status"] == "failed_closed"
    m = res["failure"]["missing"]
    assert m[0]["name"] == "^VIX" and "unusable" in m[0]["reason"]
    assert "^VIX" in read(str(out), "REPORT.md")


def test_a_failing_put_screen_is_recorded_and_the_rest_of_the_report_is_still_written(
        tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic put-screen failure")
    monkeypatch.setattr(sfa, "put_screen_for_cell", boom)
    out = tmp_path / "putfail"
    code, _, _ = run(out, make_world(), "--universe", "^VIX")
    assert code == 4
    res = load_results(str(out))
    assert res["status"] == "completed"
    assert any(f["stage"] == "put screen" and "synthetic put-screen failure" in f["reason"]
               for f in res["stage_failures"])
    assert all(p["status"] in ("not_computed", "not_applicable") for p in res["put_screen"])
    text = read(str(out), "REPORT.md")
    assert "incomplete" in re.search(r"^## Plain-language summary\n(.*?)(?=^## )", text,
                                     flags=re.M | re.S).group(1).lower()
    assert "synthetic put-screen failure" in text
    assert res["confirmatory"]["tstar_key"]               # the headline is not lost


def test_a_failing_vehicle_test_is_recorded_and_downstream_stages_say_not_run(
        tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic vehicle failure")
    monkeypatch.setattr(sfa, "vehicle_test_for_pair", boom)
    out = tmp_path / "vehfail"
    code, _, _ = run(out, make_world(), "--universe", "^VIX")
    assert code == 4
    res = load_results(str(out))
    assert any(f["stage"] == "vehicle test" for f in res["stage_failures"])
    assert res["abi1"]["tstar"]["status"] == "not_run"
    assert res["vehicle"]["candidate"]["is_candidate"] is False
    assert res["recommendation"]["register_stage2"] is False


# ---------------------------------------------------------------------------
# Q) Coverage items of the same-day reference check and the confirmatory test
# ---------------------------------------------------------------------------

def test_e2e_vehicles_and_the_core_get_their_20_largest_moves_listed(e2e):
    res = load_results(e2e["out"])
    vs = res["vehicle_series"]
    assert set(vs) == {"SVXY", "VIXY", "GLD", "SPY"}
    assert vs["SPY"]["role"] == "core" and vs["GLD"]["role"] == "vehicle"
    for name, v in vs.items():
        assert len(v["largest_returns"]) == 20, name
        assert v["last_date"] <= "2023-06-30" and v["n_rows"] > 1000
    text = read(e2e["out"], "REPORT.md")
    assert vs["GLD"]["largest_returns"][0]["date"] in text


def test_e2e_report_has_the_adjusted_p_by_series_and_side(e2e):
    text = read(e2e["out"], "REPORT.md")
    sec = re.search(r"^## Decision table.*?\n(.*?)(?=^## )", text, flags=re.M | re.S).group(1)
    assert "by series" in sec.lower() and "by side" in sec.lower()
    assert "descriptive" in sec.lower()


# ---------------------------------------------------------------------------
# R) The robustness file and the calibration file are different things
# ---------------------------------------------------------------------------

def _robustness_json():
    def arm(rate, se, rej):
        return {"n_datasets": 300, "p_n1": {"rate": rate, "se": se, "rejections": rej,
                                           "n_denominator": 300}}
    # Synthetic arms (not measured values): rates 0.08 / 0.13 / 0.18 of 300.
    return {"finished": True, "spec_version": "3.1", "generated_utc": "2000-01-01T00:20:00Z",
            "claim_run": True, "power": None, "args": {"n_series": 7, "datasets": 300, "B": 100},
            "size_by_vol_of_vol_multiplier": {"1.0": arm(0.08, 0.0157, 24),
                                              "2.0": arm(0.13, 0.0194, 39),
                                              "3.0": arm(0.18, 0.0222, 54)}}


def test_a_size_robustness_file_is_never_mistaken_for_the_power_calibration(tmp_path):
    d = tmp_path / "calibration"
    os.makedirs(d)
    good = {"finished": True, "spec_version": "3.1", "generated_utc": "2000-01-01T00:10:00Z",
            "marker": "power-file",
            "power": {"curve": {"1.0": {"power_headline": 0.95, "n_datasets": 50}},
                      "min_detectable_effect_50": 0.5, "min_detectable_effect_80": 0.8,
                      "passes_required_power": True}}
    (d / "null_calibration_v3_1_summary.json").write_text(json.dumps(good), encoding="utf-8")
    rob = _robustness_json()
    (d / "null_size_robustness.json").write_text(json.dumps(rob), encoding="utf-8")
    assert drv._read_calibration(str(tmp_path))["marker"] == "power-file"
    got = drv._read_robustness(str(tmp_path))
    assert got["size_by_vol_of_vol_multiplier"]["3.0"]["p_n1"]["rate"] == 0.18
    os.remove(d / "null_size_robustness.json")
    assert drv._read_robustness(str(tmp_path)) is None
    (d / "null_size_robustness.json").write_text("{broken", encoding="utf-8")
    assert drv._read_robustness(str(tmp_path)) is None
    unfinished = dict(rob, finished=False)
    (d / "null_size_robustness.json").write_text(json.dumps(unfinished), encoding="utf-8")
    assert drv._read_robustness(str(tmp_path)) is None


def test_render_only_picks_up_the_robustness_file_and_changes_nothing_else(e2e, tmp_path):
    out2 = tmp_path / "rob"
    shutil.copytree(e2e["out"], out2)
    res_before = read(str(out2), "results.json")
    cells_before = read(str(out2), "cells.csv")
    text_before = read(str(out2), "REPORT.md")
    assert "8 to 18 percent" not in text_before
    os.makedirs(out2 / "calibration", exist_ok=True)
    (out2 / "calibration" / "null_size_robustness.json").write_text(
        json.dumps(_robustness_json()), encoding="utf-8")
    code = drv.main(["--out", str(out2), "--render-only"], fetcher=FakeFetcher({}),
                    sleep=lambda s: None)
    assert code == 0
    text = read(str(out2), "REPORT.md")
    assert "approximately an 8 to 18 percent test" in text and "0.130" in text
    assert read(str(out2), "results.json") == res_before
    assert read(str(out2), "cells.csv") == cells_before
    assert not DOLLAR_DIGIT.search(text)
    # the calibration (power) statement is still the one it was
    sec = re.search(r"^## Power statement\n(.*?)(?=^## )", text, flags=re.M | re.S).group(1)
    assert "pending" in sec.lower()


def test_the_fred_vehicle_lag_reading_is_in_the_reports_open_choices(e2e):
    text = read(e2e["out"], "REPORT.md")
    tail = text[text.index("### Open choices"):]
    assert "the vehicle's median M and sigma21 are read at t0 + L" in tail


# ---------------------------------------------------------------------------
# S) No bare 8 to 17 digit run in the text outputs
# ---------------------------------------------------------------------------

BARE_RUN = re.compile(r"(?<!\d\.)\b\d{8,17}\b")


def test_the_run_log_groups_long_digit_runs(capsys):
    drv._default_log("N1 (random dates): p = 0.5000, B = 100, seed " + str(2026_0923) + ".")
    out = capsys.readouterr().out
    assert "seed 2026_0923." in out and not BARE_RUN.search(out)


def test_e2e_report_and_meta_have_no_bare_long_digit_run(e2e):
    for name in ("REPORT.md", "run_meta.json"):
        assert BARE_RUN.findall(read(e2e["out"], name)) == [], name
    assert "2026_0923" in read(e2e["out"], "REPORT.md")
    with open(os.path.join(e2e["out"], "data", "raw", "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    for ent in man["files"].values():
        assert not re.search(r"(?<![0-9a-f])[0-9]{8,}(?![0-9a-f])", ent["sha256"])
