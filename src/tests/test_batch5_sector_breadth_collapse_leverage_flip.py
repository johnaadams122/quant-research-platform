"""Tests for src/mining/batch5/sector_breadth_collapse_leverage_flip.py.

Synthetic data only (no network); breadth is a synthetic panel.AsOfSeries.
The one real file read is data/mining/catalog.json, for the parameter,
ticker-claim and eligibility checks.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import panel  # noqa: E402
from mining.batch5 import sector_breadth_collapse_leverage_flip as mod  # noqa: E402
from mining.rotation_engine import run_two_asset_rotation  # noqa: E402

FAMILY_ID = "sector_breadth_collapse_leverage_flip"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")


# ------------------------------------------------------------------ helpers

def _dates(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _frame(closes, dates=None) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    if dates is None:
        dates = _dates(len(closes))
    return pd.DataFrame({
        "Open": closes * 0.999,
        "High": closes * 1.01,
        "Low": closes * 0.99,
        "Close": closes,
    }, index=pd.DatetimeIndex(dates))


def _breadth(dates, values) -> panel.AsOfSeries:
    return panel.AsOfSeries(dates=pd.DatetimeIndex(dates),
                            values=np.asarray(values, dtype=float),
                            indicator="participation_above_ma",
                            content_sha256="synthetic")


def _flat(n: int, level: float = 100.0) -> np.ndarray:
    return np.full(n, level)


def _run_targets(target_fn, engine_frames, upto=None):
    calendar = next(iter(engine_frames.values())).index
    n = len(calendar) if upto is None else upto
    results = {}
    for t in range(n):
        truncated = {key: frame.iloc[:t + 1].copy(deep=True)
                     for key, frame in engine_frames.items()}
        results[calendar[t]] = target_fn(truncated)
    return results


def _catalog_entry():
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    matches = [e for e in entries if e["family_id"] == FAMILY_ID]
    assert len(matches) == 1, "family_id must appear exactly once in the catalog"
    return matches[0]


# --------------------------------------------------------- decision_target

def test_decision_target_seeds_on_first_defined_session():
    assert mod.decision_target(None, 0.50) == "SPXL"
    assert mod.decision_target(None, 0.7777) == "SPXL"
    assert mod.decision_target(None, 0.4999) == "SPXS"
    assert mod.decision_target(None, 0.0) == "SPXS"


def test_decision_target_none_value_never_changes_or_seeds_standing():
    assert mod.decision_target(None, None) is None
    assert mod.decision_target("SPXL", None) == "SPXL"
    assert mod.decision_target("SPXS", None) == "SPXS"


def test_decision_target_flips_only_at_the_exact_thresholds():
    # SPXL holds strictly above the bearish threshold.
    assert mod.decision_target("SPXL", 0.31) == "SPXL"
    assert mod.decision_target("SPXL", 0.30) == "SPXS"      # <= flips
    assert mod.decision_target("SPXL", 0.10) == "SPXS"
    # SPXS holds strictly below the bullish threshold.
    assert mod.decision_target("SPXS", 0.69) == "SPXS"
    assert mod.decision_target("SPXS", 0.70) == "SPXL"      # >= flips
    assert mod.decision_target("SPXS", 0.95) == "SPXL"


def test_decision_target_hysteresis_band_holds_both_sides():
    # Inside (0.30, 0.70) neither side flips -- the whole point of the band.
    for value in (0.31, 0.40, 0.50, 0.60, 0.69):
        assert mod.decision_target("SPXL", value) == "SPXL"
        assert mod.decision_target("SPXS", value) == "SPXS"


# --------------------------------------------------------- decision_series

def test_decision_series_is_causal_and_matches_hand_walked_path():
    dates = _dates(6)
    # 0.80 (seed SPXL) -> 0.50 (hold) -> 0.30 (flip SPXS) -> 0.50 (hold) ->
    # 0.70 (flip SPXL) -> 0.29 (flip SPXS)
    values = [0.80, 0.50, 0.30, 0.50, 0.70, 0.29]
    breadth = _breadth(dates, values)
    series = mod.decision_series(dates, breadth)
    assert list(series) == ["SPXL", "SPXL", "SPXS", "SPXS", "SPXL", "SPXS"]


def test_decision_series_stays_none_before_breadth_is_defined():
    dates = _dates(5)
    breadth = panel.AsOfSeries(
        dates=pd.DatetimeIndex([dates[2]]), values=np.asarray([0.9]),
        indicator="participation_above_ma", content_sha256="synthetic")
    series = mod.decision_series(dates, breadth)
    assert list(series) == [None, None, "SPXL", "SPXL", "SPXL"]


# -------------------------------------------------------- factory contract

def test_make_target_fn_requires_both_traded_frames():
    frame = _frame(_flat(50))
    with pytest.raises(ValueError, match="SPXS"):
        mod.make_target_fn({"SPXL": frame})
    with pytest.raises(TypeError):
        mod.make_target_fn(["SPXL", "SPXS"])


def test_make_target_fn_refuses_fewer_than_two_shared_sessions():
    spxl = _frame(_flat(50))
    spxs = _frame(_flat(50), dates=_dates(50, start="2030-01-01"))  # disjoint
    with pytest.raises(ValueError, match="fewer than two"):
        mod.make_target_fn({"SPXL": spxl, "SPXS": spxs},
                           breadth=_breadth(_dates(1), [0.5]))


def test_factory_and_engine_must_run_on_the_same_data():
    n = 60
    spxl = _frame(_flat(n))
    spxs = _frame(_flat(n))
    breadth = _breadth(spxl.index, np.linspace(0.8, 0.2, n))
    target_fn = mod.make_target_fn({"SPXL": spxl, "SPXS": spxs}, breadth=breadth)
    tampered = spxl.copy()
    tampered["Close"] = tampered["Close"] * 1.5
    with pytest.raises(ValueError, match="does not match the factory"):
        target_fn({"SPXL": tampered.iloc[:30]})


# ------------------------------------------------------------- look-ahead

def test_look_ahead_probe_perturbing_future_bars_never_changes_the_past():
    n = 80
    spxl = _frame(np.linspace(100.0, 140.0, n))
    spxs = _frame(np.linspace(50.0, 30.0, n))
    breadth = _breadth(spxl.index, np.concatenate([
        np.full(20, 0.8), np.full(20, 0.5), np.full(20, 0.25), np.full(20, 0.75)]))
    frames = {"SPXL": spxl, "SPXS": spxs}
    k = 45
    baseline = _run_targets(mod.make_target_fn(frames, breadth=breadth), frames,
                            upto=k + 1)

    perturbed = {}
    for key, frame in frames.items():
        alt = frame.copy()
        later = alt.index > frame.index[k]
        alt.loc[later, ["Open", "Close"]] *= 1.4
        alt.loc[later, "High"] *= 1.5
        alt.loc[later, "Low"] *= 1.3
        perturbed[key] = alt
    rebuilt = _run_targets(mod.make_target_fn(perturbed, breadth=breadth),
                           perturbed, upto=k + 1)
    assert rebuilt == baseline

    truncated = {key: frame.iloc[:k + 1].copy() for key, frame in frames.items()}
    values = np.concatenate([
        np.full(20, 0.8), np.full(20, 0.5), np.full(20, 0.25), np.full(20, 0.75)])[:k + 1]
    truncated_breadth = _breadth(spxl.index[:k + 1], values)
    cut = _run_targets(mod.make_target_fn(truncated, breadth=truncated_breadth),
                       truncated)
    assert cut == baseline


# ------------------------------------------------------------- end to end

def test_end_to_end_through_the_rotation_engine_switches_on_designed_breadth():
    n = 120
    spxl = _frame(np.linspace(100.0, 100.0 * 1.02 ** n, n) *
                  (1.0 + 0.001 * np.sin(np.arange(n))))
    spxs = _frame(np.linspace(50.0, 50.0 * 0.98 ** n, n) *
                  (1.0 + 0.001 * np.cos(np.arange(n))))
    # Designed breadth path: starts high (seed SPXL), collapses at session 30
    # (flip SPXS), recovers at session 70 (flip SPXL), never touches the band
    # edges elsewhere.
    values = np.full(n, 0.80)
    values[30:70] = 0.20
    values[70:] = 0.85
    breadth = _breadth(spxl.index, values)
    frames = {"SPXL": spxl, "SPXS": spxs}
    target_fn = mod.make_target_fn(frames, breadth=breadth)
    trades, active = run_two_asset_rotation(
        frames, target_fn, cost_bps_by_asset={"SPXL": 2.5, "SPXS": 3.9},
        tax_haircut_rate=0.0)
    assert list(trades["asset"]) == ["SPXL", "SPXS", "SPXL"]
    assert trades["reason"].tolist() == ["switch", "switch", "end_of_data"]
    # Breadth already seeds SPXL at session 0 itself, so the FIRST fill is
    # the next-open execution of that session-0 decision: session 1.
    # Switch at session 30 fills at session 31's open; switch at 70 -> 71.
    assert trades["entry_date"].iloc[0] == spxl.index[1]
    assert trades["exit_date"].iloc[0] == spxl.index[31]
    assert trades["entry_date"].iloc[1] == spxl.index[31]
    assert trades["exit_date"].iloc[1] == spxl.index[71]
    assert trades["entry_date"].iloc[2] == spxl.index[71]
    assert active.dropna().isin(["SPXL", "SPXS"]).all()
    assert active.iloc[0] is None  # decision fires session 0, fills session 1


def test_never_in_cash_active_asset_is_always_one_of_the_two_legs_once_seeded():
    n = 60
    spxl = _frame(_flat(n, 100.0))
    spxs = _frame(_flat(n, 40.0))
    breadth = _breadth(spxl.index, np.linspace(0.9, 0.1, n))
    frames = {"SPXL": spxl, "SPXS": spxs}
    target_fn = mod.make_target_fn(frames, breadth=breadth)
    _, active = run_two_asset_rotation(
        frames, target_fn, cost_bps_by_asset={"SPXL": 1.0, "SPXS": 1.0})
    # Exactly one bar (the seed session itself) is flat before the first fill;
    # every later bar holds a real leg.
    assert active.iloc[1:].isin(["SPXL", "SPXS"]).all()


# ------------------------------------------------------------------ catalog

def test_parameters_are_byte_identical_to_the_catalog_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert mod.TRADED == entry["tickers"] == ["SPXL", "SPXS"]
    assert mod.SHAPE == entry["shape"] == "rotation_two_asset"
    assert mod.BENCHMARK == entry["benchmark_ticker"] == "SPY"
    assert entry["status"] == "eligible"


def test_no_unexpected_eligible_catalog_family_owns_spxl_or_spxs():
    """The distinctness / owned-exposure check this candidate's catalog
    entry claims -- run for real against the live catalog, not just
    asserted in prose.

    quarterly_index_rebalance_flow_leverage_window is a genuinely distinct
    mechanism (a quarterly calendar-rebalancing-flow signal, vs. this
    family's sector-breadth-participation signal) that also legitimately
    trades SPXL/SPXS -- see that family's own catalog distinctness_note.
    The check names that one known, deliberately-distinct sibling
    explicitly rather than asserting no other claimant exists at all, so a THIRD,
    unreviewed family later claiming the same tickers still fails loudly."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    owners = {e["family_id"] for e in entries
             if e["status"] == "eligible" and e["family_id"] != FAMILY_ID
             and set(e.get("tickers", [])) & {"SPXL", "SPXS"}}
    assert owners == {"quarterly_index_rebalance_flow_leverage_window"}


def test_family_is_not_excluded_by_the_real_step1_ruled_out_check(tmp_path):
    """The pipeline's catalog-load duplicate and exclusion check
    (mining.ruled_out.filter_eligible), run against the real catalog and an
    explicitly EMPTY trial ledger (the ledger is not part of this repository;
    an empty file states that precondition instead of relying on a missing
    path) -- this is the pipeline's own duplicate-check mechanism, not a
    hand-rolled substitute for it."""
    from mining import catalog as catalog_mod
    from mining import ruled_out

    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    open(ledger_path, "w", encoding="utf-8").close()
    loaded = catalog_mod.load_catalog(CATALOG_PATH)
    eligible, excluded = ruled_out.filter_eligible(loaded, ledger_path)
    assert FAMILY_ID in {e["family_id"] for e in eligible}
    assert FAMILY_ID not in {e["family_id"] for e in excluded}


# ----------------------------------------------- driver contract validation

def test_module_passes_the_batch5_driver_contract():
    spec = importlib.util.spec_from_file_location(
        "run_batch5_mining_for_test",
        os.path.join(_SCRIPTS, "run_batch5_mining.py"))
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    validated = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert validated["family_id"] == FAMILY_ID
    assert validated["shape"] == "rotation_two_asset"
    assert validated["traded"] == ["SPXL", "SPXS"]
    assert validated["signal"] == []
    assert validated["snapshots"] == {}
    assert driver.parameter_drift(mod.PARAMETERS, _catalog_entry()["parameters"]) is None
