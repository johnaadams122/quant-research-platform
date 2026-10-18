"""Tests for src/mining/batch5/treasury_duration_volatility_timing.py.

Synthetic frames only (no network) for the contract/delegation/look-ahead
checks, mirroring test_batch5_gold_long_treasury_relative_momentum_rotation.
py's structure. This family's PARAMETERS are compared against the REAL
pre-registered catalog entry (data/mining/catalog.json), matching
test_batch5_stock_split_signaling_drift.py's own convention.

The real-data end-to-end checks (live market-history fetches) are not part of
this public file; every test here is offline and uses synthetic data.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import rotation_engine, signals  # noqa: E402
from mining.batch5 import treasury_duration_volatility_timing as mod  # noqa: E402

FAMILY_ID = "treasury_duration_volatility_timing"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
RISK, DEFENSIVE = "TLT", "SHY"


# ------------------------------------------------------------ synthetic data

def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _frame_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _random_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0001, 0.006, len(index))
    return _frame_from_close(100.0 * np.exp(np.cumsum(steps)), index)


def _random_frames(index: pd.DatetimeIndex, seed: int = 7) -> dict:
    return {RISK: _random_frame(index, seed),
            DEFENSIVE: _random_frame(index, seed + 100)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _reference():
    return signals.make_monthly_volatility_target_fn(
        RISK, DEFENSIVE, vol_lookback_days=mod.PARAMETERS["vol_lookback_days"],
        median_lookback_days=252)


# ------------------------------------------------------------------ contract

def test_module_constants_match_the_family_contract():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 0
    assert callable(mod.make_target_fn)


def test_parameters_are_byte_identical_to_the_frozen_catalog_entry():
    # catalog.json is the pre-registered, frozen source this module must
    # not reinterpret (same convention as stock_split_signaling_drift).
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == mod.FAMILY_ID]
    assert len(matches) == 1
    entry = matches[0]
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert mod.TRADED == entry["tickers"] == [RISK, DEFENSIVE]


def test_module_passes_the_driver_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module, so the result does not
    depend on whatever else sits in the real batch5 dir."""
    import run_batch5_mining as batch5
    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == [RISK, DEFENSIVE]
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    catalog_entry = next(c for c in catalog if c["family_id"] == mod.FAMILY_ID)
    assert batch5.parameter_drift(
        spec["parameters"], catalog_entry["parameters"]) is None


def test_factory_requires_both_legs():
    index = _bdays("2020-01-01", "2020-06-30")
    frames = _random_frames(index)
    try:
        mod.make_target_fn({RISK: frames[RISK]})
    except ValueError as exc:
        assert DEFENSIVE in str(exc)
    else:
        raise AssertionError("make_target_fn accepted a missing SHY frame")
    try:
        mod.make_target_fn({DEFENSIVE: frames[DEFENSIVE]})
    except ValueError as exc:
        assert RISK in str(exc)
    else:
        raise AssertionError("make_target_fn accepted a missing TLT frame")


# ------------------------------------------------------- delegation to signals.py

def test_matches_the_repo_reference_factory_on_every_bar():
    """The module is a thin wrapper: it must reproduce
    signals.make_monthly_volatility_target_fn's own decisions bar-for-bar,
    proving the math is delegated unchanged and not reimplemented."""
    index = _bdays("2015-01-02", "2020-12-31")
    frames = _random_frames(index, seed=11)
    ours = mod.make_target_fn(frames)
    reference = _reference()
    decisions = []
    for date in index[::5]:  # every 5th session is plenty to prove equality
        truncated = _truncate(frames, date)
        decisions.append(ours(truncated))
        assert decisions[-1] == reference(truncated), date
    assert RISK in decisions or DEFENSIVE in decisions


def test_returns_none_during_warmup_before_252_days_of_realized_vol_exist():
    index = _bdays("2019-01-02", "2019-12-31")
    frames = _random_frames(index, seed=3)
    target = mod.make_target_fn(frames)
    for date in index:
        assert target(_truncate(frames, date)) is None, date


# ------------------------------------------------------------ look-ahead

def test_perturbing_bars_after_k_never_changes_targets_through_k():
    index = _bdays("2015-01-02", "2020-12-31")
    frames = _random_frames(index, seed=21)
    k = 700
    target = mod.make_target_fn(frames)
    before = [target(_truncate(frames, index[j])) for j in range(0, k + 1, 3)]
    perturbed = {}
    for key, frame in frames.items():
        copy = frame.copy(deep=True)
        after = copy.index > index[k]
        copy.loc[after, "Close"] *= 1.4
        copy.loc[after, "Open"] *= 1.3
        copy.loc[after, "High"] = copy.loc[after, "High"] * 1.6
        copy.loc[after, "Low"] = copy.loc[after, "Low"] * 0.6
        perturbed[key] = copy
    target_p = mod.make_target_fn(perturbed)
    after = [target_p(_truncate(perturbed, index[j])) for j in range(0, k + 1, 3)]
    assert before == after
    assert any(value is not None for value in before)


def test_factory_ignores_prices_in_the_full_history_frames():
    """Holdout-blindness by construction: make_target_fn's ``frames``
    argument only checks required keys are present; decisions come from the
    truncated frames the engine hands over per call."""
    index = _bdays("2015-01-02", "2020-12-31")
    frames = _random_frames(index, seed=2)
    scrambled = {key: frame * 3.0 for key, frame in frames.items()}
    a, b = mod.make_target_fn(frames), mod.make_target_fn(scrambled)
    for date in index[index >= pd.Timestamp("2017-01-01")][::11]:
        truncated = _truncate(frames, date)
        assert a(truncated) == b(truncated)


# --------------------------------------------------------------- end to end

def test_end_to_end_rotation_never_goes_flat_after_warmup():
    index = _bdays("2015-01-01", "2018-12-31")
    frames = _random_frames(index, seed=17)
    target = mod.make_target_fn(frames)
    segments, active = rotation_engine.run_two_asset_rotation(
        {RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]}, target,
        cost_bps_by_asset={RISK: 3.0, DEFENSIVE: 1.0})
    assert len(segments) >= 1
    assert set(segments["asset"]).issubset({RISK, DEFENSIVE})
    assert set(segments["reason"]).issubset({"switch", "end_of_data"})
    first = segments["entry_date"].min()
    assert active.loc[active.index >= first].notna().all()
