"""Tests for src/mining/batch5/tips_nominal_breakeven_rotation.py.

This module is a THIN WRAPPER around the shared, separately tested
``mining.signals.make_monthly_relative_momentum_target_fn`` factory (see
``src/tests/test_mining_batch3_signals.py`` for TIP/IEF coverage of the
underlying momentum math itself, and ``scripts/run_batch3_mining.py``'s
``SUPPORTED_CONTRACTS``/``EXPECTED_PARAMETER_VALUES``, which define the
same family_id for the batch3 driver).
So this file proves DELEGATION, contract compliance,
and PARAMETERS byte-identity against the real catalog -- not the momentum
math a second time.

The real-data end-to-end checks (live market-history fetches) are not part of
this public file; every test here is offline and uses synthetic data.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import abi1_driver, signals  # noqa: E402
from mining.batch5 import tips_nominal_breakeven_rotation as mod  # noqa: E402

FAMILY_ID = "tips_nominal_breakeven_rotation"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
RISK, DEFENSIVE = "TIP", "IEF"


# ------------------------------------------------------------ synthetic data

def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _frame_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _random_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0001, 0.004, len(index))
    return _frame_from_close(100.0 * np.exp(np.cumsum(steps)), index)


def _random_frames(index: pd.DatetimeIndex, seed: int = 4) -> dict:
    return {RISK: _random_frame(index, seed),
            DEFENSIVE: _random_frame(index, seed + 100)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


# ------------------------------------------------------------- contract

def _catalog_entry() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [entry for entry in catalog if entry.get("family_id") == FAMILY_ID]
    assert len(matches) == 1, (
        f"expected exactly one catalog.json entry for {FAMILY_ID!r}, "
        f"found {len(matches)}")
    return matches[0]


def test_parameters_and_tickers_are_byte_identical_to_the_real_catalog_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert mod.FAMILY_ID == FAMILY_ID == entry["family_id"]
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == entry["tickers"] == [RISK, DEFENSIVE]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"  # ABI-1 always benchmarks SPY, never the
                                   # catalog's own tickers -- see module docstring


def test_module_satisfies_validate_module_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module -- other writers may touch
    the real batch5 dir concurrently, so the real dir's
    current contents are never asserted on."""
    import run_batch5_mining as driver

    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = driver.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == [RISK, DEFENSIVE]
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    assert spec["expected_min_trades"] == mod.EXPECTED_MIN_TRADES
    assert driver.parameter_drift(
        spec["parameters"], _catalog_entry()["parameters"]) is None


# --------------------------------------------------------- delegation/contract

def test_factory_requires_both_legs_and_names_the_missing_one():
    index = _bdays("2020-01-01", "2020-06-30")
    frames = _random_frames(index)
    with pytest.raises(ValueError, match=DEFENSIVE):
        mod.make_target_fn({RISK: frames[RISK]})
    with pytest.raises(ValueError, match=RISK):
        mod.make_target_fn({DEFENSIVE: frames[DEFENSIVE]})
    with pytest.raises(ValueError, match=RISK):
        mod.make_target_fn({})


def test_make_target_fn_delegates_to_the_reviewed_signals_factory_verbatim(
        monkeypatch):
    """This module must add NO new momentum math -- it must call
    ``signals.make_monthly_relative_momentum_target_fn`` with exactly
    (risk=TIP, defensive=IEF, lookback_months=3) and return its result
    unchanged."""
    captured = {}
    sentinel = object()
    real = signals.make_monthly_relative_momentum_target_fn

    def spy(risk_ticker, defensive_ticker, *, lookback_months):
        captured["args"] = (risk_ticker, defensive_ticker)
        captured["lookback_months"] = lookback_months
        return sentinel

    monkeypatch.setattr(mod.signals, "make_monthly_relative_momentum_target_fn", spy)
    index = _bdays("2020-01-01", "2020-06-30")
    frames = _random_frames(index)
    result = mod.make_target_fn(frames)

    assert captured["args"] == (RISK, DEFENSIVE)
    assert captured["lookback_months"] == mod.PARAMETERS["lookback_months"] == 3
    assert result is sentinel  # no wrapping layer -- the factory's own
                               # per-call target is returned unchanged
    monkeypatch.setattr(mod.signals, "make_monthly_relative_momentum_target_fn", real)


def test_make_target_fn_matches_the_reference_factory_bar_for_bar_on_synthetic_data():
    """Belt-and-suspenders on top of the delegation-spy test above: the
    actual (non-mocked) decisions must match the reference factory called
    directly, on a real synthetic tape."""
    index = _bdays("2018-01-02", "2021-06-30")
    frames = _random_frames(index, seed=13)
    ours = mod.make_target_fn(frames)
    reference = signals.make_monthly_relative_momentum_target_fn(
        RISK, DEFENSIVE, lookback_months=3)
    decisions = []
    for date in index:
        truncated = _truncate(frames, date)
        decisions.append(ours(truncated))
        assert decisions[-1] == reference(truncated), date
    assert RISK in decisions and DEFENSIVE in decisions and None in decisions


# ------------------------------------------------------------- real data (e2e)

