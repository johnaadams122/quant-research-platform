"""Tests for src/mining/batch5/vxn_vix_spread_derisking_regime.py.

This module expresses the same mechanism that scripts/run_batch3_mining.py
wires for this family_id, in the batch5 module contract. The ratio-regime
math itself (``mining.signals.make_ratio_regime_target_fn``) already has its
own direct unit tests (``src/tests/test_mining_batch3_signals.py``) -- this
file proves only:

  1. The batch5 driver contract is satisfied (``validate_module_contract``
     via ``run_batch5_mining.discover_modules``).
  2. PARAMETERS is byte-identical to data/mining/catalog.json's real entry
     for this family_id.
  3. The thin wrapper actually delegates to the shared factory with the
     same arguments -- proven on a tiny synthetic frame set
     (bar-for-bar agreement with signals.make_ratio_regime_target_fn
     called directly).

Synthetic data only: nothing in this file reaches the network.
"""
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

from mining import signals  # noqa: E402
from mining.batch5 import vxn_vix_spread_derisking_regime as mod  # noqa: E402

FAMILY_ID = "vxn_vix_spread_derisking_regime"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
RISK, DEFENSIVE = "QQQ", "SHY"
NUMERATOR, DENOMINATOR = "^VXN", "^VIX"
ALL_KEYS = (RISK, DEFENSIVE, NUMERATOR, DENOMINATOR)


def _catalog_entry() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [entry for entry in catalog if entry.get("family_id") == FAMILY_ID]
    assert len(matches) == 1, f"expected exactly one catalog entry for {FAMILY_ID}"
    return matches[0]


# ------------------------------------------------------------- (a) contract

def test_parameters_are_byte_identical_to_the_catalog_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert mod.FAMILY_ID == FAMILY_ID == entry["family_id"]
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]
    assert mod.SIGNAL == [NUMERATOR, DENOMINATOR]
    # catalog.json's own "tickers" list names only the risk leg + the two
    # signal series (QQQ, ^VXN, ^VIX); the defensive leg SHY is named
    # separately, via "parameters"."defensive_proxy" -- same convention
    # scripts/run_batch3_mining.py's SUPPORTED_CONTRACTS uses for this
    # family_id (catalog_tickers=("QQQ","^VXN","^VIX")).
    assert set(entry["tickers"]) == {RISK, NUMERATOR, DENOMINATOR}
    assert set(entry["tickers"]) <= set(mod.TRADED + mod.SIGNAL)
    assert entry["parameters"]["defensive_proxy"] == DEFENSIVE == mod.TRADED[1]
    assert mod.SNAPSHOTS == {}
    # ABI-1 always benchmarks SPY, never the catalog's own tickers.
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert not isinstance(mod.EXPECTED_MIN_TRADES, bool)
    assert mod.EXPECTED_MIN_TRADES >= 0


def test_module_passes_the_driver_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module (so the test does not depend
    on every other module in the real batch5 dir)."""
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
    assert spec["signal"] == [NUMERATOR, DENOMINATOR]
    assert spec["benchmark"] == "SPY"
    assert spec["parameters"] == mod.PARAMETERS


def test_factory_requires_every_traded_and_signal_frame():
    index = pd.bdate_range("2020-01-01", periods=5)
    ok_frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
                            index=index)
    for missing_key in ALL_KEYS:
        frames = {key: ok_frame for key in ALL_KEYS if key != missing_key}
        try:
            mod.make_target_fn(frames)
        except ValueError as exc:
            assert missing_key in str(exc)
        else:
            raise AssertionError(
                f"make_target_fn accepted a frames dict missing {missing_key}")


# --------------------------------------------- (b) delegation, synthetic

def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _frame_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=index)


def _random_frame(index: pd.DatetimeIndex, seed: int, level: float = 100.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0002, 0.012, len(index))
    return _frame_from_close(level * np.exp(np.cumsum(steps)), index)


def _synthetic_frames(index: pd.DatetimeIndex) -> dict:
    return {RISK: _random_frame(index, 1), DEFENSIVE: _random_frame(index, 2),
            NUMERATOR: _random_frame(index, 3, level=30.0),
            DENOMINATOR: _random_frame(index, 4, level=18.0)}


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def test_make_target_fn_matches_the_reviewed_factory_bar_for_bar():
    """The wrapper must not reimplement the math: its returned callable
    must agree, on every bar, with signals.make_ratio_regime_target_fn
    called directly with the same arguments (fast=20, slow=252,
    defensive_ticker=SHY -- the batch3 driver's wiring for this family)."""
    index = _bdays("2018-01-02", "2020-12-31")
    frames = _synthetic_frames(index)
    ours = mod.make_target_fn(frames)
    reference = signals.make_ratio_regime_target_fn(
        risk_ticker=RISK, defensive_ticker=DEFENSIVE,
        numerator_ticker=NUMERATOR, denominator_ticker=DENOMINATOR,
        fast=20, slow=252)
    decisions = []
    for date in index:
        truncated = _truncate(frames, date)
        decisions.append(ours(truncated))
        assert decisions[-1] == reference(truncated), date
    assert RISK in decisions and DEFENSIVE in decisions and None in decisions


def test_warm_up_returns_none_before_the_slow_window_is_full():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _synthetic_frames(index)
    target = mod.make_target_fn(frames)
    early = index[:mod._SLOW - 1]
    for date in early:
        assert target(_truncate(frames, date)) is None, date
    assert target(_truncate(frames, index[mod._SLOW])) is not None
