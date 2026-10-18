"""Tests for src/mining/batch5/credit_spread_hy_ig_risk_regime.py.

This family expresses its catalog mechanism in the batch5 module
contract -- see the module's own docstring, including the
factory-argument polarity: the literal argument order would be backwards
relative to the catalog mechanism, so this module and its tests pass the
risk/defensive roles swapped (the POLARITY NOTE there). Like
test_batch5_usd_strength_risk_regime.py and
test_batch5_vxn_vix_spread_derisking_regime.py, the PARAMETERS drift
check here compares against the data/mining/catalog.json entry, read
directly, rather than a hand-typed duplicate.

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
_SRC = os.path.join(_ROOT, "src")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import signals  # noqa: E402
from mining.batch5 import credit_spread_hy_ig_risk_regime as mod  # noqa: E402

FAMILY_ID = "credit_spread_hy_ig_risk_regime"
RISK, DEFENSIVE = "SPY", "SHY"
NUMERATOR, DENOMINATOR = "HYG", "LQD"
ALL_KEYS = (RISK, DEFENSIVE, NUMERATOR, DENOMINATOR)
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _random_frame(index: pd.DatetimeIndex, seed: int, level: float = 100.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0002, 0.01, len(index))
    return _ohlc_from_close(level * np.exp(np.cumsum(steps)), index)


def _synthetic_frames(index: pd.DatetimeIndex) -> dict:
    return {
        RISK: _random_frame(index, seed=1, level=100.0),
        DEFENSIVE: _random_frame(index, seed=2, level=80.0),
        NUMERATOR: _random_frame(index, seed=3, level=85.0),
        DENOMINATOR: _random_frame(index, seed=4, level=115.0),
    }


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _catalog_entry() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == FAMILY_ID]
    assert len(matches) == 1, (
        f"expected exactly one catalog.json entry for {FAMILY_ID}, found "
        f"{len(matches)}")
    return matches[0]


# -------------------------------------------------------------- (a) contract

def test_module_constants_match_the_family_contract():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]
    assert mod.SIGNAL == [NUMERATOR, DENOMINATOR]
    assert mod.BENCHMARK == "SPY"
    assert mod.SNAPSHOTS == {}
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert not isinstance(mod.EXPECTED_MIN_TRADES, bool)
    assert mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_target_fn)


def test_parameters_are_byte_identical_to_the_real_catalog_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert set(entry["tickers"]) == {RISK, NUMERATOR, DENOMINATOR}
    assert entry["parameters"]["defensive_proxy"] == DEFENSIVE == mod.TRADED[1]
    assert entry["parameters"]["signal_ratio"] == "HYG/LQD"


def test_driver_discovery_validates_this_module_in_isolation(tmp_path):
    """Discovery on a private copy of the module, so the test does not depend
    on whatever else is in the real batch5 dir."""
    import run_batch5_mining as batch5
    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["family_id"] == FAMILY_ID
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == [RISK, DEFENSIVE]
    assert spec["signal"] == [NUMERATOR, DENOMINATOR]
    assert spec["benchmark"] == "SPY"
    assert spec["expected_min_trades"] == mod.EXPECTED_MIN_TRADES
    assert spec["parameters"] == mod.PARAMETERS


# ------------------------------------------------------------------ factory

def test_make_target_fn_requires_every_traded_and_signal_frame():
    index = _bdays("2015-01-02", "2016-12-30")
    frames = _synthetic_frames(index)
    for missing_key in ALL_KEYS:
        subset = {key: frame for key, frame in frames.items() if key != missing_key}
        with pytest.raises(ValueError, match=missing_key):
            mod.make_target_fn(subset)


def test_make_target_fn_matches_the_reviewed_factory_bar_for_bar():
    """The wrapper must not reimplement the math: its returned callable
    must agree, on every bar, with signals.make_ratio_regime_target_fn
    called directly with the SAME (swapped) arguments -- proving the
    wiring in the module was not transposed or altered beyond the
    documented risk/defensive swap."""
    index = _bdays("2018-01-02", "2020-12-31")
    frames = _synthetic_frames(index)
    ours = mod.make_target_fn(frames)
    reference = signals.make_ratio_regime_target_fn(
        risk_ticker=DEFENSIVE, defensive_ticker=RISK,
        numerator_ticker=NUMERATOR, denominator_ticker=DENOMINATOR,
        fast=20, slow=100)
    decisions = []
    for date in index[::3]:
        truncated = _truncate(frames, date)
        decisions.append(ours(truncated))
        assert decisions[-1] == reference(truncated), date
    assert set(decisions) <= {RISK, DEFENSIVE, None}
    assert RISK in decisions or DEFENSIVE in decisions


def test_warm_up_returns_none_before_the_slow_window_is_full():
    index = _bdays("2020-01-01", "2020-12-31")
    frames = _synthetic_frames(index)
    target = mod.make_target_fn(frames)
    early = index[:mod._SLOW - 1]
    for date in early:
        assert target(_truncate(frames, date)) is None, date
    assert target(_truncate(frames, index[mod._SLOW])) is not None


def test_rising_hyg_lqd_ratio_selects_spy_matching_catalog_text():
    """Hand-constructed HYG/LQD path: both flat for 100+ sessions (the
    100-day ratio SMA warms up and settles), then HYG rallies sharply
    against a flat LQD so the ratio's 20-day MA sits clearly above its
    100-day MA ('20-day average above the 100-day average -- spreads
    stable or compressing' per the catalog text) -> hold SPY. And the
    inverse: HYG sells off against flat LQD so the ratio's fast MA sits
    below its slow MA ('spreads widening') -> exit to SHY. This directly
    checks the catalog mechanism's stated polarity end to end through
    make_target_fn, not just internal self-consistency against the
    reference factory (the previous test already covers that) -- the
    check that pins the polarity."""
    index = _bdays("2018-01-02", "2019-06-30")
    n = len(index)
    lqd_close = np.full(n, 115.0)

    # Rising-ratio path: HYG rallies over the final 20 sessions.
    hyg_rising = np.full(n, 85.0)
    hyg_rising[-20:] = 85.0 + np.linspace(0, 12, 20)
    frames_up = {
        RISK: _ohlc_from_close(np.full(n, 400.0), index),
        DEFENSIVE: _ohlc_from_close(np.full(n, 82.0), index),
        NUMERATOR: _ohlc_from_close(hyg_rising, index),
        DENOMINATOR: _ohlc_from_close(lqd_close, index),
    }
    target_up = mod.make_target_fn(frames_up)
    ratio_up = frames_up[NUMERATOR]["Close"] / frames_up[DENOMINATOR]["Close"]
    assert ratio_up.iloc[-20:].mean() > ratio_up.iloc[-100:].mean(), (
        "test construction check: fast MA must sit above slow MA here")
    assert target_up(_truncate(frames_up, index[-1])) == RISK, (
        "HYG/LQD ratio's 20d MA above its 100d MA must select SPY "
        "(risk-on), per the catalog's 'spreads stable or compressing "
        "-> hold SPY' mechanism text")

    # Falling-ratio path: HYG sells off over the final 20 sessions.
    hyg_falling = np.full(n, 85.0)
    hyg_falling[-20:] = 85.0 - np.linspace(0, 12, 20)
    frames_down = {
        RISK: _ohlc_from_close(np.full(n, 400.0), index),
        DEFENSIVE: _ohlc_from_close(np.full(n, 82.0), index),
        NUMERATOR: _ohlc_from_close(hyg_falling, index),
        DENOMINATOR: _ohlc_from_close(lqd_close, index),
    }
    target_down = mod.make_target_fn(frames_down)
    ratio_down = frames_down[NUMERATOR]["Close"] / frames_down[DENOMINATOR]["Close"]
    assert ratio_down.iloc[-20:].mean() < ratio_down.iloc[-100:].mean(), (
        "test construction check: fast MA must sit below slow MA here")
    assert target_down(_truncate(frames_down, index[-1])) == DEFENSIVE, (
        "HYG/LQD ratio's 20d MA below its 100d MA must select SHY "
        "(risk-off), per the catalog's 'spreads widening -> exit to "
        "SHY' mechanism text")
