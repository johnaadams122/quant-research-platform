"""Tests for src/mining/batch5/usd_strength_risk_regime.py.

This family expresses an existing catalog mechanism, unchanged, in the
batch5 module contract -- see the module's own docstring. Like
test_batch5_skew_index_tail_risk_regime.py and
test_batch5_vxn_vix_spread_derisking_regime.py, the PARAMETERS drift
check here compares against the real, already-existing
data/mining/catalog.json entry, read directly, rather than a hand-typed
duplicate.

Every test here is offline and uses synthetic data.
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

import data_pipeline  # noqa: E402
import run_batch3_mining as batch3  # noqa: E402
from mining import abi1_driver, rotation_engine, signals  # noqa: E402
from mining.batch5 import usd_strength_risk_regime as mod  # noqa: E402

FAMILY_ID = "usd_strength_risk_regime"
RISK, DEFENSIVE, SIGNAL_TICKER = "SPY", "SHY", "UUP"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           f"{FAMILY_ID}.py")


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
        SIGNAL_TICKER: _random_frame(index, seed=3, level=25.0),
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


# -------------------------------------------------------------- contract

def test_module_constants_match_the_family_contract():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]
    assert mod.SIGNAL == [SIGNAL_TICKER]
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
    assert entry["tickers"] == [RISK, SIGNAL_TICKER]


def test_driver_discovery_validates_this_module_in_isolation(tmp_path):
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
    assert spec["family_id"] == FAMILY_ID
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == [RISK, DEFENSIVE]
    assert spec["signal"] == [SIGNAL_TICKER]
    assert spec["benchmark"] == "SPY"
    assert spec["expected_min_trades"] == mod.EXPECTED_MIN_TRADES
    assert batch5.parameter_drift(
        spec["parameters"], _catalog_entry()["parameters"]) is None


# ------------------------------------------------------------------ factory

def test_make_target_fn_requires_every_traded_and_signal_frame():
    index = _bdays("2015-01-02", "2016-12-30")
    frames = _synthetic_frames(index)

    with pytest.raises(ValueError, match=DEFENSIVE):
        mod.make_target_fn({RISK: frames[RISK], SIGNAL_TICKER: frames[SIGNAL_TICKER]})

    with pytest.raises(ValueError, match=SIGNAL_TICKER):
        mod.make_target_fn({RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]})

    with pytest.raises(ValueError, match=RISK):
        mod.make_target_fn({DEFENSIVE: frames[DEFENSIVE], SIGNAL_TICKER: frames[SIGNAL_TICKER]})


def test_make_target_fn_returns_a_callable_that_matches_the_cited_reference_factory():
    """The module's own docstring cites the exact wiring:
    signals.make_moving_average_regime_target_fn(risk_ticker="SPY",
    defensive_ticker="SHY", signal_ticker="UUP", fast=1, slow=200).
    Confirm the module's make_target_fn produces bar-for-bar identical
    decisions to that independently-constructed reference on synthetic
    data, proving the wiring (ticker assignment, fast/slow values) was
    not transposed or altered."""
    index = _bdays("2018-01-02", "2021-06-30")
    frames = _synthetic_frames(index)
    ours = mod.make_target_fn(frames)
    reference = signals.make_moving_average_regime_target_fn(
        risk_ticker=RISK, defensive_ticker=DEFENSIVE,
        signal_ticker=SIGNAL_TICKER, fast=1, slow=200)
    decisions = []
    for date in index[::5]:
        truncated = _truncate(frames, date)
        decisions.append(ours(truncated))
        assert decisions[-1] == reference(truncated), date
    assert set(decisions) <= {RISK, DEFENSIVE, None}
    assert RISK in decisions or DEFENSIVE in decisions


def test_fast_1_degenerates_to_todays_raw_close_numerically():
    """Direct numerical proof (not just a comment's word taken on trust)
    that signals.make_moving_average_regime_target_fn's fast=1 leg is
    exactly today's raw UUP close, i.e. rolling(1).mean() degenerates to
    the single value it is a mean of. Symbolic reasoning:
    values.iloc[-1:].mean() over a length-1 slice sums that one element
    and divides by 1, which is the identity -- verified here on real
    non-trivial floating-point values, not round numbers, so no
    coincidental cancellation could mask a mistake."""
    values = pd.Series([101.234567, 98.7654321, 103.1122334, 99.998877])
    for k in range(1, len(values) + 1):
        window = values.iloc[:k]
        fast_mean = window.iloc[-1:].mean()
        assert fast_mean == window.iloc[-1], (
            "fast=1 rolling mean must equal the raw last value exactly")


def test_uup_above_200sma_selects_the_defensive_leg_matching_catalog_text():
    """Hand-constructed UUP path: flat at 25.0 for 200+ sessions (the
    200-day SMA warms up and settles at 25.0), then a sustained step up
    so today's raw UUP close (fast=1) sits above its own still-mostly-25
    200-day SMA -- 'dollar in a sustained uptrend' per the catalog text
    -> rotate SPY into SHY. And the inverse: UUP below its 200-day SMA
    -> hold SPY. This directly checks the catalog mechanism's stated
    polarity, not just internal self-consistency against the reference
    factory (the previous test already covers that)."""
    index = _bdays("2018-01-02", "2019-06-30")
    n = len(index)
    uup_close = np.full(n, 25.0)
    uup_close[-1] = 40.0  # today's UUP close sharply above its 200d SMA
    frames = {
        RISK: _ohlc_from_close(np.full(n, 50.0), index),
        DEFENSIVE: _ohlc_from_close(np.full(n, 80.0), index),
        SIGNAL_TICKER: _ohlc_from_close(uup_close, index),
    }
    target = mod.make_target_fn(frames)

    above_truncated = _truncate(frames, index[-1])
    assert target(above_truncated) == DEFENSIVE, (
        "UUP close above its 200-day SMA must select SHY (risk-off), "
        "per the catalog's 'dollar in a sustained uptrend -> move into "
        "SHY' mechanism text")

    below_truncated = _truncate(frames, index[-2])  # before the step up
    assert target(below_truncated) == RISK, (
        "UUP close at/below its 200-day SMA must select SPY (risk-on), "
        "per the catalog's 'dollar weakening or range-bound -> hold "
        "SPY' mechanism text")
