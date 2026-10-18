"""Tests for src/mining/batch5/vix1d_vix9d_inversion_deleverage_trigger.py.

Synthetic frames only (no network). This family has no catalog or
accepted.json entry (see the module's own docstring), so this file has no
accepted.json parameter check. Contract-shape validation still runs via the
real ``run_batch5_mining.validate_module_contract``, which needs no
catalog/accepted.json entry to work.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import rotation_engine  # noqa: E402
from mining.batch5 import vix1d_vix9d_inversion_deleverage_trigger as mod  # noqa: E402


def _index(n: int = 40) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


def _ohlc(index: pd.DatetimeIndex, closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, closes) * 1.001,
        "Low": np.minimum(opens, closes) * 0.999,
        "Close": closes,
    }, index=index)


def _frames(index: pd.DatetimeIndex | None = None, *,
            vix1d=None, vix9d=None) -> dict:
    index = _index() if index is None else index
    n = len(index)
    xlk = 100.0 * np.cumprod(1.0 + np.full(n, 0.0005))
    synthetic = 100.0 * np.cumprod(1.0 + np.full(n, 0.0005) * mod.LEVERAGE)
    if vix1d is None:
        vix1d = np.full(n, 12.0)
    if vix9d is None:
        vix9d = np.full(n, 14.0)
    return {
        mod.TRADED[0]: _ohlc(index, synthetic),
        mod.TRADED[1]: _ohlc(index, xlk),
        mod.SIGNAL[0]: _ohlc(index, vix1d),
        mod.SIGNAL[1]: _ohlc(index, vix9d),
    }


def _truncated(frames: dict, stop: int, start: int = 0) -> dict:
    return {key: frame.iloc[start:stop + 1].copy() for key, frame in frames.items()}


# ------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == "vix1d_vix9d_inversion_deleverage_trigger"
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == ["XLK_2x_synthetic", "XLK"]
    assert mod.SIGNAL == ["^VIX1D", "^VIX9D"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "XLK"
    assert mod.EXPECTED_MIN_TRADES == 0
    assert callable(mod.make_target_fn)


def test_driver_contract_validation():
    import run_batch5_mining as batch5
    spec = batch5.validate_module_contract(mod)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == ["XLK_2x_synthetic", "XLK"]
    assert spec["signal"] == ["^VIX1D", "^VIX9D"]


# ------------------------------------------------------------- naming helpers

def test_synthetic_id_round_trip():
    assert mod.synthetic_id_for("XLK") == "XLK_2x_synthetic"
    assert mod.underlying_of("XLK_2x_synthetic") == "XLK"
    with pytest.raises(ValueError):
        mod.underlying_of("XLK")


def test_synthetic_ohlc_from_close_shape():
    index = _index(5)
    close = np.array([10.0, 11.0, 9.0, 9.0, 12.0])
    frame = mod.synthetic_ohlc_from_close(index, close, name="X")
    assert list(frame.columns) == ["Open", "High", "Low", "Close"]
    assert frame["Open"].iloc[0] == close[0]
    assert (frame["Open"].to_numpy()[1:] == close[:-1]).all()
    assert (frame["High"].to_numpy() == close).all()
    assert (frame["Low"].to_numpy() == close).all()
    assert (frame["Close"].to_numpy() == close).all()


# ------------------------------------------------------------- the rule

def test_contango_selects_leveraged_leg():
    frames = _frames(vix1d=np.full(40, 12.0), vix9d=np.full(40, 14.0))
    target = mod.make_target_fn(frames)
    assert target(_truncated(frames, 10)) == mod.TRADED[0]


def test_inversion_selects_unlevered_leg():
    frames = _frames(vix1d=np.full(40, 16.0), vix9d=np.full(40, 14.0))
    target = mod.make_target_fn(frames)
    assert target(_truncated(frames, 10)) == mod.TRADED[1]


def test_tie_resolves_to_unlevered_leg():
    frames = _frames(vix1d=np.full(40, 14.0), vix9d=np.full(40, 14.0))
    target = mod.make_target_fn(frames)
    assert target(_truncated(frames, 10)) == mod.TRADED[1]


def test_reverts_immediately_no_smoothing():
    """Daily re-decision, no persistence: contango -> inversion -> contango
    on three consecutive bars flips the target each time, with no
    hysteresis or minimum-dwell smoothing anywhere in the rule."""
    vix1d = np.array([12.0, 16.0, 12.0, 16.0, 12.0])
    vix9d = np.full(5, 14.0)
    index = _index(5)
    frames = _frames(index, vix1d=vix1d, vix9d=vix9d)
    target = mod.make_target_fn(frames)
    expected = [mod.TRADED[0], mod.TRADED[1], mod.TRADED[0],
                mod.TRADED[1], mod.TRADED[0]]
    actual = [target(_truncated(frames, t)) for t in range(5)]
    assert actual == expected


def test_undefined_signal_is_flat_not_fabricated():
    index = _index(10)
    vix1d = np.array([12.0, 12.0, np.nan, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
    vix9d = np.full(10, 14.0)
    frames = _frames(index, vix1d=vix1d, vix9d=vix9d)
    target = mod.make_target_fn(frames)
    assert target(_truncated(frames, 2)) is None
    assert target(_truncated(frames, 0)) == mod.TRADED[0]


def test_rejects_missing_signal_frame_entirely():
    frames = _frames()
    incomplete = {k: v for k, v in frames.items() if k != mod.SIGNAL[1]}
    with pytest.raises(ValueError):
        mod.make_target_fn(incomplete)


def test_rejects_signal_frame_missing_ohlc_columns():
    frames = _frames()
    broken = dict(frames)
    broken[mod.SIGNAL[0]] = frames[mod.SIGNAL[0]].drop(columns=["Close"])
    with pytest.raises(ValueError):
        mod.make_target_fn(broken)


def test_a_single_nan_signal_print_does_not_raise_at_factory_time():
    """A live-feed gap on one day must degrade gracefully (see
    test_undefined_signal_is_flat_not_fabricated for the per-bar
    behavior) -- it must NOT halt construction of the whole target_fn,
    since that would blind every OTHER bar's decision too."""
    frames = _frames()
    bad = frames[mod.SIGNAL[0]].copy()
    bad.loc[bad.index[5], "Close"] = np.nan
    broken = dict(frames)
    broken[mod.SIGNAL[0]] = bad
    mod.make_target_fn(broken)  # must not raise


def test_rejects_traded_leg_with_nonfinite_close():
    """Unlike the signal legs, a TRADED leg's own Close must be clean
    across its whole history -- these prices are actually filled/marked
    by the engine, so a NaN there is a real data defect, not a benign
    live-feed gap."""
    frames = _frames()
    bad = frames[mod.TRADED[1]].copy()
    bad.loc[bad.index[5], "Close"] = np.nan
    broken = dict(frames)
    broken[mod.TRADED[1]] = bad
    with pytest.raises(ValueError):
        mod.make_target_fn(broken)


# -------------------------------------------------------------- look-ahead

def test_lookahead_probe_future_signal_bars_never_change_past_targets():
    index = _index(30)
    rng = np.random.default_rng(7)
    vix1d = 12.0 + rng.normal(0, 0.5, size=30)
    vix9d = 14.0 + rng.normal(0, 0.5, size=30)
    frames = _frames(index, vix1d=vix1d, vix9d=vix9d)
    k = 15
    before = [mod.make_target_fn(frames)(_truncated(frames, t)) for t in range(k + 1)]

    perturbed = {key: frame.copy() for key, frame in frames.items()}
    for key in mod.SIGNAL:
        tail = perturbed[key].index[k + 1:]
        perturbed[key].loc[tail, "Close"] *= 5.0  # a huge, sign-flipping jump
        perturbed[key].loc[tail, "Open"] *= 5.0
        perturbed[key].loc[tail, "High"] *= 5.0
        perturbed[key].loc[tail, "Low"] *= 5.0
    after_fn = mod.make_target_fn(perturbed)
    after = [after_fn(_truncated(perturbed, t)) for t in range(k + 1)]
    assert before == after


# -------------------------------------------------------------- end to end

def test_end_to_end_rotation_engine_switches_on_inversion():
    index = _index(20)
    # Contango for the first 8 bars, inverted for the next 6, contango again.
    vix1d = np.concatenate([np.full(8, 12.0), np.full(6, 16.0), np.full(6, 12.0)])
    vix9d = np.full(20, 14.0)
    frames = _frames(index, vix1d=vix1d, vix9d=vix9d)
    target = mod.make_target_fn(frames)
    asset_frames = {mod.TRADED[0]: frames[mod.TRADED[0]], mod.TRADED[1]: frames[mod.TRADED[1]]}
    signal_frames = {mod.SIGNAL[0]: frames[mod.SIGNAL[0]], mod.SIGNAL[1]: frames[mod.SIGNAL[1]]}
    trades, active = rotation_engine.run_two_asset_rotation(
        asset_frames, target, signal_frames=signal_frames,
        cost_bps_by_asset={mod.TRADED[0]: 2.0, mod.TRADED[1]: 1.0})
    assert len(trades) >= 2
    assert set(trades["asset"]).issubset(set(mod.TRADED))
    assert set(trades["asset"]) == set(mod.TRADED)
    switches = trades[trades["reason"] == "switch"]
    assert len(switches) >= 1
    assert active.notna().sum() > 0


def test_end_to_end_permanent_contango_never_switches():
    index = _index(20)
    frames = _frames(index, vix1d=np.full(20, 12.0), vix9d=np.full(20, 14.0))
    target = mod.make_target_fn(frames)
    asset_frames = {mod.TRADED[0]: frames[mod.TRADED[0]], mod.TRADED[1]: frames[mod.TRADED[1]]}
    signal_frames = {mod.SIGNAL[0]: frames[mod.SIGNAL[0]], mod.SIGNAL[1]: frames[mod.SIGNAL[1]]}
    trades, active = rotation_engine.run_two_asset_rotation(
        asset_frames, target, signal_frames=signal_frames,
        cost_bps_by_asset={mod.TRADED[0]: 2.0, mod.TRADED[1]: 1.0})
    assert set(trades["asset"]) == {mod.TRADED[0]}
    assert (active.dropna() == mod.TRADED[0]).all()
