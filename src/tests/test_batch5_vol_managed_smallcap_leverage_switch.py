"""Tests for src/mining/batch5/vol_managed_smallcap_leverage_switch.py.

Synthetic frames only (no network).
"""
import json
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
from mining.batch5 import vol_managed_smallcap_leverage_switch as mod  # noqa: E402

# Regime schedule by calendar-month ordinal (0-based) of the synthetic index.
# Daily LOG returns alternate in sign with the given magnitude, so the
# 21-day realized variance is 21 * magnitude**2 inside a regime.
_BASE, _CALM, _WILD = 0.01, 0.0001, 0.03
_CALM_MONTHS = {18, 19, 22, 23, 24, 25}
_WILD_MONTHS = {20, 21}


def _index(n: int = 560) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _month_ordinals(index: pd.DatetimeIndex) -> np.ndarray:
    periods = index.to_period("M")
    first = periods[0]
    return np.array([(p.year - first.year) * 12 + (p.month - first.month)
                     for p in periods])


def _month_end_positions(index: pd.DatetimeIndex) -> dict:
    """{month ordinal: position of that month's last bar}."""
    ordinals = _month_ordinals(index)
    out = {}
    for pos, m in enumerate(ordinals):
        out[int(m)] = pos
    return out


def _iwm_closes(index: pd.DatetimeIndex) -> np.ndarray:
    ordinals = _month_ordinals(index)
    magnitude = np.where(
        np.isin(ordinals, list(_CALM_MONTHS)), _CALM,
        np.where(np.isin(ordinals, list(_WILD_MONTHS)), _WILD, _BASE))
    signs = np.where(np.arange(len(index)) % 2 == 0, 1.0, -1.0)
    log_returns = magnitude * signs
    log_returns[0] = 0.0
    return 100.0 * np.exp(np.cumsum(log_returns))


def _ohlc(index: pd.DatetimeIndex, closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, closes) * 1.001,
        "Low": np.minimum(opens, closes) * 0.999,
        "Close": closes,
    }, index=index)


def _frames(index: pd.DatetimeIndex | None = None) -> dict:
    index = _index() if index is None else index
    iwm = _iwm_closes(index)
    tna = 50.0 * (iwm / iwm[0]) ** 3  # any positive series; not a signal input
    return {"TNA": _ohlc(index, tna), "IWM": _ohlc(index, iwm)}


def _truncated(frames: dict, stop: int, start: int = 0) -> dict:
    return {key: frame.iloc[start:stop + 1].copy() for key, frame in frames.items()}


def _targets_through(target, frames, k: int) -> list:
    return [target(_truncated(frames, t)) for t in range(k + 1)]


# ------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == "vol_managed_smallcap_leverage_switch"
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == ["TNA", "IWM"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "IWM"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES == 60
    assert callable(mod.make_target_fn)


# ------------------------------------------------------------- the rule

def test_rule_fires_tna_after_calm_month_and_iwm_after_wild_month():
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["IWM"].index)
    # Fully calm month (its 21-bar window lies entirely inside a calm regime).
    assert target(_truncated(frames, ends[19])) == "TNA"
    assert target(_truncated(frames, ends[23])) == "TNA"
    # Fully wild month.
    assert target(_truncated(frames, ends[21])) == "IWM"


def test_no_position_until_min_obs_rv_observations_exist():
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["IWM"].index)
    warmup_bars = mod.PARAMETERS["rv_lookback_days"] + mod.PARAMETERS["threshold_min_obs"]
    # Every month-end decision before the warm-up completes is None ...
    early = [m for m, pos in ends.items() if pos < warmup_bars]
    assert early, "fixture must contain month-ends inside the warm-up"
    for m in early:
        assert target(_truncated(frames, ends[m])) is None
    # ... and the first one after it is defined.
    late = min(m for m, pos in ends.items() if pos >= warmup_bars)
    assert target(_truncated(frames, ends[late])) in ("TNA", "IWM")
    # A mid-warm-up ordinary bar is flat too.
    assert target(_truncated(frames, 100)) is None


def test_no_intra_month_re_evaluation_between_decisions():
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["IWM"].index)
    # Month 20 is wild, but its decision only lands at ITS month-end; every
    # bar inside month 20 still carries month 19's (calm -> TNA) decision.
    for pos in range(ends[19] + 1, ends[20]):
        assert target(_truncated(frames, pos)) == "TNA"
    assert target(_truncated(frames, ends[20])) == "IWM"
    # Likewise month 22 (calm) keeps month 21's IWM until its own month-end.
    for pos in range(ends[21] + 1, ends[22]):
        assert target(_truncated(frames, pos)) == "IWM"


def test_tie_rv_equal_to_median_counts_as_low_variance():
    # Closes alternating between two exact powers of two give log returns
    # that are exact negatives, so every 21-day RV is bit-identical and
    # the expanding median EQUALS the current RV: the inclusive `<=` in
    # the pseudocode must resolve that tie to the low-variance leg.
    assert np.log(2.0) ** 2 == np.log(0.5) ** 2
    index = _index(400)
    closes = np.where(np.arange(len(index)) % 2 == 0, 1.0, 2.0)
    frames = {"TNA": _ohlc(index, np.repeat(10.0, len(index))),
              "IWM": _ohlc(index, closes)}
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(index)
    late = [pos for pos in ends.values()
            if pos >= mod.PARAMETERS["rv_lookback_days"]
            + mod.PARAMETERS["threshold_min_obs"]]
    assert late
    for pos in late:
        assert target(_truncated(frames, pos)) == "TNA"


def test_bucket_starting_mid_month_inherits_standing_decision():
    """The holdout bucket is handed to the engine padded but still starting
    long after IWM's first bar; a decision bar before the bucket start must
    still govern the bars inside it (as-of lookup, not 'flat until the
    first in-bucket month-end')."""
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["IWM"].index)
    bucket_start = ends[19] + 5          # mid month 20
    probe = ends[19] + 10                # still inside month 20
    assert target(_truncated(frames, probe, start=bucket_start)) == "TNA"
    # And the first in-bucket month-end re-decides normally.
    assert target(_truncated(frames, ends[20], start=bucket_start)) == "IWM"


# -------------------------------------------------------------- look-ahead

def test_lookahead_probe_future_bars_never_change_past_targets():
    frames = _frames()
    ends = _month_end_positions(frames["IWM"].index)
    k = ends[22] + 1  # first bar of calm month 23; a jump lands inside its RV window
    before = _targets_through(mod.make_target_fn(frames), frames, k)

    perturbed = {}
    for key, frame in frames.items():
        frame = frame.copy()
        tail = frame.index[k + 1:]
        frame.loc[tail, "Close"] *= 1.3
        frame.loc[tail, "Open"] *= 1.3
        frame.loc[tail, "High"] *= 1.5
        frame.loc[tail, "Low"] *= 0.7
        perturbed[key] = frame
    after_fn = mod.make_target_fn(perturbed)
    after = _targets_through(after_fn, perturbed, k)
    assert before == after

    # The probe has teeth: the same perturbation DOES change a later decision
    # (the +30% jump sits inside month 23's trailing RV window).
    unperturbed_fn = mod.make_target_fn(frames)
    assert unperturbed_fn(_truncated(frames, ends[23])) == "TNA"
    assert after_fn(_truncated(perturbed, ends[23])) == "IWM"


# -------------------------------------------------------------- end to end

def test_end_to_end_rotation_engine_produces_trades_switching_at_month_starts():
    frames = _frames(_index(600))
    target = mod.make_target_fn(frames)
    trades, active = rotation_engine.run_two_asset_rotation(
        frames, target, cost_bps_by_asset={"TNA": 2.0, "IWM": 1.0})
    assert len(trades) >= 1
    assert set(trades["asset"]).issubset({"TNA", "IWM"})
    assert set(trades["asset"]) == {"TNA", "IWM"}
    # Every switch executes on the first bar of a calendar month (decision
    # on the last trading day, fill at the next open).
    periods = frames["IWM"].index.to_period("M")
    first_of_month = {frames["IWM"].index[i] for i in range(1, len(periods))
                      if periods[i] != periods[i - 1]}
    switches = trades[trades["reason"] == "switch"]
    assert len(switches) >= 1
    for entry_date in trades["entry_date"]:
        assert entry_date in first_of_month
    # Held segments span whole months: no intra-month churn.
    assert (trades["held"] >= 15).all()
    assert active.notna().sum() > 0


def test_rejects_missing_or_bad_signal_frame():
    index = _index(300)
    frames = _frames(index)
    with pytest.raises(ValueError):
        mod.make_target_fn({"TNA": frames["TNA"]})
    bad = frames["IWM"].copy()
    bad.loc[bad.index[50], "Close"] = np.nan
    with pytest.raises(ValueError):
        mod.make_target_fn({"TNA": frames["TNA"], "IWM": bad})
