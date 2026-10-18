import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.rotation_engine import run_two_asset_rotation


def _frame(opens, closes=None, *, index=None):
    opens = np.asarray(opens, dtype=float)
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float)
    if index is None:
        index = pd.date_range("2024-01-02", periods=len(opens), freq="B")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes),
            "Low": np.minimum(opens, closes),
            "Close": closes,
        },
        index=pd.DatetimeIndex(index),
    )


def _basic_frames():
    return {
        "RISK": _frame([100, 100, 105, 110], [100, 101, 106, 120]),
        "SAFE": _frame([50, 50, 50, 50], [50, 50, 50, 55]),
    }


def _run(frames, target_fn, **kwargs):
    return run_two_asset_rotation(
        frames,
        target_fn,
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0},
        **kwargs,
    )


def test_risky_only_holding_is_entered_next_open_and_liquidated_final_close():
    frames = _basic_frames()
    segments, active = _run(frames, lambda decision_frames: "RISK")

    assert len(segments) == 1
    row = segments.iloc[0]
    assert row["asset"] == "RISK"
    assert row["entry_date"] == frames["RISK"].index[1]
    assert row["exit_date"] == frames["RISK"].index[-1]
    assert row["entry_px"] == 100.0
    assert row["exit_px"] == 120.0
    assert row["held"] == 2
    assert row["reason"] == "end_of_data"
    assert row["gross_ret"] == pytest.approx(0.20)
    assert list(active) == [None, "RISK", "RISK", "RISK"]


def test_defensive_only_holding_uses_defensive_prices():
    frames = _basic_frames()
    segments, active = _run(frames, lambda decision_frames: "SAFE")

    row = segments.iloc[0]
    assert row["asset"] == "SAFE"
    assert row["entry_px"] == 50.0
    assert row["exit_px"] == 55.0
    assert row["gross_ret"] == pytest.approx(0.10)
    assert list(active) == [None, "SAFE", "SAFE", "SAFE"]


def test_switch_closes_old_and_opens_new_at_same_next_open():
    frames = {
        "RISK": _frame([100, 100, 120, 130], [100, 101, 121, 130]),
        "SAFE": _frame([50, 50, 40, 45], [50, 50, 42, 50]),
    }

    def target(decision_frames):
        return "RISK" if len(decision_frames["RISK"]) == 1 else "SAFE"

    segments, active = _run(frames, target)
    assert list(segments["asset"]) == ["RISK", "SAFE"]
    first, second = segments.iloc[0], segments.iloc[1]
    assert first["exit_date"] == second["entry_date"] == frames["RISK"].index[2]
    assert first["exit_px"] == 120.0
    assert second["entry_px"] == 40.0
    assert first["reason"] == "switch"
    assert second["reason"] == "end_of_data"
    assert list(active) == [None, "RISK", "SAFE", "SAFE"]


def test_unchanged_target_does_not_churn_or_recharge_cost():
    frames = _basic_frames()
    segments, _ = run_two_asset_rotation(
        frames,
        lambda decision_frames: "RISK",
        cost_bps_by_asset={"RISK": 100.0, "SAFE": 250.0},
    )
    assert len(segments) == 1
    assert segments.iloc[0]["net_ret"] == pytest.approx(0.20 - 0.01)


def test_each_switched_segment_uses_its_own_round_trip_cost_once():
    frames = {
        "RISK": _frame([100, 100, 120, 120], [100, 100, 120, 120]),
        "SAFE": _frame([50, 50, 40, 40], [50, 50, 40, 50]),
    }

    def target(decision_frames):
        return "RISK" if len(decision_frames["RISK"]) == 1 else "SAFE"

    segments, _ = run_two_asset_rotation(
        frames,
        target,
        cost_bps_by_asset={"RISK": 100.0, "SAFE": 200.0},
    )
    assert segments.iloc[0]["net_ret"] == pytest.approx(0.20 - 0.01)
    assert segments.iloc[1]["net_ret"] == pytest.approx(0.25 - 0.02)


def test_positive_only_tax_haircut_leaves_losing_segment_unchanged():
    frames = {
        "RISK": _frame([100, 100, 120, 120], [100, 100, 120, 120]),
        "SAFE": _frame([50, 50, 40, 40], [50, 50, 40, 30]),
    }

    def target(decision_frames):
        return "RISK" if len(decision_frames["RISK"]) == 1 else "SAFE"

    segments, _ = run_two_asset_rotation(
        frames,
        target,
        cost_bps_by_asset={"RISK": 100.0, "SAFE": 200.0},
        tax_haircut_rate=0.10,
    )
    assert segments.iloc[0]["net_ret"] == pytest.approx((0.20 - 0.01) * 0.90)
    assert segments.iloc[1]["gross_ret"] == pytest.approx(-0.25)
    assert segments.iloc[1]["net_ret"] == pytest.approx(-0.25 - 0.02)


def test_none_target_stays_flat_then_returns_an_open_position_to_flat():
    frames = _basic_frames()

    def target(decision_frames):
        length = len(decision_frames["RISK"])
        if length == 1:
            return "RISK"
        return None

    segments, active = _run(frames, target)
    assert len(segments) == 1
    assert segments.iloc[0]["reason"] == "to_flat"
    assert segments.iloc[0]["exit_date"] == frames["RISK"].index[2]
    assert list(active) == [None, "RISK", None, None]


def test_none_warmup_stays_flat_until_target_can_execute_next_open():
    frames = {
        key: pd.concat([frame, frame.iloc[[-1]].set_axis(
            [frame.index[-1] + pd.offsets.BDay(1)])])
        for key, frame in _basic_frames().items()
    }

    def target(decision_frames):
        return "RISK" if len(decision_frames["RISK"]) >= 3 else None

    segments, active = _run(frames, target)
    assert list(active) == [None, None, None, "RISK", "RISK"]
    assert segments.iloc[0]["entry_date"] == frames["RISK"].index[3]


def test_empty_result_has_stable_schema_and_dtypes():
    segments, active = _run(_basic_frames(), lambda decision_frames: None)
    assert list(segments.columns) == [
        "asset", "entry_date", "exit_date", "entry_px", "exit_px",
        "held", "reason", "gross_ret", "net_ret",
    ]
    assert segments["asset"].dtype == object
    assert segments["entry_date"].dtype == "datetime64[ns]"
    assert segments["exit_date"].dtype == "datetime64[ns]"
    assert segments["entry_px"].dtype == np.float64
    assert segments["exit_px"].dtype == np.float64
    assert segments["held"].dtype == np.int64
    assert segments["reason"].dtype == object
    assert segments["gross_ret"].dtype == np.float64
    assert segments["net_ret"].dtype == np.float64
    assert active.dtype == object
    assert active.name == "active_asset"


def test_target_receives_one_dict_of_deep_copies_truncated_through_decision_bar():
    frames = _basic_frames()
    original = {key: frame.copy(deep=True) for key, frame in frames.items()}
    observations = []

    def target(decision_frames):
        observations.append({
            key: (len(frame), frame.index[-1])
            for key, frame in decision_frames.items()
        })
        decision_frames["RISK"].iloc[-1, 0] = 9999.0
        return None

    _run(frames, target)
    assert [obs["RISK"][0] for obs in observations] == [1, 2, 3]
    assert [obs["RISK"][1] for obs in observations] == list(
        frames["RISK"].index[:-1])
    for key in frames:
        pd.testing.assert_frame_equal(frames[key], original[key])


def test_signal_only_frame_is_truncated_and_never_becomes_active():
    frames = _basic_frames()
    signal = _frame([10, 10, 20, 20])
    seen = []

    def target(decision_frames):
        seen.append((set(decision_frames), len(decision_frames["SIGNAL"])))
        return "RISK" if decision_frames["SIGNAL"]["Close"].iloc[-1] < 15 else "SAFE"

    segments, active = run_two_asset_rotation(
        frames,
        target,
        signal_frames={"SIGNAL": signal},
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0},
    )
    assert seen == [
        ({"RISK", "SAFE", "SIGNAL"}, 1),
        ({"RISK", "SAFE", "SIGNAL"}, 2),
        ({"RISK", "SAFE", "SIGNAL"}, 3),
    ]
    assert set(active.dropna().unique()) <= {"RISK", "SAFE"}
    assert "SIGNAL" not in set(segments["asset"])


def test_future_asset_and_signal_values_cannot_change_prior_active_targets():
    frames = _basic_frames()
    signal = _frame([10, 10, 10, 10])
    changed_frames = {key: frame.copy(deep=True) for key, frame in frames.items()}
    changed_signal = signal.copy(deep=True)
    changed_frames["RISK"].loc[frames["RISK"].index[2]:, ["Open", "High", "Low", "Close"]] *= 5
    changed_signal.loc[signal.index[2]:, ["Open", "High", "Low", "Close"]] = 1000.0

    def target(decision_frames):
        return "RISK" if decision_frames["SIGNAL"]["Close"].iloc[-1] < 15 else "SAFE"

    _, original_active = run_two_asset_rotation(
        frames, target, signal_frames={"SIGNAL": signal},
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})
    _, changed_active = run_two_asset_rotation(
        changed_frames, target, signal_frames={"SIGNAL": changed_signal},
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})
    pd.testing.assert_series_equal(
        original_active.iloc[:3], changed_active.iloc[:3], check_exact=True)


def test_invalid_target_fails_closed():
    with pytest.raises(ValueError, match="unknown target.*CASH"):
        _run(_basic_frames(), lambda decision_frames: "CASH")


@pytest.mark.parametrize("asset_count", [1, 3])
def test_requires_exactly_two_traded_asset_frames(asset_count):
    frames = _basic_frames()
    if asset_count == 1:
        frames.pop("SAFE")
    else:
        frames["THIRD"] = frames["RISK"].copy()
    with pytest.raises(ValueError, match="exactly two"):
        run_two_asset_rotation(
            frames, lambda decision_frames: None,
            cost_bps_by_asset={key: 0.0 for key in frames})


def test_rejects_empty_or_misaligned_secondary_asset_before_target_call():
    calls = []

    def target(decision_frames):
        calls.append(1)
        return None

    frames = _basic_frames()
    frames["SAFE"] = frames["SAFE"].iloc[0:0]
    with pytest.raises(ValueError, match="nonempty"):
        _run(frames, target)
    assert calls == []

    frames = _basic_frames()
    frames["SAFE"] = frames["SAFE"].iloc[:-1]
    with pytest.raises(ValueError, match="identical.*calendar"):
        _run(frames, target)
    assert calls == []


def test_rejects_misaligned_signal_only_frame_before_target_call():
    calls = []

    def target(decision_frames):
        calls.append(1)
        return None

    signal = _frame([10, 10, 10], index=_basic_frames()["RISK"].index[:-1])
    with pytest.raises(ValueError, match="identical.*calendar"):
        run_two_asset_rotation(
            _basic_frames(), target, signal_frames={"SIGNAL": signal},
            cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})
    assert calls == []


def test_rejects_timezone_aware_traded_calendars_before_target_call():
    calls = []
    frames = {
        key: frame.set_axis(frame.index.tz_localize("UTC"))
        for key, frame in _basic_frames().items()
    }

    def target(decision_frames):
        calls.append(1)
        return None

    with pytest.raises(ValueError, match="timezone-naive"):
        _run(frames, target)
    assert calls == []


def test_rejects_timezone_aware_signal_calendar_before_target_call():
    calls = []
    signal = _frame([10, 10, 10, 10])
    signal = signal.set_axis(signal.index.tz_localize("UTC"))

    def target(decision_frames):
        calls.append(1)
        return None

    with pytest.raises(ValueError, match="timezone-naive"):
        run_two_asset_rotation(
            _basic_frames(), target, signal_frames={"SIGNAL": signal},
            cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})
    assert calls == []


@pytest.mark.parametrize("defect", ["unsorted", "duplicate"])
def test_rejects_non_strict_asset_calendar(defect):
    frames = _basic_frames()
    if defect == "unsorted":
        frames["SAFE"] = frames["SAFE"].iloc[[0, 2, 1, 3]]
    else:
        bad = frames["SAFE"].copy()
        bad.index = pd.DatetimeIndex([bad.index[0], bad.index[1],
                                      bad.index[1], bad.index[3]])
        frames["SAFE"] = bad
    with pytest.raises(ValueError, match="sorted unique"):
        _run(frames, lambda decision_frames: None)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("High", np.inf, "finite"), ("Open", 0.0, "positive"),
     ("Close", -1.0, "positive")],
)
def test_rejects_invalid_ohlc_before_target_call(column, value, message):
    calls = []
    frames = _basic_frames()
    frames["SAFE"] = frames["SAFE"].copy()
    frames["SAFE"].iloc[1, frames["SAFE"].columns.get_loc(column)] = value

    def target(decision_frames):
        calls.append(1)
        return None

    with pytest.raises(ValueError, match=message):
        _run(frames, target)
    assert calls == []


def test_rejects_missing_ohlc_column():
    frames = _basic_frames()
    frames["SAFE"] = frames["SAFE"].drop(columns="High")
    with pytest.raises(ValueError, match="required OHLC"):
        _run(frames, lambda decision_frames: None)


@pytest.mark.parametrize(
    "costs",
    [
        {"RISK": 0.0},
        {"RISK": 0.0, "SAFE": 0.0, "SIGNAL": 0.0},
        {"RISK": -1.0, "SAFE": 0.0},
        {"RISK": np.inf, "SAFE": 0.0},
    ],
)
def test_rejects_missing_extra_negative_or_nonfinite_asset_costs(costs):
    with pytest.raises(ValueError, match="cost_bps_by_asset"):
        run_two_asset_rotation(
            _basic_frames(), lambda decision_frames: None,
            cost_bps_by_asset=costs)


@pytest.mark.parametrize("rate", [-0.1, 1.0, np.inf])
def test_rejects_invalid_tax_haircut(rate):
    with pytest.raises(ValueError, match="tax_haircut_rate"):
        _run(_basic_frames(), lambda decision_frames: None,
             tax_haircut_rate=rate)


def test_active_asset_series_contains_at_most_one_traded_key_per_bar():
    frames = _basic_frames()

    def alternating_target(decision_frames):
        return "RISK" if len(decision_frames["RISK"]) % 2 else "SAFE"

    _, active = _run(frames, alternating_target)
    assert active.index.equals(frames["RISK"].index)
    assert len(active) == len(frames["RISK"])
    assert all(value is None or value in frames for value in active)


def test_negative_open_close_in_a_signal_frame_is_allowed():
    """A signal-only frame (e.g. a Treasury yield index like ^IRX, which had
    real negative closes in March 2020) is a decision input only -- it is
    never traded or priced, so `_validate_frame`'s Open/Close positivity gate
    (which exists to catch a nonsensical traded price) must not apply to it.
    Real traded asset frames keep the strict positivity gate unchanged --
    see test_rejects_invalid_ohlc_before_target_call above, which still
    covers "SAFE" (an asset_frames member)."""
    frames = _basic_frames()
    signal = frames["SAFE"].copy()
    signal.iloc[1, signal.columns.get_loc("Close")] = -0.05
    signal.iloc[1, signal.columns.get_loc("Open")] = -0.04

    calls = []

    def target(decision_frames):
        calls.append(decision_frames["SIGNAL"]["Close"].iloc[-1])
        return None

    _run(frames, target, signal_frames={"SIGNAL": signal})
    assert -0.05 in calls


def test_negative_open_close_in_an_asset_frame_still_rejected_even_when_a_signal_frame_is_present():
    """The positivity relaxation is scoped to signal_frames only -- a
    negative Open/Close in a real traded asset frame must still fail
    closed, whether or not signal_frames are also supplied."""
    frames = _basic_frames()
    frames["SAFE"].iloc[1, frames["SAFE"].columns.get_loc("Close")] = -1.0
    signal = _frame([10, 10, 10, 10])

    with pytest.raises(ValueError, match="positive"):
        _run(frames, lambda decision_frames: None,
             signal_frames={"SIGNAL": signal})


def test_non_finite_value_in_a_signal_frame_still_rejected():
    """Positivity is relaxed for signal frames; finiteness is not -- a NaN
    or inf signal value is still a real data defect, not a legitimate
    negative yield."""
    frames = _basic_frames()
    signal = _frame([10, 10, 10, 10])
    signal.iloc[1, signal.columns.get_loc("High")] = np.inf

    with pytest.raises(ValueError, match="finite"):
        _run(frames, lambda decision_frames: None,
             signal_frames={"SIGNAL": signal})
