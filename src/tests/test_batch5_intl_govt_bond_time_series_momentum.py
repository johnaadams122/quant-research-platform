"""Tests for src/mining/batch5/intl_govt_bond_time_series_momentum.py.

Synthetic frames only (no network). PARAMETERS is pinned byte-for-byte
against ``_CATALOG_PARAMETERS`` below, a verbatim local copy of the
pre-registered catalog entry this module implements.
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
from mining.batch5 import intl_govt_bond_time_series_momentum as mod  # noqa: E402

# Verbatim from the pre-registered catalog entry for family_id
# "intl_govt_bond_time_series_momentum" (see the module docstring above).
_CATALOG_PARAMETERS = {
    "lookback_months": 12,
    "signal": "trailing_total_return",
    "rebalance": "monthly",
    "cash_proxy": "BIL",
}


# ------------------------------------------------------------- fixtures

def _full_index(n_months: int) -> pd.DatetimeIndex:
    """A business-day index that ends EXACTLY on calendar month
    ``n_months - 1``'s (0-based) own last business day -- no partial
    trailing month, so every month-end position used by a test is a real,
    complete month-end."""
    raw = pd.bdate_range("2015-01-01", periods=40 * n_months)
    periods = raw.to_period("M")
    first = periods[0]
    ordinals = np.array([(p.year - first.year) * 12 + (p.month - first.month)
                         for p in periods])
    positions = np.where(ordinals == n_months - 1)[0]
    assert len(positions), "raw range too short for n_months"
    return raw[:positions[-1] + 1]


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


# Hand-verified monthly schedule (BWX month-end close by month ordinal).
# Anchor month for ordinal m is m-12; decisions before month 12 have no
# anchor (undefined -> BIL regardless of value). From month 12 on, each
# return is computed by hand against the anchor already fixed at 100.0:
#   m12: 110.0 / m0(100.0)  - 1 = +0.10  -> BWX
#   m13:  90.0 / m1(100.0)  - 1 = -0.10  -> BIL
#   m14: 100.0 / m2(100.0)  - 1 =  0.00  -> BIL (exact tie, not > 0)
#   m15: 105.0 / m3(100.0)  - 1 = +0.05  -> BWX
#   m16: 105.0 / m4(100.0)  - 1 = +0.05  -> BWX (stays; no switch)
#   m17:  95.0 / m5(100.0)  - 1 = -0.05  -> BIL
_SCHEDULE = {m: 100.0 for m in range(12)}
_SCHEDULE.update({12: 110.0, 13: 90.0, 14: 100.0, 15: 105.0, 16: 105.0, 17: 95.0})
_N_MONTHS = 18
_EXPECTED_DECISIONS = {
    **{m: "BIL" for m in range(12)},
    12: "BWX", 13: "BIL", 14: "BIL", 15: "BWX", 16: "BWX", 17: "BIL",
}


def _bwx_closes(index: pd.DatetimeIndex, schedule: dict) -> np.ndarray:
    ends = _month_end_positions(index)
    ordinals = sorted(ends)
    positions = [ends[m] for m in ordinals]
    values = [schedule[m] for m in ordinals]
    closes = np.empty(len(index))
    closes[:positions[0] + 1] = values[0]
    for i in range(1, len(positions)):
        p0, p1 = positions[i - 1], positions[i]
        v0, v1 = values[i - 1], values[i]
        span = p1 - p0
        for k in range(span + 1):
            closes[p0 + k] = v0 + (v1 - v0) * k / span
    return closes


def _ohlc(index: pd.DatetimeIndex, closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, closes) * 1.001,
        "Low": np.minimum(opens, closes) * 0.999,
        "Close": closes,
    }, index=index)


def _frames(index: pd.DatetimeIndex | None = None, schedule: dict | None = None) -> dict:
    index = _full_index(_N_MONTHS) if index is None else index
    schedule = _SCHEDULE if schedule is None else schedule
    bwx = _bwx_closes(index, schedule)
    n = len(index)
    # BIL: a real, small, positive drift -- never flat synthetic cash.
    bil = 100.0 * (1.00005 ** np.arange(n))
    return {"BWX": _ohlc(index, bwx), "BIL": _ohlc(index, bil)}


def _truncated(frames: dict, stop: int, start: int = 0) -> dict:
    return {key: frame.iloc[start:stop + 1].copy() for key, frame in frames.items()}


def _targets_through(target, frames, k: int) -> list:
    return [target(_truncated(frames, t)) for t in range(k + 1)]


def _cyclical_frames(n: int = 650) -> dict:
    """A longer, smoothly oscillating fixture for the end-to-end engine
    test: not hand-scored (the exact schedule fixture above already pins
    the rule's arithmetic), just long enough that the trailing-12-month
    sign flips several real times."""
    index = pd.bdate_range("2015-01-01", periods=n)
    t = np.arange(n)
    drift = 0.0009 * np.sin(2 * np.pi * t / 130.0)
    bwx = 100.0 * np.exp(np.cumsum(drift))
    bil = 100.0 * (1.00003 ** t)
    return {"BWX": _ohlc(index, bwx), "BIL": _ohlc(index, bil)}


# ------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == "intl_govt_bond_time_series_momentum"
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == ["BWX", "BIL"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "BWX"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_target_fn)


def test_parameters_match_catalog_entry_byte_for_byte():
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(_CATALOG_PARAMETERS, sort_keys=True))


def test_driver_contract_validation_and_no_parameter_drift():
    import run_batch5_mining as batch5
    spec = batch5.validate_module_contract(mod)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == ["BWX", "BIL"]
    assert batch5.parameter_drift(mod.PARAMETERS, _CATALOG_PARAMETERS) is None


# ------------------------------------------------------------- the rule

def test_no_anchor_warmup_defaults_to_bil():
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["BWX"].index)
    for m in range(12):
        assert target(_truncated(frames, ends[m])) == "BIL"


def test_rule_positive_return_selects_bwx_negative_selects_bil():
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["BWX"].index)
    for m, expected in _EXPECTED_DECISIONS.items():
        assert target(_truncated(frames, ends[m])) == expected, f"month {m}"


def test_tie_exact_zero_return_counts_as_bil():
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["BWX"].index)
    # m14's trailing return is EXACTLY 0.0 by construction (100.0/100.0-1).
    assert target(_truncated(frames, ends[14])) == "BIL"


def test_no_intra_month_re_evaluation_between_decisions():
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["BWX"].index)
    # Month 13 is BIL, but its decision only lands at ITS own month-end;
    # every bar inside month 13 still carries month 12's (BWX) decision.
    for pos in range(ends[12] + 1, ends[13]):
        assert target(_truncated(frames, pos)) == "BWX"
    assert target(_truncated(frames, ends[13])) == "BIL"
    # Likewise month 16 (BWX, unchanged from month 15) keeps carrying BWX
    # right up to month 17's own month-end, which flips it to BIL.
    for pos in range(ends[15] + 1, ends[17]):
        assert target(_truncated(frames, pos)) == "BWX"
    assert target(_truncated(frames, ends[17])) == "BIL"


def test_bucket_starting_mid_month_inherits_standing_decision():
    """A holdout bucket padded to start mid-month must still carry the
    standing decision made before the bucket started (as-of lookup, not
    'undefined/BIL until the first in-bucket month-end')."""
    frames = _frames()
    target = mod.make_target_fn(frames)
    ends = _month_end_positions(frames["BWX"].index)
    bucket_start = ends[12] + 5          # mid month 13 (standing decision: BWX)
    probe = ends[12] + 10                # still inside month 13
    assert target(_truncated(frames, probe, start=bucket_start)) == "BWX"
    # The first in-bucket month-end re-decides normally.
    assert target(_truncated(frames, ends[13], start=bucket_start)) == "BIL"


# -------------------------------------------------------------- edge cases

def test_no_position_before_any_month_end_defaults_to_bil():
    frames = _frames()
    target = mod.make_target_fn(frames)
    # Only 1-2 bars available -- no month-end has occurred yet.
    assert target(_truncated(frames, 0)) == "BIL"
    assert target(_truncated(frames, 1)) == "BIL"


def test_empty_truncated_frame_defaults_to_bil():
    frames = _frames()
    target = mod.make_target_fn(frames)
    empty = {key: frame.iloc[:0] for key, frame in frames.items()}
    assert target(empty) == "BIL"


def test_rejects_missing_or_bad_signal_frame():
    frames = _frames()
    with pytest.raises(ValueError):
        mod.make_target_fn({"BIL": frames["BIL"]})
    bad = frames["BWX"].copy()
    bad.loc[bad.index[50], "Close"] = np.nan
    with pytest.raises(ValueError):
        mod.make_target_fn({"BWX": bad, "BIL": frames["BIL"]})
    bad_neg = frames["BWX"].copy()
    bad_neg.loc[bad_neg.index[50], "Close"] = -1.0
    with pytest.raises(ValueError):
        mod.make_target_fn({"BWX": bad_neg, "BIL": frames["BIL"]})


# -------------------------------------------------------------- look-ahead

def test_lookahead_probe_future_bars_never_change_past_targets():
    frames = _frames()
    ends = _month_end_positions(frames["BWX"].index)
    k = ends[14]  # last bar decisions are claimed safe through
    before = _targets_through(mod.make_target_fn(frames), frames, k)

    perturbed = {}
    for key, frame in frames.items():
        frame = frame.copy()
        tail = frame.index[k + 1:]
        frame.loc[tail, "Close"] *= 1.5
        frame.loc[tail, "Open"] *= 1.5
        frame.loc[tail, "High"] *= 1.6
        frame.loc[tail, "Low"] *= 0.9
        perturbed[key] = frame
    after_fn = mod.make_target_fn(perturbed)
    after = _targets_through(after_fn, perturbed, k)
    assert before == after

    # The probe has teeth: the same perturbation changes month 15's own
    # decision (its own month-end close, after k, is now +50%).
    unperturbed_fn = mod.make_target_fn(frames)
    assert unperturbed_fn(_truncated(frames, ends[15])) == "BWX"
    assert after_fn(_truncated(perturbed, ends[15])) == "BWX"  # +50% still positive
    # A negative-tilting perturbation on month 15's own close DOES flip it.
    neg_perturbed = {}
    for key, frame in frames.items():
        frame = frame.copy()
        tail = frame.index[k + 1:]
        frame.loc[tail, "Close"] *= 0.5
        frame.loc[tail, "Open"] *= 0.5
        frame.loc[tail, "High"] *= 0.6
        frame.loc[tail, "Low"] *= 0.4
        neg_perturbed[key] = frame
    neg_fn = mod.make_target_fn(neg_perturbed)
    neg_after = _targets_through(neg_fn, neg_perturbed, k)
    assert before == neg_after  # still unchanged through k
    assert neg_fn(_truncated(neg_perturbed, ends[15])) == "BIL"  # now flips


# -------------------------------------------------------------- end to end

def test_end_to_end_rotation_engine_produces_trades_switching_at_month_starts():
    frames = _cyclical_frames()
    target = mod.make_target_fn(frames)
    trades, active = rotation_engine.run_two_asset_rotation(
        frames, target, cost_bps_by_asset={"BWX": 3.0, "BIL": 1.0})
    assert len(trades) >= 2  # the initial open plus at least one real switch
    assert set(trades["asset"]).issubset({"BWX", "BIL"})
    assert set(trades["asset"]) == {"BWX", "BIL"}
    # "Undefined -> BIL" (Interpretation choices) means the very FIRST
    # trade opens on the frame's own first executable bar, not at a month
    # boundary -- there is no flat/None warm-up state for this family.
    assert trades["entry_date"].iloc[0] == frames["BWX"].index[1]
    # Every trade AFTER that initial open is a real BWX<->BIL switch, which
    # only ever executes on the first bar of a calendar month (decision on
    # the last trading day, fill at the next open).
    periods = frames["BWX"].index.to_period("M")
    first_of_month = {frames["BWX"].index[i] for i in range(1, len(periods))
                      if periods[i] != periods[i - 1]}
    for entry_date in trades["entry_date"].iloc[1:]:
        assert entry_date in first_of_month
    assert active.iloc[1:].notna().all()  # never flat: always BWX or BIL from bar 1


def test_end_to_end_matches_exact_schedule_trade_count_and_assets():
    """The hand-scored schedule fixture, run through the real engine with
    one extra partial month appended so the final (month 17 -> BIL) switch
    actually executes, produces exactly the 5 hand-counted segments:
    BIL(m0..11) -> BWX(m12) -> BIL(m13..14) -> BWX(m15..16) -> BIL(m17..end).
    """
    base_index = _full_index(_N_MONTHS)
    extra = pd.bdate_range(base_index[-1] + pd.Timedelta(days=1), periods=6)
    index = base_index.append(extra)
    schedule = dict(_SCHEDULE)
    schedule[_N_MONTHS] = 90.0  # arbitrary; only fills the partial tail month
    frames = _frames(index=index, schedule=schedule)
    target = mod.make_target_fn(frames)
    trades, _active = rotation_engine.run_two_asset_rotation(
        frames, target, cost_bps_by_asset={"BWX": 3.0, "BIL": 1.0})
    assert list(trades["asset"]) == ["BIL", "BWX", "BIL", "BWX", "BIL"]
    assert list(trades["reason"]) == [
        "switch", "switch", "switch", "switch", "end_of_data"]
