"""Tests for src/mining/batch5/sector_ad_line_divergence_derisk.py.

Synthetic data only, no network: breadth is a synthetic panel.AsOfSeries
built in the test, and no data file is read.
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

from mining import backtest_engine, panel  # noqa: E402
from mining.batch5 import sector_ad_line_divergence_derisk as mod  # noqa: E402

FAMILY_ID = "sector_ad_line_divergence_derisk"
HIGH = 126
FLAT = 21


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
                            indicator="net_advance_ratio",
                            content_sha256="synthetic")


def _rising(n: int, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(np.full(n, 1.001))


def _signals(signal_fn, frame, positions):
    out = []
    for i in positions:
        sig = signal_fn(frame.iloc[:i + 1], i)
        out.append((bool(sig.entry), bool(sig.exit)))
    return out


def _one_flag_scenario(n: int = 420, flag_bar: int = 300):
    """SPY at a fresh high every bar; the A/D line makes a new high every
    session EXCEPT ``flag_bar`` (a single negative net-advance print), then
    jumps to fresh highs again. Exactly one divergence flag, at flag_bar."""
    closes = _rising(n)
    nar = np.full(n, 0.2)
    nar[flag_bar] = -0.2
    nar[flag_bar + 1:] = 0.5
    frame = _frame(closes)
    return frame, _breadth(frame.index, nar)


# ------------------------------------------------- (1) fires / stays flat

def test_single_flag_forces_exactly_twenty_one_flat_sessions():
    frame, breadth = _one_flag_scenario()
    state = mod.compute_state(frame, breadth)
    flag_idx = np.flatnonzero(state["flag"].to_numpy())
    assert flag_idx.tolist() == [300]
    off = state["off"].to_numpy()
    assert off[300:300 + FLAT].all()
    assert not off[299]
    assert not off[300 + FLAT]
    # position rule: long (entry, no exit) while off is False; flat while True
    assert bool(state["entry"].iloc[299]) and not bool(state["exit"].iloc[299])
    assert not bool(state["entry"].iloc[300]) and bool(state["exit"].iloc[300])
    assert not bool(state["entry"].iloc[320]) and bool(state["exit"].iloc[320])
    assert bool(state["entry"].iloc[321]) and not bool(state["exit"].iloc[321])


def test_no_flag_when_spy_is_not_at_a_six_month_high():
    n = 420
    closes = _rising(n)
    closes[300:] = closes[299] * 0.9  # SPY drops and stays below its high
    nar = np.full(n, 0.2)
    nar[300:] = -0.2  # the A/D line collapses, but SPY is not at a high
    frame = _frame(closes)
    state = mod.compute_state(frame, _breadth(frame.index, nar))
    assert not state["flag"].to_numpy()[300:].any()
    assert state["entry"].to_numpy()[300:].all()


def test_no_flag_when_ad_line_confirms_the_high():
    n = 420
    frame = _frame(_rising(n))
    state = mod.compute_state(frame, _breadth(frame.index, np.full(n, 0.2)))
    assert state["spy_hi"].to_numpy()[HIGH - 1:].all()
    assert state["adl_hi"].to_numpy()[HIGH - 1:].all()
    assert not state["flag"].any()


def test_flag_needs_a_full_ad_line_lookback_before_it_can_fire():
    # A/D line falls from its first defined session: adl_hi would be False on
    # every bar, but the 126-session A/D high is undefined until 126 defined
    # sessions exist, so no divergence can be established before that.
    n = 300
    frame = _frame(_rising(n))
    state = mod.compute_state(frame, _breadth(frame.index, np.full(n, -0.1)))
    flag = state["flag"].to_numpy()
    assert not flag[:HIGH - 1].any()
    assert flag[HIGH - 1:].all()


def test_ad_line_is_the_cumulative_sum_from_the_first_defined_session():
    n = 200
    frame = _frame(_rising(n))
    values = np.linspace(-0.5, 0.5, n - 40)
    breadth = _breadth(frame.index[40:], values)
    state = mod.compute_state(frame, breadth)
    assert state["adl"].iloc[:40].isna().all()
    np.testing.assert_allclose(state["adl"].iloc[40:].to_numpy(),
                               np.cumsum(values))


def test_no_position_before_breadth_is_defined():
    n = 200
    frame = _frame(_rising(n))
    breadth = _breadth(frame.index[40:], np.full(n - 40, 0.2))
    state = mod.compute_state(frame, breadth)
    assert not state["entry"].iloc[:40].any()
    assert not state["exit"].iloc[:40].any()
    assert state["entry"].iloc[40:].all()


# -------------------------------------------------- (2) look-ahead probe

def test_look_ahead_probe_perturbing_the_future_never_changes_the_past():
    frame, breadth = _one_flag_scenario(n=420, flag_bar=300)
    k = 310
    index = frame.index
    fn_a = mod.make_signal_fn("SPY", index, {"SPY": frame}, breadth=breadth)
    before = _signals(fn_a, frame, range(0, k + 1))

    perturbed = frame.copy()
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] *= 1.3
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Open")] *= 1.3
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("High")] *= 1.45
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Low")] *= 1.2
    nar = np.full(len(frame), 0.2)
    nar[300] = -0.2
    nar[301:] = 0.5
    nar[k + 1:] = -0.9  # breadth collapses in the future too
    breadth_b = _breadth(frame.index, nar)
    fn_b = mod.make_signal_fn("SPY", index, {"SPY": perturbed}, breadth=breadth_b)
    after = _signals(fn_b, perturbed, range(0, k + 1))
    assert before == after
    # and the perturbation genuinely changed the future, so the probe bit
    assert _signals(fn_a, frame, range(k + 1, 420)) != _signals(
        fn_b, perturbed, range(k + 1, 420))


def test_holdout_bucket_uses_only_history_through_its_last_bar():
    frame, breadth = _one_flag_scenario(n=420, flag_bar=300)
    bucket = frame.index[250:350]
    fn_full = mod.make_signal_fn("SPY", bucket, {"SPY": frame}, breadth=breadth)
    # a world in which nothing after the bucket's last bar exists at all:
    # neither SPY bars nor breadth observations
    truncated = frame.iloc[:350]
    nar = np.full(len(frame), 0.2)
    nar[300] = -0.2
    nar[301:] = 0.5
    breadth_trunc = _breadth(truncated.index, nar[:350])
    fn_trunc = mod.make_signal_fn("SPY", bucket, {"SPY": truncated},
                                  breadth=breadth_trunc)
    bucket_frame = frame.loc[bucket]
    assert (_signals(fn_full, bucket_frame, range(len(bucket)))
            == _signals(fn_trunc, bucket_frame, range(len(bucket))))
    # the warm-up before the bucket is what makes bar 300 a flag here
    sig = fn_full(bucket_frame.iloc[:51], 50)
    assert bucket[50] == frame.index[300]
    assert sig.exit and not sig.entry


# ------------------------------------------------ (3) end-to-end engine

def test_end_to_end_run_backtest_produces_trades_with_next_open_execution():
    frame, breadth = _one_flag_scenario(n=600, flag_bar=300)
    fn = mod.make_signal_fn("SPY", frame.index, {"SPY": frame}, breadth=breadth)
    trades, in_market = backtest_engine.run_backtest(frame, fn)
    assert len(trades) >= 1
    assert len(trades) == 2
    first, second = trades.iloc[0], trades.iloc[1]
    # long from the first bar the rule is defined (entry at bar 0 -> open of bar 1)
    assert first["entry_date"] == frame.index[1]
    # exit at the close of the flag session itself
    assert first["exit_date"] == frame.index[300]
    assert first["reason"] == "signal_exit"
    # re-entry 21 sessions after the flag: entry signal at 321, fill at 322 open
    assert second["entry_date"] == frame.index[322]
    assert second["reason"] == "end_of_data"
    assert not in_market.iloc[301:322].any()


def test_signal_fn_is_a_constant_time_lookup_and_refuses_misaligned_bars():
    frame, breadth = _one_flag_scenario()
    fn = mod.make_signal_fn("SPY", frame.index, {"SPY": frame}, breadth=breadth)
    # the engine hands a slice ending at bar i together with i
    sig = fn(frame.iloc[:301], 300)
    assert sig.exit
    with pytest.raises(ValueError):
        fn(frame.iloc[:300], 300)  # slice does not end at bar 300


# ----------------------------------------------- fail-closed calendar rules

def test_frame_missing_a_snapshot_session_is_refused():
    n = 300
    frame = _frame(_rising(n))
    breadth = _breadth(frame.index, np.full(n, 0.2))
    holed = frame.drop(frame.index[150])
    with pytest.raises(ValueError, match="one-to-one"):
        mod.make_signal_fn("SPY", holed.index, {"SPY": holed}, breadth=breadth)


def test_frame_starting_after_the_snapshots_first_session_is_refused():
    n = 300
    frame = _frame(_rising(n))
    breadth = _breadth(frame.index, np.full(n, 0.2))
    late = frame.iloc[10:]
    with pytest.raises(ValueError, match="first defined session"):
        mod.make_signal_fn("SPY", late.index, {"SPY": late}, breadth=breadth)


def test_bucket_outside_the_frame_is_refused():
    frame, breadth = _one_flag_scenario(n=350)
    foreign = pd.bdate_range("2030-01-01", periods=10)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", foreign, {"SPY": frame}, breadth=breadth)


def test_only_the_traded_ticker_is_accepted():
    frame, breadth = _one_flag_scenario(n=350)
    with pytest.raises(ValueError):
        mod.make_signal_fn("XLK", frame.index, {"SPY": frame, "XLK": frame},
                           breadth=breadth)
