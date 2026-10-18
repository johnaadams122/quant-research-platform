"""Tests for src/mining/batch5/capex_cycle_blue_chip_dip_pooled.py and its
sibling blue_chip_reversal_no_capex_control.py. Synthetic frames only (no
network) -- TRADED is populated (by the mechanical rule in
scripts/build_capex_watchlist.py), but every strategy-logic
test here still works with hand-built synthetic tickers/frames, not the
real pool -- only the watchlist-identity tests near the bottom of this
file touch the real, populated TRADED list.
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

from mining import backtest_engine  # noqa: E402
from mining.backtest_engine import SignalResult  # noqa: E402
from mining.batch5 import capex_cycle_blue_chip_dip_pooled as mod  # noqa: E402
from mining.batch5 import blue_chip_reversal_no_capex_control as control  # noqa: E402


def _index(n, start="2015-01-05"):
    return pd.bdate_range(start, periods=n)


def _flat_series(index, price=100.0):
    return pd.Series(price, index=index)


def _periodic_fundamentals(index, growth, *, every=200):
    """A disclosure roughly every `every` bars, well inside the 400-
    calendar-day staleness cap (200 business days =~ 280 calendar days)
    -- avoids the fixture going stale mid-test, matching a realistic
    company that files annually."""
    dates_vals = [(index[i], growth) for i in range(0, len(index), every)]
    return _fundamentals(dates_vals)


def _fundamentals(dates_vals):
    """{"usable_asof": ..., "capex_growth_yoy": ...} rows, matching
    edgar_point_in_time.build_point_in_time_annual_series's real output
    shape (only the two columns this module actually reads)."""
    rows = [{"usable_asof": pd.Timestamp(d), "capex_growth_yoy": v,
            "latest_fiscal_year_end": pd.Timestamp(d), "latest_val": 1.0,
            "prior_fiscal_year_end": pd.NaT, "prior_val": np.nan}
           for d, v in dates_vals]
    return pd.DataFrame(rows)


# ------------------------------------------------------ underperformance

def test_underperformance_spread_basic():
    idx = _index(600)
    close = pd.Series(np.linspace(100, 90, 600), index=idx)     # -10%
    bench = pd.Series(np.linspace(100, 110, 600), index=idx)    # +10%
    spread = mod.underperformance_spread(close, bench, window=504)
    assert np.isnan(spread.iloc[503])
    last = spread.iloc[-1]
    assert last < -0.15  # own -10% minus bench's own move is well below -0.15


def test_underperformance_spread_nan_before_window():
    idx = _index(300)
    close = _flat_series(idx)
    spread = mod.underperformance_spread(close, close, window=504)
    assert spread.isna().all()


# ------------------------------------------------------------ capex condition

def test_capex_condition_none_fundamentals_always_false():
    idx = _index(300)
    result = mod.capex_condition_series(idx, None)
    assert not result.any()


def test_capex_condition_true_once_disclosed_above_threshold():
    idx = _index(600)
    fundamentals = _fundamentals([(idx[100], 0.10), (idx[300], 0.40)])
    result = mod.capex_condition_series(idx, fundamentals, threshold=0.30)
    assert not result.iloc[:100].any()          # before any disclosure
    assert not result.iloc[100:300].any()        # disclosed but below threshold
    # Disclosed and above threshold, while still fresh (well inside the
    # 400-calendar-day staleness cap -- ~285 business days from idx[300]).
    assert result.iloc[300:580].all()
    # Well past the staleness cap (idx[599] is ~420 calendar days past
    # idx[300]) -- the stale disclosure correctly ages out to False.
    assert not result.iloc[599]


# --------------------------------------------------------------- decision_dates

def test_no_decisions_before_underperformance_window_full():
    idx = _index(400)
    close = pd.Series(np.linspace(100, 50, 400), index=idx)
    bench = _flat_series(idx)
    fundamentals = _periodic_fundamentals(idx, 0.50)
    entries, exits = mod.decision_dates(close, bench, fundamentals)
    assert len(entries) == 0  # window (504) never completes in 400 bars


def test_both_conditions_true_fires_entry_control_arm_fires_on_underperf_alone():
    idx = _index(900)
    # Sharp decline in the back half -> underperformance kicks in after
    # the 504-session window; capex disclosed above threshold from bar 0.
    close = np.concatenate([np.full(504, 100.0), np.linspace(100, 40, 396)])
    close = pd.Series(close, index=idx)
    bench = _flat_series(idx)
    fundamentals = _periodic_fundamentals(idx, 0.50)

    main_entries, main_exits = mod.decision_dates(close, bench, fundamentals)
    control_entries, control_exits = mod.decision_dates(
        close, bench, fundamentals=None, require_capex=False)

    assert len(main_entries) >= 1
    assert len(control_entries) >= 1
    # Both arms react to the SAME underperformance signal on this fixture
    # (capex is true throughout), so they should fire at the same bars.
    assert list(main_entries) == list(control_entries)


def test_control_arm_ignores_missing_fundamentals():
    idx = _index(900)
    close = np.concatenate([np.full(504, 100.0), np.linspace(100, 40, 396)])
    close = pd.Series(close, index=idx)
    bench = _flat_series(idx)
    entries, exits = mod.decision_dates(
        close, bench, fundamentals=None, require_capex=False)
    assert len(entries) >= 1  # control arm doesn't need fundamentals at all


def test_max_hold_exit_fires_when_signal_never_clears():
    idx = _index(1100)
    close = np.concatenate([np.full(504, 100.0), np.full(596, 40.0)])  # permanent underperformance
    close = pd.Series(close, index=idx)
    bench = _flat_series(idx)
    fundamentals = _periodic_fundamentals(idx, 0.50)
    entries, exits = mod.decision_dates(close, bench, fundamentals)
    assert len(entries) >= 1
    first_exit_held = exits[0] - (entries[0] + 1)
    assert first_exit_held == mod.MAX_HOLD_SESSIONS


def test_signal_exit_fires_before_max_hold_when_condition_clears():
    idx = _index(1100)
    # Sharp dip, then a FAST recovery (well within max_hold_sessions) back
    # above the -0.15 underperformance threshold vs. the flat benchmark.
    close = np.concatenate([
        np.full(504, 100.0), np.linspace(100, 40, 50), np.linspace(40, 95, 60),
        np.full(486, 95.0)])
    close = pd.Series(close, index=idx)
    bench = _flat_series(idx)
    fundamentals = _periodic_fundamentals(idx, 0.50)
    entries, exits = mod.decision_dates(close, bench, fundamentals)
    assert len(entries) >= 1
    first_exit_held = exits[0] - (entries[0] + 1)
    assert first_exit_held < mod.MAX_HOLD_SESSIONS


def test_rejects_mismatched_lengths():
    idx = _index(300)
    close = _flat_series(idx)
    with pytest.raises(ValueError):
        mod.decision_dates(close, close.iloc[:200], None)


def test_empty_input():
    entries, exits = mod.decision_dates(
        pd.Series(dtype=float), pd.Series(dtype=float), None)
    assert len(entries) == 0 and len(exits) == 0


# ------------------------------------------------------------ engine agreement

def _ohlc(index, close):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": close * 1.001, "Low": close * 0.999,
                         "Close": close}, index=index)


def test_end_to_end_matches_real_engine_bar_for_bar():
    idx = _index(900)
    close = np.concatenate([np.full(504, 100.0), np.linspace(100, 40, 396)])
    frame = _ohlc(idx, close)
    bench_frame = _ohlc(idx, np.full(900, 100.0))
    fundamentals = _periodic_fundamentals(idx, 0.50)

    fn = mod.make_signal_fn(
        "X", idx, {"X": frame, "SPY": bench_frame},
        fundamentals={"X": fundamentals})
    trades, _ = backtest_engine.run_backtest(frame, fn)
    assert len(trades) >= 1

    entries_np, exits_np = mod.decision_dates(
        frame["Close"], bench_frame["Close"], fundamentals)
    assert len(entries_np) == len(trades)
    for k in range(len(trades)):
        trade = trades.iloc[k]
        assert trade["entry_date"] == idx[entries_np[k] + 1]
        assert trade["exit_date"] == idx[exits_np[k]]
        assert trade["held"] == exits_np[k] - (entries_np[k] + 1)


def test_control_arm_end_to_end_matches_real_engine():
    idx = _index(900)
    close = np.concatenate([np.full(504, 100.0), np.linspace(100, 40, 396)])
    frame = _ohlc(idx, close)
    bench_frame = _ohlc(idx, np.full(900, 100.0))

    fn = control.make_signal_fn("X", idx, {"X": frame, "SPY": bench_frame})
    trades, _ = backtest_engine.run_backtest(frame, fn)
    assert len(trades) >= 1

    entries_np, exits_np = control.decision_dates(
        frame["Close"], bench_frame["Close"], fundamentals=None, require_capex=False)
    assert len(entries_np) == len(trades)
    for k in range(len(trades)):
        assert trades.iloc[k]["entry_date"] == idx[entries_np[k] + 1]
        assert trades.iloc[k]["exit_date"] == idx[exits_np[k]]


# ------------------------------------------------------------------- contract

def test_control_arm_shares_thresholds_with_main_by_construction():
    assert control.PARAMETERS["underperformance_threshold"] == \
        mod.PARAMETERS["underperformance_threshold"]
    assert control.PARAMETERS["underperformance_window_sessions"] == \
        mod.PARAMETERS["underperformance_window_sessions"]
    assert control.PARAMETERS["max_hold_sessions"] == mod.PARAMETERS["max_hold_sessions"]
    assert control.BENCHMARK == mod.BENCHMARK


def test_control_arm_parameters_inherited_live_not_hand_copied():
    """A NEW non-capex key added to
    the main module's PARAMETERS must reach the control arm automatically
    -- a hand-copied allowlist would silently omit it. Proves inheritance
    by construction, not by re-asserting the currently known keys."""
    non_capex_keys = set(mod.PARAMETERS) - {"capex_growth_threshold", "capex_growth_source"}
    for key in non_capex_keys:
        assert key in control.PARAMETERS
        assert control.PARAMETERS[key] == mod.PARAMETERS[key]
    # And the capex-only keys must NOT leak into the control arm's own
    # metadata (it never uses them).
    assert "capex_growth_threshold" not in control.PARAMETERS
    assert "capex_growth_source" not in control.PARAMETERS


def test_watchlist_is_populated_within_the_target_range():
    # The mechanical selection rule picks N=25 names (see the module's
    # WATCHLIST_SOURCE comment); this allows a small band around it.
    assert 20 <= len(mod.TRADED) <= 30
    assert len(mod.TRADED) == len(set(mod.TRADED))  # no duplicates
    assert all(isinstance(t, str) and t == t.upper() for t in mod.TRADED)


def test_control_arm_watchlist_follows_the_main_module_automatically():
    # control.TRADED = list(_main.TRADED) -- populating the main module
    # alone must be sufficient; the control arm needs no separate edit.
    assert control.TRADED == mod.TRADED
    assert len(control.TRADED) > 0


def test_rejects_missing_frames():
    idx = _index(300)
    frame = _ohlc(idx, np.full(300, 100.0))
    with pytest.raises(KeyError):
        mod.make_signal_fn("X", idx, {"X": frame})  # no SPY frame
    with pytest.raises(KeyError):
        mod.make_signal_fn("X", idx, {"SPY": frame})  # no own frame


def test_lookahead_future_bars_never_change_past_decisions():
    idx = _index(900)
    close = np.concatenate([np.full(504, 100.0), np.linspace(100, 40, 396)])
    frame = _ohlc(idx, close)
    bench_frame = _ohlc(idx, np.full(900, 100.0))
    fundamentals = _periodic_fundamentals(idx, 0.50)
    k = 700
    fn = mod.make_signal_fn(
        "X", idx[:k + 1], {"X": frame, "SPY": bench_frame},
        fundamentals={"X": fundamentals})
    before = [fn(frame.iloc[:t + 1], t) for t in range(k + 1)]

    perturbed = frame.copy()
    perturbed.loc[idx[k + 1:], ["Open", "High", "Low", "Close"]] *= 5.0
    fn2 = mod.make_signal_fn(
        "X", idx[:k + 1], {"X": perturbed, "SPY": bench_frame},
        fundamentals={"X": fundamentals})
    after = [fn2(perturbed.iloc[:t + 1], t) for t in range(k + 1)]
    assert before == after
