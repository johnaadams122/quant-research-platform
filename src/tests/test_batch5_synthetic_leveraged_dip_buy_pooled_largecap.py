"""Tests for src/mining/batch5/synthetic_leveraged_dip_buy_pooled_largecap.py.

Synthetic frames only (no network). Heavy focus on verifying
dip_entry_exit_events's bar accounting against the REAL
backtest_engine.run_backtest (not just trusting the derivation) --
mirrors test_batch5_stock_split_post_split_drift_pooled.py's own
"end-to-end matches engine semantics" pattern.
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

from mining import backtest_engine  # noqa: E402
from mining.backtest_engine import SignalResult  # noqa: E402
from mining.batch5 import (  # noqa: E402
    synthetic_leveraged_dip_buy_pooled_largecap as mod,
)


# ------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == "synthetic_leveraged_dip_buy_pooled_largecap"
    assert mod.SHAPE == "pooled_single_asset"
    assert mod.BENCHMARK == "SPY"
    assert callable(mod.make_signal_fn)
    assert len(mod.TRADED) == len(mod.SIGNAL)
    assert not (set(mod.TRADED) & set(mod.SIGNAL))  # disjoint, per the contract
    # Positional correspondence: TRADED[i] is SIGNAL[i]'s own synthetic id.
    assert mod.TRADED == [mod.synthetic_id_for(t) for t in mod.SIGNAL]
    assert all(mod.underlying_of(t) == u for t, u in zip(mod.TRADED, mod.SIGNAL))


def test_driver_contract_validation():
    import run_batch5_mining as batch5
    spec = batch5.validate_module_contract(mod)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "pooled_single_asset"


CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")


def _candidate() -> dict:
    """This family's entry in the shared catalog."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    matches = [c for c in entries if c.get("family_id") == mod.FAMILY_ID]
    assert len(matches) == 1
    return matches[0]


def test_driver_no_parameter_drift_against_catalog():
    import run_batch5_mining as batch5
    from mining import catalog as catalog_mod
    entries = catalog_mod.load_catalog(batch5.CATALOG_PATH)
    entry = next(e for e in entries if e["family_id"] == mod.FAMILY_ID)
    assert batch5.parameter_drift(mod.PARAMETERS, entry["parameters"]) is None
    assert entry["status"] == "eligible"


def test_underlying_of_and_synthetic_id_for_roundtrip():
    assert mod.synthetic_id_for("TSLA") == "TSLA_2x_synthetic"
    assert mod.underlying_of("TSLA_2x_synthetic") == "TSLA"
    with pytest.raises(ValueError):
        mod.underlying_of("TSLA")  # not a synthetic id


# ------------------------------------------------------ synthetic_ohlc_from_close

def test_synthetic_ohlc_shape_open_equals_prior_close_high_low_equal_close():
    dates = pd.bdate_range("2020-01-02", periods=5)
    close = [100.0, 102.0, 101.0, 105.0, 103.0]
    out = mod.synthetic_ohlc_from_close(dates, close, name="X_2x_synthetic")
    assert list(out.columns) == ["Open", "High", "Low", "Close"]
    assert out["Open"].iloc[0] == pytest.approx(100.0)      # Open_0 == Close_0
    assert out["Open"].iloc[1:].tolist() == pytest.approx(close[:-1])
    assert (out["High"] == out["Close"]).all()
    assert (out["Low"] == out["Close"]).all()
    assert len(out) == 5
    assert out.index.equals(pd.DatetimeIndex(dates, name="Date"))


def test_synthetic_ohlc_empty_input():
    out = mod.synthetic_ohlc_from_close([], [], name="X")
    assert len(out) == 0
    assert list(out.columns) == ["Open", "High", "Low", "Close"]


# --------------------------------------------------- dip_entry_exit_events

def _flat_underlying(n, price=100.0):
    """Underlying that never dips (always at its own high) -- a fixture
    baseline with zero entries, used as the substrate for hand-inserted
    dip windows."""
    return np.full(n, price, dtype=float), np.full(n, price, dtype=float)


def _insert_dip(close, high, start, depth=0.5, width=1):
    """Force a dip at [start, start+width): close drops to `depth` of the
    trailing high (well under the 0.80 threshold) for `width` bars, then
    price recovers to the pre-dip level and stays flat (so a profit
    target can be reached later by directly manipulating synthetic_close
    in the caller, not by this helper)."""
    close = close.copy()
    close[start:start + width] = high[start - 1] * depth
    return close


def test_no_decisions_before_252_sessions_of_history():
    n = 300
    close, high = _flat_underlying(n)
    close = _insert_dip(close, high, 100)  # inside the warm-up window
    syn_open = close.copy()
    syn_close = close.copy()
    entries, exits = mod.dip_entry_exit_events(close, high, syn_open, syn_close)
    assert len(entries) == 0 and len(exits) == 0


def test_dip_then_profit_target_hit_before_time_stop():
    n = 600
    close, high = _flat_underlying(n)
    dip_at = 260  # past the 252-session warm-up
    close = _insert_dip(close, high, dip_at)
    syn_open = close.copy()
    syn_close = close.copy()
    fill_bar = dip_at + 1
    entry_price = syn_open[fill_bar]
    target_bar = fill_bar + 30
    syn_close[target_bar:] = entry_price * 1.5  # target hit here onward
    syn_open[target_bar + 1:] = entry_price * 1.5

    entries, exits = mod.dip_entry_exit_events(close, high, syn_open, syn_close)
    assert len(entries) == 1
    assert entries[0] == dip_at
    assert exits[0] == target_bar
    assert exits[0] - fill_bar < 252


def test_dip_then_time_stop_when_target_never_hit():
    n = 700
    close, high = _flat_underlying(n)
    dip_at = 260  # past the 252-session warm-up
    close = _insert_dip(close, high, dip_at)
    syn_open = close.copy()
    syn_close = np.full(n, 90.0)  # never reaches the profit target
    fill_bar = dip_at + 1

    entries, exits = mod.dip_entry_exit_events(close, high, syn_open, syn_close)
    assert len(entries) == 1
    assert entries[0] == dip_at
    assert exits[0] == fill_bar + 252


def test_chained_reentry_when_dip_persists_at_the_exit_bar():
    """If underlying Close stays depressed forever (persistent decline),
    a time-stop exit is immediately followed by a fresh entry decided AT
    the exit bar, filling the very next bar -- not skipped, not delayed."""
    n = 900
    close, high = _flat_underlying(n)
    dip_at = 50
    close[dip_at:] = high[dip_at - 1] * 0.5  # depressed forever after the dip
    syn_open = close.copy()
    syn_close = np.full(n, 10.0)  # never reaches any profit target

    entries, exits = mod.dip_entry_exit_events(close, high, syn_open, syn_close)
    assert len(entries) >= 2
    # Second entry decided exactly at the first exit bar (chained re-entry).
    assert entries[1] == exits[0]
    assert exits[1] == entries[1] + 1 + 252


def test_no_reentry_when_dip_clears_after_exit():
    """After the position closes, if the underlying has recovered above
    the dip threshold, no new entry fires."""
    n = 600
    close, high = _flat_underlying(n)
    dip_at = 260  # past the 252-session warm-up
    close = _insert_dip(close, high, dip_at)  # single-bar dip, recovers next bar
    syn_open = close.copy()
    syn_close = np.full(n, close[0] * 1.5)  # hits target immediately

    entries, exits = mod.dip_entry_exit_events(close, high, syn_open, syn_close)
    assert len(entries) == 1  # the recovered underlying never re-flags a dip


def test_dangling_entry_at_end_of_history_is_dropped():
    n = 300
    close, high = _flat_underlying(n)
    close = _insert_dip(close, high, n - 1)  # dip on the very last bar
    syn_open = close.copy()
    syn_close = close.copy()
    entries, exits = mod.dip_entry_exit_events(close, high, syn_open, syn_close)
    assert len(entries) == 0  # no bar left to fill it


def test_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError):
        mod.dip_entry_exit_events([1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3])


def test_empty_input():
    entries, exits = mod.dip_entry_exit_events([], [], [], [])
    assert len(entries) == 0 and len(exits) == 0


# ---------------------------------------------- engine agreement (critical)

def _real_frames(n=700, seed=3):
    """A real underlying OHLC frame with a genuine dip-then-recovery
    pattern, and its matching synthetic frame built via the real
    data_pipeline.build_synthetic_leveraged + synthetic_ohlc_from_close
    pipeline -- not a hand-rolled fixture -- so the agreement test
    exercises the actual production path end to end."""
    dates = pd.bdate_range("2015-01-05", periods=n)
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    # Force one clean dip around bar 300 so a trade is guaranteed.
    close[300:303] *= 0.5
    high = close * 1.01
    low = close * 0.99
    open_ = np.concatenate([[close[0]], close[:-1]])
    underlying = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=dates)

    df_for_synth = pd.DataFrame({"Date": dates, "TSLA": close})
    import data_pipeline as dp
    synth_close_df = dp.build_synthetic_leveraged(df_for_synth, "TSLA")
    synthetic = mod.synthetic_ohlc_from_close(
        dates, synth_close_df["TSLA_2x_synthetic"].to_numpy(),
        name="TSLA_2x_synthetic")
    return underlying, synthetic


def test_end_to_end_matches_real_engine_bar_for_bar():
    """The module's own precompute must land on the IDENTICAL entry/exit
    bars, held count, and fill prices that backtest_engine.run_backtest
    itself produces when driven by this module's signal_fn -- proven by
    running the real engine, not by re-deriving the arithmetic by hand."""
    underlying, synthetic = _real_frames()
    frames = {"TSLA_2x_synthetic": synthetic, "TSLA": underlying}
    fn = mod.make_signal_fn("TSLA_2x_synthetic", synthetic.index, frames)
    trades, in_market = backtest_engine.run_backtest(synthetic, fn)

    assert len(trades) >= 1
    entries_np, exits_np = mod.dip_entry_exit_events(
        underlying["Close"].to_numpy(dtype=float),
        underlying["High"].to_numpy(dtype=float),
        synthetic["Open"].to_numpy(dtype=float),
        synthetic["Close"].to_numpy(dtype=float))
    assert len(entries_np) == len(trades)
    for k in range(len(trades)):
        trade = trades.iloc[k]
        expected_entry_date = synthetic.index[entries_np[k] + 1]
        expected_exit_date = synthetic.index[exits_np[k]]
        assert trade["entry_date"] == expected_entry_date
        assert trade["exit_date"] == expected_exit_date
        assert trade["entry_px"] == pytest.approx(
            float(synthetic["Open"].iloc[entries_np[k] + 1]))
        assert trade["exit_px"] == pytest.approx(
            float(synthetic["Close"].iloc[exits_np[k]]))
        assert trade["held"] == exits_np[k] - (entries_np[k] + 1)


def test_end_to_end_time_stop_bar_matches_engines_own_time_stop_semantics():
    """Same pattern as stock_split_post_split_drift_pooled's own
    'matches_engine_time_stop_semantics' test: this module's time-based
    exit bar must be the SAME bar the engine's built-in time_stop=252
    would independently produce (only the reason label differs)."""
    n = 700
    close, high = _flat_underlying(n)
    dip_at = 20
    close[dip_at:] = high[dip_at - 1] * 0.5  # dip, never recovers
    syn_open = close.copy()
    syn_close = np.full(n, 10.0)  # never hits any profit target
    dates = pd.bdate_range("2015-01-05", periods=n)
    underlying = pd.DataFrame(
        {"Open": close, "High": high, "Low": close, "Close": close}, index=dates)
    synthetic = mod.synthetic_ohlc_from_close(dates, syn_close, name="X_2x_synthetic")
    # Override Open on the synthetic frame to match syn_open exactly (the
    # fixture above sets Open=prior Close by construction, which already
    # equals syn_open here since syn_open IS close.copy()).
    frames = {"X_2x_synthetic": synthetic, "X": underlying}
    fn = mod.make_signal_fn("X_2x_synthetic", synthetic.index, frames)
    ours, _ = backtest_engine.run_backtest(synthetic, fn)

    def entry_only(frame_slice, i):
        return SignalResult(entry=fn(frame_slice, i).entry, exit=False)
    engine_ts, _ = backtest_engine.run_backtest(synthetic, entry_only, time_stop=252)

    assert len(ours) >= 1 and len(engine_ts) >= 1
    for column in ("entry_date", "exit_date", "held"):
        assert ours[column].iloc[0] == engine_ts[column].iloc[0]
    assert ours["reason"].iloc[0] == "signal_exit"
    assert engine_ts["reason"].iloc[0] == "time_stop"


# -------------------------------------------------------------- look-ahead

def test_lookahead_future_bars_never_change_past_decisions():
    underlying, synthetic = _real_frames(n=500, seed=11)
    frames = {"TSLA_2x_synthetic": synthetic, "TSLA": underlying}
    k = 320  # well after the forced dip at 300
    fn = mod.make_signal_fn(
        "TSLA_2x_synthetic", synthetic.index[:k + 1], frames)
    before = [fn(synthetic.iloc[:t + 1], t) for t in range(k + 1)]

    perturbed_underlying = underlying.copy()
    perturbed_synthetic = synthetic.copy()
    tail = synthetic.index[k + 1:]
    perturbed_underlying.loc[tail, ["Open", "High", "Low", "Close"]] *= 5.0
    perturbed_synthetic.loc[tail, ["Open", "High", "Low", "Close"]] *= 5.0
    perturbed_frames = {"TSLA_2x_synthetic": perturbed_synthetic,
                        "TSLA": perturbed_underlying}
    fn2 = mod.make_signal_fn(
        "TSLA_2x_synthetic", perturbed_synthetic.index[:k + 1], perturbed_frames)
    after = [fn2(perturbed_synthetic.iloc[:t + 1], t) for t in range(k + 1)]
    assert before == after


# --------------------------------------------------------------- validation

def test_rejects_missing_synthetic_or_underlying_frame():
    dates = pd.bdate_range("2020-01-02", periods=300)
    close = np.full(300, 100.0)
    synthetic = mod.synthetic_ohlc_from_close(dates, close, name="X_2x_synthetic")
    with pytest.raises(KeyError):
        mod.make_signal_fn("X_2x_synthetic", dates, {"X_2x_synthetic": synthetic})
    with pytest.raises(KeyError):
        mod.make_signal_fn("X_2x_synthetic", dates, {"X": synthetic})


def test_rejects_mismatched_calendars_between_synthetic_and_underlying():
    dates = pd.bdate_range("2020-01-02", periods=300)
    close = np.full(300, 100.0)
    synthetic = mod.synthetic_ohlc_from_close(dates, close, name="X_2x_synthetic")
    # Underlying is missing one INTERIOR bar within the bucket window --
    # both frames still cover the same nominal [first, cutoff] span after
    # truncation, so only a genuine calendar mismatch (not merely a
    # shorter pre-truncation length) can trip the guard.
    gap_dates = dates.delete(150)
    gap_close = np.delete(close, 150)
    underlying = pd.DataFrame(
        {"Open": gap_close, "High": gap_close, "Low": gap_close, "Close": gap_close},
        index=gap_dates)
    with pytest.raises(ValueError):
        mod.make_signal_fn(
            "X_2x_synthetic", dates,
            {"X_2x_synthetic": synthetic, "X": underlying})
