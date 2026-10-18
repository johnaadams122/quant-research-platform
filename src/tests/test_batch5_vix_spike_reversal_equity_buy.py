"""Tests for src/mining/batch5/vix_spike_reversal_equity_buy.py.

Mostly synthetic frames (no network) for the contract/spike-detection/
reentry-blocking/look-ahead checks, mirroring test_batch5_betting_against_
beta_low_vol.py's and test_batch5_vix_term_structure_carry.py's structure.
This family's PARAMETERS are compared against the REAL pre-registered
catalog entry (data/mining/catalog.json), matching those two modules' own
precedent. Real-data (network) end-to-end checks are not part of this file.
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

from mining.backtest_engine import SignalResult, run_backtest  # noqa: E402
from mining.batch5 import vix_spike_reversal_equity_buy as mod  # noqa: E402

FAMILY_ID = "vix_spike_reversal_equity_buy"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
TRADED_TICKER, SIGNAL_TICKER = "SPY", "^VIX"


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2015-01-02", periods=100) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _spy_close(index: pd.DatetimeIndex, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))


def _frames(index: pd.DatetimeIndex, spy_close: np.ndarray,
           vix_close: np.ndarray) -> dict:
    return {"SPY": _ohlc_from_close(spy_close, index),
            "^VIX": _ohlc_from_close(vix_close, index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SPY"]
    assert mod.SIGNAL == ["^VIX"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 0
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_the_frozen_catalog_entry():
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == mod.FAMILY_ID]
    assert len(matches) == 1
    entry = matches[0]
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert mod.TRADED == [entry["tickers"][0]] == [TRADED_TICKER]
    assert SIGNAL_TICKER in entry["tickers"]
    assert mod.SIGNAL == [SIGNAL_TICKER]


def test_module_passes_the_driver_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module (other modules in the real
    batch5 dir must not affect the assertion, so it is not asserted on)."""
    import run_batch5_mining as batch5
    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["shape"] == "single_long"
    assert spec["traded"] == [TRADED_TICKER]
    assert spec["signal"] == [SIGNAL_TICKER]
    assert spec["benchmark"] == "SPY"
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    catalog_entry = next(c for c in catalog if c["family_id"] == mod.FAMILY_ID)
    assert batch5.parameter_drift(
        spec["parameters"], catalog_entry["parameters"]) is None


def test_generic_abi1_dispatch_signature_is_usable():
    # run_abi1_mining.py's _build_curve calls make_signal_fn(symbol,
    # full_frame.index, module_frames) POSITIONALLY with no extra kwargs --
    # this module must be directly callable that way (^VIX is a normal
    # data_pipeline.fetch_ohlc-fetchable ticker, so this family needs no
    # bespoke external-data driver).
    index = _calendar(periods=60)
    frames = _frames(index, _spy_close(index), np.full(60, 15.0))
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    assert callable(signal_fn)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=30)
    frames = _frames(index, _spy_close(index), np.full(30, 15.0))
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, {"SPY": frames["SPY"]})


def test_factory_rejects_empty_or_unsorted_or_duplicate_index():
    index = _calendar(periods=15)
    frames = _frames(index, _spy_close(index), np.full(15, 15.0))
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", pd.DatetimeIndex([]), frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index[::-1], frames)
    dup_index = index.append(pd.DatetimeIndex([index[0]]))
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", dup_index, frames)


# --------------------------------------------------- vix_spike_flags (pure)

def test_spike_flags_are_false_during_warm_up():
    # First ma_window-1 entries have no defined rolling MA -- even a huge
    # value there must not read as a spike.
    values = pd.Series([100.0] + [15.0] * 8, index=_calendar(periods=9))
    flags = mod.vix_spike_flags(values, ma_window=10, threshold_mult=1.5)
    assert not flags.any()


def test_spike_flags_boundary_is_inclusive():
    # ma_window=3: two values of 10 warm up the window; the third value
    # is chosen so the ratio to the 3-value trailing mean lands exactly
    # at 1.5, just under, and just over.
    index = _calendar(periods=3)

    def _ratio_at_boundary(mult):
        # mean = (10 + 10 + C) / 3; want C / mean == mult
        # => C = mult * (20 + C) / 3 => 3C = 20*mult + mult*C
        # => C(3 - mult) = 20*mult => C = 20*mult / (3 - mult)
        return 20.0 * mult / (3.0 - mult)

    exact = pd.Series([10.0, 10.0, _ratio_at_boundary(1.5)], index=index)
    flags = mod.vix_spike_flags(exact, ma_window=3, threshold_mult=1.5)
    assert bool(flags.iloc[2]) is True  # exactly at threshold -> inclusive

    below = pd.Series([10.0, 10.0, _ratio_at_boundary(1.5) - 1e-6], index=index)
    flags = mod.vix_spike_flags(below, ma_window=3, threshold_mult=1.5)
    assert bool(flags.iloc[2]) is False

    above = pd.Series([10.0, 10.0, _ratio_at_boundary(1.5) + 1e-6], index=index)
    flags = mod.vix_spike_flags(above, ma_window=3, threshold_mult=1.5)
    assert bool(flags.iloc[2]) is True


def test_spike_flags_true_when_close_meaningfully_above_trailing_mean():
    index = _calendar(periods=11)
    values = pd.Series([20.0] * 10 + [35.0], index=index)
    flags = mod.vix_spike_flags(values, ma_window=10, threshold_mult=1.5)
    # MA at the 11th bar = mean(bars 2..11, i.e. the trailing 10) =
    # mean([20]*9 + [35]) = 21.5; 35 >= 1.5*21.5 = 32.25 -> True.
    assert bool(flags.iloc[10]) is True
    assert not flags.iloc[:10].any()


def test_spike_flags_non_finite_input_never_spikes():
    index = _calendar(periods=11)
    values = pd.Series([20.0] * 10 + [np.nan], index=index)
    flags = mod.vix_spike_flags(values, ma_window=10, threshold_mult=1.5)
    assert bool(flags.iloc[10]) is False


# --------------------------------------------- entry_exit_positions (pure)

def test_entry_exit_positions_single_spike_fills_next_bar_holds_exactly_n():
    spike = np.zeros(30, dtype=bool)
    spike[5] = True
    entries, exits = mod.entry_exit_positions(spike, hold_days=10)
    assert list(entries) == [5]
    # fill = 6, exit = fill + 10 = 16
    assert list(exits) == [16]


def test_entry_exit_positions_blocks_reentry_while_held():
    spike = np.zeros(40, dtype=bool)
    spike[5] = True   # entry -> fill 6, exit 16
    spike[10] = True  # inside the hold window [6, 16] -> must be ignored
    spike[16] = True  # exactly on the exit bar -> must ALSO be ignored
    spike[17] = True  # the very next bar -> flat again, must be a new entry
    entries, exits = mod.entry_exit_positions(spike, hold_days=10)
    assert list(entries) == [5, 17]
    assert list(exits) == [16, 28]


def test_entry_exit_positions_multiple_independent_spikes():
    spike = np.zeros(60, dtype=bool)
    spike[2] = True    # fill 3, exit 13
    spike[20] = True   # well after prior hold closed -> fill 21, exit 31
    spike[45] = True   # fill 46, exit 56
    entries, exits = mod.entry_exit_positions(spike, hold_days=10)
    assert list(entries) == [2, 20, 45]
    assert list(exits) == [13, 31, 56]


def test_entry_exit_positions_open_at_bucket_end_reports_no_exit():
    n = 20
    spike = np.zeros(n, dtype=bool)
    spike[12] = True  # fill 13, would exit at 23 >= n -> no exit reported
    entries, exits = mod.entry_exit_positions(spike, hold_days=10)
    assert list(entries) == [12]
    assert list(exits) == []


def test_entry_exit_positions_spike_too_close_to_bucket_end_is_not_entered():
    n = 20
    spike = np.zeros(n, dtype=bool)
    spike[18] = True  # fill = 19 = n-1 -> degenerate fill, no entry at all
    spike[19] = True  # fill = 20 >= n -> no fill bar exists either
    entries, exits = mod.entry_exit_positions(spike, hold_days=10)
    assert list(entries) == []
    assert list(exits) == []


def test_entry_exit_positions_rejects_non_positive_hold():
    with pytest.raises(ValueError):
        mod.entry_exit_positions(np.zeros(5, dtype=bool), hold_days=0)


# ------------------------------------------------------- integration (rule)

def test_make_signal_fn_reproduces_entry_exit_positions_on_synthetic_data():
    # Build a VIX series with exactly one clean, isolated spike, and
    # confirm the resulting signal_fn's entry/exit calls match
    # entry_exit_positions's own precomputed positions directly.
    index = _calendar(periods=60)
    vix_vals = np.full(60, 15.0)
    vix_vals[25] = 40.0  # a real spike relative to the flat 15.0 trailing MA
    spy_close = _spy_close(index)
    frames = _frames(index, spy_close, vix_vals)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    sigs = _signals(signal_fn, frames["SPY"])

    vix_close = mod._clean_close(frames["^VIX"], mod._naive_index(index)[-1])
    spike_full = mod.vix_spike_flags(vix_close, mod._MA_WINDOW_DAYS,
                                     mod._SPIKE_THRESHOLD_MULT)
    spike_on_bucket = spike_full.reindex(mod._naive_index(index)).fillna(False).to_numpy(bool)
    entries, exits = mod.entry_exit_positions(spike_on_bucket, mod._HOLD_DAYS)

    got_entries = [i for i, s in enumerate(sigs) if s.entry]
    got_exits = [i for i, s in enumerate(sigs) if s.exit]
    assert got_entries == list(entries)
    assert got_exits == list(exits)
    assert len(entries) == 1
    assert len(exits) == 1


def test_spy_prices_never_enter_the_signal():
    index = _calendar(periods=60)
    vix_vals = np.full(60, 15.0)
    vix_vals[25] = 40.0
    frames_a = _frames(index, _spy_close(index, seed=1), vix_vals)
    frames_b = _frames(index, _spy_close(index, seed=99), vix_vals)
    fn_a = mod.make_signal_fn("SPY", index, frames_a)
    fn_b = mod.make_signal_fn("SPY", index, frames_b)
    assert (_signals(fn_a, frames_a["SPY"])
            == _signals(fn_b, frames_b["SPY"]))


# ------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=80)
    rng = np.random.RandomState(3)
    vix_vals = 15.0 + rng.normal(0.0, 0.5, 80)
    vix_vals[40] = 45.0  # a real spike inside the probed window
    spy_close = _spy_close(index)
    frames = _frames(index, spy_close, vix_vals)
    k = 60
    base_fn = mod.make_signal_fn("SPY", index, frames)
    base = _signals(base_fn, frames["SPY"], upto=k)

    # Perturb ^VIX (and, for realism, SPY too, though SPY's own prices are
    # never read by the signal) strictly AFTER the cutoff. No spike was
    # seeded post-cutoff in the base series, so a large multiplicative
    # bump there pushes at least one post-cutoff bar's VIX close well
    # past 1.5x its own trailing MA -- a real, verifiable change to the
    # post-cutoff DECISION (not just to prices that are never read), which
    # is what actually probes look-ahead.
    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 3.0
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 3.0
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 3.2
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 2.8
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("SPY", index, perturbed)
    after = _signals(pert_fn, perturbed["SPY"], upto=k)
    assert base == after
    # The perturbation is real: SOME later signal differs (the post-cutoff
    # ^VIX bump creates a fresh spike that was not there before).
    assert (_signals(base_fn, frames["SPY"])
            != _signals(pert_fn, perturbed["SPY"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=80)
    rng = np.random.RandomState(7)
    vix_vals = 15.0 + rng.normal(0.0, 0.5, 80)
    vix_vals[30] = 45.0
    vix_vals[65] = 45.0
    frames = _frames(index, _spy_close(index), vix_vals)
    bucket = index[:60]
    fn_full = mod.make_signal_fn("SPY", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("SPY", bucket, truncated)
    bucket_frame = frames["SPY"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


# --------------------------------------------------- contract guards (calls)

def test_signal_fn_rejects_out_of_range_position():
    index = _calendar(periods=15)
    frames = _frames(index, _spy_close(index), np.full(15, 15.0))
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(IndexError):
        signal_fn(frames["SPY"], len(index))
    with pytest.raises(IndexError):
        signal_fn(frames["SPY"], -1)


def test_signal_fn_rejects_mismatched_bar_date():
    index = _calendar(periods=15)
    frames = _frames(index, _spy_close(index), np.full(15, 15.0))
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(ValueError):
        signal_fn(frames["SPY"].iloc[:1], 3)


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_trades():
    index = _calendar(periods=150)
    vix_vals = np.full(150, 15.0)
    vix_vals[20] = 40.0
    vix_vals[70] = 40.0
    vix_vals[110] = 40.0
    spy_close = _spy_close(index)
    frames = _frames(index, spy_close, vix_vals)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    trades, in_market = run_backtest(frames["SPY"], signal_fn, cost_bps=1.0)
    assert len(trades) == 3
    assert (trades["held"] == 10).all()
    assert set(trades["reason"]) == {"signal_exit"}
    assert in_market.any()


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2006-01-03", periods=3000)
    rng = np.random.RandomState(11)
    vix_vals = 15.0 + rng.normal(0.0, 0.5, 3000)
    spy_close = _spy_close(index)
    frames = _frames(index, spy_close, vix_vals)
    signal_fn = mod.make_signal_fn("SPY", index, frames)
    frame = frames["SPY"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
