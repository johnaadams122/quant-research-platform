"""Tests for src/mining/batch5/convertible_bond_moneyness_regime.py.

Synthetic data only (no network), deterministic, fast. All hand-computed
fixtures below use DYADIC-rational returns (+-50%, +-75%, +-25%) so every
intermediate value (beta, its trailing median, the trailing return) is
EXACTLY representable in IEEE-754 double precision through the whole price
-> return -> beta -> median chain -- no floating-point tolerance is needed
anywhere, including at an exact beta == median boundary, which several
fixtures deliberately land on to exercise the ">= is inclusive" rule.

The PARAMETERS-match test below compares against a Python literal copied
byte-for-byte from the family's FROZEN catalog entry.
"""
import os
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
from mining.batch5 import convertible_bond_moneyness_regime as mod  # noqa: E402

# Copied byte-for-byte from the family's frozen catalog entry (see this
# module's own docstring).
FROZEN_CATALOG_PARAMETERS = {
    "beta_window_days": 63,
    "regime_median_window_days": 504,
    "trailing_return_window_days": 63,
    "rebalance": "daily",
}
FROZEN_CATALOG_TICKERS = ["CWB", "SPY"]
FROZEN_CATALOG_BENCHMARK = "CWB"


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start="2015-01-02", periods=600) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _dyadic_close_series(start: float, factors: list) -> np.ndarray:
    """``start`` followed by ``start`` compounded through each multiplicative
    ``factors`` entry (e.g. 1.5 for +50%, 0.5 for -50%) -- every factor here
    is a dyadic rational, so every intermediate price and every return
    derived from it is EXACTLY representable in double precision (no
    compounding rounding drift), which is what makes the fixtures below
    hand-verifiable to exact equality rather than only approximately."""
    values = [start]
    for factor in factors:
        values.append(values[-1] * factor)
    return np.array(values, dtype=float)


def _frames(index: pd.DatetimeIndex, cwb_close: np.ndarray,
           spy_close: np.ndarray) -> dict:
    return {"CWB": _ohlc_from_close(cwb_close, index),
            "SPY": _ohlc_from_close(spy_close, index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


def _random_close(index: pd.DatetimeIndex, seed: int, mu=0.0003, sigma=0.01,
                  start=100.0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return start * np.exp(np.cumsum(rng.normal(mu, sigma, len(index))))


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == "convertible_bond_moneyness_regime"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["CWB"]
    assert mod.SIGNAL == ["SPY"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "CWB"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_signal_fn)


def test_parameters_match_frozen_catalog_entry():
    import json
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(FROZEN_CATALOG_PARAMETERS, sort_keys=True))
    assert mod.TRADED + mod.SIGNAL == FROZEN_CATALOG_TICKERS
    assert mod.BENCHMARK == FROZEN_CATALOG_BENCHMARK


def test_driver_discovery_validates_this_module():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "single_long"
    assert spec["traded"] == ["CWB"]
    assert spec["signal"] == ["SPY"]


# --------------------------------------------------------- low-level math

def test_rolling_beta_hand_computed_two_point_slope():
    # Two-point windows: beta reduces to the exact secant slope
    # (trd2-trd1)/(sig2-sig1). All dyadic, all exact.
    spy_ret = np.array([np.nan, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    cwb_ret = np.array([np.nan, 0.75, -0.75, 0.75, -0.75, 0.75,
                        -0.25, 0.25, -0.25, 0.25])
    beta = mod.rolling_beta(spy_ret, cwb_ret, window=2)
    expected = np.array([np.nan, np.nan, 1.5, 1.5, 1.5, 1.5, 1.0, 0.5, 0.5, 0.5])
    assert np.array_equal(np.isnan(beta), np.isnan(expected))
    assert np.array_equal(beta[~np.isnan(beta)], expected[~np.isnan(expected)])


def test_rolling_beta_zero_variance_denominator_yields_no_signal():
    # Two identical SPY returns in a window -> zero variance -> guarded,
    # never a manufactured beta from a degenerate denominator.
    spy_ret = np.array([np.nan, 0.5, 0.5])
    cwb_ret = np.array([np.nan, 0.3, -0.2])
    beta = mod.rolling_beta(spy_ret, cwb_ret, window=2)
    assert np.isnan(beta[2])


def test_rolling_beta_window_must_exceed_one():
    with pytest.raises(ValueError):
        mod.rolling_beta(np.array([0.1, 0.2]), np.array([0.1, 0.2]), window=1)


def test_trailing_median_expanding_then_rolling_hand_computed():
    beta = np.array([np.nan, np.nan, 1.5, 1.5, 1.5, 1.5, 1.0, 0.5, 0.5, 0.5])
    median = mod.trailing_median_expanding_then_rolling(beta, window=3)
    expected = np.array([np.nan, np.nan, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.5, 0.5])
    assert np.array_equal(np.isnan(median), np.isnan(expected))
    assert np.array_equal(median[~np.isnan(median)], expected[~np.isnan(expected)])


def test_trailing_return_hand_computed():
    close = np.array([100.0, 110.0, 99.0, 108.9, 98.01])
    ret1 = mod.trailing_return(close, window=1)
    expected1 = np.array([np.nan, 0.10, -0.10, 0.10, -0.10])
    assert np.isnan(ret1[0])
    assert ret1[1:] == pytest.approx(expected1[1:], abs=1e-12)

    ret3 = mod.trailing_return(close, window=3)
    assert np.all(np.isnan(ret3[:3]))
    assert ret3[3] == pytest.approx(108.9 / 100.0 - 1.0, abs=1e-12)
    assert ret3[4] == pytest.approx(98.01 / 110.0 - 1.0, abs=1e-12)


def test_trailing_return_rejects_bad_anchor():
    close = np.array([0.0, 10.0, -5.0, 20.0])
    ret = mod.trailing_return(close, window=1)
    # anchor 0.0 (bar 0) and anchor -5.0 (bar 2) are both invalid anchors.
    assert np.isnan(ret[1])
    assert np.isnan(ret[3])


# ------------------------------------------------------------------- rule

def test_regime_state_and_condition_hand_computed():
    # Constructed so beta cycles 1.5 -> 1.0 (mixed window) -> 0.5, exactly
    # crossing its own trailing median at the beta==median boundary (bars
    # 8-9), while trailing return alternates sign -- exercising all four
    # combinations of the AND (both true, beta-only, ret-only, neither).
    spy = _dyadic_close_series(100.0, [1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    spy_ret = spy[1:] / spy[:-1] - 1.0
    k = [1.5] * 5 + [0.5] * 4  # cwb_ret[t] = k[t-1] * spy_ret[t-1]
    cwb = [50.0]
    for factor, r in zip(k, spy_ret):
        cwb.append(cwb[-1] * (1.0 + factor * r))
    cwb = np.array(cwb, dtype=float)

    index = _calendar(periods=10)
    traded_close = pd.Series(cwb, index=index)
    signal_close = pd.Series(spy, index=index)
    state = mod.regime_state(traded_close, signal_close,
                             beta_window=2, median_window=3, return_window=1)

    # beta:   [nan, nan, 1.5, 1.5, 1.5, 1.5, 1.0, 0.5, 0.5, 0.5]
    # median: [nan, nan, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.5, 0.5]
    # ret:    [nan,0.75,-0.75,0.75,-0.75,0.75,-0.25,0.25,-0.25,0.25]
    # AND:    bar3 (beta>=med & ret>0) True; bar5 True; bar9 True
    #         (bar8: beta==median exactly, 0.5>=0.5, but ret8=-0.25<=0 -> False)
    #         bar6,7: beta<median -> False regardless of ret
    #         bar2,4: beta>=median True but ret<=0 -> False
    expected = np.array([False, False, False, True, False,
                         True, False, False, False, True])
    assert np.array_equal(state.to_numpy(), expected)


def test_beta_equal_median_boundary_is_inclusive_and_zero_return_is_not_positive():
    # A degenerate constant-beta series (cwb tracks spy 1:1) so beta always
    # equals its own median exactly (>= holds by equality at every valid
    # bar); the final bar's trailing return is EXACTLY zero, which must
    # still fail the strict ">0" test (catalog: "positive", never "zero").
    spy = _dyadic_close_series(100.0, [1.5, 0.5, 1.5, 1.0])  # last move flat
    cwb = spy / 2.0  # k=1 exactly throughout -> beta==median always
    index = _calendar(periods=5)
    traded_close = pd.Series(cwb, index=index)
    signal_close = pd.Series(spy, index=index)
    state = mod.regime_state(traded_close, signal_close,
                             beta_window=2, median_window=2, return_window=1)
    # bar2: beta==median (equality, inclusive) but ret2=-0.5 -> False
    # bar3: beta==median, ret3=+0.5 -> True
    # bar4 (FINAL bar of this bucket): beta==median (equality) but ret4==0.0
    #       exactly -> NOT positive -> False. Also proves the final bar of
    #       the bucket DOES make a real decision (interpretation choice 7),
    #       unlike the batch5 month-end modules.
    expected = np.array([False, False, False, True, False])
    assert np.array_equal(state.to_numpy(), expected)


def test_non_finite_close_poisons_only_its_own_windows():
    # Same construction as test_regime_state_and_condition_hand_computed,
    # but with a single bad print injected at CWB bar 5 -- it must poison
    # exactly the beta windows that touch that bar's return (bars 5, 6, 7)
    # and leave every other bar's decision byte-identical to the clean run,
    # including full recovery once the poisoned window rolls past (bars 8-9).
    spy = _dyadic_close_series(100.0, [1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    spy_ret = spy[1:] / spy[:-1] - 1.0
    k = [1.5] * 5 + [0.5] * 4
    cwb = [50.0]
    for factor, r in zip(k, spy_ret):
        cwb.append(cwb[-1] * (1.0 + factor * r))
    cwb = np.array(cwb, dtype=float)
    cwb[5] = np.nan  # a bad print, not a missing row

    index = _calendar(periods=10)
    traded_close = pd.Series(cwb, index=index)
    signal_close = pd.Series(spy, index=index)
    state = mod.regime_state(traded_close, signal_close,
                             beta_window=2, median_window=3, return_window=1)
    expected = np.array([False, False, False, True, False,
                         False, False, False, False, True])
    assert np.array_equal(state.to_numpy(), expected)
    # Only bar 5 differs from the clean fixture (True -> False); bars 8-9
    # recovered even though bar 6/7's beta stayed poisoned.
    clean_expected = np.array([False, False, False, True, False,
                               True, False, False, False, True])
    diff = np.flatnonzero(state.to_numpy() != clean_expected)
    assert list(diff) == [5]


def test_short_history_produces_no_signal_and_no_crash():
    # Fewer bars than beta_window_days (63): the beta series never warms
    # up, so is_long is False (cash/undefined) at every bar -- entry must
    # never fire (exit[t] = not is_long[t] is True throughout, per the
    # entry/exit mapping, but the engine only reads exit while a position
    # is open, and one is never opened here) -- must not raise.
    index = _calendar(periods=30)
    frames = _frames(index, _random_close(index, seed=1, start=50.0),
                     _random_close(index, seed=2, start=100.0))
    signal_fn = mod.make_signal_fn("CWB", index, frames)
    sigs = _signals(signal_fn, frames["CWB"])
    assert all(not s.entry for s in sigs)
    assert all(s.exit for s in sigs)
    trades, in_market = run_backtest(frames["CWB"], signal_fn, cost_bps=0.0)
    assert len(trades) == 0
    assert not in_market.any()


def test_empty_and_missing_frames_raise():
    index = _calendar(periods=10)
    frames = _frames(index, _random_close(index, seed=3, start=50.0),
                     _random_close(index, seed=4, start=100.0))
    with pytest.raises(ValueError):
        mod.make_signal_fn("CWB", index, {"CWB": frames["CWB"].iloc[0:0],
                                          "SPY": frames["SPY"]})
    with pytest.raises(ValueError):
        mod.make_signal_fn("CWB", index, {"SPY": frames["SPY"]})
    with pytest.raises(ValueError):
        mod.make_signal_fn("CWB", pd.DatetimeIndex([]), frames)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    frames = _frames(index, _random_close(index, seed=5, start=50.0),
                     _random_close(index, seed=6, start=100.0))
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPY", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("CWB", index, {"CWB": frames["CWB"]})


# -------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    index = _calendar(periods=900)
    cwb = _random_close(index, seed=11, start=50.0)
    spy = _random_close(index, seed=12, start=100.0)
    frames = _frames(index, cwb, spy)
    k = 600
    base_fn = mod.make_signal_fn("CWB", index, frames)
    base = _signals(base_fn, frames["CWB"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.6
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.6
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("CWB", index, perturbed)
    after = _signals(pert_fn, perturbed["CWB"], upto=k)
    assert base == after
    # The perturbation is real: SOME later signal differs.
    assert (_signals(base_fn, frames["CWB"])
            != _signals(pert_fn, perturbed["CWB"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    index = _calendar(periods=900)
    cwb = _random_close(index, seed=21, start=50.0)
    spy = _random_close(index, seed=22, start=100.0)
    frames = _frames(index, cwb, spy)
    bucket = index[:700]
    fn_full = mod.make_signal_fn("CWB", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("CWB", bucket, truncated)
    bucket_frame = frames["CWB"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


def test_warm_up_history_before_the_bucket_is_used():
    # A holdout bucket starting well after bar 0 must still see the prior
    # 63/504-bar history (which lies BEFORE the bucket) to decide at its
    # very first bar -- a bucket-only signal would still be warming up.
    index = _calendar(periods=900)
    cwb = _random_close(index, seed=31, start=50.0)
    spy = _random_close(index, seed=32, start=100.0)
    frames = _frames(index, cwb, spy)
    full_fn = mod.make_signal_fn("CWB", index, frames)
    full_sigs = _signals(full_fn, frames["CWB"])

    start = 650
    bucket = index[start:]
    bucket_fn = mod.make_signal_fn("CWB", bucket, frames)
    bucket_frame = frames["CWB"].loc[bucket]
    bucket_sigs = _signals(bucket_fn, bucket_frame)
    # The bucket's own signals must match the full-history run's signals
    # over the same dates -- proving the 63/504-bar warm-up before `start`
    # was actually used, not silently restarted at the bucket boundary.
    assert bucket_sigs == full_sigs[start:]
    # And the warm-up is genuinely load-bearing: at least one bucket bar is
    # long (impossible if history before the bucket were discarded, since
    # neither beta nor the 504-day median could ever be defined that early
    # relative to the bucket's own start).
    assert any(s.entry for s in bucket_sigs)


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_trades():
    index = _calendar(periods=1400)
    cwb = _random_close(index, seed=41, start=50.0, mu=0.0004, sigma=0.012)
    spy = _random_close(index, seed=42, start=100.0, mu=0.0003, sigma=0.010)
    frames = _frames(index, cwb, spy)
    signal_fn = mod.make_signal_fn("CWB", index, frames)
    trades, in_market = run_backtest(frames["CWB"], signal_fn, cost_bps=1.0)
    assert len(trades) >= 1
    assert in_market.any()
    # Every entry fills strictly after the bar that raised the entry flag
    # (next-open execution) and every trade is held at least one bar.
    assert (trades["held"] >= 0).all()
    assert (trades["exit_date"] >= trades["entry_date"]).all()


def test_signal_fn_is_constant_time_per_call():
    import time
    index = _calendar(start="2010-01-04", periods=4500)
    cwb = _random_close(index, seed=51, start=50.0)
    spy = _random_close(index, seed=52, start=100.0)
    frames = _frames(index, cwb, spy)
    signal_fn = mod.make_signal_fn("CWB", index, frames)
    frame = frames["CWB"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0


def test_signal_fn_rejects_bar_position_mismatch():
    index = _calendar(periods=100)
    frames = _frames(index, _random_close(index, seed=61, start=50.0),
                     _random_close(index, seed=62, start=100.0))
    signal_fn = mod.make_signal_fn("CWB", index, frames)
    with pytest.raises(IndexError):
        signal_fn(frames["CWB"], len(index))
    with pytest.raises(IndexError):
        signal_fn(frames["CWB"], -1)
    with pytest.raises(ValueError):
        # Bar 5's own data but claiming to be bar position 6.
        signal_fn(frames["CWB"].iloc[:6], 6)
