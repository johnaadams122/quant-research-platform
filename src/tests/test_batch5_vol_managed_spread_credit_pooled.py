"""Tests for src/mining/batch5/vol_managed_spread_credit_pooled.py.

Synthetic data only, deterministic, no network. The rule is checked
against an independent NAIVE reference implementation (explicit loops over
the trailing 21 log returns and the full expanding history), so the module's
vectorised precomputation is graded against the prose, not against itself.
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
from mining.batch5 import vol_managed_spread_credit_pooled as mod  # noqa: E402

MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           "vol_managed_spread_credit_pooled.py")

RV_LOOKBACK = 21
MIN_OBS = 252


# ------------------------------------------------------------ synthetic data

def _regime_frame(n_bars: int = 900, *, seed: int = 7,
                  block_months: int = 3) -> pd.DataFrame:
    """Business-day OHLC whose daily volatility alternates between a LOW
    and a HIGH regime every ``block_months`` calendar months, so the
    expanding median sits between the two regimes and month-end decisions
    genuinely switch. Positive drift keeps every price positive."""
    index = pd.bdate_range("2015-01-02", periods=n_bars)
    rng = np.random.RandomState(seed)
    month_number = (index.year - index.year[0]) * 12 + index.month
    high_regime = ((month_number // block_months) % 2) == 1
    sigma = np.where(high_regime, 0.030, 0.004)
    steps = rng.normal(0.0002, 1.0, n_bars) * sigma
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _reference_signals(frame: pd.DataFrame, index: pd.DatetimeIndex):
    """Naive, loop-based transcription of the candidate's rule_pseudocode.

    RV_t = sum of the 21 squared log returns ending at t (inclusive);
    MED_t = median of every RV_u with u <= t, undefined until 252 RV values
    exist; LOW_t = RV_t <= MED_t; decide only on the last trading day of
    each month (the last bar of ``index`` is never a decision bar because
    its t+1 open does not exist inside the bucket)."""
    closes = frame["Close"].to_numpy(dtype=float)
    dates = list(frame.index)
    log_ret = {}
    for pos in range(1, len(closes)):
        log_ret[dates[pos]] = np.log(closes[pos] / closes[pos - 1])
    rv = {}
    for pos in range(len(closes)):
        window = [log_ret.get(dates[k]) for k in range(pos - RV_LOOKBACK + 1, pos + 1)
                  if k >= 1]
        if len(window) == RV_LOOKBACK:
            rv[dates[pos]] = float(sum(r * r for r in window))
    med = {}
    history = []
    for pos in range(len(closes)):
        if dates[pos] in rv:
            history.append(rv[dates[pos]])
            if len(history) >= MIN_OBS:
                med[dates[pos]] = float(np.median(history))
    out = []
    n = len(index)
    for i in range(n):
        today = index[i]
        is_decision = (i < n - 1 and (index[i + 1].year, index[i + 1].month)
                       != (today.year, today.month))
        if not is_decision:
            out.append(SignalResult())
            continue
        if today in med and today in rv:
            low = rv[today] <= med[today]
            out.append(SignalResult(entry=low, exit=not low))
        else:
            out.append(SignalResult(entry=False, exit=True))
    return out


def _drive(signal_fn, frame: pd.DataFrame, upto: int | None = None):
    """Call the signal exactly the way backtest_engine does: a slice ending
    at bar i plus i."""
    n = len(frame) if upto is None else upto
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


# ------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == "vol_managed_spread_credit_pooled"
    assert mod.SHAPE == "pooled_single_asset"
    assert mod.TRADED == ["HYG", "EMB"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "HYG"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES == 50
    assert callable(mod.make_signal_fn)
    assert not hasattr(mod, "make_target_fn")


def test_module_passes_the_driver_contract_validation():
    import run_batch5_mining as driver
    module = driver._import_module_file("vol_managed_spread_credit_pooled",
                                        MODULE_PATH)
    spec = driver.validate_module_contract(module, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "pooled_single_asset"
    assert spec["traded"] == ["HYG", "EMB"]
    assert spec["snapshots"] == {}


# --------------------------------------------------------------- the rule

@pytest.mark.parametrize("ticker", ["HYG", "EMB"])
def test_rule_matches_naive_reference_on_every_bar(ticker):
    frame = _regime_frame()
    other = "EMB" if ticker == "HYG" else "HYG"
    frames = {ticker: frame, other: _regime_frame(seed=99)}
    signal_fn = mod.make_signal_fn(ticker, frame.index, frames)
    got = _drive(signal_fn, frame)
    expected = _reference_signals(frame, frame.index)
    assert got == expected
    # The synthetic regimes must actually exercise both branches, otherwise
    # the equality above proves nothing.
    entries = sum(1 for s in got if s.entry)
    exits = sum(1 for s in got if s.exit)
    assert entries >= 3 and exits >= 3


def test_no_position_before_252_rv_observations_and_never_intra_month():
    frame = _regime_frame()
    signal_fn = mod.make_signal_fn("HYG", frame.index, {"HYG": frame})
    got = _drive(signal_fn, frame)
    # first RV needs 21 returns (bar 21), the 252nd RV is at bar 21+251=272
    first_defined = RV_LOOKBACK + MIN_OBS - 1
    for i in range(first_defined):
        assert got[i].entry is False, i
    periods = frame.index.to_period("M")
    for i in range(len(frame) - 1):
        if periods[i] == periods[i + 1]:
            assert got[i] == SignalResult(), i
    # the last bar of the bucket is never a decision bar
    assert got[-1] == SignalResult()


def test_fires_long_on_low_variance_month_end_and_flat_on_high():
    """Hand-built: a moderate-vol warm-up year (which dominates the
    expanding median), then a very quiet half-year, then one violently
    noisy month, then quiet again. Quiet month-ends after warm-up must say
    LONG (entry, no exit); the noisy month-end must say FLAT (exit, no
    entry)."""
    index = pd.bdate_range("2015-01-02", periods=470)  # through ~Oct 2016
    rng = np.random.RandomState(3)
    sigma = np.where(index >= pd.Timestamp("2016-01-01"), 0.002, 0.010)
    sigma = np.where(index >= pd.Timestamp("2016-07-01"), 0.05, sigma)
    sigma = np.where(index >= pd.Timestamp("2016-08-01"), 0.002, sigma)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1.0, len(index)) * sigma))
    frame = pd.DataFrame({"Open": close, "High": close * 1.001,
                          "Low": close * 0.999, "Close": close}, index=index)
    signal_fn = mod.make_signal_fn("EMB", frame.index, {"EMB": frame})
    got = _drive(signal_fn, frame)
    periods = frame.index.to_period("M")
    month_end_pos = {}
    for i in range(len(frame) - 1):
        if periods[i] != periods[i + 1]:
            month_end_pos[str(periods[i])] = i
    # warm-up: 21 + 252 - 1 = 272 bars -> defined from 2016-01 onward
    assert got[month_end_pos["2016-03"]] == SignalResult(entry=True, exit=False)
    assert got[month_end_pos["2016-06"]] == SignalResult(entry=True, exit=False)
    assert got[month_end_pos["2016-07"]] == SignalResult(entry=False, exit=True)
    # the 21-day window still straddles July on Aug 31? No: Aug has >21
    # sessions, so by the Aug month-end the window is quiet again -> LONG.
    assert got[month_end_pos["2016-08"]] == SignalResult(entry=True, exit=False)


# ------------------------------------------------------------- look-ahead

def test_lookahead_probe_future_bars_do_not_change_past_signals():
    frame = _regime_frame()
    k = 600
    signal_fn = mod.make_signal_fn("HYG", frame.index, {"HYG": frame})
    before = _drive(signal_fn, frame, upto=k + 1)

    perturbed = frame.copy()
    future = perturbed.index > frame.index[k]
    # x1.3 on every OTHER future bar: a uniform x1.3 would move exactly one
    # log return and leave the variance path (hence the probe) untouched.
    positions = np.flatnonzero(future)
    odd = positions[::2]
    perturbed.iloc[odd, perturbed.columns.get_loc("Close")] *= 1.3
    perturbed.iloc[odd, perturbed.columns.get_loc("Open")] *= 1.3
    perturbed.loc[future, "High"] = perturbed.loc[future, "High"] * 1.5
    perturbed.loc[future, "Low"] = perturbed.loc[future, "Low"] * 0.7
    signal_fn2 = mod.make_signal_fn("HYG", perturbed.index, {"HYG": perturbed})
    after = _drive(signal_fn2, perturbed, upto=k + 1)
    assert before == after
    # and the perturbation must actually change something later, else the
    # probe is vacuous
    full_before = _drive(signal_fn, frame)
    full_after = _drive(signal_fn2, perturbed)
    assert full_before != full_after


def test_holdout_bucket_uses_prebucket_history_for_warmup_only():
    """When ``index`` is a tail bucket the factory warms up on the frame's
    earlier history (signals are defined from the bucket's first
    month-end), yet depends on nothing after the bucket's last bar."""
    frame = _regime_frame()
    bucket = frame.index[700:820]
    signal_fn = mod.make_signal_fn("HYG", bucket, {"HYG": frame})
    got = _drive(signal_fn, frame.loc[bucket])
    decisions = [s for s in got if s.entry or s.exit]
    assert len(decisions) >= 3  # defined immediately, thanks to warm-up
    # truncate the frame right after the bucket -> identical signals
    truncated = frame.loc[:bucket[-1]]
    signal_fn_t = mod.make_signal_fn("HYG", bucket, {"HYG": truncated})
    assert _drive(signal_fn_t, truncated.loc[bucket]) == got
    # perturb everything after the bucket -> identical signals
    perturbed = frame.copy()
    later = perturbed.index > bucket[-1]
    perturbed.loc[later, ["Open", "High", "Low", "Close"]] *= 2.0
    signal_fn_p = mod.make_signal_fn("HYG", bucket, {"HYG": perturbed})
    assert _drive(signal_fn_p, perturbed.loc[bucket]) == got


# ------------------------------------------------------------ performance

def test_signal_is_precomputed_not_recomputed_from_the_slice():
    """The per-bar call must be a lookup: handing it a ONE-ROW slice ending
    at bar i gives the same answer as the full growing slice, which is only
    possible if nothing is recomputed from the slice."""
    frame = _regime_frame()
    signal_fn = mod.make_signal_fn("HYG", frame.index, {"HYG": frame})
    full = _drive(signal_fn, frame)
    one_row = [signal_fn(frame.iloc[i:i + 1], i) for i in range(len(frame))]
    assert full == one_row


def test_unknown_date_is_flat():
    frame = _regime_frame()
    signal_fn = mod.make_signal_fn("HYG", frame.index, {"HYG": frame})
    alien = frame.iloc[:5].copy()
    alien.index = alien.index + pd.Timedelta(days=20000)
    assert signal_fn(alien, 4) == SignalResult()


# ------------------------------------------------------------ end to end

def test_end_to_end_backtest_produces_round_trips():
    frame = _regime_frame(n_bars=900)
    trades_total = 0
    for ticker in mod.TRADED:
        signal_fn = mod.make_signal_fn(ticker, frame.index, {ticker: frame})
        trades, in_market = backtest_engine.run_backtest(
            frame, signal_fn, cost_bps=5.0)
        assert len(trades) >= 1
        # every entry is a month-end decision executed at the NEXT open
        periods = frame.index.to_period("M")
        for _, row in trades.iterrows():
            entry_pos = frame.index.get_loc(row["entry_date"])
            assert periods[entry_pos - 1] != periods[entry_pos]
            assert row["entry_px"] == pytest.approx(float(frame["Open"].iloc[entry_pos]))
            assert row["held"] >= 1
        assert bool(in_market.any())
        trades_total += len(trades)
    assert trades_total >= 2


def test_end_to_end_short_frame_is_flat_without_error():
    """Fewer bars than the warm-up -> no signal, zero trades, no exception."""
    frame = _regime_frame(n_bars=200)
    signal_fn = mod.make_signal_fn("EMB", frame.index, {"EMB": frame})
    trades, in_market = backtest_engine.run_backtest(frame, signal_fn)
    assert len(trades) == 0
    assert not in_market.any()


def test_factory_rejects_missing_ticker_frame():
    frame = _regime_frame(n_bars=300)
    with pytest.raises(KeyError):
        mod.make_signal_fn("HYG", frame.index, {"EMB": frame})
