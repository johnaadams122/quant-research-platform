"""Tests for src/mining/batch5/vol_managed_intl_country_etf_pooled.py.

Synthetic OHLC only (deterministic, no network, no RNG). The independent
oracle in this file recomputes the rule the slow, literal way (an explicit
21-day sum and a fresh median over every prior RV observation at each
bar) so the module's vectorized precomputation is checked against a second
implementation rather than against itself.
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
from mining.batch5 import vol_managed_intl_country_etf_pooled as mod  # noqa: E402

RV_LOOKBACK = 21
MIN_OBS = 252
WARMUP = RV_LOOKBACK + MIN_OBS  # bars before the first defined median


# ------------------------------------------------------------ synthetic data

def _frame(n: int = 620, *, start: str = "2015-01-05",
           regimes: list[tuple[int, float]] | None = None) -> pd.DataFrame:
    """Deterministic OHLC on a weekday calendar. Daily log returns follow a
    slowly varying sine envelope so trailing variance moves above and below
    its own expanding median (both regimes occur after warm-up). When
    ``regimes`` is given as [(start_bar, amplitude), ...] the return
    amplitude is piecewise constant, giving unambiguous regimes."""
    index = pd.bdate_range(start, periods=n)
    k = np.arange(n, dtype=float)
    if regimes is None:
        amplitude = 0.006 + 0.004 * np.sin(k / 37.0)
    else:
        amplitude = np.zeros(n)
        for start_bar, level in regimes:
            amplitude[k >= start_bar] = level
    log_returns = amplitude * np.where(k % 2 == 0, 1.0, -1.0) * (
        1.0 + 0.1 * np.cos(k / 11.0))
    log_returns[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _oracle_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Slow literal re-implementation of the pseudocode. Returns
    (is_decision_bar, long_target) aligned to ``frame``."""
    closes = frame["Close"].to_numpy(dtype=float)
    n = len(closes)
    rv = np.full(n, np.nan)
    for t in range(n):
        if t < RV_LOOKBACK:  # needs 21 returns -> 22 closes
            continue
        total = 0.0
        for s in range(t - RV_LOOKBACK + 1, t + 1):
            total += np.log(closes[s] / closes[s - 1]) ** 2
        rv[t] = total
    med = np.full(n, np.nan)
    for t in range(n):
        history = rv[:t + 1]
        history = history[~np.isnan(history)]
        if len(history) >= MIN_OBS:
            med[t] = float(np.median(history))
    long_target = ~np.isnan(rv) & ~np.isnan(med) & (rv <= med)
    months = frame.index.year * 12 + frame.index.month
    is_decision = np.zeros(n, dtype=bool)
    is_decision[:-1] = months[1:] != months[:-1]
    return is_decision, long_target


def _signals(signal_fn, frame: pd.DataFrame) -> list[tuple[bool, bool]]:
    return [(bool(r.entry), bool(r.exit))
            for r in (signal_fn(frame.iloc[:i + 1], i) for i in range(len(frame)))]


# ------------------------------------------------------------------ contract

def test_contract_constants():
    assert mod.FAMILY_ID == "vol_managed_intl_country_etf_pooled"
    assert mod.SHAPE == "pooled_single_asset"
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "VEU"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert len(mod.TRADED) == 12 and len(set(mod.TRADED)) == 12
    assert callable(mod.make_signal_fn)
    assert not hasattr(mod, "make_target_fn")


def test_driver_discovery_validates_module():
    import run_batch5_mining as driver
    found, problems = driver.discover_modules(
        os.path.join(_ROOT, "src", "mining", "batch5"))
    ours = [p for p in problems
            if p["module"] == "vol_managed_intl_country_etf_pooled"]
    assert ours == [], ours
    assert mod.FAMILY_ID in found
    spec = found[mod.FAMILY_ID]["spec"]
    assert spec["shape"] == "pooled_single_asset"
    assert spec["traded"] == mod.TRADED


# ------------------------------------------------------------------- rule

def test_rule_matches_slow_oracle_on_every_bar():
    frame = _frame(620)
    signal_fn = mod.make_signal_fn("EWJ", frame.index, {"EWJ": frame})
    is_decision, long_target = _oracle_targets(frame)
    got = _signals(signal_fn, frame)
    expected = [((bool(d and lt)), (bool(d and not lt)))
                for d, lt in zip(is_decision, long_target)]
    assert got == expected
    # Both regimes must actually occur after warm-up, or the test is vacuous.
    after_warmup = got[WARMUP:]
    entries = sum(1 for e, x in after_warmup if e)
    exits = sum(1 for e, x in after_warmup if x)
    assert entries >= 1 and exits >= 1, (entries, exits)


def test_fires_long_on_calm_month_end_and_flat_on_wild_month_end():
    # Three regimes: moderately calm [0,250), wild [250,450), very calm
    # [450,620). Warm-up ends at bar 273 inside the wild segment, where the
    # expanding median is dominated by the ~229 earlier calm RVs, so every
    # wild month-end is unambiguously HIGH; after 450+21 the trailing RV is
    # ~4x below every earlier calm RV, so every late month-end is LOW.
    frame = _frame(620, regimes=[(0, 0.002), (250, 0.04), (450, 0.001)])
    signal_fn = mod.make_signal_fn("EWG", frame.index, {"EWG": frame})
    months = frame.index.year * 12 + frame.index.month
    month_ends = [i for i in range(len(frame) - 1) if months[i] != months[i + 1]]
    wild_ends = [i for i in month_ends if WARMUP <= i < 450]
    calm_ends = [i for i in month_ends if i >= 450 + RV_LOOKBACK]
    assert len(wild_ends) >= 3 and len(calm_ends) >= 3
    for i in wild_ends:
        result = signal_fn(frame.iloc[:i + 1], i)
        assert (result.entry, result.exit) == (False, True), i
    for i in calm_ends:
        result = signal_fn(frame.iloc[:i + 1], i)
        assert (result.entry, result.exit) == (True, False), i


def test_flat_before_threshold_defined_and_on_non_decision_bars():
    frame = _frame(620)
    signal_fn = mod.make_signal_fn("EWJ", frame.index, {"EWJ": frame})
    months = frame.index.year * 12 + frame.index.month
    for i in range(len(frame)):
        result = signal_fn(frame.iloc[:i + 1], i)
        is_month_end = i < len(frame) - 1 and months[i] != months[i + 1]
        if not is_month_end:
            assert (result.entry, result.exit) == (False, False), i
        elif i < WARMUP - 1:
            # Threshold undefined: "no position is taken" -> never an entry
            # (a flat exit signal here is a harmless no-op).
            assert result.entry is False, i
    # The final bar of the frame is never a decision bar.
    last = signal_fn(frame, len(frame) - 1)
    assert (last.entry, last.exit) == (False, False)


def test_requires_own_ticker_frame():
    frame = _frame(300)
    with pytest.raises(KeyError):
        mod.make_signal_fn("EWJ", frame.index, {"EWY": frame})


# --------------------------------------------------------------- look-ahead

def _perturb_after(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    perturbed = frame.copy()
    tail = len(frame) - (k + 1)
    # 1.3x level shift PLUS an alternating +/-5% wobble: a pure level shift
    # would leave every log return after k+1 unchanged and prove little.
    wobble = 1.3 * (1.0 + 0.05 * np.where(np.arange(tail) % 2 == 0, 1.0, -1.0))
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] *= wobble
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("High")] *= 1.5
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Low")] *= 0.7
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Open")] *= 1.2
    return perturbed


def test_look_ahead_probe_perturbing_future_bars_leaves_past_signals_unchanged():
    frame = _frame(620)
    base_fn = mod.make_signal_fn("EWJ", frame.index, {"EWJ": frame})
    base = _signals(base_fn, frame)
    is_decision, _ = _oracle_targets(frame)
    # Probe at EVERY post-warm-up decision bar (and one arbitrary mid-month
    # bar) so a window that peeks even one session ahead is caught at the
    # bar where it would actually change a decision.
    probe_points = [k for k in range(WARMUP, len(frame) - 1) if is_decision[k]]
    probe_points.append(480)
    assert len(probe_points) >= 10
    changed_somewhere = False
    for k in probe_points:
        perturbed = _perturb_after(frame, k)
        pert_fn = mod.make_signal_fn("EWJ", perturbed.index, {"EWJ": perturbed})
        after = _signals(pert_fn, perturbed)
        assert after[:k + 1] == base[:k + 1], k
        changed_somewhere |= after[k + 1:] != base[k + 1:]
    # Sanity: the perturbation is large enough to change SOME later signal,
    # otherwise the probe proves nothing.
    assert changed_somewhere


def test_holdout_bucket_is_blind_to_bars_after_its_last_date():
    frame = _frame(700)
    bucket = frame.index[400:560]  # a bucket ending mid-history
    base_fn = mod.make_signal_fn("EWJ", bucket, {"EWJ": frame})
    base = _signals(base_fn, frame.loc[bucket])

    perturbed = frame.copy()
    perturbed.iloc[560:, perturbed.columns.get_loc("Close")] *= 1.3
    pert_fn = mod.make_signal_fn("EWJ", bucket, {"EWJ": perturbed})
    assert _signals(pert_fn, perturbed.loc[bucket]) == base
    # And warm-up history BEFORE the bucket is used: the bucket's own
    # 160 bars alone could never define a 252-observation median.
    assert any(e or x for e, x in base)


# ---------------------------------------------------------------- end-to-end

def test_end_to_end_run_backtest_produces_trades_with_monthly_structure():
    frame = _frame(620)
    signal_fn = mod.make_signal_fn("FXI", frame.index, {"FXI": frame})
    trades, in_market = backtest_engine.run_backtest(frame, signal_fn)
    assert len(trades) >= 1
    months = frame.index.year * 12 + frame.index.month
    positions = {ts: i for i, ts in enumerate(frame.index)}
    for _, trade in trades.iterrows():
        entry_i = positions[trade["entry_date"]]
        # Entry fills at the t+1 open after a month-end decision, i.e. on
        # the first session of a month.
        assert months[entry_i] != months[entry_i - 1]
        assert trade["entry_px"] == pytest.approx(frame["Open"].iloc[entry_i])
        exit_i = positions[trade["exit_date"]]
        if trade["reason"] == "signal_exit":
            assert months[exit_i] != months[exit_i + 1]
        else:
            assert trade["reason"] == "end_of_data"
        assert trade["held"] >= 1
    assert bool(in_market.any())


def test_pooled_factory_runs_independently_per_ticker():
    frames = {"EWJ": _frame(620), "EWY": _frame(620, start="2016-03-01")}
    fns = {t: mod.make_signal_fn(t, frames[t].index, frames) for t in frames}
    for t, fn in fns.items():
        trades, _ = backtest_engine.run_backtest(frames[t], fn)
        assert len(trades) >= 1, t
