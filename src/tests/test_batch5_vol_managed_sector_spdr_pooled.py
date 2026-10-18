"""Tests for src/mining/batch5/vol_managed_sector_spdr_pooled.py.

Synthetic data only, deterministic, no network. The real ledger/catalog are
never touched.
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
from mining.batch5 import vol_managed_sector_spdr_pooled as mod  # noqa: E402

MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           "vol_managed_sector_spdr_pooled.py")

LOW_STEP = 0.0005    # daily log-return magnitude in a LOW-variance month
MID_STEP = 0.005     # warm-up months: sits between LOW and HIGH
HIGH_STEP = 0.03     # daily log-return magnitude in a HIGH-variance month
WARMUP_MONTHS = 20   # months of MID variance before the regime blocks start


def _regime_of(month: int) -> str:
    """Month ordinal -> regime. MID for the warm-up, then two-month blocks
    alternating HIGH/LOW (HIGH first). Two-month blocks guarantee the 21-day
    RV window ending on the block's SECOND month end lies wholly inside one
    regime, whatever the business-day count of the month."""
    if month < WARMUP_MONTHS:
        return "mid"
    return "high" if ((month - WARMUP_MONTHS) // 2) % 2 == 0 else "low"


def _is_second_of_block(month: int) -> bool:
    return month >= WARMUP_MONTHS and (month - WARMUP_MONTHS) % 2 == 1


# ------------------------------------------------------------ synthetic data

def _regime_frame(n_months: int, *, regime_of=_regime_of, start="2015-01-01",
                  seed: int = 0) -> pd.DataFrame:
    """Business-day OHLC whose daily log-return MAGNITUDE is constant within
    a calendar month and set by ``regime_of(month_ordinal)``. Signs are
    random so the price stays bounded while the realized variance of a window
    is exactly determined by the regimes it covers."""
    index = pd.bdate_range(start, periods=n_months * 30)
    periods = index.to_period("M")
    ordinal = pd.factorize(periods)[0]
    index = index[ordinal < n_months]
    ordinal = ordinal[ordinal < n_months]
    rng = np.random.RandomState(seed)
    signs = np.where(rng.rand(len(index)) < 0.5, -1.0, 1.0)
    magnitude = {"low": LOW_STEP, "mid": MID_STEP, "high": HIGH_STEP}
    step = np.array([magnitude[regime_of(int(m))] for m in ordinal])
    logret = signs * step
    logret[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(logret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _month_end_positions(index: pd.DatetimeIndex) -> list:
    periods = index.to_period("M")
    return [i for i in range(len(index) - 1) if periods[i] != periods[i + 1]]


def _signals(signal_fn, frame: pd.DataFrame, upto: int) -> list:
    return [(bool(s.entry), bool(s.exit)) for s in
            (signal_fn(frame.iloc[:i + 1], i) for i in range(upto + 1))]


def _factory(frame: pd.DataFrame, index=None, ticker="XLK"):
    return mod.make_signal_fn(ticker, frame.index if index is None else index,
                              {ticker: frame})


# ------------------------------------------------------------------ contract

def test_driver_validates_module_contract():
    import run_batch5_mining as batch5
    module = batch5._import_module_file("vol_managed_sector_spdr_pooled",
                                        MODULE_PATH)
    spec = batch5.validate_module_contract(module)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "pooled_single_asset"
    assert spec["traded"] == mod.TRADED
    assert batch5.parameter_drift(module.PARAMETERS, mod.PARAMETERS) is None


# ------------------------------------------------------ rule: fires / flat

def test_realized_variance_and_expanding_median_definitions():
    frame = _regime_frame(20)
    close = frame["Close"]
    rv, med = mod.realized_variance_and_threshold(close)
    logret = np.log(close).diff()
    # RV_t = trailing 21 squared log returns INCLUDING t -> first defined at
    # position 21 (21 returns need 22 closes).
    assert rv.iloc[:21].isna().all()
    assert np.isclose(rv.iloc[21], float((logret.iloc[1:22] ** 2).sum()))
    assert np.isclose(rv.iloc[50], float((logret.iloc[30:51] ** 2).sum()))
    # MED_t undefined until 252 RV observations exist: RV #252 sits at
    # position 21 + 251 = 272.
    assert med.iloc[:272].isna().all()
    assert np.isfinite(med.iloc[272])
    assert np.isclose(med.iloc[272], float(rv.iloc[21:273].median()))
    assert np.isclose(med.iloc[300], float(rv.iloc[21:301].median()))


def test_decision_bars_are_month_ends_and_never_the_final_bar():
    frame = _regime_frame(6)
    positions = mod.decision_positions(frame.index)
    assert list(positions) == _month_end_positions(frame.index)
    assert len(frame) - 1 not in set(positions)
    # A frame that ends exactly on a month end still excludes its final bar
    # (the calendar cannot tell it is a month end without the next session).
    last_me = _month_end_positions(frame.index)[-1]
    truncated = frame.iloc[:last_me + 1]
    assert last_me not in set(mod.decision_positions(truncated.index))


def test_rule_fires_after_low_variance_month_and_flat_after_high():
    # 20 months of MID variance warm-up, then two-month HIGH/LOW blocks. The
    # expanding median stays at the MID level (the majority of observations),
    # strictly between the LOW and HIGH RV levels, so the decision on the
    # SECOND month end of every block is unambiguous.
    n_months = 44
    frame = _regime_frame(n_months)
    signal_fn = _factory(frame)
    periods = frame.index.to_period("M")
    month_ends = _month_end_positions(frame.index)
    ordinal_of = {p: k for k, p in enumerate(periods.unique())}
    checked_long = checked_flat = 0
    for pos in month_ends:
        month = ordinal_of[periods[pos]]
        if not _is_second_of_block(month):
            continue
        sig = signal_fn(frame.iloc[:pos + 1], pos)
        if _regime_of(month) == "high":
            assert (sig.entry, sig.exit) == (False, True), (month, pos)
            checked_flat += 1
        else:
            assert (sig.entry, sig.exit) == (True, False), (month, pos)
            checked_long += 1
        assert isinstance(sig.entry, bool) and isinstance(sig.exit, bool)
        # Intra-month bars carry the decision unchanged (stay long / stay
        # flat) -- no intra-month re-evaluation, even though the following
        # month is the OTHER regime and its RV crosses the median mid-month.
        # Stops BEFORE the next month-end bar, which is the next decision.
        next_end = next((q for q in month_ends if q > pos), len(frame) - 1)
        assert next_end - pos >= 15
        for k in range(pos + 1, next_end):
            mid = signal_fn(frame.iloc[:k + 1], k)
            assert (mid.entry, mid.exit) == (sig.entry, sig.exit), (month, k)
    assert checked_long >= 5 and checked_flat >= 5


def test_no_position_before_threshold_is_defined():
    frame = _regime_frame(40)
    signal_fn = _factory(frame)
    _, med = mod.realized_variance_and_threshold(frame["Close"])
    first_defined = int(np.flatnonzero(med.notna())[0])
    for i in range(0, first_defined):
        sig = signal_fn(frame.iloc[:i + 1], i)
        assert (sig.entry, sig.exit) == (False, False), i
    # The first month-end decision at/after the threshold becomes defined
    # takes a position (long or flat -- either way not the empty result).
    first_decision = next(p for p in _month_end_positions(frame.index)
                          if p >= first_defined)
    assert signal_fn(frame.iloc[:first_decision + 1],
                     first_decision) != backtest_engine.SignalResult()


# ---------------------------------------------------------------- look-ahead

def _perturb_after(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    perturbed = frame.copy()
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] *= 1.3
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Open")] *= 1.3
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("High")] *= 1.9
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Low")] *= 0.7
    return perturbed


def test_lookahead_probe_future_bars_do_not_change_past_signals():
    """Probe at every month-end DECISION bar in the post-warm-up range plus
    mid-month bars: a one-bar look-ahead in RV only shows up when the probe
    boundary k IS a decision bar, so probing a single arbitrary k is not a
    proof. A ``rolling(...).sum().shift(-1)`` RV definition (a one-bar
    look-ahead) turns this test red."""
    frame = _regime_frame(40, seed=3)
    full = _signals(_factory(frame), frame, len(frame) - 1)
    decision_ks = [p for p in _month_end_positions(frame.index)
                   if 300 <= p <= len(frame) - 3]
    assert len(decision_ks) >= 12
    for k in decision_ks + [700, 705, 733]:
        baseline = full[:k + 1]
        perturbed = _perturb_after(frame, k)
        assert _signals(_factory(perturbed), perturbed, k) == baseline, k
        # Holdout-blind: nothing after the last bar of the bucket is needed
        # beyond the NEXT SESSION'S DATE (calendar only, to know k is a month
        # end) -- so a frame cut at k+1 must reproduce every signal through k.
        truncated = frame.iloc[:k + 2]
        assert _signals(_factory(truncated), truncated, k) == baseline, k
        # ... and cutting at k+1 with that one remaining bar's PRICES wrecked
        # must too (its date is used, its prices never are).
        wrecked = _perturb_after(truncated, k)
        assert _signals(_factory(wrecked), wrecked, k) == baseline, k


def test_holdout_bucket_uses_prior_history_for_warmup_and_matches_full_run():
    frame = _regime_frame(40, seed=5)
    split = len(frame) - 130
    bucket = frame.iloc[split:]
    full_fn = _factory(frame)
    bucket_fn = _factory(frame, index=bucket.index)
    for j in range(len(bucket)):
        expected = full_fn(frame.iloc[:split + j + 1], split + j)
        got = bucket_fn(bucket.iloc[:j + 1], j)
        assert (got.entry, got.exit) == (expected.entry, expected.exit), j
    # Warm-up came from history before the bucket: a factory given only the
    # bucket as history has no defined threshold for 130 bars.
    naked_fn = mod.make_signal_fn("XLK", bucket.index, {"XLK": bucket})
    assert all(naked_fn(bucket.iloc[:j + 1], j) == backtest_engine.SignalResult()
               for j in range(len(bucket)))


def test_signal_fn_rejects_misaligned_bar():
    frame = _regime_frame(30)
    signal_fn = _factory(frame)
    shifted = frame.iloc[1:]
    with pytest.raises(ValueError):
        signal_fn(shifted.iloc[:5], 4)


# --------------------------------------------------------------- end to end

def test_end_to_end_backtest_produces_round_trips():
    n_months = 30   # ~650 business days; threshold defined after ~273 bars
    frame = _regime_frame(n_months, seed=11)
    signal_fn = _factory(frame)
    trades, in_market = backtest_engine.run_backtest(frame, signal_fn)
    # Two-month LOW blocks after the warm-up -> at least one long round trip
    # per LOW block, closed at a HIGH-block month end.
    assert len(trades) >= 2
    assert (trades["held"] > 0).all()
    assert (trades["reason"] == "signal_exit").sum() >= 1
    assert bool(in_market.any())
    # Entries fill the session AFTER a month-end decision; exits on a
    # month-end close (or end_of_data).
    periods = frame.index.to_period("M")
    for _, row in trades.iterrows():
        entry_pos = frame.index.get_loc(row["entry_date"])
        assert periods[entry_pos - 1] != periods[entry_pos]
        if row["reason"] == "signal_exit":
            exit_pos = frame.index.get_loc(row["exit_date"])
            assert periods[exit_pos] != periods[exit_pos + 1]


def test_per_call_cost_is_constant_lookups():
    """The engine calls signal_fn once per bar with a growing slice; the
    per-call work must not grow with the slice length."""
    import time
    frame = _regime_frame(60, seed=2)
    signal_fn = _factory(frame)
    n = len(frame)
    calls = 400
    t0 = time.perf_counter()
    for i in range(50, 50 + calls):
        signal_fn(frame.iloc[:i + 1], i)
    early = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(n - calls, n):
        signal_fn(frame.iloc[:i + 1], i)
    late = time.perf_counter() - t0
    # Slicing itself is the only per-call cost the engine imposes; a
    # recompute-over-the-slice implementation would be many times slower at
    # the end than at the start.
    assert late < 5.0 * early + 0.05
