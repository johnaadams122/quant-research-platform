"""Tests for src/mining/batch5/long_horizon_index_mean_reversion.py.

Synthetic frames only (no network) for the math/contract/hysteresis/
look-ahead/holdout-blind checks. Independent oracles are used wherever
practical: ``numpy.lib.stride_tricks.sliding_window_view`` gives a
vectorized-but-independent (different code path from pandas' own
``.rolling()``) recomputation of the rolling mean/std/z-score, and
``standing_long_state``'s hysteresis is checked against a slow, literal
per-bar Python re-implementation.

PARAMETERS is compared against the REAL pre-registered catalog entry
(data/mining/catalog.json), matching
test_batch5_betting_against_beta_low_vol.py's own convention.

The real-data end-to-end checks (live market-history fetches) are not part of
this public file; every test here is offline and uses synthetic data.
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
from mining.batch5 import long_horizon_index_mean_reversion as mod  # noqa: E402

FAMILY_ID = "long_horizon_index_mean_reversion"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _calendar(start: str = "2000-01-03", periods: int = 6000) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _regime_closes(index: pd.DatetimeIndex, *,
                   regimes: list, seed: int = 0) -> np.ndarray:
    """Piecewise-constant-drift synthetic close series: ``regimes`` is a
    list of (n_bars, annualized_drift) tuples. A long calm-then-crash-then-
    recover-then-boom shape reliably produces both a deep trailing-return
    z-score trough (entry) and a subsequent peak (exit) once the warm-up
    windows are satisfied."""
    rng = np.random.RandomState(seed)
    n = len(index)
    daily = np.empty(n)
    pos = 0
    for n_bars, annual_drift in regimes:
        daily_drift = annual_drift / 252.0
        daily[pos:pos + n_bars] = daily_drift
        pos += n_bars
    assert pos == n, "regimes must cover the whole index"
    noise = rng.normal(0.0, 0.006, n)
    log_returns = daily + noise
    return 100.0 * np.exp(np.cumsum(log_returns))


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


def _oracle_rolling_zscore(trailing: np.ndarray, window: int) -> np.ndarray:
    """Independent (different code path) rolling z-score: builds every
    length-``window`` slice via ``sliding_window_view`` and takes numpy's
    own mean/std (ddof=1) over each slice -- vectorized, not pandas'
    ``.rolling()``, and not a per-bar Python loop."""
    n = len(trailing)
    out = np.full(n, np.nan)
    if n < window:
        return out
    valid_mask = ~np.isnan(trailing)
    # sliding_window_view requires no NaNs to be structurally excluded, so
    # windows touching a NaN trailing_return are left as NaN in the output,
    # matching pandas' own min_periods=window full-window requirement
    # (pandas also emits NaN for any window containing a NaN observation).
    windows = np.lib.stride_tricks.sliding_window_view(trailing, window)
    means = np.nanmean(windows, axis=1)
    stds = np.nanstd(windows, axis=1, ddof=1)
    complete = np.array([valid_mask[i:i + window].all() for i in range(n - window + 1)])
    z_tail = np.where(complete & (stds > 1e-9 * np.abs(means) + 1e-15),
                      (trailing[window - 1:] - means) / stds, np.nan)
    out[window - 1:] = z_tail
    return out


def _oracle_trailing_return(close: np.ndarray, window: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n > window:
        out[window:] = close[window:] / close[:-window] - 1.0
    return out


def _oracle_hysteresis(z: np.ndarray, decision: np.ndarray,
                       entry_z: float, exit_z: float) -> np.ndarray:
    """Slow, literal per-bar loop -- deliberately unvectorized, a second
    independent implementation of the same rule ``standing_long_state``
    itself implements (used to prove that function correct, not to replace
    it)."""
    n = len(z)
    out = np.zeros(n, dtype=bool)
    is_long = False
    for i in range(n):
        if decision[i]:
            if np.isnan(z[i]):
                is_long = False
            elif z[i] <= entry_z:
                is_long = True
            elif z[i] >= exit_z:
                is_long = False
        out[i] = is_long
    return out


# ------------------------------------------------------------------ contract

def test_contract_constants():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "pooled_single_asset"
    assert mod.TRADED == ["SPY", "QQQ", "IWM"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 0
    assert callable(mod.make_signal_fn)
    assert not hasattr(mod, "make_target_fn")


def test_parameters_are_byte_identical_to_the_frozen_catalog_entry():
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == mod.FAMILY_ID]
    assert len(matches) == 1
    entry = matches[0]
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert set(mod.TRADED) == set(entry["tickers"])
    assert mod.BENCHMARK == "SPY"  # ABI-1's own standing rule, not a catalog field


def test_module_passes_the_driver_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module, so the result does not
    depend on whatever else sits in the real batch5 dir."""
    import run_batch5_mining as batch5
    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["shape"] == "pooled_single_asset"
    assert spec["traded"] == mod.TRADED
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    catalog_entry = next(c for c in catalog if c["family_id"] == mod.FAMILY_ID)
    assert batch5.parameter_drift(
        spec["parameters"], catalog_entry["parameters"]) is None


def test_generic_dispatch_signature_is_usable():
    index = _calendar(periods=200)
    close = _regime_closes(index, regimes=[(200, 0.05)])
    frame = _ohlc_from_close(close, index)
    signal_fn = mod.make_signal_fn("SPY", index, {"SPY": frame})
    assert callable(signal_fn)


def test_design_constants_are_the_stated_trading_day_proxies():
    assert mod._RETURN_WINDOW_DAYS == 36 * 21 == 756
    assert mod._ZSCORE_LOOKBACK_DAYS == 15 * 252 == 3780
    assert mod._ENTRY_Z == -1.0
    assert mod._EXIT_Z == 1.0


# --------------------------------------------------------------------- math

def test_trailing_return_matches_naive_formula():
    rng = np.random.RandomState(1)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 2000)))
    index = _calendar(periods=2000)
    series = pd.Series(close, index=index)
    window = 50
    got = mod.trailing_return(series, window=window).to_numpy()
    expected = _oracle_trailing_return(close, window)
    np.testing.assert_allclose(got[window:], expected[window:], rtol=1e-12)
    assert np.isnan(got[:window]).all()


def test_trailing_return_rejects_nonpositive_window():
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mod.trailing_return(series, window=0)


def test_rolling_zscore_matches_independent_sliding_window_oracle():
    rng = np.random.RandomState(2)
    n = 3000
    trailing = rng.normal(0.02, 0.08, n)
    trailing[:30] = np.nan  # simulate an unwarmed prefix, as the real pipeline has
    series = pd.Series(trailing)
    window = 200
    got = mod.rolling_zscore(series, window=window).to_numpy()
    expected = _oracle_rolling_zscore(trailing, window)
    both_nan = np.isnan(got) & np.isnan(expected)
    np.testing.assert_allclose(got[~both_nan], expected[~both_nan], rtol=1e-9, atol=1e-9)
    assert (np.isnan(got) == np.isnan(expected)).all()
    # Sanity: the oracle isn't vacuously all-NaN.
    assert np.isfinite(expected).sum() > 1000


def test_rolling_zscore_zero_dispersion_is_nan_not_infinite():
    # A perfectly constant trailing_return window has std ~ 0 (float noise,
    # not exactly 0) -- must resolve to NaN (no decision), never a spurious
    # near-infinite z (matches preferred_stock_yield_spread_reversion.
    # trailing_yield_zscore's own guard).
    series = pd.Series(np.full(500, 0.05))
    z = mod.rolling_zscore(series, window=100)
    assert z.iloc[99:].isna().all()


def test_rolling_zscore_rejects_too_small_window():
    with pytest.raises(ValueError):
        mod.rolling_zscore(pd.Series([1.0, 2.0]), window=1)


def test_month_end_flags_matches_naive_calendar_check():
    index = _calendar(periods=500)
    got = mod._month_end_flags(index)
    months = index.year * 12 + index.month
    expected = np.zeros(len(index), dtype=bool)
    for i in range(len(index) - 1):
        expected[i] = months[i] != months[i + 1]
    np.testing.assert_array_equal(got, expected)
    assert got[-1] == False  # noqa: E712 -- final bar is never a decision bar


# ---------------------------------------------------------------- hysteresis

def test_standing_long_state_matches_slow_oracle_on_a_crafted_zscore_path():
    # A hand-crafted path exercising: undefined warm-up (flat), a dip that
    # does NOT cross entry (no-op), a real entry crossing, holding inside
    # the band, a real exit crossing, staying flat inside the band, and a
    # SECOND entry -- every branch of the state machine.
    z = np.array([
        np.nan, np.nan,      # undefined -> flat
        -0.5, -0.9,          # inside band, near entry but not crossing -> flat
        -1.0,                # entry boundary (inclusive) -> LONG
        -0.3, 0.5, 0.99,     # inside band while LONG -> hold LONG
        1.0,                 # exit boundary (inclusive) -> FLAT
        0.2, -0.99,          # inside band while FLAT -> hold FLAT
        -2.5,                # deep entry -> LONG again
        np.nan,              # undefined while LONG -> FLAT (undefined=flat)
        -1.5,                # entry again -> LONG
    ])
    decision = np.ones(len(z), dtype=bool)  # every bar is a decision bar here
    got = mod.standing_long_state(pd.Series(z), decision)
    expected = _oracle_hysteresis(z, decision, mod._ENTRY_Z, mod._EXIT_Z)
    np.testing.assert_array_equal(got, expected)
    # Pin the exact expected sequence too, not just agreement with the
    # oracle (the oracle could share a bug with the implementation).
    np.testing.assert_array_equal(got, [
        False, False, False, False, True, True, True, True, False, False,
        False, True, False, True])


def test_standing_long_state_ignores_non_decision_bars():
    z = np.array([-2.0, -2.0, -2.0, 2.0, 2.0])
    decision = np.array([True, False, False, False, False])
    got = mod.standing_long_state(pd.Series(z), decision)
    # Only the first bar is checked (entry at -2.0 <= -1); the state then
    # holds LONG through every later non-decision bar regardless of z.
    np.testing.assert_array_equal(got, [True, True, True, True, True])


def test_standing_long_state_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mod.standing_long_state(pd.Series([1.0, 2.0]), np.array([True]))


# -------------------------------------------------------------- full signal

def _big_regime_frame():
    """~30 years of synthetic daily data with an engineered LONG calm start
    (long enough that the 756+3780 = 4536-bar warm-up completes on a
    STABLE, low-dispersion baseline, not mid-transition), THEN a deep
    decline, a stagnant trough, a strong recovery, and a calmer resumption.

    Chosen so the rolling
    z-score both dips to a real entry crossing (<= -1, first at
    2007-10-31) and later rises to a real exit crossing (>= +1, completing
    a real signal_exit trade 2007-11-01 .. 2013-04-30 through the real
    engine). A shorter calm period would produce no entry at all: the
    rolling 3780-day z-window would still be mostly warm-up/decline history
    by the time z first becomes defined, absorbing the decline into the
    window's own baseline instead of showing it as an outlier. A long,
    stable calm PRECEDING the decline is what lets the decline register as
    a real deviation."""
    index = _calendar(start="1990-01-02", periods=7700)
    regimes = [
        (4600, 0.08),    # long, stable calm bull -- establishes the z-window baseline
        (900, -0.35),    # a sharp multi-year decline -> pushes trailing_return sharply down
        (200, 0.0),      # stagnation at the bottom (keeps trailing_return depressed)
        (1200, 0.35),    # a strong multi-year recovery -> trailing_return spikes up
        (800, 0.08),     # a calmer resumption
    ]
    n = sum(r[0] for r in regimes)
    assert n == len(index)
    close = _regime_closes(index, regimes=regimes, seed=7)
    return _ohlc_from_close(close, index)


def test_full_pipeline_produces_both_a_real_entry_and_a_real_exit():
    frame = _big_regime_frame()
    signal_fn = mod.make_signal_fn("SPY", frame.index, {"SPY": frame})
    sigs = _signals(signal_fn, frame)
    entries = sum(1 for s in sigs if s.entry)
    assert entries >= 1, "engineered decline never produced a real entry crossing"
    # entry/exit reflect the standing LONG/FLAT state on EVERY bar (see
    # _precompute -- a bucket starting already-long must see entry=True at
    # its own first bar, not just at a later decision bar), so the real
    # invariant is that the state only
    # ever TRANSITIONS on an actual month-end decision bar -- not that a
    # True flag can only appear there.
    decision = mod._month_end_flags(mod._naive_index(frame.index))
    state = [s.entry for s in sigs]
    for i in range(1, len(state)):
        if state[i] != state[i - 1]:
            assert decision[i], i
    # A REAL exit -- not merely the harmless flat-exit no-op raised on every
    # decision bar while already flat/undefined (see module docstring's
    # "undefined = flat" convention) -- is proven by a completed round-trip
    # trade through the real engine, not by counting raw exit flags.
    trades, _ = run_backtest(frame, signal_fn, cost_bps=1.0)
    assert (trades["reason"] == "signal_exit").any(), \
        "engineered recovery never produced a real completed exit trade"


def test_full_pipeline_non_decision_bars_never_change_the_standing_state():
    """entry/exit reflect the current state on every bar (continuous,
    not decision-bar-gated -- see _precompute), so a non-decision bar
    must never show a DIFFERENT state than the immediately preceding bar --
    only an actual month-end decision bar may flip it."""
    frame = _big_regime_frame()
    signal_fn = mod.make_signal_fn("QQQ", frame.index, {"QQQ": frame})
    sigs = _signals(signal_fn, frame)
    decision = mod._month_end_flags(mod._naive_index(frame.index))
    state = [s.entry for s in sigs]
    for i in range(1, len(state)):
        if not decision[i]:
            assert state[i] == state[i - 1], i


def test_requires_own_ticker_frame():
    frame = _big_regime_frame()
    with pytest.raises(KeyError):
        mod.make_signal_fn("SPY", frame.index, {"QQQ": frame})


def test_rejects_ticker_not_in_traded():
    frame = _big_regime_frame()
    with pytest.raises(ValueError):
        mod.make_signal_fn("XYZ", frame.index, {"XYZ": frame})


# ------------------------------------------------------------------ look-ahead

def _perturb_after(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    perturbed = frame.copy()
    tail = len(frame) - (k + 1)
    wobble = 1.4 * (1.0 + 0.06 * np.where(np.arange(tail) % 2 == 0, 1.0, -1.0))
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] *= wobble
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Open")] *= 1.2
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("High")] *= 1.6
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Low")] *= 0.6
    return perturbed


def test_look_ahead_probe_perturbing_future_bars_leaves_past_signals_unchanged():
    frame = _big_regime_frame()
    base_fn = mod.make_signal_fn("SPY", frame.index, {"SPY": frame})
    base = _signals(base_fn, frame)
    k = 5300  # well after warm-up (~4535) and the first entry (~4651),
              # well before the real completed exit trade (bar 6085)
    perturbed = _perturb_after(frame, k)
    pert_fn = mod.make_signal_fn("SPY", perturbed.index, {"SPY": perturbed})
    after = _signals(pert_fn, perturbed)
    assert after[:k + 1] == base[:k + 1]
    # The perturbation is real: it changes SOME later signal, or the probe
    # proves nothing.
    assert after[k + 1:] != base[k + 1:]


def test_holdout_bucket_is_blind_to_bars_after_its_last_date():
    frame = _big_regime_frame()
    # A bucket ending mid-history, well past warm-up (~4535) and inside the
    # standing-LONG regime that opened at the first entry (~4651) -- real
    # entry/exit-raising activity is confirmed to occur in this exact span
    # (measured).
    bucket = frame.index[4700:5200]
    base_fn = mod.make_signal_fn("SPY", bucket, {"SPY": frame})
    base = _signals(base_fn, frame.loc[bucket])
    assert any(s.entry or s.exit for s in base)  # sanity: not a vacuous bucket

    perturbed = frame.copy()
    perturbed.iloc[5200:, perturbed.columns.get_loc("Close")] *= 1.5
    pert_fn = mod.make_signal_fn("SPY", bucket, {"SPY": perturbed})
    assert _signals(pert_fn, perturbed.loc[bucket]) == base


def test_restricted_bucket_still_resolves_true_prior_hysteresis_state():
    """Regression-style proof (matching betting_against_beta_low_vol.py's
    own restricted-bucket precedent): a holdout bucket beginning mid-
    history must carry in the SAME standing state a full-calendar bucket
    would show at that same date, because the hysteresis state depends on
    history before the bucket's own first bar. Starting the holdout at
    4700 -- after the real entry crossing at 4651 -- means a bucket-only
    view would open FLAT (it never sees bar 4651's own entry decision),
    while the correct, full-history-aware answer must open already LONG."""
    frame = _big_regime_frame()
    full_fn = mod.make_signal_fn("SPY", frame.index, {"SPY": frame})
    full_sigs = _signals(full_fn, frame)

    holdout_index = frame.index[4700:]  # starts well after warm-up, mid-regime
    restricted_fn = mod.make_signal_fn("SPY", holdout_index, {"SPY": frame})
    restricted_frame = frame.loc[holdout_index]
    restricted_sigs = _signals(restricted_fn, restricted_frame)

    offset = frame.index.get_loc(holdout_index[0])
    assert restricted_sigs == full_sigs[offset:]


def test_restricted_bucket_starting_already_long_enters_at_first_reachable_bar():
    """Real engine-position proof, NOT a self-consistency check against
    another signal array (the gap the prior test above has -- both the
    full-history and restricted-bucket signal arrays can agree with each
    other while BOTH are wrong, if entry only fires on the NEXT decision
    bar inside the bucket rather than reflecting the carried-in state
    immediately). A bucket starting mid-regime (already LONG, per the
    prior test's own 4700 offset -- well after the real entry crossing at
    4651, well before the next month-end decision bar) must have
    run_backtest actually hold a position from the engine's own earliest
    reachable bar (bar 1, one bar of universal engine latency -- see e.g.
    value_factor_book_to_market.py's identical convention), not delay
    entry until a later month-end merely re-confirms the already-standing
    state."""
    frame = _big_regime_frame()
    holdout_index = frame.index[4700:4750]  # 50 bars: several bars before
    # the next month-end decision bar is reached, per this fixture's real
    # ~21-trading-day month-end cadence -- if entry incorrectly waits for
    # that decision bar, in_market stays False for a real, measurable span.
    restricted_fn = mod.make_signal_fn("SPY", holdout_index, {"SPY": frame})
    restricted_frame = frame.loc[holdout_index]
    trades, in_market = run_backtest(restricted_frame, restricted_fn, cost_bps=0.0)

    assert bool(in_market.iloc[1]), (
        "the state carried in from before this bucket is LONG (per the "
        "prior test's own 4700 offset) -- the engine must hold a position "
        "by its own earliest reachable bar, not wait for a later "
        "month-end decision bar to merely re-confirm a state that was "
        "already true when the bucket began")
    assert len(trades) >= 1
    assert trades.iloc[0]["entry_date"] == restricted_frame.index[1]


# ---------------------------------------------------------------- end-to-end

def test_end_to_end_run_backtest_produces_a_trade_with_monthly_fill_timing():
    frame = _big_regime_frame()
    signal_fn = mod.make_signal_fn("IWM", frame.index, {"IWM": frame})
    trades, in_market = run_backtest(frame, signal_fn, cost_bps=1.0)
    assert len(trades) >= 1
    months = frame.index.year * 12 + frame.index.month
    positions = {ts: i for i, ts in enumerate(frame.index)}
    for _, trade in trades.iterrows():
        entry_i = positions[trade["entry_date"]]
        # Entry fills the session after a month-end decision -- the first
        # session of a new month.
        assert months[entry_i] != months[entry_i - 1]
        assert trade["entry_px"] == pytest.approx(frame["Open"].iloc[entry_i])
        if trade["reason"] == "signal_exit":
            exit_i = positions[trade["exit_date"]]
            assert months[exit_i] != months[exit_i + 1]
        else:
            assert trade["reason"] == "end_of_data"
        assert trade["held"] >= 1
    assert bool(in_market.any())


def test_pooled_factory_runs_independently_per_ticker():
    frame_a = _big_regime_frame()
    index_b = _calendar(start="1990-01-02", periods=7700)
    close_b = _regime_closes(index_b, regimes=[
        (4600, 0.07), (900, -0.32), (200, 0.0), (1200, 0.32), (800, 0.07)],
        seed=11)
    frame_b = _ohlc_from_close(close_b, index_b)
    frames = {"SPY": frame_a, "QQQ": frame_b}
    for ticker, frame in frames.items():
        fn = mod.make_signal_fn(ticker, frame.index, frames)
        trades, _ = run_backtest(frame, fn, cost_bps=1.0)
        assert len(trades) >= 1, ticker


def test_signal_fn_out_of_range_position_is_a_no_op():
    frame = _big_regime_frame()
    signal_fn = mod.make_signal_fn("SPY", frame.index, {"SPY": frame})
    assert signal_fn(frame, -1) == SignalResult()
    assert signal_fn(frame, len(frame)) == SignalResult()
