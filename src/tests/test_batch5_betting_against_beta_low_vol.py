"""Tests for src/mining/batch5/betting_against_beta_low_vol.py.

Mostly synthetic frames (no network) for the contract/rebalance-cadence/
look-ahead checks, mirroring test_batch5_treasury_duration_volatility_timing.
py's structure. This family's PARAMETERS are compared against the REAL
pre-registered catalog entry (data/mining/catalog.json), matching
test_batch5_treasury_duration_volatility_timing.py's own convention.
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
from mining.batch5 import betting_against_beta_low_vol as mod  # noqa: E402

FAMILY_ID = "betting_against_beta_low_vol"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
TRADED_TICKER, SIGNAL_TICKER = "SPLV", "SPY"


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


def _close_from_returns(returns: np.ndarray, start_level: float = 100.0) -> np.ndarray:
    return start_level * np.exp(np.cumsum(returns))


def _beta_controlled_closes(index: pd.DatetimeIndex, *, regimes: list,
                            seed: int = 0) -> tuple:
    """Build SPLV/SPY close series whose rolling beta of SPLV on SPY is
    driven toward a controlled target in successive contiguous blocks.

    ``regimes`` is a list of (n_bars, beta_target) tuples. SPY's return is a
    common shock plus idiosyncratic noise; SPLV's return is
    ``beta_target * common_shock`` plus its own idiosyncratic noise, so the
    realized rolling beta over a window sitting entirely within one block
    converges close to that block's target."""
    rng = np.random.RandomState(seed)
    n = len(index)
    common_shock = rng.normal(0.0, 0.01, n)
    spy_idio = rng.normal(0.0, 0.0003, n)
    splv_idio = rng.normal(0.0, 0.0003, n)
    beta_true = np.empty(n)
    pos = 0
    for n_bars, target in regimes:
        beta_true[pos:pos + n_bars] = target
        pos += n_bars
    assert pos == n, "regimes must cover the whole index"
    spy_ret = common_shock + spy_idio
    splv_ret = beta_true * common_shock + splv_idio
    spy_close = _close_from_returns(spy_ret, 300.0)
    splv_close = _close_from_returns(splv_ret, 50.0)
    return splv_close, spy_close


def _drifting_beta_closes(index: pd.DatetimeIndex, *, beta_start: float = 0.2,
                          beta_end: float = 1.8, seed: int = 5) -> tuple:
    """Like _beta_controlled_closes, but the target beta ramps smoothly and
    continuously across the whole index instead of holding in blocks --
    used to prove the realized rolling beta keeps drifting day over day,
    while this module's own signal only updates at monthly rebalance
    boundaries."""
    n = len(index)
    beta_true = np.linspace(beta_start, beta_end, n)
    rng = np.random.RandomState(seed)
    common_shock = rng.normal(0.0, 0.01, n)
    spy_idio = rng.normal(0.0, 0.0003, n)
    splv_idio = rng.normal(0.0, 0.0003, n)
    spy_ret = common_shock + spy_idio
    splv_ret = beta_true * common_shock + splv_idio
    spy_close = _close_from_returns(spy_ret, 300.0)
    splv_close = _close_from_returns(splv_ret, 50.0)
    return splv_close, spy_close


def _frames(index: pd.DatetimeIndex, splv_close: np.ndarray,
           spy_close: np.ndarray) -> dict:
    return {"SPLV": _ohlc_from_close(splv_close, index),
            "SPY": _ohlc_from_close(spy_close, index)}


def _signals(signal_fn, frame: pd.DataFrame, upto: int | None = None) -> list:
    n = len(frame) if upto is None else upto + 1
    return [signal_fn(frame.iloc[:i + 1], i) for i in range(n)]


def _reference_rebalance_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Direct, unvectorized reference implementation of the
    last-rebalance-position rule, called on every growing prefix of
    ``index`` as a per-call recomputation would be -- used only to prove
    this module's vectorized ``rebalance_positions`` is bit-for-bit
    equivalent, not to replace it."""
    max_days = 23
    n_total = len(index)
    out = np.empty(n_total, dtype=np.int64)
    for i in range(n_total):
        sub = index[: i + 1]
        n = len(sub)
        scan_from = max(1, n - max_days)
        last = max(0, n - max_days)
        for j in range(scan_from, n):
            prev_key = (sub[j - 1].year, sub[j - 1].month)
            cur_key = (sub[j].year, sub[j].month)
            if cur_key != prev_key:
                last = j
        out[i] = last
    return out


# -------------------------------------------------------------- contract

def test_module_constants_match_candidate_contract():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["SPLV"]
    assert mod.SIGNAL == ["SPY"]
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert mod.EXPECTED_MIN_TRADES >= 0
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_the_frozen_catalog_entry():
    # catalog.json is the pre-registered, already-frozen source this module must not
    # reinterpret (same situation/convention as
    # treasury_duration_volatility_timing.py).
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
    """Discovery on a private copy of the module (other processes may write to
    the real batch5 dir concurrently, so the real dir is not asserted on)."""
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
    # this module must be directly callable that way (SPY is a normal
    # data_pipeline.fetch_ohlc-fetchable ticker, so this family needs no
    # bespoke external-data driver).
    index = _calendar(periods=200)
    splv_close, spy_close = _beta_controlled_closes(index, regimes=[(200, 0.4)])
    frames = _frames(index, splv_close, spy_close)
    signal_fn = mod.make_signal_fn("SPLV", index, frames)
    assert callable(signal_fn)


def test_factory_rejects_wrong_ticker_or_missing_frames():
    index = _calendar(periods=100)
    splv_close, spy_close = _beta_controlled_closes(index, regimes=[(100, 0.4)])
    frames = _frames(index, splv_close, spy_close)
    with pytest.raises(ValueError):
        mod.make_signal_fn("QQQ", index, frames)
    with pytest.raises(ValueError):
        mod.make_signal_fn("SPLV", index, {"SPLV": frames["SPLV"]})


# ------------------------------------------------------- rebalance cadence

def test_rebalance_dates_matches_the_reference_growing_prefix_loop():
    # On a FULL calendar starting at the true beginning of history (no
    # "before" period to consider), rebalance_dates's vectorized full-scan
    # must agree bit-for-bit with the reference per-call
    # growing-prefix loop -- scanning positions <= i within a fixed full
    # calendar is equivalent to scanning the growing prefix index[:i+1],
    # since neither can be affected by anything after position i.
    index = _calendar(start="2018-01-02", periods=760)
    ours = mod.rebalance_dates(index)
    reference_positions = _reference_rebalance_positions(index)
    reference_dates = index[reference_positions]
    assert ours.equals(reference_dates)


def test_rebalance_date_only_advances_on_a_real_month_transition():
    index = _calendar(start="2019-01-02", periods=300)
    reb_dates = mod.rebalance_dates(index)
    for i in range(1, len(index)):
        if reb_dates[i] != reb_dates[i - 1]:
            # A change in the owning rebalance date must itself land on a
            # real (year, month) transition day, or on index[i] itself
            # becoming the new rebalance day.
            assert reb_dates[i] == index[i]
            prev_key = (index[i - 1].year, index[i - 1].month)
            cur_key = (index[i].year, index[i].month)
            assert cur_key != prev_key


def test_restricted_holdout_bucket_still_resolves_true_prior_rebalance_date():
    """Guards rebalance ownership on a restricted bucket. A holdout-style
    bucket beginning mid-month (as run_batch5_mining.py's own
    compute_candidate/_signal_factory passes for in-sample/holdout
    evaluation, unlike ABI-1's own full-history bucket) has NO real prior
    transition visible within it; resolving rebalance ownership from
    ``bucket`` alone would silently treat the bucket's own start as the
    rebalance date -- wrong for up to _MAX_TRADING_DAYS_PER_MONTH bars every
    time. So the signal for the first stretch of a RESTRICTED bucket must be byte-
    identical to what a FULL-calendar bucket says for those same dates,
    because both must resolve to the SAME true prior rebalance date, even
    though that date falls entirely before the restricted bucket's own
    first bar."""
    window = mod._BETA_LOOKBACK_DAYS
    full_index = _calendar(start="2015-01-02", periods=window + 300)
    splv_close, spy_close = _beta_controlled_closes(
        full_index, regimes=[(window + 300, 0.3)])
    frames = _frames(full_index, splv_close, spy_close)

    # Ground truth: score the FULL calendar as one bucket.
    full_signal_fn = mod.make_signal_fn("SPLV", full_index, frames)
    full_sigs = _signals(full_signal_fn, frames["SPLV"])

    # Pick a restricted holdout bucket starting well past warm-up, MID-MONTH
    # (not on a rebalance day itself), so its true prior rebalance date is
    # guaranteed to fall before its own first bar.
    holdout_start_pos = window + 60
    while full_index[holdout_start_pos].day <= 3:  # avoid landing near a
        holdout_start_pos += 1                     # transition day itself
    holdout_index = full_index[holdout_start_pos:]
    assert holdout_index[0] > full_index[window]  # sanity: real mid-history start

    # The engine truncates the OHLC frame it feeds signal_fn to the bucket it
    # is scoring -- frames["SPLV"] itself stays full-history (the factory's
    # own `frames` argument), but the per-call `frame` argument only ever
    # covers the bucket, matching mining.backtest_engine.run_backtest's real
    # behavior.
    restricted_signal_fn = mod.make_signal_fn("SPLV", holdout_index, frames)
    restricted_frame = frames["SPLV"].loc[holdout_index]
    restricted_sigs = _signals(restricted_signal_fn, restricted_frame)

    # The first ~23 bars of the restricted bucket span the SAME true
    # rebalance period as whatever full-calendar bars precede
    # holdout_start_pos -- both must agree byte-for-byte with the full-
    # calendar ground truth over that stretch, not silently diverge because
    # the restricted bucket could not see back far enough.
    offset = full_index.get_loc(holdout_index[0])
    check_len = min(mod._MAX_TRADING_DAYS_PER_MONTH, len(holdout_index))
    for i in range(check_len):
        assert restricted_sigs[i] == full_sigs[offset + i], i


# -------------------------------------------------------------------- rule

def test_warm_up_period_before_full_beta_window_is_flat():
    window = mod._BETA_LOOKBACK_DAYS
    index = _calendar(periods=window + 40)
    splv_close, spy_close = _beta_controlled_closes(index, regimes=[(window + 40, 0.3)])
    frames = _frames(index, splv_close, spy_close)
    signal_fn = mod.make_signal_fn("SPLV", index, frames)
    sigs = _signals(signal_fn, frames["SPLV"])
    # Fewer than `window` common-calendar return observations exist for any
    # rebalance date this early -> beta is undefined -> flat (exit=True),
    # matching this family's own "undefined = not hold" convention (unlike
    # the batch's more common "no signal" convention).
    for i in range(window):
        assert sigs[i] == SignalResult(entry=False, exit=True), i


def test_low_beta_regime_holds_risk_on_entry():
    window = mod._BETA_LOOKBACK_DAYS
    index = _calendar(periods=window + 100)
    splv_close, spy_close = _beta_controlled_closes(
        index, regimes=[(window + 100, 0.3)])
    frames = _frames(index, splv_close, spy_close)
    signal_fn = mod.make_signal_fn("SPLV", index, frames)
    sigs = _signals(signal_fn, frames["SPLV"])
    # Well past warm-up (extra margin so every rebalance date used has its
    # own full window), a persistently low-beta regime must hold SPLV.
    tail = sigs[window + 40:]
    assert any(s.entry for s in tail)
    assert not any(s.exit for s in tail)


def test_high_beta_regime_stays_flat():
    window = mod._BETA_LOOKBACK_DAYS
    index = _calendar(periods=window + 100)
    splv_close, spy_close = _beta_controlled_closes(
        index, regimes=[(window + 100, 1.8)])
    frames = _frames(index, splv_close, spy_close)
    signal_fn = mod.make_signal_fn("SPLV", index, frames)
    sigs = _signals(signal_fn, frames["SPLV"])
    tail = sigs[window + 40:]
    assert any(s.exit for s in tail)
    assert not any(s.entry for s in tail)


def test_boundary_beta_equal_to_threshold_is_flat():
    # Rule is strict "<" for entry, so beta == 1.0 exactly is flat (exit),
    # not held -- this family's own asymmetric boundary, opposite
    # of stock_bond_correlation_regime_flip.py's inclusive entry boundary.
    beta = np.array([1.0, 0.999999, 1.000001, np.nan])
    entry, exit_ = mod.hold_flags(beta, mod._BETA_ENTRY_THRESHOLD)
    assert list(entry) == [False, True, False, False]
    assert list(exit_) == [True, False, True, True]


def test_signal_holds_constant_between_rebalance_dates_even_as_raw_beta_drifts():
    window = mod._BETA_LOOKBACK_DAYS
    index = _calendar(start="2016-01-04", periods=window + 400)
    splv_close, spy_close = _drifting_beta_closes(index)
    frames = _frames(index, splv_close, spy_close)
    signal_fn = mod.make_signal_fn("SPLV", index, frames)
    sigs = _signals(signal_fn, frames["SPLV"])

    reb_dates = mod.rebalance_dates(mod._naive_index(index))
    splv_naive = mod._clean_close(frames["SPLV"], mod._naive_index(index)[-1])
    spy_naive = mod._clean_close(frames["SPY"], mod._naive_index(index)[-1])
    raw_beta = mod.trailing_beta_vs_spy(splv_naive, spy_naive, window)
    naive_index = mod._naive_index(index)
    beta_by_date = pd.to_numeric(raw_beta.reindex(naive_index),
                                 errors="coerce").to_numpy(dtype=float)

    n = len(index)
    changed_positions = 0
    saw_within_period_pair = False
    for i in range(window + 1, n):
        if reb_dates[i] == reb_dates[i - 1]:
            # Same owning rebalance period as yesterday -> the signal must
            # be byte-identical, even though the raw daily beta reading
            # (computed independently, unlagged) may have moved.
            assert sigs[i] == sigs[i - 1], i
            if np.isfinite(beta_by_date[i]) and np.isfinite(beta_by_date[i - 1]):
                if beta_by_date[i] != beta_by_date[i - 1]:
                    saw_within_period_pair = True
        else:
            if sigs[i] != sigs[i - 1]:
                changed_positions += 1
    # The perturbation is real: the raw (unlagged) beta reading actually
    # moves day-to-day within at least one held rebalance period, proving
    # the "held constant" assertion above is not vacuous (i.e. not simply
    # because beta itself never changes).
    assert saw_within_period_pair
    # And the signal does change at least once across this long, smoothly
    # drifting-beta scenario -- exclusively on days that are themselves a
    # new rebalance position (enforced by the loop above).
    assert changed_positions >= 1


# ------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_do_not_change_past_signals():
    window = mod._BETA_LOOKBACK_DAYS
    index = _calendar(start="2016-01-04", periods=window + 200)
    splv_close, spy_close = _drifting_beta_closes(index, seed=9)
    frames = _frames(index, splv_close, spy_close)
    k = window + 120
    base_fn = mod.make_signal_fn("SPLV", index, frames)
    base = _signals(base_fn, frames["SPLV"], upto=k)

    perturbed = {}
    for symbol, frame in frames.items():
        pert = frame.copy()
        pert.iloc[k + 1:, pert.columns.get_loc("Close")] *= 1.4
        pert.iloc[k + 1:, pert.columns.get_loc("Open")] *= 1.3
        pert.iloc[k + 1:, pert.columns.get_loc("High")] *= 1.6
        pert.iloc[k + 1:, pert.columns.get_loc("Low")] *= 0.6
        perturbed[symbol] = pert
    pert_fn = mod.make_signal_fn("SPLV", index, perturbed)
    after = _signals(pert_fn, perturbed["SPLV"], upto=k)
    assert base == after
    # The perturbation is real: SOME later signal differs.
    assert (_signals(base_fn, frames["SPLV"])
            != _signals(pert_fn, perturbed["SPLV"]))


def test_holdout_blind_frames_extending_past_index_are_ignored():
    window = mod._BETA_LOOKBACK_DAYS
    index = _calendar(start="2016-01-04", periods=window + 260)
    splv_close, spy_close = _drifting_beta_closes(index, seed=13)
    frames = _frames(index, splv_close, spy_close)
    bucket = index[: window + 200]
    fn_full = mod.make_signal_fn("SPLV", bucket, frames)
    truncated = {s: f.loc[f.index <= bucket[-1]] for s, f in frames.items()}
    fn_trunc = mod.make_signal_fn("SPLV", bucket, truncated)
    bucket_frame = frames["SPLV"].loc[bucket]
    assert _signals(fn_full, bucket_frame) == _signals(fn_trunc, bucket_frame)


# ----------------------------------------------------------- end to end

def test_end_to_end_run_backtest_produces_trades():
    window = mod._BETA_LOOKBACK_DAYS
    regimes = [(window + 60, 0.3), (window + 20, 1.8), (window + 20, 0.3),
              (window + 20, 1.8), (window + 60, 0.3)]
    n = sum(r[0] for r in regimes)
    index = _calendar(periods=n)
    splv_close, spy_close = _beta_controlled_closes(index, regimes=regimes)
    frames = _frames(index, splv_close, spy_close)
    signal_fn = mod.make_signal_fn("SPLV", index, frames)
    trades, in_market = run_backtest(frames["SPLV"], signal_fn, cost_bps=1.0)
    assert len(trades) >= 1
    assert in_market.any()


def test_signal_fn_is_constant_time_per_call():
    import time
    window = mod._BETA_LOOKBACK_DAYS
    index = _calendar(start="2006-01-03", periods=3000)
    splv_close, spy_close = _beta_controlled_closes(index, regimes=[(3000, 0.4)])
    frames = _frames(index, splv_close, spy_close)
    signal_fn = mod.make_signal_fn("SPLV", index, frames)
    frame = frames["SPLV"]
    started = time.perf_counter()
    for i in range(len(index)):
        signal_fn(frame.iloc[:i + 1], i)
    assert time.perf_counter() - started < 5.0
