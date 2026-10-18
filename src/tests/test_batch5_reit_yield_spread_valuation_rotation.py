"""Tests for src/mining/batch5/reit_yield_spread_valuation_rotation.py.

This family implements its catalog entry's mechanism in the batch5 module
contract -- see the module's own docstring. The PARAMETERS drift check here
compares against the data/mining/catalog.json entry, read directly,
matching e.g.
test_batch5_defensive_sector_recession_signal_rotation.py's own identical
convention.

The real-data end-to-end checks (live market-history fetches) are not part of
this public file; every test here is offline and uses synthetic data.
"""
import json
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

import data_pipeline  # noqa: E402
import run_batch3_mining as batch3  # noqa: E402
from mining import abi1_driver, rotation_engine, signals  # noqa: E402
from mining.batch5 import reit_yield_spread_valuation_rotation as mod  # noqa: E402

FAMILY_ID = "reit_yield_spread_valuation_rotation"
RISK, DEFENSIVE = "VNQ", "SPY"
TNX = "^TNX"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")


# ------------------------------------------------------------------ helpers

def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _catalog_entry() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == FAMILY_ID]
    assert len(matches) == 1, (
        f"expected exactly one catalog.json entry for {FAMILY_ID}, found "
        f"{len(matches)}")
    return matches[0]


# -------------------------------------------------------------- contract

def test_module_constants_match_the_spec():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]
    assert mod.SIGNAL == [TNX]
    assert mod.BENCHMARK == "SPY"
    assert mod.SNAPSHOTS == {}
    assert isinstance(mod.EXPECTED_MIN_TRADES, int)
    assert not isinstance(mod.EXPECTED_MIN_TRADES, bool)
    assert mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_target_fn)


def test_parameters_are_byte_identical_to_the_real_catalog_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert set(entry["tickers"]) == {RISK, DEFENSIVE}
    assert entry["parameters"]["yield_spread_lookback_years"] == 3
    assert entry["parameters"]["entry_zscore"] == 0.5
    assert entry["parameters"]["exit_zscore"] == -0.5


def test_driver_discovery_validates_this_module_in_isolation(tmp_path):
    """Discovery on a private copy of the module, so the result depends only
    on this module and not on whatever else is in the real batch5 dir."""
    import shutil
    import run_batch5_mining as batch5
    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["family_id"] == FAMILY_ID
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == [RISK, DEFENSIVE]
    assert spec["signal"] == [TNX]
    assert spec["benchmark"] == "SPY"
    assert spec["expected_min_trades"] == mod.EXPECTED_MIN_TRADES
    assert batch5.parameter_drift(
        spec["parameters"], _catalog_entry()["parameters"]) is None


# ------------------------------------------------------------------ factory validation

def test_make_target_fn_requires_every_traded_and_signal_frame():
    index = _bdays("2015-01-02", 50)
    vnq = _ohlc_from_close(np.full(len(index), 90.0), index)
    spy = _ohlc_from_close(np.full(len(index), 400.0), index)
    tnx = _ohlc_from_close(np.full(len(index), 4.5), index)
    close = pd.Series(90.0, index=index)
    divs = pd.Series(0.0, index=index)

    with pytest.raises(ValueError, match=DEFENSIVE):
        mod.make_target_fn({RISK: vnq, TNX: tnx},
                           dividend_inputs=(close, divs))

    with pytest.raises(ValueError, match=r"\^TNX"):
        mod.make_target_fn({RISK: vnq, DEFENSIVE: spy},
                           dividend_inputs=(close, divs))

    with pytest.raises(ValueError, match=RISK):
        mod.make_target_fn({DEFENSIVE: spy, TNX: tnx},
                           dividend_inputs=(close, divs))


# ------------------------------------------------------------------ helper math

def test_month_end_flags_matches_hand_worked_calendar():
    index = pd.DatetimeIndex([
        "2021-01-28", "2021-01-29",  # Fri (month end)
        "2021-02-01", "2021-02-26",  # Fri (month end)
        "2021-03-01",                # only bar of March in this toy index
    ])
    flags = mod._month_end_flags(index)
    assert list(flags) == [False, True, False, True, False]


def test_vnq_trailing_yield_pct_matches_hand_computed_value():
    index = _bdays("2015-01-02", 5)
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index)
    dividends = pd.Series([0.0, 1.0, 0.0, 2.0, 0.0], index=index)

    yield_pct = mod.vnq_trailing_yield_pct(close, dividends, trailing_days=3)

    assert np.isnan(yield_pct.iloc[0])
    assert np.isnan(yield_pct.iloc[1])
    # window [0,1,2] -> divs sum 0+1+0=1, close[2]=102
    assert yield_pct.iloc[2] == pytest.approx(1.0 / 102.0 * 100.0)
    # window [1,2,3] -> divs sum 1+0+2=3, close[3]=103
    assert yield_pct.iloc[3] == pytest.approx(3.0 / 103.0 * 100.0)
    # window [2,3,4] -> divs sum 0+2+0=2, close[4]=104
    assert yield_pct.iloc[4] == pytest.approx(2.0 / 104.0 * 100.0)


def test_vnq_trailing_yield_pct_rejects_nonpositive_window():
    index = _bdays("2015-01-02", 3)
    close = pd.Series([100.0, 100.0, 100.0], index=index)
    divs = pd.Series([0.0, 0.0, 0.0], index=index)
    with pytest.raises(ValueError):
        mod.vnq_trailing_yield_pct(close, divs, trailing_days=0)


def test_yield_spread_series_matches_hand_computed_value_and_only_uses_common_dates():
    index = _bdays("2015-01-02", 5)
    vnq_yield = pd.Series([np.nan, np.nan, 4.0, 4.5, 5.0], index=index)
    # TNX has one extra date not shared with VNQ's yield calendar, and is
    # missing the LAST vnq-yield date -- only the intersection is used.
    tnx_index = index[:-1].append(pd.DatetimeIndex(["2015-01-12"]))
    tnx_close = pd.Series([4.0, 4.0, 4.2, 4.4, 4.6], index=tnx_index)

    spread = mod.yield_spread_series(vnq_yield, tnx_close)

    assert set(spread.index) == {index[2], index[3]}
    assert spread.loc[index[2]] == pytest.approx(4.0 - 4.2)
    assert spread.loc[index[3]] == pytest.approx(4.5 - 4.4)


def test_spread_zscore_matches_hand_computed_value_ddof1():
    index = _bdays("2015-01-02", 6)
    # window=4: first 3 z's are NaN, the 4th is defined over [1,2,3,4].
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], index=index)
    z = mod.spread_zscore(spread, window=4)

    window_vals = np.array([1.0, 2.0, 3.0, 4.0])
    expected_mean = window_vals.mean()
    expected_std = window_vals.std(ddof=1)
    expected_z = (4.0 - expected_mean) / expected_std

    assert z.iloc[:3].isna().all()
    assert z.iloc[3] == pytest.approx(expected_z)


def test_spread_zscore_zero_dispersion_is_nan_not_infinite():
    index = _bdays("2015-01-02", 5)
    spread = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0], index=index)
    z = mod.spread_zscore(spread, window=4)
    assert np.isnan(z.iloc[3])
    assert np.isnan(z.iloc[4])


def test_spread_zscore_rejects_tiny_window():
    index = _bdays("2015-01-02", 3)
    spread = pd.Series([1.0, 2.0, 3.0], index=index)
    with pytest.raises(ValueError):
        mod.spread_zscore(spread, window=1)


# ------------------------------------------------------------------ hysteresis state

def test_reit_risk_on_series_undefined_before_first_zscore_and_then_carries_forward():
    index = _bdays("2015-01-02", 10)
    # z undefined for bars 0-3, then defined (but inside the band, 0.1) for
    # every later bar; decision bars are 4 and 8 only.
    z = pd.Series([np.nan, np.nan, np.nan, np.nan,
                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1], index=index)
    decision = np.array([False, False, False, False,
                         True, False, False, False, True, False])

    state = mod.reit_risk_on_series(z, decision, entry_z=0.5, exit_z=-0.5)

    # Undefined before the z-score's own first-ever non-NaN reading (bar 4)
    assert state.iloc[:4].isna().all()
    # From bar 4 onward: never entered (0.1 is inside the band, so state
    # holds its initial SPY/False default at every decision bar) -> 0.0
    # (SPY) forward-carried through every later bar, including the
    # non-decision bars in between.
    assert (state.iloc[4:] == 0.0).all()


def test_reit_risk_on_series_stays_undefined_until_the_first_real_decision_bar():
    """The z-score's own first-ever non-NaN reading can land
    on a bar that is NOT itself a decision bar (rolling-window completion
    has nothing to do with month-end timing). The test above
    (`..._undefined_before_first_zscore_and_then_carries_forward`)
    constructs a fixture where those two dates coincide, so it does not
    cover this case: state must stay undefined (NaN) through every
    bar strictly BEFORE the first bar that is BOTH defined AND a real
    decision bar -- exposing a "real" SPY/defensive decision any earlier
    means the account would open a real
    defensive position before any real decision was ever evaluated."""
    index = _bdays("2015-01-02", 10)
    # z first becomes defined at bar 3 (a NON-decision bar); the first
    # REAL decision bar is bar 6. Bars 3-5 must stay undefined (NaN), even
    # though the code's own internal walk-forward state defaults to
    # SPY/False before any decision ever fires.
    z = pd.Series([np.nan, np.nan, np.nan, 0.1, 0.1, 0.1,
                  0.6, 0.6, 0.6, 0.6], index=index)
    decision = np.array([False, False, False, False, False, False,
                         True, False, False, False])

    state = mod.reit_risk_on_series(z, decision, entry_z=0.5, exit_z=-0.5)

    assert state.iloc[:6].isna().all(), (
        "bars 0-5 (including 3-5, where z is defined but no decision bar "
        "has fired yet) must all stay undefined -- an early 'real' "
        "decision here would open a position before any decision was made")
    # bar6: the first real decision bar, z=0.6>=0.5 -> enter VNQ (1.0),
    # carried forward through the rest.
    assert (state.iloc[6:] == 1.0).all()


def test_reit_risk_on_series_hysteresis_enter_hold_exit_hold():
    index = _bdays("2015-01-02", 6)
    # Decision bars are every index; z: cheap -> hold -> compressed -> hold
    z = pd.Series([0.6, 0.2, -0.4, -0.6, 0.1, 0.7], index=index)
    decision = np.array([True] * 6)

    state = mod.reit_risk_on_series(z, decision, entry_z=0.5, exit_z=-0.5)

    # bar0: z=0.6>=0.5 -> enter VNQ (1.0)
    # bar1: z=0.2 inside band -> hold VNQ (1.0)
    # bar2: z=-0.4 inside band (> -0.5) -> hold VNQ (1.0)
    # bar3: z=-0.6<=-0.5 -> exit to SPY (0.0)
    # bar4: z=0.1 inside band -> hold SPY (0.0)
    # bar5: z=0.7>=0.5 -> enter VNQ (1.0)
    assert list(state.to_numpy()) == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]


def test_reit_risk_on_series_nan_at_a_decision_bar_resolves_to_defensive():
    index = _bdays("2015-01-02", 4)
    z = pd.Series([0.6, np.nan, 0.6, np.nan], index=index)
    decision = np.array([True, True, True, True])
    state = mod.reit_risk_on_series(z, decision, entry_z=0.5, exit_z=-0.5)
    # bar0: enter VNQ. bar1: NaN AT a decision bar -> resolves to SPY.
    # bar2: re-enter VNQ. bar3: NaN again -> SPY.
    assert list(state.to_numpy()) == [1.0, 0.0, 1.0, 0.0]


def test_reit_risk_on_series_rejects_length_mismatch():
    index = _bdays("2015-01-02", 3)
    z = pd.Series([0.1, 0.2, 0.3], index=index)
    with pytest.raises(ValueError):
        mod.reit_risk_on_series(z, np.array([True, False]))


def test_mutation_flip_entry_exit_direction_breaks_the_selftest():
    """Deliberately sabotage the polarity of the hysteresis rule (swap
    which comparison enters vs exits), confirm the sabotaged copy disagrees
    with the correct module output on the same hand-built fixture used by
    the hysteresis test above -- proving that test actually has power to
    catch a flipped-polarity mutation. This exercises a locally
    reconstructed sabotaged copy only; the real module is never mutated in
    place."""
    index = _bdays("2015-01-02", 6)
    z = pd.Series([0.6, 0.2, -0.4, -0.6, 0.1, 0.7], index=index)
    decision = np.array([True] * 6)

    correct = mod.reit_risk_on_series(z, decision, entry_z=0.5, exit_z=-0.5)

    # Sabotaged: entry/exit comparisons swapped (>= becomes the EXIT
    # condition and <= becomes the ENTRY condition).
    values = z.to_numpy(dtype=float)
    n = len(values)
    sabotaged_raw = np.zeros(n, dtype=bool)
    is_risk_on = False
    for i in range(n):
        zz = values[i]
        if np.isnan(zz):
            is_risk_on = False
        elif zz <= -0.5:      # <-- mutated: was the ENTRY condition (>=0.5)
            is_risk_on = True
        elif zz >= 0.5:       # <-- mutated: was the EXIT condition (<=-0.5)
            is_risk_on = False
        sabotaged_raw[i] = is_risk_on
    sabotaged = pd.Series(sabotaged_raw.astype(float), index=index)

    assert not sabotaged.equals(correct), (
        "the sabotaged (flipped-polarity) series must disagree with the "
        "correct one -- if it doesn't, this test has no power to catch "
        "the mutation")
    assert sabotaged.iloc[0] != correct.iloc[0]
    assert sabotaged.iloc[3] != correct.iloc[3]


# ------------------------------------------------------------------ fetch

def test_fetch_vnq_dividend_inputs_uses_dividends_column_only_and_drops_nan_close():
    index = _bdays("2015-01-02", 4)

    def fake_fetcher(ticker):
        assert ticker == "VNQ"
        frame = pd.DataFrame({
            "Open": [90.0, 91.0, 92.0, np.nan],
            "High": [90.5, 91.5, 92.5, np.nan],
            "Low": [89.5, 90.5, 91.5, np.nan],
            "Close": [90.2, 91.2, 92.2, np.nan],   # last row still in-progress
            "Dividends": [0.0, 0.8, 0.0, 0.0],
            "Capital Gains": [0.0, 5.0, 0.0, 0.0],  # must NOT be counted
        }, index=index)
        return frame

    close, dividends = mod.fetch_vnq_dividend_inputs(fetcher=fake_fetcher)

    assert len(close) == 3, "the all-NaN in-progress final row must be dropped"
    assert list(dividends.to_numpy()) == [0.0, 0.8, 0.0]
    assert close.index.equals(dividends.index)


def test_fetch_vnq_dividend_inputs_raises_on_empty_fetch():
    with pytest.raises(ValueError):
        mod.fetch_vnq_dividend_inputs(fetcher=lambda ticker: pd.DataFrame())


def test_fetch_vnq_dividend_inputs_defaults_dividends_to_zero_when_column_absent():
    index = _bdays("2015-01-02", 3)

    def fake_fetcher(ticker):
        return pd.DataFrame({"Open": [1.0, 1.0, 1.0], "High": [1.0, 1.0, 1.0],
                             "Low": [1.0, 1.0, 1.0], "Close": [1.0, 1.0, 1.0]},
                            index=index)

    close, dividends = mod.fetch_vnq_dividend_inputs(fetcher=fake_fetcher)
    assert (dividends.to_numpy() == 0.0).all()


# ------------------------------------------------------------------ end-to-end (synthetic)

def _synthetic_warm_dataset():
    """~5.6 years of business days -- long enough to clear this module's
    real 252-session yield warm-up plus its real 756-session z-score
    window with room to spare, then hold a deliberately large, controlled
    TNX regime change well past that warm-up.

    VNQ's dividend is a PERIODIC step pattern (a payment every 65 trading
    days -- deliberately NOT a divisor of the 252-session yield window, so
    different 252-day windows catch either 3 or 4 payments depending on
    phase, giving the rolling trailing yield GENUINE non-zero variance).
    A constant daily trickle, or a period that divides 252 (e.g. every 63
    days, exactly 4 payments in every window), gives a perfectly flat
    spread with ZERO rolling variance, which ``spread_zscore``'s own
    documented zero-dispersion guard (see its docstring: "A rolling std
    below the relative tolerance floor is treated as undefined... never a
    spurious near-infinite z") treats as UNDEFINED, not "z==0 and inside
    the band" -- so no pre-jump decision would ever be made. With this
    pattern the pre-jump z-score is real and non-degenerate (measured on
    this fixture: defined well before jump_at, range roughly [-2.7, +0.4],
    never crossing the +0.5 entry threshold), so a genuine SPY/defensive
    decision is actually made and held through several real decision bars
    before the jump -- the property the tests below exercise."""
    n = 1400
    index = _bdays("2010-01-04", n)
    vnq_close = pd.Series(100.0, index=index)
    vnq_dividends = pd.Series(0.0, index=index)
    payment_dates = np.arange(20, n, 65)
    vnq_dividends.iloc[payment_dates] = 1.0

    tnx = np.full(n, 1.0)
    jump_at = 1100
    tnx[jump_at:] = -4.0   # spread jumps from ~3.0-4.0 to ~7.0-8.0 well past warm-up
    tnx_frame = _ohlc_from_close(tnx, index)

    vnq_frame = _ohlc_from_close(np.full(n, 90.0), index)   # engine fills only
    spy_frame = _ohlc_from_close(np.full(n, 400.0), index)
    frames = {RISK: vnq_frame, DEFENSIVE: spy_frame, TNX: tnx_frame}
    return index, frames, (vnq_close, vnq_dividends), jump_at


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def test_make_target_fn_returns_none_before_the_zscore_ever_warms_up():
    index = _bdays("2015-01-02", 500)   # well short of the real 1008-session warm-up
    vnq_close = pd.Series(100.0, index=index)
    vnq_dividends = pd.Series(4.0 / 252.0, index=index)
    frames = {
        RISK: _ohlc_from_close(np.full(len(index), 90.0), index),
        DEFENSIVE: _ohlc_from_close(np.full(len(index), 400.0), index),
        TNX: _ohlc_from_close(np.full(len(index), 1.0), index),
    }
    target = mod.make_target_fn(frames, dividend_inputs=(vnq_close, vnq_dividends))

    for probe in (0, 100, 250, 499):
        truncated = _truncate(frames, index[probe])
        assert target(truncated) is None


def test_make_target_fn_forward_carries_between_month_ends():
    """Proves the module docstring's central design claim: the dense,
    continuously-carried-forward hysteresis state does not react to a mid-
    month regime change until the NEXT month-end decision bar is actually
    reached -- the same forward-carry property
    defensive_sector_recession_signal_rotation.py's own sparse series
    relies on signals.make_external_regime_target_fn's as-of lookup for,
    proven here directly against THIS module's dense series."""
    index, frames, dividend_inputs, jump_at = _synthetic_warm_dataset()
    target = mod.make_target_fn(frames, dividend_inputs=dividend_inputs)

    jump_date = index[jump_at]
    # A few sessions after the jump, but still the SAME calendar month as
    # the jump -- no month-end has been reached yet, so the target must
    # still read the pre-jump regime (SPY: spread ~3.0 is inside the
    # +/-0.5 band around its own long-run rolling mean established over
    # the flat era, well below the 0.5 entry threshold).
    still_same_month = target(_truncate(frames, jump_date + pd.Timedelta(days=3)))
    assert still_same_month == DEFENSIVE, (
        "a mid-month regime change must not move the target before the "
        "next month-end decision bar")

    # Walk forward to the first REAL month-end at or after the jump.
    after_jump = pd.DatetimeIndex(index[index >= jump_date])
    month_end_mask = mod._month_end_flags(after_jump)
    first_month_end = after_jump[month_end_mask.argmax()]
    probe_date = min(first_month_end + pd.Timedelta(days=3), index[-1])
    after_month_end = target(_truncate(frames, probe_date))
    assert after_month_end == RISK, (
        "the first month-end reached after the spread widens past the "
        "entry threshold must switch the target to VNQ")


def test_make_target_fn_real_engine_end_to_end_synthetic():
    """Full wiring proof through the REAL mining.rotation_engine.
    run_two_asset_rotation (not just the raw target() callable) on the
    same controlled synthetic dataset -- confirms at least one real
    VNQ segment is opened once the spread widens."""
    index, frames, dividend_inputs, jump_at = _synthetic_warm_dataset()
    target = mod.make_target_fn(frames, dividend_inputs=dividend_inputs)
    asset_frames = {RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE]}
    signal_frames = {TNX: frames[TNX]}

    trades_df, active_asset = rotation_engine.run_two_asset_rotation(
        asset_frames, target, cost_bps_by_asset={RISK: 1.0, DEFENSIVE: 1.0},
        signal_frames=signal_frames)

    assert set(trades_df["asset"].unique()) <= {RISK, DEFENSIVE}
    assert (trades_df["asset"] == RISK).any(), (
        "the widened post-jump spread must produce at least one real VNQ "
        "segment through the real rotation engine")
    jump_date = index[jump_at]
    vnq_segments = trades_df.loc[trades_df["asset"] == RISK]
    assert (pd.to_datetime(vnq_segments["entry_date"]) >= jump_date).any()


# ------------------------- history end: second-fetch rows cannot reach the state

def _regime_state_and_targets(monkeypatch, frames, fetched, probes):
    """Build the target function over ``frames`` with the live VNQ fetch faked
    to return ``fetched`` (an unadjusted-close / dividends pair), and return the
    dense regime series handed to the target factory plus the target at each
    probe date (frames cut to end on that date)."""
    captured = {}
    real = signals.make_external_regime_target_fn

    def spy(risk_on, **kwargs):
        captured["risk_on"] = pd.Series(risk_on, copy=True)
        return real(risk_on, **kwargs)

    monkeypatch.setattr(signals, "make_external_regime_target_fn", spy)
    monkeypatch.setattr(mod, "fetch_vnq_dividend_inputs",
                        lambda fetcher=None: fetched)
    target = mod.make_target_fn(frames)
    return captured["risk_on"], [target(_truncate(frames, day)) for day in probes]


def test_second_fetch_rows_dated_after_a_history_end_never_reach_the_regime_state(
        monkeypatch):
    """The scored history can be ended early (the opt-in --history-end setting): the driver trims its frames in memory, but this
    module's live VNQ fetch is a SECOND vendor call that still returns later
    rows. Only the intersection of the fetched yield with the (trimmed) ^TNX
    frame reaches the spread, the z-score and the month-end calendar, so rows
    dated after the cutoff must change nothing. Poison rows (a tiny close, huge
    dividends) are appended after a MID-month cutoff and the whole regime series
    and the target at every date up to the cutoff must be identical. The control
    leaves ^TNX running past the cutoff: then the same poison does reach the
    regime series, so a future change that let post-cutoff second-fetch rows in
    would fail this test."""
    n_total, n_post = 1400, 60
    index = _bdays("2010-01-04", n_total)
    mid_month = (index >= index[1200]) & (index.day >= 10) & (index.day <= 20)
    cut = int(np.argmax(mid_month))
    cutoff = index[cut]
    assert cut > 1100 and index[cut + 1].month == cutoff.month
    assert not mod._month_end_flags(index)[cut], "the cutoff must be mid-month"

    kept, post = index[:cut + 1], index[cut + 1:cut + 1 + n_post]
    clean_close = pd.Series(100.0, index=kept)
    clean_dividends = pd.Series(0.0, index=kept)
    clean_dividends.iloc[20::65] = 1.0                       # a real, varying yield
    poison_close = pd.concat([clean_close, pd.Series(100.0 * 0.02, index=post)])
    poison_dividends = pd.concat([clean_dividends, pd.Series(500.0, index=post)])
    clean = (clean_close, clean_dividends)
    poisoned = (poison_close, poison_dividends)

    def frames_with_tnx(tnx_index):
        return {RISK: _ohlc_from_close(np.full(len(kept), 90.0), kept),
                DEFENSIVE: _ohlc_from_close(np.full(len(kept), 400.0), kept),
                TNX: _ohlc_from_close(np.full(len(tnx_index), 1.0), tnx_index)}

    trimmed = frames_with_tnx(kept)                     # what --history-end hands over
    untrimmed = frames_with_tnx(index[:cut + 1 + n_post])   # ^TNX runs past the cutoff
    probes = list(kept)                                  # every date up to the cutoff

    state_clean, targets_clean = _regime_state_and_targets(
        monkeypatch, trimmed, clean, probes)
    state_poison, targets_poison = _regime_state_and_targets(
        monkeypatch, trimmed, poisoned, probes)

    # non-vacuous: real decisions exist up to the cutoff
    assert state_clean.notna().sum() > 100
    assert {RISK, DEFENSIVE} & set(targets_clean)
    # the guarantee: post-cutoff second-fetch rows change nothing
    assert state_poison.index.max() <= cutoff
    pd.testing.assert_series_equal(state_poison, state_clean)
    assert targets_poison == targets_clean

    # power: with ^TNX running past the cutoff the very same poison DOES reach
    # the regime series (dates after the cutoff, and a flipped state there)
    control_clean, _ = _regime_state_and_targets(
        monkeypatch, untrimmed, clean, probes[-1:])
    control_poison, _ = _regime_state_and_targets(
        monkeypatch, untrimmed, poisoned, probes[-1:])
    pd.testing.assert_series_equal(control_clean, state_clean)   # clean: no leak
    assert not control_poison.equals(control_clean)
    later = control_poison.loc[control_poison.index > cutoff]
    assert len(later) > 0 and (later == 1.0).any(), "the poison must reach the regime series"
    pd.testing.assert_series_equal(control_poison.loc[:cutoff], state_clean)
