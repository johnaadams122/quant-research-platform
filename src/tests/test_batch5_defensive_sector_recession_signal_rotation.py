"""Tests for src/mining/batch5/defensive_sector_recession_signal_rotation.py.

This family implements its catalogued mechanism in the batch5 module
contract -- see the module's own docstring. The
PARAMETERS drift check here compares against the already-existing
data/mining/catalog.json entry, read directly, matching e.g. test_batch5_usd_strength_risk_regime.py's own
identical convention.

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

import data_pipeline  # noqa: E402
import run_batch3_mining as batch3  # noqa: E402
from mining import abi1_driver, rotation_engine, signals  # noqa: E402
from mining.batch5 import defensive_sector_recession_signal_rotation as mod  # noqa: E402

FAMILY_ID = "defensive_sector_recession_signal_rotation"
RISK, DEFENSIVE = "SPY", "XLU"
TNX, IRX = "^TNX", "^IRX"
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


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _random_frame(index: pd.DatetimeIndex, seed: int, level: float = 100.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    steps = rng.normal(0.0002, 0.01, len(index))
    return _ohlc_from_close(level * np.exp(np.cumsum(steps)), index)


def _synthetic_frames(index: pd.DatetimeIndex) -> dict:
    return {
        RISK: _random_frame(index, seed=1, level=100.0),
        DEFENSIVE: _random_frame(index, seed=2, level=80.0),
        TNX: _ohlc_from_close(np.full(len(index), 4.5), index),
        IRX: _ohlc_from_close(np.full(len(index), 4.0), index),
    }


def _truncate(frames: dict, through: pd.Timestamp) -> dict:
    return {key: frame.loc[frame.index <= through].copy(deep=True)
            for key, frame in frames.items()}


def _catalog_entry() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    matches = [c for c in catalog if c["family_id"] == FAMILY_ID]
    assert len(matches) == 1, (
        f"expected exactly one catalog.json entry for {FAMILY_ID}, found "
        f"{len(matches)}")
    return matches[0]


# -------------------------------------------------------------- contract

def test_module_constants_match_the_family_contract():
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE]
    assert mod.SIGNAL == [TNX, IRX]
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
    assert entry["parameters"]["short_yield_ticker"] == IRX
    assert entry["parameters"]["long_yield_ticker"] == TNX


def test_driver_discovery_validates_this_module_in_isolation(tmp_path):
    """Discovery on a private copy of the module (other modules in the
    real batch5 dir may change independently, so the real dir is not
    asserted on)."""
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
    assert spec["signal"] == [TNX, IRX]
    assert spec["benchmark"] == "SPY"
    assert spec["expected_min_trades"] == mod.EXPECTED_MIN_TRADES
    assert batch5.parameter_drift(
        spec["parameters"], _catalog_entry()["parameters"]) is None


# ------------------------------------------------------------------ factory

def test_make_target_fn_requires_every_traded_and_signal_frame():
    index = _bdays("2015-01-02", "2016-12-30")
    frames = _synthetic_frames(index)

    with pytest.raises(ValueError, match=DEFENSIVE):
        mod.make_target_fn({RISK: frames[RISK], TNX: frames[TNX], IRX: frames[IRX]})

    with pytest.raises(ValueError, match=r"\^TNX"):
        mod.make_target_fn({RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE], IRX: frames[IRX]})

    with pytest.raises(ValueError, match=r"\^IRX"):
        mod.make_target_fn({RISK: frames[RISK], DEFENSIVE: frames[DEFENSIVE], TNX: frames[TNX]})

    with pytest.raises(ValueError, match=RISK):
        mod.make_target_fn({DEFENSIVE: frames[DEFENSIVE], TNX: frames[TNX], IRX: frames[IRX]})


def test_month_end_flags_matches_hand_worked_calendar():
    index = pd.DatetimeIndex([
        "2021-01-28", "2021-01-29",  # Fri (month end)
        "2021-02-01", "2021-02-26",  # Fri (month end)
        "2021-03-01",                # only bar of March in this toy index
    ])
    flags = mod._month_end_flags(index)
    assert list(flags) == [False, True, False, True, False]


def test_curve_risk_on_series_only_defined_post_warmup_and_at_month_end():
    """A flat spread means the series has 4 years of daily bars; warm-up is
    curve_lookback_days=20 sessions. Confirm every defined entry lands
    exactly on a month-end date of the tnx/irx common calendar, and that
    entries exist only once warm-up has completed."""
    index = _bdays("2015-01-02", "2018-12-31")
    tnx_close = pd.Series(4.5, index=index)
    irx_close = pd.Series(4.0, index=index)
    risk_on = mod.curve_risk_on_series(tnx_close, irx_close)

    defined = risk_on.dropna()
    month_end_mask = mod._month_end_flags(index)
    month_end_dates = set(index[month_end_mask])
    assert set(defined.index) <= month_end_dates
    # First candidate month-end is 2015-01-30 -- well past the 20-session
    # warm-up (curve_lookback_days) started 2015-01-02, so it must already
    # be defined.
    assert pd.Timestamp("2015-01-30") in defined.index
    # A constant positive spread must select the risk leg at every defined
    # month-end.
    assert (defined == 1.0).all()


def test_inverted_curve_selects_defensive_and_normal_curve_selects_risk():
    """Hand-constructed TNX/IRX paths: flat/normal (TNX > IRX) for 4 years,
    then TNX drops below IRX for the final ~2 months so the smoothed spread
    at the final month-end is negative -- directly checks the catalog's own
    stated polarity ('when the spread inverts ... switch into XLU ...
    a positively-sloped curve holds SPY instead'), not just internal
    self-consistency."""
    index = _bdays("2015-01-02", "2018-12-31")
    n = len(index)
    tnx = np.full(n, 4.5)
    irx = np.full(n, 4.0)
    tnx[-40:] = 3.0  # last ~2 months: TNX well below IRX -> inverted
    tnx_close = pd.Series(tnx, index=index)
    irx_close = pd.Series(irx, index=index)

    risk_on = mod.curve_risk_on_series(tnx_close, irx_close)
    defined = risk_on.dropna()

    normal_era = defined.loc[:defined.index[-2]]
    assert (normal_era == 1.0).all(), "normal (TNX>IRX) curve must select SPY throughout"
    assert defined.iloc[-1] == 0.0, (
        "the final month-end's smoothed spread must be inverted and select XLU")


def test_make_target_fn_forward_carries_between_month_ends():
    """Proves the module docstring's central design claim: a sparse,
    month-end-only risk_on series is ALREADY forward-carried across every
    intra-month day by signals.make_external_regime_target_fn's own as-of
    lookup, with no additional restriction needed inside this module. Build
    a path where the curve flips from normal to inverted mid-month; the
    decision must not change until the NEXT month-end is reached."""
    index = _bdays("2015-01-02", "2015-06-30")
    n = len(index)
    tnx = np.full(n, 4.5)
    irx = np.full(n, 4.0)
    # Flip mid-April (well inside the month, not at a month-end) through
    # the rest of the series.
    flip_pos = index.get_loc(pd.Timestamp("2015-04-15"))
    tnx[flip_pos:] = 3.0
    frames = {
        RISK: _random_frame(index, seed=5, level=100.0),
        DEFENSIVE: _random_frame(index, seed=6, level=80.0),
        TNX: _ohlc_from_close(tnx, index),
        IRX: _ohlc_from_close(irx, index),
    }
    target = mod.make_target_fn(frames)

    # The day right after the mid-month flip: still April, curve inverted
    # intraday, but the LAST month-end (March) was still normal -> must
    # still read SPY (the flip has not yet reached a month-end).
    still_march_decision = target(_truncate(frames, index[flip_pos + 1]))
    assert still_march_decision == RISK, (
        "an intra-month spread flip must not change the target before the "
        "next month-end decision bar")

    # April's own month-end (2015-04-30): the smoothed spread by then
    # reflects the flip (curve_lookback_days=20 sessions have elapsed since
    # 2015-04-15) -> must switch to XLU.
    april_month_end = pd.Timestamp("2015-04-30")
    after_month_end = target(_truncate(frames, april_month_end + pd.Timedelta(days=3)))
    assert after_month_end == DEFENSIVE, (
        "the first month-end reached after the flip must adopt the new "
        "regime reading")


def test_no_positivity_required_for_signal_close_values():
    """The module docstring's "Real measurement 3" (negative signal yields):
    by design, this module's own code must
    tolerate a real, economically genuine negative yield print (unlike
    run_batch3_mining._normalize_ohlc's blanket equity-oriented positivity
    gate, which this module never calls on TNX/IRX)."""
    index = _bdays("2015-01-02", "2016-12-30")
    frames = _synthetic_frames(index)
    tnx = frames[TNX].copy()
    tnx.loc[tnx.index[-5]:, "Close"] = -0.05  # a genuine-shape negative print
    tnx.loc[tnx.index[-5]:, "Open"] = -0.04
    frames[TNX] = tnx

    target = mod.make_target_fn(frames)  # must not raise
    result = target(_truncate(frames, index[-1]))
    assert result in (RISK, DEFENSIVE, None)


def test_mutation_flip_direction_breaks_the_selftest():
    """Deliberately sabotage the polarity of curve_risk_on_series (flip the
    comparison), confirm the module's own polarity test above would fail
    against it, then leave the real module untouched (this test only
    exercises a locally-reconstructed sabotaged copy of the decision rule,
    never mutates the imported module in place -- that would corrupt every
    other test in this same process)."""
    index = _bdays("2015-01-02", "2018-12-31")
    n = len(index)
    tnx = np.full(n, 4.5)
    irx = np.full(n, 4.0)
    tnx[-40:] = 3.0
    tnx_close = pd.Series(tnx, index=index)
    irx_close = pd.Series(irx, index=index)

    correct = mod.curve_risk_on_series(tnx_close, irx_close).dropna()

    # Sabotaged copy: flip > to < (a "flip a comparison" mutation).
    common = tnx_close.index.intersection(irx_close.index).sort_values()
    daily_spread = tnx_close.reindex(common) - irx_close.reindex(common)
    smoothed = daily_spread.rolling(
        mod._CURVE_LOOKBACK_DAYS, min_periods=mod._CURVE_LOOKBACK_DAYS).mean()
    month_end = mod._month_end_flags(common)
    defined = smoothed.notna().to_numpy() & month_end
    sabotaged_risk_on = (smoothed < 0.0).to_numpy()  # <-- mutated: was > 0.0
    sabotaged = pd.Series(
        np.where(defined, sabotaged_risk_on.astype(float), np.nan),
        index=common).dropna()

    assert not sabotaged.equals(correct), (
        "the sabotaged (flipped-comparison) series must disagree with the "
        "correct one -- if it doesn't, this test has no power to catch the "
        "mutation")
    assert sabotaged.iloc[-1] != correct.iloc[-1]


