"""Tests for src/mining/batch5/post_opex_gamma_release_leverage_window.py.

Mechanism + tests only -- this file deliberately never touches the trial
ledger or any run output. It proves the module's contract, its delegation to the pre-existing, already-
tested mining.signals.post_opex_window_dates (no new date math re-
implemented here), and that its declared PARAMETERS match the pre-
registered catalog entry byte-for-byte -- the same conventions every other
batch5 module's test file already follows.
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

from mining import catalog as catalog_mod  # noqa: E402
from mining import signals  # noqa: E402
from mining.backtest_engine import run_backtest  # noqa: E402
from mining.batch5 import (  # noqa: E402
    post_opex_gamma_release_leverage_window as mod,
)
from mining.batch5.synthetic_leveraged_dip_buy_pooled_largecap import (  # noqa: E402
    synthetic_ohlc_from_close,
)

CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")


def _catalog_entry() -> dict:
    entries = catalog_mod.load_catalog(CATALOG_PATH)
    matches = [e for e in entries if e["family_id"] == mod.FAMILY_ID]
    assert len(matches) == 1
    return matches[0]


def _flat_frame(index) -> pd.DataFrame:
    """A trivially-shaped OHLC frame usable as a synthetic-id frame in
    tests that only need a valid DATE INDEX (the opex signal is pure
    calendar arithmetic -- no price ever enters make_signal_fn)."""
    n = len(index)
    close = np.linspace(100.0, 100.0 + n - 1, n)
    return synthetic_ohlc_from_close(index, close, name="test")


# ------------------------------------------------------------- contract

def test_contract_constants():
    assert mod.FAMILY_ID == "post_opex_gamma_release_leverage_window"
    assert mod.SHAPE == "pooled_single_asset"
    assert mod.BENCHMARK == "SPY"
    assert callable(mod.make_signal_fn)
    assert len(mod.TRADED) == len(mod.SIGNAL) == 11
    assert not (set(mod.TRADED) & set(mod.SIGNAL))  # disjoint, per the contract
    assert mod.TRADED == [mod.synthetic_id_for(t) for t in mod.SIGNAL]
    assert all(mod.underlying_of(t) == u for t, u in zip(mod.TRADED, mod.SIGNAL))
    assert mod.SIGNAL == ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU",
                          "XLV", "XLY", "XLRE", "XLC"]


def test_driver_contract_validation():
    import run_batch5_mining as batch5
    spec = batch5.validate_module_contract(mod)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "pooled_single_asset"
    assert spec["benchmark"] == "SPY"


def test_underlying_of_and_synthetic_id_for_roundtrip():
    assert mod.synthetic_id_for("XLK") == "XLK_2x_synthetic"
    assert mod.underlying_of("XLK_2x_synthetic") == "XLK"
    with pytest.raises(ValueError):
        mod.underlying_of("XLK")  # not a synthetic id


# ---------------------------------------------------------- catalog drift

def test_parameters_match_catalog_byte_for_byte():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert entry["status"] == "eligible"
    assert entry["shape"] == "pooled_single_asset"
    assert entry["benchmark_ticker"] == mod.BENCHMARK
    assert entry["pooling_note"] == (
        'resolved_pool: ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV",'
        '"XLY","XLRE","XLC"]')


def test_driver_no_parameter_drift_against_catalog():
    import run_batch5_mining as batch5
    entry = _catalog_entry()
    assert batch5.parameter_drift(mod.PARAMETERS, entry["parameters"]) is None


# ------------------------------------------------------- make_signal_fn

def test_make_signal_fn_rejects_missing_traded_frame():
    index = pd.bdate_range("2024-01-01", periods=5)
    with pytest.raises(KeyError, match="no synthetic frame"):
        mod.make_signal_fn("XLK_2x_synthetic", index, {})


def test_make_signal_fn_rejects_empty_index():
    frame = _flat_frame(pd.bdate_range("2024-01-01", periods=5))
    with pytest.raises(ValueError, match="index must not be empty"):
        mod.make_signal_fn("XLK_2x_synthetic", pd.DatetimeIndex([]),
                           {"XLK_2x_synthetic": frame})


def test_make_signal_fn_delegates_to_signals_module_and_matches_it_exactly():
    """The whole point of this module is that it adds no new opex-date
    math -- it must produce entries/exits IDENTICAL to calling
    signals.post_opex_window_dates directly on the same (truncated)
    calendar, for the exact hold this module declares."""
    index = pd.bdate_range("2024-01-01", "2024-04-30")
    frame = _flat_frame(index)
    signal_fn = mod.make_signal_fn(
        "XLK_2x_synthetic", index, {"XLK_2x_synthetic": frame})

    expected_entries, expected_exits = signals.post_opex_window_dates(
        index, hold_trading_days_after_opex=mod.HOLD_TRADING_DAYS_AFTER_OPEX)
    expected_signal_fn = signals.make_date_window_signal_fn(
        expected_entries, expected_exits)

    for i in range(len(index)):
        assert (signal_fn(frame.iloc[: i + 1], i)
                == expected_signal_fn(frame.iloc[: i + 1], i))
    # And at least one real entry/exit actually fired in this window.
    assert len(expected_entries) >= 3  # Jan, Feb, Mar expiries all resolve


def test_make_signal_fn_is_holdout_blind_pre_bucket_history_never_extends_hold():
    """Truncating the frame to the bucket's own last bar must not change
    the entry/exit dates that fall WITHIN the bucket -- the exit position
    for an expiry near the bucket's end must be resolved against the
    calendar available up to and including the bucket, not a longer
    history a real driver would also supply as warm-up (mirrors
    vol_managed_sector_spdr_pooled's and long_treasury_3x_fomc_window_
    drift's own truncation-invariance convention)."""
    full_index = pd.bdate_range("2024-01-01", "2024-06-30")
    full_frame = _flat_frame(full_index)
    bucket = full_index[full_index <= pd.Timestamp("2024-04-30")]

    truncated_fn = mod.make_signal_fn(
        "XLK_2x_synthetic", bucket, {"XLK_2x_synthetic": full_frame})
    full_fn = mod.make_signal_fn(
        "XLK_2x_synthetic", full_index, {"XLK_2x_synthetic": full_frame})

    for i, date in enumerate(bucket):
        bucket_slice = full_frame.loc[full_frame.index <= date]
        assert (truncated_fn(bucket_slice, i)
                == full_fn(bucket_slice, i))


def test_make_signal_fn_end_to_end_via_real_engine_one_trade_per_expiry():
    index = pd.bdate_range("2024-01-01", "2024-04-30")
    frame = _flat_frame(index)
    signal_fn = mod.make_signal_fn(
        "XLK_2x_synthetic", index, {"XLK_2x_synthetic": frame})
    trades, _ = run_backtest(frame, signal_fn)
    # Third Fridays Jan 19 / Feb 16 / Mar 15 / Apr 19 all have >= 5 actual
    # trading sessions of calendar remaining after them within the supplied
    # index (which runs through 2024-04-30), so all four resolve.
    assert len(trades) == 4
    entry_dates = sorted(trades["entry_date"].tolist())
    exit_dates = sorted(trades["exit_date"].tolist())
    assert entry_dates[0] == pd.Timestamp("2024-01-22")  # next-open after Jan 19
    assert exit_dates[0] == pd.Timestamp("2024-01-26")   # 5 sessions after Jan 19
    for entry, exit_ in zip(entry_dates, exit_dates):
        assert exit_ > entry


# ------------------------------------------------------- distinctness

def test_never_shares_family_id_or_pool_signal_with_vol_managed_sibling():
    from mining.batch5 import vol_managed_sector_spdr_pooled as sibling
    assert mod.FAMILY_ID != sibling.FAMILY_ID
    assert set(mod.SIGNAL) == set(sibling.TRADED)  # same panel, disclosed
    # The two modules' signals must not be computed by the same function.
    assert mod.make_signal_fn is not sibling.make_signal_fn
    # Sanity: this family's parameters contain no realized-variance/
    # threshold keys at all -- confirms the signal really is calendar-only.
    for key in ("rv_lookback_days", "threshold", "threshold_min_obs"):
        assert key not in mod.PARAMETERS
