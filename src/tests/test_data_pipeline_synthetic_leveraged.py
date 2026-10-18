"""Tests for src/data_pipeline.py's build_synthetic_leveraged -- a thin sibling of build_synthetic_3x carrying
single-stock-appropriate defaults. No new math: these tests confirm
correct delegation and naming, not the underlying formula (which
build_synthetic_3x already owns and which this wrapper does not touch).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import data_pipeline as dp  # noqa: E402


def _price_df(n: int = 300, start: str = "2020-01-02", seed: int = 7) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    return pd.DataFrame({"Date": dates, "TSLA": close})


def test_output_naming_uses_2x_not_3x():
    df = _price_df()
    out = dp.build_synthetic_leveraged(df, "TSLA")
    assert "TSLA_2x_synthetic" in out.columns
    assert "TSLA_3x_synthetic" not in out.columns
    assert list(out.columns) == ["Date", "TSLA_2x_synthetic"]
    assert len(out) == len(df)


def test_delegates_byte_identically_to_build_synthetic_3x_with_same_kwargs():
    """No new math -- calling build_synthetic_3x directly with this
    wrapper's own stated defaults must reproduce its output exactly."""
    df = _price_df()
    wrapped = dp.build_synthetic_leveraged(df, "TSLA")
    direct = dp.build_synthetic_3x(
        df, "TSLA", leverage=2.0, annual_expense_ratio=0.0,
        annual_financing_rate=None, financing_spread=0.0100,
        dividend_yield=0.0, trading_days=252)
    pd.testing.assert_frame_equal(wrapped, direct)


def test_defaults_differ_from_build_synthetic_3x_own_index_etf_defaults():
    """The whole point of the wrapper: different DEFAULTS, not different
    math. Confirm the single-stock defaults actually diverge from
    build_synthetic_3x's own (0.88% ER / 50bps spread) when neither
    caller passes them explicitly."""
    df = _price_df()
    single_stock_default = dp.build_synthetic_leveraged(df, "TSLA")
    index_etf_default = dp.build_synthetic_3x(df, "TSLA", leverage=2.0)
    # Same leverage, different ER/spread defaults -> different price paths.
    assert not single_stock_default["TSLA_2x_synthetic"].equals(
        index_etf_default["TSLA_2x_synthetic"])


def test_constant_financing_rate_path():
    df = _price_df()
    out = dp.build_synthetic_leveraged(df, "TSLA", annual_financing_rate=0.05)
    assert out["TSLA_2x_synthetic"].notna().all()
    assert (out["TSLA_2x_synthetic"] > 0).all()


def test_time_varying_financing_rate_series_path():
    df = _price_df()
    rate_series = pd.Series(
        np.full(len(df), 0.04), index=pd.to_datetime(df["Date"]))
    out = dp.build_synthetic_leveraged(
        df, "TSLA", annual_financing_rate=rate_series)
    assert out["TSLA_2x_synthetic"].notna().all()
    assert (out["TSLA_2x_synthetic"] > 0).all()


def test_leverage_kwarg_still_overridable():
    """The wrapper's own default is 2.0, but it must not hardcode leverage
    -- a future caller wanting a different multiple must still be able to
    pass one through."""
    df = _price_df()
    out = dp.build_synthetic_leveraged(df, "TSLA", leverage=1.5)
    assert "TSLA_1x_synthetic" in out.columns or "TSLA_1_5x_synthetic" not in out.columns
    # int(1.5) == 1 per build_synthetic_3x's own naming convention -- this
    # assertion documents that inherited behavior, not a new choice here.
    assert list(out.columns) == ["Date", "TSLA_1x_synthetic"]


def test_rejects_non_positive_prices_same_as_build_synthetic_3x():
    df = _price_df()
    df.loc[5, "TSLA"] = 0.0
    with pytest.raises(ValueError, match="zero or negative"):
        dp.build_synthetic_leveraged(df, "TSLA")
