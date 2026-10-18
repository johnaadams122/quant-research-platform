import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from mining import options_vol, constants


def test_dividend_yield_for_known_ticker():
    assert options_vol.dividend_yield_for("SPY") == pytest.approx(0.013)


def test_dividend_yield_for_unknown_ticker_defaults_zero():
    assert options_vol.dividend_yield_for("ZZZZ_NOT_A_REAL_TICKER") == 0.0


def test_implied_vol_for_index_tier_uses_real_vol_index(monkeypatch):
    dates = pd.date_range("2020-01-02", periods=5, freq="B")
    price_df = pd.DataFrame({"Close": [100.0, 101, 102, 101, 103]}, index=dates)

    def fake_fetch_iv_proxy(ticker, start="max"):
        assert ticker == "^VIX"
        return pd.DataFrame({"Date": dates, "^VIX": [0.20, 0.21, 0.19, 0.22, 0.20]})

    monkeypatch.setattr(options_vol.data_pipeline, "fetch_iv_proxy", fake_fetch_iv_proxy)
    result = options_vol.implied_vol_for("SPY", price_df)
    assert list(result) == pytest.approx([0.20, 0.21, 0.19, 0.22, 0.20])


def test_implied_vol_for_index_tier_forward_fills_gaps(monkeypatch):
    dates = pd.date_range("2020-01-02", periods=5, freq="B")
    vix_dates = dates[[0, 2, 4]]  # a vendor gap on days 1 and 3

    def fake_fetch_iv_proxy(ticker, start="max"):
        return pd.DataFrame({"Date": vix_dates, "^VIX": [0.20, 0.22, 0.24]})

    monkeypatch.setattr(options_vol.data_pipeline, "fetch_iv_proxy", fake_fetch_iv_proxy)
    price_df = pd.DataFrame({"Close": [100.0] * 5}, index=dates)
    result = options_vol.implied_vol_for("SPY", price_df)
    assert result.iloc[1] == pytest.approx(0.20)  # ffilled from day 0
    assert result.iloc[3] == pytest.approx(0.22)  # ffilled from day 2


def test_implied_vol_for_single_stock_tier_uses_realized_vol_times_multiplier():
    dates = pd.date_range("2020-01-02", periods=30, freq="B")
    closes = [100.0 + i * 0.1 for i in range(30)]
    price_df = pd.DataFrame({"Close": closes}, index=dates)
    result = options_vol.implied_vol_for("AAPL", price_df)
    assert pd.isna(result.iloc[0])  # 21-session realized-vol warmup
    assert result.iloc[-1] > 0
    manual_returns = pd.Series(closes, index=dates).pct_change()
    manual_realized = manual_returns.rolling(21).std() * np.sqrt(252)
    expected_last = manual_realized.iloc[-1] * constants.OPTIONS_IV_REALIZED_VOL_MULTIPLIER
    assert result.iloc[-1] == pytest.approx(expected_last)


def test_implied_vol_for_raises_on_empty_index_fetch(monkeypatch):
    monkeypatch.setattr(options_vol.data_pipeline, "fetch_iv_proxy",
                         lambda ticker, start="max": None)
    price_df = pd.DataFrame({"Close": [100.0, 101.0]},
                             index=pd.date_range("2020-01-02", periods=2, freq="B"))
    with pytest.raises(ValueError, match="no data returned"):
        options_vol.implied_vol_for("SPY", price_df)


def test_options_underlying_tier_index_etf():
    assert options_vol.options_underlying_tier("SPY") == "index_etf"
    assert options_vol.options_underlying_tier("QQQ") == "index_etf"


def test_options_underlying_tier_mega_cap():
    assert options_vol.options_underlying_tier("AAPL") == "mega_cap"


def test_options_underlying_tier_raises_outside_whitelist():
    with pytest.raises(ValueError, match="OPTIONS_LIQUID_UNDERLYINGS"):
        options_vol.options_underlying_tier("GME")
