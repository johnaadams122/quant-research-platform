"""Volatility and dividend-yield inputs for the options-chain simulator.

No IV term structure: sigma comes directly from data_pipeline's
fetch_iv_proxy/rolling_realized_vol helpers. A 30-45 DTE cycle's tenor is close enough to
VIX/VXN's own 30-day tenor that using them directly as sigma avoids the
"phantom premium" mismatch a 30-day index causes when it prices a much
shorter-dated option.
"""
from __future__ import annotations

import pandas as pd

import data_pipeline
from . import constants

# Index-tier underlyings get a dedicated vol INDEX (a real yfinance
# ticker). Anything not in this map falls back to a realized-vol proxy.
TICKER_VOL_INDEX = {
    "SPY": "^VIX",
    "QQQ": "^VXN",
}

# Static per-ticker dividend yield table. A live yfinance .info lookup
# applied uniformly across all of history would be a lookahead leak
# (today's yield applied to 1995 pricing) -- same convention as
# data_pipeline's SOX-synthetic 0.012 hardcode. The values are
# hand-maintained approximations, changed only by a deliberate commit.
DIVIDEND_YIELD_TABLE = {
    "SPY": 0.013, "QQQ": 0.006, "IWM": 0.012, "DIA": 0.018,
    "AAPL": 0.005, "MSFT": 0.008, "AMZN": 0.0, "GOOGL": 0.0,
    "META": 0.003, "NVDA": 0.0003,
}


def dividend_yield_for(ticker: str) -> float:
    """Static dividend yield for `ticker`, from DIVIDEND_YIELD_TABLE. 0.0
    for any ticker not in the table (matches data_pipeline's own
    QQQ/SPY=0.0 total-return-ETF default)."""
    return DIVIDEND_YIELD_TABLE.get(ticker, 0.0)


def implied_vol_for(ticker: str, price_df: pd.DataFrame) -> pd.Series:
    """Daily decimal IV series for `ticker`, aligned to `price_df.index`
    (`price_df` is a standard mining OHLC frame -- a "Close" COLUMN, not
    a ticker-named one).

    Index-tier names (TICKER_VOL_INDEX) use the real vol INDEX (VIX for
    SPY, VXN for QQQ) via data_pipeline.fetch_iv_proxy, reindexed onto
    `price_df.index` and forward-filled (a vendor gap means a data hole,
    not a real closed day -- VIX/VXN publish on the same NYSE calendar as
    the underlying).

    Everything else uses `rolling_realized_vol(..., window=21) *
    constants.OPTIONS_IV_REALIZED_VOL_MULTIPLIER`. rolling_realized_vol
    needs a column literally named `ticker` (data_pipeline's own "wide
    corpus" convention), which a mining OHLC frame does not have --
    adapted at the call site (same idiom as
    run_synthetic_leveraged_dip_mining._build_underlying_and_synthetic's
    own Date-plus-named-column shape adapter) rather than changing that
    function. The first 21 sessions of the realized-vol path are NaN
    (rolling_realized_vol's own warmup).
    """
    if ticker in TICKER_VOL_INDEX:
        vol_ticker = TICKER_VOL_INDEX[ticker]
        raw = data_pipeline.fetch_iv_proxy(vol_ticker, start="max")
        if raw is None or raw.empty:
            raise ValueError(f"implied_vol_for: no data returned for {vol_ticker}")
        series = raw.set_index("Date")[vol_ticker]
        series.index = pd.to_datetime(series.index)
        return series.reindex(price_df.index, method="ffill")

    close_wide = pd.DataFrame({ticker: price_df["Close"].to_numpy()}, index=price_df.index)
    realized = data_pipeline.rolling_realized_vol(close_wide, ticker, window=21)
    return realized * constants.OPTIONS_IV_REALIZED_VOL_MULTIPLIER


def options_underlying_tier(ticker: str) -> str:
    """`"index_etf"` or `"mega_cap"` per `constants.OPTIONS_UNDERLYING_TIER`
    -- raises for anything outside `constants.OPTIONS_LIQUID_UNDERLYINGS`
    (the v1 whitelist; there is no options-liquidity data source in this
    repo to derive a floor from, so this is a hand-curated call, same
    convention as any other batch5/6 curation judgment)."""
    if ticker not in constants.OPTIONS_LIQUID_UNDERLYINGS:
        raise ValueError(
            f"{ticker!r} is not in constants.OPTIONS_LIQUID_UNDERLYINGS -- "
            f"the options simulator's v1 whitelist")
    return constants.OPTIONS_UNDERLYING_TIER[ticker]
