"""
data_pipeline.py
----------------
Fetches historical adjusted closing prices and builds synthetic 3x leveraged
ETF series with realistic, time-varying cost drag.

TICKERS AND DATA COVERAGE:
  Underlying indexes/ETFs (source for synthetics):
    QQQ  Mar 1999  (~27 yr)  Nasdaq-100 ETF              -> mimics TQQQ  (ER 0.88%)
    SOX  Jun 1993  (~33 yr)  PHLX Semiconductor Index    -> mimics SOXL  (ER 0.76%)
                              (fetched as ^SOX via TICKER_MAP; price-return index)
    SPY  Jan 1993  (~33 yr)  S&P 500 ETF                 -> mimics SPXL  (ER 0.91%)

  Actual 3x ETFs (validation):
    TQQQ  Feb 2010   SOXL  Mar 2010   SPXL  Nov 2008

  NOTE on SOX vs SOXX:
    SOXL tracks the PHLX Semiconductor Sector Index (^SOX) directly.
    SOXX (iShares ETF) tracks a similar but distinct index with different weights,
    causing up to 16% single-day divergence on volatile days. ^SOX is the
    correct underlying for SOXL validation.

COST MODEL — time-varying:
  r_synth_t = lambda * r_t  -  ER/252  -  (lambda-1) * r_f_t/252

  r_f_t is the daily 13-week T-Bill rate (^IRX) + financing spread (50 bps).
  A time-varying rate matters: a fixed rate near its long-run average
  over-states the financing drag during the 2010-2021 near-zero-rate period.

TAYLOR SERIES DECOMPOSITION:
  log(1 + lambda*r - c) expanded around 0:

    = (lambda*r - c)                          1st order: trend capture
    - (lambda*r - c)^2 / 2                    2nd order: vol drag  (always negative)
    + (lambda*r - c)^3 / 6                    3rd order: skew correction
    - (lambda*r - c)^4 / 24                   4th order: kurtosis correction
    + ...

  Keeping 1st + 2nd order (ignoring c which is tiny):
    log(1 + lambda*r) approx= lambda*r - lambda^2 * r^2 / 2

  Annualizing the 2nd-order term:
    vol_drag approx= -lambda^2 / 2 * sigma^2   (lambda=3 -> -4.5 * sigma^2)

  Outperformance condition (3x beats 1x geometrically, ignoring cost):
    E[log L] > E[log S]
    3*mu - 9*sigma^2/2  >  mu - sigma^2/2
    2*mu > 4*sigma^2
    mu > 2*sigma^2          <- the core thesis of this project

  The 3rd-order term (-lambda^3 * skewness / 6) shows that negative skew
  (equity crash asymmetry) further penalizes leveraged positions.

  See taylor_vol_drag() for the rolling regime decomposition.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS = ["QQQ", "TQQQ", "SOX", "SOXL", "SPY", "SPXL", "DIA", "UDOW"]

START_DATE = "max"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Fallback annual financing rate when ^IRX data is unavailable
FINANCING_RATE_FALLBACK = 0.025

# yfinance uses "^" prefix for indexes. Map clean display names -> yfinance tickers.
# fetch_ticker() uses this so data is stored/keyed by the clean name (e.g. "SOX"),
# not the yfinance symbol ("^SOX").
TICKER_MAP = {
    "SOX": "^SOX",   # PHLX Semiconductor Sector Index — direct benchmark of SOXL
    "BXM": "^BXM",   # Cboe S&P 500 BuyWrite Index, 1988-06-01 -> present (yfinance)
    "PUT": "^PUT",   # Cboe S&P 500 PutWrite Index, 1996-08-02 -> present (yfinance)
}

# One config per synthetic series.
# financing_spread: basis-point premium above ^IRX/SOFR (repo/swap spread).
# Large leveraged ETFs (TQQQ/SOXL/SPXL) typically pay SOFR + 20-50 bps.
SYNTHETIC_CONFIGS = [
    {
        "underlying": "QQQ",
        "real_etf": "TQQQ",
        "leverage": 3.0,
        "annual_expense_ratio": 0.0088,
        "financing_spread": 0.005,
        "trading_days": 252,
    },
    {
        "underlying": "SOX",
        "real_etf": "SOXL",
        "leverage": 3.0,
        "annual_expense_ratio": 0.0076,
        "financing_spread": 0.005,
        # ^SOX is a price-return index — dividends excluded. SOXL's swaps capture
        # dividends, so we add an estimated annual yield to the underlying return.
        # Semiconductor index yield ~1.0-1.5%/yr historically; 1.2% is a mid estimate.
        # Tune this to minimize the CAGR gap vs SOXL on the 2010-2026 overlap.
        "dividend_yield": 0.012,
        "trading_days": 252,
    },
    {
        "underlying": "SPY",
        "real_etf": "SPXL",
        "leverage": 3.0,
        "annual_expense_ratio": 0.0091,
        "financing_spread": 0.005,
        "trading_days": 252,
    },
    {
        # Dow Jones Industrial Average: DIA (SPDR Dow ETF, total-return like SPY
        # -- auto_adjust folds dividends into the close, so NO dividend_yield
        # add-back) -> mimics UDOW (ProShares UltraPro Dow30, 3x, ER ~0.95%).
        # VALIDATE the CAGR gap vs UDOW on the 2010-2026 overlap when the raw
        # data is fetched (as SPY/QQQ/SOX were); tune financing_spread if needed.
        "underlying": "DIA",
        "real_etf": "UDOW",
        "leverage": 3.0,
        "annual_expense_ratio": 0.0095,
        "financing_spread": 0.005,
        "trading_days": 252,
    },
]


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, start: str = "max") -> pd.DataFrame:
    """
    Download adjusted close price history for a single ticker.

    Looks up TICKER_MAP first so clean display names (e.g. "SOX") are resolved
    to their yfinance symbols (e.g. "^SOX"). The returned DataFrame always uses
    the clean display name as its price column and key in results{}.
    """
    yf_symbol = TICKER_MAP.get(ticker, ticker)
    print(f"  Fetching {ticker} ({'=' if ticker == yf_symbol else yf_symbol})...")
    try:
        raw = yf.Ticker(yf_symbol).history(period=start, auto_adjust=True)
        if raw.empty:
            print(f"  WARNING: No data returned for {yf_symbol}.")
            return None
        df = raw[["Close"]].reset_index()
        df = df.rename(columns={"Close": ticker})   # store under clean display name
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        print(f"  OK {ticker}: {len(df)} days | {df['Date'].min()} -> {df['Date'].max()}")
        return df
    except Exception as e:
        print(f"  ERROR fetching {ticker} ({yf_symbol}): {e}")
        return None


def fetch_ohlc(ticker: str, start: str = "max") -> pd.DataFrame:
    """
    Download Open/High/Low/Close history for a single ticker.

    Used for the IC backtest's intraday exit model (simulate_trade revalues
    the iron condor at the day's High and Low). Unlike fetch_ticker(), which
    keeps only Close, this keeps all four OHLC columns.
    """
    yf_symbol = TICKER_MAP.get(ticker, ticker)
    print(f"  Fetching OHLC {ticker} ({'=' if ticker == yf_symbol else yf_symbol})...")
    try:
        raw = yf.Ticker(yf_symbol).history(period=start, auto_adjust=True)
        if raw.empty:
            print(f"  WARNING: No OHLC data returned for {yf_symbol}.")
            return None
        df = raw[["Open", "High", "Low", "Close"]].reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        print(f"  OK {ticker} OHLC: {len(df)} days | {df['Date'].min()} -> {df['Date'].max()}")
        return df
    except Exception as e:
        print(f"  ERROR fetching OHLC {ticker} ({yf_symbol}): {e}")
        return None


def fetch_iv_proxy(ticker: str, start: str = "max") -> pd.DataFrame:
    """
    Fetch a volatility index (^VIX or ^VXN) as a decimal IV series.

    yfinance returns these in vol points (e.g. 20.5 = 20.5%); divide by 100
    so the value can be passed directly as Black-Scholes sigma.
    """
    print(f"  Fetching {ticker}...")
    try:
        raw = yf.Ticker(ticker).history(period=start, auto_adjust=True)
        if raw.empty:
            print(f"  WARNING: No data returned for {ticker}.")
            return None
        df = raw[["Close"]].reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.rename(columns={"Close": ticker})
        df[ticker] = df[ticker] / 100.0
        print(f"  OK {ticker}: {len(df)} days | {df['Date'].min()} -> {df['Date'].max()} | "
              f"range [{df[ticker].min()*100:.1f}, {df[ticker].max()*100:.1f}]")
        return df
    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")
        return None


def fetch_vix1d(start: str = "max") -> pd.DataFrame:
    """
    Fetch CBOE's 1-day expected-volatility index (^VIX1D) as a decimal IV series.

    VIX1D is the same-day (0DTE-horizon) analogue of the 30-day VIX, published by
    CBOE since 2023-04-24 (yfinance coverage begins on that date). It is the
    correct Black-Scholes sigma for a 0DTE SPX/SPY iron condor — the 30-day VIX
    over-states a sub-day option's premium (a "phantom premium" when pricing
    same-day options). Returned in decimal form
    (divide by 100), identical convention to fetch_iv_proxy().

    There is no published 1-day Nasdaq-100 index; QQQ's same-day IV can be imputed
    as VIX1D * (VXN / VIX) (cross-sectional vol-ratio scaling).
    """
    return fetch_iv_proxy("^VIX1D", start)


def rolling_realized_vol(df: pd.DataFrame, ticker: str, window: int = 21) -> pd.Series:
    """
    Annualized realized volatility from close-to-close returns.

    Companion series to the VIX/VXN-based IV (which is the actual Black-Scholes
    input). The gap (IV - realized_vol) is the "vol risk premium" -- positive
    when IC selling is statistically attractive, compressed/negative when
    options are cheap relative to recent realized moves.
    """
    r = df[ticker].pct_change()
    return r.rolling(window).std() * np.sqrt(252)


def fetch_risk_free_rate(start: str = "max") -> pd.Series:
    """
    Fetch 13-week T-Bill yield (^IRX) as a daily annualized rate Series.

    ^IRX is quoted in annualized percentage points (e.g., 5.0 = 5.00%/yr).
    Returns a pd.Series indexed by date with decimal rates (e.g., 0.05).

    Used as the daily benchmark financing rate for the synthetic cost model.
    Forward-filled to cover any market-holiday gaps in T-Bill data.
    """
    print("  Fetching ^IRX (13-week T-Bill)...")
    try:
        raw = yf.Ticker("^IRX").history(period=start, auto_adjust=True)
        if raw.empty:
            print("  WARNING: ^IRX empty; falling back to constant rate")
            return None
        s = raw[["Close"]].reset_index()
        s["Date"] = pd.to_datetime(s["Date"]).dt.date
        s = s.rename(columns={"Close": "rf"})
        s["rf"] = s["rf"] / 100.0  # percent -> decimal
        print(f"  OK ^IRX: {len(s)} days | {s['Date'].min()} -> {s['Date'].max()} | "
              f"range [{s['rf'].min()*100:.2f}%, {s['rf'].max()*100:.2f}%]")
        return s.set_index("Date")["rf"]
    except Exception as e:
        print(f"  ERROR fetching ^IRX: {e}")
        return None


def fetch_and_save_irx(data_dir=OUTPUT_DIR):
    """Persist ^IRX daily annualized rate (decimal) as a dated CSV so the cash
    leg is reproducible + cache-fingerprintable. Raises if the fetch is empty."""
    rate = fetch_risk_free_rate("max")
    if rate is None or len(rate) == 0:
        raise RuntimeError("fetch_risk_free_rate returned no data for IRX")
    out = (pd.DataFrame({"Date": pd.to_datetime(rate.index),
                         "irx_rate_annual": rate.values.astype(float)})
           .dropna().drop_duplicates(subset="Date")
           .sort_values("Date").reset_index(drop=True))
    os.makedirs(data_dir, exist_ok=True)
    out.to_csv(os.path.join(data_dir, "IRX.csv"), index=False)
    return out


# ── Validation & Storage ───────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame, ticker: str) -> bool:
    """Check for NaN, zero/negative prices, extreme daily returns, duplicate dates."""
    issues = []
    nan_count = df[ticker].isna().sum()
    if nan_count > 0:
        issues.append(f"{nan_count} NaN values")
    if (df[ticker] <= 0).any():
        issues.append(f"{(df[ticker] <= 0).sum()} zero/negative prices")
    extreme = (df[ticker].pct_change().abs() > 1.0).sum()
    if extreme > 0:
        issues.append(f"{extreme} daily returns > 100%")
    dupes = df["Date"].duplicated().sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate dates")
    if issues:
        print(f"  WARNING {ticker}: {'; '.join(issues)}")
        return False
    print(f"  OK {ticker} passed all validation checks")
    return True


def save_to_csv(df: pd.DataFrame, ticker: str, output_dir: str) -> str:
    """Save ticker DataFrame to data/raw/<ticker>.csv."""
    os.makedirs(output_dir, exist_ok=True)
    fp = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(fp, index=False)
    print(f"  Saved -> {fp}")
    return fp


# ── IC Backtest Inputs ──────────────────────────────────────────────────────────

IC_OHLC_TICKERS = ["SPY", "QQQ"]
IC_IV_TICKERS = {"SPY": "^VIX", "QQQ": "^VXN"}
# True same-day (0DTE-horizon) IV for same-day option pricing.
# Single 1-day index exists (SPX-based VIX1D); QQQ is imputed from it downstream.
IC_0DTE_IV_TICKERS = ["^VIX1D"]


def fetch_ic_inputs(start: str = "max", output_dir: str = OUTPUT_DIR) -> dict:
    """
    Fetch + cache the inputs needed for the IC walk-forward backtest:
    SPY/QQQ OHLC and their VIX/VXN IV proxies.

    Saves to data/raw/{TICKER}_ohlc.csv and data/raw/{VIX_TICKER}.csv
    (^ stripped from filename). Returns a dict keyed by "SPY_ohlc", "QQQ_ohlc",
    "VIX", "VXN".
    """
    print("=" * 60)
    print("IC Backtest Data Fetch")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    for ticker in IC_OHLC_TICKERS:
        print(f"\n[{ticker} OHLC]")
        df = fetch_ohlc(ticker, start)
        if df is not None:
            save_to_csv(df, f"{ticker}_ohlc", output_dir)
            results[f"{ticker}_ohlc"] = df

    for ticker, iv_symbol in IC_IV_TICKERS.items():
        clean_name = iv_symbol.lstrip("^")
        print(f"\n[{clean_name}]  (IV proxy for {ticker})")
        df = fetch_iv_proxy(iv_symbol, start)
        if df is not None:
            df = df.rename(columns={iv_symbol: clean_name})
            save_to_csv(df, clean_name, output_dir)
            results[clean_name] = df

    for iv_symbol in IC_0DTE_IV_TICKERS:
        clean_name = iv_symbol.lstrip("^")
        print(f"\n[{clean_name}]  (true 0DTE IV for same-day option pricing)")
        df = fetch_iv_proxy(iv_symbol, start)
        if df is not None:
            df = df.rename(columns={iv_symbol: clean_name})
            save_to_csv(df, clean_name, output_dir)
            results[clean_name] = df

    n_expected = len(IC_OHLC_TICKERS) + len(IC_IV_TICKERS) + len(IC_0DTE_IV_TICKERS)
    print("\n" + "=" * 60)
    print(f"IC input fetch complete. {len(results)}/{n_expected} series built.")
    print("=" * 60)

    return results


# ── Synthetic Construction ─────────────────────────────────────────────────────

def build_synthetic_3x(
    df: pd.DataFrame,
    ticker: str,
    leverage: float = 3.0,
    annual_expense_ratio: float = 0.0088,
    annual_financing_rate=None,
    financing_spread: float = 0.005,
    dividend_yield: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Construct a synthetic leveraged ETF price series from an underlying index.

    RETURN MODEL (per day t):
      r_synth_t = leverage * (r_t + daily_div)  -  daily_expense  -  daily_financing_t

      daily_div          = annual dividend_yield / trading_days
      daily_expense      = annual_ER / trading_days
      daily_financing_t  = (leverage - 1) * (rf_t + spread) / trading_days

      The dividend term is applied INSIDE the leverage multiplier because leveraged
      ETFs gain 3x exposure to the total return of the underlying, including dividends
      captured via their equity swap contracts.

    WHEN TO SET dividend_yield > 0:
      Use for price-return indexes (e.g. ^SOX) where yfinance strips dividends from
      the Close price. Total-return ETFs (QQQ, SPY) already include dividends in
      their adjusted close — leave dividend_yield=0.0 for those.

    TIME-VARYING FINANCING RATE:
      When annual_financing_rate is a pd.Series (from fetch_risk_free_rate),
      the daily financing cost tracks the actual rate environment:
        - 2010-2021 near-zero rates: ~0.75%/yr total financing
        - 2022-2023 rate hike cycle: ~5.5-6.0%/yr total financing

    Parameters
    ----------
    dividend_yield : float
        Estimated annual dividend yield of the underlying index, as a decimal.
        Applied daily (yield/252) inside the leverage multiplier to restore
        total-return behavior for price-return indexes. Default 0.0 (no adjustment).
        For ^SOX (PHLX Semiconductor), ~1.2-1.5%/yr historically.
    annual_financing_rate : float, pd.Series, or None
        - pd.Series: daily annualized rates indexed by date (from fetch_risk_free_rate).
          financing_spread is added on top.
        - float: constant annual rate (spread ignored).
        - None: falls back to FINANCING_RATE_FALLBACK (2.5%).
    financing_spread : float
        Repo/swap spread above the rate series (default 50 bps).
        Only applied when annual_financing_rate is a pd.Series.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    if (df[ticker] <= 0).any():
        raise ValueError(f"build_synthetic_3x: {ticker} contains zero or negative prices.")

    daily_expense = annual_expense_ratio / trading_days

    # Build daily financing rate series
    if isinstance(annual_financing_rate, pd.Series):
        # Normalize index to date objects for alignment
        rf = annual_financing_rate.copy()
        rf.index = pd.to_datetime(rf.index).date
        rf.index.name = "Date"  # preserve through reset_index()
        df_dates = pd.to_datetime(df["Date"]).dt.date

        # Left-join price dates onto rate series, then forward-fill gaps
        rate_aligned = (
            pd.DataFrame({"Date": df_dates})
            .merge(rf.rename("rf").reset_index(), on="Date", how="left")
            ["rf"]
            .ffill()
            .bfill()
            .fillna(FINANCING_RATE_FALLBACK)
            .values
        )
        daily_financing = (leverage - 1) * (rate_aligned + financing_spread) / trading_days

        avg_annual = float(np.mean(rate_aligned + financing_spread))
        total_drag = annual_expense_ratio + (leverage - 1) * avg_annual
        print(f"  Time-varying rate: avg {avg_annual*100:.2f}%/yr  "
              f"total drag avg {total_drag*100:.2f}%/yr")
    else:
        rate = annual_financing_rate if annual_financing_rate is not None else FINANCING_RATE_FALLBACK
        daily_financing = (leverage - 1) * rate / trading_days
        total_drag = annual_expense_ratio + (leverage - 1) * rate
        print(f"  Constant rate: {rate*100:.2f}%/yr  total drag {total_drag*100:.2f}%/yr")

    # Dividend adjustment: add yield inside the leverage multiplier (total-return correction)
    daily_div = dividend_yield / trading_days

    # Exact discrete compounding (cumprod is exact — no Taylor approximation needed)
    r = df[ticker].pct_change().fillna(0.0)
    r_synth = leverage * (r + daily_div) - daily_expense - daily_financing
    prices = 100.0 * (1 + r_synth).cumprod()

    synthetic_name = f"{ticker}_{int(leverage)}x_synthetic"
    result = pd.DataFrame({"Date": df["Date"], synthetic_name: prices})

    if dividend_yield > 0:
        print(f"  Dividend yield adj: +{dividend_yield*100:.2f}%/yr  "
              f"(+{leverage*dividend_yield*100:.2f}%/yr leveraged)")
    print(f"  OK {synthetic_name}: {len(result)} days | "
          f"{result['Date'].min()} -> {result['Date'].max()}")
    return result


def build_synthetic_leveraged(
    df: pd.DataFrame,
    ticker: str,
    *,
    leverage: float = 2.0,
    annual_expense_ratio: float = 0.0,
    annual_financing_rate=None,
    financing_spread: float = 0.0100,
    dividend_yield: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Thin sibling of ``build_synthetic_3x`` carrying single-stock-appropriate
    DEFAULTS -- no new math, just a distinct default set so call sites never
    need the confusing "call build_synthetic_3x but pass leverage=2.0,
    annual_expense_ratio=0.0" pattern. ``leverage``/``dividend_yield`` are
    already fully generic in ``build_synthetic_3x``: ``leverage`` is a plain float used symbolically throughout
    the formula, and ``int(leverage)`` is used only in the output column
    name (``f"{ticker}_{int(leverage)}x_synthetic"``), which works cleanly
    for ``leverage=2.0`` -> ``"TICKER_2x_synthetic"``.

    Built for the synthetic 2x leveraged dip-buy family (a synthetic 2x margin-financed position on an
    individual stock, NOT a packaged leveraged-ETF product):

    * ``annual_expense_ratio=0.0`` (vs. ``build_synthetic_3x``'s 0.88% real-
      ETF default) -- there is no fund wrapper here, just real shares of the
      underlying financed 2x on standard Reg-T margin (50% initial margin =
      exactly 2:1); a margin-financed position pays financing interest
      (below) and trading spread (priced separately, at the caller's own
      cost-lookup site), never a management fee.
    * ``financing_spread=0.0100`` (100 bps, vs. the 50 bps index-ETF
      default calibrated for institutional swap financing on TQQQ/SOXL/
      SPXL) -- single-name RETAIL margin-loan economics typically run
      above institutional swap spreads. An unvalidated placeholder, same
      disclosed-default convention as every other invented constant in
      this pipeline; a sensitivity sweep over {0.0050, 0.0100, 0.0150,
      0.0200} is the intended substitute for the CAGR-vs-real-ETF
      calibration ``build_synthetic_3x``'s own defaults use, which is
      unavailable here (no long-history real 2x single-stock ETF exists
      to check against -- that is the whole reason this is synthetic).
    * ``dividend_yield=0.0`` -- every underlying's Close is already a
      total-return series here (the mining driver's OHLC loader uses
      yfinance's ``auto_adjust=True`` uniformly), so no per-ticker yield
      lookup is needed or wanted -- this is the QQQ/SPY case
      (``build_synthetic_3x``'s own docstring), never the ``^SOX``
      price-index special case.

    ``df``/``ticker`` keep ``build_synthetic_3x``'s own calling convention
    exactly (a ``Date`` column plus a column literally named ``ticker``,
    NOT an OHLC frame) -- callers adapt an OHLC ``Close`` column into this
    shape at the call site, the same translation `abi1_driver`'s curve
    builders already require of every driver.
    """
    return build_synthetic_3x(
        df, ticker, leverage=leverage,
        annual_expense_ratio=annual_expense_ratio,
        annual_financing_rate=annual_financing_rate,
        financing_spread=financing_spread,
        dividend_yield=dividend_yield, trading_days=trading_days)


# ── Taylor Series Decomposition ────────────────────────────────────────────────

def taylor_vol_drag(
    df: pd.DataFrame,
    ticker: str,
    leverage: float = 3.0,
    window: int = 21,
) -> pd.DataFrame:
    """
    Rolling Taylor series decomposition of leveraged ETF log returns by regime.

    Expands log(1 + lambda*r) around 0 to expose the contribution of each
    statistical moment of the underlying's return distribution:

      log(1 + lambda*r) = lambda*r                  1st:  trend (linear in r)
                        - lambda^2 * r^2 / 2        2nd:  vol drag (negative, quadratic)
                        + lambda^3 * r^3 / 6        3rd:  skew correction (cubic)
                        - lambda^4 * r^4 / 24       4th:  kurtosis correction (quartic)

    The exact value log(1 + lambda*r) is also computed for comparison — the
    approximation error is the residual beyond 4th order.

    REGIME CHARACTERIZATION (rolling window estimates):
      mu_annual    : annualized arithmetic mean return of underlying
      sigma_annual : annualized volatility of underlying

    OUTPERFORMANCE CONDITION (ignoring cost):
      mu > 2 * sigma^2   -> leverage geometrically outperforms
      Ratio mu / (2*sigma^2):  > 1 means leverage helps,  < 1 means it hurts

    The 2nd-order term shows WHY high-vol regimes destroy leveraged performance
    quadratically: doubling sigma quadruples the drag. The 3rd-order term shows
    that negative skewness (crash asymmetry) adds additional drag.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'Date' and <ticker> columns.
    window : int
        Rolling window in trading days for regime estimation (default 21 = 1 month).
        Use 63 (1 quarter) for smoother regime signals, 21 for faster detection.

    Returns
    -------
    pd.DataFrame
        Columns: Date, r_underlying, term1..term4, taylor_approx_4th, exact_log1p,
                 approx_error, rolling_mu_annual, rolling_sigma_annual,
                 rolling_vol_drag_annual, mu_over_2sigma2.
        Indexed from 1 (first row has NaN returns from pct_change).
    """
    r = df[ticker].pct_change()
    lr = leverage * r

    # Taylor terms — exact contributions per day
    term1 = lr                          # 1st order: trend
    term2 = -(lr ** 2) / 2              # 2nd order: vol drag
    term3 = (lr ** 3) / 6               # 3rd order: skew
    term4 = -(lr ** 4) / 24             # 4th order: kurtosis

    # Exact log(1 + lambda*r) for comparison (the simulation target)
    exact = np.log1p(lr)
    approx_4th = term1 + term2 + term3 + term4

    # Rolling regime characterization (annualized)
    mu = r.rolling(window).mean() * 252
    sigma2 = r.rolling(window).var() * 252
    sigma = sigma2 ** 0.5

    # Vol drag term annualized from 2nd-order: -lambda^2/2 * sigma^2_annual
    vol_drag_annual = -(leverage ** 2 / 2) * sigma2

    # Outperformance ratio: mu / (2 * sigma^2); >1 = leverage helps, <1 = hurts
    mu_ratio = mu / (2 * sigma2.clip(1e-10))

    return pd.DataFrame({
        "Date": df["Date"],
        "r_underlying": r,
        "term1_trend": term1,
        "term2_vol_drag": term2,
        "term3_skew": term3,
        "term4_kurtosis": term4,
        "taylor_approx_4th": approx_4th,
        "exact_log1p": exact,
        "approx_error": exact - approx_4th,
        "rolling_mu_annual": mu,
        "rolling_sigma_annual": sigma,
        "rolling_vol_drag_annual": vol_drag_annual,
        "mu_over_2sigma2": mu_ratio,      # > 1 = leverage condition met
    })


# ── Main Entry Point ───────────────────────────────────────────────────────────

def run_pipeline(
    tickers: list = TICKERS,
    start: str = START_DATE,
    synthetic_configs: list = SYNTHETIC_CONFIGS,
    use_market_rates: bool = True,
) -> dict:
    """
    Fetch all tickers, build synthetic series with time-varying financing rates.

    When use_market_rates=True (default), fetches ^IRX and uses it as the
    daily financing rate benchmark for all synthetic series. This produces
    more accurate cost estimates across rate regimes than a constant rate.

    Returns dict: ticker/synthetic_name -> DataFrame.
    """
    print("=" * 60)
    print("Data Pipeline")
    print(f"Run time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers   : {', '.join(tickers)}")
    print(f"Synthetics: {len(synthetic_configs)} series")
    print(f"Rates     : {'time-varying (^IRX)' if use_market_rates else 'constant fallback'}")
    print("=" * 60)

    results = {}

    # Step 1: Real tickers
    for ticker in tickers:
        print(f"\n[{ticker}]")
        df = fetch_ticker(ticker, start)
        if df is None:
            continue
        validate_data(df, ticker)
        save_to_csv(df, ticker, OUTPUT_DIR)
        results[ticker] = df

    # Step 2: Risk-free rate
    rf_series = None
    if use_market_rates:
        print("\n[^IRX]")
        rf_series = fetch_risk_free_rate(start)

        # Step 2b: Persist ^IRX as a dated CSV (cash leg source — reproducible +
        # cache-fingerprintable for the cash-state gate). Inside the
        # use_market_rates guard so the constant-fallback/offline mode never hits
        # the network or crashes on a transient IRX response.
        print("\n[IRX.csv]")
        fetch_and_save_irx(data_dir=OUTPUT_DIR)

    # Step 3: Synthetic series
    for cfg in synthetic_configs:
        underlying = cfg["underlying"]
        synthetic_name = f"{underlying}_{int(cfg['leverage'])}x_synthetic"
        print(f"\n[{synthetic_name}]  (mimics {cfg['real_etf']})")

        if underlying not in results:
            print(f"  SKIP: {underlying} was not fetched")
            continue

        synth_df = build_synthetic_3x(
            df=results[underlying],
            ticker=underlying,
            leverage=cfg["leverage"],
            annual_expense_ratio=cfg["annual_expense_ratio"],
            annual_financing_rate=rf_series,
            financing_spread=cfg.get("financing_spread", 0.005),
            dividend_yield=cfg.get("dividend_yield", 0.0),
            trading_days=cfg.get("trading_days", 252),
        )

        validate_data(synth_df, synthetic_name)
        save_to_csv(synth_df, synthetic_name, OUTPUT_DIR)
        results[synthetic_name] = synth_df

    print("\n" + "=" * 60)
    total = len(tickers) + len(synthetic_configs)
    print(f"Pipeline complete. {len(results)}/{total} series built.")
    print(f"Data saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_pipeline()
