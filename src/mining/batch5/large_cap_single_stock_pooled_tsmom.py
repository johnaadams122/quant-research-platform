"""Batch5 candidate: large_cap_single_stock_pooled_tsmom (pooled_single_asset).

Pooled 12-month time-series momentum on individual large-cap stocks, long/flat,
decided at each calendar month-end close (Lim, Wang & Yao 2018; Moskowitz,
Ooi & Pedersen 2012). Rule, verbatim from the curated candidate record's
``rule_pseudocode``::

    for each T in POOL independently:
      C = auto-adjusted daily close of T (total return)
      M[k] = last available close of T in calendar month k
      at the close of the last trading day of each calendar month m
      (needs 12 prior month-end closes, else no decision):
        ret12 = M[m] / M[m-12] - 1
        target = LONG if ret12 > 0 else FLAT   (ret12 == 0 -> FLAT)
      if target == LONG and position == FLAT: BUY T at the OPEN of the next
        trading session
      if target == FLAT and position == LONG: SELL T at the CLOSE of the next
        trading session
      otherwise hold; no stops, no time exits; one trade = one contiguous
        holding period

Engine mapping (mining.backtest_engine.run_backtest): an ``entry=True`` on
the month-end bar fills at the NEXT bar's open; an ``exit=True`` fills at
the SAME bar's close, so the FLAT decision is emitted on the bar AFTER the
month-end bar (the first session of the new month), which is exactly "sell
at the close of the next trading session". A LONG decision while already
long re-emits ``entry=True`` (ignored by the engine) with ``exit=False``, so
the position is held; a FLAT decision while flat emits an exit that has
nothing to close. No stops, no time stop.

Every series is precomputed ONCE per (ticker, bucket) in ``make_signal_fn``
from shifted/grouped operations on the ticker's own close history truncated
at the bucket's last bar; the per-bar call is a position lookup (with a
date-keyed fallback) -- no rolling statistics are recomputed per bar.

Interpretation choices
----------------------
* "auto-adjusted daily close (total return)": the driver's frames come from
  ``data_pipeline.fetch_ohlc`` (``yfinance ... auto_adjust=True``), so the
  ``Close`` column handed in IS the dividend- and split-adjusted series; it
  is used as given, no further adjustment.
* "last trading day of each calendar month": the last bar of that calendar
  month present in the ticker's own frame (truncated at the bucket's final
  bar). Identifying it needs only the calendar (whether a later bar exists
  in the same month), never a future price. At the very end of a truncated
  history the final bar is therefore treated as a month-end even when the
  month is incomplete; the engine never evaluates an entry on a bucket's
  final bar and the corresponding exit bar lies outside the bucket, so this
  cannot create or alter a trade.
* "needs 12 prior month-end closes, else no decision": a decision at month m
  requires a month-end close for EVERY calendar month m-1 .. m-12 (which
  includes the M[m-12] denominator). A missing calendar month suppresses
  decisions until it rolls out of the window; observations are never
  carried across a gap.
* Pool: ``TRADED`` is the candidate's ``resolved_pool`` -- the fixed 518-name
  universe named in the pseudocode (``data/mining/universe_snapshot_
  2026-08-29.json``). The ``parameters.pool`` prose additionally says to keep
  one share class per issuer (GOOG/GOOGL, FOX/FOXA, NWS/NWSA); the resolved
  pool as curated still contains both classes and the pseudocode names the
  518 list, so the pool is taken as resolved and this conflict is recorded
  rather than resolved here (dropping names would be a pool edit, not a rule
  reading).
* Buckets are independent: a position is never carried into the holdout
  bucket; the first decision inside a bucket is its first month-end, using
  pre-bucket history only for the 12-month lookback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "large_cap_single_stock_pooled_tsmom"
SHAPE = "pooled_single_asset"
TRADED = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALAB", "ALB", "ALGN", "ALL", "ALLE", "ALNY", "AMAT", "AMCR", "AMD", "AME",
    "AMGN", "AMP", "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH",
    "APO", "APP", "APTV", "ARE", "ARES", "ARM", "ASML", "ATO", "AVGO", "AVY",
    "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX",
    "BEN", "BF-B", "BG", "BIIB", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BNY",
    "BR", "BRK-B", "BRO", "BSX", "BX", "BXP", "C", "CAH", "CARR", "CASY",
    "CAT", "CB", "CBOE", "CBRE", "CCEP", "CCI", "CCL", "CDNS", "CDW", "CEG",
    "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CIEN", "CINF", "CL", "CLX",
    "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COHR", "COIN",
    "COO", "COP", "COR", "COST", "CPAY", "CPRT", "CPT", "CRH", "CRL", "CRM",
    "CRWD", "CRWV", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "CTVA", "CVNA", "CVS",
    "CVX", "D", "DAL", "DASH", "DD", "DDOG", "DE", "DECK", "DELL", "DG",
    "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ",
    "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EBAY", "ECHO", "ECL", "ED",
    "EFX", "EG", "EIX", "EL", "ELV", "EME", "EMR", "EOG", "EQIX", "EQT",
    "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD",
    "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FDXF", "FE",
    "FER", "FERG", "FFIV", "FICO", "FIS", "FISV", "FITB", "FIX", "FLEX", "FOX",
    "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC", "GEN",
    "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC",
    "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HIG",
    "HII", "HLT", "HON", "HONA", "HOOD", "HPE", "HPQ", "HRL", "HSIC", "HST",
    "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM", "ICE", "IDXX", "IEX", "IFF",
    "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR", "IRM", "ISRG", "IT",
    "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM", "KDP",
    "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KO", "KR",
    "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII", "LIN", "LITE", "LLY",
    "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA",
    "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MELI",
    "MET", "META", "MGM", "MKC", "MLM", "MMM", "MNST", "MO", "MOS", "MPC",
    "MPWR", "MRK", "MRNA", "MRSH", "MRVL", "MS", "MSCI", "MSFT", "MSI", "MSTR",
    "MTB", "MTD", "MU", "NBIS", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX",
    "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA",
    "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL",
    "ORLY", "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PDD", "PEG", "PEP",
    "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM",
    "PNC", "PNR", "PNW", "PODD", "PPG", "PPL", "PRU", "PSA", "PSKY", "PSX",
    "PTC", "PWR", "PYPL", "Q", "QCOM", "RCL", "RDDT", "REG", "REGN", "RF",
    "RJF", "RKLB", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX",
    "RVTY", "SBAC", "SBUX", "SCHW", "SHOP", "SHW", "SJM", "SLB", "SMCI", "SNA",
    "SNDK", "SNPS", "SO", "SOLV", "SPCX", "SPG", "SPGI", "SRE", "STE", "STLD",
    "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T",
    "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TKO",
    "TMO", "TMUS", "TPL", "TPR", "TRGP", "TRI", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER",
    "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VEEV",
    "VICI", "VLO", "VLTO", "VMC", "VMRK", "VRSK", "VRSN", "VRT", "VRTX", "VST",
    "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD", "WDAY", "WDC", "WEC", "WELL",
    "WFC", "WM", "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XYL", "XYZ", "YUM", "ZBH", "ZBRA", "ZTS",
]
SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied verbatim from the candidate JSON "parameters" (byte-identical under
# json.dumps(sort_keys=True); pinned by the module's test file).
PARAMETERS = {
    "lookback_months": 12,
    "decision_frequency": "monthly_at_month_end_close",
    "long_condition": "trailing_12_month_total_return_gt_0",
    "flat_condition": "trailing_12_month_total_return_le_0",
    "entry": "next_session_open",
    "exit": "next_session_close",
    "pool": (
        "the pipeline's cached expanded universe: current S&P 500 + NASDAQ "
        "100 constituents after dedup, with cached OHLC and dated bid/ask "
        "quotes from the pipeline's universe cache. Keep ONE share class per "
        "issuer (drop GOOG, keep GOOGL; likewise FOX/NWS-type duplicates) so "
        "a single corporate event is not counted twice. Survivorship caveat: "
        "current constituents only, the same convention as every other "
        "pooled family drawn from this universe; any result must disclose it."
    ),
    "pooling": "all_pool_trades_concatenated_one_trial",
}
EXPECTED_MIN_TRADES = 4000

LOOKBACK_MONTHS = PARAMETERS["lookback_months"]

_ENTRY = SignalResult(entry=True, exit=False)
_EXIT = SignalResult(entry=False, exit=True)
_NONE = SignalResult()


def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def decision_dates(close: pd.Series, *, lookback_months: int = LOOKBACK_MONTHS):
    """Return ``(entry_dates, exit_dates)`` for one ticker's close history.

    ``entry_dates``: month-end bars whose 12-month total return is > 0
    (LONG decided at that close; the engine buys at the next open).
    ``exit_dates``: the bar immediately AFTER a month-end bar whose 12-month
    return is <= 0 (FLAT decided; the engine sells at that bar's close).
    A month-end bar with fewer than ``lookback_months`` complete prior
    calendar months of closes decides nothing.
    """
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    values = pd.to_numeric(pd.Series(close), errors="coerce")
    values.index = _normalize_index(values.index)
    values = values.dropna()
    values = values[~values.index.duplicated(keep="last")].sort_index()
    n = len(values)
    if n == 0:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([])

    periods = values.index.to_period("M")
    # Last bar of each calendar month present in the (truncated) history.
    is_month_end = np.append(periods[1:] != periods[:-1], True)
    month_end_pos = np.flatnonzero(is_month_end)
    month_periods = periods[month_end_pos]
    month_close = values.to_numpy(dtype=float)[month_end_pos]
    monthly = pd.Series(month_close, index=month_periods)

    # Every one of the prior ``lookback_months`` calendar months must have a
    # month-end close; the denominator is the month exactly ``lookback_months``
    # back.
    decided = np.ones(len(month_periods), dtype=bool)
    for k in range(1, lookback_months + 1):
        decided &= (month_periods - k).isin(monthly.index)
    base = monthly.reindex(month_periods - lookback_months).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = month_close / base - 1.0
    decided &= np.isfinite(ret) & (base > 0)
    long = decided & (ret > 0)
    flat = decided & ~(ret > 0)

    entry_dates = values.index[month_end_pos[long]]
    exit_pos = month_end_pos[flat] + 1
    exit_pos = exit_pos[exit_pos < n]
    exit_dates = values.index[exit_pos]
    return entry_dates, exit_dates


def make_signal_fn(ticker: str, index, frames: dict):
    """SignalFn for one pool member over one bucket (``index``).

    Precomputes the entry/exit calendar from ``frames[ticker]`` truncated at
    the bucket's last bar (pre-bucket history is used only as lookback
    warm-up), reindexes it to ``index``, and answers each bar by position.
    """
    if ticker not in frames:
        raise KeyError(f"{FAMILY_ID}: no frame for {ticker}")
    frame = frames[ticker]
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame lacks a Close column")
    bucket = _normalize_index(index)
    if len(bucket) == 0:
        def empty_signal_fn(frame_slice, i):
            return _NONE
        return empty_signal_fn

    close = pd.Series(frame["Close"].to_numpy(dtype=float),
                      index=_normalize_index(frame.index))
    close = close.loc[close.index <= bucket[-1]]
    entry_dates, exit_dates = decision_dates(close, lookback_months=LOOKBACK_MONTHS)

    entry = np.asarray(bucket.isin(entry_dates), dtype=bool)
    exit_ = np.asarray(bucket.isin(exit_dates), dtype=bool)
    results = np.empty(len(bucket), dtype=object)
    for pos in range(len(bucket)):
        if entry[pos] and exit_[pos]:
            # Only reachable when a calendar month holds a single bar (the
            # exit bar after a FLAT month-end is itself a month-end): report
            # both flags honestly and let the engine's own collision rule
            # decide.
            results[pos] = SignalResult(entry=True, exit=True)
        elif entry[pos]:
            results[pos] = _ENTRY
        elif exit_[pos]:
            results[pos] = _EXIT
        else:
            results[pos] = _NONE
    bucket_ns = bucket.asi8
    by_date = {int(ts): pos for pos, ts in enumerate(bucket_ns)}
    n = len(bucket)

    def signal_fn(frame_slice, i):
        today = pd.Timestamp(frame_slice.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        today_ns = int(today.normalize().value)
        if 0 <= i < n and bucket_ns[i] == today_ns:
            return results[i]
        pos = by_date.get(today_ns)
        if pos is None:
            return _NONE
        return results[pos]

    return signal_fn
