"""dividend_month_premium_predicted_ex_month -- long each pool payer during
its PREDICTED ex-dividend month, exit at the ex-day close (event_pooled),
batch5 candidate module.

Source: Hartzmark & Solomon (2013), "The dividend month premium", JFE
109(3), 640-660. Rule (verbatim from the curated candidate's
rule_pseudocode, implemented per pool member independently):

    PRE: EX[t] = set of ex-dividend dates for ticker t (yfinance
         Ticker.dividends, dates only, amounts unused) -- fetched ONCE into
         data/mining/batch5_events/<family_id>/events.json (+ provenance.json)
         and loaded from that file here; no network in this module.
    PREDICTED(t, m): EX[t] contains a date whose calendar month equals
         m-3, m-6, m-9 or m-12 (calendar arithmetic).
    SIGNAL at bar i (date d, all data <= d):
      flat:        entry = (d is the last trading day of its calendar month)
                           and PREDICTED(t, next calendar month)
                   [engine fills at the next bar's open = first session of m]
      in position: exit  = (d in EX[t]) or (d is the last trading day of m)
    time_stop = 23 bars (safety cap only). One position per ticker. Long.
    POOL: every ticker in expanded_universe_2026-08-29, one share class per
         issuer; trades pooled by the driver into ONE trial.

Engine mapping (mining.backtest_engine.run_backtest): an entry signal on bar
t fills at bar t+1's open, an exit signal on bar t fills at bar t's close.
The engine only reads ``entry`` while flat and ``exit`` while in a position,
so the module emits, for EVERY bar, entry = "month-end bar whose next month
is predicted" and exit = "ex-day bar OR month-end bar"; the engine's own
state machine turns that into exactly one holding period per predicted
month. Because a month never has more than 23 NYSE sessions, the month-end
exit always fires at held <= 22 bars and the 23-bar time stop is a
non-binding safety cap (the driver's run_single_asset_windows passes none;
the module exposes TIME_STOP_BARS for callers that do).

Everything is precomputed ONCE in the factory from the full-history frame
and the ticker's ex-date set (no price enters either series; the month-end
flag needs only the NEXT bar's DATE, exchange-calendar knowledge, never its
price). The returned signal_fn does two positional array lookups per call.

Point-in-time: the entry at the end of month m-1 uses only ex-dates in
months m-3 .. m-12, all strictly before the signal bar. The exit uses the
realized ex-date, which is public at least ten calendar days before the
record date (NYSE 204.12; Nasdaq 5250(e)(6)) and therefore before the
ex-date itself; the module never reads the ex-day's price ahead of time.

Interpretation choices
----------------------
* One share class per issuer (candidate "pool"): GOOG, FOX and NWS are
  dropped in favour of GOOGL, FOXA and NWSA (the pairs in the 518-ticker
  universe that are two listings of one issuer's dividend calendar). Other
  look-alike pairs (HON/HONA, FDX/FDXF) are distinct post-spin issuers and
  are both kept. TRADED therefore has 515 members.
* "in_sample_start": "2001-01" is applied as a rule gate: no entry signal
  fires for a predicted month earlier than 2001-01, so a 1960s-inception
  frame (the driver loads period="max") does not manufacture pre-2001
  trades that the source's 12-month lookback from 2000 data never had.
  Predicted months from 2001-01 on use whatever ex-date history exists.
* "last trading day of its calendar month": a bar whose NEXT frame bar
  falls in a different calendar month. The frame's final bar has no
  successor and counts as a month-end only when no weekday of its month
  remains after it (a weekday holiday on the 31st is missed at the very
  last bar only -- nothing could fill after it anyway).
* An ex-date that is not a frame bar (data gap, or a non-session date in
  the source) exits at the close of the FIRST frame bar on or after it,
  provided that bar is still in the ex-date's calendar month; otherwise
  the month-end fallback exit already covers it. Ex-dates after the last
  frame bar are ignored.
* An ex-date on the first session of month m (the entry bar) cannot be an
  exit -- the engine never evaluates an exit on the entry bar (engine
  design note) -- so the position exits at the next ex-day in m or the
  month-end close. Disclosed, not "fixed": no filter is added.
* A ticker absent from events.json (or with no ex-dates) never enters.
* Holdout bucket: warm-up uses ex-dates before the bucket (they are the
  rule's only state); nothing after the bucket's last bar is read. The
  entry signal on the last bar BEFORE the bucket is not visible to the
  engine, so a holdout that starts on the first session of a predicted
  month misses that month's entry (same convention as every batch5
  bucket-sliced module).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "dividend_month_premium_predicted_ex_month"
SHAPE = "event_pooled"
SIGNAL: list[str] = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
EXPECTED_MIN_TRADES = 20000

# Copied verbatim (json -> Python literal, generated not retyped) from the
# catalog candidate's "parameters" -- the driver's drift check requires
# byte identity under json.dumps(sort_keys=True).
PARAMETERS = {'event_source': (
                  "yfinance Ticker.dividends per traded name; index = ex-dividend date "
                  "(yfinance dates are ex-dates: one business day before the record "
                  "date under T+2 settlement, the record date itself under T+1)"
              ),
 'predicted_month_rule': 'calendar month m is a predicted dividend month for ticker t if t had an '
                         'ex-dividend date in month m-3, m-6, m-9 or m-12 (Hartzmark-Solomon '
                         'quarterly-payer rule; also predicts semi-annual/annual payers via the '
                         '6/12 terms)',
 'entry': 'open of the first trading day of month m (entry signal fires on the last trading day of '
          'month m-1)',
 'exit': 'close of the first ex-dividend day inside month m; if no ex-dividend day occurs, close '
         'of the last trading day of month m',
 'time_stop_bars': 23,
 'yield_filter': 'none (source has none)',
 'one_position_per_ticker': True,
 'direction': 'long',
 'pool': (
     "the pipeline's cached expanded universe: current S&P 500 + NASDAQ "
     "100 constituents after dedup, with cached OHLC and dated bid/ask "
     "quotes from the pipeline's universe cache. Keep ONE share class per "
     "issuer (drop GOOG, keep GOOGL; likewise FOX/NWS-type duplicates) so "
     "a single corporate event is not counted twice. Survivorship caveat: "
     "current constituents only, the same convention as every other "
     "pooled family drawn from this universe; any result must disclose it."
 ),
 'in_sample_start': '2001-01 (first month with a 12-month dividend lookback available from 2000 '
                    'data)'}

# The candidate's resolved pool (== the universe_snapshot_2026-08-29 tickers).
_RESOLVED_POOL = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK",
    "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALAB", "ALB", "ALGN", "ALL",
    "ALLE", "ALNY", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "AON",
    "AOS", "APA", "APD", "APH", "APO", "APP", "APTV", "ARE", "ARES", "ARM", "ASML", "ATO",
    "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN",
    "BF-B", "BG", "BIIB", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BNY", "BR", "BRK-B", "BRO",
    "BSX", "BX", "BXP", "C", "CAH", "CARR", "CASY", "CAT", "CB", "CBOE", "CBRE", "CCEP", "CCI",
    "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CIEN", "CINF", "CL",
    "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COHR", "COIN", "COO",
    "COP", "COR", "COST", "CPAY", "CPRT", "CPT", "CRH", "CRL", "CRM", "CRWD", "CRWV", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL", "DASH", "DD",
    "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV",
    "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EBAY", "ECHO", "ECL", "ED", "EFX",
    "EG", "EIX", "EL", "ELV", "EME", "EMR", "EOG", "EQIX", "EQT", "ERIE", "ES", "ESS", "ETN",
    "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS",
    "FDX", "FDXF", "FE", "FER", "FERG", "FFIV", "FICO", "FIS", "FISV", "FITB", "FIX", "FLEX",
    "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC", "GEN", "GEV",
    "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS",
    "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HON", "HONA", "HOOD", "HPE",
    "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM", "ICE", "IDXX",
    "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
    "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM",
    "KKR", "KLAC", "KMB", "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII",
    "LIN", "LITE", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA",
    "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MELI", "MET", "META",
    "MGM", "MKC", "MLM", "MMM", "MNST", "MO", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRSH",
    "MRVL", "MS", "MSCI", "MSFT", "MSI", "MSTR", "MTB", "MTD", "MU", "NBIS", "NCLH", "NDAQ",
    "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS",
    "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL",
    "ORLY", "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PDD", "PEG", "PEP", "PFE", "PFG",
    "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "PPG",
    "PPL", "PRU", "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL", "Q", "QCOM", "RCL", "RDDT", "REG",
    "REGN", "RF", "RJF", "RKLB", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY",
    "SBAC", "SBUX", "SCHW", "SHOP", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNDK", "SNPS", "SO",
    "SOLV", "SPCX", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK",
    "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT",
    "TJX", "TKO", "TMO", "TMUS", "TPL", "TPR", "TRGP", "TRI", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS",
    "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VEEV", "VICI", "VLO", "VLTO", "VMC",
    "VMRK", "VRSK", "VRSN", "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD",
    "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY",
    "WYNN", "XEL", "XOM", "XYL", "XYZ", "YUM", "ZBH", "ZBRA", "ZTS",
]

# One share class per issuer (candidate "pool"): dropped -> kept sibling.
DROPPED_SHARE_CLASSES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}
TRADED = [t for t in _RESOLVED_POOL if t not in DROPPED_SHARE_CLASSES]

TIME_STOP_BARS = PARAMETERS["time_stop_bars"]
PREDICTION_LAGS_MONTHS = (3, 6, 9, 12)
# "2001-01 (first month with ...)": the leading YYYY-MM token is the value.
IN_SAMPLE_START_MONTH = pd.Period(PARAMETERS["in_sample_start"].split()[0], freq="M")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EVENTS_DIR = os.path.join(_REPO_ROOT, "data", "mining", "batch5_events", FAMILY_ID)
EVENTS_PATH = os.path.join(EVENTS_DIR, "events.json")
PROVENANCE_PATH = os.path.join(EVENTS_DIR, "provenance.json")

_EVENTS_CACHE: dict[str, dict] = {}


def load_events(path: str = EVENTS_PATH) -> dict:
    """{ticker: frozenset of tz-naive normalized ex-dividend Timestamps},
    read once per path and cached (the driver calls the factory once per
    ticker per bucket -- ~1000 times for this pool)."""
    key = os.path.abspath(path)
    cached = _EVENTS_CACHE.get(key)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{FAMILY_ID}: events file must be a JSON object")
    events = {}
    for ticker, dates in raw.items():
        if not isinstance(dates, list):
            raise ValueError(f"{FAMILY_ID}: events[{ticker!r}] must be a list")
        stamps = pd.DatetimeIndex(pd.to_datetime(dates))
        if stamps.tz is not None:
            stamps = stamps.tz_localize(None)
        events[ticker] = frozenset(stamps.normalize())
    _EVENTS_CACHE[key] = events
    return events


def is_predicted_month(ex_dates, month: pd.Period) -> bool:
    """PREDICTED(t, m): an ex-date exists in month m-3, m-6, m-9 or m-12."""
    ex_months = {pd.Timestamp(d).to_period("M") for d in ex_dates}
    return any((month - lag) in ex_months for lag in PREDICTION_LAGS_MONTHS)


def _normalized_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    out = out.normalize()
    if not out.is_monotonic_increasing or out.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    return out


def _month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on bars whose next bar is in a different calendar month; the
    final bar is True only when no weekday of its month remains after it."""
    n = len(index)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    periods = index.to_period("M")
    mask[:-1] = periods[:-1] != periods[1:]
    last = index[-1]
    month_end_day = last.to_period("M").to_timestamp(how="end").normalize()
    remaining = pd.bdate_range(last + pd.Timedelta(days=1), month_end_day)
    mask[-1] = len(remaining) == 0
    return mask


def _entry_exit_series(frame_index: pd.DatetimeIndex, ex_dates) -> tuple:
    """Per bar of the frame: entry = month-end bar whose NEXT calendar month
    is predicted (and >= in_sample_start); exit = ex-day bar or month-end
    bar. No price is read."""
    n = len(frame_index)
    periods = frame_index.to_period("M")
    is_month_end = _month_end_mask(frame_index)
    ex_months = {pd.Timestamp(d).to_period("M") for d in ex_dates}

    entry = np.zeros(n, dtype=bool)
    for pos in np.flatnonzero(is_month_end):
        next_month = periods[pos] + 1
        if next_month < IN_SAMPLE_START_MONTH:
            continue
        entry[pos] = any((next_month - lag) in ex_months
                         for lag in PREDICTION_LAGS_MONTHS)

    exit_ = is_month_end.copy()
    for d in ex_dates:
        stamp = pd.Timestamp(d)
        pos = int(frame_index.searchsorted(stamp, side="left"))
        if pos < n and periods[pos] == stamp.to_period("M"):
            exit_[pos] = True
    return (pd.Series(entry, index=frame_index),
            pd.Series(exit_, index=frame_index))


def build_signal_fn(frame: pd.DataFrame, index, ex_dates):
    """SignalFn for one instrument given its full-history OHLC `frame`, the
    bucket `index` the engine will iterate over, and its ex-date set."""
    frame_index = _normalized_index(frame.index)
    entry_frame, exit_frame = _entry_exit_series(frame_index, ex_dates)
    bucket = _normalized_index(index)
    entry = entry_frame.reindex(bucket).fillna(False).to_numpy(dtype=bool)
    exit_ = exit_frame.reindex(bucket).fillna(False).to_numpy(dtype=bool)
    size = len(bucket)

    def signal_fn(frame_slice: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= size:
            return SignalResult()
        return SignalResult(entry=bool(entry[i]), exit=bool(exit_[i]))

    return signal_fn


def make_signal_fn(ticker: str, index: pd.DatetimeIndex, frames: dict):
    """Driver entry point (event_pooled): one SignalFn per pool member.
    `frames[ticker]` is that member's full-history OHLC frame; `index` is
    the bucket (in-sample or holdout) the engine will run over."""
    frame = frames[ticker]  # KeyError on a missing frame is the right failure
    ex_dates = load_events().get(ticker, frozenset())
    return build_signal_fn(frame, index, ex_dates)
