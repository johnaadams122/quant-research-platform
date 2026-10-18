"""Batch5 candidate: stock_split_post_split_drift_pooled (event_pooled).

Post-split drift on REAL forward-split events, pooled across the expanded
large-cap universe: enter at the open of the first session after the split's
effective (ex) date -- the date yfinance's ``Ticker.splits`` carries -- and
hold for the sources' 12-month horizon (Ikenberry, Rankine & Stice 1996;
Ikenberry & Ramnath 2002). Events from every pool member are scored
together as one trial. Rule, verbatim from the candidate's
``rule_pseudocode``::

    PRE: for ticker t load auto-adjusted OHLC and SPLITS[t] = {date S: ratio}
         from Ticker.splits; keep only ratio >= 1.25 with
         abs(12*ratio - round(12*ratio)) <= 0.005.
    SIGNAL at bar i (date d, all data <= d):
      if flat: entry = (d in SPLITS[t])   [engine executes entry at the next
                bar's open = first session after the split-date bar]
      if in position: exit = False (time stop governs)
    time_stop = 252 bars -> the engine closes the trade at the close of the
      252nd bar after entry (reason time_stop). One position per ticker; a
      split occurring while a position is open is skipped. direction = long.
    POOL: run independently for every ticker in expanded_universe_2026-08-29
      (one share class per issuer), concatenate trades, score the pooled
      sample as ONE trial under the existing family_id.

Event data
----------
The split calendar is fetched ONCE (``fetch_split_events``, real yfinance
``Ticker.splits`` per pool member) and saved under
``data/mining/batch5_events/stock_split_post_split_drift_pooled/``:

* ``events.json``      -- ``{ticker: [ISO dates]}``, ONLY the events passing
                          the ratio rule (the calendar the factory reads);
* ``raw_splits.json``  -- ``{ticker: {ISO date: ratio}}``, every split-series
                          entry yfinance returned, so the ratio filter is
                          auditable after the fact;
* ``provenance.json``  -- source, fetched_at, member/event counts, failures.

The factory never touches the network: ``make_signal_fn`` reads
``events.json`` (or an explicit ``events`` mapping in tests).

Engine mapping (mining.backtest_engine.run_backtest)
----------------------------------------------------
An ``entry=True`` on the split-date bar fills at the NEXT bar's open, which is
exactly "the first session after the split-date bar". The driver
(scripts/run_batch5_mining.py -> run_batch3_mining.run_single_asset_windows)
calls ``run_backtest`` WITHOUT its ``time_stop`` argument, so the time stop is
expressed by this module: ``exit=True`` on the bar exactly ``time_stop_bars``
bars after the fill bar, i.e. bucket position ``k + 1 + 252`` for a split on
position ``k``. That is the same bar the engine's own ``time_stop=252`` would
close on (``held = i - entry_idx >= 252``); only the recorded ``reason`` label
differs (``signal_exit`` rather than ``time_stop``). Every other bar returns
``entry=False, exit=False``.

The whole entry/exit schedule is precomputed ONCE per (ticker, bucket) in
``make_signal_fn`` from the split calendar and the bucket's own
``DatetimeIndex`` -- no price is read at all, so nothing here can look ahead
(a split's effective date is public weeks before it and the signal reads
only that the event has occurred). The per-bar call is a position lookup.

Interpretation choices
----------------------
* "one share class per issuer" (pseudocode POOL line; ``parameters.pool``:
  "drop GOOG, keep GOOGL; likewise FOX/NWS-type duplicates"): the resolved
  518-name pool contains three dual-listed issuers (GOOG/GOOGL, FOX/FOXA,
  NWS/NWSA). Following the stated rule, ``TRADED`` keeps the class-A line
  (GOOGL, FOXA, NWSA) and drops GOOG, FOX and NWS -- 515 names -- because a
  simultaneous split of both classes IS a single corporate event and the
  rule exists precisely so it is not counted twice. ``RESOLVED_POOL`` keeps
  the full 518 list as curated; ``events.json`` is fetched for all 518.
* "d in SPLITS[t]": the split date is mapped onto the bucket calendar as the
  first bucket bar ON OR AFTER it (``searchsorted``). When the date is a
  session bar (the normal case -- yfinance dates splits to the first
  split-adjusted session) this is an exact match; if a date falls on a
  non-session day the first following session carries the signal, which is
  still causal (the event has already occurred). Off-session dates do
  occur in practice: a split dated inside a market closure (e.g. the
  post-9/11 closure in September 2001), a gap in a vendor's early history,
  or a split dated after a cache's last row. Split dates outside the
  bucket's own [first bar, last bar] range are ignored.
* "a split occurring while a position is open is skipped": a position is open
  from the fill bar (split position + 1) through the exit bar inclusive (the
  engine closes at that bar's CLOSE), so a second split dated anywhere in
  that span -- including the exit bar itself -- produces no entry. The
  eligible split after a hold is the first one dated AFTER the exit bar.
* Buckets are independent (the driver runs in-sample and holdout as separate
  ``run_backtest`` calls): a hold is never carried across the bucket boundary
  and a split dated before the bucket's first bar creates no position inside
  it. A hold still open at the bucket's last bar is closed by the engine's
  ``end_of_data`` path -- which, in the derived 6-month holdout bucket, is
  the outcome of EVERY holdout trade (Data hygiene item 2 below).
* A split too late in the bucket to be held for even one bar reports NO
  entry -- see "Data hygiene" below, which covers both the final-bar case
  (no fill bar exists) and the second-to-last-bar case (the fill bar is the
  bucket's last bar, so the trade would open and close on it).
* ``ratio_rule``: applied at fetch time with ``passes_ratio_rule``; reverse
  splits (ratio < 1) and non-twelfth adjustment factors (spin-offs stored in
  the same series, e.g. 1.281) are excluded, exactly as the parameter text
  says. The rule is a module function so it is unit-tested synthetically.

Data hygiene
------------
Both items below are mappings of the pre-registered rule onto a finite
bucket. Neither adds, removes or retunes a PARAMETERS key.

1. ZERO-HOLD ENTRIES ARE EXCLUDED (``entry_exit_positions``). The engine
   fills an entry at the NEXT bar's open and, if a hold is still open on the
   bucket's last bar, closes it at that bar's close. A split on bucket
   position ``n - 2`` therefore fills on the last bar and is force-closed on
   the same bar: ``held = 0``, ``reason = "end_of_data"``, a single-session
   open-to-close return recorded as a trade for a rule whose exit is "the
   close of the 252nd bar after entry". It contains no post-split drift at
   all -- not a short version of the hypothesis but a different measurement
   (one intraday session) -- so no entry is reported for it, the same
   already-declared outcome as position ``n - 1`` (no fill bar exists). The
   discriminator is mechanical and magnitude-free: a fill bar that is the
   bucket's last bar cannot produce a hold of even one bar. This matters
   most in a short holdout bucket, where a single such one-session trade
   can carry a large share of the pooled weight and
   ``backtest_engine.holdout_passes`` is a bare sign check.

2. THE 252-BAR EXIT IS UNREACHABLE IN THE HOLDOUT BUCKET, so no holdout
   trade exercises the rule's own exit. The driver derives a 6-month holdout
   (``run_batch3_mining.derive_holdout_window``; no pinned window exists for
   this family_id), and every member with a full history gets the SAME
   bucket, roughly half the 252 bars the time stop needs. The exact session
   count moves with the run date and is derived, never pinned. Recent
   listings with ZERO in-sample bars have shorter buckets, but
   ``scripts/run_batch5_mining.py`` drops them before any backtest runs
   ("insufficient history: fewer than two bars in a window bucket"), so
   they are not part of the pool this disclosure describes. Every holdout
   trade is therefore truncated by the engine's ``end_of_data`` path. The
   holdout gate for this family thus tests a partial hold of the post-split
   window, NOT the 12-month rule the cited sources (Ikenberry, Rankine &
   Stice 1996; Ikenberry & Ramnath 2002) support; the in-sample bucket is
   the only one where the pre-registered exit binds. This is structural to
   the window the driver derives, not something this module can fix, and it
   is NOT a licence to shorten ``time_stop_bars`` -- that parameter is
   pre-registered. It is disclosed here so the holdout verdict is read as
   what it measures.
"""
from __future__ import annotations

import json
import os
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "stock_split_post_split_drift_pooled"
SHAPE = "event_pooled"

# The candidate's ``resolved_pool`` exactly as curated (518 names =
# data/mining/universe_snapshot_2026-08-29.json "tickers").
RESOLVED_POOL = [
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

# "one share class per issuer" -- see Interpretation choices. Class-A line
# kept (GOOGL, FOXA, NWSA); the second listing of the same issuer dropped.
DUPLICATE_SHARE_CLASSES = ["GOOG", "FOX", "NWS"]
TRADED = [ticker for ticker in RESOLVED_POOL
          if ticker not in DUPLICATE_SHARE_CLASSES]

SIGNAL: list = []
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
# Copied mechanically from the candidate JSON "parameters" (byte-identical
# under json.dumps(sort_keys=True)).
PARAMETERS = {
    'event_source': (
        "yfinance Ticker.splits per traded name; index = first "
        "split-adjusted trading day"
    ),
    'ratio_rule': (
        "ratio >= 1.25 and abs(12*ratio - round(12*ratio)) <= 0.005 (a true "
        "forward split ratio such as 5:4, 4:3, 3:2, 2:1, 3:1, 4:1, 5:1, 7:1, "
        "10:1, 20:1); excludes spin-off / reorganization adjustment factors "
        "yfinance stores in the same series (factors that are not a "
        "whole-twelfth ratio, e.g. 1.281)"
    ),
    'entry': 'open of the first trading day after the split-date bar',
    'exit': 'time stop only: close of the 252nd bar after entry',
    'time_stop_bars': 252,
    'one_position_per_ticker': True,
    'second_split_during_hold': 'skipped',
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
    'retest': (
        "a separate family_id, charged as its own trial; it is NOT in any "
        "--retest list and has no pinned holdout to preserve. Catalog entry "
        "63 is a distinct family and is unaffected by this one."
    ),
}
EXPECTED_MIN_TRADES = 200

TIME_STOP_BARS = int(PARAMETERS["time_stop_bars"])  # 252
MIN_SPLIT_RATIO = 1.25            # parameters["ratio_rule"]: ratio >= 1.25
RATIO_TWELFTHS_TOLERANCE = 0.005  # abs(12*ratio - round(12*ratio)) <= 0.005

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EVENTS_DIR = os.path.join(_REPO_ROOT, "data", "mining", "batch5_events", FAMILY_ID)
EVENTS_PATH = os.path.join(EVENTS_DIR, "events.json")
RAW_SPLITS_PATH = os.path.join(EVENTS_DIR, "raw_splits.json")
PROVENANCE_PATH = os.path.join(EVENTS_DIR, "provenance.json")
UNIVERSE_SNAPSHOT_PATH = os.path.join(
    _REPO_ROOT, "data", "mining", "universe_snapshot_2026-08-29.json")

_ENTRY = SignalResult(entry=True, exit=False)
_EXIT = SignalResult(entry=False, exit=True)
_NONE = SignalResult()


# ------------------------------------------------------------- ratio rule

def passes_ratio_rule(ratio) -> bool:
    """``ratio >= 1.25 and abs(12*ratio - round(12*ratio)) <= 0.005`` --
    a true forward split expressed in twelfths (5:4, 4:3, 3:2, 2:1, 3:1,
    ...). Non-finite or non-numeric ratios never pass."""
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value) or value < MIN_SPLIT_RATIO:
        return False
    twelfths = 12.0 * value
    return abs(twelfths - round(twelfths)) <= RATIO_TWELFTHS_TOLERANCE


# --------------------------------------------------------------- calendar

def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _normalize_dates(dates) -> pd.DatetimeIndex:
    if dates is None:
        return pd.DatetimeIndex([])
    out = _normalize_index(list(dates))
    return out[~out.duplicated()].sort_values()


_EVENTS_CACHE: dict = {}


def load_events(path: str = EVENTS_PATH) -> dict:
    """``{ticker: DatetimeIndex}`` from ``events.json`` (cached per path).
    Raises FileNotFoundError when the fetched calendar is missing -- the
    factory must never silently run with an empty event set."""
    key = os.path.abspath(path)
    cached = _EVENTS_CACHE.get(key)
    if cached is not None:
        return cached
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{FAMILY_ID}: split calendar missing at {path}; run "
            "fetch_split_events(...) once to create it")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{FAMILY_ID}: {path} must be an object keyed by ticker")
    events = {}
    for ticker, dates in raw.items():
        if not isinstance(dates, list):
            raise ValueError(f"{FAMILY_ID}: {path}[{ticker!r}] must be a list of ISO dates")
        events[str(ticker)] = _normalize_dates(dates)
    _EVENTS_CACHE[key] = events
    return events


def entry_exit_positions(split_dates, index, *,
                         time_stop_bars: int = TIME_STOP_BARS):
    """Precompute the bucket positions carrying ``entry=True`` and
    ``exit=True`` for one ticker.

    A split is mapped to the first bucket bar on or after its date; it
    enters only when no hold is open. A hold spans the fill bar (split
    position + 1) through the exit bar (fill + ``time_stop_bars``)
    inclusive; splits dated inside that span are skipped. A hold that would
    end past the bucket is left open (the engine closes it at end of data).

    A split whose FILL bar would be the bucket's last bar -- bucket position
    ``n - 2`` -- or later reports NO entry: see "Data hygiene" in the module
    docstring. Nothing dated after such a split can fill either, so the
    remaining calendar for that bucket is closed out.
    Returns ``(entry_positions, exit_positions)`` as int arrays.
    """
    if isinstance(time_stop_bars, bool) or not isinstance(time_stop_bars, int) \
            or time_stop_bars <= 0:
        raise ValueError("time_stop_bars must be a positive int")
    bucket = _normalize_index(index)
    n = len(bucket)
    dates = _normalize_dates(split_dates)
    if n == 0 or len(dates) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: bucket index must be sorted and unique")
    dates = dates[(dates >= bucket[0]) & (dates <= bucket[-1])]
    positions = np.asarray(bucket.searchsorted(dates, side="left"), dtype=int)
    entries, exits = [], []
    busy_through = -1
    for k in positions:
        if k <= busy_through:
            continue  # split while a position is open (incl. the exit bar)
        fill = k + 1
        if fill >= n - 1:
            # The fill bar is the bucket's LAST bar (k == n - 2), or there is
            # no fill bar at all (k == n - 1). In the first case the engine
            # opens at that bar's open and force-closes at the SAME bar's
            # close (held = 0, reason end_of_data) -- a single-session
            # open-to-close return that contains none of the post-split drift
            # the rule measures. Neither case can produce a hold of even one
            # bar, so no entry is reported and the rest of this bucket's
            # calendar is closed out (nothing later could fill either).
            busy_through = n - 1
            continue
        entries.append(int(k))
        exit_pos = fill + time_stop_bars
        if exit_pos < n:
            exits.append(int(exit_pos))
            busy_through = exit_pos
        else:
            busy_through = n - 1  # held to end of data
    return np.asarray(entries, dtype=int), np.asarray(exits, dtype=int)


# ---------------------------------------------------------------- factory

def make_signal_fn(ticker: str, index, frames: Mapping,
                   events: Optional[Mapping] = None):
    """SignalFn for one pool member over one bucket (``index``).

    ``events`` defaults to the fetched calendar at ``EVENTS_PATH``; tests
    pass ``{ticker: [dates]}`` directly. ``frames[ticker]`` must exist with
    an OHLC ``Close`` column (contract check) but no price is read.
    """
    if ticker not in frames:
        raise KeyError(f"{FAMILY_ID}: no frame for {ticker}")
    if "Close" not in frames[ticker].columns:
        raise ValueError(f"{FAMILY_ID}: {ticker} frame lacks a Close column")
    calendar = load_events() if events is None else events
    split_dates = calendar.get(ticker, ())
    bucket = _normalize_index(index)
    n = len(bucket)
    if n == 0:
        def empty_signal_fn(frame_slice, i):
            return _NONE
        return empty_signal_fn

    entry_pos, exit_pos = entry_exit_positions(
        split_dates, bucket, time_stop_bars=TIME_STOP_BARS)
    results = np.empty(n, dtype=object)
    results[:] = _NONE
    results[entry_pos] = _ENTRY
    results[exit_pos] = _EXIT
    bucket_ns = bucket.asi8
    by_date = {int(ts): pos for pos, ts in enumerate(bucket_ns)}

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

    signal_fn.entry_positions = entry_pos
    signal_fn.exit_positions = exit_pos
    return signal_fn


# ------------------------------------------------------------- data fetch

def _file_sha256(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_split_series(series) -> dict:
    """``{ISO date: ratio}`` from a yfinance ``Ticker.splits`` Series
    (tz-aware index, float ratios). Empty/None -> ``{}``."""
    if series is None or len(series) == 0:
        return {}
    values = pd.Series(series, copy=True)
    values.index = _normalize_index(values.index)
    values = pd.to_numeric(values, errors="coerce")
    out = {}
    for stamp, ratio in values.items():
        if pd.isna(ratio):
            continue
        out[stamp.strftime("%Y-%m-%d")] = float(ratio)
    return dict(sorted(out.items()))


def events_from_raw(raw: Mapping) -> dict:
    """Apply ``passes_ratio_rule`` to ``{ticker: {date: ratio}}`` ->
    ``{ticker: [ISO dates]}`` (every ticker keyed, possibly empty)."""
    return {ticker: sorted(date for date, ratio in splits.items()
                           if passes_ratio_rule(ratio))
            for ticker, splits in raw.items()}


def fetch_split_events(pool, *, out_dir: str = EVENTS_DIR, retries: int = 3,
                       pause_seconds: float = 2.0,
                       fetcher=None, log=print) -> dict:
    """ONE-TIME network fetch of ``Ticker.splits`` per pool member; writes
    raw_splits.json, events.json and provenance.json under ``out_dir`` and
    returns the provenance dict. ``fetcher(ticker) -> Series`` defaults to
    yfinance (imported lazily so the module itself never needs it)."""
    import datetime as _dt
    import time as _time

    if fetcher is None:
        import yfinance as yf

        def fetcher(ticker):  # noqa: E306
            return yf.Ticker(ticker).splits
        source_detail = f"yfinance {getattr(yf, '__version__', '?')} Ticker(t).splits"
    else:
        source_detail = "injected fetcher"

    pool = list(pool)
    raw, failures = {}, {}
    for number, ticker in enumerate(pool, start=1):
        error = None
        for attempt in range(1, retries + 1):
            try:
                raw[ticker] = normalize_split_series(fetcher(ticker))
                error = None
                break
            except Exception as exc:  # network / parsing -- retried, then recorded
                error = f"{type(exc).__name__}: {exc}"
                _time.sleep(pause_seconds * attempt)
        if error is not None:
            failures[ticker] = error
        if log is not None and (number % 25 == 0 or number == len(pool)):
            log(f"[{FAMILY_ID}] {number}/{len(pool)} fetched, "
                f"{len(failures)} failed")
    events = events_from_raw(raw)
    fetched_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance = {
        "family_id": FAMILY_ID,
        "source": "yfinance Ticker.splits (Yahoo Finance corporate-actions series), "
                  "one call per pool member",
        "source_detail": source_detail,
        "fetched_at": fetched_at,
        "pool_source": os.path.relpath(UNIVERSE_SNAPSHOT_PATH, _REPO_ROOT).replace(os.sep, "/"),
        "pool_source_sha256": (_file_sha256(UNIVERSE_SNAPSHOT_PATH)
                               if os.path.exists(UNIVERSE_SNAPSHOT_PATH) else None),
        "member_count": len(pool),
        "tickers_fetched": len(raw),
        "tickers_failed": dict(sorted(failures.items())),
        "raw_split_count": int(sum(len(v) for v in raw.values())),
        "event_count": int(sum(len(v) for v in events.values())),
        "tickers_with_events": int(sum(1 for v in events.values() if v)),
        "ratio_rule": PARAMETERS["ratio_rule"],
        "date_semantics": "yfinance split date = first split-adjusted trading "
                          "session (the effective/ex date); entry at the next "
                          "session's open",
        "share_class_note": ("events fetched for all resolved-pool members; TRADED "
                             "drops " + ", ".join(DUPLICATE_SHARE_CLASSES) +
                             " (one share class per issuer)"),
        "files": {"events": "events.json", "raw_splits": "raw_splits.json"},
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, payload in (("raw_splits.json", raw), ("events.json", events),
                          ("provenance.json", provenance)):
        path = os.path.join(out_dir, name)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    _EVENTS_CACHE.pop(os.path.abspath(os.path.join(out_dir, "events.json")), None)
    return provenance
