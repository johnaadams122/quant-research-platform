"""intraday_data.py
------------------
Fetches historical 1-minute OHLCV bars from Alpaca's SIP historical-bars API
(``GET https://data.alpaca.markets/v2/stocks/{symbol}/bars``), with a local
on-disk CSV cache so repeated backtests don't re-hit the API. It is the
data layer of the intraday/swing strategy-mining tracks, which share the
daily pipeline's statistical machinery rather than modifying it.

Credentials are read ONLY at request time via ``credentials_fn`` (default:
``_environment_credentials``, reading ``ALPACA_API_KEY_ID`` /
``ALPACA_API_SECRET_KEY`` from the environment), and the HTTP layer is
injected via ``session`` (any object exposing a ``requests``-shaped
``.get(url, params=, headers=, timeout=)``). This module owns its own
client.
Never print, log, write-to-file, or embed the actual key/secret values
anywhere (including in exception messages).

Bars are filtered to regular trading hours only -- the half-open window
[9:30:00, close) ET -- before being cached or returned; pre/after-market
data is deliberately out of scope for this module. The window is half-open,
not 9:30-16:00-inclusive-both-ends, because Alpaca bar timestamps mark the
START of their minute interval: a bar timestamped exactly at the close
covers the following minute, which is AFTER the close, not the last
minute of RTH (so an "inclusive of 16:00" bound would be wrong).

``close`` is that SPECIFIC day's real scheduled close, not a hardcoded
16:00 -- a real NYSE early-close day (e.g. the day after Thanksgiving)
closes at 13:00 ET, and Alpaca's feed can still report activity in the
13:00-16:00 window on that day that is not part of the regular session
(see ``_trading_sessions``). ``_RTH_START`` (9:30 ET) never varies -- NYSE has
no scheduled early-OPEN days.

Trading days (and each one's real close time) are computed from a real
NYSE trading-calendar (``pandas_market_calendars.get_calendar("NYSE")``),
not naive Mon-Fri weekday arithmetic -- a real NYSE holiday (Thanksgiving,
July 4th, etc.) that falls on a weekday is correctly excluded rather than
treated as an expected trading day (naive weekday arithmetic would raise a
spurious gap error on every request spanning any real holiday).

Fail-closed: if Alpaca returns no usable (RTH) bars for a real trading day,
or a ``next_page_token`` chain never terminates within a bounded page
count, this module raises rather than fabricating, backfilling, or
silently dropping data. It does NOT catch its own gap/pagination errors --
the calling script decides the skip-and-log policy.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Callable
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import requests


BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, "..", ".."))
DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "data", "mining", "intraday_cache")

# Split-adjustment default. Measured (see
# scripts/validate_alpaca_intraday.py): a real NVDA 10-for-1
# split (effective after close 2024-06-07) shows a ~10.04x Friday-close-to-
# Monday-open discontinuity with adjustment="raw" and no discontinuity
# (~1.004x) with adjustment="split" -- confirming Alpaca's "raw" is truly
# unadjusted and "split" is the correct default for a client that will
# hold positions across corporate actions.
DEFAULT_ADJUSTMENT = "split"

_ET = ZoneInfo("America/New_York")
_RTH_START = dtime(9, 30, 0)  # NYSE has no scheduled early-OPEN days, so this stays fixed.

# A real NYSE trading calendar (holidays, half-days), not naive Mon-Fri
# weekday arithmetic. Construction is deterministic, offline, and
# credential-free (pure calendar rules, no network I/O), so a single
# module-level instance is safe and avoids rebuilding it on every call.
_NYSE_CALENDAR = mcal.get_calendar("NYSE")


def _trading_sessions(start: date, end: date) -> dict[date, dtime]:
    """Real NYSE trading days in [start, end] (both inclusive), mapped to
    each session's ACTUAL close time in ET.

    Most sessions close at 16:00 ET, but a real scheduled early close
    (e.g. the day after Thanksgiving) closes at 13:00 ET -- Alpaca's feed
    can still report activity in the 13:00-16:00 window on that specific
    day, which is not part of the regular session and must not be
    accepted as RTH data. `_RTH_START` (9:30) never varies; the upper
    bound must be looked up per day, not hardcoded.
    """
    schedule = _NYSE_CALENDAR.schedule(start_date=start, end_date=end)
    return {
        session_ts.date(): close_utc.tz_convert(_ET).time()
        for session_ts, close_utc in zip(schedule.index, schedule["market_close"])
    }


_CSV_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "TradeCount", "VWAP"]

MAX_RETRIES = 3
MAX_PAGES_PER_REQUEST = 10_000
_REQUEST_TIMEOUT = 30.0

# Real Alpaca tickers are uppercase alphanumeric plus "."/"-"/"_" for share
# classes (e.g. "BRK.B"). `symbol`
# and `adjustment` are both interpolated into a filesystem path (the cache
# key), so both need this same-shaped defense against path traversal, not
# just a non-emptiness check.
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9._-]+$")
# The real Alpaca-documented adjustment values (bars endpoint). Also closes
# off adjustment as a path-traversal vector, since it is part of the
# cache path.
_VALID_ADJUSTMENTS = frozenset({"raw", "split", "dividend", "all"})


def _validate_symbol(symbol: str) -> None:
    if not isinstance(symbol, str) or not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("symbol: safe_ticker_pattern_required")
    # ".." is built entirely from otherwise-allowed characters (each "."
    # is a legitimate share-class separator) -- the character-class check
    # above alone does not catch it. Reject it explicitly.
    if ".." in symbol:
        raise ValueError("symbol: path_traversal_token_rejected")


def _validate_adjustment(adjustment: str) -> None:
    if adjustment not in _VALID_ADJUSTMENTS:
        raise ValueError(
            f"adjustment: must be one of {sorted(_VALID_ADJUSTMENTS)}"
        )


class IntradayDataError(Exception):
    """Base class for this module's own errors."""


class CredentialError(IntradayDataError):
    """Credentials were not supplied by the request-time callback."""


class IntradayDataGapError(IntradayDataError):
    """Alpaca returned no usable (RTH) bars for a requested trading day."""


class PaginationError(IntradayDataError):
    """A page_token chain did not terminate within the bounded page limit."""


class AlpacaRequestError(IntradayDataError):
    """A non-retryable HTTP failure, or a retry budget was exhausted."""


def _environment_credentials() -> tuple[str | None, str | None]:
    """Read only the two accepted names, and only when a request begins."""
    return os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")


def fetch_intraday_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    session: Any = None,
    credentials_fn: Callable[[], tuple[str | None, str | None]] = _environment_credentials,
    adjustment: str = DEFAULT_ADJUSTMENT,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Return RTH-only 1-minute OHLCV bars for [start_date, end_date].

    ``start_date``/``end_date`` are "YYYY-MM-DD", both inclusive. The
    returned frame has a tz-aware ("America/New_York") DatetimeIndex named
    "timestamp" and columns Open/High/Low/Close/Volume/TradeCount/VWAP.

    A real NYSE trading day already fully cached on disk (for this exact
    ``adjustment`` value -- see below) contributes zero HTTP calls. Any
    trading day not yet cached triggers exactly one paginated fetch
    covering the full missing span (not one call per day).

    The on-disk cache key is ``(symbol, trading_date, adjustment)``, not
    just ``(symbol, trading_date)``: two different ``adjustment`` values
    for the same symbol/day are genuinely different bar values across any
    corporate action in that window, and must never share a cache entry
    (which would otherwise make the cache path collide).
    """
    _validate_symbol(symbol)
    _validate_adjustment(adjustment)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("date_range: start_must_not_exceed_end")

    sessions = _trading_sessions(start, end)
    business_days = list(sessions.keys())
    if not business_days:
        return _empty_frame()

    symbol_dir = os.path.join(cache_dir, symbol)
    missing_days = [
        d for d in business_days
        if not os.path.exists(_cache_path(symbol_dir, d, adjustment))
    ]

    if missing_days:
        active_session = session if session is not None else requests.Session()
        bars_by_day = _fetch_range(
            symbol,
            missing_days[0],
            missing_days[-1],
            session=active_session,
            credentials_fn=credentials_fn,
            adjustment=adjustment,
            sleep_fn=sleep_fn,
        )
        for day in missing_days:
            rth_end = sessions[day]
            day_frame = _bars_to_rth_frame(bars_by_day.get(day, []), rth_end)
            if day_frame.empty:
                raise IntradayDataGapError(
                    f"{symbol}: no RTH ([9:30, {rth_end.isoformat()}) ET) bars "
                    f"returned by Alpaca for trading day {day.isoformat()}"
                )
            _write_cache(symbol_dir, day, adjustment, day_frame)

    frames = [_read_cache(symbol_dir, day, adjustment) for day in business_days]
    return pd.concat(frames)


def _empty_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([], name="timestamp", tz=_ET)
    return pd.DataFrame(columns=_CSV_COLUMNS, index=index)


def _cache_path(symbol_dir: str, day: date, adjustment: str) -> str:
    # adjustment is validated against a fixed enum (_validate_adjustment)
    # before ever reaching here, so it is safe to use as a path segment.
    return os.path.join(symbol_dir, adjustment, f"{day.isoformat()}.csv")


def _write_cache(symbol_dir: str, day: date, adjustment: str, frame: pd.DataFrame) -> None:
    """Write the cache file atomically.

    ``fetch_intraday_bars`` treats "file exists" as "cache hit, trust it"
    (``os.path.exists``), so a reader must never be able to observe a
    partially-written file -- whether from a process killed mid-write or
    two fetches racing on the same (symbol, day, adjustment). Write to a
    uniquely-named temp file in the same directory, then ``os.replace()``
    it onto the final path: ``os.replace`` is atomic on both POSIX and
    Windows, so a reader only ever sees the old complete file (if any) or
    the new complete file, never a partial one.
    """
    path = _cache_path(symbol_dir, day, adjustment)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        frame.to_csv(tmp_path, index_label="timestamp")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_cache(symbol_dir: str, day: date, adjustment: str) -> pd.DataFrame:
    frame = pd.read_csv(
        _cache_path(symbol_dir, day, adjustment), index_col="timestamp", parse_dates=["timestamp"]
    )
    index = pd.DatetimeIndex(frame.index)
    # Re-express to the named zone (rather than whatever fixed UTC offset
    # the CSV round-trip parsed) so frames from different DST regimes
    # concatenate onto one consistent tz-aware index.
    frame.index = index.tz_convert(_ET) if index.tz is not None else index.tz_localize(_ET)
    frame.index.name = "timestamp"
    return frame[_CSV_COLUMNS]


def _bars_to_rth_frame(ts_bar_pairs: list[tuple[pd.Timestamp, dict]], rth_end: dtime) -> pd.DataFrame:
    """Filter to the half-open RTH window [9:30, rth_end) ET.

    ``rth_end`` is that SPECIFIC day's real scheduled close (16:00 ET on a
    normal day, 13:00 ET on a real early-close day) -- see
    ``_trading_sessions``. Bar timestamps mark interval START, so a bar
    timestamped exactly at the close covers the post-close minute and
    must be excluded (half-open, not inclusive).
    """
    rows = []
    idx = []
    for ts, bar in sorted(ts_bar_pairs, key=lambda pair: pair[0]):
        if not (_RTH_START <= ts.time() < rth_end):
            continue
        idx.append(ts)
        rows.append({
            "Open": bar["o"],
            "High": bar["h"],
            "Low": bar["l"],
            "Close": bar["c"],
            "Volume": bar["v"],
            "TradeCount": bar.get("n"),
            "VWAP": bar.get("vw"),
        })
    if not rows:
        return _empty_frame()
    index = pd.DatetimeIndex(idx, name="timestamp")
    return pd.DataFrame(rows, index=index, columns=_CSV_COLUMNS)


def _fetch_range(
    symbol: str,
    fetch_start: date,
    fetch_end: date,
    *,
    session: Any,
    credentials_fn: Callable[[], tuple[str | None, str | None]],
    adjustment: str,
    sleep_fn: Callable[[float], None],
) -> dict[date, list[tuple[pd.Timestamp, dict]]]:
    """Fetch one full paginated span and bucket every returned bar by its
    ET calendar date -- the RTH filter itself is applied by the caller.
    """
    start_dt_et = datetime.combine(fetch_start, dtime.min).replace(tzinfo=_ET)
    end_dt_et = datetime.combine(fetch_end + timedelta(days=1), dtime.min).replace(tzinfo=_ET)
    start_utc = start_dt_et.astimezone(timezone.utc)
    end_utc = end_dt_et.astimezone(timezone.utc)

    base_params: dict[str, Any] = {
        "timeframe": "1Min",
        "start": _rfc3339(start_utc),
        "end": _rfc3339(end_utc),
        "limit": 10_000,
        "adjustment": adjustment,
        "feed": "sip",
        "sort": "asc",
    }
    url = BARS_URL.format(symbol=symbol)

    bars_by_day: dict[date, list[tuple[pd.Timestamp, dict]]] = {}
    page_token: str | None = None
    pages = 0
    while True:
        params = dict(base_params)
        if page_token is not None:
            params["page_token"] = page_token
        payload = _request_page(session, url, params, credentials_fn, sleep_fn)

        bars = payload.get("bars")
        # Observed against the real Alpaca API: a date
        # range before the vendor's own coverage returns HTTP 200 with
        # {"bars": null, ...} -- a real null, not an empty list. Treat that
        # the same as zero bars (surfaced as a gap by the caller), not as a
        # malformed response.
        if bars is None:
            bars = []
        if not isinstance(bars, list):
            raise AlpacaRequestError(f"{symbol}: malformed response, 'bars' must be a list or null")
        for bar in bars:
            ts = pd.Timestamp(bar["t"]).tz_convert(_ET)
            bars_by_day.setdefault(ts.date(), []).append((ts, bar))

        page_token = payload.get("next_page_token")
        pages += 1
        if page_token is None:
            break
        if pages >= MAX_PAGES_PER_REQUEST:
            raise PaginationError(
                f"{symbol}: page_token chain exceeded {MAX_PAGES_PER_REQUEST} pages "
                "without terminating"
            )

    return bars_by_day


def _request_page(
    session: Any,
    url: str,
    params: dict[str, Any],
    credentials_fn: Callable[[], tuple[str | None, str | None]],
    sleep_fn: Callable[[float], None],
) -> dict:
    key_id, secret = credentials_fn()
    if not key_id or not secret:
        raise CredentialError(
            "credentials: ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY missing or empty"
        )
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}

    attempts = 0
    while True:
        try:
            # allow_redirects=False: unlike the standard Authorization
            # header, requests does NOT strip custom headers (our
            # APCA-API-KEY-ID/APCA-API-SECRET-KEY) on a cross-origin
            # redirect -- never auto-follow, so credentials can never be
            # sent to an unexpected host. A 3xx is a hard failure below;
            # this module never follows one itself either.
            response = session.get(
                url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.exceptions.RequestException as exc:
            # A real transient network failure (connection reset, DNS
            # blip, timeout raised by the transport itself) rather than an
            # HTTP error response -- retry it exactly like a 429/5xx.
            if attempts >= MAX_RETRIES:
                raise AlpacaRequestError(
                    f"alpaca bars request failed after {MAX_RETRIES} retries: "
                    f"network error ({type(exc).__name__})"
                ) from None
            sleep_fn(min(2.0 ** attempts, 30.0))
            attempts += 1
            continue

        status = getattr(response, "status_code", None)
        if status == 200:
            return response.json()
        if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
            if attempts >= MAX_RETRIES:
                raise AlpacaRequestError(
                    f"alpaca bars request failed after {MAX_RETRIES} retries: status {status}"
                )
            sleep_fn(min(2.0 ** attempts, 30.0))
            attempts += 1
            continue
        if isinstance(status, int) and 300 <= status <= 399:
            raise AlpacaRequestError(
                f"alpaca bars request failed: status {status} (redirect, not followed)"
            )
        raise AlpacaRequestError(f"alpaca bars request failed: status {status}")


def _rfc3339(dt_utc: datetime) -> str:
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
