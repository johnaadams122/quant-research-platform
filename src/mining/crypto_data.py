"""Free, keyless public REST clients for crypto market data used by mining
(Kraken spot OHLC + Kraken derivatives perpetual-futures historical funding
rate). Matches ``external_data.py``'s and ``edgar_data.py``'s style: raw
HTTP kept injectable via ``http_get``, fail-closed schema validation, and
a pandas return type ``.attrs``-tagged the same way the FRED/FINRA/EDGAR
drivers are, so an eventual strategy-family module can consume this
exactly like it consumes those.

Endpoints (both free, keyless, no auth, no rate-limit key required):
Kraken spot ``/0/public/OHLC`` (``https://api.kraken.com``) and Kraken
derivatives (futures) ``/derivatives/api/v3/historical-funding-rates``
(``https://futures.kraken.com``). Symbols are exposed to callers as
``BTC-USD``/``ETH-USD`` (this repo's familiar ticker style -- see
``data_pipeline.py``'s ticker table) and mapped internally to Kraken's own
pair/symbol names, which DIFFER between the two APIs (spot uses
``KRAKEN_SPOT_SYMBOL_MAP``: ``XBTUSD``/``ETHUSD``; futures uses
``KRAKEN_FUTURES_SYMBOL_MAP``: ``PF_XBTUSD``/``PF_ETHUSD`` -- Kraken's own
perpetual-future naming convention).

Why Kraken and not Binance: both ``api.binance.com`` and
``fapi.binance.com`` return HTTP 451 to US clients (Binance's Terms
'b. Eligibility' geo-block) -- a legal restriction, not a bug. Both Kraken
endpoints below were verified live (real HTTP 200 responses with real
market data, not just documentation): spot OHLC for both BTC-USD and
ETH-USD, and futures historical funding rates for both BTC-USD and ETH-USD
perpetuals.

Kraken-specific behaviour this module handles, each because Kraken's real
API genuinely works this way:
- Kraken's spot API signals failure via a **200 response with a non-empty
  top-level ``error`` array** (``{"error": ["EQuery:Unknown asset
  pair"]}``), not an HTTP error code -- verified live. ``raise_for_status``
  alone would not catch this, so the payload's own ``error`` field is
  checked explicitly.
- Kraken's spot OHLC response nests candles under a vendor-chosen key that
  echoes an INTERNAL pair name, not the requested pair string (e.g.
  requesting ``pair=XBTUSD`` returns data under ``"XXBTZUSD"``; ``ETHUSD``
  comes back under ``"XETHZUSD"``) -- verified live. The one data key
  (whichever it is) is found generically by excluding the sibling
  ``"last"`` cursor key, rather than assumed by name.
- Kraken's futures API has no ``start_time``/``end_time``/``limit``
  pagination on this endpoint (verified against the live docs and a real
  request: it simply returns its full retained history, ~8900 hourly rows
  for BTC spanning roughly a year, with only ``symbol`` as a parameter) --
  so this module exposes no pagination knobs rather than faking them.
- Kraken's futures error shape on a bad request is
  ``{"status":"BAD_REQUEST","result":"error","errors":[...]}`` with a real
  HTTP 400 (verified live) -- caught by ``raise_for_status`` for the HTTP
  layer, with a payload-level ``result != "success"`` check kept too, in
  case Kraken (or a future caller's fake) ever returns a non-2xx-coded
  failure body.
- Kraken funding-rate records carry both an absolute ``fundingRate`` and a
  dimensionless ``relativeFundingRate``; this module returns
  ``relativeFundingRate`` (a dimensionless rate, e.g. ``0.0001`` = 1bp per
  settlement period) so downstream consumers see a rate, not an absolute
  price-quoted amount. Kraken settles funding roughly hourly (not the 8h
  cadence common on other venues) -- descriptive only, no code depends on
  the cadence.

Revision/restatement note (this pipeline requires cost/data-source
honesty about revision behavior -- see
``external_data.FRED_VERIFIED_UNREVISED_SERIES``'s docstring for the same
discipline applied to a FRED series): once a Kraken OHLC candle's period
has fully elapsed, and once a funding settlement has actually occurred,
NEITHER value is ever revised or restated after the fact -- unlike many
FRED series (CPIAUCSL, GDP, payrolls, all repeatedly revised, sometimes
years later), a crypto exchange's already-settled trade/funding ledger has
no revision pipeline; there is no analog of a "preliminary" vs. "final"
print. The ONE real exception: the *currently forming* (most recent)
candle is mutable until its own period closes -- Kraken's own docs state
the last OHLC row is always "the current, not-yet-committed timeframe".
``fetch_kraken_spot_ohlcv`` flags this per-fetch via
``.attrs["last_row_closed"]`` rather than silently dropping the row, so a
caller doing anything ledger-sensitive can check it and decide, mirroring
``fetch_fred_series_point_in_time``'s own "supplies correct values only,
never availability timing" disclaimer. Funding-rate records carry no such
caveat: Kraken only ever returns a settlement record for a funding period
that has already occurred (never a currently accruing one), so every row
``fetch_kraken_perp_funding_rate`` returns is already final.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional
import hashlib

import numpy as np
import pandas as pd
import requests

KRAKEN_SPOT_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
KRAKEN_FUTURES_FUNDING_RATE_URL = (
    "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates")

# This repo's own ticker convention elsewhere (data_pipeline.py) uses plain
# tickers; crypto pairs are exposed to callers the same familiar way
# (BTC-USD/ETH-USD) and mapped internally to each Kraken API's own symbol
# naming -- see the module docstring for why spot and futures need SEPARATE
# maps (their vendor symbol conventions genuinely differ).
KRAKEN_SPOT_SYMBOL_MAP = {
    "BTC-USD": "XBTUSD",
    "ETH-USD": "ETHUSD",
}
KRAKEN_FUTURES_SYMBOL_MAP = {
    "BTC-USD": "PF_XBTUSD",
    "ETH-USD": "PF_ETHUSD",
}

OHLC_COLUMNS = ("open", "high", "low", "close", "volume")
# Kraken's documented valid `interval` values, in minutes.
KRAKEN_VALID_INTERVALS = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)


def _spot_symbol(symbol: str) -> str:
    if symbol not in KRAKEN_SPOT_SYMBOL_MAP:
        raise ValueError(
            f"crypto symbol {symbol!r} not supported -- only "
            f"{sorted(KRAKEN_SPOT_SYMBOL_MAP)} are mapped to a Kraken spot pair")
    return KRAKEN_SPOT_SYMBOL_MAP[symbol]


def _futures_symbol(symbol: str) -> str:
    if symbol not in KRAKEN_FUTURES_SYMBOL_MAP:
        raise ValueError(
            f"crypto symbol {symbol!r} not supported -- only "
            f"{sorted(KRAKEN_FUTURES_SYMBOL_MAP)} are mapped to a Kraken "
            f"futures symbol")
    return KRAKEN_FUTURES_SYMBOL_MAP[symbol]


def _validate_interval(interval: int) -> None:
    if interval not in KRAKEN_VALID_INTERVALS:
        raise ValueError(
            f"interval must be one of {KRAKEN_VALID_INTERVALS}, got {interval}")


def _to_unix_seconds(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("since must be timezone-aware")
    return int(value.astimezone(timezone.utc).timestamp())


def _retrieved_at_iso(retrieved_at: datetime) -> str:
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    return (retrieved_at.astimezone(timezone.utc).isoformat()
            .replace("+00:00", "Z"))


def fetch_kraken_spot_ohlcv(
        symbol: str, *, interval: int = 1440, since: Optional[datetime] = None,
        http_get: Callable = requests.get, timeout: float = 30.0,
        clock: Callable[[], datetime] | None = None) -> pd.DataFrame:
    """Fetch spot OHLC candles for BTC-USD/ETH-USD (daily by default).

    Returns a DataFrame indexed by each candle's OPEN time (UTC,
    tz-naive ``DatetimeIndex`` to match this repo's other pandas series),
    with columns ``open``/``high``/``low``/``close``/``volume`` (float) --
    Kraken's own ``vwap``/``count`` fields are dropped to keep this
    connector's column contract identical to the rest of this module.
    Fails closed on a non-2xx response (via ``raise_for_status``), on
    Kraken's own in-body ``error`` array, on a missing/ambiguous result
    data key, a malformed/short candle row, or a non-numeric OHLCV field --
    same convention as ``external_data.fetch_fred_series``. See the module
    docstring for ``.attrs["last_row_closed"]`` and the revision-safety
    rationale.
    """
    _validate_interval(interval)
    kraken_symbol = _spot_symbol(symbol)
    params: dict = {"pair": kraken_symbol, "interval": interval}
    if since is not None:
        params["since"] = _to_unix_seconds(since)

    response = http_get(KRAKEN_SPOT_OHLC_URL, params=params, timeout=timeout)
    response.raise_for_status()
    raw_sha256 = hashlib.sha256(bytes(response.content)).hexdigest()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(
            f"Kraken OHLC {symbol}: expected a dict payload, "
            f"got {type(payload).__name__}")
    errors = payload.get("error")
    if errors:
        raise ValueError(f"Kraken OHLC {symbol}: request failed -- {errors}")

    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(
            f"Kraken OHLC {symbol}: expected a dict 'result', "
            f"got {type(result).__name__}")
    data_keys = [k for k in result if k != "last"]
    if len(data_keys) != 1:
        raise ValueError(
            f"Kraken OHLC {symbol}: expected exactly one data key in "
            f"'result' (excluding 'last'), got {data_keys!r}")
    candles = result[data_keys[0]]
    if not isinstance(candles, list) or not candles:
        raise ValueError(
            f"Kraken OHLC {symbol}: expected a non-empty list of candles, "
            f"got {type(candles).__name__}")

    open_times = []
    rows = []
    for i, row in enumerate(candles):
        if not isinstance(row, list) or len(row) < 7:
            raise ValueError(
                f"Kraken OHLC {symbol}: candle[{i}] malformed (expected "
                f"a >=7-element list), got {row!r}")
        time_s, open_, high, low, close, _vwap, volume = row[:7]
        open_times.append(time_s)
        rows.append({"open": open_, "high": high, "low": low,
                     "close": close, "volume": volume})

    frame = pd.DataFrame(rows, columns=list(OHLC_COLUMNS))
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().to_numpy().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(
            f"Kraken OHLC {symbol}: non-numeric/non-finite OHLCV values")

    index = pd.to_datetime(open_times, unit="s", utc=True, errors="coerce")
    if pd.isna(index).any():
        raise ValueError(f"Kraken OHLC {symbol}: unparseable candle open time")

    now = clock() if clock is not None else datetime.now(timezone.utc)
    last_close_time = pd.Timestamp(open_times[-1], unit="s", tz="UTC") + \
        pd.Timedelta(minutes=interval)
    last_row_closed = bool(last_close_time < pd.Timestamp(now))

    result_frame = numeric.copy()
    result_frame.index = pd.DatetimeIndex(index).tz_convert(None)
    result_frame.index.name = "open_time"
    result_frame.attrs.update({
        "source_url": KRAKEN_SPOT_OHLC_URL,
        "symbol": symbol,
        "kraken_symbol": kraken_symbol,
        "interval": interval,
        "date_semantics": "candle_open_time_utc",
        "point_in_time": True,
        "ledgerable": True,
        "ledger_block_reason": None,
        "last_row_closed": last_row_closed,
        "retrieved_at": _retrieved_at_iso(now),
        "raw_sha256": raw_sha256,
    })
    return result_frame


def fetch_kraken_perp_funding_rate(
        symbol: str, *, http_get: Callable = requests.get, timeout: float = 30.0,
        clock: Callable[[], datetime] | None = None) -> pd.Series:
    """Fetch USD-margined perpetual funding-rate settlement history for
    BTC-USD/ETH-USD (Kraken's derivatives perpetual future, roughly hourly
    settlement cadence).

    Returns a ``Series`` indexed by each settlement's ``timestamp`` (UTC,
    tz-naive ``DatetimeIndex``), named ``funding_rate`` (float, Kraken's
    dimensionless ``relativeFundingRate`` field -- e.g. ``0.0001`` = 1bp
    per settlement period). Fails closed the same way as
    ``fetch_kraken_spot_ohlcv`` -- see that function and the module
    docstring for the shared revision-safety rationale and the documented
    Kraken-specific handling (no pagination params on this endpoint; a
    payload-level ``result`` status check).
    """
    kraken_symbol = _futures_symbol(symbol)
    params: dict = {"symbol": kraken_symbol}

    response = http_get(KRAKEN_FUTURES_FUNDING_RATE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    raw_sha256 = hashlib.sha256(bytes(response.content)).hexdigest()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(
            f"Kraken funding rate {symbol}: expected a dict payload, "
            f"got {type(payload).__name__}")
    if payload.get("result") != "success":
        raise ValueError(
            f"Kraken funding rate {symbol}: request failed -- "
            f"{payload.get('errors', payload)}")

    records = payload.get("rates")
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"Kraken funding rate {symbol}: expected a non-empty list of "
            f"records, got {type(records).__name__}")

    funding_times = []
    rates = []
    for i, record in enumerate(records):
        if (not isinstance(record, dict) or "timestamp" not in record
                or "relativeFundingRate" not in record):
            raise ValueError(
                f"Kraken funding rate {symbol}: record[{i}] missing "
                f"timestamp/relativeFundingRate, got {record!r}")
        funding_times.append(record["timestamp"])
        rates.append(record["relativeFundingRate"])

    index = pd.to_datetime(funding_times, utc=True, errors="coerce")
    if pd.isna(index).any():
        raise ValueError(f"Kraken funding rate {symbol}: unparseable timestamp")
    values = pd.to_numeric(pd.Series(rates), errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            f"Kraken funding rate {symbol}: non-numeric/non-finite "
            f"relativeFundingRate")

    result_series = pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(index).tz_convert(None),
        name="funding_rate",
    )
    result_series = result_series[
        ~result_series.index.duplicated(keep="last")].sort_index()

    now = clock() if clock is not None else datetime.now(timezone.utc)
    result_series.attrs.update({
        "source_url": KRAKEN_FUTURES_FUNDING_RATE_URL,
        "symbol": symbol,
        "kraken_symbol": kraken_symbol,
        "date_semantics": "funding_settlement_time_utc",
        "point_in_time": True,
        "ledgerable": True,
        "ledger_block_reason": None,
        "retrieved_at": _retrieved_at_iso(now),
        "raw_sha256": raw_sha256,
    })
    return result_series
