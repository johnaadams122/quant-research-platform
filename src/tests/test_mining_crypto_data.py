"""Tests for src/mining/crypto_data.py. Network-free: both fetch functions
are exercised via an injected fake ``http_get`` (same pattern as
``test_mining_external_data.py``'s ``FakeResponse`` / ``test_mining_edgar_
data.py``'s ``_FakeResponse``) -- no real network calls anywhere in this
file. Response shapes below mirror REAL bodies captured live from Kraken's
public spot (``api.kraken.com``) and futures (``futures.kraken.com``) APIs
-- see ``crypto_data.py``'s module docstring for the
verified-reachability note.
"""
import hashlib
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import crypto_data


class FakeResponse:
    def __init__(self, json_body=None, content=None, error=None, status_code=200):
        self._json_body = json_body
        self.content = content if content is not None else b"raw-bytes"
        self._error = error
        self.status_code = status_code

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._json_body


def _kraken_ohlc_row(time_s, o, h, l, c, vwap, vol, count):
    return [time_s, o, h, l, c, vwap, vol, count]


def _ohlc_payload(rows, result_key="XXBTZUSD", last=0):
    body = {"last": last}
    body[result_key] = rows
    return {"error": [], "result": body}


# --------------------------------------------------------------- spot OHLC


def test_fetch_kraken_spot_ohlcv_parses_candles_and_records_provenance():
    rows = [
        _kraken_ohlc_row(1_700_000_000, "100.0", "110.0", "90.0", "105.0",
                         "102.0", "12.5", 7),
        _kraken_ohlc_row(1_700_086_400, "105.0", "108.0", "95.0", "96.0",
                         "101.0", "20.0", 9),
    ]
    payload = _ohlc_payload(rows)
    raw_bytes = b"payload-bytes"
    response = FakeResponse(json_body=payload, content=raw_bytes)
    observed = {}

    def fake_get(url, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        return response

    out = crypto_data.fetch_kraken_spot_ohlcv(
        "BTC-USD", http_get=fake_get, timeout=9.0,
        clock=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc))

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert out.iloc[0].to_dict() == {
        "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
        "volume": 12.5}
    assert out.index[0] == pd.Timestamp(1_700_000_000, unit="s")
    assert out.index.tz is None

    assert observed["url"] == crypto_data.KRAKEN_SPOT_OHLC_URL
    assert observed["params"] == {"pair": "XBTUSD", "interval": 1440}
    assert observed["timeout"] == 9.0

    assert out.attrs["source_url"] == crypto_data.KRAKEN_SPOT_OHLC_URL
    assert out.attrs["symbol"] == "BTC-USD"
    assert out.attrs["kraken_symbol"] == "XBTUSD"
    assert out.attrs["interval"] == 1440
    assert out.attrs["point_in_time"] is True
    assert out.attrs["ledgerable"] is True
    assert out.attrs["last_row_closed"] is True
    assert out.attrs["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert out.attrs["retrieved_at"] == "2026-09-16T00:00:00Z"


def test_fetch_kraken_spot_ohlcv_flags_an_unclosed_last_candle():
    now = datetime(2026, 9, 16, tzinfo=timezone.utc)
    almost_now = int(now.timestamp()) - 3600  # candle opened 1h ago
    rows = [_kraken_ohlc_row(almost_now, "100.0", "110.0", "90.0", "105.0",
                             "102.0", "12.5", 7)]
    response = FakeResponse(json_body=_ohlc_payload(rows))
    out = crypto_data.fetch_kraken_spot_ohlcv(
        "ETH-USD", http_get=lambda *a, **k: response, clock=lambda: now)
    # daily interval (1440 min) means close_time = open + 86400s, still future
    assert out.attrs["last_row_closed"] is False


def test_fetch_kraken_spot_ohlcv_maps_eth_symbol():
    rows = [_kraken_ohlc_row(1_700_000_000, "1", "1", "1", "1", "1", "1", 1)]
    response = FakeResponse(json_body=_ohlc_payload(rows, result_key="XETHZUSD"))
    observed = {}

    def fake_get(url, **kwargs):
        observed.update(kwargs)
        return response

    crypto_data.fetch_kraken_spot_ohlcv("ETH-USD", http_get=fake_get)
    assert observed["params"]["pair"] == "ETHUSD"


def test_fetch_kraken_spot_ohlcv_rejects_unsupported_symbol():
    with pytest.raises(ValueError, match="not supported"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "DOGE-USD", http_get=lambda *a, **k: FakeResponse(json_body={}))


def test_fetch_kraken_spot_ohlcv_propagates_http_failure():
    response = FakeResponse(error=RuntimeError("HTTP 503"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_spot_ohlcv_rejects_kraken_error_array():
    # Real observed shape (HTTP 200, error surfaced in-body):
    # {"error":["EQuery:Unknown asset pair"]}
    payload = {"error": ["EQuery:Unknown asset pair"]}
    response = FakeResponse(json_body=payload)
    with pytest.raises(ValueError, match="EQuery:Unknown asset pair"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_spot_ohlcv_rejects_missing_or_ambiguous_result_key():
    payload = {"error": [], "result": {"last": 123}}  # no data key at all
    response = FakeResponse(json_body=payload)
    with pytest.raises(ValueError, match="exactly one"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_spot_ohlcv_rejects_non_dict_result_payload():
    payload = {"error": [], "result": ["not", "a", "dict"]}
    response = FakeResponse(json_body=payload)
    with pytest.raises(ValueError, match="result"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_spot_ohlcv_rejects_short_malformed_row():
    payload = _ohlc_payload([[1_700_000_000, "1", "2"]])
    response = FakeResponse(json_body=payload)
    with pytest.raises(ValueError, match="malformed"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_spot_ohlcv_rejects_non_numeric_ohlcv_field():
    rows = [_kraken_ohlc_row(1_700_000_000, "not-a-number", "2", "1", "1.5",
                             "1.5", "1", 1)]
    response = FakeResponse(json_body=_ohlc_payload(rows))
    with pytest.raises(ValueError, match="non-numeric"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_spot_ohlcv_rejects_out_of_range_interval():
    with pytest.raises(ValueError, match="interval must be one of"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", interval=7,
            http_get=lambda *a, **k: FakeResponse(json_body=_ohlc_payload([])))


def test_fetch_kraken_spot_ohlcv_passes_through_since_as_unix_seconds():
    rows = [_kraken_ohlc_row(1_700_000_000, "1", "1", "1", "1", "1", "1", 1)]
    response = FakeResponse(json_body=_ohlc_payload(rows))
    observed = {}

    def fake_get(url, **kwargs):
        observed.update(kwargs)
        return response

    since = datetime(2023, 11, 14, tzinfo=timezone.utc)
    crypto_data.fetch_kraken_spot_ohlcv("BTC-USD", http_get=fake_get, since=since)
    assert observed["params"]["since"] == int(since.timestamp())


def test_fetch_kraken_spot_ohlcv_rejects_naive_since_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        crypto_data.fetch_kraken_spot_ohlcv(
            "BTC-USD", http_get=lambda *a, **k: FakeResponse(json_body=_ohlc_payload([])),
            since=datetime(2023, 11, 14))


# --------------------------------------------------------- funding rate


def _funding_record(timestamp_iso, relative_rate, absolute_rate="0.5"):
    return {"timestamp": timestamp_iso, "fundingRate": absolute_rate,
            "relativeFundingRate": relative_rate}


def _funding_payload(rates, result="success"):
    return {"result": result, "serverTime": "2026-09-17T04:56:39.055Z",
            "rates": rates}


def test_fetch_kraken_perp_funding_rate_parses_and_records_provenance():
    payload = _funding_payload([
        _funding_record("2025-09-10T08:00:00Z", "5.827e-06"),
        _funding_record("2025-09-10T09:00:00Z", "-2.460e-06"),
    ])
    raw_bytes = b"funding-payload"
    response = FakeResponse(json_body=payload, content=raw_bytes)
    observed = {}

    def fake_get(url, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        return response

    out = crypto_data.fetch_kraken_perp_funding_rate(
        "BTC-USD", http_get=fake_get, timeout=8.0,
        clock=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc))

    assert out.name == "funding_rate"
    assert list(out.values) == pytest.approx([5.827e-06, -2.460e-06])
    assert out.index.tz is None
    assert observed["url"] == crypto_data.KRAKEN_FUTURES_FUNDING_RATE_URL
    assert observed["params"] == {"symbol": "PF_XBTUSD"}
    assert observed["timeout"] == 8.0

    assert out.attrs["source_url"] == crypto_data.KRAKEN_FUTURES_FUNDING_RATE_URL
    assert out.attrs["symbol"] == "BTC-USD"
    assert out.attrs["kraken_symbol"] == "PF_XBTUSD"
    assert out.attrs["point_in_time"] is True
    assert out.attrs["ledgerable"] is True
    assert out.attrs["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert out.attrs["retrieved_at"] == "2026-09-16T00:00:00Z"


def test_fetch_kraken_perp_funding_rate_dedupes_keeping_last_and_sorts():
    payload = _funding_payload([
        _funding_record("2025-09-10T09:00:00Z", "0.0002"),
        _funding_record("2025-09-10T08:00:00Z", "0.0001"),
        _funding_record("2025-09-10T08:00:00Z", "0.00015"),
    ])
    response = FakeResponse(json_body=payload)
    out = crypto_data.fetch_kraken_perp_funding_rate(
        "BTC-USD", http_get=lambda *a, **k: response)
    assert list(out.values) == pytest.approx([0.00015, 0.0002])
    assert out.index.is_monotonic_increasing


def test_fetch_kraken_perp_funding_rate_rejects_unsupported_symbol():
    with pytest.raises(ValueError, match="not supported"):
        crypto_data.fetch_kraken_perp_funding_rate(
            "DOGE-USD", http_get=lambda *a, **k: FakeResponse(json_body={}))


def test_fetch_kraken_perp_funding_rate_propagates_http_failure():
    response = FakeResponse(error=RuntimeError("HTTP 400"))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        crypto_data.fetch_kraken_perp_funding_rate(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_perp_funding_rate_rejects_empty_list():
    response = FakeResponse(json_body=_funding_payload([]))
    with pytest.raises(ValueError, match="non-empty list"):
        crypto_data.fetch_kraken_perp_funding_rate(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_perp_funding_rate_rejects_missing_fields():
    payload = _funding_payload([{"timestamp": "2025-09-10T08:00:00Z"}])
    response = FakeResponse(json_body=payload)
    with pytest.raises(ValueError, match="missing timestamp/relativeFundingRate"):
        crypto_data.fetch_kraken_perp_funding_rate(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_perp_funding_rate_rejects_non_numeric_rate():
    payload = _funding_payload([_funding_record("2025-09-10T08:00:00Z", "not-a-number")])
    response = FakeResponse(json_body=payload)
    with pytest.raises(ValueError, match="non-numeric"):
        crypto_data.fetch_kraken_perp_funding_rate(
            "BTC-USD", http_get=lambda *a, **k: response)


def test_fetch_kraken_perp_funding_rate_rejects_result_not_success():
    # Real observed shape on an invalid symbol (HTTP 400 body):
    # {"status":"BAD_REQUEST","result":"error","errors":[{"code":11,...}],...}
    payload = {"status": "BAD_REQUEST", "result": "error",
               "errors": [{"code": 11, "message": "Argument invalid: symbol"}],
               "serverTime": "2026-09-17T04:58:12.762Z"}
    response = FakeResponse(json_body=payload)
    with pytest.raises(ValueError, match="request failed"):
        crypto_data.fetch_kraken_perp_funding_rate(
            "BTC-USD", http_get=lambda *a, **k: response)
