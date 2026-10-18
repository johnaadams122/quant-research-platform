import json
import os
import sys
from datetime import datetime, timezone
import hashlib

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import external_data
from mining.external_data import fetch_fred_series


class FakeResponse:
    def __init__(self, text="", error=None, content=None, json_body=None):
        self.text = text
        self._error = error
        self.content = text.encode("utf-8") if content is None else content
        self._json_body = json_body

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._json_body


def test_fetch_fred_series_parses_missing_values_sorts_and_dedupes():
    response = FakeResponse(
        "DATE,RMFSL\n"
        "2024-02-01,.\n"
        "2024-01-01,100.0\n"
        "2024-01-01,101.0\n")
    observed = {}

    def fake_get(url, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        return response

    out = fetch_fred_series("RMFSL", http_get=fake_get, timeout=7.5)

    assert list(out.index) == [pd.Timestamp("2024-01-01")]
    assert out.iloc[0] == pytest.approx(101.0)
    assert out.name == "RMFSL"
    assert observed == {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv",
        "params": {"id": "RMFSL"},
        "timeout": 7.5,
    }


def test_fetch_fred_series_accepts_current_observation_date_header():
    response = FakeResponse("observation_date,WALCL\n2024-01-03,7500000\n")
    out = fetch_fred_series("WALCL", http_get=lambda *a, **k: response)
    assert out.loc[pd.Timestamp("2024-01-03")] == pytest.approx(7_500_000.0)


def test_fetch_fred_series_rejects_missing_value_column():
    response = FakeResponse("DATE,OTHER\n2024-01-01,1.0\n")
    with pytest.raises(ValueError, match="missing required column RMFSL"):
        fetch_fred_series("RMFSL", http_get=lambda *a, **k: response)


def test_fetch_fred_series_rejects_no_usable_rows():
    response = FakeResponse("DATE,RMFSL\n2024-01-01,.\nnot-a-date,5\n")
    with pytest.raises(ValueError, match="no usable finite observations"):
        fetch_fred_series("RMFSL", http_get=lambda *a, **k: response)


def test_fetch_fred_series_propagates_http_failure():
    response = FakeResponse("", error=RuntimeError("HTTP 503"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        fetch_fred_series("RMFSL", http_get=lambda *a, **k: response)


def test_fetch_fred_series_rejects_explicit_nonfinite_values():
    response = FakeResponse(
        "DATE,RMFSL\n2024-01-01,100.0\n2024-02-01,inf\n")
    with pytest.raises(ValueError, match="non-finite"):
        fetch_fred_series("RMFSL", http_get=lambda *a, **k: response)


def test_fetch_fred_series_rejects_unexpected_schema_columns():
    response = FakeResponse(
        "DATE,RMFSL,unexpected_revision_flag\n2024-01-01,100.0,1\n")
    with pytest.raises(ValueError, match="unexpected columns"):
        fetch_fred_series("RMFSL", http_get=lambda *a, **k: response)


def test_fetch_fred_series_marks_reference_dated_current_history_nonledgerable():
    response = FakeResponse("DATE,RMFSL\n2024-01-01,100.0\n")
    out = fetch_fred_series("RMFSL", http_get=lambda *a, **k: response)
    assert out.attrs["date_semantics"] == "reference_date_current_history"
    assert out.attrs["point_in_time"] is False
    assert out.attrs["ledgerable"] is False
    assert "publication" in out.attrs["ledger_block_reason"]


FINRA_COLUMNS = [
    "Year-Month",
    "Debit Balances in Customers' Securities Margin Accounts",
    "Free Credit Balances in Customers' Cash Accounts",
    "Free Credit Balances in Customers' Securities Margin Accounts",
]


def _finra_workbook(rows):
    return {"Customer Margin Balances": pd.DataFrame(rows, columns=FINRA_COLUMNS)}


def test_parse_finra_margin_debt_validates_and_returns_vintage_metadata():
    raw = b"real workbook bytes"
    observed = {}

    def fake_reader(stream, **kwargs):
        observed["raw"] = stream.read()
        observed.update(kwargs)
        return _finra_workbook([
            ["2026-01", 900.0, 10.0, 20.0],
            ["2025-12", 800.0, 9.0, 19.0],
        ])

    snapshot = external_data.parse_finra_margin_debt(
        raw,
        excel_reader=fake_reader,
        retrieved_at=datetime(2026, 8, 30, 4, 15, tzinfo=timezone.utc),
    )

    assert list(snapshot.series.index) == [
        pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-31")]
    assert list(snapshot.series) == [800.0, 900.0]
    assert snapshot.series.name == "finra_margin_debt_usd_millions"
    assert snapshot.metadata == {
        "source_url": external_data.FINRA_MARGIN_STATISTICS_URL,
        "retrieved_at": "2026-08-30T04:15:00Z",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "worksheet": "Customer Margin Balances",
        "units": "USD millions",
        "date_semantics": "reference_month_end",
    }
    assert observed == {
        "raw": raw,
        "sheet_name": None,
        "header": 0,
    }


def test_parse_finra_allows_historical_blank_in_unused_free_credit_column():
    workbook = _finra_workbook([
        ["2010-01", 266_455.0, 281_035.0, float("nan")],
        ["2010-02", 255_000.0, 280_000.0, 10_000.0],
    ])
    snapshot = external_data.parse_finra_margin_debt(
        b"raw",
        excel_reader=lambda *args, **kwargs: workbook,
        retrieved_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    assert list(snapshot.series) == [266_455.0, 255_000.0]


@pytest.mark.parametrize(
    ("workbook", "message"),
    [
        ({"Renamed Sheet": pd.DataFrame(columns=FINRA_COLUMNS)}, "worksheet"),
        ({"Customer Margin Balances": pd.DataFrame(columns=FINRA_COLUMNS[:-1])},
         "columns"),
        (_finra_workbook([
            ["2026-01", float("inf"), 10.0, 20.0],
        ]), "finite numeric"),
        (_finra_workbook([
            ["2026-01", 900.0, 10.0, 20.0],
            ["2026-01", 901.0, 11.0, 21.0],
        ]), "duplicate"),
    ],
)
def test_parse_finra_margin_debt_rejects_schema_and_value_drift(
        workbook, message):
    with pytest.raises(ValueError, match=message):
        external_data.parse_finra_margin_debt(
            b"raw",
            excel_reader=lambda *args, **kwargs: workbook,
            retrieved_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        )


def _fred_api_payload(observations):
    return {
        "realtime_start": "2026-09-15", "realtime_end": "2026-09-15",
        "observation_start": "1776-07-04", "observation_end": "9999-12-31",
        "units": "lin", "output_type": 1, "file_type": "json",
        "order_by": "observation_date", "sort_order": "asc",
        "count": len(observations), "offset": 0, "limit": 100000,
        "observations": observations,
    }


def test_fetch_fred_series_point_in_time_parses_and_marks_ledgerable(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "a" * 32)
    payload = _fred_api_payload([
        {"realtime_start": "2026-09-15", "realtime_end": "9999-12-31",
         "date": "2026-06-01", "value": "3047.2"},
        {"realtime_start": "2026-09-15", "realtime_end": "9999-12-31",
         "date": "2026-07-01", "value": "."},
    ])
    observed = {}

    def fake_get(url, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        return FakeResponse(json_body=payload)

    out = external_data.fetch_fred_series_point_in_time(
        "RMFSL", http_get=fake_get, timeout=12.0)

    assert list(out.index) == [pd.Timestamp("2026-06-01")]
    assert out.iloc[0] == pytest.approx(3047.2)
    assert out.name == "RMFSL"
    assert out.attrs["point_in_time"] is True
    assert out.attrs["ledgerable"] is True
    assert out.attrs["ledger_block_reason"] is None
    assert observed["url"] == external_data.FRED_API_OBSERVATIONS_URL
    assert observed["params"] == {
        "series_id": "RMFSL", "api_key": "a" * 32, "file_type": "json"}
    assert observed["timeout"] == 12.0


def test_fetch_fred_series_point_in_time_records_retrieval_provenance(monkeypatch):
    """Provenance files must carry a retrieval timestamp and content hash,
    not only static IDs/URL -- matching fetch_finra_margin_debt's own
    retrieved_at + raw_sha256 convention. Attach both to .attrs (the same
    metadata channel already used for point_in_time/ledgerable), never
    changing the return type, so both drivers can record real provenance."""
    monkeypatch.setenv("FRED_API_KEY", "a" * 32)
    raw_bytes = json.dumps(_fred_api_payload(
        [{"realtime_start": "x", "realtime_end": "y",
          "date": "2026-06-01", "value": "3047.2"}])).encode("utf-8")
    response = FakeResponse(content=raw_bytes)
    response._json_body = json.loads(raw_bytes)

    out = external_data.fetch_fred_series_point_in_time(
        "RMFSL", http_get=lambda *a, **k: response,
        clock=lambda: datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))

    assert out.attrs["retrieved_at"] == "2026-09-15T12:00:00Z"
    assert out.attrs["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()


def test_fetch_fred_series_point_in_time_uses_explicit_api_key_over_env(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "b" * 32)
    payload = _fred_api_payload(
        [{"realtime_start": "x", "realtime_end": "y",
          "date": "2026-01-01", "value": "1.0"}])
    observed = {}

    def fake_get(url, **kwargs):
        observed.update(kwargs)
        return FakeResponse(json_body=payload)

    external_data.fetch_fred_series_point_in_time(
        "RMFSL", api_key="c" * 32, http_get=fake_get)
    assert observed["params"]["api_key"] == "c" * 32


def test_fetch_fred_series_point_in_time_requires_a_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FRED_API_KEY"):
        external_data.fetch_fred_series_point_in_time(
            "RMFSL", http_get=lambda *a, **k: FakeResponse())


def test_fetch_fred_series_point_in_time_rejects_unverified_series(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "a" * 32)
    with pytest.raises(ValueError, match="not on the verified-unrevised allowlist"):
        external_data.fetch_fred_series_point_in_time(
            "CPIAUCSL", http_get=lambda *a, **k: FakeResponse())


def test_fetch_fred_series_point_in_time_propagates_http_failure(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "a" * 32)
    response = FakeResponse(error=RuntimeError("HTTP 503"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        external_data.fetch_fred_series_point_in_time(
            "RMFSL", http_get=lambda *a, **k: response)


def test_fetch_fred_series_point_in_time_rejects_missing_observations_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "a" * 32)
    response = FakeResponse(json_body={"count": 0})
    with pytest.raises(ValueError, match="missing observations"):
        external_data.fetch_fred_series_point_in_time(
            "RMFSL", http_get=lambda *a, **k: response)


def test_fetch_fred_series_point_in_time_rejects_no_usable_rows(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "a" * 32)
    response = FakeResponse(json_body=_fred_api_payload(
        [{"realtime_start": "x", "realtime_end": "y",
          "date": "2026-01-01", "value": "."}]))
    with pytest.raises(ValueError, match="no usable finite observations"):
        external_data.fetch_fred_series_point_in_time(
            "RMFSL", http_get=lambda *a, **k: response)


def test_fetch_finra_margin_debt_propagates_transport_and_uses_primary_url():
    calls = []
    response = FakeResponse(error=RuntimeError("HTTP 503"), content=b"")

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    with pytest.raises(RuntimeError, match="HTTP 503"):
        external_data.fetch_finra_margin_debt(
            http_get=fake_get,
            excel_reader=lambda *args, **kwargs: None,
            timeout=7.0,
        )
    assert calls == [(
        external_data.FINRA_MARGIN_STATISTICS_URL,
        {"timeout": 7.0},
    )]
