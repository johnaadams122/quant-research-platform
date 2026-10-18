"""Tests for src/mining/edgar_cik_resolution.py. Network-free: resolution
is exercised against small hand-built payloads shaped exactly like SEC's
real ``company_tickers.json`` (``{"0": {"cik_str": ..., "ticker": ...,
"title": ...}, "1": {...}, ...}``, string-indexed keys, unordered)."""
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from mining import edgar_cik_resolution  # noqa: E402


def _payload(entries):
    """entries: list of (cik_str, ticker, title) -> real payload shape."""
    return {str(i): {"cik_str": cik, "ticker": ticker, "title": title}
            for i, (cik, ticker, title) in enumerate(entries)}


# ----------------------------------------------------------- parse

def test_parse_company_tickers_payload_basic():
    payload = _payload([
        (320193, "AAPL", "Apple Inc."),
        (1067983, "BRK-B", "Berkshire Hathaway Inc."),
    ])
    table = edgar_cik_resolution.parse_company_tickers_payload(payload)
    assert list(table.columns) == ["cik", "ticker", "title"]
    assert len(table) == 2
    assert table["cik"].dtype.kind == "i"
    row = table[table["ticker"] == "AAPL"].iloc[0]
    assert row["cik"] == 320193
    assert row["title"] == "Apple Inc."


def test_parse_rejects_non_dict_payload():
    with pytest.raises(ValueError, match="dict"):
        edgar_cik_resolution.parse_company_tickers_payload([1, 2, 3])


def test_parse_rejects_entry_missing_field():
    payload = {"0": {"cik_str": 1, "ticker": "X"}}  # no "title"
    with pytest.raises(ValueError, match="missing field"):
        edgar_cik_resolution.parse_company_tickers_payload(payload)


def test_parse_rejects_empty_payload():
    with pytest.raises(ValueError, match="non-empty"):
        edgar_cik_resolution.parse_company_tickers_payload({})


# ----------------------------------------------------------- resolve

def test_resolve_cik_exact_match():
    table = edgar_cik_resolution.parse_company_tickers_payload(_payload([
        (320193, "AAPL", "Apple Inc."),
    ]))
    result = edgar_cik_resolution.resolve_cik_for_tickers(table, ["AAPL"])
    assert result == {"AAPL": 320193}


def test_resolve_cik_dot_form_in_sec_data_matches_hyphen_query():
    # SEC publishes some share classes with a dot (e.g. "BRK.B"); this
    # repo's own tickers are normalized to hyphen form (BRK-B) per
    # mining.universe.normalize_ticker_for_yfinance -- resolution must
    # bridge the two without the caller doing anything special.
    table = edgar_cik_resolution.parse_company_tickers_payload(_payload([
        (1067983, "BRK.B", "Berkshire Hathaway Inc."),
    ]))
    result = edgar_cik_resolution.resolve_cik_for_tickers(table, ["BRK-B"])
    assert result == {"BRK-B": 1067983}


def test_resolve_cik_hyphen_form_in_sec_data_matches_dot_query():
    table = edgar_cik_resolution.parse_company_tickers_payload(_payload([
        (1067983, "BRK-B", "Berkshire Hathaway Inc."),
    ]))
    result = edgar_cik_resolution.resolve_cik_for_tickers(table, ["BRK.B"])
    assert result == {"BRK.B": 1067983}


def test_resolve_cik_is_case_insensitive():
    table = edgar_cik_resolution.parse_company_tickers_payload(_payload([
        (320193, "aapl", "Apple Inc."),
    ]))
    result = edgar_cik_resolution.resolve_cik_for_tickers(table, ["AAPL"])
    assert result == {"AAPL": 320193}


def test_resolve_cik_unresolved_ticker_returns_none_not_raise():
    table = edgar_cik_resolution.parse_company_tickers_payload(_payload([
        (320193, "AAPL", "Apple Inc."),
    ]))
    result = edgar_cik_resolution.resolve_cik_for_tickers(table, ["AAPL", "NOPE"])
    assert result == {"AAPL": 320193, "NOPE": None}


def test_resolve_cik_duplicate_tickers_in_query_are_idempotent():
    table = edgar_cik_resolution.parse_company_tickers_payload(_payload([
        (320193, "AAPL", "Apple Inc."),
    ]))
    result = edgar_cik_resolution.resolve_cik_for_tickers(table, ["AAPL", "AAPL"])
    assert result == {"AAPL": 320193}
