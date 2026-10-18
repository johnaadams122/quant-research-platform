"""Ticker -> CIK resolution against SEC's public ``company_tickers.json``
(https://www.sec.gov/files/company_tickers.json), for screening the S&P
500 universe through ``edgar_data``'s CIK-keyed XBRL client (capex-cycle
watchlist build).

Fetching that file is a plain ``www.sec.gov`` GET, not the rate-limited
``data.sec.gov/api/xbrl`` endpoint ``edgar_data.py`` throttles -- this
module owns only the resolution logic (fail-closed parse + pure lookup);
the live fetch is the caller's job (``scripts/build_capex_watchlist.py``),
matching ``edgar_data.py``'s own raw-fetch/pure-parse split.

Normalization: SEC's file spells some share-class tickers with a dot
(e.g. "BRK.B") while this repo's own universe snapshots normalize to a
hyphen (``mining.universe.normalize_ticker_for_yfinance``, "BRK-B").
Resolution tries both forms so neither side has to agree on a convention
in advance.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

REQUIRED_ENTRY_FIELDS = ("cik_str", "ticker", "title")


def parse_company_tickers_payload(payload: dict) -> pd.DataFrame:
    """Fail-closed parse of SEC's ``company_tickers.json`` shape --
    string-indexed keys ("0", "1", ...), each value an object with
    ``cik_str``/``ticker``/``title``. Returns a DataFrame with columns
    ``cik`` (int), ``ticker`` (str, as published by SEC), ``title``
    (str). Raises on any schema drift rather than silently skipping a
    malformed entry -- matching ``edgar_data.parse_company_concept_payload``'s
    own convention."""
    if not isinstance(payload, dict):
        raise ValueError("company_tickers payload must be a dict")
    if not payload:
        raise ValueError("company_tickers payload must be non-empty")

    rows = []
    for key, entry in payload.items():
        if not isinstance(entry, dict):
            raise ValueError(f"company_tickers entry {key!r} is not an object")
        missing = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"company_tickers entry {key!r} missing field(s) {missing}")
        rows.append({
            "cik": int(entry["cik_str"]),
            "ticker": str(entry["ticker"]),
            "title": str(entry["title"]),
        })
    return pd.DataFrame(rows, columns=["cik", "ticker", "title"])


def _variants(ticker: str) -> set:
    upper = ticker.strip().upper()
    return {upper, upper.replace(".", "-"), upper.replace("-", ".")}


def resolve_cik_for_tickers(table: pd.DataFrame, tickers) -> dict:
    """Look up each of ``tickers`` (this repo's own normalized form,
    hyphenated share classes) against ``table`` (SEC's published form,
    dotted share classes), trying dot/hyphen/case variants on both
    sides. A ticker with no match maps to ``None`` -- never raises --
    so the caller can drop it as an excluded pool member (this family's
    own pooled-candidate convention: drop the member, not the family)
    rather than aborting the whole screen over one unresolved name."""
    by_variant = {}
    for _, row in table.iterrows():
        for variant in _variants(row["ticker"]):
            by_variant.setdefault(variant, int(row["cik"]))

    result = {}
    for ticker in tickers:
        if ticker in result:
            continue
        cik = None
        for variant in _variants(ticker):
            if variant in by_variant:
                cik = by_variant[variant]
                break
        result[ticker] = cik
    return result
