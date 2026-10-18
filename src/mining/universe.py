"""Real-market ticker universe construction for pooled multi-ticker
mining tests (every ticker in the S&P 500 and NASDAQ 100, for catalog
candidates whose signal is derived purely from one ticker's own
price/earnings history -- e.g. fifty_two_week_high_momentum and
pead_earnings_surprise_drift in data/mining/catalog.json).

This module holds only the PURE, testable normalization/dedup logic.
The actual live fetch of the two Wikipedia source pages (network I/O,
not unit-testable in the way pure functions are) lives in
scripts/fetch_index_constituents.py, which calls build_universe() with
whatever raw ticker lists it scraped.
"""
from __future__ import annotations


def normalize_ticker_for_yfinance(raw: str) -> str:
    """Wikipedia's constituent tables spell share-class tickers with a
    dot (e.g. "BRK.B", "BF.B"); yfinance/Yahoo Finance expects a hyphen
    ("BRK-B", "BF-B"). Also strips surrounding whitespace and uppercases
    -- defensive against incidental formatting noise in a scraped table."""
    return raw.strip().upper().replace(".", "-")


def build_universe(sp500_tickers: list, nasdaq100_tickers: list) -> dict:
    """Normalize and de-duplicate two raw ticker lists (as scraped,
    unnormalized) into one combined universe. Raises ValueError if
    either input list is empty -- an empty scrape result is almost
    certainly a page-structure/parsing failure, not a real 0-constituent
    index, and silently proceeding would test a wrong/partial universe
    without saying so.

    Returns {'tickers': sorted unique normalized list, 'sp500_count':
    len(sp500_tickers) (raw, pre-dedup), 'nasdaq100_count':
    len(nasdaq100_tickers) (raw, pre-dedup), 'overlap_count': how many
    entries were shared between the two normalized sets (i.e. how much
    the union shrank vs. the raw sum), 'unique_count': len(tickers)}."""
    if not sp500_tickers:
        raise ValueError("build_universe: sp500_tickers is empty -- likely "
                          "a scrape/parse failure, refusing to proceed silently")
    if not nasdaq100_tickers:
        raise ValueError("build_universe: nasdaq100_tickers is empty -- likely "
                          "a scrape/parse failure, refusing to proceed silently")

    sp500_norm = {normalize_ticker_for_yfinance(t) for t in sp500_tickers}
    nasdaq100_norm = {normalize_ticker_for_yfinance(t) for t in nasdaq100_tickers}
    union = sp500_norm | nasdaq100_norm
    overlap_count = len(sp500_norm) + len(nasdaq100_norm) - len(union)

    return {
        "tickers": sorted(union),
        "sp500_count": len(sp500_tickers),
        "nasdaq100_count": len(nasdaq100_tickers),
        "overlap_count": overlap_count,
        "unique_count": len(union),
    }
