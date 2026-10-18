import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import universe


def test_normalize_ticker_converts_dot_class_shares_to_hyphen():
    # yfinance/Yahoo Finance uses a hyphen for share-class tickers
    # (BRK.B -> BRK-B), not the dot Wikipedia's raw table uses.
    assert universe.normalize_ticker_for_yfinance("BRK.B") == "BRK-B"
    assert universe.normalize_ticker_for_yfinance("BF.B") == "BF-B"


def test_normalize_ticker_strips_whitespace_and_uppercases():
    assert universe.normalize_ticker_for_yfinance(" aapl \n") == "AAPL"


def test_normalize_ticker_leaves_plain_tickers_unchanged():
    assert universe.normalize_ticker_for_yfinance("MSFT") == "MSFT"


def test_build_universe_dedupes_overlap_between_indices():
    sp500 = ["AAPL", "MSFT", "GOOGL", "BRK.B"]
    nasdaq100 = ["AAPL", "MSFT", "NVDA", "COST"]
    result = universe.build_universe(sp500, nasdaq100)
    assert result["sp500_count"] == 4
    assert result["nasdaq100_count"] == 4
    # AAPL, MSFT overlap -> 2 dropped from the union
    assert result["overlap_count"] == 2
    assert result["unique_count"] == 6
    assert result["tickers"] == ["AAPL", "BRK-B", "COST", "GOOGL", "MSFT", "NVDA"]


def test_build_universe_normalizes_before_deduping():
    # The same underlying security listed with different raw punctuation
    # in the two source tables must collapse to one entry, not two.
    sp500 = ["BRK.B"]
    nasdaq100 = ["BRK-B"]
    result = universe.build_universe(sp500, nasdaq100)
    assert result["unique_count"] == 1
    assert result["overlap_count"] == 1
    assert result["tickers"] == ["BRK-B"]


def test_build_universe_rejects_empty_source_list():
    # An empty constituent list is almost certainly a scrape failure
    # (wrong table picked up, page structure changed) -- fail closed
    # rather than silently proceeding with a partial/wrong universe.
    import pytest
    with pytest.raises(ValueError):
        universe.build_universe([], ["AAPL"])
    with pytest.raises(ValueError):
        universe.build_universe(["AAPL"], [])
