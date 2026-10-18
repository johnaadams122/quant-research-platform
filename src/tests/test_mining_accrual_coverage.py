"""Tests for src/mining/accrual_coverage.py -- the vendor-data coverage check
for the accrual_anomaly_sloan candidate (three legs: net income, CFO,
total assets). Network-free: every fetch is injected. Mirrors
test_mining_retail_inventory_coverage.py's own fixture shapes.
"""
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from mining import accrual_coverage as ac, edgar_data  # noqa: E402


def _fact(**overrides):
    fact = {"start": "2019-01-01", "end": "2019-12-31", "val": 100.0,
            "accn": "0001", "fy": 2019, "fp": "FY", "form": "10-K",
            "filed": "2020-02-01"}
    fact.update(overrides)
    return pd.DataFrame([fact])


def _two_year_frame(val_year1=100.0, val_year2=150.0):
    return pd.concat([
        _fact(start="2018-01-01", end="2018-12-31", val=val_year1,
              fy=2018, filed="2019-02-01"),
        _fact(start="2019-01-01", end="2019-12-31", val=val_year2,
              fy=2019, filed="2020-02-01"),
    ], ignore_index=True)


def _one_year_frame():
    return _fact()


def _instant_fact(**overrides):
    fact = {"end": pd.Timestamp("2019-12-31"), "val": 100.0, "accn": "0001",
           "fy": 2019, "fp": "FY", "form": "10-K", "filed": pd.Timestamp("2020-02-01")}
    fact.update(overrides)
    if not isinstance(fact["end"], pd.Timestamp):
        fact["end"] = pd.Timestamp(fact["end"])
    if not isinstance(fact["filed"], pd.Timestamp):
        fact["filed"] = pd.Timestamp(fact["filed"])
    return pd.DataFrame([fact])


def _two_year_instant_frame(val_year1=1000.0, val_year2=1200.0):
    return pd.concat([
        _instant_fact(end="2018-12-31", val=val_year1, fy=2018, filed="2019-02-01"),
        _instant_fact(end="2019-12-31", val=val_year2, fy=2019, filed="2020-02-01"),
    ], ignore_index=True)


# --------------------------------------------- _select_and_build_duration_series

def test_select_duration_returns_none_for_empty_by_tag():
    series, tag = ac._select_and_build_duration_series({}, edgar_data.NET_INCOME_TAGS)
    assert series is None and tag is None


def test_select_duration_a_single_fiscal_year_is_already_usable():
    # Unlike the sibling capex/inventory/revenue coverage checks (which
    # need a YoY GROWTH ratio and so require >= 2 distinct fiscal years),
    # this candidate only ever reads each leg's own LATEST known LEVEL
    # (net income, CFO) -- a single disclosed fiscal year is coverage-ok.
    by_tag = {"NetIncomeLoss": _one_year_frame()}
    series, tag = ac._select_and_build_duration_series(by_tag, edgar_data.NET_INCOME_TAGS)
    assert tag == "NetIncomeLoss"
    assert series is not None
    assert series.iloc[-1]["latest_val"] == pytest.approx(100.0)


def test_select_duration_prefers_net_income_loss_over_profit_loss():
    by_tag = {
        "ProfitLoss": _two_year_frame(10.0, 20.0),
        "NetIncomeLoss": _two_year_frame(100.0, 150.0),
    }
    series, tag = ac._select_and_build_duration_series(by_tag, edgar_data.NET_INCOME_TAGS)
    assert tag == "NetIncomeLoss"
    assert series.iloc[-1]["latest_val"] == pytest.approx(150.0)


def test_select_duration_falls_back_to_profit_loss():
    by_tag = {"ProfitLoss": _two_year_frame(10.0, 25.0)}
    series, tag = ac._select_and_build_duration_series(by_tag, edgar_data.NET_INCOME_TAGS)
    assert tag == "ProfitLoss"
    assert series.iloc[-1]["latest_val"] == pytest.approx(25.0)


def test_select_duration_falls_back_when_the_higher_priority_tag_is_non_annual_only():
    # Guarded case: a company whose NetIncomeLoss tag
    # returns real facts but every one is a non-annual (10-Q) disclosure --
    # annual_facts_only correctly reduces it to zero rows -- while
    # ProfitLoss (the fallback) has real annual data for the SAME company.
    # edgar_data.select_priority_series would commit to NetIncomeLoss
    # (merely non-empty) and never try ProfitLoss; this function must not.
    non_annual_only = _fact(fp="Q2", form="10-Q")  # never annual_facts_only-usable
    by_tag = {
        "NetIncomeLoss": non_annual_only,
        "ProfitLoss": _two_year_frame(10.0, 25.0),
    }
    series, tag = ac._select_and_build_duration_series(by_tag, edgar_data.NET_INCOME_TAGS)
    assert tag == "ProfitLoss"
    assert series is not None
    assert series.iloc[-1]["latest_val"] == pytest.approx(25.0)


# --------------------------------------------- _select_and_build_instant_series

def test_select_instant_returns_none_for_empty_by_tag():
    series, tag = ac._select_and_build_instant_series({}, edgar_data.TOTAL_ASSETS_TAGS)
    assert series is None and tag is None


def test_select_instant_returns_a_usable_series_with_two_fiscal_years():
    by_tag = {"Assets": _two_year_instant_frame(1000.0, 1200.0)}
    series, tag = ac._select_and_build_instant_series(by_tag, edgar_data.TOTAL_ASSETS_TAGS)
    assert tag == "Assets"
    last = series.iloc[-1]
    assert last["latest_val"] == pytest.approx(1200.0)
    assert last["prior_val"] == pytest.approx(1000.0)


def test_select_instant_falls_back_when_the_higher_priority_tag_is_non_annual_only():
    non_annual_only = _instant_fact(fp="Q2", form="10-Q")
    by_tag = {
        "InventoryNet": non_annual_only,
        "InventoryGross": _two_year_instant_frame(1000.0, 1200.0),
    }
    series, tag = ac._select_and_build_instant_series(
        by_tag, ("InventoryNet", "InventoryGross"))
    assert tag == "InventoryGross"
    assert series is not None


# --------------------------------------------------------- check_ticker_coverage

def _fetch_ok(cik, tags):
    if tags == edgar_data.NET_INCOME_TAGS:
        return {"NetIncomeLoss": _two_year_frame(100.0, 150.0)}
    raise AssertionError(f"unexpected duration tags {tags}")


def _fetch_instant_ok(cik, tags):
    if tags == edgar_data.TOTAL_ASSETS_TAGS:
        return {"Assets": _two_year_instant_frame(1000.0, 1200.0)}
    raise AssertionError(f"unexpected instant tags {tags}")


def test_check_ticker_coverage_no_cik():
    result = ac.check_ticker_coverage(
        "ZZZZ", None, fetch_tags_fn=lambda c, t: {}, fetch_instant_tags_fn=lambda c, t: {})
    assert result["status"] == "no_cik"
    assert result["ticker"] == "ZZZZ"


def test_check_ticker_coverage_no_usable_net_income():
    result = ac.check_ticker_coverage(
        "XYZ", 1, fetch_tags_fn=lambda c, t: {}, fetch_instant_tags_fn=lambda c, t: {})
    assert result["status"] == "no_usable_net_income"


def test_check_ticker_coverage_no_usable_cfo():
    def fetch_tags_fn(cik, tags):
        if tags == edgar_data.NET_INCOME_TAGS:
            return {"NetIncomeLoss": _two_year_frame()}
        return {}
    result = ac.check_ticker_coverage(
        "XYZ", 1, fetch_tags_fn=fetch_tags_fn, fetch_instant_tags_fn=lambda c, t: {})
    assert result["status"] == "no_usable_cfo"
    assert result["ni_tag_used"] == "NetIncomeLoss"
    assert result["cfo_tag_used"] is None


def test_check_ticker_coverage_no_usable_assets():
    def fetch_tags_fn(cik, tags):
        if tags == edgar_data.NET_INCOME_TAGS:
            return {"NetIncomeLoss": _two_year_frame()}
        if tags == edgar_data.CFO_TAGS:
            return {"NetCashProvidedByUsedInOperatingActivities": _two_year_frame(80.0, 90.0)}
        raise AssertionError(f"unexpected tags {tags}")
    result = ac.check_ticker_coverage(
        "XYZ", 1, fetch_tags_fn=fetch_tags_fn, fetch_instant_tags_fn=lambda c, t: {})
    assert result["status"] == "no_usable_assets"
    assert result["ni_tag_used"] == "NetIncomeLoss"
    assert result["cfo_tag_used"] == "NetCashProvidedByUsedInOperatingActivities"
    assert result["assets_tag_used"] is None


def test_check_ticker_coverage_ok_when_all_three_present():
    def fetch_tags_fn(cik, tags):
        if tags == edgar_data.NET_INCOME_TAGS:
            return {"NetIncomeLoss": _two_year_frame(100.0, 150.0)}
        if tags == edgar_data.CFO_TAGS:
            return {"NetCashProvidedByUsedInOperatingActivities": _two_year_frame(80.0, 90.0)}
        raise AssertionError(f"unexpected tags {tags}")
    result = ac.check_ticker_coverage(
        "MSFT", 789019, fetch_tags_fn=fetch_tags_fn, fetch_instant_tags_fn=_fetch_instant_ok)
    assert result["status"] == "ok"
    assert result["ni_tag_used"] == "NetIncomeLoss"
    assert result["cfo_tag_used"] == "NetCashProvidedByUsedInOperatingActivities"
    assert result["assets_tag_used"] == "Assets"
    assert result["ni_series"] is not None
    assert result["cfo_series"] is not None
    assert result["assets_series"] is not None


def test_check_ticker_coverage_records_fetch_error_without_raising():
    def fetch_tags_fn(cik, tags):
        raise RuntimeError("boom")
    result = ac.check_ticker_coverage(
        "BAD", 1, fetch_tags_fn=fetch_tags_fn, fetch_instant_tags_fn=lambda c, t: {})
    assert result["status"] == "fetch_error"
    assert "boom" in result["detail"]


def test_check_ticker_coverage_instant_fetch_error_recorded_with_duration_tags_kept():
    def fetch_instant_error(cik, tags):
        raise RuntimeError("assets boom")
    result = ac.check_ticker_coverage(
        "XYZ", 1, fetch_tags_fn=_fetch_ok_with_cfo, fetch_instant_tags_fn=fetch_instant_error)
    assert result["status"] == "fetch_error"
    assert "assets boom" in result["detail"]
    assert result["ni_tag_used"] == "NetIncomeLoss"
    assert result["cfo_tag_used"] == "NetCashProvidedByUsedInOperatingActivities"


def _fetch_ok_with_cfo(cik, tags):
    if tags == edgar_data.NET_INCOME_TAGS:
        return {"NetIncomeLoss": _two_year_frame(100.0, 150.0)}
    if tags == edgar_data.CFO_TAGS:
        return {"NetCashProvidedByUsedInOperatingActivities": _two_year_frame(80.0, 90.0)}
    raise AssertionError(f"unexpected tags {tags}")


# -------------------------------------------------------------- run_coverage_check

def test_run_coverage_check_aggregates_and_computes_fraction():
    def fetch_tags_fn(cik, tags):
        if cik == 1:
            if tags == edgar_data.NET_INCOME_TAGS:
                return {"NetIncomeLoss": _two_year_frame()}
            return {"NetCashProvidedByUsedInOperatingActivities": _two_year_frame()}
        if cik == 2:
            if tags == edgar_data.NET_INCOME_TAGS:
                return {"NetIncomeLoss": _two_year_frame()}
            return {}
        return {}

    def fetch_instant_tags_fn(cik, tags):
        if cik == 1:
            return {"Assets": _two_year_instant_frame()}
        return {}

    cik_by_ticker = {"AAA": 1, "BBB": 2, "CCC": None}
    result = ac.run_coverage_check(
        ["AAA", "BBB", "CCC"], cik_by_ticker,
        fetch_tags_fn=fetch_tags_fn, fetch_instant_tags_fn=fetch_instant_tags_fn)

    assert result["declared_n"] == 3
    assert result["usable_n"] == 1
    assert result["coverage_fraction"] == pytest.approx(1 / 3)
    assert result["by_status"] == {"ok": 1, "no_usable_cfo": 1, "no_cik": 1}
    statuses = {r["ticker"]: r["status"] for r in result["results"]}
    assert statuses == {"AAA": "ok", "BBB": "no_usable_cfo", "CCC": "no_cik"}


def test_run_coverage_check_empty_universe_is_zero_not_a_crash():
    result = ac.run_coverage_check(
        [], {}, fetch_tags_fn=lambda c, t: {}, fetch_instant_tags_fn=lambda c, t: {})
    assert result["declared_n"] == 0
    assert result["usable_n"] == 0
    assert result["coverage_fraction"] == 0.0
    assert result["results"] == []


def test_run_coverage_check_never_calls_fetch_for_an_unresolved_cik():
    calls = []

    def fetch_tags_fn(cik, tags):
        calls.append((cik, tags))
        return {}

    ac.run_coverage_check(
        ["ZZZZ"], {"ZZZZ": None},
        fetch_tags_fn=fetch_tags_fn, fetch_instant_tags_fn=fetch_tags_fn)
    assert calls == []
