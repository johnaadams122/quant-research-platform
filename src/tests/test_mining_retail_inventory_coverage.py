"""Tests for src/mining/retail_inventory_coverage.py -- the data-coverage
check for the retail_inventory_sales_breadth_short candidate
(disclosed real risk: inventory/sales tag naming is inconsistent across
retail filers). Network-free: every fetch is injected.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from mining import edgar_data, retail_inventory_coverage as ric  # noqa: E402


def _fact(**overrides):
    fact = {"start": "2019-01-01", "end": "2019-12-31", "val": 100.0,
            "accn": "0001", "fy": 2019, "fp": "FY", "form": "10-K",
            "filed": "2020-02-01"}
    fact.update(overrides)
    return pd.DataFrame([fact])


def _two_year_frame(tag_val_year1=100.0, tag_val_year2=150.0):
    """Two distinct fiscal years, both already annual_facts_only-shaped
    (fp=FY, form=10-K) so a real growth ratio can be computed."""
    return pd.concat([
        _fact(start="2018-01-01", end="2018-12-31", val=tag_val_year1,
              fy=2018, filed="2019-02-01"),
        _fact(start="2019-01-01", end="2019-12-31", val=tag_val_year2,
              fy=2019, filed="2020-02-01"),
    ], ignore_index=True)


def _one_year_frame():
    return _fact()


def _instant_fact(**overrides):
    # Real InventoryNet (an instant concept) shape: no "start" field at all. Dates pre-parsed to Timestamp, matching
    # parse_instant_concept_payload's own output dtype (this fixture
    # feeds build_point_in_time_instant_series directly, bypassing the
    # parser, so it must already look like the parser's output).
    fact = {"end": pd.Timestamp("2019-12-31"), "val": 100.0, "accn": "0001",
           "fy": 2019, "fp": "FY", "form": "10-K", "filed": pd.Timestamp("2020-02-01")}
    fact.update(overrides)
    if not isinstance(fact["end"], pd.Timestamp):
        fact["end"] = pd.Timestamp(fact["end"])
    if not isinstance(fact["filed"], pd.Timestamp):
        fact["filed"] = pd.Timestamp(fact["filed"])
    return pd.DataFrame([fact])


def _two_year_instant_frame(val_year1=100.0, val_year2=150.0):
    return pd.concat([
        _instant_fact(end="2018-12-31", val=val_year1, fy=2018, filed="2019-02-01"),
        _instant_fact(end="2019-12-31", val=val_year2, fy=2019, filed="2020-02-01"),
    ], ignore_index=True)


# ---------------------------------------- build_point_in_time_instant_series

def test_build_instant_series_empty_input_is_empty_not_an_error():
    empty = pd.DataFrame(columns=list(edgar_data.REQUIRED_INSTANT_FACT_FIELDS))
    result = ric.build_point_in_time_instant_series(empty)
    assert result.empty
    assert list(result.columns) == [
        "usable_asof", "latest_fiscal_year_end", "latest_val",
        "prior_fiscal_year_end", "prior_val", "growth_yoy"]


def test_build_instant_series_no_growth_until_second_fiscal_year():
    single = _instant_fact()
    result = ric.build_point_in_time_instant_series(single)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["growth_yoy"])


def test_build_instant_series_computes_growth_from_two_fiscal_years():
    frame = _two_year_instant_frame(100.0, 150.0)
    result = ric.build_point_in_time_instant_series(frame)
    assert len(result) == 2
    last = result.iloc[-1]
    assert last["growth_yoy"] == pytest.approx(0.5)
    assert last["usable_asof"] == pd.Timestamp("2020-02-01")


def test_build_instant_series_keys_on_end_alone_not_start_end_pair():
    # The whole reason this is a separate function from
    # edgar_point_in_time.build_point_in_time_annual_series: an instant
    # fact has no "start" to include in the dedup key. Two facts sharing
    # the same "end" (the balance-sheet date) are the SAME fiscal year
    # even though nothing distinguishes them via "start".
    frame = pd.concat([
        _instant_fact(end="2019-12-31", val=100.0, filed="2020-02-01"),
        _instant_fact(end="2019-12-31", val=100.0, filed="2020-06-01"),  # re-disclosed, unchanged
    ], ignore_index=True)
    result = ric.build_point_in_time_instant_series(frame)
    assert len(result) == 1  # unchanged re-disclosure drops, not a new row


def test_build_instant_series_restatement_updates_the_value():
    frame = pd.concat([
        _instant_fact(end="2018-12-31", val=100.0, filed="2019-02-01"),
        _instant_fact(end="2019-12-31", val=150.0, filed="2020-02-01"),
        _instant_fact(end="2018-12-31", val=120.0, filed="2020-02-01"),  # restated FY2018
    ], ignore_index=True)
    result = ric.build_point_in_time_instant_series(frame)
    last = result.iloc[-1]
    # After the restatement, prior_val (FY2018) should reflect 120, not 100.
    assert last["prior_val"] == pytest.approx(120.0)
    assert last["growth_yoy"] == pytest.approx(150.0 / 120.0 - 1.0)


# ------------------------------------------------- select_and_build_annual_series
#
# kind="duration" (default) covers capex/revenue-shaped tags (start+end);
# kind="instant" covers balance-sheet tags like InventoryNet (end only) --
# see build_point_in_time_instant_series's own docstring for why a real
# InventoryNet fact has no "start" at all.

def test_select_and_build_returns_none_for_empty_by_tag():
    series, tag = ric.select_and_build_annual_series(
        {}, edgar_data.INVENTORY_TAGS, kind="instant")
    assert series is None
    assert tag is None


def test_select_and_build_returns_none_when_only_one_fiscal_year_known():
    by_tag = {"InventoryNet": _instant_fact()}
    series, tag = ric.select_and_build_annual_series(
        by_tag, edgar_data.INVENTORY_TAGS, kind="instant")
    assert series is None
    assert tag is None


def test_select_and_build_returns_a_usable_instant_series_with_two_fiscal_years():
    by_tag = {"InventoryNet": _two_year_instant_frame()}
    series, tag = ric.select_and_build_annual_series(
        by_tag, edgar_data.INVENTORY_TAGS, kind="instant")
    assert tag == "InventoryNet"
    assert series is not None
    assert not series["growth_yoy"].isna().all()
    last = series.iloc[-1]
    assert last["growth_yoy"] == pytest.approx(0.5)  # 150/100 - 1


def test_select_and_build_prefers_inventory_net_over_gross():
    by_tag = {
        "InventoryGross": _two_year_instant_frame(1000.0, 1500.0),
        "InventoryNet": _two_year_instant_frame(100.0, 150.0),
    }
    series, tag = ric.select_and_build_annual_series(
        by_tag, edgar_data.INVENTORY_TAGS, kind="instant")
    assert tag == "InventoryNet"
    assert series.iloc[-1]["growth_yoy"] == pytest.approx(0.5)


def test_select_and_build_filters_out_non_annual_facts_first_instant():
    # A 10-Q fact must not count as a second "fiscal year" -- only 10-K/FY
    # facts make a series usable.
    by_tag = {"InventoryNet": pd.concat([
        _instant_fact(fp="FY", form="10-K"),
        _instant_fact(fp="Q2", form="10-Q", val=999.0, filed="2019-08-01"),
    ], ignore_index=True)}
    series, tag = ric.select_and_build_annual_series(
        by_tag, edgar_data.INVENTORY_TAGS, kind="instant")
    assert series is None
    assert tag is None


def test_select_and_build_duration_kind_for_revenue():
    by_tag = {"Revenues": _two_year_frame(1000.0, 1200.0)}
    series, tag = ric.select_and_build_annual_series(
        by_tag, edgar_data.REVENUE_TAGS, kind="duration")
    assert tag == "Revenues"
    assert series.iloc[-1]["capex_growth_yoy"] == pytest.approx(0.2)


def test_select_and_build_default_kind_is_duration():
    by_tag = {"Revenues": _two_year_frame(1000.0, 1200.0)}
    series, tag = ric.select_and_build_annual_series(by_tag, edgar_data.REVENUE_TAGS)
    assert tag == "Revenues"


# ------------------------------------------------------- check_ticker_coverage

def test_check_ticker_coverage_no_cik():
    result = ric.check_ticker_coverage("ZZZZ", None, fetch_tags_fn=lambda cik, tags: {})
    assert result["status"] == "no_cik"
    assert result["ticker"] == "ZZZZ"


def test_check_ticker_coverage_no_usable_inventory():
    def fetch(cik, tags):
        return {}  # neither inventory nor (later) revenue tags ever return data
    result = ric.check_ticker_coverage("ETSY", 1370637, fetch_tags_fn=fetch)
    assert result["status"] == "no_usable_inventory"
    assert result["cik"] == 1370637


def test_check_ticker_coverage_no_usable_revenue():
    def fetch(cik, tags):
        if tags == edgar_data.INVENTORY_TAGS:
            return {"InventoryNet": _two_year_instant_frame()}
        return {}
    result = ric.check_ticker_coverage("XYZ", 42, fetch_tags_fn=fetch)
    assert result["status"] == "no_usable_revenue"
    assert result["inventory_tag_used"] == "InventoryNet"
    assert result["revenue_tag_used"] is None


def test_check_ticker_coverage_ok_when_both_present():
    def fetch(cik, tags):
        if tags == edgar_data.INVENTORY_TAGS:
            return {"InventoryNet": _two_year_instant_frame(100.0, 130.0)}
        if tags == edgar_data.REVENUE_TAGS:
            return {"Revenues": _two_year_frame(1000.0, 1040.0)}
        raise AssertionError(f"unexpected tags {tags}")
    result = ric.check_ticker_coverage("WMT", 104169, fetch_tags_fn=fetch)
    assert result["status"] == "ok"
    assert result["inventory_tag_used"] == "InventoryNet"
    assert result["revenue_tag_used"] == "Revenues"
    assert result["inventory_series"] is not None
    assert result["revenue_series"] is not None


def test_check_ticker_coverage_records_fetch_error_without_raising():
    def fetch(cik, tags):
        raise RuntimeError("boom")
    result = ric.check_ticker_coverage("BAD", 1, fetch_tags_fn=fetch)
    assert result["status"] == "fetch_error"
    assert "boom" in result["detail"]


# ------------------------------------------------------------- run_coverage_check

def test_run_coverage_check_aggregates_and_computes_fraction():
    def fetch_tags_fn(cik, tags):
        # cik 1 -> usable both; cik 2 -> inventory only; cik None never
        # reaches here (no_cik short-circuits before any fetch).
        if cik == 1:
            if tags == edgar_data.INVENTORY_TAGS:
                return {"InventoryNet": _two_year_instant_frame()}
            return {"Revenues": _two_year_frame()}
        if cik == 2:
            if tags == edgar_data.INVENTORY_TAGS:
                return {"InventoryNet": _two_year_instant_frame()}
            return {}
        return {}

    cik_by_ticker = {"AAA": 1, "BBB": 2, "CCC": None}
    result = ric.run_coverage_check(
        ["AAA", "BBB", "CCC"], cik_by_ticker, fetch_tags_fn=fetch_tags_fn)

    assert result["declared_n"] == 3
    assert result["usable_n"] == 1
    assert result["coverage_fraction"] == pytest.approx(1 / 3)
    assert result["by_status"] == {"ok": 1, "no_usable_revenue": 1, "no_cik": 1}
    assert len(result["results"]) == 3
    statuses = {r["ticker"]: r["status"] for r in result["results"]}
    assert statuses == {"AAA": "ok", "BBB": "no_usable_revenue", "CCC": "no_cik"}


def test_run_coverage_check_empty_universe_is_zero_not_a_crash():
    result = ric.run_coverage_check([], {}, fetch_tags_fn=lambda cik, tags: {})
    assert result["declared_n"] == 0
    assert result["usable_n"] == 0
    assert result["coverage_fraction"] == 0.0
    assert result["results"] == []


def test_run_coverage_check_never_calls_fetch_for_an_unresolved_cik():
    calls = []

    def fetch_tags_fn(cik, tags):
        calls.append((cik, tags))
        return {}

    ric.run_coverage_check(["ZZZZ"], {"ZZZZ": None}, fetch_tags_fn=fetch_tags_fn)
    assert calls == []
