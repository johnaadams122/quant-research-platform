"""Tests for src/mining/edgar_point_in_time.py -- the highest-risk piece
of the capex-cycle family's EDGAR infrastructure (point-in-time capex/revenue joins).
Heavy focus on proving the lookahead guard actually bites, not asserting
it in prose: a deliberately-wrong join keyed on `end` instead of `filed`
must diverge from the correct one on a constructed case.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from mining import edgar_data, edgar_point_in_time as pit  # noqa: E402

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "edgar_synthetic_capex.json")


def _fact(start, end, val, filed, *, fp="FY", form="10-K", accn="0001", fy=2020):
    return {"start": pd.Timestamp(start), "end": pd.Timestamp(end), "val": float(val),
           "filed": pd.Timestamp(filed), "fp": fp, "form": form, "accn": accn, "fy": fy}


def _facts_frame(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------- annual_facts_only

def test_annual_facts_only_filters_fp_and_form():
    rows = [
        _fact("2019-01-01", "2019-12-31", 100, "2020-02-01", fp="FY", form="10-K"),
        _fact("2019-07-01", "2019-09-30", 50, "2019-11-01", fp="Q3", form="10-Q"),
        _fact("2019-01-01", "2019-12-31", 100, "2020-02-05", fp="FY", form="10-K/A"),
    ]
    out = pit.annual_facts_only(_facts_frame(rows))
    assert len(out) == 2
    assert set(out["form"]) == {"10-K", "10-K/A"}


# ------------------------------------------------ build_point_in_time_annual_series

def test_first_disclosure_has_no_growth_yet():
    rows = [_fact("2018-01-01", "2018-12-31", 100, "2019-02-01")]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    assert len(series) == 1
    assert series.iloc[0]["usable_asof"] == pd.Timestamp("2019-02-01")
    assert series.iloc[0]["latest_val"] == 100.0
    assert np.isnan(series.iloc[0]["capex_growth_yoy"])


def test_second_fiscal_year_computes_growth():
    rows = [
        _fact("2018-01-01", "2018-12-31", 100, "2019-02-01"),
        _fact("2019-01-01", "2019-12-31", 130, "2020-02-01"),
    ]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    assert len(series) == 2
    last = series.iloc[-1]
    assert last["latest_val"] == 130.0
    assert last["prior_val"] == 100.0
    assert last["capex_growth_yoy"] == pytest.approx(0.30)


def test_real_first_disclosure_of_a_year_uses_the_earlier_filed_date_not_the_later_repeat():
    """The exact AAPL shape: FY2014's value is disclosed in
    BOTH the FY2015 10-K (filed 2015-10-28, as a comparative year) and
    the FY2016 10-K (filed 2016-10-26, as a comparative year again) --
    same (start,end,val). The series must reflect it as KNOWN starting
    2015-10-28, not silently delayed to 2016-10-26."""
    rows = [
        _fact("2012-09-30", "2013-09-28", 8165, "2015-10-28"),   # FY2013, first year
        _fact("2013-09-29", "2014-09-27", 9571, "2015-10-28"),   # FY2014, same filing
        _fact("2013-09-29", "2014-09-27", 9571, "2016-10-26"),   # FY2014, re-disclosed unchanged
    ]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    # The re-disclosure (unchanged value) must NOT create a new distinct row.
    fy2014_rows = series[series["latest_fiscal_year_end"] == pd.Timestamp("2014-09-27")]
    assert len(fy2014_rows) == 1
    assert fy2014_rows.iloc[0]["usable_asof"] == pd.Timestamp("2015-10-28")


def test_restatement_with_a_different_value_advances_the_series():
    """A GENUINE restatement (same fiscal year, DIFFERENT val, later
    filed) must be reflected at its OWN filed date -- unlike an unchanged
    re-disclosure, this is new information and must appear."""
    rows = [
        _fact("2018-01-01", "2018-12-31", 100, "2019-02-01"),
        _fact("2019-01-01", "2019-12-31", 130, "2020-02-01"),
        _fact("2018-01-01", "2018-12-31", 90, "2020-02-01"),  # restated FY2018, same filing
    ]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    last = series.iloc[-1]
    assert last["prior_val"] == 90.0  # the restated value, not the original 100
    assert last["capex_growth_yoy"] == pytest.approx(130.0 / 90.0 - 1.0)


def test_out_of_encounter_order_three_years_disclosed_at_once_uses_end_date_not_row_order():
    """A first-ever 10-K can disclose 3 comparative years in one filing.
    The 'two most recent' must be picked by fiscal year END DATE, not by
    the order rows happen to appear in the payload -- AND exactly ONE row
    must be emitted for the one filing, not one row per intermediate fold
    state. Emitting a row per fold state on this input would give 3 rows
    sharing usable_asof=2020-02-01 with capex_growth_yoy NaN -> 2.0 -> 0.5,
    and the 2.0 would be an artifact of folding the 'middle' year before
    the 'oldest' one, never an independently observable value (the whole
    filing becomes public atomically)."""
    rows = [
        _fact("2019-01-01", "2019-12-31", 300, "2020-02-01"),  # most recent, listed FIRST
        _fact("2017-01-01", "2017-12-31", 100, "2020-02-01"),  # oldest, listed SECOND
        _fact("2018-01-01", "2018-12-31", 200, "2020-02-01"),  # middle, listed THIRD
    ]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    assert len(series) == 1  # one filing, one atomic disclosure event
    last = series.iloc[-1]
    assert last["latest_fiscal_year_end"] == pd.Timestamp("2019-12-31")
    assert last["latest_val"] == 300.0
    assert last["prior_fiscal_year_end"] == pd.Timestamp("2018-12-31")
    assert last["prior_val"] == 200.0
    assert last["capex_growth_yoy"] == pytest.approx(300.0 / 200.0 - 1.0)


def test_synthetic_fixture_first_10k_discloses_three_years_as_one_atomic_row():
    """The same one-row-per-filing rule on the AAPL-shaped synthetic
    fixture: its first 10-K (filed 2015-10-28) discloses FY2013/FY2014/
    FY2015 capex in one filing. Must produce exactly one usable_asof row
    for that date, not three."""
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    facts = edgar_data.parse_company_concept_payload(
        payload, cik=1234, tag="PaymentsToAcquirePropertyPlantAndEquipment")
    annual = pit.annual_facts_only(facts)
    series = pit.build_point_in_time_annual_series(annual)
    same_date = series[series["usable_asof"] == pd.Timestamp("2015-10-28")]
    assert len(same_date) == 1


def test_empty_input_returns_empty_frame_not_error():
    series = pit.build_point_in_time_annual_series(
        pit.annual_facts_only(_facts_frame([])))
    assert len(series) == 0
    assert list(series.columns) == [
        "usable_asof", "latest_fiscal_year_end", "latest_val",
        "prior_fiscal_year_end", "prior_val", "capex_growth_yoy"]


def test_zero_prior_value_does_not_raise_a_division_error():
    rows = [
        _fact("2018-01-01", "2018-12-31", 0.0, "2019-02-01"),
        _fact("2019-01-01", "2019-12-31", 50.0, "2020-02-01"),
    ]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    assert np.isnan(series.iloc[-1]["capex_growth_yoy"])


# ------------------------------------------------------ the lookahead guard

def test_using_end_date_instead_of_filed_provably_leaks_the_future():
    """The guard this whole module exists for: build the CORRECT series
    (keyed on filed) and a DELIBERATELY WRONG one (keyed on end date, as
    if a careless implementation used the fiscal period end as the
    as-of), then prove they diverge exactly where a real leak would
    occur -- a query date between the two keys sees different growth
    values only under the correct (filed-keyed) construction."""
    # FY2019 ends 2019-12-31 (early), but ISN'T FILED until 2021-06-01 --
    # a company badly behind on its filings. FY2018 was filed earlier.
    rows = [
        _fact("2018-01-01", "2018-12-31", 100, "2019-02-01"),
        _fact("2019-01-01", "2019-12-31", 300, "2021-06-01"),  # late filer
    ]
    correct = pit.build_point_in_time_annual_series(_facts_frame(rows))
    # A query date AFTER FY2019's end (2019-12-31) but BEFORE it was
    # actually filed (2021-06-01) must NOT see FY2019's value yet.
    # staleness_cap_days is set generously wide here to isolate the
    # LOOKAHEAD property in this test from the SEPARATE staleness-cap
    # property (already covered on its own by
    # test_asof_join_respects_staleness_cap) -- this "late filer" gap is
    # deliberately larger than a normal annual refresh cadence.
    query_date = pd.DatetimeIndex([pd.Timestamp("2020-06-01")])
    correct_asof = pit.point_in_time_asof(correct, query_date, staleness_cap_days=100000)
    assert correct_asof["latest_val"].iloc[0] == 100.0  # still FY2018 -- FY2019 not yet public
    assert np.isnan(correct_asof["capex_growth_yoy"].iloc[0])

    # The WRONG construction (what a careless "as of end date" join would
    # produce): treat facts as known at their fiscal period END instead
    # of their FILED date.
    wrong_rows = [dict(r, filed=r["end"]) for r in rows]
    wrong = pit.build_point_in_time_annual_series(_facts_frame(wrong_rows))
    wrong_asof = pit.point_in_time_asof(wrong, query_date, staleness_cap_days=100000)
    assert wrong_asof["latest_val"].iloc[0] == 300.0  # LEAKS the not-yet-public FY2019 value

    assert correct_asof["latest_val"].iloc[0] != wrong_asof["latest_val"].iloc[0]


def test_filed_on_a_weekend_is_handled_by_backward_merge_with_no_special_casing():
    rows = [_fact("2018-01-01", "2018-12-31", 100, "2019-02-02")]  # a Saturday
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2019-02-01"), pd.Timestamp("2019-02-02"), pd.Timestamp("2019-02-04")])
    out = pit.point_in_time_asof(series, dates)
    assert np.isnan(out["latest_val"].iloc[0])       # before the weekend filing
    assert out["latest_val"].iloc[1] == 100.0         # the weekend date itself
    assert out["latest_val"].iloc[2] == 100.0         # the next trading day


# ---------------------------------------------------------- point_in_time_asof

def test_asof_join_respects_staleness_cap():
    rows = [_fact("2018-01-01", "2018-12-31", 100, "2019-02-01")]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    fresh_date = pd.Timestamp("2019-02-01") + pd.Timedelta(days=100)
    stale_date = pd.Timestamp("2019-02-01") + pd.Timedelta(days=500)
    dates = pd.DatetimeIndex([fresh_date, stale_date])
    out = pit.point_in_time_asof(series, dates, staleness_cap_days=400)
    assert out["latest_val"].iloc[0] == 100.0
    assert np.isnan(out["latest_val"].iloc[1])


def test_asof_join_before_any_disclosure_is_nan():
    rows = [_fact("2018-01-01", "2018-12-31", 100, "2019-02-01")]
    series = pit.build_point_in_time_annual_series(_facts_frame(rows))
    dates = pd.DatetimeIndex([pd.Timestamp("2018-06-01")])
    out = pit.point_in_time_asof(series, dates)
    assert np.isnan(out["latest_val"].iloc[0])


def test_asof_join_on_empty_series_returns_all_nan():
    empty = pit.build_point_in_time_annual_series(
        pit.annual_facts_only(_facts_frame([])))
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")])
    out = pit.point_in_time_asof(empty, dates)
    assert len(out) == 2
    assert out["latest_val"].isna().all()


# --------------------------------------------- AAPL-shaped synthetic fixture

def test_synthetic_fixture_end_to_end_point_in_time_join():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    facts = edgar_data.parse_company_concept_payload(
        payload, cik=1234, tag="PaymentsToAcquirePropertyPlantAndEquipment")
    annual = pit.annual_facts_only(facts)
    series = pit.build_point_in_time_annual_series(annual)
    assert len(series) >= 5
    assert series["usable_asof"].is_monotonic_increasing

    dates = pd.DatetimeIndex([pd.Timestamp("2016-01-01"), pd.Timestamp("2026-09-09")])
    out = pit.point_in_time_asof(series, dates, staleness_cap_days=400)
    # 2016-01-01: some capex figure must be known by then (10-K
    # filed 2015-10-28) -- not NaN.
    assert not np.isnan(out["latest_val"].iloc[0])
