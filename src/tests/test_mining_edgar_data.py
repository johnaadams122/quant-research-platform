"""Tests for src/mining/edgar_data.py. Network-free: fetch functions are
exercised via an injected fake http_get, and one pinned synthetic payload
with the exact shape of a real SEC EDGAR companyconcept response
(PaymentsToAcquirePropertyPlantAndEquipment, 105 facts; values scaled down,
company and filing ids replaced) proves the parser against a realistic
document, not just a hand-crafted mock.
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

from mining import edgar_data  # noqa: E402

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "edgar_synthetic_capex.json")
AAPL_CIK = 1234
AAPL_TAG = "PaymentsToAcquirePropertyPlantAndEquipment"
# Synthetic contact strings, built from parts so no address literal is stored.
_UA = "Example Research " + "x" + "@" + "example.org"
_CONTACT = "x " + "y" + "@" + "example.org"


def _load_fixture() -> dict:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json_body = json_body

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests_module.HTTPError(f"status {self.status_code}")


import requests as requests_module  # noqa: E402


def _fake_http_get(response_map):
    calls = []

    def http_get(url, *, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        for key, response in response_map.items():
            if key in url:
                return response
        raise AssertionError(f"unexpected URL: {url}")
    return http_get, calls


# --------------------------------------------------------------- fetch

def test_fetch_company_concept_requires_a_real_user_agent():
    with pytest.raises(ValueError, match="user_agent"):
        edgar_data.fetch_company_concept(AAPL_CIK, AAPL_TAG, user_agent="")
    with pytest.raises(ValueError, match="user_agent"):
        edgar_data.fetch_company_concept(AAPL_CIK, AAPL_TAG, user_agent="   ")


def test_fetch_company_concept_sends_the_user_agent_header_and_correct_url():
    http_get, calls = _fake_http_get({
        f"CIK{AAPL_CIK:010d}/us-gaap/{AAPL_TAG}.json": _FakeResponse(200, {"cik": AAPL_CIK}),
    })
    edgar_data.fetch_company_concept(
        AAPL_CIK, AAPL_TAG, user_agent=_UA,
        http_get=http_get)
    assert len(calls) == 1
    assert calls[0]["headers"] == {"User-Agent": _UA}
    assert "data.sec.gov/api/xbrl/companyconcept" in calls[0]["url"]
    assert f"CIK{AAPL_CIK:010d}" in calls[0]["url"]


def test_fetch_company_concept_returns_none_on_404_never_reported():
    http_get, _ = _fake_http_get({"us-gaap": _FakeResponse(404)})
    result = edgar_data.fetch_company_concept(
        AAPL_CIK, "SomeTagAAPLNeverUsed", user_agent=_CONTACT, http_get=http_get)
    assert result is None


def test_fetch_company_concept_returns_none_on_200_with_empty_units_usd():
    # Observed screening the real S&P 500: SEC signals
    # "this company has never disclosed a USD fact under this taxonomy
    # tag" via a 200 response carrying units.USD: [] just as often as via
    # a 404 (real example: BIIB CIK 875045,
    # PaymentsToAcquirePropertyPlantAndEquipment). Both must mean the
    # same thing to a fallback-tag caller -- "this tag doesn't apply,
    # try the next one" -- never a raised error that kills the whole
    # multi-tag fetch before later fallback tags are even tried.
    http_get, _ = _fake_http_get({
        "us-gaap": _FakeResponse(200, {"cik": AAPL_CIK, "tag": "SomeUnusedTag",
                                       "units": {"USD": []}}),
    })
    result = edgar_data.fetch_company_concept(
        AAPL_CIK, "SomeUnusedTag", user_agent=_CONTACT, http_get=http_get)
    assert result is None


def test_fetch_all_tags_skips_a_200_empty_units_tag_and_still_tries_the_rest():
    http_get, _ = _fake_http_get({
        "TagA": _FakeResponse(200, {"cik": 1, "tag": "TagA", "units": {"USD": []}}),
        "TagB": _FakeResponse(200, _payload([_valid_fact()], tag="TagB")),
    })
    result = edgar_data.fetch_all_tags_for_company(
        1, ["TagA", "TagB"], user_agent=_CONTACT, http_get=http_get,
        sleep=lambda s: None)
    assert set(result) == {"TagB"}


def test_fetch_company_concept_raises_on_other_http_errors():
    http_get, _ = _fake_http_get({"us-gaap": _FakeResponse(403)})
    with pytest.raises(requests_module.HTTPError):
        edgar_data.fetch_company_concept(
            AAPL_CIK, AAPL_TAG, user_agent=_CONTACT, http_get=http_get)


# ---------------------------------------------------- parse (real fixture)

def test_parse_pinned_capex_fixture():
    payload = _load_fixture()
    frame = edgar_data.parse_company_concept_payload(
        payload, cik=AAPL_CIK, tag=AAPL_TAG)
    assert len(frame) > 50
    assert list(frame.columns) == list(edgar_data.REQUIRED_FACT_FIELDS)
    for col in ("start", "end", "filed"):
        assert pd.api.types.is_datetime64_any_dtype(frame[col])
    assert frame["val"].dtype == float
    assert (frame["val"] > 0).all()
    assert frame["filed"].is_monotonic_increasing  # sorted output contract

    fy_10k = frame[(frame["fp"] == "FY") & (frame["form"].isin(["10-K", "10-K/A"]))]
    assert len(fy_10k) > 5
    # A specific fact, verified present (value scaled like the rest of the fixture).
    row = frame[(frame["start"] == pd.Timestamp("2012-09-30")) &
               (frame["end"] == pd.Timestamp("2013-09-28"))]
    assert len(row) == 1
    assert row["val"].iloc[0] == pytest.approx(816_500)
    assert row["filed"].iloc[0] == pd.Timestamp("2015-10-28")


def test_fetch_company_concept_series_end_to_end_with_real_fixture():
    payload = _load_fixture()
    http_get, _ = _fake_http_get({"us-gaap": _FakeResponse(200, payload)})
    frame = edgar_data.fetch_company_concept_series(
        AAPL_CIK, AAPL_TAG, user_agent=_CONTACT, http_get=http_get)
    assert frame is not None
    assert len(frame) > 50


# --------------------------------------------------------- schema drift

def _valid_fact(**overrides):
    fact = {"start": "2020-01-01", "end": "2020-12-31", "val": 100.0,
           "accn": "0001", "fy": 2020, "fp": "FY", "form": "10-K",
           "filed": "2021-02-01"}
    fact.update(overrides)
    return fact


def _payload(facts, tag="T"):
    return {"cik": 1, "tag": tag, "units": {"USD": facts}}


def test_parse_rejects_wrong_cik_or_tag():
    with pytest.raises(ValueError, match="cik"):
        edgar_data.parse_company_concept_payload(_payload([_valid_fact()]), cik=2, tag="T")
    with pytest.raises(ValueError, match="tag"):
        edgar_data.parse_company_concept_payload(_payload([_valid_fact()]), cik=1, tag="OTHER")


def test_parse_rejects_missing_units_usd():
    with pytest.raises(ValueError, match="units.USD"):
        edgar_data.parse_company_concept_payload({"cik": 1, "tag": "T", "units": {}},
                                                  cik=1, tag="T")


def test_parse_rejects_empty_facts_list():
    with pytest.raises(ValueError, match="non-empty"):
        edgar_data.parse_company_concept_payload(_payload([]), cik=1, tag="T")


def test_parse_rejects_missing_required_field():
    fact = _valid_fact()
    del fact["filed"]
    with pytest.raises(ValueError, match="missing field"):
        edgar_data.parse_company_concept_payload(_payload([fact]), cik=1, tag="T")


def test_parse_rejects_non_numeric_val():
    with pytest.raises(ValueError, match="non-numeric"):
        edgar_data.parse_company_concept_payload(
            _payload([_valid_fact(val="not-a-number")]), cik=1, tag="T")


def test_parse_rejects_unparseable_date():
    with pytest.raises(ValueError, match="filed"):
        edgar_data.parse_company_concept_payload(
            _payload([_valid_fact(filed="not-a-date")]), cik=1, tag="T")


def test_parse_tolerates_null_fy_and_fp_on_8k_sourced_facts():
    # Observed screening real AAPL/MSFT capex history: SEC
    # returns fy=null, fp=null, form="8-K" for comparative-year facts
    # disclosed via an 8-K restatement (not a periodic 10-K/10-Q), e.g.
    # an AAPL PaymentsToAcquireProductiveAssets fact from a 2015 8-K.
    # This is real, legitimate SEC data, not malformed input -- it must
    # parse (and then be filtered out by annual_facts_only's own
    # fp=="FY"/form-in-10-K check), not crash the whole company's fetch.
    facts = [
        _valid_fact(fy=2020, fp="FY", form="10-K"),
        _valid_fact(fy=None, fp=None, form="8-K", filed="2015-01-28"),
    ]
    frame = edgar_data.parse_company_concept_payload(_payload(facts), cik=1, tag="T")
    assert len(frame) == 2
    ten_k_row = frame[frame["form"] == "10-K"].iloc[0]
    eight_k_row = frame[frame["form"] == "8-K"].iloc[0]
    assert ten_k_row["fy"] == 2020
    assert pd.isna(eight_k_row["fy"])
    assert eight_k_row["fp"] is None


def test_parse_frame_is_sorted_by_filed():
    facts = [_valid_fact(filed="2022-01-01", val=3.0),
            _valid_fact(filed="2020-01-01", val=1.0),
            _valid_fact(filed="2021-01-01", val=2.0)]
    frame = edgar_data.parse_company_concept_payload(_payload(facts), cik=1, tag="T")
    assert frame["val"].tolist() == [1.0, 2.0, 3.0]


# ------------------------------------------------------------- multi-tag

def test_fetch_all_tags_returns_only_tags_the_company_actually_used():
    http_get, calls = _fake_http_get({
        "TagA": _FakeResponse(200, _payload([_valid_fact()], tag="TagA")),
        "TagB": _FakeResponse(404),
        "TagC": _FakeResponse(200, _payload([_valid_fact()], tag="TagC")),
    })
    result = edgar_data.fetch_all_tags_for_company(
        1, ["TagA", "TagB", "TagC"], user_agent=_CONTACT, http_get=http_get,
        sleep=lambda s: None)
    assert set(result) == {"TagA", "TagC"}


def test_fetch_all_tags_throttles_between_requests():
    sleeps = []

    def http_get(url, **kw):
        # Reflect whichever tag was requested so the tag cross-check passes.
        tag = url.rsplit("/", 1)[-1].removesuffix(".json")
        return _FakeResponse(200, _payload([_valid_fact()], tag=tag))

    edgar_data.fetch_all_tags_for_company(
        1, ["Tag1", "Tag2", "Tag3"], user_agent=_CONTACT, http_get=http_get,
        sleep=lambda s: sleeps.append(s), min_interval=0.2)
    assert sleeps == [0.2, 0.2]  # (n-1) sleeps for n requests, never before the first


def test_fetch_all_tags_skips_a_tag_whose_parse_raises_and_keeps_the_rest():
    # Observed screening the real S&P 500: TXN (CIK
    # 97476) has a stray INSTANT-typed fact (frame "CY2021Q2I", no
    # "start") filed under the nominally duration-typed
    # PaymentsToAcquireProductiveAssets concept -- a real filer tagging
    # anomaly, not a network/schema failure. One company's one bad tag
    # must not discard data already fetched from that SAME company's
    # other, cleanly-parsing tags -- matching CAPEX_TAGS's own docstring
    # intent ("fetch every tag in this list and keep whichever return
    # data").
    http_get, _ = _fake_http_get({
        "TagA": _FakeResponse(200, _payload([_valid_fact()], tag="TagA")),
        "TagB": _FakeResponse(200, {
            "cik": 1, "tag": "TagB",
            "units": {"USD": [{"end": "2021-06-30", "val": 900_000_000,
                               "accn": "0001", "fy": 2021, "fp": "Q2",
                               "form": "10-Q", "filed": "2021-07-22"}]},
            # missing "start" on the one fact -- malformed relative to
            # REQUIRED_FACT_FIELDS, parse_company_concept_payload raises
        }),
        "TagC": _FakeResponse(200, _payload([_valid_fact()], tag="TagC")),
    })
    result = edgar_data.fetch_all_tags_for_company(
        1, ["TagA", "TagB", "TagC"], user_agent=_CONTACT, http_get=http_get,
        sleep=lambda s: None)
    assert set(result) == {"TagA", "TagC"}


def test_fetch_all_tags_empty_when_none_match():
    http_get, _ = _fake_http_get({"us-gaap": _FakeResponse(404)})
    result = edgar_data.fetch_all_tags_for_company(
        1, ["TagA", "TagB"], user_agent=_CONTACT, http_get=http_get,
        sleep=lambda s: None)
    assert result == {}


# -------------------------------------------------- inventory tag plumbing

def test_revenue_tags_leads_with_the_real_dominant_modern_tag():
    # Across a real 19-company retail sample, every filer reports under
    # "...ExcludingAssessedTax", none under "...ExcludingTax" (no
    # "Assessed").
    assert edgar_data.REVENUE_TAGS[0] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert "SalesRevenueGoodsNet" in edgar_data.REVENUE_TAGS


def test_inventory_tags_is_inventory_net_first():
    # InventoryNet must be tried before InventoryGross everywhere a
    # priority pick is made -- observed that a real
    # co-reporter (Abercrombie & Fitch, CIK 1018840) discloses BOTH, and
    # Gross is structurally >= Net (gross before a valuation reserve), so
    # the order is what makes Net win.
    assert edgar_data.INVENTORY_TAGS[0] == "InventoryNet"
    assert "InventoryGross" in edgar_data.INVENTORY_TAGS


def test_inventory_tags_covers_real_retail_specific_concepts():
    # Observed: real, currently-reporting XRT-holding
    # retailers (GAP, AutoZone, Ulta Beauty, Dillard's, Kroger, Murphy
    # USA, Dollar Tree, Upbound Group) that hold real inventory but do
    # not use InventoryNet as their primary balance-sheet tag.
    for tag in ("InventoryFinishedGoodsNetOfReserves",
                "RetailRelatedInventoryMerchandise",
                "InventoryFinishedGoods"):
        assert tag in edgar_data.INVENTORY_TAGS
    # NetOfReserves before the bare tag: GAP reports both (the bare tag's
    # facts stop in 2012 when GAP switched), so the more current, more
    # complete series must win.
    assert (edgar_data.INVENTORY_TAGS.index("InventoryFinishedGoodsNetOfReserves")
           < edgar_data.INVENTORY_TAGS.index("InventoryFinishedGoods"))


# ----------------------------------------------------- priority tag pick

def test_select_priority_series_picks_the_first_present_tag():
    tag_a = pd.DataFrame({"val": [1.0]})
    tag_b = pd.DataFrame({"val": [2.0]})
    series, tag = edgar_data.select_priority_series(
        {"TagB": tag_b, "TagA": tag_a}, priority=("TagA", "TagB"))
    assert tag == "TagA"
    assert series is tag_a


def test_select_priority_series_falls_back_when_first_priority_tag_absent():
    tag_b = pd.DataFrame({"val": [2.0]})
    series, tag = edgar_data.select_priority_series(
        {"TagB": tag_b}, priority=("TagA", "TagB"))
    assert tag == "TagB"
    assert series is tag_b


def test_select_priority_series_never_merges_across_tags():
    # The core reason this exists rather than reusing capex_watchlist's
    # resolve_tag_conflicts (prefer-the-larger-value merge): that rule is
    # WRONG for InventoryNet vs InventoryGross, where Gross is always >=
    # Net by definition and would win every conflict, the opposite of the
    # intended "prefer the standard net concept" behavior. Proven here by
    # asserting the LARGER-valued tag (TagB) does NOT win when it is
    # lower priority.
    tag_a_smaller = pd.DataFrame({"val": [1.0]})
    tag_b_larger = pd.DataFrame({"val": [999.0]})
    series, tag = edgar_data.select_priority_series(
        {"TagA": tag_a_smaller, "TagB": tag_b_larger}, priority=("TagA", "TagB"))
    assert tag == "TagA"
    assert series["val"].iloc[0] == 1.0


def test_select_priority_series_returns_none_when_nothing_present():
    series, tag = edgar_data.select_priority_series({}, priority=("TagA", "TagB"))
    assert series is None
    assert tag is None


def test_select_priority_series_skips_an_empty_frame_for_a_tag():
    empty = pd.DataFrame({"val": []})
    real = pd.DataFrame({"val": [5.0]})
    series, tag = edgar_data.select_priority_series(
        {"TagA": empty, "TagB": real}, priority=("TagA", "TagB"))
    assert tag == "TagB"
    assert series is real


# ------------------------------------------------- instant-concept plumbing
#
# Observed building the retail_inventory_sales_breadth_short
# candidate's coverage check: InventoryNet is a BALANCE-SHEET ("instant")
# XBRL concept -- reported as-of a single date, never over a start..end
# period the way capex/revenue (duration concepts) are. A real SEC
# companyconcept fact under InventoryNet carries NO "start" field at all
# (confirmed against WMT CIK 104169: every InventoryNet
# fact is missing "start"). parse_company_concept_payload's existing
# REQUIRED_FACT_FIELDS correctly RAISES on a missing "start" -- that is
# deliberate, tested behavior for duration tags (see
# test_fetch_all_tags_skips_a_tag_whose_parse_raises_and_keeps_the_rest,
# TXN's real stray-instant-fact-under-a-duration-tag anomaly) and must
# not change. Inventory needs its OWN parse path instead.

def _instant_fact(**overrides):
    fact = {"end": "2019-12-31", "val": 100.0, "accn": "0001", "fy": 2019,
           "fp": "FY", "form": "10-K", "filed": "2020-02-01"}
    fact.update(overrides)
    return fact


def test_required_instant_fact_fields_excludes_start():
    assert "start" not in edgar_data.REQUIRED_INSTANT_FACT_FIELDS
    assert set(edgar_data.REQUIRED_INSTANT_FACT_FIELDS) == {
        "end", "val", "accn", "fy", "fp", "form", "filed"}


def test_parse_instant_concept_payload_accepts_a_real_shaped_no_start_fact():
    payload = _payload([_instant_fact()], tag="InventoryNet")
    frame = edgar_data.parse_instant_concept_payload(
        payload, cik=1, tag="InventoryNet")
    assert list(frame.columns) == list(edgar_data.REQUIRED_INSTANT_FACT_FIELDS)
    assert "start" not in frame.columns
    assert frame["val"].iloc[0] == 100.0
    assert pd.api.types.is_datetime64_any_dtype(frame["end"])


def test_parse_instant_concept_payload_raises_on_missing_end():
    fact = _instant_fact()
    del fact["end"]
    with pytest.raises(ValueError, match="missing field"):
        edgar_data.parse_instant_concept_payload(
            _payload([fact], tag="InventoryNet"), cik=1, tag="InventoryNet")


def test_parse_instant_concept_payload_rejects_wrong_cik_or_tag():
    with pytest.raises(ValueError, match="cik"):
        edgar_data.parse_instant_concept_payload(
            _payload([_instant_fact()], tag="InventoryNet"), cik=2, tag="InventoryNet")
    with pytest.raises(ValueError, match="tag"):
        edgar_data.parse_instant_concept_payload(
            _payload([_instant_fact()], tag="InventoryNet"), cik=1, tag="Other")


def test_fetch_instant_concept_series_end_to_end():
    http_get, _ = _fake_http_get({
        "InventoryNet": _FakeResponse(
            200, _payload([_instant_fact()], tag="InventoryNet")),
    })
    frame = edgar_data.fetch_instant_concept_series(
        1, "InventoryNet", user_agent=_CONTACT, http_get=http_get)
    assert frame is not None
    assert len(frame) == 1
    assert "start" not in frame.columns


def test_fetch_instant_concept_series_returns_none_on_404():
    http_get, _ = _fake_http_get({"us-gaap": _FakeResponse(404)})
    frame = edgar_data.fetch_instant_concept_series(
        1, "InventoryNet", user_agent=_CONTACT, http_get=http_get)
    assert frame is None


def test_fetch_all_instant_tags_for_company_tries_every_tag():
    http_get, _ = _fake_http_get({
        "InventoryNet": _FakeResponse(404),
        "InventoryGross": _FakeResponse(
            200, _payload([_instant_fact()], tag="InventoryGross")),
    })
    result = edgar_data.fetch_all_instant_tags_for_company(
        1, ["InventoryNet", "InventoryGross"], user_agent=_CONTACT,
        http_get=http_get, sleep=lambda s: None)
    assert set(result) == {"InventoryGross"}


def test_fetch_all_instant_tags_skips_a_tag_whose_parse_raises_and_keeps_the_rest():
    http_get, _ = _fake_http_get({
        "TagA": _FakeResponse(200, _payload([_instant_fact()], tag="TagA")),
        "TagB": _FakeResponse(200, {
            "cik": 1, "tag": "TagB",
            "units": {"USD": [{"val": 5.0, "accn": "0001", "fy": 2020,
                               "fp": "FY", "form": "10-K", "filed": "2021-01-01"}]},
            # missing "end" -- malformed relative to REQUIRED_INSTANT_FACT_FIELDS
        }),
        "TagC": _FakeResponse(200, _payload([_instant_fact()], tag="TagC")),
    })
    result = edgar_data.fetch_all_instant_tags_for_company(
        1, ["TagA", "TagB", "TagC"], user_agent=_CONTACT, http_get=http_get,
        sleep=lambda s: None)
    assert set(result) == {"TagA", "TagC"}
