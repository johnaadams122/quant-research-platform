"""Tests for src/mining/cftc_data.py -- the CFTC Legacy Futures-Only
open-interest fetcher, raw-response cache with a sha256 manifest, and the
publication-availability map (D+6 calendar days, plus the three publication
blackouts).

Everything here uses SMALL SYNTHETIC recorded payloads shaped like the real
Socrata response and a fake ``http_get``; nothing reads the network. Every test
that writes anything writes only under ``tmp_path`` (``fetch_and_cache_all``
has NO default cache directory on purpose, so a test cannot reach the real
``data/mining/cftc_oi/`` folder by forgetting an argument).
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import cftc_data as cd  # noqa: E402

_MODULE_PATH = os.path.join(_SRC, "mining", "cftc_data.py")
_DOLLAR_FIGURE = re.compile(r"\$\s?\d")
_YYYYMMDD_INT = re.compile(r"(?<![\d.\-_])(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?![\d\-_])")
# a bare run of 8 to 17 digits
_NUM_SHAPE = re.compile(r"(?<!\d\.)\b\d{8,17}\b")


# ------------------------------------------------------------ recorded payloads

_NAME_BY_CODE = {
    "002602": "CORN - CHICAGO BOARD OF TRADE",
    "005602": "SOYBEANS - CHICAGO BOARD OF TRADE",
    "001602": "WHEAT-SRW - CHICAGO BOARD OF TRADE",
    "080732": "SUGAR NO. 11 - ICE FUTURES U.S.",
    "083731": "COFFEE C - ICE FUTURES U.S.",
    "033661": "COTTON NO. 2 - ICE FUTURES U.S.",
    "073732": "COCOA - ICE FUTURES U.S.",
    "067651": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    "023651": "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
    "022651": "NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE",
    "111659": "GASOLINE BLENDSTOCK (RBOB) - NEW YORK MERCANTILE EXCHANGE",
    "057642": "LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE",
    "054642": "LEAN HOGS - CHICAGO MERCANTILE EXCHANGE",
    "088691": "GOLD - COMMODITY EXCHANGE INC.",
    "084691": "SILVER - COMMODITY EXCHANGE INC.",
    "085692": "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
    "076651": "PLATINUM - NEW YORK MERCANTILE EXCHANGE",
    "075651": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE",
}


def _rows(code, dates, values, name=None):
    name = _NAME_BY_CODE[code] if name is None else name
    return [{"report_date_as_yyyy_mm_dd": pd.Timestamp(d).strftime("%Y-%m-%dT00:00:00.000"),
             "cftc_contract_market_code": code,
             "open_interest_all": str(v),
             "market_and_exchange_names": name}
            for d, v in zip(dates, values)]


def _payload(rows) -> bytes:
    return json.dumps(rows).encode("utf-8")


def _weekly_payload(code, start="2010-01-05", periods=30, base=1000):
    dates = pd.date_range(start, periods=periods, freq="7D")
    return _payload(_rows(code, dates, [base + 10 * i for i in range(periods)]))


class _Response:
    def __init__(self, content: bytes, status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.content)


def _fake_get(payload_by_code=None, calls=None):
    payload_by_code = payload_by_code or {}

    def get(url, params=None, timeout=None):
        if calls is not None:
            calls.append((url, dict(params or {}), timeout))
        code = re.search(r"'([^']+)'", params["$where"]).group(1)
        return _Response(payload_by_code.get(code) or _weekly_payload(code))
    return get


_CLOCK = lambda: pd.Timestamp("2026-09-24T09:00:00Z").to_pydatetime()  # noqa: E731


# ------------------------------------------------------------- contract table

def test_the_contract_table_is_exactly_the_eighteen_pre_registered_codes_in_four_sectors():
    by_sector = cd.contracts_by_sector()
    assert set(by_sector) == {"agriculture", "energy", "livestock", "metals"}
    assert by_sector["agriculture"] == ["002602", "005602", "001602", "080732", "083731", "033661", "073732"]
    assert by_sector["energy"] == ["067651", "023651", "022651", "111659"]
    assert by_sector["livestock"] == ["057642", "054642"]
    assert by_sector["metals"] == ["088691", "084691", "085692", "076651", "075651"]
    assert len(cd.CONTRACT_CODES) == 18 == len(set(cd.CONTRACT_CODES))
    assert set(cd.CONTRACT_CODES) == set(_NAME_BY_CODE)
    assert cd.SECTORS == ("agriculture", "energy", "livestock", "metals")


def test_every_contract_names_at_least_one_keyword_its_real_names_contain():
    for contract in cd.CONTRACTS:
        assert contract.name_keywords, contract.code
        assert any(k in _NAME_BY_CODE[contract.code].upper() for k in contract.name_keywords), contract.code


def test_the_query_asks_for_one_contract_and_only_the_fields_the_signal_needs():
    params = cd.query_params("088691")
    assert params["$where"] == "cftc_contract_market_code='088691'"
    assert set(params["$select"].split(",")) == {
        "report_date_as_yyyy_mm_dd", "cftc_contract_market_code", "open_interest_all",
        "market_and_exchange_names"}
    assert params["$order"] == "report_date_as_yyyy_mm_dd"
    assert int(params["$limit"]) == cd.PAGE_LIMIT
    with pytest.raises(ValueError, match="not one of the 18"):
        cd.query_params("999999")


def test_the_dataset_is_the_legacy_futures_only_socrata_table():
    assert cd.SOCRATA_DATASET_ID == "6dca-aqww"
    assert cd.SOCRATA_URL.endswith("/resource/6dca-aqww.json")


# --------------------------------------------------------------------- parser

def test_parser_returns_a_sorted_float_series_indexed_by_the_as_of_date():
    rows = _rows("088691", ["2010-01-12", "2010-01-05", "2010-01-19"], [1100, 1000, 1200])
    series = cd.parse_open_interest("088691", _payload(rows))
    assert list(series.index) == [pd.Timestamp("2010-01-05"), pd.Timestamp("2010-01-12"), pd.Timestamp("2010-01-19")]
    assert series.tolist() == [1000.0, 1100.0, 1200.0]
    assert series.dtype == float and series.name == "088691"
    assert series.attrs["contract_code"] == "088691" and series.attrs["sector"] == "metals"
    assert series.attrs["point_in_time_status"] == cd.POINT_IN_TIME_STATUS
    assert series.attrs["date_semantics"] == "position_as_of_tuesday_not_release_date"


@pytest.mark.parametrize("mutate,reason", [
    (lambda rows: {"not": "a list"}, "payload_not_a_list"),
    (lambda rows: [], "payload_empty"),
    (lambda rows: [{k: v for k, v in rows[0].items() if k != "open_interest_all"}] + rows[1:], "row_missing_field"),
    (lambda rows: [{**rows[0], "cftc_contract_market_code": "002602"}] + rows[1:], "wrong_contract_code"),
    (lambda rows: [{**rows[0], "open_interest_all": "12x"}] + rows[1:], "non_numeric_open_interest"),
    (lambda rows: [{**rows[0], "open_interest_all": "0"}] + rows[1:], "non_positive_open_interest"),
    (lambda rows: [{**rows[0], "open_interest_all": "-5"}] + rows[1:], "non_positive_open_interest"),
    (lambda rows: [{**rows[0], "open_interest_all": "nan"}] + rows[1:], "non_numeric_open_interest"),
    (lambda rows: [{**rows[0], "report_date_as_yyyy_mm_dd": "not-a-date"}] + rows[1:], "unparseable_date"),
    (lambda rows: [{**rows[0], "report_date_as_yyyy_mm_dd": "2010-01-05T00:00:00.000Z"}] + rows[1:], "unparseable_date"),
    (lambda rows: rows + [rows[0]], "duplicate_as_of_date"),
    (lambda rows: [{**rows[0], "market_and_exchange_names": "CORN - CHICAGO BOARD OF TRADE"}] + rows[1:],
     "unexpected_contract_name"),
    (lambda rows: ["a string row"] + rows[1:], "row_not_an_object"),
])
def test_parser_refuses_every_malformed_payload_with_a_reason_code(mutate, reason):
    good = _rows("088691", pd.date_range("2010-01-05", periods=4, freq="7D"), [1000, 1010, 1020, 1030])
    bad = mutate(good)
    with pytest.raises(cd.CftcPayloadError) as caught:
        cd.parse_open_interest("088691", _payload(bad))
    assert caught.value.reason_code == reason
    assert isinstance(caught.value, ValueError)


def test_parser_refuses_bytes_that_are_not_json():
    with pytest.raises(cd.CftcPayloadError) as caught:
        cd.parse_open_interest("088691", b"<html>gateway error</html>")
    assert caught.value.reason_code == "payload_not_json"


def test_parser_refuses_a_response_that_hit_the_page_limit_because_socrata_truncates_silently(monkeypatch):
    monkeypatch.setattr(cd, "PAGE_LIMIT", 5)
    rows = _rows("088691", pd.date_range("2010-01-05", periods=5, freq="7D"), [1, 2, 3, 4, 5])
    with pytest.raises(cd.CftcPayloadError) as caught:
        cd.parse_open_interest("088691", _payload(rows))
    assert caught.value.reason_code == "page_limit_reached"
    cd.parse_open_interest("088691", _payload(rows[:4]))             # one under the limit is fine


def test_parser_accepts_the_real_renamed_contracts():
    """Codes are stable while names change over time (heating oil became ULSD,
    exchange names moved): a keyword set, not one exact name."""
    rows = (_rows("022651", ["2010-01-05"], [10], name="NO. 2 HEATING OIL, N.Y. HARBOR - NEW YORK MERCANTILE EXCHANGE")
            + _rows("022651", ["2010-01-12"], [11], name="NY HARBOR USLD - NEW YORK MERCANTILE EXCHANGE")
            + _rows("022651", ["2010-01-19"], [12], name="#2 HEATING OIL- NY HARBOR-ULSD - NEW YORK MERCANTILE EXCHANGE"))
    assert cd.parse_open_interest("022651", _payload(rows)).tolist() == [10.0, 11.0, 12.0]


# ------------------------------------------------------------- the D+6 lag

def test_a_report_is_usable_from_six_calendar_days_after_its_as_of_date_not_five():
    tuesday = pd.Timestamp("2010-06-01")
    assert tuesday.dayofweek == 1
    assert cd.availability_date(tuesday) == pd.Timestamp("2010-06-07")
    assert cd.AVAILABILITY_LAG_DAYS == 6
    assert cd.availability_date(tuesday).dayofweek == 0              # always a Monday
    idx = pd.DatetimeIndex(["2010-06-01", "2010-06-08"])
    assert list(cd.availability_dates(idx)) == [pd.Timestamp("2010-06-07"), pd.Timestamp("2010-06-14")]


def _series(dates, values, code="088691"):
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates), name=code)
    return s


def test_month_end_value_is_the_last_report_available_by_that_date_with_the_boundary_exact():
    # Tuesdays; month-end 2010-06-30 is a Wednesday
    dates = pd.date_range("2010-06-01", periods=5, freq="7D")           # 06-01, 06-08, 06-15, 06-22, 06-29
    s = _series(dates, [10, 11, 12, 13, 14])
    values, used = cd.month_end_open_interest({"088691": s}, [pd.Timestamp("2010-06-30")])
    # 06-22 becomes usable 06-28 (<= 06-30); 06-29 becomes usable 07-05 (> 06-30)
    assert values.loc[pd.Timestamp("2010-06-30"), "088691"] == 13.0
    assert used.loc[pd.Timestamp("2010-06-30"), "088691"] == pd.Timestamp("2010-06-22")
    # decision date exactly D+6 -> usable; one day earlier -> not
    v6, _ = cd.month_end_open_interest({"088691": s}, [pd.Timestamp("2010-06-28")])
    v5, _ = cd.month_end_open_interest({"088691": s}, [pd.Timestamp("2010-06-27")])
    assert v6.iloc[0, 0] == 13.0 and v5.iloc[0, 0] == 12.0


def test_month_end_value_is_nan_before_the_first_report_is_available():
    s = _series(["2010-06-01"], [10])
    values, used = cd.month_end_open_interest({"088691": s}, [pd.Timestamp("2010-06-06"), pd.Timestamp("2010-06-07")])
    assert np.isnan(values.iloc[0, 0]) and pd.isna(used.iloc[0, 0])
    assert values.iloc[1, 0] == 10.0


def test_a_later_report_never_leaks_into_an_earlier_decision_date():
    dates = pd.date_range("2010-01-05", periods=40, freq="7D")
    base = _series(dates, np.arange(40) + 100.0)
    perturbed = base.copy()
    perturbed.iloc[20:] = 9999.0
    decision = pd.Timestamp(dates[20]) + pd.Timedelta(days=5)          # one day before report 20 is usable
    a, _ = cd.month_end_open_interest({"088691": base}, [decision])
    b, _ = cd.month_end_open_interest({"088691": perturbed}, [decision])
    assert a.iloc[0, 0] == b.iloc[0, 0] == 119.0


# ------------------------------------------------------- publication blackouts

def test_the_three_blackouts_are_recorded_with_their_published_dates():
    names = [b.name for b in cd.PUBLICATION_BLACKOUTS]
    assert len(names) == 3
    b1, b2, b3 = cd.PUBLICATION_BLACKOUTS
    assert (b1.first_delayed_asof, b1.public_by) == (pd.Timestamp("2013-10-01"), pd.Timestamp("2013-11-08"))
    # the first delayed report is the Christmas-shifted Monday 2018-12-24 (the CFTC's own notice: the last report
    # published before the suspension was published 2018-12-21; publication resumed with the 2018-12-24 report)
    assert (b2.first_delayed_asof, b2.last_delayed_asof, b2.public_by) == (
        pd.Timestamp("2018-12-24"), pd.Timestamp("2019-02-26"), pd.Timestamp("2019-03-08"))
    assert (b3.first_delayed_asof, b3.public_by) == (pd.Timestamp("2025-09-30"), pd.Timestamp("2026-01-23"))
    for b in cd.PUBLICATION_BLACKOUTS:
        assert b.first_delayed_asof <= b.last_delayed_asof < b.public_by
        assert b.evidence


@pytest.mark.parametrize("month_end,expected", [
    # 2013 shutdown: reports as-of 2013-10-01 on were first public 2013-10-25 and normal from 2013-11-08
    ("2013-09-30", False),      # last normal report (as-of 09-24) was public before the shutdown
    ("2013-10-31", True),       # the plain lag would use as-of 10-22, not yet public
    ("2013-11-29", False),      # normal cadence back
    # 2018-19 shutdown
    ("2018-11-30", False),
    ("2018-12-31", True),       # plain lag uses as-of 12-24, first published 2019-02-01
    ("2019-01-31", True),       # catch-up had only just started
    ("2019-02-28", True),       # plain lag uses as-of 02-19, published about 03-01 on the announced cadence
    ("2019-03-29", False),      # current again from 2019-03-08
    # 2025-26 shutdown, the last three month-ends fall inside the ABI-1 holdout
    ("2025-09-30", False),
    ("2025-10-31", True),
    ("2025-11-28", True),       # Nov 30 2025 is a Sunday: the last session is Friday the 28th
    ("2025-12-31", True),
    ("2026-01-30", False),      # backlog cleared with the report published 2026-01-23
    ("2026-08-31", False),
    # ordinary months
    ("2010-06-30", False),
    ("2020-03-31", False),
    ("2013-10-01", False),      # the very first delayed as-of date is not yet usable under the plain lag either
])
def test_each_month_end_is_classified_blackout_or_normal(month_end, expected):
    assert (cd.blackout_at(pd.Timestamp(month_end)) is not None) is expected


def test_the_blackout_lookup_names_the_blackout_it_falls_in():
    assert cd.blackout_at(pd.Timestamp("2013-10-31")).public_by == pd.Timestamp("2013-11-08")
    assert cd.blackout_at(pd.Timestamp("2025-11-28")).public_by == pd.Timestamp("2026-01-23")
    assert cd.blackout_at(pd.Timestamp("2019-01-31")).first_delayed_asof == pd.Timestamp("2018-12-24")
    assert cd.blackout_at(pd.Timestamp("2019-02-28")).public_by == pd.Timestamp("2019-03-08")


def test_a_delayed_report_is_available_only_from_its_conservative_public_by_date():
    b1 = cd.PUBLICATION_BLACKOUTS[0]
    assert cd.availability_date(pd.Timestamp("2013-10-15")) == b1.public_by          # delayed
    assert cd.availability_date(pd.Timestamp("2013-09-24")) == pd.Timestamp("2013-09-30")   # last normal one
    assert cd.availability_date(pd.Timestamp("2013-11-05")) == pd.Timestamp("2013-11-11")   # normal again
    assert cd.availability_date(pd.Timestamp("2025-12-16")) == pd.Timestamp("2026-01-23")


def test_at_a_blackout_month_end_the_value_is_the_last_report_that_was_really_public():
    """The plain D+6 rule would read as-of 2013-10-22 at 2013-10-31. The blackout-aware
    availability map must return the last pre-shutdown report (as-of 2013-09-24)."""
    dates = pd.date_range("2013-09-03", "2013-12-31", freq="7D")          # Tuesdays
    assert pd.Timestamp("2013-09-24") in dates and pd.Timestamp("2013-10-22") in dates
    s = _series(dates, np.arange(len(dates)) + 500.0)
    values, used = cd.month_end_open_interest({"088691": s}, [pd.Timestamp("2013-09-30"), pd.Timestamp("2013-10-31"),
                                                              pd.Timestamp("2013-11-29")])
    assert used.iloc[0, 0] == pd.Timestamp("2013-09-24")
    assert used.iloc[1, 0] == pd.Timestamp("2013-09-24")                    # NOT 2013-10-22
    assert used.iloc[2, 0] == pd.Timestamp("2013-11-19")
    assert values.iloc[1, 0] == values.iloc[0, 0]
    plain = s[s.index + pd.Timedelta(days=6) <= pd.Timestamp("2013-10-31")]
    assert plain.index[-1] == pd.Timestamp("2013-10-22")                    # the plain-lag answer the map rejects


def test_every_blackout_month_end_reads_a_report_strictly_older_than_the_delayed_set():
    """Month-end by month-end over all three blackouts."""
    for blackout, month_ends in [
        (cd.PUBLICATION_BLACKOUTS[0], ["2013-10-31"]),
        (cd.PUBLICATION_BLACKOUTS[1], ["2018-12-31", "2019-01-31", "2019-02-28"]),
        (cd.PUBLICATION_BLACKOUTS[2], ["2025-10-31", "2025-11-28", "2025-12-31"]),
    ]:
        dates = pd.date_range(blackout.first_delayed_asof - pd.Timedelta(days=70),
                              blackout.public_by + pd.Timedelta(days=70), freq="7D")
        s = _series(dates, np.arange(len(dates)) + 1.0)
        for month_end in month_ends:
            _, used = cd.month_end_open_interest({"088691": s}, [pd.Timestamp(month_end)])
            assert used.iloc[0, 0] < blackout.first_delayed_asof, (blackout.name, month_end)
            assert cd.blackout_at(pd.Timestamp(month_end)) is blackout


def test_the_plain_and_corrected_maps_agree_everywhere_outside_a_blackout_month_end():
    dates = pd.date_range("2005-01-04", "2027-01-05", freq="7D")
    s = _series(dates, np.arange(len(dates)) + 1.0)
    month_ends = pd.date_range("2005-01-31", "2026-12-31", freq="ME")
    values, used = cd.month_end_open_interest({"088691": s}, month_ends)
    for month_end in month_ends:
        plain_last = s[s.index + pd.Timedelta(days=6) <= month_end].index[-1]
        same = used.loc[month_end, "088691"] == plain_last
        assert same is (cd.blackout_at(month_end) is None), month_end


def test_the_blackout_interval_uses_the_first_delayed_report_plus_the_lag_as_its_start():
    b = cd.PUBLICATION_BLACKOUTS[0]
    just_before = b.first_delayed_asof + pd.Timedelta(days=cd.AVAILABILITY_LAG_DAYS - 1)
    at_start = b.first_delayed_asof + pd.Timedelta(days=cd.AVAILABILITY_LAG_DAYS)
    assert cd.blackout_at(just_before) is None and cd.blackout_at(at_start) is b
    assert cd.blackout_at(b.public_by - pd.Timedelta(days=1)) is b and cd.blackout_at(b.public_by) is None


def test_the_blackout_boundaries_are_exact_in_2018_19():
    """The delayed interval starts with the first delayed report's plain-lag date and ends the day before public_by."""
    assert cd.blackout_at(pd.Timestamp("2018-12-29")) is None
    assert cd.blackout_at(pd.Timestamp("2018-12-30")) is cd.PUBLICATION_BLACKOUTS[1]
    assert cd.blackout_at(pd.Timestamp("2019-03-07")) is cd.PUBLICATION_BLACKOUTS[1]
    assert cd.blackout_at(pd.Timestamp("2019-03-08")) is None


def test_a_holiday_shifted_monday_report_inside_a_blackout_is_treated_as_delayed_not_public():
    """Real history has Monday as-of dates (Christmas 2018 and New Year 2019 fall on Tuesdays, so the
    reports are dated Monday 2018-12-24 and 2018-12-31). A Tuesday-only synthetic grid cannot see them."""
    tuesdays = pd.date_range("2018-09-04", "2019-05-28", freq="7D")
    dates = tuesdays[~tuesdays.isin([pd.Timestamp("2018-12-25"), pd.Timestamp("2019-01-01")])]
    dates = dates.append(pd.DatetimeIndex(["2018-12-24", "2018-12-31"])).sort_values()
    assert pd.Timestamp("2018-12-24") in dates and pd.Timestamp("2018-12-31") in dates
    assert pd.Timestamp("2018-12-25") not in dates
    s = _series(dates, np.arange(len(dates)) + 1.0)
    month_ends = [pd.Timestamp(d) for d in ("2018-12-31", "2019-01-31", "2019-02-28")]
    _, used = cd.month_end_open_interest({"088691": s}, month_ends)
    for month_end in month_ends:
        assert used.loc[month_end, "088691"] == pd.Timestamp("2018-12-18"), month_end
    # and once public_by has passed the whole backlog is readable again
    _, later = cd.month_end_open_interest({"088691": s}, [pd.Timestamp("2019-03-29")])
    assert later.iloc[0, 0] == pd.Timestamp("2019-03-19")


# The CFTC's own published dates, transcribed from cftc.gov (dates only). Each delayed report's row is the day it was
# (or, for the 2025 rows after the notice date, was planned to be) first public. They are a test ORACLE, independent of
# the availability map under test: the map must never call a delayed report usable before it was public.
_CFTC_PUBLISHED_ON = {
    # press release 6745-13 (2013-10-23): the first delayed report (originally 2013-10-04) was published 2013-10-25
    "2013-10-01": "2013-10-25",
    # press release 7864-19 (2019-01-29): the last report before the suspension was published 2018-12-21; the next
    # one, as of Monday 2018-12-24 (originally 2018-12-28), was published 2019-02-01
    "2018-12-24": "2019-02-01",
    # press release 9147-25 (2025-12-09), the new publish date column
    "2025-09-30": "2025-11-19", "2025-10-07": "2025-11-21", "2025-10-14": "2025-11-25",
    "2025-10-21": "2025-12-02", "2025-10-28": "2025-12-05", "2025-11-04": "2025-12-09",
    "2025-11-10": "2025-12-10", "2025-11-18": "2025-12-12", "2025-11-25": "2025-12-15",
    "2025-12-02": "2025-12-17", "2025-12-09": "2025-12-19", "2025-12-16": "2025-12-23",
    "2025-12-23": "2025-12-29",
}


@pytest.mark.parametrize("asof,published", sorted(_CFTC_PUBLISHED_ON.items()))
def test_a_delayed_report_is_never_usable_before_the_cftcs_own_published_date(asof, published):
    assert cd.availability_date(pd.Timestamp(asof)) >= pd.Timestamp(published)


_REAL_CACHE = cd.DEFAULT_CACHE_DIR


@pytest.mark.slow
def test_on_the_real_cache_the_early_years_are_semi_monthly_and_the_weekly_era_is_weekly():
    """The docstring's release-cadence claim: a median gap of 15 days in 1986 and in 1990, 7 days by 1994."""
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    gold = cd.load_cached_open_interest(_REAL_CACHE).series["088691"]
    gaps = gold.index.to_series().diff().dt.days
    median_by_year = gaps.groupby(gold.index.year).median()
    assert (median_by_year[1986], median_by_year[1990], median_by_year[1994]) == (15.0, 15.0, 7.0)


@pytest.mark.slow
def test_on_the_real_cache_every_blackout_month_end_reads_a_report_that_was_really_public():
    """Real as-of dates (Monday holiday-shifted ones included). At each blackout month-end every contract must
    read the last on-time report before the delayed interval, never a delayed one."""
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(_REAL_CACHE)
    for code, series in snapshot.series.items():                     # the real Monday rows exist for every contract
        assert pd.Timestamp("2018-12-24") in series.index and pd.Timestamp("2018-12-31") in series.index, code
    blackout_month_ends = {
        "2013_federal_shutdown": ["2013-10-31"],
        "2018_19_federal_shutdown": ["2018-12-31", "2019-01-31", "2019-02-28"],
        "2025_26_federal_shutdown": ["2025-10-31", "2025-11-28", "2025-12-31"],
    }
    last_on_time = {"2013_federal_shutdown": "2013-09-24", "2018_19_federal_shutdown": "2018-12-18",
                    "2025_26_federal_shutdown": "2025-09-23"}
    for blackout in cd.PUBLICATION_BLACKOUTS:
        month_ends = [pd.Timestamp(d) for d in blackout_month_ends[blackout.name]]
        _, used = cd.month_end_open_interest(snapshot.series, month_ends)
        for month_end in month_ends:
            assert cd.blackout_at(month_end) is blackout, month_end
            assert set(used.loc[month_end]) == {pd.Timestamp(last_on_time[blackout.name])}, (blackout.name, month_end)


@pytest.mark.slow
def test_on_the_real_cache_no_decision_date_ever_reads_a_report_before_its_verified_publication_date(monkeypatch):
    """Every calendar day in and around the three blackouts (not only month-ends), on the real as-of dates: the
    report read on a day must already have been published that day. The scan has power: the same scan run against
    a deliberately too-early 2018-19 table (delayed interval starting 2018-12-25, public-by 2019-02-15) finds
    violations."""
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(_REAL_CACHE)
    days = pd.DatetimeIndex([])
    for first, last in (("2013-10-01", "2013-12-31"), ("2018-12-15", "2019-04-30"), ("2025-09-15", "2026-02-28")):
        days = days.append(pd.date_range(first, last, freq="D"))

    def violations():
        _, used = cd.month_end_open_interest(snapshot.series, days)
        found = []
        for day in days:
            for code in snapshot.series:
                asof = used.loc[day, code]
                published = None if pd.isna(asof) else _CFTC_PUBLISHED_ON.get(pd.Timestamp(asof).strftime("%Y-%m-%d"))
                if published is not None and pd.Timestamp(published) > day:
                    found.append((day.date(), code, asof.date()))
        return found

    assert violations() == []
    too_early = cd.PublicationBlackout("too_early_2018_19", pd.Timestamp("2018-12-25"), pd.Timestamp("2019-01-29"),
                                       pd.Timestamp("2019-02-15"), "a deliberately too-early table for this power check")
    monkeypatch.setattr(cd, "PUBLICATION_BLACKOUTS",
                        (cd.PUBLICATION_BLACKOUTS[0], too_early, cd.PUBLICATION_BLACKOUTS[2]))
    assert violations() != []


# ---------------------------------------------------- cache and manifest

def test_fetch_and_cache_writes_raw_bytes_and_a_manifest_with_their_sha256(tmp_path):
    # contracts differ in LENGTH (staggered starts, as in the real history) but share the same LAST as-of date
    payloads = {code: _weekly_payload(code, start=(pd.Timestamp("2010-01-05") - pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d"),
                                      periods=12 + i)
                for i, code in enumerate(cd.CONTRACT_CODES)}
    calls = []
    manifest = cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(payloads, calls), clock=_CLOCK, timeout=12.5)
    assert len(calls) == 18 and {c[0] for c in calls} == {cd.SOCRATA_URL}
    # every one of the 18 requests carries ITS OWN contract's full query (order and page limit included) and the timeout
    assert [(c[1], c[2]) for c in calls] == [(cd.query_params(code), 12.5) for code in cd.CONTRACT_CODES]
    on_disk = json.loads((tmp_path / cd.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert set(on_disk["contracts"]) == set(cd.CONTRACT_CODES)
    for i, code in enumerate(cd.CONTRACT_CODES):
        raw = (tmp_path / f"{code}.json").read_bytes()
        assert raw == payloads[code]                                     # the RAW response, byte for byte
        entry = on_disk["contracts"][code]
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["bytes"] == len(raw) and entry["rows"] == 12 + i
        assert entry["first_asof"] == (pd.Timestamp("2010-01-05") - pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d")
        assert entry["file"] == f"{code}.json"
        assert entry["last_asof"] == "2010-03-23"
    assert on_disk["dataset"] == "6dca-aqww" and on_disk["retrieved_at"] == "2026-09-24T09:00:00Z"
    assert on_disk["point_in_time_status"] == cd.POINT_IN_TIME_STATUS


def test_the_point_in_time_status_says_not_verified_never_revised_and_names_the_corrections():
    assert cd.POINT_IN_TIME_STATUS.startswith("not_verified_never_revised")
    for phrase in ("2017-03-28", "2018-09-18", "2019-10-15", "corrected", "not as first published"):
        assert phrase in cd.POINT_IN_TIME_STATUS, phrase
    assert cd.POINT_IN_TIME_STATUS != "never_revised"


def test_the_module_docstring_carries_the_point_in_time_disclosure():
    doc = cd.__doc__
    for phrase in ("NOT verified never-revised", "2017-03-28", "2018-09-18", "2019-10-15",
                   "Socrata serves the corrected", "blackout", "D+6"):
        assert phrase in doc, phrase


def test_the_docstring_states_the_lag_margin_and_release_cadence_only_as_far_as_they_hold():
    """The three-day margin is a normal-week property of the weekly era, not a universal one."""
    doc = " ".join(cd.__doc__.split())
    for phrase in ("in the weekly era", "three days of margin in a normal week", "none in a holiday-shifted week",
                   "immaterial to the state"):
        assert phrase in doc, phrase
    assert "always a Monday" not in doc
    for phrase in ("dates after 2019-02-01 are inferred from that cadence", "cleared it 2025-12-29 with the report as of 2025-12-23",
                   "so the month-end 2025-12-31, whose plain-lag report was in fact public, is still carried"):
        assert phrase in doc, phrase


def test_the_docstring_discloses_the_five_holiday_week_decisions_the_early_timing_sensitivity_and_the_survivor_universe():
    doc = " ".join(cd.__doc__.split())
    for phrase in ("2009-11-30, 2011-02-28, 2015-11-30, 2020-11-30 and 2022-02-28",
                   "US federal-holiday week whose plain D+6 date falls on the decision day itself",
                   "about 30 minutes before the close on the decision day", "NOT verified against a CFTC notice",
                   "so no state depends on them", "the other 25 month-ends",
                   "sensitive to the unverified pre-2000 release timing", "an extra 7 or 14 days of lag",
                   "present-day survivor set", "a limitation, not a deviation", "No membership rule was applied at each historical date",
                   "contracts_disagree_on_last_asof", "require_recent_reads", "older than 14 days"):
        assert phrase in doc, phrase


def test_each_blackouts_evidence_text_names_the_cftc_notice_it_rests_on_and_the_conservative_bound():
    b1, b2, b3 = cd.PUBLICATION_BLACKOUTS
    assert "6745-13" in b1.evidence and "2013-10-25" in b1.evidence and "2013-11-08" in b1.evidence
    assert "7864-19" in b2.evidence and "2018-12-24" in b2.evidence and "2019-02-01" in b2.evidence
    assert "2018-12-21" in b2.evidence and "2019-03-08" in b2.evidence and "conservative" in b2.evidence
    # the per-report dates after 2019-02-01 are the announced cadence, not read from a table
    assert "dates after 2019-02-01 are inferred from that announced cadence" in b2.evidence
    assert "the last delayed report, as of 2019-02-26, on 2019-03-05" in b2.evidence
    assert "9147-25" in b3.evidence and "2026-01-23" in b3.evidence
    assert "cleared it 2025-12-29 with the report as of 2025-12-23" in b3.evidence
    assert "conservative" in b3.evidence and "is still carried as a blackout month-end" in b3.evidence


def test_a_bad_payload_writes_nothing_and_leaves_the_previous_cache_untouched(tmp_path):
    good = {code: _weekly_payload(code) for code in cd.CONTRACT_CODES}
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(good), clock=_CLOCK)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    poisoned = dict(good)
    poisoned["085692"] = b"[]"                                           # one contract returns nothing
    with pytest.raises(cd.CftcPayloadError) as caught:
        cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get({**poisoned, "088691": _weekly_payload("088691", base=5)}),
                               clock=_CLOCK)
    assert caught.value.reason_code == "payload_empty" and "085692" in str(caught.value)
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert before == after                                               # even the contracts fetched BEFORE the bad one


def test_an_http_error_is_surfaced_not_swallowed(tmp_path):
    def get(url, params=None, timeout=None):
        return _Response(b"oops", status=503)
    with pytest.raises(RuntimeError, match="503"):
        cd.fetch_and_cache_all(str(tmp_path), http_get=get, clock=_CLOCK)
    assert list(tmp_path.iterdir()) == []


def test_load_returns_a_series_per_contract_and_the_manifest(tmp_path):
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(), clock=_CLOCK)
    snapshot = cd.load_cached_open_interest(str(tmp_path))
    assert set(snapshot.series) == set(cd.CONTRACT_CODES)
    assert all(len(s) == 30 for s in snapshot.series.values())
    assert snapshot.latest_asof == pd.Timestamp("2010-01-05") + pd.Timedelta(days=7 * 29)
    assert snapshot.manifest["dataset"] == "6dca-aqww"
    assert snapshot.point_in_time_status == cd.POINT_IN_TIME_STATUS


def test_load_refuses_a_missing_cache_directory_and_a_missing_manifest(tmp_path):
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(tmp_path / "nope"))
    assert caught.value.reason_code == "cftc_cache_missing"
    (tmp_path / "empty").mkdir()
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(tmp_path / "empty"))
    assert caught.value.reason_code == "cftc_cache_missing"


def test_load_refuses_a_raw_file_that_no_longer_matches_its_manifest_hash(tmp_path):
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(), clock=_CLOCK)
    path = tmp_path / "088691.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[3]["open_interest_all"] = "123456"                              # a hand edit of ONE value
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(tmp_path))
    assert caught.value.reason_code == "cftc_cache_corrupt" and "088691" in str(caught.value)


def test_load_refuses_a_missing_raw_file_and_an_incomplete_manifest(tmp_path):
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(), clock=_CLOCK)
    os.remove(tmp_path / "076651.json")
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(tmp_path))
    assert caught.value.reason_code == "cftc_cache_corrupt" and "076651" in str(caught.value)
    fresh = tmp_path / "second"
    fresh.mkdir()
    cd.fetch_and_cache_all(str(fresh), http_get=_fake_get(), clock=_CLOCK)
    manifest = json.loads((fresh / cd.MANIFEST_NAME).read_text(encoding="utf-8"))
    del manifest["contracts"]["075651"]
    (fresh / cd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(fresh))
    assert caught.value.reason_code == "cftc_cache_incomplete" and "075651" in str(caught.value)


def test_load_refuses_a_manifest_row_count_that_disagrees_with_the_parsed_file(tmp_path):
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(), clock=_CLOCK)
    manifest = json.loads((tmp_path / cd.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["contracts"]["002602"]["rows"] += 1
    (tmp_path / cd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(tmp_path))
    assert caught.value.reason_code == "cftc_cache_corrupt" and "row count" in str(caught.value)


def test_load_refuses_a_manifest_that_claims_the_series_is_never_revised(tmp_path):
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(), clock=_CLOCK)
    manifest = json.loads((tmp_path / cd.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["point_in_time_status"] = "never_revised"
    (tmp_path / cd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(tmp_path))
    assert caught.value.reason_code == "cftc_cache_corrupt" and "never_revised" in str(caught.value)


def test_fetch_and_cache_has_no_default_cache_directory():
    import inspect
    parameter = inspect.signature(cd.fetch_and_cache_all).parameters["cache_dir"]
    assert parameter.default is inspect.Parameter.empty


def test_the_cli_writes_only_to_the_directory_it_is_given(tmp_path, capsys):
    code = cd.main(["--cache-dir", str(tmp_path / "out")], http_get=_fake_get(), clock=_CLOCK)
    assert code == 0
    assert (tmp_path / "out" / cd.MANIFEST_NAME).exists()
    assert "18 contracts" in capsys.readouterr().out
    assert os.path.abspath(cd.DEFAULT_CACHE_DIR).endswith(os.path.join("data", "mining", "cftc_oi"))


def test_the_cli_refuses_to_run_without_an_explicit_directory_or_the_default_flag(tmp_path):
    with pytest.raises(SystemExit):
        cd.main([], http_get=_fake_get(), clock=_CLOCK)


# ------------------------------------------ one last as-of date for all 18 contracts

def _reseal_contract(cache_dir, code, payload):
    """Hand-write ONE contract's raw file and re-seal its manifest entry (sha256, bytes, rows, dates), the way a
    hand-repaired cache whose contracts end on different dates would look on disk (only the fetch enforces one
    shared last as-of date; the loader accepts a sealed cache as it is)."""
    (cache_dir / f"{code}.json").write_bytes(payload)
    manifest_path = cache_dir / cd.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    series = cd.parse_open_interest(code, payload)
    manifest["contracts"][code].update({
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "rows": int(len(series)),
        "first_asof": series.index[0].strftime("%Y-%m-%d"), "last_asof": series.index[-1].strftime("%Y-%m-%d")})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_fetch_and_cache_refuses_one_lagging_contract_among_current_ones_and_writes_nothing(tmp_path):
    """17 contracts current, one stopped two weeks early: refused with a reason code naming the contract and its
    date, before any file is written (a lagging contract would otherwise be carried forward with zero growth)."""
    payloads = {code: _weekly_payload(code, periods=30) for code in cd.CONTRACT_CODES}
    payloads["067651"] = _weekly_payload("067651", periods=28)
    with pytest.raises(cd.CftcPayloadError) as caught:
        cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(payloads), clock=_CLOCK)
    assert caught.value.reason_code == "contracts_disagree_on_last_asof"
    message = str(caught.value)
    assert "067651" in message and "2010-07-13" in message and "2010-07-27" in message
    assert list(tmp_path.iterdir()) == []
    payloads["067651"] = _weekly_payload("067651", periods=30)           # all current again: accepted
    assert len(cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(payloads), clock=_CLOCK)["contracts"]) == 18


def test_fetch_and_cache_names_every_lagging_contract_not_only_the_first(tmp_path):
    payloads = {code: _weekly_payload(code, periods=30) for code in cd.CONTRACT_CODES}
    payloads["002602"] = _weekly_payload("002602", periods=29)
    payloads["075651"] = _weekly_payload("075651", periods=20)
    with pytest.raises(cd.CftcPayloadError) as caught:
        cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(payloads), clock=_CLOCK)
    assert "002602" in str(caught.value) and "075651" in str(caught.value)


def test_the_snapshots_latest_asof_is_the_newest_report_across_contracts_not_the_oldest(tmp_path):
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(), clock=_CLOCK)
    _reseal_contract(tmp_path, "085692", _weekly_payload("085692", periods=25))     # one contract ends 5 weeks early
    snapshot = cd.load_cached_open_interest(str(tmp_path))
    newest = pd.Timestamp("2010-01-05") + pd.Timedelta(days=7 * 29)
    assert snapshot.series["085692"].index[-1] < newest
    assert snapshot.latest_asof == newest                                            # the max, not the min


@pytest.mark.parametrize("field,value", [("dataset", "abcd-1234"), ("schema_version", 2)])
def test_load_refuses_a_manifest_for_another_dataset_or_schema_version(tmp_path, field, value):
    cd.fetch_and_cache_all(str(tmp_path), http_get=_fake_get(), clock=_CLOCK)
    manifest = json.loads((tmp_path / cd.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest[field] = value
    (tmp_path / cd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(cd.CftcCacheError) as caught:
        cd.load_cached_open_interest(str(tmp_path))
    assert caught.value.reason_code == "cftc_cache_corrupt"
    assert ("dataset" if field == "dataset" else "schema version") in str(caught.value)


# ------------------------------------------------------------------ hygiene

def test_no_dollar_figure_no_yyyymmdd_integer_and_no_long_digit_run_in_the_module_or_this_file():
    for path in (_MODULE_PATH, __file__):
        text = open(path, "r", encoding="utf-8").read()
        assert not _DOLLAR_FIGURE.search(text), path
        assert not _YYYYMMDD_INT.search(text), path
        assert not _NUM_SHAPE.search(text), (path, "an 8 to 17 digit run looks like an account number")


def test_the_module_never_labels_the_series_never_revised():
    text = open(_MODULE_PATH, "r", encoding="utf-8").read()
    labels = [m.start() for m in re.finditer(r"['\"]never_revised['\"]", text)]
    assert not labels        # only ever written as "NOT verified never-revised" / not_verified_never_revised
