"""Tests for src/mining/capex_watchlist.py -- the pure ranking core of the
capex-cycle mechanical watchlist-selection rule (an objective rule
over a hand-picked list). Network-free: point-in-time series are hand-built in the exact
shape ``edgar_point_in_time.build_point_in_time_annual_series`` emits
(columns: usable_asof, latest_fiscal_year_end, latest_val,
prior_fiscal_year_end, prior_val, capex_growth_yoy), never fetched.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from mining import capex_watchlist  # noqa: E402
from mining.edgar_data import REQUIRED_FACT_FIELDS  # noqa: E402

ASOF = pd.Timestamp("2019-12-31")


def _series(usable_asof, latest_val, *, growth=0.1):
    """One-row point-in-time series, shaped like
    edgar_point_in_time.build_point_in_time_annual_series's real output."""
    return pd.DataFrame({
        "usable_asof": [pd.Timestamp(usable_asof)],
        "latest_fiscal_year_end": [pd.Timestamp(usable_asof) - pd.Timedelta(days=60)],
        "latest_val": [latest_val],
        "prior_fiscal_year_end": [pd.Timestamp(usable_asof) - pd.Timedelta(days=425)],
        "prior_val": [latest_val / (1 + growth)],
        "capex_growth_yoy": [growth],
    })


# ------------------------------------------------------- trailing_capex_asof

def test_trailing_capex_asof_returns_latest_val_when_fresh():
    series = _series("2019-05-01", 1_000_000.0)
    result = capex_watchlist.trailing_capex_asof(series, ASOF)
    assert result == 1_000_000.0


def test_trailing_capex_asof_returns_none_when_stale():
    # usable_asof over 400 days before ASOF -> point_in_time_asof's own
    # staleness cap blanks the value columns to NaN.
    series = _series("2018-01-01", 1_000_000.0)
    result = capex_watchlist.trailing_capex_asof(series, ASOF, staleness_cap_days=400)
    assert result is None


def test_trailing_capex_asof_returns_none_when_series_is_none():
    assert capex_watchlist.trailing_capex_asof(None, ASOF) is None


def test_trailing_capex_asof_returns_none_when_series_is_empty():
    empty = pd.DataFrame(columns=["usable_asof", "latest_fiscal_year_end",
                                  "latest_val", "prior_fiscal_year_end",
                                  "prior_val", "capex_growth_yoy"])
    assert capex_watchlist.trailing_capex_asof(empty, ASOF) is None


def test_trailing_capex_asof_returns_none_before_first_disclosure():
    # asof predates the company's earliest usable_asof entirely.
    series = _series("2019-05-01", 1_000_000.0)
    result = capex_watchlist.trailing_capex_asof(series, pd.Timestamp("2018-01-01"))
    assert result is None


# ------------------------------------------------------- rank_by_trailing_capex

def test_rank_by_trailing_capex_orders_descending():
    capex_by_ticker = {"A": 100.0, "B": 300.0, "C": 200.0}
    result = capex_watchlist.rank_by_trailing_capex(capex_by_ticker, n=10)
    assert [row["ticker"] for row in result] == ["B", "C", "A"]
    assert [row["trailing_capex_usd"] for row in result] == [300.0, 200.0, 100.0]


def test_rank_by_trailing_capex_drops_none_and_nan_values():
    capex_by_ticker = {"A": 100.0, "B": None, "C": float("nan"), "D": 50.0}
    result = capex_watchlist.rank_by_trailing_capex(capex_by_ticker, n=10)
    assert [row["ticker"] for row in result] == ["A", "D"]


def test_rank_by_trailing_capex_respects_n_cap():
    capex_by_ticker = {"A": 100.0, "B": 300.0, "C": 200.0, "D": 400.0}
    result = capex_watchlist.rank_by_trailing_capex(capex_by_ticker, n=2)
    assert [row["ticker"] for row in result] == ["D", "B"]


def test_rank_by_trailing_capex_ties_break_alphabetically_for_determinism():
    capex_by_ticker = {"ZEBRA": 100.0, "ALPHA": 100.0, "MID": 100.0}
    result = capex_watchlist.rank_by_trailing_capex(capex_by_ticker, n=10)
    assert [row["ticker"] for row in result] == ["ALPHA", "MID", "ZEBRA"]


def test_rank_by_trailing_capex_empty_input_returns_empty_list():
    assert capex_watchlist.rank_by_trailing_capex({}, n=10) == []


def test_rank_by_trailing_capex_all_none_returns_empty_list():
    assert capex_watchlist.rank_by_trailing_capex({"A": None, "B": None}, n=10) == []


# ------------------------------------------------------- dedupe_share_classes_by_cik

def test_dedupe_share_classes_by_cik_keeps_alphabetically_first_ticker():
    # Observed in SEC data: GOOG and GOOGL share one CIK (1652044) --
    # both are Alphabet, not two distinct companies/trials.
    capex_by_ticker = {"GOOG": 25_139_000_000.0, "GOOGL": 25_139_000_000.0,
                       "AAPL": 10_495_000_000.0}
    cik_by_ticker = {"GOOG": 1652044, "GOOGL": 1652044, "AAPL": 320193}
    result = capex_watchlist.dedupe_share_classes_by_cik(capex_by_ticker, cik_by_ticker)
    assert result == {"GOOG": 25_139_000_000.0, "AAPL": 10_495_000_000.0}


def test_dedupe_share_classes_by_cik_no_collisions_is_a_passthrough():
    capex_by_ticker = {"AAPL": 100.0, "MSFT": 200.0}
    cik_by_ticker = {"AAPL": 1, "MSFT": 2}
    result = capex_watchlist.dedupe_share_classes_by_cik(capex_by_ticker, cik_by_ticker)
    assert result == capex_by_ticker


def test_dedupe_share_classes_by_cik_missing_cik_entry_is_never_collided():
    # A ticker absent from cik_by_ticker (e.g. an unresolved ticker,
    # cik=None) must never collide with another None-CIK ticker.
    capex_by_ticker = {"A": 100.0, "B": 200.0}
    cik_by_ticker = {"A": None, "B": None}
    result = capex_watchlist.dedupe_share_classes_by_cik(capex_by_ticker, cik_by_ticker)
    assert result == {"A": 100.0, "B": 200.0}


# ------------------------------------------------------- resolve_tag_conflicts

def _fact(**overrides):
    fact = {"start": pd.Timestamp("2018-01-01"), "end": pd.Timestamp("2018-12-31"),
           "val": 100.0, "accn": "0001", "fy": 2018, "fp": "FY", "form": "10-K",
           "filed": pd.Timestamp("2019-02-01")}
    fact.update(overrides)
    return fact


def _frame(*facts):
    return pd.DataFrame(list(facts), columns=list(REQUIRED_FACT_FIELDS))


def test_resolve_tag_conflicts_keeps_the_larger_value_on_same_period_overlap():
    # Observed in SEC data (Southern Co, CIK 92122): the SAME
    # (start,end,filed) can be disclosed under two DIFFERENT tags with
    # DIFFERENT values (e.g. PP&E purchases vs PP&E+capitalized
    # interest) -- not the "different tag per era" case the fallback
    # list's docstring anticipates. Preferring CAPEX_TAGS's listed
    # (narrowest-first) order instead would be actively WRONG for Devon
    # Energy (DVN, CIK 1090012) -- its "narrowest" tag,
    # PaymentsToAcquirePropertyPlantAndEquipment, captured a nearly
    # irrelevant subset (~31-55 million USD/yr) while PaymentsToAcquireProductive
    # Assets captured its real oil & gas development spend (~1.9-2.5 billion USD/
    # yr) for the SAME fiscal years -- "narrowest tag" and "most
    # representative tag" are not the same thing. Preferring the LARGER
    # value is more robust: undercounting real capex is the worse
    # failure mode for a study measuring capital intensity, and it still
    # gets Southern Co right (PP&E+capitalized-interest, the larger
    # figure, is also the more economically complete one).
    facts_by_tag = {
        "NarrowTag": _frame(_fact(val=100.0)),
        "WideTag": _frame(_fact(val=150.0)),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("NarrowTag", "WideTag"))
    assert len(result) == 1
    assert result["val"].iloc[0] == 150.0


def test_resolve_tag_conflicts_larger_value_wins_regardless_of_dict_order():
    # Same scenario, but the larger-value tag is inserted FIRST in the
    # dict -- must not affect the outcome (rules out accidental
    # dependence on dict/concat iteration order).
    facts_by_tag = {
        "WideTag": _frame(_fact(val=150.0)),
        "NarrowTag": _frame(_fact(val=100.0)),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("NarrowTag", "WideTag"))
    assert len(result) == 1
    assert result["val"].iloc[0] == 150.0


def test_resolve_tag_conflicts_ties_break_by_tag_priority_for_determinism():
    # Genuinely equal values across two tags for the same period -- the
    # value comparison alone can't decide, so tag_priority's own listed
    # order is the deterministic tie-break (rare in practice, but must
    # not be arbitrary dict-iteration order).
    facts_by_tag = {
        "SecondTag": _frame(_fact(val=100.0, accn="0002")),
        "FirstTag": _frame(_fact(val=100.0, accn="0001")),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("FirstTag", "SecondTag"))
    assert len(result) == 1
    assert result["accn"].iloc[0] == "0001"


def test_resolve_tag_conflicts_prefers_a_stable_tag_over_a_larger_unstable_one():
    # Observed in SEC data (Ventas/VTR, CIK 740260), which shows that
    # the "prefer larger" rule above: "larger" is not always
    # "more representative" either -- VTR's PaymentsToAcquireProductive
    # Assets reported FY2017 capex as 811.875 million USD when first filed
    # (2018-02-09), then RESTATED the SAME period to 1.096 billion USD a year
    # later (2019-02-08) -- a ~35% revision, known BEFORE this test's
    # own conflict row's filed date, so this is not a lookahead check.
    # Meanwhile PaymentsToAcquirePropertyPlantAndEquipment reported the
    # SAME FY2017 figure identically both times. A tag with a KNOWN
    # restatement in ITS OWN history is less trustworthy than one with
    # none, regardless of which one is larger at the row being resolved
    # (rule: use the stable tag's figure).
    facts_by_tag = {
        "StableTag": _frame(
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=100.0, filed=pd.Timestamp("2018-02-09")),
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=100.0, filed=pd.Timestamp("2019-02-08")),  # same value again
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=90.0, filed=pd.Timestamp("2019-02-08")),  # the conflict row
        ),
        "UnstableTag": _frame(
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=500.0, filed=pd.Timestamp("2018-02-09")),
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=700.0, filed=pd.Timestamp("2019-02-08")),  # RESTATED
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=600.0, filed=pd.Timestamp("2019-02-08")),  # the conflict row
        ),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("StableTag", "UnstableTag"))
    fy2018_row = result[result["start"] == pd.Timestamp("2018-01-01")]
    assert len(fy2018_row) == 1
    assert fy2018_row["val"].iloc[0] == 90.0  # StableTag wins despite being smaller


def test_resolve_tag_conflicts_instability_after_the_conflict_date_is_not_lookahead():
    # A tag that LATER gets restated (after this conflict row's own
    # filed date) must still win on size here -- the restatement was not
    # yet knowable at this row's filed date, so using it would be
    # lookahead. Same shape as the test above, but the restatement is
    # moved to AFTER the FY2018 conflict row's filed date.
    facts_by_tag = {
        "SmallTag": _frame(
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=90.0, filed=pd.Timestamp("2019-02-08")),
        ),
        "LargeTagRestatedLater": _frame(
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=500.0, filed=pd.Timestamp("2019-02-08")),
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=600.0, filed=pd.Timestamp("2019-02-08")),  # the conflict row
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=700.0, filed=pd.Timestamp("2020-02-08")),  # restated AFTER
        ),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("SmallTag", "LargeTagRestatedLater"))
    fy2018_row = result[result["start"] == pd.Timestamp("2018-01-01")]
    assert len(fy2018_row) == 1
    assert fy2018_row["val"].iloc[0] == 600.0  # the larger tag still wins here


def test_resolve_tag_conflicts_ignores_an_ancient_restatement_outside_the_lookback_window():
    # Observed in SEC data (Ventas/VTR, CIK 740260):
    # PaymentsToAcquirePropertyPlantAndEquipment ALSO shows a restatement
    # in its history -- but from 2010-2012, the notoriously messy
    # early-XBRL-adoption era, nearly a decade before the conflict the
    # stability rule above targets. An unscoped "any restatement, ever"
    # check would flag BOTH tags unstable, silently falling back to
    # "prefer larger" again and defeating the stability rule. The
    # instability check must only look at periods RECENT relative to the
    # conflict row's own period, not a tag's entire lifetime.
    facts_by_tag = {
        "TagWithOldRestatement": _frame(
            _fact(start=pd.Timestamp("2009-01-01"), end=pd.Timestamp("2009-12-31"),
                 val=10.0, filed=pd.Timestamp("2010-02-19")),
            _fact(start=pd.Timestamp("2009-01-01"), end=pd.Timestamp("2009-12-31"),
                 val=8.0, filed=pd.Timestamp("2012-02-22")),  # ancient restatement
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=100.0, filed=pd.Timestamp("2018-02-09")),
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=100.0, filed=pd.Timestamp("2019-02-08")),  # recent: consistent
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=90.0, filed=pd.Timestamp("2019-02-08")),  # the conflict row
        ),
        "UnstableTag": _frame(
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=500.0, filed=pd.Timestamp("2018-02-09")),
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=700.0, filed=pd.Timestamp("2019-02-08")),  # recent restatement
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=600.0, filed=pd.Timestamp("2019-02-08")),  # the conflict row
        ),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("TagWithOldRestatement", "UnstableTag"))
    fy2018_row = result[result["start"] == pd.Timestamp("2018-01-01")]
    assert len(fy2018_row) == 1
    assert fy2018_row["val"].iloc[0] == 90.0  # the ancient restatement must not count


def test_resolve_tag_conflicts_stability_check_respects_fp_form_not_just_period():
    # Observed in SEC data: a stability check (is_unstable_near) that
    # grouped by (start, end) alone -- unlike the main resolution key's
    # (start, end, filed, fp, form) -- would conflate two GENUINELY
    # DIFFERENT disclosure contexts reported under the SAME tag (e.g. an
    # FY/10-K annual fact and a Q4/10-Q duration-cumulative one for the
    # same period, exactly the routine SEC pattern the resolution key's
    # fp/form fields exist to handle) as if the tag were
    # disagreeing with ITSELF -- a false "restatement" signal. A tag
    # reporting a DIFFERENT fp/form combination for the same period is
    # not evidence of unreliability the way an FY/10-K fact being
    # REVISED to a different FY/10-K value is; they aren't even claiming
    # to measure the same thing.
    facts_by_tag = {
        "SelfConsistentTag": _frame(
            # Same (start,end) reported under BOTH fp/form contexts --
            # routine SEC pattern, not a restatement of either one.
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=999.0, filed=pd.Timestamp("2018-02-09"), fp="Q4", form="10-Q"),
            _fact(start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"),
                 val=100.0, filed=pd.Timestamp("2018-02-09"), fp="FY", form="10-K"),
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=90.0, filed=pd.Timestamp("2019-02-08"), fp="FY", form="10-K"),
        ),
        "SmallerTag": _frame(
            _fact(start=pd.Timestamp("2018-01-01"), end=pd.Timestamp("2018-12-31"),
                 val=50.0, filed=pd.Timestamp("2019-02-08"), fp="FY", form="10-K"),
        ),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("SelfConsistentTag", "SmallerTag"))
    fy2018_row = result[(result["start"] == pd.Timestamp("2018-01-01")) &
                        (result["fp"] == "FY")]
    assert len(fy2018_row) == 1
    # SelfConsistentTag must NOT be penalized as "unstable" merely
    # because IT ALSO reported a Q4/10-Q fact for FY2017 -- it should
    # win on size (90 > 50), not lose because a false instability signal
    # made SmallerTag's smaller value win by default.
    assert fy2018_row["val"].iloc[0] == 90.0


def test_resolve_tag_conflicts_falls_back_to_larger_when_neither_tag_is_unstable():
    # No restatement evidence on either side (Southern Co / Devon
    # Energy's actual shape) -- falls back to preferring the larger
    # value, unaffected by the stability check.
    facts_by_tag = {
        "NarrowTag": _frame(_fact(val=100.0)),
        "WideTag": _frame(_fact(val=150.0)),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("NarrowTag", "WideTag"))
    assert result["val"].iloc[0] == 150.0


def test_resolve_tag_conflicts_preserves_all_non_overlapping_periods():
    # The intended "different tag per era" case: tag A covers an early
    # period, tag B covers a later one -- both must survive untouched.
    facts_by_tag = {
        "EarlyTag": _frame(_fact(start=pd.Timestamp("2015-01-01"),
                                 end=pd.Timestamp("2015-12-31"), val=50.0,
                                 filed=pd.Timestamp("2016-02-01"))),
        "LaterTag": _frame(_fact(start=pd.Timestamp("2018-01-01"),
                                 end=pd.Timestamp("2018-12-31"), val=200.0,
                                 filed=pd.Timestamp("2019-02-01"))),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("EarlyTag", "LaterTag"))
    assert len(result) == 2
    assert set(result["val"]) == {50.0, 200.0}


def test_resolve_tag_conflicts_single_tag_is_a_passthrough():
    facts_by_tag = {"OnlyTag": _frame(_fact(val=77.0))}
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("OnlyTag",))
    assert len(result) == 1
    assert result["val"].iloc[0] == 77.0


def test_resolve_tag_conflicts_empty_input_returns_empty_frame():
    result = capex_watchlist.resolve_tag_conflicts({}, tag_priority=("A", "B"))
    assert len(result) == 0
    assert list(result.columns) == list(REQUIRED_FACT_FIELDS)


def test_resolve_tag_conflicts_never_collides_an_annual_fact_with_a_quarterly_one():
    # Observed in SEC data: SEC XBRL routinely
    # reports the IDENTICAL (start, end, filed) triple twice under one
    # tag -- once fp="FY"/form="10-K" (the real annual fact) and once
    # fp="Q4"/form="10-Q" (a duration-cumulative duplicate that is NOT
    # annual-qualifying). resolve_tag_conflicts's key must never treat
    # these as "the same period, resolve by priority" -- that would
    # silently keep whichever one happened to iterate first, discarding
    # the real annual fact with no error. Callers must pre-filter to
    # annual-only (edgar_point_in_time.annual_facts_only) BEFORE calling
    # this function; the annual and quarterly rows must both survive it
    # untouched here, be filterable afterward.
    facts_by_tag = {
        "OnlyTag": _frame(
            _fact(val=999.0, fp="Q4", form="10-Q"),
            _fact(val=100.0, fp="FY", form="10-K"),
        ),
    }
    result = capex_watchlist.resolve_tag_conflicts(
        facts_by_tag, tag_priority=("OnlyTag",))
    assert len(result) == 2
    assert set(result["val"]) == {999.0, 100.0}
    fy_row = result[result["fp"] == "FY"]
    assert len(fy_row) == 1
    assert fy_row["val"].iloc[0] == 100.0


# ------------------------------------------------------- capex_growth_multiple
# The watchlist thesis targets companies sharply GROWING their AI/
# infrastructure capex while the market penalizes them for it -- not
# "whichever companies had the biggest capex bill in 2019."
# A concrete real-world proof this matters: Oracle's real capex went from
# 1.66 billion USD (FY2019) to 55.7 billion USD (FY2026), a ~33x increase driven by AI/cloud
# buildout -- invisible to a level-as-of-2019 ranking. Ranking by GROWTH
# (recent / base) targets exactly this, while a level floor (below) keeps a
# tiny company's huge percentage swing off a small base from dominating.

def test_capex_growth_multiple_basic():
    assert capex_watchlist.capex_growth_multiple(100.0, 300.0) == 3.0


def test_capex_growth_multiple_none_base_returns_none():
    assert capex_watchlist.capex_growth_multiple(None, 300.0) is None


def test_capex_growth_multiple_none_recent_returns_none():
    assert capex_watchlist.capex_growth_multiple(100.0, None) is None


def test_capex_growth_multiple_zero_base_returns_none():
    assert capex_watchlist.capex_growth_multiple(0.0, 300.0) is None


def test_capex_growth_multiple_negative_base_returns_none():
    assert capex_watchlist.capex_growth_multiple(-50.0, 300.0) is None


def test_capex_growth_multiple_respects_min_base_val_floor():
    # Observed in SEC data (American Electric Power, CIK 4904): SEC shows
    # FY2018 capex as 14.6 million USD in the filing made before the base date, then
    # RESTATES the identical period to exactly zero in the next year's
    # comparative disclosure -- a provable filer inconsistency, not a
    # genuine tiny base. A growth ratio (unlike a level ranking) is
    # extremely sensitive to a spuriously small denominator: dividing the
    # normal recent figure below by that base gives a ~236x "growth" that
    # doesn't reflect any real acceleration. min_base_val excludes bases
    # too small to plausibly be genuine for the kind of company this
    # ranking targets (large_floor-eligible S&P 500 names).
    assert capex_watchlist.capex_growth_multiple(
        14_600_000.0, 3_453_000_000.0, min_base_val=50_000_000.0) is None
    # A base well above the floor is unaffected.
    assert capex_watchlist.capex_growth_multiple(
        131_858_000.0, 2_928_069_000.0, min_base_val=50_000_000.0) == pytest.approx(
        2_928_069_000.0 / 131_858_000.0)


def test_capex_growth_multiple_default_min_base_val_is_zero_backward_compatible():
    # Callers that don't pass min_base_val get a zero floor (any positive
    # base is usable).
    assert capex_watchlist.capex_growth_multiple(1.0, 300.0) == 300.0


# ------------------------------------------------------- rank_by_capex_growth

def test_rank_by_capex_growth_orders_by_growth_not_level():
    # C has the highest LEVEL but the lowest GROWTH; ranking must follow
    # growth, not level, once both clear the floor.
    level_by_ticker = {"A": 100.0, "B": 200.0, "C": 900.0}
    growth_by_ticker = {"A": 10.0, "B": 5.0, "C": 1.1}
    result = capex_watchlist.rank_by_capex_growth(
        level_by_ticker, growth_by_ticker, level_floor_rank=10, n=10)
    assert [row["ticker"] for row in result] == ["A", "B", "C"]
    assert [row["capex_growth_multiple"] for row in result] == [10.0, 5.0, 1.1]


def test_rank_by_capex_growth_excludes_tickers_below_the_level_floor():
    # TINY has the highest growth by far but a trivial absolute level --
    # the floor (top 2 by level) must exclude it even though its growth
    # would otherwise win outright.
    level_by_ticker = {"BIG1": 1000.0, "BIG2": 900.0, "TINY": 5.0}
    growth_by_ticker = {"BIG1": 2.0, "BIG2": 1.5, "TINY": 100.0}
    result = capex_watchlist.rank_by_capex_growth(
        level_by_ticker, growth_by_ticker, level_floor_rank=2, n=10)
    assert [row["ticker"] for row in result] == ["BIG1", "BIG2"]


def test_rank_by_capex_growth_drops_none_growth_values():
    level_by_ticker = {"A": 100.0, "B": 200.0}
    growth_by_ticker = {"A": 3.0, "B": None}
    result = capex_watchlist.rank_by_capex_growth(
        level_by_ticker, growth_by_ticker, level_floor_rank=10, n=10)
    assert [row["ticker"] for row in result] == ["A"]


def test_rank_by_capex_growth_drops_tickers_missing_from_level_dict():
    # A ticker present in growth but absent from level (shouldn't happen
    # in practice, but must not crash) is excluded, not treated as
    # automatically eligible.
    level_by_ticker = {"A": 100.0}
    growth_by_ticker = {"A": 3.0, "B": 5.0}
    result = capex_watchlist.rank_by_capex_growth(
        level_by_ticker, growth_by_ticker, level_floor_rank=10, n=10)
    assert [row["ticker"] for row in result] == ["A"]


def test_rank_by_capex_growth_ties_break_alphabetically():
    level_by_ticker = {"ZEBRA": 100.0, "ALPHA": 100.0, "MID": 100.0}
    growth_by_ticker = {"ZEBRA": 2.0, "ALPHA": 2.0, "MID": 2.0}
    result = capex_watchlist.rank_by_capex_growth(
        level_by_ticker, growth_by_ticker, level_floor_rank=10, n=10)
    assert [row["ticker"] for row in result] == ["ALPHA", "MID", "ZEBRA"]


def test_rank_by_capex_growth_respects_n_cap():
    level_by_ticker = {"A": 100.0, "B": 100.0, "C": 100.0}
    growth_by_ticker = {"A": 3.0, "B": 5.0, "C": 4.0}
    result = capex_watchlist.rank_by_capex_growth(
        level_by_ticker, growth_by_ticker, level_floor_rank=10, n=2)
    assert [row["ticker"] for row in result] == ["B", "C"]


def test_rank_by_capex_growth_empty_input_returns_empty_list():
    assert capex_watchlist.rank_by_capex_growth({}, {}, level_floor_rank=10, n=10) == []


def test_resolve_tag_conflicts_result_is_sorted_by_filed():
    facts_by_tag = {
        "TagA": _frame(_fact(filed=pd.Timestamp("2020-01-01"), val=1.0,
                             start=pd.Timestamp("2019-01-01"), end=pd.Timestamp("2019-12-31")),
                      _fact(filed=pd.Timestamp("2018-01-01"), val=2.0,
                             start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-12-31"))),
    }
    result = capex_watchlist.resolve_tag_conflicts(facts_by_tag, tag_priority=("TagA",))
    assert result["filed"].is_monotonic_increasing
