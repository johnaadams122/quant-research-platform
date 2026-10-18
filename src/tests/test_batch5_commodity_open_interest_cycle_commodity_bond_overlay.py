"""Tests for src/mining/batch5/commodity_open_interest_cycle_commodity_bond_overlay.py.

Contract shell, the open-interest cycle signal (count-based growth, sector
averaging, 12-month mean, expanding-median state, blackout carry), the per-window
leg plan, and the curve composition (two documented ``long_overlay`` builder
calls, one per leg, summed). The CFTC fetcher / availability map is tested in
test_mining_cftc_data.py; the bespoke driver in
test_run_commodity_oi_cycle_overlay_abi1.py.

Everything uses SYNTHETIC production-shaped frames and small synthetic
open-interest series; the only real-data tests (marked ``slow``) read the cached
CFTC open-interest counts (signal only: no price, no return is read).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import cftc_data as cd  # noqa: E402
from mining import long_overlay as lo  # noqa: E402
from mining.batch5 import commodity_open_interest_cycle_commodity_bond_overlay as mod  # noqa: E402
from mining.batch5 import treasury_month_end_index_demand_overlay as treasury  # noqa: E402
from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns  # noqa: E402

_FAMILY = "commodity_open_interest_cycle_commodity_bond_overlay"
_CATALOG_ENTRY_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == _FAMILY)


_DOLLAR_FIGURE = re.compile(r"\$\s?\d")
_YYYYMMDD_INT = re.compile(r"(?<![\d.\-_])(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?![\d\-_])")
# a bare run of 8 to 17 digits
_NUM_SHAPE = re.compile(r"(?<!\d\.)\b\d{8,17}\b")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _catalog_entry() -> dict:
    with open(_CATALOG_ENTRY_PATH, "r", encoding="utf-8") as fh:
        return _catalog_entry_from(fh)


# ------------------------------------------------------------------ contract

def test_module_constants_match_the_candidate_contract():
    assert mod.FAMILY_ID == _FAMILY
    assert mod.SHAPE == "pooled_single_asset"        # contract LABEL only, see the docstring
    assert mod.TRADED == ["DBC", "TLT"]
    assert mod.SIGNAL == ["^IRX"]                    # financing-rate series, never traded
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_signal_fn)


def test_parameters_are_tied_to_the_constants_they_describe():
    p = mod.PARAMETERS
    assert p["overlay_notional_fraction"] == mod.OVERLAY_NOTIONAL == 0.25
    assert p["cmi_months"] == mod.CMI_MONTHS == 12
    assert p["state_history_start"] == mod.STATE_HISTORY_START == "1986-01-02"       # decides the median baseline
    assert p["min_median_months"] == mod.MIN_MEDIAN_MONTHS == 36
    assert p["overlay_legs"] == list(mod.TRADED)
    assert p["state_to_leg"] == {"HIGH": "DBC", "LOW": "TLT"} == mod.LEG_BY_STATE
    assert p["benchmark"] == "SPY" == mod.BENCHMARK
    assert p["availability_lag_calendar_days"] == cd.AVAILABILITY_LAG_DAYS == 6
    assert p["scored_merge_same_leg"] is mod.SCORED_MERGE_SAME_LEG is False          # the ONE switch, monthly-resized
    assert "every calendar month is its own window" in p["merge"]
    assert p["contracts_by_sector"] == cd.contracts_by_sector()
    assert p["financing_rate"].startswith("^IRX") and "no margin spread" in p["financing_rate"]
    assert mod.OVERLAY_NOTIONAL <= lo.MAX_OVERLAY_NOTIONAL


def test_the_catalog_entry_is_complete_honest_and_states_every_disclosure():
    from mining import catalog as catalog_mod
    entry = _catalog_entry()
    for field in catalog_mod.REQUIRED_FIELDS:
        assert field in entry, field
    assert entry["family_id"] == _FAMILY and entry["status"] == "eligible"
    assert entry["account_types"] == ["taxable"]                # the overlay is margin-financed
    assert entry["tickers"] == ["DBC", "TLT", "SPY"]
    assert entry["timeframe"] == "daily" and entry["added_date"] == "2026-09-24"
    text = (entry["source"] + " " + entry["mechanism"] + " " + entry["distinctness_note"]).lower()
    for phrase in (
            "hong", "yogo", "not verified never-revised", "2017-03-28", "2018-09-18", "2019-10-15",
            "publication blackout", "carries the prior state", "2013-10-31", "2018-12-31",
            "2025-10-31", "holdout",
            # every deviation from the paper
            "contract counts", "not dollar open interest", "18 contracts", "median split",
            "not a regression", "tlt", "10-year", "one pooled trial",
            # composition drift and the near-static post-2008 state
            "composition drift", "1998", "static long-tlt",
            # flattering assumptions
            "no margin spread", "energy-heavy", "equal-weight",
            # control arm and merge convention
            "control arm", "always-tlt", "merge", "fixed share count",
            # the tax side effect of merging
            "tax walk", "365 days or less", "zero long-term rate", "realizes its gain untaxed",
            "risk-basis rule", "overlay-only excess account"):
        assert phrase in text, phrase
    for phrase in (
            # all FIVE deviations from the paper, each pinned
            "contract counts, not dollar open interest", "contract-level growth averaged within each sector",
            "aggregate-then-growth", "18 contracts in four sectors, not about 30 commodities",
            "a median split, not a regression", "tlt (a 20-plus-year fund), not the 10-year note",
            # the 2018-19 blackout and the tie month
            "2019-02-28", "2018-12-24", "exact tie", "2019-11-29", "accelerated plan",
            # the scored convention and its one switch
            "monthly-resized", "one switch", "every calendar month is its own window", "merged sensitivity",
            # measured composition figures
            "241 of the 476", "84, 168 and 259 days"):
        assert phrase in text, phrase
    assert "never_revised" not in text.replace("not verified never-revised", "")
    assert not _DOLLAR_FIGURE.search(json.dumps(entry))


def test_driver_discovery_validates_this_module_contract_shape():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == _FAMILY and spec["shape"] == "pooled_single_asset"
    assert spec["traded"] == ["DBC", "TLT"] and spec["signal"] == ["^IRX"] and spec["benchmark"] == "SPY"
    assert spec["parameters"] == mod.PARAMETERS


def test_generic_dispatch_call_shape_raises_instead_of_scoring_anything():
    idx = pd.bdate_range("2020-01-02", periods=30)
    frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
    with pytest.raises(RuntimeError, match="bespoke driver"):
        mod.make_signal_fn("DBC", idx, {"DBC": frame})


def test_real_generic_dispatch_builder_fails_closed_for_this_family():
    import run_abi1_mining as generic
    import run_batch5_mining as driver
    idx = pd.bdate_range("2020-01-02", periods=60)
    frame = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0}, index=idx)
    frames = {s: frame for s in ("DBC", "TLT", "SPY", "^IRX")}
    plan = {"spec": driver.validate_module_contract(mod, repo_root=_ROOT), "module": mod,
            "effective": ["DBC", "TLT"], "costs": {s: {"cost_bps": 1.0} for s in ("DBC", "TLT")}}
    with pytest.raises(RuntimeError, match="bespoke driver"):
        generic._build_curve(plan, frames)


def test_the_module_docstring_states_the_pre_registration_and_every_amendment():
    doc = mod.__doc__
    for phrase in (
            "PRE-REGISTERED", "one pooled trial", "NOT verified never-revised", "2017-03-28",
            "Hong", "Yogo", "contract COUNTS", "18 contracts", "median split", "TLT", "10-year",
            "publication blackout", "carry the prior state", "composition drift",
            "always-TLT", "control", "merge", "fixed share count", "energy-heavy", "no margin spread",
            "realizes its gain untaxed", "held beyond a year", "flatters the strategy through the tax walk",
            "risk-basis rule", "prefix", "not tuned", "expanding median", "current month",
            "Dispatch safety", "holdout"):
        assert phrase in doc, phrase
    for stale in ("never_revised\"", "point_in_time: true"):
        assert stale not in doc
    flat = _norm(doc)
    for phrase in (
            "FIVE declared ways", "contract counts, not dollar open interest",
            "contract-level growth averaged within each sector", "aggregate-then-growth",
            "18 contracts, not about 30 commodities", "a median split, not a regression",
            "TLT (a 20-plus-year fund), not the 10-year note",
            "2019-02-28", "2018-12-24", "EXACT TIE", "SCORED_MERGE_SAME_LEG", "241 of the 476 CMI values (50.6 percent)",
            "84, 168 and 259 days", "in the scored convention no window is held beyond a year",
            "three days of margin in a normal week and none in a holiday-shifted week", "immaterial to the state",
            "accelerated plan", "state-level prefix-invariance test"):
        assert phrase in flat, phrase


# ------------------------------------------------------------------ calendar

def test_month_end_sessions_are_the_last_scheduled_session_of_every_month():
    ends = mod.month_end_sessions(pd.Timestamp("2018-03-05"), pd.Timestamp("2018-05-20"))
    assert list(ends) == [pd.Timestamp("2018-03-29"),      # Good Friday the 30th is closed
                          pd.Timestamp("2018-04-30"), pd.Timestamp("2018-05-31")]
    assert mod.month_end_sessions(pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-02"))[0] == pd.Timestamp("2026-05-29")
    assert mod.month_end_sessions(pd.Timestamp("2027-12-05"), pd.Timestamp("2027-12-06"))[0] == pd.Timestamp("2027-12-31")
    assert mod.month_end_sessions(pd.Timestamp("2012-10-01"), pd.Timestamp("2012-10-02"))[0] == pd.Timestamp("2012-10-31")
    assert mod.month_end_sessions(pd.Timestamp("1986-01-02"), pd.Timestamp("1986-01-03"))[0] == pd.Timestamp("1986-01-31")


def test_month_end_sessions_equal_the_treasury_families_scheduled_exits_over_the_whole_history():
    """Whole-set cross-check against the sibling's independently tested schedule."""
    mine = list(mod.month_end_sessions(pd.Timestamp("2003-01-01"), pd.Timestamp("2027-12-31")))
    theirs = [exit_ for _, exit_ in treasury.month_end_windows(pd.Timestamp("2003-01-01"), pd.Timestamp("2027-12-31"))]
    assert mine == theirs and len(mine) == 25 * 12


def test_month_end_sessions_take_dates_only():
    import inspect
    assert list(inspect.signature(mod.month_end_sessions).parameters) == ["start", "end"]


# ------------------------------------------------------------ signal building

def _months(n, start="2010-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def test_aggregate_growth_is_the_equal_weight_of_sector_means_of_contract_log_growth():
    codes = ["088691", "084691", "002602", "057642"]     # metals x2, agriculture, livestock
    months = _months(3)
    oi = pd.DataFrame({
        "088691": [100.0, 110.0, 121.0],                 # +ln 1.1, +ln 1.1
        "084691": [100.0, 100.0, 200.0],                 # 0, +ln 2
        "002602": [50.0, 100.0, 100.0],                  # +ln 2, 0
        "057642": [10.0, 10.0, 9.0],                     # 0, ln 0.9
    }, index=months)[codes]
    table = mod.aggregate_growth(oi)
    first = table.iloc[1]
    metals = (np.log(1.1) + 0.0) / 2.0
    assert first["metals"] == pytest.approx(metals)
    assert first["agriculture"] == pytest.approx(np.log(2.0))
    assert first["livestock"] == pytest.approx(0.0)
    assert np.isnan(first["energy"])                     # no energy contract present: sector missing, not zero
    assert first["aggregate"] == pytest.approx((metals + np.log(2.0) + 0.0) / 3.0)   # three sectors, equal weight
    second = table.iloc[2]
    assert second["metals"] == pytest.approx((np.log(1.1) + np.log(2.0)) / 2.0)
    assert second["aggregate"] == pytest.approx(
        (second["metals"] + 0.0 + np.log(0.9)) / 3.0)
    assert np.isnan(table.iloc[0]["aggregate"])          # no growth without a prior month


def test_a_sector_with_many_contracts_does_not_outweigh_a_sector_with_one():
    months = _months(2)
    oi = pd.DataFrame({"088691": [100.0, 100.0 * np.e], "084691": [100.0, 100.0 * np.e],
                       "085692": [100.0, 100.0 * np.e], "057642": [100.0, 100.0]}, index=months)
    table = mod.aggregate_growth(oi)
    assert table.iloc[1]["aggregate"] == pytest.approx((1.0 + 0.0) / 2.0)   # not (3*1 + 0)/4


def test_a_contract_missing_in_a_month_is_excluded_not_zero_filled():
    months = _months(3)
    oi = pd.DataFrame({"088691": [100.0, 110.0, 121.0], "084691": [np.nan, np.nan, 50.0]}, index=months)
    table = mod.aggregate_growth(oi)
    assert table.iloc[1]["metals"] == pytest.approx(np.log(1.1))            # silver excluded, not averaged as 0
    assert table.iloc[2]["metals"] == pytest.approx(np.log(1.1))            # silver has no prior month yet
    assert mod.aggregate_growth(oi.assign(**{"084691": [50.0, 50.0, 50.0]})).iloc[2]["metals"] == pytest.approx(
        np.log(1.1) / 2.0)


def test_cmi_is_the_mean_of_the_last_twelve_aggregate_values_and_needs_all_twelve():
    aggregate = pd.Series(np.arange(1.0, 21.0) / 100.0, index=_months(20))
    cmi = mod.cmi_from_aggregate(aggregate)
    assert cmi.iloc[:11].isna().all()
    assert cmi.iloc[11] == pytest.approx(np.mean(np.arange(1.0, 13.0) / 100.0))
    assert cmi.iloc[19] == pytest.approx(np.mean(np.arange(9.0, 21.0) / 100.0))
    with_hole = aggregate.copy()
    with_hole.iloc[15] = np.nan
    holed = mod.cmi_from_aggregate(with_hole)
    assert holed.iloc[15:27].isna().all() if len(holed) > 26 else holed.iloc[15:].isna().all()


def test_cmi_equals_the_twelve_month_log_change_divided_by_twelve_when_the_contracts_are_constant():
    """The 12-month mean of monthly log growth telescopes."""
    months = _months(30)
    rng = np.random.RandomState(3)
    levels = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.05, (30, 4)), axis=0))
    oi = pd.DataFrame(levels, index=months, columns=["088691", "084691", "002602", "057642"])
    cmi = mod.cmi_from_aggregate(mod.aggregate_growth(oi)["aggregate"])
    logs = np.log(oi)
    change = logs.iloc[29] - logs.iloc[17]
    expected = (change[["088691", "084691"]].mean() + change["002602"] + change["057642"]) / 3.0 / 12.0
    assert cmi.iloc[29] == pytest.approx(expected)


def _cmi(values, start="1990-01-31"):
    return pd.Series(np.asarray(values, dtype=float), index=_months(len(values), start))


def test_state_is_high_only_when_cmi_strictly_exceeds_the_expanding_median_including_the_current_month():
    values = list(np.linspace(0.0, 0.35, 36)) + [0.34, 0.10, 0.40, 0.17, 0.17]
    cmi = _cmi(values)
    table = mod.classify_states(cmi, pd.Series(False, index=cmi.index))
    assert table["state"].iloc[:35].isna().all()                    # min 36 monthly observations
    assert table["median"].iloc[35] == pytest.approx(np.median(values[:36]))
    for k in range(35, len(values)):
        median_k = np.median(values[:k + 1])                        # the current month is INCLUDED
        assert table["median"].iloc[k] == pytest.approx(median_k)
        expected = "HIGH" if values[k] > median_k else "LOW"
        assert table["state"].iloc[k] == expected, k
    assert table["state"].iloc[35] == "HIGH"                         # 0.35 is the max: above its own median
    assert table["state"].iloc[37] == "LOW" and table["state"].iloc[38] == "HIGH"
    # a tie with the median is LOW (strictly greater is required)
    tie = _cmi([0.0] * 36)
    assert (mod.classify_states(tie, pd.Series(False, index=tie.index))["state"].iloc[35:] == "LOW").all()


def test_the_first_state_needs_exactly_thirty_six_observations():
    cmi = _cmi(np.arange(40) / 100.0)
    table = mod.classify_states(cmi, pd.Series(False, index=cmi.index))
    assert table["state"].first_valid_index() == cmi.index[35]


def test_a_blackout_month_end_carries_the_prior_state_and_is_left_out_of_the_median():
    values = list(np.sin(np.arange(60) / 3.0))
    cmi = _cmi(values)
    flags = pd.Series(False, index=cmi.index)
    flags.iloc[44] = True
    flags.iloc[45] = True                                          # two consecutive blackout month-ends
    table = mod.classify_states(cmi, flags)
    assert table["carried"].iloc[44] and table["carried"].iloc[45] and not table["carried"].iloc[43]
    assert table["state"].iloc[44] == table["state"].iloc[43] and table["state"].iloc[45] == table["state"].iloc[43]
    assert table["blackout"].iloc[44] and pd.isna(table["raw_state"].iloc[44])
    # the blackout months are not in the median history
    history = [v for k, v in enumerate(values[:47]) if k not in (44, 45)]
    assert table["median"].iloc[46] == pytest.approx(np.median(history))
    # ... so the state after them is judged against that median
    assert table["state"].iloc[46] == ("HIGH" if values[46] > np.median(history) else "LOW")


def test_a_blackout_before_any_state_exists_leaves_the_state_undefined():
    cmi = _cmi(np.arange(40) / 100.0)
    flags = pd.Series(False, index=cmi.index)
    flags.iloc[10] = True
    table = mod.classify_states(cmi, flags)
    assert pd.isna(table["state"].iloc[10]) and not table["carried"].iloc[10]


def test_a_missing_cmi_outside_a_blackout_is_not_forward_filled():
    cmi = _cmi(np.sin(np.arange(50) / 3.0))
    cmi.iloc[42] = np.nan
    table = mod.classify_states(cmi, pd.Series(False, index=cmi.index))
    assert pd.isna(table["state"].iloc[42]) and not table["carried"].iloc[42]


# ------------------------------- the full pipeline against an independent reference

_CODE_START = {c.code: "1986-01-07" for c in cd.CONTRACTS}
_CODE_START.update({"002602": "1998-01-06", "005602": "1998-01-06", "001602": "1998-01-06",
                    "054642": "1996-04-02", "023651": "1990-04-10", "085692": "1989-07-25",
                    "111659": "2006-02-14"})


def _synthetic_series(end="2016-12-27", seed=7):
    out = {}
    rng = np.random.RandomState(seed)
    for k, contract in enumerate(cd.CONTRACTS):
        dates = pd.date_range(_CODE_START[contract.code], end, freq="7D")
        t = np.arange(len(dates))
        period = 52 * (3 + (k % 5))
        level = 1000.0 * np.exp(0.7 * np.sin(2 * np.pi * t / period + k) + 0.004 * t * (1 if k % 2 else -1)
                                + rng.normal(0.0, 0.02, len(t)))
        out[contract.code] = pd.Series(level, index=dates, name=contract.code)
    return out


def _reference_state_table(series_by_code, decision_dates):
    """A deliberately naive, loop-based re-implementation of the documented rule."""
    import bisect
    sectors = cd.contracts_by_sector()
    listed = {}
    for code, s in series_by_code.items():
        avail = cd.availability_dates(s.index)                     # once per contract, not per decision
        listed[code] = (list(s.index), list(avail), list(s.to_numpy()))
    per_month = []
    for date in decision_dates:
        row = {}
        for code, (asofs, avails, levels) in listed.items():
            # walk back from the latest report dated on or before the decision until one is usable
            k = bisect.bisect_right(asofs, date) - 1
            while k >= 0 and avails[k] > date:
                k -= 1
            row[code] = None if k < 0 else float(levels[k])
        per_month.append(row)
    aggregate = [np.nan]
    for m in range(1, len(decision_dates)):
        sector_means = []
        for codes in sectors.values():
            growths = [np.log(per_month[m][c] / per_month[m - 1][c]) for c in codes
                       if per_month[m][c] is not None and per_month[m - 1][c] is not None]
            if growths:
                sector_means.append(sum(growths) / len(growths))
        aggregate.append(sum(sector_means) / len(sector_means) if sector_means else np.nan)
    cmi = [np.nan] * len(aggregate)
    for m in range(11, len(aggregate)):
        window = aggregate[m - 11:m + 1]
        if not any(np.isnan(w) for w in window):
            cmi[m] = sum(window) / 12.0
    states, history, last = [], [], None
    for m, date in enumerate(decision_dates):
        if cd.blackout_at(date) is not None:
            states.append(last)
            continue
        if np.isnan(cmi[m]):
            states.append(None)
            last = None
            continue
        history.append(cmi[m])
        if len(history) < 36:
            states.append(None)
            last = None
            continue
        last = "HIGH" if cmi[m] > float(np.median(history)) else "LOW"
        states.append(last)
    return cmi, states


@pytest.fixture(scope="module")
def synthetic_oi():
    return _synthetic_series()


@pytest.fixture(scope="module")
def synthetic_decisions():
    return mod.month_end_sessions(pd.Timestamp("1986-01-02"), pd.Timestamp("2016-12-15"))


@pytest.fixture(scope="module")
def synthetic_table(synthetic_oi, synthetic_decisions):
    return mod.signal_state_table(synthetic_oi, synthetic_decisions)


def test_the_full_pipeline_equals_a_naive_reference_on_eighteen_contracts_with_staggered_starts(
        synthetic_oi, synthetic_decisions, synthetic_table):
    cmi, states = _reference_state_table(synthetic_oi, list(synthetic_decisions))
    np.testing.assert_allclose(synthetic_table["cmi"].to_numpy(), np.array(cmi), rtol=1e-12, equal_nan=True)
    got = [None if pd.isna(s) else s for s in synthetic_table["state"]]
    assert got == states
    assert sum(s is not None for s in got) > 200
    assert {"HIGH", "LOW"} <= set(s for s in got if s)


def test_the_pipeline_table_marks_the_2013_blackout_and_carries_the_state(synthetic_table):
    row = synthetic_table.loc[pd.Timestamp("2013-10-31")]
    before = synthetic_table.loc[pd.Timestamp("2013-09-30")]
    assert bool(row["blackout"]) and bool(row["carried"]) and row["state"] == before["state"]
    assert not bool(synthetic_table.loc[pd.Timestamp("2013-11-29")]["blackout"])
    assert synthetic_table["blackout"].sum() == 1          # 2013 only (the series ends 2016)


def test_the_expanding_median_history_excludes_the_blackout_month(synthetic_table):
    valid = synthetic_table["cmi"].where(~synthetic_table["blackout"]).dropna()
    expected = valid.expanding(min_periods=mod.MIN_MEDIAN_MONTHS).median()
    later = synthetic_table.loc[pd.Timestamp("2014-01-31"):]
    np.testing.assert_allclose(later["median"].to_numpy(), expected.reindex(later.index).to_numpy(), rtol=1e-12)


def test_the_state_at_a_month_end_never_reads_a_report_that_was_not_yet_usable(synthetic_oi, synthetic_decisions):
    """Look-ahead probe: replace every report a decision date cannot yet use with garbage; the state up
    to that date must not move. Garbage is tried BOTH huge and tiny, so a leaked report flips the
    state whichever side of its median the honest state sits on."""
    base = mod.signal_state_table(synthetic_oi, synthetic_decisions)["state"]
    moved_after = False
    for cutoff in ["2003-06-30", "2010-03-31", "2013-10-31", "2013-11-29", "2015-08-31"]:
        cutoff = pd.Timestamp(cutoff)
        for garbage in (1.0e9, 1.0e-3):
            perturbed = {}
            for code, s in synthetic_oi.items():
                t = s.copy()
                unusable = cd.availability_dates(t.index) > cutoff
                t[unusable] = garbage * (1 + np.arange(int(unusable.sum())))
                perturbed[code] = t
            again = mod.signal_state_table(perturbed, synthetic_decisions)["state"]
            upto = base.index <= cutoff
            assert base[upto].equals(again[upto]), (cutoff.date(), garbage)
            moved_after = moved_after or not base.equals(again)
    # ... and the probe has power: the same garbage DOES move the states after the cutoff
    assert moved_after


def test_the_look_ahead_probe_has_power_against_a_shorter_availability_lag(monkeypatch, synthetic_oi,
                                                                          synthetic_decisions):
    """With the lag forced to 0 the same perturbation must change a state at or
    before the cutoff (proof that the test above would notice a leak)."""
    base_lag = cd.AVAILABILITY_LAG_DAYS
    cutoff = pd.Timestamp("2010-03-31")
    truthful = mod.signal_state_table(synthetic_oi, synthetic_decisions)["state"]
    monkeypatch.setattr(cd, "AVAILABILITY_LAG_DAYS", 0)
    leaky_base = mod.signal_state_table(synthetic_oi, synthetic_decisions)["state"]
    assert not truthful.loc[:cutoff].equals(leaky_base.loc[:cutoff])        # the lag genuinely matters here
    monkeypatch.setattr(cd, "AVAILABILITY_LAG_DAYS", base_lag)


def test_the_require_fresh_cache_check_refuses_a_cache_older_than_the_last_decision():
    snapshot = cd.CftcSnapshot(series={}, manifest={}, latest_asof=pd.Timestamp("2026-09-15"),
                               point_in_time_status=cd.POINT_IN_TIME_STATUS)
    mod.require_fresh_cache(snapshot, [pd.Timestamp("2026-08-31")])            # needs the report as of 2026-08-25
    with pytest.raises(mod.SignalDataError) as caught:                         # needs the report as of 2026-09-22
        mod.require_fresh_cache(snapshot, [pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-30")])
    assert caught.value.reason_code == "cftc_cache_stale"
    assert "2026-09-30" in str(caught.value) and "2026-09-15" in str(caught.value)
    # a blackout decision date needs no new report: its state is carried
    mod.require_fresh_cache(cd.CftcSnapshot(series={}, manifest={}, latest_asof=pd.Timestamp("2025-09-23"),
                                            point_in_time_status=cd.POINT_IN_TIME_STATUS),
                            [pd.Timestamp("2025-10-31"), pd.Timestamp("2025-11-28")])


# ---------------------- freshness of the open-interest cache, per contract

def _fresh_snapshot(last_asof, *, overrides=None):
    """A snapshot whose 18 contracts all end on ``last_asof`` except the ones named in ``overrides``."""
    lasts = {code: pd.Timestamp(last_asof) for code in cd.CONTRACT_CODES}
    lasts.update({code: pd.Timestamp(value) for code, value in (overrides or {}).items()})
    series = {code: pd.Series([1.0, 2.0], index=pd.DatetimeIndex([last - pd.Timedelta(days=7), last]), name=code)
              for code, last in lasts.items()}
    return cd.CftcSnapshot(series=series, manifest={}, latest_asof=max(lasts.values()),
                           point_in_time_status=cd.POINT_IN_TIME_STATUS)


def test_the_require_fresh_cache_check_refuses_one_lagging_contract_when_seventeen_are_current():
    """The newest contract alone must not vouch for the cache: a contract that stopped early would be carried
    forward with zero growth at every later month-end."""
    decision = [pd.Timestamp("2026-09-30")]                                   # needs the report as of 2026-09-22
    mod.require_fresh_cache(_fresh_snapshot("2026-09-22"), decision)
    with pytest.raises(mod.SignalDataError) as caught:
        mod.require_fresh_cache(_fresh_snapshot("2026-09-22", overrides={"067651": "2026-09-08"}), decision)
    assert caught.value.reason_code == "cftc_contract_stale"
    message = str(caught.value)
    for text in ("067651", "2026-09-08", "2026-09-22", "2026-09-30"):
        assert text in message, text
    with pytest.raises(mod.SignalDataError) as caught:                        # one day short is short
        mod.require_fresh_cache(_fresh_snapshot("2026-09-22", overrides={"075651": "2026-09-21"}), decision)
    assert caught.value.reason_code == "cftc_contract_stale" and "075651" in str(caught.value)
    mod.require_fresh_cache(_fresh_snapshot("2026-09-22", overrides={"075651": "2026-09-22"}), decision)


def test_the_require_fresh_cache_names_every_lagging_contract():
    snapshot = _fresh_snapshot("2026-09-22", overrides={"002602": "2026-09-08", "088691": "2026-08-25"})
    with pytest.raises(mod.SignalDataError) as caught:
        mod.require_fresh_cache(snapshot, [pd.Timestamp("2026-09-30")])
    assert "002602" in str(caught.value) and "088691" in str(caught.value)


def test_the_stale_cache_boundary_at_a_month_end_operating_point():
    """The 2026-09-30 decision needs the report as of 2026-09-22 (usable from 2026-09-28): a cache whose newest
    report is exactly that one is accepted, one week older is refused, and the message names the required date."""
    decisions = [pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-30")]
    mod.require_fresh_cache(_fresh_snapshot("2026-09-22"), decisions)                      # exactly on the required report
    mod.require_fresh_cache(_fresh_snapshot("2026-09-29"), decisions)                      # a later report is fine
    with pytest.raises(mod.SignalDataError) as caught:
        mod.require_fresh_cache(_fresh_snapshot("2026-09-15"), decisions)
    assert caught.value.reason_code == "cftc_cache_stale"
    message = str(caught.value)
    for text in ("2026-09-30", "needs the report as of 2026-09-22", "usable from 2026-09-28", "2026-09-15"):
        assert text in message, text
    with pytest.raises(mod.SignalDataError):                                                # one day short of the required report
        mod.require_fresh_cache(_fresh_snapshot("2026-09-21"), decisions)


# ------------- a stale read at a scored month-end is refused (mid-history holes and stopped contracts)

def _one_report_series(asof):
    return pd.Series([100.0], index=pd.DatetimeIndex([pd.Timestamp(asof)]), name="088691")


def test_the_stale_read_guard_bound_is_fourteen_days_and_starts_in_2005():
    assert mod.MAX_REPORT_AGE_DAYS == 14 and mod.STALE_READ_CHECK_START == pd.Timestamp("2005-01-01")


def test_a_read_exactly_fourteen_days_old_is_accepted_and_fifteen_is_refused():
    decision = [pd.Timestamp("2010-03-31")]
    mod.require_recent_reads({"088691": _one_report_series("2010-03-17")}, decision)          # age 14
    with pytest.raises(mod.SignalDataError) as caught:
        mod.require_recent_reads({"088691": _one_report_series("2010-03-16")}, decision)      # age 15
    assert caught.value.reason_code == "cftc_report_stale_at_month_end"
    for text in ("088691", "2010-03-31", "2010-03-16", "15 days"):
        assert text in str(caught.value), text


def test_a_mid_history_gap_in_one_contract_is_refused_naming_the_contract_and_the_month_end(synthetic_oi):
    decisions = mod.month_end_sessions(pd.Timestamp("2005-01-01"), pd.Timestamp("2016-11-30"))
    mod.require_recent_reads(synthetic_oi, decisions)                                          # the clean history passes
    holed = dict(synthetic_oi)
    gold = synthetic_oi["088691"]
    holed["088691"] = gold.drop([pd.Timestamp("2010-03-09"), pd.Timestamp("2010-03-16"), pd.Timestamp("2010-03-23")])
    with pytest.raises(mod.SignalDataError) as caught:
        mod.require_recent_reads(holed, decisions)
    assert caught.value.reason_code == "cftc_report_stale_at_month_end"
    for text in ("088691", "2010-03-31", "2010-03-02"):
        assert text in str(caught.value), text


def test_the_stale_read_guard_ignores_history_before_2005_blackout_month_ends_and_contracts_not_yet_reporting():
    stale = {"088691": _one_report_series("2004-11-02")}
    mod.require_recent_reads(stale, [pd.Timestamp("2004-12-31")])                              # before the check starts
    with pytest.raises(mod.SignalDataError):
        mod.require_recent_reads(stale, [pd.Timestamp("2005-01-31")])                          # the first checked month-end
    # a blackout month-end reads a report weeks old BY DESIGN (state carried): never refused for it
    assert cd.blackout_at(pd.Timestamp("2013-10-31")) is not None
    mod.require_recent_reads({"088691": _one_report_series("2013-09-24")}, [pd.Timestamp("2013-10-31")])
    # a contract with no report yet (RBOB lists in 2006) has no read to be stale
    mod.require_recent_reads({"111659": _one_report_series("2006-02-14")}, [pd.Timestamp("2005-06-30")])


@pytest.mark.slow
def test_on_the_real_cache_the_stale_read_guard_does_not_false_trip_and_real_reads_stay_within_the_bound():
    cache = os.path.join(_ROOT, "data", "mining", "cftc_oi")
    if not os.path.isfile(os.path.join(cache, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(cache)
    decisions = [d for d in mod.month_end_sessions(pd.Timestamp("1986-01-02"), snapshot.latest_asof)
                 if d <= snapshot.latest_asof]
    mod.require_recent_reads(snapshot.series, decisions)                                       # real history clears the bound
    scored = pd.DatetimeIndex([d for d in decisions if d >= mod.STALE_READ_CHECK_START and cd.blackout_at(d) is None])
    _, used = cd.month_end_open_interest(snapshot.series, scored)
    ages = (pd.DataFrame({c: scored - pd.DatetimeIndex(used[c]) for c in used.columns}, index=scored)
            .apply(lambda col: col.dt.days))
    assert ages.max().max() < mod.MAX_REPORT_AGE_DAYS                                          # with real slack


# -------------------------- defensive branches (coverage gaps)

@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_aggregate_growth_refuses_a_zero_or_negative_open_interest(bad):
    index = pd.DatetimeIndex(["2010-01-29", "2010-02-26"])
    oi = pd.DataFrame({code: [10.0, 12.0] for code in cd.CONTRACT_CODES}, index=index)
    oi.loc[index[1], "088691"] = bad
    with pytest.raises(ValueError, match="positive"):
        mod.aggregate_growth(oi)


def test_a_data_calendar_with_no_window_at_all_scores_the_benchmark_itself():
    idx, spy, legs, irx = _inputs(start="2011-05-02", end="2011-05-20")          # no month-end entry inside the data
    out = mod.build_curve(spy, legs, irx, _pattern_table(), leg_cost_bps={"DBC": 5.0, "TLT": 4.0},
                          spy_entry_cost_bps=1.0)
    assert out["window_info"]["windows"] == [] and out["window_info"]["monthly"] == []
    assert len(out["trades_df"]) == 0 and not out["overlay_mask"].any()
    assert (out["overlay_daily"].to_numpy() == 0.0).all()
    core = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=1.0).reindex(idx)
    np.testing.assert_allclose(out["strategy_daily"].to_numpy(), core.to_numpy(), rtol=0, atol=1e-15)


# ------------- disclosures: the five holiday-week decisions, the early-timing tie sensitivity, the sixth limitation

_HOLIDAY_WEEK_AND_SURVIVOR_PHRASES = (
    "2009-11-30, 2011-02-28, 2015-11-30, 2020-11-30 and 2022-02-28",
    "about 30 minutes before the close on the decision day", "not verified against a cftc notice",
    "sixth disclosed limitation", "a limitation, not a deviation from the paper", "the deviations stay five",
    "present-day survivor set", "no membership rule applied at each historical date")


def test_the_module_docstring_and_the_catalog_entry_carry_the_holiday_week_and_survivor_set_disclosures():
    doc = _norm(mod.__doc__).lower()
    source = _norm(_catalog_entry()["source"]).lower()
    for phrase in _HOLIDAY_WEEK_AND_SURVIVOR_PHRASES:
        assert phrase in source, ("catalog", phrase)
    for phrase in ("2009-11-30, 2011-02-28, 2015-11-30, 2020-11-30 and 2022-02-28", "about 30 minutes before the close on the decision day",
                   "not verified against a cftc notice", "sensitive to the unverified pre-2000 release timing",
                   "an extra 7 or 14 days of lag", "sixth disclosed limitation", "a limitation, not a deviation from the paper",
                   "present-day survivor set", "no membership rule applied at each historical date",
                   "require_recent_reads", "cftc_contract_stale", "older than 14 days"):
        assert phrase in doc, ("docstring", phrase)
    assert "differs in five declared ways" in doc          # the deviations stay FIVE: the survivor set is a limitation


@pytest.mark.slow
def test_on_the_real_cache_the_five_holiday_week_decisions_and_the_early_timing_tie_are_as_disclosed(monkeypatch):
    """The holiday-week and early-timing disclosures, reproduced on the cached vintage (signal only: counts and dates, no price):
    exactly five scored month-end decisions read a report from a US-federal-holiday week on its plain D+6 date, no
    state depends on those five reports, and an extra lag on the PRE-2000 reports alone flips only the 2019-11-29
    exact tie within the scored span."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(_REAL_CACHE)
    decisions = mod.month_end_sessions(pd.Timestamp("1986-01-02"), pd.Timestamp("2026-08-31"))
    base = mod.signal_state_table(snapshot.series, decisions)
    _, used = cd.month_end_open_interest(snapshot.series, decisions)
    holidays = USFederalHolidayCalendar().holidays(start="1985-01-01", end="2027-12-31")
    scored_from = pd.Timestamp("2006-02-01")
    exactly_six, holiday_week, holiday_asofs = [], [], set()
    for decision in decisions:
        if decision < scored_from or cd.blackout_at(decision) is not None:
            continue
        for asof in pd.DatetimeIndex(used.loc[decision].dropna().unique()):
            if (decision - asof).days == 6:
                exactly_six.append(decision)
                week = pd.date_range(asof - pd.Timedelta(days=asof.dayofweek), periods=5)      # Monday to Friday
                if any(day in holidays for day in week):
                    holiday_week.append(decision)
                    holiday_asofs.add(asof)
    assert len(exactly_six) == 30
    assert [d.strftime("%Y-%m-%d") for d in holiday_week] == [
        "2009-11-30", "2011-02-28", "2015-11-30", "2020-11-30", "2022-02-28"]      # the other 25 are ordinary weeks
    real = cd.availability_dates

    def moved(states):
        return [d.strftime("%Y-%m-%d") for d in base.index[(base["state"].fillna("none") != states.fillna("none")).to_numpy()]
                if d >= scored_from]

    def withheld_a_day(asof_index):
        asof = pd.DatetimeIndex(asof_index)
        late = np.array([a in holiday_asofs for a in asof])
        return pd.DatetimeIndex(np.where(late, real(asof) + pd.Timedelta(days=1), real(asof)))
    monkeypatch.setattr(cd, "availability_dates", withheld_a_day)
    assert moved(mod.signal_state_table(snapshot.series, decisions)["state"]) == []      # no state depends on them
    for extra in (7, 14):
        def early_lag(asof_index, extra=extra):
            asof = pd.DatetimeIndex(asof_index)
            early = np.array([a < pd.Timestamp("2000-01-01") for a in asof])
            return pd.DatetimeIndex(np.where(early, real(asof) + pd.Timedelta(days=extra), real(asof)))
        monkeypatch.setattr(cd, "availability_dates", early_lag)
        assert moved(mod.signal_state_table(snapshot.series, decisions)["state"]) == ["2019-11-29"], extra


# ------------------------------------------------------------ the leg plan

def _table_from_states(states, start="2008-01-31"):
    """A state table on the scheduled month-ends from ``start``; months after ``states`` are LOW."""
    dates = mod.month_end_sessions(pd.Timestamp(start), pd.Timestamp(start) + pd.offsets.MonthEnd(len(states) + 14))
    padded = list(states) + ["LOW"] * (len(dates) - len(states))
    return pd.DataFrame({"state": padded}, index=dates)


def _idx(start, end):
    return treasury.exchange_sessions(pd.Timestamp(start), pd.Timestamp(end))


def test_plan_windows_maps_states_to_legs_and_merges_consecutive_same_leg_months():
    table = _table_from_states(["HIGH", "HIGH", "LOW", "LOW", "LOW", "HIGH", "LOW", "LOW"], "2010-03-31")
    idx = _idx("2010-03-15", "2010-12-31")
    plan = mod.plan_windows(idx, table)
    legs = [(w[2], w[0].strftime("%Y-%m-%d"), w[1].strftime("%Y-%m-%d")) for w in plan["windows"]]
    assert legs[:4] == [("DBC", "2010-03-31", "2010-05-28"), ("TLT", "2010-05-28", "2010-08-31"),
                        ("DBC", "2010-08-31", "2010-09-30"), ("TLT", "2010-09-30", "2010-12-31")]
    assert len(plan["monthly"]) > len(plan["windows"])
    assert plan["windows_by_leg"]["DBC"][0] == (pd.Timestamp("2010-03-31"), pd.Timestamp("2010-05-28"))


def test_plan_windows_blocks_tile_the_span_with_shared_boundaries_and_only_switch_at_a_state_change():
    states = ["HIGH", "LOW", "LOW", "HIGH", "HIGH", "HIGH", "LOW", "HIGH"]
    table = _table_from_states(states, "2010-03-31")
    idx = _idx("2010-03-25", "2011-06-30")
    plan = mod.plan_windows(idx, table)
    blocks = plan["windows"]
    for a, b in zip(blocks, blocks[1:]):
        assert a[1] == b[0] and a[2] != b[2]                          # neighbours share the boundary close, differ in leg
    monthly = plan["monthly"]
    assert [m[2] for m in monthly[:len(states)]] == [mod.LEG_BY_STATE[s] for s in states]
    assert blocks[0][0] == monthly[0][0] and blocks[-1][1] == monthly[-1][1]


def test_plan_windows_unmerged_keeps_one_window_per_month():
    table = _table_from_states(["LOW"] * 8, "2010-03-31")
    idx = _idx("2010-03-25", "2010-12-31")
    merged = mod.plan_windows(idx, table)
    unmerged = mod.plan_windows(idx, table, merge_same_leg=False)
    assert len(merged["windows"]) == 1 and len(unmerged["windows"]) == len(unmerged["monthly"]) > 6
    assert unmerged["windows"][0][1] == unmerged["windows"][1][0]


def test_plan_windows_force_leg_overrides_every_month_and_keeps_the_same_first_entry():
    table = _table_from_states(["HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW"], "2010-03-31")
    idx = _idx("2010-03-25", "2010-12-31")
    strategy = mod.plan_windows(idx, table)
    control = mod.plan_windows(idx, table, force_leg="TLT")
    assert [w[2] for w in control["monthly"]] == ["TLT"] * len(control["monthly"])
    assert control["monthly"][0][0] == strategy["monthly"][0][0]
    assert len(control["windows"]) == 1
    with pytest.raises(ValueError, match="force_leg"):
        mod.plan_windows(idx, table, force_leg="SPY")


def test_plan_windows_skips_months_whose_entry_precedes_the_data_and_keeps_an_entry_on_the_first_date():
    table = _table_from_states(["HIGH"] * 12, "2010-03-31")
    on_entry = mod.plan_windows(_idx("2010-03-31", "2010-12-31"), table)
    assert on_entry["monthly"][0][0] == pd.Timestamp("2010-03-31") and on_entry["skipped_before_data"] == 0
    later = mod.plan_windows(_idx("2010-04-01", "2010-12-31"), table)
    assert later["monthly"][0][0] == pd.Timestamp("2010-04-30")       # March's month-end precedes the data: no window


def test_plan_windows_reports_the_end_of_data_state_exactly():
    """Windows are month-long, so the data can end before an entry (the PREVIOUS window is then
    truncated), exactly on an entry (that window opens nothing; the previous one completes), or
    inside a window (truncated)."""
    table = _table_from_states(["HIGH", "LOW", "LOW", "HIGH", "LOW", "HIGH", "HIGH", "LOW", "HIGH", "LOW"], "2010-03-31")
    full = _idx("2010-03-01", "2011-03-31")
    entry = pd.Timestamp("2010-09-30")
    following_exit = pd.Timestamp("2010-10-29")
    e = full.get_loc(entry)
    cases = {                      # last data session -> (truncated_final, entered_on_last_bar, not_started)
        full[e - 1]: (True, False, 1),
        entry: (False, True, 0),
        full[e + 1]: (True, False, 1),                      # inside the window; the next month-end has not happened
        full[e + 10]: (True, False, 1),
        full[full.get_loc(following_exit) - 1]: (True, False, 1),
        following_exit: (False, True, 0),
    }
    for last, expected in cases.items():
        plan = mod.plan_windows(full[full <= last], table)
        assert (plan["truncated_final"], plan["entered_on_last_bar"], plan["not_started"]) == expected, last


def test_a_window_entered_on_the_last_bar_is_dropped_before_merging_so_the_prior_block_closes_there():
    table = _table_from_states(["HIGH"] * 10, "2010-03-31")
    full = _idx("2010-03-01", "2011-03-31")
    entry = pd.Timestamp("2010-08-31")
    plan = mod.plan_windows(full[:full.get_loc(entry) + 1], table)
    assert plan["entered_on_last_bar"] and plan["windows"][-1][1] == entry and not plan["truncated_final"]


def test_plan_windows_refuses_a_missing_entry_or_exit_session_and_an_undefined_state():
    table = _table_from_states(["HIGH"] * 10, "2010-03-31")
    idx = _idx("2010-03-25", "2010-12-31")
    with pytest.raises(ValueError, match="2010-05"):
        mod.plan_windows(idx.difference(pd.DatetimeIndex(["2010-05-28"])), table)
    holed = table.copy()
    holed.loc[pd.Timestamp("2010-06-30"), "state"] = np.nan
    with pytest.raises(mod.SignalDataError) as caught:
        mod.plan_windows(idx, holed)
    assert caught.value.reason_code == "state_unavailable" and "2010-06-30" in str(caught.value)
    with pytest.raises(mod.SignalDataError) as caught:
        mod.plan_windows(idx, table.iloc[:3])
    assert caught.value.reason_code == "state_table_does_not_cover_the_data"


def test_plan_windows_refuses_a_missing_entry_session_of_the_first_in_data_window():
    """The middle-of-the-data test above trips the EXIT guard of the earlier window first; this one reaches the
    entry guard itself (the very first scheduled entry is missing from the data)."""
    table = _table_from_states(["HIGH"] * 10, "2010-03-31")
    idx = _idx("2010-03-25", "2010-12-31")
    with pytest.raises(ValueError, match="needs its entry session 2010-03-31"):
        mod.plan_windows(idx.difference(pd.DatetimeIndex(["2010-03-31"])), table)


def test_the_scored_window_convention_is_monthly_resized_behind_one_switch_and_both_are_pinned():
    import inspect
    assert mod.SCORED_MERGE_SAME_LEG is False
    assert inspect.signature(mod.build_curve).parameters["merge_same_leg"].default is mod.SCORED_MERGE_SAME_LEG
    idx, spy, legs, irx = _inputs()
    table = _table_from_states(["LOW"] * 16, "2011-05-31")
    kwargs = dict(leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    scored = mod.build_curve(spy, legs, irx, table, **kwargs)
    monthly = mod.build_curve(spy, legs, irx, table, merge_same_leg=False, **kwargs)
    merged = mod.build_curve(spy, legs, irx, table, merge_same_leg=True, **kwargs)
    # the default IS the monthly-resized account, bit for bit, and it is a genuinely different account from the merged one
    np.testing.assert_array_equal(scored["strategy_daily"].to_numpy(), monthly["strategy_daily"].to_numpy())
    pd.testing.assert_frame_equal(scored["trades_df"], monthly["trades_df"])
    assert len(scored["trades_df"]) == len(mod.plan_windows(idx, table, merge_same_leg=False)["windows"]) > 10
    assert len(merged["trades_df"]) == 1
    assert not np.array_equal(scored["strategy_daily"].to_numpy(), merged["strategy_daily"].to_numpy())
    # the monthly windows are each held a month or so; no scored window is held beyond a year
    held = (scored["trades_df"]["exit_date"] - scored["trades_df"]["entry_date"]).dt.days
    assert held.max() <= 33 and (merged["trades_df"]["exit_date"] - merged["trades_df"]["entry_date"]).dt.days.iloc[0] > 365


def test_the_next_open_sensitivity_follows_the_scored_window_convention():
    idx, spy, legs, irx = _inputs()
    table = _table_from_states(["LOW"] * 16, "2011-05-31")
    sens = mod.build_next_open_sensitivity(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    assert len(sens["trades_df"]) == len(mod.plan_windows(idx, table, merge_same_leg=mod.SCORED_MERGE_SAME_LEG)["windows"])


# ----------------------------------------------------------- curve composition

_COSTS = {"DBC": 4.0, "TLT": 2.0}


def _ohlc(close, index):
    close = np.asarray(close, dtype=float)
    prev = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": prev, "High": np.maximum(prev, close), "Low": np.minimum(prev, close),
                         "Close": close}, index=index)


def _walk(index, seed, vol, drift=0.0002, start=100.0):
    r = np.random.RandomState(seed).normal(drift, vol, len(index))
    r[0] = 0.0
    return start * np.cumprod(1.0 + r)


def _inputs(start="2011-05-02", end="2013-08-30", seed=5):
    idx = _idx(start, end)
    legs = {"DBC": _ohlc(_walk(idx, 11, 0.012), idx), "TLT": _ohlc(_walk(idx, 12, 0.008), idx)}
    spy = _ohlc(_walk(idx, 14, 0.01, start=120.0), idx)
    irx = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": np.linspace(0.5, 2.5, len(idx))}, index=idx)
    return idx, spy, legs, irx


_PATTERN = ["HIGH", "HIGH", "LOW", "LOW", "LOW", "HIGH", "LOW", "HIGH", "HIGH", "LOW", "LOW", "HIGH",
            "LOW", "LOW", "HIGH", "HIGH", "LOW", "HIGH", "LOW", "LOW", "HIGH", "LOW", "HIGH", "HIGH",
            "LOW", "LOW", "HIGH", "LOW"]


def _pattern_table(start="2011-05-31"):
    return _table_from_states(_PATTERN, start)


def _reference_account(idx, spy, legs, irx, blocks, *, costs, spy_entry_cost, x=0.25):
    """Independent, plain-loop implementation of the documented account rules for
    back-to-back single-leg blocks: shares fixed at the entry close and sized on the
    account equity there, financing on calendar days at the causal bill rate, the
    round-trip cost charged on the exit session."""
    core = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=spy_entry_cost).reindex(idx).to_numpy()
    rate = lo.causal_financing_rate(irx["Close"], idx).to_numpy()
    starts = {pd.Timestamp(b[0]): b for b in blocks}
    daily = np.zeros(len(idx))
    equity = 1.0
    pos = None
    for t, day in enumerate(idx):
        overlay = 0.0
        if pos is not None:
            leg = pos["leg"]
            close = legs[leg]["Close"].to_numpy()
            reference = pos["entry_px"] if t == pos["first"] else close[t - 1]
            pnl = pos["shares"] * (close[t] - reference)
            days = 1.0 if t == 0 else float((idx[t] - idx[t - 1]).days)
            fin = pos["principal"] * rate[t] * days / 365.0
            overlay = pnl - fin
            if day == pos["exit"]:
                overlay -= pos["cost"]
                pos = None
        daily[t] = core[t] + overlay / equity
        equity = equity * (1.0 + daily[t])
        if day in starts:
            entry, exit_, leg = starts[day]
            price = legs[leg]["Close"].to_numpy()[t]
            pos = {"leg": leg, "exit": exit_, "first": t + 1, "entry_px": price, "shares": x * equity / price,
                   "principal": x * equity, "cost": equity * x * costs[leg] / 1e4}
    return daily


def test_the_composed_account_equals_an_independent_reference_with_back_to_back_different_leg_blocks():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    plan = mod.plan_windows(idx, table)
    blocks = [w for w in plan["windows"] if w[1] <= idx[-1]]
    assert len(blocks) >= 8 and {b[2] for b in blocks} == {"DBC", "TLT"}
    assert any(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))            # back-to-back
    assert not plan["truncated_final"]
    curve = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3, merge_same_leg=True)
    reference = _reference_account(idx, spy, legs, irx, plan["windows"], costs=_COSTS, spy_entry_cost=0.3)
    np.testing.assert_allclose(curve["strategy_daily"].to_numpy(), reference, rtol=0, atol=1e-13)
    assert len(curve["trades_df"]) == len(plan["windows"] if not plan["entered_on_last_bar"] else plan["windows"])


def test_a_single_leg_plan_equals_the_documented_builder_bit_for_bit():
    idx, spy, legs, irx = _inputs()
    table = _table_from_states(["LOW"] * 20, "2011-05-31")
    composed = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3,
                               merge_same_leg=True)
    plan = mod.plan_windows(idx, table)
    financing = lo.causal_financing_rate(irx["Close"], idx)
    direct = lo.build_long_overlay_curve(
        spy, {"TLT": legs["TLT"]}, [(w[0], w[1]) for w in plan["windows"]], leg_cost_bps={"TLT": 2.0},
        spy_entry_cost_bps=0.3, financing_rate_annual=financing, overlay_notional=0.25)
    np.testing.assert_array_equal(composed["strategy_daily"].to_numpy(), direct["strategy_daily"].to_numpy())
    pd.testing.assert_frame_equal(composed["trades_df"], direct["trades_df"])


def test_the_curve_has_the_abi1_shape_and_declares_a_benchmark_plus_overlay_account():
    idx, spy, legs, irx = _inputs()
    curve = mod.build_curve(spy, legs, irx, _pattern_table(), leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    assert {"strategy_daily", "in_market", "trades_df", "account_kind", "core_daily", "overlay_daily",
            "overlay_mask", "financing_rate", "window_info", "leg_by_trade"} <= set(curve)
    assert curve["account_kind"] == "benchmark_plus_overlay"
    assert list(curve["trades_df"].columns) == list(lo.TRADES_COLUMNS)
    assert curve["strategy_daily"].index.equals(idx) and curve["in_market"].all()
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy(),
                                  (curve["core_daily"] + curve["overlay_daily"]).to_numpy())
    assert len(curve["leg_by_trade"]) == len(curve["trades_df"])
    assert (curve["trades_df"]["entry_date"].diff().dropna() > pd.Timedelta(0)).all()


def test_the_account_equals_the_benchmark_series_before_the_first_entry():
    idx, spy, legs, irx = _inputs()
    curve = mod.build_curve(spy, legs, irx, _pattern_table(), leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    off = ~curve["overlay_mask"].to_numpy()
    assert off.sum() >= 10 and off[0]
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy()[off], curve["core_daily"].to_numpy()[off])


def test_trade_sizes_are_the_true_combined_account_equity_at_each_entry_close():
    idx, spy, legs, irx = _inputs()
    curve = mod.build_curve(spy, legs, irx, _pattern_table(), leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    growth = np.cumprod(1.0 + curve["strategy_daily"].to_numpy())
    for _, row in curve["trades_df"].iterrows():
        assert row["size"] == pytest.approx(growth[idx.get_loc(row["entry_date"])], rel=1e-13)


def test_each_window_is_driven_only_by_its_own_leg():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    base = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    plan = mod.plan_windows(idx, table)
    dbc = [w for w in plan["windows"] if w[2] == "DBC" and w[1] <= idx[-1]][1]
    other = {**legs, "TLT": _ohlc(_walk(idx, 99, 0.03), idx)}                       # a wildly different TLT
    changed = mod.build_curve(spy, other, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    inside = (idx > dbc[0]) & (idx <= dbc[1])
    np.testing.assert_array_equal(base["strategy_daily"].to_numpy()[inside], changed["strategy_daily"].to_numpy()[inside])
    assert not np.array_equal(base["strategy_daily"].to_numpy(), changed["strategy_daily"].to_numpy())
    tlt = [w for w in plan["windows"] if w[2] == "TLT" and w[1] <= idx[-1]][1]
    other_d = {**legs, "DBC": _ohlc(_walk(idx, 98, 0.03), idx)}
    changed_d = mod.build_curve(spy, other_d, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    inside_t = (idx > tlt[0]) & (idx <= tlt[1])
    np.testing.assert_array_equal(base["strategy_daily"].to_numpy()[inside_t],
                                  changed_d["strategy_daily"].to_numpy()[inside_t])


def test_the_leg_costs_come_from_the_right_leg():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    a = mod.build_curve(spy, legs, irx, table, leg_cost_bps={"DBC": 4.0, "TLT": 2.0}, spy_entry_cost_bps=0.3)
    b = mod.build_curve(spy, legs, irx, table, leg_cost_bps={"DBC": 4.0, "TLT": 9.0}, spy_entry_cost_bps=0.3)
    ta, tb = a["trades_df"], b["trades_df"]
    for i, leg in enumerate(a["leg_by_trade"]):
        if leg == "DBC":
            assert ta["cost_ret"].iloc[i] == tb["cost_ret"].iloc[i]
        else:
            assert tb["cost_ret"].iloc[i] > ta["cost_ret"].iloc[i]
            assert ta["cost_ret"].iloc[i] == pytest.approx(0.25 * 2.0 / 1e4)
    assert a["leg_by_trade"] == b["leg_by_trade"]


def test_consecutive_same_leg_months_are_one_trade_with_one_round_trip_and_unmerged_pays_each_month():
    idx, spy, legs, irx = _inputs()
    table = _table_from_states(["LOW"] * 16, "2011-05-31")
    merged = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3,
                             merge_same_leg=True)
    unmerged = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3,
                               merge_same_leg=False)
    assert len(merged["trades_df"]) == 1
    assert len(unmerged["trades_df"]) == len(mod.plan_windows(idx, table, merge_same_leg=False)["windows"]) > 10
    assert unmerged["trades_df"]["cost_ret"].sum() > merged["trades_df"]["cost_ret"].sum() * 5
    assert merged["trades_df"]["cost_ret"].iloc[0] == pytest.approx(0.25 * 2.0 / 1e4)
    # both hold the same leg every month; only the sizing/cost bookkeeping differs
    assert set(unmerged["leg_by_trade"]) == {"TLT"}


def test_the_unmerged_account_equals_an_independent_reference_with_monthly_resizing():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    plan = mod.plan_windows(idx, table, merge_same_leg=False)
    blocks = [w for w in plan["windows"] if w[1] <= idx[-1]]
    unmerged = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)   # the DEFAULT
    reference = _reference_account(idx, spy, legs, irx, plan["windows"], costs=_COSTS, spy_entry_cost=0.3)
    np.testing.assert_allclose(unmerged["strategy_daily"].to_numpy(), reference, rtol=0, atol=1e-13)


def test_the_state_at_a_month_end_cannot_move_overlay_pnl_on_or_before_that_month_end():
    """Leg-switch bug class: flip ONE month's state (as if the CFTC report
    that decides it had come in different); no price return on or before that month-end may
    move (the decision is taken AT the close; only the cost of executing a switch there differs)."""
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    base = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    decision_dates = [d for d in table.index if idx[0] <= d < idx[-1] - pd.Timedelta(days=45)][3:14]
    for date in decision_dates:
        flipped = table.copy()
        flipped.loc[date, "state"] = "HIGH" if table.loc[date, "state"] == "LOW" else "LOW"
        again = mod.build_curve(spy, legs, irx, flipped, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
        before = idx < date
        np.testing.assert_array_equal(base["strategy_daily"].to_numpy()[before],
                                      again["strategy_daily"].to_numpy()[before], err_msg=str(date.date()))
        # ON the month-end itself only the round-trip cost of a switch executed at that close can differ
        # (the builder books it on the exit session); no price return moves
        gap = abs(base["strategy_daily"].loc[date] - again["strategy_daily"].loc[date])
        assert gap <= 0.25 * max(_COSTS.values()) / 1e4 * 1.5, (date.date(), gap)      # x 1.5: equity growth since entry
        assert not np.array_equal(base["strategy_daily"].to_numpy(), again["strategy_daily"].to_numpy())


def test_the_curve_is_prefix_invariant_with_cutoffs_one_to_four_sessions_after_each_switch():
    """PRICES ONLY are cut (the state table stays fixed), so this catches a price look-ahead in the curve
    build; a leg chosen from the next month's state is caught by the state-flip test and the state-level
    prefix test instead."""
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    zero = {"DBC": 0.0, "TLT": 0.0}
    full = mod.build_curve(spy, legs, irx, table, leg_cost_bps=zero, spy_entry_cost_bps=0.0)
    plan = mod.plan_windows(idx, table)
    switches = [w[0] for w in plan["windows"][1:-1]]
    cutoffs = []
    for entry in switches:
        e = idx.get_loc(entry)
        cutoffs += [e + 1, e + 2, e + 3, e + 4]
    checked = 0
    for c in cutoffs:
        cut = idx[:c + 1]
        part = mod.build_curve(spy.loc[cut], {s: f.loc[cut] for s, f in legs.items()}, irx.loc[cut], table,
                               leg_cost_bps=zero, spy_entry_cost_bps=0.0)
        np.testing.assert_array_equal(part["strategy_daily"].to_numpy(), full["strategy_daily"].to_numpy()[:c + 1])
        checked += 1
    assert checked == len(cutoffs) >= 20


def _a_state_flip_moves_an_earlier_return(idx, spy, legs, irx, table, dates) -> bool:
    """True when flipping ONE month-end's state moves any account return strictly BEFORE that month-end
    (the decision is taken at the close: nothing earlier may depend on it)."""
    base = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    for date in dates:
        flipped = table.copy()
        flipped.loc[date, "state"] = "HIGH" if table.loc[date, "state"] == "LOW" else "LOW"
        again = mod.build_curve(spy, legs, irx, flipped, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
        before = idx < date
        if not np.array_equal(base["strategy_daily"].to_numpy()[before], again["strategy_daily"].to_numpy()[before]):
            return True
    return False


def test_the_state_flip_check_has_power_against_a_leg_chosen_from_the_next_months_state(monkeypatch):
    """The state-flip (cannot-move) check is the one that sees a leg chosen from the NEXT month's state; the
    price-only prefix test cannot (its state table is fixed while only prices are cut). The same helper that
    passes on the honest build must fail on a build that reads the next month's state."""
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    dates = [d for d in table.index if idx[0] <= d < idx[-1] - pd.Timedelta(days=45)][3:14]
    assert _a_state_flip_moves_an_earlier_return(idx, spy, legs, irx, table, dates) is False
    original = mod.plan_windows

    def leaky(index, tbl, **kwargs):
        shifted = tbl.copy()
        shifted["state"] = tbl["state"].shift(-1).fillna(tbl["state"])       # the leg of month m from month m+1's state
        return original(index, shifted, **kwargs)
    monkeypatch.setattr(mod, "plan_windows", leaky)
    assert _a_state_flip_moves_an_earlier_return(idx, spy, legs, irx, table, dates) is True


_PREFIX_CUTOFFS = ("2003-06-30", "2009-12-31", "2013-10-31", "2013-11-29", "2014-06-30", "2016-06-30")


def _state_prefix_mismatches(oi, decisions, whole, cutoffs=_PREFIX_CUTOFFS):
    """Cut every contract's history at each cutoff, recompute the state table on the decisions up to it and list
    every (cutoff, column) whose earlier values differ from the full-history table's."""
    found = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        cut_decisions = decisions[decisions <= cutoff]
        part = mod.signal_state_table({code: s[s.index <= cutoff] for code, s in oi.items()}, cut_decisions)
        ref = whole.loc[cut_decisions]
        for column in ("cmi", "median"):
            if not np.allclose(part[column].to_numpy(dtype=float), ref[column].to_numpy(dtype=float), rtol=1e-12,
                               equal_nan=True):
                found.append((cutoff.date(), column))
        for column in ("state", "raw_state", "carried", "blackout"):
            if part[column].astype(object).where(part[column].notna(), None).tolist() !=                     ref[column].astype(object).where(ref[column].notna(), None).tolist():
                found.append((cutoff.date(), column))
    return found


def test_the_state_table_is_prefix_invariant_when_the_cftc_data_is_truncated(synthetic_oi, synthetic_decisions,
                                                                             synthetic_table):
    """State-level prefix invariance: cut every contract's history at a cutoff, recompute the state table on the
    decisions up to it, and every earlier CMI, median, state and carry flag must equal the full-history table's
    (the state table is where a future report would leak; the price-only prefix test cannot see it)."""
    assert _state_prefix_mismatches(synthetic_oi, synthetic_decisions, synthetic_table) == []


def test_the_state_level_prefix_check_has_power_against_a_median_over_the_whole_history(monkeypatch, synthetic_oi,
                                                                                      synthetic_decisions,
                                                                                      synthetic_table):
    """Power check: an expanding median replaced by a whole-history median reads the future; the same check must
    report mismatches."""
    original = mod.classify_states

    def whole_history(cmi, blackout):
        table = original(cmi, blackout)
        table["median"] = cmi.where(~blackout.reindex(cmi.index).fillna(False).astype(bool)).median()
        return table
    monkeypatch.setattr(mod, "classify_states", whole_history)
    leaky_whole = mod.signal_state_table(synthetic_oi, synthetic_decisions)
    assert _state_prefix_mismatches(synthetic_oi, synthetic_decisions, leaky_whole) != []


def test_build_curve_requires_exactly_the_registered_legs_and_refuses_calendar_gaps():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    with pytest.raises(ValueError, match="DBC, TLT"):                     # frames guard alone (costs are complete)
        mod.build_curve(spy, {"DBC": legs["DBC"]}, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    with pytest.raises(ValueError, match="DBC, TLT"):
        mod.build_curve(spy, {**legs, "SHY": legs["DBC"]}, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    with pytest.raises(ValueError, match="round-trip cost"):               # cost-mapping guard alone
        mod.build_curve(spy, legs, irx, table, leg_cost_bps={"DBC": 1.0}, spy_entry_cost_bps=0.0)
    gone = idx[100]
    drop = lambda f: f.drop(gone)  # noqa: E731
    with pytest.raises(lo.CalendarCoverageError) as caught:
        mod.build_curve(drop(spy), {s: drop(f) for s, f in legs.items()}, drop(irx), table,
                        leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    assert caught.value.reason_code == "missing_exchange_session"
    short = {**legs, "DBC": legs["DBC"].iloc[:-1]}
    with pytest.raises(lo.CalendarCoverageError) as caught:
        mod.build_curve(spy, short, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    assert caught.value.reason_code == "unequal_trailing_coverage"


def test_the_account_starts_on_the_later_listing_leg_and_spy_may_be_longer():
    idx, spy, legs, irx = _inputs()
    late = {"DBC": legs["DBC"].iloc[150:], "TLT": legs["TLT"]}
    curve = mod.build_curve(spy, late, irx, _pattern_table(), leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    assert curve["strategy_daily"].index[0] == idx[150]
    evaluator = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=0.3).reindex(curve["strategy_daily"].index)
    off = ~curve["overlay_mask"].to_numpy()
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy()[off], evaluator.to_numpy()[off])


def test_an_irx_history_that_starts_late_fails_closed_on_a_financed_session():
    idx, spy, legs, irx = _inputs()
    with pytest.raises(ValueError, match="financing"):
        mod.build_curve(spy, legs, irx.iloc[400:], _pattern_table(), leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)


def test_a_negative_bill_print_does_not_break_the_builder():
    idx, spy, legs, irx = _inputs()
    neg = irx.copy()
    neg["Close"] = np.linspace(-0.1, 0.4, len(idx))
    assert np.isfinite(mod.build_curve(spy, legs, neg, _pattern_table(), leg_cost_bps=_COSTS,
                                       spy_entry_cost_bps=0.3)["strategy_daily"].to_numpy()).all()


def test_the_truncated_final_block_is_closed_at_the_last_close_and_flagged():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    plan = mod.plan_windows(idx, table)
    entry = [w for w in plan["windows"] if w[1] <= idx[-1]][-1][0]
    e = idx.get_loc(entry)
    cut = idx[:e + 3]
    curve = mod.build_curve(spy.loc[cut], {s: f.loc[cut] for s, f in legs.items()}, irx.loc[cut], table,
                            leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    assert curve["window_info"]["truncated_final"] is True
    assert curve["trades_df"]["reason"].iloc[-1] == "end_of_data"
    assert curve["trades_df"]["exit_date"].iloc[-1] == cut[-1]


def test_next_open_sensitivity_is_a_separate_report_only_curve_with_the_same_exits():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    scored = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    sens = mod.build_next_open_sensitivity(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    assert len(sens["trades_df"]) == len(scored["trades_df"])
    assert not np.array_equal(sens["strategy_daily"].to_numpy(), scored["strategy_daily"].to_numpy())
    assert (sens["trades_df"]["exit_date"].to_numpy() == scored["trades_df"]["exit_date"].to_numpy()).all()
    assert (sens["trades_df"]["entry_date"] > scored["trades_df"]["entry_date"]).all()
    import inspect
    assert "entry_at_next_open" not in inspect.signature(mod.build_curve).parameters


# --------------------------------------------------------------- control arm

def test_the_always_tlt_control_is_one_long_tlt_position_from_the_same_first_entry():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    strategy = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    control = mod.build_control_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    assert set(control["leg_by_trade"]) == {"TLT"} and len(control["trades_df"]) == 1
    assert control["trades_df"]["entry_date"].iloc[0] == strategy["trades_df"]["entry_date"].iloc[0]
    assert control["account_kind"] == "benchmark_plus_overlay"
    monthly = mod.build_control_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3,
                                      merge_same_leg=False)
    assert set(monthly["leg_by_trade"]) == {"TLT"} and len(monthly["trades_df"]) > 20
    # the control never looks at the state: a wildly different state table gives the identical control
    flipped = table.copy()
    flipped["state"] = flipped["state"].map({"HIGH": "LOW", "LOW": "HIGH"})
    again = mod.build_control_curve(spy, legs, irx, flipped, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    np.testing.assert_array_equal(control["strategy_daily"].to_numpy(), again["strategy_daily"].to_numpy())


def test_the_control_uses_the_same_costs_financing_and_core_as_the_strategy():
    idx, spy, legs, irx = _inputs()
    table = _pattern_table()
    strategy = mod.build_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    control = mod.build_control_curve(spy, legs, irx, table, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.3)
    np.testing.assert_array_equal(strategy["core_daily"].to_numpy(), control["core_daily"].to_numpy())
    np.testing.assert_array_equal(strategy["financing_rate"].to_numpy(), control["financing_rate"].to_numpy())
    assert control["trades_df"]["cost_ret"].iloc[0] == pytest.approx(0.25 * _COSTS["TLT"] / 1e4)


# ----------------------------------------------------------------- state details

def test_state_details_reports_shares_and_switches_by_sub_period_from_a_hand_table():
    dates = pd.DatetimeIndex(["2007-01-31", "2007-02-28", "2007-03-30", "2010-01-29", "2010-02-26", "2010-03-31",
                              "2018-01-31", "2018-02-28", "2024-01-31", "2024-02-29"])
    states = ["HIGH", "HIGH", "LOW", "LOW", "LOW", "LOW", "HIGH", "LOW", "LOW", "HIGH"]
    table = pd.DataFrame({"state": states, "carried": False, "blackout": False}, index=dates)
    diag = mod.state_diagnostics(table, start=pd.Timestamp("2006-02-01"), end=pd.Timestamp("2026-08-31"),
                                 holdout_start=pd.Timestamp("2023-09-01"))
    assert diag["evaluations"] == 10 and diag["high_share"] == pytest.approx(0.4)
    assert diag["switches"] == 4
    by = diag["by_period"]
    assert by["2006-2008"]["evaluations"] == 3 and by["2006-2008"]["high_share"] == pytest.approx(2 / 3)
    assert by["2006-2008"]["switches"] == 1
    assert by["2009-2016"]["high_share"] == 0.0 and by["2009-2016"]["switches"] == 0
    assert by["2017-2022"]["evaluations"] == 2 and by["2017-2022"]["switches"] == 1
    assert by["2017-2022"]["switches_including_entering"] == 2      # LOW (2010) -> HIGH (2018) is a switch too
    assert by["2023-onward"]["high_share"] == pytest.approx(0.5) and by["2023-onward"]["switches_including_entering"] == 1
    assert diag["holdout"]["evaluations"] == 2 and diag["holdout"]["high_share"] == pytest.approx(0.5)
    assert diag["by_year"]["2007"]["high_share"] == pytest.approx(2 / 3)
    assert diag["blackout_month_ends_carried"] == 0
    # the switches counted across sub-period boundaries add up to the overall count
    assert sum(v["switches_including_entering"] for v in by.values()) == diag["switches"]
    assert by["2009-2016"]["switches_including_entering"] == 0


def test_state_details_counts_carried_blackout_months():
    dates = pd.DatetimeIndex(["2013-08-30", "2013-09-30", "2013-10-31", "2013-11-29"])
    table = pd.DataFrame({"state": ["LOW", "LOW", "LOW", "HIGH"], "carried": [False, False, True, False],
                          "blackout": [False, False, True, False]}, index=dates)
    diag = mod.state_diagnostics(table, start=pd.Timestamp("2006-02-01"), end=pd.Timestamp("2026-08-31"))
    assert diag["blackout_month_ends_carried"] == 1 and diag["evaluations"] == 4


# ---------------------------------------------- real cached counts (signal only)

_REAL_CACHE = os.path.join(_ROOT, "data", "mining", "cftc_oi")


@pytest.mark.slow
def test_the_cached_signal_reproduces_the_pinned_plain_rule_counts(monkeypatch):
    """SIGNAL ONLY: only the cached open-interest counts and the calendar are read. With the
    publication-blackout handling switched off, the signal falls back to the plain rule (plain D+6
    lag, no blackout handling); the counts pinned below are for that plain rule, and the states are
    identical whether the current month is in the median or not. Runs only where the cached open
    interest exists. Pinned to the vintage cached for decisions through 2026-08-31: a later refresh
    whose history carries corrections may legitimately move these counts, so re-derive them under
    the plain rule before touching the pin."""
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(_REAL_CACHE)
    decisions = mod.month_end_sessions(pd.Timestamp("1986-01-02"), pd.Timestamp("2026-08-31"))
    monkeypatch.setattr(cd, "PUBLICATION_BLACKOUTS", ())
    plain = mod.signal_state_table(snapshot.series, decisions)
    assert int(plain["cmi"].notna().sum()) == 476
    assert plain["cmi"].first_valid_index() == pd.Timestamp("1987-01-30")
    since = plain.loc["2006-02-01":"2026-08-31"]
    assert len(since) == 247 and since["state"].notna().all()
    assert round(float((since["state"] == "HIGH").mean()), 3) == 0.356
    assert int((since["state"].to_numpy()[1:] != since["state"].to_numpy()[:-1]).sum()) == 23
    exclusive = plain["cmi"].expanding(min_periods=mod.MIN_MEDIAN_MONTHS).median().shift(1)
    raw = np.where(plain["cmi"] > exclusive, "HIGH", "LOW")
    live = plain.loc["2006-02-01":"2026-08-31"].index
    assert (plain.loc[live, "state"].to_numpy() == pd.Series(raw, index=plain.index).loc[live].to_numpy()).all()


@pytest.mark.slow
def test_the_real_cached_signal_with_the_blackout_amendment_carries_exactly_the_seven_month_ends():
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(_REAL_CACHE)
    decisions = mod.month_end_sessions(pd.Timestamp("1986-01-02"), pd.Timestamp("2026-08-31"))
    table = mod.signal_state_table(snapshot.series, decisions)
    since = table.loc["2006-02-01":"2026-08-31"]
    assert len(since) == 247 and since["state"].notna().all()
    assert set(since.index[since["blackout"]].strftime("%Y-%m-%d")) == {
        "2013-10-31", "2018-12-31", "2019-01-31", "2019-02-28", "2025-10-31", "2025-11-28", "2025-12-31"}
    assert since["carried"].sum() == 7
    for date in since.index[since["blackout"]]:
        previous = since.index[since.index.get_loc(date) - 1]
        assert since.loc[date, "state"] == since.loc[previous, "state"]
    diag = mod.state_diagnostics(since, start=pd.Timestamp("2006-02-01"), end=pd.Timestamp("2026-08-31"),
                                 holdout_start=pd.Timestamp("2023-09-19"))
    assert diag["evaluations"] == 247
    states = since["state"].to_numpy()
    assert diag["switches"] == int((states[1:] != states[:-1]).sum())
    assert 0.0 < diag["high_share"] < 1.0
    assert diag["blackout_month_ends_carried"] == 7 and diag["holdout"]["carried_blackout_month_ends"] == 3
    assert diag["holdout"]["evaluations"] == 36
    # sensitivity: a shorter 2018-19 blackout window (delayed reports through 2019-01-29 only) would move
    # exactly two states of the scored span
    superseded = cd.PublicationBlackout("shorter_2018_19_window", pd.Timestamp("2018-12-25"), pd.Timestamp("2019-01-29"),
                                        pd.Timestamp("2019-02-15"), "a shorter 2018-19 window, for this sensitivity check")
    saved = cd.PUBLICATION_BLACKOUTS
    try:
        cd.PUBLICATION_BLACKOUTS = (saved[0], superseded, saved[2])
        old = mod.signal_state_table(snapshot.series, decisions).loc["2006-02-01":"2026-08-31"]
    finally:
        cd.PUBLICATION_BLACKOUTS = saved
    moved = since.index[since["state"].astype(str).to_numpy() != old["state"].astype(str).to_numpy()]
    assert list(moved.strftime("%Y-%m-%d")) == ["2019-11-29", "2020-02-28"]


@pytest.mark.slow
def test_the_real_signal_disclosures_the_tie_month_the_stale_base_months_the_lag_margin_and_the_composition():
    """Pins the measured figures the strategy docstring and catalog entry quote (signal only: counts and dates)."""
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(_REAL_CACHE)
    decisions = mod.month_end_sessions(pd.Timestamp("1986-01-02"), pd.Timestamp("2026-08-31"))
    table = mod.signal_state_table(snapshot.series, decisions)
    since = table.loc["2006-02-01":"2026-08-31"]
    # (1) 2019-11-29 is an EXACT tie resolved to LOW, and only because of the blackout-exclusion amendment
    ties = since.index[(since["cmi"] == since["median"]).to_numpy()]
    assert list(ties.strftime("%Y-%m-%d")) == ["2019-11-29"] and since.loc["2019-11-29", "state"] == "LOW"
    saved = cd.PUBLICATION_BLACKOUTS
    try:
        cd.PUBLICATION_BLACKOUTS = ()
        plain = mod.signal_state_table(snapshot.series, decisions)
    finally:
        cd.PUBLICATION_BLACKOUTS = saved
    assert plain.loc["2019-11-29", "state"] == "HIGH"
    differs = since.index[(since["state"] != plain.loc[since.index, "state"]).to_numpy()]
    assert list(differs.strftime("%Y-%m-%d")) == ["2019-11-29", "2020-02-28", "2025-02-28", "2025-10-31",
                                                  "2025-11-28", "2025-12-31"]
    # (2) the evaluation months whose 12-month base month is a blackout month-end (a window longer than 12 months)
    bases = []
    for date in table.index[table["blackout"]]:
        later = table.index[table.index.get_loc(date) + 12] if table.index.get_loc(date) + 12 < len(table) else None
        if later is not None and later >= pd.Timestamp("2006-02-01") and not table.loc[later, "blackout"]:
            bases.append(later.strftime("%Y-%m-%d"))
    assert bases == ["2014-10-31", "2019-12-31", "2020-01-31", "2020-02-28"]
    # (3) the D+6 margin is immaterial to the state: 7 days moves only the tie month (and one month before 2006), 5 none
    def states_with_lag(days):
        saved_lag = cd.AVAILABILITY_LAG_DAYS
        try:
            cd.AVAILABILITY_LAG_DAYS = days
            return mod.signal_state_table(snapshot.series, decisions)["state"]
        finally:
            cd.AVAILABILITY_LAG_DAYS = saved_lag
    base_states = table["state"].astype(object).fillna("none")
    seven = states_with_lag(7).astype(object).fillna("none")
    assert list(table.index[(base_states != seven).to_numpy()].strftime("%Y-%m-%d")) == ["1999-11-30", "2019-11-29"]
    assert (base_states == states_with_lag(5).astype(object).fillna("none")).all()
    # (4) composition drift, measured: partial-composition share of the CMI history and of the scored span
    values, _ = cd.month_end_open_interest(snapshot.series, decisions)
    contributing = np.log(values / values.shift(1)).notna().sum(axis=1)
    partial = (contributing.rolling(12, min_periods=12).min() < 18) & table["cmi"].notna()
    assert int(table["cmi"].notna().sum()) == 476 and int(partial.sum()) == 241
    assert contributing[contributing == 18].index[0] == pd.Timestamp("2006-03-31")
    assert int(partial.loc["2006-02-01":"2026-08-31"].sum()) == 12
    # (5) palladium's real reporting gaps and the stalest carried value
    gaps = snapshot.series["075651"].index.to_series().diff().dt.days
    assert [int(g) for g in gaps[gaps > 60]] == [84, 168, 259]
    assert cd.month_end_open_interest({"075651": snapshot.series["075651"]}, [pd.Timestamp("2002-03-28")])[1].iloc[0, 0] \
        == pd.Timestamp("2002-03-28") - pd.Timedelta(days=247)
    # (6) treating any contract older than 30 days as missing at a non-blackout month-end changes no state
    real_lookup = cd.month_end_open_interest

    def stale_is_missing(series_by_code, decision_dates):
        values, used = real_lookup(series_by_code, decision_dates)
        stamps = pd.Series(pd.DatetimeIndex(decision_dates), index=values.index)
        live = pd.Series([cd.blackout_at(d) is None for d in decision_dates], index=values.index)
        for code in values.columns:
            drop = ((stamps - used[code]).dt.days > 30) & live
            values.loc[drop, code] = np.nan
            used.loc[drop, code] = pd.NaT
        return values, used
    saved_lookup = cd.month_end_open_interest
    try:
        cd.month_end_open_interest = stale_is_missing
        strict = mod.signal_state_table(snapshot.series, decisions)["state"].astype(object).fillna("none")
    finally:
        cd.month_end_open_interest = saved_lookup
    assert (strict == base_states).all()
    # (7) leaving the current month out of the median changes no state of the scored span with the amendment either
    history, exclusive = [], {}
    for month in table.index:
        if table.loc[month, "blackout"] or pd.isna(table.loc[month, "cmi"]):
            continue
        if len(history) >= mod.MIN_MEDIAN_MONTHS:
            exclusive[month] = "HIGH" if table.loc[month, "cmi"] > float(np.median(history)) else "LOW"
        history.append(table.loc[month, "cmi"])
    moved_by_exclusion = [m for m in exclusive
                          if m >= pd.Timestamp("2006-02-01") and table.loc[m, "state"] != exclusive[m]]
    assert moved_by_exclusion == [] and len(exclusive) > 300


@pytest.mark.slow
def test_the_real_signal_has_a_defined_state_from_1990_and_the_dbc_span_has_the_expected_block_count():
    if not os.path.isfile(os.path.join(_REAL_CACHE, cd.MANIFEST_NAME)):
        pytest.skip("no cached CFTC open interest")
    snapshot = cd.load_cached_open_interest(_REAL_CACHE)
    table = mod.signal_state_table(snapshot.series, mod.month_end_sessions(pd.Timestamp("1986-01-02"),
                                                                           pd.Timestamp("2026-10-31")))
    assert table.loc["1990-01-01":"2026-08-31", "state"].notna().all()
    assert table["state"].first_valid_index() == pd.Timestamp("1989-12-29")
    idx = treasury.exchange_sessions(pd.Timestamp("2006-02-06"), pd.Timestamp("2026-09-23"))     # calendar only
    plan = mod.plan_windows(idx, table)
    assert len(plan["monthly"]) == 247 and len(plan["windows"]) == 24
    assert plan["truncated_final"] is True and plan["windows"][-1][:2] == (
        pd.Timestamp("2026-01-30"), pd.Timestamp("2026-09-30"))          # the final block closes after the data ends
    assert sum(len(w) for w in plan["windows_by_leg"].values()) == len(plan["windows"])
    assert plan["windows"][0][:2] == (pd.Timestamp("2006-02-28"), pd.Timestamp("2008-06-30"))
    assert mod.EXPECTED_MIN_TRADES <= len(plan["windows"])
    # the SCORED monthly convention, calendar only: one window per month, none held beyond a year
    scored = mod.plan_windows(idx, table, merge_same_leg=mod.SCORED_MERGE_SAME_LEG)
    assert len(scored["windows"]) == 247 == len(scored["monthly"])
    assert sum(len(w) for w in scored["windows_by_leg"].values()) == 247
    assert max((w[1] - w[0]).days for w in scored["windows"]) <= 33
    # the merged sensitivity holds some blocks beyond a year (the tax-walk disclosure); the scored one holds none
    days = np.array([(w[1] - w[0]).days for w in plan["windows"]])
    assert int((days > 365).sum()) > 0


# ------------------------------------------------------------------- hygiene

# the cached raw CFTC responses and their manifest (data this family adds under data/mining/cftc_oi): they are
# committed files like any other, so the hygiene scan reads them too
_CFTC_CACHE_DIR = os.path.join(_ROOT, "data", "mining", "cftc_oi")


def _new_file_paths():
    return [os.path.join(_ROOT, "src", "mining", "cftc_data.py"),
            os.path.join(_ROOT, "src", "mining", "batch5", f"{_FAMILY}.py"),
            os.path.join(_ROOT, "scripts", "run_commodity_oi_cycle_overlay_abi1.py"),
            os.path.join(_ROOT, "src", "tests", "test_mining_cftc_data.py"),
            os.path.join(_ROOT, "src", "tests", "test_batch5_commodity_open_interest_cycle_commodity_bond_overlay.py"),
            os.path.join(_ROOT, "src", "tests", "test_run_commodity_oi_cycle_overlay_abi1.py")]


def _raw_cache_paths():
    """The manifest plus the raw response file of every one of the 18 contracts, ENUMERATED from the module's
    own contract list (never a glob, so a missing or renamed file cannot silently shrink the scan)."""
    return [os.path.join(_CFTC_CACHE_DIR, cd.MANIFEST_NAME)] + [
        os.path.join(_CFTC_CACHE_DIR, f"{code}.json") for code in cd.CONTRACT_CODES]


def _hygiene_findings(path):
    """The three committed-content shapes (currency figure, YYYYMMDD integer, long digit run) for one file."""
    text = open(path, "r", encoding="utf-8").read()
    found = []
    if _DOLLAR_FIGURE.search(text):
        found.append("dollar_figure")
    if _YYYYMMDD_INT.search(text):
        found.append("yyyymmdd_integer")
    if _NUM_SHAPE.search(text):
        found.append("account_number_shape")
    return found


def test_no_dollar_figure_or_yyyymmdd_integer_in_any_new_file():
    for path in _new_file_paths():
        if not os.path.isfile(path):
            continue
        assert _hygiene_findings(path) == [], (path, "trips a committed-content shape")


def test_no_dollar_figure_or_yyyymmdd_integer_or_long_digit_run_in_the_manifest_or_any_raw_cache_file():
    if not os.path.isdir(_CFTC_CACHE_DIR):
        pytest.skip("no cached CFTC responses in this checkout")
    for path in _raw_cache_paths():
        assert os.path.isfile(path), (path, "a file this scan enumerates is missing from a cache that exists")
        assert _hygiene_findings(path) == [], (path, "trips a committed-content shape")


def test_the_raw_cache_scan_enumerates_the_manifest_and_all_eighteen_contract_files():
    paths = _raw_cache_paths()
    assert len(paths) == 19 == len(set(paths))
    assert paths[0] == os.path.join(_CFTC_CACHE_DIR, "manifest.json")
    assert sorted(os.path.basename(p) for p in paths[1:]) == sorted(f"{code}.json" for code in cd.CONTRACT_CODES)
    assert len(cd.CONTRACT_CODES) == 18
    assert {os.path.dirname(p) for p in paths} == {_CFTC_CACHE_DIR}
    assert os.path.abspath(_CFTC_CACHE_DIR) == os.path.abspath(cd.DEFAULT_CACHE_DIR)


@pytest.mark.parametrize("text,kind", [
    ("cost " + "$" + "5 a share", "dollar_figure"),
    ("cost " + "$ " + "12", "dollar_figure"),
    ("account " + "1" * 9 + " end", "account_number_shape"),
    ("account " + "1" * 17 + " end", "account_number_shape"),
    ('{"n": ' + "2026" + "0924" + "}", "yyyymmdd_integer"),
    ('{"n": ' + "1999" + "1231" + "}", "yyyymmdd_integer")],
    ids=["dollar_attached", "dollar_spaced", "account_short", "account_long", "date_integer_recent",
         "date_integer_old"])
def test_the_hygiene_scan_flags_each_content_shape_in_a_file(tmp_path, text, kind):
    path = tmp_path / "planted.json"
    path.write_text(text, encoding="utf-8")
    assert kind in _hygiene_findings(str(path))


@pytest.mark.parametrize("text", [
    '{"open_interest_all": "1234567", "id": "abc' + "1" * 9 + '"}',      # seven digits; a digit run inside a word
    "0." + "1" * 12,                                                    # a long decimal fraction is not an account
    "a price of 5 " "dollars and 12345 contracts",
    "2026-09-24 and 2026_0924"],
    ids=["short_count_and_digits_in_a_word", "long_decimal_fraction", "price_in_words", "separated_dates"])
def test_the_hygiene_scan_leaves_ordinary_text_alone(tmp_path, text):
    path = tmp_path / "ordinary.json"
    path.write_text(text, encoding="utf-8")
    assert _hygiene_findings(str(path)) == []
