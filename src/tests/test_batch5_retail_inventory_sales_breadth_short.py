"""Tests for src/mining/batch5/retail_inventory_sales_breadth_short.py.
Synthetic frames/fundamentals only (no network) -- the real coverage-
checked universe (the usable XRT holdings, see
data/mining/retail_inventory_coverage.json) is consumed only
by the bespoke driver (scripts/run_retail_inventory_breadth_mining.py),
never by this module's own logic, matching every sibling batch5 module's
fetch/compute split.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining.backtest_engine import SignalResult, run_backtest  # noqa: E402
from mining.batch5 import retail_inventory_sales_breadth_short as mod  # noqa: E402


def _index(n, start="2007-01-02"):
    return pd.bdate_range(start, periods=n)


def _instant_series(dates_vals):
    """{"usable_asof", "growth_yoy"} rows, matching
    retail_inventory_coverage.build_point_in_time_instant_series's real
    output shape (only the two columns this module actually reads)."""
    rows = [{"usable_asof": pd.Timestamp(d), "growth_yoy": v,
            "latest_fiscal_year_end": pd.Timestamp(d), "latest_val": 1.0,
            "prior_fiscal_year_end": pd.NaT, "prior_val": np.nan}
           for d, v in dates_vals]
    return pd.DataFrame(rows)


def _duration_series(dates_vals):
    """{"usable_asof", "capex_growth_yoy"} rows, matching
    edgar_point_in_time.build_point_in_time_annual_series's real output
    shape -- used here for the REVENUE leg (a duration concept)."""
    rows = [{"usable_asof": pd.Timestamp(d), "capex_growth_yoy": v,
            "latest_fiscal_year_end": pd.Timestamp(d), "latest_val": 1.0,
            "prior_fiscal_year_end": pd.NaT, "prior_val": np.nan}
           for d, v in dates_vals]
    return pd.DataFrame(rows)


# --------------------------------------------------------- quarter_end_mask

def test_quarter_end_mask_flags_last_bar_of_each_quarter():
    idx = pd.DatetimeIndex(["2020-03-30", "2020-03-31", "2020-04-01",
                            "2020-06-29", "2020-06-30"])
    mask = mod.quarter_end_mask(idx)
    assert mask.tolist() == [False, True, False, False, True]


def test_quarter_end_mask_single_bar_is_true():
    idx = pd.DatetimeIndex(["2020-03-31"])
    assert mod.quarter_end_mask(idx).tolist() == [True]


# ------------------------------------------------------------ breadth_series

def test_breadth_series_all_defined_all_ok_is_one():
    dates = pd.DatetimeIndex(["2020-03-31", "2020-06-30"])
    fundamentals_map = {
        f"T{i}": {
            "inventory": _instant_series([("2020-01-01", 0.50)]),
            "revenue": _duration_series([("2020-01-01", 0.10)]),
        }
        for i in range(5)
    }
    breadth = mod.breadth_series(dates, fundamentals_map, min_members=3)
    assert (breadth == 1.0).all()


def test_breadth_series_mixed_fraction():
    dates = pd.DatetimeIndex(["2020-06-30"])
    fundamentals_map = {
        "OK1": {"inventory": _instant_series([("2020-01-01", 0.50)]),
               "revenue": _duration_series([("2020-01-01", 0.10)])},
        "OK2": {"inventory": _instant_series([("2020-01-01", 0.30)]),
               "revenue": _duration_series([("2020-01-01", 0.05)])},
        "NOTOK1": {"inventory": _instant_series([("2020-01-01", 0.05)]),
                  "revenue": _duration_series([("2020-01-01", 0.20)])},
        "NOTOK2": {"inventory": _instant_series([("2020-01-01", -0.10)]),
                  "revenue": _duration_series([("2020-01-01", 0.02)])},
    }
    breadth = mod.breadth_series(dates, fundamentals_map, min_members=3)
    assert breadth.iloc[0] == pytest.approx(0.5)  # 2 of 4 ok


def test_breadth_series_undefined_when_below_min_members():
    dates = pd.DatetimeIndex(["2020-06-30"])
    fundamentals_map = {
        "T1": {"inventory": _instant_series([("2019-01-01", 0.50)]),
              "revenue": _duration_series([("2019-01-01", 0.10)])},
    }
    breadth = mod.breadth_series(dates, fundamentals_map, min_members=3)
    assert pd.isna(breadth.iloc[0])


def test_breadth_series_excludes_a_company_missing_one_leg():
    # A company with usable inventory but no usable revenue (or vice
    # versa) must not count toward EITHER the numerator or the
    # denominator -- undefined is undefined, never a False.
    dates = pd.DatetimeIndex(["2020-06-30"])
    fundamentals_map = {
        "OK1": {"inventory": _instant_series([("2020-01-01", 0.50)]),
               "revenue": _duration_series([("2020-01-01", 0.10)])},
        "OK2": {"inventory": _instant_series([("2020-01-01", 0.50)]),
               "revenue": _duration_series([("2020-01-01", 0.10)])},
        "MISSING_REV": {"inventory": _instant_series([("2020-01-01", 0.99)]),
                        "revenue": _duration_series([])},
    }
    breadth = mod.breadth_series(dates, fundamentals_map, min_members=2)
    assert breadth.iloc[0] == pytest.approx(1.0)  # only OK1/OK2 count


def test_breadth_series_respects_point_in_time_staleness():
    # A fact filed well before the evaluation date, and long enough ago
    # to exceed the staleness cap, must not count as "defined".
    dates = pd.DatetimeIndex(["2025-01-01"])
    fundamentals_map = {
        f"T{i}": {"inventory": _instant_series([("2019-01-01", 0.50)]),
                  "revenue": _duration_series([("2019-01-01", 0.10)])}
        for i in range(5)
    }
    breadth = mod.breadth_series(dates, fundamentals_map, min_members=3)
    assert pd.isna(breadth.iloc[0])


# ------------------------------------------------------------ decision_dates

def test_decision_dates_no_entry_when_never_above_threshold():
    n = 10
    breadth_ok = np.array([False] * n)
    qe = np.array([False, False, True, False, False, True,
                  False, False, True, False])
    entries, exits = mod.decision_dates(breadth_ok, qe)
    assert len(entries) == 0
    assert len(exits) == 0


def test_decision_dates_enters_and_exits_at_quarter_ends():
    n = 12
    qe = np.zeros(n, dtype=bool)
    qe[[3, 7, 11]] = True
    breadth_ok = np.zeros(n, dtype=bool)
    breadth_ok[3] = True  # triggers at first quarter-end
    breadth_ok[7] = False  # off at second -> exit
    entries, exits = mod.decision_dates(breadth_ok, qe)
    assert entries.tolist() == [3]
    assert exits.tolist() == [7]


def test_decision_dates_rolls_forward_while_still_triggered():
    n = 16
    qe = np.zeros(n, dtype=bool)
    qe[[3, 7, 11, 15]] = True
    breadth_ok = np.zeros(n, dtype=bool)
    breadth_ok[3] = True
    breadth_ok[7] = True   # still triggered -> stays short (no interim exit)
    breadth_ok[11] = False  # off -> exit here
    entries, exits = mod.decision_dates(breadth_ok, qe)
    assert entries.tolist() == [3]
    assert exits.tolist() == [11]


def test_decision_dates_keeps_an_entry_that_never_exits_within_the_data():
    # No time-stop in this mechanism (unlike the sibling capex family) --
    # a position that stays triggered through the end of available data
    # is a real, legitimate open position, censored at data end. The
    # engine's own end_of_data forced-close handles it; decision_dates
    # must NOT discard the entry the way capex_cycle_blue_chip_dip_
    # pooled's bounded-hold convention does.
    n = 8
    qe = np.zeros(n, dtype=bool)
    qe[[3, 7]] = True
    breadth_ok = np.zeros(n, dtype=bool)
    breadth_ok[3] = True
    breadth_ok[7] = True  # still triggered at the LAST quarter-end -> no exit ever fires
    entries, exits = mod.decision_dates(breadth_ok, qe)
    assert entries.tolist() == [3]
    assert exits.tolist() == []


def test_decision_dates_no_entry_without_a_following_bar():
    n = 4
    qe = np.array([False, False, False, True])
    breadth_ok = np.array([False, False, False, True])
    entries, exits = mod.decision_dates(breadth_ok, qe)
    assert len(entries) == 0  # signal fires on the FINAL bar, no fill bar exists


# ------------------------------------------------------------- make_signal_fn

def test_make_signal_fn_end_to_end_short_through_engine():
    idx = _index(1600)  # ~6.3 years of business days, several real quarters
    # XRT drifts DOWN over the window -- a short should profit.
    close = pd.Series(np.linspace(100.0, 60.0, len(idx)), index=idx)
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close,
                          "Close": close}, index=idx)

    # A permanently-triggered breadth panel: every company shows
    # inventory growth persistently outpacing sales growth, refreshed
    # periodically so point-in-time staleness never lapses.
    fundamentals_map = {}
    for i in range(6):
        dated = [(idx[k], 0.40) for k in range(0, len(idx), 150)]
        rev_dated = [(idx[k], 0.05) for k in range(0, len(idx), 150)]
        fundamentals_map[f"T{i}"] = {
            "inventory": _instant_series(dated),
            "revenue": _duration_series(rev_dated),
        }

    signal_fn = mod.make_signal_fn(
        "XRT", idx, {"XRT": frame}, fundamentals_map=fundamentals_map,
        min_members=3)  # only 6 companies in this fixture; PARAMETERS'
                        # real default (20) is sized for the full 68-name
                        # coverage-checked universe, not a unit fixture
    trades_df, in_market = run_backtest(
        frame, signal_fn, cost_bps=0.0, direction="short")

    assert len(trades_df) > 0
    # A short into a falling price must show positive gross returns.
    assert (trades_df["gross_ret"] > 0).all()


def test_make_signal_fn_never_triggers_with_empty_fundamentals():
    idx = _index(300)
    close = pd.Series(100.0, index=idx)
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close,
                          "Close": close}, index=idx)
    signal_fn = mod.make_signal_fn("XRT", idx, {"XRT": frame}, fundamentals_map={})
    for i in range(len(idx)):
        result = signal_fn(frame.iloc[:i + 1], i)
        assert result == SignalResult()


def test_make_signal_fn_rejects_wrong_ticker():
    idx = _index(50)
    close = pd.Series(100.0, index=idx)
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close,
                          "Close": close}, index=idx)
    with pytest.raises(ValueError, match="XRT"):
        mod.make_signal_fn("SPY", idx, {"SPY": frame}, fundamentals_map={})


# --------------------------------------------------------------- contract

def test_module_contract_fields():
    assert mod.FAMILY_ID == "retail_inventory_sales_breadth_short"
    assert mod.SHAPE == "panel_breadth"
    assert mod.TRADED == ["XRT"]
    assert mod.SIGNAL == []
    assert mod.BENCHMARK == "SPY"
    assert 0.0 < mod.PARAMETERS["breadth_threshold"] < 1.0
    assert mod.PARAMETERS["direction"] == "short"
