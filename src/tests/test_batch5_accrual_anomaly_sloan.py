"""Tests for src/mining/batch5/accrual_anomaly_sloan.py. Synthetic
fundamentals/frames only (no network) -- mirrors
test_batch5_capex_cycle_blue_chip_dip_pooled.py's own fixture/engine-
agreement/lookahead conventions, adapted for a cross-sectional ranking
signal instead of a per-ticker absolute-threshold one.
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

from mining import backtest_engine  # noqa: E402
from mining.batch5 import accrual_anomaly_sloan as mod  # noqa: E402
from mining.batch5 import capex_cycle_blue_chip_dip_pooled as capex_mod  # noqa: E402


def _index(n, start="2015-01-05"):
    return pd.bdate_range(start, periods=n)


def _ohlc(index, close):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": close * 1.001, "Low": close * 0.999,
                         "Close": close}, index=index)


def _pit(asof, val):
    return pd.DataFrame([{"usable_asof": pd.Timestamp(asof), "latest_val": val}])


def _assets_pit(asof, latest, prior=np.nan):
    return pd.DataFrame([{"usable_asof": pd.Timestamp(asof), "latest_val": latest,
                          "prior_val": prior}])


def _const_fundamentals(ni, cfo, assets, *, prior_assets=np.nan, asof="2000-01-01"):
    """A company whose NI/CFO/Assets are all known from `asof` onward and
    never change again -- gives a FIXED accrual ratio at every later
    query date PROVIDED the query stays within STALENESS_CAP_DAYS (400)
    of `asof`. Useful for isolating the ranking/walk logic from the
    point-in-time join machinery (already covered by its own dedicated
    tests in test_mining_edgar_data.py / test_mining_accrual_coverage.py).
    """
    return {"ni": _pit(asof, ni), "cfo": _pit(asof, cfo),
            "assets": _assets_pit(asof, assets, prior_assets)}


def _periodic_pit(index, val, *, every=250, extra_cols=None):
    """The SAME value re-disclosed every `every` bars across `index` --
    keeps a fixed accrual leg FRESH (inside STALENESS_CAP_DAYS) across a
    test window spanning multiple years, matching
    test_batch5_capex_cycle_blue_chip_dip_pooled.py's own
    `_periodic_fundamentals` convention (a real company refiles
    annually)."""
    rows = []
    for i in range(0, len(index), every):
        row = {"usable_asof": pd.Timestamp(index[i]), "latest_val": val}
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return pd.DataFrame(rows)


def _linear_pool_fundamentals(n, index, *, lowest=-0.60, step=0.10, every=250):
    """n tickers T00..T{n-1} over `index`'s own calendar, each a FIXED
    accrual ratio ascending from `lowest`, kept fresh throughout via
    periodic re-disclosure (`_periodic_pit`). Alphabetical ticker order
    == accrual-ascending order, so a ranking test can predict the
    selected set without re-deriving it."""
    return {
        f"T{i:02d}": {
            "ni": _periodic_pit(index, lowest + step * i, every=every),
            "cfo": _periodic_pit(index, 0.0, every=every),
            "assets": _periodic_pit(index, 1.0, every=every, extra_cols={"prior_val": np.nan}),
        }
        for i in range(n)
    }


# --------------------------------------------------------- accrual_ratio_at

def test_accrual_ratio_at_none_fundamentals_is_all_nan():
    dates = _index(5)
    result = mod.accrual_ratio_at(None, dates)
    assert result.isna().all()


def test_accrual_ratio_at_missing_leg_is_all_nan():
    dates = _index(5)
    fundamentals = {"ni": _pit("2000-01-01", 100.0), "cfo": _pit("2000-01-01", 50.0)}
    result = mod.accrual_ratio_at(fundamentals, dates)
    assert result.isna().all()


def test_accrual_ratio_at_computes_ni_minus_cfo_over_average_assets():
    dates = _index(3, start="2010-01-04")
    fundamentals = _const_fundamentals(150.0, 100.0, 1000.0, prior_assets=800.0,
                                       asof="2009-06-01")  # ~217 days before, fresh
    result = mod.accrual_ratio_at(fundamentals, dates)
    # avg_assets = (1000 + 800) / 2 = 900; accrual = (150 - 100) / 900
    assert result.iloc[-1] == pytest.approx(50.0 / 900.0)


def test_accrual_ratio_at_falls_back_to_latest_assets_without_a_prior_year():
    dates = _index(3, start="2010-01-04")
    fundamentals = _const_fundamentals(150.0, 100.0, 1000.0, asof="2009-06-01")
    result = mod.accrual_ratio_at(fundamentals, dates)
    assert result.iloc[-1] == pytest.approx(50.0 / 1000.0)


def test_accrual_ratio_at_zero_average_assets_is_nan_not_a_crash():
    dates = _index(3, start="2010-01-04")
    fundamentals = _const_fundamentals(150.0, 100.0, 0.0, asof="2009-06-01")
    result = mod.accrual_ratio_at(fundamentals, dates)
    assert result.isna().all()


def test_accrual_ratio_at_before_first_disclosure_is_nan():
    dates = _index(5, start="1999-12-27")  # spans before/after the disclosure date
    fundamentals = _const_fundamentals(150.0, 100.0, 1000.0, asof=str(dates[3].date()))
    result = mod.accrual_ratio_at(fundamentals, dates)
    assert result.iloc[0:3].isna().all()
    assert not pd.isna(result.iloc[-1])


# ------------------------------------------------------------- year_end_mask

def test_year_end_mask_marks_the_last_bar_of_each_calendar_year():
    idx = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-12-31"])
    mask = mod.year_end_mask(idx)
    assert list(mask) == [False, True, False, True]


def test_year_end_mask_single_bar_is_always_year_end():
    idx = pd.to_datetime(["2020-06-15"])
    assert list(mod.year_end_mask(idx)) == [True]


# ---------------------------------------------------------- selection_panel

def _linear_pool_fundamentals_single(n, asof, *, lowest=-0.60, step=0.10):
    """n tickers T00..T{n-1}, each a fixed accrual ratio ascending from
    `lowest`, disclosed ONCE at `asof` -- for a single-snapshot query
    (must stay within STALENESS_CAP_DAYS of the query date). Alphabetical
    ticker order == accrual-ascending order, so a ranking test can
    predict the selected set without re-deriving it."""
    return {
        f"T{i:02d}": _const_fundamentals(lowest + step * i, 0.0, 1.0, asof=asof)
        for i in range(n)
    }


def test_selection_panel_selects_the_bottom_fraction_among_defined():
    fundamentals_map = _linear_pool_fundamentals_single(12, "2015-06-01")  # >= min_rank_members(10)
    dates = pd.DatetimeIndex([pd.Timestamp("2015-12-31")])
    panel = mod.selection_panel(dates, fundamentals_map, bottom_fraction=0.20,
                                min_rank_members=10)
    selected = panel.loc[dates[0]]
    # round(0.20 * 12) = 2 -> the two LOWEST-accrual tickers (T00, T01).
    assert selected[selected].index.tolist() == ["T00", "T01"]
    assert selected.sum() == 2


def test_selection_panel_undefined_below_min_rank_members_selects_nobody():
    fundamentals_map = _linear_pool_fundamentals_single(5, "2015-06-01")  # < min_rank_members(10)
    dates = pd.DatetimeIndex([pd.Timestamp("2015-12-31")])
    panel = mod.selection_panel(dates, fundamentals_map, bottom_fraction=0.20,
                                min_rank_members=10)
    assert not panel.loc[dates[0]].any()


def test_selection_panel_ties_break_alphabetically():
    fundamentals_map = {
        f"T{i:02d}": _const_fundamentals(0.0, 0.0, 1.0, asof="2015-06-01") for i in range(10)
    }
    dates = pd.DatetimeIndex([pd.Timestamp("2015-12-31")])
    panel = mod.selection_panel(dates, fundamentals_map, bottom_fraction=0.20,
                                min_rank_members=10)
    selected = panel.loc[dates[0]]
    # All tied at accrual=0.0 -> the first 2 alphabetically (T00, T01).
    assert selected[selected].index.tolist() == ["T00", "T01"]


def test_selection_panel_empty_fundamentals_map():
    dates = pd.DatetimeIndex([pd.Timestamp("2015-12-31")])
    panel = mod.selection_panel(dates, {})
    assert panel.empty or not panel.any().any()


# ------------------------------------------------------------- decision_dates

def test_decision_dates_no_entries_before_the_first_year_end_bar():
    idx = _index(10, start="2015-06-01")  # < 1 full year, no year-end bar inside
    fundamentals_map = _linear_pool_fundamentals(12, idx)
    entries, exits = mod.decision_dates("T00", idx, fundamentals_map)
    assert len(entries) == 0


def test_decision_dates_entry_fires_for_a_selected_ticker_at_year_end():
    idx = _index(600, start="2014-06-01")  # spans at least one full calendar year-end
    fundamentals_map = _linear_pool_fundamentals(12, idx)
    entries, exits = mod.decision_dates("T00", idx, fundamentals_map)  # T00 = lowest accrual
    assert len(entries) >= 1


def test_decision_dates_never_fires_for_a_non_selected_ticker():
    idx = _index(600, start="2014-06-01")
    fundamentals_map = _linear_pool_fundamentals(12, idx)
    entries, exits = mod.decision_dates("T11", idx, fundamentals_map)  # highest accrual
    assert len(entries) == 0


def test_decision_dates_ticker_missing_from_fundamentals_map_never_enters():
    idx = _index(600, start="2014-06-01")
    fundamentals_map = _linear_pool_fundamentals(12, idx)
    entries, exits = mod.decision_dates("GHOST", idx, fundamentals_map)
    assert len(entries) == 0


def test_decision_dates_empty_input():
    entries, exits = mod.decision_dates("T00", pd.DatetimeIndex([]), {})
    assert len(entries) == 0 and len(exits) == 0


def test_decision_dates_exit_fires_when_ticker_falls_out_of_bottom_fraction():
    # Build two full years: T00 has the lowest accrual in year 1 (selected),
    # then a fresh disclosure just before year 2's reformation raises its
    # accrual above the cutoff (no longer selected) -- exit should fire on
    # year 2's own reformation bar.
    idx = _index(760, start="2014-01-02")  # spans two calendar year-ends
    fundamentals_map = _linear_pool_fundamentals(12, idx)
    # Give T00 a SECOND, later disclosure (well before the 2nd year-end)
    # that moves it to the top of the ranking -- the first disclosure is
    # `idx[0]` (via _linear_pool_fundamentals's own every=250 periodic
    # schedule), so this stays a genuine later revision, not a duplicate.
    later_asof = idx[500]
    fundamentals_map["T00"] = {
        "ni": pd.concat([_pit(idx[0], -0.60), _pit(later_asof, 5.0)], ignore_index=True),
        "cfo": pd.concat([_pit(idx[0], 0.0), _pit(later_asof, 0.0)], ignore_index=True),
        "assets": pd.concat([_assets_pit(idx[0], 1.0), _assets_pit(later_asof, 1.0)],
                            ignore_index=True),
    }
    entries, exits = mod.decision_dates("T00", idx, fundamentals_map)
    assert len(entries) >= 1
    assert len(exits) >= 1


# ------------------------------------------------------------ engine agreement

def test_end_to_end_matches_real_engine_bar_for_bar():
    idx = _index(600, start="2014-06-01")
    frame = _ohlc(idx, np.full(600, 100.0))
    fundamentals_map = _linear_pool_fundamentals(12, idx)

    fn = mod.make_signal_fn("T00", idx, {"T00": frame}, fundamentals_map=fundamentals_map)
    trades, _ = backtest_engine.run_backtest(frame, fn)

    entries_np, exits_np = mod.decision_dates("T00", idx, fundamentals_map)
    assert len(entries_np) == len(trades)
    for k in range(len(trades)):
        trade = trades.iloc[k]
        assert trade["entry_date"] == idx[entries_np[k] + 1]
        if k < len(exits_np):
            assert trade["exit_date"] == idx[exits_np[k]]


def test_lookahead_future_bars_never_change_past_decisions():
    idx = _index(760, start="2014-01-02")
    frame = _ohlc(idx, np.full(760, 100.0))
    fundamentals_map = _linear_pool_fundamentals(12, idx)
    k = 500
    fn = mod.make_signal_fn(
        "T00", idx[:k + 1], {"T00": frame}, fundamentals_map=fundamentals_map)
    before = [fn(frame.iloc[:t + 1], t) for t in range(k + 1)]

    perturbed = frame.copy()
    perturbed.loc[idx[k + 1:], ["Open", "High", "Low", "Close"]] *= 5.0
    fn2 = mod.make_signal_fn(
        "T00", idx[:k + 1], {"T00": perturbed}, fundamentals_map=fundamentals_map)
    after = [fn2(perturbed.iloc[:t + 1], t) for t in range(k + 1)]
    assert before == after


def test_lookahead_future_fundamentals_disclosures_never_change_past_decisions():
    """Sibling of test_lookahead_future_bars_never_change_past_decisions
    (this signal needs its own fundamentals counterpart): this signal's decisions
    depend on FUNDAMENTALS, not prices, so a price-only perturbation
    cannot detect a future-fundamentals leak. Perturbs a brand-new,
    FUTURE-dated (beyond bar k) EDGAR disclosure -- for T00 itself AND
    for every OTHER pool member (the cross-sectional ranking closes over
    all of them) -- and asserts every decision up to k is unchanged."""
    idx = _index(760, start="2014-01-02")
    frame = _ohlc(idx, np.full(760, 100.0))
    fundamentals_map = _linear_pool_fundamentals(12, idx)
    k = 500

    fn = mod.make_signal_fn(
        "T00", idx[:k + 1], {"T00": frame}, fundamentals_map=fundamentals_map)
    before = [fn(frame.iloc[:t + 1], t) for t in range(k + 1)]

    future_asof = idx[k + 100]
    perturbed_fundamentals = {
        ticker: {
            "ni": pd.concat([legs["ni"], _pit(future_asof, 999.0)], ignore_index=True),
            "cfo": legs["cfo"],
            "assets": legs["assets"],
        }
        for ticker, legs in fundamentals_map.items()
    }
    fn2 = mod.make_signal_fn(
        "T00", idx[:k + 1], {"T00": frame}, fundamentals_map=perturbed_fundamentals)
    after = [fn2(frame.iloc[:t + 1], t) for t in range(k + 1)]
    assert before == after


def test_rejects_missing_frames():
    idx = _index(300)
    with pytest.raises(KeyError):
        mod.make_signal_fn("T00", idx, {})


# ------------------------------------------------------------------- contract

def test_traded_reuses_the_capex_watchlist_exactly():
    assert mod.TRADED == capex_mod.TRADED
    assert len(mod.TRADED) == len(set(mod.TRADED))


def test_traded_is_bounded_25_to_75_names():
    assert 25 <= len(mod.TRADED) <= 75


def test_bottom_fraction_and_min_rank_members_match_parameters():
    assert mod.BOTTOM_FRACTION == mod.PARAMETERS["bottom_fraction"]
    assert mod.MIN_RANK_MEMBERS == mod.PARAMETERS["min_rank_members"]


def test_make_signal_fn_with_no_fundamentals_map_never_enters():
    idx = _index(600, start="2014-06-01")
    frame = _ohlc(idx, np.full(600, 100.0))
    fn = mod.make_signal_fn("T00", idx, {"T00": frame})
    trades, _ = backtest_engine.run_backtest(frame, fn)
    assert len(trades) == 0


def test_module_parameters_match_the_catalog_entry_exactly():
    import run_batch5_mining as batch5
    from mining import catalog as catalog_mod
    entries = {e["family_id"]: e for e in catalog_mod.load_catalog(batch5.CATALOG_PATH)}
    entry = entries[mod.FAMILY_ID]
    assert entry["status"] == "eligible"
    assert batch5.parameter_drift(mod.PARAMETERS, entry["parameters"]) is None
    assert set(entry["tickers"]) == set(mod.TRADED)
