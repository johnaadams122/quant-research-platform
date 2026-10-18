"""Tests for src/mining/batch5/treasury_month_end_index_demand_overlay.py.

Contract shell, the ex-ante month-end SCHEDULE (the load-bearing calendar
logic), the family's curve builder and dispatch safety. The account-curve
arithmetic itself is tested in test_mining_long_overlay.py; the bespoke driver
in test_run_treasury_month_end_overlay_abi1.py. This family is ALWAYS run
through its own driver (scripts/run_treasury_month_end_overlay_abi1.py), never
the generic scripts/run_abi1_mining.py dispatch. The family's catalog entry
is read from data/mining/catalog.json.
"""
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

from mining import long_overlay as lo  # noqa: E402
from mining.batch5 import treasury_month_end_index_demand_overlay as mod  # noqa: E402

_FAMILY = "treasury_month_end_index_demand_overlay"
_CATALOG_ENTRY_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == _FAMILY)


_DOLLAR_FIGURE = re.compile(r"\$\s?\d")


def _catalog_entry() -> dict:
    with open(_CATALOG_ENTRY_PATH, "r", encoding="utf-8") as fh:
        return _catalog_entry_from(fh)


# -------------------------------------------------------------- contract

def test_module_constants_match_the_candidate_contract():
    assert mod.FAMILY_ID == _FAMILY
    assert mod.SHAPE == "pooled_single_asset"       # contract LABEL only, see docstring
    assert mod.TRADED == ["IEF", "TLT", "AGG"]
    assert mod.SIGNAL == ["^IRX"]                   # financing-rate series, never traded
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_signal_fn)


def test_parameters_are_tied_to_the_constants_they_describe():
    assert mod.PARAMETERS["overlay_notional_fraction"] == mod.OVERLAY_NOTIONAL == 0.5
    assert mod.PARAMETERS["window_sessions"] == mod.WINDOW_SESSIONS == 3
    assert mod.PARAMETERS["overlay_legs"] == list(mod.TRADED)
    assert mod.PARAMETERS["benchmark"] == "SPY" == mod.BENCHMARK
    assert mod.OVERLAY_NOTIONAL <= lo.MAX_OVERLAY_NOTIONAL
    assert mod.PARAMETERS["financing_rate"].startswith("^IRX")
    assert "no spread" in mod.PARAMETERS["financing_rate"]


def test_catalog_entry_is_complete_honest_and_new():
    from mining import catalog as catalog_mod
    entry = _catalog_entry()
    for field in catalog_mod.REQUIRED_FIELDS:
        assert field in entry, field
    assert entry["family_id"] == _FAMILY
    assert entry["status"] == "eligible"
    assert entry["account_types"] == ["taxable"]        # the overlay is margin-financed
    assert entry["tickers"] == ["IEF", "TLT", "AGG", "SPY"]
    assert entry["timeframe"] == "daily"
    assert entry["added_date"] == "2026-09-23"
    source = entry["source"]
    assert "Hartley" in source and "Schwarz" in source and "SSRN 3440417" in source
    assert "UNVERIFIED" in source                       # journal-publication status of the working paper
    assert "Liberty Street" in source
    mechanism = entry["mechanism"].lower()
    assert "overlay" in mechanism and "spy" in mechanism and "margin" in mechanism
    for neighbour in ("month_end_rebalancing_flow_rotation",
                      "long_treasury_auction_cycle_3x_inverse_rotation",
                      "long_treasury_3x_fomc_window_drift"):
        assert neighbour in entry["distinctness_note"]
    assert not _DOLLAR_FIGURE.search(json.dumps(entry))     # no dollar figures in the catalog text


def test_the_pre_registration_text_says_which_constants_were_and_were_not_chosen_blind():
    """Some constants were fixed after descriptive looks at these same ETFs
    (in-sample data only). The module and the catalog must both carry that
    disclosure: which constants were and were not chosen blind, and that those
    looks ran outside the trial ledger."""
    doc = mod.__doc__
    assert "What was and was not chosen blind" in doc
    for phrase in ("fixed after", "headline example", "outside the trial ledger",
                   "next-open", "in-sample", "holdout"):
        assert phrase in doc, phrase
    source = _catalog_entry()["source"]
    lowered = source.lower()
    for phrase in ("not all were chosen blind", "fixed after those looks",
                   "headline example", "outside the trial ledger",
                   "next-open", "in-sample", "before this pipeline computed any return"):
        assert phrase in lowered, phrase


def test_the_risk_layer_disclosure_states_the_basis_rule_and_its_scope():
    """The four refusals are measured on the overlay-only excess account. The
    module and the catalog text state that risk-basis rule, and scope the
    whole-account claim to what the data shows: an SPY-core account clears the
    0.35 cap only if the overlay offsets most of SPY's own drawdown."""
    doc = mod.__doc__
    for phrase in ("risk-basis rule", "overlay-only excess account", "SPY-relative",
                   "longest losing-week streak", "worst 8-week stretch", "0.35",
                   "much less binding", "units of whole-account equity"):
        assert phrase in doc, phrase
    lowered = _catalog_entry()["source"].lower()
    for phrase in ("risk-basis rule", "overlay-only excess account", "spy-relative refusals",
                   "longest losing-week streak", "worst 8-week stretch", "0.35",
                   "much less binding", "units of whole-account equity"):
        assert phrase in lowered, phrase
    # scoped to what the data shows, not a claim about every overlay
    for phrase in ("only if the overlay offsets most of it",):
        assert phrase in doc.lower() and phrase in lowered, phrase


def test_the_distinctness_note_confronts_why_turn_of_month_was_ruled_out():
    """turn_of_month's catalog ruled_out_reason is economic immateriality at
    small-account size, NOT a mechanism failure. Dismissing it on mechanism
    alone would misstate that reason; the note must confront the materiality
    question and say why this family is still worth one trial."""
    note = _catalog_entry()["distinctness_note"].lower()
    assert "turn_of_month" in note
    for phrase in ("immaterial", "account size", "not a mechanism failure", "materiality review",
                   "does not depend on account size"):
        assert phrase in note, phrase


def test_the_dollar_figure_pattern_actually_matches_a_dollar_figure():
    # built at runtime so this file itself carries no literal dollar amount
    dollar = chr(36)
    assert _DOLLAR_FIGURE.search("costs " + dollar + "500 a year")
    assert _DOLLAR_FIGURE.search(dollar + "5")
    assert not _DOLLAR_FIGURE.search("no money here, 50% of equity")


def test_driver_discovery_validates_this_module_contract_shape():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == _FAMILY
    assert spec["shape"] == "pooled_single_asset"
    assert spec["traded"] == ["IEF", "TLT", "AGG"]
    assert spec["signal"] == ["^IRX"]
    assert spec["benchmark"] == "SPY"
    assert spec["parameters"] == mod.PARAMETERS


# ----------------------------------------------------------- dispatch safety

def test_generic_dispatch_call_shape_raises_instead_of_scoring_anything():
    idx = pd.bdate_range("2020-01-02", periods=30)
    frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
    with pytest.raises(RuntimeError, match="bespoke driver"):
        mod.make_signal_fn("IEF", idx, {"IEF": frame})


def test_real_generic_dispatch_builder_fails_closed_for_this_family():
    import run_abi1_mining as generic
    import run_batch5_mining as driver
    idx = pd.bdate_range("2020-01-02", periods=60)
    frame = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0}, index=idx)
    frames = {s: frame for s in ("IEF", "TLT", "AGG", "SPY", "^IRX")}
    plan = {"spec": driver.validate_module_contract(mod, repo_root=_ROOT),
            "module": mod, "effective": ["IEF", "TLT", "AGG"],
            "costs": {s: {"cost_bps": 1.0} for s in ("IEF", "TLT", "AGG")}}
    with pytest.raises(RuntimeError, match="bespoke driver"):
        generic._build_curve(plan, frames)


# ----------------------------------------------- the ex-ante month-end schedule

def _xnys_sessions_and_adhoc(start, end):
    import exchange_calendars as xc
    cal = xc.get_calendar("XNYS", start=start, end=end)
    sessions = pd.DatetimeIndex(cal.sessions).tz_localize(None) if cal.sessions.tz else pd.DatetimeIndex(cal.sessions)
    adhoc = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tz else pd.Timestamp(d)
                              for d in cal.adhoc_holidays])
    return sessions.normalize(), adhoc.normalize()


def test_scheduled_sessions_equal_the_exchange_calendar_plus_only_the_unscheduled_closures():
    """Whole-set cross-check, not spot dates (a spot-date check can miss a rule
    edge such as 2027-12-31, a session before a Saturday New Year's Day): over
    2001-2028 the scheduled sessions are EXACTLY the exchange calendar's
    sessions plus the ad hoc closure days (which were not known in advance)."""
    start, end = pd.Timestamp("2001-01-01"), pd.Timestamp("2028-12-31")
    sessions, adhoc = _xnys_sessions_and_adhoc(start, end)
    scheduled = mod.scheduled_sessions(start, end)
    adhoc_in_range = adhoc[(adhoc >= start) & (adhoc <= end)]
    assert scheduled.equals(sessions.union(adhoc_in_range))
    assert set(scheduled.difference(sessions)) == set(adhoc_in_range) and len(adhoc_in_range) >= 9


def test_scheduled_sessions_keep_an_unscheduled_closure_and_drop_a_scheduled_holiday():
    s = mod.scheduled_sessions(pd.Timestamp("2012-10-22"), pd.Timestamp("2012-11-02"))
    assert pd.Timestamp("2012-10-29") in s and pd.Timestamp("2012-10-30") in s     # Hurricane Sandy closure
    assert pd.Timestamp("2018-04-02") in mod.scheduled_sessions(pd.Timestamp("2018-03-26"),
                                                                pd.Timestamp("2018-04-06"))
    gf = mod.scheduled_sessions(pd.Timestamp("2018-03-26"), pd.Timestamp("2018-04-06"))
    assert pd.Timestamp("2018-03-30") not in gf                                     # Good Friday
    # Saturday New Year's Day is NOT observed on the Friday; Saturday Christmas IS
    assert pd.Timestamp("2027-12-31") in mod.scheduled_sessions(pd.Timestamp("2027-12-20"),
                                                                pd.Timestamp("2028-01-05"))
    assert pd.Timestamp("2021-12-24") not in mod.scheduled_sessions(pd.Timestamp("2021-12-20"),
                                                                    pd.Timestamp("2021-12-31"))


_HAND_WINDOWS = [
    # (year, month, entry T-3, exit T) -- computed by hand from the NYSE holiday rules
    (2018, 3, "2018-03-26", "2018-03-29"),    # Good Friday 30th closed: T is Thursday the 29th
    (2026, 5, "2026-05-26", "2026-05-29"),    # Memorial Day the 25th
    (2026, 6, "2026-06-25", "2026-06-30"),    # window straddles a weekend
    (2026, 9, "2026-09-25", "2026-09-30"),
    (2026, 11, "2026-11-24", "2026-11-30"),   # Thanksgiving the 26th inside the window's span
    (2021, 12, "2021-12-28", "2021-12-31"),   # Fri 24th closed (Christmas observed), Fri 31st open
    (2022, 12, "2022-12-27", "2022-12-30"),   # Christmas observed on Monday the 26th
    (2027, 12, "2027-12-28", "2027-12-31"),   # Saturday Jan 1 2028: the 31st is a session
    (2004, 12, "2004-12-28", "2004-12-31"),   # same calendar shape, exchange was open
    (2010, 12, "2010-12-28", "2010-12-31"),
    (2012, 10, "2012-10-26", "2012-10-31"),   # 29th and 30th were SCHEDULED sessions (closed for Sandy)
    (2016, 2, "2016-02-24", "2016-02-29"),    # leap-year month end
    (2022, 1, "2022-01-26", "2022-01-31"),
]


@pytest.mark.parametrize("year,month,entry,exit_", _HAND_WINDOWS)
def test_month_end_window_matches_a_hand_computed_case(year, month, entry, exit_):
    windows = mod.month_end_windows(pd.Timestamp(year, month, 1), pd.Timestamp(year, month, 28))
    assert windows == [(pd.Timestamp(entry), pd.Timestamp(exit_))]


def test_every_window_is_three_scheduled_sessions_before_the_last_scheduled_session_of_its_month():
    windows = mod.month_end_windows(pd.Timestamp("2003-01-01"), pd.Timestamp("2027-12-31"))
    sched = mod.scheduled_sessions(pd.Timestamp("2002-12-01"), pd.Timestamp("2028-01-31"))
    assert len(windows) == 25 * 12
    for entry, exit_ in windows:
        assert entry.to_period("M") == exit_.to_period("M")          # T-3 is in the same month
        pos = sched.get_loc(exit_)
        assert sched[pos - mod.WINDOW_SESSIONS] == entry
        assert sched[pos + 1].to_period("M") != exit_.to_period("M")  # T is the LAST scheduled session


def test_month_ends_agree_with_the_independent_easter_and_memorial_day_rule():
    """short_rate_easing_smallcap_rotation derives the scheduled month-end from
    the last weekday stepped back over Good Friday and Memorial Day, with no
    exchange calendar at all. Two independent derivations must agree on every
    month of the real history."""
    from mining.batch5 import short_rate_easing_smallcap_rotation as other
    for entry, exit_ in mod.month_end_windows(pd.Timestamp("2002-08-01"), pd.Timestamp("2027-12-31")):
        assert exit_ == other.scheduled_month_end(exit_.year, exit_.month), exit_


def test_the_schedule_never_reads_data_it_takes_dates_only():
    import inspect
    assert list(inspect.signature(mod.scheduled_sessions).parameters) == ["start", "end"]
    assert list(inspect.signature(mod.month_end_windows).parameters) == ["start", "end"]


# ------------------------------ windows against a data calendar (with power)

def _data_calendar(start, end, *, drop=()):
    s = mod.scheduled_sessions(pd.Timestamp(start), pd.Timestamp(end))
    return s.difference(pd.DatetimeIndex(list(drop)))


def test_windows_in_data_skips_a_first_window_whose_entry_precedes_the_data():
    march = (pd.Timestamp("2010-03-26"), pd.Timestamp("2010-03-31"))      # Good Friday is Apr 2 2010
    starts_before_entry = mod.windows_in_data(_data_calendar("2010-03-25", "2010-08-31"))
    assert starts_before_entry["windows"][0] == march
    assert starts_before_entry["skipped_before_data"] == 0
    starts_inside_window = mod.windows_in_data(_data_calendar("2010-03-30", "2010-08-31"))
    assert march not in starts_inside_window["windows"]                    # entry (Mar 26) is before the data
    assert starts_inside_window["skipped_before_data"] == 1
    assert starts_inside_window["windows"][0][0] >= pd.Timestamp("2010-03-30")


def test_windows_in_data_keeps_a_window_whose_entry_is_exactly_the_first_data_date():
    """The LOWER boundary of the rule: a window is warm-up only when its entry is
    strictly BEFORE the first data date. If the vendor's history began on a
    scheduled entry date the window must still open (an off-by-one here would
    silently drop a pre-registered window)."""
    march = (pd.Timestamp("2010-03-26"), pd.Timestamp("2010-03-31"))
    on_entry = mod.windows_in_data(_data_calendar("2010-03-26", "2010-08-31"))
    assert on_entry["windows"][0] == march
    assert on_entry["skipped_before_data"] == 0
    one_session_later = mod.windows_in_data(_data_calendar("2010-03-29", "2010-08-31"))
    assert march not in one_session_later["windows"]
    assert one_session_later["skipped_before_data"] == 1


def test_build_curve_opens_a_window_entered_on_the_first_data_date():
    idx, spy, legs, irx = _synthetic_inputs(start="2010-03-26", end="2010-06-30")
    assert idx[0] == pd.Timestamp("2010-03-26")            # the first bar IS a scheduled entry date
    curve = mod.build_curve(spy, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    first = curve["trades_df"].iloc[0]
    assert first["entry_date"] == idx[0] and first["exit_date"] == pd.Timestamp("2010-03-31")
    assert curve["window_info"]["skipped_before_data"] == 0
    assert list(np.flatnonzero(curve["overlay_mask"].to_numpy()[:5])) == [1, 2, 3]


def test_windows_in_data_truncated_final_window_and_not_started_window():
    full = _data_calendar("2010-01-04", "2010-12-31")
    # cut inside the December window: entry seen, exit not
    dec_entry, dec_exit = mod.month_end_windows(pd.Timestamp("2010-12-01"), pd.Timestamp("2010-12-31"))[0]
    cut = full[full <= full[full.get_loc(dec_entry) + 1]]
    info = mod.windows_in_data(cut)
    assert info["windows"][-1] == (dec_entry, dec_exit) and info["truncated_final"] is True
    assert info["not_started"] == 0
    # cut before the December window's entry: December is not started
    early = full[full < dec_entry]
    info2 = mod.windows_in_data(early)
    assert info2["not_started"] == 1 and info2["truncated_final"] is False
    assert all(w[1] <= early[-1] for w in info2["windows"])


def test_windows_in_data_separates_a_truncated_final_window_from_one_entered_on_the_last_bar():
    """Data ending EXACTLY on an entry date opens nothing (the entry close is the
    last close), so it is neither 'truncated' (no trade is closed early) nor
    'not started' (the window has entered). Only data ending inside the window
    (T-2 or T-1) truncates a real trade."""
    full = _data_calendar("2010-01-04", "2010-12-31")
    dec_entry, dec_exit = mod.month_end_windows(pd.Timestamp("2010-12-01"), pd.Timestamp("2010-12-31"))[0]
    e = full.get_loc(dec_entry)
    expected = {   # sessions of data past the entry date -> (truncated_final, entered_on_last_bar, not_started)
        -1: (False, False, 1),    # data ends the session BEFORE the entry
        0: (False, True, 0),      # data ends exactly on the entry date
        1: (True, False, 0),      # T-2
        2: (True, False, 0),      # T-1
        3: (False, False, 0),     # T: the window completes exactly
    }
    for past, (truncated, on_last_bar, not_started) in expected.items():
        info = mod.windows_in_data(full[:e + 1 + past])
        assert (info["truncated_final"], info["entered_on_last_bar"], info["not_started"]) == (
            truncated, on_last_bar, not_started), past
        if past >= 0:
            assert info["windows"][-1] == (dec_entry, dec_exit), past


def test_the_trade_count_equals_the_window_count_except_for_a_window_entered_on_the_last_bar():
    idx, spy, legs, irx = _synthetic_inputs()
    entry, _ = mod.windows_in_data(idx)["windows"][10]
    e = idx.get_loc(entry)
    for past in (0, 1, 2, 3, 4):
        cut = idx[:e + 1 + past]
        curve = mod.build_curve(spy.loc[cut], {s: f.loc[cut] for s, f in legs.items()}, irx.loc[cut],
                                leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
        info, trades = curve["window_info"], curve["trades_df"]
        assert len(trades) == len(info["windows"]) - int(info["entered_on_last_bar"]), past
        assert info["entered_on_last_bar"] is (past == 0), past
        assert info["truncated_final"] is (past in (1, 2)), past
        assert trades["reason"].iloc[-1] == ("end_of_data" if past in (1, 2) else "window"), past


def test_windows_in_data_refuses_a_missing_entry_or_exit_session():
    idx = _data_calendar("2010-01-04", "2010-12-31")
    mar_entry, mar_exit = mod.month_end_windows(pd.Timestamp("2010-03-01"), pd.Timestamp("2010-03-31"))[0]
    with pytest.raises(ValueError, match="2010-03"):
        mod.windows_in_data(idx.difference(pd.DatetimeIndex([mar_entry])))
    with pytest.raises(ValueError, match="2010-03"):
        mod.windows_in_data(idx.difference(pd.DatetimeIndex([mar_exit])))


def test_an_unscheduled_closure_inside_a_window_is_no_hole():
    """Oct 29-30 2012 were scheduled sessions the exchange never opened. The
    window (2012-10-26 -> 2012-10-31) needs only its own two dates."""
    real_sessions, _ = _xnys_sessions_and_adhoc(pd.Timestamp("2012-01-01"), pd.Timestamp("2012-12-31"))
    assert pd.Timestamp("2012-10-29") not in real_sessions
    info = mod.windows_in_data(real_sessions)
    assert (pd.Timestamp("2012-10-26"), pd.Timestamp("2012-10-31")) in info["windows"]


# ------------- the data calendar against the REAL exchange session set (calendar gaps)

def test_exchange_sessions_are_the_real_sessions_and_omit_the_unscheduled_closures():
    start, end = pd.Timestamp("2001-01-01"), pd.Timestamp("2028-12-31")
    sessions, adhoc = _xnys_sessions_and_adhoc(start, end)
    actual = mod.exchange_sessions(start, end)
    assert actual.equals(sessions)
    assert not set(actual) & set(adhoc[(adhoc >= start) & (adhoc <= end)])
    sandy = mod.exchange_sessions(pd.Timestamp("2012-10-26"), pd.Timestamp("2012-11-02"))
    assert pd.Timestamp("2012-10-29") not in sandy and pd.Timestamp("2012-10-31") in sandy


def test_session_coverage_accepts_the_exact_exchange_sessions_including_the_sandy_gap():
    real, _ = _xnys_sessions_and_adhoc(pd.Timestamp("2012-01-01"), pd.Timestamp("2012-12-31"))
    assert pd.Timestamp("2012-10-29") not in real and pd.Timestamp("2012-10-30") not in real
    mod.validate_session_coverage(real)                 # a known ad hoc closure is no hole


def test_session_coverage_refuses_ordinary_missing_sessions_and_lists_every_one():
    real, _ = _xnys_sessions_and_adhoc(pd.Timestamp("2012-01-01"), pd.Timestamp("2012-12-31"))
    missing = [real[40], real[41], real[130]]
    with pytest.raises(lo.CalendarCoverageError) as caught:
        mod.validate_session_coverage(real.difference(pd.DatetimeIndex(missing)))
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.reason_code == "missing_exchange_session"
    assert error.missing_dates == [d.strftime("%Y-%m-%d") for d in missing]
    for date in error.missing_dates:
        assert date in str(error)


def test_session_coverage_refuses_a_bar_on_a_day_the_exchange_was_closed():
    real, _ = _xnys_sessions_and_adhoc(pd.Timestamp("2012-01-01"), pd.Timestamp("2012-12-31"))
    extras = [pd.Timestamp("2012-10-29"), pd.Timestamp("2012-11-22")]     # Sandy closure, Thanksgiving
    with pytest.raises(lo.CalendarCoverageError) as caught:
        mod.validate_session_coverage(real.union(pd.DatetimeIndex(extras)))
    assert caught.value.reason_code == "bar_on_closed_day"
    assert caught.value.extra_dates == ["2012-10-29", "2012-11-22"]
    assert "2012-11-22" in str(caught.value)


def _drop_everywhere(spy, legs, irx, dates):
    gone = pd.DatetimeIndex(dates)
    return spy.drop(gone), {s: f.drop(gone) for s, f in legs.items()}, irx.drop(gone)


def test_build_curve_refuses_an_ordinary_session_missing_from_every_series():
    """A session every series lacks leaves all the pairwise calendar checks
    satisfied, and its multi-day close-to-close move would be booked as one bar."""
    idx, spy, legs, irx = _synthetic_inputs()
    gone = idx[100]
    spy2, legs2, irx2 = _drop_everywhere(spy, legs, irx, [gone])
    for build in (mod.build_curve, mod.build_next_open_sensitivity):
        with pytest.raises(lo.CalendarCoverageError) as caught:
            build(spy2, legs2, irx2, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
        assert caught.value.reason_code == "missing_exchange_session"
        assert caught.value.missing_dates == [gone.strftime("%Y-%m-%d")]


def test_a_missing_session_inside_a_window_is_refused_though_its_entry_and_exit_are_present():
    idx, spy, legs, irx = _synthetic_inputs()
    entry, exit_ = mod.windows_in_data(idx)["windows"][10]
    inside = idx[idx.get_loc(exit_) - 1]                                 # T-1
    # the window's own entry/exit check cannot see it ...
    assert (entry, exit_) in mod.windows_in_data(idx.drop(inside))["windows"]
    # ... the session-coverage rule does
    spy2, legs2, irx2 = _drop_everywhere(spy, legs, irx, [inside])
    with pytest.raises(lo.CalendarCoverageError) as caught:
        mod.build_curve(spy2, legs2, irx2, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    assert caught.value.missing_dates == [inside.strftime("%Y-%m-%d")]


def test_build_curve_refuses_unequal_trailing_coverage_instead_of_ending_the_account_early():
    """IEF and AGG lack the trailing bar that TLT and SPY have: the account must
    not silently end a session early."""
    idx, spy, legs, irx = _synthetic_inputs()
    short = {**legs, "IEF": legs["IEF"].iloc[:-1], "AGG": legs["AGG"].iloc[:-1]}
    for build in (mod.build_curve, mod.build_next_open_sensitivity):
        with pytest.raises(lo.CalendarCoverageError) as caught:
            build(spy, short, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
        assert caught.value.reason_code == "unequal_trailing_coverage"
        assert caught.value.missing_dates == [idx[-1].strftime("%Y-%m-%d")]
        assert "IEF" in str(caught.value) and "AGG" in str(caught.value)


def _windows_prefix_violations(idx, cutoffs, windows_fn):
    """A data prefix cut at ``c`` must see EXACTLY the full run's windows that
    have entered by ``c`` -- with the SAME scheduled exit."""
    full = windows_fn(idx)
    bad = []
    for c in cutoffs:
        expected = [w for w in full if w[0] <= idx[c]]
        if windows_fn(idx[:c + 1]) != expected:
            bad.append(c)
    return bad


def _data_derived_windows(index):
    """The look-ahead this module must NOT contain: month-end read off the data
    ('the next row falls in a new month')."""
    idx = pd.DatetimeIndex(index)
    out = []
    months = idx.to_period("M")
    for period in months.unique():
        days = idx[months == period]
        if len(days) > mod.WINDOW_SESSIONS:
            out.append((days[-(mod.WINDOW_SESSIONS + 1)], days[-1]))
    return out


def _cutoffs_one_to_three_sessions_after_each_entry(idx, windows):
    cutoffs = []
    for entry, _ in windows:
        e = idx.get_loc(entry)
        cutoffs += [e, e + 1, e + 2, e + 3, e + 4]
    return [c for c in cutoffs if c < len(idx) - 1]


def test_windows_in_data_is_prefix_invariant_at_cutoffs_inside_each_window():
    idx = _data_calendar("2009-06-01", "2012-12-31")
    windows = mod.windows_in_data(idx)["windows"]
    cutoffs = _cutoffs_one_to_three_sessions_after_each_entry(idx, windows)
    assert len(cutoffs) > 100
    assert _windows_prefix_violations(idx, cutoffs, lambda i: mod.windows_in_data(i)["windows"]) == []


def test_the_prefix_test_has_power_against_a_data_derived_month_end():
    """The same cutoffs must catch a month-end read off the data: a cut at
    T-1 makes 'the last row of the month' T-1, a false window."""
    idx = _data_calendar("2009-06-01", "2012-12-31")
    windows = mod.windows_in_data(idx)["windows"]
    cutoffs = _cutoffs_one_to_three_sessions_after_each_entry(idx, windows)
    caught = _windows_prefix_violations(idx, cutoffs, _data_derived_windows)
    assert len(caught) >= len(windows) // 2       # most windows expose it, not one lucky cutoff


def test_round_cutoffs_alone_would_miss_the_data_derived_month_end():
    """Why the cutoffs sit at and after each entry: cutting on a month's LAST
    scheduled session (or any date that is not inside a window) hides it."""
    idx = _data_calendar("2009-06-01", "2012-12-31")
    windows = mod.windows_in_data(idx)["windows"]
    exits = [idx.get_loc(x) for _, x in windows if idx.get_loc(x) < len(idx) - 1]
    assert _windows_prefix_violations(idx, exits, _data_derived_windows) == []


# ------------------------------------------------------------- the curve builder

def _synthetic_inputs(start="2011-06-01", end="2013-08-30", seed=5):
    """Frames on the REAL scheduled calendar minus the Sandy closure days."""
    idx = _data_calendar(start, end, drop=("2012-10-29", "2012-10-30"))

    def ohlc(level, vol, s):
        r = np.random.RandomState(s).normal(0.0002, vol, len(idx))
        r[0] = 0.0
        close = level * np.cumprod(1.0 + r)
        prev = np.concatenate([[close[0]], close[:-1]])
        return pd.DataFrame({"Open": prev, "High": np.maximum(prev, close), "Low": np.minimum(prev, close),
                             "Close": close}, index=idx)

    legs = {"IEF": ohlc(100.0, 0.004, 11), "TLT": ohlc(110.0, 0.008, 12), "AGG": ohlc(90.0, 0.003, 13)}
    spy = ohlc(120.0, 0.01, 14)
    irx = pd.DataFrame({"Open": 3.0, "High": 3.0, "Low": 3.0,
                        "Close": np.linspace(2.0, 3.0, len(idx))}, index=idx)
    return idx, spy, legs, irx


_COSTS = {"IEF": 1.0, "TLT": 2.0, "AGG": 3.0}


def test_build_curve_uses_the_pinned_parameters_and_the_scheduled_windows():
    idx, spy, legs, irx = _synthetic_inputs()
    curve = mod.build_curve(spy, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.25)
    info = mod.windows_in_data(idx)
    financing = lo.causal_financing_rate(irx["Close"], idx)
    direct = lo.build_long_overlay_curve(
        spy, {s: legs[s] for s in mod.TRADED}, info["windows"], leg_cost_bps=_COSTS,
        spy_entry_cost_bps=0.25, financing_rate_annual=financing,
        overlay_notional=mod.PARAMETERS["overlay_notional_fraction"])
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy(), direct["strategy_daily"].to_numpy())
    pd.testing.assert_frame_equal(curve["trades_df"], direct["trades_df"])
    assert len(curve["trades_df"]) == len(info["windows"]) >= 24
    assert curve["window_info"]["windows"] == info["windows"]


def test_build_curve_account_equals_the_evaluators_benchmark_outside_the_windows_with_a_longer_spy_history():
    """Production shape through the whole family stack: SPY lists long before the
    legs, and the account outside the windows must equal the benchmark series the
    ABI-1 evaluator subtracts (computed on SPY's FULL history) bit for bit."""
    from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns
    idx, spy_on_legs_span, legs, irx = _synthetic_inputs()
    lead = mod.scheduled_sessions(idx[0] - pd.Timedelta(days=200), idx[0] - pd.Timedelta(days=1))
    assert len(lead) > 100
    full_idx = lead.union(idx)
    r = np.random.RandomState(21).normal(0.0002, 0.01, len(full_idx))
    r[0] = 0.0
    close = 120.0 * np.cumprod(1.0 + r)
    prev = np.concatenate([[close[0]], close[:-1]])
    spy_full = pd.DataFrame({"Open": prev, "High": np.maximum(prev, close), "Low": np.minimum(prev, close),
                             "Close": close}, index=full_idx)
    entry_cost = 0.25
    curve = mod.build_curve(spy_full, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=entry_cost)
    evaluator_benchmark = cost_adjusted_benchmark_daily_returns(
        spy_full, entry_cost_bps=entry_cost).reindex(curve["strategy_daily"].index)
    outside = ~curve["overlay_mask"].to_numpy()
    assert spy_full.index[0] < curve["strategy_daily"].index[0]        # SPY really is longer
    assert outside[0] and outside.sum() > 300
    sliced = cost_adjusted_benchmark_daily_returns(
        spy_full.loc[idx[0]:idx[-1]], entry_cost_bps=entry_cost)
    assert sliced.iloc[0] != evaluator_benchmark.iloc[0]               # the fixture can tell the two apart
    np.testing.assert_array_equal(curve["strategy_daily"].to_numpy()[outside],
                                  evaluator_benchmark.to_numpy()[outside])


def test_build_curve_is_prefix_invariant_at_cutoffs_one_to_three_sessions_after_each_entry():
    """The whole stack (calendar -> common calendar -> causal financing rate
    -> builder) on frames cut INSIDE a window must reproduce the full-history
    curve on every session up to the cutoff (costs are zero here, so the
    truncated window's round-trip charge cannot differ)."""
    idx, spy, legs, irx = _synthetic_inputs()
    zero = {s: 0.0 for s in mod.TRADED}
    full = mod.build_curve(spy, legs, irx, leg_cost_bps=zero, spy_entry_cost_bps=0.0)
    windows = mod.windows_in_data(idx)["windows"]
    cutoffs = []
    for entry, _ in windows[3:9]:
        e = idx.get_loc(entry)
        cutoffs += [e, e + 1, e + 2, e + 3, e + 4]
    checked = 0
    for c in cutoffs:
        cut = idx[:c + 1]
        part = mod.build_curve(spy.loc[cut], {s: f.loc[cut] for s, f in legs.items()}, irx.loc[cut],
                               leg_cost_bps=zero, spy_entry_cost_bps=0.0)
        np.testing.assert_array_equal(part["strategy_daily"].to_numpy(),
                                      full["strategy_daily"].to_numpy()[:c + 1])
        checked += 1
    assert checked == len(cutoffs) == 30


def test_the_sandy_window_is_held_across_the_closure_and_financed_for_the_calendar_days():
    idx, spy, legs, irx = _synthetic_inputs()
    curve = mod.build_curve(spy, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    row = curve["trades_df"].loc[curve["trades_df"]["entry_date"] == pd.Timestamp("2012-10-26")].iloc[0]
    assert row["exit_date"] == pd.Timestamp("2012-10-31") and row["held"] == 1 and row["reason"] == "window"
    rate = curve["financing_rate"].loc["2012-10-31"]
    assert row["financing_ret"] == pytest.approx(0.5 * rate * 5.0 / 365.0, rel=1e-12)   # Fri -> Wed: 5 calendar days


def test_the_overlay_is_live_only_from_the_session_after_each_scheduled_entry_through_its_exit():
    idx, spy, legs, irx = _synthetic_inputs()
    curve = mod.build_curve(spy, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    mask = curve["overlay_mask"]
    windows = mod.windows_in_data(idx)["windows"]
    expected = np.zeros(len(idx), dtype=bool)
    for entry, exit_ in windows:
        expected[idx.get_loc(entry) + 1:idx.get_loc(exit_) + 1] = True
    np.testing.assert_array_equal(mask.to_numpy(), expected)
    assert 0.08 < mask.mean() < 0.20              # roughly 3 of every 21 sessions


def test_build_curve_requires_exactly_the_registered_legs():
    idx, spy, legs, irx = _synthetic_inputs()
    with pytest.raises(ValueError, match="IEF, TLT, AGG"):
        mod.build_curve(spy, {"IEF": legs["IEF"], "TLT": legs["TLT"]}, irx,
                        leg_cost_bps={"IEF": 1.0, "TLT": 1.0}, spy_entry_cost_bps=0.0)
    with pytest.raises(ValueError, match="IEF, TLT, AGG"):
        mod.build_curve(spy, {**legs, "SHY": legs["IEF"]}, irx,
                        leg_cost_bps={**_COSTS, "SHY": 1.0}, spy_entry_cost_bps=0.0)


def test_the_financing_rate_really_comes_from_the_irx_frame():
    idx, spy, legs, irx = _synthetic_inputs()
    a = mod.build_curve(spy, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    higher = irx.copy()
    higher["Close"] = higher["Close"] + 2.0
    b = mod.build_curve(spy, legs, higher, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    assert (b["trades_df"]["financing_ret"] > a["trades_df"]["financing_ret"]).all()
    np.testing.assert_allclose(a["trades_df"]["gross_ret"], b["trades_df"]["gross_ret"], atol=1e-15)


def test_a_negative_bill_print_does_not_break_the_builder():
    """^IRX printed at or below zero on 7 real sessions; it is a signal-only
    series and must never be forced through a positivity check."""
    idx, spy, legs, irx = _synthetic_inputs()
    neg = irx.copy()
    neg["Close"] = np.linspace(-0.1, 0.4, len(idx))
    curve = mod.build_curve(spy, legs, neg, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    assert np.isfinite(curve["strategy_daily"].to_numpy()).all()


def test_an_irx_history_that_starts_after_the_first_financed_session_fails_closed():
    idx, spy, legs, irx = _synthetic_inputs()
    late = irx.iloc[200:]
    with pytest.raises(ValueError, match="financing"):
        mod.build_curve(spy, legs, late, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)


def test_next_open_sensitivity_is_a_separate_report_only_function():
    idx, spy, legs, irx = _synthetic_inputs()
    scored = mod.build_curve(spy, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    sens = mod.build_next_open_sensitivity(spy, legs, irx, leg_cost_bps=_COSTS, spy_entry_cost_bps=0.0)
    assert len(sens["trades_df"]) == len(scored["trades_df"])
    assert not np.array_equal(sens["strategy_daily"].to_numpy(), scored["strategy_daily"].to_numpy())
    # the same exits, one session-later entries
    assert (sens["trades_df"]["exit_date"] == scored["trades_df"]["exit_date"]).all()
    assert (sens["trades_df"]["entry_date"] > scored["trades_df"]["entry_date"]).all()


def test_expected_min_trades_is_below_the_calendar_only_window_count_for_the_real_history():
    """Structural count from the schedule alone (no price is read): windows
    from AGG's first session (2003-09-29) through 2026-09-23."""
    idx = mod.scheduled_sessions(pd.Timestamp("2003-09-29"), pd.Timestamp("2026-09-23"))
    info = mod.windows_in_data(idx)
    assert mod.EXPECTED_MIN_TRADES <= len(info["windows"])
    assert info["windows"][0][0] >= pd.Timestamp("2003-09-29")
    assert info["windows"][0] == (pd.Timestamp("2003-10-28"), pd.Timestamp("2003-10-31"))


# ------------------------------------------------------ cost-source disclosure

def test_the_cost_source_disclosure_names_the_human_process_safeguard():
    doc = mod.__doc__
    for phrase in ("Cost source", "human-process safeguard", "SYNTHETIC", "live broker quote capture", "invented",
                   "pipeline-wide"):
        assert phrase in doc, phrase


def test_refuse_synthetic_quotes_only_catches_a_file_that_announces_itself_as_synthetic():
    """The pipeline-wide limit the module docstring discloses (shared code,
    deliberately unchanged): an invented bid/ask file that omits the marker is
    NOT refused, so real quotes must come from a real capture path by
    human process. If shared code ever verifies provenance, this test should
    fail and the disclosure should be updated."""
    import run_batch5_mining as driver
    invented = {"sampled_at_local": "2026-09-23T10:00:00", "market_state": "open",
                "source": "typed by hand",
                "quotes": {"IEF": {"bid": 95.00, "ask": 95.02}}}
    driver.refuse_synthetic_quotes(invented, dry_run=False)                 # no exception: passes silently
    marked = {**invented, "source": "SYNTHETIC unit-test fixture"}
    with pytest.raises(ValueError, match="SYNTHETIC"):
        driver.refuse_synthetic_quotes(marked, dry_run=False)
